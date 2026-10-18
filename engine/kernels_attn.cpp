// Launcher for prototype/attn_decode: one attention layer per body — one query row through the fused
// form (two launches above the lane staging), or a whole prefill wave through the staged form (P8,
// §06.3: prep + core, plus combine above the staging). Its own TU — attn_decode defines N_Q, N_KV, DH, ROT,
// THREADS, TILE, MAXSC, SCALE, EPS, FULL_MASK and its own bf/fb/rbf/block_sum, every one of which
// collides with another op's spelling if these files are merged.
//
// Three configuration choices, each made by a gate rather than here:
//
//   * `fused=1` for ONE query row, `fused=0` for a wave. The op's gate runs `staged` as its headline
//     plan and requires `fused` to agree with it BIT FOR BIT (`run.py`'s cross-check fails the gate if
//     the two launch plans differ at all), so the fused form is gated against the same pins, and it is
//     the one-launch form W5 wants at these cache lengths. A wave cannot use it: in the fused form each
//     core block appends only its own row's lane, and a wave's row t scores the lanes rows 0..t-1
//     appended, with no grid-wide barrier to order the two — see the launch plan below. Note that the
//     staged plan is not just one flag flipped: its core READS the staged q/k, so the launcher has to
//     point the core's input pointers at what `attn_prep` wrote, and null its output pointers so the
//     core does not re-append what prep already appended in order.
//   * `full`/`split` are chosen per row by necessity and nothing else — see the lane geometry below.
//     One-pass (`split = 1`) wherever a row fits the staging, because it is bit-exact with the
//     pins and it is the route every priced fixture rides: §07.3's decision, and the reason
//     a split is only ever reached by being forced. The blocked `(max,sum)` form cannot reproduce the
//     pins at any tile size — §05.5 rounds the *normalised* weight and a streaming row cannot round a
//     weight it has not divided yet — measured on the dec8 pins: one-pass 0 of 98 304 elements
//     differ, blocked 25 745. Its own band is §07.3's pre-authorized one, gated in
//     `prototype/attn_decode/run.py` (results.md §10), and it is the form that can be split.
//   * `score_f32=0` — the score is rounded through BF16 before the scale, which is the form the pins
//     carry (see G6's history in `bench/gate-audit.md`: the band that hid this was an audit finding).
#include "engine/launch_stats.h"
#include "kernels.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "prototype/attn_decode/attn_decode_kernels.h"
// The engine's copy of the tiled wave body with the two per-touch instruction levers applied
// (`--attn-touch 1`, the shipping default). Included AFTER the op header because it is a copy of that
// header's `attn_tile_body` and uses its vocabulary; both kernels are compiled either way, so the switch
// below selects a LAUNCH and never a build.
#include "engine/attn_slim_kernels.h"
#include "engine/attn_fattn_kernels.h"

namespace engine::kernels {
namespace {
// ---- the lane geometry (P7, §07.3) -------------------------------------------------------------
// The op stages one block's scores in shared memory: `MAXSC` lanes, a #define the kernel owns
// (`prototype/attn_decode/attn_decode_kernels.h:32`). The ONE-PASS form cannot cut a row across
// blocks — it normalises the weight matrix inside one block (§05.5), which is why the prototype
// refuses `--softmax full --split > 1` outright — so above the staging the engine switches to the
// streaming form and splits the lane axis into `split` blocks whose `(max,sum,Σ w·v)` partials
// `attn_combine` merges with the same rescaling (A-17: that state belongs to one query row and dies
// inside this launch, which is what makes it reduction state and not carry state).
//
// `kAttnMaxSplit` is how many blocks one query row may be cut into, and it is the second half of the
// engine's lane ceiling: `MAXSC × kAttnMaxSplit`. It is what makes that ceiling a bounded,
// allocatable number rather than "as much shared memory as the row happens to need", and going past
// it is a refusal, never an overrun: `attn_decode_layer` throws rather than write past the staging or
// past the `part` scratch. 9 admits the widest PINNED plan — `wide4096f`'s 4096 prompt + 8 generated
// = 4104 lanes, which 8's 4096 refused — on `wide4096p`'s 4096 judged rows (D44); nothing above 4104
// is pinned. No address reads this cap, so a plan 8 ran is bit-identical, and only the scratch grows.
//
// `engine/gate.py::lane_ceiling()` READS both halves of that product out of the files that enforce
// them, so neither constant can be retyped into the gate.
constexpr int kAttnMaxSplit = 9;

// The split a row of `lanes` needs: `split = ceil(lanes / staging)`, so no block is handed more lanes
// than the staging width the split was derived from (`ceil(n / ceil(n/s)) <= s`).
int split_for(int lanes) { return (lanes + MAXSC - 1) / MAXSC; }

int warp_width() {
    static int w = [] {
        int v = 32;
        (void)hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, 0);
        return v > 0 ? v : 32;
    }();
    return w;
}

// The engine's public boundary is `uint16_t*` for BF16 buffers (kernels.h keeps hip_bfloat16 out of
// the host API); CoreArgs is the op's own struct and wants `hip_bfloat16*`. Same 2 bytes, and one
// named cast here so no individual field assignment looks like a type error being papered over. The
// cast drops const deliberately and once: `CoreArgs` — verbatim from the pinned prototype — spells its
// read-only members (`qn`, `kn`, the three projections) non-const, and editing a kernel this box has
// no device to re-gate is the worse price for const-correctness here.
hip_bfloat16* bf16(const uint16_t* p) {
    return reinterpret_cast<hip_bfloat16*>(const_cast<uint16_t*>(p));
}

// The form banner AND the covered-share / fallback echo, printed once per DISTINCT wave geometry rather
// than once per attention layer.
//
// WHY IT EXISTS AT ALL: under `--attn-fattn 1` only the streaming rows move to the WMMA form, so any family
// speedup has to be divided by the covered share before it is believed — a 1.5x family result on 70 %
// coverage is a ~2x result on covered rows, and those are different claims. A log that cannot say what
// fraction it covered adjudicates neither. THE FALLBACK IS PRINTED, NOT IMPLIED: `fallback_rows=N-n` is its
// own field, because "rows that fell back must be visible in the log" is not satisfied by a reader having to
// subtract two numbers and then notice the subtraction matters. WHY ONCE PER GEOMETRY: the same body calls
// this from all 48 attention layers with the same numbers, and 48 identical lines per body would be noise a
// reader skims past, which is worse than one line. A geometry change (a later body with deeper lanes) prints
// again, so the burst window sees every number it needs. `decode_run`'s NUMERICS line carries the flag;
// this line carries the denominator.
//
// TWO DENOMINATORS, because rows are not the work. `covered_rows_pct` is the row share; `covered_work_pct`
// weights each row by `lanes_of(a, row)`, which is what the `attn` family's time is proportional to per row
// (one pass over the row's lanes, both forms). The ms share of the family is the second one, not the first:
// under (A) the covered rows are the LONG ones, so the work share is always the larger of the two and quoting
// the row share would understate coverage — at 4,096 lanes the row share is ~87 % and the work share is
// ~99 %, and a burst that divides by the wrong one is wrong by 1.14x. Both are printed so the reader cannot
// pick the flattering one by accident.
void fattn_echo(int rows, int split, int lvalid, int nrows) {
    static std::string last;
    // THE FORM'S OWN COSTS, read from the loaded code object rather than from a comment. `hipFuncGetAttributes`
    // is authoritative here and the compiler's `-Rpass-analysis=kernel-resource-usage` note is not: that note
    // prints a constant `Occupancy [waves/SIMD]: 16` for every kernel in the TU whatever the LDS
    // (AGENTS.md, "the counters and census traps"), and a banner whose fields cannot disagree with the build
    // is a banner that cannot catch a stale build. This is the D30 1.7' lesson applied one level over: the
    // canary must be a field the change is REQUIRED to move, read in the same window as the number.
    // `scratch=` is the private bytes per thread and `wf_per_cu=` the driver's own blocks/CU answer times the
    // block's warp count — the two fields that make a spilled or occupancy-starved build visible in the log,
    // which is the difference between the shipped form and the one that arrived (`.vgpr_spill_count 242`,
    // `.private_segment_fixed_size 332` in `engine/attn_fattn_kernels.h`'s note).
    hipFuncAttributes at{};
    const bool have_attrs =
        hipFuncGetAttributes(&at, reinterpret_cast<const void*>(attn_fattn_wave)) == hipSuccess;
    int blocks_per_cu = -1;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu,
                                                    reinterpret_cast<const void*>(attn_fattn_wave),
                                                    FATTN_THREADS, 0) != hipSuccess) {
        blocks_per_cu = -1;
    }
    CoreArgs k{};
    k.nrows = nrows;
    k.split = split;
    k.lvalid = lvalid;
    // THE SAME PLAN THE CORE IS LAUNCHED UNDER, and this line is the one place that could lie about the
    // covered share, so it is spelled out: `full = 1` is what `attn_decode_layer` sets above (`c.full = 1`),
    // and `blocks_for_row` only applies §07.3's one-pass rule when `full` is set. Left at the zero-init 0,
    // every row would look like a streaming row and this would print a 100 % covered share on a wave of
    // short rows — a printed number that cannot be wrong is not an instrument.
    k.full = 1;
    k.fattn_cover = 1;                    // this is the "if the form is on, how much would it own" query
    k.cap = 1 << 30;                      // never the binding limit here; `lanes_of` caps at the cache
    int covered = 0;
    long work_all = 0, work_cov = 0;
    for (int r = 0; r < rows; ++r) {
        const long lanes = lanes_of(k, r);
        work_all += lanes;
        if (fattn_covers(k, r)) {
            ++covered;
            work_cov += lanes;
        }
    }
    char buf[400];
    std::snprintf(buf, sizeof(buf),
                  "ATTN_FATTN form=fattn16 q=stage vgpr=%d lds=%d scratch=%d vtiles=%d wf_per_cu=%d"
                  " wave_rows=%d split=%d fattn_rows=%d/%d fallback_rows=%d"
                  " covered_rows_pct=%.1f covered_work_pct=%.1f%s",
                  have_attrs ? at.numRegs : -1,
                  have_attrs ? (int)(at.sharedSizeBytes) : -1,
                  have_attrs ? (int)(at.localSizeBytes) : -1,
                  FATTN_VTILES, blocks_per_cu * (FATTN_THREADS / 32),
                  rows, split, covered, rows, rows - covered,
                  rows ? 100.0 * covered / rows : 0.0,
                  work_all ? 100.0 * (double)work_cov / (double)work_all : 0.0,
                  (have_attrs && at.localSizeBytes > 0) ? " SPILL" : "");
    const std::string line(buf);
    if (line == last) return;
    last = line;
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}
}  // namespace

std::size_t attn_part_floats(int rows) {
    // `attn_core`'s own partial layout: one `(max, sum, acc[DH])` tuple per (query head, split
    // block), F32 — `a.part + ((qrow*N_Q + head)*split + blockIdx.y)*(DH + 2)` in the kernel. The
    // slot index never exceeds `rows*N_Q*split - 1`, and the throw in `attn_decode_layer` is what
    // keeps `split <= kAttnMaxSplit`, so sizing for the cap cannot overrun. At `kMaxWaveRows` = 512
    // it is 512*8*9*258 F32 = 38,043,648 B (33,816,576 B at cap 8), ~2.5 % of the 1.5457 GB
    // checkpoint; cap 40 (20480 lanes) would be 169,082,880 B — cheap, unfunded, no pin reaches it.
    // It is written only by rows above the staging, so a request under 512 lanes never touches it.
    return static_cast<size_t>(rows) * N_Q * kAttnMaxSplit * (DH + 2);
}

void attn_decode_layer(const AttnLayerArgs& a) {
    if (a.q_proj_out == nullptr || a.k_proj_out == nullptr || a.v_proj_out == nullptr ||
        a.q_norm == nullptr || a.k_norm == nullptr || a.k_cache == nullptr ||
        a.v_cache == nullptr || a.attn_out_pre_gate == nullptr || a.gated == nullptr ||
        a.q_rope == nullptr || a.k_rope == nullptr || a.gate_out == nullptr) {
        throw std::runtime_error(
            "attn_decode_layer: a required buffer is null. The three stage buffers are NOT "
            "optional in the fused plan: attn_decode_kernels.h's stage_head() returns early when "
            "q_rope is null, and the §05.7 cache append lives behind that same test, so nulling "
            "them to 'skip the dump' silently skips writing K and V into the cache.");
    }
    if (a.nrows < 1) {
        throw std::runtime_error("attn_decode_layer: nrows=" + std::to_string(a.nrows) +
                                 "; a body has at least one query row (a wave of zero tokens is"
                                 " `submit_wave_body`'s to refuse, and it does)");
    }
    // The wave's rows advance one position and one visible lane each (§06.3), so the WIDEST row is the
    // last one and every lane decision below is taken from it. `pos` is row 0's position.
    const int rows = a.nrows;
    const int lanes_last = a.lvalid + rows - 1;
    if (a.lvalid < 0 || a.pos < 0 || a.pos + rows - 1 >= a.cap || lanes_last > a.cap) {
        throw std::runtime_error("attn_decode_layer: pos=" + std::to_string(a.pos) + ".." +
                                 std::to_string(a.pos + rows - 1) + " lvalid=" +
                                 std::to_string(a.lvalid) + ".." + std::to_string(lanes_last) +
                                 " (nrows=" + std::to_string(rows) + ") does not fit cap=" +
                                 std::to_string(a.cap));
    }

    // ---- the form, chosen by necessity (§07.3) ------------------------------------------------
    // `split_for(lanes_last)` is 1 whenever the widest row fits the staging and >= 2 whenever it does
    // not, which is the only guarantee `blocks_for_row` needs to route each ROW on its own: a row that
    // fits takes exactly one block and the pins' bits, a row above the staging streams across `split`
    // blocks and `attn_combine` merges them. `full` is therefore no longer a per-launch choice (P7
    // spelled it `split_lane ? 0 : 1`); for a single row the two spellings pick the same branch of
    // `attn_core` at every lane count, which is what keeps `wide520f`'s route and shipped margin
    // untouched, and for a wave it is what lets a straddling wave mix the two forms in one grid instead
    // of pushing the short rows onto the streaming form they do not need.
    // What would still be mis-normalisation is a row WIDER than the staging served by ONE block, and
    // `split >= 2` whenever `lanes_last > MAXSC` is what makes that unreachable — so the one-pass
    // branch of `attn_core` can never be handed more than `MAXSC` lanes.
    const bool split_lane = lanes_last > MAXSC;
    const int split = split_for(lanes_last);
    if (split_lane) {
        if (split > kAttnMaxSplit) {
            throw std::runtime_error("attn_decode_layer: widest row is " +
                                     std::to_string(lanes_last) + " lanes (nrows=" +
                                     std::to_string(rows) + "), needs split=" +
                                     std::to_string(split) +
                                     ", more blocks than the engine's kAttnMaxSplit=" +
                                     std::to_string(kAttnMaxSplit) + " (lane ceiling " +
                                     std::to_string(MAXSC * kAttnMaxSplit) + " = staging MAXSC " +
                                     std::to_string(MAXSC) + " x split). One constant moves both"
                                     " halves: `attn_part_floats(rows)` sizes the `part` scratch from"
                                     " this same `kAttnMaxSplit`, and `gate.py::lane_ceiling()` reads"
                                     " both factors from the files that enforce them");
        }
        if (a.part == nullptr) {
            throw std::runtime_error("attn_decode_layer: widest row is " +
                                     std::to_string(lanes_last) + " lanes, which exceeds the"
                                     " one-pass staging (MAXSC=" + std::to_string(MAXSC) +
                                     ") and needs the split-K online form (§07.3), which has"
                                     " nowhere to put its partials: `part` is null. Allocate it —"
                                     " `attn_part_floats(rows)` says how big.");
        }
    }

    CoreArgs c{};
    c.qg = bf16(a.q_proj_out);
    c.kg = bf16(a.k_proj_out);
    c.vg = bf16(a.v_proj_out);
    c.kcache = bf16(a.k_cache);
    c.vcache = bf16(a.v_cache);
    c.qn = bf16(a.q_norm);
    c.kn = bf16(a.k_norm);
    c.aout = bf16(a.attn_out_pre_gate);
    c.gout = bf16(a.gated);
    // The §05.7 append writes the cache IN PLACE (§08.5's head-major layout is exactly what
    // attn_core indexes), so kc_as/vc_as are the cache itself, as in the gated host.
    c.kc_as = c.kcache;
    c.vc_as = c.vcache;
    c.qrope_out = bf16(a.q_rope);
    c.krope_out = bf16(a.k_rope);
    c.gate_out = bf16(a.gate_out);
    // rope_out is the cos||sin table the gate dumps to prove the in-kernel transcendentals; the
    // engine has no use for it. `stale` is NEG_CARRY plumbing (an injected bug this launcher never
    // injects); `part` is the split-K partial staging, null whenever one block covers every row.
    c.rope_out = nullptr;
    c.part = split_lane ? a.part : nullptr;
    c.stale = nullptr;
    c.cap = a.cap;
    c.pos = a.pos;
    c.lvalid = a.lvalid;
    c.nrows = rows;
    c.split = split;
    c.neg = NEG_NONE;
    c.full = 1;
    c.tile_kernel = 0;      // a zero-initialised struct is the shipped form; set per launch below
    // `fused` is the plan, not the row's lane width: 1 means the core stages its own q/k and appends
    // (§05.2-§05.3, §05.7 inline), 0 means it reads what `attn_prep` already staged and appended. It is
    // set per LAUNCH below, because that is the only thing that differs between the two plans; the
    // value here is the one-row plan's, which every caller gets unless it asked for a wave.
    c.fused = 1;
    c.score_f32 = 0;
    // Which wave core runs (`AttnLayerArgs::wave_tile`). One-row bodies never consult it — the fused
    // single-row form IS `attn_core`, so a generation step cannot be moved by this knob — and the
    // refusal below is `attn_tile_ok`'s, the same list `prototype/attn_decode/attn_decode.hip.cpp`
    // refuses, so the gate and the engine cannot disagree about what the tiled form supports.
    // THE GUARD PRICES THE WAVE'S OWN PLAN, NOT `c.fused`. `c.fused` above is the ONE-ROW plan's value
    // (1), and a wave cannot run that plan at all: the `rows > 1` branch below stages every row's
    // §05.7 append in an `attn_prep` launch that has finished before the core starts, and launches the
    // core with `fused = 0` — which is exactly the precondition the fused refusal exists to protect ("a
    // wave's §05.7 append must finish before any row reads the cache"). Reading the guard off `c.fused`
    // refused the tiled wave form unconditionally, in every path including `--time-waves`: each
    // `--attn-tile 1` run threw on its first wave body and wrote no logits file at all, which §13 then
    // read as a token flip inside the band region (§14). One named value now decides the wave's plan,
    // and the launch below uses it, so the guard cannot drift from the thing it guards.
    const int wave_fused = 0;
    const bool wave_tile = a.wave_tile != 0 && rows > 1;
    if (wave_tile) {
        const char* why = nullptr;
        if (!attn_tile_ok(c.score_f32, wave_fused, warp_width(), &why)) {
            throw std::runtime_error("attn_decode_layer: the tiled wave form is unavailable: " +
                                     std::string(why) + ". Pass wave_tile=0 for the shipped form.");
        }
    }
    // WHICH FORM OWNS THE STREAMING ROWS (`--attn-fattn`, `AttnLayerArgs::wave_fattn`). Like `wave_tile`
    // this is wave-only, and like it, a refusal is an ERROR and never a fallback: silently running
    // `attn_slim_wave` under `--attn-fattn 1` would price the shipped form twice and report it as the WMMA
    // form. The coverage split itself — WMMA gets the rows §07.3 already streams, the VALU wave keeps the
    // one-pass rows — is `fattn_covers`'s, called from both kernels, so the two sides cannot disagree
    // about who owns a row and no partial slot is ever dropped or double-written.
    const bool wave_fattn = a.wave_fattn != 0 && rows > 1 && a.wave_tile != 0;
    if (a.wave_fattn != 0 && rows > 1 && a.wave_tile == 0) {
        throw std::runtime_error(
            "attn_decode_layer: --attn-fattn 1 requires --attn-tile 1. The WMMA form replaces the TILED "
            "wave's streaming rows; under wave_tile=0 the per-(row,head) attn_core owns the whole wave and "
            "there is nothing here for it to replace, so refusing is the only honest answer.");
    }
    if (wave_fattn) {
        std::string why;
        if (!fattn_plan_ok(c, &why)) {
            throw std::runtime_error("attn_decode_layer: the WMMA wave form is unavailable: " + why +
                                     ". Pass --attn-fattn 0 for the shipped form.");
        }
        if (!a.wave_touch) {
            throw std::runtime_error(
                "attn_decode_layer: --attn-fattn 1 requires --attn-touch 1. The row split lives inside "
                "attn_slim_wave's early-out; the frozen attn_tile_wave spelling carries no such test, so "
                "under --attn-touch 0 both kernels would write the same streaming row's partial. Refusing "
                "is the only answer that cannot double-write a slot.");
        }
    }
    // One cache lane per warp, capped by the block's warp count — `attn_decode.hip.cpp:226`
    // (`tile = std::min(TILE, nw)`), where nw is THREADS/warpSize. On this wave32 part that is 8.
    c.tile = std::min(TILE, THREADS / warp_width());

    // ---- the launch plan ------------------------------------------------------------------------
    // TWO PLANS, and the thing that chooses between them is the row count, because that is what changes
    // the dependency structure — not the lane geometry, which is the same on both sides.
    //
    // ONE ROW (every generation step, and `decode_run --wave-rows 1`): the FUSED plan exactly as P7
    // shipped it — one `attn_core` launch of 8 query-head blocks, `split` deep in y, which recomputes
    // its own head norm + rope (§05.2-§05.3) and does the §05.7 append inline. At split 1 that is the
    // single lane of y the launcher shipped before P7; above it `stage_head`'s `blockIdx.y != 0`
    // early-out is what makes the extra blocks safe to run concurrently — exactly one of them writes
    // the stage buffers and performs the §05.7 append, and none of them reads the appended lane back out
    // of the cache (the core takes lane `pos` from shared memory, which every block computed for
    // itself), so there is no cross-block dependency to order.
    //
    // A WAVE (P8, §06.3): the STAGED plan, because the dependency a single row does not have is exactly
    // the one a wave cannot leave to chance. Row t scores the lanes rows 0..t-1 appended, and in the
    // fused plan each core block appends only its own row's lane, with nothing between the two grid
    // axes to say which ran first — a race that passes on a short prompt and corrupts a long one. So
    // `attn_prep` stages every row's norm, rope, gate and §05.7 append at that row's own position in a
    // launch that has finished before the core starts, and the core READS the staged q/k (`fused = 0`)
    // instead of recomputing them. Two launches per attention layer per wave, where one per token used
    // to be the bill: §06.4 says count them, and `engine/gate.py` (h) does.
    //
    // THE STAGED PLAN HAS READ POINTERS AS WELL AS WRITE POINTERS, AND GETTING ONLY THE WRITE SIDE RIGHT
    // IS WHAT HUNG THE DEVICE. In the staged branch `attn_core` takes its q and k from `CoreArgs::qrope`
    // /`krope`, not from `qg`/`kg`; the fused branch never reads those two fields, so a launcher that
    // flips `fused` to 0 and leaves them null has wave row `qrow` dereferencing a null base at byte
    // `ROW_H * qrow * 2` — the two fault dumps this produced named 0x3000 and 0x1000, i.e. rows 3 and 1.
    // The driver reports it asynchronously (`Memory Fault Error ... kernel: attn_core(CoreArgs)`) while
    // the host sits in a stream sync busy-polling a queue that will never drain: ~10 cores, no logits
    // written, which is how a null pointer spent an afternoon looking like a kernel deadlock. So the
    // core's inputs come from the same buffers `attn_prep` writes, and its `*_out` pointers are
    // deliberately left null: `stage_head`'s early-out is the only thing that suppresses an append, and
    // a wave's core must not append (prep already did, at each row's own slot, in a launch that has
    // already finished — which is where this differs from the prototype host on purpose: it leaves the
    // core's `*_out` set (`attn_decode.hip.cpp:370`), so its wave core re-stages and re-appends the same
    // bytes prep already wrote, a write against other rows' reads that is harmless only because the values
    // match. `attn_prep` is the staging kernel and does not read `fused`; `attn_combine` reads only
    // `part`/`aout`/`gout`/`qg`, all of which `c` carries.
    if (rows == 1) {
        attn_core<<<dim3(N_Q, split), THREADS>>>(c);
    } else {
        attn_prep<<<dim3(N_Q + N_KV, rows), THREADS>>>(c);
        CoreArgs core = c;
        core.fused = wave_fused;        // read the staged rows instead of recomputing them: the plan
                                        // the guard above was priced against
        core.qrope = bf16(a.q_rope);    // what `attn_prep` wrote at ROW_H * qrow
        core.krope = bf16(a.k_rope);    // what it wrote at ROW_KV * qrow, and appended
        core.qrope_out = nullptr;       // no staging, no gate dump, no append from this launch
        core.krope_out = nullptr;
        core.gate_out = nullptr;
        if (wave_tile) {
            // Phase 2: the grid is (KV group, lane slice, query row) and `split` already IS the slice
            // count — `split_for(lanes_last)` here and `tile_split_of(lanes_last)` in the op header are
            // the same expression, so the two forms cannot promise different partial counts from this
            // launcher. One row per block since §20, so the row extent is `rows`, and the widest-row-
            // first inversion lives inside the kernel.
            core.tile_kernel = 1;
            // WHICH SPELLING SCORES THE TOUCH (`--attn-touch`, `AttnLayerArgs::wave_touch`). Same grid,
            // same block extent, same `CoreArgs`, same `part` slot expression — `attn_slim_wave` is
            // `attn_tile_wave`'s body with the score dot's trip count made compile-time and the warp
            // reduce folded by hand, so 116 issue slots per (row, lane, head) touch become 45 and nothing
            // else moves (`engine/attn_slim_kernels.h`, `prototype/attn_valu_slim/results.md` §7).
            // Both kernels are in this binary whatever this says, which is what makes the paired timing
            // one window instead of two builds (`AGENTS.md`: a ruler re-taken in a different window is not
            // an A/B), and it is why `attn_combine` still needs no branch on the form: one address
            // expression is shared by writer and reader (§20.1).
            //
            // A WAVE-ONLY SWITCH, like `wave_tile` itself: a one-row body is `attn_core`, so no decode bit
            // can move under it. It cannot degrade into the other side either — the two kernels are both
            // the tiled form, so there is nothing here for a fallback to fall back to.
            if (a.wave_touch) {
                // Under `--attn-fattn 1` this kernel keeps ONLY the one-pass rows: `fattn_cover` is the
                // shared split flag and the early-out lives inside the body, before any barrier, so the
                // covered rows are genuinely this launch's business and not a partial write it abandons.
                core.fattn_cover = wave_fattn ? 1 : 0;
                attn_slim_wave<<<dim3(N_KV, split, tile_rtiles_of(rows)), ATTN_TILE_THREADS>>>(core);
                if (wave_fattn) {
                    // ONE EXTRA LAUNCH, and a different grid from the VALU wave's on purpose: this form's
                    // block is ONE WARP covering 16 query rows of ONE head, so the head moves into
                    // `blockIdx.x` (where the shipped wave puts the KV group) and the row-block extent is
                    // `fattn_rtiles_of` = ceil(rows/16) instead of `tile_rtiles_of` = rows. Same total warp
                    // count — 8 heads x rows/16 blocks x 1 warp = rows x 8 warps either way — and the shape
                    // is what the spill gate decided (`engine/attn_fattn_kernels.h`, Q placement).
                    // A wave whose rows are ALL one-pass still launches it and every block returns at its
                    // first ballot: an inert launch is a cheap, visible cost, a silently-skipped one is not.
                    attn_fattn_wave<<<dim3(N_Q, split, fattn_rtiles_of(rows)), FATTN_THREADS>>>(core);
                    fattn_echo(rows, split, c.lvalid, c.nrows);
                }
            } else {
                attn_tile_wave<<<dim3(N_KV, split, tile_rtiles_of(rows)), ATTN_TILE_THREADS>>>(core);
            }
        } else {
            attn_core<<<dim3(N_Q, split, rows), THREADS>>>(core);
        }
    }
    if (split > 1) {
        // The merge: `(max,sum,acc)` per (row, lane block) → one row, then §05.6's gate. Same algebra
        // as the tile loop, same lifetime (A-17: it never leaves this launch). A row the staging fits
        // is skipped here — its single core block already finished it — which is what lets a wave carry
        // one-pass and streaming rows in the same grid. The grid's x extent is N_Q either way, AND SO IS
        // THE PARTIAL LAYOUT: `attn_tile_wave` writes `attn_core`'s slot expression, keyed by the FULL
        // head, so one merge serves both forms (results.md §20 — two layouts that each agreed with
        // themselves is exactly what §19's aliasing was). `tile_kernel` is bookkeeping for the negative
        // controls and for which CORE ran; it must not decide an address.
        c.tile_kernel = wave_tile ? 1 : 0;
        attn_combine<<<dim3(N_Q, rows), THREADS>>>(c);
    }
    // One site per attention layer per body, whatever number of kernels that site submitted: the point
    // the gate has to see is the wave going through once instead of once per token, and the 1-to-3
    // kernel factor is lane depth, which is the same on both sides of the A/B. `launch_stats.h` owns
    // the wording; `gate.py`'s `launch_plan()["attn"]` is the same number from §07.
    launchstats::attn().record(rows);
}

}  // namespace engine::kernels
