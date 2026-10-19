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
// past the `part` scratch. 64 admits 64 full waves of `MAXSC` = 32,768 lanes, which is the ceiling
// item L asked for; it is `ceil(32768 / MAXSC)`, not a taste number, and the ONLY thing the raise
// costs is the scratch named below. Nothing in `data/refs/` reaches it: `wide4096f`'s 4096 prompt + 8
// generated = 4104 lanes is still the widest PINNED plan (D44), so the whole band between 4,608 and
// 32,768 is funded but unpinned, and `tools/longctx_check.sh` is what exercises it. No address reads
// this cap — `tools/attn_ceiling_proof.py` [A]/[B3] is the witness — so every plan the old 9 ran is
// bit-identical after the raise, and only the scratch and the refusal boundary move.
//
// The bytes, stated because they are the whole admission: `attn_part_floats(kMaxWaveRows)` moves
// 38,043,648 B -> 270,532,608 B (36.28 MiB -> 258 MiB), ~0.78% of this box's ~34 GiB device pool per
// scratch instance. That charge is PER SCRATCH, because `allocate_scratch()` sizes from this constant
// and not from the run's cap, so a 20-token request pays 258 MiB of partials exactly as a 32k one
// does. For one request, or eight concurrent 32k requests (~5.2 GiB of state+scratch between them),
// that is free at this model's scale; a many-request server is where it stops being free, and the fix
// there is to size by `split_for(cap)` instead of by this constant — a separate decision with its own
// memory argument, deliberately NOT taken here, because one constant that the gate can read is what
// keeps the refusal and the allocation from disagreeing (§20.1's aliasing class).
//
// `engine/gate.py::lane_ceiling()` READS both halves of that product out of the files that enforce
// them, so neither constant can be retyped into the gate.
constexpr int kAttnMaxSplit = 64;

// The split a row of `lanes` needs: `split = ceil(lanes / staging)`, so no block is handed more lanes
// than the staging width the split was derived from (`ceil(n / ceil(n/s)) <= s`).
int split_for(int lanes) { return (lanes + MAXSC - 1) / MAXSC; }

// L-c (`--attn-split-cap`, bet A9-L-c): the launch's split as an UPPER BOUND on the geometry's own.
// `split_for` answers "how few lane blocks may this row be cut into without one block exceeding the
// staging"; a cap asks for MORE lanes per block than that, which the row's slice arithmetic absorbs
// (`span = ceil(nrow / s_row)` in every core, so the round loop just runs longer) and which therefore
// needs no second rule — one expression, one authority, and the uncapped value is `split_for` verbatim.
// `kAttnSplitCapOff` leaves it alone. A cap of 1 is NOT clamped here: it is refused by the caller below,
// because a cap of 1 on a wide row is the one mis-normalisation §07.3 exists to make unreachable.
int cap_split(int split, int cap) { return cap >= 2 ? std::min(split, cap) : split; }

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
void fattn_echo(int rows, int split, int lvalid, int nrows, const void* launched,
                const char* roll_name, const char* qstage_name) {
    static std::string last;
    // `launched` is the function pointer of the kernel THIS LAUNCH submitted, not a name typed beside the
    // banner. With four spellings in one binary that distinction is the whole value of `vgpr=`/`lds=`/
    // `scratch=`/`wf_per_cu=` here: read off `attn_fattn_wave` unconditionally, the banner would describe the
    // shipped kernel while the run was pricing an alternative one, and the two sides of the A/B would print
    // byte-identical banners while their milliseconds — and their whole residency claim — differed. `roll=` and
    // `q=` name the spelling in the line, so the once-per-geometry dedup below cannot collapse two sides into
    // one printed banner, and `lds=`/`wf_per_cu=` are the fields the A6-G1 canary reads (32768/6 shipped against
    // 20480/9 for `unstaged8`).
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
        hipFuncGetAttributes(&at, launched) == hipSuccess;
    int blocks_per_cu = -1;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, launched,
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
                  "ATTN_FATTN form=fattn16 q=%s roll=%s warps=%d vgpr=%d lds=%d scratch=%d vtiles=%d"
                  " wf_per_cu=%d"
                  " wave_rows=%d split=%d fattn_rows=%d/%d fallback_rows=%d"
                  " covered_rows_pct=%.1f covered_work_pct=%.1f%s",
                  qstage_name,
                  roll_name,
                  FATTN_WARPS,
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

// THE L-c CANARY (`--attn-split-cap`, bet A9-L-c), one line per DISTINCT launch geometry — `fattn_echo`'s
// dedup rule and for the same reason: 48 attention layers call the launcher with identical numbers, and 48
// identical lines is a line a reader skims past.
//
// WHY A SEPARATE LINE EXISTS WHEN `ATTN_FATTN` ALREADY PRINTS `split=`: a paired A/B on this box is owed a
// behavioural canary, because `engine/build.sh` is not bit-reproducible and a byte-compare of two binaries
// proves only that they are two builds (AGENTS.md, the 2026-10-17 stale-binary row — the first §13 A/B timed
// a pre-change binary as its `new` side and reported a clean null while every in-binary self-check stayed
// green, because a check compiled into the binary under test is blind to that one failure mode). So the
// canary must be (a) a field the change is REQUIRED to move, (b) readable in the same breath as the number,
// and (c) ABSENT from the pre-change binary, which is exactly what a new line prefix gives a harness: base
// prints no `SPLITCAP` line at all, the new side prints one naming its cap, its effective split, and the
// lane span that cap produced. `span=` is the field that makes the central risk of the bet checkable rather
// than assumed: if a capped run prints `span > 512` and still produces sane logits, the cap widened a
// block's slice instead of dropping lanes.
void splitcap_echo(int rows, int lanes_last, int split_natural, int split_eff, int cap) {
    static std::string last;
    char buf[192];
    std::snprintf(buf, sizeof(buf), "SPLITCAP split=%d cap=%d lanes=%d natural=%d rows=%d span=%d",
                  split_eff, cap, lanes_last, split_natural, rows,
                  (split_eff + lanes_last - 1) / split_eff);
    const std::string line(buf);
    if (line == last) return;
    last = line;
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}
}  // namespace

std::size_t attn_part_floats(int rows) {    // `attn_core`'s own partial layout: one `(max, sum, acc[DH])` tuple per (query head, split
    // block), F32 — `a.part + ((qrow*N_Q + head)*split + blockIdx.y)*(DH + 2)` in the kernel. The
    // slot index never exceeds `rows*N_Q*split - 1`, and the throw in `attn_decode_layer` is what
    // keeps `split <= kAttnMaxSplit`, so sizing for the cap cannot overrun. At `kMaxWaveRows` = 512
    // it is 512*8*258 = 1,056,768 F32 = 4,227,072 B PER UNIT OF SPLIT, so the shipped 64 sizes
    // 270,532,608 B (516 KiB/token) where the old 9 sized 38,043,648 B (72.5625 KiB/token): ~0.78 %
    // of this box's ~34 GiB pool per scratch instance, and still 31x below the point where a kernel's
    // int32 element index is the question (67,633,152 elements at split 64) — and the widest buffer
    // one request touches is not this anyway, it is one KV cache pair, 2*cap*DH = 201 MB at 32,768.
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

    // ---- L-c: the split the LAUNCH runs (`--attn-split-cap`, bet A9-L-c) -----------------------------
    // TWO names, because two different questions are being asked of one number, and collapsing them is how a
    // cap would quietly widen the lane ceiling. `split` above is the count the lane GEOMETRY forces, and the
    // ceiling check is priced on it: a request of 40,000 lanes still refuses with `needs split=79` whatever
    // the cap says, so `MAXSC × kAttnMaxSplit` remains the engine's lane ceiling and `engine/gate.py`'
    // `lane_ceiling()` — which reads both factors out of these two files and asserts the run fits the product
    // — keeps describing the refusal it is meant to describe. `split_eff` is what gets launched: a shallower
    // cut of the SAME lane axis, never a deeper one, and never below 2 for a wide row (cap < 2 is refused
    // outright, because the one-pass branch of `attn_core` cannot be handed more than `MAXSC` lanes and that
    // branch is only unreachable while `split >= 2` — see the note above `split_lane`).
    //
    // NOTHING ELSE SIZES ON IT. The partial scratch is `attn_part_floats(rows)` = `rows*N_Q*kAttnMaxSplit*
    // (DH+2)`, i.e. sized for the CAP of split rather than for this run's split, so a shallower split only
    // leaves the tail of every row's slab unwritten — the slot expression strides by `a.split` and
    // `blockIdx.y < split_eff`, both of which move together. The scratch bound therefore cannot be overrun by
    // capping, only under-used, and that is why this knob costs no allocation, no LDS and no register.
    if (a.split_cap == 1) {
        throw std::runtime_error(
            "attn_decode_layer: split_cap == 1 is not a shallower cut, it is a different route: with one"
            " lane block a row wider than the staging is served by ONE block, which is the mis-normalisation"
            " §07.3 exists to make unreachable (the one-pass softmax divides inside its own block and a"
            " partial that has divided cannot be re-merged). Use split_cap = 0 (no cap) or >= 2;");
    }
    const int split_eff = cap_split(split, a.split_cap);

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
    c.split = split_eff;      // the CAPPED count: every grid below, `blocks_for_row` and `attn_combine`
                              // read this one field, so writer and merge cannot disagree about how many
                              // partials a row promises (L-c, `--attn-split-cap`)
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
    const bool wave_fattn = a.wave_fattn != 0 && rows > 1 && a.wave_tile != 0 && split_eff >= 2;
    if (a.wave_fattn != 0 && rows > 1 && split_eff < 2) {
        // Route (A)'s inert case, and it is structural rather than a choice: a one-pass wave allocates
        // NO streaming partials (`split == 1`), so the form has nowhere to write. Skipping it and saying
        // so is the inertness the split promised; launching it would dereference a null `part`, and
        // refusing would make the flag fatal on every short wave instead of inert on it.
        std::printf("ATTN_FATTN_SKIP rows=0/%d: a one-pass wave (lanes <= MAXSC %d) allocates no "
                    "streaming partials, so the WMMA form owns nothing and is not launched\n",
                    rows, MAXSC);
    }
    if (a.wave_fattn != 0 && rows > 1 && a.wave_tile == 0) {
        throw std::runtime_error(
            "attn_decode_layer: --attn-fattn 1 requires --attn-tile 1. The WMMA form replaces the TILED "
            "wave's streaming rows; under wave_tile=0 the per-(row,head) attn_core owns the whole wave and "
            "there is nothing here for it to replace, so refusing is the only honest answer.");
    }
    // THE CAP'S OWN REFUSAL (`--attn-split-cap`, bet A9-L-c) — an error, never a fallback, and never silent.
    // A cap makes a block's lane slice WIDER: `span = ceil(lanes_last / split_eff)`, which is <= MAXSC exactly
    // when the split is the geometry's own and > MAXSC whenever this cap binds. That is safe for the WMMA wave
    // and unsafe for the two VALU tiled spellings, and the difference is a declared buffer, not a style point:
    // `attn_fattn_wave` stages `FATTN_MLANES` = 16 lanes a round and its round loop runs from `lo_all` to
    // `hi_all` (`engine/attn_fattn_kernels.h:771`), so a wider slice is more rounds and nothing else; but
    // `attn_slim_wave` and `attn_tile_wave` stage a block's WHOLE slice as exact-BF16 scores in
    // `__shared__ hip_bfloat16 s_sc[ATTN_TILE_G][MAXSC]` (`engine/attn_slim_kernels.h:157`, written at `:232`
    // with `rr < ns == span`; same buffer in `attn_tile_body`) — so `span > MAXSC` is an LDS overrun whose
    // device signature is a fault presented as a hang (AGENTS.md, the wave-only-path row). `attn_core` —
    // `wave_tile=0`, and every one-row body — needs no guard: its streaming branch keeps a `tile`-sized
    // window in `sc` (`prototype/attn_decode/attn_decode_kernels.h:355-360`) and its `s_row == 1` branch is
    // only reachable at `lanes_of <= MAXSC`, so no branch there indexes `sc` by the span.
    // This is also what `prototype/attn_m2` already refuses for the same geometry ("a `split` at least
    // `split_for_lanes(lanes_last)`", `prototype/attn_m2/attn_m16.hip.cpp:1069-1075`), so the engine and that
    // prototype now have the same rule instead of two authorities with different answers.
    if (split_eff != split && rows > 1 && a.wave_tile != 0 && a.wave_fattn == 0) {
        throw std::runtime_error("attn_decode_layer: --attn-split-cap " + std::to_string(a.split_cap) +
                                 " binds here: widest row is " + std::to_string(lanes_last) +
                                 " lanes, so a split of " + std::to_string(split_eff) + " gives each block " +
                                 std::to_string((lanes_last + split_eff - 1) / split_eff) + " lanes"
                                 " against a staging of MAXSC=" + std::to_string(MAXSC) +
                                 ". The VALU tiled wave (`--attn-fattn 0`) stages a whole block slice in"
                                 " `s_sc[ATTN_TILE_G][MAXSC]`, so a block wider than the staging is an LDS"
                                 " overrun, not a slower run. The WMMA wave stages 16 lanes a round and"
                                 " absorbs a wider slice as more rounds: run --attn-fattn 1 (the shipping"
                                 " default) with a cap, or drop the cap.");
    }
    // The canary, printed only for a plan this launcher is actually about to submit (a refused run must not
    // print a line claiming it ran). See `splitcap_echo` for why a change like this owes a canary at all.
    splitcap_echo(rows, lanes_last, split, split_eff, a.split_cap);
    if (wave_fattn) {
        // NOTE: the plan guard that used to sit here read `c.fused`, which is the ONE-ROW core args — a
        // struct whose `fused` field the wave launcher never writes (`core.fused = wave_fused` is set
        // below). So it refused every fattn run as "fused plan" no matter which plan was real, and a
        // one-pass wave that should have been inert exited 1 instead. A guard may only read the struct
        // the launch uses; it now sits directly above the launch, on `core`, where every field is final.
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
        attn_core<<<dim3(N_Q, split_eff), THREADS>>>(c);
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
                attn_slim_wave<<<dim3(N_KV, split_eff, tile_rtiles_of(rows)), ATTN_TILE_THREADS>>>(core);
                if (wave_fattn) {
                    // The plan guard, in the one place it can be honest: `core` is complete here, so this
                    // reads the arguments this launch will actually dereference. An error, never a fallback
                    // — a form that silently does not run would price the shipped kernel and report it as WMMA.
                    std::string fwhy;
                    if (!fattn_plan_ok(core, &fwhy)) {
                        throw std::runtime_error(
                            "attn_decode_layer: the WMMA wave form is unavailable: " + fwhy +
                            ". Pass --attn-fattn 0 for the shipped form.");
                    }
                    // ONE EXTRA LAUNCH, and a different grid from the VALU wave's on purpose: this form's
                    // block is FATTN_WARPS warps covering FATTN_ROWS_PER_BLOCK query rows of ONE head (16 rows
                    // per warp, one shared K/V stage), so the head moves into `blockIdx.x` (where the shipped
                    // wave puts the KV group) and the row extent is `fattn_rtiles_of` = ceil(rows/16) instead
                    // of `tile_rtiles_of` = rows. Same total warp count — 8 heads x rows/16 blocks x 1 warp =
                    // rows x 8 warps either way — and the shape is what the spill gate decided
                    // (`engine/attn_fattn_kernels.h`, Q placement).
                    // A wave whose rows are ALL one-pass still launches it and every block returns at its
                    // first ballot: an inert launch is a cheap, visible cost, a silently-skipped one is not.
                    //
                    // WHICH SPELLING THAT LAUNCH RUNS. Two knobs pick one of four kernels here:
                    // `--attn-fattn-roll` (the score roll, bet A4-L1) and `--attn-fattn-qstage` (where Q lives
                    // and how deep the epilogue stages, bet A6-G1). The four spellings are four `__global__`s in
                    // this TU and this is the only place one is picked, so a knob moves a FUNCTION and nothing
                    // else: same grid expression, same block extent, same `CoreArgs`, ONE launch either way, and
                    // the only thing that differs between `stage` and `unstaged8` is what that kernel declares as
                    // its LDS. That is what makes A4-BET's and A6-BET's `launches`-identical requirement
                    // checkable rather than hoped for — §41 closed launch count as a variable and neither bet may
                    // reopen it, so the A/B harness asserts the equality instead of trusting it
                    // (`prototype/attn_percost/run_ab_a6g0.sh`, `run_ab_a4l1.sh`). Like every knob above it, an
                    // error is never a fallback: a side that silently ran another spelling would price one kernel
                    // twice and report the second price as the second spelling's win.
                    //
                    // Each branch names its kernel exactly once and records the same pointer it launched, so the
                    // banner cannot describe a kernel other than the one this launch submitted.
                    const bool roll_two = a.fattn_roll == FattnRoll::kTwo;
                    const bool q_unstaged = a.fattn_qstage == FattnQStage::kUnstaged8;
                    const dim3 fattn_grid(N_Q, split_eff, fattn_rtiles_of(rows));
                    const void* fattn_kernel = nullptr;
                    if (q_unstaged && roll_two) {
                        attn_fattn_wave_two_u8<<<fattn_grid, FATTN_THREADS>>>(core);
                        fattn_kernel = reinterpret_cast<const void*>(attn_fattn_wave_two_u8);
                    } else if (q_unstaged) {
                        attn_fattn_wave_u8<<<fattn_grid, FATTN_THREADS>>>(core);
                        fattn_kernel = reinterpret_cast<const void*>(attn_fattn_wave_u8);
                    } else if (roll_two) {
                        attn_fattn_wave_two<<<fattn_grid, FATTN_THREADS>>>(core);
                        fattn_kernel = reinterpret_cast<const void*>(attn_fattn_wave_two);
                    } else {
                        attn_fattn_wave<<<fattn_grid, FATTN_THREADS>>>(core);
                        fattn_kernel = reinterpret_cast<const void*>(attn_fattn_wave);
                    }
                    fattn_echo(rows, split_eff, c.lvalid, c.nrows, fattn_kernel,
                               fattn_roll_name(a.fattn_roll), fattn_qstage_name(a.fattn_qstage));
                }
            } else {
                attn_tile_wave<<<dim3(N_KV, split_eff, tile_rtiles_of(rows)), ATTN_TILE_THREADS>>>(core);
            }
        } else {
            attn_core<<<dim3(N_Q, split_eff, rows), THREADS>>>(core);
        }
    }
    if (split_eff > 1) {
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

// Parse-or-throw, `parse_gdn_geom`'s rule and for the same reason: this flag names which side of an A/B a run
// is, and the two sides are NOT bit-identical (two accumulators re-associate the score dot), so a value that
// silently fell back to the default would price the shipped roll twice and report the second price as the
// second accumulator's win. An unknown value is an error, never a default.
FattnRoll parse_fattn_roll(const std::string& name) {
    if (name == "single") return FattnRoll::kSingle;
    if (name == "two") return FattnRoll::kTwo;
    throw std::runtime_error("--attn-fattn-roll takes `single` (the shipped score roll: ONE accumulator,"
                             " 16 dependent v_wmma trips — THE DEFAULT) or `two` (bet A4-L1: two"
                             " accumulators, both tiles of a pair fetched before either matrix op issues,"
                             " two dependent chains of 8; engine/attn_fattn_kernels.h). Got '" + name +
                             "'. Refusing rather than guessing, because this names which roll produced the"
                             " bits in this log");
}

const char* fattn_roll_name(FattnRoll r) {
    return r == FattnRoll::kTwo ? "two" : "single";
}

// Parse-or-throw, `parse_fattn_roll`'s rule and for a sharper reason: this knob carries a NUMERICS claim (the
// `unstaged8` side is owed byte identity with the default, `prototype/attn_percost/A6-G1-impl.md`), so a value
// that silently fell back to the default would price the shipped form twice and report the second price as the
// un-stage's win — and it would do so while looking green. An unknown value is an error, never a default.
FattnQStage parse_fattn_qstage(const std::string& name) {
    if (name == "stage") return FattnQStage::kStaged;
    if (name == "unstaged8") return FattnQStage::kUnstaged8;
    throw std::runtime_error("--attn-fattn-qstage takes `stage` (THE DEFAULT, the shipped spelling: Q staged"
                             " in a per-warp 8 KiB stage, the epilogue transposing 16 rows at a time in 2"
                             " passes, declared LDS 32,768 B = 2 blocks/CU = 6 wf/CU) or `unstaged8` (bet"
                             " A6-G1: Q read from global at the use site, the epilogue staging FATTN_TRG = 8"
                             " rows a pass in 4 passes, declared LDS 20,480 B = 3 blocks/CU = 9 wf/CU at 185"
                             " VGPR and spill 0; engine/attn_fattn_kernels.h placement (E)). Got '" + name +
                             "'. Refusing rather than guessing, because this names which kernel produced the"
                             " bits in this log and the residency claim is read off its `lds=`/`wf_per_cu=`");
}

const char* fattn_qstage_name(FattnQStage s) {
    return s == FattnQStage::kUnstaged8 ? "unstaged8" : "stage";
}

}  // namespace engine::kernels
