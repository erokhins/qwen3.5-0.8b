/*
 * prototype/attn_decode/attn_decode.hip.cpp — the §05 gated full-attention decode step (k = 1).
 *
 * Scope: everything between the projections and `o_proj`. One query row arrives as the pinned
 * projection outputs (q_proj [4096] = [8 heads][q 256 | gate 256], k_proj [512], v_proj [512]),
 * the request's KV cache and the position; out come `attn_out_pre_gate` and the gated operand
 * `o_proj` consumes. §05.1 de-interleave, §05.2 zero-centred per-head RMSNorm, §05.3 partial rope
 * (first ROT=64 dims only), §05.7 cache append, §05.5 GQA (8 query heads over 2 KV heads) with
 * online (max,sum) softmax, §05.6 output gate. `o_proj` is prototype/proj_gemv's.
 *
 * ROUNDING POLICY — measured against the pins, not chosen (results.md has the table):
 *   norm   z   = bf16((x * rsqrt(mean(x^2)+eps)) * (1+w))          one round     (§05.2)
 *   rope   c,s = bf16(cos/sin) ; z' = bf16(bf16(z_i*c) + bf16(rot*s))  three rounds (§05.3)
 *   score  s   = bf16(q.k) * SCALE                                 scores live in BF16 (§10.3)
 *   softmax      F32 exp/sum over the BF16 scores                                             (§05.5)
 *   a            = bf16(sum_j w_j v_j)                             one round
 *   gate out     = a * bf16(sigmoid(gate))                                                    (§05.6)
 * The BF16 rounding of cos/sin is not in §05.3's text; it is what the pins say, and rounding them
 * in F32 instead misses 52-97 elements of every 2048-element q_rope (speccheck's apply_rope does
 * exactly that). See results.md — this is the single finding that makes this op reproducible.
 *
 * TWO LAUNCH PLANS, one math:
 *   --launch staged  prep (norm+rope+append) -> core -> combine (only when split>1)
 *   --launch fused   the core recomputes its own head norm+rope and appends inline: 1 launch at
 *                    split 1, 2 above. run.py requires the two plans to agree bit-for-bit, so the
 *                    fused form cannot drift away from the gated one.
 * W5 wants the fused form: at cache lengths 18..277 this op moves ~0.6 MB per layer, so it is
 * launch-latency-bound, not bandwidth-bound.
 *
 * SOFTMAX. `--softmax online` (the engine form, §07.3) reduces the row in tiles with (max,sum)
 * rescaling; the accumulators belong to this query row and die with it (A-17). `--softmax full` is
 * the reference form: max first, sum second, weights rounded to BF16, then the P·V product. Both
 * are gated against the same pins and the gap between them is measured by floor_probe.py — a band
 * copied from a dot product is what §10.1 warns about.
 *
 * CACHE LENGTH IS THE TRAP. The kernel takes `lvalid` (lanes this query may attend to) and `cap`
 * (rows actually allocated). `readpast` and `dropcurrent` below are the two ways an engine gets
 * that wrong, and run.py exercises cache lengths 1..277 against pinned data.
 *
 * Injected bugs — each with a declared must-catch / must-NOT-catch stage (the mlp_gemv/proj_gemv
 * convention: a negative that cannot fail is worse than none, and one that fails everywhere is
 * measuring nothing):
 *   qswap       q_proj's two halves swapped (§05.1)
 *   gatehalf    gate read as the contiguous second half — the [all q | all gate] reading (§05.1)
 *   ropefull    rope over all 256 dims in (i, i+128) pairs instead of the first 64 (§05.3)
 *   ropenone    no rope at all
 *   gqaoff      kappa(i) off by one: query head i reads KV head (i/4 + 1) mod 2 (§05.5)
 *   nomax       (max,sum) state not rescaled when the max moves — the A-17 online-softmax bug
 *   carry       the row seeded from the PREVIOUS query's (max,sum,acc) (§07.3's own failure)
 *   readpast    attend to `cap` lanes instead of `lvalid` — the causal mask is gone
 *   dropcurrent attend to lanes < pos — the appended token forgotten
 *   dimhalf     the score dot accumulates 128 of the 256 head dims
 *   slotk       the key appended to the wrong cache slot (§05.7)
 *   slotv       the value appended to the wrong cache slot (§05.7)
 * `--score-f32` is a policy variant (skip the BF16 rounding of the score matrix), reported by
 * floor_probe.py rather than judged as a negative.
 *
 * Build: prototype/attn_decode/build.sh  (-O3, -Wl,-rpath,/opt/rocm/lib)
 * Run:   ./attn_decode --mode once  --plan PLAN [--launch staged|fused] [--split N]
 *                      [--softmax online|full] [--neg NAME] [--score-f32] [--tile-kernel]
 *        ./attn_decode --mode burst --plan PLAN --iters N [--tile N] [--write-out] [same knobs]
 * Validate: prototype/attn_decode/run.py
 *
 * PLAN is a text file, one case per line:
 *   NAME QG KG VG KCACHE VCACHE QNORM KNORM POS LCACHE OUTDIR
 * QG/KG/VG are raw little-endian BF16 rows (4096/512/512 elements); KCACHE/VCACHE are raw BF16
 * [2][cap][256] with cap = filesize/(2*256*2), whose first LCACHE-1 lanes hold the existing cache;
 * QNORM/KNORM are 256 BF16. OUTDIR receives q_rope, k_rope, gate, rope_cs, attn_pre_gate, gated,
 * cache_keys, cache_values (raw BF16) in `once` mode, and in `burst` mode only with --write-out.
 * `rope_cs` is the 128-value cos||sin table built for POS: the gate uses it to prove the in-kernel
 * transcendentals match the reference table bit-for-bit instead of trusting them.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "attn_decode_kernels.h"

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ------------------------------------------------------------------ host side
static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    std::vector<char> b((size_t)f.tellg());
    f.seekg(0);
    f.read(b.data(), (std::streamsize)b.size());
    if ((size_t)f.gcount() != b.size()) {
        fprintf(stderr, "%s: short read\n", p.c_str());
        exit(1);
    }
    return b;
}

static hip_bfloat16* upload(const std::string& path, size_t expect_elems) {    const std::vector<char> raw = slurp(path);
    if (raw.size() != expect_elems * 2) {
        fprintf(stderr, "%s: %zu bytes, expected %zu BF16 elements\n", path.c_str(), raw.size(),
                expect_elems);
        exit(1);
    }
    hip_bfloat16* d = nullptr;
    CK(hipMalloc(&d, (size_t)raw.size()));
    CK(hipMemcpy(d, raw.data(), raw.size(), hipMemcpyHostToDevice));
    return d;
}

static void dump_file(const std::string& path, const void* dptr, size_t elems) {
    std::vector<char> h(elems * 2);
    CK(hipMemcpy(h.data(), const_cast<void*>(dptr), elems * 2, hipMemcpyDeviceToHost));
    std::ofstream of(path, std::ios::binary);
    of.write(h.data(), (std::streamsize)h.size());
}

// Shared across the whole plan: --neg carry reproduces §07.3's bug, which is one query's softmax
// state surviving into the NEXT query, so the buffer has to outlive a case.
static float* g_stale = nullptr;

struct Case {
    std::string name, outdir;
    int pos = 0, lvalid = 0, cap = 0;
    // P8 (§06.3): how many query rows this case is. 1 is every plan line written before P8 — the field
    // is optional and defaults to 1 — and a WAVE line carries one pinned prefill span: `pos`/`lvalid`
    // describe the FIRST row, and row r sits at position pos+r with lvalid+r lanes visible to it.
    int nrows = 1;
    hip_bfloat16 *qg = nullptr, *kg = nullptr, *vg = nullptr, *qn = nullptr, *kn = nullptr;
    std::vector<hip_bfloat16*> kc, vc;          // cache replicas (--tile)
    hip_bfloat16 *qrope = nullptr, *krope = nullptr, *gate = nullptr, *aout = nullptr;
    hip_bfloat16 *gated = nullptr, *rcs = nullptr;
    float *part = nullptr, *stale = nullptr;

    int lanes_last() const { return lvalid + nrows - 1; }

    // Bytes one launch moves, for the bandwidth sanity check only: at these cache lengths the op
    // is launch-latency-bound, so GB/s here measures how little it moves, not how well it moves it.
    size_t bytes(int split) const {
        const size_t rd = ((size_t)4096 * 2 + (size_t)512 * 2 + (size_t)512 * 2
                           + (size_t)256 * 2 * 2) * nrows
                          + (size_t)lanes_last() * 2048;
        const size_t wr = ((size_t)(DH * N_Q) * 2 * 2 + (size_t)(DH * N_KV) * 2 * 2
                           + (size_t)ROT * 2 * 2 + (size_t)(DH * N_KV) * 2 * 2) * nrows;
        return rd + wr + (split > 1 ? (size_t)nrows * N_Q * split * (DH + 2) * 4 : 0);
    }
};

int main(int argc, char** argv) {
    std::string mode = "once", plan_path, negname = "none", launch = "staged", softmax = "online";
    int split = 1, iters = 30, tile = 1, tile_rep = 1;
    bool score_f32 = false, write_out = false, tile_kernel = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--plan") plan_path = next();
        else if (a == "--launch") launch = next();
        else if (a == "--softmax") softmax = next();
        else if (a == "--split") split = atoi(next().c_str());
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--tile") tile_rep = atoi(next().c_str());
        else if (a == "--neg") negname = next();
        else if (a == "--score-f32") score_f32 = true;
        else if (a == "--tile-kernel") tile_kernel = true;
        else if (a == "--write-out") write_out = true;
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }
    const int neg = negname == "none" ? NEG_NONE
                    : negname == "qswap" ? NEG_QSWAP
                    : negname == "gatehalf" ? NEG_GATEHALF
                    : negname == "ropefull" ? NEG_ROPEFULL
                    : negname == "ropenone" ? NEG_ROPENONE
                    : negname == "gqaoff" ? NEG_GQA
                    : negname == "nomax" ? NEG_NOMAX
                    : negname == "carry" ? NEG_CARRY
                    : negname == "readpast" ? NEG_READPAST
                    : negname == "dropcurrent" ? NEG_DROPCUR
                    : negname == "dimhalf" ? NEG_DIMHALF
                    : negname == "slotk" ? NEG_SLOTK
                    : negname == "slotv" ? NEG_SLOTV : -1;
    if (neg < 0) {
        fprintf(stderr,
                "--neg must be none|qswap|gatehalf|ropefull|ropenone|gqaoff|nomax|carry|readpast|"
                "dropcurrent|dimhalf|slotk|slotv\n");
        return 1;
    }
    if (softmax != "online" && softmax != "full") {
        fprintf(stderr, "--softmax must be online|full\n");
        return 1;
    }
    const bool full = softmax == "full";
    if (launch != "staged" && launch != "fused") {
        fprintf(stderr, "--launch must be staged|fused\n");
        return 1;
    }
    if (split < 1) {
        fprintf(stderr, "--split must be >= 1\n");
        return 1;
    }
    if (neg == NEG_CARRY && split != 1) {
        fprintf(stderr, "--neg carry is only defined at --split 1 (one state per query row)\n");
        return 1;
    }
    // `--softmax full` used to mean "one block per row", and this is where that was refused with
    // `--split != 1`. P8 moved the decision from the launch to the ROW (`blocks_for_row`): a full row
    // that fits `MAXSC` takes exactly one block and the pins' bits, whatever the grid's depth, and a row
    // that does not fit streams across the launch's `split` blocks and is merged by `attn_combine`. So
    // `--softmax full --split 4` is now a meaningful request — one-pass wherever it fits, streaming
    // above it — which is exactly what one wave of mixed lane counts needs. What would still be
    // mis-normalisation is a row WIDER than the staging served by ONE block, and the per-case check
    // below (`lanes_last / split > MAXSC`) is what refuses that.
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    const int nw = THREADS / prop.warpSize;
    if (prop.warpSize != 32 && prop.warpSize != 64) {
        fprintf(stderr, "warpSize %d unsupported (TILE assumes one warp per lane)\n",
                prop.warpSize);
        return 1;
    }
    tile = std::min(TILE, nw);
    if (tile_kernel) {
        // Phase 2 A/B switch: every core launch of this run goes through `attn_tile_wave` instead of
        // `attn_core`, so one build measures both forms in one window. The refusals are
        // `attn_tile_ok`'s — the same list the engine launcher refuses — so a plan this gate can
        // price is a plan the engine can run, and vice versa.
        const char* why = nullptr;
        if (!attn_tile_ok(score_f32 ? 1 : 0, launch == "fused" ? 1 : 0, prop.warpSize, &why)) {
            fprintf(stderr, "--tile-kernel refused: %s\n", why);
            return 1;
        }
        if (neg == NEG_CARRY) {
            // `carry` seeds a row's softmax state from the PREVIOUS query's. In the shipped form
            // consecutive queries are blocks of one launch and that is expressible; in the tiled form
            // since §20 ONE QUERY ROW IS ONE BLOCK, so there is no row-to-row leak left to inject —
            // the only state a tile block carries is its own row's, per warp. Refusing is still the
            // honest answer rather than silently no-oping, and the surviving analogue (a state leaking
            // between two HEADS of the one block, which is what §12's row tiling would have permitted)
            // is injected and gated in the CPU mirror as `headcarry`, where the must-catch set follows
            // from the block geometry: heads with h%G != 0 move, the group's first head cannot.
            fprintf(stderr, "--tile-kernel refuses --neg carry: at one row per block (results.md §20) "
                            "there is no row-to-row state to leak; the head-to-head analogue is "
                            "`attn_tile_model.py gate --negative` (`headcarry`)\n");
            return 1;
        }
    }

    CK(hipMalloc(&g_stale, ((size_t)N_Q * 2 + (size_t)N_Q * DH) * 4));
    CK(hipMemset(g_stale, 0, ((size_t)N_Q * 2 + (size_t)N_Q * DH) * 4));

    std::vector<Case> cases;
    {
        std::ifstream pf(plan_path);
        if (!pf) {
            fprintf(stderr, "cannot open plan %s\n", plan_path.c_str());
            return 1;
        }
        std::string line;
        int ln = 0;
        while (std::getline(pf, line)) {
            ++ln;
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string qgf, kgf, vgf, kcf, vcf, qnf, knf;
            Case c;
            if (!(ss >> c.name >> qgf >> kgf >> vgf >> kcf >> vcf >> qnf >> knf >> c.pos
                  >> c.lvalid >> c.outdir)) {
                fprintf(stderr, "plan %s line %d: NAME QG KG VG KCACHE VCACHE QNORM KNORM POS "
                                "LCACHE OUTDIR [NROWS]\n", plan_path.c_str(), ln);
                return 1;
            }
            // P8: the optional 12th field makes the line a WAVE — NROWS pinned query rows in one
            // launch, at positions pos..pos+NROWS-1 over one shared cache. Absent means 1, which is
            // every plan file this op has ever been gated with.
            if (ss >> c.nrows && c.nrows < 1) {
                fprintf(stderr, "%s: NROWS must be >= 1 (got %d)\n", c.name.c_str(), c.nrows);
                return 1;
            }
            const std::vector<char> kraw = slurp(kcf);
            if (kraw.size() % ((size_t)N_KV * DH * 2)) {
                fprintf(stderr, "%s: cache is not a whole [2][cap][256] BF16 block\n", kcf.c_str());
                return 1;
            }
            c.cap = (int)(kraw.size() / ((size_t)N_KV * DH * 2));
            if (c.lvalid < 0 || c.lanes_last() > c.cap || c.pos + c.nrows - 1 >= c.cap) {
                fprintf(stderr, "%s: pos %d (+%d rows) / lvalid %d do not fit cap %d\n",
                        c.name.c_str(), c.pos, c.nrows - 1, c.lvalid, c.cap);
                return 1;
            }
            if (c.nrows > 1 && launch == "fused") {
                // Not a missing feature: a wave's row t READS the lanes rows 0..t-1 appended (§06.3),
                // and the fused plan appends from inside the block that scores, so there is no grid-wide
                // barrier to order it. Staging is what makes a wave legal, and saying so here is better
                // than measuring a race.
                fprintf(stderr, "%s: --launch fused cannot serve a wave of %d rows (the §05.7 append "
                                "would race the next row's read). Use --launch staged.\n",
                        c.name.c_str(), c.nrows);
                return 1;
            }
            if (c.nrows > 1 && neg == NEG_CARRY) {
                // `carry` is one query's state leaking into the NEXT query. Inside one wave launch there
                // is no previous query to leak from — the injection is only defined row at a time.
                fprintf(stderr, "--neg carry is not defined on a wave line (%d rows in one launch); "
                                "run it on single-row cases\n", c.nrows);
                return 1;
            }
            if (full && (c.lanes_last() + split - 1) / split > MAXSC) {
                fprintf(stderr,
                        "%s: --softmax full stages %d lanes per block, MAXSC is %d (raise --split)\n",
                        c.name.c_str(), (c.lanes_last() + split - 1) / split, MAXSC);
                return 1;
            }
            c.qg = upload(qgf, (size_t)4096 * c.nrows);
            c.kg = upload(kgf, (size_t)512 * c.nrows);
            c.vg = upload(vgf, (size_t)512 * c.nrows);
            c.qn = upload(qnf, DH);
            c.kn = upload(knf, DH);
            for (int r = 0; r < std::max(1, tile_rep); ++r) {
                hip_bfloat16 *kd = nullptr, *vd = nullptr;
                CK(hipMalloc(&kd, kraw.size()));
                CK(hipMalloc(&vd, kraw.size()));
                CK(hipMemcpy(kd, kraw.data(), kraw.size(), hipMemcpyHostToDevice));
                CK(hipMemcpy(vd, slurp(vcf).data(), kraw.size(), hipMemcpyHostToDevice));
                c.kc.push_back(kd);
                c.vc.push_back(vd);
            }
            CK(hipMalloc(&c.qrope, (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.krope, (size_t)N_KV * DH * 2 * c.nrows));
            CK(hipMalloc(&c.gate, (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.aout, (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.gated, (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.rcs, (size_t)ROT * 2 * 2));
            c.stale = g_stale;               // ONE buffer for the whole plan: `carry` has to leak
            // From case i into case i+1, not into itself. The tiled form needs the partials whenever
            // any row exceeds the staging, which is a per-case question the `--split` knob does not
            // own, so the allocation is the wider of the two forms' promises.
            const size_t part_elems =
                std::max(split > 1 ? (size_t)c.nrows * N_Q * split * (DH + 2) : 0,
                         tile_kernel ? (size_t)tile_part_floats(c.nrows, c.lanes_last()) : 0);
            if (part_elems) {
                // Sized for every row of the case, which is exactly the expression the kernels index
                // with: ((qrow*N_Q + head)*split + y) for the shipped form,
                // ((qrow*G + head_in_group)*windows + y) for the tiled one. Zero-filled, so a
                // partial that was never written reads as the neutral (0,0,0) instead of garbage.
                CK(hipMalloc(&c.part, part_elems * 4));
                CK(hipMemset(c.part, 0, part_elems * 4));
            }
            cases.push_back(c);
        }
    }
    if (cases.empty()) {
        fprintf(stderr, "plan %s has no cases\n", plan_path.c_str());
        return 1;
    }

    const int launches = (launch == "staged" ? 1 : 0) + 1 + (split > 1 ? 1 : 0);
    size_t plan_rows = 0;
    for (auto& c : cases) plan_rows += c.nrows;
    // The deepest core grid this plan can launch, echoed on RESULT so a control that claims to have
    // deepened the lane grid can be checked instead of believed. The tiled form's grid.y IS `a.split`,
    // i.e. the max taken in `launch_case`, so the two spellings cannot drift.
    const int core_x = tile_kernel ? N_KV : N_Q;
    int core_y_max = 1, core_z_max = 1;
    for (auto& c : cases) {
        core_y_max = std::max(core_y_max, tile_kernel ? std::max(split, tile_split_of(c.lanes_last()))
                                                      : split);
        core_z_max = std::max(core_z_max, tile_kernel ? tile_rtiles_of(c.nrows) : c.nrows);
    }
    char core_grid[32];
    std::snprintf(core_grid, sizeof core_grid, "%dx%dx%d", core_x, core_y_max, core_z_max);
    printf("plan %s | cases=%zu | rows=%zu | launch=%s (split %d) | softmax=%s | warps=%d tile=%d"
           " | neg=%s | score=%s | replicas=%d | core=%s grid=%s\n",
           plan_path.c_str(), cases.size(), plan_rows, launch.c_str(), split, softmax.c_str(), nw,
           tile, negname.c_str(), score_f32 ? "f32" : "bf16", std::max(1, tile_rep),
           tile_kernel ? "attn_tile_wave" : "attn_core", core_grid);
    if (neg != NEG_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());

    auto launch_case = [&](Case& c, int rep, bool stages) {
        CoreArgs a{};
        a.qg = c.qg; a.kg = c.kg; a.vg = c.vg;
        a.kcache = c.kc[(size_t)rep % c.kc.size()];
        a.vcache = c.vc[(size_t)rep % c.vc.size()];
        a.qn = c.qn; a.kn = c.kn;
        a.aout = c.aout; a.gout = c.gated;
        a.kc_as = a.kcache; a.vc_as = a.vcache;   // the append writes in place (§05.7)
        a.rope_out = c.rcs;
        a.part = c.part; a.stale = c.stale;
        a.cap = c.cap; a.pos = c.pos; a.lvalid = c.lvalid; a.split = split; a.nrows = c.nrows;
        a.neg = neg; a.full = full ? 1 : 0; a.score_f32 = score_f32 ? 1 : 0;
        a.fused = launch == "fused" ? 1 : 0; a.tile = tile;
        a.tile_kernel = tile_kernel ? 1 : 0;
        if (tile_kernel) {
            // The tiled form owns its lane geometry: its windows are MAXSC lanes wide by
            // construction, so `split` can never be SMALLER than `tile_split_of(lanes_last)` here —
            // leaving the knob in charge would put a second, disagreeing authority over how many
            // partials a row promises, and `tile_blocks_for_row` clamps by this field. The knob can
            // still DEEPEN the grid, and that is what `run.py`'s mixed-grid control needs: a row the
            // staging fits must take exactly one block whatever the grid's depth, which is the same
            // claim `blocks_for_row` makes for `attn_core`, and no control can test it on the tiled
            // form while `--split 2` is a silent no-op here.
            a.split = std::max(split, tile_split_of(c.lanes_last()));
        }
        if (stages) rope_table<<<1, ROT>>>(a);   // stage check, not part of the timed path
        if (launch == "staged") {
            CoreArgs p = a;
            p.qrope_out = c.qrope; p.krope_out = c.krope; p.gate_out = c.gate;
            // grid.y is the query row (P8): one launch stages the whole wave and performs every row's
            // §05.7 append at that row's own position, which is what lets the core launch below read a
            // cache that already holds the rows before it.
            attn_prep<<<dim3(N_Q + N_KV, c.nrows), THREADS>>>(p);
            a.qrope = c.qrope; a.krope = c.krope;      // core reads what prep wrote
        } else {
            a.qrope_out = c.qrope; a.krope_out = c.krope; a.gate_out = c.gate;
        }
        a.aout = c.aout; a.gout = c.gated;
        // grid.z is the query row, grid.y the lane block within it (§07.3's split axis, unchanged).
        // The tiled form's grid is (KV group, lane window, row tile), and the widest-row-first
        // inversion lives inside the kernel, not here.
        if (tile_kernel) {
            attn_tile_wave<<<dim3(N_KV, a.split, tile_rtiles_of(c.nrows)), ATTN_TILE_THREADS>>>(a);
        } else {
            attn_core<<<dim3(N_Q, split, c.nrows), THREADS>>>(a);
        }
        if (a.split > 1) attn_combine<<<dim3(N_Q, c.nrows), THREADS>>>(a);
        CK(hipGetLastError());
    };

    auto write_case = [&](Case& c) {
        const std::string& o = c.outdir;
        const size_t nr = c.nrows;
        dump_file(o + "/q_rope.bin", c.qrope, (size_t)N_Q * DH * nr);
        dump_file(o + "/k_rope.bin", c.krope, (size_t)N_KV * DH * nr);
        dump_file(o + "/gate_vec.bin", c.gate, (size_t)N_Q * DH * nr);
        dump_file(o + "/rope_cs.bin", c.rcs, (size_t)ROT * 2);
        dump_file(o + "/attn_pre_gate.bin", c.aout, (size_t)N_Q * DH * nr);
        dump_file(o + "/gated.bin", c.gated, (size_t)N_Q * DH * nr);
        dump_file(o + "/cache_keys.bin", c.kc[0], (size_t)N_KV * c.cap * DH);
        dump_file(o + "/cache_values.bin", c.vc[0], (size_t)N_KV * c.cap * DH);
    };

    size_t plan_bytes = 0;
    for (auto& c : cases) plan_bytes += c.bytes(split);

    if (mode == "once") {
        int rep = 0;
        for (auto& c : cases) launch_case(c, rep++, true);
        CK(hipDeviceSynchronize());
        for (auto& c : cases) write_case(c);
        printf("RESULT mode=once cases=%zu rows=%zu launches=%d neg=%s split=%d launch=%s"
               " softmax=%s core=%s core_grid=%s\n",
               cases.size(), plan_rows, launches, negname.c_str(), split, launch.c_str(),
               softmax.c_str(), tile_kernel ? "tile" : "attn_core", core_grid);
        return 0;
    }

    // Burst: submit the whole plan `iters` times, one sync, wall-clock/N. hipEventElapsedTime
    // reads exactly 0 below ~1 ms on this iGPU (trap #7), so wall-clock is the number.
    auto burst = [&](int n) {
        const auto t0 = std::chrono::steady_clock::now();
        size_t counter = 0;
        for (int i = 0; i < n; ++i)
            for (auto& c : cases) launch_case(c, (int)counter++, false);
        CK(hipDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    burst(1);                                     // warm: fresh pages run at ~35 GB/s
    const double ms = burst(std::max(1, iters)) / std::max(1, iters);
    const double gbs = (double)plan_bytes / (ms / 1e3) / 1e9;
    printf("RESULT mode=burst cases=%zu launches_per_case=%d ms_per_plan=%.6f us_per_case=%.3f"
           " gbs=%.2f probe_pct=%.2f neg=%s split=%d launch=%s softmax=%s bytes=%.1f KiB/plan"
           " core_grid=%s\n", cases.size(), launches, ms, ms * 1e3 / cases.size(), gbs,
           100.0 * gbs / PROBE_GBS, negname.c_str(), split, launch.c_str(), softmax.c_str(),
           plan_bytes / 1024.0, core_grid);
    if (write_out) {
        for (size_t i = 0; i < cases.size(); ++i) launch_case(cases[i], (int)i, true);
        CK(hipDeviceSynchronize());
        for (auto& c : cases) write_case(c);
    }
    return 0;
}
