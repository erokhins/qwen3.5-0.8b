/*
 * attn_decode_wide.hip.cpp — Route D3 rig: the decode attention line at long context, two cores, one
 * process, one lock hold.
 *
 * WHY A SEPARATE RIG. `prototype/attn_decode/` is the shipped op's gate and the witness §18-§20 were
 * measured with; D3 has to A/B a candidate against that witness without editing it. This rig therefore
 * INCLUDES the frozen header — `attn_core`, `attn_prep`, `attn_combine`, the geometry helpers and the
 * injections are the same source the engine compiles — and adds one arm. Every arm is the same
 * prep → core → combine sequence, so a difference between arms is the candidate and nothing else.
 *
 * THE ARMS.
 *   --core ship   the frozen `attn_core` (the reference for every comparison here).
 *   --core lean   `attn_core_lean`: the same kernel with two loop trip counts made literals, which
 *                 reclaims 32,768 B of LDS per block (38,944 → 6,176 B) and so lifts residency from
 *                 1 block/CU to 4. Claimed BIT-IDENTICAL; `gate.py` proves it on every case.
 *   --split N     overrides the launch's lane-block count. `blocks_for_row()` honours `a.split`, so a
 *                 deeper cut is a shallower serial walk with no kernel change at all — the population
 *                 lever. It RE-WINDOWS the streaming softmax, so it is a banded arm, not a bit-identical
 *                 one: §16's question, judged against the published floors and reported, never widened.
 *   --neg NAME    the frozen injections, kept because a gate that cannot fail is not a gate. `dimhalf`
 *                 matters doubly here: the runtime `nd` that the `lean` arm re-spells exists only for it.
 *
 * PLAN FORMAT is `prototype/attn_decode/`'s, verbatim, so `gate.py` can hand over the same raw BF16
 * operand files the shipped gate writes with `reference.write_case()`:
 *     NAME QG KG VG KCACHE VCACHE QNORM KNORM POS LVALID OUTDIR [NROWS]
 * and `lane_valid`, not `pos + 1`, owns the causal bound (§06.3 — a wave's last row is the widest).
 *
 * WHO MAY TIME THIS. `--mode once` is the correctness path a worker is allowed to run (AGENTS.md:
 * correctness-only device gates, under a minute, no timing). `--mode burst` prints wall-clock/N for the
 * main agent's blocking, lock-held pass; a millisecond figure from a worker run is contaminated and must
 * not be recorded anywhere. `--core` is echoed on every RESULT line so a timing script can assert the
 * behavioural canary in the same window as the number (the stale-binary trap).
 *
 * BUILD: ./build.sh   (needs -Wl,-rpath,/opt/rocm/lib; see prototype/README.md)
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "attn_decode_wide_kernels.h"

// Exits rather than returns: CK is used inside helpers and lambdas whose return type is not `int`, and
// a macro that silently returned a status code is how a failed allocation becomes a null deref.
#define CK(x)                                                                        \
    do {                                                                             \
        const hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                      \
            std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__,               \
                         hipGetErrorString(e_));                                     \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

std::vector<hip_bfloat16> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "missing %s\n", p.c_str()); std::exit(1); }
    const auto n = f.tellg();
    f.seekg(0);
    std::vector<hip_bfloat16> v((size_t)n / 2);
    f.read((char*)v.data(), n);
    return v;
}

hip_bfloat16* upload(const std::string& p, size_t want) {
    auto h = slurp(p);
    if (h.size() < want) {
        std::fprintf(stderr, "%s holds %zu elements, the plan needs %zu\n", p.c_str(), h.size(), want);
        std::exit(1);
    }
    hip_bfloat16* d = nullptr;
    CK(hipMalloc(&d, want * 2));
    CK(hipMemcpy(d, h.data(), want * 2, hipMemcpyHostToDevice));
    return d;
}

void dump_file(const std::string& p, const void* dptr, size_t n_bf16) {
    std::vector<hip_bfloat16> h(n_bf16);
    CK(hipMemcpy(h.data(), dptr, n_bf16 * 2, hipMemcpyDeviceToHost));
    std::ofstream f(p, std::ios::binary);
    f.write((const char*)h.data(), (std::streamsize)(n_bf16 * 2));
}

void dump_part(const std::string& p, const float* dptr, size_t n) {
    std::vector<float> h(n);
    CK(hipMemcpy(h.data(), dptr, n * 4, hipMemcpyDeviceToHost));
    std::ofstream f(p, std::ios::binary);
    f.write((const char*)h.data(), (std::streamsize)(n * 4));
}

hip_bfloat16* dev_copy(const void* src, size_t n_bf16) {
    hip_bfloat16* d = nullptr;
    CK(hipMalloc(&d, n_bf16 * 2));
    CK(hipMemcpy(d, src, n_bf16 * 2, hipMemcpyHostToDevice));
    return d;
}

struct Case {
    std::string name, outdir;
    int pos = 0, lvalid = 1, cap = 1, nrows = 1, split = 1;
    hip_bfloat16 *qg = nullptr, *kg = nullptr, *vg = nullptr, *qn = nullptr, *kn = nullptr;
    std::vector<hip_bfloat16> kraw, vraw;                 // host copies; the source of each replica
    std::vector<hip_bfloat16*> kc, vc;                    // one device replica per burst slot
    hip_bfloat16 *aout = nullptr, *gated = nullptr;
    hip_bfloat16 *qrope = nullptr, *krope = nullptr, *gatev = nullptr, *rcs = nullptr;
    float* part = nullptr;
    size_t part_n = 0;
    int lanes() const { return lvalid + nrows - 1; }
};

} // namespace

int main(int argc, char** argv) {
    std::string mode = "once", plan_path, core = "ship", negname = "none";
    int split_opt = 0, iters = 20, reps = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--plan") plan_path = next();
        else if (a == "--core") core = next();
        else if (a == "--split") split_opt = atoi(next().c_str());
        else if (a == "--neg") negname = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--reps") reps = atoi(next().c_str());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (core != "ship" && core != "lean") {
        std::fprintf(stderr, "--core takes ship|lean, got '%s'\n", core.c_str());
        return 1;
    }
    const int neg = negname == "none" ? NEG_NONE
                    : negname == "dimhalf" ? NEG_DIMHALF
                    : negname == "gatehalf" ? NEG_GATEHALF
                    : negname == "qswap" ? NEG_QSWAP
                    : negname == "gqaoff" ? NEG_GQA
                    : negname == "nomax" ? NEG_NOMAX
                    : negname == "ropefull" ? NEG_ROPEFULL
                    : negname == "ropenone" ? NEG_ROPENONE
                    : negname == "carry" ? NEG_CARRY
                    : negname == "dropcurrent" ? NEG_DROPCUR
                    : negname == "readpast" ? NEG_READPAST : -1;
    if (neg < 0) { std::fprintf(stderr, "unknown --neg %s\n", negname.c_str()); return 1; }

    std::vector<Case> cases;
    {
        std::ifstream pf(plan_path);
        if (!pf) { std::fprintf(stderr, "missing plan %s\n", plan_path.c_str()); return 1; }
        std::string line;
        while (std::getline(pf, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            Case c;
            std::string qgf, kgf, vgf, kcf, vcf, qnf, knf;
            ss >> c.name >> qgf >> kgf >> vgf >> kcf >> vcf >> qnf >> knf
               >> c.pos >> c.lvalid >> c.outdir;
            if (!(ss >> c.nrows)) c.nrows = 1;
            c.kraw = slurp(kcf);
            c.vraw = slurp(vcf);
            c.cap = (int)(c.kraw.size() / ((size_t)N_KV * DH));
            if (c.lanes() > c.cap) {
                std::fprintf(stderr, "%s: lane_valid %d exceeds the cache's %d lanes\n",
                             c.name.c_str(), c.lanes(), c.cap);
                return 1;
            }
            c.qg = upload(qgf, ROW_Q * c.nrows);
            c.kg = upload(kgf, ROW_KV * c.nrows);
            c.vg = upload(vgf, ROW_KV * c.nrows);
            c.qn = upload(qnf, DH);
            c.kn = upload(knf, DH);
            CK(hipMalloc(&c.aout, ROW_H * c.nrows * 2));
            CK(hipMalloc(&c.gated, ROW_H * c.nrows * 2));
            CK(hipMalloc(&c.qrope, ROW_H * c.nrows * 2));
            CK(hipMalloc(&c.krope, ROW_H * c.nrows * 2));
            CK(hipMalloc(&c.gatev, ROW_H * c.nrows * 2));
            CK(hipMalloc(&c.rcs, (size_t)ROT * 2 * 2));
            // Burst replicas of the cache: the staged route APPENDS the current lane (§05.7), so a
            // burst that reused one cache would have later passes reading a lane a previous pass wrote.
            // Values are identical (the append is idempotent), but replicas keep the read side the same
            // shape as the first pass, which is what the shipping path does per step.
            for (int r = 0; r < std::max(1, reps); ++r) {
                c.kc.push_back(dev_copy(c.kraw.data(), c.kraw.size()));
                c.vc.push_back(dev_copy(c.vraw.data(), c.vraw.size()));
            }
            const int natural = split_for_lanes(c.lanes());
            c.split = split_opt > 0 ? split_opt : natural;
            // The same refusal the shipped harness makes, and for the same reason: with `full = 1` a
            // block stages its slice's scores in `sc[MAXSC]`, so a slice wider than MAXSC is not a
            // deeper cut, it is a staging overflow that prints a plausible-looking wrong answer. The
            // first version of `gate.py` ran `--split 1` on a 4,104-lane row and got cos -0.013, which
            // was this overflow, not a route. Refuse; never measure an illegal geometry.
            const int span = (c.lanes() + c.split - 1) / c.split;
            if (span > MAXSC) {
                std::fprintf(stderr,
                             "%s: --split %d leaves %d lanes in one block, MAXSC stages %d; "
                             "the cut must be at least %d\n",
                             c.name.c_str(), c.split, span, MAXSC, split_for_lanes(c.lanes()));
                return 1;
            }
            if (c.split > 1 && natural == 1) {
                // Legal but not the shipping shape: a row the one-pass staging fits CAN be cut, and
                // `blocks_for_row` will stream it. `gate.py` uses exactly this to probe the boundary
                // from both sides, so it is recorded, not refused.
                std::printf("NOTE %s: split=%d on a %d-lane row the one-pass form would serve in 1 block\n",
                            c.name.c_str(), c.split, c.lanes());
            }
            if (c.split > 1) {
                c.part_n = (size_t)c.nrows * N_Q * c.split * (DH + 2);
                CK(hipMalloc(&c.part, c.part_n * 4));
                CK(hipMemset(c.part, 0, c.part_n * 4));
            }
            cases.push_back(c);
        }
    }
    if (cases.empty()) { std::fprintf(stderr, "plan %s has no cases\n", plan_path.c_str()); return 1; }

    auto launch = [&](Case& c, int rep) {
        CoreArgs a{};
        a.qg = c.qg; a.kg = c.kg; a.vg = c.vg;
        a.qn = c.qn; a.kn = c.kn;
        a.kcache = c.kc[(size_t)rep % c.kc.size()];
        a.vcache = c.vc[(size_t)rep % c.vc.size()];
        a.kc_as = a.kcache; a.vc_as = a.vcache;      // the §05.7 append writes in place
        a.rope_out = c.rcs;
        a.part = c.part; a.stale = nullptr;
        a.cap = c.cap; a.pos = c.pos; a.lvalid = c.lvalid; a.split = c.split; a.nrows = c.nrows;
        a.neg = neg;
        // `full = 1` is the ENGINE's decode setting (`c.full = 1` in engine/kernels_attn.cpp's
        // attn_decode_layer): one pass with the scores staged for a row at or below MAXSC, streaming
        // partials above it. Parity with the shipping route is the point of this rig, so the knob is
        // deliberately not exposed.
        a.full = 1;
        a.score_f32 = 0;                            // §10.3: the score rounds through BF16
        a.fused = 0;                                // staged route, as the engine runs it
        a.tile = TILE;
        a.tile_kernel = 0;
        a.fattn_cover = 0;
        a.aout = c.aout; a.gout = c.gated;

        // prep → core → combine, in that order: prep does the norm+rope+append the core's read side
        // depends on, so an arm without its own prep would read a cache missing the current lane. Its
        // cost sits inside every arm, which is what makes these numbers comparable with the ledger's
        // `attn` line (prep + core + combine), not with the core alone.
        CoreArgs p = a;
        p.qrope_out = c.qrope; p.krope_out = c.krope; p.gate_out = c.gatev;
        attn_prep<<<dim3(N_Q + N_KV, c.nrows), THREADS>>>(p);
        a.qrope = c.qrope; a.krope = c.krope;        // core reads what prep wrote
        if (core == "ship") attn_core<<<dim3(N_Q, c.split, c.nrows), THREADS>>>(a);
        else attn_core_lean<<<dim3(N_Q, c.split, c.nrows), THREADS>>>(a);
        if (c.split > 1) attn_combine<<<dim3(N_Q, c.nrows), THREADS>>>(a);
        CK(hipGetLastError());
    };

    auto echo = [&]() {
        std::printf(" core=%s split_opt=%d neg=%s cases=%zu grids=", core.c_str(), split_opt,
                    negname.c_str(), cases.size());
        for (auto& c : cases)
            std::printf(" %s:lanes=%d,split=%d,blocks=%d", c.name.c_str(), c.lanes(), c.split,
                        N_Q * c.split);
    };

    if (mode == "once") {
        for (size_t i = 0; i < cases.size(); ++i) launch(cases[i], (int)i);
        CK(hipDeviceSynchronize());
        for (auto& c : cases) {
            dump_file(c.outdir + "/attn_pre_gate.bin", c.aout, ROW_H * c.nrows);
            dump_file(c.outdir + "/gated.bin", c.gated, ROW_H * c.nrows);
            dump_file(c.outdir + "/q_rope.bin", c.qrope, ROW_H * c.nrows);
            dump_file(c.outdir + "/k_rope.bin", c.krope, ROW_KV * c.nrows);
            dump_file(c.outdir + "/cache_keys.bin", c.kc[0], c.kraw.size());
            dump_file(c.outdir + "/cache_values.bin", c.vc[0], c.vraw.size());
            if (c.part) dump_part(c.outdir + "/part.bin", c.part, c.part_n);
        }
        std::printf("RESULT mode=once");
        echo();
        std::printf("\n");
        return 0;
    }

    auto burst = [&](int n) {
        const auto t0 = std::chrono::steady_clock::now();
        size_t k = 0;
        for (int i = 0; i < n; ++i)
            for (auto& c : cases) launch(c, (int)k++);
        CK(hipDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    burst(1);                                       // warm: fresh pages run at ~35 GB/s
    const double ms = burst(std::max(1, iters)) / std::max(1, iters);
    double kv_bytes = 0;
    for (auto& c : cases)
        kv_bytes += (double)c.nrows * N_Q * c.lanes() * (2 * N_KV * DH * 2);  // one K row + one V row
    std::printf("RESULT mode=burst iters=%d ms_per_plan=%.6f us_per_case=%.3f kv_demanded_gbs=%.2f",
                iters, ms, ms * 1e3 / cases.size(), kv_bytes / (ms / 1e3) / 1e9);
    echo();
    std::printf("\n");
    return 0;
}
