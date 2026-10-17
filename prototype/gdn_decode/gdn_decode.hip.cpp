/*
 * prototype/gdn_decode/gdn_decode.hip.cpp — the state-carrying decode step of S04.
 *
 * Four entry points, each gated separately by check.py (the rule for this directory:
 * every function gets its own correctness check before the next one is written):
 *
 *   gates   S04.3: beta = sigmoid(b_raw), g = -exp(A_log)·softplus(a_raw + dt_bias)
 *   conv    S04.2: causal depthwise conv, kernel 4, on a [tokens][CH] tile + a [K-1][CH] window
 *   rec     S04.4: the gated-delta recurrence for one token, one GDN layer (16 heads)
 *   all     `rec` over all 18 GDN layers at once — the bandwidth-relevant case
 *
 * `gates` and `conv` take `--tokens N` (default 1) and `--neg NAME` (default none). At N = 1 both
 * launch exactly the grids they launched before P6 and read/write the same single-token files, so
 * the four original checks are the T = 1 cases unchanged; N > 1 is the prefill-wave shape (P6,
 * `progress/2026-10-12/plan-prefill-2026-10-12.md` §4 row P6) and is gated by `check.py --tokens <fixture>`.
 * The window is the part that is NOT a trivial batch: see the `GdnConvNeg` block in
 * `gdn_decode_kernels.h` for what the injections are and which of them no downstream output can see.
 *
 * The recurrence, per value head η with state S ∈ F32^{128×128} (row i over d_k, column j
 * over d_v), is `S ← γS; r = Sᵀk̃; Δv = β(v − r); S += k̃⊗Δv; o = q̃ᵀS`. One block per head
 * with 128 threads: thread j owns column j end to end, so both passes are perfectly
 * coalesced (consecutive threads touch consecutive floats of every row) and the reduction
 * for r needs no cross-thread communication at all:
 *
 *   pass A: r'_j = Σ_i S_old[i,j]·k̃[i]                                    (one read)
 *   Δv_j = β(v_j − γ r'_j)
 *   pass B: s = γ S_old[i,j]; s' = s + k̃[i]·Δv_j; S[i,j] = s';  o_j += q̃[i]·s'   (read+write)
 *
 * Pass B accumulates o from the *stored* value rather than using the algebraic split
 * o = γ(q̃ᵀS) + (q̃·k̃)Δv. That split is exactly equivalent on paper and measurably worse on
 * this data — the two terms cancel, and the first version came out at meanrel 1.5e-3
 * against the reference instead of 1e-6. Rounding element by element in the reference's
 * order is what reproduces it.
 *
 * Pass B re-reads S instead of keeping it in registers: a head is 64 KiB, which is 128
 * floats per thread at 512 threads and impossible at 128 threads. Cost is 3 state passes
 * instead of 2 — though see results.md: at 18 layers the whole state is small enough to
 * stay resident in the iGPU's cache, so the extra pass is cheaper than it looks.
 *
 * Build: prototype/gdn_decode/build.sh    Gate: .venv/bin/python prototype/gdn_decode/check.py
 *
 * TIMING in `rec`/`all` uses a burst of launches with ONE sync and steady_clock wall time — see the
 * comment at the burst for why the old per-iteration hipEventElapsedTime loop is gone, and what
 * `--restore-each` is for.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "gdn_decode_kernels.h"

#define CK(x)                                                              \
    do {                                                                   \
        hipError_t e_ = (x);                                              \
        if (e_ != hipSuccess) {                                           \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_), \
                    __FILE__, __LINE__);                                  \
            exit(1);                                                      \
        }                                                                 \
    } while (0)


/* ---- host ------------------------------------------------------------ */
static std::vector<float> load_f32(const std::string& p, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    std::vector<float> b(n);
    f.read((char*)b.data(), n * 4);
    if ((size_t)f.gcount() != n * 4) {
        fprintf(stderr, "%s: short read (%lld of %zu floats)\n", p.c_str(),
                (long long)f.gcount(), n);
        exit(1);
    }
    return b;
}
static void dump_f32(const std::string& p, const void* dptr, size_t n) {
    std::vector<float> h(n);
    CK(hipMemcpy(h.data(), (void*)dptr, n * 4, hipMemcpyDeviceToHost));
    FILE* f = fopen(p.c_str(), "wb");
    fwrite(h.data(), 4, n, f);
    fclose(f);
}

// The conv tile and the two gate rows are read by the kernels as BF16, because that is the dtype
// their producer writes; the fixtures this gate reads are the HF dumps, stored as F32. So the
// fixture must be narrowed here — and that is only legitimate if it is exactly representable, which
// it is, because HF's `mixed_qkv_pre_conv` / `in_proj_a` / `in_proj_b` outputs ARE bf16 tensors.
// Assert it rather than assume it: a dump that ever became a true F32 would otherwise be silently
// rounded here, and the gate would then be judging a kernel fed something the reference never had.
static std::vector<hip_bfloat16> narrow_exact(const std::vector<float>& v, const char* what) {
    std::vector<hip_bfloat16> out(v.size());
    size_t bad = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = hip_bfloat16(v[i]);
        if (float(out[i]) != v[i]) ++bad;
    }
    if (bad) {
        fprintf(stderr, "%s: %zu of %zu values are not BF16-exact, so this fixture cannot be fed "
                        "to a BF16-read kernel without rounding it\n", what, bad, v.size());
        exit(1);
    }
    return out;
}

int main(int argc, char** argv) {
    std::string mode, dir, out, negname = "none";
    int iters = 20, nlayers = 18, trials = 3, restore_each = 0, tokens = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--dir") dir = next();
        else if (a == "--out") out = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--trials") trials = atoi(next().c_str());
        else if (a == "--restore-each") restore_each = 1;
        else if (a == "--layers") nlayers = atoi(next().c_str());
        else if (a == "--tokens") tokens = atoi(next().c_str());
        else if (a == "--neg") negname = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (tokens < 1) { fprintf(stderr, "--tokens must be >= 1 (got %d)\n", tokens); return 1; }
    // One name per injected bug, and the kernel's own enum value beside it. An unknown name is a
    // refusal, not a silent 0: a negative that quietly became `none` would read as a passing gate.
    const int gneg = negname == "none" ? GC_NONE : negname == "tap_stride" ? GC_TAP_STRIDE
                     : negname == "win_lag" ? GC_WIN_LAG : negname == "win_from_u" ? GC_WIN_FROM_U
                     : negname == "zero_pad" ? GC_ZERO_PAD : -1;
    const int gtneg = negname == "none" ? GG_NONE : negname == "row_zero" ? GG_ROW_ZERO : -1;
    if ((mode == "conv" && gneg < 0) || (mode == "gates" && gtneg < 0)) {
        fprintf(stderr, "--neg for conv is none|tap_stride|win_lag|win_from_u|zero_pad, for"
                        " gates it is none|row_zero (got '%s' for mode %s)\n",
                negname.c_str(), mode.c_str());
        return 1;
    }
    auto path = [&](const char* n) { return dir + "/" + n; };

    if (mode == "gates") {
        const size_t n = (size_t)tokens * NH;
        auto a_raw = narrow_exact(load_f32(path("a_raw.f32"), n), "a_raw.f32");
        auto b_raw = narrow_exact(load_f32(path("b_raw.f32"), n), "b_raw.f32");
        auto A_log = load_f32(path("A_log.f32"), NH), dt = load_f32(path("dt_bias.f32"), NH);
        hip_bfloat16 *da, *db;
        float *dA, *dd, *dg, *dbt;
        CK(hipMalloc(&da, n * 2)); CK(hipMalloc(&db, n * 2));
        CK(hipMalloc(&dA, NH * 4)); CK(hipMalloc(&dd, NH * 4));
        CK(hipMalloc(&dg, n * 4)); CK(hipMalloc(&dbt, n * 4));
        CK(hipMemcpy(da, a_raw.data(), n * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(db, b_raw.data(), n * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dA, A_log.data(), NH * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dd, dt.data(), NH * 4, hipMemcpyHostToDevice));
        gates_kernel<<<tokens, NH>>>(da, db, dA, dd, dg, dbt, tokens, gtneg);
        CK(hipGetLastError()); CK(hipDeviceSynchronize());
        dump_f32(out + "/g.f32", dg, n);
        dump_f32(out + "/beta.f32", dbt, n);
        printf("RESULT mode=gates heads=%d tokens=%d neg=%s\n", NH, tokens, negname.c_str());
        return 0;
    }

    if (mode == "conv") {
        const size_t nraw = (size_t)tokens * CH;
        auto raw = narrow_exact(load_f32(path("raw.f32"), nraw), "raw.f32");
        auto win = load_f32(path("win.f32"), (K - 1) * CH);
        auto C = load_f32(path("convw.f32"), CH * K);
        hip_bfloat16 *draw;
        float *dwin, *dC, *du, *dnew;
        CK(hipMalloc(&draw, nraw * 2)); CK(hipMalloc(&dwin, (K - 1) * CH * 4));
        CK(hipMalloc(&dC, CH * K * 4)); CK(hipMalloc(&du, nraw * 4));
        CK(hipMalloc(&dnew, (K - 1) * CH * 4));
        CK(hipMemcpy(draw, raw.data(), nraw * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dwin, win.data(), (K - 1) * CH * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dC, C.data(), CH * K * 4, hipMemcpyHostToDevice));
        // grid.x is the channel strip (the single-token launch's blocks, CONV_VEC channels per lane),
        // grid.y the token. tokens = 1 is therefore the same grid, not a grid padded to one.
        const dim3 grid(CH / (256 * CONV_VEC), tokens);
        conv_kernel<<<grid, 256>>>(draw, dwin, dC, du, dnew, tokens, gneg);
        CK(hipGetLastError()); CK(hipDeviceSynchronize());
        dump_f32(out + "/u.f32", du, nraw);
        dump_f32(out + "/new_win.f32", dnew, (K - 1) * CH);
        printf("RESULT mode=conv channels=%d tokens=%d neg=%s\n", CH, tokens, negname.c_str());
        return 0;
    }

    if (mode == "rec" || mode == "all") {
        int L = (mode == "all") ? nlayers : 1;
        size_t nq = (size_t)L * NH * D, ns = (size_t)L * NH * D * D;
        std::vector<float> q, k, v, g, beta, S;
        if (mode == "all") {
            q = load_f32(path("q_all.f32"), nq); k = load_f32(path("k_all.f32"), nq);
            v = load_f32(path("v_all.f32"), nq);
            g = load_f32(path("g_all.f32"), (size_t)L * NH);
            beta = load_f32(path("beta_all.f32"), (size_t)L * NH);
            S = load_f32(path("S_all.f32"), ns);
        } else {
            q = load_f32(path("q.f32"), nq); k = load_f32(path("k.f32"), nq);
            v = load_f32(path("v.f32"), nq);
            g = load_f32(path("g.f32"), NH); beta = load_f32(path("beta.f32"), NH);
            S = load_f32(path("S.f32"), ns);
        }
        float *dq, *dk, *dv, *dg, *dbt, *dS, *do_;
        CK(hipMalloc(&dq, nq * 4)); CK(hipMalloc(&dk, nq * 4)); CK(hipMalloc(&dv, nq * 4));
        CK(hipMalloc(&dg, (size_t)L * NH * 4)); CK(hipMalloc(&dbt, (size_t)L * NH * 4));
        CK(hipMalloc(&dS, ns * 4)); CK(hipMalloc(&do_, nq * 4));
        CK(hipMemcpy(dq, q.data(), nq * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dk, k.data(), nq * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dv, v.data(), nq * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dg, g.data(), (size_t)L * NH * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dbt, beta.data(), (size_t)L * NH * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dS, S.data(), ns * 4, hipMemcpyHostToDevice));

        // The kernel mutates S in place, so keep a pristine copy: the burst timing restores from it,
        // outside the timed region (see the TIMING block for why).
        float* dS0;
        CK(hipMalloc(&dS0, ns * 4));
        CK(hipMemcpy(dS0, S.data(), ns * 4, hipMemcpyHostToDevice));
        auto launch = [&]() {
            rec_kernel<<<L * NH, D>>>(dq, dk, dv, dg, dbt, dS, do_, NH);
        };
        launch();
        CK(hipGetLastError()); CK(hipDeviceSynchronize());
        dump_f32(out + "/o.f32", do_, nq);
        dump_f32(out + "/S_new.f32", dS, ns);

        // TIMING — a burst of launches, ONE sync, steady_clock wall time divided by the launch
        // count. This used to be a per-iteration hipEventElapsedTime loop with
        // `best = std::min(best, (double)ms)` and no `ms > 0` guard, which is the exact timer
        // d81ff13 and ledger §6 disowned: on this iGPU the event timer returns exactly 0.000 ms for
        // sub-millisecond work, so a zero became the best row and the GB/s next to it was inf or
        // 918. Same idiom as prototype/glue and prototype/attn_decode now: one warm-up burst (first
        // touch of fresh pages runs ~7x slow, devices/), then `--trials` bursts of `--iters`
        // launches, best AND worst printed so the number carries its own spread instead of posing
        // as a point measurement.
        //
        // Why the state restore sits OUTSIDE the timed burst: copying the state back is
        // 2·ns·4 bytes of extra traffic (+66 % at 18 layers), so timing it would measure the copy,
        // not the recurrence. A burst therefore runs `iters` SUCCESSIVE recurrences over an evolving
        // S rather than replaying one step. That is what a token stream actually does — the engine
        // never restores either — and the kernel has no value-dependent control flow, so the bytes
        // moved are the same whatever S holds; with the same q/k/v every launch, S converges to a
        // bounded fixed point instead of decaying to zero. `--restore-each` puts the restore back
        // inside the burst if anyone wants to check that claim on the device: the two rows should
        // differ by about the copy's own time, and if they differ wildly the burst row is not
        // measuring what it says it is.
        const int l2_kb = [] {
            int kb = 0;
            (void)hipDeviceGetAttribute(&kb, hipDeviceAttributeL2CacheSize, 0);
            return kb;
        }();
        auto burst = [&](int n) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) {
                if (restore_each) CK(hipMemcpy(dS, dS0, ns * 4, hipMemcpyDeviceToDevice));
                launch();
            }
            CK(hipDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        // The correctness launch above already moved S, so restore before warming too.
        CK(hipMemcpy(dS, dS0, ns * 4, hipMemcpyDeviceToDevice));
        burst(1);                                   // warm: pages and clocks
        double best = 1e30, worst = 0.0;
        for (int t = 0; t < std::max(1, trials); ++t) {
            CK(hipMemcpy(dS, dS0, ns * 4, hipMemcpyDeviceToDevice));   // each trial starts pinned
            const double ms = burst(iters) / std::max(1, iters);
            if (!(ms > 0.0)) {
                fprintf(stderr, "burst %d measured 0.000 ms: timer not resolving, not a fast kernel\n", t);
                return 1;
            }
            best = std::min(best, ms);
            worst = std::max(worst, ms);
        }
        double bytes = 3.0 * ns * 4 + 4.0 * nq * 4;   // two S reads + one write, plus q/k/v/o
        const double gbs = bytes / (best / 1e3) / 1e9;
        printf("RESULT mode=%s layers=%d iters=%d trials=%d restore=%s ms=%.4f worst=%.4f "
               "spread_pct=%.1f gbs=%.1f state_MiB=%.1f pct_of_237=%.1f ws_KiB=%.0f l2_KiB=%d\n",
               mode.c_str(), L, iters, std::max(1, trials), restore_each ? "each" : "burst", best,
               worst, 100.0 * (worst - best) / best, gbs, ns * 4 / 1048576.0,
               bytes / (best / 1e3) / 237e9 * 100.0, bytes / 1024.0, l2_kb);
        // Printed, not asserted: whether a >237 GB/s row is skipped work (the repo's standing rule)
        // or a state that stayed resident decides how the ledger may use it. The attribute cannot
        // answer it on this box — `hipDeviceGetAttribute(L2CacheSize)` says 2 MiB while tiling shows
        // ~54 MiB resident (performance-status.md §4) — so this is a hint, and the tile sweep is the
        // measurement. Deciding the band is the ledger's call, not the kernel's.
        printf("NOTE working set %.0f KiB; L2 attribute says %d KiB (this box under-reports it — "
               "tiling, not the attribute, is the authority on residency)\n", bytes / 1024.0, l2_kb);
        return 0;
    }
    fprintf(stderr, "mode must be gates|conv|rec|all\n");
    return 1;
}
