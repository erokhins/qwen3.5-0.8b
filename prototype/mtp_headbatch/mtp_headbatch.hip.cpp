/*
 * prototype/mtp_headbatch/mtp_headbatch.hip.cpp — price the one term P7 plan A still carries as a
 * plan: does a SECOND candidate row ride the tied table read for free, or does it cost a second
 * cold pass? Prototype only; nothing here is engine code.
 *
 * Op: out[q][v] = sum_k x[q][k] * W[v][k] for q in [0,M), v in [0,V), W the tied BF16 table
 * [248320, 1024] = 508,559,360 B, x M F32 hidden rows. M=1 is also run through the SHIPPED kernel
 * (prototype/lm_head_gemv/lm_head_gemv_kernels.h, included by hb_kernels.h) inside the same process,
 * so the baseline is the code the engine runs today rather than a paraphrase of it.
 *
 * MODES
 *   --mode plan    host-only. Prints the launch plan and the byte expectation. No HIP call at all,
 *                  so it is safe outside the GPU lock (prototype/mtp_delta's --dump-plan rule).
 *   --mode check   correctness only. Witness launch, read back, print bytes-touched verdicts, then
 *                  re-poison and launch the timed form so the two can be compared bit for bit.
 *                  Prints NO ms field, in every path.
 *   --mode burst   the timed arm. Runs the witness FIRST and refuses to print ms if it is red
 *                  (AGENTS, burst_m16.sh: refuse to print a millisecond figure when admission is red).
 *
 * INJECTIONS (--inject, correctness path only; see hb_kernels.h):
 *   none | row_dup | row_shift | dropk | short_grid | extra_pass | extra_chunk
 * Each is required by prototype/mtp_headbatch/gate.py to move EXACTLY the outputs its geometry says
 * it can and to leave the rest bit-identical.
 *
 * Build: prototype/mtp_headbatch/build.sh   (-O3, -Wl,-rpath,/opt/rocm/lib, per prototype/README)
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

#include "hb_kernels.h"

#define CK(x)                                                                     \
    do {                                                                          \
        hipError_t e_ = (x);                                                      \
        if (e_ != hipSuccess) {                                                   \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),     \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

static std::vector<char> slurp(const std::string& p, size_t expect) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    std::vector<char> b(expect);
    f.read(b.data(), expect);
    if ((size_t)f.gcount() != expect) {
        fprintf(stderr, "%s: wanted %zu bytes, got %lld\n", p.c_str(), expect,
                (long long)f.gcount());
        exit(1);
    }
    return b;
}

static void usage() {
    fprintf(stderr,
            "usage: mtp_headbatch --w W.bin --x xM.bin [--out out.bin] [--out2 out2.bin]\n"
            "                   --M 1|2|4 [--arm hb|shipped] [--rpb 8]\n"
            "                   [--mode plan|check|burst] [--iters N]\n"
            "                   [--inject none|row_dup|row_shift|dropk|short_grid|extra_pass"
            "|extra_chunk] [--inject-q Q]\n"
            "  --x holds M*1024 F32 values; --arm shipped requires M=1.\n");
    exit(2);
}

static void dump(const std::string& path, const std::vector<unsigned int>& raw, size_t pad,
                 size_t elems) {
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    fwrite(raw.data() + pad, 4, elems, f);
    fclose(f);
}

int main(int argc, char** argv) {
    std::string wpath, xpath, opath, opath2, mode = "check", arm = "hb", inject = "none";
    int M = 2, rpb = ROWS_PER_BLOCK, iters = 20, inject_q = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) usage();
            return std::string(argv[++i]);
        };
        if (a == "--w") wpath = next();
        else if (a == "--x") xpath = next();
        else if (a == "--out") opath = next();
        else if (a == "--out2") opath2 = next();
        else if (a == "--mode") mode = next();
        else if (a == "--arm") arm = next();
        else if (a == "--M") M = atoi(next().c_str());
        else if (a == "--rpb") rpb = atoi(next().c_str());
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--inject") inject = next();
        else if (a == "--inject-q") inject_q = atoi(next().c_str());
        else usage();
    }
    if (wpath.empty() || xpath.empty()) usage();
    if (M != 1 && M != 2 && M != 4) {
        fprintf(stderr, "M must be 1, 2 or 4 (got %d)\n", M);
        return 2;
    }
    if (rpb < 1 || rpb > HB_MAX_RPB || (rpb & (rpb - 1))) {
        fprintf(stderr, "--rpb must be a power of two in [1,%d]: the block reduction publishes one\n"
                        "partial per warp and folds lanes < rpb, so any other value would silently\n"
                        "drop warp partials instead of failing\n", HB_MAX_RPB);
        return 2;
    }
    if (arm == "shipped" && M != 1) {
        fprintf(stderr, "--arm shipped is the M=1 shipping kernel only\n");
        return 2;
    }
    if (inject == "dropk" && (inject_q < 0 || inject_q >= M)) {
        fprintf(stderr, "dropk needs --inject-q in [0,%d)\n", M);
        return 2;
    }
    if (mode == "burst" && inject != "none") {
        printf("NO-TIMING refusing to time an injected case (inject=%s)\n", inject.c_str());
        return 2;
    }

    std::ifstream wf(wpath, std::ios::binary | std::ios::ate);
    size_t wbytes = (size_t)wf.tellg();
    wf.close();
    int V = (int)(wbytes / ((size_t)HID * 2));
    HbPlan p = hb_plan(M, V, rpb);
    if (p.weight_bytes != wbytes) {
        fprintf(stderr, "table size %zu does not match the plan's %llu\n", wbytes, p.weight_bytes);
        return 1;
    }

    if (mode == "plan") {
        // Host-only: derive the byte expectation from the plan and print it. Nothing allocated.
        printf("PLAN arm=%s M=%d V=%d rpb=%d grid=%d threads=%d chunks=%d table_bytes=%llu "
               "out_elems=%llu intended_chunks=%llu intended_fetch_per_weight=1.000000\n",
               arm.c_str(), M, V, rpb, p.grid, p.threads, p.chunks, p.weight_bytes, p.out_elems,
               p.intended_chunks);
        return 0;
    }

    auto xbuf = slurp(xpath, (size_t)M * HID * sizeof(float));
    printf("PLAN arm=%s M=%d V=%d rpb=%d grid=%d threads=%d chunks=%d table_bytes=%llu "
           "out_elems=%llu intended_chunks=%llu inject=%s\n",
           arm.c_str(), M, V, rpb, p.grid, p.threads, p.chunks, p.weight_bytes, p.out_elems,
           p.intended_chunks, inject.c_str());

    // The engine's load path: pinned host staging, one hipMemcpy (devices/strix-halo-unified-memory).
    char* hw = nullptr;
    CK(hipHostMalloc(&hw, wbytes));
    {
        std::ifstream f(wpath, std::ios::binary);
        f.read(hw, wbytes);
    }
    // A pad after the table: the extra_chunk injection reads one round past a row's end, and on the
    // LAST vocab row that is past the table. In-bounds zeros, so the injection is a wrong fetch
    // rather than a fault.
    const size_t W_PAD = 4096;
    hip_bfloat16* dw = nullptr;
    CK(hipMalloc(&dw, wbytes + W_PAD));
    CK(hipMemset(dw, 0, wbytes + W_PAD));
    CK(hipMemcpy(dw, hw, wbytes, hipMemcpyHostToDevice));

    float* dh = nullptr;
    const size_t H_PAD = 1024;                     // floats, so dropk/row_shift cannot leave the row
    CK(hipMalloc(&dh, ((size_t)M * HID + H_PAD) * sizeof(float)));
    CK(hipMemset(dh, 0, ((size_t)M * HID + H_PAD) * sizeof(float)));
    CK(hipMemcpy(dh, xbuf.data(), (size_t)M * HID * sizeof(float), hipMemcpyHostToDevice));

    const size_t PAD = 256;                        // poison words on each side of the logits
    const size_t out_words = PAD + p.out_elems + PAD;
    unsigned int* dout_raw = nullptr;
    CK(hipMalloc(&dout_raw, out_words * sizeof(unsigned int)));
    std::vector<unsigned int> poison(out_words, HB_POISON_WORD);
    float* dout = reinterpret_cast<float*>(dout_raw + PAD);
    auto repoison = [&]() {
        CK(hipMemcpy(dout_raw, poison.data(), out_words * sizeof(unsigned int),
                     hipMemcpyHostToDevice));
    };
    repoison();

    // One witness slot per block PER PASS. No atomics: the published partial is an integer, so a
    // fixed host-side sum over slots is exact and order-independent, and a second pass writes its
    // own slots instead of racing the first into one address.
    int passes = (inject == "extra_pass") ? 2 : 1;
    int grid = (inject == "short_grid") ? p.grid - 1 : p.grid;
    if (grid < 1) {
        fprintf(stderr, "short_grid needs grid > 1\n");
        return 2;
    }
    const size_t slots = (size_t)passes * grid;
    unsigned long long *cs = nullptr, *cnt = nullptr, *rawsl = nullptr;
    if (arm == "hb") {
        CK(hipMalloc(&cs, slots * 8));
        CK(hipMalloc(&cnt, slots * 8));
        CK(hipMalloc(&rawsl, slots * 8));
        CK(hipMemset(cs, 0, slots * 8));
        CK(hipMemset(cnt, 0, slots * 8));
        CK(hipMemset(rawsl, 0, slots * 8));
    }

    int dropq = (inject == "dropk") ? inject_q : -1;
    int rowdup = (inject == "row_dup") ? 0 : -1;
    int rowshift = (inject == "row_shift") ? 1 : 0;
    int chunks = p.chunks + (inject == "extra_chunk" ? 32 : 0);

    auto dispatch_timed = [&]() {
        if (arm == "shipped")
            lm_head_gemv_full<<<p.grid, p.threads>>>(dw, dh, dout, V, rpb);
        else if (M == 1)
            hb_head<1, false><<<p.grid, p.threads>>>(dw, dh, dout, nullptr, nullptr, nullptr, p.V,
                                                     p.rpb, p.chunks, -1, -1, 0);
        else if (M == 2)
            hb_head<2, false><<<p.grid, p.threads>>>(dw, dh, dout, nullptr, nullptr, nullptr, p.V,
                                                     p.rpb, p.chunks, -1, -1, 0);
        else
            hb_head<4, false><<<p.grid, p.threads>>>(dw, dh, dout, nullptr, nullptr, nullptr, p.V,
                                                     p.rpb, p.chunks, -1, -1, 0);
    };
    auto dispatch_witness = [&](int pass) {
        const int slot = pass * grid;
        if (M == 1)
            hb_head<1, true><<<grid, p.threads>>>(dw, dh, dout, cs + slot, cnt + slot, rawsl + slot,
                                                  p.V, p.rpb, chunks, dropq, rowdup, rowshift);
        else if (M == 2)
            hb_head<2, true><<<grid, p.threads>>>(dw, dh, dout, cs + slot, cnt + slot, rawsl + slot,
                                                  p.V, p.rpb, chunks, dropq, rowdup, rowshift);
        else
            hb_head<4, true><<<grid, p.threads>>>(dw, dh, dout, cs + slot, cnt + slot, rawsl + slot,
                                                  p.V, p.rpb, chunks, dropq, rowdup, rowshift);
    };

    // ---- host-side expectation: walk the INTENDED range from the plan, never the kernel's output
    unsigned long long exp_w = 0, exp_r = 0, exp_n = 0;
    if (arm == "hb") hb_expect(hw, p, &exp_w, &exp_r, &exp_n);
    if (arm == "hb" && exp_n != p.intended_chunks) {
        fprintf(stderr, "internal: expectation walk %llu != plan %llu\n", exp_n, p.intended_chunks);
        return 1;
    }
    exp_w *= (unsigned long long)passes;
    exp_r *= (unsigned long long)passes;
    exp_n *= (unsigned long long)passes;

    // ---- correctness pass: witness first; burst refuses to time if it is red --------------------
    // The shipped arm has no witness slots (it is the shipping kernel, unmodified), so its judged
    // output is the shipping kernel's own logits. Launching a witness with null slots faults the
    // device, which is how this branch was found: a path only one arm takes is invisible to every
    // other case the gate runs.
    const bool has_witness = (arm == "hb");
    if (has_witness) dispatch_witness(0);          // first touch: pages fault in here
    else dispatch_timed();
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    for (int k = 1; k < passes; ++k) {             // extra_pass = a second cold pass over the table
        dispatch_witness(k);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
    }

    std::vector<unsigned long long> hcs(slots), hcnt(slots), hraw(slots);
    unsigned long long got_w = 0, got_r = 0, got_n = 0;
    if (arm == "hb") {
        CK(hipMemcpy(hcs.data(), cs, slots * 8, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hcnt.data(), cnt, slots * 8, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hraw.data(), rawsl, slots * 8, hipMemcpyDeviceToHost));
        for (size_t b = 0; b < slots; ++b) {       // fixed order, integer: exact, no round-off
            got_w += hcs[b];
            got_r += hraw[b];
            got_n += hcnt[b];
        }
    }
    const double fetch_per_weight = arm == "hb" ? (double)got_n / (double)p.intended_chunks : -1.0;

    std::vector<unsigned int> hout(out_words);
    CK(hipMemcpy(hout.data(), dout_raw, out_words * sizeof(unsigned int), hipMemcpyDeviceToHost));
    size_t poison_left = 0, guard_written = 0;
    for (size_t i = 0; i < out_words; ++i) {
        const bool inside = i >= PAD && i < PAD + p.out_elems;
        if (inside) {
            // A logit slot still holding the poison word was never written.
            if (hout[i] == HB_POISON_WORD) ++poison_left;
        } else {
            // A guard word that no longer holds the poison word was written: out-of-range store.
            if (hout[i] != HB_POISON_WORD) ++guard_written;
        }
    }
    const bool cs_ok = (arm != "hb") || (got_w == exp_w);
    const bool raw_ok = (arm != "hb") || (got_r == exp_r);
    const bool fetch_ok = (arm != "hb") || (got_n == p.intended_chunks);
    const bool witness_ok = cs_ok && raw_ok && fetch_ok && poison_left == 0 && guard_written == 0;
    dump(opath, hout, PAD, p.out_elems);

    if (mode == "check") {
        printf("WITNESS slots=%zu passes=%d fetches=%llu fetch_per_weight=%.6f cs_ok=%d raw_ok=%d "
               "fetch_ok=%d poison_left=%zu guard_ok=%d\n",
               slots, passes, got_n, fetch_per_weight, (int)cs_ok, (int)raw_ok, (int)fetch_ok,
               poison_left, guard_written == 0);
        // Re-poison and run the TIMED form, so gate.py can require the instantiation that gets
        // clocked to produce bit-identical logits to the one being witnessed.
        repoison();
        dispatch_timed();
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        std::vector<unsigned int> hout2(out_words);
        CK(hipMemcpy(hout2.data(), dout_raw, out_words * sizeof(unsigned int),
                     hipMemcpyDeviceToHost));
        size_t poison2 = 0;
        for (size_t i = PAD; i < PAD + p.out_elems; ++i)
            if (hout2[i] == HB_POISON_WORD) ++poison2;
        dump(opath2, hout2, PAD, p.out_elems);
        printf("TIMEDFORM poison_left=%zu differs_from_witness=%s\n", poison2,
               has_witness ? ((hout == hout2) ? "no" : "YES") : "n/a");
        printf("RESULT arm=%s M=%d rows_served=%d out_elems=%llu grid=%d rpb=%d table_bytes=%llu "
               "fetch_per_weight=%.6f poison_left=%zu inject=%s\n",
               arm.c_str(), M, M, p.out_elems, grid, rpb, p.weight_bytes, fetch_per_weight,
               poison_left, inject.c_str());
        printf("VERDICT %s\n", witness_ok ? "OK" : "BROKEN");
        return witness_ok ? 0 : 1;
    }

    // ---- timed path ---------------------------------------------------------------------------
    if (!witness_ok) {
        printf("NO-TIMING witness is red (cs_ok=%d raw_ok=%d fetch_ok=%d poison_left=%zu"
               " guard_ok=%d); refusing to print ms\n",
               (int)cs_ok, (int)raw_ok, (int)fetch_ok, poison_left, guard_written == 0);
        return 1;
    }
    dispatch_timed();                              // first touch of the timed form
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    // Burst wall-clock, not hipEvents: hipEventElapsedTime returns exactly 0 for a ~2 ms kernel on
    // this box (prototype/lm_head_gemv/results.md), so burst + one sync + divide is the timer to
    // trust. best/worst over N bursts is the spread the ledger asks for.
    auto burst = [&](int n) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) dispatch_timed();
        CK(hipDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / n;
    };
    const int nb = 5;
    const int per = std::max(1, iters / nb);
    burst(1);
    double best = 1e30, worst = 0.0;
    for (int b = 0; b < nb; ++b) {
        const double ms = burst(per);
        best = std::min(best, ms);
        worst = std::max(worst, ms);
    }
    const double bytes = (double)p.weight_bytes + (double)p.out_elems * 4.0
                       + (double)M * HID * 4.0;
    printf("SPREAD best %.4f ms / worst %.4f ms (%d bursts x %d)\n", best, worst, nb, per);
    printf("RESULT arm=%s M=%d rows_served=%d out_elems=%llu grid=%d rpb=%d table_bytes=%llu "
           "fetch_per_weight=%.6f poison_left=%zu ms=%.4f gbs=%.1f\n",
           arm.c_str(), M, M, p.out_elems, grid, rpb, p.weight_bytes, fetch_per_weight,
           poison_left, best, bytes / (best / 1e3) / 1e9);

    CK(hipFree(dw));
    CK(hipFree(dh));
    CK(hipFree(dout_raw));
    if (cs) CK(hipFree(cs));
    if (cnt) CK(hipFree(cnt));
    if (rawsl) CK(hipFree(rawsl));
    CK(hipHostFree(hw));
    return 0;
}
