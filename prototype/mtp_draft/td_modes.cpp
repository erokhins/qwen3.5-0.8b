/*
 * prototype/mtp_draft/td_modes.cpp — the four modes of the draft-block rig, and the only TU that
 * includes `mtp_draft_kernels.h` (its `__global__` definitions need exactly one home; the shipped op
 * headers each get their own TU for the same kind of reason).
 *
 * `mode_chain` is the admission path: k chained §09.2 draft steps, one enqueue, no host sync inside,
 * launched entirely through the shipped kernels the engine will use. `mode_witness` is the traffic arm.
 * `mode_xcheck` is the fidelity proof for the one stage that could not be reused verbatim (the dense
 * MTP cache; DRAFT-HANDOVER.md §3). `mode_stagecheck` proves the all-BF16 activation chain is
 * bit-identical to F32 staging on this block's operands, which is what buys the missing cast launches.
 */
#include "td_host.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mtp_draft_kernels.h"
#include "td_launchers.h"

// The two globals every launcher touches, declared in `td_launchers.h` and `td_host.h`. Defined here
// because this TU is the one that drives all four families.
long td_launches = 0;
int td_rpb_proj = 32;          // engine/kernels_proj.cpp's kRowsPerBlock

namespace {

void die(const std::string& m) { throw std::runtime_error(m); }
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) die(std::string(what) + ": " + hipGetErrorString(e));
}

const Blob& need(const Plan& p, const std::string& name, int rank, long long d0, long long d1) {
    for (const auto& b : p.blobs) {
        if (b.name != name) continue;
        const long long want = (rank == 1 ? d0 : d0 * d1) * 2;
        if (b.rank != rank || (rank == 1 && b.dim[0] != d0) ||
            (rank == 2 && (b.dim[0] != d0 || b.dim[1] != d1)) || b.bytes != want) {
            std::ostringstream ss;
            ss << name << ": plan says rank " << b.rank << " [" << b.dim[0] << "," << b.dim[1] << "] "
               << b.bytes << " B, but the kernel that reads it needs rank " << rank << " [" << d0
               << "," << d1 << "] " << want << " B";
            die(ss.str());
        }
        return b;
    }
    die("block plan has no tensor named " + name);
    static Blob none;
    return none;
}

uint16_t* at(const Plan& p, const std::string& n, int r, long long d0, long long d1) {
    return p.dev + need(p, n, r, d0, d1).offset / 2;
}

uint16_t* todev(const void* h, size_t bytes, const char* what) {
    void* d = nullptr;
    ck(hipMalloc(&d, bytes ? bytes : 1), what);
    if (bytes) ck(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice), what);
    return (uint16_t*)d;
}

void xfree(void* p) { (void)hipFree(p); }   // the HIP headers mark hipFree nodiscard; a teardown
                                            // failure here is reported by the next ck(), not ignored

void back(void* h, const void* d, size_t bytes) {
    ck(hipMemcpy(h, d, bytes, hipMemcpyDeviceToHost), "readback");
}

void dump_u16(const std::string& out, const std::string& name, const void* d, long n) {
    std::vector<uint16_t> h(n);
    back(h.data(), d, n * 2);
    std::ofstream f(out + "/" + name, std::ios::binary);
    f.write((const char*)h.data(), n * 2);
}

// The scratch arena's own two guard bands (the block arena has its own pair, Plan::guard_left).
long scr_guard(uint32_t* raw, long guard, long mid_words) {
    return td_count_poison(raw, guard) + td_count_poison(raw + guard + mid_words, guard);
}

}  // namespace

void td_poison_range(uint32_t* d, long words) {
    const int threads = 256;
    const long blocks = std::max<long>(1, std::min<long>((words + threads - 1) / threads, 2048));
    td_poison<<<(int)blocks, threads>>>(d, words);
    ck(hipGetLastError(), "poison launch");
}

long td_count_poison(const uint32_t* d, long words) {
    std::vector<uint32_t> h(words);
    back(h.data(), d, words * 4);
    long left = 0;
    for (long i = 0; i < words; ++i) if (h[i] == TD_POISON) ++left;
    return left;
}

long Plan::guard_left() const { return td_count_poison(guard_lo, guard_words) +
                                      td_count_poison(guard_hi, guard_words); }

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// chain
// ═══════════════════════════════════════════════════════════════════════════════════════════════
void mode_chain(const Opts& o, const Plan& p, const Inputs& in) {
    const int k = (int)in.steps.size();
    if (k < 1) die("chain needs at least one step");
    for (auto& s : in.steps)
        if (s.y_next < 0 || s.y_next >= in.vocab) die("y_next is outside the tied table");

    uint16_t* w_fc = at(p, "mtp.fc.weight", 2, 1024, 2048);
    uint16_t* w_q = at(p, "mtp.layers.0.self_attn.q_proj.weight", 2, 4096, 1024);
    uint16_t* w_k = at(p, "mtp.layers.0.self_attn.k_proj.weight", 2, 512, 1024);
    uint16_t* w_v = at(p, "mtp.layers.0.self_attn.v_proj.weight", 2, 512, 1024);
    uint16_t* w_o = at(p, "mtp.layers.0.self_attn.o_proj.weight", 2, 1024, 2048);
    uint16_t* w_gate = at(p, "mtp.layers.0.mlp.gate_proj.weight", 2, 3584, 1024);
    uint16_t* w_up = at(p, "mtp.layers.0.mlp.up_proj.weight", 2, 3584, 1024);
    uint16_t* w_down = at(p, "mtp.layers.0.mlp.down_proj.weight", 2, 1024, 3584);
    uint16_t* w_emb = at(p, "mtp.pre_fc_norm_embedding.weight", 1, 1024, 0);
    uint16_t* w_hid = at(p, "mtp.pre_fc_norm_hidden.weight", 1, 1024, 0);
    uint16_t* w_inln = at(p, "mtp.layers.0.input_layernorm.weight", 1, 1024, 0);
    uint16_t* w_paln = at(p, "mtp.layers.0.post_attention_layernorm.weight", 1, 1024, 0);
    uint16_t* w_mn = at(p, "mtp.norm.weight", 1, 1024, 0);
    uint16_t* w_qn = at(p, "mtp.layers.0.self_attn.q_norm.weight", 1, 256, 0);
    uint16_t* w_kn = at(p, "mtp.layers.0.self_attn.k_norm.weight", 1, 256, 0);

    // One poison-prefilled scratch arena: `poison_left = 0` then MEANS "every output word of every
    // step was written", which is a byte fact rather than an assumption.
    struct Sz { const char* name; long per_step; };
    const Sz szs[] = {
        {"e", 1024}, {"pair", 2048}, {"u", 1024}, {"nrm", 1024}, {"qg", 4096}, {"kg", 512},
        {"vg", 512}, {"qrop", 2048}, {"krop", 512}, {"gate", 2048}, {"aout", 2048}, {"gout", 2048},
        {"attn_o", 1024}, {"z1", 1024}, {"u2", 1024}, {"act", 3584}, {"mlp_o", 1024}, {"z2", 1024},
        {"xd", 1024}};
    const int NS = (int)(sizeof(szs) / sizeof(szs[0]));
    std::vector<long> off(NS + 1, 0);
    for (int i = 0; i < NS; ++i) off[i + 1] = off[i] + szs[i].per_step * k;
    const long guard = 2048;                          // words, each side
    // The guard is `guard` WORDS per side, so the allocation is mid_bytes + 8*guard, not + 4*guard. The
    // first version wrote 8 KiB past this buffer with the poison fill and the guard check read the same
    // out-of-bounds tail, which is how `guard_left` came back 12160 of 12288 instead of failing loudly.
    uint32_t* scr_raw = nullptr;
    ck(hipMalloc(&scr_raw, (off[NS] + 4 * guard) * 2), "scratch");
    td_poison_range(scr_raw, off[NS] / 2 + 2 * guard);
    uint16_t* scr = (uint16_t*)(scr_raw + guard);
    auto buf = [&](int i, int j) { return scr + off[i] + szs[i].per_step * j; };
    enum { B_E, B_PAIR, B_U, B_NRM, B_QG, B_KG, B_VG, B_QROP, B_KROP, B_GATE, B_AOUT, B_GOUT,
           B_ATTNO, B_Z1, B_U2, B_ACT, B_MLPO, B_Z2, B_XD };

    // The block's own 1-layer KV cache (§09.1), dense lanes, poison-filled so a lane read before it was
    // appended cannot masquerade as a rounding difference.
    const int cap = 64;
    if (k > cap) die("chain longer than the rig's cache capacity");
    uint16_t *kc, *vc;
    ck(hipMalloc(&kc, (size_t)2 * cap * 256 * 2), "kcache");
    ck(hipMalloc(&vc, (size_t)2 * cap * 256 * 2), "vcache");
    td_poison_range((uint32_t*)kc, (long)2 * cap * 256 / 2);
    td_poison_range((uint32_t*)vc, (long)2 * cap * 256 / 2);

    std::vector<int> ids(k);
    for (int j = 0; j < k; ++j) ids[j] = (int)in.steps[j].y_next;
    int* d_ids = (int*)todev(ids.data(), k * sizeof(int), "ids");

    const bool he = (o.fc_order == "he");
    if (!he && o.fc_order != "eh") die("--fc-order must be eh or he");
    if (o.h_source != "post" && o.h_source != "pre") die("--h-source must be post or pre");
    const uint16_t* h0 = (o.h_source == "pre") ? in.h_pre : in.h_post;

    const int reps = std::max(1, o.repeat);
    auto issue = [&]() {
    for (int j = 0; j < k; ++j) {
        const long pos = in.steps[j].pos + o.pos_shift + (j == k - 1 ? o.pos_last : 0);
        // 1  the tied-table row of the token being drafted from (§09.1, shared table)
        td_gather_row<<<1, TD_CHUNKS_PER_ROW>>>((const hip_bfloat16*)in.table, d_ids + j,
                                               (hip_bfloat16*)buf(B_E, j));
        td_launches += 1;
        // 2-3  the two pre-FC norms, written straight into the halves of the fc operand. THE CONCAT
        // ORDER IS THESE TWO DESTINATION POINTERS, so the A-13 reading is an operand-staging fact the
        // gate can flip (§09.5: 0.642 accept for eh against 0.014 for he) and not a branch inside
        // arithmetic that a timing run could take by accident.
        td::norm_only(buf(B_E, j), w_emb, buf(B_PAIR, j) + (he ? 1024 : 0));
        td::norm_only(j == 0 ? h0 : buf(B_XD, j - 1), w_hid, buf(B_PAIR, j) + (he ? 0 : 1024));
        // 4  the fusion fc: [1, 2048] -> [1, 1024]
        td::gemv(w_fc, buf(B_PAIR, j), buf(B_U, j), 1024, 2048);
        // 5-8  the §05 block's operands
        td::norm_only(buf(B_U, j), w_inln, buf(B_NRM, j));
        td::gemv(w_q, buf(B_NRM, j), buf(B_QG, j), 4096, 1024);
        td::gemv(w_k, buf(B_NRM, j), buf(B_KG, j), 512, 1024);
        td::gemv(w_v, buf(B_NRM, j), buf(B_VG, j), 512, 1024);
        // 9  the attention row, over the block's OWN cache, at the drafted ABSOLUTE position
        td::AttnRow ar{};
        ar.qg = buf(B_QG, j); ar.kg = buf(B_KG, j); ar.vg = buf(B_VG, j);
        ar.qn = w_qn; ar.kn = w_kn; ar.kcache = kc; ar.vcache = vc;
        ar.aout = buf(B_AOUT, j); ar.gout = buf(B_GOUT, j);
        ar.qrope_out = buf(B_QROP, j); ar.krope_out = buf(B_KROP, j); ar.gate_out = buf(B_GATE, j);
        ar.cap = cap; ar.nlanes = j + 1; ar.self_lane = j; ar.pos = (int)pos;
        td::mtp_attn_row(ar);
        // 10-11  o_proj, then the §03.2 add fused with the next norm (the engine's pairing)
        td::gemv(w_o, buf(B_GOUT, j), buf(B_ATTNO, j), 1024, 2048);
        td::add_norm(buf(B_U, j), buf(B_ATTNO, j), w_paln, buf(B_Z1, j), buf(B_U2, j));
        // 12-14  the MLP, then the second add fused with `mtp.norm`, whose output IS x_d (§09.2)
        td::gateup(w_gate, w_up, buf(B_U2, j), buf(B_ACT, j), 1024);
        td::down(w_down, buf(B_ACT, j), buf(B_MLPO, j), 3584);
        td::add_norm(buf(B_Z1, j), buf(B_MLPO, j), w_mn, buf(B_Z2, j), buf(B_XD, j));
    }
    };
    // The burst form the ledger trusts (`engine/performance-status.md` §6: `hipEventElapsedTime` reads
    // exactly 0 for sub-millisecond work, and N launches plus one sync on `steady_clock` is the timer to
    // believe), with two warm reps first because the first GPU pass over fresh pages runs ~35 GB/s.
    if (o.time_burst) { issue(); issue(); }
    const long launches_before = td_launches;
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) issue();
    ck(hipDeviceSynchronize(), "chain sync");
    const auto t1 = std::chrono::steady_clock::now();
    const long launches_made = td_launches - launches_before;
    const double burst_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    const long poison_left = td_count_poison((const uint32_t*)scr, off[NS] / 2);
    const long guard_left = scr_guard(scr_raw, guard, off[NS] / 2) + p.guard_left();
    printf("RESULT mode=chain arm=block k=%d launches=%ld launches_expected=%d tensors=%d "
           "weight_bytes=%lld rpb=%d poison_left=%ld guard_left=%ld fc_order=%s h_source=%s "
           "pos_shift=%ld pos_last=%d\n",
           k, td_launches, tdplan::kLaunchesPerDraftStep * k * reps, (int)p.blobs.size(), p.total,
           td_rpb_proj, poison_left, guard_left, o.fc_order.c_str(), o.h_source.c_str(),
           o.pos_shift, (int)o.pos_last);
    if (poison_left) die("unwritten output words in the scratch arena");
    if (guard_left != 2 * guard + 2 * p.guard_words) die("a guard band was overwritten");
    if (launches_made != (long)tdplan::kLaunchesPerDraftStep * k * reps)
        die("launch census disagrees with the plan (D2's ~18 was never a census; this one is)");
    if (o.time_burst)
        printf("RESULT mode=chain arm=block burst_ms=%.4f reps=%d k=%d launches=%ld\n",
               burst_ms, reps, k, launches_made);

    if (o.dump) {
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < NS; ++i)
                dump_u16(o.out, "step" + std::to_string(j) + "." + szs[i].name + ".bin",
                         buf(i, j), szs[i].per_step);
        // The cache is plane-major per KV head — `prototype/attn_decode/attn_decode_kernels.h`'s own
        // `kcache[(kv * cap + lane) * DH]`, which this rig transcribes rather than invents — so the dump
        // walks the two planes and takes each plane's first k rows, and the untouched tail is checked
        // PER PLANE. A row-major assumption here would have compared a transposed cache and passed.
        const long dh = 256;
        uint16_t* caches[2] = {kc, vc};
        for (int c = 0; c < 2; ++c) {
            std::vector<uint16_t> h((size_t)2 * k * dh);
            for (int kv = 0; kv < 2; ++kv) {
                const uint16_t* plane = caches[c] + (size_t)kv * cap * dh;
                back(h.data() + (size_t)kv * k * dh, plane, (size_t)k * dh * 2);
                const long tail_words = (cap - k) * dh / 2;
                if (td_count_poison((const uint32_t*)(plane + k * dh), tail_words) != tail_words)
                    die(std::string(c ? "v" : "k") + " cache plane " + std::to_string(kv) +
                        ": a lane outside the chain's own append range was written — the append index "
                        "is not dense");
            }
            std::ofstream f(o.out + "/cache_" + (c ? "v" : "k") + ".bin", std::ios::binary);
            f.write((const char*)h.data(), (long)h.size() * 2);
        }
        printf("DUMP k=%d per_step=%d cache_rows=%d layout=plane-major[kv][cap][256]\n", k, cap, k);
    }
    xfree(scr_raw); xfree(kc); xfree(vc); xfree(d_ids);
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// launch — what a submission costs on this box, measured in the same shape the block uses
// ════════════════════════════════════════════════════════════════════════════════════════════
// The ledger's 3.10 µs/launch came from another rig; the block's cost model is 14 submissions per step,
// so the overhead term is re-measured here against a kernel that does nothing (a poison pass over zero
// words: one block, no memory traffic) and burst-timed the same way the block is. `run_locked.sh` prints
// it beside the block arms so the overhead is attributed inside one lock hold rather than carried in from
// a different window — the "a ruler re-taken in a different window is not an A/B" rule.
void mode_launch(const Opts& o) {
    uint32_t* scratch = nullptr;
    ck(hipMalloc((void**)&scratch, 64), "launch scratch");
    const int reps = std::max(1, o.repeat);
    const int n = reps * tdplan::kLaunchesPerDraftStep;
    for (int i = 0; i < tdplan::kLaunchesPerDraftStep; ++i) { td_poison_range(scratch, 0); }
    (void)hipDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) { td_poison_range(scratch, 0); td_launches += 1; }
    ck(hipDeviceSynchronize(), "launch sync");
    const auto t1 = std::chrono::steady_clock::now();
    // Without --time-burst this arm prints NO number at all, so the admission pass cannot accidentally
    // become a measurement; with it, the per-launch cost is the burst divided by the launches made.
    if (!o.time_burst) {
        printf("RESULT mode=launch launches=%d timed=0\n", n);
    } else {
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
        printf("RESULT mode=launch launches=%d timed=1 burst_us_per_launch=%.4f reps=%d\n", n, us, reps);
    }
    xfree(scratch);
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// witness — the traffic arm
// ════════════════════════════════════════════════════════════════════════════════════════════
// Reads the block's byte multiset exactly once and reports (n, raw, addr). gate.py recomputes the
// triple from ITS copy of the same bytes and the same segment order, so the verdict is never derived
// from the value the kernel accumulated (the `tools/mixed_bw` rule).
void mode_witness(const Opts& o, const Plan& p) {
    std::vector<TdSeg> hs(p.blobs.size());
    long long flat = 0;
    for (size_t i = 0; i < p.blobs.size(); ++i) {
        if (p.blobs[i].bytes % 8) die("blob is not a whole number of 8-byte units");
        hs[i].base = (const uint2*)(p.dev + p.blobs[i].offset / 2);
        hs[i].n = p.blobs[i].bytes / 8;
        hs[i].prefix = flat;
        flat += hs[i].n;
    }
    TdSeg* dsegs = nullptr;
    ck(hipMalloc(&dsegs, hs.size() * sizeof(TdSeg)), "segs");
    ck(hipMemcpy(dsegs, hs.data(), hs.size() * sizeof(TdSeg), hipMemcpyHostToDevice), "segs cp");
    const int grid = 320;
    unsigned long long* dout = nullptr;
    ck(hipMalloc(&dout, (size_t)grid * 3 * 8), "witness out");
    ck(hipMemset(dout, 0, (size_t)grid * 3 * 8), "witness zero");
    for (int rep = 0; rep < std::max(1, o.repeat); ++rep) {
        td_read_witness<256><<<grid, 256>>>(dsegs, (int)hs.size(), dout);
        td_launches += 1;
    }
    ck(hipDeviceSynchronize(), "witness sync");
    std::vector<unsigned long long> h((size_t)grid * 3);
    back(h.data(), dout, h.size() * 8);
    unsigned long long n = 0, raw = 0, addr = 0;
    for (int b = 0; b < grid; ++b) { n += h[b * 3]; raw += h[b * 3 + 1]; addr += h[b * 3 + 2]; }
    printf("RESULT mode=witness arm=witness k=%d launches=%ld bytes=%lld uint2_read=%llu raw=%llu "
           "addr=%llu\n", o.k, td_launches, p.total, n, raw, addr);
    if ((long long)n != p.total / 8)
        die("witness read a different number of 8-byte units than the census promises");
    xfree(dsegs); xfree(dout);
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// xcheck — the transcribed row vs the shipped one
// ════════════════════════════════════════════════════════════════════════════════════════════
// Positions contiguous from 0 make `pos == dense lane index`, which is the only geometry where the two
// addressing rules coincide; everywhere else a difference could be the seam rather than a mistake. On
// this geometry the requirement is bit-identity, step by step, on both attention outputs.
void mode_xcheck(const Plan& p) {
    const int L[] = {1, 2, 3, 8, 9, 17, 64, 128, 256, 512};
    const int cap = 512;
    auto lcg = [](uint32_t& s) { s = s * 1664525u + 1013904223u; return (uint16_t)((s >> 16) & 0x3F80);
                          };
    uint32_t st = 12345u;
    std::vector<uint16_t> hq(4096), hk(512), hv(512), hqn(256, 0x3F80), hkn(256, 0x3F80);
    for (auto& v : hq) v = lcg(st);
    for (auto& v : hk) v = lcg(st);
    for (auto& v : hv) v = lcg(st);
    uint16_t *dq = todev(hq.data(), hq.size() * 2, "q"), *dk = todev(hk.data(), hk.size() * 2, "k");
    uint16_t *dv = todev(hv.data(), hv.size() * 2, "v");
    uint16_t *dqn = todev(hqn.data(), hqn.size() * 2, "qn"), *dkn = todev(hkn.data(), hkn.size() * 2, "kn");
    uint16_t* stage = nullptr;
    ck(hipMalloc(&stage, (2048 + 512 + 2048) * 2), "stage");
    ck(hipMemset(stage, 0, (2048 + 512 + 2048) * 2), "stage zero");
    uint16_t* qrop = stage;
    uint16_t* krop = stage + 2048;
    uint16_t* gt = stage + 2048 + 512;

    long diff_a = 0, diff_g = 0, compared = 0, steps = 0;
    // aout/gout must be DEVICE buffers. They were host `std::vector`s in the first version, which the
    // kernel dutifully wrote to and the GPU answered with HSA_STATUS_ERROR_MEMORY_FAULT — the same shape
    // of mistake AGENTS records for a staged launch plan with a half-set pointer: the fault names an
    // address, and the address is a host mapping.
    const int LMAX = 512;
    uint16_t *da_m = nullptr, *dg_m = nullptr, *da_s = nullptr, *dg_s = nullptr;
    ck(hipMalloc((void**)&da_m, 2048L * LMAX * 2), "xcheck aout transcribed");
    ck(hipMalloc((void**)&dg_m, 2048L * LMAX * 2), "xcheck gout transcribed");
    ck(hipMalloc((void**)&da_s, 2048L * LMAX * 2), "xcheck aout shipped");
    ck(hipMalloc((void**)&dg_s, 2048L * LMAX * 2), "xcheck gout shipped");
    for (int li = 0; li < (int)(sizeof(L) / sizeof(L[0])); ++li) {
        const int n = L[li];
        std::vector<uint16_t> am(2048L * n), gm(2048L * n), as(2048L * n), gs(2048L * n);
        uint16_t *kc1, *vc1, *kc2, *vc2;
        ck(hipMalloc((void**)&kc1, (size_t)2 * cap * 256 * 2), "kc1");
        ck(hipMalloc((void**)&vc1, (size_t)2 * cap * 256 * 2), "vc1");
        ck(hipMalloc((void**)&kc2, (size_t)2 * cap * 256 * 2), "kc2");
        ck(hipMalloc((void**)&vc2, (size_t)2 * cap * 256 * 2), "vc2");
        for (int pass = 0; pass < 2; ++pass) {
            for (int lane = 0; lane < n; ++lane) {
                td::AttnRow r{};
                r.qg = dq; r.kg = dk; r.vg = dv; r.qn = dqn; r.kn = dkn;
                r.kcache = pass ? kc2 : kc1; r.vcache = pass ? vc2 : vc1;
                r.aout = (pass ? da_s : da_m) + 2048L * lane;
                r.gout = (pass ? dg_s : dg_m) + 2048L * lane;
                r.qrope_out = qrop; r.krope_out = krop; r.gate_out = gt;
                r.cap = cap; r.nlanes = lane + 1; r.self_lane = lane; r.pos = lane;
                if (pass) td::shipped_attn_row(r); else td::mtp_attn_row(r);
                steps += 1;
            }
        }
        ck(hipDeviceSynchronize(), "xcheck sync");
        back(am.data(), da_m, 2048L * n * 2);
        back(gm.data(), dg_m, 2048L * n * 2);
        back(as.data(), da_s, 2048L * n * 2);
        back(gs.data(), dg_s, 2048L * n * 2);
        for (long i = 0; i < 2048L * n; ++i) {
            compared += 2;
            if (am[i] != as[i]) diff_a += 1;
            if (gm[i] != gs[i]) diff_g += 1;
        }
        xfree(kc1); xfree(vc1); xfree(kc2); xfree(vc2);
    }
    xfree(da_m); xfree(dg_m); xfree(da_s); xfree(dg_s);
    printf("RESULT mode=xcheck steps=%ld compared=%ld diff_aout=%ld diff_gout=%ld\n",
           steps, compared, diff_a, diff_g);
    if (diff_a + diff_g) die("the transcribed attention row is NOT bit-identical to attn_core");
    (void)p;
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// stagecheck — is BF16 staging free of consequence?
// ════════════════════════════════════════════════════════════════════════════════════════════
void mode_stagecheck(const Plan& p, const Inputs& in) {
    uint16_t* w_fc = at(p, "mtp.fc.weight", 2, 1024, 2048);
    uint16_t* w_gate = at(p, "mtp.layers.0.mlp.gate_proj.weight", 2, 3584, 1024);
    uint16_t* w_up = at(p, "mtp.layers.0.mlp.up_proj.weight", 2, 3584, 1024);
    uint16_t* w_down = at(p, "mtp.layers.0.mlp.down_proj.weight", 2, 1024, 3584);
    // A pair row out of the fixture's own h, twice, so both arms read identical real bytes.
    std::vector<uint16_t> hp(1024), hp2(2048);
    back(hp.data(), in.h_post, 1024 * 2);
    for (int i = 0; i < 2048; ++i) hp2[i] = hp[i % 1024];
    uint16_t* pair = todev(hp2.data(), 2048 * 2, "pair");
    float* pair32 = nullptr; ck(hipMalloc(&pair32, 2048 * 4), "pair32");
    td_widen<<<8, 256>>>((const hip_bfloat16*)pair, pair32, 2048); td_launches += 1;
    uint16_t *o1 = nullptr, *o2 = nullptr;
    ck(hipMalloc(&o1, 1024 * 2), "o1"); ck(hipMalloc(&o2, 1024 * 2), "o2");
    td::gemv(w_fc, pair, o1, 1024, 2048, true);
    td::gemv_f32(w_fc, pair32, o2, 1024, 2048);
    ck(hipDeviceSynchronize(), "stage sync");
    std::vector<uint16_t> h1(1024), h2(1024);
    back(h1.data(), o1, 2048); back(h2.data(), o2, 2048);
    long d_fc = 0;
    for (int i = 0; i < 1024; ++i) if (h1[i] != h2[i]) ++d_fc;

    // The MLP both ways on the fc's own output.
    float* o1_32 = nullptr; ck(hipMalloc(&o1_32, 1024 * 4), "o1_32");
    td_widen<<<4, 256>>>((const hip_bfloat16*)o1, o1_32, 1024); td_launches += 1;
    uint16_t *a1 = nullptr, *a2 = nullptr, *d1 = nullptr, *d2 = nullptr;
    float* a1_32 = nullptr;
    ck(hipMalloc(&a1, 3584 * 2), "a1"); ck(hipMalloc(&a2, 3584 * 2), "a2");
    ck(hipMalloc(&d1, 1024 * 2), "d1"); ck(hipMalloc(&d2, 1024 * 2), "d2");
    ck(hipMalloc(&a1_32, 3584 * 4), "a1_32");
    td::gateup(w_gate, w_up, o1, a1, 1024);
    td::gateup_f32(w_gate, w_up, o1_32, a2, 1024);
    td_widen<<<14, 256>>>((const hip_bfloat16*)a1, a1_32, 3584); td_launches += 1;
    td::down(w_down, a1, d1, 3584);
    td::down_f32(w_down, a1_32, d2, 3584);   // the SAME act bytes, widened instead of staged BF16
    ck(hipDeviceSynchronize(), "stage sync2");
    std::vector<uint16_t> ha1(3584), ha2(3584), hd1(1024), hd2(1024);
    back(ha1.data(), a1, 3584 * 2); back(ha2.data(), a2, 3584 * 2);
    back(hd1.data(), d1, 1024 * 2); back(hd2.data(), d2, 1024 * 2);
    long d_act = 0, d_dn = 0;
    for (int i = 0; i < 3584; ++i) if (ha1[i] != ha2[i]) ++d_act;
    for (int i = 0; i < 1024; ++i) if (hd1[i] != hd2[i]) ++d_dn;
    printf("RESULT mode=stagecheck fc_diff=%ld act_diff=%ld down_diff=%ld\n", d_fc, d_act, d_dn);
    if (d_fc + d_act + d_dn)
        die("BF16 staging is NOT bit-identical to F32 staging on this block's operands");
    xfree(pair); xfree(pair32); xfree(o1); xfree(o2); xfree(o1_32); xfree(a1_32);
    xfree(a1); xfree(a2); xfree(d1); xfree(d2);
}
