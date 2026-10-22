/*
 * prototype/mtp_draft/mtp_draft.hip.cpp — host entry for the MTP draft-block rig (§09.2 at the real
 * shapes). The gate is `prototype/mtp_draft/gate.py`: it builds the inputs from the pinned fixtures and
 * the checkpoint, runs this binary, and judges every observable against bands it imports from the ops
 * that measured them. This binary decides no number and touches no `engine/` file.
 *
 * WHAT ONE DRAFT STEP IS, launch by launch: `DRAFT-HANDOVER.md` §0. The launches are counted at the
 * launch sites and asserted against the plan, because the block's published cost rests on "~18 launches"
 * which its own source admits was never a census.
 *
 * The block's blobs arrive as one file plus a plan file that STATES each tensor's rank and shape —
 * `engine/weights_contract.py` exists because a claim a checker cannot express is a bug with a
 * compile-time guarantee it will be wrong eventually — and `td_modes.cpp` re-checks every claim against
 * what the kernel about to read the blob needs.
 *
 * FILES the gate writes under --work
 *   block.txt   one line per tensor: name rank d0 d1 d2 elems
 *   block.bin   those tensors' BF16 bytes, concatenated in plan order
 *   inputs.txt  line 1: vocab; then one line per draft step: y_next pos
 *   h_post.bin  the fixture's pinned final_hidden row (1024 BF16)   — §09.1's `h`, A-12 falsified
 *   h_pre.bin   the same fixture's pinned pre_final_hidden row       — the A-12 reading, for the control
 *   W.bin       the tied table, [vocab, 1024] BF16                   — the shared embedding (§09.1)
 *
 * USAGE
 *   prototype/mtp_draft/mtp_draft --work DIR [--out DIR] [--mode chain|witness|xcheck|stagecheck]
 *                                 [--k 1|2|4] [--dump] [--repeat N] [--time-burst]
 *                                 [--fc-order eh|he] [--h-source post|pre]
 *                                 [--pos-shift N] [--pos-last N] [--perturb name:mult,...] [--rpb N]
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <hip/hip_bfloat16.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "td_host.h"

namespace {

void die(const std::string& m) { throw std::runtime_error(m); }
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) die(std::string(what) + ": " + hipGetErrorString(e));
}

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) die("cannot open " + p);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

uint16_t* upload(const std::string& path, long long expect_bytes, const char* what) {
    std::string buf = slurp(path);
    if ((long long)buf.size() != expect_bytes)
        die(std::string(what) + ": " + path + " is " + std::to_string(buf.size()) + " B, expected " +
            std::to_string(expect_bytes));
    void* d = nullptr;
    ck(hipMalloc(&d, (size_t)expect_bytes), what);       // hipMalloc, never managed (AGENTS: managed
    ck(hipMemcpy(d, buf.data(), (size_t)expect_bytes, hipMemcpyHostToDevice), what);  // degrades silently
    return (uint16_t*)d;
}

uint16_t* upload_bits(const std::vector<uint16_t>& v, const char* what) {
    void* d = nullptr;
    ck(hipMalloc(&d, v.size() * 2), what);
    ck(hipMemcpy(d, v.data(), v.size() * 2, hipMemcpyHostToDevice), what);
    return (uint16_t*)d;
}

float bf_of(uint16_t b) {
    const uint32_t u = ((uint32_t)b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t bits_of(float f) {
    const hip_bfloat16 b = hip_bfloat16(f);
    uint16_t out;
    std::memcpy(&out, &b, 2);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) die("missing value for " + a);
            return std::string(argv[++i]);
        };
        if (a == "--work") o.work = next();
        else if (a == "--out") o.out = next();
        else if (a == "--mode") o.mode = next();
        else if (a == "--k") o.k = std::stoi(next());
        else if (a == "--repeat") o.repeat = std::stoi(next());
        else if (a == "--time-burst") o.time_burst = true;
        else if (a == "--dump") o.dump = true;
        else if (a == "--fc-order") o.fc_order = next();
        else if (a == "--h-source") o.h_source = next();
        else if (a == "--pos-shift") o.pos_shift = std::stol(next());
        else if (a == "--pos-last") o.pos_last = std::stol(next());
        else if (a == "--perturb") o.perturb = next();
        else if (a == "--rpb") td_rpb_proj = std::stoi(next());
        else die("unknown argument " + a);
    }
    if (o.work.empty()) die("--work is required");
    if (o.dump && o.out.empty()) die("--dump needs --out");
    if (o.dump && o.time_burst) die("--dump and --time-burst are different runs on purpose");
    if (o.out.empty()) o.out = o.work;

    try {
        // ── the plan: name, rank, dims, element count, one per line ──────────────────────────────
        Plan p;
        {
            std::istringstream is(slurp(o.work + "/block.txt"));
            std::string line;
            long long offset = 0;
            while (std::getline(is, line)) {
                if (line.empty()) continue;
                std::istringstream ls(line);
                Blob b;
                long long elems = 0;
                ls >> b.name >> b.rank >> b.dim[0] >> b.dim[1] >> b.dim[2] >> elems;
                if (!ls) die("malformed block.txt line: " + line);
                if (b.rank != 1 && b.rank != 2) die(b.name + ": this block has rank-1 and rank-2 only");
                b.bytes = elems * 2;
                b.offset = offset;
                offset += b.bytes;
                p.blobs.push_back(b);
            }
            p.total = offset;
        }
        if (p.blobs.empty()) die("empty block plan");
        if (p.blobs.size() != (size_t)tdplan::kBlockTensors)
            die("block plan has " + std::to_string(p.blobs.size()) + " tensors; the census says " +
                std::to_string(tdplan::kBlockTensors));
        if (p.total != tdplan::kBlockBytes)
            die("block plan totals " + std::to_string(p.total) + " B; the census says " +
                std::to_string(tdplan::kBlockBytes));

        // ── the bytes, perturbed on the host copy if asked, then uploaded ────────────────────────
        std::string raw = slurp(o.work + "/block.bin");
        if ((long long)raw.size() != p.total) die("block.bin size disagrees with the plan");
        std::vector<uint16_t> hb(raw.size() / 2);
        std::memcpy(hb.data(), raw.data(), raw.size());
        if (!o.perturb.empty()) {
            std::istringstream is(o.perturb);
            std::string item;
            while (std::getline(is, item, ',')) {
                const size_t c = item.rfind(':');
                if (c == std::string::npos) die("--perturb wants name:mult[,name:mult]");
                const std::string name = item.substr(0, c);
                const double mult = std::stod(item.substr(c + 1));
                const Blob* b = nullptr;
                for (const auto& bb : p.blobs) if (bb.name == name) b = &bb;
                if (!b) die("no block tensor named " + name);
                for (long long i = 0; i < b->bytes / 2; ++i)
                    hb[b->offset / 2 + i] = bits_of((float)(bf_of(hb[b->offset / 2 + i]) * mult));
            }
        }
        uint32_t* arena = nullptr;
        ck(hipMalloc((void**)&arena, (size_t)(2 * p.guard_words) * 4 + p.total), "arena");
        p.guard_lo = arena;
        p.dev = (uint16_t*)(arena + p.guard_words);
        p.guard_hi = (uint32_t*)(p.dev + p.total / 2);
        td_poison_range(p.guard_lo, p.guard_words);
        td_poison_range(p.guard_hi, p.guard_words);
        for (const auto& b : p.blobs)
            ck(hipMemcpy(p.dev + b.offset / 2, hb.data() + b.offset / 2, (size_t)b.bytes,
                         hipMemcpyHostToDevice), "blob upload");

        // ── the step list and the fixture rows ───────────────────────────────────────────────────
        Inputs in;
        {
            std::istringstream is(slurp(o.work + "/inputs.txt"));
            std::string line;
            if (!std::getline(is, line)) die("inputs.txt is empty");
            in.vocab = std::stoll(line);
            while (std::getline(is, line)) {
                if (line.empty()) continue;
                std::istringstream ls(line);
                StepIn s;
                ls >> s.y_next >> s.pos;
                in.steps.push_back(s);
            }
        }
        if ((o.mode == "chain" || o.mode == "stagecheck") && (int)in.steps.size() != o.k)
            die("inputs.txt carries " + std::to_string(in.steps.size()) + " steps but --k is " +
                std::to_string(o.k));
        // The launch and traffic arms do not consume the step list, so they must not be held to it: the
        // timing harness runs them against the same work directory the numeric arm left behind.
        if (o.mode == "chain" || o.mode == "stagecheck") {
            in.h_post = upload(o.work + "/h_post.bin", 1024 * 2, "h_post");
            in.h_pre = upload(o.work + "/h_pre.bin", 1024 * 2, "h_pre");
        } else {
            in.h_post = in.h_pre = nullptr;
        }
        // Only the arms that price the round read the 485 MiB table; the traffic, xcheck and launch arms
        // do not, and uploading it for them would make the arm's cost a copy of the embedding.
        if (o.mode == "chain" || o.mode == "stagecheck")
            in.table = upload(o.work + "/W.bin", in.vocab * 1024 * 2, "tied table");

        if (o.mode == "chain") mode_chain(o, p, in);
        else if (o.mode == "launch") mode_launch(o);
        else if (o.mode == "witness") mode_witness(o, p);
        else if (o.mode == "xcheck") mode_xcheck(p);
        else if (o.mode == "stagecheck") mode_stagecheck(p, in);
        else die("unknown --mode " + o.mode);

        printf("PLAN tensors=%d block_bytes=%lld launches_per_step=%d steps=%zu\n",
               (int)p.blobs.size(), p.total, tdplan::kLaunchesPerDraftStep, in.steps.size());
        (void)hipDeviceSynchronize();
    } catch (const std::exception& e) {
        fprintf(stderr, "mtp_draft FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
