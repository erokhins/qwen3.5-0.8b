// §02.3's embedding lookup as ONE gather over a wave (D30 4.2′), and the only other TU in engine/ that
// contains a kernel it did not lift from a prototype. `kernels_cast.cpp` is the first: this file, like
// that one, holds a shape the engine had to grow for its own reasons rather than an op that graduated
// from a `prototype/<op>/` directory with a gate of its own.
//
// WHY A KERNEL EXISTS FOR A LOOKUP (the number, not the idea). Until now `submit_wave_body` submitted
// one `hipMemcpyAsync` of `kHidden` BF16 per token: 4096 separate 2 KiB device-to-device copies in a
// 4096-lane wave, which is 4096 of the wave's 4240 `copy` operations and 96.6 % of that family's
// submissions. `tools/submission_probe.cpp` measured exactly that shape: **2.05 µs per copy against an
// empty launch's 1.72 µs**, so the bytes of a 2 KiB row are ~0.33 µs of the op and its submission is
// 1.72 µs — 84 % — and the same 8 MiB moved as ONE operation costs **0.10 ms instead of 8.40 ms**
// (`progress/2026-10-17/glue-cast-copy-bytes-census.md` §8). The family's traffic was never the
// problem: 38 MB/wave is 0.16 ms at the read probe against a 10.86 ms row. Submissions were, so this
// kernel's job is to delete 4095 of them per wave and not to be clever about bytes.
//
// WHY IT CANNOT MOVE A BIT. There is no arithmetic here at all. Row `t` of the output is the `kHidden`
// BF16 values at `table + ids[t] * kHidden`, and each one is stored exactly as it was loaded — the same
// bytes the per-token `hipMemcpyAsync` wrote, in the same order. The vector width is a layout choice,
// not a computation: a row is `kHidden * 2 = 2048` bytes and `kHidden` is a whole number of `uint2`
// (asserted below), and both a source row and a destination row start on a 2048-byte boundary —
// `table` and `sc.embed` are `hipMalloc`'d at 256-byte alignment and a row is 2048 bytes — so every
// access below is naturally aligned and covers the row exactly once with no remainder.
//
// THE WITNESS THAT THIS IS THE SAME BYTES AS THE COPIES. `engine/gate.py` section (h) on `--gemm hand`
// compares a wide wave's logits byte-for-byte against the same prompt fed one token per body. The
// one-token-per-body side is STILL the memcpy — `submit_wave_body`'s row-count rule keeps a one-row
// body on the copy path (see the site in `decode_step.cpp`) — so that check is gather-against-copy and
// not gather-against-itself. That is also why a bit difference here is a bug and not a band question.
#include "kernels.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "engine/state.h"   // kHidden: the row width the gather is defined over

namespace {

// One row is walked as this many 8-byte chunks. `kHidden` is BF16, so a row is kHidden * 2 bytes.
constexpr int kChunksPerRow = engine::kHidden / 4;
static_assert(engine::kHidden % 4 == 0,
              "embed_gather's uint2 walk needs a row that is a whole number of uint2 chunks; a "
              "remainder would leave the tail of every row unwritten, which is a silent wrong answer.");

// One block per wave row, kChunksPerRow threads wide: lane `c` moves the `c`-th 8-byte chunk of that
// row. `ids[blockIdx.x]` is read by every lane of the block, so it is one broadcast load — the whole
// indirection cost of the gather is that one word per block, and the row itself is a straight copy.
// No bounds guard on `threadIdx.x` because the launch width is exactly kChunksPerRow, and no guard on
// `blockIdx.x` because the launch grid is exactly `rows`; the launcher refuses anything else.
__global__ void embed_gather_kernel(const hip_bfloat16* __restrict__ table,
                                    const int* __restrict__ ids,
                                    hip_bfloat16* __restrict__ out) {
    const uint2* src = (const uint2*)(table + (size_t)ids[blockIdx.x] * engine::kHidden);
    uint2* dst = (uint2*)(out + (size_t)blockIdx.x * engine::kHidden);
    dst[threadIdx.x] = src[threadIdx.x];
}

}  // namespace

namespace engine::kernels {

void embed_gather_rows(const uint16_t* table, const int* ids, uint16_t* out, int rows) {
    // A throw rather than a silent no-op for a zero-row call: unlike `cast_bf16_to_f32`, where n = 0 is
    // an empty region of a call that legitimately has one, a body of zero rows is not a thing
    // `submit_wave_body` can reach, so reaching this launcher with rows = 0 is a caller bug.
    if (rows < 1) {
        throw std::runtime_error("embed_gather_rows: rows must be >= 1, got " + std::to_string(rows));
    }
    if (table == nullptr || ids == nullptr || out == nullptr) {
        throw std::runtime_error(
            "embed_gather_rows: a null table/ids/out would be a wild write of a whole row");
    }
    embed_gather_kernel<<<rows, kChunksPerRow>>>((const hip_bfloat16*)table, ids,
                                                 (hip_bfloat16*)out);
}

}  // namespace engine::kernels
