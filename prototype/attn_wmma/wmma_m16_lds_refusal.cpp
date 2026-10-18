// wmma_m16_lds_refusal.cpp — a TU that is SUPPOSED to fail compilation, and the witness for the
// largest cost a re-tile to M >= 16 pays. `check_m16_lds_refusal.sh` asserts the failure and its text.
//
// THE CLAIM THIS FILE PROVES. The shipped kernel keeps a whole score row in LDS:
// `__shared__ hip_bfloat16 s_sc[ATTN_TILE_G][MAXSC]` = 4 x 512 x 2 = 4,096 B, affordable only because
// `static_assert(ATTN_TILE_M == 1)` pins one query row per block. The moment a block owns 16 query
// rows, the same design needs 4 heads x 16 rows x 512 lanes x 2 B = 65,536 B for the scores alone —
// the entire LDS budget of a CU on this part (`lds_size_in_kb 64`,
// `prototype/attn_wmma/evidence/device_facts.txt`) — before any K/V/Q stage. clang refuses it:
//
//     local memory (82944) exceeds limit (65536) in function 'm16_whole_row_scores'
//
// so an M = 16 attention block CANNOT keep the shipped two-pass whole-row score staging. It must either
// make the scores tile-local (which is what `wmma_m16_probe.cpp`'s primary variants do, and which costs
// a barrier inside the drain) or fuse pass 1 and pass 2 (a deeper restructure than this item covers).
// This is recorded rather than designed around, because it is a *cost* the funded band has to carry, not
// a defect: the probe's own STRIDE leg shows the drain's slot price does not care which stride is used.

#include <hip/hip_bfloat16.h>

#define DH 256
#define MAXSC 512
#define MROWS 16
#define NWARPS 4

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
m16_whole_row_scores(const hip_bfloat16* __restrict__ q, float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];                       //      8,192 B
    __shared__ hip_bfloat16 s_k[MROWS * DH];                       //      8,192 B
    __shared__ hip_bfloat16 s_sc[NWARPS][MROWS][MAXSC];             //     65,536 B  <-- the wall
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    s_q[lane] = q[lane];
    s_k[lane] = q[lane + 8];
    __syncthreads();
    s_sc[w][lane & 15][lane] = s_q[lane];
    __syncthreads();
    if (sink != nullptr) sink[lane] = (float)s_sc[w][lane & 15][lane] + (float)s_k[lane];
}
