// Minimal rocwmma-for-gfx1151 compile probe: does the library's WMMA path build for this arch?
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void one_wmma(const hip_bfloat16* a, const hip_bfloat16* b, float* out) {
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::col_major> fb;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc;
    rocwmma::fill_fragment(fc, 0.0f);
    rocwmma::load_matrix_sync(fa, a, 256);
    rocwmma::load_matrix_sync(fb, b, 256);
    rocwmma::mma_sync(fc, fa, fb, fc);
    rocwmma::store_matrix_sync(out, fc, 512, rocwmma::mem_row_major);
}
