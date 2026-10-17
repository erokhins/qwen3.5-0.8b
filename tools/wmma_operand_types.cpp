// Which WMMA operand types does this toolchain actually accept for gfx1151?
// This decides the shape of the attention matrix route: a bf16 WMMA keeps the engine's operands as they are,
// while fp16-only means every q/k operand needs a conversion (a numerics change under D13, not a free one).
// VARIANT 0 = bf16 builtin, 1 = f16 builtin (the one `devices/strix-halo-official-specs.md` already records).
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

typedef float f32 __attribute__((ext_vector_type(8)));
#if VARIANT == 0
typedef unsigned short u16 __attribute__((ext_vector_type(16)));
typedef __hip_bfloat16 T;
#define WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c)
#define TNAME "bf16"
#else
typedef unsigned short u16 __attribute__((ext_vector_type(16)));
typedef __half T;
#define WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
#define TNAME "f16"
#endif

__global__ void probe(const unsigned short* a, const unsigned short* b, float* c)
{
    u16 fa, fb;
    for (int i = 0; i < 16; ++i) { fa[i] = a[i]; fb[i] = b[i]; }
    f32 acc = {};
    acc = WMMA(fa, fb, acc);
    for (int i = 0; i < 8; ++i) c[i] = acc[i];
}

int main() { return (int)TNAME[0] & 1; }
