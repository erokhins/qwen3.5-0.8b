// Translation unit whose only job is to make the four slim wave kernels exist so the ISA census can
// read them. Compile-only: `hipcc -O3 --offload-arch=gfx1151 --save-temps -c isa_tu.cpp -o isa_tu.o`
#include "attn_valu_slim_kernels.h"
