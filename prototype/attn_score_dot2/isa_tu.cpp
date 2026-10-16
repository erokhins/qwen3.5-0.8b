// isa_tu.cpp — a translation unit whose only job is to make the shipped attention kernels compilable on their own, so `hipcc --save-temps` leaves a GCN .s for isa_census.py. No main(), no device.
#include "prototype/attn_decode/attn_decode_kernels.h"
