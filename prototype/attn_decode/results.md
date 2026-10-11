# attn_decode — §05 gated full-attention decode (k = 1)

**STATUS: work in progress. No claims yet.**

This commit carries the kernel (`attn_decode.hip.cpp`) and `build.sh` only. The gate
(`run.py`), the softmax floor probe (`floor_probe.py`), the negative-injection table, the
cache-length sweep and the timing numbers are still to come, so every number this directory
would otherwise quote is deliberately absent. Nothing here is validated yet: the kernel
compiles and links, and that is the only thing this commit asserts.

Read `prototype/proj_gemv/results.md` and `prototype/mlp_gemv/results.md` for what a finished
op directory looks like, and §05 for what the op has to compute.
