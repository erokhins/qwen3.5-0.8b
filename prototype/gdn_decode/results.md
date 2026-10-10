# prototype/gdn_decode — the state-carrying decode step (S04.2–S04.4)

Four functions, each gated before the next was trusted
(`.venv/bin/python prototype/gdn_decode/check.py`), driven entirely by reference-sourced
data: prior state from `decode0.*` dumps, new token's pre-conv raw from `decode1.*`, and
expected outputs from `decode1.*`. Nothing synthesized.

| gate | expectation | result |
|---|---|---|
| F1 `gates` (S04.3) | vs torch on identical fp32 inputs | g 1.95e-11, beta 4.5e-9 **ok** |
| F1 `gates` | vs pinned dump (can only be bounded — HF's `a_raw`/`b_raw` come from a bf16 matmul) | g 2.6e-4, beta 1.9e-3 **ok** |
| F2 `conv` (S04.2) | `q/k/v_post_conv`, new conv window | **bit-exact** (max\|Δ\| 0) |
| F3 `rec` (S04.4) | vs fp32 mirror of §04.4 | o 4.5e-8, state 2.9e-8 **ok** |
| F3 `rec` | vs pinned HF dump, after bf16 rounding (the dump is bf16-valued) | **0/2048 differ** |
| F4 `all` 18 layers | vs spec mirror, worst over layers | o 1.7e-7, state 9.9e-8 **ok** |
| F4 `all` 18 layers | vs HF dump | **within 1 bf16 ULP on every element of all 18 layers** |

## Performance

| case | time | effective bandwidth |
|---|---|---|
| one layer (16 heads, 1 MiB state) | 0.0101 ms | — |
| 18 layers, 18.0 MiB state (the real decode step) | **0.062 ms** | 918 GB/s **cache-resident** |
| same kernel, state tiled to 108 MiB (6× replicas) | 1.344 ms | **255 GB/s** = 108 % of the read probe |

So the kernel runs at the LPDDR roofline when the state is cold and ~4× that when it is
warm — which it is in a decode loop, because 18 MiB of state is touched every step and the
iGPU's cache (not reported by `hipGetDeviceProperties`, which says 2 MiB L2) holds it.
Either way the state traffic is **0.06–0.22 ms of the ~6.6 ms token budget**, so the
register-resident version of pass B (avoiding the third pass over `S`) is a real
optimization but not a needed one.

## Findings this kernel produced

1. **§04.2's conv formula had a stray term.** It read `silu(raw[t] + Σ_j C[:,j]⊙raw[t-(3-j)])`,
   which double-counts the current token; the sum alone is correct and reproduces
   `q/k/v_post_conv` **bit-exactly**. Fixed in the spec, with the measurement cited. This
   is the second real spec bug found by a per-op gate (the first was the §02 special-token
   list) — the gates are earning their keep.
2. **The dumps are bf16-valued, and that masqueraded as a 1.45e-3 arithmetic gap.** HF's
   recurrent/chunk delta-rule functions cast back to `initial_dtype` before returning, so
   `o_pre_norm` on disk is bf16 stored as fp32. Comparing an F32 kernel against it shows
   meanrel 1.45e-3 — which is almost exactly bf16's own quantization step, and I initially
   wrote it up as "HF's fused read accumulates differently". It was the dtype. Round to bf16
   and the kernel is **bit-exact except for 1-elem-in-2048 ties worth exactly 1 ULP**. Gate:
   `bf16_ulp_report` in `check.py` (≤1% of elements differ, none >1 ULP); diagnostic:
   `ulp_probe.py`. **Check the dtype of a dump before concluding anything from a residual the
   size of that dtype's epsilon.**
3. **The algebraically prettier form of `o` is numerically worse.** `o = γ(q̃ᵀS) + (q̃·k̃)Δv`
   (one pass over `S`, no re-read) measures 1.5e-3 against the reference because the two
   terms cancel, while accumulating from the stored `S` in the reference's order measures
   4.5e-8 — for free, since pass B re-reads `S` anyway. Recorded in §04.4. (Note the
   coincidence: the split form's error and the bf16-dtype artifact are both ~1.5e-3, so
   fixing one was necessary before the other made sense.)
4. **My own ">240 GB/s means skipped work" rule needed a clause.** Here the number was real
   and explained by cache residency; the way to tell is to tile the working set past any
   plausible cache and re-measure (row 3 of the table). Without that check it looked exactly
   like the `lm_head_gemv` grid bug.
5. **A shared-memory overrun that produced NaNs in one mode and silence in another**:
   `block_sum` read `red[0..31]` from an 8-entry array. 128 threads → 4 warps, so the bound
   must come from `blockDim.x/32`. Fixed, and it is the reason "no NaNs in the inputs" is
   now printed as part of the gate.
6. **`decode1.L00.s_in` is the *pre*-norm block input** — the projections take
   `rmsnorm(s_in)`, which `check.py` determines by trying both readings and keeping the one
   that reproduces the pinned `beta`, rather than assuming it.

## Shape of the kernel, for the next one

One block of 128 threads per value head; thread `j` owns column `j` of `S` end to end, so
every access is consecutive across the warp and the `r` reduction is thread-local. Pass A
reads `S` to form `r`; pass B re-reads `S`, writes `γS + k̃⊗Δv`, and accumulates `o` from
the stored value. `k̃`, `q̃` staged in shared (1 KiB each); two block reductions for the L2
norms. q/k/v arrive bf16-rounded (the reference casts before the recurrence) and everything
else is F32.
