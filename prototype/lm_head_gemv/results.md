# prototype/lm_head_gemv — first real HIP kernel

The tied logits matmul: `logits[v] = Σ_k h[k] · W[v][k]`, `W` = `embed_tokens.weight`
(bf16, `[248320, 1024]`, 508.6 MB), `h` = one normed final hidden row. Chosen first
because §01.7 makes it ~33 % of all decode bytes and because it is pure streaming with no
state to carry, so it measures the memory path without confounds.

    ./prototype/lm_head_gemv/build.sh
    .venv/bin/python prototype/lm_head_gemv/run.py --refresh    # builds input from HF, gates result

## Result (best of 20, `rows_per_block` 4, GPU clock as-is)

| metric | value |
|---|---|
| time | **2.105 ms** |
| implied bandwidth | **242.1 GB/s** = 102 % of the `bw_probe_hip` read number (237) |
| vs torch CPU bf16 (16 threads, same op) | 6.6 ms / 77.3 GB/s → **3.1× faster** |
| weight upload (pinned staging → GTT slab) | 9.4 ms (54.3 GB/s incl. first touch) |
| fused-argmax mode | 2.099 ms / 242.2 GB/s, logits never materialized |

Accuracy vs the pinned reference logits (§10.1 `logits` class: cos ≥ 0.999, meanrel ≤ 0.05):

    argmax 760 (ref 760) | cos 1.000000 | max|d| 0.0617 | meanrel 0.00141 | top5 5/5
    fused argmax 760 (24.8060) vs ref 760 (24.7500)

So the first kernel sits **on the roofline**, which is the answer to the question this
prototype existed to answer: the HIP path can pull what the memory system offers, and the
decode ceiling of ~151 tok/s (§07.4) is reachable in principle. What remains is the other
93 % of the bytes.

`rows_per_block` sweep (all bit-identical outputs, `ndiff 0`): 1 → 192.7 GB/s, 2 → 241.7,
4 → 242.1, 8 → 239.6, 16 → 239.0. Flat from 2 up; the default is 8.

What the fused argmax bought: **1 %** (2.099 vs 2.126 ms). §07.4 called a skinny argmax GEMV
"first-class", and on the GPU that turns out to be wrong in detail — writing 1 MB of logits
is nothing next to reading 485 MB. The real reasons to keep the fused form are that greedy
decoding never needs the logits (A-17), it avoids a 1 MB host copy per token, and it skips
the host-side argmax; not bandwidth. That correction belongs in §07.4 when the engine
wires this in.

## Two traps found on the way, both worth keeping

**An impossible number is a bug.** The first `--rows` sweep printed 415–710 GB/s, above
the 256 GB/s physical peak of LPDDR5x-8000. Cause: the kernel used the compile-time
`ROWS_PER_BLOCK` while the host launched with the runtime value, so `--rows 2` covered
only a quarter of the vocab rows — less work, so "faster". The fix passes `rpb` as an
argument, and `run.py` now compares each config's logits against the reference instead of
trusting the timer. Rule for every kernel here: **a bandwidth number above ~240 GB/s is
evidence of skipped work, not of a good kernel.**

**The bare dump names are decode-step values.** `data/refs/smoke/pre_final_hidden.npy`
and `input_ids.npy` have a single token because the recorder writes bare names in every
phase, so the decode step overwrote the prefill's; `logits_last.npy` however belongs to
`prefill_ids`. Norming the dumped `pre_final_hidden` therefore does *not* reproduce the
dumped logits (cos 0.25) — this looked exactly like a broken kernel until HF was run on the
real prompt. `run.py` now takes the input row from a live HF prefill (hook on the final
norm) and self-checks that it reproduces `logits_last` (cos 0.999972) before using it.

## Shape of the kernel, for the next one

One warp per vocab row; `h` staged once into 4 KiB of shared memory; 32 lanes × 8 bf16 =
256 elements per step so `H = 1024` is four steps of 16 B loads, each warp instruction
touching 512 contiguous bytes; fp32 accumulation with a 5-step shuffle reduction; one
`atomicMax` per row for the fused mode (float→order-preserving-uint, 20 bits of index,
since 248320 < 2²⁰).
