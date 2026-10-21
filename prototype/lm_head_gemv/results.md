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

## Per-element rule on the logits — gate-audit G1, closed 2026-10-12

Until today this gate asserted three aggregates over 248 320 values — argmax equality,
`cos >= 0.999`, `meanrel <= 5 %` — and printed top-5 and `max|d|` without asserting them. That is
the hole `progress/2026-10-11/gate-audit.md` G1 names: the two wrong kernels it lists walk straight through. The
gate now compares the logits element by element and asserts on it. From a `--iters 1` device run:

    argmax 760 (ref 760) | cos 1.000000 | max|d| 0.0617 | meanrel 0.00141 | top5 match 5/5
    per element: ndiff 57/248320 (0.023 %) | >1 ULP 2 | n_bad 0 | worst bad 0 ULP
                 | tightest band this would pass at 9.01e-07

**The width is measured, not typed in.** `floor_probe.py` recomputes the same 1024 products in eight
legal addition orders — one of them `lm_head_gemv_full`'s own warp pattern, emulated from the kernel
source (lane *l* takes uint4 chunks *l*, *l*+32, *l*+64, *l*+96, eight sequential `acc +=` inside
each, then the `__shfl_down_sync` butterfly at 16, 8, 4, 2, 1) — and scores each against the pin with
the shared `prototype/bf16_ulp.py`. Over all 248 320 rows, at `ulp_max 1` and `cancel_rel 1e-4`:
every legal order lands at **n_bad 0**, with ndiff 50–73 elements (0.020–0.029 % of the vocabulary)
and max ULP up to 12 — all of that ULP inside the cancellation band, the widest band any legal order
would pass at being **1.01e-05**. So `1e-4` carries ten times the measured need and the 0.10 % ndiff
limit 3.4 times. That max-ULP figure is the whole point: it is §10.1 F1–F4 in the flesh — a chained
dot reassociated is entitled to move cancelling elements many ULPs — which is why the rule is
`n_bad == 0` (nothing large moved) and not `max ULP <= 1`.

Four things are asserted, because they fail differently:

* **ULP distances outside a cancellation band** — a value moved where a ULP means something;
* **an ndiff percentage** — a vector sitting 1 ULP off *everywhere* has `n_bad 0` and would sail
  through the first check; 0.10 % of the vocabulary is the limit, against a 0.029 % legal floor;
* **zeros and NaNs, named separately** — `bf16_ulp.py`'s own docstring: a collapsed output and a
  merely wrong one can share a ULP count;
* **the top-5 set**, which is what the engine actually spends. Every legal order measured keeps the
  argmax and top-5 5/5 on this pin, so the assertion is earned rather than hopeful.

`run.py --selftest` proves the rule bites — CPU-only, no device, ~1.6 s — and declares what each case
must and must not do:

- **clean**, the kernel's own order recomputed on the CPU: passes (ndiff 0.023 %, n_bad 0). The
  device lands on the same 57 elements, which is the emulation checking itself.
- **row 623 swapped with row 332**, both non-argmax: **FAILS** — 2 elements off by 2 ULP at
  `|ref|/max = 6.82e-01`, i.e. outside the band. The old gate read it as `argmax =`, cos 1.000000,
  meanrel 0.00141, top5 5/5: the audit's row-misplacement kernel, now caught.
- **last term of every dot missing**: **FAILS** — n_bad 212 554, ndiff 95.2 %, one logit collapsed to
  zero. And the aggregates the gate used to stand on read `argmax =`, cos 0.999420, meanrel 0.03614,
  top5 5/5, i.e. **inside** the 5 % band: the audit's dropped-term kernel, now caught too.

What the rule admits: any addition order of the same products; up to 0.10 % of the vocabulary
differing by any amount; any number of elements differing by 1 ULP. What it refuses: two logits
exchanging values, and a term missing from the reduction. What it does not speak to: the reference
rounds its logits to BF16 and `lm_head_full` keeps F32 — that distance belongs to
`engine/logits_floor.py`, so the rule rounds the candidate through BF16 before counting ULPs and the
two questions stay separate.

Derivation re-runs in ~7 s, CPU-only:
`.venv/bin/python prototype/lm_head_gemv/floor_probe.py --rows 0 --chunk 8192`, and the discrimination
check is `.venv/bin/python prototype/lm_head_gemv/run.py --selftest`. One limitation to keep: the
floor was measured on the one input row this fixture pins. A second prompt would say whether the
cancellation profile moves with the input; if it did, `PER_ELEMENT_NDIFF_PCT_MAX` is the constant to
re-measure — and these are prototype-gate constants, not `engine/gate.py` `BAND` keys, so
`tools/band_history.py` does not police them and the derivation above is the record.
