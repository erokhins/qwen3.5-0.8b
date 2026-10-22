# The F16-accumulator WMMA atom is **not neutral to the sign of a zero product** — `gfx1151`

This is a behavioural finding, not a rate: nothing here is timed, and no timing could show it. It was found
while attributing a 1-BF16-ULP cross-row dependence in the shipping prefill attention path
(`prototype/mtp_verify/CAUSE.md` is the work log, `prototype/mtp_verify/zero_gate.py` is the standing gate).

## The finding, in one sentence

A dot issued as `v_wmma_f16_16x16x16_f16` on this part produces a **different accumulated F16 result
depending on the sign of a zero product in the k-range**, even when every operand that could carry
magnitude is byte-identical — the difference is exactly 1 F16 ULP, always downward, and independent of both
factors' magnitudes.

## Measured, main agent, blocking, zero subagents, `tools/gpu_lock.sh` held (util 79 % at acquire)

Probe: `tools/wmma_signed_zero/cause_probe.cpp` (`bash build.sh`, then `--charac`). One atom, `C = +0`,
`P = 0x0000` (exactly `+0.0`) and three non-zero `P` values, `V` swept over eight bit patterns. D-lane 0,
elements 2 and 6:

| `V` bits | meaning | `P=0000` | `P=3400` | `P=3c00` | `P=4000` |
|---|---|---|---|---|---|
| `0x0000` | `+0.0` | `39bc/39da` | `39bc/39da` | `39bc/39da` | `39bc/39da` |
| `0x8000` | `−0.0` | `39bb/39d9` | `39bb/39d9` | `39bb/39d9` | `39bb/39d9` |
| `0x0001` | `+subnormal` | `39bc/39da` | `39bc/39da` | `39bc/39da` | `39bc/39da` |
| `0x8001` | `−subnormal` | `39bb/39d9` | `39bb/39d9` | `39bb/39d9` | `39bb/39d9` |
| `0x0002`,`0x0004` | small + | `39bc/39da` | (same) | (same) | (same) |
| `0x8400` | small − | `39bb/39d9` | (same) | (same) | (same) |
| `0x3c00` | `+1.0` | `39bc/39da` | `3bbc/3bda` | `3ede/3eed` | `416f/4176` |
| `0xbc00` | `−1.0` | `39bb/39d9` | `3777/37b3` | `b489/b44d` | `bd22/bd13` |

Read the first two rows against the last two: **for any `P` whose product with `V` is zero, the answer still
moves by 1 ULP with the sign of `V`, and does not move at all with `V`'s magnitude** (`+subnormal`, `+1e-12`,
`+1000` all read the `+0` row). Non-zero products move it by thousands of ULP, as they should. This is the
engine datum — a non-zero D cell moved by exactly 1 BF16 ULP — reproduced at single-instruction resolution.

## The shape it needs (why three earlier probes missed it)

The full sweep is 256 configs (4 L-groups × 16 dot positions × 2 A-tuples × {1,2}-issue, `C = +0`) and
**11 of them fire**: all at `L=0`, at odd positions `p ∈ {3,5,7,9}` for `vlo` (both one- and two-issue) and
`p ∈ {5,7,9}` for `vhi` with two issues, every one `maxulp = 1` on 16–32 non-zero cells, `nan = 0`. So:

- **odd k-position**, which is the same parity fact the engine-level leak shows (a masked column landing in
  the odd half of a packed F16 pair);
- **F16 D only** — the F32-accumulator form shows no sensitivity anywhere in the same sweep, which is why
  the engine's own `--attn-fattn-wide 0` route measures `known_leaks=0` while both F16-D wide knobs leak 2;
- the earlier isolated probes were not wrong, they were **too tidy**: one dot position, a non-zero `C`, and
  one masked slot in sixteen, which is not a shape this quirk appears in. An add-level test
  (`x+(−0)` vs `x+(+0)`, both FMA forms, the packed `v_pk_mul_f16`/`v_pk_add_f16` form, 200 measures) is
  clean, so this is a property of the **matrix accumulation**, not of the add.

Do not generalise this into "matrix dots are unreliable": every non-zero product behaves exactly as specified,
and the effect is 1 ULP of the accumulated F16 value. It is a **non-associativity of a zero addend inside the
matrix unit**, and it is only observable when a row masks most of its k-range — which is what causal
attention does.

## Why it matters to this engine

`engine/attn_wide_llama_kernels.h` turns a masked cell into `−1e30f` → `expf(−1e30f − mn)` = `+0.0f` →
packed F16 half `0x0000`. The product for that k-term is then `(+0.0) × V[dead][d]` = **`±0`, whose sign is
the dead lane's V sign** — and this quirk means that sign reaches the live row's accumulator. Consequences:

- Row `q` depends, deterministically and by ≤1 BF16 ULP, on token `q+1` whenever `q+1` is odd. Argmax has
  never changed in any fixture measured; `CASCADE_BAND` is 2e-3 and the effect is 1−cos ≈ 2.7e−08, which is
  why no band ever saw it.
- **There is no cheap per-cell fix.** A product's sign is the XOR of its operands, so forcing the masked `P`
  cell to `+0.0` cannot help: `P` is *already* `0x0000`, and the sign arrives from `V`. The `V` tile is
  shared by all rows in the block, and a lane dead for row 8 is live for row 12, so `V` cannot be sign-cleaned
  per row either. The neutral accumulator is F32-D — that is a route with a cost, not a patch.
- Related hazard on the same path, witnessed separately: `0 × inf` forms **NaN in the live row**. Unreachable
  today (fp16 range census: activations peak 82.5 against f16's 65504), but it is the shape of the next bug in
  this family.

## Two probe defects found while confirming, both general

1. **My confirmation command read `rc=0` while the probe was refusing.** The probe *does* exit 1 on a failed
   must-fire control (`cause_probe.cpp:320`) and 4 on a configuration-census mismatch (`:319`) — but I ran it
   as `probe | tail -8; echo rc=$?`, so `$?` was `tail`'s status. The vacuous check was my pipeline, not the
   probe, and the honest reading of that first run was "something aborted and I cannot tell what from a pipe".
   Any harness that pipes a refusal-bearing program must take `PIPESTATUS` or write the status before the pipe.
2. One config (`vhi L=0 p=00`, one-issue) printed "no difference" while its own `slotwired` control did not
   fire, i.e. the null was **vacuous**. The probe labels it correctly; the mistake was in reading the sweep's
   headline as "sign sensitivity in every combination tried" — the number is **11 of 256**, and the
   falsifiable statement is about *which* 11.
