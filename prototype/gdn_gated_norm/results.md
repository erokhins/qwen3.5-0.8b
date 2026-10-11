# `gdn_gated_norm` — §04.5's gated RMSNorm, k = 1 (kernel + gate **compiled, never run on a device**)

Why this op exists as its own directory: the audit found that A-01's cast order for this norm is
asserted at **no bit level anywhere** — the only comparison in the repo was `run_isolated.py`'s
`cos ≥ 0.9999` band, and a cast-order difference cannot move that number (§10.2 says so for every
rounding-policy question in this model). That is the same failure mode as the rope bug, so this op
gets its own bit-exact gate before the engine calls it.

**Device status: never run.** `hipcc` compiles clean (`build.sh`, and `engine/kernels_gdn.cpp`
compiles against the same header). Every number below is from the **CPU** mirror gate, which needs
no device and is allowed under the GPU rule. The device half (`run.py --gate`) is in the hand-off
list in `engine/README.md` and has never been executed.

## 1. The finding: the pins put the affine in the activation dtype, and the oracle lands there too

`mirror_probe.py` compares BF16 **bit patterns** of six readings of §04.5 against
`decode{N}.L{ii}.gdn.gated_out`, using `gdn.o_pre_norm` and `in_proj_z.out` as inputs, over all
18 GDN layers × 8 decode steps of `dec8` — **144 cases, 294 912 elements**. Command and output:

```bash
.venv/bin/python prototype/gdn_gated_norm/mirror_probe.py dec8
```
```
dec8 — 144 cases, 294912 BF16 elements compared against gdn.gated_out
  POLICY (kernel): bf16(n) · bf16(w) rounds, then ·silu(z) rounds  ndiff       0 / 294912  worst case    0  BIT-EXACT
  A-01 literal, weight in raw F32 (not a dtype speccheck holds)    ndiff  105683 / 294912  worst case  892  refused by the pins
  single cast: f32 norm · w · silu(z), one cast at the end         ndiff  124108 / 294912  worst case  999  refused by the pins
  A-01 literal, weight pre-rounded to BF16: *bf16(w) in F32, one cast at the end ndiff   74860 / 294912  worst case  585  refused by the pins
  gate rounded: silu(z) goes through a BF16 slot first             ndiff   79900 / 294912  worst case  608  refused by the pins
  speccheck.specimpl.rmsnorm_gated, called as speccheck calls it (BF16 weight) ndiff       0 / 294912  worst case    0  BIT-EXACT
  weight-dtype ladder: bf16 weight 0  |  bf16-rounded-then-F32 74860  |  raw F32 105683
```

Both bit-exact rows compute this one policy:

```
n   = BF16( o · rsqrt(mean(o², 128) + 1e-6) )      F32 statistics, cast back — A-01 gets this far
a   = BF16( n · BF16(w_norm) )                     the affine is a BF16 tensor op: the weight
                                                   joins in the activation dtype, and the product
                                                   rounds there too
ô   = BF16( a · silu(z) )                          silu in F32 over the BF16 z (HF's z·σ(z) order)
```

Three consequences, none of them mine to decide:

1. **A-01's clause is wrong as written.** "normalize in F32 → cast the normalized value back →
   then applies `w_norm` and `silu(z)` in F32" puts the affine in F32; the dumps put it in BF16
   with the weight cast first (§04.5's "multiply by `w_norm`" says nothing about dtype, and is
   only right under a BF16 weight). §04.5/A-01 correction in place, naming the old claim, is a
   hand-off item (see §6 and the §04.5 bullet under "Open questions" in `engine/results.md`) — not
   done in this commit.
2. **`speccheck.specimpl.rmsnorm_gated` is bit-faithful: 0 of 294 912 differ.** An earlier revision
   of this file reported "105 683 (35.8 %) differ, and 0 of them ever failed a gate"; that figure
   was **this probe's own bug**, not the model's. The probe handed the oracle `w.float()` — the raw
   F32 checkpoint tensor (`operands.py` returns `get_tensor` verbatim) — a dtype combination
   `Weights` never produces, because `specimpl.py:70-86` casts every model weight to the
   activation dtype unless `keep_f32_scalars=True` (`specimpl.py:83`). The three figures the audit
   reconciles are one A-01 formula at three weight dtypes, over identical `o`, `z` and pins:

   | weight as handed to the oracle | ndiff / 294 912 |
   |---|---|
   | BF16 — what speccheck passes | **0** |
   | BF16-rounded, then widened to F32 — A-01's letter | 74 860 |
   | raw F32 checkpoint tensor — what the old probe passed | 105 683 |

   `fc444d0` retracted the claim in `bench/gate-audit.md` (G2 amendment); the probe now *asserts*
   its oracle row at ndiff 0 and asserts the ladder is ordered, so the retraction is checked on
   every run instead of remembered. One caveat the ladder exposes: bit-exactness is a property of
   the **call**, not of the function — `speccheck/run.py --keep-f32-scalars` (the A-02 A/B knob)
   leaves `linear_attn.norm.weight` in F32, and then the oracle's `n = w * n` promotes to F32 and
   lands in the 74 860 class. The engine keeps that weight F32 as well (§01.6, §3 below), so it
   must round it to the activation dtype before the multiply — which is what the kernel does.
3. The engine's `decode_step` calls the kernel, not the oracle, so the engine is on the
   pin-faithful path from day one.

## 2. The same result on the other fixtures

`mirror_probe.py` with no argument sweeps four fixtures, exit 0, all CPU-only:

| fixture | cases | elements | kernel's reading | oracle's reading |
|---|---|---|---|---|
| `dec8` decode (18 layers × 8 steps) | 144 | 294 912 | **ndiff 0 — bit-exact** | **ndiff 0** |
| `prefix24dec8` decode | 144 | 294 912 | **ndiff 0** | **ndiff 0** |
| `smoke` decode (2 steps) | 36 | 73 728 | **ndiff 0** | **ndiff 0** |
| `p300` prefill, 277 positions × 18 layers | 79 776 rows | 10 211 328 | **ndiff 0** | **ndiff 0** |

The probe asserts that the bit-exact set is *exactly* {kernel, oracle-called-as-called} and that
the weight-dtype ladder stays ordered, so a second reading tying is a failure, not a footnote.

Two notes on this table, both honest-record corrections rather than findings:

* The `p300` row could not be re-derived until now: `variants()` reshaped its input to `(16, 128)`,
  which is one token, and a prefill pin holds 277 (`RuntimeError: shape '[16,128]' is invalid for
  input of size 567296`). §04.5 reduces per head over 128 lanes, so the row shape is `[-1, 128]`
  for any token count; `cases()` now asserts the pin is a whole number of `16×128` tokens.
* A prefill-position case for the `dec8` fixture itself (18 cases / 36 864 elements, measured by
  the earlier probe) is **not** in `FIXTURES` and is not re-derived here: `data/refs/dec8/` holds
  only `decode*`-prefixed pins, and the bare names that would carry prefill live in
  `prefix24dec8`/`smoke` — where §10.5's last-writer-wins trap makes bare names the wrong handle
  to reach for across phases anyway.

The `p300` row matters most: prefill positions exercise 16 heads × 277 positions per layer, so a
per-head-vs-whole-row reduction error would show there even if a decode case hid it.

## 3. What the kernel is

One block per value head (16), one thread per value dim (128): §04.5 normalizes **per head over 128
lanes**, not over the flattened 2048, and getting that wrong is silent (16 scales instead of one).
`o` is taken as F32 because that is what `prototype/gdn_decode`'s recurrence emits, and the kernel
rounds it to the activation dtype first — HF's fused delta-rule read casts `o` back to bf16 before
the norm (gdn_decode `results.md` finding 2), so the cast is part of the reference. On the gate's
own inputs (pins that are already BF16-valued) that cast is invisible, which is why there is no
negative for it; the engine depends on it.

`z` is taken as BF16 (it is `in_proj_z`'s rows, already an activation) and `w_norm` as F32 (§01.6:
one of the 36 F32 tensors, `[128]`, shared by all 16 heads) — so the engine needs **no dtype
adapter** to call this one.

## 4. Negatives (six, declared, device half not yet run)

`norm_pre_gate` is the dumped `BF16(n)` stage, which is what makes four of the six locatable
rather than decorative: an epilogue bug must move `gated_out` and leave the stage alone.

| `--neg` | bug | must move | must NOT move |
|---|---|---|---|
| `w_f32` | the A-01 literal reading: weight not cast to the activation dtype | `gated_out` | `norm_pre_gate` |
| `affine_uncast` | `n·w` stays F32 instead of rounding | `gated_out` | `norm_pre_gate` |
| `single_cast` | nothing rounds until the end (the tempting kernel) | `gated_out` | `norm_pre_gate` |
| `silu_rounded` | `silu(z)` goes through a BF16 slot before multiplying | `gated_out` | `norm_pre_gate` |
| `bf16acc` | the squared-sum accumulator is BF16 end to end | both | — |
| `noeps` | eps dropped from the rsqrt | both | — |

The wide-reduction mistake (normalising over 2048 instead of 128) cannot be injected in this block
shape — one block holds one head — so that falsification lives on the Python side: `cases()` feeds
`[16][128]` and a whole-row reduction cannot reproduce a per-head result (the `p300` row of §2 is
the evidence).

## 5. Band for the device gate

`gated_out` will be compared at **max 1 BF16 ULP with ≤ 1 % of elements differing** — the band
`prototype/gdn_decode/check.py` already uses for exactly this op class (an F32 norm whose pins are
BF16-valued; reordering the mean-of-squares is legal and flips a few elements by 1 ULP —
`prototype/glue/floor_probe.py` measures 3–4 of 393 216). `run.py --gate` prints the run's actual
ndiff / max ULP / tightest band through `prototype/bf16_ulp.py` (whose selftest it runs first), so
the band can be tightened the day it lands at 0 rather than being loosened.

## 6. Not done

* Device half never executed (`run.py --gate`, `--gate --time`): commands in `engine/README.md`.
* `results.md` here carries no timing row for the same reason.
* §04.5/A-01 correction in `specification/` is listed as a hand-off item, not done in this commit.
