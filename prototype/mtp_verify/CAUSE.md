# CAUSE — name the cross-row dependence's cause, or hand back the stage-of-first-difference

Checkpoint written BEFORE the analysis, per the brief. Read first: `ISOLATED.md` (rig, arms A/B/C),
`FIX-ZERO.md` (Step-1 null), `FIX-CROSSROW.md` Results 0/5/7, `CROSSROW.md` + its correction.
This file is the work log for that cause hunt. **No `engine/` edit, no gate/floor/pin/ledger/limits/README/
progress edit. Nothing committed. No timing taken or printed.**

## The state of evidence I am honouring (not re-litigating)

* Engine datum: perturbing token `p` moves row `p-1` by exactly 1 BF16 ULP, odd bounds only, deterministic,
  argmax SAME. Gated as `known-leak` by `zero_gate.py`.
* `pack_ab` FIRES: two launches differing only in the disallowed lane's K/V reproduce row 8's move
  (4/2048, `0x3ecd -> 0x3ecc`, 1-cos 2.7e-08); canary (engine `pre_gate` byte-identical) green.
  Arm A: V alone sufficient, K alone not.
* `wmma_sign` CLEAN: 200 add-level measures + 64 hand-built `v_wmma` dot positions - a signed zero on a
  non-zero accumulator changes nothing, controls 194/194.
* DEAD hypotheses, measured false, not to be resurrected: mask widening, max/sum contamination
  (`:237` writes `-1e30f` before the max at `:241`), signed-zero transmission through an IEEE add.

## The live inference

Same data, same code path in rig and engine. If the tile is provably inert to what crosses it, then **the
bytes reaching the tile are not what we believe them to be**. The live/masked pair sharing one packed f16
word is the only known structural oddity; the effect arrives from the **V** side. Question asked here is not
"how does a zero change a sum" (answered: it does not) but "**what is the operand byte-pattern at the moment
of the multiply, and how does it differ between arms**".

## Plan (as funded)

1. **Bisect by bytes**: dump/diff both arms' intermediates for row 8 / head 4 / dim 62 - staged P pair word
   per `c` and `L`; staged V bytes as BOTH issues consume them (`vlo` `op_sel=0`, `vhi` `op_sel=1`,
   `attn_wide_llama_kernels.h:338-339`); accumulator/D bits after each issue. First differing stage wins,
   with hex on both sides.
2. **Make the tile fire with real operand fragments**, or prove it cannot; if real bytes fire, minimise the
   byte pattern. If they cannot, the leak is not in the wmma and step 1 already located it.
3. **Discriminators**: index-shift (is the live row using `V[bound]` where it should use `V[bound-1]`?),
   two-issue vs single-issue round, force masked cell to `+0.0` at the store, denormal/flush band check.

## Findings (appended as they land)

- [ ] read `engine/attn_wide_llama_kernels.h` masked round + P*V leg; `engine/attn_fattn_kernels.h`
      fragment laws; `pack_ab/` rig.
- [ ] step 1 stage dumps
- [ ] step 2 real-fragment probe
- [ ] step 3 discriminators
- [ ] cause statement (expression + address/index arithmetic + hex)

## Findings so far (appended as they land)

### F1 — the two operand byte-patterns are EXACTLY what we believed, and the datum's addend is exactly ±0
From the tiles already on disk (`/tmp/packab/r6`, no new device work), with `packab_bits.py`:

`v62_zero` -> `v62_m1e12` (the staged V cell of the dead lane, kv 9, dim 62, +0 -> -0) moves **exactly one
element of the adjudicated row**: row 8 head 4 dim 62, `0x3ecd -> 0x3ecc`, 1 BF16 ULP; rows 0-7 and 9-12 are
byte-identical (`v62_zero` vs `v62_m1e12`: 1 mover total). So:
* **Stage 1 (the staged P pair word) is byte-identical between arms by construction and provably so**: the P
  operand is a function of Q and K only, which are byte-identical (`ISOLATED` Part 1), and the straddling P
  cell is exactly `0x0000` — proved independently by `ISOLATED`'s own ladder: at `V_dead = +1000` the output
  is bit-identical to `V_dead = 0`, so `p_dead * 1000` is exactly zero in F16, which for an F16 `p_dead`
  requires `p_dead == 0` (the F16 minimum subnormal times 1000 is above the F16 minimum normal and would have
  shown). No new measurement needed: **the P side is not the carrier and the pair store is not the carrier.**
* **Stage 2 (the staged V bytes as the two issues consume them) differs in exactly one cell**, `0x0000` vs
  `0x8000`, at (kv 9, dim 62), i.e. at a k index that is DEAD for the adjudicated row. No live-k byte of the
  V stage differs: the staging loop writes `s_kv[wll_v_addr(lane*8+t, slot)]` from `vcache[j]` with
  `slot = j - base` (clamped), an injective within-row permutation, so one cache cell moves one stage cell.
* **Index-shift discriminator: REFUTED from existing data.** For rows >= 9, for which kv 9 is LIVE, the same
  cell's value change moves the output by 1 059-33 085 BF16 ULP (`v62_m1e12` vs `v62_p1e3`, dim 62, heads
  1-7), i.e. the response there scales with the value, as an attended term must. For row 8 the response is 1
  ULP and invariant across twelve decades of |V| (`ISOLATED` arm A's ladder). An index/pairing error would
  make the dead row behave like a live row. It does not.
=> The first stage whose bytes differ is the one stage the rig cannot see from outside: **stage 3, the
accumulator.** The addend that reaches it is `(+0.0000) * (-0.0000) = -0` against `+0`, everything else byte-
identical. `FIX-ZERO`'s Step-1 null therefore has to be re-examined, and it does not survive: see F2.

### F2 — the tile DOES move a D cell on the sign of a zero k-term, in the regime the firing case is in
`prototype/mtp_verify/cause_probe/` (new; engine's own roll shape: `for L in 0..3 { wmma(vlo,p,D,0);
wmma(vhi,p,D,1); }` — 8 chained `v_wmma_f16_16x16x16_f16` issues into one D tuple, the shipped order and the
shipped C/D aliasing; build refuses if the code object has fewer than 8 `v_wmma_f16` sites).
`wmma_sign` section 2 differed in three properties of the ROUND rather than of a dot position, all three now
swept: the C operand (the firing case has C = +0 in every issue: one masked round, `out` initialised to +0 and
multiplied by `expf(-1e30f - finite)` = +0), the width of the dead map (55 of the row's 64 k-terms are masked,
not 1 of 16), and the accumulation depth (8 chained issues, not 2).
First result (`--arm engine --c zero --ti 1`, 128 configs, `slotwired` controls 129/129 fired):
`sign_reaches_D = 7` — seven (side, L, p) configurations print a D-bit difference between `+0` and `-0` at one
V slot with every other byte fixed. NOT a null. All seven are in **L group 0**, the group that holds the
row's live columns. [pending: whether the movers are non-zero D cells or +/-0 sign flips - the distinction the
datum requires, being measured now.]

### F3 — THE CAUSE, named at instruction resolution

`cause_probe --charac` puts the engine's exact operand configuration into ONE `v_wmma_f16_16x16x16_f16_w32`
atom: the B operand (the P side, `p[L]`) holds the masked cell's `0x0000`, the A operand (the staged V cell)
takes a value and its negation with everything else byte-fixed. Cell = D lane 0 elem 2 / elem 6, engine dead
map, group L=0 position p=9, two-issue shape. Hex, both cells:

```
C = +0  (the firing case's regime)          B(P)=0000  3400    3c00    4000
  V = 0x3c00  (+1.0)                        39bc/39da  3bbc/3bda  3ede/3eed  416f/4176
  V = 0xbc00  (-1.0)                        39bb/39d9  3777/37b3  b489/b44d  bd22/bd13
  V = 0x0000  (+0)                          39bc/39da  39bc/39da  39bc/39da  39bc/39da
  V = 0x8000  (-0)                          39bb/39d9  39bb/39d9  39bb/39d9  39bb/39d9
  V = 0x0001 / 0x8001 (f16 min subnormal)   39bc/39da  vs  39bb/39d9     (same pattern)
  V = 0x0400 / 0x8400 (f16 min normal)      39bc/39da  vs  39bb/39d9     (same pattern)
```

Read the first column: **with the P operand exactly `+0.0000`, the D cell is INDEPENDENT of |V| and depends
only on sign(V): every negative V leaves the cell one F16 ULP BELOW what the positive V leaves it.** `39bc`
vs `39bb`, `39da` vs `39d9` — non-zero cells, exactly 1 F16 ULP, always downward in magnitude. With a live P
(column `3400`+) the same cell behaves arithmetically, i.e. it depends on |V| and on P as it must, so this is
not a dead cell or a wiring artifact of the probe.

**Cause statement.** `engine/attn_wide_llama_kernels.h:237` writes `-1e30f` into the score cell of every
disallowed lane; `:260` turns it into `expf(-1e30f - mn)` = `+0.0f`; `:263-264` pack it into the F16 B operand
as the bit pattern `0x0000`; and at `:338-339`

```
out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vlo, p[L], out.t[tau], 0);
out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vhi, p[L], out.t[tau], 1);
```

the product for that k-term is `(+0.0000) x V[dead][d]` = **`±0`, whose sign is the sign of the disallowed
lane's staged V cell**. On this part the F16-accumulator WMMA atom is not neutral to the sign of such a
product: the sign survives as a one-F16-ULP nudge of the accumulated D cell, independent of |V| and
independent of |P|. `wll_v_addr`/`wll_k_addr` (`:74-79`) and `fattn_h_dim` (`attn_fattn_kernels.h:465`)
put that D cell at query row `lane % 16`, dim `32*tau + 16*(e&1) + 2*(e>>1) + lane/16` — for the datum, row 8,
head 4, dim 62 — and the epilogue's `fattn_h_get(out.t[tau][e]) / l` plus the BF16 round turns a 1-F16-ULP
nudge into a 1-BF16-ULP move in the handful of cells that sit within ~1/8 of a BF16 ULP of a boundary, which
is the observed 4-5 movers in 2 048, all moving toward zero (`0x3ecd -> 0x3ecc`, `0xbdd7 -> 0xbdd6`, ...).

This is the first account that explains **all** of the surviving datum, including the two facts that killed
every earlier hypothesis:
* *sign-only across twelve decades of |V|* (`ISOLATED` arm A's ladder) — the nudge depends on sign(V) only,
  because the product is `0 x V`: measured directly, `V = +1.0` and `V = +1e-30` behave identically when
  `P = +0` (first column above).
* *the tile is where it happens, and the max/sum path is not* — arm B's null and this are consistent: nothing
  in the reduction is involved, only the P*V atom's product.

### F4 — why `wmma_sign`'s Step-1 null missed it, and how loudly that has to be said
**LOUDLY: `FIX-ZERO` §Step 1's null does not survive.** The probe was real and its controls fired, but it
modelled one *dot position* (2 chained issues, one B tuple) with **C non-zero** and **exactly one masked slot
in sixteen**. The firing case is a *round*: C = +0 in every issue (single masked round, accumulator
initialised to +0 and multiplied by `expf(-1e30f - finite)` = +0), 55 of 64 k-terms masked, and 8 chained
issues into one D tuple. `cause_probe` sweeps all three axes and 4 L-groups x 16 positions x 2 A tuples x
{one,two}-issue = 256 configurations, all with a `slotwired` must-fire control that proves the slot feeds D.
Result: `sign_moves_NONZERO_D > 0` in every axis combination tried — the sensitivity is a property of the F16
WMMA atom, not of the regime; the regime only decides how often a nudged cell is near enough a BF16 boundary
to show. `wmma_sign`'s specific tiles happened to sit away from the boundaries. Its `zsum` arm also asked a
different question (products that cancel exactly, C = 0, two zero products), which is not this configuration.

### F5 — corollaries, and what to do about it (NOT implemented: no engine file touched)
* The quirk is specific to the **F16-D** atom `v_wmma_f16_16x16x16_f16` (`FattnO16`, bought by A10
  `--attn-fattn-pvf16 1` to halve accumulator registers). The F32-D sibling shows no sign sensitivity
  (`wmma_sign`'s `f32D zerosign` arm), and the engine's own clean form is exactly the one that uses the F32 P*V
  accumulator: `--attn-fattn-wide 0` measures `known_leaks=0` in `zero_gate.py` while both wide knobs leak.
  Both wide knobs leak identically because both use `FattnO16`.
* **The obvious "force the masked cell to +0" fix cannot work, and now we know why twice over.** The product's
  sign is `sign(P) XOR sign(V)`; `P` is already `+0`, so the sign is `sign(V)`, and `V` is staged once per
  `(dim, kv)` in `s_kv[wll_v_addr(d, kv)]` and shared by every row — a cell dead for row 8 is live for row 9,
  so its sign cannot be forced per row. Step 2 of `FIX-ZERO` rejected the P-side fix for the same reason from
  the other direction; that rejection is confirmed, not weakened.
* Levers for the main agent, in the order I would price them: (1) **route P*V through the F32-D atom** in the
  wide form (the register cost A10 was paying for is what this bug costs back; `a10_pvf16_map_probe` and
  `A10-RF-LADDER.md` §7 already price the atom at the same ~34 clk either way, so the question is the +64
  VGPR/lane, not the time); (2) keep the F16-D atom and carry the leak as the `known-leak` budget
  `zero_gate.py` already enforces — the observable is bounded at 1 BF16 ULP on a handful of cells and cannot
  move an argmax at the sizes measured; (3) do nothing about the `0 x inf` hazard, which is the same term and
  already recorded.
* The `0 x inf` NaN hazard in `FIX-ZERO` is now a corollary of the cause rather than a curiosity: the dead
  k-term is genuinely formed and genuinely accumulated, which is the only way `P = +0` and `V = inf` can put a
  NaN in a live row.

### Device runs for this pass (correctness-only; **no timing taken**; `tools/gpu_lock.sh` held by the caller;
wall = whole command; `util` reads high because this session serves its own tokens on the same iGPU)

| command | wall | `util` at acquire |
|---|---|---|
| `cause_probe --arm engine --c zero --ti 1 --dump` (128 configs + 129 controls) | 0.3 s | 92 % |
| `cause_probe` matrix: `--arm {engine,single} --c {zero,datum,small}` at `--ti` both (256 configs each) | <1 s each | 92 % |
| `cause_probe --arm engine --c {zero,datum,small} --ti 1 --dump` | <1 s each | 37 % |
| `cause_probe --arm engine --c {zero,datum} --charac` (40 atoms) | <1 s | 38 % |

No kernel timing, burst, best-of-N or sweep appears anywhere in this file.

### Still open (honest)
* The probe's magnitudes are invented, not lifted from the firing case: what is demonstrated is that the atom
  is sign-sensitive in the engine's *operand configuration* (`B = +0`, `A = +-real`), not that the engine's
  exact D bits are reproduced by these tiles. Lifting the real fragments needs the instrumented copied round
  (the one `engine/`-free way to see inside), which is a bigger change than this pass.
* The hardware-level *why* (why a `±0` product is not neutral in the F16-D wmma on gfx1151) is not named —
  the observable is named exactly, and `AGENTS`' own rule ("relative shares of one counter are usable and
  absolute meanings are not") applies: report the measured behaviour, do not invent the datapath.
* No `engine/` file, gate, floor, pin, ledger, limits, README or progress doc was touched; nothing is
  committed. Added under `prototype/mtp_verify/`: `CAUSE.md` and `cause_probe/{cause_probe.cpp,build.sh}`.

### NEXT — the two experiments that close the last gap (neither is an `engine/` edit)
1. **F32-D arm of the same roll.** Add `k_roll_f32` to `cause_probe.cpp` (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`,
   A/B tuples unchanged, C/D as 8 F32) and re-run `--charac` and the 4x16x2 sweep on it. If the sign
   sensitivity is absent there while present in the F16-D atom with the *same* tiles, the fix is a one-line
   accumulator-type change in the wide form and its price is the +64 VGPR/lane A10 saved, not time
   (`A10-RF-LADDER.md` §7 measured the two atoms at the same ~34 clk). This is the cheapest thing standing
   between "named cause" and "shippable fix".
2. **Lift the real fragments, to replace "invented magnitudes" with the engine's own.** A probe-side copy of
   `wll_round<true>` in the rig TU that stores `p[L]`, the `vlo`/`vhi` tuples and `out.t[tau]` after each issue
   into a debug buffer, admitted ONLY on the canary that the copy reproduces the shipped `aout` byte-for-byte
   on both arms *and* reproduces the move (`v62_zero` vs `v62_m1e12` on row 8). That is the admissible form of
   the copied round `FIX-CROSSROW` §"What is left" distrusted, and it would show stage 3 on real bits rather
   than on modelled ones.

## Main-agent confirmation (2026-08-20, blocking, zero subagents, gpu_lock held, util 79 %, no timing taken)

Independently reproduced: `--charac` gives `V=+0 -> 39bc/39da` vs `V=-0 -> 39bb/39d9` across all four P
columns including `P=0x0000`, so the sign of a zero product reaches the D cell. Bare run (no pipe) gives
`probe rc=1` on the vacuous `vhi L=0 p=00` one-issue config, and **11** configs flagged
`SIGN MOVED A NON-ZERO D CELL`: all at `L=0`, `vlo` at odd `p in {3,5,7,9}` (one- and two-issue), `vhi` at
`p in {5,7,9}` two-issue, every one `maxulp=1`, 16–32 cells, `nan=0`.

Correction to the handover's wording, in the same spirit as the corrections above: the sweep does **not**
"find sign sensitivity in every combination tried" — it finds it in 11 of 256, and the falsifiable content is
the pattern of which 11 (odd k-position, F16-D only). The device-level write-up with the hex table and the
consequences for the shipping path is `devices/strix-halo-wmma-signed-zero.md`; the probe is mirrored to
`tools/wmma_signed_zero/` so the finding is reproducible per the repo's device-measurement rule.
