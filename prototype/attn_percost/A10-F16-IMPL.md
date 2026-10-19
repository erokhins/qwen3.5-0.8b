# A10 implementation spec — the f16 accumulator, and why it is an enabler rather than a win

Written after the rate measurement (`A10-RF-LADDER.md` §7) and before any code, because the arithmetic says
something about sequencing that is cheaper to find here than in a week of kernel work. Design decisions are
**numbered D-n** so a later reader can disagree with one without disagreeing with the file.

Inputs: `A10-REWRITE-BRIEF.md` §3 (warp mapping, settled), `A10-RF-LADDER.md` §1 (8 VGPR per fragment), §2
(register arithmetic), §7 (both atom widths measured, equal rate), `prototype/issue_rate_probe/results.md`
§2b/§6 (the atom, and the f16 chain-length bound), `engine/attn_fattn_kernels.h` (the kernel to change).

---

## D1 — the two atoms issue the same number of instructions per round, so the f16 change buys registers only

Counting per round per wave, one wave = 16 query rows × 256 output dims:

| | f32 accumulator (shipped) | f16 accumulator (llama's form) |
|---|---|---|
| output covered per `v_wmma` | 16 rows × 16 dims (4,096 MAC) | 16 rows × 32 dims **per pair of issues** (4,096 MAC per issue: `op_sel` 0 then 1, `mma.cuh:1556-1566`) |
| issues per round | `FATTN_VTILES` = 16 | 8 tiles × 2 issues = **16** |
| accumulator registers per lane | 16 tiles × 8 f32 = **128** | 8 tiles × 16 halves = **64** |
| rescale per round (the `out.t[t][l2] *= rescale` loop at `attn_fattn_kernels.h:~880`) | 128 `v_mul_f32` | 64 packed-half multiplies ⇒ **half the VALU of that loop** |
| drain of P | widen to f32 then narrow | stays 16-bit through the fold |

**The instruction count is the same (16 per round either way), so this is not a throughput change** — consistent
with §7 measuring the same ~34 clk per instruction for both. What changes is 64 registers per lane and half of
one rescale loop. Which leads to the part that matters for sequencing:

## D2 — on the current geometry, 64 freed registers buy **nothing**, because LDS already binds residency

Measured at the shipped placement (`evidence/wmma-latency.txt` cross-check rows, and
`prototype/attn_wmma/fattn_residency_read.py`):

| quantity | shipped `attn_fattn_wave_u8` | with an f16 accumulator |
|---|---|---|
| VGPR/lane | 185 | ~121 (−64) |
| block threads | 96 (3 warps) | 96 |
| register-file bound on blocks/CU | 65,536 ÷ (185 × 96) = 3.69 → **3** | 65,536 ÷ (121 × 96) = 5.6 → **5** |
| LDS bound on blocks/CU | 65,536 ÷ 20,480 = 3.2 → **3** | **3** (unchanged) |
| driver answer | `driver_blocks_per_cu = 3` ✓ | **3** — LDS-bound |

So the f16 accumulator does **not** raise residency on the current 3-warp/16-position geometry: LDS is already
the limit at 3 blocks/CU, exactly the conclusion `A10-RF-LADDER.md` §3 reached for the other geometry ("after
R1 the binding resource is registers, not LDS" — here it is the mirror image, and both are worth knowing).

**Consequence, stated bluntly: the f16 accumulator alone is not a performance change on today's kernel.** It
is the *precondition* for the two levers that are supposed to pay — R1 (8 waves × 16 rows = 128 rows/block,
which needs 1 block/CU and 256 VGPR/thread) and R2 (Q resident, which costs 128 VGPR/lane and only fits
because the accumulator dropped to 64). Fund it as part of the geometry work, not as a standalone experiment;
do not spend a device window discovering that D2 is true.

The same arithmetic applied to the other instantiated spellings, because it is the sentence that decides
whether this is worth a window on its own: `attn_fattn_wave_u8` measures 9 wavefronts/CU (3 blocks × 3 warps),
and `attn_fattn_wave_two_u8` at 221 VGPR is register-bound at 65,536 ÷ (221 × 96) = 3.09 → **3 blocks too**,
which is what its LDS already allows. So the f16 accumulator changes residency for **neither** shipping
spelling: every one of them is already sitting on the LDS limit at 3 blocks/CU. **There is no standalone device
claim in this file** — the format change buys registers that today's geometry cannot spend, which is precisely
why it belongs to the geometry work and not ahead of it.

## D3 — what has to change in the code, and which part is the hard part

Sites in `engine/attn_fattn_kernels.h` (line numbers as of tip `5ae0992`):

1. **`FattnO` (`:212`)** — `struct FattnO { fattn_CRegs t[FATTN_VTILES]; }` (16 × 8 f32 = 128/lane) gains an
   f16 sibling: `t[8]` of 16 halves = 64/lane. The `static_assert`s at `:321` and `:324` bound the transpose
   staging from the tile count and must be re-derived, not edited.
2. **The P·V leg (`:917-923`)** — one `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` per 16-dim tile becomes
   **two `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` per 32-dim tile**, `op_sel` 0 then 1, with **two different
   A fragments and the same P** (llama's exact shape, `mma.cuh:1556-1566`: `xA[0]`, then `xA[1]` against
   `xB[0]`). This is the hard part: the A operand is staged through `s_kv` with `fattn_v_addr(dim, lane)`, and
   the fragments must now be gathered for dim-halves `16τ` and `16τ+16` of the same output tile.
3. **The rescale loop (`:880`)** — `out.t[t][l2] *= rescale` over 128 floats becomes the same fold over packed
   halves. Note `rescale` is a float computed as `expf(m - mn)`; whether the fold is done in f32 per half or
   with `v_half_mul` is a measurable choice, not a detail (the probe's §6 bound says the accumulator's
   magnitude must stay near the addend's magnitude, and this loop is what keeps it there).
4. **The epilogue / drain** — `fattn_transpose_out` and the `part` publish read the accumulator's element
   layout; with 16 halves per lane per tile the address arithmetic differs and the coverage asserts (the ones
   the header note says are the only thing that caught a half-denominator bug once) must be re-derived for the
   new layout, not relaxed.
5. **Score leg** — untouched. Scores are produced f32 and converted to the P operand with
   `__float2half_rn` already (`:~868`), so P is unchanged.

## D4 — the knob, and what "inert" has to mean here

`--attn-fattn-pvf16 {0,1}`, default **0**, plumbed as one commit through the trio (`engine/decode_step.h`,
`engine/decode_run.cpp`, `engine/gate.py`) per the recorded one-commit rule, plus:

* a **behavioural canary** that the change is required to move and that a stale binary cannot fake — the
  accumulator type is not in `LaunchCounts::describe()`, so the canary is the note-block field
  (`vgpr_count` must drop by ~64 on the `1` side and not on the `0` side) read inside the same window, per the
  "assert a behavioural canary inside the window" rule in `AGENTS.md`;
* **inertness witness** = default side byte-identical to `HEAD` (a `gate.py` device run against the pinned
  dumps, not a claim), since this is the whole reason the knob exists;
* **gate population derived, not typed** (`for d in data/refs/*/; do [ -f $d/manifest.json ] …`), because three
  gate admissions in this repo have already claimed a fixture count off a hand-listed population that did not
  exist.

## D5 — what the gates have to adjudicate, and the falsifier

The user authorised the format change on 2026-10-19 with three conditions: numerics may move, no bit-identity
claim, gates re-run and not loosened. So:

* the §10.1 cos/meanrel aggregates and the argmax/decision-room checks run over the **derived** fixture
  population, both knob sides;
* **no new band and no widening.** If a gate goes red, the question is the one the standing rule asks — could a
  widened band still catch every injection in the geometry-derived must-catch set — and the answer is recorded
  beside it, not assumed;
* the risk the probe identified is the one to look for, and it is a *magnitude-sensitive* risk: an f16
  accumulator stops absorbing an add once the add falls below half a ULP of the running value
  (`issue_rate_probe/results.md` §6). Cosine will not see that (it is scale-invariant and smooths small
  drifts), so the read must include the rowsum/magnitude checks, which exist precisely because a
  half-denominator once passed every cosine in the tree;
* **falsifier for the f16 accumulator specifically:** if the §10.1 aggregates move materially on *any*
  derived fixture while the per-round rescale is in place, the format change is reported, not routed around —
  no per-fixture exception, no band widening, no `--ack-spill`-style override.

## D6 — sequencing, and what is still unknown

1. **Do not build this before `A11-attribution.md` lands.** D2 says the standalone win is ~0; the whole value
   is in R1+R2+R3, and whether *that* can clear the 6.5 TFLOP/s abort line is exactly what A11 is being asked.
   If A11 attributes the window to something R2/R3 do not touch, this file is shelved and the week is saved.
2. If it is funded: R3 (64-position stage) and R1 (8 waves) and D1's f16 accumulator land **together**, because
   the register arithmetic only closes with all three (acc 64 + Q 128 + scores 32 + ~20 ≤ 256 — see
   `A10-RF-LADDER.md` §2's measured 155 VGPR in isolation), and any pair of them is worse than the trio.
   Land the trio behind D4's knob, default inert, then run the one paired 32k window.
3. The measurement that prices the whole thing is a single paired window at 32,768 lanes against a pinned
   pre-change sha with the canary read first — the step-3 shape from the kickoff, unchanged.
