# A6-G0 verdict: **LIVE** — 7 warps buys −10.82 % of the attention family, so the residency thesis holds and G1 is funded

Measured by the main agent, 2026-10-19 14:32–14:4xZ, one `gpu_lock.sh` hold, zero subagents, blocking,
`LC_ALL=C`. Harness `prototype/attn_percost/run_ab_a6g0.sh`, both sides built from the **same pinned sha**
(`9a4bbce`, base as-is; g0 = same sha + `a6_g0_warps7.py`), logs in `tmp/attn_percost/a6g0ab/`. Rule:
`prototype/attn_percost/A6-BET.md` — family ≥ 10 % funds G1, < 3 % falsifies the thesis.

| row | base | G0 (7 warps) | move | status |
|---|---|---|---|---|
| `attn` family, untraced (**the falsifier row**) | 137.5176 ms | 122.6404 | **−10.82 %** | **LIVE** (≥ 10 %) |
| `attention` phase, traced | 149.71 | 136.81 | −8.62 % | agrees in direction |
| wave `wall_ms` @4,096 | 459.5527 | 450.8880 | −1.89 % | inside the floor; **G1's gate, not G0's** |
| wave ms @16,384 | 3,276.96 | 3,150.48 | −3.86 % | printed as G1's ship gate |
| wave ms @32,768 | 10,545.10 | 9,833.97 | **−6.74 %** | printed as G1's ship gate |

**Pairing.** No inert side exists for a compile-time geometry change, so the pairing check is rep spread:
base 2.13 %, g0 1.39 % across reps (tolerance 3 %). **Canaries.** Banners read `warps=3 lds=32768
wf_per_cu=6` on base and `warps=7 lds=65536 wf_per_cu=7` on the variant, with `vgpr=178` unchanged on both —
read off the kernel that ran, in the window. **Numerics.** A warp count re-splits the round's lane partition and
therefore the fold order, so the existing bands govern and bit-identity was never claimed:
`attn_form_band.py` on 640 ids and on a ragged 1,111-id fixture reads **PASS on both sides** (a `STATEFP` match
would have been reported as vacuous, not as a pass).

## What this establishes, and what it does not

**Establishes:** resident wavefronts are the quantity this attention wave responds to. A one-axis change — warps
3 → 7, nothing else, VGPR unchanged, no spill, LDS 65,536 = exactly one block/CU — moves the family by
**10.82 %**, close to the −10.5 % that §31's log-elasticity of 0.72 predicted for 6 → 7 wf/CU. A4-L1 moved the
same family 1.67 % by re-associating work inside a wavefront. That contrast is the result: **attention is
stalled on how few wavefronts are resident, not on what each one computes.**

**Does not establish:** that G0 ships. Its own wave row (−1.89 %) is inside the paired floor, exactly as A5 §3.1
predicted (−3.2 %), which is why G0 was funded as a falsifier and not as a bet. The wide rows move more than the
narrow one (−3.86 % at 16k, −6.74 % at 32k), consistent with §46's quadratic term being residency-sensitive.

**Externality, reported because it is not benign.** Sum of `g0 − base` over every non-`attn` family:
**+6.80 ms @4,096, +74.72 ms @16,384, −1.03 ms @32,768**. At 16k the attention family improved by more than the
wave did, i.e. a 224-thread block made *neighbours* slower — most plausibly LSU/L1 pressure on co-resident
blocks. It decides nothing here (no pre-registered line, and the wave row is the gate), but G1's window must look
at it again: G1 keeps 96-thread blocks and only drops LDS, so it should not reproduce this, and if it does, that
is a finding about co-residency rather than about the roll.

## What G1 now has to clear

G1 keeps 3 warps/96 threads and buys residency the other way — `20,480 B` per block ⇒ 3 blocks/CU ⇒ **9 wf/CU**
(compile-side: 185 VGPR, spill 0, driver grants 3 blocks). A5 §3.2 predicts **−25.3 % family / −7.7 % wave**.
A6-BET's G1 bar applies unchanged: live ≤ 0.8947 × base family, and `wall_ms` @4,096, 16k and 32k are **ship
gates**; G1 additionally owes **byte-identical logits and `STATEFP`** against the shipped default, because unlike
G0 it is supposed to move staging addresses only.
