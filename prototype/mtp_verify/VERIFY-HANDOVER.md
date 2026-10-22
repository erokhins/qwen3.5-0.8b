# P7 / MTP — fourth funded increment: draft census declaration + batched 2-row verify body

## TOPIC B RETRY — final state: gate GREEN, nothing committed (worker, this session)

Topic A is landed and committed green (`fea8d9a`). Topic B is uncommitted in the tree and its admission is
now GREEN on `dec8` (8 steps): `VERIFY GATE PASS`, 164 PASS lines, 0 failures, 15 recorded-as-deferred byte
claims. No timing was taken or printed anywhere here, and no band, floor, pin, ledger line or limit moved.

```
export LC_ALL=C && .venv/bin/python prototype/mtp_verify/gate.py --work /tmp/mtp_verify_gate
# VERIFY GATE PASS   (self-locks the GPU; wall ~10 s; correctness only; no timing taken)
```

### Two defects were found, and they were two, not one

**(1) The shadow mixer was never seeded.** `take_snapshot()` filled the checkpoint side, `shadow_state()`
repointed the shadow at the shadow side, and `restore_snapshot()` — the copy between them — was called in
exactly one place, inside `restore_witness()`, at the end of the run. So every round ran its body on mixer
memory it had never written: round 0 on whatever `hipMalloc` handed back, later rounds on the previous round's
END state, i.e. two tokens downstream of where its first row sits. `verify_body` now restores before it
submits. That copy is also what a reject would perform, so it is `ARMS.md` arm C's cost, not new work dressed
up as a fix. Effect, measured: row 0 against the plain step went from **1−cos 0.18–0.42** (argmax differing)
to the form-gap figures below, and `operands_same=1` — row 0's staged inputs are the step's inputs byte for
byte.

**(2) The body wrote inside the live cache, and that was a separate defect.** With the mixer fixed, the
guard's live before/after diff still read `gdn=0B conv=0B kv=4237B`, all of it in row 0's own anchor lane,
first visible at layer 7's K. Row 0 re-stages a lane the step already wrote, and lane `q` is inside the live
prefix, so every later reader sees the difference. The output-neutrality admission was red on exactly this.

**Row 0's append cannot be suppressed inside the wide body. The lines that prove it:**

* `prototype/attn_decode/attn_decode_kernels.h:804-805` — `attn_prep`'s KV-head branch writes
  `a.kc_as[((size_t)kv * a.cap + slot_of(row_pos, a.cap, a.neg, 0)) * DH + d] = fb(out[d]);` and the matching
  `vc_as`, with **no test on `qrow`**, so every row of the wave appends at its own position.
* `engine/kernels_attn.cpp:629` — `attn_prep<<<dim3(N_Q + N_KV, rows), THREADS>>>(c);`, i.e. `gridDim.y =
  rows`. The only append suppressor anywhere is `stage_head`'s `blockIdx.y != 0` early-out, which belongs to
  the *fused* one-row plan that a wave cannot use at all.
* `engine/attn_fattn_kernels.h:1164-1167` — in the staged plan the core reads the cache itself: it holds
  "every lane this kernel reads — including a row's own new lane". So there is neither a per-row write mask
  nor a separate write base; read-live/write-shadow is not expressible in the shipped launcher.

Suppressing it would mean a new `CoreArgs` field read by the kernel every **prefill** wave runs — a shipping-
kernel change with its own device admission, which this increment does not do. The alternative built instead
is the cheap copy the decision asked to be priced first: **`verify_body` saves one KV lane per row before it
submits and writes those lanes back immediately after**, on the same stream, before anything else is
submitted. No reader can observe a live cache that differs from the one the step wrote, and the cost is 24,576
B each way per round (`lane_revert_bytes_per_round=24576`, 96 D2D copies of 512 B) instead of a prefix copy —
47 MB at 4,032 lanes. `copy_kv_lane` walks `kv_regions`, the same address set the guard compares, and refuses
if the covered byte count disagrees with `kv_lane_bytes`, so a layout change cannot silently restore a shifted
cache. Row 1's tail lane is reverted too, for the reason in the finding below.

**The poison had to move.** The guard poisons the tail lane to witness that row 1 really appended, but the
revert puts that lane back, so a report read after `verify_body` would describe the repair instead of the
write. `verify_body` now takes a `PoisonReport*` and fills it between the body and the revert. `poison_written=1`
with `poison_regions_unwritten=0/24` in both probe runs is the witness that row 1 wrote its own lane in every
(layer, K/V, head) region.

### The finding this session produced, which is not fixed: row 0 is not bit-independent of row 1's token

Restating D3 as its own claim — same anchor, different candidate, row 0 against row 0 — found a real
arithmetic path from row 1's token into row 0's logits:

* two runs with the **same** candidate are byte-identical at every anchor: the body is deterministic;
* candidates 25 and 777 move row 0 at **4 of 8** anchors (18, 20, 22, 24) by 1−cos 1.9e-05 … **1.19e-04**,
  `ndiff` up to 209k/248k, with **argmax SAME at all eight**;
* it is **not** the live cache: with the revert in place, the wrong-candidate probe is byte-identical to the
  plain run in `sampled.txt`, all 86 `STATEFP` rows, all 25 `STEP` lines and all 9 per-position `logits*.bin`
  — including in the guard-off configuration the timing arms run in, which is now its own gate section
  `(0b)`;
* it is **not** the batched head: `engine/lm_head_batched_kernels.h:14-19` documents (and Topic A gated) that
  per output row nothing is shared between rows except the `uint4` weight both rows multiply by, and each
  row's accumulator is its own independent FP chain.

So the path is inside the wide body's own per-layer arithmetic and is **unattributed**. Attributing it needs
per-layer hidden dumps for two candidates, which is a separate pass, not this increment. What follows for
this increment: the decision is adjudicated (argmax SAME at every anchor, every row) and the arithmetic is
judged by the imported bands, with byte identity recorded rather than deleted.

### What the rows are now judged against, and where those numbers come from

Byte identity between the 2-row body and the one-row step was never a band this tree bought: the 2-row body is
the staged wave form (`attn_prep` + `fused = 0` core) and the shipped decode step is the fused one-row form.
Per the approved construction, each row is adjudicated on the existing bands and its decision, imported from
`engine/gate.py` so this rig owns **zero** numeric constants:

* `CASCADE_BAND["row_1cos_max"] = 2.0e-3` — the tree's per-logits-row band for two legal spellings of one row
  (hand GEMV vs hipBLAS). Measured row 0: **2.55e-05 … 9.13e-05** (≈22× inside); row 1: 3.13e-05 … 8.23e-05.
* `BAND["meanrel_hard"] = 5e-2`. Measured: 6.47e-03 … 1.38e-02.
* argmax SAME at every anchor for both rows, plus the engine's own in-run canary `row0_eq_step=1` at all 8.
* output byte identity of the emitted run (`(0)` with the guard on, `(0b)` with it off).

The 15 `[DEFERRED-FAIL]` lines are those byte-identity claims, each printed with its metrics so a reader sees
the exact status of the stronger claim instead of a silence. Nothing was widened to get them: the only
constants the gate now names are two it imports.

### Green machinery, still green and now checked harder

`head_passes=1`, `d2h=1` every round, `checkpoint_bytes_per_round=20201472` (derived from
`engine/state_sizes.py`, not typed), `restored_bytes == rounds × checkpoint`, `restore_witness=identical`,
D1/D2/D4 at every anchor, `(F)` flag-off byte identity, and the new census checks: `lane_revert_bytes ==
rounds × rows × kv_bytes_above_prefix` (two independent spellings of the lane layout must agree, and a round
that reverted nothing cannot pass) and the guard's `guard_bytes` consistency.

### Cost surface the timing arms must bill (no timing taken here)

Per verify round the probe now performs, in addition to what `ARMS.md` already arms: one shadow seed of
20,201,472 B (arm C, pre-existing) **plus** the lane revert — 48 device-to-device copies of 512 B out and 48
back per round (24 KV regions × 2 rows, each way; 96 copies in total), i.e. 24,576 B each way.
`ARMS.md`'s bands (plan A ≤ 0.1505,
KILL > 0.379) are untouched and its BLOCKED line is the main agent's to move; the pre-registered sentence for
this line is: **if the lane revert lands at ~0.3 ms per round, `delta` crosses KILL and plan A dies on the
verify side.** A single gather/scatter kernel would fold the 96 copies into 2 launches if the arms say the
per-copy launch cost is what bites rather than the 49 KB of traffic.

### Trap hit and recorded, again

Holding `tools/gpu_lock.sh` and then running `prototype/mtp_verify/gate.py` **deadlocks it**: the gate
self-acquires (`grep -c gpu_lock prototype/mtp_verify/gate.py` → 4). The first attempt at the green run parked
silently for the full tool timeout and showed up as a stale holder (`STALE (holder pid dead)`), which is the
signature `AGENTS.md` already documents. The gate is a self-locker; run it without holding the lock, and hold
the lock only for direct `decode_run` invocations.

### Commands for the main agent's device pass

```bash
export LC_ALL=C
./engine/build.sh                                                     # both binaries
.venv/bin/python prototype/mtp_verify/gate.py --work /tmp/mtp_verify_gate   # self-locks; no lock held by caller
# row-0 candidate dependence, if it is to be re-witnessed (decode_run does NOT self-lock):
bash tools/gpu_lock.sh acquire mtp-verify-row0probe
for c in 25 777; do ./engine/build/decode_run --model models/Qwen3.5-0.8B \
  --ids-file /tmp/mtp_verify_gate/dec8_ids.csv --steps 8 --out /tmp/row0probe3/c${c}b \
  --state-fp --mtp-verify 1 --verify-candidate $c > /tmp/row0probe3/c${c}b/log.txt 2>&1; done
bash tools/gpu_lock.sh release mtp-verify-row0probe
```

---

## SUPERSEDED — the previous session's Topic B write-up (kept: it is the evidence base)

Status at the time of writing: **Topic A code-complete and CPU-witnessed; device witness handed over. Topic B
implemented behind a default-off flag, and RED on device — row 0 is not bit-exact, so per the brief the work
stops here for a decision.** Nothing committed. No timing taken or printed anywhere in this handover.

### What was implemented

* `engine/lm_head_batched_kernels.h` — `lm_head_gemv_batched<NROWS>`: one pass over the 485 MiB tied table
  serving N query rows, one independent accumulator per row, per-row trip order/butterfly/store copied from
  the shipped one-row kernel. Only `NROWS=2` is instantiated; the launcher refuses anything else, because
  `prototype/mtp_draft/DRAFT-RESULT.md` pre-registers "fund a staging pass, do not fund deeper k".
* `engine/kernels_lmhead.cpp`, `engine/kernels.h` — `lm_head_batched_xbf16(table, h_rows, logits_rows,
  vocab, nrows)` with the same grid/warp constraints as the shipped launcher, and `kLmHeadBatchedRows = 2`.
* `engine/mtp_verify.{h,cpp}` — the side pass: a `Snapshot` with **separate checkpoint and shadow buffers**
  (so "restore" cannot be a self-copy), `verify_body()` which builds a shadow `RequestState` sharing the
  target's KV at the anchor position, forces the one-row route (hand GEMM, sequential GDN scan) for its own
  body, runs `submit_wave_body(…, 2)`, prices both rows with **one** batched head pass and **one** D2H,
  argmaxes both with §03.4's tie rule, and restores; plus `restore_witness()`, which poisons the shadow side
  and requires every byte of both regions to come back.
* `engine/decode_run.cpp` — `--mtp-verify 0|1` / `QWEN_MTP_VERIFY`, `--verify-candidate ID|sampled`. Refused
  with `--session`, `--sizes-only`, `--time-stages`, `--time-waves`, with `--wave-rows < 2`, and
  `--verify-candidate` is refused without the verify flag. Per-round `VERIFY2 …` and a final `VERIFY2DONE`
  census. Writes `verify2_<step>.bin` (F32 `[2][vocab]`) when `--out` is given.
* `engine/build.sh` — `mtp_verify.cpp` added to the `decode_run` target.

### What device said: RED, and it was the implementation's fault, not the route's arithmetic

Command (self-locked, wall 2.7 s, `util` at acquire 1 %, correctness only, **no timing taken**):

```bash
export LC_ALL=C && .venv/bin/python prototype/mtp_verify/gate.py --work /tmp/mtp_verify_gate
```

1. **The side pass was not read-only as implemented.** With `--mtp-verify 1`, `sampled.txt` and the `STATEFP`
   rows differ from the plain run starting at the step *after* the first verify round (`18 6511 314
   25.7663746` plain vs `18 6511 314 25.8213291` probe: same tokens, different max logit).
2. **Row 0 was not bit-identical to the plain step.** dec8 anchors 17/18/20/24: `ndiff` ≈ 247k/248k, **1−cos
   0.18–0.42, meanrel 0.5–0.75**, top logit off by 2–8 units, argmax differs at 2 of 8 anchors
   (`row0_eq_step=0` at the first anchor).
3. **That gap was not the wave-vs-seq form difference** — control, no new code, same conditioning:
   `wave@16 vs seq@16: ndiff=224985 1-cos=1.555e-04 meanrel=1.861e-02 argmax=760/760 SAME`. The verify body
   was ~10³ worse, attributed at the time to "mid-stream shadow conditioning". The actual mechanism is the
   unseeded shadow above; the control is still the right size for the *form* difference.
4. **The rollback and census machinery was green**: `head_passes=1`, `d2h=1` every round,
   `checkpoint_bytes_per_round=20201472` equal to `state_sizes.gdn_state + conv_window` derived in Python,
   `restored_bytes` equal to it, `restore_witness=identical`, `kv_bytes_above_prefix=12288`, rounds == the
   number of `VERIFY2` lines parsed.
5. **The derived negatives partly fired**: D1, D2 and D4 passed at every anchor; **D3 failed**, which is the
   same root cause as (1) and (2), seen from the other side.
6. **The flag-off path was byte-identical**, and that is now a section of the gate rather than a claim: `(F)`
   re-runs the fixture with `--mtp-verify 0` and with `QWEN_MTP_VERIFY=0` and compares every `logits<step>.bin`
   plus `sampled.txt` byte for byte against the run that never mentions the flag — 9 logits files +
   `sampled.txt`, all identical, in both forms.

### One arithmetic finding for the docs

The per-round checkpoint is **19.2656 MiB** (GDN 18.000 MiB + conv window 1.266 MiB **F32**), derived from
`engine/state_sizes.py`, and the device line agrees: 20,201,472 B. `prototype/mtp_headbatch/MTP-IMPL.md` §1
prices 18.66 MiB (0.63 MiB conv), i.e. it predates §08.1's 2026-10-11 correction that the window is F32. So
§3's snapshot/restore line (0.1217 ms) should be re-derived at ~0.133 ms before the headline `delta` is quoted
again. That document was not edited; it is a doc fix for whoever owns `prototype/`.

### Files this increment has touched, all uncommitted

`engine/gate.py`, `engine/lm_head_batched_kernels.h`, `engine/kernels_lmhead.cpp`, `engine/kernels.h`,
`engine/mtp_verify.h`, `engine/mtp_verify.cpp`, `engine/decode_run.cpp`, `engine/build.sh`,
`prototype/mtp_accept/run_accept.sh`, `prototype/mtp_verify/gate.py`, `prototype/mtp_verify/rowdiff.py`,
`prototype/mtp_verify/ARMS.md`, this file.
