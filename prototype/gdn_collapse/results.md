# `gdn_collapse` — route A: §04.4's recurrence with the state resident in LDS (**works, is bit-identical on device, and is DEAD: 112.0 ms against a 63 ms win line and an 84 ms kill line**)

The op is `engine/gdn_collapse_kernels.h` (kernel) + `engine/kernels_gdn.cpp::gdn_rec_collapse`
(launcher) + `--gdn-scan collapse` through `engine/decode_step.cpp`. This directory holds the priced
geometry (`collapse_price.py`), the CPU mirror (`collapse_mirror.py`), and — added with the device
pass — the fast device gate (`gdn_collapse_check.hip.cpp`, `build.sh`, `run.py`).

Route A was funded by `progress/2026-10-14/p9-phase1-2026-10-14.md` D-12 with the lines written down first:
**`gdn_rec` ≤ 63 ms at 1024 lanes lives, > 84 ms dies.** This file is the device verdict on those
lines, and the answer is: **112.04 ms best-of-3 — dead**, by the route's own pre-registered rule and
not by anyone's taste. What it did buy is a measured reason why the whole "collapse the launches"
idea does not work on this recurrence, which is §5.

## 0. What happened, in order

1. The kernel as committed (`3d838ea`) **faulted**: `HSA_STATUS_ERROR_MEMORY_FAULT`, illegal access, in
   the first body, and no timing at all (`progress/2026-10-14/p9-route-a-2026-10-14.md` §5's hand-off was never
   runnable). Each attempt cost a 5-minute HSA hang-analysis through the engine.
2. A standalone one-layer device gate was written (`run.py`, seconds per case) so the repair could be
   iterated; it found **two** defects, both of which had to be fixed before the route could be judged
   (§2).
3. Bit identity then held everywhere it was checked: 8 shapes in the harness, and all ten fixtures
   end-to-end through `engine/gate.py` (§3).
4. The timing pass then killed the route: 112.0 ms against a 63 ms line, with a best-case end-to-end
   saving of 6.8 ms on a 322 ms wave (§4). The default stays `--gdn-scan seq`; `collapse` remains
   selectable because it is bit-identical and an A/B side is cheap to keep, but nothing should plan on
   these milliseconds.

## 1. The fast device gate, and why a second gate was needed

`engine/gate.py --gdn-scan collapse` against `--gdn-scan seq` is the authority on the shipped path and
is what §3's fixture table comes from. It is not a usable repair loop: it runs 24 layers × 1024 lanes
before it can tell you anything, and on a faulting kernel it costs a queue-timeout each time.

`prototype/gdn_collapse/run.py` builds and runs a **single GDN layer** in a standalone binary:

```
$ .venv/bin/python prototype/gdn_collapse/run.py
CASE one body of 512, fresh state             tokens= 512 body=512 state=zeros         decay=0.4    seed=7
     o bit-diff        0/1048576   state bit-diff        0/262144
     o unwritten seq=0 collapse=0   guard violations o=0 state=0 operands=0
     PASS
... (8 cases) ...
COLLAPSE CHECK PASS (8/8 cases)
```

It judges the claim route A actually makes, which is **bit identity, not closeness**: both forms run
over the same operands and the same starting state and `o` and the resulting state are compared as
`uint32`. A band here would be a way of not checking the claim. Three things the value comparison
alone cannot see, all of them checks and not notes (the `tools/mixed_bw` lesson — verify what a kernel
*touched*, not only what it computed):

* **every `o` element is written** — both output buffers are pre-filled with `-1.2345e37f`, a value
  neither form can produce, so a strip that silently owns nothing shows up as an unwritten element.
  This is how `NEG_RC_STRIP_SHIFT` localises: 786,432 of 1,048,576 elements (exactly 3/4) are never
  written, while the tensor's *shape* is perfectly plausible.
* **nothing outside `o`, the state and the operands is written** — every device buffer carries 512
  floats of guard band on each side, checked on read-back.
* **the two forms run on separate state and output buffers**, so a stray write in one cannot be
  laundered by the other.

`run.py` does **not** take the GPU lock (it is a correctness loop and the caller holds the lock — the
self-locking-gate-driver deadlock is an `AGENTS.md` trap), and it takes **no timings**.

Two fixture lessons live in the harness rather than in the kernel:

* `--decay` controls the magnitude of the decay log, and therefore the state's **memory**: a body of
  *T* tokens forgets its incoming state by about `exp(-T·decay)`. At the default 0.4 a 128-token body
  erases its incoming state inside itself, and the state at the end then carries no trace of what was
  staged — which is why the first run of the `NEG_RC_STATE_BF16` build printed `state bit-diff 0`
  while `o` moved 222k elements. `--decay 0.002` keeps the state long-lived (what a real GDN head
  does) and is the shape the state-staging negative must be run on.
* and the driver's own lesson: a parser that silently returns no cases turns the whole gate into a
  check that cannot fail. `run.py` now aborts unless every `CASE` line it printed was parsed, and
  unless the matrix ran its 8 cases. It bit immediately — the first version matched zero cases and
  reported PASS.

## 2. The two defects the first working version needed

**a) The staging and write-back loops used `tid` as a column index.** The block is 128 threads and the
strip is 32 columns wide, so `for (i) Sst[slot(i,tid)] = Sh[i*D + cbase + tid]` asked for 128 columns
of a 32-column strip. Two independent consequences:

* global: strips 1–3 read (and the write-back *wrote*) up to 96 floats past every head's state, which
  for the last head is past the end of the tensor. That is the illegal access, and the fault only
  appeared in the last head's blocks, which is why it looked like a first-body mystery.
* LDS: `slot(127,127) = 4160` in a 4,096-cell array — 65 cells past the end of `Sst`, into whatever
  the allocator laid out after it. On this device that does not fault, it corrupts neighbours
  silently (trap 5 in `progress/2026-10-14/HANDOFF-2026-10-14.md`).

Both loops now walk the 4,096 cells striped over the threads, which also happens to make the global
access one contiguous 128 B row per warp and the swizzled LDS write 32 distinct banks.

**b) `q̃` was written as `qv * (rsqrtf(...) * Q_SCALE)` where `rec_kernel` computes
`(qv * rsqrtf(...)) * Q_SCALE`.** Hoisting the constant out of the per-thread multiply is a different
product tree, and the device said so loudly: **93,522 of 131,072 `o` elements differed while the state
was bit-exact**, because only `q̃` carries the factor. The CPU mirror could not have seen this — it
mirrors the *formula*, and both sides of a mirror that shares a formula agree with each other. This is
the §4-caveat-3 risk ("a CPU mirror cannot see FMA contraction and the device intrinsics") arriving in
a slightly different dress: what a mirror cannot see is **the source text's association**, and route
A's identity claim is about the source text, not about the algebra.

Neither defect was visible in the kernel's own geometry documentation, and `collapse_price.py --isa`
(a compile-only census) passed on both.

## 3. Bit identity, on device

Harness, `run.py`, 8 cases (512- and 1024-token bodies, one body and many, fresh and carried state,
short and long memory), every case: `o bit-diff 0`, `state bit-diff 0`, no unwritten elements, no
guard violations.

End-to-end through `engine/gate.py`, both `--gdn-scan seq` and `--gdn-scan collapse`, then a byte
compare of every file the two runs dumped:

```
dec8        gate_seq=OK gate_collapse=OK files=88    diff_lines=0
dec64f      gate_seq=OK gate_collapse=OK files=312   diff_lines=0
dec64t      gate_seq=OK gate_collapse=OK files=342   diff_lines=0
dec300f     gate_seq=OK gate_collapse=OK files=1402  diff_lines=0
mt32f       gate_seq=OK gate_collapse=OK files=829   diff_lines=0
p128        gate_seq=OK gate_collapse=OK files=421   diff_lines=0
chunk64b    gate_seq=OK gate_collapse=OK files=485   diff_lines=0
wide520f    gate_seq=OK gate_collapse=OK files=29    diff_lines=0
wide640f    gate_seq=OK gate_collapse=OK files=2596  diff_lines=0
wide1024f   gate_seq=OK gate_collapse=OK files=4132  diff_lines=0
```

`wide1024f` is the one that matters: 1024 lanes with **every prefill row pinned**, 4,132 files, and
the two forms agree on every byte of every one. The route is bit-identical, and that part of D-12's
argument survived contact with the device completely.

## 4. The timing, and the verdict against the lines written before it

Main agent, foreground, zero subagents, `tools/gpu_lock.sh` held, `--time-waves` at 1024 lanes
(`progress/bench/evidence/q10/ids1024.txt`), `--gemm library`:

| form | `gdn_rec` best-of-3 | runs (ms) | launches | ms/launch |
|---|---|---|---|---|
| `seq` (shipped) | **125.36** | 125.36 / 180.05 / 127.09 | 18,432 | 0.0068 |
| `collapse` (route A) | **112.04** | 112.04 / 112.37 / 113.28 | 36 | 3.112 |

| | seq | collapse | |
|---|---|---|---|
| `attn` row | 108.03–140.14 ms | 107.32–109.72 ms | unchanged, as it should be |
| wave wall, burst | 322.83 ms | **316.07 ms** | −6.8 ms = −2.1 % |
| wave wall, stage | 325.26 ms | **314.45 ms** | −10.8 ms |

**112.04 ms > the 84 ms kill line, so the route is dead.** End to end it moves a 1024-lane prefill
from 322.8 ms to 316.1 ms — which on the ledger's ruler (378.5 ms at the shipped 1024-lane shape)
would be roughly 2,706 → 2,755 prompt tok/s, +49. That is not what the route was funded for, and it
is not paid for by the complexity. `--gdn-scan` default stays `seq`.

One extra measurement, because the loop looked like the obvious suspect and was not: putting
`#pragma unroll 8` on both `i` loops is bit-identical (one accumulator, `i` ascending — an unroll is
not a re-association) and changed `gdn_rec` from 112.8 ms to **113.4 ms**, i.e. nothing. The pragma
was reverted; the loop is not LDS-latency-bound at this occupancy.

## 5. What the arithmetic got wrong, since it was wrong by 10×

`collapse_price.py --geometry` priced the route bottom-up at **11.4 ms** (1,370 instructions per token
per useful wavefront → 1,370 cycles/token → 512 tokens per site → 318 µs/site × 36 sites), cross-checked
as a rate: 33.94 GFLOP / 11.4 ms = 2.96 TF/s, sitting on the fp32 vector row rather than above it. The
device returned **112 ms — 9.8× the bottom-up model**, and 2.8× even the pessimistic "at the engine's
best hand-GEMM rate" figure of 40 ms.

The failure is the same shape as the chunked scan's (priced ≥1.5×, measured 5.9× — trap 6 in the
handoff), and the reason is visible in the geometry §2 already recorded but did not price: the grid is
64 blocks over 40 CUs, so about **1.6 blocks per CU, one of whose four wavefronts does the recurrence**.
An instruction-count model assumes an instruction is issued per cycle per SIMD; at 1.6 *working*
wavefronts per CU there is nothing else in the machine to schedule while a lane waits on a dependent
load, so the cost is latency per token, not issue slots per token. Route A pays for that tax
three ways per token — nine barriers per token per strip (`block_sum` is four of them, and the norm
tree is redone once per strip, so 36 per token per head), the two serial `i` chains, and the
`q|k|v` re-read per strip — and gets 4× redundant work for the privilege of keeping the shipped
reduction width.

What it removed, removal was real and cheap: launches 18,432 → 36, and the state round trip 38.7 GB →
75 MB per wave. What that bought is the 13.3 ms of `gdn_rec` above. The inescapable reading is that
`gdn_rec` is **not launch-bound and was never going to be fixed by reshaping launches**: the shipped
form spends 6.8 µs per (token, layer) and the resident form spends 6.1 µs per (token, layer), and the
gap between them is exactly the dispatch the route removed. §04.4's scan has 2,048 independent chains,
40 % of this box's fp32 lanes, and any form that keeps the serial recurrence pays ~6 µs per (token,
layer) whatever it does with memory. Moving 123.8 ms therefore needs **fewer dependent operations**,
which is what the chunked route tried to buy and could not, or a wider form of the state update —
neither of which is a launch-shape question.

## 6. Negatives: four, each required to localise

Built as separate `-D` binaries by `run.py` (never a runtime branch, so no defect is reachable from the
shipping binary), and each checked against a control where one exists:

```
-DNEG_RC_STATE_BF16   o 2096908/2097152  state 262077/262144  unwritten 0        -> CAUGHT
                      control (body from zeros): o 0/1048576  state 0/262144     -> CLEAN
-DNEG_RC_STRIP_SHIFT  o 786432/1048576   state 196608/262144  unwritten 786432   -> CAUGHT
-DNEG_RC_NORM_HALF    o 1048576/1048576  state 262144/262144  unwritten 0        -> CAUGHT
-DNEG_RC_I_DESC       o 848205/2097152   state 56320/262144   unwritten 0        -> CAUGHT
```

The localisation, per defect, is what the defect *does* rather than what a hand-written list hopes:
`STRIP_SHIFT` leaves 3/4 of `o` unwritten and the state half-moved; `NORM_HALF` moves everything from
the first token; `I_DESC` moves a **subset** (848k of 2.1M, 56k of 262k) because §04.4's dot cancels
and only elements with decision room change sign-relevantly; `STATE_BF16` is invisible from a zero
start (rounding a staged zero is still zero) and is only caught on a body that carries a state — the
same derived-must-catch lesson already recorded for the CPU mirror.

## 7. Not claimed

* No timing here is a device measurement except §4's, which is main-agent, foreground, lock held,
  zero subagents, and quoted with its spread. The `seq` 180 ms outlier is kept in the table because
  a spread that includes a bad run is the honest form.
* The claim that 2,048 chains is the whole serial width of §04.4 (§04.7 aside) is `collapse_price.py`'s,
  and §5 is an interpretation of one measurement against that claim, not a proof.
* Nothing here says a *register*-resident recurrence (state in 128 VGPR per lane instead of LDS, which
  would remove the LDS round trip and the parked warps at the cost of occupancy) is or is not faster.
  It is a redesign, D-12's kill line closes the route, and it has not been priced. If anyone re-funds
  GDN work, the honest starting point is §5's "6 µs per (token, layer) either way", not this file's
  kernel.
* `--gdn-scan collapse` remains selectable and defaults to off. It is not failed-closed the way
  `chunk` is: `chunk` is refused because its numerics are wrong, `collapse` is merely not a win.
