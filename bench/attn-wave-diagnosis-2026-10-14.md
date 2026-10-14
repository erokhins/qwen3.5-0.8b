# Why the attention wave runs at 0.236 TFLOP/s — Phase 1 diagnosis (2026-10-14)

Task: explain **mechanically** why the `attn` family costs 108.80 ms of a 1024-token prefill wave
(9.067 ms per 512-row launch, 12 sites) at **0.236 TFLOP/s**, and say whether a wave-kernel rewrite
can take most of it. Phase 1 (diagnosis, this document) was done **without touching the GPU**: every
number is either quoted from the ledger and `devices/`, or is geometry read out of
`prototype/attn_decode/attn_decode_kernels.h` and arithmetic on it, reproducible with

    .venv/bin/python prototype/attn_decode/wave_model.py geometry
    .venv/bin/python prototype/attn_decode/wave_model.py roof
    .venv/bin/python prototype/attn_decode/wave_model.py isa        # hipcc -c, compile-only
    .venv/bin/python prototype/attn_decode/wave_model.py selftest   # 6 checks, all green

`isa` cross-compiles the header for `gfx1151` and reads the resources and loop census out of the
assembly. No timing was taken here and no number in this file came from a worker device run.

**Answer in one line: the kernel is not near roofline — it reads exactly 1.0 byte of K/V per flop
(2,050× the bytes it could not avoid), and at the box's measured 237 GB/s read ceiling that alone
prices the wave at 108.8 ms against 109.2 ms measured. The binding constraint is the memory
pattern — no re-use — not parallelism, not occupancy, not the causal triangle.**

---

## 0. What is being priced

From `bench/performance-status.md` (§"Wave-path attribution", route-(c) flip tables):

| | value | source |
|---|---|---|
| `attn` in the 1013.07 ms wave | 108.80 ms = **10.7 %**, 12 launches, 9.067 ms each | `--time-waves`, lock `waveattr2` |
| `attn` after the route-(c) flip | **109.2 ms** = 33.3 % of a 328.5 ms wave | lock `W9b` |
| matmul in the same wave | 74.4 % hand → 21.2 % library | same table |

Twelve sites, not twelve *kinds*: 6 full-attention layers × 2 bodies of a 512-row wave. The wave
never routes attention through hipBLAS — `engine/kernels_attn.cpp`'s three-kernel plan
(`attn_prep` → `attn_core` → `attn_combine`, in `prototype/attn_decode/attn_decode_kernels.h`, which
the engine `#include`s) is the only route — so the library flip left `attn` untouched and its
*share* tripled only because everything around it got cheap. This is the line item that is now the
largest non-matmul, non-GDN cost in prefill.

Device anchors used below, all measured elsewhere in this repo: 40 physical CUs
(`devices/strix-halo-compute.md`; **HIP reports 20, which is wrong for this part and is not used
anywhere here**), wave32, 2 issue slots and max 32 waves and 1024 threads per CU, 2.206 GHz,
DRAM read **237 GB/s** (918 GB/s for an 18 MiB resident set), hand GEMMs 0.85 TF/s, hipBLAS fp32
2.93 TF/s (what the vector path measurably sustains), hipBLAS 16-bit wave mix 18.84 TF/s at M=512.

## 1. The launch plan, and what each block does

`wave_rows=512`, `N_Q=8`, `N_KV=2`, `GQA=4`, `DH=256`, `THREADS=256` (= 8 warps), `TILE=8`,
`MAXSC=512`. Row `qrow` sits at position `pos0 + qrow` and may attend `lvalid + qrow` lanes, with
`lvalid = pos0 + 1` (`decode_step.cpp`).

| | body 0 | body 1 |
|---|---|---|
| rows, `pos0`, `lvalid` | 512, 0, 1 | 512, 512, 513 |
| lanes per row | 1 … 512 | 513 … 1024 |
| form (from `split_for(lanes_last)`) | **one-pass**, all 512 rows | **split, `split=2`**, all 512 rows |
| `attn_prep` / `attn_core` / `attn_combine` blocks | 5120 / **4096** / 0 | 5120 / **8192** / 4096 |
| touches (row × head × lane) | 1,050,624 (one-pass) | 3,147,776 (split) |
| serial lane-tile steps per block | mean 32.5, max 64 | mean 48.5, max 64 |
| flops / K+V bytes read | 1.08 GFLOP / 1.08 GB | 3.22 GFLOP / 3.22 GB |

Body 0 is entirely one-pass because its widest row is `512 = MAXSC`; body 1 is entirely split
because its narrowest row is `513`. So the two forms are not a fallback edge case — the wave pays
each of them on half its rows, and body 1 carries 3× the work of body 0.

* **One block = one (query row, query head)** pair (×2 at `split=2`), 8 warps, 256 threads, one
  thread per output dim. Body 0 launches 4096 core blocks and body 1 8192; across 6 layers the wave
  launches **73,728 `attn_core` blocks = 1,843 per CU**, or 159,744 blocks counting prep/combine.
* **Each block's serial inner loop** is `ceil(lanes / split / 8)` tile iterations, and each
  iteration contains **two whole-block `__syncthreads()`** plus a 5-step butterfly per scoring
  warp: 32.5 mean / 64 max iterations for body 0, 48.5 / 64 for body 1. Nothing inside an iteration
  depends on the previous one, but every warp waits at both barriers for the slowest one.
* **One "touch"** — one (row, head, lane) triple — is: read the lane's 512-byte K row and dot it
  against the head's `d_head=256` query row, then read the lane's 512-byte V row and accumulate one
  term of `P·V`. That is `4 × 256 = 1024` flops against **1024 bytes**, and it is true regardless of
  shapes: **exactly 1.000 byte per flop.**
* The wave therefore reads **25.79 GB** of K/V to produce **25.79 GFLOP** (the model reproduces the
  ledger's 25.79 GFLOP independently — that check is in `selftest`) out of only **12.58 MB** of
  distinct KV: every K and V line is re-read **2,050×** (once per row that may attend it × 4 heads
  of its KV group).

## 2. The four candidate constraints, each priced

| candidate | number | verdict |
|---|---|---|
| **Parallelism / load balance** | 73,728 `attn_core` blocks over 40 CUs = **1,843 per CU**; one body's 6,144 blocks against the 160 all the CUs could hold at once (4 blocks/CU × 40, if residency allows 4) | not binding — the supply of blocks is not the question, residency per CU is |
| **Occupancy** | 53 VGPR / 75 SGPR, **0 spills, 0 scratch** → 16 waves/SIMD by registers and 4 blocks/CU by threads (8 warps × 4 = 32 = the CU's wave cap). But the kernel descriptor reserves **38,944 B LDS/block** | see §3 |
| **Memory pattern** | 1.000 byte/flop → **25.79 GB** → **108.8 ms** at 237 GB/s vs **109.2 ms** measured (0.4 %) | **binding** |
| — same bytes, best-case cache | the distinct KV is only 2.1 MB per layer-body, and `mixed_bw` measured **918 GB/s** for an 18 MiB resident set: if every re-read hit cache the same traffic costs **28.1 ms** | the caveat that makes the next row matter |
| **Per-block serial arithmetic (issue)** | ~18.7 G warp-instructions/wave counted *statically* off the loop bodies → floor **≤106 ms** at 1 IPC/SIMD, plausibly ~36 ms if that count is 3× pessimistic | within 1–3× of the line; not separable on paper |

**The causal triangle is not the problem people expect.** The triangle is already *packed* into the
launch: block durations go as the row's lane count (1…512 in body 0, a 512:1 spread), but with 4096
blocks over ~160 concurrent slots there are no idle half-blocks — what is left is dispatch-order
imbalance, since `z = qrow` ascending launches the *shortest* rows first and leaves the 64-tile
blocks as the tail. Its residue is a few percent, not a factor of two. Body 1 is nearly uniform
(513…1024 lanes, and `split=2` flattens it further), so the imbalance is confined to body 0, which
is only a quarter of the touches.

**Registers are fine.** `d_head=256` per row does not spill: each thread holds `qreg[8]` plus one
`acc` and a handful of pointers, 53 VGPRs, zero spill, zero scratch — the ISA question the task
asked ("does a 256-wide row fit in registers?") has a clean yes. Any rewrite that holds 4 heads'
queries in registers pays ~+32 VGPRs and drops register-occupancy from 16 to ~8 waves/SIMD, which
is affordable but is a real cost to price before it is paid.

**The honest ambiguity, stated plainly.** Two walls land on the measured line: the byte wall
(108.8 ms at the DRAM read ceiling) and the instruction wall (≤106 ms at the static instruction
count, 1 IPC/SIMD). They cannot be separated on paper, and the discriminator is a counter pass, not
another model: `SQInstrExecuted` vs the 18.7 G/wave static upper bound, and DRAM read bytes vs the
25.79 GB/wave this file predicts if nothing is caught in cache. **It does not change the direction of
the fix** — tiling for re-use divides the bytes by `G·M` *and* divides the per-touch instruction
count (one K-row load feeding `G×M` dots, one softmax shared instead of eight redundant copies), so
both walls fall together. It does change the ceiling, which is why the ceiling is quoted as a band in
§4 rather than a point. Useful work is
`1024 flops/touch ÷ 64 flops-per-warp-FMA = 16` FMA warp-instructions per touch; the ISA runs
**~1,030 warp-instructions per touch** in the one-pass form, of which the score-tile loop is 190 and
the softmax+PV loop ~840. The reason is structural: pass 2 is three loops over *all* the row's lanes
that **every warp of the block executes redundantly** (`attn_decode_kernels.h`, `for (t < ns)` ×3 —
max, sum-of-exp, then `expf`/divide/round plus the V load and FMA), so each touch pays `expf` and a
divide **8 times**, once per warp. Cutting that redundancy is an instruction win, not a traffic
win, and it is available inside the same rewrite.

## 3. One open item that a paper argument cannot close: 38,944 B of LDS

`attn_core`'s code-object descriptor declares `.amdhsa_group_segment_fixed_size = 38,944` B/block
(`wave_model.py isa` prints it from the assembler output; `attn_prep` 2,080 B, `attn_combine` 0).
The source's own six `__shared__` arrays sum to **6,176 B** (`q_rot` 4 KB + `sq`/`sk`/`sv` 3 KB +
`sc[512]` 2 KB). So the compiler reserves **6.3× more group segment than the kernel addresses**, and
on an RDNA3-class CU with 64 KB of LDS that decides residency outright: `2 × 38,944 > 65,536`, so
**one block per CU = 8 of 32 wave slots = 25 % occupancy**, instead of the 4 blocks/CU the thread
count allows.

If that is real it is a large, cheap finding — a 1-line experiment worth more than a scheduling
tweak, and it would also change what a rewrite must budget LDS for. It cannot be settled without a
device: the two commands that do are in §5. Note the trap class from `AGENTS.md`: this is exactly
the kind of claim that a CPU-only geometry selftest would "pass" vacuously, so it is stated as open
and handed over, not asserted.

## 4. Verdict and ceiling

**Binding constraint: memory pattern — absent data re-use.** One byte of K/V per flop, 2,050×
re-read of a working set that is smaller than one GDN state, priced at the measured read ceiling to
within 0.4 %. The kernel is **not** near roofline; the roofline is 4–30× away:

* Fix traffic only (share each K/V read across the `GQA=4` query heads of a group): 6.45 GB →
  **27.2 ms** of traffic; with row tiling `M=8` on top, 806 MB → 3.4 ms.
* Once the bytes are under the arithmetic term (~`G·M = 12×` re-use), the wall becomes how fast the
  box does 16-bit MACs: **8.8 ms** at the fp32 vector rate (2.93 TF/s), **30.3 ms** at the engine's
  current hand-GEMM rate (0.85 TF/s), **1.4 ms** at the measured matrix rate (18.84 TF/s).
* So a **vector-path rewrite lands ~9–30 ms** (3.6–12×), and a matrix/hipBLAS routing **~1.4–5 ms**
  (up to ~30×). Against a 328.5 ms library-route wave that is 24–32 % of prefill recovered, and
  against the hand route 8–10 %.

The honest residual: at `1 byte/flop` the flop count and the byte count price the wave at the *same
number*, and the static instruction count lands within 1–3× of it too. Removing the traffic wall
alone therefore cannot take all 109 ms — it takes it down to whichever of (issue floor ~36–106 ms,
arithmetic 8.8 ms) the rewrite also addresses. A rewrite that tiles for re-use **and** stops running
the softmax loops redundantly in all 8 warps removes both at once; that is the shape worth funding.

## 5. Bit-identity, and what a rewrite must preserve (Phase 2 not started)

Phase 2 was **not** started, which is the sanctioned stopping point for this task: writing a tiled
kernel and a device gate that no worker may run would ship an unverified kernel and an unexercised
gate, and the gate rule in this repo is that a gate with no demonstrated failure mode is not a gate.
What follows is the specification Phase 2 must satisfy, plus the CPU-only reasoning that answers the
bit-identity question in advance.

**A tiling rewrite can be bit-identical — band-free — because sharing a load is not an arithmetic
change.** The reference form's per-element arithmetic (`attn_decode_kernels.h`, §05.5) is five
ordered reductions, and a G×M tiled kernel can keep every one of them verbatim:

1. score dot = 8 serial `dot += bf(k[t]) * qreg[t]` per thread, then a 5-step `__shfl_down_sync`
   butterfly, then `× SCALE` after one BF16 round of the raw dot;
2. `m` = serial `fmaxf` over `sc[0..ns-1]` in index order;
3. `l` = serial `+= expf(sc[t] - m)` in index order;
4. `wt = bf(fb(expf(sc[t] - m) / l))` — the BF16 round of the *normalised* weight **is** the
   reference form (HF rounds the softmax output to the activation dtype before `P·V`), so it must
   stay, and the epilogue must not divide again (hence the existing rule that a split block cannot
   re-merge a range-normalised row);
5. `acc += wt * v` serial in `t`.

Reuse changes *where a K row comes from* (one load feeding `G×M` dots, from a register or LDS
instead of four separate global loads); it does not reassociate any of 1–5. **So the answer to
"is Phase 2 bit-identical or band-bearing" is: the vector tiling is bit-identical, and the numbers
that prove it are the existing pins** — `prototype/attn_decode/run.py` already reports 0 differing
elements for the one-pass form against the pinned `attn_out_pre_gate`, and the claim to be tested is
that a tiled form reproduces the same bits. Band-bearing is *not* on the table here: the only
already-authorized band is §07.3's for the online/split form body 1 uses (25,745 of 98,304 elements
differ, pre-authorized), and a rewrite must not smuggle the one-pass route into that band. If Phase
2 instead routes `Q·Kᵀ` through hipBLAS, it is band-bearing by construction (a library's F32
accumulation order is not torch's), needs its own floor probe and a pre-registered band, and must
say so before it is written.

Budget constraint that sizes the tiling, and why `M` cannot be picked freely: the one-pass form
stages **all** `ns ≤ 512` scores of a row in LDS to keep pass 2 a two-pass form — 2 KB per
(head, row) at `MAXSC` — so a `G`-head × `M`-row block needs `G × M × 2 KB` of score staging on top
of a group segment the compiler is already inflating 6.3× (§3). `G=4` costs 8 KB of scores plus 32
VGPRs of extra `q` per thread (four heads' queries at 8 dims each) and divides traffic 4×; that is
the first piece to fund, and it is affordable even in the §3 worst case. `G=4, M=2` is 16 KB, still
possible; **`G=4, M=8` is 64 KB, which cannot coexist with the 38,944 B already reserved on a 64 KB
CU** — which is the arithmetic reason row tiling past `M=2` needs either the online form (band) or
shrinking the score staging itself (e.g. BF16 scores, 1 KB per head-row, at the cost of a rounding
decision that is *not* bit-neutral). This is the arithmetic the task asked to be shown rather than
assumed, and it is why "just tile it 16×16" is not free.

**Gate for Phase 2** — the harness already exists and the two controls the brief asked for are
already in it, so the rewrite inherits them instead of inventing new ones
(`prototype/attn_decode/run.py`, `NEG_KERNEL`, each with a declared must-catch / must-not-catch stage
set so no negative can pass by construction — the vacuous-negative trap in `AGENTS.md`):

* **wrong causal mask**: `readpast` (attend to `cap` lanes instead of `lvalid`, i.e. the mask is
  gone) and `dropcurrent` (lanes `< pos` only, so the appended token is forgotten, §05.7). Under a
  `G`-head × `M`-row tiled form these must still move exactly the rows that gained or lost a lane —
  and a row whose lane count straddles a tile boundary is the new failure mode the old geometry
  cannot express;
* **swapped head pointer**: `gqaoff` (query head `i` reads KV head `(i/4 + 1) mod 2`, §05.5). Only
  the heads whose `kappa` moves may change; if the others move too, the control localises nothing.
  In a block that holds several heads of one group, this is *the* control for "did the shared K/V
  load get wired to the right group", which is exactly the bug the rewrite can introduce;
* and the ones a tiled form could break that the current geometry cannot: a **tile-boundary
  off-by-one** (a row of exactly `MAXSC`, `MAXSC±1` lanes must be priced by the same form on both
  sides of the boundary — body 0's widest row is exactly `MAXSC` and body 1's narrowest is
  `MAXSC+1`, so the pinned wave already straddles it), `carry` (a softmax state leaking from the
  previous row — with `M` rows per block the leak is *between rows of the same block*, which is new
  and needs its own injection), and `dimhalf` (the dot reads 128 of 256 dims — catches a shared
  `K`-row load that got truncated to one warp's slice).

The must-catch fixtures are the existing `--wave` ones (pinned `q_rope`/`k_rope`/`v` in,
`attn_out_pre_gate` out, compared elementwise on the BF16 bits); the positive gate stays "0 of
98,304 elements differ" as recorded in `prototype/attn_decode/results.md` §2.

## 6. The device commands I could not run (hand-off, blocking, one at a time)

The `attn` line before/after any change, on the library route (`tools/gpu_lock.sh` still applies;
`decode_run` does **not** self-lock):

    tools/gpu_lock.sh acquire "attn wave before"
    tools/gpu_lock.sh util
    ./engine/build/decode_run --ids "$(cat tmp/q10/ids1024.txt)" --steps 1 --time-waves --gemm library \
        | grep -E "WAVE gran=stage"           # before; repeat best-of-3 with zero subagents
    tools/gpu_lock.sh release
    # ... kernel change lands ...
    tools/gpu_lock.sh acquire "attn wave after"; ./engine/build/decode_run --ids "$(cat tmp/q10/ids1024.txt)" \
        --steps 1 --time-waves --gemm library | grep -E "WAVE gran=stage"; tools/gpu_lock.sh release

The two §3 residency questions, one short program (correctness-only, well under a minute):

    hipDeviceGetAttribute(&n, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, 0);  // 64 KB?
    int blocks; hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, attn_core, 256, 0);
    // blocks == 1 confirms the 38,944 B reservation caps residency at one block per CU
    // (no tool does this query yet; it is ~15 lines in the style of tools/ and needs a device)

and, if it is 1, the same query re-run with the `sc` staging moved out of LDS — that is the one-line
experiment that turns §3 from an open item into a finding. The instruction-vs-traffic discriminator,
if it is ever worth a counter pass: `rocprofiler --basic_counters` over one `attn_core` launch,
comparing `SQInstrExecuted` against this file's 18.7 G/wave static upper bound and the DRAM read
bytes against 25.79 GB/wave.

The gate itself is `prototype/attn_decode/run.py` — `--wave` prices a whole prefill wave through one
launch (results.md §11), `--negative` runs the 12 injections of §6, and `--softmax full` is the
bit-exact route. **`attn_decode/run.py` does NOT self-acquire the GPU lock** (`grep -c gpu_lock
prototype/attn_decode/run.py` is 0, unlike `prototype/glue/run.py` and friends), so the caller holds
the lock around it:

    tools/gpu_lock.sh acquire "attn wave gate"
    .venv/bin/python prototype/attn_decode/run.py --fixture dec8 --softmax full --wave --wave-rows 64
    .venv/bin/python prototype/attn_decode/run.py --softmax full --long --wave --negative
    tools/gpu_lock.sh release

Both are correctness-only and fit the worker budget in §7 once the lock is free; they were not run
for this document because Phase 1 changes no kernel, so there was nothing new to gate.

## 7. Pre-registered expectation (so a win cannot be a moved denominator)

Stated before any measurement, against the numbers in §0:

1. The measure is `WAVE gran=stage family=attn ms` on the **library** route at 1024 tokens,
   `wave_rows=512`, best-of-3, zero subagents, baseline **109.2 ms** re-taken in the same window —
   not the share, which moves when anything else gets cheaper.
2. **A win must put `attn` at ≤40 ms absolute** (GQA sharing alone predicts ~27–30 ms of traffic
   plus the softmax redundancy it does not remove). 40–80 ms is "traffic fell but something else now
   binds" and must be reported as a partial; > 80 ms is not a win even if it is faster.
3. **`proj`, `mlp_down`, `mlp_gateup`, `gdn_rec` ms must not move** beyond their own noise. If they
   move, the window was contaminated and the number is void, per the standing GPU rules.
4. **The traffic hypothesis has its own falsifier**: if KV traffic falls 4× (GQA sharing) and `attn`
   ms does not fall anywhere near 4×, then traffic was *not* the binding constraint and §4's verdict
   is wrong — that outcome is a finding about issue-bound behaviour, and the fix is the redundant
   pass-2 loops, not more tiling. Say which of the two happened; do not report only the ms.
5. Correctness gate before any timing claim: the one-pass route stays **bit-identical** (0 differing
   elements) to the pinned `attn_out_pre_gate`, and both negative controls fail for the right reason
   on the right subset. A timing on a kernel that is not bit-identical is a different project.

## 8. Relation to what is already written

* `bench/p9-chunked-scan-2026-10-14.md` §4 already derived the same **25.79 GFLOP/wave** and the same
  0.236 TFLOP/s by the same triangle arithmetic, and already named the lever as "tiling/MFMA
  (flash-style blocking over the KV)" rather than a numerics item. This file agrees with that
  conclusion and supplies what §4 did not have: **why** the rate is 80× under the matrix rate
  (1.0 byte/flop, 2,050× re-read), the launch geometry and ISA facts behind it (§1–§3), the
  conditions under which tiling stops helping (§3's score-staging budget, §4's two walls), and one
  open device question that could dominate all of it (§3). Two details to carry over from §4: it
  prices the *site* at 9.067 ms while a site is 1–3 kernels, and it writes "the same 20 CUs" — the
  CU count used here is **40**, per the `devices/strix-halo-compute.md` correction of HIP's 20.
* §4's aside that "the triangle assumption matters by exactly 2×" is worth keeping next to this
  file's §1: the triangle is already packed into the launch, so the 2× is about *what is scored*,
  not about idle blocks.
* Anything that reads `attn`'s 0.236 TF/s as "attention is small, leave it" is now contradicted with
  arithmetic: it is 33.3 % of the post-flip wave, the largest single non-matmul item after GDN, and
  it grows `O(T²)` in context while GDN is linear.
