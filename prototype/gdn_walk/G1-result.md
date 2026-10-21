# G1 result — the norm prologue hoisted out of the looped GDN body

**Bet:** `prototype/gdn_walk/BET.md` G1. **Status: implemented, bit-identical on device, ready for the
timing A/B.** The timing A/B is the main agent's and has NOT been run here — see §6.

Device gate result up front, because everything else is commentary on it:

```
shape 512/512: logits compared=46 differing=0 STATEFP_identical=YES
shape 16/8:    logits compared=46 differing=0 STATEFP_identical=YES
```

`--gdn-norm hoist` reproduces `--gdn-norm inline` bit for bit, on every prefill logit row and on every
`STATEFP` line (which carries the post-prefill GDN **state** fingerprints for all 18 GDN layers, plus the
conv windows). No band was touched, widened, or added.

---

## 1. What changed, and why

`rec_loop_kernel` (`engine/gdn_rec_loop_kernels.h`) spends **1,327 of its 1,370 issued slots per
token/wavefront** inside the state walk (`prototype/gdn_walk/evidence-tokenbill-2026-10-19.txt`: 24.8
slots of LDS read traffic, 18.4 of LDS write, 12.6 of barrier, plus the `w_from`/`w_to` walks). The token
loop also carries the per-token k/q L2-norm — a `block_sum_pair_f32` with **three** `s_barrier`s and a
shared round-trip of both rows — which is a second, unrelated job. A block at 4 wavefronts/CU has three
idle wavefronts to hide that latency with, and §23 had already established the cost is the dependency
chain, not the instructions. G1 asks the narrow question: **does the token step get cheaper when it stops
containing a barrier?**

| file | change |
|---|---|
| `engine/gdn_rec_hoist_kernels.h` | **NEW.** `gdn_norm_hoist_kernel` (the prologue, verbatim, `<<<dim3(NH, tokens), D>>>`) and `rec_loop_normed_kernel` (the walk, text unchanged, zero barriers). |
| `engine/kernels.h` | `enum class GdnNorm`, `parse_gdn_norm`/`gdn_norm_name` (parse-by-refusal, like every other engine knob), `GdnNormStage`, `gdn_rec_span_hoisted`. |
| `engine/kernels_gdn.cpp` | the hoisted launcher; its doc-comment carries the traffic arithmetic (§3). |
| `engine/decode_step.h` / `.cpp` | `StreamCfg.gdn_norm` (default `Inline`); three stage tensors `g_kn`/`g_qn` `[wave_rows][16][128]` F32 and `g_gsc` `[wave_rows][32]` F32 — **16,512 B per lane row** (8 KiB + 8 KiB + 128 B), so **8.06 MiB** at the shipping 512-lane wave, against a ~34 GiB pool. Allocated unconditionally (a branch here would be a path the default run never exercises); one branch at the `--gdn-loop` span call site. |
| `engine/decode_run.cpp` | `--gdn-norm inline\|hoist`, default **`inline`** = shipped behaviour; `gdn_norm=` echoed in `NUMERICS`; refusals (§5). |
| `engine/stage_clock.{h,cpp}` | `LaunchCounts::gdn_norm_hoisted` and the plan-bill term for the extra launches. |
| `prototype/gdn_walk/{norm_tree_cpu,check_hoist_transcription,walk_isa_diff,plan_bill_check}.py` | the four CPU checks (§4). |

The default does not move: `engine/gate.py` is untouched and judges the shipped form. Untouched forms
stay untouched — `--gdn-spell ship|unroll4` and `--gdn-loop 1` submit their existing kernels
unconditionally, and `git diff` shows no edit to `gdn_rec_fused_kernels.h`, `gdn_rec_loop_kernels.h`,
`gdn_rec_unroll4_kernels.h` or `gdn_stage_kernel.h`.

Two honest scope notes:
* **`--gdn-norm hoist` is live only with `--gdn-spell fused`** (the default), `--gdn-loop > 1` and
  `--gdn-scan seq`. The `ship`/`unroll4` bodies and the collapse scan keep their own prologue, and the
  launcher **refuses** rather than running a kernel this flag did not name — verified on device:
  `--gdn-spell unroll4 --gdn-norm hoist` → `only the \`fused\` spelling has a looped transcription`.
* `gdn_norm_hoist_kernel` **re-reads** raw q/k and writes normed copies, so the walk still reads 16 KB of
  q/k per token per layer — it reads the staged rows instead of the raw ones. See §3, which corrects
  BET.md's traffic line.

## 2. Bit-identity: the association argument, and the two bugs that actually settled it

**The argument, as required by the bet.** The reduction the norm depends on is
`block_sum_pair_f32` (`engine/gdn_glue_kernels.h`): per-lane serial partials, `__shfl_down_sync` at
offsets 16, 8, 4, 2, 1, then a second stage that folds `warps = blockDim.x >> 5` partials in the fixed
pattern `(p[0]+p[2]) + (p[1]+p[3])`. It is **not** a butterfly and **not** a serial fold — and F32
addition is neither associative nor commutative in IEEE-754, so "it sums 128 squares" does not pin a
value. What pins it: the pre-pass calls **the same function, with `blockDim.x == D == 128`**, i.e. the
same 4-warp fold width, on the same two rows, in the same order. The walk itself was **not rewritten**:
`check_hoist_transcription.py` extracts the 23 lines from `float* Sh = S + (size_t)h * D * D;` through
`o[h * D + j] = acc;` and requires them **verbatim**, pragmas included, so there is no rewritten
`+= … * ks[i]` line for contraction to disagree about. The prologue expressions are checked as exact
source strings, with a matcher that terminates at `;`, `,` or `)` so `expf(g[h]) * 1.0f` — a no-op
algebraically, a different product tree — is caught.

**What decided it was the device gate, not the argument.** Two real bugs went through the CPU checks
(blind both times) and were caught only by comparing device output:

1. **Grid-axis swap.** The launcher built `<<<dim3(tokens, NH), D>>>` while the kernel reads
   `h = blockIdx.x`. With a 46-token body that writes head-46 rows into a 16-head buffer. First device
   gate: **layer 0's post-prefill state came out all zeros** (`n2=0 maxabs=0`), and 46/46 logits rows
   differed.
2. **Operand pointers hoisted above the token loop.** `rec_loop_normed_kernel` declared
   `ks = knorm + h*D` / `qs = qnorm + h*D` once, outside `t` — reads like a legal strength reduction, and
   is exactly what the "keep the walk text verbatim" framing invites. Every token then walked **token 0's**
   normalised rows. Second device gate: layer 0's state fingerprint moved
   `n2=221.73749689549811` → `n2=45.289693062550455`, 45/46 logits rows differed. The row advance belongs
   in the declaration *inside* the loop — which is why `rec_loop_kernel` declares its operand pointers
   where it does. That comment is now in the header, because the trap is not obvious.

Lesson, in this repo's own shape: **a CPU mirror of the summation tree cannot see what the compiler does
with the operands, and a transcription check cannot see an address.** The `.s` census (§4) proves the
structural claims but is not the witness for the arithmetic; the bit-for-bit device diff is, and it had
teeth — it read red twice before it read green.

## 3. Traffic arithmetic — correcting BET.md's line

BET.md says "traffic cost is small: the prologue read **moves**, it does not double". Half right. The
per-token, per-GDN-layer (16 heads, D = 128, F32) picture:

| | inline (shipped) | hoist |
|---|---|---|
| raw q/k read | 16 KB (in the walk) | 16 KB (in the pre-pass) |
| normed k̂/q̂ written | — | **+16 KB** |
| k̂/q̂ read by the walk | — | **+16 KB** |
| walk's own q/k read | 16 KB | 0 (it reads k̂/q̂) |
| **total** | **16 KB** | **48 KB** |

The walk's operand read does not shrink — it reads the staged rows instead of the raw ones — so the honest
figure is **+32 KB per token per layer**, not +16. Over the shipping shape (4,096 lanes × 18 GDN layers,
`kNumGdnLayers`) that is **+2.42 GB** demanded per wave, on top of the 1.21 GB the stage already demands.
The L2 is 2 MB, so essentially none of it is absorbed. At the measured 237 GB/s ceiling that is
**≈ 10.2 ms of added traffic**, against a **best-case saving of 9.2 % of a ~112 ms body ≈ 10 ms** (D60's
flat, which per D61 is itself an upper bound — ~19 % of a 50 % lane-coverage body is ~5 % of the wave). So
on paper this is close to break-even, which is exactly why it is a device question and why the outcome
lines are pre-registered. **If the timing comes back "dead", the traffic column above is the reason, not
the barrier claim.**

## 4. Build and CPU verification (all run here, all green)

```
bash engine/build.sh                       # BUILD_RC=0, no new warnings
.venv/bin/python prototype/gdn_walk/norm_tree_cpu.py
    operational __shfl_down_sync sim == independent structural derivation: 2944/2944 rows bit-equal
    4-partial fold: 20000/20000 bit-equal; teeth: numpy-sum differs 1232/2944, left-to-right 2429/2944,
    reversed offsets 2944/2944; 6/6 must-catch mutations caught; PASS
.venv/bin/python prototype/gdn_walk/check_hoist_transcription.py          # PASS
.venv/bin/python prototype/gdn_walk/check_hoist_transcription.py --selftest  # PASS, 5/5 mutations caught
.venv/bin/python prototype/gdn_walk/walk_isa_diff.py --selftest            # PASS, 2/2
.venv/bin/python prototype/gdn_walk/walk_isa_diff.py --asm tmp/gdn_walk/s/kernels_gdn-*.s   # PASS
    shipped walk: 5 barriers, 39 LDS ops | normed walk: 0 barriers, 0 LDS ops
    exp 1 = 1 in the pre-pass, rsqrt 2 = 2 in the pre-pass  (the norm arithmetic moved, not rewritten)
    barriers 5 = 3 (pair reduction, now in the pre-pass) + 2 (the ks/qs write pair, deleted)
.venv/bin/python prototype/gdn_walk/plan_bill_check.py                    # PASS, 21 shapes
    THE A/B SHAPE 4096/512/512: gdn_rec launched 144 inline, 288 hoisted, plan agrees on both
    plus 513-lane ragged, wave_rows 1, span 1, span just under a body, and the 1-row-tail shape
    confirmed unreachable (it aborts in the launcher)
```

`plan_bill_check.py` exists because the plan bill's own audit (`plan_match`) is compiled into the binary
being audited — the same self-agreement blind spot `AGENTS.md` records for `wave_plan_bill` as a canary —
so the derivation is spelled a second time here, and it disagrees nowhere. It found one real thing on the
way: `--gdn-loop 1` is not a span case at all (it runs the per-token loop over every token of the body),
and the first version of the mirror got that wrong before the C++ `else if (gdn_loop_span > 1)` guard was
read.

`walk_isa_diff.py` counts **issued slots** and partitions cleanly on exactly those three classes
(`barrier`, `lds`, `exp`/`rsqrt`), which is the bet's structural claim stated where it is checkable. It
does **not** partition on the FP mix (11 of 15 classes) because `v_dual_*` hides an op inside one mnemonic
and both token loops are rolled, so a body's count depends on how far LLVM unrolled it in each caller —
hence it reports the FP mix and asserts nothing about it. The census is also **not** the per-token bill:
`prototype/gdn_rec_isa/token_bill.py` owns that (per the "list the census tools" trap).

Device gate, run under `tools/gpu_lock.sh` (TAG `g1-final`; the bug-hunting runs used `g1-bitident`),
correctness-only, **no timing taken**:

```
IDS=$(cat /tmp/g1_ids.txt)   # the 46 ids of data/refs/p64/input_ids.npy
for SHAPE in "512 512" "16 8"; do for SIDE in inline hoist; do
  ./engine/build/decode_run --model models/Qwen3.5-0.8B --ids "$IDS" --steps 0 \
      --prefill-logits all --state-fp --wave-rows $1 --gdn-loop $2 --gdn-norm $SIDE \
      --out /tmp/gf_${1}_${2}_${SIDE} > …log 2>&1
done; done
# then cmp of every logits*.bin and of the two STATEFP extracts
```
Wall: **4 s for all four runs**, on the tree as it now stands (the two bug-fix runs and the green run were
repeated after the last rebuild so the verdict matches the handed-over tree). `tools/gpu_lock.sh util`
read **90 %** at acquire and **29 %** just before release — a prior attempt read 89–90 % at acquire and
still returned a deterministic verdict, as expected for gates with no atomics. Nothing from these runs is
a timing, and no number here belongs in `engine/performance-status.md`.

**Four** short gate invocations were used rather than the one the brief allowed, because two of them found
real bugs (§2) and a green verdict on a tree that no longer matched the build would have been worthless; all
four were correctness-only, 1–4 s each, and none produced a millisecond figure.

`speccheck/run_all.py` was **not** run: nothing it covers changed (no `reference_dump.py`, no
`speccheck/specimpl.py`, no gate file). Refusals were exercised without loading a model:

```
--gdn-norm bogus                  -> refuses, names both accepted values
--gdn-norm hoist --gdn-loop 1     -> refuses (per-token path keeps its own prologue)
--gdn-norm hoist --gdn-scan collapse -> refuses (different kernel, inert knob)
```

## 5. Admissions and refusals

* `--gdn-norm` takes `inline|hoist`; anything else refuses at parse time (AGENTS: a shape claim is a
  contract, and a silent fallback would price the shipped form twice and report it as the hoisted one).
* `hoist` + `--gdn-loop 1`, and `hoist` + `--gdn-scan collapse`: refuse at parse (the knob would be inert).
* `hoist` + `--gdn-spell ship|unroll4`: refused at the launch site, **verified on device** —
  `only the \`fused\` spelling has a looped transcription … Use --gdn-spell fused (the default) or
  --gdn-loop 1`. Same reason as above: the flag names which kernel ran.
* `gdn_rec_span_hoisted` refuses `tokens < 2`, mirroring `gdn_rec_span`'s existing refusal.
* **Pre-existing defect found, NOT caused by this change, NOT fixed:** with a span that leaves a 1-row
  tail — e.g. `--wave-rows 16 --gdn-loop 13` over 46 tokens (bodies 16/16/14, tail 14−13 = 1) — the span
  loop hands `gdn_rec_span` a 1-row span and it throws: `gdn_rec_span: tokens=1`. The same happens on the
  `inline` side, i.e. in code already at HEAD: `resolve_gdn_loop` routes a 1-row *body* to the per-token
  kernels, but nothing routes a 1-row *span* of a wider body. Unreachable with the default `span ==
  wave_rows`; reachable with any narrower span whose remainder is 1. `--gdn-loop`'s help ("A span below
  the body size means what it says") overpromises. Reported, not silently widened or patched.

## 6. Handover: the A/B for the main agent

Pre-registered lines from the brief: **live ≤ 250 ms · in the middle 250–258 ms, no verdict · dead > 258
ms**, against §37's `266.24 ms` at 4,096 lanes. One window, one lock hold, blocking, zero subagents.

**The harness: `bash prototype/gdn_walk/run_ab_g1.sh`** (written for this bet, modelled on
`tools/regress_vs_origin.sh`; the base side is built from a detached worktree at `HEAD`, which is required
because the working tree now carries the change). It runs **three sides**, because two pairs answer two
different questions:

| pair | question it answers |
|---|---|
| `base` (HEAD, no flag) vs `new --gdn-norm inline` | **must be a NULL** — did adding the knob move the shipping path? |
| `new --gdn-norm inline` vs `new --gdn-norm hoist` | the bet itself, inside one binary, so no build difference can leak in |

Canaries it asserts before it reports a number (the D30 rule — a stale binary A/B reads as a clean null):
* **A** (outside the hold): the base binary must print `unknown argument --gdn-norm`; the new one must
  accept `--gdn-norm inline`.
* **bit-identity** (outside the hold, since a deterministic check needs no exclusive device): re-proves
  `base == inline == hoisted` on every logit row and every `STATEFP` line at `--wave-rows 16 --gdn-loop 8`,
  and aborts if the default side moved at all.
* **B** (inside the hold, every rep): each side must echo `gdn_norm=inline` / `gdn_norm=hoist`, and the
  `gdn_rec` family line must read **`launches=144 plan_launches=144 counts_match=1`** inline and
  **`launches=288 plan_launches=288 counts_match=1`** hoisted, with `TOTAL … plan_match=1`. Those two
  numbers are the shape `prototype/gdn_walk/plan_bill_check.py` derives (8 bodies × 18 layers, one span
  each; the hoisted side adds 144 pre-passes). If the observed and planned counts disagree, or the two
  sides read the same count, the window is discarded rather than reported.

It prints per-family medians for all three sides plus the `RESULT ms_total` line the verdict is taken on,
and says explicitly to quote the pair, the spread and the clocks — §6's dispatch-bound bimodality swallows
short-prompt rows, and §37's number is the ruler. 4,096 ids is ~24 KB, safely inside one `MAX_ARG_STRLEN`
slot; if this is ever retaken at 32k lanes, pass them by `--ids-file` (§39).

## 7. What this cannot see

* **Whether the barrier removal is worth anything.** Every number here is a slot count or a bit diff; only
  the timed A/B decides, and the traffic column in §3 is the counterweight that could make it dead.
* **D61's correction as a measurement.** The slots removed are the barrier plus the LDS round-trip, and
  D61 says a third of the body is latency that does not leave when instructions do. Expect the win to
  land below 9.2 % of the body even if traffic were free.
* **Long-context occupancy.** The stages add 16,512 B per lane row (**8.06 MiB** at the shipping 512-lane
  wave); fine on this device, unmeasured for occupancy effects, and the bet explicitly declines to size
  memory pressure.
* **Contrived-peak arithmetic.** Per D62 no arithmetic from the FMA census is trusted against a peak.
* **Decode.** Not touched: a one-row body routes to the per-token kernels, which keep their own prologue,
  and `--gdn-loop 1` is the unchanged path.
* **The `.s` listing** is a whole-kernel **slot** census; it cannot see the stall profile, and its FP mix is
  not additive across the two kernels (§4).
