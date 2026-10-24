# E1 measured on the real body: Q un-staged is zero-spill at 184 VGPR — and it buys no residency at all

Task: `prototype/attn_wmma/fattn_regfile_ladder.md` §4 **E1** (the 3-warp `attn_fattn_wave` body with Q
un-staged), plus **A0**'s missing half (why 42 of 48 traced waves go through the streaming WMMA path).
Written 2026-10-19 on tip `bb3ec13`. **Compile-only and driver-fact-only: no kernel was launched and no timing
was taken**, by anyone, in this file. Verbatim listings and commands are in
`prototype/attn_wmma/evidence/fattn_e1_noteblocks.md`; the port itself is
`prototype/attn_wmma/evidence/fattn_unstage_q.diff` and **is not applied in the tree**.

Citation legend (the repo rule is that a `§N` that names no file resolves only in a reader's memory): a bare
`§N` below means §N **of this file**; `§31/§33/§37/§26` mean `engine/performance-status.md`; `§1/§2/§3/§4/§5`
when the sentence is about residency or the ladder mean `prototype/attn_wmma/fattn_regfile_ladder.md`; `§7` on
a line about the attention family means `prototype/gdn_walk/BET.md`; `§20.1` is
`prototype/attn_decode/results.md` §20.1 and `§07.3` is `specification/07-decode-step.md`.

## 0. The answer, in four lines

| | shipped (`s_q` staged per warp) | E1 (Q read from global at the use site) |
|---|---|---|
| `vgpr_count` / `vgpr_spill_count` | **178 / 0** | **184 / 0** |
| `private_segment_fixed_size` / `sgpr` | 0 / 24 | 0 / 27 |
| `group_segment_fixed_size` | **32,768 B** | **32,768 B** — unchanged |
| blocks/CU → wf/CU (KFD LDS, rocminfo ceilings) | **2 → 6**, LDS-bound | **2 → 6**, LDS-bound |

The register leg of the ladder's admission passes (+6 VGPR, zero spill, zero scratch — the skeleton's V6 "73
spills" story does **not** survive contact with the real body, exactly as §2 predicted it might not). The
**LDS leg fails**, and LDS is the limit that binds this kernel: the Q stage was never extra LDS, it was the
epilogue's F32 transpose staging on loan. So **no 8+ wf/CU zero-spill shape exists by this lever**, the
residency axis stays at 6, and no device A/B is owed (§4 below states the rule and why its antecedent is
incomplete). BET.md §7's "spill > 0 closes the axis" branch was not the branch taken; the axis closes for a
different and more structural reason.

## 1. The two note blocks, side by side

Same TU, same `-O3 --offload-arch=gfx1151`, same 96-thread block, one kernel differs:

```
        field                     shipped (Q staged)      E1 (Q from global)
        vgpr_count                     178                     184
        vgpr_spill_count                 0                       0
        private_segment_fixed_size       0                       0
        sgpr_count / sgpr_spill         24 / 0                27 / 0
        group_segment_fixed_size     32768                 32768
        max_flat_workgroup_size         96                    96
        uses_dynamic_stack             false                 false
```

Full YAML blocks, both from `.name: _ZN6engine7kernels15attn_fattn_waveE8CoreArgs`, are in
`evidence/fattn_e1_noteblocks.md` §3. The shipped side reproduces `progress/2026-10-21/performance-status-2026-10-21.md` §33's banner
(178 / 0 / 0 / 32,768) exactly, which is what makes the comparison legal rather than a diff of two windows.
Drift check over the whole TU (`fattn_residency_read.py --all`): exactly one line differs; `attn_core`
(53 VGPR / 38,944 B), `attn_slim_wave` and `attn_tile_wave` (47 / 12,288), `attn_prep` (30 / 2,080),
`attn_combine` (10 / 0) and `rope_table` (16 / 0) are untouched.

What it decides: **+6 VGPR is the whole price of un-staging at three warps**, against Bet 1's +2 at one warp
(§31) and the skeleton's +73 spills (V6). Zero spill and zero scratch in both forms, so §33's spill tax is not
in play on either side.

## 2. Blocks/CU and wf/CU, from the facts this box was actually asked

```
KFD  /sys/devices/virtual/kfd/kfd/topology/nodes/1/properties : lds_size_in_kb 64, simd_per_cu 2, simd_count 80
rocminfo Agent 2 (gfx1151)                                   : Compute Unit 40, SIMDs per CU 2,
                                                             Max Waves Per CU 32, Max Work-item Per CU 1024
```

40 CUs from `rocminfo` agrees with KFD's `simd_count 80 / simd_per_cu 2`; HIP's capability answers and clang's
`Occupancy [waves/SIMD]` note are not consulted (AGENTS.md, "Counter, occupancy and census traps"), and the
VGPR-file-per-CU is **unknown on this part** (ladder §3) so no register-derived block count is used:

| limit | shipped and E1, 96 threads / 3 warps / 32,768 B | blocks/CU |
|---|---|---|
| LDS | floor(65,536 / 32,768) | **2** ← binding |
| work-items | floor(1,024 / 96) | 10 |
| wavefronts | floor(32 / 3) | 10 |
| register file | **UNKNOWN — not applied** | — |

**blocks/CU = 2 → wf/CU = 2 × 3 = 6**, for both forms, and the bound is LDS alone. Every limit other than LDS
would allow ≥ 3 blocks/CU, so the residency question is now entirely an LDS question.

## 3. The admission rule, verbatim, and its verdict

From `fattn_regfile_ladder.md` §4, E1:

> * `spill = 0` and VGPR ≲ 190 ⇒ there **is** an 8+ wf/CU zero-spill shape, and it is worth a device A/B
>   against the shipped 6-wf form;
> * spill > 0 ⇒ §33's spill tax re-applies and the residency axis is closed at 6 (with NW=7 as a footnote).

Applied to the measured pair:

* `spill = 0` — **passes** (0 spilled, 0 B of scratch).
* `VGPR ≲ 190` — **passes** (184).
* The consequent's premise — **fails**. The rule was written against §2's expectation that "the un-staged
  3-warp body lands near **8192 B** of LDS"; it lands at **32,768 B**, so the shape the antecedent buys is a
  zero-spill shape *at 6 wf/CU*, which is the shape that already ships. **There is nothing to A/B on residency
  grounds, and the A/B is not run.** The axis closes at 6 — not by the spill tax, but because the block's LDS
  did not move.

The ladder's own §1 said the shipped NW=3 at 6 is "LDS-bound, not register-bound… *no register cut at all moves
residency in the shipped geometry*"; E1 now says the converse half too — no register *cost* matters either, and
no Q change moves residency, because Q's stage is what the epilogue is standing on.

## 4. Why the LDS did not move, and what would have to move for 8+

| form | Q | epilogue transpose staging | K/V | declared LDS |
|---|---|---|---|---|
| shipped | 3 × 8,192 B (`s_q`, one per warp) | **rides the dead `s_q`** — 16 rows × `FATTN_TROW` F32 = 8,192 B/warp | 8,192 B | 32,768 B |
| E1 | — (global reads at the use site) | 3 × 8,192 B (`s_tr`, one per warp) | 8,192 B | 32,768 B |

The single assert that sizes all of this is the one the shipped file already carried:

```
static_assert(FATTN_MROWS * FATTN_TROW * 4 == FATTN_MROWS * DH * 2, ...)   // 16x128 F32 == 16x256 F16 == 8192 B
```

One warp's epilogue staging **is** one warp's Q stage, byte for byte, and it is per-warp on purpose (three
warps transposing through the one shared `s_kv` is §20.1's aliasing bug — the file says so, and Bet 2's NaN
history is the receipt). So the block's LDS floor with this epilogue is `3 × 8,192 + 8,192`, Q staged or not.

For 3 blocks/CU — 9 wf/CU, which is what the rule's "8+" means at 3 warps per block — the block needs
**≤ 21,845 B**. The only ways to get there that this file can see, none of which is E1 and none of which is
approved:

* stage the epilogue **8 rows at a time** instead of 16: `3 × 4,096 + 8,192 = 20,480 B` → 3 blocks/CU =
  9 wf/CU, at the cost of 4 transpose passes instead of 2 (same barriers ×2, half the bytes);
* serialise the three transposes through the one `s_kv` stage: 8,192 B → 8 blocks/CU on LDS, but that puts the
  epilogue behind block-wide barriers per warp and would then be register-file-bound at an unknown file (§3's
  canary is a precondition for even claiming it).

Both are epilogue changes, not Q changes. **That is the re-framing E1 hands back: the residency axis is the
epilogue's, not Q's.**

## 5. The port (what it changed, and why the bits cannot move)

`evidence/fattn_unstage_q.diff` — 258 lines over two files, `git apply --check` clean against the tree as it
now stands. It is a hand port, not the `tmp/gcc/fattn_unstage_q.patch` apply (4 of 5 hunks of that patch fail
against the Bet-2 body, as the ladder records). In the body:

* `s_q`'s per-warp fill (`for (qr = 0; qr < FATTN_MROWS; ++qr)` → `fattn_stage_row`) is **gone**; the buffer
  stays, renamed `s_tr` and typed `float`, because the epilogue needs it (§4).
* new `fattn_load_q_tile(const hip_bfloat16*)`: two `uint4` loads and the **same**
  `__float2half_rn(fattn_bf(x))` conversion `fattn_stage_row` applied on the way in; feeds the score leg's
  SECOND operand in place of `fattn_load_tile(&s_q[...])`.
* the shipped clamp is kept and is still per-lane: `qrs = min(qslot, qrmax - 1)` resolves to row `nrows - 1`
  when a warp's row-block is wholly past `nrows`, so no lane ever dereferences a row that does not exist (the
  staged form `break`ed; this form reads a valid row whose cells are masked out by `live` either way).
* `fattn_q_addr` (the stage index) is replaced by `fattn_q_row`, the one spelling of the global Q row
  (§20.1's one-address-expression rule, moved with the object it describes).
* barrier 1's comment changes from "Q published, spans published" to spans only — the fold through the front of
  `s_kv` still needs it, and barriers 2–6, the round protocol and the publish set are untouched.
* `engine/kernels_attn.cpp`: the banner's `q=stage` → `q=global`, so the field the change is required to move
  is a printed number (the D30 canary lesson applied here at compile level: a canary that cannot disagree with
  a stale build is not an instrument).

**Bit-identity is an argument, not a measurement**: same 16 source bytes per tile (same row, same clamp, same
order), same `bf16 → f32 → f16 RN` conversion, same masks — so the F16 operand bits are identical and the
arithmetic downstream is untouched. The ISA says the conversion moved rather than multiplied: `v_cvt_f16_f32`
sites are **48 in both listings**, `ds_store` 77 → 75, `ds_load` 71 → 69, `v_wmma` 17 and `s_barrier` 11
unchanged, and the score leg's `v_wmma` now takes `v[168:175]`, the pair produced by
`global_load_b128 … off` + `global_load_b128 … off offset:16` (the 16 halves), where shipped it took a
`ds_load` result. The device witness for bit-identity is `prototype/attn_wmma/run_fattn_admission.sh`
section 1 (`--attn-fattn` sides byte-identical on one-pass rows), **not this file**.

## 6. A0: the work split, derived — 48/42/42 reconciles exactly

`prototype/attn_wmma/a0_work_split.py` (CPU-only; parses every constant from the file that owns it and refuses
to run if any enforcing expression below has changed spelling).

**The rule.** The traced counts are not a policy about the WMMA form; they are one integer, `split`, gating two
of the four launches:

| leg | enforcing site | launched when |
|---|---|---|
| `attn_prep` | `engine/kernels_attn.cpp:413` | every wave body (`rows > 1`) — always |
| `attn_slim_wave` | `engine/kernels_attn.cpp:447` | every wave body under `--attn-touch 1` — always |
| `attn_fattn_wave` | `engine/kernels_attn.cpp:467`, gated by `:338` `wave_fattn = … && split >= 2` | **only if `split >= 2`** |
| `attn_combine` | `engine/kernels_attn.cpp:487`, gated by `:477` `if (split > 1)` | **only if `split > 1`** |
| `split` itself | `engine/kernels_attn.cpp:248` `const int split = split_for(lanes_last);` | with `:86` `split_for(l) = ceil(l / MAXSC)` and `:226` `lanes_last = a.lvalid + rows - 1` |
| a wave body's `lvalid` | `engine/decode_step.cpp:697` `a.lvalid = pos0 + 1` | so `lanes_last` of a 512-row body is `pos0 + 512` |
| a wave body's width | `engine/prefill.cpp:57` (`wave_cap = sc.wave_rows`) | `sc.wave_rows = kMaxWaveRows = 512` (`engine/decode_step.h:51`) |
| which ROWS the WMMA kernel owns once launched | `prototype/attn_decode/attn_decode_kernels.h:553` `fattn_covers` → `:220` `blocks_for_row` → `:221` `if (a.full && lanes_of(a,qrow) <= MAXSC) return 1`, with `:203` `lanes_of = lvalid + qrow`; `MAXSC` is `:32` |
| how many attention layers a pass has | `specification/00-notation-and-hyperparameters.md:93` — `full_attention` at `3,7,11,15,19,23` → **6 layers** |

**The reconciliation.** A 4,096-lane prompt is `ceil(4096/512) = 8` wave bodies × 6 attention layers:

```
 body rows  lvalid lanes_last split  fattn  combine  rows covered
     0  512      1        512     1     NO       NO     0/512
     1  512    513       1024     2    yes      yes   512/512
     2..6 (split 3..7)                          ...   512/512 each
     7  512   3585       4096     8    yes      yes   512/512
```

* `attn_prep` = 8 × 6 = **48** ✓ traced 48
* `attn_slim_wave` = 8 × 6 = **48** ✓ traced 48
* `attn_fattn_wave` = 7 × 6 = **42** ✓ traced 42 — body 0 is the one-pass wave: `lvalid` starts at 1, so
  `lanes_last = 1 + 512 - 1 = 512 = MAXSC`, `split_for(512) = 1`, and no row of it has `lanes_of > MAXSC`.
* `attn_combine` = 7 × 6 = **42** ✓ traced 42 — same gate, `split > 1`.

Independent witness that this is the launcher's own arithmetic and not a story told beside it: the script's
aggregate coverage is **3,584/4,096 rows = 87.5 %** and **98.4 % of lanes**, against the numbers
`engine/kernels_attn.cpp`'s `fattn_echo` comment claims for 4,096 lanes ("row share ~87 %, work share ~99 %"),
written by a different author in a different file. The script exits non-zero if either count or either share
fails to reproduce.

**What this buys the next decision.** The two bets are now separable and they are *not* the same size:

* **"too many launches go through it"** is dead as a framing. 42 of 48 is not a routing choice that could be
  widened or narrowed — the 6 missing launches are wave body 0 of a prompt, which `split_for` cannot give a
  partial, and the WMMA kernel has nowhere to write on a one-pass wave (`part` is null; the launcher prints
  `ATTN_FATTN_SKIP` for exactly that case, `:339-347`). The covered share of the *work* is already 98.4 %, so
  there is at most 1.6 % of streaming work outside the WMMA form to reclaim, and 12.5 % of rows outside it that
  are one-pass by §07.3's pin-equality decision and must stay there.
* **"the kernel is slow per launch"** is where all of it is: 122.72 ms ÷ 42 = 2.92 ms/launch (BET.md §7), and
  after the last round the form runs at ~3.17 TFLOP/s ≈ 6 % of the WMMA-class roof — while llama's ~15 % of the
  same roof proves neither engine is roof-bound. That is an instruction-and-latency question inside one kernel,
  the same class as §26's 4.6× gap, and it is a different and larger piece of work than anything in this file.

## 7. What was built, and which checks were run

| command | result |
|---|---|
| `engine/build.sh` on the shipped tree | built, 17.4 s |
| `hipcc -O3 --offload-arch=gfx1151 --save-temps -c engine/kernels_attn.cpp` (shipped) | clean, note block matches §33 |
| apply `evidence/fattn_unstage_q.diff` → `engine/build.sh` | built, 17.5 s — the variant compiles in the real build |
| same `--save-temps` compile (variant) | clean, note block read |
| `git apply -R` the variant → `engine/build.sh` | built, 17.3 s — tree restored to shipped and green |
| `python3 prototype/attn_wmma/fattn_residency_read.py --asm …` (both sides, and `--all`) | derived 2 blocks/CU → 6 wf/CU both sides; one TU line differs |
| `python3 prototype/attn_wmma/a0_work_split.py` | exit 0, 48/48/42/42 reconciled, coverage 87.5 % / 98.4 % |
| `git apply --check evidence/fattn_unstage_q.diff` | applies cleanly (verified after the revert; not left applied) |
| `--save-temps` compile of the **restored** tree (`/tmp/e1final`) | `vgpr_count 178 / spill 0 / private 0 / group 32768` — the revert is verified at codegen level, not only in the source text |

**The tree does not contain the variant.** It was applied, built, compiled, read, and reverted; the port lives
in `prototype/attn_wmma/evidence/fattn_unstage_q.diff`. Nothing under `engine/` is modified. Two new CPU-only
tools are added and untracked: `prototype/attn_wmma/fattn_residency_read.py` (note block + residency derived
from KFD/rocminfo, reusing `fattn_ship_census.note_blocks` rather than writing a second YAML parser) and
`prototype/attn_wmma/a0_work_split.py`.

## 8. Device command — **not owed**, and here it is if the traffic question is priced anyway

The residency admission did not pass (§3), so the A/B it would have licensed is not run and nothing here should
enter `engine/performance-status.md`. If the main agent decides to price the *traffic* leg anyway — the only
thing the un-stage actually changes — it is a two-binary paired A/B in one `gpu_lock` hold, medians of 3, 4,096
lanes, with the canary asserted in the same window as the number:

```
git apply prototype/attn_wmma/evidence/fattn_unstage_q.diff && engine/build.sh \
  && cp engine/build/decode_run /tmp/e1qg_new && git apply -R prototype/attn_wmma/evidence/fattn_unstage_q.diff \
  && engine/build.sh && cp engine/build/decode_run /tmp/e1qg_base
tools/gpu_lock.sh acquire e1q && tools/gpu_lock.sh util
for side in base new; do for rep in 1 2 3; do
  /tmp/e1qg_$side/decode_run --model models/Qwen3.5-0.8B --ids "$(seq 1 4096 | tr '\n' ' ')" \
      --steps 0 --time-waves > /tmp/e1qg_$side-$rep.log 2>&1
  grep -h "ATTN_FATTN " /tmp/e1qg_$side-$rep.log        # CANARY: base must read q=stage wf_per_cu=6, new q=global wf_per_cu=6
  grep -h "gran=stage family=attn " /tmp/e1qg_$side-$rep.log | tail -1
done; done
tools/gpu_lock.sh release e1q
```

`wf_per_cu=6` on **both** sides is the expected read and is the point: if the `new` side prints anything else,
the build is stale (AGENTS.md, "a paired A/B against a stale binary reads as a clean null"). The pre-registered
lines from BET.md §7 (family **live if ≥ 8 %**, dead inside §37's ±3 %) were written for a residency win; with
no residency delta they would be judging a traffic change against a bar set for something else, and that should
be said out loud in whatever log records it.

## 9. What this cannot see

* **Not a timing, not a speed.** Nothing here says the un-stage is fast or slow. What it says is narrower and
  firmer: there is no residency difference to measure, so the funded question ("does an 8+ wf/CU zero-spill shape
  exist") is answered *no* by declaration arithmetic, and the device pass is not needed to answer it.
* **The traffic leg is arithmetic, not measurement.** At the shipping shape (split 8, `span` 512, 32 rounds) a
  warp re-reads its 8 KiB of Q every round: **16 KiB demanded / 8 KiB distinct per warp per round**, so
  524,288 B demanded per warp per launch where the staged form read 8,192 B once. At 6 resident wavefronts/CU
  the distinct Q footprint per round is 48 KiB against the 32 KiB L1 `rocminfo` prints, so it cannot be
  L1-only; §31 measured the 1-warp-per-block version of this as cheap (8 blocks/CU, one warp's 8 KiB resident),
  and at 3 warps per block it is 3× denser. Demanded ≠ DRAM (AGENTS.md, 2026-10-16 preflight row) — this is a
  cost *risk*, sized, not a cost.
* **Bit-identity is argued at source level only.** The admission script's section 1 is the witness; nobody has
  run it against this variant.
* **The register file per CU is still unknown**, so any residency claim above what LDS allows — including the two
  epilogue routes in §4 — is model until the ladder's E2 canary prints. E1 removes the *need* for that
  measurement at 6 wf/CU, and makes it a precondition for anything above.
* **`prototype/attn_wmma/fattn_ship_census.py` was not run and not touched.** It is a known-red gate (§33's
  trailing note; the ladder §5 says a separate task owns it). Note for whoever repairs it: its `q_unstaged_text`
  mutation models the un-stage by deleting the Q stage outright while `fattn_transpose_out` still indexes it —
  a text that cannot compile at three warps, for the exact reason §4 gives. When the census learns the un-stage it
  must learn that `lds_B` does **not** fall, and `geometry()` must stop deriving a per-warp LDS saving from the
  absence of a Q stage.
* **`--attn-fattn` defaults to 1** (`engine/decode_run.cpp:349`), so this kernel is on the shipping path, and the
  known-red ship census is a live gate on that path. Neither fact is changed by anything in this file.
