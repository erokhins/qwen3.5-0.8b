# S1 follow-on — fold the `glue` and `copy` launches into their consumers

**Status: implemented, CPU-proven, dec8-gated on device both arms, A/B script handed back. Nothing is
committed.** No millisecond figure in this file is a measurement this route is entitled to bank — the
ones that appear are quoted from `engine/generation-limits.md`, `prototype/launch_price/census.py` and
`tools/step_traffic_model.py` and labelled where they come from. `engine/performance-status.md`,
`README.md`, `engine/generation-limits.md` and everything under `progress/` are untouched.

## 0. The headline finding, before the detail

The population is 68 launches per decode step: `glue` 49 and `copy` 19 (quoted:
`engine/generation-limits.md` §1; re-derived from source by
`prototype/glue_copy_fold/census_sites.py`, which refuses on a mismatch).

**19 of the 68 fold. 49 do not.**

* `copy`'s 19 all fold, and both folds are **exact** — no arithmetic moves at all, so no band is
  needed and no gate can go red on numerics:
  - the embedding row copy (1 launch) is pure data movement whose only reader is layer 0's norm
    launch, which is pointed at the tied-table row instead of at a copy of it;
  - the conv roll-back (18 launches) is absorbed by letting the conv write its own end-condition rows
    into the state window instead of into a scratch row somebody then copies. The prototype's own
    comment already argues this is per-thread safe; the scratch exists only so both pointers can stay
    `__restrict__`.
* `glue`'s 49 do **not** fold, and the reason is arithmetic rather than taste — quantified in §3:
  **394 MB/token of extra reads to buy 0.152 ms, 10.9× the prize it is buying.** The band question the
  brief anticipated therefore does not arise on what actually folds: nothing that folds does
  arithmetic, and the only fold that would have needed a band is the one that cannot be bought.

Prize scales to 19/68 of the population: at the measured 3.10 µs/launch (quoted, §4) that is
**0.0589 ms of submission cost**, plus 2.658 MB/token of deleted copy traffic ≈ 0.011 ms at the ~237
GB/s read plateau (quoted). **A full win here is about 1 % of the step, not 2.7 %** — the population
ceiling is 0.2108 ms and only 19/68 of it is reachable. Its value is that it is the last per-launch
population left in the decode step, not that it is large. Pre-registered lines: WIN ≥ 0.10 × 19/68 =
**0.0279 ms**, KILL ≤ 0.03 × 19/68 = **0.0084 ms** on the 4,032-lane step wall
(`engine/generation-limits.md` §7, quoted, scaled by the `folded=` field the A/B reads from its own
runs rather than typing 19).

## 1. The site table (derived, not inherited)

Run: `.venv/bin/python prototype/glue_copy_fold/census_sites.py` (and `--selftest`, 13 deliberately
broken-source cases, all of which it refuses).

| site | call text (keyed on text, not line) | launches/token | read B | write B | classification |
|---|---|---:|---:|---:|---|
| `glue_normonly_L0` | `glue_add_norm_rows(s, nullptr, …, nullptr, sc.norm, n)` | 1 | 4 096 | 2 048 | not-foldable |
| `glue_addnorm_postattn` | `add_norm(sc.delta, …kPostAttnNorm)` | 24 | 147 456 | 98 304 | not-foldable |
| `glue_addnorm_nextinput` | `add_norm(sc.delta, next_norm)` | 24 | 147 456 | 98 304 | not-foldable |
| `copy_embed_row` | `hipMemcpyAsync(sc.embed, table + tokens[0]*kHidden, …)` | 1 | 2 048 | 2 048 | **foldable-now → folded** |
| `copy_conv_rollback` | `hipMemcpyAsync(st.conv_window_layer(layer), sc.conv_new, …)` | 18 | 1 327 104 | 1 327 104 | **needs-kernel-signature-change → folded** |
| **total** | | **68** | 1 628 160 | 1 527 808 | **19 folded / 49 not-foldable** |

Excluded, and named so the census can say what it left out: the wide-body-only submissions
(`hipMemcpyAsync(ids_dev, ids_pin, …)` and `kernels::embed_gather_rows(…)`) — a decode step never
reaches either, so they are not in the 68.

The census re-derives the two totals twice — once from the sites' own control flow (each site's guard
and loop nesting is read back out of `engine/decode_step.cpp`, including that the roll-back now sits
on the shipped side of `if (fold_copy)`) and once from `planned_launch_counts()`'s own expressions,
which it translates out of C++ rather than retyping. It also evaluates the FOLDED bill and requires
`copy → 0` with `glue` unmoved.

**Byte reconciliation against `tools/step_traffic_model.py`**, row by row at the model's own print
precision (±500 B, because it prints `%.3f` MB), with every difference named: the norm-only glue call
passes `delta == nullptr` and `s_out == nullptr`, so the model over-bills it by exactly one BF16 row on
each side; the model has no READ row for the embedding copy and its printer drops every row under
4,096 B, so its 2 KB write never appears; and §1's `glue` read of 0.407 MB is 0.299 MB plus the 100
352 B of norm-weight rows that `launch_price/census.py`'s `STAGE_OF` folds into the same stage — the
same bytes counted twice, and the 6 144 B of `q_norm`/`k_norm`, which belong to no glue launch.

## 2. Why the two `copy` sites fold

**Embedding row.** Shipped, the tied table's row is D2D-copied into `sc.embed` and the residual pointer
`s` starts there. Folded, `s` starts at the table row and the copy is gone. Safe because `s` has exactly
two owners in `submit_wave_body` — its own declaration and the ping-pong's `s = nxt;`, which only ever
yields `sc.s_a`/`sc.s_b` — so the table row is never a write target, and `sc.embed`'s only two write
sites are the shipped D2D and the wave gather, neither of which a folded one-row body reaches. Both
the copy's source and the folded pointer are the same expression `table + (size_t)tokens[0] * kHidden`,
which is what makes the equality identity rather than argument. Checked in §5.

**Conv roll-back.** At `tokens = 1` the end condition is literally
`win row 0 ← old win row 1, win row 1 ← old win row 2, win row 2 ← widen(raw row 0)`, and a lane owns
`CONV_VEC` consecutive channels of **every** row it touches: the three tap rows it reads and the three
new rows it writes are the same `c0 … c0+CONV_VEC-1`. No lane can read a column another lane is
writing, and no lane reads after it writes. What blocks is only the qualifier — one buffer through two
`__restrict__` parameters is the case the qualifier forbids, and the compiler is entitled to sink the
window loads below the window stores because it was promised they cannot alias. So `conv_kernel` stays
untouched (it is a pinned reference and its qualifiers are part of what its gate tests) and
`engine/gdn_conv_roll_kernels.h` holds the aliased-legal spelling: same strip, same 6 × 256 grid, every
read staged into `tap[]` before a single byte is written, and the roll written from those registers.
Same arrangement as `engine/gdn_rec_unroll4_kernels.h` and `engine/gdn_collapse_kernels.h`.

**One-row only, and that is geometry, not policy.** At `tokens > 1` `blockIdx.y` is the token and every
token-block claims the same channel strip, so the end condition belongs to `last_token_only` while the
other blocks are still reading the window. A wave therefore keeps the shipped pair. Same for the
embedding lookup, which at width is one gather over the tile.

## 3. Why the 49 `glue` launches do not

`glue_add_norm_rows` is already the maximal fusion of a residual add and a row norm in one launch.
Splitting the question in half is the only way to see the cost, and both halves fail:

* **The add** is per-element and could ride the GEMV that emitted `delta` (block *i* owns `delta[i]`
  and could write `s_out[i]`). But moving only the add leaves the launch standing — it still owes
  `n_out` — so it saves **zero** submissions.
* **The norm** needs the whole row of the *added* residual, which no launch between the producer and
  the consumer has. Its consumer is a GEMV, whose blocks are one **output** row wide, so riding it
  recomputes the whole-row norm once per consumer block.

The consumer geometry is read from the sources, not assumed (`mlp_gemv`'s `ROWS_PER_BLOCK` = 8,
`kernels_proj.cpp`'s `kProjGroupRowsPerBlock` = 8, `lm_head_gemv`'s `ROWS_PER_BLOCK` = 8), and the walk
is printed row by row by the census:

```
layer 0 norm-only   -> layer 0 input-projection group   1028 blocks
24 × post-attn      -> mlp_gateup                        448 blocks each
23 × next-input     -> next layer's input group   1028 (GDN) / 640 (attention)
layer 23 next-input -> lm_head (final norm)            31040 blocks
= 64 136 consumer blocks per token × 6 144 B of re-read operands
= 394 051 584 B = 394 MB/token of EXTRA reads
```

against a `glue` population worth **0.152 ms** at the measured 3.10 µs/launch (quoted). The extra reads
alone are **1.663 ms** at the ~237 GB/s read plateau (quoted) — **10.9× the prize they would buy**,
before any of the redundant VALU is priced. Excluding `lm_head`'s 31 040 blocks does not rescue it
either: 33 096 blocks is still 203 341 824 B = 203 MB and **0.858 ms**, 5.6× the prize (arithmetic on
the census's own rows, not a printed line). `--proj-plan split` makes it worse
(a packed launch has one norm to read; a split one has four consumers each re-deriving it).

And it would not be bit-identical even if it were free: a consumer block is 1024 threads over a
1024-element row, which is a **different F32 summation order** for the mean of squares than glue's 256
threads × 4 elements — `engine/kernels_glue.cpp`'s own note on exactly that. `prototype/glue/floor_probe.py`
sizes a norm's legal floor at 1 ULP with `ndiff` doing the work, so a band would have to be bought and
sized for a fold that pays 1/11th of its own cost. **No band was requested, loosened, or needed by
anything in this change.**

## 4. The engine change (diff summary)

One knob, `QWEN_COPY_FOLD=1`, environment-gated for the reason `QWEN_CAST_FOLD` is (so
`engine/gate.py` can judge the folded path by inheriting it, with no edit to the gate). Default off =
HEAD's machine code. Both sites are applied only where `n == 1`, next to `fold_cast`.

| file | what moved |
|---|---|
| `engine/gdn_conv_roll_kernels.h` | **new.** `conv_roll_kernel`: the §04.2 conv with the window roll inside it. `win` is both operands, non-`__restrict__` by design; all reads staged into `tap[]` before the first store; the roll is `tap[j+1]`. Provenance, the aliasing argument, and the one-row rule are in the file header. |
| `engine/kernels.h`, `engine/kernels_gdn.cpp` | `kernels::gdn_conv_roll(raw, win, conv_weight, u)`: same grid as `gdn_conv` at `tokens = 1`, stated from the same constants, and it records `launchstats::gdn_conv().record(1)` so the `gdn_conv` family row reads the same on both sides of the flag. |
| `engine/decode_step.h` | `DecodeScratch::copy_fold`; `planned_launch_counts(bool cast_fold, bool copy_fold)` and the comment tying `c.other` to the five sites. |
| `engine/decode_step.cpp` | `fold_copy = sc.copy_fold && (n == 1)`; the layer-0 `s` names the table row when folded and the D2D is skipped; the conv branch picks `gdn_conv_roll` and issues no `kCopy` note and no `hipMemcpyAsync`; `c.other = (copy_fold ? 0 : 1 + kNumGdnLayers) + 1`. |
| `engine/stage_clock.h/.cpp` | `WaveShape::copy_fold`, `Clock::set_plan_copy_fold`, the wave bill's `kCopy` row for a folded one-row body, and a new self-check field `body1_copyfold_plan_matches_token_plan` (the wave bill and the step bill must agree on the FOLDED number, not against the shipped one). |
| `engine/decode_run.cpp` | the env parse (0/1 or a hard refusal), `sc.copy_fold`, `shape.copy_fold`, `clock->set_plan_copy_fold`, the `COPYFOLD on=… plan=… folded=… kept=… sites=5 step_bill=…` echo beside `LAUNCH`, and its `--help` entry. |

`LAUNCH` is untouched: it still prints `planned_launch_counts()`'s shipped bill (340), which is what
`gate.py` section (d) re-derives in Python — the fold-aware number is a separate line, exactly as with
`CASTFOLD`.

**Build**: `engine/build.sh` → both binaries, clean, 23.1 s. Nothing staged (`git diff --cached` empty).

## 5. CPU evidence

`.venv/bin/python prototype/glue_copy_fold/equiv_copy_fold.py` (and `--selftest`, 4 injected-bug cases,
all refused — counts are as printed: census 13 refusals, equivalence 4). No GPU call, no timing.

* **roll, bit-exact.** 8 cases × [3][6144] F32, exactly-BF16-representable inputs, comparing the copied
  path (scratch → copy) against the in-place path (`win[j] = tap[j+1]`), both checked against the
  prototype's prose rule ("the window after this tile is its last K-1 raw columns"). Every destination
  is pre-filled with a sentinel no roll writes, so a write that did not happen **fails** instead of
  surviving — `AGENTS.md`: a write side verified against a buffer pre-filled with the value the writes
  store proves nothing. Injected `GC_WIN_LAG` (shift the wrong way) and a missing newest row are both
  caught.
* **roll rule, independently.** The kernel's `tap[j+1]` mapping reproduces the prose rule at `tokens = 1`
  (K = 4), the newest row is this token's own raw column, and the oldest new row is the window's second
  row — which is precisely the `GC_WIN_LAG` boundary.
* **the transcription is text-checked, not trusted.** The fold kernel re-spells `conv_value`'s dot rather
  than calling it, because calling it would pass one buffer through two `__restrict__` parameters. That
  is the `engine/gdn_collapse_kernels.h` q̃ hazard, and a CPU mirror cannot see it (two sides that share
  a formula agree), so the checker diffs five expressions **character for character** between the two
  headers, and additionally requires the three arithmetic ones to still be what the shipped kernel
  contains — editing `conv_value` without re-transcribing goes red.
* **the embedding fold is a source property, and is checked as one.** `s` has one declaration (naming
  either the table row or `sc.embed`) and exactly one assignment (`s = nxt;`); `sc.embed` has two write
  sites and six read sites; the folded pointer and the copy's source are the same expression. Injecting
  a third write to `s` goes red.

**No numeric band was needed.** Both folds are data movement, so the admission is operand equality plus
byte-identity, not `floor_probe.py`. Nothing in `engine/gate.py`, no floor and no threshold was touched.

## 6. Device evidence, and what is handed back

Correctness only, one run at a time, each inside its own `tools/gpu_lock.sh` hold with an EXIT trap;
**no timing taken and no millisecond figure reported** for any of it.

| what | command | result |
|---|---|---|
| fixture pairing, no device | `.venv/bin/python engine/gate.py --no-engine --fixture dec8` | `ENGINE GATE PASS` |
| dec8, folded arm | `tools/gpu_lock.sh acquire … && QWEN_COPY_FOLD=1 .venv/bin/python engine/gate.py --fixture dec8 --work tmp/glue_copy_fold/gate_on` (10 s wall, util 90 %) | `ENGINE GATE PASS` — re-taken after the final rebuild, PASS again (inside the 10 s hold below) |
| dec8, shipped arm | same without the env (9 s wall, util 92 %) | `ENGINE GATE PASS` (taken before the final rebuild; identical source) |
| plan canary, no device | `engine/build/decode_run --model … --sizes-only` both sides | `COPYFOLD on=0 … plan=19 folded=0 kept=19 step_bill=340` / `on=1 … folded=19 kept=0 step_bill=321`; `LAUNCH` unchanged at `other=20 total=340` on both |
| **state + plan, both arms, one hold** | `decode_run --ids-file tmp/glue_copy_fold/ids_dec8.csv --steps 8 --state-fp --time-stages` per arm, `grep '^STATEFP '` diffed (**10 s wall for all four runs plus the gate**, util 93 %) | **86 STATEFP rows byte-identical, 36 of them `family=conv` at n=18432 = the folded buffer itself**; `sampled.txt` byte-identical (26 lines); shipped arm `launches_per_step=340 plan_match=1` with the `family=copy` stage row **present**, folded arm `321` (340 − 19), `plan_match=1`, `family=copy` **absent**; `COPYFOLD on=1 … folded=19 kept=0 step_bill=321` vs `on=0 … folded=0 kept=19 step_bill=340` |

All six device rows above were taken one at a time inside their own lock hold with an EXIT trap; the
last four were re-taken against a clean `engine/build.sh` (23 s, rc 0) so the binary matches the diff
byte for byte in provenance. **No timing was taken or is reported** — the `--time-stages` runs were read
for their `plan_match` / `launches_per_step` / row-presence fields only.

The `family=conv` STATEFP identity is the one that matters on this route: the folded write targets
state, and a window written one row late or one row wrong can leave the first token's logits untouched
and be decisive by the tenth.

`wide4096f` is **handed back** rather than run: it is a 4,096-lane prefill plus its steps, which does not
fit the ~60 s a worker may hold the device for, and it is the *inert* side of this fold anyway (a wide
body folds neither site). It is inside the A/B script's own numerics phase. To run it alone:

```
tools/gpu_lock.sh acquire gluecopy-wide4096f
QWEN_COPY_FOLD=1 .venv/bin/python engine/gate.py --fixture wide4096f --work tmp/glue_copy_fold/gate_wide_on
.venv/bin/python engine/gate.py --fixture wide4096f --work tmp/glue_copy_fold/gate_wide_base
tools/gpu_lock.sh release gluecopy-wide4096f
```

## 7. The A/B and its pre-registration

`bash prototype/glue_copy_fold/ab_gluecopy.sh` — **main agent only**; it takes milliseconds, acquires
`tools/gpu_lock.sh` itself and holds it for the whole window, and must run blocking with zero subagents
active. `REPS=5 GATE_FIXTURES="dec8 wide4096f" bash …` to widen it. It reuses `cast_fold/ab_item1.sh`'s
discipline: BASE is built from HEAD in a scratch worktree (never from the touched working tree), the two
binaries are `cmp`-checked to be different, `LC_ALL=C` is exported, every parsed field aborts on an empty
or non-numeric parse, and every canary is asserted **before** a millisecond is spent, with
`NO-TIMING` printed on any red.

Pre-registered in the script header, quoted from `engine/generation-limits.md` §7 and scaled by
`folded/68` read out of its own runs:

```
WIN   >= 0.10 x 19/68 = 0.0279 ms saved on the 4,032-lane step wall
KILL  <= 0.03 x 19/68 = 0.0084 ms saved on the 4,032-lane step wall
between                -> neither, reported as neither
```

and the header says in as many words that **the full win is about 1 % of the step, not 2.7 %**.

Canaries, all inside the window: census + census selftest + equivalence + equivalence selftest on CPU;
three-arm `COPYFOLD`/`LAUNCH` agreement (`plan=` must equal `other − 1` from the untouched `LAUNCH`
line); base prints **no** `COPYFOLD` line, so a stale "new" binary is caught rather than A/Bed against
itself; `new-off` is inert (`folded=0 kept=19`); the stage clock's `family=copy` row is **present** on
base and `new-off` and **absent** on `new-on`; `launches_per_step` is 340 on the unfolded arms and 321
on the folded one and equals `step_bill=` from the other line of the same run; `PREFILL tokens=` is
asserted against the field count of the id file (a space-joined fixture reads as ONE token and the run
still succeeds); `STAGEPARITY tokens_identical=1 logits_bitidentical=1`; `gate.py` per fixture per arm;
and byte-identity of `sampled.txt` **and of the STATEFP rows** at 128 and 4,032 lanes against both base
and `new-off`, with the width the byte claim covers stated rather than dropped.

## 8. Open risks

1. **The conv arithmetic is transcribed, not shared.** `conv_roll_kernel` re-spells `conv_value`'s tap
   fetch and dot because calling it would be the `__restrict__` lie. The text check in §5 is what makes a
   future divergence loud rather than silent, but it is a copy until someone factors the shared part.
2. **`wide4096f` was not run here.** It is the inert arm, and the A/B's numerics phase covers it.
3. **The fold's whole value is submission cost, and the route's own verdict may well be KILL.** If
   `cast`'s A/B reads ≤ 0.05 ms when it runs, this route's scaled line is 0.0084 ms and the honest
   outcome is "launches are free when pipelined, here too". That is a legitimate answer and the script
   is written to print it as one.
4. **Line-form citations rotted.** Every file this change touches is cited from `progress/` and from
   `engine/HANDOFF-*.md` by line number, and these edits are not line-count-neutral
   (`decode_step.h` 528 → 548, `decode_step.cpp` 1384 → 1417, `decode_run.cpp` 2501 → 2566,
   `stage_clock.h` 376 → 392, `stage_clock.cpp` 643 → 664, `kernels.h` 928 → 936,
   `kernels_gdn.cpp` 491 → 511). **18 archived coordinates no longer name what they claimed** — listed
   in §9. `tools/citation_rot_check.py` cannot see this (its baseline is `HEAD`, which moved with the
   change); the guard that does see it is `diff <(git show HEAD:file) file`, which is how this list was
   made. Re-pointing them means editing `progress/` and the handoffs, which is out of this route's scope
   and needs the ledger rule applied by whoever owns it.
5. Nothing is committed, and no timing document was touched.

## 9. The rotted coordinates (for whoever re-points them)

`engine/decode_run.cpp:397`, `:1797`; `engine/decode_step.cpp:524`, `:527`, `:530`, `:537`, `:540`,
`:596`, `:599`, `:653`, `:654`, `:687`, `:692`, `:697`, `:732`; `engine/kernels_gdn.cpp:32`, `:40`;
`engine/kernels.h:414`. Cited by `progress/2026-10-12/plan-prefill-2026-10-12.md`,
`progress/2026-10-16/prefill-decision-log-2026-10-16.md`,
`progress/2026-10-17/HANDOFF_17prime_gdn-cast-deletion.md`,
`progress/2026-10-17/HANDOFF_41a_cast-deletion.md`,
`progress/2026-10-17/HANDOFF_42prime_embed-gather.md`,
`progress/2026-10-17/HANDOFF_ceiling_raise.md`,
`progress/2026-10-21/performance-status-2026-10-21.md`, `engine/HANDOFF-D1-ENGINE.md`,
`engine/HANDOFF-D1-FLIP.md`. Coordinates that survived (nothing above them moved): `decode_step.h:49`,
`:51`, `:114`, `:155`, `:174`, `:178`; `decode_step.cpp:45`, `:298`, `:423`, `:465`, `:479`, `:481`;
`decode_run.cpp:64`, `:104`, `:106`, `:325`; `stage_clock.h:4`; `stage_clock.cpp:32`;
`kernels.h:48`, `:84`, `:171`.
