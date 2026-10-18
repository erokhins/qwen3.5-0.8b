# Item 2.5 — attention lane ceiling raised: `kAttnMaxSplit` 8 → 9 (lane ceiling 4096 → 4608)

Repo `/home/erokhins/erokhins_projects/qwen3.5-0.8b`, base `3d4d293`. Worker pass. No commits, nothing
staged, no `specification/`, `bench/`, `devices/`, `progress/` edit. No timing taken and no `ms` figure
appears in this file; the gates printed their own by-product `ms/step` lines and those are deliberately
not carried here (`AGENTS.md`: no number from a worker device run enters a timing doc).

## 0. What landed

One line changes behaviour. The other 27 changed lines are comments and the refusal's own message text,
all of which became false without it.

```
engine/kernels_attn.cpp             | 14 +++++-----   ← the constant (1 line) + its justification,
                                                    the `attn_part_floats` sizing note, the refusal text
engine/decode_run.cpp               |  3 +--          ← `kSessionDefaultCap` comment (its reason died)
prototype/attn_m2/attn_m2_kernels.h |  2 +-
                                                    ← the parked prototype's mirror claim (comment only)
tools/dump_fixtures.sh              |  9 +++++-----    ← the wide4096f/wide4096p block ("cannot run today")
4 files changed, 28 insertions(+), 28 deletions(-)   (git diff --numstat: 14/14, 3/3, 2/2, 9/9)
```

Every edited file is **line-count neutral** (`engine/kernels_attn.cpp` 329 → 329, `engine/decode_run.cpp`
1519 → 1519, `tools/dump_fixtures.sh` 252 → 252, `prototype/attn_m2/attn_m2_kernels.h` 945 → 945), so no
line-form citation moved. `tools/citation_rot_check.py` reports **1** rotted cite, described in §6.

The behaviour change:

```c
-constexpr int kAttnMaxSplit = 8;
+constexpr int kAttnMaxSplit = 9;
```

`kMaxWaveRows` (512) was not touched — wave width stays 512, and no plan bill moved (§1 "the plan/wave
bill"). `MAXSC` (512) was not touched.

## 1. Step 1 — every place the split count or the lane ceiling appears

Answer to "if a single constant does not do it, say so": **a single constant does do it.** The
allocation, the refusal and the gate's check all derive from `kAttnMaxSplit`; nothing was half-raised.
P7's note at `engine/results.md:3300` handed on "the two coupled knobs `kAttnMaxSplit` and
`attn_part_floats()`, and `gate.py::lane_geometry()` follows them automatically" — as the code now
stands `attn_part_floats()` is not a second knob but a derived reader of the first, which is why one
line is enough.

| # | site | what it enforces | what happened |
|---|---|---|---|
| 1 | `prototype/attn_decode/attn_decode_kernels.h:32` `#define MAXSC 512` | first factor: lanes one block stages in shared memory | **untouched** |
| 2 | same header `:180` `split_for_lanes()` | `ceil(lanes / MAXSC)`, `__device__ __host__`, one spelling | untouched; reads MAXSC only, never the cap |
| 3 | `engine/kernels_attn.cpp:65` `constexpr int kAttnMaxSplit` | **the only second-factor constant in the tree** | **8 → 9** |
| 4 | `engine/kernels_attn.cpp:69` `split_for()` | engine's spelling of #2 | untouched; cap-free |
| 5 | `engine/kernels_attn.cpp:178` `attn_part_floats()` `= rows*N_Q*kAttnMaxSplit*(DH+2)` | the `part` allocation, derived from the cap | numbers in its comment restated (38,043,648 B) |
| 6 | `engine/kernels_attn.cpp:232` `if (split > kAttnMaxSplit) throw` | the refuse-to-truncate check | message tail corrected: one constant moves both halves |
| 7 | `engine/decode_step.cpp:288` `sc.a_part = alloc<float>(kernels::attn_part_floats(wave_rows))` | where the scratch is actually allocated | follows #5 automatically, no edit |
| 8 | `engine/decode_step.h:174` `float* a_part; // [wave_rows][8][kAttnMaxSplit][258]` | shape comment — names the **symbol**, and its `8` is `N_Q` not the split | no edit needed (checked: it cannot go stale) |
| 9 | `engine/gate.py:396-436` `lane_geometry()` / `lane_ceiling()` / `lane_ceiling_note()` | the gate's ceiling: greps `#define MAXSC (\d+)` from #1 and `constexpr int kAttnMaxSplit\s*=\s*(\d+)` from #3 | **no literal exists here**, so nothing moved with the constant — see §4.3 for the ±1 re-proof |
| 10 | `engine/gate.py:6635` (smoke) and `:6951` (fixture run) | `check(lanes <= lane_ceiling(), …)` | untouched; now adjudicates against 4608 |
| 11 | `engine/decode_run.cpp:64` `kSessionDefaultCap = 4096` | a **slot** count (KV capacity), not the lane ceiling — but it equalled the old ceiling and its comment justified itself by "attention refuses long before 4096" | comment corrected (that claim is now false: 4608 > 4096). **Value left at 4096 — a decision for you, §6** |
| 12 | `engine/prefill.cpp:50-57` | wave-split loop bound; comment says "the lane ceiling stops there … Raising the lane ceiling is P7's, not this file's" | untouched; no literal, still true (wave bound is `sc.wave_rows` = 512) |
| 13 | `engine/kernels.h:400-410`, `:457-460` | `AttnLayerArgs::part` docs + `attn_part_floats` declaration; ceiling described as a product | untouched, no literal |
| 14 | `prototype/attn_decode/attn_decode.hip.cpp:154-160, 355-364` | the op's own host harness `part` sizing, from `--split N` | untouched — **the prototype has no cap at all**; the cap is engine-only |
| 15 | `prototype/attn_decode/attn_decode_kernels.h:523-526` `tile_part_floats()` | the tiled form's own footprint claim, `nrows*N_Q*split_for_lanes(lanes)*(DH+2)` | untouched; derived, cap-free |
| 16 | `prototype/attn_m2/attn_m2_kernels.h:191` `ATTN_M2_MAXSPLIT 8` | a **mirror in the parked M=2 prototype (item 2.2)** whose `ATTN_M2_STAGE`/`VSLOT` derive from it, and whose comment claimed it *was* the engine's constant and cited `engine/kernels_attn.cpp:60` | **value left at 8** (2.2 is not on the shipping path and its gate numbers were taken there); the false mirror claim and the rotting line cite were rewritten into a symbol cite |
| 17 | `prototype/attn_tile_model.py:371-376` `part_offset`, `:338-341` `split_for_lanes` | the CPU mirror's slot expression and split deriver (parses #1) | untouched; used as the third-party witness in §3 |
| 18 | `prototype/attn_m2/m2_model.py:905-907, 1105-1144` | enumerates splits up to `ATTN_M2_MAXSPLIT` (#16) | untouched, consistent with leaving 2.2 at 8 |
| 19 | **the plan / wave bill** | `launchstats::attn().record(rows)` at `kernels_attn.cpp` end-of-site, and `stage_clock.h`'s `WaveShape.bodies = ceil(tokens / wave_rows)` | **unaffected.** `split` deepens a *grid*, never a launch site: `attn_combine` is an extra kernel inside the same one recorded site, and `wave_plan_bill`'s "split" is wave bodies. No bill line moves |
| 20 | `static_assert` census | `attn_decode_kernels.h:500-513` (`ATTN_TILE_M == 1`, `kTileWarps == ATTN_TILE_G`, `ATTN_TILE_L >= TILE`, `kTileScoreElems <= 32768`, `kTileLdsBytes <= 65536`), `attn_slim_kernels.h:102`, `kernels_embed.cpp:44`, `kernels_gdn.cpp:33` | **none of them involves the split count or the cap**; all are staging/LDS/trip-count bounds and none needed moving. No new assert added: a `static_assert(split <= kAttnMaxSplit)` would be decoration, because the bound is a runtime property of the request and `attn_decode_layer`'s throw is where it is enforced |
| 21 | `tools/dump_fixtures.sh:229-237` | prose asserting "caps at MAXSC 512 x kAttnMaxSplit 8 = 4096" and "the decode-bearing fixture cannot run today" | rewritten to the new state (that fixture is now the judge, §5) |

`tools/prefill_row_rooms.py:58` and `tools/reference_dump.py:142` mention `MAXSC` only — untouched.

## 2. Step 2 — sizing, honestly, including the long-context target

`attn_part_floats(kMaxWaveRows = 512)`, i.e. the `a_part` scratch one wave costs:

| cap | lane ceiling `MAXSC × cap` | bytes `512 × 8 × cap × 258 × 4` | MB / MiB | % of the 1.5457 GB bf16 checkpoint |
|---|---|---|---|---|
| 8 (was) | 4096 | 33,816,576 B | 33.82 MB / 32.25 MiB | 2.19 % |
| **9 (now)** | **4608** | **38,043,648 B** | **38.04 MB / 36.28 MiB** | **2.46 %** |
| 40 (20480 lanes) | 20,480 | 169,082,880 B | 169.08 MB / 161.25 MiB | 10.94 % |

* cost of the shipped raise: **+4,227,072 B** (one `a_part` per wave scratch, reused by every body).
* **A correction to the brief's arithmetic:** the 20480-lane row is `512 × 8 × 40 × 258 × 4 B =
  169,082,880 B = 161.25 MiB`, not ~213 MB. Nothing reproduces 213 MB from that product — 213 MB needs
  split ≈ 50 (`213e6 / (512*8*258*4) = 50.4`). The number is stated in the code comment as
  169,082,880 B so the next reader does not inherit the wrong one.
* **Nothing else scales with the cap.** Checked individually: `kEmbedIdSlots` (the §02.3 gather ring) is
  **per body, not per split** — 8 slots × 512 rows × 4 B = 16 KiB device + 16 KiB pinned, and its
  `hipEventSynchronize` wait path is already exercised today (a `wide1024f` run is 2 prefill bodies + 8
  generation bodies = 10 > 8), so the comment at `decode_step.h:207` stands. Per-block LDS is MAXSC-based
  and unchanged. KV is per slot (4105 slots = 48.1 MiB). `--prefill-logits all` cost is per **row**, not
  per split, and is already paid at 4096 rows (1.9 GiB in BF16).
* **9 is minimal and the ceiling is a multiple of 512 by construction** (`split` is an integer), so
  admitting 4104 lanes necessarily admits up to 4608. That means widths 4105..4608 are *legal but
  unpinned*: this is a capacity bound, not a numerics claim about them. If you want the admitted band to
  equal the pinned band exactly, the only way is a second, uglier constant (`kAttnMaxLanes`), which is a
  decision rather than a detail.
* **split 40 is not funded and this raise does not imply it.** 20480 lanes needs a 20k-token pinned
  reference, which D41 already flagged as a deliberate step (an eager 20k pass materialises ~13 GB of
  attention per layer on CPU). No pin reaches it; the constant stays where the evidence is.

## 3. Step 3 — one address expression, shared (results.md §20.1)

§20.1's bug was two self-consistent layouts joined by a branch. Nothing in this raise touches an
address, and here are all five spellings of it as they now stand, with the allocation:

```c
// attn_core, the shipped per-(row,head) form — prototype/attn_decode/attn_decode_kernels.h:382
float* p  = a.part + (((size_t)qrow * N_Q + head) * a.split + blockIdx.y) * (DH + 2);

// attn_tile_body, the §20 tiled wave form — prototype/attn_decode/attn_decode_kernels.h:705
float* pp = a.part + (((size_t)row  * N_Q + head) * a.split + blockIdx.y) * (DH + 2);

// attn_slim_wave, the shipping default touch form — engine/attn_slim_kernels.h:337
float* pp = a.part + (((size_t)row  * N_Q + head) * a.split + blockIdx.y) * (DH + 2);

// attn_combine, THE READER — prototype/attn_decode/attn_decode_kernels.h:728-733
const float* p = a.part + (((size_t)qrow * N_Q + head) * a.split) * (DH + 2);
    …  p[(size_t)s * (DH + 2) + 0 / 1 / 2 + d]   for s in [0, s_row),  s_row = blocks_for_row(a, qrow)

// the allocation — engine/kernels_attn.cpp:178
return static_cast<size_t>(rows) * N_Q * kAttnMaxSplit * (DH + 2);

// the CPU mirror's own, used as the third authority — prototype/attn_tile_model.py:376
((row * N_Q + head) * split + window) * (DH + 2)
```

The writer and the reader are the same function `((row * N_Q + head) * split + block) * (DH + 2)`: keyed
by the **full** head (so the two KV groups own different slots), strided by the **launch's** `split`. The
reader's base is that expression at `block = 0` and its per-block stride is `(DH + 2)`, so it reads
exactly the blocks the grid can write (`blockIdx.y < split`, `s_row == split` for a streaming row).
**No line above contains `kAttnMaxSplit`,** and that is the whole reason the raise is bit-identity-safe:
buffer containment follows from `split <= kAttnMaxSplit`, which is enforced by the throw at
`kernels_attn.cpp:181`, not by the address.

`attn_combine` still has one merge and does not branch on `tile_kernel` (§20.1's fix), and this commit
adds no branch and no second layout.

## 4. Step 4 — the admission: bit-identity below the old ceiling

### 4.1 CPU witness (no device, no timing)

`ceiling_proof.py` (this artifact dir; run
`.venv/bin/python <path>/ceiling_proof.py <repo>`, 0.15 s) reads the two constants with **gate.py's own
regexes**, parses the four C++ address expressions and the mirror's with `ast`, and evaluates them.
Every claim is paired with an injection that must turn it red. Result: `PROOF PASS`.

```
[A] census of `kAttnMaxSplit` in engine/: allocation=1, comment=8, declaration=1, message=2, refusal=1
    -> functional uses are exactly the declaration, the allocation and the refusal;
       address expressions reading the cap: 0 (must be 0)
[B1] three writers + the CPU mirror parse to ONE AST; reader base == writer slot at window 0;
     reader stride `(DH + 2)` per block, read off `p[(size_t)s * (DH + 2) + 0]`
[B2] writer slot set == reader access set: 16/16 streaming plans (rows 1,4 x splits 2..9)
     must-catch (§20.1 group-collapsed writer re-injected): caught in 3/3 shapes
     containment: max slot + 1032 B <= attn_part_floats(rows) at rows 1/8/512 x caps 8/9/40 — no overrun
[B3] 6496 slot evaluations at the split boundaries of every split cap 8 admitted:
     offsets that move between cap 8 and cap 9 = 0
     must-catch (`* kAttnMaxSplit` injected into the writer text): the caps then differ = True
[C]  newly admitted: 4097..4608 (512 widths), all split 9; legal at 8 and refused at 9: 0
       4096 -> split  8 legal | 4097 -> split 9 legal | 4104 -> split 9 legal | 4608 -> split 9 legal
       4609 -> split 10 REFUSED | 20480 -> split 40 REFUSED
```

Why this is the right instrument: `wide4096f` at 4104 lanes is new ground and cannot witness "nothing
below the old ceiling moved", and the small fixtures can only show that *their* rows are unchanged. The
general claim is about which quantities the constant is allowed to reach, and that is checkable without
a device: the cap reaches the allocation and the refusal and nothing else, so for every plan the old
ceiling admitted the `split`, the grid, the slot addresses and therefore the bits are identical, and the
buffer is a strict superset whose extra tail nothing writes.

### 4.2 device gates — three small fixtures, one at a time, under `tools/gpu_lock.sh`

`engine/gate.py` does not self-lock (`grep -c gpu_lock engine/gate.py` → 0), so the lock was held around
each. Run blocking, one subagent, correctness only, **no timing taken**, no `ms` figure reported. Each
was run on the first build and again on the final (line-neutral) build; the wall times below are the
final ones. `util` is reported as read and discounted (`AGENTS.md`: flash_serve polls at ~90 %).

| fixture | command (each inside its own acquire/release) | rc | wall | `util` at acquire | verdict |
|---|---|---|---|---|---|
| `dec64f` | `.venv/bin/python engine/gate.py --fixture dec64f` | 0 | 11 s | 93 % | `ENGINE GATE PASS — with 2 step(s) reported TIE` (the fixture's own standing ties), `[FAIL]` lines: 0 |
| `wide640f` | `.venv/bin/python engine/gate.py --fixture wide640f` | 0 | 34 s | 92 % | `ENGINE GATE PASS`, `[FAIL]` lines: 0 — **640 of 640 pinned prefill rows judged**, worst cos deficit 4.3696e-04 at row 2, worst meanrel 3.5902 % at row 436 |
| `wide1024f` | `.venv/bin/python engine/gate.py --fixture wide1024f` | 0 | 52 s | 92 % | `ENGINE GATE PASS`, `[FAIL]` lines: 0 — **1024 of 1024 rows judged**, worst cos deficit 4.3696e-04 at row 2, worst meanrel 3.5902 % at row 436 |

Full form of one of them, as run:

```
tools/gpu_lock.sh acquire "2.5 ceiling raise (final build): wide1024f gate"; tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture wide1024f
tools/gpu_lock.sh release "2.5 ceiling raise (final build): wide1024f gate"
```

Worth noting rather than glossing: those worst-deficit numbers are **identical to the figures D44
recorded for `wide4096p`**, at the same row indices. That is expected and it is a cross-check, not a
copy-paste — `wide640f` / `wide1024f` / `wide4096f` / `wide4096p` are one prompt at 640 / 1024 / 4096
lanes (`tools/dump_fixtures.sh:222-227` records `ids1024.txt` as a bit-exact prefix of `ids4096.txt`),
so rows 2 and 436 are the same rows of the same prompt, and their bits did not move under the raise.

Binary freshness (the D30 canary rule, since `engine/build.sh` is not bit-reproducible): `engine/build.sh`
does one whole-program `hipcc` compile with no object reuse, run after the final edit
(`engine/kernels_attn.cpp` 2026-10-18T01:35:50 → `engine/build/decode_run` 2026-10-18T01:36:20). The
**behavioural** canary for this change is not available at these widths — split 9 only happens above 4096
lanes — so the canary is yours: `wide4096f` must run at all, where D44 recorded it exiting 1. If you want
a device bit-identity A/B, `engine/build/decode_run.base` (pre-2.5, 2026-10-17 13:28) against the new
binary on `wide640f` with `md5sum` over the dumped logits is ~70 s, carrying the usual caveat that
`.base` is not `3d4d293`.

### 4.3 the gate's ceiling check, re-proved to bite at ±1

`gate.py` carries **no lane-ceiling literal**, so nothing had to move with the constant — it reads both
factors out of the enforcing files (that was the design, `gate.py:398-411`; the brief allowed a literal
to follow the engine, and there is none). `ceiling_gate_follows.py` (this artifact dir, no device, ~1 s)
proves the check is live rather than decorative, which is the ±1 re-proof §07's launch bill got in D39:

```
[1] gate.py::lane_geometry() -> (512, 9, 4608)
    check(4096 <= ceiling) = True | check(4104 <= ceiling) = True (the wide4096f plan)
    check(4608 <= ceiling) = True | check(4609 <= ceiling) = False
[2] same gate, same code, source reverted to 8 in a scratch copy: (512, 8, 4096)
    check(4096) = True | check(4097) = False | check(4104) = False   ← D44's device refusal, reproduced
WITNESS PASS
```

## 5. The two commands that are yours (I did not run them)

~3 minutes each, blocking, zero subagents, in your exclusive window. `gate.py` does not self-lock, so
hold the lock yourself. Both fixtures are on disk (`data/refs/wide4096f` 2.1 GiB, `wide4096p` 16 GiB) and
both manifests pin every prefill row, so `gate.py` raises `--prefill-logits all` itself before argv.

```
tools/gpu_lock.sh acquire "2.5: wide4096f gate 4104 lanes"; tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture wide4096f   ; # the plan must now be accepted: 4104 lanes
tools/gpu_lock.sh release "2.5: wide4096f gate 4104 lanes"

tools/gpu_lock.sh acquire "2.5: wide4096p gate re-take"; tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture wide4096p   ; # unchanged geometry, re-take under cap 9
tools/gpu_lock.sh release "2.5: wide4096p gate re-take"
```

What to look for, in this order (D44's lesson: read how many checks ran, not only how many failed):

1. the lane-ceiling line must now read `4104 lanes vs staging MAXSC 512 x kAttnMaxSplit 9 = 4608 lanes`
   and **pass**; if the run exits 1 again, the row sections never ran and the failure count means nothing;
2. **4104 rows judged** in the prefill-row section (640/1024/4096 came back with `N of N judged`; a
   `missing` count means the coverage forcing did not fire, not that the engine failed);
3. the decode steps at 4104 lanes are the first adjudication of the streaming route with a generated
   token above the old ceiling — that is the thing `wide4096p` could not say.

**If `wide4096f` fails on floors rather than on the plan, stop.** Do not widen, tune or special-case:
that would mean 4104 lanes is the first width the floors in force do not cover, which is a finding to
report, not a bug to route around. The revert is one line (`kAttnMaxSplit = 8`) plus the comment edits.
Anything that enters the ledger, every paired A/B, and the `bench/performance-status.md` rows are yours.

## 6. Decisions I did not make, and residual risks

1. **`kSessionDefaultCap` is still 4096** (`engine/decode_run.cpp:64`). It is a session's slot count, not
   the lane ceiling, but it used to sit exactly on the ceiling; now the ceiling (4608) is above it, so a
   `--session` run hits its cap first and `--cap` is how a session reaches the wider geometry. Raising it
   to 4608 costs +6 MiB of KV per request (12 KiB/token) and no gate asserts its value. That is a product
   default, not a mechanical follow-on, so I corrected the now-false comment and left the number to you.
2. **One rotted line-citation, and it is unavoidable.** `tools/citation_rot_check.py` now reports
   `ROTTED engine/kernels_attn.cpp:60 | cited by progress/2026-10-16/prefill-decision-log-2026-10-16.md:16`
   ("`engine/kernels_attn.cpp:60` sets `kAttnMaxSplit = 8`"). That cite was already 5 lines off at
   `3d4d293` (the constant is at :65), and the sentence it lands on is the sentence this change falsifies,
   so no wording keeps it honest — it needs an archive re-point, which is your call (`progress/` is frozen
   and my brief excluded it). I kept every edited file line-count neutral precisely so this is the *only*
   rot: at the first attempt (+5 lines) the checker reported four, and the other three were pure line
   shift. The one live rotting cite in code — `prototype/attn_m2/attn_m2_kernels.h:191`'s
   `engine/kernels_attn.cpp:60` — I converted to a symbol cite, so it cannot rot again.
3. **`engine/results.md` has no entry for this yet** (P7's note at `:3300` and the P8 note at `:3413`
   both predicted exactly this raise and both still read as history). Journal prose is commit-time writing
   and yours; flagging rather than writing it, since the brief asked for small targeted edits and no
   journal work.
4. **The band 4105..4608 is legal and unpinned** (§2). Nothing runs there today. It is a capacity bound,
   not an admitted numerics claim; the only alternative is a second constant, which is a decision.
5. **`wide4096f` numerics at 4104 lanes are unjudged until you run §5**, so the raise is not yet a
   completed admission — it is the plan check passing plus the structural bit-identity proof below the
   old ceiling. If it goes red, §5's stop rule applies.
6. `speccheck/` was neither touched nor run: nothing here can reach `specimpl.py` or a spec gate
   (`AGENTS.md` budget rule). The `prototype/attn_tile_model.py selftest` (CPU, 13/13 ok) and
   `prototype/attn_m2/*` were run/inspected only to confirm they parse these headers; `attn_tile_model
   gate --negative` was deliberately **not** re-run — its inputs (the two attention headers) are
   untouched by this diff, so it could not have changed colour.
7. The two CPU witnesses are scratch files in this artifact directory. If you want them permanent, the
   natural home is `tools/attn_ceiling_proof.py` (they are self-contained, CPU-only, and each carries
   its own must-catch injections); that is a new-tracked-tool decision I did not make for you.

## 7. Changed files and commands

Changed (working tree, nothing staged, no commits):
`engine/kernels_attn.cpp`, `engine/decode_run.cpp`, `prototype/attn_m2/attn_m2_kernels.h`,
`tools/dump_fixtures.sh`.

Commands run, in order:

```
git log --oneline -3 ; git status --short                          # base 3d4d293, clean tree
grep -rn "kAttnMaxSplit|lane_ceiling|MAXSC" engine prototype tools # enumeration, §1
python3 …/ceiling_proof.py .            # first version, FAIL then PASS (injections reproduced)
engine/build.sh                          # full hipcc rebuild after the final edits
python3 …/ceiling_proof.py .             # PROOF PASS  (§4.1)
.venv/bin/python …/ceiling_gate_follows.py .   # WITNESS PASS (§4.3)
.venv/bin/python prototype/attn_tile_model.py selftest   # 13/13 passed (CPU mirror, headers re-parsed)
tools/gpu_lock.sh … ; .venv/bin/python engine/gate.py --fixture dec64f      # PASS, 11 s, util 93 %
tools/gpu_lock.sh … ; .venv/bin/python engine/gate.py --fixture wide640f    # PASS, 34 s, util 92 %
tools/gpu_lock.sh … ; .venv/bin/python engine/gate.py --fixture wide1024f   # PASS, 52 s, util 92 %
.venv/bin/python tools/citation_rot_check.py   # 1 rotted cite, explained in §6.2
.venv/bin/python tools/doc_ref_check.py        # no BROKEN-PATH attributed to a touched file
tools/gpu_lock.sh status                          # free
```
