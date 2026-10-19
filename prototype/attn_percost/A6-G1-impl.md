# A6-G1-impl — G1 (Q un-staged + 8-row epilogue) graduated into `engine/` behind `--attn-fattn-qstage stage|unstaged8`

Written 2026-10-20 by the A6-G1 worker, on tip `42c8e09`, against `prototype/attn_percost/A6-BET.md` (the spec:
G1's bar, its numerics policy, G1-M1), `prototype/attn_percost/A6-G0-verdict.md` (the falsifier came back LIVE:
7 warps = `attn` family **−10.82 %**, so the residency thesis holds and G1 is funded),
`prototype/attn_percost/A6-result.md` §1 (compile-side G1: **185 VGPR / spill 0 / private 0 / group_segment
20,480 B**, driver grants 3 blocks ⇒ 9 wf/CU) and `prototype/attn_percost/A5-GEOMETRY.md` §3.2/§6.

**This file is a live checkpoint.** It is appended as each milestone lands, so a compaction cannot lose work.

## 0. Status board

| milestone | state |
|---|---|
| engine implementation (knob, `if constexpr`, launcher, echo, `gate.py` passthrough) | **DONE** — four kernel instantiations, launcher selects by the two knobs, `q=`/`lds=`/`wf_per_cu=`/`vgpr=` echoed from the function that ran, `gate.py` argv passthrough only. Wiring listed in §6; `engine/build.sh` green; note-block table in §2 |
| compile-side note-block read on the built tree | **DONE — all four instantiations reproduce A6-result §1.3 to the byte**, see §2 below |
| **G1-M1 bit-identity device gate** (default vs HEAD-built; `unstaged8` vs same baseline; ≥ 2 shapes incl. a ragged tail) | **GREEN — byte-identical on 640, 1,111 (ragged tail) and 2,000 ids, logits AND `STATEFP`, both comparisons.** Full evidence in §3 |
| `a6_epi8_model.py`'s coverage assertion kept as a test beside the tree, `--negative` included | **DONE — and it is now a test OF the shipping header, not of a patcher's output**: geometry read from `engine/attn_fattn_kernels.h`'s own `#define`s, 19 index expressions + 2 coverage asserts required verbatim, `--header` for its own negative control. See §4 |
| `prototype/attn_percost/run_ab_a6g1.sh` handed over (never run beyond `--dry-run`/`PARSE_CHECK=1`) | **DONE — `DRY_RUN=1` and `PARSE_CHECK=1` both green (6 scenarios incl. the NULL leg); no device touched, no timing taken.** See §5 |

**No timing figure may ever appear in this file.** The main agent owns every measurement
(`AGENTS.md`, "GPU execution is serialized and belongs to the main agent").

## 1. Design decisions taken before the first edit

- **Two template axes on one body.** `attn_fattn_body` becomes `template <bool kTwoAcc, bool kQUnstaged8>`, so
  the four spellings are four `__global__`s in the same TU: `attn_fattn_wave` (shipped: single roll, Q staged),
  `attn_fattn_wave_two` (A4-L1), `attn_fattn_wave_u8` (G1), `attn_fattn_wave_two_u8`. Templated, not a runtime
  `if`, for the reason `84f35f8` gave for the roll axis: a runtime branch makes both operand sets live in every
  build and moves the register count of the DEFAULT path, which would stop the default being HEAD's machine code
  and therefore stop "default is inert" from being a byte-compare. The two knobs stay orthogonal and no
  combination is a dead route.
- **ONE per-warp buffer, sized by the spelling.** The shipped form parks Q in it (16 rows × DH F16 = 8,192 B per
  warp) and the epilogue reuses it as the F32 transpose stage; the un-staged form parks only the transpose
  (FATTN_TRG = 8 rows × FATTN_TROW F32 = 4,096 B per warp). Two separate `__shared__` arrays would **sum** in
  the block's LDS and move the default's declared 32,768 B — the exact number the residency claim and the
  banner's `lds=` field are about — so it is one array whose element count is a constexpr expression in the
  template parameter, measured in halves (2 halves per transpose float).
- **One address expression each.** `fattn_tr_addr(row_in_pass, d) = row_in_pass * FATTN_TROW + d` becomes the
  only spelling of a transpose-stage slot for BOTH epilogues (the shipped 16-row two-pass form passes
  `row_in_pass = fattn_c_row(lane)`/`srow`, which is its current expression verbatim), and
  `fattn_score_q_tile<kQUnstaged8>` is the only spelling of the Q operand tile for both Q spellings. The
  aliasing class recorded in `prototype/attn_decode/results.md` §20.1 is the failure of the alternative: two
  self-consistent layouts joined by a branch, each agreeing with itself, serving every streaming row half of the
  other group's answer.
- **The epilogue's 4 passes are half-major** (`half = pass / FATTN_NGRP`, `rg = pass % FATTN_NGRP`), so a float's
  destination order is the shipped one and only the staging depth moves.
- `gate.py`: argv passthrough + NUMERICS-echo assertion only, mirroring `--attn-fattn-roll`. No band, floor,
  threshold or admission edits; `ATTN_FORM_FLAGS` is untouched (the roll knob is not in it either).

## 2. The note block of the graduated tree — G1-M0 re-read on `engine/`, not on a quarantined copy

CPU only, no device, no timing:

```sh
mkdir -p /tmp/a6g1nb && cd /tmp/a6g1nb && export REPO=$PWD        # REPO = repo root
hipcc -O3 -Wno-unused-command-line-argument -I"$REPO" -I"$REPO/engine" --save-temps \
    -c "$REPO/engine/kernels_attn.cpp" -o k.o
cd "$REPO" && python3 prototype/attn_wmma/fattn_residency_read.py \
    --asm /tmp/a6g1nb/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s --all
```

The four `attn_fattn_wave` instantiations, read out of the object the shipping TU now builds, against the
compile-side table of `prototype/attn_percost/A6-result.md` §1.3:

```
  kernel                  vgpr  spill  sgpr  private  group_segment  max_flat  this file   A6-result §1.3
  attn_fattn_wave          178     0     24       0         32768         96     (default)  178 / 0 / 0 / 32768
  attn_fattn_wave_two      200     0     24       0         32768         96     (roll two) 200 / 0 / 0 / 32768
  attn_fattn_wave_u8       185     0     27       0         20480         96     **G1**     185 / 0 / 0 / 20480
  attn_fattn_wave_two_u8   221     0     26       0         20480         96               221 / 0 / 0 / 20480
```

Four things this witnesses, and the first is the one the default side owes:

- **The default did not move.** `attn_fattn_wave` is `178 / 0 / 0 / 32768`, the same four numbers §33's banner
  and `A6-result` §0 record for HEAD, so adding the second template axis did not touch the shipped kernel's
  register or LDS budget — which is the whole reason the axis is `if constexpr` and not a runtime branch. (A
  note-block match is not byte-identity of the machine code; that is what device leg 1 is for.)
- **The graduation is faithful to the quarantined variant.** `attn_fattn_wave_u8` lands on `185 VGPR /
  spill 0 / private 0 / group_segment 20,480 B` — every field equal to the compile-side G1 that `A6-BET`'s
  G1-M0 line admitted — and the `two_u8` cell reproduces `A6-result` §1.3's `221 / 0 / 0 / 20480` too, so the
  match is not one coincidence in one cell.
- **`attn_fattn_wave_two` is untouched at 200 / 32768**, i.e. A4-L1's admitted side did not move when the qstage
  axis was added beside it.
- The other six kernels of the TU (`attn_core`, `attn_slim_wave`, `attn_tile_wave`, `attn_prep`, `attn_combine`,
  `rope_table`) are unchanged from their pre-edit values in the same listing (12288 / 12288 / 38944 / 2080 /
  0 / 0 of group segment, vgprs 47/47/53/30/10/16) — the census names its population: 10 kernels in this TU
  now, 4 of them `attn_fattn_wave` instantiations.

Derived, with its divisor read rather than quoted (`/sys/.../kfd/topology/nodes/1/properties`, `lds_size_in_kb
64`): `floor(65536 / 20480) = 3` blocks/CU × 3 warps = **9 wf/CU** against the shipped
`floor(65536 / 32768) = 2` × 3 = 6. The driver's own grant was already read by `A6-result` §3 (3 blocks at
185 VGPR) and is not re-derived here.

## 3. G1-M1 — the bit-identity gate: **GREEN**, and it is the hard one

`A6-BET` owes G1 byte identity with the shipped default, and this is the device witness for it. Harness
`prototype/attn_percost/a6_g1_bits.sh`; logs and dumps in `/tmp/a6g1-bits/`. Baseline is a `git worktree` at the
pinned pre-change sha `ed658fe` (`/tmp/a6g1-bits-base-ed658fe`), built by that worktree's own `engine/build.sh`;
`ed658fe` **is the tip the change was made on** (`git rev-list --count ed658fe..HEAD` = 0), so “base vs stage” is
literally HEAD-built versus the changed default, not an older tree compared with a newer one. The two changed
sides are the working tree's `engine/build/decode_run`, run with the knob unset and with
`--attn-fattn-qstage unstaged8`. Three sides per shape, and the two comparisons are different claims: `base` vs
`stage` is default-inertness, `base` vs `u8` is the bet. Comparing only `stage` against `u8` would pass if both
had moved off HEAD together, and comparing only `base` against `stage` would pass if the knob were a no-op.

**The commands, verbatim, with what AGENTS requires of a worker device run.**

```sh
BUILD_ONLY=1 bash prototype/attn_percost/a6_g1_bits.sh   # CPU only: builds the base worktree, reads note blocks
bash prototype/attn_percost/a6_g1_bits.sh                # shapes 640 and 1,111 — THE GATE
SHAPES="2000" bash prototype/attn_percost/a6_g1_bits.sh  # the extra wide/ragged shape, run as its own window
```

Device legs ran blocking, foreground, one subagent, under one `tools/gpu_lock.sh` hold each (TAG `a6g1-bits`,
acquired and released by the script). `util` read **43 % at acquire / 29 % at release** for the 640+1,111 window
and **45 % / 15 %** for the 2,000 window; the script's wall was **25 s** and **25 s** (correctness legs — six and
three `decode_run` invocations, `--steps 0 --prefill-logits all --state-fp`, no decode steps, no bursts, no
repetitions for timing). **NO TIMING WAS TAKEN: no `ms` figure exists in this section, and none may be copied
from it into a timing document.**

**The verdict, verbatim out of the two runs:**

```
CANARY 1 ok: base attn_fattn_wave 178 0 0 32768 | new attn_fattn_wave 178 0 0 32768 | new attn_fattn_wave_u8 185 0 0 20480   (vgpr spill private group, read off the LINKED binaries)
WINDOW a6g1-bits util@acquire=gpu use: 43%
LEG 640 base:  q=stage     lds=32768 wf_per_cu=6 vgpr=178 fattn_rows=128/128 plan_attn_launches=6 statefp_rows=44
LEG 640 stage: q=stage     lds=32768 wf_per_cu=6 vgpr=178 fattn_rows=128/128 plan_attn_launches=6 statefp_rows=44
LEG 640 u8:    q=unstaged8 lds=20480 wf_per_cu=9 vgpr=185 fattn_rows=128/128 plan_attn_launches=6 statefp_rows=44
BYTE-IDENTICAL 640: base vs stage — 641 dumped files equal, 44 STATEFP rows equal, attn plan bill equal (6), coverage fields equal (vtiles=16 wave_rows=512 split=2 fattn_rows=128/128 fallback_rows=0 covered_rows_pct=100.0 covered_work_pct=100.0 )
BYTE-IDENTICAL 640: base vs u8 — 641 dumped files equal, 44 STATEFP rows equal, attn plan bill equal (6), coverage fields equal (… identical to the line above)
LEG 1111 base:  q=stage     lds=32768 wf_per_cu=6 vgpr=178 fattn_rows=512/512 plan_attn_launches=6 statefp_rows=44
LEG 1111 stage: q=stage     lds=32768 wf_per_cu=6 vgpr=178 fattn_rows=512/512 plan_attn_launches=6 statefp_rows=44
LEG 1111 u8:    q=unstaged8 lds=20480 wf_per_cu=9 vgpr=185 fattn_rows=512/512 plan_attn_launches=6 statefp_rows=44
BYTE-IDENTICAL 1111: base vs stage — 1112 dumped files equal, 44 STATEFP rows equal, attn plan bill equal (6), coverage fields equal
BYTE-IDENTICAL 1111: base vs u8   — 1112 dumped files equal, 44 STATEFP rows equal, attn plan bill equal (6), coverage fields equal
util@release=gpu use: 29%
A6-G1-M1 PASS: default-inert and unstaged8 byte-identical on shapes [640 1111] (logits files + STATEFP), with the geometry canaries proving the unstaged8 kernel is the one that ran.
PROTOCOL: command=bash prototype/attn_percost/a6_g1_bits.sh SHAPES="640 1111"; wall=25s (script wall, correctness only); NO TIMING TAKEN
```

and the wider shape:

```
LEG 2000 base:  q=stage     lds=32768 wf_per_cu=6 vgpr=178 fattn_rows=512/512 plan_attn_launches=6 statefp_rows=44
LEG 2000 stage: q=stage     lds=32768 wf_per_cu=6 vgpr=178 fattn_rows=512/512 plan_attn_launches=6 statefp_rows=44
LEG 2000 u8:    q=unstaged8 lds=20480 wf_per_cu=9 vgpr=185 fattn_rows=512/512 plan_attn_launches=6 statefp_rows=44
BYTE-IDENTICAL 2000: base vs stage — 2001 dumped files equal, 44 STATEFP rows equal, …
BYTE-IDENTICAL 2000: base vs u8   — 2001 dumped files equal, 44 STATEFP rows equal, …
PROTOCOL: command=bash prototype/attn_percost/a6_g1_bits.sh SHAPES="2000"; wall=25s; NO TIMING TAKEN
```

**What was compared, and why each half can fail.** “Byte-identical” here is not a cosine: `diff -rq` over every
file the run dumped into `--out` (one raw F32 `logits<row>.bin` per prefill row — 641 / 1,112 / 2,001 of them,
which is every row of the prompt including the ragged tail — plus `sampled.txt`), and `cmp` over the 44 `STATEFP`
rows (`--state-fp`, the §08.6 fingerprints of the GDN state, the conv window and the KV cache, read out of the
run’s own stdout). File-count equality is asserted before the diff, because a comparison over a missing file is a
comparison over nothing.

**Why this is not vacuous** — the part that makes the green mean something. A kernel that never ran satisfies byte
identity trivially, and this tree has been bitten by that class three times (AGENTS’ `partalias` control scoring 0
of 1,024, the CPU mirror that agreed with itself, the empty-parse gate that printed PASS). So every leg had to
print, in its own log:

- `q=unstaged8` on the u8 side and `q=stage` on the other two — the knob named the spelling that ran;
- **`lds=20480 wf_per_cu=9` on the u8 side against `lds=32768 wf_per_cu=6` on the staged sides**, read by
  `hipFuncGetAttributes` / `hipOccupancyMaxActiveBlocksPerMultiprocessor` off **the function this launch
  submitted** (`fattn_echo`’s `launched` pointer, `engine/kernels_attn.cpp`), not off a name typed beside the
  print. If those two fields agreed, the identity below would be one run compared with itself;
- `vgpr=185` on the u8 side and `vgpr=178` on the staged sides, each agreeing with the **note block of the linked
  binary** (CANARY 1) — the stale-build guard recorded in `bench/performance-status.md` §44, since
  `engine/build.sh` is not bit-reproducible and every check compiled into a binary agrees with itself;
- `fattn_rows=` **strictly greater than zero** on every side (128/128, 512/512, 512/512): a wave of only one-pass
  rows launches no WMMA kernel, and byte identity over rows this kernel never wrote would prove nothing;
- all seven coverage fields (`vtiles wave_rows split fattn_rows fallback_rows covered_rows_pct
  covered_work_pct`) equal across the three sides, collected as a 7-line file and `cmp`-ed — `fattn_covers` does
  not read a staging plan, so a difference there would mean the knob changed ROUTING and the byte comparison
  would be judging different rows;
- the planned attn launch bill equal (6) on every side.

**What this establishes.** G1’s premise holds on device: an un-staged Q operand plus a 4-pass, 8-row-deep F32
transpose publishes **the same bytes** — every prefill row’s logits and every state fingerprint — on three shapes
including a ragged tail, while the block that produced them declares 20,480 B and runs 9 wf/CU instead of 32,768 B
and 6. It also establishes that the default is inert at the byte level, not merely by the `if constexpr` argument.
What it does **not** establish is that the change is faster: that is `A6-BET`’s G1 bar, adjudicated by
`prototype/attn_percost/run_ab_a6g1.sh` in a main-agent window, and nothing here speaks to it. Nor does it speak to
the two-accumulator roll crossed with this knob (`attn_fattn_wave_two_u8` compiles at 221 / 0 / 20480 and is
selectable by naming both knobs, but this gate ran the default roll only).

## 4. G1-M2 — the epilogue witness is now a test of the shipping header

Before the graduation `a6_epi8_model.py` enumerated the index arithmetic of a *patcher’s* output. That would have
been the mirror-that-agrees-with-itself failure from the moment the code moved into `engine/`: a model of a
remembered source cannot notice the source moved. So the file was rebuilt as a test of
`engine/attn_fattn_kernels.h` in three layers, CPU-only, no device, no timing:

1. **Geometry is read, not typed.** `DH`, `FATTN_MROWS`, `FATTN_TROW`, `FATTN_TBLK`, `FATTN_PBLK`, `FATTN_TRG`,
   `FATTN_NGRP`, `FATTN_NPASS`, `FATTN_LBLK`, `FATTN_KTILES` are parsed out of the header’s own `#define` lines and
   resolved in dependency order (C integer division, since that is what the device does). A model that hard-codes
   `TRG = 8` cannot see the header change it.
2. **Source agreement.** 19 index expressions and both `static_assert` coverage lines must appear **verbatim** in
   the header — the two publish splits (`lane >> 1` × `FATTN_PBLK` versus `lane >> 2` × `FATTN_LBLK`), the single
   `fattn_tr_addr`, each form’s stage write and stage read, the pass/half/row-group decomposition, and the `d`
   expression shared by both. If the C++ is re-worded without re-deriving the model, the test **refuses** instead
   of quietly adjudicating a geometry the engine no longer has.
3. **The three claims**, over all 16 row-slots × 256 dims of a warp: coverage (every destination written exactly
   once, in each form), identity (a shared destination is fed by the *same* `(lane, t, l)` accumulator element in
   both forms — which is what makes the published bytes the shipped ones whatever the stage layout), and bounds
   (no index leaves its own stage, which is the half-depth trap: an old-geometry index in a 4,096 B stage walks
   into the next warp’s stage).

Green, with the population named:

```sh
python3 prototype/attn_percost/a6_epi8_model.py          # PASS — 4096/4096 destinations, identity 4096/4096
python3 prototype/attn_percost/a6_epi8_model.py --negative  # NEGATIVE-CONTROL PASS (3 of 3 injections caught)
```

**Its own checks have teeth, witnessed rather than asserted.** `--negative` injects three ways the G1 mapping could
have been transcribed from the shipped one and be wrong — publish `FATTN_PBLK` blocks instead of `FATTN_LBLK`, use
the shipped 2-lanes-per-row split in a pass that holds four, or index the stage by absolute row-slot. All three
change the map (4,096 / 2,048 / 2,048 bound-or-coverage faults respectively). And the SOURCE AGREEMENT layer was
witnessed the same way, on mutated copies reached through `--header`:

```sh
sed 's/const int srow = row0 + (lane >> 2);/const int srow = row0 + (lane >> 1);/' engine/attn_fattn_kernels.h > /tmp/a6g1nb/neg/srow.h
sed '/int fattn_tr_addr(int row_in_pass, int d)/d'                                       engine/attn_fattn_kernels.h > /tmp/a6g1nb/neg/noaddr.h
sed 's|/ 32)|/ 16)|' on the FATTN_LBLK line                                              engine/attn_fattn_kernels.h > /tmp/a6g1nb/neg/lblk.h
python3 prototype/attn_percost/a6_epi8_model.py --header /tmp/a6g1nb/neg/srow.h    # exit 1 — spelling gone
python3 prototype/attn_percost/a6_epi8_model.py --header /tmp/a6g1nb/neg/noaddr.h  # exit 1 — spelling gone
python3 prototype/attn_percost/a6_epi8_model.py --header /tmp/a6g1nb/neg/lblk.h    # exit 1 — reads LBLK=8, coverage fails
```

The third mutation matters most: it proves the geometry really is read from the header (the run prints
`LBLK=8` from the mutated file) and that the coverage claim fails when the header’s arithmetic stops agreeing
with itself — a check that cannot fail is the failure this tree keeps recording.

## 5. G1-M3 — the A/B harness handed over: `run_ab_a6g1.sh` (+ `mk_a6g1_parsecheck.py`)

**Not run against the device.** Exercised only as `DRY_RUN=1` and `PARSE_CHECK=1`, both of which touch no device,
build nothing and take no lock. No `ms` figure from a synthetic fixture belongs in this file or in a report; the
fixtures exist to colour-test a parser.

Three sides, one window, one lock hold: `base` (worktree at the pinned pre-change sha `ed658fe`, built by its own
`engine/build.sh`), `stage` (the changed binary, knob UNSET — the NULL leg, A4-style) and `u8` (the same changed
binary with `--attn-fattn-qstage unstaged8`). `stage` and `u8` being one binary is deliberate: two rulers re-taken
in different weather are not an A/B.

The bar is A6-BET’s G1 bar, and the verdict is a conjunction printed with the reason:

- **LIVE** — `attn` family ≤ 0.8947 × base **and** the three ship gates (wall @4,096 ≤ 0.9435, 16,384 ≤ 0.9189,
  32,768 ≤ 0.8997) **and** the traced `attention` row ≤ 0.8991, i.e. agreeing in direction;
- **DEAD** — family > 0.9699 **or** a ship gate past its own dead line;
- **UNRESOLVED** — anything between: not shipped, not narrated.
- **UNPAIRED** — a refusal, not a verdict: the NULL leg moved > 2 % on the family row or any ship gate, so the
  harness prints no verdict at all and exits non-zero.

The traced row cannot decide DEAD on its own — a worse traced row beside a clearing family is a disagreement, which
is UNRESOLVED by construction. That is a deliberate choice and it is written in the script, because the alternative
lets one noisy row bury a real win.

What it refuses, and why each refusal exists: a binary whose **note block** and **banner** disagree (stale build);
a launch-count or coverage-field difference between sides (§41 closed launch count as a variable); an empty parse,
emphasised by `need`-ing every field and requiring all 7 coverage fields per log; a rep spread > 3 %; a wide prompt
whose id count is not the width it is labelled with; and a numerics leg that went red — bit-identity is taken
**before** the lock is acquired, via `a6_g1_bits.sh`, because that script self-locks and holding the lock around a
self-locking gate is the recorded deadlock.

`PARSE_CHECK=1` regenerates its fixtures from the harness’s own ratio constants (it refuses if a live line is not
below its dead line, or if the injected NULL move does not exceed `NULL_TOL`), then re-runs the harness in
`SUMMARIZE=1` mode over them and refuses unless each scenario prints the verdict named for it:

```
PARSE-CHECK ok: scenario live -> LIVE (expected LIVE)
PARSE-CHECK ok: scenario unresolved -> UNRESOLVED (expected UNRESOLVED)
PARSE-CHECK ok: scenario shipgate_blocks -> UNRESOLVED (expected UNRESOLVED)
PARSE-CHECK ok: scenario traced_disagrees -> UNRESOLVED (expected UNRESOLVED)
PARSE-CHECK ok: scenario dead -> DEAD (expected DEAD)
PARSE-CHECK ok: scenario null_unpaired -> UNPAIRED (expected UNPAIRED)
PARSE-CHECK PASS: the parser distinguishes LIVE / UNRESOLVED (family row, ship gate, traced disagreement) / DEAD / and refuses an unpaired NULL. This says nothing about the engine.
```

`shipgate_blocks` and `traced_disagrees` exist because G1’s verdict is a conjunction: a one-row rule can be
colour-tested row by row, a five-row rule needs a fixture in which each row is the *only* reason. `null_unpaired`
is the leg G0’s harness could not have, since G0 had no inert side.

**The commands the main agent owns** (all blocking, zero subagents, one lock hold — the harness takes and releases
`a6g1-ab` itself, so it must not be run from inside another hold):

```sh
DRY_RUN=1 bash prototype/attn_percost/run_ab_a6g1.sh        # audit the plan first; touches nothing
BASE=ed658fe bash prototype/attn_percost/run_ab_a6g1.sh     # the window: numerics, 3×3 legs @4,096, traced, 16k/32k
SUMMARIZE=1 OUT=<the window's OUT> bash prototype/attn_percost/run_ab_a6g1.sh   # re-judge the logs, no device
```

Externality (sum of `u8 − base` over every non-`attn` family) is printed at 4,096 / 16,384 / 32,768 against G0’s
**+74.72 ms at 16,384** and decides nothing; the script flags it if the 16k reading exceeds 20 ms, because G1 keeps
96-thread blocks and only drops LDS, so reproducing G0’s neighbour-slowing would itself be a finding.

## 6. The control-plane wiring, and what it is not allowed to do

Following the `--attn-fattn-roll` graduation (`84f35f8`) shape for shape:

| piece | where | what it does |
|---|---|---|
| `FattnQStage` enum | `engine/kernels.h` | `kFattnQStage = 0` default, `kFattnQStageUnstaged8 = 1`; the args struct carries `attn_fattn_qstage` |
| `parse_fattn_qstage` / `fattn_qstage_name` | `engine/kernels_attn.cpp` | **parse by refusal**: an unrecognised value aborts with the accepted spellings, it does not fall back to the default |
| `--attn-fattn-qstage` | `engine/decode_run.cpp` | CLI flag; inertness is refused **keyed on whether the knob was given**: `--attn-fattn 0 --attn-fattn-qstage …` aborts, because a knob that selects a kernel the shipping path never launches would otherwise echo as if it had run |
| launcher | `engine/kernels_attn.cpp` | selects one of the four `__global__`s on `(roll, qstage)`; **one launch of one kernel, same grid, on both sides** — the knob moves an operand’s address space, not the launch plan (§41) |
| `fattn_echo` | `engine/kernels_attn.cpp` | prints `q=stage` / `q=unstaged8` plus `lds=`, `wf_per_cu=`, `vgpr=` read by `hipFuncGetAttributes` / `hipOccupancyMaxActiveBlocksPerMultiprocessor` off **the function this launch submitted**, not off a name |
| NUMERICS line | `engine/decode_run.cpp` | carries `attn_fattn_qstage=…`, so a dump directory records the spelling that produced it |
| `engine/gate.py` | argv passthrough + an assertion that the echoed `attn_fattn_qstage` matches what was asked, mirroring `--attn-fattn-roll`. **No band, floor, threshold, admission or default-engine behaviour changed**; `ATTN_FORM_FLAGS` untouched |

Verification actually run on the control plane: `engine/build.sh` green (both binaries); `engine/gate.py --selftest`
→ `SELFTEST PASS` after the passthrough edit. One pre-existing defect was found and deliberately left alone:
`engine/gate.py --help` crashes (`ValueError: unsupported format character 'v'`) because an unrelated `--dry-run`
help string at `engine/gate.py:6936` contains a literal `5 %`, which argparse’s `_expand_help` reads as a format
specifier. None of the 45 lines this change adds contain a `%` (checked against the diff), so it is not this
change’s and is not routed around here — but it means `--help` is not a usable way to list the flags; `--selftest`
and `--attn-fattn-qstage` are.

**Limits of this package, stated plainly.** (a) No timing exists anywhere in it, and none may be inferred: the only
device legs run were correctness legs. (b) Byte-identity was taken on three shapes at `--wave-rows 512` with the
default roll; the `roll two` × `unstaged8` cell compiles and is selectable but has no device witness of its own.
(c) The bit-identity gate compares what the run dumps (logits, `STATEFP`); it is not a gate on the `two_u8`
instantiation, and it is not a numerics band — no band moved, per the repo’s delegated-format rule, because
nothing moved at all. (d) The A/B harness is untested against a device by design here; its parser is colour-tested
and its refusals are asserted in-code, but the first real window will be the first time its canaries see real
binaries — if CANARY A or B goes red, the harness is right and the build is wrong.
