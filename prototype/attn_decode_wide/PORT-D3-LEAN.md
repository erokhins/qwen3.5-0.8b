# PORT-D3-LEAN — the `lean` dot-loop spelling, ported into the shipping `attn_core`

Written by the implementation worker, 2026-10-21. **Nothing here is a measurement.** No `ms` figure
appears except values quoted from the ledger or from `prototype/attn_decode_wide/results.md`, each
marked *quoted*. The device work was correctness-only, one run at a time, under `tools/gpu_lock.sh`,
and each run is reported with its exact command line, wall time, `tools/gpu_lock.sh util` at acquire
and release, and the words **no timing taken**. Nothing was committed. `engine/performance-status.md`,
`README.md`, `progress/`, and every gate threshold/floor are untouched — `git status --short progress/`
is empty, which is the proof, not an assertion.

Scope as briefed: four lines in `prototype/attn_decode/attn_decode_kernels.h::attn_core`, the gate that
proves the copy claim, every other consumer of that header audited and re-run on CPU, the code-object
witness for both the prototype and the engine, and a paired 4k A/B script for the main agent.

---

## 0. Baseline, taken before any edit (CPU only, compile metadata)

`bash prototype/attn_decode_wide/census.sh`, against the `.s` the main agent's D3 build left behind
(2026-10-21 13:31 CEST):

```
simd_count=80 max_waves_per_simd=16 lds_size_in_kb=64 simd_per_cu=2
_Z9attn_core8CoreArgs                lds= 38944 B  next_free_vgpr=  53  next_free_sgpr=73
_Z14attn_core_lean8CoreArgs          lds=  6176 B  next_free_vgpr=  45  next_free_sgpr=72
```

Engine side, same fields, from `engine/build/kernels_attn.s` (the `--save-temps` listing of the shipping
TU, dated 2026-10-19 = the pre-port build), read with the repo's own code-object reader
`prototype/attn_percost/a11_read_vgpr.py`, i.e. `.amdhsa_next_free_vgpr` — not `.vgpr_count`, not a
driver query (AGENTS.md, 2026-10-20 row):

```
attn_core(CoreArgs)    vGPU 53  alloc 56  LDS 38944  blk/CU 1  wf/CU 3  binder LDS  (regs allow 24, LDS allows 1)
```

So both objects started at 38,944 B / 1 resident block per CU, `attn_core_lean` at 6,176 B / 4.

---

## 1. STEP 1 — the edit: four lines, line-count-neutral, with the warning on the lines

`prototype/attn_decode/attn_decode_kernels.h`, inside `attn_core` only, at the four score-dot sites
(now lines 315, 320, 352, 357 — same coordinates as before the change):

```
-  for (int t = 0; t < nd; ++t) dot += sq[lane * 8 + t] * sk[lane * 8 + t];
+  for (int t = 0; t < 8; ++t) if (t < nd) dot += sq[lane * 8 + t] * sk[lane * 8 + t];   // D3: literal trip + guard, NOT `nd` — promote-alloca-to-LDS; see "D3 LDS" at the foot of this file
-  for (int t = 0; t < nd; ++t) dot += bf(k[t]) * qreg[t];
+  for (int t = 0; t < 8; ++t) if (t < nd) dot += bf(k[t]) * qreg[t];   // D3: literal trip + guard, NOT `nd` — promote-alloca-to-LDS; see "D3 LDS" at the foot of this file
```
(twice: once in the one-pass `if (s_row == 1 && a.full)` branch, once in the streaming branch.)

Line 641 (`attn_tile_body`'s `for (int t = 0; t < nd; ++t) dot += bf(k[t]) * q[t];`) was **not**
touched, nor was anything in `attn_m2_kernels.h` or `attn_valu_slim_kernels.h` — that is a different
kernel (the tiled wave form) and an unmeasured lead.

### 1a. Why the warning is a trailing marker plus a foot note, not a block comment at the site

The brief asked for a comment at the first site. A block comment there costs 10 lines, and this header
is cited **by line** from 45 named coordinates whose highest is `:767` — 28 of them sit below line 314,
so a 10-line insert at the site rotates all 28 and `tools/citation_rot_check.py` reports 26 rotted rows
(11 of them in frozen `progress/` journals). The census, re-runnable: `tmp/cite_endpoint_check.py`
→ `distinct named coordinates: 45 | highest: 767 | named endpoints that ARE an edited line: NONE |
endpoints > 314 (rotated by a +10 insert at 315): 28`. The supervisor ruled for the line-count-neutral
shape on that evidence, so:

* the four lines carry the warning as a **trailing comment** — zero lines added, and it lands on exactly
  the line someone would rewrite;
* the full mechanism note is appended at the **foot** of `attn_decode_kernels.h` under the banner
  `D3 LDS`, below every cited coordinate, and it carries the mechanism, the numbers, the trip-set
  argument, the pointer to `prototype/attn_decode_wide/results.md`, the instruction not to "simplify"
  the guard back to `t < nd`, and one line saying why the note lives at the foot;
* both are mirrored verbatim into `attn_core_lean`, so the two files stay textually identical apart from
  the kernel name, which is what `check_source` adjudicates.

### 1b. The witness that the edit moved no lines

```
git show HEAD:prototype/attn_decode/attn_decode_kernels.h | wc -l   ->  821
head -821 prototype/attn_decode/attn_decode_kernels.h | diff against HEAD  ->  4 lines changed, all four are the dot loops
wc -l prototype/attn_decode/attn_decode_kernels.h                   ->  849   (821 + the foot note)
.venv/bin/python tools/citation_rot_check.py                        ->  0 rotted line-citation(s)
git status --short progress/                                        ->  empty (no journal was edited)
```

`wc -l` is the witness and the checker alone is not: `citation_rot_check.py` compares against `HEAD`,
and `HEAD` moves with the commit being checked, so it cannot see drift introduced inside that commit
(AGENTS.md, the row recording exactly this). The foot note adds 28 lines **below** coordinate 767, which
is why it is free; the code change itself is 821 → 821.

### 1c. Two kinds of citation, kept separate (mixing them is what made this look like a 26-file commit)

* **Coordinate cites** (`path:NNN`) must keep pointing at the right line. Repaired by moving the number,
  same line. For this change: none needed, because none of the 45 endpoints is an edited line and the
  edit is length-neutral above 767.
* **Verbatim quotes** record what the code said when written. Never rewritten, not even when the code
  moves. For this change: `progress/` contains **zero** occurrences of `t < nd` (checked:
  `git grep -n 't < nd' -- progress/` is empty), so no journal quoted the old header at all and no
  journal was edited.

Mentions of the old spelling elsewhere, classified:

| where | kind | action |
|---|---|---|
| `attn_decode_kernels.h:641`, `attn_m2_kernels.h:775`, `attn_valu_slim_kernels.h:209` | live code of a **different** kernel (`attn_tile_body` and its two copies) | untouched by design |
| `attn_valu_slim/touch_probe.py:53` | live, still-true quote of that other kernel's `bf(k[t]) * q[t]` | untouched |
| `attn_decode_wide/HANDOFF-D3.md:97,121`, `attn_decode_wide_kernels.h:9,30,31` | the handoff and the variant's own before→after illustration | accurate as written, left |
| `attn_decode_wide/gate.py:69,70,255` | the checker's own forbidden-string constant | intended |
| `prototype/attn_decode/results.md` (prose above the foot) | historical record of the pre-port spelling | left as written; a dated foot note was appended at the foot of that file (highest coordinate citing it is 1122 of 1,509, so an append moves nothing) |

---

## 2. STEP 2 — the gate that had to move in the same change

`prototype/attn_decode_wide/gate.py::check_source()` used to prove the variant is a copy by *requiring*
the frozen sites to read `t < nd` and the variant to read the re-sailed form, and erroring if fewer than
3 differences appeared. After the port that check would have failed by construction. It was rewritten to
adjudicate the new contract instead of merely passing:

1. **The shipping core must still be `lean`.** `LITERAL_TRIP` (`for (int t = 0; t < 8; ++t) if (t < nd)`)
   must appear **exactly 4 times** in `attn_core`, and `RUNTIME_TRIP` (`for (int t = 0; t < nd; ++t)`)
   **zero** times, in both files. Failure message: "SHIPPING KERNEL REGRESSED to a runtime trip count:
   … has N of 4 literal-trip dot sites … A runtime `nd` bound makes LLVM promote `qreg[8]` and the K-row
   `uint4` to LDS: 38,944 B reserved instead of 6,176 B, 1 resident block per CU instead of 4." This is
   the anti-rewrite half: a "simplification" back to a runtime bound keeps every numerics gate green and
   costs 4× residency silently.
2. **The variant must still be a copy.** Line counts equal, every line identical modulo the rename, and
   the difference count must be **exactly 1** (the name). The old "comment differences are allowed"
   latitude was *not* added — allowing comment drift is the vacuous-check class.
3. **Anti-vacuity**: the matched count is asserted `== 4` explicitly, a body that cannot be sliced is an
   error, and `allowed != 1` fails. A check over an empty match list cannot fail, which AGENTS records at
   least three times.

Current output:

```
   shipping attn_core: 4/4 dot sites on the literal trip count, 0 on a runtime bound
   attn_core_lean: 4/4 dot sites on the literal trip count, 0 on a runtime bound
   variant is the frozen kernel: 151 lines, 1 permitted difference (the name; …)
```

**Its own negative control, run and passing**: `prototype/attn_decode_wide/check_source_witness.py`
runs `check_source` against scratch copies of the two headers under five mutations (CPU only, real
headers copied and never touched) — A as-shipped PASS, B one site reverted FAIL, C all four reverted
FAIL, D variant body edited FAIL, E variant line deleted FAIL → `WITNESS PASS: 5/5 cases behaved as
required`. Without this, "the gate is green" would have been decoration.

Two honest consequences of the port, recorded rather than waved through:

* `gate.py` §1 (lean vs ship, bit-identity) is now a **tautology by construction** — the two bodies are
  identical source. It can still catch a build that linked a stale object, and nothing more. The
  pre-port evidence that `lean` was bit-identical to the pre-port core is `results.md` §1 and it stands
  on its own; the live authority on the shipping path is now `engine/gate.py`. The module docstring says
  so where §1 is described.
* `results.md`'s and `HANDOFF-D3.md`'s canary line — `variant is the frozen kernel: 151 lines, 5
  permitted differences` — now prints `151 lines, 1 permitted difference`. Both documents are
  measurement records and were **not** edited; anyone re-taking that canary block will see the new
  wording, which is the correct reading after the port, not a drift.

---

## 3. STEP 3 — every consumer of that header, audited and re-run (CPU only)

| consumer | does it read the dot loop? | verdict, and the command that shows it |
|---|---|---|
| `prototype/attn_decode_wide/gate.py --check-source` | yes, that is its job | PASS, 4/4 + 4/4 sites, 1 permitted difference (§2) |
| `prototype/attn_m2/check_m16_transcription.py` | **no.** It reads `SHIP` (this header) for exactly two comparisons — the `lanes_of` body and the `blocks_for_row` body. Every `streaming step:` entry passes the shipped text only into the *label*; `C.has()` searches `mine` (`attn_m16.hip.cpp`). The one dot-loop label names `attn_tile_body`'s line 641, which is not part of this port. | **RED, but red at HEAD**: it looks for `__device__ __forceinline__ int lanes_of` while the header has declared `__device__ __host__ inline int lanes_of` since before this change. Witnessed: `git worktree add tmp/wt-d3head HEAD --detach` and running the checker there fails identically → pre-existing, unrelated to the port, NOT fixed here (out of scope; the fix is those two signature strings). With those two strings patched **in a scratch copy only**, it reports `PASS: check_m16_transcription — 23/23` against the ported header, and `--expect 23` exits 0. |
| `tools/attn_ceiling_proof.py` | no — it parses the three `a.part +` address expressions and `MAXSC` | PASS: `PROOF PASS: the cap reaches only the allocation and the refusal; …` |
| `prototype/attn_decode/run.py` | no — it regexes `#define TILE` (and imports `compare`/`ndiff_limit`) | resolves: `KERNEL_TILE = 8`, floors `0.9999 / 0.01 / 2.0 / 1`. The gate itself is a device run of unknown length and is handed to the main agent. |
| `prototype/attn_decode/wave_model.py` | constants only; `--isa` re-compiles the header and prints the reservation + named loop spans | `selftest: 6/6 passed`; `isa` now prints `attn_core: group_segment_fixed_size = 6176 B/block` (HEAD's own worktree run prints `38944` for the same line). The named spans it uses to guard `WARP_INSTR_TOUCH` moved little: score-tile smallest span 190 → 194 instrs (`fma` 15 → 17, `lds` 8 → 11), softmax+PV 156 → 155. **Not re-derived**: `WARP_INSTR_TOUCH = {one_pass: 1030, split: 647}` is a frozen 2026-10-16 census citation and this worker does not price instruction bills from a dry census; the fresh spans are printed beside it by the same command so a reader can see the drift. |
| `tools/kernel_occupancy_probe.cpp` | includes the header | compiles: `hipcc -O3 --offload-arch=gfx1151 -I. -Iengine -c tools/kernel_occupancy_probe.cpp -o tmp/d3_probe.o` → rc 0. Not run (it queries the driver; AGENTS 2026-10-20 says the code object is the authority). |
| `prototype/attn_decode_wide/{census.sh,loop_bill.py,isa_bill.txt}` | no — read the build's `.s` | re-taken after rebuilding, see §3a/§3b |
| `engine/gate.py`, `engine/attn_form_band.py`, `prototype/attn_tile_model.py` | read `MAXSC`/`TILE`/`DH`/address text, not the dot loop | not re-run: `engine/gate.py` is a device gate whose runtime this worker cannot bound (handed over); the other two consume constants this edit does not touch. |

`git grep` for any other file reproducing the four lines verbatim: `attn_decode_kernels.h` (the four,
now literal), `attn_decode_wide_kernels.h` (the mirrored four, plus the header's before→after
illustration at lines 30-31), and `progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md:173` — prose
quoting the loop **body** `dot += bf(k[t]) * qreg[t]` ("8 serial … per thread"), which is unchanged in
substance and is accurate history. No live file outside the two headers reproduces the old header text.

### 3a. `census.sh` — a hard-coded residency pair that had become a lie

The residency lines used to be a literal `(38944, 6176)` pair printed *underneath* the parsed readings,
so after the port it would have printed `attn_core 38944 B -> 1 block(s)/CU` beside a header reserving
6,176 — a census contradicting its own output above it, and readable. Now every number is derived from
the `.s`, and a missing kernel is a hard error (the empty-parse failure class). Re-taken on the final
build:

```
_Z9attn_core8CoreArgs                lds=  6176 B  next_free_vgpr=  45  next_free_sgpr=72
_Z14attn_core_lean8CoreArgs          lds=  6176 B  next_free_vgpr=  45  next_free_sgpr=72
  attn_core          6176 B -> 10 blocks/CU by LDS -> 4 block(s)/CU resident
  attn_core_lean     6176 B -> 10 blocks/CU by LDS -> 4 block(s)/CU resident
```

Negative control run: pointed at a `.s` with no `attn_core_lean` (the engine's), it aborts —
`census refuses to report: no .amdhsa_kernel block for _Z14attn_core_lean8CoreArgs … ` (rc 1).

### 3b. `isa_bill.txt` — the caveat that no longer describes reality, corrected in place

The shipped core's `nd`-bounded dots were rolled and unannotated, so its per-loop numbers under-counted
(`fma32=1` was one rolled FMA standing for eight) while `attn_core_lean`'s literal trips unrolled and
billed completely — and the file said the two cores' bills are NOT comparable. That was a property of
the two spellings, not of the tool. Both cores are now the same spelling, both unroll, and the bills
agree loop for loop. Re-taken from one build of the ported header (no device, no timing):

| | pre-port `attn_core` (record kept in the file) | post-port `attn_core` | post-port `attn_core_lean` |
|---|---|---|---|
| `group_segment_fixed_size` | 38,944 B | **6,176 B** | 6,176 B |
| `next_free_vgpr` | 53 | **45** | 45 |
| total instructions | 3,648 (498 in annotated loops) | 3,621 (591) | 3,624 (591) |
| tile-score loop | 226 (fma32=1) | **270 (fma32=17)** | 270 (fma32=17) |
| streaming (max,sum) | 137 | **187** | 187 |
| softmax+PV | 118 | **117** | 117 |
| inner d2 loop | 17 | 17 | 17 |

So the complete figure HANDOFF-D3 §1 wanted is now readable from the shipping core: **270 warp-instructions
per (head, lane) touch one-pass, 187 + 117 = 304 streaming**. The residual 3-instruction difference in the
"outside annotated loops" bucket is unexamined: identical source text, so whatever it is, LLVM chose it,
not the kernel. The pre-port block is kept verbatim underneath the new one, labelled, because it is the
evidence that the 226 / 137+118 figures quoted in `HANDOFF-D3.md` §1 and `results.md` were real.

---

## 4. STEP 4 — builds and the code-object witness (compile/metadata only, no device work)

All three builds clean, from the final sources: `bash prototype/attn_decode/build.sh`,
`bash prototype/attn_decode_wide/build.sh`, and `EXTRA_FLAGS="--save-temps=obj" bash engine/build.sh`
(two binaries; the extra flag only keeps the device listings that the canary reads — `-O3` unchanged).

**The two witness lines the report was owed** — `.amdhsa_kernel` metadata, `.amdhsa_next_free_vgpr` and
`.amdhsa_group_segment_fixed_size`, read by `prototype/attn_percost/a11_read_vgpr.py`:

```
prototype/attn_decode_wide/attn_decode_wide.hip-…-gfx1151.s   attn_core        LDS 6176  next_free_vgpr 45
engine/build/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s     attn_core(CoreArgs)  vGPU 45  alloc 48  LDS 6176
```

**THE PORT LANDED ON THE SHIPPING PATH.** The engine's `attn_core` reserves 6,176 B with
`next_free_vgpr` 45, exactly the `lean` figure, against 38,944 / 53 in §0's pre-port reading. The
`blk/CU 10 / wf/CU 30` a11 prints is the LDS-only bound at its own register assumption; at
THREADS = 256 the 1,024-thread ceiling caps it at 4 blocks/CU, which is what `census.sh` prints as
`4 block(s)/CU resident`.

### STEP 7 (observation only, no edit, no device, no claim)

Do the wave kernels show the same promote-alloca-to-LDS behaviour? Their reservations equal their
declared staging exactly, so no: `kTileLdsBytes = (kTilePairs*MAXSC + 2*ATTN_TILE_L*DH)*2 =
(4*512 + 2*8*256)*2 = 12,288 B`, and the code objects read 12,288 B for both
`attn_tile_wave` and `attn_slim_wave` (47 VGPR, 5 blocks/CU by LDS, THREADS=128). `attn_tile_body`'s
`float q[8]` with its runtime-`nd` loop at line 641 therefore is **not** being promoted — there is no
analogous reclaim on the wave path. For reference from the same object: `attn_prep` 2,080 B / 30 VGPR,
`attn_combine` 0 B / 10 VGPR, the `attn_fattn_wave*` family 20,480–32,768 B at 241–248 VGPR, and
`attn_wide_wave` / `attn_wide_llama_wave` 32,768 B at 256 VGPR.

---

## 5. STEP 5 — the device gates this worker ran, and what is handed over

Ran two device commands, one at a time, each under its own `tools/gpu_lock.sh` hold, blocking,
correctness-only, no timing taken. `prototype/attn_decode_wide/gate.py` does not self-lock (verified:
its `self_locks()` check looks only inside `subprocess` calls, and the only self-locking drivers in the
tree are `prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`).

1. `.venv/bin/python prototype/attn_decode_wide/gate.py --check-source` — CPU only, 0.75 s wall, no
   device touched, no timing taken. PASS.
2. Under lock tag `d3-lean-port`: `.venv/bin/python prototype/attn_decode_wide/gate.py` — wall **4.4 s**,
   `tools/gpu_lock.sh util` **91 %** at acquire / **7 %** at release, **no timing taken**. Result:
   `RESULT d3gate cases=21 bitdiff_stages=0 pinned_off_floor=8 negative_failures=0`,
   `verdict: BIT-IDENTICAL (126 stage comparisons over 21 cases)`, `0 negative-control failures`, and
   the known `12 pinned cases, 8 outside the floors` — identical in shape to the main agent's pre-port
   canary, including the `split 9/18/36` cut probe printing `bytes moved vs the natural cut: 0`.
3. Under lock tag `d3-lean-port2`, after the final source state and a full rebuild, the same command
   again: wall **4.3 s**, util **93 %** at acquire / **7 %** at release, **no timing taken**, same
   `RESULT d3gate cases=21 bitdiff_stages=0 pinned_off_floor=8 negative_failures=0`.

Nothing else was run on the device: the runtime of `prototype/attn_decode/run.py`'s gate and
`engine/gate.py` cannot be bounded inside a minute, so they are handed over below.

---

## 6. STEP 6 — the paired A/B script, `prototype/attn_decode_wide/ab_d3_engine.sh`

Main-agent-only (it takes ms). Run blocking, zero subagents, one lock hold — the script acquires
`tools/gpu_lock.sh` itself with tag `d3-lean-ab` and releases it on exit.

```bash
bash prototype/attn_decode_wide/ab_d3_engine.sh                       # REPS=3, GATE_FIXTURES=dec8
REPS=5 GATE_FIXTURES="dec8 wide4096f" bash prototype/attn_decode_wide/ab_d3_engine.sh
```

What it does, in order:

1. **Builds NEW from the working tree and BASE from a scratch `git worktree add $OUT/base HEAD --detach`**
   — never from the touched working tree — both with `--save-temps=obj`, binaries kept under
   `tmp/d3ab/`. Builds happen before the lock is taken (a compile is not device work). `REBUILD_BASE=1`
   forces a fresh HEAD build; a cached base is used otherwise and says so.
2. **CANARY, before any ms**: prints `attn_core`'s LDS and `next_free_vgpr` from **each side's own**
   code object and ABORTS unless base reads 38,944 and new reads 6,176, and unless new's `vgpr` is below
   base's. `engine/build.sh` is not bit-reproducible, so `cmp` between two binaries proves nothing
   (AGENTS 2026-10-17) — this is the behavioural canary the rule asks for.
3. **Prompts** as id **files**, via `prototype/attn_decode_wide/mkids_d3.py` (the
   `tools/context_sweep.py` recipe; 112 and 4,000 ids, field count asserted by the writer) — the
   `MAX_ARG_STRLEN` and `--ids-file`-splits-on-commas traps together.
4. **Numerics first**: `engine/gate.py --binary <each side's own binary> --fixture $GATE_FIXTURES
   --work …`, both arms must print `ENGINE GATE PASS`, plus a direct byte-diff of the two arms'
   `sampled.txt` at 112+16 lanes. A red numerics line outranks the timing entirely, so the window
   aborts before any ms is spent and prints `NO-TIMING:`. *Ordering note:* the brief listed numerics
   after the reps; running it first is a deliberate deviation so a red does not waste a whole window,
   and the verdict is re-asserted before anything prints. Say so if you re-shuffle it.
5. **Exercised-path canary, asserted on every timed run**: `PREFILL tokens=` equals the id file's field
   count; a `SPLITCAP split=8 … rows=1` line exists **and** one of them reads `lanes=4032` (at 128
   lanes: `split=1`, `lanes=128`); a `STAGE gran=stage family=attn` row exists; and the run's own
   `STAGEPARITY … tokens_identical=1 logits_bitidentical=1` holds. `splitcap_echo` is called by
   `attn_decode_layer` before the `rows == 1` branch, so the decode path prints it — that is what makes
   "the decode attention ran at that width and split" a reading rather than a hope.
6. **Interleaved reps**: `decode_run --time-stages` at 4,032 lanes / `--steps 32` (default 3 reps per
   arm) and 128 lanes / `--steps 16`, arms alternating rep by rep so clock drift shows as disagreement
   between reps of one arm. Reports median + spread for the pre-registered `attn` line
   (`STAGE gran=stage family=attn ms_per_step`) and for the headline `STAGE gran=burst family=TOTAL
   ms_per_step`, plus `tok/s`.
7. **Pre-registered verdict printed**: WIN ≤ 1.50 ms, KILL ≥ 2.55 ms at 4,032 lanes (both quoted from
   HANDOFF-D3.md's head), between = neither;
   it also prints `X = 2*(3.023 − lean)` since X is the number the route was really after (3.023 is the
   ledger's quoted `attn` line at 4,032 lanes, progress/2026-10-21/decode-attribution-2026-10-21.md §2). The 128-lane
   row is labelled **in advance** as a FLAT = PREDICTED PASS (8 blocks on 40 CUs is one round at either
   residency; HANDOFF-D3.md §4.3 declared 0.40 ms out of reach for this route).
8. Header records the projection the script exists to falsify: −45.2 % on the isolated body and
   X ≈ 0.64 → 78.9 → ≈84.7 tok/s at 4,032 lanes (*all quoted from
   `prototype/attn_decode_wide/results.md` and the ledger, none measured here*).

One bug the dry run caught before handover: this box's locale uses a comma as the decimal separator, so
`sort -n` truncated every key at the `.` and awk read `"3.0230"` as `3` — medians, spreads and the
WIN/KILL verdict would have been computed off truncated integers while still printing plausible digits
(`attn line: 3,0000 -> 1,0000 ms/step (-66,67 %)`). The script now `export LC_ALL=C`. The report block
was then re-run against synthetic data and reads correctly (`attn line: 3.0230 -> 1.7300 ms/step
(-42.77 %)`, `VERDICT … NEITHER`), and `bash -n` is clean. `prototype/attn_decode_wide/mkids_d3.py` and
`gate.py` are `py_compile` clean.

---

## 7. Exact command sequence for the main agent's device pass

```bash
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b
# everything below: blocking, zero subagents, never backgrounded.

# 1. the two prototype gates + the code-object census (~15 s of device work total)
bash tools/gpu_lock.sh acquire d3-lean-pass && bash tools/gpu_lock.sh util
.venv/bin/python prototype/attn_decode_wide/gate.py --check-source      # CPU only
.venv/bin/python prototype/attn_decode_wide/check_source_witness.py     # CPU only, 5/5 expected
bash prototype/attn_decode_wide/census.sh                               # CPU only
.venv/bin/python prototype/attn_decode/run.py                       # the op's own device gate (dec8, all layers/steps)
.venv/bin/python prototype/attn_decode/run.py --long --wave         # if the long/wave routes are wanted too
bash tools/gpu_lock.sh release d3-lean-pass

# 2. the engine numerics gate on the ported build, and at the wide pin (one lock hold, run blocking)
bash tools/gpu_lock.sh acquire d3-lean-gate
.venv/bin/python engine/gate.py --fixture dec8
.venv/bin/python engine/gate.py --fixture wide4096f
bash tools/gpu_lock.sh release d3-lean-gate

# 3. the paired A/B — it builds base from HEAD, runs its own canaries, holds its own lock
REPS=3 GATE_FIXTURES="dec8 wide4096f" bash prototype/attn_decode_wide/ab_d3_engine.sh
```

`prototype/attn_decode/run.py` does **not** self-lock (its only `gpu_lock` mention is a docstring), so
holding the lock once around the block is correct; `engine/gate.py` does not self-lock either.

---

## 8. Residual risks, stated plainly

1. **The engine's own numerics gate has not been run on the ported build.** This worker could not bound
   its runtime. It is the one check that decides "the port is byte-identical on the shipping path", and
   it is first in the handover list. `prototype/attn_decode/run.py`'s gate is likewise unrun here.
2. **`gate.py` §1 is now vacuous** (§2) and `attn_core_lean`'s reason to exist is historical. If the D3
   rig is ever retired, the pre-port evidence lives in `results.md` and must move with it.
3. **`check_m16_transcription.py` is red at HEAD**, independently of this change (stale
   `__device__ __forceinline__` signature strings for `lanes_of` / `blocks_for_row`). Witnessed on a
   clean HEAD worktree; verified that with those two strings patched in a scratch copy it reports
   23/23 against the ported header. Not fixed: not this commit's subject.
4. **`wave_model.py`'s `WARP_INSTR_TOUCH` (1030/647) is a frozen census citation from the rolled form.**
   Its guard (`isa`) now prints 194/155 against the recorded 190/156, i.e. within a few percent — close
   enough that nobody should be misled, far enough that it is a stale citation, and re-deriving it is a
   pricing exercise this worker is not funded for.
5. **The 128-lane row is expected to be flat** and must be reported as a predicted pass, not a miss.
6. **The projection will probably not survive.** X ≈ 0.64 came from an isolated per-layer body (*quoted*);
   the in-situ family has prep, combine and launch structure around the core.
7. **Not committed.** `git status --short` shows 6 modified files
   (`attn_decode_kernels.h`, `attn_decode/results.md`, `attn_decode_wide/{attn_decode_wide_kernels.h,
   census.sh, gate.py, isa_bill.txt}`) and 4 new ones (`PORT-D3-LEAN.md`, `ab_d3_engine.sh`,
   `mkids_d3.py`, `check_source_witness.py`); nothing staged, nothing in `progress/`, no gate threshold
   touched. Scratch left under the gitignored `tmp/`: `tmp/d3_m16` (the patched-checker experiment),
   `tmp/d3_census_neg`, `tmp/d3_abtest` (the report dry run), `tmp/d3_probe.o`, and the supervisor's
   `tmp/cite_endpoint_check.py`. The scratch HEAD worktree this session used to witness the
   `check_m16_transcription.py` pre-existing failure was removed and `git worktree prune` run.
