# A6-result — G1's byte target is reachable at 185 VGPR (20,480 B, 9 wf/CU), G0's is 65,536 B at 7 wf/CU, and the register-file rung is answered by a driver ladder

Written 2026-10-19 by the A6 worker, on tip `9a4bbce`, against `prototype/attn_percost/A6-BET.md` (the spec:
order, lines, numerics policy) and `prototype/attn_percost/A5-GEOMETRY.md` §3.1/§3.2/§4.1/§4.2.

**Compile-only and driver-fact-only. No kernel of the engine was launched by anyone in this file, and no timing
was taken by anyone in this file: there is not one `ms` figure here, and none may be copied from here into
`engine/performance-status.md`.** The one step that touches a device is Step 3's occupancy *query*
(`hipOccupancyMaxActiveBlocksPerMultiprocessor`), which submits no work; its command, wall, `util` and the words
"no timing taken" are in §3.

Citation legend (AGENTS' citation-hygiene rule: a `§N` that names no file resolves only in a reader's memory):
a bare `§N` below means §N **of this file**; `§19/§20.1` mean `prototype/attn_decode/results.md`; `§28/§31/§33/§34/§37/§47`
mean `engine/performance-status.md`; `A6-BET`, `A5`, `A3-result.md` and `E1` mean
`prototype/attn_percost/A6-BET.md`, `prototype/attn_percost/A5-GEOMETRY.md`,
`prototype/attn_percost/A3-result.md` and `prototype/attn_wmma/E1-result.md`, and a `§` after one of those names is
that file's section. Every number is labelled **read** (a `path:line` or a note-block field, with
the file), **derived** (the sum or formula is printed), or **quoted** (a `§`).

## 0. The answers, in four lines

| question | answer |
|---|---|
| **Is `group_segment_fixed_size == 20,480 B` reachable at spill 0?** (G1-M0, the cheap falsifier of the whole geometry thesis) | **YES — read 20,480 B, `vgpr_spill_count 0`, `private_segment_fixed_size 0`, `vgpr_count 185` (≤ E1's 190 bound).** G1 is *not* dead at the source the way A3 was. this file's §1. |
| Is G0's 65,536 B reachable, at 1 block/CU → 7 wf/CU? (G0-M0) | **YES — read 65,536 B, spill 0, private 0, `vgpr_count 178` unchanged.** The 7-warp form compiles without touching one register. this file's §2. |
| Does the epilogue change move any published value? | **No, arithmetically: all 4,096 `(row-slot, dim)` destinations of a warp are written exactly once by both spellings and each is fed by the same accumulator element.** `prototype/attn_percost/a6_epi8_model.py`, negative-controlled. this file's §1.4. |
| Does the driver allow 3 blocks/CU at 20,480 B? (the rung A3-result §4 named and nobody ran) | **YES — 3 blocks/CU, 9 wf/CU, at the driver's own `numRegs` 185**, and its LDS-only stub of the same size says the same 3, so there is no register shortfall to attribute. The ladder rungs themselves are flat (the minBlocks hint never squeezed these bodies) and this file's §3.3 says what that means. G1 is owed its device A/B. this file's §3. |

Two harness-legality canaries passed before any variant number was read. **The `base` side reproduces the shipped
note block exactly** — `178 / 0 / 0 / 32,768` (§33's banner, as re-read by `E1` §1) — so the deltas below are
deltas against the shipping object and not against a second harness. **The un-stage-only side reproduces `E1`
§1 exactly** — `184 VGPR / 0 spill / 0 private / 32,768 B`, and even `sgpr 24 → 27` — which is the witness that
my re-anchored port is `prototype/attn_wmma/evidence/fattn_unstage_q.diff` and not something else: that patch no
longer applies to this tree (`git apply --check` fails at `engine/attn_fattn_kernels.h:445` and
`engine/kernels_attn.cpp:170`, because the `--attn-fattn-roll` knob moved the roll behind
`fattn_score_q_tile`/`fattn_score_k_tile`), so the port had to be re-anchored and needed re-witnessing.

## 1. G1-M0 — Q un-staged plus the 8-row epilogue: the byte target is reached

### 1.1 The four note blocks, side by side

Read out of `--save-temps` listings built by `bash prototype/attn_percost/a6_build.sh /tmp/a6build` (CPU only),
one scratch directory per side, all four sides compiled from the *same three files* copied into their own
quarantine dir — `engine/kernels_attn.cpp`, `engine/attn_fattn_kernels.h`,
`prototype/attn_decode/attn_decode_kernels.h` — with `-I$side` ahead of `-I$REPO` so the quote-includes resolve
to the copies. `engine/` was never written to. Pattern taken from `prototype/attn_percost/a4_l1_build.sh`.

```
        field                     base (shipped)   unstage (E1)   g1 (the bet)   g0 (7 warps)
        vgpr_count                    178             184            185            178
        vgpr_spill_count                0               0              0              0
        sgpr_count / sgpr_spill       24 / 0          27 / 0         27 / 0         24 / 0
        private_segment_fixed_size      0               0              0              0
        group_segment_fixed_size    32768           32768          20480          65536
        max_flat_workgroup_size        96              96             96            224
        uses_dynamic_stack            false           false          false         false
        derived blocks/CU (LDS)          2               2              3              1
        derived wf/CU                    6               6              9              7
```

`max_flat_workgroup_size` is `__launch_bounds__(FATTN_THREADS)` reaching the object, which makes it the free
third canary for G0 (96 → 224) beside `group_segment_fixed_size`, and `uses_dynamic_stack false` on every side is
the same no-scratch claim `private_segment_fixed_size 0` makes from the other direction.

**G1's admission, against `A6-BET`'s G1-M0 line** — `group_segment_fixed_size == 20,480` ✓ read;
`vgpr_spill_count == 0` ✓ read; `private_segment_fixed_size == 0` ✓ read (the task's wording asks for it and
`A6-BET`'s milestone line asks for spill; both are zero); `vgpr_count ≤ 190` ✓ read at **185**, which is E1's
184 plus one register for the extra pass bookkeeping. **G1-M0 passes.** A5 §7 step 1 — "If G1's note block cannot
reach 20,480 B at spill 0 and private 0, G1 is dead on §33's precedent" — did not trigger.

The byte count with its sum behind it (AGENTS: a byte count without a sum is a guess). Declared in the header by
the assert the patcher rewrites, and evaluated by the compiler:

```
3 warps x (FATTN_TRG=8 rows x FATTN_TROW=128 F32 x 4 B) = 3 x 4,096 = 12,288 B   (transpose stages, per warp)
1 aliased K/V stage = 16 lanes x 256 dims x 2 B         =             8,192 B
Q stage                                                 =                 0 B
                                                            ---------------
block LDS                                             =  20,480 B
blocks/CU = floor(65,536 / 20,480) = 3   ->  3 x 3 warps = 9 wf/CU               [derived]
```

Divisors, re-read on this box at the time of writing rather than quoted from the ladder:
`/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties` → `lds_size_in_kb 64`, `simd_per_cu 2`,
`simd_count 80`; `rocminfo` Agent 2 → `Compute Unit 40`, `SIMDs per CU 2`, `Max Waves Per CU 32`,
`Max Work-item Per CU 1024`. **HIP's capability answers and clang's `Occupancy [waves/SIMD]` remark are not used
anywhere here** (AGENTS' counter/occupancy trap row), and the register file per CU is still unknown, so no
register-derived block count appears in any table in this file — which is why this file's §3 exists.

The other two residency limits were read too, and none of them binds: the tool's own `derived blocks/CU` is the
minimum of three terms, and at 96 threads the work-item limit allows 1024 / 96 = **10** blocks and the wave limit
32 / 3 = **10**, while at 224 threads they allow 1024 / 224 = **4** and 32 / 7 = **4** [derived]. So LDS is what
binds in every side (2 < 10 shipped, 3 < 10 for G1, 1 < 4 for G0), and no block count in this file rests on a
register-file assumption.

### 1.2 What the two halves are, and why they were built separately

`prototype/attn_percost/a6_g1_unstage.py` = `E1`'s port re-anchored to the current header.
`prototype/attn_percost/a6_g1_epi8.py` = the epilogue staging 8 rows at a time, i.e. `E1` §4's first named route
quoted verbatim in A5 §3.2 (*"stage the epilogue 8 rows at a time instead of 16: `3 x 4,096 + 8,192 = 20,480 B`
→ 3 blocks/CU = 9 wf/CU, at the cost of 4 transpose passes instead of 2"*). They are one candidate but two
patchers precisely so the LDS move is **attributable**: `E1` established that un-staging Q alone leaves the block
at 32,768 B, so the intermediate side is a measured waypoint, not an extra option — and it reproduced that
reading to the byte, which is what licenses attributing the whole 12,288 B drop to the epilogue.

The epilogue change in one sentence: the stage becomes `FATTN_TRG = 8` rows deep, the epilogue runs
`FATTN_NPASS = 4` passes (half-major over DH, two row groups each), and a staged row is addressed by
`fattn_tr_addr(row - row0, d)` — one expression, writer and reader both (§20.1's rule, moved with the object it
sizes). Because a pass holds 8 rows instead of 16, four lanes per row publish instead of two, so
`FATTN_LBLK = (FATTN_TRG × (FATTN_TROW / FATTN_TBLK)) / 32 = 4` blocks per lane per pass, and the shipped
coverage assert is re-derived for the narrower stage rather than asserted beside it:
`FATTN_TRG * FATTN_TROW == 32 * FATTN_LBLK * FATTN_TBLK` → `8 × 128 == 32 × 4 × 8` ✓ (both 1,024).

### 1.3 The TU drift check — what else moved

`prototype/attn_wmma/fattn_residency_read.py --asm <side>/kernels_attn-…-gfx1151.s --all` diffs the note-block
census of **every kernel this TU emits — all 8 of them** (`attn_core`, `attn_tile_wave`, `attn_combine`,
`attn_prep`, `rope_table`, `attn_slim_wave`, and the two `attn_fattn_wave` instantiations; 8, not the 40 kernels
the engine has across all TUs, and the census names its population because AGENTS' rule is that a clean bill
measures only what was scanned). **Only the two `attn_fattn_wave` kernels move:**

```
   kernel (vgpr/spill/private/group)   base           unstage         g1            g0
   attn_fattn_wave                     178/0/0/32768  184/0/0/32768  185/0/0/20480  178/0/0/65536
   attn_fattn_wave_two                 200/0/0/32768  220/0/0/32768  221/0/0/20480  200/0/0/65536
```

The shipping default roll is `single` (`engine/decode_run.cpp:219` `kSingle`), so `attn_fattn_wave` at **185** is
the kernel the bet is about. Worth recording anyway because a device A/B might want the other side: the
`--attn-fattn-roll two` instantiation costs +40 VGPR just from un-staging (200 → 220, and `sgpr 24 → 26`) and +1
more from the narrower epilogue (220 → 221), and stays spill-free — the narrower epilogue is cheap, the global-Q
read is what the two-accumulator form pays for. `attn_core`, `attn_slim_wave`, `attn_tile_wave`, `attn_prep`,
`attn_combine`, `rope_table` are byte-for-byte identical on every side (the four census files are
`/tmp/a6final/drift_*.txt`; the command is in this file's §6).

### 1.4 The epilogue's bit-identity claim, taken as arithmetic rather than as prose

`prototype/attn_percost/a6_epi8_model.py` (CPU only, no device) transcribes `fattn_c_row`/`fattn_c_col`
(`engine/attn_fattn_kernels.h:280-281`) and both epilogues' index arithmetic, and enumerates the set of
publishes as `{(row-slot, dim): (lane, t, l)}`:

```
shipped  :  4096 published (row, dim) pairs   stage = 16 rows x 128 F32 = 8192 B/warp x 2 passes
g1       :  4096 published (row, dim) pairs   stage =  8 rows x 128 F32 = 4096 B/warp x 4 passes
identity : 4096/4096 destinations fed by the SAME accumulator element in both forms
```

Three claims, all checked: **coverage** (4,096 = 16 rows × 256 dims, exactly one writer each — the shipped file's
own defect #2 on `fattn_transpose_out`, "16 dims published per row, 240 never written", is invisible to a bound
check and visible here), **identity** (same source register per destination, so the published bytes are the
shipped bytes whatever the stage layout, which is what `A6-BET`'s "no numerics band re-opens for G1" rests on),
and **bounds** (no index leaves its own stage — G1's stage is half as deep, so an index left over from the
16-row geometry walks into the next warp's stage, which is §20.1's aliasing class).

`--negative` runs the must-catch set, because AGENTS' record is that a control which cannot fire is not a
control (three of this tree's bugs were found that way, and one of them — `partalias` scoring 0 of 1,024 — was a
control that passed because the geometry could not make it fire):

```
  lblk  : CAUGHT   4096 bound faults          (publishing the shipped 8 blocks/lane into a 4-block pass)
  row   : CAUGHT   2048/4096 published       (the shipped 2-lanes-per-row split in a 4-lane pass)
  stage : CAUGHT   4096 bound faults         (absolute row-slot instead of row-within-pass)
```

**What this is not:** it is not the device witness. `A6-BET`'s G1-M1 (logits and `STATEFP` byte-identical to the
shipped default on ≥ 2 shapes including a ragged tail, plus default-inertness of whatever knob carries it) is
still owed and is a device run this worker does not take. What it removes is the specific worry that the 4-pass
epilogue silently re-points a published float.

## 2. G0-M0 — seven warps: 65,536 B, spill 0, 1 block/CU → 7 wf/CU

`prototype/attn_percost/a6_g0_warps7.py` moves the one geometry fact in the two headers that own it
(`FATTN_WARPS 3 → 7` + `FATTN_THREADS 96 → 224` in `engine/attn_fattn_kernels.h:204-205`, and
`FATTN_WARPS_PER_BLOCK 3 → 7` in `prototype/attn_decode/attn_decode_kernels.h:543`, which is what
`fattn_rtiles_of` counts blocks of). AGENTS' §34 trap is exactly this one — a geometry fact committed in one of
two headers left the tip not compiling — so the guard is the one that cannot be skipped: `a6_build.sh` compiles
the patched **pair**, and the kernel's own
`static_assert(FATTN_ROWS_PER_BLOCK == FATTN_WARPS * FATTN_MROWS)` is the compile-time witness that they moved
together. Nothing else is changed: M stays 16 rows/warp, Q stays staged, the epilogue stays 16 rows deep.

Against `A6-BET`'s G0-M0 line: `group_segment_fixed_size == 65,536` ✓ read; `vgpr_spill_count == 0` ✓;
`private_segment_fixed_size == 0` ✓; derived blocks/CU == 1 → **7 wf/CU** ✓. And a free extra: `vgpr_count` is
**unchanged at 178**, i.e. the warp count costs no registers at all, which matters because G0 is the one
candidate in A5 §3 whose residency was claimed decidable *without* the register-file question — now confirmed
rather than asserted (1 block/CU × 224 threads × 178 VGPR = 39,872 VGPR/CU, derived, under any file that exists).

```
7 warps x 16 rows x 256 dims x 2 B (Q, staged)  = 57,344 B
1 aliased K/V stage                             =  8,192 B
                                                  -------
block LDS                                       = 65,536 B   = the entire CU budget
blocks/CU = floor(65,536 / 65,536) = 1  ->  1 x 7 warps = 7 wf/CU        [derived]
```

Grid consequence, recorded because it constrains the A/B's canaries: `fattn_rtiles_of(512)` goes
`ceil(512/48) = 11 → ceil(512/112) = 5`, so **grid.z changes and the launch count does not** (one
`attn_fattn_wave` per attention layer per wave body either way). The A/B's plan/launch-count canary therefore
stays an equality, and G0's "the change is live" canary must be `warps=`, `lds=`, `wf_per_cu=` — not the plan
bill. §47's harness lesson again: the canary has to be a field the change is *required* to move.

## 3. G0-M0b / G1's gate — the occupancy ladder, and what it says about the register file

### 3.1 What was built, and where it departs from the brief

`prototype/attn_percost/a6_rf_ladder.py` inserts eight `__global__` kernels
(`a6_lb1 … a6_lb8`), one per `__launch_bounds__(FATTN_THREADS, N)`, into a quarantined header copy just inside
`namespace engine::kernels`; `a6_rf_probe.inc` supplies the `main()` that asks the driver per kernel; and
`a6_rf_ladder.sh` builds one probe binary per side (shipped body and G1 body). One departure from the wording of
the brief, stated rather than slipped: **it does not extend `prototype/attn_wmma/occupancy_fattn_probe.cpp`, it
copies its pattern into a new probe.** The brief's file links `wmma_fattn_score.cpp` — the prototype's own kernels
— while this question has to be asked of the *engine* body, and the first attempt did exactly what the brief
said (append `main()` to the launcher TU `engine/kernels_attn.cpp`) and failed to link: the launcher references
`engine::launchstats::attn()` and `Family::record(int)`, which live in another TU. The header defines both
`attn_fattn_wave` and `attn_fattn_body`, so a TU that merely includes the header carries them and needs no link
partners. Everything the old probe taught is kept: `dynamicLDS` stays 0 because every byte is static, and the stub
canaries come first because a query that silently fails is the empty-parse class with a number attached.

### 3.2 The driver's answers — the stub ladder first, because it is what makes the rest readable

`bash prototype/attn_percost/a6_rf_run.sh /tmp/a6rf`, run under the lock. The stubs hold LDS and almost nothing
else (2 VGPR), so their blocks/CU **is** the driver's LDS term — and it reproduces `A5` §4.1's derived column at
every rung, which is what licenses reading a real kernel's shortfall as a register question:

```
  stub            LDS/block   driver blocks/CU   A5 §4.1 derived    wf/CU
  a6_stub_0            0            21                —              63
  a6_stub_8192     8,192             8                 8             24
  a6_stub_12288   12,288             5                 5             15
  a6_stub_20480   20,480             3                 3              9      <- G1's figure
  a6_stub_32768   32,768             2                 2              6      <- shipped figure
  a6_stub_65536   65,536             1                 1              3      <- G0's figure
  CANARY PASS (12,288 B -> 5 want 5; 20,480 B -> 3 want 3)
```

The `65,536 B -> 1` row is also G0-M0b's witness without building a 7-warp body: the driver allows exactly one
block of G0's size, which is the whole of G0's residency claim (A5 §3.1 said the shape was "driver-confirmed" for
a stub; it is now confirmed by one measured on this box today).

And the real kernels:

| side | body | VGPR (driver `numRegs` = note block) | LDS | **driver blocks/CU** | wf/CU | its own LDS term (stub) |
|---|---|---|---|---|---|---|
| base | `attn_fattn_wave` | 178 | 32,768 | **2** | **6** | 2 |
| g1 | `attn_fattn_wave` | 185 | 20,480 | **3** | **9** | 3 |

**The gate in `A6-BET` — "if the driver refuses 3 blocks at ≤ 190 VGPR, G1 is dead without any device A/B" — did
not fire.** The driver grants 3, at 185 VGPR and 20,480 B, and grants the same number of blocks as a 2-VGPR stub
of the same LDS size. There is no register-file shortfall to attribute, so **the device A/B for G1 is owed** (that
is main-agent window work, and this file's §4's harness is for G0 first, per A6-BET's order).

### 3.3 The rung table is flat, and that is the finding about the register file

`__launch_bounds__(96, N)` did not squeeze either body at any rung: `numRegs` is **178 on every base rung and 185
on every G1 rung, scratch 0 at all 16 rungs** (read from the probe rows: every `a6_lb*` row prints the same
`numRegs` and `scratch=0` as the unhinted kernel, and the stubs print `numRegs=2` — and the note blocks in the
same objects confirm it, `vgpr_spill_count 0` everywhere). So there is no "rung where the driver's answer stops rising":
the driver computes its answer from the emitted object, and the emitted object did not change. What LLVM did
instead was say so in the build log, and those lines **are** the compiler's own register-file statement, quoted
as emitted:

```
base: a6_lb4 … a6_lb8   "failed to meet occupancy target given by 'amdgpu-waves-per-eu'… desired occupancy was N,
                         final occupancy is 3"                     (5 refusals: N = 4,5,6,7,8)
g1:   a6_lb6 … a6_lb8   "… desired occupancy was N, final occupancy is 5"  (3 refusals: N = 6,7,8)
```

(counted by the ladder script into `/tmp/a6rf/<side>/occupancy_refusals.txt` as 5 and 3, and quoted from the
side's `compile.log` so the N values are visible rather than counted by hand)

Both finals are exactly the LDS-derived block count translated into wavefronts per SIMD — base: 2 blocks × 3 warps
÷ 2 SIMDs = 3; g1: 3 blocks × 3 warps ÷ 2 = 4.5, rounded up to 5 — so in both bodies **the limit the compiler
hits is LDS, not registers, and no rung shows a register ceiling before the LDS limit.** Read the other way, as a
bound: to satisfy `minBlocks=4` the base body would have had to fit 4 × 96 × 178 = 68,208 lane-VGPRs and LLVM said
it could not, while at N = 3 the same arithmetic is 51,264 and it said it could. That brackets the file between
those two numbers for the shipped body, and between 53,280 (3 × 96 × 185, which G1 needs and got) and 71,040
(4 × 96 × 185, which it was not asked for) for G1.

The arithmetic that settles A5 §3.2's honest limit, labelled **derived**: at 96 threads and 185 VGPR, three
blocks need `3 × 96 × 185 = 53,280` lane-VGPRs. The most pessimistic file this part has ever been quoted at is
65,536 — **53,280 ≤ 65,536, so 3 blocks fit even there**; the trace's rival figure of 248 VGPR would need 71,424
and would not fit a 64K file, which is exactly the caveat A5 §3.2 wrote. What this resolves is which of those two
figures the runtime is using: the driver's `numRegs` on the loaded object — the same field `engine/kernels_attn.cpp`'s
`wf_per_cu=` banner reads — is 185, agreeing with the note block, so the 248 in `A3-result.md` §2b is not what
this driver loaded. **The register file is still not named as a number**, and G1 does not need it named: the
question A6-BET asked was "3 blocks at ≤ 190 VGPR, yes or no", and the answer is yes.

### 3.4 The device protocol, reported as AGENTS requires of a worker device run

```
command : bash prototype/attn_percost/a6_rf_run.sh /tmp/a6rf          (blocking, foreground, one subagent)
lock    : tools/gpu_lock.sh acquire a6rf-ladder … release a6rf-ladder; status before it read `free`
util    : 92 % at acquire, 68 % at release (this box serves its own tokens; discount it)
wall    : 0.26 s for the two driver-query runs — the SCRIPT's wall, reported because AGENTS asks for it
runs    : taken twice with byte-identical ROW/CANARY output (the second after an `LC_ALL=C` fix to the script's
          own wall print). Full transcript: /tmp/a6rf/run_output.txt; LLVM's refusals are per side in
          /tmp/a6rf/<side>/occupancy_refusals.txt and its compile.log.
NO TIMING TAKEN. No kernel was launched: the binaries call hipSetDevice, hipGetDeviceProperties,
hipFuncGetAttributes and hipOccupancyMaxActiveBlocksPerMultiprocessor and nothing else. Nothing in this
section may be copied into a timing document.
```

The two rows the bet turns on, verbatim out of that transcript:

```
ROW attn_fattn_wave  threads=96  numRegs=178  scratch=0  lds=32768  blocks/CU=2  wf/CU=6   base
ROW attn_fattn_wave  threads=96  numRegs=185  scratch=0  lds=20480  blocks/CU=3  wf/CU=9   g1
```

## 4. The A/B the main agent runs for G0

`prototype/attn_percost/run_ab_a6g0.sh`, modelled on `run_ab_a4l1.sh` and carrying its fixed discipline, plus
`prototype/attn_percost/mk_a6g0_parsecheck.py` for the colour test.

```
BASE=<pre-change sha> bash prototype/attn_percost/run_ab_a6g0.sh     # the window (main agent, blocking, zero subagents)
PARSE_CHECK=1 bash prototype/attn_percost/run_ab_a6g0.sh            # CPU only: colour-tests the verdict parser
SUMMARIZE=1  bash prototype/attn_percost/run_ab_a6g0.sh              # CPU only: re-judge the logs, touch no device
```

`PARSE_CHECK=1` was run by this worker and passes — four scenarios, four colours, including the tie-break case:

```
scenario live            family +11.0 %                        -> LIVE        (expected LIVE)
scenario unresolved      family +6.5 %, traced +2.0 %          -> UNRESOLVED  (expected UNRESOLVED)
scenario live_by_trace   family +6.5 %, traced +7.0 %          -> LIVE        (expected LIVE)
scenario dead            family +1.0 %                         -> DEAD        (expected DEAD)
```

The fixtures are generated from the harness's own constants — the five row ratios **and** `FAMILY_LIVE_IMPROVE /
FAMILY_DEAD_IMPROVE / TRACE_TIEBREAK`, read out of `run_ab_a6g0.sh` by regex, with a refusal if the constants are
out of order — so a line moved in the harness moves the fixture or the generator aborts. That is
`mk_a4l1_parsecheck.py`'s lesson applied before the fact rather than after one.

What the harness guarantees, each one a refusal rather than a comment: the baseline is a `git worktree` at a
**pinned pre-change sha** with `BASEDIR`/`NEWDIR` named after it (not `HEAD`, A4's defect), and the g0 side is the
**same sha** plus `a6_g0_warps7.py`, so the patch is the only difference; the note blocks of both **linked**
binaries are read in-window and must be `178 0 0 32768` and `spill 0 / group 65,536`; the banners that produced
the milliseconds must print `warps=3 lds=32768 wf_per_cu=6` and `warps=7 lds=65536 wf_per_cu=7`; `launches ==
plan_launches` and `counts_match=1` on both sides (G0 moves grid.z, never the launch count); the coverage fields
`vtiles / wave_rows / split / fattn_rows / fallback_rows / covered_*` must be **identical**, because `fattn_covers`
does not read a warp count; every parse is `need`-checked; the wide legs go through `--ids-file` with the field
count asserted; there is no inert side for a compile-time change, so the window must pass a **rep-spread** check
(≤ `SPREAD_TOL`, default 3 %) in place of A4's base-vs-single NULL; `LC_ALL=C`; `SUMMARIZE=1` re-judges without
touching the device; externality is printed and decides nothing. The numerics legs — 640 ids and a ragged 1,111,
outside the hold because they are deterministic — go through `engine/attn_form_band.py`, which imports the bands
from the file that owns them and refuses a missing row instead of calling a crashed run a band difference; a
STATEFP match is reported as *vacuous*, not as a pass, because for G0 A6-BET expects bits to move and the banner
canary is the live-change witness.

The verdict is A6-BET's falsifier rule and nothing else: the **family** row decides, `≥ 10 %` → LIVE / `< 3 %` →
DEAD / between → UNRESOLVED unless the traced row improves `≥ 5 %`, and the wave/16k/32k rows are printed with
their A6-BET ratios marked **[G1 ship gate, informational here]** because G0's predicted −3.2 % wave move is
inside §37's floor and therefore cannot adjudicate anything.

## 5. What this cannot see

* **Nothing here is a speed.** Not one millisecond figure exists in this file. Every residency number is an
  *allowance* — the driver
  saying how many blocks fit — and §28's measured null (GRP4 → GRP2, a 66 % occupancy increase, ≤ 5 % time) is
  the live alternative hypothesis that only the A/B can adjudicate. G0 is precisely the experiment that decides
  between §28 and §31, and it has not been run.
* **The grid has to be big enough for an allowance to matter.** Derived from the launcher's own grid expression,
  `engine/kernels_attn.cpp:485` — `dim3 fattn_grid(N_Q, split, fattn_rtiles_of(rows))` — with
  `prototype/attn_decode/attn_decode_kernels.h:23` `#define N_Q 8`: a 4,096-lane prompt's widest body at `split = 8` is
  `8 × 8 × 11 = 704` blocks shipped against `40 CUs × 2 blocks = 80` blocks resident, and `8 × 8 × 5 = 320`
  blocks under G0 against `40 × 1 = 40` — so both are grid-saturated and 9-or-7 wavefronts/CU is realisable, not
  just allowed. A body with a small `split` at a small `rtiles` need not be, and nothing here measures which
  bodies are.
* **G1's bit-identity is argued and modelled, not measured on a device.** this file's §1.4's model is arithmetic over the
  index functions; `A6-BET`'s G1-M1 (logits and `STATEFP` byte-identical on ≥ 2 shapes with a ragged tail, plus
  default-inertness of whatever knob carries it) is still owed, and so is the same for the un-stage half.
* **The register file is still not a named number.** this file's §3.3 brackets it between 53,280 and 71,040 lane-VGPRs and
  shows the driver's model allows G1's 3 blocks, but no field on this part states the file, and the trace-vs-object
  disagreement about VGPRs (`A3-result.md` §2b) is *sidestepped* here — the probe reads the same field the
  launcher's banner reads, so it cannot contradict the banner.
* **The driver's `blocks/CU` is a capacity, not a schedule.** It says nothing about L1/L2 residency of G1's
  re-read Q (A5 §3.2 prices the extra demanded traffic at 568 KB/token/layer and names VMEM queue depth as the
  first ceiling), nothing about the two extra epilogue barrier pairs G1 pays, and nothing about the A5 §3.1 hazard
  — at 1 block/CU the four barriers per round have no second resident block to overlap with.
* **`prototype/attn_wmma/fattn_ship_census.py` was not run and not touched.** It is a known-red gate on this path
  (`E1` §9). Its `q_unstaged_text` mutation still models the un-stage by deleting the Q stage while the epilogue
  indexes it — which is the thing `E1` §9 warned a future repair about, and `a6_g1_epi8.py` is the first spelling
  in the tree that shows what the un-stage must be paired with.
* **Nothing under `engine/` was modified**, and no G0/G1 variant exists on the shipping path: the variants are
  quarantined patchers, and the A/B builds them into scratch worktrees from a pinned sha. If G0 is ever adopted, its
  two-header move must land in **one commit** and be built from the committed tree (AGENTS' two-header trap).

## 6. The commands, and what each one is

```sh
# compile-and-read, CPU only, no device touched
bash prototype/attn_percost/a6_build.sh /tmp/a6build            # base | unstage | g1 | g0 note blocks
SIDES="base g1" bash prototype/attn_percost/a6_build.sh /tmp/a6g1only
# the whole-TU drift census (8 kernels), per side, then diff base vs the variant:
python3 prototype/attn_wmma/fattn_residency_read.py --asm /tmp/a6build/g1/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s --all
python3 prototype/attn_percost/a6_epi8_model.py                 # the G1 epilogue witness (coverage/identity/bounds)
python3 prototype/attn_percost/a6_epi8_model.py --negative      # its must-catch set
# the register-file ladder: build is CPU only; running the probe needs the lock
bash prototype/attn_percost/a6_rf_ladder.sh /tmp/a6rf           # build both sides' probes
bash prototype/attn_percost/a6_rf_run.sh /tmp/a6rf              # driver queries only, NO TIMING TAKEN
# the G0 A/B — MAIN AGENT ONLY (timings)
PARSE_CHECK=1 bash prototype/attn_percost/run_ab_a6g0.sh        # CPU only, colour test
BASE=<sha> bash prototype/attn_percost/run_ab_a6g0.sh           # the window
```

Files this task added, all under `prototype/attn_percost/`: `a6_g1_unstage.py`, `a6_g1_epi8.py`,
`a6_g0_warps7.py`, `a6_build.sh`, `a6_epi8_model.py`, `a6_rf_ladder.py`, `a6_rf_probe.inc`,
`a6_rf_ladder.sh`, `a6_rf_run.sh`, `run_ab_a6g0.sh`, `mk_a6g0_parsecheck.py`, and this file.

