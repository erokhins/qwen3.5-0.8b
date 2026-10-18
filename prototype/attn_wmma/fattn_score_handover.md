# `m16_fattn_score` — what was built, what it costs on paper, and the device commands handed over unrun

Built for **D62** (`progress/2026-10-16/prefill-decision-log-2026-10-16.md`), whose funded probe is
`progress/2026-10-18/llama-fattn-anatomy.md` §4.2. Everything in this file is **CPU-side**: a compile, a
census, and one host-side driver query. **No timing was taken and no kernel was launched.** No timing figure
appears anywhere in these artifacts.

Shorthands used below so no `§label` is left without its file: `anatomy §N` =
`progress/2026-10-18/llama-fattn-anatomy.md`; `ledger §N` = `bench/performance-status.md`;
`results §20.1` = `prototype/attn_decode/results.md`; a bare `§5` is this file's own section.

## 0. The state of the tree, and the choice the task asked to be stated

New files, all under `prototype/attn_wmma/` (plus `tmp/` scratch):

| file | what it is |
|---|---|
| `wmma_fattn_score.cpp` | the form, plus its two baselines. **Sibling TU**, not an edit to `wmma_m16_probe.cpp` |
| `build_fattn_score.sh` | compile-only build; leaves the listing and clang's own `-Rpass-analysis` remarks |
| `fattn_census.py` | the price: pipe-weighted slots/touch, occupancy rules, the four named report items |
| `fattn_census.output.txt` | the recorded census output (with the driver's occupancy answer recorded in it) |
| `occupancy_fattn_probe.cpp`, `build_occupancy_fattn.sh` | the driver-authority occupancy query |
| `evidence/occupancy-fattn.txt` | that query's recorded output, canary included |
| `run_fattn_burst_window.sh` | claim (iii)'s burst, **written, not run** |
| `fattn_score_handover.md` | this file |

**Why a sibling TU and not an extension of `wmma_m16_probe.cpp`.** Three reasons, all checkable:
`m16_census.output.txt` is the D57/D58 calibration witness and a new kernel in that listing would move
it; `m16_census.py`'s trip rule (`round_slots`, `m16_census.py:271`) derives the loop expansion from the
v_wmma count in the whole kernel body, which aborts on a kernel carrying a rolled 1-wmma score loop *and*
an unrolled 16-wmma P·V block (16 % 32 ≠ 0), so the new form needs its own per-region rule; and the new
form's staging writes LDS, which that file's source-text assertions forbid in its FEED kernels.

## 1. The three changes, as built

| §4.2 change | how it is built | the witness |
|---|---|---|
| (1) score tile `v_wmma_f32_16x16x16_f16`, **transcribed** map | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` with `short[16]` A/B and `float[8]` C; C read as row `lane&15`, columns `2l + lane>>4` | listing emits `v_wmma_f32_16x16x16_f16 v[1:8], v[15:22], v[23:30], v[1:8]`; `sizeof(CRegs)==32`, `sizeof(ARegs)==32` static_asserts; the `get_half2` round-trip below is the cross-check that the C map and the A/B map were read correctly |
| (2) fold = 1 cross-lane op per row per tile | serial local max over the lane's 8 values, then **one** `__shfl_xor_sync(mask, m, 16)`; the rowsum is a register accumulate with its combine outside the round | exactly **10** lane-movement ops per warp-round in the listing: 9 in the round (1 + 8 exchanges) + 1 deferred rowsum |
| (3) ≥ 8 independent **f32** P·V chains | 16 accumulator tiles over DH = 256, fully unrolled, P converted by the transcribed `get_half2` | `MMA.wmma` = 32 per round = 16 score + 16 P·V; `private_segment_fixed_size = 0` |

The roll/unroll choice on (3) is a finding: with `#pragma unroll 1` the accumulator array is indexed by
the loop variable and LLVM moves all 128 accumulator registers into **scratch**
(`private_segment_fixed_size: 544` B/thread, `.enable_private_segment 1`) while clang's own remark still
reported `VGPRs Spill: 0`. Fully unrolled: 193 VGPR, 0 B scratch. The census prints `private_segment`
beside `vgpr_spill_count` for exactly this reason.

Everything else was left alone on purpose: the shipped score round `fb(bf(fb(dot)) * SCALE)` is kept and
applied in registers (so the only numerics difference this form can introduce is the dot's summation order
plus 16-bit operands); `NWARPS = 4`, one warp per head of the GQA group; 16 Q rows × 16 KV lanes per
warp-round = 256 touches, the same denominator `m16_census.py:98` uses.

## 2. Claim (i) — occupancy: **PASSES at exactly the bar (4 blocks/CU), and only by driver authority**

```
m16_fattn_score   193 VGPR (gran 200), 18 SGPR, LDS 16,384 B, vgpr_spill 0, private_segment 0
driver            4 blocks/CU, 16 wavefronts/CU of the 32 the part allows
canary            65,536-B stub -> 1, 0-B stub -> 16, 12,288-B stub -> 5 (the recorded attn_slim_wave row)
```

Four things the main agent should not miss here:

- **The LDS budget has zero slack.** Q(8 KiB) + K(8 KiB) = 16,384 B and 65,536/16,384 = 4 exactly. V is
  **aliased over the K stage** (the `nstages == 0` aliasing llama.cpp does at `fattn-mma-f16.cuh:1252`),
  with a `__syncthreads()` between the last K reader and the first V writer; V is stored in a second
  layout (`v_addr(d, kv)`, dim-major with the 16 KV halves contiguous) because the P·V A operand needs the
  KV run contiguous. A separate V stage is 24 KiB → 2 blocks/CU, which fails the bar. So any future
  addition to `__shared__` — double-buffering, a score publish, an O staging tile — drops this form to
  3 blocks/CU by arithmetic.
- **The two occupancy rules disagree and only the driver settles it.** 193 VGPR × 128 threads: a
  98,304 lane-VGPR-per-CU file (HIP's `MaxRegistersPerMultiprocessor = 196,608`, halved per AGENTS.md's
  halved-SM-count trap) gives 3; reading 196,608 as anatomy §3.4 does gives 7, so LDS binds at 4. Every recorded
  row in `evidence/occupancy-m16.txt` is LDS-bound, so **the recorded evidence does not separate the two
  figures** — this form is the first case that does, and the driver says 4. `fattn_census.py` prints all
  three rules, validates them against five recorded rows, and records the driver's answer.
- **Passing claim (i) does not predict a speed win, and `ledger §28` is why.** That section timed the
  M = 16 re-tile at 3 blocks/CU and at 5 (`attn_m16_grp2`, shipped parity), giving back half its traffic
  advantage, and the time moved by **at most 5 %** across six fixtures; its own words are "this kernel was
  never occupancy-limited". Claim (i) is a funded bar, not a mechanism. If §28 generalises, a 4-blocks/CU
  ceiling is necessary and buys nothing on its own, and the only lever this form has is the instruction
  stream of claim (ii). Say that before the burst, not after.
- **`m16_census.py`'s register term is retired by the same evidence.** Its 512-lane-VGPR-per-SIMD budget
  predicts 1 block/CU for `m16_bf16rt` (recorded: 3) and 1 for this form (driver: 4). It reproduces the
  shipped row only because that row is LDS-bound. `fattn_census.py`'s `validate_rules()` prints this
  mismatch instead of inheriting it.

Corrections to two of my own first attempts, both live in the files: the first occupancy probe passed the
kernels' *static* LDS again as the `dynamicLDS` argument, which double-charged it and read 2 blocks/CU for
the 12,288 B control (the canary caught it: it must be 5); and the first census expanded every back-edge
region it found, which double-billed the unrolled P·V block through a pair of nested LLVM back edges
(48 v_wmma per round where the geometry says 32). Each instruction line now belongs to the widest region
covering it, and the derived canary aborts if a line ever falls inside two *expanding* regions.

## 3. Claim (ii) — pipe-weighted cost: **PASSES with room (9.72 of a 20 budget)**

Currency matters, so both are printed. PIPE-WEIGHTED = one instruction line with waits counted as work,
each `v_wmma` billed at 32 clocks of matrix-pipe time (the class roof: 4096 MAC / (32 lanes × 4
MAC/lane/clk), `m16_census.output.txt:91`); that is anatomy §3.1's currency, in which its round is 3.29 raw and
7.16 pipe-weighted. SLOTS = `m16_census.py`'s currency (waits excluded), comparable to D57's ≤ 19 bar.

| kernel | slots/touch | instr/touch | **pipe-weighted** | (36 clk sensitivity) | blocks/CU |
|---|---|---|---|---|---|
| `m16_fattn_empty` (staging + 3 barriers + sink) | 1.535 | 1.672 | 1.672 | 1.672 | 4 |
| `m16_fattn_scoreleg` (+ score tile, fold, exp, P) | 4.207 | 4.891 | 6.828 | 7.078 | 4 |
| **`m16_fattn_score`** (the form) | **4.855** | 5.844 | **9.719** | 10.219 | 4 |

Leg differences, which is where the interesting part is:

- score tile + fold + exp + P over the staging baseline: **5.16 pipe-weighted/touch** (of which
  `LDS.load_b128` 64 ops = 0.25/touch, `MMA.wmma` 16).
- P·V leg (16 independent f32 chains, 32 more LDS loads): **2.89 pipe-weighted/touch**, 0.953 raw.
- the matrix term alone is exactly 4.00/touch (32 tiles × 32 clk / 256 touches) — identical to llama's, as
  it must be, since MACs per touch are a geometry invariant. **The whole remaining budget is non-matrix:
  5.72/touch.**

Why this form is 36 % above anatomy §3.1's 7.16 for llama's round, named rather than waved:

1. **The bf16→f16 operand conversion is billed here and does not exist there.** The engine's weights and
   KV cache are BF16 (`AGENTS.md` §00.5; `data/refs` shards are BF16); llama.cpp's fattn reads an f16 KV
   cache and stages it with `cp.async`. Every staged element needs a conversion: 96 halves per lane per
   round, and the census measures **0.79 CVT-class ops/touch in staging alone**, 1.35/touch overall.
2. **The shipped score round is kept on purpose** (so the numerics difference stays confined to the dot
   order and the 16-bit operands). It is ~0.5/touch that llama's round does not pay.
3. **V is staged transposed with 32 narrow `ds_store_b16` per lane per round** (0.125/touch). llama.cpp
   transposes on the SRAM *load* instead; at DH = 256 that spelling costs 256 narrow LDS reads per lane
   per round instead of 32 narrow stores, so the store side is ~8× cheaper in instruction count. **Bank
   behaviour is unmeasured and is the open risk**: a lane's 8 contiguous dims land 32 B apart, which by
   the 4 B/wide-bank arithmetic is a 32-way conflict on the store side. `SQ_LDS_BANK_CONFLICT` over one
   launch would settle it in seconds and is listed below as a device command nobody has run.
4. The DCE sink (128 accumulator adds + one store, ≈0.5/touch by the geometry rather than isolated by the
   census) is *not* the shipping epilogue;
   §4.2 of the anatomy explicitly does not re-price pass 2's drain. Deleting the `s_sc` publish is a consequence of
   changes (1)–(2), not a fourth change: with the fold living in registers the score never leaves them,
   so the shipped `ds_store_b16_d16_hi` per touch and pass 2's re-read both disappear. That flatters this
   number by roughly the store it no longer pays, and the reader should hold it against the round (7.16),
   not against the whole-kernel 89.9 warp-inst/touch, whose denominator also carries `attn_prep`, the
   score publish, and the O epilogue.

Both baselines for context, cited not re-made: `m16_bf16gq` at 1.98 slots/touch (`m16_census.output.txt:42`,
the D58 winner, map-assumed and no softmax in the round) and `m16_scalar_ob` at 59.86 slots/touch (the
D58 control: the shipped geometry copied into the WMMA shape).

## 4. The four named report items

- **Fragment-relabelling ops:** 10 lane-movement ops per warp-round = **0.039/touch** — 1 max fold +
  8 `get_half2` C→B exchanges in the round + 1 deferred rowsum combine outside it. The emitted form is
  `ds_bpermute_b32` (the listing has 19 across the two non-baseline kernels: 9 + 10 ✓), not
  `ds_swizzle_b32`. Plus the pair packing: `CVT.bf16-widen-or-addr`-class ops, 183 per round overall.
  anatomy §3.3 prices *their* relabelling at **16 cross-lane ops per round plus ~385 `v_mov_b16_*` packing
  ops, 23 % of their round**. This form's in-round cross-lane bill is **9** — one fold plus one 8-op
  exchange, because a round here is one 16×16 KV tile rather than two — and the packing sits in the CVT
  classes above. The reason it is smaller is structural, not cleverness: one C→B exchange per round, and no
  separate KQ staging round trip.
- **Spills:** `vgpr_spill_count` 0 for all three kernels and clang's remark says `VGPRs Spill: 0` for all
  three — **but** the rolled-P·V spelling spilled 544 B/thread to private memory while that remark stayed
  0, so the note block's `.private_segment_fixed_size` (0 / 0 / 0 as built) is the figure that must be
  quoted. `fattn_census.py` prints both.
- **Does Q stay in registers?** **No, by construction** — anatomy §4.2's claim wording is "with A staged", and the
  A operand is read from LDS: 64 of the 96 `ds_load_b128` per round are the score leg's A+B operand tiles
  (2 per operand per K-step, visible as `ds_load_b128 v[15:18], v27` + `offset:16` and the same pair at
  `offset:8192` for K — one base register, two stages, the 8 KiB separation being the aliasing). Keeping Q
  in registers instead would free 8 KiB of LDS (→ 8 blocks/CU by LDS) but cost 8 VGPRs of live operand
  per K-step on top of the 128 accumulator registers, and 193 VGPR is already the term that decides
  occupancy under the more conservative register-file reading.
- **One address expression per staged object:** `q_addr` / `k_addr` / `v_addr` are the only ways to name a
  staged address; `fattn_census.py:discipline()` fails the census if any raw index into `s_q`, `s_kv` or a
  staging `dst` appears anywhere in the source (3 + 4 + 1 sites, all through a helper). The known sharp
  edge, stated rather than hidden: `s_kv` carries **two** layouts (K row-major via `k_addr`, V
  transposed via `v_addr`) on purpose — which is legal only because a `__syncthreads()` separates the last
  K reader from the first V writer. That is one step away from the `results §20.1` bug, and any refactor that moves
  a K read after the V stage would produce exactly its symptom (half a row answering with the other
  operand). The remedy if the port ever needs both live at once: un-alias and accept 2 blocks/CU, or
  split the KV round into two 8-lane sub-blocks (which breaks the 16×16 tile).

## 5. HANDED OVER, WRITTEN, NOT RUN

Nothing below was executed. A worker may not take a timing (`AGENTS.md`, "GPU execution is serialized"),
and claim (iii) is a timing claim.

### 5a. The prerequisite neither command can avoid

`m16_fattn_score` is a probe kernel: it has no launcher, no engine geometry, and no engine path. Claim
(iii) is about the engine's `attn` family row, so **both commands below require a port into `engine/`**,
which is outside this worker's write scope (the brief names `engine/`, `bench/`, `tools/` as off-limits).
The port, as one commit:

1. `engine/attn_fattn_kernels.h` + a launch site in `engine/kernels_attn.cpp`, gated by a new
   `--attn-fattn {0,1}` switch; `engine/gate.py`'s default and `decode_run.cpp`'s must move in **one**
   commit (the `--attn-tile` trap in AGENTS.md: a gate that silently judges a different binary than the
   engine ships).
2. The engine's 16-row Q block, 16-lane KV sub-block, and its real O epilogue feeding
   `attn_combine`. **The staged-Q consequence must be decided by the port, not here**: this form stages Q
   per block and so spends 8 KiB of the LDS budget that `ledger §28` shows is not what limits this family.
   The alternative is register-resident Q (8 live VGPRs per K-step on top of the 128 accumulator registers)
   with V un-aliased, which is 24 KiB of LDS and 2 blocks/CU. Whoever ports it should re-census at the
   placement it actually ships rather than assume 9.72 carries over.
3. The barrier count moves from 2 to 3 per round (Q+K published / K consumed / V published). At 4 blocks/CU
   the cost is 4 block-wide arrivals and 4 LDS round trips per round instead of 3. Unpriced by any census.
4. A one-line init banner, which is what makes the burst canary work (below):
   `ATTN_FATTN form=m16_fattn_score vgpr=193 lds=16384 vtiles=16` printed once when the path is active.
   And a per-run echo line `attn_fattn=<0|1>` beside the existing `attn_touch=<0|1>`, so every timed log
   names its own side (`run_fattn_burst_window.sh` refuses a log that does not).

### 5b. Admission — numerical acceptance (cos/meanrel + floor probe, never max-ULP)

```bash
# 0. the floor for a reordered score dot, so the band is measured and not borrowed
.venv/bin/python prototype/attn_decode/floor_probe.py --fixture dec8 --long-fixture p300
#    F1 is the score dot with a pairwise 256-way sum, which is the legal order a 16x16x16 tile computes.

# 1. the form against the frozen witness, both scopes, wide fixtures
python3 engine/gate.py --fixture dec8       --attn-fattn 1 --attn-tile 1 --scope form
python3 engine/gate.py --fixture wide1024   --attn-fattn 1 --attn-tile 1 --scope form
python3 engine/gate.py --fixture corpus48   --attn-fattn 1 --attn-tile 1 --scope full
# 2. same fixtures at --attn-fattn 0: must reproduce the shipped pins exactly (the switch is inert off)
# 3. the must-catch set, DERIVED not hand-listed: the negative controls that gate the tile wave
python3 prototype/attn_tile_model.py --neg-wide
```

The band is cos/meanrel against the pinned dumps, sized by `floor_probe.py`'s F1, plus the cancellation
band the §10.1 rule requires. **Max-ULP is not admissible here**, for two recorded reasons: a 16-bit
operand plus a legal reordering moves a normal share of elements by ≤ 1 ULP and any max-ULP bound passes
vacuously (AGENTS.md, "max-ULP bands are blind to whole-tensor 1-ULP error"), and §10.5's pins pair with
`greedy[s+1]`, so a free run must be aligned with the `feed`/`gen` index stated (`prototype/attn_decode`
and `engine/results.md`'s Q2 section). One expectation to set before running, not after: this form is
**not** bit-identical to the shipped path — it moves the dot into a 16×16×16 f16 tile and converts the
operands to 16 bits — so the admission question is exactly "is the move inside the floor-probe-sized
band", and `ndiff` here is meaningless (a chained dot's legal `ndiff` is 29–47 %, `mlp_gemv` F4).

### 5c. Burst — claim (iii), ≥ 1.5× cut of the `attn` family at 4,096 lanes

`bash prototype/attn_wmma/run_fattn_burst_window.sh` — MAIN AGENT ONLY, blocking, zero subagents, one
`gpu_lock` hold. It reuses `run_attn_attrib_window.sh`'s shape: `--time-waves` runs at 1,024 and 4,096
lanes, best-of-3 reps inside one lock hold, `gran=stage family=attn` parsed for `ms=` **and** `launches=`,
abort on a null parse. Bit-identity is not asserted (this is not a bit-identical pair); the numerics leg
is §5b and it must be green before any millisecond is quoted — per the D59 rule, the harness refuses to
print a millisecond figure when admission is red.

**The behavioural canary (the task's requirement; `plan_match` cannot detect a stale binary).** Three
checks, all read in the same window that produces the number, all aborting the script:

1. **Flag liveness.** `--attn-fattn 1` must be accepted and `--attn-fattn 7` must be refused. A binary
   whose flag is inert, or absent, aborts before timing — exactly `run_attn_attrib_window.sh`'s CANARY 0.
2. **Form banner, with fields.** `new` must print
   `ATTN_FATTN form=m16_fattn_score vgpr=193 lds=16384 vtiles=16` and `base` must print nothing like it.
   The `vgpr=`/`lds=` fields are read back against the listing's note block by the script, so a binary
   built from an *older* copy of the form — the D30 1.7′ failure, a pre-change binary timed as `new` — is
   caught by the field values, not by hoping `cmp` on the binary means something (it cannot:
   `engine/build.sh` is not bit-reproducible here).
3. **A bit-difference witness.** One 1-row prefill logits dump per side must differ (`cmp` must fail).
   Here, unlike a bit-identical pair, identical bytes would *prove* one side is stale: an f16 tile dot
   cannot reproduce a BF16 serial 256-term dot bit for bit. A negative control that must fire, with its
   reason stated, rather than a "must be equal" check that passes for the wrong reason.

### 5d. Three cheap device checks the census cannot substitute for, also unrun

- `SQ_LDS_BANK_CONFLICT` (one launch, one wave) for the transposed V store pattern named above.
  If it confirms a 32-way conflict, the fix is the alternative spelling (row-major V + 256 narrow A
  reads), which costs +0.75 slots/touch (256 narrow A reads = 1.0, against the 0.25 this spelling pays) — that is a priced choice, not an open question.
- Realised occupancy by counter, not by calculator (`SQ_WAVE_CYCLES / SQ_BUSY_CYCLES`, remembering the
  recorded correction that the architectural max is 32 wavefronts per CU, not 64). The form claims 16 of
  32; `prototype/gdn_collapse` §4–§5 is the recorded cost of trusting a calculator about realised
  occupancy instead.

#- **The fragment map itself has never been executed against.** Everything in §1 is transcription plus a
  consistency argument (the `get_half2` output shape equals the A/B operand shape, the emitted operand
  register counts are 8/8/8→8, `sizeof` is static_asserted). The one empirical witness is a **map probe**:
  one launch with a patterned Q and K (say `Q[q][d] = q`, `K[kv][d] = kv`), storing each lane's 8 C values
  and its lane id to memory, and checking that lane t holds row `t%16` at columns `2l + t/16`. That is 30
  seconds of device time and it is the check `m16_bf16frag` never had — it assumed the map and was
  disqualified for it. `prototype/attn_m2/check_m16_transcription.py` is the precedent for a witness built
  as a separate checker rather than argued in prose.

### 5e. The two extras anatomy §4.2 asks for in the same window, and where they were NOT done

§4.2 asks, in the same window, for a **WMMA latency leg** in `prototype/issue_rate_probe` (§3.2's gap: the
32-clk bill this census uses is the class roof, not a measured latency for this instruction on this part)
and, if the fixture is built at all, the **512-row × 4096-lane fixture with its must-catch set derived**
(§4.4). Neither is in this package: `prototype/issue_rate_probe/` is outside this worker's write scope, and
the fixture belongs to the gate work that only becomes mandatory if (iii) wins. Both are named here so the
window can take them alongside the burst rather than discover them after it.

## 6. Verdict, in the words the task asked for

Neither claim failed. Claim (i) passes at **exactly** 4 blocks/CU, with zero LDS slack, and only under the
driver's reading of the register file — which is the instrument anatomy §4.2 names ("per the occupancy
probe"), and its canary reproduces the recorded shipped row. Claim (ii) passes at **9.719** pipe-weighted slots/touch against a
budget of 20, of which 4.00 is the matrix itself and 5.72 is everything else — the bet anatomy §4.2 funded was
that the non-matrix term stays small when the matrix does the dot, and on paper it does, at ~1.35× the
anatomy's own round, with the gap explained by the bf16→f16 conversion the engine must pay and llama
does not. **Claim (iii) is untouched**: it needs the engine port of §5a plus a main-agent device window.
