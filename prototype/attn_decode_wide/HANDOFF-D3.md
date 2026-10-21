# HANDOFF-D3 — decode attention at long context: attribution, what was built, what to measure

The briefed route: `attn = 3.023 ms/step` at **4,032 lanes** against a KV-read floor of **0.21 ms/step**,
and `attn = 0.496 ms/step` at **128 lanes**. Pre-registered lines: **WIN ≤ 1.50 ms** at 4,032 lanes and
**≤ 0.40 ms** at 128 lanes; **KILL ≥ 2.55 ms** at 4,032 lanes; between the lines is neither and gets
reported as neither.

Everything is prototype-only in `prototype/attn_decode_wide/`. Nothing in `engine/`,
`engine/performance-status.md`, `README.md`, `progress/`, `speccheck/` or any gate threshold was touched,
and nothing was committed (`git status --short` shows one untracked directory). **No millisecond figure in
this file was measured by the worker who wrote it**: each is quoted from the ledger/`results.md` or is
arithmetic from `attribute_d3.py`. The device work done here was correctness-only — `gate.py`, ~7 s wall
per run, `tools/gpu_lock.sh util` 26-92 % at the time, **no timing taken**.

| file | what it is |
|---|---|
| `attn_decode_wide_kernels.h` | `attn_core_lean` — `attn_core` copied from `prototype/attn_decode/attn_decode_kernels.h` lines 251-401 with five permitted differences (the name, and four `nd` loop spellings). The frozen header was **not** edited; `gate.py --check-source` re-derives that claim from the two files. |
| `attn_decode_wide.hip.cpp` | the rig: same plan format as the shipped op, arms `--core ship\|lean`, `--split N`, `--neg NAME`, `--mode once\|burst`. |
| `gate.py` | device gate: source-provenance, bit-identity, pin report, legal-cut probe, derived negatives. Does **not** take the GPU lock itself. |
| `census.sh` | compile-only witness: LDS/VGPR of both cores out of the generated GCN, plus the residency each gets on this part. |
| `attribute_d3.py` | the arithmetic in §1, runnable, CPU-only. |
| `loop_bill.py`, `isa_bill.txt` | the static per-loop instruction bill of both cores, and its output. `isa_bill.txt`'s header records why the two cores' bills must not be compared with each other. |
| `build.sh` | `-O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib --save-temps=obj` (temps kept on purpose — `census.sh` and `loop_bill.py` read them). |

`.gitignore` gained eight exact-path lines so the binary and `--save-temps` debris are ignored the way the
AGENTS `.gitignore` trap demands — named by file, never `prototype/**/attn_decode_wide`, which would match
the directory and hide the sources. `git status --ignored` was checked after adding them: the seven source
files above are trackable, the build outputs are not.

---

## 1. The attribution — with the bound classification, and the size of what it leaves over

Reproduce every number with `python3 prototype/attn_decode_wide/attribute_d3.py`.

**Geometry, from the launcher's own expressions** (`engine/kernels_attn.cpp:92` `split_for()`, and
`blocks_for_row`/`split_for_lanes` in the op header): grid `(N_Q=8 heads, split, nrows)`,
`split = ceil(lanes/MAXSC)`, `MAXSC = 512`, `TILE = 8` lanes per tile, 6 attention layers per step
(3, 7, 11, 15, 19, 23).

| | 4,032 lanes | 128 lanes |
|---|---|---|
| split, span per block, tiles per block | 8, 504, **63** | 1, 128, **16** |
| blocks per layer / per step | 64 / 384 | 8 / 48 |
| launches per step (staged: prep + core, + combine above MAXSC) | 18 | 12 |
| CUs busy at 1 block/CU, and rounds | 64 over 40 → **2 rounds** | 8 over 40 → 1 round |
| tile-trips one CU must serialise, per layer | **126** | 16 |

**Traffic: not the wall, and the cheap check says so plainly.**

| quantity | 4,032 lanes | 128 lanes |
|---|---|---|
| unique KV bytes/step (2 KV heads × 256 dims × BF16 × K+V × 6 layers) | 49.55 MB | 1.57 MB |
| floor at the measured 237 GB/s plateau | **0.209 ms** (the route's figure, reproduced) | 0.007 ms |
| **demanded** bytes/step, re-use factor **4.00× = GQA** | **198.18 MB** | 6.29 MB |
| demanded ÷ measured time | **65.6 GB/s = 27.7 % of the plateau** | 12.7 GB/s = 5.4 % |

The re-use factor the route suspected is confirmed and read off the loops, not asserted: one K row and one
V row of the *shared* KV head are read once per (query head, lane) pair by exactly one warp and staged
only for the width of one tile, so the four heads of a GQA group each fetch the same 1,024 B — a re-use of
**1× per touch, 4× wasted across the group**. Closing that waste entirely buys at most 0.836 − 0.209 =
0.63 ms of the 3.023, and only if the line were bandwidth-bound. It is not: bytes demanded divided by the
time actually taken is **27.7 %** of the ceiling this part can reach, which is the AGENTS-shaped
refutation (a demanded-÷-time figure *far below* the plateau cannot be a bandwidth wall, and D21's warning
that a figure *near* it proves nothing either).

**Issue: not the wall either.** The static bill is in `isa_bill.txt`, produced by `loop_bill.py` from this
build's own annotations, and reading it has one trap: the shipped core's `nd`-bounded dot loops are
**rolled and unannotated**, so their bodies fall outside every annotated loop and its per-loop numbers
(226 one-pass, 137 + 118 streaming) under-count — one rolled `v_fmac_f32` sits there standing for eight.
`attn_core_lean`, whose trip counts are literals, is unrolled and bills completely: **270 one-pass,
187 + 117 = 304 warp-instructions per (head, lane) touch**. Use the complete figure: 304 × 8 heads × 4,032
lanes = **58.3 M warp-instructions per step**, which against the *measured* scalar `v_fma_f32` rate of
7.30 TFLOP/s = 114 G warp-instructions/s is **0.511 ms — 5.9× headroom**; billing the mixed
LDS/VMEM/barrier body at half or a third of that rate still leaves 3.0× and 2.0×. At 128 lanes the
headroom is 10-30×. The two cores' bills are not comparable with this tool and no claim here depends on
comparing them; the comparable facts are the LDS reservation and the VGPR.

**Bound classification: latency per tile at an occupancy the grid cannot fill — with an honest remainder.**
Per layer at 4,032 lanes the step budget is 0.504 ms over 126 serialised tile-trips per CU = **4.0 µs =
~7,200 cycles per tile-trip** — an upper bound on what one tile-trip may cost, not a measurement of what it
does cost — against a per-tile dependency chain of a few hundred cycles (K row load, 8
serial FMAs, a 5-step `ds_bpermute_b32` butterfly, a barrier, V row load, 8 more FMAs). So the two
defects below are real and both are fixable, and neither one alone accounts for the gap: the walk is
serial, under-filled and exposed, but the step also contains 18 launches, 6 prep and 6 combine kernels,
and the remainder sits there. §6 states the predictions as a function of the walk's unknown share of the
line rather than pretending to know it; the two ratio measurements in §5 are what turn that remainder into
a number.

**The structural defect, found in the code object rather than a profiler.** `attn_core` declares 6,176 B
of `__shared__` (sq/sk/sv/zs 1,024 + `red` 24 + `sc` 2,048 + pad, THREADS = 256) and its code object
reserves **38,944 B** — 32,768 B more, exactly 128 B per thread, which on this part's 64 KiB LDS per CU
(from KFD, per the AGENTS rule that LDS comes from KFD and never from a HIP query) is **one resident block
per CU**. `results.md` §12 recorded that 6.3× inflation as an unexplained asymmetry; the mechanism is
LLVM's promote-alloca-to-LDS for the two per-thread arrays whose subscripts are bounded by a **runtime**
trip count: `float qreg[8]` and the `uint4 u` the score loop indexes as `k[t]`, both looped
`for (t = 0; t < nd; ++t)` where `nd = (a.neg == NEG_DIMHALF) ? 4 : 8` is runtime because `a.neg` is a
kernel argument. Bisected on this box (`census.sh`, compile-only, no device):

| spelling | `attn_core` LDS | next_free_vgpr | resident blocks/CU |
|---|---|---|---|
| as shipped | **38,944 B** | 53 | **1** |
| only the `k[t] * qreg[t]` loop made constant-trip | 22,560 B | 52 | 2 |
| both `nd` loops made constant-trip | **6,176 B** | **45** | **4** (10 by LDS, 4 by the 1,024-thread ceiling) |
| only `float e[TILE]`'s `nt` loop made constant-trip | 38,944 B | — | 1 (no effect — that array is not the cost) |

So the shipping decode attention runs at **8 of 32 wavefronts per CU**, and the extra 32 KiB is bought
with zero information: both arrays are 8 elements, indexed 0..7, always, and the runtime bound exists
**only so the `dimhalf` negative control can be injected**.

**The second defect is population, and it binds the other width.** At 128 lanes the launch is 8 blocks on
40 CUs: 32 CUs idle, and no residency or traffic lever can reach that — 1.57 MB of unique KV at 0.007 ms
floor inside a 0.496 ms line says the line is mostly not the walk at all. At 4,032 lanes, 64 blocks over
40 CUs is 1.6 rounds: the 24 doubled CUs, at 1 block/CU, run their two blocks *serially*.

---

## 2. What was built, and exactly what is bit-identical

**Arm `lean`** (`attn_core_lean`) — the LDS reclaim. Two loop spellings:
`for (t = 0; t < nd; ++t) dot += …` → `for (t = 0; t < 8; ++t) if (t < nd) dot += …`. `nd` ∈ {4, 8}, so the
trip set and the order of the additions are unchanged and the skipped iterations add nothing; the `dot`
chain is data-dependent so reassociation is not available anyway, but the argument is the trip-set
identity, not the optimiser's mood. Measured:

* **`gate.py` section 1: 126 of 126 stage comparisons byte-identical** (`attn_pre_gate`, `gated`, `q_rope`,
  `k_rope`, `cache_keys`, `cache_values`) over **21 cases** spanning 1, 18-25, 524, 648, 1,032 and 4,104
  lanes — i.e. the one-pass route, the streaming route, the `MAXSC` boundary (511/512/513/514/515) and the
  two-to-three-window boundary (1023/1024/1025).
* Under the `dimhalf` injection — the knob that makes `nd` runtime in the first place — the two cores
  still agree byte for byte, so the guard preserves the injection it exists to serve.

**Arm `--split N`** (no kernel change) — the population lever, honoured because `blocks_for_row` reads
`a.split`. On the real 4,104-lane pin, cuts 9 (natural), 18 and 36 produce **byte-identical BF16
`attn_pre_gate`** (same md5, and each RESULT line echoes `blocks=72/144/288`, so the geometry provably
changed). Do not over-read that: it says the BF16 output did not move at these three cuts on this case, not
that re-windowing is invariant. It is consistent with §12's measured re-windowing rate rather than
proof of anything — §12 found 65 elements of 2,097,152 moved by a split change, i.e. 3.1e−5 per element,
which over 2,048 elements predicts **0.06 expected moves**, so 0 moves is the expected observation either
way. The F32 partials are not dumped, so their identity is not established and is not claimed.

**An illegal cut is refused, not measured.** With `full = 1` a row whose block count is 1 takes the one-pass
branch (`if (s_row == 1 && a.full)`) whatever its width, and that branch stages every scored lane of its
slice in `sc[(MAXSC < TILE) ? TILE : MAXSC]`. So a slice wider than `MAXSC` is an out-of-bounds LDS staging
run: the first cut of this probe ran `--split 1` on the 4,104-lane row and printed cos 0.049 and **−0.013**
— plausible-looking numbers from garbage, not a route. The binary now refuses `span > MAXSC` (the same
refusal the shipped harness makes) and `gate.py` derives its probe list from the geometry so an illegal cut
cannot appear at all.

**Case population, derived rather than typed** (§20.6's rule). Every pinned decode step at layer 3 of
`wide520f`, `wide640f`, `wide1024f`, `wide4096f` (524/648/1,032/4,104 lanes) and `dec8` (18-25 lanes),
plus nine geometries obtained by truncating a real pinned cache to 1, 511, 512, 513, 514, 515, 1023,
1024, 1025 lanes. The truncated set has no HF pin and is a **form-vs-form witness only**; it exists because
the boundary is where §19/§20's bug class lives and a route no case reaches has no tests.

**Negatives, both halves derived from the geometry.** The must-catch set follows from what an injection
touches and which branch the row lands in; the must-not-catch set falls out of the same rule, and the gate
fails if an exempt row moves or a must-catch row does not:

| injection | adjudicated on | result | derived exemption |
|---|---|---|---|
| `dimhalf` | `attn_pre_gate`, `gated` | 20/20 moved | 1-lane row: softmax of a singleton is 1.0 whatever the score arrives as (§18) |
| `nomax` | same | 10/10 moved | the 11 one-pass rows: at ≤ `MAXSC` the max comes from a separate pass, so a never-re-anchored running max cannot reach them |
| `gqaoff` | same | 21/21 moved | **none** — mis-mapping the KV head selects a different **V** as well as a different score, so there is no singleton exemption (the first run listed it as score-side and the gate caught the mistake) |
| `qswap` | all stages | 21/21 moved | none |
| `gatehalf` | `gated` | 21/21 moved | none |

Two traps this prototype reinstalled from AGENTS.md before being fixed, recorded because they are the reason
the gate is written the way it is: unnamespaced case names made three fixtures share the operand file
`decode7.L03.kc` (last-writer-wins, and the plan then reads a cache that belongs to another case), and a
driver that greps its own source for `gpu_lock` matches its own docstring — so the self-lock check looks
only inside `subprocess` calls.

**What is NOT bit-identical, reported rather than waved through.** Against the HF pins, `attn_pre_gate` on
the four > `MAXSC` cases misses this op's flat §10.1 fallback floors — **identically for both cores, at
every legal cut**:

| case | lanes | ndiff/2048 | n_bad | 1−cos | meanrel |
|---|---|---|---|---|---|
| `wide520f.decode23.L03` | 524 | 379 | 48 | 1.76e−06 | 6.7e−04 |
| `wide640f.decode7.L03` | 648 | 302 | 47 | 1.09e−06 | 4.2e−04 |
| `wide1024f.decode7.L03` | 1032 | 308 | 45 | 7.71e−07 | 3.8e−04 |
| `wide4096f.decode7.L03` | 4104 | 132 | 23 | 2.08e−07 | 1.2e−04 |
| `dec8.decode0…7.L03`, 8 steps | 18-25 | **0** | 0 | 0 | 0 |

cos and meanrel are 3-4 decades inside the floors; `n_bad` and ndiff are not. The one-pass route is
bit-exact, the streaming route is not, and the number does not depend on the cut — so what it measures is
the streaming form against a pinned single-pass reference, which is §16's question and is priced per
fixture by a route allowance (D-2) that this prototype driver deliberately does **not** read. It is not
caused by `lean` (byte-identical to `ship` everywhere). Flagged for a decision, not re-banded here.

---

## 3. Exact commands for the timing pass — blocking, one lock hold, zero subagents

Neither `gate.py` nor the binary self-locks, so the caller holds the lock once around the whole block.
Run blocking, never backgrounded (`tools/gpu_lock.sh status` before and after).

```bash
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b
bash prototype/attn_decode_wide/build.sh      # ~20 s. Keep the --save-temps=obj: census.sh reads it.
bash prototype/attn_decode_wide/census.sh     # ~1 s, CPU only

bash tools/gpu_lock.sh acquire d3-timing
bash tools/gpu_lock.sh util                   # record it; discount it

# CANARY BLOCK — read these before believing any ms below. ~8 s wall, correctness only.
.venv/bin/python prototype/attn_decode_wide/gate.py --check-source
.venv/bin/python prototype/attn_decode_wide/gate.py | tee tmp/d3_gate_out.txt
#   required readings, all four:
#     variant is the frozen kernel: 151 lines, 5 permitted differences
#     _Z9attn_core8CoreArgs                lds= 38944 B  next_free_vgpr=  53   (from census.sh)
#     _Z14attn_core_lean8CoreArgs          lds=  6176 B  next_free_vgpr=  45   <- two distinct code objects
#     RESULT d3gate cases=21 bitdiff_stages=0 negative_failures=0
#   A stale or single-kernel binary cannot print both lds lines. cases=0 or a missing RESULT line is an
#   error exit here, not a PASS: the driver asserts the arm's echoed case count against the plan.

P=tmp/attn_decode_wide/ship.plan       # 21 cases, 1-4,104 lanes, written by the gate run above
grep wide4096f $P > tmp/d3_wide.plan   # the one 4,104-lane real case: the route's width, alone

# A/B #1 — one variable (the core), traffic held identical, interleaved so drift shows as agreement.
#           ~10 s wall at iters=200 on the wide plan.
for i in 1 2 3; do
  prototype/attn_decode_wide/attn_decode_wide --mode burst --plan tmp/d3_wide.plan --core ship --iters 200
  prototype/attn_decode_wide/attn_decode_wide --mode burst --plan tmp/d3_wide.plan --core lean  --iters 200
done

# A/B #2 — the population lever at the same width, lean held fixed: one variable (the cut). ~20 s.
for s in 9 18 36; do
  prototype/attn_decode_wide/attn_decode_wide --mode burst --plan tmp/d3_wide.plan --core lean --split $s --iters 200
done
# A/B #3 — the same pair at the 128-lane line's shape (short cases, 12/48 blocks per step). ~15 s.
grep -E "dec8|bnd0511" $P > tmp/d3_short.plan
for i in 1 2 3; do
  prototype/attn_decode_wide/attn_decode_wide --mode burst --plan tmp/d3_short.plan --core ship --iters 400
  prototype/attn_decode_wide/attn_decode_wide --mode burst --plan tmp/d3_short.plan --core lean  --iters 400
done
bash tools/gpu_lock.sh release d3-timing
```

Every `RESULT` line echoes `core=`, `split_opt=`, `neg=` and a per-case `lanes=,split=,blocks=` triple, so
the canary and the measurement are in the same window (the stale-binary trap). Burst is warm-up plus
wall-clock/N over one sync, because `hipEventElapsedTime` reads exactly 0 at these sizes. **These plan
mixes are not the engine's decode step** — the route verdict needs the candidate wired into `engine/` and
the ledger's own `attn` line re-taken at 4,032 and 128 lanes; that is a decision for the parent, not
something to infer from these ratios. Quote each arm as a pair with the spread across reps, not a best.

---

## 4. Stated predictions — arithmetic, and what would refute the account

**Write X for the ms/step of the 3.023 that belongs to `attn_core`'s tile walk at 4,032 lanes.** X is
unknown — pricing it is what the A/B is for — so the predictions are stated as thresholds on X, not as
millisecond promises. The model behind them is one sentence: at 1 resident block per CU every other
wavefront on that CU is idle, so the walk costs what one CU's **serial tile-trips** cost,
`trips/CU/layer = ceil(8·split / (40 · resident_blocks)) × ceil(span / TILE)`. Every arm below is
**traffic-invariant** — each lane is touched once per query head whatever the cut, so demanded bytes stay
198.18 MB/step at 4,032 lanes — which is what makes them a test of one model instead of a mixture.

| arm | trips/CU/layer | walk cost | predicted line, from 3.023 ms |
|---|---|---|---|
| ship: residency 1, split 8 | 126 | X | 3.023 |
| deeper cut only, residency still 1 (split 18/32/36) | 112 | 0.889 X | 3.023 − 0.111 X |
| `lean`: residency 4, split 8 | 63 | 0.500 X | 3.023 − 0.500 X |
| `lean` + split 18 or 36 | 28 | 0.222 X | 3.023 − 0.778 X |

1. **`lean` alone cannot reach WIN on this model, and that is worth knowing before the run.** WIN needs
   3.023 − 0.5 X ≤ 1.50, i.e. X ≥ 3.05 ms — more than the whole line. Its ceiling is the walk being
   literally everything: 1.51 ms, which is the WIN line and misses it by a rounding error. So read `lean`
   alone as a **measurement of X**, not as a candidate: wherever it lands, X = 2 × (3.023 − it).
2. **The pair (`lean` + a deeper cut) is the candidate, and WIN is its test of "the walk is two thirds of
   the line".** 3.023 − 0.778 X ≤ 1.50 needs **X ≥ 1.96 ms = 65 % of the line**. Conversely KILL (≥ 2.55 ms)
   is what the model predicts from the pair if **X ≤ 0.61 ms = 20 %**, and from `lean` alone if X ≤ 0.95 ms.
   That is the decision content of the run: it separates "the walk is most of the line" from "the walk is a
   fifth of it" with two arms, and both answers are useful — the second one means ~2.4 ms sits somewhere
   other than the tile loop and the route as briefed is dead.
3. **At 128 lanes this route predicts nothing, and the WIN line there is out of its reach.** 8 blocks per
   launch on 40 CUs is one round at either residency, so the population lever cannot bind and the residency
   lever has nothing to fill; the unique-KV floor is 0.007 ms and the issue floor 0.014 ms of a 0.496 ms
   line, so the 128-lane line is not the walk. **0.40 ms at 128 lanes will not come from either arm here** —
   it needs the launch/prep side priced (12 launches/step there), which is a different route and the
   parent's call.
4. **The deeper cut's own cost is negligible and its numerics are bounded.** At split 36 the partials are
   8 × 36 × 258 F32 × 6 layers = 1.79 MB written **and** read per step = 0.0150 ms round trip at the
   plateau, with no extra launches, and the output at the real 4,104-lane pin did not move by one byte (§2) —
   consistent with §12's measured re-windowing rate, which predicts 0.06 expected moves over 2,048 elements,
   so treat the numerics as "not disproven at this size" rather than "proven".
5. **Falsifiers.** If neither arm moves the wide plan by more than the rep-to-rep spread, the line is not
   makespan-bound, and then — despite §1's 27.7 % — the only account left is traffic, whose decisive arm is
   a G=4 shared K/V stage (`attn_tile_body`'s lever ported to the decode route), which the predicted 4×
   demand cut would make obvious; record the attribution as refuted in that case rather than amending it.
   Free second falsifier: if `lean` moves ≈2× while `--split 36` on the shipped core moves ≈1.1× as the table
   says, residency is the binding axis and the model survives; if the deeper cut alone moves much more than
   1.1×, the model's "a round is work-conserving" step is wrong and the grid is emptier than the arithmetic.

## 5. Open items for whoever picks this up

* **X, the walk's share of the line, is the number this route was really after.** §4 turns each arm's
  measurement into it (`X = 2 × (3.023 − lean)` from the `lean` arm alone), and it is worth a ledger line
  whichever way it lands, because every remaining decode idea is sized against it.
* **The `attn_core` LDS reservation is a shipping-path finding** (the engine runs at 1 block/CU today).
  Porting `lean` into `engine/` is the same five-line spelling change to the frozen kernel plus a rebuild and
  the engine's own gate set — deliberately not done here, and the decision is the parent's.
* `attn_prep` and `attn_combine` were not timed apart. If the A/B ratios come back flat, the next split of
  the 3.023 ms is per-kernel and per-launch (18 launches/step), not per-shape, and §11's measured ~0.04 ms
  per launch doubling is the scale to price against.
* The streaming-route deviation from the pins in §2's last table needs a ruling: flat §10.1 floors, or the
  wide fixtures' priced route allowance. Two instruments currently disagree about a red, and the engine's
  decode route at > `MAXSC` should be checked against those four pins under whichever is correct.
* `gate.py` deliberately does not read a route allowance. If it is taught one, it should import it the way
  it imports the floors (`prototype/attn_decode/run.py`'s `compare`/`ndiff_limit`), not copy the numbers.
