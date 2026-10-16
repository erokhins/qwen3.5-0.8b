# gdn_rec_isa — handoff

CPU/ISA census of the shipping `gdn_rec` family, modelled on
`prototype/attn_valu_slim/`. Findings and the pricing are in `results.md` (same directory).
Nothing here was run on a device; no timing was taken by this worker. Read `results.md` first.

## State of the tree when this was written

* `engine/`, `prototype/attn_m2/`, `prototype/issue_rate_probe/` — untouched.
* New files, all under `prototype/gdn_rec_isa/`:
  * `isa_census.py` — the census. 76 patterns, each self-tested against its own example line
    (`rules … selftest PASS`). Imports the shipped-classifier machinery from
    `prototype/attn_score_dot2/isa_census.py` through `prototype/attn_valu_slim/isa_census.py`;
    nothing was copied, and the pipe table is the base's (which fixes the `v_movrels_b32`-is-SALU
    mistake `attn_valu_slim` §4 corrects).
  * `gdn_rec_isa_kernels.h` — MODE 0 `rec0_kernel` (the shipped `rec_kernel` copied character for
    character), MODE 1 `rec1_kernel`, MODE 2 `rec2_kernel`, MODE 2 control `rec2c_kernel`.
    MODE 3 (norms out of the launch) is described in the header comment and **not implemented**.
  * `isa_tu.cpp` — compile-only TU that instantiates all four modes.
  * `gdn_rec_isa.hip.cpp` + `build.sh` — the device A/B harness (§3): MODE 0 as the reference, MODEs
    1/2/2c bit-diffed against it on `o` and final state, plus the `--burst` falsifier path and two
    CPU-only selftests. Written later than the rest of this directory, so that the census verdict
    stands on the census and the harness only tests it.
  * `isa_census.output.txt` / `.s` — the census run over `isa_tu.cpp` and the listing it read.
  * `isa_census.harness.output.txt` — the same census over the **harness** TU, i.e. the listing the
    binary actually runs. Its two control rows and the three lever deltas are `results.md` §6's table.
  * `rec_kernel.ship.s` — the shipped listing, extracted from the engine TU, kept so the numbers
    here can be re-diffed without a rebuild.
* `tmp/gdn_isa/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s` — the engine TU built with
  `--save-temps` (scratch, disposable). `tmp/gdn_isa/gdn_rec_isa.s` is the harness TU's listing, from
  `SAVE_TEMPS=1 ./build.sh`, and is likewise disposable.

## What the census says (one line)

A token-step — one `rec_kernel` launch, one (token, GDN layer), 16 blocks × 128 threads, wave32 —
costs **1,523 issue slots per wavefront**, 97,472 per launch; the launch's own issue floor is
~0.7 µs against a measured 6.8–6.9 µs, so the family runs at ~10 % of its issue roof and
**instruction surgery is not the wall**. The xlane spill attention had *is* here (195 slots,
12.8 % of the bill, same instruction names) and is already 10× too small to pay. Details, numbers
and the falsifier: `results.md`.

## Commands for the main agent

Every device command here is run blocking, zero subagents, under `tools/gpu_lock.sh`.
`isa_census.py` is CPU-only and safe to run any time.

### 1. The census itself (no device)

```bash
.venv/bin/python prototype/gdn_rec_isa/isa_census.py \
    --out prototype/gdn_rec_isa/isa_census.output.txt \
    --engine-asm tmp/gdn_isa/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s
```

If `tmp/gdn_isa/` is gone, rebuild the engine listing first (`prototype/gdn_decode`'s
`--save-temps`; **`--save-temps` writes next to the `-o` path**, so pass a `-o` under `tmp/`):

```bash
mkdir -p tmp/gdn_isa && /opt/rocm/bin/hipcc -x hip --offload-arch=gfx1151 -O3 -Wl,-rpath,/opt/rocm/lib \
    -I engine -I prototype/gdn_decode --save-temps engine/kernels_gdn.cpp -o tmp/gdn_isa/kernels_gdn.o
```

Useful flags: `--dump loopB --kernel rec_kernel` (full instruction dump of a region),
`--categories` (every mnemonic seen, classified or not), `--engine-asm ''` to skip the cross-TU row.
`--help` prints the unit definition and the trip-count derivation.

### 2. Compile check for the modes (no device, ~60 s)

```bash
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -I prototype/gdn_rec_isa \
    -c prototype/gdn_rec_isa/isa_tu.cpp -o /tmp/isatu.o
```

Passes as of this writing. This is the only check that the hand-written `ds_bpermute_b32` /
`ds_swizzle_b32` operand forms assemble.

### 3. The A/B that decides the lever — **the harness exists now: `gdn_rec_isa.hip.cpp`**

Written after the verdict, to make §4 of `results.md` checkable. Shape: one process, one input tile,
one starting state, every MODE on **its own** state and `o` buffers with guard bands and an
unwritten-element sentinel (`prototype/gdn_collapse/gdn_collapse_check.hip.cpp`), and every MODE timed
inside each rep so the falsifier is one window (`prototype/attn_m2/attn_m2.hip.cpp`'s `--forms`). It
does **not** fork: `fork()` after HIP has touched the device is not an isolation mechanism
(`tools/mixed_bw`'s `run_dev_case` forks *before* `hipInit`, which is a different design), so the
isolated-process case is `--modes <one mode> --out <dir>` plus `cmp -l`, in commands 3.5-3.6 below.
This binary does **not** acquire `tools/gpu_lock.sh` — the caller holds it, and releases it before
invoking any self-locking driver (`grep -c gpu_lock prototype/*/run.py`: glue, proj_gemv,
gdn_gated_norm self-lock; this op has no `run.py`).

**3.0 Build and CPU-only checks (no device, no lock, ~90 s).**

```bash
./prototype/gdn_rec_isa/build.sh                     # -> prototype/gdn_rec_isa/gdn_rec_isa
./prototype/gdn_rec_isa/gdn_rec_isa --selftest-lane  # CPU-only, expect: SELFTEST lane cases=80660 bad=0 ok=1
./prototype/gdn_rec_isa/gdn_rec_isa --selftest-mirror# CPU-only, expect: SELFTEST mirror cases=524304 bad=0 ok=1
```

`--selftest-mirror` passing is what licenses the `*_vs_cpu` column; `--selftest-lane` passing is the
CPU-side witness that MODEs 1/2 read the source lane the compiler's own lowering reads, in every lane
(both selftests carry a must-catch control — `bad=0` is not vacuous).

**3.1 Hold the lock, and say what the box was doing.**

```bash
tools/gpu_lock.sh acquire gdn_rec_isa_ab
tools/gpu_lock.sh util                                # record this beside the numbers it belongs to
```

**3.2 The admission gate — bit-identity, no timing taken** (correctness-only, deterministic, well under
a minute, so a worker may run this part; it prints no ms):

```bash
mkdir -p tmp/gdn_rec_isa && ./prototype/gdn_rec_isa/gdn_rec_isa \
    | tee tmp/gdn_rec_isa_admission.log
```

Expected shape (4 cases × 4 MODEs): one `plan` line, then per case a `CASE` line, four `RESULT case=…
mode=…` lines, one `CPU-FLOOR` line, one `NOT TIMED` line, and one `RESULT mode=summary …` line at the
end. Read it in this order:

* `ndiff_o_vs0=0` and `ndiff_state_vs0=0` on the `rec1`/`rec2`/`rec2c` lines — **this is the gate**. Any
  non-zero value is a numerics change, not a slower lever, and `results.md` §6's last paragraph says
  which MODE means which defect.
* `rerun_ndiff_o=0 rerun_ndiff_state=0` — a non-zero value means the hand-written `s_waitcnt
  lgkmcnt(0)` in `shfl_down_bits` is racy (`prototype/attn_valu_slim`'s failure mode: run-to-run
  disagreement, not a stable wrong answer).
* `unwritten_o=0 guard=0` — bytes touched, not just values.
* `maxrel_o_vs_cpu`/`maxrel_state_vs_cpu` of order **1e-6 or smaller, and the same number on every
  MODE**. They are not zero and never will be — the device's `v_rsq_f32_e32`/`v_exp_f32_e32` are not
  glibc's `rsqrtf`/`expf`; `CPU-FLOOR` states MODE 0's own value, which every bit-identical MODE
  shares. An order-1 value instead means the *fixture* or the mirror is wrong (a q|k|v row stride, a
  head index, a transposed state) — report it, do not proceed to the burst.

**3.3 The census row, so a reader knows the harness did not move MODE 0** (no device, ~60 s):

```bash
SAVE_TEMPS=1 ./prototype/gdn_rec_isa/build.sh        # leaves tmp/gdn_isa/gdn_rec_isa.s
.venv/bin/python prototype/gdn_rec_isa/isa_census.py --asm tmp/gdn_isa/gdn_rec_isa.s \
    --engine-asm tmp/gdn_isa/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s \
    --kernels rec_kernel,rec0_kernel,rec1_kernel,rec2_kernel,rec2c_kernel \
    --out prototype/gdn_rec_isa/isa_census.harness.output.txt
```

Both CONTROL lines must read IDENTICAL at 1523 vs 1523, and the DELTA section must reproduce
1488 / 1391 / 1492 (−2.3 % / −8.7 % / −2.0 %). Taken as of this writing: they do, which is what
`results.md` §6's table records. (`--save-temps` writes after the *source* stem in the CWD, not the
`-o` path; `build.sh` moves it under `tmp/`.)

**3.4 The falsifier — MODE 0 vs MODE 2 in one window, still inside the same lock hold.** This is a
timing; the main agent takes it, and only the main agent may put a number from it into
`bench/performance-status.md`.

```bash
./prototype/gdn_rec_isa/gdn_rec_isa --modes rec0,rec2 --burst 16 --reps 5 \
    | tee tmp/gdn_rec_isa_falsifier_rec0_rec2.log
```

Expected shape: the same admission lines first (it re-gates before it times — if admission fails, the
run prints `NOT TIMED … reason=admission_not_clean` and no timing at all, because a timing taken on a
MODE that is not bit-identical to the reference is not a lever measurement), then per rep one `BURST
case=… mode=… ms_per_sequence=… ms_per_launch=…` line per MODE, then per case a `RESULT case=…
mode=burst … best_rec0_ms=… best_rec2_ms=… no_timing_taken=0` line and a `FALSIFIER case=… mode=rec2
against=rec0 … rec0_over_mode=… per_launch_rec0_us=… band=…` line. `burst=16` means 16 copies of the
whole 128-token sequence (2,048 launches per timed sample), steady_clock, one sync per sample, state
restored before each sample so every MODE walks the trajectory that was gated; `ms_per_launch` is the
number the ledger bills the family at (6.8-6.9 µs today), so a reader can check the harness's own MODE
0 against it. Quote the pair (`best_rec0_ms`, `best_rec2_ms`), state the spread across reps, and record
`util` — the ledger's rule (`bench/performance-status.md` §6) is that a short-prompt row means nothing
unless both sides were measured in one window, which is why the burst alternates MODEs inside each rep
rather than looping one MODE and then the other.

**3.5 Then the attribution pair, same window, same lock hold** (so 2c-vs-2 separates "fusion" from
"hand spelling", and ship-vs-rec0 re-witnesses the census's control row on device if wanted):

```bash
./prototype/gdn_rec_isa/gdn_rec_isa --modes rec0,rec2,rec2c --burst 16 --reps 5 \
    | tee tmp/gdn_rec_isa_falsifier_triple.log
./prototype/gdn_rec_isa/gdn_rec_isa --modes ship,rec0,rec1,rec2,rec2c --burst 8 --reps 3   # optional, all five
```

**3.6 If a MODE faults instead of answering** (the hand `ds_bpermute_b32`/`ds_swizzle_b32` forms are the
candidates, and per the traps table a device fault on a path nothing else exercises presents as a hang,
not a crash — Ctrl-C and isolate rather than waiting):

```bash
mkdir -p tmp/gdn_rec_isa/dump
./prototype/gdn_rec_isa/gdn_rec_isa --modes rec0        --out tmp/gdn_rec_isa/dump   # reference dumps
./prototype/gdn_rec_isa/gdn_rec_isa --modes rec2 --out tmp/gdn_rec_isa/dump         # one MODE, own process
cmp -l tmp/gdn_rec_isa/dump/t128-zeros-d0.4.rec0.o.bin tmp/gdn_rec_isa/dump/t128-zeros-d0.4.rec2.o.bin
```

`--out` writes `<case>.<mode>.o.bin` and `<case>.<mode>.state.bin`, so a MODE that dies alone can still
be bit-compared against MODE 0's files. Note that a process that dies mid-way silently skips every case
after it, which is why the isolated runs print a `CASE` line before doing anything.

**3.7 Release the lock** (before anything that self-locks, and before the counter pass in §4 if that is
next):

```bash
tools/gpu_lock.sh release gdn_rec_isa_ab
tools/gpu_lock.sh status        # stale holders show as `STALE (holder pid dead)`
```

**What each threshold decision means** (`results.md` §6 restates this with the reasoning):

| `rec0_over_rec2` | decision |
|---|---|
| **> 1.10** | The latency reading is refuted — the prologue chain is on the critical path, so the barrier/guard cut is a lever after all. Fund MODE 3 (norms out of the launch, one pre-pass per token for all 18 layers; still unwritten, and it costs one launch per token, which is exactly what `prototype/m2_launch`'s D22 measured as nearly free). |
| **< 1.02** | The instruction route is dead for this family and D23's null stands as a measurement rather than a model. Close this directory: `gdn_rec` moves by raising work per launch (the scan route), not by slimming the stream. |
| **between** | Inconclusive — re-take on the long-memory cases (`t128-carried-d0.002`, `t128-zeros-d0.002`, whose state resembles a real wave), then the `rocprofv3` counters in §4 as the tie-breaker. Do not average the two bands into a verdict. |

### 4. If it survives, the counters that explain it

```bash
tools/gpu_lock.sh status
/opt/rocm/bin/rocprofv3 --kernel-name-recapture 0 -f csv -k "rec*_kernel" -- \
    counters --token-batch 1024 --token-seq 1
```

`SQ_WAIT_ANY` against `SQ_BUSY`, plus `SQ_LDS_BANK_CONFLICT` and `SQ_LDS_ACTIVE`. Two things the
census could not decide and this can: whether the wavefront is waiting on LDS or on VMEM, and
whether the 512-byte-stride LDS coefficient read (64 distinct banks × 16 B) actually conflicts.

## What is *not* built, and the decisions sitting with the main agent

1. ~~**The harness.**~~ **Built** — `gdn_rec_isa.hip.cpp` + `build.sh`, protocol in §3. It compiles and
   both CPU selftests pass; **no device run has been made with it**, so the admission gate and the burst
   are still unrun, and D23's null therefore still rests on the pricing until §3.2 and §3.4 are executed.
2. **MODE 3 (norms out of the launch)** is specified in the header but unwritten, because MODE 2
   has to survive first, and because moving the norms changes the launch *count* — which is exactly
   the lever `prototype/m2_launch` D22 measured and falsified for this family (removing 33,792
   launches bought 13 ms of 125). Say the word if you want it written anyway.
3. `.gitignore`: `prototype/gdn_rec_isa/gdn_rec_isa` (the binary) **is** ignored now, by exact path,
   per the trap noted there. `git status --ignored` after adding an op's files still catches a line
   that matches the directory instead. New files still untracked from this pass: `gdn_rec_isa.hip.cpp`
   and `build.sh` (plus the `results.md`/`HANDOFF.md`/`isa_census.harness.output.txt` edits).
4. Nothing is committed by this pass — `git status` shows `prototype/gdn_rec_isa/{gdn_rec_isa.hip.cpp,
   build.sh}` untracked and the two docs modified; `tmp/gdn_isa/` is scratch (disposable, or keep it as
   the census's input).

## Corrections to anything I said verbally earlier

* The SIMD count: `devices/strix-halo-compute.md`'s own `hipGetDeviceProperties` print says 20 CUs
  (a reporting artifact — use 40), and `rocminfo` there prints "SIMDs per CU: 2" while the roofline
  section of the same file uses 128 FP32 lanes/CU (= 4 × 32). `results.md` §3 carries the roofline
  both ways, because the conclusion (≤1 wavefront per SIMD for a 16-block launch) is the same under
  both readings. I did not resolve which is right and did not invent a third number.
* My first version of `shfl_down_bits` clamped the out-of-range source lane to 31. That is **not**
  what the shipped lowering does and would have changed lane 0's sum; the committed version emits
  the compiler's own clamp (`delta -> 0`, i.e. read your own lane) and keeps the range-select, which
  is why MODE 1 saves less than the raw guard count suggests. Written up at the top of
  `gdn_rec_isa_kernels.h`.
