# `gemm_algo_sweep` — handoff

Worker-authored, 2026-10-17, amended same day after two device faults. **Pass 1 was CPU-only: no
device run, no timing, no `tools/gpu_lock.sh`, nothing committed** (that was the brief). Pass 2 ran
short correctness-only device gates under the lock — fault localisation and `--check-only` — and
**still took no timing**: no `ms` or TFLOP/s figure in this file came from a worker. §1 says exactly
what is and is not proven, §2 is the device window for the main agent, §5 is the decision rule.

## 1. State of the tree right now

| thing | state | how it is known |
|---|---|---|
| it compiles | **yes**, zero warnings | `bash prototype/gemm_algo_sweep/build.sh` → `BUILD ok: …` |
| every API call resolves | **yes** | `ldd -r ./prototype/gemm_algo_sweep/gemm_algo_sweep` prints no `undefined symbol`, including the five `hipblaslt_ext` entry points |
| the classic enum API is absent | **yes, and provably** | `nm -D --defined-only /opt/rocm/lib/libhipblaslt.so \| grep " T hipblasLtMatmulAlgo"` → `hipblasLtMatmulAlgoGetHeuristic` only. `AlgoGetIds`/`AlgoInit`/`AlgoCapGetAttribute`/`AlgoConfigGet`/`AlgoConfigSet` are in neither the `.so` nor `/opt/rocm/include/hipblaslt/*.h` → calling one is a link error, not a refusal. README §3 |
| shape census | **green** | `./gemm_algo_sweep --list` → 7 distinct shapes, `TOTAL … = 150`, exit 0 (the `--list` return code *is* the 150 assertion) |
| shapes agree with the checkpoint | **yes** | `engine/weights_contract.header_shapes` over `models/Qwen3.5-0.8B` gives `[4096,1024] [512,1024] [1024,2048] [6144,1024] [2048,1024] [3584,1024] [1024,3584]` — every README row |
| **`--verify` works on device** | **yes** — run by the operator: exit 0, `VERIFY ok`, `problems=21 refused=0 heuristic_empty=0`, 15,750 `ALGO` lines across all 7 shapes | so enumeration is real, the heuristic answers every problem, and `getAllAlgos` is not empty. The one gap found: `problems` was counted and not itemised — fixed, see the `CASE` line below |
| **`--time` faulted, twice** | **no fault has been seen since** — the operator's next window ran to completion and died on the band instead | Fault 1 was the operand pairing (README §1a — localised by the rule *fault ⟺ n ≠ m* over 10 observations, and the band independently refused the one shape that could not fault). Fault 2 arrived after that fix with `LAYOUT ok=1` printed immediately before it, so the extents were fine and something else in the call was not. The instrumentation built to name it (`validate_args` + `FAULT-ARGS` + `--selftest-sys`) went in first, and the following window faulted zero times. **Which of those changes removed it is not known and no cause is being asserted** — what is known is that the window now reports numerics instead of dying |
| **the band refused 7 of 7 shapes** | found and fixed twice: first the floor was imported rather than measured, then the per-element test was the wrong instrument | First pass: every candidate came back `worst_ulp=1.0000 survivors=0 cos=0.999998169 verdict=WRONG cos<floor` against the shipped `--cos-floor 0.999999`, missing by 1.8e-06, with **one cos shared by all 12 candidates while ndiff wandered 79 → 236**. That deficit is now measured and named: it is **bf16 output quantisation** (three legs, spread ≤ 1 % of it — `CONFLUENCE`), and `--floor-probe` calibrates to it instead of accommodating a guess. Second pass: a flat `worst_ulp ≤ 1` also refused the engine's own call, on **one element per leg**, at `rel 1.96e-07` / `2.34e-07` of the tensor max — cancellations, which the repo's existing band form (`prototype/mlp_gemv`) excuses by magnitude rather than by widening the bound. Both halves are measured now and the ULP bound itself is **still 1.0** |
| **`--floor-probe` + `--check-only` are green on device** | **yes**, run by a worker under the lock at `--rows 512 --check-rows 8 --seeds 4` | `FLOOR ok - 7 CAL row(s)`, `refused=0 could_not_run=0`, then `CHECK-ONLY-SUMMARY problems=7 cases_printed=7 refused=0 band_passed=7` with `band_ok=12_of=12` and `engine_verdict=ok` on every shape, exit 0. `floor_calibration.tsv` is committed beside the tool. No timing was taken by either run |
| `--selftest-layout` | **PASS**, no device | `layout_lines_printed=21 expected=21 bad=0 swap_caught=17 swap_structurally_blind=4`. The 4 blind ones are exactly the 4 (shape, M) pairs where `n == m` — and exactly the 4 that did **not** fault on device, which is the two methods agreeing |
| `--selftest-sys` | **works**, no device; audited against hand-authored fixtures, **not yet against a real capture** | legal fixture: `lines_parsed=4 algos_checked=4 bad=0 neg_expected=17 neg_caught=17`, exit 0. Bad fixture (5 lines, one defect each): 5 `VIOLATION` lines naming `algo_state_success`, `algo_not_all_zero`, `workspace_requested_fits`, `algo_supported`, `algo_max_workspace_fits`, exit 1. Running it on a `--algo-dump` capture is step 1 of §2 |
| no timing numbers exist | `results.md` is a skeleton | by design |

Three defects found and fixed while writing this, because a reviewer will want to know what the first
draft was worth: (1) the winner-vs-default index comparison in the timing print loop compared a
`timed[]` loop counter against a `cand[]` index, which silently left `default_ms = -1` as soon as any
algorithm was dropped by the band; (2) a launch failure inside the timed loop returned instantly and
would have looked like a spectacular kernel — the status is now checked and the shape is voided;
(3) the correctness pass memsets, so the timed loop must **not** memset a 28–48 MiB `D` inside the
measured window — it doesn't.

And four from the fault pass, all of them the same shape of mistake:

* the operand pairing (README §1a), and `hipblasLtMatmulAlgo_t` is **24 bytes, not 16** — the first
  `--selftest-sys` fixture was rejected by the tool's own length check, which is the only reason the
  stale "16 opaque bytes" claim in README §3 got caught before it was copied anywhere.
* `state` and `support_status` were **printed and never enforced**: an entry the library marked
  not-SUCCESS still got launched. Now predicates, at the launch site.
* algos dropped for workspace or support were dropped **silently**. Now `SKIP` + `SKIP-SUMMARY`.
* `--selftest-sys` first judged its own injections by "did any tag fire", which one unrelated
  violation would have made vacuous; each injection now has to be caught by the predicate named for
  it. The instrument caught this in itself before anyone could trust it.

And one from the band pass, found only because a device window ran long enough to hit it: the cos
floor was a constant copied from another op's gate. A band copied across ops is the same category of
mistake as a must-catch set copied across geometries — it looks strict and it adjudicates nothing.

And four from the band's *second* pass, three of them caught by the negative controls that were written
to catch algorithm-level defects — which is the argument for deriving a control set rather than
hand-listing one:

* the v1 must-catch set was built from chain arithmetic (drop a term of the dot) and then refused to
  calibrate, because the worst seed of that injection sat 7e-06 from a correct kernel. No algorithm
  swap drops a term of a dot, so the set was adjudicating a failure class the tool cannot reach. It is
  now derived from what a swap can break (`ld`, operand roles, split-K partials, partial-slot alias,
  stale tile, `beta` left at 1, `alpha` doubled), and it calibrates with ~4 digits of headroom.
* `judge` reported a **NaN row as `cos = 1.0`**, because `na > 0` is false for NaN and the row cosine
  fell back to its "no data" value. The `beta_left_at_1_nan` control printed `NEG-FAIL` with
  `cos=1.000000` and that is how it was found. Non-finite output is its own predicate now.
* the confluence test demanded bit-equality across the three legs and called a `3.3e-16`…`1.2e-09`
  spread "disagreement". Agreement now has to be *relative to the deficit being measured*, which is the
  only denominator that means anything here.
* the cancellation band was sized from the wrong witness first: the double reference reordered moves
  nothing (`ndiff=0`), and F32 accumulation of bf16 products turned out to be bit-identical too, so a
  band sized only from "what the reference licenses" would have excused nothing and condemned the
  engine's own correct element. Both legs feed it now, and `cancel_rel_source=` says which one did.

## 2. The device window

**This tool does not self-lock.** Evidence: `grep -n gpu_lock prototype/gemm_algo_sweep/*.cpp` finds
two prose mentions (a header line and the `--help` footer) and no `system`/`popen`/`fork`/`flock`
call anywhere. Hold the lock *outside* it. Do not hold the lock around anything that self-locks
(`prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`) — AGENTS.md
trap row — and nothing in this directory is one of those.

```bash
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b

# ---- step 0: no device at all, free, and it checks the tool before it is aimed at the GPU.
tools/gpu_lock.sh util                      # record it; flash_serve parks at ~98 %, so it is context
./prototype/gemm_algo_sweep/gemm_algo_sweep --selftest-layout        # expect SELFTEST-LAYOUT PASS
./prototype/gemm_algo_sweep/gemm_algo_sweep --list > /dev/null       # expect exit 0 (= the 150 claim)

# ---- step 1, no lock needed: does the plan build, and can this build enumerate at all?
#          no operand buffer, no matmul launched, no timing. Seconds. --algo-dump captures the
#          algo bytes so step 1b can audit them offline with no device in sight.
./prototype/gemm_algo_sweep/gemm_algo_sweep --verify --shapes q_proj --rows 512 | head -60
tools/gpu_lock.sh acquire "gas verify"; tools/gpu_lock.sh util
./prototype/gemm_algo_sweep/gemm_algo_sweep --verify --algo-dump /tmp/gas_algos.txt \
    > /tmp/gas_verify.txt 2>&1; echo "verify exit=$?"
tools/gpu_lock.sh release "gas verify"
grep -E '^(CASE|SKIP-SUMMARY|VERIFY)' /tmp/gas_verify.txt | head -40

# ---- step 1b: still no device. The audit of what the library actually offered. If a heuristic entry
#          carries state != SUCCESS or an algo that wants more workspace than exists, this says so
#          before a kernel is asked to demonstrate it.
./prototype/gemm_algo_sweep/gemm_algo_sweep --selftest-sys --algo-file /tmp/gas_algos.txt; echo "exit=$?"

# ---- step 1c: calibrate the band. Takes no timing, but it is a device run: one default-algo call
#          + one engine call per (shape, M, seed), plus the host license/reorder legs and the derived
#          injections. The CAL key is (shape, M, check_rows), so the --rows and --check-rows used HERE
#          are the ones steps 2 and 3 may use, or they refuse.
#          STATUS: run for --rows 512 --check-rows 8 --seeds 4 (7 CAL rows, FLOOR ok) and the file is
#          committed as prototype/gemm_algo_sweep/floor_calibration.tsv. Re-run it after any change to
#          the comparator, the reference, or the library.
tag="gas probe"; tools/gpu_lock.sh acquire "$tag"; tools/gpu_lock.sh util
./prototype/gemm_algo_sweep/gemm_algo_sweep --floor-probe \
    --calibration-out prototype/gemm_algo_sweep/floor_calibration.tsv \
    --rows 512 --check-rows 8 --seeds 4 > /tmp/gas_floor512.txt 2>&1; echo "probe exit=$?"
tools/gpu_lock.sh release "$tag"
grep -E '^(FLOOR-RESULT|PROBE-REFUSE|NUMERICS-FINDING|FLOOR )' /tmp/gas_floor512.txt
#          A DISCRIMINATING green looks like this, and the difference between it and a vacuous one is
#          the whole point of the mode (see §2b):
#            FLOOR ok - 7 CAL row(s) ...          and  PROBE-REFUSE absent, NUMERICS-FINDING absent
#            CONFLUENCE ... agree=1               every shape (the legs agree; the deficit is the format)
#            NEG-CAUGHT x8 per shape, NEG-BLIND x1 (alpha_doubled, by the ULP leg), NEG-FAIL absent
#            RESOLUTION ... verdict=below_cos_resolution_at_this_K   the recorded blind spot, expected
#            FLOOR-RESULT ... cos_floor_suggested=0.9999877 mustcatch_max_cos=0.980-0.986   ~4 digits apart
#            BAND-SIZE ... ulp_bound=1.000 unchanged=1   the bound was not widened to get here
#          What MUST fire: NEG-FAIL, PROBE-REFUSE reason=negative_control_escaped (an injection escaped
#          its predicate), NUMERICS-FINDING (a >1-ULP element above --fullmag-rel: a real library
#          numerics finding, written up before any timing of that shape is quoted).
#          Re-do this for --rows 1024,4096 before step 3 if those widths are being timed, and keep the
#          same --check-rows throughout the session. Commit the .tsv: it is the provenance of every
#          band verdict below it.

# ---- step 2: the correctness pass only. Worker-safe by construction: it takes no timing, so there
#          is no ms figure to contaminate. One shape at a time keeps it short.
#          STATUS: run at --rows 512 --check-rows 8 --max-algos 12 against the committed CAL: 7 of 7
#          shapes band_ok=12_of=12, engine_verdict=ok, not_launched=0, CHECK-ONLY-SUMMARY refused=0.
tag="gas check-only"; tools/gpu_lock.sh acquire "$tag"; tools/gpu_lock.sh util
./prototype/gemm_algo_sweep/gemm_algo_sweep --time --check-only --rows 512 --max-algos 12 \
    --check-rows 8 > /tmp/gas_c512.txt 2>&1; echo "check exit=$?"
grep -E '^(CALIBRATION|LAYOUT|COMPARE-MODE|CHECK|CHECK-ENGINE|SIGNATURE|SKIP|FAULT|CHECK-SUMMARY|CASE|CHECK-ONLY)' /tmp/gas_c512.txt
tools/gpu_lock.sh release "$tag"

# ---- step 3: the measurement. Blocking, zero subagents, lock held for the whole thing.
#          NOTE --check-rows 8 on both commands: the CAL key is (shape, M, check_rows), so a step-3 run
#          at the default 16 would exit 2 with CALIBRATION-NEEDED unless step 1c also probed at 16.
tag="gemm_algo_sweep time"
tools/gpu_lock.sh acquire "$tag"; trap "tools/gpu_lock.sh release '$tag'" EXIT
tools/gpu_lock.sh util
./prototype/gemm_algo_sweep/gemm_algo_sweep --time --rows 512 --inner 8 --check-rows 8 \
    > /tmp/gas_t512.txt 2>&1; echo "exit=$?"
#          the 1024/4096 widths need their own probe first (step 1c with --rows 1024,4096)
./prototype/gemm_algo_sweep/gemm_algo_sweep --time --rows 1024,4096 --check-rows 8 \
    > /tmp/gas_tbig.txt 2>&1; echo "exit=$?"
tools/gpu_lock.sh release "$tag"
trap - EXIT

# ---- step 4: read it. grep RESULT, not ms (the RESULT line is last per shape on purpose).
grep -E '^(RESULT|SWEEP|VERDICT|REFUSE)' /tmp/gas_t512.txt /tmp/gas_tbig.txt
# ---- and if it faulted, the fault line is now self-describing:
grep -E '^(FAULT|FAULT-ARGS|ARGS-REJECT)' /tmp/gas_t512.txt
```

Step 0 and step 1b cost nothing and are the two steps that did not exist when the first fault
happened. Step 1 first, as before: if `hipblasLtCreate` fails or `getAllAlgos` returns 0, everything
after it is noise, and it costs a second instead of a window.

**And a host-only step 0b that costs nothing and did not exist before the band refused everything:**
`bash prototype/gemm_algo_sweep/cpu_check.sh` — 17 invocations covering both selftests, the census,
every band-source refusal and every probe-flag validation. None of them reach a HIP call, so it needs
no lock and can be run with the box busy.

### 2b. A calibrated green versus a vacuous green

A band that was loosened until it passed proves nothing; a band that was measured proves a great deal.
These are the lines that tell them apart, and they are all in the log, so the distinction is not a
matter of the reader's mood. The right-hand column is not hypothetical: v1 of this tool was a vacuous
green in exactly the first row's sense, and two of the harness bugs the v2 injections were written for
were found by running them once.

| check | calibrated green (what step 1c + step 2 measured on this box) | vacuous green |
|---|---|---|
| first line of the run | `CALIBRATION mode=measured file=… rows_loaded=7 check_rows=8 cancel_margin=4.00 fullmag_reject=1.0e-03 cos_override=none` | `cos_override=` anything, or `… cancellation band OVERRIDDEN …` in the provenance: an A/B against thresholds this build never measured, and any ratio quoted from it says so |
| `COMPARE-MODE … cos_floor_source=` | `measured on this build v2: cos floor 0.999988 = min over 4 seeds of {algo 0.999998, engine 0.999998} minus margin 1.0e-05; confluence ref_reorder=… license=… quantisation_deficit=2.0e-06 \| cancellation band cancel_rel=… cancel_tol=… \| check_rows=8 device="Radeon 8060S" source=…` | `… cos floor OVERRIDDEN by --cos-floor …, typed explicitly, NOT a measurement of this build` |
| `CHECK-ENGINE … verdict=` | `ok` — a known-correct shipping path (`hipblasGemmEx`, the engine's own call) is inside the band, on all 7 shapes | missing, or `WRONG` while every algo is `WRONG` too: then the band is what is broken and no algorithm could pass it. `REFUSE … attribution=engine_call_also_outside_band_so_band_or_reference_is_suspect` says this out loud |
| `CONFLUENCE … agree=` | `1` on every shape: legs `{algo, engine, ref_reorder}` agree to ≤ 1 % of the deficit they all sit at (`q_proj 0.999997974322` on three legs, spread `3.3e-16`). This is the evidence that the floor is the output format's resolution, not a tolerance being tolerated | `0`, i.e. one leg is doing something the others are not — which is a subject to name, not a floor to write |
| `BAND-SIZE … ulp_bound=` | `1.000 unchanged=1`, `license_n_off=0 device_n_off=0` on five shapes, `1` or `2` on the other two with `worst_offender=ulp2.00@rel1.96e-07` / `ulp6.00@rel2.34e-07` — cancellations, ~150× inside `mlp_gemv`'s band, each one element out of 49 152 / 28 672 | any `--max-ulp` other than 1.0, or `--cancel-margin`/`--cancel-rel`/`--cancel-tol` on the line. The bound has no calibration path on purpose; the band's thresholds come from the probe or they are an override and print as one |
| `FLOOR-RESULT … mustcatch_max_cos` vs the floor | `mustcatch_max_cos=0.985226` (`beta_left_at_1`) against `cos_floor=0.999988` — the floor still fails its hardest reachable defect by `1.5e-02`, which is ~7 000× the margin it was set with | the two converge, or `PROBE-REFUSE reason=band_would_not_discriminate` / `reason=negative_control_escaped` and no CAL row exists. A floor that cannot fail on its own negative controls is not a floor |
| `NEG-*` lines | `NEG-CAUGHT` × 8 per shape, `NEG-BLIND` × 1 (`alpha_doubled`, caught by the ULP leg at 255 ULP while cos sits `8e-09` from a correct kernel), **no `NEG-FAIL`** | `NEG-FAIL` present, or fewer than 9 `NEG-*` lines per shape — an injection was skipped, and a control set that can be empty cannot fail |
| `RESOLUTION … verdict=` | `below_cos_resolution_at_this_K`, always printed, `ratio` printed next to it (`4.54` at `q_proj`). A recorded blind spot: expected on every shape | treating it as a bug to fix, or deleting the line. It is also why `one_term_short` is not in the must-catch set: it is below a cos floor's resolution **and** unreachable by any algorithm swap |
| `SIGNATURE … verdict=` | informational either way. On this build it fires `constant_cos_reference_limited` on 4 of 7 shapes, and confluence explains why: the shared aggregate is bf16 output quantisation, not reference sloppiness. It is **never** an automatic pass and never a reason to widen anything | using the signature as a licence to ignore the band. The floor moves by probe or by typed override only |
| expected `CHECK-SUMMARY` after calibration | `CHECK-SUMMARY shape=q_proj M=512 candidates=12 band_ok=12 band_wrong=0 not_launched=0` and `CHECK-ONLY-SUMMARY problems=7 cases_printed=7 refused=0 band_passed=7 no_timing_taken=1` | the same line reached by an override, or with `band_ok` nonzero only because `--scan-filler 0` was used and `COMPARE-MODE` says `scan=partial` |

One more thing a green must not be: green with `not_launched>0` read as `band_ok`. `CHECK … verdict=NOT-LAUNCHED` entries carry a `cos` of 1.0 because they never ran, which is why `SIGNATURE` counts `launched=` separately.

### 2a. If a fault survives: one process per candidate

A device fault poisons the HIP context, so every later launch in that process fails silently — the
`mixed_bw` lesson, and the reason "skip the bad algo" cannot work in-process. The isolation unit is
the process, and `--algo-seq N` selects exactly one candidate (it implies `--check-only`), so the
shell does the forking:

```bash
tag="gas bisect"; tools/gpu_lock.sh acquire "$tag"
for s in q_proj k_proj+v_proj o_proj+out_proj in_proj_qkv in_proj_z gate_proj+up_proj down_proj; do
  for n in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./prototype/gemm_algo_sweep/gemm_algo_sweep --time --check-only --shapes "$s" --rows 512 \
        --algo-seq "$n" --check-rows 8 > "/tmp/gas_b_${s}_${n}.txt" 2>&1
    rc=$?; line=$(grep -E '^(FAULT|ARGS-REJECT|CHECK-ONLY)' "/tmp/gas_b_${s}_${n}.txt" | head -1)
    printf '%-18s seq=%-3s rc=%-3s %s\n' "$s" "$n" "$rc" "${line:0:120}"
  done
done
tools/gpu_lock.sh release "$tag"
```

That is ~84 short processes; a `FAULT` line names the (shape, seq) that kills, and the line count is
the bisection. Coarser first if time is short: `--shapes` one family at a time, then `--max-algos 1
2 4 8`. Cheaper still before any of it — read the `FAULT-ARGS` line from step 3. It exists now
precisely so this loop is a last resort rather than a first resort.

**Why `--check-rows 8` above and not 4:** the loop inherits the band, and the band needs a CAL row keyed
on `(shape, M, check_rows)`. An uncovered key exits 2 with `CALIBRATION-NEEDED` for **every**
candidate, which in a bisect loop prints 84 identical `rc=2` lines and reads exactly like "every
candidate faults". Keep the bisect's `--check-rows` equal to step 1c's.

**Budget note, arithmetic and not measurement:** per (shape, M) the run costs one launch + one full
`D` read-back per candidate (up to 48 MiB each) plus a `nsamp·N·K` host dot (up to ~1.0e8 MACs), and
then `slots × (warm + reps·inner)` launches. With a few dozen candidates × 21 problems that is well
past a minute, which is precisely why it is main-agent work. Trims, in order of value: `--max-algos
12` (the interesting part of a catalogue is nearly always in the first dozen), `--check-rows 8`,
`--shapes` one family at a time, `--scan-filler 0` (the full-`D` survivor scan is the read-back).

## 3. Flags that change what a number means

| flag | default | why you would move it |
|---|---|---|
| `--inner` | 1 | one launch per timed sample. Below ~100 µs the sync floor (~10 µs) is inside the number, so `k_proj`/`in_proj_z` at M=512 want 8. Quote the pair, not the absolute ms, for those two |
| `--reps` / `--warm` | 5 / 2 | best-of-N per algorithm, alternating within a rep |
| `--requested` | 32 | heuristic width. Raising it past what comes back changes nothing; the returned count is printed |
| `--max-algos` | 0 (all) | caps the *timed* list after `matmulIsAlgoSupported` filtering |
| `--check-rows` | 16 | rows of M dotted on the host. First and last always included |
| `--scan-filler` | 1 | 0 narrows the survivor check to the sampled rows and prints `scan=partial` in `COMPARE-MODE`, so the weakened check is visible in the line that weakened it |
| `--engine-ref` | 1 | 0 drops the `ENGINE-CALL` row (only if hipBLAS itself is the suspect) |
| `--max-ulp` | 1.0 | the ULP half of the band. **There is no calibration path for it and it has one legal value here.** A device leg past 1 ULP off the rounded reference is a bug; `--floor-probe` refuses to write a calibration when that happens |
| `--cos-floor X` | **no default** | the aggregate half, typed by the operator. Legitimate for an A/B, and every line of the run then says `cos_floor_source=operator_override`. Reaching for it to make a red gate green is the thing §2b exists to catch |
| `--floor-probe` / `--calibration-out F` | — | measures the floor: seeds × (default algo + engine call) + three reference orders + three injections, per `(shape, M)`, and writes the `CAL` rows. No timing. `--calibration-out` is required: the file *is* the product |
| `--calibration F` | `floor_calibration.tsv` beside the binary | where `--time` reads the floor from. A missing file or a missing `(shape, M, check_rows)` key exits 2 with `CALIBRATION-NEEDED` **before any HIP call**, so this failure costs no GPU window |
| `--seeds` / `--floor-margin` | 4 / 1e-05 | probe only. The margin is `cos_floor = measured worst cos − margin`, printed on every line that quotes the floor |
| `--workspace-mib` | 128 | matches `engine/kernels_gemm.cpp:60`. Lowering it changes split-K eligibility and so changes which kernels exist to compare |
| `--check-only` | off | `--time` with the timing removed: enumerate, allocate, one correctness launch per algo, stop. **No `ms` or TFLOP/s line exists in this mode**, which is what makes it legal for a worker to run under AGENTS.md |
| `--algo-seq N` | off | one candidate, implies `--check-only`. The unit of fault isolation, because only a new process escapes a poisoned context (§2a) |
| `--trace-launch 1` | 0 | print `ARGS` for every launch, not just the warm-up one. Verbose; for the window where the warm-up line is not enough |
| `--algo-dump FILE` | off | with `--verify`: save every enumerated algo (24 raw bytes as hex, `state`, support, workspace) for `--selftest-sys` |
| `--algo-file FILE` | — | with `--selftest-sys`: the capture to audit. Required; absence exits 3 rather than passing over an empty list |
| `--selftest-layout`, `--selftest-sys` | — | host-only, no HIP call at all. Exit 0 pass, 1 violation, 3 incomplete input |

## 4. Failure patterns, and what each one means

| line | meaning | what to do |
|---|---|---|
| `VERIFY ok` / `PLAN-SUMMARY … supported=N` | this build enumerates | proceed |
| `HEUR … returned=0` + `PROBLEM … heuristic refused` | the heuristic has nothing for bf16/16-bit-`C` on this arch — the real "this build cannot do the job" answer | that *is* the result. Record `status` and stop; there is nothing to time |
| `ENUM … status=… returned=0` while `HEUR` returned >0 | enumeration is heuristic-only on this build after all (the C++ ext is present but empty) | the README's fallback answer; the heuristic-ranking version is still the useful deliverable and `--time` still works over the 32 ranked algos |
| `CHECK … verdict=WRONG worst_ulp>…` on some algos | that kernel is not at the rounding point; it is excluded and never timed | nothing — that is the check working |
| `CHECK … verdict=WRONG` with `cos=nan` and `survivors=0` on **every** algo | a kernel is reading `C` at `beta=0`, so the `0xFF` (bf16 NaN) prefill poisoned it | a real `beta` defect in that kernel family, but it voids the sweep. Report it; do **not** "fix" it by prefilling with zeros (zeros are a value writes *do* store, so the survivor leg becomes decoration — the trap is the other way round here) |
| `REFUSE … default-ranked algorithm not at the band (…) attribution=` | the harness/pairing/band is suspect for that shape, and the attribution field says which side | `attribution=engine_at_band_so_the_algorithm_is_the_suspect` → that kernel really is off; leave it excluded. `attribution=engine_call_also_outside_band_so_band_or_reference_is_suspect` → the **engine's own shipping call** missed too, so the floor or the reference is the defect: re-run `--floor-probe` (step 1c), do not widen by hand and do not time the rest. `tools/prefill_mix.cpp --check` covers the same shape with the same comparator as a second opinion |
| `CALIBRATION-NEEDED file=… missing=K of P problems missing_keys=…` (exit 2) | no floor, or no floor for that `(shape, M, check_rows)` key. **No HIP call was made** | run the printed `--floor-probe --calibration-out …` command, with `--rows`/`--check-rows` matching the run you want. Or pass `--cos-floor` deliberately, knowing every line will call it an override |
| `CHECK-ENGINE … verdict=ok` | the engine's own `hipblasGemmEx` call is inside the band, so a `WRONG` on some algorithm means the algorithm | nothing — this is the witness working. It is printed in `--check-only` too |
| `CHECK-ENGINE … verdict=WRONG` | the shipping path misses the band. Since it is what `engine/kernels_gemm.cpp:171` runs today, the band is what is wrong (or the engine has a real problem, which would be far bigger news than this sweep) | re-probe. If `--floor-probe` also prints `NUMERICS-FINDING` for that shape, stop and report: that is a numerics defect on the shipping path, with an element index attached |
| `SIGNATURE … verdict=constant_cos_reference_limited` | ≥3 launched candidates share one cos bit-for-bit while `ndiff` varies: the aggregate is measuring the reference | diagnostic, not a pass and not a failure. It is the reason the floor is measured. If it is still there after calibration, more seeds (the floor is fine; the spread is just narrow) |
| `PROBE-REFUSE reason=negative_control_escaped` | one of the nine derived injections was not caught by the predicate named for it, so the band cannot be trusted to catch what it claims to | **the band or the injection is wrong, and it is never fixed by deleting the injection.** The line prints which injection, what its cos/ULP/nonfinite counts were. Two harness bugs were found this way on the first v2 run: a NaN row reporting `cos=1.0`, and a confluence test demanding bit-equality |
| `NEG-BLIND … injection=alpha_doubled` | expected, once per shape: cosine is scale-invariant, so a doubled α leaves `cos` within `8e-09` of a correct kernel. The ULP leg is what catches it (255 ULP on every element), and the line says `verdict=covered_by_ulp_leg` | nothing, unless it says `NOT_COVERED` — then the ULP leg is not firing and the band has a hole |
| `RESOLUTION … verdict=below_cos_resolution_at_this_K` | expected, once per shape: `one_term_short` moves cos by the printed multiple of what bf16 output quantisation costs (4.54 at `q_proj`), so no cos floor can both admit a correct kernel and catch it — and no algorithm swap produces it either | nothing. It is a recorded blind spot with a `PERMANENT` note, not a bug. Its sibling error is deleting it because it will not pass |
| `NUMERICS-FINDING … worst_offender_rel=` | an element more than `--fullmag-rel` (default `1e-3`) of the tensor max sits past 1 ULP on a **correct** leg, on the shipping path. That is not a cancellation and no band excuses it | write it up before quoting any timing of that shape. This did **not** fire on the 2026-10-16 census: the two crossings were at `rel 1.96e-07` and `2.34e-07`, one element each, which is what a cancellation looks like |
| `CHECK … nonfinite=<n>0` / `why=nonfinite_output(n)` | the device wrote a NaN or an inf. A NaN passes every ULP comparison for free, so it is a predicate of its own now | same root cause as the `cos=nan` row above (a kernel reading `C` at `beta=0`); the `survivors` fields say whether the filler survived too, which separates "read the NaN prefill" from "the dot overflowed" |
| `PROBE-REFUSE reason=nonfinite_aggregate` | a device leg came back with a NaN/inf in it, so the floor would be built from NaN | as above; no calibration is written, so `--time` cannot run on that shape |
| `PROBE-REFUSE reason=band_would_not_discriminate` | measured floor ≤ the hardest injection's cos, so the band could not catch its own negative controls | the reference or the sampling is wrong (a `--check-rows` of 2 on a wide shape, say). Fix the probe's inputs; do not lower the injections |
| `CONFLUENCE … agree=0` | the algo leg, the engine leg and the reordered reference do not agree to within 1 % of the deficit they all sit at | one leg is doing something the others are not. Name which before writing any band; `spread_over_quantisation_deficit` says how far off it is |
| `BAND-SIZE … cancel_rel_source=no_crossing_on_this_shape_so_nothing_is_exempt` | no element of any correct leg moved more than 1 ULP on this shape, so the cancellation band exempts nothing here (it prints `cancel_rel=1e-12`, which reads arbitrary and is not: it means "nothing to excuse") | nothing. A future crossing on that shape is then a real one and will fail closed, which is the desired direction |
| `FLOOR FAILED` / `FLOOR ok` | the probe's own verdict, with `rows_written + refused + could_not_run` required to equal `problems` | `could_not_run>0` means some `(shape, M)` never got measured — the sweep is not complete even if rows were written for the rest |
| `REFUSE … launch failed inside the timed loop` | a candidate died mid-measurement | the shape has no `RESULT`; re-run. A `TIME` line with an absurd ms is the failure mode this guard exists for |
| `PAIR … verdict=DIVERGENT` | the winner beat the default on a **different product** | the win is void regardless of the ratio. `worst_pair_ulp` and `diff_elements` say how far |
| `CASE mode=… status=…` | the itemised per-problem line: there is exactly one per problem, and `VERIFY`/`SWEEP` print `cases_printed=N` which must equal `problems=N` or the run exits 3. A count with no list is how `--verify` looked green while hiding 21 unnamed problems | if `cases_printed != problems`, the run is broken; read nothing else |
| `SKIP … reason=workspace_exceeds_allocation` / `not_supported` | a catalogue entry that cannot run this problem, or cannot run it with 128 MiB. `SKIP-SUMMARY` carries the counts even when the individual lines were collapsed to keep the log readable | nothing, unless `skipped_workspace` is most of the catalogue — then the workspace flag, not the kernel, is deciding the answer |
| `ARGS-REJECT … why=<tag>` | an argument list the predicates refused, so **no kernel was launched** for it | the tag is the diagnosis. If `algo_state_success` fires on *every* shape, the heuristic is returning non-SUCCESS states on this build: that is the library talking, report it, do not delete the predicate |
| `FAULT …` + `FAULT-ARGS …` | a launch that was accepted and still died. The context is poisoned, so that process is over — exit 4 | the `FAULT-ARGS` line is the whole argument list (pointers, extents, `ws`, algo hex, state, dtypes). That line is the report to bring; if it does not identify the bad field, go to §2a rather than guessing |
| `SELFTEST-SYS INCOMPLETE` (exit 3) | no capture was supplied, or the file had no `ALGODUMP` lines | **not a pass.** Produce the capture with `--verify --algo-dump` (§2 step 1) |
| `SELFTEST-LAYOUT` / `SELFTEST-SYS` `FAIL` (exit 1) | a real invariant violation, or an injection that escaped its named predicate | both are defects in the tool or in the capture; neither is something to waive |
| nonzero exit from `--time` | ≥ 1 shape was refused | read the `REFUSE` lines before believing `SWEEP` |

## 5. What result justifies a hand WMMA GEMM

`SWEEP … min_ratio= max_ratio=` and `VERDICT` print this verdict directly; the rule the operator
asked for, spelled out:

* **`max_ratio < 1.10` on every shape** → the library default is already within 10 % of the best
  thing the library owns, at every width measured. Tuning is closed as a route, and **a hand WMMA
  kernel has to beat the *default* by more than the gap it is chasing, on its own merits** — i.e.
  against 33–38 TFLOP/s, not against the 4.7 TFLOP/s retired figure (decision log D17). The
  `issue_rate_probe` ceiling says a hand loop *can* reach 52.63 TFLOP/s, so this verdict does not
  kill the hand kernel; it kills "just re-tune hipBLASLt instead".
* **Some shape with `ratio ≥ 1.10`** → there is a free win, and it is *not* free in code: the engine
  calls hipBLAS (`hipblasGemmEx`), so capturing an hipBLASLt winner means either a
  `hipblasLtMatmul` call path in `engine/kernels_gemm.cpp` plus a per-shape algo table, or an
  hipBLASLt-algo-cache keyed by (N, K, M). Price that per family: `mlp_gateup` (48 calls/wave, 24.99
  ms) and `proj` (150-call mix, 38.81 ms) are where a ratio lands; a ratio on a 6.42 ms family is a
  rounding error. Before funding it, re-measure the winning pair **as a pair inside one lock hold** —
  the AGENTS.md trap row "a ruler re-taken in a different window is not an A/B".
* **`engine_vs_best ≫ ratio`** (engine call much slower than hipBLASLt's best) → the story is not
  "the default heuristic is bad" but "hipBLAS and hipBLASLt pick differently". That is a route
  question — switching the engine's call to hipBLASLt with rank-0 alone — and is worth more than any
  algo table.
* **Nothing timed (`SWEEP timed=0`)** → read the `REFUSE` lines; there is no verdict, and a
  `heuristic returned 0` answer is itself a reportable result.

Whatever the numbers, the engine-shape rate the hand kernel must beat is `default_tflops` on the
`RESULT` lines, and the honest comparison is at M = 512 (the shipped per-call cap) with M = 1024/4096
as the route-width sensitivity — see README §2 before quoting either.

## 6. What is owed after the first device window

1. `results.md` — the `RESULT`/`SWEEP`/`VERDICT` lines verbatim, with the window name, `util` at
   acquire, `--rows`, and the spread across reps runs (AGENTS.md: absolute ms moves a few %, the
   ratio vs the engine call is the stable number).
2. A note in `engine/performance-status.md` **only if** a ratio changes a route decision — that file
   is main-agent-owned.
3. If enumeration turns out to be `getAllAlgos`-only-empty, README §3's fallback sentence gets a
   pointer to the actual `ENUM` line instead of the static `nm` evidence.
4. The `SELFTEST-SYS` summary line from step 1b, verbatim, beside the `--verify` counts. It is the
   only statement in the whole package about the algo bytes themselves rather than about what the
   library said about them, and if it reports `bad>0` the fault is explained before any kernel ran.
5. If the fault survives all of this, the `FAULT-ARGS` line plus the §2a bisect output. Both are
   designed to be pasted, not interpreted.
6. The `FLOOR-RESULT` lines verbatim and the committed `floor_calibration.tsv`. Without them a
   `band_ok=12_of=12` is an orphan: the number that made it green has no provenance, and §2b's first
   row cannot be filled in by a later reader. One `FLOOR-RESULT` per `(shape, M)`, plus the
   `CALIBRATION mode=measured` line from the judging run itself.
   **DONE for `M=512, check_rows=8`**: §0a of `results.md` carries all seven, the `.tsv` is committed,
   and the judging run's `CALIBRATION mode=measured … rows_loaded=7` line is quoted there.
7. **Still owed, in order:** (a) the timing pass at `M=512` (step 3), which is main-agent work because
   it is a measurement; (b) a `--floor-probe` at `--rows 1024,4096` before those widths can be judged
   at all — with no CAL row for them, `--time --rows 1024` exits 2 on the host, which is the designed
   behaviour and not a fault; (c) the real `--verify --algo-dump` capture fed to `--selftest-sys`, so
   the algo-bytes audit is done on library output rather than on hand-authored fixtures; (d) `results.md`
   §1/§3 once (a) exists.

## 7. The `--gemm lt` path exists in the engine — what to time, what to gate, what a pass looks like

Implementation landed 2026-10-17 in `engine/kernels_gemm.cpp` (+ `kernels.h`, `decode_step.cpp`,
`decode_run.cpp`, `build.sh`, `gate.py`). `--gemm lt` sends the same ten fat shapes to
`hipblasLtMatmul` instead of `hipblasGemmEx`, with **one algorithm per `(shape, M)` chosen at plan
time** (`gemm_route_plan(rows)`, called once from `decode_run` when `--gemm` was named explicitly,
and lazily on first use otherwise) and printed as one `GEMM-LT` line each. Default remains `library`;
`legacy` is an accepted synonym so a shell loop can spell both sides of the pair it is timing.
Workspace budget unchanged (128 MiB, the same buffer size, and `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`
is set to it so an algo needing more is never returned). `algo_pins()` is the static per-shape override
table — empty as shipped, keyed by `(shape, M)`, one 48-hex-char `hipblasLtMatmulAlgo_t` per entry — so a
pinned algo from §1's `RESULT` lines drops in without touching a call site. A shape with no usable algo
**throws**; nothing falls back silently.

`gate.py` knows the route: `GEMM_ROUTES = ("hand", "library", "lt")`, and `lt` is judged by the library
cascade floor (`route_is_library()` is true for both) but logged and cross-checked as its own route —
`--gemm lt` requires `NUMERICS … gemm=lt` *and* a `GEMM route=hipblaslt` line, so a gate cannot quietly
judge the binary the other route ships.

### 7a. Correctness already taken (worker, device, **no timing taken**)

Every line below is a correctness verdict. None of these may be quoted as a duration; the wall figures
are a by-product of a worker holding the lock for seconds at a time, and the AGENTS rule stands: no
worker number enters a timing doc.

| command | result |
|---|---|
| `.venv/bin/python engine/gate.py --gemm lt` (dec8) | **PASS**, 0 fails, util 91 % at acquire |
| `.venv/bin/python engine/gate.py --gemm lt --fixture wide1024f` | **PASS**, 0 fails, util 93 % — this is the 1024-lane prompt, so two 512-row bodies and the seven planned keys are the ones actually used |
| `.venv/bin/python engine/gate.py --gemm lt --fixture dec64f` | **FAIL — 1 check, §7c below**, util 93 % |
| `.venv/bin/python engine/gate.py --gemm library --fixture dec64f` (same window, for the pair) | PASS |
| `.venv/bin/python engine/gate.py --gemm hand` (dec8) | PASS — the `--gemm hand` branch of `decode_run`'s init still works |
| `.venv/bin/python engine/gate.py` (default library, dec8) | PASS — shipped route unaffected |
| `.venv/bin/python engine/gate.py --selftest --gemm lt` | SELFTEST PASS (CPU only, no device) |

One-shape bit-parity was **not** the acceptance test and was not achieved: `lt` is a different
accumulation order, so its output is inside the cascade band, not bit-identical. A same-prompt
`lt` vs `legacy` logits diff (21 rows, dec8) reads worst per-row `1-cos 6.920e-04`, worst per-row
`meanrel 3.699e-02`, pooled within band, and **both routes sit the same distance from the pin**
(`cos` to `dec8/logits_last`: lt 0.999858737, legacy 0.999819815; `meanrel` 1.827e-02 vs 2.006e-02)
with the same argmax at every step. `lt` is marginally closer to HF here, which is what §3a's
1.072× in accuracy-neutral terms looks like from the inside.

### 7b. The timing pass (main agent only)

Both sides in **one** lock hold, alternating, medians of 3 — the §1 row of
`engine/performance-status.md` §6 and the "a ruler re-taken in a different window is not an A/B" rule
make an unpaired pair of runs worthless here.

```bash
mkdir -p tmp/gemmlt && tools/gpu_lock.sh acquire gemm-lt-ab && tools/gpu_lock.sh util
for run in 1 2 3; do
  for route in library lt; do
    ./engine/build/decode_run --model models/Qwen3.5-0.8B \
        --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" \
        --steps 1 --time-waves --gemm "$route" > "tmp/gemmlt/${route}_${run}.log" 2>&1
  done
done
grep -hE "WAVE gran=stage (GROUP|family=(proj|mlp_gateup|mlp_down) )|WAVEPARITY|TOTAL" tmp/gemmlt/*.log
tools/gpu_lock.sh util && tools/gpu_lock.sh release gemm-lt-ab
```

`--time-waves` replays the prompt wave from a fresh state at both submission granularities, so the
family rows it prints are the same instrument as ledger §3 — that is what makes `library` vs `lt`
inside one hold an A/B rather than two rulers. Quote three things: the `GROUP matmul_ms` pair, the
three family `ms=` pairs, and `WAVEPARITY tokens_identical logits_bitidentical` (expect
`tokens_identical=1 logits_bitidentical=1` **within the pass**, which compares burst against stage on
the same route, not `lt` against `library` — cross-route bit-identity is not owed and §7c explains
why the band, not equality, is the witness).

Then a prefill TTFT row from the existing ruler, both routes, same window:

```bash
tools/prefill_sweep.py            # library
tools/prefill_sweep.py --gemm lt  # if it forwards the flag; otherwise call decode_run directly
```

If `tools/prefill_sweep.py` has no `--gemm` passthrough, do not add one for this: the §7b `decode_run`
pair above is the A/B, and the sweep exists to price widths, not routes.

### 7c. What a passing gate looks like, per family

The gate judges **end-to-end logits**, not families, so "per family" below means "which shape carries
the risk, and what green looks like there". Bands are `CASCADE_BAND` (`engine/gate.py`): per-row
`1-cos ≤ 2e-03`, pooled `meanrel ≤ 3e-02`, pooled `1-cos` deficit `≤ 2e-04`.

| family | shapes at risk | green looks like | measured now |
|---|---|---|---|
| `mlp_down` | `down_proj` N=1024 K=3584 | dec8 + wide1024f inside the band; §3 says the engine call is 1.231× off the best algo here, so this is where an algo pin should move time most | PASS on both fixtures |
| `mlp_gateup` | `gate_proj`/`up_proj` N=3584 K=1024, 96 calls/wave — the largest share of `matmul` | same, and this family's `ms` also carries the `silu·mul` epilogue (§3b), so do not read its `lt` delta as pure GEMM | PASS on both fixtures |
| `proj` | `in_proj_qkv` (1.075× off best), `in_proj_z` (engine call 1.359× off), `q_proj` (1.220×), `o_proj`/`out_proj` (1.138×) | same bands; `in_proj_a`/`in_proj_b` are untouched by this route (hand GEMV at any width), so a `proj` delta is entirely the five fat shapes | PASS on dec8 and wide1024f; **the one red is here**, below |

**The red, and what it actually is.** On `dec64f`, section (h)'s per-row check
(`batched` at `--wave-rows 512` vs `one-row` at `--wave-rows 1`, which on the lt route is
hand-GEMV-vs-hipBLASLt) reports worst per-row `1-cos = 2.204e-03` at `logits45.bin` against the
2e-03 band. On the *same row in the same fixture* the library route reads `1.889e-03` — lt is 17 %
further from the one-row side, not diverging (its **pooled** numbers are better than library's:
meanrel 1.429e-02 vs 1.433e-02, pooled `1-cos` deficit 1.459e-04 vs 1.488e-04). Every pin-facing
section passes: 64/64 steps paired, worst step at 2.604× its own measured route spread against a
3.50 allowance, and the sampled ids match the pins at all 64 steps.

So this is the band question, not a bug finding, and it is an operator call, not a worker call: the
band was **measured for hipBLAS-vs-hand** (§A2 of `progress/2026-10-14/route-c-audit-2026-10-14.md`),
and hipBLASLt's heuristic rank-0 reassociates slightly differently. The two ways forward are (a) widen
`CASCADE_BAND` per-row `1-cos` to ~2.5e-03 — which AGENTS' numeric-format rule allows only if the
widened band still catches every injection in the derived must-catch set, i.e. re-run
`gate.py --selftest --gemm lt` plus the two negative controls below with the new number and show it
still bites — or (b) pin, per shape, the algo whose accumulation order matches what the gate was
calibrated against, using `algo_pins()`. Do not do (a) by editing the constant alone.

The must-catch witnesses for that decision, and they already bite under `--gemm lt` **today** (CPU-only,
`--dry-run`, so they cost no device window and can be re-run after any band edit):

```bash
.venv/bin/python engine/gate.py --dry-run --gemm lt --break-band 3       # rc=1 — noise 4x outside the cap in force
.venv/bin/python engine/gate.py --dry-run --gemm lt --break-ab-logits 2  # rc=1 — a differ in section (h)'s one-row side
.venv/bin/python engine/gate.py --selftest --gemm lt                     # SELFTEST PASS
```

If a widened band still fails all three, it is still discriminating and can be adopted with its new
number recorded beside this measurement; if any of them goes green, the widening weakened the check and
the answer is (b).

### 7d. Still owed, in order

1. §7b's timing A/B (main agent, one hold). Until it exists the honest headline is §3a's, unchanged.
2. The §7c band decision.
3. The algo pin table, if 2 lands: the §1 `RESULT` lines' winning `algo` hex for `down_proj` (1.231×),
   `k_proj`/`v_proj` (engine call 1.625× off), `in_proj_z` (1.359×), `q_proj` (1.220×), written as
   `algo_pins()` entries at `M = 512`. A pin is also a different accumulation order, so it is re-gated,
   not trusted — the same three fixtures, same sections.
4. `NUMERICS` still prints `hipblas=<version>` from hipBLAS even on the lt route, and
   `numerics_provenance.json` records hipBLAS only; the hipBLASLt version is on the `GEMM` line
   (`version=10.4.1`, `hipblaslt`). Harmless today because the provenance check compares hipBLAS's
   version against itself, but if `lt` ever ships as the default that file needs the other library's
   version beside it, or the provenance witness is describing a library the run did not use.

---

**Archived 2026-10-21** from `prototype/gemm_algo_sweep/HANDOFF.md`, where it sat beside the rig it briefed. The device pass it hands over is done and its verdict is in that op's `results.md` and in the ledger, so `progress/README.md` rule 1 puts it here. The filename now carries the op name because a dated folder holds handoffs from several ops at once (rule 3's collision clause); citations naming the old path were re-pointed in the same commit.
