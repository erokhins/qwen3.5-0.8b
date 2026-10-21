# A4-L1 implementation notes — the two-accumulator score roll, graduated behind `--attn-fattn-roll`

Written by the implementing worker on tip `47d390e`, against `prototype/attn_percost/A4-BET.md` (the gates),
`A4-L1-gates.md` (Gate 0/Gate 1, and its §5 handover) and `a4_l1_two_acc.py` / `a4_l1_build.sh` (the
quarantined variant being graduated). **Status: engine change built and building; compile-side witnesses
green; the two worker device legs run and green (default inert bit-identical, `two` green through the whole
gate on the smallest fixture that launches the form); `gate.py` passthrough in and selftest-green; the A/B
harness written and its verdict parser colour-tested. Nothing committed — the tree is dirty on purpose and
the device A/B belongs to the main agent.**

## 0. What was built

`--attn-fattn-roll single|two`, default `single`. The knob is a **runtime** choice between two `__global__`s
that are both in the binary, so an A/B is one window rather than two builds (AGENTS: a ruler re-taken in a
different window is not an A/B), and so the default side is not a branch that the register allocator has to
price.

| file | what changed |
|---|---|
| `engine/attn_fattn_kernels.h` | `attn_fattn_body` is now `template <bool kTwoAcc>`; the score roll is an `if constexpr` between the shipped roll (verbatim, `#pragma unroll 1` included) and the two-accumulator roll (both tiles of a pair fetched before either `v_wmma` issues, one add after the roll). Two new shared tile loaders (`fattn_score_k_tile`, `fattn_score_q_tile`). Two kernels: `attn_fattn_wave` = `attn_fattn_body<false>` (unchanged name), `attn_fattn_wave_two` = `<true>`. |
| `engine/kernels.h` | `enum class FattnRoll { kSingle, kTwo }`; `AttnLayerArgs::fattn_roll` (default `kSingle`); `parse_fattn_roll` / `fattn_roll_name` declared. |
| `engine/kernels_attn.cpp` | `parse_fattn_roll`/`fattn_roll_name` (parse-or-throw, `parse_gdn_geom`'s shape); the launcher picks `attn_fattn_wave` or `attn_fattn_wave_two` with the **same grid, same block extent, one launch either way**; `fattn_echo` now reads `hipFuncGetAttributes` on **the kernel that actually ran** and prints a `roll=` field. |
| `engine/decode_step.h`, `engine/decode_step.cpp` | `StepConfig::attn_fattn_roll` → `AttnLayerArgs::fattn_roll`, same riding path as `attn_fattn`. |
| `engine/decode_run.cpp` | the option (parse-by-refusal), the inertness refusal, the `NUMERICS ... attn_fattn_roll=` echo, `--help`, `sc.attn_fattn_roll`. |
| `engine/gate.py` | **argv passthrough only** (§4 below): `--attn-fattn-roll` forwarded on every engine side and asserted against the run's own echo. No threshold, band, admission or floor edit. |
| `prototype/attn_percost/a4_l1_canary.py` | rewritten to read mangled names and both listing forms, + `--selftest` (§5). |
| `prototype/attn_percost/a4_l1_note_read.py` | new: reads a kernel's note block out of a **built host binary** (the stale-binary canary's instrument). |
| `prototype/attn_wmma/fattn_ship_census.py` | its two text anchors re-pointed (its selftest's Q-operand mutation and its `score_rolled` probe) so adding a second roll spelling does not silently weaken it — §3b. |

## 1. The knob, and how it is keyed

Mirror of `--gdn-geom`, including the part that is easy to get wrong:

- **parse by refusal**: `parse_fattn_roll("single"|"two")`, anything else throws with both values and what
  each means. `--attn-fattn-roll sine` exits non-zero before any weights are loaded.
- **`attn_fattn_roll_given`**: the one interaction refusal keys on *was the knob given*, not on the default.
  `--attn-fattn 0` alone stays legal (it is the shipped VALU wave path). `--attn-fattn 0` **with** the knob
  named refuses: naming a spelling of a form that does not run would echo a kernel that did not.
- **echo**: `NUMERICS ... attn_fattn=1 attn_fattn_roll=two ...`, i.e. beside the form field it qualifies, and
  `gate.py` asserts the echo equals the flag it was given (a missing field fails too — a pre-knob binary
  cannot be judged as if it carried the knob).
- **launch counts and the plan-bill row are identical**: the launcher picks a *function*, not a grid; the
  `attn` row of `LaunchCounts::describe()` and `launchstats::attn().record(rows)` are untouched, so the
  A/B's `launches=`/`plan_launches=`/`plan_match` canaries are equality checks on both sides.

Echo example (device leg, §6 — quoted there with its command line):

```
NUMERICS wave_rows_cap=512 wave_rows=16 gemm=lt hipblas=... attn_tile=1 attn_touch=1 attn_fattn=1 attn_fattn_roll=single gdn_scan=seq ...
```

## 2. One address expression, and the §20.1 reason

The two roll spellings do not each write their own stage arithmetic. Both call:

```cpp
fattn_score_k_tile(s_kv, lane, t)   // fattn_load_tile(&s_kv[fattn_kv_addr(fattn_ab_row(lane), 16 * t)])
fattn_score_q_tile(s_q, w, lane, t) // fattn_load_tile(&s_q[w * FATTN_MROWS * DH + fattn_q_addr(...)])
```

The `two` side calls them twice per trip (`t` and `t + 1`), which is the only duplication in the pair, and it
is a tile *index*, not an offset formula. §20.1's failure mode was two self-consistent layouts joined by a
branch, each agreeing with itself — a second roll spelling writing its own `&s_q[...]` is that shape one
notch down, so the header does not allow it. The file's own standing rule ("ONE ADDRESS EXPRESSION PER STAGED
OBJECT, writer and reader alike") is what these two lines extend to the roll.

## 3. Compile-side witnesses (all CPU, no device, no timing)

`engine/build.sh` (real output: `built engine/build/loader_test`, `built engine/build/decode_run`, 17.8 s).
Note blocks read out of the **linked binary**, not out of a listing of the tree
(`prototype/attn_percost/a4_l1_note_read.py`, which carves with `tools/carve_codeobjects.py` and re-uses
`fattn_ship_census.note_blocks`):

```
KERNEL attn_fattn_wave     mangled=_ZN6engine7kernels15attn_fattn_waveE8CoreArgs     vgpr_count=178 vgpr_spill_count=0 sgpr_count=24 sgpr_spill_count=0 private_segment_fixed_size=0 group_segment_fixed_size=32768
KERNEL attn_fattn_wave_two mangled=_ZN6engine7kernels19attn_fattn_wave_twoE8CoreArgs vgpr_count=200 vgpr_spill_count=0 sgpr_count=24 sgpr_spill_count=0 private_segment_fixed_size=0 group_segment_fixed_size=32768
```

- The default side's residency facts are **unchanged from HEAD's** (178 / 0 / 0 / 32,768), so adding the knob
  cost the shipping kernel nothing on the note-block axis.
- The `two` side reproduces Gate 1's quarantined reading **exactly** (200 / 0 / 0 / 32,768,
  `A4-L1-gates.md` §2.2), so the patch transplanted into the engine without changing what the compiler does
  with it.
- `v_wmma` sites in the engine build: **17** in `attn_fattn_wave`, **18** in `attn_fattn_wave_two`.
- `a4_acc_chain.py` on the engine's own listing: `--require-chains 1` OK on `attn_fattn_wave` (one tuple,
  `v[129:136]`), `--require-chains 2` OK on `attn_fattn_wave_two` (tuples `v[137:144]` and `v[129:136]` —
  the same two Gate 1 read).

### 3b. What the refactor did to the shipped kernel's machine code — a finding, reported

After stripping the code-end/alignment padding that `llvm-objdump` prints after whichever function is last
in `.text` (which moves when a second kernel joins the TU, so a raw body-length diff charges it to the
kernel), the shipped `attn_fattn_wave` body is **2048 instructions on both HEAD and this tree**, and the only
class-count difference is `v_mov_b32_e32` −2 / `v_dual_mov_b32` +2: two pairs of ordinary MOVs came out
fused. 61 aligned lines differ, all of them branch displacements shifted by that fusion. VGPR, spill,
private and LDS are identical (§3).

**So the default side is not byte-identical machine code, and the two things that moved it are the shared
tile loaders and the presence of the second instantiation.** What the task requires byte-identity of is the
output (§"numerics": logits and `STATEFP`), and that is the device leg in §6 — the instruction stream is
reported here instead of being quietly described as identical. If the device leg comes back byte-identical,
the MOV fusion changed no bit; if it does not, that is the finding, not an obstacle.

`fattn_ship_census.py`'s verdict on the shipped placement is **unchanged by this work**: the same
derived-vs-listing table (v_wmma want=17 got=17) and the same two pre-existing red lines it has on HEAD
(`lds_write_B` 588 vs derived 572; `global_read_B` 64 vs derived 48 — both present on HEAD before any of this,
reproduced in a clean `git worktree` of `47d390e`), and the same 3 `--selftest` escapes on HEAD and here
(two "an access mnemonic went missing and the census did not notice", one "the one-Q-stage-per-block
mutation's needle is gone"). No new escape was introduced, and two of its text anchors had to be re-pointed
because adding a second roll spelling made them weaker — the `score_rolled` probe used to ask "is there a
`#pragma unroll 1` anywhere in the body", which with two rolled loops in the body is a check that cannot
fail; it now names the shipped loop (`++t`, as against `t += 2`).

## 4. `engine/gate.py` — passthrough only

What is there: an argparse choice `--attn-fattn-roll single|two` (default `single`, documented as the shipped
roll), forwarded on the main run, on section (h)'s width A/B sides and on section (b2)'s teacher-forced side
— i.e. everywhere `--gdn-geom` is forwarded — and asserted against the run's `NUMERICS` echo with the same
"silence is a failure" wording the sibling checks use.

What is deliberately **not** there: no band, floor, threshold, admission or route-allowance edit. A4-BET fixes
the judge in advance as the existing §10.1 aggregates, so nothing needed adding. If the `two` side falls
outside them, that is reported (§7), not accommodated.

## 5. The canary that was inherited, and its fix (DONE)

`a4_l1_canary.py` refused real listings. Two causes, both fixed:

1. It only understood `llvm-objdump`'s `0000...<label>:` form. `--save-temps` `.s` listings — the form
   `fattn_census_build.sh` and the engine build leave behind — carry the mangled name as a **column-0
   label**, and the canary reported `REFUSE: no function matching 'attn_fattn_wave'`. Witness:
   `a4_l1_canary.py /tmp/a4l1/kernels_attn-...gfx1151.s ...` refused before the fix.
2. Names are now matched by exact short name, exact mangled name, exact demangled
   (`engine::kernels::attn_fattn_wave`) or **unambiguous** substring — and ambiguity is a refusal. That
   matters now that `attn_fattn_wave` is a substring of `attn_fattn_wave_two`: a first-match rule would let a
   canary certify one kernel and report it as the other.

`--selftest` witnesses both directions, and refuses rather than passing on a partial input set. The two
directions matter: a comparator that always says DIFFERENT would abort every honest A/B, and one that always
says IDENTICAL would certify a stale binary as a changed one (the class of failure AGENTS.md records for an
empty-parse gate — a check that cannot fail):

```
  [PASS] synthetic identical pair: expected IDENTICAL, got IDENTICAL
  [PASS] synthetic ONE injected instruction: expected DIFFERENT, got DIFFERENT
  [PASS] 'attn_fattn_wave' -> attn_fattn_wave
  [PASS] 'attn_fattn_wave_two' -> attn_fattn_wave_two
  [PASS] 'fattn_wave' -> None (refused: it is a substring of both kernels)
  [PASS] 'attn_fattn_wave' resolves to attn_fattn_wave (got _ZN6engine7kernels15attn_fattn_waveE8CoreArgs)
  [PASS] '_ZN6engine7kernels15attn_fattn_waveE8CoreArgs' resolves to attn_fattn_wave (got ...)
  [PASS] 'engine::kernels::attn_fattn_wave' resolves to attn_fattn_wave (got ...)
  [PASS] real: base vs itself: expected IDENTICAL, got IDENTICAL
  [PASS] real: base vs the two-accumulator build: expected DIFFERENT, got DIFFERENT
SELFTEST PASS: the canary says identical when the streams agree and different when they do not
```

Exit codes, checked directly rather than through a pipe (`| tail` reports `tail`'s status, which is how this
gets missed): identical pair → rc 0, differing pair → rc 1.

The real pair is `/tmp/a4l1v/{base,new}/objdump.s`, rebuilt through `a4_l1_build.sh` if absent.

## 6. Device gate — RUN, green, no timing taken

Both legs ran under `tools/gpu_lock.sh` TAG `a4l1-worker-gate`, blocking, one at a time, correctness only.
`gpu_lock.sh util` read **86 %** at acquire and **92 %** at release; per `engine/performance-status.md` §6 an
idle-but-serving box reads 24–91 % on identical clean runs, and none of these legs takes a timing, so
contention cannot move a result here. **No timing taken, and no number from this section may enter a timing
doc.** Each wall time is the wall clock of the whole command (model load included), reported as AGENTS.md
requires of a worker's device run, not a measurement of anything.

Two protocol notes, one of them a trap worth recording:

* **`tools/gpu_lock.sh` records the pid of the `acquire` call**, which for a worker is a per-bash-call shell,
  so the lock read `STALE (holder pid dead)` between my calls and the next caller could reclaim it. The
  mutual-exclusion guarantee a worker wants requires the whole sequence inside ONE shell call (which is
  exactly how `run_ab_g4.sh` and `run_ab_a4l1.sh` do it: acquire, run, release in one process).
* **`wide520f` does not launch the WMMA form at all.** Its prompt is 500 tokens, i.e. ≤ `MAXSC` 512, so every
  wave is one-pass, the form prints `ATTN_FATTN_SKIP` and is never launched. The first `two` leg therefore
  passed the entire gate while measuring nothing about the roll — the same "a control whose geometry cannot
  fire" class as `partalias`. `wide640f` (640 lanes) is the smallest fixture that does launch it, and that is
  the leg below.

### (a) default inertness — the new default against a HEAD-built binary

```bash
# HEAD built in a `git worktree` of 47d390e (CPU-only build, 17 s), then, per side:
<binary> --model models/Qwen3.5-0.8B --ids-file /tmp/a4l1_ids640.txt --steps 0 \
         --prefill-logits all --state-fp --out <dir>
```

640 ids taken from `data/refs/wide4096p/input_ids.npy[0:640]` — above `MAXSC` on purpose, so the WMMA form
owns rows (`ATTN_FATTN ... wave_rows=128 split=2 fattn_rows=128/128 fallback_rows=0 covered_rows_pct=100.0`).

| side | wall | result |
|---|---|---|
| HEAD binary | 4.15 s | 641 files + 42 `STATEFP` lines |
| new binary, default (= `--attn-fattn-roll single`) | 3.08 s | **`diff -r` over all 641 files: identical (rc 0); `STATEFP` lines identical (md5 `6a730e1f…`)** |
| new binary, `--attn-fattn-roll two` | 3.11 s | 126 of 641 logits files differ — exactly `logits515…logits639`, the 128 rows the form owned — and 34 of 42 state fingerprints differ, worst relative `n2` delta 3.1e-03 |

The third row is the proof the knob is not decorative: `two` changes exactly the rows the WMMA form writes and
nothing else, and the delta then propagates downstream (a `family=conv` fingerprint two layers on moves by
3.1e-03 relative — the bf16 cascade, not a state bug).

### (b) the `two` side through the existing gate path

```bash
.venv/bin/python engine/gate.py --fixture wide640f --attn-fattn-roll two --work /tmp/a4l1_gate_two640
# real 0m43.16s — ENGINE GATE PASS, 289 [PASS], 0 [FAIL]
```

The form launched inside that run: `ATTN_FATTN form=fattn16 q=stage roll=two warps=3 vgpr=200 lds=32768
scratch=0 vtiles=16 wave_rows=128 split=2 fattn_rows=128/128 covered_work_pct=100.0`, and the new passthrough
assertion fired in the right direction:
`[PASS] the run ran the WMMA score roll this gate is judging it under — NUMERICS echoed attn_fattn_roll=two`.
The `wide520f` run (29.9 s, also `ENGINE GATE PASS`, 373/0) is kept as the negative witness above.

CLI legs, no device semantics: `--attn-fattn-roll sine` → rc 2 naming `sine`; `--attn-fattn 0 --attn-fattn-roll
two` → rc 2 with "re-spells the WMMA wave"; `--attn-fattn 0` alone still runs (echo `attn_fattn=0
attn_fattn_roll=single`); `engine/gate.py --selftest` → `SELFTEST PASS` with the new flag in place.

## 7. Numerics — the `two` side is inside the existing bands, nothing widened

`engine/gate.py` applied its own adopted bands, unmodified, to a run whose every engine invocation carried
`--attn-fattn-roll two`. The worst readings on that run:

| check (as gate.py words it) | worst on the `two` run | band |
|---|---|---|
| every logits row inside the cascade band, per-row 1-cos | **2.406e-04** (`logits237.bin`) | ≤ 2.75e-03 |
| same, second pairing | 2.075e-04 (`logits26.bin`) | ≤ 2.75e-03 |
| pooled stage meanrel | inside band | ≤ 3e-02 |
| pooled 1-cos | inside band | ≤ 2e-04 |
| per-decode-step 1-cos median | 6.718e-05 (worst step 0.999851) | cascade floor |

No floor, band, threshold, admission or route-allowance was touched, and no check was widened, skipped or
relabelled: the diff to `engine/gate.py` is 43 inserted lines and zero deleted ones. Per A4-BET the judge for
`two` is the §10.1 aggregates rather than bit-identity, and they hold with 11x of headroom on the tightest
line. What that does *not* buy: the pinned fixtures only reach 640 lanes, so bands at 4,096+ lanes are
unpriced (§9).

## 8. The A/B harness — `prototype/attn_percost/run_ab_a4l1.sh`

Modelled on `prototype/gdn_walk/run_ab_g4.sh`. **Main-agent device window only.** One `gpu_lock.sh` hold
(acquire/run/release inside one process), `LC_ALL=C`, blocking, nothing backgrounded, and it invokes only
non-self-locking programs (`decode_run`, `tools/prefill_kernel_trace.py`).

Three sides — `base` (HEAD worktree build, pre-knob), `single` (new binary at the default: must be a NULL),
`two` (the verdict side). Legs: 4,096 lanes untraced (`--time-waves`) for the `attn` family row and
`wall_ms`; the same prompt traced through `tools/prefill_kernel_trace.py` + `trace_phase_split.py
--calibrate attn_prep --per-pass 48` for the `attention` phase row (divisor derived, not typed); and the two
wide rows at 16,384 and 32,768 lanes through `--ids-file` with the field count asserted (a 32,768-id prompt
does not fit in one argv slot at all, §39). Externality = Σ(new − base) over every family *other than* `attn`,
printed per width and gated.

Canaries, all of them asserted non-empty and all of them able to abort the window:

* **A** — `base` refuses the knob; `new` accepts `two` and refuses `sine` *by naming it*.
* **B** — the stale-binary canary, read off the **linked binaries** with `a4_l1_note_read.py`: base
  `attn_fattn_wave` = 178/0/0/32768 and **no** `attn_fattn_wave_two`; new default = 178/0/0/32768 (so the
  default side cannot have paid for the second instantiation); new `attn_fattn_wave_two` = 200/0/0/32768.
  A self-consistency check compiled into the binary under test is blind to staleness; this is not.
* **C** — inside the window, the `ATTN_FATTN` banner of the kernel that actually ran: `roll=` matches the
  side, `vgpr=` is 178/178/200 for base/single/two, and `warps`/`lds`/`vtiles`/`wave_rows`/`split`/
  `fattn_rows`/`fallback_rows` are **identical across all three sides** (the knob picks a function, never a
  grid). A side that echoed the other roll's banner would make the halves of the A/B indistinguishable while
  their ms differed.
* **D** — `family=attn` `launches=` == `plan_launches=` == `counts_match=1` and equal across sides, plus
  `TOTAL ... plan_match=1`.

Verdict: **all five** Gate-2 rows must clear their live line for `LIVE`; **any** dead line, or any externality
line, gives `DEAD`; anything else is `UNRESOLVED` and prints "do not ship, do not narrate". The line values
live in one place in the script (env-overridable) and are re-printed inside the verdict, so the table and the
verdict cannot drift. `SUMMARIZE=1` re-judges existing logs without touching the device.

**The verdict parser is colour-tested**: `mk_a4l1_parsecheck.py` writes three synthetic windows and
`PARSE_CHECK=1 bash prototype/attn_percost/run_ab_a4l1.sh` re-runs the harness over each and refuses unless
the colours come out `LIVE` / `UNRESOLVED` / `DEAD`. It earned its keep twice on its first run: the child
inherited the exported `PARSE_CHECK=1` and recursed (fixed with `env -u`), and the `unresolved` fixture had
put 32,768 lanes at 10,300 ms — past the 10,135 dead line — so the harness correctly called it `DEAD` and the
fixture was the thing that got fixed.

The command for the main agent (blocking, zero subagents, ~15 min at the default reps):

```bash
bash prototype/attn_percost/run_ab_a4l1.sh                 # the window
SUMMARIZE=1 bash prototype/attn_percost/run_ab_a4l1.sh     # re-judge the logs, no device
PARSE_CHECK=1 bash prototype/attn_percost/run_ab_a4l1.sh   # colour-test the parser, no device
# and outside the window, because it self-locks:
.venv/bin/python engine/gate.py --fixture wide640f --attn-fattn-roll two
```

## 9. What this cannot see

* **It cannot say the default side is byte-identical machine code.** It is not: two MOV pairs come out fused
  (§3b). What it can say — and does, on device — is that the default side produces byte-identical logits and
  state fingerprints against a HEAD-built binary at a geometry where the form runs at 100 % coverage.
* **No band has been priced for the `two` side above 640 lanes.** Every pinned fixture is ≤ 4,096 and only
  `wide640f`/`wide1024f`/`wide4096*` launch the form at all; the numerics admission is therefore a statement
  about the widths the gate reaches. If the A/B comes back `LIVE`, the wide rows need their own numerics leg
  before the flip becomes a default.
* **The A/B is not an admission.** It adjudicates five timing lines and one externality sum; it says nothing
  about the bands at wide widths, and it cannot see a corruption that preserves a moment sum (the `STATEFP`
  caveat, restated in gate section (f)).
* **Occupancy is inferred from the note block, not measured.** LDS 32,768 B/block bounds blocks/CU at 5; the
  compiler's `-Rpass-analysis` `waves/SIMD` line is a constant on this box (AGENTS.md) and was not used.
  A `kernel_occupancy_probe`-class read of the *new* kernel is not in this package.
* **The 512-lane boundary is a structural blind spot of every gate in this package**: at or below `MAXSC` the
  form is not launched, so `single` would pass and `two` would pass and neither would have run the kernel.
  The A/B's canary C aborts if no `roll=` banner appears, which is the guard; the worker legs above found the
  blind spot by reading the fixture's lane count.
* The `two` side was run through the gate on **one** fixture (`wide640f`) and the direct-dump comparison on
  one 640-id prompt. The gate's own fixture list is wider than that, and a full `gate.py --fixture` sweep of
  the nine fixtures at `two` was not run (each is ~30–45 s of device and the brief capped a worker's legs).
