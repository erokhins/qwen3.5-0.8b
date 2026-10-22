# FIX-CROSSROW — the work log for the one-lane causal-bound investigation

This is the running checkpoint for the funded correctness fix that follows
`prototype/mtp_verify/CROSSROW.md`.

> **CORRECTION (this session).** CROSSROW's inference — "the wide block lets query row `q` attend lane
> `lanes_of(a,q)` when that bound is odd" — is **not confirmed as a mask widening, and the dump below
> shows the effect is two orders of magnitude smaller than an attended lane would be.** The *datum*
> reproduces, the *carrier* is the wide block, and a new dump-only engine flag localises it INTO the wide
> core (`--dump-attn-rows`): row `q`'s attention output changes when lane `bound(q)` changes, while every
> input that row is entitled to is byte-identical. What changes is a **few BF16 ULPs in a handful of the
> row's 2048 output elements**, not a lane's worth of V. So it is a dependence, not an attendance — and
> §"Result 5" is what the four structural readouts do and do not exclude. **No engine fix has been
> written: naming the expression needs one more measurement, described in §"What is left".**

## State

- [x] read CROSSROW.md, both wide headers, `attn_fattn_kernels.h`'s C/A/B fragment laws, the launcher's
      wide branch (`engine/kernels_attn.cpp:638-676`), `lanes_of`/`nrows_of`/`blocks_for_row`
      (`prototype/attn_decode/attn_decode_kernels.h:194-230`).
- [ ] instrument: `prototype/mtp_verify/causal_bound.*` — a device probe that launches the ENGINE's own
      kernels (`attn_wide_wave`, `attn_wide_llama_wave`, `attn_slim_wave` as the clean control) on a
      synthetic attention problem whose output reads out, per query row, exactly which cache lanes were
      attended. Design below.
- [x] prove the predicate, name the expression and line — **NOT PROVEN, refuted at cell resolution.** The
      shipped mask text is correct under four readouts; see §"Result 2".
- [ ] minimal fix in the shipping path, no flag — **BLOCKED**: no defect located. Writing one would be
      guessing.
- [x] permanent gate for the *reproduction*, geometry-derived, with must-catch controls on the probe side:
      `prototype/mtp_verify/form_ladder.sh` (form ladder with a flag-reached-the-binary canary) and
      `crossrow_arm1.py sweep` (per-row byte-identity over the rows that must not move). The cell-level
      gate is `prototype/mtp_verify/causal_bound` with `--must-catch` (33 injections, all RED).
- [x] device confirm list for the main agent — §"Device confirm list".
- [x] instrument: `prototype/mtp_verify/causal_bound.{cpp,sh}` — five encodings, three forms, three
      injections, all implemented and run on device.
- [x] dump-only engine flag `--dump-attn-rows` (default off, byte-identical off AND on) plus
      `attn_dump_diff.py` / `attn_dump_shape.py` — the instrument that localises the defect to the wide
      core's own arithmetic. See Results 4-6.
- [ ] CoreArgs parity table, probe vs engine — §"Result 0". Matches on every field the wide kernel can
      read; the two differences (`cap`, and the pointers the kernel provably does not read) are named.

## Why an output-encoded probe, and not a `printf` of `acc`

The mask reads correctly at every inspection site (`attn_wide_kernels.h:200,259-262`,
`attn_wide_llama_kernels.h:379,391`), so a dump of `acc[L][c]` after the mask would be read through the
same expression that is under suspicion. The probe instead measures the mask's *effect* end to end, with
no mirror of the mask anywhere:

- `Q = 0`, `K = 0` ⇒ every score is exactly 0.0, so every **live** cell gets weight exactly 1 and every
  masked cell exactly 0 (`expf(-1e30f - 0) = 0`), and `l` is exactly the count of live cells.
- The output is then the plain mean of `V` over the attended set, divided by an exact integer:
  `aout[q][h][d] = bf(fb(Σ_{j ∈ live(q)} V[j][d] / |live(q)|)`.
- Encode the test in `V`, one column `d` per threshold: `V[j][d] = (j >= t_d ? 1 : 0)`. Then
  `aout[q][h][d] == 0` **exactly** iff row `q` attended no lane `>= t_d`. Zero-vs-nonzero, no tolerance,
  no bf16 resolution limit, and it works at 4k lanes as well as at 16.
- Sweeping `t_d = d + 256w` over `w` recovers `max attended lane` for every row, exactly; a second
  encoding `V[j][d] = (j == d ? 1 : 0)` returns the whole live bitmap for lanes 0..255 in one launch.
- Assert `max attended lane == lanes_of(a, q) - 1` for every row and head — the engine's own predicate,
  host-callable, on both sides.

## Commands run

All device work was correctness-only, one run at a time, each inside `tools/gpu_lock.sh acquire/release`.
`util` was 1 % on one pass and 90 % on another (the session serves its own tokens on this iGPU); no
kernel timing is taken or printed anywhere in this file — the only wall figures are whole-command and are
reported as "no timing taken" per the repo's GPU rule.

```sh
export LC_ALL=C
bash prototype/mtp_verify/build_causal_bound.sh            # -> tmp/mtp_verify/causal_bound
./tmp/mtp_verify/causal_bound                              # clean path, 3 forms x 4 encs, 1 s
./tmp/mtp_verify/causal_bound --must-catch                 # 33 injections, all RED, 1 s
./tmp/mtp_verify/causal_bound --form wide_llama --enc poison
bash prototype/mtp_verify/form_ladder.sh /tmp/cr_fix wide0 --attn-fattn-wide 0
bash prototype/mtp_verify/form_ladder.sh /tmp/cr_fix wide1 --attn-fattn-wide 1
bash prototype/mtp_verify/form_ladder.sh /tmp/cr_fix wrows1 --wave-rows 1
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py prep    --fixture p128 --pool wide640f --prefix 64 --work /tmp/cr_fix
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py perturb --fixture p128 --pool wide640f --positions 3,4,5,6,7,127 --work /tmp/cr_fix
./engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file /tmp/cr_fix/pert5.csv --steps 0 --prefill-logits all --out ...
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py sweep --fixture p128 --ref /tmp/cr_fix/ref --work /tmp/cr_fix --positions 3,4,5,6,7,127
```

## Result 0 — does the probe launch what the engine launches? (the parity table)

A probe can be exact about a kernel the shipping path does not run, so this is checked, not assumed. The
engine's own launch record comes from the new `ATTN_DUMP` line (`engine/kernels_attn.cpp`), which prints
every scalar `CoreArgs` field and marks every pointer field set/unset; the probe passes the same field
names in its own banner. Prompt `p128`, one body of 128 rows, the default form:

```
ATTN_DUMP seq=1 layer=3 core=attn_wide_llama_wave rows=128 lanes_last=128 split_eff=1 combine=0
  cap=130 pos=0 lvalid=1 neg=0 full=1 score_f32=0 fused=0 tile=8 tile_kernel=1 fattn_cover=0
  set_qg=1 set_kg=1 set_vg=1 set_qn=1 set_kn=1 set_qrope=1 set_krope=1 set_kcache=1 set_vcache=1
  set_aout=1 set_gout=1 set_kc_as=1 set_vc_as=1 set_part=0 set_stale=0 set_qrope_out=0
  set_krope_out=0 set_gate_out=0
```

| field | engine (`ATTN_DUMP`) | probe (`causal_bound`) | verdict |
|---|---|---|---|
| kernel launched | `attn_wide_llama_wave` (= `--attn-fattn-wide 2`, the default) | same function, from `engine/attn_wide_llama_kernels.h` | match |
| `rows`, `nrows` | 128 | 128 (one of the derived cases) | match |
| `lvalid` | 1 | 1 | match |
| `pos` | 0 | `lvalid - 1` = 0 | match |
| `lanes_last` | 128 | 128 | match |
| `split`, `split_eff` | 1 | 1 | match |
| `fused` | 0 (staged plan; `wide_plan_ok` refuses otherwise) | 0 | match |
| `full` | 1 | 1 | match |
| `score_f32` | 0 (the wide form has no F32-score spelling; also a refusal) | 0 | match |
| `tile` | 8 (`TILE`) | `TILE` | match |
| `tile_kernel` | 1 | 1 | match |
| `fattn_cover` | 0 | 0 | match |
| `neg` | `NEG_NONE` | `NEG_NONE` on the clean path, injections on the control path | match |
| `cap` | **130** | `lanes_last + 64` = 192 | **differs — and only in address arithmetic.** `cap` enters the cache index `(kv*cap + j)*DH` and nothing else: there is no `cap` in any mask, bound or loop end in either wide kernel — `grep -n cap engine/attn_wide_kernels.h engine/attn_wide_llama_kernels.h` returns index arithmetic only. The probe needs slack past `lanes_last` to make an over-the-bound lane observable at all, so matching 130 exactly would shrink the observable window. |
| `part`, `stale`, `qrope_out`, `krope_out`, `gate_out` | unset | unset | match, and **provably unread**: the probe ran the same kernels with these null and no launch faulted. |
| grid | `(2, 1, 4)` = (KV group, 1, `wide_rtiles_of(128)`=4) | same expression, `dim3(N_KV, 1, wide_rtiles_of(nrows))` | match |
| block | `WIDE_THREADS` = 256 | same constant | match |

So the refutation in Result 2 is about the kernel the engine runs, with the arguments the engine passes.
The one unmatched field is `cap`, which the round structure cannot read as a bound — and note that `cap`
tracks the request's token budget rather than the body's width: the same p128 prefill body prints `cap=130`
with `--steps 0` and `cap=137` with `--steps 8`, so a reader comparing two logs should expect that. As a
second canary, the same dump under `--steps 8` prints 48 `core=attn_core` lines (the one-row decode bodies)
beside the 6 `core=attn_wide_llama_wave` prefill bodies — the flag names the core of every body, so a wave
body mistaken for a decode body, or the reverse, is visible on the line. `ATTN_DUMP` is also the reason the
probe carries its mirror-image canary: every `CASE` line now prints
`core=attn_wide_llama_wave` / `attn_wide_wave` / `attn_slim_wave`, the function pointer the launch actually
went to, so an arm that is not the arm it is named for cannot read as a null.

## Result 1 — the engine datum reproduces, and it is exactly CROSSROW's

`p128` (128 tokens, one body: `ATTN_WIDE ... rows_per_block=128 grid=(2,1,4) wide_rows=128/128
lanes_last=128 split=1`), one token changed at a time, whole logits file compared byte for byte, rows
BELOW the perturbed position are the only possible cross-row edges:

| perturbed position | rows < p that moved | size |
|---|---|---|
| 3 | row 2 | 1−cos 4.19e-05, meanrel 1.02e-02, argmax SAME |
| 4 | none | — |
| 5 | row 4 | 1−cos 2.61e-05, meanrel 7.04e-03, argmax SAME |
| 6 | none | — |
| 7 | row 6 | 1−cos 4.47e-05, meanrel 9.68e-03, argmax SAME |
| 127 | row 126 | 1−cos 9.55e-05, meanrel 1.35e-02, argmax SAME |

So: row `q` depends on token `q+1` for even `q >= 2`; odd `q` and `q = 0` are clean; the reach is exactly
one row back. Equivalently in CROSSROW's words: the row whose bound `lvalid + q` is odd is the one that
reaches one lane past it.

## Result 1b — a void pass, recorded as a trap

The first form ladder was written as a shell function that took its extra engine flags in `$@` and
**never expanded them**. Seven arms (`--attn-fattn-wide 0|1`, `--attn-fattn 0`, `--wave-rows 1`,
`--gdn-loop 1`, `--gdn-chunk 64`, `--gdn-spell 1`) all reported "row 4 moved", and the honest-looking
conclusion drawn from them — "every form leaks, so the carrier is the GDN/conv, not attention" — was
false. It was one arm, the default configuration, priced seven times. The tell was `grep NUMERICS` on
the seven logs: every field identical. `form_ladder.sh` is the fix and it prints the canary fields on
every arm, plus the `ATTN_WIDE` line, before it prints a verdict.

## Result 2 — the wide kernels, launched directly, honour the bound exactly

`prototype/mtp_verify/causal_bound.cpp` launches `engine::kernels::attn_wide_wave` (knob 1),
`engine::kernels::attn_wide_llama_wave` (knob 2, the shipping default) and `attn_slim_wave` (the clean
control) with the launcher's own `CoreArgs` values (`neg=NEG_NONE`, `full=1`, `fused=0`,
`score_f32=0`, `tile_kernel=1`, `split=1`, `pos=lvalid-1`) over a derived geometry list (rows
1,2,4,31,32,33,63,64,65,128,129,130,512; `lvalid` 1,513,1025,1537,2049). The oracle is the probe's own
statement `causal_bound(lvalid,q) = lvalid + q`; it deliberately does not call `lanes_of`, which is part
of the suspect path. Four encodings, each a different question:

| encoding | question it answers | clean-path result on the wide forms |
|---|---|---|
| `threshold` (`V[j][d] = j >= base+d`) | did any lane `>= t` contribute? | topmost attended lane = `bound − 1` for **every** row, every head, every geometry up to 512 rows |
| `indicator` (`V[j][d] = j == d`) | the whole attended bitmap, exactly, plus the rowsum `l` = `1/aout[0]` | attended set = `[0, bound)` exactly; `l == bound` on every `(row,head)` with `bound <= 64` (512 readings on one case) |
| `poison` (one cache lane's K hot, Q dim 0 = 16) | **which cell read lane `j*`'s key?** | `j* >= bound` → read by no cell; `j* < bound` → read by cell `j*`, ratio `e` as predicted |
| `identity` (`Q[r][h][d] = 16 @ d=r`, `K[j][d] = 16 @ d=j`, `qg` hot on one row) | **whose query and whose gate produced this row?** | argmax dim = `r` for every row (peak = `1/l` = 1.000); gate `sigmoid` lands on the hot row only |

Must-catch: `--must-catch` runs 33 (form, encoding, injection) controls — `dropcur` (bound one short),
`readpast` (mask gone, bound `cap`), `widen1` (kernel told `B`, caller states `B−1`) — and all 33 come
back RED, so the four encodings demonstrably respond to a one-lane bound change. `widen1` in particular
is exactly the defect CROSSROW named, injected from the host, and every encoding that could see it does.

The gap this leaves is stated in the file: the first two encodings are content-independent by construction
(`Q = K = 0`), which is also what makes them unable to see a content-dependent operand bug — hence
`poison` and `identity`, which are the two readouts a Q-row or gate-row or K-lane mixup cannot hide from.

## Result 3 — the carrier is the wide form, canary-confirmed

`form_ladder.sh`, one 128-token prompt, `pert5` against `ref`, rows < 5:

| arm | canary from its own echo | rows < 5 that moved |
|---|---|---|
| `--attn-fattn-wide 0` | `attn_fattn_wide=fattn` | none |
| `--attn-fattn-wide 1` | `attn_fattn_wide=wide` | **row 4** |
| `--attn-fattn-wide 2` (default) | `attn_fattn_wide=wide_llama` | **row 4** |
| `--wave-rows 1` | `wave_rows=1`, wide never runs (`rows > 1` guard) | none |
| `--attn-fattn 0` | refused by the engine, with a reason | — |

## Result 4 — a dump-only engine flag, proven inert on both sides

`--dump-attn-rows` (`engine/decode_run.cpp` → `DecodeScratch::attn_dump_dir` → `AttnLayerArgs`, written by
`dump_attention_body()` in `engine/kernels_attn.cpp`). Default off; when on it writes, per attention body
(6 bodies in a 128-lane prefill: layers 3, 7, 11, 15, 19, 23):

| tile | what it is |
|---|---|
| `in_q_rope` | the rope'd Q tile the core reads (`a.q_rope`) |
| `in_gate` | the gate half the `finish_head` epilogue multiplies by |
| `in_k_rope` | the rope'd K of the body's own lanes |
| `in_kcache0/1`, `in_vcache0/1` | the staged KV cache the core reads, both KV groups, `lanes_last+1` lanes |
| `pre_gate` | the core's attention output before gating (`aout`) |
| `gated` | after `finish_head` (`gout`) |

named `attnNNN_L<layer>_<what>_r<rows>_lanes<lanes_last>.bin`, BF16, `[rows][8][256]` (caches:
`[lanes][2][256]`). `attn_dump_shape.py` prints the dtype inferred from the byte count, because reading a
BF16 tile as F32 — or back — yields a plausible `ndiff` and a NaN cos (AGENTS' dtype row).

Inertness is proved, not asserted (`prototype/mtp_verify/dump_ab.sh`, three binaries in one window):

- baseline binary (copied before the edit) vs new binary, flag **off** → **BYTE-IDENTICAL** on every output;
- new binary flag **off** vs new binary flag **on** → **BYTE-IDENTICAL** on every pre-existing output; the
  dump files are the only addition.

So what the dump shows is a property of the shipped computation, not of the instrument.

## Result 5 — the dependence set, measured at the attention output

Same perturbation as Result 1 (one token swapped for an out-of-pool id), but the comparison is `pre_gate`
from the dump, and the question is which **earlier** rows move (`dump_sweep.sh` + `attn_dump_diff.py`):

| perturbed position | rows ≤ 8 whose L3 `pre_gate` moved |
|---|---|
| 6 | none earlier (rows 6,7,8 moved legitimately) |
| 8 | none earlier |
| **9** | **row 8** — 1−cos 0.0, i.e. a few elements by ≤1 BF16 ULP |
| 10, 11, 12 | none earlier |
| 32 | none |

plus Result 1's `5 → row 4`, `7 → row 6`, `127 → row 126`. The dependence is therefore **exactly one lane
past the bound, and only when that bound is odd** — not "every lane past the bound", which is what a mask
widening would produce. Row 0 never moves.

For row 4 at L3, with lane 5 perturbed:

- `in_q_rope` row 4, `in_gate` row 4, `in_k_rope` row 4 — **identical**;
- `in_kcache0/1`, `in_vcache0/1` lanes 0..4 — row 4's entire in-bound set — **identical**; lane 5 differs,
  as it must;
- `pre_gate` row 4 — **6 of 2048 elements differ, each by ≤1 BF16 ULP, 1−cos = −2.4e−7**.

Three consequences, and they are the substance of this pass:

1. **The dependence is inside the wide core.** Its legitimate inputs for that row are byte-identical and
   its output is not; nothing downstream can do that (the epilogue is per-row), and `--attn-fattn-wide 0` —
   the slim core, same pipeline, same everything else — is byte-identical (Result 3).
2. **It is a numerical dependence, not an attended lane.** If lane 5 entered row 4's softmax, `1−cos` would
   be about `V[5]/(5·|out|)`, not `2e−7`. A handful of elements by one ULP is the signature of a
   **last-F32-bit change in the row's scale** (`m` or the rowsum): softmax is invariant to a uniform scale,
   so such a change survives only as rounding, and a few outputs cross a BF16 rounding boundary. That is why
   no gate has caught it, and why a logit-level reading (CROSSROW) overshoots into an attendance story.
3. **It is deterministic, so it is not the pipeline.** Two identical runs of the same binary and prompt give
   byte-identical dumps (`attn001_L3_pre_gate`, `attn001_L3_gated`, `attn006_L23_pre_gate` all IDENTICAL).
   The race hypothesis in an earlier draft of this file is dead on that evidence.

## Result 6 — what the synthetic probe now excludes, including the new fifth encoding

The probe runs **five** encodings, because the first four were structural and a numerical channel is
invisible to structure:

| encoding | what it can see |
|---|---|
| `indicator` | exactly which lanes got weight, per row and head (V as a one-hot ruler) |
| `threshold` | the topmost lane that got weight |
| `poison` | which output cell read a named lane's K operand |
| `identity` | which query row and which gate row produced a row's output |
| **`boundary`** | **whether a masked lane's score reaches the running max or the rowsum**: Q is zero on every in-bound dim and `40·16` from `dim == bound(r)` on, K is one-hot per lane, so the first lane past the bound scores exactly **+40** and everything else scores 0. If that score reached `m`, the live weights become `exp(0−40)`, F16 flushes them, and the row publishes zeros or a wrong count — unmistakable. |

Clean path: **345 cases, 0 failures, `rc=0`** across all three forms. Controls: **39 injections, all RED,
`rc=1`** (`dropcur`, `readpast`, `widen1`). So on synthetic content the wide kernels honour the bound on
V-attendance, K-operand mapping, query-row identity, gate-row identity, exact rowsum, **and** resistance to
a +40 masked score.

What `boundary` does *not* say matters: it refutes "a masked score reaches `m`" **at the magnitudes a probe
can build**, where live scores are 0. It cannot see a relative 1e−7 change of `l` — the rowsum check reads
`1/out[0] == bound` at BF16 resolution, a tolerance of 0.4 out of `bound`, not 1e−7. That is precisely the
magnitude Result 5 points at.

## Result 7 — the parity is a *pair* parity (the strongest remaining clue)

Both wide knobs pack the P operand in **f16x2 column pairs**: `p[L][2c]`, `p[L][2c+1]`, where a lane owns
column `base + 16L + 2c + hi` and gets its partner's column through `__shfl_xor_sync(..., e, 16)` — the
`hi_half ? other : e` / `hi_half ? e : other` pair store. A packed pair straddles the causal boundary exactly
when `bound` is **odd**: then `(bound−1, bound)` — one live cell and one masked cell — land in one pair.
When `bound` is even the boundary falls between pairs. With the engine's `lvalid = 1`, `bound = q + 1`, so
odd bound ⟺ even query row, **which is exactly Result 5's dependence set**; and the packing is shared by
knob 1 and knob 2, consistent with those two being byte-identical to each other and both leaking.

That is a clue, not a finding: every readout above confirms the masked half of the pair is exactly 0 in the
cases the probe can build. For it to move a live output by one ULP, the pair would have to change a *live*
value's rounding — an FMA-contraction or packing/rounding dependency that only real-magnitude operands
exercise. Proving or killing that needs the kernel's own intermediates on real content: §"What is left".

## What is left

The shipping wide core depends on one lane past each row's causal bound, deterministically, by last-F32-bit
amounts, with that row's own inputs byte-identical. None of five synthetic encodings — 345 cases, 39
must-catch controls — reproduces it, because each checks an output property at BF16 resolution and none can
see a 1e−7 change of the row scale. Naming the expression needs the kernel's intermediates on the engine's
real content, which is no longer a probe-side trick. Options, in cost order:

1. **A copied-kernel rig.** Take `attn_wide_llama_wave`'s masked round into a probe TU and replay the
   engine's real per-row `qrope`/`kcache`/`vcache` tiles — already dumpable byte-exactly by
   `--dump-attn-rows`, so this is a replay, not a re-implementation — printing `m`, `l`, and
   `p[L][2c..2c+1]` for the straddling pair in both arms. First instrument able to say *which* quantity
   moved; Result 5 only says something did.
2. **A bitwise CPU mirror of the masked round** on those tiles. Weaker, and AGENTS says why: a mirror
   reproduces a formula and agrees with itself, and the F32 accumulation order inside `v_wmma_*` is not
   reproducible on the CPU, so a mirror can chase association for days.
3. **Gate the symptom and record it.** Make the working instruments permanent (`crossrow_arm1.py sweep`
   per-row byte identity + `form_ladder.sh`, both already run) and record the leak as a known sub-ULP
   numerical dependence. **That is a decision, not a fix**, and it is the user's: CROSSROW currently claims a
   one-lane *attendance* leak, and this pass shows the datum is right but the mechanism is not that.

Also outstanding and cheap: **`prototype/mtp_verify/CROSSROW.md` still states the mask-widening predicate**
and should carry a pointer here. I did not edit another document's claims without being asked.

## Device confirm list

Everything below is correctness-only, blocking, one at a time, `tools/gpu_lock.sh` held by the caller
unless the script self-locks (none of these do). No arm takes a kernel timing.

1. **Probe, clean path** (~3 s wall; expected `rc=0`, `CAUSAL_BOUND PASS`,
   `RUN cases=23 forms=3 encs=5 parsed=345 failures=0`, and a `core=` field on every `CASE` line naming the
   kernel that case launched):
   `./tmp/mtp_verify/causal_bound`
2. **Probe, controls** (~3 s; expected `rc=1`, 39 `control ... -> RED, as it must be` lines and no
   `control ... GREEN` line):
   `./tmp/mtp_verify/causal_bound --must-catch`
3. **Reproduction** (~10 s, expected: odd positions list `row p-1`, even positions list nothing):
   `crossrow_arm1.py prep` + `perturb --positions 3,4,5,6,7,127` + one `decode_run --steps 0
   --prefill-logits all` per csv + `sweep` (commands in §Commands run).
4. **Form ladder with canary** (~12 s, expected exactly the Result 3 table):
   `bash prototype/mtp_verify/form_ladder.sh <workdir> <tag> [flags]` per arm; a `RUN FAILED` carrying the
   `--attn-fattn 0` refusal text is the expected fifth row, not a breakage.
5. **Dump inertness — run before trusting any dump** (~15 s; expected `BYTE-IDENTICAL` twice). Needs a
   pre-change `decode_run` at `$work/baseline_decode_run`:
   `bash prototype/mtp_verify/dump_ab.sh <workdir>`
6. **Row-4 datum at the attention output** (~6 s for 4 runs; expected: `pre_gate` row 4 moves by ≤1 ULP in a
   handful of elements while `in_q_rope`/`in_gate`/`in_k_rope` row 4 and cache lanes 0..4 are identical, and
   the `--attn-fattn-wide 0` pair is identical everywhere):
   `bash prototype/mtp_verify/dump_pert.sh <workdir> pert5`
7. **Dependence set** (~15 s for 7 runs; expected exactly the Result 5 table — only odd `p` moves row
   `p-1`, and its L3 1−cos is at or below 1e−7). Needs `$work/dump-refwide` from arm 6 and the `pert<p>.csv`
   files from arm 3's `perturb` step (positions 6, 8..12, 32):
   `bash prototype/mtp_verify/dump_sweep.sh <workdir> 6 8 9 10 11 12 32`
8. **Dump determinism** (~2 s; expected every compared tile `IDENTICAL` — this is what kills the race
   hypothesis): two `decode_run --steps 0 --prefill-logits all --dump-attn-rows` runs of one prompt, then
   `cmp` their `attn*pre_gate*.bin` files.
9. **Streaming-region width** (the one geometry the pinned fixtures never reach, ~1 min): rerun arm 3 at
   1024+ lanes (`wide640f` truncated to 1024) so rows above `MAXSC` are covered, then rerun arm 4 on it.
10. **Timing arms: not needed.** Nothing here moves a `ms` figure; the dump flag is default-off and its
    writes sit behind that flag only.

Build note: the engine needs a rebuild after the dump-flag edits (`bash engine/build.sh`, ~17 s). The
working tree compiles and `engine/build/decode_run` carries the flag. **Nothing is committed.**
