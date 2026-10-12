# `gdn_gated_norm` — §04.5's gated RMSNorm, k = 1 (kernel + gate **compiled; the first device run failed and the cause is not established**)

Why this op exists as its own directory: the audit found that A-01's cast order for this norm is
asserted at **no bit level anywhere** — the only comparison in the repo was `run_isolated.py`'s
`cos ≥ 0.9999` band, and a cast-order difference cannot move that number (§10.2 says so for every
rounding-policy question in this model). That is the same failure mode as the rope bug, so this op
gets its own bit-exact gate before the engine calls it.

**Device status: run once, failed, cause not established.** `hipcc` compiles clean (`build.sh`, and
`engine/kernels_gdn.cpp` compiles against the same header). Every bit-level number below comes from
the **CPU** mirror gate, which needs no device and is allowed under the GPU rule. The device half
(`run.py --gate`) has now been run by the operator and did not pass:

```
plan /tmp/.../plan.txt | jobs=144 | 16 blocks x 128 threads per job | neg=none
RESULT mode=once jobs=144 neg=none
  gated_out vs pins over 294912 elements: worst case ndiff 2048/2048, max ULP 49174, tightest band these results pass at 1.0e+00
  c000: ndiff 2048/2048 maxulp 49174 n_bad 1905
  c001: ndiff 2048/2048 maxulp 49174 n_bad 1952
  c002: ndiff 2048/2048 maxulp 49170 n_bad 1942
```

Nothing aborted: the clean launch and all six negative launches printed
`RESULT mode=once jobs=144`, and the output files are present at their full 4096 bytes. What three of
those files held, on inspection, was 1727 zeros, 320 NaN and exactly one plausible element out of
2048 — against pins of mean −0.007, std 0.109, absmax 2.391, no NaN anywhere. **The cause is not
established.** The leading hypothesis is that the buffers were never written and the file is whatever
the pool held (freshly touched `hipMalloc` pages read back as +0), which fits a completed launch and
a completed copy-back, and which nothing above proves. §7 records what was ruled out, what was
specifically *not* the cause despite being proposed as it, and what the gate now prints so the re-run
classifies itself instead of needing a probe.

One consequence worth stating while it is fresh: because that run's baseline was wrong in every case,
the negatives' `OK` rows proved nothing — an injection "moved" `gated_out` by moving it away from
garbage. `neg_probe.py` (§4) re-checks all six on the CPU against the pins, where they do hold up.

What the first hand-off attempt did expose, before it reached a kernel: it aborted in the loader.

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  WeightStore: no tensor named model.language_model.layers.0.linear_attn.norm.weight
```

The plan writer was naming the weight by its **checkpoint** key — the same key `mirror_probe.py`
reads the file with — while `engine::WeightStore` is keyed on the **§02.5 normalised** name
(`lm.layers.N.linear_attn.norm.weight`). `at()` refuses a name it does not hold rather than
normalising what it is given, which is the right refusal (a store that tolerated both spellings would
be answering for tensors it does not hold), so the producer was wrong and now converts through
`prototype/weight_names.py`, the one Python mirror of `engine/loader.cpp::normalize_name`. Plan files
now come out of exactly one function, `write_plan()`, which refuses to return a plan whose weight
names are *not* store keys — so no launch path can skip the check, and a naming mistake prints the
offending name on the CPU instead of dying with a C++ `std::out_of_range` in a subprocess. The store's
key set
(336 names, from `model.safetensors.index.json`, tied to `loader_test`'s 335 loaded + tied-head count)
is derivable without allocating anything, and the CPU command for the whole check is

```bash
.venv/bin/python prototype/gdn_gated_norm/run.py --preflight
# == pre-flight (no device): assemble the gate's plan and resolve its weight names ==
#   plan /tmp/.../plan.txt: 144 weight name(s), all present in the store's 336-name key set
# PREFLIGHT: PASS
```

Nothing about the kernel, its pins, or its bands changed: `--preflight` writes the same 144 rows × 6
fields the gate writes, with only the name column moved; the CPU mirror still passes (144 cases, ndiff
0) and the device half has since been run once and failed — §7.

## 1. The finding: the pins put the affine in the activation dtype, and the oracle lands there too

`mirror_probe.py` compares BF16 **bit patterns** of six readings of §04.5 against
`decode{N}.L{ii}.gdn.gated_out`, using `gdn.o_pre_norm` and `in_proj_z.out` as inputs, over all
18 GDN layers × 8 decode steps of `dec8` — **144 cases, 294 912 elements**. Command and output:

```bash
.venv/bin/python prototype/gdn_gated_norm/mirror_probe.py dec8
```
```
dec8 — 144 cases, 294912 BF16 elements compared against gdn.gated_out
  POLICY (kernel): bf16(n) · bf16(w) rounds, then ·silu(z) rounds  ndiff       0 / 294912  worst case    0  BIT-EXACT
  A-01 literal, weight in raw F32 (not a dtype speccheck holds)    ndiff  105683 / 294912  worst case  892  refused by the pins
  single cast: f32 norm · w · silu(z), one cast at the end         ndiff  124108 / 294912  worst case  999  refused by the pins
  A-01 literal, weight pre-rounded to BF16: *bf16(w) in F32, one cast at the end ndiff   74860 / 294912  worst case  585  refused by the pins
  gate rounded: silu(z) goes through a BF16 slot first             ndiff   79900 / 294912  worst case  608  refused by the pins
  speccheck.specimpl.rmsnorm_gated, called as speccheck calls it (BF16 weight) ndiff       0 / 294912  worst case    0  BIT-EXACT
  weight-dtype ladder: bf16 weight 0  |  bf16-rounded-then-F32 74860  |  raw F32 105683
```

Both bit-exact rows compute this one policy:

```
n   = BF16( o · rsqrt(mean(o², 128) + 1e-6) )      F32 statistics, cast back — A-01 gets this far
a   = BF16( n · BF16(w_norm) )                     the affine is a BF16 tensor op: the weight
                                                   joins in the activation dtype, and the product
                                                   rounds there too
ô   = BF16( a · silu(z) )                          silu in F32 over the BF16 z (HF's z·σ(z) order)
```

Three consequences, none of them mine to decide:

1. **A-01's clause is wrong as written.** "normalize in F32 → cast the normalized value back →
   then applies `w_norm` and `silu(z)` in F32" puts the affine in F32; the dumps put it in BF16
   with the weight cast first (§04.5's "multiply by `w_norm`" says nothing about dtype, and is
   only right under a BF16 weight). §04.5/A-01 correction in place, naming the old claim, is a
   hand-off item (see §6 and the §04.5 bullet under "Open questions" in `engine/results.md`) — not
   done in this commit.
2. **`speccheck.specimpl.rmsnorm_gated` is bit-faithful: 0 of 294 912 differ.** An earlier revision
   of this file reported "105 683 (35.8 %) differ, and 0 of them ever failed a gate"; that figure
   was **this probe's own bug**, not the model's. The probe handed the oracle `w.float()` — the raw
   F32 checkpoint tensor (`operands.py` returns `get_tensor` verbatim) — a dtype combination
   `Weights` never produces, because `specimpl.py:70-86` casts every model weight to the
   activation dtype unless `keep_f32_scalars=True` (`specimpl.py:83`). The three figures the audit
   reconciles are one A-01 formula at three weight dtypes, over identical `o`, `z` and pins:

   | weight as handed to the oracle | ndiff / 294 912 |
   |---|---|
   | BF16 — what speccheck passes | **0** |
   | BF16-rounded, then widened to F32 — A-01's letter | 74 860 |
   | raw F32 checkpoint tensor — what the old probe passed | 105 683 |

   `fc444d0` retracted the claim in `bench/gate-audit.md` (G2 amendment); the probe now *asserts*
   its oracle row at ndiff 0 and asserts the ladder is ordered, so the retraction is checked on
   every run instead of remembered. One caveat the ladder exposes: bit-exactness is a property of
   the **call**, not of the function — `speccheck/run.py --keep-f32-scalars` (the A-02 A/B knob)
   leaves `linear_attn.norm.weight` in F32, and then the oracle's `n = w * n` promotes to F32 and
   lands in the 74 860 class. The engine keeps that weight F32 as well (§01.6, §3 below), so it
   must round it to the activation dtype before the multiply — which is what the kernel does.
3. The engine's `decode_step` calls the kernel, not the oracle, so the engine is on the
   pin-faithful path from day one.

## 2. The same result on the other fixtures

`mirror_probe.py` with no argument sweeps four fixtures, exit 0, all CPU-only:

| fixture | cases | elements | kernel's reading | oracle's reading |
|---|---|---|---|---|
| `dec8` decode (18 layers × 8 steps) | 144 | 294 912 | **ndiff 0 — bit-exact** | **ndiff 0** |
| `prefix24dec8` decode | 144 | 294 912 | **ndiff 0** | **ndiff 0** |
| `smoke` decode (2 steps) | 36 | 73 728 | **ndiff 0** | **ndiff 0** |
| `p300` prefill, 277 positions × 18 layers | 79 776 rows | 10 211 328 | **ndiff 0** | **ndiff 0** |

The probe asserts that the bit-exact set is *exactly* {kernel, oracle-called-as-called} and that
the weight-dtype ladder stays ordered, so a second reading tying is a failure, not a footnote.

Two notes on this table, both honest-record corrections rather than findings:

* The `p300` row could not be re-derived until now: `variants()` reshaped its input to `(16, 128)`,
  which is one token, and a prefill pin holds 277 (`RuntimeError: shape '[16,128]' is invalid for
  input of size 567296`). §04.5 reduces per head over 128 lanes, so the row shape is `[-1, 128]`
  for any token count; `cases()` now asserts the pin is a whole number of `16×128` tokens.
* A prefill-position case for the `dec8` fixture itself (18 cases / 36 864 elements, measured by
  the earlier probe) is **not** in `FIXTURES` and is not re-derived here: `data/refs/dec8/` holds
  only `decode*`-prefixed pins, and the bare names that would carry prefill live in
  `prefix24dec8`/`smoke` — where §10.5's last-writer-wins trap makes bare names the wrong handle
  to reach for across phases anyway.

The `p300` row matters most: prefill positions exercise 16 heads × 277 positions per layer, so a
per-head-vs-whole-row reduction error would show there even if a decode case hid it.

## 3. What the kernel is

One block per value head (16), one thread per value dim (128): §04.5 normalizes **per head over 128
lanes**, not over the flattened 2048, and getting that wrong is silent (16 scales instead of one).
`o` is taken as F32 because that is what `prototype/gdn_decode`'s recurrence emits, and the kernel
rounds it to the activation dtype first — HF's fused delta-rule read casts `o` back to bf16 before
the norm (gdn_decode `results.md` finding 2), so the cast is part of the reference. On the gate's
own inputs (pins that are already BF16-valued) that cast is invisible, which is why there is no
negative for it; the engine depends on it.

`o` is taken as F32 because that is what `prototype/gdn_decode`'s recurrence emits, and the kernel
rounds it to the activation dtype first — HF's fused delta-rule read casts `o` back to bf16 before
the norm (gdn_decode `results.md` finding 2), so the cast is part of the reference. On the gate's
own inputs (pins that are already BF16-valued) that cast is invisible, which is why there is no
negative for it; the engine depends on it.

`z` is taken as BF16 (it is `in_proj_z`'s rows, already an activation) and `w_norm` as **F32**, and
that dtype is now read off the checkpoint rather than trusted:

```bash
.venv/bin/python tools/weights_report.py models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors linear_attn.norm.weight
# model.language_model.layers.0.linear_attn.norm.weight   F32   [128]   (×18 GDN layers)
# bytes by dtype: {'BF16': 1618.6, 'F32': 6.4}   — the header's own histogram is 452 BF16 + 36 F32
```

§01.6 names this one of "the 36 F32 tensors", the header agrees for all 18 GDN layers, and
`engine/loader.cpp` uploads file bytes verbatim and records the dtype the header gave (`slot.w.dtype
= parse_dtype(p.t.dtype_str, …)`) — `WeightStore` performs no dtype conversion anywhere, so the engine
needs **no dtype adapter** to call this op. `speccheck.specimpl.Weights` is the thing that holds model
weights in BF16 (its own cast, on the Python side); the store is not, and treating the two as one
claim is what §7 records.

One measured detail the CPU probe turned up while it was replaying the norm: `o` is small on these
pins. `mean(o²)` per value head has median **3.56e-7** and minimum 1.6e-13 across dec8's 144 cases,
against §00.2's `eps = 1e-6` — so eps is about **281 % of the median head's variance**, not a
rounding detail. That is why the `noeps` injection moves 286 378 of 294 912 elements (worst 1454 ULP)
while the other four epilogue injections move 70 000–125 000: on nearly every head the eps term
outvalues the variance it is added to. `neg_probe.py` prints these numbers.

## 4. Negatives (six, declared; replayed on the CPU, and now verified on a device too)

`norm_pre_gate` is the dumped `BF16(n)` stage, which is what makes four of the six locatable
rather than decorative: an epilogue bug must move `gated_out` and leave the stage alone.

| `--neg` | bug | must move | must NOT move | CPU replay: gated_out moved | norm_pre_gate moved |
|---|---|---|---|---|---|
| `w_f32` | the A-01 literal reading: weight not cast to the activation dtype | `gated_out` | `norm_pre_gate` | **69 560** | 0 |
| `affine_uncast` | `n·w` stays F32 instead of rounding | `gated_out` | `norm_pre_gate` | **74 864** | 0 |
| `single_cast` | nothing rounds until the end (the tempting kernel) | `gated_out` | `norm_pre_gate` | **124 105** | 0 |
| `silu_rounded` | `silu(z)` goes through a BF16 slot before multiplying | `gated_out` | `norm_pre_gate` | **79 896** | 0 |
| `bf16acc` | the squared-sum accumulator is BF16 end to end | both | — | **24 817** | **29 307** |
| `noeps` | eps dropped from the rsqrt | both | — | **286 378** | **287 803** |

The two right-hand columns come from `neg_probe.py`, which replays the kernel's arithmetic — including
the two-wavefront shuffle ladder a 128-thread block runs on this target — against the same 144 pins,
on the CPU, so the six injections are checkable without a device:

```bash
.venv/bin/python prototype/gdn_gated_norm/neg_probe.py dec8
# dec8 — 144 cases, 294912 elements compared per policy
#   none            gated_out 0/294912  norm_pre_gate 0/294912  ndiff vs pin 4  worst ULP 1  clean reference
#   w_f32           gated_out 69560     norm_pre_gate 0             ndiff 69562   worst 2   moves gated_out only, as declared
#   affine_uncast   gated_out 74864     norm_pre_gate 0             ndiff 74862   worst 1   moves gated_out only, as declared
#   single_cast     gated_out 124105    norm_pre_gate 0             ndiff 124105  worst 3   moves gated_out only, as declared
#   silu_rounded    gated_out 79896     norm_pre_gate 0             ndiff 79900   worst 1   moves gated_out only, as declared
#   bf16acc         gated_out 24817     norm_pre_gate 29307         ndiff 24821   worst 4   moves both stages, as declared
#   noeps           gated_out 286378    norm_pre_gate 287803        ndiff 286378  worst 1454 moves both stages, as declared
#   none [bf16_store world]   0 / 0   identical to the clean policy
#   none [f32_as_bf16 world]  290991 / 0   DIFFERS from the clean policy: that world is broken
#   w_f32  [bf16_store world] 0 / 0   identical to the clean policy: this injection cannot exist in that world
```

Four things that probe is the record for.

1. **The clean emulation is 4 / 294 912 off the pins, worst 1 ULP.** That is the reduction-order
   floor (`prototype/glue/floor_probe.py` measures 3–4 of 393 216 for the same reason), not a policy
   error: the emulator uses a correctly-rounded reciprocal square root where the device calls
   `rsqrtf`, and its summation tree is the device's rather than torch's.
2. **Each of the four epilogue injections lands on the reading `mirror_probe.py` already refuses** —
   `affine_uncast` 74 864 against that probe's 74 860, `silu_rounded` 79 896 against 79 900,
   `single_cast` 124 105 against 124 108. The small offsets are the same reduction-order floor, and
   the two probes agreeing to three significant figures is the cross-check.
3. **`w_f32` is only an injection because the store holds F32.** If the pointer were `hip_bfloat16`,
   `gn_bf(float(w[d]))` would be the identity and the row would move 0 elements — the gate would
   report `MISCATCHED gated_out` for a kernel that was correct. The probe prints that world explicitly
   (last line above) so the dependency is on the record rather than in a comment.
4. **No two injections overlap.** The four epilogue rows differ from each other by 5 000–50 000
   elements, and the two reduction rows are the only ones that touch `norm_pre_gate` — which is the
   `mlp_gemv` rule (a bug must show up where the bug lives) checked rather than asserted.

The wide-reduction mistake (normalising over 2048 instead of 128) cannot be injected in this block
shape — one block holds one head — so that falsification lives on the Python side: `cases()` feeds
`[16][128]` and a whole-row reduction cannot reproduce a per-head result (the `p300` row of §2 is
the evidence).

The device run of these six has been executed and its verdicts **do** count now (§9); the earlier run's
did not, because its clean baseline was garbage in all 144 cases, so "moved gated_out 294 912/294 912"
was a statement about the garbage, not about the injection (§7).

## 5. Band for the device gate

`gated_out` is compared at **max 1 BF16 ULP with ≤ 1 % of elements differing** — the band
`prototype/gdn_decode/check.py` already uses for exactly this op class (an F32 norm whose pins are
BF16-valued; reordering the mean-of-squares is legal and flips a few elements by 1 ULP —
`prototype/glue/floor_probe.py` measures 3–4 of 393 216, and `neg_probe.py`'s clean row lands at
exactly that floor). `run.py --gate` prints the run's actual ndiff / max ULP / tightest band through
`prototype/bf16_ulp.py` (whose selftest it runs first), so the band can be tightened the day it lands
at 0 rather than being loosened. **The band has now been exercised on a device and the run landed inside
it** — worst case ndiff 1/2048 at max ULP 1, zeros 0 / NaN 0 (§9). The one earlier run failed far
outside it for a reason unrelated to it (§7).

The verdict logic is unchanged as of this writing, but the *report* around it grew, because the first
run could not be read from its own log:

* each failing case now carries `| zeros N nan M` (`bf16_ulp.zero_nan`), and the summary prints a
  run-wide profile line — `ndiff 2048/2048, max ULP 49174` is the headline for both "the epilogue
  never saw real values" and "the rounding is wrong", since a +0 against a pin of −2.39 is ~49 000
  ULP; the counts tell them apart and the second sentence names which one to chase;
* each negative line carries its own zeros/NaN pair, because "moved" alone cannot say whether the
  injection moved a value or destroyed one;
* the driver prefills every output buffer with a `0x3F3F` sentinel before the launch and prints a
  `sentinel:` line (elements still holding it, first-16-bytes hash before and after) for the first
  job, the last job and every job still holding it, plus a `write-back:` total — so "nothing was
  written" and "a wrong thing was written" are different log lines, and an output-mapping bug that
  wrote only the last job is visible;
* the driver copies each resolved weight row back and range-checks it (`weights: … all finite,
  |value| <= …`), which is the dtype question answered by observation instead of by reading the spec.

## 6. Not done

* ~~**The device gate has never passed.**~~ **Closed by §9: `DEVICE GATE PASS`, max ULP 1, ndiff
  1/2048 worst case, six negatives localising, and a burst row (2.936 µs/launch, spread 1.4 %).**
* The §04.5/A-01 correction in `specification/` is a hand-off item, listed under "Open questions" in
  `engine/results.md`, not done in this commit.
* ~~The negative verdicts are CPU-verified only (§4).~~ Their device counterparts have now run against a
  verified baseline and all six localise (§9); the CPU replay in §4 is what predicted their sizes.

## 7. The first device run: what happened, and what was ruled out

> **Closed by §8.** The buffers were written, the weights were the right dtype, and the kernel was
> never at fault: `write_cases` manufactured the NaN itself. The account below is kept as it was
> written at the time — including the hypothesis §8 refutes — because the reasoning in it is what
> produced the diagnostics that made §8 visible.

Run by the operator, 2026-10-11, on a rebuilt binary, after the naming fix (`8eace84`/`eb88914`) let
the plan reach the store at all. It reached the kernels and failed every case:

```
RESULT mode=once jobs=144 neg=none
  gated_out vs pins over 294912 elements: worst case ndiff 2048/2048, max ULP 49174, tightest band these results pass at 1.0e+00
c000: ndiff 2048/2048 maxulp 49174 n_bad 1905
c001: ndiff 2048/2048 maxulp 49174 n_bad 1952
c002: ndiff 2048/2048 maxulp 49170 n_bad 1942
```

All six negative plans also printed their own `RESULT mode=once jobs=144`, and the output files exist
at their full 4096 bytes. Inspection of three of them found **1727 zeros, 320 NaN and exactly one
plausible element per 2048**, against pins with mean −0.007, std 0.109, absmax 2.391 and no NaN at
all. **Cause not established.** The leading hypothesis is that the buffers were never written and the
files carry whatever the allocator's pages held — consistent with a completed launch, a completed
copy-back and a zero exit code, and unproven.

What was ruled out, in the order the evidence arrived:

* **"`WeightStore` holds every model weight in BF16, so `const float* w` is reading BF16 bytes as
  F32."** It does not, for this tensor: the shard header stores all 18 `linear_attn.norm.weight` rows
  as F32 [128] (452 BF16 + 36 F32 total — §01.6's 36 exactly), `loader.cpp` records the header's
dtype and uploads the file bytes unchanged, and the driver aborts on anything that is not
  F32/512 B/[128] and prints the dtype it found. Had the store held BF16, the run would have died on
  plan line 1 with `… is BF16 [128] (256 B); §01.6 says …` and `run.py` would have raised
  `binary failed (rc 1)` — it could not have produced 144 compared cases.
* **The same claim, tested numerically rather than by reading code.** A BF16 store read through a BF16
  pointer is *identical* to what this kernel computes: rounding the F32 row to BF16 and reading it
  back is the same value, so no 49 000-ULP failure can come from that pair at all (`neg_probe.py`'s
  `bf16_store` row: 0/294 912 differ). What does produce that order of failure is the reverse
  confusion — a BF16 read of the F32 row, where lane d gets the d-th BF16 **half** of the F32 words:
  **290 991 of 294 912 elements wrong, worst ULP 48 182, c000 = 160 exact zeros and no NaN**. That is
  the shape of the reported symptom, which is the reason the dtype hypothesis was plausible and also
  the reason the ordered change (make the kernel read BF16) would have *manufactured* it rather than
  fixed it. The kernel keeps `const float* w`.
* **NaN arriving through the arithmetic.** The gate feeds no NaN anywhere: across dec8's 144 cases the
  `o`, `z` and pin tensors contain zero NaN bit patterns (`neg_probe.py` prints the counts), and
  `a·g` cannot synthesise one from finite operands. So the NaNs in those files did not come from the
  pins through this kernel. *(True of the tensors, and the reason this bullet was kept; wrong about the
  files — the gate manufactured 320 NaN per case on the way to disk. §8.)*
* **The throwaway probe's candidate-match numbers** (`POLICY 2004/2048`, `z-all-zero 1727/2048`,
  `affine only 0/2048`) are **not evidence** and are not built on: against an output that is mostly
  zeros, "matches the all-zero candidate for the 1727 elements that are zero" measures the zeros, not
  a mechanism. They are recorded here only so that nobody re-derives them from the same files.

Re-run, with the report lines above in place and the driver rebuilt (the driver changed, so a stale
binary would tell us nothing):

```bash
cd prototype/gdn_gated_norm && ./build.sh && cd ../.. && .venv/bin/python prototype/gdn_gated_norm/run.py --gate --time
```

What each new line will say, if the hypothesis is right: `sentinel: … left 2048/2048 … NOT WRITTEN` and
`write-back: 144/144 gated buffers never written` while `RESULT … unwritten=144`, with the output
profile reading `zeros 0, NaN 0` because the sentinel (not +0) is then what the files carry. If
instead the buffers *are* written and the comparison still fails, `sentinel:` reports `written`, the
zeros/NaN profile shows what the kernel really produced, and `weights:` has already said whether the
rows it read were affine weights at all.

## 8. Resolution: the gate's own fixture writer manufactured the NaN

§7's open cause is closed, and the kernel was never at fault. The second device run (rebuilt binary,
`--gate --time`, rc 1) came back with the two lines that made the input path the only suspect left:

```
weights: 144 [128] F32 rows read back from the store, all finite, |value| <= 1.194
write-back: 0/144 gated buffers never written, 0 partially written; 0 norm buffers never written
output profile over 294912 elements: zeros 228249, NaN 66557 (0.0 % of the output is neither)
```

Every output buffer really was written, and the weight rows the store handed over are finite O(1) F32
— so neither "the launch did not write" nor "the weight dtype is wrong" can be the story. The
remaining suspect was the other two inputs, and that is where the bug was:

```python
o.numpy().astype(np.float32).tofile(of)      # fine: o is float32, astype copies
z.to(torch.uint16).numpy().tofile(zf)        # BUG: this converts, it does not reinterpret
```

`mirror_probe.cases()` yields both `o` and `z` as `.float()`, so `.to(torch.uint16)` is a **numeric
truncation of the activation values**: 0.195 → the integer 0, and −1.3 wraps to 0xFFFF. The kernel
then read those integers back as BF16 payload, i.e. as +0 and NaN. Measured on case 0's `z`, which is
exactly what the on-disk `c000.z.bin` contains:

| writer | read back as BF16 |
|---|---|
| `z.to(torch.uint16)` (what shipped) | **1525 zeros, 320 NaN** — matches the file |
| `z.to(torch.bfloat16).view(torch.uint16)` (the fix) | 0 zeros, 0 NaN |

Every symptom in §7 follows from that one line, with no further mechanism:

* **gated_out all zeros and NaN.** `g = z·σ(z)` runs on the destroyed `z`, and `ô = a·g`, so the output
  is +0 wherever the truncated integer was 0 (|z| < 1, 77.4 % of lanes) and NaN wherever a negative z
  wrapped into the all-ones exponent (22.6 %). That is why the NaN share was **not** a multiple of 128:
  it is a per-lane property of z's signs, not a head-stride effect. The `o` side was untouched, which
  is also why `norm_pre_gate` — the one stage that never reads `z` — came out clean.
* **all six negatives "passed".** They were perturbing a tensor the writer had already destroyed; every
  cast-order difference rounds to the same 0 or NaN. Their verdicts from that run carry no information
  (as §7 warned), and the `ndiff 2048/2048, max ULP 49 174` headline was the fixture, not the kernel.
* **the "66 557 NaN = `w_f32` moved 66 557" identity.** Not a clue about NaN positions: the negative
  counter compared with `!=`, and NaN is not equal to itself, so every NaN element in the clean output
  was charged as "moved" in every negative. The five others landing within 57 of the same number was
  the same artifact. Fixed by comparing BF16 bit patterns (`f248aff`).
* **the 252 `PARTIALLY WRITTEN` norm lines next to `0 norm buffers never written`.** Both were true and
  neither said the right thing: the summary counted only *never*-written norm buffers and was silent
  about partials, while the per-job line counted elements still holding the sentinel — and the sentinel
  was 0x3F3F = 0.746, a perfectly ordinary value for `norm_pre_gate`, so one written lane collided
  with the prefill. The sentinel is now 0x0101 (a BF16 denormal, ~1.8e-38, unwritable from O(1)
  inputs), survivors print their `(h,d)`, and the summary counts partials for both buffers (`32ce26c`).
* **why `--mirror` passed the whole time.** The CPU mirror consumes the tensors from `mirror_probe`
  and never touches the files, so the broken artifact was invisible to it. Mirror green and device red
  were consistent from the start; reading the mirror's ndiff 0 as evidence about the files was the error.

Fixes, in landing order: `f248aff` bit-pattern `moved` counting; `3be849a` the bitcast writer plus
`verify_cases`, which reads **every** file back after `write_cases` and asserts length, "no NaN where
the pin has none", and element-by-element bit equality — so the gate can no longer compare a file it
mangled against a pin it did not mangle; `32ce26c` the sentinel fixes. `run.py --selftest` proves the
guard is not vacuous by writing one case both ways:

```
  bitcast write  -> accepted
  numeric-cast   -> rejected: t000.z: 320 NaN in the file where the pin has none — a numeric cast
                    (Tensor.to(uint16)) instead of a bitcast (.view(uint16)) is what does that
FIXTURE SELFTEST: PASS
```

The lesson generalises past this op: **a gate that manufactures its own inputs has to verify the
manufacturing**, because a mangled fixture and a mangled kernel produce the same red log and only one
of them is fixable by changing the kernel. `--quick` speccheck stayed green throughout (26/26 before
this work; nothing here touches `speccheck/`).

Re-run with the writer fixed and the driver rebuilt:

```bash
cd prototype/gdn_gated_norm && ./build.sh && cd ../.. && .venv/bin/python prototype/gdn_gated_norm/run.py --gate --time
```

What a pass looks like: `fixture verified: 144 cases … bit-identical to the pins` (the new writer
guard), `weights: … |value| <= ~1.2`, `write-back: gated 0/144 never written, 0 partial; norm 0/144
never written, 0 partial`, `output profile … zeros 0, NaN 0`, `max ULP 1` with ndiff a handful of
294 912 (§5's floor), then six negative lines inside a few percent of the `neg_probe.py` counts with
the four epilogue injections at `norm_pre_gate 0/294912`, the burst row, and `DEVICE GATE PASS`.

## 9. The device gate PASSES (2026-10-11, foreground zero-subagent window)

`.venv/bin/python prototype/gdn_gated_norm/run.py --gate --time` after rebuilding the driver, in the
same window that re-took `gdn_decode`'s timing row and settled E1/E2 for `engine/results.md`. Result,
every line read off the run:

| line | measured | what it was supposed to be |
|---|---|---|
| fixture guard | `fixture verified: 144 cases`, bit-identical to the pins | §8's writer guard, now doing its job |
| weights | 144 `[128]` F32 rows, all finite, `\|value\| <= ~1.2` | §7's refuted dtype theory, again refuted |
| write-back | gated 0/144 never written, norm 0/144 never written, 0 partial | §7's sentinel, now silent |
| `gated_out` vs pins | worst case **ndiff 1/2048**, **max ULP 1**, zeros 0, NaN 0 | §5's band: max 1 ULP, ≤1 % differing |
| six negatives | all six localise; the four epilogue ones at `norm_pre_gate 0/294912` | §4's `neg_probe.py` CPU counts |
| burst | **0.4227 ms per 144-job plan = 2.936 µs/launch, spread 1.4 %** | first timing row this op has ever had |
| verdict | `DEVICE GATE PASS` | — |

**Why max ULP 1 is the floor and not luck.** `gdn_gated_norm` reduces a 128-element sum of squares in a
two-ladder warp tree; the pins come from torch's reduction of the same 128 numbers. `neg_probe.py`'s
clean row — the CPU mirror, which reproduces the kernel's order — already landed at exactly this floor,
so a device run at ndiff 1/2048 with max ULP 1 is the op agreeing with its own reference and disagreeing
with torch by one ULP on one element per worst case. The band was set from that prediction (§5) and the
measurement landed inside it, which is the order this project wants: a floor sized before the run, then a
run that confirms it. Do not read max ULP 1 as "one bad element" — the rest of the tensor is bit-exact.

**What the timing row is worth.** 18 launches per token (one per GDN layer) × 2.936 µs = **≈0.053
ms/token**, which is 0.66 % of the ~8.0 ms measured token — and it is pure launch cost, because this op's
traffic is ~0.37 MB/token, i.e. ~1.6 µs of roofline. It sits in the same family as glue's 49 launches: the
lever is a packed launch over the 18 layers or fusion into `rec_kernel`'s epilogue, not a better
reduction. The 1.4 % burst spread is what makes 2.936 µs quotable at all — this is a launch-bound
number, the class `bench/performance-status.md` §6 found bimodal for glue, so it needed its own spread.
Ledger rows: `bench/performance-status.md` §1 and §6.

**Open after this run, all of them inherited rather than new.** The §04.5/A-01 wording correction in
`specification/` is still the spec owner's call (§6). And the fact that the first two attempts at this
gate failed 144/144 on a fixture the gate itself mangled (§7, §8) is why the writer guard, the sentinel,
the zeros/NaN profile and `--selftest` all stay: they are what made the failure legible enough to
attribute to the harness in three commits instead of a week.

## 10. P6: the same op over a wave of tokens (plan §4 P6)

`gdn_gated_norm(o, z, w, out, norm_out, neg, tokens)` now launches `16·tokens` blocks, one per
(token, value head), with the head index innermost — so `tokens == 1` is exactly the `<<<16, 128>>>`
launch §9 gated, same grid and same per-head 128-thread reduction, and a wave is that block tiled
down the token axis. The norm is per (token, head), so there is no reduction to reorder and no
legitimate way for the wide form to differ from the narrow one; the gate requires that rather than
assuming it. The plan line gained an optional 7th field (`… [NORMFILE] [TOKENS]`, default 1), and every
device buffer, memset and readback is sized from it instead of from the literal 2048 — a fixed 2048
readback would have made a 46-row job look written when only its first row was.

New injection: **`GN_ROW_ZERO` moves the input row only** — every token reads row 0's `o`/`z` while the
kernel still writes row `t`. That is the tile bug worth injecting (a wide kernel that forgot to advance
the read pointer), and unlike "index the output by row 0 too", it leaves the write-back sentinel clean
and gives a locality claim a value comparison can see. The earlier version indexed both by row 0; the
driver caught it as `L00 left 92160/94208 first@2048 (t1,h0,d0) PARTIALLY WRITTEN`, which is the
sentinel doing its job but not a claim about the arithmetic.

Gate: `run.py --tokens p64 --layers N` (T=46; run on device, foreground, `gpu_lock.sh util` 47–93 %
at the time; **correctness only, no timing taken**). Results for `--layers 2`:

| claim | result |
|---|---|
| tile `gated_out` vs the pinned `gdn.gated_out`, 2 × 94 208 elements | **1 element differs per tile, max 1 ULP, `n_bad 0`** — the same floor §9 measured on the decode fixture, i.e. the wide form did not move it |
| write-back sentinel | `unwritten 0, partial 0` for gated and norm on every job |
| `tokens == 1`, launched per row, vs the same rows of the tile | **92/92 bit-identical** (uint16 payload compare, so NaN pairs count as equal) |
| the six arithmetic negatives | each moves **every row** of both tiles and only its own stage (`norm_pre_gate 0` for the four epilogue ones) |
| `row_zero` | **45/46 rows move per tile, row 0 stays exact** — the injection appears exactly where it lives |

The round-trip guard of §8 applies to the tile writer too: this gate writes both files through
`write_f32_raw`/`write_bf16_raw` and reads them back and `SystemExit`s if a tile file is not what was
written, because on a 46-row tile a `.to(torch.uint16)` mistake is 94 % garbage rather than one case's
worth — the failure §7 was, at 46× the size.
