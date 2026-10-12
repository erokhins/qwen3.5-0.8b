# prototype/gdn_decode — the state-carrying decode step (S04.2–S04.4)

Four functions, each gated before the next was trusted
(`.venv/bin/python prototype/gdn_decode/check.py`), driven entirely by reference-sourced
data: prior state from `decode0.*` dumps, new token's pre-conv raw from `decode1.*`, and
expected outputs from `decode1.*`. Nothing synthesized.

| gate | expectation | result |
|---|---|---|
| F1 `gates` (S04.3) | vs torch on identical fp32 inputs | g 1.95e-11, beta 4.5e-9 **ok** |
| F1 `gates` | vs pinned dump (can only be bounded — HF's `a_raw`/`b_raw` come from a bf16 matmul) | g 2.6e-4, beta 1.9e-3 **ok** |
| F2 `conv` (S04.2) | `q/k/v_post_conv`, new conv window | **bit-exact** (max\|Δ\| 0) |
| F3 `rec` (S04.4) | vs fp32 mirror of §04.4 | o 4.5e-8, state 2.9e-8 **ok** |
| F3 `rec` | vs pinned HF dump, after bf16 rounding (the dump is bf16-valued) | **0/2048 differ** |
| F4 `all` 18 layers | vs spec mirror, worst over layers | o 1.7e-7, state 9.9e-8 **ok** |
| F4 `all` 18 layers | vs HF dump | **within 1 bf16 ULP on every element of all 18 layers** |

## Performance

| case | time | effective bandwidth |
|---|---|---|
| one layer (`rec` alone: 16 heads, 1 MiB state) | **0.0064 ms** | — |
| 18 layers, 18.0 MiB state (the real decode step) | **0.0646 ms** (worst 0.0650, **spread 0.5 %**) | 876 GB/s **cache-resident** = 370 % of the probe |
| same kernel, state tiled to 108 MiB (6× replicas) | 1.344 ms | **255 GB/s** = 108 % of the read probe |

**The first two rows are the re-take** — run 2026-10-11 in a foreground, zero-subagent window with the
burst timer below, one `util` record, and `check.py` in the same window: **F1–F4 still PASS**, `o`
within 1 BF16 ULP of the pinned dump on every element of all 18 layers, with **3 layers holding 1-ULP
ties**. They replace the **0.062 ms / 918 GB/s** row this file carried, and `bench/performance-status.md`
§1, §2 and §6 now quote 0.0646 ms. The re-take came in ~4 % *slower* than the figure it replaces, which
is the direction a `min()` over a partly-zero event series is expected to move.

**What those rows were before, kept on the record.** They came from the per-iteration
`hipEventElapsedTime` loop this directory has now removed: the same timer `d81ff13` disowned
(`hipEventElapsedTime` returns exactly 0.000 ms for sub-millisecond work on this iGPU, and the loop
kept `min(best, ms)` with no `ms > 0` guard), with no `util` record and no repeat spread — contrary to
`bench/performance-status.md` §5's own provenance rule, and `0.062` had been carried straight into §6's
token sum. The timing path is now the burst steady_clock idiom (one warm-up, `--trials` bursts of
`--iters` launches, best **and** worst printed), which is what produced the numbers above.

Two things the re-take was asked to settle while it was at it, and where they stand: whether
`--restore-each` (state copied back inside the timed burst) differs from the default burst by more than
the copy's own time — **not run in that window, still open** — and whether a number above the ~240 GB/s
package ceiling is admissible here at all (`check.py` prints `% of the 237 probe` and asserts nothing,
unlike `lm_head_gemv/run.py` and `proj_gemv/run.py`, which fail on it) — **still open too**: 876 GB/s is
above the package ceiling for the same reason 918 was, namely that 18 MiB of warm state is not coming
from the package. That is what "cache-resident" in the row asserts; it is now stated in the bandwidth
column instead of left for the reader to notice. And one thing this re-take **cannot** settle: the
"state tiled to 108 MiB" row is not reproducible from
the binary as it stands — the parser takes `mode|dir|out|iters|layers` (plus the two new timing
flags) and has no tiling knob — so 255 GB/s is quoted from a build that no longer exists. Either
re-add the tiling knob or retire that row from the table and from §1's ledger row for it.

### Re-take (device; operator only, zero subagents, `tools/gpu_lock.sh`)

```bash
prototype/gdn_decode/gdn_decode --mode all --iters 200 --trials 5   # warm: 0.0646 ms best / 0.0650 worst
prototype/gdn_decode/gdn_decode --mode rec --iters 200 --trials 5  # 1 layer: 0.0064 ms
prototype/gdn_decode/gdn_decode --mode all --iters 200 --trials 5 --restore-each   # still not run
.venv/bin/python prototype/gdn_decode/check.py --iters 200         # correctness + the timing line
```

So the kernel runs at the LPDDR roofline when the state is cold and ~4× that when it is
warm — which it is in a decode loop, because 18 MiB of state is touched every step and the
iGPU's cache (not reported by `hipGetDeviceProperties`, which says 2 MiB L2) holds it.
Either way the state traffic is **0.06–0.22 ms of the ~6.6 ms token budget**, so the
register-resident version of pass B (avoiding the third pass over `S`) is a real
optimization but not a needed one.

## Findings this kernel produced

1. **§04.2's conv formula had a stray term.** It read `silu(raw[t] + Σ_j C[:,j]⊙raw[t-(3-j)])`,
   which double-counts the current token; the sum alone is correct and reproduces
   `q/k/v_post_conv` **bit-exactly**. Fixed in the spec, with the measurement cited. This
   is the second real spec bug found by a per-op gate (the first was the §02 special-token
   list) — the gates are earning their keep.
2. **The dumps are bf16-valued, and that masqueraded as a 1.45e-3 arithmetic gap.** HF's
   recurrent/chunk delta-rule functions cast back to `initial_dtype` before returning, so
   `o_pre_norm` on disk is bf16 stored as fp32. Comparing an F32 kernel against it shows
   meanrel 1.45e-3 — which is almost exactly bf16's own quantization step, and I initially
   wrote it up as "HF's fused read accumulates differently". It was the dtype. Round to bf16
   and the kernel is **bit-exact except for 1-elem-in-2048 ties worth exactly 1 ULP**. Gate:
   `bf16_ulp_report` in `check.py` (≤1% of elements differ, none >1 ULP); diagnostic:
   `ulp_probe.py`. **Check the dtype of a dump before concluding anything from a residual the
   size of that dtype's epsilon.**
3. **The algebraically prettier form of `o` is numerically worse.** `o = γ(q̃ᵀS) + (q̃·k̃)Δv`
   (one pass over `S`, no re-read) measures 1.5e-3 against the reference because the two
   terms cancel, while accumulating from the stored `S` in the reference's order measures
   4.5e-8 — for free, since pass B re-reads `S` anyway. Recorded in §04.4. (Note the
   coincidence: the split form's error and the bf16-dtype artifact are both ~1.5e-3, so
   fixing one was necessary before the other made sense.)
4. **My own ">240 GB/s means skipped work" rule needed a clause.** Here the number was real
   and explained by cache residency; the way to tell is to tile the working set past any
   plausible cache and re-measure (row 3 of the table). Without that check it looked exactly
   like the `lm_head_gemv` grid bug.
5. **A shared-memory overrun that produced NaNs in one mode and silence in another**:
   `block_sum` read `red[0..31]` from an 8-entry array. 128 threads → 4 warps, so the bound
   must come from `blockDim.x/32`. Fixed, and it is the reason "no NaNs in the inputs" is
   now printed as part of the gate.
6. **`decode1.L00.s_in` is the *pre*-norm block input** — the projections take
   `rmsnorm(s_in)`, which `check.py` determines by trying both readings and keeping the one
   that reproduces the pinned `beta`, rather than assuming it.

## Shape of the kernel, for the next one

One block of 128 threads per value head; thread `j` owns column `j` of `S` end to end, so
every access is consecutive across the warp and the `r` reduction is thread-local. Pass A
reads `S` to form `r`; pass B re-reads `S`, writes `γS + k̃⊗Δv`, and accumulates `o` from
the stored value. `k̃`, `q̃` staged in shared (1 KiB each); two block reductions for the L2
norms. q/k/v arrive bf16-rounded (the reference casts before the recurrence) and everything
else is F32.

## P6: the wave form — `conv` and `gates` over a `[T][CH]` tile (plan §4 P6)

Until P6 the engine called `conv_kernel` and `gates_kernel` once per token, so this file had never
seen a tile. Both now take a `tokens` argument: `conv_kernel` launches `dim3(CH/256, tokens)` — grid.x
is still the channel strip the single-token launch used, so `tokens == 1` is the same grid and not a
grid padded to one — and `gates_kernel` launches `<<<tokens, NH>>>`. The per-element arithmetic did
not move: F32 taps in `j` order, one BF16 round, SiLU; the gates are still one thread per
(token, head).

What the width adds is the part that had no test before: **the conv window**. A tile beginning at
token `t > 0` has three columns of history that are not in the tile, and §08.2/A-04 say they are the
carried **pre-conv** `raw` values, not zeros. `conv_tap` states that rule once for both the body and
the window write, and `new_win` is `raw[T−3..T−1]` — read through the same tap rule, so a tile that
begins mid-request writes the same window a token-at-a-time walk would.

Gate: `check.py --tokens p64` (T=46) / `--tokens p300` (T=277), `--layers N` (default 2). Run on
device, foreground, `tools/gpu_lock.sh util` 47–93 % at the time; **correctness only, no timing taken**.

| stage (per layer) | result |
|---|---|
| one wave of T, window = zeros: `u` vs `q/k/v_post_conv` | `max\|d\| 0`, bitexact on all T rows, both fixtures |
| window written by that wave vs `raw[T−3..T−1]` | **bitexact** (contents, not a fingerprint) |
| two waves, interior boundary at `T−7` and `T−5` | wave A's window **bitexact**, wave B's `u` **bitexact** against the single-wave rows, final window **bitexact** |
| `zero_pad` injected into wave B | moves **rows {0,1,2} of wave B and no other row** — the carry is load-bearing and localized |
| `--tokens 1` on nine probed rows (incl. both sides of a boundary) | **bitexact** with the wide launch, row for row |
| `tap_stride` (tile indexed as HF's `[CH][T]`) | invisible at `tokens=1`; **46/46 rows** move at `tokens=46`, window untouched |
| `win_lag`, `win_from_u` | `u` **bitexact**, window moved — see finding 1 |
| gates, T rows: `g`,`beta` vs torch on the same bytes | meanrel 2.0e−09 / 4.5e−09 |
| gates, T rows: vs the pins | `g` 2.0e−09; `beta` 1.3e−03 — bounded, see finding 4 |
| `row_zero` injected into the gates | rows 1..T−1 move, **row 0 stays exact** |

1. **A window bug is invisible in `u`, so the gate compares window contents.** Both window
   injections — write the window one row late, or write it from the post-conv `u` instead of the
   pre-conv `raw` (A-04's exact falsifier) — leave every element of `u` bit-identical, because
   neither changes what the current wave convolves. Any gate that checks only `u` passes both. That is
   why this gate compares `new_win` against `raw[T−3..T−1]` as bytes, and why the two-wave stage
   compares the window a wave *received* as well as the one it wrote.
2. **Zero-window fixtures hide the whole carry question.** `p64`/`p300` start a request, so their
   incoming window is zeros and a kernel that ignored the window entirely would still reproduce every
   pinned column. The split-wave stage is what makes the carry testable; the same argument on the
   engine's side is `engine/gate.py` section (h)'s `--wave-rows 5` side.
3. **A tile-indexing bug is invisible at `tokens == 1`.** `GC_TAP_STRIDE` reads the same buffer as if
   it were HF's channel-major `[CH][T]`; at one token both indexings are the same address, so the bug
   is undetectable on a decode step and moves every row of a wave. The negative is required to show
   both halves, which is the locality rule from `prototype/mlp_gemv/results.md` applied to a width
   rather than to a stage.
4. **`beta` cannot be pinned bit-for-bit even when the kernel is unit-exact.** The reference dumped
   `beta` BF16-valued, so feeding the kernel the *pinned* `in_proj_b.out` and comparing bit-for-bit
   charges the dump's own rounding to the kernel (meanrel 1.3e−03, `max|d|` 0.00195). The gate says
   so and keeps two levels: against torch on the same input bytes (1e−6 band, the sharp one) and
   against the pin (bounded, 5e−3, the same split F1 above already made for the same reason).
