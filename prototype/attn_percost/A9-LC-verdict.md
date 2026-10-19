# A9-L-C-VERDICT — L-c (cap the split) is **DISPLACED**: the attention family cleared its line (−4.61 %) and the wave did not (−2.38 %). Not shipped; the knob stays inert (2026-10-19)

Bet: `A9-BET.md`. Implementation: `A9-LC-impl.md`. Prediction under test: A9 §4.2 of
`A9-LONGCTX-CENSUS.md` — **−4.2 … −4.7 % of the 32,768-lane wave**, from −5.9 … −6.6 % of attention.
Window: `prototype/attn_percost/run_ab_a9lc.sh`, logs in `tmp/attn_percost/a9lcab/`, main agent, blocking,
zero subagents, one `gpu_lock` hold, `util` 91 % at acquire / 55 % at release, 3 min 09 s wall for the whole
window (20:37:44 → 20:40:53 Z). `base` = the worktree at the pre-change sha `8b0a3b4`; `cap` = the committed
tree's binary with `--attn-split-cap 8`; `null` = that same binary with the flag unnamed.

## The two pre-registered lines, at the deciding width 32,768

| line | base | cap 8 | ratio | live line | verdict |
|---|---|---|---|---|---|
| **LINE 1** `attn` family ms | 5,961.62 | 5,686.51 | **0.9539 (−4.61 %)** | ≤ 0.9700 | **cleared** |
| **LINE 2** wave `wall_ms` (gran=stage) | 8,607.94 | 8,403.26 | **0.9762 (−2.38 %)** | ≤ 0.9700 | **missed** |

**VERDICT: DISPLACED** — attention got faster and the wave did not, by the bet's own definition. **Not shipped
as a wave lever.** The knob stays in the tree with its default inert, which is witnessed (below) rather than
assumed.

Supporting facts from the same window, all pre-registered requirements of the harness:

* **NULL arm** (changed binary, cap unnamed, vs base): family **−0.75 %**, wall **−0.40 %** → the window is
  **paired** (both inside ±2 %).
* **Rep quality**: `attn` family rep spread base **1.49 %**, cap **0.15 %** (tolerance 3.0 %).
* **The verdict is not a rep artifact.** Per-rep paired `wall_ms`: rep 1 8,691.42 → 8,439.64 (**−2.90 %**),
  rep 2 8,524.45 → 8,366.87 (**−1.85 %**). Both reps are individually under the −3.0 % line, so a third rep
  could not have flipped it. Stating this matters: a near-miss verdict read off two opposite-signed reps would be
  a coin toss, and this is not one.
* **Canaries both fired**: `base` prints no `SPLITCAP` line and its geometry is its own (`split=9` at 4,608); the
  capped arm prints `SPLITCAP split=8 cap=8 lanes=4608 natural=9 span=576` — `span > MAXSC = 512`, i.e. the cap
  genuinely bound and widened the slice instead of timing the same geometry twice.
* **Inertness of the shipping default, witnessed on device** (`tmp/a9lc/inert2.sh`, one hold, correctness only,
  **no timing taken**): the committed binary with no flag vs the pre-change binary at 4,608 lanes →
  `logits4607.bin` and `sampled.txt` **byte-identical**, with identical geometry lines
  (`form=fattn16 q=unstaged8 roll=single warps=3 vgpr=185 lds=20480 wf_per_cu=9 wave_rows=512 split=9`) and
  `SPLITCAP split=9 cap=0 lanes=4608 natural=9 span=512`. So the shipped path is unchanged by this commit.

## What the missing quarter of the win is, and what it is not

The attention family saved **275.1 ms**; the wave moved **204.7 ms**. The gap (~70 ms) is what the harness calls
the externality, and it printed **+70.43 ms**, dominated by `gdn_rec` at 758.52 → 837.98 ms. **That is not
established as a displacement.** Per-rep `gdn_rec` at 32,768 reads **786.67 / 730.37 ms (base)** and
**885.94 / 790.01 ms (cap)** — a rep-to-rep spread of 7.7 % and 12.1 %, whose intervals overlap. So this window
**cannot distinguish** "co-scheduling absorbed the saving" from "`gdn_rec` is noisy at 32k". The honest words are
**unattributed**, not "GDN got slower". (The harness says the externality decides nothing alone; here it
decides nothing at all, and the reason is measurable: a family whose own spread is ±10 % cannot adjudicate a
70 ms claim.)

## What this retires, and what it keeps

* **Retired:** A9 §4.2's claim that capping the split takes **−4.2 … −4.7 % off the 32k wave**. Measured:
  −2.38 %. The census' byte arithmetic was right about direction and roughly right about size, and its
  *rate-times-bytes* conversion overstates by ~1.35× even when the rate is the traced one (208 GB/s) rather than
  a modelled demand.
* **Kept — and this is why the verdict is DISPLACED, not FALSIFIED:** the mechanism is real. Removing 77 % of a
  105 GB/pass scratch round-trip moved the phase it lives in by −4.61 %, clearing the family line, so
  A9 §2.3's "attn_combine reads the partials back at 88 % of the DRAM roof" is **not** retired. What the
  `attn_combine` reading was wrong about is how much of a phase-wide win that buys.
* **Still standing, and it was never in question:** this lever is inert below 4,609 lanes, because
  `split_for(4,096) = 8` already. So even a SHIP would have changed nothing at the headline width — which is the
  same conclusion §50 reached from the other direction (short context is not where the remaining gap lives).
* **Not measured this window** (out of the funded scope): the cap at 16,384, where A9's model says the depth cost
  is 0.00 % and the traffic saving is smaller. If anything ever revisits L-c, that is the row to price first, and
  it needs the externality to be *controlled*, not just printed — i.e. paired per-rep `gdn_rec` with enough reps
  to resolve ±3 %, which at 32k means ~5+ reps and a second timer than a median of two.

## Consequence for the plan

L-c was the last targeted attention lever with a *measured* mechanism, and it bought a family win that does not
reach the wave. The remaining distance to llama.cpp at long context — **1.91× at 32,768** — is therefore not
reachable by anything in the targeted-lever space, which is exactly what A9 §5 predicted from the numbers
(best-case stack 1.494×) and which this verdict now confirms with the cheapest of those levers spent. What is
left is the pre-gated geometry rewrite (`SQ_WAIT*` stall census first, prototype must clear **6.5 TFLOP/s**
before it earns a second week), and that is the user's scope decision, recorded in `bench/performance-status.md`
§52 and not started.
