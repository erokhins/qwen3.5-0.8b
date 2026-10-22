# P7 gate zero — MTP speculative drafting on the GPU path: priced before anyone writes a kernel

**Status: gate zero PASSED — P7 is not killed, and it is the largest prize left in the queue.**
No timing was taken by the author of this document. Every `ms` figure is QUOTED from
`engine/generation-limits.md` / the queue's operating point and is labelled `quoted`; the device commands
this hands to the main agent are in §6 and none of them has been run. Gate zero cost so far: one CPU run
of each script below, zero device seconds — the same shape as P3's death (`prototype/vocab_prune/RESULTS.md`),
except this one came out alive.

**Question.** `specification/09-mtp-draft-layer.md` §09.5 records MTP as bit-lossless but **slower on torch
CPU** (speccheck 23.3 tok/s plain vs 18.3 with mtp1, accept 0.582) and says the honest conclusion is "MTP
does not help *in Python*, while the roofline says ~1.15× at k=1 if the draft is written as a kernel". A
Python number is not a verdict about a GPU path, and this repo's rule is that an idea is not closed until
measured. This gate prices it before funding a kernel.

## 1. The arithmetic

With k=1: draft every step, verify in the same step, emit `1 + a` tokens for a cost of `1 + c` steps.

```
speedup = (1 + a_eff) / ((1 + delta) * (1 + c))        wins iff a_eff > (1 + delta)(1 + c) - 1
```

* `c` — one draft proposal as a fraction of one target step.
* `a` — accept rate of the draft against the same loop that verifies it (§09.4: greedy accept iff the
  draft equals the target's own argmax at that position).
* `delta` — what **verifying two rows per step** costs over verifying one. Unmeasured, so it is not assumed
  away; §4 prints the largest `delta` each route survives.

Quoted operating point (**4,032 lanes**, `engine/generation-limits.md` §0–§1): step **10.9357 ms** `quoted`
reading **1532.3 MB** `quoted`, of which `lm_head` is **509.552 MiB in 2.1685 ms** `quoted` — 246 GB/s
demanded against a ~237–240 GB/s plateau, i.e. **104 % of the wall**. Standing bar for an engine change:
**+0.30 ms/step = 2.7 %** (`progress/2026-10-21/generation-idea-queue-2026-10-21.md`).

### 1.1 What a draft step actually reads — witnessed, not arithmetic

`mtp_census.py` reads the shard; `draft_golden.py` runs the real `speccheck.mtp` drafter through a weights
dict that records every key it indexes, so the traffic claim is a census of tensors touched:

| per draft step | bytes | measured how |
|---|---|---|
| `mtp.*` block, **15 tensors** | **40,905,728 B = 39.0107 MiB** | shard enumeration (`mtp_census.py`) |
| tied table `model.language_model.embed_tokens.weight` BF16 `[248320, 1024]` | **508,559,360 B = 485.0000 MiB** | same |
| **read ledger of one draft step: 16 distinct tensors** | **549,465,088 B = 524.011 MiB** | recorder inside a live `MTPDraft.draft` |

so **92.6 % of a draft step's weight traffic is the tied head re-read**, 7.4 % is the block everyone has
been calling "39 MiB, trivial". The must-not-touch witness passed: every tensor read is `mtp.*` or the tied
table — nothing from the 24 target layers.

That is the whole difficulty, and §09.4 already said so (`draft cost ≈ 0.55 GB/token`). It is also why the
torch-CPU result is what it is: the draft pays a second full tied GEMV on a machine whose CPU path is
75–117 GB/s.

### 1.2 Can a proposal be produced without a second full `lm_head` pass?

Three candidate answers, and the answer turned out to be (a):

* **(a) restricted head over a candidate set.** The freshest full logits row the target has *already*
  computed when it drafts is the row whose argmax was the token being drafted behind — call it the row in
  hand. Take its top-k as a candidate set `C_k`; compute the draft's argmax over only those `k` table rows
  (`k · 2048 B`: 2 MiB at k=1024). **Exactly safe in the direction that matters:** if the draft's own
  argmax `d* ∈ C_k` the restricted head returns `d*`; if not, it returns something else and the target
  rejects it. Restriction can only lose accepts, never buy a wrong one, because the accept test is the
  target's own argmax. Survival measured in §2 and §3.
* **(b) argmax fusion.** Rejected without measuring. The traffic is reading the table to *have* logits;
  fusing the argmax removes host work, not 509 MB. `engine/generation-limits.md` §1 already caps that class
  at ~1 % of `host_tail` (queue P2).
* **(c) a full second pass is required.** Priced in §4 too, and it clears the bar on its own.

**Why this is not P3 in a hat.** P3 tried to prune the *target's* head with an **exact** Cauchy–Schwarz
bound and measured **100 % survival** (`prototype/vocab_prune/RESULTS.md`). The difference is that a
candidate set is **empirical, not provable**: it needs no bound, only to contain the drafted token often
enough — and its output is *allowed to be wrong*, which a provable prune never is. P3's fixture discipline
is carried over deliberately: a candidate-set question needs a valid `(x, logits)` pair and bare dump names
are last-writer-wins (§10.5, cos 0.245), so both studies below *prove their pairing* before printing a rate.

## 2. Study A — target-side survival, on the pinned dumps (`topk_survival.py`)

`survival(k) = P(the target's token at position i+1 is inside top-k of the row that produced token i)`,
computed over every pinned `decode{N}.logits_last` row that has a confirmed pairing. Alignment is proved
per fixture before any rate is printed — `argmax(logits_last) == greedy[0]`,
`argmax(decode{s}.logits_last) == greedy[s+1]`, `decode{s}.input_ids == greedy[s]` — and a fixture whose
pairing fails is dropped, not guessed at. **396 confirmed pairs over 14 fixtures.**

That guard earned its keep on the first run. The `logits_storage = bf16` fixtures (`dec300f`, `dec64f`,
`mt32f`, `chunk64b`, `wide*f`) store the row as raw **BF16 bits in a uint16 array**, and uint16 order is
not float order for negatives, so reading them as numbers produced a confident survival table over
garbage. Rows are now decoded by the dtype the manifest *declares*, and the first honest read of the same
data is this:

| rank of the next token inside the previous row | value |
|---|---|
| median | 19 |
| p90 | 1 047 |
| p99 | 17 584 |

| k | survival(k) | restricted head bytes | 
|---|---|---|
| 16 | 47.98 % | 39.04 MiB |
| 64 | 66.67 % | 39.14 MiB |
| 256 | 81.57 % | 39.51 MiB |
| 1024 | 89.90 % | 41.01 MiB |
| 4096 | 95.45 % | 47.01 MiB |

The head size barely moves with k — k=4096 adds 8 MiB to a 39 MiB block — so **the binding quantity is
survival, not k's cost.** (Strict ranks, stored in `topk_survival.json` as `rank_strict`; the tie-inclusive
survival column differs by ≤ 0.6 points, since the pins are BF16 and boundary ties exist. The 14 contributing
fixtures and their pair counts are stored too, so the population behind 396 is enumerable.)

**Decision room** (does an `a` measured on one bit stream transfer to another?): over the same 396 rows the
target's top1−top2 logit gap is median **6.500**, p10 **0.875**, p1 0.000. A legal 1-ULP BF16 change at the
top of a row is ~3.9e-3 relative, and **1.26 %** of rows have a relative gap under one ULP. So the share of
decisions a legal rounding change can flip is on the order of 1 %, not 10 % — which is what makes it
legitimate to carry a torch-CPU accept rate into a GPU decision at all (it is still a carry, and §5 still
says to measure it).

## 3. Study B — the accept rate, and the draft-side survival (`draft_survival.py`)

Runs the shipped §09 drafter (`speccheck.mtp.MTPDraft` + `greedy_mtp`, `verify="steps"`, which G8a blesses
as bit-identical to plain greedy) and instruments it from outside by subclassing — the loop itself is never
re-implemented, so the measurement cannot drift from the form. Per round it records the drafted position,
the proposal, and that proposal's rank inside the row the target already has.

**Alignment, because this repo has been burned by exactly this.** The accept test is a *join on recorded
positions*: `accept ⟺ d* == pred_at[p-1]`, where `pred_at` is the target's own per-position argmax read off
`Forward.logits`, and every round additionally checks `argmax(row in hand) == y_next` and aborts the run on
any failure. The independent witness that the bookkeeping is right: the per-round join reproduces the loop's
own accept counter **exactly on all 20 prompts** (343/468 both ways).

20 corpus prompts × 40 steps, k=1, 468 rounds:

| | value |
|---|---|
| accept `a` | **0.7329** (343/468) |
| §09.5 recorded | 0.642 (86/134, 9 prompts × 24 tokens) |
| this rig at 6 corpus prompts × 24 | 0.6977 (60/86) |
| accept by depth | pos 0–7 **0.588**, 8–15 **0.656**, 16–31 **0.714**, 32–63 **0.808** |
| per-prompt accept spread | 0.414 … 0.905 |
| rank of the draft's proposal in the row in hand | median 32, p75 266, p90 1227, p99 21 008 |

Three honest readings:

1. **`a` does not decay with depth — it rises.** That is the opposite of the A-18 worry as it applies here:
   A-18 says a *continuation* cannot be reproduced past ~8 tokens, which says nothing about the accept rate
   of a drafter standing next to its own target. But it rises partly because these corpus lines run into
   repetitive prose, so treat 0.73 as prose-set-dependent, not as a constant.
2. **The gap to §09.5's 0.642 is prompt choice, not method**: per-prompt spread (0.414–0.905) is far larger
   than the 0.09 difference. Consequence, and it is a decision this gate refuses to make alone: **the prose
   set that funds P7 has to be pinned**, or the number can be moved at will. Proposal: the 49 lines of
   `speccheck/fixtures/corpus.txt`, ≥ 24 generated tokens each, stopping on id 248046 in the generation
   phase only.
3. **`a` here is torch-CPU bits verifying torch-CPU bits.** The engine's bits differ. §2's decision room
   says that moves ~1 % of decisions, not 20 %, and the free-run/`feed`–`gen` index trap is handled by the
   position join above — but the number that *finally* funds the engine path must be produced by the engine
   drafting and the engine verifying (§5).

## 4. What each route buys (`route_price.py`)

Block priced at 0.725 of the plateau (the ledger's measured GEMV efficiencies: `mlp` 70 %, `proj` 75 %) =
0.2217 ms, plus **an estimated 18 launches** × 3.10 µs (`generation-limits.md` §1's measured per-launch
cost, from the `cast` deletion) = 0.0558 ms; the full head adds the quoted 2.1685 ms. The 18 is an
op-count read off the block's shape list (3 norms, 4 projection GEMVs + `fc`, rope, attention, act, the
tied pass), *not* a measured launch census — D2 is where that gets replaced by a real count, and the whole
launch term is 0.5 % of a step, so it cannot change the verdict.

**Cross-check against the spec's own roofline.** §09.5 prices break-even at k=1 as
`tokens/forward > (1.55+0.55)/1.55 = 1.36`, i.e. `accept > 0.36`. That is this same formula with
`c = 0.55/1.55 = 0.355` and speedup 1.0. With the `c` measured here (0.224) break-even accept is 0.224 and
the 1.027× bar needs 0.257. So the two arithmetics agree given the same `c`, and the whole difference is
whether the draft head is priced at 0.55 GB of assumption or at the ledger's measured 2.1685 ms pass —
D1 is exactly the measurement that settles it. The two causes §09.5 names for the CPU loss map onto the two
knobs above, one each: the 248k-row tied GEMV is what the restricted route attacks, and "an extra target
forward to roll state back on rejection" *is* the `refwd` model.

```
=== verify model: batched (one 2-row body) ===   a = 0.7329
route                      a_eff       c  speedup  ms/step     d_max a_eff_min a_raw_min
FULL tied head            0.7329  0.2237   1.416x   +4.551     0.379     0.257     0.257
RESTRICTED top-256        0.5403  0.0256   1.502x   +5.488     0.462     0.053     0.072
RESTRICTED top-1024       0.6389  0.0261   1.597x   +6.531     0.555     0.054     0.062
RESTRICTED top-4096       0.7000  0.0284   1.653x   +7.142     0.610     0.056     0.059

=== verify model: refwd (rejection re-feeds the accepted prefix — what speccheck does today) ===
FULL tied head            0.7329  0.2237   1.162x   +1.776       inf     0.633     0.633
RESTRICTED top-256        0.5403  0.0256   1.037x   +0.405       inf     0.533     0.723
RESTRICTED top-4096       0.7000  0.0284   1.280x   +3.060       inf     0.534     0.559
```

`a_eff_min` is the accept the route needs; `a_raw_min` divides it by `survival(k)`, i.e. the accept the
full drafter must achieve for the restricted route to be worth that `k`. Every line above the bar of
1.027× (the standing +0.30 ms). The three reads that matter:

* **The verify question does not kill it.** `delta` would have to exceed **0.379 of a whole extra step** to
  push the FULL head under the bar. The engine's `n`-row wave body (`engine/decode_step.h:submit_wave_body`)
  batch-extends exactly the families that carry the weight traffic and records them bit-identical to the
  per-token forms, so `delta ≪ 0.38` is the expectation — but it is §6/D3's measurement, not this document's
  assumption.
* **Even the wasteful rejection path survives.** `speccheck/mtp.py` re-forwards the accepted prefix on
  rejection, which is a large part of why torch CPU lost 21 %. Priced as its own model, the FULL head still
  gives **1.162×** (needing `a ≥ 0.633`, and we measure 0.733). So an engine that never learns to avoid the
  re-forward is still worth building — the *shape* of the win does not depend on that optimisation, only its
  size does. The restricted routes are much more fragile there (k=256 needs `a_raw ≥ 0.723`, k=128 falls
  below the bar), which is a second, independent reason to build the FULL head first.
* **The restricted head is worth ~0.24× more and is not free.** It needs a device top-k over the previous
  logits row (993 KB, already on the host — see `host_tail` 0.7163 ms `quoted`, on the critical path), i.e.
  it is coupled to P2's device-side argmax. Its price is `a_eff = a · survival(k)`, and at k ≤ 256 that loss
  is bigger than the traffic it saves: the first version should be the **FULL head**, with the restricted
  head as a follow-up once a device top-k exists anyway for P2.

## 5. What was built, and what gates it

| file | what it is |
|---|---|
| `mtp_census.py` | `mtp.*` tensor census from the shard (15 tensors, 39.0107 MiB) + per-route byte tables |
| `topk_survival.py` | Study A: target-side survival + decision room, alignment-guarded, dtype-aware pins |
| `draft_survival.py` | Study B: accept rate, depth profile, draft-side survival, per-route price; alignment guard aborts on any mismatch |
| `pin_pair_search.py` | which pinned file is the lm_head's input, decided by reproduction: prices every candidate `x` (raw pins, and each re-normed) against every pinned logits row, and exits non-zero with a stated reason if no pair reaches cos 0.9999 |
| `norm_recompute_probe.py` | the companion negative: `gamma*RMSnorm(pre_final_hidden)` does **not** reproduce the pinned `final_hidden` (cos 0.998653, median elementwise error 22.4 %), so the two bare names are not interchangeable |
| `draft_golden.py` | **the rig a future `mtp_layer` kernel gets signed against**: pinned input contract, byte-witnessed read ledger, bit-exact golden intermediates, derived controls |
| `block_pin_gate.py` | gates the block code path the drafter reuses: the `prefix=` branch against the ordinary branch, bit-identical, plus the HF-pin comparison as reported evidence |
| `route_price.py` | the arithmetic above, with `delta` and `a` sensitivities and both verify models |
| `DEVICE-HANDOVER.sh` | `--list` prints the five device items with priced values and kill lines; `--run D1` runs the only one possible today, under the GPU lock |

**There is no HF reference for this block** (transformers has no `qwen3_5` MTP class — §09 header), so
"gate it against the pinned dumps the way `speccheck/` gates everything" cannot mean a reference dump. It
means three things instead, and all three now run:

1. **Input contract, from pins.** `h` is taken from `final_hidden.npy`, and `E @ h` must reproduce the
   paired pinned logits row: `draft_golden.py`'s witness line prints **cos 1.000000000** against
   `decode7.logits_last` on `dec8`. `pin_pair_search.py` is the independent witness that `final_hidden` —
   not `pre_final_hidden` — is the vector the lm_head saw, by pricing every candidate `x` against every
   pinned logits row with a plain F32 matmul: best pair `final_hidden` × `decode7.logits_last` at
   **0.999988556**, runner-up the re-normed pre-norm pin at **0.999340475** — close enough to look right,
   far enough to be the wrong vector. The script exits non-zero with a stated reason if no pair reaches
   cos 0.9999, and the same fixture's bare `logits_last` pairs with **nothing** (best 0.246): §10.5's
   last-writer-wins trap, reproduced inside the fixture this gate depends on.
   Two findings fell out of doing this properly. (a) Recomputing `gamma * RMSnorm(pre_final_hidden)` does
   **not** reproduce the pinned `final_hidden` (cos 0.9987, median elementwise difference 22 %), so the norm
   recomputation across two bare pin names is not a valid witness — the §10.5 last-writer-wins trap again,
   one file over from where it is documented (`norm_recompute_probe.py` prints cos 0.998653, median elementwise
   error 22.4 %). (b) A first attempt at a survival study that read the uint16
   bit-form rows as numbers produced a plausible table over garbage (cos 0.25-class divergence in a different
   guise); the pins are now decoded by declared dtype only.
2. **Golden intermediates, bit-exact.** `golden/dec8.step{0,1}.{u,x_d,draft_logits}.npy` (BF16 as uint16
   bits — torch refuses `bf16.numpy()`, and `.to(uint16)` casts where `.view(uint16)` bitcasts) plus
   `golden/dec8.json` carrying the case spec and the read census. `--check` re-derives and compares
   bit-for-bit and re-verifies the census. This is what an engine kernel is admitted against. Like
   `data/refs/`, these files are **gitignored and regenerable** (`.gitignore:10` is `*.npy`), so `--write`
   is the re-make command, `--check` needs a `--write` before it in a fresh clone, and re-pinning is a
   decision rather than a fix.
3. **Controls derived from geometry** (AGENTS.md: an underived must-catch rots into "everything"):
   * MUST-CATCH `h_source=pre_norm` and `fc_order=he` — the two knobs §09.5 settled by accept rate — both move `x_d` ✔
   * MUST-CATCH moving **only the last step's** position +1 ✔, and MUST-NOT-CATCH shifting **every** position
     +1 (rope is relative-distance-only, so a uniform shift must move nothing) ✔
   * MUST-NOT-CATCH moving only the last step's position, judged on **step 0**: with a **one-entry MTP cache the
     attention row is a singleton and softmax of a singleton is 1.0 whatever the scores are**, so no score
     change can move it ✔ — the same geometry fact AGENTS.md records for a score-only injection on a one-lane
     attention row, one layer over. **This is why the gated case is two draft steps long**: a one-step golden
     cannot see rope, the MTP cache, or the position at all.
   * MUST-NOT-CATCH zeroing all 319 tensors outside the read set → both steps' `x_d` bit-identical ✔ (the read
     ledger is therefore complete)
   * MUST-NOT-CATCH perturbing step 2's input hidden → step 1 unchanged, step 2 moved ✔ (the chain feeds forward)

   `block_pin_gate.py` also came out red first and the red was in *my* comparator, not in the path: comparing
   `q_proj.out` against the raw layer input gave cos 0.999 / meanrel 0.8 (wrong operand — `q_proj`'s input is
   the normed row), and comparing the KV against the pin without the pin's `[1, heads, lanes, dim]` layout gave
   cos 0.38 (a layout bug, not a kernel). Both fixed; the gate now passes with `q_proj.out` and the post-append
   `attn.v` **bit-exact** against the pins, and the tensors that carry the HF cascade — `attn.q_rope`,
   `attn.k_rope`, `attn.attn_out_pre_gate`, `o_proj.out`, `mlp.*`, `s_out` — are printed as
   `[OVER-BAND, reported not gated]` with their numbers (dec8/L07: `o_proj.out` cos 0.999945760 /
   meanrel 9.95e-3, the widest of the group). Those belong to §10.2's **G4 attention-cascade band**
   (cos ≥ 0.999, meanrel ≤ 5 %) — this fixture is a cascade, a prefix run feeding the step, not G3i's
   isolated feed of the reference's own `s_in`, so G4 is the band that applies and it passes. A new gate
   prints numbers that live in another gate's band; it does not move that band.

One more trap worth recording: the first read ledger iterated the *recorder* (which **is** the weights dict)
instead of its `counts`, so it reported all 335 checkpoint tensors as "touched" and the must-not-touch check
would have passed vacuously. `draft_golden.py` iterates `rec.counts` and says so in a comment.

**One engine-behaviour note for whoever implements this, found by the restricted-head probe:** the draft can
propose the end-of-turn id **248046** (at `dec8` step 0 a top-16 restricted head proposes it where the full
head proposes 2064). A speculative engine must run §07.5's stop check on **accepted** tokens only — never on a
draft — or an unverified draft ends the turn. That is the same rule AGENTS.md already states for prompt feeding,
in a new place.

## 6. Handover — the device pass (main agent; nothing below was run by the author)

The runnable half is scripted: `bash prototype/mtp_gatezero/DEVICE-HANDOVER.sh --list` prints each item with
its priced value and kill line, and `--run D1` executes the only one that is possible today (lock protocol
included, self-locking drivers kept out of the hold). D2–D5 need an MTP kernel or a 2-row decode switch that
does not exist yet, and say so rather than pretending to a command.

Pre-registered lines, stated at **4,032 lanes** against the **0.30 ms/step** bar, and the arithmetic of §4
already decided KILL-1/KILL-2 in P7's favour:

* **KILL-1** (would have closed P7 with no device time): viable route is the full head **and** `a ≤ c ≈ 0.22`.
  Not triggered — `a = 0.7329`, and the accept needed to clear the bar at `delta = 0` is **0.257**.
* **KILL-2**: restricted route's survival so low that only the full head remains. Not triggered — survival
  95.5 % at k=4096 — but it *did* demote the restricted head to a follow-up (§4).
* **WIN line**: a route whose priced speedup ≥ 1.027× at 4,032 lanes with `a` measured by D4 below, and D1–D3
  confirming the priced `c` inside the margins stated there.

| # | what to measure | command / how | priced value | margin to pre-register |
|---|---|---|---|---|
| D1 | a **second** consecutive tied-table pass costs the same as the first (nothing in the step leaves 485 MiB resident: the fast tier is ~32 MiB, `devices/strix-halo-memory-bandwidth.md`) | burst `prototype/lm_head_gemv/run.py --iters 20 --rows 8` twice back-to-back inside one lock hold; the two bursts must agree | second pass = 2.1685 ms `quoted` | ±5 % on the head alone moves `c` 0.2237→0.2336 and the speedup 1.416→1.405× — not decisive either way |
| D2 | the draft block at engine shape: 39.0107 MiB of GEMVs + ~18 launches | no MTP kernel exists, so this is *not* measurable yet; price it from the ledger's achieved GEMV rates (70–75 %) = **0.22–0.25 ms** and re-check when the kernel exists | `c_block ≈ 0.28 ms` incl. launches | if the block came in at 2× the priced value (0.50 ms incl. launches), the FULL head still gives 1.39×; the conclusion is not block-cost-sensitive |
| D3 | **`delta`**, the 2-row verify's marginal cost over 1 row at 4,032 lanes | needs a decode path that submits a 2-row body over a populated KV; `decode_run` has no such switch today, so this is the funded work's **first** measurement, not a gate-zero one | assumed 0 | **delta_max = 0.379** for the FULL head (printed by `route_price.py`); above that it is under the bar |
| D4 | **`a` on the engine's own bits** — the number that finally funds P7 | engine drafts, engine verifies, on the pinned prose set (49 corpus lines, ≥ 24 tokens, stop on id 248046 in generation phase only). Free-run alignment trap in mind: state the index alignment used, always | 0.7329 carried from torch CPU | `a_needed = 0.257` (batched verify, FULL head); below that P7 is dead on the engine's bits |
| D5 | device top-k over the previous logits row (the restricted route's enabler; shares its mechanism with P2) | not built; price before choosing k | k=4096 head = 8 MiB extra, i.e. ~0.034 ms | if top-k extraction costs > 0.3 ms/step the restricted route is dead and the FULL head stands alone |

Correctness-only device checks the author *could* have run: none were needed — nothing here touches `engine/`,
and every gate above is CPU-only. **No device run was taken; no timing is reported by this document.**
`tools/gpu_lock.sh status` reads `free`, unchanged by this work.

## 7. Verdict and recommended shape

**Fund it, in this order, and do not start with the restricted head.**

1. An engine `mtp_layer` draft kernel (one §05 block + MLP + `fc`, 39.01 MiB) plus a **2-row verify body**, and
   measure D3 (`delta`) and D4 (`a` on engine bits) as its first acts. Priced headline: **1.42× at 4,032 lanes**
   (`a = 0.7329`, full head) — larger than every other item left in the queue put together, whose remaining
   estimates are 0.05–0.09 ms (P6), ~0.1 ms (P2) and ≤ 1.1 ms with a poor prior (P5).
2. Only then the restricted head (top-4096 priced at **1.65×**), which is worth doing **if and after** a
   device top-k exists for P2; building it before then buys 0.24× at the cost of a mechanism P2 also needs.
3. The gates in §5 are the admission path: `draft_golden.py --write` pins the reference intermediates (re-pin
   is a decision, not a fix), `--check` is what a kernel is judged against, `block_pin_gate.py` is the plumbing
   gate, and `draft_survival.py`'s position join is how the accept rate should be reported on device.

What would change this answer: an engine-bit accept below ~0.26; a `delta` above ~0.38; a prose set whose accept
rate is materially below the corpus set used here (which is why §3 asks for the prose set to be pinned rather
than leaving 0.73 as the number of record); or a device top-k that costs more than the head it saves.
