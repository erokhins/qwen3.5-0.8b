# P7 build plan — MTP speculative drafting, in four sections, each preceded by the measurement that de-risks it

Prototype-only planning artifact. **No engine file was touched to write it, and it contains no
measurement of its own**: every number is either imported from `prototype/mtp_gatezero/route_price.py`
/ `prototype/mtp_delta/D3-RESULT.md` / `engine/generation-limits.md` §1 or is labelled `quoted`. The
one thing this directory adds to the arithmetic is the batched-head ratio `R`, which
`prototype/mtp_headbatch/` exists to measure and which has **not** been measured yet
(`HEADBATCH-HANDOVER.md` carries the bands and the runner).

Bar and headline, standing: decode is reported at **4,032 lanes**, the step is **10.9357 ms** there, and
the funding bar is **+0.30 ms/step = 1.027×**.

## 0. The shape being built, and the four terms that decide it

One P7 round, k=1: the drafter proposes one token `d` from the hidden state the target already
computed; the target verifies `[t, d]` in **one** body of two rows at positions `q, q+1`; `d` is accepted
iff row 0's argmax is `d`, and row 1's argmax is then a free bonus token.

```
speedup = (1 + a) / ((1 + delta) * (1 + c) + rho)      # route_price's form, plus the rollback term §3 adds

a      accepted tokens per draft round. 0.7329 measured on torch-CPU bits (draft_survival.json).
       THE load-bearing unknown: the engine's own bits must reproduce it (D4, §4).
delta  marginal cost of the 2-row verify over a 1-row step, as a fraction of a step. D3 measured the
       parts reachable from a rig: plan A 0.1505 (1.6454 ms), pessimistic 0.4729 = KILLED. The gap
       between the two is the batched head, i.e. `R`, i.e. this directory.
c      draft cost / step. 0.2237 on the FULL tied-head route (the draft needs its own pass over the
       same 485 MiB table); restricted-head routes at top-4..top-256 are `c ≈ 0.0147` (`mtp_census.json`).
rho    rollback per round / step. 0.12 ms = 0.0110 (§3). Not in anyone's headline yet; it is real.
```

`price_head.py --bands` now prints the accept rate the bar needs at each `delta`, because gate zero's
`a_needed = 0.257` is the **`delta = 0`** row (its own footnote says so: “the accept needed to clear the
bar at `delta = 0` is 0.257”, MTP-GATEZERO.md:297) and is not the number D4 should be judged against:

| `delta` | `a_needed` at 1.027× | against `a` = 0.7329 |
|---|---|---|
| 0 (gate zero's framing) | 0.2567 | clears |
| **plan A, as priced (1.6454 ms)** | **0.4458** | clears |
| plan A + rollback | 0.4568 | clears |
| pessimistic arm (5.1715 ms) | 0.8511 | **fails** — the same fact D3 reported as “KILLED”, from the accept side |

So P7's survival band on `a` is **[0.457, 0.733] measured, not [0.257, 0.733]**: the engine's bits may
lose up to ~0.28 of the torch-CPU accept rate before the route is under the bar. That is D4's real kill
line and it must be stated in D4's own report.

---

## 1. The batched 2-row verify body over populated KV

**The measurement that de-risks it, already banked.** Three of them, and the enumeration below is short
because of them:

1. `submit_wave_body` **exists and is shipping**. `engine/decode_step.h:411-434`: the same §07.1 order
   over `n` consecutive tokens, layer-major — per layer all `n` tokens' mixers run in position order,
   then ONE glue launch over the `[n, H]` tile, then ONE batched MLP, then ONE more glue launch. Glue
   (row-indexed), the MLP (`--tokens`) and the nine projections (`--tokens`) are batched; the GDN
   conv/gates/rec and the attention row are still per token (P6/P7/P8). The batched forms are recorded
   **bit-identical** to the per-token ones on device: 612,352 elements 0 differing in glue, 23,040 per
   layer in the MLP, 138,400 per fixture in the projections.
2. `submit_token_body` **is** `submit_wave_body` with `n = 1`, so the n=1 launch sequence is the ledger
   sequence and there is exactly one implementation of the order. A 2-row verify therefore does not
   introduce a second code path through the 24 layers.
3. The verify rows are **consecutive positions of one sequence** — speccheck's `greedy_mtp` consumes
   `[t, d1..dk]` at positions `q..q+k` (`speccheck/mtp.py`) — which is exactly the wave body's contract
   (“positions are `st.prefix_len .. st.prefix_len + n - 1`”). **Nothing about a *candidate* row requires
   a branching position.** That is the P8 finding, and it is what makes this section a seam audit rather
   than a new kernel.

**What actually blocks reuse, item by item.** A table of the rollback surface, with whether bytes move:

| surface | what the 2-row body does to it | on **reject**, what must happen | bytes moved |
|---|---|---|---|
| target KV lanes | row 0 appends lane `q`, row 1 appends lane `q+1` | nothing to the data: roll `prefix_len` back to `q+1`. §08.6 inv. 3 (“`K,V` appends never overwrite a position `< P`”) permits re-appending at `q+1` since `q+1 == P`, and inv. 4 (`prefix_len` **is** the KV length) makes the stale lane unreachable — the attention launcher's per-row lane count comes from position (`kernels_attn.cpp:13`: a wave's row *t* scores lanes rows `0..t-1` plus its own appended lane). It is overwritten by the next round before it could ever be read. | **0** |
| GDN state (18 GDN layers) | advanced by **both** rows; not position-addressed | restore the snapshot. ~18.0 MiB/request (§00.5) | **18.0 MiB each way** |
| conv window | advanced by both rows | restore the snapshot. ~0.63 MiB/request | **0.63 MiB each way** |
| `st.prefix_len` | `+= n` | set to the accepted length | 0 |
| `st.sampled_token`, `st.stopped` | **untouched by the body** (by design: “there is no sample until `finish_token`”) | the accept loop writes them — this is where the stop-token rule lives (§3) | 0 |
| MTP block's own KV | one lane per drafted step | length truncation, same argument as target KV (speccheck's `MTPDraft` restores `kv` by truncating) | 0 |
| draft carry (`x_d`, `y_next`) | one `[1024]` row per draft | recomputed next round | 0 |
| host `logits` buffer | one vocab row today (`sc.logits` is `[kVocab]` F32, `decode_step.h:325`) | needs two rows (§1a) | 0 (one extra scratch row) |

So the whole rollback problem is **one mixer-state snapshot of 18.66 MiB**, and the whole KV problem is
a length field. That is a much smaller surface than “speculative decoding needs transactional state”, and
§3 prices it.

**(1a) `finish_token` is one row deep, and that is the seam.** `decode_step.h:382` names this exact
waste: a wave that called `decode_step` per token “would pay T `lm_head_full`s, T 1 MB vocabulary D2Hs
and T syncs to throw T−1 of them away”. Today `finish_token(weights, st, sc, logits, norm_row)`
(`decode_step.cpp:1172`) takes a `norm_row` and so **can already price row 1 of the norm tile** —
`sc.norm` is `wave_rows` deep, because the final norm falls out of the last glue call over the whole
tile. What does not exist:

* a head kernel that reads *two* norm rows in one pass over the table → `lm_head_batched`, the thing
  `prototype/mtp_headbatch/` prices. The rig's cross-M bit-identity result says that batching rows does
  not move one bit of any row's logits (`HEADBATCH-HANDOVER.md` §3), which is the property §09.4 demands
  of a batched verify;
* `sc.logits` as a **2-row** tile, and **one** D2H over both rows (2 × 993,280 floats = 1.99 MB) with
  two host argmaxes. Both argmaxes are genuinely needed: row 0's decides the accept, row 1's is the bonus
  token. This is a critical-path cost, not a hidden one: `host_tail` (`quoted`, **0.7163 ms**) is today
  “one blocking D2H of the whole 993 KB logits row plus §03.4's argmax, sampling and bookkeeping”, and
  `engine/generation-limits.md` §1 flags it as **on the critical path, not hidden**, because greedy needs
  the token id before the next step can be submitted. Doubling the D2H+argmax portion could in the worst
  case approach that whole 0.7163 ms — which is exactly why it is measured in M1 and why no number for it
  is asserted here. One fused 1.99 MB D2H is cheaper than two 993 KB ones, and §07.4's own reading is that
  fusing argmax itself is worth ~1 %, so the D2H is the term to watch, not the loop.
* **and that is the only scratch the change needs.** Checked rather than assumed:
  `allocate_scratch(o.wave_rows)` runs at `kMaxWaveRows = 512` by default (`decode_run.cpp:90`,
  `decode_step.cpp:226`) and `decode_step.cpp:227-230` records why — “a decode step uses row 0 of each and
  issues exactly the launches it issued when these were one row wide; a prefill wave uses rows `0..n-1`”.
  So the whole `[wave_rows, …]` row family (`norm`, the `[n, 6144]` tiles, `block_out`, …) **already has 512
  rows on the decode path and needs no re-sizing**; only `sc.logits` is single-row (`decode_step.h:325`),
  so the addition is one extra 993,280-float scratch row (+3.97 MB), which the same comment settles as
  “the wave buys the batching with scratch rather than with a second code path”.

**(1b) What is *not* a blocker and should not be re-litigated:** the layer-major order (mixers in position
order is what a causal verify needs anyway), `kMaxWaveRows = 512` (`decode_step.h:49`) is 256× more room
than k needs, §08.6's invariants are re-checked once per wave rather than once per row, and the batched
families' bit-identity is measured, so a 2-row verify over a populated KV is bit-lossless **except** at
the head, which is the term this directory measures.

**Open question this section cannot close (not a decision I may make):** whether the engine's accept rule
should be `verify="steps"` (bit-lossless, speccheck's own verdict) or `verify="batch"` (not bit-lossless).
§09.4 says an engine that must be exactly lossless has to make its verify path reuse the decode kernels
bit for bit; the wave body does, per the citations above, and the rig says the batched head does too — so
the *cheap* prediction is that a wave-body verify is bit-lossless and the answer is “batch”. But that is a
prediction about a path that does not exist, and the only honest form is M1's acceptance criterion:
**the 2-row body's logits rows must be bit-identical to two sequential 1-row bodies.** If they are not,
the spec stops here and comes back as a decision.

## 2. The draft-block kernel

**The measurement that de-risks it, and the one that does not.** What is banked: the block's *numerics*
are already gated on CPU (`prototype/mtp_gatezero/draft_golden.py`, `block_pin_gate.py`) — so a kernel's
admission is bit-exactness against pinned intermediates, not a tolerance. What is **not** banked: **D2 is
a plan, not a measurement** — MTP-GATEZERO.md:306 prices the block at 0.22–0.25 ms “from the ledger's
achieved GEMV rates (70–75 %)” with “~18 launches”, and says in terms that the launch count “is *not* a
measured launch census”.

**D2's own lesson, applied: a rig that stages the block differently from the engine is measuring the rig.**
Concretely, that means three things for how this block must be built and priced:

1. the weight blobs are extracted from the shard **shape-checked** (the §07.4/`check_weights` trap:
   `linear_attn.conv1d.weight` is BF16 `[6144,1,4]`, rank 3, and a checker with no rank field cannot say
   so) and staged by the engine's `WeightStore`, not by a prototype's flat file;
2. the launch bill comes from the engine's own plan derivation (`planned_launch_counts()`'s vocabulary),
   so that “~18 launches” is replaced by a real number before anyone prices a per-launch overhead;
3. the block is timed by the **stage clock on the engine's stream**, in a wave body shape, not by a
   prototype's burst of a hand-written loop. `prototype/mtp_delta` is the cautionary example: its rig
   measured a *transfer* of the `proj` family onto the head, and the difference between that transfer and
   the truth is the ±2.17 ms this whole directory exists to remove.

**The block, and what is genuinely new.** `prototype/mtp_gatezero/mtp_census.json`: **15 tensors,
40,905,728 B = 39.0107 MiB.**

| tensor | shape | engine kernel that already serves this shape | new? |
|---|---|---|---|
| `mtp.layers.0.self_attn.q_proj` | `[4096,1024]` | the target's own `q_proj` shape (8 heads × 256 × 2, `attn_output_gate: true`) — `kernels_proj.cpp` / `prototype/proj_gemv --tokens` | no |
| `…k_proj`, `…v_proj` | `[512,1024]` | target `k_proj`/`v_proj` (2 kv heads × 256) | no |
| `…o_proj` | `[1024,2048]` | target `o_proj` | no |
| `…mlp.gate_proj`, `up_proj` | `[3584,1024]` | target MLP (`--tokens`) | no |
| `…mlp.down_proj` | `[1024,3584]` | target MLP | no |
| `input_layernorm`, `post_attention_layernorm`, `mtp.norm`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden` | `[1024]` (×5) | glue's RMSnorm | no |
| `q_norm`, `k_norm` | `[256]` (×2) | §05 attention block's head norms | no |
| `mtp.fc` | `[1024,2048]` | an `(out,in)` the target already has (`o_proj` is `[1024,2048]`), so the GEMV kernel is reused | **input staging is new** |

**Tensor-for-tensor the MTP block is one target attention layer.** Every projection, the MLP, the norms and
the attention form are shapes and kernels the engine already ships, so the reusable surface is nearly all
of it and the new part is small and named:

* **the `fc` input concat.** `u = (e' ⊕ h') · W_fcᵀ` with `e'` = RMSnorm of the embedding row of the
  *next* token and `h'` = RMSnorm of the hidden state, order **embedding ‖ hidden** (`speccheck/mtp.py`,
  `MTPDraft(fc_order="eh")`; §09 with A-12 falsified: `h` is the row **after** the final RMSnorm —
  post-norm cos 0.642 vs pre-norm 0.232). Two norms then a 2048-wide staged row then a GEMV. An order bug
  here is invisible to any max-ULP check on `u` (both halves are the same distribution) and blindingly
  obvious to bit-exactness against the golden — which is the admission test, so use it. Useful accident:
  `h'` being post-final-norm means the engine **already has that row** — it is what `submit_token_body`
  leaves in `sc.norm`. No new dump, no new pass.
* **its own 1-layer KV cache**, appended per drafted step, truncated on reject. Per §1 that is a length
  field, not a copy.
* **rope at the drafted positions.** Successive draft steps sit at consecutive positions
  (`speccheck/mtp.py` uses the target's absolute `p`), so within one k=2 round the two rows are at `p` and
  `p+1` — the “positions 0/1” case. **The admission case must be ≥ 2 drafted steps, and this is a
  derived-blindness point rather than a preference:** with a single drafted step the MTP attention row is a
  *singleton*, and softmax of a singleton is 1.0 whatever the scores arrive as, so a one-step-only test
  cannot observe an attention bug at all. (Same class as the score-only injection in the AGENTS traps
  table: the must-catch set has to be derived from the geometry, and here the geometry says k≥2.)
* **the second tied-head pass**: draft logits are `x_d · Eᵀ`, i.e. a second full 485 MiB pass per round.
  That is `c ≈ 0.2237` and it is the largest single term in the draft. The restricted-head routes
  (`mtp_census.json`: top-4…top-256, `c ≈ 0.0147`, draft 0.16 ms) are a **later step gated on D5**
  (device top-k > 0.3 ms/step kills them), not part of this build; the FULL head is what P7 ships first.
* `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` (the draft shares the target's table —
  which is why the head term is a table pass and not a new weight blob).

## 3. The accept loop, checkpoint/rollback, and where the rollback cost enters the ledger

**The measurement that de-risks it, already banked.** `prototype/mtp_gatezero/draft_survival.py` measured
the accept statistics with a **position join** rather than a token-index join: a draft at position *p* is
accepted iff `d* == pred_at[p-1]`, with a per-round guard `argmax(row in hand) == y_next` and an abort when
the guard fails. The independent witness that the join is right is that it reproduced the drafting loop's
own counter on 20/20 prompts (343/468 accepted). **Carry that join into the engine accept loop verbatim**:
it is alignment-free by construction, which is what immunises it against the index trap below.

**Accept rule (k=1), with the rollback options priced.** Feed `[t, d]` at `[q, q+1]` in one body:

* `accept ⟺ argmax(row0) == d`. Accepted: commit `d`, then the bonus `argmax(row1)`; accepted length
  `q+2`; next round drafts from the bonus token's hidden row.
* `reject`: commit only row 0's argmax (which **is** the correct next token — so a rejected round still
  yields one token, which is why its cost is not wasted), roll `prefix_len` to `q+1`, restore the mixer
  snapshot, drop the MTP KV lane (a length field), and continue.

| rollback option | what it costs | verdict |
|---|---|---|
| **(i) snapshot before the body, restore on reject** | snapshot 18.66 MiB every round (~0.079 ms) + restore on a reject (~0.16 ms), i.e. `0.079 + 0.16·(1−a)` = **0.1217 ms/round** at `a`=0.7329 — reproduces D3 §4's ~0.12 ms from its own two halves | **the route to build.** D3 excluded it as out of scope, not as rejected. |
| (ii) no snapshot: on reject, re-forward from the correct token | re-pays a whole target step per reject. `route_price.py` prices the re-forward route at **1.162×** vs **1.416×** batched | strictly worse; keep only as the fallback if (i)'s copies turn out to contend with something |
| (iii) snapshot *after* row 0 (split the body in half) | smaller snapshot, but it requires a **sync inside the body**, and the body's entire design property is that it waits for none of its launches | not free. Recorded so it is not chosen silently later. |

**Where `rho` enters the ledger, and the decision that is not mine.** `rho` is neither `delta` (the row
cost at fixed state handling — D3 explicitly kept it out and flagged it upward rather than folding it in)
nor `c` (the draft's own cost). Recommendation: **a new stage-clock family `rollback` beside `glue`/`copy`**
— it is 1–2 launch pairs per round, so it is directly measurable as its own family once it exists, and it
then enters the formula as the `+ rho` term the code above uses. At plan-A numbers it costs
**0.0096×** (1.2309 → 1.2213, i.e. +2.525 → +2.421 ms/step) and raises `a_needed` from 0.4458 to 0.4568 —
inside D3's rounding, but it must be carried and not discovered later. Choosing between “new family” and
“fold into `c`” changes a ledger row's identity, so it is a supervisor call; this document states the
numbers and takes nothing.

**D4 protocol, as pre-registered, with the alignment trap attached.** The accept rate must be measured on
the **pinned prose set: 49 lines of `speccheck/fixtures/corpus.txt`** (verified: 49 lines), **≥ 24
generated tokens per line**, stopping **only on id 248046 and only in the generation phase** — a prompt
fixture legitimately *contains* 248046, so §07.5's stop check must never be applied while feeding a prompt
(`engine/decode_run.cpp`, §07.5). Two more clauses that fall out of recorded traps:

* **the drafter may propose 248046 and it must not stop the round**: gate zero found the top-16 restricted
  head proposing 248046 at dec8 step 0, so the stop test runs on *accepted* tokens only.
* **state the index alignment in the output, always.** The recorded trap: §10.5 pins
  `argmax(decode{s}.logits_last) == greedy_continuation[s+1]`, so a `kind=decode` sample at step *s* pairs
  with `greedy[s+1]`; the chat REPL is the other index (`gen[i]` pairs with `greedy[i]`, because its first
  output is the sample already pending from the last fed token). Aligned on `[0:]`, a 128-token free run
  read 0/128 and a 64-step one read 1/64, both false. The position join is the fix; a token-index
  comparison is only allowed if it prints which alignment it used.
* **speculation must not change the output**, so the end-to-end gate is the P7 free run against the plain
  greedy run on the same prompts. At long depth token equality is *not* a gate (the bf16 chaos floor,
  0.8776 sdpa-vs-eager, A-18), so: decision-level aggregates plus pooled 1−cos at the long widths, exact
  token equality at the short pinned ones — the same split `prototype/attn_wide/a13_identity_metrics.py`
  settled on.

## 4. Order of measurement, cheapest kill first

| # | measurement | what it costs | kill line (pre-registered) |
|---|---|---|---|
| **M0** | **`R`**, the batched-head ratio. `prototype/mtp_headbatch/`, no engine change, one lock window, already written and admitted green | one device window (`run_locked.sh`, main agent only) | `R > 1.35` → re-price before any kernel. **And `R_KILL = 2.2279 > 2.0`, so no head number alone kills P7** — it sizes the prize. `R ≥ 1.8617` dissolves the batched-head requirement (two passes beat it by the 0.30 ms bar) |
| **M1** | **`delta` on the engine's own 2-row path.** A `decode_run` switch that submits `[t, d]` at `[q, q+1]` over a populated KV and reads two logits rows — two separate `lm_head_full`s first (pessimistic shape), the batched head after. No new kernel family; it also measures the second D2H + second argmax, which nothing has priced | one flag + one extra logits row (the row family is already 512 deep, §1) | **`delta_max = 0.379` of a step (4.143 ms)** — `route_price.py`'s own line. Plus the bit-losslessness criterion of §1: the 2 rows must be bit-identical to two sequential 1-row bodies, or the spec comes back as a decision |
| **M2** | **the draft block, priced as §2 says** (engine `WeightStore`, engine launch plan, engine stage clock), admitted bit-exact against `draft_golden.py`'s two-step intermediates | one new kernel family | gate zero's own note: block at 2× priced (0.50 ms incl. launches) does **not** kill (FULL head still 1.39×) — it re-prices `c`. The real risk in this row is the **launch census**, which is why D2 must be replaced by a derived count, not re-asserted |
| **M3** | **`a` on the engine's own bits** (D4, §3's protocol) | needs M2 to exist | **`a_needed = 0.4568`** at plan A + rollback — *not* gate zero's 0.257, which is the `delta = 0` row. This is the only measurement here that can kill P7 by itself: the torch-CPU number is 0.7329 and the gap between it and the engine's bits is a mechanism question (decision-room ~1 % of tokens, chaos floor 0.8776), not a rounding error |
| **M4** | **end-to-end free run**: accept loop + snapshot/restore + MTP block + batched verify, on the 49-line corpus at 4,032 lanes; prints the decode headline and the new `rollback` family | the whole feature | gate = same token stream as the plain greedy run (§3's split rule). **A correctness defect proved on the shipping path outranks all pending speed work** (branch-2 rule), so if speculation changes the output, everything above is reverted, not waived |

**Why this order.** M0 is free of engine risk and already built; M1 is one flag and can kill plan A's
*shape* before a single MTP kernel is written; M2 is the expensive unblocker for M3; M3 is the number that
actually decides the route (`a`, not `delta` — `delta` would have to more than double from plan A to fail,
while `a` has only a 0.28 margin against its kill line); M4 is the ledger row and the only one that may
touch `engine/performance-status.md`. Nothing here pre-empts a numerics decision: M1's bit-identity
criterion and M4's output-equality gate are both existing-gate tests, which is what the delegated
numeric-format rule allows.

**What this increment did not do:** no `engine/` change, no timing (no `ms` was taken or is quoted except
figures labelled `quoted`), no MTP kernel, no commit, no gate threshold touched.
