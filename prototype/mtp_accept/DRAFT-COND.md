# DRAFT-COND — the draft's guess: ranked differences from llama.cpp, three variants, two nulls

**Scope and its limit, stated first.** The cost side of MTP is closed: at 4,032 lanes a k=1 round costs
1.617 steps (tip `9609c8b`, after the round-form routing fix), so parity needs an accept rate
`a* = round/step - 1 = 0.617` and the pre-registered +10 % adopt line needs `a >= 1.10 x 1.617 - 1 = 0.779`.
(On the corpus's own joined 4k case `CORPUS-WALL.md` bills the round at 1.701 steps, so `a*` there is 0.701;
the two differ by fixture, and neither is quoted here as a millisecond figure.)
The remaining term is the draft's guess. **No millisecond figure appears in this file** — every number here
is a count of tokens or a fraction, and the wall re-take is the main agent's (`AGENTS.md`: GPU timings belong
to the main agent, and this pass took none).

Everything below obeys one invariant: **a legitimate variant may change only the PROPOSED token.** The
target's verify pass stays the same forward pass over the same tokens; the accept/reject arithmetic and the
bands are untouched. That is not a promise in prose, it is what `flagoff_bytecheck.sh` and the side-pass
control measure (§Controls).

Read beside `CORPUS-WALL.md`, which carries the wall verdicts and the width caveat that §3 of this file
depends on. Rates here are deterministic and width-labelled; millisecond figures are deliberately absent from
both files where this file is concerned.

---

## 1. The ranked difference list, from source

Theirs = `engines/llama.cpp` @ `797de63`, `common_speculative_impl_draft_mtp`
(`common/speculative.cpp:1431-1872`) driven by `tools/server/server-context.cpp`. Ours =
`engine/mtp_draft.{h,cpp}`, `engine/mtp_accept.{h,cpp}`, the accept-loop branch of
`engine/decode_run.cpp`, and `kernels::mtp_attn_row` (`engine/kernels_attn.cpp`).

Ranking is by expected effect on the drafted token. D1 and D2 are the two that can be tested without
touching anything else; D3-D7 turned out to be identical or provably null, and saying so is part of the
result.

### D1 — the draft cache has no prompt. Rank 1, and the only one with room to reach the adopt line.

llama.cpp's drafter runs a **catch-up pass over every target batch** before it drafts
(`speculative.cpp:1621-1660`: one `batch.add(t.id, t.pos[0], seq_id, false)` per target row with
`set_embd_state(idx, {h_row, 1, n_embd})`, then `llama_process(ctx_dft, …)`), and that hook fires on the
prompt path too — `server-context.cpp:4345-4352` calls `common_speculative_process` after each target
submission, and `impl::begin` warns when `ctx_dft`'s `pos_max < N-1` precisely because the hook is expected
on every prefill ubatch (`speculative.cpp:1560-1573`). So llama's draft KV carries **one lane per prompt
position**: its drafter attends over the context.

Ours starts empty. `engine/mtp_draft.h::State` begins at `nlanes = 0`, `allocate_state` fills nothing, and
the accept loop appends **one lane per round** — nothing at all for the prompt. So on a 4,142-token prompt
our draft block's whole attention layer has read, at most, a few dozen generation rows.

What token it changes: every one of them. The block's mixer is a dense attention layer over its own cache
(`mtp.layers.0.self_attn.*`, `engine/mtp_draft.h`'s submission list), so an empty cache means the guess is
made from `h` and `y_next` alone with a one-lane mixer.

### D2 — our lane positions have holes on every accept. Rank 2.

A round feeds two rows on an accept and appends one draft lane, so consecutive draft lanes sit **2 positions
apart where the block expects 1**, and the rows for the skipped positions never enter the cache at all.
llama.cpp has no such hole: its catch-up runs one draft row per *target* row, pairing token at position `q`
with the target's h at `q-1` (`speculative.cpp:1625-1637`), so its draft KV is dense over every position,
accepted or not.

What it changes: the rope distance between every pair of neighbouring cached lanes, i.e. the attention
weights of the block's only mixer, plus the content it could have read. Note this is *not* the same thing as
D3: shifting every lane by a constant is free (see D3's measured null), leaving holes out is not.

### D3 — which position the draft row carries. **MEASURED NULL.**

Ours: `ar.pos = draft_pos`, the position of the token being *proposed* — `specification/09-mtp-draft-layer.md`
§09.1 (`p`: position of the drafted token), and `speccheck/mtp.py:151` `m.draft(yd, hd, q + 1 + j)`.
llama's: `batch.add(id, dp.pos0 + i + 1, …)` (`speculative.cpp:1830`) — the position of the token the row
*feeds*, which is exactly one less, uniformly over the whole cache.

Because `mtp_attn_row_kernel` rotates the query and every cached key with the same `head_norm_rope`
(`engine/kernels_attn.cpp`), a score depends only on the **difference** of positions, so a uniform shift is
an exact null. That was an argument; it is now also a measurement: `--mtp-pos-shift ±1` produces a
**byte-identical emitted stream** and identical rates over a 174-round fixture (§Rate table). The convention
disagreement is therefore *not* where the accept gap lives, and the §09.1 reading stands.

### D4 — which hidden row conditions the draft after an accept vs a reject. Identical; no variant.

llama: `accept()` sets `pending_h = verify_h[min(n_accepted, n_rows-1)]` (`speculative.cpp:1858-1871`) and
the caller passes `accepted.size() - 1` — the number of *draft* tokens accepted
(`server-context.cpp:4590`) — so a k=1 reject (`n_accepted=0`) takes row 0's h and an accept takes row 1's.
Ours: `decode_run.cpp` uses `sc.norm + kHidden` (row 1) on an accept and `sc.norm` (row 0) on a reject. Same
rows, same rule. And both h's are **post-final-norm**: llama sets `res->t_h_nextn = cur` *after*
`build_norm(cur, model.output_norm, …)` (`src/models/qwen35.cpp:210-213`), which is §09.1's A-12 as
falsified-and-corrected (0.642 post vs 0.232 pre). Independent confirmation of the same reading.

### D5 — does the draft see the token the target emitted, or the one it proposed? Identical; no variant.

llama: `dp.id_last = slot.sampled`, the last **accepted** token (`server-context.cpp:4622`, `:3497-3498`).
Ours: `tok_next = emitted_second >= 0 ? emitted_second : emitted_first` — after an accept that is the bonus
(the target's own row-1 argmax), after a reject the verifier's token. The draft's own proposal is never
fed back to itself across rounds; only within a multi-token draft would it be, and k=1 does not.

### D6 — "is the draft's GDN/conv state the post-row-0 or the post-round state?" — the question does not apply, and it is worth recording why.

The MTP block has **no recurrent state**. Its tensors are `mtp.fc`, four norms, `mtp.layer0.self_attn.*`
(a dense attention layer with its own 1-layer KV) and `mtp.layer0.mlp.*` (`engine/mtp_draft.h::Weights`,
15 tensors). The mid-checkpoint exists for the *target's* GDN state, and the only way "post-row-0 vs
post-round" can reach the draft at all is through which of the body's two norm rows is handed over — which
is D4, and is identical. So this suspect collapses into D4 rather than becoming a variant.

### D7 — the draft's sampler. Identical; no variant.

llama builds a `top_k=10` chain and takes `cur_p->data[0].id` (`speculative.cpp:1700-1712`, `:1777`), with
`p_min = 0.0` and `n_min = 0` defaults (`common/common.h:330-333`) so nothing is dropped for low confidence;
with only a top-k filter and no DIST/temp sampler in that chain, `data[0]` is the argmax. Ours is
`argmax_greedy` over the draft's own logits row — the same function `finish_token` uses.

### D8 — the h a round hands over is row 0/row 1 of a **2-row body**, not of a 1-row step. Not a conditioning difference and not movable here.

Our accept-case `h` is row 1 of `submit_wave_body` at width 2 on the round's route (`engine/mtp_accept.h`),
so it is a legal bf16 form of the same quantity rather than the bits a plain step would have produced.
llama's is also a row of a multi-row verify batch. The residual shows up as near-tie token flips, which is
exactly what `m3_loop_rate.py`'s runner-up + sub-ULP admission adjudicates, and it cannot be removed without
changing what the target computes — which the invariant forbids.

---

## 2. What was implemented — three knobs, all default-off

| flag | default | what it does | cost it pays |
|---|---|---|---|
| `--mtp-dense 0\|1` | 0 | on an **accept**, also stage the draft lane for the position the round skipped: content = the accepted candidate token with the row-0 norm (h at `p`), position `p+1` in this engine's content→position map. This is exactly llama's lane at that position | 9 submissions per accept, no head pass (`engine/mtp_draft.h::draft_lane`) |
| `--mtp-prompt-lanes N` | 0 | before generation, seed the draft cache with the last N prompt positions' lanes (token at `p`, target norm row at `p-1`; the last prompt lane uses the wave's own `y0`, which makes the seeded run contiguous with the first generated lane) | 9 submissions per lane, once per run |
| `--mtp-pos-shift N` | 0 | add N to every rope position the draft block sees, uniformly over the cache. Exists to *measure* the convention question (D3), not to ship a change | 0 |

All three change the proposed token and nothing else. `draft_lane` writes only buffers the draft owns (the
same arena a round writes) and allocates nothing new; it is stages 1-9 of `draft_round` and stops, via a
**shared** `submit_block_prefix` so a lane and a round cannot drift into being different objects.
`--mtp-prompt-lanes` reads only norm rows the prefill wave already computed, and the index into
`sc.norm` is asserted, not assumed.

Three refusals, all reachable (each was tested to actually fire):

* `--mtp-dense` without `--mtp-accept` — the side pass never skips a position, so the flag could only be inert;
* `--mtp-dense`/`--mtp-prompt-lanes` with `--time-stages` — neither has an integer per-step `mtp` bill
  (same argument `--mtp-sched loop` already gives);
* `--mtp-dense` with `--mtp-meter` — no metered phase owns a lane append, so it would be billed to
  `overhead`, which `METER-PLAN.md` §9 calls a phase waiting to be named.

Two honest limits, both printed rather than hidden:

* **`--mtp-prompt-lanes` is clamped by `PrefillResult::last_wave_rows`.** Only the rows the *last* prefill
  wave left in `sc.norm` exist, so on a 4,142-token prompt (waves of 512) the ceiling is **46 lanes**, and
  `--mtp-prompt-lanes 128` silently means 46 unless the run says otherwise — so it does:
  `MTPPROMPT wanted=128 fitted=46 norm_rows=46 pos=… clamped=norm_rows_in_scratch`. **Every prompt-lane
  number below is therefore a ≤46-lane number, not a 128-lane one.**
* The **reject** side of D2 is still open: the lane a reject skips is (fed token at `p`, h at `p-1`) and
  `h_{p-1}` is not in hand after the round, so closing it needs a 2 KiB carry of the previous round's row-0
  norm. `--mtp-dense` closes the accept side only, and says so at the site.

Also fixed in passing, because the census line had to be able to talk about the cache: `MTPDRAFT` now prints
`lane_appends=`, `prompt_lanes=` and `pos_shift=`, and `lanes_max` — which the accept loop never updated, so
it printed `lanes_max=0 cap=512` next to a cache of 83 lanes — is now the high-water mark from both callers.

---

## 3. Rate table

Same-window arms, one process per invocation, `--gdn-loop 1 --no-stop` on every accept arm. `±` is the arm's
own `1.96*sqrt(a(1-a)/n)`.

**Llama-markdown fixture** — `head -c 10200 progress/2026-10-21/performance-status-2026-10-21.md`, 4,142
ids, `--steps 256` (the round budget ends at ~171-174 rounds on `context_exhausted`):

| arm | rounds | accepted | a | ± | lane_appends | prompt_lanes fitted | note |
|---|---|---|---|---|---|---|---|
| `loop` (shipping) | 174 | 82 | 0.4713 | 0.074 | 0 | 0 | reference arm |
| `dense` | 171 | 85 | **0.4971** | 0.075 | 85 | 0 | +0.0258 |
| `pl32` | 171 | 85 | **0.4971** | 0.075 | 32 | 32 | +0.0258 |
| `pl128` | 171 | 85 | **0.4971** | 0.075 | 46 | **46 (clamped)** | +0.0258 |
| `pos-shift −1` | 174 | 82 | 0.4713 | 0.074 | — | — | **byte-identical stream** |
| `pos-shift +1` | 174 | 82 | 0.4713 | 0.074 | — | — | **byte-identical stream** |

At `--steps 120` the same loop arm reads `86 rounds, 34 accepted, a=0.3953`; the denominator, not the
model, moved. This fixture is a machine-generated ledger document, which is exactly why it is the right one
to compare against llama.cpp's 0.750-at-pos1 on the same bytes and the wrong one to read as an engine rate.

**Prose fixture** — `dec8` manifest `prefill_ids` (17 ids), `--steps 32` → 19 rounds:
`loop` 13/19 = 0.6842, `dense` 13/19, `pl16` 13/19 — **no movement, and 19 rounds cannot see one**
(±0.209 per arm). Reported because a null with a stated denominator beats a suggestive number with a
smaller one.

**On the per-position ladder the brief asked for.** k=1 means the drafter proposes exactly one token per
round, so a per-draft-position ladder has exactly one rung: every accept fraction in this file **is** the
pos1 number, which is what makes it comparable with llama.cpp's 0.750-at-pos1 on the same bytes
(`prototype/mtp_llama/LLAMA-HOW.md`). What still varies is *where in a run* the accepts land, so the probe
prints accept-by-round-ordinal instead (`ladder()`, one bit per `ROUND` line) — that row is the closest
honest reading of "per position" at this k. It goes to stdout and is not written to a file, so `tee` the
probe if a run's profile needs to survive it.

**Corpus witness, the 12-line subset** (`LIMIT=12 MIN_ROUNDS=120`, i.e. the denominator the brief calls the
noise reference):

| binary / arm | rounds | accepted | a | admission |
|---|---|---|---|---|
| pre-change (`b697843`, scratch worktree build) | 223 | 155 | **0.6951** | PASS, 12/12 lines stream-identical |
| patched, defaults (= shipping loop) | 223 | 155 | **0.6951** | PASS, 12/12 lines stream-identical |
| patched, `--mtp-dense 1` | 219 | 158 | **0.7215** | PASS, 12/12 lines stream-identical |

`0.6951` reproduced exactly on both binaries is the strongest available statement that the patch is inert on
the shipping path.

**Do not read those three rates against the parity line, and here is the width error that would cause.**
`a* = 0.617` is the parity line for a round at **4,032 lanes**, and every corpus rate above is a
**near-empty-context** number: the corpus prompts are ~6 lanes each, so the same population stretched to the
reporting width reads `a = 0.2903` against `a* ≈ 0.70` (`CORPUS-WALL.md`, §"The corpus cannot answer the 4k
question"). A corpus rate can therefore say nothing about parity at the width the headline is set at — it can
only rank draft-conditioning variants, which is what this file uses them for. The one 4k comparison that exists
says the mechanism is real and the width is not yet paid for: dense moves the joined-case rate
**0.2903 → 0.3548** (`CORPUS-WALL.md`), a bigger absolute move than the +0.026 here, which is what you would
expect if what the drafter is missing is context.

**Is it real?** Paired read (`draft_cond_paired.py`, per line, same prompts, same greedy stream):
**5 lines up, 1 down, 6 flat**; round-weighted mean per-line delta **+0.0263**; sign-flip permutation
p = **0.157**. The pooled pair's binomial half-width is ±0.084. So: **consistently positive on three
independent populations (corpus +0.026, ledger +0.026, dec8 0.000), not yet sized, and below the 0.05 bar the
brief set.** Nothing cleared 0.05, so per that instruction both knobs stay default-off.

---

## 4. Controls — what separates a better guess from a different model

* `flagoff_bytecheck.sh` — same fixture through the pre-change binary and the patched one with **no MTP flag
  at all**, `diff -r` over the dump dirs: **34 files byte-identical** (`sampled.txt`, `emitted.txt`, every
  `logits*.bin`). A binary `cmp` proves nothing here (`engine/build.sh` is not reproducible on this box);
  this compares what the two binaries *compute*.
* Side-pass control (`draft_cond_probe.py`'s `CONTROL` line): `--mtp-draft 1 --mtp-sched every` prices
  proposals without letting them drive the run, so the token stream and the per-row logits are pinned by
  construction. With vs without `--mtp-prompt-lanes 32`: **target dumps IDENTICAL across 122 files; proposals
  DIFFER (7 of 120)**. Same input tokens ⇒ same verify logits ⇒ different guess — which is precisely the
  separation the invariant demands, and the "proposals DIFFER" half is the behavioural canary that the knob
  was not a no-op (identical dumps *and* identical proposals would be reported as `CONTROL INCONCLUSIVE`).
* Per-arm admission (`m3_loop_rate.check_line`, imported not reimplemented): every accept arm's emitted
  stream against the flag-off greedy stream. All arms passed: either identical or a runner-up flip below one
  bf16 ULP of that row's top logit — the 2-row-body-vs-1-row-step form gap `ACCEPT-PLAN` already carries.
* `engine/gate.py --fixture dec8` → **ENGINE GATE PASS** with all three knobs compiled in and off.

Device passes, as required: each ran blocking under its own `tools/gpu_lock.sh` TAG
(`worker draft-cond-probe`, `worker flagoff-bytecheck`, `worker draft-cond-final-gate`), one at a time,
correctness only, util recorded at acquire and release, **no timing taken**. Harness walls (their own
seconds, not engine timings): `engine/gate.py --fixture dec8` 8 s (util 92 → …) on the final binary and 11 s
(91 → 57 %) on the first one; `flagoff_bytecheck.sh` 2-3 s (92 → 38 %); rate probe 15-25 s per invocation
(util 89-99 → 71-100 %); corpus witness 30 s per arm (62 → 45 %).

**Every number above was re-taken on the final binary** after `--mtp-pos-shift` landed: the 12-line witness
reproduced `155/223` (defaults) and `158/219` (`--mtp-dense 1`) byte-for-byte in the counters, the gate
re-passed, and the probe's `loop` arm reproduced `174 rounds / 82 accepted / 0.4713` on the ledger fixture.
The only new code since the first takes was `pos_shift`, which is added to a rope position and defaults to 0,
and the byte-identical `pos-shift ±1` arms are the witness that it moves nothing when untouched.

---

## 5. The two experiments that would actually move it, in cost order

Nothing crossed the 0.05 bar, so the honest answer to "get us the tokens" is: the guess moved a little under
the mechanisms that were cheap to build, and the gap at the reporting width is not in them. Note the width
before spending any of this: at 4,032 lanes the same population is at `a = 0.2903` against `a* ≈ 0.70`
(`CORPUS-WALL.md`), so what is missing there is not 0.08 of accept rate but more than double it — and the
mechanism that helped **most** at that width was the one that gives the drafter context back. In cost order:

1. **Size the dense knob on the whole corpus — no code at all.** `EXTRA="--mtp-dense 1"
   prototype/mtp_accept/m3_loop_rate.sh` (all 49 lines, both arms, ~2 min device) halves the error bar to
   about ±0.04 and either kills +0.026 or makes it a number a funding line can quote. Do this first: it is
   the only step that can turn §3's "if the delta is real" into a verdict.
2. **Make prompt seeding reach the tail it is meant to reach (medium).** `--mtp-prompt-lanes` is currently
   clamped to `prompt_len % 512` — **46 lanes on a 4,142-token prompt** — because only the last prefill
   wave's norm rows survive in `sc.norm`. Staging the last K norm rows of each wave into a K-row buffer
   (one D2D per wave, K ≤ 512) would let the seed reach 256-512 lanes. This is the **only remaining D-class
   difference with room to matter at 4k**: llama's drafter attends over the whole prompt and ours still sees
   ≤46 positions of it, the 32-vs-46-lane results were already indistinguishable (which says the effect, if
   any, saturates above ~32 lanes, so the current clamped number is close to meaningless), and the one 4k
   datum says context is where the drafter's headroom is — dense, the other context knob, moved 0.2903→0.3548
   at 4,032 lanes against +0.026 at natural width.
3. (Small, and cheaper than 2 but expected smaller) **Close the reject side of D2**: carry the previous
   round's row-0 norm row (a 2 KiB D2D per round) so a reject can stage the lane it skips too. Completes the
   dense property; expected ≤ +0.01 on its own.

One thing deliberately **not** tried, because it is not a draft-side change: changing which row the round
hands over (D4/D8), or making the round's body cheaper so a lower `a` still pays. The first touches what the
target computes; the second is the cost side, which is closed.

## 6. Files

* `engine/mtp_draft.{h,cpp}` — `draft_lane` + `submit_block_prefix` (stages 1-9, shared with `draft_round`),
  `kLaneStageSubs`, `State::lane_appends`, `State::pos_shift`.
* `engine/prefill.{h,cpp}` — `PrefillResult::last_wave_rows`: the row count a caller needs to read
  `sc.norm` legitimately instead of re-deriving the wave split.
* `engine/decode_run.cpp` — the three flags, their refusals, the prompt seeding (with its index assertion
  and `MTPPROMPT` line), the dense lane in the accept branch, `DENSE-SKIP`, the census fields, and the
  `lanes_max` fix.
* `prototype/mtp_accept/draft_cond_probe.{py,sh}` — the rate probe (arms, admission, control pair, noise
  half-widths, no clock).
* `prototype/mtp_accept/flagoff_bytecheck.sh` — pre-change vs patched flag-off byte control.
* `prototype/mtp_accept/draft_cond_paired.py` — the paired per-line read of two witnesses.
* `prototype/mtp_accept/m3_loop_rate.{py,sh}` — `--extra` / `EXTRA=`, appended to the **accept arm only**
  (never to the flag-off pair run, which is the greedy reference).
* `prototype/mtp_accept/mtp_wall_ab.sh` — `ACCEPT_EXTRA`, so the main agent can price a dense round in the
  same harness whose canaries already work.

## 7. The commands the main agent runs next

Everything self-locks except `engine/gate.py`, which needs the lock held around it. The wall pass is
main-agent only: blocking, zero subagents, one lock hold for the whole paired run.

```bash
# --- the 4k wall re-take, shipping pair and the dense arm priced in the SAME window -------------------
# ACCEPT_EXTRA is the priced difference, so it belongs on the accept arm only (unlike --gdn-loop 1).
IDS=tmp/glue_copy_fold/ab/ids4032.csv OUT=tmp/mtp_speed_4k_ship \
    prototype/mtp_accept/mtp_wall_ab.sh                      # base vs loop: unchanged from M3-RESULT
IDS=tmp/glue_copy_fold/ab/ids4032.csv OUT=tmp/mtp_speed_4k_dense \
    ACCEPT_EXTRA="--mtp-dense 1" prototype/mtp_accept/mtp_wall_ab.sh
# short prose leg, same two arms:
IDS=/tmp/dec8_ids.csv OUT=tmp/mtp_speed_short ACCEPT_EXTRA="--mtp-dense 1" \
    prototype/mtp_accept/mtp_wall_ab.sh

# --- the corpus witness, full population, both arms ----------------------------------------------------
prototype/mtp_accept/m3_loop_rate.sh                                        # shipping loop, 49 lines
EXTRA="--mtp-dense 1" OUT=/tmp/m3_dense_full prototype/mtp_accept/m3_loop_rate.sh
EXTRA="--mtp-prompt-lanes 32" OUT=/tmp/m3_pl32_full prototype/mtp_accept/m3_loop_rate.sh
# and the paired read of any two witnesses, which is the honest noise statement:
.venv/bin/python3 prototype/mtp_accept/draft_cond_paired.py \
    /tmp/m3_full/m3_loop_rate.json /tmp/m3_dense_full/m3_loop_rate.json --label dense

# --- the rate probe on any fixture (no timing, any arm set) --------------------------------------------
prototype/mtp_accept/draft_cond_probe.sh                                    # llama-markdown defaults
FIXTURE=/tmp/dcp/dec8.csv STEPS=32 prototype/mtp_accept/draft_cond_probe.sh
# fixture builders (CPU):
head -c 10200 progress/2026-10-21/performance-status-2026-10-21.md > /tmp/dcp/llama_md.txt
.venv/bin/python3 prototype/mtp_accept/ids_from_text.py --text /tmp/dcp/llama_md.txt --out /tmp/dcp/llama_md.csv
.venv/bin/python3 -c "import json,pathlib; m=json.loads(pathlib.Path('data/refs/dec8/manifest.json').read_text()); pathlib.Path('/tmp/dec8_ids.csv').write_text(','.join(str(i) for i in m['prefill_ids'])+'\n')"

# --- the byte control against whatever the pre-change binary is ----------------------------------------
git worktree add /tmp/prewt HEAD --detach && (cd /tmp/prewt && engine/build.sh)
prototype/mtp_accept/flagoff_bytecheck.sh /tmp/prewt/engine/build/decode_run engine/build/decode_run
```
