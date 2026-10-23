# PROMPT-FAN — D1, closed: the draft cache does have a prompt now, and it does not reach parity at 4k

Task: `DRAFT-COND.md`'s **D1 — the draft cache has no prompt**. This branch retains the prompt's per-position
post-final-norm `h` rows through the prefill wave and plumbs `--mtp-prompt-lanes N` so a draft round can be
seeded with prompt context instead of being silently clamped to the last wave's 46 rows. Every number here is
an accept-rate fraction or a count. **No millisecond figure appears in this file**: rate-only evaluation, per
the brief, and a worker's device pass takes no timing anyway (`AGENTS.md`).

Two sentences of result, then the evidence:

* The mechanism works and is real: at the 4,142-lane reporting width the widest fan the shipping step budget
  can reach moves `a` from **0.4713 to 0.5148**, and a paired test on the same positions puts that at
  **p = 0.0074** — it is not window noise. On a mid-length fixture the same knob moves `a` from 0.3913 to
  **0.9394**.
* It does **not** reach the line. Parity for a k=1 round at 4,142 lanes is `a* = 0.652` (round cost 1.652
  step); the best arm measured at that width is **0.5329** (dense geometry + fan 146), and the largest fan the
  draft cache can physically hold is **420 lanes of a 4,142-token prompt**, which reaches **0.5366** on 41
  rounds (p = 0.125, small denominator). **The answer to "does 4k accept reach 0.65 with prompt seeding" is
  no, and widening the fan is not how it gets there** — see §4.

Fixtures and binaries: `models/Qwen3.5-0.8B`, `engine/build/decode_run` built from this tree by
`engine/build.sh`; the pre-change side is a scratch worktree of `0705093` (`/tmp/prewt`). Long fixture
`tmp/llama_input/ids_long.csv` (4,142 ids); mid fixture `/tmp/pf/mid.csv` (1,285 ids, built from
`head -c 4200 specification/09-mtp-draft-layer.md` with `prototype/mtp_accept/ids_from_text.py`).

---

## 1. What the retention is, and why it had to be a window

`PrefillScratch::norm` holds the post-final-norm `h` rows of **one wave body**, and `sc.norm` is handed out
pointing at the **last** body — 46 rows of a 4,142-token prompt at `--wave-rows 512` (the last wave holds
`4142 − 8×512 = 46` rows). That is why a prompt-lane seed could never exceed 46: the rows it needed had
already been overwritten. `specification/09-mtp-draft-layer.md` §09 is the form; the row the draft block wants
for position `p` is the norm'd hidden state at `p−1` paired with the token at `p`.

`engine/prefill.h::NormTailRetention` is the smallest thing that fixes that without changing the wave:

* a **caller-owned** device buffer of `rows` rows (`rows × 2,048` B), filled by `prefill_wave(..., keep)`;
* the window is the **last `rows` prompt positions** — the ones a fan can use — and `first_pos` records which
  prompt position row 0 holds, so the use site indexes by `position − first_pos`;
* one **stream-ordered D2D per wave that intersects the window**, written right after that wave body is
  submitted, so the copy is ordered against the body that produced the bytes;
* `keep = nullptr` (every existing caller) adds **zero** submissions — the flag-off byte control (§5.1) is the
  witness that nothing else moved.

Full-prompt staging (4,142 × 2,048 B = 8.5 MB, one copy per wave = 9 at 4k) is admissible on this box by
size, and was **not** built, because §2 shows the fan can never use more than 512 lanes: a whole-prompt
retention would be bytes nobody reads. `MTPPROMPT` prints `retain_copies`/`retain_bytes`, so the price is never
implicit.

The staged fan always includes the already-generated first token `y0` paired with `h@L−1` — which is also the
lane the loop's first plain step appends, so with the fan on that pair sits in the cache **twice**. That is a
real geometry wart and it is measured rather than waved away: staging *only* that pair (`fan=1`) leaves the
emitted stream **byte-identical to the loop arm's** on both fixtures, and `fan=2` does the same on the mid
fixture (§3) — so the duplicate carries no measurable accept-rate effect and the gain comes from the older
prompt rows. Fixing it properly means changing what a lane means, which is a separate decision.

## 2. The hard ceiling: the draft attention is one-pass at 512 lanes

`mtp_attn_row_kernel` (`prototype/attn_decode/attn_decode_kernels.h`, `engine/kernels_attn.cpp`) reads its whole
lane buffer in one pass and refuses `mtp_attn_max_lanes() = 512` lanes and above. So

```
usable_lanes = min(--mtp-cap 512, 512) = 512          # the whole cache
reserve(steps) = ceil((steps*20 + 13)/14)             # room this run's own rounds still need
fan_max(steps) = 512 − reserve(steps)
```

| `--steps` | reserve | **fan_max** | fan actually reached | `retain_copies` / `retain_bytes` | `lane_subs` (9 per lane) | rounds | `lane_subs`/round | draft `per_round` |
|---|---|---|---|---|---|---|---|---|
| 256 | 366 | **146** | 146 | 2 / 299,008 | 1,314 | 169 | 7.8 | 16 |
| 128 | 183 | **329** | 329 | 2 / 673,792 | 2,961 | 90 | 32.9 | 16 |
| 64 | 92 | **420** | 420 (asked 443) | 2 / 860,160 | 3,780 | 41 | 92.2 | 16 |

So **seeding the whole 4k prompt is structurally unreachable**: 420 lanes is 10 % of a 4,142-token prompt and
it is the ceiling, not the retention, that binds — printed as `clamped=cache_ceiling_minus_reserve` (§6). The
third column is why §4's verdict is about the draft attention's form and not about how many rows were kept.

## 3. `a(fan_width)` — the table this was commissioned to produce

All arms are accept-loop arms: `--mtp-draft 1 --mtp-accept 1 --mtp-sched loop --gdn-loop 1 --no-stop`
(`--gdn-loop 1` on **every** arm — `--mtp-accept` pins it, and a required flag on one arm only prices the flag,
`AGENTS.md`). Long fixture = 4,142 ids; the denominator every row uses is defined under the first table.

**Panel A — long fixture, `--steps 256` (fan_max 146):**

| arm | `a` | accepted/rounds | `lanes_max` | Δ vs loop | paired McNemar vs loop |
|---|---|---|---|---|---|
| `loop` (fan 0) | 0.4713 | 82/174 | 83 | — | — |
| `dense` | 0.4971 | 85/171 | 171 | +0.0258 | — |
| fan 1 (the duplicated pair only) | 0.4713 | 82/174 | — | ±0.0000 | emitted stream identical to `loop` |
| fan 2 | 0.4798 | 83/173 | — | +0.0085 | — |
| fan 46 | 0.4971 | 85/171 | 132 | +0.0258 | 12 discordant (10 vs 2), **p = 0.0386** |
| fan 96 | 0.5059 | 86/170 | 183 | +0.0346 | 14 discordant (11 vs 3), p = 0.0574 |
| fan 146 | 0.5148 | 87/169 | 234 | +0.0435 | 15 discordant (13 vs 2), **p = 0.0074** |
| `dense` + fan 146 | 0.5329 | 89/167 | 325 | +0.0616 | not run |

Rate = `accepted / rounds` where `rounds` is the run's own `DECWALL rounds` — the same number as the count of
`ACCEPT … accept=` lines and as `MTPDRAFT adjudicated`. Stated because `MTPDRAFT rounds` is one larger (the
last round's proposal is never settled inside the run) and the two denominators must not be mixed across a
table; every row here uses the first one.

**Panel B — long fixture, `--steps 128` (fan_max 329):** loop 0.3913 (36/92) → fan 146 0.4222 (38/90,
p = 0.0703) → fan 329 0.4222 (38/90, p = 0.0391). **fan 146 vs fan 329: 1 discordant position of 89, p = 1** —
doubling the seed changes one verdict in eighty-nine. The curve saturates near 146 lanes on this fixture.

**Panel C — mid fixture (1,285 ids), `--steps 128`:** loop 0.3913 (36/92) → fan 46 0.5422 (45/83) → fan 146
**0.9394** (62/66) → fan 329 0.9104 (61/67); fan 1 and fan 2 both 0.3913 (36/92) and their emitted streams are
byte-identical to the loop arm's, so the duplicated pair is worth exactly nothing. Here the fan does not
saturate until well past 46 lanes, and every accepted round emits two tokens (`emit/round 1.9394`).

Why the two fixtures disagree, because it is the whole reading of this result: the mid fixture's continuation
is a **templated enumeration whose template sits in the prompt tail** — the run emits
`2. **G8b — … (binding, bit-exact).** With \`verify="batch"\`, …` then the same sentence with `G8c`, and the
prompt tail contains item 1 (`prototype/mtp_accept/emitted_text.py` prints this and the shape statistics:
57 unique tokens in 129). A drafter that can read the tail can copy the pattern almost perfectly; a drafter
with only round lanes cannot. The long fixture's continuation is prose that does not repeat itself (72 unique
tokens in 129, best 8-gram seen once), and there prompt context buys +0.04. **So the fan's size is a property
of the continuation, not a constant to be added to the ledger headline**, and the 4k number is the one the
headline has to use.

## 4. The verdict against the pre-registered line, stated plainly

At the 4,142-lane reporting width, k=1 round cost 1.652 step ⇒ parity `a* = 0.652`, +10 % line `a ≥ 0.817`
(the arithmetic carried in `DRAFT-COND.md`; llama.cpp's MTP sits at 0.750 for reference). The best 4k arm
here is **0.5329**, and the widest fan the cache can hold gives **0.5366** on 41 rounds. **D1 does not reach
0.65, and §2 says no reachable fan does.** Two things follow, and they are the useful output of this branch:

1. **More lanes are not the missing term.** On the 4k fixture 146 → 329 lanes is 1 discordant position in 89
   (§3). Removing the 512-lane ceiling without changing anything else would buy roughly nothing on this text.
   The remaining gap to llama.cpp is not "the draft cannot see the prompt"; it is what `DRAFT-COND.md`'s other
   differences describe, and the one that this table cannot separate is the *draft attention's form* — a
   streaming/tiled draft attention that could read a whole prompt is a separate piece of work, not a knob.
2. **The mechanism is worth keeping and shipping dark.** It is `--mtp-prompt-lanes`, default 0, flag-off byte
   identical (§5.1), cheap at the shipping budget (§6), and on text whose continuation is templated off its
   own prompt tail it is worth +0.55 of accept. That is a knob to have available, not a headline.

Per the brief's instruction ("if `a` at the widest reachable fan is still under ~0.65 at 4k, say so plainly and
stop"): that condition is met, so nothing further is funded from this branch. Band widths, the verifier, the
accept arithmetic and `engine/performance-status.md` were **not** touched.

## 5. Controls — what was re-checked, and what each one can see

Each line is the command, its verdict, and what it would catch. All device passes: one at a time, own
`tools/gpu_lock.sh` TAG, correctness only, under a minute, **no timing taken**.

1. **Flag-off byte control** — `prototype/mtp_accept/flagoff_bytecheck.sh /tmp/prewt/engine/build/decode_run
   engine/build/decode_run` on `dec8` (34 files) and on the mid fixture at 128 steps (130 files):
   `CONTROL: IDENTICAL`. Catches the patch changing one bit of what the *target* computes with the knobs off.
2. **Admission** — `.venv/bin/python3 engine/gate.py --fixture dec8` on the patched binary: `ENGINE GATE PASS`.
3. **Side-pass control with the fan ON** (the invariant: only the PROPOSED token may change) —
   `draft_cond_probe.py --arm loop --arm pl146=… --control side --control sidepl146=…`, long fixture/256:
   `CONTROL target-dumps IDENTICAL across 258 files; proposals DIFFER (25 of 256)`; mid fixture/128:
   `IDENTICAL across 130 files; proposals DIFFER (36 of 128)`. Panels A/B/C each carried the same pair and
   printed the same verdict. Catches the fan secretly moving the target (a leak of future context into the
   target's own rows) — which is the failure mode this knob could have.
4. **Cross-binary row addressing (LEG 1)** — `prompt_fan_rowcheck.sh`: at a fan the *pre-change* binary could
   also reach (mid, fan 146 ≤ its 261 available rows) the old and new binaries emit byte-identical streams and
   the same `a` (0.939394 both). This is the witness that the **retention buffer is indexed the same way the
   scratch slice was**; a `−1` on `first_pos` would show as a different rate here and nowhere else.
   Scope, stated: it can only witness fans ≤ `last_wave_rows`, because the old binary cannot seed beyond those.
5. **Wave-split invariance (LEG 2)** — same script, fan 261 (> 46, so the retention spans several waves) run
   at `--wave-rows 512` and `--wave-rows 64`: emitted streams identical; the fan-0 precondition pair is
   identical too. Catches a wrong per-wave copy offset, which LEG 1 cannot reach.
6. **fan 0 is a no-op** — `--mtp-prompt-lanes 0` vs no flag: emitted streams identical, `prompt_lanes=0`.
7. **The clamps are loud** — `--mtp-prompt-lanes 99999` at 4k prints
   `wanted=99999 fitted=146 … clamped=cache_ceiling_minus_reserve`; asking 443 at `--steps 64` prints
   `wanted=443 fitted=420 clamped=cache_ceiling_minus_reserve`. Neither is silent, and neither blames the
   retention for a limit the retention did not apply.
8. **Refusals** — `--session` and `--time-stages` each refuse `--mtp-prompt-lanes` with a reason (a session has
   no single prompt wave to retain; `--time-stages` has no integer per-step bill once lane submissions are not
   one-per-step). Both verified firing.
9. **The paired test** — `prompt_fan_paired.sh` (exact McNemar on `ACCEPT … cand_pos= … accept=`, pairing by
   position, positions settled by only one arm reported separately and not counted). It resolves where the
   ±binomial error bars on the two rates overlap, which is exactly the situation in panel A (±0.074 bars on a
   +0.044 delta).

## 6. The price, in counts

* **Retention, once per prompt:** `retain_copies` stream-ordered D2D (1 if the window fits inside the last
  wave, 2 at the fans that matter at 4k, at most one per wave — 9 for a 4k prompt), `retain_bytes = fan ×
  2,048` (146 → 292 KB, 420 → 840 KB). Against a 1.5 GB checkpoint this is not a memory question
  (`AGENTS.md` scope note), so it was sized for reach, not for capacity.
* **Staging, once per prompt:** 9 block submissions per lane (`kLaneStageSubs`, `engine/mtp_draft.h`) plus one
  gather — `lane_subs` on `MTPPROMPT`: 414 / 864 / 1,314 / 3,780 for fans 46 / 96 / 146 / 420. Amortised over
  the run's own rounds (171 / 170 / 169 / 41) that is **0.15 / 0.32 / 0.49 / 5.8 draft-rounds' worth of
  submissions per round** against the draft's own `per_round=16`. The fan is cheap at the shipping budget and
  ruinous at a short one, which is another reason the wide fans are not the route.
* **Staging traffic, once per prompt:** lane `i` reads `i` lanes of draft KV, so ~`fan²/2` lane-row reads —
  10,658 lane-rows at fan 146, 88,200 at fan 420.
* **Recurring cost the seed imposes on every round:** the seeded lanes stay in the cache, so each round's draft
  attention row reads them: `lanes_max` 234 at fan 146 vs 83 at fan 0 (2.8×), 391 vs 37 at fan 329 on the mid
  fixture. Submission counts do not grow (the attention is one row either way), so this is a traffic cost and
  it is printed, not implied.

## 7. Commands, as run (the main agent owns any millisecond figure)

```bash
# GPU lock protocol used by every pass below; one TAG per pass, released at the end.
tools/gpu_lock.sh acquire "worker prompt-fan-sweep"; tools/gpu_lock.sh util | tail -1
tools/gpu_lock.sh release "worker prompt-fan-sweep"

# §3 rate tables (panels A/B/C; deletes the logits dumps after each panel, `/` is at 98 %)
bash prototype/mtp_accept/prompt_fan_sweep.sh A      # also B, C, ALL
# §5.9 paired test, same window on both arms
bash prototype/mtp_accept/prompt_fan_paired.sh        # STEPS= FANS= OUT= TAG= are env
# §5.4/§5.5 rowcheck (cross-binary + wave-split)
bash prototype/mtp_accept/prompt_fan_rowcheck.sh
# §5.1 flag-off byte control; §5.2 admission
bash prototype/mtp_accept/flagoff_bytecheck.sh /tmp/prewt/engine/build/decode_run engine/build/decode_run /tmp/pf/dec8.csv 32
bash prototype/mtp_accept/flagoff_bytecheck.sh /tmp/prewt/engine/build/decode_run engine/build/decode_run /tmp/pf/mid.csv 128
.venv/bin/python3 engine/gate.py --fixture dec8
# §5.3 side-pass control with the fan on
.venv/bin/python3 prototype/mtp_accept/draft_cond_probe.py --model models/Qwen3.5-0.8B \
    --fixture /tmp/pf/mid.csv --steps 128 --no-admission --arm "loop=" \
    --arm "pl146=--mtp-prompt-lanes,146" \
    --control "side=--mtp-draft,1,--mtp-sched,every,--gdn-loop,1,--no-stop" \
    --control "sidepl146=--mtp-draft,1,--mtp-sched,every,--gdn-loop,1,--no-stop,--mtp-prompt-lanes,146"
# §3's fixture characterisation
.venv/bin/python3 prototype/mtp_accept/emitted_text.py /tmp/pf/fan/C/acc_loop/emitted.txt
# a single fan arm, if you want the MTPPROMPT line yourself
engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file tmp/llama_input/ids_long.csv \
    --steps 256 --no-stop --gdn-loop 1 --mtp-draft 1 --mtp-accept 1 --mtp-sched loop \
    --mtp-prompt-lanes 146 --out /tmp/pf/rc/146
```

If the timing question is ever asked (it is the main agent's, in an exclusive window, blocking): the two things
worth pricing are the fan's staging against `--mtp-prompt-lanes 0` at a fixed step budget, and the recurring
`lanes_max` growth inside the draft's attention stage — both are `--time-stages`-incompatible *with* the flag
by design (§5.8), so the price has to come from two whole-run arms, not from a per-stage bill.

## 8. Traps this pass found, in the house form

| trap | where it shows |
|---|---|
| **a grep for a result line can match the help text instead**, because a help string that documents a print line contains the line's name. `grep -oE "MTPPROMPT.*"` over a run that *rejected* its arguments returns the `--mtp-prompt-lanes` help paragraph — a confident-looking "result" from a run that never ran. Correct: anchor on the printed field grammar (`fan=[0-9]+ norm_rows=…`), not on the tag, and check the wall time of the batch is not 0 s | this session, §3's re-take; `decode_run.cpp`'s help text |
| **a run that exits in 0 s is an argument refusal, not a fast kernel.** Four harness lines came out blank because `--mtp-accept` pins `--gdn-loop 1` and my ad-hoc loop omitted it; the same class as the empty-parse gates already in `AGENTS.md`, one level up (here the *harness*, not a parser, silently produced nothing). Correct: assert a field the run must print before trusting anything downstream from it | §3 re-take, `prompt_fan_paired.sh` |
| **a knob's "widest setting" is set by a kernel form three layers away.** The fan is bounded by `mtp_attn_max_lanes()` (one-pass draft attention, 512) minus a reserve that grows with `--steps`, so lowering `--steps` to widen the fan also shrinks the denominator it amortises over (§6: 0.5 → 5.8 draft-rounds per round). Any future write-up of this lever must carry both numbers or it will read as "just raise N" | §2, §6 |
| **two lanes at one rope position is legal to the engine and invisible to every gate.** Staging includes the pair the first plain step also appends, and no check notices; only staging *just* that pair (`fan=1`) and seeing the loop's rate reproduce made it visible. It is inert here (both fixtures), which is why it is recorded instead of fixed | §1, §3 |
