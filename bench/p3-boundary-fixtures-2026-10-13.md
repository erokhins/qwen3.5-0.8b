# P3 — the three boundary fixtures, and what each one can falsify

Date 2026-10-13. Plan Q5 package **P3** (`bench/plan-prefill-2026-10-12.md` §5.1, finding **F3**).
CPU-only: torch dumper plus the CPU gate paths. **No device run in this package**, and no timing taken.

## Why these three exist

F3: the five shipped fixtures sit at 17, 17, 27, 232 and 295 prompt tokens, so **no prompt of theirs
ends on a §04.7 chunk boundary**, and none ever crossed the attention stage's 512-lane ceiling
(`MAXSC`; the deepest run any fixture reaches is `dec300f` at 295 + 128 = 423 lanes). Every §04.7
argument about "the last chunk is partial" and every §07.3
argument about "what happens above 512 lanes" was therefore unwritten-territory *ahead of* the batched
kernels that would break on it — and P4/P7 are the packages that would have had to invent their own gates
while also writing their kernels. These three are those gates, written before the kernels.

## The three pins

| tag | rendered prompt | forced steps | run-end lanes | pins | size | deep step | first stop id (248046) |
|---|---|---|---|---|---|---|---|
| `p128` | **128** = 64·2, exactly | 8 | 136 = 128 + 8 | 546 | 32.1 MiB | `decode7` | step 92 (legacy 248044 at 94) |
| `chunk64b` | **120** = 64 + 56 | 30 | 150 = 128 + 22 | 656 | 43.1 MiB | `decode29` | step 129 (legacy 248044 at 131) |
| `wide520f` | **500** | 24 | **524** = 512 + 12 | 626 | 53.4 MiB | `decode23` | **none in 300** |

Recipes are committed in `tools/dump_fixtures.sh`; the prompt texts are committed in
`speccheck/fixtures/`. All three are `--lean --logits-bits bf16` with **logits at every pinned step**, so
`engine/gate.py --no-engine` pairs the whole chain (8/8, 30/30, 24/24, `0 declared gaps`). Stop indices
were probed before the window was chosen, not after —
`.venv/bin/python tools/stop_probe.py --message "$(cat speccheck/fixtures/<tag>.txt)" --max-new 300` —
and recorded in each manifest under `stop_probe` (with the id it probed for, 248046). None of the three is
priced — no `route_allowance.json` — so they ride the flat §10.1 floors, and a copy that carried margin
values would read as new band keys to `tools/band_history.py`.

The budget arithmetic that sized all three comes off `dec64f`'s own pins on disk: 64 BF16 logits vectors
cum **30.32 MiB**, i.e. **0.4738 MiB** per step, and its one `--deep-steps` dump is **25.96 MiB**, so a
~60 MB fixture buys **~126 pinned steps without a deep step, ~71 with one**. That number is the reason
`wide520f` is shaped the way it is, below.

## What each one can falsify that the five shipped fixtures cannot

**`p128` — prompt-end exactly on a wave boundary.** This is §04.7's end-condition test: at 64·2 the
forward ends with a whole number of chunks, so there is no partial chunk and no zero-β / zero-g padding
anywhere in the prompt. A chunk loop that pads the last chunk and then **carries that padded state forward
as if it were real** — the shape of bug D11 was for conv windows, and the reason §04.7's third bullet says
the end-condition case "should be a gate of its own and is not" — is invisible in the 17/27-token prompts
(the whole prompt is one partial chunk, so a padding bug is indistinguishable from the normal case) and
blurred in the 232/295-token ones (the padding is real there, so the bug shows up as a diffuse state
drift rather than a clean "padding must not exist here"). `p128` also pins the *transition*: 8 replayed
steps take the run to 136 lanes, i.e. 8 lanes past the boundary the prompt landed on, so a kernel that
handles a boundary-exact prefill but mis-tracks the chunk counter for the decode steps is caught by the
same fixture. `decode7.L*.gdn.state_new` (18 real GDN rows, [1, 16, 128, 128] BF16) is the array §04.7's
bullet is about — it is the state whose last `L-4` columns must be exactly equal to the `L` columns of the
fingerprint after the no-padding case.

**`chunk64b` — the ragged tail, both readings of the plan's ambiguous row.** The plan's §5.1 row said
"not a multiple of 64" and the task told me to resolve the ambiguity by covering both readings, so this
one does: the **prompt** ends at 64·1 + 56 (a partial final chunk, with a full chunk carried across the
last boundary — which `p64`/`p300` never do: their prompts sit inside one partial chunk), and the **chain**
crosses 64·2 = 128 at `decode8` and ends at 150 = 64·2 + 22, so a second ragged tail appears mid-chain
and the chunk a token lands in changes during the chain. That makes it the fixture for the §04.7
zero-β / zero-g claim specifically: `decode29.L00.gdn.beta` must show the padding pattern *at the lanes
where padding exists and nowhere else*, and `gated_out` must be exactly zero at those lanes. `dec300f`,
the only other long fixture, ends at 423 lanes inside a chunk that is not the last, so it cannot say that.

**`wide520f` — the first pin above `MAXSC`.** 500 + 24 = **524 lanes**, i.e. 12 pinned steps past the
ceiling that no engine can currently serve, and the first fixture whose over-cap lanes are
**prefill-side** rather than only decode-side. Two things to be clear about, both said in the recipe:

- **THE RECIPE IS NOT THE PLAN'S.** Plan §5.1 sketched "~250 prompt tokens + ~275 forced steps, crossing
  512 inside the forced chain", and claimed that costs ~60 MiB. It costs **130 MiB**: the gate pairs every
  pinned step against its own logits (0.4738 MiB each), so a 275-step chain is over 130 MiB of logits
  before the deep step. The supervisor approved the substitution: **500 prompt tokens + 24 forced steps**,
  which crosses at the same place and fits at 53.4 MiB. "Step" below uses the gate's 0-based `decode<i>`
  names: `decode11` reads exactly 512 lanes (the last the one-pass stage can serve), **`decode12` is the
  first step the stage cannot serve**, and `decode12…decode23` — a dozen steps — live above the ceiling.
- **IT IS INERT FOR DEVICE GATING UNTIL P7.** `gate.py` refuses it at the lane-ceiling preflight, and that
  refusal is the expected state, not a failure to fix:
  `engine/gate.py --fixture wide520f --dry-run` →
  `[FAIL] the planned run fits the attention stage's lane ceiling — 500 prompt + 24 generation = 524 lanes
  vs MAXSC 512 (read from prototype/attn_decode/attn_decode_kernels.h; kernels_attn.cpp throws above it
  rather than truncating)`. What *is* self-consistent today is the pin: `--no-engine` pairs all 24 steps,
  the §08 fingerprints pass their own validity checks, and `--verify` reproduces it byte-for-byte. When P7
  raises the ceiling this fixture turns on with no other change, which is the point of making it now.
  **For P5 specifically**: P5 runs the gate over all 5 attention fixtures (§5.4) and will find this ceiling
  on its first device pass; the expected response is to report the ceiling, not to shrink the fixture.

## What none of the three can do — read this before trusting one as a gate

1. **A pin is an answer, not a comparison.** These fixtures buy nothing until a gate *compares* something
   at them. The chunked-vs-full equivalence of §04.7 and the >512 behaviour of §07.3 are P4/P6/P7's gates;
   if those packages do not wire these tags in, the fixtures are dead weight and should be deleted rather
   than kept as decoration.
2. **They cannot falsify a padding *difference*, because the §04.7 probe has never been run.** Plan §5.1's
   measurement — full 296-token prefill vs three chunks of 64 + 104, `gdn.state_new` compared at token
   295 — is still unrun, and `data/refs/p64/prefill_state_mismatch.json` still does not exist. So none of
   these pins can answer "does padding change the state at all?"; they can only answer "does the engine
   agree with HF at these lane counts?". P4 owes that probe first, and if it finds no difference then no
   gate built on these fixtures can show one either.
3. **They carry no margin.** Unpriced means the flat floors, so a wide prefill whose real route spread is
   wider than the §10.1 flat floor will read as a *violation* the first time it is priced. That is the
   safer direction (D-2), but it means a first device failure at `wide520f` is more likely to be "band too
   tight for a 500-lane route" than a wrong kernel — the D-14 rule applies before any band is moved, and
   fixing a band needs a ledger line, not an edit.
4. **`wide520f` cannot gate anything device-side until P7** (above), and `p128`'s exact-boundary case is
   only useful if the batched kernel keeps the boundary-exact path *distinct* — a kernel that pads
   everything to 64 will still match HF's unfilled lanes, so it passes `p128` while still doing the
   wasted work §04.7 is trying to forbid. p128 pins correctness; it cannot pin *efficiency*, and the
   "wasted work" half of F3 is a timing claim, which no fixture can make.
5. **The state is fingerprinted, not pinned.** §08's `state_fp`/`conv_fp`/`kv_fp` rows are moment sums
   (n2, maxabs, mean). A corruption that preserves that triple — an element permutation inside `S`, the
   same blind already written up for the §08 fingerprints — passes at all three fixtures. The one full
   per-layer step per fixture (`decode7` / `decode29` / `decode23`) is what limits this, and it is one
   step: a padding bug that first bites on the third chunk after the deep step is not caught by the
   per-element arrays, only by the fingerprint chain.
6. **The lane counts are a property of the prompt file, and only the recipe checks them.** `--verify`
   compares the manifest, which pins the prompt's token ids, so editing `speccheck/fixtures/p128.txt`
   breaks reproduction of the *existing* pin. But regenerating with an edited prompt regenerates a
   self-consistent fixture that silently guards a different lane count. The recipe therefore re-counts
   the rendered tokens (tokenizer only, ~2 s, no model) and **refuses to dump** if the count is not the
   documented one — a boundary fixture that drifts by one token is worse than none, because it reads as
   coverage. If you regenerate by hand with `reference_dump.py` directly, that guard does not run.
7. **Stop-token safety is probed, not structurally impossible.** The stop rule (§07.5) makes it
   uninformative to force past the point where the model would have stopped. All three windows sit inside
   a probed first-stop index (92 / 129 / none-in-300) and the probes went into the manifests, so a reader
   can see the window is honest rather than take my word for it. `wide520f` is the strongest case: no stop
   id appears in 300 generated tokens at all. The probes were taken at `thinking=False`, the same setting
   the dumps use — a `--thinking` dump would need the probe re-taken.

## One negative test per fixture, and the tool bugs this package found

The plan asked for one negative test per fixture. The honest version of "negative test" for a *pin* is
"an injection that must be caught by the pairing, at the fixture whose property it attacks":

- **`p128`, a prompt that is not what it claims.** Its whole value is the number 128, so the injection is
  the prompt drifting by one token: `want_tokens "$FIX/p128.txt" 127` →
  `FATAL: speccheck/fixtures/p128.txt renders to 128 tokens, but the recipe says 127 — the boundary case is
  gone.`, **rc 1, and no dump is taken**. The same guard fires if someone edits the prompt file and regenerates
  by hand, which is the only way this fixture can rot.
- **`chunk64b` / `p128`, missing-logits blindness.** Deleting a step's logits is exactly what would make a
  boundary fixture quietly stop guarding its own chain — that is why `gate.py` now refuses a gap it was
  not told about (`undeclared gap: no logits at decode13 and no manifest declaration`). Selftest injection:
  a copy of `dec64t` with `decode13.logits_last.npy` removed and no declaration → **exit 1**
  (`paired 0 of 32 steps`; the 31 undeclared steps listed). Also injected: a declaration that lies (names
  a step whose logits are present) → caught; a declaration whose step is no longer in the chain (a fixture
  regenerated with a shorter chain) → caught; and a fixture declaring *every* step absent → caught.
  `engine/gate.py --selftest` runs all six and passes.
- **`wide520f`, the ceiling is the fixture.** Its negative test is the refusal itself: the same gate at
  the same fixture with the ceiling arithmetic unchanged must fail at the lane check and nowhere else. It
  does — one failed check, the lane ceiling — and the same binary run at `dec64f` passes that check
  (`17 prompt + 64 generation = 81 lanes vs MAXSC 512`), so the refusal is specific to the over-cap
  fixture and not a broken preflight.

Two things worth keeping as traps, both found while building this:

- **A `--verify` run with a *different* provenance flag fails, and that is the check working.** My first
  `--verify` invocations left `--first-stop` off the command line; all three fixtures reported
  `REPRODUCE FAIL` with no byte-differing files, because the manifest's `stop_probe` value no longer
  matched. A reproduction check that compared tensor bytes only would have passed, and the fixture would
  have quietly lost its stop-probe provenance. Copy the recipe, do not retype it.
- **`tools/dump_fixtures.sh` is a shell script, and a comment block that trails off into unquoted
  multi-line text is a syntax error** — `bash -n tools/dump_fixtures.sh` is the cheapest check in this
  package and it caught a broken script that every later package would have inherited. It is green now.

## Re-check commands (all CPU, all short)

```bash
bash -n tools/dump_fixtures.sh                                                   # script parses
./tools/dump_fixtures.sh p128 chunk64b wide520f                                  # token guard + byte-exact reproduction
.venv/bin/python tools/reference_dump.py --message "$(cat speccheck/fixtures/p128.txt)" \
    --decode 8 --lean --deep-steps 7 --logits-bits bf16 \
    --first-stop "92 (probed to 300 generated tokens; legacy stop 248044 at 94)" \
    --verify data/refs/p128                                                      # §10.5 REPRODUCE PASS
.venv/bin/python engine/gate.py --fixture wide520f --no-engine                   # pin is self-consistent
.venv/bin/python engine/gate.py --fixture wide520f --dry-run                     # lane-ceiling refusal = expected
.venv/bin/python engine/gate.py --selftest                                       # includes the declared-gap injections
```
