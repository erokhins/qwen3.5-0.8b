# ROUND-FORM — an MTP round's body does not enter the wide attention block

Written 2026-10-23, same day as the four-arm wall A/B (`wide_body_ab.sh`, tip `af46c37`).
This is the finding and the admission, not the timing: no timing was taken here, and the numbers quoted
are cited to the window that measured them.

## 1. The predicate as it stood

One line, in the attention launcher — `engine/kernels_attn.cpp`, in `attn_decode_layer`:

```cpp
const bool wave_wide = a.fattn_wide != 0 && rows > 1;
```

`a.fattn_wide` arrives from `DecodeScratch::attn_fattn_wide`, copied at the one attention site in
`submit_wave_body` (`engine/decode_step.cpp`, the `if (attn_layer)` branch), which `decode_run` fills from
`--attn-fattn-wide` — default **2** since A13 shipped (`decode_run.cpp`'s option field; the documented
contract is `kernels.h:AttnLayerArgs::fattn_wide` and `decode_step.h:DecodeScratch::attn_fattn_wide`).
So the whole rule the engine had was "more than one row ⇒ the A13 wide block", and the block's warrant is a
PREFILL measurement: −27.4 % warm at 16k lanes (`prototype/attn_wide/A13-RESULT.md`, ledger
`progress/2026-10-21/performance-status-2026-10-21.md` §59). What it was never priced at is a round: the
A12 finding that retired the same block for a *different* reason is the launch-population one
(`prototype/attn_wide/A12-WHY-NOT-FASTER.md`, ledger
`progress/2026-10-21/performance-status-2026-10-21.md` §56), and `wide_rtiles_of(2)` is one row-tile — a
2-row round submits `dim3(N_KV, 1, 1)` = 2 workgroups on a 40-CU machine.

The wall A/B came back: taking the block out of a 2-row verify body at 4,032 lanes removes −4.93 ms
(−29 %) of that body, and on the accept loop **98.690 tok/s against 78.444** with **both accept arms
accepting the same 25 of 31 rounds** — so the delta is not bought by a different draft.

## 2. The change, and where the rule lives

`engine/decode_step.h` now carries the rule as a named value and one arming function:

```cpp
constexpr int kMtpRoundAttnWide = 0;
int arm_mtp_round_attn_form(DecodeScratch& sc);
```

and the two call sites that submit a round — `mtpverify::verify_body` and `mtpaccept::accept_round` — arm it
beside the forcing they already do (`wave_gemm`/`gdn_scan`/`gdn_ckpt`), restoring on every exit path
including a throw. `attn_decode_layer` is untouched: it still reads the scratch, so the wide route is
reachable exactly as before for anything that is not a round.

**How the engine distinguishes a round from a prefill body of the same width: it cannot, and that is why the
rule is not a row-count rule.** `prefill_wave` splits at `kMaxWaveRows`, so a prompt whose length is 2 mod
512 ends in a 2-row body and `--wave-rows 2` makes every prefill body 2 rows; everything downstream of
`submit_wave_body` sees `n`, a position and a state. A predicate on `rows` would therefore re-form prefill
waves to buy a round's win, and the prefill win above is measured and stands. The discriminator that does
exist is the call site — two and only two sites feed a speculation round — and both already state their
geometry by forcing a route, so this rides there instead of inventing a heuristic the launcher cannot check.

Consequence, stated plainly: `--attn-fattn-wide` is **inert on the round path** and unchanged everywhere else,
the same standing exception `resolve_wave_gemm` makes for a one-row body. Because a compiled-in rule has no
flag to watch, the form that RAN is echoed off the scratch — `attn_fattn_wide=fattn|wide|wide_llama` on each
`VERIFY2` line and on `MTPSTAT`, from one shared spelling `kernels::attn_wide_name`.

## 3. Numerics: identical to the flag, at the width where the flag reaches only the round

`round_form_identity.sh`, four arms plus two narrow ones, one lock hold, ~8 s wall, no timing taken:

| check | width | verdict |
|---|---|---|
| new default **byte-identical** to new + `--attn-fattn-wide 0` | 1-token prompt | **PASS** (`verify2_*.bin`, `logits*.bin`, `sampled.txt` all match) |
| new + `--attn-fattn-wide 0` **byte-identical** to the pre-change binary with the same flag | 4,032 lanes | **PASS** — the rule is inert when the flag already asks for that core |
| new default **differs** from the pre-change default | 4,032 lanes | **PASS**, and the diff is exactly the six `verify2_<step>.bin` round dumps — the other 8 files (`sampled.txt`, the prefill rows) are identical, so nothing outside the round moved |

So the answer to "identical / within-band / differs" is **identical to `--attn-fattn-wide 0`**, proved at the
one width where that flag reaches nothing else (at 4,032 lanes the flag also re-forms the prefill waves, so a
byte-compare there would fail for a reason unrelated to this rule — the honest claim there is the third row
of the table: the round moved, nothing else did). On the accept loop at 4,032 lanes, 6 steps:
`emitted.txt` and the whole `MTPSTAT` tally (rounds/accepted/rejected/emitted/accept_rate) are identical
between the new default, the new default with the flag at 0, and the pre-change default — same emitted
tokens, same statistics; only the round's own logits rows moved, which is the point.

## 4. Gate admission — what `gate.py` can and cannot say

* `.venv/bin/python engine/gate.py --fixture dec8` (lock held by the caller; `gate.py` does not self-lock)
  → **`ENGINE GATE PASS`, 258 PASS / 0 FAIL**, ~identical verdict lines to the same gate run against the
  pre-change binary (checked: `[PASS]/[FAIL]` sets identical with numbers stripped). Inertness, not
  admission.
* **`engine/gate.py` cannot adjudicate this path at all, and no band or check was touched for it.** No gate
  side runs `--mtp-verify` or `--mtp-accept`: the only MTP traffic the gate sees is the `MTPDRAFT` census line
  under `QWEN_MTP_DRAFT=1`, which section **(h) THE BATCHED PATH** folds into its launch-bill expectation.
  The section that would have to grow is that one — an `(h)`-shaped pair that runs a round body twice (rule
  in force vs `--attn-fattn-wide 0`) and asserts the round's two logits rows and the `MTPSTAT` tally — and it
  has no pinned fixture to judge against, because no fixture in `data/refs/` pins a round. Naming the gap
  rather than routing around it: `round_form_identity.sh` is a *witness against a pre-change binary*, which is
  weaker than a gate against a pinned reference, and it should be read that way.

## 5. The `--wave-rows` money is not on this path

`--wave-rows` is a **scratch height**, not a body width: `allocate_scratch` sizes the row family and
`attn_part_floats(wave_rows)` from it, `submit_wave_body` refuses `n > sc.wave_rows`, and the §02.3 id ring
strides by it. None of those change what a 2-row round submits — and `--wave-rows 1` **cannot run a round at
all**: `decode_run` refuses `--mtp-verify`/`--mtp-accept` below `kRows`, and `verify_body`/`accept_round`
repeat the refusal. The figure the brief carries (2.349 ms "at 4 rows") comes from `row_scale.sh`, whose
own header says what `--wave-rows N` means there: *"a PREFILL pass chunked with `--wave-rows N` IS a
sequence of N-row bodies"* — i.e. there the knob **is** the body width, and the saving is per-row staging:
`ROW-SCALE.md` §2's 1-row→2-row split at 4k is `proj` +1.54 ms (the projections leave the one-row hand GEMV)
and `attn` +2.79 ms (the second row re-reads the KV scan under the wide form), plus ~0.4 ms of MLP and
recurrence. Taking it would mean making the round one row, and a round is definitionally `[cur, candidate]`:
`mtp_verify.h`'s `kRows` is 2 and the mid-checkpoint exists only at `kGdnCheckpointRows = 2`, because
"the state after row 0" is a two-row statement. So it is neither a routing change nor a scratch-geometry
change — it is the algorithm's row count, and it stops here. (Provenance note: neither `BODY-PROBE.md` nor
`METER-RESULT.md` exists in the tree at this tip — commit `48c824c`'s message says its headline is in
`METER-RESULT.md` but the file was not in the commit — so this section answers from the code and from
`ROW-SCALE.md`, and the 3.367/2.349 ms figures are the brief's, not traceable to a file.)

## 6. What the main agent should run

* Wall re-take, four arms with the derived canary (accept arms must accept the same drafts):
  `IDS=tmp/verify_body_scale/ids4032.csv STEPS=32 REPS=3 prototype/mtp_accept/wide_body_ab.sh`
  — note that `accept` and `accept_wide0` are now the **same** attention form, so the arm pair that used to
  price the block now prices the flag's effect on the prefill waves only; the headline to re-take is
  `accept` (no flag) against `base`.
* Corpus accept-rate witness (the 0.6556 carried figure, not re-run here):
  `STAGES=corpus prototype/mtp_accept/run_accept.sh`
* Form identity, if the rule is ever suspected of not taking:
  `PRE_BIN=<decode_run built at af46c37> prototype/mtp_accept/round_form_identity.sh`
