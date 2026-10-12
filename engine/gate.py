#!/usr/bin/env python3
"""Engine gate: does a greedy token come out right?

    .venv/bin/python engine/gate.py                 # dec8; runs engine/build/decode_run
    .venv/bin/python engine/gate.py --fixture dec64f  # the 64-step FORCED pin (P1)
    .venv/bin/python engine/gate.py --smoke 300     # free run, 300 tokens, smoke only
    .venv/bin/python engine/gate.py --no-engine     # fixture pairing + state sizes, zero device
    .venv/bin/python engine/gate.py --selftest      # proves THIS gate discriminates: no engine, no device
    .venv/bin/python engine/gate.py --dry-run       # every section against a CPU stand-in
    .venv/bin/python engine/gate.py --build         # run engine/build.sh first
    .venv/bin/python engine/gate.py --binary P --model DIR --work DIR

Two fixtures, because they prove different things and only one of them can be free-running:

  dec8     8 steps, the engine feeding ITSELF. Below A-18's chaos bound a correct engine produces
           the reference's tokens by itself, so the chain can be re-derived rather than replayed.
  dec64f   64 steps, REPLAYED: `decode_run --replay` feeds the REFERENCE's token at every step and
           the engine's own sample is compared, not consumed. That is teacher forcing, and it is the
           only way depth can be pinned at all: A-18 measures two legal bf16 continuations of the
           same prompt agreeing only 0.8776 of the way over 49 prompts x 24 tokens, so a 64-token
           FREE-run id match fails by construction for a correct engine. What depth actually tests
           is state carry-over — GDN state, conv window, KV — 64 deep instead of 8.

`--smoke N` is neither: a free run of N tokens that must not crash, must not emit a NaN, must stay
inside its state, and must end by length or by the stop id. It is not a parity claim and it is not a
timing; parity is claimed only over the forced pins.

`--selftest` re-runs this file in subprocesses against injected bugs — a wrong sampled id, a swapped
top-two logits, 5 % logits noise that leaves the argmax alone, a §07.5 stop check firing during the
prompt phase, a replay step fed the wrong token, and — at a step the pin ties — the engine landing on
an untied id (must FAIL), on the pin's other tied id (must PASS as TIE), and on the pin's own top-1
with its logit 1 BF16 ULP off the tied payload (must PASS, because a step where engine and pin agree
is an ordinary match: D-1) — and requires each to fail at the step it was injected at, because a
negative that every check would catch proves nothing (`prototype/mlp_gemv/results.md`). It is the only
part of a gate that can be validated before the engine has ever run, and it is why a first device run
need not double as a test of this file.

Sections, in the order the task fixed them plus the ones the deep fixture needed:

  (pre) BAND HISTORY — every number in `BAND`, and every priced fixture's route margin, against
      `bench/band-history.jsonl`. An unrecorded move and this gate refuses to run at all
      (`tools/band_history.py`); it runs before any fixture read or device work, and it is the mechanical
      half of `AGENTS.md`'s "never loosen a gate to make it pass".

  (0) FIXTURE PAIRING, proved rather than assumed (§10.5). See `pairing_checks`.
  (a) SAMPLED IDS — the engine's sample after the prompt's last token plus its generation samples
      must equal the manifest's `greedy_continuation`, exactly. §10.1 G7 requires token-for-token
      exactness, and a token id is an integer: there is no band to argue about. The ONE exception is a
      step where the pin itself is an exact tie AND the engine picked one of the other tied ids, which
      leaves an argmax with no defined winner to agree with; that is a named, proved exception (TIE
      POLICY, next to the band rationale), every use of it is printed as a TIE line and counted in the
      summary, and a step where the engine reproduced the pin's own id is never one of its uses.
  (b) THE LOGITS — each step's logits against that step's pinned `logits_last.npy`, inside the band
      rule printed for that step: the measured ROUTE ALLOWANCE where `engine/route_allowance.py` has
      priced this fixture, the flat published §10 floors where it has not. The rule in force is named on
      every step line, and the derivation of its margins is in the ROUTE ALLOWANCE block below.
  (c) STATE SIZES — the bytes the engine reports for its request state must equal what
      `engine/state_sizes.py` derives from `state.h`; that file is *run*, not re-typed.
  (d) LAUNCH PLAN — one token's launch bill, counted from §07 in Python and compared with the engine's.
  (e) THE CURVE — the per-step logits and state-delta table across the whole forced run, with a drift
      reading, because the point of depth is whether anything degrades with step count. It reports;
      it does not invent a new band, and the reason is in the BAND comment below.
  (f) FIXTURE INTEGRITY of the state deltas — the pinned fingerprints must be finite and the pinned KV
      delta must be exactly 0 (the cache is append-only). A check on the dump, not the engine: it is
      what caught a misaligned delta in `tools/reference_dump.py`'s first version of this code.
  (g) G5-STATE — the engine's own §08 state, D2H'd at the prompt end and the run end by
      `decode_run --state-fp`, against those same pinned fingerprints. What is gated is what §08.6
      makes exact — append-only Δ² of 0.0, one lane per token, a state that moved, a window that
      rolled, no family left at the §08.4 zero — plus every row's geometry and the shared helper
      proved against the pins themselves. The engine-vs-pin DISTANCES are printed and not gated: plan
      `bench/plan-prefill-2026-10-12.md` §2.6 ships this as "a measurement and a tripwire, not a
      band", and the band is a main-agent decision with a `D-x` and a ledger line behind it.

The two phases, and the line protocol that shows them. Since P2 the prompt is fed as ONE prefill wave
(`engine/prefill.h`, §06.1) and its STEP lines say `kind=prefill`; generation steps say `kind=decode`.
§07.5's stop check is armed in the second phase only, and this gate asserts both halves of that rule:
no prefill row may carry `stop=1`, and the wave's LAST row must carry the fixture's first answer token
(`greedy_continuation[0]`, §06.5's `y0`). Under `--prefill-logits last` — §06.5's shape, and the default
— the wave argmaxes only that last row, so the earlier prefill rows print `sampled=-1`; `all` prices
every row and is the shape the wave's byte-identity comparison against the pre-P2 per-token prompt loop
is made in (`bench/plan-prefill-2026-10-12.md` §4, row P2).

Why this file contains no engine arithmetic. Its value is independence. A Python mirror of the
kernels would check the engine against the engine, which is the failure `bench/gate-audit.md` exists
to find. Everything Python-side here is either a §10 metric, the §00/§08 state derivation, or the
§03.4 argmax used to prove the fixture pairs with itself. Nothing here recomputes a model output.

CPU-only in `--no-engine` mode; the default mode runs the device binary as a subprocess and reads its
dumps. It does not import torch (the reference's own dumps are read with numpy), so it runs under the
same `.venv` the rest of the gates use. The one exception is `(g)`'s fingerprint-identity case, which
lazy-imports torch because "recompute the pinned fingerprint with the dumper's own function" is the
only way to prove the helper still matches the pins; nothing else in this file touches torch, and
`--no-engine` on a fixture with no full state tensors never imports it.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import band_history    # the band ledger this gate refuses to run against otherwise

REFS = ROOT / "data" / "refs" / "dec8"   # replaced by main() from --fixture; see `set_fixture`
SHALLOW_REFS = ROOT / "data" / "refs" / "dec8"   # the cross-check target for a deep fixture
DEFAULT_BINARY = ROOT / "engine" / "build" / "decode_run"
WORK = ROOT / "tmp" / "engine_gate"

# §07.5 / A-10: the id greedy decoding stops on — the end-of-turn token. Spelled as an id, never as
# template text, and it is the same constant as `engine::kStopToken`. The `dec8` prompt is a rendered
# chat template, so this id occurs INSIDE the prompt (index 8) as well as in a continuation.
STOP_ID = 248046

# A-18's measured bound, used as a policy line and not as a guess: below this many steps the engine
# can be gated feeding ITSELF (dec8 does, and that is the stronger property — the chain is
# re-derived, not replayed); above it a pin has to be teacher-forced, because two legal bf16
# continuations of one prompt agree only 0.8776 of the way over 49 prompts x 24 tokens. So
# `n_decode > FREE_RUN_BOUND` is exactly when this gate switches to `decode_run --replay`, and a
# divergence then shows up at one step instead of cascading through the rest of the fixture.
FREE_RUN_BOUND = 8

# The step kinds `decode_run` emits. P2 gave the prompt phase its own entry (`engine/prefill.h`), so a
# prompt-phase step now says `kind=prefill` where it used to say `kind=prompt`; what this file judges is
# the PHASE (§07.5's rule is about the prompt, not about which loop ran it), and the check names keep
# saying "prompt" for that reason. One spelling per concept: the engine and the `--dry-run` stand-in
# below both emit `prefill`, and nothing emits the old spelling.
PREFILL_KIND = "prefill"
DECODE_KIND = "decode"

# §06.1/§06.5: the two shapes a prefill wave's logits can take, exactly as `decode_run
# --prefill-logits` spells them. `last` is the engine's default and §06.5's shape — only the wave's
# last row is argmaxed, so every earlier prefill STEP line carries `sampled=-1 top_logit=none` (-1 is
# the engine's "nothing pending" marker, not a token id). `all` prices every row, which is what the
# byte-identity comparison against the pre-P2 per-token prompt loop runs in, and it is the only shape
# in which a prompt-phase argmax exists at all to be misread as a stop.
PREFILL_LOGITS_LAST = "last"
PREFILL_LOGITS_ALL = "all"


def set_fixture(name: str) -> None:
    """Point every pin read at `data/refs/<name>`. Module state rather than a parameter on nine
    functions because `load_pin` is called from the pairing, curve and integrity checks alike."""
    global REFS
    REFS = ROOT / "data" / "refs" / name
    if not (REFS / "manifest.json").exists():
        raise SystemExit(f"gate: fixture {name!r} has no manifest at {REFS}/manifest.json"
                         " — make it with tools/reference_dump.py (see engine/results.md)")


def lane_ceiling() -> int:
    """MAXSC — the largest lane count the gated attention stage can serve — read out of the kernel
    header that enforces it, never retyped here. `kernels_attn.cpp` throws above it rather than
    truncating, so a planned run over the ceiling is a guaranteed failure this gate can name up
    front instead of a HIP message half way through a device window."""
    path = ROOT / "prototype" / "attn_decode" / "attn_decode_kernels.h"
    m = re.search(r"#define\s+MAXSC\s+(\d+)", path.read_text())
    if not m:
        raise SystemExit(f"gate: cannot read MAXSC from {path} — the lane ceiling is the"
                         " kernel's to own, so this gate refuses to guess it")
    return int(m.group(1))

# ---- the band ----------------------------------------------------------------------------------
# A token's logits vector is the end of a 24-layer cascade, so §10.1/§10.2 have already decided what
# it is allowed to look like, and this gate uses their published numbers instead of inventing one:
#
#   cos >= 0.9995   §10.1 G5, "incremental == batch": prefill T then decode D, compared with the
#                   reference at the same positions, is held to sampled ids exact + logits
#                   cos >= 0.9995. That is the closest published analogue of what this gate does, so
#                   it is the hard floor here.
#   meanrel <= 5 %  §10.2's cascade ceiling ("cos >= 0.999, meanrel <= 5 %"), the band G2/G3/G4 use
#                   for every whole-cascade tensor comparison.
#   cos >= 0.9999   §10.1 G6, decode logits under online softmax. REPORTED, not gated — DECIDED
#                   2026-10-11, do not promote it; the reasoning is below, not in a results file,
#                   because a band comment is the thing a future reader edits.
#
#   ndiff / max ULP REPORTED, NOT GATED. §10.1 F1-F4 is explicit about why: on a chained dot the
#   LEGAL ndiff is 29-47 %, and a max-ULP bound is blind to a whole-tensor 1-ULP shift. A threshold
#   on either would be theatre.
#
# Why G6's 0.9999 is advisory here and G5's 0.9995 is the floor — and why the three §10 cos figures
# are not interchangeable thresholds for the same quantity:
#
#   * G3i (cos >= 0.9999) measures an ISOLATED single layer fed the reference's own `s_in`, and it is
#     the tightest figure precisely because "no upstream cascade can enter" (§10.1). It is the right
#     bar for one kernel and the wrong bar for a token.
#   * G6's 0.9999 is a decode-step logits vector in the REFERENCE, i.e. the whole 24-layer chain
#     already upstream of it, with the softmax form as the only variable. It is a bf16-cascade figure,
#     not a per-op one.
#   * G5's 0.9995 is the incremental-vs-batch invariant: same model, same cascade, two ways of getting
#     to position T+D.
#   The rule, as decided: an end-to-end quantity must not be judged by a per-layer or per-operation
#   bar, because the error it is allowed to carry is the accumulated error of all 24 layers and the
#     whole token chain, not the error of one stage. G3i/G5/G6 therefore compare different things —
#     one stage, two routes to one position, and one form of one stage — and a threshold moved between
#     them is a different measurement, not a stricter or looser one.
#   What actually decides correctness on this gate is **token identity (G7, exact ids)**; the logits
#     band is supporting evidence about *how* the engine got there, which is why a step whose cos sits
#     between the two figures is reported (the G6 note below) and not failed. The first end-to-end
#     pass had exactly that profile: step 1 at cos 0.999886 — under G6, over the G5 floor, with every
#     one of the eight ids exact and the reference's own top-1/top-2 margin 7-23x the largest element
#     error at every step. It stays a documented observation. Two CPU probes price it, both with no
#     engine and no device: `engine/logits_floor.py` is the LAST-OP floor (legal reassociation of the
#     reference's own logits dot, its answer 0.99999862), and `engine/route_price.py` is the
#     END-TO-END floor (the same reference run through two legal GDN prompt forms; its spread is
#     1-cos 4.7e-5..1.9e-4 and it is wider than the engine's deviation at every step). The engine is
#     below the first and inside the second, which is why the flag is informational: a per-stage floor
#     applied to an end-to-end quantity will always fire, and the quantity that could really be wrong
#     is tokens, which is G7. engine/results.md records both tables and the reasoning.
#
# §10.2 also fixes the DEFINITIONS: cos is over the flattened tensor, meanrel = mean|Δ| / mean|ref|
# (a ratio of means, not a mean of ratios — the latter explodes on the near-zero logits that most of
# a 248 320-wide vocabulary is).
BAND = {
    "cos_hard": 0.9995,      # §10.1 G5 — the flat floor, in force where no route allowance is priced
    "meanrel_hard": 0.05,    # §10.2 cascade ceiling — same fallback rule
    "cos_report": 0.9999,    # §10.1 G6, decode logits
    "route_margin": 3.50,    # cos: FALLBACK margin, used only for a cached allowance that carries no
                             # margin of its own (the pre-D-2 global scheme). See ROUTE ALLOWANCE below.
    "route_margin_rel": 0.42,  # meanrel: the same fallback.
}

# ROUTE ALLOWANCE — the logits band is keyed on the reference's own measured route spread, per fixture
# and per step. DECIDED 2026-10-12 by the operator: option 1 of the three written up in
# `engine/results.md` ("Decision 2 of the deep gate"). The full write-up with its tables is there; what
# belongs here, next to the code that applies it, is the rule, where its numbers come from, and what was
# deliberately not chosen.
#
# Why the flat floors above had to go. They are absolute, and they were measured to be unreachable by the
# reference itself. `engine/route_price.py` runs one pinned prompt twice through `speccheck/specimpl.py`
# — once with the chunked GDN prompt form (§04.7, the one G3 canonicalises) and once with the recurrent
# one (§04.4, the form an engine with no prefill necessarily takes) — feeds the reference's own tokens,
# and measures how far apart those two LEGAL forms land on the logits vector at each step. On `dec64f`
# that spread is 1-cos 2.71e-5 .. 1.714e-3 and it goes under G5 at steps 27, 28, 29, 30, 31, 32 — six of
# the seven steps the engine goes under it, same window, same shape, no engine anywhere in the
# measurement — and its meanrel is over §10.2's 5 % at 44 of the 64 steps (worst 60.8 %, i.e. 12x the
# ceiling). An absolute floor there is not measuring the engine. `dec64t` (same pin policy, a prompt
# whose own chain never reaches its stop id) supplies the other half: there the spread never exceeds
# 1.258e-4 and the engine's worst deficit is 2.410e-4, so the same instrument says a well-conditioned
# step is entitled to be judged far more tightly than 5e-4.
#
# The rule, at step j:
#
#     1 - cos(engine_j, pin_j) <= margin     x  route_1cos(j)        (the "shape" half)
#     meanrel(engine_j, pin_j) <= margin_rel x  route_meanrel(j)     (the "scale" half)
#
# `route_*(j)` is the measurement above, cached per fixture in
# `data/refs/<fixture>/route_allowance.json` by `engine/route_allowance.py`. This gate READS that file
# and never re-prices it: re-pricing is ~5 s of torch CPU per fixture and a gate run must not silently
# re-run the reference, and a cached spread carries a provenance header (command, date, and the hashes
# `load_allowance` checks) that a freshly computed one would not.
#
# The margins are DERIVED, not chosen — that is the entire difference between this and a relabelled band.
# Definition, PER FIXTURE (D-2, 2026-10-12, which retired the single global constant): `margin` for one
# fixture = max over THAT fixture's steps of (engine deficit / route spread), i.e. the tightest constant
# that admits every step that fixture was measured on, rounded UP at two decimals by
# `engine/route_allowance.py` and stored in the fixture's own file. Why per-fixture: while the margin was
# one global max, `dec300f`'s pricing would have moved the constant every other fixture is judged by — its
# worst ratio is 7.98 at step 86 while its route spread is *tighter* than the 64-step fixtures' (median
# route 1-cos 4.07e-5 against `dec64t`'s 6.71e-5), so pricing it globally meant 3.50 → 7.99 for everybody.
# Unpriced it sat on the flat floor with 3.7e-6 of headroom at step 86, and the 4k work is a perturbation.
# Keying per fixture means the derivation cannot read another fixture's rows — `--selftest` in that tool
# asserts exactly that, and asserts it is falsifiable by checking a global max would have answered
# differently. Measured, and re-derived from the shipped allowance files rather than from memory:
#
#     fixture   margin / margin_rel   own worst cos ratio        own worst meanrel ratio   steps
#     dec64f    3.50 / 0.42           3.0368 at step 32          0.3344 at step 54         64
#     dec64t    3.50 / 0.42           3.4939 at step 54          0.4111 at step 4          64
#     dec300f   7.99 / 0.72           7.9837 at step 86          0.7198 at step 72         128
#     mt32f     2.93 / 0.24           2.9221 at step 19          0.2303 at step 3          32
#
# `dec64t`, `dec300f` and `mt32f` are carrying their own derivations. `dec64f` carries 3.50/0.42 while its
# own derivation is 3.04/0.34: those two files were written under the global rule and D-2 recorded keeping
# them, so their `margin_basis` field is absent and `load_allowance` says out loud that the margin in force
# is one the fixture did not set for itself. Re-pricing `dec64f` REFUSES until somebody deletes its file
# deliberately, which is how "keep 3.50" stays a decision on the record rather than drift.
#
# Say the inconvenient halves plainly. (1) A margin of 3.0 or less leaves `dec64f` step 32 (3.0368) and
# `dec64t` step 54 (3.4939) RED — the constant is a measured property of this engine, not a taste call.
# (2) A margin defined as `max(engine deficit / route spread)` puts that fixture's worst step exactly on
# the boundary BY CONSTRUCTION — `dec300f` step 86 now sits at 7.984 against a 7.99 cap, i.e. the rule is
# calibrated to the very errors it is meant to tolerate. That mild self-reference is known and NOT fixed
# here: D-3 (plan §3) / Q4b schedules an engine-independent multiplier from legal-route disagreement alone,
# and if that number comes out smaller than the engine needs, it is a finding about the engine, not a
# licence to loosen. Per-fixture keying removed the cross-fixture leak, not the self-reference.
#
# What it costs and what it buys, in both directions. Outside the excursion the allowance is TIGHTER than
# the floor it replaces — on `dec64t` all 64 steps get a cos cap under §10.1's 5e-4 (tightest 1.2274e-4,
# step 41), and on `dec64f` 54 of 64 do (tightest 9.4788e-05, step 24) — so this is not a loosening of
# the gate, it is a band that is tight where the reference is tight and loose only where the reference is
# provably loose. Inside the excursion it is looser: `dec64f` step 28's cap becomes 5.9983e-3, 12x the
# flat floor it replaces. What still binds there is unchanged and hard: the
# per-step argmax (the two legal routes agree with each other at 64/64 steps, and with the pin at every
# step of that window) plus the tie policy's proof.
#
# meanrel gets the SAME treatment rather than becoming reported-only, and the reason is what cos cannot
# see: cos is scale-invariant, so a mis-scaled final norm — exactly the class §10.2's meanrel ceiling was
# put in for — scores cos 1.000000 and only meanrel can catch it. Reported-only would have handed that
# failure class back, which is the hole option 2 below was rejected for; keyed on the measured spread it
# keeps the check and stops judging a step by a figure two legal reference forms exceed 12x over.
#
# What was NOT chosen:
#   * option 2, "gate the decision, not the vector": per-step argmax only, logits demoted to reported
#     evidence with the §10.2 margin statistic as the check. Cheapest and honest, and it says what the
#     gate really verifies — but it gives up any claim on the logits vector's shape, i.e. the whole
#     wrong-but-argmax-preserving class. The argmax rule is a hard gate HERE too, unchanged by this
#     decision, underneath the band in section (b), with decision 1's proved-tie exception as its only
#     qualification. It is just not the only thing this gate checks.
#   * option 3, "leave the bands and label the window": keep 0.9995 hard and carry a per-fixture list of
#     steps known to be inside legal spread, printed as `SPREAD` lines that are reported but not armed.
#     REJECTED, and the reason is on the record: an exempted step is judged by NOTHING, so a real bug
#     that lands on an exempted step passes silently — and the list is hand-maintained from a measurement
#     that can move. Under the allowance there is no exempted step and no list to keep in sync: every
#     step is judged, by a number measured at that step.
#
# Staleness is designed to be loud. Each cached file names the fixture it was priced for and records the
# prompt-id hash, the forced-feed hash, the `speccheck/specimpl.py` hash and the step count, because those
# are what the measurement was made from; `load_allowance` FAILS the gate if any of them disagrees with the
# fixture in front of it, so a re-pinned fixture or a changed reference cannot leave a stale band behind.
# It also checks the file against itself: every step's cap must equal the file's own margin times that step's
# spread, and the margin must admit the engine deficits recorded in the same file, so a hand-edited number
# fails instead of quietly becoming the band. A fixture with NO cached file — `dec8` today, whose route table
# `route_price.py` only prints — falls back to the flat published floors and prints `flat G5` on every step
# line, so the fallback is visible rather than assumed.

# TIE POLICY — an exact tie in the pin is a named, proved exception to "ids exact per step", and it
# is the strictest thing available, not a concession. DECIDED 2026-10-12 by the operator; the band
# above is untouched by it, and the two are not the same question.
#
# SCOPE, and it is the part that was wrong first (D-1, 2026-10-12): the policy is an EXCEPTION, so it
# applies only where the engine needs excusing — i.e. where the engine's id is NOT the pin's top-1 id.
# A step where engine and pin picked the same id is an ordinary exact match and must not consult the
# tie rule at all; `mt32f` step 11 proved why. There the pin ties 24.25 (payload 0x41c2) between ids
# [4016, 15048], the engine picked 4016 — the PIN'S OWN top-1 — and the step was nevertheless RED,
# because the predicate as first written also demanded the engine's F32 at the pin's top-1 round back
# to the pinned payload, and it rounded 1 ULP high (24.391083 -> 0x41c3). G7 was satisfied outright at
# that step; a tied step was being judged MORE strictly than an untied one, by a half-ULP band this
# file's own comment declines to invent. `tie_verdict` now returns three kinds — "no-tie", "match"
# (policy not consulted, the tie still noted on the log), "tie" (the clauses below armed) — and section
# (a) only arms a check on the third. `--selftest --fixture dec64f` (or `--dry-run --tie-same-ulp 17`) is
# the regression: same id, logit 1 ULP off the payload, must be GREEN.
#
# Why an exception is admissible at all: `argmax` over an exact tie is not defined by the mathematics
# (§03.4 pins ties to the lowest id, which is a *tie-BREAK* — an arbitrary choice among values the
# model called equal), so "engine's id == pin's id" cannot be demanded at such a step. Measured in
# `dec64f`: at step 17 the top two pinned logits are both 17.25 (ids 2018, 10407) and at step 18 both
# 19.75 (ids 1379, 64704) — 2 ways tied at both. That the engine's other choice is not a numerics
# failure is shown independently by `engine/route_price.py`: both legal specimpl routes also pick
# 10407 at step 17, i.e. two legal forms of the reference disagree with the pin's tie-break exactly
# as the engine does. Those two steps are what the policy is FOR, and they still resolve through it.
#
# Why the predicate is what it is, at a step the policy is actually invoked at. The exception is
# admissible only if the tie is PROVABLE FROM THE PIN, and the proof has to live in the space the tie
# exists in: the pin's BF16 payload. So:
#   1. >= 2 ids share the exact maximum **on the BF16 payload**, read through `load_pin`'s own
#      bit-shift widening (§10.5: a bitcast, never a numeric cast) — never by a float `isclose`;
#   2. the engine's sampled id is one of the tied ids;
#   3. RNE(the engine's F32 logit AT THE ENGINE'S OWN CHOSEN ID) equals the shared tied payload, i.e.
#      the engine's arithmetic kept the value the model called tied inside its own BF16 bin. All the
#      tied ids share one payload — that is what a payload tie IS — so this is one check, and it is
#      made at the id the engine actually stands on, not at the one the pin's tie-break happened to
#      name. (Checking at the pin's top-1 was D-1's other half of the same mistake: at `dec64f` step 17
#      the engine's value at 10407 is 17.237261 -> 0x418a, at 2018 it is 17.194519 -> 0x418a; both
#      hold there, which is why the old wording survived two fixtures.)
# Clause 3 in raw F32 space instead of payload space would be a category error, not strictness:
# `lm_head`'s FULL form writes an F32 accumulator with no BF16 rounding
# (`prototype/lm_head_gemv/lm_head_gemv_kernels.h:64`), so an F32-equals-BF16 test can only pass by
# luck. And clause 3 is not vacuous in payload space either: only 48 317 of 248 320 logits at step 17
# (19 %) land in their pinned BF16 bin, so it constrains real behaviour.
#
# What it deliberately does NOT do: it does not compare the engine's value at the *other* tied ids
# against a tolerance. At 17.25 one BF16 ULP is 0.125, so "within one ULP" would be a brand-new band
# with no floor probe behind it, which `AGENTS.md` forbids inventing. The engine's value at every tied
# id is therefore PRINTED (F32 and payload) and not gated, so a human can see whether the engine saw
# them as near-tied. If that ever looks load-bearing, size it with a floor probe and promote it then.
#
# A tie must never pass silently: every step the policy actually resolves is reported as TIE with the
# tied ids, the tied value and the engine's per-id values, and counted in the summary line; a tied step
# the engine matched outright is printed as an ordinary-match note, so the tie is still visible. An
# engine that picks a THIRD, untied id at a tied step fails loudly, and `--selftest` proves all three
# directions: the untied injection FAILS, the other-tied-id injection PASSES as TIE, and the
# same-id-with-a-1-ULP-off-logit injection PASSES as the ordinary match it is.
TIE_POLICY = ("exact tie in the pin (>= 2 ids share the max on the BF16 payload), the engine picked an"
              " id other than the pin's top-1, that id is one of the tied ones, and the engine's own"
              " logit at it rounds back to the shared tied payload")

fails: list[str] = []
notes: list[str] = []
ties: list[str] = []        # one entry per generation step resolved by the tie policy; counted in report()
tie_notes: list[str] = []   # tied steps the engine matched outright: printed, NOT policy decisions (D-1)


def check(ok: bool, name: str, detail: object = "") -> bool:
    # `detail` is stringified rather than typed as str because several call sites pass a list of ids,
    # which is exactly the thing an operator needs to see when a step fails.
    text = detail if isinstance(detail, str) else repr(detail)
    print(("  [PASS] " if ok else "  [FAIL] ") + name + ((" — " + text) if text else ""))
    if not ok:
        fails.append(name + (": " + text if text else ""))
    return bool(ok)


def section(title: str) -> None:
    print("\n== " + title)


def load_pin(name: str) -> np.ndarray:
    """One pinned dump. numpy has no BF16, so a uint16 payload is widened by the bit shift (§10.5:
    a bitcast, never a numeric cast — `Tensor.to(uint16)`-style truncation is the bug that ate the
    first §04.5 device run)."""
    path = REFS / name
    if not path.exists():
        raise SystemExit(f"gate: pinned dump missing: {path}")
    a = np.load(path, allow_pickle=False)
    if a.dtype == np.uint16:
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a


def argmax_low(a: np.ndarray) -> int:
    """§03.4: greedy argmax, ties to the LOWEST id. numpy's argmax returns the first maximum, which
    is the lowest index; spelled out because the C++ side has to do the same with a strict `>`.
    Note what that means at a tie: the id it returns is a property of the tie-BREAK, not of the model
    — see the TIE POLICY block above and `pin_tie` below."""
    return int(np.argmax(a))


def bf16_payload(x) -> np.ndarray:
    """F32 -> BF16 payload, round-to-nearest-even, done on the bits (no torch, no numeric cast).

    On a value that already IS BF16 this is a bitcast in disguise: it returns the payload it went in
    with, which is what makes it usable on a pin stored as F32 (dec8) as well as on the engine's F32
    logits. `engine/pin_bits_probe.py` is the measurement that the pinned logits are BF16-exact (0
    values moved in either fixture); a fixture that stopped being so says so in `pin_tie`.
    """
    u = np.asarray(x, dtype=np.float32).reshape(-1).view(np.uint32)
    lsb = (u >> 16) & 1                       # RNE, not truncation: add half an ULP plus the lsb
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bump(vec: np.ndarray, index: int) -> np.ndarray:
    """A copy of `vec` with element `index` moved to the NEXT BF16 value up, on the bits.

    The smallest change that leaves the argmax alone while moving the value out of its own BF16 bin,
    which is what the D-1 selftest injection needs: an engine whose F32 logit at the id it chose rounds
    1 ULP above the pinned payload (`mt32f` step 11). Steps up in payload space, and refuses a negative
    rather than silently stepping it DOWN — payload order is numeric order only for non-negatives.
    """
    out = np.array(vec, dtype=np.float32, copy=True).reshape(-1)
    x = float(out[index])
    if not np.isfinite(x) or x < 0:
        raise SystemExit(f"gate: the ULP bump only knows how to step a non-negative value up"
                         f" (asked for index {index}, value {x!r})")
    payload = int(bf16_payload(x)[0]) + 1
    out[index] = np.array([payload << 16], dtype=np.uint32).view(np.float32)[0]
    if not float(out[index]) > x:      # a bit trick that went sideways is worse than no injection
        raise SystemExit(f"gate: the ULP bump at index {index} moved {x!r} to"
                         f" {float(out[index])!r}, which is not one BF16 ULP up")
    return out.reshape(np.shape(vec))


def pin_tie(step: int):
    """The exact-maximum tie set of one pinned logits dump, or None if that step has a plain winner.

    Everything is on the PIN'S payload. `decode{j}.logits_last.npy` is read raw; the payload is the
    stored uint16 (or, for an older F32-stored fixture, the BF16 payload of a value the probe already
    proved is BF16-exact); the winner is located by widening that payload with `load_pin`'s bit shift,
    because BF16 payload order is not numeric order once the sign bit is set, so the MAX has to be
    taken on widened floats while MEMBERSHIP is proved on the payload. Ties are payload equality, never
    a float tolerance.
    """
    path = REFS / f"decode{step}.logits_last.npy"
    raw = np.load(path, allow_pickle=False).reshape(-1)
    stored = raw.dtype == np.uint16
    payload = raw if stored else bf16_payload(raw)
    wide = (payload.astype(np.uint32) << 16).view(np.float32)   # load_pin's bit-shift widening
    if not stored and not np.array_equal(wide, raw.astype(np.float32)):
        notes.append(f"{path.name}: its pinned F32 logits are not BF16-exact, so a payload-space tie"
                     " claim would not be sound at this step. Re-pin with --logits-bits bf16"
                     " (engine/pin_bits_probe.py is the probe that prices this).")
        return None
    top = wide.max()
    ids = [int(i) for i in np.flatnonzero(wide == top)]
    if len(ids) < 2:
        return None
    return {"ids": ids, "value": float(top), "payload": int(payload[ids[0]]),
            "top1": ids[0], "payload_is_stored": stored}


def tie_verdict(step: int, eng, sampled: int) -> tuple[str, bool, str]:
    """Decide one generation step against the tie policy: `(kind, admissible, the line to print)`.

    The single predicate for the whole file — section (a) gates on it and section (b) reuses the answer
    for its argmax line — so exactly one place decides what an engine may do at a tied step. Three
    kinds, because the policy is an exception and an exception is only needed by an engine that deviated
    from the pin (D-1, see the TIE POLICY block):

      "no-tie" — the pin has a single winner here, so the policy has nothing to say and
                 section (a) does not even print; `admissible` is False because nothing was admitted.
      "match"  — the engine's id IS the pin's top-1 id. That is G7 satisfied outright: an ordinary
                 exact match, the rule is not consulted, and the line only records that the step
                 happened to be tied. Still not `admissible`, because there was nothing to admit.
      "tie"    — the engine picked some other id at a tied step. Only now do the clauses apply, and
                 the predicate fails closed: no engine logits vector, an id outside the tied set, or a
                 logit that does not round back to the shared tied payload all mean not admissible.
    """
    tie = pin_tie(step)
    if tie is None:
        return "no-tie", False, ""
    if sampled == tie["top1"]:
        return "match", False, (
            f"step {step}: the pin is an exact {len(tie['ids'])}-way tie at {tie['value']!r}"
            f" (payload 0x{tie['payload']:04x}) among ids {tie['ids']}, and the engine picked the"
            f" pinned top-1 {tie['top1']} — an ordinary id match, so the tie policy is not consulted")

    def shown(i: int) -> str:
        if eng is None:
            return f"{i} -> (no engine logits)"
        v = float(eng[i])
        return f"{i} -> {v:.8g} (bf16 0x{int(bf16_payload(v)[0]):04x})"

    head = (f"TIE step {step}: pinned {tie['value']!r} (payload 0x{tie['payload']:04x}) tied"
            f" {len(tie['ids'])} ways among ids {tie['ids']}; engine's value at each tied id:"
            f" " + ", ".join(shown(i) for i in tie["ids"]))
    if eng is None:
        return "tie", False, head + " | NOT ADMISSIBLE: no engine logits vector, so the tie cannot be proved"
    if sampled not in tie["ids"]:
        return "tie", False, (head + f" | engine id {sampled} is NOT one of the tied ids"
                              " | NOT ADMISSIBLE")
    got = int(bf16_payload(float(eng[sampled]))[0])
    held = got == tie["payload"]
    return ("tie", held, head + f" | engine id {sampled} is one of the tied ids; engine's logit at"
                                f" that id = {float(eng[sampled]):.8g} -> bf16 0x{got:04x}"
            + f" {'==' if held else '!='} the shared tied payload 0x{tie['payload']:04x}"
            + ("" if held else " | NOT ADMISSIBLE"))


def read_engine_logits(work: Path, n_decode: int, prompt_len: int, vocab: int) -> list:
    """Every generation step's full logits vector, read once, before the section that needs them.

    Section (a) needs them because the tie policy is proved against the engine's OWN numbers, not
    around them, and a tie that cannot be proved has to fail at (a) rather than be waived. Entries are
    `(array | None, why_not)`; (b) keeps naming its own reasons so a missing dump still reads as one.
    """
    out = []
    for j in range(n_decode):
        path = work / f"logits{prompt_len + j}.bin"
        if not path.exists():
            out.append((None, "file missing — check --out"))
            continue
        a = np.fromfile(path, dtype="<f4")
        out.append((a, "") if a.size == vocab else (None, f"{a.size} floats, expected {vocab}"))
    return out


def tie_other(step: int) -> int:
    """The pin's second tied id — the legitimate alternative the tie policy exists to admit. Used by
    `--tie-alt` so the SELFTEST can prove the policy admits, not only that it refuses."""
    tie = pin_tie(step)
    if tie is None:
        raise SystemExit(f"gate: --tie-alt {step} names a step with no exact tie in the pin, so the"
                         " injection would prove nothing (the vacuous-negative trap"
                         " `prototype/mlp_gemv/results.md` warns about)")
    return tie["ids"][1]


def untied_id(step: int) -> int:
    """An id that is NOT in the pin's tie set — what an engine must never produce at a tied step.
    Deterministic (the lowest id whose pinned value is not the tied maximum), and it refuses a step
    that is not tied, for the same reason as `tie_other`."""
    tie = pin_tie(step)
    if tie is None:
        raise SystemExit(f"gate: --break-tie {step} names a step with no exact tie in the pin; that"
                         " injection would just be the ordinary wrong-id test wearing a tie costume")
    wide = load_pin(f"decode{step}.logits_last.npy")
    return int(np.flatnonzero(wide != np.float32(tie["value"]))[0])


def metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    g, r = got.astype(np.float64), ref.astype(np.float64)
    d = g - r
    order = np.argsort(r)[::-1]
    maxabs = float(np.max(np.abs(d)))
    gap = float(r[order[0]] - r[order[1]])
    return {
        "cos": float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r))),
        "meanrel": float(np.mean(np.abs(d)) / np.mean(np.abs(r))),   # §10.2's definition
        "ndiff": int(np.count_nonzero(d)),
        "maxabs": maxabs,
        "gap12": gap,
        # How many times bigger the reference's own top-1/top-2 margin is than our worst element
        # error — i.e. how much further the error would have had to grow to flip the token. Guarded:
        # a bit-exact step has maxabs 0, and dividing by an epsilon there prints a ludicrous number.
        "margin": (float("inf") if maxabs == 0.0 else gap / maxabs),
    }


def load_fp(step: int, kind: str):
    """One step's state fingerprint: [layer, 4] F64 with columns n2, maxabs, mean, dn2 (see
    `tools/reference_dump.py:fp_row`). None when the fixture predates the pins — dec8 was made
    before there were fingerprints, and everything that reads them degrades to reporting that
    rather than failing."""
    path = REFS / f"decode{step}.{kind}_fp.npy"
    return np.load(path, allow_pickle=False) if path.exists() else None


def fp_total(step: int, kind: str, col: int):
    a = load_fp(step, kind)
    return None if a is None else float(a[:, col].sum())


# ---- the route allowance -----------------------------------------------------------------------
# The rule and its derivation live in the ROUTE ALLOWANCE block next to `BAND`, above. Everything here
# is a reader: no number in this section is computed by a gate run.
ALLOWANCE_FILE = "route_allowance.json"
SELFTEST_ALLOW_FACTOR = 1.5   # how far over its own allowance the selftest injection puts a step


def _ids_sha(ids) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode()).hexdigest()[:16]


def allowance_inputs(n_steps: int) -> dict:
    """The identity of the measurement a cached allowance must have been priced from.

    Four fields, and each is there because changing it invalidates a cached spread:
      * `prefill_ids` / `feed_ids` — the route spread is a statement about ONE prompt fed ONE forced
        token chain; re-pin the fixture and the number belongs to a different prompt.
      * `n_steps` — an allowance priced with `--max-steps` (or against a shorter chain) must not be
        used for the steps it never priced.
      * `specimpl_sha256` — the spread is measured inside `speccheck/specimpl.py`, so a change to the
        reference's arithmetic is a change to the band. `engine/route_price.py`'s own hash is NOT in
        here on purpose: a doc-comment edit to the measurer is not a change to the measurement.
    """
    man = json.loads((REFS / "manifest.json").read_text())
    return {"prefill_ids": _ids_sha(man["prefill_ids"]),
            "feed_ids": _ids_sha(man["greedy_continuation"][:-1]),
            "n_steps": n_steps,
            "specimpl_sha256": hashlib.sha256(
                (ROOT / "speccheck" / "specimpl.py").read_bytes()).hexdigest()}


def allowance_margins(doc) -> tuple[float, float, str]:
    """(cos margin, meanrel margin, and where they came from) for one cached allowance file.

    PER-FIXTURE since D-2 (2026-10-12): the file states its own margins, because the derivation is a max
    over that fixture's own steps and nothing else — so a fixture that measures a wide engine/route ratio
    (`dec300f`: 7.98 at step 86, on a route spread *tighter* than the 64-step fixtures) cannot loosen the
    gate for anybody else. `BAND["route_margin"]`/`route_margin_rel` are the documented FALLBACK for a
    file written under the old global scheme, which carried no margins of its own; when they are used the
    caller says so out loud rather than applying them silently.
    """
    if doc.get("margin") is not None and doc.get("margin_rel") is not None:
        if doc.get("margin_basis"):
            return (float(doc["margin"]), float(doc["margin_rel"]),
                    "this fixture's own, derived from its own steps by engine/route_allowance.py")
        return (float(doc["margin"]), float(doc["margin_rel"]),
                "carried in this file from the retired global-max scheme (no `margin_basis`): the"
                " number binds, but it was a max over several fixtures, so it may be looser than this"
                " fixture's own derivation — see derivation.worst_ratio_by_fixture")
    return (BAND["route_margin"], BAND["route_margin_rel"],
            "the documented global fallback: this file carries no margins at all")


def read_allowance_doc():
    """The cached file's raw contents, or None if there is none. Reads only — `load_allowance` judges."""
    path = REFS / ALLOWANCE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        check(False, f"{ALLOWANCE_FILE} is readable", f"{type(e).__name__}: {e}")
        return None


def allowance_problems(doc, n_steps: int) -> list:
    """Every reason this cached file must not be trusted for this fixture. Empty list = use it.

    Four groups, in the order a reader needs them: it is the right fixture's file; its provenance still
    matches the pins and the reference in front of it; its margins are self-consistent with the caps they
    produced and with the engine deficits recorded beside them; and it covers the steps being judged.
    """
    bad = []
    if doc.get("fixture") != REFS.name:
        bad.append(f"fixture={doc.get('fixture')!r} but this is {REFS.name!r} — a per-fixture margin read"
                   " by the wrong fixture is exactly the mix-up per-fixture keying makes possible, so a"
                   " file has to name the fixture it was priced for")
    margin, margin_rel, _whence = allowance_margins(doc)
    want_inputs = allowance_inputs(n_steps)
    got_inputs = doc.get("inputs", {})
    for field, want in want_inputs.items():
        if got_inputs.get(field) != want:
            bad.append(f"inputs.{field}={got_inputs.get(field)!r} but this fixture is {want!r} — the"
                       " pins or the reference moved under this cached spread; re-price it with"
                       " engine/route_allowance.py")
    steps = doc.get("steps", [])
    for r in steps:
        j = int(r["step"])
        for col, m in (("one_cos", margin), ("meanrel", margin_rel)):
            spread, cap = float(r[f"route_{col}"]), float(r[f"allowance_{col}"])
            if abs(cap - m * spread) > 1e-9 * max(abs(cap), 1e-30):
                bad.append(f"step {j}: allowance_{col}={cap:.6e} is not margin x route_{col}"
                           f" = {m:.2f} x {spread:.6e} — the margin in force and the caps it produced"
                           " have to be the same numbers, so a hand-edited cap is a band nobody measured")
            # The margin must at least admit the engine this same file was priced against. That is not a
            # claim about TODAY's engine — (b)'s band check judges that — only that the file cannot
            # contradict itself: a margin below its own recorded ratios was edited after the fact.
            eng = r.get(f"engine_{col}_when_priced")
            if eng is not None and float(eng) / spread > m:
                bad.append(f"step {j}: the engine measured when this file was priced was"
                           f" {float(eng) / spread:.3f}x the route spread, over this file's own margin"
                           f" {m:.2f} — it could not have passed the day it was written, so the margin"
                           " or the spread was edited after the fact")
    priced = {int(r["step"]) for r in steps}
    missing = [j for j in range(n_steps) if j not in priced]
    if missing:
        bad.append(f"{len(priced)} of {n_steps} steps priced; no allowance row for step(s)"
                   f" {missing[:6]}{' …' if len(missing) > 6 else ''}")
    return bad


def load_allowance(n_steps: int):
    """The per-step allowance for this fixture as `{margin, margin_rel, margin_source, steps}`, or None.

    None means "judge by the flat published floors", which is the STRICTER fallback and is always
    announced per step, never assumed. A file that is present but does not match what it is being read
    for is a FAIL, not a fallback: a stale spread is a band nobody measured, and the whole point of
    caching it is that its provenance is checkable.
    """
    doc = read_allowance_doc()
    if doc is None:
        notes.append(f"no {ALLOWANCE_FILE} next to {REFS.name}'s pins: every step is judged by the flat"
                     " §10.1 G5 / §10.2 floors. Price it (CPU, ~5 s per fixture, no device):"
                     " `engine/route_price.py --fixture " + REFS.name + " --engine-log <gate log>"
                     " --out tmp/route_" + REFS.name + ".json` then `engine/route_allowance.py --table"
                     " tmp/route_" + REFS.name + ".json` — one fixture per run is enough now that a"
                     " margin is derived from that fixture alone (D-2).")
        return None
    bad = allowance_problems(doc, n_steps)
    check(not bad, f"{ALLOWANCE_FILE} is valid for {REFS.name}", bad[:4])
    if bad:
        return None
    gen = doc.get("generated", {})
    margin, margin_rel, whence = allowance_margins(doc)
    print(f"  allowance: {ALLOWANCE_FILE} — priced {gen.get('date', 'date unknown')} by"
          f" `{gen.get('command', 'command unknown')}`; margins cos {margin:.2f} / meanrel"
          f" {margin_rel:.2f} ({whence})")
    return {"margin": margin, "margin_rel": margin_rel, "margin_source": whence,
            "steps": {int(r["step"]): r for r in doc["steps"]}}


def rule_for_step(allowance, j: int) -> tuple:
    """(cos cap, meanrel cap, the line naming the rule in force) for step j.

    Named on every step because a reader of a gate log must never have to guess why a step passed, and
    the two rules give different answers at the same step: the allowance is tighter than the flat floor
    most places and looser in the low-energy window. The margin quoted here is THIS FIXTURE's own, taken
    from its own allowance file (D-2), because that is the only margin whose derivation looked at these
    steps.
    """
    if allowance is None:
        return (1.0 - BAND["cos_hard"], BAND["meanrel_hard"],
                f"flat G5 (cos deficit <= {1 - BAND['cos_hard']:.4e}, meanrel <="
                f" {BAND['meanrel_hard']:.0%}) — no route allowance priced for {REFS.name}")
    row = allowance["steps"][j]
    spread = float(row["route_one_cos"])
    spread_rel = float(row["route_meanrel"])
    return (float(row["allowance_one_cos"]), float(row["allowance_meanrel"]),
            f"route allowance for {REFS.name} (this step's measured spread={spread:.4e},"
            f" {REFS.name}'s own margin={allowance['margin']:.2f} -> cos deficit <="
            f" {float(row['allowance_one_cos']):.4e}; meanrel spread={spread_rel:.4%},"
            f" margin_rel={allowance['margin_rel']:.2f} -> <="
            f" {float(row['allowance_meanrel']):.4%})")


def _counterfactual(value: float, flat_cap: float, flat_name: str) -> str:
    """What a failed band line says besides its own number: whether the FLAT published floor would have
    caught it too. A failure under the flat cap is exactly the failure the old band could not see, and
    that is the sentence a reader of a gate log needs; it is printed only on a failure, because a passing
    step does not need a counterfactual.
    """
    return (f"the flat {flat_name} ({flat_cap:.4e}) would have PASSED this step — the route allowance"
            " is what caught it" if value <= flat_cap else
            f"it is outside the flat {flat_name} ({flat_cap:.4e}) too")


def fixture_step_count() -> int:
    """How many generation steps this fixture gates — the number an allowance must cover."""
    return len(json.loads((REFS / "manifest.json").read_text())["greedy_continuation"]) - 1


def allowance_row(j: int):
    """Step j's cached row for a DRY RUN, quietly, or None. `dry_run` cannot use `load_allowance`: it
    runs before section (b), and an injection whose allowance is stale must refuse rather than double-
    report the same FAIL."""
    doc = read_allowance_doc()
    if doc is None or allowance_problems(doc, fixture_step_count()):
        return None
    for row in doc["steps"]:
        if int(row["step"]) == j:
            return row
    return None


def relative_noise(vec: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Aggregate corruption that LEAVES the argmax alone: `fraction` multiplicative Gaussian noise.

    At 5 % this puts cos at ≈ 1 − σ²/2 ≈ 0.9987 — under G5's 0.9995 and over §10.2's 0.999 cascade
    floor. This is the case that asks whether the band has teeth of its own, separate from the token id.
    Shared by the injection and by `selftest`, which has to know what deficit the injection actually
    produces before it can claim the band will catch it.
    """
    rng = np.random.default_rng(4)          # fixed: the injection must be reproducible
    return (vec * (1.0 + fraction * rng.standard_normal(vec.shape))).astype(np.float32)


def allowance_noise(vec: np.ndarray, j: int, factor: float) -> np.ndarray:
    """Noise that puts step j's logits deficit at `factor` x its own cached route allowance.

    Sized, not guessed: for g = x + n with n ~ N(0, s²), 1-cos ≈ s²/(2·mean(x²)) to first order; s is
    then corrected once against the deficit actually achieved, so the injection lands on its target
    instead of near it. Constant amplitude rather than per-element proportional, because a proportional
    σ big enough to move cos by 1e-4 is a couple of percent of the LARGEST logits and can shuffle the
    top pair — that would make the argmax line the thing that caught it, which is a different test.
    """
    row = allowance_row(j)
    if row is None:
        raise SystemExit(f"gate: --break-allow {j} needs a valid {ALLOWANCE_FILE} for {REFS.name}:"
                         " the injection's whole meaning is 'above this step's measured allowance,"
                         " under the flat floor' (engine/route_allowance.py writes the file, CPU only)")
    target = factor * float(row["allowance_one_cos"])
    flat = 1.0 - BAND["cos_hard"]
    if target >= flat:
        raise SystemExit(f"gate: --break-allow {j}: 2x the allowance at this step is {target:.4e},"
                         f" which is at or over the flat §10.1 G5 floor {flat:.4e}. An injection the"
                         " old band would have caught anyway proves nothing"
                         " (`prototype/mlp_gemv/results.md`); aim it at a step with a tighter spread.")
    x = vec.astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    rng = np.random.default_rng(7)          # fixed: the injection must be reproducible
    n = rng.standard_normal(x.shape) * (rms * np.sqrt(2.0 * target))
    achieved = 1.0 - metrics(x + n, x)["cos"]
    return (x + n * np.sqrt(target / max(achieved, 1e-30))).astype(np.float32)


def break_allow_step():
    """The step `--break-allow` should be aimed at: the tightest priced allowance in this fixture whose
    injected deficit still fits under the flat G5 floor — i.e. the step where the old band was most
    nearly blind. None when no step qualifies, which `selftest` reports as a SKIP rather than quietly
    running a negative that any band would catch.
    """
    doc = read_allowance_doc()
    if doc is None:
        return None
    if allowance_problems(doc, fixture_step_count()):
        return None
    flat = 1.0 - BAND["cos_hard"]
    cands = [(float(r["allowance_one_cos"]), int(r["step"])) for r in doc["steps"]
             if SELFTEST_ALLOW_FACTOR * float(r["allowance_one_cos"]) < flat]
    return min(cands)[1] if cands else None



# ---- (e) the curve ---------------------------------------------------------------------------
def curve_checks(per_step: list, n_decode: int) -> None:
    """The per-step table, and what it says about depth.

    Reports; does not gate on a new threshold. G5's floor is already applied per step in (b) and
    inventing a second figure for "how much drift is allowed by step 64" would be exactly the
    taste-based band §10.1 F1–F4 warns about. What the table is FOR is the shape: a flat curve
    says the extra depth bought nothing beyond dec8's 8 steps, a rising one says state carry-over
    is decaying and names the step where it started. Either is a result; both are printed.
    """
    section("(e) the curve — logits and reference state deltas against step count")
    have_fp = load_fp(0, "state") is not None
    print("  step  fed        sampled    cos        meanrel   |  S n2      S dn2     conv n2"
          "  conv dn2   kv dn2")
    for j, m in enumerate(per_step):
        if m is None:
            continue
        cells = []
        for kind, col in (("state", 0), ("state", 3), ("conv", 0), ("conv", 3), ("kv", 3)):
            v = fp_total(j, kind, col)
            cells.append("-" if v is None else f"{v:.4g}")
        print(f"  {j:4d}  {m['fed']:8d}  {m['sampled']:8d}  {m['cos']:.6f}  "
              f"{m['meanrel']:8.4%}  |  " + "  ".join(f"{c:>9}" for c in cells)
              + ("" if m["argmax_ok"] else "   <- ARGMAX DISAGREES"))
    ok = [m for m in per_step if m is not None]
    if len(ok) < 2:
        notes.append("(e) has fewer than two steps to compare, so there is no curve to read.")
        return
    om = [1.0 - m["cos"] for m in ok]          # the useful orientation: 0 is perfect
    head, tail = om[:FREE_RUN_BOUND], om[-FREE_RUN_BOUND:]
    slope = float(np.polyfit(np.arange(len(om), dtype=np.float64), om, 1)[0])
    worst = int(np.argmax(om))
    print(f"  drift reading: 1-cos median over steps 0..{len(head) - 1} is "
          f"{np.median(head):.3e}, over the last {len(tail)} steps {np.median(tail):.3e}"
          f" (ratio {np.median(tail) / max(np.median(head), 1e-300):.2f}); least-squares slope"
          f" {slope:+.3e} per step; worst step {worst} at cos {ok[worst]['cos']:.6f}"
          f" ({ok[worst]['ndiff']}/{ok[worst]['n']} elements differing).")
    print("  [info] read it as: flat means the depth bought nothing beyond dec8's 8 steps; a rise"
              " means state carry-over is decaying and the step index names where. Neither is"
              " gated — (b) already applies §10.1 G5 per step, and a second threshold for "
              "\"how much drift by step 64\" would be the taste-based band §10.1 F1–F4 forbids.")
    if not have_fp:
        notes.append("(e) ran without state fingerprints: this fixture predates them (dec8 was"
                     " pinned first). Re-make it with tools/reference_dump.py to get the state"
                     " columns.")


# ---- (f) fixture integrity of the state deltas ------------------------------------------------
def fingerprint_checks(man: dict, n_decode: int) -> None:
    """Are the pinned state deltas self-consistent? A check on the DUMP, not the engine.

    The kv row's dn2 must be exactly 0: the reference's KV is append-only, so nothing already
    written may move. That is not decoration — it is the check that caught a first version of
    `fp_row` comparing a flattened prefix, which lined head 1 of a grown cache against the tail of
    head 0 and reported dn2 3.0e3 on a cache that is bit-identical (verified separately).
    """
    section("(f) fixture integrity — the pinned state deltas must describe a real cache")
    if load_fp(0, "state") is None:
        notes.append("(f) skipped: this fixture carries no state fingerprints.")
        return
    cfg = man["text_config"]
    gdn = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
    attn = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    rows = man.get("state_fingerprint", {}).get("rows_are_model_layers", {})
    check(rows.get("state") == gdn and rows.get("conv") == gdn and rows.get("kv") == attn,
          "fingerprint rows map to the §03.5 layer routing read from the manifest",
          f"state {len(rows.get('state', []))} conv {len(rows.get('conv', []))}"
          f" kv {len(rows.get('kv', []))} vs routing {len(gdn)}+{len(attn)}")
    bad_kv, bad_finite = [], []
    for s in range(n_decode):
        for kind in ("state", "conv", "kv"):
            a = load_fp(s, kind)
            if a is None:
                continue
            if not np.isfinite(a).all():
                bad_finite.append(f"decode{s}.{kind}")
            if kind == "kv" and np.any(a[:, 3] != 0.0):
                bad_kv.append(f"decode{s}: dn2 {a[:, 3].max():.3g}")
    check(not bad_finite, "every fingerprint value is finite", bad_kv[:4])
    check(not bad_kv, "the pinned KV delta is exactly 0 at every step (append-only cache)",
          bad_kv[:4])
    print(f"  [info] the S/conv delta columns are REPORTED, not gated: a moment-sum fingerprint"
          f" catches a state that decayed, was zeroed, was scaled or was left behind, and cannot"
          f" see a corruption that preserves the multiset of values. Full per-element state"
          f" gating would cost 18 MiB per step, which is the 1.7 GB this fixture exists to avoid.")


def deep_matches_shallow(n_decode: int) -> None:
    """If this is a deep fixture made from the same prompt, its first steps must be BIT-identical
    to dec8's. Two runs of the same reference on the same weights have one answer, so this is a
    free cross-check between fixtures — and it also proves the bf16-bit storage of the deep pins
    is lossless against dec8's f32 ones."""
    if REFS == SHALLOW_REFS or not (SHALLOW_REFS / "manifest.json").exists():
        return
    section("(f2) the deep fixture must extend the shallow one, not merely resemble it")
    deep_man = json.loads((REFS / "manifest.json").read_text())
    shallow_man = json.loads((SHALLOW_REFS / "manifest.json").read_text())
    if deep_man["prefill_ids"] != shallow_man["prefill_ids"]:
        notes.append(f"{REFS.name} has a different prompt from dec8, so the overlap check does"
                     " not apply.")
        return
    n_shallow = len(shallow_man["greedy_continuation"]) - 1
    agree = []
    for s in range(min(n_shallow, n_decode)):
        deep = load_pin(f"decode{s}.logits_last.npy")
        shallow = np.load(SHALLOW_REFS / f"decode{s}.logits_last.npy", allow_pickle=False)
        if shallow.dtype == np.uint16:
            shallow = (shallow.astype(np.uint32) << 16).view(np.float32)
        agree.append(np.array_equal(deep, shallow))
    check(all(agree) and bool(agree),
          f"the first {len(agree)} steps of {REFS.name} are bit-identical to dec8's",
          f"{sum(agree)}/{len(agree)} steps bit-equal (ids also equal:"
          f" {deep_man['greedy_continuation'][:len(agree) + 1] == shallow_man['greedy_continuation'][:len(agree) + 1]})")



# ---- (g) the engine's own §08 state, against the pinned fingerprints ---------------------------
# The state half of G5 ("prefill `T` then decode `D` equals one prefill of `T+D` (states, logits,
# sampled ids", §10). Sections (a)/(b) have always judged the ids and the logits; the STATES were the
# half no gate could see, because nothing read the engine's §08.1 tuple. `decode_run --state-fp` now
# D2Hs it at the two end conditions a fixture has — after the last prompt token, after the last
# generation step — and this section compares those rows with what the dumper pinned.
#
# What gets GATED here is everything §08.6 settles with exact arithmetic rather than a tolerance: the
# KV may not be rewritten (inv 3: Δ² exactly 0.0), the cache must advance by exactly one lane per
# token (inv 4), a state that did not move while tokens were fed is a state that was left behind
# (inv 1), a conv window that did not roll is not a window (inv 2), and a state that is still all zero
# after 17+ tokens was never written (the §08.4 zeroed start, left behind). Everything mechanical is
# gated too: each row's element count, the row→layer map, the §08.4 no-previous convention on both
# sides of the comparison, and the helper identity below.
#
# What is NOT gated is how far the engine's state is ALLOWED to be from the dump's.
# `bench/plan-prefill-2026-10-12.md` §2.6 assigns that in terms: "P1 ships a measurement and a
# tripwire, not a band", and whether the recurrent-route state is "close enough" to the chunk-route
# state is "decided by the main agent against those printed numbers" in the D-3/D-5 idiom, with the
# route spread at the same position as the natural denominator — which needs `route_price` extended
# to state fingerprints, a measurement that does not exist yet. Adopting a threshold here would be a
# band move without a `D-x` and a `bench/band-history.jsonl` line, and `tools/band_history.py` exists
# to refuse exactly that. So every distance below is printed at full precision and judged by nobody in
# this file. The nearest recorded yardstick for a state-level difference between two legal forms is
# G3's `cos >= 0.99999` per layer (`speccheck/gates/run_chunkcheck.py`, A-08's verification), which on
# a moment sum means an n2 ratio inside ~2e-5 for the same quantity under a different sum order. It
# is printed as a yardstick and never applied as a limit.
#
# THE CONV FAMILY IS NOT THE SAME EXTENT ON BOTH SIDES, and that is a finding, not an engine bug. The
# engine's window is `k_conv−1 = 3` pre-conv columns (§08.2, `state.h`'s `kConvWindow`); the pinned
# `conv_fp` row pools the REFERENCE's conv buffer, which is `k_conv = 4` columns wide because
# `causal_conv1d_update` cats `[conv_state, x]` before convolving (A-04). Verified on the pins, not
# argued: `dec8` and `dec64f` share one 17-token prompt, `dec8` pins the prefill raw columns
# `L{ll}.gdn.mixed_qkv_pre_conv`, and pooling dec8's newest 4 of them reproduces dec64f's pinned
# `conv_fp` row bit-for-bit for every GDN layer (newest 3 and newest 2 do not). So a correct engine
# and a correct pin differ on this family by the energy of one raw column — 19–31 % of n2 on the three
# layers checked by hand — and no engine change fixes that. The comparable quantity is the pinned
# buffer's newest 3 columns, which only the deep-step fixtures have; `conv_window_pin` derives it and
# its absence is reported, never skipped quietly.

FP_FAMILIES = ("state", "conv", "kv")
FP_COLS = ("n2", "maxabs", "mean", "dn2")


def prefill_fp(kind: str):
    """The untagged fingerprint arrays: the reference's state at the END OF THE PROMPT.

    The manifest records them with `step: "prefill"`, which is what makes them the pair for
    `decode_run --state-fp`'s `phase=prefill` rows. None when the fixture carries none: `dec8` was
    pinned before fingerprints existed, and plan §5.1 says plainly that it has no `(g)` rows.
    """
    path = REFS / f"{kind}_fp.npy"
    return np.load(path, allow_pickle=False) if path.exists() else None


def deep_state_step():
    """The deepest step whose FULL per-layer state tensors were pinned, or None.

    `--deep-steps` keeps them at one step (63 for `dec64f`/`dec64t`, 31 for `mt32f`) and
    `--deep-steps none` keeps none (`dec300f`, whose 128 steps are fingerprint-only). The full
    tensors are what both the helper-identity case and the conv-window comparison need, so this one
    question decides whether `(g)` can prove its own arithmetic here and whether it can compare a
    conv window at all — and both answers get printed, because a missing comparison must not be
    mistaken for a passing one.
    """
    steps = [int(p.name.split(".")[0][len("decode"):])
             for p in REFS.glob("decode*.L00.gdn.state_new.npy")]
    return max(steps) if steps else None


def index_state_fp(lines: list) -> tuple[dict, dict]:
    """Group the engine's `STATEFP` lines as {(phase, family, row): fields} plus {phase: prefix}.

    The prefix arrives on its own line (`STATEFP phase=prefill prefix=17`) because it is a property of
    the snapshot, not of a row; a `note=` line (a run that never left the prompt) carries neither and
    is kept under `note:<phase>` so a caller can say WHY a phase is missing instead of guessing.
    """
    rows: dict = {}
    meta: dict = {}
    for line in lines:
        phase = line.get("phase")
        if phase is None:
            continue
        if "family" not in line:
            if "prefix" in line:
                meta[phase] = int(line["prefix"])
            if "note" in line:
                meta["note:" + phase] = line["note"]
            continue
        fields = {"layer": int(line["layer"]), "n": int(line["n"]),
                  **{c: float(line[c]) for c in FP_COLS}}
        for extra in ("cols", "lanes"):
            if extra in line:
                fields[extra] = int(line[extra])
        rows.setdefault((phase, line["family"]), {})[int(line["row"])] = fields
    return rows, meta


def fp_expectations(man: dict) -> dict:
    """The row→layer map and per-row element counts, DERIVED from the manifest's `text_config`.

    Nothing here is copied from the engine or read off the pin files: the layer map comes from
    `layer_types`, the GDN row from `linear_num_value_heads × linear_key_head_dim ×
    linear_value_head_dim`, the conv row from `(linear_conv_kernel_dim − 1) ×` the q/k/v row width
    (§04.1's 6144), the pinned conv extent from `linear_conv_kernel_dim ×` it, and the KV row from
    `2 key/value heads × 2 tensors × lanes × head_dim`. So a row count or extent that stops matching
    fails here whether it moved in the engine, in the checkpoint, or in the dumper: the geometry is
    checked, not assumed.
    """
    cfg = man["text_config"]
    gdn = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
    attn = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    k_conv = cfg["linear_conv_kernel_dim"]
    qkv_rows = (2 * cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
                + cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"])
    return {
        "layers": {"state": gdn, "conv": gdn, "kv": attn},
        "n": {"state": (cfg["linear_num_value_heads"] * cfg["linear_key_head_dim"]
                        * cfg["linear_value_head_dim"]),
              "conv": (k_conv - 1) * qkv_rows},
        "pin_n": {"conv": k_conv * qkv_rows},
        "k_conv": k_conv,
        "head_dim": cfg["head_dim"],
    }


def conv_window_pin(step: int):
    """Fingerprint rows for the engine's conv WINDOW, derived from the pinned conv BUFFER.

    The newest `k_conv−1` columns of `decode{step}.L{ll}.gdn.conv_state` are what §08.2's window
    holds — A-04's `cat([conv_state, x])` is what makes the pinned buffer one column wider — so this
    is the only place the engine's conv row has something of its own extent to be compared against.
    Returns None when those tensors are not pinned, and the caller says so out loud.
    """
    from tools.state_fp import fp_row_np
    rows = json.loads((REFS / "manifest.json").read_text())[
        "state_fingerprint"]["rows_are_model_layers"]["conv"]
    paths = [REFS / f"decode{step}.L{layer:02d}.gdn.conv_state.npy" for layer in rows]
    if not all(p.exists() for p in paths):
        return None
    return np.array([fp_row_np([load_pin(p.name)[..., 1:]], None) for p in paths], dtype=np.float64)


def fp_helper_identity(man: dict) -> None:
    """The mechanical half of a `(g)` PASS: does the shared helper still reproduce the pinned
    fingerprint arrays from the pinned tensors, BIT FOR BIT?

    This is the check that the *gate's* arithmetic matches the *dumper's*, not the engine's. It matters
    because everything else in `(g)` rests on the claim that the engine's rows, the pinned rows, and
    any row this file derives all mean the same thing — and that claim is a piece of code shared
    between two programs, so it gets proved by importing the dumper's own function and running it on
    the pinned tensors. This is the only place in the file that imports torch.

    Bit-for-bit is a real constraint and a narrow one. `fp_row` sums in F32 because that is the
    arithmetic the pins were made with (§10.5: a fresh dump must reproduce them), and an F32 tree sum
    is a property of torch's reduction order, which follows the thread count: on this box the default
    reproduces all 42 rows of a deep step exactly, and `torch.set_num_threads(1)` does not (largest
    relative deviation 4.4e-7 on `mt32f`, measured while writing this). That asymmetry is exactly why
    the ENGINE's rows are compared with an F64 accumulation floor and reported, while this case can
    demand equality — this one compares two uses of one function, and the engine does not use torch.

    Column 3 (Δ²) is excluded wherever the predecessor step's tensors were not kept; `--deep-steps`
    keeps one step, so there is nothing on disk to difference against. The Δ² convention that IS
    checkable without them — §08.4's, that a first row's Δ² equals its n2 — is asserted against the
    pins in `state_fp_checks` below.
    """
    step = deep_state_step()
    if step is None:
        notes.append(f"(g) could NOT prove its own helper on {REFS.name}: this fixture keeps no full"
                     " per-layer state tensors (`--deep-steps none`), so the fp-of-pins identity has"
                     " nothing to recompute from, and the conv family cannot be priced at its own"
                     " extent either. The engine-vs-pin numbers printed for this fixture are still"
                     " real; what is missing is the proof of the ruler. Run this section on"
                     " dec64f/dec64t/mt32f, or re-make the fixture with --deep-steps (a re-pin, so a"
                     " main-agent call: plan §5.1 forbids touching the shipped fixtures).")
        return
    import torch     # the only torch this file imports; see the module header
    from tools.state_fp import fp_row

    rows = man.get("state_fingerprint", {}).get("rows_are_model_layers", {})
    if not all(v in rows for v in FP_FAMILIES):
        check(False, "the manifest states which model layer each fingerprint row is", str(rows))
        return
    total, ok = 0, True
    for family in FP_FAMILIES:
        pin = load_fp(step, family)
        if pin is None:
            check(False, f"fp-of-pins: decode{step}.{family}_fp.npy exists")
            ok = False
            continue
        if family == "kv":
            paths = [REFS / f"decode{step}.L{layer:02d}.cache_{w}.npy"
                     for layer in rows[family] for w in ("keys", "values")]
        else:
            which = "state_new" if family == "state" else "conv_state"
            paths = [REFS / f"decode{step}.L{layer:02d}.gdn.{which}.npy"
                     for layer in rows[family]]
        if not all(p.exists() for p in paths):
            check(False, f"fp-of-pins: {family}'s full tensors are pinned at decode{step}",
                  f"{sum(1 for p in paths if not p.exists())} of {len(paths)} missing")
            ok = False
            continue
        bad, worst = [], 0.0
        for i, layer in enumerate(rows[family]):
            parts = ([torch.from_numpy(load_pin(p.name)) for p in paths[2 * i:2 * i + 2]]
                     if family == "kv" else [torch.from_numpy(load_pin(paths[i].name))])
            got = fp_row(parts, None)
            total += 1
            if got[:3] != [float(x) for x in pin[i][:3]]:
                bad.append(f"{family} row {i} (layer {layer})")
                worst = max([worst] + [abs(g - p) / abs(p) if p else abs(g)
                                       for g, p in zip(got[:3], pin[i][:3])])
        check(not bad, f"fp-of-pins identity at decode{step}: {family} rows reproduce BIT-EXACTLY"
                       " with the dumper's own function (n2/maxabs/mean)",
              "" if not bad else f"{len(bad)} row(s) differ, largest relative deviation {worst:.3e};"
                                 f" first {bad[:3]} — if the deviation is at the 1e-7 level this is"
                                 " torch's reduction order (thread count), not the data")
        ok = ok and not bad
    if ok:
        print(f"  [info] {total} rows recomputed with `tools/state_fp.py::fp_row` at decode{step} and"
              " matched bit-for-bit: the pinned arrays, this file, and — up to the F64 accumulation"
              " floor `decode_run.cpp` states — the engine's rows are one definition. Whether a fresh"
              " HF run reproduces these pins byte-for-byte is §10.5's question, checked by"
              " `tools/reference_dump.py --verify`, not here.")


def fp_distance(eng: dict, pin: np.ndarray, pin_n: int) -> dict:
    """Engine-vs-pin distance over one family at one phase. Reports; never judges.

    `n2 ratio` is the summary a moment-sum fingerprint is actually for — two routes' energies divided.
    `maxabs` is an absolute deviation because it is itself a magnitude. `mean` is printed against the
    pin's own RMS (`sqrt(n2/n)`) because the mean of a state tensor is a small number whose relative
    deviation is meaningless on its own. `pin_n` is the pin's pooled extent, which differs from the
    engine's only for the conv family; where it differs, the ratio carries that mismatch and the
    caller says so rather than presenting it as a state error.
    """
    pin_rms = np.sqrt(pin[:, 0] / max(pin_n, 1))
    out = {"n2": (0.0, None), "maxabs": (0.0, None), "mean": (0.0, None), "ratios": [],
           "n2_worst": 1.0}
    for r, got in sorted(eng.items()):
        ratio = got["n2"] / pin[r][0] if pin[r][0] else float("nan")
        out["ratios"].append((r, got["layer"], ratio))
        for key, value in (("n2", abs(ratio - 1.0)),
                           ("maxabs", abs(got["maxabs"] - pin[r][1])),
                           ("mean", (abs(got["mean"] - pin[r][2]) / pin_rms[r]
                                     if pin_rms[r] else 0.0))):
            if value > out[key][0]:
                out[key] = (value, (r, got["layer"]))
    # The ratio is printed as itself, never as `1 + |deviation|`: a conv row that pools three columns
    # where the pin pooled four reads 0.72, and "1.28" would read as the engine carrying MORE energy
    # than the reference when it carries less. Sign is the finding.
    if out["ratios"] and out["n2"][1] is not None:
        out["n2_worst"] = next(r for rr, layer, r in out["ratios"]
                               if (rr, layer) == out["n2"][1])
    return out


def print_fp_distance(label: str, dist: dict, caveat: str = "") -> None:
    """One distance line, formatted so two fixtures can be diffed by eye.

    A zero worst-deviation prints as `no row differs` rather than naming row 0: a `--dry-run` prints
    all-zero lines by construction, and "worst at row 0" would read like a finding about row 0.
    """
    def at(value, cell):
        return f"row {cell[0]} (layer {cell[1]})" if value else "no row differs"
    print(f"  {label:<44} n2 ratio {dist['n2_worst']:.6f}"
          f" at {at(dist['n2'][0], dist['n2'][1])}; maxabs |Δ| {dist['maxabs'][0]:.4g}"
          f" at {at(dist['maxabs'][0], dist['maxabs'][1])}; mean |Δ|/RMS {dist['mean'][0]:.3e}"
          f" at {at(dist['mean'][0], dist['mean'][1])}"
          + ("" if not caveat else "\n" + " " * 46 + caveat))


def parse_state_fp_break(spec: str):
    """Validate `--break-state-fp WHAT:FAMILY:ROW` into `(what, family, row)`; None when unset.

    Validation lives here because an injection that silently does nothing is worse than no injection:
    it reads as a passing negative. So every impossible request is a refusal, in the same spirit as
    `pin_tie` refusing to inject a tie at an untied step. `frozen` on the KV is refused for the same
    reason — a KV row that stands still is the correct behaviour (§08.6 inv 3), so "freezing" it
    cannot be the bug; `rewrote` on `state`/`conv` is refused because those two are SUPPOSED to move,
    and the bug there is the opposite one, `frozen`.
    """
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"gate: --break-state-fp wants WHAT:FAMILY:ROW, got {spec!r}")
    what, family, row = parts
    if what not in ("zero", "frozen", "rewrote"):
        raise SystemExit(f"gate: --break-state-fp WHAT must be zero, frozen or rewrote, not"
                         f" {what!r}")
    if family not in FP_FAMILIES:
        raise SystemExit(f"gate: --break-state-fp FAMILY must be one of {FP_FAMILIES}, not"
                         f" {family!r}")
    if what == "frozen" and family == "kv":
        raise SystemExit("gate: --break-state-fp frozen:kv is not a bug — §08.6 inv 3 wants the KV"
                         " to stand still. Use rewrote:kv:ROW for the append-only failure.")
    if what == "rewrote" and family != "kv":
        raise SystemExit(f"gate: --break-state-fp rewrote:{family} is not a bug — a moving"
                         " state is the point (inv 1/2). Use frozen:"
                         + family + ":ROW for a state that was left behind.")
    try:
        return what, family, int(row)
    except ValueError:
        raise SystemExit(f"gate: --break-state-fp ROW must be an integer row index, not {row!r}")


def state_fp_stand_in_lines(man: dict, prompt_len: int, n_decode: int, break_spec: str = "") -> list:
    """`decode_run --state-fp`'s line protocol, filled with the pinned rows: a perfect engine.

    The rows ARE the pins — prompt-end rows from the untagged arrays, run-end rows from
    `decode{n_decode-1}` — so a clean `(g)` on this input proves plumbing and nothing about the
    engine, exactly like the rest of `--dry-run`. What it does prove is worth a CPU run: that the line
    protocol parses, that the geometry and maps agree with what `(g)` derives from the manifest, that
    the §08.4 and §08.6 convention lines fire on real numbers, and — with `--break-state-fp` — that
    each of the three state failures lights exactly the row it was aimed at.

    The conv rows carry the pinned FOUR-column pool under the engine's three-column label, so the conv
    distance a dry run prints is the extent mismatch and nothing else (block comment above `(g)`).
    A real engine's row adds its own arithmetic on top of that, which is precisely the difference
    between this stand-in and a device pass.
    """
    inj = parse_state_fp_break(break_spec)
    want = fp_expectations(man)
    lines = []
    phases = [("prefill", prompt_len, {k: prefill_fp(k) for k in FP_FAMILIES})]
    if n_decode > 0:
        phases.append(("end", prompt_len + n_decode,
                       {k: load_fp(n_decode - 1, k) for k in FP_FAMILIES}))
    if inj and inj[1] == "kv" and inj[2] >= len(want["layers"]["kv"]):
        raise SystemExit(f"gate: --break-state-fp {break_spec}: this fixture has"
                         f" {len(want['layers']['kv'])} KV rows")
    for family in FP_FAMILIES:
        if inj and inj[2] >= len(want["layers"][family]):
            raise SystemExit(f"gate: --break-state-fp {break_spec}: {family} has"
                             f" {len(want['layers'][family])} rows here")
    for phase, prefix, pin in phases:
        if not any(p is not None for p in pin.values()):
            continue                       # a fixture with no pins at all: (g) says so itself
        # A perfect engine's conv row is the window, i.e. the newest k_conv−1 columns of the pinned
        # buffer — so where those tensors exist at this phase's step, the stand-in uses them and the
        # window comparison reads as the identity it is. Where they do not (the prompt end of every
        # deep fixture, which keeps no prefill tensors), the stand-in falls back to the pinned
        # four-column row and the extent caveat in (g) is what carries it.
        if phase == "end" and deep_state_step() == n_decode - 1:
            window = conv_window_pin(n_decode - 1)
            if window is not None:
                pin = dict(pin, conv=window)
        lines.append(f"STATEFP phase={phase} prefix={prefix}")
        for family in FP_FAMILIES:
            rows = pin[family]
            if rows is None:
                continue
            for r in range(rows.shape[0]):
                vals = [float(x) for x in rows[r]]
                if family == "kv":
                    extent, tail = 4 * prefix * want["head_dim"], f" lanes={prefix}"
                elif family == "conv":
                    extent = want["n"]["conv"]
                    tail = f" cols={want['k_conv'] - 1}"
                else:
                    extent, tail = want["n"]["state"], ""
                # `frozen` and `rewrote` are run-end bugs by definition: at the prompt end there is no
                # earlier state to have been left behind, so the injection passes through untouched.
                if inj and inj[1] == family and inj[2] == r and (
                        inj[0] == "zero" or (inj[0] == "frozen" and phase == "end")
                        or (inj[0] == "rewrote" and phase == "end")):
                    if inj[0] == "zero":
                        vals = [0.0, 0.0, 0.0, 0.0]
                    elif inj[0] == "frozen":
                        vals[3] = 0.0
                    else:
                        vals[3] = 1.0
                lines.append(f"STATEFP phase={phase} family={family} row={r}"
                             f" layer={want['layers'][family][r]}{tail} n={extent}"
                             f" n2={vals[0]:.17g} maxabs={vals[1]:.17g} mean={vals[2]:.17g}"
                             f" dn2={vals[3]:.17g}")
    return lines


def state_fp_checks(man: dict, prompt_len: int, n_decode: int, ran_decode: int,
                    statefp: list | None, stand_in: bool = False) -> None:
    """(g) G5-STATE — the engine's §08 state against the pinned fingerprints (§08.6 inv 1–4, §10 G5).

    `statefp` is the run's `STATEFP` lines, or None when there was no engine run at all. The pin-side
    checks run either way: they are most of what P1 can prove before a device is touched, and
    `--state-fp` means the driver has been asked to look at the state at all.

    `stand_in` marks a `--dry-run`: the fingerprint rows came from this file, not from the engine, and
    the helper-identity case is skipped — not because it matters less, but because it is a property of
    the pins and the shared helper rather than of an engine, and it is the one thing in this file that
    imports torch. `--selftest` runs it once, in its own process, and says so.
    """
    section("(g) G5-STATE — the engine's §08 state against the pinned fingerprints (§08.6, §10 G5)")
    pins = {k: prefill_fp(k) for k in FP_FAMILIES}
    if all(p is None for p in pins.values()):
        notes.append(f"(g) has nothing to compare: {REFS.name} carries no state fingerprints at all"
                     " (`dec8` was pinned before there were any). Nothing in this section ran, and"
                     " that is not a pass on the state — plan §5.1 says this fixture simply has no"
                     " (g) rows.")
        return
    want = fp_expectations(man)
    fmap = man.get("state_fingerprint", {}).get("rows_are_model_layers", {})
    check(fmap == want["layers"],
          "the pinned fingerprint row map is the §03.5 routing, re-derived from the manifest's"
          " layer_types (rows are model layers, not call order)",
          f"manifest {len(fmap.get('state', []))}/{len(fmap.get('conv', []))}/"
          f"{len(fmap.get('kv', []))} vs re-derived {len(want['layers']['state'])}/"
          f"{len(want['layers']['conv'])}/{len(want['layers']['kv'])}")

    # ---- the pin side: conventions the comparisons below stand on --------------------------------
    for kind in FP_FAMILIES:
        pin = pins[kind]
        if pin is None:
            notes.append(f"(g): {kind} has no untagged (prompt-end) fingerprint in {REFS.name}, so"
                         " that family's prompt-end comparison did not run.")
            continue
        check(bool(np.isfinite(pin).all()),
              f"the pinned prompt-end {kind}_fp values are finite")
        same = bool(np.array_equal(pin[:, 3], pin[:, 0]))
        check(same, f"the pinned prompt-end {kind}_fp rows carry §08.4's convention: no predecessor,"
                    " so Δ² IS the state (dn2 == n2)",
              "" if same else f"largest |dn2−n2| {np.max(np.abs(pin[:, 3] - pin[:, 0])):.3g}")

    if stand_in:
        print("  [info] fp-of-pins identity not re-run here: this is a --dry-run, and the identity is"
              " a property of the pins and `tools/state_fp.py`, not of an engine. `--selftest` runs it"
              " once in its own process; a real gate run runs it every time.")
    else:
        fp_helper_identity(man)

    if statefp is None:
        notes.append("(g) ran the pin side only: no engine run (--no-engine), so the engine's own"
                     " state was never observed and none of the §08.6 invariants below were tested."
                     " Re-run with a device pass.")
        return
    if not statefp:
        check(False, "the run printed its state fingerprints",
              "no STATEFP line in the output, so the engine's §08 state is invisible and G5's state"
              " half is unjudged. Pass `--state-fp` to decode_run (this gate passes it by default);"
              " a silent skip here is exactly the gap P1 was written to close.")
        return

    rows, meta = index_state_fp(statefp)
    phases = [p for p in ("prefill", "end") if p in meta]
    missing = [p for p in (["prefill"] + (["end"] if ran_decode > 0 else [])) if p not in meta]
    check(not missing, "the run emitted every end condition it reached",
          "; ".join(f"{p}: " + (meta.get("note:" + p) or "no phase line — the run stopped before"
                                " it got there") for p in missing))

    # ---- the mechanical block: geometry of every row ---------------------------------------------
    geometry, finite = [], []
    for phase in phases:
        for family in FP_FAMILIES:
            got = rows.get((phase, family), {})
            if [got[r]["layer"] for r in sorted(got)] != want["layers"][family]:
                check(False, f"{phase}: {family} rows map to the §03.5 layer routing",
                      f"engine {[got[r]['layer'] for r in sorted(got)][:6]}… vs routing"
                      f" {want['layers'][family][:6]}…")
            for r in sorted(got):
                g = got[r]
                if not np.isfinite([g[c] for c in FP_COLS]).all():
                    finite.append(f"{phase} {family} row {r}")
                if family == "state" and g["n"] != want["n"]["state"]:
                    geometry.append(f"{phase} state row {r}: n={g['n']}, §08.1 says"
                                    f" {want['n']['state']}")
                if family == "conv":
                    if g["n"] != want["n"]["conv"]:
                        geometry.append(f"{phase} conv row {r}: n={g['n']}, §08.2's window is"
                                        f" (k_conv−1)×{want['n']['conv'] // (want['k_conv'] - 1)}"
                                        f" = {want['n']['conv']}")
                    if g.get("cols") not in (None, want["k_conv"] - 1):
                        geometry.append(f"{phase} conv row {r}: cols={g['cols']}, §08.2 keeps"
                                        f" k_conv−1 = {want['k_conv'] - 1}")
                if family == "kv":
                    lanes = g.get("lanes")
                    if g["n"] != 4 * (lanes or 0) * want["head_dim"]:
                        geometry.append(f"{phase} kv row {r}: n={g['n']} is not 2 heads × 2 tensors"
                                        f" × lanes {lanes} × head_dim {want['head_dim']} — a KV"
                                        " fingerprint over unwritten slots is meaningless")
                    if lanes is not None and phase in meta and lanes != meta[phase]:
                        geometry.append(f"{phase} kv row {r}: cache claims {lanes} lanes but the"
                                        f" snapshot's prefix is {meta[phase]}")
            check(bool(got), f"{phase}: the engine emitted the {family} rows",
                  f"{len(got)} of {len(want['layers'][family])}")
    check(not geometry, "every emitted row's extent matches the geometry derived from the checkpoint",
          geometry[:4])
    check(not finite, "every emitted fingerprint value is finite", finite[:4])

    # ---- §08.6, invariant by invariant, engine side ---------------------------------------------
    inv: dict[str, list[str]] = {"1": [], "2": [], "3": [], "4": []}
    if "prefill" in meta:
        ok = meta["prefill"] == prompt_len
        inv["4"].append(f"prompt end at prefix {meta['prefill']}"
                        + ("" if ok else f", expected {prompt_len}"))
        check(ok, f"inv 4: the prompt-end snapshot sits at the fixture's prefix ({prompt_len})")
    if "end" in meta:
        want_prefix = prompt_len + ran_decode
        ok = meta["end"] == want_prefix
        inv["4"].append(f"run end at prefix {meta['end']}")
        check(ok, f"inv 4: the run-end snapshot advanced the prefix by one per generated token"
                  f" ({want_prefix})", f"engine says {meta['end']}")
    for phase in phases:
        zeroed = [f"{family} row {r}" for family in FP_FAMILIES for r in rows.get((phase, family), {})
                  if rows[(phase, family)][r]["maxabs"] == 0.0]
        check(not zeroed, f"§08.4: no state family is still all zero at the {phase} end — a zeroed"
                          " moment sum after tokens were fed means the state was never written",
              "offending: [" + ", ".join(zeroed[:6]) + "]")
    if "end" in meta and ran_decode > 0:
        for family, number in (("state", "1"), ("conv", "2")):
            got = rows.get(("end", family), {})
            still = [f"{family} row {r}" for r in sorted(got) if got[r]["dn2"] == 0.0]
            check(not still,
                  f"inv {number}: every {family} row moved between the prompt end and the run end"
                  f" ({ran_decode} tokens) — a Δ² of exactly 0 is a state left behind",
                  "offending: [" + ", ".join(still[:6]) + "]")
            inv[number].append(f"{len(got) - len(still)}/{len(got)} rows moved")
        got = rows.get(("end", "kv"), {})
        rewrote = [f"kv row {r} dn2 {got[r]['dn2']:.6g}" for r in sorted(got)
                   if got[r]["dn2"] != 0.0]
        check(not rewrote,
              "inv 3: the KV's already-written lanes did not move — Δ² exactly 0.0 over the common"
              " lane extent, part-wise (K vs K, V vs V)",
              "offending: [" + ", ".join(rewrote[:6]) + "]")
        inv["3"].append(f"{len(got) - len(rewrote)}/{len(got)} rows bit-still")
    elif "end" in meta:
        notes.append("(g) inv 1/2 could not be tested: no generation step ran, so there is no pair of"
                     " states to difference.")

    # ---- the distances: printed, priced, judged by nobody in this file ---------------------------
    print("  engine-vs-pin distance, per family and end condition. REPORTED, not gated: no threshold"
          " is adopted here (plan §2.6 gives that call to the main agent in the D-3/D-5 idiom, and a"
          " band move needs a `D-x` plus a bench/band-history.jsonl line). Yardstick for scale:"
          " G3's measured chunk-vs-recurrent agreement, A-08's cos >= 0.99999/layer, which on a moment"
          " sum is an n2 ratio inside ~2e-5 for the SAME quantity under a different sum order.")
    last = max(ran_decode - 1, 0)
    step_pins = {k: (load_fp(last, k) if "end" in meta else None) for k in FP_FAMILIES}
    distances: dict = {}
    for phase, pins_here in (("prefill", pins), ("end", step_pins)):
        if phase not in meta:
            continue
        for family in FP_FAMILIES:
            got = rows.get((phase, family), {})
            pin = pins_here.get(family)
            if not got or pin is None:
                notes.append(f"(g): no {family} comparison at the {phase} end — "
                             + ("the engine emitted no rows" if not got
                                else f"the pin is missing ({family}_fp)"))
                continue
            pin_n = want["pin_n"].get(family, want["n"].get(family, 1))
            caveat = ""
            if family == "conv":
                caveat = (f"EXTENT MISMATCH, NOT A STATE ERROR: the engine pooled"
                          f" k_conv−1 = {want['k_conv'] - 1} raw columns"
                          f" ({want['n']['conv']} elements), the pin pooled k_conv ="
                          f" {want['k_conv']} ({pin_n}); see the block comment for the pin evidence"
                          f" and `conv_window_pin` for the comparable number.")
            distances[(phase, family)] = fp_distance(got, pin, pin_n)
            print_fp_distance(f"engine({phase}) vs pin({phase}) {family}",
                              distances[(phase, family)], caveat)
            if family == "conv" and phase == "end":
                win = conv_window_pin(deep_state_step() or -1)
                if win is None:
                    notes.append("(g) could NOT price the conv window on this fixture: the pinned"
                                 " conv buffer is not kept at a step whose newest 3 columns can be"
                                 " pooled (`--deep-steps none`), so inv 2 rests on the engine-side"
                                 " Δ² line above and the extent-mismatched distance, and nothing"
                                 " else. A re-pin with --deep-steps is the only way past that, and"
                                 " re-pinning a shipped fixture is a main-agent call (plan §5.1).")
                else:
                    win_d = fp_distance(got, win, want["n"]["conv"])
                    print_fp_distance("engine(end) vs pin conv WINDOW (newest k_conv−1 cols)",
                                      win_d)
                    inv["2"].append(f"window at its own extent: n2 ratio worst"
                                    f" {win_d['n2_worst']:.6f}"
                                    + (f" at row {win_d['n2'][1][0]}" if win_d["n2"][1]
                                       else " (every row identical)"))
    for family, number in (("state", "1"), ("conv", "2"), ("kv", "3")):
        for phase in ("prefill", "end"):
            d = distances.get((phase, family))
            if d and d["ratios"]:
                if d["n2"][0] == 0.0:
                    inv[number].append(f"{phase}-end n2 ratio: every row identical")
                    continue
                worst = max(d["ratios"], key=lambda t: abs(t[2] - 1.0))
                inv[number].append(f"{phase}-end n2 ratio worst {worst[2]:.6f}"
                                   f" (row {worst[0]}, layer {worst[1]})")
    print("  §08.6, invariant by invariant — what this fixture proves here and what it cannot:")
    for number, text in (("1", "S is updated by every token, in all 18 GDN layers"),
                         ("2", f"the conv window holds the last k_conv−1 raw columns (A-04: the pin"
                               f" buffers k_conv)"),
                         ("3", "the KV is append-only: already-written slots never move"),
                         ("4", "the cache advanced by one per token; prefix == position")):
        print(f"    inv {number} — {text}: " + ("; ".join(inv[number]) if inv[number]
                                                     else "not testable on this run"))
    print("    not proved by fingerprints, at any tolerance: a corruption that preserves the multiset"
          " of values (a rotation inside a tensor), the ORDER of anything, or the per-element state —"
          " moment sums cannot see them, and full per-element state pins cost 18 MiB per step, which"
          " is the 1.7 GB a deep fixture exists to avoid (§08.6's own note, restated in (f)).")
    notes.append("(g) printed engine-vs-pin state distances and gated none of them. That is the"
                 " package as planned (§2.6: 'P1 ships a measurement and a tripwire, not a band'); the"
                 " adopted-threshold decision needs the route spread at the same position, i.e."
                 " `engine/route_price.py` extended to state fingerprints, then a `D-x` and a"
                 " band_history line. Nothing in this run is a band.")

# ---- (0) the fixture pairing -------------------------------------------------------------------
def pairing_checks(man: dict) -> tuple[list, list, np.ndarray]:
    section("(0) fixture pairing — proved from the dumps (§10.5)")
    print(f"  manifest: {REFS / 'manifest.json'}")
    prompt_ids = man["sources"][0]["ids"]
    greedy = man["greedy_continuation"]
    print(f"  sources[0].ids        = {prompt_ids}")
    print(f"  greedy_continuation   = {greedy}")

    dec_logits = [load_pin(f"decode{s}.logits_last.npy").reshape(-1) for s in range(len(greedy) - 1)]
    dec_inputs = [load_pin(f"decode{s}.input_ids.npy").reshape(-1) for s in range(len(greedy) - 1)]
    prefill_logits = load_pin("logits_last.npy").reshape(-1)

    for s, (lg, ip) in enumerate(zip(dec_logits, dec_inputs)):
        meta = man["pins"][f"decode{s}.logits_last"]
        check(meta["step"] == f"decode{s}", f"manifest declares decode{s}.logits_last step",
              f'{meta["step"]} / {meta["shape"]} / {meta["dtype"]}')
        check(len(ip) == 1 and int(ip[0]) == greedy[s],
              f"decode{s}.input_ids == greedy_continuation[{s}]", ip.astype(int).tolist())
        check(argmax_low(lg) == greedy[s + 1],
              f"argmax(decode{s}.logits_last) == greedy_continuation[{s + 1}]",
              f"{argmax_low(lg)} vs {greedy[s + 1]}")
    check(argmax_low(prefill_logits) == greedy[0],
          "argmax(bare logits_last.npy) == greedy_continuation[0] — so the bare name is the PREFILL",
          f"{argmax_low(prefill_logits)} vs {greedy[0]}")
    bare_inputs = load_pin("input_ids.npy").reshape(-1)
    check(int(bare_inputs[0]) == greedy[-2],
          f"bare input_ids.npy is decode{len(greedy) - 2}'s input, NOT the prompt — recorded so it"
          " is never used as one (§10.5's last-writer-wins trap)",
          f"{bare_inputs.astype(int).tolist()} == greedy_continuation[-2] == {greedy[-2]}")
    print(f"  [info] prompt ids come from the manifest's sources[0].ids; the prompt's own logits"
          f" come from the bare logits_last.npy, whose argmax was just proved to be"
          f" greedy_continuation[0]. Pinned logits read as {dec_logits[0].dtype},"
          f" shape {dec_logits[0].shape}, stored as"
          f" {man.get('logits_storage', 'f32 (pre-dates --logits-bits)')}.")
    stop_at = [i for i, t in enumerate(prompt_ids) if t == STOP_ID]
    print(f"  [info] the prompt itself contains the stop id {STOP_ID} at index {stop_at}, so a"
          " prompt-phase step can legitimately sample it. §07.5 is a generation-phase rule and such"
          " a match must be ignored (section (a) asserts both halves of that).")
    return prompt_ids, dec_logits, dec_inputs


# ---- (c) state sizes ---------------------------------------------------------------------------
def state_checks(state_line: dict | None) -> None:
    section("(c) state sizes — against engine/state_sizes.py, which is run rather than retyped")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            from engine import state_sizes as ss
        ok = True
        detail = "imported clean; its own §00.5/spec-page checks passed"
    except SystemExit as e:      # state_sizes.py exits 1 when state.h stops matching §00.5
        ok, detail = False, f"state_sizes.py exited with {e.code}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    check(ok, "engine/state_sizes.py PASS", detail)
    if not ok:
        print(buf.getvalue().rstrip())
        return
    check(ss.gdn_state == 18 * 1024 * 1024, "GDN recurrent state is 18.0 MiB", f"{ss.gdn_state} B")
    check(ss.kv_per_token == 2 * 2 * 256 * 2 * 6,
          "KV is 12288 B/token = 2 caches × 2 kv-heads × 256 × 2 B × 6 layers",
          f"{ss.kv_per_token} B")
    if state_line is None:
        notes.append("(c) ran on the Python side only: --no-engine did not start decode_run.")
        return
    cap = int(state_line["cap"])
    want = {
        "gdn": ss.gdn_state,
        "conv": ss.conv_window,
        "kv": ss.kv_per_token * cap,
        "per_token": ss.kv_per_token,
        "total": ss.gdn_state + ss.conv_window + ss.kv_per_token * cap,
    }
    for k, v in sorted(want.items()):
        check(int(state_line[k]) == v, f"decode_run's STATE {k} matches state_sizes.py",
              f"engine {state_line[k]} vs python {v} (cap {cap})")


# ---- (d) the launch plan -----------------------------------------------------------------------
def launch_plan(n_layers: int, n_gdn: int, n_attn: int) -> dict:
    """One token's launch bill, counted from §07 — independent of the C++'s own arithmetic.

    glue   1 + 2·24. Layer 0's mixer norm has no add in front of it (§03.2's first call), and every
           other add is fused with the norm that follows it: mixer add + post-attention norm, MLP add
           + the next layer's input norm, and after layer 23 the MLP add + the FINAL norm. That last
           pairing is why the final norm is not a launch of its own, and why this is the same 49 that
           `prototype/glue/run.py` pins its token plan against — the number §07's audit in
           `bench/gate-audit.md` demanded.
    proj   GDN: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a (§04.1) + out_proj = 5.
           attention: q_proj, k_proj, v_proj (§05.1) + o_proj = 4.
    gdn    conv (§04.2), gates (§04.3), recurrence (§04.4), gated norm (§04.5) = 4.
    attn   one fused §05.1-§05.7 launch per attention layer (§07.3).
    mlp    gate+up in one launch, down in another (§03.3) = 2 per layer.
    cast   the widenings the F32-input kernels force: GDN 5 per layer (mixer norm row, one packed
           qkv+a+b, gated-norm output, MLP norm row, MLP act) and attention 4 (mixer norm row, the
           gated o_proj operand, MLP norm row, MLP act), plus ONE more for the final norm before
           lm_head. Every one of these is a dtype disagreement between graduated kernels, not model
           math — `kernels_cast.cpp` says why they cannot be dropped today.
    other  the embedding row read (§02.3), one conv-window roll-back per GDN layer (§08.2), lm_head.
    """
    return {
        "glue": 1 + 2 * n_layers,
        "proj": 5 * n_gdn + 4 * n_attn,
        "gdn": 4 * n_gdn,
        "attn": n_attn,
        "mlp": 2 * n_layers,
        "cast": 5 * n_gdn + 4 * n_attn + 1,
        "other": 1 + n_gdn + 1,
    }


def parse_run(text: str) -> dict:
    out = {"steps": [], "state": {}, "launch": {}, "result": {}, "statefp": []}
    for line in text.splitlines():
        if line.startswith("STEP "):
            out["steps"].append(dict(re.findall(r"(\w+)=([^\s]+)", line)))
        elif line.startswith("STATEFP "):
            # Its own key, and matched BEFORE `STATE ` would matter: these are the §08.6 state rows
            # section (g) judges, one dict per printed line (phase markers included). "STATEFP" does
            # not match "STATE " so the two readers cannot eat each other's lines.
            out["statefp"].append(dict(re.findall(r"(\w+)=([^\s]+)", line)))
        elif line.startswith("STATE ") or line.startswith("LAUNCH "):
            key = line.split()[0].lower()
            out[key] = dict(re.findall(r"(\w+)=([^\s]+)", line))
        elif line.startswith("RESULT "):
            out["result"] = dict(re.findall(r'(\w+)=(".*?"|[^\s]+)', line))
    return out


def dry_run(man: dict, prompt_ids: list, dec_logits: list, work: Path, break_ids: int = -1,
            break_logits: int = -1, break_noise: int = -1, break_prompt_stop: int = -1,
            replay: bool = False, break_feed: int = -1, break_tie: int = -1,
            tie_alt: int = -1, break_allow: int = -1, tie_same_ulp: int = -1,
            break_state_fp: str = "", prefill_logits: str = PREFILL_LOGITS_LAST) -> tuple[
                str, int]:
    """A stand-in for `decode_run`, on the CPU, that replays the pins as if the engine were perfect.

    Why this is in a gate: sections (a)-(d) parse a line protocol, read binary dumps, and do float
    maths that has never once run against a real logits file. Discovering a typo there costs a
    quiescent-GPU window, which is the scarcest thing in this repo (`AGENTS.md`). So the plumbing gets
    exercised here, on the CPU, against input built from the same pins the checks will compare with —
    which is exactly why it cannot be read as a result: it passes by construction. Anything it fails
    is a bug in this file, not in the engine. (`--selftest` is what proves it can also FAIL.)

    `break_ids`/`break_logits` inject one deliberate error at that generation step, the way
    `prototype/*/run.py --negative` does: the injected bug has to make the gate fail *at that step*,
    which is the property worth testing — a negative that every check would catch proves nothing
    (`prototype/mlp_gemv/results.md` has that trap written down). `break_prompt_stop` injects the one
    bug the engine actually shipped with: the §07.5 stop check firing on a prompt-phase sample, which
    truncates the run there and leaves both the rest of the prompt and all generation steps unsaid.

    `prefill_logits` picks which row shape the stand-in prints, so that it matches the `decode_run`
    whose output this is a stand-in for. Under `last` (§06.5, the default) only the wave's LAST row has
    a sample and the earlier prefill rows print `sampled=-1`; under `all` every row is priced, which is
    the shape the byte-identity gate compares. It matters for `break_prompt_stop`: an intermediate
    prompt row that was never argmaxed cannot be the row that fires a stop check, so an injection aimed
    at one is refused rather than allowed to pass vacuously.
    """
    greedy = man["greedy_continuation"]
    cfg = man["text_config"]
    lines = []
    cap = len(prompt_ids) + len(dec_logits) + 1
    plan = launch_plan(cfg["num_hidden_layers"],
                       sum(1 for t in cfg["layer_types"] if t == "linear_attention"),
                       sum(1 for t in cfg["layer_types"] if t == "full_attention"))
    lines.append(f"STATE cap={cap} gdn={ss_sizes()[0]} conv={ss_sizes()[1]}"
                 f" kv={ss_sizes()[2] * cap} total={ss_sizes()[0] + ss_sizes()[1] + ss_sizes()[2] * cap}"
                 f" per_token={ss_sizes()[2]}")
    lines.append("LAUNCH " + " ".join(f"{k}={v}" for k, v in plan.items())
                 + f" total={sum(plan.values())}")
    for i, t in enumerate(prompt_ids):
        is_last_row = i == len(prompt_ids) - 1
        # Only the LAST prefill row's sample is checked by section (a) — that is the one the fixture
        # pins and the one §06.5 says the wave is required to produce. Under `last` the earlier rows
        # were never argmaxed and say so with -1, exactly as `decode_run` does; under `all` every row is
        # priced and the earlier ones echo the next prompt token.
        if prefill_logits != PREFILL_LOGITS_ALL and not is_last_row:
            sampled, top = -1, "none"
        else:
            sampled = greedy[0] if is_last_row else (
                prompt_ids[i + 1] if i + 1 < len(prompt_ids) else greedy[0])
            top = "1"
        # `stop=` is the §07.5 CHECK being armed and matched — never the raw match. A prompt-phase step
        # reports stop=0 even if it sampled the stop id, because its sample is discarded; that is what
        # decode_run prints and what section (a) asserts. `--break-prompt-stop` is the one case where
        # it is 1, and it reproduces the first device run: the stop match ends the run AT the prompt,
        # so the remaining prompt tokens and all 8 generation steps are simply missing.
        if i == break_prompt_stop:
            if sampled < 0:
                # No argmax on this row, so there is no stop check to fire: the injection would be
                # caught by nothing except itself. Refuse it (the same refusal `--break-tie` and
                # `--break-allow` make when aimed at a step that cannot carry their bug) and name the
                # two ways to make it representable.
                raise SystemExit(
                    f"--break-prompt-stop {i}: prefill row {i} of {len(prompt_ids)} carries no"
                    " argmax under --prefill-logits last (§06.5 prices only the last row), so the bug"
                    f" cannot be injected there. Aim at the last row ({len(prompt_ids) - 1}), or"
                    " pass --prefill-logits all.")
            # The stop check fires BECAUSE the sample is the stop id, so the stand-in has to say both;
            # `decode_run` prints `stop=1` only under that condition.
            lines.append(f"STEP kind={PREFILL_KIND} index={i} input={t} sampled={STOP_ID}"
                         f" top_logit=1 pos_before={i} pos_after={i + 1} stop=1")
            lines.append(f"RESULT mode=run steps={cap - 1} ms_total=0 ms_per_step=0"
                         f" last_sampled={STOP_ID} prefix_len={i + 1} stopped=1")
            return "\n".join(lines), 0
        lines.append(f"STEP kind={PREFILL_KIND} index={i} input={t} sampled={sampled}"
                     f" top_logit={top} pos_before={i} pos_after={i + 1} stop=0")
    for j in range(len(dec_logits)):
        pos = len(prompt_ids) + j
        sampled = greedy[j + 1]
        logits = dec_logits[j]
        # What this step is fed. In a replay that is the caller's list (the reference's own token);
        # `--break-feed` is the driver feeding the wrong entry of it, an off-by-one in the replay
        # list being a completely plausible way to get this wrong.
        fed = greedy[j]
        if j == break_feed:
            fed = greedy[j] + 1
        if j == break_ids:
            sampled = greedy[j + 1] + 1        # one wrong id, exactly where a real divergence starts
        if j == break_tie:
            # The failure the tie policy exists to catch: at a step where the PIN is tied, the engine
            # produces an id that is not one of the tied ones. `untied_id` refuses a step that is not
            # tied, so this injection cannot silently degrade into the ordinary wrong-id test.
            sampled = untied_id(j)
        if j == tie_alt:
            # The other direction, which is the property the policy buys: the engine's id is the
            # pin's OTHER tied id, and the gate must admit that as a TIE rather than fail it.
            sampled = tie_other(j)
        if j == tie_same_ulp:
            # D-1's regression, in the direction the old predicate got wrong: the engine keeps the
            # pin's own top-1 id — so G7 is satisfied and the policy must not be consulted at all —
            # and its F32 logit there sits one BF16 ULP above the pinned payload, which is exactly
            # what the old clause-3-at-the-pin's-top-1 refused (`mt32f` step 11). `pin_tie` refusing an
            # untied step is what stops this degrading into a plain noise injection.
            logits = bf16_bump(logits, pin_tie(j)["top1"])
        if j == break_logits:
            # Flip the argmax: raise the runner-up past the top. This is the smallest logits-level
            # failure that matters — a token chosen wrongly, not a vector that is merely noisy.
            logits = logits.copy()
            order = np.argsort(logits)[::-1]
            logits[order[1]] = logits[order[0]] + 1.0
        if j == break_noise:
            # Aggregate corruption that LEAVES the argmax alone: 5 % relative noise puts cos at
            # ≈ 1 − σ²/2 ≈ 0.9987, under G5's 0.9995 and over §10.2's 0.999 cascade floor. This is the
            # case that asks whether the band has teeth of its own, separate from the token id.
            logits = relative_noise(logits)
        if j == break_allow:
            # The same class — argmax untouched, aggregate wrong — sized against THIS step's measured
            # route allowance instead of against the flat floor: a deficit over the allowance that still
            # fits under the flat §10.1 G5 floor, i.e. a failure the old band could not have seen. The
            # meanrel line usually goes with it (any noise that moves cos moves meanrel); the assertion
            # is about the cos line failing AT THIS STEP, which is what `selftest` checks.
            logits = allowance_noise(logits, j, SELFTEST_ALLOW_FACTOR)
        lines.append(f"STEP kind={DECODE_KIND} index={pos} input={fed} sampled={sampled}"
                     f" top_logit=1 pos_before={pos} pos_after={pos + 1} stop=0")
        (work / f"logits{pos}.bin").write_bytes(logits.astype(np.float32).tobytes())
    lines += state_fp_stand_in_lines(man, len(prompt_ids), len(dec_logits), break_state_fp)
    lines.append(f"RESULT mode={'replay' if replay else 'run'} steps=25 ms_total=0 ms_per_step=0"
                 f" last_sampled={greedy[-1]} prefix_len={cap - 1} stopped=0")
    return "\n".join(lines), 0


def ss_sizes() -> list:
    """The three constant state figures, from engine/state_sizes.py (same source section (c) uses)."""
    with contextlib.redirect_stdout(io.StringIO()):
        from engine import state_sizes as ss
    return [ss.gdn_state, ss.conv_window, ss.kv_per_token]


def selftest(fixture: str = "dec8") -> int:
    """Prove the gate discriminates, on the CPU, before anyone spends a GPU window on it.

    Runs: the clean dry run must pass; each injected bug must fail, and fail *at its own step*.
    A negative that any check would catch is worth nothing (`prototype/mlp_gemv/results.md`), so
    the assertion is about where the failure landed, not that one happened.

    The tie policy gets three cases, because a policy with only a refusal test is indistinguishable from
    the policy that refuses everything: the engine on the pin's OTHER tied id must PASS as a TIE, the
    engine on an untied third id must FAIL, and the engine that keeps the pin's own top-1 while its logit
    sits 1 BF16 ULP off the tied payload must PASS as the ordinary match it is (D-1, the `mt32f` step 11
    shape). All three are aimed at a step the fixture really has a tie at, and all are skipped out loud if
    this fixture has none.

    The route allowance gets one: a step's logits deficit pushed past its own measured route spread
    while still inside the flat §10.1 G5 floor — i.e. a failure the band this allowance replaced could
    not have seen. Skipped out loud on a fixture with no priced allowance (dec8 today). The plain 5 %
    noise case is likewise skipped out loud when a fixture's OWN allowance at that step is wider than
    the noise's deficit, because then the band admits the injection and expecting a FAIL there would be
    a claim about the flat floor, not about the rule in force.

    And the band history gets one: a scratch tree whose `engine/gate.py` loosens `BAND.cos_hard` with no
    line recording it, which must stop the gate at startup — before a fixture is read, a binary started,
    or a device touched. `tools/band_history.py --selftest` covers the checker's own behaviour; this
    case covers the wiring, because a checker nobody calls is the same as no checker.

    The state fingerprints get three, one per §08.6 failure a moment-sum fingerprint can see (a family
    left at the §08.4 zero, a state or window that never moved, KV lanes that moved when they should
    not have), each aimed at a single row and each required to name that row and no other — the
    locality is the assertion, because a negative every check catches proves nothing. They need a
    fixture that carries fingerprints, so `--selftest --fixture dec64f` is the run that exercises
    section (g); on `dec8` the skip is printed. The section's other mechanical case — recomputing a
    pinned fingerprint from pinned tensors with `tools/state_fp.py::fp_row` and requiring bit equality
    — runs once in this process, since it is a fact about the pins and not about an injection.
    """
    base = [sys.executable, str(Path(__file__).resolve()), "--fixture", fixture]
    # The cos band line is NAMED by the rule that produced it, so the needle has to know which rule this
    # fixture is under: with a priced allowance the same injected noise reads as
    # `step 5: cos deficit … inside the route allowance …`. `allow_step` is where the allowance
    # injection will be aimed, or None when there is nothing priced to aim it at.
    allow_step = break_allow_step()
    cos_fail_5 = (f"[FAIL] step 5: cos deficit" if allow_step is not None
                  else f"[FAIL] step 5: cos >= {BAND['cos_hard']} (§10.1 G5)")
    # The 5 % noise injection is sized against the FLAT §10.1 G5 floor, so on a fixture whose OWN
    # allowance at step 5 is wider than what that noise produces it is not a negative at all — the band
    # admits the deficit because two legal reference forms are further apart than that at the step. The
    # honest response is to say so and skip, exactly as `--break-allow` refuses a step where its own
    # injection would not discriminate; the allowance-sized case below asks the same question at the
    # right size for this fixture.
    noise_pin = load_pin("decode5.logits_last.npy").reshape(-1)
    noise_deficit = 1.0 - metrics(relative_noise(noise_pin), noise_pin)["cos"]
    noise_row = allowance_row(5)
    noise_cap = None if noise_row is None else float(noise_row["allowance_one_cos"])
    noise_caught = noise_cap is None or noise_deficit > noise_cap
    cases = [
        ([], 0, None, "clean dry run"),
        (["--break-ids", "3"], 1, "[FAIL] generation step 3 samples greedy_continuation[4]",
         "wrong id injected at step 3"),
        (["--break-logits", "2"], 1, "[FAIL] step 2: argmax agrees with the pin",
         "top two logits swapped at step 2"),
        # The bug the engine actually shipped with, replayed in the stand-in: id 248046 sampled at
        # prompt index 6 ended a 17-token prompt after 7 steps. Both checks have to name it.
        #
        # It is aimed with `--prefill-logits all` on purpose. Since P2 the prompt is a wave and §06.5's
        # default prices only its last row, so row 6 has no argmax at all and there is nothing at that
        # position to fire a stop check on — `dry_run` refuses the injection rather than letting it pass
        # vacuously (the vacuous-negative trap `prototype/mlp_gemv/results.md` names). In the `all` shape
        # every row is argmaxed, the bug is representable again, and this is where the regression that
        # actually happened is caught. The `last` shape gets its own case below, aimed at the one row it
        # does price.
        (["--break-prompt-stop", "6", "--prefill-logits", PREFILL_LOGITS_ALL], 1,
         ["[FAIL] no prompt-phase step fires §07.5's stop check",
          "[FAIL] one step per prompt token",
          "that last step's §07.5 stop check HAD fired"],
         "the stop check fired at prompt index 6 (the first device run's bug: a 17-token prompt"
         " ended after 7 steps and generation never started), in the row shape where an intermediate"
         " prompt row is argmaxed at all"),
        # The forced driver's own failure mode: an off-by-one in the replay list. Everything else
        # can stay perfect, so only the fed-token check can catch it.
        (["--break-feed", "4"], 1,
         "[FAIL] the tokens the engine fed itself are the ones the reference fed",
         "generation step 4 was fed the wrong token (a replay list off by one)"),
    ]
    if noise_caught:
        size = ("" if noise_cap is None else
                f" — deficit {noise_deficit:.3e} over this step's own cap {noise_cap:.3e}")
        cases.append((["--break-noise", "5"], 1, cos_fail_5,
                      "5 % noise on step 5's logits (argmax untouched, aggregate wrong) — must FAIL"
                      f" under the rule in force{size}"))
    else:
        print(f"  [SKIP] 5 % noise at step 5: it produces a deficit of {noise_deficit:.3e}, inside this"
              f" fixture's own allowance at that step ({noise_cap:.3e}), so the band legitimately admits"
              " it — two legal reference routes are further apart than that there. `--break-allow` asks"
              " the same question at this fixture's own size.")
    # The tie policy, both directions, at the first step this fixture actually has a tie at — an
    # injection aimed at a step that is not tied would be caught by every check no matter what the
    # policy does, which is the vacuous-negative trap `prototype/mlp_gemv/results.md` names. dec8 has
    # no tied step; dec64f has two (17, 18), so `--selftest --fixture dec64f` is the one that exercises
    # this pair.
    man = json.loads((REFS / "manifest.json").read_text())
    tie_step = next((j for j in range(len(man["greedy_continuation"]) - 1) if pin_tie(j)), None)
    # §07.5 under the DEFAULT row shape. The case above needs `all` because an unpriced row cannot fire a
    # stop check, but the default wave prices exactly one prompt row — the last — and that row's sample IS
    # §06.5's first answer token, so a stop check armed there would end a run before generation began.
    # This is the case that proves the rule is still testable without `all`, so the negative did not go
    # extinct when the intermediate argmaxes did.
    cases.append((
        ["--break-prompt-stop", str(len(man["sources"][0]["ids"]) - 1)], 1,
        ["[FAIL] no prompt-phase step fires §07.5's stop check",
         "[FAIL] the prompt's last token samples greedy_continuation[0]"],
        "the stop check fired on the wave's last row — the only row `--prefill-logits last` prices,"
        " and the one whose sample §06.5 turns into the first generated token; §07.5 still says a"
        " prompt-phase match does not end the run"))
    if tie_step is None:
        print(f"  [SKIP] tie injections: no exact tie anywhere in {fixture}'s pinned logits."
              " Run `--selftest --fixture dec64f` (ties at 17 and 18) or"
              " `--selftest --fixture mt32f` (ties at 11) to exercise the tie policy.")
    else:
        cases += [
            (["--tie-alt", str(tie_step)], 0,
             f"TIE step {tie_step}",
             f"engine sampled the pin's other tied id at tied step {tie_step} — must be ADMITTED"
             f" (the id is {tie_other(tie_step)}, the pinned top-1 is {pin_tie(tie_step)['top1']})"),
            (["--break-tie", str(tie_step)], 1,
             [f"[FAIL] generation step {tie_step}: exact tie in the pin",
              "is NOT one of the tied ids"],
             f"engine sampled an UNTIED id at tied step {tie_step} — must fail loudly"),
            # D-1's regression. The old predicate consulted the tie rule whenever the PIN was tied, so a
            # step where the engine reproduced the pin's own id could still go red on a half-ULP payload
            # question (`mt32f` step 11: engine 24.391083 rounds 0x41c3, pin 0x41c2, both ids 4016).
            # This injection reproduces that shape and requires GREEN.
            (["--tie-same-ulp", str(tie_step)], 0,
             "the tie policy is not consulted",
             f"engine kept the pin's own top-1 {pin_tie(tie_step)['top1']} at tied step {tie_step} with"
             " its logit one BF16 ULP above the tied payload (the mt32f step 11 shape) — must PASS as"
             " an ordinary id match, rule not consulted"),
        ]
    # The route allowance has to still bite, and bite where the old flat band could not: noise sized to
    # exceed this step's own measured spread while staying inside §10.1's flat 5e-4 deficit. Aimed at the
    # tightest priced allowance in the fixture (the step where the flat floor was most nearly blind), and
    # skipped out loud if this fixture has no priced allowance or no such step — a negative every band
    # would catch is the vacuous one `prototype/mlp_gemv/results.md` names.
    if allow_step is None:
        print(f"  [SKIP] route-allowance injection: {REFS.name} carries no valid"
              f" {ALLOWANCE_FILE}, or no step whose allowance leaves room for a deficit under the flat"
              " §10.1 G5 floor. Price it with engine/route_allowance.py (CPU, no device) to exercise"
              " this case.")
    else:
        cases += [
            (["--break-allow", str(allow_step)], 1,
             [f"[FAIL] step {allow_step}: cos deficit", "would have PASSED this step"],
             f"step {allow_step}'s logits deficit pushed to {SELFTEST_ALLOW_FACTOR}x its own measured"
             " route allowance while still inside the flat G5 floor the old band used — must FAIL there"
             " (this is the failure class the allowance exists to make visible)"),
        ]
    # The state fingerprints get three injections, one per §08.6 failure a moment-sum fingerprint can
    # see, each aimed at ONE row and required to name only that row. They need a fixture that carries
    # fingerprints — `dec8` predates them — so on the default fixture this is skipped out loud rather
    # than quietly passing.
    if prefill_fp("state") is None:
        print(f"  [SKIP] state-fingerprint injections: {fixture} carries no state fingerprints at"
              " all (dec8 was pinned before there were any), so section (g) has nothing to perturb."
              " Run `--selftest --fixture dec64f` (or dec64t / mt32f) to exercise it.")
    else:
        cases += [
            (["--break-state-fp", "zero:state:5"], 1,
             ["[FAIL] §08.4: no state family is still all zero at the prefill end",
              "offending: [state row 5]",
              "![state row 4]", "![state row 6]", "![conv row 5]", "![kv row 5]"],
             "GDN state row 5 zeroed at both end conditions — the §08.4 zeroed start, left behind"
             " (and at the run end, a state that never moved: the same row named twice and no other"
             " row named)"),
            (["--break-state-fp", "frozen:conv:2"], 1,
             ["[FAIL] inv 2: every conv row moved", "offending: [conv row 2]",
              "![conv row 1]", "![conv row 3]", "![state row 2]"],
             "conv window row 2 did not roll across the whole run (§08.6 inv 2, the A-04 carry) —"
             " must fail at row 2 and nowhere else"),
            (["--break-state-fp", "rewrote:kv:1"], 1,
             ["[FAIL] inv 3: the KV's already-written lanes did not move",
              "offending: [kv row 1 dn2 1]",
              "![kv row 0", "![kv row 2"],
             "KV row 1's already-written lanes moved (§08.6 inv 3, append-only storage) — the one"
             " invariant whose correct value is exactly 0.0, so it needs no tolerance to violate"),
        ]
    bad = []
    for extra, want_rc, want_in, what in cases:
        p = subprocess.run(base + ["--dry-run"] + extra, capture_output=True, text=True)
        needles = [] if want_in is None else ([want_in] if isinstance(want_in, str) else list(want_in))
        missing = []
        for needle in needles:
            # A needle starting with "!" must be ABSENT. That is what makes an injection local: "the
            # gate failed" is cheap — a broken parser fails everything — while "the gate failed at
            # exactly the row I broke" is the property worth requiring. It is the trap
            # `prototype/mlp_gemv/results.md` names: a negative every check catches proves nothing.
            absent = needle.startswith("!")
            hit = needle.lstrip("!") in p.stdout
            if hit == absent:
                missing.append(needle)
        ok = p.returncode == want_rc and not missing
        where = "" if not needles else (" — every expected FAIL line present" if not missing
                                        else " — missing: " + "; ".join(missing))
        print(f"  [{'PASS' if ok else 'FAIL'}] {what} — rc={p.returncode} want {want_rc}{where}")
        if not ok:
            print("      " + "\n      ".join(l for l in p.stdout.splitlines() if "FAIL" in l)[:400])
            bad.append(what)
    # The fingerprint-identity case is a property of the pins and the shared helper rather than of an
    # injection, and it is the only thing in this file that imports torch — so it runs once here, in
    # this process, instead of once per dry-run subprocess. A fixture with no fingerprint pins has
    # nothing to prove it with, and says so in the SKIP line above.
    if prefill_fp("state") is not None:
        section("(g) fp-of-pins identity, run once in the selftest process")
        before = len(fails)
        fp_helper_identity(man)
        if len(fails) > before:
            bad.append("fp-of-pins identity (section (g)'s mechanical case): " + fails[-1][:160])
    # The band history has to bite here too, and bite before anything runs: an unrecorded edit to `BAND`
    # must stop the gate, not merely look suspicious in review. The injection is the real thing — a
    # scratch tree whose engine/gate.py carries a looser flat floor, sitting beside the shipped ledger —
    # so this exercises the startup check in the real startup path rather than a stand-in for it
    # (`tools/band_history.py --selftest` is where the checker's own cases live).
    with tempfile.TemporaryDirectory(prefix="gate_bandhist_") as td:
        tdp = Path(td)
        for entry in ROOT.iterdir():
            if entry.name not in ("engine", "bench", ".git"):
                os.symlink(entry.resolve(), tdp / entry.name)
        (tdp / "bench").mkdir()
        for entry in (ROOT / "bench").iterdir():
            if entry.name != Path(band_history.HISTORY_REL).name:
                os.symlink(entry, tdp / "bench" / entry.name)
        shutil.copyfile(ROOT / band_history.HISTORY_REL, tdp / band_history.HISTORY_REL)
        (tdp / "engine").mkdir()
        for entry in (ROOT / "engine").iterdir():
            if entry.name != "gate.py":
                os.symlink(entry, tdp / "engine" / entry.name)
        # Loosen the flat floor — a cos FLOOR going down is the band widening — and refuse to run the
        # case if the edit found nothing to edit, so it cannot pass by editing nothing.
        src = (ROOT / "engine" / "gate.py").read_text()
        hit = re.search(r'^(\s*"cos_hard":\s*)([0-9.eE+-]+)', src, re.M)
        injected = False
        if hit:
            looser = format(float(hit.group(2)) - 0.0001, "g")
            (tdp / "engine" / "gate.py").write_text(
                src[:hit.start()] + hit.group(1) + looser + src[hit.end():])
            injected = True
        p = subprocess.run([sys.executable, str(tdp / "engine" / "gate.py"),
                            "--fixture", fixture, "--no-engine"], capture_output=True, text=True)
        needles = ["BAND HISTORY FAIL", "BAND.cos_hard", "LOOSER than recorded",
                   "band_history.py record"]
        missing = [n for n in needles if n not in p.stdout]
        ok = injected and p.returncode == 1 and not missing
        print(f"  [{'PASS' if ok else 'FAIL'}] an unrecorded band edit stops the gate before any"
              " fixture read or device work — rc=" + str(p.returncode) + " want 1"
              + ("" if not missing else " — missing: " + "; ".join(missing))
              + ("" if injected else " — the injection could not find BAND.cos_hard to loosen"))
        if not ok:
            print("      " + "\n      ".join(p.stdout.splitlines()[:6])[:400])
            bad.append("unrecorded band edit must stop the gate")
    print("SELFTEST " + ("FAIL — " + "; ".join(bad) if bad else
                        "PASS (the gate passes clean input and catches each injected bug at its"
                        " injected step; this says nothing about the engine)"))
    return 1 if bad else 0


# ---- the smoke run ---------------------------------------------------------------------------
def smoke_run(n_steps: int, args) -> int:
    """`--smoke N`: a FREE run of N generated tokens. Not a parity claim, and not a timing.

    What "smoke" has to mean to be worth a device window: it did not crash, no step produced a
    NaN or an out-of-range id, the state stayed inside what was allocated, and the run ended by
    its length budget or by the stop id — one of those two, not anything else. The forced pins
    own depth and parity; this is the question they cannot ask, because a teacher-forced chain
    never lets the engine's own divergence happen and a long context is exactly where a
    carry-over bug that only shows up past the pin lives.
    """
    man = json.loads((REFS / "manifest.json").read_text())
    prompt_ids = man["sources"][0]["ids"]
    vocab = man["text_config"]["vocab_size"]
    ceiling = lane_ceiling()
    section(f"(smoke) free run of {n_steps} generated tokens on {REFS.name}'s prompt")
    lanes = len(prompt_ids) + n_steps
    check(lanes <= ceiling, f"the planned run fits the attention stage's lane ceiling",
          f"{len(prompt_ids)} prompt + {n_steps} generated = {lanes} lanes vs MAXSC {ceiling}"
          " (`kernels_attn.cpp` reads the ceiling from its own #define and throws rather than"
          " truncating; the split-block §07.3 form that would raise it is a prefill item)")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob("logits*.bin"):
        stale.unlink()
    cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
           "--steps", str(n_steps), "--out", str(work)]
    if args.smoke_no_stop:
        # Without this the run stops where the model decides to, which on this prompt is 38 tokens
        # in — and a 300-token smoke that only grew the state 38 slots would be a 38-token smoke.
        cmd.append("--no-stop")
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        check(False, "decode_run exits 0", f"rc={proc.returncode}: "
              + (proc.stderr.strip().splitlines() or [""])[-1])
        return report()
    check(True, "decode_run exits 0")
    run = parse_run(proc.stdout)
    prompt_steps = [s for s in run["steps"] if s["kind"] == PREFILL_KIND]
    decode_steps = [s for s in run["steps"] if s["kind"] == DECODE_KIND]
    result = run["result"]
    stopped = result.get("stopped") == "1"
    check(len(prompt_steps) == len(prompt_ids), "one step per prompt token",
          f"{len(prompt_steps)} of {len(prompt_ids)}")
    check(not [s for s in prompt_steps if s.get("stop") == "1"],
          "no prompt-phase step fired §07.5", f"{len(prompt_steps)} prompt steps")
    check(len(decode_steps) == n_steps or stopped,
          "the run ended by its length budget or by the stop id, not by anything else",
          f"{len(decode_steps)} generation steps, stopped={stopped}, RESULT={result}")
    if args.smoke_no_stop:
        check(not stopped and len(decode_steps) == n_steps,
              "--no-stop: the run went the full length and §07.5 never fired",
              f"{len(decode_steps)} steps, stopped={stopped}")
        if stopped:
            notes.append("--no-stop was passed but the run stopped anyway: the depth this smoke"
                         " claims was not exercised.")
    if stopped:
        check(decode_steps and int(decode_steps[-1]["sampled"]) == STOP_ID
              and decode_steps[-1].get("stop") == "1",
              "a stopped run stopped ON the stop id, in the generation phase",
              f"last step sampled={decode_steps[-1]['sampled'] if decode_steps else 'none'}"
              f" stop={decode_steps[-1].get('stop') if decode_steps else '-'}")
    ids = [int(s["sampled"]) for s in decode_steps]
    bad = [i for i in ids if not 0 <= i < vocab]
    check(not bad, f"every sampled id is a vocabulary id (0..{vocab - 1})", bad[:6])
    top_bad = [s["top_logit"] for s in decode_steps if not re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", s["top_logit"])]
    check(not top_bad, f"every step printed a finite top logit (nan/inf would land here)",
          top_bad[:4])
    # The full logits vectors are the only place a NaN can hide: a step whose argmax looks sane can
    # still be one element away from an inf. ~1 MB per step, read then deleted.
    nonfinite, missing, worst = [], [], None
    for j in range(len(decode_steps)):
        path = work / f"logits{len(prompt_steps) + j}.bin"
        if not path.exists():
            missing.append(path.name)
            continue
        eng = np.fromfile(path, dtype="<f4")
        if eng.size != vocab:
            missing.append(f"{path.name} has {eng.size} floats")
            continue
        if not np.isfinite(eng).all():
            nonfinite.append(f"step {j}: {int((~np.isfinite(eng)).sum())} non-finite")
        else:
            mx = float(np.max(np.abs(eng)))
            worst = mx if worst is None else max(worst, mx)
    check(not missing, "one full logits dump per generation step", missing[:4])
    check(not nonfinite, "no NaN or infinity in any logits vector", nonfinite[:4])
    print(f"  [info] largest |logit| seen anywhere: {worst if worst is not None else 'n/a'}.")
    cap = int(run["state"].get("cap", -1)) if run["state"] else -1
    prefix = int(result.get("prefix_len", -1))
    check(0 <= prefix <= cap <= 10 ** 9, "the state ended inside its allocation",
          f"prefix_len={prefix} cap={cap} (§08.6 invariant: P must stay within the slots)")
    written = sum(p.stat().st_size for p in work.glob("logits*.bin"))
    for p in work.glob("logits*.bin"):
        p.unlink()
    notes.append(f"smoke run dumped and then deleted {written / (1024 * 1024):.0f} MiB of logits."
                 " decode_run also printed its own ms line: a smoke run's timing is NOT a"
                 " measurement and is deliberately not quoted here (bench/performance-status.md"
                 " §6 owns what a clean window takes).")
    return report()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--fixture", default="dec8",
                    help="which data/refs/<name> to gate against: dec8 (8 free steps, the engine"
                         " feeds itself) or dec64f (64 forced steps, replayed — see the header)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="free-run N generated tokens and check only that nothing broke: no crash,"
                         " no NaN, ids in range, state inside its allocation, ended by length or"
                         " the stop id. No parity claim, no timing.")
    ap.add_argument("--smoke-no-stop", action="store_true",
                    help="with --smoke: leave §07.5 unarmed so the run really reaches N tokens"
                         " (the model stops on this prompt after ~38, so without this a 300-token"
                         " smoke is a 38-token smoke)")
    ap.add_argument("--no-engine", action="store_true",
                    help="fixture pairing + state arithmetic only; starts no binary, touches no device")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every section except the engine itself, against a CPU stand-in that"
                         " replays the pins; exercises this file's parsing and metrics. NOT a result.")
    ap.add_argument("--break-ids", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: make that generation step sample a wrong id, and require"
                         " the gate to catch it there")
    ap.add_argument("--break-logits", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: swap that step's top two logits, and require the gate to"
                         " catch it there")
    ap.add_argument("--break-noise", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: put 5 % relative noise on that step's logits, which moves"
                         " cos under the floor without changing the argmax")
    ap.add_argument("--break-allow", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: size that step's noise against ITS OWN cached route"
                         " allowance — a deficit over the allowance that still fits under the flat"
                         " §10.1 G5 floor, i.e. the failure the old flat band could not see. Refuses a"
                         " step with no priced allowance, so it cannot degrade into the --break-noise"
                         " test")
    ap.add_argument("--break-prompt-stop", type=int, default=-1, metavar="PROMPT_INDEX",
                    help="with --dry-run: fire the §07.5 stop check at that prefill row and end the"
                         " run there, which is the bug the first device run had. Under the default"
                         " --prefill-logits last only the wave's LAST row has an argmax to fire on"
                         " (§06.5), so an earlier index is refused unless --prefill-logits all is")
    ap.add_argument("--break-feed", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: feed that step the wrong token, which is an off-by-one"
                         " in the replay list — the plausible way a forced driver breaks")
    ap.add_argument("--break-tie", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: at a step where the PIN is an exact tie, sample an id that"
                         " is NOT one of the tied ids, and require the gate to fail there. Refuses a"
                         " step that is not tied, so the negative cannot quietly become the"
                         " ordinary wrong-id test")
    ap.add_argument("--tie-alt", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: at a tied step, sample the pin's OTHER tied id — what the"
                         " engine really does at dec64f steps 17/18 — and require the gate to admit"
                         " it as a TIE instead of failing it")
    ap.add_argument("--tie-same-ulp", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: at a step where the PIN is tied, keep the pin's own top-1 id"
                         " (so the ids match) but put the engine's F32 logit there one BF16 ULP above"
                         " the pinned payload — the `mt32f` step 11 shape. The step must PASS: an"
                         " engine that reproduced the pin's id is not invoking the tie exception, so"
                         " the rule must not be consulted (D-1). Refuses an untied step, for the same"
                         " reason as --break-tie")
    ap.add_argument("--selftest", action="store_true",
                    help="re-run this file in subprocesses: clean dry run must pass, each injected"
                         " bug must fail at its own step")
    ap.add_argument("--prefill-logits", choices=[PREFILL_LOGITS_LAST, PREFILL_LOGITS_ALL],
                    default=PREFILL_LOGITS_LAST,
                    help="which logits the prompt wave is asked for: `last` (the default, §06.5 — only"
                         " the wave's last row is argmaxed, so earlier prefill STEP lines carry"
                         " `sampled=-1`) or `all` (every row priced, which is what the wave's"
                         " byte-identity comparison against the pre-P2 per-token prompt loop runs in,"
                         " and the only shape where a prompt-phase argmax exists mid-wave)")
    ap.add_argument("--state-fp", dest="state_fp", action="store_true", default=True,
                    help="ask decode_run for its §08 state fingerprints (the default, because a"
                         " fixture that carries fingerprint pins cannot judge G5's state half"
                         " without them). Accepted explicitly for the commands in the prefill plan's"
                         " §2.6; section (g) reads whatever arrived either way")
    ap.add_argument("--no-state-fp", dest="state_fp", action="store_false",
                    help="do not ask for them: section (g) then fails loudly on a fixture that has"
                         " pins, which is the honest reading of a gate that could not see the state")
    ap.add_argument("--break-state-fp", default="", metavar="WHAT:FAMILY:ROW",
                    help="with --dry-run: corrupt one emitted state-fingerprint row and require"
                         " section (g) to fail at exactly that row. WHAT is `zero` (a state that was"
                         " never written), `frozen` (a state or window that did not move over the"
                         " whole run — §08.6 inv 1/2) or `rewrote` (already-written KV lanes moved"
                         " — inv 3). Refuses an impossible target, so it cannot become a generic"
                         " perturbation test")
    ap.add_argument("--build", action="store_true", help="run engine/build.sh first")
    args = ap.parse_args()

    # (pre) BAND HISTORY — checked before a fixture is read, before a binary is started, before a
    # device is touched. `tools/band_history.py` compares every number in `BAND`, and every priced
    # fixture's route margin, against `bench/band-history.jsonl`; an unrecorded move and this gate
    # refuses to run, because a gate must not be judged by a band nobody agreed to (`AGENTS.md`: bands
    # are derived from measured floors and never loosened to pass). It reads the same small files this
    # gate reads anyway — one `BAND` block and one margin per priced fixture.
    if (rc := band_history.for_gate(ROOT)):
        return rc

    set_fixture(args.fixture)

    if args.selftest:
        return selftest(args.fixture)

    if args.build:
        subprocess.run([str(ROOT / "engine" / "build.sh")], check=True)

    if args.smoke:
        if args.dry_run or args.no_engine:
            raise SystemExit("gate: --smoke runs the engine; it is not a --dry-run or --no-engine"
                             " thing (there is no stand-in for 'nothing crashed')")
        return smoke_run(args.smoke, args)

    man = json.loads((REFS / "manifest.json").read_text())
    prompt_ids, dec_logits, dec_inputs = pairing_checks(man)
    greedy = man["greedy_continuation"]
    n_decode = len(dec_logits)

    if args.no_engine:
        state_checks(None)
        fingerprint_checks(man, n_decode)
        deep_matches_shallow(n_decode)
        state_fp_checks(man, len(prompt_ids), n_decode, 0, None)
        return report()

    section("(run) decode_run — one prefill wave (§06.1), then one decode_step per generated token")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    # Section (b) reads `logits{pos}.bin` by name, so a run that stops early is judged against whatever
    # the PREVIOUS run left behind. The first device run stopped at prompt index 6, so it wrote
    # logits0..6 — and still reported only six failures, all in section (a), which is only possible if
    # section (b) found logits17..24.bin that run cannot have written (its default `--work` is the same
    # directory `--dry-run` and `--selftest` write into). Clear them: (b) must judge the run it is
    # judging, and a missing dump has to read as missing.
    for stale in work.glob("logits*.bin"):
        stale.unlink()
    # A replay above A-18's bound, a free run below it: see FREE_RUN_BOUND's comment. `--replay`
    # also tells `decode_run` not to let §07.5 end the run, because dec64f's own chain samples the
    # stop id at generation step 24 and teacher forcing feeds it straight back in.
    replay = n_decode > FREE_RUN_BOUND
    lanes = len(prompt_ids) + n_decode
    ceiling = lane_ceiling()
    check(lanes <= ceiling, "the planned run fits the attention stage's lane ceiling",
          f"{len(prompt_ids)} prompt + {n_decode} generation = {lanes} lanes vs MAXSC {ceiling}"
          " (read from `prototype/attn_decode/attn_decode_kernels.h`; `kernels_attn.cpp` throws"
          " above it rather than truncating)")
    cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
           "--steps", str(n_decode), "--out", str(work),
           # Named on every run rather than left to the binary's default, so the echoed command says
           # which row shape the dumps were taken in: `last` is §06.5's and the one the fixtures are
           # gated in, `all` is the one the wave's byte-identity comparison against the pre-P2 per-token
           # prompt loop is made from.
           "--prefill-logits", args.prefill_logits]
    if args.state_fp:
        # On by default: a fixture that carries fingerprint pins cannot judge G5's state half without
        # them, and a section that silently did not run is the gap P1 was written to close (§2.6). The
        # cost is two D2Hs of the §08.1 tuple per run — ~26 MB each at 63–70 GB/s, on the correctness
        # path, outside every timed region — which is why plan §2.6 gives it no timing significance.
        cmd.append("--state-fp")
    if replay:
        cmd += ["--replay", ",".join(str(t) for t in greedy[:-1])]
    feeding = ("REPLAY — the reference's own token is fed at every step, because this fixture is"
               f" deeper than A-18's {FREE_RUN_BOUND}-token bound and a self-fed run past that"
               " could not be pinned (a divergence then shows at one step, not in all of them)"
               if replay else
               "SELF-FED — the engine feeds its own sample, which is the stronger claim: the"
               " chain is re-derived rather than replayed")
    print("  feeding: " + feeding)
    if args.dry_run:
        stdout, rc = dry_run(man, prompt_ids, dec_logits, work, args.break_ids,
                             args.break_logits, args.break_noise, args.break_prompt_stop,
                             replay, args.break_feed, args.break_tie, args.tie_alt,
                             args.break_allow, args.tie_same_ulp, args.break_state_fp,
                             args.prefill_logits)
        notes.append("DRY RUN: the gate's own plumbing was exercised against a stand-in that replays"
                     " the pins. NOTHING ENGINE-RELATED WAS MEASURED. It exists so the first device"
                     " run is not also the first time this file parses a STEP line. Section (d) is"
                     " Python comparing with itself in this mode; only a real decode_run makes it a"
                     " cross-check.")
        notes.append(f"prefill row shape in force: --prefill-logits {args.prefill_logits}"
                     + (" — only the wave's last row is argmaxed, so the earlier prefill rows print"
                        " `sampled=-1` (§06.5)" if args.prefill_logits == PREFILL_LOGITS_LAST else
                        " — every prefill row is argmaxed, which is the shape the wave's"
                        " byte-identity comparison runs in"))
        if args.break_ids >= 0:
            notes.append(f"injected: generation step {args.break_ids} sampled the wrong id")
        if args.break_logits >= 0:
            notes.append(f"injected: generation step {args.break_logits} had its top two logits"
                         " swapped")
        if args.break_noise >= 0:
            notes.append(f"injected: generation step {args.break_noise} got 5 % relative noise"
                         " (wrong in aggregate, argmax untouched)")
        if args.break_allow >= 0:
            notes.append(f"injected: generation step {args.break_allow} got noise sized to"
                         f" {SELFTEST_ALLOW_FACTOR}x its own cached route allowance — over the"
                         " allowance, under the flat §10.1 G5 floor")
        if args.break_prompt_stop >= 0:
            notes.append(f"injected: the §07.5 stop check fired at prefill row"
                         f" {args.break_prompt_stop} and ended the run there")
        if args.break_tie >= 0:
            notes.append(f"injected: generation step {args.break_tie} sampled an id that is not one"
                         " of the tied ids the pin offers at that step")
        if args.tie_alt >= 0:
            notes.append(f"injected: generation step {args.tie_alt} sampled the pin's other tied id"
                         " — the case the tie policy exists to admit")
        if args.tie_same_ulp >= 0:
            notes.append(f"injected: generation step {args.tie_same_ulp} kept the pin's own top-1 id"
                         " with its logit one BF16 ULP above the tied payload — an ordinary match, so"
                         " the tie policy must not be consulted (D-1)")
        if args.break_state_fp:
            notes.append(f"injected: state fingerprint {args.break_state_fp} — section (g) must fail"
                         " at exactly that family and row, and nowhere else")
    else:
        print("  $ " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout, rc = proc.stdout, proc.returncode
        for line in proc.stdout.splitlines():
            print("      " + line)
    if rc != 0:
        check(False, "decode_run exits 0", f"rc={rc}")
        return report()
    check(True, "decode_run exits 0")
    run = parse_run(stdout)
    check(bool(run["steps"]), "one STEP line per token", f"{len(run['steps'])} lines")
    if not run["steps"]:
        notes.append("no STEP lines at all: decode_run printed nothing the gate could parse — read"
                     " the raw log above (a load failure or a thrown HIP error exits before any step)")
        return report()

    # ---- (a) sampled ids -----------------------------------------------------------------------
    section("(a) sampled token ids — §03.4 greedy, §10.1 G7 exactness")
    # The prompt phase is `kind=prefill` now (P2's wave entry); the check names keep saying "prompt"
    # because §07.5's rule belongs to the phase, not to whichever loop ran it.
    prompt_steps = [s for s in run["steps"] if s["kind"] == PREFILL_KIND]
    decode_steps = [s for s in run["steps"] if s["kind"] == DECODE_KIND]
    # WHY the prompt came up short, not just that it did. The run log already says what ended it, and
    # reading that log on a device window is exactly the round trip these two lines exist to skip.
    if len(prompt_steps) != len(prompt_ids):
        last = run["steps"][-1] if run["steps"] else {}
        fired = last.get("stop") == "1"
        why = (f"{len(prompt_steps)} of {len(prompt_ids)} prompt tokens fed — the run ended early."
               f" Last STEP: kind={last.get('kind', 'no STEP line at all')}"
               f" index={last.get('index', '-')} sampled={last.get('sampled', '-')}"
               f" stop={last.get('stop', '-')}; RESULT: {run['result'] or 'no RESULT line'} — "
               + ("that last step's §07.5 stop check HAD fired. A stop match during the prompt phase"
                  " is not a stop: the sample is discarded and the next fed token comes from the"
                  f" fixture, and a chat prompt legitimately contains the stop id {STOP_ID} (dec8's"
                  " does, at index 8). `decode_run.cpp` owns the generation-only rule; it is the"
                  " first device run's bug, reproduced here so it cannot come back quietly"
                  if fired else
                  "no stop check on the last step, so this is an early exit or a thrown error rather"
                  " than §07.5 — read the lines above for the HIP message"))
    else:
        why = f"{len(prompt_steps)} vs {len(prompt_ids)}"
    check(len(prompt_steps) == len(prompt_ids), "one step per prompt token", why)
    # The same bug seen from the other side, as a standing tripwire: no prompt-phase step may be the one
    # that fires §07.5. The raw match is NOT the failure here — prompt-phase steps sampling the stop id is
    # expected on a chat prompt and prints `stop=0` with that id in `sampled=`. `stop=` is what
    # decode_run's human-readable [STOP] marker means (same printf condition), so this covers it too.
    prompt_stop = [s for s in prompt_steps if s.get("stop") == "1"]
    check(not prompt_stop, "no prompt-phase step fires §07.5's stop check",
          "" if not prompt_stop else
                  f"kind=prefill index={prompt_stop[0].get('index')}"
          f" sampled={prompt_stop[0].get('sampled')} ended the run; §07.5 arms only during"
          f" generation (and {STOP_ID} inside a prompt is not news — see decode_run.cpp)")
    check(len(decode_steps) == n_decode, f"{n_decode} generation steps", f"{len(decode_steps)}")
    check(int(prompt_steps[-1]["sampled"]) == greedy[0] if prompt_steps else False,
          "the prompt's last token samples greedy_continuation[0]",
          f"{prompt_steps[-1]['sampled'] if prompt_steps else 'no prompt step ran'} vs {greedy[0]}")
    got_ids = [int(s["sampled"]) for s in decode_steps]
    # The engine's logits vectors, read here rather than in (b): the tie policy is proved against what
    # the engine actually computed, so (a) needs them, and a tie that cannot be proved must FAIL at (a)
    # rather than be waived.
    eng_logits = read_engine_logits(work, n_decode, len(prompt_ids), dec_logits[0].size)
    # One tie decision per step, in the pin's payload space (see TIE POLICY). `tie_ok` is what (b)'s
    # argmax line reuses, so the two sections cannot disagree about what a tied step means.
    tie_ok: dict[int, bool] = {}
    for j in range(min(n_decode, len(got_ids))):
        kind, admissible, line = tie_verdict(j, eng_logits[j][0], got_ids[j])
        if kind == "no-tie":
            continue                       # the ordinary case: the pin has a single winner here
        if kind == "match":
            # D-1: engine and pin picked the same id, so G7 is satisfied outright and there is no
            # exception to grant. The line is printed so a reader can still see the pin was tied here,
            # but nothing is armed and nothing is counted as a use of the policy.
            tie_notes.append(f"step {j} -> {got_ids[j]}")
            print("  [info] " + line)
            continue
        ties.append(f"step {j} -> {got_ids[j]}")
        check(admissible,
              f"generation step {j}: exact tie in the pin and the engine picked a different id"
              " — tested against the tie policy", line)
        tie_ok[j] = admissible
    for j, (g, w) in enumerate(zip(got_ids, greedy[1:])):
        by_tie = g != w and tie_ok.get(j, False)
        check(g == w or by_tie, f"generation step {j} samples greedy_continuation[{j + 1}]",
              f"{g} vs {w}" + (" — admitted as a proved tie, see the TIE line above" if by_tie else ""))
    unresolved = [j for j, (g, w) in enumerate(zip(got_ids, greedy[1:]))
                  if g != w and not tie_ok.get(j, False)]
    check(not unresolved,
          f"all {n_decode} generated ids match, in order — exactly, or as a tie proved from the pin",
          f"{len(tie_ok)} step(s) resolved by the tie policy"
          f" ({', '.join(sorted(ties)) if ties else 'none'}); unmatched steps {unresolved}"
          f" | engine {got_ids} / pinned {greedy[1:]}")
    fed = [int(s["input"]) for s in decode_steps]
    check(fed == greedy[:-1],
          "the tokens the engine fed itself are the ones the reference fed (the engine feeds itself,"
          " so this is the chain being re-derived, not replayed)", f"{fed}")
    if replay:
        # Same check, different meaning, so it is said out loud: here the fed list is what THIS gate
        # handed `decode_run --replay`, so it proves the delivery. What proves the model is the
        # `sampled=` values above — the engine's own argmax at each step, compared with the
        # reference's and deliberately not fed back.
        check(not [s for s in decode_steps if s.get("stop") == "1"],
              "no generation step ended a replay run (§07.5 is a free-run rule)",
              f"{len(decode_steps)} replayed steps")
        where = [j for j, s in enumerate(decode_steps) if int(s["sampled"]) == STOP_ID]
        print(f"  [info] the pinned chain samples the stop id {STOP_ID} at generation step(s)"
              f" {where if where else 'none'}; teacher forcing feeds that token straight back in,"
              " so the engine prints it and keeps going — that is why a replay does not arm §07.5.")
    positions = [int(s["pos_before"]) for s in decode_steps]
    check(positions == list(range(len(prompt_ids), len(prompt_ids) + n_decode)),
          f"positions run {len(prompt_ids)}..{len(prompt_ids) + n_decode - 1} (§02.4's P)",
          f"{positions}")
    if stdout.count("[STOP]") and greedy[-1] not in man["special_ids"].values():
        notes.append(f"engine stopped on id {STOP_ID} in the generation phase but the manifest's"
                     " continuation does not")

    # ---- (b) logits ------------------------------------------------------------------------------
    section("(b) logits vs the pinned dumps — the rule in force is named on every step")
    print(f"  where an allowance is priced, the rule is cos deficit <= margin x the step's measured"
          f" route spread and meanrel <= margin_rel x it, and the margins are THIS FIXTURE's own — read"
          f" from its own {ALLOWANCE_FILE} and derived from its own steps only, so pricing another"
          f" fixture cannot move them (D-2; the fallback in BAND is"
          f" {BAND['route_margin']:.2f} / {BAND['route_margin_rel']:.2f}, used only by a file priced"
          f" before per-fixture keying). Where nothing is priced, the flat published floors apply:"
          f" cos >= {BAND['cos_hard']} (G5 incremental==batch), meanrel <="
          f" {BAND['meanrel_hard']:.0%} (§10.2 cascade ceiling).")
    print(f"  reported: cos vs {BAND['cos_report']} (G6 decode logits); ndiff and max ULP are "
          "reported but not gated (§10.1 F1–F4: a chained dot's legal ndiff is 29–47 %, and a "
          "max-ULP bound is blind to a whole-tensor 1-ULP shift)")
    allowance = load_allowance(n_decode)
    per_step: list = [None] * n_decode
    under_g6: list = []
    ratios: list = []            # (engine deficit / that step's route spread), for the summary line
    for j in range(n_decode):
        path = work / f"logits{len(prompt_ids) + j}.bin"
        eng, why = eng_logits[j]
        if eng is None:
            check(False, f"engine wrote {path.name}", why + " — check --out")
            continue
        ref = dec_logits[j]
        if eng.shape != ref.shape:
            check(False, f"step {j}: vocabulary length", f"{eng.shape} vs {ref.shape}")
            continue
        m = metrics(eng, ref)
        deficit = 1.0 - m["cos"]
        gap_x = ("exact" if m["margin"] == float("inf") else f"{m['margin']:.0f}×")
        cos_cap, rel_cap, rule_line = rule_for_step(allowance, j)
        if allowance is not None:
            ratios.append((j, deficit / float(allowance["steps"][j]["route_one_cos"])))
        argmax_ok = argmax_low(eng) == argmax_low(ref)
        fed = int(decode_steps[j]["input"]) if j < len(decode_steps) else int(dec_inputs[j][0])
        sampled = (int(decode_steps[j]["sampled"]) if j < len(decode_steps)
                   else argmax_low(ref))
        per_step[j] = {"cos": m["cos"], "meanrel": m["meanrel"], "ndiff": m["ndiff"],
                       "n": ref.size, "argmax_ok": argmax_ok, "fed": fed, "sampled": sampled}
        # `deficit` is printed at full precision because it is what `engine/route_allowance.py` re-derives
        # the margins from; the 6-decimal `cos` is not enough (at a 2.3e-4 deficit it is ±0.4 %).
        check(True, f"step {j}: metrics",
              f"cos {m['cos']:.6f} deficit {deficit:.6e} meanrel {m['meanrel']:.4%}"
              f" ndiff {m['ndiff']}/{ref.size}"
              f" maxabs {m['maxabs']:.4g} | ref top1−top2 gap {m['gap12']:.3f}"
              f" = {gap_x} the largest element error | rule: {rule_line}")
        if allowance is None:
            check(deficit <= cos_cap, f"step {j}: cos >= {BAND['cos_hard']} (§10.1 G5)")
            check(m["meanrel"] <= rel_cap,
                  f"step {j}: meanrel <= {BAND['meanrel_hard']:.0%} (§10.2)")
        else:
            ok_cos = deficit <= cos_cap
            check(ok_cos,
                  f"step {j}: cos deficit {deficit:.4e} inside the route allowance {cos_cap:.4e}",
                  "" if ok_cos else _counterfactual(deficit, 1.0 - BAND["cos_hard"],
                                                    "§10.1 G5 floor"))
            ok_rel = m["meanrel"] <= rel_cap
            check(ok_rel,
                  f"step {j}: meanrel {m['meanrel']:.4%} inside the route allowance {rel_cap:.4%}",
                  "" if ok_rel else _counterfactual(m["meanrel"], BAND["meanrel_hard"],
                                                    "§10.2 ceiling"))
        check(argmax_ok or tie_ok.get(j, False), f"step {j}: argmax agrees with the pin",
              f"{argmax_low(eng)}" + ("" if argmax_ok else
                                      (" — but the pin is tied here and the engine's id is one of the"
                                       " tied ids, so the tie policy admits it"
                                       if tie_ok.get(j, False) else
                                       " — and the pin is tied but the engine's id is not one of the"
                                       " tied ids")))
        if m["cos"] < BAND["cos_report"]:
            under_g6.append(f"{j}:{m['cos']:.6f}")
    if ratios:
        worst_j, worst_r = max(ratios, key=lambda t: t[1])
        tighter = sum(1 for j in range(n_decode)
                      if allowance["steps"][j]["allowance_one_cos"] <= 1 - BAND["cos_hard"])
        notes.append(f"all {len(ratios)} of {n_decode} steps judged by the route allowance; the"
                     f" allowance is TIGHTER than the flat §10.1 G5 floor at {tighter} of them. Worst"
                     f" step {worst_j}: deficit = {worst_r:.3f}x its own measured route spread"
                     f" (this fixture's margin in force {allowance['margin']:.2f}, derived as the max"
                     f" ratio inside {REFS.name} alone — engine/route_allowance.py; no other fixture"
                     " can move it, and no step of this fixture sits outside it by construction).")
    if under_g6:
        notes.append(f"{len(under_g6)} of {n_decode} steps sit under G6's decode-logits figure "
                     f"{BAND['cos_report']} (step:cos — " + ", ".join(under_g6[:12])
                     + (" …" if len(under_g6) > 12 else "") + "). G6 is ADVISORY here by decision,"
                     " not an open question: an end-to-end token carries the whole 24-layer"
                     " cascade, so it is not judged by a figure belonging to a narrower"
                     " comparison — see the BAND comment above and the two measured floors in"
                     " engine/results.md: last-op reassociation 0.99999862"
                     " (engine/logits_floor.py) and end-to-end two-legal-routes 1-cos"
                     " 4.7e-5..1.9e-4 (engine/route_price.py).")

    state_checks(run["state"])
    curve_checks(per_step, n_decode)
    fingerprint_checks(man, n_decode)
    deep_matches_shallow(n_decode)
    state_fp_checks(man, len(prompt_ids), n_decode, len(decode_steps), run["statefp"],
                    stand_in=args.dry_run)

    # ---- (d) launch plan ------------------------------------------------------------------------
    section("(d) the launch plan — §07's structure, counted in Python")
    cfg = man["text_config"]
    n_layers = cfg["num_hidden_layers"]
    n_gdn = sum(1 for t in cfg["layer_types"] if t == "linear_attention")
    n_attn = sum(1 for t in cfg["layer_types"] if t == "full_attention")
    want = launch_plan(n_layers, n_gdn, n_attn)
    got = {k: int(v) for k, v in run["launch"].items() if k != "total"}
    for k in sorted(want):
        check(got.get(k) == want[k], f"launch count {k}", f"engine {got.get(k)} vs §07 {want[k]}")
    total = int(run["launch"].get("total", -1))
    check(total == sum(want.values()), "launch total", f"{total} vs {sum(want.values())}")
    check(want["glue"] == 49, "glue is the 49 `prototype/glue/run.py` pins its token plan against",
          f"{want['glue']}")
    check(n_gdn + n_attn == n_layers and n_attn == 6,
          "layer routing read from the manifest matches §03.5", f"{n_gdn}+{n_attn} of {n_layers}")
    if "ms_per_step" in run["result"]:
        notes.append(f"decode_run reported {run['result']['ms_per_step']} ms/step. Timing from a"
                     " gate run is a by-product, not a measurement: see bench/performance-status.md"
                     " §6 for what a clean window takes.")
    return report()


def report() -> int:
    section("summary")
    for n in notes:
        print("  [note] " + n)
    if ties:
        print(f"  [TIE] {len(ties)} generation step(s) were exact ties in the pin and were decided by"
              f" the tie policy: " + ", ".join(ties))
    if tie_notes:
        print(f"  [TIE-NOTE] {len(tie_notes)} step(s) had an exact tie in the pin that the engine"
              " matched outright — ordinary id matches, tie policy not consulted (D-1):"
              " " + ", ".join(tie_notes))
    if fails:
        print(f"ENGINE GATE FAIL — {len(fails)} failed"
              + (f" ({len(ties)} step(s) reported TIE and not counted as failures)" if ties else ""))
        for f in fails:
            print("  - " + f)
        return 1
    print("ENGINE GATE PASS" + (f" — with {len(ties)} step(s) reported TIE" if ties else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
