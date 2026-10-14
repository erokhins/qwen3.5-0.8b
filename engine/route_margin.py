#!/usr/bin/env python3
"""D-3 / Q4b: what is a legal route worth, measured with NO engine in the numerator?

    # one fixture, its own gate log for the engine column, three alternative §04.7 chunk widths
    .venv/bin/python engine/route_margin.py --fixture dec64f --engine-log progress/engine/evidence/d6_gate_dec64f.log \
            --chunk-alt 8,16,32,128 --out progress/engine/evidence/route_margin_dec64f.json
    .venv/bin/python engine/route_margin.py --selftest          # synthetic: no weights, no device

WHY THIS EXISTS. Every margin `engine/gate.py` applies is `max over steps of engine_deficit(j) /
route_spread(j)` — the numerator is the engine's own error, so the worst measured step sits on the
boundary by construction and each new fixture asks for a bigger number (3.50, 7.99, 2.93). D-3 asks for
a multiplier that stands without reference to the engine: price TWO independent legal spreads on the
same fixture and compare them. `engine/route_price.py` prices one pair (a vs b, below); this script adds
a third legal route and prices it against the same two, which yields a ratio in which the engine appears
nowhere.

THE ROUTES. All of them are `speccheck/specimpl.py` with the same weights and the same forced token list;
the only thing that differs is how the prompt gets into the GDN state, because `specimpl.gdn` takes the
chunked branch only when T > 1 and every forced decode step is T = 1.

  a  chunked §04.7 at the shipped width, `C = specimpl.GDN_CHUNK = 64`. This is the pin's own family:
     §04.7 names the chunked form as HF-ref's prefill path `[REF]`, and `deficit()` measures every other
     route against a, i.e. against the route the dump came from.
  b  recurrent §04.4, the token loop — the form the engine is forced into today, since it has no prefill
     and feeds the prompt one token per decode step. `route(a,b)` is what `route_price.py` caches in
     `data/refs/<fixture>/route_allowance.json`, so `def_b` below IS the priced spread.
  c… chunked §04.7 at a DIFFERENT width (`--chunk-alt`). Legality is §04.7 verbatim: "Any prefill
     algorithm is correct iff, for every t, it returns the o[t] and final S of §04.4 applied
     token-by-token … vLLM-ref and the engine **may differ in chunk size** or fusion strategy." §10 G3 /
     `speccheck/gates/run_chunkcheck.py` is the gate that licenses that sentence (chunk vs recurrent
     >= 0.99999 per layer, both equidistant from the reference; A-08 closed). Nothing is perturbed,
     dropped or re-typed: the same F32 UT-transform computes the same recurrence with a different block
     width, i.e. a different association of the same sums — `prototype/mlp_gemv/floor_probe.py`'s
     reassociation argument (§10.1 F1-F4: legal, and sized by a probe rather than by taste) applied to
     the whole prompt->state map instead of to one dot. It is not a knob, because §04.7 names chunk size
     as an implementation choice and G3 already proves every width reproduces §04.4 token-for-token.
     Several widths are priced per fixture precisely so the answer cannot be one width's artifact.
  d  chunked §04.7 at the shipped width with §07.3's BLOCKED `(max, sum)` decode softmax instead of the
     one-pass softmax — a second, orthogonal legal freedom. Legal by §07.3 verbatim: "The blocked form is
     not banned — it is the shape a long-context path has to take (W7 split-K / flash-decoding …), and it
     stays available in the oracle (`specimpl.attention(softmax="online")`) … When that path is built it
     must be gated on a cos/meanrel band", with the measured band in A-17 (26 % of `attn_out_pre_gate`
     elements differ, cos >= 0.99999, meanrel <= 1.8e-3). Two things follow and both are respected here:
     it is a legitimate thing to price — it is what the 4k context target will require — AND the shipped
     engine does NOT take it (§07.3's decision is the one-pass form, because that one is bit-exact), so
     its spread is reported on the `M_wide` line and kept out of the conservative `M_1`.
  e  recurrent §04.4 with the blocked softmax: both freedoms at once, i.e. the widest legal point from
     route a that this oracle can construct. Included because the engine's deficit is a composition of
     many legal choices, not one, so a single-axis spread understates the legal envelope by construction.

A WIDTH THAT DOES NOT REPARTITION THE PROMPT IS NOT A THIRD ROUTE, and the script says so out loud
instead of pricing it. `gdn_chunk_scan` right-pads to a multiple of C, so on a 17-token prompt C=32 and
C=64 are the same single-chunk computation and come back bit-identical (measured: deficit 0.0 at every
step). Such a width is reported INERT and excluded from the multiplier — silently averaging a run of
exact zeros into a max would understate the multiplier, which is a loosening.

WHY THE WIDTH IS REBOUND RATHER THAN PLUMBED. `specimpl.gdn` calls `gdn_chunk_scan(qn, kn, v, g, beta,
S)` and leaves `C` at its default, so a second width would otherwise mean editing `speccheck/specimpl.py`.
This script refuses to: `engine/gate.py:allowance_inputs` pins specimpl's sha256 into every priced
fixture, so an oracle edit would invalidate all four cached allowances at once (and the oracle is not
this task's object). The rebind lives in this process only, is counted so a silent no-op is impossible,
and `--check-patch` asserts it reproduces route a BIT-EXACTLY when asked for the shipped width — the
falsifiable half of "the rebind is the only thing that moved".

WHAT COMES OUT. Every legal deficit is measured against route a, i.e. against the pin's route, so it is
the same quantity the gate compares the engine against. `def_x(j)` is route x's 1-cos (or meanrel)
against route a at step j, and `def_b` is the priced spread:

    fitted   = max_j engine_def(j) / def_b(j)     <- today's basis: the engine in the numerator
    M_1(alt) = max_j def_c(j)      / def_b(j)     <- same statistic, a legal route in the engine's seat
    M_wide   = max_x M_1(x) over every axis priced, §07.3 included
    M_env    = max_j max_x def_x(j) / def_b(j)    <- widest legal deficit the priced route set can make

`fitted`, `M_1` and `M_env` are one functional with three different numerators, so comparing them is
arithmetic rather than argument. The statistic is a MAX because the rule it calibrates is a per-step cap
every step must clear: a p95 multiplier would leave the p95+1 step red by construction, and D-2's whole
point was that a margin is the tightest constant that admits EVERY measured step. p95 and median are
printed for shape — so a reader can see whether the max is one outlier or the whole tail — and are not
offered as the constant.

THREE COLUMNS, because "meanrel" means two different things on the two sides of the shipped rule. The
engine's meanrel comes from `engine/gate.py:metrics`, which is §10.2's DEFINITION (mean|Δ| / mean|ref|, a
ratio of means — gate.py's own BAND comment rejects the other form as exploding on the near-zero logits
that most of a 248 320-wide vocabulary is). The route side, `route_price.meanrel`, IS that mean of ratios.
Nothing is wrong with the gate's arithmetic — it derives and applies the same two quantities — but the
constant `margin_rel` is then a ratio of two different statistics and cannot be read as "the engine's
relative error is N× the legal spread". So this script prints:

    one_cos             engine 1-cos   / route 1-cos          (unambiguous, the primary column)
    meanrel_shipped     engine rom     / route mean-of-ratios (the mixture the shipped constant uses, so
                                                             the number here IS the number in force)
    meanrel_consistent  engine rom     / route rom            (both sides on §10.2; the only version that
                                                             means what it appears to say)

THE BASELINE, which is what the comparison actually hinges on. The engine's deficit is measured against the
PINNED HF dump, while every legal route's deficit above is measured against specimpl's route a. Different
references. So `oracle_vs_pin = max_j def(pin_j vs route a_j) / def_b(j)` is priced too: an engine-
independent quantity against the SAME dump the engine is judged on. If it is the same size as `fitted`, the
margin is measuring specimpl-vs-HF rather than the engine, and no pair of specimpl-internal routes can ever
produce that term — which is a statement about the rule's shape, not about anybody's kernel.

WHAT THIS SCRIPT MAY NOT DO. It writes nothing under `data/refs/`, imports neither
`engine/route_allowance.py` nor `tools/band_history.py`, and moves no band: a multiplier is a number for
the operator to decide on, not an edit. If the multiplier is SMALLER than the margin in force that is a
finding about the engine and the only honest action is to report it — `verdict()` encodes that as `keep`,
and `--selftest` case 4 fails if anyone makes it possible to adopt a smaller multiplier as a way of
turning the gate green.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speccheck"))
sys.path.insert(0, str(ROOT))
import specimpl as si                                    # the module object: rebound, never edited  # noqa: E402
from specimpl import Forward, Weights                    # noqa: E402
from engine import gate as egate                         # noqa: E402
from engine import route_price as rp                     # noqa: E402

torch.set_grad_enabled(False)

SHIPPED_WIDTH = si.GDN_CHUNK          # 64, §04.7's reference default
PRICED = "b_recurrent"                # route b: the spread the shipped allowances already key on


# ---- running one route ------------------------------------------------------------------------------

@contextlib.contextmanager
def chunk_width(c: int):
    """Run the oracle's own chunked GDN at width `c`. Counts calls so "the patch did not apply" cannot be
    misread as "the two routes agree". A default-argument tweak is not enough (`C=GDN_CHUNK` is bound at
    def time), so the module attribute is swapped and restored."""
    original = si.gdn_chunk_scan
    seen: list[int] = []

    def patched(qn, kn, v, g, beta, S=None):
        seen.append(c)
        return original(qn, kn, v, g, beta, S, C=c)

    si.gdn_chunk_scan = patched
    try:
        yield seen
    finally:
        si.gdn_chunk_scan = original


def run_route(w: Weights, prompt: list[int], feed: list[int], form: str, width: int | None,
              attn: str = "full") -> tuple[list, int]:
    """One legal prompt route over the fixture's own forced chain: (logits per step, chunked calls)."""
    if width is None:
        f = Forward(w, gdn_form=form, attn_softmax=attn)
        f.prefill(torch.tensor(prompt))
        return [f.logits(f.decode(tok))[-1] for tok in feed], 0
    with chunk_width(width) as seen:
        f = Forward(w, gdn_form=form, attn_softmax=attn)
        f.prefill(torch.tensor(prompt))
        logits = [f.logits(f.decode(tok))[-1] for tok in feed]
        calls = len(seen)
    # Anti-vacuity, the shape A-17's gate uses: if the chunked branch never ran, this "third route" is
    # route a wearing a hat and every ratio against it is a silent 1.0.
    # Raised, not asserted: this is the falsifiable half of "the rebind applied", and a check that a
    # `python -O` invocation can delete is not a check. Same reason in `ratios()` below.
    if calls <= 0:
        raise SystemExit(f"route_margin: the chunked GDN ran {calls} times at width {width}: the rebind"
                         " did not apply, so this route is route a wearing a hat")
    return logits, calls


def n_chunks(n_prompt: int, width: int) -> int:
    """How many chunks §04.7 splits the prompt into at this width (right-padded to a multiple of C).
    Printed because this, not the width itself, is what changes the arithmetic: two widths that both fit
    the prompt in one chunk are the same computation."""
    return -(-n_prompt // width)


def route_set(chunk_alts: list[int], n_prompt: int, attn_axis: bool = True) -> list[dict]:
    """a (shipped width) and b (recurrent) always; one c per alternative §04.7 width; and, if asked, the
    two routes that exercise §07.3's second legal freedom (the blocked `(max, sum)` decode softmax),
    alone and composed with the recurrent GDN form."""
    routes = [{"name": "a_chunk64", "form": "chunk", "width": None, "attn": "full", "axis": "—",
               "chunks": n_chunks(n_prompt, SHIPPED_WIDTH)},
              {"name": PRICED, "form": "recurrent", "width": None, "attn": "full", "axis": "priced",
               "chunks": n_prompt}]
    for c in chunk_alts:
        routes.append({"name": f"c_chunk{c}", "form": "chunk", "width": c, "attn": "full",
                       "axis": "§04.7 width", "chunks": n_chunks(n_prompt, c)})
    if attn_axis:
        routes.append({"name": "d_attn_online", "form": "chunk", "width": None, "attn": "online",
                       "axis": "§07.3 softmax", "chunks": n_chunks(n_prompt, SHIPPED_WIDTH)})
        routes.append({"name": "e_both", "form": "recurrent", "width": None, "attn": "online",
                       "axis": "both", "chunks": n_prompt})
    return routes


def meanrel_rom(a, b) -> float:
    """§10.2's definition, which `engine/gate.py:metrics` uses: mean|Δ| / mean|ref|, a RATIO OF MEANS.
    gate.py's own BAND comment says the mean-of-ratios form "explodes on the near-zero logits that most
    of a 248 320-wide vocabulary is" — which is exactly what `route_price.meanrel` is, and the reason
    this script prices both (see COLUMNS in the docstring)."""
    a64, b64 = rp.f64(a).ravel(), rp.f64(b).ravel()
    return float(((a64 - b64).abs().mean() / b64.abs().mean()).item())


def deficit(run: list, ref_a: list, n: int) -> list[dict]:
    """Route x's deviation from ROUTE A at each step — the same comparison the gate makes against the pin,
    because the pins came from the chunked family (§04.7 [REF]). `a` is the denominator of both meanrels
    for the same reason `route_price.price()` keeps the chunked run there."""
    return [{"step": j,
             "one_cos": 1.0 - rp.cos(run[j], ref_a[j]),
             "meanrel_rom": meanrel_rom(run[j], ref_a[j]),
             "meanrel_mor": rp.meanrel(run[j], ref_a[j]),
             "argmax": int(torch.argmax(run[j]))} for j in range(n)]


# The columns to price. Each is (label, engine field, route field), and the third one is the point:
# `engine/gate.py` prints the ENGINE's meanrel as a ratio of means (§10.2), while `route_price.py`'s
# route_meanrel — the number every cached `margin_rel` divides by — is a mean of ratios. So the shipped
# constant is a ratio of two DIFFERENT statistics. `meanrel_shipped` re-derives it on the shipped
# mixture so the number in the table is the number in force; `meanrel_consistent` puts both sides on
# §10.2's definition, which is the only version that means what it appears to say.
COLUMNS = (("one_cos", "one_cos", "one_cos"),
           ("meanrel_shipped", "meanrel", "meanrel_mor"),
           ("meanrel_consistent", "meanrel", "meanrel_rom"))


# ---- statistics --------------------------------------------------------------------------------------

def summarize(pairs: list[tuple[float, int]]) -> dict:
    """max (+ its step), p95, median over per-step ratios. max is the number a cap needs; the others are
    shape, so a reader can see whether the max is a single outlier or the whole tail."""
    vals = np.array([v for v, _ in pairs], dtype=np.float64)
    worst_v, worst_step = max(pairs, key=lambda p: p[0])
    return {"max": float(worst_v), "max_step": int(worst_step),
            "p95": float(np.percentile(vals, 95)), "median": float(np.median(vals)),
            "n": int(vals.size)}


def ratios(numer: list[dict], denom: list[dict], field: str) -> list[tuple[float, int]]:
    """Per-step ratio of two same-step quantities. A zero denominator is a bug in the caller, not
    something to paper over: it is a step where two legal routes agreed exactly, and there is no ratio
    there — dropping it would quietly shrink a max."""
    out = []
    for n_row, d_row in zip(numer, denom):
        if n_row["step"] != d_row["step"]:
            raise SystemExit("route_margin: tried to take a ratio between two different steps")
        if d_row[field] <= 0:
            raise SystemExit(f"route_margin: step {n_row['step']} has {field} {d_row[field]!r} in the"
                             " denominator. A step where two legal routes agree exactly cannot carry a"
                             " ratio; stop and look at it rather than skipping it.")
        out.append((n_row[field] / d_row[field], n_row["step"]))
    return out


def ceil2(x: float) -> float:
    """Same rule as `engine/route_allowance.py::ceil2` — up at two decimals, so a re-derived constant is
    never quoted more permissive than the measurement behind it."""
    return math.ceil(round(x, 6) * 100) / 100.0


def verdict(inforce: float, fitted: float, lo: float, hi: float) -> dict:
    """What the operator may do with the route multiplier, given the margin actually in force.

    Several legal widths give several multipliers, so two readings are taken and they are deliberately
    asymmetric in the safe direction:

      * to conclude THAT THE ENGINE EXCEEDS legal routes, use `hi` — the most generous legal estimate.
        If even the widest legal choice falls short, the finding does not depend on which width was
        picked.
      * to conclude that the basis MAY CHANGE, use `lo` — the most conservative estimate. Adopting on
        `hi` would let the choice of width set a band, which is the knob D-3 exists to refuse.

    Pure, so `--selftest` can prove the anti-loosening branches cannot be lost: adopting a multiplier
    SMALLER than the one in force is never returned as an action, and neither is one larger.
    """
    if hi <= 0:
        return {"action": "no multiplier",
                "why": "every alternative route came back bit-identical to route a, so this fixture has"
                       " no second legal spread to price at all (see INERT above)"}
    if hi < fitted:
        return {"action": "keep",
                "why": f"the engine needs {fitted:.4f}; the widest legal alternative route reaches only"
                       f" {hi:.4f}, so the engine sits {fitted / hi:.2f}x outside what a legal prompt"
                       " route produces at that step — and the most conservative width gives"\
                       f" {lo:.4f}. That is a finding ABOUT THE ENGINE, not a licence to loosen: do"
                       " not touch a band or re-price a fixture on the strength of this number"}
    if lo >= fitted:
        if lo <= inforce + 1e-12:
            return {"action": "adopt-with-changed-basis",
                    "why": f"every priced legal route explains the engine's worst step (lo = {lo:.4f} >="
                           f" fitted {fitted:.4f}) and the conservative multiplier is no looser than"
                           f" the {inforce:.2f} in force, so the constant could be restated with no"
                           " engine in its derivation — a band move either way, so it still needs a"
                           " recorded decision and a band_history line that this script cannot write"}
        return {"action": "keep (adopting would LOOSEN)",
                "why": f"legal routes explain the engine, but even the conservative multiplier"
                       f" {lo:.4f} is looser than the {inforce:.2f} in force; adopting it is a band"
                       " move, so it needs a recorded decision in progress/2026-10-12/plan-2026-10-12.md §3 and a"
                       " tools/band_history.py line — neither of which this script may make"}
    return {"action": "keep (width-sensitive)",
            "why": f"the multiplier straddles what the engine needs (lo {lo:.4f} < fitted {fitted:.4f}"
                   f" <= hi {hi:.4f}), i.e. the answer depends on which legal width was priced, which"
                   " is a knob rather than a calibration. Keep the margin in force and say the"
                   " engine-independent basis did not settle it"}


# ---- one fixture -------------------------------------------------------------------------------------

def price_fixture(w: Weights, fixture: str, engine_log: Path | None, chunk_alts: list[int],
                  max_steps: int = 0, check_patch: bool = False, attn_axis: bool = True) -> dict:
    egate.set_fixture(fixture)
    man = json.loads((egate.REFS / "manifest.json").read_text())
    prompt, chain = man["prefill_ids"], man["greedy_continuation"]
    feed = chain[:-1]                                   # exactly what gate.py hands `decode_run --replay`
    if max_steps:
        feed = feed[:max_steps]
    n, pin_ids = len(feed), chain[1:]
    routes = route_set(chunk_alts, len(prompt), attn_axis)

    print(f"\n== {fixture}: prompt {len(prompt)} tokens, {n} forced steps")
    print(f"   {'route':<14}{'form':<18}{'width':>6}{'chunks':>8}  {'axis':<16}what it is")
    for r in routes:
        form = "recurrent §04.4" if r["form"] == "recurrent" else "chunked §04.7"
        width = "-" if r["width"] is None else str(r["width"])
        same_as_shipped = r["width"] is None or r["chunks"] == n_chunks(len(prompt), SHIPPED_WIDTH)
        print(f"   {r['name']:<14}{form:<18}{width:>6}{r['chunks']:>8}  {r['axis']:<16}"
              + {"a_chunk64": "the pin's family (§04.7 [REF]); every deficit below is measured against it",
                 PRICED: "the engine's own prompt form today — THIS is the priced spread the"
                         " allowances key on, so its ratio to itself is 1.0 by construction",
                 "d_attn_online": "§07.3 blocked (max,sum) decode softmax on top of route a. Legal"
                                  " (\"the blocked form is not banned\", §07.3; A-17), and it is the form"
                                  " the 4k path will need — but the SHIPPED engine uses the one-pass form"
                                  " by §07.3's decision, so this prices a freedom nobody has taken",
                 "e_both": "both freedoms at once: recurrent GDN form + blocked softmax",
                 }.get(r["name"],
                       f"third legal route: §04.7 at width {r['width']}"
                       + ("" if not same_as_shipped else
                          "  <-- INERT: same chunk count as the shipped width, so the same arithmetic")
                       ))

    runs: dict[str, list] = {}
    for r in routes:
        runs[r["name"]], _ = run_route(w, prompt, feed, r["form"], r["width"], r["attn"])
    a = runs["a_chunk64"]

    if check_patch:
        # The falsifiable half of "the rebind is the only thing that moved": at the SHIPPED width the
        # patched call must be bit-identical to the unpatched one. A rebind that leaked a second change
        # (a dtype, a padding rule, a different tensor) shows here instead of passing unexamined.
        patched_run, _ = run_route(w, prompt, feed, "chunk", SHIPPED_WIDTH)
        same = all(torch.equal(patched_run[j], a[j]) for j in range(n))
        print(f"   patch check: route a at width {SHIPPED_WIDTH} under the rebind is "
              + ("BIT-IDENTICAL to route a — the rebind itself moves nothing"
                 if same else "DIFFERENT from route a — the rebind is not a pure width choice; stop"))
        if not same:
            raise SystemExit("route_margin: the chunk-width rebind is not inert at the shipped width")
        del patched_run

    # Bit-equality, not a magnitude test: two routes computing the same thing give cos 1 - 2.2e-16 in F64,
    # so a threshold would call a genuinely inert width "live" and feed a run of zeros into a max, which
    # undercuts the multiplier by exactly the factor D-3 is supposed to measure.
    inert = [k for k in runs if k != "a_chunk64" and all(torch.equal(runs[k][j], a[j]) for j in range(n))]
    if inert:
        print(f"   INERT: {', '.join(inert)} came back bit-identical to route a at every step — same"
              " chunk count, so not a third route. Excluded from the multiplier.")
    alts = [k for k in runs if k not in ("a_chunk64", PRICED, *inert)]
    if not alts:
        print("   no live alternative width on this prompt length: this fixture cannot price a second"
              " legal spread at all (see the chunk-count column).")

    # Alignment: each route must reproduce the fixture's own pinned ids. A miss here is
    # specimpl-vs-HF-at-depth, not the engine, but it is exactly the thing that would make a spread
    # meaningless, so it is printed rather than assumed. (§10.5: step j pairs with greedy_continuation[j+1].)
    agree: dict[str, list[bool]] = {}
    for name, run in runs.items():
        ok = [int(torch.argmax(run[j])) == int(pin_ids[j]) for j in range(n)]
        agree[name] = ok
        print(f"   route {name:<14} argmax vs the pinned ids: {sum(ok)}/{n} agree"
              + ("" if all(ok) else f" — first miss at step {ok.index(False)}"))

    eng = rp.engine_metrics(engine_log)["steps"] if engine_log else {}
    if not eng:
        print("   no --engine-log: the engine column is empty, so `fitted` cannot be computed and the"
              " comparison against the margin in force is report-only")
    elif any(eng[j].get("meanrel") is None for j in eng) or len(eng) < n:
        raise SystemExit(f"route_margin: this log covers {len(eng)} of {n} steps or has no"
                         " `deficit`/`meanrel` metrics lines. `fitted` is a max over steps, so a max"
                         " over a subset is a different (smaller) number. Re-run engine/gate.py.")
    # A step where a legal route picks a different id is not a band question; `route_allowance.read_table`
    # refuses such a table outright and this script says the same thing out loud rather than pricing it.
    disagree = {k: [j for j, ok in enumerate(agree[k]) if not ok] for k in [PRICED] + alts}
    if any(disagree.values()):
        print(f"   ARGMAX DISAGREEMENT vs the pin: " +
              "; ".join(f"{k} at steps {v[:8]}" for k, v in disagree.items() if v) +
              " — those steps are coin-flips, not spread. Priced anyway and flagged, because a tie is"
              " a fact about the reference (D-4: every corpus first flip was one), not a reason to"
              " exclude a step from a max.")

    defs = {name: deficit(run, a, n) for name, run in runs.items()}
    # THE BASELINE THE WHOLE COMPARISON HINGS ON. The engine's deficit is measured against the pinned HF
    # dump; every legal route's deficit above is measured against specimpl's route a. Those are different
    # references, so before concluding anything about the engine, price how far the ORACLE itself sits from
    # the pin: `def_a_vs_pin(j) / def_b(j)` is an engine-independent quantity against the same dump the
    # engine is judged on, and if it is the same size as the fitted margin then the margin is measuring
    # specimpl-vs-HF, not the engine. Costs nothing but reading pins that are already on disk.
    pin_logits = [torch.from_numpy(egate.load_pin(f"decode{j}.logits_last.npy").reshape(-1).astype(
        np.float32)) for j in range(n)]
    defs_pin = deficit(pin_logits, a, n)
    del pin_logits
    shown = [PRICED] + alts
    rows = [{"step": j, "fed": feed[j], "pinned_sampled": pin_ids[j],
             **{f"def_{k}": {f: defs[k][j][f] for f in ("one_cos", "meanrel_rom", "meanrel_mor")}
                for k in shown},
             "engine": ({"one_cos": eng[j]["one_cos"], "meanrel": eng[j]["meanrel"]}
                        if j in eng else None),
             "route_argmax_vs_pin": {k: bool(agree[k][j]) for k in agree}}
            for j in range(n)]

    print(f"\n   step  def_b = priced spread(a,b) " +
          "".join(f" {k[2:] + '/b':>12}" for k in alts) + f" {'engine/b':>10}")
    for j, row in enumerate(rows):
        db = row[f"def_{PRICED}"]["one_cos"]
        line = f"   {j:4d} {db:21.3e} " + "".join(f" {row['def_' + k]['one_cos'] / db:12.4f}" for k in alts)
        if row["engine"]:
            line += f" {row['engine']['one_cos'] / db:10.3f}"
        print(line)

    out: dict = {"fixture": fixture, "n_steps": n, "routes": routes, "rows": rows,
                 "inert_routes": inert, "argmax_disagreement": disagree, "columns": {}}
    axis = {k: next(r["axis"] for r in routes if r["name"] == k) for k in shown}
    # M_1 is the CONSERVATIVE adoption number: §04.7 widths only, because that is the legal freedom the
    # shipped engine is actually entitled to take. The §07.3 axis is reported on M_wide but kept out of
    # M_1 — pricing a freedom the engine does not exercise would make the gate greener for nobody's
    # benefit, which is the failure D-3 exists to catch.
    width_alts = [k for k in alts if axis[k] == "§04.7 width"]
    for label, eng_field, def_field in COLUMNS:
        priced = [defs[PRICED][j][def_field] for j in range(n)]
        per_route = {k: summarize(ratios(defs[k], defs[PRICED], def_field)) for k in shown}
        col: dict = {
            "engine_field": eng_field, "route_field": def_field,
            "priced_spread": {"min": float(min(priced)), "max": float(max(priced)),
                              "median": float(np.median(priced))},
            "per_route": per_route, "axis_of": axis,
            "M_1": ({"lo": min(per_route[k]["max"] for k in width_alts),
                     "hi": max(per_route[k]["max"] for k in width_alts),
                     "lo_route": min(width_alts, key=lambda k: per_route[k]["max"]),
                     "hi_route": max(width_alts, key=lambda k: per_route[k]["max"])}
                    if width_alts else None),
            "M_wide": ({"hi": max(per_route[k]["max"] for k in alts),
                        "hi_route": max(alts, key=lambda k: per_route[k]["max"])}
                       if alts else None),
            "envelope": summarize([(max(row[f"def_{k}"][def_field] for k in shown) / priced[j], j)
                                   for j, row in enumerate(rows)]),
        }
        if eng:
            col["engine_fitted"] = summarize(
                ratios([{"step": j, def_field: eng[j][eng_field]} for j in range(n)],
                       defs[PRICED], def_field))
        # The oracle-vs-pin baseline, in the same units as the margin: how much of the deficit the gate
        # attributes to the engine is already there between specimpl's own route a and the HF dump.
        col["oracle_vs_pin"] = summarize(ratios(defs_pin, defs[PRICED], def_field))
        if eng:
            # ...and the same numerator against the pin-side baseline: how much the engine goes BEYOND the
            # oracle's own distance from the dump, per step. Not an engine-free multiplier (the engine is in
            # the numerator by construction) — it is the decomposition that says whether the margin is buying
            # the reference's HF gap or the engine's own error.
            col["engine_over_oracle"] = summarize(
                ratios([{"step": j, def_field: eng[j][eng_field]} for j in range(n)], defs_pin,
                       def_field))
        if eng:
            # Two numbers the write-up needs, both awkward to re-derive by hand: how many steps would go
            # RED if a route-derived margin were actually adopted (the cost of the adoption being argued
            # against, as a count rather than an opinion), and where the fitted max sits among the
            # DENOMINATORS, because `max_j num(j)/den(j)` can be inflated by a small denominator as
            # easily as by a large numerator, and those two say very different things about the engine.
            m_lo = col["M_1"]["lo"] if col["M_1"] else None
            j_star = col["engine_fitted"]["max_step"]
            col["red_if_adopted"] = {
                "at_M_1_lo": None if m_lo is None else sum(
                    1 for j in range(n) if eng[j][eng_field] > m_lo * defs[PRICED][j][def_field]),
                "at_1.00": sum(1 for j in range(n) if eng[j][eng_field] > defs[PRICED][j][def_field]),
                "at_margin_in_force": None,      # filled once the margin in force is known, below
                "n": n}
            col["fitted_step_context"] = {
                "step": j_star,
                "engine_deficit": eng[j_star][eng_field],
                "priced_spread_there": defs[PRICED][j_star][def_field],
                "spread_percentile": 100.0 * sum(1 for k in range(n)
                                                 if priced[k] <= priced[j_star]) / n,
                "priced_spread_median": float(np.median(priced))}
        out["columns"][label] = col

        print(f"\n  == {label}: each legal deficit against route a, divided by the priced spread (a vs b)"
              f"   [engine column: {eng_field}; route column: {def_field}]")
        print(f"     priced spread: min {col['priced_spread']['min']:.3e}"
              f"  median {col['priced_spread']['median']:.3e}  max {col['priced_spread']['max']:.3e}")
        print(f"     {'route':<16}{'axis':<17}{'max ratio':>10}{'at step':>8}{'p95':>9}{'median':>9}"
              f"{'worst deficit':>15}")
        for k in shown:
            s = col["per_route"][k]
            wdef = max(defs[k][j][def_field] for j in range(n))
            print(f"     {k:<16}{axis[k]:<17}{s['max']:10.4f}{s['max_step']:>8}{s['p95']:9.4f}"
                  f"{s['median']:9.4f}{wdef:15.3e}"
                  + ("  <- ratio 1.0 by construction" if k == PRICED else ""))
        env, e_def = col["envelope"], max(max(row[f"def_{k}"][def_field] for k in shown) for row in rows)
        print(f"     {'ENVELOPE':<16}{'any legal route':<17}{env['max']:10.4f}{env['max_step']:>8}"
              f"{env['p95']:9.4f}{env['median']:9.4f}{e_def:15.3e}")
        op = col["oracle_vs_pin"]
        print(f"     {'ORACLE a vs PIN':<16}{'reference-side':<17}{op['max']:10.4f}{op['max_step']:>8}"
              f"{op['p95']:9.4f}{op['median']:9.4f}"
              f"{max(d[def_field] for d in defs_pin):15.3e}"
              "  <- the oracle's own route a against the same dump the engine is judged on")
        if eng and "engine_over_oracle" in col:
            eo = col["engine_over_oracle"]
            print(f"     {'ENGINE/ORACLE':<16}{'same dump both sides':<21}{eo['max']:10.4f}"
                  f"{eo['max_step']:>8}{eo['p95']:9.4f}{eo['median']:9.4f}"
                  f"{max(eng[j][eng_field] for j in range(n)):15.3e}"
                  "  <- how much BEYOND the oracle's own gap to the pin the engine goes at its worst"
                  " step: ~1.0 means the engine adds nothing to the reference's own HF gap")
        if eng:
            e, c = col["engine_fitted"], col["fitted_step_context"]
            print(f"     {'ENGINE':<16}{'vs the pin':<17}{e['max']:10.4f}{e['max_step']:>8}"
                  f"{e['p95']:9.4f}{e['median']:9.4f}{max(eng[j][eng_field] for j in range(n)):15.3e}"
                  "  <- what the margin is fitted to today")
            if col["M_1"]:
                print(f"     if the route margin were adopted at M_1 lo {col['M_1']['lo']:.4f}:"
                      f" {col['red_if_adopted']['at_M_1_lo']} of {n} steps RED"
                      f" (at margin 1.00: {col['red_if_adopted']['at_1.00']} of {n})")
            print(f"     the fitted max is step {c['step']}, whose priced spread is the"
                  f" {c['spread_percentile']:.0f}th percentile here ({c['priced_spread_there']:.3e} vs"
                  f" median {c['priced_spread_median']:.3e}) — so the ratio is driven by"
                  f" {'a LARGE numerator, not a small denominator' if c['spread_percentile'] >= 50 else 'a small denominator as much as a large numerator'}")
        if col["M_1"]:
            print(f"     M_1 (§04.7 widths only): lo {col['M_1']['lo']:.4f} ({col['M_1']['lo_route']}),"
                  f" hi {col['M_1']['hi']:.4f} ({col['M_1']['hi_route']})")
        if col["M_wide"]:
            print(f"     M_wide (any axis priced here, §07.3 included):"
                  f" hi {col['M_wide']['hi']:.4f} ({col['M_wide']['hi_route']})")

    # Against the margin in force. Read-only: `allowance_margins` is the gate's own reader, so the number
    # compared here is the number the gate applies, not a copy of it.
    doc = egate.read_allowance_doc()
    inforce = egate.allowance_margins(doc) if doc else (None, None, "no allowance file")
    out["in_force"] = {}
    print(f"\n  == against the margin in force — {inforce[2]}")
    for i, (label, eng_field, def_field) in enumerate(COLUMNS):
        key = "margin" if label == "one_cos" else "margin_rel"
        col = out["columns"][label]
        force = inforce[0] if label == "one_cos" else inforce[1]
        fitted = col.get("engine_fitted", {}).get("max")
        detail = {"fitted_max": fitted, "in_force": force, "whence": inforce[2],
                  "per_route": col["per_route"], "M_1": col["M_1"], "M_wide": col["M_wide"],
                  "envelope": col["envelope"], "priced_spread": col["priced_spread"],
                  "oracle_vs_pin": col["oracle_vs_pin"]}
        print(f"   {label:<19} in force {('%.2f' % force) if force is not None else '-':>7}"
              f" | engine-fitted from this log {('%.4f' % fitted) if fitted is not None else '-':>9}"
              f" | M_1 "
              + (f"lo {col['M_1']['lo']:.4f} hi {col['M_1']['hi']:.4f}" if col["M_1"] else "no width axis")
              + (f" | M_wide hi {col['M_wide']['hi']:.4f} ({col['M_wide']['hi_route']})"
                 if col["M_wide"] else "")
              + f" | oracle a vs pin {col['oracle_vs_pin']['max']:.4f}"
                f"@{col['oracle_vs_pin']['max_step']}")
        if fitted is not None and force is not None and col["M_1"]:
            detail["verdict"] = verdict(force, fitted, col["M_1"]["lo"], col["M_1"]["hi"])
            col["red_if_adopted"]["at_margin_in_force"] = sum(
                1 for j in range(n) if eng[j][eng_field] > force * defs[PRICED][j][def_field])
            print(f"      -> {detail['verdict']['action']}: {detail['verdict']['why']}")
        out["in_force"].setdefault(key, {})[label] = detail
    out["provenance"] = {
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "specimpl_sha256": hashlib.sha256((ROOT / "speccheck" / "specimpl.py").read_bytes()).hexdigest(),
        "engine_log": str(engine_log) if engine_log else None,
        "note": "reads data/refs/<fixture>/route_allowance.json and never writes it; no band moved",
    }
    return out


# ---- selftest ----------------------------------------------------------------------------------------

def selftest() -> int:
    """Synthetic: no weights, no fixture, no device. The properties that make the multiplier honest."""
    bad: list[str] = []
    def eq(label, got, want):
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} — got {got}, want {want}")
        if not ok:
            bad.append(label)

    def rows(*triples):
        return [{"step": s, "one_cos": e, "meanrel": e} for s, e, _ in triples]

    denom = rows((0, 1.0e-4, 0), (1, 2.0e-4, 0), (2, 5.0e-5, 0))
    eq("1. a legal route whose deficit is 2x the priced spread at some step gives multiplier 2.5 there"
       " and 2.0 elsewhere — the max is what a cap needs",
       summarize(ratios(rows((0, 2.0e-4, 0), (1, 4.0e-4, 0), (2, 1.0e-4, 0)), denom, "one_cos"))["max"], 2.0)
    twice = rows((0, 2.0e-4, 0), (1, 5.0e-4, 0), (2, 1.0e-4, 0))
    s = summarize(ratios(twice, denom, "one_cos"))
    eq("2. the max carries the step that set it, and p95/median sit under it (percentiles are shape,"
       " never the constant)",
       (s["max_step"], s["max"], s["p95"] <= s["max"], s["median"] <= s["p95"]), (1, 2.5, True, True))
    eq("3. a width that repartitions the prompt the same way is not a third route: an all-zero deficit"
       " yields no multiplier rather than a 0.0 that a max would swallow",
       verdict(3.5, 3.0, 0.0, 0.0)["action"], "no multiplier")
    # The anti-loosening rule, in every direction.
    eq("4. a multiplier SMALLER than what the engine needs never becomes an action — it is a finding"
       " about the engine (the D-3 trap)",
       verdict(3.5, 3.5, 1.0, 1.2)["action"], "keep")
    eq("5. adopting a multiplier LOOSER than the margin in force is refused as a band move",
       verdict(3.5, 3.0, 4.2, 4.6)["action"], "keep (adopting would LOOSEN)")
    eq("6. the one case where the basis may change: every priced width explains the engine AND the"
       " conservative multiplier is no looser than what is in force",
       verdict(3.5, 3.0, 3.2, 3.4)["action"], "adopt-with-changed-basis")
    eq("7. when two legal widths straddle the fitted number the answer is a knob and not a"
       " calibration, so the margin stays where it is",
       verdict(3.5, 3.0, 2.4, 4.0)["action"], "keep (width-sensitive)")
    eq("8. ceil2 rounds UP at 2 dp, so a re-derived constant cannot be quoted looser than its"
       " measurement", ceil2(3.4939), 3.50)
    refused = False
    try:
        ratios(rows((0, 1.0, 0)), rows((0, 0.0, 0)), "one_cos")
    except SystemExit:
        refused = True
    eq("9. a step where two legal routes agree exactly is refused, not skipped (a skipped step is a"
       " smaller max, which is a loosening nobody measured)", refused, True)
    eq("10. §04.7's chunk arithmetic: a 17-token prompt is ONE chunk at width 32 and at width 64, so both"
       " widths are the same computation — the reason an inert width must be reported, not priced",
       (n_chunks(17, 32), n_chunks(17, 64), n_chunks(17, 16), n_chunks(295, 32), n_chunks(295, 64)),
       (1, 1, 2, 10, 5))
    eq("11. a 17-token prompt at width 8 really is 3 chunks, so the short fixtures do have a live width"
       " axis once the width is below the prompt",
       (n_chunks(17, 8), n_chunks(27, 8), n_chunks(232, 64), n_chunks(232, 32)), (3, 4, 4, 8))
    # The two meanrel statistics this file prices are NOT the same quantity, and the difference is not a
    # rounding matter: `engine/gate.py:metrics` uses §10.2's ratio of means ("not a mean of ratios — the
    # latter explodes on the near-zero logits that most of a 248 320-wide vocabulary is", its own words),
    # while `route_price.meanrel` — the number every cached margin_rel divides by — IS that mean of ratios.
    # On a vocabulary-shaped tensor they differ by an order of magnitude, so `margin_rel` is not a
    # like-for-like ratio and must not be read as one (engine/results.md, D-3).
    vocab_like = torch.full((1000,), 1e-4)
    vocab_like[0] = 1.0                      # one real logit, 999 near-zero ones: the §10.2 argument
    perturbed = vocab_like + 1e-5            # the SAME absolute perturbation everywhere
    eq("12. a uniform perturbation prices ~11x higher as a mean of ratios (route_price's route_meanrel)"
       " than as a ratio of means (§10.2, what gate.py prints for the engine), which is why this file"
       " reports both and why margin_rel cannot be compared to either alone",
       round(rp.meanrel(perturbed, vocab_like) / meanrel_rom(perturbed, vocab_like)), 11)
    print("SELFTEST " + ("FAIL — " + "; ".join(bad) if bad else
                         "PASS (the multiplier is comparable to the fitted margin; neither a looser"
                         " multiplier, a dropped step, nor an inert width can be silently adopted)"))
    return 1 if bad else 0


# ---- driver ------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", action="append", default=[], metavar="NAME",
                    help="a priced fixture (dec64f, dec64t, dec300f, mt32f). Repeatable; one fixture"
                         " per run is the norm because every multiplier is per-fixture (D-2).")
    ap.add_argument("--engine-log", default=None,
                    help="an `engine/gate.py --fixture X > log` log, for the engine column only. The"
                         " engine appears in the `fitted` column and NOWHERE in the multiplier.")
    ap.add_argument("--chunk-alt", default="8,16,32,128",
                    help="comma-separated alternative §04.7 chunk widths. Several turn the single third"
                         " route into a width-sensitivity check, which is what shows the multiplier is a"
                         " property of the legal-freedom axis and not of a knob; widths that do not"
                         " repartition the prompt are reported INERT and excluded.")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="price only the first N forced steps (cheap smoke; not a result)")
    ap.add_argument("--check-patch", action="store_true",
                    help="also run route a through the rebind at the shipped width and require it"
                         " bit-identical to the unpatched route a (proves the width rebind is inert;"
                         " costs one extra reference pass)")
    ap.add_argument("--no-attn-axis", action="store_true",
                    help="skip the §07.3 blocked-softmax routes (d, e). They are reported on M_wide and"
                         " never in M_1, but they cost two reference passes each fixture.")
    ap.add_argument("--out", default=None, help="write the per-fixture tables as JSON here (tmp/, never"
                                               " data/refs/ — that directory is what the gate reads)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.fixture:
        raise SystemExit("route_margin: pass at least one --fixture (or --selftest)")
    alts = [int(x) for x in args.chunk_alt.split(",") if x.strip()]
    if SHIPPED_WIDTH in alts:
        raise SystemExit(f"route_margin: --chunk-alt {alts} contains the shipped width"
                         f" {SHIPPED_WIDTH}; that is route a, not a third route")
    if args.out and "data/refs" in args.out:
        raise SystemExit(f"route_margin: refusing to write {args.out}; this instrument reports, it does"
                         " not price. data/refs/<fixture>/route_allowance.json belongs to"
                         " engine/route_allowance.py alone.")

    w = Weights(ROOT / "models" / "Qwen3.5-0.8B")
    docs = []
    for fx in args.fixture:
        log = Path(args.engine_log) if args.engine_log and len(args.fixture) == 1 else None
        docs.append(price_fixture(w, fx, log, alts, args.max_steps, args.check_patch,
                                  attn_axis=not args.no_attn_axis))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fixtures": docs}, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
