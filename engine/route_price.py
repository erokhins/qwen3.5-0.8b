#!/usr/bin/env python3
"""What is a GDN prefill route worth on the LOGITS vector — measured inside the reference, with no engine?

Two modes, because the question grew a second half:

    .venv/bin/python engine/route_price.py                              # dec8, the 8-step table
    .venv/bin/python engine/route_price.py --fixture dec64f \
          --engine-log progress/engine/evidence/gate_dec64f.log                              # the whole forced chain, 64 steps
    .venv/bin/python engine/route_price.py --fixture dec64f --engine-log ... \
          --out progress/engine/evidence/route_dec64f.json                                   # same table as JSON

Why this exists. `engine/logits_floor.py` prices one op (final norm + tied `lm_head`) with the reference's
own `L23.s_out` held fixed, so its 0.99999862 is the floor for *reassociation-only change on the
reference's own arithmetic at that one op*. It is not a floor for an end-to-end token-level logits vector,
because it removes by construction the 23 other layers, the whole token chain, and the legitimate choice of
how a prompt gets into the GDN state. This script prices that missing term: it runs the dec8 prompt plus its
greedy continuation twice through `speccheck/specimpl.py` — once with the chunked GDN prefill (§04.7, the
form G3 canonicalises) and once with the recurrent form (§04.4, the form an engine with no prefill
necessarily takes, since it feeds the prompt one token per decode step) — and compares the two runs'
logits vector at every step using `engine/gate.py::metrics`' F64 cos. Same weights, same tokens, no engine,
no device. If the spread between two legal reference forms is the same order as the engine's deviation from
the pin, the engine's deviation needs no bug to explain it.

Why the deep mode exists. `engine/gate.py --fixture dec64f` FAILs on 7 steps whose logits `cos` sits under
G5's 0.9995 (worst 0.997844 at step 32), and the only thing known about that window is that the reference's
own GDN state energy halves inside it. Two readings survive that fact — the engine lost state there, or the
comparison stopped being a fair one because the thing it compares against got small — and the question
"how far apart are two LEGAL reference routes at those steps?" is the one that separates them. So the deep
mode replays the fixture's own forced chain (the manifest's `greedy_continuation`, exactly the list the
gate hands `decode_run --replay`) through both forms and prices them at every step. Deciding what the
answer *means* for the band is the operator's call, not this script's: it prints the table and the verdict
per step, and changes no band.

Step numbering: the deep table uses `engine/gate.py`'s, i.e. step j is the step fed `greedy_continuation[j]`
and compared with pinned `decode{j}.logits_last.npy`. (The 8-step table below predates that and is numbered
from the prefill position, so its row 2 is the gate's step 0.)

dec8, measured 2026-10-11, the same 8 continuation tokens fed to both forms, argmax agreement all True:

| step (gate numbering) | 1-cos, route vs route | × the last-op floor | engine's own 1-cos vs the pin |
|---|---|---|---|
| 1 | 1.900e-04 | 138× | 1.14e-4 |
| 2 | 5.262e-05 | 38× | 4.1e-5 |
| 3 | 6.989e-05 | 51× | 3.7e-5 |
| 4 | 5.314e-05 | 39× | 3.5e-5 |
| 5 | 4.730e-05 | 34× | 3.5e-5 |
| 6 | 5.628e-05 | 41× | 3.4e-5 |
| 7 | 1.025e-04 | 74× | 3.2e-5 |
| 8 | 1.080e-04 | 78× | 4.8e-5 |

The route spread is larger than the engine's entire deviation **at every step**, and it is worst at step 1
— the same shape the engine shows. So the engine sits inside the band the reference leaves open between two
of its own legal forms, and the fair end-to-end floor is this table, not 0.99999862.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speccheck"))
sys.path.insert(0, str(ROOT))
from specimpl import Weights, Forward  # noqa: E402
from engine import curve_probe, gate as egate  # noqa: E402  (the gate owns the pin readers and the (e) parser)

torch.set_grad_enabled(False)

FLOOR = 1.38e-6  # engine/logits_floor.py: last-op floor, distance from 1
G5 = 0.9995      # §10.1 G5, the hard floor engine/gate.py applies per step. Read here, never edited.


def f64(t):
    return t.double() if t.dtype == torch.bfloat16 else t.float().double()


def cos(a, b):
    a, b = f64(a).ravel(), f64(b).ravel()
    return float((a @ b / (a.norm() * b.norm() + 1e-30)).item())


def meanrel(a, b):
    a, b = f64(a).ravel(), f64(b).ravel()
    return float(((a - b).abs() / b.abs().clamp_min(1e-6)).mean().item())


# One row of `engine/gate.py`'s (b) section: `step 54: metrics — cos 0.999767 deficit 2.33e-04 meanrel
# 2.2452% …`. The deficit field is there so this script can read the engine's own numbers at full
# precision instead of at the (e) table's six decimals of `cos`.
B_LINE = re.compile(r"step (\d+): metrics .*?deficit ([0-9.eE+-]+) meanrel ([0-9.]+)%")


def engine_metrics(log: Path | None) -> dict:
    """The engine's own per-step numbers, read out of a `engine/gate.py` log. Read-only, no device.

    The (e) table prints cos at six decimals, which on a 2.3e-4 deficit is ±0.4 % — enough to move a
    derived margin — so the (b) metrics line is preferred: it prints the deficit in scientific notation
    and meanrel to four decimals of a percent. The (e) row is the fallback for logs written before that
    field existed, and `source` says which a given log supplied.
    """
    out: dict = {}
    if log is None or not log.exists():
        return {"source": "no log given", "steps": {}}
    steps = {step: {"cos": cos, "one_cos": 1.0 - cos, "meanrel": None, "precise": False}
             for step, cos, _n2, _dn2 in curve_probe.parse(log)}
    by_b = 0
    for line in log.read_text().splitlines():
        m = B_LINE.search(line)
        if not m:
            continue
        j, deficit, rel = int(m.group(1)), float(m.group(2)), float(m.group(3)) / 100.0
        row = steps.setdefault(j, {"cos": None, "meanrel": None})
        row.update({"one_cos": deficit, "meanrel": rel, "precise": True,
                    "cos": 1.0 - deficit if row.get("cos") is None else row["cos"]})
        by_b += 1
    return {"steps": steps, "source": f"{by_b} of {len(steps)} steps from the (b) metrics lines"
            + ("" if by_b else " (this log predates the `deficit` field: 6-decimal cos instead)")}



def price(w: Weights, prompt: list[int], feed: list[int]) -> tuple[list, list]:
    """The prompt, then the forced feed list, through both legal GDN prefill forms.

    Returns the two logits lists, index-aligned with `feed`: element j is the logits vector produced
    after consuming `feed[j]`, which is exactly what the gate compares with `decode{j}.logits_last`.
    The two runs differ in ONE thing — `gdn_form`, i.e. chunked §04.7 vs recurrent §04.4 for getting the
    prompt into the GDN state. The decode path is the same recurrent code in both, and the feed list is
    the reference's own tokens, so neither run can diverge from the other by picking a different token.
    """
    runs = {}
    for form in ("chunk", "recurrent"):
        f = Forward(w, gdn_form=form)
        f.prefill(torch.tensor(prompt))
        runs[form] = [f.logits(f.decode(tok))[-1] for tok in feed]
    return runs["chunk"], runs["recurrent"]


def dec8_price(w: Weights) -> None:
    """The published 8-step table: dec8's prompt plus its own greedy continuation."""
    fix = ROOT / "data" / "refs" / "dec8"
    ids = json.loads((fix / "manifest.json").read_text())["prefill_ids"]
    print(f"dec8 prefill_ids: {len(ids)} tokens = {ids}")

    def run(form, cont):
        """One prefill pass over the whole prompt, then the continuation token-at-a-time. Returns the logits
        vector that predicts each next token: position len(ids)-1, then one per continuation token."""
        f = Forward(w, gdn_form=form)
        lg = [f.logits(f.prefill(torch.tensor(ids)))[-1]]
        for tok in cont:
            lg.append(f.logits(f.decode(tok))[-1])
        return lg

    cont = Forward(w, gdn_form="chunk").greedy(ids, 8)
    print(f"chunk-form continuation: {cont}")
    chunk = run("chunk", cont)
    recur = run("recurrent", cont)

    print("\nroute-vs-route on the logits vector (chunked prefill vs recurrent prefill, same tokens):")
    print("step  cos        meanrel      1-cos        x the last-op floor")
    for j in range(8):
        c, mr = cos(recur[j], chunk[j]), meanrel(recur[j], chunk[j])
        print(f"  {j + 1:>2}   {c:.6f}   {mr:.3e}   {1 - c:.3e}   {(1 - c) / FLOOR:>7.1f}x")
    print("\nengine observed vs the pinned dump, per step: "
          "0.999886 0.999959 0.999963 0.999965 0.999965 0.999966 0.999968 0.999952")
    print("engine 1-cos per step: 1.14e-4 4.1e-5 3.7e-5 3.5e-5 3.5e-5 3.4e-5 3.2e-5 4.8e-5")
    print("last-op floor (engine/logits_floor.py): 0.99999862")
    print("argmax agreement between the two routes, per step:",
          [int(torch.argmax(a)) == int(torch.argmax(b)) for a, b in zip(chunk, recur)])


def forced_price(w: Weights, fixture: str, engine_log: Path | None,
                 out: Path | None = None, max_steps: int = 0) -> int:
    """Every step of a forced fixture: two legal routes against each other, and (if a gate log is
    supplied) the engine against the pin, on the same row.

    `--max-steps` exists to make this script cheap enough to smoke-test: the full 64-step run is two
    CPU forwards over the whole chain and takes minutes, and a typo in the table code should not cost
    that to find.
    """
    egate.set_fixture(fixture)
    man = json.loads((egate.REFS / "manifest.json").read_text())
    prompt = man["prefill_ids"]
    chain = man["greedy_continuation"]
    feed = chain[:-1]                       # the same list engine/gate.py hands `decode_run --replay`
    if max_steps:
        feed = feed[:max_steps]
    n = len(feed)
    print(f"fixture {fixture}: prompt {len(prompt)} tokens, forced chain {n} steps")
    print(f"feed list = greedy_continuation[:-1] = {feed}")
    stop_steps = [j for j in range(n) if chain[j + 1] == egate.STOP_ID]
    print(f"the pinned chain samples the stop id {egate.STOP_ID} at step(s) {stop_steps}"
          " — everything after the first of those is the model being driven past its own end of turn")

    chunk, recur = price(w, prompt, feed)

    # Alignment check, so a mis-numbered table cannot be read as a result: if this replay is the same
    # chain the fixture pinned, each route's own argmax must reproduce the pinned next id. A miss here
    # is specimpl-vs-HF, not the engine — but it is worth seeing at depth.
    pin_ids = chain[1:]
    for form, runs in (("chunk", chunk), ("recurrent", recur)):
        ok = [int(torch.argmax(a)) == int(b) for a, b in zip(runs, pin_ids)]
        miss = [f"{j}: got {int(torch.argmax(runs[j]))} want {pin_ids[j]}"
                for j in range(n) if not ok[j]]
        print(f"  route {form:9s} argmax vs the pinned ids: {sum(ok)}/{n} agree"
              + ("" if not miss else f" — misses {miss}"))

    eng_log = engine_metrics(Path(engine_log) if engine_log else None)
    engine = eng_log["steps"]
    if engine:
        print(f"  engine-vs-pin read from {engine_log} ({len(engine)} steps; {eng_log['source']})")
    else:
        print("  no --engine-log given: the engine columns are empty (the device run is engine/gate.py's,"
              " this script never touches the device)")

    print("\nstep  fed      pin S n2   route cos   route 1-cos  route mrel  routes-argmax |"
          " engine cos  engine 1-cos  verdict")
    table = []
    for j in range(n):
        c = cos(recur[j], chunk[j])
        mr = meanrel(recur[j], chunk[j])
        same = int(torch.argmax(chunk[j])) == int(torch.argmax(recur[j]))
        n2 = egate.fp_total(j, "state", 0)          # col 0 = n2 = sum x^2, summed over the GDN layers
        ec = engine.get(j, {}).get("cos")
        eoc = engine.get(j, {}).get("one_cos")
        emr = engine.get(j, {}).get("meanrel")
        bad_route = c < G5
        bad_eng = ec is not None and ec < G5
        verdict = ("both legal routes under G5 too -> engine is inside legal spread" if bad_route and bad_eng
                   else "OURS: engine under G5 where BOTH legal routes are clean" if bad_eng
                   else "routes alone under G5; engine clean" if bad_route
                   else "ok")
        table.append({"step": j, "fed": feed[j], "pinned_sampled": pin_ids[j],
                      "pin_s_n2": n2, "route_cos": c, "route_one_cos": 1.0 - c, "route_meanrel": mr,
                      "routes_argmax_agree": bool(same),
                      "engine_cos": ec, "engine_one_cos": eoc, "engine_meanrel": emr,
                      "engine_ratio": None if eoc is None else eoc / (1.0 - c),
                      "engine_rel_ratio": None if emr is None else emr / mr,
                      "after_first_stop": bool(stop_steps) and j > stop_steps[0],
                      "verdict": verdict})
        r = table[-1]
        print(f"{j:4d}  {feed[j]:8d}  {('%g' % n2) if n2 is not None else '-':>9}"
              f"   {c:.6f}   {1 - c:.3e}   {mr:8.4%}   {'yes' if same else 'NO ':>4}"
              f"        | "
              + (f"{ec:.6f}   {eoc:.3e}    " if ec is not None else "     -           -      ")
              + verdict)

    print("\n== what the table says")
    rc = np.array([r["route_cos"] for r in table])
    n2 = np.array([r["pin_s_n2"] if r["pin_s_n2"] is not None else np.nan for r in table])
    eng = np.array([r["engine_cos"] if r["engine_cos"] is not None else np.nan for r in table])
    print(f"  route-vs-route: worst cos {rc.min():.6f} at step {int(rc.argmin())};"
          f" median 1-cos {np.median(1 - rc):.3e}; {int((rc < G5).sum())} of {n} steps under G5 {G5}")
    if np.isfinite(eng).any():
        print(f"  engine-vs-pin:  worst cos {np.nanmin(eng):.6f} at step {int(np.nanargmin(eng))};"
              f" median 1-cos {np.median(1 - eng):.3e}; {int(np.nansum(eng < G5))} of"
              f" {n} steps under G5")
        both = int(np.nansum((eng < G5) & (rc < G5)))
        ours = int(np.nansum((eng < G5) & (rc >= G5)))
        reading = ("the legal routes cover the engine everywhere it fails"
                   if ours == 0 else
                   "the engine fails at steps the two legal routes do not: those are ours")
        print(f"  under-G5 steps: engine {int(np.nansum(eng < G5))}, route-vs-route"
              f" {int((rc < G5).sum())}, BOTH {both}, engine-only {ours} -> {reading}")
    # The two ratios `engine/gate.py`'s route allowance is keyed on. They are printed here so the number
    # that sets a margin is visible in the same table it came from; `engine/route_allowance.py` is what
    # turns them into the constant, over EVERY priced fixture rather than this one.
    for field, label in (("engine_ratio", "engine deficit / route 1-cos    "),
                         ("engine_rel_ratio", "engine meanrel / route meanrel")):
        pairs = [(r[field], r["step"]) for r in table if r[field] is not None]
        if not pairs:
            continue
        vals = np.array([v for v, _ in pairs])
        worst = max(pairs)
        print(f"  {label}: median {np.median(vals):.3f}, max {worst[0]:.4f} at step {worst[1]}"
              "  <- the per-step price of a legal route choice at that step")
    print(f"  corr(route 1-cos, 1/(S n2)) = "
          f"{np.corrcoef(1 - rc, 1.0 / n2)[0, 1]:.3f}"
          f"   corr(route 1-cos, step) = {np.corrcoef(1 - rc, np.arange(n))[0, 1]:.3f}")
    if np.isfinite(eng).any():
        print(f"  corr(engine 1-cos, route 1-cos) = {np.corrcoef(1 - eng, 1 - rc)[0, 1]:.3f}"
              "  (do the two independent things dip in the same place?)")
    if stop_steps:
        s0 = stop_steps[0]
        below = [j for j in range(n) if rc[j] < G5]
        print(f"  eos crossing: the pinned chain samples the stop id at step {s0}; route-under-G5 steps"
              f" {below} sit {sum(j > s0 for j in below)} after it and {sum(j <= s0 for j in below)}"
              " before it")
        e_below = [r["step"] for r in table if r["engine_cos"] is not None and r["engine_cos"] < G5]
        if e_below:
            print(f"  engine-under-G5 steps {e_below}: {sum(j > s0 for j in e_below)} after the"
              f" crossing, {sum(j <= s0 for j in e_below)} before it")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fixture": fixture, "stop_steps": stop_steps, "rows": table},
                                  indent=1))
        print(f"  wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="dec8",
                    help="dec8 (the published 8-step table) or a forced fixture, e.g. dec64f")
    ap.add_argument("--engine-log", default=None,
                    help="an `engine/gate.py --fixture X > log` log, whose (e) table supplies the"
                         " engine-vs-pin cos column. Read-only; produced by a device run.")
    ap.add_argument("--out", default=None, help="write the per-step table as JSON here")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="price only the first N forced steps (a cheap way to test this script;"
                         " the published tables come from the full chain)")
    args = ap.parse_args()
    w = Weights(ROOT / "models" / "Qwen3.5-0.8B")
    if args.fixture == "dec8":
        dec8_price(w)
        return 0
    return forced_price(w, args.fixture, args.engine_log,
                        Path(args.out) if args.out else None, args.max_steps)


if __name__ == "__main__":
    sys.exit(main())
