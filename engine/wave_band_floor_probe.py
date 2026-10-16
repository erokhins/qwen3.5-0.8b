#!/usr/bin/env python3
"""Measure the spread of the pairing gate section (h) actually makes, per GEMM route, and price a
floor from it. No timings; the numbers here are per-row `1-cos` between two runs of the engine.

    .venv/bin/python engine/wave_band_floor_probe.py                       # both routes, the default set
    .venv/bin/python engine/wave_band_floor_probe.py --fixtures dec8,dec64f
    .venv/bin/python engine/wave_band_floor_probe.py --routes lt --margin 1.25
    .venv/bin/python engine/wave_band_floor_probe.py --routes lt --gemm-pins none   # rank-0 instead

WHAT IS BEING MEASURED, AND WHY IT HAS TO BE RE-MEASURED AT ALL. Section (h) runs one prompt twice —
once at the wave scratch's full width (`--wave-rows 512`) and once with `--wave-rows 1` — and compares
the two logits corpora. Because `resolve_wave_gemm` forces the hand GEMV at one row, the two sides are
two DIFFERENT GEMM implementations on any library route: hand-vs-hipBLAS under `--gemm library`, and
hand-vs-hipBLASLt under `--gemm lt`. The band it gates against (`CASCADE_BAND.row_1cos_max` = 2e-3)
was measured for the first of those (§A2 of progress/2026-10-14/route-c-audit-2026-10-14.md: worst row
1.889e-3 over 1694 rows, line drawn 6 % above it). `lt` is a different library with a different
reassociation, so the pairing's independent variable changed and the old number is a claim about a
pair this gate no longer runs. This probe measures the new pair the way `prototype/glue/floor_probe.py`
and `prototype/mlp_gemv/floor_probe.py` measure theirs: over several fixtures, with the frozen
comparison spelled by the shipping code rather than re-derived here, and a proposal that is the
measured max plus a stated margin — never a number anyone has to remember.

ONE SPELLING OF THE SIDES AND OF THE STATISTIC. The two sides come from `engine/gate.py`'s own
`run_batched_ab` and the distance from its own `corpus_drift`, so the number this probe prints IS the
number section (h) will gate, not a near-copy of it. A probe that reimplemented the comparison could
agree with itself and be wrong, which is the trap `prototype/attn_tile_model.py`'s `compare` records.

WHICH ALGO CONFIG IS BEING MEASURED, and why the probe has to say. A spread is a property of the
algorithms that ran, and on the hipBLASLt route the algorithm per shape is chosen by `algo_pins()`
minus whatever `--gemm-pins` withholds. The band that came out of the 2026-10-16 run was therefore
stated against a three-entry table, and the table was trimmed to one entry the same day (D28) — a
provenance claim about a config no binary runs. So `--gemm-pins` is a named input of this probe, it
reaches every side of every pair, and it is printed on the `FLOOR-PROPOSE` line beside the max.

CORRECTNESS ONLY. Nothing here times anything and nothing here may be quoted in a timing document
(AGENTS.md: a worker's device run takes no timings). Each side is one engine run; the wall cost is
the engine's, not a measurement of anything.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import gate  # the shipping gate: its sides, its statistic, its route/fixture state

# The fixtures, in the order engine/route_c_audit.py uses for its own corpus, minus the ones whose
# one-row side is a >4-minute run on this box (dec300f: 295 prompt + 128 generated rows, every one of
# them a separate wave on the `--wave-rows 1` side). The set is chosen for SHAPE variety, not to make a
# number look small: a short prompt (dec8), a replayed 64-step prompt (dec64f, the one that first read
# red), a prompt with tied steps (dec64t), a prompt that crosses wave boundaries (chunk64b), a prompt
# of exactly one wave (p128), a mid-width one (mt32f), and three that straddle or exceed the wave width
# (wide520f 500, wide640f 640 = 512+128, wide1024f = two full waves).
DEFAULT_FIXTURES = ("dec8", "dec64f", "dec64t", "chunk64b", "p128", "mt32f",
                    "wide520f", "wide640f", "wide1024f")
DEFAULT_ROUTES = ("library", "lt")
# The margin is stated because a band that is only as wide as the worst row it has seen is a band that
# fails on the next prompt. D-11 drew its line 6 % above a 1694-row, 8-fixture corpus; this corpus is
# smaller, so it gets a wider margin rather than a borrowed one. `--margin` prints what it used either
# way, and the gate line quotes the number, not this rationale.
DEFAULT_MARGIN = 1.25


def measure_side(args, fixture: str, route: str, work: Path) -> dict | None:
    """One (fixture, route): run the two sides section (h) runs, then price the drift between them.

    Returns `corpus_drift`'s dict plus what it was measured on, or None if a side did not run — and a
    missing side is a REFUSAL, not a short list, because a check that averages over the pairs that
    happened to run is the empty-must-catch-set trap again.
    """
    gate.set_fixture(fixture)
    gate.set_route(route)
    # The pin set is part of what is being measured, not an ambient default: the D28 band was cited
    # against a three-entry algo table and the binary then shipped one, and the band text was left
    # describing a config nobody runs. So the spec comes from the command line, goes to every side of
    # every pair through `gate.GEMM_PINS`, and is printed on the proposal line. `none`/csv only means
    # something on the lt route — hipBLAS has no algo pins, so the library leg always runs `all`.
    gate.set_pins(a_spec_for(route, args))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        man = json.loads((gate.REFS / "manifest.json").read_text())
        prompt_ids, dec_logits, _inputs = gate.pairing_checks(man)
        greedy = man["greedy_continuation"]
        n_decode = len(dec_logits)
        replay = n_decode > gate.FREE_RUN_BOUND
        out_dir = work / fixture / route
        out_dir.mkdir(parents=True, exist_ok=True)
        wave_rows = gate.wave_row_ceiling()
        sides = gate.run_batched_ab(args, prompt_ids, n_decode, greedy, replay, out_dir,
                                   wave_rows, 0)
        failed = list(gate.fails)
    # The two command lines are this leg's whole provenance — which binary, which route, which widths —
    # so they are printed. The gate's own section (0) chatter is not, because it repeats per fixture.
    for line in buf.getvalue().splitlines():
        if line.strip().startswith("$"):
            print("  " + line.strip())
    if failed:
        print(f"  REFUSE {fixture}/{route}: a side did not run clean — {failed[-1]}")
        return None
    drift = gate.corpus_drift(sides["batched"]["dir"], sides["perrow"]["dir"])
    if drift is None or "mismatch" in drift:
        print(f"  REFUSE {fixture}/{route}: no comparable corpora ({drift})")
        return None
    drift["fixture"], drift["route"] = fixture, route
    drift["rows_expected"] = len(prompt_ids) + n_decode
    drift["wave_rows"] = wave_rows
    return drift


def a_spec_for(route: str, args) -> str:
    """The `--gemm-pins` spec this leg runs under: the requested one on `lt`, and `all` on the hipBLAS
    leg, where there is no pin concept for it to select. Said out loud rather than silently dropped,
    because a report that quietly ignored a requested spec is the attribution hole all over again."""
    if route != "lt" and args.gemm_pins != "all":
        print(f"  [note] --gemm-pins {args.gemm_pins!r} ignored on the `library` leg: an algo pin is a"
              " hipBLASLt record and hipBLAS runs none. Only the `lt` leg is pin-sensitive.")
    return args.gemm_pins if route == "lt" else "all"


def print_leg(route: str, rows: list[dict], band: float, margin: float, pins: str) -> dict | None:
    """Per-fixture table for one route, then the aggregate and the proposal it licenses."""
    print(f"\n=== pairing: hand GEMV at --wave-rows 1 vs the library at --wave-rows "
          f"{rows[0]['wave_rows']} — route `--gemm {route}` --gemm-pins {pins} ===")
    print(f"{'fixture':12s} {'rows':>5} {'bit=':>5} {'worst 1-cos':>12} {'at':>14} "
          f"{'pooled meanrel':>15} {'pooled 1-cos':>13} {'vs band':>9}")
    worst = None
    for d in rows:
        at = f"logits{d['worst_1cos_row']}.bin"
        print(f"{d['fixture']:12s} {d['rows']:5d} {d['identical']:5d} {d['worst_1cos']:12.4e}"
              f" {at:>14} {d['meanrel']:15.4e} {d['one_cos']:13.4e}"
              f" {d['worst_1cos'] / band:9.3f}x")
        if worst is None or d["worst_1cos"] > worst["worst_1cos"]:
            worst = d
    total_rows = sum(d["rows"] for d in rows)
    proposed = worst["worst_1cos"] * margin
    print(f"  MAX over {len(rows)} fixtures and {total_rows} compared rows: "
          f"{worst['worst_1cos']:.4e} at {worst['fixture']}/logits{worst['worst_1cos_row']}.bin")
    print(f"  band in force for this pairing: {band:.3e}  "
          f"({'inside' if worst['worst_1cos'] <= band else 'OUTSIDE — that is the red being chased'})")
    print(f"  FLOOR-PROPOSE route={route} gemm_pins={pins} measured_max={worst['worst_1cos']:.6e} "
          f"margin={margin:.2f}x proposed={proposed:.3e} "
          f"worst_at={worst['fixture']}:logits{worst['worst_1cos_row']}.bin "
          f"corpus={len(rows)} fixtures/{total_rows} rows "
          f"sides=hand(--wave-rows 1) vs {route}(--wave-rows {worst['wave_rows']})")
    return {"route": route, "max": worst["worst_1cos"], "at": worst["fixture"], "rows": total_rows,
            "proposed": proposed, "fixtures": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", default=",".join(DEFAULT_FIXTURES))
    ap.add_argument("--routes", default=",".join(DEFAULT_ROUTES),
                    help="comma list of the GEMM routes whose pairing is priced. Narrowing it to "
                         "`lt` is allowed but the report says which legs are missing, because the "
                         "point of running both is that the two pairings sit side by side.")
    ap.add_argument("--binary", default=str(gate.DEFAULT_BINARY))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--work", default=str(ROOT / "tmp" / "wave_band_floor"))
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--attn-tile", default="1")
    ap.add_argument("--attn-touch", default="1")
    ap.add_argument("--gdn-scan", default="seq")
    ap.add_argument("--gdn-chunk", type=int, default=64)
    # The pin set the band is being measured on, named on the command line because a band whose
    # provenance does not name the algo config it was measured under is a band that goes stale the day
    # the table is trimmed (D28). `all` means "whatever `algo_pins()` holds right now", which is the
    # shipping config; `none` prices the same corpus with the heuristic's rank-0 everywhere.
    ap.add_argument("--gemm-pins", default="all")
    a = ap.parse_args()

    fixtures = [f for f in a.fixtures.split(",") if f]
    routes = [r for r in a.routes.split(",") if r]
    for r in routes:
        if r not in gate.GEMM_ROUTES or r == "hand":
            raise SystemExit(f"probe: --routes takes the library routes ({', '.join(DEFAULT_ROUTES)});"
                             " `hand` is bit-identical to itself and has no band to price")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    # The `args` that `gate.run_batched_ab` reads. Only these fields are consulted, and they are the
    # same ones the gate's own defaults carry — a width A/B whose two sides ran a different attention
    # form or GDN scan measures two things at once, which is why they are named here rather than left
    # to a stand-in's defaults. `gemm_pins` is read by `a_spec_for` below and reaches the engine through
    # `gate.GEMM_PINS`, which `measure_side` sets per leg.
    args = argparse.Namespace(binary=a.binary, model=a.model, attn_tile=a.attn_tile,
                              attn_touch=a.attn_touch, gdn_scan=a.gdn_scan,
                              gdn_chunk=a.gdn_chunk, gemm_pins=a.gemm_pins)

    print(f"fixtures: {', '.join(fixtures)}")
    print(f"routes:   {', '.join(routes)}   margin: {a.margin:.2f}x   work: {work}")
    print(f"algo pins: --gemm-pins {a.gemm_pins} (hipBLASLt leg only; the engine echoes"
          " `pins=<in force>/<in the table>` and `gate.py` refuses a run whose echo disagrees)")
    if sorted(routes) != sorted(DEFAULT_ROUTES):
        print(f"  [note] running {len(routes)} of {len(DEFAULT_ROUTES)} pairings — this report does"
              " NOT show the two pairings side by side, so it cannot say whether a proposal for one"
              " would have been adequate for the other.")

    summary = []
    for route in routes:
        rows = []
        for fixture in fixtures:
            # gate.fails is module state shared with the gate's own report(); the probe reads it as
            # "did a side fail" and clears it, so one bad fixture cannot tint the next one's answer.
            del gate.fails[:]
            del gate.notes[:]
            d = measure_side(args, fixture, route, work)
            if d is None:
                print(f"PROBE INCOMPLETE: {fixture}/{route} could not be measured")
                return 1
            rows.append(d)
        # The band printed beside the measurement is the one THAT pairing is judged by, not the one
        # constant: `wave_row_1cos_band()` reads the route module state `measure_side` just set, so the
        # `lt` leg reports against the hipBLASLt pairing's line and the hipBLAS leg against its own.
        # Printing 2e-03 on the `lt` leg is what made the first run of this probe read "OUTSIDE" about a
        # band that was never meant for it.
        summary.append(print_leg(route, rows, gate.wave_row_1cos_band()[0], a.margin,
                                 a_spec_for(route, a)))

    print("\n=== side by side ===")
    for s in summary:
        print(f"  route {s['route']:8s} worst per-row 1-cos {s['max']:.4e}"
              f"  ({s['fixtures']} fixtures, {s['rows']} rows)  proposed {s['proposed']:.3e}")
    print("  The two legs differ by which library sits on the wide side of the pair, and nothing else:"
          " same prompts, same replay, same rows, same statistic. A single band over both is only"
          " honest if it is the max of both rows above; a per-route band is honest about the tighter"
          " one and is what engine/gate.py's CASCADE_BAND carries.")
    print("  PROBE COMPLETE"
          f" planned={len(routes) * len(fixtures)} measured={len(summary) * len(fixtures)}"
          " — no timing taken, and none of these lines is one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
