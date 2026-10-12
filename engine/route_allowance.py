#!/usr/bin/env python3
"""Turn `engine/route_price.py`'s tables into the allowance files `engine/gate.py` reads.

    # 1. the engine column comes from a gate run (device, correctness only, ~5 s)
    .venv/bin/python engine/gate.py --fixture dec64f > tmp/gate_dec64f.log
    .venv/bin/python engine/route_price.py --fixture dec64f --engine-log tmp/gate_dec64f.log \
            --out tmp/route_dec64f.json                              # CPU, ~5 s, no device
    # ... the same pair for every forced fixture being priced ...
    .venv/bin/python engine/gate.py --fixture dec64t > tmp/gate_dec64t.log
    .venv/bin/python engine/route_price.py --fixture dec64t --engine-log tmp/gate_dec64t.log \
            --out tmp/route_dec64t.json

    # 2. derive the margins from those tables and write one file per fixture
    .venv/bin/python engine/route_allowance.py --table tmp/route_dec64f.json \
            --table tmp/route_dec64t.json

Writes `data/refs/<fixture>/route_allowance.json` for every `--table` given. `engine/gate.py` READS
that file and never re-prices it: re-pricing is a torch CPU run against the checkpoint, and a gate run
that silently re-ran the reference would (a) cost minutes nobody budgeted and (b) throw away the
provenance that makes a cached allowance safe — the command, the date, and the hashes this file records
and the gate re-checks.

WHY THE MARGIN IS DERIVED AND NOT CHOSEN. The rule is
`engine 1-cos(step j) <= margin x route 1-cos(step j)` (and the same shape for meanrel), where
`route 1-cos(j)` is the measured disagreement between the reference's own two legal GDN prompt forms at
that step. If `margin` were picked by taste the exercise would be a relabelled band, so it is a maximum:

    margin     = max over every priced fixture and every step j of  engine_1cos(j) / route_1cos(j)
    margin_rel = max over every priced fixture and every step j of  engine_meanrel(j) / route_meanrel(j)

— the tightest constant that admits every step ever measured — rounded UP at two decimals, with no
safety factor. Two consequences, both intended. The constant is reproducible (the engine is deterministic
and the reference run is fixed by the same hashes the file records), and it has no slack to absorb a
regression: an engine that gets worse moves this number, and moving it is a visible edit to
`engine/gate.py`'s `BAND` plus the numbers quoted in its ROUTE ALLOWANCE comment. This script
therefore REFUSES to write when the derived value disagrees with the constant in force: it can report a
measurement, it cannot quietly move a band.

Pass every priced fixture each time. The margin is a max over the whole priced set, so a run that names
only one fixture would derive a smaller constant than the one `engine/gate.py` carries — which is
exactly the kind of silent narrowing (or widening) this file exists to prevent, so it is an error here.
"""
from __future__ import annotations
import argparse
import datetime
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import gate as egate      # the margins and the input hashes live in the gate, not here


def ceil2(x: float) -> float:
    """Round UP at two decimals. Up, because a margin rounded down fails a step that was measured to
    pass it; the two-decimal grain is what makes the constant readable and re-derivable."""
    return math.ceil(round(x, 6) * 100) / 100.0


def read_table(path: Path) -> dict:
    """One `engine/route_price.py --out` table, with every reason not to trust it turned into an exit."""
    doc = json.loads(path.read_text())
    rows = doc.get("rows", [])
    if not rows:
        raise SystemExit(f"route_allowance: {path} has no rows")
    missing = [r["step"] for r in rows if r.get("engine_one_cos") is None
               or r.get("engine_meanrel") is None]
    if missing:
        raise SystemExit(f"route_allowance: {path} has no ENGINE column for step(s) {missing[:6]}"
                         f" ({len(missing)} of {len(rows)}). The margin is defined by the engine's own"
                         " deficit, so the table must come from"
                         " `engine/route_price.py --fixture X --engine-log <gate log> --out ...`")
    if any(r.get("routes_argmax_agree") is False for r in rows):
        raise SystemExit(f"route_allowance: {path} has a step where the two legal routes pick"
                         " different ids. That is a different question from a band, and it needs"
                         " looking at before any of this table is cached — gate.py's argmax rule is"
                         " hard and a route that disagrees with itself is not a spread to key on.")
    return doc


def ratios(doc: dict) -> tuple[list, list]:
    """(cos ratios, meanrel ratios) as (value, step) pairs for one fixture."""
    cos_r = [(r["engine_one_cos"] / r["route_one_cos"], r["step"]) for r in doc["rows"]]
    rel_r = [(r["engine_meanrel"] / r["route_meanrel"], r["step"]) for r in doc["rows"]]
    return cos_r, rel_r


def write_one(doc: dict, out_root: Path, dry: bool) -> Path:
    """One fixture's allowance file, with the provenance the gate re-checks."""
    fx = doc["fixture"]
    egate.set_fixture(fx)
    man = json.loads((egate.REFS / "manifest.json").read_text())
    n_steps = len(man["greedy_continuation"]) - 1
    priced = sorted(int(r["step"]) for r in doc["rows"])
    if priced != list(range(n_steps)):
        raise SystemExit(f"route_allowance: the {fx} table prices steps {priced[:6]}…"
                         f" ({len(priced)} of them) but the fixture has {n_steps} gated steps. Price"
                         " the whole chain: a missing row would leave that step judged by a floor"
                         " nobody compared anything against.")
    steps = [{"step": int(r["step"]),
              "route_one_cos": r["route_one_cos"],
              "route_meanrel": r["route_meanrel"],
              "allowance_one_cos": egate.BAND["route_margin"] * r["route_one_cos"],
              "allowance_meanrel": egate.BAND["route_margin_rel"] * r["route_meanrel"],
              # Recorded, not used by the gate: what the engine measured when this file was priced, so
              # a later reader can see which measurement set which number.
              "engine_one_cos_when_priced": r["engine_one_cos"],
              "engine_meanrel_when_priced": r["engine_meanrel"]}
             for r in sorted(doc["rows"], key=lambda r: int(r["step"]))]
    file_doc = {
        "fixture": fx,
        "rule": ("engine 1-cos(step j) <= margin x route_1cos(j), and meanrel(j) <="
                 " margin_rel x route_meanrel(j), where route_*(j) is the measured disagreement"
                 " between specimpl's chunked (§04.7) and recurrent (§04.4) prompt forms at step j."
                 " The margins and the reason for them live in engine/gate.py's ROUTE ALLOWANCE"
                 " comment; engine/results.md has the tables."),
        "margin": egate.BAND["route_margin"],
        "margin_rel": egate.BAND["route_margin_rel"],
        "derivation": {
            "definition": "max over every priced fixture and step of engine/route (cos, and meanrel"
                          " separately), rounded up at 2 dp — see this tool's docstring",
            "worst_ratio_by_fixture": doc.get("_worst", {}),
        },
        "generated": {
            "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "routes": "specimpl chunked §04.7 vs recurrent §04.4 prefill of the same prompt, then the"
                      " same forced feed — engine/route_price.py",
            "source_table": str(doc["_table"]),
        },
        "inputs": egate.allowance_inputs(n_steps),
        "steps": steps,
    }
    out = out_root / fx / egate.ALLOWANCE_FILE
    if dry:
        print(f"  --dry-run: would write {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(file_doc, indent=1))
    print(f"  wrote {out} ({len(steps)} steps)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="append", default=[], metavar="PATH",
                    help="an `engine/route_price.py --out` table. Repeat once per priced fixture — the"
                         " margin is a max over all of them, so naming a subset is refused.")
    ap.add_argument("--out-root", default=str(ROOT / "data" / "refs"),
                    help="where the fixture directories live")
    ap.add_argument("--dry-run", action="store_true", help="derive and print, write nothing")
    args = ap.parse_args()
    if not args.table:
        raise SystemExit("route_allowance: pass at least one --table (see this file's docstring)")

    tables = []
    for name in args.table:
        path = Path(name)
        doc = read_table(path)
        doc["_table"] = path
        tables.append(doc)

    cos_all, rel_all, per_fixture = [], [], []
    for doc in tables:
        cos_r, rel_r = ratios(doc)
        cos_all += cos_r
        rel_all += rel_r
        wc, wr = max(cos_r), max(rel_r)
        doc["_worst"] = {"max_cos_ratio": wc[0], "max_cos_ratio_step": wc[1],
                         "max_meanrel_ratio": wr[0], "max_meanrel_ratio_step": wr[1],
                         "steps": len(cos_r)}
        per_fixture.append((doc["fixture"], wc, wr, len(cos_r)))

    print("fixture   steps  worst engine/route cos ratio      worst engine/route meanrel ratio")
    for fx, wc, wr, n in per_fixture:
        print(f"  {fx:<8} {n:>4}   {wc[0]:>8.4f} at step {wc[1]:<3}            {wr[0]:>8.4f}"
              f" at step {wr[1]}")
    cos_max = max(v for v, _ in cos_all)
    rel_max = max(v for v, _ in rel_all)
    derived, derived_rel = ceil2(cos_max), ceil2(rel_max)
    print(f"\nderived from {len(cos_all)} steps across {len(tables)} fixture(s):")
    print(f"  cos:     max ratio {cos_max:.6f} -> ceil2 = {derived:.2f}"
          f"   (engine/gate.py BAND['route_margin'] = {egate.BAND['route_margin']:.2f})")
    print(f"  meanrel: max ratio {rel_max:.6f} -> ceil2 = {derived_rel:.2f}"
          f"   (engine/gate.py BAND['route_margin_rel'] = {egate.BAND['route_margin_rel']:.2f})")

    problems = []
    if derived != egate.BAND["route_margin"] or derived_rel != egate.BAND["route_margin_rel"]:
        problems.append(
            "the derived margins are "
            f"{derived:.2f}/{derived_rel:.2f} and the gate carries "
            f"{egate.BAND['route_margin']:.2f}/{egate.BAND['route_margin_rel']:.2f}. This tool does not"
            " edit a band: if this measurement is the one to believe, set both numbers in"
            " engine/gate.py's BAND, update the figures quoted in its ROUTE ALLOWANCE comment, then"
            " re-run this command. If it is not, work out why the engine moved before caching anything.")
    named = {d["fixture"] for d in tables}
    stale_dir = [p.parent.name for p in Path(args.out_root).glob(f"*/{egate.ALLOWANCE_FILE}")
                 if p.parent.name not in named]
    if stale_dir:
        problems.append(f"{sorted(stale_dir)} already carry an {egate.ALLOWANCE_FILE} but were not"
                        " priced in this run. The margin is a max over EVERY priced fixture, so pass"
                        " their tables too (or delete their files deliberately and say why).")
    if problems:
        print("\nREFUSED — nothing was written:")
        for p in problems:
            print("  * " + p)
        return 2

    out_root = Path(args.out_root)
    for doc in tables:
        write_one(doc, out_root, args.dry_run)
    print("\nOK — the cached margins and engine/gate.py's BAND agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
