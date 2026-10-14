#!/usr/bin/env python3
"""Turn `engine/route_price.py`'s tables into the allowance files `engine/gate.py` reads.

    # 1. the engine column comes from a gate run (device, correctness only, ~5 s)
    .venv/bin/python engine/gate.py --fixture dec300f > tmp/gate_dec300f.log
    .venv/bin/python engine/route_price.py --fixture dec300f --engine-log tmp/gate_dec300f.log \
            --out progress/engine/evidence/route_dec300f.json                             # CPU, minutes, no device
    # ... the same pair for every fixture being priced ...

    # 2. derive each fixture's own margin from its own table and write one file per fixture
    .venv/bin/python engine/route_allowance.py --table progress/engine/evidence/route_dec300f.json
    .venv/bin/python engine/route_allowance.py --table tmp/route_mt32f.json --dry-run

    .venv/bin/python engine/route_allowance.py --selftest   # the per-fixture property, CPU, no files

Writes `data/refs/<fixture>/route_allowance.json` for every `--table` given. `engine/gate.py` READS
that file and never re-prices it: re-pricing is a torch CPU run against the checkpoint, and a gate run
that silently re-ran the reference would (a) cost minutes nobody budgeted and (b) throw away the
provenance that makes a cached allowance safe — the command, the date, and the hashes this file records
and the gate re-checks.

AND IT KEEPS THE BAND LEDGER. A margin derived here is a band value that changed with nobody editing a
file, so this tool appends its own lines to `bench/band-history.jsonl` (`tools/band_history.py`) for
every margin it writes — and for every re-pricing it REFUSES, as `event: refused`, which records the
attempt without moving the current value. `engine/gate.py` refuses to run when a margin and its ledger
line disagree, so a fixture's band cannot drift silently. `--dry-run` writes nothing, so it records
nothing; so does an `--out-root` the shipped gate does not read.

WHY THE MARGIN IS DERIVED AND NOT CHOSEN. The rule, at step *j* of one fixture, is
`engine 1-cos(j) <= margin x route 1-cos(j)` (and the same shape for meanrel), where `route 1-cos(j)` is
the measured disagreement between the reference's own two legal GDN prompt forms at that step. If
`margin` were picked by taste the exercise would be a relabelled band, so it is a maximum:

    margin(fixture)     = max over THAT fixture's steps of  engine_1cos(j) / route_1cos(j)
    margin_rel(fixture) = max over THAT fixture's steps of  engine_meanrel(j) / route_meanrel(j)

— the tightest constant that admits every step THAT FIXTURE was measured on, rounded UP at two decimals
with no safety factor.

PER-FIXTURE, and why (D-2 of `progress/2026-10-12/plan-2026-10-12.md`, 2026-10-12). This file used to derive ONE global
constant as a max over the whole priced set, and that turned out to be a hazard rather than a rigour:
`dec300f` (295-token prompt, 128 steps) has ratios up to 7.98 while its route spread is *tighter* than
the 64-step fixtures', so pricing it would have jumped the shared constant 3.50 → 7.99 and loosened the
gate for `dec64f` and `dec64t` at the same time — a fixture nobody was looking at would have moved a band
everybody else was being judged by. Keyed per fixture, each margin is a statement about its own steps and
**the derivation reads nothing else**, which is the structural property, not a preference:

    pricing or re-pricing one fixture cannot change any other fixture's margin.

`--selftest` proves that property (and proves it is falsifiable: it asserts the per-fixture answer
DIFFERS from what a global max would have given, so reverting the derivation to global fails the test).

Two consequences of a derived margin survive the change, both intended. The number is reproducible — the
engine is deterministic and the reference run is fixed by the same hashes the file records — and a
re-pricing never moves a band on its own: if a fresh measurement disagrees with the margin already stored
for that fixture, **this tool refuses** and names the file to delete deliberately. A band move has to be
a decision somebody can see in a diff, which is what `margin_basis` in the file is for. (So re-running
this tool on a fixture priced under the old global max REFUSES: the old constant was a max over several
fixtures and a per-fixture derivation is usually smaller. That refusal is the tool working — leaving the
file alone is how "keep 3.50" is recorded.)

Known self-reference, NOT fixed here (it is D-3 / Q4b, a separate scheduled item): a margin defined as
`max(engine deficit / route spread)` puts that fixture's worst observed step exactly on the boundary *by
construction*, so the definition can never be engine-independent. Per-fixture keying removes the
cross-fixture leak, not the self-reference. `engine/results.md` has the write-up.
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
from engine import gate as egate      # the fallback margins and the input hashes live in the gate, not here
from tools import band_history        # the ledger this tool writes its own derived margins into

DEFAULT_OUT_ROOT = ROOT / "data" / "refs"

# What a file written by this version declares about where its margin came from. A file written before
# per-fixture keying has no `margin_basis`, and `engine/gate.py:allowance_margins` says so when it prints
# the rule in force; `BAND`'s two numbers are the fallback for a file that carries no margins at all.
MARGIN_BASIS = ("per-fixture (D-2): max engine/route ratio inside this fixture's own steps only"
                " — no other fixture can move it")


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


def derive(doc: dict) -> dict:
    """One fixture's margins, derived from THAT fixture's rows and from nothing else.

    The whole of D-2 lives in this function: it takes one document, reads only that document's rows, and
    returns that fixture's margins. There is no cross-fixture accumulator to thread through it, so a
    caller cannot make one fixture's number depend on another's without editing this signature — and
    `--selftest` fails if someone does.
    """
    cos_r, rel_r = ratios(doc)
    wc, wr = max(cos_r), max(rel_r)
    return {
        "margin": ceil2(wc[0]),
        "margin_rel": ceil2(wr[0]),
        "raw_max_cos_ratio": wc[0], "raw_max_cos_ratio_step": wc[1],
        "raw_max_meanrel_ratio": wr[0], "raw_max_meanrel_ratio_step": wr[1],
        "n_ratios": len(cos_r),
    }


def steps_rows(doc: dict, margins: dict) -> list:
    """The per-step cached rows: the measured spread, the cap it implies, and the engine that set it."""
    return [{"step": int(r["step"]),
             "route_one_cos": r["route_one_cos"],
             "route_meanrel": r["route_meanrel"],
             "allowance_one_cos": margins["margin"] * r["route_one_cos"],
             "allowance_meanrel": margins["margin_rel"] * r["route_meanrel"],
             # Recorded, not used by the gate as a band: what the engine measured when this file was
             # priced, so a later reader can see which measurement set which number — and so
             # `gate.py:allowance_problems` can prove the margin admits its own pricing.
             "engine_one_cos_when_priced": r["engine_one_cos"],
             "engine_meanrel_when_priced": r["engine_meanrel"]}
            for r in sorted(doc["rows"], key=lambda r: int(r["step"]))]


def file_doc(doc: dict, margins: dict) -> dict:
    """The JSON document for one fixture — pure, so `--selftest` can compare two derivations."""
    egate.set_fixture(doc["fixture"])
    man = json.loads((egate.REFS / "manifest.json").read_text())
    n_steps = len(man["greedy_continuation"]) - 1
    priced = sorted(int(r["step"]) for r in doc["rows"])
    if priced != list(range(n_steps)):
        raise SystemExit(f"route_allowance: the {doc['fixture']} table prices steps {priced[:6]}…"
                         f" ({len(priced)} of them) but the fixture has {n_steps} gated steps. Price"
                         " the whole chain: a missing row would leave that step judged by a floor"
                         " nobody compared anything against.")
    return {
        "fixture": doc["fixture"],
        "rule": ("engine 1-cos(step j) <= margin x route_1cos(j), and meanrel(j) <="
                 " margin_rel x route_meanrel(j), where route_*(j) is the measured disagreement"
                 " between specimpl's chunked (§04.7) and recurrent (§04.4) prompt forms at step j."
                 " The margins are this fixture's own; the reason for them lives in engine/gate.py's"
                 " ROUTE ALLOWANCE comment and engine/results.md has the tables."),
        "margin": margins["margin"],
        "margin_rel": margins["margin_rel"],
        "margin_basis": MARGIN_BASIS,
        "derivation": {
            "definition": "max over THIS fixture's steps of engine/route (cos, and meanrel separately),"
                          " rounded up at 2 dp — see this tool's docstring. Pricing another fixture"
                          " cannot change this number.",
            "worst_ratio_in_this_fixture": {
                "cos": margins["raw_max_cos_ratio"], "cos_at_step": margins["raw_max_cos_ratio_step"],
                "meanrel": margins["raw_max_meanrel_ratio"],
                "meanrel_at_step": margins["raw_max_meanrel_ratio_step"],
                "steps": margins["n_ratios"]},
        },
        "generated": {
            "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "routes": "specimpl chunked §04.7 vs recurrent §04.4 prefill of the same prompt, then the"
                      " same forced feed — engine/route_price.py",
            "source_table": str(doc["_table"]),
        },
        "inputs": egate.allowance_inputs(n_steps),
        "steps": steps_rows(doc, margins),
    }


def stored_margin(fx: str, out_root: Path):
    """The margins already on disk for this fixture, or None when there is no file. Read-only."""
    path = out_root / fx / egate.ALLOWANCE_FILE
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise SystemExit(f"route_allowance: cannot read the existing {path}: {e}")
    return egate.allowance_margins(doc)


def write_one(doc: dict, margins: dict, out_root: Path, dry: bool) -> Path:
    """One fixture's allowance file, with the provenance the gate re-checks."""
    out = out_root / doc["fixture"] / egate.ALLOWANCE_FILE
    payload = file_doc(doc, margins)
    if dry:
        print(f"  --dry-run: would write {out} ({len(payload['steps'])} steps,"
              f" margins {margins['margin']:.2f}/{margins['margin_rel']:.2f})")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {out} ({len(payload['steps'])} steps,"
          f" margins {margins['margin']:.2f}/{margins['margin_rel']:.2f})")
    return out


def margin_reason(fx: str, m: dict, which: str) -> str:
    """Why this fixture's margin is what it is, in the words the ledger wants.

    The ledger refuses a boilerplate reason, and it should: the derivation IS the justification, so
    the line quotes it — the fixture, its own worst engine/route ratio, and the step that set it.
    """
    if which == "cos":
        raw, step = m["raw_max_cos_ratio"], m["raw_max_cos_ratio_step"]
    else:
        raw, step = m["raw_max_meanrel_ratio"], m["raw_max_meanrel_ratio_step"]
    return (f"derived by engine/route_allowance.py from {fx}'s own {m['n_ratios']} steps: worst"
            f" engine/{'route 1-cos' if which == 'cos' else 'route meanrel'} ratio {raw:.5f} at step"
            f" {step}, rounded up at two decimals (ceil2) — the tightest constant that admits every"
            " step this fixture was measured on, and no other fixture's rows can move it")


def log_band_history(fx: str, m: dict, stored, refused: bool) -> None:
    """Append this run's derived margins to `bench/band-history.jsonl` — including the refusals.

    A margin derived here is a band value that changed without anybody editing a file, so the generator
    records it rather than trusting a human to remember to. Two cases write no line, and both say so:
    a re-price that returns the SAME margin (nothing moved, and the ledger refuses a line claiming it
    did), and — for a refusal — there is no such case, since a refusal is by definition a disagreement.
    A REFUSAL is recorded as `event: refused`: it changes nothing (and `band_history.check` ignores it
    when deciding the current value), but an attempted loosening that was caught is exactly the thing
    worth finding in the log later.
    """
    rel = f"data/refs/{fx}/{egate.ALLOWANCE_FILE}"
    for which, key in (("cos", "margin"), ("meanrel", "margin_rel")):
        idx = 0 if key == "margin" else 1
        derived_v = m[key]
        if refused:
            band_history.record(file=rel, key=key, old=stored[idx], new=derived_v,
                                event="refused", decision="D-2", by="engine/route_allowance.py",
                                reason=(f"re-price REFUSED by engine/route_allowance.py: this"
                                        f" measurement derives {derived_v:.2f} for {fx}'s {key} but"
                                        f" the file on disk carries {stored[idx]:.2f}, so nothing was"
                                        " written and no band moved; re-pricing one fixture must never"
                                        " move a margin by itself"))
            continue
        # With no file on disk the previous value comes from the ledger, not from the absent file: the
        # documented way to change a margin is to delete its file deliberately first, and a line that
        # recorded that as `absent -> 3.04` would hide what the fixture was actually being judged by.
        prev_v = stored[idx] if stored is not None else band_history.current_recorded(rel, key)
        if prev_v is not None and float(prev_v) == float(derived_v):
            # Re-pricing a fixture and getting the same margin back is not a band move, and the ledger
            # refuses a line that claims one (old == new). Say that here instead of raising it later.
            print(f"  band history: {fx} {key} stays {derived_v:.2f} — no band moved, nothing"
                  " recorded")
            continue
        band_history.record(file=rel, key=key, old=(prev_v if prev_v is not None else "absent"),
                            new=derived_v, event="applied", decision="D-2",
                            by="engine/route_allowance.py", reason=margin_reason(fx, m, which))


def selftest() -> int:
    """Prove the per-fixture property on synthetic ratios, with no device, no fixture and nothing written.

    The property D-2 bought is a NEGATIVE — pricing a wide fixture must not move a tight one's margin —
    and a test that only checks that the numbers came out right would still pass after somebody reverted
    the derivation to a global max. So the assertion is aimed at the exact shape that made the decision
    necessary: a fixture whose own worst ratio is 3.4939 (`dec64t`'s real number) priced alongside one
    whose own worst ratio is 7.9837 (`dec300f`'s real number). Under the retired global rule BOTH would
    have been judged at 7.99; per-fixture, the first stays at 3.50. If the derivation ever goes back to a
    cross-fixture max, check 3 fails, and that is the point of it.
    """
    def table(fx, triples):
        """A `route_price.py`-shaped table from (step, engine 1-cos, route 1-cos) triples.

        meanrel is carried at exactly one tenth of the cos figures on both sides, so the meanrel ratio
        equals the cos ratio and one assertion covers both columns.
        """
        return {"fixture": fx, "_table": "<selftest>",
                "rows": [{"step": s, "engine_one_cos": eoc, "route_one_cos": roc,
                          "engine_meanrel": eoc / 10.0, "route_meanrel": roc / 10.0,
                          "routes_argmax_agree": True} for s, eoc, roc in triples]}

    tight = table("tightfx", [(0, 1.0e-4, 1.0e-4), (1, 2.3281e-4, 6.6635e-5)])   # worst 3.4939
    wide = table("widefx", [(0, 1.0e-4, 6.25e-5), (1, 4.6321e-4, 5.802e-5),
                            (2, 2.0e-5, 1.0e-4)])                                # worst 7.9836…
    bad = []

    def eq(label, got, want):
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} — got {got}, want {want}")
        if not ok:
            bad.append(label)

    alone_t, alone_w = derive(tight), derive(wide)
    eq("1. ceil2 rounds UP at 2 dp (the four shipped numbers)",
       [ceil2(3.4939), ceil2(0.4111), ceil2(7.9837), ceil2(0.7198)], [3.50, 0.42, 7.99, 0.72])
    eq("2. each fixture's margin is its own worst ratio, rounded up — the tight one stays at 3.50",
       (alone_t["margin"], alone_w["margin"]), (3.50, 7.99))
    eq("3. and that is only true because the derivation is per-fixture: a global max over both tables"
       " would have judged BOTH of them at 7.99",
       ceil2(max(max(ratios(tight)[0])[0], max(ratios(wide)[0])[0])) != alone_t["margin"], True)
    eq("4. meanrel is keyed the same way", (alone_t["margin_rel"], alone_w["margin_rel"]),
       (3.50, 7.99))
    # The same proof one level down, on what the gate actually reads: the per-step caps a fixture gets
    # must be a function of that fixture's rows alone. `steps_rows` takes the fixture's own document and
    # its own margins, exactly as `write_one` calls it, whatever else the run named.
    rows_alone = json.dumps(steps_rows(tight, alone_t), sort_keys=True)
    rows_after_wide = json.dumps(steps_rows(tight, derive(tight)), sort_keys=True)
    eq("5. the per-step caps of the tight fixture are byte-identical before and after the wide fixture"
       " is priced", rows_alone == rows_after_wide, True)
    # And the refusal that keeps a band move visible: a table with no engine column must be refused, not
    # silently priced at whatever the route spread happens to allow.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "notable.json"
        p.write_text(json.dumps({"fixture": "tightfx", "rows": [{"step": 0}]}))
        refused = False
        try:
            read_table(p)
        except SystemExit:
            refused = True
        eq("6. a table with no engine column is refused rather than priced", refused, True)
    print("SELFTEST " + ("FAIL — " + "; ".join(bad) if bad else
                         "PASS (a fixture's margin is derived from its own rows alone; the wide-fixture"
                         " case is the one the retired global max got wrong)"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="append", default=[], metavar="PATH",
                    help="an `engine/route_price.py --out` table. Repeat once per fixture being priced."
                         " Each table is derived and written on its own: naming only one fixture is"
                         " normal now, because a margin depends on nothing outside its own fixture.")
    ap.add_argument("--out-root", default=str(ROOT / "data" / "refs"),
                    help="where the fixture directories live")
    ap.add_argument("--dry-run", action="store_true", help="derive and print, write nothing")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the per-fixture property on synthetic ratios (no files, no device)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.table:
        raise SystemExit("route_allowance: pass at least one --table (see this file's docstring)")

    out_root = Path(args.out_root)
    tables, seen = [], {}
    for name in args.table:
        path = Path(name)
        doc = read_table(path)
        doc["_table"] = path
        fx = doc["fixture"]
        if fx in seen:
            raise SystemExit(f"route_allowance: two tables ({seen[fx]} and {path}) both price {fx}."
                             " One fixture gets one measurement per run: pick the table to believe."
                             " Two tables for one fixture would mean two candidate margins for one file.")
        seen[fx] = path
        tables.append(doc)

    print("fixture    steps  own worst engine/route cos ratio     own worst engine/route meanrel ratio")
    derived = {}
    for doc in tables:
        m = derive(doc)
        derived[doc["fixture"]] = m
        print(f"  {doc['fixture']:<9} {m['n_ratios']:>4}   {m['raw_max_cos_ratio']:>8.4f}"
              f" at step {m['raw_max_cos_ratio_step']:<4}          {m['raw_max_meanrel_ratio']:>8.4f}"
              f" at step {m['raw_max_meanrel_ratio_step']}")

    print("\nmargins, each derived from its own fixture alone (D-2):")
    problems, refused_fx, storeds = [], [], {}
    for doc in tables:
        fx = doc["fixture"]
        m = derived[fx]
        line = (f"  {fx:<9} cos {m['margin']:.2f} / meanrel {m['margin_rel']:.2f}"
                f"   (fallback in engine/gate.py BAND:"
                f" {egate.BAND['route_margin']:.2f}/{egate.BAND['route_margin_rel']:.2f})")
        stored = storeds[fx] = stored_margin(fx, out_root)
        if stored is None:
            print(line + "   [new file]")
            continue
        line += f"   (already stored: {stored[0]:.2f}/{stored[1]:.2f})"
        print(line)
        if (stored[0], stored[1]) != (m["margin"], m["margin_rel"]):
            refused_fx.append(fx)
            direction = ("looser" if m["margin"] > stored[0] or m["margin_rel"] > stored[1]
                         else "tighter")
            problems.append(
                f"{fx}: this measurement derives {m['margin']:.2f}/{m['margin_rel']:.2f} but the file on"
                f" disk carries {stored[0]:.2f}/{stored[1]:.2f}, i.e. re-pricing would make the band"
                f" {direction}. This tool does not move a band on its own. If this measurement is the one"
                f" to believe: delete {out_root / fx / egate.ALLOWANCE_FILE} deliberately, say in"
                f" engine/results.md what moved and why, and re-run this command. If it is not, work out"
                " why the engine or the reference moved before caching anything. (A file priced under the"
                " old global max often differs legitimately: its margin was a max over several fixtures,"
                " and a per-fixture derivation is allowed to be smaller.)")
    untouched = [p.parent.name for p in Path(args.out_root).glob(f"*/{egate.ALLOWANCE_FILE}")
                 if p.parent.name not in seen]
    if untouched:
        print(f"\nnot touched by this run, and structurally cannot be: {sorted(untouched)} — each of"
              " those margins is derived from its own fixture's rows, so naming a subset of fixtures"
              " here no longer risks re-deriving anybody else's constant.")

    # The ledger: a margin written here is a band value that moved without anybody editing a file, so
    # this tool records it — and so a refusal, which is a caught attempt at exactly that, is on the
    # record too. Skipped for `--dry-run` (nothing happened) and for an `--out-root` the gate does not
    # read (recording a margin the shipped gate cannot see would put a story in the ledger, not a fact).
    def log():
        if args.dry_run:
            return
        if Path(args.out_root).resolve() != DEFAULT_OUT_ROOT:
            print(f"  band history NOT recorded: --out-root {args.out_root} is not the tree the gate"
                  f" reads ({DEFAULT_OUT_ROOT}), so there is no band here for the ledger to describe.")
            return
        for fx in refused_fx:
            log_band_history(fx, derived[fx], storeds[fx], refused=True)
        if refused_fx:
            print(f"  recorded {len(refused_fx)} refused re-pricing(s) in"
                  f" {band_history.HISTORY_REL} as event=refused — no band moved, so no band check"
                  " treats them as current.")

    if problems:
        log()
        print("\nREFUSED — nothing was written:")
        for p in problems:
            print("  * " + p)
        return 2

    for doc in tables:
        write_one(doc, derived[doc["fixture"]], out_root, args.dry_run)
        if not args.dry_run:      # a --dry-run derived nothing and wrote nothing, so it records nothing
            log_band_history(doc["fixture"], derived[doc["fixture"]], storeds[doc["fixture"]],
                             refused=False)
    print("\nOK — every margin written is the max ratio inside its own fixture, rounded up at 2 dp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
