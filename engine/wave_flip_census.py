#!/usr/bin/env python3
"""Which rows flip their argmax between two widths of one route, bucketed by the room they had.

CPU-only — it reads logits dumps a device run already made and starts no engine:

    # one explicit pair (e.g. the two sides of section (h)'s batched-vs-carry comparison)
    .venv/bin/python engine/wave_flip_census.py --pair tmp/w/ab_batched tmp/w/ab_carry \
        --label batched --label carry

    # every fixture of a `engine/wave_band_floor_probe.py` run, i.e. batched(512) vs one-row(1)
    .venv/bin/python engine/wave_flip_census.py --probe-work tmp/wave_band_floor --route lt

WHY THIS EXISTS AND NOT A GREP OVER THE GATE LOG. The gate's flip rule (`flip_admissible`) admits a
flip when the smaller of the two sides' own top-1/top-2 gaps is under `CASCADE_BAND.flip_room_max`
= 0.10, and that 0.10 is a MEASURED number: §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md
bucketed 1694 hand-vs-hipBLAS rows and found 0 of the 1566 with ≥ 0.10 of room flipping, with all 24
flips under it. A single flip at or above the line therefore does not by itself say the line is wrong —
but it does say the pairing being judged now is not the pairing the line was measured on, and the
honest instrument is the same census re-run on the new pairing, not a bump of the constant. That
census is what this prints: rows per room bucket, flips per bucket, and every flip at or over the line
named by row, id and both gaps.

ONE DEFINITION, IMPORTED. The room quantity, the bucket edges and the argmax rule all come from
`tools/route_c_audit.py` (`logit_row_decision`, `BUCKETS`) — the same code that produced the table the
0.10 line is derived from — and the threshold comes from `engine/gate.py`'s `CASCADE_BAND`. A census
that re-derived any of the three could disagree with the rule it is auditing and be green while doing
so, which is the trap `prototype/attn_tile_model.py`'s `compare` records for the attention mirror. Note
the pairing here has no reference side: `room` is the min of the two sides' own gaps, symmetric, and
that is exactly how §A2 defined it.

ROOM IN TWO UNITS, AND WHY BOTH. The line is an ABSOLUTE top-1/top-2 difference (0.10 of a logit), but a
logit gap means something only relative to the magnitude it sits at: 0.10 is a fraction of a BF16 step
at 6 and forty steps at 40 (`engine/gate.py`'s own `bf16_ulp` says "0.125 is nothing at 40 and two whole
ULP at 15.6"). So the census prints the §A2 buckets in logits AND the same rows bucketed by room
measured in BF16 ULPs of the row's own winner. The second column is not a rival threshold — this file
moves nothing — it is the quantity a re-derivation of the line would have to be asked in, because a
threshold that cannot tell "one ULP apart at logit 20" from "one ULP apart at logit 0.5" is a
magnitude proxy, not a decision rule.

NO TIMINGS. Nothing here measures anything; it is float arithmetic on files (`AGENTS.md`: a worker's
device run takes no timings, and this takes not even a device).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import route_c_audit  # the frozen decision/room definitions the 0.10 line came out of
import gate           # CASCADE_BAND.flip_room_max — the line being audited, read not retyped

ROW_RE = re.compile(r"logits(\d+)\.bin$")
# Room in BF16 ULPs of the row's own top-1 magnitude. Edges are ULP counts, not logits: 1 ULP is the
# floor of what a reassociating route can legally do to a decision, so [0,1) is "the two candidates are
# inside one representable step", and the wide end is "no rounding story reaches this".
ULP_BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf"))]


def room_in_ulp(row_a: np.ndarray, row_b: np.ndarray) -> float:
    """min over the two sides of (that side's top-1/top-2 gap ÷ one BF16 step at that side's top-1).

    Same `min` shape as §A2's `room` — a row is a coin toss if EITHER side called it close — just with
    each side's gap divided by the step size that side was writing at. `gate.bf16_ulp` is the same
    step function section (b) states a near-tie in, so no third definition of ULP enters the tree.
    """
    ratios = []
    for v in (row_a, row_b):
        _i, top, gap = gate.top2(v)
        step = gate.bf16_ulp(top)
        ratios.append(gap / step if step and step == step else float("inf"))
    return min(ratios)


def row_indices(d: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in d.glob("logits*.bin"):
        m = ROW_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def show_rows(name: str, rows: list[int], a_dir: Path, b_dir: Path, labels: tuple[str, str]) -> None:
    """The decision itself, printed rather than asserted, for the named rows of one pair.

    A room figure in a gate line is one number derived from two vectors, and the question a red row
    raises is never "is 0.1248 above 0.10" but "what was the decision, and how close behind it was
    the runner-up". So: both sides' top-3 ids and values, each side's own top-1/top-2 gap, and the
    room. This is the same arithmetic as the census above (same `top2`, same `min`), just not collapsed
    into a bucket.
    """
    ia, ib = row_indices(a_dir), row_indices(b_dir)
    for i in rows:
        if i not in ia or i not in ib:
            print(f"  {name} row {i}: not dumped on both sides, nothing to show")
            continue
        print(f"  {name} row {i} — top-3, per side (id:logit)")
        for label, d, idx in ((labels[0], a_dir, ia), (labels[1], b_dir, ib)):
            v = route_c_audit.read_raw(idx[i], "F32")
            top_id, top_v, gap = gate.top2(v)
            order = np.argsort(-v)[:3]
            t3 = " ".join(f"{int(j)}:{v[j]:.6f}" for j in order)
            second = float(np.sort(v)[-2])
            print(f"    {label:9s} top-1 {top_id} ({top_v:.6f}), top-2 {second:.6f},"
                  f" own gap {gap:.6f} | top-3: {t3}")
        va = route_c_audit.read_raw(ia[i], "F32")
        vb = route_c_audit.read_raw(ib[i], "F32")
        dec = route_c_audit.logit_row_decision(vb, va)
        room = min(dec["gap_ref"], dec["gap_got"])
        print(f"    decision: {labels[0]} {dec['argmax_ref']} vs {labels[1]} {dec['argmax_got']}"
              f" — {'SAME' if dec['argmax_match'] else 'FLIPPED'}, room {room:.6f}"
              f" vs the {gate.CASCADE_BAND['flip_room_max']:.2f} line"
              + (" — ABOVE" if room >= gate.CASCADE_BAND["flip_room_max"] else " — inside"))


def census_pair(name: str, a_dir: Path, b_dir: Path, labels: tuple[str, str],
                show: list[int] = ()) -> dict | None:
    """One pair of corpora → the bucket table, the flips at or over the line, and the raw flip list.

    A row present on only one side is a REFUSAL, not a smaller denominator: the census' whole claim is
    about a fixed population of rows, and silently averaging over the rows that happened to be dumped
    twice is the empty-must-catch-set trap one layer over (`prototype/attn_tile_model.py`,
    `prototype/gdn_collapse/results.md` §1 — a check whose input list can shrink cannot fail).
    """
    ia, ib = row_indices(a_dir), row_indices(b_dir)
    only_a, only_b = sorted(set(ia) - set(ib)), sorted(set(ib) - set(ia))
    if only_a or only_b:
        print(f"REFUSE {name}: the two sides did not dump the same rows"
              f" (only in {labels[0]}: {len(only_a)} {only_a[:5]},"
              f" only in {labels[1]}: {len(only_b)} {only_b[:5]}) — a census over the intersection"
              " would be a claim about a population that was never compared")
        return None
    rows = sorted(ia)
    if not rows:
        print(f"REFUSE {name}: no rows at all in {a_dir} / {b_dir}")
        return None
    buckets: dict[tuple[float, float], list[int]] = {k: [0, 0] for k in route_c_audit.BUCKETS}
    ulp_buckets: dict[tuple[float, float], list[int]] = {k: [0, 0] for k in ULP_BUCKETS}
    flips, wide = [], []
    for i in rows:
        va = route_c_audit.read_raw(ia[i], "F32")
        vb = route_c_audit.read_raw(ib[i], "F32")
        dec = route_c_audit.logit_row_decision(vb, va)   # (got, ref); `room` is min(), so symmetric
        room = min(dec["gap_ref"], dec["gap_got"])
        key = next(k for k in route_c_audit.BUCKETS if k[0] <= room < k[1])
        buckets[key][0] += 1
        # The same row in the other unit: each side's own gap over one BF16 step at that side's own
        # top-1, then the same min(). Two tables over one pass, because they are the same rows.
        r_ulp = room_in_ulp(va, vb)
        ukey = next(k for k in ULP_BUCKETS if k[0] <= r_ulp < k[1])
        ulp_buckets[ukey][0] += 1
        if not dec["argmax_match"]:
            buckets[key][1] += 1
            ulp_buckets[ukey][1] += 1
            flips.append((i, room, r_ulp, dec))
            if room >= gate.CASCADE_BAND["flip_room_max"]:
                wide.append((i, room, r_ulp, dec))
    print(f"\n=== {name}: {labels[0]} {a_dir} vs {labels[1]} {b_dir} — {len(rows)} rows"
          f" compared, {len(flips)} argmax flip(s), {len(wide)} of them at or over the"
          f" {gate.CASCADE_BAND['flip_room_max']:.2f} coin-toss line ===")
    for (lo, hi), (n, f) in buckets.items():
        hi_txt = "999" if hi == float("inf") else f"{hi:.2f}"
        print(f"  room [{lo:>5.2f},{hi_txt:>5}): {n:5d} rows, {f:3d} flipped"
              f" = {100 * f / n if n else 0.0:5.1f}%")
    print(f"  the same {len(rows)} rows, room in BF16 ULPs of each side's own top-1:")
    for (lo, hi), (n, f) in ulp_buckets.items():
        hi_txt = "999" if hi == float("inf") else f"{hi:.0f}"
        print(f"  room [{lo:>4.0f},{hi_txt:>4}) ULP: {n:5d} rows, {f:3d} flipped"
              f" = {100 * f / n if n else 0.0:5.1f}%")
    for i, room, r_ulp, dec in flips:
        over = " ≥ LINE" if room >= gate.CASCADE_BAND["flip_room_max"] else ""
        print(f"  row {i}: {labels[0]} {dec['argmax_ref']} (gap {dec['gap_ref']:.6g}) /"
              f" {labels[1]} {dec['argmax_got']} (gap {dec['gap_got']:.6g})"
              f" — room {room:.6g} = {r_ulp:.2f} BF16 ULP{over}")
    if show:
        show_rows(name, list(show), a_dir, b_dir, labels)
    return {"name": name, "rows": len(rows), "flips": len(flips), "wide": len(wide),
            "buckets": {f"{lo}-{hi if hi != float('inf') else 'inf'}": v
                        for (lo, hi), v in buckets.items()},
            "ulp_buckets": {f"{lo:.0f}-{hi:.0f}" if hi != float("inf") else f"{lo:.0f}-inf": v
                            for (lo, hi), v in ulp_buckets.items()},
            "wide_rows": [i for i, _, _, _ in wide]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", nargs=2, metavar=("A_DIR", "B_DIR"),
                    help="one pair of dump directories to compare, row file by row file")
    ap.add_argument("--label", action="append", default=[],
                    help="labels for the two sides of --pair (--label batched --label carry)")
    ap.add_argument("--probe-work",
                    help="a `engine/wave_band_floor_probe.py` work root: pairs each fixture's"
                         " ab_batched side against its ab_perrow side (the two sides the probe priced"
                         " the row band from)")
    ap.add_argument("--route", default="lt", help="which route's legs of a probe run to census")
    ap.add_argument("--fixtures", default="", help="restrict a --probe-work census to a comma list")
    ap.add_argument("--show-rows", default="",
                    help="comma list of row indices whose top-3, per side, get printed after the"
                         " table — the decision behind a room figure, visible instead of asserted")
    a = ap.parse_args()
    if not a.pair and not a.probe_work:
        raise SystemExit("census: give --pair A B or --probe-work DIR")

    results = []
    planned = 0
    show = [int(x) for x in a.show_rows.split(",") if x.strip() != ""]
    if a.pair:
        planned += 1
        labels = (a.label + ["a", "b"])[:2]
        r = census_pair("pair", Path(a.pair[0]), Path(a.pair[1]), (labels[0], labels[1]), show)
        if r is not None:
            results.append(r)
    if a.probe_work:
        root = Path(a.probe_work)
        fixtures = [f for f in a.fixtures.split(",") if f] or sorted(
            p.name for p in root.iterdir() if p.is_dir())
        planned += len(fixtures)
        for fx in fixtures:
            a_dir = root / fx / a.route / "ab_batched"
            b_dir = root / fx / a.route / "ab_perrow"
            if not (a_dir.is_dir() and b_dir.is_dir()):
                # Refused, not fatal: every other fixture still gets its table, and the
                # planned/measured count at the end is what makes the run exit non-zero. A loop that
                # returns on the first missing pair reports a census of the fixtures before it.
                print(f"REFUSE probe census: {fx}/{a.route} has no pair to compare"
                      f" ({a_dir}, {b_dir}) — the aggregate below will not be a census of the"
                      " requested fixtures")
                continue
            r = census_pair(fx, a_dir, b_dir, ("batched", "one-row"), show)
            if r is None:
                continue
            results.append(r)

    rows = sum(r["rows"] for r in results)
    flips = sum(r["flips"] for r in results)
    wide = sum(r["wide"] for r in results)
    agg: dict[str, list[int]] = {}
    for r in results:
        for k, (n, f) in r["buckets"].items():
            acc = agg.setdefault(k, [0, 0])
            acc[0] += n
            acc[1] += f
    print(f"\n=== aggregate over {len(results)} pair(s): {rows} rows, {flips} flips,"
          f" {wide} at or over the line ===")
    for k in route_c_audit.BUCKETS:
        key = f"{k[0]}-{k[1] if k[1] != float('inf') else 'inf'}"
        n, f = agg.get(key, [0, 0])
        hi_txt = "999" if k[1] == float("inf") else f"{k[1]:.2f}"
        print(f"  room [{k[0]:>5.2f},{hi_txt:>5}): {n:5d} rows, {f:3d} flipped"
              f" = {100 * f / n if n else 0.0:5.1f}%")
    print(f"  the line under audit: CASCADE_BAND.flip_room_max ="
          f" {gate.CASCADE_BAND['flip_room_max']:.2f}, measured by §A2 of progress/2026-10-14/"
          "route-c-audit-2026-10-14.md on the hand<->hipBLAS pairing. Flips at or over it are what a"
          " re-derivation would have to account for — this file reports them, it does not move them.")
    for r in results:
        if r["wide"]:
            print(f"  {r['name']}: rows at or over the line: {r['wide_rows']}")
    print("  CENSUS COMPLETE"
          f" planned={planned} measured={len(results)} — no timing taken, and none of these lines"
          " is one" + ("" if planned == len(results) else
           " — MISMATCH: a pair was asked for and produced no table, so the aggregate above is not a"
           " census of what was requested"))
    if planned != len(results):
        print("  A census whose pair list shrank cannot fail; the count is derived from the request,"
              " not from what came back (`prototype/gdn_collapse/results.md` §1).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
