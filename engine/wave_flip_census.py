#!/usr/bin/env python3
"""Which rows flip their argmax between two widths of one route, bucketed by the room they had.

CPU-only — it reads logits dumps a device run already made and starts no engine:

    # one explicit pair (e.g. the two sides of section (h)'s batched-vs-carry comparison)
    .venv/bin/python engine/wave_flip_census.py --pair tmp/w/ab_batched tmp/w/ab_carry \
        --label batched --label carry

    # every fixture of a `engine/wave_band_floor_probe.py` run, i.e. batched(512) vs one-row(1)
    .venv/bin/python engine/wave_flip_census.py --probe-work tmp/wave_band_floor --route lt

    # every fixture x route x pairing of a directory of gate work dirs, with the retired absolute line
    # printed beside the ULP one so the colour changes are named (this is how §3f's table was made)
    .venv/bin/python engine/wave_flip_census.py --work-root tmp/flipc --retired-line 0.10 \
        --fixtures dec8,dec64f,dec64t,chunk64b,p128,mt32f,wide520f,wide640f,wide1024f \
        --routes library,lt

WHY THIS EXISTS AND NOT A GREP OVER THE GATE LOG. The gate's flip rule admits a flip when the room is
under `CASCADE_BAND.flip_room_max_ulp` BF16 ULP at the row's own top-1 — since 2026-10-16 a ULP line,
before it a flat 0.10 of logits — and that figure is a MEASURED number: §A2 of
progress/2026-10-14/route-c-audit-2026-10-14.md bucketed 1694 hand-vs-hipBLAS rows and found 0 of the
1566 with ≥ 0.10 of room flipping, with all 24 flips under it. A single flip at or above the line
therefore does not by itself say the line is wrong — but it does say the pairing being judged now is not
the pairing the line was measured on, and the honest instrument is the same census re-run on the new
pairing, not a bump of the constant. That census is what this prints: rows per room bucket, flips per
bucket, every flip at or over the line named by row, id and both gaps — and, since the constant moved,
which flips CHANGE COLOUR between the retired absolute line and the ULP one (`--retired-line`, and
`--work-root` runs the whole corpus in one command so the figures a write-up quotes come from this file).

ONE DEFINITION, IMPORTED. The room quantity, the bucket edges and the argmax rule all come from
`tools/route_c_audit.py` (`logit_row_decision`, `BUCKETS`) — the same code that produced the table the
line came out of — the ULP conversion is `engine/gate.py`'s own `room_in_ulp`/`flip_room_ulp`, and the
threshold is `gate.CASCADE_BAND`. A census that re-derived any of the three could disagree with the rule
it is auditing and be green while doing so, which is the trap `prototype/attn_tile_model.py`'s `compare`
records for the attention mirror. Note the pairing here has no reference side: `room` is the min of the
two sides' own gaps, symmetric, and that is exactly how §A2 defined it — and it is still the reading the
gate applies to a cross-width pair, but no longer to a pin-vs-engine one (`gate.reference_room_ulp`).

ROOM IN TWO UNITS, AND WHY BOTH. The retired line was an ABSOLUTE top-1/top-2 difference (0.10 of a
logit), but a logit gap means something only relative to the magnitude it sits at: 0.10 is a fraction of
a BF16 step at 6 and forty steps at 40 (`engine/gate.py`'s own `bf16_ulp` says "0.125 is nothing at 40
and two whole ULP at 15.6"). So the census prints the §A2 buckets in logits AND the same rows bucketed
by room measured in BF16 ULPs of the row's own winner. Both are kept now that the line is the ULP one:
the logits column is what the retired threshold was asked in, and it is the only way to show a reader
which rows the change moved, in each direction.

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

import route_c_audit  # the frozen decision/room definitions the flip line came out of
import gate           # CASCADE_BAND.flip_room_max_ulp + room_in_ulp: the line, read not retyped

ROW_RE = re.compile(r"logits(\d+)\.bin$")
# Room in BF16 ULPs of the row's own top-1 magnitude. Edges are ULP counts, not logits: 1 ULP is the
# floor of what a reassociating route can legally do to a decision, so [0,1) is "the two candidates are
# inside one representable step", and the wide end is "no rounding story reaches this".
ULP_BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf"))]
# What the line carried before 2026-10-16, as the default for `--retired-line`. A historical figure, not
# a band: `engine/gate.py` no longer holds it, and only the census needs it, to say which rows changed
# colour when the line moved into ULP.
RETIRED_LINE_LOGITS = 0.10


def room_in_ulp(row_a: np.ndarray, row_b: np.ndarray) -> float:
    """min over the two sides of (that side's top-1/top-2 gap ÷ one BF16 step at that side's top-1).

    A one-line call into `engine/gate.py` now, and deliberately so: the step function and the min()
    shape are the gate's own (`gate.flip_room_ulp`, `gate.room_in_ulp`), so the census cannot drift from
    the rule it is auditing, and this pairing — two engine sides, no reference — is exactly the one the
    min() reading still governs.
    """
    return gate.flip_room_ulp(*[gate.top2(v)[1:] for v in (row_a, row_b)])


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
        r_ulp = room_in_ulp(va, vb)
        print(f"    decision: {labels[0]} {dec['argmax_ref']} vs {labels[1]} {dec['argmax_got']}"
              f" — {'SAME' if dec['argmax_match'] else 'FLIPPED'}, room {room:.6f} logits"
              f" = {r_ulp:.4g} BF16 ULP, against the"
              f" {gate.CASCADE_BAND['flip_room_max_ulp']:.2f} BF16 ULP line"
              + (" — ABOVE" if r_ulp >= gate.CASCADE_BAND["flip_room_max_ulp"] else " — inside"))


def census_pair(name: str, a_dir: Path, b_dir: Path, labels: tuple[str, str],
                show: list[int] = (), retired_line: float | None = None) -> dict | None:
    """One pair of corpora → the bucket table, the flips at or over the line, and the raw flip list.

    A row present on only one side is a REFUSAL, not a smaller denominator: the census' whole claim is
    about a fixed population of rows, and silently averaging over the rows that happened to be dumped
    twice is the empty-must-catch-set trap one layer over (`prototype/attn_tile_model.py`,
    `prototype/gdn_collapse/results.md` §1 — a check whose input list can shrink cannot fail).

    `retired_line`, when given, is the absolute-logit threshold the line used to carry; every flip is
    then also classified by whether the two lines answer differently, which is the only way to say out
    loud how many rows a unit change moved and in which direction.
    """
    line = gate.CASCADE_BAND["flip_room_max_ulp"]
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
    flips, wide, colour = [], [], []
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
            if r_ulp >= line:
                wide.append((i, room, r_ulp, dec))
            if retired_line is not None and (room >= retired_line) != (r_ulp >= line):
                colour.append((i, room, r_ulp, dec,
                               "newly ADMITTED by the ULP line" if room >= retired_line
                               else "newly RED under the ULP line"))
    print(f"\n=== {name}: {labels[0]} {a_dir} vs {labels[1]} {b_dir} — {len(rows)} rows"
          f" compared, {len(flips)} argmax flip(s), {len(wide)} of them at or over the"
          f" {line:.2f} BF16 ULP coin-toss line ===")
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
        over = " ≥ LINE" if r_ulp >= line else ""
        print(f"  row {i}: {labels[0]} {dec['argmax_ref']} (gap {dec['gap_ref']:.6g}) /"
              f" {labels[1]} {dec['argmax_got']} (gap {dec['gap_got']:.6g})"
              f" — room {room:.6g} = {r_ulp:.2f} BF16 ULP{over}")
    for i, room, r_ulp, dec, why in colour:
        print(f"  row {i} COLOUR CHANGE: {why} — was {room:.6g} logits"
              f" vs the retired {retired_line:.2f}, is {r_ulp:.4g} ULP vs {line:.2f}"
              f" ({dec['argmax_ref']} vs {dec['argmax_got']})")
    if show:
        show_rows(name, list(show), a_dir, b_dir, labels)
    return {"name": name, "rows": len(rows), "flips": len(flips), "wide": len(wide),
            "max_room": max((r for _i, r, _u, _d in flips), default=0.0),
            "max_room_ulp": max((u for _i, _r, u, _d in flips), default=0.0),
            # Named, not just measured: "the worst room in the corpus is 0.9985 ULP" is a claim a reader
            # cannot check without re-running; "...at wide520f row 180" is.
            "max_ulp_row": (max(flips, key=lambda f: f[2])[0] if flips else None),
            "buckets": {f"{lo}-{hi if hi != float('inf') else 'inf'}": v
                        for (lo, hi), v in buckets.items()},
            "ulp_buckets": {f"{lo:.0f}-{hi:.0f}" if hi != float("inf") else f"{lo:.0f}-inf": v
                            for (lo, hi), v in ulp_buckets.items()},
            "wide_rows": [i for i, _, _, _ in wide],
            "colour_rows": [(i, room, ulp, why) for i, room, ulp, _d, why in colour]}


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
    ap.add_argument("--retired-line", type=float, default=None, metavar="LOGITS",
                    help=f"also classify every flip against the absolute-logit line the gate carried"
                         f" before the ULP one (the tool's RETIRED_LINE_LOGITS ="
                         f" {RETIRED_LINE_LOGITS} is the historical value), and print each flip the two"
                         " lines answer differently — the colour change a unit switch actually caused")
    ap.add_argument("--work-root",
                    help=f"a directory of gate work dirs named <fixture>_<route>, each holding the"
                         " `ab_*` legs section (h) wrote (as `engine/gate.py --work` leaves them). Runs"
                         " every fixture x route x pairing combination in one pass and prints the"
                         " per-(route, pairing) aggregate the write-up quotes")
    ap.add_argument("--routes", default="library,lt",
                    help="comma list of route suffixes for --work-root")
    ap.add_argument("--pairings", default="ab_batched:ab_carry,ab_batched:ab_perrow",
                    help="comma list of side:side leg pairs for --work-root (the subdirectory names)")
    a = ap.parse_args()
    if not a.pair and not a.probe_work and not a.work_root:
        raise SystemExit("census: give --pair A B, --probe-work DIR or --work-root DIR")

    results = []
    planned = 0
    show = [int(x) for x in a.show_rows.split(",") if x.strip() != ""]
    retired = a.retired_line
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

    if a.work_root:
        # One pass over `<fixture>_<route>` dirs. The fixture list is REQUIRED to be the caller's, not
        # globbed from what happens to be on disk, for the same reason as everywhere else in this file:
        # a census whose denominator is whatever it found cannot fail.
        root = Path(a.work_root)
        fixtures = [f for f in a.fixtures.split(",") if f] or sorted(
            p.name for p in root.iterdir() if p.is_dir())
        combos = [(r, p.split(":")) for r in a.routes.split(",") if r.strip()
                  for p in a.pairings.split(",") if p.strip()]
        summary: dict[tuple[str, tuple[str, str]], list[dict]] = {}
        for route, (side_a, side_b) in combos:
            per: list[dict] = []
            for fx in fixtures:
                a_dir = root / f"{fx}_{route}" / side_a
                b_dir = root / f"{fx}_{route}" / side_b
                planned += 1
                if not (a_dir.is_dir() and b_dir.is_dir()):
                    print(f"REFUSE {fx} {route} {side_a}-vs-{side_b}: no such pair ({a_dir}, {b_dir})"
                          " — the aggregate below will not be a census of the requested fixtures")
                    continue
                r = census_pair(fx, a_dir, b_dir, (side_a, side_b), show, retired)
                if r is None:
                    continue
                r["fixture"] = fx
                per.append(r)
                results.append(r)
            summary[(route, (side_a, side_b))] = per
        print("\n=== per (route, pairing): the numbers a write-up quotes, from this tool ====")
        for (route, (sa, sb)), per in summary.items():
            rows = sum(r["rows"] for r in per)
            flips = sum(r["flips"] for r in per)
            wide = sum(r["wide"] for r in per)
            mx = max((r["max_room"] for r in per), default=0.0)
            mxu = max((r["max_room_ulp"] for r in per), default=0.0)
            worst = max(per, key=lambda r: r["max_room_ulp"], default=None)
            where = (f" at {worst['fixture']} row {worst['max_ulp_row']}" if worst else "")
            colour = [(r["fixture"], i, room, ulp, why) for r in per
                      for i, room, ulp, why in r["colour_rows"]]
            print(f"  {route} {sa}-vs-{sb}: {len(per)}/{len(per)} fixtures, {rows} rows, {flips}"
                  f" flips, {wide} at or over the {gate.CASCADE_BAND['flip_room_max_ulp']:.2f} ULP"
                  f" line; worst room {mx:.6g} logits = {mxu:.4g} ULP{where}")
            for fx, i, room, ulp, why in colour:
                print(f"    {fx} row {i}: {why} ({room:.6g} logits vs"
                      f" {a.retired_line:.2f}; {ulp:.4g} ULP vs"
                      f" {gate.CASCADE_BAND['flip_room_max_ulp']:.2f})")
            if not colour:
                print("    no flip in this pairing changes colour between the two lines")

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
    print(f"  the line under audit: CASCADE_BAND.flip_room_max_ulp ="
          f" {gate.CASCADE_BAND['flip_room_max_ulp']:.2f} BF16 ULP at the row's own top-1, measured by"
          " §A2 of progress/2026-10-14/"
          "route-c-audit-2026-10-14.md on the hand<->hipBLAS pairing (in the logits unit it retired) and"
          " re-measured on both cross-width pairings in prototype/gemm_algo_sweep/results.md §3f."
          " Flips at or over it are what a re-derivation would have to account for — this file reports"
          " them, it does not move them.")
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
