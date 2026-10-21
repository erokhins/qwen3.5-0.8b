#!/usr/bin/env python3
"""check_result_line.py — asserts one RESULT line is well-formed, and nothing more.

Used as a canary by sweep.sh before it takes any real timing, and by check_wave_shape.sh against
synthetic lines. It deliberately does NOT look at whether the millisecond values are plausible: this
file must never become a place where a number is judged, because the same parser is used on numbers
taken in a contaminated window. What it does check is that the line is a complete record:

  * every field a reader needs is present,
  * the counters agree with the geometry (launches_executed == launches, and a matrix cell's
    blocks_total == launches * grid),
  * per_launch_us == ms_min * 1000 / launches (so a printed rate cannot come from a different cell
    than the printed time),
  * verify=ok is present, because a row whose witnesses are not clean must carry NO-TIMING instead.

Usage: check_result_line.py "<one RESULT line>"   (or - for stdin)
Exits 0 with `RESULT-LINE ok ...`, or 1 with `RESULT-LINE BAD <reason>`.
"""
import re
import sys

REQUIRED = ["arm", "family", "launches", "read_mib", "write_mib", "demand_mib", "verify",
            "launches_executed", "syncs_executed", "ms_min", "ms_median", "ms_max",
            "per_launch_us", "per_block_us", "reps", "iters"]


def check(line):
    d = dict(re.findall(r"([a-z_0-9]+)=(\S+)", line))
    if not line.startswith("RESULT "):
        return "line does not start with RESULT"
    for f in REQUIRED:
        if f not in d:
            return "missing field %s" % f
    if d["verify"] != "ok":
        return "verify=%s: a row that is not clean must print NO-TIMING, never a millisecond figure" \
               % d["verify"]
    try:
        ms = float(d["ms_min"]); L = int(d["launches"]); plu = float(d["per_launch_us"])
        pbu = float(d["per_block_us"]); bt = int(d["blocks_total"])
        le = int(d["launches_executed"]); se = int(d["syncs_executed"])
        reps = int(d["reps"]); iters = int(d["iters"])
    except ValueError as exc:
        return "unparseable number: %s" % exc
    if L <= 0 or bt <= 0 or reps <= 0 or iters <= 0:
        return "non-positive geometry (launches=%d blocks_total=%d reps=%d iters=%d)" % (
            L, bt, reps, iters)
    if le != L:
        return "launches_executed=%d disagrees with launches=%d: the timed step did not run the " \
               "booked number of launches" % (le, L)
    if se < 1:
        return "syncs_executed=%d: an untimed step that never syncs measures queueing, not work" % se
    if d["family"] == "mat":
        if "grid" not in d:
            return "matrix row without grid"
        if bt != L * int(d["grid"]):
            return "blocks_total=%d != launches*grid=%d" % (bt, L * int(d["grid"]))
    if abs(plu - ms * 1000.0 / L) > 1e-3 * max(plu, 1e-9):
        return "per_launch_us=%s is not ms_min*1000/launches (=%.4f): a rate printed next to a time " \
               "that is not the time it divides" % (plu, ms * 1000.0 / L)
    if pbu <= 0:
        return "per_block_us=%s" % pbu
    return ""


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "-":
        lines = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    else:
        lines = [arg]
    bad = 0
    for ln in lines:
        why = check(ln)
        if why:
            bad += 1
            print("RESULT-LINE BAD %s :: %s" % (why, ln[:160]))
        else:
            print("RESULT-LINE ok fields=%d (structural check only, the value is not judged here)"
                  % len(re.findall(r"([a-z_0-9]+)=\S+", ln)))
    return 1 if bad or not lines else 0


if __name__ == "__main__":
    sys.exit(main())
