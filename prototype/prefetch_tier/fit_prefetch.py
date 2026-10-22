#!/usr/bin/env python3
"""fit_prefetch.py — the only reader of the P1 rig's timed output that may print a verdict.

Reads RESULT/CASE lines printed by `sweep.sh` (main agent only) and answers ONE question:

    does an explicit early load leave its bytes in the ~32 MiB tier until the use site, and is that
    worth enough of a 10.9357 ms step to be worth an engine change?

Two arms are compared inside one stratum (target, regime, setup_mib), and the strata never mix:

    cold - warm       = the prize if nothing sits between the prefetch and the use. Upper bound.
    cold - polluted   = the prize with the real intervening stream in the way. THE ENGINE NUMBER.

The rig times the USE-SITE READ ONLY: the evictor, the prefetch and the intervening stream all run outside
the timer, synchronously. That is not an implementation detail — a prefetch is an EXTRA read and the
consumer's read still happens afterwards, so if the whole sequence were timed the warm arm would move
strictly more bytes than the cold arm and `cold - warm` would be negative whatever the cache did. Timing
the consumer's read is the only formulation in which a prefetch can win, and it is the shape
`prototype/gdn_insitu` used to measure the +0.070 ms state penalty.

WHY THE NOISE FLOOR IS RELATIVE AND STRATIFIED. `engine/generation-limits.md` §7's corrections log
records a noise floor defined as the largest ABSOLUTE within-arm spread, which let one 311 ms cell make
a 7 ms comparison unadjudicable. Here the arms span two orders of magnitude (a 0.06 MiB target next to a
256 MiB setup), so an absolute floor is meaningless: it is set by the biggest cell and blinds every small
one. The floor is therefore computed per stratum as a RELATIVE spread of the arm's own ms_min across
passes, and converted to ms at that stratum's own magnitude. A stratum whose own repeats disagree by more
than --max-rel is refused outright rather than averaged: launch-bound rows on this box are bimodal
(`engine/performance-status.md` §6) and a bimodal mean is worse than no number.

REFUSALS (prints `REFUSE <reason>`, exit 2). A check whose input list can be empty cannot fail
(AGENTS: prototype/gdn_collapse/run.py):
  empty     — zero RESULT lines.
  partial   — a CASE without its RESULT, a RESULT without its CASE, a NO-TIMING row, a row whose
              verify is not ok, or a missing member of the expected arm set.
  count     — parsed record count != records the rig printed (its own RESULTS-PRINTED trailer).
  footprint — inside one stratum, the three arms do not carry identical target_mib and setup_mib. The
              rig's whole contract is "the arms share setup bytes", so a drifted row makes every delta a
              traffic delta instead of a residency delta.
  workgroup — inside one stratum, the arms' wg_per_launch differ. The three arms run the same grid on
              the same use-site kernel; a difference means they are not measuring the same read, and
              `prototype/attn_wide/A12-WHY-NOT-FASTER.md` is the cautionary tale for launch population.
  bimodal   — some stratum's relative repeat spread exceeds --max-rel.

PRE-REGISTERED LINES (written before any timing was taken; see sweep.sh's header, same numbers):
  bar       0.30 ms per step  [queue P1 preamble: what both of this week's wins cleared]
  WIN       cold - polluted >= 0.30 ms per step at an engine-shaped stratum, adjudicable against that
            stratum's own noise floor
  KILL-1    |cold - warm| <= noise for EVERY stratum  -> an explicit prefetch changes nothing at all;
            the mechanism is not present on this part
  KILL-2    |cold - polluted| <= noise for EVERY stratum while some stratum shows cold - warm > noise
            -> the bytes are evicted before the use site, which is the failure mode P1 was built to
               avoid, and prefetch does not belong in this engine
  else      INCONCLUSIVE — a result, not a failure, and reported with the stratum that could not be
            adjudicated and why

Per-step extrapolation is printed and labelled as an extrapolation: the rig prices ONE staged read, and
the per-step figure multiplies by how many times that slice is consumed per step. A target larger than
the tier is not extrapolable at all (it cannot be staged whole) and says so.

No timing is taken here; this only reads numbers the rig printed.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import size_targets  # noqa: E402  the single place any byte size in this package is spelled

FIELD_RE = re.compile(r"([a-z_0-9]+)=(\S+)")
BAR_MS = 0.30            # quoted: queue P1 preamble
TIER_MIB = 32.0          # quoted: KFD level 3, printed by p1c_query
# How many times one staged slice of this target is consumed per decode step. Not measured: derived
# from the layer counts engine/state.h publishes (18 GDN layers read their state once each per step,
# 6 attention layers each walk their own KV slice once, one layer's weight block is read once).
CONSUMPTIONS_PER_STEP = {"state": 18, "kv-layer": 6, "weights": 24, "tiny": 0}


def per_step_factor(target, staged_mib):
    """Turn a one-staged-set delta into a per-decode-step delta, or say why it cannot be done.

    The rig measures the saving on ONE staged set; the engine would prefetch slice by slice, so the
    multiplier is that target's consumptions per step divided by how many slices the staged set actually
    holds. Staging all 18 state slices once and staging one slice per layer are the same engine behaviour
    and must not become two different prizes: 18 consumptions / 18 staged slices = 1, and 18/1 = 18 for a
    one-slice stage. A staged set bigger than the tier cannot be staged whole, and a set that is not a
    whole number of slices has an undefined multiplier, because part of what was staged is never consumed.
    """
    if staged_mib > TIER_MIB + 1e-9:
        return None, (f"staged {staged_mib:.3f} MiB exceeds the {TIER_MIB:.0f} MiB tier, so it cannot be "
                      "staged whole and has no per-step meaning")
    t = target.lower()
    if CONSUMPTIONS_PER_STEP.get(t, 0) == 0:
        return None, f"target {target} is a synthetic scale probe with no per-step meaning"
    try:
        slice_mib = size_targets.rig_slice_bytes(t) / size_targets.MIB
    except SystemExit:
        return None, f"target {target} is not a target size_targets.py can derive a slice for"
    slices = staged_mib / slice_mib
    if abs(slices - round(slices)) > 0.02:
        return None, (f"staged {staged_mib:.4f} MiB is {slices:.4g} slices of {target} "
                      f"(one slice = {slice_mib:.4g} MiB); a partial slice's bytes are never all "
                      "consumed, so the multiplier is undefined")
    return (CONSUMPTIONS_PER_STEP[t] / max(slices, 1.0),
            f"{CONSUMPTIONS_PER_STEP[t]} consumptions/step over {slices:.4g} staged slice(s) "
            f"of {slice_mib:.4g} MiB")


def die(msg):
    print("REFUSE " + msg)
    sys.exit(2)


def parse(paths):
    rows, trailers, problems = [], {}, []
    pending = None
    n_case = n_result = n_notiming = 0
    for path in paths:
        with open(path, errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("CASE "):
                    if pending is not None:
                        problems.append("CASE-without-RESULT %s" % pending)
                    pending = line.split()[1]
                    continue
                if line.startswith("RESULT "):
                    n_result += 1
                    d = dict(FIELD_RE.findall(line))
                    if pending is None:
                        problems.append("RESULT-without-CASE arm=%s" % d.get("arm", "?"))
                    else:
                        d["target_name"] = pending
                        pending = None
                    if d.get("verify") != "ok":
                        problems.append("verify_not_ok arm=%s target=%s"
                                        % (d.get("arm", "?"), d.get("target_name", "?")))
                        continue
                    for k in ("ms_min", "ms_median", "ms_max", "target_mib", "setup_mib",
                              "intervene_mib"):
                        if k not in d:
                            problems.append("missing_field %s arm=%s" % (k, d.get("arm", "?")))
                            break
                        d[k] = float(d[k])
                    else:
                        for k in ("blocks", "threads", "wg_per_launch", "reps", "iters"):
                            try:
                                d[k] = int(float(d[k]))
                            except (KeyError, ValueError):
                                problems.append("missing_field %s" % k)
                        d["pass"] = int(d.get("pass", "1"))
                        rows.append(d)
                    continue
                if line.startswith("NO-TIMING "):
                    n_notiming += 1
                    d = dict(FIELD_RE.findall(line))
                    problems.append("NO-TIMING arm=%s reason=%s"
                                    % (d.get("arm", "?"), d.get("reason", "?")))
                    pending = None
                    continue
                if line.startswith("RESULTS-PRINTED") or line.startswith("CASES-PRINTED"):
                    key, _, val = line.partition("=")
                    trailers[key.strip()] = int(val.split()[0])
    if pending is not None:
        problems.append("CASE-without-RESULT %s" % pending)
    return rows, trailers, problems, (n_case, n_result, n_notiming)


def spread(vals):
    lo, hi = min(vals), max(vals)
    return (hi - lo), (0.0 if lo <= 0 else (hi - lo) / lo)


def selftest():
    """Proves this reader can fail. A gate driver whose parser silently matches nothing prints PASS over
    a matrix it never read (AGENTS: prototype/gdn_collapse/run.py), so every refusal is asserted against
    synthetic text here, and so are the verdict shapes. No device, no timing."""
    import os
    import tempfile

    fails = []

    def line(target, arm, ms, regime="idle", setup=64.0, tmib=18.0, wg=1152, passno=1, verify="ok"):
        return ("CASE %s\nRESULT %s target=%s target_mib=%.4f arm=%s regime=%s inject=none "
                "setup_mib=%.4f intervene_mib=21.0000 blocks=%d threads=256 wg_per_launch=%d "
                "verify=%s ms_min=%.4f ms_median=%.4f ms_max=%.4f reps=5 iters=3 pass=%d\n"
                % (target, target, target, tmib, arm, regime, setup, wg, wg, verify, ms, ms, ms,
                   passno))

    def run(text, argv_extra=()):
        fd, path = tempfile.mkstemp(text=True)
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        old = sys.argv
        sys.argv = ["fit", path, "--quiet"] + list(argv_extra)
        try:
            rc = main()
            return ("ok", rc)
        except SystemExit as exc:
            return ("exit", exc.code)
        finally:
            sys.argv = old
            os.unlink(path)

    def expect(name, got, want):
        if got != want:
            fails.append("%s: got %r want %r" % (name, got, want))

    expect("empty", run("nothing here\n"), ("exit", 2))
    expect("partial", run("CASE state\n"), ("exit", 2))
    expect("verify-red", run(line("state", "cold", 1.0, verify="FAIL")
                             + line("state", "warm", 1.0) + line("state", "polluted", 1.0)),
           ("exit", 2))
    drift = (line("state", "cold", 1.0) + line("state", "warm", 0.5)
             + line("state", "polluted", 1.0, tmib=20.0))
    expect("footprint", run(drift), ("exit", 2))
    wgdrift = (line("state", "cold", 1.0, wg=1152) + line("state", "warm", 0.5, wg=576)
               + line("state", "polluted", 1.0, wg=1152))
    expect("workgroup", run(wgdrift), ("exit", 2))
    bimodal = ""
    for p in (1, 2):
        ms = 1.0 if p == 1 else 1.5
        bimodal += line("state", "cold", ms, passno=p) + line("state", "warm", 0.9, passno=p) \
            + line("state", "polluted", 0.95, passno=p)
    expect("bimodal", run(bimodal), ("exit", 2))
    flat = ""
    for p in (1, 2):
        flat += line("state", "cold", 1.0, passno=p) + line("state", "warm", 1.0, passno=p) \
            + line("state", "polluted", 1.0, passno=p)
    expect("kill1-shape", run(flat), ("ok", 0))
    kill2 = ""
    for p in (1, 2):
        kill2 += line("state", "cold", 1.0, passno=p) + line("state", "warm", 0.9, passno=p) \
            + line("state", "polluted", 1.0, passno=p)
    expect("kill2-shape", run(kill2), ("ok", 0))
    for delta in (0.40, 0.05):
        moving = ""
        for p in (1, 2):
            moving += line("state", "cold", 1.0, passno=p)
            moving += line("state", "warm", 1.0 - delta, passno=p)
            moving += line("state", "polluted", 1.0 - delta, passno=p)
        expect("verdict-shape-%.2f" % delta, run(moving), ("ok", 0))
    expect("missing-arm", run(line("state", "cold", 1.0)), ("exit", 2))
    print("FIT-SELFTEST ok=%d fails=%d" % (0 if fails else 1, len(fails)))
    for f in fails:
        print("  FIT-FAIL " + f)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="files printed by sweep.sh")
    ap.add_argument("--selftest", action="store_true",
                    help="assert every refusal and verdict path against synthetic text (no device)")
    ap.add_argument("--max-rel", type=float, default=0.02,
                    help="max relative ms_min spread for one arm across passes before the stratum is "
                         "refused as bimodal (default 2 %)")
    ap.add_argument("--bar", type=float, default=BAR_MS)
    ap.add_argument("--json", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.inputs:
        die("no input files given; this reader only fits numbers the rig printed")

    rows, trailers, problems, counts = parse(args.inputs)
    n_case, n_result, n_notiming = counts
    out = (lambda s: None) if args.quiet else (lambda s: print(s))

    if n_result == 0 or not rows:
        die("empty: zero usable RESULT lines in %d file(s); %d CASE and %d NO-TIMING lines seen. "
            "Either nothing was timed or this reader matched nothing." % (len(args.inputs), n_case,
                                                                          n_notiming))
    if problems:
        die("partial/unusable input: %d problem(s): %s" % (len(problems), "; ".join(problems[:8])))
    want = trailers.get("RESULTS-PRINTED")
    if want is not None and want != len(rows):
        die("count: the rig printed %d RESULT rows, this reader parsed %d" % (want, len(rows)))

    def strat(r):
        # intervene_mib is part of the stratum on purpose: the residency-cliff sweep varies it inside one
        # (target, regime, setup) cell, and pooling those rows would take a median across four different
        # amounts of eviction and call it one reading.
        return (r["target"], r["regime"], "%.4f" % r["setup_mib"], "%.4f" % r["intervene_mib"])

    by = defaultdict(list)
    for r in rows:
        by[strat(r)].append(r)

    out("== P1 prefetch residency fit — %d RESULT rows, %d strata, bar %.2f ms/step [quoted]"
        % (len(rows), len(by), args.bar))
    out("   arms: cold / warm / polluted;  prize = cold - polluted (engine number), cold - warm (bound)")

    strata, unusable = [], []
    for key in sorted(by):
        rs = by[key]
        arms = defaultdict(list)
        for r in rs:
            arms[r["arm"]].append(r)
        missing = [a for a in ("cold", "warm", "polluted") if a not in arms]
        if missing:
            unusable.append("%s: missing arms %s" % ("/".join(map(str, key)), ",".join(missing)))
            continue
        # footprint: the three arms must book identical bytes or a delta is a traffic delta
        for field in ("target_mib", "setup_mib"):
            vals = set("%.4f" % r[field] for a in arms for r in arms[a])
            if len(vals) != 1:
                die("footprint: stratum %s carries %d different %s values %s"
                    % (key, len(vals), field, sorted(vals)))
        wgs = set(r["wg_per_launch"] for a in arms for r in arms[a])
        if len(wgs) != 1:
            die("workgroup: stratum %s arms launch %d different workgroup populations %s; the three "
                "arms must run the same grid or they are not measuring the same read"
                % (key, len(wgs), sorted(wgs)))
        # relative noise floor, judged PER COMPARISON rather than per stratum. A comparison may only be
        # judged against the spread of the two arms it actually uses: refusing a whole stratum because
        # an arm it never touches is bimodal throws away the reading that was asked for, and on this rig
        # it threw away nearly all of them - the warm arm of a 1,152-workgroup single read repeats 2.0-
        # 2.5 % apart at REPS=5 and at REPS=12 alike (launch-bound rows are bimodal on this box,
        # `engine/performance-status.md` §6), so a per-stratum cap was a check no arm could pass, which
        # makes it a defect rather than a standard. The cap itself is NOT moved: each comparison is still
        # refused when either of its own two arms repeats by more than --max-rel.
        rel = {}
        for arm, ar in arms.items():
            rel[arm] = spread([r["ms_min"] for r in ar])[1]
        pairs = {"warm": ("cold", "warm"), "pol": ("cold", "polluted")}

        def pair_ok(a, b):
            return rel[a] <= args.max_rel and rel[b] <= args.max_rel

        def noise_of(a, b):
            m = max(max(r["ms_min"] for r in arms[a]), max(r["ms_min"] for r in arms[b]))
            return max(rel[a], rel[b]) * m

        refused = [k for k, (a, b) in pairs.items() if not pair_ok(a, b)]
        if len(refused) == len(pairs):
            worst_arm = max(rel, key=lambda a: rel[a])
            vals = [r["ms_min"] for r in arms[worst_arm]]
            unusable.append("%s: both comparisons bimodal, worst arm=%s repeats %.4f..%.4f (%.2f %%) "
                            "> --max-rel %.2f %%" % ("/".join(map(str, key)), worst_arm,
                                                     min(vals), max(vals), 100 * rel[worst_arm],
                                                     100 * args.max_rel))
            continue
        for k in refused:
            a, b = pairs[k]
            bad = max((a, b), key=lambda x: rel[x])
            vals = [r["ms_min"] for r in arms[bad]]
            unusable.append("%s: comparison cold-%s refused, arm=%s repeats %.4f..%.4f (%.2f %%) "
                            "> --max-rel %.2f %% (the other comparison in this stratum is still "
                            "adjudicated)" % ("/".join(map(str, key)), k, bad, min(vals), max(vals),
                                              100 * rel[bad], 100 * args.max_rel))
        magnitude = max(r["ms_min"] for a in arms for r in arms[a])
        noise_ms = max([noise_of(a, b) for k, (a, b) in pairs.items() if k not in refused] or [0.0])
        worst_rel = max(rel.values())
        def med(a):
            vals = sorted(r["ms_median"] for r in arms[a])
            return vals[len(vals) // 2]
        d_warm = med("cold") - med("warm")
        d_pol = med("cold") - med("polluted")
        ok_warm, ok_pol = pair_ok("cold", "warm"), pair_ok("cold", "polluted")
        tgt = key[0]
        per_step, why_step = per_step_factor(tgt, float(arms["cold"][0]["target_mib"]))
        fits = float(arms["cold"][0]["target_mib"]) <= TIER_MIB
        strata.append({
            "target": tgt, "regime": key[1], "setup_mib": float(key[2]),
            "intervene_mib": float(key[3]),
            "target_mib": arms["cold"][0]["target_mib"],
            "wg_per_launch": arms["cold"][0]["wg_per_launch"],
            "noise_rel": worst_rel, "noise_ms": noise_ms,
            "cold": med("cold"), "warm": med("warm"), "polluted": med("polluted"),
            "d_warm": d_warm, "d_pol": d_pol,
            "adjudicable_warm": ok_warm and abs(d_warm) > max(noise_of("cold", "warm"), 1e-9),
            "adjudicable_pol": ok_pol and abs(d_pol) > max(noise_of("cold", "polluted"), 1e-9),
            "ok_warm": ok_warm, "ok_pol": ok_pol,
            "per_step_factor": per_step, "fits_tier": fits, "why_step": why_step,
            "per_step_pol": (d_pol * per_step) if (fits and per_step) else None,
            "per_step_warm": (d_warm * per_step) if (fits and per_step) else None,
        })

    for s in strata:
        out("\n  stratum target=%-9s regime=%-4s setup=%.0f MiB intervene=%.4g MiB  "
            "(cold %.4f warm %.4f polluted %.4f ms)"
            % (s["target"], s["regime"], s["setup_mib"], s["intervene_mib"],
               s["cold"], s["warm"], s["polluted"]))
        out("    workgroups per launch: %d (same on all three arms, asserted)" % s["wg_per_launch"])
        out("    noise floor: %.2f %% worst arm; each comparison is judged against its own two arms; "
            "stratum floor shown = %.4f ms" % (100 * s["noise_rel"], s["noise_ms"]))
        lab_w = "REFUSED (an arm of it is bimodal)" if not s["ok_warm"] else (
            "ADJUDICABLE" if s["adjudicable_warm"] else "within noise")
        lab_p = "REFUSED (an arm of it is bimodal)" if not s["ok_pol"] else (
            "ADJUDICABLE" if s["adjudicable_pol"] else "within noise")
        out("    cold - warm     %+.4f ms  %s" % (s["d_warm"], lab_w))
        out("    cold - polluted %+.4f ms  %s   <- the engine reading" % (s["d_pol"], lab_p))
        if s["fits_tier"] and s["per_step_factor"]:
            out("    extrapolated to a step: %+.4f ms/step (x%.4g, %s; target fits the tier)"
                % (s["per_step_pol"], s["per_step_factor"], s["why_step"]))
        else:
            out("    extrapolated to a step: NOT EXTRAPOLABLE (%s)" % s["why_step"])
    for u in unusable:
        out("  UNUSABLE %s" % u)

    if not strata:
        die("no stratum survived its own noise floor (%d unusable: %s); no verdict is available from "
            "this window" % (len(unusable), "; ".join(unusable[:4])))

    # ---- verdict against the pre-registered lines --------------------------------------------
    kill1 = not any(s["adjudicable_warm"] for s in strata)
    kill2 = (not any(s["adjudicable_pol"] for s in strata)) and any(s["adjudicable_warm"] for s in strata)
    winners = [s for s in strata
               if s["per_step_pol"] is not None and s["adjudicable_pol"]
               and s["per_step_pol"] >= args.bar]
    out("\n== VERDICT")
    if kill1:
        out("  KILL-1: no stratum shows cold - warm above its own noise floor. An explicit prefetch "
            "changes nothing at the use site on this part. The mechanism is not present and P1 is dead; "
            "the 18.9 MiB state warm-up in D4's +0.070 ms [quoted] is the only piece of it that was ever "
            "measured, and it is already inside the shipped numbers.")
        verdict = "KILL-1"
    elif kill2:
        out("  KILL-2: prefetch helps when nothing follows it, and helps by nothing measurable once the "
            "real intervening traffic runs. The staged bytes are gone by the use site, which is exactly "
            "the failure P1 was built to dodge. Prefetch does not belong in this engine.")
        verdict = "KILL-2"
    elif winners:
        best = max(winners, key=lambda s: s["per_step_pol"])
        out("  WIN: %s/%s gives cold - polluted %+.4f ms per staged read = %+.4f ms/step against a "
            "%.2f ms bar. The bytes are still resident at the use site with the intervening stream run."
            % (best["target"], best["regime"], best["d_pol"], best["per_step_pol"], args.bar))
        verdict = "WIN"
    else:
        out("  INCONCLUSIVE: some strata moved, none reached the %.2f ms bar per step and none was flat "
            "enough to kill. Quote the stratum table, not a single number, and say which stratum would "
            "have to be re-taken in one window to settle it." % args.bar)
        verdict = "INCONCLUSIVE"
    out("  strata adjudicable on the engine reading: %d of %d; unusable strata: %d"
        % (sum(1 for s in strata if s["adjudicable_pol"]), len(strata), len(unusable)))
    out("  REMEMBER: a rig delta is not a step delta. The extrapolation column carries the assumption "
        "that each slice is staged once per step and survives; the engine path has to be A/B'd at "
        "4,032 lanes before anything is claimed for it.")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"verdict": verdict, "strata": strata, "unusable": unusable,
                       "bar_ms": args.bar}, fh, indent=1)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
