#!/usr/bin/env python3
"""gate.py — Route D3's device gate: does the LDS-reclaim core change any bit, at any context length?

Run under `tools/gpu_lock.sh`, one holder, blocking (`prototype/attn_decode/run.py` does not self-lock;
neither does this driver — see `self_locks()` below, which checks rather than assumes).

WHAT IT ADJUDICATES, in the order it prints
  1. BIT-IDENTITY. `attn_core_lean` must write the same bytes as the frozen `attn_core` on every case
     and every stage. This is the whole claim of the candidate, so it is asserted per stage per case and
     an empty case list is an error, not a pass (AGENTS.md, "a gate driver whose parser silently matches
     nothing reports PASS").
     POST-PORT READING (2026-10-21, PORT-D3-LEAN.md): the `lean` spelling now lives in the shipping
     header, so the two bodies are identical and this section is a tautology by construction — it can
     still catch a build that linked a stale object, and nothing else. The evidence that `lean` was
     bit-identical to the pre-port core is `results.md` §1, taken before the port; the live authority on
     the shipping path is now `engine/gate.py`, not this section.
  2. AGAINST THE PINS. Where a case has a pinned HF reference, both cores are also judged by the floors
     this op already publishes (`prototype/attn_decode/run.py`'s COS_MIN/MEANREL_MAX/NDIFF_*), imported,
     never restated. A candidate that is bit-identical to a form that passes the pins needs no new band;
     a candidate that is not is reported, not re-banded.
  3. NEGATIVES, DERIVED. The must-catch / must-not-catch split is COMPUTED from each case's geometry
     (lane count, and whether the injection acts on the score or on the epilogue), not hand-listed —
     §20.6's lesson: a hand-listed set rots into "everything". The rule for the score-side injections is
     §18's: a score-only change cannot move a row with exactly ONE lane, because softmax of a singleton
     is 1.0 whatever the score arrives as.

THE CASE POPULATION is derived too, in two halves: every pinned decode step of the long fixtures and of
`dec8`, and then the MAXSC-boundary geometries obtained by truncating a real pinned cache to 511/512/
513/514/515 and 1023/1024/1025 lanes. The second half has no HF pin — it exists to make the
streaming-route boundary reachable per form-vs-form comparison, which is what §20.1's aliasing bug and
§20.6's vacuous control were both about: a route no case reaches has no tests.

PLAN/fixture plumbing is `prototype/attn_decode/reference.py`'s, so the operand files are the same bytes
the shipped gate feeds its own binary.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OP = ROOT / "prototype" / "attn_decode"
sys.path.insert(0, str(OP))
sys.path.insert(0, str(ROOT / "engine"))

import torch                                                # noqa: E402
import reference as R                                       # noqa: E402
import run as OPRUN                                         # noqa: E402  the shipped op's own gate

BIN = HERE / "attn_decode_wide"
WORK = ROOT / "tmp" / "attn_decode_wide"
N_Q, DH, N_KV = R.N_Q, R.DH, R.N_KV
MAXSC = 512
# The floors are imported from the shipped gate rather than restated here: a D3 number and a §17/§18
# number must mean the same thing, and a copied constant is a second authority that drifts. The
# comparator is `bf16_ulp.report` behind `run.py`'s `compare()` — the cancellation band is part of the
# floor and a hand-rolled cos/meanrel would silently be a different (looser) test.
compare, ndiff_limit = OPRUN.compare, OPRUN.ndiff_limit
COS_MIN, MEANREL_MAX = OPRUN.COS_MIN, OPRUN.MEANREL_MAX

STAGES = ["attn_pre_gate", "gated", "q_rope", "k_rope", "cache_keys", "cache_values"]
BOUNDARY = [1, 511, 512, 513, 514, 515, 1023, 1024, 1025]

# The dot-loop spellings `check_source()` adjudicates. Written as fragments rather than regexes because
# the test is "this exact text appears in the shipping kernel", and a regex that matches nothing reads as
# a pass (AGENTS.md). Two score dots per branch (self-row and cache-row) x two branches = 4 sites.
LITERAL_TRIP = "for (int t = 0; t < 8; ++t) if (t < nd)"
RUNTIME_TRIP = "for (int t = 0; t < nd; ++t)"
DOT_SITES_EXPECT = 4


def slice_kernel(lines: list[str], signature: str) -> list[str]:
    """The brace-balanced body of the kernel whose definition starts with `signature`, or [] if absent.
    Braces are counted per line, which is the same simplification the original made and is sound for
    this header's house style (one brace per line at the statement level, no brace in a string)."""
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith(signature))
    except StopIteration:
        return []
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            return lines[start:i + 1]
    return []

# What each injection can reach, and which geometry exempts it. BOTH halves are derived, not listed by
# taste — §20.6's lesson is that a hand-listed must-catch set rots into "everything" and proves nothing,
# and a must-not-catch set that is not derived hides a dead injection.
#   stages: adjudicate only what the injection can move. `qswap` also rewrites `q_rope`, so judging a
#           softmax-shaped exemption on "did any stage change" reports an exemption as violated when the
#           reduction in fact did not move.
#   exempt: "singleton"   — §18: a change that reaches ONLY THE SCORE cannot move a row of exactly one
#                           lane, because softmax of a singleton is 1.0 whatever the score arrives as.
#           "one_pass_row"— `nomax` never re-anchors a RUNNING max, and a running max is a streaming-form
#                           object: a row at or below MAXSC is served by the one-pass form, which takes
#                           its max in a separate pass over the staged scores. Which branch a row lands
#                           in is a fact about the geometry, so that is what the must-catch set follows.
#           "none"        — nothing exempts it.
# The exemption is a claim about what an injection TOUCHES, and the first run of this gate proved the
# distinction matters: `gqaoff` was listed as score-side and the 1-lane row moved anyway, because
# mis-mapping the KV head selects a different V as well as a different score. A value-side injection has
# no singleton exemption.
NEGS = {
    "dimhalf":  {"stages": ["attn_pre_gate", "gated"], "exempt": "singleton"},
    "nomax":    {"stages": ["attn_pre_gate", "gated"], "exempt": "one_pass_row"},
    "gqaoff":   {"stages": ["attn_pre_gate", "gated"], "exempt": "none"},
    "qswap":    {"stages": STAGES, "exempt": "none"},
    "gatehalf": {"stages": ["gated"], "exempt": "none"},
}


def exempt_reason(c: dict, kind: str):
    if kind == "none":
        return None
    if kind == "singleton" and c["lvalid"] == 1:
        return "1 lane: softmax of a singleton is 1.0 whatever the score is (SS18)"
    if kind == "one_pass_row" and c["lvalid"] <= MAXSC:
        return "at or below MAXSC: one-pass form takes its max in a separate pass"
    return None


def self_locks() -> bool:
    """This driver must NOT take `tools/gpu_lock.sh` itself: the caller holds it, and a driver that
    re-acquires a held lock parks silently with no CPU and no GPU work, which reads as a hung box
    (AGENTS.md, the three-self-locking-drivers row). The check looks only inside a `subprocess` call — a
    grep for the bare tool name matches this file's own prose, and a check that fires on its own
    comments is the same failure class as a parser that matches nothing."""
    calls = re.findall(r"subprocess\.\w+\((.*?)\)", (HERE / "gate.py").read_text(), re.S)
    return any("gpu_lock.sh" in c for c in calls)


def long_cases(verbose=True) -> list[dict]:
    """Every pinned decode step of every long fixture, as decode-shaped single-row cases."""
    out = []
    for fx in ("wide520f", "wide640f", "wide1024f", "wide4096f"):
        d = ROOT / "data" / "refs" / fx
        if not d.exists():
            continue
        steps = sorted({p.name.split(".")[0] for p in d.glob("decode*.L03.cache_keys.npy")})
        for st in steps:
            c = R.decode_case(d, st, 3)
            # Namespaced by fixture, deliberately: `decode7.L03` exists in wide640f, wide1024f AND
            # wide4096f, and `reference.write_case` names its operand files after the case — so an
            # unnamespaced plan is last-writer-wins over files an earlier line still points at.
            # That is AGENTS.md's bare-dump-name trap, which this gate reinstall on its first run.
            c["name"] = "%s.%s" % (fx, c["name"])
            c["pin"] = d
            out.append(c)
            if verbose:
                print(f"  case {c['name']:24s} lanes={c['lvalid']:5d}  pinned")
    return out


def short_cases() -> list[dict]:
    """The one-pass route: `dec8`'s pinned steps, all under MAXSC, so `blocks_for_row` gives one block."""
    d = ROOT / "data" / "refs" / "dec8"
    out = []
    for st in sorted({p.name.split(".")[0] for p in d.glob("decode*.L03.cache_keys.npy")}):
        c = R.decode_case(d, st, 3)
        c["name"] = "dec8." + c["name"]
        c["pin"] = d
        out.append(c)
    return out


def boundary_cases(src: dict) -> list[dict]:
    """The same real query and the same real cache PREFIX, cut to lane counts around every structural
    boundary: 1 (the singleton §18 exempted), MAXSC-1/MAXSC/MAXSC+1 (the one-pass/streaming decision),
    and 2xMAXSC +-1 (two streaming windows and the first row that needs a third).

    No HF pin exists at these widths, so these cases are form-vs-form witnesses. They are NOT evidence
    about correctness against the reference; they are evidence that the two cores agree where the route
    changes shape, which is where the §19/§20 class of bug lives.
    """
    kc_full = src["kcache"].reshape(N_KV, src["cap"], DH)
    vc_full = src["vcache"].reshape(N_KV, src["cap"], DH)
    out = []
    for lanes in BOUNDARY:
        if lanes > src["lvalid"]:
            continue
        c = dict(src)
        c["name"] = "bnd%04d.L03" % lanes
        c["lvalid"] = lanes
        c["pos"] = lanes - 1
        c["kcache"] = kc_full[:, :lanes].reshape(-1).contiguous()
        c["vcache"] = vc_full[:, :lanes].reshape(-1).contiguous()
        c["cap"] = lanes
        c["pin"] = None
        c["ref"] = None
        out.append(c)
    return out


def run_arm(cases, plan_name, extra=()) -> Path:
    """One binary invocation over one plan; returns the plan's output root."""
    indir, outroot = WORK / "in", WORK / "out" / plan_name
    indir.mkdir(parents=True, exist_ok=True)
    outroot.mkdir(parents=True, exist_ok=True)
    lines = []
    for c in cases:
        outdir = outroot / c["name"]
        outdir.mkdir(parents=True, exist_ok=True)
        lines.append(R.write_case(c, indir, outdir))
    with open(WORK / (plan_name + ".plan"), "w") as f:
        for fields in lines:
            f.write(" ".join(str(x) for x in fields) + "\n")
    cmd = [str(BIN), "--mode", "once", "--plan", str(WORK / (plan_name + ".plan"))] + list(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        sys.stdout.write(proc.stdout + proc.stderr)
        raise SystemExit("arm %s failed: %s" % (plan_name, " ".join(cmd)))
    res = [l for l in proc.stdout.splitlines() if l.startswith("RESULT")]
    if len(res) != 1:
        raise SystemExit("arm %s printed %d RESULT lines, expected exactly 1:\n%s"
                         % (plan_name, len(res), proc.stdout))
    fields = dict(kv.split("=", 1) for kv in res[0].split()[1:])
    if int(fields["cases"]) != len(cases):
        raise SystemExit("arm %s ran %s cases, the plan has %d — the plan did not reach the kernel"
                         % (plan_name, fields["cases"], len(cases)))
    return outroot


def stage_bytes(outroot: Path, case: dict, stage: str) -> bytes:
    p = outroot / case["name"] / (stage + ".bin")
    if not p.exists():
        raise SystemExit("missing %s — the arm wrote no such stage, so the comparison is vacuous" % p)
    return p.read_bytes()


def bitdiff(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        return -1
    return sum(x != y for x, y in zip(a, b))


def judge_pin(outroot: Path, case: dict) -> tuple[bool, str]:
    """`attn_pre_gate` against the pinned HF tensor, under this op's published floors."""
    n = N_Q * DH * case.get("nrows", 1)
    got = R.read_out(outroot / case["name"], "attn_pre_gate", n)
    ref = case["ref"]["attn_pre_gate"].reshape(-1)
    rep = compare(got, ref)
    ok = not rep["n_bad"] and rep["ndiff"] <= ndiff_limit(rep["n"]) \
        and rep["cos"] >= COS_MIN and rep["meanrel"] <= MEANREL_MAX
    return ok, ("ndiff=%d/%d n_bad=%d cos=%.9f meanrel=%.3e"
                % (rep["ndiff"], rep["n"], rep["n_bad"], rep["cos"], rep["meanrel"]))


def check_source() -> int:
    """Two claims about one kernel, both re-derived from the two files rather than asserted in a comment.

    (1) THE SHIPPING CORE IS STILL `lean`. `attn_core`'s four score-dot loops must be spelled with a
        literal trip count and the `t < nd` test inside the body. A runtime `t < nd` bound is not a
        style question: LLVM's promote-alloca-to-LDS then moves `qreg[8]` and the `uint4` K-row read
        into LDS, the kernel reserves 38,944 B instead of 6,176 B, and residency on this part's
        64 KiB LDS/CU falls from 4 blocks/CU to 1 — the shipping decode attention at 8 of 32
        wavefronts per CU. So this half of the check exists to catch a REWRITE, not a drift: someone
        "simplifying" the guard back to a runtime bound keeps every numerics gate green and quietly
        costs 4x residency. `prototype/attn_decode_wide/results.md` is the measurement; the count is
        asserted equal to 4 because a check that matched nothing would pass (AGENTS.md, several rows).

    (2) THE VARIANT IS STILL A COPY. `attn_core_lean` was the candidate; after the port it is the same
        kernel under a second name, so the bodies must agree line for line modulo that name and the
        difference count must be exactly 1. Line counts equal, every line identical after the rename:
        a copy claim that is re-derived cannot rot the way a comment about a copy does (AGENTS.md, the
        geometry-fact-in-two-headers row — the working tree is where the uncommitted half hides).
    """
    frozen = (OP / "attn_decode_kernels.h").read_text().splitlines()
    body = slice_kernel(frozen, "__global__ void attn_core(CoreArgs")
    variant_text = (HERE / "attn_decode_wide_kernels.h").read_text().splitlines()
    variant = slice_kernel(variant_text, "__global__ void attn_core_lean(CoreArgs")
    if not body or not variant:
        print("   SOURCE DRIFT: could not slice a kernel body out of %s / %s"
              % (OP.name, HERE.name))
        return 1

    # ---- (1) the four dot sites, in the file the engine actually compiles ------------------------
    for label, lines in (("shipping attn_core", body), ("attn_core_lean", variant)):
        literal = [i for i, l in enumerate(lines) if LITERAL_TRIP in l]
        runtime = [i for i, l in enumerate(lines) if RUNTIME_TRIP in l]
        if len(literal) != DOT_SITES_EXPECT or runtime:
            print("   SHIPPING KERNEL REGRESSED to a runtime trip count: %s has %d of %d "
                  "literal-trip dot sites (%s) and %d spelled `%s`."
                  % (label, len(literal), DOT_SITES_EXPECT, LITERAL_TRIP, len(runtime), RUNTIME_TRIP))
            print("   A runtime `nd` bound makes LLVM promote `qreg[8]` and the K-row `uint4` to LDS: "
                  "38,944 B reserved instead of 6,176 B, 1 resident block per CU instead of 4. "
                  "Witness and numbers: prototype/attn_decode_wide/results.md; re-take with census.sh.")
            return 1
        print("   %s: %d/%d dot sites on the literal trip count, %d on a runtime bound"
              % (label, len(literal), DOT_SITES_EXPECT, len(runtime)))

    # ---- (2) the copy claim ----------------------------------------------------------------------
    if len(body) != len(variant):
        print("   SOURCE DRIFT: %d frozen lines vs %d variant lines" % (len(body), len(variant)))
        return 1
    allowed = 0
    for a, b in zip(body, variant):
        if a == b:
            continue
        if a.replace("attn_core", "attn_core_lean") == b:
            allowed += 1
            continue
        print("   SOURCE DRIFT:\n     frozen:  %s\n     variant: %s" % (a.strip(), b.strip()))
        return 1
    print("   variant is the frozen kernel: %d lines, %d permitted difference (the name; the four "
          "`nd` loop sites are now identical because the shipping kernel carries this spelling)"
          % (len(body), allowed))
    if allowed != 1:
        print("   UNEXPECTED: the name is the only difference a copy is allowed, and there %s %d"
              % ("is" if allowed == 1 else "are", allowed))
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one long case, for a fast device check")
    ap.add_argument("--no-neg", action="store_true")
    ap.add_argument("--check-source", action="store_true",
                    help="CPU only: re-derive the copy claim from the two headers and exit")
    args = ap.parse_args()
    if args.check_source:
        return check_source()
    if self_locks():
        raise SystemExit("this driver must not hold the GPU lock itself")
    if not BIN.exists():
        raise SystemExit("build first: ./build.sh")

    pinned = long_cases(verbose=False) + short_cases()
    long_src = max((c for c in pinned if c["lvalid"] > 2 * MAXSC), key=lambda c: c["lvalid"])
    cases = pinned + boundary_cases(long_src)
    if args.quick:
        cases = [long_src]
    print("cases: %d (%d pinned HF decode steps, %d derived MAXSC-boundary geometries from %s)"
          % (len(cases), len(pinned), len(cases) - len(pinned), long_src["name"]))

    ship = run_arm(cases, "ship")
    lean = run_arm(cases, "lean", ["--core", "lean"])

    print("\n1. BIT-IDENTITY  lean vs ship, every stage of every case")
    bad = 0
    for c in cases:
        line = []
        for st in STAGES:
            nd = bitdiff(stage_bytes(ship, c, st), stage_bytes(lean, c, st))
            if nd:
                bad += 1
            line.append("%s=%s" % (st, "0" if nd == 0 else nd))
        print("   %-22s lanes=%5d  %s" % (c["name"], c["lvalid"], " ".join(line)))
    print("   verdict: %s (%d stage comparisons over %d cases)"
          % ("BIT-IDENTICAL" if bad == 0 else "%d STAGES DIFFER" % bad, len(cases) * len(STAGES),
             len(cases)))

    print("\n2. AGAINST THE PINS  (floors imported: cos >= %s, meanrel <= %s)"
          % (COS_MIN, MEANREL_MAX))
    print("   REPORTED, NOT ADJUDICATED: these are the flat §10.1 fallback floors. The wide fixtures"
          " carry their")
    print("   own priced route allowance (`engine/gate.py`'s ROUTE ALLOWANCE, D-2), which this driver"
          " does not")
    print("   read, and both cores print the SAME numbers, so a difference here cannot be the"
          " candidate's.")
    pin_bad = 0
    for c in cases:
        if not c.get("ref"):
            continue
        for arm, root in (("ship", ship), ("lean", lean)):
            ok, line = judge_pin(root, c)
            pin_bad += 0 if ok else 1
            print("   %-22s %-5s lanes=%5d  %s  %s" % (c["name"], arm, c["lvalid"],
                                                       "ok " if ok else "FAIL", line))
    print("   verdict: %d pinned cases, %d outside the floors"
          % (sum(1 for c in cases if c.get("ref")), pin_bad))

    if not args.no_neg:
        print("\n3b. THE CUT, as one variable (same operands, legal cuts only)")
        print("   A block may not cover more than MAXSC lanes with `full = 1`: the staging is MAXSC")
        print("   floats, and the shipped harness refuses that geometry rather than measuring it. So")
        print("   `--split 1` on a wide row is NOT probed here — the first version of this section ran")
        print("   it and printed cos 0.049 and -0.013, which are the numbers an out-of-bounds staging")
        print("   produces, not a route. The legal question is the other direction: how deep can the")
        print("   cut go before the streaming sum moves?")
        deep = [c for c in cases if c["lvalid"] >= 2048]
        # The probe set is derived per case from the geometry: the natural cut (what the engine books,
        # `split_for(lanes)`) and then each doubling up to the engine's own ceiling `kAttnMaxSplit = 64`
        # (`engine/kernels_attn.cpp:88`). A probe below the natural cut is not a deeper cut, it is the
        # staging overflow the binary now refuses, so it cannot appear in this list by construction.
        K_MAX_SPLIT = 64
        for c in deep:
            natural = -(-c["lvalid"] // MAXSC)
            probes, s = [], natural
            while s <= K_MAX_SPLIT:
                probes.append(s)
                s *= 2
            for s in probes:
                root = run_arm([c], "lean_split%d" % s, ["--core", "lean", "--split", str(s)])
                ok, line = judge_pin(root, c)
                vs = bitdiff(stage_bytes(root, c, "attn_pre_gate"),
                             stage_bytes(lean, c, "attn_pre_gate"))
                span = -(-c["lvalid"] // s)
                print("   %-22s lanes=%5d split=%3d span=%4d tiles=%3d blocks=%4d  vs pin: %s  |"
                      " bytes moved vs the natural cut: %d"
                      % (c["name"], c["lvalid"], s, span, -(-span // 8), N_Q * s, line, vs))

    if not args.no_neg:
        print("\n3. NEGATIVES  (must-catch and must-not-catch both DERIVED from the geometry)")
        neg_bad = 0
        for neg, spec in NEGS.items():
            a = run_arm(cases, "ship_neg_" + neg, ["--neg", neg])
            b = run_arm(cases, "lean_neg_" + neg, ["--core", "lean", "--neg", neg])
            moved = must_catch = must_not = 0
            for c in cases:
                why = exempt_reason(c, spec["exempt"])
                must_catch += 0 if why else 1
                must_not += 1 if why else 0
                # Only the stages the injection can reach are adjudicated. Which those are is a property
                # of the injection: `gatehalf` moves `gated` and cannot move `attn_pre_gate`, and the
                # first run of this gate watched `attn_pre_gate` alone and reported a live injection as
                # 0-of-21 — the vacuous-control failure, reached from the other direction.
                moved_st, differ_st = [], []
                for st in spec["stages"]:
                    if bitdiff(stage_bytes(a, c, st), stage_bytes(ship, c, st)):
                        moved_st.append(st)
                    if bitdiff(stage_bytes(a, c, st), stage_bytes(b, c, st)):
                        differ_st.append(st)
                if differ_st:
                    neg_bad += 1
                    print("   %s %s: lean and ship DIFFER under this injection (%s)"
                          % (neg, c["name"], ",".join(differ_st)))
                if moved_st:
                    moved += 1
                if why and moved_st:
                    neg_bad += 1
                    print("   %s %s: an EXEMPT row moved (%s) — exemption was '%s'"
                          % (neg, c["name"], ",".join(moved_st), why))
                if not why and not moved_st:
                    neg_bad += 1
                    print("   %s %s: MOVED NOTHING among %s — dead injection, or the exemption rule "
                          "is wrong" % (neg, c["name"], ",".join(spec["stages"])))
            print("   %-9s moved %2d/%2d must-catch; %d must-not-catch (stages: %s)"
                  % (neg, moved, must_catch, must_not, ",".join(spec["stages"])))
        print("   verdict: %d negative-control failures" % neg_bad)
    else:
        neg_bad = 0

    print("\nRESULT d3gate cases=%d bitdiff_stages=%d pinned_off_floor=%d negative_failures=%d"
          % (len(cases), bad, pin_bad, neg_bad))
    print("no timing taken by this driver; --mode burst is the main agent's, under its own lock hold.")
    return 1 if (bad or neg_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
