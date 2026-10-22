#!/usr/bin/env python3
"""Correctness gate for prototype/mtp_headbatch — no timing, no ms, no verdict about speed.

WHAT IS BEING CLAIMED, and each claim gets its own catcher:
  C1 one pass over the table produces logits for all M rows            -> byte witness (fetches)
  C2 the batched head's row q equals that row run alone                -> cross-M bit-identity
  C3 the batched head is the shipped head, widened                     -> shipped-kernel bit-identity
  C4 the logits are the right numbers on real pinned data              -> §10.1 per-element rule
  C5 every output element is written and nothing is written outside it -> poison + guard bands
  C6 the instantiation that gets clocked equals the one being witnessed-> witness/timed bit-identity

THE MUST-CATCH SET IS DERIVED FROM THE GEOMETRY, not typed. The geometry is one warp per vocab row
and M staged query rows read as `hs[q * HID + ...]`, so the control families follow:
  D1 for each q in [0,M): an injection confined to query row q must move EVERY element of output row
     q (must-catch) and NO element of any other row (must-not-catch). A must-catch set with only the
     must-catch half is vacuous — AGENTS: a negative test that catches everything catches nothing.
     Count = sum over M in {2,4} of M cases, and M*(M-1) frozen-row assertions.
  D2 `row_dup` (every query row reads staged row 0) must move rows 1..M-1 and leave row 0
     bit-identical. Row 0 is the derived must-not-catch: the bug is exactly "row 1 rode row 0's data",
     and the correct row is the proof the detector is not just flagging every change.
  D3 `short_grid` (one block fewer) must leave exactly the last grid_eff*rpb..V vocab rows
     unwritten in EVERY query row, and leave everything else bit-identical. The hole set is derived
     from the grid the binary ECHOED, not from a hand-listed span (hand-listed sets rot into
     "everything" the moment the fixture changes). A one-pass row has no partial slot to steal, which
     is the `partalias` lesson re-appearing one layer over.
  D4 `extra_pass` (two launches, two accumulation windows) must drive fetch_per_weight to 2.000000
     while leaving the logits bit-identical. This is the calibration of the byte witness: it is
     required to detect the exact failure plan A rules out, and the logits alone cannot see it —
     if the witness misses this, C1 is decoration.
  D5 `extra_chunk` must move the checksum and the fetch count and therefore the values.
Every case runs in its own process (AGENTS, `tools/mixed_bw`): one faulting kernel poisons the
context and every case after it silently never ran.

NO TIMING. `--mode check` prints no ms field and this script prints no millisecond figure; wall time
is reported as wall time only. The timing arms live in run_locked.sh and belong to the main agent.

    .venv/bin/python prototype/mtp_headbatch/gate.py            # all cases (device)
    .venv/bin/python prototype/mtp_headbatch/gate.py --quick    # the load-bearing subset
    .venv/bin/python prototype/mtp_headbatch/gate.py --selftest # CPU only: the control machinery
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
WORK = Path("/tmp/mtp_headbatch")
BIN = HERE / "mtp_headbatch"
V, H = 248320, 1024
POISON = np.uint32(0x7FC00001)
RPB = 8                      # rows per block, the rig default (= lm_head_gemv's default)
Ms = (2, 4)                  # batch widths under test; M=1 is the degenerate case


# ---------------------------------------------------------------- the numbers the band is made of
def head_gate():
    """The shipped head gate, imported rather than re-typed, so this rig cannot quietly own a second
    set of thresholds. prototype/lm_head_gemv/run.py owns them and `floor_probe.py` sized them."""
    path = ROOT / "prototype/lm_head_gemv/run.py"
    spec = importlib.util.spec_from_file_location("lm_head_gate_owner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lm_head_gate_owner"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- running one case (own process)
def run_case(name, M, xfile, arm="hb", rpb=RPB, inject="none", inject_q=None, mode="check"):
    out = WORK / "out" / f"{name}.bin"
    out2 = WORK / "out" / f"{name}.timed.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "--w", str(WORK / "W.bin"), "--x", str(WORK / xfile), "--M", str(M),
           "--arm", arm, "--rpb", str(rpb), "--mode", mode, "--inject", inject,
           "--out", str(out)]
    if inject_q is not None:
        cmd += ["--inject-q", str(inject_q)]
    if mode == "check":
        cmd += ["--out2", str(out2)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    fields = {}
    for line in p.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        if tag in ("PLAN", "WITNESS", "RESULT", "TIMEDFORM", "VERDICT", "NO-TIMING"):
            if tag == "VERDICT":
                fields[tag] = {"VERDICT": line.split()[1]}
            else:
                fields[tag] = dict(kv.split("=", 1) for kv in line.split(" ", 1)[1].split())
            fields[tag]["_tag"] = tag
    if "RESULT" not in fields and "NO-TIMING" not in fields:
        return {"err": f"no RESULT line (rc={p.returncode})\n{p.stdout}\n{p.stderr}", "rc": p.returncode}
    fields["_wall"] = wall
    fields["_rc"] = p.returncode
    fields["_path"] = out
    fields["_path2"] = out2
    fields["_cmd"] = " ".join(cmd)
    return fields


def rows_of(path, M):
    """The logits as written: M rows of V F32, read as raw bits so 'bit-identical' is literal."""
    a = np.fromfile(path, dtype=np.float32)
    if a.size != M * V:
        raise SystemExit(f"{path}: {a.size} elements, expected {M * V} — a wrong-size read would "
                         f"make every comparison below meaningless")
    return a.view(np.uint32).reshape(M, V)


# ---------------------------------------------------------------- the derived expectations
def derived_expectations(M, rpb=RPB, grid=None):
    """Everything a case is required to do, as a function of the geometry alone."""
    grid = grid or (V + rpb - 1) // rpb
    e = {}
    e["dropk"] = {"moved": [q for q in range(M)], "frozen": [], "verdict_ok": True,
                  "why": "a per-query-row bug must move its own row only"}
    e["row_dup"] = {"moved": list(range(1, M)), "frozen": [0], "verdict_ok": True,
                    "why": "row 0 is the derived must-not-catch for 'row 1 rode row 0'"}
    e["row_shift"] = {"moved": list(range(M)), "frozen": [], "verdict_ok": True,
                      "why": "a cyclic relabelling moves every row when M>1"}
    holes_start = (grid - 1) * rpb
    e["short_grid"] = {"moved": [], "frozen": list(range(M)), "verdict_ok": False,
                       "holes": (M, holes_start, V),
                       "why": f"grid-1 leaves vocab rows [{holes_start},{V}) unwritten in all "
                              f"{M} query rows"}
    e["extra_pass"] = {"moved": [], "frozen": list(range(M)), "verdict_ok": False,
                       "fetch": 2.0, "why": "two cold passes must demand two table reads"}
    e["extra_chunk"] = {"moved": list(range(M)), "frozen": [], "verdict_ok": False,
                        "fetch": (H // 8 + 32) / (H // 8),
                        "why": "an over-read changes the fetched byte multiset and the values"}
    return e


def counts():
    """The derived control counts. A control set that can be empty cannot fail."""
    mc_dropk = sum(M for M in Ms)
    mnc_dropk = sum(M * (M - 1) for M in Ms)
    mc_dup = sum(M - 1 for M in Ms)
    mnc_dup = len(Ms)
    mc_short = sum(M for M in Ms)
    mc_pass = len(Ms)
    return dict(mc_dropk=mc_dropk, mnc_dropk=mnc_dropk, mc_dup=mc_dup, mnc_dup=mnc_dup,
                mc_short_holes=mc_short, mc_extra_pass=mc_pass,
                crossM=sum(M for M in Ms), solo=4)


def check_injection(name, M, inj, clean_bits, exp, res):
    """Compare an injected run against the clean run of the same shape, per query row."""
    got = res.get("_bits")
    if got is None:
        got = rows_of(res["_path"], M)
    bad = []
    moved = [bool(np.any(got[q] != clean_bits[q])) for q in range(M)]
    for q in exp["moved"]:
        if not moved[q]:
            bad.append(f"MUST-CATCH missed: {inj} did not move output row {q} ({exp['why']})")
    for q in exp["frozen"]:
        if "holes" in exp:
            continue        # a coverage hole is a legitimate difference; the holes block below
                            # checks the frozen part over exactly the covered prefix
        if moved[q]:
            n = int(np.sum(got[q] != clean_bits[q]))
            bad.append(f"MUST-NOT-CATCH fired: {inj} moved {n} elements of output row {q}")
    if "holes" in exp:
        n_rows, lo, hi = exp["holes"]
        for q in range(n_rows):
            if not np.all(got[q][lo:hi] == POISON):
                bad.append(f"MUST-CATCH missed: {inj} left no poison in rows[{lo}:{hi}] of query "
                           f"row {q}; the coverage hole is not where the grid says it is")
            if np.any(got[q][:lo] != clean_bits[q][:lo]):
                bad.append(f"MUST-NOT-CATCH fired: {inj} touched query row {q} before the hole")
    if "fetch" in exp:
        got_f = float(res["RESULT"]["fetch_per_weight"])
        if abs(got_f - exp["fetch"]) > 1e-6:
            bad.append(f"byte witness is blind: {inj} must report fetch_per_weight "
                       f"{exp['fetch']:.6f}, got {got_f:.6f}")
    want = "OK" if exp["verdict_ok"] else "BROKEN"
    if res["VERDICT"]["VERDICT"] != want:
        bad.append(f"rig verdict is {res['VERDICT']['VERDICT']}, expected {want} "
                   f"for {inj} ({exp['why']})")
    return bad, moved


# ---------------------------------------------------------------- the case list
def build_cases(quick):
    """(name, kwargs) in execution order. The clean runs come first: every injection is judged
    against them, so a failed clean run must stop the run rather than produce 20 false greens."""
    cs = [("solo_row0", dict(M=1, xfile="xrow0.bin")),
          ("solo_row1", dict(M=1, xfile="xrow1.bin")),
          ("solo_row2", dict(M=1, xfile="xrow2.bin")),
          ("solo_row3", dict(M=1, xfile="xrow3.bin")),
          ("clean_M1", dict(M=1, xfile="x1.bin")),
          ("shipped_M1", dict(M=1, xfile="xrow0.bin", arm="shipped")),
          ("clean_M2", dict(M=2, xfile="x2.bin"))]
    if not quick:
        cs.append(("clean_M4", dict(M=4, xfile="x4.bin")))
    for M in Ms:
        if quick and M == 4:
            continue
        for q in range(M):
            cs.append((f"dropk_M{M}_q{q}", dict(M=M, xfile=f"x{M}.bin", inject="dropk", inject_q=q)))
        cs.append((f"row_dup_M{M}", dict(M=M, xfile=f"x{M}.bin", inject="row_dup")))
        if not quick:
            cs.append((f"row_shift_M{M}", dict(M=M, xfile=f"x{M}.bin", inject="row_shift")))
            cs.append((f"short_grid_M{M}", dict(M=M, xfile=f"x{M}.bin", inject="short_grid")))
            cs.append((f"extra_pass_M{M}", dict(M=M, xfile=f"x{M}.bin", inject="extra_pass")))
            cs.append((f"extra_chunk_M{M}", dict(M=M, xfile=f"x{M}.bin", inject="extra_chunk")))
    return cs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="the load-bearing subset only")
    ap.add_argument("--selftest", action="store_true", help="CPU-only: prove the machinery can fail")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not BIN.exists():
        subprocess.run(["bash", str(HERE / "build.sh")], check=True)
    else:
        # Rebuild if any source is newer than the binary. This is not politeness: a stale binary was
        # run into a device fault here, and AGENTS' 2026-10-17 rule says a stale arm reads a clean
        # null — the gate must not be able to judge a build that is not the code on disk.
        newest_src = max((p.stat().st_mtime for p in
                          (HERE / "mtp_headbatch.hip.cpp", HERE / "hb_kernels.h",
                           HERE.parent / "lm_head_gemv" / "lm_head_gemv_kernels.h")), default=0)
        if newest_src > BIN.stat().st_mtime:
            print("binary older than its sources; rebuilding")
            subprocess.run(["bash", str(HERE / "build.sh")], check=True)

    c = counts()
    print("derived control counts (from the geometry, not typed): " +
          " ".join(f"{k}={v}" for k, v in c.items()))
    for k, v in c.items():
        if v <= 0:
            print(f"FAIL: derived count {k} is {v}; an empty control set cannot fail")
            return 1

    cases = build_cases(args.quick)
    results, failures = {}, []
    print(f"\n== {len(cases)} cases, one process each (no timing: wall is wall, not a measurement) ==")
    for name, kw in cases:
        res = run_case(name, **kw)
        results[name] = res
        if "err" in res:
            failures.append(f"{name}: {res['err']}")
            print(f"  {name:20} CRASH  {res['err'].splitlines()[0]}")
            break
        v = res["VERDICT"]["VERDICT"]
        print(f"  {name:20} {v:6} rc={res['_rc']} "
              f"fetch={res['RESULT'].get('fetch_per_weight')} poison={res['RESULT'].get('poison_left')}"
              f" grid={res['RESULT'].get('grid')}")
        # Every clean case must be green before anything is judged against it.
        if name.startswith(("clean_", "solo_", "shipped_")):
            if v != "OK":
                failures.append(f"{name}: clean case is {v}; the injections below would be judged "
                                f"against a broken reference")
                break
            if res["_rc"] != 0:
                failures.append(f"{name}: VERDICT OK but rc={res['_rc']}")
            if ("TIMEDFORM" in res and res["TIMEDFORM"]["differs_from_witness"] not in ("no", "n/a")):
                failures.append(f"{name}: the timed instantiation does not produce the witnessed "
                                f"logits (C6) — the instrument would be the subject")
            if ("TIMEDFORM" in res and res["TIMEDFORM"]["poison_left"] != "0"):
                failures.append(f"{name}: the timed instantiation left unwritten logits")
            if name == "clean_M1" and res["RESULT"].get("arm") != "hb":
                failures.append("clean_M1 is not the hb arm; the M=1 identity below is meaningless")

    if not failures:
        failures += cross_checks(results)
        failures += pin_checks(results)
        failures += injection_checks(results, args.quick)

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS — claims C1..C6 hold on the pinned inputs; the byte witness, the per-element rule and "
          "the poison coverage each caught something the others could not (see the per-case lines).")
    print("No millisecond figure was printed by this gate. Timing arms: run_locked.sh (main agent).")
    return 0


def cross_checks(results):
    """C2 and C3: batched rows are the same bits the same rows produce alone, and the same bits the
    shipping M=1 kernel produces. Bit-exact, because the dot order was transcribed, not paraphrased."""
    bad = []
    solo = {i: rows_of(results[f"solo_row{i}"]["_path"], 1)[0]
            for i in range(4) if f"solo_row{i}" in results}
    shipped = rows_of(results["shipped_M1"]["_path"], 1)[0]
    hb1 = rows_of(results["clean_M1"]["_path"], 1)[0]
    n = int(np.sum(hb1 != shipped))
    if n:
        bad.append(f"C3: hb M=1 differs from the SHIPPED lm_head_gemv_full in {n}/{V} elements; "
                   f"the generalisation moved the arithmetic")
    for M in (2, 4):
        key = f"clean_M{M}"
        if key not in results:
            continue
        bits = rows_of(results[key]["_path"], M)
        for q in range(M):
            if q not in solo:
                continue
            n = int(np.sum(bits[q] != solo[q]))
            if n:
                bad.append(f"C2: M={M} row {q} differs from that row run alone in {n}/{V} elements")
        n = int(np.sum(bits[0] != shipped))
        if n:
            bad.append(f"C3: M={M} row 0 differs from the shipped kernel in {n}/{V} elements")
    return bad


def pin_checks(results):
    """C4 on real pinned data: row 0 is dec8's `final_hidden`, and dec8/decode7.logits_last is the
    row that x produced — `make_inputs.py` proved that pairing before writing anything."""
    gate = head_gate()
    ref = np.load(ROOT / "data/refs/dec8/decode7.logits_last.npy").reshape(-1)
    got = rows_of(results["clean_M1"]["_path"], 1)[0].view(np.float32)
    rep, violations = gate.per_element_rule(got, ref)
    cos = float(got.astype(np.float64) @ ref.astype(np.float64)
                / (np.linalg.norm(got) * np.linalg.norm(ref)))
    meanrel = float(np.abs(got - ref).mean() / np.abs(ref).mean())
    print(f"\n== row 0 vs the pinned logits (dec8/decode7.logits_last), §10.1 logits class ==\n"
          f"   argmax {int(np.argmax(got))} vs {int(np.argmax(ref))} | cos {cos:.6f} | "
          f"meanrel {meanrel:.5f} | ndiff {rep['ndiff']} ({rep['ndiff'] / rep['n'] * 100:.3f} %) "
          f"| n_bad {rep['n_bad']}")
    return [f"C4 per-element rule: {v}" for v in violations]


def injection_checks(results, quick):
    """Judge every injection case that actually ran, deriving its expectation from the case name —
    i.e. from the run, not from a hand-listed table that can rot when the case list changes."""
    bad = []
    pat = re.compile(r"(dropk|row_dup|row_shift|short_grid|extra_pass|extra_chunk)_M(\d+)"
                     r"(?:_q(\d+))?$")
    for name in results:
        m = pat.match(name)
        if not m:
            continue
        inj, M, q = m.group(1), int(m.group(2)), m.group(3)
        clean = rows_of(results[f"clean_M{M}"]["_path"], M)
        exp = derived_expectations(M)[inj]
        if inj == "dropk":
            qi = int(q)
            exp = dict(exp, moved=[qi], frozen=[x for x in range(M) if x != qi])
        b, moved = check_injection(name, M, inj, clean, exp, results[name])
        bad += b
        print(f"   {name:22} moved={''.join('1' if mm else '0' for mm in moved)} "
              f"required={''.join('1' if x in exp['moved'] else '0' for x in range(M))} "
              f"{'ok' if not b else 'BAD'}")
    return bad


# ---------------------------------------------------------------- CPU-only self-witness
def selftest() -> int:
    """The gate must be shown to be capable of failing (AGENTS: a check whose input list can be
    empty cannot pass; a negative control with an empty must-catch set is not a control)."""
    print("== gate selftest (CPU only; no device, no timing) ==")
    ok = True
    c = counts()
    for k, v in c.items():
        good = v > 0
        ok &= good
        print(f"  derived count {k:16} = {v:3}  {'ok' if good else '<<< EMPTY SET, cannot fail'}")

    # A synthetic clean pair, and three synthetic "injections" whose only difference is which rows
    # moved. The machinery must catch two and refuse to congratulate itself on the third.
    rng = np.random.default_rng(7)
    clean = rng.integers(0, 1 << 31, size=(2, 64)).astype(np.uint32)

    def fake(inj, moved_rows, holes=None, fetch=1.0, verdict="OK"):
        got = clean.copy()
        for q in moved_rows:
            got[q] = (got[q] + 1).astype(np.uint32)
        if holes:
            _, lo, hi = holes
            got[:, lo:hi] = POISON
        return {"_bits": got, "VERDICT": {"VERDICT": verdict},
                "RESULT": {"fetch_per_weight": f"{fetch:.6f}"}}

    exps = derived_expectations(2)
    cases = [
        ("row_dup correct behaviour", exps["row_dup"], fake("dup_ok", [1]), 0),
        ("row_dup but row 1 did NOT move (must-catch miss)", exps["row_dup"], fake("dup_bad", []), 1),
        ("row_dup but row 0 also moved (must-not-catch fire)", exps["row_dup"],
         fake("dup_bad2", [0, 1]), 1),
        ("extra_pass but the witness stayed at 1.0 (blind witness)", exps["extra_pass"],
         fake("pass_bad", [], fetch=1.0, verdict="BROKEN"), 1),
        ("short_grid with the hole where the grid says (must-catch hit)", exps["short_grid"],
         fake("sg_ok", [], holes=(2, 56, 64), verdict="BROKEN"), 0),
        ("short_grid with the hole in the wrong place", exps["short_grid"],
         fake("sg_bad", [], holes=(2, 0, 8), verdict="BROKEN"), 1),
    ]
    for name, exp, res, want_bad in cases:
        b, _ = check_injection(name, 2, name, clean, exp, res)
        good = (len(b) > 0) == bool(want_bad)
        ok &= good
        print(f"  {name:52} -> {'caught' if b else 'passed'}  "
              f"{'ok' if good else '<<< MACHINERY IS ASLEEP'}")
        for x in b:
            print(f"       {x}")

    # The band must be importable, and it must be the shipped head gate's own numbers.
    try:
        gate = head_gate()
        print(f"  band imported from prototype/lm_head_gemv/run.py: ulp<={gate.PER_ELEMENT_ULP_MAX} "
              f"cancel<={gate.PER_ELEMENT_CANCEL_REL:g} ndiff<={gate.PER_ELEMENT_NDIFF_PCT_MAX} %")
    except Exception as e:
        ok = False
        print(f"  band import FAILED: {e}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
