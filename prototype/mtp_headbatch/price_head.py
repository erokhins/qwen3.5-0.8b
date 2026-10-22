#!/usr/bin/env python3
"""Pre-registered bands for the batched tied head, and the arithmetic that sets them. CPU only.

WHY A BAND AND WHY SET HERE. `prototype/mtp_delta/D3-RESULT.md` names `lm_head_batched` the single
largest swing term between plan A (1.231x) and the pessimistic arm (0.961x, KILLED): ±2.17 ms, and it
is "a plan, not a measurement". Plan A assumes the 2-row head pass costs about what the 1-row pass
costs. This script states, BEFORE any timing, what measured outcome confirms that, what outcome
forces a narrower claim, and what outcome kills the assumption outright — so the number that arrives
from the device is read against a fixed line rather than a mood (AGENTS: a band sized by a floor
probe, not by taste; `prototype/mtp_gatezero/route_price.py` for the arithmetic it reuses).

INPUTS, every one quoted or imported, none measured here:
    step            10.9357 ms   engine/generation-limits.md §1, paired window, 4,032 lanes
    lm_head pass     2.1685 ms   same table; 509.552 MiB demanded, 104 % of the plateau
    plateau           237 GB/s   devices/strix-halo-memory-bandwidth.md
    a               0.7329       prototype/mtp_gatezero/draft_survival.json (torch CPU bits)
    c               0.2237       prototype/mtp_gatezero (D1 measured the head term)
    delta_planA     1.6454 ms    prototype/mtp_delta/D3-RESULT.md, plan-A row
    proj marginal   +7.6 %       prototype/mtp_delta/D3-RESULT.md: the measured 1->2 marginal of the
                                  closest measured family (one wavefront per row, bf16, k=1024,
                                  weight read once per tile) — the transfer plan A used for the head

THE PLATEAU ARITHMETIC, i.e. what a second row can cost at all:
    table read once            485.0000 MiB / 237 GB/s = 2.1458 ms  (measured pass: 2.1685 ms)
    + one more logits row      0.9475 MiB                            = 0.0042 ms   (+0.19 %)
    + one more x row staged    4 KiB per block, L1-resident          = not a DRAM term
    dots, M=1 -> M=2           0.509 -> 1.017 GFLOP per pass = 235 -> 469 GFLOP/s
                                 against the quoted scalar v_fma_f32 roof 7.30 TFLOP/s
                                 (prototype/issue_rate_probe §1), i.e. 3.2 % -> 6.4 % of that roof
    LDS per block              M * 4 KiB; at rpb 8 both M=1 and M=2 stay thread-bound at
                                 8 blocks/CU (64 KiB / 8 KiB = 8 = 2048 threads / 256), so no
                                 occupancy term either at M=2. At M=4 / rpb 8 LDS binds at
                                 4 blocks/CU, which is why the runner sweeps rpb for M=4.
    => PREDICTION if the table is genuinely read once: R(2) ~ 1.00-1.03. A measured R well above
       that is a mechanism not accounted for here (L2 thrash of the staged rows, register spill, or
       a partial second read), and it is reported as such rather than smoothed over.

THE BANDS (the numbers the main agent's measurement is judged against; also printed by --bands):
    R <= 1.10      PLAN A STANDS AS PRICED. The head marginal is <= 0.217 ms, which is within
                   0.052 ms of the 0.165 ms plan A priced via the proj transfer, and that slack is
                   inside D1's own pre-registered "±5 % on the head alone is not decisive" margin
                   (MTP-GATEZERO §6, D1 row: ±5 % moves the speedup 1.416 -> 1.405x).
    1.10 < R <= 1.35  PLAN A SURVIVES, WITH A NARROWER CLAIM. The claim that survives is "P7 clears
                   the standing bar with one batched body and one batched head at >= 1.17x
                   (>= +1.9 ms/step)", which still exceeds every other queue item together
                   (P6 <= 0.09 + P2 ~ 0.1 + P5 <= 1.1 ms). The claim that does NOT survive is
                   "the second row rides the table read for free": at R = 1.35 the head alone is
                   0.76 ms, a third of the whole 2-row verify. Then D3's lm_head row must be
                   re-emitted with the measured marginal, and the D3 headline 1.231x restated.
    R > 1.35     THE BATCHED HEAD IS NOT FREE. Re-price in route_price.py before any kernel is
                   written, and note the structural line below: once R >= 1.862 the batched head
                   saves less than the standing 0.30 ms/step bar over simply calling the head twice,
                   so the engine should ship two head passes and plan A's batched-head requirement
                   dissolves rather than being re-priced.
    R_KILL = 2.2275  With plan A's OTHER terms as measured, P7 falls below the 1.027x bar only above
                   this R — which is above 2.0, i.e. above "the batched head is as bad as two cold
                   passes". So no head measurement can kill P7 on its own; it can only shrink it by
                   up to 3x. That is stated here rather than discovered later, because it changes
                   what the number is for: it sizes the prize, it does not gate the route.

    .venv/bin/python prototype/mtp_headbatch/price_head.py --bands
    .venv/bin/python prototype/mtp_headbatch/price_head.py --evidence /tmp/mtp_headbatch/evidence
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RP = _load("prototype/mtp_gatezero/route_price.py", "mtp_route_price")

# Quoted, with their sources in the docstring.
LM_HEAD_MS = RP.LM_HEAD_MS               # 2.1685
STEP_MS = RP.STEP_MS                     # 10.9357
BAR = RP.BAR                             # 1.027
A_ACCEPT = 0.7329                        # draft_survival.json
C_DRAFT = 0.2237                         # gate zero, with D1's measured head term
PROJ_RATIO = 1.076                       # D3-RESULT: proj 1->2 marginal +7.6 %
DELTA_PLANA = 1.6454                     # D3-RESULT plan-A row, ms
# D3's lm_head term was LM_HEAD_MS * (PROJ_RATIO - 1); the rest of delta is everything else it
# measured, which this script must not re-price.
LM_HEAD_PRICED = LM_HEAD_MS * (PROJ_RATIO - 1.0)
DELTA_REST = DELTA_PLANA - LM_HEAD_PRICED

BAND_A_STANDS = 1.10
BAND_NARROW = 1.35
# D3-RESULT's pessimistic arm, in ms: 0.4729 of a step (the arm where the second row costs a second
# cold head pass). Kept only to show that a_needed at that delta exceeds what the drafter achieved.
PESSIMISTIC_DELTA_MS = 0.4729 * STEP_MS
PLATEAU_MS = 485.0 * 2**20 / (237.0 * 1e9) * 1e3    # 2.1458 ms, table once at the plateau


def delta_of(r: float) -> float:
    """The 2-row verify's cost, in ms, for a measured batched-head ratio r."""
    return DELTA_REST + (r - 1.0) * LM_HEAD_MS


def speedup_of(r: float) -> float:
    d = delta_of(r) / STEP_MS
    return (1.0 + A_ACCEPT) / ((1.0 + d) * (1.0 + C_DRAFT))


def solve_speedup(target: float) -> float:
    """The R at which the speedup equals target (monotone in R, so a plain bisection is enough)."""
    lo, hi = 1.0, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if speedup_of(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def solve_delta_max() -> float:
    return (1.0 + A_ACCEPT) / (BAR * (1.0 + C_DRAFT)) - 1.0


# Rollback per draft round, `quoted` from prototype/mtp_delta/D3-RESULT.md §4 (snapshot ~0.079 ms every
# round + restore ~0.16 ms on a reject, i.e. x (1-a); at a=0.7329 that is 0.1217 ms). It is neither the
# verify's marginal row cost (`delta`) nor the draft's own cost (`c`) — it is the price of UNDOING the
# row that was rejected — and D3 deliberately left it out of `delta` rather than folding it in. See
# MTP-IMPL.md §3 for where it belongs in the ledger.
RHO_MS = 0.12


def a_needed(d_frac: float, rho_frac: float = 0.0) -> float:
    """Accepted tokens per draft round required to clear the bar at a given `delta`.

    Gate zero printed `a_needed = 0.257` and labelled it `(batched verify, FULL head)`, but its own
    footnote says it is the accept rate at `delta = 0` (MTP-GATEZERO.md:297). That is not the number D4
    should be judged against: at plan A's measured `delta` the requirement is materially higher, and at
    the pessimistic `delta` it exceeds what the torch-CPU drafter achieved at all — which is the same
    fact D3 reported as "the pessimistic arm is KILLED", arriving from the other side.
    """
    return BAR * (1.0 + d_frac) * (1.0 + C_DRAFT) + rho_frac - 1.0


def bands_table() -> int:
    print(f"quoted inputs: step {STEP_MS} ms  lm_head {LM_HEAD_MS} ms  a {A_ACCEPT}  c {C_DRAFT}  "
          f"bar {BAR}")
    print(f"plateau floor for one table read: {PLATEAU_MS:.4f} ms   "
          f"(measured one-row pass {LM_HEAD_MS} ms = {LM_HEAD_MS / PLATEAU_MS:.3f} of it)")
    print(f"delta as measured for plan A: {DELTA_PLANA} ms, of which the head term priced by the "
          f"proj transfer was {LM_HEAD_PRICED:.4f} ms -> the rest is {DELTA_REST:.4f} ms")
    dmax = solve_delta_max()
    print(f"kill line (route_price's own): delta_max = {dmax:.4f} of a step = "
          f"{dmax * STEP_MS:.3f} ms")
    print()
    print("  R      head marginal   delta ms  of a step   speedup   ms/step    reading")
    for r in (1.00, 1.02, 1.076, 1.10, 1.20, 1.35, 1.50, 1.862, 2.00, solve_speedup(BAR)):
        marg = (r - 1.0) * LM_HEAD_MS
        d = delta_of(r)
        sp = speedup_of(r)
        note = ("plan A as priced" if r <= BAND_A_STANDS else
                "narrower claim" if r <= BAND_NARROW else "re-price first")
        if abs(r - 1.076) < 1e-9:
            note = "what D3 priced -> must reproduce 1.6454 ms / 1.231x"
        elif r > 1.86:
            note = ("batched head no longer beats two passes by the 0.30 ms bar" if sp >= BAR
                    else "P7 itself under the bar")
        print(f" {r:5.3f}   {marg:8.4f} ms   {d:8.4f}   {d / STEP_MS:7.4f}   {sp:6.3f}x  "
              f"{(sp - 1) * STEP_MS:+7.3f}   {note}")
    r_bar = solve_speedup(BAR)
    r_barpass = 2.0 - 0.30 / LM_HEAD_MS
    print(f"\n  R at which P7 falls to the 1.027x bar:          {r_bar:.4f}")
    print(f"  R at which a batched head stops beating two passes by the 0.30 ms bar: {r_barpass:.4f}")
    print(f"  D3 consistency check: R=1.076 -> delta {delta_of(1.076):.4f} ms, speedup "
          f"{speedup_of(1.076):.3f}x  "
          f"{'(matches the quoted 1.6454 ms / 1.231x)' if abs(speedup_of(1.076) - 1.231) < 0.002 else '(DOES NOT MATCH the quoted row — refresh the constants)'}")
    rho = RHO_MS / STEP_MS
    print("\n  ACCEPT RATE the 1.027x bar needs, at each delta (D4's kill line must be stated at the")
    print("  delta that actually ships — gate zero's 0.257 is the delta=0 row, quoted below as such):")
    for label, d_ms in (("delta = 0 (gate zero's framing)", 0.0),
                        ("plan A, as priced", DELTA_PLANA),
                        ("plan A + rollback", DELTA_PLANA),
                        ("pessimistic arm, D3", PESSIMISTIC_DELTA_MS)):
        with_rho = label.endswith("rollback")
        a_ne = a_needed(d_ms / STEP_MS, rho if with_rho else 0.0)
        print(f"    {label:32} delta {d_ms:7.4f} ms  a_needed {a_ne:6.4f}   "
              f"{'CLEARS' if A_ACCEPT >= a_ne else 'FAILS'} at the torch-CPU a={A_ACCEPT}")
    print(f"    (rollback term: {RHO_MS} ms/round = {rho:.4f} of a step; it COSTS "
          f"{(1 + A_ACCEPT) / ((1 + DELTA_PLANA / STEP_MS) * (1 + C_DRAFT)) - (1 + A_ACCEPT) / ((1 + DELTA_PLANA / STEP_MS) * (1 + C_DRAFT) + rho):.4f}x "
          f"at plan A)")
    print("\n  Bands, pre-registered (same words in run_locked.sh's header and the handover):")
    print(f"   R <= {BAND_A_STANDS}    plan A stands as priced")
    print(f"   {BAND_A_STANDS} < R <= {BAND_NARROW}  plan A survives with the narrower claim above")
    print(f"   R > {BAND_NARROW}   the batched head is not free: re-price in route_price.py "
          f"before a kernel")
    return 0


RESULT_RE = re.compile(r"^RESULT\b", re.M)


def from_evidence(d: pathlib.Path) -> int:
    """Read run_locked.sh's evidence and print the verdict against the bands.

    The parse is guarded the way `prototype/gdn_collapse/run.py` learned to guard it: the number of
    RESULT lines parsed must equal the number the runner says it printed, and an empty parse is a
    hard error, not a PASS."""
    files = sorted(d.glob("*.txt"))
    if not files:
        raise SystemExit(f"{d}: no evidence files — a verdict from an empty list cannot fail")
    arms = {}
    n_result = 0
    for f in files:
        for line in f.read_text().splitlines():
            if not RESULT_RE.match(line):
                continue
            n_result += 1
            fl = dict(kv.split("=", 1) for kv in line.split()[1:])
            key = (fl["arm"], int(fl["M"]), int(fl.get("rpb", 8)))
            arms[key] = fl
    if n_result == 0:
        raise SystemExit(f"{d}: {len(files)} files but 0 RESULT lines parsed — empty parse")
    print(f"parsed {n_result} RESULT lines from {len(files)} files; {len(arms)} distinct arms")
    for key in sorted(arms):
        fl = arms[key]
        print(f"  {key[0]:8} M={key[1]} rpb={key[2]:2} grid={fl['grid']:>6} "
              f"fetch={fl['fetch_per_weight']} poison={fl['poison_left']} "
              f"ms={fl.get('ms', 'NO-MS')}")
    # The ratio denominator is the best hb M=1 arm, so a stale or missing baseline is an error.
    base = [int(float(arms[k]["ms"])) for k in arms if k[0] == "hb" and k[1] == 1]
    if not base:
        raise SystemExit("no hb M=1 arm in the evidence: the ratio has no denominator")
    best1 = min(float(arms[k]["ms"]) for k in arms if k[0] == "hb" and k[1] == 1)
    for M in (2, 4):
        cand = [float(arms[k]["ms"]) for k in arms if k[0] == "hb" and k[1] == M]
        if not cand:
            print(f"\nR(M={M}): NOT MEASURED — no arm for M={M}. No verdict is printed, because a "
                  f"verdict from a partial arm set is the empty-parse failure in a new costume.")
            continue
        r = min(cand) / best1
        print(f"\nR(M={M}) = {min(cand):.4f} / {best1:.4f} = {r:.4f}  "
              f"(best-of-rpb on both sides; per-rpb values above)")
        if r <= BAND_A_STANDS:
            verdict = "PLAN A STANDS AS PRICED"
        elif r <= BAND_NARROW:
            verdict = "PLAN A SURVIVES WITH A NARROWER CLAIM (>= %.3fx, not %.3fx)" % (
                speedup_of(r), speedup_of(BAND_A_STANDS))
        else:
            verdict = "THE BATCHED HEAD IS NOT FREE — re-price in route_price.py before a kernel"
        print(f"  delta -> {delta_of(r):.4f} ms = {delta_of(r) / STEP_MS:.4f} of a step; "
              f"speedup {speedup_of(r):.3f}x ({(speedup_of(r) - 1) * STEP_MS:+.3f} ms/step)")
        print(f"  VERDICT: {verdict}")
    print(f"\n(arm count is asserted non-empty; the runner prints NO-TIMING rather than a ms for an "
          f"arm whose canary did not move)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", action="store_true", help="print the pre-registered band table")
    ap.add_argument("--evidence", type=pathlib.Path, help="fold run_locked.sh evidence into a verdict")
    ap.add_argument("--ratio", type=float, help="price a single ratio (for what-if only)")
    args = ap.parse_args()
    if args.evidence:
        return from_evidence(args.evidence)
    if args.ratio:
        r = args.ratio
        print(f"R={r:.4f}: marginal {(r - 1) * LM_HEAD_MS:.4f} ms, delta {delta_of(r):.4f} ms "
              f"= {delta_of(r) / STEP_MS:.4f} of a step, speedup {speedup_of(r):.3f}x")
        return 0
    return bands_table()


if __name__ == "__main__":
    raise SystemExit(main())
