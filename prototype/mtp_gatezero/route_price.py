#!/usr/bin/env python3
"""The k=1 speculative arithmetic, with the assumptions that could kill it made explicit.

P7 gate zero, part 6. Everything that decides P7 fits in one line:

    speedup = (1 + a_eff) / ((1 + delta) * (1 + c))

  `a_eff` accepted drafts per round: `a` for the full tied head, `a * survival(k)` for the restricted
      head of S1.2(a) - a proposal outside the candidate set is a LOST accept, never a wrong one,
      because the accept test is the target's own argmax (S09.4).
  `c`     one draft proposal as a fraction of one target step (bytes from `mtp_census.py` /
      `draft_golden.py`'s read ledger, ms quoted from `engine/generation-limits.md`).
  `delta` what verifying TWO rows per step costs over verifying one. UNMEASURED - the engine has an
      `n`-row wave body (`engine/decode_step.h:submit_wave_body`) whose batched families are recorded
      bit-identical to the per-token ones, and weight traffic does not grow with `n`, but nobody has
      clocked it. So this script does not assume `delta = 0`; it prints the largest `delta` each route
      survives.

Every threshold here is the standing one: a prize must clear 0.30 ms/step = 2.7 % of the quoted
10.9357 ms step at 4,032 lanes (`progress/2026-10-21/generation-idea-queue-2026-10-21.md`).

Three verify models are priced, because the engine does not have to pick the cheap one:
  batched   - one 2-row body, cost (1+delta)(1+c). What an engine should do.
  refwd     - rejection re-feeds the accepted prefix through one more target step (what
              `speccheck/mtp.py` does today, and a large part of why torch CPU lost 21 %):
              cost (1+c) + (1-a) per round, and `delta` is inside each of those steps.
  none      - no speculation at all, speedup 1 by definition, the arm everything is compared to.

    .venv/bin/python prototype/mtp_gatezero/route_price.py [--accept 0.7329]
"""
from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# QUOTED from engine/generation-limits.md S1 (paired window, 4,032 lanes). Nothing here measures.
STEP_MS = 10.9357
LM_HEAD_MS = 2.1685
PLATEAU_GIBS = 237.0
LAUNCH_US = 3.10
BLOCK_MIB = 39.0107
BAR = 1.027          # the standing 0.30 ms / 2.7 % bar, as a speedup


def ms_of_mib(mib: float, eff: float = 1.0) -> float:
    return mib * 2**20 / (PLATEAU_GIBS * 2**30) * 1e3 / eff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept", type=float, default=None,
                    help="accept rate; default = read prototype/mtp_gatezero/draft_survival.json")
    ap.add_argument("--survival", default=None,
                    help="comma list of k to price as restricted routes; default = the ones "
                         "draft_survival.py measured")
    ap.add_argument("--eff", type=float, default=0.725,
                    help="achieved fraction of the copy plateau for the block walk; the ledger's "
                         "GEMV families run mlp 70 %% and proj 75 %%, lm_head 104 %%")
    args = ap.parse_args()

    surv: dict[int, float] = {}
    a = args.accept
    if a is None or args.survival is None:
        try:
            got = json.loads((HERE / "draft_survival.json").read_text())
        except FileNotFoundError:
            raise SystemExit("run draft_survival.py first (or pass --accept and --survival)")
        a = got["accept"] if a is None else a
        surv = {r["k"]: r["survival"] for r in got["routes"]}
    if args.survival:
        ks = [int(x) for x in args.survival.split(",")]
        surv = {k: surv.get(k, float("nan")) for k in ks}

    launches_ms = LAUNCH_US * 18 / 1e3
    block_ms = ms_of_mib(BLOCK_MIB, args.eff)
    c_full = (LM_HEAD_MS + block_ms + launches_ms) / STEP_MS
    print(f"accept a = {a:.4f}   (bar: speedup >= {BAR} = +0.30 ms/step at 4,032 lanes, quoted)")
    print(f"draft block priced at {args.eff:.3f} of the plateau = {block_ms:.4f} ms, "
          f"+{launches_ms:.4f} ms of launches, +{LM_HEAD_MS:.4f} ms quoted if the head is full")

    def line(name, a_eff, c, delta, model, survival=1.0):
        if model == "batched":
            den = (1 + delta) * (1 + c)
            amin_eff = BAR * den - 1                      # (1+a_eff)/(den) = BAR
        else:
            den = (1 + c) + (1 - a_eff)
            # refwd: den itself contains a_eff, so solve (1+a)/((1+c)+(1-a)) = BAR for a
            amin_eff = (BAR * (2.0 + c) - 1.0) / (1.0 + BAR)
        sp = (1 + a_eff) / den
        prize = (sp - 1) * STEP_MS
        dmax = (1 + a_eff) / (BAR * (1 + c)) - 1 if model == "batched" else float("inf")
        amin_raw = amin_eff / survival if survival > 0 else float("inf")
        verdict = "WIN" if sp >= BAR else "under bar"
        print(f"{name:<24} {a_eff:>7.4f} {c:>7.4f} {sp:>7.3f}x {prize:>+8.3f} {dmax:>9.3f} "
              f"{amin_eff:>9.3f} {amin_raw:>9.3f}  {verdict}")
        return sp

    for model in ("batched", "refwd"):
        print(f"\n=== verify model: {model} ===")
        print(f"{'route':<24} {'a_eff':>7} {'c':>7} {'speedup':>8} {'ms/step':>8} "
              f"{'d_max':>9} {'a_eff_min':>9} {'a_raw_min':>9}")
        line("FULL tied head", a, c_full, 0.0, model)
        for k in sorted(surv):
            if surv[k] != surv[k]:      # nan
                continue
            c = (block_ms + launches_ms + ms_of_mib(k * 1024 * 2 / 2**20, 1.0)) / STEP_MS
            line(f"RESTRICTED top-{k}", a * surv[k], c, 0.0, model, surv[k])

    print(f"\n{'delta':>7}  what the verify width costs, batched model, FULL head")
    for delta in (0.0, 0.05, 0.10, 0.20, 0.30, 0.379):
        sp = (1 + a) / ((1 + delta) * (1 + c_full))
        print(f"{delta:>7.3f}  speedup {sp:.3f}x  {(sp - 1) * STEP_MS:+.3f} ms/step  "
              f"{'WIN' if sp >= BAR else 'KILLED'}")
    print("\na_eff_min is the accept the ROUTE needs; a_raw_min divides it by survival(k), i.e. the\n"
          "accept the FULL drafter has to achieve for the restricted route to be worth that k.")
    print("\nThe decisive reads: (1) the FULL head clears the bar at every delta up to the printed "
          "delta_max, so P7 is not killed by the verify question; (2) the restricted head is worth "
          "more per accepted token but needs a device top-k over the previous logits row, and its "
          "margin over the FULL head is small enough that it is an optimisation to defer, not a "
          "reason to wait.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
