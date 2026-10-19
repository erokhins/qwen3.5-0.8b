#!/usr/bin/env python3
"""Synthetic A/B logs for `run_ab_a6g0.sh`'s verdict parser — a colour test, not a measurement. Touches no device.

WHY. A6-BET's falsifier verdict is computed by an awk block over parsed rows, and AGENTS' record is full of the
same failure: a driver whose regex matched nothing printed PASS over a matrix it never read; a check whose input
list can be empty cannot fail. So each scenario below carries the real field names (`WAVE gran=stage family=…
ms=`, `WAVE gran=stage TOTAL … wall_ms=`, `trace_phase_split`'s `attention <ms/pass>` row, the wide logs) and
lands at a known distance from A6-BET's rule, and `run_ab_a6g0.sh` (PARSE_CHECK=1) re-runs itself over the logs
and REFUSES unless each prints the verdict named here:

    live           family improvement above FAMILY_LIVE_IMPROVE                 -> LIVE
    live_by_trace  family in between, traced improvement above TRACE_TIEBREAK   -> LIVE
    unresolved     family in between, traced improvement below the tie-break    -> UNRESOLVED
    dead           family improvement below FAMILY_DEAD_IMPROVE                 -> DEAD

THE FIXTURES ARE DERIVED, NOT TYPED. Every improvement factor is arithmetic on the constants read out of
`run_ab_a6g0.sh` itself — `FAMILY_LIVE_IMPROVE`, `FAMILY_DEAD_IMPROVE`, `TRACE_TIEBREAK` and the five row ratios —
because `mk_a4l1_parsecheck.py`'s history is exactly what this refuses: a hand-typed `unresolved` fixture put one
row past its dead line, read DEAD, and the colour test caught the *fixture*, not the parser. Deriving removes the
recurrence, and if someone moves a line in the harness the fixture moves with it or the generator refuses.

    SCEN=live OUT=tmp/attn_percost/pc_live python3 prototype/attn_percost/mk_a6g0_parsecheck.py
"""
import os
import re
from pathlib import Path

FAMS = ("gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head "
        "host_tail").split()
# Bases: A6-BET cites `prototype/attn_percost/A4-L1-verdict.md`'s in-window bases (attn family 136.2548 ms, wave
# 446.2689 ms) and its own traced row's base is A4-BET's 155.71 ms/pass (the file rewrites 0.8991 = 140/155.71).
# The wide bases are A4-BET's Gate-2 absolutes. These are FIXTURE scaffolding for a parser colour test, not a
# claim about any run; the harness re-derives its own lines from whatever base it measures.
BASE_MS = {"gdn_rec": 90.0, "gdn_conv": 18.0, "gdn_gates": 6.0, "gdn_gated_norm": 14.0, "glue": 8.0,
           "proj": 60.0, "mlp_gateup": 55.0, "mlp_down": 30.0, "attn": 136.2548, "cast": 20.0, "copy": 5.0,
           "lm_head": 3.0, "host_tail": 2.93}
BASE_TRACED = 155.71
BASE_WALL = {"4096": 446.2689, "16384": 3319.25, "32768": 10448.13}


def _constants() -> dict:
    txt = Path("prototype/attn_percost/run_ab_a6g0.sh").read_text()
    want = {
        "rat_a": r"RAT_A_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "rat_at": r"RAT_AT_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "rat_w": r"RAT_W_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "rat_16": r"RAT_16_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "rat_32": r"RAT_32_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "fam_live": r"FAMILY_LIVE_IMPROVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "fam_dead": r"FAMILY_DEAD_IMPROVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "trace_tb": r"TRACE_TIEBREAK=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
    }
    out = {}
    for key, pat in want.items():
        m = re.search(pat, txt)
        if not m:
            raise SystemExit(f"REFUSE: cannot read {key} out of run_ab_a6g0.sh; a fixture built on a constant "
                             "that does not exist is not a colour test.")
        out[key] = float(m.group(1))
    if not (out["fam_dead"] < out["trace_tb"] < out["fam_live"]):
        raise SystemExit(f"REFUSE: the harness's constants are out of order "
                         f"(dead {out['fam_dead']} < tiebreak {out['trace_tb']} < live {out['fam_live']} is "
                         "required to build the four scenarios). Fix the harness, not this file.")
    return out


def _fixture(mode: str) -> dict:
    c = _constants()
    mid = 0.5 * (c["fam_dead"] + c["fam_live"])
    if mode == "live":
        fam, trace = c["fam_live"] + 0.01, c["fam_live"] + 0.01
    elif mode == "dead":
        fam, trace = max(c["fam_dead"] - 0.02, 0.0), 0.0
    elif mode == "unresolved":
        fam, trace = mid, max(c["trace_tb"] - 0.03, 0.0)
    elif mode == "live_by_trace":
        fam, trace = mid, c["trace_tb"] + 0.02
    else:
        raise SystemExit(f"REFUSE: unknown scenario {mode!r}")
    # The ship-gate rows are parked just inside their live lines in every scenario: they do not decide G0 (A6-BET's
    # falsifier rule is the family row), and parking them at a fixed point keeps a colour test about ONE row.
    return {"fam": fam, "trace": trace,
            "wall": {"4096": 1.0 - c["rat_w"], "16384": 1.0 - c["rat_16"], "32768": 1.0 - c["rat_32"]}}


SCENARIOS = {s: _fixture(s) for s in ("live", "unresolved", "live_by_trace", "dead")}


def log_lines(fams_ms: dict, wall: float) -> list:
    total = sum(fams_ms.values())
    lines = []
    for fam, ms in fams_ms.items():
        lines.append(f"WAVE gran=stage family={fam} reps=1 launches=48 plan_launches=48 counts_match=1 "
                     f"ms={ms:.2f} avg_ms={ms:.2f} per_unit_ms=0.01 share_pct={100.0 * ms / total:.2f}")
    lines.append(f"WAVE gran=stage TOTAL waves=8 units=4096 wall_ms={wall:.2f} plan_match=1")
    lines.append(f"RESULT decode_steps=0 prompt_tokens=4096 prefill_tokens=4096 ms_total={wall:.2f} tok_per_s=1000")
    return lines


def main() -> int:
    scen = os.environ.get("SCEN", "").strip()
    if scen not in SCENARIOS:
        raise SystemExit(f"REFUSE: SCEN must be one of {sorted(SCENARIOS)} (got {scen!r}). A fixture that"
                         " silently defaults to one scenario would colour-test one colour and claim four.")
    mv = SCENARIOS[scen]
    out = Path(os.environ.get("OUT", f"tmp/attn_percost/a6g0ab/pc_{scen}"))
    out.mkdir(parents=True, exist_ok=True)
    reps = int(os.environ.get("REPS", "1"))
    reps_wide = int(os.environ.get("REPS_WIDE", "1"))
    n = 0
    for rep in range(1, reps + 1):
        for side in ("base", "g0"):
            fams = dict(BASE_MS)
            traced, wall = BASE_TRACED, BASE_WALL["4096"]
            if side == "g0":
                fams["attn"] = round(BASE_MS["attn"] * (1.0 - mv["fam"]), 2)
                traced = round(BASE_TRACED * (1.0 - mv["trace"]), 2)
                wall = round(BASE_WALL["4096"] * (1.0 - mv["wall"]["4096"]), 2)
            (out / f"{side}_{rep}.log").write_text("\n".join(log_lines(fams, wall)) + "\n")
            (out / f"trace_{side}_{rep}.txt").write_text(
                f"{'phase':<12} {'ms/pass':>10} {'share':>7} {'launches/pass':>14}\n"
                f"{'attention':<12} {traced:>10.2f} {'33.0':>7} {48:>14}\n")
            n += 2
    for wid in ("16384", "32768"):
        for rep in range(1, reps_wide + 1):
            for side in ("base", "g0"):
                fams = dict(BASE_MS)
                wall = BASE_WALL[wid]
                if side == "g0":
                    fams["attn"] = round(BASE_MS["attn"] * (1.0 - mv["fam"]), 2)
                    wall = round(BASE_WALL[wid] * (1.0 - mv["wall"][wid]), 2)
                (out / f"wide{wid}_{side}_{rep}.log").write_text("\n".join(log_lines(fams, wall)) + "\n")
                n += 1
    print(f"wrote {n} synthetic files to {out}/ for scenario {scen}: family improvement {mv['fam']:.3f}, "
          f"traced improvement {mv['trace']:.3f}, wall factors {mv['wall']} (bases attn {BASE_MS['attn']}, "
          f"traced {BASE_TRACED}, wall {BASE_WALL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
