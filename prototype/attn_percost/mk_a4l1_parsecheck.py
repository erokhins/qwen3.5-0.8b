#!/usr/bin/env python3
"""Synthetic A/B logs for `run_ab_a4l1.sh`'s verdict parser — a colour test, not a measurement.

Touches no device. It exists because AGENTS.md records a gate driver whose regex matched zero lines and
printed PASS over a matrix it had never read: the A/B summary reads `WAVE gran=stage family=... ms=`,
`WAVE gran=stage TOTAL ... wall_ms=`, `trace_phase_split`'s `attention <ms/pass>` row and the wide logs, and
if ANY of those field names were wrong the harness would print `NA` columns and still print a verdict. So
each scenario below carries the real field names and lands at a known distance from the pre-registered lines,
and `run_ab_a4l1.sh` (PARSE_CHECK=1) re-runs itself over the logs and REFUSES unless each scenario prints the
verdict named here:

    live        every one of the five rows inside its live line            -> LIVE
    unresolved  `attn` between the live and dead lines, nothing dead       -> UNRESOLVED
    dead        `attn` past its dead line                                  -> DEAD

The `single` side is written identical to `base`, because that pairing is the NULL the harness also checks.

    SCEN=live OUT=tmp/attn_percost/pc python3 prototype/attn_percost/mk_a4l1_parsecheck.py
"""
import os
from pathlib import Path

FAMS = ("gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head "
        "host_tail").split()
# Sizes near the re-based Gate-2 window: `attn` 91.27 of a 1,196.89 ms wave, the traced attention phase
# 73.34 ms/pass. Only `attn` and the totals move per scenario; the other families stay equal across sides,
# so the externality sum is 0.0 in every scenario and the colour can only come from the five rows.
BASE_MS = {"gdn_rec": 260.0, "gdn_conv": 40.0, "gdn_gates": 12.0, "gdn_gated_norm": 30.0, "glue": 8.0,
           "proj": 60.0, "mlp_gateup": 55.0, "mlp_down": 30.0, "attn": 91.27, "cast": 20.0, "copy": 5.0,
           "lm_head": 3.0, "host_tail": 2.93}
BASE_TRACED_ATTN = 73.34
BASE_WALL = {"4096": 1196.89, "16384": 3319.25, "32768": 10448.13}
# What each scenario moves on the `two` side: (attn ms, traced attention ms, wall at each width).
SCENARIOS = {
    "live":       {"attn": 84.00, "traced": 66.00, "wall": {"4096": 1120.0, "16384": 2950.0, "32768": 9200.0}},
    # every row strictly between its live and dead lines. Its first version put 32,768 at 10,300 ms, which is
    # PAST the 10,135 dead line, and the colour test correctly read DEAD — the fixture was wrong, not the
    # parser, which is the whole point of colour-testing a fixture as well as a parser.
    "unresolved": {"attn": 88.00, "traced": 72.00, "wall": {"4096": 1180.0, "16384": 3200.0, "32768": 10100.0}},
    "dead":       {"attn": 96.00, "traced": 82.00, "wall": {"4096": 1260.0, "16384": 3400.0, "32768": 11000.0}},
}


def log_lines(fams_ms: dict, wall: float) -> list:
    total = sum(fams_ms.values())
    lines = []
    for fam, ms in fams_ms.items():
        lines.append(f"WAVE gran=stage family={fam} reps=1 launches=48 plan_launches=48 counts_match=1 "
                     f"ms={ms:.2f} avg_ms={ms:.2f} per_unit_ms=0.01 share_pct={100.0 * ms / total:.2f}")
    # The synthetic total is the scenario's own wall, not the family sum: the harness reads the two from
    # different fields and the point is that BOTH are parsed, not that they reconcile.
    lines.append(f"WAVE gran=stage TOTAL waves=8 units=4096 wall_ms={wall:.2f} plan_match=1")
    lines.append(f"RESULT decode_steps=0 prompt_tokens=4096 prefill_tokens=4096 ms_total={wall:.2f} tok_per_s=1000")
    return lines


def main() -> int:
    scen = os.environ.get("SCEN", "").strip()
    if scen not in SCENARIOS:
        raise SystemExit(f"REFUSE: SCEN must be one of {sorted(SCENARIOS)} (got {scen!r}). A fixture that"
                         " silently defaults to one scenario would colour-test one colour and claim three.")
    mv = SCENARIOS[scen]
    out = Path(os.environ.get("OUT", "tmp/attn_percost/a4l1ab/pc_" + scen))
    out.mkdir(parents=True, exist_ok=True)
    reps = int(os.environ.get("REPS", "1"))
    reps_wide = int(os.environ.get("REPS_WIDE", "1"))
    n = 0
    for rep in range(1, reps + 1):
        for side in ("base", "single", "two"):
            fams = dict(BASE_MS)
            traced, wall = BASE_TRACED_ATTN, BASE_WALL["4096"]
            if side == "two":
                fams["attn"] = mv["attn"]
                traced = mv["traced"]
                wall = mv["wall"]["4096"]
            tag = "base" if side == "base" else f"new_{side}"
            (out / f"{tag}_{rep}.log").write_text("\n".join(log_lines(fams, wall)) + "\n")
            # trace_phase_split.py's table row: `phase ms/pass share launches/pass`. Header included so the
            # parser is exercised against the real shape, not against a bare number.
            (out / f"trace_{tag}_{rep}.txt").write_text(
                f"{'phase':<12} {'ms/pass':>10} {'share':>7} {'launches/pass':>14}\n"
                f"{'attention':<12} {traced:>10.2f} {'33.0':>7} {48:>14}\n")
            n += 2
    for wid in ("16384", "32768"):
        for rep in range(1, reps_wide + 1):
            for side in ("base", "single", "two"):
                fams = dict(BASE_MS)
                wall = BASE_WALL[wid]
                if side == "two":
                    fams["attn"] = mv["attn"]
                    wall = mv["wall"][wid]
                tag = "base" if side == "base" else f"new_{side}"
                (out / f"wide{wid}_{tag}_{rep}.log").write_text("\n".join(log_lines(fams, wall)) + "\n")
                n += 1
    print(f"wrote {n} synthetic files to {out}/ for scenario {scen}: attn={mv['attn']} traced={mv['traced']} "
          f"wall={mv['wall']} (base attn={BASE_MS['attn']} traced={BASE_TRACED_ATTN} wall={BASE_WALL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
