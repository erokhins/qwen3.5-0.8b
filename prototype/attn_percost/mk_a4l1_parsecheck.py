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
# Bases are A4-BET.md's Gate-2 bases (the §45 figures). The worker's fixture carried the fabricated "re-based"
# absolutes (attn 91.27 of a 1,196.89 ms wave); those were 2.6x off the fixture this harness actually runs,
# which is how a colour test kept passing over lines that described nothing.
BASE_MS = {"gdn_rec": 90.0, "gdn_conv": 18.0, "gdn_gates": 6.0, "gdn_gated_norm": 14.0, "glue": 8.0,
           "proj": 60.0, "mlp_gateup": 55.0, "mlp_down": 30.0, "attn": 133.00, "cast": 20.0, "copy": 5.0,
           "lm_head": 3.0, "host_tail": 2.93}
BASE_TRACED_ATTN = 155.71
BASE_WALL = {"4096": 453.65, "16384": 3319.25, "32768": 10448.13}

# The fixtures are DERIVED from the harness's own ratio constants, not hand-typed: a fixture whose numbers are
# typed to match a line set rots the moment the lines move, and that is the exact defect this file is here to
# catch. Ratios are read out of run_ab_a4l1.sh, so there is one source of truth.
def _ratios() -> dict:
    import re, pathlib
    txt = pathlib.Path("prototype/attn_percost/run_ab_a4l1.sh").read_text()
    want = {"a": "RAT_A", "at": "RAT_AT", "w": "RAT_W", "w16": "RAT_16", "w32": "RAT_32"}
    out = {}
    for key, name in want.items():
        for side in ("LIVE", "DEAD"):
            m = re.search(name + "_" + side + r"=\$\{[A-Z_0-9]+:-([0-9.]+)\}", txt)
            if not m:
                raise SystemExit(f"REFUSE: cannot read {name}_{side} out of the harness; a fixture built on a"
                                 " default that does not exist is not a colour test.")
            out[(key, side)] = float(m.group(1))
    return out


def _fixture(mode: str) -> dict:
    r = _ratios()
    base = {"a": BASE_MS["attn"], "at": BASE_TRACED_ATTN, "w": BASE_WALL["4096"],
            "w16": BASE_WALL["16384"], "w32": BASE_WALL["32768"]}
    if mode == "live":            # every row just inside its live line
        fac = {k: r[(k, "LIVE")] * 0.99 for k in base}
    elif mode == "unresolved":    # every row strictly between live and dead (midpoint of the two ratios)
        fac = {k: 0.5 * (r[(k, "LIVE")] + r[(k, "DEAD")]) for k in base}
    elif mode == "dead":          # exactly one row past its dead line, the rest live
        fac = {k: r[(k, "LIVE")] * 0.99 for k in base}
        fac["a"] = r[("a", "DEAD")] * 1.01
    else:
        raise SystemExit(f"REFUSE: unknown scenario {mode!r}")
    return {"attn": round(base["a"] * fac["a"], 2), "traced": round(base["at"] * fac["at"], 2),
            "wall": {"4096": round(base["w"] * fac["w"], 2), "16384": round(base["w16"] * fac["w16"], 2),
                     "32768": round(base["w32"] * fac["w32"], 2)}}


# The `unresolved` fixture is derived too; the hand-typed version put 32,768 PAST its dead line and read DEAD,
# which the colour test caught - the fixture was wrong, not the parser, and deriving it removes the recurrence.
SCENARIOS = {s: _fixture(s) for s in ("live", "unresolved", "dead")}


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
