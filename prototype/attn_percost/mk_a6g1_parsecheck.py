#!/usr/bin/env python3
"""Synthetic A/B logs for `run_ab_a6g1.sh`'s verdict parser — a colour test, not a measurement. Touches no device.

WHY. G1's verdict is computed by an awk block over parsed rows, and this tree's record is unambiguous about that
shape: a driver whose regex matched nothing printed PASS over a matrix it never read, and a check whose input list
can be empty cannot fail. So each scenario below lands at a known distance from A6-BET's pre-registered G1 lines,
and `run_ab_a6g1.sh` (PARSE_CHECK=1) re-runs itself over the logs and REFUSES unless each prints the verdict named
here:

    live              family, traced and all three ship gates at or under their live lines      -> LIVE
    unresolved        family between the lines, nothing else decisive                           -> UNRESOLVED
    shipgate_blocks   family clears, but wave ms@16,384 sits between ITS lines                   -> UNRESOLVED
    traced_disagrees  family and the ship gates clear, the traced row moves the wrong way        -> UNRESOLVED
    dead              family past its dead line                                                  -> DEAD
    null_unpaired     every line clears, but the inert `stage` side moved > NULL_TOL vs base     -> UNPAIRED (abort)

Two things this colour-tests that G0's could not. G1 has an inert side, so the NULL leg is testable at all — a
harness that printed a verdict while the NULL had moved would be re-running A4's "the window was lying" failure.
And `shipgate_blocks` / `traced_disagrees` exist because G1's verdict is a CONJUNCTION (family AND three ship
gates AND the traced row agreeing); a one-row rule cannot be colour-tested row by row, so each row gets a fixture
in which it alone is the reason.

THE FIXTURES ARE DERIVED, NOT TYPED. Every factor is arithmetic on the ratio constants read out of
`run_ab_a6g1.sh` itself, because `mk_a4l1_parsecheck.py`'s history is exactly what this refuses: a hand-typed
`unresolved` fixture put one row past its dead line, read DEAD, and the colour test caught the *fixture*, not the
parser. If someone moves a line in the harness, the fixture moves with it or this file refuses.

    SCEN=live OUT=tmp/attn_percost/pc_live python3 prototype/attn_percost/mk_a6g1_parsecheck.py
    SCEN=null_unpaired OUT=tmp/attn_percost/pc_null python3 prototype/attn_percost/mk_a6g1_parsecheck.py
"""
import os
import re
from pathlib import Path

FAMS = ("gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head "
        "host_tail").split()
# Fixture scaffolding, not a claim about any run: A6-BET cites A4-L1's in-window bases (attn family 136.2548 ms,
# wave 446.2689 ms) and A4-BET's traced base (155.71 ms/pass) and wide absolutes. The harness re-derives its own
# lines from whatever base it measures, so these only have to be plausible magnitudes with the right shape.
BASE_MS = {"gdn_rec": 90.0, "gdn_conv": 18.0, "gdn_gates": 6.0, "gdn_gated_norm": 14.0, "glue": 8.0,
           "proj": 60.0, "mlp_gateup": 55.0, "mlp_down": 30.0, "attn": 136.2548, "cast": 20.0, "copy": 5.0,
           "lm_head": 3.0, "host_tail": 2.93}
BASE_TRACED = 155.71
BASE_WALL = {"4096": 446.2689, "16384": 3319.25, "32768": 10448.13}
# One inert-side move, in percent, for the fixture whose whole point is that the NULL moved.
NULL_MOVE_PCT = 5.0


def _constants() -> dict:
    txt = Path("prototype/attn_percost/run_ab_a6g1.sh").read_text()
    want = {
        "ra_l": r"RAT_A_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}", "ra_d": r"RAT_A_DEAD=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "rat_l": r"RAT_AT_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}", "rat_d": r"RAT_AT_DEAD=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "rw_l": r"RAT_W_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}", "rw_d": r"RAT_W_DEAD=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "r16_l": r"RAT_16_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}", "r16_d": r"RAT_16_DEAD=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "r32_l": r"RAT_32_LIVE=\$\{[A-Z_0-9]+:-([0-9.]+)\}", "r32_d": r"RAT_32_DEAD=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
        "null_tol": r"NULL_TOL=\$\{[A-Z_0-9]+:-([0-9.]+)\}",
    }
    out = {}
    for key, pat in want.items():
        m = re.search(pat, txt)
        if not m:
            raise SystemExit(f"REFUSE: cannot read {key} out of run_ab_a6g1.sh. A fixture built on a constant "
                             "that does not exist is not a colour test.")
        out[key] = float(m.group(1))
    for live, dead in (("ra", ("ra_l", "ra_d")), ("rat", ("rat_l", "rat_d")), ("rw", ("rw_l", "rw_d")),
                       ("r16", ("r16_l", "r16_d")), ("r32", ("r32_l", "r32_d"))):
        if not out[dead[0]] < out[dead[1]]:
            raise SystemExit(f"REFUSE: {live}: the harness's live line {out[dead[0]]} is not below its dead line "
                             f"{out[dead[1]]}. There is no between to build a fixture in; fix the harness.")
    if NULL_MOVE_PCT <= out["null_tol"]:
        raise SystemExit(f"REFUSE: NULL_MOVE_PCT {NULL_MOVE_PCT} does not exceed the harness's NULL_TOL "
                         f"{out['null_tol']}, so the unpaired fixture would not be unpaired.")
    return out


def mid(c, k_live, k_dead):
    return 0.5 * (c[k_live] + c[k_dead])


def scenario_factors(c: dict, mode: str) -> dict:
    """Ratios (new / base) for the five timed rows, plus the inert side's own move."""
    ok = lambda k: c[k] - 0.005                       # just inside a live line
    if mode == "live":
        f = {"fam": ok("ra_l"), "trace": ok("rat_l"), "w4096": ok("rw_l"),
             "w16384": ok("r16_l"), "w32768": ok("r32_l")}
    elif mode == "unresolved":
        f = {"fam": mid(c, "ra_l", "ra_d"), "trace": mid(c, "rat_l", "rat_d"),
             "w4096": mid(c, "rw_l", "rw_d"), "w16384": mid(c, "r16_l", "r16_d"),
             "w32768": mid(c, "r32_l", "r32_d")}
    elif mode == "shipgate_blocks":
        # ONE row alone is the reason: the 16k ship gate sits between its own two lines, everything else clears.
        f = {"fam": c["ra_l"] - 0.01, "trace": c["rat_l"] - 0.01, "w4096": c["rw_l"] - 0.01,
             "w16384": mid(c, "r16_l", "r16_d"), "w32768": c["r32_l"] - 0.01}
    elif mode == "traced_disagrees":
        # The traced row is the only miss, and it moves the wrong way: not DEAD (it cannot decide that alone),
        # not LIVE (A6-BET requires direction agreement), so UNRESOLVED.
        f = {"fam": ok("ra_l"), "trace": 1.0 + 0.005, "w4096": ok("rw_l"),
             "w16384": ok("r16_l"), "w32768": ok("r32_l")}
    elif mode == "dead":
        f = {"fam": c["ra_d"] + 0.01, "trace": c["rat_d"] + 0.01, "w4096": c["rw_d"] + 0.01,
             "w16384": c["r16_d"] + 0.01, "w32768": c["r32_d"] + 0.01}
    elif mode == "null_unpaired":
        f = {"fam": ok("ra_l"), "trace": ok("rat_l"), "w4096": ok("rw_l"),
             "w16384": ok("r16_l"), "w32768": ok("r32_l")}
    else:
        raise SystemExit(f"REFUSE: unknown scenario {mode!r}. A fixture that silently defaults is a colour test"
                         " of one colour pretending to be six.")
    # The inert side is byte-identical work, so its factors are 1.0 — except in the fixture about it moving.
    null = (1.0 + NULL_MOVE_PCT / 100.0) if mode == "null_unpaired" else 1.0
    return {"f": f, "null": null}


def log_lines(fams_ms: dict, wall: float, width: int) -> list:
    total = sum(fams_ms.values())
    lines = []
    for fam, ms in fams_ms.items():
        lines.append(f"WAVE gran=stage family={fam} reps=1 launches=48 plan_launches=48 counts_match=1 "
                     f"ms={ms:.2f} avg_ms={ms:.2f} per_unit_ms=0.01 share_pct={100.0 * ms / total:.2f}")
    lines.append(f"WAVE gran=stage TOTAL waves={max(1, width // 512)} units={width} wall_ms={wall:.2f} plan_match=1")
    lines.append(f"RESULT decode_steps=0 prompt_tokens={width} prefill_tokens={width} ms_total={wall:.2f}")
    return lines


def main() -> int:
    scen = os.environ.get("SCEN", "").strip()
    c = _constants()
    mv = scenario_factors(c, scen)
    f, null = mv["f"], mv["null"]
    out = Path(os.environ.get("OUT", f"tmp/attn_percost/a6g1ab/pc_{scen}"))
    out.mkdir(parents=True, exist_ok=True)
    reps = int(os.environ.get("REPS", "1"))
    reps_wide = int(os.environ.get("REPS_WIDE", "1"))
    n = 0
    for rep in range(1, reps + 1):
        for side in ("base", "stage", "u8"):
            fams, wall, traced = dict(BASE_MS), BASE_WALL["4096"], BASE_TRACED
            if side == "u8":
                fams["attn"] = round(BASE_MS["attn"] * f["fam"], 2)
                wall = round(BASE_WALL["4096"] * f["w4096"], 2)
                traced = round(BASE_TRACED * f["trace"], 2)
            elif side == "stage":
                fams["attn"] = round(BASE_MS["attn"] * null, 2)
                wall = round(BASE_WALL["4096"] * null, 2)
            lines = log_lines(fams, wall, 4096)
            # The banner fields CANARY B reads. Kept in the fixture so a parser change that starts depending on
            # them is visible here rather than in a device window.
            if side != "base":
                q, lds, wf, vg = (("unstaged8", 20480, 9, 185) if side == "u8" else ("stage", 32768, 6, 178))
                lines.insert(0, f"ATTN_FATTN wave_rows=512 split=8 fattn_rows=512 fallback_rows=0 "
                                f"covered_rows_pct=100.00 covered_work_pct=100.00 vtiles=8 q={q} "
                                f"lds={lds} wf_per_cu={wf} vgpr={vg}")
            (out / f"{side}_{rep}.log").write_text("\n".join(lines) + "\n")
            if side != "stage":
                (out / f"trace_{side}_{rep}.txt").write_text(
                    f"{'phase':<12} {'ms/pass':>10} {'share':>7} {'launches/pass':>14}\n"
                    f"{'attention':<12} {traced:>10.2f} {'33.0':>7} {48:>14}\n")
            n += 2 if side != "base" else 1
    for wid in ("16384", "32768"):
        for rep in range(1, reps_wide + 1):
            for side in ("base", "stage", "u8"):
                fams, wall = dict(BASE_MS), BASE_WALL[wid]
                if side == "u8":
                    fams["attn"] = round(BASE_MS["attn"] * f["fam"], 2)
                    wall = round(BASE_WALL[wid] * f["w" + wid], 2)
                elif side == "stage":
                    fams["attn"] = round(BASE_MS["attn"] * null, 2)
                    wall = round(BASE_WALL[wid] * null, 2)
                (out / f"wide{wid}_{side}_{rep}.log").write_text("\n".join(log_lines(fams, wall, int(wid))) + "\n")
                n += 1
    print(f"wrote {n} synthetic files to {out}/ for scenario {scen}: u8/base factors {f}, inert-side factor "
          f"{null} (bases attn {BASE_MS['attn']}, traced {BASE_TRACED}, wall {BASE_WALL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
