#!/usr/bin/env python3
"""Synthetic `--time-waves` logs, to prove `run_ab_g2.sh`'s summary parser parses.

Not a measurement and touches no device: it exists because `AGENTS.md` records a gate driver whose regex
matched zero lines and printed PASS over a matrix it had never read. The summary block of the A/B script
reads `WAVE gran=stage family=... ms=`, `TOTAL ... wall_ms=` and `RESULT ... ms_total=`; if any of those
field names were wrong the script would print `NA` columns and a verdict computed from nothing. So the
fake logs below carry the real field names and deliberately NOT the real shapes, and the expected output
is a specific table and two specific verdict lines. Delete-after-use helper, kept because re-running the
check is 0.2 s.

    .venv/bin/python prototype/gdn_walk/mk_parsecheck.py && \
      SUMMARIZE=1 OUT=tmp/gdn_walk/parsecheck bash prototype/gdn_walk/run_ab_g2.sh
"""
from pathlib import Path

FAMS = ("gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head "
        "host_tail").split()
BASE_MS = {"gdn_rec": 266.60, "gdn_conv": 40.0, "gdn_gates": 12.0, "gdn_gated_norm": 30.0, "glue": 8.0,
           "proj": 60.0, "mlp_gateup": 55.0, "mlp_down": 30.0, "attn": 90.0, "cast": 20.0, "copy": 5.0,
           "lm_head": 3.0, "host_tail": 2.0}
# wf4 moves gdn_rec past BOTH dead lines, wf2 lands between the live and dead lines: one side must read
# DEAD and the other UNRESOLVED, or the verdict logic is not reading its own numbers.
SCALE = {"base": 1.0, "block": 1.0, "wf4": 0.60, "wf2": 0.80}


def main() -> int:
    out = Path("tmp/gdn_walk/parsecheck")
    out.mkdir(parents=True, exist_ok=True)
    for rep in (1, 2, 3):
        for side, scale in SCALE.items():
            name = "base" if side == "base" else f"new_{side}"
            lines, total = [], 0.0
            for fam in FAMS:
                ms = BASE_MS[fam] * (scale if fam == "gdn_rec" else 1.0)
                total += ms
                lines.append(f"WAVE gran=stage family={fam} reps=1 launches=144 plan_launches=144 "
                             f"counts_match=1 ms={ms:.2f} avg_ms={ms:.2f} per_unit_ms=0.01 share_pct=1.0")
            lines.append(f"WAVE gran=stage TOTAL waves=8 units=4096 wall_ms={total:.2f}")
            lines.append(f"RESULT decode_steps=0 prompt_tokens=4096 prefill_tokens=4096 "
                         f"ms_total={total:.2f} tok_per_s=1000")
            (out / f"{name}_{rep}.log").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(FAMS) * 4 * 3 + 8} synthetic log lines to {out}/ "
          "(wf4 gdn_rec 160 ms = past both live lines; wf2 213 ms = between the lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
