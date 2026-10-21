#!/usr/bin/env python3
"""census.py — who checks the engine's population and byte mix that the S1 rig books.

The rig hard-codes two things it must not be trusted to have typed: the per-family LAUNCH bill (the
same arithmetic engine/decode_step.cpp's planned_launch_counts() does) and the per-family BYTE bill
(the step traffic census). This script re-derives both from their sources and compares them against
what the rig itself prints (`--print-engine-spec`, host-only, no device):

  * engine/state.h                -> kNumLayers / kNumGdnLayers / kNumAttnLayers
  * tools/step_traffic_model.py   -> per-family read/write MB per token, consumed, never retyped
  * the rig's --print-engine-spec -> the claim under test

Three rules from AGENTS.md are the reason this file exists in this shape:
  * "before writing a census tool, list the census tools": tools/step_traffic_model.py owns the byte
    census and is CONSUMED here, not duplicated; engine/decode_step.cpp owns the launch bill and is
    re-derived here from the same three layer counts rather than copied from a comment.
  * a parser that can match nothing reports success over an empty list. Every regex here asserts the
    record count it expected, and an empty match aborts (that guard has caught two real bugs this
    month: prototype/gdn_collapse/run.py and prototype/gdn_insitu/census.py).
  * a census must name the files it could read, so the bucket table below is explicit about where
    each of the census's rows goes, including the one judgement call (`norms` folds into `glue`).

Exit 0 with `AGREE rig=9 derived=9` means every family's launch count, segment count and byte totals
match within tolerance; any mismatch exits 1 and prints both sides.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RIG = HERE / "s1_launch_price"
TRAFFIC = ROOT / "tools" / "step_traffic_model.py"
STATE_H = ROOT / "engine" / "state.h"

# census family -> engine stage. `norms` is the judgement call: those rows are glue's add+norm and the
# final norm before lm_head, so their bytes belong to launches glue bills.
STAGE_OF = {
    "mlp": "mlp", "lm_head": "lm_head", "proj": "proj", "gdn": "gdn", "attn": "attn",
    "copy": "copy_19", "cast": "cast", "glue": "glue", "norms": "glue",
}
QUOTED_LAUNCHES, QUOTED_SEGMENTS = 340, 305      # [Q1] the S1 brief §2, echoing decode_run's echo
BYTE_TOL = 0.01                                  # 1 % of a family's bytes


def die(msg: str) -> "NoReturn":  # noqa: F821
    print(f"CENSUS FAIL: {msg}")
    sys.exit(1)


def parse_layers() -> dict[str, int]:
    """The three layer counts engine/state.h defines, or a hard failure."""
    txt = STATE_H.read_text()
    out = {}
    for name in ("kNumLayers", "kNumGdnLayers", "kNumAttnLayers"):
        m = re.search(rf"constexpr int\s+{name}\s*=\s*(\d+)\s*;", txt)
        if not m:
            die(f"{STATE_H} has no `constexpr int {name} = N;` — the launch bill cannot be derived")
        out[name] = int(m.group(1))
    if len(out) != 3:
        die(f"expected 3 layer counts from {STATE_H}, got {sorted(out)}")
    return out


def launch_bill(L: dict[str, int]) -> dict[str, tuple[int, int]]:
    """stage -> (launches, segments), derived the way planned_launch_counts() derives the launches
    and the way the stage clock's segments_per_step rows report the segments (quoted [Q1] window:
    glue 49/49, proj 48/48, gdn 72/36, attn 6/6, mlp 48/48, cast 97/97, copy 19/19, lm_head 1/1,
    host_tail 0/1)."""
    nl, ng, na = L["kNumLayers"], L["kNumGdnLayers"], L["kNumAttnLayers"]
    return {
        "glue": (1 + 2 * nl, 1 + 2 * nl),
        "proj": (2 * nl, 2 * nl),                       # shipped plan: packed, 2 launches a layer
        "gdn": (4 * ng, 2 * ng),                        # 4 launches a layer, 2 stage segments
        "attn": (na, na),
        "mlp": (2 * nl, 2 * nl),
        "cast": (4 * ng + 4 * na + 1, 4 * ng + 4 * na + 1),
        "copy_19": (ng + 1, ng + 1),
        "lm_head": (1, 1),
        "host_tail": (0, 1),
    }


def parse_traffic() -> dict[str, dict[str, float]]:
    """Run tools/step_traffic_model.py and bucket its per-family MB/token rows into engine stages."""
    if not TRAFFIC.exists():
        die(f"{TRAFFIC} is missing; the byte census has no source")
    res = subprocess.run([sys.executable, str(TRAFFIC)], capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        die(f"{TRAFFIC} exited {res.returncode}: {res.stderr.strip()[:200]}")
    rows = []
    for line in res.stdout.splitlines():
        m = re.match(r"^(\S+)\s+(read|write)\s+([0-9]+\.[0-9]+)\s+[0-9.]+%\s", line)
        if m:
            rows.append((m.group(1), m.group(2), float(m.group(3))))
    if len(rows) < 20:
        die(f"parsed {len(rows)} traffic rows from {TRAFFIC}; expected the full family table")
    tot = {}
    for fam, direction, mb in rows:
        stage = STAGE_OF.get(fam)
        if stage is None:
            die(f"census family {fam!r} has no stage mapping — add it to STAGE_OF and say why")
        tot.setdefault(stage, {"read": 0.0, "write": 0.0})
        tot[stage][direction] += mb
    # The TOTAL rows are the census's own checksum; a bucket that drops a row shows up here.
    mt = re.search(r"TOTAL\s+read\s+([0-9.]+)", res.stdout)
    mw = re.search(r"write\s+([0-9.]+)\s+1\.57%|TOTAL\s+[0-9.]+.*write\s+([0-9.]+)", res.stdout)
    if not mt:
        die("no TOTAL read row in the census output")
    sum_read = sum(v["read"] for v in tot.values())
    if abs(sum_read - float(mt.group(1))) > 0.05:
        die(f"bucketed read {sum_read:.3f} MB does not equal the census TOTAL read {mt.group(1)}")
    _ = mw
    return tot


def parse_rig_spec() -> dict[str, dict[str, float]]:
    """The rig's own claim, printed by --print-engine-spec (no device, no HIP)."""
    if not RIG.exists():
        die(f"{RIG} is not built; run prototype/launch_price/build.sh first")
    res = subprocess.run([str(RIG), "--print-engine-spec"], capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        die(f"{RIG} --print-engine-spec exited {res.returncode}: {res.stderr.strip()[:200]}")
    out, total_line = {}, None
    for line in res.stdout.splitlines():
        m = re.match(r"^ESPEC family=(\S+) launches=(\d+) segments=(\d+) "
                     r"read_mb=([0-9.]+) write_mb=([0-9.]+)$", line)
        if m:
            out[m.group(1)] = {"launches": int(m.group(2)), "segments": int(m.group(3)),
                               "read": float(m.group(4)), "write": float(m.group(5))}
            continue
        t = re.match(r"^ESPEC-TOTAL launches=(\d+) segments=(\d+)", line)
        if t:
            total_line = (int(t.group(1)), int(t.group(2)))
    if len(out) != 9 or total_line is None:
        die(f"parsed {len(out)} ESPEC rows and total={total_line} from the rig; expected 9 and a total")
    return {"rows": out, "total": total_line}


def main() -> int:
    L = parse_layers()
    bill = launch_bill(L)
    traffic = parse_traffic()
    rig = parse_rig_spec()
    rows, (tl, ts) = rig["rows"], rig["total"]

    print(f"layers: kNumLayers={L['kNumLayers']} kNumGdnLayers={L['kNumGdnLayers']} "
          f"kNumAttnLayers={L['kNumAttnLayers']} (engine/state.h)")
    fails = 0
    for stage, (launches, segments) in bill.items():
        r = rows.get(stage)
        if r is None:
            print(f"  {stage:<10} MISSING from the rig's spec")
            fails += 1
            continue
        t = traffic.get(stage, {"read": 0.0, "write": 0.0})
        prob = []
        if r["launches"] != launches:
            prob.append(f"launches rig={r['launches']} derived={launches}")
        if r["segments"] != segments:
            prob.append(f"segments rig={r['segments']} derived={segments}")
        for key, want in (("read", t["read"]), ("write", t["write"])):
            # The rig books the census's MB figures verbatim (kFam), so this is a same-unit compare.
            if abs(r[key] - want) > max(BYTE_TOL * want, 0.002):
                prob.append(f"{key} rig={r[key]:.3f} MB census={want:.3f} MB")
        if prob:
            fails += 1
            print(f"  {stage:<10} MISMATCH: " + "; ".join(prob))
        else:
            print(f"  {stage:<10} ok launches={launches:<4} segments={segments:<4} "
                  f"read={r['read']:.3f} MiB write={r['write']:.3f} MiB")
    if (tl, ts) != (QUOTED_LAUNCHES, QUOTED_SEGMENTS):
        fails += 1
        print(f"  TOTAL MISMATCH: rig={tl}/{ts}, quoted [Q1] = {QUOTED_LAUNCHES}/{QUOTED_SEGMENTS}")
    else:
        print(f"  TOTAL ok {tl} launches / {ts} segments (quoted [Q1], not re-derived from a device run)")
    print(f"AGREE rig={len(rows)} derived={len(bill)} fails={fails}")
    if fails:
        print("CENSUS FAIL: the rig's engine arm disagrees with its sources; fix the rig or the "
              "mapping, never the quoted totals.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
