"""Run the D-1/D-2 verification set in one blocking pass and print what each command reported.

Correctness only. The wall figures below are the cost of running a check, printed so a reader can see
each fit inside the `<60 s` worker allowance in `AGENTS.md`; they are NOT engine timings and nothing here
goes near `bench/performance-status.md`. `decode_run`'s own `ms_per_step` line is deliberately not parsed.

    .venv/bin/python tmp/d6_final_pass.py            # device gates + CPU selftests, ~30 s total
    .venv/bin/python tmp/d6_final_pass.py --cpu      # selftests only, no device
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOCKS = ROOT / "tools" / "gpu_lock.sh"

FIXTURES = ["dec8", "dec64f", "dec64t", "dec300f", "mt32f"]
cpu_only = "--cpu" in sys.argv


def sh(argv):
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    return time.time() - t0, p.returncode, (p.stdout or "") + (p.stderr or "")


def summarize(out):
    lines = out.splitlines()
    verdict = next((l for l in lines if l.startswith(("ENGINE GATE", "SELFTEST"))), "NO VERDICT LINE")
    npass = sum(1 for l in lines if "[PASS]" in l)
    nfail = sum(1 for l in lines if "[FAIL]" in l)
    margin = next((l.strip() for l in lines if "allowance: route_allowance.json" in l), "")
    m = re.search(r"margins cos ([\d.]+) / meanrel ([\d.]+)", margin)
    worst = next((l for l in lines if "Worst step" in l), "")
    wm = re.search(r"Worst step (\d+): deficit = ([\d.]+)x", worst)
    ties = [l.strip() for l in lines if l.strip().startswith(("[TIE]", "[TIE-NOTE]"))]
    skips = [l.strip() for l in lines if "[SKIP]" in l]
    return (f"{verdict} | {npass} PASS / {nfail} FAIL | margins"
            f" {m.group(1)}/{m.group(2)}" if m else verdict) + (
        f" | worst {wm.group(1)}@{wm.group(2)}x" if wm else "") + (
        f" | {'; '.join(ties)}" if ties else "") + (
        f" | {'; '.join(s.replace('[SKIP] ', '') for s in skips)}" if skips else "")


if __name__ == "__main__":
    _, _, util = sh([str(LOCKS), "util"])
    print(f"gpu_lock: {util.strip()}  (holding tag 'engine gate D-1/D-2 last worker pass')"
          f"{' --cpu mode' if cpu_only else ''}")
    for fx in FIXTURES:
        wall, rc, out = sh([PY, "engine/gate.py", "--selftest", "--fixture", fx])
        print(f"selftest {fx:<7} {wall:5.1f} s rc={rc}  {summarize(out)}")
    if not cpu_only:
        for fx in FIXTURES:
            wall, rc, out = sh([PY, "engine/gate.py", "--fixture", fx])
            log = ROOT / "tmp" / f"d6_gate_{fx}.log"
            log.write_text(out)
            print(f"gate     {fx:<7} {wall:5.1f} s rc={rc}  {summarize(out)}  (log tmp/{log.name})")
    _, _, util = sh([str(LOCKS), "util"])
    print(f"gpu_lock at the end: {util.strip()}")
