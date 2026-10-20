#!/usr/bin/env python3
"""A13 byte-identity witness: `--attn-fattn-wide 2` must equal `--attn-fattn-wide 1`, element for element.

WHY A BYTE COMPARE AND NOT A BAND. A13 (`engine/attn_wide_llama_kernels.h`, `prototype/attn_wide/LLAMA-PORT-PLAN.md`
items 1, 2, 4 and 5) ports four structural features onto the A12 wide block, and every one of them is
value-preserving: a swizzle permutes bytes INSIDE a staged row, a prefetch moves WHEN a load is issued and not
what it loads, a specialised predicate that provably cannot fire computes the same value, and the softmax scale
is `SCALE = 0.0625f = 2^-4`, which commutes with rounding and with the F32 WMMA sum. So the two forms owe each
other identical bits, and a cos/meanrel band would be too weak to say so: a swizzle that aliased a tile would
still land inside §10.1's bands at these prompt lengths and would be tabled as a win.

WHY NOT `engine/gate.py`. Its `--attn-fattn-wide` is `choices=("0","1")`, and admitting a third form is a
gate-policy decision rather than this script's to make. This is the op-level witness that answers "is the port
correct?" while that decision is pending, in the same spirit as `prototype/attn_m2/gate_m16.py` beside `gate.py`.

WHAT IT RUNS. One binary, one window, one variable: the same pinned prompt fed twice with `--steps 0
--prefill-logits all`, so every prefill row's logits are dumped and every attention row of every layer is on
show, and then every output file is hashed and compared. Each pair also greps the two echo fields that tell a
stale binary from a fresh one (`NUMERICS attn_fattn_wide=` and `ATTN_WIDE form=`), because a check whose input
can silently fail to arrive must be shown to have arrived (AGENTS.md, 2026-10-17).

CORRECTNESS ONLY. No timing, no burst, no best-of-N, no ms figure. The GPU lock is held per pair, and the wall
figures printed are there to show the run stayed inside a worker's budget, not to be quoted.

    python3 prototype/attn_wide/a13_byte_identity.py                 # the default fixture list
    python3 prototype/attn_wide/a13_byte_identity.py p128 wide520f
    KEEP_OUTPUT=1 python3 prototype/attn_wide/a13_byte_identity.py  # leave tmp dumps in place
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "engine" / "build" / "decode_run"
REFS = ROOT / "data" / "refs"
WORK = ROOT / "tmp" / "a13" / "identity"
LOCK = ROOT / "tools" / "gpu_lock.sh"

# One row per reason the port could break, so a green run means something specific:
#   p8         8 rows: every bound inside the first round, so the TAIL body does all the work and the full
#              body runs zero or one times — the geometry the item-4 split is about.
#   p128       four 32-position row tiles, 2 rounds at the widest row, full and tail mixed.
#   dec300f    295 rows, not a multiple of 32, so the last tile is short.
#   wide520f   500 rows: 16 rounds, 16 tiles, above MAXSC so the wide block owns lanes the one-pass route
#              would have split into partials.
#   wide1024f  the widest pinned single-body prompt: two 512-row bodies (`kMaxWaveRows`), and the row-slot
#              past `nrows` clamp at the top tile of each.
DEFAULT_FIXTURES = ("p8", "p128", "dec300f", "wide520f", "wide1024f")


def prompt_ids(fixture: str) -> list[int]:
    """The fixture's pinned prompt, read from its manifest — never a prompt invented here."""
    with open(REFS / fixture / "manifest.json") as fh:
        return [int(t) for t in json.load(fh)["prefill_ids"]]


def write_ids(fixture: str) -> Path:
    ids = prompt_ids(fixture)
    path = WORK / f"ids_{fixture}.txt"
    path.write_text(",".join(str(t) for t in ids))
    return path


def run_one(knob: int, ids: Path, out: Path) -> dict[str, str]:
    """One `decode_run` prefill, and the two echo fields that name the form that actually ran."""
    if out.exists():
        subprocess.run(["rm", "-rf", str(out)], check=True)
    cmd = [str(RUN), "--ids-file", str(ids), "--steps", "0", "--prefill-logits", "all",
           "--attn-fattn", "1", "--attn-fattn-wide", str(knob), "--out", str(out)]
    log = subprocess.run(cmd, capture_output=True, text=True)
    if log.returncode != 0:
        raise SystemExit(f"knob {knob} exited {log.returncode}:\n{(log.stdout + log.stderr)[-1200:]}")
    fields = {"numerics": "", "form": ""}
    for line in log.stdout.splitlines():
        if line.startswith("NUMERICS "):
            fields["numerics"] = dict(
                kv.split("=", 1) for kv in line.split()[1:] if "=" in kv).get("attn_fattn_wide", "")
        if line.startswith("ATTN_WIDE "):
            fields["form"] = line.split()[1].split("=", 1)[1]
    # Assert the canary ARRIVED, on the side that must show it: a form field that read the default would make
    # this whole comparison price knob 1 twice and call the second price knob 2.
    want = "wide_llama" if knob == 2 else "wide"
    if fields["numerics"] != want or not fields["form"]:
        raise SystemExit(f"knob {knob}: expected attn_fattn_wide={want}, got {fields!r}")
    return fields


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(a: Path, b: Path) -> tuple[int, list[str]]:
    """Compare by name RELATIVE TO EACH DIR — comparing the absolute paths instead makes every pair differ by
    its own directory, which reads as `files=0` and a wall of DIFFERS. That is exactly what the first version
    of this function did, and it failed loudly rather than passing, which is the only good kind of wrong."""
    na = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    nb = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    bad = sorted(str(r) for r in (na ^ nb))
    same = 0
    for r in sorted(na & nb):
        if digest(a / r) == digest(b / r):
            same += 1
        else:
            bad.append(str(r))
    return same, bad


def main() -> int:
    fixtures = sys.argv[1:] or list(DEFAULT_FIXTURES)
    WORK.mkdir(parents=True, exist_ok=True)
    util = subprocess.run(["bash", str(LOCK), "util"], capture_output=True, text=True).stdout.strip()
    print(f"# {RUN}  fixtures={' '.join(fixtures)}  {util}")
    failures = []
    for fx in fixtures:
        ids = write_ids(fx)
        subprocess.run(["bash", str(LOCK), "acquire", f"a13_identity_{fx}"], check=True)
        try:
            t0 = time.time()
            f1 = run_one(1, ids, WORK / "knob1")
            f2 = run_one(2, ids, WORK / "knob2")
            wall = time.time() - t0
        finally:
            subprocess.run(["bash", str(LOCK), "release", f"a13_identity_{fx}"], check=True)
        same, bad = compare(WORK / "knob1", WORK / "knob2")
        verdict = "BYTE-IDENTICAL" if not bad else f"NOT-IDENTICAL ({len(bad)} differ: {bad[:4]})"
        print(f"{fx:12s} rows={len(ids.read_text().split(',')):5d} files={same:5d} "
              f"knob1={f1['form']}/{f1['numerics']} knob2={f2['form']}/{f2['numerics']} "
              f"wall_s={wall:4.1f}  {verdict}")
        if bad:
            failures.append(fx)
        if not os.environ.get("KEEP_OUTPUT"):
            subprocess.run(["rm", "-rf", str(WORK / "knob1"), str(WORK / "knob2")], check=True)
    print("A13 BYTE-IDENTICAL: PASS" if not failures else f"A13 BYTE-IDENTICAL: FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
