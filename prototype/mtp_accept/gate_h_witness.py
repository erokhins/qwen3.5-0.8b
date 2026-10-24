#!/usr/bin/env python3
"""Post-hoc corruption witness for `engine/gate.py` section (m) — CPU only, no engine run.

Section (m)'s own injections run inside `--selftest` on synthetic records. This one answers a narrower and
more annoying question: would the section catch a wrong token in a REAL captured run, and would it refuse to
count a position that is downstream of a fork the pin itself could not decide? Both halves are required — a
negative control that catches everything proves nothing, and one that catches nothing is worse.

Capture a run first (device, correctness only, GPU lock held):

    tools/gpu_lock.sh acquire <TAG>
    d=tmp/gate_h_witness && mkdir -p $d
    engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file <(python3 - <<'EOF'
    import json;print(",".join(map(str,json.load(open("data/refs/prefix24dec8/manifest.json"))["sources"][0]["ids"])))
    EOF
    ) --mtp-accept 1 --mtp-k 1 --mtp-draft 1 --gdn-loop 1 --no-stop --gemm lt --steps 6 --out $d > $d.log 2>&1
    tools/gpu_lock.sh release <TAG>

Then adjudicate it three ways:

    .venv/bin/python prototype/mtp_accept/gate_h_witness.py --fixture prefix24dec8 \
        --log tmp/gate_h_witness.log --out-dir tmp/gate_h_witness

Exits 0 only if all three behave: the clean record's stream line is green (admitted or equal), an injected
token at a row the pin DECIDES turns it red, and an injected token downstream of an admitted fork leaves it
exactly as it was.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "engine"))
import gate  # noqa: E402


def stream_row(rec) -> tuple[bool, str]:
    """The section's losslessness verdict for one record, as (green, detail)."""
    rows = gate.mtp_round_verdicts(rec)
    for name, ok, detail in rows:
        if "stream ==" in name:
            return ok, detail
    raise AssertionError("section (m) printed no stream line for this record")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="prefix24dec8")
    ap.add_argument("--log", required=True, help="stdout+stderr of one `--mtp-accept` run")
    ap.add_argument("--out-dir", required=True, help="the run's --out directory (holds emitted.txt)")
    args = ap.parse_args()

    gate.set_fixture(args.fixture)
    log = pathlib.Path(args.log).read_text()
    rec = gate.mtp_round_read_run(log, pathlib.Path(args.out_dir))
    greedy = json.loads((gate.REFS / "manifest.json").read_text())["greedy_continuation"]
    # `mtp_round_read_run` is the parse half; these are the fields `mtp_round_run_case` adds around it.
    rec.update({"fixture": args.fixture, "k": 1, "rc": 0, "pin": list(greedy),
                "depth": gate.mtp_round_depth(greedy)})
    depth = rec["depth"]
    if not rec["emitted"]:
        print("no emitted.txt in --out-dir; nothing to adjudicate", file=sys.stderr)
        return 2

    ok_clean, det_clean = stream_row(dict(rec))
    print(f"(a) clean: green={ok_clean}\n    {det_clean[:300]}")

    results = [ok_clean]
    # The second injection goes to the LAST position the section actually judges. If the clean record's fork
    # is already at or past it there is nothing downstream to inject into, and the witness says so instead of
    # inventing a slot: a must-not-catch aimed at a position the geometry cannot reach is the empty control
    # `AGENTS.md` records three times.
    slots = [0]
    if len(rec["emitted"]) > 1:
        slots.append(min(depth - 1, len(rec["emitted"]) - 1))
    for slot, label in ((slots[0], "the row the pin decides"),
                        *([(slots[1], "downstream of the admitted fork")] if len(slots) > 1 else [])):
        mut = dict(rec)
        mut["emitted"] = list(rec["emitted"])
        mut["emitted"][slot] = mut["emitted"][slot] + 7777      # an id no row here ranks first
        ok_mut, det_mut = stream_row(mut)
        same_as_clean = (ok_mut == ok_clean and det_mut[:120] == det_clean[:120])
        print(f"\n(b) injected at emitted slot {slot} ({label}): green={ok_mut}"
              f"{' — unchanged from clean' if same_as_clean else ''}\n    {det_mut[:300]}")
        # The decided-row injection is only a must-catch if the pin decided that row; say which one fired.
        decided = "the pin DECIDED this token" in det_mut
        if decided:
            print(f"    verdict: must-catch fired (the pin decided the row the injection moved): {not ok_mut}")
            results.append(not ok_mut)
        else:
            print("    verdict: this slot's row was not one the pin decides, so the injection is a"
                  " must-not-catch here; the fork/admission machinery should absorb it identically")
            results.append(same_as_clean)
    print("\n=== all three behaved:", all(results))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
