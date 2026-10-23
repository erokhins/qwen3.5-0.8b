#!/usr/bin/env python3
"""draft_cond_probe.py — does a DRAFT-SIDE conditioning change move the accept rate?

WHY THIS EXISTS. The round's cost is done: at 4,032 lanes a k=1 round costs 1.617 steps and parity needs
an accept rate a* = round/step - 1 = 0.617, while the +10 % adopt line needs a >= 1.10 x 1.617 - 1 = 0.779.
The corpus loop measures 0.6556. llama.cpp measures 0.750 at pos1 on the SAME text we score 0.516 on
(`prototype/mtp_llama/LLAMA-HOW.md` §Q5), so the gap is the draft's guess, not the round's price. This
harness measures the guess, one knob at a time, on one fixture, and it takes NO TIMING: no millisecond
figure appears anywhere in its output, by construction rather than by discipline — nothing in this file
reads a clock.

THE INVARIANT IT ENFORCES. A legitimate variant may change only the PROPOSED token. So every arm is
adjudicated twice:

  * ADMISSION (imported from `m3_loop_rate.py`, same criterion, same band): the accept arm's emitted stream
    against the flag-off greedy stream. A divergence must be that row's RUNNER-UP at a margin below one
    bf16 ULP of the row's top logit — i.e. a legal gap between two bf16 forms — and anything else is a
    bookkeeping bug, not a better guess. That is what catches a variant that reached into the target.
  * CONTROL (this file): `--mtp-draft 1 --mtp-sched every` prices proposals without letting them drive the
    run, so the target's stream and its per-row logits are pinned by construction. Those dumps must come
    out BYTE-identical between the plain and the variant arm while the `DRAFT ... proposal=` lines
    DIFFER. Identical-but-no-difference is a no-op, not a null result, and the line says which.

THE NOISE FLOOR IS PRINTED, NOT ASSUMED. The last form change moved the 12-file corpus witness from 0.6951
to 0.7027, i.e. about one accept in 222. A single-fixture rate carries a denominator of `rounds`, so every
arm prints its own binomial-style half-width (`1.96*sqrt(a(1-a)/n)`); a variant that moves the rate by less
than that on this fixture has not been measured, it has been observed.

USAGE
    prototype/mtp_accept/draft_cond_probe.sh                       # default arms, default fixture
    FIXTURE=/tmp/dcp/dec8.csv STEPS=32 prototype/mtp_accept/draft_cond_probe.sh
    prototype/mtp_accept/draft_cond_probe.py --fixture dec8 --steps 32 --arm dense=--mtp-dense,1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

# Imported, not reimplemented: the admission criterion, the near-tie band and the per-relation checks are
# `m3_loop_rate.py`'s, and two spellings of a band is how one of them gets loosened by accident.
from m3_loop_rate import ROUND_RE, check_line, fields, run_binary  # noqa: E402
from run_accept import DEPTH_BUCKETS, die  # noqa: E402

MODEL_DEFAULT = "models/Qwen3.5-0.8B"

# The ledger document llama.cpp's MTP A/B was measured on (`prototype/mtp_llama/llama_mtp_ab.sh:52`),
# truncated to the same 10,200 bytes, because that is the only text on which our rate and llama.cpp's
# 0.750-at-pos1 are the same measurement rather than two measurements.
LLAMA_TEXT_SRC = "progress/2026-10-21/performance-status-2026-10-21.md"
LLAMA_TEXT_BYTES = 10200

# Every accept-loop arm carries these: `--gdn-loop 1` pins the per-token recurrence the round's
# mid-checkpoint is a product of, and `--no-stop` keeps a stop id from ending the run early and shrinking
# the denominator mid-comparison (the rate work is about the guess, not about §07.5).
BASE_ARGS = "--mtp-draft 1 --mtp-accept 1 --gdn-loop 1 --no-stop"

# The default arm set: the shipping loop, then one knob per arm, then the two together. An `--arm`
# spec carries only the VARIANT KNOBS; `BASE_ARGS` is prepended to every accept-loop arm by this file, so
# a shell cannot accidentally drop `--gdn-loop 1` from one arm and price the flag instead of the knob
# (AGENTS.md: a flag one arm requires belongs on both arms). `side`/`side-pl32` are the CONTROL pair —
# full argument strings, because the control is not an accept loop.
DEFAULT_ARMS = [
    ("loop", ""),
    ("dense", "--mtp-dense 1"),
    ("pl32", "--mtp-prompt-lanes 32"),
    ("pl128", "--mtp-prompt-lanes 128"),
    ("pl32dense", "--mtp-prompt-lanes 32 --mtp-dense 1"),
]
DEFAULT_CONTROL = [
    ("side", "--mtp-draft 1 --mtp-sched every --gdn-loop 1 --no-stop"),
    ("side-pl32", "--mtp-draft 1 --mtp-sched every --gdn-loop 1 --no-stop --mtp-prompt-lanes 32"),
]


def build_fixture(name: str, out_dir: pathlib.Path, model: str) -> pathlib.Path:
    """Materialise a named fixture as a COMMA-separated ids file and assert the field count.

    `--ids-file` splits on commas, so a space-joined fixture reads as ONE token and the run still
    "succeeds" at width 1 (AGENTS.md). The count is read back off the file, never off the builder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if name.endswith(".csv"):
        src = pathlib.Path(name)
        if not src.exists():
            die(f"fixture {src} does not exist")
        dest = out_dir / src.name
        dest.write_text(src.read_text())
        path = dest
    elif name == "llama":
        txt = out_dir / "llama_md.txt"
        raw = (ROOT / LLAMA_TEXT_SRC).read_bytes()[:LLAMA_TEXT_BYTES]
        txt.write_bytes(raw)
        path = out_dir / "llama_md.csv"
        cmd = [sys.executable, str(HERE / "ids_from_text.py"), "--model", model,
               "--text", str(txt), "--out", str(path)]
        print("    $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if p.returncode != 0:
            die(f"ids_from_text failed: {p.stdout}{p.stderr}")
        print("    " + p.stdout.strip(), flush=True)
    elif name == "dec8":
        man = json.loads((ROOT / "data/refs/dec8/manifest.json").read_text())
        ids = man.get("prefill_ids") or (man.get("sources") or [{}])[0].get("ids")
        if not ids:
            die(f"dec8 manifest has no prefill_ids and no sources[0].ids; keys={sorted(man)}")
        path = out_dir / "dec8.csv"
        path.write_text(",".join(str(i) for i in ids) + "\n")
    else:
        die(f"unknown fixture '{name}' (use llama|dec8|<path.csv>)")
    n = len(path.read_text().strip().split(","))
    if n < 8:
        die(f"{path}: {n} comma fields — that is the one-token silent failure")
    return path


def ladder(log: str) -> list[bool]:
    return [ROUND_RE.match(ln).group(4) == "1" for ln in log.splitlines() if ROUND_RE.match(ln)]


def half_width(accepted: int, rounds: int) -> float:
    if rounds == 0:
        return float("nan")
    a = accepted / rounds
    return 1.96 * (a * (1.0 - a) / rounds) ** 0.5


def run_one(args, tag: str, extra: str, ids_file: pathlib.Path, out: pathlib.Path,
            base: bool = True) -> dict:
    full = (BASE_ARGS + " " + extra) if base else extra
    extra_list = full.split()
    rc, log = run_binary(args.binary, args.model, ids_file, out, ["--steps", str(args.steps)] + extra_list)
    if rc != 0:
        die(f"{tag}: rc={rc}\n" + "\n".join(log.splitlines()[-8:]))
    return {"log": log, "dir": out}


def rate_summary(tag: str, info: dict, run: dict, n_ids: int) -> dict:
    """Fold one accept run into the numbers this probe prints, and re-check it via m3_loop_rate."""
    m, d = info["m"], info["d"]
    rounds, accepted = int(m["rounds"]), int(m["accepted"])
    lad = ladder(run["log"])
    dense_skip = sum(1 for ln in run["log"].splitlines() if ln.startswith("DENSE-SKIP"))
    draft_skip = sum(1 for ln in run["log"].splitlines() if ln.startswith("DRAFT-SKIP"))
    mtpdraft = [fields(ln) for ln in run["log"].splitlines() if ln.startswith("MTPDRAFT")]
    # The engine prints the census twice: once when the side computation is armed (all zeros, an echo that
    # the flag reached the engine) and once with the run's own tally. The LAST one is the run's.
    if not mtpdraft:
        die(f"{tag}: no MTPDRAFT line; the census is the only place `lane_appends` is printed, so without "
            "it the lane counters are unreadable")
    mtpdraft = [mtpdraft[-1]]
    mp = [fields(ln) for ln in run["log"].splitlines() if ln.startswith("MTPPROMPT")]
    mp = mp[0] if mp else {}
    prop = [ln for ln in run["log"].splitlines() if ln.startswith("DRAFT round=")]
    rec = {
        "arm": tag, "prompt_ids": n_ids, "rounds": rounds, "accepted": accepted,
        "rejected": int(m["rejected"]), "plain": int(d["plain"]),
        "rate": accepted / rounds if rounds else None,
        "half_width": half_width(accepted, rounds),
        "mean_emit": int(m["emitted"]) / rounds if rounds else None,
        "draft_skip": draft_skip, "dense_skip": dense_skip,
        "lane_appends": mtpdraft[0].get("lane_appends", "?"),
        "prompt_lanes_fitted": mp.get("fitted", "0"),
        "lanes_max": mtpdraft[0].get("lanes_max", "?"),
        "ended": d.get("ended", "?"), "n_proposal_lines": len(prop),
        "identical_to_greedy": info["identical"],
        "flip": (info["flip"] or {}).get("near_tie") if info["flip"] else None,
    }
    print(f"  {tag:>12} rounds={rounds:>4} accept={accepted:>4} rate={rec['rate']:.4f}"
          f" +-{rec['half_width']:.4f} emit/round={rec['mean_emit']:.4f} plain={rec['plain']:>2}"
          f" dense_skip={dense_skip} draft_skip={draft_skip} lanes_max={rec['lanes_max']}"
          f" prompt_lanes={rec['prompt_lanes_fitted']} lane_appends={rec['lane_appends']}"
          f" ended={rec['ended']}"
          + ("" if rec["identical_to_greedy"] is None else
             (" stream=same-as-greedy" if rec["identical_to_greedy"] else
              (f" stream=near-tie-flip@{info['flip']['flip']}" if info["flip"] and info["flip"]["near_tie"]
               else " stream=STRUCTURAL-FLIP")))
          , flush=True)
    print(f"  {'':>12} ladder (accept by round ordinal):", end="")
    for lo, hi in DEPTH_BUCKETS:
        sel = [ok for i, ok in enumerate(lad) if lo <= i <= hi]
        if sel:
            print(f"  {lo}-{hi if hi < 10**8 else '+'}={sum(sel)}/{len(sel)}", end="")
    print("", flush=True)
    return rec


def control_stream_equal(a: pathlib.Path, b: pathlib.Path, names: list[str]) -> list[str]:
    """Which of the named target-side files differ between two runs (empty = byte-identical)."""
    bad = []
    for n in names:
        pa, pb = a / n, b / n
        if pa.exists() != pb.exists():
            bad.append(f"{n}:present={pa.exists()}/{pb.exists()}")
        elif pa.exists() and pa.read_bytes() != pb.read_bytes():
            bad.append(n)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "engine/build/decode_run"))
    ap.add_argument("--model", default=str(ROOT / MODEL_DEFAULT))
    ap.add_argument("--fixture", default="llama")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--out", default="/tmp/draft_cond_probe")
    ap.add_argument("--arm", action="append", default=[],
                    help="name=VARIANT-KNOBS (comma or space separated); repeats; replaces the default "
                         "set. The accept-loop base flags are added by this file, never by the spec.")
    ap.add_argument("--control", action="append", default=[],
                    help="name=EXTRA ARGS for the target-pinned control pair; `none` disables it."
                         " Defaults to the plain/side-pl32 pair, which is the pair that adjudicates the"
                         " prompt-lane knob")
    ap.add_argument("--no-admission", action="store_true",
                    help="skip the flag-off pair run (the near-tie admission then cannot be adjudicated and"
                         " every arm reports stream=? instead)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids_file = build_fixture(args.fixture, out, args.model)
    n_ids = len(ids_file.read_text().strip().split(","))

    arms = []
    for spec in (args.arm or DEFAULT_ARMS):
        name, _, extra = spec.partition("=")
        arms.append((name, extra.replace(",", " ") if extra else ""))
    controls = []
    for spec in (args.control or DEFAULT_CONTROL):
        if spec == "none":
            controls = []
            break
        if isinstance(spec, tuple):        # a DEFAULT_CONTROL entry, already split
            name, extra = spec
        else:                              # a `--control name=KNOBS` from the command line
            name, _, extra = spec.partition("=")
            extra = extra.replace(",", " ") if extra else ""
        controls.append((name, extra))

    print(f"=== draft-conditioning rate probe — fixture={args.fixture} ids={n_ids}"
          f" steps={args.steps} binary={args.binary}")
    print("    NO TIMING: this harness reads no clock and prints no millisecond figure.")

    # ---- the flag-off pair: the greedy stream every accept arm is adjudicated against -----------------
    base_dir = out / "greedy"
    if args.no_admission:
        base_log, base_dir = None, None
        print("    admission SKIPPED (--no-admission): stream legality is not adjudicated below.")
    else:
        rc, base_log = run_binary(args.binary, args.model, ids_file, base_dir,
                                  ["--steps", str(args.steps)])
        if rc != 0:
            die(f"flag-off pair run rc={rc}\n" + "\n".join(base_log.splitlines()[-6:]))
        greedy_steps = [ln for ln in base_log.splitlines() if ln.startswith("STEP kind=decode")]
        if not greedy_steps:
            die("flag-off run printed no `STEP kind=decode` line; the admission has no reference")
        print(f"    flag-off greedy stream: {len(greedy_steps)} generation steps")

    recs = []
    for name, extra in arms:
        d = out / f"acc_{name}"
        run = run_one(args, f"arm {name}", extra, ids_file, d)
        info = check_line(f"arm {name}", run["log"], d, base_dir, base_log, n_ids)
        recs.append(rate_summary(name, info, run, n_ids))

    # ---- CONTROL: same input tokens => same verify logits, different guess ---------------------------
    # The side pass prices a proposal and settles it against a later ordinary step, so the run's own token
    # stream is the greedy one whatever the draft does. Byte-identical target dumps with a DIFFERENT
    # proposal list is exactly the separation this probe needs: a better guess, not a different model.
    ctrl = []
    for name, extra in controls:
        d = out / f"ctl_{name}"
        run = run_one(args, f"control {name}", extra, ids_file, d, base=False)
        props = [ln for ln in run["log"].splitlines() if ln.startswith("DRAFT round=")]
        rec = {"arm": name, "n_proposals": len(props),
               "proposals": [fields(ln).get("proposal") for ln in props]}
        ctrl.append(rec)
        print(f"  control {name:>10} proposals={len(props)}", flush=True)
    if len(ctrl) == 2:
        a, b = out / f"ctl_{ctrl[0]['arm']}", out / f"ctl_{ctrl[1]['arm']}"
        files = ["sampled.txt"] + [p.name for p in sorted(a.glob("logits*.bin"))]
        bad = control_stream_equal(a, b, files)
        pa, pb = ctrl[0]["proposals"], ctrl[1]["proposals"]
        n_diff = sum(1 for x, y in zip(pa, pb) if x != y)
        same_prop = pa == pb
        verdict = "IDENTICAL" if not bad else f"DIFFERS({len(bad)}): {bad[:4]}"
        prop_line = "SAME" if same_prop else f"DIFFER ({n_diff} of {len(pa)})"
        print(f"CONTROL target-dumps {verdict} across {len(files)} files; proposals {prop_line}")
        if bad:
            print("CONTROL FAIL — the variant moved the target's own outputs. That is not a better guess;")
            print("              stop and read the diff before reading any rate above.")
        elif same_prop:
            print("CONTROL INCONCLUSIVE — identical dumps AND identical proposals means the knob did")
            print("                   nothing at all in this run; it is a no-op, not a null result.")
    else:
        print("CONTROL not run: two control arms are needed to byte-compare. Name them with `--control`,"
              " or drop `--control none`.")

    # ---- the comparison the whole probe exists for ----------------------------------------------------
    base = next((r for r in recs if r["arm"] == "loop"), recs[0] if recs else None)
    print("\n=== movement against the shipping loop (the only comparison that is inside one window) ===")
    for r in recs:
        if r is base:
            continue
        d = r["rate"] - base["rate"]
        # Both arms ran in this one process, on this one fixture, with the same pair run — but they are
        # separate runs, so the honest error bar is the pair's, not one arm's.
        band = (base["half_width"] ** 2 + r["half_width"] ** 2) ** 0.5
        clears = abs(d) >= max(band, 0.05)
        print(f"  {r['arm']:>12}: a={r['rate']:.4f} vs {base['rate']:.4f}  delta={d:+.4f}"
              f"  pair noise=+-{band:.4f}  threshold 0.05  ->"
              f" {'CLEARS' if clears else 'within noise'}"
              f"   (a accept = {r['accepted'] - base['accepted']:+d} of {r['rounds']} rounds)")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(
            {"fixture": args.fixture, "ids": n_ids, "steps": args.steps, "binary": args.binary,
             "arms": recs, "controls": [{k: v for k, v in c.items() if k != "proposals"} for c in ctrl]},
            indent=2) + "\n")
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
