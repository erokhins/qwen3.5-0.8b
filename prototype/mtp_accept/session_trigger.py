#!/usr/bin/env python3
"""session_trigger.py — what geometry makes `--session --mtp-accept 1` disagree with the plain step, and
does every disagreement adjudicate as a near-tie?

THE QUESTION THIS EXISTS TO ANSWER. `session_longturn.sh` localized a defect-shaped symptom: a session whose
first turn crosses the one-pass attention ceiling (MAXSC = 512 lanes) and then takes a second `feed` emits a
different stream with rounds on (`SESSION-ACCEPT.md` §8). A symptom that appears in one geometry and not in
its neighbours is either a geometric bug or a coincidence, and the only way to tell is to sweep the geometry
and read the DECISION MARGIN at every position a round decided. A flip that lands on a sub-ULP tie is a
coincidence (two legal bf16 forms broke a tie differently); a flip at an ordinary margin is a bug.

WHAT EACH COLUMN MEANS, and which ones have teeth:
  verdict      LOSSLESS, or DIFFER with the first differing GEN position.
  rnd          how many positions in the dump window a ROUND decided (0 means the case speculated nowhere and
               proves nothing about rounds).
  min/ULP      over those positions: the flag-off row's top1-top2 margin divided by one bf16 ULP of that row's
               top logit. This is the column that predicts a flip: a case with no sub-ULP position CANNOT
               flip by the tie explanation, and a case that flips without one is unexplained.
  sub-ULP      how many of the round positions are inside that band.
  flip/ULP     the ratio AT THE FIRST FLIP (`-` when nothing flipped). The verdict and this column are printed
               side by side so a reader can disagree with the reading.
  roundgap     median 1-cos between a round row and the flag-off step row at the same position: the size of
               the body-form difference itself (~1e-5 for a re-associated long dot; ~1e-1 for something else).
  step-pre     the two arms' plain STEP rows at positions at or before the arm's first round row, as
               differing/total. In the `no-rounds` case (the first row of the table, a one-token second turn
               in which no round can run) these MUST be 0/n: that is the control with teeth, and it is what
               separates "the round's form differs" from "turning the flag on changed a plain step".
  step-post    step rows AFTER the first round, same 1-cos measure, reported and not gated: a multi-row
               body writes a different legal spelling of the same state, and every later row inherits that
               hair. That inheritance is the mechanism, so its size is the number worth reading.

THE POPULATION IS DERIVED. Prompts come from `data/refs/*/manifest.json` (the first fixture of 250-900 ids, so
turn 1 can cross 512 while generating), the case list is built from the command-line axes, and the harness
asserts `cases_ran == |case list|` before printing a verdict.

GPU: this does NOT take `tools/gpu_lock.sh`; run it through `session_trigger.sh`, which does. Correctness and
margins only — no millisecond figure is read or printed anywhere in this file.

USAGE
    prototype/mtp_accept/session_trigger.sh                            # the standard grid, shipping config
    prototype/mtp_accept/session_trigger.sh --config k1fixed --at 512,540,600
    prototype/mtp_accept/session_trigger.py --at 540 --dump 0          # token-only, no logits dumped
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from session_rowdiff import (BF16_REL_ULP, greedy_argmax, parse_dumps, read_row)  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
REFS = ROOT / "data" / "refs"
PURE_CASE = "no-rounds"
# The three ids the other session gates feed at a turn boundary (13 = " The", 198 = newline, 760 = " I").
FEED_TOKENS = [13, 198, 760]

# The accept-arm flag sets the table sweeps. Each one is a geometry the engine can actually be run at, and
# the flag-off arm is the same for all of them: `--mtp-accept 0 --mtp-draft 0`.
FLAGS_BY_CONFIG = {
    "off": ["--mtp-accept", "0", "--mtp-draft", "0"],
    "ship": ["--mtp-accept", "1"],                                  # k=2, adaptive on: what ships
    "k1": ["--mtp-accept", "1", "--mtp-k", "1"],
    "fixed": ["--mtp-accept", "1", "--mtp-adaptive", "0"],
    "k1fixed": ["--mtp-accept", "1", "--mtp-k", "1", "--mtp-adaptive", "0"],
    "gdn1": ["--mtp-accept", "1", "--gdn-loop", "1"],                # case G: the span the round forces
    # Case G, exactly: `--gdn-loop 1` on BOTH arms, so the span the round forces is no longer a difference
    # between them at all. If the divergence were the span, this would be lossless; if it is a tie, this is
    # the arm where the tie can break the other way — which is what §8 observed of G.
    "gdn1both": ["--mtp-accept", "1", "--gdn-loop", "1"],
}
# The flag-off side, per config. Only `gdn1both` needs a non-default one.
OFF_BY_CONFIG = {"gdn1both": ["--mtp-accept", "0", "--mtp-draft", "0", "--gdn-loop", "1"]}


def feed_ids(n: int) -> list[int]:
    """The second turn's `feed`, for any size n.

    The default text is the one `session_longturn.sh` and `session_lossless.py` feed, so the table's
    `at=535 feed=3` row IS case B and not a look-alike: whether a stream contains a sub-ULP tie is a property
    of the text, so a table that silently fed different text could not reproduce the case it claims to sweep.
    Longer feeds cycle the same small list, which keeps every size made of real token ids.
    """
    base = FEED_TOKENS
    return [base[i % len(base)] for i in range(n)]


def fixture_ids(lo: int, hi: int) -> list[int]:
    """The first DERIVED fixture whose prompt length is inside [lo, hi]."""
    for d in sorted(REFS.glob("*/manifest.json")):
        try:
            j = json.loads(d.read_text())
        except (OSError, ValueError):
            continue
        ids = None
        for key in ("sources", "prompts"):
            for row in j.get(key) or []:
                if isinstance(row, dict) and isinstance(row.get("ids"), list) and row["ids"]:
                    ids = row["ids"]
                    break
            if ids:
                break
        if ids is None:
            ids = j.get("ids")
        if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids) and lo <= len(ids) <= hi:
            return ids
    raise SystemExit(f"NO VERDICT: no fixture under {REFS} has a prompt of {lo}-{hi} ids")


class Arm:
    """One resident `decode_run --session`, reused across cases and separated by `reset` (§08.4)."""

    def __init__(self, binary: str, model: str, flags: list[str], out_dir: pathlib.Path, cap: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("session_*.bin"):
            old.unlink()
        self.p = subprocess.Popen([binary] + flags + ["--model", model, "--session",
                                                      "--cap", str(cap), "--out", str(out_dir)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lines: list[str] = []
        while True:
            raw = self.p.stdout.readline()
            if not raw:
                raise SystemExit("engine closed the session before READY")
            raw = raw.rstrip("\n")
            self.lines.append(raw)
            if raw.startswith("READY"):
                break

    def run(self, cmds: list[str]) -> list[str]:
        out: list[str] = []
        for c in cmds:
            self.p.stdin.write(c + "\n")
            self.p.stdin.flush()
            while True:
                raw = self.p.stdout.readline()
                if not raw:
                    raise SystemExit(f"engine closed the session on {c!r}")
                raw = raw.rstrip("\n")
                self.lines.append(raw)
                word = raw.split(" ", 1)[0]
                if word in ("GEN", "FED", "END", "PREFIX", "RESET", "DUMP", "DUMP-MISS", "ERROR"):
                    out.append(raw)
                # One reply line per command. `DUMP` needs its own test: the ARMING reply ends the `dump`
                # command, while the `DUMP seq=` lines that follow belong to the NEXT command's bodies and
                # must not end it — breaking on those would leave the engine's output one line ahead of the
                # command that produced it and desynchronise every later transcript line.
                if word in ("FED", "END", "PREFIX", "RESET", "ERROR", "BYE", "READY"):
                    break
                if raw.startswith("DUMP armed"):
                    break
        return out

    def close(self) -> None:
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        except (BrokenPipeError, ValueError, subprocess.TimeoutExpired):
            self.p.kill()


def join_cases(off_lines: list[str], on_lines: list[str], off_dir: pathlib.Path, on_dir: pathlib.Path):
    """Join the two arms' dumps by position; report flips, the round/step split, and both step gaps."""
    dg_off, _ = parse_dumps("\n".join(off_lines))
    dg_on, _ = parse_dumps("\n".join(on_lines))
    if not dg_off or not dg_on:
        return None
    cache: dict[tuple[str, int], np.ndarray] = {}

    def row(which: str, pos: int):
        dg = dg_off if which == "off" else dg_on
        rec = dg.get(pos)
        if rec is None:
            return None
        key = (which, pos)
        if key not in cache:
            cache[key] = read_row(off_dir if which == "off" else on_dir, rec["file"], rec["vocab"],
                                  rec["bytes"])
        return cache[key]

    def gap(a: np.ndarray, b: np.ndarray) -> float:
        """1 - cos between two rows, with bit-identity reported as exactly 0.

        Bit-identical float32 rows still give a dot/normal product of 1 +/- 1e-7 in float64, which prints as
        -8e-08 and reads like a defect. `ndiff == 0` is the fact being reported, so report it as 0."""
        if not np.count_nonzero(a != b):
            return 0.0
        nrm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if nrm == 0 else 1.0 - float(np.dot(a, b)) / nrm

    shared = sorted(set(dg_off) & set(dg_on))
    if not shared:
        return None
    round_first = min([p for p in shared if dg_on[p]["kind"] == "round"], default=None)
    # A step row at or before the arm's first round row is a pure-step row: same code path, same state, so
    # it must agree BIT FOR BIT. A step row AFTER a round is not a control at all — a round's writes are a
    # different legal spelling of the same state and every later row inherits that hair — so the two are
    # measured separately and only the first is gated (by the `no-rounds` case, which has no rounds).
    flips, round_pos, round_gaps, pre_gaps, pre_n, post_pairs = set(), [], [], [], 0, []
    off_rows = {}
    all_rows: dict[int, np.ndarray] = {}
    for pos in shared:
        a, b = row("off", pos), row("on", pos)
        if a is None or b is None:
            continue
        kind = dg_on[pos]["kind"]
        all_rows[pos] = a
        if greedy_argmax(a) != greedy_argmax(b):
            flips.add(pos)
        if kind == "round":
            round_pos.append(pos)
            round_gaps.append(gap(a, b))
            off_rows[pos] = a
        elif kind == "step":
            if round_first is None or pos <= round_first:
                pre_n += 1
                if np.count_nonzero(a != b):
                    pre_gaps.append(pos)
            else:
                post_pairs.append((pos, gap(a, b)))
    post_keep = [g for p, g in post_pairs if not flips or p < min(flips)]
    return {"flips": flips, "round_positions": round_pos, "off_rows": off_rows,
            "off_rows_all": all_rows,
            "median_round_row_gap": (float(np.median(round_gaps)) if round_gaps else None),
            "max_round_row_gap": (max(round_gaps) if round_gaps else None),
            "step_pre_round": pre_n, "step_pre_round_differing": len(pre_gaps),
            # Only the pairs BEFORE the first flip measure the state's own form gap. Past the flip the two
            # arms are fed different tokens, so their rows differ in content and the number is meaningless.
            "median_step_post_gap": (float(np.median(post_keep)) if post_keep else None),
            "step_post_pairs": len(post_pairs)}


def margin_stats(off_rows: dict[int, np.ndarray], round_pos: list[int], flips: set[int]) -> dict:
    """Margin/ULP over the positions a round decided, plus the ratio at each flipped position.

    The flip ratios are read from `off_rows`, which the caller fills for EVERY shared position: a flip can
    land on a position the accept arm decided with a plain step (after a rejected round or a budget
    demotion), and the row that adjudicates it is still the flag-off row at that position. Adjudicating only
    the round rows would call a legal sub-ULP tie “unexplained” because of which body happened to print it.
    A flipped position with no row yields None, which the caller turns into NO VERDICT — never a pass."""
    def ratio(pos: int):
        r = off_rows.get(pos)
        if r is None or not len(r):
            return None
        o = np.argsort(r)[::-1]
        margin = float(r[o[0]] - r[o[1]])
        return margin / max(abs(float(r[o[0]])) * BF16_REL_ULP, 1e-30)

    ratios = [x for x in (ratio(p) for p in round_pos) if x is not None]
    flip_ratios = {p: ratio(p) for p in sorted(flips)}
    known = [x for x in flip_ratios.values() if x is not None]
    return {"n_round_positions": len(ratios),
            "min_margin_over_ulp": (min(ratios) if ratios else None),
            "n_sub_ulp_positions": sum(1 for x in ratios if x < 1.0),
            "flip_margin_over_ulp": (min(known) if known else None),
            "flip_ratios": flip_ratios}


def selftest() -> int:
    """Proves the classification can go red: a flip at an ordinary margin must NOT be called a tie."""
    wide = np.zeros(8, np.float32); wide[3] = 10.0; wide[4] = 9.0        # margin 1.0, top logit 10
    tight = np.zeros(8, np.float32); tight[3] = 10.0; tight[4] = 10.0 - 0.001
    checks = [
        ("wide margin is not a tie",
         margin_stats({7: wide}, [7], {7})["flip_margin_over_ulp"] >= 1.0),
        ("sub-ULP margin is a tie",
         margin_stats({7: tight}, [7], {7})["flip_margin_over_ulp"] < 1.0),
        ("no round positions yields no verdict, not a pass",
         margin_stats({}, [], set())["flip_margin_over_ulp"] is None),
        ("a flip on a non-round position is still adjudicated by that row's margin",
         margin_stats({7: tight}, [], {7})["flip_margin_over_ulp"] < 1.0),
        ("a flip whose row is missing yields None, not a pass",
         margin_stats({}, [], {7})["flip_margin_over_ulp"] is None),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("SELFTEST", "PASS" if all(ok for _, ok in checks) else "FAIL")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=str(ROOT / "engine" / "build" / "decode_run"))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--out", default="/tmp/session_trigger")
    ap.add_argument("--maxsc", type=int, default=512)
    ap.add_argument("--gen2", type=int, default=20, help="tokens generated after the second feed")
    ap.add_argument("--config", default="ship", help=f"one of {sorted(FLAGS_BY_CONFIG)[1:]}")
    ap.add_argument("--at", default="400,505,510,511,512,513,520,535,540",
                    help="prefix_len at the moment of the second feed; > maxsc means turn 1 crossed. The"
                         " default stops where the derived prompt's first turn still reaches: a turn that"
                         " prints the end-of-turn id earlier is reported TURN-STOPPED, not silently kept")
    ap.add_argument("--feed", default="3", help="comma list of second-feed sizes; size n feeds the first n"
                         " of FEED_TOKENS (cycled), which is the same text `session_longturn.sh` feeds")
    ap.add_argument("--dump", default="1", choices=("0", "1"),
                    help="1 (default) arms the session logits dump and reports margins; 0 is a token-only "
                         "verdict table")
    ap.add_argument("--log-prefix", default="",
                    help="write each arm's transcript to <prefix>.off.log / <prefix>.on.log so"
                         " session_rowdiff.py can adjudicate the flips in detail")
    ap.add_argument("--cleanup", default="1", choices=("0", "1"),
                    help="delete each case's dump files once its margins are computed (the full grid is a"
                         " few GB of F32 rows otherwise)")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true", help="check the classifier, no device")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    if args.config not in FLAGS_BY_CONFIG:
        raise SystemExit(f"--config must be one of {sorted(FLAGS_BY_CONFIG)}")
    PROMPT = fixture_ids(250, 900)
    L0 = len(PROMPT)
    at_list = [int(x) for x in args.at.split(",") if x.strip()]
    feed_list = [int(x) for x in args.feed.split(",") if x.strip()]
    cases: list[tuple[str, list[str], int, int]] = []
    if args.config != "off":
        # One token after the feed is decided by the FIRST body of the turn, which is always a plain step
        # (there are no proposals yet), so no round runs anywhere in the window: the two arms must agree bit
        # for bit. Without this row the table could not tell a form gap from the accept machinery itself.
        # Round-free by construction: after a `feed` there are no proposals, so the FIRST body is a plain
        # step, and asking for exactly one token makes that step the only body the turn ever runs. Any
        # difference here is the accept MACHINERY (flag parse, draft block, dump arming) touching a plain
        # step — not a form gap between body shapes. Kept at the prompt's own last position for the same
        # reason: turn 1 of every other case is already speculating.
        cases.append((PURE_CASE, ["reset", f"feed {' '.join(map(str, PROMPT))}",
                                  f"dump {L0 - 2} {L0 + 1}", "gen 1"], L0, 0, L0 + 1))
    for at in at_list:
        if at <= L0:
            print(f"skip at={at}: not above the prompt ({L0} ids), so it cannot test a crossing")
            continue
        for nf in feed_list:
            # The dump window is the whole second turn plus the `feed`'s own deciding row (`at + nf - 1`)
            # and a lane of headroom, so every position the second turn can decide is covered.
            window = ([f"dump {at} {at + nf + args.gen2 + 4}"] if args.dump == "1" else [])
            cmds = ["reset", f"feed {' '.join(map(str, PROMPT))}"] + window + [
                f"gen {at - L0}", f"feed {' '.join(map(str, feed_ids(nf)))}", f"gen {args.gen2}"]
            cases.append((f"at={at} feed={nf}", cmds, at, nf, at))
    if not cases:
        raise SystemExit("NO VERDICT: every requested `at` is at or below the prompt length")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = max(c[2] for c in cases) + max(c[3] for c in cases) + args.gen2 + 64
    off_dir, on_dir = out / f"{args.config}.off", out / f"{args.config}.on"
    a_off = Arm(args.binary, args.model, OFF_BY_CONFIG.get(args.config, FLAGS_BY_CONFIG["off"]),
                off_dir, cap)
    a_on = Arm(args.binary, args.model, FLAGS_BY_CONFIG[args.config], on_dir, cap)

    print(f"=== session trigger table: config={args.config}  prompt={L0} ids (derived from data/refs/)"
          f"  MAXSC={args.maxsc}  gen2={args.gen2}  dump={args.dump} ===")
    print(f"{'case':>16} {'cross?':>7} {'verdict':>16} {'rnd':>4} {'min/ULP':>8} {'sub-ULP':>7}"
          f" {'flip/ULP':>9} {'roundgap':>9} {'step-pre':>9} {'step-post':>10}")
    rows, bad = [], 0
    for name, cmds, at, nf, expect_end in cases:
        lo = a_off.run(cmds)
        ln = a_on.run(cmds)
        gen_off = [l for l in lo if l.startswith("GEN ")]
        gen_on = [l for l in ln if l.startswith("GEN ")]
        first = next((x.split()[2] for x, y in zip(gen_off, gen_on) if x != y), None)
        j = join_cases(lo, ln, off_dir, on_dir) if args.dump == "1" else {"disabled": True}
        # Did turn 1 even reach the requested geometry? A turn that prints the end-of-turn token early
        # leaves `prefix_len` short of `at`, the dump window then covers nothing, and the case tests a
        # different prompt shape than the one the row claims. Report that as itself, not as a numerics result.
        end1 = next((l for l in lo if l.startswith("END ")), "")
        reached = int(end1.split("len=")[1].split()[0]) if "len=" in end1 else None
        if args.cleanup == "1":
            for d in (off_dir, on_dir):
                for f in d.glob("session_*.bin"):
                    f.unlink()      # the full grid is a few GB of F32 rows otherwise
        crossed = "crossed" if at > args.maxsc else "no"
        verdict = "LOSSLESS" if first is None else f"DIFFERS at {first}"
        if reached is not None and reached != expect_end:
            print(f"{name:>16} {crossed:>7} TURN-STOPPED: turn 1 ended at {reached} ids, expected"
                  f" {expect_end}; this row tests no requested geometry")
            rows.append({"case": name, "config": args.config, "at": at, "feed": nf,
                         "crossed": at > args.maxsc, "turn_stopped_at": reached, "first_diff": None})
            continue
        if j is None:
            print(f"{name:>16}  NO VERDICT: no joined dump rows — the window missed every body row")
            bad += 1
            continue
        if "disabled" in j:
            print(f"{name:>16} {crossed:>7} {verdict:>16} {'-':>4} {'-':>8} {'-':>7} {'-':>9}"
                  f" {'-':>9} {'-':>9} {'-':>10}")
            rows.append({"case": name, "config": args.config, "at": at, "feed": nf,
                         "crossed": at > args.maxsc, "first_diff": first})
            continue
        m = margin_stats(j["off_rows_all"], j["round_positions"], j["flips"])
        if name == PURE_CASE:
            # Round-free by construction (see the case builder), so bit-identity is the requirement, and
            # `rnd` must read 0 or the case stopped being a control.
            if first is not None or j["step_pre_round_differing"] or j["flips"] or m["n_round_positions"]:
                verdict += (f" +NO-ROUND-PARITY-BROKEN(rnd={m['n_round_positions']}"
                            f",stepdiff={j['step_pre_round_differing']})")
                bad += 1
        if first is not None and not m["n_round_positions"]:
            print(f"{name:>16}  NO VERDICT: streams differ but no round decided a dumped position")
            bad += 1
            continue
        fmt = lambda x, f: "-" if x is None else (f % x)    # noqa: E731 - one-line formatter for the table
        print(f"{name:>16} {crossed:>7} {verdict:>16} {m['n_round_positions']:>4}"
              f" {fmt(m['min_margin_over_ulp'], '%.4f'):>8} {m['n_sub_ulp_positions']:>7}"
              f" {fmt(m['flip_margin_over_ulp'], '%.4f'):>9}"
              f" {fmt(j['median_round_row_gap'], '%.2e'):>9}"
              f" {str(j['step_pre_round_differing']) + '/' + str(j['step_pre_round']):>9}"
              f" {fmt(j['median_step_post_gap'], '%.2e'):>10}")
        row = {"case": name, "config": args.config, "at": at, "feed": nf,
               "crossed": at > args.maxsc, "first_diff": first,
               "step_pre_round_differing": j["step_pre_round_differing"],
               "median_round_row_gap": j["median_round_row_gap"],
               "max_round_row_gap": j["max_round_row_gap"],
               "median_step_post_gap": j["median_step_post_gap"], **m}
        rows.append(row)
        if first is not None and any(x is None for x in m["flip_ratios"].values()):
            print("    ^ NO VERDICT: a flipped position has no flag-off row in the dump window")
            bad += 1
        if first is not None and (m["flip_margin_over_ulp"] is None
                                  or m["flip_margin_over_ulp"] >= 1.0):
            print("    ^ DIFFERS at an ordinary margin: NOT explicable as a tie between two legal forms")
    if args.log_prefix:
        # Written from the in-memory transcripts, so the logs are exactly what the comparator would have to
        # reconstruct from the process otherwise. `dump` lines included: they are the join key.
        for tag, arm in (("off", a_off), ("on", a_on)):
            pathlib.Path(f"{args.log_prefix}.{tag}.log").write_text("\n".join(arm.lines) + "\n")
    a_off.close()
    a_on.close()
    if len(rows) != len(cases):
        print(f"NO VERDICT: ran {len(rows)} of {len(cases)} cases")
        return 1
    diff = [r for r in rows if r["first_diff"] is not None]
    unexplained = [r for r in diff if args.dump == "1" and (r.get("flip_margin_over_ulp") is None
                   or r["flip_margin_over_ulp"] >= 1.0)]
    print(f"\ncases {len(rows)} (derived geometry grid); DIFFER {len(diff)}; of which unexplained by a"
          f" sub-ULP margin {len(unexplained)}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"config": args.config, "prompt_ids": L0,
                                                       "maxsc": args.maxsc, "cases": rows}, indent=2) + "\n")
        print(f"json: {args.json}")
    return 1 if (bad or unexplained) else 0


if __name__ == "__main__":
    sys.exit(main())
