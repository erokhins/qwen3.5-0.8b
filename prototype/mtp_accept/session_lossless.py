#!/usr/bin/env python3
"""Does `--session --mtp-accept 1` emit the same tokens, at the same positions, as plain decode?

THE CLAIM. Greedy speculation cannot change the output: a round emits a candidate only when the device
proved the target's own argmax equals it, and otherwise emits the target's own argmax
(`engine/mtp_accept.h`'s accept rule). On the RUN path that is admitted by
`prototype/mtp_accept/flagoff_bytecheck.sh`. This is its sibling for the SESSION path, where the object is
not a `sampled.txt` but the line protocol: every `GEN <token> <pos>` the caller sees, plus the `FED` /
`END` / `PREFIX` lines that carry `prefix_len`. Those must be byte-identical between two arms of ONE
scripted transcript:

    arm plain   --mtp-accept 0 --mtp-draft 0     (the plain autoregressive step)
    arm mtp     --mtp-accept 1                   (the engine's default: k=2, adaptive on)

`prefix_len` is part of the compared set on purpose. A round that printed the right tokens but left the
state one lane ahead or behind would still pass a GEN-only compare and would corrupt the NEXT turn,
because `feed` re-anchors on the state's `prefix_len`.

THE POPULATION IS DERIVED, never typed. Every `data/refs/*/manifest.json` is a case, with the prompt read
out of `sources[0].ids` or `prompts[0].ids`; the harness asserts `cases == |population|` and refuses to
print a verdict for a case that did not run. (`AGENTS.md`: an admission over a fixture list typed from
memory read 18/18 for a population of which seven names did not exist.) Four structural cases ride along
and are asserted separately: a multi-turn `feed/gen/feed/gen`, a generation-phase stop, a prompt that
CONTAINS the stop id (which must NOT stop anything — §07.5 is generation-phase only), and one long enough
to hit the draft cache's lane ceiling.

A round emitting the stop token is the interesting case, and it is decided by the driver, not here: a
round may emit 3 tokens and the stop may sit in the middle of them, in which case the tokens behind it are
suppressed and the state is cut back to the printed prefix (`engine::mtpaccept::session_body`). If that
cut were wrong, the next turn's `FED`/`GEN` positions would move and this file would go red — which is why
the stop case is run as a turn FOLLOWED BY ANOTHER TURN rather than as a one-turn run.

NO TIMING. Nothing here reads or prints a millisecond figure: this is a token-agreement instrument and a
worker-run device number is contaminated by the session's own serving (`AGENTS.md`). Token ids only.

GPU PROTOCOL: like every other `decode_run` caller, this tool does NOT take `tools/gpu_lock.sh` itself.
The caller holds it — see `session_lossless.sh`, which does.

USAGE
    .venv/bin/python3 prototype/mtp_accept/session_lossless.py [--quick] [--gen 8] [--verbose]
    prototype/mtp_accept/session_lossless.sh            # takes the lock and runs this
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REFS = ROOT / "data" / "refs"
STOP_ID = 248046                      # engine::kStopToken; never printed as its literal text
TERMINAL = ("FED", "END", "RESET", "PREFIX", "ERROR", "BYE", "READY")
# `GEN` is the streamed line: it belongs to the protocol and is COMPARED, but it does not end a reply, so
# it is collected without being a terminal (`chat.py`'s driver has the same split, between `GEN` and
# `TERMINAL`).
STREAM = ("GEN",)
COMPARE_FIELDS = TERMINAL + STREAM
# The compared protocol lines. `GEN` carries (token, pos); `FED`/`END`/`PREFIX`/`RESET` carry
# `prefix_len`, which is the state's side of the same claim.
COMPARE = ("GEN ", "FED ", "END ", "PREFIX ", "RESET ")
DRAFT_CEILING_LANES = 512             # --mtp-cap default, and usable_lanes() is min(cap, MAXSC)


def fixture_prompt(manifest: pathlib.Path) -> list[int] | None:
    """The prompt a fixture pins, or None when it pins no token list at all."""
    try:
        j = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return None
    for key in ("sources", "prompts"):
        rows = j.get(key)
        if isinstance(rows, list):
            for row in rows:
                ids = row.get("ids") if isinstance(row, dict) else None
                if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids):
                    return ids
    ids = j.get("ids")
    return ids if isinstance(ids, list) and ids else None


def population() -> list[tuple[str, list[int]]]:
    out = []
    for d in sorted(REFS.glob("*/manifest.json")):
        ids = fixture_prompt(d)
        if ids:
            out.append((d.parent.name, ids))
    return out


class Session:
    """One resident `decode_run --session` process. The load is 1.55 GB, so one process serves every
    case, separated by `reset` (§08.4's fresh request, which also resets the draft cache)."""

    def __init__(self, binary: str, model: str, cap: int, extra: list[str]):
        self.p = subprocess.Popen([binary] + extra + ["--model", model, "--session",
                                                      "--cap", str(cap)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lines: list[str] = []
        self.pending: list[str] = []

    def send(self, line: str) -> list[str]:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out: list[str] = []
        while True:
            raw = self.p.stdout.readline()
            if not raw:
                raise RuntimeError(f"engine closed the session (rc={self.p.wait()}) on {line!r}")
            raw = raw.rstrip("\n")
            self.lines.append(raw)
            word = raw.split(" ", 1)[0]
            if word in COMPARE_FIELDS:
                out.append(raw)
            if word in TERMINAL:
                return out
            # Non-terminal lines are diagnostics (`ACCEPT`, `MTPSTAT`, `ROUND-SKIP`, `GEMM-LT`, ...);
            # they are kept for the round-activity assertions and compared by NOTHING.

    def rounds_seen(self) -> list[str]:
        return [l for l in self.lines if l.startswith("ACCEPT round=")]

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except (BrokenPipeError, ValueError, subprocess.TimeoutExpired):
            self.p.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=str(ROOT / "engine" / "build" / "decode_run"))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--gen", type=int, default=8, help="tokens asked for per case")
    ap.add_argument("--quick", action="store_true",
                    help="only fixtures whose prompt is at most 64 ids (a worker-sized pass)")
    ap.add_argument("--only", default=None, help="comma-separated fixture names")
    ap.add_argument("--maxsc", type=int, default=512,
                    help="the one-pass attention ceiling in lanes (kernels::mtp_attn_max_lanes())")
    ap.add_argument("--cross-lo", type=int, default=250,
                    help="fixture-prompt bounds for the case that crosses the ceiling WHILE generating")
    ap.add_argument("--cross-hi", type=int, default=900, help="bounds, same fixture search")
    ap.add_argument("--cross-first", type=int, default=240,
                    help="minimum tokens for turn 1 of the crossing case; the value used is whichever of"
                         " this and (maxsc - prompt_len + 25) is larger")
    ap.add_argument("--long-gen", type=int, default=700,
                    help="tokens for the draft-cache-ceiling case (512 usable lanes ≈ 170 rounds at k=2)")
    args = ap.parse_args()

    all_pops = population()
    pops = list(all_pops)
    if not pops:
        print("NO VERDICT: no fixture under data/refs/ yielded a prompt")
        return 1
    if args.only:
        want = set(args.only.split(","))
        pops = [p for p in pops if p[0] in want]
        missing = want - {p[0] for p in pops}
        if missing:
            print(f"NO VERDICT: named fixtures have no prompt under data/refs/: {sorted(missing)}")
            return 1
    if args.quick:
        pops = [p for p in pops if len(p[1]) <= 64]
    print(f"population: {len(pops)} fixtures derived from {REFS}/*/manifest.json; "
          f"gen={args.gen} long_gen={args.long_gen}")

    cap = max(len(i) for _, i in pops) + max(args.long_gen, args.gen) + 64
    # The transcript is built once and replayed identically on both arms; each case is bracketed by
    # `reset` so no case can inherit another's state (§08.4).
    plan: list[tuple[str, list[str]]] = []
    # (a) one case per DERIVED fixture.
    for name, ids in pops:
        plan.append((f"fixture:{name}", ["reset", f"feed {' '.join(map(str, ids))}",
                                         f"gen {args.gen}", "stats"]))
    first_ids = pops[0][1]
    # (b) MULTI-TURN: turns 2 and 3 must still speculate. The draft cache and the adaptive tracker are
    #     reset at every `feed` (decision 3), so a driver that only worked on the first turn would still
    #     look green here unless later turns are required to anchor rounds.
    plan.append(("multi-turn", ["reset", f"feed {' '.join(map(str, first_ids))}", f"gen {args.gen}",
                                "feed 13 198 760", f"gen {args.gen}", "feed 5156",
                                f"gen {args.gen}", "stats"]))
    # (c) a generation-phase STOP, FOLLOWED BY ANOTHER `gen`: if the mid-round cut left the state ahead of
    # the printed transcript, this is where the next turn's positions would move.
    dec8 = next((i for n, i in pops if n == "dec8"), first_ids)
    plan.append(("stop-terminated", ["reset", f"feed {' '.join(map(str, dec8))}",
                                     f"gen {max(args.gen, 64)}", f"gen {args.gen}"]))
    # (d) a prompt that CONTAINS the stop id: §07.5 is generation-phase only, so this must run the whole
    # budget and end `stop=0` rather than stopping on a fed token.
    plan.append(("stop-id-in-prompt", ["reset",
                                       f"feed {' '.join(map(str, dec8))} {STOP_ID} 13",
                                       f"gen {args.gen}"]))
    # (e) long enough to fill the draft block's lane ceiling: `DRAFT-RESET` must fire there and rounds must
    # keep happening after it (decision 3, `prototype/mtp_accept/SESSION-ACCEPT.md`).
    plan.append(("draft-ceiling", ["reset", f"feed {' '.join(map(str, first_ids))}",
                                   f"gen {args.long_gen}"]))
    # (f) CROSS THE ATTENTION CEILING DURING GENERATION, then feed. Every case above builds its prefix with
    #     a `feed`, so the one-pass ceiling (MAXSC lanes) is crossed by PREFILL and a round only ever READS a
    #     prefix somebody else built. This is the geometry `SESSION-ACCEPT.md` §8/§9 found the form gap in:
    #     turn 1 grows past MAXSC under the decoder's own tokens, a second `feed` re-arms the adaptive
    #     tracker, and the rounds that follow decide positions whose lanes crossed while generating. The
    #     assertions below require the turn to have crossed and a round to have straddled the ceiling, so the
    #     case cannot sit in the population as decoration.
    cross_ids = next((i for _, i in all_pops if args.cross_lo <= len(i) <= args.cross_hi), None)
    if cross_ids is None:
        print(f"NO VERDICT: no derived fixture of {args.cross_lo}-{args.cross_hi} ids, so the"
              " cross-then-feed case cannot be built (a missing population member is named, not skipped)")
        return 1
    cross_gen = max(args.cross_first, args.maxsc - len(cross_ids) + 25)
    plan.append(("cross-then-feed",
                 ["reset", f"feed {' '.join(map(str, cross_ids))}", f"gen {cross_gen}",
                  f"feed {' '.join(map(str, cross_ids[-3:]))}", f"gen {args.gen}", "stats"]))

    # The crossing case's turn-1 length can exceed every fixture, so the ring buffer has to cover it too.
    cap = max(cap, len(cross_ids) + cross_gen + 3 + args.gen + 64)
    names = [n for n, _ in plan]
    assert len(names) == len(set(names)), "case names must be unique"
    n_fixture_cases = sum(1 for n, _ in plan if n.startswith("fixture:"))
    if n_fixture_cases != len(pops):
        print(f"NO VERDICT: {n_fixture_cases} fixture cases for a population of {len(pops)}")
        return 1
    print(f"cases: {len(plan)} = {n_fixture_cases} derived fixtures + {len(plan) - n_fixture_cases}"
          " structural")

    arms: dict[str, dict[str, list[str]]] = {}
    sess: dict[str, Session] = {}
    for label, extra in (("plain", ["--mtp-accept", "0", "--mtp-draft", "0"]),
                         ("mtp", ["--mtp-accept", "1"])):
        s = Session(args.binary, args.model, cap, extra)
        # Drain up to and including READY.
        while True:
            raw = s.p.stdout.readline().rstrip("\n")
            s.lines.append(raw)
            if raw.startswith("READY") or not raw:
                break
        per: dict[str, list[str]] = {}
        per_case_diag: dict[str, list[str]] = {}
        for case_name, cmds in plan:
            got: list[str] = []
            mark = len(s.lines)
            for c in cmds:
                got.extend(s.send(c))
            per[case_name] = [l for l in got if l.startswith(COMPARE)]
            # The ACCEPT lines of THIS case only. The global `rounds_seen()` cannot say which case produced
            # a round, and a case in which no round ran adjudicates nothing about rounds.
            per_case_diag[case_name] = [l for l in s.lines[mark:] if l.startswith("ACCEPT ")]
            if any(l.startswith("ERROR") for l in got):
                print(f"NO VERDICT: [{label}] case {case_name} returned an ERROR line: "
                      + next(l for l in got if l.startswith("ERROR")))
                s.close()
                return 1
        sess[label], arms[label] = s, per
    rounds = sess["mtp"].rounds_seen()
    cross_diag = per_case_diag.get("cross-then-feed", [])
    per_turn = [l for l in sess["mtp"].lines if l.startswith("MTPSTAT")]
    resets = [l for l in sess["mtp"].lines if l.startswith("DRAFT-RESET")]
    skips = [l for l in sess["mtp"].lines if l.startswith("DRAFT-SKIP")]
    sess["mtp"].close()
    sess["plain"].close()

    bad = 0
    for case_name, _ in plan:
        a, b = arms["plain"][case_name], arms["mtp"][case_name]
        gens_a = [l for l in a if l.startswith("GEN ")]
        gens_b = [l for l in b if l.startswith("GEN ")]
        if not gens_a or not gens_b:
            print(f"NO VERDICT: case {case_name} printed no GEN lines "
                  f"(plain={len(gens_a)} mtp={len(gens_b)}); a check that compared nothing cannot pass")
            bad += 1
            continue
        if a == b:
            print(f"ok    {case_name}: {len(gens_a)} GEN lines identical, END agrees")
        else:
            bad += 1
            print(f"FAIL  {case_name}: the two arms differ")
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"        first difference at line {i}: plain={x!r} mtp={y!r}")
                    break
            if len(a) != len(b):
                print(f"        line counts differ: plain={len(a)} mtp={len(b)}")
    # The structural assertions, so a green run cannot be green because speculation never happened.
    if not rounds:
        print("FAIL  no ACCEPT round ran anywhere: the mtp arm never speculated, so the compare above "
              "compared two plain runs")
        bad += 1
    else:
        acc = sum(1 for l in rounds if " accept=1" in l)
        print(f"info  rounds={len(rounds)} full-accept={acc} reject={len(rounds) - acc}; "
              f"DRAFT-RESET={len(resets)} DRAFT-SKIP={len(skips)}")
    if len(per_turn) < 3:
        print(f"FAIL  only {len(per_turn)} turn(s) ran a round: multi-turn speculation is unwitnessed"
              " (the draft-cache reset at `feed` must not cost a later turn its speculation)")
        bad += 1
    else:
        print(f"info  {len(per_turn)} turns ran at least one round")
    stop_lines = [l for l in sess["mtp"].lines if l.startswith("END") and " stop=1" in l]
    if not stop_lines:
        print("FAIL  no END line reported stop=1: the stop case never reached §07.5 and proved nothing")
        bad += 1
    prompt_stop = arms["mtp"]["stop-id-in-prompt"]
    if not any(l.startswith("END steps=") and " stop=0" in l for l in prompt_stop):
        print("FAIL  the prompt containing the stop id did not end with stop=0")
        bad += 1
    # The crossing case's assertions. A case that never reached the ceiling, or reached it with no round
    # straddling it, is not a witness for the above-ceiling geometry and must say so rather than sit green.
    cross_plain = arms["plain"]["cross-then-feed"]
    ends = [l for l in cross_plain if l.startswith("END ")]
    lens = [int(l.split("len=")[1].split()[0]) for l in ends if "len=" in l]
    if not lens:
        print("FAIL  cross-then-feed printed no END line; the case ran nothing")
        bad += 1
    elif max(lens) <= args.maxsc:
        print(f"FAIL  cross-then-feed never crossed: longest turn ended at {max(lens)} lanes, at or below"
              f" MAXSC={args.maxsc} (the turn stopped early — a different fixture is needed)")
        bad += 1
    else:
        print(f"info  cross-then-feed crossed: longest turn ended at {max(lens)} lanes"
              f" (MAXSC={args.maxsc})")
    anchors = []
    for l in cross_diag:
        if "anchor=" in l:
            anchors.append(int(l.split("anchor=")[1].split()[0]))
    straddle = [a for a in anchors if a + 2 > args.maxsc]
    if not anchors:
        print("FAIL  cross-then-feed ran no round at all: nothing above the ceiling was adjudicated")
        bad += 1
    elif not straddle:
        print(f"FAIL  cross-then-feed ran {len(anchors)} rounds but none decided a position above"
              f" MAXSC={args.maxsc} (max anchor {max(anchors)})")
        bad += 1
    else:
        print(f"info  cross-then-feed: {len(anchors)} rounds, {len(straddle)} deciding at or above the"
              f" ceiling (max anchor {max(anchors)})")
    if "cross-then-feed" in arms and arms["plain"]["cross-then-feed"] != arms["mtp"]["cross-then-feed"]:
        # Not a loosening: a pointer. The exact-equality predicate above stays exactly as strict as it was,
        # and this case is red until the band decision in `SESSION-ACCEPT.md` §9.7 is taken. What belongs in
        # the log is the fact that the divergence is adjudicated elsewhere, with the instrument that does it.
        print("note  cross-then-feed is the §8 geometry. Its divergence is adjudicated by"
              " session_trigger.sh / session_rowdiff.py (margin vs one bf16 ULP of the row's own top"
              " logit); §9 established the flips there are sub-ULP ties between two legal bf16 forms.")
    print("== " + ("LOSSLESS: the mtp arm emitted the same tokens at the same positions on every case"
                   if bad == 0 else f"{bad} case(s) FAILED or unadjudicable"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
