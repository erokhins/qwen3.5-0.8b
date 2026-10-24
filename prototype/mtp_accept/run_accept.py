#!/usr/bin/env python3
"""P7 / MTP — the accept rate `a` on the ENGINE's OWN bits.

P7 gate three. `prototype/mtp_gatezero/draft_survival.py` measured `a` on torch-CPU bits (0.7329 at
k=1, 20 prompts x 40 steps, `verify="steps"`) and said in terms that the number which finally funds
P7 has to come from the engine's own bits on device — A-18 says bf16 continuations are chaotic, and a
chaos floor arrives in an accept path exactly as it arrives anywhere else. This is that measurement.

WHAT IT DRIVES. One resident `decode_run --session --mtp-draft 1` process, because the load is 1.55 GB
of weights per process and 49 of them would spend their whole wall on H2D — the same reason
`tools/corpus_freerun.py` exists. Per corpus line: `reset`, `feed <ids>`, `gen <steps>`, `stats`. The
draft is a side computation inside the engine (`engine/mtp_draft.h`): no candidate row is fed to the
target, nothing is snapshotted or rolled back, and the target runs plain greedy. So what this tallies is
the accept rate of a k=1 drafter whose proposals are checked against the target's own greedy tokens.

THE POPULATION. The 49 plain-text lines of `speccheck/fixtures/corpus.txt`, tokenised from
`data/refs/corpus48/manifest.json`, which also carries the HF reference's 24-token continuations. The
manifest's texts and count are ASSERTED against the corpus file before anything runs, because an
admission over a fixture list typed from memory is not an admission (AGENTS.md: seven of nine names on
one such list did not exist, and a harness then printed `PASS=0` for them). The reference continuations
are used only to report how far the engine's token stream agrees with the reference's — context for
reading `a`, not part of its definition.

THE JOIN, and the alignment it prints. A draft made after the step whose row predicted position `p`
proposes the token at position `p+2` and is settled by the target's prediction AT position `p+1` — the
NEXT step's token, never this one. No index is assumed: every `DRAFT` line carries
`adjudicated_by_pred_pos` and the settling token is looked up in the engine's own printed
`GEN <id> <pos>` stream at that position (the session prints the PREDICTING position, which is why
`gen[i]` pairs with `greedy[i]` — the note at the head of `tools/corpus_freerun.py`). The tally is then
cross-checked against the engine's own `MTPACCEPT` lines: two independent implementations of the join
that must agree. A free run aligned on `[0:]` read a false 0/128 in this repo's history, and the fix
was never to assume an index.

PER LINE, deliberately. Positions restart with each line's own prompt, so one line's position could
settle another line's proposal if the map were global. Every join is line-scoped and the aggregates are
folded from the per-line pairs.

DENOMINATOR, named on the rate line. `proposals made` = every `DRAFT` plus every `DRAFT-SKIP`;
`adjudicated` = those with a target token at the settling position; the rest are `skipped`, each with a
reason. A skipped proposal is in `made` and never in the numerator — counting one as a miss would
understate `a`. A rate whose denominator is unstated is not a rate.

STOP RULE. id 248046 (end-of-turn) is armed in the GENERATION phase only, and the engine owns that rule
(§07.5: the `dec8` prompt legitimately contains the id, and a prompt-phase match must never stop
anything). A proposal that happens to BE 248046 settles its round like any other id — the stop test
applies to tokens the target emitted, never to a draft's proposal.

NO TIMINGS. Nothing here prints a millisecond figure: a worker-run device number is contaminated by the
session sitting on the same GPU (`AGENTS.md`) and this is a token-agreement instrument. The shell
runner prints the wall of the whole harness run, labelled as that and nothing more. Token ids only.

GPU PROTOCOL: like every other `decode_run` caller, this tool does NOT take `tools/gpu_lock.sh` itself.
The caller holds it:

    tools/gpu_lock.sh acquire "p7 accept on engine bits"
    .venv/bin/python3 prototype/mtp_accept/run_accept.py --steps 32
    tools/gpu_lock.sh release "p7 accept on engine bits"

USAGE
    .venv/bin/python3 prototype/mtp_accept/run_accept.py [--steps 32] [--sched loop|every]
        [--start 0] [--limit 49] [--binary engine/build/decode_run] [--json PATH] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]

CORPUS = ROOT / "speccheck" / "fixtures" / "corpus.txt"
REFS = ROOT / "data" / "refs" / "corpus48" / "manifest.json"
STOP_ID = 248046            # end-of-turn token; referred to by id, never rendered

# The depth buckets `draft_survival.py` reported the carried profile with (0.588 -> 0.808), so the two
# readings diff line for line instead of being two tables that happen to look similar.
DEPTH_BUCKETS = ((0, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 9))


def die(msg: str):
    raise SystemExit(f"run_accept: {msg}")


def load_population() -> list[dict]:
    """The corpus prompts, DERIVED from the two files that define them and cross-checked."""
    lines = [ln.strip() for ln in CORPUS.read_text().splitlines() if ln.strip()]
    if not lines:
        die(f"no prompt lines in {CORPUS}")
    man = json.loads(REFS.read_text())
    prompts = man.get("prompts") or []
    if len(prompts) != len(lines):
        die(f"{REFS} carries {len(prompts)} prompts but {CORPUS} has {len(lines)} lines; the accept "
            "rate would be measured on something other than the corpus")
    for i, (p, text) in enumerate(zip(prompts, lines)):
        if p.get("text") != text:
            die(f"prompt {i}: manifest text {p.get('text')!r} != corpus line {text!r}")
        if not p.get("ids"):
            die(f"prompt {i} has no token ids in {REFS}")
    return prompts


class Session:
    """Line-oriented wrapper over `decode_run --session`.

    An `ERROR` line is fatal, not a short transcript: a session that threw mid-`gen` has a state that no
    longer matches what was printed, and the tokens it did print still look perfectly fine.
    """

    def __init__(self, cmd: list[str]) -> None:
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self.log: list[str] = []

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, until) -> list[str]:
        out: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                die(f"session ended early (rc={self.proc.poll()}) waiting for a line; "
                    f"last: {out[-4:]}")
            line = line.rstrip("\n")
            self.log.append(line)
            if line.startswith("ERROR"):
                die(f"engine refused a session command: {line}")
            out.append(line)
            if until(line):
                return out

    def close(self) -> None:
        self.send("quit")
        self.read_until(lambda l: l.startswith("BYE"))


def parse_records(lines: list[str]) -> dict:
    """One line's transcript -> pred_at map, draft/accept/skip records, and the census echo."""
    pred_at: dict[int, int] = {}
    drafts: list[dict] = []
    accepts: list[dict] = []
    skips: list[dict] = []
    census: dict[str, str] = {}
    last_sample: dict[str, int] = {}
    gen_tokens, stop = 0, 0

    def fields(line: str, prefix: str) -> dict[str, str]:
        """`KEY a=1 b=2` -> {a: "1", b: "2"}. Tokens without `=` are ignored, not guessed at."""
        return dict(kv.split("=", 1) for kv in line[len(prefix):].split() if "=" in kv)

    for ln in lines:
        if ln.startswith("GEN "):
            _, tok, pos = ln.split()
            pred_at[int(pos)] = int(tok)
        elif ln.startswith("END "):
            f = fields(ln, "END ")
            gen_tokens, stop = int(f["steps"]), int(f["stop"])
        elif ln.startswith("MTPDRAFT "):
            census = fields(ln, "MTPDRAFT ")
        elif ln.startswith("LASTSAMPLE "):
            f = fields(ln, "LASTSAMPLE ")
            last_sample = {"pos": int(f["pos"]), "token": int(f["token"])}
        elif ln.startswith(("DRAFT ", "MTPACCEPT ", "DRAFT-SKIP ", "DRAFT-UNADJUDICATED ")):
            head, _, rest = ln.partition(" ")
            rec = {"kind": head}
            for kv in rest.split():
                if kv.startswith("reason="):
                    rec["reason"] = kv.split("=", 1)[1]
                elif "=" in kv:
                    k, v = kv.split("=", 1)
                    rec[k] = float(v) if k == "top_logit" else int(v)
            if rec["kind"] == "DRAFT":
                drafts.append(rec)
            elif rec["kind"] == "MTPACCEPT":
                accepts.append(rec)
            else:
                skips.append(rec)
    return {"pred_at": pred_at, "drafts": drafts, "accepts": accepts, "skips": skips,
            "last_sample": last_sample, "gen_tokens": gen_tokens, "stop": stop, "census": census}


def join_line(rec: dict) -> dict:
    """Settle each of a line's proposals against that line's own target predictions."""
    pairs: list[tuple[dict, bool]] = []      # (draft, accepted) — the depth tables fold from here
    skipped: list[dict] = []
    # The budget path's last sample is a real target prediction that the turn does not print as a token
    # (the driver closes the turn instead), and the draft can already have been settled by it. The engine
    # prints it on `LASTSAMPLE` so this join sees exactly what the engine saw; `final_pos` keeps it
    # separable, because a rate that includes it and a rate that excludes it are two different claims and
    # the reader is owed both rather than an unlabelled average of them.
    pred_at = dict(rec["pred_at"])
    final_pos = -1
    if rec["last_sample"]:
        final_pos = rec["last_sample"]["pos"]
        pred_at[final_pos] = rec["last_sample"]["token"]
    # The engine prints a `DRAFT-UNADJUDICATED` line at the end of a turn for the proposal no later
    # step settled. That is the SAME proposal as the unsettled `DRAFT` line, so it merges into it here
    # rather than being counted twice — a double count would inflate `proposals made` and understate
    # the rate by exactly one per line, which is the kind of error a denominator note cannot fix.
    flagged = {s["adjudicated_by_pred_pos"] for s in rec["skips"]
               if s["kind"] == "DRAFT-UNADJUDICATED"}
    for d in rec["drafts"]:
        pos = d["adjudicated_by_pred_pos"]
        if pos in pred_at:
            pairs.append((d, d["proposal"] == pred_at[pos]))
        elif pos in flagged:
            skipped.append({**d, "why": "engine-flagged unadjudicated: no later generation step in"
                                        " this turn settles it"})
        else:
            skipped.append({**d, "why": "no target token at the settling position: the line hit the"
                                        " stop id or the step budget before that position"})
    for s in rec["skips"]:
        if s["kind"] != "DRAFT-UNADJUDICATED":
            skipped.append({**s, "why": s.get("reason", s["kind"])})
    accepted = sum(1 for _, ok in pairs if ok)
    # CROSS-CHECK against the engine's own verdicts. Two implementations of the join that agree is the
    # only form of witness here that is not self-reference.
    mine = sorted((d["adjudicated_by_pred_pos"], d["proposal"], int(ok)) for d, ok in pairs)
    theirs = sorted((a["settled_pos"], a["proposal"], int(a["accept"])) for a in rec["accepts"])
    if mine != theirs:
        only_mine = [x for x in mine if x not in theirs]
        only_theirs = [x for x in theirs if x not in mine]
        die(f"harness join and the engine's MTPACCEPT lines disagree: len {len(mine)} vs"
            f" {len(theirs)}; only-mine {only_mine[:4]}; only-engine {only_theirs[:4]}")
    n_final = sum(1 for d, _ in pairs if d["adjudicated_by_pred_pos"] == final_pos)
    n_final_ok = sum(1 for d, ok in pairs if ok and d["adjudicated_by_pred_pos"] == final_pos)
    return {"pairs": pairs, "adjudicated": len(pairs), "accepted": accepted, "skipped": skipped,
            "settled_by_final_sample": n_final, "final_sample_accepted": n_final_ok}


def depth_table(pairs: list[tuple[dict, bool]], key: str) -> list[dict]:
    rows = []
    for lo, hi in DEPTH_BUCKETS:
        sel = [int(ok) for d, ok in pairs if lo <= d[key] <= hi]
        if sel:
            rows.append({"bucket": f"{lo}-{hi if hi < 10 ** 9 else '+'}", "n": len(sel),
                         "accept": sum(sel) / len(sel)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=32,
                    help="generation budget per line (>= 24 is what the protocol asks for; a line that "
                         "hits the stop id sooner is reported short, never padded)")
    ap.add_argument("--sched", default="loop", choices=("loop", "every"),
                    help="the engine's round schedule; `loop` matches the geometry the carried 0.7329 "
                         "was measured on, `every` drafts after every step")
    ap.add_argument("--binary", default=str(ROOT / "engine" / "build" / "decode_run"))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.steps < 24:
        die("--steps must be >= 24: the protocol prices >= 24 generated tokens per line")

    prompts = load_population()
    end = len(prompts) if args.limit == 0 else min(len(prompts), args.start + args.limit)
    chunk = list(enumerate(prompts[args.start:end], start=args.start))
    if not chunk:
        die(f"empty slice: --start {args.start} --limit {args.limit} of {len(prompts)} lines")

    sess = Session([args.binary, "--mtp-accept", "0", "--mtp-draft", "0", "--model", args.model, "--session",
                    "--mtp-draft", "1", "--mtp-sched", args.sched])
    sess.read_until(lambda l: l.startswith("READY"))

    pairs_all: list[tuple[dict, bool]] = []
    skipped_all: list[dict] = []
    per_line: list[dict] = []
    agree_rows: list[dict] = []
    made = 0
    final_n = 0
    final_ok = 0
    last_census: dict[str, str] = {}
    for li, p in chunk:
        sess.send("reset")
        sess.read_until(lambda l: l.startswith("RESET"))
        sess.send("feed " + " ".join(str(i) for i in p["ids"]))
        sess.read_until(lambda l: l.startswith("FED"))
        sess.send(f"gen {args.steps}")
        lines = sess.read_until(lambda l: l.startswith("END "))
        sess.send("stats")
        lines += sess.read_until(lambda l: l.startswith("MTPDRAFT"))
        rec = parse_records(lines)
        j = join_line(rec)
        pairs_all += j["pairs"]
        skipped_all += j["skipped"]
        final_n += j["settled_by_final_sample"]
        final_ok += j["final_sample_accepted"]
        made += len(rec["drafts"]) + sum(1 for s in rec["skips"]
                                         if s["kind"] == "DRAFT-SKIP")
        last_census = rec["census"]

        # Engine-vs-reference agreement, for context only: how far the engine's own greedy tokens are
        # from the pinned HF continuation on the same prompt (A-18's chaos floor lives here too).
        base = len(p["ids"]) - 1          # GEN prints the PREDICTING position
        ref = p.get("continuation") or []
        compared = sum(1 for i in range(len(ref)) if base + i in rec["pred_at"])
        agree = sum(1 for i in range(len(ref))
                    if base + i in rec["pred_at"] and rec["pred_at"][base + i] == ref[i])
        agree_rows.append({"line": li, "ref_tokens": len(ref), "compared": compared, "agree": agree})
        per_line.append({
            "line": li, "text": p["text"][:48], "prompt_ids": len(p["ids"]),
            "gen_tokens": rec["gen_tokens"], "stop_fired": rec["stop"],
            "reached_budget": rec["gen_tokens"] >= args.steps,
            "drafts_made": len(rec["drafts"]) + sum(1 for s in rec["skips"]
                                                    if s["kind"] == "DRAFT-SKIP"),
            "adjudicated": j["adjudicated"], "accepted": j["accepted"],
            "settled_by_final_sample": j["settled_by_final_sample"],
            "final_sample_accepted": j["final_sample_accepted"],
            "skipped": len(j["skipped"]),
            "engine_vs_reference": f"{agree}/{compared}",
        })
        if not args.quiet:
            print(f"  line {li:>2} prompt={len(p['ids']):>3} gen={rec['gen_tokens']:>3}"
                  f" drafts={len(rec['drafts']) + len(rec['skips']):>3}"
                  f" adj={j['adjudicated']:>3} accept={j['accepted']:>3}"
                  f" skip={len(j['skipped']):>2} engine_bits_vs_ref={agree}/{compared}")
    census_lines = [ln for ln in sess.log if ln.startswith("MTPDRAFT")]
    sess.close()

    adjudicated = len(pairs_all)
    accepted = sum(1 for _, ok in pairs_all if ok)
    rate = accepted / adjudicated if adjudicated else float("nan")

    # GLOBAL CROSS-CHECK: the folded per-line totals must equal the engine's own session census, which is
    # a counter the engine keeps independently of any printed line. A harness that dropped a transcript
    # line, or counted one twice, fails here rather than reporting a plausible lower rate.
    if last_census:
        c = {k: int(last_census[k]) for k in ("rounds", "adjudicated", "accepted", "refused",
                                              "unadjudicated")}
        if (adjudicated, accepted, made) != (c["adjudicated"], c["accepted"],
                                             c["rounds"] + c["refused"]):
            die(f"harness totals disagree with the engine census: harness made/adjudicated/accepted = "
                f"{made}/{adjudicated}/{accepted} vs census "
                f"{c['rounds']}+{c['refused']}/{c['adjudicated']}/{c['accepted']}")
        census_ok = (f"harness {made}/{adjudicated}/{accepted} == census "
                     f"{c['rounds']}+{c['refused']}/{c['adjudicated']}/{c['accepted']}")
    else:
        census_ok = "no census line parsed"

    if made != adjudicated + len(skipped_all):
        die(f"bucket leak: {made} proposals made but {adjudicated} adjudicated +"
            f" {len(skipped_all)} skipped = {adjudicated + len(skipped_all)}. Every proposal must land"
            " in exactly one bucket; a rate printed over a leaking one is wrong by construction.")
    print()
    print(f"=== accept on ENGINE bits — {len(chunk)} corpus lines x {args.steps} steps,"
          f" schedule={args.sched} ===")
    print(f"proposals made        {made}")
    print(f"proposals adjudicated {adjudicated}")
    print(f"proposals skipped     {len(skipped_all)}   (each carries a printed reason; a skip is"
          " never a miss)")
    print(f"accept OVERALL        {accepted}/{adjudicated} = {rate:.4f}"
          f"   [denominator: proposals adjudicated / proposals made = {adjudicated}/{made}]")
    if final_n:
        # One settlement per line comes from the sample the budget path produced but never printed as a
        # token. It is a real target prediction on the same bits, so it belongs in the headline; it is
        # also the one a reader may reasonably want excluded, so both numbers are on the page.
        ex_n, ex_ok = adjudicated - final_n, accepted - final_ok
        print(f"  of which settled by the turn's unprinted final sample: {final_ok}/{final_n}"
              f"; excluding those: {ex_ok}/{ex_n} = "
              f"{(ex_ok / ex_n if ex_n else float('nan')):.4f}")
    print("accept by draft position   " + "  ".join(
        f"{r['bucket']}:{r['accept']:.3f}(n={r['n']})" for r in depth_table(pairs_all, "draft_pos")))
    # `round` is the engine's own per-request round counter, zeroed at each `reset`, so it is the
    # round index WITHIN the line — the depth measure the task asks for beside the absolute position
    # the carried profile is bucketed by.
    print("accept by round in line    " + "  ".join(
        f"{r['bucket']}:{r['accept']:.3f}(n={r['n']})" for r in depth_table(pairs_all, "round")))
    if agree_rows:
        tc = sum(r["compared"] for r in agree_rows)
        ta = sum(r["agree"] for r in agree_rows)
        print(f"engine-vs-reference   {ta}/{tc} tokens agree with the pinned HF continuations"
              " (context for reading `a`, not part of it)")
    if last_census:
        print(f"engine census (last)    rounds={last_census.get('rounds')}"
              f" adjudicated={last_census.get('adjudicated')}"
              f" accepted={last_census.get('accepted')} refused={last_census.get('refused')}"
              f" unadjudicated={last_census.get('unadjudicated')}"
              f" lanes_max={last_census.get('lanes_max')} sched={last_census.get('sched')}")
        print(f"census cross-check    {census_ok}")
    short = [r["line"] for r in per_line if not r["reached_budget"]]
    if short:
        print(f"lines short of {args.steps} tokens: {len(short)} {short[:12]}"
              " (stop id reached sooner; reported, not padded)")
    none = [r["line"] for r in per_line if r["adjudicated"] == 0]
    if none:
        print(f"lines with nothing adjudicated: {len(none)} {[r['line'] for r in none][:12]}")
    if skipped_all:
        reasons: dict[str, int] = {}
        for s in skipped_all:
            reasons[s.get("why", "?")[:60]] = reasons.get(s.get("why", "?")[:60], 0) + 1
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"skipped x{n}: {why}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({
            "population": {"corpus_lines": len(prompts), "driven": len(chunk),
                           "start": args.start, "limit": args.limit,
                           "steps_per_line": args.steps, "schedule": args.sched,
                           "binary": args.binary},
            "proposals_made": made, "proposals_adjudicated": adjudicated,
            "proposals_accepted": accepted, "proposals_skipped": len(skipped_all),
            "accept_overall": rate,
            "denominator": "proposals adjudicated / proposals made",
            "settled_by_unprinted_final_sample": final_n,
            "final_sample_accepted": final_ok,
            "accept_excluding_final_sample": ((accepted - final_ok) / (adjudicated - final_n)
                                              if adjudicated > final_n else None),
            "accept_by_draft_position": depth_table(pairs_all, "draft_pos"),
            "accept_by_round_in_line": depth_table(pairs_all, "round"),
            "per_line": per_line, "engine_vs_reference": agree_rows,
            "engine_census_last": last_census,
            "census_cross_check": census_ok, "session_census_lines": census_lines,
        }, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
