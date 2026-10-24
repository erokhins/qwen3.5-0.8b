#!/usr/bin/env python3
"""M3 — the accept rate of the SHIPPING LOOP, adjudicated on the corpus-49 population.

WHY THIS EXISTS AND `run_accept.py` DOES NOT COVER IT. `run_accept.py` measures the accept rate of the
read-only SIDE PASS (`--mtp-draft`, which prices a proposal and settles it against the next ordinary
step). `--mtp-accept` is a different object: a round consumes one or two positions, drafts from the last
FED row of the round that consumed the previous proposal, and owns the token stream. The kill line
(`ACCEPT-PLAN.md`: kill if measured `a` on the shipping path falls below 0.4568) is stated about THE
SHIPPING PATH, so it has to be adjudicated on this loop, not on the side pass whose 0.6641 was carried.
`ACCEPT-BUILD.md` §4i.3 hands over exactly this measurement, and notes the geometry difference that makes
exact agreement with 0.6641 not owed.

THE POPULATION IS DERIVED, NEVER TYPED. It comes from `run_accept.load_population()`, which cross-checks
`speccheck/fixtures/corpus.txt` against `data/refs/corpus48/manifest.json` (same count, same texts). The
AGENTS.md trap row about an admission over a list typed from memory is the reason this file imports that
function instead of reimplementing it: two populations that drift apart produce a plausible rate on
neither corpus.

THREE INDEPENDENT COUNTS OF THE SAME RUN, per line. `MTPSTAT`'s fields come from the device verdicts,
`DECWALL`'s from the host emit sites, `emitted.txt` from the file, and the `ACCEPT round=` lines from the
transcript. The harness re-derives every relation the engine asserts (rounds == accepted + rejected,
DECWALL.emitted == MTPSTAT.emitted + plain - suppressed, DECWALL.rounds == MTPSTAT.rounds, emitted.txt
lines == DECWALL.emitted, ACCEPT lines == MTPSTAT.rounds) because a check that lives only inside the thing
under test is blind to the one failure that matters — the same reason `run_accept.sh` refuses a green
gate whose `MTPDRAFT` line is missing.

LOSSLESSNESS IS RE-ADJUDICATED ON EVERY LINE, not just on `dec8`, and it is adjudicated as a DECISION
QUESTION rather than as byte equality. `ACCEPT-BUILD.md` §4g names the failure mode a wrong candidate
would produce: the stream moves and nothing else does. So every line also runs flag-off and the two streams
are compared. What was NOT known in advance is that they will not always be equal: the accept arm prices
row 0 in the 2-row wave body and the flag-off arm prices it in the 1-row step, and `ACCEPT-PLAN.md` already
records that those two forms are not bit-identical (argmax SAME, sub-band magnitudes) — which is A-18's
chaos floor wearing a different hat. Byte equality at depth was therefore never an admissible ask, and
"DIFFERS" at depth cannot tell a re-associated summation from dropped work.

So the discriminator is mechanical, and it is the flag-off run's OWN logits at the flip position:

  a flip is a legal tie-break iff the token the accept arm emitted is the flag-off run's RUNNER-UP, and
  the top1-top2 margin at that row is below ONE BF16 ULP of the winning logit — i.e. the two candidates
  are closer than the format they were computed in can express.

Anything else — a token from further down the ranking, or a margin a legal form difference cannot reach —
is a bookkeeping bug and exits non-zero, which by the standing fix-first rule outranks every speed item.
The criterion is computed from the format (bf16 mantissa 8 bits -> relative 2^-8), NOT from the observed
flips; the raw margins, ranks and each line's own margin distribution are printed regardless, so any
reader can re-adjudicate with a different threshold. Stated plainly because it happened: the first flip
(corpus line 3) was inspected BEFORE this criterion was written down, and it passed it (margin 0.0311 vs
ULP bound 0.0853, token = runner-up, tightest tie of its line's 32 steps).

THE DENOMINATOR IS PART OF THE VERDICT. The carried 0.6641 adjudicated 926 proposals; a "rate" from a
handful of rounds cannot trip or clear a kill line drawn at four digits, so below `--min-rounds` (default
400) this prints INSUFFICIENT-SAMPLE instead of a verdict. A rate of exactly 0.0000 is reported as a bug,
not a low rate: a loop that never accepts is a broken loop.

GPU: this does NOT take the lock. Run it under `m3_loop_rate.sh`, which acquires and releases
`tools/gpu_lock.sh` in one long-lived shell (a short-lived subprocess would record a dead holder pid, and
`gpu_lock.sh` deliberately reclaims a dead holder's lock).

USAGE
    prototype/mtp_accept/m3_loop_rate.sh                       # all 49 lines, both arms
    prototype/mtp_accept/m3_loop_rate.sh --limit 8 --steps 32  # a slice, for a short window
    PROXY of the same for a manual run: see the printed command line per line.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_accept import DEPTH_BUCKETS, die, load_population  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The kill line and the carried figure, imported by value from the documents that own them rather than
# restated per run. Both live in ACCEPT-PLAN.md / ACCEPT-RESULT.md; changing one here would be moving a
# band in a script that is supposed to be judged by it.
KILL_LINE = 0.4568
CARRIED_SIDE_PASS = 0.6641
CARRIED_SIDE_PASS_FRAC = "615/926"

# `key=value` pairs, values being any non-space run: `census=ok` and `ended=context_exhausted` are fields
# too, and a numeric-only pattern silently dropped them, which read as "the engine refused to bill the run"
# when in fact the harness had failed to parse a word.
KV = re.compile(r"(\w+)=([^\s]+)")
VOCAB = 248320          # asserted against the dump's byte size, never assumed
BF16_REL_ULP = 2.0 ** -8  # bf16 keeps 8 mantissa bits: one ULP of x is x * 2^-8
# `ACCEPT round=.. anchor=.. fed=.. cand=.. cand_pos=.. accept=.. verifier=.. bonus=.. emitted=.. pos=A->B
#  restored_bytes=..`
ROUND_RE = re.compile(
    r"^ACCEPT round=(\d+) anchor=\d+ fed=\d+ cand=(-?\d+) cand_pos=(-?\d+) accept=(\d)"
    r" verifier=(-?\d+) bonus=(-?\d+) emitted=(\d+) pos=\d+->(\d+) restored_bytes=(\d+)")


def fields(line: str) -> dict[str, str]:
    return {k: v for k, v in KV.findall(line)}


def run_binary(binary: str, model: str, ids_file: pathlib.Path, out: pathlib.Path,
               extra: list[str], gdn_loop: str = "") -> tuple[int, str]:
    out.mkdir(parents=True, exist_ok=True)
    # `--gdn-loop` is passed ONLY when the caller named one. Empty means "the engine default", which since
    # 2026-10-25 is what the SHIPPING loop runs: the accept round's own bodies are forced to a one-token-step
    # span by `submit_wave_body` (their mid-round checkpoint is a per-token recurrence's end state) and every
    # other body — the prefill wave, the plain steps — keeps `--gdn-loop 512`. This file's own subject is the
    # shipping loop, so inheriting the default is the honest shape and pinning 1 here would be pricing a
    # configuration the engine no longer runs. `--gdn-loop 1` reproduces the whole-run-span record every
    # carried number in `M3-RESULT.md` was taken with, and it goes on BOTH arms from one place below, so the
    # pair can never differ in it (`AGENTS.md`: a required flag belongs on both arms or the pair prices the
    # flag).
    cmd = [binary, "--mtp-accept", "0", "--mtp-draft", "0", "--model", model,
           "--ids-file", str(ids_file)]
    if gdn_loop:
        cmd += ["--gdn-loop", gdn_loop]
    cmd += ["--no-stop", "--out", str(out)] + extra
    # `--mtp-accept 0 --mtp-draft 0` here, on the SHAPE of every run this driver makes, not on one arm:
    # the accept arm names `--mtp-accept 1` in `extra` and the last name wins, while the flag-off pair run
    # gets the plain path explicitly instead of by default. The engine now defaults speculation ON, so a
    # pair run that inherits the default would write `emitted.txt` and no `logits<N>.bin`, and the lossless
    # comparison below would be reading a different object than it thinks it is.
    # The command line is printed so any single line of this pass is re-runnable by hand, and so the
    # device population is auditable from the log rather than from this script's argparse defaults.
    print("    $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def logits_row(d: pathlib.Path, step: int) -> np.ndarray:
    """One flag-off logits row, dtype read off the file size, not assumed.

    The engine's `--out` rows are F32 (AGENTS.md: read a dump's dtype off its byte size — read as BF16 they
    give half the elements and NaN aggregates with a plausible `ndiff` attached). Size is asserted against
    the vocab, so a truncated dump is an error and not a short row.
    """
    p = d / f"logits{step}.bin"
    if not p.exists():
        die(f"no {p} — the flag-off run did not dump the row the comparison needs")
    n = p.stat().st_size
    if n == 4 * VOCAB:
        a = np.fromfile(p, dtype=np.float32)
    elif n == 2 * VOCAB:
        raise SystemExit(f"{p}: BF16-sized logits ({n} B); this probe knows how to read F32 only")
    else:
        raise SystemExit(f"{p}: {n} B is neither {4 * VOCAB} (F32) nor {2 * VOCAB} (BF16) for vocab "
                         f"{VOCAB}; refusing to guess a dtype")
    assert a.size == VOCAB, (p, a.size)
    return a


def flip_witness(base_dir: pathlib.Path, base_steps: list[int], flip: int,
                 acc_token: str, greedy_token: str) -> dict:
    """Adjudicate ONE stream flip against the flag-off run's own logits at that position.

    Returns the margin, the rank of the token the accept arm emitted, the line's own margin distribution,
    and the verdict. See the module docstring for why runner-up + sub-ULP margin is the criterion.
    """
    step = base_steps[flip]
    r = logits_row(base_dir, step)
    order = np.argsort(r)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    margin = float(r[top1] - r[top2])
    try:
        rank = int(np.where(order == int(acc_token))[0][0])
    except IndexError:
        rank = -1
    ms = []
    for s in base_steps:
        x = logits_row(base_dir, s)
        part = np.partition(x, x.size - 2)[-2:]
        ms.append(float(part.max() - part.min()))
    ulp = abs(float(r[top1])) * BF16_REL_ULP
    ok = rank == 1 and margin < ulp
    return {"flip": flip, "step": step, "greedy": greedy_token, "accept": acc_token,
            "top1": top1, "top2": top2, "margin": margin, "ulp_bound": ulp, "rank": rank,
            "line_median_margin": float(np.median(ms)), "line_min_margin": float(min(ms)),
            "line_steps": len(ms), "near_tie": bool(ok)}


def check_line(tag: str, acc_log: str, acc_dir: pathlib.Path, base_dir: pathlib.Path | None,
               base_log: str | None, want_ids: int) -> dict:
    """Re-derive every relation from ONE accept run, plus its lossless pair. Dies on a broken one."""
    decwall = [ln for ln in acc_log.splitlines() if ln.startswith("DECWALL")]
    mtpstat = [ln for ln in acc_log.splitlines() if ln.startswith("MTPSTAT")]
    rounds_lines = [ln for ln in acc_log.splitlines() if ROUND_RE.match(ln)]
    if len(decwall) != 1:
        die(f"{tag}: {len(decwall)} DECWALL lines, expected exactly 1 "
            "(a run that printed none never reached the window)")
    if len(mtpstat) != 1:
        die(f"{tag}: {len(mtpstat)} MTPSTAT lines, expected exactly 1 — no accept tally means the "
            "flag did not reach the engine, so nothing below this line is a rate")
    d, m = fields(decwall[0]), fields(mtpstat[0])
    for k in ("bodies", "rounds", "plain", "accepted", "rejected", "suppressed", "emitted"):
        if k not in d:
            die(f"{tag}: DECWALL has no `{k}` field: {decwall[0]}")
    if "note" in d:
        die(f"{tag}: DECWALL note={d['note']} NO-RATE — the engine refused to bill this run")
    if d.get("census") != "ok":
        die(f"{tag}: DECWALL census={d.get('census', 'MISSING')} — the engine itself refused to bill "
            "this run; a rate from it would be a number the binary told us not to read")

    # The checks. Each names the two sources that disagree, because "mismatch" without provenance is
    # the reader doing the forensics the harness should have done.
    def eq(got, want, what):
        if int(got) != int(want):
            die(f"{tag}: {what}: {got} != {want}")

    eq(int(m["rounds"]), int(m["accepted"]) + int(m["rejected"]), "MTPSTAT rounds vs accepted+rejected")
    # The emit identity, in the shape the geometry supports. At k=1 (`--mtp-k`, default) an accept emits 2
    # and a reject emits 1, which is the relation this check used to state unconditionally. At k=2 a round
    # emits n_accepted + 1, so the relation the engine asserts is `emitted == candidates_accepted +
    # rounds`; MTPSTAT prints that field only when k>1, and the branch is on the FIELD'S PRESENCE rather
    # than on a knob this harness did not set — a k=1 run cannot accidentally be judged by the k=2 rule.
    if "candidates_accepted" in m:
        eq(int(m["emitted"]), int(m["candidates_accepted"]) + int(m["rounds"]),
           "MTPSTAT emitted vs candidates_accepted+rounds")
        eq(int(m["depth1"]) >= int(m["accepted"]), True,
           "MTPSTAT depth1 vs accepted: a depth-2 accept is also a depth-1 accept, so a depth1 below the"
           " full-prefix count means a candidate behind a rejected one was accepted")
    else:
        eq(int(m["emitted"]), 2 * int(m["accepted"]) + int(m["rejected"]),
           "MTPSTAT emitted vs 2*accepted+rejected")
    eq(int(d["rounds"]), int(m["rounds"]), "DECWALL rounds vs MTPSTAT rounds")
    eq(int(d["accepted"]), int(m["accepted"]), "DECWALL accepted vs MTPSTAT accepted")
    eq(int(d["rejected"]), int(m["rejected"]), "DECWALL rejected vs MTPSTAT rejected")
    eq(int(d["emitted"]),
       int(m["emitted"]) + int(d["plain"]) - int(d["suppressed"]),
       "DECWALL emitted (host emit sites) vs MTPSTAT tally + plain steps - suppressed bonus")
    eq(int(d["bodies"]), int(m["rounds"]) + int(d["plain"]), "DECWALL bodies vs rounds + plain")
    eq(len(rounds_lines), int(m["rounds"]), "ACCEPT round lines vs MTPSTAT rounds")
    n_accept_lines = sum(1 for ln in rounds_lines if ROUND_RE.match(ln).group(4) == "1")
    # `accept=` on the ACCEPT line is the FIRST candidate's equality: at k=1 that is the whole accept bit,
    # and at k=2 it is the depth-1 hit, which MTPSTAT prints as `depth1`. Comparing the line count against
    # whichever field the geometry defines keeps one rule per independent variable (§18's attribution point).
    eq(n_accept_lines, int(m.get("depth1", m["accepted"])),
       "ACCEPT accept=1 lines vs MTPSTAT " + ("depth1" if "depth1" in m else "accepted"))

    emitted_file = [ln.strip() for ln in (acc_dir / "emitted.txt").read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")]
    eq(len(emitted_file), int(d["emitted"]), "emitted.txt lines vs DECWALL emitted")

    # Losslessness, on this line: the accept arm's stream against the flag-off stream. The reference comes
    # from the flag-off run's `STEP kind=decode` lines, NOT from its `sampled.txt`: that file also carries
    # the prefill rows, and with the default `--prefill-logits last` those print `sampled=-1`, so a reader
    # of column 3 compares the prompt against the generation and finds every position "different".
    # `base_log is None` means the caller skipped the pair run (`--lossless 0`); that is recorded as None
    # below, never as a passing comparison.
    if base_log is None:
        return {"d": d, "m": m, "rounds_lines": rounds_lines, "lossless_common": None,
                "accept_len": len(emitted_file), "greedy_len": None, "prompt_ids": want_ids,
                "identical": None, "flip": None}
    base = [fields(ln)["sampled"] for ln in base_log.splitlines()
            if ln.startswith("STEP kind=decode")]
    base_steps = [int(fields(ln)["index"]) for ln in base_log.splitlines()
                  if ln.startswith("STEP kind=decode")]
    if not base:
        die(f"{tag}: the flag-off run printed no `STEP kind=decode` line, so the lossless comparison has "
            "no reference")
    n = min(len(emitted_file), len(base))
    bad = [i for i in range(n) if emitted_file[i] != base[i]]
    # WHO emitted each token. An `ACCEMIT step=` whose step matches an `ACCEPT round=` came out of the
    # 2-row body (row 0 on a reject, the candidate or row 1 on an accept); any other `ACCEMIT` came out of
    # a plain 1-row `decode_step` — the same call the flag-off arm makes. That splits the two competing
    # explanations for a flip: the body's arithmetic is a different legal form (row 0 of a wave at width 2
    # vs the fused 1-row step), versus the loop's bookkeeping corrupting state that a plain step then
    # reads. A flip on a plain step is the second one, and it is not explicable by the form gap.
    emit_steps = [int(fields(ln)["step"]) for ln in acc_log.splitlines()
                  if ln.startswith("ACCEMIT ")]
    round_steps = {int(ROUND_RE.match(ln).group(1)): ROUND_RE.match(ln).group(4)
                   for ln in rounds_lines}
    flip = None
    if bad:
        # Only the FIRST flip is adjudicable: after a different token the two continuations are different
        # texts, so every later position differs by construction and says nothing.
        f = bad[0]
        flip = flip_witness(base_dir, base_steps, f, emitted_file[f], base[f])
        st = emit_steps[f] if f < len(emit_steps) else None
        flip["emitted_by"] = ("plain_step(1-row)" if st not in round_steps
                              else ("round_accept" if round_steps.get(st) == "1" else "round_reject"))
    return {"d": d, "m": m, "rounds_lines": rounds_lines,
            "lossless_common": n, "accept_len": len(emitted_file), "greedy_len": len(base),
            "prompt_ids": want_ids, "identical": not bad, "flip": flip,
            "n_div": len(bad)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="engine/build/decode_run")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = to the end of the population")
    ap.add_argument("--lossless", type=int, default=1, choices=(0, 1))
    ap.add_argument("--extra", default="",
                    help="extra accept-arm flags, e.g. '--mtp-dense 1'. Appended to the ACCEPT arm only, "
                         "never to the flag-off pair run — that run is the greedy reference the stream is "
                         "adjudicated against, and a draft-side flag must not reach it. Used by "
                         "prototype/mtp_accept/draft_cond_probe.py's follow-up adjudication of a "
                         "draft-conditioning knob on the corpus population.")
    ap.add_argument("--gdn-loop", default="",
                    help="passed to the engine on BOTH arms; '' (the default) passes nothing, i.e. the "
                         "engine default — the span the SHIPPING loop runs, since the round's own bodies "
                         "are forced to 1 by the engine and everything else keeps 512. Pass '1' to "
                         "re-take the whole-run-span record the carried M3-RESULT.md figures were taken "
                         "with; a value here moves both arms together, never one.")
    ap.add_argument("--min-rounds", type=int, default=400)
    ap.add_argument("--out", default="/tmp/m3_loop_rate")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    prompts = load_population()
    end = len(prompts) if args.limit == 0 else min(len(prompts), args.start + args.limit)
    chunk = list(enumerate(prompts[args.start:end], start=args.start))
    if not chunk:
        die(f"empty slice: --start {args.start} --limit {args.limit} of {len(prompts)} lines")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== the SHIPPING LOOP's accept rate — {len(chunk)} corpus lines x {args.steps} steps "
          f"(rounds), lossless={args.lossless} ===")
    print(f"population: {len(prompts)} lines from speccheck/fixtures/corpus.txt, cross-checked against "
          f"data/refs/corpus48/manifest.json; driving lines {args.start}..{end - 1}")
    print(f"accept-arm extra flags: {args.extra if args.extra else '(none — the shipping loop)'}"
          f"  [appended to the accept arm only, never to the flag-off pair run]")
    # Named on the record because BOTH arms ran at it and the carried figures were taken at the other one.
    # The round's own bodies are at 1 either way; what this moves is the prefill wave and the plain steps.
    print(f"gdn-loop on both arms: {args.gdn_loop if args.gdn_loop else '(engine default)'}"
          "  [engine/gate.py section (m) asserts the engine's `MTP mode=` echo of the two spans]")

    tot = {"rounds": 0, "accepted": 0, "rejected": 0, "emitted": 0, "plain": 0, "suppressed": 0,
           "bodies": 0, "depth1": 0, "candidates": 0, "stat_emitted": 0}
    by_depth: list[tuple[int, bool]] = []
    per_line = []
    flips: list[dict] = []
    for li, p in chunk:
        ids_file = out / f"line{li:02d}.csv"
        ids_file.write_text(",".join(str(i) for i in p["ids"]) + "\n")
        # `--ids-file` splits on COMMAS, so a space-joined file reads as one token and the run then
        # "succeeds" at width 1 (AGENTS.md). Assert the field count off the file, not off the intent.
        got = len(ids_file.read_text().strip().split(","))
        if got != len(p["ids"]):
            die(f"line {li}: wrote {len(p['ids'])} ids and the file has {got} comma fields")
        tag = f"line {li}"
        ad, bd = out / f"acc{li:02d}", out / f"base{li:02d}"
        rc, log = run_binary(args.binary, args.model, ids_file, ad,
                             ["--steps", str(args.steps), "--mtp-draft", "1", "--mtp-accept", "1"]
                             + args.extra.split(), args.gdn_loop)
        if rc != 0:
            die(f"{tag}: accept run rc={rc}\n" + "\n".join(log.splitlines()[-6:]))
        if args.lossless:
            rc, blog = run_binary(args.binary, args.model, ids_file, bd,
                                  ["--steps", str(args.steps)], args.gdn_loop)
            if rc != 0:
                die(f"{tag}: flag-off pair run rc={rc}\n" + "\n".join(blog.splitlines()[-6:]))
        else:
            blog = None
        info = check_line(tag, log, ad, bd if args.lossless else None, blog, got)
        d, m = info["d"], info["m"]
        for k in ("rounds", "accepted", "rejected", "emitted", "plain", "suppressed", "bodies"):
            tot[k] += int(d[k])           # the run-window counts, off DECWALL (what the RUN did)
        for k in ("depth1", "candidates"):
            # Round-tally counts that exist only at k=2, off MTPSTAT (what the ROUNDS did). A k=1 run has
            # neither field and contributes 0, which is what keeps one accumulator shape for both geometries
            # without inventing a k=1 value the engine never printed.
            tot[k] += int(m.get(k, 0))
        # MTPSTAT's own emit tally, which counts ONLY what rounds emitted. `tot["emitted"]` is DECWALL's,
        # which also carries one token per plain step, so dividing it by `rounds` mixes the two
        # denominators — visible as a `1+n1+n2` figure that refused to match the engine's
        # `mean_tokens_per_round` on a run with plain steps in it.
        tot["stat_emitted"] += int(m["emitted"])
        for idx, ln in enumerate(info["rounds_lines"]):
            mt = ROUND_RE.match(ln)
            by_depth.append((idx, mt.group(4) == "1"))
        fl = info["flip"]
        if fl:
            flips.append({"line": li, **fl})
        per_line.append({"line": li, "prompt": got, "rounds": int(m["rounds"]),
                         "accepted": int(m["accepted"]), "emitted": int(d["emitted"]),
                         "plain": int(d["plain"]), "ended": d.get("ended", "?"),
                         "rate": int(m["accepted"]) / int(m["rounds"]) if int(m["rounds"]) else None,
                         "lossless_common": info["lossless_common"] if args.lossless else None,
                         "identical": info["identical"], "first_flip": (fl or {}).get("flip"),
                         "near_tie": (fl or {}).get("near_tie")})
        status = ""
        if args.lossless:
            if info["identical"]:
                status = f" stream=identical({info['lossless_common']}tok)"
            elif fl:
                status = (f" stream=FLIP@{fl['flip']} margin={fl['margin']:.4f}"
                          f"<ulp{fl['ulp_bound']:.4f} rank={fl['rank']}"
                          f" {'NEAR-TIE' if fl['near_tie'] else 'NOT-NEAR-TIE'}")
            else:
                status = " stream=DIFFERS"
        print(f"  {tag:>8} prompt={got:>3} rounds={m['rounds']:>3} accepted={m['accepted']:>3}"
              f" plain={d['plain']:>2} emitted={d['emitted']:>3} rate={per_line[-1]['rate']:.4f}"
              f" ended={d.get('ended', '?')}{status}", flush=True)

    rounds, accepted = tot["rounds"], tot["accepted"]
    a = accepted / rounds if rounds else 0.0
    print(f"\n--- harness wall {time.time() - t0:.0f}s (its own minutes, not an engine timing) ---")

    # The stream admission, printed BEFORE the rate because it gates it: a rate from a loop that moved the
    # target's stream is not a number (`run_accept.sh`'s NO-NUMBER rule), and the only way to know that is
    # to compare the streams, which is what every line above did.
    if args.lossless:
        n_id = sum(1 for r in per_line if r["identical"])
        hard = [f for f in flips if not f["near_tie"]]
        print("=== ADMISSION: the accept arm's stream against the flag-off greedy stream ===")
        print(f"lines compared {len(per_line)}   identical {n_id}   flipped {len(flips)}"
              f"   of which outside the near-tie band {len(hard)}")
        for f in flips:
            print(f"  line {f['line']:>2}: first flip at emitted index {f['flip']:>2} (step {f['step']:>3}):"
                  f" greedy={f['greedy']} accept={f['accept']} (rank {f['rank']} of that row)"
                  f"  margin {f['margin']:.4f} vs bf16-ULP bound {f['ulp_bound']:.4f}"
                  f"  | line median {f['line_median_margin']:.3f}, min {f['line_min_margin']:.4f},"
                  f" {f['line_steps']} steps  emitted_by={f.get('emitted_by', '?')}  -> "
                  + ("NEAR-TIE: a legal form gap" if f["near_tie"] else "NOT A NEAR-TIE: STRUCTURAL"))
        if hard:
            print(f"ADMISSION: FAIL — {len(hard)} flip(s) are not explicable as a tie between two legal"
                  " bf16 forms. Per the fix-first rule this outranks every pending speed item.")
            print("NO-NUMBER: the accept rate above this line is not printed, because the loop moved the"
                  " target's stream in a way the form gap cannot reach.")
            if args.json:
                pathlib.Path(args.json).write_text(json.dumps(
                    {"admission": "FAIL", "flips": flips, "per_line": per_line}, indent=2) + "\n")
            return 1
        print("ADMISSION: PASS — every divergence is the flag-off row's own runner-up at a margin below"
              " one bf16 ULP of that row's top logit, i.e. the two candidates were not distinguishable in"
              " the format they were computed in. That is the 2-row-body-vs-1-row-step form gap ACCEPT-PLAN"
              " already carries, arriving as a token flip at depth rather than as a ULP count; byte"
              " equality of the two streams at depth was never an admissible ask (§4i proved it to 16 tokens"
              " on `dec8`, and A-18 says no bf16 implementation keeps token equality past a few dozen).")

    print("=== VERDICT: the shipping loop's accept rate ===")
    print(f"proposals adjudicated (rounds)   {rounds}")
    print(f"accepted                         {accepted}   rejected {tot['rejected']}")
    print(f"tokens emitted                   {tot['emitted']}   plain steps {tot['plain']},"
          f" suppressed bonus {tot['suppressed']}")
    print(f"mean tokens per round            {tot['emitted'] / rounds:.4f}" if rounds else "n/a")
    print(f"a = accepted/rounds              {accepted}/{rounds} = {a:.4f}")
    # The k=2 reading of the same tally. The engine's own emit identity is `emitted == candidates +
    # rounds`, and `candidates == depth1 + accepted` at k=2 (a depth-2 accept is also a depth-1 one), so
    # tokens per round is `1 + n1 + n2` — the same decomposition
    # `prototype/mtp_accept/DRAFTK-RESULT.md` priced the ladder with, which is what makes the two numbers
    # comparable instead of merely adjacent. Printed only when the geometry produced the fields; a k=1 run
    # prints the block above and nothing here.
    if tot["depth1"] or tot["candidates"]:
        n1 = tot["depth1"] / rounds if rounds else 0.0
        n2 = accepted / rounds if rounds else 0.0
        pred = 1.0 + n1 + n2
        got = tot["stat_emitted"] / rounds if rounds else 0.0
        print(f"k=2 prefix ladder               n1 (cand1 accepted) = {tot['depth1']}/{rounds} = {n1:.4f}"
              f"   n2 (cand2 as well) = {accepted}/{rounds} = {n2:.4f}")
        print(f"tokens per round 1+n1+n2      {pred:.4f}   MTPSTAT mean_tokens_per_round {got:.4f}"
              f"   {'consistent' if abs(pred - got) < 1e-9 else 'INCONSISTENT with the engine tally'}"
              f"   [both over ROUNDS only; the plain steps' {tot['plain']} tokens are in"
              f" `tokens emitted` above and not here]")
    print(f"carried side-pass figure          {CARRIED_SIDE_PASS} ({CARRIED_SIDE_PASS_FRAC})"
          f"  — geometry differs (§4i.3), agreement is expected and not owed")
    # WHICH QUANTITY THE KILL LINE JUDGES. `KILL_LINE` was pre-registered in ACCEPT-PLAN against the rate of
    # the FIRST proposal. At k=1 that is `a`. At k=2 the first proposal's rate is the depth-1 hit `n1`, and
    # `a` is the whole-prefix rate, a strictly smaller quantity — a round that stops after candidate 1 still
    # emits two tokens — so judging `a` at k=2 against a line drawn on the first proposal points the ruler at
    # a different object than the bar names (the same error class as the `--scope form` row: a comparison
    # whose independent variable is not what the bar is about). The quantity judged is printed, always.
    judged = a
    judged_name = "a = accept/rounds"
    if tot["depth1"] or tot["candidates"]:
        judged = tot["depth1"] / rounds if rounds else 0.0
        judged_name = (f"n1 = depth-1 (first candidate) accept/rounds = {tot['depth1']}/{rounds}, which"
                       " is the quantity ACCEPT-PLAN's line was drawn on")
    print(f"kill line                         a < {KILL_LINE}   judged as: {judged_name}")
    if rounds == 0:
        print("VERDICT: NO-DENOMINATOR — nothing was adjudicated; this run is an error, not a result")
    elif rounds < args.min_rounds:
        print(f"VERDICT: INSUFFICIENT-SAMPLE — {rounds} rounds is below the {args.min_rounds}-round floor"
              f" for adjudicating a line drawn on {CARRIED_SIDE_PASS_FRAC}; run the full population")
    elif judged == 0.0:
        print("VERDICT: BROKEN-LOOP — the judged rate is 0 exactly, i.e. a loop that never accepts, which"
              " is a bug report, not a measured rate")
    elif judged < KILL_LINE:
        print(f"VERDICT: KILL — the loop measures {judged:.4f} on the judged quantity against the"
              f" pre-registered kill line {KILL_LINE}. Per ACCEPT-PLAN.md this is not movable by the"
              " implementation.")
    else:
        print(f"VERDICT: the loop's measured {judged:.4f} does not trip the kill line {KILL_LINE}"
              f" (margin {judged - KILL_LINE:+.4f}); the rate work may proceed to timing.")
    if tot["depth1"] or tot["candidates"]:
        # The k=2 bars are not rate bars: the funded question is what the 3-row round COSTS against the extra
        # token it buys, and only a timing pass answers that. Stating it here is what stops a reader from
        # taking the line above as the k=2 verdict (`prototype/mtp_accept/K2-RESULT.md` carries both bars).
        print("NOTE(k=2): the bar on this geometry is the round COST — adopt at tok/s >= +13.6 % over base,"
              " kill at round/step >= 1.95 — and neither is adjudicable from an accept rate. No millisecond"
              " figure appears in this output: the timing pass is the main agent's, run blocking under one"
              " GPU-lock hold.")

    print("\naccept by round ordinal in run (same buckets the carried profile used; the loop's ordinal"
          " counts ROUNDS, the side pass counted proposals):")
    for lo, hi in DEPTH_BUCKETS:
        sel = [ok for (i, ok) in by_depth if lo <= i <= hi]
        if sel:
            print(f"  {lo:>3}-{hi if hi < 10**8 else '+':<3} n={len(sel):>4}"
                  f" accept={sum(sel) / len(sel):.3f}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({
            "binary": args.binary, "steps": args.steps, "lines": len(chunk),
            # Which recurrence span BOTH arms ran at, spelled the way `draftk_ladder.py` spells it, so a
            # reader can tell the shipping-default re-take from the carried whole-run-span figure. The
            # round's bodies were at 1 in either case; this names the prefill and plain-step span.
            "gdn_loop": args.gdn_loop or "engine-default",
            "population": len(prompts), "totals": tot, "a": a, "kill_line": KILL_LINE,
            "carried_side_pass": CARRIED_SIDE_PASS, "per_line": per_line, "flips": flips,
            "bf16_rel_ulp": BF16_REL_ULP, "identical_lines": n_id if args.lossless else None,
        }, indent=2) + "\n")
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
