#!/usr/bin/env python3
"""D-4: put the ENGINE free-running over the corpus prompt set and judge it against the reference.

`tools/corpus_dump.py` pins the reference's own greedy continuations for 49 plain-text prompts
x 24 tokens (`data/refs/corpus48`), and `data/refs/corpus48sdpa` is the same set through a second
legal rounding of the same math — the chaos floor A-18 quotes (0.8776). What the repo did not have is
the engine's side of that comparison: the two free-run fixtures it does have (`dec64t`, `dec300f`)
are prompts whose next token is near-certain, so their 64/64 and 128/128 say nothing about divergence
DEPTH. This is the run that measures it.

Two modes, and only one of them touches the device:

  drive (default)  one `decode_run --session` process; for each prompt: reset, feed the corpus
                   prompt ids, `gen <steps>`, record what came out. Writes an engine manifest in
                   `tools/corpus_dump.py`'s schema, so `tools/corpus_agreement.py corpus48 <out>`
                   answers the mean-agreement question with an instrument that already exists.
  --report-only    no device, no engine. Reads the three manifests + the margin file and prints the
                   per-prompt depth table and the aggregates (mean/median first divergence, the
                   distribution, and agreement conditioned on the reference's decision margin).

Why the session protocol rather than one `decode_run` per prompt: the load is 1.55 GB of weights per
process, so 49 processes would spend most of their wall on H2D and each one is a separate window to
lose. One resident process, one load, `reset` between prompts (08.4 fresh request), which is also
what makes a chunked run (`--start`/`--limit`) cheap.

Indexing, because the wrong index reads as total divergence (see 10.5 and `engine/results.md`): the
session's first `gen` token is the sample already pending from the last fed prompt token, so
`gen[i]` pairs with `greedy_continuation[i]`. Run mode's `kind=decode` step s pairs with s+1. This
tool uses the session, so the pairing is the direct one.

NO TIMINGS. Nothing here prints a duration, and none should be added: a worker-run device number is
contaminated by the session that is sitting on the same GPU (`AGENTS.md`), and this instrument is a
token-agreement instrument. Token ids only.

GPU protocol: like every other `decode_run` caller, this tool does NOT take `tools/gpu_lock.sh`
itself (the self-locking drivers are `prototype/{glue,proj_gemv,gdn_gated_norm}/run.py`; holding the
lock and then invoking one of those deadlocks it). The caller holds the lock:

    tools/gpu_lock.sh acquire "corpus free run"
    .venv/bin/python tools/corpus_freerun.py --start 0 --limit 25
    tools/gpu_lock.sh release "corpus free run"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def manifest_path(tag_or_path: str) -> Path:
    """`corpus48`, `data/refs/corpus48`, or a direct path to a manifest.json."""
    q = resolve(tag_or_path)
    if q.is_dir():
        return q / "manifest.json"
    if q.suffix == ".json":
        return q
    return ROOT / "data" / "refs" / tag_or_path / "manifest.json"


# ---- the device half: drive one resident decode_run session -------------------------------

def drive(args) -> Path:
    ref = json.loads(manifest_path(args.corpus).read_text())
    prompts = ref["prompts"]
    if args.start or args.limit:
        prompts = prompts[args.start: args.start + args.limit if args.limit else None]
    steps = args.steps or ref["steps"]

    binary = resolve(args.binary)
    if not binary.exists():
        raise SystemExit(f"corpus_freerun: engine binary {binary} missing — run engine/build.sh")
    proc = subprocess.Popen(
        [str(binary), "--model", str(resolve(args.model)), "--session"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        return drive_prompts(args, ref, prompts, steps, proc)
    finally:
        # A SystemExit from inside the loop would otherwise leave a process holding 1.55 GB of device
        # memory and the GPU (`AGENTS.md`: a wedged box is what a leaked device process looks like).
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


def drive_prompts(args, ref, prompts, steps, proc) -> Path:
    """The session exchange itself, split out so `drive` can own the process lifetime."""

    def ask(line: str, expect: str) -> list[str]:
        """Write one command, collect its reply lines up to the terminal marker `expect`."""
        assert proc.stdin and proc.stdout
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        out = []
        while True:
            got = proc.stdout.readline()
            if not got:
                raise SystemExit("corpus_freerun: decode_run died. Last replies: "
                                 + " / ".join(out) + f" (while sending: {line[:60]}...)")
            out.append(got.strip())
            if out[-1].startswith(expect) or out[-1].startswith("ERROR"):
                return out

    # `--session` prints its load line and the §00/§01 checkpoint-check lines BEFORE `READY`, so the
    # preamble is skipped rather than assumed to be one line. Bounded: a binary that never says READY
    # is a broken binary, and hanging here would hold the GPU lock for nothing.
    for _ in range(200):
        ready = proc.stdout.readline()
        if not ready:
            raise SystemExit("corpus_freerun: decode_run closed stdin without saying READY")
        if ready.startswith("READY"):
            break
        if ready.startswith("ERROR"):
            raise SystemExit("corpus_freerun: " + ready.strip())
    else:
        raise SystemExit("corpus_freerun: no READY line in the preamble — engine binary mismatch?")

    items = []
    for i, p in enumerate(prompts):
        rep = ask("reset", "RESET")
        if rep[-1].startswith("ERROR"):
            raise SystemExit(f"corpus_freerun: reset failed on {p['text'][:40]!r}: {rep[-1]}")
        fed = ask("feed " + " ".join(str(t) for t in p["ids"]), "FED")
        want = len(p["ids"])
        got_n = int(fed[-1].split("n=")[1].split()[0])
        if got_n != want:
            raise SystemExit(f"corpus_freerun: feed reported n={got_n}, prompt has {want} ids "
                             f"({p['text'][:40]!r}); a half-fed prompt would make the whole run")
        gen = ask(f"gen {steps}", "END")
        ids = [int(w.split()[1]) for w in gen if w.startswith("GEN ")]
        end = gen[-1]
        stop = int(end.split("stop=")[1].split()[0])
        if stop == 0 and len(ids) != steps:
            raise SystemExit(f"corpus_freerun: gen returned {len(ids)} tokens for a budget of "
                             f"{steps} with stop=0 ({p['text'][:40]!r}) — {end}")
        items.append({"text": p["text"], "ids": [int(t) for t in p["ids"]],
                      "continuation": ids, "stop_token": stop})
        print(f"  {args.start + i + 1}/{len(ref['prompts'])}  {p['text'][:40]!r}"
              f"  n_gen={len(ids)} stop={stop}")

    ask("quit", "BYE")
    proc.wait(timeout=30)

    out_path = manifest_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "kind": "corpus",
        "source": "engine free run (decode_run --session, self-fed samples)",
        "model": str(resolve(args.model)),
        "binary": str(resolve(args.binary)),
        "reference_corpus": args.corpus,
        "steps": steps,
        "n": len(items),
        "start": args.start,
        "note": ("plain-text prompts, same ids as the reference corpus; engine feeds its own "
                 "samples; gen[i] pairs with the pinned greedy_continuation[i]"),
        "prompts": items}, indent=1))
    print(f"wrote {out_path}: {len(items)} prompts, {steps}-token budget")
    return out_path


# ---- the judging half: no device, no engine -----------------------------------------------

def prefix_agree(got: list[int], want: list[int]) -> int:
    """Length of the matching prefix, capped by the reference length. Same definition
    `tools/corpus_agreement.py` and `speccheck/gates/run_g7.py` use (matched / len(reference))."""
    n = min(len(got), len(want))
    k = 0
    while k < n and got[k] == want[k]:
        k += 1
    return k


def prefix_agree_tie_aware(got: list[int], want: list[int], marg_row) -> int:
    """The same prefix count, but a step where the REFERENCE's own logits are exactly tied at the top
    and the engine picked one of the tied ids counts as agreement.

    That is §10.1 G7's existing tie policy, applied per step instead of per fixture, and it is
    reported here as a SECOND metric, not a replacement: strict equality is still what the gate
    enforces. The reason it is worth a number is that greedy breaks a tie by lowest index, which is
    an arbitrary convention no other legal rounding of the same math reproduces, so a strict count
    cannot tell "the engine made a different choice" from "the reference made a choice it had not
    actually made".
    """
    n = min(len(got), len(want))
    k = 0
    while k < n:
        if got[k] == want[k]:
            k += 1
            continue
        if marg_row:
            ids, vals = marg_row["top_ids"][k], marg_row["top_vals"][k]
            if got[k] in ids and vals[ids.index(got[k])] == vals[0] and vals[0] == vals[1]:
                k += 1
                continue
        break
    return k


def stats(xs: list[float]) -> dict:
    s = sorted(xs)
    med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return {"mean": sum(s) / len(s), "median": med, "min": s[0], "max": s[-1]}


def first_div_list(ks: list[int], budget: int) -> list[float]:
    """First-divergence index with the capping made explicit: a prompt that never diverges has no
    index, so it is quoted at `budget` (i.e. "at least this far"), never at 0 and never dropped."""
    return [float(k) if k < budget else float(budget) for k in ks]


def buckets(rows: list[dict], budget: int) -> list[dict]:
    """Agreement conditioned on the reference's own decision margin. The headline D-4 suspects is
    that depth is a property of the prompt, not of the engine, and a mean hides that: bucketing by
    median top1-top2 gap is what shows it."""
    edges = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf")]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [r for r in rows if r["gap_median"] is not None and lo <= r["gap_median"] < hi]
        if not sel:
            continue
        ke = [r["engine_agree"] for r in sel]
        kf = [r["floor_agree"] for r in sel if r["floor_agree"] is not None]
        out.append({
            "gap_median_range": [lo, None if hi == float("inf") else hi],
            "n_prompts": len(sel),
            "engine_mean_agree_tokens": sum(ke) / len(ke),
            "engine_never_diverged": sum(1 for k in ke if k >= budget),
            # None, not 0.0, when there is no floor manifest: a mean over an empty selection would
            # print as a real zero next to the engine's number.
            "floor_mean_agree_tokens": (sum(kf) / len(kf)) if kf else None,
            "floor_never_diverged": (sum(1 for k in kf if k >= budget)) if kf else None,
        })
    return out


def flip_kind(m, gaps, k, token, ref_len):
    """Classify a first divergence at step `k` against the REFERENCE's own ranking at that step.

    Three claims live here and must not be conflated:
      gap12 == 0      the reference sat on an exact BF16 tie; greedy took the lowest id, and no other
                      legal rounding of the same math is obliged to reproduce that (D-1's tie policy
                      in `engine/gate.py` exists for exactly this case)
      tied_at_top     the id this implementation picked has the SAME reference value as the
                      reference's top-1, so the flip is still a tie-break and not a decision
      gap_distinct    how far below the top the first differently-valued candidate sits: the logit
                      error a flip at that step actually required
    Returns None when there is no divergence at `k` or no margin row for the prompt.
    """
    if m is None or k >= ref_len:
        return None
    rank = (m["top_ids"][k].index(token) if token in m["top_ids"][k] else None)
    return {
        "step": k,
        "gap12": gaps[k],
        "gap_distinct": m["gap_distinct"][k],
        "engine_rank_in_ref": rank,
        "tied_at_top": (rank is not None and m["top_vals"][k][rank] == m["top_vals"][k][0]),
    }


def flip_str(f, top_k):
    """One flip, one line: `gap12 0.00 distinct 0.19 rank 1 tied` says the reference was exactly tied,
    the first distinct value sat 0.19 logits lower, this implementation's id was the reference's #2 by
    index, and it was in the tied set."""
    return ("   flip gap12 %6.2f distinct %6.2f rank %s%s"
            % (f["gap12"], f["gap_distinct"],
               ">" + str(top_k) if f["engine_rank_in_ref"] is None else f["engine_rank_in_ref"],
               " tied" if f["tied_at_top"] else ""))


def flip_summary(rows, key):
    """The aggregate over one side's first flips, for the JSON artifact (the print side has the same
    numbers in words; this is the machine-readable half)."""
    sel = [r[key] for r in rows if r[key]]
    if not sel:
        return {"n_diverged": 0}
    return {
        "n_diverged": len(sel),
        "gap12_at_flip": stats([f["gap12"] for f in sel]),
        "at_exact_tie": sum(1 for f in sel if f["gap12"] == 0.0),
        "its_id_also_tied_at_top": sum(1 for f in sel if f["tied_at_top"]),
        "its_id_beyond_ref_topk": sum(1 for f in sel if f["engine_rank_in_ref"] is None),
        "gap_distinct_at_flip": stats([f["gap_distinct"] for f in sel]),
    }


def report(args) -> None:
    ref = json.loads(manifest_path(args.corpus).read_text())
    eng = json.loads(manifest_path(args.engine).read_text())
    flr = json.loads(manifest_path(args.floor).read_text()) if args.floor else None
    marg = json.loads(resolve(args.margins).read_text()) if args.margins else None
    budget = eng["steps"]
    by_text_ref = {p["text"]: p for p in ref["prompts"]}
    by_text_flr = {p["text"]: p for p in flr["prompts"]} if flr else {}
    gap_by_text = {p["text"]: p for p in marg["prompts"]} if marg else {}

    rows = []
    for p in eng["prompts"]:
        refp = by_text_ref.get(p["text"])
        if refp is None:
            raise SystemExit(f"corpus_freerun: engine prompt {p['text'][:40]!r} is not in the "
                             "reference corpus — the two manifests are from different prompt lists")
        ke = prefix_agree(p["continuation"], refp["continuation"])
        kf = None
        if p["text"] in by_text_flr:
            kf = prefix_agree(by_text_flr[p["text"]]["continuation"], refp["continuation"])
        m = gap_by_text.get(p["text"])
        gaps = m["gaps"] if m else None
        ke_tok = p["continuation"][ke] if ke < len(p["continuation"]) else None
        eng_flip = (flip_kind(m, gaps, ke, ke_tok, len(refp["continuation"]))
                    if ke_tok is not None else None)
        fl = by_text_flr.get(p["text"])
        fl_tok = (fl["continuation"][kf] if fl and kf < len(fl["continuation"]) else None)
        flp_flip = (flip_kind(m, gaps, kf, fl_tok, len(refp["continuation"]))
                    if fl_tok is not None else None)
        rows.append({
            "text": p["text"],
            "n_prompt_tokens": len(p["ids"]),
            "engine_agree": ke,
            "engine_agree_tie_aware": (prefix_agree_tie_aware(p["continuation"],
                                                             refp["continuation"], m)
                                       if m else ke),
            "engine_tokens": len(p["continuation"]),
            "engine_stop": p.get("stop_token", 0),
            # A stream that ended before the budget with everything it emitted matching is NOT full
            # agreement. The counts already handle it (never-diverged needs k >= budget), but saying so
            # out loud is what keeps a stop id from quietly reading as a shorter success.
            "engine_truncated": ke == len(p["continuation"]) < len(refp["continuation"]),
            "floor_agree": kf,
            "gap_median": (stats(gaps)["median"] if gaps else None),
            "gap_min": (min(gaps) if gaps else None),
            "engine_flip": eng_flip,
            "floor_flip": flp_flip,
        })

    ke = [r["engine_agree"] for r in rows]
    kt = [r["engine_agree_tie_aware"] for r in rows]
    kf = [r["floor_agree"] for r in rows if r["floor_agree"] is not None]
    frac_t = sum(k / budget for k in kt) / len(kt)
    frac_e = sum(k / budget for k in ke) / len(ke)
    frac_f = sum(k / budget for k in kf) / len(kf) if kf else None
    div_e = first_div_list(ke, budget)
    div_f = first_div_list(kf, budget) if kf else None
    hist = {}
    for k in ke:
        hist[k] = hist.get(k, 0) + 1
    hist_f = {}
    for k in kf:
        hist_f[k] = hist_f.get(k, 0) + 1

    print(f"corpus={args.corpus} engine={args.engine} floor={args.floor} budget={budget} tokens"
          f"  prompts={len(rows)}")
    print()
    trunc = [r["text"] for r in rows if r["engine_truncated"]]
    if trunc:
        print(f"  NOTE: {len(trunc)} prompt(s) whose engine stream ended before the budget (a stop id"
              " or a short gen), counted as diverging at the last token they produced: "
              + ", ".join(t[:30] for t in trunc))
    print("mean prefix agreement (A-18's metric: matched-prefix / budget, averaged over prompts)")
    print(f"  ENGINE vs reference      {frac_e:.4f}")
    if args.margins:
        print(f"  ENGINE, tie-aware        {frac_t:.4f}   <- same run; an exactly-tied decision the "
              "engine resolved inside the tie set counts as agreement")
    if frac_f is not None:
        print(f"  REFERENCE vs reference   {frac_f:.4f}   <- the chaos floor (a legal rounding variant)")
    print()
    print("first divergence index (quoted at the budget when the prompt never diverged)")
    print(f"  engine     mean {stats(div_e)['mean']:.2f}  median {stats(div_e)['median']:.1f}"
              f"  min {stats(div_e)['min']:.0f}  never-diverged {sum(1 for k in ke if k >= budget)}"
              f"/{len(rows)}")
    if args.margins:
        dt = first_div_list(kt, budget)
        print(f"  engine, tie-aware  mean {stats(dt)['mean']:.2f}"
              f"  median {stats(dt)['median']:.1f}  min {stats(dt)['min']:.0f}"
              f"  never-diverged {sum(1 for k in kt if k >= budget)}/{len(rows)}")
    if div_f:
        print(f"  floor      mean {stats(div_f)['mean']:.2f}  median {stats(div_f)['median']:.1f}"
              f"  min {stats(div_f)['min']:.0f}  never-diverged {sum(1 for k in kf if k >= budget)}"
              f"/{len(kf)}")
    print()
    print("distribution of engine agreement depth (tokens agreed out of %d):" % budget)
    for k in sorted(hist):
        print(f"   {k:3d} tokens: {hist[k]:2d} prompt(s)" + ("  <- never diverged" if k >= budget else ""))
    if hist_f:
        print("same distribution for the reference-vs-reference floor:")
        for k in sorted(hist_f):
            print(f"   {k:3d} tokens: {hist_f[k]:2d} prompt(s)"
                  + ("  <- never diverged" if k >= budget else ""))
    print()
    if any(r["gap_median"] is not None for r in rows):
        print("agreement conditioned on the reference's own decision margin (top1-top2, logits):")
        print("   gap median range        n   engine mean tokens (never-div)   floor mean tokens (never-div)")
        for b in buckets(rows, budget):
            hi = b["gap_median_range"][1]
            fl = ("  %6.2f (%2d)" % (b["floor_mean_agree_tokens"], b["floor_never_diverged"])
                  if b["floor_mean_agree_tokens"] is not None else "        -")
            print("   %6s - %-8s %3d   %6.2f (%2d)          %s"
                  % (b["gap_median_range"][0], "inf" if hi is None else hi, b["n_prompts"],
                     b["engine_mean_agree_tokens"], b["engine_never_diverged"], fl))
        gmin = [r["gap_min"] for r in rows if r["gap_min"] is not None]
        if gmin:
            s = stats(gmin)
            print(f"   per-prompt MINIMUM gap over its 24 steps: median {s['median']:.3f},"
                  f" min {s['min']:.3f}, max {s['max']:.3f}")
        # The conditioning that decides how to read the headline: at WHAT kind of decision each
        # implementation flipped, judged by the reference's own ranking at that step. A flip at an
        # exact tie is a tie-break; a flip with a real gap is a different claim.
        def flip_line(who, key):
            sel = [r[key] for r in rows if r[key]]
            if not sel:
                return None
            g = stats([f["gap12"] for f in sel])
            d = stats([f["gap_distinct"] for f in sel])
            return (f"   {who}: {len(sel)} prompts diverged. gap12 at the flip:"
                    f" median {g['median']:.3f}, min {g['min']:.3f}, max {g['max']:.3f}."
                    f" At an exact tie: {sum(1 for f in sel if f['gap12'] == 0.0)}."
                    f" Of those, its own id was tied at the top in the reference:"
                    f" {sum(1 for f in sel if f['tied_at_top'])}."
                    f" Id beyond the reference's top-{marg['top_k']}:"
                    f" {sum(1 for f in sel if f['engine_rank_in_ref'] is None)}."
                    f" First DISTINCT value below the top at the flip: median {d['median']:.3f}"
                    f" (min {d['min']:.3f}) — the logit error a flip there needed.")
        eng_line, fl_line = flip_line("ENGINE", "engine_flip"), flip_line("FLOOR", "floor_flip")
        if eng_line or fl_line:
            print("\nwhat the first flip was, judged by the reference's own ranking at that step:")
            for line in (eng_line, fl_line):
                if line:
                    print(line)
            never = [r for r in rows if not r["engine_flip"]]
            if never:
                ns = stats([r["gap_min"] for r in never if r["gap_min"] is not None])
                print(f"   the {len(never)} prompts that never diverged: minimum gap12 over their 24"
                      f" steps median {ns['median']:.3f}, min {ns['min']:.3f}")
            tied_steps = sum(1 for r in rows if r["gap_min"] is not None and r["gap_min"] == 0.0)
            print(f"   prompts whose 24 steps contain at least one exact tie (gap12 0):"
                  f" {tied_steps}/{len(rows)}")
    else:
        print("no margin file (pass --margins); see tools/corpus_margin.py")
    print()
    print("per prompt (engine agree / budget, floor agree / budget, ref gap median|min):")
    for r in sorted(rows, key=lambda r: (r["engine_agree"], r["text"])):
        g = ("%6.2f|%6.2f" % (r["gap_median"], r["gap_min"])) if r["gap_median"] is not None else "     -"
        fl = str(r["floor_agree"]) if r["floor_agree"] is not None else "-"
        star = " STOP" if r["engine_stop"] else ""
        flip = flip_str(r["engine_flip"], marg["top_k"]) if r["engine_flip"] else ""
        print(f"   {r['engine_agree']:2d}/{budget:<2d} floor {fl:>2s}  gap {g}  {r['text'][:40]!r}"
              f"{star}{flip}")

    doc = {
        "corpus": args.corpus, "engine_manifest": args.engine, "floor": args.floor,
        "margins": args.margins or None, "budget_tokens": budget, "n_prompts": len(rows),
        "engine_mean_prefix_agreement": frac_e,
        "engine_mean_prefix_agreement_tie_aware": frac_t,
        "engine_never_diverged_tie_aware": sum(1 for k in kt if k >= budget),
        "floor_mean_prefix_agreement": frac_f,
        "engine_first_divergence": stats(div_e), "floor_first_divergence":
            (stats(div_f) if div_f else None),
        "engine_never_diverged": sum(1 for k in ke if k >= budget),
        "floor_never_diverged": (sum(1 for k in kf if k >= budget) if kf else None),
        "engine_depth_histogram": {str(k): v for k, v in sorted(hist.items())},
        "floor_depth_histogram": {str(k): v for k, v in sorted(hist_f.items())},
        "by_gap_bucket": buckets(rows, budget),
        "engine_first_flip": flip_summary(rows, "engine_flip"),
        "floor_first_flip": flip_summary(rows, "floor_flip"),
        "prompts": rows,
    }
    if args.json:
        jp = resolve(args.json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(doc, indent=1))
        print(f"\nwrote {jp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus48", help="reference manifest (pinned greedy)")
    ap.add_argument("--engine", default="corpus48engine", help="engine manifest to write / read")
    ap.add_argument("--floor", default="corpus48sdpa",
                    help="reference-vs-reference manifest (the chaos floor); '' to skip")
    ap.add_argument("--margins", default="data/refs/corpus48-margins.json",
                    help="per-step reference gaps from tools/corpus_margin.py; '' to skip")
    ap.add_argument("--out", default=None, help="where to write the engine manifest (default --engine)")
    ap.add_argument("--steps", type=int, default=0, help="token budget (default: the corpus' own)")
    ap.add_argument("--start", type=int, default=0, help="first prompt index (chunking)")
    ap.add_argument("--limit", type=int, default=0, help="how many prompts (0 = rest)")
    ap.add_argument("--binary", default="engine/build/decode_run")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--report-only", action="store_true", help="no device: judge what is on disk")
    ap.add_argument("--json", default=None, help="write the full per-prompt report here")
    args = ap.parse_args()
    if args.out is None:
        args.out = args.engine
    if args.report_only:
        report(args)
        return 0
    drive(args)
    if args.start == 0 and not args.limit:
        # A chunked run has nothing to judge yet; only the run that covered the whole corpus can be
        # compared against the floor and the margins without a misleading partial mean.
        args.floor = args.floor or None
        if not (args.margins and resolve(args.margins).exists()):
            args.margins = None
        report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
