#!/usr/bin/env python3
"""P7 / MTP — the WITNESS: the same protocol, on torch bits instead of engine bits.

`run_accept.py` measured a = 0.6641 on the engine's own bits against a carried torch-vs-pins number of
0.7329. Two different things could produce that gap, and they demand opposite responses: either the
engine's bf16 bits are worse drafter fuel than the reference's (A-18's chaos floor arriving in the
accept path, which is expected and is not a bug), or MY JOIN is subtly wrong and the engine number is
 fiction. The only way to tell them apart is to run this protocol — this population, this step budget,
this round schedule, this settlement rule — on the torch bits the carried number came from. If the
torch side moves down from 0.7329 towards the engine's number, the protocol explains the gap and the
engine's bits are exonerated. If it stays at 0.7329, the engine side has a real defect.

WHAT "THE SAME PROTOCOL" MEANS, spelled out because the carried 0.7329 was NOT measured this way:
  population   the same 49 `speccheck/fixtures/corpus.txt` lines, with the SAME token ids the engine
               was fed (from `data/refs/corpus48/manifest.json`), not a re-tokenisation
  steps        32 per line, same stop rule (id 248046 in the generation phase only)
  rounds       the engine's `loop` schedule: a round after step s settles against step s+1's sample;
               after an accept the next round is two steps out, after a reject one step out — which is
               `greedy_mtp(k=1, verify="steps")`'s round geometry on a plain greedy stream
  settlement   proposal == the target's own next greedy token, by position
  draft        `speccheck.mtp.MTPDraft`, h_source="post_norm", fc_order="eh" — the settings
               `gates/run_mtp.py` chose by measurement, and what the engine ported

The carried 0.7329 itself is not re-made here: it is on disk as
`prototype/mtp_gatezero/draft_survival.json` (468 rounds, 343 accepted, 20 prompts x 40 steps,
stop_on_eos=False) and nothing that feeds it changed, so citing it is the correct budget (`AGENTS.md`:
if a result is already on disk and nothing that feeds it changed, cite it). `--protocol carried` below
will re-make it if someone wants that witnessed too.

This is torch CPU only — no device, no GPU lock, no timing claim about the engine.

USAGE
    .venv/bin/python3 prototype/mtp_accept/witness_torch.py [--protocol same|carried] [--sched loop]
        [--steps 32] [--limit 49] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "prototype" / "mtp_accept")):
    if p not in sys.path:
        sys.path.insert(0, p)

from speccheck.mtp import MTPDraft, greedy_mtp          # noqa: E402
from speccheck.specimpl import Forward, Weights         # noqa: E402
from speccheck.textio import Tokenizer                  # noqa: E402

import run_accept as ra                                 # noqa: E402

STOP_ID = 248046          # end-of-turn token, generation phase only (ids only, never rendered)


def same_protocol(w: Weights, prompts: list[dict], steps: int, sched: str) -> dict:
    """The engine's geometry on torch bits: plain greedy target, one draft round per due step.

    Mirrors `RoundTracker::after_step` (engine/mtp_draft.cpp) line for line, because that is what makes
    the two instruments comparable: settle the pending proposal, then draft unless a `loop` accept just
    advanced q by two, and truncate the block's cache lane on a reject (greedy_mtp's `m.restore(msnap)`).
    Positions are the engine's own: step i's row is `p_i = len(ids) + i - 1`, which is the label the
    engine's `GEN` stream prints that step's sample under, the draft is filed against `p_i + 1` and
    proposes the token at `p_i + 2`.
    """
    pairs: list[tuple[dict, bool]] = []
    made = skipped = 0
    per_line = []
    for li, p in enumerate(prompts):
        ids = p["ids"]
        L = len(ids)
        f = Forward(w, gdn_form="chunk")
        m = MTPDraft(w, h_source="post_norm", fc_order="eh")
        normed = f.prefill(torch.tensor(ids))
        t = int(f.logits(normed)[-1].argmax())       # t_0, the token at position L
        h = normed[-1:]                              # the row that produced it, post-final-norm
        pending: dict | None = None
        round_no = 0
        emitted, stop = 0, False
        for i in range(1, steps + 1):
            if t == STOP_ID:                         # §07.5: emitted, then the turn ends
                stop = True
                break
            row = f.decode(t)
            t_next = int(f.logits(row)[-1].argmax())  # the sample of step i, at position L + i - 1
            h = row[-1:]
            p_i = L + i - 1

            settled, accepted = False, False
            if pending is not None:
                if pending["adjudicated_by_pred_pos"] != p_i:
                    raise SystemExit(f"witness: proposal filed for position"
                                     f" {pending['adjudicated_by_pred_pos']} met step {p_i}; the index"
                                     " slid, so no number here would mean anything")
                settled, accepted = True, pending["proposal"] == t_next
                pairs.append((pending, accepted))
                pending = None

            if not settled:
                due = True                            # the previous step drafted nothing
            elif sched == "every":
                due = True                            # every draft context is priced
            elif accepted:
                due = False                           # kLoop: q advanced by 2, no round here
            else:
                due = True
                m.kv["k"].pop()                       # kLoop reject: truncate the lane, move 0 bytes
                m.kv["v"].pop()
                m.n -= 1
            if due:
                round_no += 1
                proposal, _ = m.draft(t_next, h, p_i + 2)
                pending = {"round": round_no, "draft_pos": p_i + 2,
                           "adjudicated_by_pred_pos": p_i + 1, "proposal": proposal,
                           "y_next": t_next}
                made += 1
            t = t_next
            emitted += 1
            if pending is not None and i == steps:
                # The budget path's sample is never printed as a token but is a real target prediction,
                # and the engine lets it settle (the `LASTSAMPLE` line). Mirrored so both instruments
                # count the same settlements.
                pairs.append((pending, pending["proposal"] == t_next))
                pending = None
        if pending is not None:
            skipped += 1                              # stop id ended the turn before it settled
        per_line.append({"line": li, "gen_tokens": emitted, "stop_fired": int(stop),
                         "rounds": round_no})
    accepted = sum(1 for _, ok in pairs if ok)
    return {"proposals_made": made + skipped, "proposals_adjudicated": len(pairs),
            "proposals_accepted": accepted,
            "proposals_skipped": skipped,
            "accept_overall": accepted / len(pairs) if pairs else float("nan"),
            "accept_by_draft_position": ra.depth_table(pairs, "draft_pos"),
            "accept_by_round_in_line": ra.depth_table(pairs, "round"),
            "lines": per_line}


def carried(w: Weights, tok: Tokenizer, prompts: list[dict], steps: int) -> dict:
    """`draft_survival.py`'s own protocol: greedy_mtp, verify="steps", no stop rule, 20x40 by default.
    Reproducing 343/468 = 0.7329 here is the literal form of "the harness/join is not the problem".

    It was run on 2026-10-22 and returned 343/468 exactly, which is the carried figure to the last
    digit; that is why the same-protocol witness above is the informative one."""
    drafted = accepted = 0
    for p in prompts:
        f = Forward(w, gdn_form="chunk")
        m = MTPDraft(w, h_source="post_norm", fc_order="eh")
        out, st = greedy_mtp(f, m, p["ids"] or tok.encode(p["text"]), steps, k=1,
                             stop_on_eos=False, verify="steps")
        drafted += st["drafted"]
        accepted += st["accepted"]
    return {"protocol": "carried (greedy_mtp k=1 verify=steps stop_on_eos=False)",
            "prompts": len(prompts), "steps": steps,
            "proposals_made": drafted, "proposals_adjudicated": drafted, "proposals_accepted": accepted,
            "proposals_skipped": 0, "rounds": drafted, "accepted": accepted,
            "accept_overall": accepted / drafted if drafted else float("nan"),
            "carried_on_disk": 343 / 468}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="same", choices=("same", "carried"))
    ap.add_argument("--sched", default="loop", choices=("loop", "every"))
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--json", default="")
    ap.add_argument("--engine-json", default="",
                    help="the engine run's JSON, so the two numbers are compared rather than one of "
                         "them being remembered in prose here and quietly rotting")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    prompts = ra.load_population()
    if args.limit:
        prompts = prompts[:args.limit]
    w = Weights(pathlib.Path(args.model))
    if args.protocol == "carried":
        tok = Tokenizer(args.model)
        res = carried(w, tok, prompts, args.steps)
    else:
        res = same_protocol(w, prompts, args.steps, args.sched)
        res["population"] = {"lines": len(prompts), "steps": args.steps, "schedule": args.sched}
        res["denominator"] = "proposals adjudicated / proposals made"
    print(json.dumps(res, indent=2))
    print(f"WITNESS torch bits, protocol={args.protocol}: accept = "
          f"{res['proposals_accepted']}/{res['proposals_adjudicated']} = {res['accept_overall']:.4f}")
    # The engine side is read from its own artifact rather than quoted here: a number typed into a script
    # is a number that will be wrong the next time somebody runs the pass.
    if args.engine_json:
        eng = json.loads(pathlib.Path(args.engine_json).read_text())
        print(f"ENGINE bits, same protocol: accept = {eng['proposals_accepted']}"
              f"/{eng['proposals_adjudicated']} = {eng['accept_overall']:.4f}"
              f"  (from {args.engine_json})")
        print(f"difference on the same protocol: {eng['accept_overall'] - res['accept_overall']:+.4f}"
              " — the size of the engine's bits, as opposed to the size of the protocol")
    else:
        print("no --engine-json given, so no engine comparison is claimed here")
    print("reference points: carried torch-vs-pins 0.7329 (343/468, "
          "prototype/mtp_gatezero/draft_survival.json); kill line a_needed = 0.257")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
