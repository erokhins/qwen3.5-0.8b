#!/usr/bin/env python3
"""Study B - the accept rate and the draft-side candidate survival, measured on the S09 form.

P7 gate zero, part 3. Study A asked whether the TARGET's next token sits in the previous row's top-k;
this script asks the two questions that actually price the route, using the shipped S09 drafter
(`speccheck/mtp.py`) rather than a re-derivation of it:

  `a`  the accept rate of a k=1 draft against the same loop that will verify it. `greedy_mtp` with
       `verify="steps"` is the reading G8a blesses as bit-identical to plain greedy, which is the
       closest CPU stand-in there is for "the draft proposes and the same engine verifies". It is a
       STAND-IN, not the measurement: A-18 says bf16 continuations are chaotic, so the number that
       finally funds P7 has to come from the engine's own bits on device (MTP-GATEZERO.md S3). What
       this script can do is show whether `a` is stable with depth and prompt, and how big the
       decision room is.

  survival(k)  whether the DRAFT's own proposal d* lies in the previous target row's top-k - the real
       condition for the restricted head (argmax over a k-row gather equals argmax over the whole
       table iff the whole-table argmax is in the set). Study A measured a proxy for this with the
       target's own token; this measures d* itself.

ALIGNMENT, the part that has burned this repo before (AGENTS.md: a free run compared at the wrong
index reads 0/128 and looks like total divergence). Nothing is assumed about indices. Each round
records the drafted position `p` and reads the target's own per-position argmax out of the wrapped
`Forward.logits`, and the accept test is a JOIN on those recorded positions:

    accept  <=>  d* == pred_at[p-1]          (p-1 is the position of the token being drafted behind)
    guard   <=>  argmax(previous target row) == y_next     checked EVERY round, aborts if it fails

so a wrong index shows up as a guard failure or as a 0 % accept rate, never as a plausible number.

    .venv/bin/python prototype/mtp_gatezero/draft_survival.py [--prompts 10] [--steps 32] [--k 1]

CPU only. torch CPU bf16, one process, no device, no timing figure reported.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speccheck.mtp import MTPDraft, greedy_mtp                      # noqa: E402
from speccheck.specimpl import Forward, Weights                     # noqa: E402
from speccheck.textio import Tokenizer                              # noqa: E402

# QUOTED from engine/generation-limits.md S1 (4,032 lanes). Nothing here measures time.
STEP_MS = 10.9357
LM_HEAD_MS = 2.1685
PLATEAU_GIBS = 237.0
LAUNCH_US = 3.10
BLOCK_MIB = 39.0107          # mtp_census.py, from the shard
HIDDEN, EBYTES = 1024, 2

DEFAULT_KS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


class RecForward(Forward):
    """`Forward` that records the target's own argmax per position, as the verify produces it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pred_at: dict[int, int] = {}
        self.last_row: torch.Tensor | None = None

    def logits(self, normed: torch.Tensor) -> torch.Tensor:
        out = super().logits(normed)
        rows = out.reshape(-1, out.shape[-1])
        base = self.P - rows.shape[0]          # positions these rows belong to
        for i in range(rows.shape[0]):
            self.pred_at[base + i] = int(rows[i].argmax())
        self.last_row = rows[-1].detach().float().reshape(-1).cpu()
        return out


class Probe(MTPDraft):
    """`MTPDraft` that records one record per round: position, proposal, and its rank in the row in hand."""

    def __init__(self, w, rec: RecForward, **kw):
        super().__init__(w, **kw)
        self.rec = rec
        self.rounds: list[dict] = []
        self.guard_fail = 0

    def draft(self, y_next: int, h: torch.Tensor, p: int, dbg: dict | None = None):
        row = self.rec.last_row
        if row is None:
            raise SystemExit("draft called before the target produced any logits row; the "
                             "candidate-set question cannot be answered for this round")
        # the alignment guard: the row in hand is the row that emitted y_next
        if int(row.argmax()) != y_next:
            self.guard_fail += 1
        d, x_d = super().draft(y_next, h, p, dbg or {})
        v = float(row[d])
        self.rounds.append({
            "p": p,
            "draft": d,
            "y_next": y_next,
            "rank_strict": int(np.count_nonzero(row > v)) + 1,
            "rank_tie": int(np.count_nonzero(row >= v)),
        })
        return d, x_d


def ms_of_mib(mib: float) -> float:
    return mib * 2**20 / (PLATEAU_GIBS * 2**30) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--verify", default="steps", choices=["steps", "batch"],
                    help="steps = the G8a reading, bit-identical to plain greedy")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.k != 1:
        raise SystemExit("this rig prices the k=1 model (1+a)/(1+c); k>1 needs its own model")

    torch.set_grad_enabled(False)
    w = Weights(pathlib.Path(args.model))
    tok = Tokenizer(args.model)
    prompts = [ln.strip() for ln in
               (ROOT / "speccheck" / "fixtures" / "corpus.txt").read_text().splitlines()
               if ln.strip()][:args.prompts]
    if not prompts:
        raise SystemExit("no prompts read from speccheck/fixtures/corpus.txt")

    all_rounds: list[dict] = []
    guard_fail = 0
    drafted = accepted = 0
    per_prompt = []
    for pi, p in enumerate(prompts):
        f = RecForward(w, gdn_form="chunk")
        m = Probe(w, f, h_source="post_norm", fc_order="eh")
        out, st = greedy_mtp(f, m, tok.encode(p), args.steps, k=args.k,
                             stop_on_eos=False, verify=args.verify)
        acc_here = 0
        for r in m.rounds:
            oracle = f.pred_at.get(r["p"] - 1)
            if oracle is None:
                r["accept"] = None          # position never verified in this run
            else:
                r["accept"] = (r["draft"] == oracle)
                acc_here += int(r["accept"])
        all_rounds += m.rounds
        guard_fail += m.guard_fail
        drafted += st["drafted"]
        accepted += st["accepted"]
        per_prompt.append({"prompt": p[:36], "loop_accept": st["accepted"], "loop_draft": st["drafted"],
                           "probe_accept": acc_here, "probe_rounds": len(m.rounds)})
        print(f"  prompt {pi:>2} rounds={len(m.rounds):>3} loop accept={st['accepted']}/{st['drafted']}"
              f"  probe accept={acc_here}  out={len(out)-1} tok")

    scored = [r for r in all_rounds if r.get("accept") is not None]
    print(f"\n=== Study B: the S09 drafter, k={args.k}, verify={args.verify}, "
          f"{len(prompts)} prompts x {args.steps} steps ===")
    print(f"rounds drafted={drafted} accepted={accepted} -> loop accept rate "
          f"{accepted/max(drafted,1):.4f}   (S09.5 recorded 0.642 at k=1, 9 prompts x 24)")
    print(f"rounds scored by position join: {len(scored)} of {len(all_rounds)}   "
          f"probe accept {sum(r['accept'] for r in scored)}/{len(scored)}")
    if guard_fail:
        raise SystemExit(f"ALIGNMENT GUARD FAILED on {guard_fail} rounds: the row in hand did not "
                         "emit y_next. The candidate-set question is not answerable until the index "
                         "relation is fixed - do not read a survival rate.")
    if not scored:
        raise SystemExit("no round could be scored; nothing to price")
    if len(scored) != len(all_rounds):
        print(f"  note: {len(all_rounds) - len(scored)} rounds had no verify position and are excluded")

    a = sum(r["accept"] for r in scored) / len(scored)
    rs = np.array([r["rank_strict"] for r in scored])
    rt = np.array([r["rank_tie"] for r in scored])
    acc = np.array([r["accept"] for r in scored])
    print(f"\nrank of the DRAFT's proposal inside the row the target already has:")
    print(f"  strict median {np.median(rs):>7.0f}  p75 {np.percentile(rs,75):>7.0f}  "
          f"p90 {np.percentile(rs,90):>7.0f}  p99 {np.percentile(rs,99):>7.0f}  max {rs.max():.0f}")

    print(f"\naccept by depth into the generation (does `a` decay?):")
    for lo, hi in ((0, 7), (8, 15), (16, 31), (32, 63), (64, 10**9)):
        sel = [(r, k) for r, k in zip(scored, acc) if lo <= r["p"] <= hi]
        if sel:
            print(f"  positions {lo:>3}-{hi if hi < 10**9 else '+':<3}: accept "
                  f"{sum(int(k) for _, k in sel)}/{len(sel)} = "
                  f"{np.mean([k for _, k in sel]):.4f}")

    block_ms = ms_of_mib(BLOCK_MIB)
    launches_ms = LAUNCH_US * 18 / 1e3
    full_c = (LM_HEAD_MS + block_ms + launches_ms) / STEP_MS
    print(f"\n=== what each route buys with THIS accept rate (a={a:.4f}) ===")
    print(f"{'route':<26} {'c':>7} {'eff. accept':>12} {'tokens/step':>12} {'speedup':>8}")
    print(f"{'FULL tied head':<26} {full_c:>7.4f} {a:>12.4f} {(1+a)/(1+full_c):>12.4f} "
          f"{(1+a)/(1+full_c):>8.3f}x")
    table = []
    for kk in DEFAULT_KS:
        in_set = rs <= kk
        # a restricted proposal whose true argmax is outside the set is a LOST accept (the target
        # rejects whatever else the k-row head picks), so the effective accept is the joint event
        # P(accepted AND proposal inside the set), taken over ALL rounds.
        a_eff = float(np.mean([bool(k) and bool(s) for k, s in zip(acc, in_set)]))
        c = (block_ms + launches_ms + kk * HIDDEN * EBYTES / (PLATEAU_GIBS * 2**30) * 1e3) / STEP_MS
        sp = (1 + a_eff) / (1 + c)
        surv = float(np.mean(in_set))
        print(f"{'restricted top-' + str(kk):<26} {c:>7.4f} {a_eff:>12.4f} {(1+a_eff)/(1+c):>12.4f} "
              f"{sp:>8.3f}x   survival {100*surv:5.1f}%")
        table.append({"k": kk, "c": c, "survival": surv, "accept_eff": a_eff, "speedup": sp})
    best = max(table, key=lambda r: r["speedup"])
    print(f"\nbest restricted setting: top-{best['k']} at {best['speedup']:.3f}x "
          f"(full head {(1+a)/(1+full_c):.3f}x)")
    print("c prices the block at the copy plateau, which is a FLOOR: the ledger's GEMV families run at "
          "70-75 % of that, so add ~0.05 to every restricted c before quoting it.")

    if args.json:
        pcts = {f"p{q}": float(np.percentile(rs, q)) for q in (50, 75, 90, 99)}
        depth = []
        for lo, hi in ((0, 7), (8, 15), (16, 31), (32, 63)):
            sel = [int(k) for r, k in zip(scored, acc) if lo <= r["p"] <= hi]
            if sel:
                depth.append({"positions": f"{lo}-{hi}", "n": len(sel),
                              "accept": sum(sel) / len(sel)})
        pathlib.Path(args.json).write_text(json.dumps({
            "prompts": len(prompts), "steps": args.steps, "verify": args.verify,
            "rounds": len(scored), "accept": a,
            "loop_accept": accepted, "loop_draft": drafted,
            "rank_strict": pcts,
            "accept_by_position": depth,
            "routes": table, "full_c": full_c, "full_speedup": (1 + a) / (1 + full_c),
            "per_prompt": per_prompt,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
