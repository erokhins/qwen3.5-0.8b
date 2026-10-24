#!/usr/bin/env python3
"""session_rowdiff.py — adjudicate a SESSION stream flip against BOTH arms' logits rows.

WHY THIS EXISTS. `prototype/mtp_accept/session_longturn.sh` can say that a multi-turn session past the
one-pass attention ceiling (MAXSC = 512 lanes, `kernels::mtp_attn_max_lanes()`) emits a different token
stream with rounds on (`SESSION-ACCEPT.md` §8), but a token id cannot adjudicate WHY. The repo already has
the discriminator for a stream flip — `m3_loop_rate.py`'s `flip_witness`: a flip is a legal gap between two
bf16 forms only if the token the other arm emitted is this row's RUNNER-UP and the top1-top2 margin is
below ONE bf16 ULP of the winning logit — and it has always taken its reference from a RUN-mode `--out`
dump (`logits<step>.bin`). A session wrote no dumps at all, so the discriminator could not be run on the one
geometry that needs it. `--session --out DIR` plus the `dump <pos>` command (`decode_run.cpp`,
`SessionDumper`) closes that; this file is its adjudicator.

THE JOIN IS THE POSITION, AND THE POSITION CONVENTION IS THE ENGINE'S. A `DUMP` line names the position of
the row that DECIDED a token, which is the same key as the `GEN <id> <pos>` label on both arms, so no index
shift is involved and both sides of the comparison are FILES of the same dtype — never a dump compared
against a printed token (the mismatched-pair error `AGENTS.md` bills for, and the reason `emitted.txt` and
`sampled.txt` are not index-aligned).

WHAT IT REFUSES TO DO.
  * Read a dtype off an assumption: a row is F32 and the byte count is checked against the `vocab=` the
    engine echoed, because a BF16 misread yields NaN aggregates with a plausible `ndiff` attached.
  * Adjudicate past the FIRST divergence: after that the two arms are different texts and every later row
    differs by construction, which says nothing.
  * Call a flip legal on vibes: the criterion is computed from the format (bf16, 8 mantissa bits), and the
    logits, margin, ULP bound, rank and emitter are printed for every flip so any reader can re-adjudicate
    at a different threshold.
  * Pass on an empty population: zero joined positions is `NO VERDICT`, not green (`--selftest` proves the
    predicate bites, per the standing rule that a check which cannot fail is decoration).

VERDICT. `NEAR-TIE` = every divergence is the other arm's runner-up at a sub-ULP margin (two legal bf16
forms disagreeing at a tie). `STRUCTURAL` = anything else: a token from further down the ranking, or a
margin a legal form difference cannot reach. Exit non-zero on STRUCTURAL, because per the fix-first rule
that outranks speed work.

CPU-ONLY. It reads dumps a device pass already wrote; it runs no model and takes no lock.

USAGE
    prototype/mtp_accept/session_rowdiff.py --off-log A.off.log --on-log A.on.log \
        --off-dir /tmp/a.off --on-dir /tmp/a.on [--json out.json]
    prototype/mtp_accept/session_rowdiff.py --selftest
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

BF16_REL_ULP = 2.0 ** -8      # bf16 keeps 8 mantissa bits: one ULP of x is x * 2^-8
DUMP_RE = re.compile(r"^DUMP seq=(\d+) kind=(\w+) pos=(\d+) emitted=(\d+) vocab=(\d+) bytes=(\d+)"
                     r" file=(\S+)")
GEN_RE = re.compile(r"^GEN (-?\d+) (-?\d+)$")


def read_row(d: pathlib.Path, name: str, vocab: int, nbytes: int) -> np.ndarray:
    """One dumped logits row, dtype read off the BYTE COUNT and cross-checked against the echoed vocab."""
    p = d / name
    if not p.exists():
        raise SystemExit(f"{p}: the engine named a dump file that is not on disk")
    size = p.stat().st_size
    if size != nbytes:
        raise SystemExit(f"{p}: {size} B on disk, {nbytes} B in the engine's DUMP line")
    if size == 4 * vocab:
        a = np.fromfile(p, dtype=np.float32)
    elif size == 2 * vocab:
        raise SystemExit(f"{p}: BF16-sized ({size} B); this adjudicator reads F32 only")
    else:
        raise SystemExit(f"{p}: {size} B is neither 4*vocab nor 2*vocab for vocab={vocab}; refusing to"
                         " guess a dtype")
    if a.size != vocab:
        raise SystemExit(f"{p}: {a.size} elements, expected {vocab}")
    return a


def parse_dumps(log: str) -> tuple[dict[int, dict], list[int]]:
    """pos -> the LAST DECIDING row dumped there, plus the positions that were priced and thrown away.

    A round computes `rows` rows and emits at most that many, so one position can be dumped twice: once by
    a round whose row a verdict discarded (`emitted=0`) and once by the body that actually settled it. Only
    `emitted=1` rows are deciding rows, and the LAST one wins. A position with only `emitted=0` rows is
    returned separately rather than silently skipped — a comparison that reads nothing has to say so.
    """
    out: dict[int, dict] = {}
    dropped: set[int] = set()
    for line in log.splitlines():
        m = DUMP_RE.match(line)
        if not m:
            continue
        seq, kind, pos, emitted, vocab, nbytes, name = m.groups()
        pos_i = int(pos)
        rec = {"seq": int(seq), "kind": kind, "pos": pos_i, "emitted": int(emitted),
               "vocab": int(vocab), "bytes": int(nbytes), "file": name}
        if int(emitted) == 0:
            dropped.add(pos_i)
            continue
        dropped.discard(pos_i)
        out[pos_i] = rec
    return out, sorted(dropped)


def parse_gens(log: str) -> list[tuple[int, int]]:
    return [(int(m.group(1)), int(m.group(2)))
            for m in (GEN_RE.match(l) for l in log.splitlines()) if m]


def rank_of(order: np.ndarray, token: int) -> int:
    hit = np.where(order == token)[0]
    return int(hit[0]) if hit.size else -1


def greedy_argmax(row: np.ndarray) -> int:
    """§03.4's rule, re-derived rather than read off a line: strict `>`, so the LOWEST id wins a tie."""
    best = 0
    for i in range(1, row.size):
        if row[i] > row[best]:
            best = i
    return int(best)


def adjudicate(off: np.ndarray, on: np.ndarray, off_got: int, on_got: int) -> dict:
    """The flip's numbers, from the OFF arm's row as the reference side, plus the row-vs-row distance."""
    o_off = np.argsort(off)[::-1]
    o_on = np.argsort(on)[::-1]
    t1, t2 = int(o_off[0]), int(o_off[1])
    margin = float(off[t1] - off[t2])
    ulp = abs(float(off[t1])) * BF16_REL_ULP
    dot = float(np.dot(off, on))
    nrm = float(np.linalg.norm(off) * np.linalg.norm(on))
    diff = np.abs(off - on)
    mx = float(diff.max())
    at = int(np.argmax(diff))
    return {
        "off_top1": t1, "off_top1_logit": float(off[t1]),
        "off_top2": t2, "off_top2_logit": float(off[t2]),
        "off_margin": margin, "ulp_bound": ulp,
        "margin_over_ulp": (margin / ulp) if ulp > 0 else float("inf"),
        "off_argmax": off_got, "on_emitted": on_got,
        "rank_of_on_token_in_off_row": rank_of(o_off, on_got),
        "on_top1": int(o_on[0]), "on_top1_logit": float(on[o_on[0]]),
        "on_margin": float(on[o_on[0]] - on[o_on[1]]),
        "rank_of_off_token_in_on_row": rank_of(o_on, off_got),
        "one_minus_cos": (1.0 - dot / nrm) if nrm > 0 else float("nan"),
        "ndiff": int(np.count_nonzero(off != on)),
        "max_abs_diff": mx, "max_abs_at": at,
        "max_rel_diff": float(diff[at] / max(abs(float(off[at])), 1e-30)),
        # The criterion, from the format, not from the observed flips.
        "near_tie": bool(rank_of(o_off, on_got) == 1 and margin < ulp),
    }


def selftest() -> int:
    """The predicate has to be able to say NO: three of these four fabricated flips are STRUCTURAL."""
    v = 64
    tight = np.zeros(v, dtype=np.float32)
    tight[7], tight[9] = 20.0, 19.97        # runner-up INSIDE one ULP of 20.0 (0.0781)
    wide = np.zeros(v, dtype=np.float32)
    wide[7], wide[9] = 20.0, 12.0           # same pair, margin far outside one ULP
    third = np.zeros(v, dtype=np.float32)
    third[7], third[9], third[3] = 20.0, 19.97, 19.9   # the other arm's token is THIRD, not second
    cases = [
        ("runner-up, sub-ULP margin", tight, 9, True),
        ("runner-up, wide margin", wide, 9, False),
        ("third-ranked token at a sub-ULP margin", third, 3, False),
        ("third-ranked token, wide margin", wide, 3, False),
    ]
    bad = 0
    for name, row, other, want in cases:
        a = adjudicate(row, row, greedy_argmax(row), other)
        got = a["near_tie"]
        print(f"  {'ok  ' if got == want else 'FAIL'} {name}: near_tie={got} want={want}"
              f"  rank={a['rank_of_on_token_in_off_row']}"
              f"  margin {a['off_margin']:.4f} vs ulp {a['ulp_bound']:.4f}")
        if got != want:
            bad += 1
    print("SELFTEST " + ("PASS" if bad == 0 else f"FAIL ({bad} case(s))"))
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off-log", default="")
    ap.add_argument("--on-log", default="")
    ap.add_argument("--off-dir", default="")
    ap.add_argument("--on-dir", default="")
    ap.add_argument("--maxsc", type=int, default=512,
                    help="the one-pass attention ceiling; printed beside each position")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    for a in ("off_log", "on_log", "off_dir", "on_dir"):
        if not getattr(args, a):
            raise SystemExit(f"--{a.replace('_', '-')} is required (or run --selftest)")
    off_log = pathlib.Path(args.off_log).read_text()
    on_log = pathlib.Path(args.on_log).read_text()
    off_dir, on_dir = pathlib.Path(args.off_dir), pathlib.Path(args.on_dir)

    dg_off, dropped_off = parse_dumps(off_log)
    dg_on, dropped_on = parse_dumps(on_log)
    if not dg_off or not dg_on:
        # A comparison that read nothing cannot pass — the empty-parse trap, in a new costume.
        print(f"NO VERDICT: the arms produced {len(dg_off)} (off) and {len(dg_on)} (on) deciding rows")
        return 1

    gens_off, gens_on = parse_gens(off_log), parse_gens(on_log)
    first_div, first_pair = None, None
    for i, (a, b) in enumerate(zip(gens_off, gens_on)):
        if a != b:
            first_div, first_pair = a[1], (a, b)
            print(f"first differing GEN line: index {i}  off=GEN {a[0]} {a[1]}"
                  f"  on=GEN {b[0]} {b[1]}")
            break
    if first_div is None:
        print("the two arms' GEN streams agree; nothing to adjudicate")
        return 0

    shared = sorted(set(dg_off) & set(dg_on))
    print(f"deciding rows dumped: off={len(dg_off)} on={len(dg_on)}"
          f" (priced-then-dropped: off={len(dropped_off)} on={len(dropped_on)});"
          f" joinable positions={len(shared)}; first divergence at {first_div}; MAXSC={args.maxsc}")
    flips, agrees, skipped = [], 0, []
    for pos in shared:
        if pos > first_div:
            skipped.append(pos)          # past the first flip the two arms are different texts
            continue
        ro, rn = dg_off[pos], dg_on[pos]
        a = read_row(off_dir, ro["file"], ro["vocab"], ro["bytes"])
        b = read_row(on_dir, rn["file"], rn["vocab"], rn["bytes"])
        tok_off, tok_on = greedy_argmax(a), greedy_argmax(b)
        if tok_off == tok_on:
            agrees += 1
            continue
        r = adjudicate(a, b, tok_off, tok_on)
        r.update({"position": pos, "above_ceiling": pos >= args.maxsc,
                  "off_kind": ro["kind"], "on_kind": rn["kind"],
                  "off_seq": ro["seq"], "on_seq": rn["seq"]})
        flips.append(r)
    print(f"compared positions <= {first_div}: argmax agrees {agrees}, flips {len(flips)},"
          f" not adjudicable (past the first flip) {len(skipped)}")
    for r in flips:
        print(f"\n  position {r['position']}  {'ABOVE' if r['above_ceiling'] else 'below'} the ceiling"
              f"  emitted_by: off={r['off_kind']}(seq {r['off_seq']})"
              f" on={r['on_kind']}(seq {r['on_seq']})")
        print(f"    OFF row:  top1={r['off_top1']} logit {r['off_top1_logit']:.6f}"
              f"   top2={r['off_top2']} logit {r['off_top2_logit']:.6f}"
              f"   margin {r['off_margin']:.6f}")
        print(f"    bf16 ULP of that top1 = {r['ulp_bound']:.6f}"
              f"   margin/ULP = {r['margin_over_ulp']:.4f}")
        print(f"    ON row emitted {r['on_emitted']}: its rank in the OFF row ="
              f" {r['rank_of_on_token_in_off_row']} (0 = top1, 1 = runner-up)")
        print(f"    ON row's own top1={r['on_top1']} logit {r['on_top1_logit']:.6f}"
              f"  margin {r['on_margin']:.6f}; the OFF token's rank there ="
              f" {r['rank_of_off_token_in_on_row']}")
        print(f"    row vs row: 1-cos {r['one_minus_cos']:.3e}  ndiff {r['ndiff']}"
              f"  max_abs {r['max_abs_diff']:.3e} at idx {r['max_abs_at']}"
              f"  max_rel {r['max_rel_diff']:.3e}")
        print("    -> " + ("NEAR-TIE: a legal gap between two bf16 forms" if r["near_tie"]
                           else "STRUCTURAL: no legal form gap reaches this flip"))
    hard = [r for r in flips if not r["near_tie"]]
    print("\n=== VERDICT ===")
    if not flips:
        print("NO-FLIP: the GEN streams diverged but no dumped position at or before the divergence"
              " disagrees — the dump window missed it. Widen `dump <min> <max>`.")
        return 1
    if hard:
        print(f"STRUCTURAL — {len(hard)} flip(s) are not the other arm's own runner-up at a sub-ULP"
              " margin. A legal form difference cannot produce this; treat it as a defect.")
    else:
        print("NEAR-TIE — every flip is the other arm's runner-up at a margin below one bf16 ULP of that"
              " row's top logit, i.e. the two candidates were not distinguishable in the format they were"
              " computed in. That is the multi-row-body-vs-1-row-step form gap `ACCEPT-PLAN.md` already"
              " carries, arriving as a token flip at depth — not a bookkeeping bug.")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(
            {"first_divergence": first_div, "first_pair": first_pair, "flips": flips,
             "agrees": agrees, "bf16_rel_ulp": BF16_REL_ULP}, indent=2) + "\n")
        print(f"json: {args.json}")
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
