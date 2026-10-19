#!/usr/bin/env python3
"""A0: derive the attention launch counts a traced pass shows, out of the launcher, and reconcile them.

WHAT QUESTION THIS ANSWERS. `prototype/gdn_walk/BET.md` §7 reports the traced 4,096-lane attention family as
`attn_slim_wave 48 / attn_fattn_wave 42 / attn_combine 42 / attn_prep 48` per pass and explicitly records that
nobody derived WHY 42 of 48 waves go through the streaming WMMA path. "The fattn kernel is slow per launch" and
"too many launches go through it" are different bets, and the next attention decision has to be between them, so
the split needs a rule and an arithmetic that closes — an unreconciled count is a finding, a fudged one is a bug.

THE RULE IT READS OUT (all three legs are `split`, and `split` is one expression):
  * `split_for(lanes_last)` (engine/kernels_attn.cpp) sizes the lane split from the WIDEST row of the wave;
  * `attn_prep` and `attn_slim_wave` are launched on EVERY wave body of a tiled wave, whatever `split` is;
  * `attn_fattn_wave` is launched only under `wave_fattn`, which requires `split >= 2`;
  * `attn_combine` is launched only `if (split > 1)`;
  * which ROWS the WMMA kernel owns inside a launched body is `fattn_covers` (op header), i.e.
    `blocks_for_row > 1`, i.e. `full && lanes_of(row) > MAXSC`.
So the 42 is not a policy about the WMMA form at all — it is "the first wave body of a prompt is a one-pass
wave", because `lvalid` starts at 1 and 512 rows of a 512-wide staging fit it exactly.

CPU-ONLY. It reads the enforcing source and asserts the enforcing expressions are still there; it launches
nothing. The constants (MAXSC, kMaxWaveRows, the full-attention layer list) are PARSED from the files that own
them, not typed, for the reason AGENTS.md records under "an admission over a fixture list typed from memory".

    python3 prototype/attn_wmma/a0_work_split.py [--lanes 4096] [--wave-rows N]

Exit 0 only if every count reconciles to the traced 48/48/42/42 at 4,096 lanes; the assertion is parameterised
by `--lanes` so a different width prints its own expectation instead of being forced into this one.
"""
import argparse
import math
import re
import sys

# The traced numbers to reconcile against, and where they came from (BET.md §7's table, which is §38's trace).
TRACED = {"attn_prep": 48, "attn_slim_wave": 48, "attn_fattn_wave": 42, "attn_combine": 42}

# (file, regex) pairs for every constant and every enforcing expression. The second column is what makes the
# script go red when the launcher changes under it, rather than continuing to describe the old launcher.
FACTS = {
    "MAXSC": ("prototype/attn_decode/attn_decode_kernels.h", r"#define MAXSC (\d+)"),
    "kMaxWaveRows": ("engine/decode_step.h", r"constexpr int kMaxWaveRows = (\d+)"),
    "layers": ("specification/00-notation-and-hyperparameters.md",
               r"`full_attention` \(§05\) at `([\d, ]+)`"),
}
ENFORCEMENTS = [
    # (label, file, text that must still be present)
    ("split = split_for(lanes_last)", "engine/kernels_attn.cpp",
     "const int split = split_for(lanes_last);"),
    ("split_for is ceil(lanes/MAXSC)", "engine/kernels_attn.cpp",
     "int split_for(int lanes) { return (lanes + MAXSC - 1) / MAXSC; }"),
    ("lanes_last = lvalid + rows - 1", "engine/kernels_attn.cpp",
     "const int lanes_last = a.lvalid + rows - 1;"),
    ("fattn needs split >= 2", "engine/kernels_attn.cpp",
     "const bool wave_fattn = a.wave_fattn != 0 && rows > 1 && a.wave_tile != 0 && split >= 2;"),
    ("attn_fattn_wave launch site", "engine/kernels_attn.cpp", "attn_fattn_wave<<<dim3(N_Q, split,"),
    ("attn_combine gated on split > 1", "engine/kernels_attn.cpp", "if (split > 1) {"),
    ("attn_combine launch site", "engine/kernels_attn.cpp", "attn_combine<<<dim3(N_Q, rows), THREADS>>>"),
    ("attn_prep on every wave body", "engine/kernels_attn.cpp", "attn_prep<<<dim3(N_Q + N_KV, rows), THREADS>>>"),
    ("attn_slim_wave on every tiled wave body", "engine/kernels_attn.cpp",
     "attn_slim_wave<<<dim3(N_KV, split, tile_rtiles_of(rows)), ATTN_TILE_THREADS>>>"),
    ("fattn_covers is blocks_for_row > 1", "prototype/attn_decode/attn_decode_kernels.h",
     "return a.fattn_cover != 0 && qrow < nrows_of(a) && blocks_for_row(a, qrow) > 1 ? 1 : 0;"),
    ("blocks_for_row's one-pass rule", "prototype/attn_decode/attn_decode_kernels.h",
     "if (a.full && lanes_of(a, qrow) <= MAXSC) return 1;"),
    ("lanes_of is lvalid + qrow", "prototype/attn_decode/attn_decode_kernels.h",
     "return a.lvalid + qrow;"),
    ("a wave body's lvalid is pos0 + 1", "engine/decode_step.cpp", "a.lvalid = pos0 + 1;"),
    ("a wave body is wave_cap rows wide", "engine/prefill.cpp", "const size_t wave_cap = (size_t)sc.wave_rows;"),
]


def read(path):
    with open(path) as fh:
        return fh.read()


def fact(name):
    path, pat = FACTS[name]
    m = re.search(pat, read(path))
    if not m:
        raise SystemExit("%s: no match for %r in %s — the constant moved, do not carry an old number"
                         % (name, pat, path))
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=4096)
    ap.add_argument("--wave-rows", type=int, default=None, help="default: kMaxWaveRows, the shipped ceiling")
    a = ap.parse_args()

    maxsc = int(fact("MAXSC"))
    wave_rows = a.wave_rows or int(fact("kMaxWaveRows"))
    attn_layers = [int(x) for x in fact("layers").split(",")]

    missing = ["%s: %s" % (label, path) for label, path, needle in ENFORCEMENTS if needle not in read(path)]
    if missing:
        raise SystemExit("the launcher no longer spells the rule this script derives:\n  " + "\n  ".join(missing))

    print("constants read, not typed:")
    print("  MAXSC                     %d   [prototype/attn_decode/attn_decode_kernels.h]" % maxsc)
    print("  kMaxWaveRows              %d   [engine/decode_step.h]" % wave_rows)
    print("  full_attention layers     %d   %s [specification/00 §00.3]"
          % (len(attn_layers), attn_layers))
    print("  all %d enforcing expressions matched the launcher source verbatim" % len(ENFORCEMENTS))

    # The wave bodies, exactly as `prefill_wave` cuts them: `wave_cap` rows at a time, `pos0` the running
    # prefix, `lvalid = pos0 + 1` (decode_step.cpp), and the widest row of a body is its LAST one.
    bodies = []
    begin = 0
    while begin < a.lanes:
        n = min(wave_rows, a.lanes - begin)
        pos0 = begin
        lvalid = pos0 + 1
        lanes_last = lvalid + n - 1
        split = (lanes_last + maxsc - 1) // maxsc
        # rows this WMMA form owns: `fattn_covers` = `blocks_for_row > 1` = (full && lanes_of > MAXSC),
        # and `lanes_of(row) = lvalid + row`.
        covered = sum(1 for r in range(n) if lvalid + r > maxsc)
        lanes_all = sum(lvalid + r for r in range(n))
        lanes_cov = sum(lvalid + r for r in range(n) if lvalid + r > maxsc)
        bodies.append(dict(body=len(bodies), rows=n, lvalid=lvalid, lanes_last=lanes_last, split=split,
                           covered=covered, lanes_all=lanes_all, lanes_cov=lanes_cov))
        begin += n

    print("\none 4,096-lane prompt is %d wave bodies of at most %d rows x %d attention layers = %d attention "
          "layers-worth of launches per kernel site"
          % (len(bodies), wave_rows, len(attn_layers), len(bodies) * len(attn_layers)))
    print("\n body rows  lvalid lanes_last split  fattn  combine  rows covered (lanes covered)")
    for b in bodies:
        print("  %4d %4d %6d %10d %5d  %5s  %7s  %4d/%-4d (%5.1f%%)"
              % (b["body"], b["rows"], b["lvalid"], b["lanes_last"], b["split"],
                 "yes" if b["split"] >= 2 else "NO", "yes" if b["split"] > 1 else "NO",
                 b["covered"], b["rows"], 100.0 * b["lanes_cov"] / b["lanes_all"]))

    counts = {
        "attn_prep": len(bodies) * len(attn_layers),
        "attn_slim_wave": len(bodies) * len(attn_layers),
        "attn_fattn_wave": sum(1 for b in bodies if b["split"] >= 2) * len(attn_layers),
        "attn_combine": sum(1 for b in bodies if b["split"] > 1) * len(attn_layers),
    }
    print("\nlaunches derived per pass (bodies x layers, the two split-gated legs dropping the bodies that "
          "cannot pay them):")
    ok = True
    for k in ("attn_prep", "attn_slim_wave", "attn_fattn_wave", "attn_combine"):
        want = TRACED.get(k)
        mark = "" if want is None else ("  <- traced %d %s" % (want, "MATCH" if want == counts[k] else "MISMATCH"))
        if want is not None and want != counts[k]:
            ok = False
        print("  %-16s %3d%s" % (k, counts[k], mark))

    rows_cov = sum(b["covered"] for b in bodies)
    rows_all = sum(b["rows"] for b in bodies)
    lanes_cov = sum(b["lanes_cov"] for b in bodies)
    lanes_all = sum(b["lanes_all"] for b in bodies)
    print("\ncovered share of the WMMA form, both denominators `fattn_echo` prints:")
    print("  rows %d/%d = %.1f%%   lanes(work) %d/%d = %.1f%%"
          % (rows_cov, rows_all, 100.0 * rows_cov / rows_all, lanes_cov, lanes_all,
             100.0 * lanes_cov / lanes_all))
    # `engine/kernels_attn.cpp`'s own comment claims "~87 % rows / ~99 % work" at 4,096 lanes. If the model of
    # the launcher were wrong this would not agree, and the agreement is with a number written by a different
    # author in a different file, which is what makes it a witness rather than a restatement.
    if a.lanes == 4096:
        rows_pct = 100.0 * rows_cov / rows_all
        work_pct = 100.0 * lanes_cov / lanes_all
        if not (86.5 <= rows_pct <= 87.5):
            ok = False
            print("  MISMATCH: row share %.2f%% is not the ~87%% `fattn_echo`'s comment claims" % rows_pct)
        if not (98.0 <= work_pct <= 99.9):
            ok = False
            print("  MISMATCH: work share %.2f%% is not the ~99%% `fattn_echo`'s comment claims" % work_pct)

    print("\n%s" % ("RECONCILED: the rule reproduces the traced counts." if ok else
                    "NOT RECONCILED: read the arithmetic above; do not fudge it."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
