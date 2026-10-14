#!/usr/bin/env python3
"""Gate the §05 gated full-attention decode step (k = 1) at every pinned decode step.

Nothing here is synthesized. Every input comes from the pins — `q_proj.out`, `k_proj.out`,
`v_proj.out`, `cache_keys`, `cache_values` (cut back to the pre-append shape by `reference.py`,
which first proves the cut is right and the wrong readings wrong) — and every expectation is a pin:

    A1  attn.q_rope            §05.1 de-interleave + §05.2 zero-centred norm + §05.3 partial rope
    A2  attn.k_rope            the same, for the two KV heads
    A3  attn.v                 §05.7: the value this step appends, read back out of the cache
    A4  cache_keys             the whole valid cache: the read side plus the appended key lane
    A5  cache_values           the same for values
    A6  attn_out_pre_gate      §05.5 GQA attention with (max,sum) softmax — the headline
    A7  gated                  §05.6 `a * sigmoid(gate)`, the operand o_proj consumes
    A8  rope_cs                the kernel's own cos/sin table for this position, bit for bit

A1-A5 and A8 are required BIT-EXACT (they are roundings and index math, not reductions); A6/A7 get
the shared §10.1 judgement through `prototype/bf16_ulp.py` inside the cancellation band
`floor_probe.py` measures for THIS op — softmax has a different floor from a dot product, and
`proj_gemv`'s number is only a starting point, printed as "tightest band that would pass" so the
margin is on the record instead of inherited.

`--negative` injects kernel bugs one at a time, each with a declared must-catch and must-not-catch
stage set (the `mlp_gemv`/`proj_gemv` rule: a negative that cannot fail is worse than none, and one
that fails everywhere is measuring nothing). `--long` adds the prefill-derived long-cache cases,
which are the only pinned data in the repo with more than ~25 lanes. `--cross` re-checks the two
launch plans against each other bit-for-bit.

`--softmax online` is the route the ENGINE takes above the score staging (§07.3, and the engine's
launcher selects it at `lanes > MAXSC`), and it is judged differently, by a band §07.3 pre-authorized
rather than one this gate invented: see `ONLINE_*` below for what is held, what is demoted to a
reported number, and the anti-vacuity that stops an online-labelled run from silently being the
two-pass form. The two commands P7 re-gates are:

    .venv/bin/python prototype/attn_decode/run.py --softmax online --split 2 --long --negative
    .venv/bin/python prototype/attn_decode/run.py --softmax online --split 4 --long --negative

    .venv/bin/python prototype/attn_decode/run.py [--fixture dec8] [--layers all] [--steps all]
                                                 [--launch staged|fused] [--split N]
                                                 [--softmax online|full] [--long] [--negative]
                                                 [--no-verify] [--no-cross]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype"))
sys.path.insert(0, str(ROOT / "prototype" / "attn_decode"))
import bf16_ulp                                    # noqa: E402
import reference as R                              # noqa: E402

BIN = Path(__file__).resolve().parent / "attn_decode"
WORK = Path("/tmp/attn_decode")
# The cancellation band, sized for THIS op rather than inherited. Two measurements set it:
#
#   * floor_probe.py reorders the reference's own arithmetic (score dot in three orders, softmax
#     denominator forwards and reversed, sigmoid/rope evaluated two ways) and gets 0 of 15592 scores
#     and 2 of 108544 output elements moved, with no large element among them: the ordering floor
#     here is far BELOW the projections' 3e-5, so borrowing their band would have been too loose.
#   * run.py prints, from its own 1764 comparisons (3.6M elements), the tightest band everything
#     would still pass at: 1.5e-4, from one element of `gated` at 2 ULP at cache length 192. That
#     is the device libm's 1-ULP transcendental class (floor_probe's F3b measures its rate: the
#     kernel's rope table differs from torch's in 2 of 128 entries at one position in 68).
#
# 2e-4 clears the measured offender with ~1.3x margin. The shared §10.1 3e-5 would fail one element
# in 3.6 million with no bug present. See results.md for the table and floor_probe.py for the floors.
CANCEL_REL = 2e-4
NDIFF_PCT_MAX, NDIFF_ABS_MIN = 2.0, 1
COS_MIN, MEANREL_MAX = 0.9999, 1e-2
LONG_ROWS = [0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 255, 256, 257, 276]

# Cache lanes one streaming tile covers, read out of the kernel that uses it (the same pattern
# `engine/gate.py` uses for MAXSC: the constant belongs to the file that enforces it, and a copy here
# would drift). `attn_core` scores `TILE` lanes per pass of its tile loop, so a block that streams
# `ceil(lvalid / split)` lanes runs ceil(lvalid / split / TILE) tiles — and one tile is not a
# streaming reduction at all, which is what the negative harness has to respect.
KERNEL_TILE = int(re.search(r"#define\s+TILE\s+(\d+)",
                            (Path(__file__).resolve().parent
                             / "attn_decode_kernels.h").read_text()).group(1))

# The two stages the softmax reduction produces. Nothing else this gate judges is touched by the
# reduction ORDER: q_rope/k_rope/rope_cs are rope norm+transcendentals, gate_vec is a de-interleave,
# cache_keys/cache_values/v_append are copies and index math.
SOFTMAX_STAGES = {"attn_pre_gate", "gated"}

# §07.3's PRE-AUTHORIZED band for the online (blocked `(max,sum)`) route — cited, not invented. The
# spec's sentence: "When that path is built it must be gated on a cos/meanrel band — §10 G6's
# thresholds are already of that shape — and not on agreement with the pins, which it cannot reach."
# §10 G6 (specification/10-reference-dumps-and-gates.md:40) states that shape as `sampled ids exact;
# decode logits cos >= 0.9999`, and this file's COS_MIN/MEANREL_MAX already ARE those published
# numbers (0.9999 / 1e-2, §10.1), so the band introduces no new constant. What changes is which TEST
# a stage is held to, and only for the two stages in SOFTMAX_STAGES:
#
#   held    cos >= COS_MIN and meanrel <= MEANREL_MAX against the pins (aggregate agreement)
#   exact   unchanged, and bit-exact at every length: the copies, the de-interleave, the rope stages
#   demoted ndiff and the 1-ULP large-element count, i.e. PIN AGREEMENT. §07.3 measured why they are
#           unreachable at any tile size — §05.5 rounds the *normalised* weight before P·V and a
#           streaming row cannot round a weight it has not divided yet — so holding a split-K row to
#           them would fail correct code. Measured here, and unchanged by P7's re-run: 26.2 % of
#           `attn_pre_gate` differs, 6.2 % of those large, worst-case cos 0.99999134, meanrel <= 1.8e-3
#           (results.md §3). Those counts are still PRINTED for every online run; they are demoted
#           from a verdict to a number on the record, which is not the same thing as dropped.
#
# The ids half of G6 is discharged one level up, by `engine/gate.py` section (a) on the only
# >512-lane fixture (`wide520f`): this op has no tokens to sample.
#
# Anti-vacuity travels with the band, because a loosened test that the wrong branch could satisfy is
# no test at all. Two of them, both mechanical:
#   1. `--neg nomax` (A-17's un-rescaled running max) exists ONLY in the online branch of
#      `attn_core`, so injecting it under the run's OWN knobs must move the output; under `--softmax
#      full` it is inert. `run_negatives` now injects it at the run's `--split`, so it proves the
#      streaming accumulation ran in the split blocks, not just at split 1.
#   2. An `--softmax online` run whose reduction stages are bit-identical to the pins on every case
#      is not the online route (it moved 26 % of them for §07.3). Checked and printed as the
#      `anti-vacuity` line; a run that moved nothing FAILS.
# The merge itself cannot go unobserved either: with `split > 1` the core writes ONLY the `part`
# buffer, so if `attn_combine` did not run the output would be uninitialised memory and no aggregate
# could come near 0.9999.

# Which stages are pure index math and copies (must be BIT-EXACT: one wrong element is a bug) and
# which are floats produced by a reduction or a transcendental (judged by the §10.1 criteria, with
# the band floor_probe.py measures for THIS op). Everything except the cache copies and the gate
# de-interleave falls in the second class: rope calls cosf/sinf/powf, the score dot is a 256-term
# F32 sum in a different order than torch's, and the softmax is a reduction over the row.
EXACT_STAGES = {"cache_keys", "cache_values", "v_append", "gate_vec"}

# name -> (why it is a plausible engine bug, which softmax form it lives under, must-catch stages,
#         must-not-catch stages). "pinned" = injected under the form the pins pin down; "online" =
# the bug only exists in the streaming reduction, so injecting it there is the only way to see it.
NEG_KERNEL = {
    "qswap": ("q_proj's per-head halves swapped (§05.1)", "pinned",
              ["q_rope", "gate_vec", "attn_pre_gate", "gated"],
              ["k_rope", "v_append", "cache_keys", "cache_values", "rope_cs"]),
    "gatehalf": ("gate read as the contiguous second half, the [all q | all gate] reading (§05.1)",
                 "pinned", ["gate_vec", "gated"],
                 ["q_rope", "k_rope", "v_append", "cache_keys", "cache_values", "attn_pre_gate",
                  "rope_cs"]),
    "ropefull": ("rope over all 256 dims in (i,i+128) pairs instead of the first 64 (§05.3)",
                 "pinned", ["q_rope", "k_rope", "cache_keys", "attn_pre_gate", "gated"],
                 ["v_append", "cache_values", "gate_vec", "rope_cs"]),
    "ropenone": ("no rope at all", "pinned",
                 ["q_rope", "k_rope", "cache_keys", "attn_pre_gate", "gated"],
                 ["v_append", "cache_values", "gate_vec", "rope_cs"]),
    "gqaoff": ("kappa(i) off by one: query head i reads KV head (i/4+1) mod 2 (§05.5)", "pinned",
               ["attn_pre_gate", "gated"],
               ["q_rope", "k_rope", "v_append", "cache_keys", "cache_values", "gate_vec",
                "rope_cs"]),
    "nomax": ("(max,sum) state never rescaled when the running max moves (A-17 / §07.3)",
              "online", ["attn_pre_gate", "gated"],
              ["q_rope", "k_rope", "v_append", "cache_keys", "cache_values", "gate_vec",
               "rope_cs"]),
    "carry": ("the row seeded from the PREVIOUS query's (max,sum,acc) (§07.3's own failure)",
              "online", ["attn_pre_gate", "gated"],
              ["q_rope", "k_rope", "v_append", "cache_keys", "cache_values", "gate_vec"]),
    "readpast": ("attend to `cap` lanes instead of `lvalid` — the causal mask is gone", "pinned",
                 ["attn_pre_gate", "gated"],
                 ["q_rope", "k_rope", "v_append", "cache_keys", "cache_values", "gate_vec",
                  "rope_cs"]),
    "dropcurrent": ("attend to lanes < pos: the appended token is forgotten (§05.7)", "pinned",
                    ["attn_pre_gate", "gated"],
                    ["q_rope", "k_rope", "v_append", "cache_keys", "cache_values", "gate_vec"]),
    "dimhalf": ("the score dot accumulates 128 of the 256 head dims", "pinned",
                ["attn_pre_gate", "gated"],
                ["q_rope", "k_rope", "v_append", "cache_keys", "cache_values", "gate_vec",
                 "rope_cs"]),
    # The two slot bugs are invisible to the CURRENT query on purpose: the core reads the lane it
    # appends from shared memory rather than round-tripping it through the cache, so a wrong slot
    # first shows up in the NEXT token's read side. run.py cannot express that — each pinned case
    # rebuilds its own cache — so these two are declared against the cache stage only, and the
    # limit is stated instead of being papered over with a must-catch that does not hold.
    "slotk": ("the key appended to the wrong cache slot (§05.7)", "pinned", ["cache_keys"],
              ["q_rope", "k_rope", "v_append", "cache_values", "gate_vec", "rope_cs",
               "attn_pre_gate", "gated"]),
    "slotv": ("the value appended to the wrong cache slot (§05.7)", "pinned",
              ["cache_values", "v_append"],
              ["q_rope", "k_rope", "cache_keys", "gate_vec", "rope_cs", "attn_pre_gate", "gated"]),
}


def compare(got, ref, **kw) -> dict:
    kw.setdefault("cancel_rel", CANCEL_REL)
    return bf16_ulp.report(got, ref, **kw)


def fmt(c: dict) -> str:
    return ("ndiff %5d/%d (%.3f%%) max %d ULP | over-tol large: %d (worst %d ULP at |ref|/max"
            " %.1e) | cos %.8f meanrel %.2e | in-band %d (tightest band that would pass: %.1e)"
            % (c["ndiff"], c["n"], 100.0 * c["ndiff"] / c["n"], c["maxulp"], c["n_bad"],
               c["worst_bad_ulp"], c["worst_bad_rel"], c["cos"], c["meanrel"], c["n_band"],
               c["worst_off_rel"]))


def ndiff_limit(n: int) -> float:
    return max(NDIFF_PCT_MAX * n / 100.0, NDIFF_ABS_MIN)


def judge(tag: str, c: dict, fails: list[str], worst: dict, exact=False,
          pin_agreement=True) -> None:
    """`exact` stages are index math and roundings, not reductions: one element out of place is a
    bug, not arithmetic noise, so they get a zero-element allowance instead of the §10.1 one.

    `pin_agreement=False` is §07.3's online route (see the ONLINE band note at COS_MIN): the stage is
    a reduction in an order the pins cannot have, so ndiff and the 1-ULP large-element count are
    REPORTED and only the aggregate pair is judged. Every other call — the pinned form, and every
    non-reduction stage of the online form — is judged by all three as it always was.
    """
    key = tag.split()[-1]
    prev = worst.setdefault(key, [0, 0, 0, 0.0, 0])
    prev[0] = max(prev[0], c["ndiff"])
    prev[1] = max(prev[1], c["maxulp"])
    prev[2] = max(prev[2], c["n_bad"])
    prev[3] = max(prev[3], c["worst_off_rel"])
    prev[4] = max(prev[4], c["n"])
    if exact:
        if c["ndiff"]:
            fails.append("%s: %d of %d elements differ (this stage is exact arithmetic)"
                         % (tag, c["ndiff"], c["n"]))
        return
    if pin_agreement:
        if c["n_bad"]:
            fails.append("%s: %d large element(s) differ by >1 ULP (worst %d ULP at |ref|/max %.1e)"
                         % (tag, c["n_bad"], c["worst_bad_ulp"], c["worst_bad_rel"]))
        if c["ndiff"] > ndiff_limit(c["n"]):
            fails.append("%s: %d elements differ (limit %.1f)" % (tag, c["ndiff"], ndiff_limit(
                c["n"])))
    if c["cos"] < COS_MIN or c["meanrel"] > MEANREL_MAX:
        fails.append("%s: cos %.8f meanrel %.2e outside G3i/§10.1" % (tag, c["cos"],
                                                                      c["meanrel"]))


def judge_neg(c: dict) -> bool:
    """True when the comparison violates the same criteria the clean gate applies."""
    return bool(c["n_bad"] or c["ndiff"] > ndiff_limit(c["n"]) or c["cos"] < COS_MIN
                or c["meanrel"] > MEANREL_MAX)


def run_bin(args: list[str], label: str, quiet=True) -> dict:
    proc = subprocess.run([str(BIN)] + args, capture_output=True, text=True)
    if proc.returncode:
        sys.stdout.write(proc.stdout + proc.stderr)
        raise SystemExit("kernel failed: %s" % label)
    if not quiet:
        for line in proc.stdout.splitlines():
            print("      | " + line)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return dict(kv.split("=", 1) for kv in line.split()[1:])
    raise SystemExit("no RESULT line:\n" + proc.stdout)


# ------------------------------------------------------------------ cases and stages
def write_plan(path: Path, lines) -> Path:
    with open(path, "w") as f:
        for fields in lines:
            f.write(" ".join(str(x) for x in fields) + "\n")
    return path


def stages_for(case: dict, outdir: Path) -> list[tuple]:
    """(stage, kernel output, pinned reference) for one case, in report order. Whether a stage is
    bit-exact or judged by the §10.1 criteria is `EXACT_STAGES`, not a flag here.

    The cache comparisons cover the valid lanes only — for a WAVE case that is the LAST row's lane
    count, `lvalid + nrows - 1`, because every earlier row's lanes are a prefix of the same cache.
    Lanes past that are `reference.POISON` slack the engine would never read, and comparing them would
    make every clean run fail.

    A wave case's refs are already stacked in row order and flattened the way the kernel writes them,
    so each stage is one flat comparison; `v_append` is the wave's own appended span, which is the
    §05.7 check for every row of it at once.
    """
    ref, cap = case["ref"], case["cap"]
    nrows = case.get("nrows", 1)
    n = case["lvalid"] + nrows - 1
    kc = R.read_out(outdir, "cache_keys", 2 * cap * R.DH).reshape(2, cap, R.DH)
    vc = R.read_out(outdir, "cache_values", 2 * cap * R.DH).reshape(2, cap, R.DH)
    cos, sin = R.rope_cs(torch.tensor([case["pos"]]), None)
    if nrows > 1:
        p0, p1 = case["span"]
        gate = case["qg"].reshape(nrows, R.N_Q, 2 * R.DH)[:, :, R.DH:].reshape(-1)
        return [
            ("q_rope", R.read_out(outdir, "q_rope", R.N_Q * R.DH * nrows),
             ref["q_rope"].reshape(-1)),
            ("k_rope", R.read_out(outdir, "k_rope", R.N_KV * R.DH * nrows),
             ref["k_rope"].reshape(-1)),
            ("v_append", vc[:, p0:p1 + 1].reshape(-1), ref["v"].reshape(-1)),
            ("cache_keys", kc[:, :n].reshape(-1), ref["cache_keys"].reshape(-1)),
            ("cache_values", vc[:, :n].reshape(-1), ref["cache_values"].reshape(-1)),
            ("gate_vec", R.read_out(outdir, "gate_vec", R.N_Q * R.DH * nrows), gate),
            # The one cos||sin table this launch wrote is row 0's; the per-row positions are pinned by
            # q_rope/k_rope being BIT-EXACT at every row of the wave, which is the stronger claim.
            ("rope_cs", R.read_out(outdir, "rope_cs", 2 * R.ROT),
             torch.cat([cos.to(torch.bfloat16), sin.to(torch.bfloat16)]).reshape(-1)),
            ("attn_pre_gate", R.read_out(outdir, "attn_pre_gate", R.N_Q * R.DH * nrows),
             ref["attn_pre_gate"].reshape(-1)),
            ("gated", R.read_out(outdir, "gated", R.N_Q * R.DH * nrows), ref["gated"].reshape(-1)),
        ]
    return [
        ("q_rope", R.read_out(outdir, "q_rope", R.N_Q * R.DH), ref["q_rope"].reshape(-1)),
        ("k_rope", R.read_out(outdir, "k_rope", R.N_KV * R.DH), ref["k_rope"].reshape(-1)),
        ("v_append", vc[:, n - 1, :].reshape(-1), ref["v"][:, case["pos"]].reshape(-1)),
        ("cache_keys", kc[:, :n].reshape(-1), ref["cache_keys"].reshape(-1)),
        ("cache_values", vc[:, :n].reshape(-1), ref["cache_values"].reshape(-1)),
        ("gate_vec", R.read_out(outdir, "gate_vec", R.N_Q * R.DH),
         case["qg"].reshape(R.N_Q, 2 * R.DH)[:, R.DH:].reshape(-1)),
        ("rope_cs", R.read_out(outdir, "rope_cs", 2 * R.ROT),
         torch.cat([cos.to(torch.bfloat16), sin.to(torch.bfloat16)]).reshape(-1)),
        ("attn_pre_gate", R.read_out(outdir, "attn_pre_gate", R.N_Q * R.DH),
         ref["attn_pre_gate"].reshape(-1)),
        ("gated", R.read_out(outdir, "gated", R.N_Q * R.DH), ref["gated"].reshape(-1)),
    ]


def gate_cases(cases, outroot: Path, tag: str, fails, worst, ncmp, extra=(), quiet=True,
               online_route=False, band=None, print_cases=True):
    """Run one plan and judge every stage of every case; returns the raw reports per stage.

    `online_route` is §07.3's split-K/streaming form under test: it relaxes pin agreement for
    `SOFTMAX_STAGES` only (the `pin_agreement` argument to `judge`), and `band` — a dict owned by the
    caller — collects the aggregate numbers that band is made of, plus whether ANY case moved, which
    is the anti-vacuity that keeps an online-labelled run from being the two-pass form in disguise.

    `print_cases=False` is for a set whose per-case lines would bury the report (the 277 one-row cases
    the wave stage compares itself against); everything else about the run is identical.
    """
    lines = [R.write_case(c, WORK / "in", outroot / c["name"]) for c in cases]
    plan = write_plan(WORK / ("plan_%s.txt" % tag), lines)
    run_bin(["--mode", "once", "--plan", str(plan)] + list(extra), tag, quiet=quiet)
    per: dict[str, list[dict]] = {}
    for c in cases:
        row = []
        for name, got, ref in stages_for(c, outroot / c["name"]):
            if (c.get("nrows", 1) > 1 and name in SOFTMAX_STAGES and not online_route):
                # A WAVE's reduction stages are judged one ROW at a time by `wave_row_bad`, under
                # exactly the tolerance a single-row case gets. Judged flat, §10.1's ndiff limit would
                # be diluted by the row count and the verdict would no longer be comparable with the
                # one-row-per-launch run it is being compared against. Under `online_route` these stages
                # are band-judged (aggregates), so flat is the right shape there.
                continue
            rep = compare(got, ref)
            band_stage = online_route and name in SOFTMAX_STAGES
            judge("%s %s" % (c["name"], name), rep, fails, worst, exact=(name in EXACT_STAGES),
                  pin_agreement=not band_stage)
            per.setdefault(name, []).append(rep)
            ncmp.append("%s.%s" % (c["name"], name))
            if band is not None and band_stage:
                # [worst cos, max meanrel, differing elements, elements, cases that moved]
                b = band.setdefault(name, [1.0, 0.0, 0, 0, 0])
                b[0] = min(b[0], rep["cos"])
                b[1] = max(b[1], rep["meanrel"])
                b[2] += rep["ndiff"]
                b[3] += rep["n"]
                b[4] += 1 if rep["ndiff"] else 0
            row.append("%s %d/%d" % (name, rep["ndiff"], rep["n"]))
        if print_cases:
            print("  %-16s pos %3d lvalid %3d cap %3d rows %3d | %s"
                  % (c["name"], c["pos"], c["lvalid"], c["cap"], c.get("nrows", 1), " ".join(row)))
    return per


def cross_variant(label, knobs, cases, base, alt, fails, ncmp, gate_it=True, online_route=False):
    """Re-run the same cases under a different launch plan / softmax form / split, and report what
    moved. Two different questions get answered here:

      * bit-identity (the caller checks the return value): the fused launch recomputes its own norm
        and rope instead of staging them, so it MUST produce the same bits. Anything else means one
        of the two plans is wrong, not that they differ.
      * gate criteria (checked here): a different REDUCTION order — the streaming softmax, or a
        split row — cannot be bit-identical to the pinned one, so the question is whether it stays
        inside the same 1-ULP/aggregate gate. `ndiff` is printed, not judged.
    """
    lines = [R.write_case(c, WORK / "in", alt / c["name"]) for c in cases]
    plan = write_plan(WORK / ("plan_x_%s.txt" % label.replace(" ", "_")), lines)
    run_bin(["--mode", "once", "--plan", str(plan)] + knobs, label)
    agg = {}
    bits = 0
    for c in cases:
        # Bit-identity is measured on the WHOLE file the kernel wrote, not on the compared slice:
        # the cache files are [2][cap][256] and the slice is per-lane, so slicing before comparing
        # would compare differently ordered bytes and report a difference that is not there.
        for stage, n in (("q_rope", R.N_Q * R.DH), ("k_rope", R.N_KV * R.DH),
                         ("gate_vec", R.N_Q * R.DH), ("rope_cs", 2 * R.ROT),
                         ("attn_pre_gate", R.N_Q * R.DH), ("gated", R.N_Q * R.DH),
                         ("cache_keys", 2 * c["cap"] * R.DH), ("cache_values", 2 * c["cap"] * R.DH)):
            bits += int((R.read_out(base / c["name"], stage, n)
                         != R.read_out(alt / c["name"], stage, n)).sum())
        for name, got, ref in stages_for(c, alt / c["name"]):
            if name == "v_append":
                continue        # a slice of cache_values, which is compared on its own
            rep = compare(got, ref)
            ncmp.append("%s.%s" % (c["name"], name))
            a = agg.setdefault(name, [0, 0, 1.0, 0.0, 0, 0])
            a[0] += rep["ndiff"]
            a[1] = max(a[1], rep["maxulp"])
            a[2] = min(a[2], rep["cos"])
            a[3] = max(a[3], rep["meanrel"])
            a[4] += rep["n"]
            a[5] += rep["n_bad"]
    for name, (nd, mx, cs, mr, n, bad) in sorted(agg.items()):
        # The same §10.1 judgement the gate uses: a >1 ULP element only counts when it is a large
        # element (|ref| above the band). Raw max ULP across a whole tensor is exactly what the band
        # exists to excuse — a cancelling element a few BF16 steps off is arithmetic, not error. The
        # one exception is the online route's own reduction stages, which `judge` relaxes the same way
        # for the same reason (see the ONLINE band note at COS_MIN).
        pin_ok = bad == 0 or (online_route and name in SOFTMAX_STAGES)
        flag = "" if (pin_ok and cs >= COS_MIN and mr <= MEANREL_MAX) else (
            "   <-- OUTSIDE GATE" if gate_it else "   (reported: a different policy, not a bug)")
        print("  %-16s %-14s %7d/%d differ (%.2f%%) max %5d ULP (large-element %d)  cos %.8f"
              "  meanrel %.2e%s" % (label, name, nd, n, 100.0 * nd / n, mx, bad, cs, mr, flag))
        if flag and gate_it:
            fails.append("%s: stage %s outside the gate (%d large elements over 1 ULP, cos %.8f,"
                         " meanrel %.2e)" % (label, name, bad, cs, mr))
    return bits


def long_rows(spec: str, dl: Path) -> list[int]:
    """Which pinned prefill rows `--long` prices. `sample` is the shipped boundary list; `all` is every
    row the fixture pins (277 for p300), which is what separates a real regression from the sample
    happening to miss it — at ~30 s for one layer, a diagnosis cost worth knowing the answer to.
    """
    if spec == "sample":
        return list(LONG_ROWS)
    if spec == "all":
        return list(range(R.prefill_tokens(dl, R.ATTN_LAYERS[0])))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def run_plan(cases, outroot: Path, tag: str, extra) -> Path:
    """Write the case files, run one `--mode once` pass, return the output root."""
    lines = [R.write_case(c, WORK / "in", outroot / c["name"]) for c in cases]
    plan = write_plan(WORK / ("plan_%s.txt" % tag), lines)
    run_bin(["--mode", "once", "--plan", str(plan)] + list(extra), tag)
    return outroot


# Stages a wave launch writes per row, with the element count one row of each occupies. Same numbers
# the kernel's ROW_H / ROW_KV strides use, taken from the reference's own constants.
WAVE_STAGE_ROWS = (("q_rope", R.N_Q * R.DH), ("k_rope", R.N_KV * R.DH), ("gate_vec", R.N_Q * R.DH),
                   ("attn_pre_gate", R.N_Q * R.DH), ("gated", R.N_Q * R.DH))
# The two of those that are reductions, i.e. the only ones whose bits a softmax policy can move.
WAVE_RED_STAGES = WAVE_STAGE_ROWS[3:]


def wave_row_divergences(dl: Path, layer: int, chunks, stacked: bool) -> set:
    """{(stage, absolute row)} for every prefill row whose reduction output breaks the shipped §10.1
    rule — a large element more than 1 ULP off, or more differing elements than §10.1 allows a
    comparison of that size.

    `chunks` is `[(case dir, [absolute rows in launch order]), ...]`; `stacked` says whether each case
    directory holds all of its rows (one wave launch) or one row (one launch per row). The rule and
    the tolerance are the ones `judge` applies, and the reference is the same pin either way, so the
    two calls of this function differ ONLY in how the rows were launched: any difference between the
    two sets is a difference P8 introduced, and nothing else can produce it.
    """
    pre_gate = R.pin(dl, "L%02d.attn.attn_out_pre_gate" % layer).reshape(-1, R.N_Q, R.DH)
    qproj = R.pin(dl, "L%02d.q_proj.out" % layer).reshape(-1, R.N_Q * 2 * R.DH)
    bad = set()
    for stage, elems in WAVE_RED_STAGES:
        for dirdir, rows in chunks:
            flat = R.read_out(dirdir, stage, elems * (len(rows) if stacked else 1))
            for i, p in enumerate(rows):
                ref = pre_gate[p].reshape(-1)
                if stage == "gated":
                    ref = ref * torch.sigmoid(
                        qproj[p].reshape(R.N_Q, 2 * R.DH)[:, R.DH:].reshape(-1))
                got = flat[i * elems:(i + 1) * elems] if stacked else flat
                rep = compare(got, ref)
                if rep["n_bad"] or rep["ndiff"] > ndiff_limit(rep["n"]):
                    bad.add((stage, p))
    return bad


def wave_stage(args, fails, worst, ncmp, online_route, band):
    """P8 (§06.3): a whole prefill wave of queries through ONE launch, gated three ways.

    1. **The wave against the same rows run ONE AT A TIME, bit for bit.** This is the claim
       `engine/submit_wave_body` leans on: P8 replaced one attention launch per token with one per
       layer per wave, and the five shipped fixtures must not move by a bit. The pins cannot make that
       claim on their own — two forms can each agree with a pin and still differ from each other — so
       it is measured directly, stage by stage and row by row.
    2. **Every row against its own pinned prefill row** (`L{ll}.attn.q_rope` / `k_rope` /
       `attn_out_pre_gate`, §10.5's pairing), under the same rule a single-row case gets. The spans
       that do not start at row 0 are the multi-wave carry: those lanes arrive from the pins exactly as
       a previous wave would have left them, so row `p0` has to read lanes it did not append. Any row
       this flags is then priced one at a time, so the report says whether the wave caused it.
    3. The online route's anti-vacuity, in the direction this stage needs: an `--softmax online` wave
       whose reduction stages are bit-identical to the `--softmax full` wave is a wave that never
       streamed. §07.3 measured 26 % of these elements move, so identity here is a bug report.
    """
    dl = ROOT / "data" / "refs" / args.long_fixture
    if not dl.exists():
        raise SystemExit("no long fixture %s (needed for --wave)" % dl)
    # A wave is per attention layer, exactly as in the engine: one launch serves the layer's rows.
    layers = (R.ATTN_LAYERS[:1] if len(R.ATTN_LAYERS) > 1 and args.layers == "all"
              else [int(x) for x in args.layers.split(",")])
    # A wave cannot be fused: the §05.7 append would race the next row's read (the host says so too),
    # so this stage always prices the staged plan.
    knobs = ["--launch", "staged", "--split", str(args.split), "--softmax", args.softmax]
    if args.tile_kernel:
        # Phase 2's A/B. Every core launch of this run goes through `attn_tile_wave`; the pins are the
        # same pins, so the comparison is the §5 contract itself: the tiled wave must reproduce, row by
        # row and bit for bit, what the shipped form writes against the dumps.
        knobs.append("--tile-kernel")
    print("\nP8 waves of %d rows from %s prefill rows (one launch per layer, positions per row, "
          "staged plan): %s" % (args.wave_rows, args.long_fixture, " ".join(knobs)))
    for layer in layers:
        seq = R.prefill_tokens(dl, layer)
        spans = R.wave_spans(seq, args.wave_rows)
        rows_here = [r for p0, p1 in spans for r in range(p0, p1 + 1)]
        wcases = R.wave_cases(dl, layer, spans)
        for c in wcases:
            p0, p1 = c["span"]
            # The operand rebuild, at the two ends of the span: the pairing this whole gate stands on
            # (§10.5) is a per-row fact, and checking both ends covers the span's positions without
            # re-running the torch chain once per row.
            for probe in R.prefill_cases(dl, layer, [p0, p1]):
                for ok, line in R.verify(dl, probe):
                    if not ok:
                        fails.append("verify %s (span %s-%s): %s" % (probe["name"], p0, p1, line))
        wchunks = [(WORK / "out_wave" / c["name"], list(range(c["span"][0], c["span"][1] + 1)))
                   for c in wcases]
        gate_cases(wcases, WORK / "out_wave", "wave", fails, worst, ncmp, extra=knobs,
                   online_route=online_route, band=band)

        # Claim 1: the same rows, one launch each, must produce the same bits.
        per_row = R.prefill_cases(dl, layer, rows_here)
        gate_cases(per_row, WORK / "out_wave_rows", "wave-row", [], worst, ncmp, extra=knobs,
                   online_route=online_route, band=None, quiet=True, print_cases=False)
        rchunks = [(WORK / "out_wave_rows" / ("prefill%03d.L%02d" % (p, layer)), [p])
                   for p in rows_here]
        bits = 0
        for wdir, wrows in wchunks:
            for stage, row_elems in WAVE_STAGE_ROWS:
                got_wave = R.read_out(wdir, stage, row_elems * len(wrows))
                for r, p in enumerate(wrows):
                    one = R.read_out(WORK / "out_wave_rows" / ("prefill%03d.L%02d" % (p, layer)),
                                     stage, row_elems)
                    bits += int((got_wave[r * row_elems:(r + 1) * row_elems] != one).sum())
        rows_done = len(rows_here)
        print("  wave-vs-one-row (L%02d): %d elements across %d rows x %d stages differ"
              % (layer, bits, rows_done, len(WAVE_STAGE_ROWS)))
        if bits:
            fails.append("wave L%02d: %d elements differ between one launch over the wave and one "
                         "launch per row. They must be bit-identical: P8 changes how a wave is "
                         "submitted, not what is computed (this is the claim the five shipped "
                         "fixtures' bit-identity rests on)" % (layer, bits))
        else:
            print("    -> a wave of %d rows is BIT-IDENTICAL to %d single-row launches (L%02d)"
                  % (rows_done, rows_done, layer))

        # Claim 2: the pins. Row by row, and attributed: a row that misses the shipped rule in a wave
        # and also misses it one row at a time is this op's long-cache floor, which P8 neither caused
        # nor fixes; a row that misses ONLY in a wave is P8's and is fatal.
        if not online_route:
            bad_wave = wave_row_divergences(dl, layer, wchunks, stacked=True)
            bad_rows = wave_row_divergences(dl, layer, rchunks, stacked=False)
            only_wave = sorted(bad_wave - bad_rows)
            print("  wave-vs-pins (L%02d): %d of %d (stage, row) reduction comparisons over the "
                  "shipped §10.1 rule; one-row launches on the same rows: %d; wave-only: %d"
                  % (layer, len(bad_wave), 2 * rows_done, len(bad_rows), len(only_wave)))
            if bad_wave:
                print("    rows over the rule (wave): %s" % sorted(p for _, p in bad_wave))
            if bad_rows:
                print("    the same rows, priced ONE AT A TIME, miss identically — pre-existing at"
                      " these lane counts, not caused by P8. Reproduce with --long --long-rows %s"
                      % ",".join(str(p) for p in sorted(set(p for _, p in bad_rows))))
            for stage, p in only_wave:
                fails.append("wave L%02d row %d %s: over the §10.1 rule in a wave but NOT one row at"
                             " a time, so the wave itself caused it" % (layer, p, stage))

        # Claim 2b: the mixed-grid mechanism, priced. `--split 2` under `--softmax full` must not move
        # a single bit, because `blocks_for_row` hands every row that fits the staging EXACTLY ONE block
        # whatever the grid's depth — the upper blocks return before touching anything, and
        # `attn_combine` skips a row its core already finished. That is the shape the engine's launcher
        # produces for a wave whose widest row is above the staging (deep y, one-pass rows underneath),
        # so the mechanism is gated here even though no shipped fixture has a wave that needs it.
        if not online_route:
            run_plan(wcases, WORK / "out_wave_deep", "wave_deep",
                     ["--launch", "staged", "--split", "2", "--softmax", "full"])
            bits_deep = 0
            for wdir, wrows in wchunks:
                for stage, row_elems in WAVE_STAGE_ROWS:
                    a = R.read_out(wdir, stage, row_elems * len(wrows))
                    b = R.read_out(WORK / "out_wave_deep" / wdir.name, stage, row_elems * len(wrows))
                    bits_deep += int((a != b).sum())
            print("  wave at --split 2 under --softmax full (L%02d): %d elements moved vs --split 1"
                  " (%s)" % (layer, bits_deep,
                             "one block per row that fits, as designed" if bits_deep == 0 else
                             "a deeper grid changed the arithmetic"))
            if bits_deep:
                fails.append("wave L%02d: %d elements moved when the lane grid was deepened to 2 under"
                             " --softmax full. A row the staging fits must take exactly one block"
                             " (blocks_for_row); if it moved, the one-pass form is reading the split"
                             " grid and normalising per block — §07.3's trap, back again"
                             % (layer, bits_deep))

        # Claim 3: the wave's own anti-vacuity, online runs only.
        if online_route:
            run_plan(wcases, WORK / "out_wave_full", "wave_full",
                     ["--launch", "staged", "--split", "1", "--softmax", "full"])
            moved = 0
            for wdir, wrows in wchunks:
                for stage, row_elems in WAVE_RED_STAGES:      # the two reduction stages
                    a = R.read_out(wdir, stage, row_elems * len(wrows))
                    b = R.read_out((WORK / "out_wave_full" / wdir.name), stage, row_elems * len(wrows))
                    moved += int((a != b).sum())
            print("  wave anti-vacuity (L%02d): %d reduction elements differ from the same waves run"
                  " --softmax full (%s)" % (layer, moved,
                                            "streaming ran" if moved else "NOTHING MOVED"))
            if not moved:
                fails.append("anti-vacuity: the --softmax online waves produced output bit-identical "
                             "to --softmax full on every stage of every wave, which is what a wave "
                             "that never entered the streaming branch does (§07.3 measures 26 %% of "
                             "these elements moving)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--long-fixture", default="p300")
    ap.add_argument("--layers", default="all", help="comma list of attention layers, or 'all'")
    ap.add_argument("--steps", default="all", help="comma list of decodeN, or 'all'")
    ap.add_argument("--launch", default="staged", choices=["staged", "fused"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--softmax", default="full", choices=["online", "full"])
    ap.add_argument("--long", action="store_true", help="long-cache cases from the prefill pins")
    ap.add_argument("--wave", action="store_true",
                    help="P8: pinned prefill spans as WAVES (one launch per layer over many query "
                         "rows), each row judged against its own pin and the whole wave against the "
                         "same rows run one at a time")
    ap.add_argument("--long-rows", default="sample",
                    help="--long row list: 'sample' (the shipped boundary rows), 'all' (every pinned "
                         "prefill row), or a comma-separated row list")
    ap.add_argument("--wave-rows", type=int, default=64,
                    help="rows per wave for --wave (default 64, plan §2.1's chunk width)")
    ap.add_argument("--tile-kernel", action="store_true",
                    help="Phase 2: run the G x M tiled wave core (`attn_tile_wave`) instead of the "
                         "shipped per-(row,head) `attn_core`. Wave-only — the host refuses it with "
                         "--launch fused, --score-f32 or --neg carry. `run.py` takes the GPU lock "
                         "itself, so hold `tools/gpu_lock.sh` before invoking it (see AGENTS.md)")
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the operand-rebuild proof (it is the gate's own foundation)")
    ap.add_argument("--no-cross", action="store_true",
                    help="skip the fused-vs-staged bit-equality check")
    args = ap.parse_args()

    d = ROOT / "data" / "refs" / args.fixture
    if not d.exists():
        raise SystemExit("no fixture %s" % d)
    all_steps = sorted({p.name.split(".")[0] for p in d.glob("decode*.L03.*.npy")})
    steps = all_steps if args.steps == "all" else args.steps.split(",")
    layers = R.ATTN_LAYERS if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    if not BIN.exists():
        subprocess.run(["bash", str(BIN.parent / "build.sh")], check=True)
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed - the gate cannot be trusted")
    WORK.mkdir(parents=True, exist_ok=True)

    knobs = ["--launch", args.launch, "--split", str(args.split), "--softmax", args.softmax]
    print("fixture %s | %d layers x %d steps | launch %s split %d softmax %s | band |ref|/max <= %g"
          % (args.fixture, len(layers), len(steps), args.launch, args.split, args.softmax,
             CANCEL_REL))

    fails: list[str] = []
    worst: dict = {}
    ncmp: list[str] = []
    # The route under test, and the aggregate column the online band is judged on (None otherwise).
    online_route = args.softmax == "online"
    band: dict[str, list] = {} if online_route else None
    n_cases_run = 0
    cases = [R.decode_case(d, s, l) for l in layers for s in steps]

    # 1. The gate's own foundation. Without this every comparison below is against the wrong
    #    tensor, which is the trap §10.5 documents twice in this project.
    if not args.no_verify:
        print("\noperand + fixture rebuild (cache cut back to pre-append, pos = lvalid-1, "
              "rope policy):")
        nbad = 0
        for c in cases:
            for ok, line in R.verify(d, c):
                if not ok:
                    nbad += 1
                    print("    FAIL %s: %s" % (c["name"], line))
        print("    %d cases x %d claims checked, %d failures"
              % (len(cases), len(R.verify(d, cases[0])), nbad))
        if nbad:
            raise SystemExit("operand rebuild failed - stopping: a gate against the wrong tensor"
                             " verifies nothing")

    # 2. The gate.
    print("\nkernel stages vs the pins (%s, split %d, softmax %s%s):"
          % (args.launch, args.split, args.softmax,
             " — online route: the two softmax stages are judged on §07.3's pre-authorized"
             " cos/meanrel band, pin agreement reported not judged" if online_route else ""))
    gate_cases(cases, WORK / "out", "%s_%s_%d" % (args.launch, args.softmax, args.split), fails,
               worst, ncmp, extra=knobs, online_route=online_route, band=band)
    n_cases_run += len(cases)

    # 3. The other knobs. The two launch plans must agree BIT FOR BIT (one of them recomputes the
    #    stages the other stages, that is all); a different reduction ORDER cannot, so the streaming
    #    softmax and a split row are judged by the same gate instead and the gap is printed.
    if not args.no_cross:
        print("\ncross-checks against the gated run above (bit-identical for the launch plan,"
              " same 1-ULP gate for a different reduction order):")
        base, alt = WORK / "out", WORK / "out_cross"
        other = "online" if args.softmax == "full" else "full"
        bits = cross_variant("fused", ["--launch", "fused", "--split", str(args.split),
                                       "--softmax", args.softmax], cases, base, alt, fails, ncmp,
                             online_route=online_route)
        if bits:
            fails.append("the fused launch plan does not agree with the staged one bit-for-bit"
                         " (%d elements of %d differ across the whole output files) - one of the"
                         " two is wrong"
                         % (bits, sum(2 * c["cap"] * R.DH * 2 + 3 * R.N_Q * R.DH
                                       + R.N_KV * R.DH + 2 * R.ROT for c in cases)))
        cross_variant("softmax-" + other,
                      ["--launch", args.launch, "--split", "1", "--softmax", other],
                      cases, base, alt / other, fails, ncmp, gate_it=False)
        if args.split == 1:
            # Only the streaming form can be split at all: the reference form normalises the weight
            # matrix across the whole row, which one block per row is the only way to reproduce.
            cross_variant("split-4-online",
                          ["--launch", args.launch, "--split", "4", "--softmax", "online"],
                          cases, base, alt / "split4", fails, ncmp, gate_it=False)

    # 4. Long caches. The decode fixtures stop at 25 lanes.
    if args.long:
        dl = ROOT / "data" / "refs" / args.long_fixture
        if not dl.exists():
            raise SystemExit("no long fixture %s (needed for --long)" % dl)
        rows = long_rows(args.long_rows, dl)
        lc: list[dict] = []
        for layer in (layers[:1] if len(layers) > 1 else layers):
            lc += R.prefill_cases(dl, layer, rows)
        print("\nlong-cache cases from %s prefill rows (each row is a decode-shaped query at"
              " position p over lanes 0..p): %d cases, %d..%d lanes"
              % (args.long_fixture, len(lc), lc[0]["lvalid"], lc[-1]["lvalid"]))
        for c in lc:
            for ok, line in R.verify(dl, c):
                if not ok:
                    fails.append("verify %s: %s" % (c["name"], line))
        gate_cases(lc, WORK / "out_long", "long", fails, worst, ncmp, extra=knobs,
                   online_route=online_route, band=band)
        n_cases_run += len(lc)

    # 4b. The online band, on the record in one place: the numbers §07.3 pre-authorized the route to
    #     be judged by, and the anti-vacuity that says the streaming branch is the one that ran.
    if online_route:
        print("\n§07.3 online band (G6's shape, held against the pins; ndiff/1-ULP pin agreement"
              " demoted to a reported number because §05.5's normalised-weight round makes the pins"
              " unreachable at any tile size):")
        moved = 0
        for name, (cs, mr, nd, n, nc) in sorted(band.items()):
            print("    %-14s worst cos %.8f >= %.4f | max meanrel %.2e <= %.0e | %d/%d differ from"
                  " the pin (%.1f %%, reported, not judged) | %d case(s) moved"
                  % (name, cs, COS_MIN, mr, MEANREL_MAX, nd, n, 100.0 * nd / n, nc))
            moved = max(moved, nc)
        if moved == 0:
            fails.append("anti-vacuity: --softmax online --split %d produced output bit-identical to"
                         " the pins on EVERY case, which is what the two-pass form does (0 of 98 304"
                         " differ, results.md §3) and the online form provably cannot (§07.3 measured"
                         " 26 %% of elements moved). The streaming branch did not run"
                         % args.split)
        else:
            print("    anti-vacuity: the online branch ran — %d of %d case(s) moved against the pins"
                  " on at least one softmax stage (a two-pass run moves none). `--negative` proves it"
                  " again from the other side: `nomax` exists only in the online branch and is"
                  " injected at this run's own split" % (moved, n_cases_run))

    # 4c. P8: waves. Runs after `--long` because it reuses the same fixture, and before the negatives
    # so a wave failure is visible next to the row failures it mirrors.
    if args.wave:
        wave_stage(args, fails, worst, ncmp, online_route, band)

    # 5. Negatives.
    if args.negative:
        run_negatives(args, d, fails)

    print("\nworst BF16 ULP by stage -- gated (large elements only) | raw (also the cancelling"
          " elements floor_probe.py attributes to arithmetic, not error) | most elements any one"
          " comparison put inside the %g band | tightest band these results would pass at:"
          % CANCEL_REL)
    slack = None
    for k, (nd, raw, bad, minband, n) in sorted(worst.items()):
        print("    %-16s ndiff %6d   raw %5d ULP   over-tol %3d   in-band %8d   tightest band %s"
              % (k, nd, raw, bad, n, "%.1e" % minband if minband else "-"))
        if k in ("attn_pre_gate", "gated", "q_rope", "k_rope", "rope_cs") and minband:
            slack = min(slack or 1e9, CANCEL_REL / minband)
    if slack:
        print("    band in use |ref|/max <= %g; worst offender above it anywhere -> %.1fx of"
              " slack. floor_probe.py puts this op's own floor at the numbers it prints; a"
              " tighter band has to clear both." % (CANCEL_REL, slack))
    print("\n%d stage comparisons, %d failures" % (len(ncmp), len(fails)))
    if fails:
        print("FAIL")
        for f in fails[:40]:
            print("  -", f)
        return 1
    print("PASS: the attention decode step reproduces the reference's own stages.")
    return 0


def neg_cases(args, d: Path, layer: int, split: int) -> list[dict]:
    """The two cases one negative is injected into, WIDE ENOUGH for that split to have a streaming
    reduction at all.

    `nomax` is invisible inside a single tile: with one tile per block the subtraction constant cancels
    in the final divide, so the un-rescaled running max leaves the bits untouched. A block streams
    `ceil(lvalid / split)` lanes in tiles of `KERNEL_TILE`, and the two decode cases are 18/19 lanes —
    three tiles at split 1, one tile at split 4. Above split 2 the long-cache rows of `--long-fixture`
    are therefore the only pinned data wide enough, and they are used instead. A negative that cannot
    fail because the harness handed it one tile is the vacuous test this file's own rule forbids.
    """
    steps = sorted({p.name.split(".")[0] for p in d.glob("decode*.L%02d.*.npy" % layer)})[:2]
    cases = [R.decode_case(d, s, layer) for s in steps]

    def streams(lvalid: int) -> bool:
        return (lvalid + split - 1) // split > KERNEL_TILE

    if any(streams(c["lvalid"]) for c in cases):
        return cases
    dl = ROOT / "data" / "refs" / args.long_fixture
    if not dl.exists():
        raise SystemExit("--negative at --split %d needs pinned rows wider than %d lanes so a block"
                         " streams more than one tile, and %s (the only such data in the repo) is"
                         " missing — run with --long-fixture pointing at one"
                         % (split, split * KERNEL_TILE, dl))
    wide = [r for r in LONG_ROWS if streams(r + 1)]      # a row r attends lanes 0..r
    if not wide:
        raise SystemExit("--negative at --split %d needs rows wider than %d lanes; %s has none"
                         % (split, split * KERNEL_TILE, dl))
    return R.prefill_cases(dl, layer, wide[:2])


def run_negatives(args, d: Path, fails) -> None:
    """Inject one bug at a time and ask which stages' BITS moved, against a clean run under the
    identical knobs. Comparing against the pins instead would fold the softmax-policy gap into every
    injection (the first version did, and every negative "caught" the attention output no matter
    what it was given — a negative that fails everywhere measures the knob, not the bug).

    Two cases per negative: `carry` needs a previous query to leak out of, and one case would let
    every other injection pass by doing nothing."""
    layer = R.ATTN_LAYERS[0]
    print("\nnegative tests (rejected where declared, and nowhere else):")
    for name, (why, form, must, must_not) in NEG_KERNEL.items():
        # Injected at THIS run's split, not at a forced 1. That is what makes `nomax` the
        # anti-vacuity for a split row: the un-rescaled running max lives inside the online tile loop
        # each split block runs, so an injection that moves nothing at --split 4 means the split
        # blocks never ran the streaming accumulation. `carry` is the one exception the host itself
        # imposes — one softmax state per query row is undefined across split blocks.
        neg_split = 1 if name == "carry" else args.split
        cases = neg_cases(args, d, layer, neg_split)
        knobs = ["--launch", args.launch, "--split", str(neg_split),
                 "--softmax", args.softmax if form == "pinned" else "online"]
        dirs = {}
        for tag, extra in (("clean", []), ("neg", ["--neg", name])):
            out = WORK / "neg" / name / tag
            lines = [R.write_case(c, WORK / "in", out / c["name"]) for c in cases]
            plan = write_plan(WORK / ("plan_neg_%s_%s.txt" % (name, tag)), lines)
            run_bin(["--mode", "once", "--plan", str(plan)] + knobs + extra, "%s/%s" % (name,
                                                                                          tag))
            dirs[tag] = out
        caught = set()
        for c in cases:
            for stage, n in (("q_rope", R.N_Q * R.DH), ("k_rope", R.N_KV * R.DH),
                             ("gate_vec", R.N_Q * R.DH), ("rope_cs", 2 * R.ROT),
                             ("attn_pre_gate", R.N_Q * R.DH), ("gated", R.N_Q * R.DH),
                             ("cache_keys", 2 * c["cap"] * R.DH),
                             ("cache_values", 2 * c["cap"] * R.DH)):
                if int((R.read_out(dirs["clean"] / c["name"], stage, n)
                        != R.read_out(dirs["neg"] / c["name"], stage, n)).sum()):
                    caught.add(stage)
        # v_append is the appended lane of cache_values: it moves exactly when cache_values does,
        # so it is reported from those bytes rather than read as a file of its own.
        if "cache_values" in caught:
            caught.add("v_append")
        missing = [m for m in must if m not in caught]
        wrong = [m for m in must_not if m in caught]
        print("    --neg %-12s [%-6s] L%02d %-52s caught %-40s must-not: %s%s"
              % (name, form, layer, why, ",".join(sorted(caught)) or "NOTHING",
                 ",".join("%s=%s" % (m, "clean" if m not in caught else "MOVED")
                          for m in must_not) or "-",
                 "" if not wrong else "  <-- " + ",".join(wrong)))
        if missing:
            fails.append("negative %s was NOT rejected at %s - the gate is vacuous there"
                         % (name, ",".join(missing)))
        if wrong:
            fails.append("negative %s moved %s, which it should not touch"
                         % (name, ",".join(wrong)))


if __name__ == "__main__":
    raise SystemExit(main())
