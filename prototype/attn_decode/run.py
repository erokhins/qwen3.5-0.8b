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

    .venv/bin/python prototype/attn_decode/run.py [--fixture dec8] [--layers all] [--steps all]
                                                 [--launch staged|fused] [--split N]
                                                 [--softmax online|full] [--long] [--negative]
                                                 [--no-verify] [--no-cross]
"""
from __future__ import annotations

import argparse
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


def judge(tag: str, c: dict, fails: list[str], worst: dict, exact=False) -> None:
    """`exact` stages are index math and roundings, not reductions: one element out of place is a
    bug, not arithmetic noise, so they get a zero-element allowance instead of the §10.1 one."""
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

    The cache comparisons cover lanes [0, lvalid) only: lanes past that are `reference.POISON`
    slack the engine would never read, and comparing them would make every clean run fail.
    """
    ref, cap, n = case["ref"], case["cap"], case["lvalid"]
    kc = R.read_out(outdir, "cache_keys", 2 * cap * R.DH).reshape(2, cap, R.DH)
    vc = R.read_out(outdir, "cache_values", 2 * cap * R.DH).reshape(2, cap, R.DH)
    cos, sin = R.rope_cs(torch.tensor([case["pos"]]), None)
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


def gate_cases(cases, outroot: Path, tag: str, fails, worst, ncmp, extra=(), quiet=True):
    """Run one plan and judge every stage of every case; returns the raw reports per stage."""
    lines = [R.write_case(c, WORK / "in", outroot / c["name"]) for c in cases]
    plan = write_plan(WORK / ("plan_%s.txt" % tag), lines)
    run_bin(["--mode", "once", "--plan", str(plan)] + list(extra), tag, quiet=quiet)
    per: dict[str, list[dict]] = {}
    for c in cases:
        row = []
        for name, got, ref in stages_for(c, outroot / c["name"]):
            rep = compare(got, ref)
            judge("%s %s" % (c["name"], name), rep, fails, worst, exact=(name in EXACT_STAGES))
            per.setdefault(name, []).append(rep)
            ncmp.append("%s.%s" % (c["name"], name))
            row.append("%s %d/%d" % (name, rep["ndiff"], rep["n"]))
        print("  %-16s pos %3d lvalid %3d cap %3d | %s" % (c["name"], c["pos"], c["lvalid"],
                                                          c["cap"], " ".join(row)))
    return per


def cross_variant(label, knobs, cases, base, alt, fails, ncmp, gate_it=True):
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
        # exists to excuse — a cancelling element a few BF16 steps off is arithmetic, not error.
        flag = "" if (bad == 0 and cs >= COS_MIN and mr <= MEANREL_MAX) else (
            "   <-- OUTSIDE GATE" if gate_it else "   (reported: a different policy, not a bug)")
        print("  %-16s %-14s %7d/%d differ (%.2f%%) max %5d ULP (large-element %d)  cos %.8f"
              "  meanrel %.2e%s" % (label, name, nd, n, 100.0 * nd / n, mx, bad, cs, mr, flag))
        if flag and gate_it:
            fails.append("%s: stage %s outside the gate (%d large elements over 1 ULP, cos %.8f,"
                         " meanrel %.2e)" % (label, name, bad, cs, mr))
    return bits


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
    print("\nkernel stages vs the pins (%s, split %d, softmax %s):"
          % (args.launch, args.split, args.softmax))
    gate_cases(cases, WORK / "out", "%s_%s_%d" % (args.launch, args.softmax, args.split), fails,
               worst, ncmp, extra=knobs)

    # 3. The other knobs. The two launch plans must agree BIT FOR BIT (one of them recomputes the
    #    stages the other stages, that is all); a different reduction ORDER cannot, so the streaming
    #    softmax and a split row are judged by the same gate instead and the gap is printed.
    if not args.no_cross:
        print("\ncross-checks against the gated run above (bit-identical for the launch plan,"
              " same 1-ULP gate for a different reduction order):")
        base, alt = WORK / "out", WORK / "out_cross"
        other = "online" if args.softmax == "full" else "full"
        bits = cross_variant("fused", ["--launch", "fused", "--split", str(args.split),
                                       "--softmax", args.softmax], cases, base, alt, fails, ncmp)
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
        lc: list[dict] = []
        for layer in (layers[:1] if len(layers) > 1 else layers):
            lc += R.prefill_cases(dl, layer, LONG_ROWS)
        print("\nlong-cache cases from %s prefill rows (each row is a decode-shaped query at"
              " position p over lanes 0..p): %d cases, %d..%d lanes"
              % (args.long_fixture, len(lc), lc[0]["lvalid"], lc[-1]["lvalid"]))
        for c in lc:
            for ok, line in R.verify(dl, c):
                if not ok:
                    fails.append("verify %s: %s" % (c["name"], line))
        gate_cases(lc, WORK / "out_long", "long", fails, worst, ncmp, extra=knobs)

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


def run_negatives(args, d: Path, fails) -> None:
    """Inject one bug at a time and ask which stages' BITS moved, against a clean run under the
    identical knobs. Comparing against the pins instead would fold the softmax-policy gap into every
    injection (the first version did, and every negative "caught" the attention output no matter
    what it was given — a negative that fails everywhere measures the knob, not the bug).

    Two cases per negative: `carry` needs a previous query to leak out of, and one case would let
    every other injection pass by doing nothing."""
    layer = R.ATTN_LAYERS[0]
    steps = sorted({p.name.split(".")[0] for p in d.glob("decode*.L%02d.*.npy" % layer)})[:2]
    cases = [R.decode_case(d, s, layer) for s in steps]
    print("\nnegative tests (rejected where declared, and nowhere else):")
    for name, (why, form, must, must_not) in NEG_KERNEL.items():
        knobs = ["--launch", args.launch, "--split", "1",
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
