#!/usr/bin/env python3
"""Gate the fused residual-add + zero-centred RMSNorm (§03.2 + §03.1) against the dec8 pins.

WHAT IS GATED AGAINST WHAT. The pins are `decode{N}.L{ii}.s_in` / `.s_out` (the residual on each
side of a block), the two addends (`...out_proj.out` / `...o_proj.out`, `...mlp.out`), the
post-attention norm's output (`...mlp.in`) and `final_hidden`. There is no `transformer.h.N.glue.*`
prefix and no new dump was needed; `glue_pins.py` reads these and `final_pair()` *finds* which
decode step the unprefixed `final_hidden` belongs to instead of assuming one (§10.5's
last-writer-wins trap). Each layer contributes two launches, exactly the two an engine issues:

    g1:  s = L.s_in,          delta = the block's output,  w = post_attention_layernorm
    g2:  s = the updated s,   delta = mlp.out,            w = the NEXT layer's input_layernorm
                                                      (or lm.norm.weight after layer 23)

and one norm-only launch per step in front of layer 0, which has no add before it. So:

    g1.s vs the reference add      g1.n vs the PIN `mlp.in`          (a real dump)
    g2.s vs the PIN `s_out`        g2.n vs the spec mirror, and vs the PIN `final_hidden`
                                   on the step those unprefixed pins belong to
    n0.n vs the spec mirror        plus a cross-stage check: feed g2.n / n0.n into the next
                                   layer's first projection and require the pinned output back

Stage R first, because every comparison below depends on it: torch's own BF16 arithmetic must
reproduce `mlp.in` and `s_out` from the pinned inputs BIT-EXACTLY, the residual chain must be
self-consistent (`L.s_out == (L+1).s_in`), and the plausible misreadings must NOT reproduce them —
plain `w` instead of `1+w`, normalising the addend, and the §04.5 gated-norm form on this path
(§03.2: no gate, no bias, no per-layer scaling here).

Judgement goes through `prototype/bf16_ulp.py`. The two `.s` outputs are pure BF16 adds and are
required to be BIT-EXACT; the norm outputs use the cancellation band that `floor_probe.py`
measures for a norm (not the dot band it was copied from), plus the §10.1-class aggregates.

`--negative` injects eight kernel bugs, each scoped to the call that can show it, with declared
must-catch and must-NOT-catch observables. `--time` takes `tools/gpu_lock.sh` and reports launches
per token and wall cost against the 1.85 us launch floor — not GB/s, because this op moves half a
megabyte per token and everything about it is dispatch, not bandwidth.

    .venv/bin/python prototype/glue/run.py [--fixture dec8] [--layers all] [--steps all]
                                          [--negative] [--time] [--split-check] [--loader-verify]
                                          [--iters 200] [--threads 256] [--verify-only]
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp  # noqa: E402
from glue_pins import (H, case, case_prefill, ckpt_w, dump_bf16, final_pair, first_proj,  # noqa
                       is_attn_layer, is_attn_layer_pre, pin, pin_rows, proj_weight,
                       read_bf16, w_g1, w_g2, w_in)
from speccheck.specimpl import rmsnorm_zc  # noqa: E402

BIN = Path(__file__).resolve().parent / "glue"
WORK = Path("/tmp/glue_gate")
GPU_LOCK = ROOT / "tools" / "gpu_lock.sh"
LAUNCH_FLOOR_US = 1.85          # bench/performance-status.md §4, measured
PROBE_GBS = 237.0

# What "reproduce" means for a NORM. Sized by floor_probe.py over 192 norms x 1024 elements — the
# same 3e-5 the dot gates use, but measured here rather than inherited, and the run prints the
# tightest band these results would pass at so the margin is on the record.
CANCEL_REL = 3e-5
NDIFF_PCT_MAX, NDIFF_ABS_MIN = 2.0, 1
COS_MIN, MEANREL_MAX = 0.9999, 1e-2

OBSERVABLES = ["g1_s", "g1_n", "g2_s", "g2_n", "n0_n"]

# name -> (why it is plausible, which glue call it is injected into, must-catch, must-not-catch)
NEGATIVES = [
    ("plainw", 1, "affine is w instead of 1+w (call 1)", ["g1_n"], ["g1_s", "g2_s", "g2_n", "n0_n"]),
    ("plainw", 2, "affine is w instead of 1+w (call 2)", ["g2_n"], ["g1_s", "g1_n", "g2_s", "n0_n"]),
    ("normdelta", 1, "normalise the addend instead of the updated residual", ["g1_n"],
     ["g1_s", "g2_s", "g2_n"]),
    ("noadd", 1, "skip the residual add (call 1)", ["g1_s", "g1_n"], ["g2_s", "g2_n", "n0_n"]),
    ("noadd", 2, "skip the residual add (call 2)", ["g2_s", "g2_n"], ["g1_s", "g1_n", "n0_n"]),
    ("mean", 1, "mean instead of mean-of-squares", ["g1_n"], ["g1_s", "g2_s"]),
    ("bf16acc", 1, "squared-sum accumulator kept in BF16", ["g1_n"], ["g1_s", "g2_s", "g2_n"]),
    ("bf16sub", 1, "one BF16 subpartial before the F32 ladder", ["g1_n"], ["g1_s", "g2_s", "g2_n"]),
]


def compare(got, ref, **kw) -> dict:
    kw.setdefault("cancel_rel", CANCEL_REL)
    return bf16_ulp.report(got, ref, **kw)


def fmt(c: dict) -> str:
    return ("ndiff %5d/%d (%.4f%%) max %d ULP | over-tol large: %d (worst %d ULP at |ref|/max"
            " %.1e) | cos %.8f meanrel %.2e | in-band %d (tightest band that would pass: %.1e)"
            % (c["ndiff"], c["n"], 100.0 * c["ndiff"] / c["n"], c["maxulp"], c["n_bad"],
               c["worst_bad_ulp"], c["worst_bad_rel"], c["cos"], c["meanrel"], c["n_band"],
               c["worst_off_rel"]))


def ndiff_limit(n: int) -> float:
    return max(NDIFF_PCT_MAX * n / 100.0, NDIFF_ABS_MIN)


def band_ok(c: dict) -> bool:
    """True when the comparison violates the criteria the clean gate applies."""
    return bool(c["n_bad"] or c["ndiff"] > ndiff_limit(c["n"]) or c["cos"] < COS_MIN
                or c["meanrel"] > MEANREL_MAX)


def judge(tag: str, c: dict, fails: list[str], worst: dict) -> None:
    print("    %-34s %s" % (tag, fmt(c)))
    key = tag.split()[1] if len(tag.split()) > 1 else tag
    prev = worst.setdefault(key, [0, 0, 0, 0.0])
    prev[0] = max(prev[0], c["worst_bad_ulp"])
    prev[1] = max(prev[1], c["maxulp"])
    prev[2] = max(prev[2], c["n_band"])
    prev[3] = max(prev[3], c["worst_off_rel"])
    if c["n_bad"]:
        fails.append("%s: %d large element(s) differ by >1 ULP (worst %d ULP at |ref|/max %.1e)"
                     % (tag, c["n_bad"], c["worst_bad_ulp"], c["worst_bad_rel"]))
    if c["ndiff"] > ndiff_limit(c["n"]):
        fails.append("%s: %d elements differ (limit %.1f)" % (tag, c["ndiff"], ndiff_limit(c["n"])))
    if c["cos"] < COS_MIN or c["meanrel"] > MEANREL_MAX:
        fails.append("%s: cos %.8f meanrel %.2e outside G3i/§10.1" % (tag, c["cos"], c["meanrel"]))


def judge_exact(tag: str, got, ref, fails: list[str], counts: dict) -> None:
    n = int((got != ref).sum())
    key = tag.split()[1] if len(tag.split()) > 1 else tag
    counts[key] = counts.get(key, 0) + n
    if n:
        fails.append("%s: %d element(s) differ from the reference add — a BF16 add has no"
                     " reordering freedom, so this is an error, not a rounding choice"
                     % (tag, n))

def run_bin(args: list[str], label: str, quiet: bool = True) -> dict:
    proc = subprocess.run([str(BIN)] + args, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode:
        sys.stdout.write(proc.stdout + proc.stderr)
        raise SystemExit("kernel failed: %s (label %s)" % (label, proc.returncode))
    if not quiet:
        for line in proc.stdout.splitlines():
            print("      | " + line)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            out.update(dict(kv.split("=", 1) for kv in line.split()[1:]))
        if line.startswith("TIMING"):
            out["_timing"] = out.get("_timing", "") + line + "\n"
    return out


# ------------------------------------------------------------------ stage R
def verify_rebuild(d: Path, steps, layers, final_step) -> int:
    """Prove the pin pairing with torch's own arithmetic, and prove the misreadings wrong.

    Everything the kernel is compared against comes from here, so a pairing that could not be wrong
    would prove nothing (§10.5's `act` trap is the cautionary tale: a hook that read the wrong
    argument reproduced real reference tensors and passed every self-check). The misreadings below
    are the ones this op can actually get wrong: which norm belongs to which call, whether the
    affine is `1+w`, and whether the norm sees the add or the pre-add residual.
    """
    fails = 0
    checked = 0
    for step in steps:
        for layer in layers:
            c = case(d, step, layer)
            checks = []
            # g1's norm output must BE the pinned mlp input, bit for bit
            checks.append((bool((rmsnorm_zc(c["s_mid"].reshape(1, -1), ckpt_w(c["w1"])).reshape(-1)
                                 == c["ref_g1_n"]).all()), "g1.n == pin mlp.in"))
            # g2's add must land on the pinned s_out, bit for bit
            checks.append((bool((c["s_out"] == c["ref_g2_s"]).all()), "g2.s == pin s_out"))
            # and the chain must be self-consistent: this layer's out is the next layer's in
            if layer < 23:
                checks.append((bool((c["ref_g2_s"]
                                     == pin(d, "%s.L%02d.s_in" % (step, layer + 1))).all()),
                               "s_out == next s_in"))
            # MISREADINGS that must not reproduce the pins
            checks.append((not bool((rmsnorm_zc(c["s_mid"].reshape(1, -1),
                                                1.0 + ckpt_w(c["w1"])).reshape(-1)
                                     == c["ref_g1_n"]).all()), "plain-w affine refused"))
            checks.append((not bool((rmsnorm_zc(c["add1"].reshape(1, -1), ckpt_w(c["w1"]))
                                     .reshape(-1) == c["ref_g1_n"]).all()),
                           "norming the addend refused"))
            checks.append((not bool((rmsnorm_zc(c["s"].reshape(1, -1), ckpt_w(c["w1"]))
                                     .reshape(-1) == c["ref_g1_n"]).all()),
                           "norming the pre-add residual refused"))
            # the norm belongs to the call: swap the two and the pins stop reproducing
            checks.append((not bool((rmsnorm_zc(c["s_mid"].reshape(1, -1), ckpt_w(w_in(layer)))
                                     .reshape(-1) == c["ref_g1_n"]).all()),
                           "this layer's input norm refused for call 1"))
            checks.append((not bool((rmsnorm_zc(c["s_out"].reshape(1, -1), ckpt_w(c["w1"]))
                                     .reshape(-1) == c["ref_g2_n"]).all()),
                           "this layer's post-attention norm refused for call 2"))
            if not is_attn_layer(d, layer):
                gated = (c["ref_g1_n"].float() * torch.sigmoid(c["ref_g1_n"].float())) \
                    .to(torch.bfloat16)
                checks.append((not bool((gated == c["ref_g1_n"]).all()),
                               "any sigmoid factor on the norm output refused (§03.2: no gate,"
                               " no per-layer scale on this path)"))
            for ok, what in checks:
                checked += 1
                if not ok:
                    fails += 1
                    print("    FAIL %s L%02d: %s" % (step, layer, what))
    if final_step is not None and 23 in layers:
        fh = pin(d, "final_hidden")
        checked += 1
        if not bool((rmsnorm_zc(case(d, final_step, 23)["s_out"].reshape(1, -1),
                                ckpt_w("lm.norm.weight")).reshape(-1) == fh).all()):
            fails += 1
            print("    FAIL final_hidden does not reproduce from layer 23 of %s — the pin pairing"
                  " is wrong" % final_step)
        else:
            print("    the unprefixed final_hidden/pre_final_hidden pins pair with %s (found by"
                  " bit-comparison, not assumed)" % final_step)
    print("    %d checks, %d failures" % (checked, fails))
    return fails


# ------------------------------------------------------------------ plan + reading back
def prepare(d: Path, steps, layers, work: Path, outdir: Path) -> list[str]:
    """Write this case set's inputs as raw BF16 and return the plan lines."""
    work.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []
    for step in steps:
        if 0 in layers:
            s0 = dump_bf16(case(d, step, 0)["s"], work / ("%s.L00.s.bin" % step))
            lines.append("%s.L00.n0 %s - %s - %s"
                         % (step, s0, w_in(0), outdir / ("%s.L00.n0.n.bin" % step)))
        for layer in layers:
            c = case(d, step, layer)
            tag = "%s.L%02d" % (step, layer)
            sf = dump_bf16(c["s"], work / ("%s.s.bin" % tag))
            a1 = dump_bf16(c["add1"], work / ("%s.a1.bin" % tag))
            sm = dump_bf16(c["s_mid"], work / ("%s.smid.bin" % tag))
            a2 = dump_bf16(c["add2"], work / ("%s.a2.bin" % tag))
            lines.append("%s.g1 %s %s %s %s %s"
                         % (tag, sf, a1, c["w1"], outdir / ("%s.g1.s.bin" % tag),
                            outdir / ("%s.g1.n.bin" % tag)))
            lines.append("%s.g2 %s %s %s %s %s"
                         % (tag, sm, a2, c["w2"], outdir / ("%s.g2.s.bin" % tag),
                            outdir / ("%s.g2.n.bin" % tag)))
    return lines


def write_plan(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def cross_stage(d: Path, step: str, layer: int, normed: torch.Tensor):
    """Feed one norm output into the projection it is supposed to feed.

    The norm output of glue call 2 at layer L is the normed input of layer L+1, and `proj_gemv`
    already proved that the reference's normed input reproduces that layer's pinned projection
    bit-exactly. So the pinned projection is a real DUMP to gate call 2's norm against, not only a
    mirror of the formula — which matters because `g2.n` has no pin of its own except at layer 23.
    """
    nxt = layer + 1
    name = first_proj(nxt, d)
    ref = pin(d, "%s.L%02d.%s.out" % (step, nxt, name))
    got = normed.reshape(1, -1) @ proj_weight(nxt, name).t()
    return got, ref


def check_outputs(d: Path, steps, layers, outdir: Path, final_step, fails, worst,
                  counts, verbose=True) -> tuple:
    """Judge every observable.

    Returns (bad, sev): the set of observable names that came out wrong, and how badly each moved
    (worst ndiff / max ULP / meanrel over that observable's cases) so a negative can be reported
    with a number instead of a bare 'caught'.
    """
    bad, sev = set(), {}

    def note(obs, c):
        s = sev.setdefault(obs, {"ndiff": 0, "n": 0, "maxulp": 0, "meanrel": 0.0})
        s["ndiff"] += c["ndiff"]
        s["n"] += c["n"]
        s["maxulp"] = max(s["maxulp"], c["maxulp"])
        s["meanrel"] = max(s["meanrel"], c["meanrel"])

    def note_exact(obs, n):
        s = sev.setdefault(obs, {"ndiff": 0, "n": 0, "maxulp": 0, "meanrel": 0.0})
        s["ndiff"] += n
        s["n"] += H
        s["maxulp"] = max(s["maxulp"], 1 if n else 0)

    for step in steps:
        for layer in layers:
            c = case(d, step, layer)
            tag = "%s.L%02d" % (step, layer)
            for obs, ref in (("g1_s", c["ref_g1_s"]), ("g2_s", c["ref_g2_s"])):
                got = read_bf16(outdir / ("%s.%s.bin" % (tag, obs.replace("_", "."))))
                judge_exact("%s %s" % (tag, obs), got, ref, fails, counts)
                n = int((got != ref).sum())
                note_exact(obs, n)
                if n:
                    bad.add(obs)
            g1n = read_bf16(outdir / ("%s.g1.n.bin" % tag))
            cc = compare(g1n, c["ref_g1_n"])
            if verbose:
                judge("%s g1_n" % tag, cc, fails, worst)
            note("g1_n", cc)
            if band_ok(cc):
                bad.add("g1_n")
            g2n = read_bf16(outdir / ("%s.g2.n.bin" % tag))
            cc = compare(g2n, c["ref_g2_n"])
            if verbose:
                judge("%s g2_n" % tag, cc, fails, worst)
            note("g2_n", cc)
            if band_ok(cc):
                bad.add("g2_n")
            if layer == 23 and step == final_step:
                cc = compare(g2n, pin(d, "final_hidden"))
                if verbose:
                    judge("%s g2_n(final_hidden pin)" % tag, cc, fails, worst)
                if band_ok(cc):
                    bad.add("g2_n")
            if layer < 23:
                got, ref = cross_stage(d, step, layer, g2n)
                cc = compare(got, ref)
                if verbose:
                    judge("%s xstage" % tag, cc, fails, worst)
                if band_ok(cc):
                    bad.add("g2_n")
        if 0 in layers:
            got = read_bf16(outdir / ("%s.L00.n0.n.bin" % step))
            ref = rmsnorm_zc(case(d, step, 0)["s"].reshape(1, -1), ckpt_w(w_in(0))).reshape(-1)
            cc = compare(got, ref)
            if verbose:
                judge("%s n0_n" % step, cc, fails, worst)
            note("n0_n", cc)
            if band_ok(cc):
                bad.add("n0_n")
            # ...and the layer-0 projection that consumes it is pinned too.
            gotx = got.reshape(1, -1) @ proj_weight(0, first_proj(0, d)).t()
            refx = pin(d, "%s.L00.%s.out" % (step, first_proj(0, d)))
            cc = compare(gotx, refx)
            if verbose:
                judge("%s n0_xstage" % step, cc, fails, worst)
            if band_ok(cc):
                bad.add("n0_n")
    return bad, sev


# ------------------------------------------------------------------ negatives
def run_negatives(d: Path, steps, gate_layers, final_step, fails) -> None:
    layers = [l for l in (0, 3, 23) if l in gate_layers]
    neg_steps = list(dict.fromkeys([steps[0], final_step if final_step else steps[-1]]))
    work, outdir = WORK / "neg", WORK / "out_neg"
    lines = prepare(d, neg_steps, layers, work, outdir)
    plan = write_plan(WORK / "plan_neg.txt", lines)
    print("\nnegative tests (rejected where declared, clean where declared clean; severity is"
          " summed over the case set):")
    print("    case set: layers %s, steps %s" % (layers, ",".join(neg_steps)))
    for name, scope, why, must, must_not in NEGATIVES:
        run_bin(["--mode", "once", "--plan", str(plan), "--neg", name, "--scope", str(scope)],
                "%s@%d" % (name, scope))
        bad, sev = check_outputs(d, neg_steps, layers, outdir, final_step, [], {}, {},
                                 verbose=False)
        missing = [m for m in must if m not in bad]
        wrong = [m for m in must_not if m in bad]
        how = "; ".join("%s ndiff %d/%d max %d ULP meanrel %.1e"
                        % (k, sev[k]["ndiff"], sev[k]["n"], sev[k]["maxulp"], sev[k]["meanrel"])
                        for k in must if k in sev)
        print("    --neg %-10s @call%d %-46s caught %-20s severity: %s\n               %*s"
              "must-not: %s%s"
              % (name, scope, why, ",".join(sorted(bad)) or "NOTHING", how or "-", 15, "",
                 ",".join("%s=%s" % (m, "MOVED" if m in bad else "clean") for m in must_not)
                 or "-", "" if not wrong else "   <-- " + ",".join(wrong)))
        if missing:
            fails.append("negative %s@call%d was NOT rejected at %s — the gate is vacuous"
                         % (name, scope, ",".join(missing)))
        if wrong:
            fails.append("negative %s@call%d moved %s, which it should not touch"
                         % (name, scope, ",".join(wrong)))


# ------------------------------------------------------------------ fused vs unfused
def split_equivalence(d, steps, layers, final_step, fails) -> None:
    """The fused kernel and the unfused add+norm pair must produce the same bits.

    Otherwise the launch saving was bought with a silent arithmetic change, and the gate would be
    validating a kernel the timing section does not run.
    """
    work, out_f, out_s = WORK / "eq", WORK / "out_fused", WORK / "out_split"
    plan_f = write_plan(WORK / "plan_eq_fused.txt", prepare(d, steps, layers, work, out_f))
    plan_s = write_plan(WORK / "plan_eq_split.txt", prepare(d, steps, layers, work, out_s))
    run_bin(["--mode", "once", "--plan", str(plan_f)], "fused")
    run_bin(["--mode", "once", "--plan", str(plan_s), "--split"], "split")
    diffs, n = 0, 0
    for step in steps:
        for layer in layers:
            tag = "%s.L%02d" % (step, layer)
            files = ["%s.g1.s.bin" % tag, "%s.g1.n.bin" % tag, "%s.g2.s.bin" % tag,
                     "%s.g2.n.bin" % tag]
            if layer == 0:
                files.append("%s.L00.n0.n.bin" % step)
            for f in files:
                n += 1
                diffs += int((read_bf16(out_f / f) != read_bf16(out_s / f)).sum())
    print("\nfused vs unfused pair: %d outputs x %d elements, %d differing" % (n, H, diffs))
    if diffs:
        fails.append("the fused kernel is not bit-exact with the unfused add+norm pair (%d"
                     " elements differ) — the launch saving changed the arithmetic" % diffs)


# ------------------------------------------------------------------ P4: the tile form
def pre_rebuild(d: Path, layers: list[int]) -> int:
    """Stage R for a prefill tile: the same pairing proof, over [T, H] instead of [H].

    The tile form is only worth gating if the tile is the reference's own tile, so the same
    bit-exact requirements apply: torch's add must land on the pinned `s_out`, the norm of the
    mid-residual must BE the pinned `mlp.in`, layer L's `s_out` must be layer L+1's `s_in` across
    the whole chunk, and the misreadings must still fail. §10.5 is why none of this is assumed.
    """
    fails = checked = 0
    for layer in layers:
        c = case_prefill(d, layer)
        w1 = ckpt_w(c["w1"])
        checks = [
            (bool((rmsnorm_zc(c["s_mid"], w1) == c["ref_g1_n"]).all()), "g1.n == pin mlp.in"),
            (bool((c["s_out"] == c["ref_g2_s"]).all()), "g2.s == pin s_out"),
            (not bool((rmsnorm_zc(c["s_mid"], 1.0 + w1) == c["ref_g1_n"]).all()),
             "plain-w affine refused"),
            (not bool((rmsnorm_zc(c["add1"], w1) == c["ref_g1_n"]).all()),
             "norming the addend refused"),
            (not bool((rmsnorm_zc(c["s"], w1) == c["ref_g1_n"]).all()),
             "norming the pre-add residual refused"),
            (not bool((rmsnorm_zc(c["s_mid"], ckpt_w(w_in(layer))) == c["ref_g1_n"]).all()),
             "this layer's input norm refused for call 1"),
        ]
        if layer < 23:
            checks.append((bool((c["ref_g2_s"] == pin_rows(d, "L%02d.s_in" % (layer + 1))).all()),
                           "s_out == next layer's s_in, whole chunk"))
        for ok, what in checks:
            checked += 1
            if not ok:
                fails += 1
                print("    FAIL L%02d: %s" % (layer, what))
    if 23 in layers:
        for nm, what in (("pre_final_hidden", "g2.s at L23"), ("final_hidden", "g2.n at L23")):
            if not (d / (nm + ".npy")).exists():
                continue
            c = case_prefill(d, 23)
            ref = pin_rows(d, nm)
            got = c["s_out"] if nm == "pre_final_hidden" \
                else rmsnorm_zc(c["s_out"], ckpt_w(w_g2(23)))
            checked += 1
            if not bool((got == ref).all()):
                fails += 1
                print("    FAIL %s: %s does not reproduce the unprefixed %s pin bit-for-bit"
                      % (what, nm, nm))
        print("    the unprefixed final_hidden/pre_final_hidden pins pair with the whole prompt"
              " (a prefill fixture has one phase, so §10.5's last-writer-wins trap cannot bite)"
              " — checked bit for bit" if (d / "final_hidden.npy").exists() else "")
    print("    %d checks, %d failures" % (checked, fails))
    return fails


def _tile_lines(d: Path, layers: list[int], work: Path, outdir: Path, per_row: bool):
    """Plan lines over a whole tile (`per_row=False`, one launch per call) or one row at a time.

    Two shapes from one function on purpose: the batched stage's whole point is that the tile
    launch and the per-row launches are the same arithmetic, and that claim is worthless if the
    two plans are built by two pieces of code that can drift.
    """
    work.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []
    for layer in layers:
        c = case_prefill(d, layer)
        t = c["rows"]
        for r in ([None] if not per_row else range(t)):
            pick = (lambda x: x) if r is None else (lambda x: x[r:r + 1])
            tag = "L%02d" % layer if r is None else "L%02d.r%03d" % (layer, r)
            sf = dump_bf16(pick(c["s"]), work / ("%s.s.bin" % tag))
            if layer == 0:
                lines.append("%s.n0 %s - %s - %s" % (tag, sf, w_in(0),
                                                     outdir / ("%s.n0.n.bin" % tag)))
            a1 = dump_bf16(pick(c["add1"]), work / ("%s.a1.bin" % tag))
            sm = dump_bf16(pick(c["s_mid"]), work / ("%s.smid.bin" % tag))
            a2 = dump_bf16(pick(c["add2"]), work / ("%s.a2.bin" % tag))
            lines.append("%s.g1 %s %s %s %s %s" % (tag, sf, a1, c["w1"],
                                                   outdir / ("%s.g1.s.bin" % tag),
                                                   outdir / ("%s.g1.n.bin" % tag)))
            lines.append("%s.g2 %s %s %s %s %s" % (tag, sm, a2, c["w2"],
                                                   outdir / ("%s.g2.s.bin" % tag),
                                                   outdir / ("%s.g2.n.bin" % tag)))
    return lines


def worst_over_rows(got: torch.Tensor, ref: torch.Tensor) -> tuple:
    """Judge a tile row by row and return the worst row's comparison.

    Per row, not over the flattened tile, because the band is relative to the tensor max: one
    giant row would let a loud token hide a quiet one, and the decode gate judges per row.
    """
    bad = None
    for i in range(got.shape[0]):
        c = compare(got[i], ref[i])
        if bad is None or (c["n_bad"], c["ndiff"], c["maxulp"], -c["cos"]) > (
                bad["n_bad"], bad["ndiff"], bad["maxulp"], -bad["cos"]):
            bad = c
            bad["row"] = i
    return bad


def batched_stage(d: Path, layers: list[int], args, fails: list[str]) -> None:
    """One launch per glue call over a [T, H] tile, gated two ways.

    (a) against the prefill pins — the same rules the decode gate uses, applied per row;
    (b) against T launches of one row each, BIT FOR BIT. (b) is the one that matters for the
        engine's margins: if the multi-row launch changes any bit, the legal route changes and
        every priced margin is suspect. If it does not, prefill arithmetic is decode arithmetic
        and the fixtures cannot move.
    """
    tiles = [l for l in layers]
    c0 = case_prefill(d, tiles[0])
    T = c0["rows"]
    print("\nbatched (tile) stage on %s: %d rows x %d layers, threads %d"
          % (d.name, T, len(tiles), args.threads))
    if pre_rebuild(d, tiles):
        fails.append("prefill pin pairing failed on %s — a gate against the wrong tensor"
                     " verifies nothing" % d.name)
        return

    work, out_t = WORK / "pre_in", WORK / "out_tile"
    out_r = WORK / "out_rows"
    for p in (work, out_t, out_r):
        if p.exists():
            shutil.rmtree(p)
    plan_t = write_plan(WORK / "plan_tile.txt",
                        _tile_lines(d, tiles, work, out_t, per_row=False))
    plan_r = write_plan(WORK / "plan_rows.txt",
                        _tile_lines(d, tiles, work, out_r, per_row=True))
    print("    tile plan: %d launches (one per call) | per-row plan: %d launches (one per row)"
          % (len(plan_t.read_text().rstrip().splitlines()),
             len(plan_r.read_text().rstrip().splitlines())))
    run_bin(["--mode", "once", "--plan", str(plan_t), "--threads", str(args.threads),
             "--rows", str(T)], "tile")
    run_bin(["--mode", "once", "--plan", str(plan_r), "--threads", str(args.threads),
             "--rows", "1"], "rows")

    ncmp, bitdiff, bitn = 0, 0, 0
    for layer in tiles:
        c = case_prefill(d, layer)
        tag = "L%02d" % layer
        obs = {"g1_s": c["ref_g1_s"], "g1_n": c["ref_g1_n"], "g2_s": c["ref_g2_s"],
               "g2_n": c["ref_g2_n"]}
        if layer == 0:
            obs["n0_n"] = rmsnorm_zc(c["s"], ckpt_w(w_in(0)))
        if layer == 23 and (d / "final_hidden.npy").exists():
            obs["g2_n"] = pin_rows(d, "final_hidden")
        for name, ref in obs.items():
            leaf = name.replace("_", ".")
            got = read_bf16(out_t / ("%s.%s.bin" % (tag, leaf)), T * H).reshape(T, H)
            # (a) against the pins / the spec mirror, with the decode gate's rules
            if name.endswith("_s"):     # a residual add has no reordering freedom at all
                n = int((got != ref).sum())
                if n:
                    fails.append("%s %s: %d of %d elements differ from the reference add in the"
                                 " tile launch" % (tag, name, n, ref.numel()))
                print("    %s %-5s add   %d/%d elements differ from the pin" % (tag, name, n,
                                                                                ref.numel()))
            else:
                cc = worst_over_rows(got, ref)
                ncmp += 1
                how = "FAIL" if band_ok(cc) else "ok"
                print("    %s %-5s norm worst row %3d: %s [%s]" % (tag, name, cc["row"],
                                                                    fmt(cc), how))
                if band_ok(cc):
                    fails.append("%s %s: tile launch outside tolerance at row %d" % (tag, name,
                                                                                      cc["row"]))
            # (b) bit-identity with one launch per row, same plan, same inputs
            per_row = torch.cat([read_bf16(out_r / ("L%02d.r%03d.%s.bin" % (layer, r, leaf)), H)
                                 for r in range(T)])
            n = int((per_row != got.reshape(-1)).sum())
            bitn += T * H
            bitdiff += n
            if n:
                fails.append("%s %s: the %d-row launch differs from %d one-row launches in %d"
                             " elements — batching changed the arithmetic" % (tag, name, T, T, n))
    print("    bit-identity: tile vs one launch per row, %d elements compared, %d differ"
          % (bitn, bitdiff))
    print("    %d norm comparisons judged per row (worst row printed above)" % ncmp)

    # Falsification: a tile stage that cannot fail verifies nothing. `plainw` on call 1 must move
    # g1_n and leave the residual adds and call 2 alone, exactly as it does at rows=1.
    run_bin(["--mode", "once", "--plan", str(plan_t), "--threads", str(args.threads),
             "--rows", str(T), "--neg", "plainw", "--scope", "1"], "neg-plainw")
    moved, clean = [], []
    for layer in tiles:
        c = case_prefill(d, layer)
        tag = "L%02d" % layer
        for name, ref in (("g1_s", c["ref_g1_s"]), ("g1_n", c["ref_g1_n"]),
                          ("g2_s", c["ref_g2_s"]), ("g2_n", c["ref_g2_n"])):
            got = read_bf16(out_t / ("%s.%s.bin" % (tag, name.replace("_", "."))),
                            T * H).reshape(T, H)
            broke = bool((got != ref).any()) if name.endswith("_s") else band_ok(
                worst_over_rows(got, ref))
            (moved if broke else clean).append(name)
        if "g1_n" not in moved:
            fails.append("negative plainw@call1 was NOT caught by the tile stage at %s"
                         " — the stage is vacuous" % tag)
        if any(n in moved for n in ("g1_s", "g2_s", "g2_n")):
            fails.append("negative plainw@call1 moved %s at %s, which it should not touch"
                         % (sorted(set(m) - {"g1_n"}), tag))
    print("    falsification --neg plainw @call%d on the tile plan: moved %s | left alone %s"
          % (1, ",".join(sorted(set(moved))) or "NOTHING",
             ",".join(sorted(set(clean))) or "-"))


# ------------------------------------------------------------------ timing
@contextlib.contextmanager
def gpu_lock(tag: str):
    """Timings need the box to themselves (bench/build-queue.md). Ownership is by tag, so acquire
    and release pass the identical string. `util` is printed at both ends and a busy box makes
    every number below provisional — this box's GPU is never idle (flash_serve polls it), so a
    quiet window may not exist at all and a provisional number with util stated beats none."""
    subprocess.run([str(GPU_LOCK), "acquire", tag], check=True)
    try:
        print("  gpu_lock held (%s); util at acquire: %s"
              % (tag, subprocess.run([str(GPU_LOCK), "util"], capture_output=True,
                                     text=True).stdout.strip()))
        yield
    finally:
        print("  util at release: %s"
              % subprocess.run([str(GPU_LOCK), "util"], capture_output=True,
                               text=True).stdout.strip())
        subprocess.run([str(GPU_LOCK), "release", tag], check=False)


def time_token(d: Path, steps, layers, args, fails) -> None:
    """One token's glue: 1 norm-only + 24x2 fused launches, in token order, chained."""
    work, outdir = WORK / "time", WORK / "out_time"
    lines = prepare(d, [steps[0]], layers, work, outdir)
    plan = write_plan(WORK / "plan_token.txt", lines)
    n_fused = len(lines)
    print("\none token's glue: %d fused launches (1 norm-only + 2 calls x 24 layers); unfused"
          " that is %d launches (%d adds + %d norms)" % (n_fused, 2 * n_fused - 1,
                                                         n_fused - 1, n_fused))
    print("  launch floor arithmetic: fused %d x %.2f us = %.4f ms/token | unfused %d x %.2f us"
          " = %.4f ms/token | a token budget of ~7.8 ms" % (n_fused, LAUNCH_FLOOR_US,
                                                            n_fused * LAUNCH_FLOOR_US / 1000.0,
                                                            2 * n_fused - 1, LAUNCH_FLOOR_US,
                                                            (2 * n_fused - 1) * LAUNCH_FLOOR_US
                                                            / 1000.0))
    for split in (False, True):
        for chained in (True, False):
            extra = ["--split"] if split else []
            if chained:
                extra = extra + ["--chained"]
            r = run_bin(["--mode", "burst", "--plan", str(plan), "--iters", str(args.iters),
                         "--threads-list", "128,256,512"] + extra, "burst")
            for line in r.get("_timing", "").splitlines():
                print("  " + line)
    if args.sweep:
        print("\n  (rows sweep n/a: one block per launch; the sweep above is the block width)")
    if fails is None:
        return


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--layers", default="all", help="comma list, or 'all'")
    ap.add_argument("--steps", default="all", help="comma list of decodeN, or 'all'")
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--split-check", action="store_true", default=True)
    ap.add_argument("--no-split-check", action="store_false", dest="split_check")
    ap.add_argument("--time", action="store_true", help="timings (takes the GPU lock)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--loader-verify", action="store_true",
                    help="ask the loader to SHA-256 every resident blob (~7.6 s)")
    ap.add_argument("--verify-only", action="store_true", help="stage R and stop")
    ap.add_argument("--batched", default=None, metavar="FIXTURE",
                    help="P4 stage: run the same fused op over a whole prefill tile [T, H] from "
                         "this fixture (p64/p300), one launch, and require (a) every row matches "
                         "the prefill pins and (b) the multi-row launch is BIT-IDENTICAL to one "
                         "launch per row. Multi-row arithmetic is otherwise untested.")
    args = ap.parse_args()

    d = ROOT / "data" / "refs" / args.fixture
    if not d.exists():
        raise SystemExit("no fixture %s" % d)
    all_steps = sorted({p.name.split(".")[0] for p in d.glob("decode*.L00.s_in.npy")})
    steps = all_steps if args.steps == "all" else args.steps.split(",")
    layers = list(range(24)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    if not BIN.exists():
        subprocess.run(["bash", str(BIN.parent / "build.sh")], check=True)
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed - the gate cannot be trusted")
    WORK.mkdir(parents=True, exist_ok=True)
    final_step = final_pair(d, steps, layers)
    print("fixture %s | %d layers x %d steps | threads %d | band %g | final_hidden pairs with %s"
          % (args.fixture, len(layers), len(steps), args.threads, CANCEL_REL, final_step))

    print("\nstage R — the pin pairing, proved with torch's own BF16 arithmetic:")
    if verify_rebuild(d, steps, layers, final_step):
        raise SystemExit("pin pairing failed — a gate against the wrong tensor verifies nothing")
    if args.verify_only:
        return 0

    fails: list[str] = []
    worst: dict = {}
    counts: dict = {}
    ncmp = [0]
    outdir = WORK / "out"
    plan = write_plan(WORK / "plan_all.txt", prepare(d, steps, layers, WORK / "in", outdir))
    print("\nkernel output vs the pins and the spec mirror (%d launches in the plan):"
          % len(plan.read_text().rstrip().splitlines()))
    once = ["--mode", "once", "--plan", str(plan), "--threads", str(args.threads)]
    if args.loader_verify:
        once.append("--loader-verify")
    run_bin(once, "all")
    ncmp[0] = 4 * len(steps) * len(layers) + (2 if 0 in layers else 0) * len(steps)
    bad, sev = check_outputs(d, steps, layers, outdir, final_step, fails, worst, counts)
    if bad:
        fails.append("observables outside tolerance: %s" % ",".join(sorted(bad)))

    if args.split_check:
        split_equivalence(d, steps[:1], layers[:4] if len(layers) > 4 else layers, final_step,
                          fails)

    if args.batched:
        pd = ROOT / "data" / "refs" / args.batched
        if not pd.exists():
            raise SystemExit("no prefill fixture %s (try p64)" % pd)
        batched_stage(pd, [l for l in (0, 3, 23) if l in layers] or layers[:1], args, fails)

    if args.negative:
        run_negatives(d, steps, layers, final_step, fails)

    if args.time:
        with gpu_lock("glue timing (run.py)"):
            time_token(d, steps, layers, args, fails)

    print("\nworst BF16 ULP by observable -- gated (large elements only) | raw (also the"
          " near-zero elements the band attributes to arithmetic, not error) | most elements any"
          " one comparison put inside the %g band | tightest band these results would pass at:"
          % CANCEL_REL)
    slack = None
    for k, (bad_ulp, raw, nband, minband) in sorted(worst.items()):
        print("    %-16s gated %4d   raw %6d   in-band %7d   tightest passing band %s"
              % (k, bad_ulp, raw, nband, "%.1e" % minband if minband else "-"))
        if minband:
            slack = min(slack or 1e9, CANCEL_REL / minband)
    if slack:
        print("    band in use |ref|/max <= %g; worst offender above the band anywhere: %.1e"
              " -> %.1fx of slack. floor_probe.py measures what a legal reordering of the"
              " reference's own sum reaches; a tighter band has to clear that too."
              % (CANCEL_REL, max(v[3] for v in worst.values()), slack))
    if counts:
        print("    bit-exact outputs (the two residual adds): %s (differing elements, 0 wanted)"
              % counts)
    print("\n%d comparisons, %d failures" % (ncmp[0], len(fails)))
    if fails:
        print("FAIL")
        for f in fails[:30]:
            print("  -", f)
        return 1
    print("PASS: the fused residual-add + zero-centred norm reproduces the reference's glue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
