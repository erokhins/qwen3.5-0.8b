#!/usr/bin/env python3
"""Gate the projection GEMV family (§04.1, §04.5, §05.1, §05.6) at k = 1 against the W1 pins.

Nothing here is synthesized. The operands are rebuilt from the pinned dumps by `operands.py` —
which first proves the rebuild right (bit-exactly) and proves the wrong readings wrong — the
weights come from the checkpoint, and the reference outputs are the pins
`decode{N}.L{ii}.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,q_proj,k_proj,v_proj,o_proj}
.out` (§10.5). For every layer and decode step each projection is launched the way the engine
will launch it — one launch for {qkv, z, b, a}, one for {q, k, v}, one for out_proj/o_proj — and
each output slice is compared against its own pin:

    P1  input projections, from the reference's normed layer input   (§04.1 / §05.1 dots)
    P2  out_proj / o_proj, from the REFERENCE block output           (§04.5 / §05.6 dots)
    cross  the same projection through speccheck's torch CPU expression: a third path, and it
           is bit-exact to the pin, so any disagreement belongs to the kernel

Judgement goes through `prototype/bf16_ulp.py` inside the cancellation band that `floor_probe.py`
measures for THESE K values (1024/2048, not 3584), plus the §10.1-class aggregates: no element
above 1e-4 of the tensor max may exceed 1 BF16 ULP; at most max(2 % of elements, 1 element) may
differ at all; cos >= 0.9999 and meanrel <= 1e-2.

`--negative` injects seven kernel bugs and three wrong operands, each with a declared must-catch
and must-not-catch block (the rule `mlp_gemv` learned: a negative that cannot fail is worse than
none, and one that "catches" everywhere is measuring the wrong thing). Every injection writes and
loads its own layer's weights and operands, so nothing leaks in from the main loop.

`--time` takes `tools/gpu_lock.sh` and measures cold: each shape across every layer that has it
(18 or 6 distinct weight blobs, tiled past the iGPU cache when the shape alone is too small to
be cold), and the whole 48-launch token plan in the order a token runs it. `--sweep` adds the
rows/block and x-packing sweeps. Both write outputs and correctness-check the timed config
(prototype/README.md rule 2).

    .venv/bin/python prototype/proj_gemv/run.py [--fixture dec8] [--layers all] [--steps all]
                                               [--negative] [--time] [--sweep] [--rows 8]
                                               [--iters 30] [--no-verify]
    .venv/bin/python prototype/proj_gemv/run.py --tokens p64      # P5: the token-batched form
    .venv/bin/python prototype/proj_gemv/run.py --time --only gdn  # one ledger row at a time

`--only gdn|attn` is a TIMING selector and needs `--time`: it narrows the cold timing path to the
layers of one family, which is how `bench/performance-status.md` §2's two "not measured on its own"
rows (GDN projections 0.3796 GB, attention projections 0.0881 GB) get measured instead of
arithmetically apportioned — `bench/plan-2026-10-12.md` Q6 item 5. The P1/P2 correctness gate never
narrows: it runs every projection of every layer whatever this flag says, because a gate that can be
told to look at half the projections is not the gate the numbers above were taken with. Refusing the
flag without `--time` is the same point: an inert `--only` would read like a subset gate.
"""
from __future__ import annotations

import argparse
import contextlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp  # noqa: E402
from operands import (CKPT, attn_operand, dump_bf16, is_attn_layer, operands_for,  # noqa: E402
                      pack_offsets, pin, pin_tile, prefill_operands_for, prefill_tokens,
                      prefill_verify, projections_of, proj_weight, read_bf16, verify,
                      wrong_normed_inputs)

BIN = Path(__file__).resolve().parent / "proj_gemv"
WORK = Path("/tmp/proj_gemv")
GPU_LOCK = ROOT / "tools" / "gpu_lock.sh"
PROBE_GBS = 237.0
# What "reproduce" means, sized by floor_probe.py for K = 1024/2048: reordering the reference's
# own F32 sum already moves ~1 element in 1e5 by >1 ULP at cancelling elements, so a 16-element
# output (in_proj_b/a) gets a one-element allowance rather than a zero-element one.
NDIFF_PCT_MAX, NDIFF_ABS_MIN = 2.0, 1
COS_MIN, MEANREL_MAX = 0.9999, 1e-2
# The cancellation band. `mlp_gemv` tightened its own from a guessed 1e-4 to a measured 3e-5;
# this gate uses the same number and prints, at the end of every run, the tightest band these
# results would pass at, so the margin is on the record instead of assumed.
CANCEL_REL = 3e-5

IN_PROJ_BLOCKS = ["in_proj_qkv.q", "in_proj_qkv.k", "in_proj_qkv.v", "in_proj_z", "in_proj_b",
                  "in_proj_a"]
ATTN_BLOCKS = ["q_proj", "k_proj", "v_proj"]

# name -> (why it is plausible, which block type it is exercised on, which launches get the bug,
#         must-catch blocks, must-not-catch blocks)
NEG_KERNEL = {
    "tail": ("drop the last word of K (off-by-one reduction length)", "gdn", "all",
             IN_PROJ_BLOCKS + ["out_proj"], []),
    "half": ("accumulate half of K", "gdn", "all", IN_PROJ_BLOCKS + ["out_proj"], []),
    "trunc": ("round toward zero instead of RNE", "gdn", "all",
              IN_PROJ_BLOCKS + ["out_proj"], []),
    "rowswap": ("write each dot to the neighbouring output row", "gdn", "all",
                IN_PROJ_BLOCKS + ["out_proj"], []),
    "transpose": ("read the weight buffer transposed", "gdn", "all",
                  IN_PROJ_BLOCKS + ["out_proj"], []),
    "qhalf": ("q_proj rows written in the contiguous-half layout §05.1 forbids", "attn", "inproj",
              ["q_proj"], ["k_proj", "v_proj"]),
    "kvswap": ("k/v sections of a packed in_proj_qkv swapped (§04.1)", "gdn", "inproj",
               ["in_proj_qkv.k", "in_proj_qkv.v"],
               ["in_proj_qkv.q", "in_proj_z", "in_proj_b", "in_proj_a"]),
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


def judge(tag: str, c: dict, fails: list[str], worst: dict) -> None:
    print("    %-30s %s" % (tag, fmt(c)))
    key = tag.split()[-1]
    prev = worst.setdefault(key, [0, 0, 0, 0.0])
    prev[0] = max(prev[0], c["worst_bad_ulp"])
    prev[1] = max(prev[1], c["maxulp"])
    prev[2] = max(prev[2], c["n_band"])
    prev[3] = max(prev[3], c["worst_off_rel"])
    if c["n_bad"]:
        fails.append("%s: %d large element(s) differ by >1 ULP (worst %d ULP at |ref|/max %.1e)"
                     % (tag, c["n_bad"], c["worst_bad_ulp"], c["worst_bad_rel"]))
    if c["ndiff"] > ndiff_limit(c["n"]):
        fails.append("%s: %d elements differ (limit %.1f)" % (tag, c["ndiff"],
                                                             ndiff_limit(c["n"])))
    if c["cos"] < COS_MIN or c["meanrel"] > MEANREL_MAX:
        fails.append("%s: cos %.8f meanrel %.2e outside G3i/§10.1" % (tag, c["cos"],
                                                                      c["meanrel"]))


def judge_neg(c: dict) -> bool:
    """True when the comparison violates the same criteria the clean gate applies."""
    return bool(c["n_bad"] or c["ndiff"] > ndiff_limit(c["n"]) or c["cos"] < COS_MIN
                or c["meanrel"] > MEANREL_MAX)


def run_bin(args: list[str], label: str, quiet: bool = True) -> dict:
    proc = subprocess.run([str(BIN)] + args, capture_output=True, text=True)
    if proc.returncode:
        sys.stdout.write(proc.stdout + proc.stderr)
        raise SystemExit("kernel failed: %s" % label)
    if not quiet:
        for line in proc.stdout.splitlines():
            print("      | " + line)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            out = dict(kv.split("=", 1) for kv in line.split()[1:])
            out["_rowsweep"] = " ".join(l.strip() for l in proc.stdout.splitlines()
                                        if l.strip().startswith("rows/block"))
            return out
    raise SystemExit("no RESULT line:\n" + proc.stdout)


# ------------------------------------------------------------------ plan construction
def write_plan(path: Path, jobs) -> Path:
    """One line per launch: NAME ROWS IN XFILE WFILE[,WFILE...] [OUTFILE]."""
    with open(path, "w") as f:
        for name, rows, in_dim, xfile, wfiles, outfile in jobs:
            f.write("%s %d %d %s %s%s\n" % (name, rows, in_dim, xfile,
                                            ",".join(str(w) for w in wfiles),
                                            (" " + str(outfile)) if outfile else ""))
    return path


def pack_rows(pack) -> int:
    return sum(n for _, n in pack)


def sections_of(name: str, off: int, n: int):
    """Named blocks to report on: the §04.1 q/k/v thirds of in_proj_qkv, else the pin's name."""
    if name == "in_proj_qkv":
        third = n // 3
        return [("in_proj_qkv.q", off, third), ("in_proj_qkv.k", off + third, third),
                ("in_proj_qkv.v", off + 2 * third, third)]
    return [(name, off, n)]


def layer_files(layer: int, d: Path, steps: list[str]) -> dict:
    """This layer's weights and rebuilt operands, as raw BF16, written once per layer."""
    names = projections_of(d, layer)
    files = {"w": {n: dump_bf16(proj_weight(layer, n), WORK / ("L%02d_%s.bin" % (layer, n)))
                   for n in names}, "xin": {}, "xout": {}}
    for step in steps:
        ops = operands_for(d, step, layer)
        files["xin"][step] = dump_bf16(ops["x_in"], WORK / ("L%02d_%s_xin.bin" % (layer, step)))
        files["xout"][step] = dump_bf16(ops["x_out"],
                                       WORK / ("L%02d_%s_xout.bin" % (layer, step)))
    return files


def layer_jobs(layer: int, d: Path, steps, files, outdir: Path):
    """The launches a real token issues in this layer: the packed input projection, then the
    output projection. Returns (jobs, pack, out_name)."""
    ops = operands_for(d, steps[0], layer)
    pack, out_name = ops["pack"], ops["out_proj"]
    jobs = []
    for step in steps:
        jobs.append(("%s.inproj" % step, pack_rows(pack), 1024, files["xin"][step],
                     [files["w"][n] for n, _ in pack], outdir / ("%s.inproj.bin" % step)))
        jobs.append(("%s.%s" % (step, out_name), 1024, 2048, files["xout"][step],
                     [files["w"][out_name]], outdir / ("%s.%s.bin" % (step, out_name))))
    return jobs, pack, out_name


def blocks_after_run(layer: int, d: Path, step: str, pack, outdir: Path, out_name: str,
                     want_out: bool = True):
    """Which named output blocks the last run left outside tolerance. `want_out=False` for a plan
    that only contained the input-projection launch (the two layout negatives), so a missing
    output file is not mistaken for a clean block."""
    offs = pack_offsets(pack)
    caught = []
    for name, n in pack:
        off = offs[name][0]
        whole = read_bf16(outdir / ("%s.inproj.bin" % step), pack_rows(pack))
        ref = pin(d, "%s.L%02d.%s.out" % (step, layer, name))
        for sname, soff, sn in sections_of(name, off, n):
            if judge_neg(compare(whole[soff:soff + sn], ref[soff - off:soff - off + sn])):
                caught.append(sname)
    if not want_out:
        return caught
    whole = read_bf16(outdir / ("%s.%s.bin" % (step, out_name)), 1024)
    if judge_neg(compare(whole, pin(d, "%s.L%02d.%s.out" % (step, layer, out_name)))):
        caught.append(out_name)
    return caught


# ------------------------------------------------------------------ correctness
def gate_correctness(args, d: Path, steps, layers, fails, worst, ncmp) -> None:
    from speccheck.specimpl import Weights
    w = Weights(CKPT)          # loads by the §01 names and re-checks every shape (G1)
    outdir = WORK / "out"
    outdir.mkdir(exist_ok=True)
    for layer in layers:
        files = layer_files(layer, d, steps)
        jobs, pack, out_name = layer_jobs(layer, d, steps, files, outdir)
        plan = write_plan(WORK / ("plan_L%02d.txt" % layer), jobs)
        run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(args.rows)],
                "L%02d" % layer)
        offs = pack_offsets(pack)
        print("  L%02d %-4s pack %s + %s" % (layer, "attn" if is_attn_layer(d, layer) else "gdn",
                                             ",".join(n for n, _ in pack), out_name))
        xkey_pre = "self_attn" if is_attn_layer(d, layer) else "linear_attn"
        for step in steps:
            x_in = operands_for(d, step, layer)["x_in"].reshape(1, -1)
            for name, n in pack:
                off = offs[name][0]
                whole = read_bf16(outdir / ("%s.inproj.bin" % step), pack_rows(pack))
                ref = pin(d, "%s.L%02d.%s.out" % (step, layer, name))
                judge("%s %s" % (step, name), compare(whole[off:off + n], ref), fails, worst)
                ncmp.append(name)
                # third path: speccheck's own torch CPU expression on the same operand
                sc = x_in @ w.p["model.language_model.layers.%d.%s.%s.weight"
                                % (layer, xkey_pre, name)].t()
                if int((sc.reshape(-1) != ref).sum()):
                    fails.append("L%02d %s %s: speccheck's own matmul is not bit-exact to the"
                                 " pin - the operand rebuild or the pin pairing is wrong"
                                 % (layer, step, name))
            whole = read_bf16(outdir / ("%s.%s.bin" % (step, out_name)), 1024)
            ref = pin(d, "%s.L%02d.%s.out" % (step, layer, out_name))
            judge("%s %s" % (step, out_name), compare(whole, ref), fails, worst)
            ncmp.append(out_name)


# ------------------------------------------------------------------ negatives
def run_negatives(args, d: Path, fails) -> None:
    gdn = next(l for l in range(24) if not is_attn_layer(d, l))
    attn = next(l for l in range(24) if is_attn_layer(d, l))
    step = sorted(p.name.split(".")[0] for p in d.glob("decode*.L%02d.s_in.npy" % gdn))[0]
    outdir = WORK / "neg"
    outdir.mkdir(exist_ok=True)
    print("\nnegative tests (rejected where declared, and nowhere else):")
    cases = {"gdn": gdn, "attn": attn}
    for name, (why, kind, scope, must, must_not) in NEG_KERNEL.items():
        layer = cases[kind]
        files = layer_files(layer, d, [step])
        jobs, pack, out_name = layer_jobs(layer, d, [step], files, outdir)
        if scope == "inproj":
            jobs = [j for j in jobs if j[0].endswith(".inproj")]
        plan = write_plan(WORK / ("plan_neg_%s.txt" % name), jobs)
        run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(args.rows), "--neg", name],
                name)
        caught = blocks_after_run(layer, d, step, pack, outdir, out_name,
                                  want_out=scope == "all")
        missing = [m for m in must if m not in caught]
        wrong = [m for m in must_not if m in caught]
        print("    --neg %-10s L%02d %-46s caught %-44s must-not: %s%s"
              % (name, layer, why, ",".join(caught) or "NOTHING",
                 ",".join("%s=%s" % (m, "clean" if m not in caught else "MOVED")
                          for m in must_not) or "-",
                 "" if not wrong else "  <-- " + ",".join(wrong)))
        if missing:
            fails.append("negative %s was NOT rejected at %s - the gate is vacuous"
                         % (name, ",".join(missing)))
        if wrong:
            fails.append("negative %s moved %s, which it should not touch" % (name,
                                                                              ",".join(wrong)))
    # Operand negatives: same kernel, wrong INPUT tensor. Must move the launches that read it and
    # must leave the one that reads the other operand alone.
    for kind, layer in cases.items():
        files = layer_files(layer, d, [step])
        jobs, pack, out_name = layer_jobs(layer, d, [step], files, outdir)
        if kind == "gdn":
            variants = wrong_normed_inputs(d, step, layer)
            where = "inproj"
            must = IN_PROJ_BLOCKS
            must_not = [out_name]
        else:
            variants = {"contiguous-half gate (§05.1)":
                        attn_operand(d, step, layer, interleave="half")}
            where = "outproj"
            must = [out_name]
            must_not = ATTN_BLOCKS
        target = ".inproj" if where == "inproj" else "." + out_name
        for label, xx in variants.items():
            slug = re.sub(r"\W+", "_", label)[:24]
            badx = dump_bf16(xx, WORK / ("L%02d_bad_%s.bin" % (layer, slug)))
            j2 = [(nm, r, ind, badx if nm.endswith(target) else xf, wf, of) for nm, r, ind, xf,
                  wf, of in jobs]
            plan = write_plan(WORK / ("plan_neg_in_%s.txt" % slug), j2)
            run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(args.rows)], label)
            caught = blocks_after_run(layer, d, step, pack, outdir, out_name)
            missing = [m for m in must if m not in caught]
            wrong = [m for m in must_not if m in caught]
            print("    x=%-30s L%02d %-30s caught %-44s must-not: %s%s"
                  % (label[:30], layer, "(wrong operand)", ",".join(caught) or "NOTHING",
                     ",".join("%s=%s" % (m, "clean" if m not in caught else "MOVED")
                              for m in must_not) or "-",
                     "" if not wrong else "  <-- " + ",".join(wrong)))
            if missing:
                fails.append("wrong operand %s was NOT rejected at %s" % (label,
                                                                          ",".join(missing)))
            if wrong:
                fails.append("wrong operand %s moved %s, which it should not touch"
                             % (label, ",".join(wrong)))


# ------------------------------------------------------------------ the token-batched stage (P5)
def worst_over_rows(got, ref, **kw) -> dict:
    """Judge a [T, dim] tile one row at a time and return the worst row's comparison.

    Per row rather than over the flattened tile for the reason `mlp_gemv` gives: the cancellation
    band and the 2 % ndiff rule are relative to the tensor being judged, and flattened one loud token
    could hide a quiet one. The rules themselves are the k=1 gate's, unchanged — `floor_probe.py`
    sized them for K = 1024/2048, which is a property of the reduction length, not of how many rows
    a launch carries. `kw` is passed through so a caller can widen `ulp_max` exactly where the decode
    stage does.
    """
    bad = None
    for i in range(got.shape[0]):
        c = compare(got[i], ref[i], **kw)
        if bad is None or (c["n_bad"], c["ndiff"], c["maxulp"], -c["cos"], -c["meanrel"]) > (
                bad["n_bad"], bad["ndiff"], bad["maxulp"], -bad["cos"], -bad["meanrel"]):
            bad = c
            bad["row"] = i
    return bad


def prefill_layer_jobs(layer, files, outdir, ops):
    """The two launches a wave issues in this layer: the packed input projection over the whole
    [T,1024] norm tile, then the output projection over the [T,2048] gated block output."""
    pack, out_name = ops["pack"], ops["out_proj"]
    return [("tile.inproj", pack_rows(pack), 1024, files["xin"],
             [files["w"][n] for n, _ in pack], outdir / ("tile.inproj.%d.bin" % layer)),
            ("tile." + out_name, 1024, 2048, files["xout"],
             [files["w"][out_name]], outdir / ("tile.outproj.%d.bin" % layer))], pack, out_name


def run_tokens(args, fails, worst) -> None:
    """P5: the token-batched projections, gated against the prefill pins and against the k=1 kernel.

    Three claims, in the order the engine depends on them:

    1. **bit-identity vs the shipped single-row kernel.** A batched kernel that moved one bit would
       move the legal route every fixture margin was priced against, and the fixtures cannot see that
       (P4b: identical arithmetic submitted one row at a time produces identical bits). So for
       sampled tokens the k=1 kernel's answer must BE the tile's row, bit for bit. The reference is
       `--kernel k1` — the original kernel, a different code path, not the same template with a
       different tile width.
    2. **the pins.** The same four rules the k=1 gate applies, judged per row, over every row the
       fixture pins.
    3. **the identity check can fail.** One injected bug (`tail`, one word of K dropped) into the tile
       run only: if the tile still matched the per-row reference the comparison would be vacuous,
       which is the trap `mlp_gemv/results.md` names.
    """
    dp = ROOT / "data" / "refs" / args.tokens
    if not dp.exists():
        raise SystemExit("no prefill fixture %s (try p64)" % dp)
    layers = [l for l in (list(range(24)) if args.layers == "all"
                          else [int(x) for x in args.layers.split(",")])
              if (dp / ("L%02d.s_in.npy" % l)).exists()]
    if not layers:
        raise SystemExit("%s pins no L{ll}.s_in for %s" % (dp.name, args.layers))
    T = prefill_tokens(dp, layers[0])
    sample = sorted({0, 1, T // 2, max(0, T - 2), T - 1})
    outdir = WORK / "tokens"
    outdir.mkdir(exist_ok=True)
    print("\ntoken-batched stage on %s: T=%d rows, layers %s, tile sampled at tokens %s"
          % (dp.name, T, layers, sample))

    # 0. The gate's foundation, on the tile: the rebuilt operands reproduce every row of the pins
    #    bit-exactly and the wrong readings do not (the §10.5 pairing proof, at T rows).
    print("  operand rebuild on the prefill tiles:")
    nbad = 0
    for layer in layers:
        for ok, line in prefill_verify(dp, layer):
            if not ok:
                nbad += 1
                print("    FAIL %s" % line)
    print("    %d layers checked, %d failures" % (len(layers), nbad))
    if nbad:
        raise SystemExit("prefill operand rebuild failed - stopping: a gate against the wrong tile"
                         " verifies nothing")

    ncomp = ndiff_bits = njudge = 0
    for layer in layers:
        ops = prefill_operands_for(dp, layer)
        pack, out_name = ops["pack"], ops["out_proj"]
        names = [n for n, _ in pack] + [out_name]
        files = {"w": {n: dump_bf16(proj_weight(layer, n),
                                    WORK / ("T%02d_%s.bin" % (layer, n))) for n in names},
                 "xin": dump_bf16(ops["x_in"], WORK / ("T%02d_xin.bin" % layer)),
                 "xout": dump_bf16(ops["x_out"], WORK / ("T%02d_xout.bin" % layer))}
        refs = {}
        for name, rows in pack:
            refs[name] = pin_tile(dp, "L%02d.%s.out" % (layer, name), rows)
        refs[out_name] = pin_tile(dp, "L%02d.%s.out" % (layer, out_name), 1024)

        jobs, _, _ = prefill_layer_jobs(layer, files, outdir, ops)
        plan = write_plan(WORK / ("plan_tile_L%02d.txt" % layer), jobs)
        run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(args.rows),
                 "--tokens", str(T)], "L%02d tile" % layer)
        got = {}
        whole = read_bf16(outdir / ("tile.inproj.%d.bin" % layer), pack_rows(pack) * T)
        whole = whole.reshape(T, pack_rows(pack))
        offs = pack_offsets(pack)
        for name, rows in pack:
            got[name] = whole[:, offs[name][0]:offs[name][0] + rows]
        got[out_name] = read_bf16(outdir / ("tile.outproj.%d.bin" % layer),
                                  1024 * T).reshape(T, 1024)

        print("  L%02d %-4s tile T=%d, pack %s + %s"
              % (layer, ops["kind"], T, ",".join(n for n, _ in pack), out_name))
        for name in names:
            c = worst_over_rows(got[name], refs[name])
            judge("L%02d r%03d %s" % (layer, c["row"], name), c, fails, worst)
            njudge += got[name].shape[0]

        # 1. bit-identity vs the k=1 kernel, run one row at a time on the same operands
        layer_comp = layer_diff = 0
        bad_bits = 0
        for t in sample:
            xone = {"xin": dump_bf16(ops["x_in"][t:t + 1], WORK / ("T%02d_one_in.bin" % layer)),
                    "xout": dump_bf16(ops["x_out"][t:t + 1], WORK / ("T%02d_one_out.bin" % layer))}
            one = [("one.inproj", pack_rows(pack), 1024, xone["xin"],
                    [files["w"][n] for n, _ in pack], outdir / ("one.inproj.%d.bin" % layer)),
                   ("one." + out_name, 1024, 2048, xone["xout"],
                    [files["w"][out_name]], outdir / ("one.outproj.%d.bin" % layer))]
            plan = write_plan(WORK / ("plan_one_L%02d_t%d.txt" % (layer, t)), one)
            run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(args.rows),
                     "--kernel", "k1"], "L%02d k1 t%d" % (layer, t))
            w1 = read_bf16(outdir / ("one.inproj.%d.bin" % layer), pack_rows(pack))
            row1 = {name: w1[offs[name][0]:offs[name][0] + rows] for name, rows in pack}
            row1[out_name] = read_bf16(outdir / ("one.outproj.%d.bin" % layer), 1024)
            for name in names:
                many = got[name][t].reshape(-1)
                n = int((many != row1[name]).sum())
                ncomp += many.numel()
                ndiff_bits += n
                layer_comp += many.numel()
                layer_diff += n
                bad_bits += n
                if n:
                    fails.append("L%02d token %d: the %d-token tile's `%s` row differs from the"
                                 " k=1 kernel in %d of %d elements — batching changed the"
                                 " arithmetic" % (layer, t, T, name, n, many.numel()))
        print("    bit-identity vs the k=1 kernel: tokens %s, all %d projections, %d elements"
              " compared, %d differ" % (sample, len(names), layer_comp, layer_diff))

        # 3. the falsification: the same tile plan with one word of K dropped, which must move the
        #    tile. Proves both that the batched kernel honours the injected-bug vocabulary and that
        #    the element comparison above is not comparing a buffer with itself.
        jobs_bad = [list(j) for j in jobs]
        for j in jobs_bad:
            j[5] = Path(str(j[5]) + ".bad")
        plan = write_plan(WORK / ("plan_bad_L%02d.txt" % layer), [tuple(j) for j in jobs_bad])
        run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(args.rows),
                 "--tokens", str(T), "--neg", "tail"], "L%02d tile --neg tail" % layer)
        bad_in = read_bf16(outdir / ("tile.inproj.%d.bin.bad" % layer),
                           pack_rows(pack) * T).reshape(T, pack_rows(pack))
        moved = int((bad_in != whole).sum())
        moved += int((read_bf16(outdir / ("tile.outproj.%d.bin.bad" % layer), 1024 * T)
                      != got[out_name].reshape(-1)).sum())
        print("    falsification (--neg tail on the tile only): %d of %d tile elements moved"
              % (moved, pack_rows(pack) * T + 1024 * T))
        if not moved:
            fails.append("L%02d: --neg tail on the tile moved nothing — the batched kernel ignores"
                         " the injected-bug vocabulary and/or the tile comparison is vacuous"
                         % layer)
        if not judge_neg(compare(bad_in.reshape(-1), whole.reshape(-1))):
            fails.append("L%02d: the --neg tail tile is not judged different from the clean tile by"
                         " the gate's own comparator" % layer)
    print("  tile vs k=1: %d elements compared, %d differ. %d per-row pin comparisons (%d layers x "
          "every row the fixture pins), worst row reported per block."
          % (ncomp, ndiff_bits, njudge, len(layers)))


# ------------------------------------------------------------------ timing
@contextlib.contextmanager
def gpu_lock(tag: str):
    """Timings need the box to themselves (bench/build-queue.md). Another agent may hold it and
    waiting is normal; the release happens whatever this section does. Ownership is by tag, so
    acquire and release pass the identical string. `util` is printed at both ends and a busy box
    makes every number below provisional — a cold bandwidth figure measured while something else
    owns the memory controller is not comparable to the ledger's."""
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


def families(d: Path, layers, step: str) -> dict:
    """shape -> launches across every layer that has it, real weights, real operands. `PACK_*`
    entries are the launches an engine would actually issue; the named ones are the shapes the
    queue asks to report separately."""
    fam: dict[str, dict] = {}
    for layer in layers:
        if not (d / ("%s.L%02d.s_in.npy" % (step, layer))).exists():
            continue
        files = layer_files(layer, d, [step])
        ops = operands_for(d, step, layer)
        pack, out_name = ops["pack"], ops["out_proj"]
        key = "PACK_qkv" if out_name == "o_proj" else "PACK_inproj"
        for entry, files_list in [(n, [files["w"][n]]) for n, _ in pack] + \
                                 [(key, [files["w"][n] for n, _ in pack])]:
            rows = pack_rows(pack) if entry == key else dict(pack)[entry]
            f = fam.setdefault(entry, {"w": [], "x": [], "rows": rows, "in": 1024})
            f["w"].append(files_list)
            f["x"].append(files["xin"][step])
        f = fam.setdefault(out_name, {"w": [], "x": [], "rows": 1024, "in": 2048})
        f["w"].append([files["w"][out_name]])
        f["x"].append(files["xout"][step])
    return fam


def time_families(args, d: Path, layers, step: str, fails) -> None:
    print("\ncold per-shape timings (real weights from every layer that has the shape, %s;"
          " tile = replicas deep, added when the shape alone is smaller than the iGPU cache)"
          % step)
    for name, f in sorted(families(d, layers, step).items()):
        mib = sum(sum(Path(x).stat().st_size for x in ws) for ws in f["w"]) / 1048576.0
        # 16-row shapes are launch cost, not bandwidth: no tile makes 0.6 MiB cold, and that is
        # the point of folding them into the in_proj launch.
        tile = 1 if mib < 1.0 else max(1, int(np.ceil(96.0 / mib)))
        jobs = [("%s/%d" % (name, i), f["rows"], f["in"], f["x"][i], f["w"][i], None)
                for i in range(len(f["w"]))]
        plan = write_plan(WORK / ("plan_fam_%s.txt" % name), jobs)
        t = run_bin(["--mode", "burst", "--plan", str(plan), "--rows", str(args.rows),
                     "--iters", str(max(args.iters, 30)), "--tile", str(tile),
                     "--write-out"], name)
        line = ("  %-14s [%6d,%4d] x%-3d tile %-3d %8.2f us/launch %7.1f GB/s (%3.0f%% of probe)"
                % (name, f["rows"], f["in"], len(jobs), tile, float(t["ms_per_launch"]) * 1e3,
                   float(t["gbs"]), 100.0 * float(t["gbs"]) / PROBE_GBS))
        print(line)
        if args.sweep:
            sw = run_bin(["--mode", "burst", "--plan", str(plan), "--rows-list",
                          "1,2,4,8,16,32", "--iters", str(max(args.iters, 30)),
                          "--tile", str(tile), "--write-out"], name)
            pk = run_bin(["--mode", "burst", "--plan", str(plan), "--rows", str(args.rows),
                          "--iters", str(max(args.iters, 30)), "--tile", str(tile),
                          "--x-packed", "--write-out"], name)
            print("       rows/block sweep: %s" % sw["_rowsweep"])
            print("       x left BF16 in shared: %8.2f us/launch  %s"
                  % (float(pk["ms_per_launch"]) * 1e3, pk["_rowsweep"]))
        if float(t["gbs"]) > 240.0:
            fails.append("%s cold reports %.1f GB/s, above the ~240 GB/s ceiling: work is being"
                         " skipped or the working set is still cache-resident (raise --tile)"
                         % (name, float(t["gbs"])))


def time_token(args, d: Path, layers, step: str, fails) -> None:
    outdir = WORK / "out"
    jobs = []
    for layer in layers:
        if not (d / ("%s.L%02d.s_in.npy" % (step, layer))).exists():
            continue
        files = layer_files(layer, d, [step])
        js, _, _ = layer_jobs(layer, d, [step], files, outdir)
        jobs += js
    plan = write_plan(WORK / "plan_token.txt", jobs)
    # The plan's own bytes, computed the way the launcher accounts for them (weights + x + output,
    # one pass each at tokens = 1), so the roofline printed under `--only gdn|attn` is that family's
    # roofline and not the whole 0.4677 GB that all 24 layers cost.
    gb = sum(sum(Path(w).stat().st_size for w in wfiles) + Path(xfile).stat().st_size
             + rows * 2 for _, rows, _, xfile, wfiles, _ in jobs) / 1e9
    print("\ncold token plan (%s): %d launches, every layer's packed input projection then its"
          " output projection, in token order" % (step, len(jobs)))
    t = run_bin(["--mode", "burst", "--plan", str(plan), "--rows", str(args.rows),
                 "--iters", str(max(args.iters, 20))], "token")
    print("  projections for one token: %.4f ms | %.1f GB/s | %.0f%% of the %.0f GB/s probe"
          "  (roofline for %.4f GB: %.3f ms)"
          % (float(t["ms_per_plan"]), float(t["gbs"]), 100.0 * float(t["gbs"]) / PROBE_GBS,
             PROBE_GBS, gb, gb / PROBE_GBS * 1e3))
    if args.sweep:
        run_bin(["--mode", "burst", "--plan", str(plan), "--rows-list", "2,4,8,16,32",
                 "--iters", str(max(args.iters, 20))], "token sweep", quiet=False)
        print("  x left BF16 in shared:")
        run_bin(["--mode", "burst", "--plan", str(plan), "--rows-list", "2,4,8,16,32",
                 "--iters", str(max(args.iters, 20)), "--x-packed"], "token sweep packed",
                quiet=False)
    if float(t["gbs"]) > 240.0:
        fails.append("token plan %.1f GB/s above the ceiling" % float(t["gbs"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--layers", default="0,3,11,23", help="comma list, or 'all'")
    ap.add_argument("--steps", default="all", help="comma list of decodeN, or 'all'")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--tokens", default=None, metavar="PREFILL_FIXTURE",
                    help="P5 stage: run the token-batched form over this PREFILL fixture's"
                         " L{ll}.{in_proj_*,q_proj,k_proj,v_proj,out_proj,o_proj}.out pins"
                         " (p64/p300), judge them per row with the k=1 rules, and require the tile"
                         " rows to be BIT-IDENTICAL to the k=1 kernel run one row at a time.")
    ap.add_argument("--time", action="store_true", help="cold timings (takes the GPU lock)")
    ap.add_argument("--sweep", action="store_true", help="with --time: rows/block + x packing")
    ap.add_argument("--only", choices=["all", "gdn", "attn"], default="all",
                    help="with --time: measure one family's layers only (needs --time; the P1/P2"
                         " correctness gate always covers all nine projections of all 24 layers)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the operand-rebuild proof (it is the gate's own foundation)")
    args = ap.parse_args()
    if args.only != "all" and not args.time:
        raise SystemExit("--only is a timing selector: it needs --time. Without it the flag would"
                         " look like a correctness subset and do nothing, and the P1/P2 gate below"
                         " runs every projection of every layer whatever it says.")

    d = ROOT / "data" / "refs" / args.fixture
    if not d.exists():
        raise SystemExit("no fixture %s" % d)
    all_steps = sorted({p.name.split(".")[0] for p in d.glob("decode*.L00.s_in.npy")})
    steps = all_steps if args.steps == "all" else args.steps.split(",")
    if not steps:
        raise SystemExit("%s has no decode steps; this kernel is the k=1 path" % d)
    layers = list(range(24)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    if not BIN.exists() or any(src.stat().st_mtime > BIN.stat().st_mtime for src in
                               (BIN.parent / "proj_gemv.hip.cpp", BIN.parent / "proj_gemv_kernels.h")):
        # Staleness, not just existence: a gate that silently runs the previous binary passes
        # vacuously — one P5 pass did exactly that, green, on a source that had not compiled.
        subprocess.run(["bash", str(BIN.parent / "build.sh")], check=True)
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed - the gate cannot be trusted")
    WORK.mkdir(parents=True, exist_ok=True)
    print("fixture %s | %d layers x %d steps | rows/block %d | iters %d"
          % (args.fixture, len(layers), len(steps), args.rows, args.iters))

    fails: list[str] = []
    worst: dict = {}
    ncmp: list[str] = []

    # 1. The gate's own foundation: the rebuilt operands reproduce the pins bit-exactly and the
    #    wrong readings do not. Without this, every comparison below is against the wrong tensor.
    if not args.no_verify:
        print("\noperand rebuild (§03.2 norm, §04.5 gated norm, §05.6 output gate):")
        nbad = 0
        for layer in layers:
            for step in steps:
                for ok, line in verify(d, step, layer):
                    if not ok:
                        nbad += 1
                        print("    FAIL L%02d %s: %s" % (layer, step, line))
        print("    %d layers x %d steps checked, %d failures" % (len(layers), len(steps), nbad))
        if nbad:
            raise SystemExit("operand rebuild failed - stopping: a gate against the wrong tensor"
                             " verifies nothing")

    print("\nP1/P2 kernel output vs the pins:")
    gate_correctness(args, d, steps, layers, fails, worst, ncmp)

    if args.negative:
        run_negatives(args, d, fails)

    if args.tokens:
        run_tokens(args, fails, worst)

    if args.time:
        # `--only` narrows the timed layers to one family and nothing else. `is_attn_layer` reads
        # the fixture's own dump names, so the split is the checkpoint's (§01.3: 18 `linear_attn`
        # layers, 6 `self_attn`), not a layer list hardcoded here that could drift from the shard.
        timed = layers if args.only == "all" else [
            L for L in layers if is_attn_layer(d, L) == (args.only == "attn")]
        if not timed:
            raise SystemExit("--only %s matched no layer in --layers %s; the fixture names the"
                             " families in its dump files, so check the fixture"
                             % (args.only, args.layers))
        with gpu_lock("proj_gemv timing (run.py)"):
            print("\ntimed layers: %s (--only %s of %s); the gate above ran all %d"
                  % (",".join(str(L) for L in timed), args.only, args.layers, len(layers)))
            time_families(args, d, timed, steps[0], fails)
            time_token(args, d, timed, steps[0], fails)

    print("\nworst BF16 ULP by output -- gated (large elements only) | raw (also the cancelling"
          " elements floor_probe.py attributes to arithmetic, not error) | most elements any one"
          " comparison put inside the %g band | tightest band these results would still pass at:"
          % CANCEL_REL)
    slack = None
    for k, (bad, raw, nband, minband) in sorted(worst.items()):
        print("    %-16s gated %4d   raw %6d   in-band %7d   tightest passing band %s"
              % (k, bad, raw, nband, "%.1e" % minband if minband else "-"))
        if minband:
            slack = min(slack or 1e9, CANCEL_REL / minband)
    if slack:
        print("    band in use |ref|/max <= %g; worst offender above the band anywhere: %.1e"
              " -> %.1fx of slack. floor_probe.py puts the reference's own reordering floor"
              " (the largest |ref|/max that still moves >1 ULP) at the numbers it prints; a"
              " tighter band has to clear both." % (CANCEL_REL, max(v[3] for v in worst.values()),
                                                   slack))
    print("\n%d comparisons, %d failures" % (len(ncmp), len(fails)))
    if fails:
        print("FAIL")
        for f in fails[:30]:
            print("  -", f)
        return 1
    print("PASS: the projection family reproduces the reference's own operands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
