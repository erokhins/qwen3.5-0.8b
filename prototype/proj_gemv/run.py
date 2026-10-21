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
                                               [--negative] [--time] [--sweep] [--rows 0]
                                               [--variant all] [--threads 0] [--grid-file F]
                                               [--iters 30] [--no-verify]
    .venv/bin/python prototype/proj_gemv/run.py --tokens p64      # P5: the token-batched form
    .venv/bin/python prototype/proj_gemv/run.py --time --only gdn  # one ledger row at a time
    .venv/bin/python prototype/proj_gemv/run.py --layers all --steps decode0 --emit-plans DIR

`--variant` (default `all`) is the D1 axis: the P1/P2 gate runs EVERY selected variant over every
projection of every layer against its own pin, diffs each non-reference variant against `v0-baseline`
(bit-exactly for the forms that claim bit-identity, band-wise for the lanes-per-row forms), and runs
every injected kernel bug through every variant — 49 cases for 7 bugs x 7 variants, each with its
must-catch and must-not-catch set intact. The names, the bit-identity claim and the launch geometry are
read back out of the binary (`--dump-plan`, which touches no device), so the C++ `g_var` table and this
file cannot drift apart silently. `--rows 0` (the default) means each variant's own rows/block; pass a
number to force one geometry. `--threads 0` (the default) means each variant's own
`rpb * lanes_per_row`, which is the shape every published number in `results.md` was taken with.

`--grid-file F` is what keeps the gate and the sweep honest. F names one point per line as
`variant[:rpb][@threads]`, and the gate then judges every projection of every layer at exactly those
(variant, rows/block, threads/block) triples instead of at each variant's favourite geometry, printing
`GATED n point(s)` for `sweep_decode.sh` to compare against the points it times. That is the D59 trap
named in AGENTS.md — a timing harness carrying a wider case list than the gate, so the report says
"PASS" about different bytes than the ones measured. Point 0 of a grid is the reference every other
point is diffed against, so a grid file starts with the shape the engine ships. `--emit-plans` writes the whole-token plans (packed / unpacked / mixed,
see `--plan-shape`) and exits without launching anything, which is how `sweep_decode.sh` times the same
plans the gate judged. See `HANDOFF-D1.md` for what the sweep is for and which levers it cannot price.

`--only gdn|attn` is a TIMING selector and needs `--time`: it narrows the cold timing path to the
layers of one family, which is how `engine/performance-status.md` §4's two projection rows — once "not
measured on its own" (progress/2026-10-16/performance-status-2026-10-16.md:81-82, GDN projections 0.3796 GB and attention
projections 0.0881 GB) and measured at last on 2026-10-16 — got their numbers instead of apportioned ones, the ask of `progress/2026-10-12/plan-2026-10-12.md` Q6 item 5. The P1/P2 correctness gate never
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


# ------------------------------------------------------------------ the D1 variant axis
# The variant table lives in the BINARY (`g_var` in proj_gemv.hip.cpp), not here: a list typed from
# memory in two places is one commit away from the `FATTN_WARPS` trap AGENTS.md records. The gate
# asks the binary what it supports and refuses to run on an empty parse.
VARIANT_EXPECT = ["v0-baseline", "v1-unroll", "v2-lpw16", "v3-lpw8", "v4-lpw4", "v5-vx",
                  "v6-lpw8-vx"]
# Which variants claim to be BIT-identical to v0 and which are band-judged because they reassociate
# the dot (lanes per row < 32). The kernel header states why each claim is what it is; the binary
# echoes `exact=0/1` on every run and `gate_variants` cross-checks this table against that echo, so
# a re-spelled kernel that quietly stops being bit-exact cannot keep a claim it no longer holds.
VARIANT_EXACT = {"v0-baseline": True, "v1-unroll": True, "v2-lpw16": False, "v3-lpw8": False,
                 "v4-lpw4": False, "v5-vx": True, "v6-lpw8-vx": False}
# Lanes per output row and the default rows per block, per variant. These are the geometry the sweep
# table must print, and `gate_variants` requires the binary's `--dump-plan` echo to agree, so the
# numbers in the report are read off the launcher rather than typed under it.
VARIANT_LPW = {"v0-baseline": 32, "v1-unroll": 32, "v2-lpw16": 16, "v3-lpw8": 8, "v4-lpw4": 4,
               "v5-vx": 32, "v6-lpw8-vx": 8}
VARIANT_RPB = {"v0-baseline": 8, "v1-unroll": 8, "v2-lpw16": 8, "v3-lpw8": 16, "v4-lpw4": 32,
               "v5-vx": 8, "v6-lpw8-vx": 16}
# The rows/block the CPU gate exercises per variant. The launcher's contract is that
# `rows_per_block * lanes_per_row` is a whole number of wavefronts, so for LPW = 4 the smallest
# legal value is 8 and the list below is legal for every variant; a value that is not is refused by
# the launcher with both numbers in the message, never silently rounded.
VARIANT_ROWS = {v: [8, 16, 32] for v in VARIANT_EXPECT}


def vrows(variant: str) -> list[int]:
    return VARIANT_ROWS[variant]


def binary_variants() -> list[str]:
    """The variant names the built binary accepts, read off its own refusal of a bad name.

    Reading them off the binary rather than typing them here is the guard for the `FATTN_WARPS` trap
    (AGENTS.md): a geometry or variant fact held in two files gets committed in one of them.
    """
    proc = subprocess.run([str(BIN), "--variant", "__no_such_variant__"], capture_output=True,
                          text=True)
    txt = proc.stdout + proc.stderr
    # One capture group, so findall hands back the names themselves. Written as `m[1] for m in` over
    # a single group it silently yields the SECOND CHARACTER of each name ("0" for 0=v0-baseline),
    # which is the empty-parse failure mode wearing a plausible-looking list.
    names = re.findall(r"\d+=([A-Za-z0-9_-]+)", txt)
    if not names:
        raise SystemExit("could not read the variant table out of %s; it must print "
                         "`--variant must be one of 0=name ...`. A gate that parsed zero variants "
                         "would judge nothing and print PASS." % BIN)
    if "v0-baseline" != names[0]:
        raise SystemExit("the binary's variant list does not start with v0-baseline, so the "
                         "control every other variant is diffed against is not at index 0")
    return names


def probe_operands() -> tuple[Path, Path]:
    """Throwaway [1024,1024] weight and [1024] input files so `--dump-plan` has something to size.
    Never launched: --dump-plan reads the weight file's LENGTH and nothing else."""
    w = WORK / "probe_w.bin"
    x = WORK / "probe_x.bin"
    if not w.exists() or w.stat().st_size != 1024 * 1024 * 2:
        w.write_bytes(bytes(1024 * 1024 * 2))
    if not x.exists() or x.stat().st_size != 1024 * 2:
        x.write_bytes(bytes(1024 * 2))
    return x, w


def gate_variants(args) -> list[str]:
    """The variant list this run will judge, cross-checked against the binary three ways: the names,
    the exact/bit-identical claim, and the geometry each variant launches with.

    The geometry check runs `--dump-plan`, which loads nothing and launches nothing, so it is free
    and it is what stops the sweep table from printing a thread count the launcher never uses."""
    names = binary_variants()
    if args.variant == "all":
        sel = names
    else:
        sel = [v.strip() for v in args.variant.split(",") if v.strip()]
        unknown = [v for v in sel if v not in names]
        if unknown:
            raise SystemExit("--variant %s is not in the binary's table %s" % (",".join(unknown),
                                                                               ",".join(names)))
    missing = [v for v in VARIANT_EXPECT if v not in names]
    if missing:
        raise SystemExit("the binary does not offer %s; the gate's table (g_var in "
                         "proj_gemv.hip.cpp) and this file disagree" % ",".join(missing))
    noband = [v for v in sel if VARIANT_EXACT.get(v) is None]
    if noband:
        raise SystemExit("no exact/not-exact claim for %s: add it to VARIANT_EXACT, next to the "
                         "claim it is supposed to record" % ",".join(noband))
    plan = WORK / "plan_probe.txt"
    px, pw = probe_operands()
    plan.write_text("probe 1024 1024 %s %s\n" % (px, pw))
    for v in sel:
        r = run_bin(["--mode", "once", "--plan", str(plan), "--dump-plan", "--variant", v], v)
        want = {"label": v, "lanes_per_row": str(VARIANT_LPW[v]),
                "rows_per_block": str(VARIANT_RPB[v]),
                "threads": str(VARIANT_LPW[v] * VARIANT_RPB[v]),
                "exact": str(int(VARIANT_EXACT[v]))}
        # v0 is the shipping kernel: its lanes_per_row is the wavefront width the binary reports as
        # `lanes_per_row=0` (0 means "whatever the wavefront is"), which is 32 on this part.
        if v == "v0-baseline":
            want["lanes_per_row"] = "0"
            want["threads"] = str(32 * VARIANT_RPB[v])
        bad = {k: (r.get(k), want[k]) for k, vv in want.items() if r.get(k) != vv}
        if bad:
            raise SystemExit("variant %s launched with %s, the gate expected %s (got %s) - the C++ "
                             "g_var table and VARIANT_LPW/VARIANT_RPB/VARIANT_EXACT have drifted"
                             % (v, bad, want, r))
    print("  variants %s; geometry and bit-identity claims read off the binary's --dump-plan echo"
          % ",".join(sel))
    return sel


def judge(tag: str, c: dict, fails: list[str], worst: dict, key: str | None = None) -> None:
    print("    %-40s %s" % (tag, fmt(c)))
    # `key` names the row of the worst-case table. Callers judging more than one variant pass
    # "<variant>/<block>", so the table says which form produced which worst case instead of
    # collapsing seven variants onto one line per block.
    key = tag.split()[-1] if key is None else key
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
        raise SystemExit("kernel failed: %s (%s)" % (label, proc.returncode))
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
    """One line per launch: NAME ROWS IN XFILE WFILE[,WFILE...] [OUTFILE] [VAR=name[:rows]].

    The 7th field is optional and per-launch; see `Job::variant` in proj_gemv.hip.cpp for why a plan
    has to be able to carry more than one variant.
    """
    with open(path, "w") as f:
        for job in jobs:
            name, rows, in_dim, xfile, wfiles, outfile = job[:6]
            f.write("%s %d %d %s %s%s%s\n"
                    % (name, rows, in_dim, xfile, ",".join(str(w) for w in wfiles),
                       (" " + str(outfile)) if outfile else "",
                       (" VAR=%s" % job[6]) if len(job) > 6 and job[6] else ""))
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


def layer_jobs(layer: int, d: Path, steps, files, outdir: Path, variant: str | None = None,
               rows: int | None = None, threads: int | None = None):
    """The launches a real token issues in this layer: the packed input projection, then the
    output projection. Returns (jobs, pack, out_name).

    `variant`, with the rows/block and threads/block that point was timed at, tags every launch
    (`VAR=name:rows[:threads]`) so ONE binary call runs every gated point's plan over ONE weight
    upload — and each point is judged at the geometry it will be timed at rather than at whatever the
    first point in the list uses. That is what makes "gated N, timed M, N==M" more than a slogan: the
    gate launches the same (variant, rows, threads) triples the sweep will time.
    """
    ops = operands_for(d, steps[0], layer)
    pack, out_name = ops["pack"], ops["out_proj"]
    tag = None
    if variant:
        # The field values only; `write_plan` spells the `VAR=` prefix.
        tag = variant + (":%d" % rows if rows else "") + (":%d" % threads if threads else "")
    jobs = []
    for step in steps:
        jobs.append(("%s.inproj" % step, pack_rows(pack), 1024, files["xin"][step],
                     [files["w"][n] for n, _ in pack], outdir / ("%s.inproj.bin" % step), tag))
        jobs.append(("%s.%s" % (step, out_name), 1024, 2048, files["xout"][step],
                     [files["w"][out_name]], outdir / ("%s.%s.bin" % (step, out_name)), tag))
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


def token_layer_jobs(layer: int, d: Path, step: str, files, outdir, shape: str,
                     small_variant: str = "v3-lpw8", small_rpb: int = 32, small_max: int = 100):
    """One layer's launches in a whole-token plan. `shape` is one of:

    `packed`   the prototype's folded form — one [8224,1024] input projection for a GDN layer and
               [5120,1024] for an attention layer, plus the output projection: 48 launches/token.
    `unpacked` one launch per projection: the **114** the engine ships today (`engine/decode_step.h`:
               18 GDN layers x {qkv,z,b,a,out} = 90 and 6 attention layers x {q,k,v,o} = 24). Same
               weight bytes as `packed`; what differs is 66 launches.
    `mixed`    those 114 launches with ONLY the small shapes — the [16,1024] `in_proj_b`/`in_proj_a`,
               which `results.md` times at 3.2 us to read 32 KiB (~5 % of the probe: launch and
               block-count cost, nothing else) — tagged `name:rows` for `small_variant` at
               `small_rpb`. Untagged jobs take the run's own `--variant`, so the big shapes run
               exactly as shipped and the small ones are priced in isolation.
    """
    ops = operands_for(d, step, layer)
    pack, out_name = ops["pack"], ops["out_proj"]
    jobs = []

    def tag(rows):
        # The field value only — `write_plan` spells the `VAR=` prefix. Tagging happens in `mixed`
        # only, and only for shapes at or below `small_max` rows.
        if shape != "mixed" or rows > small_max:
            return None
        return "%s:%d" % (small_variant, small_rpb)

    if shape == "packed":
        jobs.append(("L%02d.inproj" % layer, pack_rows(pack), 1024, files["xin"][step],
                     [files["w"][n] for n, _ in pack], outdir / ("L%02d_inproj.bin" % layer)))
    else:
        for name, rows in pack:
            jobs.append(("L%02d.%s" % (layer, name), rows, 1024, files["xin"][step],
                         [files["w"][name]], outdir / ("L%02d_%s.bin" % (layer, name)),
                         tag(rows)))
    jobs.append(("L%02d.%s" % (layer, out_name), 1024, 2048, files["xout"][step],
                 [files["w"][out_name]], outdir / ("L%02d_%s.bin" % (layer, out_name)),
                 tag(1024)))
    return jobs, pack, out_name


def emit_token_plans(args, d: Path, layers, step: str, out: Path) -> None:
    """Write the whole-token plans and leave. No launch, no allocation: the operand and weight files
    are the same ones the correctness gate judged, so the sweep times gated operands."""
    out.mkdir(parents=True, exist_ok=True)
    shapes = (["packed", "unpacked", "mixed"] if args.plan_shape == "both"
              else [args.plan_shape])
    for shape in shapes:
        jobs = []
        for layer in layers:
            if not (d / ("%s.L%02d.s_in.npy" % (step, layer))).exists():
                continue
            files = layer_files(layer, d, [step])
            j, _, _ = token_layer_jobs(layer, d, step, files, out, shape)
            jobs += j
        path = write_plan(out / ("plan_token_%s.txt" % shape), jobs)
        gb = sum(sum(Path(w).stat().st_size for w in wf) for _, _, _, _, wf, *_ in jobs) / 1e9
        ntag = sum(1 for j in jobs if len(j) > 6 and j[6])
        print("plan %s: %d launches, %.4f GB of weights (%.4f ms at the %.0f GB/s probe)"
              "%s" % (path.name, len(jobs), gb, gb / PROBE_GBS * 1e3, PROBE_GBS,
                      "; %d jobs variant-tagged" % ntag if shape == "mixed" else ""))


# ------------------------------------------------------------------ correctness
def variant_args(args, pt: dict) -> list[str]:
    """The run-level flags for one gated point: its variant, its rows/block and its threads/block.
    Every job in the plan also carries its own `VAR=` tag, so these are what the launcher ECHOES (the
    canary reads them back) as much as what it obeys. `threads` 0 says "the variant's own", which is
    `rpb * lanes_per_row` — the shape every published number in results.md was taken with.
    """
    return ["--variant", pt["variant"], "--rows", str(pt["rpb"]), "--threads", str(pt["threads"] or 0)]


def rows_or(args) -> int:
    """`--rows 0` (the default) means "each variant's own", which for v0 is the shipped 8. Every
    call site that is not variant-aware reads it through here, so no code path can pass 0 to the
    launcher by accident."""
    return args.rows or 8


def layer_outputs(outdir: Path, steps, pack, out_name) -> dict:
    """This layer's launched outputs as {(step, block): array}, read once per variant so the pin
    comparison and the cross-variant comparison are looking at the same bytes."""
    offs = pack_offsets(pack)
    got = {}
    for step in steps:
        whole = read_bf16(outdir / ("%s.inproj.bin" % step), pack_rows(pack))
        for name, n in pack:
            off = offs[name][0]
            got[(step, name)] = whole[off:off + n]
        got[(step, out_name)] = read_bf16(outdir / ("%s.%s.bin" % (step, out_name)), 1024)
    return got


def pkey(pt: dict) -> str:
    """The label for one gated/timed point: variant, rows/block and threads/block. Threads show as
    `t0` when the point asked for the variant's own (rpb*lanes), which is what every published number
    here was taken with."""
    return "%s@rpb%dt%s" % (pt["variant"], pt["rpb"] or 0, pt["threads"] or 0)


def parse_grid(args, variants) -> list[dict]:
    """The (variant, rows/block, threads/block) points to gate, as a list of dicts.

    With `--grid-file`, one point per line: `variant[:rpb][@threads]` (blank lines and `#` comments
    ignored). Without it, one point per selected variant at its own rows/block and `--threads`.

    The point list is what makes the gate and the sweep talk about the same thing. Timing a
    (variant, rpb, threads) triple that was never gated is the D59 trap — a report that says "40/40
    PASS" about a case list that is not the case list that was measured — so `sweep_decode.sh` passes
    the same file to both phases and asserts `gated N == timed M`.
    """
    def one(spec: str) -> dict:
        body, _, thr = spec.partition("@")
        name, _, rpb = body.partition(":")
        if name not in variants:
            raise SystemExit("grid point %r names %r, which is not a selected variant (%s)"
                             % (spec, name, ",".join(variants)))
        return {"variant": name,
                "rpb": int(rpb) if rpb else (rows_or(args) or VARIANT_RPB[name]),
                "threads": int(thr) if thr else args.threads}

    if args.grid_file:
        pts = [one(s.strip()) for s in Path(args.grid_file).read_text().splitlines()
               if s.strip() and not s.strip().startswith("#")]
        if not pts:
            raise SystemExit("--grid-file %s has no points; a grid that can be empty cannot fail"
                             % args.grid_file)
        seen = [pkey(p) for p in pts]
        if len(set(seen)) != len(seen):
            dup = sorted({s for s in seen if seen.count(s) > 1})
            raise SystemExit("grid file repeats a point: %s" % ",".join(dup))
        return pts
    return [one(v) for v in variants]


def gate_correctness(args, d: Path, steps, layers, fails, worst, ncmp, points) -> None:
    """Every projection of every layer, run through every gated POINT — a variant at a specific
    rows/block and threads/block — judged against its own pin with the k=1 bands, AND against the
    first point's output element for element.

    The two comparisons answer different questions and both are needed. **vs the pin** is the
    variant reproducing the reference: for an `exact` variant that is the same test the shipping
    form already passes, and for an LPW variant it is the band test — the module's own constants,
    unchanged, which is the statement that no band was widened. **vs point 0** is the only check
    that can see a variant move a bit it promised not to: judged only against a pin inside a band, a
    variant could differ from the shipping kernel anywhere inside that band and nothing would
    notice, which for the bit-exact forms is exactly the failure that moves the legal route every
    fixture margin was priced against.

    Every comparison is counted and the count is asserted against the population the loop claims to
    have walked — a check over a list that can silently arrive empty cannot fail (decision log D40).
    """
    from speccheck.specimpl import Weights
    w = Weights(CKPT)          # loads by the §01 names and re-checks every shape (G1)
    variants = [p["variant"] for p in points]
    npins = nagree = nbit = 0
    pop = 0                    # (layer, step, block) cells the loop actually walked
    for layer in layers:
        files = layer_files(layer, d, steps)
        outs, vof, jobs, pack, out_name = {}, {}, [], None, None
        offs = None
        for pt in points:
            outdir = WORK / ("out-%s" % pkey(pt))
            outdir.mkdir(exist_ok=True)
            jv, pack, out_name = layer_jobs(layer, d, steps, files, outdir, pt["variant"],
                                            pt["rpb"], pt["threads"])
            jobs += jv
            vof[pkey(pt)] = outdir
        plan = write_plan(WORK / ("plan_L%02d.txt" % layer), jobs)
        res = run_bin(["--mode", "once", "--plan", str(plan)] + variant_args(args, points[0]),
                      "L%02d %s" % (layer, ",".join(pkey(p) for p in points)))
        # The binary echoes what it ACTUALLY ran. Trusting the flags we passed instead is how a
        # stale binary produces a clean null (AGENTS.md, the D30 1.7-prime A/B): the plan carried
        # len(variants) tagged groups, so the launch count is the canary that it ran all of them.
        if res.get("launches") != str(len(jobs)):
            fails.append("L%02d: the plan carried %d launches, the binary reported %s"
                         % (layer, len(jobs), res.get("launches")))
        for k in vof:
            outs[k] = layer_outputs(vof[k], steps, pack, out_name)
        offs = pack_offsets(pack)
        # The population is the layer's OWN pack: 5 blocks for a GDN layer (qkv, z, b, a, out_proj),
        # 4 for an attention one (q, k, v, o_proj). Hardcoding a number here would make the count
        # assertion fail on a correct run, and a gate that cries wolf gets ignored, so the count is
        # derived from the pack the layer actually has.
        pop += len(steps) * (len(pack) + 1)
        print("  L%02d %-4s pack %s + %s  points: %s"
              % (layer, "attn" if is_attn_layer(d, layer) else "gdn",
                 ",".join(n for n, _ in pack), out_name, ",".join(pkey(p) for p in points)))
        xkey_pre = "self_attn" if is_attn_layer(d, layer) else "linear_attn"
        for step in steps:
            x_in = operands_for(d, step, layer)["x_in"].reshape(1, -1)
            for name in [n for n, _ in pack] + [out_name]:
                ref = pin(d, "%s.L%02d.%s.out" % (step, layer, name))
                for pi, pt in enumerate(points):
                    tag, variant = pkey(pt), pt["variant"]
                    got = outs[tag][(step, name)]
                    judge("%s %-21s %s" % (step, tag, name), compare(got, ref), fails, worst,
                          key=name if pi == 0 else "%s/%s" % (tag, name))
                    ncmp.append(name)
                    npins += 1
                    if pi != 0 or name == out_name:
                        continue
                    # Third path: speccheck's own torch CPU expression on the same operand — for
                    # the packed input projections only, exactly as before this file grew variants
                    # (out_proj/o_proj reads the [*,2048] gated block output, a different weight and
                    # a different x, and the pin check above is its third path). It re-derives the
                    # PIN, so it says nothing about a kernel variant, and running it per variant
                    # would bill a torch matmul seven times for one fact about the dump.
                    sc = x_in @ w.p["model.language_model.layers.%d.%s.%s.weight"
                                    % (layer, xkey_pre, name)].t()
                    if int((sc.reshape(-1) != ref).sum()):
                        fails.append("L%02d %s %s: speccheck's own matmul is not bit-exact to the"
                                     " pin - the operand rebuild or the pin pairing is wrong"
                                     % (layer, step, name))
                base = outs[pkey(points[0])][(step, name)]
                for pt in points[1:]:
                    tag, variant = pkey(pt), pt["variant"]
                    got = outs[tag][(step, name)]
                    nd = int((got != base).sum())
                    nbit += got.numel()  # torch tensor: .size is a method, .numel() is the count
                    nagree += 1
                    if VARIANT_EXACT[variant]:
                        if nd:
                            fails.append("L%02d %s %s: %s differs from %s in %d of %d elements, but"
                                         " the variant table claims it is bit-identical"
                                         % (layer, step, variant, name, nd, got.numel()))
                        continue
                    cv = compare(got, base)
                    judge("%s %-21s %s vs %s" % (step, tag, name, pkey(points[0])), cv, fails,
                          worst, key="%s/%s-agree" % (tag, name))
                    # Deliberately NO extra `maxulp <= 1` rule here, tempting as it looks for "a
                    # re-associated sum may only move ~1 ULP". Raw max-ULP is the wrong instrument
                    # and the gate's own numbers prove it: on L18 `in_proj_z` the SHIPPING kernel is
                    # already 40 ULP from the pin at one element of 2048 whose |ref|/max is 1.9e-08,
                    # inside the cancellation band (results.md §F1: the reference's own F32 sum
                    # already moves cancelling elements), and the LPW forms move that same element
                    # 96-384 ULP and no other. `over-tol large` is 0 for all of them, which is the
                    # band-relative statement of the same fact. So the bound stays the one `judge`
                    # already applies — no element OUTSIDE the band past 1 ULP, ndiff inside the
                    # band, cos, meanrel — and it is byte for byte the four criteria and constants the
                    # pins are judged by, not a set widened to accommodate a variant.
    print("  GATED %d point(s): %s" % (len(points), ", ".join(pkey(p) for p in points)))
    print("  %d pin comparisons, %d cross-point comparisons over %d elements"
          % (npins, nagree, nbit))
    want = pop * len(points)
    if npins != want:
        fails.append("the variant gate judged %d block comparisons but the population is %d"
                     " (every layer's own blocks x steps x points) - the loop did not walk what it"
                     " claims to" % (npins, want))
    if nagree != pop * (len(points) - 1):
        fails.append("the variant gate made %d cross-point comparisons, expected %d"
                     " - every non-reference point must be diffed against %s on every block"
                     % (nagree, pop * (len(points) - 1), pkey(points[0])))


# ------------------------------------------------------------------ negatives
def run_negatives(args, d: Path, fails, variants) -> None:
    gdn = next(l for l in range(24) if not is_attn_layer(d, l))
    attn = next(l for l in range(24) if is_attn_layer(d, l))
    step = sorted(p.name.split(".")[0] for p in d.glob("decode*.L%02d.s_in.npy" % gdn))[0]
    outdir0 = WORK / "neg"
    outdir0.mkdir(exist_ok=True)
    print("\nnegative tests (rejected where declared, and nowhere else), every variant:")
    # The negatives are per VARIANT at its default geometry, not per gated point: an injected bug lives
    # in the kernel body, and rows/block and threads/block are launch geometry around that body. The
    # grid answer to "was the timed shape judged?" is the pin comparison and the cross-point diff in
    # gate_correctness; the count printed here says which variants were bug-tested, not which points.
    cases = {"gdn": gdn, "attn": attn}
    ncatch = 0
    for name, (why, kind, scope, must, must_not) in NEG_KERNEL.items():
        layer = cases[kind]
        files = layer_files(layer, d, [step])
        for variant in variants:
            outdir = WORK / ("neg-%s" % variant)
            outdir.mkdir(exist_ok=True)
            jobs, pack, out_name = layer_jobs(layer, d, [step], files, outdir, variant,
                                             rows_or(args) if args.rows else VARIANT_RPB[variant])
            if scope == "inproj":
                jobs = [j for j in jobs if j[0].endswith(".inproj")]
            plan = write_plan(WORK / ("plan_neg_%s_%s.txt" % (name, variant)), jobs)
            run_bin(["--mode", "once", "--plan", str(plan), "--neg", name,
                     "--variant", variant, "--rows",
                     str(rows_or(args) or VARIANT_RPB[variant])], name)
            caught = blocks_after_run(layer, d, step, pack, outdir, out_name,
                                      want_out=scope == "all")
            ncatch += 1
            missing = [m for m in must if m not in caught]
            wrong = [m for m in must_not if m in caught]
            print("    --neg %-10s %-11s L%02d %-34s caught %-44s must-not: %s%s"
                  % (name, variant, layer, why, ",".join(caught) or "NOTHING",
                     ",".join("%s=%s" % (m, "clean" if m not in caught else "MOVED")
                              for m in must_not) or "-",
                     "" if not wrong else "  <-- " + ",".join(wrong)))
            if missing:
                fails.append("negative %s on %s was NOT rejected at %s - the gate is vacuous"
                             % (name, variant, ",".join(missing)))
            if wrong:
                fails.append("negative %s on %s moved %s, which it should not touch"
                             % (name, variant, ",".join(wrong)))
    if ncatch != len(NEG_KERNEL) * len(variants):
        raise SystemExit("ran %d kernel-injection cases, expected %d (%d injections x %d variants):"
                         " a negatives loop that skips a variant or an injection cannot fail"
                         % (ncatch, len(NEG_KERNEL) * len(variants), len(NEG_KERNEL),
                            len(variants)))
    # Operand negatives: same kernel, wrong INPUT tensor. Must move the launches that read it and
    # must leave the one that reads the other operand alone. Run on the reference variant only: what
    # they falsify is the COMPARISON ("the operands are not compared against themselves"), which is a
    # property of the dump and the reading, not of the split; the per-variant falsification is the
    # cross-variant diff in `gate_correctness` plus the injected-bug loop above.
    for kind, layer in cases.items():
        files = layer_files(layer, d, [step])
        jobs, pack, out_name = layer_jobs(layer, d, [step], files, outdir0)
        if kind == "gdn":
            bad_inputs = wrong_normed_inputs(d, step, layer)
            where = "inproj"
            must = IN_PROJ_BLOCKS
            must_not = [out_name]
        else:
            bad_inputs = {"contiguous-half gate (§05.1)":
                          attn_operand(d, step, layer, interleave="half")}
            where = "outproj"
            must = [out_name]
            must_not = ATTN_BLOCKS
        target = ".inproj" if where == "inproj" else "." + out_name
        for label, xx in bad_inputs.items():
            slug = re.sub(r"\W+", "_", label)[:24]
            badx = dump_bf16(xx, WORK / ("L%02d_bad_%s.bin" % (layer, slug)))
            j2 = [(nm, r, ind, badx if nm.endswith(target) else xf, wf, of)
                  for nm, r, ind, xf, wf, of, *_ in jobs]
            plan = write_plan(WORK / ("plan_neg_in_%s.txt" % slug), j2)
            run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(rows_or(args))], label)
            caught = blocks_after_run(layer, d, step, pack, outdir0, out_name)
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
        run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(rows_or(args)),
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
            run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(rows_or(args)),
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
        run_bin(["--mode", "once", "--plan", str(plan), "--rows", str(rows_or(args)),
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
    """Timings need the box to themselves (progress/2026-10-15/build-queue.md). Another agent may hold it and
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
        t = run_bin(["--mode", "burst", "--plan", str(plan), "--rows", str(rows_or(args)),
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
            pk = run_bin(["--mode", "burst", "--plan", str(plan), "--rows", str(rows_or(args)),
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
    t = run_bin(["--mode", "burst", "--plan", str(plan), "--rows", str(rows_or(args)),
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
    ap.add_argument("--rows", type=int, default=0,
                    help="rows per block; 0 (the default) means each variant's own, which for "
                         "v0-baseline is the shipped 8")
    ap.add_argument("--variant", default="all",
                    help="comma list of variant names, or 'all' (default): the P1/P2 gate runs "
                         "every one and diffs each against v0-baseline")
    ap.add_argument("--threads", type=int, default=0,
                    help="threads/block for the timed/gated shape; 0 (default) = each variant's own"
                         " rpb*lanes, which is what every published number here used. Only the"
                         " prototype-only variants (v1-v6) take a free count: v0 is the shipped"
                         " kernel and the launcher refuses a shape the engine cannot emit.")
    ap.add_argument("--grid-file", default="",
                    help="gate exactly the (variant[:rpb][@threads]) points listed here, one per"
                         " line, so the gated set IS the timed set. sweep_decode.sh writes this file"
                         " and passes the same one to both phases, then asserts gated == timed.")
    ap.add_argument("--plan-shape", choices=["packed", "unpacked", "mixed", "both"],
                    default="both",
                    help="with --time/--emit-plans: the 48-launch packed plan the prototype builds, "
                         "the 114-launch plan the engine ships today, the same 114 launches with "
                         "only the 16-row shapes variant-tagged, or all three (default both)")
    ap.add_argument("--emit-plans", metavar="DIR", default=None,
                    help="write the token plan file(s) into DIR and exit: no GPU work at all, so "
                         "sweep_decode.sh can build the plans before it takes the lock")
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
    # After the build: `gate_variants` runs the binary to read its variant table, so on a fresh
    # checkout it would otherwise be probing a binary that does not exist yet.
    variants = gate_variants(args)
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed - the gate cannot be trusted")
    WORK.mkdir(parents=True, exist_ok=True)
    print("fixture %s | %d layers x %d steps | rows/block %s | variants %s | iters %d"
          % (args.fixture, len(layers), len(steps),
             "per-variant" if not args.rows else str(args.rows), ",".join(variants), args.iters))

    if args.emit_plans:
        # Plan construction only — operands, weights and plan files, no launch. This is what
        # sweep_decode.sh runs before it takes the GPU lock, so the plans the sweep times are the
        # same ones the gate judged rather than a second spelling of the token plan.
        emit_token_plans(args, d, layers, steps[0], Path(args.emit_plans))
        return 0

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
    points = parse_grid(args, variants)
    gate_correctness(args, d, steps, layers, fails, worst, ncmp, points)

    if args.negative:
        run_negatives(args, d, fails, variants)

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
