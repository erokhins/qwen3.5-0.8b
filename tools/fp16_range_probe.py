#!/usr/bin/env python3
"""fp16_range_probe.py — would any pinned tensor overflow or underflow fp16?

Why this exists. The matrix instruction on gfx1151 is WMMA and it is fp16-only (there is no
v_pk_fma_bf16 and no bf16 WMMA on this part — tools/isa_probe.sh, and devices/
strix-halo-official-specs.md section 7). So a matrix fast tier means fp16 operands with an f32
accumulator, and fp16 has a 5-bit exponent: max finite 65504, min normal 6.104e-5, min subnormal
6e-8. bf16 has the same exponent as f32, so "does it fit in bf16" has never been a question this
repo had to ask and there is no existing gate that would notice a value above 65504.

Standing authority for the decision this feeds (user, 2026-10-16): any 16-bit or 32-bit arithmetic
may be adopted if the gates still pass, or if the checks cannot be widened significantly. This
script does not make that decision; it measures the one thing fp16 has that bf16 does not — a
ceiling and a floor — so the decision has a number under it.

Read-only over the pinned dumps in data/refs. CPU only, no device, no GPU lock.

Comparison is done on the BF16 BIT PATTERN, not on converted floats: for IEEE binary formats the
magnitude ordering equals the ordering of the sign-stripped bit pattern, and a bf16 value is the
top 16 bits of the f32, so a bf16 tensor can be census-ed straight out of its uint16 dump with no
widening to f32 and no copy the size of the tensor. Sign-stripping means AND 0x7FFF. It does NOT
mean abs() of an int16 view: -12.5 is 0xC190, whose int16 absolute value is 0x3E80, which is the
bf16 pattern for +2.5. The first version of this script made that mistake and reported 351 tensors
"overflowing fp16"; the real offenders are the fp64 state dumps, not the logits.

  65504.0  as f32 bits 0x477FE000 -> bf16 bits 0x477F   (fp16 max: 0x477FE000 is exactly 65504)
  6.103515625e-05 as 0x38800000   -> bf16 bits 0x3880   (fp16 min normal)
  5.96e-08 as 0x33800000          -> bf16 bits 0x3380   (fp16 min subnormal)

Usage:
  tools/fp16_range_probe.py [dump ...]           # default: the three biggest pinned dump dirs
  tools/fp16_range_probe.py --verbose            # list every file, not just the offenders
  tools/fp16_range_probe.py --only lm_head       # substring filter on the file name
"""

import json
import sys
from pathlib import Path

import numpy as np

REFS = Path(__file__).resolve().parent.parent / "data" / "refs"
DEFAULT_DUMPS = ["dec300f", "p300", "wide1024f"]

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05
FP16_MIN_SUBNORMAL = 5.9604644775390625e-08

# bf16 bit patterns of the fp16 limits (see header)
BF_FP16_MAX = 0x477F
BF_FP16_MIN_NORMAL = 0x3880
BF_FP16_MIN_SUBNORMAL = 0x3380


def bf16_bits(x):
    """uint16 -> float32 by placing bf16 in the high half of an f32."""
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def census(path: Path):
    """Return a dict of range facts for one pinned tensor file, or None if not a numeric dump."""
    try:
        a = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - a JSON/npz side file in the dump dir is not a tensor
        return {"file": path.name, "skipped": f"unreadable: {exc}"}

    dt = a.dtype
    if dt == np.uint16:
        # Every pinned bf16 dump is stored as raw uint16 bits (torch refuses bf16.numpy();
        # see prototype/README.md). Magnitude = sign-stripped bits, i.e. AND 0x7FFF. See header.
        mag = np.asarray(a, dtype=np.uint16) & np.uint16(0x7FFF)
        finite = np.ones(mag.shape, dtype=bool)
        # 0x7F80 and above is inf/NaN in bf16
        not_finite = mag >= np.uint16(0x7F80)
        n_over = int((mag > BF_FP16_MAX).sum())
        n_sub = int(((mag > 0) & (mag < BF_FP16_MIN_NORMAL)).sum())
        n_flush = int(((mag > 0) & (mag < BF_FP16_MIN_SUBNORMAL)).sum())
        mx_bits = int(mag[~not_finite].max()) if (~not_finite).any() else 0
        maxabs = float(bf16_bits(np.array([mx_bits], dtype=np.uint16))[0]) if mx_bits else 0.0
        minabs_bits = mag[(mag > 0)]
        minabs = float(bf16_bits(np.array([int(minabs_bits.min())], dtype=np.uint16))[0]) if minabs_bits.size else 0.0
        nan_inf = int(not_finite.sum())
    elif dt.kind in "fc":
        f = np.asarray(a).ravel()
        fin = f[np.isfinite(f)]
        absf = np.abs(fin)
        maxabs = float(absf.max()) if absf.size else 0.0
        minabs = float(absf[absf > 0].min()) if (absf > 0).any() else 0.0
        n_over = int((absf > FP16_MAX).sum())
        n_sub = int(((absf > 0) & (absf < FP16_MIN_NORMAL)).sum())
        n_flush = int(((absf > 0) & (absf < FP16_MIN_SUBNORMAL)).sum())
        nan_inf = int(f.size - fin.size)
    else:
        return {"file": path.name, "skipped": f"dtype {dt}"}

    n = int(a.size)
    scale_needed = maxabs / FP16_MAX if maxabs > FP16_MAX else 1.0
    return {
        "file": path.name,
        "dtype": str(dt),
        "n": n,
        "max_abs": maxabs,
        "min_abs_nonzero": minabs,
        "n_overflow_fp16": n_over,
        "n_below_fp16_min_normal": n_sub,
        "n_below_fp16_min_subnormal": n_flush,
        "n_nonfinite_in_bf16": nan_inf,
        "max_abs_over_fp16_max": scale_needed,
    }


def classify(name: str) -> str:
    """Role of a pinned file, so a token-id tensor or a fingerprint row is not census-ed as if it
    were a value. The classes are the ones a datatype decision is actually made per, not the ones
    the dump directory happens to be sorted by.

    `*_fp` files are NOT full-precision values: each row is a fingerprint quadruple
    (magnitude-as-bits, count, delta, magnitude-as-bits) written by tools/reference_dump.py, so its
    first and fourth columns run to ~1e6 by construction. Census-ing them as values produced 276
    phantom "fp16 overflow" reports on the first run of this script.
    """
    n = name.lower()
    if "input_ids" in n or n in ("greedy_tokens.npy", "tokens.npy"):
        return "ids (not a value tensor)"
    if n.endswith("_fp.npy"):
        return "fingerprint rows (metadata)"
    if "logits" in n:
        return "logits"
    if "weight" in n or "lm_head" in n or "embed" in n:
        return "weights"
    return "activations"


def main(argv):
    verbose = "--verbose" in argv
    args = [a for a in argv if not a.startswith("--")]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1]
        args = args[:i] + args[i + 2:]
    dumps = args or DEFAULT_DUMPS

    rows, problems = [], []
    for name in dumps:
        d = REFS / name
        if not d.is_dir():
            print(f"skip {name}: no such dump dir", file=sys.stderr)
            continue
        for p in sorted(d.rglob("*.npy")):
            if only and only not in p.name:
                continue
            r = census(p)
            if r is None:
                continue
            if "skipped" in r:
                if verbose:
                    print(f"  skip {p.name}: {r['skipped']}")
                continue
            r["dump"] = name
            r["role"] = classify(p.name)
            rows.append(r)
            if r["n_overflow_fp16"] and r["role"] in ("activations", "logits", "weights"):
                problems.append(r)

    if not rows:
        print("nothing scanned — check the dump names", file=sys.stderr)
        return 2

    roles = {}
    for r in rows:
        g = roles.setdefault(r["role"], {"tensors": 0, "values": 0, "over": 0, "sub": 0,
                                         "flush": 0, "max": 0.0, "min": float("inf")})
        g["tensors"] += 1
        g["values"] += r["n"]
        g["over"] += r["n_overflow_fp16"]
        g["sub"] += r["n_below_fp16_min_normal"]
        g["flush"] += r["n_below_fp16_min_subnormal"]
        g["max"] = max(g["max"], r["max_abs"])
        if r["min_abs_nonzero"]:
            g["min"] = min(g["min"], r["min_abs_nonzero"])

    tot = sum(r["n"] for r in rows)
    sub_tot = sum(r["n_below_fp16_min_normal"] for r in rows)
    flush_tot = sum(r["n_below_fp16_min_subnormal"] for r in rows)

    print(f"scanned {len(rows)} pinned tensors, {tot:,} values, dumps: {', '.join(dumps)}")
    print()
    print("The only thing fp16 has that bf16 lacks is a range: 10 mantissa bits beats bf16's 7, so")
    print("inside its range fp16 is strictly MORE precise than the format the gates are calibrated")
    print("to. What can break is overflow above 65504 and flush-to-zero below ~5.96e-08. Those two")
    print("are what this table counts, per role, because the datatype decision is made per role:")
    print()
    print(f"  {'role':26s} {'tensors':>8s} {'values':>14s} {'max|x|':>11s} {'min|x|':>11s} "
          f"{'over 65504':>11s} {'subnormal %':>12s} {'flushed':>9s}")
    for role, g in sorted(roles.items()):
        mn = g["min"] if g["min"] != float("inf") else 0.0
        print(f"  {role:26s} {g['tensors']:8d} {g['values']:14,d} {g['max']:11.4g} {mn:11.4g} "
              f"{g['over']:11,d} {100.0*g['sub']/max(g['values'],1):11.4f}% {g['flush']:9,d}")

    print()
    real_problems = [r for r in problems]
    if real_problems:
        print(f"tensors with values above fp16 range ({len(real_problems)}), by role:")
        for r in sorted(real_problems, key=lambda r: -r["n_overflow_fp16"])[:12]:
            print(f"  {r['dump']}/{r['file']:42s} [{r['role']}] max {r['max_abs']:.6g}  "
                  f"{r['n_overflow_fp16']:,}/{r['n']:,} values over 65504  "
                  f"(scale {r['max_abs_over_fp16_max']:.3g} would fix)")
    else:
        print("no value in any scanned value-tensor exceeds fp16 range.")

    print()
    print(f"values below fp16 min normal overall: {sub_tot:,} ({100.0*sub_tot/tot:.4f} %); "
          f"below fp16 min subnormal, i.e. flushed to zero: {flush_tot:,}")
    verdict = ("PASS: fp16 range covers every scanned value-tensor; mantissa is a strict "
               "improvement on bf16, so a bf16-calibrated gate cannot be loosened by the switch"
               if not real_problems else
               "NOTE: at least one value-tensor exceeds fp16 range; that role needs a scale factor "
               "or must stay bf16/fp32 (see list above)")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
