#!/usr/bin/env python3
"""Does bf16 -> fp16 hold this checkpoint exactly? Counted, not asserted.

    .venv/bin/python tools/fp16_range_audit.py [--model models/Qwen3.5-0.8B]
                                              [--refs data/refs/p300]
                                              [--selftest] [--all-tensors]

Why this exists. Route (a) of the prefill plan (`bench/plan-prefill-2026-10-12.md`, "Funded: prefill
compute") is fp16 arithmetic on the prefill path, and it rests on one claim about data: that the bf16
the checkpoint ships in converts to fp16 without loss. That claim is nearly the whole story and is
not the whole story. bf16 and fp16 are both 16 bits wide, but bf16 has 8 exponent bits and fp16 has 5,
so bf16 can name numbers fp16 cannot: everything above fp16's largest finite 65504 is in trouble, and
everything below fp16's smallest normal 2^-14 = 6.103515625e-5 lands in the subnormal band, where the
mantissa loses a bit per halving until one bit survives at 2^-24. In between, every bf16 value is
exactly representable in fp16 — and this script verifies that EXHAUSTIVELY, over all 65 536 bf16 bit
patterns, rather than spot-checking, because "every bf16 in range is exact" is exactly the sort of
sentence that gets written as a law and turns out to have a subnormal exception.

What is counted, per tensor and in total (`classify` is the single place that defines these, and the
pattern census below runs the same function on the whole bf16 space so the two cannot disagree):

  absmax / absmin      largest and smallest nonzero |value| seen;
  overflow             finite in the source, Inf after the cast. This is a ROUNDING question, not a
                       `> 65504` question: the round-half-even boundary against infinity is 65520, so
                       a finite value in (65504, 65520) converts to 65504. For the bf16 grid
                       specifically the two counts turn out to be equal — reported, not assumed;
  subnormal            finite, nonzero, |x| < 2^-14: representable in fp16, at reduced precision;
  inexact              the fp16 image is not equal to the source value (so: exact = bit-faithful).
                       Split into three so a reader can see WHERE the loss is:
        inexact_low      ... and |x| < 2^-14  (the subnormal band)
        inexact_normal   ... and inside fp16's normal range — ZERO for a BF16 source (the claim),
                             nonzero for an F32 source, which has 23 mantissa bits to fp16's 10;
        inexact_zero     ... and the value is 0 in the source or became 0 (a flush);
  nan / inf / zero / neg_zero     as present in the source.

The conversion is numpy's float64 -> float16 cast (IEEE round-half-even, subnormals honoured); the
helper doing it is self-tested on hand-computed cases including the 65520 boundary and a value only
round-half-even gets right, plus one injected tensor containing every failure mode so each counter is
seen to fire. A counter that has never fired is a decoration (`prototype/mlp_gemv/results.md`).

No device, no timing, no engine code, no torch.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

FP16_MAX = float(np.finfo(np.float16).max)        # 65504.0
FP16_MIN_NORMAL = float(np.float16(2.0 ** -14))   # 6.103515625e-5
FP16_MIN_SUBNORMAL = float(np.float16(2.0 ** -24))


# ---------------------------------------------------------------- conversion under audit
def to_fp16_widened(x: np.ndarray) -> np.ndarray:
    """Cast to fp16 with round-half-to-even, then widen back to float64 for an EXACT comparison.
    numpy's cast is the IEEE-conforming one, and `--selftest` proves that on the boundaries this
    audit leans on: a converter that flushed subnormals to zero, or rounded toward zero, would make
    the subnormal and inexact counts read low and the verdict would pass vacuously."""
    with np.errstate(over="ignore", invalid="ignore"):
        return x.astype(np.float16).astype(np.float64)


def from_bf16(bits: np.ndarray) -> np.ndarray:
    """BF16 bit patterns (uint16) -> float32, exactly. BF16 is the top half of an F32, so widening is
    a shift, not a conversion: no rounding, no libm, no torch."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------- the one definition of the counts
def classify(x: np.ndarray) -> dict:
    """Count what a bf16/F32 -> fp16 conversion does to `x`. `x` is the float64 image of the source
    values (exact for BF16, exact for F32). Every counter in this file comes out of here."""
    x64 = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x64)
    absx = np.abs(x64)
    y = to_fp16_widened(x64)

    band = finite & (absx > 0.0) & (absx < FP16_MIN_NORMAL)
    both_nan = np.isnan(x64) & np.isnan(y)
    eq = (x64 == y) | both_nan
    # +0 and -0 compare equal under IEEE, so sign is compared separately; a source zero that becomes
    # -0 (or the reverse) is a lost bit even though the values compare equal.
    sign_mismatch = (np.signbit(x64) != np.signbit(y)) & ~np.isnan(x64)
    bad = (~eq | sign_mismatch) & finite

    overflow = finite & np.isinf(y)
    zero_src = finite & (x64 == 0.0)
    became_zero = finite & ~zero_src & (y == 0.0)     # a flush: inexact, and worth naming
    # Four DISJOINT buckets, so `inexact` is their sum and a reader can see WHERE the loss is. An
    # overlap here is not cosmetic: the first version counted a flushed subnormal twice and the
    # partition check in --selftest is what caught it.
    inexact_sub = bad & band & ~became_zero
    inexact_flush = bad & became_zero
    inexact_normal = bad & ~band & ~overflow

    nz = finite & (absx > 0.0)
    # The single number that decides whether the loss matters: the largest ABSOLUTE change the
    # conversion makes to any value. A bf16 weight near 1.0 has a quantum of ~4e-3, so a conversion
    # error many orders below that cannot move a rounded result, and the gate bands do not have to
    # change on its account. Counting inexact values without this is a list with no scale. (The errstate
    # is not decoration: the pattern census feeds Inf and NaN through here, and an unlabelled warning in
    # an audit output is one more thing a reader has to decide to ignore.)
    with np.errstate(invalid="ignore"):
        err = np.abs(y - x64)
    err = np.where(finite & ~overflow, err, 0.0)
    return {
        "n": int(x64.size),
        "absmax": float(absx[nz].max()) if nz.any() else 0.0,
        "absmin": float(absx[nz].min()) if nz.any() else float("inf"),
        "zero": int(zero_src.sum()),
        "neg_zero": int((zero_src & np.signbit(x64)).sum()),
        "nan": int(np.isnan(x64).sum()),
        "inf_in": int(np.isinf(x64).sum()),
        "overflow": int(overflow.sum()),
        "subnormal": int(band.sum()),
        "inexact": int(bad.sum()),
        "inexact_sub": int(inexact_sub.sum()),
        "inexact_flush": int(inexact_flush.sum()),
        "inexact_normal": int(inexact_normal.sum()),
        "max_abs_err": float(err.max()) if err.size else 0.0,
    }


KEYS = ["n", "absmax", "absmin", "zero", "neg_zero", "nan", "inf_in", "overflow", "subnormal",
        "inexact", "inexact_sub", "inexact_flush", "inexact_normal", "max_abs_err"]


class Census:
    """One tensor (or a merge of tensors): the counters plus what they were counted over."""

    def __init__(self, name: str, dtype: str, shape: list, counts: dict | None = None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.c = counts or {k: (0.0 if k in ("absmax", "max_abs_err") else float("inf") if k == "absmin" else 0)
                            for k in KEYS}

    def feed(self, x: np.ndarray) -> None:
        self.merge(Census(self.name, self.dtype, self.shape, classify(x)))

    def merge(self, o: "Census") -> None:
        for k in KEYS:
            if k == "absmax":
                self.c[k] = max(self.c[k], o.c[k])
            elif k == "max_abs_err":
                self.c[k] = max(self.c[k], o.c[k])
            elif k == "absmin":
                self.c[k] = min(self.c[k], o.c[k]) if o.c["n"] else self.c[k]
            else:
                self.c[k] += o.c[k]

    def clean(self) -> bool:
        c = self.c
        return not (c["overflow"] or c["inexact"] or c["nan"] or c["inf_in"])

    def row(self) -> str:
        c = self.c
        return (f"{self.name:52s} {self.dtype:5s} [{','.join(map(str, self.shape)) or 'scalar'}]"
                f"{' ':<12} n={int(c['n']):>9d} absmax={c['absmax']:<10.6g} "
                f"ovf={int(c['overflow'])} inexact={int(c['inexact'])}"
                f"[sub {int(c['inexact_sub'])} norm {int(c['inexact_normal'])}"
                f" flush {int(c['inexact_flush'])}] subnorm={int(c['subnormal'])} "
                f"nan={int(c['nan'])} inf={int(c['inf_in'])} zero={int(c['zero'])}"
                f"(-0 {int(c['neg_zero'])})")


def audit_tensor(name: str, dtype: str, shape: list, raw: bytes) -> Census:
    c = Census(name, dtype, shape)
    if dtype == "BF16":
        c.feed(from_bf16(np.frombuffer(raw, dtype=np.uint16)))
    elif dtype == "F32":
        c.feed(np.frombuffer(raw, dtype=np.float32))
    elif dtype == "F16":
        c.feed(np.frombuffer(raw, dtype=np.float16))
    elif dtype in ("F64", "I64", "I32", "I16", "I8", "U8", "BOOL"):
        c.feed(np.frombuffer(raw, dtype=dtype.lower()))
    else:
        raise SystemExit(f"{name}: unhandled safetensors dtype {dtype}")
    return c


def iter_safetensors(model_dir: Path):
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        with open(model_dir / shard, "rb") as f:
            head_len = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(head_len).decode())
            base = 8 + head_len
            for name, meta in head.items():
                if name == "__metadata__":
                    continue
                s, e = meta["data_offsets"]
                f.seek(base + s)
                yield name, meta["dtype"], list(meta["shape"]), f.read(e - s)


def audit_model(model_dir: Path) -> list:
    return [audit_tensor(n, d, s, r) for n, d, s, r in iter_safetensors(model_dir)]


# Which audit bucket a tensor belongs to. Route (a) converts what a prefill GEMM multiplies by, so the
# headline verdict is about rank-2 tensors under `model.language_model.*` — everything else is
# reported, but it does not get to colour that verdict:
#   text-gemm  the projections, the MLP, and `embed_tokens` (which §01.1 aliases as `lm_head`);
#   text-aux   1-D norms / dt_bias / A_log and the rank-3 `conv1d.weight`: read by glue, gates and the
#              conv, not by a GEMM. The 36 F32 ones live here, and an F32 source has 23 mantissa bits
#              so it CANNOT convert exactly — lumping them in would make the verdict read "lossy" for
#              a reason that has nothing to do with the matrix units;
#   mtp        the draft block (39 MiB, §01) — same formats, separate verdict;
#   visual     the vision tower. Out of scope per AGENTS.md ("no work on the vision tower"), audited
#              because the task said the whole checkpoint, and excluded from the route verdict.
def group_of(name: str, shape: list) -> str:
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.visual."):
        return "visual"
    return "text-gemm" if len(shape) == 2 else "text-aux"


# `model.language_model.layers.12.mlp.up_proj.weight` -> `...layers.*.mlp.up_proj.weight`, so the
# printed table is one row per tensor KIND (15 rows) rather than one per tensor (488).
def leaf_of(name: str) -> str:
    return re.sub(r"layers\.\d+\.", "layers.*.", name)


def audit_refs(refs_dir: Path) -> list:
    """The activation side. The dumper writes BF16 tensors as a uint16 view (torch refuses
    `bf16.numpy()` — AGENTS.md), so a uint16 array here IS bf16 bits; float arrays are the F32 dumps.
    Read with mmap so a 2.4 GB fixture does not become a 2.4 GB RSS spike.

    Two families of file in a fixture directory are NOT activations and must not be audited as if they
    were: `*_fp.npy` FNV fingerprint arrays (float64 hashes of state, whose "values" are 64-bit
    integers wearing a float dtype — they produced `absmax=248069` and nine phantom "overflow"s on the
    first run of this tool), and `*input_ids*` token id arrays, which the dumper writes as float32 and
    which contain token ids, not activations. A carried file is named and skipped, not silently
    ignored — same rule as `tools/reference_dump.py --verify`."""
    out, skipped = [], []
    for p in sorted(refs_dir.rglob("*.npy")):
        rel = str(p.relative_to(refs_dir))
        if rel.endswith("_fp.npy") or "input_ids" in rel:
            skipped.append(rel)
            continue
        arr = np.load(p, mmap_mode="r")
        if str(arr.dtype) == "uint16":
            c = Census(rel, "BF16", list(arr.shape))
            c.feed(from_bf16(np.asarray(arr)))
        elif arr.dtype.kind in "fc":
            c = Census(rel, str(arr.dtype), list(arr.shape))
            c.feed(np.asarray(arr))
        else:
            skipped.append(f"{rel} (dtype {arr.dtype})")
            continue
        out.append(c)
    if skipped:
        print(f"  carried, not audited ({len(skipped)}): "
              + ", ".join(skipped[:4]) + (", ..." if len(skipped) > 4 else ""))
    return out


# ---------------------------------------------------------------- exhaustive pattern census
def pattern_census() -> dict:
    """All 65 536 BF16 bit patterns through `classify`. This is what turns 'every bf16 value inside
    fp16's range is exact' from an assertion into a checked fact. Two counters exist only here —
    how many patterns sit above 65504, and where the overflowing ones start and stop — because a
    per-tensor census has no use for them and `classify` stays one concept."""
    bits = np.arange(1 << 16, dtype=np.uint16)
    with np.errstate(invalid="ignore"):
        x = from_bf16(bits).astype(np.float64)
    pat = classify(x)
    pat["above_fp16_max"] = int((np.isfinite(x) & (np.abs(x) > FP16_MAX)).sum())
    ov = np.isfinite(x) & np.isinf(to_fp16_widened(x))
    pat["smallest_overflowing"] = float(np.abs(x)[ov].min()) if ov.any() else float("nan")
    pat["largest_overflowing"] = float(np.abs(x)[ov].max()) if ov.any() else float("nan")
    band = np.isfinite(x) & (np.abs(x) > 0) & (np.abs(x) < FP16_MIN_NORMAL)
    pat["smallest_in_band"] = float(np.abs(x)[band].min()) if band.any() else float("nan")
    return pat


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    fails: list[str] = []

    def check(ok: bool, what: str):
        print(("  [ok  ] " if ok else "  [FAIL] ") + what)
        if not ok:
            fails.append(what)

    cases = [
        (65504.0, 65504.0, "fp16 max finite stays finite"),
        (65519.0, 65504.0, "65519, below the 65520 half-way point, rounds DOWN to 65504 not Inf"),
        (65536.0, float("inf"), "65536, one fp16 ULP above the max, rounds to Inf"),
        (2.0 ** -14, 2.0 ** -14, "smallest fp16 normal is exact"),
        (2.0 ** -24, 2.0 ** -24, "smallest fp16 subnormal is exact, not flushed to zero"),
        (3.0 * 2.0 ** -24, 3.0 * 2.0 ** -24, "an odd multiple of the min subnormal is exact"),
        (2.0 ** -25, 0.0, "half the min subnormal rounds half-even to +0"),
        (3.0 * 2.0 ** -25, 4.0 * 2.0 ** -25, "a half-ULP value rounds half-even UP"),
        (5.0 * 2.0 ** -25, 4.0 * 2.0 ** -25, "a half-ULP value rounds half-even DOWN to even"),
        (-65504.0, -65504.0, "negative max finite"),
    ]
    for src, want, what in cases:
        got = float(to_fp16_widened(np.array([src]))[0])
        check(got == want, f"{what}  (got {got})")
    check(bool(np.signbit(to_fp16_widened(np.array([-0.0]))[0])), "-0.0 survives the round trip")

    # One injected tensor containing every failure mode, each with its expectation written next to it.
    poison = np.array([0.0,          # exact, counted as +0
                       -0.0,         # exact, counted as -0
                       70000.0,      # overflow -> Inf, and inexact
                       -70000.0,     # overflow -> -Inf, and inexact
                       1e-7,         # subnormal band, not a multiple of 2^-24 -> inexact_low
                       2.0 ** -24,   # subnormal band, exact
                       1234.5,       # normal range, 11 mantissa bits -> inexact_normal
                       float("nan"),
                       float("inf")], dtype=np.float32)
    c = Census("injected", "F32", [9])
    c.feed(poison)
    for key, want, what in [
        ("overflow", 2, "both injected overflows counted"),
        ("inexact", 4, "inexact = 2 overflow + 1 subnormal + 1 long-mantissa"),
        ("inexact_sub", 1, "the subnormal-band inexact is counted in its own bucket"),
        ("inexact_normal", 1, "a long-mantissa normal value is counted in its own bucket"),
        ("inexact_flush", 0, "nothing flushed here"),
        ("subnormal", 2, "both injected subnormals counted, exact or not"),
        ("nan", 1, "source NaN counted"),
        ("inf_in", 1, "source Inf counted"),
        ("zero", 2, "both zeros counted"),
        ("neg_zero", 1, "-0 distinguished from +0"),
        ("absmax", 70000.0, "absmax is the largest FINITE source value (Inf/NaN excluded)"),
    ]:
        got = c.c[key]
        check(got == want, f"injection: {what} (want {want}, got {got})")

    # The claim route (a) needs, checked over the whole bf16 space rather than believed.
    pat = pattern_census()
    check(pat["inexact_normal"] == 0,
          f"OVER ALL 65536 bf16 PATTERNS: none inexact inside fp16's normal range "
          f"(got {pat['inexact_normal']})")
    check(pat["inexact"] == pat["overflow"] + pat["inexact_sub"] + pat["inexact_flush"]
          + pat["inexact_normal"],
          "the bf16 inexact set partitions into overflow + subnormal + flush + normal "
          f"= {pat['overflow']}+{pat['inexact_sub']}+{pat['inexact_flush']}"
          f"+{pat['inexact_normal']} against inexact={pat['inexact']}")
    check(pat["overflow"] > 0 and pat["overflow"] == pat["above_fp16_max"],
          f"every finite bf16 pattern above 65504 also overflows to Inf "
          f"({pat['overflow']} vs {pat['above_fp16_max']})")
    check(pat["subnormal"] > 0 and pat["inexact_sub"] > 0,
          f"the subnormal band is populated ({pat['subnormal']} patterns) and "
          f"{pat['inexact_sub']} of them are inexact")
    return 1 if fails else 0


# ---------------------------------------------------------------- report
def by_leaf(censuses: list) -> list:
    """Aggregate per tensor KIND (layers.* folded). A reader wants `up_proj loses weight in every
    layer the same way`, not 24 identical rows."""
    out: dict = {}
    for c in censuses:
        key = (leaf_of(c.name), c.dtype)
        if key not in out:
            out[key] = Census(key[0], c.dtype, [])
        out[key].merge(c)
    return sorted(out.values(), key=lambda c: (-c.c["inexact"], -c.c["absmax"]))


def report(title: str, censuses: list, label: str) -> dict:
    print(f"\n=== {title}")
    total = Census(f"TOTAL {label}", "", [])
    for c in censuses:
        total.merge(c)
    if not censuses:
        print("  (empty)")
        return total.c
    # Split the totals by SOURCE dtype, because the two ask different questions. A BF16 source either
    # converts exactly or it does not, and that is the route question. An F32 source cannot convert
    # exactly, ever — 24-bit mantissa into 11 — so its "inexact in the normal range" count is the width
    # change doing what widths do, and reporting it as a defect would be theatre. What matters for an
    # F32 source is the RANGE: an fp16 operand must not overflow and must not need the subnormal band.
    by_src: dict = {}
    for c in censuses:
        by_src.setdefault(c.dtype, Census(f"  {c.dtype}-source", "", [])).merge(c)
    print(f"  {len(censuses)} tensors, {int(total.c['n'])} values")
    print(f"  {'kind':<50s}{'dtype':6s}{'n':>10s}{'absmax':>11s}{'ovf':>5s}{'inexact':>9s}"
          f"{'insub':>8s}{'flush':>7s}{'innorm':>7s}{'subnorm':>9s}{'zero':>6s}")
    for c in by_leaf(censuses):
        k = c.c
        print(f"  {c.name:<50s}{c.dtype:6s}{int(k['n']):>10d}{k['absmax']:>11.4g}"
              f"{int(k['overflow']):>5d}{int(k['inexact']):>9d}{int(k['inexact_sub']):>8d}"
              f"{int(k['inexact_flush']):>7d}{int(k['inexact_normal']):>7d}"
              f"{int(k['subnormal']):>9d}{int(k['zero']):>6d}")
    hi = max(censuses, key=lambda c: c.c["absmax"])
    lo = min((c for c in censuses if c.c["absmin"] < float("inf")), key=lambda c: c.c["absmin"])
    print(f"  largest |value|   {hi.c['absmax']:.6g}   in {hi.name}")
    print(f"  smallest nonzero  {lo.c['absmin']:.6g}   in {lo.name}")
    if hi.c["absmax"] > 0:
        print(f"  headroom to fp16's 65504: {FP16_MAX / hi.c['absmax']:.4g}x on the largest value")
    print(f"  TOTAL  absmax={total.c['absmax']:.6g} ovf={int(total.c['overflow'])} "
          f"inexact={int(total.c['inexact'])} [insub {int(total.c['inexact_sub'])} "
          f"innorm {int(total.c['inexact_normal'])} flush {int(total.c['inexact_flush'])}] "
          f"subnorm={int(total.c['subnormal'])} nan={int(total.c['nan'])} inf={int(total.c['inf_in'])} "
          f"zero={int(total.c['zero'])} (-0 {int(total.c['neg_zero'])})")
    for dt, sub in sorted(by_src.items()):
        if len(by_src) > 1:
            k = sub.c
            print(f"  {dt}-source  n={int(k['n'])} absmax={k['absmax']:.6g} ovf={int(k['overflow'])} "
                  f"inexact={int(k['inexact'])} [insub {int(k['inexact_sub'])} "
                  f"innorm {int(k['inexact_normal'])} flush {int(k['inexact_flush'])}] "
                  f"subnorm={int(k['subnormal'])}")
    return total.c


def verdict(label: str, c: dict, n_tensors: int) -> None:
    """One line per bucket, in the words the audit question asks: is the conversion lossless, and if
    not WHAT would have to be handled."""
    if not (c["overflow"] or c["inexact"] or c["nan"] or c["inf_in"]):
        print(f"  {label:<14s} LOSSLESS — 0 overflow, 0 inexact, 0 NaN, 0 Inf over {int(c['n']):,} "
              f"values in {n_tensors} tensors; max |value| {c['absmax']:.6g}.")
        return
    parts = []
    if c["overflow"]:
        parts.append(f"{int(c['overflow'])} would become Inf")
    if c["inexact_normal"]:
        parts.append(f"{int(c['inexact_normal'])} INEXACT IN FP16'S NORMAL RANGE")
    if c["inexact_sub"]:
        parts.append(f"{int(c['inexact_sub'])} rounded inside the subnormal band")
    if c["inexact_flush"]:
        parts.append(f"{int(c['inexact_flush'])} flushed to zero")
    if c["inexact"]:
        parts.append(f"largest absolute change any value suffers = {c['max_abs_err']:.3g}")
    if c["nan"] or c["inf_in"]:
        parts.append(f"{int(c['nan'])} NaN / {int(c['inf_in'])} Inf already present")
    print(f"  {label:<14s} NOT lossless — {'; '.join(parts)}. "
          f"{int(c['inexact']):,} of {int(c['n']):,} values "
          f"({100.0 * c['inexact'] / max(c['n'], 1):.5f} %); max |value| {c['absmax']:.6g}.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.5-0.8B"))
    ap.add_argument("--refs", default=None, help="a data/refs/<fixture> dir, for the activation side")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--all-tensors", action="store_true",
                    help="also print one row per tensor, not just per kind")
    args = ap.parse_args()

    if args.selftest:
        rc = selftest()
        print("SELFTEST " + ("PASS" if rc == 0 else "FAIL"))
        return rc

    pat = pattern_census()
    print("=== bf16 -> fp16, exhaustively over all 65 536 bf16 bit patterns")
    print(f"  exact {int(pat['n'] - pat['inexact'])} / {int(pat['n'])}   inexact {int(pat['inexact'])}")
    print(f"  of the inexact set: {int(pat['overflow'])} overflow to Inf, "
          f"{int(pat['inexact_sub'])} inside the subnormal band, "
          f"{int(pat['inexact_normal'])} in fp16's normal range, "
          f"{int(pat['inexact_flush'])} flushed to zero")
    print(f"  finite patterns > 65504: {int(pat['above_fp16_max'])}; the cast sends "
          f"{int(pat['overflow'])} of them to Inf (smallest {pat['smallest_overflowing']:.6g}, "
          f"largest {pat['largest_overflowing']:.6g})")
    print(f"  finite nonzero patterns in fp16's subnormal band (< 2^-14): {int(pat['subnormal'])}, "
          f"smallest {pat['smallest_in_band']:.6g}")
    print(f"  source NaN {int(pat['nan'])}, source Inf {int(pat['inf_in'])}, zeros {int(pat['zero'])}")
    print("  => the inexact set is EMPTY inside fp16's normal range, and every finite pattern above "
          "65504 becomes Inf: the whole risk surface is the subnormal band plus the overflow ceiling.")

    model = Path(args.model)
    if not model.is_absolute():
        model = ROOT / model
    censuses = audit_model(model)
    if args.all_tensors:
        for c in censuses:
            print("    " + c.row())

    groups: dict = {}
    for c in censuses:
        groups.setdefault(group_of(c.name, c.shape), []).append(c)
    print(f"\n  {len(censuses)} tensors, bucketed by what route (a) would actually convert "
          "(see group_of above).")
    totals = {g: report(f"{g} — {len(v)} tensors", v, g) for g, v in sorted(groups.items())}
    counts = {g: len(v) for g, v in groups.items()}

    if args.refs:
        refs = Path(args.refs)
        if not refs.is_absolute():
            refs = ROOT / refs
        acens = audit_refs(refs)
        if acens:
            totals["activations"] = report(f"pinned activations — {len(acens)} arrays: {refs}",
                                           acens, "activations")
            counts["activations"] = len(acens)

    print("\nVERDICT  (is bf16 -> fp16 lossless, and if not what has to be handled)")
    for g in ("text-gemm", "text-aux", "mtp", "visual", "activations"):
        if g in totals:
            verdict(g, totals[g], counts[g])
    print("  reading the buckets: text-gemm is what a prefill GEMM multiplies by and is therefore the "
          "route verdict; text-aux is norms / dt_bias / A_log / conv1d, which are not GEMM operands "
          "and include every F32 tensor (§01.6) — an F32 source cannot convert exactly and does not "
          "need to.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
