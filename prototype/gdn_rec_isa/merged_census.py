#!/usr/bin/env python3
"""The MERGED-form census (D46 / `bench/performance-status.md` §18): does `rec2c`'s fused prologue survive
being merged into the shipping unroll4 walks, and do the walks survive the merge?

WHY A CENSUS AND NOT ANOTHER BURST. The §18 screen compared `rec2c` (fused prologue, PRE-unroll4 walks)
against `rec0` (also pre-unroll4), while the engine runs `engine/gdn_rec_unroll4_kernels.h` (D33). A merge
that loses the unroll to register pressure produces a kernel that is fused and slower, and no burst of the
two unmerged forms can see it, because that kernel is not in the binary the burst ran. So the question is
answered off a listing, compile-only, before any device window is spent.

WHAT IT PRINTS, in the order the questions are asked:
  0. PARSE ASSERTIONS. every named kernel must exist in the listing, the parsed kernel count must be
     non-zero, and each censused kernel must bill non-zero slots. A census that parsed nothing used to print
     silence as a result (`prototype/gdn_collapse/run.py`'s class of defect, and `gdn_collapse` §1 of its
     own results), so all three are assertions, not decoration.
  1. UNROLL SURVIVAL, per walk: `state-accesses/trip` is the unrolled-body width. The shipping unroll4
     walks are 16 accesses x 8 trips in BOTH loops; the rolled form is loop A 16x8 with loop B 8x16. Loop
     B's width is the discriminator, because that is the loop the rolled fallback shortens.
  2. PROLOGUE SURVIVAL: the fused pair leaves a fingerprint the two-pass form cannot produce — 5 `s_barrier`
     instead of 11, and 3 permute-then-barrier groups instead of 5 (three inside `block_sum_pair_comp`, two
     around the `ks`/`qs` staging).
  3. THE BILL, per token-step per wavefront, split by region, and a per-category diff of the two named forms.
  4. RESOURCES, per kernel: VGPRs, spill, SGPRs, LDS, scratch, occupancy. Read from a compile pass, and
     keyed by the demangled name — `isa_census.py`'s own resource column truncates mangled names at the
     first digit, which prints `rec1_kernel`, `rec2_kernel` and `rec2c_kernel` as three identical `rec`
     rows (visible in `isa_census.output.txt`), i.e. a table that cannot attribute a register count.

`--expect merged` turns 1-2 into a verdict and a non-zero exit: the A/B script calls it as its build-time
canary, because "the merged kernel really is fused AND really is unrolled" is exactly the property a stale
or silently-regressed binary would fail.

Usage:
    prototype/gdn_rec_isa/merged_census.py <device.s> rec_unroll4_kernel,rec2c_kernel,rec2c4_kernel \
        [--resource-tu prototype/gdn_rec_isa/isa_tu_merged.cpp --include . --include engine \
--include prototype/gdn_rec_isa] [--expect merged]
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def load_census(name="gdn_rec_isa_census"):
    """Import the gdn census under a UNIQUE module name.

    It imports the slim parser as `isa_census` (`prototype/attn_valu_slim/isa_census.py:39` does
    `import isa_census as base`), so registering this directory's file in sys.modules under the plain name
    `isa_census` shadows that import and hands slim the wrong parser module.
    """
    spec = importlib.util.spec_from_file_location(name, HERE / "isa_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = load_census()
RES_FIELDS = ("VGPRs", "VGPRs Spill", "TotalSGPRs", "SGPRs Spill", "LDS Size [bytes/block]",
              "ScratchSize [bytes/lane]", "Occupancy [waves/SIMD]")


def permute_groups(prologue_insts) -> int:
    """How many separate cross-lane sweeps the prologue makes: a barrier that follows a permute ends one."""
    groups, seen = 0, False
    for _, ln in prologue_insts:
        if "ds_bpermute" in ln or "ds_swizzle" in ln:
            seen = True
        elif seen and re.match(r"\s*s_barrier\b", ln):
            groups += 1
            seen = False
    return groups


def bill(kernels: dict[str, list[str]], names: list[str]) -> dict[str, dict]:
    out = {}
    for n in names:
        c = C.census_kernel(kernels[n])
        slots = C.slots(c["step"])
        assert slots > 0, f"{n}: billed 0 slots — a broken parse, not a cheap kernel"
        cats = C.tally_cats(c["step"])
        pro_cats = C.tally_cats(c["regions"]["prologue"])
        assert pro_cats.get("xlane.permute", 0) > 0, \
            f"{n}: no cross-lane permute in the prologue — the norm reductions are not where the census "\
            f"expects them, so the prologue columns below mean nothing"
        out[n] = {
            "total": slots,
            "prologue": C.slots(c["regions"]["prologue"]),
            "loopA": C.slots(c["regions"]["loopA"]) * c["trips"]["loopA"],
            "loopB": C.slots(c["regions"]["loopB"]) * c["trips"]["loopB"],
            "cats": cats,
            "barriers": cats.get("barrier.sync", 0),
            "permutes": cats.get("xlane.permute", 0),
            "groups": permute_groups(c["regions"]["prologue"]),
            "walks": {w: (c["walks"][w]["per_trip"], c["walks"][w]["trips"])
                      for w in c["walks"]},
        }
    return out


def resources(tu: Path, includes: list[str], names: list[str]) -> dict[str, dict[str, str]]:
    cmd = ["hipcc", "-O3", f"--offload-arch={C.ARCH}", "-Rpass-analysis=kernel-resource-usage",
           "-c", str(tu), "-o", "/dev/null"]
    for d in includes:
        cmd[3:3] = ["-I", d]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit("resource compile failed")
    rows, name = [], None
    for ln in (p.stdout + p.stderr).splitlines():
        m = re.search(r"Function Name: (\S+)", ln)
        if m:
            name, _ = m.group(1), rows.append((m.group(1), {}))
            continue
        m = re.search(r"remark:\s+([^:]+): (.+?) \[", ln)
        if m and rows:
            rows[-1][1][m.group(1).strip()] = m.group(2).strip()
    if not rows:
        raise SystemExit("zero kernels in the -Rpass-analysis output — a broken parse, not an empty table")
    by = {C.demangle_fn(k): v for k, v in rows}
    missing = [n for n in names if n not in by]
    if missing:
        raise SystemExit(f"{missing} absent from the resource pass; it saw {sorted(by)}")
    for n in names:
        gap = [f for f in ("VGPRs", "VGPRs Spill", "LDS Size [bytes/block]") if f not in by[n]]
        assert not gap, f"{n}: field(s) {gap} did not parse"
    return by


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asm", type=Path)
    ap.add_argument("kernels", help="comma-separated kernel names, in report order")
    ap.add_argument("--resource-tu", type=Path, default=None,
                    help="compile this TU for the VGPR/LDS pass (it must contain the named kernels)")
    ap.add_argument("--include", action="append", default=[],
                    help="extra -I dir for --resource-tu (repeat it; `.` is needed for engine/ headers)")
    ap.add_argument("--expect", choices=["merged"], default=None,
                    help="assert the merged form keeps BOTH halves; non-zero exit if it does not")
    ap.add_argument("--baseline", default=None,
                    help="for --expect, the shipping form to bill against (default: kernels[0])")
    args = ap.parse_args()

    names = [n for n in args.kernels.split(",") if n]
    lines = args.asm.read_text().splitlines()
    kernels = C.split_kernels_local(lines)
    print(f"listing   {args.asm}")
    print(f"assert    kernels parsed = {len(kernels)} (must be > 0) -> {', '.join(sorted(kernels))}")
    if not kernels:
        raise SystemExit("zero kernels parsed — the listing or the splitter is broken; nothing below is a result")
    absent = [n for n in names if n not in kernels]
    if absent:
        raise SystemExit(f"named kernel(s) {absent} NOT in the listing; it holds {sorted(kernels)}")
    print(f"assert    every named kernel present: {', '.join(names)}")

    b = bill(kernels, names)
    print(f"\n== the bill, slots per token-step per wavefront ({C.WF_PER_LAUNCH} wavefronts per launch)")
    print(f"{'kernel':22}{'prologue':>10}{'loopA':>8}{'loopB':>8}{'TOTAL':>8}{'vs first':>11}")
    first = b[names[0]]["total"]
    for n in names:
        v = b[n]
        print(f"{n:22}{v['prologue']:>10}{v['loopA']:>8}{v['loopB']:>8}{v['total']:>8}"
              f"{(v['total']-first)/first:>+10.2%}")

    print("\n== 1. did the unroll survive? state-accesses per trip x trips, per walk")
    print(f"{'kernel':22}{'loopA':>16}{'loopB':>16}   (unrolled bodies/trip in brackets)")
    for n in names:
        w = b[n]["walks"]
        fa, ta = w["loopA_state_dot"]
        fb, tb = w["loopB_state_update"]
        print(f"{n:22}{f'{fa}x{ta} [{fa // 4}]':>16}{f'{fb}x{tb} [{fb // 4}]':>16}")

    print("\n== 2. did the fused prologue survive? barrier/permute fingerprint of the prologue")
    print(f"{'kernel':22}{'barrier.sync':>13}{'xlane.permute':>14}{'permute groups':>15}")
    for n in names:
        v = b[n]
        print(f"{n:22}{v['barriers']:>13}{v['permutes']:>14}{v['groups']:>15}")

    if len(names) >= 2:
        a_name, bb_name = names[0], names[-1]
        print(f"\n== 3. category diff {a_name} -> {bb_name}")
        ta, tb = b[a_name]["cats"], b[bb_name]["cats"]
        for k in sorted(set(ta) | set(tb), key=lambda k: tb.get(k, 0) - ta.get(k, 0)):
            if ta.get(k, 0) != tb.get(k, 0):
                print(f"  {k:24}{ta.get(k, 0):>8}{tb.get(k, 0):>8}{tb.get(k, 0) - ta.get(k, 0):>+8}")
        print("  (unchanged: %s)" % ", ".join(sorted(k for k in set(ta) & set(tb)
                                                    if ta[k] == tb[k])))

    if args.resource_tu:
        res = resources(args.resource_tu, args.include, names)
        print(f"\n== 4. resources (compile pass on {args.resource_tu})")
        print(f"{'kernel':22}" + "".join(f"{f:>26}" for f in RES_FIELDS))
        for n in names:
            print(f"{n:22}" + "".join(f"{res[n][f]:>26}" for f in RES_FIELDS))

    if args.expect == "merged":
        base = args.baseline or names[0]
        merged = names[-1]
        m, r = b[merged], b[base]
        okA = m["walks"]["loopB_state_update"] == r["walks"]["loopB_state_update"]
        okA = okA and m["walks"]["loopA_state_dot"] == r["walks"]["loopA_state_dot"]
        okB = m["barriers"] < r["barriers"] and m["groups"] < r["groups"]
        print("\n== VERDICT (--expect merged)")
        print(f"  unroll kept  : {okA}  ({base} loopB "
              f"{r['walks']['loopB_state_update']} vs {merged} {m['walks']['loopB_state_update']})")
        print(f"  prologue kept: {okB}  (barrier.sync {r['barriers']} -> {m['barriers']},"
              f" permute groups {r['groups']} -> {m['groups']})")
        print(f"  bill         : {r['total']} -> {m['total']} ({m['total'] - r['total']:+d} slots,"
              f" {(m['total'] - r['total']) / r['total']:+.3%})")
        if not (okA and okB):
            print("  RESULT: the merge gave a half back. That is a closure, not a build.")
            return 1
        print("  RESULT: both halves survived; the bill moved down. Continue to a paired burst.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
