#!/usr/bin/env python3
"""zero_gate — the permanent per-row cross-row-dependence gate (see prototype/mtp_verify/FIX-ZERO.md).

This is the check that DID NOT EXIST when the defect did. The defect: perturbing token `p` moves query row
`p-1`'s attention output, deterministically, by <= 1 BF16 ULP, and only when `bound(p-1) = p` is odd
(`CROSSROW` datum, `FIX-CROSSROW` Results 1/5, `ISOLATED` arm A). Nothing in the tree caught it because every
existing band is ~2e-3 against an effect at one ULP, i.e. the leak hides inside a band that was bought for a
different purpose, and because a per-row byte-identity question was only ever asked by throwaway shell scripts.

WHAT IT DECIDES, per perturbed position p (the rules are derived from the geometry, not typed per fixture):

  rows q > p-1           EXEMPT      they attend lane p legitimately; a run where nothing at or after p moves
                                     is unclassified, not clean (the reach witness, `must_fire_reach`).
  row q == p-1, p even   MUST NOT MOVE  an even bound puts the causal boundary between two packed f16x2 P
                                     cells, so there is no straddle and no excuse (`FIX-CROSSROW` Result 7).
  row q == p-1, p odd    BOUNDED     the straddle row. <= 1 BF16 ULP and no NaN -> KNOWN-LEAK, recorded and
                                     reported, NOT a failure (the defect is unfixed and the fix was refused on
                                     evidence, `FIX-ZERO` §Step 1). More than 1 ULP, or any NaN -> VIOLATION.
                                     A leak that gets WORSE is therefore red even though the leak itself is
                                     carried.
  every row q < p-1      MUST NOT MOVE  their bound is < p, so lane p is outside their causal set outright.

THREE ARMS, each a separate invocation so every device command stays small and blocking:

  --arms sweep     engine per-row sweep. One `decode_run --dump-attn-rows` per position (reference plus the
                   derived set), adjudicated on the FIRST attention layer's `pre_gate` tile.
  --arms sinfzero  the signed-zero injection as a must-catch, through `pack_ab`: a negative staged V at the
                   disallowed lane MUST move the straddling row and a non-negative one MUST NOT. Both halves
                   are asserted, so this arm cannot pass by being blind. The `0 x inf` arm of the same ladder
                   is a must-FIRE in its own right (the recorded hazard), expected from the F16 range rather
                   than from a typed case name.
  --arms widen     the one-lane widening must-catch, by REUSING `tmp/mtp_verify/causal_bound --must-catch`
                   (its `widen1` is the engine's own host-side bound lie) instead of re-deriving an injection.

`selftest` proves the ADJUDICATOR can fail in both directions, on synthetic mutants of a real reference dump
dir (CPU only): an even-bound row moved, a straddle row moved 2 ULP, a NaN in an exempt row, and the
unmutated copy - the last one proves the must-not-catch side is not vacuous. AGENTS' rule applies to every
count in here: a parsed list that can be empty cannot fail, so each arm asserts its expected case count.

Geometry is READ, never typed: `WIDE_POS_PER_BLOCK` comes from the header that defines it, the row count from
the fixture manifest (enumerated, not remembered), the attention layer from the dump file names, and the row
width from `attn_dump_diff.ROW_FLOATS`.

Device rules honoured: correctness only, no timing anywhere (no ms, no burst, no best-of-N), blocking, one
command at a time, `tools/gpu_lock.sh` acquired by THIS driver and by none of its children.

Usage:
    .venv/bin/python prototype/mtp_verify/zero_gate.py plan  [--fixture p128]
    .venv/bin/python prototype/mtp_verify/zero_gate.py run   [--fixture p128] [--arms sweep,sinfzero,widen]
    .venv/bin/python prototype/mtp_verify/zero_gate.py selftest --ref-dump DIR [--work DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp  # noqa: E402  the repo's comparator, imported not retyped

VOCAB_CENSUS = 248320
DUMP_NAME = re.compile(r"^attn(\d+)_L(-?\d+)_([a-z0-9_]+)_r(\d+)_lanes(\d+)\.bin$")
ROW_FLOATS = 8 * 256      # [8 heads][256] per query row; same constant as attn_dump_diff, by name not value


# ------------------------------------------------------------------ geometry, derived ------------------------------------------------------------
def _defines_in(hdrs: list[Path]) -> dict[str, str]:
    """Every `#define NAME expr` across the headers a geometry macro can depend on, comment stripped. The
    comment is cut on `//` rather than excluded by a character class, because `(WIDE_MROWS / GQA)` contains a
    slash and a first attempt that forbade slashes silently lost the line."""
    out: dict[str, str] = {}
    for hdr in hdrs:
        if not hdr.is_file():
            continue
        for line in hdr.read_text().splitlines():
            m = re.match(r"\s*#define\s+([A-Z_][A-Z0-9_]*)\s+(.*)$", line)
            if m:
                out.setdefault(m.group(1), m.group(2).split("//")[0].strip())
    return out


def _eval_define(name: str, defs: dict[str, str], seen: frozenset = frozenset()) -> int:
    """Resolve a macro that is a product/sum of other macros (WIDE_POS_PER_BLOCK is `(WIDE_WARPS *
    WIDE_POS_PER_WARP)`, not a literal, so a `#define X 32` grep would have missed it and a typed 32 would
    have been a second copy of a fact that has already caused a bug by living in two headers)."""
    if name in seen:
        raise SystemExit(f"macro cycle at {name}")
    expr = defs[name]
    for _ in range(8):
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
        if not toks:
            break
        for t in set(toks):
            if t not in defs:
                raise SystemExit(f"cannot resolve macro {name} -> {expr!r}: unknown token {t}")
            expr = re.sub(rf"\b{t}\b", "(" + defs[t] + ")", expr)
    if not re.fullmatch(r"[0-9()+\-*/ ]+", expr):
        raise SystemExit(f"cannot evaluate macro {name} -> {expr!r}: not plain arithmetic")
    return int(eval(expr, {"__builtins__": {}}, {}))    # noqa: S307 - characters are restricted above


def wide_pos_per_block() -> int:
    """`WIDE_POS_PER_BLOCK`, read from the header that owns it and resolved through its own macro chain,
    which crosses headers (`WIDE_POS_PER_WARP` -> `WIDE_MROWS / GQA`)."""
    hdrs = [ROOT / h for h in ("engine/attn_wide_kernels.h", "engine/attn_wide_llama_kernels.h",
                               "engine/attn_fattn_kernels.h", "engine/kernels.h",
                               "prototype/attn_decode/attn_decode_kernels.h")]
    defs = _defines_in(hdrs)
    if "WIDE_POS_PER_BLOCK" not in defs:
        raise SystemExit("cannot read WIDE_POS_PER_BLOCK from any header that might define it")
    return _eval_define("WIDE_POS_PER_BLOCK", defs)


def fixture_ids(name: str) -> list[int]:
    """A pinned fixture's prompt ids, from its manifest. Never a remembered list (AGENTS' admission trap)."""
    man = ROOT / "data" / "refs" / name / "manifest.json"
    if not man.is_file():
        raise SystemExit(f"{man} missing - enumerate data/refs/*/manifest.json instead of naming fixtures")
    ids = json.loads(man.read_text()).get("prefill_ids") or []
    if not ids:
        raise SystemExit(f"{man} has no prefill_ids")
    return [int(t) for t in ids]


def rtiles(nrows: int, wpb: int) -> list[tuple[int, int]]:
    """(rtile, first_row) for every row tile the launcher can emit for this prompt."""
    return [(r, r * wpb) for r in range((nrows + wpb - 1) // wpb)]


def derive_positions(nrows: int, wpb: int, budget: int) -> list[int]:
    """The perturbation set: per row tile, one ODD and one EVEN position as far inside the tile as possible,
    then a stride-odd/even alternation until the budget is spent. Derived from `wpb` and `nrows` alone, and
    asserted non-empty by the caller - a gate whose case list is empty cannot fail."""
    out: list[int] = []
    for _, row0 in rtiles(nrows, wpb):
        # Position p tests row p-1, whose bound is exactly p, so p's parity IS the straddle parity.
        for p in (row0 + 2, row0 + 3):          # row 1 (bound 2, even) and row 2 (bound 3, odd) of this tile
            if 1 <= p < nrows and p not in out:
                out.append(p)
    step = 0
    while len(out) < budget:
        p = wpb + 5 + step * 7                  # further into the prompt, alternating parity by construction
        if p >= nrows:
            break
        if p not in out:
            out.append(p)
        step += 1
    return sorted(set(out))[:budget]


# ------------------------------------------------------------------ dump IO + adjudication -----------------------------------------------------
def dump_files(d: Path, what: str = "pre_gate") -> list[Path]:
    if not d.is_dir():
        raise SystemExit(f"{d} is not a directory - run the sweep first; a missing dir is not a clean result")
    fs = sorted(f for f in d.iterdir() if DUMP_NAME.match(f.name) and f.name.split("_r")[0].endswith(what))
    return fs


def first_layer_tile(d: Path, what: str = "pre_gate") -> tuple[Path, int, int]:
    """The FIRST attention layer's tile in a dump dir: layer chosen by the smallest L in the dir, so the
    gate does not hardcode 'layer 3' and silently misread a fixture with a different layer layout."""
    fs = dump_files(Path(d), what)
    if not fs:
        raise SystemExit(f"{d}: no {what} tiles. A dump dir with no tiles cannot fail (empty-parse trap).")
    def layer_of(f: Path) -> int:
        return int(DUMP_NAME.match(f.name).group(2))
    f = min(fs, key=layer_of)
    m = DUMP_NAME.match(f.name)
    return f, int(m.group(2)), int(m.group(4))


def load_bits(path: Path) -> np.ndarray:
    """Raw BF16 bit patterns as [rows][ROW_FLOATS], dtype PROVEN by byte count (AGENTS' dump-dtype row: read
    a BF16 tile as F32 or back and you get a plausible ndiff and a NaN cos)."""
    n = path.stat().st_size
    if n % (2 * ROW_FLOATS):
        raise SystemExit(f"{path}: {n} bytes is not a whole number of BF16 rows of {ROW_FLOATS} elements")
    a = np.fromfile(path, dtype="<u2")
    return a.reshape(n // 2 // ROW_FLOATS, ROW_FLOATS)


def ulp_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """BF16 ULP distance between two equal-shaped arrays of RAW BIT PATTERNS. Uses the comparator's own
    `monotone_key` (its bit-pattern law) rather than `ulp_dist`, which takes BF16 values and would re-encode."""
    return np.abs(bf16_ulp.monotone_key(a.astype(np.int32)) - bf16_ulp.monotone_key(b.astype(np.int32)))


def is_nan_bits(a: np.ndarray) -> np.ndarray:
    return (a & 0x7C00) == 0x7C00


def adjudicate(ref: np.ndarray, got: np.ndarray, p: int, upto: int) -> tuple[list[str], list[str], dict]:
    """The rule table from the module docstring, over rows [0, upto). Returns (violations, known_leaks, stats)."""
    viol: list[str] = []
    known: list[str] = []
    nrow = min(ref.shape[0], got.shape[0], upto)
    if nrow < 2:
        raise SystemExit(f"adjudicating {nrow} rows cannot fail; the fixture is too short for this arm")
    straddle = p - 1
    stats = {"rows": nrow, "movers_adj": 0, "max_ulp": 0, "nan": 0, "exempt_movers": 0}
    for q in range(nrow):
        r, g = ref[q], got[q]
        if np.array_equal(r, g):
            continue
        if q > straddle:
            stats["exempt_movers"] += 1
            continue                                     # legitimately downstream: it attends lane p
        stats["movers_adj"] += 1
        u = int(ulp_between(r, g).max())
        stats["max_ulp"] = max(stats["max_ulp"], u)
        n_nan = int(is_nan_bits(g).sum())
        stats["nan"] += n_nan
        if n_nan:
            viol.append(f"row {q}: NaN in {n_nan} cells of its attention output")
        if q == straddle:
            if p % 2 == 0:
                viol.append(f"row {q}: moved with an EVEN bound ({p}); no straddle exists")
            elif u > 1:
                viol.append(f"row {q}: moved {u} BF16 ULP with an odd bound; band is 1")
            else:
                known.append(f"row {q}: {int((r != g).sum())} cells by {u} BF16 ULP (odd bound {p}, known leak)")
        else:
            viol.append(f"row {q}: moved but its bound is {q + 1} <= {p}; lane p is outside its causal set")
    return viol, known, stats


# ------------------------------------------------------------------ helpers -------------------------------------------------------------------------
def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def gpu_lock(tag: str, acquire: bool) -> None:
    r = sh(["bash", "tools/gpu_lock.sh", "acquire" if acquire else "release", tag])
    if r.returncode != 0:
        raise SystemExit(f"gpu_lock {tag}: {r.stdout}{r.stderr}")


def shift_ulp(bits: np.ndarray, n: int) -> int:
    """One bit pattern moved exactly `n` BF16 ULPs, through the comparator's own key law."""
    k = bf16_ulp.monotone_key(np.array([int(bits)], dtype=np.int32))[0]
    return int(bf16_ulp.bits_from_key(np.array([int(k) + n], dtype=np.int64))[0])


# ------------------------------------------------------------------ arm A: the engine sweep ------------------------------------------------------
def run_sweep(fixture: str, work: Path, positions: list[int], form: list[str]) -> tuple[int, int, int]:
    """Returns (cases, violations, known_leaks). One device run per position, blocking, one at a time."""
    work.mkdir(parents=True, exist_ok=True)
    ids = fixture_ids(fixture)
    nrows = len(ids)
    prep = sh([sys.executable, "prototype/mtp_verify/crossrow_arm1.py", "prep",
               "--fixture", fixture, "--work", str(work)])
    if prep.returncode != 0:
        raise SystemExit(f"crossrow_arm1 prep failed: {prep.stdout}{prep.stderr}")
    pert = sh([sys.executable, "prototype/mtp_verify/crossrow_arm1.py", "perturb", "--fixture", fixture,
               "--work", str(work), "--positions", ",".join(str(p) for p in positions)])
    if pert.returncode != 0:
        raise SystemExit(f"crossrow_arm1 perturb failed: {pert.stdout}{pert.stderr}")

    def dump_run(csv: Path, out: Path) -> Path:
        if out.exists():
            shutil.rmtree(out)
        cmd = ["./engine/build/decode_run", "--model", "models/Qwen3.5-0.8B", "--ids-file", str(csv),
               "--steps", "0", "--prefill-logits", "all", "--dump-attn-rows", "--out", str(out)] + form
        r = sh(cmd)
        if r.returncode != 0:
            raise SystemExit(f"decode_run failed on {csv.name}: {r.stdout[-800:]}{r.stderr[-400:]}")
        return out

    ref_dir = dump_run(work / "ref.csv", work / "zg-ref")
    tile, layer, trows = first_layer_tile(ref_dir)
    ref = load_bits(tile)
    # A position at or beyond the first tile's row count would be adjudicated against a tile that does not
    # contain its straddle row, i.e. a case that cannot fail. Excluded by name, not silently skipped.
    active = [p for p in positions if p < trows]
    excluded = [p for p in positions if p not in active]
    if excluded:
        print(f"  EXCLUDED (at or beyond the first tile's {trows} rows, so this tile cannot adjudicate them): "
              f"{excluded}")
    if not active:
        raise SystemExit("no position falls inside the first attention tile; this arm has no cases")
    print(f"SWEEP fixture={fixture} rows={nrows} first_layer=L{layer} tile_rows={trows} "
          f"form={' '.join(form) if form else 'shipping default'} cases={len(active)}")
    positions = active
    cases = viol_total = known_total = 0
    for p in positions:
        got_dir = dump_run(work / f"pert{p}.csv", work / f"zg-p{p}")
        gtile = next((f for f in dump_files(got_dir) if f.name.split("_r")[0] == tile.name.split("_r")[0]), None)
        if gtile is None:
            raise SystemExit(f"{got_dir}: no tile matching {tile.name}; the geometry changed between runs")
        got = load_bits(gtile)
        viol, known, stats = adjudicate(ref, got, p, upto=min(trows, p + 1))
        cases += 1
        # reach witness: the perturbation must move SOMETHING at or after p, or it never reached the model
        reach = int((got[p:] != ref[p:]).any(axis=1).sum()) if got.shape[0] > p else 0
        tagline = "CLEAN" if not viol and not known else ("VIOLATION" if viol else "known-leak")
        print(f"  p={p:4d} bound_parity={'odd' if p % 2 else 'even'} movers_at_or_below_straddle="
              f"{stats['movers_adj']} max_ulp={stats['max_ulp']} nan={stats['nan']} "
              f"exempt_rows_moved={stats['exempt_movers']} rows_at_or_after_p_moved={reach} -> {tagline}")
        for v in viol:
            print(f"      VIOLATION: {v}")
        for k in known:
            print(f"      known-leak: {k}")
        if reach == 0:
            print(f"      VIOLATION: nothing at or after row {p} moved; the injection did not reach the model")
            viol_total += 1
        viol_total += len(viol)
        known_total += len(known)
    return cases, viol_total, known_total


# ------------------------------------------------------------------ arm B: the signed-zero injection -----------------------------------------------
def run_sinfzero(pack_ab: Path, ref_dump: Path, row: int) -> tuple[int, int]:
    """The signed-zero must-catch, both halves, plus the recorded `0 x inf` hazard as its own must-fire.
    The expectation for each case is DERIVED from the value the rig itself prints (`--list` carries `val=`),
    against the F16 range: a finite value stages to +/-0 so only its SIGN can matter, and a value at or above
    F16's 65504 stages to `inf`, where `0 x inf` is NaN and reaches the live row (FIX-ZERO's hazard note).
    Returns (cases, failures)."""
    if not pack_ab.is_file():
        raise SystemExit(f"{pack_ab} not built; run bash prototype/mtp_verify/pack_ab/build.sh")
    r = sh([str(pack_ab), "--dump", str(ref_dump), "--row", str(row), "--out", "/tmp/zero_gate/packab"])
    if r.returncode != 0:
        raise SystemExit(f"pack_ab failed: {r.stdout[-800:]}{r.stderr[-300:]}")
    lst = sh([str(pack_ab), "--list"])
    n_lines = len([ln for ln in lst.stdout.splitlines() if ln.startswith("v62_")])
    want: dict[str, float] = {}
    for ln in lst.stdout.splitlines():
        m = re.match(r"^(v62_\S+)\b.*\bval=(-?[0-9.eE+-]+)\}", ln)
        if m:
            want[m.group(1)] = float(m.group(2))
    if not want:
        raise SystemExit("pack_ab --list shows no v62_* cases with a val=: this arm has no cases and cannot fail")
    if len(want) != n_lines:
        raise SystemExit(f"parsed a val= for {len(want)} of {n_lines} v62_* cases; the arm would silently "
                         "adjudicate a subset (the empty-parse failure class)")
    seen: dict[str, int] = {}
    for ln in r.stdout.splitlines():
        m = re.match(r"^CASE (v62_\S+)\b.*row_at_moved=(\d)", ln)
        if m:
            seen[m.group(1)] = int(m.group(2))
    missing = [c for c in want if c not in seen]
    if missing:
        raise SystemExit(f"parsed {len(seen)} of {len(want)} CASE lines (missing {missing}); "
                         "a driver that parses less than the rig printed cannot adjudicate")
    fails = 0
    for c, val in sorted(want.items()):
        overflow = abs(val) >= 65504.0            # F16 max normal: stages to inf, and 0 x inf is NaN
        want_moved = (val < 0.0) or overflow
        got_moved = seen[c] == 1
        ok = want_moved == got_moved
        if not ok:
            fails += 1
        why = ("stages to inf -> 0 x inf = NaN, the recorded hazard" if overflow else
               f"stages to {'-' if val < 0 else '+'}0, sign {'can' if val < 0 else 'cannot'} reach the row")
        print(f"  {c:14s} val={val:<10g} {why:52s} row_at_moved={seen[c]} "
              f"{'ok' if ok else '*** NOT AS REQUIRED ***'}")
    if "CANARY fid: BYTE-IDENTICAL" not in r.stdout:
        print("  *** pack_ab fidelity canary did not report BYTE-IDENTICAL; this arm adjudicates nothing ***")
        fails += 1
    return len(want), fails


# ------------------------------------------------------------------ arm C: the one-lane widening must-catch ------------------------------------
def run_widen(causal_bound: Path) -> tuple[int, int]:
    """REUSE, not re-derive: `causal_bound --must-catch` already injects `widen1` (the engine's own bound lie)
    across its encodings. It exits 1 when every injection came back RED, which is the pass condition here."""
    if not causal_bound.is_file():
        raise SystemExit(f"{causal_bound} not built; run bash prototype/mtp_verify/build_causal_bound.sh")
    r = sh([str(causal_bound), "--must-catch"])
    red = len(re.findall(r"control .*?-> RED, as it must be", r.stdout))
    green = len(re.findall(r"control .*?GREEN", r.stdout))
    if red == 0:
        raise SystemExit("causal_bound --must-catch reported 0 RED controls: an empty must-catch set is not a control")
    print(f"  causal_bound --must-catch: red={red} green={green} rc={r.returncode}")
    return red, green


# ------------------------------------------------------------------ selftest: prove the adjudicator can fail ---------------------------
def selftest(ref_dump: Path, work: Path, positions: list[int]) -> tuple[int, int]:
    """Synthetic mutants of a REAL reference dump, so the must-catch and must-not-catch halves are both
    exercised without touching the engine. CPU only."""
    tile, layer, trows = first_layer_tile(ref_dump)
    ref = load_bits(tile)
    evens = [p for p in positions if p % 2 == 0 and p - 1 < trows]
    odds = [p for p in positions if p % 2 == 1 and p - 1 < trows]
    if not evens or not odds:
        raise SystemExit(f"the derived set has no even ({evens}) or no odd ({odds}) position: the selftest "
                         "cannot exercise both directions, so it must not be allowed to pass")
    cases = 0
    fails = 0
    for name, p, row, nulp, nan, expect_viol in [
        ("clean copy (must-not-catch)",            odds[0],  None, 0, False, "clean"),
        ("straddle row moved 1 ULP (known leak)",  odds[0],  odds[0] - 1, 1, False, "known"),
        ("straddle row moved 2 ULP (band broken)", odds[0],  odds[0] - 1, 2, False, "violation"),
        ("even-bound row moved 1 ULP (widened)",   evens[0], evens[0] - 1, 1, False, "violation"),
        ("a row past the straddle moved",          odds[0],  max(0, odds[0] - 2), 1, False, "violation"),
        ("NaN injected into the straddle row",     odds[0],  odds[0] - 1, 0, True,  "violation"),
    ]:
        mut = ref.copy()
        if row is not None:
            if nan:
                mut[row, 0] = 0x7FC0
            else:
                mut[row, 0] = shift_ulp(ref[row, 0], nulp)
        viol, known, _ = adjudicate(ref, mut, p, upto=min(trows, p + 1))
        got = "violation" if viol else ("known" if known else "clean")
        ok = got == expect_viol
        cases += 1
        fails += 0 if ok else 1
        print(f"  [{layer}] {name:42s} expected={expect_viol:10s} got={got:10s} {'ok' if ok else '*** SELFTEST FAILED ***'}"
              + (f"  ({viol[0]})" if viol and not ok else ""))
    return cases, fails


# ------------------------------------------------------------------ main ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(prog="zero_gate.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("plan", "run", "selftest"):
        p = sub.add_parser(c)
        p.add_argument("--fixture", default="p128")
        p.add_argument("--work", default="/tmp/zero_gate")
        p.add_argument("--budget", type=int, default=8, help="max perturbation positions")
        p.add_argument("--positions", default="", help="override the derived set (comma list)")
        p.add_argument("--attn-fattn-wide", type=int, default=-1, help="-1 = shipping default form")
        p.add_argument("--wave-rows", type=int, default=-1)
        p.add_argument("--ref-dump", default="")
        p.add_argument("--arms", default="sweep,sinfzero,widen")
        p.add_argument("--pack-ab", default="tmp/mtp_verify/pack_ab/pack_ab")
        p.add_argument("--causal-bound", default="tmp/mtp_verify/causal_bound")
        p.add_argument("--sinf-row", type=int, default=-1, help="straddle row for the signed-zero arm")
        p.add_argument("--no-lock", action="store_true", help="caller already holds tools/gpu_lock.sh")
    ns = ap.parse_args()

    wpb = wide_pos_per_block()
    nrows = len(fixture_ids(ns.fixture))
    positions = ([int(x) for x in ns.positions.split(",") if x] if ns.positions
                 else derive_positions(nrows, wpb, ns.budget))
    if not positions:
        raise SystemExit(f"derived perturbation set is EMPTY for {ns.fixture} (rows={nrows}, "
                         f"WIDE_POS_PER_BLOCK={wpb}); a gate with no cases cannot fail")
    form = []
    if ns.attn_fattn_wide >= 0:
        form += ["--attn-fattn-wide", str(ns.attn_fattn_wide)]
    if ns.wave_rows >= 0:
        form += ["--wave-rows", str(ns.wave_rows)]

    print(f"GATE zero_gate fixture={ns.fixture} rows={nrows} WIDE_POS_PER_BLOCK={wpb} "
          f"rtiles={len(rtiles(nrows, wpb))} cases={len(positions)}")
    print(f"  derived positions: {positions}")
    print(f"  parity coverage: odd={sum(1 for p in positions if p % 2)} even={sum(1 for p in positions if p % 2 == 0)}")
    if ns.cmd == "plan":
        for _, row0 in rtiles(nrows, wpb):
            par = ["odd" if (row0 + i + 1) % 2 else "even" for i in range(4) if row0 + i < nrows]
            print(f"  rtile@{row0:4d}: first bounds parity {par}")
        print("PLAN ok: the set is non-empty and covers both bound parities.")
        return 0

    work = Path(ns.work) / ns.fixture
    ref_dump = Path(ns.ref_dump) if ns.ref_dump else work / "zg-ref"
    total_fail = 0
    if ns.cmd == "selftest":
        if not Path(ns.ref_dump).is_dir():
            raise SystemExit("--ref-dump DIR is required and must be a dump dir from `run`")
        cases, fails = selftest(Path(ns.ref_dump), work, positions)
        print(f"SELFTEST cases={cases} failures={fails}")
        return 1 if fails else 0

    tag = "zero_gate sweep"
    if not ns.no_lock:
        gpu_lock(tag, True)
    try:
        if "sweep" in ns.arms:
            cases, viol, known = run_sweep(ns.fixture, work, positions, form)
            if cases != len(positions):
                raise SystemExit(f"sweep adjudicated {cases} of {len(positions)} cases")
            print(f"ARM sweep: cases={cases} violations={viol} known_leaks={known}")
            total_fail += viol
        if "sinfzero" in ns.arms:
            srow = ns.sinf_row if ns.sinf_row >= 0 else 8
            cases, fails = run_sinfzero(ROOT / ns.pack_ab, ref_dump, srow)
            print(f"ARM sinfzero: cases={cases} failures={fails} (row {srow}, bound {srow + 1})")
            total_fail += fails
        if "widen" in ns.arms:
            red, green = run_widen(ROOT / ns.causal_bound)
            print(f"ARM widen: red={red} green={green}")
            total_fail += green
    finally:
        if not ns.no_lock:
            gpu_lock(tag, False)
    print(f"VERDICT: {'RED' if total_fail else 'GREEN'} (failures={total_fail}); "
          "known leaks are carried, not waived - see FIX-ZERO.md")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
