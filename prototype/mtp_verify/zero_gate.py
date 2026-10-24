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
  --arms sinfzero  the signed-zero injection as a must-catch, through `pack_ab`, over a DERIVED population of
                   (attention layer, straddle row) points, adjudicated on the rig's own output FILES rather
                   than on the counter it prints. What is hard at every point: a non-negative staged V must
                   not move the straddle row or any row behind it, a negative one may move it by <= 1 BF16 ULP
                   with no NaN, all negatives at a point must agree with each other (the dead product is
                   0 x V = +-0, so a response that depends on |V| means the dead weight is no longer exactly
                   +0), and the `0 x inf` arm MUST fire (the recorded hazard). What is NOT hard anywhere: that
                   the sign nudge is VISIBLE at all - it is a 1-F16-ULP nudge of the F16 accumulator and shows
                   in the BF16 output only where a cell sits near a BF16 rounding boundary, which is a
                   property of the fixture's values, not of its geometry. So the arm SEARCHES its derived
                   population for at least one point where the sign channel is visible and, if the budget
                   holds none, it REFUSES loudly. A silent pass on an uninhabited question is the failure mode
                   this arm had: see FIX-ZERO §"Correction: the signed-zero arm was asserting one point's luck".
  --arms widen     the one-lane widening must-catch, by REUSING `tmp/mtp_verify/causal_bound --must-catch`
                   (its `widen1` is the engine's own host-side bound lie) instead of re-deriving an injection.

`selftest` proves the ADJUDICATOR can fail in both directions, on synthetic mutants of a real reference dump
dir (CPU only): an even-bound row moved, a straddle row moved 2 ULP, a NaN in an exempt row, and the
unmutated copy - the last one proves the must-not-catch side is not vacuous - plus eight mutants of the
signed-zero arm's own case tiles (a sensitive point that must read adjudicable, a straddle move of 2 ULP, a
non-negative that moves, an `inf` that does not fire, a row behind the straddle that moves, a value-dependent
negative, a rig counter that disagrees with its own bytes, and the blind pattern, which must read as NOT
adjudicable rather than as clean). AGENTS' rule applies to
every count in here: a parsed list that can be empty cannot fail, so each arm asserts its expected case count.

Geometry is READ, never typed: `WIDE_POS_PER_BLOCK` comes from the header that defines it, the row count from
the fixture manifest (enumerated, not remembered), the attention layer from the dump file names, and the row
width from `attn_dump_diff.ROW_FLOATS`.

Device rules honoured: correctness only, no timing anywhere (no ms, no burst, no best-of-N), blocking, one
command at a time, `tools/gpu_lock.sh` acquired by THIS driver and by none of its children.

Usage:
    .venv/bin/python prototype/mtp_verify/zero_gate.py plan  [--fixture p128]
    .venv/bin/python prototype/mtp_verify/zero_gate.py run   [--fixture p128] [--arms sweep,sinfzero,widen]
    .venv/bin/python prototype/mtp_verify/zero_gate.py run   --fixture dec8 --arms sinfzero --sinf-rows 4
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
    if not path.is_file():
        raise SystemExit(f"{path}: no such file. A missing tile is an unadjudicated case, not a clean one, so "
                         "it aborts rather than reading as a pass.")
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
        cmd = ["./engine/build/decode_run", "--mtp-accept", "0", "--mtp-draft", "0", "--model", "models/Qwen3.5-0.8B", "--ids-file", str(csv),
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
# This arm used to assert ONE fact at ONE hard-coded point - the smallest-L tile of the dump, row 8 - namely
# that a negative staged V at the disallowed lane moves the straddling row. `--fixture dec8` read 10 failures
# and the bytes say why, and the rig's counter was NOT the liar:
#
#   dec8 (engine echo `core=attn_wide_llama_wave rows=17 lanes_last=17 lvalid=1 cap=19`, rig canary
#   BYTE-IDENTICAL, so the wide F16-D form really does run at 17 rows): at L3, every one of rows 2,4,6,8,10,
#   12,14 leaves its straddle row BYTE-IDENTICAL under all ten negative cases - 0 of 2 048 cells move, which
#   is what the rig's counter printed and what `bf16_ulp` confirms on the files. At L23 row 8 the same fixture
#   moves exactly 1 cell, `h5/d62 3fe3 -> 3fe2`, 1 BF16 ULP, all ten negatives together.
#   p128: L3 row 8 moves 1 cell (`h4/d62 3ecd -> 3ecc`); L3 rows 2,4,6,10,20,40,60,80,98,100,110 do not.
#   wide640f body 1 (512 rows): L3 rows 2,4,6,8 do not; L7 row 8 moves 1 cell (`h5/d62 3e4c -> 3e4b`).
#   The counter was compared against the files on 80 case-reads across five of those points: 0 disagreements.
#
# So "row 8 moves" is neither geometry nor a law: it is whether one cell of that row's 2 048 sits within one
# F16 ULP of a BF16 rounding boundary, because that is the only way the nudge that `CAUSE.md` F3 names (the
# F16-D WMMA atom is not neutral to the sign of a `0 x V` product) survives the BF16 epilogue. The leak is
# present at dec8 whether or not it shows - `v62_p1e6` puts NaN in 8 cells of row 8 there, which proves the
# dead k-term is formed and accumulated at that row. An expectation imported from another fixture's row 8 was
# therefore a check about the wrong thing. What this arm now asserts:
#
#   HARD, at every derived point (fixture-independent):
#     canary                                  the rig reproduces the engine's own tile, else it adjudicates
#                                             nothing (pack_ab aborts; re-checked here by name);
#     0 <= val < 65504                        straddle row and every row behind it BYTE-IDENTICAL;
#     val < 0, finite                         straddle row may move, by <= 1 BF16 ULP and NaN-free, and no
#                                             row behind it may move. Move recorded as known-leak;
#     all negatives at a point                must AGREE (product is 0 x V = +-0 for every finite V, so a
#                                             response that depends on |V| means the dead weight stopped
#                                             being exactly +0 - the thing most worth catching here);
#     |val| >= 65504                          MUST move the straddle row. 0 x inf = NaN is IEEE-forced, and
#                                             movement of rows behind it is the recorded hazard, printed and
#                                             not charged;
#     rig counter vs the rig's own files      must agree, per case. `ISOLATED.md` records one build where
#                                             `row_at_moved` disagreed with its files, so the FILES adjudicate
#                                             and a disagreement is a failure of the rig, loudly.
#   SEARCHED, at the fixture as a whole:
#     at least one derived point where the sign channel is VISIBLE. If the derived population holds none
#     within `--sinf-probes`, the arm REFUSES (non-zero exit). That is not a pass and not a violation: the
#     fixture cannot adjudicate that half, and saying so is the whole point.
F16_MAX_NORMAL = 65504.0     # the largest finite F16: at or above it the staged V is inf and 0 x inf is NaN


def sinf_kind(val: float) -> str:
    """The three expectations, derived from the VALUE the rig prints for a case, never from its name."""
    if abs(val) >= F16_MAX_NORMAL:
        return "overflow"          # stages to inf: 0 x inf = NaN, must reach the live row
    return "negative" if val < 0.0 else "nonnegative"


def sinf_case_table(pack_ab: Path) -> dict[str, float]:
    """case -> value, read from `pack_ab --list` (that branch returns before any HIP call, so it is device
    free). The arm and the selftest share this one source so the two cannot drift apart."""
    if not pack_ab.is_file():
        raise SystemExit(f"{pack_ab} not built; run bash prototype/mtp_verify/pack_ab/build.sh")
    lst = sh([str(pack_ab), "--list"])
    lines = [ln for ln in lst.stdout.splitlines() if ln.startswith("v62_")]
    want: dict[str, float] = {}
    for ln in lines:
        m = re.match(r"^(v62_\S+)\b.*\bval=(-?[0-9.eE+-]+)\}", ln)
        if m:
            want[m.group(1)] = float(m.group(2))
    if not want:
        raise SystemExit("pack_ab --list shows no v62_* cases with a val=: this arm has no cases and cannot fail")
    if len(want) != len(lines):
        raise SystemExit(f"parsed a val= for {len(want)} of {len(lines)} v62_* cases; the arm would silently "
                         "adjudicate a subset (the empty-parse failure class)")
    return want


def row_diff(ref_row: np.ndarray, got_row: np.ndarray) -> dict:
    """One row's byte difference as {moved, cells, max_ulp, nan}. ULP is measured only where NEITHER side is
    NaN, because a NaN's bit pattern is 48 700-odd ULPs from any number and a band checked over it passes or
    fails on the encoding rather than on the arithmetic."""
    d = ref_row != got_row
    cells = int(d.sum())
    if cells == 0:
        return {"moved": False, "cells": 0, "max_ulp": 0, "nan": 0}
    both_real = ~(is_nan_bits(ref_row) | is_nan_bits(got_row))
    sel = d & both_real
    mx = int(ulp_between(ref_row[sel], got_row[sel]).max()) if int(sel.sum()) else 0
    return {"moved": True, "cells": cells, "max_ulp": mx, "nan": int(is_nan_bits(got_row).sum())}


def adjudicate_sinfzero(ref: np.ndarray, tiles: dict[str, np.ndarray], counter: dict[str, tuple[int, int, int]],
                        want: dict[str, float], row: int) -> tuple[list[str], list[str], dict]:
    """The rule table above, at one point, over the rig's FILES. `counter` is the rig's own printed
    (at, below, above) triple per case, kept only to be contradicted. Returns (violations, notes, stats) with
    stats['sensitive'] non-zero iff a negative case visibly moved the straddle row here."""
    viol: list[str] = []
    notes: list[str] = []
    if ref.shape[0] < row + 2:
        raise SystemExit(f"cannot adjudicate row {row} of a {ref.shape[0]}-row tile: there is no row after "
                         "it to witness that the injection reached the kernel")
    stats = {"cases": 0, "sensitive": 0, "cells": 0, "max_ulp": 0, "nan": 0}
    neg_move: dict[str, bool] = {}
    for case, val in sorted(want.items()):
        if case not in tiles:
            viol.append(f"{case}: the rig printed a CASE line but no tile {case}; an unadjudicated case "
                        "cannot be counted as clean")
            continue
        stats["cases"] += 1
        kind = sinf_kind(val)
        at = row_diff(ref[row], tiles[case][row])
        below = sum(1 for q in range(row) if not np.array_equal(ref[q], tiles[case][q]))
        above = sum(1 for q in range(row + 1, min(ref.shape[0], tiles[case].shape[0]))
                    if not np.array_equal(ref[q], tiles[case][q]))
        c = counter.get(case)
        if c is None:
            viol.append(f"{case}: no CASE line in the rig's output; a driver that parses less than the rig "
                        "printed cannot adjudicate")
        elif c != (int(at["moved"]), below, above):
            viol.append(f"{case}: the rig's counter says at={c[0]} below={c[1]} above={c[2]} but its own "
                        f"files say at={int(at['moved'])} below={below} above={above}; the files adjudicate "
                        "and the counter is the suspect (ISOLATED.md's loose end)")
        if kind == "nonnegative":
            if at["moved"]:
                viol.append(f"{case}: val={val:g} is non-negative, the dead product is exactly +0, yet the "
                            f"straddle row moved {at['cells']} cells by {at['max_ulp']} ULP")
            if below:
                viol.append(f"{case}: {below} row(s) BEHIND the straddle row moved on a non-negative value")
        elif kind == "negative":
            neg_move[case] = at["moved"]
            if below:
                viol.append(f"{case}: {below} row(s) behind the straddle row moved; their bounds are smaller, "
                            "so this lane is outside their causal set")
            if at["moved"]:
                stats["sensitive"] += 1
                stats["cells"] = max(stats["cells"], at["cells"])
                stats["max_ulp"] = max(stats["max_ulp"], at["max_ulp"])
                if at["nan"]:
                    viol.append(f"{case}: NaN in {at['nan']} cells of the straddle row; the signed-zero band "
                                "is 1 ULP with no NaN")
                elif at["max_ulp"] > 1:
                    viol.append(f"{case}: straddle row moved {at['max_ulp']} BF16 ULP; the carried band is 1, "
                                "so the leak got worse and a carried leak stops being carried")
        else:  # overflow: the recorded `0 x inf` hazard, and a must-fire in its own right
            if not at["moved"]:
                viol.append(f"{case}: val={val:g} stages to inf, 0 x inf is NaN by IEEE, and nothing in the "
                            "straddle row moved; the dead k-term is no longer being formed at all")
            if at["nan"]:
                notes.append(f"hazard: {at['nan']} NaN cells in the straddle row ({case})")
            if below:
                notes.append(f"hazard: {below} row(s) behind it moved too ({case}); the staged V is shared by "
                             "every row of the block, so a NaN in it is not contained to one row")
            stats["nan"] += at["nan"]
    uniq = set(neg_move.values())
    if len(uniq) > 1:
        viol.append(f"the negative cases disagree about moving the straddle row "
                    f"({sorted(neg_move)}): the dead product is 0 x V = +-0 for every finite V, so a "
                    "response that depends on |V| means the masked weight stopped being exactly +0")
    return viol, notes, stats


def sinf_points(dump: Path, lvalid: int, n_rows: int, n_layers: int) -> list[dict]:
    """The derived probe population for this arm: one point per (attention layer of the dump's FIRST body,
    straddle row). Everything in it is read or derived, nothing is typed -
      * the layers and the `seq` index come from the `attnNNN_L<M>_pre_gate_r<R>_lanes<L>.bin` names, and only
        the FIRST body's occurrence of each layer is taken, which is the same tile the sweep arm adjudicates;
      * a straddle row needs an ODD bound (`bound = lvalid + row`), because only then does the causal boundary
        fall inside a packed f16x2 P cell (`FIX-CROSSROW` Result 7); an even bound has nothing to adjudicate;
      * `row >= 1` because the rig's own must-fire control is `rows_below > 0` and row 0 has no row below it,
        which is why running it used to exit 1 with three controls that structurally could not fire;
      * `row + 1 < rows` because the injected lane must have a legitimate live consumer, else the injection is
        inert and the point is vacuous (dec8 row 16, whose dead lane is the last cached lane, moves 0 bytes);
      * `bound <= lanes` because the rig cannot vary a lane the dump does not carry.
    `lvalid` is the value the body's own ATTN_DUMP line prints for a `--steps 0` prefill (FIX-CROSSROW Result
    0); a wrong one is falsified by the rig's fidelity canary, which cannot reproduce the tile."""
    by_layer: dict[int, dict] = {}
    for f in dump_files(dump):
        m = DUMP_NAME.match(f.name)
        if not m:
            continue
        seq, layer, rows, lanes = int(m.group(1)), int(m.group(2)), int(m.group(4)), int(m.group(5))
        if layer not in by_layer:            # first occurrence of the layer = the first body
            by_layer[layer] = {"seq": seq, "layer": layer, "tile": f, "rows": rows, "lanes": lanes}
    if not by_layer:
        raise SystemExit(f"{dump}: no attention tiles to probe; this arm has no points and cannot fail")
    layers = sorted(by_layer)[:n_layers] if n_layers > 0 else sorted(by_layer)
    points: list[dict] = []
    for layer in layers:
        info = by_layer[layer]
        rows = [q for q in range(1, info["rows"] - 1)
                if (lvalid + q) % 2 == 1 and lvalid + q <= info["lanes"]]
        for q in rows[:n_rows]:
            points.append({**info, "row": q, "bound": lvalid + q})
    if not points:
        raise SystemExit(f"{dump}: no straddle row (odd bound, a row before it and one after it) in a "
                         f"{info['rows']}-row body at lvalid={lvalid}; the arm has no cases")
    return points


def run_sinfzero(pack_ab: Path, ref_dump: Path, points: list[dict], out_root: Path,
                 budget: int) -> tuple[int, int]:
    """Device part: probe the derived points, in layer-major order, until one is sensitive; adjudicate every
    one that was probed. Returns (cases adjudicated, failures). REFUSES (SystemExit) if none is sensitive."""
    want = sinf_case_table(pack_ab)
    probes = points[:budget]
    if len(probes) < len(points):
        print(f"  note: {len(points)} points derived, budget --sinf-probes={budget} probes the first "
              f"{len(probes)}; raise it if this fixture's sensitive point is further along")
    total_cases = fails = 0
    probed = sensitive_points = hazard_nan = 0
    for pt in probes:
        tile, layer, seq, row = pt["tile"], pt["layer"], pt["seq"], pt["row"]
        r = sh([str(pack_ab), "--dump", str(ref_dump), "--seq", str(seq), "--layer", str(layer),
                "--row", str(row), "--out", str(out_root / f"s{seq:03d}_L{layer}_r{row}")])
        if r.returncode != 0:
            raise SystemExit(f"pack_ab --seq {seq} --layer {layer} --row {row} failed rc={r.returncode}: "
                             f"{r.stdout[-600:]}{r.stderr[-300:]}")
        if "CANARY fid: BYTE-IDENTICAL" not in r.stdout:
            raise SystemExit(f"pack_ab's fidelity canary did not report BYTE-IDENTICAL at seq={seq} "
                             f"layer={layer}: the rig does not reproduce the engine's tile, so nothing this "
                             "arm prints about that point means anything")
        counter: dict[str, tuple[int, int, int]] = {}
        for ln in r.stdout.splitlines():
            m = re.match(r"^CASE (v62_\S+)\b.*rows_below=(\d+) first_below=(-?\d+) row_at_moved=(\d) "
                         r"rows_above=(\d+)", ln)
            if m:
                counter[m.group(1)] = (int(m.group(4)), int(m.group(2)), int(m.group(5)))
        missing = [c for c in want if c not in counter]
        if missing:
            raise SystemExit(f"parsed {len(counter)} of {len(want)} CASE lines at seq={seq} layer={layer} "
                             f"row={row} (missing {missing}); a driver that parses less than the rig printed "
                             "cannot adjudicate")
        ref = load_bits(tile)
        tiles = {c: load_bits(out_root / f"s{seq:03d}_L{layer}_r{row}" / f"{c}_L{layer}.bin")
                 for c in want}
        viol, notes, stats = adjudicate_sinfzero(ref, tiles, counter, want, row)
        probed += 1
        total_cases += stats["cases"]
        hazard_nan += stats["nan"]
        fails += len(viol)
        n_neg = sum(1 for v in want.values() if sinf_kind(v) == "negative")
        head = (f"  point seq={seq} layer=L{layer} row={row} bound={pt['bound']} (odd, straddling) "
                f"cases={stats['cases']} negatives_visible={stats['sensitive']}/{n_neg} "
                f"max_ulp={stats['max_ulp']} -> ")
        if stats["sensitive"]:
            sensitive_points += 1
            print(head + f"SENSITIVE, adjudicating the sign half here ({stats['cells']} cell(s) at "
                         f"{stats['max_ulp']} BF16 ULP; per-case NaN counts below)")
        else:
            print(head + "no BF16-boundary crossing here, so the sign half is not adjudicable AT THIS POINT "
                         "(see the arm's note: this is the fixture's values, not its geometry)")
        for c, val in sorted(want.items()):
            kind = sinf_kind(val)
            at = row_diff(ref[row], tiles[c][row])
            bad = any(v.startswith(f"{c}:") for v in viol)
            print(f"      {c:12s} val={val:<10g} {kind:12s} straddle_row_moved={int(at['moved'])} "
                  f"cells={at['cells']} max_ulp={at['max_ulp']} nan={at['nan']} "
                  f"{'*** NOT AS REQUIRED ***' if bad else 'ok'}")
        for v in viol:
            print(f"      VIOLATION: {v}")
        for n in notes:
            print(f"      recorded: {n}")
        if sensitive_points and not viol:
            break          # one adjudicable point is what the sign half needs; the hard half ran on all
    if sensitive_points == 0:
        raise SystemExit(
            f"REFUSED: the signed-zero arm found NO adjudicable point in {probed} probed "
            f"(layer, row) point(s) of {ref_dump}: no negative case moved a straddle row anywhere, so this "
            "fixture cannot adjudicate the sign channel and a green here would be a green about nothing. "
            "This is not a violation of the engine - the leak is present whether or not it shows (the "
            "`0 x inf` case fires). Raise --sinf-rows / --sinf-probes to search more of the derived "
            "population, or run this arm at a fixture recorded as adjudicable in FIX-ZERO.md.")
    print(f"  points_probed={probed} sensitive={sensitive_points} cases={total_cases} "
          f"hazard_nan_cells={hazard_nan}")
    return total_cases, fails


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


def selftest_sinfzero(ref_dump: Path, pack_ab: Path) -> tuple[int, int]:
    """Mutants of the SIGNED-ZERO arm's own case tiles, built from a real reference tile (CPU only, no device
    run). Every mutant goes through `adjudicate_sinfzero`, i.e. through the same file-shaped data the arm uses,
    so what is being proved is that the arm can fail - not that a hand-written list agrees with itself. The
    counter side is derived from the tiles for every case except the one whose whole point is a lying counter.
    """
    want = sinf_case_table(pack_ab)
    tile, layer, trows = first_layer_tile(ref_dump)
    ref = load_bits(tile)
    row = next((q for q in range(1, trows - 1) if (1 + q) % 2 == 1), 0)
    if row == 0:
        raise SystemExit(f"a {trows}-row dump has no straddle row to mutate; the selftest would adjudicate "
                         "nothing and must not be allowed to pass")
    negs = [c for c, v in want.items() if sinf_kind(v) == "negative"]
    nonnegs = [c for c, v in want.items() if sinf_kind(v) == "nonnegative"]
    overs = [c for c, v in want.items() if sinf_kind(v) == "overflow"]
    if not (negs and nonnegs and overs):
        raise SystemExit(f"the rig's ladder has no {negs and 'negative'} / {nonnegs and 'non-negative'} / "
                         f"{overs and 'overflow'} case to mutate; it cannot exercise both directions")

    def build(mut: dict[str, str], moved: list[str] | None = None) -> tuple[dict, dict]:
        """tiles = copies of ref with per-case mutations; `moved` overrides WHICH cases the derived counter
        claims moved, so a counter can be made to lie."""
        tiles: dict[str, np.ndarray] = {}
        for c in want:
            a = ref.copy()
            how = mut.get(c)
            if how == "ulp1":
                a[row, 0] = shift_ulp(ref[row, 0], 1)
            elif how == "ulp2":
                a[row, 0] = shift_ulp(ref[row, 0], 2)
            elif how == "nan":
                a[row, 0] = 0x7FC0
            elif how == "behind":
                a[row - 1, 0] = shift_ulp(ref[row - 1, 0], 1)
            tiles[c] = a
        counter = {}
        for c, a in tiles.items():
            at = 1 if (moved is None and mut.get(c) in ("ulp1", "ulp2", "nan")) or \
                     (moved is not None and c in moved) else 0
            counter[c] = (at, 1 if mut.get(c) == "behind" else 0, 0)
        return tiles, counter

    def verdict(tiles: dict, counter: dict) -> tuple[str, list[str]]:
        viol, _, stats = adjudicate_sinfzero(ref, tiles, counter, want, row)
        if viol:
            return "violation", viol
        return ("sensitive" if stats["sensitive"] else "blind"), []

    cases = fails = 0
    mutations = [
        # the pattern a sensitive point really shows: every negative moves the straddle row 1 ULP, the
        # non-negatives do nothing, the overflow case puts a NaN there. Must read adjudicable, not red.
        ("sensitive point (all negatives move 1 ULP)", {c: "ulp1" for c in negs} | {c: "nan" for c in overs},
         None, "sensitive"),
        # the dec8/L3 pattern: no negative moves anywhere, though the overflow case still fires. That is NOT
        # clean - it is a point the sign half cannot adjudicate, and reading it as clean is exactly the defect
        # this fix removes.
        ("blind point (no negative moves)",        {c: "nan" for c in overs}, None, "blind"),
        ("a negative moves 2 ULP (band broken)",   {c: "ulp2" for c in negs} | {c: "nan" for c in overs},
         None, "violation"),
        ("a non-negative moves 1 ULP",             {c: "ulp1" for c in negs + nonnegs[:1]} | {c: "nan" for c in overs},
         None, "violation"),
        ("the 0 x inf case does not fire",          {c: "ulp1" for c in negs}, None, "violation"),
        ("a negative moves a row behind it",       {c: "behind" for c in negs} | {c: "nan" for c in overs},
         None, "violation"),
        ("only the biggest negatives move",        {negs[-1]: "ulp1", negs[-2]: "ulp1"} | {c: "nan" for c in overs},
         None, "violation"),
        # the rig prints a counter as well as files; ISOLATED.md records one build where they disagreed, so a
        # disagreement must be a failure of the rig rather than a silent adoption of whichever the code read.
        ("counter claims a move the bytes lack",    {c: "ulp1" for c in negs} | {c: "nan" for c in overs},
         [], "violation"),
    ]
    for name, mut, moved, expect in mutations:
        tiles, counter = build(mut, moved)
        got, viol = verdict(tiles, counter)
        ok = got == expect
        cases += 1
        fails += 0 if ok else 1
        print(f"  [{layer}] sinf: {name:40s} expected={expect:10s} got={got:10s} "
              f"{'ok' if ok else '*** SELFTEST FAILED ***'}"
              + (f"  ({viol[0]})" if viol and not ok else ""))
    if fails == 0 and not any(e == "violation" for _, _, _, e in mutations):
        raise SystemExit("the signed-zero selftest asserts no violation, so it cannot fail")
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
        p.add_argument("--sinf-rows", type=int, default=4,
                       help="signed-zero arm: how many derived straddle rows per attention layer")
        p.add_argument("--sinf-layers", type=int, default=0,
                       help="signed-zero arm: how many attention layers of the first body (0 = all)")
        p.add_argument("--sinf-probes", type=int, default=24,
                       help="signed-zero arm: cap on pack_ab invocations per run (search budget)")
        p.add_argument("--sinf-lvalid", type=int, default=1,
                       help="lvalid the body's own ATTN_DUMP line printed (1 for a --steps 0 prefill)")
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
        scases, sfails = selftest_sinfzero(Path(ns.ref_dump), ROOT / ns.pack_ab)
        print(f"SELFTEST cases={cases + scases} failures={fails + sfails} "
              f"(sweep adjudicator {cases}/{fails}, signed-zero adjudicator {scases}/{sfails})")
        return 1 if (fails + sfails) else 0

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
            if not ref_dump.is_dir():
                raise SystemExit(f"{ref_dump} is not a dump dir; run the sweep arm first, or pass --ref-dump "
                                 "(a signed-zero arm with no reference tile cannot fail)")
            points = sinf_points(ref_dump, ns.sinf_lvalid, ns.sinf_rows, ns.sinf_layers)
            print(f"  sinfzero points: {len(points)} derived "
                  f"({len({p['layer'] for p in points})} layer(s) x up to {ns.sinf_rows} straddle row(s), "
                  f"lvalid={ns.sinf_lvalid}); first={points[0]['layer']}/r{points[0]['row']} "
                  f"last={points[-1]['layer']}/r{points[-1]['row']}")
            cases, fails = run_sinfzero(ROOT / ns.pack_ab, ref_dump, points,
                                        work / "packab", ns.sinf_probes)
            print(f"ARM sinfzero: cases={cases} failures={fails}")
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
