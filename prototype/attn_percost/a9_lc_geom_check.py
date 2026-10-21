#!/usr/bin/env python3
"""a9_lc_geom_check.py — the CPU witness for bet A9-L-c (`--attn-split-cap`): what a capped split may and may not do.

NO DEVICE, NO TIMING. Three arithmetic claims, one source-order claim and one band check — the five things
the bet cannot be run without, each answered from the engine's own expressions, plus the injections that prove
they can fail.

  [A] DOES A CAP LOSE LANES?  The central risk, stated as arithmetic. `split` is not a work knob, it is the
      number of contiguous slices a row is cut into, and every core computes the slice the same way —
      `span = ceil(nrow / s_row)`, block y owns `[y*span, min(nrow, y*span+span))`
      (`engine/attn_fattn_kernels.h:473-479`, `prototype/attn_decode/attn_decode_kernels.h:288-290`,
      `engine/attn_slim_kernels.h:168-171`). Requiring those slices to tile `[0, nrow)` exactly — no gap, no
      overlap, nothing past the row — for every body of every width and every cap is what "the round loop just
      runs longer, it does not drop lanes" has to mean before anyone spends a 32k window on it. A cap of 8 at
      32,768 lanes is `ceil(32768/8) = 4096` lanes per block and 256 rounds per block, the row
      `A9-LONGCTX-CENSUS.md` §4.2's table prints; this is the witness that the row is arithmetic and not hope
      (AGENTS: "a device fault on a wave-only path presents as a hang, not a crash").
      The case population is DERIVED, not hand-listed: it includes non-power-of-two caps and the remainder body a
      real prompt ends with, because a case list on which the injected defect cannot fire is not a control
      (the `partalias` lesson, applied to a check written ten minutes ago — the first version of this file
      listed only 512-multiple widths and power-of-two caps, where floor and ceil division agree exactly, and
      its must-catch scored 0 of 0).

  [B] CAN A CAP OVERRUN THE SCRATCH?  `attn_part_floats(rows)` sizes `part` from `kAttnMaxSplit`
      (`engine/kernels_attn.cpp:241`), not from this run's split, and the slot expression strides by
      `a.split`, so a shallower split can only leave the tail of a row's slab unwritten. The must-catch is the
      mismatch `tools/attn_ceiling_proof.py` [B3] injects for the same reason: stride the slot by the ceiling
      while the allocation follows the run's split, and the two caps differ.

  [C] WHERE IS A BINDING CAP LEGAL AT ALL?  `span > MAXSC` is where the two VALU tiled spellings overrun
      their `s_sc[ATTN_TILE_G][MAXSC]` score staging (`engine/attn_slim_kernels.h:157`, written at `:232`);
      `span <= MAXSC` is where they do not. The census of which (cap, width) pairs cross the line is the
      domain of the launcher's refusal, printed so the refusal's bound is a derived fact rather than a
      sentence in an error message.

  [D] THE CEILING IS PRICED ON THE UNCAPPED SPLIT.  A plan whose `split_for` exceeds `kAttnMaxSplit` must be
      refused whatever the cap says, because `engine/gate.py::lane_ceiling()` reads `MAXSC × kAttnMaxSplit`
      out of the enforcing files and asserts every plan fits it. That guarantee is not arithmetic — it is the
      ORDER of two statements in `attn_decode_layer`: the ceiling throw must read `split`, and it must come
      before the cap is applied. So [D] reads the source and checks the order, and its injection moves the
      comparison onto `split_eff`, which is the edit that would silently widen the ceiling the gate believes.

  [E] WHAT DOES THE CAP COST IN ARITHMETIC?  The cap removes partials and `attn_combine` merges the ones that
      remain, so a capped row is the SAME lane set summed in a DIFFERENT order — not bit-identical, and the bet
      claims no bit-identity. That claim is checked here rather than asserted, on the house's own reductions
      (`prototype/attn_tile_model.py`'s `online` = `attn_core`'s streaming algebra, `merge_partials` =
      `attn_combine`'s merge, imported not retyped), against the published band for a REASSOCIATING route —
      `engine/gate.py`'s `CASCADE_BAND`, imported by the same mechanism `attn_tile_model.py` uses, because a
      judge may not retype a band. The injection is §07.3's mis-normalisation (a partial that divided inside
      itself and is merged anyway), which is the defect this route is most likely to commit and the one the
      mirror's own docstring records hiding in this file until §20.

Usage:
    .venv/bin/python prototype/attn_percost/a9_lc_geom_check.py            # the five sections
    .venv/bin/python prototype/attn_percost/a9_lc_geom_check.py --inject   # the controls must go red
    .venv/bin/python prototype/attn_percost/a9_lc_geom_check.py --no-merge # skip [E] (no torch import)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LAUNCHER = ROOT / "engine" / "kernels_attn.cpp"
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "attn_decode")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAXSC = 512                 # `prototype/attn_decode/attn_decode_kernels.h:32`
KMAXSPLIT = 64              # `engine/kernels_attn.cpp`'s `kAttnMaxSplit` — the ceiling's second factor
N_Q, DH = 8, 256            # `attn_decode_kernels.h:23`, `:26`
ROWS = 512                  # `engine/decode_step.h` `kMaxWaveRows` — one body's rows
WIDTHS = [512, 1024, 4104, 4608, 8192, 16384, 32768, 40000]
# 0 = `kAttnSplitCapOff` (no cap). The odd caps exist so [A]'s injection can fire; 129 > the ceiling and must
# be inert, because `min` cannot bind above it.
CAPS = [0, 2, 3, 4, 5, 8, 16, 32, 64, 129]
# [E]'s population: rows above MAXSC (below it the split path never runs), a straddler, the widest pinned
# fixture, and the width of the worker's device gate.
MERGE_WIDTHS = [513, 1030, 4104, 4608]
MERGE_CAPS = [2, 4, 8]


def split_for(lanes: int) -> int:
    """`engine/kernels_attn.cpp`'s `split_for` verbatim."""
    return (lanes + MAXSC - 1) // MAXSC


def cap_split(split: int, cap: int) -> int:
    """`engine/kernels_attn.cpp`'s `cap_split` verbatim: a cap >= 2 is an upper bound, anything else is OFF."""
    return min(split, cap) if cap >= 2 else split


def span_of(nrow: int, s_row: int, floor_div=False) -> int:
    """The engine's slice width. `floor_div` is [A]'s injected defect: the off-by-one a `nrow//s_row` cut has."""
    return (nrow // s_row) if floor_div else (nrow + s_row - 1) // s_row


def bodies(width: int):
    """A prefill of `width` lanes is `width/512` sequential bodies plus a remainder (§06.3). Yields
    `(lvalid, nrows)`; body i has `lvalid = 512i + 1` and its widest row is `lvalid + nrows - 1`
    (`engine/decode_step.cpp:696-688`). The remainder body is the one that makes widths like 4,104 —
    the widest PINNED fixture — part of the population instead of a case nobody runs."""
    full, rest = divmod(width, ROWS)
    for i in range(full):
        yield i * MAXSC + 1, ROWS
    if rest:
        yield full * MAXSC + 1, rest


def slices(nrow: int, s_row: int, floor_div=False):
    out, span = [], span_of(nrow, s_row, floor_div)
    for y in range(s_row):
        lo = y * span
        if lo >= nrow:
            break
        out.append((lo, min(nrow, lo + span)))
    return out


def check_source(inject: bool):
    """[D] the ceiling throw reads the UNCAPPED split, and it precedes the cap. Both halves are the guarantee;
    either one broken means a plan of 33,000+ lanes becomes admissible by asking for a cap, and
    `gate.py::lane_ceiling()` starts asserting a product the engine no longer enforces.

    The injection is the REAL widening, not a swapped variable: rewrite the throw to compare `split_eff`.
    """
    lines = LAUNCHER.read_text().splitlines()
    if inject:
        lines = [ln.replace("if (split > kAttnMaxSplit)", "if (split_eff > kAttnMaxSplit)") for ln in lines]

    def at(needle):
        hits = [i for i, ln in enumerate(lines) if needle in ln and not ln.strip().startswith("//")]
        return hits[0] if len(hits) == 1 else None

    refuse = at("if (split > kAttnMaxSplit)")
    capped = at("const int split_eff = cap_split(split, a.split_cap);")
    notes = []
    if refuse is None:
        notes.append("no unique `if (split > kAttnMaxSplit)` — the ceiling is not compared on the uncapped"
                     " split, so a cap can admit a plan past MAXSC x kAttnMaxSplit")
    if capped is None:
        notes.append("no unique `const int split_eff = cap_split(split, a.split_cap);` — the cap site moved"
                     " or was re-named, and this check cannot locate it")
    if notes:
        return False, notes
    if refuse > capped:
        notes.append(f"the ceiling throw (line {refuse + 1}) does not precede the cap (line {capped + 1}):"
                     " `split_eff` would not be in scope, and a re-order that fixes the scope by comparing"
                     " `split_eff` is the silent-widening failure above")
    return not notes, notes


def merge_rows(inject: bool) -> tuple[list, list, dict]:
    """[E] the capped merge, on the mirror's own arithmetic.

    `attn_tile_model.online` reproduces `attn_core`'s streaming tile algebra (running max, weights NOT
    rounded, the divide left to the merge) and `merge_partials` reproduces `attn_combine`'s merge in window
    order, both in the kernel's order rather than in numpy's. A row run through them at `split` partials and
    at `min(split, cap)` partials is the engine's re-association, not a model of a model.

    The band is `engine/gate.py`'s `CASCADE_BAND` — the published floor for a reassociating route — imported,
    never retyped, by the mechanism `attn_tile_model._load_gate_band()` uses.
    """
    import numpy as np
    import attn_tile_model as M

    band = M._load_gate_band()
    cos_deficit_max = band["row_1cos_max"]          # 2.0e-3, hand<->hipBLAS cascade floor
    meanrel_max = band["stage_meanrel_max"]         # 3.0e-2

    def row_out(sc, V, s_row, normalized=True):
        """one row's DH-vector after the merge and the epilogue's single divide."""
        parts = [M.online(sc[lo:hi], V[lo:hi], M.TILEW) for lo, hi in M.row_bounds(len(sc), s_row)]
        _, l, acc = M.merge_partials(parts)
        # `normalized=False` is §07.3's bug: the partial divided inside its own block and the merge re-merged
        # it anyway. It is applied to the CAPPED side ONLY — injecting it into both sides would compare two
        # wrong answers and pass, which is the vacuous-control failure this file already records for [A].
        return np.asarray(acc, dtype=np.float32) if not normalized \
            else np.asarray(acc, dtype=np.float32) / np.float32(l)

    rng = np.random.default_rng(20261019)
    rows, fails = [], []
    for lanes in MERGE_WIDTHS:
        natural = split_for(lanes)
        # Scores in the range a real row occupies (the engine scales by 1/sqrt(DH) and the dot of two
        # unit-norm-ish 256-vectors), V ~ N(0,1). Content is synthetic; the ARITHMETIC is the kernel's.
        sc = np.float32(rng.normal(0.0, 4.0, lanes))
        import torch
        V = torch.from_numpy(np.float32(rng.normal(0.0, 1.0, (lanes, DH))))
        base = row_out(sc, V, natural)
        for cap in MERGE_CAPS:
            s_row = cap_split(natural, cap)
            if s_row == natural:
                continue                      # a cap that cannot bind has nothing to witness
            got = row_out(sc, V, s_row, normalized=not inject)
            d = got - base
            cos = float(np.dot(base, got) / (np.linalg.norm(base) * np.linalg.norm(got)))
            meanrel = float(np.abs(d).mean() / np.abs(base).mean())
            deficit, top = 1.0 - cos, int(np.argmax(got)) == int(np.argmax(base))
            ok = deficit <= cos_deficit_max and meanrel <= meanrel_max
            rows.append((lanes, natural, s_row, cos, meanrel, top, ok))
            if not ok:
                fails.append((lanes, cap, f"1-cos={deficit:.3e} meanrel={meanrel:.3e}"))
    return rows, fails, {"row_1cos_max": cos_deficit_max, "stage_meanrel_max": meanrel_max}


def main() -> int:
    inject = "--inject" in sys.argv[1:]
    bad = {k: [] for k in "ABC"}

    scratch_slots = ROWS * N_Q * KMAXSPLIT        # attn_part_floats(ROWS) / (DH+2)
    worst_slot, over_span = 0, {}
    plans = 0
    for width in WIDTHS:
        for cap in CAPS:
            for lvalid, nrows in bodies(width):
                nrow = lvalid + nrows - 1         # the body's widest row: lanes_of(row) = lvalid + row
                natural = split_for(nrow)
                if natural > KMAXSPLIT:
                    continue                      # [D] owns this band: it is a refusal, not a plan
                s_row = 1 if nrow <= MAXSC else cap_split(natural, cap)
                plans += 1
                # [A] the slices must tile [0, nrow) exactly.
                cur = 0
                for lo, hi in slices(nrow, s_row, floor_div=inject):
                    if lo != cur or hi <= lo:
                        bad["A"].append((width, cap, "gap/overlap", lo, hi))
                    cur = hi
                if cur != nrow:
                    bad["A"].append((width, cap, "row not fully covered", cur, nrow))
                if s_row > KMAXSPLIT:
                    bad["A"].append((width, cap, "split exceeds ceiling", s_row, KMAXSPLIT))
                # [B] the reachable slot index inside the ceiling-sized scratch. The injected defect strides
                # the slot by the ceiling while the allocation follows the run's split — [B3]'s mismatch.
                stride = KMAXSPLIT if inject else s_row
                sized = ROWS * N_Q * (s_row if inject else KMAXSPLIT)
                last_slot = ((nrows - 1) * N_Q + (N_Q - 1)) * stride + (stride - 1)
                worst_slot = max(worst_slot, last_slot)
                if last_slot >= sized:
                    bad["B"].append((width, cap, "slot past scratch", last_slot, sized))
                if s_row > 1 and span_of(nrow, s_row) > MAXSC:
                    over_span[(cap, width)] = span_of(nrow, s_row)

    d_ok, d_notes = check_source(inject)
    e_rows, e_fails, e_band = ([], [], None)
    if "--no-merge" not in sys.argv[1:]:
        e_rows, e_fails, e_band = merge_rows(inject)

    print(f"[A] slices tile the row exactly — {plans} plans "
          f"({len(WIDTHS)} widths x {len(CAPS)} caps, {ROWS} rows/body plus remainder bodies);"
          f" defects: {len(bad['A'])}")
    print(f"[B] partial scratch sized from kAttnMaxSplit={KMAXSPLIT}: {scratch_slots} slots;"
          f" worst reachable index {worst_slot} -> inside: {worst_slot < scratch_slots};"
          f" defects: {len(bad['B'])}")
    print(f"[C] (cap,width) pairs whose block span exceeds MAXSC={MAXSC} — legal only for the WMMA wave;"
          f" the VALU spellings stage the whole span in s_sc[G][MAXSC] and must be refused:")
    for (cap, width) in sorted(over_span):
        print(f"      cap={cap:>3} width={width:>6} -> span={over_span[(cap, width)]}")
    if not over_span:
        print("      none")
    print(f"[D] the ceiling throw reads the uncapped split and precedes the cap — "
          f"{'ok' if d_ok else 'NOT ok'} (`{LAUNCHER.name}`); notes: {d_notes or ['none']}")
    if e_band is not None:
        print(f"[E] the capped merge against the uncapped one, on `attn_tile_model`'s own reductions, judged"
              f" by `engine/gate.py`'s CASCADE_BAND (1-cos <= {e_band['row_1cos_max']:.1e},"
              f" meanrel <= {e_band['stage_meanrel_max']:.1e}): {len(e_rows)} rows,"
              f" outside: {len(e_fails)}")
        for lanes, nat, s_row, cos, meanrel, top, ok in e_rows:
            print(f"      {lanes:>6} lanes: split {nat:>2} -> {s_row:>2}  cos={cos:.9f}"
                  f"  meanrel={meanrel:.3e}  top-1 of the DH vector {'same' if top else 'MOVED'}"
                  f"  {'ok' if ok else 'OUTSIDE BAND'}")
    else:
        print("[E] skipped (--no-merge)")

    defects = sum(len(v) for v in bad.values()) + (0 if d_ok else 1) + len(e_fails)
    if inject:
        caught = len(bad["A"]) > 0 and len(bad["B"]) > 0 and not d_ok and (
            "--no-merge" in sys.argv[1:] or len(e_fails) > 0)
        print(("INJECT PASS" if caught else "INJECT FAIL")
              + " — all four injected defects (floor-divided span, strided slot with a run-sized scratch,"
                " ceiling compared on the capped value, a partial that divided inside its own block) must be"
                " caught. A control that cannot fire is not a control.")
        return 0 if caught else 1
    print("SELFTEST PASS" if defects == 0 else f"SELFTEST FAIL ({defects} defects)")
    return 0 if defects == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
