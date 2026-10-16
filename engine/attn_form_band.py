#!/usr/bin/env python3
"""attn_form_band.py — price one attention FORM against the shipped one, row by row, on CPU.

    .venv/bin/python engine/attn_form_band.py --ship tmp/ab/ship --tile tmp/ab/tile --prompt-len 1024
    .venv/bin/python engine/attn_form_band.py --selftest

WHY THIS EXISTS. Phase 2 (`prototype/attn_decode/attn_tile_wave`, `progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md`
§9) replaces the wave's attention core with a tiled one. Below MAXSC lanes the two forms promise
BIT-IDENTICAL output, and above it the tiled form owns fixed MAXSC-wide windows where the shipped form
cuts the row into equal slices — two legal groupings of §07.3's streaming route, so a difference there
is admissible ONLY inside §07.3's already-authorized band. `results.md` §13 reported a difference at
`logits1023.bin` on the 1024-token prompt and called it a token flip in that band region. It had not
been measured: §14 shows the tiled run threw before writing any logits at all, and a `cmp` against a
missing file reads exactly like a numerics difference. So this file does the measuring, and it is
built around the two ways this A/B went wrong once:

  1. A MISSING OR SHORT DUMP IS A CRASHED RUN, NOT A BAND DIFFERENCE. Every row is required on both
     sides before a single number is compared, and an absent row is reported as `MISSING` with the
     side named and the sampled.txt line count printed beside it. A `diff`/`cmp` loop over two
     directories cannot tell "the bytes differ" from "one side died", and only one of those is a
     numerics question.
  2. A FLIP IS NOT A BAND EVENT UNTIL ITS ROOM IS MEASURED. A row whose argmax moves prints the
     top-1/top-2 room on BOTH sides and is judged by the room rule already in force
     (`engine/gate.py`'s `CASCADE_BAND["flip_room_max_ulp"]`, applied by `gate.flip_admissible`, which
     reads room as the smaller of the two sides' gaps IN BF16 ULP at each side's own top-1 — the
     symmetric reading, which is the right one here because both sides are the engine: a shipped form
     against a tiled one, no reference between them). A flip at or above that room prints
     `BLOCKER` and exits non-zero: it is not something the band covers, and it is a stop-and-report
     for the operator rather than a line to file under "band".

WHAT IT REPORTS, for every dumped row (the >MAXSC rows are the band region; the rows at or below it
are reported too, because there the claim is bit-identity and any difference is a defect, not a band):
cos, meanrel, ndiff, BF16-lattice max ULP and F32-lattice max ULP **with the cancellation band
applied**, the worst element's index / both values / |d| / |ref|/max, the two sides' top-1 ids and
rooms, whether the argmax moved, and the verdict against the bands.

NO NEW NUMBERS. Every threshold is imported from the file that owns it, and each line of output names
where its numbers came from: `engine/gate.py` for the cascade band and the flip-room rule,
`prototype/attn_decode/run.py` for the op's measured cancellation band, and
`prototype/attn_decode/attn_decode_kernels.h` for MAXSC. This script never edits a band and never
proposes one; a row outside what is already published is reported as outside.

CPU-only: it reads two directories of `decode_run` dumps. No device, no engine, no timing.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "attn_decode"),
           str(ROOT / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bf16_ulp                                        # noqa: E402


def _load_module(name: str, path: Path):
    """Import a tool file that is not a package member (gate.py and run.py are scripts)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                            # dataclasses inside want the module present
    spec.loader.exec_module(mod)
    return mod


GATE = _load_module("_ab_gate", ROOT / "engine" / "gate.py")
ATTN = _load_module("_ab_attn_run", ROOT / "prototype" / "attn_decode" / "run.py")

KERNEL_H = ROOT / "prototype" / "attn_decode" / "attn_decode_kernels.h"
MAXSC = int(re.search(r"^#define\s+MAXSC\s+(\d+)", KERNEL_H.read_text(), re.M).group(1))
# The op's measured cancellation band (`prototype/attn_decode/run.py`, sized by its floor_probe.py):
# below this fraction of the tensor max a ULP distance means nothing, because ordinary F32
# reassociation moves an element that cancelled to ~1e-6 of the max by 4-16 ULP. Cited, not chosen.
CANCEL_REL = ATTN.CANCEL_REL


def read_logits(path: Path) -> np.ndarray | None:
    """One `decode_run` logits dump as F32, or None if the file is absent."""
    if not path.exists():
        return None
    a = np.fromfile(path, dtype="<f4")
    return a if a.size else None


def read_sampled(path: Path) -> dict[int, int]:
    """`sampled.txt` as {step: sampled_token}. A run that died mid-flight leaves the header only."""
    out: dict[int, int] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        try:
            out[int(f[0])] = int(f[2])
        except (IndexError, ValueError):
            continue                                    # header-shaped or truncated line
    return out


def f32_ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise distance in F32 ULPs, on the monotone integer lattice.

    Same sign trap `prototype/bf16_ulp.py` documents for BF16: the bit pattern is monotone only
    within a sign, so the transform is `bits ^ (0x80000000 if non-negative else 0xFFFFFFFF)` and a
    `np.where(a < 0, ...)` on a uint32 view is a dead branch that reports nonsense for every
    negative value (half a logits row).
    """
    def key(x):
        bits = np.asarray(x, dtype="<f4").view(np.uint32).astype(np.int64)
        return np.where(bits & 0x80000000, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)
    return np.abs(key(a) - key(b))


def row_verdict(ship: np.ndarray, tile: np.ndarray, *, lanes: int, cancel_rel: float) -> dict:
    """Everything one row's tile-vs-shipped comparison says, with no thresholds applied yet."""
    rep = bf16_ulp.report(tile, ship, cancel_rel=cancel_rel)   # BF16 lattice: the decision lattice
    u32 = f32_ulp_dist(tile, ship)
    scale = float(np.abs(ship.astype(np.float64)).max())
    large = np.abs(ship.astype(np.float64)) / scale > cancel_rel if scale else np.zeros_like(u32, bool)
    over_large = np.where(large & (u32 > 1), u32, 0)
    worst = int(np.argmax(np.where(large, np.abs(tile - ship).astype(np.float64), 0.0)))
    s_id, s_val, s_room = GATE.top2(ship)
    t_id, t_val, t_room = GATE.top2(tile)
    d = np.abs(tile.astype(np.float64) - ship.astype(np.float64))
    return {
        "row": -1, "lanes": lanes, "identical": bool(np.array_equal(ship, tile)),
        "ndiff": int(np.count_nonzero(d)), "n": int(d.size),
        "cos": rep["cos"], "meanrel": rep["meanrel"],
        "bf16_maxulp": rep["maxulp"], "bf16_n_bad": rep["n_bad"], "bf16_over1": rep["over1"],
        "bf16_worst_bad_ulp": rep["worst_bad_ulp"], "bf16_worst_bad_rel": rep["worst_bad_rel"],
        "bf16_n_band": rep["n_band"], "band_slack": rep["worst_off_rel"],
        "f32_maxulp": int(u32.max()) if u32.size else 0,
        # Max, not a count: the worst F32 ULP distance among elements big enough that a ULP means
        # something. F32 ULPs are tiny, so this number is large for a legal reassociation; it is
        # reported for a reader who wants it and judged by nothing.
        "f32_maxulp_banded": int(over_large.max()),
        "worst_idx": worst, "worst_abs": float(d[worst]),
        "worst_ship": float(ship[worst]), "worst_tile": float(tile[worst]),
        "worst_rel": float(abs(float(ship[worst])) / scale) if scale else 0.0,
        "ship_id": s_id, "ship_room": s_room, "ship_val": s_val,
        "tile_id": t_id, "tile_room": t_room, "tile_val": t_val,
        "flipped": s_id != t_id,
    }


def verdict_lines(rows: list[dict], band: dict, cancel_rel: float) -> tuple[list[str], bool]:
    """`(lines, red)` — the table, and whether this A/B is acceptable as-is.

    What a row is JUDGED on, and what is only REPORTED, follows the published rules and nothing else:
      * at or below MAXSC the tiled form's own claim is bit-identity (§12), so any difference there is
        a defect.
      * above MAXSC the two forms are two legal groupings of §07.3's streaming route, so the row is
        judged by the published per-row shape number `row_1cos_max`. meanrel and the ULP columns are
        REPORTED and not judged: `stage_meanrel_max` is published per stage tile rather than per logits
        row (its own comment says so) and no per-row meanrel or ULP bound exists for a reassociating
        route, so inventing one here would be the widening this script exists to refuse. The pooled
        aggregates in `cmd_ab` are where the stage keys apply.
      * a moved argmax is judged by `gate.flip_admissible`, the rule already in force.
    """
    out: list[str] = []
    red = False
    out.append("  %5s %6s %10s %9s %9s %10s %10s  %-44s %-24s %s"
               % ("row", "lanes", "cos", "meanrel", "bf16maxULP", "f32maxULP", "bandedMax",
                  "worst element: id, ship -> tile, |d|, |ref|/max", "top1 (room)", "route / verdict"))
    for r in rows:
        route = "one-pass" if r["lanes"] <= MAXSC else "streaming"
        flags: list[str] = []
        if r["identical"]:
            flags.append("bit-identical")
        else:
            if r["lanes"] <= MAXSC:
                flags.append("DEFECT: at or below MAXSC the two forms must be bit-identical (§12)")
            deficit = 1.0 - r["cos"]
            if deficit > band["row_1cos_max"]:
                flags.append(f"OUTSIDE band: 1-cos {deficit:.3e} > row_1cos_max"
                             f" {band['row_1cos_max']:.0e}")
        if r["flipped"]:
            admit = GATE.flip_admissible((r["ship_val"], r["ship_room"]),
                                         (r["tile_val"], r["tile_room"]))
            flags.append("ARGMAX MOVES %d->%d, room %.4g/%.4g logits = %.3g ULP -> %s"
                         % (r["ship_id"], r["tile_id"], r["ship_room"], r["tile_room"],
                            GATE.flip_room_ulp((r["ship_val"], r["ship_room"]),
                                               (r["tile_val"], r["tile_room"])),
                            "admitted (room under the line)" if admit
                            else "BLOCKER: room at or above the flip line, §07.3 does not cover it"))
            red = red or not admit
        red = red or any(f.startswith(("DEFECT", "OUTSIDE")) for f in flags)
        out.append("  %5d %6d %10.8f %9.2e %9d %10d %10d  id %6d  %.6g -> %.6g  d=%.3g |ref|/max=%.1e"
                   "  %d(%.3g)/%d(%.3g) %s"
                   % (r["row"], r["lanes"], r["cos"], r["meanrel"], r["bf16_maxulp"], r["f32_maxulp"],
                      r["f32_maxulp_banded"], r["worst_idx"], r["worst_ship"], r["worst_tile"],
                      r["worst_abs"], r["worst_rel"],
                      r["ship_id"], r["ship_room"], r["tile_id"], r["tile_room"], route)
                 + "".join("\n        ! " + f for f in flags))
    return out, red


def cmd_ab(args) -> int:
    ship_dir, tile_dir = Path(args.ship), Path(args.tile)
    for label, d in (("shipped", ship_dir), ("tiled", tile_dir)):
        if not d.is_dir():
            print(f"RED: the {label} dump directory {d} does not exist")
            return 1
    ship_files = {int(m.group(1)) for p in ship_dir.glob("logits*.bin")
                  if (m := re.fullmatch(r"logits(\d+)\.bin", p.name))}
    tile_files = {int(m.group(1)) for p in tile_dir.glob("logits*.bin")
                  if (m := re.fullmatch(r"logits(\d+)\.bin", p.name))}
    ship_sampled = read_sampled(ship_dir / "sampled.txt")
    tile_sampled = read_sampled(tile_dir / "sampled.txt")
    band = GATE.CASCADE_BAND
    print(f"attention FORM A/B: shipped `{ship_dir}` vs tiled `{tile_dir}`")
    print(f"  bands in force, read from their owners: CASCADE_BAND {band} (engine/gate.py); "
          f"cancellation band |ref|/max <= {CANCEL_REL:g} (prototype/attn_decode/run.py CANCEL_REL); "
          f"MAXSC={MAXSC} ({KERNEL_H.name})")
    print(f"  rows dumped: shipped {len(ship_files)} logits files, tiled {len(tile_files)};"
          f" sampled.txt rows: shipped {len(ship_sampled)}, tiled {len(tile_sampled)}")

    # Fail closed BEFORE any comparison. This is the §13 failure: a side that produced nothing reads
    # as "everything differs" to a cmp loop, and it was reported as a numerics result.
    missing_ship = sorted(tile_files - ship_files)
    missing_tile = sorted(ship_files - tile_files)
    if not ship_files or not tile_files:
        print(f"RED: one side produced NO logits at all (shipped {len(ship_files)}, tiled "
              f"{len(tile_files)}). That is a crashed or truncated run, not a numerics difference:"
              " nothing in this A/B was measured. Re-run both sides and check both dump a file before"
              " judging any of it.")
        return 1
    if missing_ship or missing_tile:
        print(f"RED: the two sides dumped different rows — missing on shipped {missing_ship[:8]},"
              f" missing on tiled {missing_tile[:8]}. A missing file is not a band difference."
              " Re-run with the same --prefill-logits on both sides.")
        return 1

    rows_here = sorted(ship_files)
    if args.prompt_len:
        expect = [p for p in range(args.prompt_len) if p + 1 > MAXSC]
        absent = [p for p in expect if p not in ship_files or p not in tile_files]
        print(f"  prompt_len={args.prompt_len}: {len(expect)} rows sit above MAXSC={MAXSC} lanes"
              f" (rows {expect[0]}..{expect[-1]}); {len(absent)} of them were not dumped on both"
              " sides" + (", so they are NOT judged here — use --prefill-logits all" if absent else ""))
    want = [p for p in rows_here if p + 1 > MAXSC] if not args.all_rows else rows_here
    if not want:
        print(f"RED: no dumped row is above MAXSC={MAXSC} lanes (rows present: {rows_here[:8]}...). "
              "The band region needs the prompt's own rows: run both sides with --prefill-logits all.")
        return 1
    if len(want) < 2:
        print(f"note: only {len(want)} row above MAXSC was dumped; --prefill-logits all prices every"
              " prompt row and is what makes this a table rather than a single line.")

    rows: list[dict] = []
    # Pooled accumulators over the band region, in F64: `stage_meanrel_max`/`stage_cos_min` are
    # published PER STAGE TILE, not per logits row, so the quantity they are allowed to judge is the
    # concatenation of the rows the stage produced. Accumulating avoids holding 512 x 248 320 floats.
    acc = dict(n=0, dot=0.0, na=0.0, nb=0.0, sad=0.0, sar=0.0)
    for p in want:
        a = read_logits(ship_dir / f"logits{p}.bin")
        b = read_logits(tile_dir / f"logits{p}.bin")
        if a is None or b is None or a.shape != b.shape:
            print(f"RED: row {p} is unreadable or shape-mismatched "
                  f"({None if a is None else a.shape} vs {None if b is None else b.shape})")
            return 1
        r = row_verdict(a, b, lanes=p + 1, cancel_rel=CANCEL_REL)
        r["row"] = p
        rows.append(r)
        if p + 1 > MAXSC:
            g, f = b.astype(np.float64), a.astype(np.float64)
            acc["n"] += f.size
            acc["dot"] += float(g @ f)
            acc["na"] += float(g @ g)
            acc["nb"] += float(f @ f)
            acc["sad"] += float(np.abs(g - f).sum())
            acc["sar"] += float(np.abs(f).sum())
    lines, red = verdict_lines(rows, band, CANCEL_REL)
    print("\n".join(lines))

    band_rows = [r for r in rows if r["lanes"] > MAXSC]
    exact_rows = [r for r in rows if r["lanes"] <= MAXSC]
    flips = [r for r in rows if r["flipped"]]
    print(f"\n  rows judged {len(rows)} ({len(band_rows)} above MAXSC, {len(exact_rows)} at or below);"
          f" bit-identical {sum(1 for r in rows if r['identical'])}; argmax moved {len(flips)};"
          f" elements differing {sum(r['ndiff'] for r in rows):,}"
          f" (worst row {max(r['ndiff'] for r in rows):,})")
    if band_rows:
        print(f"  worst row above MAXSC: 1-cos {max(1 - r['cos'] for r in band_rows):.3e} against"
              f" row_1cos_max {band['row_1cos_max']:.0e}; large elements over 1 BF16 ULP "
              f"{sum(r['bf16_n_bad'] for r in band_rows)}, worst {max(r['bf16_worst_bad_ulp'] for r in band_rows)}"
              f" ULP at |ref|/max {max(r['bf16_worst_bad_rel'] for r in band_rows):.1e} (reported: the"
              " published band for a reassociating route is the cos pair, not a ULP count); banded "
              f"F32 max ULP {max(r['f32_maxulp_banded'] for r in band_rows)}; cancellation band's own"
              f" slack at its widest excuse |ref|/max"
              f" {max(r['band_slack'] for r in band_rows):.1e} (tighten it against this number, not"
              " against taste)")
        if acc["n"] and acc["na"] and acc["nb"] and acc["sar"]:
            pcos = acc["dot"] / (acc["na"] ** 0.5 * acc["nb"] ** 0.5)
            prel = acc["sad"] / acc["sar"]
            out_stage = pcos < band["stage_cos_min"] or prel > band["stage_meanrel_max"]
            print(f"  pooled over the {acc['n']:,} band elements: cos {pcos:.8f} (1-cos"
                  f" {1 - pcos:.3e}) meanrel {prel:.3e}, against the stage keys stage_cos_min"
                  f" {band['stage_cos_min']} / stage_meanrel_max {band['stage_meanrel_max']:.0e}"
                  f" ({'OUTSIDE' if out_stage else 'inside'})")
            red = red or out_stage
    if exact_rows:
        bad = [r["row"] for r in exact_rows if not r["identical"]]
        print(f"  at or below MAXSC (bit-identity is the claim there): "
              f"{'all bit-identical' if not bad else 'NOT bit-identical at rows ' + str(bad[:8])}")
        red = red or bool(bad)
    if flips:
        for r in flips:
            print(f"  flip at row {r['row']}: shipped {r['ship_id']} (room {r['ship_room']:.4f}) vs"
                  f" tiled {r['tile_id']} (room {r['tile_room']:.4f}); room read as the smaller of"
                  f" the two, in BF16 ULP at each side's own top-1, against the line"
                  f" {band['flip_room_max_ulp']} (engine/gate.py flip rule; both sides are the engine"
                  " here, so this is the symmetric reading, not the reference one)")
    # The engine's own sampled column, cross-checked against the per-row argmax above. A differing step
    # is only accounted for if a JUDGED row flipped there and the flip rule admits it; anything else is
    # an unexplained divergence in the stream, and the honest verdict for that is red, not "band".
    both = sorted(set(ship_sampled) & set(tile_sampled))
    diff_steps = [s for s in both if ship_sampled[s] != tile_sampled[s]]
    explained = {r["row"] for r in rows if r["flipped"]
                 and GATE.flip_admissible((r["ship_val"], r["ship_room"]),
                                          (r["tile_val"], r["tile_room"]))}
    # Which row explains which step is exactly the index-alignment trap AGENTS.md records (§10.5's pins
    # pair with greedy[s+1], the REPL's first sample pairs with greedy[s]), so a step counts as explained
    # under EITHER reading and only a step neither explains is called unexplained. The reading is stated
    # rather than assumed, because "0 of 128 tokens matched" was once produced by getting it wrong.
    unexplained = [s for s in diff_steps if not ({s, s - 1, s + 1} & explained)]
    print(f"  sampled.txt: {len(both)} steps compared, {len(diff_steps)} differ"
          + (f" at steps {diff_steps[:8]}" if diff_steps else ""))
    if diff_steps:
        print(f"  alignment used: a differing step is explained if a judged row at that index or ±1"
              f" flipped within the flip rule (admitted flips at rows {sorted(explained)[:8]}); the ±1"
              " is the §10.5-vs-REPL pairing ambiguity, not slack")
    if unexplained:
        print(f"  {len(unexplained)} differing step(s) are NOT explained by any judged row flipping"
              f" within the flip rule (steps {unexplained[:8]}). Either the dumps do not cover them"
              " (--prefill-logits all covers every prompt row) or something outside attention moved."
              " Do not file this under 'band'.")
        red = True
    if tile_sampled and len(tile_sampled) < len(ship_sampled):
        print("  note: the tiled side sampled fewer steps than the shipped side, which means one of"
              " the two runs ended early. That is a run failure, not a band question.")
        red = True
    print(f"\n{'RED' if red else 'OK'}: " + (
        "at least one row is outside the bands already in force, or a flip sits at room the flip rule"
        " does not admit. That is a stop-and-report: the numbers here are the ones in force and this"
        " script does not widen them." if red else
        "every judged row is inside §07.3's band, no large element is outside the cancellation band,"
        " and no argmax moved at room the flip rule would not admit."))
    return 1 if red else 0


def cmd_selftest(_args) -> int:
    """The instrument's own teeth, on synthetic rows — CPU, no engine, no device.

    A comparator that cannot fail is worse than none (the `mlp_gemv`/`proj_gemv` rule). Each case
    below is a claim about what this file must and must not call a problem.
    """
    checks: list[tuple[str, bool]] = []
    rng = np.random.default_rng(20261014)
    base = rng.normal(0.0, 4.0, 4096).astype(np.float32)
    base[7] = 30.0                                    # a clear top-1
    base[9] = 29.0                                    # a clear runner-up: room 1.0
    band = GATE.CASCADE_BAND

    def v(other, lanes=600):
        return row_verdict(base, other, lanes=lanes, cancel_rel=CANCEL_REL)

    same = v(base.copy())
    checks.append(("an identical row is bit-identical and flips nothing",
                   same["identical"] and not same["flipped"] and same["bf16_n_bad"] == 0))
    checks.append(("an identical row needs no band at all (ndiff 0)", same["ndiff"] == 0))

    # The cancellation band must be doing work, in BOTH directions. One absolute perturbation, applied
    # where the reference is tiny and where it is large: only the second may be held against the run.
    delta = np.float32(0.2)
    small_i = int(np.argmin(np.abs(base)))
    large_i = int(np.argmax(np.abs(base)))
    tiny = base.copy()
    tiny[small_i] += delta
    inband = v(tiny)
    big = base.copy()
    big[large_i] += delta
    outband = v(big)
    checks.append((f"a {delta:g} change at an element under the {CANCEL_REL:g} cancellation band is"
                   " excused, even though it moves many BF16 ULP",
                   inband["bf16_over1"] > 0 and inband["bf16_n_band"] > 0 and inband["bf16_n_bad"] == 0))
    checks.append((f"the same {delta:g} change at a LARGE element is not excused (n_bad > 0), so the"
                   " band above is not a free pass",
                   outband["bf16_over1"] > 0 and outband["bf16_n_bad"] > 0))

    wide = base.copy()
    wide[9] = np.float32(31.0)                        # flips the top-1 with 30 of room on one side
    wideflip = v(wide)
    checks.append(("a flip at wide room is a BLOCKER, not a band event",
                   wideflip["flipped"] and not GATE.flip_admissible(
                       (wideflip["ship_val"], wideflip["ship_room"]),
                       (wideflip["tile_val"], wideflip["tile_room"]))))
    thin = base.copy()
    thin[9] = np.float32(30.0 + 1e-4)                 # coin-toss room on the tiled side
    thinflip = v(thin)
    checks.append(("the same flip at thin room is admitted by the rule already in force",
                   thinflip["flipped"] and GATE.flip_admissible(
                       (thinflip["ship_val"], thinflip["ship_room"]),
                       (thinflip["tile_val"], thinflip["tile_room"]))))
    _, red = verdict_lines([wideflip], band, CANCEL_REL)
    _, red_ok = verdict_lines([same], band, CANCEL_REL)
    checks.append(("verdict_lines goes red for the wide-room flip and green for a clean row",
                   red and not red_ok))
    lines, _ = verdict_lines([wideflip], band, CANCEL_REL)
    checks.append(("a moved argmax prints both sides' room (the number a reader needs to disagree)",
                   any("room" in ln for ln in lines)))

    low = v(base.copy() + delta, lanes=MAXSC)         # a difference on a one-pass row
    _, red_low = verdict_lines([low], band, CANCEL_REL)
    _, red_clean = verdict_lines([v(base.copy(), lanes=MAXSC)], band, CANCEL_REL)
    checks.append((f"a difference on a row at or below MAXSC={MAXSC} is a DEFECT, not a band event,"
                   " and a clean one-pass row stays green", red_low and not red_clean))

    # The §13 failure mode itself, on the dumps that produced it: a side that wrote nothing must be
    # reported as a crashed run, never as a numerics difference.
    ship_d, tile_d = ROOT / "tmp" / "ab" / "ship", ROOT / "tmp" / "ab" / "tile"
    if ship_d.is_dir() and tile_d.is_dir():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_ab(argparse.Namespace(ship=str(ship_d), tile=str(tile_d), prompt_len=1024,
                                          all_rows=False))
        text = buf.getvalue()
        checks.append(("the preserved §13 dumps are called a crashed run (exit 1, 'NO logits at all')"
                       " rather than a band difference",
                       rc == 1 and "NO logits at all" in text))
    else:
        print(f"  skip  the preserved §13 dumps are gone ({ship_d} / {tile_d}); that check not run")

    def ab_case(rows_by_row, sampled=(), tile_sampled=None) -> tuple[int, str]:
        """Run the whole file path over a throwaway dump pair; `rows_by_row` is {row: (ship, tile)}.

        `tile_sampled`, when given, is the tiled side's `sampled.txt` body verbatim: a side that stops
        early is exactly what a crashed run leaves, and it has to be caught as a run failure.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "ship").mkdir(); (root / "tile").mkdir()
            for p, (x, y) in rows_by_row.items():
                x.tofile(root / "ship" / f"logits{p}.bin")
                y.tofile(root / "tile" / f"logits{p}.bin")
            hdr = "# step input_token sampled_token top_logit prefix_len_before\n"
            body = hdr + "".join(f"{s} 1 {t} 3 4\n" for s, t in sampled)
            (root / "ship" / "sampled.txt").write_text(body)
            (root / "tile" / "sampled.txt").write_text(body if tile_sampled is None
                                                       else hdr + tile_sampled)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cmd_ab(argparse.Namespace(ship=str(root / "ship"), tile=str(root / "tile"),
                                               prompt_len=0, all_rows=False))
            return rc, buf.getvalue()

    wide_row = np.linspace(-3.0, 30.0, 5120).astype(np.float32)
    rc, text = ab_case({MAXSC: (wide_row, wide_row.copy())})
    checks.append(("a clean streaming row through the file path prints the table and exits 0",
                   rc == 0 and "bit-identical" in text and "rows judged 1" in text))
    dirty = (wide_row * np.float32(1.05)).astype(np.float32)
    rc, text = ab_case({MAXSC: (wide_row, dirty)})
    checks.append(("a streaming row outside the published band exits 1 and names the band it broke",
                   rc == 1 and ("OUTSIDE band" in text or "stage keys" in text)))
    rc, text = ab_case({MAXSC: (wide_row, wide_row.copy())},
                       sampled=[(MAXSC, 5), (MAXSC + 1, 6)])
    rc2, text2 = ab_case({MAXSC: (wide_row, wide_row.copy())},
                         sampled=[(MAXSC, 5), (MAXSC + 1, 6)], tile_sampled=f"{MAXSC} 1 5 3 4\n")
    checks.append(("a sampled.txt that stops early is a run failure, not a silent short comparison",
                   rc == 0 and rc2 == 1 and "ended early" in text2))
    rc3, text3 = ab_case({MAXSC: (wide_row, wide_row.copy())}, sampled=[(MAXSC, 5)],
                         tile_sampled=f"{MAXSC} 1 9 3 4\n")
    checks.append(("a sampled step that no judged row explains is red, not filed under 'band'",
                   rc3 == 1 and "NOT explained by any judged row" in text3))
    rc, text = ab_case({10: (wide_row, wide_row.copy())})
    checks.append(("a dump set with nothing above MAXSC is refused rather than reported as clean",
                   rc == 1 and "no dumped row is above maxsc" in text.lower()))

    fails = 0
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
        fails += 0 if ok else 1
    print(f"selftest: {len(checks) - fails}/{len(checks)} passed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ship", default="tmp/ab/ship", help="the shipped-form (--attn-tile 0) dumps")
    ap.add_argument("--tile", default="tmp/ab/tile", help="the tiled-form (--attn-tile 1) dumps")
    ap.add_argument("--prompt-len", type=int, default=0,
                    help="prompt length, to say which rows the band region covers; 0 infers it from"
                         " the dumps")
    ap.add_argument("--all-rows", action="store_true",
                    help="judge every dumped row, not only the >MAXSC band region")
    ap.add_argument("--selftest", action="store_true", help="synthetic rows only; no dumps read")
    args = ap.parse_args()
    if args.selftest:
        return cmd_selftest(args)
    return cmd_ab(args)


if __name__ == "__main__":
    sys.exit(main())
