#!/usr/bin/env python3
"""Does the wide-scoped pooled cosine band still bite? A CPU witness for `gate.pooled_cos_band`.

`engine/gate.py` widened exactly one pooled line, and only for pairings with an A12 wide side
(`CASCADE_BAND.stage_cos_min_wide`, `pooled_cos_band()`, `side_owns_wide_form()`). `AGENTS.md`'s
numeric-format rule says a widened band must still catch every injection in the must-catch set, and the
existing injections cannot answer that question for this line: `--selftest`'s `--break-ab-logits` runs
`--dry-run`, so its sides start no binary, echo no `attn_fattn_wide`, and are therefore judged by the
ORDINARY line by design. So the wide line needed its own geometry-derived case list, derived from the
two lines it has to be distinguishable from — the same "derive the must-catch set from the geometry"
rule `prototype/attn_tile_model.py` and `prototype/gdn_walk/BET.md` record, applied to a band.

THE GEOMETRY. `corpus_drift` pools `cos = Σdot / sqrt(Σna·Σnb)`, so a row of `a = [1,0,…]` against
`b = [1,x,0,…]` has pooled `1-cos` = per-row `1-cos` = `1 − 1/sqrt(1+x²)`, exactly, and with one knob
the other statistics are computable rather than hoped about: the per-row `1-cos` equals the pooled one
(so it stays inside the 2e-03 `library` row line for every case below except the last) and the pooled
`meanrel` equals `x = sqrt(2d + d²)` (so it stays inside 3e-02 for `d <= ~4.5e-04`). That is what lets
this file require the LOCALITY of each case rather than merely a red.

THE CASES, and what each one is the must-catch for. The boundary targets are read from the bands
themselves rather than aimed at a round number, and each line is STRADDLED at one part in a thousand
rather than hit exactly: the dumps are F32, so the deficit the gate computes sits within ~1e-16 of the
target and could land either side of a `<=`, and a case that fails on a float rather than on a band is
noise. An injection canary (below) asserts each injected figure really is the figure it claims.

    1.5625e-04   the ordinary line's own worst known-good (dec8, `engine/wave_band_floor_probe.py`)
                 — both lines admit it. If this one stopped passing, a band would have been tightened
                 by accident, which is its own kind of damage.
    1.900e-04    inside the ordinary line — both admit.
    2.028112e-04 THE measured wide figure (smoke, wide batched-vs-carry). Ordinary REFUSES, wide
                 admits. This is the whole reason the scope exists, and the case that would go missing
                 if someone "fixed" it by moving the ordinary line instead.
    ordinary x 0.999 / x 1.001
                 one part in a thousand either side of the ORDINARY line: it admits its own boundary
                 and refuses just above it, and the wide line admits both — the shape of the A12 case.
    wide x 0.999 / x 1.02
                 one part in a thousand either side of the WIDE line: refused by the ordinary line and
                 admitted by the wide one below it, refused by both above it. The above-the-line case
                 is what keeps the wide band from being a band that cannot fail — the failure
                 `prototype/gdn_collapse` §1 and `prototype/attn_tile_model.py` exist to retire.
    4.300e-03    the size of the smallest defect `--break-ab-logits` injects at this check (per the
                 gate's own band comment) — refused by both, so nothing the old line caught has been
                 let go. At this size the construction necessarily trips the per-row and pooled-meanrel
                 lines too (`x` grows as `sqrt(2d)`), which is declared as an expected companion
                 rather than hidden.

Plus a direct witness of the SCOPE predicate on the three echoes a side can carry (`wide`, `fattn`,
no NUMERICS line at all), because a scope that is read off a run echo has to be shown to read it.

CPU only: two directories of synthetic `logitsN.bin` files, no engine, no device, no timing.

    .venv/bin/python prototype/attn_wide/a12_wide_band_witness.py
"""
from __future__ import annotations

import contextlib
import io
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import gate  # the shipping gate: its band, its statistic, its judging function

ROWS = 4
WIDTH = 64
WORK = ROOT / "tmp" / "a12_wide_band_witness"
COS_CHECK = "pool to a cosine inside the cascade band"

# (target pooled 1-cos deficit, what the ordinary line must say, what the wide line must say, why the
# case exists, whether the other three checks of `wave_band_checks` must stay green). The two boundary
# targets are appended in `main()` from the bands themselves, so this file cannot drift out of
# agreement with the numbers it exists to test.
CASES = [
    (1.5625e-04, "admit", "admit",
     "the ordinary line's worst known-good (dec8, wave_band_floor_probe 2026-10-16)", True),
    (1.9000e-04, "admit", "admit",
     "inside the ordinary line, so the wide move changed nothing here", True),
    (2.028112e-04, "refuse", "admit",
     "THE measured wide figure (smoke wide batched-vs-carry, A12-FIXES 0.2)", True),
    (4.3000e-03, "refuse", "refuse",
     "the --break-ab-logits size of defect, refused by both lines", False),
]


def x_for(deficit: float) -> float:
    """The off-axis offset that makes `1 − 1/sqrt(1+x²)` exactly `deficit`."""
    return math.sqrt(1.0 / (1.0 - deficit) ** 2 - 1.0)


def build_pair(tag: str, deficit: float) -> tuple[Path, Path]:
    """Two run directories whose pooled `1-cos` deficit is `deficit`, exactly to F64 rounding."""
    a_dir, b_dir = WORK / f"{tag}-a", WORK / f"{tag}-b"
    for d in (a_dir, b_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    x = x_for(deficit)
    a = np.zeros(WIDTH, dtype="<f4")
    a[0] = 1.0
    b = np.zeros(WIDTH, dtype="<f4")
    b[0], b[1] = 1.0, np.float32(x)
    for i in range(ROWS):
        a.tofile(a_dir / f"logits{i}.bin")
        b.tofile(b_dir / f"logits{i}.bin")
    return a_dir, b_dir


def judge(a_dir: Path, b_dir: Path, wide_pairing: bool) -> dict[str, str]:
    """Run the shipping check and return {check title: "admit"|"refuse"} for the four it makes.

    stdout is parsed rather than `gate.fails` alone because the printed line is what an operator reads,
    and a parse that found nothing is a failure here, never a pass — the `prototype/gdn_collapse` §1
    rule that a check whose input list can be empty cannot fail.
    """
    del gate.fails[:]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gate.wave_band_checks(a_dir, b_dir, "batched", "carry", "witness pair",
                              wide_pairing=wide_pairing)
    out: dict[str, str] = {}
    for line in buf.getvalue().splitlines():
        s = line.strip()
        for tag, verdict in (("[PASS]", "admit"), ("[FAIL]", "refuse")):
            if s.startswith(tag):
                name = s[len(tag):].strip().split(" — ")[0]
                out[name] = verdict
    if len(out) != 4:
        raise SystemExit(f"witness: expected 4 checks from wave_band_checks, parsed {len(out)} from"
                         f" {buf.getvalue()!r} — a witness that parses nothing cannot fail")
    return out


def main() -> int:
    gate.set_fixture("smoke")      # only for the module state `wave_row_1cos_band()` reads
    gate.set_route("library")      # the per-row line here is 2e-03; none of these cases aims at it
    WORK.mkdir(parents=True, exist_ok=True)
    bad: list[str] = []
    ordinary = 1.0 - gate.CASCADE_BAND["stage_cos_min"]
    wide = 1.0 - gate.CASCADE_BAND["stage_cos_min_wide"]
    print(f"bands in force, read from the shipping gate: ordinary pooled 1-cos <= {ordinary:.3e}"
          f" (floor {gate.CASCADE_BAND['stage_cos_min']}), wide-scoped <= {wide:.3e}"
          f" (floor {gate.CASCADE_BAND['stage_cos_min_wide']})")
    if not wide > ordinary:
        print("  [FAIL] the wide line is not wider than the ordinary one, so the scope is decoration")
        bad.append("wide line not wider")
    cases = list(CASES[:3]) + [
        (ordinary * 0.999, "admit", "admit", "one part in a thousand inside the ordinary line", True),
        (ordinary * 1.001, "refuse", "admit",
         "one part in a thousand above the ordinary line — the shape of the A12 case", True),
        (wide * 0.999, "refuse", "admit", "one part in a thousand inside the wide line", True),
        (wide * 1.02, "refuse", "refuse",
         "above the wide line: the wide-scoped band STILL BITES", True),
    ] + list(CASES[3:])

    for deficit, want_ord, want_wide, why, others_green in cases:
        a_dir, b_dir = build_pair(f"d{deficit:.6e}", deficit)
        got = gate.corpus_drift(a_dir, b_dir)["one_cos"]
        # The canary: the injected figure must BE the figure, in this window. A stale or no-op
        # injection otherwise reads as a verdict about the band (AGENTS.md, 2026-10-17).
        if abs(got - deficit) > 1e-9:
            print(f"  [FAIL] injection did not land: aimed {deficit:.6e}, corpus_drift reads"
                  f" {got:.6e} — {why}")
            bad.append(f"injection {deficit:.3e}")
            continue
        for label, wide_pairing, want in (("ordinary", False, want_ord), ("wide", True, want_wide)):
            verdicts = judge(a_dir, b_dir, wide_pairing)
            cos_keys = [k for k in verdicts if COS_CHECK in k]
            if len(cos_keys) != 1:
                print(f"  [FAIL] {label}: expected exactly one pooled-cosine check, got {cos_keys}")
                bad.append(f"{label} {deficit:.3e} parse")
                continue
            got_v = verdicts[cos_keys[0]]
            strays = {k: v for k, v in verdicts.items() if k not in cos_keys and v != "admit"}
            ok = got_v == want and (not strays or not others_green)
            print(f"  [{'PASS' if ok else 'FAIL'}] 1-cos {got:.6e} at the {label} line ->"
                  f" {got_v} (want {want}) — {why}"
                  + (f"; companion failures at this size: {len(strays)}"
                     if strays and not others_green
                     else (f"; other checks that also failed: {sorted(strays)}" if strays else "")))
            if not ok:
                bad.append(f"{label} {deficit:.3e}")

    # The scope predicate itself, on the three echoes a side can carry.
    echo_cases = [({"run": {"numerics": {"attn_fattn_wide": "wide"}}}, True, "echoed wide"),
                  ({"run": {"numerics": {"attn_fattn_wide": "fattn"}}}, False, "echoed fattn"),
                  ({"run": {}}, False, "no NUMERICS line at all (the --dry-run stand-in)")]
    for side, want, what in echo_cases:
        got = gate.side_owns_wide_form(side)
        ok = got is want
        print(f"  [{'PASS' if ok else 'FAIL'}] side_owns_wide_form({what}) -> {got}, want {want}"
              " — an unknown form must fall back to the TIGHTER line")
        if not ok:
            bad.append(f"echo {what}")
    shutil.rmtree(WORK, ignore_errors=True)
    verdict = "PASS" if not bad else "FAIL"
    reading = ("the wide-scoped band admits what it measured and refuses what is above it, and the"
               " ordinary band is unchanged" if not bad else str(bad))
    print(f"WITNESS {verdict} — {len(cases) * 2 + len(echo_cases)} cases run ({len(bad)} wrong):"
          f" {reading}")
    print("no timing taken; nothing above is a measurement and none of it may enter a timing doc.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
