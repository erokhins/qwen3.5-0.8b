#!/usr/bin/env python3
"""check_fattn_map.py — executes the WMMA fragment map claim instead of arguing it.

WHAT IT DECIDES
`prototype/attn_wmma/wmma_fattn_score.cpp` and any port of it read the C tile of
`v_wmma_f32_16x16x16_f16` as "lane t holds C[t%16][2l + t/16]" and build both operands as "lane t holds
the 16 contiguous halves of row t%16". Those sentences were transcribed from llama.cpp's `mma.cuh` and
cross-checked against each other; they had never been executed against (`fattn_score_handover.md` §5d).
This file runs the probe once per operand order, with four coordinate-encoding patterns, and decides the
claim by comparison rather than by prose. `prototype/attn_m2/check_m16_transcription.py` is the precedent
for building a witness as a separate checker instead of arguing one in a kernel header.

THE ANSWER IT PRODUCED ON FIRST RUN (and still asserts, so the finding cannot be quietly lost)
With the operands in the order the transcription uses, the claimed map is FALSE on this device: every
orientation-adjudicating pattern decodes to the exchanged law — the wmma's FIRST operand's row index
lands in the C tile's COLUMN (2l + t/16) and the SECOND operand's row index lands in its ROW (t%16).
Running the same patterns with the two operand slots exchanged makes the transcribed drain exactly right.
Those two statements are one fact, and no probe can separate "the C map is transposed" from "the operand
roles are exchanged": the only observable is which staged data lands in which fragment slot, so the law
below is what a port needs, with or without a verdict on which sentence of `mma.cuh` the transcription
misread.

    THE LAW (measured on gfx1151, 2026-10-18): with both operands staged row-major-with-run-along-k,
    lane t register l holds  sum_k  FIRST[row = 2l + t/16][k] * SECOND[row = t%16][k].
    So a score tile S[q][kv] = sum_d Q[q][d] K[kv][d] must be built with K in the FIRST operand slot and
    Q in the SECOND to land in the transcribed drain, and P*V O[d][q] must be built with P first and the
    transposed V second.

WHY THE PATTERNS ENCODE COORDINATES
The pattern §5d suggests (`Q[q][d] = q`, `K[kv][d] = kv`) gives C[q][kv] = 256*q*kv, which is symmetric
in (q, kv) — a transposed tile satisfies it identically, so that probe would pass on a transposed map.
Every pattern here instead makes each cell hold a value that names its own coordinates, so a transpose, a
parity swap, a lane permutation or a register-order error is a WRONG NUMBER, never a plausible sum.
`--selftest-host` re-derives that fact rather than asserting it: it shows a symmetric pattern is blind to
the transpose (movement count 0), which is the "empty must-catch set is not a control" trap, one level
down.

WHAT IS AND IS NOT ADJUDICATED, per pattern, stated in the source rather than inferred from a green
line: P1 and P3 adjudicate orientation (injective in (row, col), asymmetric), P2 adjudicates coverage and
accumulation only (symmetric on purpose), P4 adjudicates the port's own address spellings (`v_addr`
transposed V, P q-major) with the roles exchanged. The `get_half2` C->B relabelling is NOT witnessed
here — it moves values across lanes, so it needs the numerics admission of §5b, not this probe.

DEVICE PROTOCOL (AGENTS.md, "GPU execution is serialized")
Correctness-only, one launch per case, PASS/FAIL, no millisecond figure, and **this script does not take
`tools/gpu_lock.sh` itself** — three gate drivers in this tree self-lock, and a caller that holds the lock
and then runs one of them deadlocks it. Whoever runs this holds the lock around it and releases before
invoking a self-locking driver. Each device case runs in its own subprocess: a faulting kernel poisons
the context and every later case in the same process would silently never run.

  python3 prototype/attn_wmma/check_fattn_map.py                    # both operand orders, 2 launches
  python3 prototype/attn_wmma/check_fattn_map.py --order swapped    # one launch, the corrected order
  python3 prototype/attn_wmma/check_fattn_map.py --selftest         # + the injections that MUST fail
  python3 prototype/attn_wmma/check_fattn_map.py --expect N         # the check-count guard

No timing is taken by anything in this file.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "fattn_map_probe.cpp"
BUILD = HERE / "build_map_probe.sh"
DEFAULT_BIN = ROOT / "tmp" / "attn_wmma_map" / "fattn_map_probe"
DUMP_DIR = ROOT / "tmp" / "attn_wmma_map"
LANES, CELLS, PATTERNS = 32, 8, 4

# ---- the candidate maps: (lane t, register l) -> (row, col) of the logical tile --------------------
# Each is the sentence the hardware would have to be satisfying. `claimed` is what
# `wmma_fattn_score.cpp` and any port assume; the rest are the ways that assumption can be wrong, named so
# that a failure reports which one the machine implements rather than leaving a reader to decode a table.
MAPS = {
    "claimed-rdna3":       lambda t, l: (t % 16, 2 * l + t // 16),
    "parity-swap":         lambda t, l: (t % 16, 2 * l + (t // 16) ^ 1),
    "transposed":          lambda t, l: (2 * l + t // 16, t % 16),
    "contig8-per-halfrow": lambda t, l: (t % 16, 8 * (t // 16) + l),
    "flat-run":            lambda t, l: ((t * 8 + l) // 16, (t * 8 + l) % 16),
    "adjacent-lane-swap":  lambda t, l: ((t ^ 1) % 16, 2 * l + (t ^ 1) // 16),
}
CLAIMED = "claimed-rdna3"
# The map this device produces with the operands in the transcribed order. Recorded, not re-derived, so
# that a future run which disagrees is a FINDING printed in red rather than a silent re-fit.
MEASURED_AS_TRANSCRIBED = "transposed"

MAP_MEANING = {
    CLAIMED: "lane t holds C[t%16][2l + t/16] — the transcription in wmma_fattn_score.cpp's header",
    "transposed": "lane t holds C[2l + t/16][t%16] — the exchange: the FIRST operand's row index lands "
                  "in the C tile's column, the SECOND's in its row",
    "parity-swap": "lane t holds C[t%16][2l + (t/16)^1] — the two half-warps' column parities exchanged",
    "contig8-per-halfrow": "lane t holds C[t%16][8*(t/16) + l] — the contiguous 8-run assumption that "
                           "disqualified D58's m16_bf16frag",
    "flat-run": "lane t holds a flat 8-run of the row-major tile, lane order outer",
    "adjacent-lane-swap": "lane t holds C[(t^1)%16][2l + (t^1)/16] — an off-by-one in the lane index",
}

# ---- the patterns, with what each one is allowed to adjudicate -------------------------------------
# `orientation=True` requires the value function to be injective AND asymmetric in (row, col); the
# checker re-derives both properties from the function instead of trusting the flag, and a pattern that
# turns out blind to transposes is reported as NOT ADJUDICABLE rather than as a pass.
PATTERN_SPECS = {
    1: dict(name="P1 cell-id (16q+kv)", value=lambda r, c: 16 * r + c, orientation=True,
            axes=("q row", "kv lane"),
            why="the orientation witness: every cell names itself, row-weight on A, col-weight on B"),
    2: dict(name="P2 coverage (256(q+1)(kv+1))", value=lambda r, c: 256 * (r + 1) * (c + 1),
            orientation=False, axes=("q row", "kv lane"),
            why="symmetric by construction: witnesses that all 256 dims contribute exactly once across "
                "the 16 K-steps and that the accumulator carries, and CANNOT adjudicate orientation"),
    3: dict(name="P3 weights exchanged (16kv+q)", value=lambda r, c: 16 * c + r, orientation=True,
            axes=("q row", "kv lane"),
            why="P1 with the axis weights exchanged: an A/B asymmetry moves P1 and P3 oppositely"),
    4: dict(name="P4 port spellings (d + 16q)", value=lambda r, c: r + 16 * c, orientation=True,
            axes=("head_dim", "q row"),
            why="the P*V roles exchanged: A's row index is a head_dim (the transposed `v_addr` claim), "
                "B's row index is a query row, and the drain reads C as C[dim][row]"),
}

# The two operand orders. `expect` is the map each must produce, so the checker asserts the whole law,
# not one half of it: the transcription's order yields the exchange, the exchanged order yields the
# transcription's drain.
ORDERS = {
    "as-transcribed": dict(swap=0, expect=MEASURED_AS_TRANSCRIBED,
                           why="first operand slot = the pattern's q/dim operand, as the header reads it"),
    "swapped-operands": dict(swap=1, expect=CLAIMED,
                             why="the two operand slots exchanged: the transcribed drain then holds"),
}

# The store-side injections the binary supports, and the shape of error each one models.
DEVICE_INJECTIONS = {
    "none": "no injection — the truth",
    "lane-half-swap": "lane t's values land in lane t^16's slots (the half-warp swap)",
    "reg-reverse": "the 8 register slots are written reversed",
    "lane-rot": "every value lands one lane up",
}


class Checks:
    def __init__(self) -> None:
        self.n = 0
        self.bad = 0

    def raw(self, what: str, ok: bool, detail: str = "") -> None:
        self.n += 1
        if ok:
            print(f"  ok   {what}")
        else:
            self.bad += 1
            print(f"  FAIL {what}")
            if detail:
                print("       " + detail.replace("\n", "\n       "))

    def note(self, text: str) -> None:
        print(f"  note {text}")


# ------------------------------------------------------------------ device / dump plumbing
def build_binary(binary: Path) -> None:
    if binary.exists():
        return
    print(f"building {binary} ...")
    subprocess.run(["bash", str(BUILD)], check=True, stdout=subprocess.DEVNULL)


def run_probe(binary: Path, inject: str, swap: int, save: Path | None = None) -> dict:
    """One launch of the probe. Returns {'cells': {p: {(t,l): value}}, 'laneids': [...]}."""
    proc = subprocess.run([str(binary), f"--inject={inject}", f"--swap={swap}"],
                          capture_output=True, text=True, timeout=300)
    out = proc.stdout
    if save is not None:
        save.write_text(out + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0 or "MAPPROBE DONE" not in out:
        raise RuntimeError(f"probe(inject={inject}, swap={swap}) exit={proc.returncode}\n"
                           f"{out}\n{proc.stderr}")
    laneids: list[int] = []
    cells: dict[int, dict[tuple[int, int], float]] = {p: {} for p in PATTERN_SPECS}
    for line in out.splitlines():
        m = re.match(r"LANE (\d+) laneid=(-?\d+)(.*)$", line.strip())
        if not m:
            continue
        t, lid, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        laneids.append(lid)
        for pm in re.finditer(r"p(\d+)=([-\deEafnan +]+)", rest):
            p = int(pm.group(1))
            vals = [float(x) for x in pm.group(2).split()]
            if len(vals) != CELLS:
                raise RuntimeError(f"pattern {p} of lane {t} printed {len(vals)} values, want {CELLS}")
            for l, v in enumerate(vals):
                cells[p][(t, l)] = v
    if len(laneids) != LANES:
        raise RuntimeError(f"dump carried {len(laneids)} lanes, want {LANES}")
    return {"cells": cells, "laneids": laneids}


# ------------------------------------------------------------------ the comparator
def expected_map(spec: dict, mapping) -> dict:
    """The table a given pattern must produce, read through a given map."""
    return {(t, l): float(spec["value"](*mapping(t, l)))
            for t in range(LANES) for l in range(CELLS)}


def compare(observed: dict, spec: dict, mapping) -> list[tuple[tuple[int, int], float, float]]:
    want = expected_map(spec, mapping)
    return [(k, observed[k], want[k]) for k in sorted(observed)
            if abs(observed[k] - want[k]) > 1e-4]


def classify(observed: dict, spec: dict) -> list[str]:
    """Every candidate map the observed table is consistent with (the answer is 1 or 0, never many)."""
    return [name for name, mp in MAPS.items() if not compare(observed, spec, mp)]


def movement(spec: dict, mapping, against: str = CLAIMED) -> int:
    """How many of the 256 cells a candidate map would report differently from `against`.

    DERIVED, not hand-listed: it is a property of the two maps and the pattern's value function, so a
    candidate that moves nothing is invisible to that pattern and the checker must say so instead of
    banking a pass. This is the must-catch set, computed.
    """
    claim = expected_map(spec, MAPS[against])
    other = expected_map(spec, mapping)
    return sum(1 for k in claim if abs(claim[k] - other[k]) > 1e-4)


def decode_cells(observed: dict, spec: dict) -> dict | None:
    """Invert the pattern: value -> (row, col). None when the pattern is not injective."""
    inv: dict[float, tuple[int, int]] = {}
    for r in range(16):
        for c in range(16):
            v = float(spec["value"](r, c))
            if v in inv and inv[v] != (r, c):
                return None                      # not injective: no decoding, so no orientation claim
            inv[v] = (r, c)
    if len(inv) != 256:
        return None
    return {k: inv.get(v) for k, v in observed.items()}


# ------------------------------------------------------------------ the checks
def check_order(C: Checks, dump: dict, expect: str, order: str) -> None:
    C.raw(f"[{order}] lane ids: each lane stored its own id",
          dump["laneids"] == list(range(LANES)), f"got {dump['laneids'][:6]}...")

    # The must-catch derivation, printed before any verdict, so a reader sees what this pattern can see.
    C.note(f"[{order}] derived movement of the transcription's claim, out of 256 cells: " +
           ", ".join(f"{name}={movement(PATTERN_SPECS[1], mp)}"
                     for name, mp in MAPS.items() if name != CLAIMED))

    for p, spec in PATTERN_SPECS.items():
        obs = dump["cells"][p]
        tag = f"[{order}] {spec['name']} [{spec['axes'][0]} x {spec['axes'][1]}]"
        bad = compare(obs, spec, MAPS[expect])
        C.raw(f"{tag}: all 256 cells match '{expect}'", not bad,
              "; ".join(f"lane {t} l={l}: got {g:g} want {w:g} (which '{expect}' reads as "
                        f"row {MAPS[expect](t, l)[0]} col {MAPS[expect](t, l)[1]})"
                        for (t, l), g, w in bad[:6]) +
              (f" ... {len(bad)} cells" if len(bad) > 6 else ""))
        fits = classify(obs, spec)
        if spec["orientation"]:
            C.raw(f"{tag}: exactly one candidate map fits, and it is '{expect}'",
                  fits == [expect],
                  f"fits={fits}; {MAP_MEANING.get(expect, '')}")
            # The blind set is measured against the map this order is expected to produce: a candidate
            # that cannot move a single cell relative to THAT map is one this fixture cannot rule out.
            blind = [n for n in MAPS if n != expect and movement(spec, MAPS[n], against=expect) == 0]
            C.raw(f"{tag}: no candidate other than the reference is invisible to this pattern",
                  not blind, f"invisible: {blind}")
            if expect != CLAIMED:
                # The falsification stays asserted. A checker that only tests the corrected map would
                # pass silently if the device ever changed, or if someone re-typed the expectation.
                n_claim = len(compare(obs, spec, MAPS[CLAIMED]))
                C.raw(f"{tag}: the transcribed '{CLAIMED}' map does NOT fit this order "
                      f"({n_claim} of 256 cells disagree)", n_claim > 0, "")
        else:
            C.note(f"{tag}: NOT ADJUDICABLE for orientation by design — {spec['why']}")


def selftest_host(C: Checks) -> None:
    """Inject a wrong map HOST-side: rebuild the expectation from the wrong formula and require failure.

    This is the witness that the comparator discriminates. The must-catch side is derived per candidate
    (movement > 0), and the anti-vacuity case is checked too: a symmetric pattern is shown to be blind to
    the transpose, which is what the real patterns were built to avoid.
    """
    spec = PATTERN_SPECS[1]
    for name, mp in MAPS.items():
        synth = {(t, l): float(spec["value"](*mp(t, l)))
                 for t in range(LANES) for l in range(CELLS)}
        fits = classify(synth, spec)
        bad = compare(synth, spec, MAPS[CLAIMED])
        if name == CLAIMED:
            C.raw("selftest: a table built from the claimed map PASSES the claimed comparison",
                  not bad and fits == [CLAIMED], f"fits={fits}")
        else:
            C.raw(f"selftest: a table built from the WRONG map '{name}' FAILS the claimed comparison "
                  f"({movement(spec, mp)} of 256 cells move) and is classified as '{name}'",
                  bool(bad) and fits == [name], f"mismatches={len(bad)}, fits={fits}")

    symmetric = dict(name="symmetric", value=lambda r, c: 256 * r * c, orientation=False,
                     axes=("q", "kv"), why="")
    mv = movement(symmetric, MAPS["transposed"])
    C.raw("anti-vacuity: the symmetric pattern §5d suggested (C = 256*q*kv) cannot see a transpose at "
          "all, which is why the shipped probe encodes coordinates instead",
          mv == 0, f"movement for 'transposed' under the symmetric pattern = {mv}")
    asym_mv = movement(spec, MAPS["transposed"])
    C.raw(f"and the coordinate-encoding pattern does see it (movement {asym_mv})", asym_mv > 0,
          f"movement = {asym_mv}")
    # The classifier must be able to name the answer this device gives, or the headline is unverifiable.
    law = {(t, l): float(spec["value"](*MAPS[MEASURED_AS_TRANSCRIBED](t, l)))
           for t in range(LANES) for l in range(CELLS)}
    C.raw(f"selftest: the recorded answer '{MEASURED_AS_TRANSCRIBED}' is one the classifier can name",
          classify(law, spec) == [MEASURED_AS_TRANSCRIBED], f"fits={classify(law, spec)}")


def selftest_device(C: Checks, binary: Path) -> None:
    """Re-inject on the device and require the whole pipeline to go red, in its own process per case."""
    for inj, why in DEVICE_INJECTIONS.items():
        if inj == "none":
            continue
        try:
            dump = run_probe(binary, inj, ORDERS["swapped-operands"]["swap"])
        except Exception as exc:                                   # noqa: BLE001
            C.raw(f"injection {inj}: probe did not produce a dump", False, str(exc))
            continue
        spec = PATTERN_SPECS[1]
        obs = dump["cells"][1]
        bad = compare(obs, spec, MAPS[CLAIMED])
        fits = classify(obs, spec)
        C.raw(f"injection {inj} ({why}) is caught: {len(bad)} cells disagree with the claimed map and "
              f"it is NOT classified as '{CLAIMED}'",
              bool(bad) and CLAIMED not in fits, f"mismatches={len(bad)}, fits={fits}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", default=str(DEFAULT_BIN))
    ap.add_argument("--order", choices=("both",) + tuple(ORDERS), default="both")
    ap.add_argument("--selftest", action="store_true", help="host + device injections must all fail")
    ap.add_argument("--selftest-host", action="store_true")
    ap.add_argument("--selftest-dev", action="store_true")
    ap.add_argument("--expect", type=int, default=0,
                    help="assert this many checks ran: a check list that shrinks silently cannot fail "
                         "(prototype/gdn_collapse/run.py)")
    args = ap.parse_args()
    t0 = time.time()
    C = Checks()

    if args.selftest_host or args.selftest:
        selftest_host(C)

    orders = list(ORDERS) if args.order == "both" else [args.order]
    device_work = args.selftest_dev or args.selftest or not args.selftest_host
    if device_work:
        binary = Path(args.binary)
        build_binary(binary)
        if args.selftest_dev or args.selftest:
            selftest_device(C, binary)
        if not (args.selftest_dev or args.selftest):
            for order in orders:
                spec = ORDERS[order]
                dump_path = DUMP_DIR / f"dump-{order}.txt"
                DUMP_DIR.mkdir(parents=True, exist_ok=True)
                print(f"one launch: {binary} --swap={spec['swap']} "
                      f"(dump kept at {dump_path.relative_to(ROOT)}) — {spec['why']}")
                check_order(C, run_probe(binary, "none", spec["swap"], save=dump_path),
                            spec["expect"], order)
            if len(orders) == 2:
                C.raw("the two operand orders disagree with each other, so the answer is about the "
                      "operand slots and not about the fixture",
                      ORDERS["as-transcribed"]["expect"] != ORDERS["swapped-operands"]["expect"], "")

    print(f"\n{'FAIL' if C.bad else 'PASS'}: check_fattn_map — {C.n - C.bad}/{C.n} checks; "
          f"driver wall {int(time.time() - t0)} s (process bookkeeping); no timing taken")
    if args.expect and C.n != args.expect:
        print(f"FAIL: expected {args.expect} checks, ran {C.n} — a check list that can be empty cannot "
              f"fail")
        return 1
    return 1 if C.bad else 0


if __name__ == "__main__":
    sys.exit(main())
