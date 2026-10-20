#!/usr/bin/env python3
"""Derive and PIN the D-block map of `v_wmma_f16_16x16x16_f16` (w32) from the probe's raw dump.

`a10_pvf16_map_probe.cpp` runs one warp with operands whose every output cell holds `d + 32q` (its own
coordinates), so the map is READ OFF the dump rather than transcribed from `mma.cuh` — the same rule
`prototype/attn_wmma/check_fattn_map.py` applies to the F32 accumulator, and for the same reason: the first
transcription of the F32 map was wrong and produced a transposed score.

The claim this checks, in the engine's words (row = query row = `fattn_c_row`, dim = output dim within the
32-dim pair):

    lane t, element e:  q(t,e) = t % 16                                one row per lane, as in the F32 tile
                      d(t, 2l) = 2l + t/16        <- the op_sel-0 A fragment (dims  0..15 of the pair)
                      d(t, 2l+1) = 16 + 2l + t/16 <- the op_sel-1 A fragment (dims 16..31)

i.e. the F16 D block is the F32 C tile's law applied to each op_sel half, with the two halves interleaved
BY HALF OF EACH VGPR (which is what rocwmma's `AccumBits LOW = bits[15:0] / HIGH = bits[31:16]` says, and
what `--opsel-high-first` cross-checks: swapping which A fragment gets op_sel 0 must swap the two dim
halves and move nothing else).

Run it on the probe's stdout (`--dump -` reads stdin):

    tools/gpu_lock.sh acquire pvf16-map
    tmp/pvf16_map/a10_pvf16_map_probe > tmp/pvf16_map/lowfirst.txt
    tmp/pvf16_map/a10_pvf16_map_probe --opsel-high-first > tmp/pvf16_map/highfirst.txt
    tools/gpu_lock.sh release
    python3 prototype/attn_percost/check_a10_pvf16_map.py --dump tmp/pvf16_map/lowfirst.txt \
             --swapped tmp/pvf16_map/highfirst.txt

Correctness only. No timing, no ms, nothing about performance.
"""
from __future__ import annotations

import argparse
import re
import sys


def parse(text):
    """`LANE t` + 16 raw F16 bit patterns -> per-lane list of (q, d) decoded from `d + 32q`."""
    lanes = {}
    for line in text.splitlines():
        m = re.match(r"^LANE (\d+)((?: \d+)+)$", line.strip())
        if not m:
            continue
        bits = [int(x) for x in m.group(2).split()]
        lanes[int(m.group(1))] = bits
    return lanes


def half_to_int(bits):
    """Exact for the probe's integer patterns: decode an IEEE-754 binary16 as an int.

    `value = (-1)^s * (1 + m/1024) * 2^(e-15)` for a normal number. A self-check runs first, because a
    decoder that silently mis-decodes would report a wrong MAP instead of a broken checker (the same reason
    `bf16_ulp.py` carries a selftest): 1.0 must read 1 and 1.5 must read 1 at the integer step."""
    s, e, m = (bits >> 15) & 1, (bits >> 10) & 0x1F, bits & 0x3FF
    if e == 0:
        return 0 if m == 0 else None            # subnormal: not in this probe's vocabulary
    if e == 0x1F:
        return None                             # inf/NaN: not in this probe's vocabulary
    val = (1.0 + m / 1024.0) * (2.0 ** (e - 15))
    iv = int(round(val))
    if abs(iv - val) > 1e-9:
        return None                             # not an integer: outside the pattern vocabulary
    return -iv if s else iv


assert (half_to_int(0x3C00), half_to_int(0x4000), half_to_int(0x5000), half_to_int(0x3C01),
        half_to_int(0x4E80)) == (1, 2, 32, None, 26), \
    "the F16 decoder itself is broken — every number below would be meaningless"


def decode(bits):
    """[(q, d) per element]; None where the value is not an integer coordinate pair."""
    out = []
    for b in bits:
        v = half_to_int(b)
        out.append(None if v is None else (v // 32, v % 32))
    return out


def law(t, e):
    """The claim under test: element 2l is op_sel-0's column 2l + t/16, element 2l+1 is op_sel-1's +16."""
    l, hi = e // 2, e % 2
    return (t % 16, 16 * hi + 2 * l + (t // 16))


def check(lanes, label, expect=law):
    bad, unseen = [], []
    for t, bits in sorted(lanes.items()):
        got = decode(bits)
        for e, cell in enumerate(got):
            if cell is None:
                unseen.append((t, e))
                continue
            if cell != expect(t, e):
                bad.append((t, e, cell, expect(t, e)))
    n = sum(len(v) for v in lanes.values())
    print(f"[{label}] {len(lanes)} lanes x {n} cells: {len(bad)} disagree with "
          f"q=t%16, d=16*(e&1) + 2*(e>>1) + t/16; {len(unseen)} not an integer coordinate")
    for t, e, g, w in bad[:12]:
        print(f"    lane {t:2d} elem {e:2d}: got q={g[0]} d={g[1]}, expected q={w[0]} d={w[1]}")
    if unseen:
        print(f"    first non-integer cells: {unseen[:8]}")
    return not bad and not unseen and len(lanes) == 32


def show(lanes, n=4):
    """Print the DERIVED map for the first `n` lanes: what (q, dim) each element of the D block holds."""
    for t in sorted(lanes)[:n]:
        cells = decode(lanes[t])
        print("    lane %2d: " % t + " ".join("e%02d=(q%2d,d%02d)" % (e, q, d) for e, (q, d) in enumerate(cells)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="-")
    ap.add_argument("--swapped", default=None,
                    help="the --opsel-high-first dump; its map must be the SAME law, which is what "
                         "witnesses that the two op_sel halves are independent tiles")
    ap.add_argument("--show", type=int, default=0, help="print the derived (q, dim) map of the first N lanes")
    a = ap.parse_args()
    text = sys.stdin.read() if a.dump == "-" else open(a.dump).read()
    if "PVF16MAP" not in text:
        raise SystemExit("REFUSE: no PVF16MAP lines in the dump — the probe did not run (a check whose "
                         "input list is empty cannot fail)")
    ok = check(parse(text), "op_sel low-first")
    if a.show:
        print("  derived map (row q, dim within the 32-dim pair):")
        show(parse(text), a.show)
    if a.swapped:
        st = open(a.swapped).read() if a.swapped != "-" else sys.stdin.read()
        if "PVF16MAP" not in st:
            raise SystemExit("REFUSE: --swapped file carries no PVF16MAP lines")
        # With the A fragments exchanged between op_sel 0 and 1, dims 0..15 and 16..31 swap places: the
        # law's `16*(e&1)` term flips sign within the pair. That is the mirror, not a new map.
        def mirror(t, e):
            q, d = law(t, e)
            return (q, d + 16 if d < 16 else d - 16)
        ok = check(parse(st), "op_sel high-first (A fragments exchanged)", mirror) and ok
    print("PVF16 MAP PASS" if ok else "PVF16 MAP FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
