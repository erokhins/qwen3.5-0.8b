#!/usr/bin/env python3
"""Print the declared LDS of the two kernels `time_wide` launches, read from the CODE OBJECT.

One job, so `run_a12_window.sh` can assert that the binary it is about to time declares what the listing of the
same build says it declares — the canary a stale `new` side slips past (AGENTS.md, 2026-10-17: every
self-consistency field compiled into the binary under test agrees with itself, and `engine/build.sh` is not
bit-reproducible here, so the only witness is the code object produced by the same invocation).

The field is `.amdhsa_group_segment_fixed_size` in the `.amdhsa_kernel` metadata block — the same field
`prototype/attn_percost/a11_read_vgpr.py` and `prototype/attn_wide/wide_regread.py` read, and deliberately not
the LLVM note block's `.group_segment_fixed_size`/`.vgpr_count` pair (AGENTS.md, the note-block row: the note
entries are not in definition order). Exit code 1 if either kernel is missing: a table that matches nothing is
the failure class this repo keeps meeting, and here a silent miss would let a stale binary through the window.

Usage: listing_lds.py <kernel.s>     ->  "<wide_lds> <fattn_lds>" on stdout
"""
import re
import sys

WANT = ("attn_wide_wave", "attn_fattn_wave_u8")

if len(sys.argv) != 2:
    print(__doc__, file=sys.stderr)
    sys.exit(2)

text = open(sys.argv[1], errors="replace").read()
found = {name: None for name in WANT}
for mangled, body in re.findall(r"\.amdhsa_kernel\s+(\S+)(.*?)\.end_amdhsa_kernel", text, re.S):
    for name in WANT:
        # `attn_fattn_wave_u8` is a substring of `attn_fattn_wave_u8_pvf16` and of `attn_fattn_wave_two_u8...`, so
        # a bare substring test would report a different spelling's LDS — same 20,480 B today, which is exactly
        # how a match-by-accident passes until the day the spellings differ. The boundaries that actually work on
        # a mangled name: the name is preceded by its mangled LENGTH prefix, i.e. a digit (so the lookbehind may
        # not exclude digits: `_Z15attn_wide_wave8WideArgs`), and every variant suffixes with `_`, so refusing a
        # following `_` is what separates the plain kernel from its `_pvf16` / `_two_` siblings.
        if re.search(r"(?<![A-Za-z])" + re.escape(name) + r"(?!_)", mangled):
            m = re.search(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", body)
            found[name] = int(m.group(1)) if m else -1

missing = [n for n, v in found.items() if v is None or v < 0]
if missing:
    print("REFUSAL: %s has no readable .amdhsa_group_segment_fixed_size for %s"
          % (sys.argv[1], ", ".join(missing)), file=sys.stderr)
    sys.exit(1)
print(" ".join(str(found[n]) for n in WANT))
