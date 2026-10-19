#!/usr/bin/env python3
"""Read a kernel's note block and DERIVE blocks/CU and wf/CU from the device facts, compile-only.

WHY THIS EXISTS. `prototype/attn_wmma/fattn_ship_census.py` is the note-block parser and the spill gate for
`attn_fattn_wave`, and this script imports `note_blocks` from it rather than writing a second YAML parser (the
AGENTS.md trap "before writing a census tool, list the census tools"). What is missing from the census is the
residency arithmetic: the ladder in `prototype/attn_wmma/fattn_regfile_ladder.md` §1 books LDS against the KFD
number and calls a shape LDS-bound or register-bound, and that division has been asserted in prose from memory
of two constants. This makes it one command over a listing.

WHAT IT REFUSES TO GUESS. Three things, all recorded traps:
  * LDS per CU comes from KFD (`/sys/.../nodes/1/properties` -> `lds_size_in_kb`), NOT from
    `hipGetDeviceProperties`, which on this SKU reports against a halved SM count (AGENTS.md, "Counter,
    occupancy and census traps").
  * The thread/wave ceilings come from `rocminfo` (`Max Work-item Per CU`, `Max Waves Per CU`, `Compute Unit`),
    same reason.
  * clang's `-Rpass-analysis=kernel-resource-usage` "Occupancy [waves/SIMD]" line is a CONSTANT that prints 16
    for every kernel in the TU; it is never consulted, and neither is any `hipOccupancy*` answer here (this
    script never creates a context — it is a listing read).
  * The VGPR-file-per-CU is UNKNOWN on this part (ladder §3), so NO register-derived block count is printed. A
    shape is called register-bound only in the sense that its LDS allowance is not what binds, which is what
    §3's canary has to turn into a number.

    python3 prototype/attn_wmma/fattn_residency_read.py --asm PATH [--kernel attn_fattn_wave]
        [--threads 96] [--warps 3] [--lds-kb 64] [--work-items-per-cu 1024] [--waves-per-cu 32]

The `--lds-kb` / `--work-items-per-cu` / `--waves-per-cu` defaults are the values THIS BOX WAS MEASURED AT and
they are printed with their source; pass them explicitly from a fresh `cat`/`rocminfo` read when re-running.
Exit 1 if the kernel is not in the listing (a parser that matches nothing prints a healthy-looking nothing).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fattn_ship_census import note_blocks   # the one YAML note-block parser in the tree

NOTE_KEYS = ("vgpr_count", "vgpr_spill_count", "sgpr_count", "sgpr_spill_count",
             "private_segment_fixed_size", "group_segment_fixed_size")


def note_for(asm_text, kernel):
    """The note block of `kernel`, plus the mangled name it came from. Raises if absent or ambiguous."""
    notes = note_blocks(asm_text)
    hits = [k for k in notes if k == kernel or k.startswith(kernel + "E")]
    if len(hits) != 1:
        raise SystemExit("expected exactly one note block matching %r, found %d: %s"
                         % (kernel, len(hits), sorted(hits)))
    return hits[0], notes[hits[0]]


def report(kernel, note, threads, warps, lds_kb, work_items_per_cu, waves_per_cu):
    lds_cu = lds_kb * 1024
    lds_block = note["group_segment_fixed_size"]
    print("kernel                 : %s" % kernel)
    for k in NOTE_KEYS:
        print("  %-24s %d" % (k, note[k]))
    print("  %-24s %d" % ("block threads", threads))
    print("  %-24s %d" % ("block warps", warps))
    print("device facts (must be re-read to re-take this):")
    print("  LDS per CU             : %d B  [KFD nodes/1/properties lds_size_in_kb %d]" % (lds_cu, lds_kb))
    print("  work items per CU      : %d   [rocminfo 'Max Work-item Per CU']" % work_items_per_cu)
    print("  waves per CU           : %d   [rocminfo 'Max Waves Per CU']" % waves_per_cu)
    print("  register file per CU   : UNKNOWN on this part (fattn_regfile_ladder.md §3) — not applied")
    # Every limit that is knowable without the register file. A limit of -1 means "does not bind".
    limits = {}
    limits["LDS"] = (lds_cu // lds_block) if lds_block > 0 else None
    limits["threads"] = work_items_per_cu // threads
    limits["wavefronts"] = waves_per_cu // warps
    finite = {k: v for k, v in limits.items() if v is not None}
    blocks = min(finite.values())
    binding = sorted(k for k, v in finite.items() if v == blocks)
    print("derived blocks/CU        : %d   (%s)" % (blocks, "  ".join("%s=%s" % (k, limits[k])
                                                                      for k in sorted(limits))))
    print("  binding limit(s)       : %s" % "+".join(binding))
    print("derived wf/CU            : %d   (= blocks/CU %d x warps/block %d)" % (blocks * warps, blocks, warps))
    print("  LDS per block / budget : %d B / %d B  = %.2f of the CU's LDS"
          % (lds_block, lds_cu, lds_block / float(lds_cu)))
    return blocks, blocks * warps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", required=True)
    ap.add_argument("--kernel", default="attn_fattn_wave")
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--warps", type=int, default=3)
    ap.add_argument("--lds-kb", type=int, default=64)
    ap.add_argument("--work-items-per-cu", type=int, default=1024)
    ap.add_argument("--waves-per-cu", type=int, default=32)
    ap.add_argument("--all", action="store_true",
                    help="print one line per kernel in the listing instead of the derived residency for one")
    a = ap.parse_args()
    text = open(a.asm).read()
    if "amdhsa.kernels:" not in text:
        raise SystemExit("%s has no amdhsa.kernels: note section — wrong listing, not an empty result"
                         % a.asm)
    print("listing                : %s" % a.asm)
    if a.all:
        # The drift check: a change to one kernel in a shared TU must move ONE line of this table. Printed as
        # `name vgpr spill private group` so two listings diff with `diff`.
        #
        # `note_blocks` keys each record TWICE, by mangled name and by short name, and it can only shorten the
        # `_ZN6engine7kernels<digits>` prefix — kernels at global scope (`attn_core`, `attn_prep`,
        # `attn_combine`, ...) keep their `_Z` name in both keys. Filtering on "not starting with _Z" therefore
        # silently drops most of the TU, which is the census trap "a checker's clean bill measures the population
        # it scanned". Group by object identity instead and prefer the readable name where one exists.
        notes = note_blocks(text)
        by_id = {}
        for k, v in notes.items():
            by_id.setdefault(id(v), []).append(k)
        rows = []
        for keys in by_id.values():
            short = [k for k in keys if not k.startswith("_Z")]
            name = short[0] if short else sorted(keys)[0]
            r = notes[keys[0]]
            rows.append("  %-24s %s" % (name, "  ".join("%s=%d" % (key, r[key]) for key in NOTE_KEYS)))
        print("\n".join(sorted(rows)))
        return
    name, note = note_for(text, a.kernel)
    target = re.search(r"^\s*\.target\s+(\S+)", text, re.M)
    if target:
        print("listing target         : %s" % target.group(1))
    report(name, note, a.threads, a.warps, a.lds_kb, a.work_items_per_cu, a.waves_per_cu)


if __name__ == "__main__":
    main()
