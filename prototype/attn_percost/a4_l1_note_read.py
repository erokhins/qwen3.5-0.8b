#!/usr/bin/env python3
"""Read a kernel's NOTE BLOCK out of a BUILT host binary — the stale-binary canary's instrument.

WHY THIS EXISTS. `engine/build.sh` is not bit-reproducible on this box, so `cmp` between two builds proves
they are two builds and nothing about which code either carries (AGENTS.md, 2026-10-17: "a paired A/B against
a stale binary reads as a clean null"). Any self-consistency check compiled INTO the binary under test —
`plan_match`, a plan bill, a grid echo — agrees with itself and is blind to exactly that failure. The one
reading that survives is a fact about the machine code, and for a register/residency claim that fact is the
AMDHSA note block of the code object embedded in the executable that will actually run.

`prototype/attn_wmma/fattn_ship_census.py` already parses note blocks, but only out of a `--save-temps`
listing — a compile of the tree, not the linked binary. This reads the same YAML out of the linked binary by
carving its code objects with `tools/carve_codeobjects.py` and `llvm-readelf -n`-ing each one, and it reuses
that census's `note_blocks()` rather than writing a second parser (AGENTS.md, "before writing a census tool,
list the census tools").

    python3 prototype/attn_percost/a4_l1_note_read.py BINARY [--kernel NAME]... [--keep DIR]

With no `--kernel` it prints every kernel it found. `NAME` matches the short name (`attn_fattn_wave`), the
mangled one (`_ZN6engine7kernels15attn_fattn_waveE8CoreArgs`), or any unambiguous substring of either — and a
name that matches zero kernels or two of them is a REFUSAL, because a canary whose input silently failed to
arrive is the failure class this tree keeps meeting (a parser that matches nothing prints a healthy-looking
nothing, then the verdict computed from it reads like a measurement).

Exit 0 = every requested kernel was found exactly once. Nothing here touches a device: carving and readelf are
file reads.
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
CARVE = REPO / "tools" / "carve_codeobjects.py"
READELF = "/opt/rocm/llvm/bin/llvm-readelf"
NOTE_KEYS = ("vgpr_count", "vgpr_spill_count", "sgpr_count", "sgpr_spill_count",
             "private_segment_fixed_size", "group_segment_fixed_size")

sys.path.insert(0, str(REPO / "prototype" / "attn_wmma"))
from fattn_ship_census import note_blocks      # the one note-block YAML parser in the tree


def short_name(mangled: str) -> str:
    """`_ZN6engine7kernels15attn_fattn_waveE8CoreArgs` -> `attn_fattn_wave`; a short name passes through.

    Same derivation `fattn_ship_census.note_blocks` uses for its alias keys, kept here so the CLI can print
    and accept one spelling of a kernel name without importing a second copy of the rule.
    """
    return re.sub(r"^_ZN6engine7kernels\d+", "", mangled).split("E")[0]


def carve(binary: pathlib.Path, outdir: pathlib.Path) -> list:
    """Carve the AMDGPU code objects out of a host binary; the carver's own canary must pass."""
    proc = subprocess.run([sys.executable, str(CARVE), str(binary), str(outdir)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"REFUSE: carve_codeobjects.py failed on {binary} (rc={proc.returncode}): "
                         + (proc.stderr or proc.stdout).strip()[-300:])
    objs = sorted(outdir.glob("*.co"))
    if not objs:
        raise SystemExit(f"REFUSE: no code objects carved out of {binary}. Nothing was read, so nothing "
                         "here is a fact about this binary.")
    return objs


def read_notes(objs: list) -> dict:
    """{kernel name -> note dict}, merged across the carved objects, refusing on a disagreement."""
    merged, where = {}, {}
    for obj in objs:
        proc = subprocess.run([READELF, "-n", str(obj)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"REFUSE: llvm-readelf -n failed on {obj} (rc={proc.returncode}): "
                             + proc.stderr.strip()[-200:])
        for name, rec in note_blocks(proc.stdout).items():
            if name in merged and merged[name] != rec:
                raise SystemExit(f"REFUSE: kernel {name} appears in two carved objects with DIFFERENT note "
                                 f"blocks ({where[name]}: {merged[name]} vs {obj.name}: {rec}). Two code "
                                 "objects in one binary disagreeing about one kernel is a build problem, not "
                                 "a reading, and picking either silently is worse.")
            merged[name] = rec
            where.setdefault(name, obj.name)
    if not merged:
        raise SystemExit("REFUSE: carved objects yielded ZERO note blocks. The parser matched nothing, which "
                         "is the empty-parse failure class, not an absence of kernels.")
    return merged


def resolve(spec: str, notes: dict) -> list:
    """The kernels `spec` names: exact short/mangled name first, then substring. Zero or many is a refusal.

    `note_blocks` keys every engine kernel twice — once by the mangled name the object carries, once by the
    short name derived from it — so candidates are de-duplicated by identity before anything is counted and
    the mangled key is the one reported (printing the alias would send a reader hunting for a kernel that
    does not exist). An ambiguous substring is refused, not first-taken: `attn_fattn_wave` is a substring of
    BOTH `attn_fattn_wave` and `attn_fattn_wave_two`, and a canary that read whichever the dict happened to
    yield first would certify one kernel and report it as the other — the two-sides-are-the-same null.
    """
    by_ident = {}                      # short name -> the keys that mean it
    for k in notes:
        by_ident.setdefault(short_name(k), set()).add(k)

    def is_exact(ident_keys, s):
        return any(k == s or short_name(k) == s for k in ident_keys)

    exact = sorted(i for i, keys in by_ident.items() if is_exact(keys, spec))
    if exact:
        return exact
    many = sorted(i for i, keys in by_ident.items() if any(spec in k for k in keys))
    if len(many) != 1:
        raise SystemExit(f"REFUSE: --kernel {spec!r} matched {len(many)} kernels in the binary"
                         + (": " + ", ".join(many) if many else "")
                         + ". Name one exactly (short name or mangled); a canary that guesses which kernel"
                         " it read cannot tell the two sides of an A/B apart.")
    return many


def key_for(ident: str, notes: dict) -> str:
    """The key to report for one identity: the mangled name when the binary carries one."""
    keys = [k for k in notes if short_name(k) == ident]
    return sorted(keys, key=lambda k: (not k.startswith("_Z"), k))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=pathlib.Path)
    ap.add_argument("--kernel", action="append", default=[],
                    help="kernel to report; repeatable. Default: everything found.")
    ap.add_argument("--keep", type=pathlib.Path, default=None,
                    help="carve into this directory instead of a temporary one (for post-mortem)")
    a = ap.parse_args()
    if not a.binary.exists():
        raise SystemExit(f"REFUSE: no such binary: {a.binary}")
    tmp = tempfile.TemporaryDirectory(prefix="a4l1note_")
    outdir = a.keep or pathlib.Path(tmp.name)
    outdir.mkdir(parents=True, exist_ok=True)
    objs = carve(a.binary, outdir)
    notes = read_notes(objs)
    specs = a.kernel or sorted({short_name(k) for k in notes})
    rows = []
    for spec in specs:
        for ident in resolve(spec, notes):
            key = key_for(ident, notes)
            rows.append((ident, key, notes[key]))

    for short, key, rec in rows:
        print(f"KERNEL {short} mangled={key} " +
              " ".join(f"{k}={rec[k]}" for k in NOTE_KEYS) +
              " note_found=1")
    print(f"SUMMARY binary={a.binary} objects={len(objs)} note_blocks={len(notes)} "
          f"requested={len(specs)} resolved={len(rows)}")
    if not rows:
        raise SystemExit("REFUSE: zero kernels resolved. A canary that read nothing cannot pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
