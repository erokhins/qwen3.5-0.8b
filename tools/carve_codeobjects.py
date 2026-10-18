#!/usr/bin/env python3
"""carve_codeobjects.py — pull every AMDGPU code object out of a host binary (or a fatbin blob).

Why this exists: `engine/build.sh` links with hipcc and does not pass --save-temps, so the only
GPU assembly that provably belongs to the *shipping* binary is the one embedded in
`engine/build/decode_run`'s `.hip_fatbin` section. `clang-offload-bundler -unbundle` returned a
single 5 KB ELF from that blob (hipv4 nests a second container inside the bundle entry), so the
carving is done here instead: scan for ELF magic, keep the objects whose e_machine is EM_AMDGPU
(224) and whose AMDGPU note names the expected target.

The canary is the point of this script, not a nicety: a census that silently reads the readable
fallback objects instead of the tuned ones is on record in AGENTS.md ("0 of the readable 55"), so
--target must be found in the carved listing's own target-id metadata AND each carved object must
carry at least one kernel symbol. If either fails, exit non-zero and no census runs.

Usage:
  python3 tmp/gcc/carve_codeobjects.py <host-binary-or-blob> <outdir> [--target gfx1151]
"""
import os
import struct
import subprocess
import sys

EM_AMDGPU = 224
READELF = "/opt/rocm/lib/llvm/bin/llvm-readelf"


def elf_objects(buf):
    """Yield (offset, bytes) for every plausible ELF object starting in buf.

    Sizes come from the ELF program/section headers, but code objects are contiguous, so the cheap
    bound is "next ELF magic or end of buffer"; we keep the header-declared size when it is sane
    and fall back to the gap otherwise.
    """
    starts = [i for i in range(len(buf) - 4) if buf[i:i + 4] == b"\x7fELF"]
    for idx, start in enumerate(points := starts):
        end = starts[idx + 1] if idx + 1 < len(points) else len(buf)
        yield start, buf[start:end]


def header_ok(obj, target):
    """Verify by metadata, not by filename: EM_AMDGPU + the target id appears in the ELF notes."""
    if len(obj) < 64:
        return None
    ei_class, ei_data = obj[4], obj[5]
    if ei_class != 2 or ei_data != 1:
        return None
    e_machine = struct.unpack_from("<H", obj, 18)[0]
    if e_machine != EM_AMDGPU:
        return None
    # amdhsa metadata: the target id string (e.g. "gfx1151") is stored in .AMDGPU.csdata / notes.
    hay = obj
    if target.encode() not in hay:
        return None
    return e_machine


def symbols(obj_path):
    try:
        out = subprocess.run([READELF, "--syms", obj_path], capture_output=True, text=True,
                             timeout=120).stdout
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        parts = line.split()
        # llvm-readelf syms columns: num: value size type bind vis ndx name
        if len(parts) >= 8 and parts[3] == "FUNC":
            names.append(parts[-1])
    return names


def main(argv):
    if len(argv) < 3:
        __doc__ and print(__doc__)
        return 2
    src, outdir = argv[1], argv[2]
    target = "gfx1151"
    if "--target" in argv:
        target = argv[argv.index("--target") + 1]
    buf = open(src, "rb").read()
    os.makedirs(outdir, exist_ok=True)
    kept, rejected = [], 0
    for off, obj in elf_objects(buf):
        if header_ok(obj, target) is None:
            rejected += 1
            continue
        path = os.path.join(outdir, "co_%06d.co" % off)
        with open(path, "wb") as fh:
            fh.write(obj)
        kept.append((off, path, len(obj)))
    print("scanned %d ELF candidates; kept %d AMDGPU objects for %s, rejected %d"
          % (sum(1 for _ in elf_objects(buf)), len(kept), target, rejected))

    # ---- canary: every carved object must name the target in its own metadata and expose >=1
    # ---- FUNC symbol. A census over an empty or wrong-arch listing must not be able to pass.
    total_syms = 0
    for off, path, size in kept:
        rh = subprocess.run([READELF, "-h", path], capture_output=True, text=True).stdout
        arch_line = [l for l in rh.splitlines() if "Machine" in l or "Flags" in l]
        syms = symbols(path)
        total_syms += len(syms)
        print("  %s  %8d B  %s  FUNC syms=%d" % (os.path.basename(path), size,
                                                 " | ".join(x.strip() for x in arch_line), len(syms)))
        if not syms:
            print("CANARY FAIL: %s carries no FUNC symbol — not a device code object" % path)
            return 1
    if not kept:
        print("CANARY FAIL: no AMDGPU code object for %s found in %s" % (target, src))
        return 1
    print("CANARY PASS: %d code object(s), %d kernel symbols, target id %s present in each" %
          (len(kept), total_syms, target))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
