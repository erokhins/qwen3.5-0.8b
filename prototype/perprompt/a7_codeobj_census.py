#!/usr/bin/env python3
"""Are this binary's GPU code objects embedded, or read from somewhere at first launch?

A7 candidate F3 asks whether the first launch of each engine kernel pays an image load inside
the timed wave. That question only exists if a kernel image is NOT already resident with the
process. This tool answers the residency half, from the host ELF:

  * `.hip_fatbin`        — the container holding every device image, linked into the ELF
  * `.hipFatBinSegment`  — the registration segment the CUDA/HIP runtime walks
  * `__hipRegisterFatBinary` — the static-initialiser entry point that does the registration,
                           i.e. it runs before `main`, hence before any engine timer starts
  * `external_code_object` — the marker clang emits when images are shipped as side files

So a non-zero `.hip_fatbin` with the registrar present and no external marker means: at the
first `hipLaunchKernelGGL` there is no file to read; what remains is the driver's map+patch of
an image already in memory. That is a much smaller cost, and it bounds F3 downward.

This tool deliberately does NOT count the code objects inside the container — the container is
not exposed as a section table, and the first tool I wrote for that matched `.amdgpu_fat_*`
section names, found 0 in the host ELF (those names live inside the container, not beside it),
and would have printed a confident "0 code objects, nothing embedded". Counting them needs
`llvm-objdump` over the extracted container, which is a separate item.

    python3 prototype/perprompt/a7_codeobj_census.py [ELF]
"""
import re
import shutil
import subprocess
import sys

elf = sys.argv[1] if len(sys.argv) > 1 else "engine/build/decode_run"
readelf = shutil.which("llvm-readelf") or "/opt/rocm/lib/llvm/bin/llvm-readelf"

sec = subprocess.run([readelf, "-S", elf], capture_output=True, text=True)
sym = subprocess.run([readelf, "--symbols", elf], capture_output=True, text=True)
if sec.returncode != 0:
    sys.exit(f"{readelf} failed on {elf}: {sec.stderr.strip()[:200]}")

size = None
for line in sec.stdout.splitlines():
    # Column order is Name Type Address Off Size — taking the *second* hex field (a tempting
    # off-by-one here) reports the file OFFSET as the size, which on this binary reads 90,112 B
    # instead of the true 413,624 B. Both look like plausible byte counts.
    m = re.search(r"\]\s+\.hip_fatbin\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
    if m:
        size = int(m.group(3), 16)
        print(f"  (addr={m.group(1)} off={m.group(2)} size={m.group(3)})")

has_segment = ".hipFatBinSegment" in sec.stdout
has_registrar = "__hipRegisterFatBinary" in sym.stdout
external = "external_code_object" in sec.stdout or "external_code_object" in sym.stdout

print(f"elf: {elf}")
print(f"  .hip_fatbin      : {size if size is not None else 'ABSENT'} bytes"
      + (f" ({size / 1048576:.2f} MiB)" if size else ""))
print(f"  .hipFatBinSegment: {'present' if has_segment else 'ABSENT'}")
print(f"  __hipRegisterFatBinary: {'present (registration runs before main)' if has_registrar else 'ABSENT'}")
print(f"  external_code_object  : {'PRESENT - images ship as side files' if external else 'none - nothing to read at first launch'}")

if size is None or not has_segment or not has_registrar or external:
    print("NOTE: the embedded-image picture is incomplete for this ELF; F3 stays open.")
else:
    print("VERDICT: device images are embedded and registered before main; F3 cannot be an "
          "image file read inside the timed wave.")
