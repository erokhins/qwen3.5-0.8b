#!/usr/bin/env python
"""Per-kernel device-ISA fingerprint of a generated `.s`, to answer one question: did an edit move a kernel?

Why not `cmp` of two binaries: `engine/build.sh` is not bit-reproducible on this box, so a binary cmp proves
only that two builds are two builds (AGENTS' stale-binary row). Why not a whole-file diff: the TU carries a
content-hash symbol (`__hip_cuid_<hash>`) that changes when a COMMENT changes, so a file diff over-reports.
This hashes the instruction run of each device kernel and nothing else, so a comment edit or a host-side edit
reports `0 changed` while a kernel edit names the kernel that moved.

Three traps this is written around, all of them hit while writing it:
  * clang emits a kernel body under a BARE column-0 label with a trailing comment
    (`_Z9attn_core8CoreArgs:                  ; @_Z9attn_core8CoreArgs`), not under `.globl`, so a
    `.globl`-based slicer finds a stub (AGENTS, `prototype/gdn_walk/BET.md` §2).
  * the SAME `.s` carries the HOST translation unit too, and the host stubs carry the SAME symbol names as
    the kernels. Their blocks contain x86 (`movq %rax, 144(%rsp)`, `.LBB16_162`) and shift with any host-side
    edit, so a slicer that does not require `.amdhsa_kernel` in the block reports a device change when only
    the launcher moved. That is not hypothetical: the first version of this file did exactly that and called a
    comment-only edit a kernel change.
  * an instruction run is contiguous: the AMDGPU note block (`.amdhsa_kernel`, `.vgpr_count`, ...) comes
    AFTER the body, so collection stops at the first directive once instructions have started. Hashing
    everything between two labels instead would swallow whatever follows, including host code.

Usage:
  a12_isa_fingerprint.py BEFORE.s AFTER.s          # per-kernel hashes for both, then the diff; exit 1 if any moved
  a12_isa_fingerprint.py --dump FILE KERNEL        # the exact lines hashed for one kernel
  a12_isa_fingerprint.py --selftest FILE           # perturb one operand, require exactly one kernel to move
  a12_isa_fingerprint.py FILE                      # hashes only (one build, for eyeballing)
Regenerate a side with:
  hipcc -std=c++17 -O3 -w -S --offload-arch=gfx1151 -I engine -I . engine/kernels_attn.cpp -o out.s
"""
import hashlib
import re
import sys

KERNEL_FIELD = ".amdhsa_kernel"
LABEL_RE = re.compile(r"^(_Z[A-Za-z0-9_]+):\s*(?:;.*)?$")
MIN_INSTR = 32          # every kernel in this tree is far above it; below it means the slicer slipped
INSTR_RE = re.compile(r"^\t[a-z][a-z0-9_.]*\b")


def blocks(lines):
    """-> (name, instruction lines) for every device kernel label.

    The terminator is clang's own `; -- End function`, not "the next thing that is not an instruction": an
    earlier version stopped at the first non-instruction line and silently returned 13 instructions for a
    13,000-instruction kernel — a fingerprint that cannot see the body cannot fail, which is the same defect
    class as a parser that matches zero records and prints PASS. The host stubs under the same symbol are
    dropped by the `.amdhsa_kernel` test, and the AMDGPU note block after the body contributes nothing
    because every line of it starts with a dot.
    """
    starts = [i for i, line in enumerate(lines) if LABEL_RE.match(line)]
    for start in starts:
        end = start + 1
        while end < len(lines) and "-- End function" not in lines[end]:
            end += 1
        window = lines[start + 1:end]
        if not any(KERNEL_FIELD in line for line in window):
            continue                       # the host stub under the same symbol, or a data symbol
        yield (lines[start].split(":")[0],
               [line.rstrip() for line in window if INSTR_RE.match(line)])


def fingerprint(path):
    """Per-kernel hash, with the parse itself under test.

    A silently empty or half-empty parse is the failure mode that prints a confident wrong answer (AGENTS' gate
    drivers that parsed zero records and reported PASS), so a table with no kernels, or a kernel body under
    `MIN_INSTR` instructions, is an error here rather than a number. `--selftest` proves the count moves when
    one instruction operand moves.
    """
    table = {}
    for name, body in blocks(open(path).read().splitlines()):
        if len(body) < MIN_INSTR:
            raise SystemExit(f"{path}: parsed only {len(body)} instructions for {name}; the slicer is "
                             f"broken, not the kernel. Refusing to report a fingerprint.")
        table[name] = hashlib.md5("\n".join(body).encode()).hexdigest()[:16]
    if not table:
        raise SystemExit(f"{path}: no device kernels found (no `.amdhsa_kernel` block under a bare label?); "
                         "refusing to report an empty fingerprint.")
    return table


def selftest(path):
    """Prove the slicer can fail, which is the only way its `0 changed` verdict means anything.

    Mutates one instruction OPERAND in memory — the first scalar move of the first kernel — and requires
    exactly one kernel's fingerprint to move. A slicer that hashed the whole file, or nothing, would report
    either 15 of 15 or 0 of 15 here and be caught. Witnessed on this build: `1 of 15`, and the named kernel is
    the one the mutation sits in.
    """
    lines = open(path).read().splitlines()
    clean = {name: hashlib.md5("\n".join(body).encode()).hexdigest()[:16]
             for name, body in blocks(lines)}
    target, moved = None, None
    for start in (i for i, line in enumerate(lines) if LABEL_RE.match(line)):
        end = start + 1
        while end < len(lines) and "-- End function" not in lines[end]:
            end += 1
        for idx in range(start + 1, end):
            if re.match(r"^\ts_mov_b32\b", lines[idx]):
                target, moved = (idx, lines[idx].replace(", s5", ", s6"))
                break
        if target is not None:
            break
    if target is None:
        print("SELFTEST SKIP: no `s_mov_b32 ..., s5` instruction to perturb in this file")
        return 0
    hot = list(lines)
    hot[target] = moved
    after = {}
    for name, body in blocks(hot):
        after[name] = hashlib.md5("\n".join(body).encode()).hexdigest()[:16]
    diff = [n for n in clean if clean[n] != after.get(n)]
    print(f"SELFTEST {'PASS' if len(diff) == 1 else 'FAIL'}: one operand perturbed -> "
          f"{len(diff)} of {len(clean)} kernels changed ({', '.join(n.split('E8')[0] for n in diff)})")
    return 0 if len(diff) == 1 else 1


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "--selftest":
        return selftest(argv[2])
    if argv[1] == "--dump":
        want = argv[3]
        for name, body in blocks(open(argv[2]).read().splitlines()):
            if want in name:
                print("\n".join(body))
                return 0
        print(f"{argv[2]}: no device kernel matching {want!r}", file=sys.stderr)
        return 2
    tables = [fingerprint(p) for p in argv[1:]]
    for path, table in zip(argv[1:], tables):
        print(f"{path}: {len(table)} device kernels")
        for name in sorted(table):
            print(f"  {table[name]}  {name}")
    if len(tables) != 2:
        return 0
    old, new = tables
    print("=== diff ===")
    moved = 0
    for name in sorted(set(old) | set(new)):
        if name not in old:
            print(f"ADDED   {name}")
            moved += 1
        elif name not in new:
            print(f"REMOVED {name}")
            moved += 1
        elif old[name] != new[name]:
            print(f"CHANGED {name}")
            moved += 1
    print(f"changed device kernels: {moved} of {len(set(old) | set(new))}")
    return 1 if moved else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
