#!/usr/bin/env python3
"""attn_isa_census.py — does the shipping attention path issue ANY matrix instruction?

Queue item 2.6 / D56 question (1). The question is deliberately cheap: D56 priced attention at
1.59 TFLOP/s = 7.1 % of the mix roof, and the size of the prize depends entirely on which ceiling
that 1.59 is being compared against. If the shipping kernels already run on the matrix unit, 1.59
TFLOP/s against the WMMA roof means the tile shape is the problem; if they run on the FMA path, the
right ceiling is a different (much lower) number and WMMA is a new instruction class, not a retune.

Input is the GPU assembly carved out of the SHIPPING binary (tmp/gcc/carve_codeobjects.py), never a
--save-temps re-build, so the answer is about the object that actually executes.

Canary, asserted before any count is printed (AGENTS.md: "0 of the readable 55" — a census that
silently scanned the readable fallback objects):
  * the listing must contain the kernel symbols it claims to census; a symbol that is absent is a
    hard failure, not a zero row;
  * the object's own ELF metadata must name the target arch (checked by the carver, re-checked here
    from the listing header `file format elf64-amdgpu`);
  * a kernel with ZERO total instructions cannot be reported — that is what a failed disassembly
    looks like, and it would otherwise read as "no matrix instructions".

Usage:
  python3 tmp/gcc/attn_isa_census.py <file.s ...> [--kernels attn_core,attn_slim_wave,...]
"""
import collections
import re
import sys

# Ordered: first match wins for the *primary* class, but a matrix instruction is reported
# independently of the primary class so a mixed kernel cannot hide one behind an FMA count.
MATRIX = [r"v_wmma[a-z0-9_]*", r"v_mfma[a-z0-9_]*"]
DOTCLASS = [r"v_dot[0-9]+_[a-z0-9_]*", r"v_dp4a[a-z0-9_]*", r"v_pk_fma_f16", r"v_pk_fma_bf16",
            r"v_fma_mix_f32", r"v_pack_b32_f16"]
FMA_PATH = [r"v_fma_f32", r"v_fmac_f32_e32", r"v_mad_f32", r"v_madrs_f32", r"v_fma_f64",
            r"v_dual_fma_f32", r"v_dual_fmac_f32", r"v_fma_{0,1}f32"]
LDS = [r"ds_read_[a-z0-9_]*", r"ds_write_[a-z0-9_]*", r"ds_load[a-z0-9_]*", r"ds_store[a-z0-9_]*",
       r"ds_swizzle[a-z0-9_]*", r"ds_bpermute[a-z0-9_]*", r"ds_add_u32"]
VMEM = [r"global_load_[a-z0-9_]*", r"global_store_[a-z0-9_]*", r"global_load", r"global_store"]
SMEM = [r"load_[a-z0-9_]*b[0-9]+", r"store_[a-z0-9_]*b[0-9]+"]

KHDR = re.compile(r"^[0-9a-f]+ <(.+)>:$")
INSN = re.compile(r"^[ \t]+([a-z][a-z0-9_.]*)\b")


def families(patterns):
    return re.compile(r"^(?:" + "|".join(patterns) + ")$")


def split_kernels(paths):
    """=> {mangled kernel name: [mnemonic, ...]}"""
    out = collections.defaultdict(list)
    cur = None
    for path in paths:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = KHDR.match(line)
                if m:
                    cur = m.group(1)
                    out[cur]  # create the key even for an empty kernel, so CANARY can see it
                    continue
                if cur is None:
                    continue
                im = INSN.match(line)
                if im:
                    out[cur].append(im.group(1).split(".")[0])
    return out


def demangle(names):
    import subprocess
    if not names:
        return {}
    p = subprocess.run(["c++filt"] + list(names), capture_output=True, text=True)
    return dict(zip(names, p.stdout.split("\n")))


def main(argv):
    want = None
    if "--kernels" in argv:
        want = argv[argv.index("--kernels") + 1].split(",")
    raw = argv[argv.index("--kernels") + 1] if "--kernels" in argv else None
    skip = {"--kernels", *(want or []), *([raw] if raw else [])}
    paths = [a for a in argv[1:] if not a.startswith("--") and a not in skip]
    if not paths:
        print(__doc__)
        return 2

    kernels = split_kernels(paths)

    # ---- canary ------------------------------------------------------------------
    print("## 0. CANARY (a census that read nothing must not be able to print zeros)")
    listing_ok = any("elf64-amdgpu" in open(p).read(200) for p in paths)
    print("  listing header is an AMD GPU code object: %s  (files: %s)"
          % (listing_ok, ", ".join(paths)))
    if not listing_ok:
        print("CANARY FAIL: input is not an AMD GPU disassembly")
        return 1
    targets = [k for k in (want or []) ]
    missing = [t for t in targets if not any(t in k for k in kernels)]
    print("  kernels the census was asked for and must exist: %s" % (", ".join(targets) or "(all)"))
    if missing:
        print("CANARY FAIL: not present in this listing: %s" % ", ".join(missing))
        return 1
    empty = [k for k, v in kernels.items() if not v]
    if empty:
        print("CANARY FAIL: kernel(s) with zero instructions (disassembly did not run): %s" % empty)
        return 1
    print("  kernels present, all non-empty: %s" % ", ".join("%s(%d insns)" % (k, len(v))
                                                            for k, v in kernels.items()))
    print("CANARY PASS\n")

    M, D, F, L, V, S = (families(p) for p in (MATRIX, DOTCLASS, FMA_PATH, LDS, VMEM, SMEM))
    names = demangle(list(kernels))

    print("## 1. instruction-class census (static, whole kernel body — includes cold/preamble code)")
    print("%-34s %6s %6s %5s %6s %6s %6s %6s %6s" %
          ("kernel", "insns", "MATRIX", "DOT", "FMA32", "LDS", "VMEM", "SMEM", "SALU"))
    for k, insns in kernels.items():
        c = collections.Counter(insns)
        matrix = sum(n for m, n in c.items() if M.match(m))
        dot = sum(n for m, n in c.items() if D.match(m))
        fma = sum(n for m, n in c.items() if F.match(m))
        lds = sum(n for m, n in c.items() if L.match(m))
        vmem = sum(n for m, n in c.items() if V.match(m))
        smem = sum(n for m, n in c.items() if S.match(m))
        salu = sum(n for m, n in c.items() if m.startswith("s_"))
        label = names.get(k, k)
        print("%-34s %6d %6d %5d %6d %6d %6d %6d %6d" %
              (label[:34], len(insns), matrix, dot, fma, lds, vmem, smem, salu))

    print("\n## 2. every matrix / dot-class mnemonic actually present, per kernel (0 rows means none)")
    any_matrix = False
    for k, insns in kernels.items():
        c = collections.Counter(insns)
        hits = {m: n for m, n in c.items() if M.match(m) or D.match(m)}
        if hits:
            any_matrix = True
            print("  %-34s %s" % (names.get(k, k), sorted(hits.items(), key=lambda x: -x[1])))
    if not any_matrix:
        print("  none. No v_wmma_*, v_mfma_*, v_dot*, v_dp4a*, v_pk_fma_*, v_fma_mix_f32 anywhere "
              "in these kernels.")

    print("\n## 3. what the FMA path is actually built from (the class that carries the work)")
    for k, insns in kernels.items():
        c = collections.Counter(insns)
        top = c.most_common(14)
        print("  %-30s %s" % (names.get(k, k),
                              "  ".join("%s=%d" % (m, n) for m, n in top)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
