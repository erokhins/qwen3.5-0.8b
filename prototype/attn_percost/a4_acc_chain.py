#!/usr/bin/env python3
"""A4-L1: which accumulator registers does each `v_wmma` site chain on?

NOT a fourth census tool (AGENTS.md, "before writing a census tool, list the census tools"). It bills nothing and
classifies nothing:

  * `prototype/gdn_walk/g3_isa_bill.py` finds loop spans but prints only the five widest and its per-span key set
    has no matrix rung, so it cannot name the INNERMOST loop around a chosen instruction;
  * `prototype/attn_percost/a2_round_bill.py` histograms a span someone else located, and cannot see registers;
  * `prototype/gdn_rec_isa/token_bill.py` splits per-launch from per-token blocks for GDN and does not apply to
    this kernel at all — against this listing it exits 1 with `.LBB6_65: 33 state accesses/trip does not divide
    D=128` (its walk finder is GDN-shaped), and the symbol it needs is `engine::kernels::attn_fattn_wave`;

The question none of them asks — and the one A4's Gate 0 and Gate 1 turn on — is register identity: does this
`v_wmma` consume as its accumulator (4th operand) the tuple its own destination writes, and if so, is that tuple
written by a PRECEDING site in the same loop (chained) or by a different tuple (independent chain)? A "two
accumulators" change that the compiler folded back into one chain must read as one chain here, which is what makes
Gate 1 non-vacuous.

    python3 prototype/attn_percost/a4_acc_chain.py OBJDUMP.S [--kernel attn_fattn_wave]
            [--require-chains N]        # exit 1 unless the matrix loop holds exactly N distinct accumulator tuples
            [--show N]                  # print up to N matrix sites verbatim

Refuses: a listing with no matrix site, a site whose operands do not parse, and (with --require-chains) a count
that does not match — a check whose input list can be empty cannot fail.
"""
import argparse
import re

FUNC = re.compile(r"^([0-9a-fA-F]{16}) <(.*)>:")
INSN = re.compile(r"^\s+([a-z_][a-z0-9_.]*)\s(.*?)//\s*([0-9A-Fa-f]{8,16}):")
TUPLE = re.compile(r"^v\[(\d+):(\d+)\]$|^v(\d+)$")


def tup(op):
    """`v[129:136]`/`v129` -> a canonical tuple key; anything else (sgpr, imm, `off`) -> None."""
    m = TUPLE.match(op.split()[0].strip()) if op else None
    if not m:
        return None
    if m.group(1):
        return (int(m.group(1)), int(m.group(2)))
    return (int(m.group(3)), int(m.group(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    ap.add_argument("--kernel", default="attn_fattn_wave")
    ap.add_argument("--require-chains", type=int, default=None)
    ap.add_argument("--show", type=int, default=4)
    a = ap.parse_args()

    lines = open(a.listing).read().splitlines()
    start, base, name = None, None, None
    for i, ln in enumerate(lines):
        m = FUNC.match(ln)
        if m and a.kernel in m.group(2):
            start, base, name = i, int(m.group(1), 16), m.group(2)
            break
    if start is None:
        raise SystemExit(f"REFUSE: no function matching {a.kernel!r} in {a.listing}")

    body = []                       # [addr, mnemonic, operands-text, branch-target, raw line]
    for ln in lines[start + 1:]:
        if FUNC.match(ln):
            break
        m = INSN.match(ln)
        if not m:
            continue
        body.append([int(m.group(3), 16), m.group(1), m.group(2), None, ln])
    if not body:
        raise SystemExit(f"REFUSE: {name} parsed to 0 instructions")
    idx = {r[0]: n for n, r in enumerate(body)}

    # Backward branches, so each matrix site can be attributed to its innermost enclosing loop. llvm-objdump
    # prints an absolute branch target as `<symbol+0xOFFSET>` in the trailing comment, which is the same route
    # g3_isa_bill.py takes; the bare decimal form it prints for a branch with no resolvable symbol is a
    # PC-relative displacement in INSTRUCTION units, so it is scaled by 4 here — read off this very listing
    # (0xe650 `s_cbranch_scc0 65520` -> signed -16 -> 0xe654 - 64 = 0xe614, the score-roll header).
    for n, rec in enumerate(body):
        addr, mn, _ops, _t, raw = rec
        if not (mn.startswith("s_cbranch") or mn.startswith("s_cjump")):
            continue
        m = re.search(r"\+0x([0-9A-Fa-f]+)>", raw)
        if m:
            tgt = base + int(m.group(1), 16)
        else:
            m2 = re.match(r"\s*(-?\d+)\s*$", raw.split("//")[0].split(None, 1)[1].strip())
            if not m2:
                continue
            v = int(m2.group(1))
            if v >= 32768:
                v -= 65536          # llvm prints the 16-bit displacement unsigned
            tgt = addr + 4 + 4 * v
        if tgt < addr and tgt in idx:
            rec[3] = tgt

    sites = [n for n, r in enumerate(body) if r[1].startswith(("v_wmma", "v_mfma"))]
    if not sites:
        raise SystemExit(f"REFUSE: no matrix-class site in {name} — a census over nothing.")

    def innermost_loop(site):
        """Smallest backward span [tgt, branch] that contains `site` and whose back edge is at/after it."""
        best = None
        for n, rec in enumerate(body):
            addr, tgt = rec[0], rec[3]
            if tgt is None or n < site or body[site][0] < tgt:
                continue
            span = addr - tgt
            if best is None or span < best[0]:
                best = (span, tgt, addr)
        return best

    print(f"==== {name}  (base 0x{base:x}, {len(body)} instructions, {len(sites)} matrix sites)")
    loops = {}
    for n in sites:
        addr, mn, ops = body[n][0], body[n][1], body[n][2]
        parts = [p.strip() for p in ops.split(",")]
        if len(parts) < 4:
            raise SystemExit(f"REFUSE: matrix site at 0x{addr:x} has {len(parts)} operands, cannot read the "
                             "accumulator. Wrong disassembler output, not a zero result.")
        dst, a_op, b_op, c_op = (tup(p) for p in parts[:4])
        lp = innermost_loop(n)
        loops.setdefault(lp, []).append(n)
        print(f"  site 0x{addr:5x}  dst={dst}  A={a_op}  B={b_op}  acc-in={c_op}"
              f"   dst==acc-in: {'YES' if dst == c_op else 'no'}"
              f"   loop: {'none (straight line)' if lp is None else '0x%x..0x%x' % (lp[1], lp[2])}")
        if n - start < a.show * 1000 and a.show and sites.index(n) < a.show:
            print(f"        raw: {mn} {ops.split('//')[0].strip()}")

    # The roll is the matrix loop holding the FIRST site; report the chain structure inside it.
    roll_loop = innermost_loop(sites[0])
    members = loops.get(roll_loop, [sites[0]])
    accs = []
    for n in members:
        parts = [p.strip() for p in body[n][2].split(",")]
        t = tup(parts[3])
        if t not in accs:
            accs.append(t)
    if roll_loop is None:
        print(f"# first matrix site is NOT in a loop; {len(accs)} distinct accumulator tuple(s) among the "
              f"{len(members)} sites in that span.")
    else:
        trips = ""
        print(f"# matrix loop 0x{roll_loop[1]:x}..0x{roll_loop[2]:x}: {len(members)} site(s), "
              f"{len(accs)} distinct accumulator tuple(s): {accs}{trips}")
        if len(accs) == 1:
            print("#   ONE accumulator: every site in the loop chains on the same tuple it writes "
                  "-> a serial recurrence of (trips x sites) matrix ops.")
        else:
            print(f"#   {len(accs)} accumulators: {len(accs)} chains progress at once inside this loop.")
    if a.require_chains is not None:
        ok = len(accs) == a.require_chains
        print(f"# REQUIRE-CHAINS {a.require_chains}: {'OK' if ok else 'FAIL'} (got {len(accs)})")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
