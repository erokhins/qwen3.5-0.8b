#!/usr/bin/env python3
"""token_bill.py — what ONE token-step of a GDN kernel costs, and how much of it is address arithmetic.

WHY THIS IS NOT `isa_census.py`. `isa_census.py` bills a token-step as `prologue + trips x loopA +
trips x loopB`. That is exactly right for the per-token kernels (`rec_kernel`, `rec_fused_kernel`: one
launch IS one token-step) and it over-bills the looped ones, because their "prologue" mixes blocks that
run ONCE PER LAUNCH (kernarg load, the `tokens <= 0` guard, the pre-loop pointer setup) with blocks that
run ONCE PER TOKEN (the norms, the shuffles, the per-token address recomputation, the loop control), and
bills both at x1. At span 1 the two sets coincide and nothing shows; at the shipped span of 512 the
per-launch part amortises 512 ways, so the honest per-token bill is the FLAT number minus the per-launch
part. This file splits them and prints both. The flat number stays the one to compare against
`progress/2026-10-21/performance-status-2026-10-21.md` §24's "+156 slots/token-step", because that is how it was taken.

THE SECOND THING IT MEASURES — the whole point of the addressing item: how much of a token-step is
ADDRESS ARITHMETIC. The category table in `isa_census.py` cannot answer it, because LLVM's per-token
address math here is not one mnemonic: it is `s_mul_hi_u32` + `s_mul_i32` + `s_lshl_b64` (the `t*CH`
byte offset), a `v_add_co_u32` + `v_add_co_ci_u32_e64` pair (folding that offset into the per-thread row
address), and `s_lshl_b64`s for `t*NH` and `t*NH*D`. Only the shifts even reach its `addr.math` bucket;
the multiplies land in `unclassified`. So classification here is by USE, not by mnemonic:

    an instruction is address arithmetic iff the register it writes is written ONLY so that it can reach
    a memory instruction's address operand — i.e. every use of that register is either a memory address
    or an operand of another address-arithmetic instruction.

That is a fixpoint seeded from the memory instructions' address operands, grown backwards through defs.
`--selftest` proves it can move BOTH ways on the real listing: pointing a `global_load`'s address at a
register an FMA defines must make an FMA join the set; cutting a real address operand's producer out of
the chain must make it leave; and a must-not-catch census (no `v_fma_*`/`v_mul_*` in the set as shipped)
proves it is not simply counting everything. A classifier whose answer cannot move is decoration —
the `prototype/gdn_collapse/run.py` trap, a parse that matched nothing and printed PASS.

CPU-only. Reads a `.s` produced by `hipcc --save-temps`; launches nothing, times nothing, prints no ms.

Usage:
    prototype/gdn_rec_isa/token_bill.py --asm tmp/gcc/gdn_addr/kernels_gdn-...-gfx1151.s \
        --kernels rec_fused_kernel,rec_loop_kernel
    prototype/gdn_rec_isa/token_bill.py --asm SAME --kernels rec_loop_kernel --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cen = _load("gdn_rec_isa_census", HERE / "isa_census.py")   # shipped census: parser, rules, walk finder
base = cen.base

# --- what counts as a memory instruction, and which of its operands is the address -----------------
MEM_RE = re.compile(r"^((global|flat|scratch|queued|buffer)_\w*(load|store)|ds_)")


def is_mem(mn: str) -> bool:
    return bool(MEM_RE.match(mn))


def split_ops(text: str) -> list[str]:
    """Operands of an instruction line, split on top-level commas only (`v[2:3]` stays one operand)."""
    ops, depth, cur = [], 0, ""
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            ops.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        ops.append(cur.strip())
    return ops


REG_RE = re.compile(r"^([vs])\[(\d+):(\d+)\]$|^([vs])(\d+)$")


def regs_of(op: str) -> list[str]:
    """Physical registers named by one operand, as `v12`/`s6`. Immediates, `scc`, `exec_lo`, `vcc_lo`
    and `off` come back empty — which is what keeps the fixpoint out of the exec bookkeeping."""
    m = REG_RE.match(op.strip())
    if not m:
        return []
    if m.group(1):
        return [f"{m.group(1)}{i}" for i in range(int(m.group(2)), int(m.group(3)) + 1)]
    return [f"{m.group(4)}{m.group(5)}"]


def body_of(text: str) -> str:
    return text.split(None, 1)[1] if " " in text else ""


def sub_ops(text: str) -> list[tuple[str, list[str]]]:
    """One listing line -> [(mnemonic, operands)]. A `v_dual_*` line carries TWO operations separated by
    `::`, and LLVM uses them to bundle a float op with an address op in one issue slot, so a classifier
    that reads only the first half mis-attributes the second half's destination."""
    out = []
    for part in text.split("::"):
        part = part.strip()
        if not part:
            continue
        mn, _, rest = part.partition(" ")
        out.append((mn.strip(), [o.split()[0] for o in split_ops(rest)]))
    return out


def addr_ops(mn: str, text: str) -> list[str]:
    """The registers a memory instruction uses as an ADDRESS.

    Operand order differs by verb, and getting it backwards bills the recurrence to the address bill:
        global_load_b32   v2, v[2:3], off offset:-4096   vdata, vaddr, soff  -> v2, v3
        global_load_b32   v6, v1, s[4:5]                 vdata, vaddr, soff  -> v1, s4, s5
        global_store_b32  v[3:4], v20, off               vaddr, vdata, soff  -> v3, v4
        ds_load_b128      v[10:13], v50 offset:1024      vdata, vaddr(+imm)  -> v50
        ds_store_b32      v11, v10 offset:1024           vdata, vaddr(+imm)  -> v10
    A b32 store cannot take a VGPR pair as DATA, which is what fixes the global order; the DS order is
    the opposite (`ds_write_b32 vdst, vaddr` is the documented form)."""
    out: list[str] = []
    for m2, ops in sub_ops(text):
        if not ops:
            continue
        if m2.startswith("ds_"):
            cand = ops[1:2]
        elif "_store" in m2:
            cand = ops[0:1] + ops[2:3]
        else:
            cand = ops[1:3]
        for op in cand:
            if op.startswith(("off", "offset")):
                continue
            out += regs_of(op)
    return out


def all_reads(mn: str, text: str) -> list[str]:
    """Every register the line reads. A load's destination is not a read; a store's address AND its data
    both are. Coarser than ideal only where a dual's second destination is also listed as a read, which
    cannot inflate the address set (it can only make a producer look less pure)."""
    out: list[str] = []
    for m2, ops in sub_ops(text):
        if not ops:
            continue
        keep = ops if ("_store" in m2 and not m2.startswith("ds_")) else ops[1:]
        for op in keep:
            if op.startswith(("off", "offset")):
                continue
            out += regs_of(op)
    return out


NO_DEF = ("s_cbranch", "s_branch", "s_endpgm", "s_waitcnt", "s_delay_alu", "s_nop", "s_barrier",
          "s_clause", "s_setimm", "buffer_gl0_inv", "s_sendmsg", "s_or_b32", "s_and_b32",
          "s_and_saveexec", "s_xor_b32", "s_mov_b32", "s_mov_b64", "s_setpc", "s_nop")


def defs_of(mn: str, text: str) -> list[str]:
    """The registers a line writes: the first operand of each operation on the line (both halves of a
    `v_dual_*`), when that operand is a register.

    A LOAD's destination is a def even though it is not address math: the loaded value later flows into the
    recurrence, and if the load does not claim that def every later read of `v2` attributes to whatever
    produced the ADDRESS, which makes a clean pointer add look impure and drops it out of the set — the bug
    that made the shipped `t*CH` multiply invisible on the first run. A store writes nothing.

    Two coarseness notes. (1) The NO_DEF list covers exec-mask and comparison bookkeeping (`s_mov_b32`,
    `s_xor_b32`, `s_and_b32`), whose destinations the shuffle guards own; counting them would bill the guards
    as address arithmetic. (2) Carries are not counted: `v_add_co_u32 v2, vcc_lo, ...` also writes `vcc_lo`,
    and skipping that only ever makes the address set smaller."""
    if mn.startswith(NO_DEF):
        return []
    if is_mem(mn) and "_store" not in mn and not mn.startswith("ds_store"):
        ops = [o.split()[0] for o in split_ops(body_of(text))]        # a load defines its destination
        return regs_of(ops[0]) if ops else []
    if is_mem(mn):
        return []
    out: list[str] = []
    for _m2, ops in sub_ops(text):
        if ops:
            out += regs_of(ops[0])
    return out


def uses_of(mn: str, text: str) -> list[str]:
    return all_reads(mn, text)


# The candidate mnemonics: arithmetic that can only be serving a pointer or an index in this kernel.
# The gate is required, not cosmetic. The def-use rule alone also admits instructions whose result happens
# to be read as an address because LLVM reused a dead register — a float add whose destination was
# recycled as an index — and that bills the RECURRENCE to the address bill (the first version counted 46
# `v_fmac_f32_e32` slots in the state walk, which is nonsense). With the gate an instruction qualifies only
# if it is one of these AND its destination is read as an address, or by another qualifying instruction
# that reaches one.
ADDR_MNEMONICS = (
    "s_add_u32", "s_addc_u32", "s_add_i32", "s_lshl_b64", "s_lshr_b64", "s_mul_i32", "s_mul_hi_u32",
    "v_add_co_u32", "v_add_co_ci_u32_e64", "v_add_nc_u32", "v_add_nci_u32_e64",
    "v_or_b32", "v_lshl_add_u32", "v_lshl_or_b32", "v_mov_b32_e32", "v_mov_b32_e64",
)
# A `v_dual_*` slot bundles two operations in one issue slot and LLVM uses that to pair a float op with
# an address op (`v_dual_add_f32 v24, ... :: v_dual_lshlrev_b32 v23, 2, v25`). Such a SLOT is partly
# address work, and it is the slot that is billed, so the second half gets a vote.
DUAL_ADDR_SUBS = ("v_lshlrev_b32", "v_lshrrev_b32", "v_add3_u32", "v_or_b32", "v_add_co",
                  "v_add_nc", "v_mov_b32")


def addr_candidate(mn: str, line: str) -> bool:
    if mn.startswith(ADDR_MNEMONICS):
        return True
    if mn.startswith("v_dual"):
        return any(sub.startswith(DUAL_ADDR_SUBS) for sub, _o in sub_ops(line))
    return False


def address_math(insts: list[tuple[str, str]]) -> set[int]:
    """Indices of `insts` that are address arithmetic, per the rule in the module docstring.

    Register-level def-use with the reuse handled, which is the whole difficulty on real assembly: LLVM
    writes an address into `v2` and then writes the LOADED VALUE into the same `v2` (`v_add_co_u32 v2,
    ...` then `global_load_b32 v2, v[2:3], off`). Treating "uses of v2" as uses of the ADD makes the add
    look impure — its register appears to feed float math — and the first version of this function
    therefore counted 5 address slots where the listing has dozens. So each use is attributed to the
    LATEST preceding def of that register (the assembly's own SSA order), and a use with no preceding def
    in the region is external and does not bind to anything here."""
    defs: dict[str, list[int]] = defaultdict(list)
    for i, (mn, ln) in enumerate(insts):
        for r in defs_of(mn, ln):
            defs[r].append(i)

    # use attribution: for each instruction, each read register maps to the def that produced it.
    reads: dict[int, list[tuple[str, int]]] = defaultdict(list)     # i -> [(reg, defining idx)]
    addr_reads: dict[int, set[str]] = {}                           # i -> regs read AS ADDRESSES
    for i, (mn, ln) in enumerate(insts):
        rr = set()
        if is_mem(mn):
            rr = set(addr_ops(mn, ln))
        addr_reads[i] = rr
        for r in uses_of(mn, ln):
            ds = defs[r]
            lo, hi = 0, len(ds)
            while lo < hi:                                          # last def strictly before i
                mid = (lo + hi) // 2
                if ds[mid] <= i:
                    lo = mid + 1
                else:
                    hi = mid
            if lo:                                                  # else: defined outside the region
                reads[i].append((r, ds[lo - 1]))

    uses_of_def: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, rs in reads.items():
        for r, d in rs:
            uses_of_def[(r, d)].append(i)

    want: set[int] = set()
    for _ in range(24):                            # chains are short; the bound keeps a cycle honest
        grew = False
        for i, (mn, ln) in enumerate(insts):
            if i in want or is_mem(mn) or not addr_candidate(mn, ln):
                continue
            ds = defs_of(mn, ln)
            if not ds:
                continue
            connected, pure = False, True
            for r in ds:
                for j in uses_of_def[(r, i)]:
                    if j == i or j in want or r in addr_reads[j]:
                        connected = True
                        continue
                    pure = False
                    break
                if not pure:
                    break
            if pure and connected:
                want.add(i)
                grew = True
        if not grew:
            break
    return want


# --- loop membership, read off LLVM's own labels ---------------------------------------------------
HDR_RE = re.compile(r"This (?:Inner |Sub ?)*Loop Header: Depth=(\d+)")
IN_RE = re.compile(r"in Loop: Header=(\S+) Depth=(\d+)")
PAR_RE = re.compile(r"Parent Loop (\S+) Depth=(\d+)")


def loop_membership(body: list[str]) -> dict[str, set[str]]:
    """label -> the set of loop-header labels that block belongs to (header included).

    LLVM prints the membership INLINE on the label line (`.LBB5_2:  ; in Loop: Header=BB5_4 Depth=1`,
    `.LBB5_32: ; Parent Loop BB5_4 Depth=1` plus `; => This Inner Loop Header: Depth=2` on the next
    line), and `split_blocks_full` drops an inline tail with the label, so the annotations have to be
    read here instead. Labels are normalised with `base.norm_key`, so `.LBB5_4` and `BB5_4` agree."""
    out: dict[str, set[str]] = {}
    cur: str | None = None
    for ln in body:
        lab = base.LABEL_RE.match(ln) or base.BB_COMMENT_RE.match(ln)
        tail = ln.split(";", 1)[1] if ";" in ln else ""
        if lab:
            cur = base.norm_key(lab.group(1))
            out.setdefault(cur, set())
            m = HDR_RE.search(tail)
            if m:
                out[cur].add(cur)
        if cur is None:
            continue
        for rx in (IN_RE, PAR_RE):
            for m in rx.finditer(tail):
                out[cur].add(base.norm_key(m.group(1)))
    return out


# --- regions --------------------------------------------------------------------------------------
def split_regions(body: list[str]):
    """(walks, per-launch blocks, per-token blocks) for one kernel body.

    The token loop is found structurally, not by name: it is the loop header that (a) is an ancestor of
    a state-walk block and (b) is not itself a walk header. In a per-token kernel the two walks sit at
    depth 1 with no such ancestor, so there is no token loop, every live block is per-token and none is
    per-launch — which is exactly the statement that the two bills coincide at span 1.
    """
    blocks = cen.split_blocks_full(cen.annotate_loops(body))
    live, dead = cen.executed_blocks(blocks)
    walks = cen.find_walks(live)
    if len(walks) != 2:
        raise SystemExit(f"expected 2 state walks, found {sorted(walks)} — the fingerprint moved")
    in_walk = {id(b) for v in walks.values() for b in v["blocks"]}
    walk_hdr = {base.norm_key(v["hdr"]): v for v in walks.values()}
    memb = {base.norm_key(b["label"]): loop_membership(body).get(base.norm_key(b["label"]), set())
            for b in live}
    # The token loop is a loop the WALKS belong to that is not itself a walk header. Read off the
    # walk blocks' own membership sets rather than a depth number, because LLVM labels the walk headers
    # `This Inner Loop Header: Depth=2` under `Parent Loop BB5_4 Depth=1` and rotates the token loop so
    # that its back edge is a fall-through (`%Flow310` -> header), which no branch-target scan sees.
    hdr_of = {base.norm_key(v["hdr"]) for v in walks.values()}
    seen: set[str] = set()
    for v in walks.values():
        for b in v["blocks"]:
            seen |= memb.get(base.norm_key(b["label"]), set())
    outer = sorted((seen | set()) - hdr_of)
    tok_hdr = outer[0] if outer else None
    per_tok, per_launch = [], []
    for b in live:
        if id(b) in in_walk:
            continue
        if tok_hdr is not None:
            (per_tok if tok_hdr in memb.get(base.norm_key(b["label"]), set()) else per_launch).append(b)
        else:
            per_tok.append(b)
    return walks, per_launch, per_tok, tok_hdr, dead


def slots_of(insts) -> int:
    return cen.slots(insts)


def bill(name: str, body: list[str], w, verbose: bool = True) -> dict:
    walks, per_launch, per_tok, tok_hdr, dead = split_regions(body)
    ta, tb = walks["loopA_state_dot"]["trips"], walks["loopB_state_update"]["trips"]
    ia = [ln for b in walks["loopA_state_dot"]["blocks"] for ln in b["insts"]]
    ib = [ln for b in walks["loopB_state_update"]["blocks"] for ln in b["insts"]]
    ip = [ln for b in per_launch for ln in b["insts"]]
    it = [ln for b in per_tok for ln in b["insts"]]

    flat = slots_of(ip) + slots_of(it) + slots_of(ia) * ta + slots_of(ib) * tb
    persub = slots_of(ip)
    pertok = slots_of(it) + slots_of(ia) * ta + slots_of(ib) * tb

    ad_t = address_math(it)
    ad_a, ad_b = address_math(ia), address_math(ib)

    def live_addr(idx: set[int], insts) -> list[tuple[str, str]]:
        return [(mn, ln) for i, (mn, ln) in enumerate(insts)
                if i in idx and not mn.startswith(base.SKIP_AS_WORK)]

    a_t, a_a, a_b = live_addr(ad_t, it), live_addr(ad_a, ia), live_addr(ad_b, ib)
    addr_tok = len(a_t) + (len(a_a) * ta + len(a_b) * tb)

    w(f"===== {name}")
    w(f"  walks                 : loopA x{ta} ({walks['loopA_state_dot']['per_trip']} state-accesses/trip)"
      f", loopB x{tb} ({walks['loopB_state_update']['per_trip']}/trip)")
    w(f"  token loop            : "
      + ("NONE — a per-token kernel, so every live block runs once per token and once per launch"
         if tok_hdr is None else f"{tok_hdr}"))
    w(f"  per-LAUNCH straight   : {len(per_launch):3} blocks, {persub:4} slots"
      + ("" if persub else "   (nothing sits outside the token loop: flat == per-token)"))
    w(f"  per-TOKEN straight    : {len(per_tok):3} blocks, {slots_of(it):4} slots")
    w(f"  walks per token       : loopA {slots_of(ia)}x{ta}={slots_of(ia)*ta},"
      f"  loopB {slots_of(ib)}x{tb}={slots_of(ib)*tb}")
    w(f"  FLAT token-step bill  : {flat} slots/wavefront     <- comparable with isa_census.py and §24 of progress/2026-10-21/performance-status-2026-10-21.md")
    w(f"  PER-TOKEN bill        : {pertok} slots/wavefront"
      + (f"   (flat - {persub} per-launch slots)" if persub else ""))
    w(f"  ADDRESS ARITHMETIC    : {len(a_t)} straight-line + {len(a_a)*ta + len(a_b)*tb} in walks"
      f" = {addr_tok} slots = {100.0*addr_tok/pertok:.1f}% of the per-token bill,"
      f" {100.0*addr_tok/flat:.1f}% of the flat bill")
    if verbose:
        cp = Counter(cen.category(mn, ln) for mn, ln in ip
                     if not mn.startswith(base.SKIP_AS_WORK))
        ct = Counter(cen.category(mn, ln) for mn, ln in it
                     if not mn.startswith(base.SKIP_AS_WORK))
        cw = Counter(cen.category(mn, ln) for x in (ia, ib)
                     for mn, ln in x if not mn.startswith(base.SKIP_AS_WORK))
        w("  category rollup       : per-launch | per-token-straight | per trip of the walks")
        for k in sorted(set(cp) | set(ct) | set(cw), key=lambda k: -(ct.get(k, 0))):
            w(f"    {k:<24} {cp.get(k,0):5} | {ct.get(k,0):5} | {cw.get(k,0):4}")
        w("  address-arithmetic mnemonics, straight line (per token) / walks (per trip):")
        st = Counter(mn for mn, _ in a_t)
        stw = Counter(mn for mn, _ in a_a + a_b)
        for k in sorted(set(st) | set(stw), key=lambda k: (-(st.get(k, 0) + stw.get(k, 0)), k)):
            w(f"    {k:<26} {st.get(k,0):4} / {stw.get(k,0):3}")
        bad = pure_float_lines(a_t + a_a + a_b)
        w(f"  must-not-catch        : address-set lines with no address-class operation at all = {len(bad)}"
          + ("" if not bad else f"  <-- {bad[:3]}"))
    return {"flat": flat, "pertok": pertok, "persub": persub, "addr": addr_tok,
            "lines": [ln for _mn_, ln in a_t + a_a + a_b], "bad": pure_float_lines(a_t + a_a + a_b),
            "mn": Counter(mn for mn, _ in a_t)}


# --- selftest mutations ---------------------------------------------------------------------------
def _mn(line: str) -> str:
    s = line.strip()
    return s.split(None, 1)[0] if s else ""


def mutate(lines: list[str], kind: str) -> list[str]:
    """Mutations that move the address count in a direction fixed in advance. See `--selftest`.

    They are SYNTHETIC additions rather than edits of existing lines, because on a real listing every
    physical register is reused: deleting or renaming one address instruction also changes who the later
    reads of that register belong to, so its effect on the count is not predictable and a check built on it
    would fail for reasons that say nothing about the classifier. Appending a self-contained chain does:

    `append_addr`    adds `v_add_co_u32 v9500, vcc_lo, s95, v9501` + `global_load_b32 v9502, v[9500:9501]`
                     into the token loop. The add's only use is that load's address, so it must JOIN the
                     set: the count must GROW. If it does not, the census is not reading the loop body.
    `append_float`   adds `v_add_f32_e32 v9503, v9504, v9505`, whose result is read by nothing. The count
                     must be UNCHANGED: the census is not simply counting instructions inside the loop.
    Both are inserted in the same place, so the only difference between them is the character of the work.
    """
    out = list(lines)
    _walks, _pl, per_tok, _h, _d = split_regions(lines[1:])
    if not per_tok:
        raise SystemExit("no per-token block found — there is no loop body to mutate")
    lbl = per_tok[len(per_tok) // 2]["label"]
    # Find the line that BEGINS the block, not the first line that mentions it: a branch target or a comment
    # earlier in the file would put the mutation in another region and the count would move (or not) for the
    # wrong reason.
    at = None
    for i, l in enumerate(out):
        lab = base.LABEL_RE.match(l) or base.BB_COMMENT_RE.match(l)
        if lab and lab.group(1) == lbl:
            at = i + 1
            break
    if at is None:
        raise SystemExit(f"cannot find the block {lbl} to insert into")
    if kind == "append_addr":
        add = ["\tv_add_co_u32 v9500, vcc_lo, s95, v9501",
               "\tglobal_load_b32 v9502, v[9500:9501], off"]
    elif kind == "append_float":
        add = ["\tv_add_f32_e32 v9503, v9504, v9505"]
    else:
        raise SystemExit(f"unknown mutation {kind}")
    return out[:at] + add + out[at:]


def attached_to_memory(body: list[str]) -> int:
    """How many counted address lines write a register a memory instruction reads AS AN ADDRESS.
    The non-vacuity control: the census is attached to real memory traffic, not to a chain of arithmetic
    that reaches nothing. Zero means the set is decoration."""
    n = 0
    _walks, _pl, per_tok, _hdr, _dead = split_regions(body)
    regions = [[ln for b in per_tok for ln in b["insts"]]]
    regions += [[ln for b in w["blocks"] for ln in b["insts"]] for w in _walks.values()]
    for insts in regions:
        idx = address_math(insts)
        addrs: set[str] = set()
        for mn, ln in insts:
            if is_mem(mn):
                addrs |= set(addr_ops(mn, ln))
        n += sum(1 for i in idx if set(defs_of(*insts[i])) & addrs)
    return n


def addr_lines(body: list[str]) -> list[str]:
    """The per-token straight-line instructions the classifier calls address arithmetic, as text.
    The positive control below reads these, so a reader can see WHICH lines the census is counting."""
    _walks, _pl, per_tok, _h, _d = split_regions(body)
    insts = [ln for b in per_tok for ln in b["insts"]]
    idx = address_math(insts)
    return [ln for i, (_mn_, ln) in enumerate(insts) if i in idx]


def pure_float_lines(lines: list[tuple[str, str]]) -> list[str]:
    """Address-set lines carrying NO address-class operation, i.e. the classifier reached a line that only
    does float work. On a `v_dual_*` slot one half may legitimately be an address op — the SLOT is what gets
    billed — so the test is per line, not per mnemonic."""
    out = []
    for mn, ln in lines:
        parts = sub_ops(ln) or [(mn, [])]
        if any(sub[len("v_dual_"):] in DUAL_ADDR_SUBS or sub.startswith(ADDR_MNEMONICS)
               for sub, _o in parts):
            continue
        out.append(ln)
    return out


def addr_count(lines: list[str], name: str) -> int:
    ks = cen.split_kernels_local(lines)
    if name not in ks:
        raise SystemExit(f"mutation deleted the kernel label for {name} — the mutant is not a listing")
    return bill_counts(ks[name])


def bill_counts(body: list[str]) -> int:
    walks, _pl, per_tok, _h, _d = split_regions(body)
    it = [ln for b in per_tok for ln in b["insts"]]
    n = len(live_only(address_math(it), it))
    for k in ("loopA_state_dot", "loopB_state_update"):
        insts = [ln for b in walks[k]["blocks"] for ln in b["insts"]]
        n += len(live_only(address_math(insts), insts)) * walks[k]["trips"]
    return n


def live_only(idx: set[int], insts) -> list[tuple[str, str]]:
    return [(mn, ln) for i, (mn, ln) in enumerate(insts) if i in idx
            and not mn.startswith(base.SKIP_AS_WORK)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asm", type=Path, required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-addr", action="store_true",
                    help="print the per-token straight-line lines the classifier calls address arithmetic")
    ap.add_argument("--dump-blocks", action="store_true",
                    help="print the per-launch / per-token block split, with slot counts")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lines = args.asm.read_text().splitlines()
    kernels = cen.split_kernels_local(lines)
    names = [n for n in args.kernels.split(",") if n]
    missing = [n for n in names if n not in kernels]
    if missing:
        raise SystemExit(f"kernel(s) {missing} not in {args.asm}; here: {sorted(kernels)}")

    out: list[str] = []
    w = out.append
    w(f"listing  {args.asm}")
    w("unit     one token-step = one iteration of the token loop (or one launch, for a per-token kernel),"
      " per wavefront; grid 16x128 threads, wave32 -> 64 wavefronts")
    w("")
    res = {n: bill(n, kernels[n], w) for n in names}

    for n in names:
        if args.dump_blocks:
            walks, pl, pt, hdr, dead = split_regions(kernels[n])
            w(f"===== BLOCK SPLIT {n} (token loop = {hdr})")
            for tag, group in (("launch", pl), ("token", pt)):
                for b in group:
                    w(f"  {tag:6} {b['label']:<12} {cen.slots(b['insts']):4} slots  "
                      f"{' '.join(b['parents'][0]) if b['parents'] else ''}")
        if args.dump_addr:
            walks, pl, pt, hdr, _d = split_regions(kernels[n])
            it = [ln for b in pt for ln in b["insts"]]
            idx = address_math(it)
            w(f"===== ADDRESS LINES {n}: {len(idx)} of {len(it)} per-token straight-line lines")
            for i, (mn, ln) in enumerate(it):
                if i in idx:
                    w(f"    {ln}")

    ref = names[0]
    w("===== DELTA against the first kernel named, per token-step per wavefront")
    for n in names[1:]:
        a, b = res[ref], res[n]
        w(f"  {ref} -> {n}: flat {a['flat']} -> {b['flat']} ({b['flat']-a['flat']:+d}),"
          f" per-token {a['pertok']} -> {b['pertok']} ({b['pertok']-a['pertok']:+d}),"
          f" address-arithmetic {a['addr']} -> {b['addr']} ({b['addr']-a['addr']:+d})")
        for m in sorted(set(a["mn"]) | set(b["mn"])):
            if a["mn"].get(m, 0) != b["mn"].get(m, 0):
                w(f"    {m:<26} {a['mn'].get(m,0):4} -> {b['mn'].get(m,0):4}")

    rc = 0
    if args.selftest:
        n = names[-1]
        # Mutate a one-kernel listing: a mutation that finds its site in an EARLIER kernel of the same
        # file would leave this kernel's answer unmoved and report a false `DID-NOT-MOVE`.
        solo = [f"_Z{len(n)}{n}:"] + kernels[n]
        want = addr_count(solo, n)
        w("")
        w(f"===== SELFTEST on {n}: address slots as shipped = {want}")
        for kind, require in (("append_addr", ">"), ("append_float", "=")):
            try:
                got = addr_count(mutate(solo, kind), n)
            except SystemExit as e:
                w(f"SELFTEST {kind:14} NOT-APPLICABLE {e}")
                rc = 1
                continue
            ok = (got > want) if require == ">" else (got == want)
            w(f"SELFTEST {kind:14} {'OK' if ok else 'WRONG'} {want} -> {got}"
              f" (required {require}{want})")
            if not ok:
                rc = 1
        # NON-VACUITY, and the control that matters for this item: a counted address line must write a
        # register a memory instruction reads AS AN ADDRESS. If the census cannot show that, every number it
        # prints about addressing is decoration — which is the failure the first version of this file had,
        # when it billed 46 state-recurrence FMAs to the address bill.
        att = attached_to_memory(kernels[n])
        w(f"SELFTEST attached       : {att} counted address lines write a register a memory operand reads"
          f" as an address (required > 0)")
        if att <= 0:
            rc = 1
        floaty = res[n]["bad"]
        w(f"SELFTEST must-not-catch : {len(floaty)} counted lines that are pure float work (required 0)")
        if floaty:
            rc = 1
        w(f"SELFTEST ok={0 if rc else 1}")
    txt = "\n".join(out)
    print(txt)
    if args.out:
        args.out.write_text(txt + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
