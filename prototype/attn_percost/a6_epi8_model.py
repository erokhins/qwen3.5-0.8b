#!/usr/bin/env python3
"""G1's epilogue witness — since the graduation, a test OF THE SHIPPING ENGINE: does an 8-row transpose stage
publish the SAME floats the shipped 16-row stage does, and does it publish each of them exactly once?
CPU only — no device, no timing.

WHY THIS EXISTS. G1's bit-identity claim (A6-BET.md "Numerics", A5-GEOMETRY.md §3.2) rests on one sentence:
*"the epilogue change moves staging addresses only, never the published values"*. That is an argument, and this
tree's trap list is unambiguous about arguments of that shape — a check whose reference side is derived from the
thing under test cannot adjudicate it, and an aggregate that does not name its denominator cannot fail. So this
file enumerates both epilogues' writes as arithmetic — no kernel, no floats, just the index functions — and asks
three questions each of which can be wrong:

  1. COVERAGE: is every (row-slot, dim) of the warp's published half-row written EXACTLY once, in each form?
     The shipped file's own defect #2 on `fattn_transpose_out` — 16 dims published per row, 240 never written
     — is invisible to a bound check and visible to exactly this one.
  2. IDENTITY: when a (row-slot, dim) is written in both forms, does the same accumulator element
     `(lane, t, l)` — the same `out.t[t][l]` of the same lane — land in it? If yes, the published bytes are the
     shipped ones whatever the stage layout, and no numerics band re-opens.
  3. BOUNDS: is every stage index below the stage's declared size, in each form? G1's stage is half as deep, so
     an index derived from the old geometry walks off it into the next warp's stage — the §19/§20.1 aliasing
     class of prototype/attn_decode/results.md, caught here instead of on a device.

AND A FOURTH, added when the candidate was graduated into `engine/attn_fattn_kernels.h`: SOURCE AGREEMENT. Before
the graduation this file modelled a patcher's output; now it must model the shipping file, and a model of a
remembered source is the same failure class as a mirror that agrees with itself. So the geometry constants (DH,
FATTN_MROWS, FATTN_TROW, FATTN_TBLK, FATTN_PBLK, FATTN_TRG, FATTN_LBLK, …) are READ OUT of the header's own
`#define` lines, and every index expression this file enumerates is required to appear verbatim in it — both
epilogues' publish splits, the shared `fattn_tr_addr`, and each form's stage write/read pair. If the header moves,
this refuses rather than quietly modelling the old geometry.

`--negative` runs the must-catch set: three ways the G1 mapping could have been transcribed from the shipped one
and would be WRONG. Each must change the map, or the PASS above is decoration (AGENTS: "a negative control with
an empty must-catch set is not a control").

Run:  python3 prototype/attn_percost/a6_epi8_model.py [--negative] [--header PATH]
      --header points the whole check at another copy of the header, which is how the SOURCE AGREEMENT check
      itself is witnessed: mutate a copy, and the model must go red instead of modelling the old geometry.
Exit 0 = the header agrees with the model, and the three claims hold for all 16 row-slots x 256 dims of a warp.
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
HEADER = REPO / "engine" / "attn_fattn_kernels.h"
for _i, _a in enumerate(sys.argv):
    if _a == "--header" and _i + 1 < len(sys.argv):
        HEADER = pathlib.Path(sys.argv[_i + 1])
OPHEADER = REPO / "prototype" / "attn_decode" / "attn_decode_kernels.h"


def parse_geometry(path: pathlib.Path, opheader: pathlib.Path) -> dict:
    """Read the `#define`d geometry out of the headers that own it.

    Constants are resolved in dependency order by repeated passes, because `FATTN_TROW` is `(DH / 2)` and
    `FATTN_LBLK` is `((FATTN_TRG * (FATTN_TROW / FATTN_TBLK)) / 32)`: a model that hard-codes the answers cannot
    notice the header changed them. C's integer division is what the device does, so `/` becomes `//`.
    """
    text = path.read_text()
    defs = {}
    for m in re.finditer(r"^\s*#define\s+(\w+)\s+(.+?)\s*$", text, re.M):
        name, val = m.group(1), m.group(2).split("//")[0].strip()   # a #define's trailing comment is prose
        if re.fullmatch(r"[0-9]+", val):
            defs[name] = int(val)
        elif re.fullmatch(r"[A-Za-z0-9_()+*/ -]+", val):
            defs[name] = val          # an expression, resolved below
    m = re.search(r"^\s*#define\s+DH\s+(\d+)", opheader.read_text(), re.M)
    if not m:
        raise SystemExit("REFUSE: cannot read DH from prototype/attn_decode/attn_decode_kernels.h. A model whose"
                         " width is a guess is not a witness.")
    env = {"DH": int(m.group(1))}
    env.update({k: v for k, v in defs.items() if isinstance(v, int)})
    pending = {k: v for k, v in defs.items() if not isinstance(v, int)}
    while pending:
        progress = False
        for k, expr in list(pending.items()):
            try:
                env[k] = eval(expr.replace("/", "//"), {"__builtins__": {}}, dict(env))  # noqa: S307
                del pending[k]
                progress = True
            except NameError:
                pass
        if not progress:
            raise SystemExit(f"REFUSE: cannot resolve the header's own defines {sorted(pending)} from what this"
                             " file knows. The geometry moved; re-read it instead of modelling the old one.")
    want = ["FATTN_MROWS", "FATTN_MLANES", "FATTN_TROW", "FATTN_TBLK", "FATTN_PBLK", "FATTN_TRG",
            "FATTN_NGRP", "FATTN_NPASS", "FATTN_LBLK", "FATTN_KTILES", "FATTN_VTILES"]
    missing = [w for w in want if w not in env]
    if missing:
        raise SystemExit(f"REFUSE: {path.name} no longer defines {missing}, which this model needs. The epilogue"
                         " geometry moved; a model of the old one proves nothing about the new one.")
    env["DH"] = int(m.group(1))
    return env


# Every index expression this file enumerates must exist, verbatim, in the header that runs it. Not a style
# check: `fattn_tr_addr` being ONE spelling is what retires the aliasing class of prototype/attn_decode/
# results.md §20.1 in the graduated file, and the
# two publish splits are what the coverage claim is about. If either is re-worded in the C++ without being
# re-worded here, this model would keep printing PASS about a geometry the engine no longer has.
SOURCE_SPELLINGS = {
    "c_row": "int fattn_c_row(int lane) { return lane & 15; }",
    "c_col": "int fattn_c_col(int lane, int l) { return 2 * l + (lane >> 4); }",
    "one stage address expression": "int fattn_tr_addr(int row_in_pass, int d) { return row_in_pass * FATTN_TROW + d; }",
    "shipped stage write": "stage[fattn_tr_addr(fattn_c_row(lane), d)] = out.t[t][l];",
    "shipped stage read": "pp[2 + dbase + u] = stage[fattn_tr_addr(srow, sbase + u)];",
    "shipped publish split": "const int srow = lane >> 1;",
    "shipped block split": "const int sblk = (lane & 1) * FATTN_PBLK;",
    "shipped publish count": "for (int k = 0; k < FATTN_PBLK; ++k)",
    "g1 stage write": "stage[fattn_tr_addr(fattn_c_row(lane) - row0, d)] = out.t[t][l];",
    "g1 stage read": "pp[2 + dbase + u] = stage[fattn_tr_addr(lane >> 2, sbase + u)];",
    "g1 publish split": "const int srow = row0 + (lane >> 2);",
    "g1 block split": "const int sblk = (lane & 3) * FATTN_LBLK;",
    "g1 publish count": "for (int k = 0; k < FATTN_LBLK; ++k)",
    "g1 pass loop": "for (int pass = 0; pass < FATTN_NPASS; ++pass)",
    "g1 half-major": "const int half = pass / FATTN_NGRP;",
    "g1 row group": "const int rg = pass % FATTN_NGRP;",
    "d expression, one spelling in both forms": "const int d = 16 * t + fattn_c_col(lane, l) - half * FATTN_TROW;",
    "coverage assert (shipped stage)": "static_assert(FATTN_MROWS * FATTN_TROW == 32 * FATTN_PBLK * FATTN_TBLK,",
    "coverage assert (g1 pass)": "static_assert(FATTN_TRG * FATTN_TROW == 32 * FATTN_LBLK * FATTN_TBLK,",
}


def source_agreement(path: pathlib.Path) -> list:
    text = path.read_text()
    return [f"{name}: `{needle}` is not in {path.name} — this model no longer describes the code that runs"
            for name, needle in SOURCE_SPELLINGS.items() if needle not in text]


_G = parse_geometry(HEADER, OPHEADER)
DH = _G["DH"]
MROWS = _G["FATTN_MROWS"]
TROW = _G["FATTN_TROW"]
TBLK = _G["FATTN_TBLK"]
PBLK = _G["FATTN_PBLK"]          # blocks per lane per half, the shipped 16-row stage
TRG = _G["FATTN_TRG"]            # rows per pass, G1
LBLK = _G["FATTN_LBLK"]          # blocks per lane per G1 pass
NPASS = _G["FATTN_NPASS"]
NGRP = _G["FATTN_NGRP"]
KT = _G["FATTN_KTILES"]          # == FATTN_VTILES == DH // 16

c_row = lambda lane: lane & 15                    # engine/attn_fattn_kernels.h, fattn_c_row
c_col = lambda lane, l: 2 * l + (lane >> 4)       # engine/attn_fattn_kernels.h, fattn_c_col


def shipped():
    """-> (writes, faults): {(row_slot, dim): (lane, t, l)} for one warp's 16 rows x DH dims."""
    writes, bad = {}, []
    for half in range(2):
        stage = {}
        for lane in range(32):
            for t in range(half * (KT // 2), (half + 1) * (KT // 2)):
                for l in range(8):
                    d = 16 * t + c_col(lane, l) - half * TROW
                    idx = c_row(lane) * TROW + d
                    if not 0 <= idx < MROWS * TROW:
                        bad.append(("stage write out of the 16-row stage", half, lane, t, l, idx))
                    if stage.setdefault((c_row(lane), d), (lane, t, l)) != (lane, t, l):
                        bad.append(("two lanes claim one stage cell", half, c_row(lane), d))
        for lane in range(32):
            srow, sblk = lane >> 1, (lane & 1) * PBLK
            for k in range(PBLK):
                sbase = (sblk + k) * TBLK
                dbase = half * TROW + sbase
                for u in range(TBLK):
                    src = (srow, sbase + u)
                    if src not in stage:
                        bad.append(("publish reads an unstaged cell", half, lane, src))
                        continue
                    dest = (srow, dbase + u)
                    if writes.setdefault(dest, stage[src]) != stage[src]:
                        bad.append(("two sources claim one dest", dest, writes[dest], stage[src]))
    return writes, bad


def g1(bug=""):
    """The same maps for the G1 spelling: FATTN_TRG rows per pass, FATTN_LBLK blocks per lane per pass.

    `bug` injects one of three transcriptions of the SHIPPED geometry into the narrower stage — the must-catch
    set, run by `--negative`:
      `lblk`   publish the shipped 8 blocks/lane instead of the 4 an 8-row pass has;
      `row`    publish with the shipped 2-lanes-per-row split (`lane >> 1`) into a pass that holds four;
      `stage`  index the stage by the absolute row-slot instead of the row-within-pass.
    """
    lblk = PBLK if bug == "lblk" else LBLK
    writes, bad = {}, []
    for pass_ in range(NPASS):
        half, rg = pass_ // NGRP, pass_ % NGRP
        row0 = rg * TRG
        stage = {}
        for lane in range(32):
            if not (row0 <= c_row(lane) < row0 + TRG):
                continue
            for t in range(half * (KT // 2), (half + 1) * (KT // 2)):
                for l in range(8):
                    d = 16 * t + c_col(lane, l) - half * TROW
                    r_st = c_row(lane) if bug == "stage" else c_row(lane) - row0
                    idx = r_st * TROW + d                       # fattn_tr_addr(row - row0, d)
                    if not 0 <= idx < TRG * TROW:
                        bad.append(("stage write out of the 8-row stage", pass_, lane, t, l, idx))
                    if stage.setdefault((r_st, d), (lane, t, l)) != (lane, t, l):
                        bad.append(("two lanes claim one stage cell", pass_, r_st, d))
        for lane in range(32):
            srow_g = (lane >> 1) if bug == "row" else (lane >> 2)
            sblk = (lane & 3) * lblk
            srow = row0 + srow_g
            for k in range(lblk):
                sbase = (sblk + k) * TBLK
                dbase = half * TROW + sbase
                for u in range(TBLK):
                    src = (srow_g, sbase + u)
                    if src not in stage:
                        bad.append(("publish reads an unstaged cell", pass_, lane, src))
                        continue
                    dest = (srow, dbase + u)
                    if writes.setdefault(dest, stage[src]) != stage[src]:
                        bad.append(("two sources claim one dest", dest, writes[dest], stage[src]))
    return writes, bad


def _check(writes, bad, want, name):
    fails = []
    if bad:
        fails.append(f"{name}: {len(bad)} bound/aliasing faults, first: {bad[0]}")
    if len(writes) != len(want):
        fails.append(f"{name}: published {len(writes)} of {len(want)} (row, dim) pairs — "
                     f"{len(want) - len(writes)} never written")
    if set(writes) != want:
        fails.append(f"{name}: set mismatch, missing {sorted(want - set(writes))[:3]} "
                     f"outside {sorted(set(writes) - want)[:3]}")
    return fails


def main() -> int:
    src = source_agreement(HEADER)
    if src:
        print("FAIL — the header no longer spells what this file models:")
        for s in src:
            print("  -", s)
        print("This is not a numerics result and not something to widen: re-read the epilogue and re-derive the"
              " model from it. A model of a geometry the engine no longer has cannot adjudicate anything.")
        return 1
    print(f"source   : {HEADER.name} read — {len(SOURCE_SPELLINGS)} index expressions and 2 coverage asserts"
          f" present; geometry read from its own #defines (DH={DH} MROWS={MROWS} TROW={TROW} TBLK={TBLK}"
          f" PBLK={PBLK} TRG={TRG} LBLK={LBLK} NGRP={NGRP} NPASS={NPASS} KTILES={KT})")
    want = {(r, d) for r in range(MROWS) for d in range(DH)}
    a, ba = shipped()
    if "--negative" in sys.argv:
        print("must-catch set — each injection is a wrong transcription of the shipped geometry into the "
              "narrower stage and MUST change the map:")
        missed = 0
        for bug in ("lblk", "row", "stage"):
            b, bb = g1(bug)
            moved = [k for k in (want & set(a) & set(b)) if a[k] != b[k]]
            caught = bool(bb) or set(b) != want or bool(moved)
            print(f"  {bug:<6}: {'CAUGHT' if caught else 'MISSED — this model cannot adjudicate it'}"
                  f"   published {len(b)}/{len(want)}, {len(bb)} bound faults, {len(moved)} destinations"
                  f" re-pointed")
            missed += 0 if caught else 1
        print("NEGATIVE-CONTROL " + ("PASS: every injection changes the map, so the PASS above can disagree."
                                     if not missed else
                                     f"FAIL: {missed} injection(s) went unnoticed."))
        return 1 if missed else 0

    b, bb = g1()
    fails = _check(a, ba, want, "shipped") + _check(b, bb, want, "g1")
    moved = [k for k in (want & set(a) & set(b)) if a[k] != b[k]]
    if moved:
        k = moved[0]
        fails.append(f"identity: {len(moved)} of {len(want & set(a) & set(b))} published floats come from a"
                     f" different accumulator element, first: {k} shipped={a[k]} g1={b[k]}")
    print(f"shipped  : {len(a):>5} published (row, dim) pairs, each from one (lane, t, l)   "
          f"stage = {MROWS} rows x {TROW} F32 = {MROWS * TROW * 4} B/warp x 2 passes")
    print(f"g1       : {len(b):>5} published (row, dim) pairs, each from one (lane, t, l)   "
          f"stage = {TRG} rows x {TROW} F32 = {TRG * TROW * 4} B/warp x {NPASS} passes")
    print(f"identity : {len(want & set(a) & set(b)) - len(moved)}/{len(want & set(a) & set(b))} destinations "
          f"fed by the SAME accumulator element in both forms")
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: every (row-slot, dim) is written exactly once in each form, the two forms agree on the source "
          "element of every destination, and no index leaves its own stage. G1's epilogue moves staging "
          "addresses and nothing else — the bit-identity claim A6-BET's numerics policy owes, taken as arithmetic"
          " over the index functions the shipping header actually contains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
