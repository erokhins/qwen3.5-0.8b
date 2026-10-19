#!/usr/bin/env python3
"""A6 Step 1's witness for G1's epilogue: does an 8-row transpose stage publish the SAME floats the shipped
16-row stage does, and does it publish each of them exactly once? CPU only — no device, no timing.

WHY THIS EXISTS. G1's bit-identity claim (A6-BET.md "Numerics", A5-GEOMETRY.md §3.2) rests on one sentence:
*"the epilogue change moves staging addresses only, never the published values"*. That is an argument, and this
tree's trap list is unambiguous about arguments of that shape — a check whose reference side is derived from
the thing under test cannot adjudicate it, and an aggregate that does not name its denominator cannot fail. So
this file enumerates both epilogues' writes as arithmetic — no kernel, no floats, just the index functions
transcribed from the source that owns them (`engine/attn_fattn_kernels.h:280-281` for `fattn_c_row` /
`fattn_c_col`) — and asks three questions each of which can be wrong:

  1. COVERAGE: is every (row-slot, dim) of the warp's published half-row written EXACTLY once, in each form?
     The shipped file's own defect #2 on `fattn_transpose_out` — 16 dims published per row, 240 never written
     — is invisible to a bound check and visible to exactly this one.
  2. IDENTITY: when a (row-slot, dim) is written in both forms, does the same accumulator element
     `(lane, t, l)` — the same `out.t[t][l]` of the same lane — land in it? If yes, the published bytes are the
     shipped ones whatever the stage layout, and no numerics band re-opens.
  3. BOUNDS: is every stage index below the stage's declared size, in each form? G1's stage is half as deep, so
     an index derived from the old geometry walks off it into the next warp's stage — the §19/§20.1 aliasing
     class, caught here instead of on a device.

`--negative` runs the must-catch set: three ways the G1 mapping could have been transcribed from the shipped one
and would be WRONG. Each must change the map, or the PASS above is decoration (AGENTS: "a negative control with
an empty must-catch set is not a control").

Run:  python3 prototype/attn_percost/a6_epi8_model.py [--negative]
Exit 0 = the three claims hold for all 16 row-slots x 256 dims of a warp.
"""
import sys

DH = 256
MROWS = 16
TROW = DH // 2          # FATTN_TROW
TBLK = 8                # FATTN_TBLK
PBLK = 8                # FATTN_PBLK — blocks per lane per half, the shipped 16-row stage
TRG = 8                 # FATTN_TRG — rows per pass, G1
KT = DH // 16           # FATTN_KTILES == FATTN_VTILES == 16

c_row = lambda lane: lane & 15                    # engine/attn_fattn_kernels.h:280
c_col = lambda lane, l: 2 * l + (lane >> 4)       # engine/attn_fattn_kernels.h:281


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
    lblk = PBLK if bug == "lblk" else (TRG * (TROW // TBLK)) // 32      # 4 when correct
    writes, bad = {}, []
    for pass_ in range(2 * (MROWS // TRG)):
        half, rg = pass_ // (MROWS // TRG), pass_ % (MROWS // TRG)
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
        print("NEGATIVE-CONTROL " + ("PASS: every injection changes the map, so the PASS below can disagree."
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
          f"stage = {TRG} rows x {TROW} F32 = {TRG * TROW * 4} B/warp x 4 passes")
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
          " over the index functions instead of as prose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
