#!/usr/bin/env python3
"""A2 census: attention's demanded work and its per-launch cost, derived from GEOMETRY only.

Why this exists and why it is not another ISA census. The census tools that exist
(`tools/kernel_isa_census.py`, `prototype/gdn_walk/g3_isa_bill.py`,
`prototype/attn_wmma/fattn_ship_census.py`) answer "what did the compiler emit". This one answers the
other half of G3's method (`prototype/gdn_walk/g3_work_table.py`): what does the geometry DEMAND, before
any instruction count, and what would that demand cost at a measured ceiling. Every input is either
(a) a constant read out of a header at run time by regex, so a geometry edit moves the output, (b) a
traced ms figure from `engine/performance-status.md` named in its own label, or (c) a measured roof from
`prototype/issue_rate_probe/results.md` §1. Nothing here launches, times, or touches a device.

The staging question is answered the way `prototype/attn_tile_model.py:stage_copies` answers it — read
the LOOP, do not book a re-use factor — but the shape is different here and so is the reader: `attn_fattn_wave`
fills ONE block-shared `s_kv` strided over the block's warps, and every warp of the block then reads that
stage for all `FATTN_MROWS` of its own rows.

UNIT DISCIPLINE (this is the second version; the first printed 5.6e12 % of a roof). Every roof is held in
SI units (FLOP/s, B/s, slots/s) and every rate is computed in SI, and `pct_of` refuses to print a
percentage-of-a-roof above 1000 %. A census that prints 5e12 % and keeps going is a check that cannot fail,
which is the same defect as the empty-parse gates AGENTS.md records, pointing the other way.

Three different K/V byte denominators exist and they are NOT interchangeable, so each is named in its own
column rather than being called "unique":
  * `demand1x` — one fetch per (row, lane, kv-head): the zero-cross-row-reuse demand, i.e. what the LSU would
    see if a block did not share a stage across its `FATTN_WARPS_PER_BLOCK * FATTN_MROWS` rows.
  * `staged`   — what the kernel's own loops fetch: `MLANES` lanes per (block, round), summed over rounds.
  * `uniq`     — the lower bound: each visible lane of a body read once, summed over bodies.
Read literally the first two differ by ~48x, which IS the staging lever, and the third sits under both.

    python3 prototype/attn_percost/a2_work_table.py [--widths 2048,4096,8192,16384,32768]
"""
import argparse
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROWS_PER_BODY = 512          # engine/decode_step.h:49 kMaxWaveRows — a wave is at most 512 rows

# Roofs, all in SI. Sources: prototype/issue_rate_probe/results.md §1 (T·FLOP/s column, which counts an FMA
# as 2 FLOP — the same convention as the `4*DH FLOP per touch` demand below),
# devices/strix-halo-memory-bandwidth.md (237 GB/s DRAM plateau; 918 GB/s at an 18 MiB working set).
WMMA = 52.63e12
DOT2 = 23.28e12
SCALAR = 7.30e12
LSU = 918e9
DRAM = 237e9
SLOT_S = 80 * 2.894e9        # wavefront issue slots/s: 80 SIMD x 1 slot x 2.894 GHz


def pct_of(part_si, whole_si, what):
    """A percentage of a roof, with the sanity bound the first version lacked.

    Above 1000 % the two operands cannot be the same kind of quantity (a unit slip of 1e12 was the original
    bug), so it is a hard stop rather than a warning — a census line that prints a plausible-looking absurdity
    gets tabulated by the next reader anyway.
    """
    pct = 100.0 * part_si / whole_si
    if abs(pct) > 1000.0:
        raise SystemExit("REFUSE: %s = %.4g %% of its roof. A percentage-of-a-roof over 1000 %% means the "
                         "numerator and the denominator are not in the same units (the first version of this "
                         "file divided a TFLOP/s rate by an FLOP/s roof). Fix the units; do not widen the cap."
                         % (what, pct))
    return pct


def header_raw(path, names):
    """`#define NAME value` out of a header, unresolved. Resolution is deliberately deferred to
    `resolve` because the constants are split across two headers on purpose
    (`DH` lives in `prototype/attn_decode/attn_decode_kernels.h`, `FATTN_KTILES (DH / 16)` lives in
    `engine/attn_fattn_kernels.h`), so a per-file evaluator cannot see its own operands."""
    text = open(os.path.join(REPO, path)).read()
    raw = {}
    for n in names:
        m = re.search(r"^#define\s+%s\s+(.+?)\s*(?://.*)?$" % re.escape(n), text, re.M)
        if not m:
            raise SystemExit("REFUSE: %s does not #define %s" % (path, n))
        raw[n] = m.group(1).strip()
    return raw, text


def resolve(raw):
    out = {}
    for _ in range(len(raw)):
        progressed = False
        for n, v in raw.items():
            if n in out:
                continue
            try:
                out[n] = int(eval(v, {"__builtins__": {}}, dict(out)))
                progressed = True
            except Exception:
                pass
        if not progressed:
            break
    missing = [n for n in raw if n not in out]
    if missing:
        raise SystemExit("REFUSE: cannot evaluate %s" % missing)
    return out


def stage_loops(text):
    """The staging loops the header actually spells, as (induction var, lo, bound var, stride var).

    AGENTS.md's §20.5 row is a booked 4x re-use that the loop shape made a 1x, so the shape is read out
    and printed rather than asserted from a write-up.
    """
    got = re.findall(r"for\s*\(\s*int\s+(slot|kv|rr|qr)\s*=\s*(\w+)\s*;\s*\1\s*<\s*(\w+)\s*;\s*\1\s*\+=\s*(\w+)\)",
                     text)
    if not got:
        raise SystemExit("REFUSE: no stage-fill loop found — the header stopped spelling staging in a "
                         "shape this census can read, which is a FAILURE of the census, not a free pass")
    return got


def bodies(total_lanes):
    """lvalid of each wave body, exactly as the engine advances a prompt: body 0 starts at 1 and each
    body's rows advance one position and one visible lane each."""
    out, lvalid = [], 1
    while lvalid <= total_lanes:
        out.append(lvalid)
        lvalid += ROWS_PER_BODY
    return out


def split_of(lv):
    """`split_for(lanes_last)` of `engine/kernels_attn.cpp:86`, with lanes_last = lv + ROWS_PER_BODY - 1."""
    return (lv + ROWS_PER_BODY - 1 + MAXSC - 1) // MAXSC


def body_plan(lv, split):
    """Replay fattn_row_of/fattn_warp_min/max per (lane-slice y, row-block-of-blocks rb).

    Returns [(blocks, rounds, useful_cells, issued_cells)] for one body: `blocks` counts the
    (head, y, rb) launches that reach the round loop, `rounds` the block-uniform round count of each,
    `useful_cells` the (row, lane) pairs the mask lets through and `issued_cells` the MR x ML the round
    computes whatever the mask says. Both are per WARP, so the head axis is not multiplied in here.

    Cross-check, and it is not a small one: `blocks * NQ` reproduces the traced grid of every geometry row
    of `attn_fattn_wave` in the g4ab trace — grid_threads (768, split, 11) over a 96-thread block is
    exactly 8 heads x split x 11 row-tiles.
    """
    rt = (ROWS_PER_BODY + NROWS_BLK - 1) // NROWS_BLK
    tot = [0, 0, 0, 0]
    for y in range(split):
        for rb in range(rt):
            lo, hi = 1 << 30, 0
            useful = 0
            rows_here = 0
            for w in range(NW):
                rs = [rb * NROWS_BLK + w * MR + r for r in range(MR)
                      if rb * NROWS_BLK + w * MR + r < ROWS_PER_BODY]
                if not rs:
                    continue
                rows_here += 1
                n = [lv + r for r in rs]                       # lanes_of(a, row) with neg=NONE, lvalid=lv
                s = [(x + split - 1) // split for x in n]      # span
                blo = [y * v for v in s]
                bhi = [min(x, (y + 1) * v) for x, v in zip(n, s)]
                for i in range(len(rs)):
                    if bhi[i] > blo[i]:
                        useful += bhi[i] - blo[i]
                        lo, hi = min(lo, blo[i]), max(hi, bhi[i])
            if hi <= lo:
                continue
            rounds = (hi - lo + 15) // 16
            tot[0] += 1                       # one (head, y, rb) block reaches the round loop
            tot[1] += rounds                  # ... and runs `rounds` of them
            tot[2] += useful                  # summed over the rows of all warps
            tot[3] += rounds * MR * 16 * rows_here
    return tot


def work(W):
    """Everything the geometry determines for one prompt width, per LAYER unless the name says pass."""
    bs = bodies(W)
    visits = sum(sum(lv + r for r in range(ROWS_PER_BODY)) for lv in bs)
    covered_visits = sum(sum(lv + r for r in range(ROWS_PER_BODY)) for lv in bs if split_of(lv) >= 2)
    rows_covered = sum(ROWS_PER_BODY for lv in bs if split_of(lv) >= 2)
    blocks = rounds = 0
    for lv in bs:
        if split_of(lv) < 2:
            continue
        b, r, _, _ = body_plan(lv, split_of(lv))
        blocks += b * NQ
        rounds += r * NQ
    uniq_lanes = sum(lv + ROWS_PER_BODY - 1 for lv in bs)     # lanes each body must read at least once
    return {
        "bodies": bs, "n": len(bs), "visits": visits, "covered_visits": covered_visits,
        "rows_covered": rows_covered, "blocks": blocks, "rounds": rounds,
        "uniq_lanes": uniq_lanes,
        "mac_layer": visits * NQ * 2 * DH,                    # 2*DH MAC per (row, lane, head) touch
        "flop_pass": 2 * visits * NQ * 2 * DH * LAYERS,
        "flop_pass_covered": 2 * covered_visits * NQ * 2 * DH * LAYERS,
        "demand1x_layer": visits * NQ * 2 * DH * 2,   # per (row, HEAD, lane): its KV group's K+V, zero reuse
        "uniq_layer": uniq_lanes * NKV * 2 * DH * 2,  # per lane-instance, both groups of the layer
        "staged_layer": rounds * ML * DH * 2 * 2,             # MLANES lanes per (block, round), K and V
        "part_layer": sum(ROWS_PER_BODY * NQ * split_of(lv) * (DH + 2) * 4
                          for lv in bs if split_of(lv) >= 2),
        # attn_combine is gated on split > 1 (engine/kernels_attn.cpp:248, bench §41), so it runs on the
        # split bodies only: 7 bodies x 6 layers = 42 launches/pass at 4,096 lanes, which the g4ab trace
        # confirms by instance count (attn_prep 48 / slim 48 / fattn 42 / combine 42).
        "combine_wf": NQ * ROWS_PER_BODY * COMBINE_WF * sum(1 for lv in bs if split_of(lv) >= 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="2048,4096,8192,16384,32768")
    ap.add_argument("--selftest-pct", action="store_true",
                    help="witness that the unit guard cannot pass vacuously, then exit")
    a = ap.parse_args()

    if a.selftest_pct:
        # The bug this guard was written for: a TFLOP/s rate divided by an FLOP/s roof, which prints
        # 5,644,434,225,721 % and keeps going. The guard must refuse THAT shape, and must not refuse the
        # legal >100 % case (llama's attention is at 109 % of the scalar roof).
        try:
            pct_of(2.971e12, WMMA / 1e12, "selftest: the original unit slip")
        except SystemExit as exc:
            print("SELFTEST PASS: the unit slip is refused — %s" % str(exc).split(".")[0])
        else:
            raise SystemExit("SELFTEST FAIL: a 1e12 unit slip passed the guard")
        if abs(pct_of(7.985e12, SCALAR, "selftest: llama vs scalar") - 109.4) > 0.2:
            raise SystemExit("SELFTEST FAIL: a legal >100 % ratio is mis-computed")
        print("SELFTEST PASS: a legal 109 %% of roof still prints (%.1f %%)"
              % pct_of(7.985e12, SCALAR, "selftest: llama vs scalar"))
        return

    global MAXSC, NQ, NKV, DH, MR, ML, KT, VT, NW, NROWS_BLK, COMBINE_WF, LAYERS
    fat, fattn_text = header_raw("engine/attn_fattn_kernels.h",
                                 ["FATTN_MROWS", "FATTN_MLANES", "FATTN_KTILES", "FATTN_VTILES",
                                  "FATTN_WARPS", "FATTN_THREADS", "FATTN_TROW"])
    op, _ = header_raw("prototype/attn_decode/attn_decode_kernels.h",
                       ["N_Q", "N_KV", "GQA", "DH", "MAXSC", "THREADS", "FATTN_WARPS_PER_BLOCK"])
    d = resolve(dict(fat, **op))
    MR, ML, KT, VT = d["FATTN_MROWS"], d["FATTN_MLANES"], d["FATTN_KTILES"], d["FATTN_VTILES"]
    NW = d["FATTN_WARPS"]
    NROWS_BLK = d["FATTN_WARPS_PER_BLOCK"] * MR
    NQ, NKV, DH, MAXSC = d["N_Q"], d["N_KV"], d["DH"], d["MAXSC"]
    COMBINE_WF = d["THREADS"] // 32
    LAYERS = 6              # attention layers per pass: 48 attn_prep sites / 8 bodies (bench §38/§41)
    LDS = NW * MR * DH * 2 + ML * DH * 2
    assert LDS == 32768 and NROWS_BLK == 48 and MR == ML == 16

    print("# engine geometry, read from headers (not typed):")
    print("#   N_Q=%d N_KV=%d GQA=%d DH=%d MAXSC=%d attention layers=%d wave rows/body=%d"
          % (NQ, NKV, d["GQA"], DH, MAXSC, LAYERS, ROWS_PER_BODY))
    print("#   attn_fattn_wave MROWS=%d MLANES=%d KTILES=%d VTILES=%d WARPS=%d THREADS=%d rows/block=%d"
          % (MR, ML, KT, VT, NW, d["FATTN_THREADS"], NROWS_BLK))
    print("#   LDS/block = %d (s_q, %d warps x 8192) + %d (s_kv) = %d B -> %d blocks/CU at KFD's 65536 B"
          % (NW * MR * DH * 2, NW, ML * DH * 2, LDS, 65536 // LDS))
    print("#     = %d wavefronts/CU = %.2f wf/SIMD (KFD simd_count 80, simd_per_cu 2 -> 40 CU; never HIP)"
          % ((65536 // LDS) * NW, (65536 // LDS) * NW / 2.0))
    print("#   registers, as read rather than asserted: clang's `.vgpr_count` is 178 with 0 spill and 0")
    print("#     private segment (prototype/attn_wmma/fattn_residency_read.py on the carved object) while")
    print("#     rocprofv3's VGPR_Count column for the same kernel says 248 — a grant/rounding difference")
    print("#     this census does NOT resolve and does not need: at 96 threads either number allows 3 blocks")
    print("#     only if the register file per CU is >= 62.5K, and that file size is UNKNOWN on this part")
    print("#     (fattn_regfile_ladder.md §3), so the residency claim rests on LDS alone. The 196,608 in")
    print("#     engine/attn_fattn_kernels.h:736-747 is LLVM's occupancy *hint* budget (196,608/1024 = 192")
    print("#     VGPR/thread without __launch_bounds__), not a measured register file, and the first version")
    print("#     of this file quoted it as one. LDS is the binding limit either way: 2 blocks/CU.")
    print("#   staging loops read out of the header (var, lo, bound, stride):")
    for var, lo, hi, step in stage_loops(fattn_text):
        print("#     for (int %s = %s; %s < %s; %s += %s)" % (var, lo, var, hi, var, step))
    print("#     -> s_kv is filled ONCE per (block, round) across the block's %d warps, and all %d warps"
          % (NW, NW))
    print("#        then read it, each for its own %d rows: K/V global re-use per block = %dx (rows in a"
          % (MR, NROWS_BLK))
    print("#        block), which is the only lever of this kind the form has left.")
    print()

    W0 = 4096
    ws = {W: work(W) for W in [int(x) for x in a.widths.split(",")]}
    z = ws[W0]

    print("# =============================== 1. WORK DEMANDED (geometry only) ==========================")
    print("# a (row, lane, head) touch costs 2*DH MAC (score dot + P*V) = 4*DH FLOP.")
    print("# K/V byte columns are PER LAYER, all x%d for a pass. demand1x = one fetch per (row,lane,kvhead);")
    print("# uniq = each visible lane of a body read once, summed over bodies (the DRAM lower bound).")
    print("%-9s %7s %9s %16s %16s %16s %18s %16s"
          % ("lanes", "bodies", "split", "row-lane visits", "MAC/layer", "TFLOP/pass",
             "K+V demand1x B/layer", "K+V uniq B/layer"))
    for W, t in ws.items():
        splits = sorted(split_of(lv) for lv in t["bodies"])
        print("%-9d %7d %9s %16s %16s %16.4f %18s %16s"
              % (W, t["n"], "%d..%d" % (splits[0], splits[-1]),
                 format(t["visits"], ","), format(t["mac_layer"], ","), t["flop_pass"] / 1e12,
                 format(t["demand1x_layer"], ","), format(t["uniq_layer"], ",")))
    print("# NOTE the visits grow ~W^2/2: doubling the prompt quadruples this row (3.44e10 MAC/layer at")
    print("# 4,096 -> 5.50e11 at 16,384 -> 2.20e12 at 32,768), which is the `b` term of S46's fit. The uniq")
    print("# column grows the same way (it is sum_i 512*(i+1)), so demand1x/uniq is a CONSTANT in W,")
    print("# %.0fx at every width: the cross-row sharing is a property of ROWS_PER_BODY, not of the width."
          % (z["demand1x_layer"] / float(z["uniq_layer"])))
    print()

    print("# =============================== 2. ROUNDS RUN vs CELLS WANTED =============================")
    print("# per body: blocks that reach the round loop, block-uniform rounds, and the fraction of the")
    print("# MR x ML cells a round actually gets to keep (the union-of-%d-rows range masks nothing out)."
          % NROWS_BLK)
    print("%-9s %5s %6s %8s %10s %12s %14s %9s"
          % ("lanes", "body", "split", "rtiles", "rounds", "blocks*8h", "useful cells", "waste %"))
    for W, t in ws.items():
        for bi, lv in enumerate(t["bodies"]):
            if split_of(lv) < 2:
                continue
            blocks, rounds, useful, issued = body_plan(lv, split_of(lv))
            print("%-9d %5d %6d %8d %10s %12s %14s %9.1f"
                  % (W, bi, split_of(lv), (ROWS_PER_BODY + NROWS_BLK - 1) // NROWS_BLK,
                     format(rounds, ","), format(blocks * NQ, ","), format(useful * NQ, ","),
                     100.0 * (1.0 - useful / float(issued))))
    print("# waste is 4-7 % at every body of every width: the tile-shape candidate for any rate gap is")
    print("# 1.07x and is dead on magnitude. Rows below MAXSC are refused by this kernel by design")
    print("# (engine/attn_fattn_kernels.h COVERAGE), so body 0 runs on attn_slim_wave: that is")
    print("# %.1f %% of rows and %.1f %% of the lane-work, which reproduces S41's 87.5/98.4 from geometry."
          % (100.0 * z["rows_covered"] / (z["n"] * ROWS_PER_BODY),
             100.0 * z["covered_visits"] / z["visits"]))
    print()

    print("# =============================== 3. CEILINGS vs MEASURED TIME ==============================")
    tr = {"attn_fattn_wave": 138.83, "attn_combine": 6.15, "attn_slim_wave": 6.04, "attn_prep": 4.69}
    fam = sum(tr.values())
    print("# traced ms/pass, g4ab trace (tmp/gdn_walk/g4ab/trace_reg16), passes derived from 48 attn_prep")
    print("# at 48 sites; re-read per-kernel with prototype/attn_percost/a2_trace_rows.py, whose geometry")
    print("# rows agree with section 2's block counts.")
    for k, v in tr.items():
        print("#   %-16s %7.2f ms/pass  (%4.1f %% of family)" % (k, v, 100 * v / fam))
    print("#   family total %.2f ms traced = S44's 155.71 ms attention phase; S41's UNTRACED `attn`"
          % fam)
    print("#   family row is 133 ms (bench 41), i.e. tracing inflates it by %.0f %%. Quote ratios."
          % (100 * (fam / 133.0 - 1.0)))
    print("# demanded math per pass at %d lanes: %.4f TFLOP whole-family, %.4f TFLOP on the rows fattn owns"
          % (W0, z["flop_pass"] / 1e12, z["flop_pass_covered"] / 1e12))
    # Each rate pairs ITS OWN numerator with ITS OWN denominator. The first version divided the whole
    # family's FLOP by the fattn kernel's ms, which overstates fattn by the ~1.6 % body-0 work it never runs.
    rows = [("fattn row / covered rows", tr["attn_fattn_wave"], z["flop_pass_covered"]),
            ("untraced family", 133.0, z["flop_pass"]),
            ("frozen VALU form", 263.12, z["flop_pass"])]
    for nm, ms, flop in rows:
        tf = flop / (ms / 1e3)                     # FLOP/s
        print("#   %-24s %7.2f ms -> %6.3f TFLOP/s = %5.2f %% of the WMMA roof | %5.1f %% of the scalar "
              "roof | %5.1f %% of dot2"
              % (nm, ms, tf / 1e12, pct_of(tf, WMMA, nm + " vs WMMA"),
                 pct_of(tf, SCALAR, nm + " vs scalar"), pct_of(tf, DOT2, nm + " vs dot2")))
    print("#   FLOORS for the same math: WMMA roof %.2f ms | 50 %% of WMMA %.2f ms | 25 %% %.2f ms | "
          "dot2 %.2f ms | scalar %.2f ms"
          % (z["flop_pass"] / WMMA * 1e3, z["flop_pass"] / (.5 * WMMA) * 1e3,
             z["flop_pass"] / (.25 * WMMA) * 1e3, z["flop_pass"] / DOT2 * 1e3,
             z["flop_pass"] / SCALAR * 1e3))
    ll = 51.65                                     # §38's traced flash_attn_ext_f16 ms/pass
    tf_ll = z["flop_pass"] / (ll / 1e3)
    print("# llama.cpp's flash_attn_ext_f16 does the SAME math in %.2f ms traced (S38) = %.3f TFLOP/s"
          % (ll, tf_ll / 1e12))
    print("#   = %.1f %% of the WMMA roof and %.1f %% of the SCALAR roof, so it cannot be a scalar kernel."
          % (pct_of(tf_ll, WMMA, "llama vs WMMA"), pct_of(tf_ll, SCALAR, "llama vs scalar")))
    print("#   Both sides are matrix-class: 17 v_wmma sites in attn_fattn_wave (a2_round_bill.py on the")
    print("#   carved engine object) and v_wmma_f32_16x16x16_f16 x1584 + v_wmma_f16_16x16x16_f16 x1008 in")
    print("#   the carved flash_attn_ext_f16 object. The class is therefore NOT the gap; S26's \"parity is a")
    print("#   matrix-class question\" is closed by having reached the class, not by paying it.")
    print("#   our ratio to llama on this kernel row: %.3fx (S38's family row said 2.65x on 136.83 vs"
          % (tr["attn_fattn_wave"] / ll))
    print("#   51.65; the family denominator is %.3fx now)." % (fam / ll))
    print()

    print("# =============================== 4. K/V TRAFFIC ===========================================")
    print("# staged = %d lanes x DH x 2 B x (K,V) per (block, round), summed over the kernel's own rounds;"
          % ML)
    print("# the first version booked %d lanes per block instead, which is the split window, not the loop."
          % MAXSC)
    for nm, div in (("per LAYER", 1), ("per PASS", LAYERS)):
        print("#   %s: demand1x %s B | staged %s B | uniq %s B"
              % (nm, format(z["demand1x_layer"] * div, ","), format(z["staged_layer"] * div, ","),
                 format(z["uniq_layer"] * div, ",")))
    print("#   staging cuts LSU-level demand %.1fx below the no-reuse figure — the loop shape books %dx (rows"
          % (z["demand1x_layer"] / float(z["staged_layer"]), NROWS_BLK))
    print("#   per block sharing one s_kv), so the shape does about what it claims; the residual is edge")
    print("#   rounds and the partly-masked tail row-tile. The staged column is still %.1fx the unique bytes:"
          % (z["staged_layer"] / float(z["uniq_layer"])))
    print("#   each body re-reads its prefix once per 48-row tile, and the 4 query heads of a KV group are 4")
    print("#   separate blocks, so cross-head re-use in global is zero.")
    for nm, ms in (("traced fattn", tr["attn_fattn_wave"]), ("untraced family", 133.0)):
        st = z["staged_layer"] * LAYERS / (ms / 1e3)
        uq = z["uniq_layer"] * LAYERS / (ms / 1e3)
        d1 = z["demand1x_layer"] * LAYERS / (ms / 1e3)
        print("#   %s at %.2f ms: staged %.0f GB/s (%.0f %% of the 918 GB/s cache roof) | uniq %.2f GB/s "
              "(%.1f %% of it, %.0f %% of the 237 GB/s DRAM roof) | demand1x %.0f GB/s"
              % (nm, ms, st / 1e9, pct_of(st, LSU, "staged vs cache"), uq / 1e9,
                 100 * uq / LSU, pct_of(uq, DRAM, "uniq vs DRAM"), d1 / 1e9))
    print("# AGENTS.md's D21 caveat applies and now bites the other way. uniq K+V per pass is %.2f GB = %.1fx"
          % (z["uniq_layer"] * LAYERS / 1e9, z["uniq_layer"] * LAYERS / (18 * 1024 * 1024)))
    print("#   the 18 MiB working set the 918 GB/s roof was measured with, and the staged demand is %.2f GB"
          % (z["staged_layer"] * LAYERS / 1e9))
    print("#   per pass = %.0fx that footprint, so nearly all of it must be L1/L2-served and the GB/s columns"
          % (z["staged_layer"] / float(z["uniq_layer"])))
    print("#   are a census, not a wall. demand1x at %.0f GB/s exceeds the 237 GB/s DRAM roof by construction,"
          % (z["demand1x_layer"] * LAYERS / (tr["attn_fattn_wave"] / 1e3) / 1e9))
    print("#   which is exactly the point of naming the three columns apart: bytes demanded are not bytes")
    print("#   fetched are not bytes from DRAM (the §26/D21 lesson, read from the geometry side this time).")
    print()

    print("# =============================== 5. SPLIT / COMBINE SCALING ================================")
    print("# part slot = (2 + DH) F32 = %d B per (row, head, split-block); combine grid (N_Q, rows) x 256."
          % ((DH + 2) * 4))
    print("# kAttnMaxSplit=64 -> attn_part_floats(512) = %s B per scratch instance (S39: 258.00 MiB)."
          % format(512 * NQ * 64 * (DH + 2) * 4, ","))
    print("# Columns are PER PASS (x%d layers). The first version printed per-layer bytes and divided them"
          % LAYERS)
    print("# by per-layer K/V demand, which is why it could call the traffic \"1.6 %\" and stop there.")
    print("%-9s %10s %14s %14s %16s %14s %14s %12s"
          % ("lanes", "maxsplit", "part WR B", "part RD B", "wr+rd B", "vs demand1x", "vs uniq", "vs staged"))
    for W, t in ws.items():
        wr = t["part_layer"] * LAYERS
        print("%-9d %10d %14s %14s %16s %14.3f %14.2f %12.3f"
              % (W, max(split_of(lv) for lv in t["bodies"]), format(wr, ","), format(wr, ","),
                 format(2 * wr, ","), 2 * wr / float(t["demand1x_layer"] * LAYERS),
                 2 * wr / float(t["uniq_layer"] * LAYERS), 2 * wr / float(t["staged_layer"] * LAYERS)))
    print("# Every one of those three ratios is CONSTANT in W (part traffic is rows x split = rows x")
    print("# lanes/MAXSC, and so is the K/V demand), so the split path scales exactly like the attention")
    print("# math's own quadratic term: it is part of `b`, it is not a separate blow-up, and it cannot by")
    print("# itself explain a ratio between two engines' `b` unless llama pays no such traffic at all —")
    print("# which it does not: flash_attn_ext_f16 writes its O tile once and keeps the online softmax in")
    print("# registers, so on OUR side of the comparison the whole round-trip is differential.")
    print("# combine traced: %.2f ms/pass = %.1f %% of the family (S41's 'satellites are 3 %%' holds),"
          % (tr["attn_combine"], 100 * tr["attn_combine"] / fam))
    print("#   and attn_combine is gated on split > 1, so it runs on %d of the %d bodies per layer"
          % (sum(1 for lv in z["bodies"] if split_of(lv) >= 2), z["n"]))
    print("#   (%d launches/pass, which is the 42 the g4ab trace counts by instance) and carries the" % (z["n"] and sum(1 for lv in z["bodies"] if split_of(lv) >= 2) * LAYERS))
    print("#   merged-row write for the covered rows only: %s x N_Q x DH x 4 B = %s B/pass. The uncovered"
          % (format(z["rows_covered"], ","), format(z["rows_covered"] * NQ * DH * 4 * LAYERS, ",")))
    print("#   body-0 rows are finalised by attn_slim_wave and never reach this scratch at all.")
    print("# Time sizing of the whole round-trip, DERIVED not measured (bytes / 237 GB/s, NOT a timing):")
    wr = z["part_layer"] * LAYERS
    comb_w = z["rows_covered"] * NQ * DH * 4 * LAYERS
    print("#   writes %s B + reads %s B + merged write %s B = %.3f GB/pass; at the 237 GB/s DRAM roof that"
          % (format(wr, ","), format(wr, ","), format(comb_w, ","), (2 * wr + comb_w) / 1e9))
    print("#   is %.0f ms/pass of pure traffic, i.e. %.1f %% of the %.2f ms traced family row"
          % ((2 * wr + comb_w) / DRAM * 1e3, 100 * ((2 * wr + comb_w) / DRAM * 1e3) / fam, fam))
    for W in sorted(ws):
        t = ws[W]
        w2 = t["part_layer"] * LAYERS
        gb = (2 * w2 + t["rows_covered"] * NQ * DH * 4 * LAYERS) / 1e9
        print("#   %-6d lanes: %.2f GB round-trip = %6.0f ms at the DRAM roof"
              % (W, gb, gb * 1e12 / DRAM))
    print("#   Compare that last column with §46's excess quadratic term at the same width (census §4).")


if __name__ == "__main__":
    main()
