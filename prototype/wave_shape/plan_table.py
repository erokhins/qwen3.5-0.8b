#!/usr/bin/env python3
"""plan_table.py — a second, independent spelling of the rig's (L,G) plan, diffed against the binary.

Why: every claim in the timed matrix rests on "every cell touches the same bytes, and the booked bytes
are the step's own census". If the plan arithmetic in the binary is wrong, the device witnesses still
pass — they check the kernel against the plan the same binary built, so a wrong plan is self-consistent
(AGENTS: a check whose reference side is derived from the thing under test cannot adjudicate it). This
file re-derives the table from typed constants that came from the ledger, not from the binary, prints
the same fields in the same order, and exits 1 on any line that differs.

Run with no arguments: it runs `wave_shape --print-plan` and diffs. It takes no timing and touches no
device beyond the one process the binary needs for --print-device (which is a properties query).
"""
import re
import subprocess
import sys

HERE = __import__("pathlib").Path(__file__).resolve().parent
BIN = HERE / "wave_shape"

# ---- typed constants. Nothing here is read back from the binary. ----
READ_ELEMENTS = 191_889_408          # 8-byte words; 1,464.0000 MiB
WORD_BYTES = 8
WRITE_DIVISOR = 64                   # the write stream is one word per 64 read words
THREADS = 256
# The step's real read/write census, QUOTED (not measured here): the decode step's read census is
# 1,532.311 MB and its write census 24.409 MB, where MB is 1e6 bytes. Both spellings of the plan convert
# MB -> whole 8-byte words (rounding down to a whole word) and words -> MiB the same way, so the two
# derivations can still be compared digit for digit; a plan that disagreed on the conversion would look
# like a 0.01 % traffic gap and be chased as a bug. The rig's read total is +0.183 % off the census read
# and -1.73 % off the census write; both are printed on PLAN_TOTALS so the gap is on the record.
CENSUS_READ_MB = 1532.311
CENSUS_WRITE_MB = 24.409
CENSUS_READ_EL = round(CENSUS_READ_MB * 1e6 / WORD_BYTES)
CENSUS_WRITE_EL = round(CENSUS_WRITE_MB * 1e6 / WORD_BYTES)
CENSUS_READ_MIB = CENSUS_READ_EL * WORD_BYTES / 2**20
CENSUS_WRITE_MIB = CENSUS_WRITE_EL * WORD_BYTES / 2**20
CUS = 40                             # from KFD simd_count/simd_per_cu, NOT from HIP on this SKU
WAVE = 32
MAT_L = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
MAT_G = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
DOWN_WPR = [1, 2, 4, 8]
DOWN_SPL = [2, 4, 7, 8, 14]
DOWN_ROWS = 1024                     # kHidden: down_proj output rows for ONE token
DOWN_K = 3584                        # kIntermediate: down_proj's reduction length
DOWN_THREADS = 256
DOWN_RPB = 8                         # engine/prototype mlp_down ships 8 rows per block
# The down arms whose geometry the plan must state, in the order the binary prints them.
DOWN_ARMS = ([("down|ship|f32", "ship", 1)] +
             [("down|intra%d|f32" % w, "intra", w) for w in DOWN_WPR] +
             [("down|two%d|f32" % s, "two", s) for s in DOWN_SPL] +
             [("down|intra4|bf16", "intra", -4)])   # -4 marks the bf16-activation twin of wpr=4


def down_arm(tag, form, arg):
    """One PLAN_DOWN line, derived the way dgeom() derives it, spelled independently."""
    act_bf16 = arg < 0
    wpr = -arg if arg < 0 else arg
    if form == "ship":
        block_rows, grid_x, grid_y, launches, slices = DOWN_RPB, DOWN_ROWS // DOWN_RPB, 1, 1, 1
    elif form == "intra":
        block_rows = DOWN_RPB // wpr
        grid_x, grid_y, launches, slices = DOWN_ROWS // block_rows, 1, 1, wpr
    else:
        block_rows, grid_x, grid_y = DOWN_RPB, DOWN_ROWS // DOWN_RPB, arg
        launches, slices = 2, arg
    reduce_blocks = (DOWN_ROWS + 255) // 256 if form == "two" else 0
    blocks_total = grid_x * grid_y + reduce_blocks
    wavefronts = grid_x * grid_y * (DOWN_THREADS // WAVE) + reduce_blocks * (DOWN_THREADS // WAVE)
    stage = grid_x * grid_y * DOWN_K * (2 if act_bf16 else 4)
    # part_bytes books the partial stream TWICE: written by the split launch, read back by the reduce
    # launch. That is demanded traffic, which is what this column is for.
    part = arg * DOWN_ROWS * 4 * 2 if form == "two" else 0
    return ("PLAN_DOWN arm=%s launches=%d grid=%dx%d blocks_total=%d wavefronts_total=%d slices=%d "
            "weight_mib=%.4f stage_kib=%.1f part_kib=%.1f"
            % (tag, launches, grid_x, grid_y, blocks_total, wavefronts, slices,
               DOWN_ROWS * DOWN_K * 2 / 2**20, stage / 1024.0, part / 1024.0))


def expect_lines():
    tot_r = READ_ELEMENTS
    tot_w = READ_ELEMENTS // WRITE_DIVISOR
    out = []
    line = ("PLAN_TOTALS read_el=%d read_mib=%.4f write_el=%d write_mib=%.4f census_read_mib=%.4f "
            "census_write_mib=%.4f dev_read_pct=%+.5f dev_write_pct=%+.5f threads=%d cus=%d wave=%d"
            % (tot_r, tot_r * WORD_BYTES / 2**20, tot_w, tot_w * WORD_BYTES / 2**20,
               CENSUS_READ_MIB, CENSUS_WRITE_MIB,
               100.0 * (tot_r / CENSUS_READ_EL - 1.0),
               100.0 * (tot_w / CENSUS_WRITE_EL - 1.0), THREADS, CUS, WAVE))
    out.append(line)
    n = nre = 0
    for L in MAT_L:
        for G in MAT_G:
            B = L * G
            n += 1
            ok = (tot_r % B == 0 and tot_w % B == 0 and G * THREADS <= tot_r and G <= 65535)
            if ok:
                nre += 1
                out.append("PLAN_CELL L=%d G=%d B=%d blocks_per_launch=%d per_launch_mib=%.6f "
                           "per_block_kib=%.4f read_mib=%.4f write_mib=%.4f demand_mib=%.4f "
                           "wavefronts=%d wavefronts_per_cu=%.2f reachable=1"
                           % (L, G, B, G, tot_r * WORD_BYTES / 2**20 / L,
                              tot_r * WORD_BYTES / 2**10 / B,
                              tot_r * WORD_BYTES / 2**20, tot_w * WORD_BYTES / 2**20,
                              (tot_r + tot_w) * WORD_BYTES / 2**20,
                              B * THREADS // WAVE, B * THREADS // WAVE / CUS))
            else:
                # The binary collapses every unreachability into the divisibility reason; if a cell ever
                # fails a different rule the two spellings will disagree, which is the point.
                why = ("block_total_does_not_divide_the_pinned_bytes" if (tot_r % B or tot_w % B) else
                       "one_launch_would_have_more_blocks_than_bytes" if G * THREADS > tot_r else
                       "grid_above_65535")
                out.append("PLAN_CELL L=%d G=%d B=%d blocks_per_launch=%d reachable=0 reason=%s"
                           % (L, G, B, G, why))
    out.append("PLAN_CELLS total=%d reachable=%d" % (n, nre))
    for tag, form, arg in DOWN_ARMS:
        out.append(down_arm(tag, form, arg))
    return out


def main():
    got = subprocess.run([str(BIN), "--print-plan"], capture_output=True, text=True)
    if got.returncode != 0:
        print("ABORT: %s --print-plan exited %d\n%s" % (BIN, got.returncode, got.stdout + got.stderr))
        return 2
    have = [ln.rstrip() for ln in got.stdout.splitlines() if ln.strip()]
    want = expect_lines()
    # Compare the two families of line separately: a count mismatch is reported as a population change,
    # not as a wall of diffs.
    bad = 0
    for tag in ("PLAN_TOTALS", "PLAN_CELL", "PLAN_CELLS", "PLAN_DOWN"):
        h = [x for x in have if x.startswith(tag)]
        w = [x for x in want if x.startswith(tag)]
        if len(h) != len(w):
            print("POPULATION %s: the binary prints %d, this re-derivation expects %d" % (tag, len(h), len(w)))
            bad += 1
            continue
        for a, b in zip(h, w):
            if a != b:
                bad += 1
                print("DIFF\n  binary: %s\n  python: %s" % (a, b))
    print("PLAN-DIFF %s lines_compared=%d mismatches=%d"
          % ("ok" if bad == 0 else "FAILED", len(want), bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
