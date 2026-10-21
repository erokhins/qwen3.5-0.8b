#!/usr/bin/env bash
# a6_build.sh — A6 Step 1/2 compile-and-read: build the G1 and G0 variants beside the shipped form, CPU only.
#
# NOTHING HERE LAUNCHES A KERNEL AND NOTHING HERE TAKES A TIMING. It is `hipcc --save-temps -c` over a COPY, and
# the numbers it produces are note-block fields (`vgpr_count`, `vgpr_spill_count`, `private_segment_fixed_size`,
# `group_segment_fixed_size`), which are facts about the object file. `AGENTS.md`'s GPU rule is about timing; a
# compile does not touch the device at all.
#
# THE QUARANTINE, copied from `prototype/attn_percost/a4_l1_build.sh` and load-bearing for the same reason:
# every side is compiled from a copy of the SAME three files (`engine/kernels_attn.cpp`,
# `engine/attn_fattn_kernels.h`, `prototype/attn_decode/attn_decode_kernels.h`) in its own scratch directory,
# with `-I$side` ahead of `-I$REPO` so the quote-includes resolve to the copies. `engine/` is never written to.
# The `base` side is an unpatched copy and doubles as the behavioural canary: if `base` does not reproduce the
# shipped note block (178 / 0 / 0 / 32,768 — `progress/2026-10-21/performance-status-2026-10-21.md` §33's banner, read back by
# `prototype/attn_wmma/E1-result.md` §1), then the harness changed the code and every delta below is a harness
# artefact rather than a result.
#
# SIDES (all four compile by default; SIDES="base g1" narrows it):
#   base     the shipped pair, unpatched                      -> expect 178 VGPR / spill 0 / private 0 / 32,768 B
#   unstage  `a6_g1_unstage.py` only (E1's port)              -> expect 184 VGPR / spill 0 / 32,768 B
#            This side is the port-faithfulness canary: E1 measured exactly those numbers
#            (`prototype/attn_wmma/E1-result.md` §1). If it disagrees, my port is not E1's diff and the G1
#            reading below would be a diff against something nobody has measured.
#   g1       un-stage + `a6_g1_epi8.py` (the 8-row epilogue)  -> the bet: 20,480 B at spill 0, vgpr <= 190
#   g0       `a6_g0_warps7.py` (7 warps / 224 threads)        -> expect 65,536 B at spill 0, 1 block/CU -> 7 wf/CU
#
#   bash prototype/attn_percost/a6_build.sh [outdir]           # default /tmp/a6build
#   SIDES="base g1" bash prototype/attn_percost/a6_build.sh /tmp/a6g1only
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
OUT=${1:-/tmp/a6build}
SIDES=${SIDES:-"base unstage g1 g0"}
HERE="$REPO/prototype/attn_percost"

for side in $SIDES; do
    d="$OUT/$side"
    rm -rf "$d"
    mkdir -p "$d/engine" "$d/prototype/attn_decode"
    cp "$REPO/engine/kernels_attn.cpp"                       "$d/kernels_attn.cpp"
    cp "$REPO/engine/attn_fattn_kernels.h"                   "$d/engine/attn_fattn_kernels.h"
    cp "$REPO/prototype/attn_decode/attn_decode_kernels.h"   "$d/prototype/attn_decode/attn_decode_kernels.h"
    case "$side" in
        base) : ;;
        unstage) python3 "$HERE/a6_g1_unstage.py" --out "$d/engine/attn_fattn_kernels.h" >/dev/null ;;
        g1)   python3 "$HERE/a6_g1_unstage.py" --out "$d/engine/attn_fattn_kernels.h" >/dev/null
              python3 "$HERE/a6_g1_epi8.py" --in "$d/engine/attn_fattn_kernels.h" \
                                           --out "$d/engine/attn_fattn_kernels.h" >/dev/null ;;
        g0)   python3 "$HERE/a6_g0_warps7.py" --out-header "$d/engine/attn_fattn_kernels.h" \
                                              --out-decode "$d/prototype/attn_decode/attn_decode_kernels.h" >/dev/null ;;
        *)    echo "unknown side $side" >&2; exit 1 ;;
    esac
done

for side in $SIDES; do
    d="$OUT/$side"
    ( cd "$d" && hipcc -O3 -Wno-unused-command-line-argument \
        -I"$d" -I"$REPO" -I"$REPO/engine" --save-temps -c kernels_attn.cpp -o k.o ) \
        > "$d/compile.log" 2>&1 \
        || { echo "COMPILE FAILED: $side — see $d/compile.log"; tail -30 "$d/compile.log"; exit 1; }
    # The threads/warps the launcher will actually pass, read out of the side's own header rather than typed
    # here, so `fattn_residency_read.py`'s thread and wave limits are derived from the source under test.
    THR=$(sed -n 's/^#define FATTN_THREADS \([0-9]*\).*/\1/p' "$d/engine/attn_fattn_kernels.h" | tail -1)
    WRP=$(sed -n 's/^#define FATTN_WARPS \([0-9]*\).*/\1/p' "$d/engine/attn_fattn_kernels.h" | tail -1)
    { [ -n "$THR" ] && [ -n "$WRP" ]; } || { echo "could not read FATTN_THREADS/FATTN_WARPS out of $side"; exit 1; }
    echo "== side $side  (threads=$THR warps=$WRP)"
    python3 "$REPO/prototype/attn_wmma/fattn_residency_read.py" \
        --asm "$d/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s" \
        --kernel attn_fattn_wave --threads "$THR" --warps "$WRP" \
        | sed -n '2,30p'
done
