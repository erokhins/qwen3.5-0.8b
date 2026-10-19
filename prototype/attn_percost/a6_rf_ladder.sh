#!/usr/bin/env bash
# a6_rf_ladder.sh — build the A6 Step 3 register-file probe: the fattn body, eight `__launch_bounds__(,N)` rungs,
# and LDS-only stubs, as one executable per side. BUILD IS CPU ONLY.
#
# Running the binary creates a device context (it submits nothing — every call in `a6_rf_probe.inc` is a driver
# query — but a context is a context), so it is executed under `tools/gpu_lock.sh` by whoever runs it, with its
# wall time and `util` reported and the words "no timing taken". Hence RUN=1 is opt-in:
#
#   bash prototype/attn_percost/a6_rf_ladder.sh /tmp/a6rf              # build only (default)
#   RUN=1 bash prototype/attn_percost/a6_rf_ladder.sh /tmp/a6rf        # build, then query the driver
#
# The two sides are the SAME quarantine as `a6_build.sh` (same three copied files, same `-I$side` ordering), plus
# `a6_rf_ladder.py`'s rungs and `a6_rf_probe.inc`'s `main()`. `base` is the shipped body — the reference E1 §1
# already read at 6 wf/CU — and `g1` is the bet whose residency claim needs the register file to answer.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
OUT=${1:-/tmp/a6rf}
HERE="$REPO/prototype/attn_percost"
RUN=${RUN:-0}

for side in base g1; do
    d="$OUT/$side"
    rm -rf "$d"
    mkdir -p "$d/engine" "$d/prototype/attn_decode"
    cp "$REPO/engine/attn_fattn_kernels.h"                 "$d/engine/attn_fattn_kernels.h"
    cp "$REPO/prototype/attn_decode/attn_decode_kernels.h" "$d/prototype/attn_decode/attn_decode_kernels.h"
    if [ "$side" = g1 ]; then
        python3 "$HERE/a6_g1_unstage.py" --out "$d/engine/attn_fattn_kernels.h" >/dev/null
        python3 "$HERE/a6_g1_epi8.py" --in "$d/engine/attn_fattn_kernels.h" \
                                     --out "$d/engine/attn_fattn_kernels.h" >/dev/null
    fi
    python3 "$HERE/a6_rf_ladder.py" --in "$d/engine/attn_fattn_kernels.h" \
                                    --out "$d/engine/attn_fattn_kernels.h" >/dev/null
    # ONE TU, and deliberately NOT the launcher TU: `engine/kernels_attn.cpp` drags in `engine::launchstats`,
    # which lives in another translation unit and would make this a link exercise. The fattn kernel and the
    # rungs are both DEFINED in the header, so including the header is enough.
    printf '#include "engine/attn_fattn_kernels.h"\n' > "$d/a6_rf_tu.cpp"
    cat "$HERE/a6_rf_probe.inc" >> "$d/a6_rf_tu.cpp"
    ( cd "$d" && hipcc -O3 --offload-arch=gfx1151 -Wno-unused-command-line-argument \
        -Wl,-rpath,/opt/rocm/lib -I"$d" -I"$REPO" -I"$REPO/engine" --save-temps \
        a6_rf_tu.cpp -o a6_rf_probe ) > "$d/compile.log" 2>&1 \
        || { echo "BUILD FAILED: $side"; tail -30 "$d/compile.log"; exit 1; }
    # The build log carries a third instrument for free: LLVM's own refusal when a minBlocks hint cannot be
    # met (`failed to meet occupancy target given by 'amdgpu-waves-per-eu' ... desired 4, final 3`). Kept as it
    # was emitted rather than paraphrased.
    grep -c "failed to meet occupancy target" "$d/compile.log" > "$d/occupancy_refusals.txt"
    echo "built $side: $d/a6_rf_probe  (llvm occupancy refusals: $(cat "$d/occupancy_refusals.txt"))"
done

echo
echo "== note blocks (compile side: what the squeeze cost per rung)"
for side in base g1; do
    echo "-- $side"
    python3 "$REPO/prototype/attn_wmma/fattn_residency_read.py" \
        --asm "$OUT/$side/a6_rf_tu-hip-amdgcn-amd-amdhsa-gfx1151.s" --all \
        | grep -E "attn_fattn_wave |a6_lb|a6_stub"
done

if [ "$RUN" = 1 ]; then
    echo
    echo "== driver answers (no kernel launched, no timing taken)"
    for side in base g1; do
        echo "-- $side"
        "$OUT/$side/a6_rf_probe"
        echo "exit=$?"
    done
fi
