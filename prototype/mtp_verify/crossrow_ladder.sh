#!/usr/bin/env bash
# crossrow_ladder — which FORM carries the cross-row edge?
#
# For each named configuration, run the reference prompt and the single-position perturbation at
# position 127, and byte-compare row 126 — the row that must NOT move (`crossrow_arm1.py compare`, CPU
# side). A config in which the edge disappears is a config in which the coupling does not live: that is
# the attribution instrument available without touching engine/.
#
# Usage: crossrow_ladder.sh [case ...]     (no args = every case)
#
# Correctness only. No timing is taken or printed anywhere here.
set -u
export LC_ALL=C
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
W=${CROSSROW_WORK:-/tmp/crossrow_ladder}
FIX=/tmp/crossrow_p128
mkdir -p "$W"

# One case per line, `name|extra flags`. A newline-delimited list read with IFS='|' — a `for spec in
# $CASES` word-splits the flags and silently turns `--attn-fattn-wide 1` into two cases, which is what
# the first version of this script did and it reported three bogus RUN FAILEDs.
CASES="default|
wide1|--attn-fattn-wide 1
nowide|--attn-fattn-wide 0
nofattn|--attn-fattn 0
notile|--attn-fattn 0 --attn-tile 0
valuwave|--attn-fattn 0 --attn-tile 1 --attn-touch 0
gemmhand|--gemm hand
onerow|--wave-rows 1
gdnloop1|--gdn-loop 1
gdncollapse|--gdn-scan collapse
gdnunroll4|--gdn-spell unroll4
gdnwf|--gdn-scan wf"

keep="$*"
bash "$ROOT/tools/gpu_lock.sh" acquire crossrow-ladder >/dev/null
while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    if [ -n "$keep" ] && ! echo " $keep " | grep -q " $name "; then continue; fi
    ok=1
    for side in ref pert127; do
        d="$W/$name/$side"
        mkdir -p "$d"
        # shellcheck disable=SC2086
        "$ROOT/engine/build/decode_run" --mtp-accept 0 --model "$ROOT/models/Qwen3.5-0.8B" \
            --ids-file "$FIX/$side.csv" --steps 0 --prefill-logits all \
            $flags --out "$d" > "$d/log.txt" 2>&1 || { ok=0; reason=$(tail -1 "$d/log.txt"); }
    done
    if [ "$ok" = 0 ]; then
        echo "### $name [$flags] REFUSED/FAILED: $reason"
        continue
    fi
    echo "### $name [$flags]"
    "$ROOT/.venv/bin/python" "$ROOT/prototype/mtp_verify/crossrow_arm1.py" compare \
        "$W/$name/ref" "$W/$name/pert127" --edge-row 126 | sed 's/^/    /'
done <<EOF
$CASES
EOF
bash "$ROOT/tools/gpu_lock.sh" release crossrow-ladder >/dev/null
