#!/usr/bin/env bash
# form_ladder — which attention form carries CROSSROW's cross-row dependence, and it prints the
# NUMERICS canary per arm because an arm whose flag did not reach the binary is a repeat of the
# default run, not a control (AGENTS' "paired A/B against a stale binary" row; an earlier draft of
# this script dropped "$@" and priced the default form five times under five names).
#
# Correctness only. Compares the logits dumps of `ref` against `pert<p>` for rows BELOW p, which are
# the rows that must not move. No timing is taken or printed; the only wall figure is the whole
# script's, printed once, and it is not a kernel timing.
#
# usage: form_ladder.sh <workdir-with-csvs> <outtag> [extra decode_run flags...]
set -u
export LC_ALL=C
work=$1; outtag=$2; shift 2
root=$(cd "$(dirname "$0")/../.." && pwd)
run=$root/engine/build/decode_run

arm=$outtag
for v in ref pert5 pert4; do
  if ! "$run" --model "$root/models/Qwen3.5-0.8B" --ids-file "$work/$v.csv" --steps 0 \
        --prefill-logits all --out "$work/$arm-$v" "$@" > "$work/$arm-$v.log" 2>&1; then
    echo "ARM[$arm] RUN FAILED on $v"
    sed -n '1,4p' "$work/$arm-$v.log"
    exit 1
  fi
done

# The canary: the fields the flags are required to move, read from the run's OWN echo. Printed with
# `tr` rather than a regex so the canary cannot silently fail to match and print nothing — an earlier
# draft of this script printed no canary at all because its sed pattern did not fit the line, and an
# earlier ladder dropped "$@" and priced the default form under four arm names.
grep -h '^NUMERICS' "$work/$arm-ref.log" | tr ' ' '\n' | grep -E
'^(wave_rows_cap|wave_rows|attn_tile|attn_touch|attn_fattn|attn_fattn_roll|attn_fattn_qstage|attn_fattn_pvf16|attn_fattn_wide|gdn_scan|gdn_chunk|gdn_spell|gdn_loop|gdn_norm|gdn_geom)=' | tr '\n' ' '
echo "  <- CANARY[$arm]"
grep -h '^ATTN_WIDE' "$work/$arm-ref.log" | head -1

back5=""
for i in 0 1 2 3 4; do
  cmp -s "$work/$arm-ref/logits$i.bin" "$work/$arm-pert5/logits$i.bin" || back5="$back5 $i"
done
back4=""
for i in 0 1 2 3; do
  cmp -s "$work/$arm-ref/logits$i.bin" "$work/$arm-pert4/logits$i.bin" || back4="$back4 $i"
done
echo "ARM[$arm] pert5 -> rows<5 that moved:${back5:- none} | pert4 -> rows<4 that moved:${back4:- none}"
