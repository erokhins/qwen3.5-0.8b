#!/usr/bin/env bash
# dump_sweep — which cache lanes does a query row's attention output actually depend on?
#
# For each perturbed position p, dump every attention body's input and output tiles and report which
# (layer, row) of the FIRST attention layer moved among the rows that must not depend on p. This turns
# CROSSROW's one-edge observation into the dependence set: a row that moves when lane bound(q) changes is
# a one-lane overhang; a row that also moves when lane bound(q)+k changes is a whole-round leak.
#
# Correctness only; no timing is taken or printed.
set -u
export LC_ALL=C
root=$(cd "$(dirname "$0")/../.." && pwd)
work=${1:?workdir}; shift
cd "$root"
bash tools/gpu_lock.sh acquire fix-crossrow-sweep >/dev/null
s=$(date +%s)
for p in "$@"; do
  if ! ./engine/build/decode_run --mtp-accept 0 --mtp-draft 0 --model models/Qwen3.5-0.8B --ids-file "$work/pert$p.csv" --steps 0 \
        --prefill-logits all --dump-attn-rows --out "$work/dsw-p$p" > "$work/dsw-p$p.log" 2>&1; then
    echo "RUN FAILED pert$p"; sed -n '1,3p' "$work/dsw-p$p.log"; continue
  fi
  echo "== perturbed position $p: which earlier rows moved, at the first attention layer (L3)"
  .venv/bin/python prototype/mtp_verify/attn_dump_diff.py "$work/dump-refwide" "$work/dsw-p$p" \
      --rows 0,1,2,3,4,5,6,7,8 --what pre_gate 2>&1 | grep -E "L3|VERDICT"
done
e=$(date +%s)
bash tools/gpu_lock.sh release fix-crossrow-sweep >/dev/null
echo "wall=$((e-s))s for $(( $# )) runs (whole-command wall; no timing taken)"
