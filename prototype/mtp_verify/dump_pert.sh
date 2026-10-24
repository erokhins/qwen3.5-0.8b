#!/usr/bin/env bash
# dump_pert — dump the attention block's own output tiles for `ref` and `pert<p>` and say which
# (layer, row) moved. Correctness only: no timing is taken or printed, one device run at a time, and the
# lock is held only around the runs.
#
# usage: dump_pert.sh <workdir> <pert-tag e.g. pert5> [--extra decode_run flags...]
set -u
export LC_ALL=C
root=$(cd "$(dirname "$0")/../.." && pwd)
work=${1:?workdir}; tag=${2:?pert tag}; shift 2
cd "$root"
run() {  # run <csv-stem> <outtag>
  stem=$1; out=$2; shift 2          # shift, or "$@" below re-feeds this function's own name to the binary
  if ! ./engine/build/decode_run --mtp-accept 0 --mtp-draft 0 --model models/Qwen3.5-0.8B --ids-file "$work/$stem.csv" \
        --steps 0 --prefill-logits all --dump-attn-rows --out "$work/dump-$out" "$@" \
        > "$work/dump-$out.log" 2>&1; then
    echo "RUN FAILED $out"; sed -n '1,3p' "$work/dump-$out.log"; return 1
  fi
  echo "ran dump-$out ($(grep -c ATTN_DUMP "$work/dump-$out.log") bodies, cores: $(grep ATTN_DUMP "$work/dump-$out.log" | sed 's/.*core=\([a-z_*]*\).*/\1/' | sort -u | tr '\n' ','))"
}
bash tools/gpu_lock.sh acquire fix-crossrow-dumppert >/dev/null
s=$(date +%s)
run ref refwide
run "$tag" "perth${tag#pert}"
run ref refwide0 --attn-fattn-wide 0
run "$tag" "perth${tag#pert}w0" --attn-fattn-wide 0
e=$(date +%s)
bash tools/gpu_lock.sh release fix-crossrow-dumppert >/dev/null
echo "wall=$((e-s))s for 4 runs (whole-command wall; no timing taken)"
for pair in "refwide perth${tag#pert}" "refwide0 perth${tag#pert}w0"; do
  set -- $pair
  echo "== $1 vs $2 (rows below the perturbed position)"
  .venv/bin/python prototype/mtp_verify/attn_dump_diff.py "$work/dump-$1" "$work/dump-$2" --rows 0,1,2,3,4
done
