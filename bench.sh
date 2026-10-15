#!/usr/bin/env bash
# Run the testdata/ benchmark tests against one or more engines and print a table.
#
#   ./bench.sh                          # both engines, both tests
#   ./bench.sh --engine speccheck       # only the specification path
#   ./bench.sh --engine hf              # only the transformers reference
#   ./bench.sh --test decode            # long generation only (decode speed)
#   ./bench.sh --test prefill_summarize # one named test
#   ./bench.sh --reps 3                 # median of 3 per test
#   ./bench.sh --mtp 2                  # also run speccheck drafting 2 tokens/step
#   ./bench.sh --max-new 256            # override the generation ceiling
#   ./bench.sh --quick                  # both tests, tiny generation cap
#   ./bench.sh --keep                   # keep this run's jsonl (default: timestamped copy)
#
# Every engine is started in a fresh process (the two libraries should not share a
# process) with the same OMP_NUM_THREADS, so thread count cannot skew the comparison.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="$here/.venv/bin/python"
[[ -x "$py" ]] || { echo "no venv at $py" >&2; exit 1; }

ENGINE=both
TEST=all
REPS=1
MTP=""
MAXNEW=()
EXTRA=()
THREADS="${OMP_NUM_THREADS:-8}"
RUNDIR="$here/data/bench"
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --test)   TEST="$2"; shift 2 ;;
    --reps)   REPS="$2"; shift 2 ;;
    --mtp)    MTP="${2:-1}"; shift 2 ;;
    --max-new) MAXNEW=(--max-new "$2"); shift 2 ;;
    --quick)  MAXNEW=(--max-new 64); shift ;;
    --no-text) EXTRA+=(--no-text); shift ;;
    --threads) THREADS="$2"; shift 2 ;;
    --keep)   KEEP=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$here/models/Qwen3.5-0.8B" ]]; then
  echo "no checkpoint at $here/models/Qwen3.5-0.8B" >&2; exit 1
fi

mkdir -p "$RUNDIR"
stamp="$(date +%Y%m%d-%H%M%S)"
out="$RUNDIR/bench-$stamp.jsonl"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

args=(--test "$TEST" --reps "$REPS" "${MAXNEW[@]+"${MAXNEW[@]}"}" "${EXTRA[@]+"${EXTRA[@]}"}")
files=()
for e in $( [[ "$ENGINE" == "both" ]] && echo "speccheck hf" || echo "$ENGINE" ); do
  f="$RUNDIR/bench-$stamp-$e.jsonl"
  echo "=== engine=$e threads=$THREADS ==="
  "$py" "$here/testdata/run.py" --engine "$e" --jsonl "$f" "${args[@]}"
  files+=("$f")
  # MTP is a speccheck-only path (transformers has no MTP class for qwen3_5), so the
  # baseline row above is exactly the number it has to beat.
  if [[ -n "$MTP" && "$e" == "speccheck" ]]; then
    fm="$RUNDIR/bench-$stamp-$e-mtp$MTP.jsonl"
    echo "=== engine=$e+mtp$MTP threads=$THREADS ==="
    "$py" "$here/testdata/run.py" --engine "$e" --mtp "$MTP" --jsonl "$fm" "${args[@]}"
    files+=("$fm")
  fi
done

echo
echo "=== summary (threads=$THREADS, reps=$REPS) ==="
"$py" "$here/testdata/run.py" --table "${files[@]}"

merged="$RUNDIR/bench-$stamp.jsonl"
: > "$merged"
for f in "${files[@]}"; do
  grep '"engine"' "$f" >> "$merged" || true
  [[ "$KEEP" == 1 ]] || rm -f "$f"
done
echo
echo "raw results: $merged"
