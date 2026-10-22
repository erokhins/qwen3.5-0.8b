#!/usr/bin/env bash
# dump_ab — does the attention-body dump move a bit? Three arms, one lock hold:
#   base = the binary as it was before the dump code existed
#   new  = the new binary, flag ABSENT   (must be byte-identical to base)
#   dump = the new binary, flag PRESENT  (must be byte-identical to new, plus the dump files)
# Correctness only; no kernel timing is taken or printed.
set -u
export LC_ALL=C
root=$(cd "$(dirname "$0")/../.." && pwd)
work=${1:-/tmp/cr_fix}
cd "$root"
run() {  # run <binary> <dir> [flag...]
  bin=$1; dir=$2; shift 2
  if ! "$bin" --model models/Qwen3.5-0.8B --ids-file "$work/ref.csv" --steps 8 \
        --out "/tmp/$dir" "$@" > "/tmp/$dir.log" 2>&1; then
    echo "RUN FAILED $dir"; sed -n '1,3p' "/tmp/$dir.log"; return 1
  fi
  echo "ran $dir"
}
bash tools/gpu_lock.sh acquire fix-crossrow-dump >/dev/null
s=$(date +%s)
run /tmp/baseline_decode_run ab_base
run ./engine/build/decode_run ab_new
run ./engine/build/decode_run ab_dump --dump-attn-rows
e=$(date +%s)
bash tools/gpu_lock.sh release fix-crossrow-dump >/dev/null
echo "wall=$((e-s))s for 3 runs (whole-command wall; no timing taken)"

echo "== ARM 1: pre-change binary vs new binary, flag absent"
if diff -r /tmp/ab_base /tmp/ab_new >/dev/null; then
  echo "  BYTE-IDENTICAL"
else
  echo "  DIFFERS:"; diff -r /tmp/ab_base /tmp/ab_new | head -6
fi

echo "== ARM 2: new binary, flag absent vs flag present (dump must not move a bit)"
moved=0
for f in /tmp/ab_new/*; do
  b=$(basename "$f")
  cmp -s "$f" "/tmp/ab_dump/$b" || { echo "  DIFFERS in $b"; moved=1; }
done
[ "$moved" = 0 ] && echo "  BYTE-IDENTICAL on every pre-change output file"
echo "  dump files added: $(ls /tmp/ab_dump | grep -c '^attn[0-9]')"
echo "== ATTN_DUMP lines: $(grep -c ATTN_DUMP /tmp/ab_dump.log) (one per attention layer per body)"
grep ATTN_DUMP /tmp/ab_dump.log | head -2
grep ATTN_DUMP /tmp/ab_dump.log | sed 's/.*core=\([a-z_*]*\).*/\1/' | sort | uniq -c
