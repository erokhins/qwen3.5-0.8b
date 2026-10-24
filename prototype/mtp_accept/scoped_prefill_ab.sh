#!/usr/bin/env bash
# Prefill A/B for the scoped MTP GDN span: three arms, two widths, three reps, medians by eye.
# Canaries: PREFILL tokens= must equal the id count in the file; MTP mode= names the spans actually used.
set -u
cd "$(dirname "$0")/.."
pick() { grep '^'"$1"' ' | tr ' ' '\n' | sed -n "s/^$2=//p" | head -1; }
tools/gpu_lock.sh acquire main-agent scoped-prefill >/dev/null
for f in tmp/verify_body_scale/ids1024.csv tmp/verify_body_scale/ids4032.csv; do
  n=$(awk -F',' '{print NF}' "$f")
  echo "### $f has $n ids"
  for arm in scoped plain oldpin; do
    case $arm in
      scoped) EXTRA="" ;;
      plain) EXTRA="--mtp-accept 0 --gdn-loop 512" ;;
      oldpin) EXTRA="--gdn-loop 1" ;;
    esac
    for rep in 1 2 3; do
      log=/tmp/scoped_pf_${arm}_${rep}.log
      engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file "$f" --steps 0 $EXTRA \
        --out /tmp/pf_x > "$log" 2>&1
      tk=$(pick PREFILL tokens < "$log")
      ms=$(grep '^DECWALL' "$log" | tr ' ' '\n' | sed -n 's/^prefill_ms=//p' | head -1)
      sp=$(grep '^MTP mode=' "$log" | tr ' ' '\n' | sed -n 's/^gdn_span_other=//p' | head -1)
      printf '  %-7s rep %s tokens=%-6s prefill_ms=%-9s gdn_span_other=%s\n' "$arm" "$rep" "$tk" "$ms" "${sp:-off}"
    done
  done
done
tools/gpu_lock.sh release main-agent >/dev/null
