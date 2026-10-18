#!/usr/bin/env bash
# tools/prefill_sampler.sh — record CPU %, VRAM and GPU-busy around a prefill run, on either engine.
#
# Why: D48 compares our engine to llama.cpp, and a tok/s figure says nothing about *where* the time
# went unless you also know whether the host was the bottleneck or the GPU was idle. Two columns are
# honest and one is not: CPU % and VRAM are trustworthy; **GPU use % saturates on this box** because
# the inference engine serving this very session busy-polls the iGPU (AGENTS.md: identical clean runs
# read 24-91 %). The GPU column is therefore a sanity check that the process is on the device at all,
# never a contention gate and never evidence of a busy GPU.
#
# Usage: tools/prefill_sampler.sh TAG -- <command...>
# Writes tmp/samples/TAG.csv: ts_s,cpu_pct_all,mem_avail_mb,gpu_use_pct,vram_used_mb
# Exits 1 if the sampler produced no samples (a silent sampler must not read as a clean run).
set -uo pipefail
export LC_ALL=C          # otherwise bc/awk print "49,0" and inject a CSV field

TAG="${1:?usage: prefill_sampler.sh TAG -- cmd...}"; shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no command given" >&2; exit 2; }

OUT="tmp/samples/${TAG}.csv"; mkdir -p tmp/samples
T0NS=$(date +%s%N)

cpu_pct() {   # busy % across all CPUs over one 0.5 s window, from /proc/stat (no external deps)
  local -a a b; read -r -a a <<<"$(head -1 /proc/stat)"; sleep 0.5; read -r -a b <<<"$(head -1 /proc/stat)"
  local i ta=0 tb=0 ia=$(( ${a[4]:-0} + ${a[5]:-0} )) ib=$(( ${b[4]:-0} + ${b[5]:-0} ))
  for i in 1 2 3 4 5 6 7 8 9 10; do ta=$((ta + ${a[i]:-0})); tb=$((tb + ${b[i]:-0})); done
  local dt=$((tb - ta)) di=$((ib - ia))
  if [ "$dt" -gt 0 ]; then awk -v d="$dt" -v i="$di" 'BEGIN{printf "%.2f", 100*(d-i)/d}'; else echo 0; fi
}

gpu_pct() {   # "<use_pct> <vram_used_mb>", or empty on parse failure
  local out u v
  out=$(rocm-smi --showuse --showmeminfo vram 2>/dev/null)
  u=$(awk -F: '/GPU use/{gsub(/[^0-9]/,"",$NF); print $NF; exit}' <<<"$out")
  v=$(awk -F: '/VRAM Total Used Memory/{gsub(/[^0-9]/,"",$NF); print $NF; exit}' <<<"$out")
  local unit=$(awk '/VRAM Total Used Memory/{if ($0 ~ /\(MiB\)/) print "MiB"; else print "B"; exit}' <<<"$out")
  if [ -z "$u" ] || [ -z "$v" ]; then echo ""; return; fi
  if [ "$unit" = "B" ]; then v=$(awk -v v="$v" 'BEGIN{printf "%.0f", v/1048576}'); fi
  echo "$u $v"
}

echo "ts_s,cpu_pct_all,mem_avail_mb,gpu_use_pct,vram_used_mb" > "$OUT"
( while :; do
    NOW=$(date +%s%N); ts=$(awk -v a="$T0NS" -v b="$NOW" 'BEGIN{printf "%.2f", (b-a)/1e9}')
    cpu=$(cpu_pct); gpu=$(gpu_pct)
    if [ -z "$gpu" ]; then echo "PARSE-FAIL: rocm-smi gave no GPU use / VRAM fields at ts=$ts" >&2; gpu="NA NA"; fi
    avail=$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo)
    echo "$ts,$cpu,$avail,${gpu% *},${gpu#* }" >> "$OUT"
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT INT TERM

TS=$(date +%s.%N); "$@"; RC=$?; TE=$(date +%s.%N)
kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null

awk -F, -v rc="$RC" -v a="$TS" -v b="$TE" '
  NR>1 { n++; c+=$2; if($2>cm)cm=$2; v=$5+0; if(v>vm)vm=v;
         if($4=="NA")na++; else {g+=$4; if($4>gm)gm=$4; if($4<5)idle++} }
  END {  if(!n){print "samples=0 — sampler never ran; unclassified run, not a clean one"; exit 1}
         printf "wall_s=%.2f rc=%d samples=%d\n", b-a, rc, n
         printf "CPU %%       mean=%.1f max=%.1f\n", c/n, cm
         printf "VRAM MB    peak=%.0f\n", vm
         if(na) printf "GPU use%%  UNPARSED in %d/%d samples (do not report this column)\n", na, n
         else   printf "GPU use%%  mean=%.1f max=%.1f  [saturates on this box; sanity check only]\n", g/n, gm
      }' "$OUT"
echo "csv=$OUT"
exit $RC
