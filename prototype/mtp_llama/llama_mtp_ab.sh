#!/usr/bin/env bash
# llama_mtp_ab.sh — does llama.cpp's MTP speculative decoding pay on THIS box, and where does it go?
#
# WHY. Our own accept loop measures a verify round at 2.029x a decode step at 4,000 lanes, which caps k=1
# speculation below parity at ANY accept rate (prototype/mtp_accept/M3-RESULT.md §4). Before concluding
# anything about MTP from one implementation, price the same idea in the implementation that shipped it:
# llama.cpp's `--spec-type draft-mtp`, on the SAME model (this repo's HF checkpoint, converted with its MTP
# head: blk.24.nextn.*), the SAME iGPU, the SAME greedy sampling, the same prompt files. One variable: arm.
#
# WHAT MAKES THIS MORE THAN A tok/s PAIR. llama.cpp logs its own accept statistics at trace level, and the
# statistics line carries the DRAFT time separately from the verify time (`dur(b,g,a)`), plus the counters
# needed to recover the number that matters:
#
#     rounds       = predicted_n - accepted_tokens   (each round emits its bonus token + its accepted ones)
#     ms/round     = eval_ms / rounds
#     tokens/round = predicted_n / rounds = 1 + accepted/rounds
#
# So the round-vs-step ratio comes from the program's own counters, not from a wall-time guess — the same
# arithmetic M3-RESULT applies to our engine, which is what makes the two runtimes comparable.
#
# CANARIES (per rep, before its numbers are used; a stale flag reads as a clean null -- AGENTS.md): the MTP
# arms must print a `statistics ... draft-mtp` line with #gen drafts > 0, the `none` arm must print none,
# parsed numbers must be numbers, and the prompt token count must be IDENTICAL across arms — a differing pp
# means the arms priced different requests, i.e. two experiments instead of one A/B.
#
# Protocol: main agent, blocking, one lock hold for the run, util recorded at both ends and discounted,
# median of REPS, greedy (--temp 0), fixed seed, all layers offloaded, flash attention on for every arm.
#
# Usage:  prototype/mtp_llama/llama_mtp_ab.sh [shape...]   # tiny|short|long (default all three)
#         REPS=2 TMO=420 ARMS="none mtp3" ...
# NOTE: shapes come from ALL args ($*), not $1 — reading $1 alone silently ran only the first shape.
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

BIN=${BIN:-engines/llama.cpp/build-hip/bin/llama-cli}
MODEL=${MODEL:-tmp/mtp_llama/Qwen3.5-0.8B.mtp.bf16.gguf}
OUT=${OUT:-tmp/mtp_llama/run}
REPS=${REPS:-2}
NGEN=${NGEN:-128}
ARMS=${ARMS:-none mtp1 mtp2 mtp3 mtp4}
CTX=${CTX:-5120}

mkdir -p "$OUT"; rm -f "$OUT"/*.tsv "$OUT"/*.log 2>/dev/null

nmax_of() { case "$1" in none) echo 0 ;; mtp1) echo 1 ;; mtp2) echo 2 ;; mtp3) echo 3 ;; mtp4) echo 4 ;; esac; }

prompt_of() { # $1 shape -> file; built here so the token count is a measured fact, not a filename
  case "$1" in
    tiny)  printf 'The capital of France is' > "$OUT/p_tiny.txt"; echo "$OUT/p_tiny.txt" ;;
    short) cp speccheck/fixtures/corpus.txt "$OUT/p_short.txt";   echo "$OUT/p_short.txt" ;;
    long)  head -c 10200 progress/2026-10-21/performance-status-2026-10-21.md > "$OUT/p_long.txt"
           echo "$OUT/p_long.txt" ;;
    *) echo "unknown shape $1" >&2; exit 1 ;;
  esac
}

run_arm() { # $1 prompt-file $2 arm $3 rep -> logfile
  local pf="$1" arm="$2" rep="$3" spec=()
  local log="$OUT/$(basename "$pf" .txt)_${arm}_${rep}.log"
  if [ "$arm" != none ]; then
    spec=(--spec-type draft-mtp --spec-draft-n-max "$(nmax_of "$arm")")
  fi
  # --log-verbose on EVERY arm, not only the MTP ones: without it this build prints the REPL banner and no
  # print_timing lines at all, and a harness that gets perf from one arm and not the other is comparing a
  # measurement to a guess. The trace lines cost both arms host I/O (~1 line per draft call), same order
  # on both sides. -p "$(cat file)" rather than -f for the same reason: -f drops to the REPL.
  timeout "${TMO:-420}" "$BIN" -m "$MODEL" -ngl 99 -fa on -c "$CTX" --temp 0 -s 42 \
      -p "$(cat "$pf")" -n "$NGEN" "${spec[@]}" --log-verbose > "$log" 2>&1
  echo "$log"
}

# llama.cpp prints this locale's decimal separator (here a comma) in perf and stats lines. Normalise ONLY
# digit-comma-digit, so `dur(b,g,a)` keeps its letters and `292,51` becomes 292.51. A blanket `tr ',' '.'`
# (this harness's first version) rewrites the separators too and then matches nothing — a silent empty parse
# on the very line the accept rate comes from.
norm() { sed -E 's/([0-9]),([0-9])/\1.\2/g' "$1"; }
num_ok() { case "$1" in ''|*[!0-9.]*|*.*.*) return 1 ;; *) return 0 ;; esac; }

# `eval time = 292.51 ms / 32 tokens ( 9.44 ms per token, 105.98 tokens per second)`
# The prompt line is `prompt eval time = ...` and matches the same pattern (and comes FIRST, so head -1
# would price the prompt as if it were the decode) — hence the grep -v.
parse_eval() { norm "$1" | grep -v "prompt eval time" \
    | grep -oE "eval time = +[0-9.]+ ms / +[0-9]+ tokens" | head -1 \
    | sed -E 's/[^0-9]+([0-9.]+) ms \/ +([0-9]+) tokens/\1 \2/'; }
parse_pp() { norm "$1" | grep -oE "prompt eval time = +[0-9.]+ ms / +[0-9]+ tokens" | head -1 \
    | sed -E 's/[^0-9]+([0-9.]+) ms \/ +([0-9]+) tokens/\1 \2/'; }
# `statistics draft-mtp: ... #gen drafts = 9, ... #acc tokens = 22, ... dur(b,g,a) = 0.01, 231.2, 0.01 ms`
parse_spec() { norm "$1" | grep -oE "statistics +draft-mtp:.*" | head -1 \
    | sed -nE 's/.*#gen drafts = +([0-9]+).*#acc tokens = +([0-9]+).*dur\(b,g,a\) = +([0-9.]+), +([0-9.]+), +([0-9.]+).*/\1 \2 \4/p'; }

if [ "$#" -gt 0 ]; then SHAPES="$*"; else SHAPES="tiny short long"; fi
tools/gpu_lock.sh acquire main-agent llama-mtp-ab || exit 1
trap 'tools/gpu_lock.sh release main-agent >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util | tail -1); model=$MODEL ngen=$NGEN reps=$REPS ctx=$CTX arms='$ARMS'"

for shape in $SHAPES; do
  pf=$(prompt_of "$shape")
  echo
  echo "=== shape=$shape (prompt file $pf, $(wc -c < "$pf") bytes) ==="
  for rep in $(seq 1 "$REPS"); do
    for arm in $ARMS; do
      log=$(run_arm "$pf" "$arm" "$rep")
      ev=$(parse_eval "$log"); [ -n "$ev" ] || { echo "ABORT: $shape/$arm/rep$rep printed no eval line"; tail -3 "$log"; exit 1; }
      pp=$(parse_pp "$log");  [ -n "$pp" ] || { echo "ABORT: $shape/$arm/rep$rep printed no prompt-eval line"; exit 1; }
      sp=$(parse_spec "$log")
      set -- $ev; ems="$1"; ntok="$2"; set -- $pp; pms="$1"; ptok="$2"
      num_ok "$ems" || { echo "ABORT: unparsable eval ms ($ems) in $shape/$arm/rep$rep"; exit 1; }
      num_ok "$pms" || { echo "ABORT: unparsable prompt ms ($pms) in $shape/$arm/rep$rep"; exit 1; }
      echo "$ptok" >> "$OUT/${shape}_ptok.txt"
      if [ "$arm" = none ]; then
        [ -z "$sp" ] || { echo "ABORT: the none arm printed draft statistics ($sp)"; exit 1; }
        printf '%-5s rep %s  pp %5s tok %8.1f ms  eval %8.2f ms / %4s tok  %8.3f tok/s\n' \
          "$arm" "$rep" "$ptok" "$pms" "$ems" "$ntok" "$(awk -v a="$ems" -v n="$ntok" 'BEGIN{printf "%.3f",1000.0*n/a}')"
        echo "$ems $ntok 0 0" >> "$OUT/${shape}_${arm}_${rep}.tsv"
      else
        [ -n "$sp" ] || { echo "ABORT: $shape/$arm printed NO draft-mtp statistics: the MTP arm drafted nothing, so it is a null, not a slow arm"; exit 1; }
        set -- $sp; gdrafts="$1"; acc="$2"; dms="$3"
        [ "$gdrafts" -gt 0 ] || { echo "ABORT: $shape/$arm #gen drafts = 0"; exit 1; }
        num_ok "$dms" || { echo "ABORT: unparsable draft ms ($dms) in $shape/$arm"; exit 1; }
        rounds=$(awk -v n="$ntok" -v a="$acc" 'BEGIN{print n-a}')
        [ "$rounds" -gt 0 ] || { echo "ABORT: $shape/$arm rounds=0 from n=$ntok acc=$acc"; exit 1; }
        printf '%-5s rep %s  pp %5s tok %8.1f ms  eval %8.2f ms / %4s tok  %8.3f tok/s  rounds %3s acc_tok %3s  draft %6.1f ms (%.0f%% of eval)\n' \
          "$arm" "$rep" "$ptok" "$pms" "$ems" "$ntok" \
          "$(awk -v a="$ems" -v n="$ntok" 'BEGIN{printf "%.3f",1000.0*n/a}')" \
          "$rounds" "$acc" "$dms" "$(awk -v d="$dms" -v e="$ems" 'BEGIN{printf "%.0f",100*d/e}')"
        echo "$ems $ntok $rounds $dms" >> "$OUT/${shape}_${arm}_${rep}.tsv"
      fi
    done
  done
  echo "  prompt tokens seen: $(uniq -c "$OUT/${shape}_ptok.txt" | tr '\n' ' ')"
  [ "$(uniq -c "$OUT/${shape}_ptok.txt" | wc -l)" = 1 ] \
    || { echo "ABORT: arms disagree on prompt token count for $shape -- the pair priced different requests"; exit 1; }

  med() { cat "$OUT"/${shape}_$1_*.tsv | awk -v c="$2" '{print $c}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
  echo "  -- medians, shape=$shape --"
  bstep=$(awk -v a="$(med none 1)" -v n="$(med none 2)" 'BEGIN{printf "%.6f",a/n}')
  btps=$(awk -v a="$(med none 1)" -v n="$(med none 2)" 'BEGIN{printf "%.6f",1000.0*n/a}')
  printf '   %-5s %8.3f tok/s   step cost %7.3f ms   (the denominator for every ratio below)\n' none "$btps" "$bstep"
  for arm in $ARMS; do
    [ "$arm" = none ] && continue
    awk -v arm="$arm" -v ems="$(med "$arm" 1)" -v n="$(med "$arm" 2)" -v r="$(med "$arm" 3)" -v d="$(med "$arm" 4)" \
        -v bs="$bstep" -v btps="$btps" 'BEGIN{
      rcost = ems/r; tps = 1000.0*n/ems; tpr = n/r;
      printf "   %-5s %8.3f tok/s   round %7.3f ms (draft %5.3f)  round/step = %.3f  tok/round = %.2f  accept-tok/round = %.2f  -> %.3f x base (%+.1f %%);  parity needs tok/round >= %.3f\n",
             arm, tps, rcost, d/r, rcost/bs, tpr, tpr-1, tps/btps, 100.0*(tps-btps)/btps, rcost/bs; }'
  done
done
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
