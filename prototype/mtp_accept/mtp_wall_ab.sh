#!/usr/bin/env bash
# mtp_wall_ab.sh — the paired wall-clock A/B for the MTP accept loop, taken on the engine's OWN window.
#
# WHAT CHANGED, and why the old version is deleted rather than kept as a variant. It used to subtract a
# `--steps 0` run from a `--steps 32` run to remove prefill. That is invalid for the accept arm: a round
# can end the loop early (§07.5's stop, `RUN-END reason=context_exhausted`, `ROUND-SKIP`), so the two runs
# do not cover the same work — and the arithmetic produced a 2.17 ms decode for 45 emitted tokens (20,689
# tok/s, physically impossible) and negative windows on two other reps. The engine now prints `DECWALL`:
# one wall clock around the generation loop, the prefill leg reported separately rather than subtracted,
# and the denominator counted at the emit sites. This harness reads that line and does no arithmetic across
# runs at all.
#
# CANARY, asserted inside the measurement window, not before it (`AGENTS.md`: a paired A/B against a stale
# binary reads as a clean null). Every rep must show: a `DECWALL` line with `census=ok`; NO `MTPSTAT` on
# base; `MTPSTAT` with `rounds>0` on accept. A rep that fails a canary aborts the script before its
# milliseconds are used, because "both sides printed plausible numbers" was exactly what the stale-binary
# null looked like.
#
# Also asserted: `PREFILL tokens=` must equal the field count read off the IDS FILE — `--ids-file` splits
# on commas, so a space-joined fixture reads as ONE token and the run then "succeeds" at width 1 in ~1 s,
# which is the trap row in AGENTS.md and the reason the canary here is a field count and not a wall time.
#
# Protocol (fixed in `prototype/mtp_accept/ACCEPT-PLAN.md` before any MTP timing existed): main agent only,
# blocking, zero subagents, ONE lock hold for the whole paired pass so the arms share a window, median of
# REPS, util recorded at both ends and then discounted. Pre-registered decision line: adopt at >= +10 %
# tok/s at 4,000 lanes with gates green.
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

MODEL=${MODEL:-models/Qwen3.5-0.8B}
IDS=${IDS:-tmp/glue_copy_fold/ab/ids4032.csv}
STEPS=${STEPS:-32}
REPS=${REPS:-5}
CAP=${CAP:-4300}
BIN=${BIN:-engine/build/decode_run}
OUT=${OUT:-tmp/mtp_speed_wall}
ADOPT_PCT=${ADOPT_PCT:-10}
# Extra flags for the ACCEPT arm only — the thing being priced. Empty by default, which leaves the pair
# exactly as M3-RESULT took it. Unlike `--gdn-loop 1` (a required flag, therefore on both arms), this one
# IS the difference between the arms, so arm-specific is correct here: e.g.
#   ACCEPT_EXTRA="--mtp-dense 1" IDS=tmp/glue_copy_fold/ab/ids4032.csv prototype/mtp_accept/mtp_wall_ab.sh
ACCEPT_EXTRA=${ACCEPT_EXTRA:-}
# `GDN=''` (the default, and the empty default is the point) means "pass no `--gdn-loop` at all", i.e. the
# engine default — which since a4f2670 is what the SHIPPING loop runs: the round's bodies are forced to a
# one-token-step span inside the engine and every other body keeps 512. `GDN=1` pins BOTH arms to the
# whole-run span, which is how every carried figure in `M3-RESULT.md` and the §7 decode table were taken, and
# the two arms still move together either way (`AGENTS.md`: a flag one arm requires belongs on both, and after
# a4f2670 NO span is required — so the default stopped pinning one rather than pinning it on one arm, which
# would price the flag instead of the loop).
GDN=${GDN-}

want_tk=$(awk -F',' '{print NF}' "$IDS") || { echo "no ids file $IDS"; exit 1; }
[ "$want_tk" -ge 8 ] || { echo "$IDS has $want_tk fields; too few to be a fixture"; exit 1; }
mkdir -p "$OUT"
# Stale per-rep tsv files would be median-ed with the fresh ones (and a 3-column line from an older
# harness silently yields an empty 4th field, which reads as "a median came back missing").
rm -f "$OUT"/*.tsv

tools/gpu_lock.sh acquire main-agent mtp-wall-ab || exit 1
trap 'tools/gpu_lock.sh release main-agent >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util | tail -1); ids fields=$want_tk steps=$STEPS reps=$REPS cap=$CAP"
echo "== binary: $BIN ($(git log -1 --format=%h -- engine/ 2>/dev/null))  fixture: $IDS"
echo "== accept-arm extra flags: ${ACCEPT_EXTRA:-(none — the shipping loop)}  [priced difference, so arm-specific on purpose]"
echo "== gdn-loop on both arms: ${GDN:-(engine default — the shipping spans)}"

run() { # $1 arm $2 rep -> log path
  local arm="$1" rep="$2" extra="" log
  # The span goes on BOTH arms from one variable, or the pair would differ in the recurrence's submission
  # shape as well as in speculation — the first take of this A/B showed exactly that: a prefill gap between
  # the arms that had nothing to do with the accept loop at all, and that swamped the quantity being priced.
  # What it says now is "whatever
  # `$GDN` says", and `$GDN` is empty by default because after a4f2670 the round's span is not a flag the
  # caller can mis-set: `submit_wave_body` forces it wherever a mid-round checkpoint is armed. One variable
  # per A/B.
  [ -n "$GDN" ] && extra="--gdn-loop $GDN"
  # Default-on means the BASE arm has to name the plain path explicitly: since `aceee97` the engine default is
  # `--mtp-accept 1 --mtp-k 2`, so an unpinned base arm silently runs the accept loop and the A/B times the
  # accept loop against itself. The MTPSTAT canary below is what caught this — it fired as "base rep 1 printed
  # 1 MTPSTAT lines", which is exactly the class of silent-baseline bug this harness exists to prevent.
  [ "$arm" = base ] && extra="$extra --mtp-accept 0 --mtp-draft 0"
  [ "$arm" = accept ] && extra="$extra --mtp-draft 1 --mtp-accept 1 $ACCEPT_EXTRA"
  log="$OUT/${arm}_${rep}.log"; mkdir -p "$OUT/out_${arm}_${rep}"
  $BIN --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" --cap "$CAP" --out "$OUT/out_${arm}_${rep}" \
      $extra > "$log" 2>&1
  echo "$log"
}

# kv LINE KEY -> value. Deliberately scoped to ONE named line, so a stray `emitted=` in some other line
# cannot be picked up by a bare grep across the log.
kv() { # kv LINE-PREFIX FILE KEY -> the KEY's value on the first line starting with LINE-PREFIX.
       # Scoped to one named line on purpose, so a stray `emitted=` somewhere else in the log cannot be
       # picked up by a bare grep. Split on whitespace rather than with one long sed program, because the
       # sed form failed in this script and worked in a shell (both seds reported "unknown option to `s'"
       # at char 37 on a program of length 36) — a harness whose parser fails intermittently is the
       # empty-parse failure this repo has already been bitten by three times.
  grep -m1 "^$1 " "$2" | tr ' \t' '\n\n' | sed -n "s|^$3=||p"
}

canary() { # $1 arm $2 rep $3 log -> aborts the script on a bad side
  local arm="$1" rep="$2" log="$3" dw tk st
  dw=$(grep -c '^DECWALL' "$log")
  [ "$dw" = 1 ] || { echo "ABORT: $arm rep $rep printed $dw DECWALL lines; the window was never closed"; exit 1; }
  [ "$(kv DECWALL "$log" census)" = "ok" ] || { echo "ABORT: $arm rep $rep DECWALL census=$(kv DECWALL "$log" census) — the engine refused to bill this run"; exit 1; }
  tk=$(kv PREFILL "$log" tokens)
  [ "$tk" = "$want_tk" ] || { echo "ABORT: $arm rep $rep prefilled tokens=$tk but $IDS has $want_tk comma fields — the fixture was not read as a list"; exit 1; }
  st=$(grep -c '^MTPSTAT' "$log")
  # THE SPAN CANARY, in the same window as the milliseconds (`AGENTS.md`: a paired A/B against a stale binary
  # reads as a clean null, and every self-consistency check compiled into the binary cannot see that a
  # different binary ran). This change is REQUIRED to move the spans the two arms print — `gdn_span_round=1`
  # with `gdn_span_other` at whatever `$GDN` asked for on the accept arm, and the plain arm's single
  # `gdn_span=` at the same value — so a rep whose echo disagrees is a rep of some other build, and its
  # milliseconds are discarded before they are medianed.
  local want_span round_span mode_span
  want_span="${GDN:-512}"
  round_span=$(kv "MTP mode=accept" "$log" gdn_span_round)
  mode_span=$(kv "MTP mode=off" "$log" gdn_span)
  if [ "$arm" = accept ]; then
    [ "$round_span" = 1 ] || { echo "ABORT: accept rep $rep echoed gdn_span_round=${round_span:-<none>}; the round is forced to 1 by the engine, so this binary is not the one under test"; exit 1; }
    [ "$(kv "MTP mode=accept" "$log" gdn_span_other)" = "$want_span" ] \
      || { echo "ABORT: accept rep $rep ran its non-round bodies at gdn_span_other=$(kv "MTP mode=accept" "$log" gdn_span_other), asked $want_span"; exit 1; }
  else
    [ "$mode_span" = "$want_span" ] \
      || { echo "ABORT: base rep $rep ran at gdn_span=${mode_span:-<none>}, asked $want_span — the arms would differ in the recurrence span, which is not the thing being priced"; exit 1; }
  fi
  st=$(grep -c '^MTPSTAT' "$log")
  if [ "$arm" = base ]; then
    [ "$st" = 0 ] || { echo "ABORT: base rep $rep printed $st MTPSTAT lines; the flag is on where it must be off, so this is not a baseline"; exit 1; }
  else
    [ "$st" = 1 ] || { echo "ABORT: accept rep $rep printed $st MTPSTAT lines; nothing was adjudicated, so there is no accept arm to time"; exit 1; }
    local r; r=$(kv MTPSTAT "$log" rounds)
    [ "${r:-0}" -gt 0 ] || { echo "ABORT: accept rep $rep printed rounds=0; the loop ran no round"; exit 1; }
  fi
}

for rep in $(seq 1 "$REPS"); do
  for arm in base accept; do
    log=$(run "$arm" "$rep")
    canary "$arm" "$rep" "$log"
    line=$(grep '^DECWALL' "$log")
    printf '%-7s rep %s  prefill %8.1f ms  decode %9.3f ms  emitted %4s  bodies %3s rounds %3s plain %2s  rate %s  tok/s %8.3f  ended=%s\n' \
      "$arm" "$rep" \
      "$(kv DECWALL "$log" prefill_ms)" "$(kv DECWALL "$log" decode_ms)" \
      "$(kv DECWALL "$log" emitted)" "$(kv DECWALL "$log" bodies)" "$(kv DECWALL "$log" rounds)" \
      "$(kv DECWALL "$log" plain)" \
      "$(kv MTPSTAT "$log" accept_rate)" "$(kv DECWALL "$log" tok_per_s)" "$(kv DECWALL "$log" ended)"
    # one field per line for the median: tok/s, decode_ms, emitted, bodies
    printf '%s %s %s %s\n' "$(kv DECWALL "$log" tok_per_s)" "$(kv DECWALL "$log" decode_ms)" \
        "$(kv DECWALL "$log" emitted)" "$(kv DECWALL "$log" bodies)" >> "$OUT/${arm}_${rep}.tsv"
  done
done

med() { # $1 arm $2 column -> median over reps
  cat "$OUT"/$1_*.tsv 2>/dev/null | awk -v c="$2" '{print $c}' | sort -n |
    awk '{a[NR]=$1} END{if (NR==0) {print "nan"; exit} print a[int((NR+1)/2)]}'
}

echo
echo "== medians of $REPS reps, same lock hold, $want_tk lanes =="
for arm in base accept; do
  printf '  %-7s tok/s %8s   decode_ms %9s   emitted %6s\n' "$arm" "$(med $arm 1)" "$(med $arm 2)" "$(med $arm 3)"
done
bt=$(med base 1); at=$(med accept 1)
bstep=$(awk -v d="$(med base 2)" -v e="$(med base 3)" 'BEGIN{if (e+0>0) printf "%.4f", d/e; else print "nan"}')
around=$(awk -v d="$(med accept 2)" -v b="$(med accept 4)" 'BEGIN{if (b+0>0) printf "%.4f", d/b; else print "nan"}')
echo "  cost per body: base $(printf '%6.3f' "$bstep") ms/step   accept $(printf '%6.3f' "$around") ms/round"
# How many tokens a round can emit is k+1, not a constant. This line used to hardcode k=1's 2, which made it
# print "the ceiling on any accept rate is 1.089 x base" directly above a measured +16.53 % — a print that
# contradicts the number beside it is worse than no print, because a reader tabulates the ceiling. The parity
# test is the same generalisation: a round is at parity when tokens/round >= round/step, and it can never emit
# more than kmax, so parity is unreachable iff round/step >= kmax.
# How many tokens a round can emit is k+1, not a constant. Two bugs lived here: the line originally hardcoded
# k=1's 2 and printed "ceiling 1.089 x base" above a measured +16.53 %; then the engine default became k=2, so
# reading k out of ACCEPT_EXTRA alone went silently wrong again (an empty ACCEPT_EXTRA now means k=2, not k=1).
# A print that contradicts the number beside it is worse than no print, because a reader tabulates the ceiling.
# So: the default is 3 (k=2 shipped by aceee97), and only an explicit `--mtp-k 1` in ACCEPT_EXTRA lowers it.
case "$ACCEPT_EXTRA" in
  *"--mtp-k 1"*) kmax=2 ;;
  *) kmax=3 ;;
esac
awk -v bs="$bstep" -v ar="$around" -v kmax="$kmax" 'BEGIN{
  if (bs=="nan" || ar=="nan") { print "  NO-COST-ARITHMETIC: a median came back missing"; exit }
  r = ar/bs;
  printf "  one round costs %.3f x one step; a round emits at most %d tokens, so the ceiling on any", r, kmax;
  printf " accept rate is %.3f x base\n", kmax/r;
  printf "  parity would need tokens/round >= %.3f  (a round emits at most %d, and a is a rate: it cannot exceed 1.000)", r, kmax;
  if (r >= kmax) print "  — so PARITY IS UNREACHABLE AT THIS WIDTH FOR THIS k AT ANY ACCEPT RATE";
  else print "  — parity is reachable in principle if the measured tokens/round clears it";
}'
awk -v b="$bt" -v a="$at" -v p="$ADOPT_PCT" -v w="$want_tk" 'BEGIN{
  if (b=="nan" || a=="nan" || b+0<=0) { print "  NO-VERDICT: a median came back missing"; exit }
  d = 100.0*(a-b)/b;
  printf "  accept vs base: %+.2f %% tok/s at %s lanes (pre-registered adopt line: >= +%s %%)\n", d, w, p;
  if (d >= p) print "  VERDICT: clears the adopt line on wall time — subject to gates being green and the rate witness having survived its kill line";
  else print "  VERDICT: does NOT clear the adopt line on wall time";
}'
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
echo "== logs in $OUT (per-arm per-rep .log and .tsv); no number here is best-of-N and none is a burst"
