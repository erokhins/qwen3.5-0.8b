#!/usr/bin/env bash
# meter_round.sh — where do an MTP round's extra milliseconds go? One row per phase, per shape.
#
# WHY. `prototype/mtp_accept/M3-RESULT.md` and `ROW-SCALE.md` established that an accept round costs
# ~1.96-2.09 steps at 4,032 lanes while the known extras (mid-checkpoint capture+restore ~19.3 MiB, verdict
# kernel + 16 B readback, one draft) do not add up to it. That is a budget with no line items, and B1/B2 of
# `BEAT-MTP.md` cannot be sized from it: funding "make the round cheaper" without naming the phase is how a
# route gets funded for the wrong reason. This harness runs the engine's own per-phase meter
# (`--mtp-meter 1`, `engine/mtp_meter.h`) and prints the line items.
#
# WHAT IT RUNS, per width and route. Two arms, because the two rounds are different machines:
#   verify  --mtp-verify 1 --mtp-meter 1                                the read-only side pass -> role=verify
#   accept  --mtp-draft 1 --mtp-accept 1 --gdn-loop 1 --mtp-meter 1      the shipping accept loop -> role=accept
# `MIDSPLIT=1` adds a third arm, `--mtp-midsplit 1`, whose `body_with_checkpoint - body` is the ONLY honest
# price of the mid-round checkpoint: inside the accept loop that capture is 2 submissions per GDN layer in
# the middle of `submit_wave_body`, and the meter prints it `status=not_metered` rather than putting 18
# device waits inside the call whose pipelining is the thing being priced.
#
# WHAT IT REFUSES TO PRINT, and the failure each refusal exists to catch, all enforced in
# `meter_table.awk` and exercised by `meter_round_selftest.sh`:
#   * coverage outside 0.8-1.2. `overhead` is DERIVED as wall - sum(phases) and is kept out of the sum, so
#     coverage is precisely "do the named phases account for the window"; a list accounting for 40 % of it has
#     a missing phase, and tabulating that would present a wrong attribution as a finding.
#   * sum(phases) + overhead != wall. An arithmetic identity over the printed numbers.
#   * no `MTPMETER ... on=1` header, or zero phase lines. A binary that ignored the flag would print an empty
#     table that reads as a null result. This is the behavioural canary: the field the change is REQUIRED to
#     move, read in the same breath as the number (`AGENTS.md`, "a paired A/B against a stale binary reads as
#     a clean null").
#   * the meter's `rounds=` != `MTPSTAT rounds=`. Two reports whose denominators name different populations
#     cannot be compared, and every `ms_per_round` on this table divides by one of them.
#   * `PREFILL tokens=` != the field count read off the ids FILE. `--ids-file` splits on commas, so a
#     space-joined fixture reads as ONE token and the whole table prices a one-token prefill.
#   * the run's own route echo disagrees with `--mtp-route`. `VERIFY2 mtp_route=`/`gemm=` and `MTPSTAT
#     route=`/`gemm=` name the route that ACTUALLY ran; a flag that never took would otherwise be reported as
#     the arm it did not run.
#   * `RESULT` is not the run's last stdout line, or `meter=sync_failed`.
#
# EVERY NUMBER HERE IS AN UPPER BOUND IN PARTS. Phases labelled `sync_added=1` are bounded by a device wait
# the shipping path does not perform, so they serialize and inflate; `rollback` inflates hardest because a
# real round deliberately overlaps its reject restore with the next draft. The label is on each line and the
# reasoning is in `METER-PLAN.md`. Read the phases against EACH OTHER and against `DECWALL`, never as a
# production cost, and never write these numbers into `engine/performance-status.md`.
#
# PROTOCOL. A timing harness, so it belongs to the main agent: blocking, one lock hold, zero subagents active
# (`AGENTS.md`, "GPU execution is serialized and belongs to the main agent"). It self-acquires
# `tools/gpu_lock.sh` — do NOT call it while holding the lock yourself, or it parks silently (`AGENTS.md`, the
# self-locking-runners row).
#
# Usage:
#   prototype/mtp_accept/meter_round.sh [width ...]              # default: 128 1024 4032
#   prototype/mtp_accept/meter_round.sh 4032 --route amortised
#   MIDSPLIT=1 prototype/mtp_accept/meter_round.sh 1024
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-engine/build/decode_run}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
OUT=${OUT:-tmp/meter_round}
STEPS=${STEPS:-16}
SRC_IDS=${SRC_IDS:-/tmp/dec8_ids.csv}
ROUTE=hand
WIDTHS=""
next_is_route=0
for a in "$@"; do
    if [ "$next_is_route" = 1 ]; then ROUTE="$a"; next_is_route=0; continue; fi
    case "$a" in
        --route) next_is_route=1 ;;
        --route=*) ROUTE="${a#--route=}" ;;
        hand|amortised) ROUTE="$a" ;;
        128|1024|4032) WIDTHS="$WIDTHS $a" ;;
        *) echo "usage: $0 [128|1024|4032 ...] [--route hand|amortised]"; exit 2 ;;
    esac
done
[ -n "$WIDTHS" ] || WIDTHS="128 1024 4032"
case "$ROUTE" in
    hand)      want_gemm=hand ;;
    amortised) want_gemm="${AMORTISED_GEMM:-lt}" ;;
    *) echo "ABORT: route '$ROUTE' is not hand or amortised"; exit 2 ;;
esac
[ -f "$SRC_IDS" ] || { echo "ABORT: $SRC_IDS missing. SRC_IDS=... a comma-separated id list (any data/refs input list, or /tmp/dec8_ids.csv)"; exit 2; }
mkdir -p "$OUT"; rm -f "$OUT"/*.log 2>/dev/null

ids_of() { # $1 width -> an ids file of EXACTLY that many comma-separated fields, cycled from a real fixture
    local w="$1" f="$OUT/ids$1.csv" n
    awk -v n="$w" 'BEGIN{RS="[,\n]"} {ids[c++]=$0} END{if(c<1){exit 1} for(i=0;i<n;i++) printf "%s%s", ids[i%c], (i+1<n?",":"")}' \
        "$SRC_IDS" > "$f"
    n=$(awk -F',' '{print NF}' "$f")
    [ "$n" = "$w" ] || { echo "ABORT: $f has $n fields, wanted $w — a short fixture silently feeds a smaller prefill"; exit 1; }
    echo "$f"
}
kv() { grep -m1 "^$1 " "$2" | tr ' \t' '\n\n' | sed -n "s|^$3=||p"; }
# The meter's verdict fields ride the run's OWN RESULT line — a second RESULT line would make every
# `grep -m1 '^RESULT'` harness in the tree read the wrong one — so read them off that one line.
result_kv() { grep -m1 '^RESULT mode=' "$1" | tr ' \t' '\n\n' | sed -n "s|^$2=||p"; }
instr_kv() { grep -m1 'phase=instrument' "$1" | tr ' \t' '\n\n' | sed -n "s|^$2=||p"; }
table() { awk -v role="$2" -v w="$3" -v logname="$1" -f "$HERE/meter_table.awk" "$1"; }

run() { # $1 arm $2 width $3 ids-file -> log path
    local arm="$1" w="$2" ids="$3" extra log
    case "$arm" in
        verify)   extra="--mtp-verify 1" ;;
        midsplit) extra="--mtp-verify 1 --mtp-midsplit 1" ;;
        accept)   extra="--mtp-draft 1 --mtp-accept 1" ;;
    esac
    # `--gdn-loop 1` on EVERY arm, the side pass included: a flag one arm requires belongs on both, or the
    # pair prices the flag instead of the round (`AGENTS.md`, the `mtp_wall_ab.sh` row). The accept loop pins
    # it, so leaving it off the verify arm would compare a looped body against an unlooped one.
    log="$OUT/${w}_${ROUTE}_${arm}.log"
    mkdir -p "$OUT/out_${w}_${ROUTE}_${arm}"
    "$BIN" --model "$MODEL" --ids-file "$ids" --steps "$STEPS" --cap "$((w + 300))" \
           --out "$OUT/out_${w}_${ROUTE}_${arm}" --no-stop --gdn-loop 1 \
           --mtp-meter 1 --mtp-route "$ROUTE" $extra > "$log" 2>&1
    echo "$log"
}

tools/gpu_lock.sh acquire main-agent meter-round || exit 1
trap 'tools/gpu_lock.sh release main-agent >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util | tail -1); widths='$WIDTHS' route=$ROUTE steps=$STEPS midsplit=${MIDSPLIT:-0}"

ARMS="verify accept"
[ "${MIDSPLIT:-0}" = 1 ] && ARMS="verify midsplit accept"
FAILED=0
for w in $WIDTHS; do
    ids=$(ids_of "$w")
    echo
    echo "=== lanes=$w  ids=$(basename "$ids") ($(awk -F',' '{print NF}' "$ids") fields)  cap=$((w + 300))  route=$ROUTE ==="
    for arm in $ARMS; do
        log=$(run "$arm" "$w" "$ids")
        # ---- canaries first; milliseconds are never read before they pass ----
        [ "$(grep -c '^RESULT mode=' "$log")" = 1 ] || { echo "ABORT: $w/$arm printed no single RESULT line"; exit 1; }
        [ "$(tail -1 "$log" | cut -c1-7)" = "RESULT " ] || { echo "ABORT: $w/$arm's last stdout line is not RESULT — anything printed after it would be what a `tail -1` harness reads instead of the run verdict"; exit 1; }
        grep -q '^MTPMETER role=.* on=1' "$log" || { echo "ABORT: $w/$arm printed no 'MTPMETER ... on=1' header: this binary does not meter. Rebuild with engine/build.sh, and do not discard its log"; exit 1; }
        tk=$(kv PREFILL "$log" tokens); [ "$tk" = "$w" ] || { echo "ABORT: $w/$arm prefilled $tk tokens, the file has $w"; exit 1; }
        [ "$(kv DECWALL "$log" census)" = "ok" ] || { echo "ABORT: $w/$arm census=$(kv DECWALL "$log" census)"; exit 1; }
        case "$arm" in
            verify|midsplit)
                grep -q '^VERIFY2' "$log" || { echo "ABORT: $w/$arm printed no VERIFY2: the side pass never ran, so this arm is a duplicate of a plain step"; exit 1; }
                grep '^VERIFY2 ' "$log" | grep -q "mtp_route=$ROUTE" || { echo "ABORT: $w/$arm did not echo mtp_route=$ROUTE on its VERIFY2 lines"; exit 1; }
                grep '^VERIFY2 ' "$log" | grep -q "gemm=$want_gemm" || { echo "ABORT: $w/$arm did not echo gemm=$want_gemm: route $ROUTE never reached the GEMM seam (resolve_wave_gemm, decode_step.cpp)"; exit 1; } ;;
            accept)
                [ "$(kv MTPSTAT "$log" rounds)" -gt 0 ] 2>/dev/null || { echo "ABORT: $w/$arm rounds not > 0"; exit 1; }
                [ "$(kv MTPSTAT "$log" route)" = "$ROUTE" ] || { echo "ABORT: $w/$arm MTPSTAT route=$(kv MTPSTAT "$log" route), asked $ROUTE: the round ran the other route"; exit 1; }
                [ "$(kv MTPSTAT "$log" gemm)" = "$want_gemm" ] || { echo "ABORT: $w/$arm MTPSTAT gemm=$(kv MTPSTAT "$log" gemm), expected $want_gemm for route $ROUTE"; exit 1; }
                [ "$(result_kv "$log" rounds)" = "$(kv MTPSTAT "$log" rounds)" ] || { echo "ABORT: $w/$arm RESULT rounds=$(result_kv "$log" rounds) but MTPSTAT rounds=$(kv MTPSTAT "$log" rounds): the meter and the round loop counted different populations"; exit 1; } ;;
        esac
        printf '   %-9s rounds=%s meter=%s coverage=%s plain_steps=%s syncs_added=%s sync_errors=%s\n' \
            "$arm" "$(result_kv "$log" rounds)" "$(result_kv "$log" meter)" "$(result_kv "$log" coverage)" \
            "$(instr_kv "$log" plain_steps)" "$(instr_kv "$log" syncs_added)" "$(instr_kv "$log" sync_errors)"
    done
    for arm in $ARMS; do
        log="$OUT/${w}_${ROUTE}_${arm}.log"
        [ "$arm" = accept ] && role=accept || role=verify
        table "$log" "$role" "$w" || FAILED=1
    done
done
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
[ "$FAILED" = 0 ] || { echo "== one or more arms refused their table; read the NO-TABLE lines above"; exit 1; }
