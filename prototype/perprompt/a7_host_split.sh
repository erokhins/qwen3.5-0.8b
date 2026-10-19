#!/usr/bin/env bash
# a7_host_split.sh — attribute §49's ~65–71 ms "per-prompt constant" (`RESULT ms_total` minus
# the wave's own `wall_ms`) into named causes, with both operands taken from ONE process.
#
# WHY THIS SHAPE. `RESULT ms_total` is the wall of `prefill_wave` plus the generation steps
#   (engine/decode_run.cpp:1787-1795, printed at :1990) and NOTHING else — the weight load is
#   timed separately into `load_ms` (:1588-1593) and never added in. `wall_ms` is printed by
#   `wave_pass`, which calls `st.reset()` and then runs `prefill_wave` AGAIN on the same
#   prompt (engine/decode_run.cpp:1397-1421). So §49's subtraction is cold-first-pass minus
#   warm-re-pass of identical work; it cannot contain host work outside the wave, because
#   there is none inside that bracket. The legs below split that cold cost by cause:
#
#   A  cold-vs-warm in one process: RESULT ms_total (pass 1) vs WAVEPASS gran=burst wall_ms
#      (pass 2) vs gran=stage (pass 3), plus the load wall the headline excludes.
#   B  the same with `--gemm hand`: is hipBLASLt's first execution per algo the cost?
#   C  (patched build) A7_REPEAT=3: the decay curve pass 1 → pass 4.
#   D  (patched build) A7_WARMUP=1: `RESULT ms_total` becomes the WARM number, i.e. the one
#      comparable with llama-bench, which discards a full-length warmup run by default
#      (engines/llama.cpp/tools/llama-bench/llama-bench.cpp:2415-2428; `no_warmup` defaults
#      false at :422; the timer starts at :2484).
#   E  (patched build) A7_BODY=1 A7_SYNC1=1: the wave's host-submission timeline (no sync
#      added) plus the first body's device wall — "the host stalls inside a launch" vs "the
#      first pass is slower on the device".
#
# MAIN-AGENT DEVICE WINDOW. This takes TIMINGS, so per AGENTS.md it is main-agent work: zero
#   subagents active, blocking foreground call, ONE `tools/gpu_lock.sh` hold, `util` and
#   `sclk` recorded (§6). It runs no numerics and touches no pins, so it is not a gate. No
#   `--ack-spill`, no gate edits, and no `engine/` edit in place: legs C/D/E run a binary built
#   from a scratch worktree (`prototype/perprompt/a7_phase.py --apply`), which is CPU-only.
#
# Usage
#   bash prototype/perprompt/a7_host_split.sh --dry-run     # print the commands, take nothing
#   bash prototype/perprompt/a7_host_split.sh               # legs A+B, shipping binary
#   WT=$(mktemp -d/a7wt) && git worktree add "$WT" HEAD --detach
#   python3 prototype/perprompt/a7_phase.py --apply "$WT" && (cd "$WT" && bash engine/build.sh)
#   bash prototype/perprompt/a7_host_split.sh --worktree "$WT/engine/build/decode_run"
#   WIDTHS="512 1024 4096" REPS=3 TAG=a7-hostsplit bash prototype/perprompt/a7_host_split.sh
#
# Every field is asserted non-empty: a phase that parses empty aborts, it is never read as zero
# (the empty-parse failure mode AGENTS.md records for `gdn_collapse/run.py` and `gate.py`).
set -uo pipefail
cd "$(dirname "$0")/../.."
export LC_ALL=C

BIN=${BIN:-engine/build/decode_run}
WT_BIN=""
WIDTHS=${WIDTHS:-"512 1024 4096"}
REPS=${REPS:-3}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
TAG=${TAG:-a7-hostsplit}
OUT=${OUT:-tmp/a7}
DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --worktree) WT_BIN=$2; shift 2 ;;
        --widths) WIDTHS=$2; shift 2 ;;
        --reps) REPS=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

die() {
    echo "ABORT: $*" >&2
    [ "$DRY" = 1 ] || tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1
    exit 1
}
clk() { rocm-smi --showclocks 2>/dev/null | tr -d '\r' | grep -iE "sclk" | tr '\n' ' '; }

# The readers live in a7_lib.sh so `a7_parse_selftest.sh` can prove they fail loudly on a
# missing line without running anything. `die` is defined above and exits this shell.
# shellcheck source=prototype/perprompt/a7_lib.sh
. "$(dirname "$0")/a7_lib.sh"

# The ruler's flags, same as `tools/llama_phase_compare.sh`'s prompt-only bench leg, so the
# cold number here is comparable with §49's rows and not with a fourth spelling.
ENG_FLAGS=(--steps 0 --gemm lt --gemm-pins all --gdn-loop 512)

ids_file() {  # one width; written to a file and passed with --ids-file (single argv slot caps
    local w=$1 f="$OUT/ids_$w.csv"   # at MAX_ARG_STRLEN=131072 B — AGENTS.md, §39)
    seq -s, 1 "$w" > "$f"
    local n
    n=$(tr ',' '\n' < "$f" | grep -c .)
    [ "$n" = "$w" ] || die "ids file for $w holds $n ids, not $w"
    printf '%s' "$f"
}

# The patch prints exactly these phase names (`a7_phase.py`'s EDITS). Requiring the count and
# each name is the guard `AGENTS.md` demands — a driver that parses another program's output
# must require the number of records it read, or an empty parse reports success. An empty
# phase census is a failure, not a zero.
A7_PHASE_NAMES="check_weights state_alloc_and_reset scratch_alloc gemm_route_init gemm_route_plan prepare_weight_adapters row_prints_and_dumps teardown"
phases_checked() {   # phases_checked LOG LEGNAME — dies if the phase census is incomplete
    local log=$1 leg=$2 n name
    n=$(grep -c '^A7PHASE' "$log")
    [ "$n" = 8 ] || die "$leg printed $n A7PHASE lines, expected 8 ($A7_PHASE_NAMES)"
    for name in $A7_PHASE_NAMES; do
        grep -q "^A7PHASE name=$name " "$log" || die "$leg has no A7PHASE line for $name"
    done
}

run_one() {  # run_one BIN LOG WIDTH [ENV=VAL]... -- [EXTRA ENGINE FLAGS]...
    local bin=$1 log=$2 w=$3
    shift 3
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    env "${envs[@]}" "$bin" --model "$MODEL" --ids-file "$(ids_file "$w")" \
        "${ENG_FLAGS[@]}" "$@" --out "$OUT/dump" > "$log" 2>&1 \
        || die "run failed: $bin width=$w ${envs[*]} $* (see $log)"
    grep -q '^RESULT ' "$log" || die "no RESULT line in $log — nothing to attribute"
}

echo "== a7_host_split: the §49 constant, split with both operands from one process =="
echo "widths='$WIDTHS' reps=$REPS unpatched=$BIN${WT_BIN:+ patched=$WT_BIN}"

if [ "$DRY" = 1 ]; then
    echo "--dry-run: these are the commands, one hold, blocking, zero subagents, no device work done here--"
    echo "tools/gpu_lock.sh acquire $TAG; tools/gpu_lock.sh util; rocm-smi --showclocks | grep -i sclk"
    for w in $WIDTHS; do
        for r in $(seq 1 "$REPS"); do
            echo "$BIN ${ENG_FLAGS[*]} --ids-file $OUT/ids_$w.csv --time-waves   # leg A: RESULT ms_total vs WAVEPASS burst/stage wall_ms, both from this one process"
            echo "$BIN ${ENG_FLAGS[*]} --ids-file $OUT/ids_$w.csv --time-waves --gemm hand   # leg B: same with no library (F2)"
            if [ -n "$WT_BIN" ]; then
                echo "A7_PHASE=1 A7_REPEAT=3 $WT_BIN ${ENG_FLAGS[*]} --ids-file $OUT/ids_$w.csv   # leg C: pass-1..4 decay"
                echo "A7_PHASE=1 A7_WARMUP=1 $WT_BIN ${ENG_FLAGS[*]} --ids-file $OUT/ids_$w.csv   # leg D: warm RESULT = the llama-methodology number"
                echo "A7_PHASE=1 A7_BODY=1 A7_SYNC1=1 $WT_BIN ${ENG_FLAGS[*]} --ids-file $OUT/ids_$w.csv   # leg E: host submit timeline + first-body device wall + A7PHASE rows"
            fi
        done
    done
    echo "tools/gpu_lock.sh release $TAG"
    exit 0
fi

[ -x "$BIN" ] || die "engine binary missing at $BIN (engine/build.sh)"
[ -z "$WT_BIN" ] || [ -x "$WT_BIN" ] || die "patched binary missing at $WT_BIN"
mkdir -p "$OUT"
: > "$OUT/summary.tsv"

tools/gpu_lock.sh acquire "$TAG" || exit 1
echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
echo "CLK@start: $(clk)"

for w in $WIDTHS; do
    ids_file "$w" > /dev/null
    echo
    echo "== width $w lanes =="
    for r in $(seq 1 "$REPS"); do
        LA="$OUT/A_${w}_${r}.log"
        run_one "$BIN" "$LA" "$w" -- --time-waves
        LOAD=$(load_ms "$LA")
        COLD=$(kv "$LA" ms_total "A.RESULT ms_total width=$w")
        WB=$(wave_wall "$LA" burst "A width=$w")
        WS=$(wave_wall "$LA" stage "A width=$w")
        PAR=$(kv "$LA" logits_bitidentical "A.WAVEPARITY width=$w")
        [ "$PAR" = 1 ] || die "the warm re-pass is NOT bit-identical at $w lanes (parity=$PAR): it is not the same work, so its wall cannot be subtracted"
        GAP=$(diff_ms "$COLD" "$WB")
        echo "A width=$w rep=$r load_ms=$LOAD cold_ms=$COLD warm_burst_ms=$WB warm_stage_ms=$WS gap_ms=$GAP parity=$PAR"
        printf '%s\tA\t%s\t%s\t%s\t%s\t%s\n' "$w" "$r" "$COLD" "$WB" "$GAP" "$LOAD" >> "$OUT/summary.tsv"
        LB="$OUT/B_${w}_${r}.log"
        run_one "$BIN" "$LB" "$w" -- --time-waves --gemm hand
        CH=$(kv "$LB" ms_total "B.RESULT ms_total hand width=$w")
        WH=$(wave_wall "$LB" burst "B width=$w")
        GH=$(diff_ms "$CH" "$WH")
        echo "B width=$w rep=$r hand cold_ms=$CH warm_burst_ms=$WH gap_ms=$GH   (compare A's gap at this width; compare DIFFERENCES — the hand wave itself is slower, so a ratio would lie)"
        printf '%s\tB\t%s\t%s\t%s\t%s\t-\n' "$w" "$r" "$CH" "$WH" "$GH" >> "$OUT/summary.tsv"

        if [ -n "$WT_BIN" ]; then
            LC_="$OUT/C_${w}_${r}.log"
            run_one "$WT_BIN" "$LC_" "$w" A7_PHASE=1 A7_REPEAT=3 --
            phases_checked "$LC_" "leg C width=$w"
            n=$(grep -c '^A7REPEAT kind=repeat' "$LC_")
            [ "$n" = 3 ] || die "leg C printed $n A7REPEAT lines, expected 3 — a repeat did not run"
            C3=$(kv "$LC_" ms_total "C.RESULT ms_total width=$w")
            echo "C width=$w rep=$r cold_ms=$C3 warm repeats (passes 2,3,4 -- if they agree with each other and pass 1 is the outlier, the constant is one-time cost and equals cold minus warm exactly): $(grep '^A7REPEAT kind=repeat' "$LC_" | sed -E 's/.*idx=([0-9]+) wall_ms=([0-9.]+).*/ pass\2/' | tr -d '\n')"
            printf '%s\tC\t%s\t%s\t-\t-\t-\n' "$w" "$r" "$C3" >> "$OUT/summary.tsv"

            LD="$OUT/D_${w}_${r}.log"
            run_one "$WT_BIN" "$LD" "$w" A7_PHASE=1 A7_WARMUP=1 --
            phases_checked "$LD" "leg D width=$w"
            WM=$(kv "$LD" ms_total "D.warm RESULT ms_total width=$w")
            echo "D width=$w rep=$r WARM_ms=$WM warm_tok_s=$(awk -v m="$WM" -v w="$w" 'BEGIN{printf "%.0f", w*1000/m}')   (leg D discards a full-length wave first, so this is the number comparable to llama-bench)"
            printf '%s\tD\t%s\t%s\t-\t-\t-\n' "$w" "$r" "$WM" >> "$OUT/summary.tsv"

            LE="$OUT/E_${w}_${r}.log"
            run_one "$WT_BIN" "$LE" "$w" A7_PHASE=1 A7_BODY=1 A7_SYNC1=1 A7_REPEAT=2 --
            phases_checked "$LE" "leg E width=$w"
            nb=$(grep -c '^A7BODY' "$LE")
            ns=$(grep -c '^A7SYNC1' "$LE")
            # 3 = the run's own wave + the 2 repeats; `A7_SYNC1` prints once per wave execution
            # (body 0 only), so a leg that also passed --time-waves would print more, and that
            # would be a different question. Asserting the count is the empty-parse guard.
            [ "$ns" = 3 ] || die "leg E printed $ns A7SYNC1 lines, expected 3 (run + 2 repeats) — first-body timing is incomplete"
            [ "$nb" -ge 3 ] || die "leg E printed $nb A7BODY lines, expected >= 3 — A7_BODY did not take effect"
            echo "E width=$w rep=$r bodies=$nb first-body device walls (1st is COLD, then warm): $(grep '^A7SYNC1' "$LE" | sed -E 's/^A7SYNC1 .*device_ms=//' | tr '\n' ' ')"
            echo "E   host submit wall of the run's body 0: $(grep '^A7BODY' "$LE" | head -1 | sed -E 's/.*submit_ms=//') ms   (small here while the gap is large means the host is not stalling inside submission, so the cost is device-side)"
            echo "E   host phases: $(grep '^A7PHASE' "$LE" | sed -E 's/^A7PHASE //' | tr '\n' ' ')"
        fi
    done
    echo "-- width $w medians: cold=$(awk -F'\t' -v w="$w" '$1==w && $2=="A"{print $4}' "$OUT/summary.tsv" | med) ms  warm=$(awk -F'\t' -v w="$w" '$1==w && $2=="A"{print $5}' "$OUT/summary.tsv" | med) ms  gap=$(awk -F'\t' -v w="$w" '$1==w && $2=="A"{print $6}' "$OUT/summary.tsv" | med) ms  gap_with_hand_gemm=$(awk -F'\t' -v w="$w" '$1==w && $2=="B"{print $6}' "$OUT/summary.tsv" | med) ms"
done

echo
echo "CLK@end: $(clk)"
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
tools/gpu_lock.sh release "$TAG"
echo "DONE. logs and summary.tsv in $OUT/"

cat <<'EOF'

HOW TO READ IT — what falsifies each hypothesis
  leg A `gap_ms` IS §49's constant, re-taken inside one process. ~0 at every width ⇒ the
     65–71 ms was a cross-process subtraction (§49's own bench-leg logs carry no WAVEPASS
     line) and there is nothing to chase. Tens of ms ⇒ H1 stands: it is first-execution cost.
  leg B vs leg A: gap(hand) < 0.2 × gap(lt) ⇒ hipBLASLt's first execution per algo (F2) owns
     most of it. gap(hand) ≈ gap(lt) within 20 % ⇒ the library is not it, and the residual is
     kernel first-launch (F3) plus the page bound F1 (~8.5 ms at the quoted 35 GB/s, see
     `a7_scratch_bytes.py`, which is arithmetic and needs no re-run).
  leg C: if passes 2..4 agree with each other to a few ms and pass 1 is the outlier, the
     constant is one-time and equals cold − warm exactly.
  leg D vs §49's llama column (pp512 7,378.72 / pp1024 8,014.23 / pp4096 10,259.19 tok/s):
     if warm_tok_s exceeds llama's at 512, the short-context ratio in §46/§49 has the wrong
     SIGN, and what is owed is a methodology decision (report cold AND warm, or pass
     `--no-warmup` to llama-bench), not a kernel.
  leg E: A7BODY walls small while gap is large ⇒ the host is not stalling inside submission,
     so the cost is device-side; A7SYNC1 (cold) vs leg C's warm pass-2 first body sizes that.
  What this CANNOT see: GPU-side per-kernel durations (a `rocprofv3 --kernel-trace` pass plus
     the pass-count discipline is a separate item), and any split of first-touch pages from
     code-object load finer than the F1 bound.
EOF
