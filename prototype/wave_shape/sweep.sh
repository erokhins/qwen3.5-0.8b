#!/usr/bin/env bash
# sweep.sh — THE TIMED RUN. Main-agent only.
#
# Why this file is not runnable by a builder: every number it prints is a timing, it must hold the GPU
# exclusively (this box serves the session's own tokens on the same iGPU), and a contaminated number is
# worse than no number (AGENTS.md: "GPU execution is serialized and belongs to the main agent").
# Run it BLOCKING, in the foreground, with ZERO subagents active. Never background it, never `&`, never
# nohup: a backgrounded run keeps the token stream alive and contends with itself.
#
# What it prices (Route S1 item 2): launch SHAPE at fixed bytes and fixed launch count, separately from
# launch COUNT. S1 gate zero conflated them (340 launches at grid_max=271 vs 48 at grid_max=1918) and
# its 13.08 us/launch could not be read. Here every matrix cell touches identical bytes (1,464.0000 MiB
# read + 22.8750 MiB write, which is the step's own read census to +0.183 %), so:
#   * cells sharing L and differing in G  -> the shape term
#   * cells sharing G and differing in L  -> the count term, at fixed bytes
#   * cells sharing L*G (the iso-population diagonal) -> whether the machine cares how a fixed
#     workgroup population is spread over launches at all
#
# Discipline enforced below, in this order, and the run aborts rather than printing a number:
#   1. --selftest must be ok (no timing is taken for a red admission),
#   2. a behavioural canary: the shipped down arm must reproduce its CPU mirror bit-for-bit
#      (mirror=fma frozen_fma=0 vs_ship_ndiff=0 grid=128x1). engine/build.sh is not bit-reproducible
#      on this box, so cmp between binaries proves nothing; a stale binary would time the wrong shape
#      and every parity field inside the binary would agree with itself.
#   3. every RESULT row must carry the booked footprint (read_mib=1464.0000) or the row is a traffic
#      delta, not a shape delta,
#   4. two full passes inside ONE lock hold, so a "flat" claim is a within-window comparison and not a
#      comparison of two windows (AGENTS: a ruler re-taken in a different window is not an A/B),
#   5. the lock is acquired and released exactly once, in this one shell block, and the binary itself
#      never self-locks (checked below, because a self-locking child parked on a held lock looks like a
#      hung box).
set -u
export LC_ALL=C
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
BIN="$HERE/wave_shape"
REPS=${REPS:-5}            # outer best-of reps
ITERS=${ITERS:-3}          # inner best-of per rep
OUT=${OUT:-$HERE/out}
mkdir -p "$OUT"
CELLS="$OUT/cells.txt"
DOWNS="$OUT/downs.txt"
LOG="$OUT/sweep.log"
: > "$CELLS"; : > "$DOWNS"; : > "$LOG"

# ---- the matrix cells. Every (L,G) here must divide both pinned totals exactly (--print-plan). ----
# axis A: fixed G=8, L swept 16..1024        -> the count term at one shape
# axis B: fixed L=48, G swept 8..1024        -> the shape term at one count  (the item under test)
# axis C: iso-population pop=1024 and pop=4096, including G=1,2,4 (a launch of one block is the
#         one-row-launch end of the axis, which is what the engine's down_proj actually looks like)
mat_cells() {
  local L G
  for L in 16 32 64 128 256 512 1024; do echo "$L 8"; done
  for G in 8 16 32 64 128 256 512 1024; do echo "48 $G"; done
  echo "16 64"; echo "32 32"; echo "64 16"; echo "128 8"; echo "256 4"; echo "512 2"; echo "1024 1"
  echo "16 256"; echo "32 128"; echo "64 64"; echo "128 32"; echo "256 16"; echo "512 8"; echo "1024 4"
}
mat_cells | sort -u -n -k1,1 -k2,2 > "$OUT/cell_list.txt"
NCELL=$(wc -l < "$OUT/cell_list.txt")
DOWN_ARMS=("ship|" "intra|1" "intra|2" "intra|4" "intra|8" "two|2" "two|4" "two|7" "two|8" "two|14")

die() { echo "ABORT: $*" | tee -a "$LOG"; }

# =============================== ONE blocking lock hold, start to finish ==========================
cd "$HERE/../.." || exit 2
TOOLS=tools/gpu_lock.sh
[ -x "$TOOLS" ] || { echo "ABORT: $TOOLS not found; run from the repo"; exit 2; }
grep -q gpu_lock "$BIN" && { echo "ABORT: the binary self-locks; holding it here would deadlock"; exit 2; }
TAG="wave_shape item2 timed matrix (main agent, exclusive)"
echo "util at acquire: $("$TOOLS" util)" | tee -a "$LOG"
"$TOOLS" acquire "$TAG" || { echo "ABORT: could not acquire the GPU lock"; exit 2; }
trap '"$TOOLS" release "$TAG" >/dev/null 2>&1 || true' EXIT INT TERM

# --- gate 1: admission must be green before any timing ---
if ! "$BIN" --selftest 2>&1 | tee -a "$LOG" | grep -q "^SELFTEST ok"; then
    die "--selftest is not ok. No timing is taken for a red admission."
    exit 1
fi

# --- gate 2: behavioural canary (see header note 2) ---
CANARY=$("$BIN" --down ship 2>&1)
echo "CANARY-LINE $CANARY" | tee -a "$LOG"
case "$CANARY" in
  *"mirror=fma"*) ;; *) die "canary: down|ship did not report mirror=fma; the binary is not the build " \
                              "whose reduction order this rig was validated with"; exit 1;;
esac
case "$CANARY" in
  *"frozen_fma=0"*) ;; *) die "canary: frozen_fma!=0, the shipped form no longer reproduces its mirror"; exit 1;;
esac
case "$CANARY" in
  *"grid=128x1"*) ;; *) die "canary: down|ship is not 1024 rows over 128 blocks; the shape under test " \
                            "is not the shipped shape"; exit 1;;
esac
case "$CANARY" in
  *"vs_ship_ndiff=0"*) ;; *) die "canary: down|ship vs itself is not ndiff=0"; exit 1;;
esac

# --- gate 2b: the RESULT line itself, smoke-tested at reps=1 iters=1 before the real passes ---
# The RESULT path is otherwise reached only by this script, so it is exercised here on the smallest
# cell: fields present, and per_launch_us consistent with ms_min and the launch count. One rep of one
# iteration is a SHAPE check on the printer, not a measurement; nothing here may be quoted as a number.
SMOKE=$("$BIN" --launches 1024 --grid 8 --time --reps 1 --iters 1 2>&1 | grep '^RESULT' | head -1)
echo "SMOKE-LINE $SMOKE" | tee -a "$LOG"
[ -n "$SMOKE" ] || { die "gate 2b: --time printed no RESULT line at all"; exit 1; }
python3 "$HERE/check_result_line.py" "$SMOKE" | tee -a "$LOG"
[ "${PIPESTATUS[0]:-1}" -eq 0 ] || { die "gate 2b: the RESULT line is not a well-formed record"; exit 1; }

# --- gate 3+4: the timed matrix, two passes inside this one hold ---
run_cell() {  # $1=L $2=G $3=pass
    local line res
    line=$("$BIN" --launches "$1" --grid "$2" --time --reps "$REPS" --iters "$ITERS" 2>&1)
    res=$(printf '%s\n' "$line" | grep '^RESULT' | head -1)
    if [ -z "$res" ]; then
        echo "$line" | sed 's/^/RAW /' >> "$CELLS"
        die "cell L=$1 G=$2 printed no RESULT line (see RAW rows in $CELLS)"
        return 1
    fi
    case "$res" in
      *"read_mib=1464.0000"*) ;;
      *) die "cell L=$1 G=$2 printed a footprint that is not the booked one: $res"; return 1;;
    esac
    printf 'CASE cell-%dx%s\n' "$1" "$2" >> "$CELLS"
    printf '%s pass=%s\n' "$res" "$3" >> "$CELLS"
    return 0
}
run_down() {  # $1=arm $2=extra $3=pass
    local line res
    line=$("$BIN" --down "$1" $2 --time --reps "$REPS" --iters "$ITERS" 2>&1)
    res=$(printf '%s\n' "$line" | grep '^RESULT' | head -1)
    if [ -z "$res" ]; then
        echo "$line" | sed 's/^/RAW /' >> "$DOWNS"
        die "down arm $1 $2 printed no RESULT line (see RAW rows in $DOWNS)"
        return 1
    fi
    printf 'CASE %s\n' "$(printf '%s' "$line" | sed -n 's/^CASE \(.*\)$/\1/p' | head -1)" >> "$DOWNS"
    printf '%s pass=%s\n' "$res" "$3" >> "$DOWNS"
    return 0
}
BAD=0
for PASS in 1 2; do
    while read -r L G; do
        run_cell "$L" "$G" "$PASS" || BAD=1
    done < "$OUT/cell_list.txt"
    for a in "${DOWN_ARMS[@]}"; do
        arm=${a%%|*}; arg=${a##*|}
        case "$arm" in
            ship)  run_down ship "" "$PASS" ;;
            intra) run_down intra "--wpr $arg" "$PASS" ;;
            two)   run_down two  "--split $arg" "$PASS" ;;
        esac || BAD=1
    done
done
# The trailer counts RECORDS (cells x 2 passes), not distinct cells: fit_shape.py's `count` check
# compares "records printed" against "records parsed", and a distinct-cell trailer on a two-pass file
# refuses every real run (found that way — the fitter refused a complete 54-record file). Distinct-cell
# coverage is --expect's job, and it stays there.
echo "CELLS-PRINTED=$((NCELL * 2))" >> "$CELLS"
echo "DOWNS-PRINTED=$(( ${#DOWN_ARMS[@]} * 2 ))" >> "$DOWNS"
echo "util at release: $("$TOOLS" util)" | tee -a "$LOG"
"$TOOLS" release "$TAG" | tee -a "$LOG"
trap - EXIT INT TERM

echo "wrote $CELLS ($NCELL cells x 2 passes) and $DOWNS (${#DOWN_ARMS[@]} arms x 2 passes)" | tee -a "$LOG"
if [ "$BAD" -ne 0 ]; then
    echo "SWEEP INCOMPLETE: at least one row printed no RESULT or the wrong footprint. Do not fit it." \
        | tee -a "$LOG"
    exit 1
fi
echo "Next: python3 prototype/wave_shape/fit_shape.py $CELLS $DOWNS --expect $((NCELL + ${#DOWN_ARMS[@]}))" \
    | tee -a "$LOG"
