#!/usr/bin/env bash
# ab_d3_engine.sh's discipline applied to D2: one lock hold, blocking, canaries read INSIDE the window
# before any millisecond is used, an empty-parse guard on every parser that aborts rather than printing
# blanks, and NO-TIMING on any red.
#
#   bash prototype/mlp_insitu/ab_d2_sweep.sh
#   REPS=5 ITERS=8 ROUNDS=2 bash prototype/mlp_insitu/ab_d2_sweep.sh
#   QUOTED_BURST=2.240 bash prototype/mlp_insitu/ab_d2_sweep.sh     # if the burst was re-taken in-window
#
# ============================================================================ WHAT IT IS FOR
# D2 has to say WHERE the MLP's ~1.09 ms burst->in-situ penalty comes from, and the pre-registered
# lines are on a STAGE-GRANULARITY family row: WIN <= 2.90 ms, KILL >= 3.20 ms
# (progress/2026-10-21/decode-attribution-2026-10-21.md §5, D2). The rig decomposes the penalty into
# named terms; this script is the window that fills them in. Prototype-only: nothing here changes the
# engine, so there is no `sampled.txt` decision check — see D2-DIAGNOSIS.md §2 for why that check would
# be vacuous here and what replaces it (the burst reproduction, the rig's byte witnesses on every row,
# and the drain cross-check against the ledger's own 7.2 us per-segment constant).
#
# ============================================================================ THE CANARIES
# 1. BUILD/ARM canary — the rig `#include`s the shipping kernel header, so there is no second object to
#    be stale; the canary is instead that the CPU gate and the census must pass BEFORE the lock is taken
#    (a compile or a parse failure must not eat the window), and that `--selftest-geom` prints
#    `SELFTEST-GEOM ok fails=0`.
# 2. BURST canary — row R0 (blob3, --drain burst) must reproduce the QUOTED 2.275 ms within +-3 %.
#    If it does not, the rig is not the published burst and nothing downstream is admissible.
# 3. PHENOMENON canary — P = R7 - R0 must be >= 0.35 ms. A rig that cannot synthesize the penalty
#    cannot decompose it: that is NO-DIAGNOSIS, and the rest of the matrix is not run.
# 4. BYTE canary — EVERY row must print `verify=ok`. A millisecond figure from a run whose bytes were
#    not accounted is a measurement of something unknown, so a red verify aborts the window.
# 5. EMPTY-PARSE guard — every field pulled out of a row is checked non-empty and numeric, and the row
#    count at the end must equal the number of rows this script asked for. This is the guard that
#    caught ab_d3_engine.sh parsing a `STAGE gran=burst family=TOTAL` line the engine does not emit.
#
# Every ratio is between two rows printed IN THIS WINDOW. Quoted absolutes are used only as canary
# targets, and are labelled QUOTED at the point of use.
set -uo pipefail
cd "$(dirname "$0")/../.."
export LC_ALL=C        # not cosmetic: a comma-decimal locale makes `sort -n` and awk read "3.3628" as 3
VENV=${VENV:-.venv/bin/python}
OUT=${OUT:-tmp/d2insitu}
TAG=${TAG:-d2-mlp-insitu}
# Override RIG only to dry-run this script against a stub rig on CPU; the real window uses the default.
RIG=${RIG:-prototype/mlp_insitu/mlp_insitu}
REPS=${REPS:-3}                 # reps per row (>=3, pre-registered: report median + spread)
ITERS=${ITERS:-6}               # tokens per rep
ROUNDS=${ROUNDS:-2}             # whole matrix passes; drift shows up as round disagreement
LAYERS=${LAYERS:-24}
GAP=${GAP:-42}                  # MiB of other-weight span per layer, layout=shipped72
QUOTED_BURST=${QUOTED_BURST:-2.275}    # QUOTED: engine/performance-status.md §4
MIN_PENALTY=${MIN_PENALTY:-0.35}        # §3.1 of D2-DIAGNOSIS.md, written before the window
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ROWS="$OUT/rows.tsv"; : > "$ROWS"
LOG="$OUT/rows.log"; : > "$LOG"

fail() { echo "ABORT: $*" >&2; echo "NO-TIMING: no millisecond figure from this run is usable" >&2 \
        ; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
num() { case "$1" in ''|*[!0-9.]*|*..*) return 1;; *) return 0;; esac; }
med() { sort -n "$1" | awk '{a[NR]=$1} END{if(NR==0){print "nan";exit} m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "%.4f", m}'; }
# Defined here, not in the report section, because the floor gate needs them before the matrix runs.
m() { awk -v t="$1" '$1==t && $3!="no_launch"{print $3}' "$ROWS" > "$OUT/.$1"; med "$OUT/.$1"; }
sp() { awk -v t="$1" '$1==t{print $3}' "$ROWS" | sort -n | awk 'NR==1{lo=$1} {hi=$1} END{printf "%s..%s", lo, hi}'; }
spread() { sort -n "$1" | awk '{if(n==""||$1<n)n=$1; if($1>x)x=$1} END{printf "%s..%s", n, x}'; }

# --------------------------------------------------------- CPU-only gates, before the lock ------
echo "== build (CPU) =="
bash prototype/mlp_insitu/build.sh > "$OUT/build.log" 2>&1 || fail "build failed, see $OUT/build.log"
echo "== --selftest-geom (CPU) =="
"$RIG" --selftest-geom > "$OUT/geom.log" 2>&1 || fail "--selftest-geom red, see $OUT/geom.log"
grep -q 'SELFTEST-GEOM ok fails=0' "$OUT/geom.log" || fail "--selftest-geom did not print ok fails=0"
echo "== census (CPU): the interferer doses are parsed, never typed =="
$VENV prototype/mlp_insitu/census.py --emit-stepspec > "$OUT/census.log" 2>&1 \
    || fail "census.py failed, see $OUT/census.log"
spec=$(grep -m1 '^STEPMIX ' "$OUT/census.log") || fail "census printed no STEPMIX line"
rpl=$(echo "$spec" | sed -n 's/.*read_mib_per_layer=\([0-9.]*\).*/\1/p')
wpl=$(echo "$spec" | sed -n 's/.*write_mib_per_layer=\([0-9.]*\).*/\1/p')
head=$(echo "$spec" | sed -n 's/.*head_mib=\([0-9.]*\).*/\1/p')
for v in rpl wpl head; do num "${!v}" || fail "census STEPMIX field $v did not parse ('${!v:-}')"; done
IF_READ=$(awk -v r="$rpl" -v L="$LAYERS" 'BEGIN{printf "%.3f", r*L}')
IF_WRITE=$(awk -v w="$wpl" -v L="$LAYERS" 'BEGIN{printf "%.3f", w*L}')
echo "  doses (MiB/token): if_read=$IF_READ if_write=$IF_WRITE head=$head layers=$LAYERS"
[ "$ROUNDS" -ge 2 ] 2>/dev/null || fail "ROUNDS must be >= 2: a single-pass matrix cannot show drift"

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
echo "== lock held: $(tools/gpu_lock.sh status 2>&1 | head -1) ; util at acquire: $(tools/gpu_lock.sh util 2>&1 | tail -1)"
# Untimed device witnesses, in-window, before any ms. Prints no ms and no GB/s.
if ! "$RIG" --selftest > "$OUT/selftest.log" 2>&1; then
    fail "--selftest red in-window (see $OUT/selftest.log): a rig whose witnesses do not behave is not an instrument"
fi
grep -q 'SELFTEST ok fails=0' "$OUT/selftest.log" || fail "--selftest did not print ok fails=0"
echo "  SELFTEST ok (76 cases, untimed, no ms printed)"

# ------------------------------------------------------- same-window normalization rows -------
if [ -x tools/build/mixed_bw ]; then
    for spec in "A_read_512m --mode read --footprint-mib 512" \
                "A_read_1536m --mode read --footprint-mib 1536" \
                "C_mixlaunch_62 --mode mixlaunch --ratio 62 --nbuf 48 --buf-mib 11.01"; do
        set -- $spec; tag=$1; shift
        if ! tools/build/mixed_bw "$@" --threads 256 --pages warm --iters 8 2>&1 | tee -a "$LOG"; then
            fail "mixed_bw $tag failed — the normalization rows are not optional"
        fi
    done
else
    echo "NOTE tools/build/mixed_bw is not built; the footprint-size control row is MISSING and the"
    echo "     (b) verdict in the report must be labelled as resting on the rig's rows alone."
fi

# ------------------------------------------------------------------------- the row runner -----
# `row <tag> <args...>` runs one arm, asserts it printed exactly one INSITU line with a numeric ms and
# verify=ok, and appends "tag round ms". Anything else aborts the window with NO-TIMING.
want_rows=0
row() {
    local tag="$1"; shift
    local log="$OUT/${tag}.log"
    if ! "$RIG" --time --tag "$tag" --layers "$LAYERS" --reps "$REPS" --iters "$ITERS" \
            "$@" > "$log" 2>&1; then
        sed 's/^/#   /' "$log" >&2
        fail "row $tag exited non-zero (see $log)"
    fi
    # DEFECT 3: an arm whose timed window launches nothing (--mlp 0) prints no milliseconds at all and
    # says so, because a 0.0001 from an empty sub-ms window gets tabulated as "free". Such a row is a
    # WITNESS row: it is recorded, it is not a number, and nothing subtracts it.
    if grep -q "^NO-TIMING tag=$tag no_launch=1" "$log"; then
        local v; v=$(grep -m1 "^NO-TIMING tag=$tag no_launch=1" "$log" | sed -n 's/.* verify=\([A-Za-z0-9,_]*\).*/\1/p')
        [ "$v" = ok ] || fail "witness row $tag: verify=$v"
        echo -e "$tag\t$ROUND\tno_launch" >> "$ROWS"
        grep '^NO-TIMING' "$log" >> "$LOG"
        printf '  %-6s %-4s %8s       (%s)\n' "$tag" "$ROUND" "no_launch" "$*"
        return 0
    fi
    local lines; lines=$(grep -c '^INSITU tag='"$tag"' ' "$log")
    [ "$lines" = 1 ] || fail "row $tag printed $lines INSITU lines, wanted exactly 1"
    local ms verify
    ms=$(grep '^INSITU tag='"$tag"' ' "$log" | sed -n 's/.* ms_per_token=\([0-9.]*\).*/\1/p')
    verify=$(grep '^INSITU tag='"$tag"' ' "$log" | sed -n 's/.* verify=\([A-Za-z0-9,_]*\).*/\1/p')
    num "$ms" || fail "row $tag: ms_per_token did not parse ('$ms') — empty-parse guard"
    [ "$verify" = ok ] || fail "row $tag: verify=$verify — a millisecond figure from unaccounted bytes is not a result"
    echo -e "$tag\t$ROUND\t$ms" >> "$ROWS"
    grep '^INSITU' "$log" >> "$LOG"
    printf '  %-6s %-4s %8s ms  (%s)\n' "$tag" "$ROUND" "$ms" "$*"
}

floor_row() { want_rows=$((want_rows+1)); row "$@"; }

# -------------------------------------------------------------------- FLOOR GATE, FIRST ------
# The first window spent its reproduction gate AFTER the matrix, which is why it had to be aborted by
# hand. These four rows run first, alone, and nothing else is paid for until they agree:
#   R0   thin  (48 launches of grid 448/128)          the shipped shape, alone
#   R0f  fat   (2 launches of grid 10,752/3,072)      the shape the published 2.275 ms was taken in
#   R0b  thin, 8 tokens between syncs                 were R0's launches pipelined after all?
#   R1   thin, one sync per launch                    the drain term (ledger: 48 x 7.2 us)
echo "== floor gate (reproduction + shape), before the matrix is spent =="
ROUND=0
floor_row R0  --layout blob3 --drain burst
floor_row R0f --layout blob3 --drain burst --geometry fat
floor_row R0b --layout blob3 --drain burst --token-burst 8
floor_row R1  --layout blob3 --drain each
# R7 is the expensive one, but canary 3 cannot be evaluated without it, and spending 20 rows to learn
# that P is under the floor is exactly what the first window did. So it runs in the gate too.
floor_row R7  --layout shipped72 --drain each --gap-mib "$GAP" \
              --if-read-mib "$IF_READ" --if-write-mib "$IF_WRITE" --head-mib "$head"

m0=$(m R0); m0f=$(m R0f); m0b=$(m R0b); m1=$(m R1); m7=$(m R7)
Pearly=$(awk -v a="$m7" -v b="$m0" 'BEGIN{printf "%.4f", a-b}')
dev=$(awk -v a="$m0f" -v q="$QUOTED_BURST" 'BEGIN{d=(a-q)/q*100; if(d<0)d=-d; printf "%.2f", d}')
num "$dev" || fail "reproduction delta did not parse ('$dev')"
echo "  reproduction: R0f(fat) $m0f vs QUOTED $QUOTED_BURST = $dev % (gate: <= 3 %)"
echo "  T_geo  = R0 - R0f = $(awk -v a="$m0" -v b="$m0f" 'BEGIN{printf "%.4f", a-b}') ms  (launch granularity: published shape vs shipped shape)"
echo "  T_pipe = R0 - R0b = $(awk -v a="$m0" -v b="$m0b" 'BEGIN{printf "%.4f", a-b}') ms  (per-token drain vs never-drained burst)"
echo "  T_serial = R1 - R0 = $(awk -v a="$m1" -v b="$m0" 'BEGIN{printf "%.4f", a-b}') ms  (band 0.25-0.45, ledger 0.3456 QUOTED)"
awk -v d="$dev" 'BEGIN{exit (d<=3.0)?0:1}' \
    || fail "canary C2: the rig cannot reproduce the published burst even at the published 2-launch geometry ($dev % > 3 %). This is a NO-RESULT, not a NO-DIAGNOSIS: the published 2.275 ms floor is not reachable here, so the 1.11 ms penalty needs re-measuring before any of this is admissible. NO-TIMING."
awk -v p="$Pearly" -v m="$MIN_PENALTY" 'BEGIN{exit (p>=m)?0:1}' \
    || fail "canary C3 (pre-registered, NOT moved): against the shipped-shape floor R0 the step reproduces only $Pearly ms of in-situ penalty, under the $MIN_PENALTY ms floor. That is a finding, not a fault: at 48-launch geometry there is no ~1 ms memory-state penalty to decompose — the QUOTED 1.09 ms gap is mostly T_geo (launch shape, above) plus the drain term. The share table would be shares of a number below its own measurement floor. Report the finding; do not fund code from it. NO-TIMING."
echo "  P (R7 - R0, shipped-shape floor) = $Pearly ms (gate: >= $MIN_PENALTY) — matrix is admissible"

for ROUND in $(seq "$ROUNDS"); do
    echo "== round $ROUND/$ROUNDS =="
    # --- the burst reproduction and the serialization term -------------------------------------
    want_rows=$((want_rows+1)); row R0  --layout blob3     --drain burst          # shipped shape, alone
    want_rows=$((want_rows+1)); row R0f --layout blob3     --drain burst --geometry fat
    want_rows=$((want_rows+1)); row R0b --layout blob3     --drain burst --token-burst 8
    want_rows=$((want_rows+1)); row R1  --layout blob3     --drain each           # T_serial = R1-R0
    # --- placement: region count, then span (bytes identical in all three) ----------------------
    want_rows=$((want_rows+1)); row R2  --layout blob72    --drain each
    want_rows=$((want_rows+1)); row R3  --layout shipped72 --drain each --gap-mib "$GAP"
    want_rows=$((want_rows+1)); row R3b --layout shipped72 --drain each --gap-mib 12
    # --- carry-over doses, drained (the only form a drained family can inherit) ----------------
    want_rows=$((want_rows+1)); row R4a --layout shipped72 --drain each --gap-mib "$GAP" --if-read-mib $(awk -v x="$IF_READ" 'BEGIN{printf "%.1f", x/4}')
    want_rows=$((want_rows+1)); row R4b --layout shipped72 --drain each --gap-mib "$GAP" --if-read-mib $(awk -v x="$IF_READ" 'BEGIN{printf "%.1f", x/2}')
    want_rows=$((want_rows+1)); row R4c --layout shipped72 --drain each --gap-mib "$GAP" --if-read-mib "$IF_READ"
    want_rows=$((want_rows+1)); row R5  --layout shipped72 --drain each --gap-mib "$GAP" --if-write-mib "$IF_WRITE"
    # --- the MLP's own stores: same bytes, destinations that cannot stay resident ----------------
    want_rows=$((want_rows+1)); row R6a --layout blob3     --drain each --writes spread --write-mul 1
    want_rows=$((want_rows+1)); row R6b --layout blob3     --drain each --writes spread --write-mul 16
    # --- the reproduction row: the step's own non-MLP bytes, drained ahead of every launch ------
    want_rows=$((want_rows+1)); row R7  --layout shipped72 --drain each --gap-mib "$GAP" \
                                        --if-read-mib "$IF_READ" --if-write-mib "$IF_WRITE" --head-mib "$head"
    # --- the concurrency BOUND (not a reproduction): two non-default streams ---------------------
    want_rows=$((want_rows+1)); row R8  --layout shipped72 --drain burst --gap-mib "$GAP" \
                                        --if-when concurrent --if-read-mib "$IF_READ" \
                                        --if-write-mib "$IF_WRITE" --head-mib "$head"
    # --- interferer alone: a WITNESS row, not a number. With no MLP in flight it proves the
    # interferer's byte accounts stand on their own; it prints no_launch=1 and nothing subtracts it.
    want_rows=$((want_rows+1)); row R9  --mlp 0 --layout shipped72 --drain each --gap-mib "$GAP" \
                                        --if-read-mib "$IF_READ" --if-write-mib "$IF_WRITE" --head-mib "$head"
    # --- the first fork: fixed per launch vs proportional to bytes -------------------------------
    want_rows=$((want_rows+1)); row RAg --kernels gateup  --layout blob3 --drain burst
    want_rows=$((want_rows+1)); row RBg --kernels gateup  --layout blob3 --drain each
    want_rows=$((want_rows+1)); row RAu --kernels down    --layout blob3 --drain burst
    want_rows=$((want_rows+1)); row RBu --kernels down    --layout blob3 --drain each
done
tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1
trap - EXIT

# ------------------------------------------------------------------------- the report -------
got=$(wc -l < "$ROWS")
[ "$got" = "$want_rows" ] || fail "printed $got rows but asked for $want_rows — a row list whose count is not its own is not a result"

R0=$(m R0) R0f=$(m R0f) R0b=$(m R0b)
R1=$(m R1) R2=$(m R2) R3=$(m R3) R3b=$(m R3b) R4a=$(m R4a) R4b=$(m R4b) R4c=$(m R4c)
R5=$(m R5) R6a=$(m R6a) R6b=$(m R6b) R7=$(m R7) R8=$(m R8)
RAg=$(m RAg) RBg=$(m RBg) RAu=$(m RAu) RBu=$(m RBu)
# Canary 2 was evaluated in the floor gate BEFORE the matrix was paid for; it is recomputed here on the
# full median set. It is the SAME-SHAPE comparison: the published 2.275 ms is a 2-launch burst
# (mlp_gemv derives rows from the weight FILE SIZE, so --tile 24 is one launch of 86,016 rows per
# kernel, divided by 24), and R0f is that shape. R0 — the shipped 48-launch shape — is NOT expected to
# hit 2.275; the gap between them is T_geo, a named term. The 2.275 target is not moved.
dev=$(awk -v a="$R0f" -v q="$QUOTED_BURST" 'BEGIN{d=(a-q)/q*100; if(d<0)d=-d; printf "%.2f", d}')
awk -v d="$dev" 'BEGIN{exit (d<=3.0)?0:1}' \
    || fail "canary 2 (recomputed on all rounds): R0f (fat, the published shape) reads $R0f ms against the QUOTED $QUOTED_BURST ($dev % > 3 %). The published floor is not reproducible from this rig at all, so the 1.11 ms penalty itself needs re-measuring before any code is funded. NO-TIMING."
T_geo=$(awk -v a="$R0" -v b="$R0f" 'BEGIN{printf "%.4f", a-b}')
T_pipe=$(awk -v a="$R0" -v b="$R0b" 'BEGIN{printf "%.4f", a-b}')
# Canary 3, PRE-REGISTERED AT 0.35 ms and NOT moved. What changed is what it is measured against: P is
# R7 - R0 where R0 is now explicitly the SHIPPED-shape floor, so a trip here no longer means "the rig
# cannot see the penalty", it means the penalty at the shipped geometry is small and the ledger's 1.11 ms
# is mostly T_geo. That is a result, and it is reported as one rather than routed around.
P=$(awk -v a="$R7" -v b="$R0" 'BEGIN{printf "%.4f", a-b}')
awk -v p="$P" -v m="$MIN_PENALTY" 'BEGIN{exit (p>=m)?0:1}' \
    || fail "canary 3: NO-DIAGNOSIS. Against the shipped-shape floor R0 the rig reproduces only $P ms of in-situ penalty (pre-registered floor $MIN_PENALTY ms). Read it as the finding it is: at 48-launch geometry the step's memory-state carry-over is ~$P ms, and the QUOTED 1.09 ms burst-to-stage gap is dominated by T_geo=$T_geo ms of launch shape plus the drain term, not by a memory mechanism. The D2 prize must be re-cut against that before code is funded. NO-TIMING."

echo
echo "=== D2 attribution, all ratios within this window (QUOTED absolute used only as the canary)"
printf "  %-30s %8s   spread %s\n" "R0  shipped-shape floor (thin, 48 launches)" "$R0" "$(sp R0)"
printf "  %-30s %8s   spread %s\n" "R0f published-shape floor (fat, 2 launches)" "$R0f" "$(sp R0f)"
printf "  %-30s %8s   spread %s\n" "R0b thin, 8 tokens per sync" "$R0b" "$(sp R0b)"
printf "  %-30s %8s   spread %s\n" "R7  step-shaped carry-over" "$R7" "$(sp R7)"
printf "  %-30s %8s  %s\n" "T_geo = R0 - R0f" "$T_geo" "launch granularity: published shape minus shipped shape (not a term of P)"
printf "  %-30s %8s  %s\n" "T_pipe = R0 - R0b" "$T_pipe" "per-token drain vs a never-drained burst (tests the pipelining claim)"
T_serial=$(awk -v a="$R1" -v b="$R0" 'BEGIN{printf "%.4f", a-b}')
T_region=$(awk -v a="$R2" -v b="$R1" 'BEGIN{printf "%.4f", a-b}')
T_span=$(awk -v a="$R3" -v b="$R2" 'BEGIN{printf "%.4f", a-b}')
T_read=$(awk -v a="$R4c" -v b="$R3" 'BEGIN{printf "%.4f", a-b}')
T_write=$(awk -v a="$R5" -v b="$R3" 'BEGIN{printf "%.4f", a-b}')
# The lm_head's 486 MiB is placed ONCE per step, not once per layer, so the --if-read-mib dose rows never
# carry it: R4c prices 471 MiB of per-layer traffic and the head sits in R7 alone. Attributing it to
# T_readcarry would be a mislabeled 486 MiB, and leaving it unnamed would hide the step's single largest
# read inside the residue. Estimator: whatever R7 costs over R4c and over the write term.
T_head=$(awk -v a="$R7" -v b="$R4c" -v w="$T_write" 'BEGIN{printf "%.4f", a-b-w}')
T_own=$(awk -v a="$R6b" -v b="$R1" 'BEGIN{printf "%.4f", a-b}')
T_conc=$(awk -v a="$R8" -v b="$R0" 'BEGIN{printf "%.4f", a-b}')   # R9 prints no number (no_launch)
sum=$(awk -v a="$T_serial" -v b="$T_region" -v c="$T_span" -v d="$T_read" -v e="$T_write" -v f="$T_head" 'BEGIN{printf "%.4f", a+b+c+d+e+f}')
resid=$(awk -v p="$P" -v s="$sum" 'BEGIN{printf "%.4f", p-s}')
# One awk for the whole table. The per-row `$(awk ...)` version of this was a syntax error, and it is
# also the shape that hides a bad parse: one program, all the terms as -v args, verdicts derived from
# the same pre-registered thresholds for every row.
awk -v P="$P" -v ts="$T_serial" -v tr="$T_region" -v tp="$T_span" -v tc="$T_read" \
    -v tw="$T_write" -v th="$T_head" -v to="$T_own" -v tx="$T_conc" -v sm="$sum" -v rs="$resid" 'BEGIN{
  printf "  %-30s %8s\n", "P = R7 - R0 (reproduced penalty)", P
  n = split(ts "\n" tr "\n" tp "\n" tc "\n" tw "\n" th "\n" to "\n" tx "\n" sm "\n" rs, v, "\n")
  lab[1]="T_serial (48 drains, not memory)"; lab[2]="T_region (3 -> 72 blobs)"
  lab[3]="T_span (504 MiB -> 1.5 GiB)";      lab[4]="T_readcarry (per-layer, 471 MiB)"
  lab[5]="T_writecarry (candidate c-prime)"; lab[6]="T_headcarry (lm_head, 486 MiB)"
  lab[7]="T_ownwrites (candidate c, upper bd)"; lab[8]="cross_concurrent (a BOUND only)"
  lab[9]="sum of named terms"; lab[10]="residue (UNEXPLAINED, stays named)"
  note[7]="  not additive: same bytes, destinations that cannot stay resident"
  note[8]="  not additive: t(R8) - t(R0), the cost of adding the step traffic concurrently"
  note[9]=""
  for (i = 1; i <= 10; i++) {
    sh = v[i] / P
    if (i == 7 || i == 8) verdict = ""
    else if (i >= 9)      verdict = sprintf("share %.3f", sh)
    else verdict = sprintf("share %.3f  %s", sh, (sh >= 0.25 ? "CONFIRMED" : (sh <= 0.10 ? "refuted" : "partial")))
    printf "  %-34s %8s  %s%s\n", lab[i], v[i], verdict, note[i]
  }
  print "  (thresholds pre-registered in D2-DIAGNOSIS.md §1: CONFIRMED >= 0.25 of P, refuted <= 0.10)"
}'

echo
echo "=== the ledger cross-check on the per-segment constant (attribution §4: 7.2 us x segments)"
awk -v t="$T_serial" 'BEGIN{printf "  T_serial = %.4f ms; 48 x 7.2 us = 0.3456 ms QUOTED; ratio %.2f\n", t, t/0.3456
    if (t < 0.25 || t > 0.45) print "  OUTSIDE the pre-registered 0.25-0.45 ms band: the per-family prize table built on that constant is the finding, not the attribution."
    else print "  inside the band: the rig drain shape and the stage clock agree."}'
echo
echo "=== the first fork: fixed per launch, or proportional to bytes?"
awk -v ag="$RAg" -v bg="$RBg" -v au="$RAu" -v bu="$RBu" 'BEGIN{
    dg=bg-ag; du=bu-au;
    printf "  gateup (14.7 MiB weights) burst %.4f -> each %.4f  delta %.4f ms\n", ag, bg, dg
    printf "  down    (7.34 MiB weights) burst %.4f -> each %.4f  delta %.4f ms\n", au, bu, du
    r=(du>0)?dg/du:0
    printf "  delta ratio gateup/down = %.2f; bytes ratio is 2.00.\n", r
    if (r > 1.6) print "  -> proportional: the penalty scales with BYTES. Memory mechanism."
    else if (r < 1.25) print "  -> fixed per launch: the penalty is drain/dispatch/empty-queue. Step shape, not memory."
    else print "  -> between: report it as between and let the dose rows carry the split."
}'
echo
echo "=== the prediction, as pre-registered (D2-DIAGNOSIS.md §3)"
awk -v P="$P" -v ts="$T_serial" -v tr="$T_read" -v th="$T_head" -v tw="$T_write" -v tsp="$T_span" 'BEGIN{
    printf "  (traffic state)  (readcarry+writecarry+headcarry)/P = %.2f  predicted >= 0.40 : %s\n", (tr+tw+th)/P, ((tr+tw+th)/P>=0.40?"held":"REFUTED")
    printf "  (placement)      T_span/P                          = %.2f  predicted <= 0.15 : %s\n", tsp/P, (tsp/P<=0.15?"held":"REFUTED — page walk is the mechanism")
    printf "  (instrument)     T_serial/P                        = %.2f  refutation line >= 0.55 : %s\n", ts/P, (ts/P>=0.55?"REFUTED the route as written":"not crossed")
}'
echo
echo "# raw: $ROWS ; every row's full log: $OUT/*.log ; INSITU lines: $LOG"
echo "# quote each term with its two source rows and the window's spread, never as a best."
