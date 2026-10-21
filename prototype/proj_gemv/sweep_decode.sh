#!/usr/bin/env bash
# prototype/proj_gemv/sweep_decode.sh — the D1 decode-projection shape search, in ONE GPU lock hold.
#
# THE MAIN AGENT RUNS THIS. It times, and a timing taken while this session is streaming tokens is
# contaminated (AGENTS.md: "a GPU number taken while the main agent is streaming ... is contended by
# its own tokens"). Everything in this file is rig: the worker who wrote it took no millisecond figure
# anywhere, and the published numbers it compares against are quoted from results.md with their commit.
#
# WHAT THIS VERSION FIXES, from the first window (main agent, 2026-10-21, one hold, util 9 -> 79 %):
#
#  1. ROWS/BLOCK AND THREADS/BLOCK WERE ONE NUMBER. `threads = rpb * lanes_per_row` made the thread
#     count untestable independently of rows/block. `--threads` is now a real axis — on the
#     prototype-only variants, because the shipped kernel refuses a shape the engine cannot emit — and
#     it is a column of the arm table and a canary field. See the control-arm note below for what the
#     published rows' thread counts actually were.
#  2. A REP RAN 5x SLOW AND WAS REPORTED AS THE RESULT. One config read min 3.544 / max 17.497 ms and
#     `ms_per_plan` was the LAST rep, so 17.497 — 26 GB/s, the fresh-page rate — was the headline.
#     Every timed point is now warmed first (`--warm`, default 2 whole-plan passes), `ms_per_plan` is
#     the MIN (best-of-N, prototype/README.md's convention), max/min > 1.10 prints `provisional`, and a
#     point still above 2x its own min after warming prints `VOID` and no millisecond figure at all.
#  3. THE GATED SET WAS NOT THE TIMED SET. Admission judged each variant at its default rows/block while
#     the sweep timed three of them — the D59 trap. One file, `$OUT/gate_grid.txt`, now lists the
#     points, is passed to `run.py --grid-file` for admission and drives the timed loop, and the script
#     prints `gated N, timed M, N==M` and refuses to call any row adjudicable otherwise.
#
# And from the second window (the DVFS window: main agent, 2026-10-21, blocking, zero subagents, one
# hold, util 9 -> 30 %, 13 s of GPU work, gated 15 = timed 15, 1710 comparisons 0 failures):
#
#  4. NO CLOCK WARM-UP. 13 s of GPU work is shorter than the ramp to boost, so the whole window drifted:
#     the three shipped-kernel controls came in 18.3 / 19.8 / 17.7 % above their published rows while
#     agreeing with each other to 1-2 %, and the arms ran monotonically faster down the table. Now: a
#     sustained untimed `--mode burn` on the real plan before arm 1 (`BURN` seconds, wall-timed out), a
#     `rocm-smi --showclocks` reading on EVERY row plus the window edges, and any row below `CLK_MIN`
#     (default 2300 MHz) prints `LOWCLK` and is excluded from every pool and verdict.
#  5. ONE PASS, ONE ORDER. Arms now run in a FIXED pseudo-random interleaved order (seed `SEED`, printed
#     and recorded in `$OUT/arm_order.txt`) and the whole set runs `PASSES` times (default 2); the
#     reported figure per point is the MIN across passes, and two ramp diagnostics (pass-over-pass, and
#     position within pass 1) say out loud whether the ramp survived interleaving.
#  6. PHASE C ASKED THE WRONG VARIANT. The plan pair ran `--threads 1024` against `v0-baseline`, which
#     refuses a thread count the engine cannot emit, so both arms exited 1. The pair is now driven by the
#     winning SEARCH arm's own variant (v1-unroll or v5-vx, both bit-identical to v0) at the shape the
#     search won, which is the only variant that can hold that shape.
#
# Usage:  prototype/proj_gemv/sweep_decode.sh [ITERS] [REPS] [ADMISSION_STEPS]
#           ITERS  burst length in whole plans per rep (default 30, the published convention)
#           REPS   independent timed bursts per point, one upload (default 3)
#           ADMISSION_STEPS  run.py --steps for the gate (default decode0; `all` is ~8x slower)
#         Env: SKIP_GATE=1 skips phase 1 (rows are then printed as un-admitted)
#              NO_TIMING=1 phases 0-1 only (grid + admission), no burn, nothing measured
#              WARM=<n>     warm passes per point (default 2)
#              OUT=<dir>    plans and logs (default /tmp/proj_gemv/sweep)
#              BURN=<sec>   clock burn before arm 1, wall seconds (default 5; 0 disables it — say so)
#              PASSES=<n>   times the whole interleaved set is run (default 2)
#              SEED=<s>     the order's seed; the order is printed either way (default d1-interleave-2)
#              CLK_MIN=<MHz>floor for a row to count as a measurement (default 2300)
#              CLK_TOOL=<path> clock reader (default rocm-smi; `none` disables it and the LOWCLK rule,
#                       which is reported loudly rather than silently, because rows would all be clk=?)
#         `sweep_decode.sh selftest` checks the canary, the verdict thresholds, the clock parser and the
#         field parser without a GPU — including a VOID line and a sub-floor clock. The parser cannot be
#         validated on a real burst, because that would be this worker taking a timing.
#
# Refuses to print a millisecond figure if the admission gate is red — precedent
# prototype/attn_m2/burst_m16.sh ("NO-TIMING: admission is red"). A number from a run whose correctness
# was not checked in the same window is not a result, it is a rumour.

set -u

# Doubles as CLK_TOOL: `CLK_TOOL=<this file> --showclocks` must look like rocm-smi, so a dry run can also
# exercise the LOWCLK path (STUB_CLK=800 puts every row under the floor).
if [ "${1:-}" = --showclocks ]; then
  printf 'GPU[0]\t\t: sclk clock level: 1: (%sMhz)\n' "${STUB_CLK:-2700}"
  exit 0
fi
# Numeric parsing must not depend on who runs this. Under a comma-decimal locale (this box's shell is
# LC_NUMERIC=nl_NL.UTF-8) awk reads "3.550000" as the integer 3, which would have made every minimum,
# control offset and ramp ratio here meaningless — min across passes would compare 3 with 3 and 2 with 2.
# The first window escaped this only because that shell happened to be a C locale.
export LC_ALL=C
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="${BIN:-$HERE/proj_gemv}"   # override to stub_result.sh for a no-GPU dry run of the timed path
PY="$ROOT/.venv/bin/python"
LOCK="$ROOT/tools/gpu_lock.sh"
ITERS="${1:-30}"
REPS="${2:-3}"
STEPS="${3:-decode0}"
WARM="${WARM:-2}"
OUT="${OUT:-/tmp/proj_gemv/sweep}"

# --------------------------------------------------------------------------- the arm table
# One row is one launcher invocation = one weight upload, and prints one line per rows/block value.
# Fields: tag | variant | plan | rows/block-list | threads | want_ms | want_tol
#   threads 0   = the variant's own rpb*lanes_per_row (v0: 256/512/1024 at rpb 8/16/32).
#   want_ms     = a published number this point MUST reproduce within want_tol (a fraction); empty
#                 means no published number exists at this geometry: report it, adjudicate nothing.
#
# WHY A1-A3 ARE THE CONTROLS, AND AT THOSE THREAD COUNTS. `results.md`'s rows/block sweep on the token
# plan reads rpb 8 -> 2.9895, rpb 16 -> 2.946, rpb 32 -> 2.785. Those came out of the launcher at
# 5d6405f, whose thread line was `rpb * warpSize` — so the published sweep never held threads fixed:
# its 512-thread point IS rpb 16, and 2.785 was rpb 32 at 1024 threads. The prose at results.md:34
# ("512 threads, one work item = one output row") describes spec §07.3's form, not the kernel those
# numbers came out of. Each control therefore reproduces its own published row at the thread count that
# row actually used; A4/A5 price the reading the stale prose invites ("rpb 8/32 at 512 threads"), on
# v1-unroll, which is bit-identical to v0 and takes a free thread count — so any difference there is
# geometry and nothing else.
ARMS=(
  "A1|v0-baseline|packed|8|0|2.9895|0.03"
  "A2|v0-baseline|packed|16|0|2.946|0.03"
  "A3|v0-baseline|packed|32|0|2.785|0.03"
  "A4|v1-unroll|packed|8|512||"
  "A5|v1-unroll|packed|32|512||"
  "B-v1-t512|v1-unroll|packed|8,16,32|512||"
  "B-v1-t1024|v1-unroll|packed|8,16,32|1024||"
  "B-v5-t512|v5-vx|packed|8,16,32|512||"
  "B-v5-t1024|v5-vx|packed|8,16,32|1024||"
)

expected_launches() { case "$1" in packed) echo 48;; unpacked|mixed) echo 114;; *) echo 0;; esac; }
lanes_of() { case "$1" in v2-lpw16) echo 16;; v3-lpw8|v6-lpw8-vx) echo 8;; v4-lpw4) echo 4;; *) echo 32;; esac; }
# The threads/block the launcher must echo: the request, or rpb*lanes when 0 was asked for.
want_threads() { local thr="$1" rpb="$2" var="$3"; [ "$thr" != 0 ] && { echo "$thr"; return; }
                 echo $(( rpb * $(lanes_of "$var") )); }

# ---- the clock -------------------------------------------------------------------------------
# The DVFS confound the second window exposed: 13 s of GPU work is shorter than the clock's ramp to
# boost, so an arm that ran early was slower for a reason that has nothing to do with its geometry.
# progress/2026-10-21/performance-status-2026-10-21.md §36 measured this part at 819/827/838 MHz idle
# and 2432/2437/2423 MHz immediately after a rep. Three responses, all in the rig: a sustained untimed
# BURN before arm 1 (`--mode burn`, timed out by wall seconds), a clock reading attached to EVERY row
# with a LOWCLK marker below CLK_MIN, and a fixed interleaved order run twice so a ramp cannot
# masquerade as a ranking.
CLK_TOOL="${CLK_TOOL:-rocm-smi}"
CLK_MIN="${CLK_MIN:-2300}"      # below this a row is LOWCLK, i.e. not a measurement
CLK_NORM="${CLK_NORM:-2400}"    # divisor of the clock-normalised column; a READING aid, never a verdict
ARM_BURN="${ARM_BURN:-1.5}"     # seconds of in-process burn immediately before each timed burst
CLKS_OK=0; CLKS_BAD=0
# parse_sclk — pull the MHz out of `rocm-smi --showclocks`, whose line reads
#   GPU[0]  : sclk clock level: 1: (2642Mhz)
# Written as a filter so selftest can feed it a canned line: a clock probe whose parser matches nothing
# turns every row into `clk=?` and the LOWCLK rule into decoration (AGENTS.md, the empty-parse family).
parse_sclk() { sed -n 's/.*sclk clock level: [0-9]*: (\([0-9]*\)Mhz).*/\1/p' | head -1; }
clock_mhz() {
  [ -n "$CLK_TOOL" ] || { return; }
  local v; v="$("$CLK_TOOL" --showclocks 2>/dev/null | parse_sclk)"
  printf '%s' "$v"
}
# clk_note — called in the CALLER's shell, because clock_mhz runs in a command substitution and a counter
# it bumped would die with the subshell: the tally would read 0 parsed on a window with 60 good reads.
clk_note() { if [ -n "${1:-}" ]; then CLKS_OK=$((CLKS_OK + 1)); else CLKS_BAD=$((CLKS_BAD + 1)); fi; }
# clk_verdict CLK — empty when the row stands; LOWCLK below the floor; CLK-UNKNOWN when nothing could
# be read. A clock you cannot see is not a clock that looks fine.
clk_verdict() {
  local c="${1:-}"
  [ -z "$c" ] && { echo "CLK-UNKNOWN"; return; }
  [ "$c" -lt "$CLK_MIN" ] && { echo "LOWCLK"; return; }
  echo ""
}
# norm_ms MS CLK — the clock-normalised figure: ms a row would have read at CLK_NORM MHz, given the clock
# it actually ran at. It exists because what moves between windows is sclk while mclk sits at 1000 MHz, so
# two rows of a latency-bound kernel at different clocks are not comparable as ms. It is a READING column:
# the WIN/KILL call is made on raw ms at a clock that passed the floor, never on this.
norm_ms() { local ms="${1:-}" clk="${2:-}"
  case "$ms" in ""|VOID) echo "-"; return;; esac
  [ -z "$clk" ] && { echo "-"; return; }
  awk -v m="$ms" -v c="$clk" -v n="$CLK_NORM" 'BEGIN{printf "%.6f", m*c/n}'
}

# row / row_hdr — the one table format. A column added in the print sites but forgotten in one of the four
# error paths is how a table starts lying about its own header.
row() { printf '%-4s %-11s %-12s %-4s %-7s %-5s %-5s %-8s %9s %9s %8s %6s %6s %-11s  %s\n' "$@"; }
row_hdr() { row PASS TAG VARIANT ROWS THREADS LANES LOADS PLAN MS "NORM@$CLK_NORM" GB/S %PROBE CLK VERDICT NOTE; }
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }
# g FIELD LINE — one `key=value` field out of a RESULT line. Top level so selftest can exercise it.
g() { local kv; for kv in $2; do [ "${kv%%=*}" = "$1" ] && { printf '%s' "${kv#*=}"; return; }; done; }

# The canary. Given the point's variant / expected launches / rows / threads and one RESULT line, prints
# "" when the launcher agrees with the request and a reason when it does not. A canary that cannot fail
# is the failure mode AGENTS.md records three times over, so selftest feeds it five wrong lines.
canary_check() { # $1=variant $2=want_loads $3=want_rpb $4=want_threads $5=resultline
    local var="$1" want="$2" wrpb="$3" wthr="$4" line="$5" bad=""
    local label="" launches="" rpb="" threads=""
    for kv in $line; do
      case "$kv" in
        label=*) label="${kv#label=}";;
        launches=*) launches="${kv#launches=}";;
        rpb=*) rpb="${kv#rpb=}";;
        threads=*) threads="${kv#threads=}";;
      esac
    done
    [ -n "$label" ] || bad+="no-label "
    [ -n "$rpb" ] && [ -n "$threads" ] || bad+="no-geometry "
    [ "$label" = "$var" ] || bad+="label=$label wants $var "
    [ "$launches" = "$want" ] || bad+="launches=$launches wants $want "
    [ "$rpb" = "$wrpb" ] || bad+="rpb=$rpb wants $wrpb "
    [ "$threads" = "$wthr" ] || bad+="threads=$threads wants $wthr "
    printf '%s' "$bad"
}

# verdict LINE — what a RESULT line may be reported as: ok | provisional | void. The launcher already
# prints `pt`, and the ratio is re-read here rather than trusted: a launcher that stopped flagging its
# own bimodal points would still be caught by the sweep.
verdict() { # $1=resultline
    local line="$1" pt ratio
    pt="$(g pt "$line")"; ratio="$(g ratio "$line")"
    [ "$pt" = void ] && { echo void; return; }
    [ -n "$ratio" ] && awk -v r="$ratio" 'BEGIN{ exit !(r > 2.0) }' && { echo void; return; }
    [ -n "$ratio" ] && awk -v r="$ratio" 'BEGIN{ exit !(r > 1.10) }' && { echo provisional; return; }
    echo "${pt:-ok}"
}

# ---- the summary: pooled min per point, and whether the ramp survived ----------------------------
# One row per point: both passes, their min, and pass2/pass1. A point whose every row was VOID or below
# the clock floor has no pooled figure, which is a window failure and not a slow arm.
#
# A function taking (record file, summary file) and returning its verdicts in SUM_FAIL / SUM_WIN_* so
# selftest can drive it with a canned file: this worker may not take a timing, so the pooling rules get
# their witness there or nowhere.
summarise() {
  local REC="${1:?summarise: need the record file}" SUM="${2:?summarise: need the summary file}"
  local tag var shape rpb thr p1 p2 mn ratio want tol nuse ctrl rv
  SUM_FAIL=0; SUM_WIN_TAG=""; SUM_WIN_VAR=""; SUM_WIN_RPB=""; SUM_WIN_THR=""; SUM_WIN_MS=""
  : > "$SUM"
  # `|`-separated, not tab: TAB IS IFS WHITESPACE, so `while IFS=$'\t' read` collapses runs of tabs and an
  # empty field (a search arm has no published number) shifts every column after it. That is how a canned
  # selftest row read its usable-row count as its published target and reported a 20.0 % control miss.
  awk -F'|' '
    { key = $3 "," $4 "," $5 "," $6 "," $7
      if (!(key in seen)) { seen[key] = 1; order[++n] = key; want[key] = $12; tol[key] = $13 }
      if ($11 == "1") {
        usable[key]++
        if (!(key in mn) || $8 + 0 < mn[key] + 0) mn[key] = $8
        if ($14 != "" && $14 != "-") { if (!(key in nm) || $14 + 0 < nm[key] + 0) nm[key] = $14 }
        p[$1 "|" key] = $8
      }
    }
    END {
      for (i = 1; i <= n; i++) {
        k = order[i]; split(k, a, ",")
        r = (("1|" k) in p && ("2|" k) in p && p["1|" k] + 0 > 0) ? p["2|" k] / p["1|" k] : 0
        printf "%s|%s|%s|%s|%s|%s|%s|%s|%.3f|%s|%s|%d|%s\n", a[1], a[2], a[3], a[4], a[5], \
               (("1|" k) in p ? p["1|" k] : "-"), (("2|" k) in p ? p["2|" k] : "-"), \
               ((k in mn) ? mn[k] : "NONE"), r, want[k], tol[k], usable[k] + 0, \
               ((k in nm) ? nm[k] : "-")
      }
    }' "$REC" > "$SUM"

  printf '%-11s %-12s %-8s %-4s %-7s %9s %9s %9s %9s %7s %7s  %s\n' \
         TAG VARIANT PLAN ROWS THREADS PASS1 PASS2 MIN "NORMMIN" P2/P1 CTRL VERDICT
  while IFS='|' read -r tag var shape rpb thr p1 p2 mn ratio want tol nuse nrm; do
    ctrl="-"; rv="ok"
    if [ "$mn" = NONE ]; then
      rv="NO-USABLE-ROW"; SUM_FAIL=$((SUM_FAIL + 1))
    elif [ -n "$want" ]; then
      ctrl=$(awk -v m="$mn" -v w="$want" -v t="$tol" 'BEGIN{o=(m-w)/w; if(o<0)o=-o; printf "%.1f%%", 100*o}')
      if awk -v m="$mn" -v w="$want" -v t="$tol" 'BEGIN{o=(m-w)/w; if(o<0)o=-o; exit !(o>t)}'; then
        rv="off-published"; SUM_FAIL=$((SUM_FAIL + 1))
      fi
    fi
    # The plan pair is priced at the best packed SEARCH arm (a tag starting in B), and only at a variant
    # that owns the threads axis: the shipped kernel refuses a thread count the engine cannot emit, which
    # is what broke C in the second window.
    case "$tag" in
      B*) if [ "$mn" != NONE ] && [ "$shape" = packed ] &&
            { [ -z "$SUM_WIN_MS" ] || awk -v a="$mn" -v b="$SUM_WIN_MS" 'BEGIN{exit !(a < b)}'; }; then
            SUM_WIN_MS="$mn"; SUM_WIN_RPB="$rpb"; SUM_WIN_THR="$thr"; SUM_WIN_VAR="$var"; SUM_WIN_TAG="$tag"
          fi;;
    esac
    printf '%-11s %-12s %-8s %-4s %-7s %9s %9s %9s %9s %7s %7s  %s\n' \
           "$tag" "$var" "$shape" "$rpb" "$thr" "$p1" "$p2" "$mn" "$nrm" "$ratio" "$ctrl" \
           "$rv ($nuse usable row(s))"
  done < "$SUM"
  log "NORMMIN = min over usable rows of ms * clk / $CLK_NORM MHz. It is a READING column: it makes a row "
  log "taken at 1750 comparable with one taken at 2450, which is all the cross-window comparison can lean on. "
  log "The WIN/KILL call is made on raw ms from a row whose clock passed the $CLK_MIN MHz floor, never on this."

  # Did the ramp survive interleaving? Two independent reads, printed either way because a null here is
  # the result. (a) pass 2 against pass 1 on the same arms: a clock still climbing makes one pass uniformly
  # faster. (b) a PAIRED position statistic: each arm's residual off its own two-pass mean, correlated with
  # how far its position moved between the two passes. Identity-free by construction, which a
  # first-half/second-half mean comparison is not.
  awk -F'|' '
    $11 == "1" { s[$1] += $8; c[$1]++ }
    END {
      if (c[1] && c[2]) {
        d = (s[2] / c[2]) / (s[1] / c[1])
        printf "ramp, pass over pass: mean pass1 %.4f ms vs pass2 %.4f ms, ratio %.4f\n", s[1]/c[1], s[2]/c[2], d
        if (d < 0.98) printf "  -> pass 2 ran %.1f%% faster: the clock was STILL CLIMBING across the whole set. The min across passes is the only quotable figure; raise BURN.\n", (1-d)*100
        else if (d > 1.02) printf "  -> pass 2 ran %.1f%% slower: drift the other way (thermal or contention); same remedy, quote the min.\n", (d-1)*100
        else printf "  -> the two passes agree within 2%%: no monotone clock ramp across the set.\n"
      } else printf "ramp, pass over pass: not enough usable rows in both passes to compare\n"
    }' "$REC"
  awk -F'|' '
    # key -> per-arm mean of the two passes, and the position offset within the arm.
    $11 == "1" { key = $3 "~" $4 "~" $5 "~" $6 "~" $7
                 m[key "#" $1] = $8; p[key "#" $1] = $2; k[++n] = key }
    END {
      np = 0
      for (i = 1; i <= n; i++) {
        key = k[i]
        if (!((key "#1") in m) || !((key "#2") in m)) continue
        mean = (m[key "#1"] + m[key "#2"]) / 2
        pm = (p[key "#1"] + p[key "#2"]) / 2
        xp[++np] = p[key "#1"] - pm; yp[np] = m[key "#1"] / mean - 1
        xp[++np] = p[key "#2"] - pm; yp[np] = m[key "#2"] / mean - 1
      }
      if (np < 4) { print "ramp, position (paired): not enough arms measured in both passes"; exit }
      sx = 0; sy = 0; sxx = 0; syy = 0; sxy = 0
      for (i = 1; i <= np; i++) { sx += xp[i]; sy += yp[i]; sxx += xp[i]*xp[i]; syy += yp[i]*yp[i]; sxy += xp[i]*yp[i] }
      den = sqrt((np*sxx - sx*sx) * (np*syy - sy*sy))
      r = (den == 0) ? 0 : (np*sxy - sx*sy) / den
      printf "ramp, position (paired, %d rows from %d arms measured in both passes): correlation between position offset and ms residual r = %+.3f\n", np, np/2, r
      if (r <= -0.30) printf "  -> later positions are faster even within an arm across the two orders: a monotone ramp SURVIVED interleaving. Treat the ranking as clock-contaminated; raise BURN and re-take.\n"
      else if (r >= 0.30) printf "  -> later positions are SLOWER within an arm: drift the other way (thermal throttling or contention); the min across passes is the quotable figure.\n"
      else printf "  -> no position effect within arms: interleaving did its job, the order is not what ranked the arms.\n"
    }' "$REC"
}

selftest() {
  # Field-consistency check first: the canned lines below are only a witness if they are the lines the
  # launcher prints. Rename a field in the C++ and this fails instead of quietly parsing a string that
  # no longer matches the binary.
  local src="$HERE/proj_gemv.hip.cpp" f blob
  blob="$(grep -A5 'RESULT mode=burst' "$src" | tr -d '\n')"
  for f in label rpb lanes_per_row threads launches ms_per_plan ms_min ms_max reps ratio pt warm preburn_s preburn_plans; do
    case "$blob" in *"$f="*) ;; *) echo "SELFTEST-FAIL: '$f' is not in the burst RESULT printf in $src"; return 1;; esac
  done
  echo "SELFTEST ok   every field the parser reads exists in the burst RESULT printf (incl. the per-arm burn)"
  # The clock-normalised figure, and the rule that it never adjudicates.
  [ "$(norm_ms 3.500000 2400)" = "3.500000" ] \
    || { echo "SELFTEST-FAIL: norm_ms at the divisor itself read $(norm_ms 3.500000 2400), wants 3.500000"; return 1; }
  [ "$(norm_ms 2.500000 1200)" = "1.250000" ] \
    || { echo "SELFTEST-FAIL: norm_ms at half the divisor read $(norm_ms 2.500000 1200), wants 1.250000"; return 1; }
  [ "$(norm_ms VOID 2700)" = "-" ] && [ "$(norm_ms 2.5 "")" = "-" ] \
    || { echo "SELFTEST-FAIL: norm_ms printed a figure for a VOID row or a row with no clock"; return 1; }
  echo "SELFTEST ok   norm_ms: ms at the divisor is itself, half the divisor halves it, VOID/no-clock is a dash"
  local ok='RESULT mode=burst label=v5-vx variant=5 rpb=16 lanes_per_row=32 vx=1 threads=512 jobs=48 launches=48 preburn_s=1.51 preburn_plans=340 preburn_launches=16320 ms_per_plan=3.272000 ms_min=3.272000 ms_max=3.395000 reps=3 warm=2 ratio=1.038 pt=ok us_per_launch=68.167 gbs=149.30 probe_pct=63.00 x_packed=0 tile=1 neg=none'
  local void='RESULT mode=burst label=v0-baseline variant=0 rpb=16 lanes_per_row=32 vx=0 threads=512 jobs=48 launches=48 ms_per_plan=VOID ms_min=3.544000 ms_max=17.497000 reps=3 warm=2 ratio=4.938 pt=void us_per_launch=VOID gbs=VOID probe_pct=VOID x_packed=0 tile=1 neg=none'
  local prov='RESULT mode=burst label=v1-unroll variant=1 rpb=8 lanes_per_row=32 vx=0 threads=256 jobs=48 launches=48 ms_per_plan=3.000000 ms_min=3.000000 ms_max=3.400000 reps=3 warm=2 ratio=1.133 pt=provisional us_per_launch=62.500 gbs=155.00 probe_pct=65.40 x_packed=0 tile=1 neg=none'
  local pair
  for pair in label:v5-vx rpb:16 threads:512 launches:48 ms_per_plan:3.272000 ms_min:3.272000 ms_max:3.395000 reps:3 ratio:1.038 pt:ok warm:2 gbs:149.30 probe_pct:63.00; do
    [ "$(g "${pair%%:*}" "$ok")" = "${pair#*:}" ] \
      || { echo "SELFTEST-FAIL: g ${pair%%:*} read '$(g "${pair%%:*}" "$ok")', wants ${pair#*:}"; return 1; }
  done
  echo "SELFTEST ok   every field the table prints extracts cleanly from the canned line"
  [ "$(verdict "$ok")" = ok ] || { echo "SELFTEST-FAIL: a clean line read $(verdict "$ok")"; return 1; }
  [ "$(verdict "$prov")" = provisional ] || { echo "SELFTEST-FAIL: max/min 1.133 read $(verdict "$prov")"; return 1; }
  [ "$(verdict "$void")" = void ] || { echo "SELFTEST-FAIL: max/min 4.938 read $(verdict "$void")"; return 1; }
  echo "SELFTEST ok   verdict thresholds: 1.038 -> ok, 1.133 -> provisional, 4.938 -> void"
  # The clock probe, since a parser that matches nothing would make every row clk=? and the LOWCLK rule
  # decoration. This is the same failure family as the empty-parse gates in AGENTS.md, and the one case
  # here that a real window cannot catch: by the time a window prints, the parse already agreed with it.
  local clk_line='GPU[0]		: sclk clock level: 1: (2642Mhz)'
  [ "$(printf '%s\n' "$clk_line" | parse_sclk)" = 2642 ] \
    || { echo "SELFTEST-FAIL: parse_sclk read '$(printf '%s\n' "$clk_line" | parse_sclk)', wants 2642"; return 1; }
  [ -z "$(printf 'no clocks here\n' | parse_sclk)" ] \
    || { echo "SELFTEST-FAIL: parse_sclk matched a line that has no sclk in it"; return 1; }
  [ -z "$(clk_verdict 2711)" ] || { echo "SELFTEST-FAIL: 2711 MHz read $(clk_verdict 2711), wants a plain row"; return 1; }
  [ "$(clk_verdict 820)" = LOWCLK ] || { echo "SELFTEST-FAIL: 820 MHz read $(clk_verdict 820), wants LOWCLK"; return 1; }
  [ "$(clk_verdict "")" = CLK-UNKNOWN ] || { echo "SELFTEST-FAIL: no reading read $(clk_verdict ""), wants CLK-UNKNOWN"; return 1; }
  echo "SELFTEST ok   clock: 2642 parses, junk does not, 2711 -> plain, 820 -> LOWCLK, missing -> CLK-UNKNOWN"
  # The pooled summary, driven by a canned record file. This worker may not take a timing, so the pooling
  # rules — min across passes, a control judged pooled, a point with no usable row, the winner being the
  # best packed SEARCH arm and not the fastest row — get their witness here or nowhere.
  local rec="$OUT/selftest_rows.tsv" sum="$OUT/selftest_summary.tsv" n fail rv
  mkdir -p "$OUT"
  # pass|pos|tag|variant|plan|rpb|threads|ms|clk|verdict|usable|want|tol|norm
  # The norm column is ms*clk/CLK_NORM exactly as time_point writes it, so a LOWCLK row carries a norm too
  # — and must not be allowed to use it. B-y's second row is the trap: 2.3 ms at 820 MHz normalises to
  # 0.786, which would look like the fastest arm in the table if an unusable row could enter the pool.
  printf '%s\n' \
    '1|1|A1|v0-baseline|packed|8|0|3.550000|2700|ok|1|2.9895|0.03|3.993750' \
    '2|1|A1|v0-baseline|packed|8|0|3.400000|2700|ok|1|2.9895|0.03|3.825000' \
    '1|2|B-x|v5-vx|packed|16|1024|2.600000|2700|ok|1|||2.925000' \
    '2|2|B-x|v5-vx|packed|16|1024|2.400000|2700|ok|1|||2.700000' \
    '1|3|B-y|v1-unroll|packed|8|512|2.300000|2700|ok|1|||2.587500' \
    '2|3|B-y|v1-unroll|packed|8|512|2.300000|820|LOWCLK|0|||0.785833' \
    '1|4|B-low|v5-vx|packed|32|512|9.000000|820|LOWCLK|0|||3.075000' \
    '2|4|B-low|v5-vx|packed|32|512|9.100000|820|LOWCLK|0|||3.109167' \
    '1|5|A3|v0-baseline|packed|32|0|2.785000|2700|ok|1|2.785|0.03|3.133125' \
    > "$rec"
  summarise "$rec" "$sum" > "$OUT/selftest_summary.out" 2>&1
  # A1: pooled min is 3.400, not the 3.550 of pass 1, and it is 13.7 % off 2.9895 -> off-published.
  grep -q '^A1|v0-baseline|packed|8|0|3.550000|3.400000|3.400000' "$sum" \
    || { echo "SELFTEST-FAIL: pooled min for A1 is not the min of the two passes:"; cat "$sum"; return 1; }
  grep -q '^A1 .*off-published' "$OUT/selftest_summary.out" \
    || { echo "SELFTEST-FAIL: a pooled control 13.7 % off published was not flagged"; return 1; }
  # B-y: its only usable row is pass 1; a LOWCLK row must not enter the pool, so the min stays 2.300 and
  # the empty want/tol fields must not shift the columns (the tab-separator bug).
  grep -q '^B-y|v1-unroll|packed|8|512|2.300000|-|2.300000|0.000|||1|2.587500' "$sum" \
    || { echo "SELFTEST-FAIL: B-y did not pool as one usable row, or its normmin came from the LOWCLK row:"; cat "$sum"; return 1; }
  # The normalised pool: A1's min is the normalised value of its faster row, and B-low — whose every row
  # was LOWCLK — has no normalised figure at all.
  grep -q '^A1|v0-baseline|packed|8|0|3.550000|3.400000|3.400000|0.958|2.9895|0.03|2|3.825000' "$sum" \
    || { echo "SELFTEST-FAIL: A1's normalised minimum is wrong:"; cat "$sum"; return 1; }
  grep -q '^B-low|v5-vx|packed|32|512|-|-|NONE|0.000|||0|-$' "$sum" \
    || { echo "SELFTEST-FAIL: a point with no usable row still published a normalised minimum:"; cat "$sum"; return 1; }
  # B-low: no usable row at all -> NO-USABLE-ROW, and it must count as a window failure.
  grep -q '^B-low .*NO-USABLE-ROW' "$OUT/selftest_summary.out" \
    || { echo "SELFTEST-FAIL: a point with zero usable rows did not report NO-USABLE-ROW"; return 1; }
  # The winner must be B-y at 2.300 (the best packed search arm), not A3 at 2.785 (faster, but a control)
  # and not B-x. This is the bug that broke phase C: the pair must be priced on a variant that owns the
  # threads axis, and controls do not own it.
  [ "$SUM_WIN_VAR" = v1-unroll ] && [ "$SUM_WIN_RPB" = 8 ] && [ "$SUM_WIN_THR" = 512 ] \
    || { echo "SELFTEST-FAIL: winner is $SUM_WIN_VAR rpb $SUM_WIN_RPB threads $SUM_WIN_THR, wants v1-unroll rpb 8 threads 512"; return 1; }
  [ "$SUM_FAIL" = 2 ] || { echo "SELFTEST-FAIL: summary reported $SUM_FAIL failures, wants 2 (A1 off published, B-low unusable)"; cat "$OUT/selftest_summary.out"; return 1; }
  # A3 has one pass only; it must still be adjudicable and must not claim a pass2 figure.
  grep -q '^A3|v0-baseline|packed|32|0|2.785000|-|2.785000' "$sum" \
    || { echo "SELFTEST-FAIL: a single-pass control was not pooled correctly"; return 1; }
  grep -q 'no position effect within arms' "$OUT/selftest_summary.out" \
    || { echo "SELFTEST-FAIL: the flat canned file did not read as position-free; the check cries wolf"; cat "$OUT/selftest_summary.out"; return 1; }
  # …and that it CAN fire. Same rig, canned rows where within each arm the later position is the faster
  # one — a clock ramp wearing the costume of a ranking. If this does not fire, the check is decoration.
  local ramp="$OUT/selftest_ramp.tsv"
  printf '%s\n' \
    '1|1|B-r1|v1-unroll|packed|8|512|3.000000|2700|ok|1|||3.375000'   '2|5|B-r1|v1-unroll|packed|8|512|2.600000|2700|ok|1|||2.925000' \
    '1|2|B-r2|v1-unroll|packed|16|512|3.000000|2700|ok|1|||3.375000'  '2|4|B-r2|v1-unroll|packed|16|512|2.600000|2700|ok|1|||2.925000' \
    '1|4|B-r3|v1-unroll|packed|8|1024|2.600000|2700|ok|1|||2.925000'  '2|2|B-r3|v1-unroll|packed|8|1024|3.000000|2700|ok|1|||3.375000' \
    '1|5|B-r4|v1-unroll|packed|16|1024|2.600000|2700|ok|1|||2.925000' '2|1|B-r4|v1-unroll|packed|16|1024|3.000000|2700|ok|1|||3.375000' \
    > "$ramp"
  summarise "$ramp" "$OUT/selftest_ramp_sum.tsv" > "$OUT/selftest_ramp.out" 2>&1
  grep -q 'SURVIVED interleaving' "$OUT/selftest_ramp.out" \
    || { echo "SELFTEST-FAIL: a monotone ramp (later position faster, within every arm) did not fire the position verdict"; cat "$OUT/selftest_ramp.out"; return 1; }
  # And it must be quiet on a position-free record where the same arms swap orders with no trend at all.
  printf '%s\n' \
    '1|1|B-r1|v1-unroll|packed|8|512|2.800000|2700|ok|1|||3.150000'   '2|5|B-r1|v1-unroll|packed|8|512|2.800000|2700|ok|1|||3.150000' \
    '1|2|B-r2|v1-unroll|packed|16|512|2.800000|2700|ok|1|||3.150000'  '2|4|B-r2|v1-unroll|packed|16|512|2.800000|2700|ok|1|||3.150000' \
    '1|4|B-r3|v1-unroll|packed|8|1024|2.800000|2700|ok|1|||3.150000'  '2|2|B-r3|v1-unroll|packed|8|1024|2.800000|2700|ok|1|||3.150000' \
    '1|5|B-r4|v1-unroll|packed|16|1024|2.800000|2700|ok|1|||3.150000' '2|1|B-r4|v1-unroll|packed|16|1024|2.800000|2700|ok|1|||3.150000' \
    > "$ramp"
  summarise "$ramp" "$OUT/selftest_ramp_sum2.tsv" > "$OUT/selftest_ramp2.out" 2>&1
  grep -q 'no position effect within arms' "$OUT/selftest_ramp2.out" \
    || { echo "SELFTEST-FAIL: a flat record with swapped orders read as a position effect; the check cannot tell the two apart"; cat "$OUT/selftest_ramp2.out"; return 1; }
  echo "SELFTEST ok   position verdict: fires on a within-arm monotone ramp, quiet when orders swap with no trend"
  # And the position verdict must be able to FIRE, not just to reassure: same four arms, ms made a linear
  # function of position, and the two passes given opposite orders. If the paired statistic cannot see that,
  # its "no position effect" line in a real window means nothing.
  local ramp="$OUT/selftest_ramp_rows.tsv"
  printf '%s\n' \
    '1|1|B-p1|v1-unroll|packed|8|512|2.590000|2700|ok|1|||2.913750' \
    '1|2|B-p2|v1-unroll|packed|16|512|2.580000|2700|ok|1|||2.902500' \
    '1|3|B-p3|v1-unroll|packed|8|1024|2.570000|2700|ok|1|||2.891250' \
    '1|4|B-p4|v1-unroll|packed|16|1024|2.560000|2700|ok|1|||2.880000' \
    '2|4|B-p1|v1-unroll|packed|8|512|2.560000|2700|ok|1|||2.880000' \
    '2|3|B-p2|v1-unroll|packed|16|512|2.570000|2700|ok|1|||2.891250' \
    '2|2|B-p3|v1-unroll|packed|8|1024|2.580000|2700|ok|1|||2.902500' \
    '2|1|B-p4|v1-unroll|packed|16|1024|2.590000|2700|ok|1|||2.913750' \
    > "$ramp"
  summarise "$ramp" "$OUT/selftest_ramp_summary.tsv" > "$OUT/selftest_ramp.out" 2>&1
  grep -q 'SURVIVED interleaving' "$OUT/selftest_ramp.out" \
    || { echo "SELFTEST-FAIL: the paired position check did not fire on a synthetic monotone ramp; it cannot fail, so its reassurance means nothing"; cat "$OUT/selftest_ramp.out"; return 1; }
  echo "SELFTEST ok   the paired position check fires on a synthetic ramp and stays quiet on a flat one"
  n=0; fail=0; rv=""
  # Drift guard on the live tool (a sysfs query, not a kernel run, so it needs no lock and takes no timing):
  # if rocm-smi is installed, its output must still match the parser, or every real row would be clk=?
  if command -v "$CLK_TOOL" >/dev/null 2>&1; then
    local live; live="$("$CLK_TOOL" --showclocks 2>/dev/null | parse_sclk)"
    [ -n "$live" ] || { echo "SELFTEST-FAIL: $CLK_TOOL is installed but its --showclocks output does not parse; the LOWCLK rule cannot work"; return 1; }
    echo "SELFTEST ok   live $CLK_TOOL --showclocks parses (reads ${live} MHz right now, floor $CLK_MIN)"
  else
    echo "SELFTEST note   $CLK_TOOL not on PATH: rows would read clk=? and every one would be CLK-UNKNOWN"
  fi
  local -a cases=(
    "ok|v5-vx|48|16|512|$ok|pass"
    "stale-binary:label|v0-baseline|48|16|512|$ok|fail"
    "stale-plan:launches|v5-vx|114|16|512|$ok|fail"
    "wrong-rpb|v5-vx|48|8|512|$ok|fail"
    "welded-threads|v5-vx|48|16|1024|$ok|fail"
    "void-line-parses|v0-baseline|48|16|512|$void|pass")
  local n=0 bad=0 name want_v want_l want_r want_t line want got
  for c in "${cases[@]}"; do
    IFS='|' read -r name want_v want_l want_r want_t line want <<<"$c"
    got="$(canary_check "$want_v" "$want_l" "$want_r" "$want_t" "$line")"
    n=$((n + 1))
    if { [ "$want" = pass ] && [ -n "$got" ]; } || { [ "$want" = fail ] && [ -z "$got" ]; }; then
      bad=$((bad + 1)); echo "SELFTEST-FAIL $name: expected $want, got '${got}'"
    else
      echo "SELFTEST ok   $name -> ${got:-clean}"
    fi
  done
  [ "$bad" -eq 0 ] || { echo "SELFTEST: $bad of $n canary cases wrong"; return 1; }
  echo "SELFTEST PASS: $n canary cases + 3 verdict cases + 5 clock cases + the pooled summary (min across passes, LOWCLK excluded, winner is a search arm) + the position verdict firing and staying quiet; wrong label, launches, rpb and threads all fire"
  return 0
}

# Dispatched here, not beside the function: `selftest` calls `g`, and bash only knows a function once
# its definition has been executed.
[ "${1:-}" = selftest ] && { selftest; exit $?; }

[ -x "$BIN" ] || die "$BIN missing: run bash $HERE/build.sh"
mkdir -p "$OUT"
# Explicit if, not an `A && B || C && D && cmd` chain: && and || are left-associative, so a chain of two
# source tests silently drops the rebuild when only the first file is newer than the binary.
if [ -f "$HERE/proj_gemv.hip.cpp" ] && [ "$HERE/proj_gemv.hip.cpp" -nt "$BIN" ]; then
  log "binary older than proj_gemv.hip.cpp; rebuilding"; bash "$HERE/build.sh" || die "build failed"
elif [ -f "$HERE/proj_gemv_kernels.h" ] && [ "$HERE/proj_gemv_kernels.h" -nt "$BIN" ]; then
  log "binary older than proj_gemv_kernels.h; rebuilding"; bash "$HERE/build.sh" || die "build failed"
fi

# The GRID FILE is the contract between the gate and the timer: one line per distinct
# (variant, rows/block, threads/block) point, in the launcher's own `variant[:rpb][@threads]` spelling.
# Built before the lock is taken; both phases read THIS file instead of each deriving its own list.
# Plan shape is deliberately not part of a point: packed and unpacked produce the same per-projection
# output (the gate judges each projection's block, which is what both plan shapes emit), so the plan
# pair is one gated point timed against two plan files, not a second geometry.
GRID="$OUT/gate_grid.txt"
: > "$GRID"
for arm in "${ARMS[@]}"; do
  IFS='|' read -r tag var shape rlist thr _ _ <<<"$arm"
  IFS=',' read -r -a rpbs <<<"$rlist"
  for r in "${rpbs[@]}"; do
    line="$var:$r"; [ "$thr" != 0 ] && line="$line@$thr"
    grep -qx "$line" "$GRID" || echo "$line" >> "$GRID"
  done
done
NGRID=$(grep -c . "$GRID")
log "phase 0: grid written — $NGRID distinct points (no GPU)"
"$PY" "$HERE/run.py" --layers all --steps decode0 --emit-plans "$OUT" --plan-shape both \
  || die "run.py --emit-plans failed"
for s in packed unpacked mixed; do
  [ -s "$OUT/plan_token_$s.txt" ] || die "no plan file $OUT/plan_token_$s.txt"
done

if [ "${SKIP_GATE:-0}" = "1" ]; then
  log "phase 1: SKIPPED by SKIP_GATE=1 — timings below are un-admitted"
  NGATE=0
else
  log "phase 1: admission — the SAME grid, all 24 layers, --steps $STEPS (device, correctness only)"
  # No pipe here on purpose: with `cmd | tee`, `$?` is tee's status and a red gate would read as green.
  if ! "$PY" "$HERE/run.py" --layers all --steps "$STEPS" --negative --variant all \
        --grid-file "$GRID" >"$OUT/admission.log" 2>&1; then
    cat "$OUT/admission.log"
    echo "NO-TIMING: admission is red (run.py exited non-zero)"
    echo "  see $OUT/admission.log; no millisecond figure is printed for a red case list"
    exit 1
  fi
  cat "$OUT/admission.log"
  grep -q '^PASS' "$OUT/admission.log" || { echo "NO-TIMING: admission is red"; exit 1; }
  NGATE=$(grep -oE 'GATED [0-9]+ point' "$OUT/admission.log" | grep -oE '[0-9]+' | head -1)
  [ -n "$NGATE" ] || { echo "NO-TIMING: admission printed no point count"; exit 1; }
  [ "$NGATE" = "$NGRID" ] || { echo "NO-TIMING: grid lists $NGRID points, the gate judged $NGATE"; exit 1; }
  log "admission green: gated $NGATE of $NGRID grid points"
fi

log "phase 2: timing under the lock"
# NO_TIMING=1 stops after admission. It exists so the grid, the plan files and the canaries can be
# checked in a window that is not being spent on measurements — and so the person who wrote the timer
# (who may not take a timing) can still prove phases 0 and 1 run.
if [ "${NO_TIMING:-0}" = "1" ]; then
  log "NO-TIMING: NO_TIMING=1 — phases 0-1 only, grid and admission checked, nothing measured"
  log "grid ($NGRID points): $(tr '\n' ' ' < "$GRID")"
  exit 0
fi
"$LOCK" acquire "proj_gemv D1 sweep (sweep_decode.sh)" || die "could not acquire the GPU lock"
trap '"$LOCK" release "proj_gemv D1 sweep (sweep_decode.sh)"' EXIT
log "gpu_lock util at acquire: $("$LOCK" util 2>/dev/null)"

# ---- the clock burn: no arm is timed until the part is off its idle floor ----------------------
# progress/2026-10-21/performance-status-2026-10-21.md §36 measured this box at 819/827/838 MHz idle and
# 2432/2437/2423 MHz immediately after a rep. The second D1 window then read a monotone early-to-late
# drift over 13 s of GPU work — that ramp wearing the costume of a ranking. One sustained untimed burn
# on the real plan and the real weights runs before arm 1, timed out by WALL seconds (so it costs the
# same on a faster machine) rather than by a launch count.
BURN="${BURN:-5}"
CLK_PRE_SET="$(clock_mhz)"; clk_note "$CLK_PRE_SET"
log "clock burn: ${BURN}s of untimed launches (${CLK_PRE_SET:-?} MHz now)"
"$BIN" --mode burn --burn "$BURN" --plan "$OUT/plan_token_packed.txt" \
       --variant v1-unroll --rows 8 >"$OUT/burn.log" 2>&1 \
  || log "burn FAILED (rc=$?) — the clock never got load; every row below is suspect"
log "burn: $(grep '^RESULT mode=burn' "$OUT/burn.log" || echo 'no burn RESULT line')"
CLK_POST_BURN="$(clock_mhz)"; clk_note "$CLK_POST_BURN"
log "clock after burn: ${CLK_POST_BURN:-?} MHz (a row must be at or above $CLK_MIN MHz to count)"

# ---- the order: one fixed pseudo-random permutation, run twice ---------------------------------
# Interleaving makes arm identity independent of position, so a monotone clock ramp can no longer rank
# the arms; running the whole set twice and reporting the MIN across passes means a ranking that survives
# both has survived the ramp. Whether it did is printed as a verdict below, not assumed.
SEED="${SEED:-d1-interleave-2}"
PASSES="${PASSES:-2}"
POINTS=()
for arm in "${ARMS[@]}"; do
  IFS='|' read -r tag var shape rlist thr want tol <<<"$arm"
  IFS=',' read -r -a rpbs <<<"$rlist"
  for r in "${rpbs[@]}"; do POINTS+=("$tag|$var|$shape|$r|$thr|$want|$tol"); done
done
# One order per pass, not one order reused. With the same permutation every pass, an arm's position is
# identical in both and position cannot be separated from arm identity — a first-half/second-half
# comparison then just measures which geometries happened to land where (the stub dry run showed a
# spurious 4.6 % "ramp" from geometry mix alone). Two independent permutations give each arm a different
# position in each pass, which is what makes the paired position statistic in summarise mean something.
ORDER_N=0
for p in $(seq 1 "$PASSES"); do
  of="$OUT/arm_order_p$p.txt"; : > "$of"
  for pt in "${POINTS[@]}"; do
    printf '%s\t%s\n' "$(printf '%s' "$SEED|p$p|$pt" | cksum | cut -d' ' -f1)" "$pt" >> "$of"
  done
  sort -n -k1,1 -o "$of" "$of"
  ORDER+=("$of")
  ORDER_N=$((ORDER_N + 1))
  log "order pass $p: ${#POINTS[@]} points, fixed pseudo-random (seed $SEED|p$p), in $of"
done
[ "$PASSES" -eq 1 ] || log "positions differ between passes by construction: that is what makes the position verdict readable"

row_hdr
FAILED=0; VOIDS=0; LOWCLK_N=0; NTIMED=0
GEOM="$OUT/timed_geometry.txt"; : > "$GEOM"        # distinct points actually timed
REC="$OUT/pass_rows.tsv"; : > "$REC"              # one line per timed row; the summary reads this

# time_point PASS POS TAG VARIANT PLAN RPB THREADS [WANT [TOL]] — one launcher invocation, one upload.
# Prints one row and appends a record for the summary. A VOID point prints no millisecond figure. A
# LOWCLK point prints one but is marked and kept out of the summary: a number taken below the clock
# floor measures the ramp, not the shape.
time_point() {
  local pass="${1:-1}" pos="${2:-0}" tag="$3" var="$4" shape="$5" rpb="$6" thr="${7:-0}" want="${8:-}" tol="${9:-}"
  local plan="$OUT/plan_token_$shape.txt" want_loads bad line vd ms gbs pct note off clk clk_pre clkv usable rc nrm
  want_loads="$(expected_launches "$shape")"
  local out="$OUT/burst_p${pass}_${tag}_r${rpb}_t${thr}.log"
  # Which clock adjudicates: the one read AFTER the launcher returns. With `--preburn` the boost happens
  # inside that process, immediately before the reps, so the read taken before the call is the decayed clock
  # the burn lifted itself away from (the clock-controlled window: 1218 MHz before, 2718 after). The after-
  # read is the one the timed reps ran at; the before-read is kept in the note because its level is itself
  # the evidence of how far the queue decays between arms.
  clk_pre="$(clock_mhz)"; clk_note "$clk_pre"
  "$BIN" --mode burst --plan "$plan" --variant "$var" --rows "$rpb" \
         --threads "$thr" --iters "$ITERS" --reps "$REPS" --warm "$WARM" --preburn "$ARM_BURN" >"$out" 2>&1
  rc=$?
  clk="$(clock_mhz)"; clk_note "$clk"; clkv="$(clk_verdict "$clk")"
  if [ "$rc" -ne 0 ]; then
    row "$pass" "$tag" "$var" "$rpb" "$thr" "-" "$want_loads" "$shape" "RUN-FAILED" \
        "-" "-" "-" "${clk:-?}" "RUN-FAILED" "rc=$rc, see $out"
    FAILED=$((FAILED + 1)); return 1
  fi
  line="$(grep '^RESULT mode=burst' "$out" | head -1)"
  [ -n "$line" ] || { log "$tag r$rpb: no burst RESULT line in $out — nothing was measured"
                      FAILED=$((FAILED + 1)); return 1; }
  NTIMED=$((NTIMED + 1))
  echo "$var:$rpb$( [ "$thr" != 0 ] && echo "@$thr" )" >> "$GEOM"
  local rpb_g thr_g
  rpb_g="$(g rpb "$line")"; thr_g="$(g threads "$line")"
  bad="$(canary_check "$var" "$want_loads" "$rpb" "$(want_threads "$thr" "$rpb" "$var")" "$line")"
  if [ -n "$bad" ]; then
    row "$pass" "$tag" "$var" "$rpb_g" "$thr_g" "$(g lanes_per_row "$line")" "$want_loads" \
        "$shape" "CANARY-FAIL" "-" "-" "-" "${clk:-?}" "CANARY-FAIL" "$bad"
    FAILED=$((FAILED + 1)); return 1
  fi
  # The per-arm burn is not optional if it was asked for: a launcher that quietly ignored --preburn would
  # reproduce exactly the decay this exists to prevent, and the row would still look well-formed.
  if [ "$ARM_BURN" != 0 ]; then
    local pb ppb
    pb="$(g preburn_s "$line")"; ppb="$(g preburn_plans "$line")"
    if [ -z "$pb" ] || [ -z "$ppb" ] || [ "$ppb" = 0 ]; then
      row "$pass" "$tag" "$var" "$rpb_g" "$thr_g" "$(g lanes_per_row "$line")" "$want_loads" \
          "$shape" "CANARY-FAIL" "-" "-" "-" "${clk:-?}" "CANARY-FAIL" \
          "no per-arm burn: the line carries no preburn_s/preburn_plans, so --preburn $ARM_BURN was ignored"
      FAILED=$((FAILED + 1)); return 1
    fi
    if awk -v a="$pb" -v w="$ARM_BURN" 'BEGIN{exit !(a + 0 < w * 0.8)}'; then
      row "$pass" "$tag" "$var" "$rpb_g" "$thr_g" "$(g lanes_per_row "$line")" "$want_loads" \
          "$shape" "CANARY-FAIL" "-" "-" "-" "${clk:-?}" "CANARY-FAIL" \
          "per-arm burn ran only ${pb}s of ${ARM_BURN}s requested: the burn did not complete a plan"
      FAILED=$((FAILED + 1)); return 1
    fi
  fi
  vd="$(verdict "$line")"; usable=0; nrm="$(norm_ms "$(g ms_per_plan "$line")" "$clk")"
  if [ "$vd" = void ]; then
    ms=VOID; gbs=VOID; pct=VOID; nrm=-; VOIDS=$((VOIDS + 1))
    note="max/min $(g ratio "$line") after $WARM warm passes: not measured, no millisecond printed"
  else
    ms="$(g ms_per_plan "$line")"; gbs="$(g gbs "$line")"; pct="$(g probe_pct "$line")"
    nrm="$(norm_ms "$ms" "$clk")"
    note="min..max $(g ms_min "$line")..$(g ms_max "$line") over $(g reps "$line") reps; burn $(g preburn_s "$line")s/$(g preburn_plans "$line") plans"
    if [ -n "$want" ]; then
      off=$(awk -v m="$ms" -v w="$want" -v t="$tol" \
                'BEGIN{o=(m-w)/w; if(o<0)o=-o; printf "%.1f%%", 100*o}')
      # The control check is adjudicated on the POOLED min across passes, not here: a first-pass row taken
      # before the box settled would abort a window that the second pass settles.
      note="$note | published $want, off $off (tol $(awk -v t="$tol" 'BEGIN{printf "%.0f%%", 100*t}'), judged pooled)"
    fi
    if [ -n "$clkv" ]; then
      # A number taken below the clock floor measures the ramp, not the shape: it prints, clearly marked,
      # and is kept out of every pool, minimum and verdict below.
      vd="$clkv"; note="$note | clock ${clk:-?} MHz after the burst, ${clk_pre:-?} before it; floor $CLK_MIN MHz"
      LOWCLK_N=$((LOWCLK_N + 1))
    else
      usable=1
    fi
  fi
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
         "$pass" "$pos" "$tag" "$var" "$shape" "$rpb" "$thr" "$ms" "${clk:-}" "$vd" "$usable" \
         "$want" "$tol" "$nrm" >> "$REC"
  row "$pass" "$tag" "$var" "$rpb_g" "$thr_g" "$(g lanes_per_row "$line")" "$want_loads" \
      "$shape" "$ms" "$nrm" "$gbs" "$pct" "${clk:-?}" "$vd" "$note"
  return 0
}

for pass in $(seq 1 "$PASSES"); do
  pos=0
  order_file="$OUT/arm_order_p$pass.txt"   # one order per pass; see the note where they are written
  [ -s "$order_file" ] || die "no order file $order_file for pass $pass"
  while IFS=$'\t' read -r _h p; do
    IFS='|' read -r tag var shape rpb thr want tol <<<"$p"
    pos=$((pos + 1))
    time_point "$pass" "$pos" "$tag" "$var" "$shape" "$rpb" "$thr" "$want" "$tol"
  done < "$order_file"
done

log "summary: pooled across $PASSES passes (min over usable rows: canary ok, self-consistent, clock >= $CLK_MIN MHz)"
summarise "$REC" "$OUT/summary.tsv"
FAILED=$((FAILED + SUM_FAIL))
WIN_VAR="$SUM_WIN_VAR"; WIN_RPB="$SUM_WIN_RPB"; WIN_THR="$SUM_WIN_THR"; WIN_MS="$SUM_WIN_MS"; WIN_TAG="$SUM_WIN_TAG"

CLK_END="$(clock_mhz)"; clk_note "$CLK_END"
log "clock at end of the timed set: ${CLK_END:-?} MHz (was ${CLK_PRE_SET:-?} before the burn, ${CLK_POST_BURN:-?} after it)"
log "clock reads: $CLKS_OK parsed, $CLKS_BAD unreadable (floor $CLK_MIN MHz, tool ${CLK_TOOL:-none})"
if [ "$CLKS_OK" -eq 0 ]; then
  log "FATAL: no clock reading parsed, so LOWCLK could not be enforced and no row here carries a clock"
  FAILED=$((FAILED + 1))
fi
log "rows: $NTIMED timed over $PASSES passes, $VOIDS VOID, $LOWCLK_N LOWCLK"

# ---- phase 3: the plan pair, at the shape the SEARCH won, on the variant that owns the axis -------
# The second window lost both C arms to rc=1 because it asked v0-baseline for --threads 1024, and v0
# refuses a thread count the engine cannot emit. The pair is driven by $WIN_VAR instead — v1-unroll or
# v5-vx, both bit-identical to v0 — so it still measures the launch plan and nothing else. The figure
# banked from the first window is packed 3.272 vs unpacked 3.902 = -16.2 %.
if [ -n "$WIN_VAR" ]; then
  log "phase 3: plan pair at the winning packed shape — $WIN_VAR rpb $WIN_RPB threads $WIN_THR (best $WIN_MS ms, from $WIN_TAG)"
  time_point 1 91 "D-packed"   "$WIN_VAR" packed   "$WIN_RPB" "$WIN_THR"
  time_point 1 92 "D-unpacked" "$WIN_VAR" unpacked "$WIN_RPB" "$WIN_THR"
  pk="$(awk -F'|' -v t="D-packed"   '$3==t && $11=="1" {print $8; exit}' "$REC")"
  up="$(awk -F'|' -v t="D-unpacked" '$3==t && $11=="1" {print $8; exit}' "$REC")"
  if [ -n "$pk" ] && [ -n "$up" ]; then
    log "PLAN PAIR: packed $pk vs unpacked $up = $(awk -v a="$pk" -v b="$up" 'BEGIN{printf "%+.1f%%", 100*(a-b)/b}') from the 48-launch plan alone, same kernels, same window"
  else
    log "PLAN PAIR: no usable pair (packed=${pk:-none} unpacked=${up:-none}) — not adjudicable"
    FAILED=$((FAILED + 1))
  fi
else
  log "phase 3: SKIPPED — no packed search arm survived as a measurement, so there is no shape to price the plan pair at"
  FAILED=$((FAILED + 1))
fi

log "gpu_lock util at release: $("$LOCK" util 2>/dev/null)"
"$LOCK" release "proj_gemv D1 sweep (sweep_decode.sh)"
trap - EXIT

NTPTS=$(sort -u "$GEOM" | grep -c .)
log "gated $NGATE, timed $NTPTS distinct points, N==M: $([ "$NTPTS" = "$NGATE" ] && echo yes || echo NO)"
if [ "${SKIP_GATE:-0}" != "1" ] && [ "$NTPTS" -ne "$NGATE" ]; then
  log "the timed set is not the gated set — these rows are not adjudicable:"
  diff <(sort -u "$GRID") <(sort -u "$GEOM") | sed 's/^/  /'
  die "gated $NGATE, timed $NTPTS"
fi
[ "$FAILED" -eq 0 ] || die "$FAILED point(s) failed the canary, missed a published control, missed the clock floor, or did not run"
log "sweep complete: $NGRID grid points x $PASSES passes, 0 canary failures, $VOIDS VOID, $LOWCLK_N LOWCLK"
