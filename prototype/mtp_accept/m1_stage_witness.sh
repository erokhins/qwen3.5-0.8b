#!/usr/bin/env bash
# M1 witness — the `mtp` stage family. Correctness fields only.
#
# WHAT IS WITNESSED, and by which part:
#   A  flag OFF, new binary            — the bill §07 always billed; the baseline of the pair.
#   B  flag OFF, PRE-MTP binary        — inertness: same counts, and `sampled.txt` / every logits file /
#                                        every STATEFP row byte-identical between the two binaries.
#   C  flag ON, `--mtp-sched every`    — the `mtp` row exists, the TOTAL still says `plan_match=1`, and
#                                        the target's own bytes did not move.
#   D  canaries                        — the pre-MTP binary cannot name `--mtp-draft` (so A and B are two
#                                        builds, not one build compared with itself: `AGENTS.md`), and the
#                                        combinations the clock cannot bill are refused, not misbilled.
#
# NO TIMING IS TAKEN OR READ HERE. `--time-stages` prints ms columns on every line it prints; they are
# grepped out below and are not quoted in any report, because a worker may not report a timing figure
# (`AGENTS.md`, "GPU execution is serialized and belongs to the main agent"). What is compared is
# `launches_per_step`, `plan_per_step`, `plan_match`, `uniform` — counts, which contention cannot move.
#
# USAGE   tools/gpu_lock.sh acquire <tag>   (this script does NOT self-lock; `decode_run` does not either)
#         PART=AB prototype/mtp_accept/m1_stage_witness.sh
#         PART=CD prototype/mtp_accept/m1_stage_witness.sh
set -euo pipefail
export LC_ALL=C
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
BIN=engine/build/decode_run
OLD=engine/build/decode_run.premtp
MODEL=models/Qwen3.5-0.8B
IDS=${IDS:-/tmp/dec8_ids.csv}
OUT=${OUT:-/tmp/m1_stage}
STEPS=${STEPS:-8}
PART=${PART:-ALL}
mkdir -p "$OUT"

want() { case ",$PART,ALL," in *",ALL,"*) return 0 ;; esac; case ",$PART," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# The counts, with every ms column stripped. Two sed passes: the per-family rows and the TOTAL row.
counts() {
    grep -E '^(STAGE gran=[a-z]+ (TOTAL|family=mtp)|STAGEMTP|STAGEPARITY|MTPDRAFT)' "$1" \
        | sed -E 's/ ms_per_step=[0-9.]+ ms_total=[0-9.]+//; s/ ms=[0-9.]+ share_pct=[0-9.]+//; s/ ms_total=[0-9.]+//; s/ us_per_launch=[0-9.]+//; s/ ms=[0-9.]+//' || true
}
cmpdir() {   # cmpdir DIRA DIRB LABEL — byte-compare the dumps the two runs wrote
    local a=$1 b=$2 lbl=$3 bad=0 n=0 f
    cmp -s "$a/sampled.txt" "$b/sampled.txt" && echo "   [$lbl] SAME sampled.txt" || echo "   [$lbl] DIFF sampled.txt"
    for f in "$a"/logits*; do [ -e "$f" ] || continue; n=$((n+1)); cmp -s "$f" "$b/$(basename "$f")" || bad=$((bad+1)); done
    echo "   [$lbl] logits files compared=$n differing=$bad"
    if diff <(grep '^STATEFP' "$a.txt") <(grep '^STATEFP' "$b.txt") >/dev/null; then
        echo "   [$lbl] SAME STATEFP rows ($(grep -c '^STATEFP' "$a.txt"))"
    else
        echo "   [$lbl] DIFF STATEFP rows"
    fi
    diff <(grep '^STEP ' "$a.txt") <(grep '^STEP ' "$b.txt") >/dev/null \
        && echo "   [$lbl] SAME STEP lines" || echo "   [$lbl] DIFF STEP lines"
}

if want A; then
  echo "== A flag-off, new binary, --time-stages"
  mkdir -p "$OUT/off_new"; t=$SECONDS
  "$BIN" --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" --out "$OUT/off_new" --state-fp \
      --time-stages > "$OUT/off_new.txt" 2>&1
  counts "$OUT/off_new.txt"
  echo "   wall=$((SECONDS-t))s — no timing taken, counts only"
fi

if want B; then
  echo "== B flag-off, PRE-MTP binary, --time-stages"
  if [ -x "$OLD" ]; then
    mkdir -p "$OUT/off_old"; t=$SECONDS
    "$OLD" --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" --out "$OUT/off_old" --state-fp \
        --time-stages > "$OUT/off_old.txt" 2>&1
    counts "$OUT/off_old.txt"
    echo "   wall=$((SECONDS-t))s"
    cmpdir "$OUT/off_new" "$OUT/off_old" "new-vs-preMTP, flag off"
  else
    echo "   MISSING $OLD — inertness side NOT run"
  fi
fi

if want C; then
  echo "== C flag-ON (--mtp-sched every), --time-stages"
  mkdir -p "$OUT/on_new"; t=$SECONDS
  QWEN_MTP_DRAFT=1 "$BIN" --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" --out "$OUT/on_new" \
      --state-fp --time-stages --mtp-sched every > "$OUT/on_new.txt" 2>&1
  counts "$OUT/on_new.txt"
  echo "   wall=$((SECONDS-t))s"
  cmpdir "$OUT/on_new" "$OUT/off_new" "flag-on-vs-off"
fi

if want D; then
  echo "== D canaries and refusals"
  if "$OLD" --model "$MODEL" --ids-file "$IDS" --steps 1 --mtp-draft 1 >/dev/null 2>"$OUT/c1.txt"; then
      echo "   CANARY FAILED: the pre-MTP binary accepted --mtp-draft, so A and B are not two builds"
  else
      echo "   canary OK — pre-MTP binary refuses the flag: $(sed -n 1p "$OUT/c1.txt")"
  fi
  for pair in "--time-stages --mtp-sched loop" "--time-waves --mtp-sched every"; do
    # shellcheck disable=SC2086
    if "$BIN" --model "$MODEL" --ids-file "$IDS" --steps 1 --mtp-draft 1 $pair >/dev/null 2>"$OUT/c2.txt"; then
        echo "   REFUSAL FAILED: --mtp-draft 1 $pair ran"
    else
        echo "   refusal OK for [$pair]: $(sed -n 1p "$OUT/c2.txt")"
    fi
  done
fi
echo "== done"
