#!/usr/bin/env bash
# run_fattn_burst_window.sh — the device window that decides claim (iii) of D62's funded probe
# (progress/2026-10-18/llama-fattn-anatomy.md §4.2): does the WMMA attention form cut the engine's `attn`
# family row by >= 1.5x at 4,096 lanes?
#
# WRITTEN TO BE RUN BY THE MAIN AGENT, blocking, zero subagents, one `gpu_lock` hold. NOTHING HERE HAS
# BEEN RUN. No number in this file, in fattn_score_handover.md, or in fattn_census.output.txt is a
# measurement, and a worker may not take one (`AGENTS.md`, "GPU execution is serialized"). Shape is copied
# from `prototype/attn_wmma/run_attn_attrib_window.sh` for one reason: that script already encodes this
# box's rules (one lock hold for a paired A/B, `gran=stage family=attn` as the claim rather than a harness
# ratio, abort on a null parse, refuse an inert flag).
#
# IT DOES NOT SELF-ACQUIRE `tools/gpu_lock.sh`. Three gate drivers in this tree do (`AGENTS.md`: a caller
# that holds the lock and then runs a self-locking one deadlocks it, and the child parks silently). This one
# takes the lock itself because it IS the window; nothing it calls takes it.
#
# WHAT THE PORT DECIDED, and this file prices it, not the prototype
# `--attn-fattn 1` runs `engine/attn_fattn_kernels.h`'s `attn_fattn_wave` BESIDE `attn_slim_wave`, and the
# WMMA kernel owns only the rows §07.3 already streams (lanes > MAXSC). One-pass rows stay on the VALU wave,
# bit-for-bit, because their bits are pinned and a WMMA round is streaming by construction. Two consequences
# this script has to carry rather than hide:
#   * The family denominator is only partly covered, so the covered share is read out of the run's own log
#     and printed beside the ms. A 1.5x family result on 70 % coverage is a ~2x result on covered rows, and
#     those are different claims (`fattn_score_handover.md` §5a item 2, and the launcher's `ATTN_FATTN`
#     line exists to make this a printed number rather than an assumption).
#   * Inertness is asserted (CANARY D), not assumed: a short prompt must come back byte-identical.
#
# THE CANARIES (AGENTS.md, "a paired A/B against a stale binary reads as a clean null"). `plan_match` is
# compiled into `decode_run` and cannot see staleness, and `engine/build.sh` is not bit-reproducible here,
# so `cmp` on two binaries proves nothing. Four checks, each read in the same window that produces the
# number, each aborting before any ms is printed:
#   A  flag liveness: `--attn-fattn 1` accepted, `--attn-fattn 7` refused.
#   B  form banner, with fields, from the ENGINE's own listing — `new` prints
#        ATTN_FATTN form=fattn16 vgpr=… lds=… vtiles=16 wave_rows=… fattn_rows=n/N covered_pct=… split=…
#      and `base` prints no such line. `lds` must equal the listing's `.group_segment_fixed_size` and
#      `vgpr` its `.vgpr_count` (the runtime reads them off the loaded code object, so a binary built from
#      an older copy of the form cannot agree with a fresh listing — the D30 1.7′ failure was exactly a
#      pre-change binary timed as `new` with every self-consistency field green); `SPILL` must be absent,
#      because the §1 spill failure returning is a regression and not a tuning knob; and `fattn_rows` must
#      be > 0 on a wave whose rows all stream, which is what makes the line evidence that a row was routed
#      rather than a line a binary prints while routing nothing.
#   C  a bit-difference witness ON A COVERED ROW: at 4,096 lanes every row of the last wave streams, so the
#      last-row logits MUST differ. An f16 16x16x16 tile dot cannot reproduce a BF16 serial 256-term dot
#      bit for bit, so identical bytes here would prove one side ran the other side's kernel.
#   D  INERTNESS ON A ONE-PASS PROMPT: at 64 lanes no row is covered and the two sides must be
#      BYTE-IDENTICAL. Coverage that leaked into a pinned one-pass row shows here, not in the ratio.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=models/Qwen3.5-0.8B
IDSDIR=progress/bench/evidence/prefill-occupancy-2026-10-16
BIN=engine/build/decode_run
# The engine's own code-object listing, for CANARY B. Reproduce it with the compile-only recipe in
# `engine/attn_fattn_kernels.h`'s census note (hipcc --save-temps over `engine/kernels_attn.cpp`); it
# touches no device. Overridable so a re-census in a different tmp dir is not a script edit.
LIST=${LIST:-tmp/fattn_census/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s}
KERNEL_NAME=attn_fattn_wave
OUT=tmp/gcc/attn_fattn/w
TAG=attn-fattn-62
REPS=${REPS:-3}
WIDTHS=${WIDTHS:-"1024 4096"}          # 4,096 is the width §4.2's bar names; 1,024 is the pair
INERT_IDS=${INERT_IDS:-64}             # every row one-pass: CANARY D's must-be-identical side
mkdir -p "$OUT"
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
ids_of() {
    if [ -f "$IDSDIR/ids$1.txt" ]; then cat "$IDSDIR/ids$1.txt"; else seq 1 "$1" | tr '\n' ' '; fi
}
gmsof() { grep -h "gran=stage family=attn " "$1" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; }
lnsof() { grep -h "gran=stage family=attn " "$1" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1; }
field() { grep -oh "ATTN_FATTN .*" "$1" | head -1 | sed -E "s/.*$2=([0-9m_a-zA-Z/.%]+).*/\1/"; }

[ -f "$BIN" ] || fail "no $BIN - run engine/build.sh first"
[ -f "$LIST" ] || fail "no $LIST - CANARY B needs the engine listing's own vgpr/lds for $KERNEL_NAME; build it with the compile-only recipe in engine/attn_fattn_kernels.h's census note (no device needed)"
# The expected values come from the listing, never from this script, so a form that moved cannot agree with
# a stale script. `.group_segment_fixed_size` is the dynamic LDS the launcher would report; the static
# arrays this kernel declares are reported in `.uds_size` on some LLVM shapes, so take the larger.
EXP_PAIR=$(python3 - "$LIST" "$KERNEL_NAME" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for chunk in re.split(r"\n\s*- \.args:", text):
    # The listing carries the MANGLD name (`_ZN6engine7kernels15attn_fattn_waveE8CoreArgs`), so an exact
    # `.name: attn_fattn_wave` never matched and the burst aborted on a kernel that was present and correct.
    # Substring match, but still one kernel's own block: a chunk is one function's metadata.
    if not re.search(r"^\s*\.name:\s+\S*" + re.escape(sys.argv[2]) + r"\S*$", chunk, re.M):
        continue
    vgpr = re.search(r"\.vgpr_count:\s+(\d+)", chunk)
    # The block's LDS can be reported as a dynamic segment or as `.uds_size` depending on the LLVM shape, so
    # take whichever the listing actually carries (whichever is larger, if it carries both).
    sizes = [int(m.group(1)) for m in
             re.finditer(r"\.(?:group_segment_fixed_size|uds_size):\s+(\d+)", chunk)]
    if not vgpr or not sizes:
        sys.exit(1)
    print(vgpr.group(1), max(sizes))
    break
else:
    sys.exit(1)
PY
) || fail "cannot read .vgpr_count/lds for $KERNEL_NAME out of $LIST"
EXP_VGPR=${EXP_PAIR%% *}
EXP_LDS=${EXP_PAIR##* }

tools/gpu_lock.sh acquire "$TAG" || exit 1
tools/gpu_lock.sh util | tail -1

# ---------------------------------------------------------------------------------------------
echo "== CANARY A: the knob exists and refuses garbage (an unported binary aborts here)"
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-fattn 1 >/dev/null 2>&1 \
    || fail "--attn-fattn is not accepted: the form is not ported (fattn_score_handover.md §5a). Nothing to time."
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-fattn 7 >/dev/null 2>&1 \
    && fail "--attn-fattn accepted 7: the flag is parsed but inert. Refusing to time an inert knob."
# A refusal is a canary too: the WMMA form is a staged-plan kernel, so a fused-plan request must be an ERROR
# and not a quiet run of the shipped form. `--attn-touch 0` and `--attn-tile 0` are its other two refusals.
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-fattn 1 --attn-tile 0 >/dev/null 2>&1 \
    && fail "--attn-fattn 1 with --attn-tile 0 ran instead of refusing: the launcher would be routing rows to a kernel that has no rows to take."
echo "CANARY A OK"

# ---------------------------------------------------------------------------------------------
echo
echo "== CANARY B: the form banner, with fields that must match the listing being priced"
for SIDE in 0 1; do
    "$BIN" --model "$MODEL" --ids "$(ids_of 4096)" --steps 0 --attn-fattn "$SIDE" --time-waves \
        > "$OUT/canaryB_$SIDE.log" 2>&1 || fail "CANARY B side $SIDE did not finish"
done
grep -q "ATTN_FATTN" "$OUT/canaryB_1.log" \
    || fail "side --attn-fattn 1 printed no ATTN_FATTN banner: which form did this binary run?"
grep -q "ATTN_FATTN" "$OUT/canaryB_0.log" \
    && fail "side --attn-fattn 0 also printed the banner: the switch does not select the shipped path."
GOT_FORM=$(field "$OUT/canaryB_1.log" form)
GOT_VGPR=$(field "$OUT/canaryB_1.log" vgpr)
GOT_LDS=$(field "$OUT/canaryB_1.log" lds)
GOT_ROWS=$(field "$OUT/canaryB_1.log" fattn_rows)
GOT_VTILES=$(field "$OUT/canaryB_1.log" vtiles)
GOT_WF=$(field "$OUT/canaryB_1.log" wf_per_cu)
# `wf_per_cu` is in the banner precisely so this line exists: at one warp per block the LDS budget admits
# 4 blocks/CU = 4 WAVEFRONTS/CU, and §28's "occupancy did not matter" evidence was taken at 12->20. A burst
# whose annotation does not carry the wavefront count is a burst that will be read against the wrong machine.
case "$GOT_WF" in (''|*[!0-9]*) fail "banner carries no wf_per_cu; the covered rows cannot be annotated with the occupancy they actually got";; *) echo "  note: covered rows run at wf_per_cu=$GOT_WF wavefronts/CU (§28's null was shown at 12-20)";; esac
[ "$GOT_FORM" = "fattn16" ] || fail "banner says form=$GOT_FORM, expected attn_fattn_wave's own name"
[ "$GOT_VTILES" = "16" ] || fail "banner vtiles=$GOT_VTILES, the form's accumulator chain is 16 wide by static_assert"
[ "$GOT_VGPR" = "$EXP_VGPR" ] || fail "banner vgpr=$GOT_VGPR, listing says $EXP_VGPR: this binary was built from a different copy of the form. That is exactly the stale-'new'-side failure the canary exists for."
[ "$GOT_LDS" = "$EXP_LDS" ] || fail "banner lds=$GOT_LDS, listing says $EXP_LDS: same failure as vgpr, stated separately so a one-field drift names itself"
if grep -q "SPILL" "$OUT/canaryB_1.log"; then
    # D63: the operator may time a spilling placement — the only datum we own on spill cost for this exact
    # shape is llama.cpp shipping 358 spilled VGPRs on it and winning >=5x. The ack arrives from the
    # environment (never from a worker) and is echoed here, so the log says out loud that its `ms` carry
    # scratch traffic. Refusal stays the default, and the engine flag stays default 0 either way.
    [ -n "${FATTN_SPILL_ACK:-}" ] || fail "the WMMA kernel spills to scratch: §1's failure mode (128 live accumulator registers with a rolled accumulator loop) is back and every number below would be about a kernel hitting scratch. Set FATTN_SPILL_ACK=<reason> to time it anyway (D63)."
    echo "  SPILL ACKNOWLEDGED (D63, reason: $FATTN_SPILL_ACK): every ms below carries scratch traffic."
fi
case "$GOT_ROWS" in
    */*) [ "${GOT_ROWS%/*}" -gt 0 ] || fail "banner fattn_rows=$GOT_ROWS: zero rows routed to the WMMA form on a 4,096-lane wave, every row of which streams. A banner printed while nothing is covered is a banner that cannot fail."; ;;
    *) fail "banner fattn_rows=$GOT_ROWS is not n/N: cannot tell coverage from decoration"; ;;
esac
echo "CANARY B OK (form=$GOT_FORM vgpr=$GOT_VGPR lds=$GOT_LDS vtiles=$GOT_VTILES wf=$GOT_WF fattn_rows=$GOT_ROWS, listing agrees)"

# ---------------------------------------------------------------------------------------------
echo
echo "== CANARY C: a bit-difference witness on a COVERED row — identical bytes would mean one side is stale"
# 4,096 lanes, not 1,024: under this port only rows above MAXSC are covered, so a short prompt would prove
# nothing here (that is CANARY D, which asks the opposite question). `--prefill-logits last` keeps the dump
# at one row per side; the last row is the widest and is certainly covered.
for SIDE in 0 1; do
    "$BIN" --model "$MODEL" --ids "$(ids_of 4096)" --steps 0 --attn-fattn "$SIDE" \
        --out "$OUT/canC_$SIDE" > "$OUT/canC_$SIDE.log" 2>&1 || fail "CANARY C side $SIDE did not finish"
done
a=$(ls "$OUT"/canC_0/logits*.bin 2>/dev/null | tail -1)
b=$(ls "$OUT"/canC_1/logits*.bin 2>/dev/null | tail -1)
[ -n "$a" ] && [ -n "$b" ] || fail "CANARY C: no logits file on one side; the witness would be vacuous"
cmp -s "$a" "$b" && fail "CANARY C: the last row's logits are byte-identical across the two forms on a wave whose rows are all covered. A 16x16x16 f16 tile dot cannot reproduce a BF16 serial dot exactly, so this means one side ran the other side's kernel. Not timed."
echo "CANARY C OK ($a differs from $b)"

# ---------------------------------------------------------------------------------------------
echo
echo "== CANARY D: inertness on a one-pass prompt — the coverage rule, asserted rather than trusted"
for SIDE in 0 1; do
    "$BIN" --model "$MODEL" --ids "$(ids_of "$INERT_IDS")" --steps 0 --attn-fattn "$SIDE" \
        --prefill-logits all --out "$OUT/canD_$SIDE" > "$OUT/canD_$SIDE.log" 2>&1 \
        || fail "CANARY D side $SIDE did not finish"
    grep -q "attn_fattn=$SIDE" "$OUT/canD_$SIDE.log" \
        || fail "CANARY D side $SIDE: the NUMERICS line does not echo its own side; a log that cannot name its side adjudicates nothing"
done
ndiff=0
for f in "$OUT"/canD_0/*.bin; do
    [ -e "$f" ] || fail "CANARY D: no dump files; the witness would be vacuous"
    cmp -s "$f" "${f/canD_0/canD_1}" || ndiff=$((ndiff + 1))
done
[ "$ndiff" -eq 0 ] || fail "CANARY D: $ndiff dump file(s) differ on a prompt whose rows are all one-pass. The coverage rule leaked into a pinned row, which is the §07.3 decision being relaxed by a kernel flag — stop, do not time."
grep -q "ATTN_FATTN" "$OUT/canD_1.log" \
    && echo "note: the one-pass side printed an ATTN_FATTN line. That is only honest if it reads fattn_rows=0/N; check it."
echo "CANARY D OK (all dumps byte-identical at $INERT_IDS lanes)"

# ---------------------------------------------------------------------------------------------
echo
echo "== CLAIM (iii): the attn family row, paired inside one lock hold, best-of-$REPS"
# Admission first, and the rule from D59 applies: a timing harness must refuse to print a millisecond
# figure when the numerics admission is red. The admission is `run_fattn_admission.sh` (handover §5b) — a
# gate run, not a timing — and it must be green before a single line below is believed.
for R in $(seq 1 "$REPS"); do
    for W in $WIDTHS; do
        for SIDE in 0 1; do
            "$BIN" --model "$MODEL" --ids "$(ids_of "$W")" --steps 0 --time-waves --gemm lt \
                --gemm-pins all --attn-fattn "$SIDE" --out "$OUT/t_${W}_${SIDE}" \
                > "$OUT/t_${W}_${SIDE}_r${R}.log" 2>&1 \
                || fail "timing T=$W side=$SIDE rep=$R did not finish"
            grep -q "attn_fattn=$SIDE" "$OUT/t_${W}_${SIDE}_r${R}.log" \
                || fail "T=$W side=$SIDE: the run does not echo which side it was. Attributing a number to the wrong side is the classic false win; not timed."
            if [ "$SIDE" = "1" ]; then
                grep -q "ATTN_FATTN" "$OUT/t_${W}_1_r${R}.log" \
                    || fail "T=$W rep=$R: the timed log carries no ATTN_FATTN line, so its ms has no covered share beside it. A family ratio with an unknown denominator is the aggregate that cannot fail (AGENTS.md, §07.4's correction); not timed."
            fi
        done
    done
done
for W in $WIDTHS; do
    M0=$(for R in $(seq 1 "$REPS"); do gmsof "$OUT/t_${W}_0_r$R.log"; done | sort -g | head -1)
    M1=$(for R in $(seq 1 "$REPS"); do gmsof "$OUT/t_${W}_1_r$R.log"; done | sort -g | head -1)
    L0=$(for R in $(seq 1 "$REPS"); do lnsof "$OUT/t_${W}_0_r$R.log"; done | sort -g | head -1)
    L1=$(for R in $(seq 1 "$REPS"); do lnsof "$OUT/t_${W}_1_r$R.log"; done | sort -g | head -1)
    [ -n "$M0" ] && [ -n "$M1" ] || fail "T=$W: no family=attn ms parsed; a null parse must not read as a null result"
    python3 - "$M0" "$M1" "$W" "$L0" "$L1" "$OUT/t_${W}_1_r1.log" <<'PY'
import re, sys
m0, m1, w, l0, l1 = (float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
log = sys.argv[6]
# The covered share, read out of the timed log itself rather than assumed. One ATTN_FATTN line per distinct
# wave geometry; the last is the widest wave, which is the one the quadratic work lives in.
cov = re.findall(r"ATTN_FATTN .*?fattn_rows=(\d+)/(\d+)", open(log).read())
share = "unknown"
if cov:
    c, n = int(cov[-1][0]), int(cov[-1][1])
    share = f"{c}/{n} = {100.0 * c / max(n, 1):.1f} % of rows in the widest wave"
r = m0 / m1
print(f"T={w}: attn family {m0:.4f} ms / launches={l0}  ->  {m1:.4f} ms / launches={l1}  = {r:.4f}x")
print(f"     covered share of the family: {share}")
print("     NOTE: the ratio is the WHOLE family row, covered rows and uncovered ones together. Divide by"
      " the covered share before reading it as a verdict on the form; a 1.5x family ratio at 60 % coverage"
      " is a ~2.5x result on covered rows and a 0.9x family ratio at 10 % coverage is not a null.")
if r < 1.5:
    sys.exit(f"CLAIM (iii) FAILS at T={w}: {r:.4f}x against a 1.5x bar. Report it as a failure; do not"
             " re-quote the CPU-side census as the answer, and do not re-take the row hoping for a better"
             " window - quote the pair and the spread (engine/performance-status.md §6: a ruler re-taken in"
             " a different window is not an A/B).")
print(f"CLAIM (iii) PASSES at T={w}")
PY
done

# ---------------------------------------------------------------------------------------------
echo
echo "== CLOSE: release the lock, print the util seen"
tools/gpu_lock.sh util | tail -1
tools/gpu_lock.sh release "$TAG"
echo "done. Numbers above are the only ms figures this artifact produces, and only if every canary printed OK."
