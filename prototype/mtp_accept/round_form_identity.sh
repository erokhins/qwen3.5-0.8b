#!/usr/bin/env bash
# round_form_identity.sh — does the compiled-in round-form rule equal `--attn-fattn-wide 0`, and does it
# move ONLY the round?
#
# WHAT IS BEING WITNESSED. `engine/decode_step.h`'s `kMtpRoundAttnWide` makes an MTP round's body run the
# 16-row WMMA wave core whatever `--attn-fattn-wide` says, because the WIDE block's win is a PREFILL number
# (`prototype/attn_wide/A13-RESULT.md`, 27 % of the 16k wave) and `wide_body_ab.sh` measured it as a large
# loss at a round's row count. A rule that is compiled in has no flag to watch, so three things have to be
# proved by differencing runs instead of by reading the source:
#
#   1. IDENTITY-VS-THE-FLAG. At a width where the flag touches nothing but the round — a ONE-token prompt,
#      so every prefill and every plain-step body is one row and `rows > 1` is reachable only by a round —
#      the new binary's default run must be BYTE-IDENTICAL to the new binary run with `--attn-fattn-wide 0`.
#      That is the brief's "identical to the flag", and this width is the only one where it can be true: at
#      4,032 lanes the flag also re-forms the prefill waves, so the two sides legitimately differ there and
#      a byte-compare there would fail for a reason that has nothing to do with the rule.
#   2. INERTNESS. At the real width, `--attn-fattn-wide 0` on the new binary must be byte-identical to the
#      same flag on the PRE-change binary: when the flag already asks for the non-wide core the rule adds
#      nothing, so nothing else in the run can have moved either.
#   3. THE CANARY. A rule that silently did not take would print run 2's numbers with the rule "applied".
#      So the new default run must DIFFER from the pre-change default run at the real width — same prefill
#      form on both sides, so the only thing that can move is the round — and the new binary must echo
#      `attn_fattn_wide=fattn` on every `VERIFY2` line while the pre-change binary cannot print that field
#      at all. Per `AGENTS.md` (the 2026-10-17 row): assert a behavioural canary inside the window.
#
# WHAT IT COMPARES. `--mtp-verify 1` dumps each round's two logits rows as raw F32 (`verify2_<step>.bin`,
# [2][kVocab], F32 like every logits dump in this driver — a comparator guessing dtype from file size is
# `AGENTS.md`'s other trap), and the `VERIFY2` lines carry both rows' argmax and top logit. So the round's
# numerics are compared bit for bit, not by a summary. The `VERIFY2` comparison STRIPS the
# `attn_fattn_wide=` field, because a pre-change binary has no such field and its absence is check 3's
# business, not check 1's or 2's.
#
# WHAT IT REFUSES TO PRINT. Any arm that died (no `VERIFY2` line at all, or a round count that disagrees
# with `--steps`), and any new-binary arm whose `VERIFY2` echo does not read `attn_fattn_wide=fattn` — a
# rule that did not take makes all three verdicts meaningless, and a green verdict from a harness that
# parsed nothing is this repo's oldest failure.
#
# NO TIMING. Nothing here reads a millisecond field; the dumps either match or they do not.
#
# USAGE
#   tools/gpu_lock.sh status                          # free, and no STALE holder
#   PRE_BIN=<binary built at the tip before the change> \
#       prototype/mtp_accept/round_form_identity.sh
# Env: PRE_BIN (required), BIN (default engine/build/decode_run), MODEL, WIDE_IDS (default
# tmp/verify_body_scale/ids4032.csv), OUT (default tmp/round_form_identity), STEPS (rounds at the wide
# width, default 8), NARROW_TOKENS (default 1).
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

BIN=${BIN:-engine/build/decode_run}
PRE_BIN=${PRE_BIN:?PRE_BIN=<path to a decode_run built before the change> is required}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
WIDE_IDS=${WIDE_IDS:-tmp/verify_body_scale/ids4032.csv}
OUT=${OUT:-tmp/round_form_identity}
STEPS=${STEPS:-8}
NARROW_TOKENS=${NARROW_TOKENS:-1}

[ -x "$BIN" ] || { echo "no binary at $BIN"; exit 1; }
[ -x "$PRE_BIN" ] || { echo "no pre-change binary at $PRE_BIN"; exit 1; }
[ -f "$WIDE_IDS" ] || { echo "no wide fixture at $WIDE_IDS (see prototype/mtp_accept/verify_body_scale.sh)"; exit 1; }
wide_tk=$(awk -F',' '{print NF}' "$WIDE_IDS") || { echo "cannot read $WIDE_IDS"; exit 1; }
[ "$wide_tk" -ge 2 ] || { echo "$WIDE_IDS has $wide_tk fields; too few to be a fixture"; exit 1; }

# The narrow fixture: a prompt of exactly `NARROW_TOKENS` ids, so no body but a round can be wider than one
# row. Token 248045 is the rendered-turn opener the dec8 prompt starts with, i.e. a real id.
mkdir -p "$OUT" || exit 1
narrow="$OUT/ids_narrow.csv"
awk -v n="$NARROW_TOKENS" 'BEGIN { for (i = 0; i < n; i++) printf "%s%s", 248045, (i + 1 < n ? "," : "") }' \
    > "$narrow"
narrow_tk=$(awk -F',' '{print NF}' "$narrow")
[ "$narrow_tk" = "$NARROW_TOKENS" ] || { echo "narrow fixture has $narrow_tk fields"; exit 1; }

tools/gpu_lock.sh acquire round-form-identity || exit 1
trap 'tools/gpu_lock.sh release round-form-identity >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util | tail -1) (discounted: a worker window); no timing taken"

arm() {  # $1 label $2 binary $3 ids-file $4 steps $5 extra-flags
    local label="$1" bin="$2" ids="$3" steps="$4" extra="$5"
    local dir="$OUT/$label"
    rm -rf "$dir"; mkdir -p "$dir"
    "$bin" --model "$MODEL" --ids-file "$ids" --steps "$steps" --cap 4332 --no-stop \
           --gdn-loop 1 --mtp-verify 1 $extra --out "$dir" > "$OUT/$label.log" 2>&1
    local rc=$?
    local n; n=$(grep -c '^VERIFY2 ' "$OUT/$label.log")
    if [ "$rc" -ne 0 ] || [ "$n" -eq 0 ]; then
        echo "FAIL: arm '$label' rc=$rc printed $n VERIFY2 lines (see $OUT/$label.log)"
        return 1
    fi
    # The per-round logits dumps are the comparison object; a run that wrote none compared nothing.
    local f; f=$(find "$dir" -name 'verify2_*.bin' | wc -l)
    if [ "$f" -eq 0 ]; then echo "FAIL: arm '$label' wrote no verify2_*.bin dumps"; return 1; fi
    echo "  $label: $n rounds, $f logit dumps"
    return 0
}

echo "== four arms: (new|pre) x (default | --attn-fattn-wide 0), at $wide_tk lanes"
arm wide_new      "$BIN"     "$WIDE_IDS" "$STEPS" ""              || exit 1
arm wide_new_w0   "$BIN"     "$WIDE_IDS" "$STEPS" "--attn-fattn-wide 0" || exit 1
arm wide_pre      "$PRE_BIN" "$WIDE_IDS" "$STEPS" ""              || exit 1
arm wide_pre_w0   "$PRE_BIN" "$WIDE_IDS" "$STEPS" "--attn-fattn-wide 0" || exit 1
echo "== narrow arms: $narrow_tk-token prompt, so only a round is wider than one row"
arm narrow_new    "$BIN"     "$narrow"   4        ""              || exit 1
arm narrow_new_w0 "$BIN"     "$narrow"   4        "--attn-fattn-wide 0" || exit 1

# The echo check, before any verdict: every new-binary VERIFY2 line must name the core that ran, and it must
# name `fattn`. Counting rather than grepping once, because one round on another form is still a rule that
# did not take.
new_rounds=$(cat "$OUT"/wide_new.log "$OUT"/wide_new_w0.log "$OUT"/narrow_new.log "$OUT"/narrow_new_w0.log \
             | grep -c '^VERIFY2 ')
new_fattn=$(cat "$OUT"/wide_new.log "$OUT"/wide_new_w0.log "$OUT"/narrow_new.log "$OUT"/narrow_new_w0.log \
            | grep '^VERIFY2 ' | grep -c 'attn_fattn_wide=fattn')
pre_echo=$(grep '^VERIFY2 ' "$OUT/wide_pre.log" | grep -c 'attn_fattn_wide=')
echo "== echo: $new_fattn of $new_rounds new-binary rounds print attn_fattn_wide=fattn;"
echo "   pre-change binary prints the field $pre_echo times (0 is correct: it predates the rule)"
if [ "$new_rounds" -eq 0 ] || [ "$new_fattn" -ne "$new_rounds" ] || [ "$pre_echo" -ne 0 ]; then
    echo "ABORT: the form echo does not establish the rule; no verdict below is attributable."
    exit 1
fi

cmp_arms() {  # $1 a $2 b -> empty on identical, else a diff note
    local a="$OUT/$1" b="$OUT/$2"
    local d
    d=$(find "$a" -type f -printf '%f\n' | sort)
    [ -n "$d" ] || { echo "no dumps in $1"; return; }
    # Every file the arm wrote, not just the round's: the per-step `logits*.bin` and `sampled.txt` are the
    # plain path, and if any of THOSE moved the rule reached outside the round.
    local dd; dd=$(diff -rq "$a" "$b" 2>&1)
    if [ -n "$dd" ]; then echo "$dd" | head -3; return; fi
    # The VERIFY2 lines, with the field a pre-change binary cannot print removed (see header).
    local la lb
    la=$(grep '^VERIFY2 ' "$OUT/$1.log" | sed 's/ attn_fattn_wide=[a-z_0-9]*//')
    lb=$(grep '^VERIFY2 ' "$OUT/$2.log" | sed 's/ attn_fattn_wide=[a-z_0-9]*//')
    [ "$la" = "$lb" ] || { echo "VERIFY2 lines differ"; return; }
    echo ""
}

v1=$(cmp_arms narrow_new narrow_new_w0)
v2=$(cmp_arms wide_new_w0 wide_pre_w0)
v3=$(cmp_arms wide_new wide_pre)
echo "== verdicts"
[ -z "$v1" ] && echo "  1 IDENTITY-VS-THE-FLAG at $narrow_tk lanes: PASS (byte-identical to --attn-fattn-wide 0)" \
             || echo "  1 IDENTITY-VS-THE-FLAG at $narrow_tk lanes: FAIL ($v1)"
[ -z "$v2" ] && echo "  2 INERTNESS at $wide_tk lanes with the flag at 0: PASS (identical to the pre-change binary)" \
             || echo "  2 INERTNESS at $wide_tk lanes with the flag at 0: FAIL ($v2)"
[ -n "$v3" ] && echo "  3 CANARY at $wide_tk lanes: PASS (the round's bits moved: $v3)" \
             || echo "  3 CANARY at $wide_tk lanes: FAIL (new default identical to pre-change: the rule took no effect)"
echo "== logs and dumps in $OUT; correctness only, no timing taken"
[ -z "$v1" ] && [ -z "$v2" ] && [ -n "$v3" ]
