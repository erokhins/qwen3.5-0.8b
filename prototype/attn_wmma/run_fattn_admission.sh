#!/usr/bin/env bash
# run_fattn_admission.sh — the numerical admission for `--attn-fattn` (the WMMA attention form,
# `engine/attn_fattn_kernels.h`). Handover §5b of `prototype/attn_wmma/fattn_score_handover.md` is the
# contract; this file is that sequence with the three things the port decided bolted into it.
#
# WRITTEN TO BE RUN BY THE MAIN AGENT. It runs device gates (correctness only, no timings), and each gate
# acquires `tools/gpu_lock.sh` on its own where that gate is self-locking — so DO NOT run this script while
# holding the lock (AGENTS.md: a caller that holds the lock and then invokes a self-locking driver parks the
# child silently and looks like a hung box). Blocking, one at a time, no subagents, as always for device work.
#
# NOTHING HERE HAS BEEN RUN. No number in this file is a measurement.
#
# WHY THESE SECTIONS, IN THIS ORDER
#   0  the two fragment-map checkers. They are permanent admission cases, not a one-off investigation: the
#      score map was transcribed wrong, shipped into a prototype kernel, and every per-form check passed
#      anyway (`fattn_map_results.md`). A checker that already caught one real bug and is then dropped from
#      the gate set is a checker that permits the bug to return. Both run with their inject selftests,
#      because a control that cannot fire is not a control.
#   1  inertness. Under the coverage rule this port ships (decision (A): the WMMA kernel owns only the rows
#      §07.3 already streams, `blocks_for_row > 1`), every pinned one-pass row must come back
#      byte-identical. "Inert" is asserted here, on the pinned corpus, at both flag values — not assumed
#      from reading the launcher.
#   2  the covered rows, against the pins, inside a band that was FLOOR-PROBED rather than borrowed. The form
#      is NOT bit-identical to the shipped path — it moves the score dot into a 16x16x16 f16 tile and
#      converts the operands to 16 bits — so the admission question is exactly "is the move inside the band",
#      and max-ULP is not admissible (AGENTS.md: a 1-bit operand plus a legal reordering moves a normal
#      share of elements by <= 1 ULP and any max-ULP bound passes vacuously; and a chained dot's legal ndiff
#      is 29-47 %, so ndiff there is meaningless — `prototype/mlp_gemv/results.md` F4).
#   3  the BOUNDARY, both ways. `wide520f`/`wide640f` are the fixtures whose waves straddle MAXSC=512: below
#      the line a row is one-pass and stays on the VALU wave, above it the row streams and the WMMA form owns
#      it. This is where a coverage rule that is one lane off shows up, and §20.6's rule applies: a route no
#      fixture reaches has no tests, including tests of the gate. The mandatory rows are the last one-pass row
#      and the first streaming row of the same wave (511 and 513 lanes), and the must-catch set below includes
#      the two injections that ONLY a straddling wave can see.
#   4  the derived must-catch set, derived from the GEOMETRY of the new code and not hand-listed
#      (AGENTS.md: a hand-listed must-catch set rots into "everything" the moment the fixture changes).
#   5  the empty-list rule. Every case count this script parses is asserted equal to the count it expected
#      before any verdict is printed, because a driver whose parser matches nothing reports PASS
#      (`prototype/gdn_collapse/results.md` §1).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
GATE=engine/gate.py
# The fixtures. Named rather than globbed, and every one of them must exist: an admission that silently skips
# a missing fixture is an admission over the fixtures that happened to be present.
FIX_INERT=${FIX_INERT:-"dec8 dec64f dec300f"}     # all rows one-pass: must be byte-identical at both values
FIX_STRADDLE=${FIX_STRADDLE:-"wide520f wide640f"} # waves that cross MAXSC: both routes in one wave
FIX_COVERED=${FIX_COVERED:-"wide1024f wide4096f"} # every row of the last wave streams
for f in $FIX_INERT $FIX_STRADDLE $FIX_COVERED; do
    [ -d "data/refs/$f" ] || { echo "ABORT: fixture '$f' has no directory under data/refs/. An admission run over a population that silently shrank adjudicates nothing (AGENTS.md, the admission-population trap)."; exit 1; }
done
fail() { echo "ABORT: $*" >&2; exit 1; }
count_is() {  # count_is <label> <got> <want>  — the empty-parse rule, applied to every count read here
    [ "$2" = "$3" ] || fail "$label: parsed '$2', expected '$3'. A driver whose parser matches nothing reports PASS, so a count that cannot be wrong cannot be believed either (prototype/gdn_collapse/results.md §1)."
    echo "  $1: $2 as expected"
}
checks_of() { grep -o '[0-9]*/[0-9]* checks' "$1" | head -1 | sed 's/ checks//'; }
green_of() {  # the checker's own verdict line, e.g. "PASS: check_fattn_map — 26/26 checks" -> "26/26"
    grep -oh 'PASS: [a-z_]*[ —-]*[0-9]*/[0-9]* checks' "$1" | head -1 | sed 's/.*[ —-]\([0-9]*\/[0-9]*\) checks/\1/'
}
half() { echo "$1" | cut -d/ -f1; }   # "26/26" -> 26

echo "== SECTION 0a: the shipping placement's own census — compile-only, and it gates everything below"
# This is not a formality. The census reads the note block of the kernel the engine LAUNCHES and fails on any
# non-zero spill, which is the gate the Q placement was decided under; it also re-derives the instruction
# skeleton from the geometry constants instead of trusting a copy of the probe's numbers. A covered row from a
# kernel that spills 49 VGPRs into scratch is not a candidate for a default, so nothing below is worth a device
# minute until this prints PASS — and `--attn-fattn` stays default 0 while it does not.
CENSUS_RC=0
$PY prototype/attn_wmma/fattn_ship_census.py || CENSUS_RC=$?
if [ "$CENSUS_RC" != 0 ] && [ "${SKIP_CENSUS:-0}" != "1" ]; then
    fail "fattn_ship_census.py is red (rc=$CENSUS_RC). Read the reason it prints: as of this writing it FAILS on 49 spilled VGPRs and 72 B/thread of scratch, which means this admission characterises a non-default path and is not a route to making it the default (prototype/attn_wmma/fattn_ship_results.md §1, §6). To run the device sections anyway, say so explicitly with SKIP_CENSUS=1."
fi

echo
echo "== SECTION 0b: the fragment-map checkers, with their inject selftests (permanent admission cases)"
# Both checkers, clean and selftest. Four device runs, each a few hundred ms, correctness only. Each one's
# own record count is asserted, because "PASS" printed over an empty case list is the failure this repo has
# been bitten by three times.
bash prototype/attn_wmma/build_map_probe.sh || fail "the score-map probe does not build"
bash prototype/attn_wmma/build_fattn_cb.sh || fail "the C->B probe does not build"
mkdir -p tmp/gcc/attn_fattn
for probe in "check_fattn_map.py:" "check_fattn_map.py:--selftest" "check_fattn_cb.py:" "check_fattn_cb.py:--selftest"; do
    script=${probe%%:*}; flags=${probe##*:}
    tag=$(basename "$script" .py)${flags:+-self}
    $PY "prototype/attn_wmma/$script" $flags > "tmp/gcc/attn_fattn/$tag.txt" 2>&1 \
        || fail "$script $flags is red. If it is the score map: the operand law this kernel is written against no longer holds and the port must stop, not be re-banded. If it is an injection that did NOT fail, the checker can no longer see the bug it was built for and its PASS is worth nothing."
    label="$script${flags:+ $flags}"
    # Each checker guards its own population: `--expect` defaults to its whole case table and the script FAILS
    # if the count it ran is not that (`check_fattn_cb.py:189`, `check_fattn_map.py:380`). The driver does not
    # re-derive that number by regexing identifiers out of the source — it would disagree with the checker for
    # reasons unrelated to the kernel, and today it would: `check_fattn_cb.py --selftest` drives
    # `3 * len(CASES)` checks, not a count of `cb_*` literals. So: require the checker's own green verdict,
    # require a non-zero count, and require a SELFTEST to drive strictly MORE checks than the clean run did.
    # A selftest that adds no case built no injection, and that is the empty-parse failure wearing a costume.
    got=$(green_of "tmp/gcc/attn_fattn/$tag.txt")
    case "$got" in */*) ;; *) fail "$label: no green verdict line (got '$got')." ;; esac
    n=$(half "$got")
    [ "$n" -ge 1 ] || fail "$label: $got — a check list that can be empty cannot fail (prototype/gdn_collapse/results.md §1)."
    if [ -n "$flags" ]; then
        clean=$(green_of "tmp/gcc/attn_fattn/$(basename "$script" .py).txt")
        cn=$(half "${clean:-0/0}")
        [ "$n" -gt "$cn" ] || fail "$label: drove $n checks where the clean run alone drove $cn, so the selftest added nothing and its injections were never built."
        echo "  $label: $got, i.e. $((n - cn)) more checks than the clean run — the injections ran"
    else
        echo "  $label: $got checks"
    fi
done
echo "SECTION 0 OK"

echo
echo "== SECTION 1: inertness on the pinned corpus (--attn-fattn 0 vs 1, one-pass rows only)"
for f in $FIX_INERT; do
    $GATE --fixture "$f" --attn-fattn 0 || fail "gate.py --fixture $f --attn-fattn 0 is red. The shipped path itself is broken; nothing below can be interpreted."
    $GATE --fixture "$f" --attn-fattn 1 || fail "gate.py --fixture $f --attn-fattn 1 is red on a fixture with NO covered rows. The flag moved a pinned bit on a route it is not supposed to touch: that is decision (A) being violated, not a band question."
done
echo "SECTION 1 OK (both values green on: $FIX_INERT)"

echo
echo "== SECTION 2: the covered rows, against the pins, in a floor-probed band"
# 2a. the floor, so the band is measured. F1 of this probe is the score dot with a pairwise sum, which is the
#     legal order a 16x16x16 tile computes; the band this admission uses is sized by that number and by the
#     cancellation band §10.1 requires, not by taste.
$PY prototype/attn_decode/floor_probe.py --fixture dec8 --long-fixture p300 \
    || fail "floor_probe.py did not run: without it the band below is a borrowed number, not a measured one."
# 2b. the form against the pins. `--scope form` defers the wave-carry comparisons that hold the attention
#     form identical on both sides and vary only --wave-rows: they cannot adjudicate this change
#     (`engine/gate.py`'s SCOPES, `prototype/attn_decode/results.md` §18.2), and `--scope full` on the
#     straddling fixtures below is where they are judged again.
for f in $FIX_COVERED; do
    $GATE --fixture "$f" --attn-fattn 1 --scope form \
        || fail "gate.py --fixture $f --attn-fattn 1 --scope form is red. Report WHICH rows fail and their decision room before choosing a remedy (the branch-2 rule: a correctness defect on the shipping path outranks speed work, and a gate is never loosened to route around it)."
done
echo "SECTION 2 OK (covered rows inside the band on: $FIX_COVERED)"

echo
echo "== SECTION 3: the boundary, both ways — one-pass rows bit-identical, streaming rows inside the band"
for f in $FIX_STRADDLE; do
    # The straddling wave is the only place the split itself is tested: a coverage rule one lane off either
    # drops a row (combine merges an uninitialised slot) or double-writes one (two kernels, one slot), and
    # only a wave that carries BOTH routes can show it. Both scopes, because this is also the wave where the
    # deferred wave-carry comparisons become meaningful again.
    $GATE --fixture "$f" --attn-fattn 0 --scope full || fail "gate.py --fixture $f --attn-fattn 0 --scope full is red: the frozen side of the boundary is broken, so there is no reference to compare against."
    $GATE --fixture "$f" --attn-fattn 1 --scope full || fail "gate.py --fixture $f --attn-fattn 1 --scope full is red on a wave that straddles MAXSC. Do not read this as a band question first: check whether the failing rows are the ones exactly at the boundary (rows of 511 and 513 lanes), because that is the split being wrong, not the arithmetic."
done
echo "SECTION 3 OK (boundary waves: $FIX_STRADDLE)"
# The boundary is not "a straddling fixture ran"; it is two named rows. Row of 511 lanes = the last one-pass
# row (one pass covers 512 lanes), row of 513 = the first streaming row, and one fattn block of 16 rows that
# STRADDLES 512 so a block owning rows on both sides is exercised, not just a block wholly on one side. If a
# gate log shows the fixture green without naming those rows, the boundary was never adjudicated.
# The boundary is asserted by the FIXTURE CHOICE here (wide520f/wide640f straddle MAXSC=512, and gate.py
# judges every row of the wave), not by grepping a log format this worker has never seen — a grep written
# against an output format nobody ran is a check that fails for a reason unrelated to the kernel. What is
# recorded instead is the exact conversion to make when the first real log exists.
for row in 511 512 513 528; do
    echo "TODO(first-run): name the field gate.py prints for a judged row's lane count and make row $row a hard must-appear case. Until then the boundary is carried by the fixture straddle, and this NOTE is the honest state of the check."
done

echo
echo "== SECTION 4: the must-catch set, DERIVED from the geometry of the new code"
# Each entry names the geometry that can make it fire. An injection that fires everywhere is not a control
# (§20.6), and one that fires nowhere is decoration, so both sets are read off the shape of the change:
#   * cb_score_slot_swap  — the transposed score operand. Reaches every covered row; caught at the operand
#                           dump as well as the output, because a transposed score folded down the wrong axis
#                           can still print a plausible row.
#   * cb_no_shfl          — the C->B butterfly. Its BLIND SPOT is recorded in `check_fattn_cb.py`'s header:
#                           the output is unchanged, only the operand dump moves, because each half-warp
#                           supplies its own parity of the KV axis. Kept as an operand-level case for that
#                           reason, and never quoted as an output-level control.
#   * tile-model negative set (`--neg-wide`) — the tiled wave's own injections, run at the WIDE anchors,
#                           because the partial-slot aliasing class (§20.1) only exists above MAXSC: a
#                           one-pass row writes no partial and so has no slot to steal.
$PY prototype/attn_tile_model.py --neg-wide || fail "attn_tile_model.py --neg-wide is red: the shared-partial-slot class of bug is live, and this form writes those same slots."
# And what NO existing injection reaches, stated rather than quietly omitted: a WRONG ROW SLICE inside the
# WMMA block (the `fattn_row_of` union of 16 rows' [blo, bhi) ranges) has no NEG_* hook, and no pinned
# fixture with a single-wave straddle of exactly 16 rows can be built from the existing dumps. The two
# things that DO cover it are the per-row geometry being computed by one function and the wide-anchor
# comparisons of Section 2 — and if this form ever becomes the default for streaming rows, that gap needs a
# named injection before, not after, the first streaming-row-only bug report.
echo
echo "-- Section 4b: magnitude, because cosine is scale-invariant and the rowsum defect is a scale"
# The numbers are not intuition here: `.venv/bin/python prototype/attn_wmma/fattn_rowsum_witness.py` measures
# the exact defect the committed port had (a row's denominator missing its cross-pair half) over 4096 rows and
# reports cos = 1.000000 and argmax unchanged on EVERY row, with meanrel at a median of ~1.0. So a cosine band
# of any tightness admits that bug, and the admission has to gate on a magnitude-sensitive statistic beside the
# band. Both are printed by `engine/gate.py`; this section requires the magnitude one to be present in the log,
# not inferred from the cos line.
.venv/bin/python prototype/attn_wmma/fattn_rowsum_witness.py | tee tmp/gcc/attn_fattn/rowsum_witness.txt
grep -q "cos      min=1.000000" tmp/gcc/attn_fattn/rowsum_witness.txt || fail "the rowsum witness no longer reports cosine invariance. Either numpy changed or the metric being gated on is not the one this file assumes; re-read it before trusting any cos floor."
grep -q "meanrel" engine/gate.py \
    || fail "engine/gate.py carries no magnitude-sensitive band. A cosine-only admission would have passed the half-denominator defect the rowsum witness above measures, so this check is the difference between an admission and a formality: add the meanrel floor (its value comes from floor_probe.py, Section 2) before judging a covered row."
echo "  gate.py carries a meanrel band: that is the statistic which can see a uniform scale (review point D)."

echo
echo "SECTION 4 OK (see the comments: one named gap, stated rather than hidden)"

echo
echo "== SECTION 5: the re-census at the shipped placement, and the occupancy claim, both after the swap"
# Compile-only, no device. `fattn_census.py` prices the pipe-weighted slots/touch and the driver probe reads
# blocks/CU from KFD with its three-control canary; both were re-taken after the operand swap because the
# swap moves LDS addressing, and both are re-checked here so the admission log carries its own occupancy
# story instead of citing one from a previous form.
$PY prototype/attn_wmma/fattn_census.py || fail "fattn_census.py did not run"
bash prototype/attn_wmma/build_occupancy_fattn.sh || fail "the occupancy probe does not build"
echo
echo "ADMISSION SEQUENCE COMPLETE — every section above had to print OK."
echo "Only then does prototype/attn_wmma/run_fattn_burst_window.sh get to print a millisecond figure (D59)."
