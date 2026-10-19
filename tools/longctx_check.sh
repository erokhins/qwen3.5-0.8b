#!/usr/bin/env bash
# longctx_check.sh — does the shipped engine SERVE a context past the old 4,608-lane wall?
#
# WHY THIS EXISTS. `kAttnMaxSplit` (`engine/kernels_attn.cpp`) is the second half of the engine's lane
# ceiling, `MAXSC × kAttnMaxSplit` (`MAXSC` in `prototype/attn_decode/attn_decode_kernels.h`). At the
# old 9 the ceiling was 4,608 lanes and a wider prompt was a THROWN exception, not a slow run, so no
# gate in this tree has ever fed it a deep row. Raising the constant moves a boundary, and a boundary
# needs both halves of a test: the widths it now admits must actually run, and the width just past it
# must still refuse. `tools/attn_ceiling_proof.py` is the CPU half (which quantities the cap may
# reach); this is the device half (the admitted widths really run, and the row they produce is the same
# row the short way produces).
#
# WHY ONE WIDTH COSTS TWO DEEP RUNS. Every cheap check that could catch a WRONG PARTIAL INDEX has to
# see the same row produced two ways, because a mis-indexed partial is not a missing row — it is a row
# with the wrong answer in it (results.md §20.1's aliasing, §19's room-13.25 rows). So per width:
#   deep    --ids P[0:W]             --steps 0
#           row W-1 arrives as the last row of a 512-row wave, through the staged tiled wave kernel;
#   prefix  --ids P[0:W-1] --steps 1 --replay P[W-1]
#           the SAME row W-1 arrives as one generation step off a one-token-shorter prompt, through
#           the one-row fused kernel. Same tokens, same position, same lanes, two kernels and two GEMM
#           routes. Both sides write `logits{W-1}.bin`, so the comparison is file against file.
# `--steps 1` on the prefix side is the one deviation from a pure `--steps 0` pass, and the reason is a
# host-memory wall, not taste: the only way to price an INTERIOR prefill row is `--prefill-logits all`,
# which at 32,768 rows would hold 32,768 × kVocab × 4 B ≈ 30.5 GiB of logits on the host.
#
# NO TIMING. No `--time-*` flag is passed and no millisecond figure is printed. The engine's own
# `plan_match=` field lives in the timed clock path (`engine/stage_clock.cpp`), and those passes re-run
# the prompt two or three times, so the plan check here uses the UNTIMED `LAUNCHSTAT` counters and
# derives what they must read from `engine/gate.py`'s own `launch_plan()`/`wave_expect()`. The check
# prints its own `plan_match=1` from that comparison and says where the number came from.
#
# USAGE
#   tools/longctx_check.sh [--model DIR] [--bin PATH] [--widths 8192,16384,32768]
#                          [--work DIR] [--wave-rows N] [--with-refusal]
#   --with-refusal  also runs ceiling+1 lanes and requires the engine to REFUSE. Opt-in because the
#                   refusal happens in the last wave, so seeing it costs one full deep prefill.
#   --selftest      NO ENGINE, NO GPU: writes synthetic logs and dumps for a 1,024-lane width, once
#                   clean and once with four injected defects (a clamped STATE cap, a NaN logit, a
#                   wrong `attn` launch count, a diverged prefix row), and requires the clean pair to
#                   read PASS and the injected pair to read FAIL. A driver that can only print PASS is
#                   decoration — this tree has been bitten by that at least four times — so the driver
#                   carries its own must-catch set and can be run by a reviewer on a CPU.
# Exits nonzero if any width fails. One `WIDTH <W>: PASS|FAIL` line per width, then a summary.
#
# This is a device check: hold `tools/gpu_lock.sh` around it, one run at a time, and take no timing from
# it (it takes none). Typical cost is 2 deep prefills per width, so the 32,768-lane width dominates.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 2

MODEL=models/Qwen3.5-0.8B
BIN=engine/build/decode_run
WIDTHS=8192,16384,32768
WORK=tmp/longctx
WAVE=512
WITH_REFUSAL=0
FIXTURE=0
SELFTEST=0
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python3

while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODEL=$2; shift 2 ;;
        --bin)          BIN=$2; shift 2 ;;
        --widths)       WIDTHS=$2; shift 2 ;;
        --work)         WORK=$2; shift 2 ;;
        --wave-rows)    WAVE=$2; shift 2 ;;
        --with-refusal) WITH_REFUSAL=1; shift ;;
        --fixture)      FIXTURE=1; shift ;;
        --selftest)     SELFTEST=1; shift ;;
        -h|--help)      sed -n '2,44p' "$0"; exit 0 ;;
        *)              echo "longctx: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

[ "$FIXTURE" -eq 1 ] || { [ -x "$BIN" ] || { echo "longctx: no engine binary at $BIN — run engine/build.sh first" >&2; exit 2; }; }
[ "$FIXTURE" -eq 1 ] || { [ -d "$MODEL" ] || { echo "longctx: no model directory at $MODEL" >&2; exit 2; }; }
mkdir -p "$WORK" || exit 2
# The ceiling is READ, never restated — gate.py's own readers, so this check cannot quietly disagree
# with the gate about what the engine admits, and cannot be made to pass by editing a copy here.
CEILING=$("$PY" -c 'import sys; sys.path.insert(0, "engine"); import gate; print(gate.lane_ceiling())') \
    || { echo "longctx: cannot read the lane ceiling through engine/gate.py::lane_ceiling()" >&2; exit 2; }
CEIL_NOTE=$("$PY" -c 'import sys; sys.path.insert(0, "engine"); import gate; print(gate.lane_ceiling_note())')
MAXWAVE=$("$PY" -c 'import sys; sys.path.insert(0, "engine"); import gate; print(gate.wave_row_ceiling())') \
    || { echo "longctx: cannot read kMaxWaveRows through gate.wave_row_ceiling()" >&2; exit 2; }
if [ "$WAVE" -gt "$MAXWAVE" ]; then
    echo "longctx: --wave-rows $WAVE exceeds kMaxWaveRows $MAXWAVE (engine/decode_step.h); using $MAXWAVE" >&2
    WAVE=$MAXWAVE
fi

# One token stream, cut to prefixes with `cut`, so "the same prompt prefix" is structural rather than a
# promise: the prefix run's ids are literally the deep run's ids minus the last field. Fixed seed, so a
# rerun is the same run; ids stay in the normal text range, away from the special ids at the top.
MAXW=$(echo "$WIDTHS" | tr ',' ' ' | tr ' ' '\n' | sort -n | tail -1)
if [ "$WITH_REFUSAL" -eq 1 ] && [ "$((CEILING + 1))" -gt "$MAXW" ]; then
    # The refusal case asks for ceiling+1 lanes, so the id pool must reach that far. It did not, and the
    # first run of this check therefore fed the REFUSAL case 32,768 ids, ran a perfectly legal prompt and
    # reported "the ceiling moved" — a vacuous case wearing a real failure (AGENTS.md: a negative control
    # whose geometry cannot fire is not a control). Generating to max(widths, ceiling+1) is the fix; the
    # field count assertion in ids_file_of() is the second one, because a silently short prompt is the
    # failure mode, not the missing width per se.
    MAXW=$((CEILING + 1))
fi
IDS="$WORK/ids.csv"
"$PY" - "$MAXW" "$IDS" <<'PYIDS'
import random, sys
n, path = int(sys.argv[1]), sys.argv[2]
rng = random.Random(20261019)
with open(path, "w") as f:
    f.write(",".join(str(rng.randrange(1000, 60000)) for _ in range(n)))
print(f"longctx: {n} deterministic token ids (seed 20261019) -> {path}")
PYIDS
[ -s "$IDS" ] || { echo "longctx: no token ids generated" >&2; exit 2; }

echo "longctx: bin=$BIN model=$MODEL wave_rows=$WAVE work=$WORK"
echo "longctx: lane ceiling as the gate reads it: $CEILING lanes"
echo "longctx:   ($CEIL_NOTE)"
echo "longctx: widths requested: $WIDTHS"

FAILED=0
RESULTS=""
ids_of() { cut -d, -f"1-$1" "$IDS"; }     # the first N ids, comma separated
ids_file_of() {                           # the first N ids, in a FILE — and this is not a style choice:
    local N=$1 OUT=$2
    cut -d, -f"1-$N" "$IDS" > "$OUT"
    local have; have=$(awk -F, '{print NF}' "$OUT")
    if [ "$have" != "$N" ]; then
        echo "longctx: BUG: width $N needs $N ids and $OUT holds $have — refusing to judge a run whose" \
             "prompt is not the width under test" >&2
        return 1
    fi
}                                         # an argv slot is capped at MAX_ARG_STRLEN 131,072 B, so a
                                          # 32,768-id prompt (191,588 B) cannot be passed at all. See
                                          # `--ids-file` in engine/decode_run.cpp.
id_at()  { cut -d, -f"$1"  "$IDS"; }      # the Nth id, 1-based
run_engine() { local LOG=$1; shift; "$BIN" "$@" > "$LOG" 2>&1; }

make_fixtures() {   # make_fixtures clean|bad — a synthetic 1,024-lane pair of runs, for --selftest
    "$PY" - "$WORK/w1024" "$1" <<'PYFIX'
"""Synthetic `decode_run` dumps for one width, so the CHECK can be checked without a GPU.

`clean` is a pair of runs whose deepest row agrees bit for bit: every line the driver parses is the line
the engine really prints, with the counters it really prints. `bad` starts from that same pair and
injects four defects at once, because a must-catch set with one item proves little: a clamped STATE cap
(the silent-clamp failure), a NaN logit (the un-priced run), a wrong `attn` launch count (a plan that
does not match), and a deepest row that differs between the two routes (the mis-indexed partial). Each
of the four has been shown to turn its own check red; see the report at
prototype/attn_wmma/evidence/longctx.md.
"""
import os
import shutil
import sys
import numpy as np

d, mode = sys.argv[1], sys.argv[2]
W = 1024
for side, rows in (("deep", W), ("prefix", W - 1)):
    out = os.path.join(d, side)
    os.makedirs(out, exist_ok=True)
    a = (np.random.default_rng(7).standard_normal(248320) * 8).astype("<f4")
    a.tofile(os.path.join(out, f"logits{W - 1}.bin"))
    open(os.path.join(out, "config.json"), "w").write("{}\n")
    with open(os.path.join(out, "sampled.txt"), "w") as f:
        f.write("# step input_token sampled_token top_logit prefix_len_before\n")
        f.write(f"{W - 1} 7 9 3.5 {W - 2}\n")
shutil.copyfile(os.path.join(d, "deep", f"logits{W - 1}.bin"),
                os.path.join(d, "prefix", f"logits{W - 1}.bin"))


def state(cap):
    return (f"STATE cap={cap} gdn=18874368 conv=1326080 kv={12288 * cap} "
            f"total={18874368 + 1326080 + 12288 * cap} per_token=12288")


with open(os.path.join(d, "deep.log"), "w") as f:
    f.write("loaded 335 tensors, 1.5457 GB resident\n")
    f.write(state(W) + "\n")
    f.write(f"PREFILL tokens={W} logits=last prefix_len={W} wave_rows=512\n")
    f.write(f"STEP kind=prefill index={W - 1} input=123 sampled=456 top_logit=35.5\n")
    f.write("LAUNCHSTAT family=attn calls=12 calls_batched=12 rows_total=6144 rows_max=512\n")
    f.write("RESULT mode=prompt steps=0 ms_total=1.000 ms_per_step=0.000 last_sampled=456\n")
with open(os.path.join(d, "prefix.log"), "w") as f:
    f.write(state(W) + "\n")
    f.write(f"PREFILL tokens={W - 1} logits=last prefix_len={W - 1} wave_rows=512\n")
    f.write(f"STEP kind=decode index={W - 1} input=789 sampled=456 top_logit=35.5 "
            f"pos_before={W - 2} pos_after={W - 1}\n")
    f.write("LAUNCHSTAT family=attn calls=18 calls_batched=12 rows_total=6144 rows_max=512\n")
    f.write("RESULT mode=steps steps=1 ms_total=1.000 ms_per_step=1.000 last_sampled=456\n")

if mode == "bad":
    p = os.path.join(d, "deep", f"logits{W - 1}.bin")
    a = np.fromfile(p, dtype="<f4")
    a[500] = np.nan                       # 1. an un-priced / NaN row
    a.tofile(p)
    q = os.path.join(d, "prefix", f"logits{W - 1}.bin")
    b = np.fromfile(q, dtype="<f4")
    b[:2000] *= 1.05                      # 2. the two routes disagree on the deepest row
    b.tofile(q)
    t = open(os.path.join(d, "deep.log")).read()
    t = t.replace(f"STATE cap={W}", "STATE cap=4096")          # 3. a silently clamped width
    t = t.replace("calls=12 calls_batched=12", "calls=13 calls_batched=12")   # 4. plan mismatch
    open(os.path.join(d, "deep.log"), "w").write(t)
print(f"longctx: wrote {'clean' if mode == 'clean' else 'INJECTED'} fixtures for width {W} in {d}")
PYFIX
}

if [ "$SELFTEST" -eq 1 ]; then
    echo "longctx: --selftest — no engine and no GPU. The driver judges synthetic fixtures twice: once"
    echo "         clean, once with four injected defects. Both colours must appear or the driver is."
    echo "         decoration, and this tree has been bitten by that more than four times."
    make_fixtures clean
    "$0" --fixture --widths 1024 --work "$WORK"
    CLEAN=$?
    make_fixtures bad
    "$0" --fixture --widths 1024 --work "$WORK"
    BAD=$?
    echo
    echo "SELFTEST: clean fixtures exited $CLEAN (want 0); injected fixtures exited $BAD (want nonzero)"
    if [ "$CLEAN" -eq 0 ] && [ "$BAD" -ne 0 ]; then
        echo "SELFTEST PASS — the driver turns green on a clean pair and red on every injected defect"
        exit 0
    fi
    echo "SELFTEST FAIL — the driver cannot see a difference it is supposed to see"
    exit 1
fi

for W in $(echo "$WIDTHS" | tr ',' ' '); do
    D="$WORK/w$W"
    echo
    echo "======== WIDTH $W lanes ========"
    if [ "$W" -gt "$CEILING" ]; then
        echo "WIDTH $W: SKIP — above the engine's lane ceiling ($CEILING lanes). This check does not"
        echo "         ask the engine for a width it is not built to serve; that is --with-refusal's job."
        continue
    fi
    if [ "$W" -le 512 ]; then
        echo "WIDTH $W: SKIP — at or below MAXSC every row is one-pass and no partial is ever written,"
        echo "         so this width cannot adjudicate anything about the split."
        continue
    fi

    if [ "$FIXTURE" -eq 0 ]; then
        rm -rf "$D/deep" "$D/prefix"; mkdir -p "$D/deep" "$D/prefix"
    fi
    if [ "$FIXTURE" -eq 1 ]; then
        RC_DEEP=0; RC_PRE=0
        echo "longctx: --fixture — judging the logs and dumps already in $D; the engine was not run"
    else
        t_start=$(date +%s)
        ids_file_of "$W" "$D/ids_deep.csv"
        ids_file_of "$((W - 1))" "$D/ids_prefix.csv"
        run_engine "$D/deep.log" --model "$MODEL" --ids-file "$D/ids_deep.csv" --cap "$W" --steps 0 \
            --wave-rows "$WAVE" --out "$D/deep"; RC_DEEP=$?
        run_engine "$D/prefix.log" --model "$MODEL" --ids-file "$D/ids_prefix.csv" --cap "$W" \
            --steps 1 --replay "$(id_at "$W")" --wave-rows "$WAVE" --out "$D/prefix"; RC_PRE=$?
        t_end=$(date +%s)
        echo "longctx: two deep runs finished (exit $RC_DEEP / $RC_PRE) after $((t_end - t_start))s of wall"
        echo "         clock — NO TIMING TAKEN; that elapsed figure is not a measurement of anything."
    fi

    # The prefix run prices its own last prefill row as well (row W-2 under `--prefill-logits last`),
    # so the two dump directories do not hold the same file set, and `attn_form_band.py` correctly calls
    # that a crashed run rather than a band difference. The compared row is W-1 and only W-1, so the
    # comparison runs over a dedicated pair of directories of symlinks; nothing in the original dumps is
    # moved, renamed or deleted, and `ls -l $D/cmp/*/` shows exactly what was judged.
    rm -rf "$D/cmp"; mkdir -p "$D/cmp/deep" "$D/cmp/prefix"
    for side in deep prefix; do
        SD=$(cd "$D/$side" && pwd)
        [ -f "$D/$side/logits$((W - 1)).bin" ] && ln -s "$SD/logits$((W - 1)).bin" "$D/cmp/$side/"
        [ -f "$D/$side/config.json" ] && ln -s "$SD/config.json" "$D/cmp/$side/"
        # The sampled columns are filtered to the compared row for the same reason as the logits: each
        # side prices its own last prefill row, so an unfiltered file differs at row W-2 by construction
        # (one side prints -1 there because it never priced it), and `attn_form_band.py` would rightly
        # report an unexplained step. Row W-1 is the pair, so row W-1 is all either side gets to show.
        [ -f "$D/$side/sampled.txt" ] && awk -v want=$((W - 1)) 'NR==1 || $1+0 == want' \
            "$D/$side/sampled.txt" > "$D/cmp/$side/sampled.txt"
    done

    "$PY" - "$D" "$W" "$RC_DEEP" "$RC_PRE" "$WAVE" <<'PYEOF'
"""One width's verdict: reads the two logs and the two dumped logits rows, prints PASS/FAIL lines, and
exits nonzero on any FAIL. No threshold is defined here. The band and the flip-room rule are read from
`engine/gate.py`, the file that owns them, and the launch plan from that same file's `launch_plan()` and
`wave_expect()`; the numerics adjudication is `engine/attn_form_band.py` run verbatim.
"""
import argparse
import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "engine"))
import gate                                    # noqa: E402  owner of every band and rule used here
import numpy as np                             # noqa: E402

d = Path(sys.argv[1])
W = int(sys.argv[2])
rc_deep, rc_pre, wave = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
fails = []


def mark(ok, what, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(what)


def read_log(name):
    p = d / name
    return p.read_text(errors="replace") if p.exists() else ""


# ---- (a) each run finishes, and finishes by succeeding --------------------------------------------
deep, pre = read_log("deep.log"), read_log("prefix.log")
for label, rc, text in (("deep", rc_deep, deep), ("prefix", rc_pre, pre)):
    bad = [s for s in ("terminate", "what():", "HSA_STATUS_ERROR", "MEMORY_FAULT", "Killed", "out_of_memory")
           if s in text]
    mark(rc == 0 and "RESULT" in text and not bad, f"{label}: exits 0 and prints a RESULT line",
         f"exit {rc}, crash markers {bad or 'none'}" if (rc or "RESULT" not in text or bad) else "")

# ---- (d) the requested width was really used — no silent clamp -------------------------------------
# Every number is read off the engine's own line and cross-checked against what was asked for.
# `kv == per_token x cap` is the allocation witness (the state really is cap slots wide), `prefix_len`
# and the priced row index are the position witnesses (the run reached row W-1 rather than stopping at
# a smaller cap), and `wave_rows` proves the wide geometry the launcher was asked for was the one used.
for label, text, want in (("deep", deep, W), ("prefix", pre, W - 1)):
    pf = re.search(r"PREFILL tokens=(\d+) logits=(\S+) prefix_len=(\d+) wave_rows=(\d+)", text)
    if not pf:
        mark(False, f"{label}: PREFILL line reports the prompt it consumed", "no PREFILL line")
    else:
        tok, lg, plen, wrows = pf.groups()
        mark((int(tok), int(plen), int(wrows)) == (want, want, wave) and lg in ("last", "all"),
             f"{label}: PREFILL consumed all {want} tokens at wave_rows={wave}",
             f"tokens={tok} prefix_len={plen} wave_rows={wrows} logits={lg}")
    st = re.search(r"STATE cap=(\d+) gdn=\d+ conv=\d+ kv=(\d+) total=\d+ per_token=(\d+)", text)
    if not st:
        mark(False, f"{label}: STATE line reports the allocation", "no STATE line")
    else:
        cap, kv, per = (int(x) for x in st.groups())
        mark(cap == W and kv == per * W, f"{label}: state allocated for cap={W}, not clamped",
             f"cap={cap} kv={kv} per_token={per} (expected cap={W}, kv={per * W})")
    idx = re.findall(r"STEP kind=\w+ index=(\d+)", text)
    mark(bool(idx) and int(idx[-1]) == W - 1, f"{label}: deepest row {W - 1} was priced",
         f"last STEP index={idx[-1] if idx else 'none'}")

# ---- (b) the logits are finite, and not vacuously so -----------------------------------------------
vocab = int(re.search(r"constexpr int kVocab\s*=\s*(\d+)",
                      (ROOT / "engine" / "state.h").read_text()).group(1))
rows = {}
for label in ("deep", "prefix"):
    f = d / label / f"logits{W - 1}.bin"
    if not f.exists():
        mark(False, f"{label}: logits{W - 1}.bin exists", "the deepest row was never written")
        continue
    a = np.fromfile(f, dtype="<f4")
    rows[label] = a
    finite = bool(np.isfinite(a).all())
    distinct = int(len(np.unique(a)))
    mark(a.size == vocab and finite and distinct > 1000,
         f"{label}: logits are finite and not vacuous",
         f"{a.size} floats (want {vocab}), all finite={finite}, "
         f"maxabs={float(np.abs(a).max()):.4g}, distinct={distinct}")

# ---- (c) the launch counts equal the plan (plan_match, derived, untimed) ---------------------------
plan = gate.launch_plan(24, 18, 6)
for label, text, plen, steps in (("deep", deep, W, 0), ("prefix", pre, W - 1, 1)):
    got = None
    for m in re.finditer(r"LAUNCHSTAT family=attn calls=(\d+) calls_batched=(\d+) "
                         r"rows_total=(\d+) rows_max=(\d+)", text):
        got = {"calls": int(m.group(1)), "calls_batched": int(m.group(2)),
               "rows_total": int(m.group(3)), "rows_max": int(m.group(4))}
    exp = gate.wave_expect({"attn": plan["attn"]}, plen, steps, wave)["attn"]
    ok = got == exp
    print(f"  plan_match={1 if ok else 0} {label}: attn counters observed {got} vs the derived plan "
          f"{exp} — gate.launch_plan(24,18,6)['attn']={plan['attn']} sites per body x "
          f"{len(gate.wave_row_sizes(plen, wave)) + steps} bodies "
          f"({plen} prompt rows at wave_rows={wave} + {steps} generation step{'' if steps == 1 else 's'})")
    if not ok:
        fails.append(f"{label}: attn launch counts equal the plan")
print("  note: this check's plan_match is its own derived comparison over the untimed LAUNCHSTAT "
      "counters; the engine's field of that name is printed only by the timed clocks "
      "(engine/stage_clock.cpp), which this check does not run, and the full §07 per-family bill is "
      "engine/gate.py section (h)'s to judge.")

# ---- (e) prefix consistency: the same row, reached two ways ----------------------------------------
if "deep" in rows and "prefix" in rows:
    a, b = rows["deep"], rows["prefix"]
    same = bool((a == b).all())
    n = int((a != b).sum())
    dot = float(np.dot(a.astype(np.float64), b.astype(np.float64))
                / (np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(b.astype(np.float64))))
    ia, va, ga = gate.top2(a)
    ib, vb, gb = gate.top2(b)
    flip = ia != ib
    adm = gate.flip_admissible((va, ga), (vb, gb))
    print(f"  measured: bit-identical={int(same)}, differing elements {n}/{a.size}, cos={dot:.12g}, "
          f"argmax {ia} vs {ib} (flipped={int(flip)}), top-1 room "
          f"{gate.room_in_ulp(va, ga):.3f}/{gate.room_in_ulp(vb, gb):.3f} BF16 ULP, flip admissible "
          f"under gate.flip_admissible = {int(adm)} (the room rule and the band below are gate.py's, "
          "not invented here)")
    spec = importlib.util.spec_from_file_location("_ab", ROOT / "engine" / "attn_form_band.py")
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ab.cmd_ab(argparse.Namespace(ship=str(d / "cmp" / "deep"), tile=str(d / "cmp" / "prefix"),
                                          prompt_len=W, all_rows=False))
    print("  engine/attn_form_band.py --ship cmp/deep --tile cmp/prefix --prompt-len "
          f"{W} (existing bands, run verbatim; every number it prints comes from engine/gate.py. The "
          "two directories hold symlinks to each side's row " + str(W - 1) + " and nothing else, "
          "because the prefix side legitimately also prices its own last prefill row, and that row is "
          "not half of this pair)")
    for ln in buf.getvalue().splitlines():
        print(f"    | {ln}")
    mark(rc == 0, "prefix consistency: row " + str(W - 1) + " is inside the published band across "
                  "the wave route and the one-row route",
         "" if rc == 0 else "attn_form_band.py exited nonzero; read its table above — a flip with "
                            "room above the published line is a defect, and one under it is band noise")
print(f"  VERDICT width {W}: " + ("PASS" if not fails else "FAIL — " + "; ".join(fails)))
sys.exit(1 if fails else 0)
PYEOF
    if [ $? -ne 0 ]; then
        FAILED=1
        RESULTS="$RESULTS\n  $W lanes: FAIL"
    else
        RESULTS="$RESULTS\n  $W lanes: PASS"
    fi
done

if [ "$WITH_REFUSAL" -eq 1 ]; then
    R=$((CEILING + 1))
    echo
    echo "======== REFUSAL at $R lanes (opt-in: the refusal happens in the last wave, so it costs"
    echo "         one full deep prefill to see) ========"
    D="$WORK/refuse"; rm -rf "$D"; mkdir -p "$D"
    ids_file_of "$R" "$D/ids_refuse.csv"
    run_engine "$D/refuse.log" --model "$MODEL" --ids-file "$D/ids_refuse.csv" --cap "$R" --steps 0 \
        --wave-rows "$WAVE" --out "$D/out"
    RC=$?
    if [ "$RC" -ne 0 ] && grep -q "kAttnMaxSplit" "$D/refuse.log"; then
        echo "  [PASS] $R lanes REFUSED, naming kAttnMaxSplit — the ceiling is a wall, not a clamp"
        grep -m1 "kAttnMaxSplit" "$D/refuse.log" | sed 's/^/         /'
    else
        echo "  [FAIL] $R lanes exited $RC without naming kAttnMaxSplit — the ceiling moved somewhere"
        echo "         other than the launcher's refusal, which is §20.1's aliasing failure"
        FAILED=1
        RESULTS="$RESULTS\n  refusal at $R lanes: FAIL"
    fi
fi

echo
echo "================ SUMMARY ================"
printf "%b\n" "$RESULTS"
echo "longctx: exit $FAILED (0 = every width that could be judged passed)"
exit $FAILED
