#!/usr/bin/env python3
"""run.py — plan writer and driver for `prototype/attn_m2` (R1: attention row-tiling at M = 2).

WHAT THIS DOES. Writes device plans whose WAVES sit on the boundaries M = 2 moves — a pair of adjacent
query rows at exactly MAXSC and MAXSC+1 lanes (one one-pass, one streaming), pairs straddling lane
blocks 2 and 3, the widest pair a 1024-token context holds, the shortest pairs a wave has, and the
single-row decode steps where the pair's second row does not exist — then runs `./attn_m2`, which
launches BOTH forms over the same operands and prints a per-row bit-diff count.

WHAT THIS DOES NOT DO, AND THAT IS DELIBERATE.

  * It does NOT acquire `tools/gpu_lock.sh`. Three gate drivers in this repo self-lock, and a caller that
    holds the lock and then runs one of them deadlocks it silently (AGENTS.md trap row). This one does
    not, so the CALLER holds the lock around it — HANDOFF.md step 0 says so in those words.
  * It takes no timing and prints no ms. `attn_m2`'s `--burst/--reps` path is the main agent's; this
    driver never passes those flags.
  * It does not widen a check. Bit-identity means zero differing elements per row.

NEGATIVE MODE, and why it is shaped the way it is (results.md §3). Two ways a negative control is vacuous
were both hit here, and the driver now refuses both in advance:

  1. An injection whose DERIVED must-catch set is empty on every plan it would run. `partrow` aliases a
     SPLIT partial and only a streaming row writes one, so on a plan of 1–8-lane rows it moves nothing
     no matter what the kernel does. The driver computes `m2_model.neg_expect` per injection per plan
     and runs only the plans where the set is non-empty; if NO plan qualifies it stops with an
     assertion, before touching the device. An empty list cannot print PASS.
  2. A "moved" count measured as form-vs-form. That is the wrong baseline for a FORM-NEUTRAL injection:
     `readpast` makes both forms do the same wrong thing, so they agree and the count is zero even
     though the bug fired everywhere. So negative mode runs each plan TWICE — `--neg none` and
     `--neg X` — and compares each form's injected output against **that same form's clean output**, per
     (row, head), from the dumps the harness already writes. The form-vs-form agreement is still
     asserted separately (it is the §1.2 claim), and it is asserted to be zero.

Run (repo root, inside one gpu_lock hold, blocking):
    .venv/bin/python prototype/attn_m2/run.py plans
    .venv/bin/python prototype/attn_m2/run.py ab
    .venv/bin/python prototype/attn_m2/run.py negative --neg partrow
CPU-only, no device, no lock:
    .venv/bin/python prototype/attn_m2/run.py negmatrix      # which plan proves which injection
    .venv/bin/python prototype/attn_m2/m2_model.py selftest|guard|compare|negative|traffic
"""
from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "attn_decode")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reference as R                       # noqa: E402
import attn_tile_model as T                 # noqa: E402  MAXSC, windows_of
import m2_model as M                        # noqa: E402  the mirror owns the case builders and the sets

HERE = Path(__file__).resolve().parent
WORK = Path("/tmp/attn_m2")   # scratch lives outside the tree, like `attn_decode/run.py`
BIN = HERE / "attn_m2"
MAXSC, N_Q, DH = T.MAXSC, T.N_Q, T.DH
HEAD_BYTES = DH * 2          # one (row, head) of a dump: the harness writes [row][head][dim] BF16

# The device cases. Content comes from `m2_model.wide_plan` — real pinned rows cycled over a POPULATED
# cache — because no fixture on this box has a prompt long enough to reach these positions (§20.2:
# `wide1024f` has long decode rows but no per-layer prefill stages), and a bit-identity A/B needs
# identical operands, not HF ground truth. The numeric anchor for the streaming route stays `wide1024f`'s
# pinned decode rows, which are the tail cases below.
#
# (tag, start position, nrows, why)
WAVE_SPECS = [
    ("pair-at-maxsc", MAXSC - 1, 4, "lanes 512..515: pair 0 straddles one-pass/streaming"),
    ("maxsc-minus-1", MAXSC - 2, 4, "lanes 511..514: the same boundary one row earlier"),
    ("pair-just-above", MAXSC, 8, "lanes 513..520, split 2, unequal spans: the V-overlap case"),
    ("straddles-block-2", 2 * MAXSC - 1, 4, "lanes 1024..1027, split 3"),
    ("straddles-block-3", 3 * MAXSC - 1, 4, "lanes 1536..1539, split 4"),
    ("pair-widest-1024", 1020, 4, "lanes 1021..1024: the union at block 0 is exactly MAXSC"),
    ("pair-shortest", 0, 8, "lanes 1..8: the pairs with the least to share (0.778, not 0.5)"),
]

# Waves with enough rows to sit ABOVE the bandwidth plateau, which is the only regime where a traffic
# saving can show at all: M = 2 halves the grid as well as the bytes, and this box's read bandwidth only
# reaches 233-240 GB/s from ~80 blocks up (results.md §7). The four §4 waves were 16-32 blocks, so they
# measured the wrong regime. These plans are written here so the decisive burst can be run; this driver
# still never passes `--burst`, which is the main agent's flag.
TIMING_SPECS = [
    ("wave64-at-1024", 1020, 64, "384 ship blocks / 192 m2 blocks: §7 predicts ~1.95x here"),
    ("wave64-at-2048", 2044, 64, "640/320 blocks"),
    ("wave64-at-4096", 4032, 64, "1024/512 blocks: the width the 194.6 ms prize is booked at"),
    ("wave16-at-4096", 4080, 16, "256/128 blocks: the smallest wave still above the knee at 4096"),
]

# Which injections the SHIPPED kernel can even see. `NEG_M2_*` ids are unknown to it, so the shipped form
# is a control group for them: its output must not move at all.
M2_ONLY_NEG = ("maskwide", "partrow", "partalias")
# Injected inside `attn_prep`, on the cache. The CPU mirror does not model `attn_prep`, so there is no
# derived set for these; what is assertable on device is that they fire on BOTH forms, which is the whole
# point of them here (a shared stage must not launder a cache perturbation).
DEVICE_ONLY_NEG = ("slotk", "slotv")
# Not runnable on a device at all: two rows in one block sharing a state slot would race, and the winner
# is whichever warp landed last. The CPU mirror is the witness for that class.
MIRROR_ONLY_NEG = ("headcarry", "gqa_head0")
NEG_NAMES = ("readpast", "dropcurrent", "dimhalf", "gqaoff", "slotk", "slotv",
             "maskwide", "partrow", "partalias")


@dataclasses.dataclass
class Plan:
    tag: str
    path: Path
    case: dict
    lanes: list[int]
    split: int

    @property
    def cap(self) -> int:
        return int(self.case.get("cap", 1 << 30))

    @property
    def nrows(self) -> int:
        return len(self.lanes)

    def expect(self, neg: str) -> tuple[set, set]:
        """The derived must-catch / must-not-catch sets for this plan, from the mirror."""
        return M.neg_expect(neg, self.lanes, self.split, True, self.cap)


# ============================================================================================
# Plan building
# ============================================================================================
def emit(cases, tag: str) -> Path:
    """Write one plan file in the `attn_decode` plan format (12 fields, NROWS optional)."""
    indir = WORK / "in" / tag
    outdir = WORK / "out" / tag
    indir.mkdir(parents=True, exist_ok=True)
    lines = [R.write_case(c, indir, outdir / c["name"]) for c in cases]
    plan = WORK / f"plan-{tag}.txt"
    plan.write_text("\n".join(" ".join(f) for f in lines) + "\n")
    return plan


def mkplan(tag: str, case: dict, why: str = "") -> Plan:
    case.setdefault("nrows", 1)     # a decode case is one row; `write_case` defaults the same way
    lanes = [case["lvalid"] + r for r in range(case["nrows"])]
    split = max(1, T.windows_of(lanes[-1]))
    path = emit([case], tag)
    print(f"  {tag:<19} {len(lanes)} rows, lanes {lanes[0]}..{lanes[-1]}, split {split}  | {why}")
    return Plan(tag=tag, path=path, case=case, lanes=lanes, split=split)


def build_plans(fixture: str, tail_fixture: str, layer: int) -> list[Plan]:
    dl = Path("data/refs") / fixture
    specs = WAVE_SPECS + TIMING_SPECS
    plans = [mkplan(tag, M.wide_plan(dl, layer, start, nrows, f"m2.{tag}.L{layer:02d}"), why)
             for tag, start, nrows, why in specs]
    tail = []
    for step in [f"decode{i}" for i in range(8)]:
        try:
            c = R.decode_case(Path("data/refs") / tail_fixture, step, layer)
        except FileNotFoundError:
            continue
        c["name"] = f"m2.tail.{step}.L{layer:02d}"
        tail.append(c)
    if tail:
        # Single-row decode waves: the pair's SECOND row does not exist (`g.exists == false`), the one
        # route M = 2 adds that a full pair cannot exercise. Also the only rows above MAXSC on this box
        # that carry HF pins at all.
        plans.append(mkplan("tail", tail[0], "single row above MAXSC: no partner row, real HF pins"
                                            + (f" (+{len(tail) - 1} more steps)" if len(tail) > 1 else "")))
    return plans


# ============================================================================================
# Running the harness
# ============================================================================================
def build() -> bool:
    if BIN != HERE / "attn_m2":
        print(f"$ using --bin {BIN} (no build: the driver runs the binary it was given)")
        return BIN.exists()
    print("$ bash prototype/attn_m2/build.sh   (compile-only; no device is touched)")
    r = subprocess.run(["bash", str(HERE / "build.sh")], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        return False
    print("  " + r.stdout.strip())
    return True


def run_bin(plan: Plan, neg: str, dump: Path, echo: bool = True) -> tuple[int, str, str]:
    """One harness run over one plan. Returns (rc, stdout, stderr); a refusal is a FAILURE, not a skip."""
    dump.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "--plan", str(plan.path), "--neg", neg, "--split", str(plan.split),
           "--out", str(dump)]
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if echo:
        for line in r.stdout.splitlines():
            print("  | " + line)
    if r.stderr.strip():
        for line in r.stderr.strip().splitlines():
            print("  ! " + line)
    return r.returncode, r.stdout, r.stderr


def results_of(stdout: str) -> dict[str, dict]:
    """Parse the harness's per-case RESULT lines. Keyed by case name; empty means nothing launched."""
    out = {}
    for line in stdout.splitlines():
        if not line.startswith("RESULT case="):
            continue
        f = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        out[f["case"]] = f
    return out


def moved_heads(clean: Path, inj: Path, nrows: int) -> set[tuple[int, int]]:
    """(row, head) pairs whose BF16 output differs between two dumps of the SAME form."""
    a, b = clean.read_bytes(), inj.read_bytes()
    if len(a) != len(b):
        raise SystemExit(f"dump size mismatch {clean} {len(a)} vs {inj} {len(b)}")
    moved = set()
    for row in range(nrows):
        for head in range(N_Q):
            off = (row * N_Q + head) * DH * 2
            if a[off:off + HEAD_BYTES] != b[off:off + HEAD_BYTES]:
                moved.add((row, head))
    return moved


def rows_of(heads: set[tuple[int, int]]) -> set[int]:
    return {r for r, _ in heads}


# ============================================================================================
# Modes
# ============================================================================================
def negmatrix(plans: list[Plan]) -> int:
    """CPU-only: which plan can prove which injection. Derived, not typed — a hand-listed must-catch set
    rots into "everything" the moment the plan list changes."""
    head = f"{'injection':<12} {'kind':<17}" + "".join(f"{p.tag[:15]:>16}" for p in plans)
    print(head)
    print("-" * len(head))
    empty_any = []
    for neg in NEG_NAMES:
        cells, any_nonempty = [], False
        for p in plans:
            if neg in DEVICE_ONLY_NEG:
                cells.append("any")            # attn_prep-level: no mirror model, fires on any geometry
                any_nonempty = True
                continue
            must, _blind = p.expect(neg)
            any_nonempty = any_nonempty or bool(must)
            cells.append(f"{len(must)}")
        if not any_nonempty:
            empty_any.append(neg)
        print(f"{neg:<12} {M.NEG_KIND.get(neg, 'device-only'):<17}" + "".join(f"{c:>16}" for c in cells))
    print("\n(cell = size of the derived must-catch set in (row, head) pairs; `any` = no mirror model)")
    if empty_any:
        print(f"ASSERTION FAIL: {', '.join(empty_any)} has an empty derived must-catch set on EVERY "
              "plan — those controls cannot fail and must not be run")
        return 1
    print(f"\nnegmatrix: every one of the {len(NEG_NAMES)} device injections has a non-empty derived "
          f"must-catch set on at least one of the {len(plans)} plans.")
    return 0


def do_ab(plans: list[Plan]) -> int:
    """Bit-identity A/B, one verdict line per plan."""
    rc = 0
    for p in plans:
        dump = WORK / "dump" / "ab" / p.tag
        code, out, err = run_bin(p, "none", dump)
        res = results_of(out)
        if not res:
            print(f"VERDICT plan={p.tag} verdict=NO-LAUNCH (exit {code}) "
                  f"{err.strip().splitlines()[0] if err.strip() else 'no RESULT line'}")
            rc = 1
            continue
        bad = [c for c, f in res.items()
               if int(f["ndiff_aout"]) or int(f["ndiff_gated"])]
        for c, f in res.items():
            print(f"VERDICT plan={p.tag} case={c} rows={f['rows']} lanes={f['lanes']} "
                  f"split={f['split']} ndiff_aout={f['ndiff_aout']} ndiff_gated={f['ndiff_gated']} "
                  f"bytes_m2/ship={f['ratio']} verdict="
                  f"{'PASS' if c not in bad else 'FAIL'}")
        rc |= 1 if bad else 0
    print(f"\n{'FAIL' if rc else 'PASS'}: ab over {len(plans)} plan(s) — every row of every plan must "
          "be ndiff 0; NO-LAUNCH counts as a failure, never a skip.")
    return rc


def do_negative(plans: list[Plan], neg: str) -> int:
    """Injected-bug controls, per plan, against the derived sets. Two baselines, one verdict per plan."""
    if neg in MIRROR_ONLY_NEG:
        print(f"ASSERTION: --neg {neg} is not runnable on a device ({M.NEG_WHY.get(neg, '')[:120]}). "
              "Run it in the mirror: `m2_model.py negative`.")
        return 1
    # ---- 1. Plan selection, per injection, and the empty-set assertion BEFORE any device work. ----
    chosen, empty = [], []
    for p in plans:
        if neg in DEVICE_ONLY_NEG:
            chosen.append(p)
            continue
        must, blind = p.expect(neg)
        (chosen if must else empty).append(p)
    if not chosen:
        wide = max((max(p.lanes) for p in plans), default=0)
        print(f"ASSERTION FAIL: injection `{neg}` has an EMPTY derived must-catch set on all "
              f"{len(plans)} plans (widest row here: {wide} lanes), so it cannot fail and will not be "
              f"run. Why it needs a different geometry: {M.NEG_WHY.get(neg, 'n/a')}")
        return 1
    print(f"selection: `{neg}` runs on {len(chosen)}/{len(plans)} plan(s) — the ones whose derived "
          f"must-catch set is non-empty; skipped as structurally blind: "
          f"{[p.tag for p in empty] or 'none'}")

    rc = 0
    for p in chosen:
        must, blind = (set(), set()) if neg in DEVICE_ONLY_NEG else p.expect(neg)
        clean = WORK / "dump" / "clean" / p.tag
        inj = WORK / "dump" / f"neg-{neg}" / p.tag
        code_c, out_c, err_c = run_bin(p, "none", clean, echo=False)
        code_i, out_i, err_i = run_bin(p, neg, inj)
        res_c, res_i = results_of(out_c), results_of(out_i)
        if not res_i or not res_c:
            why = (err_i or err_c).strip().splitlines()
            print(f"NEGVERDICT plan={p.tag} neg={neg} verdict=NO-LAUNCH (exit {code_i or code_c}: "
                  f"{why[0] if why else 'no RESULT line'})")
            rc = 1
            continue
        # forms must still agree bit for bit under a shared injection (the §1.2 claim), and under an
        # M2-only injection too — the shipped form does not know that injection id.
        disagree = [c for c, f in res_i.items()
                    if int(f["ndiff_aout"]) or int(f["ndiff_gated"])]
        problems = []
        moved_m2: set = set()
        moved_ship: set = set()
        for cname in res_i:
            for form in ("ship", "m2"):
                diff = moved_heads(clean / f"{cname}.{form}.aout.bin", inj / f"{cname}.{form}.aout.bin",
                                   p.nrows)
                if form == "m2":
                    moved_m2 |= diff
                else:
                    moved_ship |= diff
        if neg in M2_ONLY_NEG:
            if moved_ship:
                problems.append(f"shipped form moved {len(moved_ship)} heads although it does not "
                                f"implement {neg}")
            missing = must - moved_m2
            hit_blind = blind & moved_m2
        elif neg in DEVICE_ONLY_NEG:
            # No mirror model for an `attn_prep`-level injection, so the assertion is that it fires on
            # both forms at all: a shared stage that swallowed a cache perturbation would show up here.
            if not moved_m2:
                problems.append("m2 form did not move: the shared stage laundered the cache perturbation")
            if not moved_ship:
                problems.append("shipped form did not move")
            hit_blind = set()
        else:                                   # form-neutral and self-independent: both forms, same set
            missing = must - moved_m2
            hit_blind = blind & moved_m2
            if moved_m2 != moved_ship:
                problems.append(f"the two forms moved differently: m2 rows {sorted(rows_of(moved_m2))} "
                                f"vs ship rows {sorted(rows_of(moved_ship))}")
        if disagree:
            problems.append(f"forms disagree under the injection: {disagree}")
        if missing:
            problems.append(f"must-catch rows that did not move: {sorted(rows_of(missing))}")
        if hit_blind:
            problems.append(f"derived-blind rows that DID move: {sorted(rows_of(hit_blind))}")
        print(f"NEGVERDICT plan={p.tag} neg={neg} rows={p.nrows} lanes={p.lanes[0]}-{p.lanes[-1]} "
              f"split={p.split} must={len(must)} moved_m2={len(moved_m2)} "
              f"moved_ship={len(moved_ship)} blind_hits={len(hit_blind)} "
              f"forms_agree={'no' if disagree else 'yes'} verdict={'FAIL' if problems else 'PASS'}"
              + (" | " + "; ".join(problems) if problems else ""))
        rc |= 1 if problems else 0
    print(f"\n{'FAIL' if rc else 'PASS'}: negative {neg} over {len(chosen)} plan(s). A PASS here means "
          "the injection moved every row the geometry says it must, moved no row the geometry says it "
          "cannot, and left the two forms bit-identical to each other.")
    return rc


def main() -> int:
    global BIN                                  # the driver runs whatever harness you point it at
    ap = argparse.ArgumentParser(description="device plan writer and A/B driver for attn_m2")
    ap.add_argument("what", choices=["plans", "negmatrix", "ab", "negative"])
    ap.add_argument("--fixture", default="p300",
                    help="fixture with per-layer prefill stage pins (p300: 277 rows, cycled to any "
                         "context length by wide_plan)")
    ap.add_argument("--tail-fixture", default="wide1024f",
                    help="fixture with pinned decode rows above 1024 lanes")
    ap.add_argument("--layer", type=int, default=R.ATTN_LAYERS[0])
    ap.add_argument("--neg", default="maskwide", help="injection for `negative`: " + "|".join(NEG_NAMES))
    ap.add_argument("--bin", default=None,
                    help="the harness binary to run. Pointing it at a stub is how the driver's own "
                         "selection/verdict logic gets checked without a device window")
    args = ap.parse_args()
    if args.bin:
        BIN = Path(args.bin)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.what == "negative" and args.neg not in NEG_NAMES:
        ap.error(f"--neg {args.neg} is not a device injection; the harness knows: " + "|".join(NEG_NAMES))
    plans = build_plans(args.fixture, args.tail_fixture, args.layer)
    if args.what == "plans":
        print("\nno device touched. HANDOFF.md numbers the runs; this driver does NOT self-lock.")
        return 0
    if args.what == "negmatrix":
        return negmatrix(plans)
    if not build():
        return 1
    if args.what == "ab":
        return do_ab(plans)
    return do_negative(plans, args.neg)


if __name__ == "__main__":
    sys.exit(main())
