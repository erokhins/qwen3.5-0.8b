#!/usr/bin/env python3
"""gate_m16.py — bit-identity gate for the M = 16 re-tile (`attn_m16.hip.cpp`).

THE CLAIM, IN ONE LINE: for every row of every case, the M = 16 form's `aout` and gated output differ
from the form the engine ships (`attn_slim1_wave`) in ZERO 16-bit patterns. Not "within a band": the
harness compares raw `uint16` patterns per row and this gate requires the count to be 0. A re-tile that
shares a walk but keeps private reductions has no licence to differ at all, and the second the number is
allowed to be non-zero the gate stops being a gate.

WHAT IS CHECKED, AND WHY EACH PIECE IS NEEDED

  * `identity`  — the spans the item is named after: a 16-row tile holding 1, 3 and 16 query rows, at
                  1024 and 4096 lanes, plus the 0-row tile (a padded grid: `--pad-tiles 1`, whose empty
                  groups are the route no fixture would otherwise reach).
  * `control`   — the harness also prints `shipslim` vs the frozen tiled form `attn_tile_wave`. That pair
                  is not this item's claim; it is the witness that the REFERENCE side of the first pair
                  is sound. A check whose reference is derived from the thing under test cannot adjudicate
                  it (`prototype/attn_tile_model.py compare`, and the §18 `--scope form` row).
  * `inject`    — two injections, one that must NOT move the diff and one that MUST:
                    `dimhalf` halves the dot on BOTH forms, so ndiff must stay 0 — the positive witness
                      that this form's `nd` dispatch folds the same way;
                    `slotk` corrupts the SHIPPED form's self-lane read only, so ndiff must become
                      non-zero — the witness that this form's self-lane path is live and reads its own row
                      (every streaming row attends its own appended lane by construction, so the must-catch
                      set is derived from the geometry, not hand-listed: see the trap row).
  * `refuse`    — a one-pass row (lanes <= MAXSC with `full = 1`) must be REFUSED, with the reason naming
                  the one-pass staging rule. A refusal for the wrong reason is a different bug wearing a
                  passing result.

NO SELF-Lock, NO TIMING. Like `prototype/attn_m2/run.py`: three gate drivers in this repo acquire
`tools/gpu_lock.sh` themselves and a caller that holds the lock and runs one deadlocks it silently, so the
CALLER holds the lock around this. This script never passes `--burst`, prints no `ms`, and takes no
best-of-N: a worker's device run is correctness-only (AGENTS.md).

USAGE
    .venv/bin/python prototype/attn_m2/gate_m16.py cpu              # host-only, no device, no lock
    .venv/bin/python prototype/attn_m2/gate_m16.py device           # the device gate (caller holds the lock)
    .venv/bin/python prototype/attn_m2/gate_m16.py plans            # write the plans, run nothing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np                             # noqa: E402  only for comparing two runs' dumps

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "attn_decode")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reference as R                    # noqa: E402
import attn_tile_model as T              # noqa: E402  MAXSC, windows_of
import m2_model as M                     # noqa: E402  wide_plan: the case builder this gate reuses

HERE = Path(__file__).resolve().parent
WORK = Path("/tmp/attn_m16_gate")       # scratch outside the tree, like every other op here
BIN = HERE / "attn_m16"

# (kind, tag, lvalid, nrows, full, extra). `full = 0` forces every row to stream, which is the only way
# a row at or below MAXSC can reach this form at all — the form refuses one-pass rows (the header, and
# the `refuse` kind below proves the refusal is live).
CASES = [
    ("identity", "span01-1024",  1020,  1, 1, []),
    ("identity", "span03-1024",  1020,  3, 1, []),
    ("identity", "span16-1024",  1020, 16, 1, []),
    ("identity", "span16-4096",  4080, 16, 1, []),
    ("identity", "span01-4096",  4095,  1, 1, []),
    # The 0-row span: a 16-row tile with no live row. No real wave produces one, so the grid is padded
    # on purpose; without this case the empty-group route is asserted only on the host.
    ("identity", "span00-pad",   1020,  1, 1, ["--pad-tiles", "1"]),
    # The geometry the burst times, gated: a timing on a shape this gate did not check is not the shape
    # the burst's number describes.
    ("identity", "wave64-at-4096", 4032, 64, 1, []),
    ("inject",   "span16-4096",  4080, 16, 1, []),
    ("refuse",   "pair-at-maxsc", 511,  4, 1, []),
    # The widths `prototype/attn_m2` timed, so the tax is read off the same harness at the same widths.
    # `burst` cases are written by `plans` and run by `burst_m16.sh`; `gate_m16.py device` does not run
    # them (it is a numerics gate, not a timing), except wave64-at-4096, which is ALSO an identity case
    # above so the shape the headline burst times is a shape this gate bit-checked.
    ("burst", "wave64-at-4096",   4032, 64, 1, []),
    ("burst", "wave64-at-2048",   2044, 64, 1, []),
    ("burst", "wave64-at-1024",   1020, 64, 1, []),
    ("burst", "wave16-at-4096",   4080, 16, 1, []),
    ("burst", "span16-4096",      4080, 16, 1, []),
    ("burst", "span16-1024",      1020, 16, 1, []),
    # Short rows, streaming-forced: under `full = 1` these are one-pass and this form refuses them, so a
    # number here is the STREAMING route at a short width, not the shipping short-row route. The label is
    # printed by the harness's own `full=0` field; burst_m16.sh says it in words too.
    ("burst", "pair-at-maxsc",     511,  4, 0, []),
    ("burst", "short-64",           64, 64, 0, []),
]


def split_of(lvalid: int, nrows: int) -> int:
    return T.windows_of(lvalid + nrows - 1)


def emit_plan(tag: str, cases: list, dl: Path, layer: int) -> Path:
    """Write one plan file in the `attn_decode` 12-field format, reusing the shipped case builder."""
    indir = WORK / "in" / tag
    outdir = WORK / "out" / tag
    indir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []
    for kind, name, lvalid, nrows, full, extra in cases:
        c = M.wide_plan(dl, layer, lvalid - 1, nrows, name)
        lines.append(" ".join(R.write_case(c, indir, outdir / name)))
    path = WORK / f"plan-{tag}.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def run(plan: Path, split: int, full: int, neg: str = "none", extra: list[str] | None = None,
        dump: Path | None = None):
    out = dump if dump is not None else WORK / "dump"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "--plan", str(plan), "--neg", neg, "--split", str(split),
           "--full", str(full), "--out", str(out)] + (extra or [])
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        print("  | " + line)
    return r


def results_of(stdout: str) -> dict[str, dict]:
    """Parse `RESULT case=...` lines. An empty dict means nothing launched, which is never a PASS."""
    out = {}
    for line in stdout.splitlines():
        if not line.startswith("RESULT case=") or " mode=" in line:
            continue
        fields = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = v
        out[fields.get("case", "?")] = fields
    return out


def moved_elems(dump_a: Path, dump_b: Path, name: str, form: str) -> int:
    """How many 16-bit patterns of one form's output differ between two runs of the SAME plan.

    This is the baseline `prototype/attn_m2/run.py` learned it needs: a form-neutral injection makes the
    two forms AGREE, so "the forms match" cannot show the injection fired. Each form has to be compared
    against its own clean output, or a control that moves nothing scores as a pass.
    """
    fa, fb = dump_a / f"{name}.{form}.aout.bin", dump_b / f"{name}.{form}.aout.bin"
    if not (fa.exists() and fb.exists()):
        return -1
    a = np.frombuffer(fa.read_bytes(), dtype=np.uint16)
    b = np.frombuffer(fb.read_bytes(), dtype=np.uint16)
    return -1 if a.shape != b.shape else int((a != b).sum())


class Gate:
    def __init__(self) -> None:
        self.n = 0
        self.bad = 0

    def ok(self, what: str, cond: bool, detail: str = "") -> None:
        self.n += 1
        if cond:
            print(f"  ok   {what}")
        else:
            self.bad += 1
            print(f"  FAIL {what}\n{detail}")


def adjudicate_identity(g: Gate, res: dict, plan_cases: list, pad: bool) -> None:
    g.ok(f"every case launched ({len(res)}/{len(plan_cases)})", len(res) == len(plan_cases),
         "a gate driver whose parser matches nothing reports PASS (prototype/gdn_collapse/run.py). "
         f"Parsed {sorted(res)}, expected {sorted(c[1] for c in plan_cases)}.")
    if len(res) != len(plan_cases):
        return
    for kind, name, lvalid, nrows, full, extra in plan_cases:
        r = res[name]
        for field in ("ndiff_aout_m16", "ndiff_gated_m16"):
            g.ok(f"{name}: {field} == 0 (bit-identity vs attn_slim1_wave)",
                 r.get(field) == "0",
                 f"read {r.get(field)} at rows={r.get('rows')} lanes={r.get('lanes')} "
                 f"widest_round={r.get('widest_round')}. Non-zero means the re-tile changed a number; "
                 "the band is zero, not wide.")
        g.ok(f"{name}: control pair ndiff_aout_shipslim == 0 (shipslim vs the frozen attn_tile_wave)",
             r.get("ndiff_aout_shipslim") == "0",
             f"read {r.get('ndiff_aout_shipslim')}. If the reference of the gate does not agree with the "
             "frozen shipped form, a 0 on the first pair is meaningless.")
        if pad:
            g.ok(f"{name}: the padded grid ran empty groups (route reached, not skipped)",
                 int(r.get("empty_groups", "0")) >= 4,
                 f"empty_groups={r.get('empty_groups')}: --pad-tiles must actually launch a tile with no "
                 "live row, or this case proves nothing about the empty-group route.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cpu", "plans", "device"])
    ap.add_argument("--fixture", default="p300")
    ap.add_argument("--layer", type=int, default=R.ATTN_LAYERS[0])
    ap.add_argument("--expect-checks", type=int, default=0)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "dump").mkdir(parents=True, exist_ok=True)   # the harness's --out; a missing dir would make
    # every dump_file() a silent no-op, and a diff read from an empty file is not a witness
    dl = ROOT / "data" / "refs" / args.fixture
    if not dl.is_dir():
        print(f"FAIL: fixture {dl} does not exist. Enumerate data/refs/*/ rather than typing a name "
              "(the admission-over-a-typed-list trap).")
        return 1
    g = Gate()

    # ---- host-only part: runs in every mode, needs no device and no lock --------------------------
    print("== host-only: the geometry guard (no HIP call, no device, no lock)")
    r = subprocess.run([str(BIN), "--geom-guard"], capture_output=True, text=True)
    print("\n".join("  | " + x for x in r.stdout.splitlines()[-3:]))
    g.ok("--geom-guard exits 0 (bound / phantom / leak / real-plan / cover assertions)",
         r.returncode == 0, f"rc={r.returncode}\n{r.stdout[-2000:]}{r.stderr[-800:]}")
    g.ok("--geom-guard reported the walk it did (the line exists, so the count is not an empty parse)",
         "geometries walked" in r.stdout and "real plan geometries" in r.stdout, r.stdout[-500:])

    print("== host-only: transcription of the shipped form")
    r = subprocess.run([sys.executable, str(HERE / "check_m16_transcription.py")],
                       capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    print("  | " + tail)
    g.ok("check_m16_transcription.py passes (expressions transcribed, not re-spelled)",
         r.returncode == 0, r.stdout[-2000:] + r.stderr[-500:])

    if args.what == "cpu":
        print("\nno device touched. The device half is `gate_m16.py device`; the caller holds "
              "tools/gpu_lock.sh around it and this driver does not self-lock.")
        return finish(g, args)

    if not BIN.exists():
        print(f"FAIL: {BIN} not built. bash prototype/attn_m2/build_m16.sh")
        return 1

    if args.what == "plans":
        # Write every plan and print the runs, so a caller with the lock can execute them verbatim. This
        # is the hand-over shape: the plans exist, the commands are exact, no device was touched.
        print("== plans written (no device touched)")
        for case in CASES:
            kind, name, lvalid, nrows, full, extra = case
            plan = emit_plan(name, [case], dl, args.layer)
            # Machine-readable on purpose: `burst_m16.sh` reads these lines instead of re-typing a split,
            # so the split authority stays `split_for_lanes` in one file.
            print(f"PLAN {name} {plan} {split_of(lvalid, nrows)} {full} {kind} {' '.join(extra)}")
        print("  device commands: see the header of this file, or run `gate_m16.py device` under the lock")
        return finish(g, args)

    # ---- device part ------------------------------------------------------------------------------
    t0 = time.time()
    refuse = [c for c in CASES if c[0] == "refuse"]

    print("== device: the identity spans (spans 0/1/3/16 rows in a 16-row tile, + the burst's wave)")
    # ONE PLAN PER CASE, because `split` is part of the geometry under test: it decides how many lane
    # blocks a row is cut into, which decides which rows straddle a block boundary. One plan for the whole
    # set would force every case to the widest `split` in the set and silently test a different geometry
    # than the engine launches.
    for case in [c for c in CASES if c[0] in ("identity")]:
        kind, name, lvalid, nrows, full, extra = case
        plan = emit_plan(name, [case], dl, args.layer)
        res = results_of(run(plan, split_of(lvalid, nrows), full, extra=extra).stdout)
        adjudicate_identity(g, res, [case], bool(extra))

    print("== device: injections. Both are form-neutral, so the assertion is agreement PLUS evidence the "
          "injection was not inert, measured against each form's own clean run")
    inj = [c for c in CASES if c[0] == "inject"][0]
    kind, name, lvalid, nrows, full, extra = inj
    plan = emit_plan("inject", [inj], dl, args.layer)
    split = split_of(lvalid, nrows)
    run(plan, split, full, neg="none", dump=WORK / "d_clean")
    res = results_of(run(plan, split, full, neg="dimhalf", dump=WORK / "d_dimhalf").stdout)
    r = res.get(name, {})
    g.ok(f"dimhalf on {name}: ndiff_aout_m16 == 0 (both forms fold 4 dims the same way)",
         r.get("ndiff_aout_m16") == "0",
         f"read {r.get('ndiff_aout_m16')}; if this form's `nd` dispatch disagreed with the shipped slim "
         "form's, this is where it shows. Empty means nothing launched.")
    mv = moved_elems(WORK / "d_clean", WORK / "d_dimhalf", name, "m16")
    g.ok(f"dimhalf moved {name}/m16 away from its own clean run (the nd dispatch is reached, not dead)",
         mv > 0,
         f"moved={mv} elements. If the `nd == 4` copy of the score loop were unreachable, agreement would "
         "still read 0 for the wrong reason.")
    res = results_of(run(plan, split, full, neg="slotk", dump=WORK / "d_slotk").stdout)
    r = res.get(name, {})
    g.ok(f"slotk on {name}: ndiff_aout_m16 == 0 (a form-neutral cache injection must move both forms "
         "alike, so agreement is the claim here — see why below)",
         r.get("ndiff_aout_m16") == "0",
         f"read {r.get('ndiff_aout_m16')}. `slotk` corrupts the CACHE slot `attn_prep` appends to "
         "(slot_of, in prototype/attn_decode/attn_decode_kernels.h), which every form reads the same way, "
         "so a non-zero diff here would mean THIS form reads the appended lane differently.")
    for form in ("m16", "shipslim", "ship"):
        mv = moved_elems(WORK / "d_clean", WORK / "d_slotk", name, form)
        g.ok(f"slotk moved {form}'s output away from its OWN clean run (the injection is not inert)",
             mv > 0,
             f"moved={mv} elements. A form-neutral injection scored against the other form always reads "
             "0 and proves nothing (prototype/attn_m2/run.py, defect 2); -1 means a dump is missing, "
             "which is a failure too.")

    print("== device: the one-pass refusal must be loud, and for the right reason")
    kind, name, lvalid, nrows, full, extra = refuse[0]
    plan = emit_plan("refuse", refuse, dl, args.layer)
    r = run(plan, split_of(lvalid, nrows), 1)
    g.ok(f"{name} at full=1: REFUSED with exit 2, reason naming the one-pass staging",
         r.returncode == 2 and "REFUSED" in (r.stdout + r.stderr)
         and "one-pass" in (r.stdout + r.stderr),
         f"rc={r.returncode}. A refusal for the wrong reason (or an accepted one-pass row, which would "
         "score the whole slice into an LDS window that cannot hold it) both read as a red run here and "
         "must be told apart by the reason string.")
    res = results_of(run(plan, split_of(lvalid, nrows), 0).stdout)
    g.ok(f"{name} at full=0: launched and bit-identical (the streaming-forced route the burst uses)",
         res.get(name, {}).get("ndiff_aout_m16") == "0",
         f"read {res.get(name, {}).get('ndiff_aout_m16')}; empty means the harness refused or crashed.")

    print(f"\nwall {time.time() - t0:.1f} s for the device half. NO TIMING TAKEN: no burst, no best-of-N, "
          "no ms figure; the wall above is this driver's own clock, not a kernel measurement.")
    return finish(g, args)


def finish(g: Gate, args) -> int:
    print(f"\n{'FAIL' if g.bad else 'PASS'}: gate_m16 — {g.n - g.bad}/{g.n} assertions")
    if args.expect_checks and g.n != args.expect_checks:
        print(f"FAIL: expected {args.expect_checks} assertions, ran {g.n} — a check list that shrinks "
              "silently cannot fail")
        return 1
    return 1 if g.bad else 0


if __name__ == "__main__":
    sys.exit(main())
