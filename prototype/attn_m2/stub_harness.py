#!/usr/bin/env python3
"""stub_harness.py — a stand-in for `./attn_m2`, for testing THIS driver's verdict logic without a device.

`run.py selftest` runs the real driver end-to-end against this file through `--bin`. The thing under test
is the adjudication in `run.py:do_negative` — plan selection, the two baselines, the class scoping, the
exit code — and that is only visible end-to-end. Both defects this file exists to witness were verdict bugs:

  * a `moved` count taken form-vs-form, which is structurally blind for a form-neutral injection;
  * a `forms disagree` problem appended for EVERY class, which made the three m2-only injections
    (maskwide, partrow, partalias) impossible to PASS — the device run of 2026-10-16 scored a working
    `maskwide` control (`must=24 moved_m2=24 moved_ship=0 blind_hits=0`) as a FAILURE.

WHAT IT DOES. Same CLI as the harness (`--plan`, `--neg`, `--split`, `--out`), same `RESULT case=…` line,
same dump layout (`<case>.<form>.aout.bin`, [row][head][dim] BF16). Buffer content is a pure function of
(form, row, head, dim) and the instruction below, so a re-run with the same instruction writes the SAME
bytes and the driver's "this form, injected vs clean" diff sees exactly the movement asked for and no
more. `ndiff_aout` is computed the way `diff_rows` computes it — by comparing the two buffers just written.

WHAT IT DOES NOT MODEL, and that is on purpose. It has no attention, no lanes, no cache, no geometry: it
moves (row, head) blocks because it was TOLD to. It therefore cannot adjudicate whether a derived
must-catch set is correct — that is `m2_model.py negative`'s job, against the Python mirror. What it can
adjudicate is whether the driver accepts a legitimate PASS and rejects each way a PASS could be fake,
which is a property of the driver rather than of the sets.

    ATTN_M2_STUB="<key>=<value>[;<key>=<value>…]"      (all keys optional, defaults in brackets)
      m2=<none|all|comma-separated row indices>   rows whose m2-form output differs from clean   [none]
      ship=<none|all|…>                           same, for the shipped form                     [none]
      form_delta=<0|1>  the two forms' injected values differ, so the forms disagree under a SHARED
                        injection while both move the same rows. Without this knob no stub can ever
                        trip the retained agreement assertion, since a stub that moves both forms
                        identically leaves the two buffers equal                             [0]
      clean_diff=<0|1>  the m2 form's output differs from the shipped form's even with nothing
                        injected, i.e. §2's bit-identity gate is red and there is no control group [0]
      no_result=<0|1>   print no RESULT line at all, i.e. the launcher refused (§2's failure mode) [0]

The `bytesdemanded_*`/`ratio` fields are stubbed to 0 on purpose: no selftest reads them and a
plausible-looking fake byte census is worse than an obvious zero. `ndiff_gated` reports the same number as
`ndiff_aout` because this stub keeps one buffer per form and the driver's predicate ORs the two.

CPU-only: no HIP, no device, no gpu_lock. Exit status is 0 whenever it launches, like the real harness's
A/B path — the driver decides PASS/FAIL from the RESULT lines, not from the exit code.
"""
from __future__ import annotations

import array
import os
import sys
from pathlib import Path

N_Q, DH = 8, 256          # the constants the harness's dumps are laid out with


def instruction() -> dict[str, str]:
    kv: dict[str, str] = {}
    for part in os.environ.get("ATTN_M2_STUB", "").split(";"):
        if part.strip():
            key, _, val = part.partition("=")
            kv[key.strip()] = val.strip()
    return kv


def rows_selected(val: str, nrows: int) -> set[int]:
    if val in ("", "none"):
        return set()
    if val == "all":
        return set(range(nrows))
    return {int(x) for x in val.split(",") if x.strip() != ""}


def pattern(form: str, row: int, head: int, dim: int, instr: dict[str, str], moved: bool) -> int:
    """Any 16-bit pattern will do: the driver compares these BYTES and never reads them as numbers. Two
    rules make the stub usable — equal inputs mean equal outputs (so a re-run's diff is exactly the
    movement asked for), and `clean_diff`/`form_delta` are the only ways the two forms can differ."""
    bits = 0x3F80 ^ (row << 8) ^ (head << 3) ^ dim
    if instr.get("clean_diff") == "1" and form == "m2":
        bits ^= 0x0040                       # clean run: the forms already differ -> no control group
    if moved:
        bits ^= 0x0002 if (instr.get("form_delta") == "1" and form == "m2") else 0x0001
    return bits & 0xFFFF


def write_form(path: Path, nrows: int, form: str, moved: set[int], instr: dict[str, str]) -> array.array:
    buf = array.array("H")
    for row in range(nrows):
        for head in range(N_Q):
            hit = row in moved
            buf.extend(pattern(form, row, head, dim, instr, hit) for dim in range(DH))
    path.write_bytes(buf.tobytes())
    return buf


def main() -> int:
    argv = sys.argv[1:]
    plan, outdir, split, neg = None, ".", "1", "none"
    for i, a in enumerate(argv):
        if a == "--plan":
            plan = argv[i + 1]
        elif a == "--out":
            outdir = argv[i + 1]
        elif a == "--split":
            split = argv[i + 1]
        elif a == "--neg":
            neg = argv[i + 1]
        elif a in ("--tile", "--rows", "--burst", "--reps", "--geom-guard"):
            pass                                   # accepted and ignored, like a knob the stub lacks
        elif a.startswith("--"):
            print(f"stub: ignoring unrecognised argument {a}", file=sys.stderr)
    if plan is None:
        print("stub: --plan required", file=sys.stderr)
        return 2
    instr = instruction()
    if instr.get("no_result") == "1":
        print("stub: refusing to launch; no RESULT line printed")
        return 0

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"stub plan {plan} | neg={neg} | split={split} | ATTN_M2_STUB="
          f"{os.environ.get('ATTN_M2_STUB', '')}")
    for line in Path(plan).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        name, lvalid, nrows = fields[0], int(fields[9]), int(fields[11]) if len(fields) > 11 else 1
        # The movement instruction belongs to the INJECTED run only. The driver runs `--neg none` first
        # to get each form's own clean baseline, and a stub that perturbed both runs identically would
        # report "nothing moved" no matter what it was told to do.
        if neg == "none":
            moved = {"ship": set(), "m2": set()}
        else:
            moved = {"ship": rows_selected(instr.get("ship", "none"), nrows),
                     "m2": rows_selected(instr.get("m2", "none"), nrows)}
        bufs = {form: write_form(out / f"{name}.{form}.aout.bin", nrows, form, moved[form], instr)
                for form in ("ship", "m2")}
        ndiff = sum(1 for x, y in zip(bufs["ship"], bufs["m2"]) if x != y)
        print(f"STUBCASE case={name} rows={nrows} ndiff={ndiff} of {len(bufs['ship'])} elems"
              f" moved_m2_rows={sorted(moved['m2'])} moved_ship_rows={sorted(moved['ship'])}")
        print(f"RESULT case={name} rows={nrows} lanes={lvalid}-{lvalid + nrows - 1} split={split}"
              f" ndiff_aout={ndiff} ndiff_gated={ndiff}"
              f" bytesdemanded_ship=0 bytesdemanded_m2=0 ratio=0.0000 stub=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
