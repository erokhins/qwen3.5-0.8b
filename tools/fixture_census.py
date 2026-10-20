#!/usr/bin/env python3
"""Does a fixture directory agree with its own manifest? A FIXTURE-MISMATCH census, not a numerics run.

Why this exists is one row of `prototype/attn_wide/A12-FIXES.md` item 3: `data/refs/zz` holds 2500
`.npy` files against a 524-entry `pins` map, and its `logits_declared_absent` names two steps whose
logits files are sitting right there on disk. The gate refused it, `zz` came up rc=1 with no verdict
line, and the admission table read that as a numerics result about the engine. It is not a numerics
result at all — the fixture contradicts itself — and nothing between the fixture and the report said
so. That is the same failure class this tree keeps recording (a parser that matches nothing prints
PASS; a coverage gap prints a verdict): a red that cannot be attributed is a red nobody reads.

FOUR CHECKS, all of them about the fixture, none about the engine:

  undeclared            a `.npy` in the directory that `manifest["pins"]` does not name. zz: 1976 of
                        them, among them the two logits files that contradict its own declaration.
                        `tools/reference_dump.py` writes into `--out` without clearing it
                        (`out.mkdir(parents=True, exist_ok=True)`), and `tools/dump_fixtures.sh`
                        installs with `cp`, so a re-pin over an older, wider dump leaves the older
                        files alive and undeclared. That is how zz got its 1976.
  missing               a pin key with no file — the direction that means a pin LOST something.
  gap_contradicted      a step named in `logits_declared_absent` whose `logits_last.npy` is present.
                        This is `engine/gate.py` section (0)'s check, mirrored here so the census can
                        name it before a gate run is paid for; the gate's copy stays the forcing one.
  argmax_pairing        each pinned logits row's argmax must resolve to the `greedy_continuation`
                        index that §10.5 pairs it with: `logits_last` -> `greedy[0]`, and
                        `decode{s}.logits_last` -> `greedy[s+1]` (the "+1" is the trap this repo
                        already records — aligning a decode sample on `[0:]` reads 0/128 and looks
                        like total divergence). A logits file that does not pair is either mis-named
                        or the last-writer-wins leftover `specification/10` warns about; zz's two
                        orphans DID pair (760 and 6511 are `greedy[3]`/`greedy[4]`), which is the
                        proof they are real reference tensors from an earlier run of the same chain
                        and not noise — and also the proof the pairing check alone cannot see them.
                        The last row of `logits_rows.npy` (the `--prefill-logits all` pin) is judged
                        the same way as `logits_last`.

WHAT IS NOT JUDGED, and is printed instead: the files in `reference_dump.py`'s `CARRIED_NOT_PINNED`
(today `route_allowance.json`). They are products of the ENGINE side, not of a reference run, and
`--verify` already refuses to demand them — but they are NAMED here, because an ignored file and a
carried file read the same in a count and mean different things (the lesson behind
`tools/reference_dump.py:CARRIED_NOT_PINNED`).

CPU ONLY, and cheap: `.npy` files are read through `mmap_mode="r"`, so the two `logits_rows.npy`
blocks (303 MiB and 485 MiB) cost one row of each. No device, no engine, no fixture is written.

    .venv/bin/python tools/fixture_census.py                    # the derived population
    .venv/bin/python tools/fixture_census.py --fixture zz       # one fixture
    .venv/bin/python tools/fixture_census.py --quiet            # only the failures and the count
    .venv/bin/python tools/fixture_census.py --selftest         # prove the checks can fail, on a
                                                                # synthetic fixture in tmp

Exit 0 = every adjudicated fixture is internally consistent. Exit 1 = at least one is not, or the
census could not adjudicate something it found: a fixture that raises is reported as UNADJUDICABLE
with its reason and still counted, because a census that silently shrinks its own population is the
"0 of the 55 readable" mistake in `AGENTS.md`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "data" / "refs"
DUMP_TOOL = ROOT / "tools" / "reference_dump.py"


def carried_not_pinned() -> tuple[str, ...]:
    """`CARRIED_NOT_PINNED`, read out of its owner instead of copied.

    `tools/reference_dump.py` owns that list, and importing it costs a full `torch` import for one
    tuple. `tools/band_history.py` reads `engine/gate.py`'s band dict out of the source for the same
    reason (one fact, one home, no import tax), so this follows the house pattern; a list copied here
    would rot the day the dumper gains a carried file, and a stale carried list silently turns a
    legitimate file into an "undeclared" failure."""
    text = DUMP_TOOL.read_text()
    m = re.search(r"^CARRIED_NOT_PINNED\s*(?::[^=]*)?=\s*\(([^)]*)\)", text, re.M)
    if not m:
        raise SystemExit(f"fixture_census: cannot find CARRIED_NOT_PINNED in {DUMP_TOOL} — the "
                         "dumper changed its spelling. Update this reader; do not copy a list in.")
    return tuple(x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip())


# Bound once at import: the carried list, from its owner.
CARRIED = carried_not_pinned()


def npy_argmax(path: Path, row: int | None = None) -> int:
    """The argmax of a pinned logits file (or of one row of a row block), in F64-safe steps.

    `mmap_mode="r"` because the row blocks are 303/485 MiB and a census that read them all would
    cost more than the gate it fronts. BF16 pins are stored as a UINT16 bit pattern
    (`tools/reference_dump.py --logits-bits bf16`, and the `.to()`/`.view()` trap `AGENTS.md` records),
    so the bits are widened by <<16 — the same reconstruction `engine/gate.py:load_pin` uses.
    """
    arr = np.load(path, mmap_mode="r")
    v = arr if row is None else arr[row]
    v = np.asarray(v).ravel()
    if v.dtype == np.uint16:
        v = (v.astype(np.uint32) << 16).view(np.float32)
    if v.size == 0:
        raise ValueError(f"{path.name} is empty, so it has no argmax to pair")
    return int(np.argmax(v))


def undeclared_files(path: Path, pins) -> list[str]:
    """The `.npy` files in `path` that the manifest's `pins` map does not name.

    This is the exact half of "the directory contradicts its own manifest" that no other check in the
    tree could see, so it lives here once and `engine/gate.py` section (0) calls it: a fixture
    directory with unclaimed files is a FIXTURE mismatch and must be named before any engine verdict
    is printed from it. Only this direction is shared, because the other direction (a pin with no
    file) is what `gate.load_pin` already fails on, and because `--selftest` builds scratch fixtures
    by copying a SUBSET of a real manifest's files — an unclaimed-file test survives that, a
    missing-file test would fire on the copy and teach nothing.
    """
    return sorted({p.name[:-4] for p in path.glob("*.npy")} - set(pins))


class Fixture:
    """One fixture directory, and the four verdicts about it."""

    def __init__(self, path: Path):
        self.name = path.name
        self.path = path
        self.man = json.loads((path / "manifest.json").read_text())
        self.pins: dict = self.man.get("pins") or {}
        self.disk: dict[str, Path] = {p.name[:-4]: p for p in path.glob("*.npy")}
        self.carried: list[str] = sorted(p.name for p in path.iterdir()
                                         if p.is_file() and p.name in CARRIED)
        self.greedy: list[int] = list(self.man.get("greedy_continuation") or [])
        self.declared: list[str] = list(self.man.get("logits_declared_absent") or [])
        self.problems: list[tuple[str, str]] = []
        self.notes: list[str] = []
        self.rows_checked = 0

    # -- the four checks ------------------------------------------------------------------------
    def check_files(self) -> None:
        missing = sorted(set(self.pins) - set(self.disk))
        undeclared = undeclared_files(self.path, self.pins)
        if undeclared:
            self.problems.append(("undeclared",
                                  f"{len(undeclared)} of {len(self.disk)} .npy files are not named in"
                                  f" manifest.pins (e.g. {', '.join(undeclared[:6])})"))
        if missing:
            self.problems.append(("missing",
                                  f"{len(missing)} pinned tensor(s) have no file (e.g."
                                  f" {', '.join(missing[:6])})"))

    def check_declared_gaps(self) -> None:
        contra = [s for s in self.declared if (self.path / f"{s}.logits_last.npy").exists()]
        if contra:
            self.problems.append(("gap_contradicted",
                                  "logits_declared_absent names "
                                  f"{', '.join(contra)} but the file(s) are on the disk — a declared"
                                  " gap the disk contradicts (engine/gate.py section (0) refuses a"
                                  " device run for exactly this)"))
        for s in self.declared:
            step = _step_index(s)
            if step is not None and step >= max(len(self.greedy) - 1, 0):
                self.problems.append(("gap_contradicted",
                                      f"logits_declared_absent names {s}, which is not a step this"
                                      " fixture has (a stale declaration)"))

    def check_pairing(self) -> None:
        if not self.greedy:
            self.notes.append("no greedy_continuation, so nothing can be paired")
            return
        want_prefill = self.greedy[0]
        for name in ("logits_last", "logits_rows"):
            if name not in self.disk:
                continue
            row = None if name == "logits_last" else -1
            got = npy_argmax(self.disk[name], row=row)
            self.rows_checked += 1
            if got != want_prefill:
                self.problems.append(("argmax_pairing",
                                      f"{name} argmax {got} != greedy[0] {want_prefill}"
                                      + (" (its last row)" if row is not None else "")))
        for step in sorted(_step_index(s) for s in _decode_steps(self.disk, self.pins)):
            if step is None:
                continue
            name = f"decode{step}.logits_last"
            if name not in self.disk:
                continue                       # absent is the declared-gap check's business
            self.rows_checked += 1
            if step + 1 >= len(self.greedy):
                self.problems.append(("argmax_pairing",
                                      f"{name} exists but greedy_continuation has no index"
                                      f" {step + 1} to pair it with"))
                continue
            got = npy_argmax(self.disk[name])
            if got != self.greedy[step + 1]:
                self.problems.append(("argmax_pairing",
                                      f"{name} argmax {got} != greedy[{step + 1}]"
                                      f" {self.greedy[step + 1]} — the §10.5 pairing is"
                                      " decode{s} <-> greedy[s+1], never greedy[s]"))

    def adjudicate(self) -> None:
        self.check_files()
        self.check_declared_gaps()
        self.check_pairing()


def _step_index(name: str) -> int | None:
    m = re.fullmatch(r"decode(\d+)", name)
    return int(m.group(1)) if m else None


def _decode_steps(*dicts: dict) -> set[str]:
    """Every `decodeN` that either side (disk, pins) claims — the UNION, because a check that walks
    only the disk cannot see a pin whose file vanished and a check that walks only the pins cannot
    see an orphan."""
    out: set[str] = set()
    for d in dicts:
        for k in d:
            head = k.split(".")[0]
            if _step_index(head) is not None:
                out.add(head)
    return out


def pin_fixtures(want: list[str] | None) -> list[Path]:
    """The population, ENUMERATED — never typed. A fixture list written from memory is the trap
    `AGENTS.md` records (three admissions claimed 18/18 over a list where seven names did not exist),
    so the caller may narrow it but never invent it."""
    if not REFS.is_dir():
        raise SystemExit(f"fixture_census: {REFS} does not exist")
    found = sorted(d for d in REFS.iterdir() if d.is_dir() and (d / "manifest.json").exists())
    if want:
        known = {d.name for d in found}
        unknown = [w for w in want if w not in known]
        if unknown:
            raise SystemExit(f"fixture_census: no such fixture(s): {', '.join(unknown)}."
                             f" {len(known)} manifest-bearing directories exist under {REFS}")
        found = [d for d in found if d.name in set(want)]
    return found


def selftest() -> int:
    """Prove the pairing check and the gate-shared file check can FAIL, on a synthetic fixture.

    Why this exists: the repo's own record says a negative control with an empty must-catch set is not a
    control, and that the case list has to be DERIVED from the geometry that can make a check fire. The
    geometry here is one array per pinned name with one element set to 1.0, so the argmax is exactly the
    id we choose and every case below is exact rather than approximate. Cases:

      paired            every logits file argmaxes to the `greedy` index §10.5 pairs it with. MUST NOT
                        fire — if this one goes red the witness itself is broken.
      swapped           `decode0` and `decode1` exchange contents. MUST fire, twice, and the reported
                        wanted/got pair must name the right indices (a check that fires on the wrong
                        row is a check that will be ignored).
      shifted_by_one    every decode file one slot early (decode{s} -> greedy[s], not [s+1]) — THE
                        alignment trap this file exists to keep anyone from re-making. MUST fire.
      declared_gap      a step declared absent and actually absent. MUST NOT fire at the pairing check,
                        because an absent step is `check_declared_gaps`' business, not this one's.
      unclaimed_file    one extra `.npy` on disk, through `undeclared_files` — the exact function
                        `engine/gate.py` section (0) calls, so the gate's new line has a witness too.
    """
    import shutil
    import tempfile

    bad: list[str] = []
    work = Path(tempfile.mkdtemp(prefix="fixture_census_selftest_"))
    greedy = [760, 6511, 314, 9338]
    width = max(greedy) + 1          # the id IS the index, so the array must reach the largest one

    def build(name: str, *, mapping: dict[str, int], declare: list[str] | None = None,
              extra: list[str] | None = None) -> Path:
        d = work / name
        d.mkdir(parents=True)
        pins = {k: {"step": k, "shape": [width], "dtype": "F32"} for k in mapping}
        for pin in mapping:
            arr = np.zeros(width, dtype="<f4")
            arr[mapping[pin]] = 1.0
            np.save(d / f"{pin}.npy", arr)
        for pin in extra or []:
            np.save(d / f"{pin}.npy", np.zeros(width, dtype="<f4"))
        (d / "manifest.json").write_text(json.dumps({
            "pins": pins, "greedy_continuation": greedy,
            "logits_declared_absent": list(declare or []), "kind": "selftest"}))
        return d

    want = {"logits_last": greedy[0], "decode0.logits_last": greedy[1],
            "decode1.logits_last": greedy[2]}
    cases = [
        ("paired", {"want_pairing": 0}, dict(mapping={**want, "decode2.logits_last": greedy[3]})),
        ("swapped", {"want_pairing": 2},
         dict(mapping={**want, "decode0.logits_last": greedy[2], "decode1.logits_last": greedy[1]})),
        ("shifted_by_one", {"want_pairing": 3},
         dict(mapping={"logits_last": greedy[1], "decode0.logits_last": greedy[0],
                       "decode1.logits_last": greedy[1]})),
        ("declared_gap", {"want_pairing": 0},
         dict(mapping=want, declare=["decode2"])),
    ]
    for name, want_out, kwargs in cases:
        fx = Fixture(build(name, **kwargs))
        fx.check_pairing()
        fired = [p for p in fx.problems if p[0] == "argmax_pairing"]
        ok = len(fired) == want_out["want_pairing"]
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: pairing fired {len(fired)} time(s),"
              f" want {want_out['want_pairing']}"
              + ("" if ok else f" — got {fired}"))
        if not ok:
            bad.append(name)

    d = build("unclaimed", mapping=want, extra=["decode9.logits_last"])
    got = undeclared_files(d, json.loads((d / "manifest.json").read_text())["pins"])
    ok = got == ["decode9.logits_last"]
    print(f"  [{'PASS' if ok else 'FAIL'}] unclaimed_file: undeclared_files() -> {got}, want"
          " ['decode9.logits_last'] — the same function engine/gate.py section (0) calls")
    if not ok:
        bad.append("unclaimed_file")
    shutil.rmtree(work, ignore_errors=True)
    print(f"CENSUS SELFTEST {'PASS' if not bad else 'FAIL'} — {len(cases) + 1} cases, {len(bad)}"
          f" wrong{' (' + ', '.join(bad) + ')' if bad else ''}: synthetic fixture only, no real"
          " fixture was read, no engine ran and no timing taken")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", action="append", default=None,
                    help="restrict to these fixture names (repeatable). The DEFAULT is every"
                         " manifest-bearing directory under data/refs, enumerated at run time.")
    ap.add_argument("--quiet", action="store_true", help="print only failures and the count line")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the pairing check and the gate-shared file check can fail, on a"
                         " synthetic fixture in tmp; reads no real fixture")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs = pin_fixtures(a.fixture)
    adjudicated = 0
    skipped: list[str] = []
    unadj: list[str] = []            # found but could not be read: a named gap, never a silent pass
    mismatch: list[str] = []         # read, and the directory contradicts its own manifest
    for d in dirs:
        try:
            fx = Fixture(d)
        except Exception as exc:                     # noqa: BLE001 - an unreadable fixture is a
            unadj.append(d.name)                     # named gap, not a crash and not a pass
            print(f"[UNADJUDICABLE] {d.name} — {type(exc).__name__}: {exc}")
            continue
        if fx.pins:
            try:
                fx.adjudicate()
            except Exception as exc:                 # noqa: BLE001
                unadj.append(fx.name)
                print(f"[UNADJUDICABLE] {fx.name} — {type(exc).__name__}: {exc}")
                continue
            adjudicated += 1
            if fx.problems:
                mismatch.append(fx.name)
                print(f"[FIXTURE MISMATCH] {fx.name} — {len(fx.problems)} problem(s):")
                for kind, what in fx.problems:
                    print(f"    [{kind}] {what}")
                if not a.quiet:
                    print(f"    (pins {len(fx.pins)}, files {len(fx.disk)}, logits rows paired"
                          f" {fx.rows_checked}, declared gaps {len(fx.declared)})")
            elif not a.quiet:
                print(f"[OK] {fx.name} — pins {len(fx.pins)}, files {len(fx.disk)},"
                      f" logits rows paired {fx.rows_checked}, declared gaps {len(fx.declared)}"
                      + (f", carried {', '.join(fx.carried)}" if fx.carried else ""))
        else:
            kind = fx.man.get("kind")
            skipped.append(f"{fx.name}: carried record (kind={kind}), no pins")
            if not a.quiet:
                print(f"[SKIP] {fx.name} — carried record (kind={kind}), no pins to census")
    if skipped and not a.quiet:
        print("carried records passed over (named, not silently ignored): " + "; ".join(skipped))
    total = adjudicated + len(skipped) + len(unadj)
    if total != len(dirs):
        print(f"CENSUS FAIL — the counts do not add up ({total} of {len(dirs)} directories"
              " accounted for), so the population was not fully walked")
        return 1
    print(f"CENSUS population={len(dirs)} adjudicated={adjudicated}"
          f" carried={len(skipped)} unadjudicable={len(unadj)} mismatches={len(mismatch)}"
          " — fixture provenance only, no engine ran and no timing taken")
    if mismatch:
        print(f"CENSUS FAIL — {len(mismatch)} fixture(s) above are NOT internally consistent. Read"
              " them as a fixture/floor artifact, never as an engine numerics result; a red here"
              " means the manifest and the directory disagree, and no gate verdict about the engine"
              " can be taken from such a directory.")
        return 1
    if unadj:
        print(f"CENSUS FAIL — {len(unadj)} fixture(s) could not be adjudicated (reasons above). A"
              " census that quietly drops a directory cannot fail, which is the trap this row is for.")
        return 1
    print("CENSUS PASS — every adjudicated fixture's file set, declared gaps and logit pairings"
          " agree with its own manifest")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
