#!/usr/bin/env python3
#!/usr/bin/env python3
"""Was any numeric tolerance loosened to make something pass? A recorded band history, policed.

Four invocations. `check` is the one that matters — `engine/gate.py` calls it before it does any
work, and it is what turns `AGENTS.md`'s "never loosen a gate to make it pass" from a review comment
into a failing command.

    .venv/bin/python tools/band_history.py check            # (= `--check`); what the gate runs
    .venv/bin/python tools/band_history.py record \
        --key BAND.cos_hard --old 0.9995 --new 0.9994 \
        --reason "route_price.py puts dec64t's own spread at 3.4939 at step 54" --decision D-2
    .venv/bin/python tools/band_history.py --selftest       # CPU only: no fixture, no device, no write
    .venv/bin/python tools/band_history.py --since "2026-10-11 00:00"    # the original git walk

WHY A HISTORY AT ALL. The numbers that decide pass/fail live in two places: `engine/gate.py`'s band
blocks — `BAND` (flat §10 floors plus the demoted global route margins) and `CASCADE_BAND` (the
measured hand<->library cascade, D-11) — and each priced fixture's
`data/refs/<fixture>/route_allowance.json` margin. Nothing mechanical noticed an edit to any of them,
so "the band was widened on evidence" and "the band was widened to make a bug go green" looked
identical in a diff. `check` compares the working tree against the newest RECORDED value for every key
and fails on anything that moved, appeared, or vanished without a line saying so. Every numeric dict in
the gate that gates is therefore listed in `POLICED_BLOCKS`: a band declared in some other dict is a
band outside the ledger, which is exactly what `CASCADE_BAND` was before its four figures were admitted.

The record is `bench/band-history.jsonl`, one JSON object per line — `data/` is gitignored and the
allowance JSONs are regenerable artifacts, so a record that lived beside them would be untracked and
throwaway. The tracked file is the ledger:

    {ts, file, key, old, new, reason, decision, by}   + optional "event": applied|refused|removed

A KEY is the pair (file, key). For `engine/gate.py` the keys are a policed block's field names prefixed
with that block (`BAND.cos_hard`, `CASCADE_BAND.stage_cos_min` — two dicts in one file, so the block has
to be part of the name or a ledger line could not say which number it describes); for an allowance file
they are `margin` and `margin_rel`, and the fixture is in the file name — so each
fixture's margin is its own key, which is the whole point of D-2 (`progress/2026-10-12/plan-2026-10-12.md` §3): one
global margin key would let one fixture's measurement excuse every other fixture's.

`record` REFUSES a reason that is empty or boilerplate, because "we needed it to pass" is a motive and
not a justification: pass-seeking phrases are rejected whatever their length, and a reason that says
nothing beyond band-mover boilerplate ("loosen", "tune", "adjust the margin") is rejected for having no
content left once the boilerplate is stripped. It also refuses a record whose `--new` is not the value
the working tree actually holds, so the ledger cannot drift from the code it claims to describe. A
`D-x` decision has to exist in the plan's decision log.

Where the value is generated the line is written by the generator, not by a human:
`engine/route_allowance.py` appends its own lines when it prices a per-fixture margin — and when it
REFUSES to, which is what `event: "refused"` is for. A refused line is evidence, not a band move:
`check` counts it and ignores it when deciding what the current value is.

WHAT THIS DOES NOT STOP, said plainly because a checker that overstates itself is worse than none: an
author determined to loosen a band can edit the parser in this file, or write a fluent false `reason`,
or delete the history (which `check` reports as an error, not as a pass). Nor does it police each
allowance file's per-step route spreads — `engine/gate.py:allowance_problems` checks those against the
file's own margin, its own recorded engine deficits and the prompt/feed/specimpl hashes. The backstop
for this tool's own source is the git walk below and ordinary review, not itself.

The git-walk mode is unchanged from 2026-10-11 and is still the default with no subcommand, so
`progress/2026-10-11/commit-review-2026-10-11.md` §2 keeps working verbatim. It walks every commit after `--since`
that touched a gate file (prototype/*/run.py, */check.py,
*/floor_probe.py, */reference.py and speccheck/gates/*.py), reads that revision's tolerance
constants straight out of the file, and reports only the places where a threshold's VALUE moved —
raised, tightened or deleted — with the commit that moved it. Two passes:

  1. named constants (`F4_MEANREL_MAX = 5e-3`, tuple forms included);
  2. thresholds that are not constants: keyword arguments (`ulp_max=2`, `cancel_rel=1e-4`,
     `meanrel_max=1e-3`), argparse defaults on threshold flags, and inline comparisons
     (`cos < 0.999`, `meanrel > 0.05`).

A threshold that appears for the first time in a file created after `--since` is NEW, not loosened,
so it is counted and not reported as a move — that distinction is the whole point: a gate written
today says nothing about whether a gate was widened today. Read-only; touches no device.

The answer this tool was written for is in `progress/2026-10-11/commit-review-2026-10-11.md` §2.
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY_REL = "bench/band-history.jsonl"
GATE_FILE_REL = "engine/gate.py"
ALLOWANCE_GLOB = "data/refs/*/route_allowance.json"
PLAN_REL = "progress/2026-10-12/plan-2026-10-12.md"

# Keys where a LOWER number is the looser band. `BAND.cos_hard`/`BAND.cos_report` and
# `CASCADE_BAND.stage_cos_min` are cos FLOORS (a deficit ceiling like `CASCADE_BAND.row_1cos_max` goes
# the other way); every other policed key is a ceiling or a multiplier, where higher is looser. Without
# this the tool can say a number moved but not that it was widened.
LOWER_IS_LOOSER = {"BAND.cos_hard", "BAND.cos_report", "CASCADE_BAND.stage_cos_min"}


class BandHistoryError(Exception):
    """Something this tool cannot judge without guessing. Always a failure, never a warning."""


def _num(value, what="value"):
    """A band value as a float, or None for a key that does not exist. `absent` is spelled, not empty,
    so a typo cannot silently turn a moved number into a deleted key."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("absent", "-", "null", "None", "(absent)"):
        return None
    try:
        return float(text)
    except ValueError:
        raise BandHistoryError(f"{what} {value!r} is not a number"
                               " (write 'absent' for a key that does not exist)")


def _fmt(value):
    return "(absent)" if value is None else format(float(value), "g")


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return float(a) == float(b)


# ---------------- the values in the working tree -------------------------------------------------

# Every `engine/gate.py` dict whose numbers gate. A block named here and missing from the file is a
# failure, not an empty set: the point of the list is that a band cannot escape the ledger by being
# renamed or folded into a prettier dict. `CASCADE_BAND` joined when the route-(c) audit admitted its
# four figures (D-11); until then they were live numeric bands with no line behind them.
POLICED_BLOCKS = ("BAND", "CASCADE_BAND")
BAND_ENTRY = re.compile(r'^\s*"([A-Za-z_][A-Za-z_0-9]*)"\s*:\s*([^#,]+?)\s*(?:,\s*(?:#.*)?)?$')


def _block_pattern(name):
    # `^` with re.M is what keeps `BAND` from matching the `CASCADE_BAND = {` line below it.
    return re.compile(r"^" + re.escape(name) + r"\s*=\s*\{(.*?)^\}\s*$", re.M | re.S)


def gate_band_values(gate_path: Path) -> dict:
    """Every number in `engine/gate.py`'s policed band blocks, keyed `<BLOCK>.<name>`.

    Parsed rather than imported, for two reasons. `engine/gate.py` imports numpy and this tool has to
    stay importable and cheap; and the value that decides pass/fail is the one in the file a reviewer
    diffs, not one a running process could have rebound. Fails closed: no file, no block from
    `POLICED_BLOCKS`, or an entry that is not a bare number, is an error — a checker that quietly found
    zero keys would turn a rename into a clean bill of health.
    """
    gate_path = Path(gate_path)
    if not gate_path.exists():
        raise BandHistoryError(f"{gate_path} does not exist — there is no band to read, which is"
                               " judged a failure rather than an empty set")
    text = gate_path.read_text()
    out = {}
    for block_name in POLICED_BLOCKS:
        block = _block_pattern(block_name).search(text)
        if block is None:
            raise BandHistoryError(f"{gate_path}: no top-level `{block_name} = {{ ... }}` block found"
                                   " — this tool exists to read that block, so a rename or a move is a"
                                   " failure, not an empty set")
        for line in block.group(1).splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            m = BAND_ENTRY.match(line)
            if m is None:
                raise BandHistoryError(f"{gate_path}: cannot read a {block_name} entry from"
                                       f" {line.strip()[:70]!r}")
            key = block_name + "." + m.group(1)
            out[key] = _num(m.group(2), key)
    if not out:
        raise BandHistoryError(f"{gate_path}: the policed blocks hold no numeric entries")
    return out


def tree_values(root=ROOT) -> dict:
    """{(file_rel, key): value} for every band number the working tree currently holds."""
    root = Path(root)
    values = {(GATE_FILE_REL, k): v for k, v in gate_band_values(root / GATE_FILE_REL).items()}
    for path in sorted(root.glob(ALLOWANCE_GLOB)):
        rel = path.relative_to(root).as_posix()
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise BandHistoryError(f"{rel}: cannot be read as JSON, so its margin cannot be"
                                   f" judged: {e}")
        for key in ("margin", "margin_rel"):
            if key not in doc:
                raise BandHistoryError(f"{rel} has no {key!r} field; the gate would fall back to"
                                       " engine/gate.py's BAND, which is a band change in itself")
            values[(rel, key)] = _num(doc[key], f"{rel}:{key}")
    return values


# ---------------- the ledger ---------------------------------------------------------------------

def read_history(path) -> list:
    """Every line of the ledger, in order. Append order is the timeline; a later line wins."""
    path = Path(path)
    if not path.exists():
        raise BandHistoryError(f"no band history at {path} — this tool cannot judge a band with no"
                               " record of it. Seed it (one line per key, with its justification) or"
                               " restore it from git; an absent history is a failure, not a pass.")
    out = []
    for no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise BandHistoryError(f"{path}:{no} is not a JSON object: {e}")
        for f in ("file", "key", "new"):
            if f not in rec:
                raise BandHistoryError(f"{path}:{no} has no {f!r} field")
        rec.setdefault("event", "applied")
        out.append(rec)
    return out


def newest_values(records: list) -> dict:
    """{(file, key): (value, record)} from the newest applied/removed line for each key.

    `event: refused` lines are a re-pricing that was NOT written, so they carry no current value —
    they are the paper trail of an attempted loosening, and `check` counts them without letting one
    move a band.
    """
    newest = {}
    for rec in records:
        if rec["event"] not in ("applied", "removed"):
            continue
        value = None if rec["event"] == "removed" else _num(rec["new"], f"{rec['file']}:{rec['key']}")
        newest[(rec["file"], rec["key"])] = (value, rec)
    return newest

CONST_NAME = re.compile(r"(MIN|MAX|REL|FLOOR|TOL|BAND|ACCEPT|AGREE|NDIFF|COS|MEANREL|ULP|EPS)")
ASSIGN = re.compile(r"^([A-Za-z][A-Za-z_0-9]*(?:\s*,\s*[A-Za-z][A-Za-z_0-9]*)*)\s*=\s*"
                    r"([^#]+?)\s*(?:#.*)?$")
KWARG = re.compile(r"(cancel_rel|ulp_max|meanrel_max|cos_min|ndiff_pct_max|cancel_band|"
                   r"min-accept|min-agreement|--floor|--cos-min|--meanrel-max)"
                   r"[^0-9A-Za-z_]{0,4}([0-9]*\.?[0-9]+(?:e-?\d+)?)")
INLINE = re.compile(r"(cos >=|cos <|meanrel <=|meanrel >)\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)")


def current_recorded(file, key, root=ROOT, history=None):
    """The newest recorded value for one key, or None if it has never been recorded.

    For a generator that writes a value it did not type: the file may be absent (deleted deliberately
    before a re-price) while the ledger still knows what the value WAS, and a record whose `old` is a
    guess is a ledger line that cannot be audited.
    """
    try:
        records = read_history(Path(history) if history else Path(root) / HISTORY_REL)
    except BandHistoryError:
        return None
    found = newest_values(records).get((str(file), str(key)))
    return None if found is None else found[0]


@dataclass
class BandProblem:
    file: str
    key: str
    why: str
    recorded: object
    current: object

    def way(self) -> str:
        if self.current is None:
            return "the key was REMOVED from the tree"
        if self.recorded is None:
            return "the key is NEW — no line introduces it"
        looser = (float(self.current) < float(self.recorded)
                  if self.key in LOWER_IS_LOOSER else float(self.current) > float(self.recorded))
        return "LOOSER than recorded" if looser else "tighter than recorded"

    def label(self) -> str:
        return f"{self.file} :: {self.key}"

    def record_command(self) -> str:
        """The exact line that would put this on the record. Copyable, so the fix is not a puzzle."""
        return (f'.venv/bin/python tools/band_history.py record --file "{self.file}"'
                f' --key "{self.key}" --old {_fmt(self.recorded)} --new {_fmt(self.current)}'
                ' --reason "<the measurement that moved this, and where it was measured>"'
                ' --decision D-x')

    def describe(self) -> str:
        return (f"{self.label()} — recorded {_fmt(self.recorded)}, working tree {_fmt(self.current)}"
                f" ({self.way()}) — no line in {HISTORY_REL} records it")


@dataclass
class Report:
    ok: bool
    keys: int = 0
    problems: list = field(default_factory=list)   # list[BandProblem]
    skipped: list = field(default_factory=list)    # recorded keys whose fixture is not on this box
    refused: int = 0
    newest_ts: str = ""
    newest_decision: str = ""
    error: str = ""

    def summary(self) -> str:
        if self.error:
            return f"band history: CANNOT JUDGE — {self.error}"
        line = (f"band history: {self.keys} band value(s) checked against {HISTORY_REL}"
                f" (newest {self.newest_ts or 'no timestamp'},"
                f" {self.newest_decision or 'no decision'})")
        if self.skipped:
            line += f"; {len(self.skipped)} recorded key(s) skipped, fixture directory not present"
        if self.refused:
            line += f"; {self.refused} refused re-pricing(s) on the record, none applied"
        return line


def check_tree(root=ROOT, history=None) -> Report:
    """Compare the working tree's band values with the newest recorded value for each key.

    Three failures, not one: a value that MOVED without a line, a key that APPEARED without one, and a
    key that DISAPPEARED without one. The last matters as much as the first — deleting an allowance
    file drops a fixture back onto the flat floors, and deleting a BAND entry deletes the criterion.
    A recorded key whose fixture directory is not on this box is skipped out loud instead: `data/` is
    gitignored, so an absent fixture is a missing artifact, while a present directory with no
    `route_allowance.json` is a fixture that stopped being priced and that is a band change.
    """
    root = Path(root)
    hist = Path(history) if history else root / HISTORY_REL
    try:
        tree = tree_values(root)
        records = read_history(hist)
    except BandHistoryError as e:
        return Report(ok=False, error=str(e))
    newest = newest_values(records)
    rep = Report(ok=True, keys=len(tree),
                 refused=sum(1 for r in records if r["event"] == "refused"))
    for fk in sorted(set(tree) | set(newest)):
        f, k = fk
        if fk in tree and fk in newest:
            if not _same(newest[fk][0], tree[fk]):
                rep.problems.append(BandProblem(f, k, "moved", newest[fk][0], tree[fk]))
            continue
        if fk in tree:
            rep.problems.append(BandProblem(f, k, "unrecorded", None, tree[fk]))
            continue
        if newest[fk][0] is None:
            continue                       # a recorded removal is a recorded decision
        if not (root / f).parent.exists():
            rep.skipped.append(f"{f} :: {k}")
            continue
        rep.problems.append(BandProblem(f, k, "removed", newest[fk][0], None))
    applied = [r for r in records if r["event"] in ("applied", "removed")]
    if applied:
        rep.newest_ts = str(applied[-1].get("ts", ""))
        rep.newest_decision = str(applied[-1].get("decision", ""))
    rep.ok = not rep.problems
    return rep


def render(rep: Report, quiet_ok: bool = False) -> int:
    """Print a report the way the CLI and the gate both print it. Returns the exit code.

    On success it is one indented line under its own section header, unbracketed, so a gate run's
    `[PASS]`/`[FAIL]` tally is untouched and `engine/route_price.py --engine-log` (which parses a gate
    log for its engine column) sees nothing new to trip on.
    """
    if not (quiet_ok and rep.ok):
        print("\n== band history — every band value must have a recorded justification")
    if rep.error:
        print(f"  band history: CANNOT JUDGE — {rep.error}")
        return 1
    if rep.ok:
        if not quiet_ok:
            print("  " + rep.summary())
        return 0
    print("BAND HISTORY FAIL — a band value in the working tree has no recorded justification.")
    for p in rep.problems:
        print(f"  [FAIL] {p.describe()}")
        print(f"      record it: {p.record_command()}")
    print(f"BAND HISTORY FAIL — {len(rep.problems)} band value(s) moved with no recorded"
          f" justification. A band edit needs a measurement and a decision on the record in"
          f" {HISTORY_REL}; this tool does not decide whether either is true, review does. Nothing"
          " else was run: no fixture was read and no device work was started.")
    return 1


def for_gate(root=ROOT) -> int:
    """What `engine/gate.py` calls at startup: 0 = every band value is recorded, 1 = refuse to run."""
    return render(check_tree(root))


def git(root, *args):
    return subprocess.run(["git", "-C", root] + list(args),
                          capture_output=True, text=True).stdout


# ---------------- mode: review — the original 2026-10-11 git walk --------------------------------

GLOBS = ["prototype/*/run.py", "prototype/*/check.py", "prototype/*/floor_probe.py",
         "prototype/*/reference.py", "speccheck/gates/*.py"]


def consts_of(text):
    """Top-level-looking `NAME = number` (or tuple form) whose name looks like a threshold."""
    out = {}
    for line in text.splitlines():
        m = ASSIGN.match(line.strip())
        if not m:
            continue
        names = [n.strip() for n in m.group(1).split(",")]
        vals = [v.strip() for v in m.group(2).split(",")]
        if len(names) != len(vals):
            continue
        for name, val in zip(names, vals):
            if CONST_NAME.search(name):
                try:
                    float(val)
                except ValueError:
                    continue
                out["const:" + name] = val
    return out


def thresholds_of(text):
    """Thresholds that live in call sites and CLI defaults rather than in constants."""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        hits = set(m.group(0) for m in KWARG.finditer(s)) | set(m.group(0) for m in INLINE.finditer(s))
        if hits:
            out[s[:110]] = hits
    return out


def review_main(since, root):
    """The 2026-10-11 git walk, unchanged in behaviour; `main` below dispatches to it."""
    touched = git(root, "log", "--since=" + since, "--name-only", "--format=%x09").split()
    files = sorted({f for f in touched for g in GLOBS if fnmatch.fnmatch(f, g)})

    moved, brand_new = [], set()
    for f in files:
        revs = list(reversed(git(root, "log", "--since=" + since, "--format=%h",
                                 "--", f).split()))
        prev, first = None, None
        for rev in revs:
            text = git(root, "show", "%s:%s" % (rev, f))
            if not text:
                continue
            cur = consts_of(text)
            if first is None:
                first = set(cur)
            if prev is not None:
                for name, val in cur.items():
                    if name in prev and prev[name] != val:
                        moved.append((f, name[5:], prev[name], val, rev))
                for name in prev:
                    if name not in cur:
                        moved.append((f, name[5:], prev[name], "(deleted)", rev))
            prev = cur
        brand_new |= (first or set())

    print("since %s | gate files touched: %d | threshold constants first seen today: %d"
          % (since, len(files), len(brand_new)))
    print("\n== pass 1: named threshold constants that MOVED ==")
    if not moved:
        print("(none)")
    for f, name, old, new, rev in moved:
        try:
            way = "RAISED (looser)" if float(new) > float(old) else "tightened"
        except ValueError:
            way = "DELETED the criterion"
        print("  %-30s %-20s %-9s -> %-11s %-20s %s  %s"
              % (f, name, old, new, way, rev,
                 git(root, "log", "-1", "--format=%s", rev).strip()[:58]))

    print("\n== pass 2: call-site / CLI thresholds that MOVED (line identity, so expect noise) ==")
    any2 = False
    for f in files:
        revs = list(reversed(git(root, "log", "--since=" + since, "--format=%h",
                                 "--", f).split()))
        prev = None
        for rev in revs:
            text = git(root, "show", "%s:%s" % (rev, f))
            if not text:
                continue
            cur = thresholds_of(text)
            if prev is not None:
                gone = set(prev) - set(cur)
                added = set(cur) - set(prev)
                for g in sorted(gone):
                    pre = [a for a in added if a[:40] == g[:40]]
                    if pre:
                        print("  %-30s %s  %s  ->  %s" % (f, rev, prev[g], cur[pre[0]]))
                    else:
                        print("  %-30s %s  disappeared: %s %s" % (f, rev, prev[g], g[:60]))
                    any2 = True
            prev = cur
    if not any2:
        print("(none)")


# ---------------- recording ----------------------------------------------------------------------

# A reason whose stated goal is the gate passing, rather than a measurement that moved. Refused at any
# length, because this is the exact motive `AGENTS.md` forbids acting on. Each entry is (label, pattern)
# so the refusal names the shape it rejected instead of printing a regex at the operator.
PASS_SEEKING = [
    ("so that it passes", re.compile(r"\bso\s+that?\s+(it|this|that|the\s+gate|the\s+fixture|"
                                     r"the\s+run|things)\s+"
                                     r"(will\s+|would\s+|can\s+|could\s+|may\s+)?"
                                     r"(pass|passes|passed|green|go\s+green)\b")),
    ("for the gate to pass", re.compile(r"\b(to|for)\s+(the\s+gate|it|this|things)\s+to\s+pass\b")),
    ("make it pass", re.compile(r"\bmake\s+(it|this|that|the\s+gate|the\s+fixture|things)\s+"
                                r"(pass|green|go\s+green)\b")),
    ("needed to pass", re.compile(r"\b(needed|want|wanted|need)\s+to\s+(pass|be\s+green|go\s+green|"
                                  r"be\s+passing)\b")),
    ("for now", re.compile(r"\bfor\s+now\b")),
    ("temporarily", re.compile(r"\btemporar(y|ily)\b")),
    ("flaky", re.compile(r"\bflak(y|iness)\b")),
]
# Words that describe the ACT of moving a band and nothing else. A reason built only out of these (plus
# stop-words) names no measurement, which is the thing this exists to refuse. This is a THINNESS test,
# not a word ban: a real reason that happens to use "loosen" while citing a probe passes, and it has to,
# because half the honest write-ups in this repo use that word about the retired global margin.
BOILERPLATE = {
    "band", "bands", "value", "values", "key", "keys", "margin", "margins", "floor", "floors",
    "tolerance", "tolerances", "threshold", "thresholds", "gate", "gates", "fixture", "fixtures",
    "engine", "reason", "edit", "edits", "change", "changed", "changes", "update", "updated",
    "number", "numbers", "constant", "constants", "fix", "fixed", "fixes", "tweak", "tweaked",
    "tweaks", "tune", "tuned", "tunes", "tuning", "adjust", "adjusted", "adjusts", "adjusting",
    "adjustment", "adjustments", "loosen", "loosens", "loosening", "loosened", "relax", "relaxes",
    "relaxed", "relaxing", "widen", "widens", "widened", "widening", "fudge", "fudged", "hack",
    "hacked", "hacks", "misc", "stuff", "things", "again", "minor", "small", "slightly", "little",
    "bit", "needed", "want", "wanted", "because", "just", "new", "old", "moved", "moves", "moving",
    "raised", "raise", "raises", "lowered", "lower", "lowers", "bumped", "bump", "bumps", "tighter",
    "looser", "tighten", "tightened", "different", "wrong", "correct", "corrected", "whatever",
    "etc", "todo", "tbd", "a",
}
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "as", "from",
    "than", "then", "so", "no", "not", "but", "if", "we", "our", "i", "their", "there", "which",
    "over", "under", "into", "out", "up", "down", "all", "any", "per", "each", "own", "same",
    "one", "two", "three", "step", "steps", "see", "used", "use", "uses", "using",
}
MIN_REASON_CHARS = 24
MIN_CONTENT_WORDS = 4
PLACEHOLDER_DECISIONS = {"n/a", "na", "none", "plan", "see plan", "todo", "tbd", "x", "d-x", "?"}


def justify(reason, decision, root=ROOT):
    """Reject a reason that is empty, boilerplate, or a motive instead of a measurement.

    Two independent tests, because either one alone is trivially gameable: pass-seeking phrases are
    refused at any length, and a thin reason is refused by what survives de-boilerplating. The
    `decision` half is the other side of the record — a `D-x` has to exist in the plan's decision log
    (so an invented one fails) or be a real commit subject.
    """
    text = (reason or "").strip()
    if not text:
        raise ValueError("reason is empty. A band edit needs the measurement that moved it.")
    if len(text) < MIN_REASON_CHARS:
        raise ValueError(f"reason is {len(text)} characters — too short to name a measurement."
                         " Say what was measured, on what, and where it is written down.")
    for label, pat in PASS_SEEKING:
        if pat.search(text.lower()):
            raise ValueError(f"reason states the goal of making the gate pass (it says {label!r}). A"
                             " band move needs the measurement that forced it, not the outcome you"
                             " wanted.")
    words = re.findall(r"[a-z0-9][a-z0-9._/%-]*", text.lower())
    content = [w for w in words if w not in STOPWORDS and w not in BOILERPLATE]
    if len(content) < MIN_CONTENT_WORDS:
        raise ValueError("reason is boilerplate — once band-mover words are removed only"
                         f" {content or ['(nothing)']} is left. Name the measurement, the fixture and"
                         " step, or the spec reference that moved (and the decision that accepted"
                         " it).")
    dec = (decision or "").strip()
    if not dec:
        raise ValueError("decision is empty: give a D-x from the plan's decision log, or the subject"
                         " line of the commit that decided it.")
    if re.fullmatch(r"D-\d+", dec):
        plan = Path(root) / PLAN_REL
        if not plan.exists():
            raise ValueError(f"decision {dec} cannot be checked: {plan} is missing")
        if f"**{dec}**" not in plan.read_text():
            raise ValueError(f"decision {dec} is not in {PLAN_REL}'s decision log. Record the"
                             " decision first, or cite the commit subject that carries it.")
    elif len(dec) < 6 or dec.lower() in PLACEHOLDER_DECISIONS:
        raise ValueError(f"decision {dec!r} is a placeholder. Use a D-x from the decision log or a"
                         " real commit subject.")


def default_author() -> str:
    who = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.strip()
    return who or os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def record(*, file, key, old, new, reason, decision, by=None, event="applied",
           root=ROOT, history=None, quiet=False):
    """Append one line to the ledger — the only way a band move becomes legal.

    The ledger has to describe the code, so the tree is checked against the line before it is written:
    an `applied` record must land on the value the tree now holds, and a `refused` one must NOT have
    been applied. Without that, `record` would be a place to write a story about a band that nobody
    moved, which is the false-justification hole this cannot otherwise close.
    """
    root = Path(root)
    hist = Path(history) if history else root / HISTORY_REL
    justify(reason, decision, root=root)
    old_v, new_v = _num(old, "old"), _num(new, "new")
    if event not in ("applied", "refused", "removed"):
        raise ValueError(f"unknown event {event!r} (applied | refused | removed)")
    if event != "refused" and _same(old_v, new_v):
        raise ValueError(f"old and new are both {_fmt(new_v)} — that is not a band move, so there is"
                         " nothing to record.")
    if event == "removed" and new_v is not None:
        raise ValueError("event=removed needs --new absent")
    want = new_v if event in ("applied", "removed") else old_v
    if want is not None and (root / str(file)).exists():
        now = tree_values(root).get((str(file), str(key)))
        if now is not None and not _same(now, want):
            raise ValueError(f"the working tree holds {_fmt(now)} for {file} :: {key}, but this"
                             f" record says the move lands on {_fmt(want)}. Edit the file (or fix the"
                             " numbers) so the ledger describes the code.")
    if event == "removed":
        new_v = None
    line = {"ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "file": str(file), "key": str(key), "old": old_v, "new": new_v,
            "reason": reason.strip(), "decision": decision.strip(),
            "by": (by or default_author()).strip(), "event": event}
    if not hist.exists():
        raise BandHistoryError(f"{hist} does not exist — the ledger is a tracked file; create it"
                               " (seeded, with its justifications) before recording into it.")
    with hist.open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    if not quiet:
        print(f"  recorded: {line['file']} :: {line['key']} {_fmt(old_v)} -> {_fmt(new_v)}"
              f" [{line['decision']}]" + (f" (event={event})" if event != "applied" else ""))
    return line


def check_and_exit(root, history=None, quiet=False) -> int:
    return render(check_tree(root, history), quiet_ok=quiet)


def record_main(a) -> int:
    try:
        record(file=a.file, key=a.key, old=a.old, new=a.new, reason=a.reason,
               decision=a.decision, by=a.by, event=a.event, root=a.root)
    except (ValueError, BandHistoryError) as e:
        print(f"REFUSED — the band was not recorded: {e}")
        return 1
    rep = check_tree(a.root)
    if not rep.ok:
        print(f"  recorded, but the ledger and the tree still disagree"
              f" ({len(rep.problems)} problem(s)) — a record of one key does not excuse another:")
        render(rep)
        return 1
    return 0


SYNTH_GATE = '''#!/usr/bin/env python3
BAND = {
    "cos_hard": 0.9995,       # flat floor
    "meanrel_hard": 0.05,     # cascade ceiling
    "cos_report": 0.9999,     # reported, not gated
}

CASCADE_BAND = {
    "row_1cos_max": 0.002,      # one logits row against another route's row
    "stage_meanrel_max": 0.03,  # one stage tile, NOT a logits row
    "stage_cos_min": 0.9998,    # the stage cos FLOOR: lower is looser
    "flip_room_max": 0.10,      # room a flipped token is allowed to have had
}
'''

# The scratch gate's cascade values, so a case can name the recorded number instead of retyping it.
SYNTH_CASCADE = {"row_1cos_max": 0.002, "stage_meanrel_max": 0.03,
                 "stage_cos_min": 0.9998, "flip_room_max": 0.10}


def _set_band(text: str, key: str, value) -> str:
    """Rewrite one BAND entry in a synthetic gate file. Raises if the entry is not there, so a test can
    never pass by editing nothing."""
    pat = re.compile(r'^(\s*"' + re.escape(key) + r'"\s*:\s*)([0-9.eE+-]+)(,?)', re.M)
    out, n = pat.subn(lambda m: m.group(1) + format(float(value), "g") + m.group(3), text)
    if n != 1:
        raise AssertionError(f"selftest could not rewrite BAND.{key} (matched {n})")
    return out


def _drop_band(text: str, key: str) -> str:
    pat = re.compile(r'^\s*"' + re.escape(key) + r'"\s*:[^\n]*\n', re.M)
    out, n = pat.subn("", text)
    if n != 1:
        raise AssertionError(f"selftest could not delete BAND.{key} (matched {n})")
    return out


def selftest_main() -> int:
    """Prove the checker discriminates, on synthetic files, with no device and no fixture touched.

    The scratch tree is a real tree-shaped directory (a gate file with a BAND, two priced fixtures, a
    ledger), so every case goes through the same read/compare/record code the gate uses — and the last
    cases run against the SHIPPED tree and the shipped ledger, which is what proves the bootstrap is
    honest: a seed line that does not match reality fails here, not in a gate run next week.
    """
    import tempfile
    bad = []

    def eq(label, got, want):
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} — got {got}, want {want}")
        if not ok:
            bad.append(label)

    def raises(label, fn):
        try:
            fn()
        except (ValueError, BandHistoryError) as e:
            print(f"  [PASS] {label} — refused: {str(e)[:96]}")
            return
        print(f"  [FAIL] {label} — accepted, and it should not have")
        bad.append(label)

    with tempfile.TemporaryDirectory(prefix="band_history_selftest_") as td:
        tdp = Path(td)
        (tdp / "engine").mkdir()
        (tdp / "bench").mkdir()
        # The plan's decision log is what validates a `D-x`, so the scratch tree points at the real one
        # rather than faking one — a fabricated decision still has to fail here.
        os.symlink(Path(ROOT) / PLAN_REL, tdp / "bench" / Path(PLAN_REL).name)
        gate = tdp / "engine" / "gate.py"
        gate.write_text(SYNTH_GATE)
        fx = {"fxA": (3.5, 0.42), "fxB": (7.99, 0.72)}
        for name, (m, mr) in fx.items():
            d = tdp / "data" / "refs" / name
            d.mkdir(parents=True)
            (d / "route_allowance.json").write_text(
                json.dumps({"fixture": name, "margin": m, "margin_rel": mr}))
        hist = tdp / "bench" / "band-history.jsonl"
        hist.write_text("")

        def rec(**kw):
            kw.setdefault("root", tdp)
            return record(quiet=True, **kw)

        # Seed the scratch ledger the way the shipped one was seeded: one honest line per key.
        rec(file=GATE_FILE_REL, key="BAND.cos_hard", old="absent", new=0.9995,
            reason="seeded from the shipped gate: §10.1 G5 incremental-vs-batch floor",
            decision="D-2")
        rec(file=GATE_FILE_REL, key="BAND.meanrel_hard", old="absent", new=0.05,
            reason="seeded from the shipped gate: §10.2 cascade ceiling, meanrel of means",
            decision="D-2")
        rec(file=GATE_FILE_REL, key="BAND.cos_report", old="absent", new=0.9999,
            reason="seeded from the shipped gate: §10.1 G6 decode logits, reported not gated",
            decision="D-2")
        for name, (m, mr) in fx.items():
            rel = f"data/refs/{name}/route_allowance.json"
            rec(file=rel, key="margin", old="absent", new=m,
                reason=f"seeded: {name}'s own worst engine/route ratio, rounded up at 2 dp",
                decision="D-2")
            rec(file=rel, key="margin_rel", old="absent", new=mr,
                reason=f"seeded: {name}'s own worst meanrel ratio, rounded up at 2 dp",
                decision="D-2")
        for name, val in SYNTH_CASCADE.items():
            rec(file=GATE_FILE_REL, key=f"CASCADE_BAND.{name}", old="absent", new=val,
                reason="seeded from the shipped gate: CASCADE_BAND." + name + ", one figure from the"
                       " route-(c) audit's measured hand<->library cascade",
                decision="D-2")

        eq("1. a tree whose every value is recorded passes",
           [p.describe() for p in check_tree(tdp).problems], [])

        gate.write_text(_set_band(SYNTH_GATE, "cos_hard", 0.9994))
        rep = check_tree(tdp)
        eq("2. an unrecorded band edit FAILS, and names the key that moved",
           [p.label() for p in rep.problems], [f"{GATE_FILE_REL} :: BAND.cos_hard"])
        eq("   ... and calls it LOOSER (a cos floor going down is the band widening)",
           rep.problems[0].way(), "LOOSER than recorded")
        eq("   ... and hands over the command that would record it",
           "band_history.py record" in rep.problems[0].record_command(), True)

        try:
            rec(file=GATE_FILE_REL, key="BAND.cos_hard", old=0.9995, new=0.9994,
                reason="loosen", decision="D-2")
            eq("3. a boilerplate reason is refused", "accepted", "refused")
        except ValueError:
            eq("3. a boilerplate reason is refused", "refused", "refused")
        eq("   ... and the unrecorded edit still fails after the refusal", check_tree(tdp).ok, False)

        rec(file=GATE_FILE_REL, key="BAND.cos_hard", old=0.9995, new=0.9994,
            reason="dec64t's own worst engine/route ratio is 3.4939 at step 54, so a tighter floor"
                   " would fail a step two legal reference routes disagree on",
            decision="D-2")
        eq("4. the same edit, recorded with a measurement and a decision, passes",
           check_tree(tdp).ok, True)

        for junk in ["", "   ", "loosen", "to pass", "tune", "tweaked the margin", "adjust the band",
                     "so it passes", "needed to pass", "make it pass", "relax the floor for now",
                     "values were wrong", "flaky", "temporarily", "band edits"]:
            raises(f"5. reason {junk!r} is refused",
                   lambda j=junk: rec(file=GATE_FILE_REL, key="BAND.meanrel_hard", old=0.05,
                                      new=0.06, reason=j, decision="D-2"))
        eq("   ... and nothing was appended by the refusals",
           len(read_history(hist)), 3 + len(SYNTH_CASCADE) + 1 + len(fx) * 2)
        raises("6. a fluent reason cannot smuggle in a value the tree does not hold",
               lambda: rec(file=GATE_FILE_REL, key="BAND.meanrel_hard", old=0.05, new=0.06,
                           reason="measured on the dec300f prose fixture at step 86, where two legal"
                                  " reference routes disagree by more than the flat ceiling",
                           decision="D-2"))
        raises("   ... and an invented decision is refused",
               lambda: rec(file=GATE_FILE_REL, key="BAND.meanrel_hard", old=0.05, new=0.06,
                           reason="measured on the dec300f prose fixture at step 86, where two legal"
                                  " reference routes disagree", decision="D-99"))

        gate.write_text(_drop_band(gate.read_text(), "cos_report"))
        rep = check_tree(tdp)
        eq("7. a DELETED key fails (a band that vanishes is a band change)",
           [p.label() for p in rep.problems], [f"{GATE_FILE_REL} :: BAND.cos_report"])
        rec(file=GATE_FILE_REL, key="BAND.cos_report", old=0.9999, new="absent",
            reason="§10.1 G6 stays a published figure but this gate stopped reporting it; the"
                   " decision is on the plan's log", decision="D-2", event="removed")
        eq("   ... and a RECORDED removal passes", check_tree(tdp).ok, True)

        # The D-2 property, at the key level: one fixture's margin is its own key.
        (tdp / "data" / "refs" / "fxB" / "route_allowance.json").write_text(
            json.dumps({"fixture": "fxB", "margin": 8.5, "margin_rel": 0.72}))
        eq("8. re-pricing fxB moves ONLY fxB's key — fxA and the gate file are untouched",
           [p.label() for p in check_tree(tdp).problems],
           ["data/refs/fxB/route_allowance.json :: margin"])
        rec(file="data/refs/fxB/route_allowance.json", key="margin", old=7.99, new=8.5,
            reason="fxB's own worst engine/route ratio is 8.41 at step 86 (route_price table),"
                   " rounded up at 2 dp by route_allowance.py", decision="D-2")
        eq("   ... recorded, the tree is clean again", check_tree(tdp).ok, True)
        (tdp / "data" / "refs" / "fxA" / "route_allowance.json").write_text(
            json.dumps({"fixture": "fxA", "margin": 4.0, "margin_rel": 0.42}))
        eq("   and fxA is judged by ITS OWN key, not by the line that recorded fxB",
           [p.label() for p in check_tree(tdp).problems],
           ["data/refs/fxA/route_allowance.json :: margin"])
        rec(file="data/refs/fxA/route_allowance.json", key="margin", old=3.5, new=4.0,
            reason="fxA's own worst engine/route ratio is 3.97 at step 54 (route_price table),"
                   " rounded up at 2 dp", decision="D-2")

        rec(file=GATE_FILE_REL, key="BAND.meanrel_hard", old=0.05, new=0.08,
            reason="a re-price derived 0.08 but the shipped file carries 0.05, so nothing was"
                   " written (route_allowance refuses)", decision="D-2", event="refused")
        rep = check_tree(tdp)
        eq("9. a refused re-price does NOT move the current value",
           (rep.ok, rep.refused), (True, 1))

        gate.write_text(gate.read_text().replace("}", '    "route_margin": 3.5,\n}', 1))
        eq("10. a NEW key with no line introducing it fails",
           [p.label() for p in check_tree(tdp).problems], [f"{GATE_FILE_REL} :: BAND.route_margin"])
        rec(file=GATE_FILE_REL, key="BAND.route_margin", old="absent", new=3.5,
            reason="fallback margin for an allowance file that carries no margin of its own; the"
                   " number is dec64t's own worst ratio rounded up", decision="D-2")
        eq("   ... and passes once recorded", check_tree(tdp).ok, True)

        gate.write_text(SYNTH_GATE.replace('"cos_hard": 0.9995,', '"cos_hard": 1 - 5e-4,', 1))
        rep = check_tree(tdp)
        eq("11. a BAND value this tool cannot read fails closed instead of being skipped",
           rep.ok, False)
        eq("   ... and says so in the error, not in the problem list",
           "BAND.cos_hard" in rep.error, True)
        # The case above left the gate file unreadable on purpose; put the scratch tree back into the
        # state its ledger agrees with (cos_hard moved and recorded, cos_report removed and recorded,
        # route_margin added and recorded) before testing the fixture-directory cases.
        gate.write_text(_drop_band(_set_band(SYNTH_GATE, "cos_hard", 0.9994), "cos_report")
                        .replace("}", '    "route_margin": 3.5,\n}', 1))
        eq("12a. the tree is clean again before the fixture-dir cases",
           [p.describe() for p in check_tree(tdp).problems], [])
        import shutil
        shutil.rmtree(tdp / "data" / "refs" / "fxA")
        rep = check_tree(tdp)
        eq("12. a fixture directory that is not installed is SKIPPED, not reported as removed",
           (rep.ok, sorted(s.split(" :: ")[0] for s in rep.skipped)),
           (True, ["data/refs/fxA/route_allowance.json", "data/refs/fxA/route_allowance.json"]))
        fxd = tdp / "data" / "refs" / "fxA"
        fxd.mkdir(parents=True)
        (fxd / "route_allowance.json").write_text(
            json.dumps({"fixture": "fxA", "margin": 4.0, "margin_rel": 0.42}))
        eq("    ... and it is judged again the moment the file is back",
           [p.label() for p in check_tree(tdp).problems], [])
        (fxd / "route_allowance.json").unlink()
        eq("    ... but a PRESENT fixture whose allowance file was deleted is a failure: it drops"
           " back to the flat floors",
           sorted(p.label() for p in check_tree(tdp).problems),
           ["data/refs/fxA/route_allowance.json :: margin",
            "data/refs/fxA/route_allowance.json :: margin_rel"])
        (fxd / "route_allowance.json").write_text(
            json.dumps({"fixture": "fxA", "margin": 4.0, "margin_rel": 0.42}))

        # The second block. `CASCADE_BAND` shares the gate FILE with `BAND`, so the only way the ledger
        # can tell "the §10 flat floor moved" from "the cascade band moved" is a key that names the
        # block. These cases are what would catch a parser that read one dict and silently missed the
        # other — the exact hole this extension closes.
        gate.write_text(_set_band(gate.read_text(), "stage_meanrel_max", 0.05))
        rep = check_tree(tdp)
        eq("13a. a CASCADE_BAND edit with no line fails too (the second dict is policed)",
           [p.label() for p in rep.problems], [f"{GATE_FILE_REL} :: CASCADE_BAND.stage_meanrel_max"])
        eq("      ... and a ceiling going UP is the widening",
           rep.problems[0].way(), "LOOSER than recorded")
        gate.write_text(_set_band(gate.read_text(), "stage_meanrel_max", 0.03))
        gate.write_text(_set_band(gate.read_text(), "stage_cos_min", 0.9995))
        rep = check_tree(tdp)
        eq("13b. a cascade cos FLOOR going DOWN is the widening, not the tightening",
           [(p.label(), p.way()) for p in rep.problems],
           [(f"{GATE_FILE_REL} :: CASCADE_BAND.stage_cos_min", "LOOSER than recorded")])
        rec(file=GATE_FILE_REL, key="CASCADE_BAND.stage_cos_min", old=0.9998, new=0.9995,
            reason="a re-measured stage ladder bottomed at cos 0.9995 on the scratch fixture, so the"
                   " floor moved down with it",
            decision="D-2")
        eq("      ... recorded, and the scratch tree is clean again", check_tree(tdp).ok, True)
        gate.write_text(_drop_band(gate.read_text(), "flip_room_max"))
        rep = check_tree(tdp)
        eq("13c. a cascade key that VANISHES fails like any other (the room rule would be gone)",
           [p.label() for p in rep.problems], [f"{GATE_FILE_REL} :: CASCADE_BAND.flip_room_max"])
        gate.write_text(_set_band(_drop_band(SYNTH_GATE, "cos_report"), "cos_hard", 0.9994)
                        .replace('"stage_cos_min": 0.9998,', '"stage_cos_min": 0.9995,')
                        .replace("}", '    "route_margin": 3.5,\n}', 1))
        eq("13d. the scratch tree is back to what its ledger says before the shipped-tree case runs",
           [p.describe() for p in check_tree(tdp).problems], [])

    shipped = check_tree(ROOT)
    eq("14. the SHIPPED tree matches the SHIPPED ledger (the bootstrap is honest)",
       (shipped.error, [p.describe() for p in shipped.problems]), ("", []))
    eq("    ... and it found the real per-fixture keys, not just BAND and CASCADE_BAND",
       shipped.keys >= (5 + len(SYNTH_CASCADE)) + 2 * len(list(Path(ROOT).glob(ALLOWANCE_GLOB))), True)
    print("SELFTEST " + ("FAIL — " + "; ".join(bad) if bad else
                         "PASS (an unrecorded band move fails, a recorded one passes, and a reason"
                         " that only says 'we needed it to pass' is not a reason)"))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    selftest = False
    if "--selftest" in argv:
        argv.remove("--selftest")
        selftest = True
    mode = None
    if "--check" in argv:
        argv.remove("--check")
        mode = "check"
    elif argv and argv[0] in ("check", "record", "review"):
        mode = argv.pop(0)
    else:
        mode = "review"        # the bare 2026-10-11 invocation, kept working verbatim
    if selftest:
        mode = "selftest"      # `--selftest` alongside anything else is still the selftest

    if mode == "check":
        ap = argparse.ArgumentParser(prog="band_history.py check")
        ap.add_argument("--root", default=str(ROOT))
        ap.add_argument("--history", default=None)
        ap.add_argument("--quiet", action="store_true", help="print nothing when the tree is clean")
        a = ap.parse_args(argv)
        return check_and_exit(a.root, a.history, a.quiet)

    if mode == "record":
        ap = argparse.ArgumentParser(prog="band_history.py record")
        ap.add_argument("--file", default=GATE_FILE_REL,
                        help="the file the band lives in (default engine/gate.py; for a fixture's"
                             " margin give data/refs/<fixture>/route_allowance.json)")
        ap.add_argument("--key", required=True,
                        help="BAND.<name> for the gate file, margin|margin_rel for an allowance file")
        ap.add_argument("--old", required=True, help="the recorded value, or 'absent' for a new key")
        ap.add_argument("--new", required=True, help="the value the tree now holds, or 'absent'")
        ap.add_argument("--reason", required=True)
        ap.add_argument("--decision", required=True,
                        help="a D-x from progress/2026-10-12/plan-2026-10-12.md §3, or a commit subject")
        ap.add_argument("--by", default=None, help="who is responsible (default: git user)")
        ap.add_argument("--event", default="applied", choices=["applied", "refused", "removed"])
        ap.add_argument("--root", default=str(ROOT))
        a = ap.parse_args(argv)
        return record_main(a)

    if mode == "selftest":
        return selftest_main()

    ap = argparse.ArgumentParser(prog="band_history.py review",
                                 description="the 2026-10-11 git walk over gate-file thresholds")
    ap.add_argument("--since", default="2026-10-11 00:00")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    return review_main(a.since, a.root)


if __name__ == "__main__":
    sys.exit(main())

