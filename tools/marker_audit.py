"""CPU-only audit: every submission inside a measured unit has a stage-clock marker before it.

`engine/stage_clock.h` charges a family by the marker that PRECEDES its launch, so a submission with no
marker ahead of it is not missing from the report — it is silently charged to whatever family was noted
last, which is invisible on a body where most segments are one launch long. That is the failure this looks
for mechanically instead of by someone reading the file and being confident.

Scope is the two functions a measured unit is made of: `submit_wave_body` (which `submit_token_body` is a
call to) and `finish_token`, because those are the spans a clock is armed around. Submissions outside them
(`prepare_weight_adapters`' two adapter casts) are counted and reported as outside-a-unit rather than
failed: they run once before any clock is armed, and calling them a miss would make the audit wrong about
its own subject.

A submission is `kernels::NAME(` for a launcher, or a `hipMemcpy`/`hipMemcpyAsync`. `NAME` counts unless it
is in `NON_LAUNCHERS` — the two `kernels::` helpers that compute a plan or a size and submit nothing, named
rather than guessed at by a shape regex.

The rule checked is the one the clock actually relies on, stated the way the clock sees it: **between two
consecutive submissions, and before the first one in a unit, there is at least one `stageclock::note`.** That
is what a per-family segment is made of. A fixed line window would look equivalent and is not — two adjacent
launches of the same family sit five lines apart, so a window would pass a body with a marker deleted in
front of one of them, which is precisely the injection this must catch.

The negative is `--break-note N`: delete the Nth marker in the file and re-run the check on the patched
text, so the audit is required to fail on a real missing marker instead of passing because it cannot see
anything. Usage:

    python3 tools/marker_audit.py                 # the tree; exit 1 on a miss
    python3 tools/marker_audit.py --list          # every site and its verdict
    python3 tools/marker_audit.py --break-note 7  # the negative: must exit 1
"""
import re
import sys

PATH = "engine/decode_step.cpp"
UNITS = ("void submit_wave_body(", "StepResult finish_token(")
NON_LAUNCHERS = {"resolve_mlp_gate_map", "attn_part_floats"}
SITE = re.compile(r"kernels::([a-z0-9_]+)|hipMemcpyAsync\(|hipMemcpy\(")
NOTE = re.compile(r"stageclock::note\(")


def is_submission(line):
    names = [n for n in SITE.findall(line) if n]
    if not SITE.search(line):
        return False
    if names and all(n in NON_LAUNCHERS for n in names):
        return False
    return True


def unit_spans(lines):
    """The line ranges of the measured-unit bodies, taken from their own column-0 braces."""
    out = []
    for head in UNITS:
        start = next(i for i, line in enumerate(lines) if line.startswith(head))
        depth, i = 0, start
        while True:
            i += 1
            depth += lines[i].count("{") - lines[i].count("}")
            if depth <= 0:
                break
        out.append((start, i + 1))
    return out


def main():
    argv = sys.argv[1:]
    path = argv[argv.index("--path") + 1] if "--path" in argv else PATH
    lines = open(path).read().splitlines()
    if "--break-note" in argv:
        n = int(argv[argv.index("--break-note") + 1])
        seen = 0
        for i, line in enumerate(lines):
            # Real markers only: the file's header comment quotes `stageclock::note(family)` in prose, and
            # deleting a sentence of a comment would be a negative that proves nothing.
            if not NOTE.search(line) or line.strip().startswith(("//", "*", "/*")):
                continue
            seen += 1
            if seen == n:
                print(f"[negative] deleting marker #{n} at {path}:{i + 1}: {line.strip()}")
                del lines[i]
                break
    inside = set()
    for a, b in unit_spans(lines):
        inside.update(range(a, b))

    sites_so_far = [i for i, line in enumerate(lines) if is_submission(line) and i in inside]
    spans = unit_spans(lines)

    def unit_start(i):
        """The line the enclosing unit opens at, so the first submission is judged against the unit."""
        return max(a for a, b in spans if a <= i <= b)

    def marked(i):
        """True when a `stageclock::note` sits after the previous submission and at or before line `i`.

        `i` itself may carry both, which is how the one-line forms read.
        """
        prev = max([s for s in sites_so_far if s < i], default=unit_start(i))
        return any(NOTE.search(lines[j]) for j in range(prev + 1, i + 1))

    sites = misses = outside = 0
    for i, line in enumerate(lines):
        if not is_submission(line):
            continue
        if i not in inside:
            outside += 1
            continue
        sites += 1
        ok = marked(i)
        if "--list" in argv:
            print(f"  {path}:{i + 1}: {line.strip()[:74]:<74} {'marked' if ok else 'UNMARKED'}")
        if not ok:
            misses += 1
            print(f"  MISSING marker: {path}:{i + 1}: {line.strip()[:90]}")
    print(f"{path}: submissions inside a measured unit: {sites}, marked: {sites - misses}, "
          f"outside a unit (not the clock's subject): {outside}")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
