#!/usr/bin/env python3
"""M1 wrong-count witness, on the real census line a real binary printed.

`gate.py::mtp_plan_verdicts` is the comparison section (h) runs, so this feeds it the `MTPDRAFT` line and
the `DRAFT round=` line count read out of a run's own log — not a hand-written echo. Two directions:

  clean  — the census as the shipped constants print it, which must be green.
  WRONGCOUNT — the same log with the declared block total incremented by one (what the deliberately
               mis-declared build prints: `block=1+14 per_round=17` while the block really submits 13
               launches). Both the §09 block row and the stage-bill row must go red, and the stage-bill
               row must be the one that catches it when the declaration is internally consistent.

Usage: m1_wrongcount_check.py LOG [LOG...]
"""
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gate", ROOT / "engine" / "gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def read_run(log: pathlib.Path) -> tuple[dict, int]:
    text = log.read_text()
    echoes = [dict(re.findall(r"(\w+)=([^\s]+)", ln))
              for ln in text.splitlines() if ln.startswith("MTPDRAFT ")]
    lines = sum(1 for ln in text.splitlines() if ln.startswith("DRAFT round="))
    if not echoes:
        raise SystemExit(f"{log}: no MTPDRAFT line, so there is no census to adjudicate")
    return echoes[-1], lines  # the last line is the post-run census; the first is the declared plan


def failing(echo: dict, lines: int) -> list[str]:
    return [name for name, ok, _ in gate.mtp_plan_verdicts(echo, lines) if not ok]


bad = 0
for path in sys.argv[1:]:
    log = pathlib.Path(path)
    echo, lines = read_run(log)
    print(f"== {log}")
    print(f"   census: block={echo.get('block')} head={echo.get('head')}"
          f" per_round={echo.get('per_round')} rounds={echo.get('rounds')},"
          f" {lines} `DRAFT round=` line(s)")

    hits = failing(echo, lines)
    print(f"   as printed: {'GREEN' if not hits else 'RED ' + str(hits)}")

    # The same log made internally consistent but still wrong: increment block AND per_round together, so
    # `per_round == block + head` still holds and only the §09 count and the stage bill can catch it. That
    # is the mis-declaration an in-binary `plan_match` cannot see, because both sides of it moved together.
    try:
        copies, launches = (int(x) for x in str(echo["block"]).split("+"))
        consistent = {**echo, "block": f"{copies}+{launches + 1}",
                      "per_round": str(int(echo["per_round"]) + 1)}
    except (KeyError, ValueError):
        consistent = None
    if consistent is not None:
        hits2 = failing(consistent, lines)
        print(f"   internally consistent (+1 on both sides):"
              f" {'GREEN — NOTHING CAUGHT IT' if not hits2 else 'RED ' + str(hits2)}")
        if not hits2:
            bad += 1
print("WRONGCOUNT WITNESS PASS" if bad == 0 else f"WRONGCOUNT WITNESS FAIL ({bad})")
sys.exit(1 if bad else 0)
