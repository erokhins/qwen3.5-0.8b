#!/usr/bin/env python3
"""STEP 1 of the >MAXSC instrument question: what is admitted, what is priced, which fixtures exercise
rows above MAXSC, and which instrument each is judged by. Derived by directory walk.

    .venv/bin/python prototype/maxsc_allowance/census.py
    .venv/bin/python prototype/maxsc_allowance/census.py --markdown

Why a script and not a table typed into a document: `AGENTS.md` records an admission that cited
"nine canonical fixtures" of which seven names did not exist in `data/refs/`, and `tools/fixture_census.py`
exists because `data/refs/zz` was being read as a numerics result when the fixture contradicted its own
manifest. Every row here is read out of the directory it describes, and the population is
`every data/refs/<dir> with a manifest.json`, so a fixture cannot be forgotten and a name cannot be
invented.

`MAXSC` is read from `prototype/attn_decode/attn_decode_kernels.h` through `engine/gate.py::lane_geometry`,
never retyped. A query row with more than MAXSC lanes to attend is the STREAMING form (§07.3), and the
engine reaches it two different ways, judged by two different instruments:

  * a PREFILL ROW i of a prompt longer than MAXSC (rows MAXSC..L-1 stream). `engine/gate.py` section (i)
    judges those rows against the FLAT §10.1/§10.2 floors and says so in its own rule line: "no route
    allowance is priced for prefill rows (route_allowance.json prices decode steps)". Only a fixture
    dumped with `reference_dump.py --prefill-logits all` carries the row block at all; for the others the
    streaming prefill rows are not judged by anything, because nothing is pinned for them.
  * a DECODE ROW of a fixture whose prompt already exceeds MAXSC: every generation step of such a fixture
    attends prompt_len + step lanes, so every step of it is a streaming row. Sections (b)/(b2) judge those,
    and THAT is the section that consults `route_allowance.json` — so a fixture with no allowance file is
    judged by the flat fallback in the one section where a priced instrument was supposed to be in force.

`lanes_prefill_max` = prefill_ids length (the widest prefill row).
`lanes_decode_max`  = prefill_ids length + (len(greedy_continuation) - 1), the widest row any generation
step reaches; the KV cache is append-only and none of these fixtures wraps it.

Exit 0 always; this is a census, not a gate. A directory it cannot read is reported UNADJUDICABLE with a
reason and still counted, for the reason in `tools/fixture_census.py`'s header: a census that silently
shrinks its own population is the "0 of the readable 55" mistake.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import gate as egate    # the shipped readers: lane_geometry, NOT_A_PIN_FIXTURE, ALLOWANCE_FILE

REFS = ROOT / "data" / "refs"


def row_for(d: Path) -> dict:
    """One fixture directory, read entirely off its own files."""
    r: dict = {"fixture": d.name, "kind": "UNADJUDICABLE"}
    try:
        man = json.loads((d / "manifest.json").read_text())
    except Exception as e:                       # noqa: BLE001 — a census names a failure, it does not raise
        r["note"] = f"manifest unreadable ({type(e).__name__}: {e})"
        return r
    r["note"] = ""
    if man.get("kind") == "corpus" or "pins" not in man:
        r["kind"] = "corpus record"
        r["note"] = (gate_says(d) or "manifest carries no `pins` and the name is NOT in"
                     " engine/gate.py::NOT_A_PIN_FIXTURE, so `--fixture` has nothing to pair against")
        r["n_npy"] = len(list(d.glob("*.npy")))
        return r
    ids = man.get("prefill_ids") or []
    cont = man.get("greedy_continuation") or []
    n_steps = max(len(cont) - 1, 0)
    r.update({
        "kind": "pin",
        "prompt_len": len(ids),
        "n_steps": n_steps,
        "lanes_prefill_max": len(ids),
        "lanes_decode_max": (len(ids) + n_steps) if n_steps else 0,
        "n_npy": len(list(d.glob("*.npy"))),
        "logit_rows_file": (d / "logits_rows.npy").exists(),
        "prefill_rows_pinned": "prefill_logit_rows" in man,
        "replayed": n_steps > egate.FREE_RUN_BOUND,
    })
    ap = d / egate.ALLOWANCE_FILE
    r["has_allowance"] = ap.exists()
    if ap.exists():
        try:
            ad = json.loads(ap.read_text())
            r["margin"], r["margin_rel"], r["margin_whence"] = egate.allowance_margins(ad)
            r["margin_basis"] = ("own (per-fixture, D-2)" if ad.get("margin_basis")
                                 else "carried (pre-D-2 global max)")
            r["allowance_steps"] = len(ad.get("steps", []))
        except Exception as e:                   # noqa: BLE001
            r["margin"] = r["margin_rel"] = None
            r["margin_basis"] = f"UNREADABLE ({type(e).__name__})"
    return r


def gate_says(d: Path) -> str:
    return egate.NOT_A_PIN_FIXTURE.get(d.name, "")


def judged_by(r: dict, maxsc: int) -> str:
    """Which rule each judging section applies to this fixture, given what is priced on disk."""
    if r["kind"] != "pin":
        return r["note"] or r["kind"]
    parts = []
    if r["n_steps"] == 0:
        parts.append("no decode steps pinned (prefill-only fixture)")
    elif r["lanes_decode_max"] > maxsc:
        parts.append(f"DECODE steps stream ({r['lanes_decode_max']} lanes) -> "
                     + ("priced route allowance" if r["has_allowance"]
                        else "FLAT FALLBACK (§10.1 G5 / §10.2), no allowance priced"))
    else:
        parts.append(f"decode steps one-pass ({r['lanes_decode_max']} lanes)")
    if r["lanes_prefill_max"] > maxsc:
        parts.append(f"PREFILL rows stream ({r['lanes_prefill_max']}) -> flat floors, always"
                     + (" (row block pinned, section (i) can judge them)" if r["logit_rows_file"]
                        else " — and NO row block pinned, so section (i) judges NOTHING above MAXSC here"))
    return "; ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = ap.parse_args()
    staging, maxsplit, ceiling = egate.lane_geometry()

    dirs = sorted(p for p in REFS.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    rows = [row_for(d) for d in dirs]

    hdr = ["fixture", "kind", "prompt", "steps", "lanes prefill", "lanes decode",
           "allowance", "margin", "margin_rel", "margin basis", "steps priced", "judged by"]
    out = []
    for r in rows:
        pin = r["kind"] == "pin"
        out.append([
            r["fixture"], r["kind"],
            str(r.get("prompt_len", "")), str(r.get("n_steps", "")),
            str(r.get("lanes_prefill_max", "")), str(r.get("lanes_decode_max", "")),
            ("yes" if r.get("has_allowance") else "NO") if pin else "—",
            (f"{r['margin']:.2f}" if r.get("margin") else "—") if pin else "—",
            (f"{r['margin_rel']:.2f}" if r.get("margin_rel") else "—") if pin else "—",
            r.get("margin_basis", "—") if pin else "—",
            str(r.get("allowance_steps", "")) if pin else "—",
            judged_by(r, staging),
        ])

    if args.markdown:
        print(f"`MAXSC = {staging}` (from `prototype/attn_decode/attn_decode_kernels.h`, read by"
              f" `engine/gate.py::lane_geometry`), `kAttnMaxSplit = {maxsplit}`, lane ceiling"
              f" `= {ceiling}` lanes. Population: every `data/refs/<dir>/` carrying a `manifest.json`,"
              f" {len(dirs)} of them, enumerated by directory walk at run time.")
        print()
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
        for c in out:
            print("| " + " | ".join(c) + " |")
    else:
        print(f"MAXSC={staging} kAttnMaxSplit={maxsplit} ceiling={ceiling} lanes;"
              f" population {len(dirs)} directories")
        for c in out:
            print("  " + " | ".join(c))

    pins = [r for r in rows if r["kind"] == "pin"]
    n_allow = sum(1 for r in pins if r.get("has_allowance"))
    over_dec = [r["fixture"] for r in pins if r["lanes_decode_max"] > staging]
    over_dec_raw = [r["fixture"] for r in pins if r["lanes_decode_max"] > staging
                    and not r["has_allowance"]]
    over_pre = [r["fixture"] for r in pins if r["lanes_prefill_max"] > staging]
    over_pre_pinned = [r["fixture"] for r in pins if r["lanes_prefill_max"] > staging
                       and r["logit_rows_file"]]
    print(f"\npopulation {len(rows)} directories under data/refs/ with a manifest.json: {len(pins)} pin"
          f" fixtures, {len(rows) - len(pins)} corpus records named by"
          f" engine/gate.py::NOT_A_PIN_FIXTURE. Priced route allowances: {n_allow} of {len(pins)}.")
    print(f"DECODE rows above MAXSC ({staging}): {len(over_dec)} of {len(pins)} fixtures ->"
          f" {over_dec}")
    print(f"  ...of those, NO allowance file (so sections (b)/(b2) fall back to the flat floors, the"
          f" section the allowance exists to serve): {over_dec_raw or 'none'}")
    print(f"PREFILL rows above MAXSC: {len(over_pre)} of {len(pins)} fixtures -> {over_pre}")
    print(f"  ...of those, carrying a `logits_rows.npy` pin block (the only ones section (i) can judge"
          f" at all): {over_pre_pinned or 'none'}")
    print("  Section (i)'s own rule line says the prefill-row verdict is the flat floors BY DESIGN:"
          " `route_allowance.json` prices DECODE STEPS, not prefill rows — pricing a wide fixture does"
          " not change what section (i) applies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
