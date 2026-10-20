#!/usr/bin/env python3
"""A12 engine-plumbing admission PREPARATION — commands and pre-flight, not a verdict.

`--attn-fattn-wide` is plumbed and INERT (default 0). Nothing here adjudicates the wide form: this script
enumerates the population, proves each fixture is adjudicable by the harness at all, derives the wide
geometry's edges from the headers that enforce them, says which of those edges the pinned population can
reach and which it cannot, and prints the two arms' exact `engine/gate.py` command lines for the main
agent's device pass. It touches no device and takes no timing.

WHY A SCRIPT AND NOT A LIST IN A DOC. Three gate admissions in this tree claimed a case count off a
hand-typed fixture list and seven of the nine names did not exist (`bench/performance-status.md` §15.2,
D40), so the population here is enumerated from disk with `engine/gate.py`'s OWN discriminator between a
pinned fixture and a carried record — imported, not retyped, because a second copy of a rule is the
two-authorities failure this repo keeps recording. The geometry constants are read out of the headers that
enforce them for the same reason.

Usage:
    .venv/bin/python prototype/attn_wide/a12_engine_admission.py            # census + commands
    .venv/bin/python prototype/attn_wide/a12_engine_admission.py --check    # add the CPU-only adjudicability pass
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFS_DIR = ROOT / "data" / "refs"
GATE = ROOT / "engine" / "gate.py"
WIDE_H = ROOT / "engine" / "attn_wide_kernels.h"
OP_H = ROOT / "prototype" / "attn_decode" / "attn_decode_kernels.h"
FATTN_H = ROOT / "engine" / "attn_fattn_kernels.h"
STEP_H = ROOT / "engine" / "decode_step.h"


def load_gate_module():
    """Import `engine/gate.py` for its carried-record list, rather than copying the list here."""
    spec = importlib.util.spec_from_file_location("engine_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # module level only: gate's `main()` is under __main__
    return mod


def macro(path: Path, name: str) -> int:
    """Read one `#define NAME <integer>` out of a header. A constant quoted into a doc is a constant
    that can drift; the value is taken from the file that enforces it."""
    text = path.read_text()
    m = re.search(r"^#define\s+" + re.escape(name) + r"\s+\(?(\d+)\)?", text, re.M)
    if not m:
        raise SystemExit(f"{path.name}: cannot read `#define {name}`, so the geometry census has no"
                         " constants to derive from. Refusing rather than typing numbers here.")
    return int(m.group(1))


def constexpr(path: Path, name: str) -> int:
    text = path.read_text()
    m = re.search(r"constexpr\s+int\s+" + re.escape(name) + r"\s*=\s*(\d+)", text)
    if not m:
        raise SystemExit(f"{path.name}: cannot read `{name}`, refusing to guess the wave row cap.")
    return int(m.group(1))


def constants() -> dict:
    n_q = macro(OP_H, "N_Q")
    n_kv = macro(OP_H, "N_KV")
    m_rows = macro(FATTN_H, "FATTN_MROWS")
    warps = macro(WIDE_H, "WIDE_WARPS")
    stage = macro(WIDE_H, "WIDE_STAGE")
    c = {
        "N_Q": n_q, "N_KV": n_kv, "GQA": n_q // n_kv,
        "FATTN_MROWS": m_rows,
        "WIDE_WARPS": warps, "WIDE_STAGE": stage,
        "MAXSC": macro(OP_H, "MAXSC"),
        "kMaxWaveRows": constexpr(STEP_H, "kMaxWaveRows"),
    }
    c["WIDE_POS_PER_BLOCK"] = warps * (m_rows // c["GQA"])
    c["WIDE_ROWS_PER_BLOCK"] = warps * m_rows
    return c


def population(gate_mod) -> tuple[list[Path], list[str]]:
    """The fixture population, derived from disk with gate.py's own discriminator.

    A `kind`-carrying manifest with no `pins` is a carried record, not a pin set — the same test
    `refuse_non_pin_fixture` uses to exit 3. Names listed there must still read that way, which is
    gate.py's own derived check; here it is re-checked so this census cannot disagree with the gate.
    """
    pop, carried = [], []
    for d in sorted(REFS_DIR.iterdir()):
        mf = d / "manifest.json"
        if not mf.exists():
            carried.append(f"{d.name}: no manifest.json")
            continue
        man = json.loads(mf.read_text())
        if man.get("kind") is not None and "pins" not in man:
            why = "carried record (kind=%s, no pins)" % man.get("kind")
            if d.name in gate_mod.NOT_A_PIN_FIXTURE:
                why += " — listed in NOT_A_PIN_FIXTURE"
            carried.append(f"{d.name}: {why}")
            continue
        pop.append(d)
    if not pop:
        raise SystemExit(f"population is EMPTY over {REFS_DIR}: a census that read nothing cannot fail."
                         " Check the path before believing any number this script prints.")
    return pop, carried


def geometry(d: Path, c: dict) -> dict:
    """What this fixture presents to the wide block, derived from its manifest and the constants."""
    man = json.loads((d / "manifest.json").read_text())
    rows = len(man.get("prefill_ids") or [])
    steps = max(len(man.get("greedy_continuation") or []) - 1, 0)
    lanes_last = rows + steps            # row qrow attends lvalid+qrow lanes; lvalid=1 for a fresh prompt
    bodies = max(math.ceil(rows / c["kMaxWaveRows"]), 1)
    widest_body = min(rows, c["kMaxWaveRows"])
    g = {
        "fixture": d.name, "rows": rows, "steps": steps, "lanes_last": lanes_last,
        "wave_bodies": bodies,
        "rtiles": math.ceil(widest_body / c["WIDE_POS_PER_BLOCK"]),
        "tail_rows": widest_body % c["WIDE_POS_PER_BLOCK"],
        "rounds_64": math.ceil(lanes_last / c["WIDE_STAGE"]),
        "rounds_16": math.ceil(lanes_last / macro(FATTN_H, "FATTN_MLANES")),
        "streaming": lanes_last > c["MAXSC"],
        "single_round": lanes_last <= c["WIDE_STAGE"],
    }
    g["edges"] = derived_edges(g, c)
    return g


def derived_edges(g: dict, c: dict) -> list[str]:
    """Which wide-geometry edges THIS fixture sits on — derived from the constants, never hand-listed.

    A hand-listed must-hit set rots the moment the fixture changes (the `attn_tile_model.py` lesson); these
    are the boundaries the block's own index arithmetic has to survive: the block tail in positions, the
    round tail in lanes, the wave-row cap, the one-pass/streaming boundary where knob 0 changes route and
    knob 1 does not, and the whole-wave-past-`nrows` clamp.
    """
    e = []
    pos, stage = c["WIDE_POS_PER_BLOCK"], c["WIDE_STAGE"]
    if g["rows"] == 0:
        e.append("no prefill rows: the wide block is never reached (decode-only fixture)")
        return e
    if g["tail_rows"] == 0:
        e.append(f"rows is an exact multiple of POS_PER_BLOCK {pos}: NO tail block")
    elif g["tail_rows"] == 1:
        e.append(f"tail block holds exactly 1 of {pos} positions (narrowest clamp)")
    elif g["tail_rows"] == pos - 1:
        e.append(f"tail block holds {pos - 1} of {pos} positions (widest partial tail)")
    else:
        e.append(f"tail block holds {g['tail_rows']} of {pos} positions")
    if g["rows"] < pos:
        e.append("fewer rows than one wide block: row-slots run past `nrows`, the clamp is load-bearing")
    if g["wave_bodies"] > 1:
        e.append(f"{g['rows']} rows exceeds kMaxWaveRows {c['kMaxWaveRows']}: {g['wave_bodies']} wave bodies")
    if g["lanes_last"] == c["MAXSC"]:
        e.append("widest row is exactly MAXSC: the last one-pass wave the shipping route serves")
    elif g["lanes_last"] == c["MAXSC"] + 1:
        e.append("widest row is MAXSC+1: the first row §07.3 streams, so knob 0 gains a combine knob 1 has")
    elif g["streaming"]:
        e.append("streams above MAXSC: knob 0 merges partials, knob 1 launches no combine at all")
    else:
        e.append("entirely one-pass at MAXSC or below: knob 1 moves rows that are BIT-EXACT at knob 0")
    rem = g["lanes_last"] % stage
    if rem == 0:
        e.append(f"lanes is an exact multiple of the {stage}-position stage: no short round")
    elif rem == 1:
        e.append(f"lanes is 1 past a {stage}-position round: shortest possible round")
    elif rem == stage - 1:
        e.append(f"lanes is 1 short of a {stage}-position round: widest partial round")
    else:
        e.append(f"partial round of {rem} of {stage} positions")
    if g["single_round"]:
        e.append("one round at 64 positions in the wide form (multi-round at 16): the edge where the two"
                 " forms differ only by the weight's rounding, not by re-association")
    return e


EDGES = [
    ("a tail block of exactly 1 position", lambda g, c: g["tail_rows"] == 1 and g["rows"] >= c["WIDE_POS_PER_BLOCK"]),
    ("rows below one whole wide block", lambda g, c: 0 < g["rows"] < c["WIDE_POS_PER_BLOCK"]),
    ("more rows than kMaxWaveRows (multi-body prefill)", lambda g, c: g["wave_bodies"] > 1),
    ("lanes exactly at MAXSC (last one-pass wave)", lambda g, c: g["lanes_last"] == c["MAXSC"]),
    ("lanes at MAXSC+1 (first streamed row)", lambda g, c: g["lanes_last"] == c["MAXSC"] + 1),
    ("a row above MAXSC at all (combine exists at knob 0, not at knob 1)", lambda g, c: g["streaming"]),
    ("every row one-pass (knob 1 moves bit-exact rows)", lambda g, c: g["rows"] > 0 and not g["streaming"]),
    ("lanes an exact multiple of the 64-position stage", lambda g, c: g["lanes_last"] % c["WIDE_STAGE"] == 0),
    ("lanes 1 past a 64-position round", lambda g, c: g["lanes_last"] % c["WIDE_STAGE"] == 1),
    ("one round at 64 positions", lambda g, c: g["single_round"] and g["rows"] > 0),
    (">16 rounds at 64 positions (a long sweep)", lambda g, c: g["rounds_64"] > 16),
]


def gate_command(fixture: str, wide: str, gate_mod) -> str:
    arm = f" --attn-fattn-wide {wide}" if wide == "1" else ""
    return (f".venv/bin/python engine/gate.py --fixture {fixture}"
            f" --binary {gate_mod.DEFAULT_BINARY}{arm}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="also run `engine/gate.py --no-engine --fixture X` for every fixture: CPU-only,"
                         " starts no binary, touches no device, and catches the invocation gaps (a manifest"
                         " the harness itself cannot read) before a device window is spent on them")
    args = ap.parse_args()

    gate_mod = load_gate_module()
    c = constants()
    print("constants read from the files that enforce them (not typed here):")
    for k in ("N_Q", "N_KV", "GQA", "FATTN_MROWS", "WIDE_WARPS", "WIDE_STAGE", "WIDE_POS_PER_BLOCK",
              "WIDE_ROWS_PER_BLOCK", "MAXSC", "kMaxWaveRows"):
        print(f"  {k:20s} = {c[k]}")

    pop, carried = population(gate_mod)
    print(f"\n1. THE POPULATION, derived from {REFS_DIR.relative_to(ROOT)}/*/manifest.json")
    print(f"   {len(pop)} pinned fixtures, {len(carried)} carried or manifest-less directories excluded")
    for line in carried:
        print(f"     excluded  {line}")
    if not pop:                                   # cannot happen (population() raises), kept visible
        raise SystemExit("empty population")

    geo = [geometry(d, c) for d in pop]
    print(f"\n2. WHAT EACH FIXTURE PRESENTS TO THE WIDE BLOCK ({len(geo)} derived rows,"
          f" assert count == population: {len(geo) == len(pop)})")
    print("   fixture            rows steps lanes_last bodies rtiles tail rounds64 stream")
    for g in geo:
        print(f"   {g['fixture']:18s} {g['rows']:5d} {g['steps']:5d} {g['lanes_last']:9d}"
              f" {g['wave_bodies']:6d} {g['rtiles']:6d} {g['tail_rows']:4d} {g['rounds_64']:8d}"
              f" {'yes' if g['streaming'] else 'no':>6s}")

    print("\n3. EDGE COVERAGE, the edge set DERIVED from the constants above")
    missing = []
    for label, pred in EDGES:
        hit = [g["fixture"] for g in geo if pred(g, c)]
        if hit:
            print(f"   COVERED    {label}: {', '.join(hit[:6])}" + (" ..." if len(hit) > 6 else ""))
        else:
            missing.append(label)
            print(f"   UNCOVERED  {label}")
    if missing:
        print("   The uncovered edges above need SYNTHETIC cases, and each is stated as the geometry that"
              " produces it rather than as a fixture name: a prompt of exactly POS_PER_BLOCK+1="
              f"{c['WIDE_POS_PER_BLOCK'] + 1} tokens for the 1-position tail; a body whose widest row is"
              f" MAXSC+1={c['MAXSC'] + 1} lanes for the first streamed row; a prompt of"
              f" {c['kMaxWaveRows'] + 1} tokens for the multi-body case; a row at"
              f" {c['WIDE_STAGE'] + 1} lanes for the shortest second round. Any of those has to be DUMPED"
              " from the reference to become a gate fixture — which is a fixture-list change and therefore"
              " NOT made here, and `tools/reference_dump.py` is the tool that would make it.")

    print("\n4. THE TWO ARMS, exact commands (device runs — the main agent's pass, one gpu_lock hold each;"
          " `engine/gate.py` does NOT self-acquire the lock, so the caller holds it)")
    for g in geo:
        print("   default arm : " + gate_command(g["fixture"], "0", gate_mod))
        print("   wide arm    : " + gate_command(g["fixture"], "1", gate_mod))
    n_cmds = 2 * len(geo)
    print(f"   commands emitted: {n_cmds}, expected 2 x population = {2 * len(pop)}"
          f" -> {'OK' if n_cmds == 2 * len(pop) else 'MISMATCH'}")
    print("   Read every run's own tail before tabulating it: a `PASS=0` line is an unclassified run until"
          " its tail has been read (an error exit from the harness prints no PASS line either), and exit 3"
          " is gate.py's SKIP, which is neither verdict.")

    print("\n5. CPU PRE-FLIGHT (already re-runnable here, no device)")
    print("   .venv/bin/python prototype/attn_wide/a12_wide_mirror.py")
    print("   the mirror's f16@64 arm IS this routing: F16 accumulator, P rounded to F16 un-normalised and")
    print("   divided after widening, rounds of 64 positions over the whole row. Its headline against the")
    print("   PIN is cos >= 0.99999083 for the wide arm against the shipping f32@16 arm's 0.99999068, which")
    print("   is the number to compare against the §10.1 floors — the gate, not this script, does that.")
    print("   The mirror's population is NARROWER than this one by its own rule: it needs a complete §05")
    print("   attention pin set per fixture and prints a SKIP line for each fixture without one, so its")
    print("   fixture count can be below the 18 above. That difference is a fact about which intermediate")
    print("   tensors exist, not about the wide form, and the SKIP list is printed by the mirror itself.")
    if args.check:
        print("\n6. ADJUDICABILITY PRE-FLIGHT (`gate.py --no-engine`, CPU-only)")
        bad = []
        for d in pop:
            r = subprocess.run([sys.executable, str(GATE), "--no-engine", "--fixture", d.name],
                               cwd=ROOT, capture_output=True, text=True)
            tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
            status = {0: "PASS", 3: "SKIP(carried)"}.get(r.returncode, f"EXIT {r.returncode}")
            if r.returncode not in (0, 3):
                bad.append((d.name, r.returncode, tail))
            print(f"   {d.name:18s} {status:14s} {tail[:64]}")
        if bad:
            print(f"   {len(bad)} fixture(s) the harness itself cannot adjudicate. Two things before any of")
            print("   it is read as a wide-knob result: `--no-engine` starts NO binary, so a failure here")
            print("   cannot have been caused by the kernel under test — it is a pin-side or harness gap; and")
            print("   the base witness is HEAD's own gate.py, which costs one command:")
            print("     git show HEAD:engine/gate.py > engine/gate_head_witness.py")
            print("     .venv/bin/python engine/gate_head_witness.py --no-engine --fixture <name>   # then")
            print("     rm engine/gate_head_witness.py                                                # delete it")
            print("   Same failure at HEAD means pre-existing, and it gets reported as an invocation gap")
            print("   (D40), not dropped and not counted as a numerics verdict.")
            return 1
        print(f"   all {len(pop)} adjudicable by the harness at --no-engine")
    print("\nRISK STATEMENT, for whoever reads the result: knob 1 re-associates the softmax at EVERY width")
    print("   (a 64-position round against the shipping 16, and the accumulated weight is an un-normalised")
    print("   F16 exp divided at the end, where the one-pass route rounds the NORMALISED weight to BF16")
    print("   before the P*V), so no bit-identity claim exists anywhere under this knob, including between")
    print("   knob 0 and knob 1. And the per-round accumulator rescale is mandatory: with the fold removed")
    print("   the worst head-row drops its FIRST add at round 1 of a 64-position round and the accumulator is")
    print("   no longer finite in fp16 by round 1721 (a12_wide_mirror.py section 4; the shipping 16-wide")
    print("   arm's own counterfactual is round 1 / 13 / 99 worst-median-best, A10-PVF16.md section 4).")
    print("   No band, floor, must-catch set or fixture list was edited to let this knob exist; if it cannot")
    print("   pass what is already there, that is the answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
