#!/usr/bin/env python3
"""D3 — `delta`, the cost of verifying TWO query rows per decode step instead of ONE.

CPU-only. It takes no timing and touches no device: it (a) DERIVES the component list from
`tools/step_traffic_model.py`'s traffic census, (b) folds in rig measurements if the runner has produced
them, and (c) prints the verdict against the kill line from `prototype/mtp_gatezero/route_price.py`.

Why the component list has to be derived. AGENTS.md: a must-catch set hand-listed from memory rots into
"everything", and three gate admissions this project claimed from a list nobody enumerated were wrong.
The population here is the traffic census: `step_traffic_model.rows()` emits 33 tagged rows
(family, direction, bytes, weight-or-state) for one token, read out of `engine/state.h`. This file
re-reads it at runtime and assigns every row a scaling class from a rule table keyed on a unique
substring of the model's own note, then REQUIRES an exact 1:1 cover. A new census row is a hard error
here, not a silent omission; `--selftest` injects a fake row and proves the classifier rejects it, and
proves it accepts a real one.

One caveat carried, not hidden: `rows(k, lanes)` is written for n = 1 query row — its `--lanes` is KV
lanes, not rows — so it cannot emit a 2-row census by itself. The scaling class below is the extra
information; the bytes it produces are demanded-bytes (what a second row asks the memory system for),
and per the AGENTS traffic row, demanded bytes are not DRAM bytes.

Every ms figure is QUOTED from `engine/generation-limits.md` §1 (4,032 lanes, paired window) unless it
came from an evidence file written by `d3_run_locked.sh`, in which case it is labelled RIG. The verdict
is an ESTIMATE OF `delta` FROM RIG COMPONENTS — it is not a measurement of the engine's verify path,
which cannot exist until the draft path is built (`decode_run` has no 2-row-body switch).

    .venv/bin/python prototype/mtp_delta/components.py [--selftest] [--evidence DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "prototype" / "mtp_gatezero"))

from instep_family_budget import state_constants          # noqa: E402
import step_traffic_model as STM                          # noqa: E402
import route_price as RP                                  # noqa: E402 (shared constants, one spelling)

LANES = 4032                       # the standing operating point (AGENTS: decode is reported at 4k)

# QUOTED: engine/generation-limits.md §1, family stage lines at 4,032 lanes (pre-cast-fold values, as
# that file states; they summed to 11.175 ms and the step now measures 10.9357 ms).
FAMILY_MS = {
    "mlp": 3.3628, "proj": 2.7578, "lm_head": 2.1685, "attn": 1.6121, "cast": 0.0,
    "gdn": 0.6458, "glue": 0.5593, "host_tail": 0.7163, "copy": 0.2141, "norms": 0.0,
}
# The one traffic row whose class depends on how the engine batches the head: a second M=1 pass re-reads
# the tied table (per_row), one batched M=2 pass does not (per_step). --selftest asserts exactly one row
# carries this class and that it is this one.
FORM_DEPENDENT = "one row of the tied 248320x1024 bf16 table"

# (unique substring of the census note, class, why). Classes: per_step | per_row | form.
SCALING_RULES = [
    ("24 layers x (gate+up+down)", "per_step", "weight table; one read per launch whatever n is"),
    ("18 GDN layers: qkv/z/a/b in", "per_step", "weight table"),
    ("6 attn layers: q/k/v in", "per_step", "weight table"),
    (FORM_DEPENDENT, "form", "batched head: read once per body. Two M=1 passes: read twice."),
    ("input/post-attn norms", "per_step", "tiny per-launch weight reads, once per launch"),
    ("recurrent state [18][16][128][128]", "per_row",
     "the recurrence re-reads S per token (pass A + pass B); a 2-row chain walks it twice, "
     "the second walk warm (state is 18.9 MiB, under the 32 MiB fast tier)"),
    ("conv1d weight read as F32", "per_step", "widened once into g_conv_w per step"),
    ("conv window (k_conv-1) rows", "per_row", "conv reads the window plus one row per query row"),
    ("roll-back D2D copy reads conv_new", "per_row", "the ring must be restorable per token"),
    ("conv output tile u", "per_row", "per-token activation tile"),
    ("gated_norm reads o (F32)", "per_row", "per-token activation tile"),
    ("the 97 cast inputs", "per_row", "per-token tiles — but `cast` is 0 ms since the fold, so 0 cost"),
    ("GEMV x inputs as F32", "per_row", "one x row per query row"),
    ("mlp_down reads mlp_act_f32", "per_row", "one act row per query row"),
    ("add+norm reads s, delta, weight", "per_row", "glue reads one more row, launch count unchanged"),
    ("attn_prep reads q_raw + k_raw", "per_row", "one staged row per query row"),
    ("KV read at", "per_row", "every query row walks every lane it may attend (unique KV is shared, "
                               "the DEMAND is not)"),
    ("A_log + dt_bias", "per_step", "per-launch weight read"),
    ("gated-norm weight [128]", "per_step", "per-launch weight read"),
    ("rec_kernel writes S back", "per_row", "one more state write, same warm tier"),
    ("conv_kernel writes conv_new", "per_row", "one more window row"),
    ("roll-back D2D copy writes the window", "per_row", "see its read twin"),
    ("u tile + g/beta stage rows", "per_row", "per-token stage"),
    ("gated_norm BF16 out", "per_row", "per-token stage"),
    ("the 97 cast outputs", "per_row", "per-token tiles; 0 ms since the fold"),
    ("GEMV BF16 outputs", "per_row", "one output row per query row"),
    ("act (BF16) + delta per layer", "per_row", "one output row per query row"),
    ("s_out + norm_out per glue launch", "per_row", "one output row per glue launch"),
    ("q_rope, k_rope, gate, gated stage tiles", "per_row", "one stage row per query row"),
    ("logits [248320] F32", "per_row", "a verify step needs BOTH rows' logits (row 1 to accept, "
                                       "row 2 to emit)"),
    ("the same logits read back by the D2H", "per_row", "one D2H per logits row"),
    ("KV append: one lane", "per_row", "each query row appends a lane"),
    ("embedding row copy", "per_row", "each query row looks up its own tied-table row"),
]


def census(k, lanes=LANES):
    """(rows, uncovered, ambiguous): the derived classification, with coverage as a hard gate."""
    rows = STM.rows(k, lanes)
    out, uncovered, ambiguous = [], [], []
    for fam, d, b, wt, note in rows:
        hits = [r for r in SCALING_RULES if r[0] in note]
        if len(hits) == 0:
            uncovered.append((fam, d, b, note))
            continue
        if len(hits) > 1:
            ambiguous.append((fam, d, b, note, [h[0] for h in hits]))
            continue
        key, cls, why = hits[0]
        out.append({"family": fam, "dir": d, "bytes": b, "wt": wt, "note": note,
                    "class": cls, "rule": key, "why": why})
    return out, uncovered, ambiguous


def split_by_class(rows_):
    one = sum(r["bytes"] for r in rows_)
    two = sum(r["bytes"] * (2 if r["class"] == "per_row" else 1) for r in rows_)
    per_row = sum(r["bytes"] for r in rows_ if r["class"] == "per_row")
    form = sum(r["bytes"] for r in rows_ if r["class"] == "form")
    return one, two, per_row, form


def by_family(rows_, cls):
    agg = {}
    for r in rows_:
        if cls and r["class"] != cls:
            continue
        agg[r["family"]] = agg.get(r["family"], 0) + r["bytes"]
    return agg


# --------------------------------------------------------------------- rig evidence

RESULT_KEYS = {
    "proj": ("ms_per_plan", "tokens"),
    "proj_i1024": ("ms_per_plan", "tokens"),
    "mlp_gateup": ("ms", "tokens"),
    "mlp_down": ("ms", "tokens"),
    "attn": ("ms_per_plan", "core_grid"),   # RESULT echoes core_grid; `rows=` lives on the plan echo
}


def read_evidence(d: Path):
    """One dict per rig run: {component, arm, ms, canary, nresults, file}.

    The runner stores the rig's OWN stdout per arm, and an mlp arm's file carries one RESULT line per
    sampled layer, so ms values are SUMMED here rather than last-writer-wins. Two strengthenings that
    follow this repo's rules: a file whose RESULT lines disagree on the canary field is an error (the
    wiring moved, not the row count), and a VOID / missing ms is reported as no-data rather than 0.
    """
    out = []
    for p in sorted(d.glob("*.txt")):
        txt = p.read_text(errors="replace")
        results = [ln for ln in txt.splitlines() if ln.startswith("RESULT")]
        if not results:
            out.append({"file": p.name, "error": "no RESULT line"})
            continue
        parts = p.stem.split("-")           # <pass>-<component>-<arm>
        if len(parts) < 3:
            out.append({"file": p.name, "error": "filename is not pass-component-arm"})
            continue
        comp, arm = parts[1], parts[-1]
        ms_field, canary_field = RESULT_KEYS.get(comp, ("ms", None))
        total, voids, canaries = 0.0, 0, set()
        for ln in results:
            fields = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
            v = fields.get(ms_field)
            if v in (None, "VOID"):
                voids += 1
                continue
            total += float(v)
            if canary_field and canary_field in fields:
                canaries.add(fields[canary_field])
        rec = {"file": p.name, "component": comp, "arm": arm, "ms": total or None,
               "void": voids, "nresults": len(results), "header": txt.splitlines()[0]}
        if len(canaries) > 1:
            rec["error"] = (f"RESULT lines disagree on {canary_field}: {sorted(canaries)} — the arms "
                            f"are not what the file names claim")
        rec["canary"] = sorted(canaries)
        out.append(rec)
    return out


def ratio(ms_a, ms_b):
    return None if not ms_a or not ms_b else ms_b / ms_a


# --------------------------------------------------------------------- pricing

def price(evidence, a_accept, quiet=False):
    """Fold rig ms into per-family deltas.

    Returns {name: (ms, how, kind, role)} with role one of
      planA   - enters the plan-A total (batched body, batched head)
      control - a decomposition of a planA term, already inside it, never added
      witness - a self-check on the reading of a kernel, carries no ms of its own
      bound   - an alternative pessimistic arm, printed beside the total, never summed into it
    """
    by = {}
    for e in evidence:
        if e.get("ms"):
            by.setdefault(e["component"], {})[e["arm"]] = e["ms"]
    k = state_constants()
    rows_, _, _ = census(k)
    per_row_bytes = by_family(rows_, "per_row")
    out = {}

    def put(name, ms, how, kind, role):
        out[name] = (ms, how, kind, role)

    # proj: the rig plan IS the family's 48-launch plan, so a relative marginal transfers directly.
    # proj: the rig plan IS the family's 48-launch plan at engine shapes, so a RELATIVE marginal
    # transfers directly (the absolute rig ms would be a different working set, the ratio is not).
    if "proj" in by:
        p = by["proj"]
        r = ratio(p.get("t1"), p.get("t2"))
        if r:
            put("proj", FAMILY_MS["proj"] * (r - 1.0),
                f"rig ratio ms(tokens=2)/ms(tokens=1)={r:.4f} over the 48-launch engine plan, applied "
                f"to the quoted 2.7578 ms", "measured", "planA")
    # The tile/occupancy control: the SAME 24 in=1024 launches at one token, with only the staged x
    # tile widened (shared 8 KiB -> 32 KiB). Splits proj's marginal into "more dots" and "fatter tile".
    if "proj_i1024" in by and len(by["proj_i1024"]) >= 2:
        c1 = by["proj_i1024"].get("tpb1")
        c4 = by["proj_i1024"].get("tpb4")
        c = ratio(c1, c4)
        if c:
            ng, na = k["kNumGdnLayers"], k["kNumAttnLayers"]
            w1024 = (ng * (k["kQkvRows"] + k["kValueDim"] + 2 * k["kNumValueHeads"])
                     + na * (k["kQprojRows"] + 2 * k["kKvProjRows"])) * k["kHidden"]
            w2048 = (ng * k["kValueDim"] + na * k["kQwidth"]) * k["kHidden"]
            share = w1024 / (w1024 + w2048)
            put("proj_tpb4_control", FAMILY_MS["proj"] * share * (c - 1.0),
                f"same 24 in=1024 launches, one token, staged x tile 4x wider (shared 8 KiB -> 32 "
                f"KiB): rig ratio {c:.4f} over {share:.3f} of the proj bytes. Part of proj's marginal "
                f"is occupancy rather than dots, and that term is already inside the proj row above, "
                f"so this line is NOT added", "measured-control", "control")
    # mlp: gateup [3584,1024] at TPB=4 over-computes (TPB dots for nt tokens), down [1024,3584] at
    # TPB=2 is exact. Split the family by weight bytes, which are 2:1 by shape, derived not typed.
    gb = 2 * k["kNumLayers"] * 2 * k["kIntermediate"] * k["kHidden"] * 2
    db = 2 * k["kNumLayers"] * k["kHidden"] * k["kIntermediate"] * 2
    if "mlp_down" in by:
        d = by["mlp_down"]
        r = ratio(d.get("t1"), d.get("t2"))
        put("mlp_down", FAMILY_MS["mlp"] * db / (gb + db) * (r - 1.0),
            f"exact 2-token tile (TPB=2, nt=2): rig ratio {r:.4f}; share of the mlp line from shape "
            f"bytes = {db / (gb + db):.4f}", "measured", "planA")
    if "mlp_gateup" in by:
        g = by["mlp_gateup"]
        bound = ratio(g.get("t1"), g.get("t2"))
        lin = None
        if g.get("t4") and g.get("t1"):
            lin = (g["t4"] - g["t1"]) / (3.0 * g["t1"])
        share = FAMILY_MS["mlp"] * gb / (gb + db)
        # The witness is not decoration: `(ms(t4)-ms(t1))/3` is only the per-dot marginal if tokens=2
        # and tokens=4 cost the same (the kernel runs TPB=4 dots either way). If they differ by more
        # than a timing window's spread, that reading is wrong and the linear number is NOT used -
        # the upper bound goes into plan A instead, which is the defensible figure.
        WITNESS_TOL = 0.05
        witness_ok = bool(g.get("t2") and g.get("t4")
                          and abs(g["t4"] - g["t2"]) / g["t2"] <= WITNESS_TOL)
        put("mlp_gateup_bound", share * (bound - 1.0),
            f"UPPER bound on the gateup marginal: mlp_gateup_tokens runs TPB=4 dots for nt=2 tokens "
            f"(no nt break in the accumulate loop), rig ratio {bound:.4f}. An alternative to "
            f"mlp_gateup_linear, never additive with it", "measured-upper", "bound")
        if lin is not None and witness_ok:
            put("mlp_gateup_linear", share * lin,
                f"(ms(t4)-ms(t1))/3, the per-extra-dot marginal at the same tile and grid; the witness "
                f"below reads equal, so this reading of the kernel is admissible", "measured-linear",
                "planA")
        elif lin is not None:
            put("mlp_gateup_linear", share * (bound - 1.0),
                f"the linear estimate is VOID (the witness differs by more than {WITNESS_TOL:.0%}), so "
                f"plan A carries the upper bound instead: {(bound - 1.0):.4f} of the gateup share",
                "measured-upper-after-witness", "planA")
        if g.get("t2") and g.get("t4"):
            put("witness_gateup", 0.0, f"ms(t2)={g['t2']:.4f} vs ms(t4)={g['t4']:.4f} (rel diff "
                f"{abs(g['t4'] - g['t2']) / g['t2']:.4%}, tolerance {WITNESS_TOL:.0%}) -> "
                + ("equal within spread: the TPB-dots reading of the kernel is confirmed and the "
                   "linear estimate is admissible" if witness_ok else
                   "NOT equal: the TPB-dots reading is wrong or the window was noisy; the bound stands"),
                "witness", "witness")
    # attn: a ratio again, because the rig's absolute is a different form from the engine's wave core.
    if "attn" in by:
        a = by["attn"]
        r = ratio(a.get("n1"), a.get("n2"))
        put("attn", FAMILY_MS["attn"] * (r - 1.0),
            f"rig ratio ms(nrows=2)/ms(nrows=1)={r:.4f} at lanes_last 4,032/4,033 applied to the "
            f"quoted 1.6121 ms. TRANSFER: the rig's split/combine attn_core wave, not the engine's "
            f"fattn/wide wave core (A12 geometry risk, see handover §4)", "measured-transfer", "planA")
    put("attn_whole_pass", FAMILY_MS["attn"],
        "PESSIMISTIC alternative: the second row costs a whole extra attention pass. The arm to fall "
        "back on if the rig ratio does not transfer — bytes cannot price this family (13 % of its own "
        "floor), which is exactly what the brief warned about", "bound-quoted", "bound")
    put("mlp_gateup_family", FAMILY_MS["mlp"] * gb / (gb + db),
        "floor-of-floors for the gateup term with NO evidence at all: the marginal cannot exceed the "
        "family line itself, i.e. gateup doubles outright. Used by the pessimistic arm only when the "
        "gateup rig arms are missing", "bound-quoted", "bound")
    # Not measurable here: bounds, labelled, from the quoted lines and the census bytes.
    put("lm_head_two_passes", FAMILY_MS["lm_head"],
        "ARM A: two sequential M=1 passes. The second tied-table pass IS a cold pass (D1 measured "
        "2.127-2.221 ms back-to-back) — not assumed", "bound-quoted", "bound")
    r_proj = None
    if "proj" in by:
        r_proj = ratio(by["proj"].get("t1"), by["proj"].get("t2"))
    if r_proj:
        put("lm_head_batched", FAMILY_MS["lm_head"] * (r_proj - 1.0),
            f"ARM B: one batched pass, so the extra is the second dot plus a second 993 KB logits row. "
            f"TRANSFER: the measured proj in=1024 ratio {r_proj:.4f}, same kernel class (one wavefront "
            f"per row, bf16, k=1024, weight read once per tile). lm_head_gemv has no --tokens and "
            f"gemm_algo_sweep has no [248320,1024] shape or band, so this is a transfer, not a "
            f"measurement", "estimate-transfer", "planA")
    else:
        put("lm_head_batched", float("nan"), "needs the proj rig ratio to price ARM B", "pending",
            "planA")
    put("gdn", 0.1243 + STM.ms(max(0.0, per_row_bytes.get("gdn", 0) - 2 * 18.8744e6), 237.0),
        "quoted per-token recurrence share (0.1243 ms of the quoted 0.6458 gdn line — the state walk is "
        "serial per token so it is paid again, warm) + the per-row gdn NON-state bytes at the 237 GB/s "
        "plateau. The state's own 2 x 18.9 MB is subtracted from the byte term because it is already "
        "inside the 0.1243 ms figure; charging it in both places is the double-count this line exists to "
        "avoid. gdn_decode --tokens does NOT batch `rec`, so this is bounded, not measured",
        "bound-quoted", "planA")
    put("glue", STM.ms(per_row_bytes.get("glue", 0), 237.0),
        "launch count is unchanged in a 2-row body, so only the extra row bytes count", "bound-quoted",
        "planA")
    put("copy", STM.ms(per_row_bytes.get("copy", 0), 237.0), "same reasoning as glue",
        "bound-quoted", "planA")
    put("host_tail", k["kVocab"] * 4 / 62.8e9 * 1e3 + 0.007,
        "a second logits row: 993,280 B D2H at the quoted 62.8 GB/s device->host rate, + ~7 us for the "
        "second argmax (§07.4 measured fusing argmax at ~1 %). host_tail is serial per token, not hidden",
        "bound-quoted", "planA")
    put("launches", 0.0, "0 by construction: a 2-row body adds NO launches (charging a second step's "
        "~340 x 3.10 us ~ 1.05 ms is the double-count), and `cast` is deleted so it moves either way",
        "structural-zero", "planA")
    return out


def verdicts(a):
    dmax = (1 + a) / (RP.BAR * (1 + 0.2237)) - 1.0     # route_price's own arithmetic, c from D1
    return dmax


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence", default=str(HERE / "evidence"))
    ap.add_argument("--accept", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    a = args.accept
    if a is None:
        a = json.loads((ROOT / "prototype" / "mtp_gatezero" / "draft_survival.json").read_text())["accept"]
    k = state_constants()
    rows_, uncov, amb = census(k)
    if uncov or amb:
        print("CENSUS CLASSIFICATION INCOMPLETE — refusing to price.")
        for f, d, b, n in uncov:
            print(f"  UNCLASSIFIED: {f} {d} {b/1e6:.4f} MB  {n[:80]}")
        for f, d, b, n, keys in amb:
            print(f"  AMBIGUOUS ({len(keys)} rules): {f} {d} {b/1e6:.4f} MB  {n[:60]}")
        return 1

    one, two, per_row, form = split_by_class(rows_)
    dmax = verdicts(a)
    print(f"== the component census, DERIVED from tools/step_traffic_model.py at {LANES} lanes ==")
    print(f"{len(SCALING_RULES)} rules covered {len(rows_)} census rows, 1:1, no gaps, no double-hits")
    print(f"one query row: {one/1e6:9.3f} MB demanded   two query rows: {two/1e6:9.3f} MB"
          f"   (+{per_row/1e6:.3f} MB per-row traffic)")
    print(f"form-dependent (the tied table, once per body iff the head is batched): {form/1e6:.3f} MB")
    print("\nper family: bytes a SECOND query row demands again, and its price at the 237 GB/s plateau")
    print(f"{'family':10} {'per-row MB':>11} {'byte floor ms':>14} {'quoted family ms':>17}")
    pr = by_family(rows_, "per_row")
    for fam in sorted(FAMILY_MS, key=lambda f: -FAMILY_MS[f]):
        print(f"{fam:10} {pr.get(fam, 0)/1e6:11.3f} {STM.ms(pr.get(fam, 0), 237.0):14.4f}"
              f" {FAMILY_MS[fam]:17.4f}")

    ev_path = Path(args.evidence)
    evidence = read_evidence(ev_path) if ev_path.is_dir() else []
    usable = [e for e in evidence if e.get("ms")]
    errs = [e for e in evidence if e.get("error")]
    print(f"\n== rig evidence: {len(evidence)} file(s) in {ev_path}, {len(usable)} with a ms ==")
    for e in usable:
        print(f"  {e['file']:34} {e['component']}/{e['arm']:6} ms={e['ms']:.6f}"
              f" over {e['nresults']} RESULT line(s) canary={e['canary']}")
    for e in evidence:
        if e.get("error"):
            print(f"  {e['file']}: ERROR {e['error']}")
        elif not e.get("ms"):
            print(f"  {e['file']}: no usable ms ({'VOID' if e.get('void') else e.get('error', 'n/a')})")
        elif e.get("void"):
            print(f"  {e['file']}: {e['void']} of {e['nresults']} RESULT lines read VOID")

    priced = price(evidence, a)
    if errs:
        print(f"\nREFUSING to price: {len(errs)} evidence file(s) failed their own consistency check "
              f"above. A mis-wired arm is worse than a missing one.")
        return 1
    print(f"\n== components of delta (a = {a:.4f}; step {RP.STEP_MS} ms quoted; kill line "
          f"delta_max = {dmax:.3f} of a step = {dmax * RP.STEP_MS:.3f} ms) ==")
    print(f"{'component':22} {'ms':>8} {'of step':>8}  role     how")
    total, pending = 0.0, []
    for name, (ms, how, kind, role) in priced.items():
        if ms != ms:                                   # nan = pending rig evidence
            pending.append(name)
            print(f"{name:22} {'pending':>8} {'-':>8}  {role:8}  {how}")
            continue
        mark = " SIGMA" if role == "planA" else ""
        if role in ("planA", "control", "witness"):
            if role == "planA":
                total += ms
            print(f"{name:22} {ms:8.4f} {ms / RP.STEP_MS:8.4f}  {role:8}{mark} {kind}: {how}")
    print("\nalternative arms and bounds (printed beside the total, NOT summed into it):")
    for name, (ms, how, kind, role) in priced.items():
        if role == "bound" and ms == ms:
            print(f"{name:22} {ms:8.4f} {ms / RP.STEP_MS:8.4f}  {role:8}  {kind}: {how}")
    # The pessimistic arm keeps the plan-A terms that have no worse alternative and swaps the three
    # that do: gateup linear -> its upper bound, batched head -> two M=1 passes, attn ratio -> a
    # whole extra pass. Derived from the swap table so the two totals cannot silently disagree.
    swap = {"mlp_gateup_linear": ("mlp_gateup_bound", "mlp_gateup_family"),
            "lm_head_batched": ("lm_head_two_passes",),
            "attn": ("attn_whole_pass",)}
    pessimistic = total
    for out_, alts in swap.items():
        v_out = priced.get(out_, (float("nan"),))[0]
        v_in = float("nan")
        for alt in alts:                     # first bound that is actually available
            cand = priced.get(alt, (float("nan"),))[0]
            if cand == cand:
                v_in = cand
                break
        if v_in != v_in:
            continue
        # A missing measurement (nan) contributes 0 to plan A, so its bound has to be ADDED whole in
        # the pessimistic arm rather than skipped - otherwise the bounds-only print would read sunnier
        # than the evidence-free truth.
        pessimistic += v_in - (v_out if v_out == v_out else 0.0)
    if pending:
        print(f"\n(pending rig evidence, so BOTH totals are incomplete: {pending})")
    # A verdict from an incomplete sum is the empty-parse failure in a new costume: a check whose input
    # can silently be missing must not print a PASS. Every term that needs rig evidence is named, and
    # no speedup is printed until all of them are present.
    need = [n for n in ("proj", "mlp_down", "mlp_gateup_linear", "attn", "lm_head_batched",
                        "gdn", "glue", "copy", "host_tail")
            if priced.get(n, (float("nan"),))[0] != priced.get(n, (0,))[0]]
    print(f"\nPLAN-A HEADLINE (batched body, batched head, linear gateup marginal): delta ~ {total:.4f}"
          f" ms = {total / RP.STEP_MS:.4f} of a step")
    print(f"PESSIMISTIC ARM (batched body, head run twice, a whole extra attn pass, gateup at its "
          f"upper bound): delta ~ {pessimistic:.4f} ms = {pessimistic / RP.STEP_MS:.4f} of a step")
    print("TWO-SEQUENTIAL-1-ROW-BODIES ARM: delta ~ {0:.4f} ms = 1.000 of a step -> KILLED. The route "
          "therefore REQUIRES a batched 2-row body: a verify path built as two one-row bodies is dead "
          "on the verify side whatever this harness measures.".format(RP.STEP_MS))
    if need:
        print("NO-VERDICT: delta is incomplete, so no speedup is printed. Terms still needing rig "
              f"evidence: {need}. (A verdict computed from a partial sum is how an empty parse reports "
              "PASS.) The bounds-only pessimistic figure above is still the honest worst case, because "
              "it substitutes a whole extra pass wherever a rig term is missing.")
        return 0
    for delta, tag in ((total, "plan A      "), (pessimistic, "pessimistic ")):
        d = delta / RP.STEP_MS
        sp = (1 + a) / ((1 + d) * (1 + 0.2237))
        print(f"  {tag}: speedup {sp:.3f}x  {(sp - 1) * RP.STEP_MS:+.3f} ms/step"
              f"  {'WIN (>= 1.027x bar)' if sp >= RP.BAR else 'KILLED (below 1.027x bar)'}"
              f"   [estimate of delta from rig components, not a verify-path measurement]")
    print(f"\nEvery number above is an ESTIMATE FROM RIG COMPONENTS. The engine's verify path does not"
          f"\nexist yet; nothing here measures it. See handover §4 for what is double-counted and what"
          f"\nis missing (state checkpoint/rollback ~0.12 ms per round is deliberately OUTSIDE delta).")
    return 0


def selftest() -> int:
    """The classifier must be able to fail. A check whose input list cannot be empty cannot fail."""
    k = state_constants()
    rows_, uncov, amb = census(k)
    bad = []
    if uncov or amb:
        bad.append(f"real census did not classify cleanly: {len(uncov)} uncovered, {len(amb)} ambiguous")
    if len(rows_) != len(SCALING_RULES):
        bad.append(f"{len(rows_)} census rows vs {len(SCALING_RULES)} rules: the 1:1 cover is the gate")
    form_rows = [r for r in rows_ if r["class"] == "form"]
    if len(form_rows) != 1 or form_rows[0]["note"].find(FORM_DEPENDENT) < 0:
        bad.append("exactly one row may be form-dependent and it must be the tied table")
    # MUST-CATCH: a census row nobody classified has to stop the pricing, not be skipped.
    fake = STM.rows(k, LANES) + [("brand", "read", 1 << 30, 0, "a row nobody wrote a rule for")]
    keep = STM.rows
    STM.rows = lambda kk, lanes: fake
    try:
        _, unc, _ = census(k)
        if len(unc) != 1:
            bad.append(f"injected row was NOT caught (uncovered={len(unc)})")
        # MUST-CATCH: an ambiguous row (two rules matching) must also be caught.
        STM.rows = lambda kk, lanes: [("mlp", "read", 1, 1,
                                       "24 layers x (gate+up+down) and 18 GDN layers: qkv/z/a/b in")]
        _, _, amb2 = census(k)
        if len(amb2) != 1:
            bad.append("ambiguous injection was NOT caught")
    finally:
        STM.rows = keep
    # MUST-NOT-CATCH: a real row, re-read, is still accepted (the must-not-catch half).
    if len(census(k)[0]) != len(rows_):
        bad.append("re-reading the real census changed its size — the classifier is not stable")
    # The claim that justifies the whole measurement: if the naive bound were below the kill line there
    # would be nothing to measure. If this ever flips, STOP and say so (the brief's escape hatch).
    a = json.loads((ROOT / "prototype" / "mtp_gatezero" / "draft_survival.json").read_text())["accept"]
    dmax = verdicts(a)
    naive = sum(v for f, v in FAMILY_MS.items()
                if f in ("mlp", "proj", "attn", "gdn", "glue", "copy", "host_tail", "lm_head"))
    # A units witness: a 993,280 B row over a 62.8 GB/s link cannot be anything but ~0.016 ms. The
    # first draft of this line divided by 62.8e6 and printed 15.8 ms of host_tail, which still looked
    # like a number - a scale error in a byte-to-ms helper has to be caught by a bound, not by reading.
    d2h_ms = state_constants()["kVocab"] * 4 / 62.8e9 * 1e3
    if not 0.005 < d2h_ms < 0.05:
        bad.append(f"logits D2H reads {d2h_ms:.4f} ms; 1 MB at 62.8 GB/s is 0.016 ms — a unit error")
    if naive / RP.STEP_MS <= dmax:
        bad.append(f"naive bound {naive/1000:.3f} ms is now BELOW the kill line {dmax:.3f}: the "
                   f"measurement may no longer be needed — re-read the brief before spending device time")
    print("SELFTEST", "PASS" if not bad else "FAIL")
    for b in bad:
        print("  " + b)
    print(f"  census rows {len(rows_)}, rules {len(SCALING_RULES)}, delta_max {dmax:.4f} of a step,"
          f" naive bound {naive / RP.STEP_MS:.4f} of a step (above the kill line, so D3 is needed)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
