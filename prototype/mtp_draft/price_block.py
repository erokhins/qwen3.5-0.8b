#!/usr/bin/env python3
"""Pre-register the MTP draft block's price BEFORE it is measured, and print the WIN/narrow/KILL lines.

This is arithmetic on three published things and nothing else:

  * the byte census — `prototype/mtp_gatezero/mtp_census.json`: 15 tensors, 40,905,728 B (39.0107 MiB);
  * the machine ceiling — 237-240 GB/s LPDDR5x plateau, `devices/strix-halo-memory-bandwidth.md`, and the
    per-kernel efficiencies the ops themselves measured: `proj_gemv` 75 % of plateau, `mlp_gemv` 70 %
    (`prototype/*/results.md`);
  * the ledger's economics — `prototype/mtp_headbatch/HEADBATCH-RESULT.md`: accept `a = 0.7329` carried
    (kill line 0.257), a decode step of 10.933 ms (its own `delta = 1.4847 ms = 0.1358 of a step`), and
    the 1.247× headline that pricing produced, with the head folded into the round. The head-batch result
    is what made the ~485 MiB head NOT scale with draft depth, so from here the round's cost is the block.

The point of writing it down first is the one every `results.md` in this repo repeats: a band chosen after
the measurement is a rationalisation. The numbers below are computed, so the bands move if the inputs
move — rerun this file, do not edit its output.

CPU only. It prints no measurement of the device; `run_locked.sh` produces those.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CENSUS = ROOT / "prototype" / "mtp_gatezero" / "mtp_census.json"

# ── inputs, each with the file it came from ───────────────────────────────────────────────────────
PLATEAU_GBS = 237.0          # devices/strix-halo-memory-bandwidth.md, the LPDDR5x plateau
PROJ_EFF = 0.75              # prototype/proj_gemv/results.md — sustained share of plateau
MLP_EFF = 0.70               # prototype/mlp_gemv/results.md
LAUNCH_US = 3.10             # engine/performance-status.md, per-launch overhead on this box
STEP_MS = 1.4847 / 0.1358    # HEADBATCH-RESULT.md's own arithmetic: 10.933 ms
ACCEPT = 0.7329              # carried accept on the engine's bits (HEADBATCH-RESULT.md)
ACCEPT_KILL = 0.257          # its kill line
HEADLINE_REF = 1.247         # the 1.247× the plan already promises
LAUNCHES_PER_STEP = 14       # counted at the launch sites in td_modes.cpp, asserted by the rig

# Which tensors each stage reads, as (name, bytes). The block reads all 15 per step; the split is only so
# the right efficiency is applied to the right bytes (fc/o_proj/q/k/v are proj-shaped; gate/up/down are
# the MLP's). The two embedding reads are NOT here: the head is batched into the target's M rows, which is
# exactly what HEADBATCH-RESULT.md bought.
GEMV = {
    "proj": [("mtp.fc.weight", 1024 * 2048 * 2), ("mtp.layers.0.self_attn.o_proj.weight", 1024 * 2048 * 2),
             ("mtp.layers.0.self_attn.q_proj.weight", 4096 * 1024 * 2),
             ("mtp.layers.0.self_attn.k_proj.weight", 512 * 1024 * 2),
             ("mtp.layers.0.self_attn.v_proj.weight", 512 * 1024 * 2)],
    "mlp": [("mtp.layers.0.mlp.gate_proj.weight", 3584 * 1024 * 2),
            ("mtp.layers.0.mlp.up_proj.weight", 3584 * 1024 * 2),
            ("mtp.layers.0.mlp.down_proj.weight", 1024 * 3584 * 2)],
}
NORMS_BYTES = sum(s * 2 for s in (1024, 1024, 1024, 1024, 1024, 256, 256))   # 7 weight vectors, read once


def gbs(bytes_read: int, ms: float) -> float:
    return bytes_read / 1e9 / (ms / 1e3)


def predict() -> dict:
    """Cost of one draft step, from bytes and published efficiencies and nothing else."""
    pb = sum(n for _, n in GEMV["proj"])
    mb = sum(n for _, n in GEMV["mlp"])
    cen = json.loads(CENSUS.read_text())
    if pb + mb + NORMS_BYTES != cen["mtp_block_bytes"]:
        raise SystemExit(f"stage bytes {pb + mb + NORMS_BYTES} != census {cen['mtp_block_bytes']}: a "
                         "tensor joined or left the block since this model was written")
    t_proj = pb / (PLATEAU_GBS * PROJ_EFF * 1e6)          # ms: bytes / (GB/s x 1e9) x 1e3
    t_mlp = mb / (PLATEAU_GBS * MLP_EFF * 1e6)
    return {"proj_bytes": pb, "mlp_bytes": mb, "norm_bytes": NORMS_BYTES,
            "block_bytes": pb + mb + NORMS_BYTES,
            "ms_proj": t_proj, "ms_mlp": t_mlp, "ms_block": t_proj + t_mlp,
            "ms_launch": LAUNCHES_PER_STEP * LAUNCH_US / 1e3,
            "ms_round": t_proj + t_mlp + LAUNCHES_PER_STEP * LAUNCH_US / 1e3}


def speedup(k: int, block_ms: float, accept: float = ACCEPT) -> float:
    """`a·(k+1) / (1 + k·c)` — HEADBATCH-RESULT.md's own form, `c` = block cost as a fraction of a step.
    The head is absent on purpose: batching it into the target's M rows is what the previous commit bought
    (R(M=2) = 1.0019), so the block is the whole marginal cost of a round."""
    return accept * (k + 1) / (1.0 + k * block_ms / STEP_MS)


def main() -> int:
    p = predict()
    print(f"=== predicted cost of ONE draft step, from bytes and published efficiencies ===")
    print(f"  proj bytes {p['proj_bytes']:>10,} @ {PLATEAU_GBS:.0f} GB/s x {PROJ_EFF:.2f}"
          f"  -> {p['ms_proj']:.4f} ms")
    print(f"  mlp  bytes {p['mlp_bytes']:>10,} @ {PLATEAU_GBS:.0f} GB/s x {MLP_EFF:.2f}"
          f"  -> {p['ms_mlp']:.4f} ms")
    print(f"  norms+vecs {p['norm_bytes']:>8,} B (negligible: {p['norm_bytes']/p['block_bytes']:.2%}"
          f" of the block, read once)")
    print(f"  BLOCK TOTAL (block-only)            {p['ms_block']:.4f} ms   "
          f"= {p['ms_block']/STEP_MS:.4%} of a {STEP_MS:.2f} ms step")
    print(f"  launch floor  {LAUNCHES_PER_STEP} x {LAUNCH_US:.2f} us = {p['ms_launch']:.4f} ms"
          f"   (submissions; overlapped or exposed is what run_locked.sh's `empty` arm decides)")
    print(f"  BLOCK + LAUNCH                      {p['ms_round']:.4f} ms")

    print("\n=== what draft depth k is worth at the carried accept a = "
          f"{ACCEPT} (kill line {ACCEPT_KILL}) ===")
    for k in (1, 2, 4):
        c = p["ms_round"] * k
        print(f"  k={k}: round cost {c:.4f} ms  ->  {speedup(k, p['ms_round']):.4f}x   "
              f"(vs {HEADLINE_REF}x already promised)")
    print("  caveat: k>1 reuses the k=1 accept for every drafted position. Accept decays with draft depth,"
          "\n          so these are upper bounds in k; the engine's own accept-by-depth curve (D4) is what "
          "prices depth.")
    # Sensitivity, in the units the queue actually asks in.
    b0, b1 = p["ms_round"], p["ms_round"] + 0.1
    d = speedup(1, b1) - speedup(1, b0)
    print(f"  sensitivity: +0.1 ms on a k=1 round moves the headline by {d:+.4f}x "
          f"({speedup(1, b0):.4f}x -> {speedup(1, b1):.4f}x)")

    print("\n=== break-evens (arithmetic, not taste) ===")
    print(f"  k=1 pays only if a > 0.5 at any block cost; at a={ACCEPT} the block may cost up to "
          f"{STEP_MS * (2 * ACCEPT - 1):.3f} ms before the round stops being a win at all")
    print(f"  the block may cost up to {STEP_MS * (ACCEPT * 2 / HEADLINE_REF - 1):.3f} ms before the "
          f"projected headline falls below the {HEADLINE_REF}x already promised")

    win = 1.25 * p["ms_block"]
    narrow = 2.5 * p["ms_block"]
    kill = STEP_MS * (ACCEPT * 2 / HEADLINE_REF - 1)
    print("\n=== PRE-REGISTERED BANDS for the k=1 block-only arm (measured by run_locked.sh) ===")
    print(f"  WIN     <= {win:.4f} ms  ({1.25:.2f}x the census prediction: overhead and staging slack, "
          f"still >= {speedup(1, win + p['ms_launch']):.3f}x projected)")
    print(f"  NARROW  <= {narrow:.4f} ms ({2.5:.2f}x: legal-but-slow, fund a staging pass, do not fund "
          f"deeper k)")
    print(f"  KILL    >  {kill:.4f} ms  (would drop the projection under {HEADLINE_REF}x; and since the "
          f"census says {p['ms_block']:.3f} ms, hitting it means the design re-reads weights, not a slow "
          f"kernel)")
    print(f"  KILL(also) measured(k=2)/measured(k=1) > 2.35 — superlinear in draft depth means the second "
          f"round re-reads what the first should have kept (cache/route defect), not a slow kernel")
    print(f"  KILL(also) any arm admitted at structural level only (no numeric band) may not be quoted as "
          f"a headline")
    # ---- apply the bands to a measurement -------------------------------------------
    # Added by the main agent on 2026-10-22. The handover's own command passed `--evidence DIR` to this
    # file, which had NO argument handling at all: the flag was accepted by silence and ignored, so the
    # tool printed bands it had never compared against a number - the same defect class as a parser that
    # matches zero records and then reports PASS. It now requires the runner's log, refuses a missing or
    # partial parse, and prints the verdict the bands were written to produce.
    ev = None
    for flag in ("--evidence", "--log"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 >= len(sys.argv):
                sys.exit("%s needs a path" % flag)
            ev = sys.argv[i + 1]
    if ev is None:
        print("\n  NO MEASUREMENT ADJUDICATED. Pass --evidence <log from run_locked.sh> to apply these "
              "bands; bands without a measurement are not a verdict.")
        return 0
    import pathlib as _pl, re as _re
    pth = _pl.Path(ev)
    files = sorted(pth.glob("*.txt")) if pth.is_dir() else ([pth] if pth.exists() else [])
    if not files:
        sys.exit("evidence %r does not exist or is empty: refusing to adjudicate nothing" % ev)
    pat = _re.compile(r"pass\s*(\d)\s+(\S+)\s+admission=(\S+)\s+([0-9.]+)")
    arms = {}
    for f in files:
        for line in f.read_text(errors="replace").splitlines():
            m = pat.search(line)
            if m:
                arm, adm, val = m.group(2), m.group(3), float(m.group(4))
                if arm == "empty":
                    continue  # the empty arm reports us/launch, not ms: a different unit, see runner
                arms.setdefault(arm, {"adm": adm, "v": []})["v"].append(val)
                arms[arm]["adm"] = adm
    if "block-k1" not in arms or "block-k2" not in arms:
        sys.exit("parsed arms %s lack block-k1/block-k2: refusing to adjudicate a partial parse"
                 % sorted(arms))
    best = {a: min(d["v"]) for a, d in arms.items()}
    k1, k2 = best["block-k1"], best["block-k2"]
    print("\n=== MEASURED, ADJUDICATED against the bands above ===")
    for a in sorted(best):
        print("  %-8s min=%.4f ms  admission=%-11s passes=%s" % (a, best[a], arms[a]["adm"],
              " ".join("%.4f" % x for x in arms[a]["v"])))
    if arms["block-k1"]["adm"] != "numeric":
        print("  NOT QUOTABLE: block-k1 was admitted at %s level only (pre-registered rule)." %
              arms["block-k1"]["adm"])
        return 0
    band = ("WIN" if k1 <= win else "NARROW" if k1 <= narrow else
            "KILL" if k1 > kill else "BAND-GAP")
    print("  block-k1 = %.4f ms -> %s   (census prediction %.4f ms)" % (k1, band, p["ms_block"]))
    if band == "BAND-GAP":
        print("  The pre-registration fixed WIN <= %.4f, NARROW <= %.4f, KILL > %.4f and left the span "
              "between them undefined; it is reported as undefined, not back-filled, because inventing a "
              "band after the number exists is what this repo forbids." % (win, narrow, kill))
    ratio = k2 / k1
    print("  block-k2 / block-k1 = %.4f -> %s" % (ratio, "KILL(also): superlinear in depth" if
          ratio > 2.35 else "linear: depth is a serial chain, not a re-read of the block"))
    if "block-k4" in arms:
        print("  block-k4 / block-k1 = %.4f (k=4 admitted structural-only: it sizes the curve, it is not "
              "a headline)" % (best["block-k4"] / k1))
    print("  projected headline at the MEASURED block cost: %.4fx  (census model said %.4fx)"
          % (speedup(1, k1 + p["ms_launch"]), speedup(1, p["ms_block"] + p["ms_launch"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
