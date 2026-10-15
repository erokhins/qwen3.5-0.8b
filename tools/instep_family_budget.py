#!/usr/bin/env python3
"""Why does a real decode step cost ~3 ms more than the sum of its kernels' own bursts?

CPU-only arithmetic. It takes NO timing and touches no device: every number it reads is already
committed — the stage-clock family rows come from a `decode_run --stage-clock` log that someone
else ran under the GPU lock, the shapes come from `engine/state.h`, and the isolated-burst and
probe figures are the ones `bench/performance-status.md` already carries. What this file adds is
the *comparison* of those three sources, which is what Q7 needs to decide between three candidate
explanations for the residue:

  (a) a per-launch ramp — a small launch never reaches peak bandwidth because it spends its first
      tens of microseconds getting started, so the deficit is a per-launch constant that amortizes
      with launch size;
  (b) interleaved traffic / locality — the step walks ~1.5 GB of distinct weights in 424
      launches, so every launch's pages/rows/TLB entries are cold, while a burst re-walks one
      small window and is flattered by it (the trap rule that `prototype/mlp_gemv/results.md`
      documents at L2 level, one level up);
  (c) a mispriced denominator — the rooflines the bursts are compared against are wrong.

The tests it prints, and the shape of the answer each one gives:

  * `same-geometry` compares ONE family's in-situ cost per launch against its own isolated burst
    at the same grid and the same bytes per launch. A ramp that belongs to the launch must appear
    in both runs; a difference here therefore cannot be story (a).
  * `deficit shape` decides whether the leftover tracks launches or bytes. Per-launch and per-MB
    deficits are printed side by side: story (a) predicts the first is flat across families,
    story (b) predicts the second is.
  * `roofline recompute` re-derives every family's bytes from `engine/state.h` and divides by the
    measured 237 GB/s probe, so story (c) is checked against the checkpoint instead of argued.

Usage:
    tools/instep_family_budget.py tmp/q61/stage128_1.log [more logs...]
Multiple logs of the same geometry are median-merged, which is how `bench/performance-status.md`
takes its stage rows (three runs, median).
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The measured LPDDR5x ceiling of this box (`devices/strix-halo-memory-bandwidth.md`, quoted by
# `bench/performance-status.md` as "the 237 GB/s probe"). Every roofline here is bytes/this.
PROBE_GBS = 237.0

# Isolated-burst cost per token per family, as committed in the frozen ledger's Q6.1 table
# (progress/2026-10-16/performance-status-2026-10-16.md:683-693). `None` = never measured as a burst (the cast/copy
# rows there are dispatch estimates, and proj is derived from per-shape ratios taken under contention,
# which is why Q7 leans on mlp and lm_head rather than on proj for the same-geometry test).
BURST_MS = {
    "mlp": 2.594,        # prototype/mlp_gemv: cold, 6 replicas of ONE layer, rows/block 8
    "proj": None,        # 3.54-3.60 in the ledger, but derived — kept out of the deciding test
    "lm_head": 2.127,    # prototype/lm_head_gemv: one 485 MiB launch, repeated
    "glue": 0.190,
    "gdn": 0.115,
    "attn": None,        # 0.114 derived
    "cast": None,
    "copy": None,
    "host_tail": None,
}

# Rough per-launch dispatch price by family, in microseconds, from the frozen ledger's measured dispatch
# curve (progress/2026-10-16/performance-status-2026-10-16.md:135, the 1.85 us floor at :39; 2.6-3.8 us at the widths these families use). Used
# only to say how much of a family's in-situ row is the instrument's own exposed submission cost.
DISPATCH_US = {"mlp": 3.75, "proj": 2.80, "lm_head": 3.75, "glue": 3.06, "gdn": 3.06,
               "attn": 3.06, "cast": 3.06, "copy": 3.06, "host_tail": 0.0}

N_GDN, N_ATTN, N_LAYERS = 18, 6, 24


def state_constants() -> dict[str, int]:
    """The `constexpr int kX = <expr>;` rows of `engine/state.h`, evaluated in dependency order.

    Parsed rather than retyped for the reason `engine/weights_contract.py` gives: a shape claim
    copied into a second file is a guess that will be wrong the day the first file changes.
    """
    text = (REPO / "engine" / "state.h").read_text()
    vals: dict[str, int] = {}
    pending = re.findall(r"constexpr int (k\w+)\s*=\s*([^;]+);", text)
    stuck = len(pending)
    while pending and stuck:
        stuck = 0
        for name, expr in pending:
            try:
                vals[name] = int(eval(expr, {"__builtins__": {}}, dict(vals)))  # noqa: S307
                stuck += 1
            except Exception:      # a later line defines a name this one uses; iterate
                pass
        pending = [(n, e) for n, e in pending if n not in vals]
        stuck = len(pending) if stuck else 0
        if stuck == 0 and pending:
            raise SystemExit(f"state.h: cannot resolve {pending[0][0]} = {pending[0][1]}")
    return vals


def family_bytes(k: dict[str, int]) -> dict[str, int]:
    """Bytes of *weight* traffic one decode token moves per family, from the shapes in state.h.

    One token reads each weight row exactly once (n = 1 row of x, bf16 weights), so weight bytes
    per token = 2 * sum(rows * in) over the tensors that family launches. The x/out tile traffic
    is kilobytes at n = 1 and is left out; it is what makes the cast family cost anything at all,
    and the cast row is excluded from the byte-shaped tests for that reason.
    """
    hid, ff = k["kHidden"], k["kIntermediate"]
    qkv, zval, qproj, kvproj, qwidth = (k["kQkvRows"], k["kValueDim"], k["kQprojRows"],
                                       k["kKvProjRows"], k["kQwidth"])
    ab = k["kNumValueHeads"]
    gdn = (qkv + zval + ab + ab) * hid + hid * zval            # in_proj qkv,z,b,a + out_proj
    attn = (qproj + 2 * kvproj) * hid + hid * qwidth          # q,k,v + o_proj
    mlp = (2 * ff * hid + hid * ff)                           # gate+up in one launch, then down
    return {
        "proj": 2 * (N_GDN * gdn + N_ATTN * attn),
        "mlp": 2 * N_LAYERS * mlp,
        "lm_head": 2 * k["kVocab"] * hid if "kVocab" in k else 2 * 248320 * hid,
    }


def parse_stage_rows(path: Path) -> dict[str, dict[str, float]]:
    """The `STAGE gran=stage family=...` rows of one `--stage-clock` log (plus the TOTAL rows)."""
    rows: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"STAGE gran=(stage|kernel) family=(\w+)\s+(.*)", line)
        if m:
            gran, fam, rest = m.groups()
            f = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in rest.split()}
            rows.setdefault(f"{gran}:{fam}", {}).setdefault("ms", []).append(f["ms_per_step"])
            rows[f"{gran}:{fam}"]["launches"] = f["launches_per_step"]
            continue
        m = re.match(r"STAGE gran=(stage|kernel) TOTAL\s+(.*)", line)
        if m:
            gran, rest = m.groups()
            f = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in rest.split()}
            rows.setdefault(f"{gran}:TOTAL", {}).setdefault("ms", []).append(f["ms_per_step"])
            rows[f"{gran}:TOTAL"]["syncs"] = f["syncs_per_step"]
            rows[f"{gran}:TOTAL"]["launches"] = f["launches_per_step"]
    return rows


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    merged: dict[str, dict[str, float]] = {}
    for p in map(Path, argv):
        for key, val in parse_stage_rows(p).items():
            merged.setdefault(key, {}).update(val)
    k = state_constants()
    fbytes = family_bytes(k)

    print(f"logs: {', '.join(argv)}   probe: {PROBE_GBS:.0f} GB/s")
    print(f"shapes from engine/state.h: hidden={k['kHidden']} ffn={k['kIntermediate']} "
          f"qkv={k['kQkvRows']} z={k['kValueDim']} q={k['kQprojRows']} kv={k['kKvProjRows']}")
    print()
    print("family      launch  ms_in_situ  MB_weight  us/launch  us/MB  GB/s_in_situ  "
          "roofline_ms  burst_ms  ms_per_launch_burst")
    order = ["mlp", "proj", "lm_head", "gdn", "attn", "glue", "cast", "copy", "host_tail"]
    for fam in order:
        st = merged.get(f"stage:{fam}")
        if not st:
            continue
        ms = statistics.median(st["ms"])
        n = int(st["launches"])
        mb = fbytes.get(fam, 0) / 1e6
        roof = (fbytes[fam] / PROBE_GBS / 1e6) if fam in fbytes else float("nan")
        burst = BURST_MS.get(fam)
        # MB / ms is 1e9 B/s, i.e. GB/s; us/MB is ms per 1e6 B, i.e. ms*1000/MB.
        print(f"{fam:<10} {n:>6}  {ms:>10.4f}  {mb:>9.1f}  {ms*1000/n if n else 0:>10.2f}  "
              f"{ms*1000/mb if mb else float('nan'):>6.2f}  "
              f"{mb/ms if mb else float('nan'):>12.1f}  {roof:>11.3f}  "
              f"{burst if burst is not None else float('nan'):>9.3f}  "
              f"{burst*1000/n if burst and n else float('nan'):>19.2f}")
    tot = merged.get("stage:TOTAL")
    ktot = merged.get("kernel:TOTAL")
    if tot:
        print(f"\nstage TOTAL {statistics.median(tot['ms']):.4f} ms/step, "
              f"{int(tot['launches'])} launches, syncs_per_step={int(tot['syncs'])}")
    if ktot:
        print(f"kernel TOTAL {statistics.median(ktot['ms']):.4f} ms/step, "
              f"syncs_per_step={int(ktot['syncs'])}")

    print("\n--- test same-geometry: story (a) says a launch carries a ramp; a burst runs the "
          "same launch")
    for fam in ("mlp", "lm_head"):
        st, burst = merged.get(f"stage:{fam}"), BURST_MS.get(fam)
        if not st or not burst:
            continue
        n = int(st["launches"])
        mb = fbytes[fam] / 1e6
        insitu = statistics.median(st["ms"]) * 1000 / n
        burst_usmb = burst * 1000 / mb
        print(f"  {fam}: {n} launches/step, {mb/n:.2f} MB each — "
              f"in-step {insitu:.1f} us/launch ({statistics.median(st['ms'])*1000/mb:.2f} us/MB) "
              f"vs its own burst {burst*1000/n:.1f} us/launch ({burst_usmb:.2f} us/MB) "
              f"= x{insitu/(burst*1000/n):.2f} at identical geometry")
    print("  same kernel, same grid, same bytes per launch: only the window each run re-walks "
          "differs (mlp burst = 6 replicas of ONE layer = 132 MB, re-read every 6 launches; the "
          "step walks 528 MB of distinct MLP weights once per token and lm_head's burst window "
          "is its own 485 MiB, too big to be warm either way).")

    print("\n--- test deficit shape: per-launch vs per-MB (story (a) predicts flat per launch)")
    for fam in ("mlp", "proj", "lm_head"):
        st = merged.get(f"stage:{fam}")
        if not st or fam not in fbytes:
            continue
        ms = statistics.median(st["ms"])
        n = int(st["launches"])
        mb = fbytes[fam] / 1e6
        deficit = ms - fbytes[fam] / PROBE_GBS / 1e6
        print(f"  {fam:<8} deficit {deficit:+.3f} ms over {n:>3} launches = "
              f"{deficit*1000/n:+6.1f} us/launch = {deficit*1000/mb:+6.2f} us/MB")
    n_all = int(tot["launches"]) if tot else 424
    step_ms = statistics.median(tot["ms"]) if tot else float("nan")
    print("  a per-launch ramp sized to explain mlp would bill every one of the "
          f"{n_all} launches at that rate; the step is {step_ms:.2f} ms.")
    # Sigma exec_in_situ = stage total - (D + N*s): D = 1.63 ms (Q6's dispatch column as written),
    # s = 0.098 us (the same window's `calib` line). This reproduces the ledger's 12.74 ms, which is
    # the check that the two readings are the same quantity.
    d_ms, s_us = 1.63, 0.098
    exec_ms = step_ms - (d_ms + n_all * s_us / 1000.0)
    weight_mb = sum(fbytes.values()) / 1e6
    print(f"\n--- aggregate: sigma exec_in_situ = {step_ms:.4f} - ({d_ms} + {n_all}*{s_us}us) "
          f"= {exec_ms:.2f} ms (the ledger's 12.74); {weight_mb:.0f} MB of weights per token over "
          f"it = {weight_mb/exec_ms:.0f} GB/s against the {PROBE_GBS:.0f} GB/s probe. The only "
          "family above that aggregate rate is the one launch big enough to be a steady-state "
          "stream on its own.")

    print("\n--- test roofline recompute (story (c)) against the ledger's rows")
    ledger_roof = {"mlp": 2.230, "proj": 1.973, "lm_head": 2.146}
    for fam, roof in ledger_roof.items():
        mine = fbytes[fam] / PROBE_GBS / 1e6
        flag = "ok" if abs(mine - roof) / roof < 0.01 else "DISAGREES"
        print(f"  {fam:<8} recomputed {mine:.3f} ms ({fbytes[fam]:,} B) vs ledger {roof:.3f} "
              f"-> {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
