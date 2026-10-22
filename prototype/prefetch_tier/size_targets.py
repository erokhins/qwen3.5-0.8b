#!/usr/bin/env python3
"""size_targets.py — what P1 could possibly win, computed before any millisecond is spent.

This is the CPU half of route P1: the ceiling per prefetch target, from byte counts and QUOTED rates.
It exists so a reader can tell whether P1 can reach the repo's prize bar (~0.30 ms on a 10.9357 ms
step) without paying for a timed matrix, and so the timed matrix, when it is paid for, is asked a
question whose size is already known.

WHAT IS DERIVED HERE, and from what
  * Byte counts are DERIVED, not typed: `tools/step_traffic_model.py` is executed and its rows parsed
    — including the layer count written in each row's own annotation, which is then cross-checked
    against `engine/state.h`. A retyped shape is how `check_weights` shipped a rank-3 tensor as rank-2
    (AGENTS trap row), and the first draft of this table picked a per-layer proj block with min()/max()
    over aggregated rows and printed 95 MiB for a 22 MiB layer. Derive, then cross-check.
  * Rates and stage costs are QUOTED. Every one carries a [Qn] tag and a source in QUOTED/STAGES.
    Nothing here measures anything; no figure printed here may be quoted as a measurement.

THE MODEL, stated so it can be attacked
  A prefetch moves a read earlier; it does not remove it. So the wall prize is NOT "the bytes are free",
  it is the difference in WHERE the bytes come from AT THE USE SITE:
      ceiling_transfer(bytes) = min(tier_bytes, bytes) * (1/plateau_read - 1/tier_read)
  which is the whole prize only if the consuming family is transfer-bound. For a latency-bound family
  (attn at 13 % of floor, gdn at 30 %) exposed DRAM latency is also in play, and the bound on that is
  the family's own non-floor time:
      ceiling_latency(family) = cost(family) - floor(family)
  Both columns are printed and neither is measured here.

REFUSALS (print `REFUSE <reason>`, exit 2) — a number that cannot fail is not a number:
  * any quoted rate <= 0, or a stage cost materially below its own byte floor (the ledger's lm_head row
    legitimately runs at 104 % of floor, so a small overshoot is accepted and annotated, a large one
    is drift and refuses);
  * a traffic-model row or a state.h constant missing, or an annotation's layer count disagreeing with
    state.h;
  * a derived target byte count of 0, or a ceiling that comes out NaN.

Usage: .venv/bin/python prototype/prefetch_tier/size_targets.py [--json out.json]
"""

import argparse
import functools
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------------------------
# Quoted inputs. [Qn] tags are how P1-ARITHMETIC.md cites them. NOT measured in this file.
# ---------------------------------------------------------------------------------------------
QUOTED = {
    "Q1_plateau_read_gbs": (237.0, "devices/strix-halo-memory-bandwidth.md: iGPU read plateau"),
    "Q2_plateau_copy_gbs": (210.0, "devices/strix-halo-memory-bandwidth.md: iGPU copy (r+w) plateau"),
    "Q3_tier_read_gbs":    (876.7, "devices/strix-halo-memory-bandwidth.md: 32 MiB warm read"),
    "Q4_tier_copy_gbs":    (325.7, "devices/strix-halo-memory-bandwidth.md: 32 MiB warm copy"),
    "Q5_tier_bytes":       (33554432, "prototype/p1c_query: KFD node1 cache level 3 = 32768 KiB"),
    "Q6_l2_bytes":         (2097152, "prototype/p1c_query: attr L2CacheSize == persistingL2CacheMaxSize"),
    "Q7_step_ms_4032":     (10.9357, "engine/generation-limits.md operating point (4,032 lanes)"),
    "Q8_prize_bar_ms":     (0.30, "queue P1 preamble: the bar both of this week's wins cleared"),
    "Q9_state_cold_ms":    (0.070, "prototype/gdn_insitu/D4-DIAGNOSIS.md S7: state-coldness penalty, "
                                   "saturates at 32 MiB"),
    "Q10_mlp_memstate_ms": (0.06, "engine/generation-limits.md S2: mlp gap decomposition, the "
                                  "memory-state term"),
    "Q11_dispatch_us":     (3.10, "engine/generation-limits.md S4: measured production cost of a launch"),
}

# Stage lines at 4,032 lanes: family -> (cost ms, byte floor ms), from engine/generation-limits.md S1.
# The table is pre-cast-fold, as that document states; host_tail carries no bytes and no floor.
STAGES = {
    "mlp":       (3.3628, 2.340),
    "proj":      (2.7578, 2.069),
    "lm_head":   (2.1685, 2.254),   # 104 % of floor, i.e. at the wall: zero idle fetch capacity
    "attn":      (1.6121, 0.209),
    "gdn":       (0.6458, 0.190),
    "glue":      (0.5593, 0.0),
    "copy":      (0.2141, 0.0),
    "host_tail": (0.7163, 0.0),     # GPU idle: the whole line is free fetch time
}

MB = 10 ** 6
MIB = 1024 ** 2


def die(msg):
    print("REFUSE " + msg)
    sys.exit(2)


def q(key):
    v = QUOTED[key][0]
    if not (v > 0):
        die("quoted %s = %r is not positive; every ceiling here divides by it" % (key, v))
    return float(v)


def ms_at(byts, rate_gbs):
    """Time in ms for `byts` bytes at `rate_gbs` GB/s (10^9 B/s). One place, so the units cannot drift."""
    if rate_gbs <= 0:
        die("rate %r not positive" % rate_gbs)
    return byts / (rate_gbs * 1e9) * 1e3


def read_state_constants():
    """Reads the hyperparameters out of engine/state.h instead of retyping them."""
    path = os.path.join(ROOT, "engine", "state.h")
    if not os.path.exists(path):
        die("engine/state.h not found; cannot derive shapes")
    text = open(path, errors="replace").read()
    consts = {}
    for name, val in re.findall(r"constexpr int (k\w+)\s*=\s*([0-9]+)\s*;", text):
        consts[name] = int(val)
    # kConvWindow is `kConvKernel - 1`, i.e. derived, so a literal scan cannot see it. Derive it from
    # the constant that IS a literal rather than refusing for the wrong reason.
    if "kConvWindow" not in consts and "kConvKernel" in consts:
        consts["kConvWindow"] = consts["kConvKernel"] - 1
    need = ["kNumLayers", "kNumGdnLayers", "kNumAttnLayers", "kNumValueHeads", "kDv",
            "kNumKvHeads", "kHeadDim"]
    missing = [n for n in need if n not in consts]
    if missing:
        die("engine/state.h did not yield %s; this tool would have to guess a shape" % missing)
    return consts


def traffic_rows():
    """Runs the repo's traffic model and returns (aggregated bytes, annotated rows)."""
    exe = os.path.join(ROOT, ".venv", "bin", "python")
    tool = os.path.join(ROOT, "tools", "step_traffic_model.py")
    if not os.path.exists(tool):
        die("tools/step_traffic_model.py not found; family bytes cannot be derived")
    env = dict(os.environ, LC_ALL="C")
    try:
        out = subprocess.run([exe, tool], cwd=ROOT, env=env, capture_output=True,
                             text=True, timeout=600).stdout
    except Exception as exc:                                    # noqa: BLE001
        die("traffic model failed: %s" % exc)
    fam, rows = {}, []
    for line in out.splitlines():
        m = re.match(r"^(\w+)\s+(read|write)\s+([0-9.]+)\s+[0-9.]+%\s+(.*)$", line.strip())
        if not m:
            continue
        key, d, byts, desc = (m.group(1), m.group(2)), m.group(2), float(m.group(3)) * MB, m.group(4)
        fam[key] = fam.get(key, 0.0) + byts
        rows.append((key, byts, desc))
    for key in [("mlp", "read"), ("proj", "read"), ("lm_head", "read"), ("gdn", "read")]:
        if key not in fam:
            die("traffic model printed no %s line; the population is not what this reader expects"
                % (key,))
    return fam, rows


def per_layer(rows, key, want):
    """Per-layer bytes for one annotated row: the divisor comes from the row's own text, so a
    re-annotation of the traffic model moves this table with it instead of silently rotting."""
    hits = [(b, d) for (k, b, d) in rows if k == key and re.search(want, d)]
    if len(hits) != 1:
        die("family %s: %d rows match %r; refusing to pick a per-layer block by size"
            % (key, len(hits), want))
    byts, desc = hits[0]
    m = re.search(r"(\d+)\s+(?:GDN |attn |model )?layers", desc)
    if not m:
        die("row %r carries no layer count; cannot derive a per-layer block" % desc)
    n = int(m.group(1))
    if n <= 0 or byts <= 0:
        die("per-layer derivation got %r bytes over %r layers" % (byts, n))
    return byts / n, n


def fmt_bytes(b):
    if b >= 1e9:
        return "%.2f GB" % (b / 1e9)
    if b >= 1e6:
        return "%.2f MB" % (b / 1e6)
    if b >= 1024:
        return "%.1f KiB" % (b / 1024.0)
    return "%d B" % b


# The rig and the fitter both need "how big is ONE engine slice", and a size spelled twice is a size that
# drifts. This maps the rig's --target names onto derived names so the ceiling table, the rig's
# --target-mib values and the fitter's per-step multiplier all read one derivation.
RIG_SLICE_TARGETS = {"state": "gdn_state_1layer", "kv-layer": "kv_1layer_at_4k",
                     "weights": "weights_1gdn_layer"}


@functools.lru_cache(maxsize=1)
def derive():
    """The derived byte census as {name: bytes}, importable without printing the report.

    Reads engine/state.h and runs tools/step_traffic_model.py once per process (hence the cache); the
    layer count for a per-layer block comes from that tool's own annotation, never from a typed number.
    Refuses rather than guesses, exactly as main() does.
    """
    c = read_state_constants()
    fam, rows = traffic_rows()
    gdn_state = (c["kNumGdnLayers"] * c["kNumValueHeads"] * c["kDv"] * c["kDv"] * 4)   # F32, §08.1
    kv_per_token = c["kNumAttnLayers"] * 2 * c["kNumKvHeads"] * c["kHeadDim"] * 2      # BF16 K and V
    lanes = 4032                                                    # the standing operating point
    kv_step = kv_per_token * lanes
    kv_layer = kv_step / c["kNumAttnLayers"]
    mlp_layer = fam[("mlp", "read")] / c["kNumLayers"]
    proj_gdn_layer, n_gdn = per_layer(rows, ("proj", "read"), r"GDN layers")
    proj_attn_layer, n_attn = per_layer(rows, ("proj", "read"), r"attn layers")
    if (n_gdn, n_attn) != (c["kNumGdnLayers"], c["kNumAttnLayers"]):
        die("traffic model annotates %d GDN / %d attn layers, engine/state.h says %d / %d"
            % (n_gdn, n_attn, c["kNumGdnLayers"], c["kNumAttnLayers"]))
    out = {
        "gdn_state_all18": gdn_state,
        "gdn_state_1layer": gdn_state / c["kNumGdnLayers"],
        "kv_all6_at_4k": kv_step,
        "kv_1layer_at_4k": kv_layer,
        "kv_per_token": kv_per_token,
        "mlp_layer": mlp_layer,
        "proj_1gdn_layer": proj_gdn_layer,
        "proj_1attn_layer": proj_attn_layer,
        "weights_1gdn_layer": mlp_layer + proj_gdn_layer,
        "weights_1attn_layer": mlp_layer + proj_attn_layer,
    }
    for name, v in out.items():
        if not (v > 0):
            die("derived target %s came out %r; refusing to divide by it" % (name, v))
    return out


def rig_slice_bytes(name):
    """Bytes in ONE engine slice for a rig target name. Refuses a name it cannot derive."""
    d = derive()
    if name not in RIG_SLICE_TARGETS:
        die("rig target %r has no derived slice size (known: %s)"
            % (name, ", ".join(sorted(RIG_SLICE_TARGETS))))
    return d[RIG_SLICE_TARGETS[name]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--slice-mib", default="", metavar="TARGET",
                    help="print ONE engine slice of TARGET in MiB and exit 0, so a shell script can "
                         "derive a rig --target-mib instead of typing one")
    ap.add_argument("--bytes-mib", default="", metavar="NAME",
                    help="print the derived size of any NAME in derive() in MiB and exit 0")
    args = ap.parse_args()
    if args.slice_mib:
        print("%.6f" % (rig_slice_bytes(args.slice_mib) / MIB))
        return 0
    if args.bytes_mib:
        d = derive()
        if args.bytes_mib not in d:
            die("--bytes-mib: unknown name %r (known: %s)"
                % (args.bytes_mib, ", ".join(sorted(d))))
        print("%.6f" % (d[args.bytes_mib] / MIB))
        return 0

    plateau = q("Q1_plateau_read_gbs")
    tier_rate = q("Q3_tier_read_gbs")
    tier_bytes = q("Q5_tier_bytes")
    step_ms = q("Q7_step_ms_4032")
    bar = q("Q8_prize_bar_ms")
    if tier_rate <= plateau:
        die("tier rate %.1f <= plateau %.1f: the fast tier would not be fast, so the model is wrong"
            % (tier_rate, plateau))

    c = read_state_constants()
    d = derive()
    gdn_state = d["gdn_state_all18"]
    kv_per_token = d["kv_per_token"]
    lanes = 4032
    kv_step = d["kv_all6_at_4k"]
    kv_layer = d["kv_1layer_at_4k"]
    mlp_layer = d["mlp_layer"]
    proj_gdn_layer = d["proj_1gdn_layer"]
    proj_attn_layer = d["proj_1attn_layer"]
    w_gdn_layer = d["weights_1gdn_layer"]
    w_attn_layer = d["weights_1attn_layer"]

    print("== [0] derived byte census (derived; rates are quoted)")
    print("  gdn state, all %d layers        : %s (%.3f MiB)   per layer %s"
          % (c["kNumGdnLayers"], fmt_bytes(gdn_state), gdn_state / MIB,
             fmt_bytes(gdn_state / c["kNumGdnLayers"])))
    print("  KV per token (%d attn layers)  : %s   per attn layer %s"
          % (c["kNumAttnLayers"], fmt_bytes(kv_per_token), fmt_bytes(kv_layer)))
    print("  KV at %d lanes                 : %s (%.3f MiB)" % (lanes, fmt_bytes(kv_step),
                                                                kv_step / MIB))
    print("  mlp weights per layer          : %s" % fmt_bytes(mlp_layer))
    print("  proj weights per GDN layer     : %s   per attn layer %s"
          % (fmt_bytes(proj_gdn_layer), fmt_bytes(proj_attn_layer)))
    print("  ONE GDN layer's weight block   : %s (%.3f MiB)"
          % (fmt_bytes(w_gdn_layer), w_gdn_layer / MIB))
    print("  ONE attn layer's weight block  : %s (%.3f MiB)"
          % (fmt_bytes(w_attn_layer), w_attn_layer / MIB))
    print("  tier capacity                  : %s (%.1f MiB) [Q5]; per-GPC L2 %s [Q6]"
          % (fmt_bytes(tier_bytes), tier_bytes / MIB, fmt_bytes(q("Q6_l2_bytes"))))

    # ---- [1] free fetch capacity per window ---------------------------------------------------
    print("\n== [1] idle fetch capacity per candidate window (costs/floors quoted, arithmetic derived)")
    print("  %-10s %9s %9s %10s %12s" % ("family", "cost ms", "floor ms", "idle ms", "free bytes"))
    windows, free_total_ms = {}, 0.0
    for famname, (cost, floor) in STAGES.items():
        if cost < floor:
            # lm_head legitimately reads this way: the ledger records it at 104 % of its byte floor,
            # i.e. faster than the plateau divides out. A real reading, so it becomes idle = 0 with the
            # ratio printed. A cost far below its own floor is not that story, it is a mistyped row.
            if floor > 0 and cost * 1.10 >= floor:
                windows[famname] = 0.0
                print("  %-10s %9.4f %9.4f %10.4f %12s  at/over the wall (%.0f %% of floor)"
                      % (famname, cost, floor, 0.0, "0 B", 100.0 * floor / cost))
                continue
            die("quoted %s cost %.4f ms is %.2fx below its own floor %.4f ms; the quotes drifted"
                % (famname, cost, floor / max(cost, 1e-9), floor))
        idle = cost - floor
        free_total_ms += idle
        windows[famname] = idle
        print("  %-10s %9.4f %9.4f %10.4f %12s" % (famname, cost, floor, idle,
                                                   fmt_bytes(idle * plateau * 1e6)))
    print("  idle stage time sums to %.4f ms of a %.4f ms step [Q7]" % (free_total_ms, step_ms))
    print("  -> bandwidth is NOT P1's binding constraint (queue P1 says the same); what binds is tier"
          " capacity and whether the bytes survive until the use site.")
    if free_total_ms > step_ms:
        die("idle stage time %.4f ms exceeds the step %.4f ms; stage lines overlap and this table"
            " cannot add them" % (free_total_ms, step_ms))

    # ---- [2] ceilings per target --------------------------------------------------------------
    print("\n== [2] ceiling per target")
    print("  transfer ceiling = min(tier, bytes) * (1/%.0f - 1/%.1f) [Q1,Q3];"
          "  latency ceiling = cost - floor" % (plateau, tier_rate))
    targets = [
        ("gdn_state_all18", gdn_state, "gdn",
         "quoted cold penalty %.3f ms [Q9] lands beside this" % QUOTED["Q9_state_cold_ms"][0]),
        ("gdn_state_1layer", gdn_state / c["kNumGdnLayers"], "gdn", "the slice one launch actually uses"),
        ("kv_all6_at_4k", kv_step, "attn", "six slices, one per attention layer"),
        ("kv_1layer_at_4k", kv_layer, "attn", "the slice one attention layer walks"),
        ("weights_1gdn_layer", w_gdn_layer, "mlp+proj",
         "quoted mlp memory-state term %.2f ms [Q10]" % QUOTED["Q10_mlp_memstate_ms"][0]),
        ("weights_1attn_layer", w_attn_layer, "mlp+proj", "as above"),
    ]
    print("  %-20s %11s %8s %6s %9s %9s  %s" %
          ("target", "bytes", "MiB", "fits", "xfer ms", "lat ms", "note"))
    rows_out = []
    for name, byts, consumer, note in targets:
        fits = byts <= tier_bytes
        xfer = (ms_at(min(tier_bytes, byts), plateau) - ms_at(min(tier_bytes, byts), tier_rate))
        cost, floor = STAGES.get(consumer, (0.0, 0.0))
        lat = max(0.0, cost - floor) if consumer in STAGES else float("nan")
        if xfer != xfer or xfer <= 0:
            die("target %s produced a non-positive/NaN transfer ceiling" % name)
        rows_out.append({"target": name, "bytes": byts, "fits_tier": fits,
                         "ceiling_transfer_ms": xfer, "ceiling_latency_ms": lat, "note": note})
        print("  %-20s %11s %8.3f %6s %9.4f %9s  %s"
              % (name, fmt_bytes(byts), byts / MIB, "yes" if fits else "NO", xfer,
                 ("%.4f" % lat) if lat == lat else "n/a", note))
    print("  A target larger than the tier cannot be staged whole; its xfer column bounds only the part"
          " that fits.")
    print("  The xfer column assumes the consuming family is TRANSFER-bound. It is not for attn/gdn:")
    print("  for those the prize can run up to the lat column, and only the rig can say where it lands.")

    # ---- [3] residency headroom ---------------------------------------------------------------
    print("\n== [3] does the staged target still FIT after the traffic that really sits in between?")
    print("  staged + intervening must be <= %.1f MiB [Q5], or the bytes are gone before the use site."
          % (tier_bytes / MIB))
    intervene = {
        "kv slice, staged in layer l-1's MLP": mlp_layer,
        "kv slice, staged in layer l's own qkv proj": proj_attn_layer,
        "gdn state, staged in its own layer's in_proj": proj_gdn_layer,
        "next layer's weights, staged in this layer's MLP": mlp_layer,
    }
    pair = [("kv_all6_at_4k", kv_step, "kv slice, staged in layer l-1's MLP"),
            ("kv_1layer_at_4k", kv_layer, "kv slice, staged in layer l-1's MLP"),
            ("kv_1layer_at_4k", kv_layer, "kv slice, staged in layer l's own qkv proj"),
            ("gdn_state_all18", gdn_state, "gdn state, staged in its own layer's in_proj"),
            ("gdn_state_1layer", gdn_state / c["kNumGdnLayers"],
             "gdn state, staged in its own layer's in_proj"),
            ("weights_1gdn_layer", w_gdn_layer, "next layer's weights, staged in this layer's MLP")]
    for tgt, byts, wkey in pair:
        head = intervene[wkey]
        total = byts + head
        verdict = "fits" if total <= tier_bytes else "EVICTED by arithmetic"
        print("  %-22s %-46s %9s + %9s = %9s  %s"
              % (tgt, wkey, fmt_bytes(byts), fmt_bytes(head), fmt_bytes(total), verdict))

    # ---- [4] aggregate ------------------------------------------------------------------------
    print("\n== [4] can P1 reach the bar on arithmetic alone?")
    state_x = ms_at(min(tier_bytes, gdn_state), plateau) - ms_at(min(tier_bytes, gdn_state), tier_rate)
    kv_x = ms_at(min(tier_bytes, kv_step), plateau) - ms_at(min(tier_bytes, kv_step), tier_rate)
    honest_new = state_x + kv_x
    print("  state transfer ceiling %.4f ms + KV transfer ceiling %.4f ms = %.4f ms"
          % (state_x, kv_x, honest_new))
    print("  a weight-stage ceiling is already measured at %.2f ms [Q10] and is not new money for P1"
          % QUOTED["Q10_mlp_memstate_ms"][0])
    print("  prize bar: %.2f ms [Q8]" % bar)
    print("  verdict: the TRANSFER model puts the new money at %.3f ms, %s the bar."
          % (honest_new, "BELOW" if honest_new < bar else "at or above"))
    print("  What the transfer model cannot bound is exposed DRAM latency in a latency-bound family:")
    print("  attn's non-floor time is %.4f ms and gdn's is %.4f ms, so only the rig's arm (c) decides"
          % (windows["attn"], windows["gdn"]))
    print("  whether a staged slice also removes the latency term. P1 is NOT dead on arithmetic;")
    print("  its expected value is the transfer column and its upside is the latency column.")
    print("  The rig therefore prints both questions, and the KILL line in sweep.sh says which"
          " reading kills the route.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"quoted": QUOTED, "targets": rows_out, "windows_idle_ms": windows,
                       "new_money_transfer_ms": honest_new}, fh, indent=1)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
