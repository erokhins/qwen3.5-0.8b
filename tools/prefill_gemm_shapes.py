#!/usr/bin/env python3
"""Every matmul a prefill wave performs, at the wave widths the plan uses. Paper, not measurement.

    .venv/bin/python tools/prefill_gemm_shapes.py [--waves 64,512,1024,3072]
                                                 [--model models/Qwen3.5-0.8B] [--rate fp16]

Three questions from `bench/plan-prefill-2026-10-12.md` ("Funded: prefill compute", route (b)/(c)
spike), all of them arithmetic on shapes and on rates that are already measured elsewhere:

1. SHAPE COVERAGE. The vendor-library question cannot be answered from a square. The model issues 12
   projection/MLP matmul shapes plus `lm_head`, two of which (`k/v_proj`, `in_proj_a/b`) are far
   skinnier than anything `tools/gemm_peak.cpp` has ever run, and skinny is where library kernels are
   most likely to be bad. Note the brief called `k/v_proj` "N=256"; the shard says [512,1024] —
   `kKvProjRows = kNumKvHeads * kHeadDim = 2 x 256` (`engine/state.h`) — so N=512, and the actual
   skinny shapes are `in_proj_a`/`in_proj_b` at N=16. This table is computed from the shard, so the
   claim gets corrected rather than repeated.
2. ARITHMETIC INTENSITY at each shape, against the ridge implied by the measured matrix rate and the
   measured memory rate. Rates are QUOTED from `devices/strix-halo-compute.md` (fp16 6.885, bf16
   5.667, fp32 2.932 TFLOP/s at the engine's own shape) and the bandwidth from the memory probe; no
   timing appears anywhere in this file, and no timing may be derived from it — ms columns are labelled
   arithmetic on a quoted rate.
3. THE EPILOGUE BILL. Four things happen today inside a kernel that a library matmul cannot do: the
   `silu(gate)*up` multiply (fused in `mlp_gateup`), the attention gate de-interleave and q/k head-norm
   + rope (fused in `attn_prep`), the gated RMSNorm after the mixer, and the residual add fused with the
   next RMSNorm (`glue_add_norm`). For each: does a library matmul actually cost an extra pass, and how
   many bytes is that pass at each wave width.

Shapes come from the shard header via `engine/weights_contract.header_shapes`; the layer split comes
from `config.json`'s own `full_attention_interval`; the constants come from `engine/state.h`. The
matmul SET is what `engine/decode_step.cpp` calls: `proj_gemv` for the nine projections plus
o_proj/out_proj, `mlp_gateup`/`mlp_down`, and one `lm_head_full` for the whole request (§06.5: prefill
prices exactly one logits row).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import weights_contract as wc  # noqa: E402

# Rates and bandwidth this file divides by. Quoted from devices/, measured by someone else, in a
# window this script does not own.
RATES_TFLOPS = {"fp16": 6.885, "bf16": 5.667, "fp32": 2.932}
DEFAULT_BW_GBS = 240.0

# (checkpoint leaf, which layers). attn = the 6 full-attention layers, gdn = the 18 linear ones.
MATMULS = [
    ("self_attn.q_proj.weight", "attn"),
    ("self_attn.k_proj.weight", "attn"),
    ("self_attn.v_proj.weight", "attn"),
    ("self_attn.o_proj.weight", "attn"),
    ("linear_attn.in_proj_qkv.weight", "gdn"),
    ("linear_attn.in_proj_z.weight", "gdn"),
    ("linear_attn.in_proj_a.weight", "gdn"),
    ("linear_attn.in_proj_b.weight", "gdn"),
    ("linear_attn.out_proj.weight", "gdn"),
    ("mlp.gate_proj.weight", "all"),
    ("mlp.up_proj.weight", "all"),
    ("mlp.down_proj.weight", "all"),
]


def fmt_b(x: float) -> str:
    for unit, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if x >= div:
            return f"{x / div:.4g} {unit}"
    return f"{x:.0f} B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.5-0.8B"))
    ap.add_argument("--waves", default="64,512,1024,3072")
    ap.add_argument("--rate", default="fp16", choices=sorted(RATES_TFLOPS))
    ap.add_argument("--bw-gbs", type=float, default=DEFAULT_BW_GBS)
    args = ap.parse_args()

    model = ROOT / args.model if not Path(args.model).is_absolute() else Path(args.model)
    header = wc.header_shapes(model)
    C = wc.consts()
    interval = json.loads((model / "config.json").read_text())["text_config"][
        "full_attention_interval"]
    attn = [i for i in range(C["kNumLayers"]) if (i + 1) % interval == 0]
    counts = {"attn": len(attn), "gdn": C["kNumLayers"] - len(attn), "all": C["kNumLayers"]}

    rows = []
    for leaf, scope in MATMULS:
        L = attn[0] if scope == "attn" else 0
        dtype, (n, k) = header[f"model.language_model.layers.{L}.{leaf}"]
        rows.append({"leaf": leaf, "scope": scope, "count": counts[scope], "n": n, "k": k})
    head_dtype, (hn, hk) = header["model.language_model.embed_tokens.weight"]
    rows.append({"leaf": "lm_head (= embed, §01.1)", "scope": "once", "count": 1, "n": hn, "k": hk})

    waves = [int(w) for w in args.waves.split(",")]
    esz = 2                                   # bf16 and fp16 are the same width: the whole point (a)
    rate = RATES_TFLOPS[args.rate] * 1e12
    ridge = rate / (args.bw_gbs * 1e9)        # flops/byte where compute and traffic tie

    print(f"model {model.name}; operands {args.rate} in / f32 accumulate / 16-bit C (see the")
    print(f"constraint at the bottom). rate = {RATES_TFLOPS[args.rate]} TFLOP/s and bandwidth =")
    print(f"{args.bw_gbs} GB/s, BOTH QUOTED from devices/ — neither is measured by this script.")
    print(f"ridge = rate / bandwidth = {ridge:.2f} flops per byte = {1 / ridge:.5f} bytes per flop.")
    print("A shape below the ridge is bandwidth-bound however good the kernel is; that is a statement")
    print("about the shape, not about the route. Byte columns are the MINIMUM traffic (each operand")
    print("touched once); a badly blocked kernel re-reads A per N-tile and W per M-tile, so real")
    print("numbers can only be worse than what is printed.\n")

    block_flops = sum(2 * r["n"] * r["k"] * r["count"] for r in rows if r["scope"] != "once")
    head_flops = 2 * hn * hk
    print(f"flop accounting: {block_flops / 1e9:.4f} GFLOP per token in block matmuls "
          f"({sum(r['count'] for r in rows if r['scope'] != 'once')} matmuls), "
          f"+ {head_flops / 1e9:.4f} GFLOP per priced logits row.")
    print(f"The plan divides its ceilings by ~1.6 GFLOP/token; the matmul part alone is "
          f"{(block_flops + head_flops) / 1e9:.4f} GFLOP/token when every token's logits are priced,")
    print("and prefill prices ONE row per request (§06.5), so the prefill figure is the block part.\n")

    for M in waves:
        print(f"=== wave of M = {M} rows")
        print(f"  {'matmul':<34s}{'n/layers':>8s}{'M':>6s}{'N':>8s}{'K':>6s}{'GFLOP':>9s}"
              f"{'weights':>11s}{'act r+w':>11s}{'B/flop':>9s}{'f/B':>7s}  bound")
        tot_f = tot_w = tot_a = 0.0
        for r in rows:
            n, k, cnt = r["n"], r["k"], r["count"]
            m = 1 if r["scope"] == "once" else M
            flops = 2.0 * m * n * k * cnt
            wbytes = esz * n * k * cnt
            if r["scope"] == "once":
                wbytes = esz * n * k           # streamed once per request, whatever the wave
            abytes = esz * (m * k + m * n) * cnt
            f = flops / (wbytes + abytes)
            print(f"  {r['leaf']:<34s}{cnt:>8d}{m:>6d}{n:>8d}{k:>6d}{flops / 1e9:>9.3f}"
                  f"{fmt_b(wbytes):>11s}{fmt_b(abytes):>11s}{1 / f:>9.4f}{f:>7.1f}"
                  f"  {'compute' if f >= ridge else 'BANDWIDTH'}")
            tot_f += flops
            tot_w += wbytes
            tot_a += abytes
        agg = tot_f / (tot_w + tot_a)
        print(f"  {'TOTAL':<34s}{'':>8s}{'':>6s}{'':>8s}{'':>6s}{tot_f / 1e9:>9.3f}"
              f"{fmt_b(tot_w):>11s}{fmt_b(tot_a):>11s}{1 / agg:>9.4f}{agg:>7.1f}"
              f"  {'compute' if agg >= ridge else 'BANDWIDTH'}")
        print(f"    arithmetic on the quoted rate: {tot_f / rate * 1e3:.2f} ms of matmul for the wave "
              f"({tot_f / M / 1e9:.4f} GFLOP/token). Weights are "
              f"{100 * tot_w / (tot_w + tot_a):.0f} % of the {fmt_b(tot_w + tot_a)} touched per wave.")
        skinny = [r for r in rows if r["scope"] != "once" and r["n"] * 1.0 / r["k"] < 0.05]
        if skinny and M >= 512:
            sk = ", ".join(f"{r['leaf']}({r['count']}/layer)" for r in skinny)
            print(f"    N/K < 0.05 at this width: {sk} — these are the shapes a square probe says")
            print("    nothing about, and they are below the ridge at every wave width.")
        print()

    print("=== epilogue bill: what a library matmul unfuses, in extra bytes per wave")
    print("  'fused today' is what the engine already does inside the matmul kernel.")
    sites = [
        ("silu(gate)*up (mlp_gateup)", "yes — inside the GEMM",
         lambda m: C["kNumLayers"] * 4 * m * C["kIntermediate"] * esz),
        ("attn gate de-interleave", "no — attn_prep is its own", lambda m: 0),
        ("q/k head-norm + rope", "no — attn_prep is its own", lambda m: 0),
        ("gdn gated RMSNorm (§04.5)", "no — own kernel already", lambda m: 0),
        ("residual add + next RMSNorm", "no — glue is its own pass", lambda m: 0),
    ]
    print(f"  {'site':<34s}{'fused today?':<24s}" + "".join(f"{'M=' + str(m):>12s}" for m in waves))
    for name, fused, fn in sites:
        print(f"  {name:<34s}{fused:<24s}" + "".join(f"{fmt_b(fn(m)):>12s}" for m in waves))
    print("  Only the MLP costs anything. `mlp_gateup` writes `act` once; a library gate+up writes both")
    print("  halves (2 tiles), a silu-mul kernel reads both (2 tiles) and writes act (1 tile) = 5 tiles")
    print("  of M x 3584 x 2 B per layer instead of 1. Concatenating gate|up into one [7168,1024] weight")
    print("  merges the two calls but does NOT change that count — it buys launches, not traffic.")
    print("  The three attention/GDN epilogues are already separate passes reading a matmul output that")
    print("  is already written to memory, so a library matmul changes nothing there.")
    print("\n  What the library route gives BACK: `proj_gemv` runs PACK_X=false and so needs F32")
    print("  activations, which is most of why 115 cast launches per token exist (engine/kernels_proj.cpp,")
    print("  `engine/results.md` E2). A library GEMM taking a 16-bit A operand reads what glue wrote, so")
    print("  those casts disappear for the projections — a launch-and-dispatch saving at the measured")
    print("  dispatch floor, not a traffic one.")
    print("\n  CONSTRAINT no shape table can express — the rounding point. Every shipped pin rounds a")
    print("  projection's product to BF16 where the reference does, so a library call must be given a")
    print("  16-bit C dtype (HIPBLAS_COMPUTE_32F with C bf16/fp16). Asking for F32 C would both double")
    print("  the activation traffic above and move a rounding, and a moved rounding is a route change")
    print("  (D-3/D-5), not a tuning knob.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
