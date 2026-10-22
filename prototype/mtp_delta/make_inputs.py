#!/usr/bin/env python3
"""D3 prep: the inputs the delta arms need. CPU-only — it launches nothing.

Writes into WORK (default /tmp/mtp_delta):

  weights/     the engine's own proj and mlp weight blobs, raw BF16, rank- and shape-checked
               against the shard header (AGENTS: a shape claim typed from memory is a bug with a
               compile-time guarantee it will be wrong eventually)
  x/           synthetic GEMV input tiles, one per (input width, token count) the arms need
  attn/        synthetic attention inputs: q/k/v rows and a [2][cap][256] BF16 cache per attn layer
  plan_*.txt   the four+one rig plans (proj t1/t2, attn n1/n2), which the runner passes verbatim

Why synthetic inputs are admissible HERE and not in a correctness gate: these rigs' rule that "nothing
is synthesized" exists so a numerics verdict cannot drift from the pinned reference. Timing does not
depend on the values — bf16 FMA on this part is constant-time, and the kernels' control flow does not
read the data — while every address, byte count and launch geometry comes from the shapes, which are the
engine's own. The geometry is therefore not asserted by this file but echoed back by the rigs
(`tokens=`, `rows=`, `core_grid=`) and checked by the runner's canary.

    .venv/bin/python prototype/mtp_delta/make_inputs.py [--work /tmp/mtp_delta] [--verify]

`--verify` re-reads every file size against every plan line and the shapes the rigs require, touches no
device, and exits non-zero on any mismatch (including "the 2-row x tile is not exactly twice the 1-row
tile", which is the mistake that would make both arms run the same shape).
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import sys
import torch
from pathlib import Path
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models" / "Qwen3.5-0.8B" / "model.safetensors-00001-of-00001.safetensors"
PRE = "model.language_model.layers."

# Shapes as the engine issues them (engine/generation-limits.md §1 launch counts, `proj_gemv.hip.cpp`
# header). Checked against the shard, not trusted from here.
PROJ_SHAPES = {
    "linear_attn.in_proj_qkv": (6144, 1024), "linear_attn.in_proj_z": (2048, 1024),
    "linear_attn.in_proj_b": (16, 1024), "linear_attn.in_proj_a": (16, 1024),
    "linear_attn.out_proj": (1024, 2048),
    "self_attn.q_proj": (4096, 1024), "self_attn.k_proj": (512, 1024), "self_attn.v_proj": (512, 1024),
    "self_attn.o_proj": (1024, 2048),
}
MLP_SHAPES = {"mlp.gate_proj": (3584, 1024), "mlp.up_proj": (3584, 1024), "mlp.down_proj": (1024, 3584)}
# 6 attn layers at every 4th index; verified against the shard by the presence of self_attn.q_proj.
ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
GDN_LAYERS = [i for i in range(24) if i not in ATTN_LAYERS]
LANES = 4032                 # the operating point: 4,000 prompt + 32 steps
CAP = 4096                   # cache rows allocated; must exceed lanes_last of the 2-row arm
N_Q, N_KV, DH = 8, 2, 256


def dump_raw(t: torch.Tensor, path: Path):
    """BF16 tensor -> raw little-endian BF16 bits. `.view(torch.uint16)` BITCASTS; `.to(uint16)` CASTS
    (the trap that wrote garbage into a fixture in prototype/gdn_gated_norm)."""
    assert t.dtype == torch.bfloat16, f"{path.name}: {t.dtype}, expected bf16"
    t.contiguous().view(torch.uint16).numpy().tofile(path)


def extract(work: Path, mlp_layers):
    wdir = work / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    want = {}
    for i in GDN_LAYERS:
        for suf in PROJ_SHAPES:
            if suf.startswith("linear_attn."):
                want[f"{PRE}{i}.{suf}.weight"] = (i, suf)
    for j in ATTN_LAYERS:
        for suf in PROJ_SHAPES:
            if suf.startswith("self_attn."):
                want[f"{PRE}{j}.{suf}.weight"] = (j, suf)
    for i in mlp_layers:
        for suf in MLP_SHAPES:
            want[f"{PRE}{i}.{suf}.weight"] = (i, suf)
    got, manifest = 0, {}
    with safe_open(str(CKPT), framework="pt") as f:
        for key, (idx, suf) in want.items():
            if key not in f.keys():
                raise SystemExit(f"shard has no {key} — the layer map here is wrong")
            t = f.get_tensor(key)
            shape = tuple(t.shape)
            exp = (PROJ_SHAPES.get(suf) or MLP_SHAPES[suf])
            if shape != exp or len(shape) != 2:
                raise SystemExit(f"{key}: rank {len(shape)} {shape}, expected {exp}")
            name = f"L{idx:02d}_{suf.replace('.', '_')}.bin"
            path = wdir / name
            if not path.exists():
                dump_raw(t, path)
            manifest[key] = {"file": str(path), "shape": list(shape), "bytes": path.stat().st_size}
            got += 1
    (work / "weights.json").write_text(json.dumps(manifest, indent=1))
    print(f"extracted {got} weight blobs ({sum(m['bytes'] for m in manifest.values())/2**20:.1f} MiB)")
    return manifest


def synth_x(work: Path, widths=(1024, 2048, 3584), tokens=(1, 2, 4)):
    """Synthetic GEMV x tiles: values in [-1,1], deterministic, one file per (width, tokens)."""
    xdir = work / "x"
    xdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261022)
    for inw in widths:
        for t in tokens:
            v = rng.uniform(-1.0, 1.0, size=(t, inw)).astype(np.float32)
            torch.from_numpy(v).to(torch.bfloat16).view(torch.uint16).numpy().tofile(
                xdir / f"x{inw}_t{t}.bin")


def synth_attn(work: Path):
    """Synthetic attention cases at the operating point: 6 attn layers, one cache each.

    Geometry (checked by the rig, not here): `pos=4031, lvalid=4032` gives lanes_last 4,032 for the
    1-row arm and 4,033 for the 2-row arm, i.e. row 1 reads exactly the lane row 0 appended — the
    causal shape of a verify body, at the width the ledger's attn line was taken.
    """
    adir = work / "attn"
    adir.mkdir(parents=True, exist_ok=True)
    outdir = adir / "out"
    outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)

    def bf16(shape, lo=-1.0, hi=1.0):
        v = rng.uniform(lo, hi, size=shape).astype(np.float32)
        return torch.from_numpy(v).to(torch.bfloat16).view(torch.uint16).numpy()

    for nrows in (1, 2):
        for i, layer in enumerate(ATTN_LAYERS):
            # qg/kg/vg are per-row vectors, so the file size must scale with nrows (the rig checks it).
            (adir / f"qg_n{nrows}.bin").write_bytes(bf16((nrows, N_Q * 2 * DH)).tobytes())
            (adir / f"kg_n{nrows}.bin").write_bytes(bf16((nrows, N_KV * DH)).tobytes())
            (adir / f"vg_n{nrows}.bin").write_bytes(bf16((nrows, N_KV * DH)).tobytes())
            if nrows == 1:                      # shared per-layer state: cache, norms
                (adir / f"qn_{layer}.bin").write_bytes(bf16((DH,)).tobytes())
                (adir / f"kn_{layer}.bin").write_bytes(bf16((DH,)).tobytes())
                (adir / f"k_{layer}.bin").write_bytes(bf16((2, CAP, DH)).tobytes())
                (adir / f"v_{layer}.bin").write_bytes(bf16((2, CAP, DH)).tobytes())
    return outdir


def plans(work: Path, outdir: Path):
    """The rig plan files. Same packed launch shape the engine issues: 8224 rows on the GDN block,
    5120 on the attention block, out_proj/o_proj separate at in=2048 — 48 launches, the ledger's
    `proj` launch count, reached by enumeration rather than typed."""
    w = work / "weights"

    def proj_plan(tokens):
        lines = []
        for i in GDN_LAYERS:
            lines.append(f"L{i:02d}.inproj 8224 1024 {work}/x/x1024_t{tokens}.bin "
                         f"{w}/L{i:02d}_linear_attn_in_proj_qkv.bin,"
                         f"{w}/L{i:02d}_linear_attn_in_proj_z.bin,"
                         f"{w}/L{i:02d}_linear_attn_in_proj_b.bin,"
                         f"{w}/L{i:02d}_linear_attn_in_proj_a.bin")
            lines.append(f"L{i:02d}.out_proj 1024 2048 {work}/x/x2048_t{tokens}.bin "
                         f"{w}/L{i:02d}_linear_attn_out_proj.bin")
        for j in ATTN_LAYERS:
            lines.append(f"L{j:02d}.inproj 5120 1024 {work}/x/x1024_t{tokens}.bin "
                         f"{w}/L{j:02d}_self_attn_q_proj.bin,"
                         f"{w}/L{j:02d}_self_attn_k_proj.bin,"
                         f"{w}/L{j:02d}_self_attn_v_proj.bin")
            lines.append(f"L{j:02d}.o_proj 1024 2048 {work}/x/x2048_t{tokens}.bin "
                         f"{w}/L{j:02d}_self_attn_o_proj.bin")
        return lines

    (work / "plan_proj_t1.txt").write_text("\n".join(proj_plan(1)) + "\n")
    (work / "plan_proj_t2.txt").write_text("\n".join(proj_plan(2)) + "\n")
    # The tile/occupancy control needs an in=1024-ONLY plan: the binary instantiates TPB 4 at in=1024
    # and TPB 2 at in=2048 and refuses anything else, so `--tpb 4` over the mixed plan would die on the
    # out_proj jobs rather than price anything.
    (work / "plan_proj_i1024_t1.txt").write_text(
        "\n".join(l for l in proj_plan(1) if l.split()[2] == "1024") + "\n")

    def attn_plan(nrows):
        lines = []
        for layer in ATTN_LAYERS:
            a = work / "attn"
            lines.append(f"attn-L{layer:02d} {a}/qg_n{nrows}.bin {a}/kg_n{nrows}.bin "
                         f"{a}/vg_n{nrows}.bin {a}/k_{layer}.bin {a}/v_{layer}.bin "
                         f"{a}/qn_{layer}.bin {a}/kn_{layer}.bin {LANES - 1} {LANES} {outdir} {nrows}")
        return lines

    (work / "plan_attn_n1.txt").write_text("\n".join(attn_plan(1)) + "\n")
    (work / "plan_attn_n2.txt").write_text("\n".join(attn_plan(2)) + "\n")
    print("wrote plan_proj_t1.txt plan_proj_t2.txt plan_attn_n1.txt plan_attn_n2.txt")


def verify(work: Path) -> int:
    bad = []
    man = json.loads((work / "weights.json").read_text())
    for key, m in man.items():
        if not Path(m["file"]).exists():
            bad.append(f"missing {m['file']}")
        elif Path(m["file"]).stat().st_size != m["bytes"] or m["bytes"] != 2 * int(
                np.prod(m["shape"])):
            bad.append(f"{key}: file size {Path(m['file']).stat().st_size} != bf16 {m['shape']}")
    for inw in (1024, 2048, 3584):
        sizes = {}
        for t in (1, 2, 4):
            p = work / "x" / f"x{inw}_t{t}.bin"
            sizes[t] = p.stat().st_size if p.exists() else -1
            if sizes[t] != t * inw * 2:
                bad.append(f"{p.name}: {sizes[t]} bytes, want {t * inw * 2}")
        if sizes[1] * 2 != sizes[2]:
            bad.append(f"in={inw}: the 2-row tile is not exactly 2x the 1-row tile — both arms would "
                       f"run different byte counts than the ratio assumes")
    for plan, want_launches in (("plan_proj_t1.txt", 48), ("plan_proj_t2.txt", 48),
                                ("plan_proj_i1024_t1.txt", 24)):
        lines = [l for l in (work / plan).read_text().splitlines() if l.strip()]
        if len(lines) != want_launches:
            bad.append(f"{plan}: {len(lines)} launches, want {want_launches} (the ledger's proj count)")
        for ln in lines:
            f = ln.split()
            rows, inw, xfile = int(f[1]), int(f[2]), f[3]
            toks = 1 if "t1" in xfile else 2
            if Path(xfile).stat().st_size != toks * inw * 2:
                bad.append(f"{plan}: {f[0]} x file is not tokens*in bf16")
            for wf in f[4].split(","):
                if not Path(wf).exists():
                    bad.append(f"{plan}: no weight file {wf}")
    for plan, nrows in (("plan_attn_n1.txt", 1), ("plan_attn_n2.txt", 2)):
        lines = [l for l in (work / plan).read_text().splitlines() if l.strip()]
        if len(lines) != len(ATTN_LAYERS):
            bad.append(f"{plan}: {len(lines)} cases, want {len(ATTN_LAYERS)} attn layers")
        for ln in lines:
            f = ln.split()
            pos, lvalid, got_rows = int(f[8]), int(f[9]), int(f[11])
            lanes_last = lvalid + got_rows - 1
            if got_rows != nrows:
                bad.append(f"{plan}: NROWS={got_rows}")
            if lanes_last > CAP or pos + got_rows - 1 >= CAP:
                bad.append(f"{plan}: lanes_last {lanes_last} does not fit cap {CAP}")
            for pth in f[1:8]:
                sz = Path(pth).stat().st_size
                if sz == 0:
                    bad.append(f"{plan}: empty {pth}")
            qg = Path(f[1]).stat().st_size
            if qg != nrows * N_Q * 2 * DH * 2:
                bad.append(f"{plan}: qg is {qg} B, want {nrows * N_Q * 2 * DH * 2} for {nrows} rows")
            kc = Path(f[4]).stat().st_size
            if kc != 2 * CAP * DH * 2:
                bad.append(f"{plan}: cache is {kc} B, want [2][{CAP}][{DH}] bf16")
    print("VERIFY", "PASS" if not bad else "FAIL")
    for b in bad:
        print("  " + b)
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="/tmp/mtp_delta")
    ap.add_argument("--mlp-layers", default="0,3,11,23",
                    help="layers whose MLP weights to extract; the runner prices these and scales "
                         "by len(this)/24 (weight shapes are identical across layers)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    work = Path(args.work)
    if args.verify:
        return verify(work)
    work.mkdir(parents=True, exist_ok=True)
    mlp_layers = [int(x) for x in args.mlp_layers.split(",")]
    extract(work, mlp_layers)
    synth_x(work)
    outdir = synth_attn(work)
    plans(work, outdir)
    return verify(work)


if __name__ == "__main__":
    raise SystemExit(main())
