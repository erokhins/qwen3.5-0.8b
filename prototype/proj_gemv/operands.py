#!/usr/bin/env python3
"""Reconstruct the projection operands from the pins — and prove the reconstruction is right.

A projection kernel is only gated if the tensor that goes *into* it is known. The W1 pins give the
outputs (`decode{N}.L{ii}.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,q_proj,k_proj,
v_proj,o_proj}.out`) but not the inputs, so each input is rebuilt from other pins:

    x_normed = RMSnorm_zc(L{ii}.s_in, input_layernorm.weight)       # §03.1/§03.2 -> §04.1/§05.1
    gdn_in   = L{ii}.gdn.gated_out                                  # §04.5's gated norm output
    attn_in  = L{ii}.attn.attn_out_pre_gate * sigmoid(gate)          # §05.6, gate from q_proj.out

**None of that is trusted from the name.** `verify()` recomputes each pinned projection output
from the rebuilt operand with torch's own BF16 matmul and requires bit equality — §10.2 already
claims "the raw projections reproduce the reference bit-exactly", so anything short of ndiff 0
means the operand is wrong and the gate would be measuring the wrong tensor (the `mlp_gemv` `act`
trap, §10.5). `verify()` also requires the wrong readings *not* to reproduce it: the raw `s_in`,
the other norm's weight, the norm-without-affine form, and §05.1's contiguous-half gate. A
reconstruction that could not be wrong proves nothing.

The norm comes from `speccheck.specimpl.rmsnorm_zc`, so the operand itself comes from the
independent torch CPU implementation rather than being hand-rolled here.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from speccheck.specimpl import rmsnorm_zc  # noqa: E402  (S03.1, torch CPU)

CKPT = ROOT / "models" / "Qwen3.5-0.8B"
LAYER_PREFIX = "model.language_model.layers.%d."

# Packed-launch layouts, in the row order the engine will concatenate them: §04.1 (raw =
# [q key_dim | k key_dim | v value_dim]), §05.1 (q_proj rows are [8 heads][q 256 | gate 256]),
# widths from §01.2/§01.3.
GDN_PACK = [("in_proj_qkv", 6144), ("in_proj_z", 2048), ("in_proj_b", 16), ("in_proj_a", 16)]
ATTN_PACK = [("q_proj", 4096), ("k_proj", 512), ("v_proj", 512)]
H_KV, D_KV = 16, 128          # §00: 16 value heads x 128 = the 2048 lanes §04.5 normalizes
N_Q, D_HEAD = 8, 256

_SHARDS: dict[str, object] = {}
_INDEX: dict[str, str] = {}


def ckpt_tensor(key: str) -> torch.Tensor:
    """One tensor by its full §01 checkpoint name. Shard handles stay open: this is called once
    per projection per layer per step and re-opening the file each time dominates the gate."""
    if not _INDEX:
        for f in sorted(glob.glob(str(CKPT / "*.safetensors"))):
            _SHARDS[f] = safe_open(f, "pt")
            for k in _SHARDS[f].keys():
                _INDEX[k] = f
    if key not in _INDEX:
        raise KeyError(key)
    return _SHARDS[_INDEX[key]].get_tensor(key)


def proj_weight(layer: int, name: str) -> torch.Tensor:
    """The §01 checkpoint key for one projection (the GDN and self_attn namespaces differ)."""
    pre = LAYER_PREFIX % layer
    if name.startswith(("in_proj", "out_proj")):
        return ckpt_tensor(pre + "linear_attn." + name + ".weight")
    return ckpt_tensor(pre + "self_attn." + name + ".weight")


def dump_bf16(t: torch.Tensor, path: Path) -> Path:
    """torch refuses bf16.numpy(); view the bits instead (prototype/README.md)."""
    t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tofile(path)
    return path


def read_bf16(path: Path, n: int) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16)[:n]).view(torch.bfloat16)


def pin(d: Path, name: str) -> torch.Tensor:
    return torch.from_numpy(np.load(d / (name + ".npy"))).to(torch.bfloat16).reshape(-1)


def is_attn_layer(d: Path, layer: int, step: str = "decode0") -> bool:
    """Which block type a layer is, read off the pins themselves — §10.5 says the pinned names are
    the check to trust rather than a transcription of `layer_types`. `step=""` asks the PREFILL
    pass, whose names carry no step prefix (`L03.q_proj.out` rather than `decode0.L03.q_proj.out`)."""
    stem = "L%02d.q_proj.out" % layer if not step else "%s.L%02d.q_proj.out" % (step, layer)
    return (d / (stem + ".npy")).exists()


def pack_offsets(pack):
    """name -> (row offset, rows) inside one packed launch."""
    off, out = 0, {}
    for name, n in pack:
        out[name] = (off, n)
        off += n
    return out


def normed_input(d: Path, step: str, layer: int) -> torch.Tensor:
    """x entering §04.1 / §05.1: the zero-centered RMSNorm of the layer input (§03.2)."""
    s_in = pin(d, "%s.L%02d.s_in" % (step, layer))
    return rmsnorm_zc(s_in, ckpt_tensor(LAYER_PREFIX % layer + "input_layernorm.weight"))


def wrong_normed_inputs(d: Path, step: str, layer: int) -> dict[str, torch.Tensor]:
    """The plausible mistakes about which norm feeds a projection (§03.2 order), so the gate can
    require each of them to be refused."""
    s_in = pin(d, "%s.L%02d.s_in" % (step, layer))
    pre = LAYER_PREFIX % layer
    return {
        "pre-norm s_in": s_in,
        "post_attention_layernorm output": rmsnorm_zc(
            s_in, ckpt_tensor(pre + "post_attention_layernorm.weight")),
        "norm without the (1+w) affine": (s_in.float()
                                         * torch.rsqrt((s_in.float() ** 2).mean(-1, keepdim=True)
                                                       + 1e-6)).to(torch.bfloat16),
    }


def gdn_operand(d: Path, step: str, layer: int) -> torch.Tensor:
    """out_proj's operand: §04.5's gated RMSNorm output, [16 heads][128] -> 2048 lanes."""
    return pin(d, "%s.L%02d.gdn.gated_out" % (step, layer))


def attn_operand(d: Path, step: str, layer: int, interleave: str = "per-head") -> torch.Tensor:
    """o_proj's operand: §05.6 `attn_out * sigmoid(gate)`, the gate read out of the pinned q_proj
    output. `interleave='half'` is the §05.1 mistake, kept for the negative tests."""
    pre_gate = pin(d, "%s.L%02d.attn.attn_out_pre_gate" % (step, layer))
    qg = pin(d, "%s.L%02d.q_proj.out" % (step, layer))
    if interleave == "per-head":
        gate = qg.reshape(N_Q, D_HEAD * 2)[:, D_HEAD:].reshape(-1)
    elif interleave == "half":
        gate = qg.reshape(2, N_Q * D_HEAD)[1]
    else:
        raise ValueError(interleave)
    return pre_gate * torch.sigmoid(gate)


def operands_for(d: Path, step: str, layer: int) -> dict:
    """What one layer's projection launches consume: two x vectors, the pack layout, and the name
    of that layer's output projection."""
    if is_attn_layer(d, layer):
        return {"kind": "attn", "x_in": normed_input(d, step, layer).reshape(-1),
                "x_out": attn_operand(d, step, layer).reshape(-1), "pack": ATTN_PACK,
                "out_proj": "o_proj"}
    return {"kind": "gdn", "x_in": normed_input(d, step, layer).reshape(-1),
            "x_out": gdn_operand(d, step, layer).reshape(-1), "pack": GDN_PACK,
            "out_proj": "out_proj"}


# ---- the PREFILL pass: the same operands as [T, in] tiles (§06.1, plan P5) ----------------
#
# The pins are the prefill-pass ones, `L{ll}.s_in` / `L{ll}.gdn.gated_out` /
# `L{ll}.attn.attn_out_pre_gate` / `L{ll}.{projection}.out`, whose leading axes are just the batch
# and the token count (§10.5: every batched kernel family already has a prefill pin). Nothing about
# the reconstruction changes — the same RMSNorm, the same gated norm, the same §05.6 output gate —
# which is the point: a token-batched projection is gated against the same tensors the k=1 gate is,
# at T rows instead of one. `prefill_verify` re-runs the same proof per row, because a mispaired
# tile is the §10.5 trap and it must be refused here exactly as it is at k=1.

def pin_tile(d: Path, name: str, cols: int) -> torch.Tensor:
    """One prefill pin as a [T, cols] BF16 tile (the pins carry F32 files of BF16-representable
    values, the same cast `pin()` makes at k=1)."""
    return torch.from_numpy(np.load(d / (name + ".npy"))).to(torch.bfloat16).reshape(-1, cols)


def prefill_tokens(d: Path, layer: int) -> int:
    """How many rows this prefill fixture pins for one layer — read off `s_in`, never assumed."""
    return int(np.load(d / ("L%02d.s_in.npy" % layer)).shape[1])


def prefill_normed_input(d: Path, layer: int) -> torch.Tensor:
    s_in = pin_tile(d, "L%02d.s_in" % layer, 1024)
    return rmsnorm_zc(s_in, ckpt_tensor(LAYER_PREFIX % layer + "input_layernorm.weight"))


def prefill_wrong_normed_inputs(d: Path, layer: int) -> dict[str, torch.Tensor]:
    """The tile versions of the k=1 wrong readings: same three mistakes, at T rows."""
    s_in = pin_tile(d, "L%02d.s_in" % layer, 1024)
    pre = LAYER_PREFIX % layer
    return {
        "pre-norm s_in": s_in,
        "post_attention_layernorm output": rmsnorm_zc(
            s_in, ckpt_tensor(pre + "post_attention_layernorm.weight")),
        "norm without the (1+w) affine": (s_in.float()
                                         * torch.rsqrt((s_in.float() ** 2).mean(-1, keepdim=True)
                                                       + 1e-6)).to(torch.bfloat16),
    }


def prefill_operands_for(d: Path, layer: int) -> dict:
    """The tiles one layer's two launches consume on the prefill pass."""
    if is_attn_layer(d, layer, step=""):
        pre_gate = pin_tile(d, "L%02d.attn.attn_out_pre_gate" % layer, N_Q * D_HEAD)
        qg = pin_tile(d, "L%02d.q_proj.out" % layer, 2 * N_Q * D_HEAD)
        gate = qg.reshape(-1, N_Q, D_HEAD * 2)[:, :, D_HEAD:].reshape(-1, N_Q * D_HEAD)
        return {"kind": "attn", "x_in": prefill_normed_input(d, layer),
                "x_out": pre_gate * torch.sigmoid(gate), "pack": ATTN_PACK, "out_proj": "o_proj"}
    return {"kind": "gdn", "x_in": prefill_normed_input(d, layer),
            "x_out": pin_tile(d, "L%02d.gdn.gated_out" % layer, H_KV * D_KV),
            "pack": GDN_PACK, "out_proj": "out_proj"}


def prefill_verify(d: Path, layer: int) -> list[tuple[bool, str]]:
    """(ok, line) per projection, over the whole tile: the rebuilt operand must reproduce every row
    of the pin BIT-EXACTLY with torch's own matmul, and each wrong reading must fail to.

    The pin widths are the projection's own row count, so a mispaired file (the §10.5
    last-writer-wins trap) shows up as a shape or bit mismatch rather than as a quiet kernel
    disagreement."""
    ops = prefill_operands_for(d, layer)
    out_name = ops["out_proj"]
    out = []
    for name, rows in ops["pack"]:
        ref = pin_tile(d, "L%02d.%s.out" % (layer, name), rows)
        ndiff = int(((ops["x_in"] @ proj_weight(layer, name).t()).reshape(-1)
                     != ref.reshape(-1)).sum())
        out.append((ndiff == 0, "%s: rebuilt tile reproduces the pin (ndiff %d/%d)"
                    % (name, ndiff, ref.numel())))
    ref = pin_tile(d, "L%02d.%s.out" % (layer, out_name), 1024)
    ndiff = int(((ops["x_out"] @ proj_weight(layer, out_name).t()).reshape(-1)
                 != ref.reshape(-1)).sum())
    out.append((ndiff == 0, "%s: rebuilt tile reproduces the pin (ndiff %d/%d)"
                % (out_name, ndiff, ref.numel())))

    first = ops["pack"][0][0]
    first_rows = ops["pack"][0][1]
    ref = pin_tile(d, "L%02d.%s.out" % (layer, first), first_rows)
    bad = [label for label, xx in prefill_wrong_normed_inputs(d, layer).items()
           if int(((xx @ proj_weight(layer, first).t()).reshape(-1) != ref.reshape(-1)).sum()) == 0]
    out.append((not bad, "wrong-norm readings rejected: %s" % (
        "pre-norm, other norm, no-affine" if not bad else "STILL MATCHES " + ",".join(bad))))
    return out


def projections_of(d: Path, layer: int, step: str = "decode0") -> list[str]:
    if is_attn_layer(d, layer, step):
        return [n for n, _ in ATTN_PACK] + ["o_proj"]
    return [n for n, _ in GDN_PACK] + ["out_proj"]


def verify(d: Path, step: str, layer: int) -> list[tuple[bool, str]]:
    """(ok, line) per projection: the rebuilt operand must reproduce the pin BIT-EXACTLY with
    torch's own matmul, and each wrong reading must fail to."""
    ops = operands_for(d, step, layer)
    out = []
    for name in projections_of(d, layer):
        ref = pin(d, "%s.L%02d.%s.out" % (step, layer, name))
        xx = ops["x_in"] if name != ops["out_proj"] else ops["x_out"]
        ndiff = int(((xx.reshape(1, -1) @ proj_weight(layer, name).t()).reshape(-1)
                     != ref).sum())
        out.append((ndiff == 0, "%s: rebuilt operand reproduces the pin (ndiff %d/%d)"
                    % (name, ndiff, ref.numel())))
    bad = []
    for label, xx in wrong_normed_inputs(d, step, layer).items():
        name = projections_of(d, layer)[0]
        ref = pin(d, "%s.L%02d.%s.out" % (step, layer, name))
        if int(((xx.reshape(1, -1) @ proj_weight(layer, name).t()).reshape(-1) != ref).sum()) \
                == 0:
            bad.append(label)
    out.append((not bad, "wrong-norm readings rejected: %s" % (
        "pre-norm, other norm, no-affine" if not bad else "STILL MATCHES " + ",".join(bad))))
    if ops["kind"] == "attn":
        ref = pin(d, "%s.L%02d.o_proj.out" % (step, layer))
        wrong = attn_operand(d, step, layer, interleave="half")
        out.append((int(((wrong.reshape(1, -1) @ proj_weight(layer, "o_proj").t()).reshape(-1)
                         != ref).sum()) > 0,
                    "contiguous-half gate (§05.1) does not reproduce o_proj"))
    return out
