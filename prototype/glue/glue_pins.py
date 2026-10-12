#!/usr/bin/env python3
"""The pinned tensors one glue call consumes, and the engine names its weight comes from.

`data/refs/dec8/` pins the residual chain per layer — `decode{N}.L{ii}.s_in` (the residual going
in) and `decode{N}.L{ii}.s_out` (the residual coming out) — plus each addend (`...out_proj.out` /
`...o_proj.out` for the block, `...mlp.out` for the MLP), the post-attention norm's output
(`...mlp.in`, which is exactly glue call 1's `out`) and the final norm's output (`final_hidden`).
So a glue kernel does not need new dumps; it needs the right pairing, and §10.5's trap is that the
bare `final_hidden` / `pre_final_hidden` names are last-writer-wins across phases: `final_pair()`
below *finds* which decode step they belong to by bit-comparison instead of assuming one.

Nothing here is trusted from the name. `run.py --verify-only`/the gate's stage R recomputes each
pin from these pairings with torch's own BF16 arithmetic and requires bit equality, and requires
the plausible misreadings (gated norm, plain `w`, normalising the addend) to fail.

Weight names come in two flavours and they are not interchangeable: the §01 checkpoint key
(`model.language_model.layers.N.*`) for reading the file, and the §02.5 normalised engine name
(`lm.layers.N.*`) for fetching the resident blob through `engine::WeightStore`. The kernel gets
its weights only by the second form — that is the loader's first real consumer.
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
CKPT_LAYER = "model.language_model.layers.%d."
CKPT_FINAL = "model.language_model.norm.weight"
ENG_LAYER = "lm.layers.%d."
ENG_FINAL = "lm.norm.weight"
H = 1024

_SHARDS: dict[str, object] = {}
_INDEX: dict[str, str] = {}


def ckpt_tensor(key: str) -> torch.Tensor:
    """One checkpoint tensor by its §01 name; shard handles stay open (this is called per layer)."""
    if not _INDEX:
        for f in sorted(glob.glob(str(CKPT / "*.safetensors"))):
            _SHARDS[f] = safe_open(f, "pt")
            for k in _SHARDS[f].keys():
                _INDEX[k] = f
    if key not in _INDEX:
        raise KeyError(key)
    return _SHARDS[_INDEX[key]].get_tensor(key)


def pin(d: Path, name: str) -> torch.Tensor:
    return torch.from_numpy(np.load(d / (name + ".npy"))).to(torch.bfloat16).reshape(-1)


def pin_rows(d: Path, name: str) -> torch.Tensor:
    """A prefill pin as [rows, H] (§08.2: a chunk dump is [1, T, dim]; the wave's tile is [T, H]).

    `pin()` flattens, which is right for one token and wrong for a tile: §03.1 normalises over the
    LAST axis, so a flattened T*1024 vector would be read as one giant row. Anything the norm sees
    keeps this shape; the kernel's own buffer is the same bytes flattened, which is why
    `dump_bf16` needs no variant.
    """
    flat = torch.from_numpy(np.load(d / (name + ".npy"))).to(torch.bfloat16).reshape(-1)
    if flat.numel() % H:
        raise SystemExit("%s: %d elements is not a whole number of %d-element rows"
                         % (name, flat.numel(), H))
    return flat.reshape(-1, H)


def pin_rows(d: Path, name: str) -> torch.Tensor:
    """A prefill pin as [rows, H] (§08.2: a chunk dumps as [1, T, dim]; the wave's tile is [T, H]).

    `pin()` flattens, which is right for one token and wrong for a tile: §03.1 normalises over the
    LAST axis, so a flattened 10240-element vector would be one giant row. Anything that feeds the
    norm keeps this shape; anything that only feeds an elementwise add may flatten.
    """
    t = torch.from_numpy(np.load(d / (name + ".npy"))).to(torch.bfloat16)
    flat = t.reshape(-1)
    if flat.numel() % H:
        raise SystemExit("%s: %d elements is not a whole number of %d-element rows"
                         % (name, flat.numel(), H))
    return flat.reshape(-1, H)


def dump_bf16(t: torch.Tensor, path: Path) -> Path:
    """torch refuses bf16.numpy(); view the bits instead (prototype/README.md)."""
    t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tofile(path)
    return path


def read_bf16(path: Path, n: int = H) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16)[:n]).view(torch.bfloat16)


def is_attn_layer(d: Path, layer: int, step: str = "decode0") -> bool:
    """Block type read off the pins themselves (§10.5), not off a transcription of layer_types."""
    return (d / ("%s.L%02d.o_proj.out.npy" % (step, layer))).exists()


def is_attn_layer_pre(d: Path, layer: int) -> bool:
    """Same question for a prefill fixture, whose pins carry no `decode{N}.` prefix (§10.5)."""
    return (d / ("L%02d.o_proj.out.npy" % layer)).exists()


def block_proj(layer: int, d: Path) -> str:
    """The projection whose output is this layer's first residual addend."""
    return "o_proj" if is_attn_layer(d, layer) else "out_proj"


def first_proj(layer: int, d: Path) -> str:
    """The first projection the NEXT norm output feeds — the cross-stage check's other half."""
    return "q_proj" if is_attn_layer(d, layer) else "in_proj_qkv"


def proj_weight(layer: int, name: str) -> torch.Tensor:
    pre = CKPT_LAYER % layer
    if name.startswith(("in_proj", "out_proj")):
        return ckpt_tensor(pre + "linear_attn." + name + ".weight")
    return ckpt_tensor(pre + "self_attn." + name + ".weight")


def w_g1(layer: int) -> str:
    """Engine name of glue call 1's weight: this layer's post-attention norm (§03.2)."""
    return ENG_LAYER % layer + "post_attention_layernorm.weight"


def w_g2(layer: int) -> str:
    """Engine name of glue call 2's weight: the NEXT layer's input norm, or the final norm after
    layer 23 (§03.2's order means the add and the next norm are one fused step)."""
    return ENG_FINAL if layer == 23 else ENG_LAYER % (layer + 1) + "input_layernorm.weight"


def w_in(layer: int) -> str:
    """Engine name of the norm that precedes layer `layer`'s mixer (no add in front of it)."""
    return ENG_LAYER % layer + "input_layernorm.weight"


def ckpt_w(engine_name: str) -> torch.Tensor:
    """The same vector by its §01 name, for the torch-side references."""
    if engine_name == ENG_FINAL:
        return ckpt_tensor(CKPT_FINAL)
    assert engine_name.startswith("lm.layers."), engine_name
    layer, leaf = engine_name[len("lm.layers."):].split(".", 1)
    return ckpt_tensor(CKPT_LAYER % int(layer) + leaf)


def case(d: Path, step: str, layer: int) -> dict:
    """Everything one layer's two glue calls consume, and what each output must equal.

    s_in, add1, add2 come from pins; `s_mid` is the reference's own BF16 add of the first two
    (stage R proves `(s_in + add1) + add2` reproduces the pinned `s_out` bit-for-bit, which is
    what makes `s_mid` a legitimate input rather than a guess).
    """
    s_in = pin(d, "%s.L%02d.s_in" % (step, layer))
    add1 = pin(d, "%s.L%02d.%s.out" % (step, layer, block_proj(layer, d)))
    add2 = pin(d, "%s.L%02d.mlp.out" % (step, layer))
    s_mid = s_in + add1                      # §03.2: BF16 add, rounded before the norm sees it
    s_out = s_mid + add2
    return {
        "step": step, "layer": layer,
        "s": s_in, "add1": add1, "add2": add2, "s_mid": s_mid, "s_out": s_out,
        "w1": w_g1(layer), "w2": w_g2(layer),
        # call 1's norm output IS the pinned mlp input (§10.5: "post_attention_layernorm output")
        "ref_g1_n": pin(d, "%s.L%02d.mlp.in" % (step, layer)),
        "ref_g1_s": s_mid,
        "ref_g2_s": pin(d, "%s.L%02d.s_out" % (step, layer)),
        "ref_g2_n": rmsnorm_zc(s_out, ckpt_w(w_g2(layer))),
    }


def case_prefill(d: Path, layer: int) -> dict:
    """The same two calls over a WHOLE PREFILL TILE (§08.2: a chunk is one §03.x op over [T, H]).

    A prefill fixture (`data/refs/p64`, `data/refs/p300`) pins the bare `L{ll}.*` names with the
    leading length-1 axis: [1, T, dim]. Here that axis is dropped and T kept, because the wave
    hands the kernel a [T, H] tile and §03.1 must normalise each row on its own. Stage R runs the
    identical pairing checks on these tiles (`run.py --batched`), so the tile pairing is proved
    the same way the per-token pairing is — not assumed from the file names.
    """
    s_in = pin_rows(d, "L%02d.s_in" % layer)
    add1 = pin_rows(d, "L%02d.%s.out" % (layer, "o_proj" if is_attn_layer_pre(d, layer)
                                         else "out_proj"))
    add2 = pin_rows(d, "L%02d.mlp.out" % layer)
    s_mid = s_in + add1                      # §03.2: BF16 add, rounded before the norm sees it
    s_out = s_mid + add2
    return {
        "layer": layer, "rows": s_in.shape[0],
        "s": s_in, "add1": add1, "add2": add2, "s_mid": s_mid, "s_out": s_out,
        "w1": w_g1(layer), "w2": w_g2(layer),
        "ref_g1_n": pin_rows(d, "L%02d.mlp.in" % layer),
        "ref_g1_s": s_mid,
        "ref_g2_s": pin_rows(d, "L%02d.s_out" % layer),
        "ref_g2_n": rmsnorm_zc(s_out, ckpt_w(w_g2(layer))),
    }


def final_pair(d: Path, steps: list[str], layers: list[int]) -> str | None:
    """Which decode step the bare `final_hidden` / `pre_final_hidden` pins belong to.

    They are unprefixed, so every phase overwrote them (§10.5: pre_final_hidden + logits_last is
    not a valid pair and cos 0.25 is the proof). Rather than hardcode a step, find the one whose
    reconstructed layer-23 residual matches `pre_final_hidden` bit for bit, and insist it is
    unique. Returns the step name, or None if the fixture has no layer-23 case.
    """
    if 23 not in layers:
        return None
    try:
        pre = pin(d, "pre_final_hidden")
    except FileNotFoundError:
        return None
    hits = [st for st in steps if bool((case(d, st, 23)["s_out"] == pre).all())]
    if len(hits) > 1:                       # cannot happen with distinct steps, but say so loudly
        raise SystemExit("final_pair: %s all match pre_final_hidden; the fixture is degenerate"
                         % ",".join(hits))
    return hits[0] if hits else None
