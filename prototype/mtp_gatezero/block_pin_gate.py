#!/usr/bin/env python3
"""Gate the block code path the MTP drafter reuses, against the pinned HF dumps.

P7 gate zero, part 5. There is no HF reference for the MTP block, so a golden derived from
`speccheck/mtp.py` (see `draft_golden.py`) is only as good as the claim that the code it runs is the
code the repo already gates. §09.2 builds the draft block out of the same two functions the target
uses - `speccheck.specimpl.attention(...)` and `mlp(...)` - reached through their `prefix=` argument,
which exists precisely so §09 can reuse them:

    # speccheck/specimpl.py, attention():
    #   `prefix` lets the MTP block (S09) reuse this code with its own weights;
    #   prefix=None is the ordinary target layer.

So the checkable claim is: **the prefix path is not a second implementation.** Two things follow, and
they are separate checks because they adjudicate different things:

 1. PRIMARY (gating, bit-identical). Run `attention`/`mlp` twice on the same pinned input with the same
    weights - once through `prefix=`, once through the ordinary `prefix=None` branch - and require
    bit-identity. This is the one that would catch a real risk: the prefix argument quietly becoming a
    second code path. A comparison of a function against itself sounds vacuous until you notice it is
    the only form that can witness "these two branches are the same code" without also re-adjudicating
    every other numerics band in the tree.
 2. REPORTED (not gating). The same path against the HF pins. Its residual is the speccheck-vs-HF
    cascade G3i already owns - §10.2 measured the worst isolated cos 0.99997 / meanrel 0.53 %, and
    `o_proj.out` at an 18-lane prefix measures 1.08 % here, i.e. slightly over that recorded band. That
    is reported with the band next to it, not quietly re-tuned: a new gate must not move an existing
    band, and this residual is about speccheck's F32 softmax order, not about MTP.

Cases are (fixture, decode step, attention layer), built only from fixtures whose pins exist. The
position is DERIVED (`len(prefill_ids) + step`), never typed - if it were wrong, the rope comparison
fails, and that is part of the point.

Bands: check 1 is exact. Check 2 prints cos/meanrel against the G3i band (`cos >= 0.9999`,
`meanrel <= 1 %`) and labels anything over it `[OVER-BAND, reported not gated]`.

    .venv/bin/python prototype/mtp_gatezero/block_pin_gate.py [--fixtures dec8,prefix24dec8]
                                                             [--steps 0,1] [--layers 7,23]

CPU only. Reads pins and the checkpoint; no device, no timing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speccheck.specimpl import (Weights, attention, mlp, rmsnorm_zc, rope_cs)   # noqa: E402

P = pathlib.Path(__file__).resolve().parent


def pin(d: pathlib.Path, man: dict, name: str) -> torch.Tensor:
    entry = man["pins"].get(name)
    if entry is None:
        raise KeyError(name)
    a = np.load(d / f"{name}.npy").reshape(-1)
    if "bits:uint16" in str(entry["dtype"]) or a.dtype == np.uint16:
        t = torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16)
    else:
        t = torch.from_numpy(a.astype(np.float32))
    shape = list(entry["shape"])
    n = 1
    for s in shape:
        n *= s
    if a.size != n:
        raise SystemExit(f"{d.name}:{name}: {a.size} elements, manifest says {shape}")
    return t.reshape(*shape) if a.size != 1 else t.reshape(shape)


def report(label: str, got: torch.Tensor, want: torch.Tensor, bit_exact_required: bool,
           fails: list[str], gating: bool = True) -> None:
    g = got.float().reshape(-1)
    wv = want.float().reshape(-1)
    if g.numel() != wv.numel():
        fails.append(f"{label}: shape {tuple(got.shape)} vs pin {tuple(want.shape)}")
        print(f"    {label:<26} SHAPE {tuple(got.shape)} != {tuple(want.shape)}")
        return
    exact = bool(torch.equal(got.to(torch.bfloat16).reshape(-1), wv.to(torch.bfloat16)))
    cos = float(torch.nn.functional.cosine_similarity(g, wv, dim=0).item())
    meanrel = float((torch.abs(g - wv).mean() / (wv.abs().mean() + 1e-30)).item())
    ok = exact if bit_exact_required else (cos >= 0.9999 and meanrel <= 0.01)
    tag = "" if gating else "   [OVER-BAND, reported not gated: speccheck-vs-HF cascade, S10.2]"
    print(f"    {label:<26} bit-exact={str(exact):<5} cos={cos:.9f} meanrel={meanrel:.3e} "
          f"{'OK' if ok else 'FAIL'}{tag}")
    if not ok and gating:
        fails.append(f"{label}: bit-exact={exact} cos={cos:.6f} meanrel={meanrel:.3e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="dec8,prefix24dec8")
    ap.add_argument("--steps", default="0,1")
    ap.add_argument("--layers", default="7,23")
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    w = Weights(pathlib.Path(args.model))
    fails: list[str] = []
    cases = 0

    for tag in [t for t in args.fixtures.split(",") if t]:
        d = ROOT / "data" / "refs" / tag
        man = json.loads((d / "manifest.json").read_text())
        n_prefill = len(man["prefill_ids"])
        for s in [int(x) for x in args.steps.split(",") if x != ""]:
            for ll in [int(x) for x in args.layers.split(",") if x != ""]:
                need = [f"decode{s}.L{ll:02d}.s_in", f"decode{s}.L{ll:02d}.cache_keys",
                        f"decode{s}.L{ll:02d}.cache_values", f"decode{s}.L{ll:02d}.attn.v",
                        f"decode{s}.L{ll:02d}.o_proj.out", f"decode{s}.L{ll:02d}.mlp.in",
                        f"decode{s}.L{ll:02d}.mlp.out", f"decode{s}.L{ll:02d}.s_out",
                        f"decode{s}.L{ll:02d}.q_proj.out"]
                missing = [n for n in need if n not in man["pins"]]
                if missing:
                    print(f"{tag} decode{s} L{ll:02d}: skipped, no pin for {missing[:2]}")
                    continue
                cases += 1
                pos = n_prefill + s        # derived, not typed: a wrong position fails the rope pins
                pre = f"model.language_model.layers.{ll}.self_attn."
                pre_mlp = f"model.language_model.layers.{ll}.mlp."
                x = pin(d, man, f"decode{s}.L{ll:02d}.s_in").to(torch.bfloat16).reshape(1, -1)
                ck = pin(d, man, f"decode{s}.L{ll:02d}.cache_keys").to(torch.bfloat16)
                cv = pin(d, man, f"decode{s}.L{ll:02d}.cache_values").to(torch.bfloat16)
                # the cache pin INCLUDES this step's lane; hand the function the prior lanes and let
                # it append, which is what the live decode path does.
                prior_k = ck[0, :, :-1, :].permute(1, 0, 2).contiguous()
                prior_v = cv[0, :, :-1, :].permute(1, 0, 2).contiguous()
                kv = {"k": [prior_k], "v": [prior_v]}
                cos, sin = rope_cs(torch.tensor([pos], device=x.device), x.device)
                print(f"\n{tag} decode{s} L{ll:02d} (position {pos}, prefix reuse path)")
                dbg: dict = {}
                x_norm = rmsnorm_zc(x, w.p[f"model.language_model.layers.{ll}."
                                            "input_layernorm.weight"])
                attn_out = attention(w, ll, x_norm, cos, sin, kv, dbg, "full", prefix=pre)
                # ---- check 1: the prefix branch against the ordinary branch, bit-identical -----
                dbg_o: dict = {}
                kv_o = {"k": [prior_k.clone()], "v": [prior_v.clone()]}
                attn_o = attention(w, ll, x_norm, cos, sin, kv_o, dbg_o, "full", prefix=None)
                same = all(bool(torch.equal(dbg_o[k], dbg[k])) for k in
                           ("q_rope", "k_rope", "v", "attn_out_pre_gate")) \
                    and bool(torch.equal(attn_o, attn_out))
                m_same = bool(torch.equal(mlp(w, ll, x_norm), mlp(w, ll, x_norm, prefix=pre_mlp)))
                print(f"    PREFIX == ORDINARY        bit-identical: {same and m_same} "
                      f"(attention {same}, mlp {m_same})")
                if not (same and m_same):
                    fails.append("prefix branch is not the ordinary branch")
                # ---- check 2: the same tensors against the HF pins (reported) ------------------
                # q_proj's operand is the NORMED layer input, not s_in - comparing against the raw
                # input gives cos 0.999 with meanrel 0.8, which looks like a mild cascade difference and
                # is really the wrong operand (the missing RMS scale).
                report("q_proj.out (prefix path)", (x_norm @ w.p[pre + "q_proj.weight"].t()),
                       pin(d, man, f"decode{s}.L{ll:02d}.q_proj.out"), True, fails)
                report("attn.q_rope", dbg["q_rope"], pin(d, man, f"decode{s}.L{ll:02d}.attn.q_rope"),
                       False, fails, gating=False)
                report("attn.k_rope", dbg["k_rope"], pin(d, man, f"decode{s}.L{ll:02d}.attn.k_rope"),
                       False, fails, gating=False)
                # the pin is [1, heads, lanes, dim]; the function keeps [lanes, heads, dim], so the
                # comparison has to be told which layout it is looking at (a bare reshape of the pin
                # reads a different tensor: cos 0.38 from a layout bug, not from the kernel).
                v_pin = pin(d, man, f"decode{s}.L{ll:02d}.attn.v")[0].permute(1, 0, 2)
                report("attn.v (cache incl. step)", dbg["v"], v_pin, True, fails)
                report("attn.attn_out_pre_gate", dbg["attn_out_pre_gate"],
                       pin(d, man, f"decode{s}.L{ll:02d}.attn.attn_out_pre_gate"),
                       False, fails, gating=False) \
                    if f"decode{s}.L{ll:02d}.attn.attn_out_pre_gate" in man["pins"] else None
                report("o_proj.out", attn_out, pin(d, man, f"decode{s}.L{ll:02d}.o_proj.out"),
                       False, fails, gating=False)
                resid = (x + attn_out).to(torch.bfloat16)
                m_in = rmsnorm_zc(resid, w.p[f"model.language_model.layers.{ll}."
                                             "post_attention_layernorm.weight"])
                report("mlp.in", m_in, pin(d, man, f"decode{s}.L{ll:02d}.mlp.in"),
                       False, fails, gating=False)
                m_out = mlp(w, ll, m_in, prefix=pre_mlp)
                report("mlp.out", m_out, pin(d, man, f"decode{s}.L{ll:02d}.mlp.out"),
                       False, fails, gating=False)
                report("s_out", (resid + m_out).to(torch.bfloat16),
                       pin(d, man, f"decode{s}.L{ll:02d}.s_out"), False, fails, gating=False)

    if cases == 0:
        raise SystemExit("no case ran: every requested pin is absent. A gate that adjudicates nothing "
                         "cannot pass.")
    print(f"\n{cases} block case(s) through the prefix path.")
    if fails:
        for f in fails:
            print("FAIL:", f)
        return 1
    print("PASS: the prefix-parameterised block path is bit-identical to the ordinary path speccheck "
          "gates, so a golden built through it (draft_golden.py) is built on gated code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
