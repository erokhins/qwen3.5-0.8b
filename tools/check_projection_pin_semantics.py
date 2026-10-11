#!/usr/bin/env python3
"""Throwaway semantic check: does each new projection pin hold what its name says?

Each check rebuilds an ALREADY-PINNED downstream tensor from the NEW projection pin plus
checkpoint weights. If a pin meant the wrong thing (transposed, wrong slice, wrong head
interleaving), the rebuild would not reproduce the existing pin bit-for-bit.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/erokhins/erokhins_projects/qwen3.5-0.8b")
sys.path.insert(0, str(ROOT))
from speccheck.specimpl import (Weights, rmsnorm_zc, rmsnorm_gated, rope_cs,  # noqa: E402
                                apply_rope, N_Q, N_KV, D_HEAD, Q_WIDTH)

D = Path(sys.argv[1])
L_GDN, L_ATTN, STEP = 0, 3, "decode0"
POS = len(json.loads((D / "manifest.json").read_text())["prefill_ids"])  # rope pos of decode0
w = Weights(ROOT / "models/Qwen3.5-0.8B")
PRE = "model.language_model."
fails = []


def pin(name):
    return torch.from_numpy(np.load(D / f"{name}.npy"))


def bits(name):
    """pin (stored F32, exactly representing bf16 values) -> bf16 tensor"""
    return pin(name).to(torch.bfloat16)


def head_pin(name, heads):
    """[1, heads, 1, D_HEAD] (reference layout [B,H,T,D]) -> [1, heads, D_HEAD]"""
    return bits(name).permute(0, 2, 1, 3).reshape(1, heads, D_HEAD)


def report(label, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"   [{extra}]" if extra else ""))
    if not ok:
        fails.append(label)


# A: in_proj_qkv.out is the operand the depthwise conv receives, channels-first ---------
qkv = bits(f"{STEP}.L{L_GDN:02d}.in_proj_qkv.out").squeeze(0).squeeze(0)          # [6144]
conv_in = pin(f"{STEP}.L{L_GDN:02d}.gdn.mixed_qkv_pre_conv").squeeze(0).squeeze(1)
report("A in_proj_qkv.out == gdn.mixed_qkv_pre_conv (row vs channels-first)",
       torch.equal(qkv.double(), conv_in.double()), f"{tuple(qkv.shape)}")

# B: in_proj_b / in_proj_a are the pre-activations of beta and g -----------------------
b_raw = bits(f"{STEP}.L{L_GDN:02d}.in_proj_b.out").squeeze(0).squeeze(0)          # [16]
a_raw = bits(f"{STEP}.L{L_GDN:02d}.in_proj_a.out").squeeze(0).squeeze(0)
beta_ok = torch.equal(b_raw.sigmoid().reshape(-1), bits(f"{STEP}.L{L_GDN:02d}.gdn.beta").reshape(-1))
A_log = w.p[PRE + f"layers.{L_GDN}.linear_attn.A_log"].to(torch.bfloat16)
dt_bias = w.p[PRE + f"layers.{L_GDN}.linear_attn.dt_bias"]
g = -A_log.float().exp() * torch.nn.functional.softplus(a_raw.float() + dt_bias.float())
g_pin = pin(f"{STEP}.L{L_GDN:02d}.gdn.g_decay_log")
g_ok = torch.equal(g.reshape(-1), g_pin.reshape(-1))
report("B sigmoid(in_proj_b)=gdn.beta and g(in_proj_a)=gdn.g_decay_log",
       beta_ok and g_ok,
       f"beta bit-equal={beta_ok} maxd={float((b_raw.sigmoid().float() - pin(f'{STEP}.L{L_GDN:02d}.gdn.beta')).abs().max()):.3g}; "
       f"g bit-equal={g_ok} maxd={float((g - g_pin).abs().max()):.3g} maxrel="
       f"{float(((g - g_pin).abs() / (g_pin.abs() + 1e-30)).max()):.3g}")

# C: in_proj_z is the gate argument of the gated RMSNorm -------------------------------
z = bits(f"{STEP}.L{L_GDN:02d}.in_proj_z.out").squeeze(0).reshape(-1, 128)        # [16,128]
o = bits(f"{STEP}.L{L_GDN:02d}.gdn.o_pre_norm").reshape(-1, 128)
rebuilt = rmsnorm_gated(o, z, w.p[PRE + f"layers.{L_GDN}.linear_attn.norm.weight"])
ref = bits(f"{STEP}.L{L_GDN:02d}.gdn.gated_out")
bit_eq = torch.equal(rebuilt, ref)
cos = float(torch.nn.functional.cosine_similarity(rebuilt.double().ravel(),
                                                  ref.double().ravel(), dim=0))
report("C rmsnorm_gated(gdn.o_pre_norm, in_proj_z) == gdn.gated_out",
       bit_eq or cos > 0.99999, f"bit-equal={bit_eq} cos={cos:.8f}")

# D: q_proj's interleaved q half is what became attn.q_rope ----------------------------
cos_, sin_ = rope_cs(torch.arange(POS + 1), torch.device("cpu"))
cos_, sin_ = cos_[POS:POS + 1], sin_[POS:POS + 1]
qg = bits(f"{STEP}.L{L_ATTN:02d}.q_proj.out").squeeze(0).view(1, N_Q, D_HEAD * 2)
q_rope = head_pin(f"{STEP}.L{L_ATTN:02d}.attn.q_rope", N_Q)
qw = w.p[PRE + f"layers.{L_ATTN}.self_attn.q_norm.weight"]
interleaved = apply_rope(rmsnorm_zc(qg[..., :D_HEAD], qw), cos_, sin_)
naive_half = apply_rope(rmsnorm_zc(qg.reshape(-1)[:Q_WIDTH].view(1, N_Q, D_HEAD), qw), cos_, sin_)


def worst_row_cos(a, b):
    return float(torch.min(torch.nn.functional.cosine_similarity(
        a.double(), b.double(), dim=-1)))


good, bad = worst_row_cos(interleaved, q_rope), worst_row_cos(naive_half, q_rope)
report("D q_proj = [8 heads][q 256 | gate 256], q half -> attn.q_rope",
       good > 0.9999 and bad < 0.9,
       f"worst row cos: interleaved={good:.8f}, contiguous-half split={bad:.6f}")

# E: o_proj.out is o_proj(attn_out * sigmoid(gate)), gate = interleaved second half ----
gate = qg[..., D_HEAD:].reshape(Q_WIDTH)
pre = bits(f"{STEP}.L{L_ATTN:02d}.attn.attn_out_pre_gate").reshape(Q_WIDTH)
ow = w.p[PRE + f"layers.{L_ATTN}.self_attn.o_proj.weight"]
built = torch.nn.functional.linear(pre * torch.sigmoid(gate), ow)
built_naive = torch.nn.functional.linear(pre * torch.sigmoid(qg.reshape(-1)[Q_WIDTH:]), ow)
ref = bits(f"{STEP}.L{L_ATTN:02d}.o_proj.out").reshape(-1)
report("E o_proj(attn_out_pre_gate * sigmoid(interleaved gate)) == o_proj.out",
       torch.equal(built, ref) and not torch.equal(built_naive, ref),
       f"contiguous-half gate would differ in {int((built_naive != ref).sum())}/{ref.numel()} elems")
report("E2 the L{ii}.attn.out pin the dumper's attn_io_hook intends does NOT exist "
       "(self_attn is called with keyword args, so its positional `inp` is empty)",
       not (D / f"{STEP}.L{L_ATTN:02d}.attn.out.npy").exists(),
       "o_proj.out is therefore the only pin of that tensor")

# F: k_proj is pre-norm/pre-rope, v_proj is pre-cache-write ----------------------------
k_raw = bits(f"{STEP}.L{L_ATTN:02d}.k_proj.out").squeeze(0).view(1, N_KV, D_HEAD)
rebuilt_k = apply_rope(rmsnorm_zc(k_raw, w.p[PRE + f"layers.{L_ATTN}.self_attn.k_norm.weight"]),
                       cos_, sin_)
k_ref = head_pin(f"{STEP}.L{L_ATTN:02d}.attn.k_rope", N_KV)
k_cos = float(torch.nn.functional.cosine_similarity(rebuilt_k.double().ravel(), k_ref.double().ravel(), dim=0))
k_maxrel = float(((rebuilt_k.float() - k_ref.float()).abs().max() / (k_ref.float().abs().max() + 1e-30)))
k_ok = k_cos > 0.9999
v = bits(f"{STEP}.L{L_ATTN:02d}.v_proj.out").squeeze(0).reshape(N_KV, D_HEAD)
v_ok = torch.equal(v, bits(f"{STEP}.L{L_ATTN:02d}.cache_values")[0, :, -1, :])
report("F k_proj -> attn.k_rope (norm+rope), v_proj -> last row of cache_values",
       k_ok and v_ok, f"k cos={k_cos:.8f} maxrel={k_maxrel:.2e} (bit-equal={torch.equal(rebuilt_k, k_ref)}), v bit-equal={v_ok}")

# G: both block projections are the residual addend that feeds post_attention_layernorm
for l, which in ((L_GDN, "out_proj"), (L_ATTN, "o_proj")):
    s_in = bits(f"{STEP}.L{l:02d}.s_in").squeeze(0)
    proj = bits(f"{STEP}.L{l:02d}.{which}.out").squeeze(0)
    rebuilt = rmsnorm_zc(s_in + proj, w.p[PRE + f"layers.{l}.post_attention_layernorm.weight"])
    report(f"G rmsnorm_zc(s_in + {which}.out) == mlp.in (layer {l})",
           torch.equal(rebuilt, bits(f"{STEP}.L{l:02d}.mlp.in").squeeze(0)))

# H: the prefill copy of the same hook exists, unprefixed, with T rows -----------------
prefill = bits("L00.in_proj_qkv.out")
report("H prefill pin L00.in_proj_qkv.out exists with T rows",
       tuple(prefill.shape) == (1, POS, 6144), f"shape {tuple(prefill.shape)}")

print("\nALL SEMANTIC CHECKS PASS" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
sys.exit(0 if not fails else 1)
