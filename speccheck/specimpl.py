#!/usr/bin/env python3
"""Engine reference implementation of specification/02..07 (torch CPU).

Written directly from `specification/`, deliberately **not** by calling
transformers: if this code and `tools/reference_dump.py` agree, the spec is
correct and can be trusted as the contract for real kernels (gates G2-G7).

Sections quoted as "S0x.y" refer to specification/0x-*.md.

    .venv/bin/python -m speccheck.specimpl --prompt-ids-file data/refs/smoke/manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen3.5-0.8B"

# S00.2 constants (verified in specification/00).
D_MODEL = 1024
N_LAYERS = 24
FULL_ATTN_INTERVAL = 4
N_Q, N_KV, D_HEAD = 8, 2, 256
D_FF = 3584
VOCAB = 248320
EPS = 1e-6
H_K = H_V = 16
D_K = D_V = 128
K_CONV = 4
GDN_CHUNK = 64                              # S04.7 chunk size (reference default)
EOS_PRIMARY = 248046                        # <|im_end|> — the stop token (A-10)
EOS_LEGACY = 248044                         #  — accepted as stop too
EOS_IDS = (EOS_PRIMARY, EOS_LEGACY)
ONLINE_BLK = 64                             # S07.3 block width for online softmax

# Test-only bug injection, set by gates/run_negative.py to prove the gates
# discriminate (a gate that cannot fail verifies nothing). Empty in normal use.
NEGATIVE: set[str] = set()
ROPE_THETA = 1e7
PARTIAL_ROT = 0.25
ROT_DIM = int(D_HEAD * PARTIAL_ROT)          # 64
S_SCALE = D_HEAD ** -0.5                    # 1/16
Q_SCALE = D_K ** -0.5                       # S04.4 / A-05
KV_HEADS = N_KV * D_HEAD                    # 512
Q_WIDTH = N_Q * D_HEAD                      # 2048
QPROJ_WIDTH = Q_WIDTH * 2                   # 4096 (q|gate interleaved per head, A-15)
KEY_DIM = H_K * D_K                         # 2048
VAL_DIM = H_V * D_V                         # 2048
CONV_CH = 2 * KEY_DIM + VAL_DIM             # 6144
GDN_LAYERS = [i for i in range(N_LAYERS) if (i + 1) % FULL_ATTN_INTERVAL != 0]
ATTN_LAYERS = [i for i in range(N_LAYERS) if (i + 1) % FULL_ATTN_INTERVAL == 0]

SKIP_PREFIXES = ("model.visual.",)          # S01.8 / A-09
EXPECTED_TENSORS = 488                      # G0
EXPECTED_LOADED = 335                      # G0: 488 - 153 vision tensors


class Weights:
    """Loader + registry (S01, S02.5). `params` are cast to the model dtype
    exactly like HF `from_pretrained(dtype=...)`; A-02 A/B knob keeps the two
    F32 scalars (linear_attn.A_log, linear_attn.norm.weight) in F32."""

    def __init__(self, model_dir: Path, dtype=torch.bfloat16, keep_f32_scalars=False):
        self.dtype = dtype
        self.p: dict[str, torch.Tensor] = {}
        self.all_names: list[str] = []
        self.skipped: list[str] = []
        path = next(Path(model_dir).glob("*.safetensors"))
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                self.all_names.append(name)
                if name.startswith(SKIP_PREFIXES):
                    self.skipped.append(name)
                    continue
                t = f.get_tensor(name)
                f32_scalar = keep_f32_scalars and name.endswith(
                    ("linear_attn.A_log", "linear_attn.norm.weight"))
                self.p[name] = t.to(torch.float32 if f32_scalar else dtype)
        self._check()

    def _check(self):
        assert len(self.all_names) == EXPECTED_TENSORS, len(self.all_names)
        assert len(self.p) == EXPECTED_LOADED, len(self.p)
        assert len(self.skipped) == EXPECTED_TENSORS - EXPECTED_LOADED, len(self.skipped)
        want = {                                          # S01 tables
            "model.language_model.embed_tokens.weight": (VOCAB, D_MODEL),
            "model.language_model.norm.weight": (D_MODEL,),
            "mtp.fc.weight": (D_MODEL, 2 * D_MODEL),
            "mtp.layers.0.self_attn.q_proj.weight": (QPROJ_WIDTH, D_MODEL),
            "mtp.layers.0.self_attn.k_proj.weight": (KV_HEADS, D_MODEL),
            "mtp.layers.0.self_attn.o_proj.weight": (D_MODEL, Q_WIDTH),
        }
        for l in range(N_LAYERS):
            pre = f"model.language_model.layers.{l}."
            want[pre + "input_layernorm.weight"] = (D_MODEL,)
            want[pre + "post_attention_layernorm.weight"] = (D_MODEL,)
            want[pre + "mlp.gate_proj.weight"] = (D_FF, D_MODEL)
            want[pre + "mlp.up_proj.weight"] = (D_FF, D_MODEL)
            want[pre + "mlp.down_proj.weight"] = (D_MODEL, D_FF)
            if l in GDN_LAYERS:
                want.update({
                    pre + "linear_attn.in_proj_qkv.weight": (CONV_CH, D_MODEL),
                    pre + "linear_attn.in_proj_z.weight": (VAL_DIM, D_MODEL),
                    pre + "linear_attn.in_proj_b.weight": (H_V, D_MODEL),
                    pre + "linear_attn.in_proj_a.weight": (H_V, D_MODEL),
                    pre + "linear_attn.conv1d.weight": (CONV_CH, 1, K_CONV),
                    pre + "linear_attn.A_log": (H_V,),
                    pre + "linear_attn.dt_bias": (H_V,),
                    pre + "linear_attn.norm.weight": (D_V,),
                    pre + "linear_attn.out_proj.weight": (D_MODEL, VAL_DIM),
                })
            else:
                want.update({
                    pre + "self_attn.q_proj.weight": (QPROJ_WIDTH, D_MODEL),
                    pre + "self_attn.k_proj.weight": (KV_HEADS, D_MODEL),
                    pre + "self_attn.v_proj.weight": (KV_HEADS, D_MODEL),
                    pre + "self_attn.q_norm.weight": (D_HEAD,),
                    pre + "self_attn.k_norm.weight": (D_HEAD,),
                    pre + "self_attn.o_proj.weight": (D_MODEL, Q_WIDTH),
                })
        for name, shape in want.items():
            got = tuple(self.p[name].shape)
            assert got == shape, f"G1 shape mismatch {name}: {got} != {shape}"
        self.want = want


# ------------------------------- norms (S03.1, S04.5) -----------------------
def rmsnorm_zc(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """S03.1 zero-centered RMSNorm: f32 math, affine (1+w) in f32, cast back."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (out * (1.0 + w.float())).to(x.dtype)


def rmsnorm_gated(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """S04.5 gated RMSNorm, plain affine, cast order pinned by A-01."""
    of = o.float()
    n = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + EPS)).to(o.dtype)
    n = w * n
    return (n * torch.nn.functional.silu(z.float())).to(o.dtype)


# ------------------------------- rope (S05.3) --------------------------------
def rope_cs(positions: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of width ROT_DIM (=64): f32, freqs f_i = theta^(-2i/64)."""
    i = torch.arange(ROT_DIM // 2, device=device, dtype=torch.float32)
    inv = 1.0 / (ROPE_THETA ** (2 * i / ROT_DIM))
    ang = positions.to(torch.float32)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)


def rotate_half(a: torch.Tensor) -> torch.Tensor:
    h = a.shape[-1] // 2
    return torch.cat([-a[..., h:], a[..., :h]], -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """S05.3: rotate the first ROT_DIM dims of each head, pass the rest through.
    x: [T, heads, D_HEAD]; cos/sin: [T, ROT_DIM] — F32 from rope_cs, cast here.

    The table and BOTH products round to the activation dtype:
    `bf16(bf16(z*c) + bf16(rot*s))`. That is what the reference does (transformers' rotary embedding
    returns `emb.cos().to(dtype)`, so the multiply never sees an F32 table), and it is what the
    dumps carry: with an F32 table this function missed 52-97 of every 2048 `attn.q_rope` elements
    on dec8 (cos 0.99983, meanrel 0.0174 — inside the S10 attn threshold, so no gate failed, which
    is exactly why the wrong policy survived). Found by prototype/attn_decode reproducing the same
    pins bit-for-bit with the BF16 policy; see S05.3 and `speccheck/rope_table_probe.py`.
    """
    rot, rest = x[..., :ROT_DIM], x[..., ROT_DIM:]
    c, s = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)
    return torch.cat([(rot * c + rotate_half(rot) * s).to(x.dtype), rest], -1)


# ------------------------------- GDN block (S04) ----------------------------
def gdn_chunk_scan(qn, kn, v, g, beta, S=None, C=GDN_CHUNK):
    """S04.7 — the chunked form of the same recurrence, used for prefill.

    UT-transform formulation: intra-chunk attention with pairwise decay, one
    unit-triangular solve per chunk for the delta corrections, then a sequential
    scan over *chunks* only (not tokens). Sequences are right-padded to a
    multiple of `C`; padded positions get beta=0 and g=0, so they neither read
    nor write the state (A-08). Everything is F32, matching the reference.

    qn/kn are [T,H,Dk] (already l2-normalised, qn pre-scaled), v [T,H,Dv],
    g/beta [T,H] with g the log decay. Returns o [T,H,Dv] and S [H,Dk,Dv].
    """
    T, H, DK = kn.shape
    DV = v.shape[-1]
    pad = (-T) % C

    def padseq(x):
        if pad == 0:
            return x
        return torch.cat([x, x.new_zeros((pad,) + tuple(x.shape[1:]))], 0)

    q = padseq(qn).float().transpose(0, 1).reshape(H, -1, C, DK)
    k = padseq(kn).float().transpose(0, 1).reshape(H, -1, C, DK)
    vv = padseq(v).float().transpose(0, 1).reshape(H, -1, C, DV)
    de = padseq(g).float().transpose(0, 1).reshape(H, -1, C)
    be = padseq(beta).float().transpose(0, 1).reshape(H, -1, C)
    NC = q.shape[1]

    v_beta, k_beta = vv * be[..., None], k * be[..., None]
    cum = de.cumsum(-1)                                     # log decay since chunk start
    upper = torch.ones(C, C, dtype=torch.bool).triu(1)
    # pair[..., i, j] = decay accumulated from j to i, 0 for j > i (strict upper)
    pair = (cum[..., :, None] - cum[..., None, :]).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pair               # unit-triangular system
    intra = (q @ k.transpose(-1, -2)) * pair
    kcd = k_beta * cum.exp()[..., None]
    nv = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    kcd = torch.linalg.solve_triangular(ut, kcd, upper=False, unitriangular=True)

    q = q * cum.exp()[..., None]
    k = k * (cum[..., -1:] - cum).exp()[..., None]
    cdec = cum[..., -1].exp()[..., None, None]
    S = torch.zeros((H, DK, DV), dtype=torch.float32) if S is None else S.float()
    out = torch.zeros((H, NC, C, DV), dtype=torch.float32)
    for i in range(NC):
        v_new = nv[:, i] - kcd[:, i] @ S                     # correction only (delta rule)
        out[:, i] = q[:, i] @ S + intra[:, i] @ v_new        # inter-chunk read + intra
        S = S * cdec[:, i] + k[:, i].transpose(-1, -2) @ v_new
    return out.reshape(H, NC * C, DV)[:, :T].transpose(0, 1).contiguous(), S


def gdn(w: Weights, l: int, x: torch.Tensor, dbg: dict | None = None, win=None, S=None,
        form: str = "recurrent"):
    """S04 for a whole segment (T tokens). `form` picks S04.4 (recurrent, token
    loop — the decode path and the numerical contract) or S04.7 (chunked —
    prefill). Both must agree: gate `run_chunkcheck.py` asserts it against the
    reference dumps too.
    `win` is the S08.2 conv window: [k_conv-1, CONV_CH] of **pre-conv** raw."""
    p = f"model.language_model.layers.{l}.linear_attn."
    T = x.shape[0]
    raw = x @ w.p[p + "in_proj_qkv.weight"].t()                 # S04.1 [T,6144]
    z = x @ w.p[p + "in_proj_z.weight"].t()                     # [T,2048]
    b_raw = x @ w.p[p + "in_proj_b.weight"].t()                 # [T,16]
    a_raw = x @ w.p[p + "in_proj_a.weight"].t()                 # [T,16]

    # S04.2 causal depthwise conv (kernel 4), zero window before the request.
    C = w.p[p + "conv1d.weight"].reshape(CONV_CH, K_CONV)       # [6144,4]
    head = torch.zeros((K_CONV - 1, CONV_CH), dtype=raw.dtype) if win is None else win
    if "conv_write_first" in NEGATIVE and T == 1:
        # ring-buffer bug: the new value overwrites the oldest slot before the taps
        # read it (S07.2 says read the old window, then roll).
        head = torch.cat([head[1:], raw[:1]], 0)
    padded = torch.cat([head, raw], 0)
    new_win = padded[-(K_CONV - 1):].clone()                    # S07.2: roll, then read
    u = torch.zeros(raw.shape, dtype=torch.float32)
    for j in range(K_CONV):                                     # tap j -> token t-(3-j)
        u += padded[j:j + T].float() * C[:, j].float()
    # Reference behaviour: the depthwise conv accumulates in F32 and rounds once to
    # the activation dtype, and silu applies to that rounded value. Accumulating the
    # four taps in bf16 instead costs ~0.4 % per element in the q/k/v of *every* GDN
    # layer, which is the largest single share of the deep-layer drift.
    u = torch.nn.functional.silu(u.to(raw.dtype))

    q = u[:, :KEY_DIM].reshape(T, H_K, D_K)
    k = u[:, KEY_DIM:2 * KEY_DIM].reshape(T, H_K, D_K)
    v = u[:, 2 * KEY_DIM:].reshape(T, H_V, D_V)

    # S04.3 gates, f32.
    beta = torch.sigmoid(b_raw.float())
    g = -w.p[p + "A_log"].float().exp() * torch.nn.functional.softplus(
        a_raw.float() + w.p[p + "dt_bias"].float())
    gam = g.exp()                                               # [T,16]

    # S04.4 L2 norms + scales, then the recurrence.
    qn = q.float() * torch.rsqrt(q.float().pow(2).sum(-1, keepdim=True) + 1e-6) * Q_SCALE
    kn = k.float() * torch.rsqrt(k.float().pow(2).sum(-1, keepdim=True) + 1e-6)
    S = torch.zeros((H_V, D_K, D_V), dtype=torch.float32) if S is None else S
    if form == "chunk" and T > 1:
        o, S = gdn_chunk_scan(qn, kn, v, g, beta, S)
    else:
        o = torch.zeros((T, H_V, D_V), dtype=torch.float32)
        for t in range(T):
            gam_t, kt, qt, vt, bt = gam[t], kn[t], qn[t], v[t].float(), beta[t]
            S = S * gam_t[:, None, None]                        # decay
            r = torch.einsum("hkd,hk->hd", S, kt)               # retrieve
            dv = bt[:, None] * (vt - r)                         # delta
            S = S + torch.einsum("hk,hd->hkd", kt, dv)          # write
            o[t] = torch.einsum("hk,hkd->hd", qt, S)            # read

    if dbg is not None:
        dbg["mixed_qkv_pre_conv"] = raw
        dbg["q_post_conv"] = q
        dbg["k_post_conv"] = k
        dbg["v_post_conv"] = v
        dbg["g_decay_log"] = g
        dbg["beta"] = beta
        dbg["o_pre_norm"] = o
    gated = rmsnorm_gated(o.reshape(T * H_V, D_V), z.reshape(T * H_V, D_V),
                          w.p[p + "norm.weight"]).to(raw.dtype)
    if dbg is not None:
        dbg["gated_out"] = gated
        dbg["state_new"] = S                                    # S04.6 / S08.2
        dbg["conv_state"] = new_win
    return gated.reshape(T, VAL_DIM) @ w.p[p + "out_proj.weight"].t(), new_win, S


# ----------------------------- attention block (S05) -------------------------
# Test-only bug injection, used by `gates/run_negative.py` to prove the gates
# actually discriminate (a gate that cannot fail verifies nothing). Empty in normal
# use; each name reproduces a bug this project really had or nearly had.
NEGATIVE: set[str] = set()


def attention(w: Weights, l: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              kv: dict, dbg: dict | None = None, softmax: str = "full",
              prefix: str | None = None):
    # `prefix` lets the MTP block (S09) reuse this code with its own weights;
    # prefix=None is the ordinary target layer.
    p = prefix if prefix is not None else f"model.language_model.layers.{l}.self_attn."
    T = x.shape[0]
    qg = x @ w.p[p + "q_proj.weight"].t()                       # S05.1
    qg = qg.view(T, N_Q, D_HEAD * 2)
    q_raw, gate = qg[..., :D_HEAD], qg[..., D_HEAD:]
    if "deinterleave" in NEGATIVE:          # S05.1 says [q|gate]; this is the wrong split
        q_raw, gate = gate, q_raw
    k_raw = (x @ w.p[p + "k_proj.weight"].t()).view(T, N_KV, D_HEAD)
    v_raw = (x @ w.p[p + "v_proj.weight"].t()).view(T, N_KV, D_HEAD)

    qh = rmsnorm_zc(q_raw, w.p[p + "q_norm.weight"])            # S05.2
    kh = rmsnorm_zc(k_raw, w.p[p + "k_norm.weight"])
    qh = apply_rope(qh, cos, sin)                               # S05.3
    kh = apply_rope(kh, cos, sin)

    kv.setdefault("k", []).append(kh)                           # S05.7 post-rope
    kv.setdefault("v", []).append(v_raw)
    K = torch.cat(kv["k"], 0)                                   # [P+T, 2, 256]
    V = torch.cat(kv["v"], 0)
    P = K.shape[0]

    out = torch.zeros((T, N_Q, D_HEAD), dtype=x.dtype)
    run = softmax == "online" and T == 1                      # S07.3 online reduction
    for i in range(N_Q):
        j = i // (N_Q // N_KV)                                  # kappa(i)
        if run:
            # Online (block-wise) softmax over the prefix for THIS query row: the same
            # math as one torch.softmax, reduced in blocks with (max, sum)
            # rescaling - which is what a split-K / flash-decoding kernel does.
            # These accumulators are the reduction state of a single row; they are
            # NOT carried into the next decode step, because the next step has a
            # different query. An earlier version did carry them across steps and
            # parity died: attention cos 0.876 after one step, and the output matched
            # the *previous* prefix (0.9946) rather than the current one.
            srow = ((qh[0, i, :][None, :] @ K[:, j, :].t()) * S_SCALE).float()[0]
            m = torch.full((), float("-inf"), dtype=torch.float32)
            l = torch.zeros((), dtype=torch.float32)
            acc = torch.zeros(D_HEAD, dtype=torch.float32)
            for lo in range(0, P, ONLINE_BLK):
                blk = srow[lo:lo + ONLINE_BLK]
                mb = torch.maximum(m, blk.max())
                rescale = torch.exp(m - mb)
                e = torch.exp(blk - mb)
                l = l * rescale + e.sum()
                acc = acc * rescale + e @ V[lo:lo + ONLINE_BLK, j, :].float()
                m = mb
            out[0, i, :] = (acc / l).to(x.dtype)
            continue
        # Bit-faithful to the reference: the score matrix is produced and kept in
        # the activation dtype (bf16) and only the softmax runs in F32. Computing
        # scores in F32 skips a bf16 rounding of ~3e-3 per score and was a
        # measurable share of the deep-layer drift.
        s = (qh[:, i, :] @ K[:, j, :].t()) * S_SCALE          # [T, P], activation dtype
        qi = torch.arange(T)[:, None] + (P - T)                 # global query pos
        allowed = torch.arange(P)[None, :] <= qi                # causal, S05.5
        s = s.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(s.float(), dim=-1).to(x.dtype)
        out[:, i, :] = attn @ V[:, j, :].to(x.dtype)            # [T, 256]
    pre_gate = out.reshape(T, Q_WIDTH)
    if dbg is not None:
        dbg["q_rope"] = qh
        dbg["k_rope"] = kh
        dbg["v"] = V                                            # full cache incl. this step
        dbg["attn_out_pre_gate"] = pre_gate
    gated = pre_gate * torch.sigmoid(gate.reshape(T, Q_WIDTH))  # S05.6
    return gated @ w.p[p + "o_proj.weight"].t()


def mlp(w: Weights, l: int, x: torch.Tensor, prefix: str | None = None) -> torch.Tensor:
    p = prefix if prefix is not None else f"model.language_model.layers.{l}.mlp."  # S03.3
    gate = x @ w.p[p + "gate_proj.weight"].t()
    up = x @ w.p[p + "up_proj.weight"].t()
    # HF-ref applies silu and the gate product in the activation dtype (bf16).
    # Doing it in F32 (A-07 variant) costs ~0.3 % meanrel per layer and
    # accumulates, so the spec implementation stays bit-faithful here.
    return (torch.nn.functional.silu(gate) * up) @ w.p[p + "down_proj.weight"].t()


class Forward:
    """S03.5 chain with the S08 state; prefill then incremental decode."""

    def __init__(self, w: Weights, gdn_form: str = "recurrent", attn_softmax: str = "full"):
        self.w = w
        self.gdn_form = gdn_form            # S04.4 vs S04.7 for prefill
        self.attn_softmax = attn_softmax    # S07.3 full torch softmax vs online reduction
        self.generated: list[int] = []
        self.reset()

    def reset(self):
        self.P = 0                                              # S08.1 prefix_len
        self.kv = {l: {} for l in ATTN_LAYERS}                  # S08.1 KV
        self.conv_win = {l: None for l in GDN_LAYERS}           # S08.2 windows
        self.rec = {l: None for l in GDN_LAYERS}                # S08.1 recurrent S

    def _layer(self, l: int, x: torch.Tensor, cos, sin, dbg: dict | None, form="recurrent"):
        w = self.w
        pre = f"model.language_model.layers.{l}."
        residual = x
        h = rmsnorm_zc(x, w.p[pre + "input_layernorm.weight"])  # S03.2
        sub = {} if dbg is not None else None
        if l in GDN_LAYERS:
            h, self.conv_win[l], s_new = gdn(
                w, l, h, sub, self.conv_win[l], self.rec[l], form)
            if "no_state_persist" in NEGATIVE:      # the historical bug, kept testable
                s_new = None
            self.rec[l] = s_new
        else:
            h = attention(w, l, h, cos, sin, self.kv[l], sub, self.attn_softmax)
        if dbg is not None and sub:
            tag = "gdn." if l in GDN_LAYERS else "attn."
            for k, val in sub.items():
                dbg[f"L{l:02d}.{tag}{k}"] = val
            dbg[f"L{l:02d}.s_in"] = residual
        x = residual + h
        residual = x
        h = rmsnorm_zc(x, w.p[pre + "post_attention_layernorm.weight"])
        x = residual + mlp(w, l, h)
        if dbg is not None:
            dbg[f"L{l:02d}.s_out"] = x
        return x

    def _rope(self, T: int, device):
        pos = torch.arange(T, device=device) + self.P           # S02.4
        return rope_cs(pos, device)

    def prefill(self, ids: torch.Tensor, dbg: dict | None = None):
        x = self.w.p["model.language_model.embed_tokens.weight"][ids]   # S02.3
        cos, sin = self._rope(len(ids), x.device)
        for l in range(N_LAYERS):
            x = self._layer(l, x, cos, sin, dbg, self.gdn_form)
        self.P += len(ids)
        self.x_final = x
        return rmsnorm_zc(x, self.w.p["model.language_model.norm.weight"])  # S03.4

    def decode(self, tok: int, dbg: dict | None = None):
        x = self.w.p["model.language_model.embed_tokens.weight"][[tok]]     # S07.1
        cos, sin = self._rope(1, x.device)
        for l in range(N_LAYERS):
            x = self._layer(l, x, cos, sin, dbg)
        self.P += 1
        self.x_final = x
        return rmsnorm_zc(x, self.w.p["model.language_model.norm.weight"])

    def logits(self, normed: torch.Tensor) -> torch.Tensor:
        return normed @ self.w.p["model.language_model.embed_tokens.weight"].t()

    def snapshot(self) -> dict:
        """S08.5 rollback point. Cheap by design: the decode path never mutates a
        state tensor in place - `gdn` returns a new `S` and KV is appended to a list -
        so saving the dict/list references is enough to undo a speculative step. An
        engine that updates state in place needs a real double buffer instead."""
        return {"P": self.P,
                "rec": dict(self.rec),
                "conv_win": dict(self.conv_win),
                "kv": {l: {n: list(v.get(n, [])) for n in ("k", "v")}
                       for l, v in self.kv.items()}}

    def restore(self, snap: dict):
        self.P = snap["P"]
        self.rec = dict(snap["rec"])
        self.conv_win = dict(snap["conv_win"])
        self.kv = {l: {n: list(v.get(n, [])) for n in ("k", "v")}
                   for l, v in snap["kv"].items()}

    def greedy(self, ids: list[int], steps: int, dbg_prefill=None, dbg_decode=None,
               stop_on_eos: bool = True):
        """S07.5 greedy loop. `argmax` returns the lowest id on ties, as S07.1 pins.
        With stop_on_eos the loop ends after emitting an EOS id (A-10: 248046, with
        248044 accepted as legacy stop); the EOS token itself is part of the output."""
        out: list[int] = []
        last = self.logits(self.prefill(torch.tensor(ids), dbg_prefill))[-1]
        nxt = int(last.argmax())
        if dbg_prefill is not None:
            dbg_prefill["logits_last"] = last[None, :]
        out.append(nxt)
        for s in range(steps):
            if stop_on_eos and out[-1] in EOS_IDS:
                break
            d = {} if dbg_decode is not None else None
            last = self.logits(self.decode(out[-1], d))[-1]
            nxt = int(last.argmax())
            out.append(nxt)
            if dbg_decode is not None:
                dbg_decode[f"decode{s}"] = d | {"logits_last": last}
        self.generated = out
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--prompt-ids-file", default=str(ROOT / "data/refs/smoke/manifest.json"),
                    help="JSON with prefill_ids (e.g. a reference manifest)")
    ap.add_argument("--decode", type=int, default=2)
    ap.add_argument("--keep-f32-scalars", action="store_true", help="A-02 A/B knob")
    ap.add_argument("--gdn-prefill", default="recurrent", choices=["recurrent", "chunk"],
                    help="S04.4 token loop or S04.7 chunked scan for the prefill pass")
    ap.add_argument("--attn-softmax", default="full", choices=["full", "online"],
                    help="S07.3: one torch.softmax over the prefix (full), or the same "
                         "reduction block-wise with (max, sum) rescaling (online)")
    ap.add_argument("--no-stop-eos", action="store_true",
                    help="emit exactly --decode tokens instead of stopping at EOS (A-10)")
    ap.add_argument("--dump", default=None, help="write engine tensors here (npz, f32)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    w = Weights(Path(args.model), keep_f32_scalars=args.keep_f32_scalars)
    print(f"G0/G1 ok: {len(w.all_names)} tensors, {len(w.p)} loaded, {len(w.skipped)} skipped")
    ids = json.loads(Path(args.prompt_ids_file).read_text())["prefill_ids"]
    f = Forward(w, gdn_form=args.gdn_prefill, attn_softmax=args.attn_softmax)
    dbg_p, dbg_d = {}, {}
    gen = f.greedy(ids, args.decode, dbg_p, dbg_d, stop_on_eos=not args.no_stop_eos)
    print("greedy:", gen)
    if args.dump:
        blob = {k: np.asarray(v.detach().float().cpu()) for k, v in dbg_p.items()}
        for step, d in dbg_d.items():
            for k, v in d.items():
                blob[f"{step}.{k}"] = np.asarray(v.detach().float().cpu())
        blob["generated_ids"] = np.asarray(gen, dtype=np.int64)   # read by the G7 token gate
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump, **blob)
        print("wrote", args.dump, len(blob), "tensors")


if __name__ == "__main__":
    main()
