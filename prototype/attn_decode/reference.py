#!/usr/bin/env python3
"""Rebuild the §05 attention-decode operands from the pins, and prove the rebuild is right.

The W1 pins give this op everything that comes in — `q_proj.out`, `k_proj.out`, `v_proj.out`,
`cache_keys`, `cache_values` — and everything that goes out — `attn.q_rope`, `attn.k_rope`,
`attn.v`, `attn.attn_out_pre_gate`. What they do NOT give is the cache in the shape the engine
holds it at decode time: the pins dump the cache AFTER the append (§10.5), while the kernel is
handed the cache with one lane still free and writes that lane itself (§05.7). So `decode_case()`
takes the pinned post-append cache, cuts the appended lane back off, and hands over a buffer whose
capacity is a round number of lanes above the valid length — which is also what makes the
`readpast`/`slotk` negatives observable, since with cap == lvalid there would be no slack lane for
a wrong slot or a runaway causal mask to land on. Slack lanes hold `POISON`, never zeros: a
buffer overrun that reads zeros looks merely small, one that reads 1.0 is unmistakable.

`verify()` is the foundation the rest of the gate stands on, and it is the same shape as
`proj_gemv/operands.verify()`: the torch chain in this file must reproduce the pinned stages
BIT-EXACTLY, and each plausible misreading must fail to. Two of those misreadings are live
problems rather than straw men:

  * rope with F32 cos/sin (`speccheck.specimpl.apply_rope`'s policy) misses 52-97 of every 2048
    `attn.q_rope` elements; the pins say cos/sin are rounded to the activation dtype first.
  * the §05.1 contiguous-half gate reading.

Nothing here is trusted because the name matches: `decode_case()` also requires the pinned cache's
last lane to equal the pinned `attn.k_rope`, which is what proves `pos = lvalid - 1` and the
fixture pairing (§10.5's last-writer-wins trap) rather than assuming it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "proj_gemv")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from speccheck.specimpl import Weights, rmsnorm_zc, rope_cs  # noqa: E402
from operands import CKPT, ckpt_tensor, dump_bf16, pin, read_bf16  # noqa: E402

SCALE = 0.0625                      # 1/sqrt(256), §00.2
N_Q, N_KV, DH, ROT = 8, 2, 256, 64   # §01.3
HR = ROT // 2                       # rope pairs are (i, i+32) inside the first 64 dims
ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
POISON = 1.0                        # slack cache lanes, see the module docstring
CAP_ROUND = 8                       # capacity is a round number of lanes, like a real allocation

_W: Weights | None = None


def weights() -> Weights:
    global _W
    if _W is None:
        _W = Weights(CKPT)          # re-checks every shape (G1) on load
    return _W


def attn_norms(layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    pre = "model.language_model.layers.%d.self_attn." % layer
    return (weights().p[pre + "q_norm.weight"], weights().p[pre + "k_norm.weight"])


def o_proj_weight(layer: int) -> torch.Tensor:
    return weights().p["model.language_model.layers.%d.self_attn.o_proj.weight" % layer]


# --------------------------------------------------------------------- the torch chain
def rot_half(a: torch.Tensor) -> torch.Tensor:
    """§05.3's rotate over the ROT_DIM prefix only: pairs (i, i+32)."""
    return torch.cat([-a[..., HR:ROT], a[..., :HR]], -1)


def rope_subset(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cast_bf16=True,
                round_parts=True) -> torch.Tensor:
    """Partial rope on the first 64 dims of each head, rest passed through (§05.3).

    `cast_bf16=True, round_parts=True` is what the pins say: cos/sin are rounded to the activation
    dtype and each product is rounded before the add. `cast_bf16=False` is speccheck's
    `apply_rope`, which promotes to F32 and rounds once at the end.
    """
    c, s = (cos.to(torch.bfloat16), sin.to(torch.bfloat16)) if cast_bf16 else (cos, sin)
    head = x[..., :ROT]
    if c.dtype == torch.bfloat16:
        part = head * c + rot_half(head) * s if not round_parts else \
            (head * c).to(torch.bfloat16) + (rot_half(head) * s).to(torch.bfloat16)
    else:
        part = (head.float() * c + rot_half(head).float() * s).to(x.dtype)
    return torch.cat([part, x[..., ROT:]], -1)


def head_stages(qg_row, k_row, v_row, layer, pos, rope_policy="pins"):
    """§05.1 de-interleave + §05.2 norms + §05.3 rope for one query row.

    Returns (q_rope [8,256], k_rope [2,256], v [2,256], gate [8,256]).
    """
    qn, kn = attn_norms(layer)
    qg = qg_row.reshape(N_Q, 2 * DH)
    q_raw, gate = qg[:, :DH], qg[:, DH:]
    fn = {"pins": dict(), "f32_cs": dict(cast_bf16=False), "none": None}[rope_policy]
    cos, sin = rope_cs(torch.tensor([pos]), None)
    if fn is None:
        return (rmsnorm_zc(q_raw, qn), rmsnorm_zc(k_row.reshape(N_KV, DH), kn),
                v_row.reshape(N_KV, DH), gate)
    return (rope_subset(rmsnorm_zc(q_raw, qn), cos, sin, **fn),
            rope_subset(rmsnorm_zc(k_row.reshape(N_KV, DH), kn), cos, sin, **fn),
            v_row.reshape(N_KV, DH), gate)


def attend(q_rope, k_rope, v, kcache, vcache, pos, softmax="full", score_round=True,
           blk=8) -> torch.Tensor:
    """§05.5 GQA attention over lanes 0..pos, §05.7's appended lane included from `k_rope`/`v`.

    `kcache`/`vcache` are the PRE-append caches ([2, cap, 256]); the appended lane is this step's
    own `k_rope`/`v`, exactly as the kernel reads it. `softmax="full"` is torch's one-pass form,
    `softmax="online"` is the §07.3 (max,sum) block reduction the engine uses — two different
    orders over the same numbers, which is what floor_probe.py sizes.
    """
    n = pos + 1
    K = torch.cat([kcache[:, :pos], k_rope[:, None, :]], 1)     # [2, pos+1, 256]
    V = torch.cat([vcache[:, :pos], v[:, None, :]], 1)
    assert K.shape[1] == n, "the pre-append cache must hold exactly `pos` lanes"
    out = torch.zeros(N_Q, DH, dtype=torch.bfloat16)
    for i in range(N_Q):
        j = i // (N_Q // N_KV)                                  # kappa(i) = i/4, §05.5
        s = (q_rope[i:i + 1] @ K[j, :n].t())
        s = (s.to(torch.bfloat16) if score_round else s.float()) * SCALE
        if softmax == "full":
            w = torch.softmax(s.float(), -1).to(torch.bfloat16)
            out[i:i + 1] = w @ V[j, :n]
            continue
        m = torch.zeros(1, 1)
        l = torch.zeros(1, 1)
        acc = torch.zeros(1, DH, dtype=torch.float32)
        for lo in range(0, n, blk):
            sb = s[0, lo:lo + blk].float()
            mb = torch.maximum(m, sb.max(-1, keepdim=True).values)
            rescale = torch.exp(m - mb)
            e = torch.exp(sb - mb)
            l = l * rescale + e.sum(-1, keepdim=True)
            acc = acc * rescale + e @ V[j, lo:lo + blk].float()   # engine form: weights stay F32
            m = mb
        out[i:i + 1] = (acc / l).to(torch.bfloat16)
    return out


def gate_apply(a: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """§05.6: plain sigmoid on the BF16 gate operand."""
    return a.reshape(-1) * torch.sigmoid(gate.reshape(-1))


# --------------------------------------------------------------------- cases
def _cap_for(n_lanes: int) -> int:
    return max(CAP_ROUND, -(-n_lanes // CAP_ROUND) * CAP_ROUND)


def _slack_cache(cap: int) -> torch.Tensor:
    return torch.full((N_KV, cap, DH), POISON, dtype=torch.bfloat16)


def decode_case(d: Path, step: str, layer: int) -> dict:
    """One pinned decode step, as the kernel takes it. `pos = lvalid - 1` is PROVED in verify()."""
    kc = pin(d, "%s.L%02d.cache_keys" % (step, layer)).reshape(N_KV, -1, DH)   # post-append
    vc = pin(d, "%s.L%02d.cache_values" % (step, layer)).reshape(N_KV, -1, DH)
    lvalid = kc.shape[1]
    cap = _cap_for(lvalid)
    k_slack, v_slack = _slack_cache(cap), _slack_cache(cap)
    k_slack[:, :lvalid - 1] = kc[:, :lvalid - 1]
    v_slack[:, :lvalid - 1] = vc[:, :lvalid - 1]
    qn, kn = attn_norms(layer)
    return {
        "name": "%s.L%02d" % (step, layer), "kind": "decode", "pos": lvalid - 1, "lvalid": lvalid,
        "cap": cap,
        "qg": pin(d, "%s.L%02d.q_proj.out" % (step, layer)),
        "kg": pin(d, "%s.L%02d.k_proj.out" % (step, layer)),
        "vg": pin(d, "%s.L%02d.v_proj.out" % (step, layer)),
        "kcache": k_slack.reshape(-1), "vcache": v_slack.reshape(-1),
        "qn": qn, "kn": kn,
        "ref": {
            "q_rope": pin(d, "%s.L%02d.attn.q_rope" % (step, layer)).reshape(N_Q, DH),
            "k_rope": pin(d, "%s.L%02d.attn.k_rope" % (step, layer)).reshape(N_KV, DH),
            "v": pin(d, "%s.L%02d.attn.v" % (step, layer)).reshape(N_KV, -1, DH),
            "cache_keys": kc, "cache_values": vc,
            "attn_pre_gate": pin(d, "%s.L%02d.attn.attn_out_pre_gate" % (step, layer))
            .reshape(N_Q, DH),
            "gated": pin(d, "%s.L%02d.attn.attn_out_pre_gate" % (step, layer))
            .reshape(N_Q, DH).reshape(-1)
            * torch.sigmoid(pin(d, "%s.L%02d.q_proj.out" % (step, layer))
                            .reshape(N_Q, 2 * DH)[:, DH:].reshape(-1)),
            "o_proj": pin(d, "%s.L%02d.o_proj.out" % (step, layer)),
        },
        "stage_ref": {  # reference stages for the same input, independent of the pins
            "q_rope": None, "k_rope": None, "v": None,
        },
    }


def prefill_cases(d: Path, layer: int, rows) -> list[dict]:
    """Long-cache cases from a prefill dump: each row p is a decode-shaped query at position p
    attending lanes 0..p, with the SAME pinned cache the prefill built (§05.7 post-rope).

    The decode fixtures only reach ~25 lanes; the prefill pins reach 277 (p300), so this is the
    only pinned data in the repo that exercises a long read side.
    """
    qproj = pin(d, "L%02d.q_proj.out" % layer).reshape(-1, N_Q * 2 * DH)     # [seq, 4096]
    kproj = pin(d, "L%02d.k_proj.out" % layer).reshape(-1, N_KV * DH)
    vproj = pin(d, "L%02d.v_proj.out" % layer).reshape(-1, N_KV * DH)
    kc = pin(d, "prefill.L%02d.cache_keys" % layer).reshape(N_KV, -1, DH)
    vc = pin(d, "prefill.L%02d.cache_values" % layer).reshape(N_KV, -1, DH)
    pre_gate = pin(d, "L%02d.attn.attn_out_pre_gate" % layer).reshape(-1, N_Q, DH)
    qn, kn = attn_norms(layer)
    out = []
    for p in rows:
        cap = _cap_for(p + 1)
        k_slack, v_slack = _slack_cache(cap), _slack_cache(cap)
        k_slack[:, :p] = kc[:, :p]
        v_slack[:, :p] = vc[:, :p]
        out.append({
            "name": "prefill%03d.L%02d" % (p, layer), "kind": "prefill", "pos": p, "lvalid": p + 1,
            "cap": cap,
            "qg": qproj[p], "kg": kproj[p], "vg": vproj[p],
            "kcache": k_slack.reshape(-1), "vcache": v_slack.reshape(-1), "qn": qn, "kn": kn,
            "ref": {
                "q_rope": pin(d, "L%02d.attn.q_rope" % layer).reshape(N_Q, -1, DH)[:, p],
                "k_rope": pin(d, "L%02d.attn.k_rope" % layer).reshape(N_KV, -1, DH)[:, p],
                "v": pin(d, "L%02d.attn.v" % layer).reshape(N_KV, -1, DH), "cache_keys": kc[:, :p + 1],
            "cache_values": vc[:, :p + 1],
                "attn_pre_gate": pre_gate[p],
                "gated": pre_gate[p].reshape(-1)
                * torch.sigmoid(qproj[p].reshape(N_Q, 2 * DH)[:, DH:].reshape(-1)),
                "o_proj": None,
            },
            "stage_ref": {"q_rope": None, "k_rope": None, "v": None},
        })
    return out


def write_case(case: dict, indir: Path, outdir: Path) -> list:
    """Raw BF16 input files for the kernel, plus the plan line's POS/LCACHE/OUTDIR fields.
    The kernel names its outputs by stage (q_rope.bin, ...), so each case gets its own OUTDIR."""
    indir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    f = indir / (case["name"] + ".%s")
    qg = dump_bf16(case["qg"], Path(str(f) % "qg"))
    kg = dump_bf16(case["kg"], Path(str(f) % "kg"))
    vg = dump_bf16(case["vg"], Path(str(f) % "vg"))
    kc = dump_bf16(case["kcache"], Path(str(f) % "kc"))
    vc = dump_bf16(case["vcache"], Path(str(f) % "vc"))
    qn = dump_bf16(case["qn"], Path(str(f) % "qn"))
    kn = dump_bf16(case["kn"], Path(str(f) % "kn"))
    return [case["name"], str(qg), str(kg), str(vg), str(kc), str(vc), str(qn), str(kn),
            str(case["pos"]), str(case["lvalid"]), str(outdir)]


def read_out(outdir: Path, what: str, n: int) -> torch.Tensor:
    """One stage the kernel wrote into this case's own OUTDIR (the kernel names outputs by stage,
    which is why every case gets a directory)."""
    return read_bf16(Path(outdir) / (what + ".bin"), n)


# --------------------------------------------------------------------- the foundation
def verify(d: Path, case: dict) -> list[tuple[bool, str]]:
    """(ok, line) per claim the gate rests on. Any False and every comparison downstream is
    against the wrong tensor, which is the §10.5 trap this project has been bitten by twice."""
    c = case
    ok: list[tuple[bool, str]] = []
    # 1. the fixture pairing: the cache's last lane IS this step's rope/v, so pos = lvalid-1.
    if c["kind"] == "decode":
        nd_k = int((c["ref"]["cache_keys"][:, c["pos"]] != c["ref"]["k_rope"]).sum())
        nd_v = int((c["ref"]["cache_values"][:, c["pos"]] != c["ref"]["v"][:, c["pos"]]).sum())
        ok.append((nd_k == 0 and nd_v == 0,
                   "cache lane %d is attn.k_rope/attn.v (ndiff k %d v %d) -> pos = lvalid-1"
                   % (c["pos"], nd_k, nd_v)))
    qr, kr, vv, gate = head_stages(c["qg"], c["kg"], c["vg"], _layer_of(c), c["pos"])
    ok.append((int((qr != c["ref"]["q_rope"]).sum()) == 0,
               "torch chain reproduces attn.q_rope (ndiff %d/%d)"
               % (int((qr != c["ref"]["q_rope"]).sum()), qr.numel())))
    ok.append((int((kr != c["ref"]["k_rope"]).sum()) == 0,
               "torch chain reproduces attn.k_rope (ndiff %d/%d)"
               % (int((kr != c["ref"]["k_rope"]).sum()), kr.numel())))
    if c["ref"]["v"] is not None:
        ok.append((int((vv != c["ref"]["v"][:, c["pos"]]).sum()) == 0,
                   "torch chain reproduces attn.v at the appended lane"))
    a = attend(qr, kr, vv, c["kcache"].reshape(N_KV, c["cap"], DH),
               c["vcache"].reshape(N_KV, c["cap"], DH), c["pos"])
    nd = int((a != c["ref"]["attn_pre_gate"]).sum())
    ok.append((nd == 0, "torch chain reproduces attn_out_pre_gate (ndiff %d/%d)"
               % (nd, a.numel())))
    c["chain"] = {"q_rope": qr, "k_rope": kr, "v": vv, "gate": gate, "a_full": a}
    # 2. the misreadings must NOT reproduce it.
    qr_bad = head_stages(c["qg"], c["kg"], c["vg"], _layer_of(c), c["pos"], "f32_cs")[0]
    nd_bad = int((qr_bad != c["ref"]["q_rope"]).sum())
    if c["pos"] == 0:
        # Every angle is 0 at position 0: cos is exactly 1.0 and sin exactly 0.0, so rounding
        # cos/sin to BF16 cannot change anything and this case cannot pin the policy. Demanding a
        # difference here would be demanding something the arithmetic does not allow.
        ok.append((nd_bad == 0, "position 0: both rope policies give the same table (cos 1, sin"
                                " 0), %d/%d differ - the policy is pinned by the other positions"
                   % (nd_bad, qr_bad.numel())))
    else:
        ok.append((nd_bad > 0, "rope with F32 cos/sin (speccheck's apply_rope) is refused: it"
                               " misses %d/%d q_rope elements" % (nd_bad, qr_bad.numel())))
    kd = c["kcache"].reshape(N_KV, c["cap"], DH)
    vd = c["vcache"].reshape(N_KV, c["cap"], DH)
    if c["pos"] >= 1:
        a_off = attend(qr, kr, vv, kd, vd, c["pos"] - 1)      # the cache cut one lane short
        ok.append((int((a_off != c["ref"]["attn_pre_gate"]).sum()) > 0,
                   "a read side one lane short of the appended token is refused"))
    a_noround = attend(qr, kr, vv, kd, vd, c["pos"], score_round=False)
    nd_nr = int((a_noround != c["ref"]["attn_pre_gate"]).sum())
    ok.append((nd_nr >= nd,
               "BF16 scores are not worse than F32 scores against the pins (bf16 ndiff %d, f32"
               " ndiff %d): §10.3's 'scores live in BF16' is a policy the pins cannot distinguish"
               " at these cache lengths, so the kernel keeps the pinned one" % (nd, nd_nr)))
    if c["ref"]["o_proj"] is not None:
        gated = gate_apply(a, gate)
        nd_o = int(((gated @ o_proj_weight(_layer_of(c)).t()) != c["ref"]["o_proj"]).sum())
        ok.append((nd_o == 0, "the gated operand through o_proj reproduces the pin (ndiff %d)"
                   % nd_o))
    return ok


def _layer_of(case: dict) -> int:
    return int(case["name"].split(".L")[1])
