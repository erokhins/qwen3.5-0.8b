#!/usr/bin/env python3
"""The MTP draft block, mirrored in the ENGINE'S stage structure — plus the two things that make a
mirror admissible as a reference at all.

WHY A MIRROR. `speccheck/mtp.py`'s `MTPDraft.draft` is the shipped MTP reference and it is what the
pinned golden came from, but it computes §05's attention in one shape (F32 rope table, F32 scores —
`speccheck/specimpl.py:164-181` says so in terms and records that making it bit-faithful to the model's
own pinned policy "was tried and reverted"), while the kernels this rig launches use the pinned policy:
`prototype/attn_decode` is gated bit-exact against a BF16 rope table with each product rounded (§05.3),
and its gated `full` softmax scores as `bf(fb(q·k)) * SCALE`. So a single reference cannot judge both
sides honestly. This module therefore computes the same §09.2 step under TWO policies and lets each
adjudicate what it is the authority on:

    policy="oracle"   speccheck's plumbings, verbatim calls into `speccheck.specimpl`. WITNESS ONLY:
                      the mirror's right to exist is that this column reproduces `MTPDraft.draft`
                      bit-for-bit on `u`, `x_d` and the proposal (checked in `--selftest`, and it is the
                      same class of check as `prototype/attn_decode`'s `reference.attend` witness —
                      "the mirror of the shipped form must BE the shipped form").
    policy="pinned"   the model's own policy, stage by stage as the engine stages it: norm → bf16 rope
                      table with rounded products → bf16-rounded scores → BF16-rounded normalised
                      weights → P·V. This is the column the DEVICE is judged against, and the difference
                      between the two columns is REPORTED as a number (DRAFT-HANDOVER.md §1), not
                      absorbed into a widened band.

`fc_order`, `h_source` and the position of each step are parameters here because they are the knobs
§09.5 settled by accept rate, and because the negative-control sets the gate derives are computed by
diffing this function against itself.

CPU only, torch CPU bf16. No device, no timing.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from speccheck.specimpl import (EPS, N_Q, D_HEAD, Weights, apply_rope, attention,  # noqa: E402
                                mlp as s0_mlp, rmsnorm_zc, rope_cs)
import bf16_ulp as U                                            # noqa: E402
from speccheck.mtp import MP, MTPDraft                          # noqa: E402

HIDDEN = 1024
EMBED = "model.language_model.embed_tokens.weight"
SHARD = ROOT / "models" / "Qwen3.5-0.8B" / "model.safetensors-00001-of-00001.safetensors"
GOLDEN = ROOT / "prototype" / "mtp_gatezero" / "golden"
WORK = pathlib.Path("/tmp/mtp_draft")

# The block's 15 tensors, with the RANK each is claimed to have. `engine/weights_contract.py` exists
# because a checker that cannot state a rank will eventually be wrong about it, and the AGENTS entry
# names the exact failure. None of these are rank-3, and saying so is what makes the claim checkable.
BLOCK = [
    ("mtp.fc.weight",                                       (1024, 2048)),
    ("mtp.layers.0.input_layernorm.weight",                (1024,)),
    ("mtp.layers.0.mlp.down_proj.weight",                  (1024, 3584)),
    ("mtp.layers.0.mlp.gate_proj.weight",                  (3584, 1024)),
    ("mtp.layers.0.mlp.up_proj.weight",                    (3584, 1024)),
    ("mtp.layers.0.post_attention_layernorm.weight",       (1024,)),
    ("mtp.layers.0.self_attn.k_norm.weight",               (256,)),
    ("mtp.layers.0.self_attn.k_proj.weight",               (512, 1024)),
    ("mtp.layers.0.self_attn.o_proj.weight",               (1024, 2048)),
    ("mtp.layers.0.self_attn.q_norm.weight",               (256,)),
    ("mtp.layers.0.self_attn.q_proj.weight",               (4096, 1024)),
    ("mtp.layers.0.self_attn.v_proj.weight",               (512, 1024)),
    ("mtp.norm.weight",                                     (1024,)),
    ("mtp.pre_fc_norm_embedding.weight",                    (1024,)),
    ("mtp.pre_fc_norm_hidden.weight",                       (1024,)),
]
BLOCK_BYTES = sum(int(np.prod(s)) * 2 for _, s in BLOCK)


# ── band imports: this rig owns NO numeric constant of its own ──────────────────────────────────
def _consts(path: pathlib.Path, names: dict) -> dict:
    """Read module-level literal constants out of another op's gate driver.

    WHY READ THE SOURCE INSTEAD OF IMPORTING IT: those drivers import their own siblings (`operands`,
    `reference`, …) by bare name, and four different op directories define modules with the same names,
    so importing all four to harvest a number would let the first `operands.py` on the path answer for
    every one of them — a silent cross-feed of exactly the kind every other row of the AGENTS trap table
    is about. An AST read of the assignment cannot be cross-fed, and it fails loudly when a name is
    missing or was never a literal. Values are only ever taken from a top-level assignment to one of the
    requested names; nothing in the imported file is executed.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    got = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        # Both `A = 1` and `A, B = 1, 2` forms appear in these drivers, so flatten the target side and the
        # value side the same way and pair them positionally.
        tg, vals = [], []
        for t in node.targets:
            tg += list(t.elts) if isinstance(t, ast.Tuple) else [t]
        vals = list(node.value.elts) if isinstance(node.value, ast.Tuple) else [node.value] * len(tg)
        for t, v in zip(tg, vals):
            if isinstance(t, ast.Name) and t.id in names and isinstance(v, ast.Constant) \
                    and isinstance(v.value, (int, float)):
                got[t.id] = v.value
    missing = [n for n in names if n not in got]
    if missing:
        raise SystemExit(f"{path}: cannot read band constant(s) {missing}; the band is not this rig's to "
                         "invent, so refusing to run with a default")
    return got


def bands() -> dict:
    """The per-observable rules, each taken from the op whose floor probe measured it. The `ndiff` for
    the MLP class is `prototype/glue/run.py`'s 2 %, which is also `prototype/mlp_gemv/run.py:129`'s own
    `judge(ndiff_pct_max=2.0)` default — stated rather than imported because a function default is not a
    module-level literal and this rig will not invent a third number for it."""
    P = ROOT / "prototype"
    glue = _consts(P / "glue" / "run.py", {"CANCEL_REL": 0, "NDIFF_PCT_MAX": 0,
                                           "COS_MIN": 0, "MEANREL_MAX": 0})
    proj = _consts(P / "proj_gemv" / "run.py", {"CANCEL_REL": 0, "NDIFF_PCT_MAX": 0,
                                                "COS_MIN": 0, "MEANREL_MAX": 0})
    attn = _consts(P / "attn_decode" / "run.py", {"CANCEL_REL": 0, "NDIFF_PCT_MAX": 0,
                                                  "COS_MIN": 0, "MEANREL_MAX": 0})
    mlp = _consts(P / "mlp_gemv" / "run.py", {"CANCEL_REL": 0, "F4_COS_MIN": 0,
                                              "F4_MEANREL_MAX": 0})
    return {
        "norm":   dict(cancel=glue["CANCEL_REL"], ndiff_pct=glue["NDIFF_PCT_MAX"], ulp=1,
                       cos=glue["COS_MIN"], meanrel=glue["MEANREL_MAX"], src="prototype/glue/run.py"),
        "proj":   dict(cancel=proj["CANCEL_REL"], ndiff_pct=proj["NDIFF_PCT_MAX"], ulp=1,
                       cos=proj["COS_MIN"], meanrel=proj["MEANREL_MAX"],
                       src="prototype/proj_gemv/run.py"),
        "attn":   dict(cancel=attn["CANCEL_REL"], ndiff_pct=attn["NDIFF_PCT_MAX"], ulp=1,
                       cos=attn["COS_MIN"], meanrel=attn["MEANREL_MAX"],
                       src="prototype/attn_decode/run.py"),
        "mlp":    dict(cancel=mlp["CANCEL_REL"], ndiff_pct=glue["NDIFF_PCT_MAX"], ulp=1,
                       cos=0.9999, meanrel=1e-2, src="prototype/mlp_gemv/run.py (F1..F3)"),
        "chain":  dict(cancel=None, ndiff_pct=None, ulp=None, cos=mlp["F4_COS_MIN"],
                       meanrel=mlp["F4_MEANREL_MAX"],
                       src="prototype/mlp_gemv/run.py (F4, aggregates only)"),
    }


# ── the fixture: which pinned file is the drafter's `h`, decided by reproduction ────────────────
def fixture(fixture: str) -> dict:
    """`h`, `y_next` and the position, all READ OFF the fixture, plus the witness that the pairing is
    right. `prototype/mtp_gatezero/pin_pair_search.py` exists for this question and §10.5's
    last-writer-wins trap is why it cannot be answered from a filename."""
    from safetensors import safe_open
    d = ROOT / "data" / "refs" / fixture
    man = json.loads((d / "manifest.json").read_text())
    n_prefill = len(man["prefill_ids"])
    decs = [int(q.name.split(".")[0][len("decode"):])
            for q in d.glob("decode*.logits_last.npy")]
    last = max(decs or [-1])

    def pin(name):
        a = np.load(d / f"{name}.npy").reshape(-1)
        if a.dtype == np.uint16:
            a = torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16)
            return torch.from_numpy(a.float().numpy())
        return torch.from_numpy(a.astype(np.float32))

    with safe_open(str(SHARD), framework="pt") as f:
        E = f.get_tensor(EMBED)
    h_pin = pin("final_hidden").to(torch.bfloat16).reshape(1, HIDDEN)
    pre_pin = pin("pre_final_hidden").to(torch.bfloat16).reshape(1, HIDDEN)
    logits = pin(f"decode{last}.logits_last").reshape(-1)
    pred = (h_pin.float() @ E.float().t()).reshape(-1)
    cos = float(torch.dot(pred, logits) /
                (torch.linalg.norm(pred) * torch.linalg.norm(logits)))
    if cos < 0.9999:
        raise SystemExit(
            f"{fixture}: final_hidden does not reproduce decode{last}.logits_last (cos {cos:.6f}); the "
            "pin pairing is not what this rig assumes. Run "
            "prototype/mtp_gatezero/pin_pair_search.py before trusting any number here.")
    y_next = int(logits.argmax())
    return {"fixture": fixture, "h": h_pin, "h_pre": pre_pin, "y_next": y_next,
            "pos": n_prefill + last + 1, "n_prefill": n_prefill, "last_decode": last,
            "witness_cos": cos, "E": E, "logit_row": logits}


D_HEAD_N = HIDDEN   # kept for readers of the first draft of this file


# ── the pinned golden ───────────────────────────────────────────────────────────────────────────
def golden(fixture: str) -> dict:
    """`prototype/mtp_gatezero/golden` holds `u`, `x_d` and the draft logits per step, written by
    `draft_golden.py --write`, which derived them from `speccheck.mtp.MTPDraft` itself. Bit-exactness
    against these is the admission the MTP-IMPL §2 plan asks for."""
    base = GOLDEN / f"{fixture}.json"
    if not base.exists():
        raise SystemExit(f"no golden for {fixture}; run "
                         "prototype/mtp_gatezero/draft_golden.py --fixture {} --write".format(fixture))
    meta = json.loads(base.read_text())
    steps = []
    for j in range(meta["n_steps"]):
        rec = {}
        for name in ("u", "x_d"):
            bits = np.load(GOLDEN / f"{fixture}.step{j}.{name}.npy")
            rec[name] = torch.from_numpy(bits).view(torch.bfloat16)
        steps.append(rec)
    return {"meta": meta, "steps": steps}


# ── BF16 helpers ────────────────────────────────────────────────────────────────────────────────
def rb(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 and keep computing in F32 — what `bf(fb(...))` is in the kernels."""
    return x.to(torch.bfloat16).float()


class _Shim:
    """`specimpl.attention`/`mlp` only ever reach the weights through `w.p[...]`, so a dict in a hat is
    enough to call the shipped functions on a perturbed block without going through `Weights`' shape
    census (which belongs to the target model's key set, not this rig's)."""

    def __init__(self, p: dict):
        self.p = p


def rope_bf16(pos: int, dev) -> tuple:
    """§05.3 as the PINS have it: the table itself is BF16 (attn_decode's `rope_cs`: `bf(fb(cosf(ang)))`),
    which is the difference this rig exists to keep visible."""
    cos, sin = rope_cs(torch.tensor([pos], device=dev), dev)
    return rb(cos[0]), rb(sin[0])


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def apply_rope_pinned(x, cos, sin):
    """Rounded table AND rounded products, so `rb(rb(z*c) + rb(rh*s))` — the shipped expression."""
    rot, rest = x[..., :64], x[..., 64:]
    c, s = cos[None, :], sin[None, :]
    r = rb(rb(rot * c) + rb(rotate_half(rot) * s))
    return torch.cat([r, rest], -1)


# ── the mirror ──────────────────────────────────────────────────────────────────────────────────
def mirror_step(wp: dict, h: torch.Tensor, y_next: int, pos: int, kv: dict, policy: str,
                fc_order: str = "eh", E: torch.Tensor | None = None):
    """One §09.2 draft step, staged the way the engine stages it. Returns a dict of every intermediate
    the device dumps, keyed by the dump name, plus the proposal.

    `policy` selects the AUTHORITY for the attention stage only ("oracle" = the shipped
    `specimpl.attention`, verbatim; "pinned" = the model's BF16 policy as the engine stages it). The two
    columns share every other stage, so their difference IS the documented policy gap and nothing else."""
    pre = MP + "layers.0."
    E = wp[EMBED] if E is None else E
    e_raw = E[[y_next]]                                     # §09.1, shared table
    e = rmsnorm_zc(e_raw, wp[MP + "pre_fc_norm_embedding.weight"])
    hh = rmsnorm_zc(h, wp[MP + "pre_fc_norm_hidden.weight"])
    if fc_order == "eh":
        pair = torch.cat([e, hh], -1)
    elif fc_order == "he":
        pair = torch.cat([hh, e], -1)
    else:
        raise ValueError(fc_order)
    u = pair @ wp[MP + "fc.weight"].t()                     # -> [1, 1024]

    nrm = rmsnorm_zc(u, wp[pre + "input_layernorm.weight"])
    qg = (nrm @ wp[pre + "self_attn.q_proj.weight"].t()).view(1, N_Q, D_HEAD * 2)
    q_raw, gate = qg[..., :D_HEAD], qg[..., D_HEAD:]
    kg = (nrm @ wp[pre + "self_attn.k_proj.weight"].t()).view(1, 2, D_HEAD)
    vg = (nrm @ wp[pre + "self_attn.v_proj.weight"].t()).view(1, 2, D_HEAD)
    qh0 = rmsnorm_zc(q_raw, wp[pre + "self_attn.q_norm.weight"])
    kh0 = rmsnorm_zc(kg, wp[pre + "self_attn.k_norm.weight"])

    if policy == "oracle":
        # The shipped form VERBATIM — `speccheck.specimpl.attention` with the MTP prefix, called exactly
        # the way `MTPDraft.draft` calls it. The oracle column is not a paraphrase of the reference; it
        # IS the reference, which is what makes it admissible as the witness the selftest checks against
        # the pinned golden.
        cos, sin = rope_cs(torch.tensor([pos], device=u.device), u.device)
        dbg: dict = {}
        attn_o = attention(_Shim(wp), 0, nrm, cos, sin, kv, dbg, "full",
                           prefix=pre + "self_attn.")
        qh, kh = dbg["q_rope"], dbg["k_rope"]
        aout = dbg["attn_out_pre_gate"].reshape(1, N_Q, D_HEAD)
        gated = aout.reshape(1, N_Q * D_HEAD) * torch.sigmoid(gate.reshape(1, N_Q * D_HEAD))
        attn_o = attn_o.reshape(1, HIDDEN)
    elif policy == "pinned":
        # The model's own policy — BF16 rope table with rounded products (§05.3, which `attn_decode` is
        # bit-exact against) and scores that round through BF16 before the scale (§05.5, same form
        # `attn_decode`'s gated `full` softmax uses). This is the column the DEVICE is judged against.
        cos, sin = rope_bf16(pos, u.device)
        qh = apply_rope_pinned(qh0, cos, sin)
        kh = apply_rope_pinned(kh0, cos, sin)
        kv.setdefault("k", []).append(kh)
        kv.setdefault("v", []).append(vg)
        kc = torch.cat(kv["k"], 0)                          # [lanes, 2, 256]
        vc = torch.cat(kv["v"], 0)
        aout = torch.zeros((1, N_Q, D_HEAD), dtype=torch.float32)
        for i in range(N_Q):
            j = i // (N_Q // 2)
            s = rb((qh[0, i, :][None, :] @ kc[:, j, :].t()).float()) * 0.0625
            s = s[0]                                          # bf(fb(q·k)) * SCALE, an exact ×1/16
            m = float(s.max())                                # the max-subtracted form §05.5
            l = float(torch.exp(s - m).sum())
            wt = rb(torch.exp(s - m) / l)                     # the BF16 round of §05.5
            aout[0, i] = (wt[:, None] * vc[:, j, :].float()).sum(0)
        # §05.6 transcribed from the shipped epilogue rather than paraphrased: `av = bf(acc)` is ROUNDED
        # to BF16 first, the gate value's sigmoid is rounded, and only then are the two multiplied and
        # rounded — `bf(av * bf(fb(1/(1+exp(-g)))))`. Mirroring the algebra instead of the source text is
        # what made the k=2 run 0.2 % mean-relative off everything downstream of here while `aout` itself
        # was bit-exact, which is the same trap as a hoisted scalar constant in a norm.
        gated = rb(rb(aout).reshape(1, N_Q * D_HEAD) * rb(
            1.0 / (1.0 + torch.exp(-gate.reshape(1, N_Q * D_HEAD).float()))))    # §05.6
        attn_o = gated.to(torch.bfloat16) @ wp[pre + "self_attn.o_proj.weight"].t()
    else:
        raise ValueError(policy)

    z1 = (u + attn_o).to(torch.bfloat16)                    # §03.2, one BF16 round
    u2 = rmsnorm_zc(z1, wp[pre + "post_attention_layernorm.weight"])
    if policy == "oracle":
        mlp_o = s0_mlp(_Shim(wp), 0, u2, prefix=pre + "mlp.")   # verbatim, like MTPDraft calls it
        g = u2 @ wp[pre + "mlp.gate_proj.weight"].t()
        up = u2 @ wp[pre + "mlp.up_proj.weight"].t()
        act = (torch.nn.functional.silu(g) * up).to(torch.bfloat16)
    else:
        g = u2 @ wp[pre + "mlp.gate_proj.weight"].t()
        up = u2 @ wp[pre + "mlp.up_proj.weight"].t()
        act = (torch.nn.functional.silu(g) * up).to(torch.bfloat16)
        mlp_o = act @ wp[pre + "mlp.down_proj.weight"].t()
    z2 = (z1 + mlp_o).to(torch.bfloat16)
    x_d = rmsnorm_zc(z2, wp[MP + "norm.weight"])
    logits = x_d @ E.t()
    proposal = int(logits[0].argmax())
    return {"e": e_raw.reshape(-1), "pair": pair.reshape(-1), "u": u.reshape(-1),
            "nrm": nrm.reshape(-1), "qg": qg.reshape(-1), "kg": kg.reshape(-1),
            "vg": vg.reshape(-1), "qrop": qh.reshape(-1), "krop": kh.reshape(-1),
            "gate": gate.reshape(-1), "aout": aout.reshape(-1).to(torch.bfloat16),
            "gout": gated.reshape(-1).to(torch.bfloat16), "attn_o": attn_o.reshape(-1),
            "z1": z1.reshape(-1), "u2": u2.reshape(-1), "act": act.reshape(-1),
            "mlp_o": mlp_o.reshape(-1), "z2": z2.reshape(-1), "xd": x_d.reshape(-1),
            "cache_k": torch.cat(kv["k"], 0).reshape(-1), "cache_v": torch.cat(kv["v"], 0).reshape(-1),
            "proposal": proposal, "logits": logits.reshape(-1)}


def mirror_chain(wp: dict, h: torch.Tensor, steps: list, policy: str, fc_order="eh",
                 pos_shift=0, pos_last=0, E=None):
    """k chained steps. `h` for step j>0 is the previous step's own `x_d` (§09.3 — the shipped dataflow,
    so a rounding difference is allowed to propagate exactly as it does on the device). The token id is
    taken from the STEP LIST, never from the mirror's own proposal: the gate teacher-forces the drafted
    id from the pinned golden so that a divergence at step j is attributable to numerics and not to a
    different token. Free-running accept rate is D3/D4's measurement, not this gate's.

    The MTP cache is DENSE and each row carries its own absolute position — DRAFT-HANDOVER.md §3."""
    kv: dict = {}
    out = []
    hd = h
    for j, st in enumerate(steps):
        y = st.y_next if hasattr(st, "y_next") else st[0]
        pos = (st.pos if hasattr(st, "pos") else st[1]) + pos_shift \
              + (pos_last if j == len(steps) - 1 else 0)
        rec = mirror_step(wp, hd, y, pos, kv, policy, fc_order, E=E)
        out.append(rec)
        hd = rec["xd"].unsqueeze(0)
    return out


__all__ = ["BLOCK", "BLOCK_BYTES", "bands", "fixture", "golden", "mirror_step", "mirror_chain",
           "MTPDraft", "attention", "ROOT", "WORK", "EMBED", "rb"]
