#!/usr/bin/env python3
"""S04.4's gated delta-rule recurrence in two arithmetic forms, CPU only (P9 Phase 0).

`recurrent_scan` is S04.4 verbatim: one step per token, the state `S` read after the
decay and before the write. `chunked_scan` is S04.7's UT-transform form at a chosen
chunk size `C`: intra-chunk work that is matmul-shaped, plus one serial carry of `S`
across chunks. Both take the same operands and return the same two things, so a diff
between them is only the arithmetic form.

The derivation, per value head, `d_k = d_v = 128`, S[i, j] with i over d_k and j over
d_v (§04.4's orientation). Write the recurrence with S~_t for the state after the decay
and before the write, so the read is from S~_t and never from S_t:

    S~_t = gamma_t S_{t-1}                 gamma_t = exp(g_t)
    r_t  = S~_t^T k_t                      (r_t[j] = sum_i S~_t[i, j] k_t[i])
    dv_t = beta_t (v_t - r_t)
    S_t  = S~_t + k_t (x) dv_t
    o_t  = q_t^T S_t                        (sum over i, so o_t is a row of d_v)

Unroll S~_t from the chunk start S_0 with Gamma_t = prod_{s<=t} gamma_s (so Gamma_t =
exp(cum_t), cum the within-chunk prefix sum of g). Every write is strictly *after* its
own read, so the sum runs over s < t only — that strict inequality is why the UT system
is unit-diagonal and why dropping it is a bug the negative cases here catch:

    S~_t = Gamma_t S_0 + sum_{s<t} (Gamma_t / Gamma_s) k_s (x) dv_s
    r_t  = Gamma_t (S_0^T k_t) + sum_{s<t} (Gamma_t / Gamma_s)(k_s . k_t) dv_s

Substitute r_t into dv_t and move the s < t terms to the left. With
L[t, s] = beta_t (Gamma_t / Gamma_s)(k_s . k_t) for s < t and L[t, t] = 0 (unit diagonal),
and the right-hand side linear in t:

    L dv = (beta . v) - ((beta . Gamma) . k) S_0

so one unit-lower-triangular solve per chunk gives dv for all C tokens, and it splits
into two solves because the right side is a sum of two terms:

    dv = U ((beta . v)) - U ((beta . Gamma) . k) S_0        U = L^-1

The read keeps the diagonal term, because a token's own write *is* in the state it reads:

    o_t = Gamma_t (q_t^T S_0) + sum_{s<=t} (Gamma_t / Gamma_s)(q_t . k_s) dv_s

and the chunk's end condition — the state the next chunk inherits — decays every write
it holds to the chunk boundary:

    S_C = Gamma_C S_0 + [((Gamma_C / Gamma) . k)]^T dv

`initial_state` is the first clause and `S_C` the second; §08.6's cross-body carry and a
resumed request both land on the first one, so `chunked_scan` takes it and returns the
second. Padding: a ragged tail is right-padded with beta = 0 and g = 0 (§04.7, A-08).
beta = 0 zeroes that row of L and of the right-hand side, so a pad row neither reads nor
writes state, and g = 0 leaves the chunk's total decay alone.

`mma=True` models the device shape rather than the oracle shape: every matmul operand is
rounded to BF16 and the accumulate stays F32, which is what the matrix units on this box
do (bf16 in, f32 accumulate — `devices/strix-halo-compute.md`). It is a *model* of a
kernel, not a kernel: it is how Phase 0 prices the band a real chunked kernel would need
before one is written.

    .venv/bin/python prototype/gdn_chunk/chunk_scan.py --selftest
"""
from __future__ import annotations

import argparse
import math

import torch

D_K = D_V = 128
H_V = 16
EPS = 1e-6
Q_SCALE = D_K ** -0.5                     # S04.4 / A-05: on q only, never on k


def qk_normalized(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """S04.4's l2 normalization: F32, epsilon inside the root, 128^-1/2 on q only."""
    qf, kf = q.float(), k.float()
    qn = qf * torch.rsqrt(qf.pow(2).sum(-1, keepdim=True) + EPS) * Q_SCALE
    kn = kf * torch.rsqrt(kf.pow(2).sum(-1, keepdim=True) + EPS)
    return qn, kn


def _round_bf16(x: torch.Tensor) -> torch.Tensor:
    """The value an MFMA operand would see: BF16 in, F32 back out."""
    return x.to(torch.bfloat16).float()


def _mm(a: torch.Tensor, b: torch.Tensor, mma: bool) -> torch.Tensor:
    """One matmul, either in F32 (the oracle's shape) or with BF16 operands and an F32
    accumulator (the matrix units' shape). Accumulation is F32 in both."""
    if not mma:
        return a @ b
    return _round_bf16(a) @ _round_bf16(b)


def recurrent_scan(qn, kn, v, g, beta, S=None, bug=""):
    """S04.4 token loop — the definition. qn/kn [T,H,dk], v [T,H,dv], g/beta [T,H],
    S [H,dk,dv] (None = the request-start zero). Returns (o [T,H,dv], S)."""
    T, H, DK = kn.shape
    DV = v.shape[-1]
    gam = g.float().exp()
    o = torch.zeros((T, H, DV), dtype=torch.float32)
    S = torch.zeros((H, DK, DV), dtype=torch.float32) if S is None else S.float().clone()
    for t in range(T):
        s_t = S * gam[t][:, None, None]           # decay first (HF-ref order, S04.4)
        if bug == "read_before_decay":            # §04.4's order, reversed
            s_read = S
        else:
            s_read = s_t
        r = torch.einsum("hkd,hk->hd", s_read, kn[t])       # retrieve from the decayed S
        dv = beta[t][:, None] * (v[t].float() - r)          # delta: the correction only
        if bug == "read_after_write":
            # retrieve from a state that already holds an update: the order §04.4 forbids
            naive = s_t + torch.einsum("hk,hd->hkd", kn[t], beta[t][:, None] * v[t].float())
            r = torch.einsum("hkd,hk->hd", naive, kn[t])
            dv = beta[t][:, None] * (v[t].float() - r)
        s_t = s_t + torch.einsum("hk,hd->hkd", kn[t], dv)   # rank-1 write
        o[t] = torch.einsum("hk,hkd->hd", qn[t], s_t)       # read, from the stored S
        S = s_t
    return o, S


def ut_inverse(strict_lower, mma=False):
    """Inverse of the unit lower-triangular L = I + strict_lower, [..., C, C] -> [..., C, C].

    Forward substitution down the rows, batched over every (chunk, head): row i of U is
    e_i minus the already-known rows above it. A kernel blocks this (invert 16x16
    diagonal blocks, combine the off-diagonal blocks by matmul); the row order and hence
    the rounding chain are the same either way, so this is the honest CPU mirror of it.
    `mma` is deliberately NOT applied inside the substitution: the C x C coefficients are
    scalar F32 products there, not a matrix-unit tile, and pretending otherwise would
    price a rounding the kernel does not have to take.
    """
    lead, C = strict_lower.shape[:-2], strict_lower.shape[-1]
    A = strict_lower.reshape(-1, C, C)
    U = torch.zeros_like(A)
    for i in range(C):
        row = torch.zeros_like(U[:, i])                    # e_i, one row per (chunk, head)
        row[:, i] = 1.0
        if i:
            # [B,1,i] @ [B,i,C] -> [B,1,C]: the 1 is written out so the matmul can never
            # read the batch axis as the row axis, which is what a bare [B,i] would risk.
            row = row - _mm(A[:, i, :i].unsqueeze(1), U[:, :i, :], mma).squeeze(1)
        U[:, i, :] = row
    return U.reshape(*lead, C, C)


def chunked_scan(qn, kn, v, g, beta, S=None, C=64, mma=False, bug=""):
    """S04.7 chunked UT-transform form of the same recurrence. Same operands and return
    values as `recurrent_scan`; `C` is the chunk size, `mma` models BF16 matmul operands.

    `bug` injects one plausible mistake per name, for the negative cases in `spike.py`
    (a gate that cannot fail verifies nothing)."""
    T, H, DK = kn.shape
    DV = v.shape[-1]
    NC = -(-T // C)
    pad = NC * C - T

    def ascend(x):                                  # [T,H,last] -> [H,NC,C,last]
        if pad:
            x = torch.cat([x, x.new_zeros((pad,) + tuple(x.shape[1:]))], 0)
        return x.float().transpose(0, 1).reshape(H, NC, C, -1)

    q, k, vv = ascend(qn), ascend(kn), ascend(v)
    de = ascend(g).squeeze(-1)                     # [H,NC,C]
    be = ascend(beta).squeeze(-1)

    cum = de.cumsum(-1)                            # log of Gamma_t, within the chunk
    upper = torch.ones(C, C, dtype=torch.bool).triu(1)
    # dec[t, s] = Gamma_t / Gamma_s for s <= t, 0 above the diagonal
    dec = (cum[..., :, None] - cum[..., None, :]).masked_fill(upper, float("-inf")).exp()
    if bug == "causality_upper":                   # keep the wrong triangle
        dec = (cum[..., :, None] - cum[..., None, :]).masked_fill(upper.flip(-1, -2),
                                                                 float("-inf")).exp()
    if bug == "no_pairwise_decay":                 # the intra-chunk decay dropped
        dec = torch.ones_like(dec).tril()

    k_beta = k * be[..., None]
    ut = _mm(k_beta, k.transpose(-1, -2), mma) * dec
    intra = _mm(q, k.transpose(-1, -2), mma) * dec          # diagonal kept: a token reads
    #                                                             its own write (§04.4)
    dec_k_beta = k_beta * cum.exp()[..., None]
    U = ut_inverse(torch.tril(ut, -1))
    nv = _mm(U, vv * be[..., None], mma)
    kcd = _mm(U, dec_k_beta, mma)
    if bug == "no_inter_chunk_read":               # the state's prediction never subtracts
        kcd = torch.zeros_like(kcd)

    q = q * cum.exp()[..., None]                   # apply the chunk-prefix decay once
    k = k * (cum[..., -1:] - cum).exp()[..., None]
    cdec = cum[..., -1].exp()[..., None, None]
    if bug == "no_chunk_decay":                    # the carry forgets the chunk's decay
        cdec = torch.ones_like(cdec)

    S = torch.zeros((H, DK, DV), dtype=torch.float32) if S is None else S.float().clone()
    out = torch.zeros((H, NC, C, DV), dtype=torch.float32)
    last = NC if bug != "drop_last_chunk" else NC - 1     # end condition: last chunk left
    for i in range(last):                                  # out of the state and the output
        v_new = nv[:, i] - _mm(kcd[:, i], S, mma)
        out[:, i] = _mm(q[:, i], S, mma) + _mm(intra[:, i], v_new, mma)
        S = S * cdec[:, i] + _mm(k[:, i].transpose(-1, -2), v_new, mma)
    o = out.reshape(H, NC * C, DV)[:, :T].transpose(0, 1).contiguous()
    return o, S


def _selftest() -> int:
    """The two forms must agree on random operands — including a nonzero initial state
    and a ragged tail — and every injected bug must move the stage it lives in."""
    torch.manual_seed(0)
    fails: list[str] = []
    for (T, C) in [(64, 64), (46, 64), (277, 64), (46, 16), (46, 32), (200, 16),
                   (200, 32), (512, 64), (33, 16)]:
        # Operands the shape of the real ones (measured on data/refs/p300 L00: kn unit
        # norm, qn 0.0884, |v| ~ 0.04, g mean -0.72 with max -0.0000, beta in (0,1)). The
        # `** 2` matters: the pins have heads whose gate is essentially 0, so the state
        # carries indefinitely, and a flat -2·rand decay would let an implementation that
        # drops the inter-chunk carry pass here and fail on the pins.
        qn, kn = qk_normalized(torch.randn(T, H_V, D_K), torch.randn(T, H_V, D_K))
        v = 0.04 * torch.randn(T, H_V, D_V)
        g = -2.0 * torch.rand(T, H_V) ** 2
        beta = torch.sigmoid(torch.randn(T, H_V))
        ref, S_ref = recurrent_scan(qn, kn, v, g, beta)
        for mma in (False, True):
            o, S = chunked_scan(qn, kn, v, g, beta, C=C, mma=mma)
            cos = float((o.flatten() @ ref.flatten()
                         / (o.flatten().norm() * ref.flatten().norm())))
            if not math.isfinite(cos) or cos < (0.9999 if not mma else 0.99):
                fails.append(f"T={T} C={C} mma={mma}: chunk vs recurrent cos={cos:.3e}")
        # nonzero initial state: the carry clause, with and without a ragged tail
        o, S = chunked_scan(qn, kn, v, g, beta, S=torch.randn(H_V, D_K, D_V), C=C)
        o2, S2 = chunked_scan(qn, kn, v, g, beta, C=C)
        if float((o - o2).abs().max()) < 1e-6:
            fails.append(f"T={T} C={C}: a nonzero initial state changed nothing")
        # a beta=0 tail is inert: padding must not move any real row or the final state
        Tp = T + 7
        qp = torch.cat([qn, torch.zeros(7, H_V, D_K)])
        kp = torch.cat([kn, torch.zeros(7, H_V, D_K)])
        vp = torch.cat([v, torch.zeros(7, H_V, D_V)])
        gp = torch.cat([g, torch.zeros(7, H_V)])
        bp = torch.cat([beta, torch.zeros(7, H_V)])
        op, _ = chunked_scan(qp, kp, vp, gp, bp, C=C)
        if not torch.equal(op[:T], o2):
            fails.append(f"T={T} C={C}: zero-beta padding moved a real row (A-08 broken)")
    # each injection must be caught by *its own* stage, and only there
    T, C = 200, 64
    torch.manual_seed(1)
    qn, kn = qk_normalized(torch.randn(T, H_V, D_K), torch.randn(T, H_V, D_K))
    v = 0.04 * torch.randn(T, H_V, D_V)
    g, beta = -2.0 * torch.rand(T, H_V) ** 2, torch.sigmoid(torch.randn(T, H_V))
    ref, S_ref = recurrent_scan(qn, kn, v, g, beta)
    good_o, good_S = chunked_scan(qn, kn, v, g, beta, C=C)

    def meanrel(a, b):
        return float((a - b).abs().mean() / b.abs().mean()) if b.abs().mean() else 0.0

    # The legal delta is the route change itself: chunked-vs-sequential in F32. Every injected
    # bug has to clear it by a wide margin, otherwise the negative is calibrated by taste
    # (§10.1 F1-F4: a gate threshold has to be measured, not picked).
    legal = max(meanrel(good_o, ref), meanrel(good_S, S_ref))
    print(f"  legal chunk-vs-sequential meanrel: {legal:.3e} "
          f"(every bug below must exceed {100 * legal:.1e})")
    for name in ("no_inter_chunk_read", "causality_upper", "no_pairwise_decay",
                 "no_chunk_decay", "drop_last_chunk", "read_before_decay",
                 "read_after_write"):
        fn = recurrent_scan if name.startswith("read_") else chunked_scan
        kw = {} if name.startswith("read_") else {"C": C}
        o, S = fn(qn, kn, v, g, beta, bug=name, **kw)
        finite = bool(torch.isfinite(o).all() and torch.isfinite(S).all())
        ro, rs = meanrel(o, ref), meanrel(S, S_ref)
        cos = float((o.flatten() @ ref.flatten()
                     / (o.flatten().norm() * ref.flatten().norm() + 1e-30)))
        if finite and max(ro, rs) < 100.0 * legal:
            fails.append(f"negative {name}: NOT caught (o meanrel {ro:.3e}, S meanrel {rs:.3e})")
        print(f"  negative {name:22s} " + (f"o meanrel {ro:.3e}  state meanrel {rs:.3e}  o cos {cos:.6f}"
              if finite else "non-finite output (caught: a real kernel never writes NaN)"))
    print(f"selftest: {len(fails)} failures" + ("".join("\n  " + f for f in fails) if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
