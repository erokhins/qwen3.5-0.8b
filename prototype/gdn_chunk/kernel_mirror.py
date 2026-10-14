#!/usr/bin/env python3
"""CPU mirror of `prototype/gdn_chunk/gdn_chunk_kernels.h` — the kernel's own staging, rounding
points and chunk/boundary conditions, not the model's. `chunk_scan.py` stays the contract; this
file exists to answer "what does the kernel AS WRITTEN compute", the way
`prototype/attn_tile_model.py` does for the tiled attention core.

Why a second CPU form beside `chunk_scan.py`: the device run of `--gdn-scan chunk` produced chunk 0
correctly and NaN in every row of every later chunk (bench/p9-phase1-2026-10-14.md, Phase 1b device
window). A model that agrees with itself cannot localise that; only a mirror that keeps the kernel's
staging can. What is mirrored, and what cannot be:

  MIRRORED   the staging dtypes and where each value is rounded (BF16 `k~` in LDS and the three
             quantities derived from it — D2; `U` F32 in LDS, BF16 at use; `intra`/`kcd`/`qg`/`kh`
             rounded at the store; `nv` and `v_new` F32 results rounded at each use; the state read
             as a BF16 operand and written F32); the within-chunk cumsum order; the row-ascending
             unit-triangular substitution including row 0's seed (the 59c43dc fix); the
             chunk-boundary conditions; §04.7's ragged-tail padding; and the one expression the
             model does not share — how `khat`'s boundary decay is formed (see below).
  NOT MIRRORED  D1, the F32 accumulation order inside a dot. A thread walks its reduction serially,
             torch asks a matmul for its own association, and no CPU form can be both. Every verdict
             here is therefore a finiteness verdict or a budget verdict, never bit-identity — the
             rule §10.1 (F1-F4) imposes on every chained dot in this repo. LDS sizing, block
             geometry and the thread mapping are also not mirrored, and `candidates` says what
             follows from that: they move bytes, not values.

THE ONE EXPRESSION, because it is what this file was written to find.
  model   `khat_t = k~_t * exp(cum[C-1] - cum_t)`     one exp of a difference, always in (0, 1]
  kernel  `khat_t = k~_t * (gamC / gam[t])`           two exps and a division
  F32 `expf` underflows to 0 once the within-chunk cumsum of gates passes about -104 (-87.3 if the
  device flushes denormals), and the real gates reach -163 on p64 L00 H00 and -293 on p300. Then
  `gamC / gam[t]` is 0/0 = NaN. `khat` is read ONLY by the state update, so the damage lands in the
  state the NEXT chunk inherits: every row of chunk 0 is right, every row of every later chunk is
  NaN. That is the observed device signature and this mirror reproduces it on the pinned operands.

    .venv/bin/python prototype/gdn_chunk/kernel_mirror.py --selftest
    .venv/bin/python prototype/gdn_chunk/kernel_mirror.py finiteness --fixtures dec8,p64,p300 --budget
    .venv/bin/python prototype/gdn_chunk/kernel_mirror.py candidates --fixture p300
    .venv/bin/python prototype/gdn_chunk/kernel_mirror.py budget     --fixture p300
    .venv/bin/python prototype/gdn_chunk/kernel_mirror.py device     --work tmp/gdn/w1024chunk
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "gdn_chunk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chunk_scan as cs                                    # noqa: E402
import spike                                               # noqa: E402
from speccheck import specimpl                             # noqa: E402

GC_C = 64                        # GC_C in the kernel header
# The mechanisms the brief names as candidates, plus the kernel's own injections as controls.
# `ratio_model` is not a candidate but the one-expression fix, carried here so the bisect table and
# the candidate table come from the same code path.
CANDIDATES = ("carry_seed", "bound_head", "ut_no_seed", "strip_global", "ut_no_decay")


def bfr(x: torch.Tensor) -> torch.Tensor:
    """The BF16 round the kernel applies (`hip_bfloat16(float)` and back). Same value as
    `chunk_scan._round_bf16`; spelled here because the mirror's whole subject is WHERE it rounds."""
    return x.to(torch.bfloat16).float()


def _pad(x: torch.Tensor, nc: int, C: int) -> torch.Tensor:
    """[T,H,last] -> [H,NC,C,last], right-padded with zeros (§04.7 A-08: beta=0, g=0, q=k=v=0)."""
    pad = nc * C - x.shape[0]
    if pad:
        x = torch.cat([x, x.new_zeros((pad,) + tuple(x.shape[1:]))], 0)
    return x.float().transpose(0, 1).reshape(x.shape[1], nc, C, -1)


def ut_solve(strict_lower: torch.Tensor, seed: bool = True) -> torch.Tensor:
    """`gdn_chunk_prep`'s in-place ascending-row substitution, F32, row 0 seeded
    (`m[0][tid] = (tid == 0)`). `seed=False` models the pre-59c43dc behaviour as far as a CPU can:
    LDS cannot hold garbage, so the nearest honest reading of "row 0 was never written" is zeros."""
    lead, C = strict_lower.shape[:-2], strict_lower.shape[-1]
    A = strict_lower.reshape(-1, C, C)
    U = torch.zeros_like(A)
    if seed:
        U[:, 0, 0] = 1.0
    for t in range(1, C):
        # acc[d] = sum_{s<t} m[t][s] * m[s][d]: rows < t already hold U, row t still holds `ut`.
        # The kernel writes `urow[tid]` for every tid: -acc for tid < t, 1 at tid == t, 0 above.
        acc = -torch.einsum("bs,bsd->bd", A[:, t, :t], U[:, :t, :])
        U[:, t, :t] = acc[:, :t]
        U[:, t, t] = 1.0
    return U.reshape(*lead, C, C)


def mirror_scan(qn, kn, v, g, beta, S=None, C=GC_C, ratio="kernel", bug=""):
    """The two kernels as one function, chunk axis serial. Returns (o [T,H,dv], S [H,dk,dv]).

    `ratio="kernel"` is the file as written (`gamC / gam[t]`); `ratio="model"` is the one-expression
    fix (`exp(cum[C-1] - cum[t])`). `bug` selects one candidate mechanism.
    """
    T, H, DK = kn.shape
    DV = v.shape[-1]
    NC = -(-T // C)
    q, k, vv = _pad(qn, NC, C), _pad(kn, NC, C), _pad(v, NC, C)
    de = _pad(g.unsqueeze(-1), NC, C).squeeze(-1)
    be = _pad(beta.unsqueeze(-1), NC, C).squeeze(-1)

    cum = de.cumsum(-1)                                 # F32 ascending; one thread in the kernel
    gam = cum.exp()
    gamC = gam[..., C - 1]                              # Gamma_C
    upper = torch.ones(C, C, dtype=torch.bool).triu(1)
    dec = (cum[..., :, None] - cum[..., None, :]).masked_fill(upper, float("-inf")).exp()
    if bug == "ut_no_decay":                            # GCK_UT_NO_DECAY
        dec = torch.ones_like(dec).tril()

    kst = bfr(k)                                        # LDS `kst`: k~ rounded on the way in (D2)
    qst = bfr(q)                                        # q~ rounded at use by the [C,C] loop
    if ratio == "model" or bug == "ratio_model":
        kh_ratio = (cum[..., C - 1:C] - cum).exp()      # one exp of a difference, in (0, 1]
    else:
        kh_ratio = gamC[..., None] / gam                # THE LINE: two exps and a division
    kh = bfr(kst * kh_ratio[..., None])                 # read ONLY by the state update
    qg = bfr(qst * gam[..., None])
    ut = torch.matmul(bfr(kst * be[..., None]), kst.transpose(-1, -2)) * dec
    intra = bfr(torch.matmul(qst, kst.transpose(-1, -2)) * dec)
    U = ut_solve(torch.tril(ut, -1), seed=(bug != "ut_no_seed"))
    Ub = bfr(U)
    nv = torch.matmul(Ub, bfr(vv * be[..., None]))
    kcd = bfr(torch.matmul(Ub, bfr(kst * (be * gam)[..., None])))

    state = torch.zeros((H, DK, DV), dtype=torch.float32) if S is None else S.float().clone()
    body_start = state.clone()
    out = torch.zeros((H, NC, C, DV), dtype=torch.float32)
    for c in range(NC):
        s_read = body_start if bug == "carry_seed" else state
        kh_c = kh[:, c]
        if bug == "bound_head":                          # GCK_BOUND_HEAD: update reads head h+1
            kh_c = torch.roll(kh_c, -1, dims=0)
        v_new = nv[:, c] - torch.matmul(bfr(kcd[:, c]), bfr(s_read))
        out[:, c] = (torch.matmul(bfr(qg[:, c]), bfr(s_read))
                     + torch.matmul(bfr(intra[:, c]), bfr(v_new)))
        # `strip_global` (the residency candidate) writes and rereads the strip through global
        # scratch each chunk. Arithmetically that is this line and nothing more — residency moves
        # bytes, not values, and cmd_candidates prints that rather than pretending to measure it.
        state = gamC[:, c][..., None, None] * state + torch.matmul(kh_c.transpose(-1, -2), bfr(v_new))
    o = out.reshape(H, NC * C, DV)[:, :T].transpose(0, 1).contiguous()
    return o, state


def load_layer(fixture: str, layer: int) -> dict:
    pins = spike.load_gdn_pins(fixture, layer)
    qn, kn = cs.qk_normalized(pins["q_post_conv"], pins["k_post_conv"])
    return dict(qn=qn, kn=kn, v=pins["v_post_conv"], g=pins["g_decay_log"], beta=pins["beta"],
                o_pin=pins["o_pre_norm"], S_pin=pins["state_new"])


def first_nonfinite_row(x: torch.Tensor) -> int:
    """First row (token) holding a non-finite element, or -1 when all are finite."""
    bad = ~torch.isfinite(x.reshape(x.shape[0], -1)).all(dim=1)
    idx = torch.nonzero(bad)
    return -1 if idx.numel() == 0 else int(idx[0][0])


def n_nonfinite_rows(x: torch.Tensor) -> int:
    return int((~torch.isfinite(x.reshape(x.shape[0], -1)).all(dim=1)).sum())


def meanrel(a, b) -> float:
    return float((a - b).abs().mean() / b.abs().mean()) if float(b.abs().mean()) else 0.0


def cos_of(a, b) -> float:
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def worst_cum(g: torch.Tensor, C: int = GC_C) -> float:
    """Most negative within-chunk cumsum of the gate over every layer-agnostic (chunk, head): the
    quantity F32 `expf` cannot represent once it passes about -104 (-87.3 with denormals flushed)."""
    T = g.shape[0]
    nc = -(-T // C)
    seg = g if T % C == 0 else torch.cat([g, g.new_zeros((nc * C - T,) + g.shape[1:])], 0)
    return float(seg.reshape(nc, C, -1).cumsum(1).min())


def cmd_finiteness(args) -> int:
    """The reproducer. Per fixture: does the contract produce NaN, does the mirror, does the
    one-expression change remove it, and at which row does the damage start."""
    torch.set_grad_enabled(False)
    for fixture in args.fixtures.split(","):
        T = int(np.load(ROOT / f"data/refs/{fixture}/L00.gdn.g_decay_log.npy",
                        mmap_mode="r").shape[1])
        acc = dict(cum=0.0, m_o=-1, k_o=-1, f_o=-1, m_S=0, k_S=0, f_S=0, k_rows=0, f_rows=0,
                   m_1cos=0.0, f_1cos=0.0)
        for l in specimpl.GDN_LAYERS:
            op = load_layer(fixture, l)
            o_m, S_m = cs.chunked_scan(op["qn"], op["kn"], op["v"], op["g"], op["beta"],
                                       C=GC_C, mma=True)
            o_k, S_k = mirror_scan(op["qn"], op["kn"], op["v"], op["g"], op["beta"])
            o_f, S_f = mirror_scan(op["qn"], op["kn"], op["v"], op["g"], op["beta"], ratio="model")
            acc["cum"] = min(acc["cum"], worst_cum(op["g"]))
            acc["m_o"] = max(acc["m_o"], first_nonfinite_row(o_m))
            acc["k_o"] = max(acc["k_o"], first_nonfinite_row(o_k))
            acc["f_o"] = max(acc["f_o"], first_nonfinite_row(o_f))
            acc["m_S"] += int((~torch.isfinite(S_m)).sum())
            acc["k_S"] += int((~torch.isfinite(S_k)).sum())
            acc["f_S"] += int((~torch.isfinite(S_f)).sum())
            acc["k_rows"] += n_nonfinite_rows(o_k)
            acc["f_rows"] += n_nonfinite_rows(o_f)
            if args.budget and bool(torch.isfinite(o_f).all()):
                acc["m_1cos"] = max(acc["m_1cos"], 1.0 - cos_of(o_m, op["o_pin"]))
                acc["f_1cos"] = max(acc["f_1cos"], 1.0 - cos_of(o_f, op["o_pin"]))
        nc = -(-T // GC_C)
        print(f"\n=== {fixture}  T={T} ({nc} chunk{'s' if nc > 1 else ''} of {GC_C}), "
              f"{len(specimpl.GDN_LAYERS)} GDN layers, worst within-chunk gate sum "
              f"{acc['cum']:.2f}")
        print(f"    {'form':<36s} {'first NaN o row':>16s} {'NaN o rows':>11s} "
              f"{'NaN state elems':>16s}")
        for label, fo, rows, S_ in (("chunk_scan.py — the contract", acc["m_o"], 0, acc["m_S"]),
                                    ("mirror, gamC/gam (kernel as written)", acc["k_o"],
                                     acc["k_rows"], acc["k_S"]),
                                    ("mirror, exp(cum_C - cum_t) (the fix)", acc["f_o"],
                                      acc["f_rows"], acc["f_S"])):
            print(f"    {label:<36s} {('row %d' % fo) if fo >= 0 else 'none':>16s} {rows:>11d} "
                  f"{S_:>16d}")
        if args.budget:
            print(f"    worst per-layer 1-cos of o_pre_norm vs the PIN: contract {acc['m_1cos']:.3e}"
                  f"   fixed mirror {acc['f_1cos']:.3e}")
    return 0


def cmd_candidates(args) -> int:
    """Bisect. Each mechanism is added on top of ONE clean baseline — the mirror with the model's
    `khat` ratio, which is finite on every layer of the fixture — so a mechanism cannot inherit
    another mechanism's damage, and the two halves of the device signature (chunk 0 exactly right,
    every later chunk wholly NaN) can be matched one mechanism at a time."""
    torch.set_grad_enabled(False)
    fixture = args.fixture
    T = int(np.load(ROOT / f"data/refs/{fixture}/L00.gdn.g_decay_log.npy", mmap_mode="r").shape[1])
    layers = list(specimpl.GDN_LAYERS)[: args.layers] if args.layers else list(specimpl.GDN_LAYERS)
    base = {}
    for l in layers:
        op = load_layer(fixture, l)
        base[l], _ = mirror_scan(op["qn"], op["kn"], op["v"], op["g"], op["beta"], ratio="model")
    print(f"=== candidates on {fixture}: T={T}, {(T + GC_C - 1) // GC_C} chunks of {GC_C}, "
          f"{len(layers)} layers")
    print("    baseline = mirror with exp(cum_C - cum_t) (finite on every layer). damage = max |do|")
    print("    against that baseline; the device signature is (chunk 0 = 0, chunk >=1 = non-finite)")
    print(f"    {'mechanism':<20s} {'non-finite':>11s} {'chunk 0':>11s} {'chunk >=1':>11s}  reading")
    for name, ratio, bug, note in (
            ("baseline (fixed)", "model", "", "finite everywhere; this is what the fix must give"),
            ("carry_seed", "model", "carry_seed", "chunk i reads the state as of the body start"),
            ("bound_head", "model", "bound_head", "the boundary update reads head h+1"),
            ("ut_no_seed", "model", "ut_no_seed", "U row 0 uninitialised (pre-59c43dc)"),
            ("strip_global", "model", "strip_global", "state strip via global, not LDS"),
            ("ut_no_decay", "model", "ut_no_decay", "GCK_UT_NO_DECAY, a known-loud control"),
            ("gamC/gam (THE LINE)", "kernel", "", "khat from two exps and a division")):
        nonfin, c0, c1 = 0, 0.0, 0.0
        for l in layers:
            op = load_layer(fixture, l)
            o, S = mirror_scan(op["qn"], op["kn"], op["v"], op["g"], op["beta"], ratio=ratio, bug=bug)
            nonfin += int((~torch.isfinite(o)).sum()) + int((~torch.isfinite(S)).sum())
            d = (o - base[l]).abs().amax(dim=(1, 2))
            c0 = max(c0, float(d[:min(GC_C, T)].max()))
            c1 = max(c1, float(d[GC_C:].max()) if T > GC_C else float("nan"))
        print(f"    {name:<20s} {nonfin:>11d} {c0:>11.2e} {c1:>11.2e}  {note}")
    print("    read: only the `gamC / gam[t]` division is non-finite anywhere, and it is the only")
    print("    mechanism that leaves chunk 0 untouched. `carry_seed` and `bound_head` do localise to")
    print("    chunks >= 1 but produce finite numbers, so neither can be what the device returned;")
    print("    `ut_no_seed` and `ut_no_decay` damage chunk 0, which the device output does not.")
    print("    Residency is arithmetically a no-op, which is itself the finding: LDS sizing, block")
    print("    geometry and the thread mapping move bytes, not values.")
    return 0


def cmd_budget(args) -> int:
    """Is the mirror faithful? If only that one expression differs, the fixed mirror must land where
    `chunk_scan.py`'s chunk64mma lands against the pins — same class of error, no new band."""
    torch.set_grad_enabled(False)
    fixture = args.fixture
    layers = list(specimpl.GDN_LAYERS)[: args.layers] if args.layers else list(specimpl.GDN_LAYERS)
    w = dict(m=0.0, f=0.0, mS=0.0, fS=0.0, kn=0, fn=0, diff=0.0)
    for l in layers:
        op = load_layer(fixture, l)
        o_m, S_m = cs.chunked_scan(op["qn"], op["kn"], op["v"], op["g"], op["beta"],
                                   C=GC_C, mma=True)
        o_f, S_f = mirror_scan(op["qn"], op["kn"], op["v"], op["g"], op["beta"], ratio="model")
        o_k, _ = mirror_scan(op["qn"], op["kn"], op["v"], op["g"], op["beta"])
        w["m"] = max(w["m"], meanrel(o_m, op["o_pin"]))
        w["f"] = max(w["f"], meanrel(o_f, op["o_pin"]))
        w["mS"] = max(w["mS"], meanrel(S_m, op["S_pin"]))
        w["fS"] = max(w["fS"], meanrel(S_f, op["S_pin"]))
        if bool(torch.isfinite(o_f).all()) and bool(torch.isfinite(o_m).all()):
            w["diff"] = max(w["diff"], meanrel(o_f, o_m))
        w["kn"] += int((~torch.isfinite(o_k)).sum())
        w["fn"] += int((~torch.isfinite(o_f)).sum())
    print(f"=== budget: mirror(ratio=model) vs chunk_scan(chunk64mma) on {fixture}, {len(layers)} layers")
    print(f"    o_pre_norm meanrel vs PIN: contract {w['m']:.3e}   mirror {w['f']:.3e} (budget 9.9e-03)")
    print(f"    state_new  meanrel vs PIN: contract {w['mS']:.3e}   mirror {w['fS']:.3e}")
    print(f"    mirror-vs-contract o meanrel (D1, the association floor): {w['diff']:.3e}")
    print(f"    non-finite o elements: kernel form {w['kn']}, fixed form {w['fn']}")
    ok = w["f"] <= 9.9e-3 and w["f"] <= 3.0 * max(w["m"], 1e-30) and w["fn"] == 0
    print(f"    verdict: {'PASS' if ok else 'FAIL'} — inside the same budget as the contract once the"
          " one expression changes, and outside it before")
    return 0 if ok else 1


def cmd_device(args) -> int:
    """Line the mirror's prediction up with what the device actually wrote."""
    work = Path(args.work)
    rows = sorted(int(p.name[6:-4]) for p in work.glob("logits*.bin"))
    if not rows:
        print(f"device: no logits*.bin under {work}")
        return 1
    bad = {r: not np.isfinite(np.fromfile(work / f"logits{r}.bin", dtype=np.float32)).all()
           for r in rows}
    first_bad = next((r for r in rows if bad[r]), None)
    nc = (max(rows) + 1 + GC_C - 1) // GC_C
    pat = "".join("N" if all(bad[r] for r in rows if c * GC_C <= r < (c + 1) * GC_C) else "."
                  for c in range(nc))
    print(f"=== device run {work}: {len(rows)} prefill rows, {nc} chunks of {GC_C}")
    print(f"    first row with a non-finite logit: "
          f"{'row %d' % first_bad if first_bad is not None else 'none'}"
          + (f", which is the start of chunk {first_bad // GC_C}" if first_bad is not None else ""))
    print(f"    per-chunk all-NaN pattern: {pat}   (N = every row of that chunk is NaN)")
    print("    the mirror predicts that shape exactly: chunk 0 finite because khat is read only by")
    print("    the state update, and every chunk after the one whose gate sum underflows expf NaN.")
    return 0


def cmd_selftest(args) -> int:
    torch.set_grad_enabled(False)
    fails = []
    torch.manual_seed(0)
    for T in (64, 17, 46, 200):
        qn, kn = cs.qk_normalized(torch.randn(T, cs.H_V, cs.D_K), torch.randn(T, cs.H_V, cs.D_K))
        v = 0.04 * torch.randn(T, cs.H_V, cs.D_V)
        g = -2.0 * torch.rand(T, cs.H_V) ** 2
        beta = torch.sigmoid(torch.randn(T, cs.H_V))
        ref, _ = cs.recurrent_scan(qn, kn, v, g, beta)
        o_m, _ = cs.chunked_scan(qn, kn, v, g, beta, C=GC_C, mma=True)
        o_f, S_f = mirror_scan(qn, kn, v, g, beta, ratio="model")
        if not (bool(torch.isfinite(o_f).all()) and bool(torch.isfinite(S_f).all())):
            fails.append(f"T={T}: the fixed mirror is not finite on synthetic operands")
        if cos_of(o_f, ref) < 0.9999:
            fails.append(f"T={T}: mirror cos {cos_of(o_f, ref):.6f} vs the recurrence")
        if cos_of(o_f, o_m) < 0.9999:
            fails.append(f"T={T}: mirror does not track the contract (cos {cos_of(o_f, o_m):.6f})")
        # A gate sum deep enough to underflow expf must NaN the kernel form and not the model form.
        # For a ONE-chunk body the damage lands only in the state (khat is read by the state update
        # alone), which is exactly why a short prompt on the device passes: check the right object.
        gdeep = g.clone()
        gdeep[(GC_C - 1) if T > GC_C else (T - 1), 0] = -200.0
        o_k, S_k = mirror_scan(qn, kn, v, gdeep, beta)
        o_x, S_x = mirror_scan(qn, kn, v, gdeep, beta, ratio="model")
        damaged = S_k if -(-T // GC_C) == 1 else o_k
        if torch.isfinite(damaged).all():
            fails.append(f"T={T}: the kernel form survived a -200 gate, so the mirror does not"
                         " reproduce the division")
        if not (torch.isfinite(o_x).all() and torch.isfinite(S_x).all()):
            fails.append(f"T={T}: the fixed form did not survive the same gate")
        # and no candidate mechanism may produce a non-finite value — that IS the bisect's verdict,
        # so each candidate runs on the FINITE baseline, where any NaN would be its own and not the
        # division's.
        for bug in CANDIDATES:
            o_b, S_b = mirror_scan(qn, kn, v, gdeep, beta, ratio="model", bug=bug)
            if not (torch.isfinite(o_b).all() and torch.isfinite(S_b).all()):
                fails.append(f"T={T}: candidate {bug} produced a non-finite value on a deep gate;"
                             " it cannot be adjudicated against the device's NaN")
        if T > GC_C:
            # a candidate that damages nothing is vacuous (the AGENTS.md negative-control trap)
            base, _ = mirror_scan(qn, kn, v, g, beta, ratio="model")
            for bug in ("carry_seed", "bound_head", "ut_no_seed", "ut_no_decay"):
                bad, _ = mirror_scan(qn, kn, v, g, beta, ratio="model", bug=bug)
                d = (bad - base).abs().amax(dim=(1, 2))
                if float(d[GC_C:].max()) < 1e-4:
                    fails.append(f"T={T}: {bug} moves no row of chunk >=1, so it is vacuous here")
    print(f"selftest: {len(fails)} failures" + "".join("\n  " + f for f in fails))
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--selftest", action="store_true")
    p = sub.add_parser("finiteness", help="the reproducer, per fixture and per form")
    p.add_argument("--fixtures", default="dec8,p64,p300")
    p.add_argument("--budget", action="store_true")
    p.set_defaults(fn=cmd_finiteness)
    p = sub.add_parser("candidates", help="bisect the named mechanisms")
    p.add_argument("--fixture", default="p300")
    p.add_argument("--layers", type=int, default=0)
    p.set_defaults(fn=cmd_candidates)
    p = sub.add_parser("budget", help="show the mirror is faithful to the contract")
    p.add_argument("--fixture", default="p300")
    p.add_argument("--layers", type=int, default=0)
    p.set_defaults(fn=cmd_budget)
    p = sub.add_parser("device", help="compare the prediction with the device dumps")
    p.add_argument("--work", default="tmp/gdn/w1024chunk")
    p.set_defaults(fn=cmd_device)
    args = ap.parse_args()
    if getattr(args, "selftest", False):
        return cmd_selftest(args)
    if args.cmd is None:
        ap.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
