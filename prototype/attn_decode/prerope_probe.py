#!/usr/bin/env python3
"""Why is the device's `q_rope` one element off at position 64? Attribution, CPU-only, no device.

THE OPEN ITEM (now closed; `prototype/attn_decode/results.md` §4, `bench/performance-status.md`
§5(g)): the device kernel reproduces the pinned `attn.q_rope` bit-for-bit at every long-cache position
except `p = 64`, where it is a handful of elements off, every delta exactly 1 BF16 ULP, and the cause
was unattributed. Two candidates were left standing, and the ledger's plan was to dump `z` from the
kernel and diff it:

  A. reduction order — §05.2's mean of squares is a 256-term F32 sum, warp-tree here and torch-order
     in the pins;
  B. the function — the device's `powf`/`cosf`/`sinf` (its rope table) or its `rsqrtf` (the norm)
     disagrees with torch's by one F32 ULP.

No device dump is needed, because the candidates have disjoint signatures and each one's reach is
measurable on the pinned rows:

* **Where each can show up.** `z` is post-norm, pre-rope, and §05.3 passes the dims past `ROT_DIM`
  straight through (`q_rope[d] == z[d]` for d >= 64). Anything that moves `z` — the sum's order, or
  the `rsqrtf` that closes it — moves the **192 pass-through dims** too; a rope-table disagreement
  moves only the **64 rotated dims** of **one position**. The observed residual is inside the rope
  dims at one position and clean in the pass-through dims.
* **How far each reaches.** For every candidate the probe counts the `q_rope` elements that then
  disagree with the pin, in which dims, at how many positions, and asks which candidate reproduces
  the observed signature: elements wrong at p=64, per attention layer,
  {3: 1, 7: 1, 11: 2, 15: 0, 19: 3, 23: 2}, nothing at the other 276 positions. (The original write-up
  in `results.md` §4 said 1/1/2 for layers 3/7/11 because `run.py --long` runs long cases at the
  first layer asked for; the six-layer signature above is `prerope_probe.py --device-root` reading the
  gate's own stage dumps for all six.)

Candidates, each applied with everything else held at the pin:

  A1  the kernel's summation exactly as written in `attn_decode_kernels.h::block_sum` (a
      `__shfl_down_sync` butterfly at offsets 16,8,4,2,1 per warp, then thread 0 adding the 8 warp
      partials in order);
  A2  the other orders a kernel here might use: naive forward/reverse, fold-in-halves, split-K 8x32
      and 32x8, and the correctly rounded sum (`math.fsum`) as the truth the others sit ~1 ULP from;
  B1  one F32 ULP on `r = rsqrtf(mean+eps)` — `rsqrtf` is an estimate-based intrinsic, so this is the
      norm-side half of candidate B;
  B2  one BF16 ULP on ONE rope-table frequency at one position — both slots of it, because §05.3's
      table is the 32 distinct values written twice — which is the class F3b measured on the device
      (2 of 128 `rope_cs` entries differ at p=64, 0 at 67 other sampled positions). The nudge must be
      applied AFTER the table is rounded to BF16: the pins round it there before it multiplies
      anything, so an F32-domain nudge of 1 or 2 ULP is a no-op. (That is why the ±2-F32-ULP search
      recorded in results.md §4 came back empty; it could not have moved anything.)

Attribution only: this file changes no kernel, no band and no gate.

VERDICT, as measured: **the function.** The kernel's own summation moves nothing at p=64; one BF16
step on the rope table's frequency 3 reproduces the whole signature; the device's own dumps say the
pass-through dims (a copy of `z`) are clean at all six layers and its table differs at `cos[3]` and
`cos[35]` by one BF16 step. Written up in `prototype/attn_decode/results.md` §4.

    .venv/bin/python prototype/attn_decode/prerope_probe.py            # the CPU candidate sweep
    .venv/bin/python prototype/attn_decode/prerope_probe.py --device-root /tmp/attn_decode/out_long
        # read the attn gate's own stage dumps for p=64 (after `run.py --long`, one layer per run)
    other flags: [--fixture p300] [--positions all|64] [--layers 3,7,11] [--b2-positions 64] [-v]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from speccheck import specimpl as S                                        # noqa: E402

DH, ROT, HR, N_Q, EPS = 256, 64, 32, 8, 1e-6
ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
OBSERVED = {3: 1, 7: 1, 11: 2, 15: 0, 19: 3, 23: 2}   # elements of q_rope wrong at p=64, per layer


def bf16(x):
    return x.to(torch.bfloat16).float()


def ulp_dir(x, direction):
    return np.nextafter(np.asarray(x, dtype=np.float32), np.float32(direction)).astype(np.float32)


def bf16_ulp(x, n):
    """The n-th adjacent BF16 value of an F32-with-BF16-payload tensor, by bumping the BF16 bit
    pattern. Perturbing a rope table entry in F32 instead is a no-op: the pins round the table to
    BF16 before it multiplies anything, so an F32-ULP nudge never survives that round (which is why
    the ±1..2 ULP search in results.md §4's F3a step came back empty)."""
    bits = x.to(torch.bfloat16).view(torch.int16).clone()
    flat = bits.reshape(-1)
    flat += n
    return flat.view(torch.bfloat16).float().reshape(x.shape)


def warp_tree_sum(sq):
    """`block_sum` from attn_decode_kernels.h, lane 0's result, in F32.

    `sq[..., d]` is thread d's partial (`x*x`, one thread per head dim). A warp covers dims
    [32w, 32w+32) and `warp_sum` runs `v += __shfl_down_sync(mask, v, off)` for off = 16,8,4,2,1;
    every lane reads pre-step values, which is what numpy's evaluate-RHS-first update does. Lane 0
    then holds that warp's partial and thread 0 adds the 8 partials sequentially into `shred`.

    The input is copied: the butterfly is written as in-place updates, and `np.ascontiguousarray`
    would hand back the caller's own array, quietly squaring the wrong numbers afterwards.
    """
    v = np.array(sq, dtype=np.float32, copy=True).reshape(*sq.shape[:-1], 8, 32)
    for off in (16, 8, 4, 2, 1):
        v[..., :32 - off] = (v[..., :32 - off] + v[..., off:32]).astype(np.float32)
    w = v[..., 0]                      # [.., 8]: each warp's lane-0 partial
    acc = w[..., 0].copy()
    for i in range(1, 8):
        acc = (acc + w[..., i]).astype(np.float32)   # thread 0's sequential add of `shred`
    return acc


def order_variants(sq):
    """Alternative F32 summations of the same squares, named by what they are. `sq` is never
    modified: the sequential and butterfly orders are written as running updates on a copy."""
    x = np.array(sq, dtype=np.float32, copy=True)

    def seq(a, rev=False):
        n = a.shape[-1]
        acc = a[..., -1].copy() if rev else a[..., 0].copy()
        for i in (range(n - 2, -1, -1) if rev else range(1, n)):
            acc = (acc + a[..., i]).astype(np.float32)
        return acc

    def fold(a):
        v, n = a.copy(), a.shape[-1]
        while n > 1:
            n //= 2
            v = (v[..., :n] + v[..., n:2 * n]).astype(np.float32)
        return v[..., 0]

    def splitk(groups, inner):
        v = x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups)
        part = seq(v) if inner == "seq" else fold(v)
        acc = part[..., 0].copy()
        for i in range(1, groups):
            acc = (acc + part[..., i]).astype(np.float32)
        return acc

    flat = x.reshape(-1, x.shape[-1])
    exact = np.array([math.fsum(r) for r in flat], dtype=np.float64)
    return {"A1 warp tree (the kernel)": warp_tree_sum(x),
            "A2 naive forward": seq(x),
            "A2 naive reverse": seq(x, rev=True),
            "A2 fold in halves": fold(x),
            "A2 split-K 8 x 32 seq": splitk(8, "seq"),
            "A2 split-K 32 x 8 seq": splitk(32, "seq"),
            "A2 split-K 8 x 32 fold": splitk(8, "fold"),
            "A0 correctly rounded fsum": exact.astype(np.float32).reshape(x.shape[:-1])}


def z_from_mean(mean, x, w):
    """§05.2 after the sum, unrounded: (x * rsqrt(mean+eps)) * (1+w). /256 is exact in binary, so the
    summation order is the only thing a variant can move."""
    return (x * torch.rsqrt(mean + EPS)[..., None]) * (1.0 + w)


def rope(z, cos, sin):
    """§05.3 under the pinned policy: the table entry rounded to BF16 first, each product rounded
    before the add, pairs (i, i+32) over the first ROT dims; dims >= ROT pass through untouched.

    Everything stays F32-with-BF16-payloads so the pass-through dims and the rope dims come back in
    one dtype and can be compared against the pin, which is also F32-with-BF16-payloads.
    """
    c, s = bf16(cos), bf16(sin)
    head = z[..., :ROT]
    rot = torch.cat([-z[..., HR:ROT], z[..., :HR]], -1)
    out = bf16(bf16(head * c) + bf16(rot * s))
    return torch.cat([out, z[..., ROT:]], -1)


def load_layer(w, d, layer):
    q_norm = w.p["model.language_model.layers.%d.self_attn.q_norm.weight" % layer]
    q_raw = torch.from_numpy(np.load(d / ("L%02d.q_proj.out.npy" % layer)))[0].to(torch.bfloat16)
    pin_qr = torch.from_numpy(np.load(d / ("L%02d.attn.q_rope.npy" % layer)))[0]   # [N_Q, T, DH]
    T = q_raw.shape[0]
    return (q_raw.view(T, N_Q, 2 * DH)[..., :DH].contiguous(),
            q_norm.to(torch.bfloat16).float(), pin_qr.permute(1, 0, 2).contiguous(), T)


def device_dump_mode(layers, positions, root) -> int:
    """Read the attn gate's own stage dumps for one long-cache position and say what they prove.

    The gate (`prototype/attn_decode/run.py --long`) writes each case's stages into its own
    directory, so the device's `q_rope` and the device's own `rope_cs` table are both on disk after a
    normal gate run. Three questions, in order:

      1. where does the device's `q_rope` differ from the pin — inside the 64 rope dims, or in the 192
         dims that are a straight copy of `z`? Only the second can be caused by the norm's reduction.
      2. does the device's rope table differ from torch's, at which entries, by how much?
      3. if the CPU path is handed the DEVICE's table, does it now reproduce the device's `q_rope`
         bit-for-bit? If yes, the device's `z` and the pin's `z` agree everywhere the rope does not
         touch, which is the reduction being exonerated by the device's own bits rather than by a
         model of it.
    """
    sys.path.insert(0, str(ROOT / "prototype" / "attn_decode"))
    import reference as R                                                   # noqa: E402
    w = S.Weights(ROOT / "models" / "Qwen3.5-0.8B")
    bad_any = False
    for L in layers:
        p = 64
        case = Path(root) / ("prefill%03d.L%02d" % (p, L))
        if not case.exists():
            print("   %-22s missing (%s)" % (case.name, case))
            bad_any = True
            continue
        dev_q = R.read_out(case, "q_rope", N_Q * DH).reshape(N_Q, DH)
        dev_tab = R.read_out(case, "rope_cs", 2 * ROT)
        d = ROOT / "data" / "refs" / "p300"
        pin_q = torch.from_numpy(np.load(d / ("L%02d.attn.q_rope.npy" % L)))[0][:, p, :].float()
        cos, sin = S.rope_cs(torch.tensor([p]), None)
        tab_torch = torch.cat([bf16(cos[0]), bf16(sin[0])])

        diff = (dev_q != pin_q)
        n_rope, n_pass = int(diff[:, :ROT].sum()), int(diff[:, ROT:].sum())
        tab_diff = (dev_tab != tab_torch)
        n_tab = int(tab_diff.sum())
        where = []
        for i in tab_diff.nonzero().flatten().tolist():
            dv, tv = dev_tab[i].float(), tab_torch[i]
            step = (dv.to(torch.bfloat16).view(torch.int16).item()
                    - tv.to(torch.bfloat16).view(torch.int16).item())
            where.append("%s[%d]: device %.9g, torch %.9g, %d BF16 steps"
                         % ("cos" if i < ROT else "sin", i % ROT, float(dv), float(tv), step))
        # 3. hand the CPU path the device's own table
        q_norm = w.p["model.language_model.layers.%d.self_attn.q_norm.weight" % L]
        q_raw = torch.from_numpy(np.load(d / ("L%02d.q_proj.out.npy" % L)))[0].to(torch.bfloat16)
        q_half = q_raw.view(-1, N_Q, 2 * DH)[p, :, :DH]
        z = bf16(S.rmsnorm_zc(q_half, q_norm).float())
        cpu_with_dev_tab = rope(z, dev_tab[:ROT].float(), dev_tab[ROT:].float())
        n_emul = int((cpu_with_dev_tab != dev_q).sum())
        print("   L%-3d q_rope vs pin: %d wrong (rope dims %d, pass-through %d) | device table: %d"
              " of %d entries differ | CPU z + device table vs device q_rope: %d wrong"
              % (L, n_rope + n_pass, n_rope, n_pass, n_tab, 2 * ROT, n_emul))
        for wd in where:
            print("       %s" % wd)
        bad_any |= bool(n_rope + n_pass)
    return 1 if bad_any else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="p300")
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--positions", default="all")
    ap.add_argument("--layers", default=",".join(str(x) for x in ATTN_LAYERS))
    ap.add_argument("--b2-positions", default="64")
    ap.add_argument("--device-root", default=None,
                    help="instead of the candidate sweep, read the attn gate's stage dumps from this "
                         "directory (e.g. /tmp/attn_decode/out_long) and attribute the p=64 residual "
                         "from the device's own bits")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    d = ROOT / "data" / "refs" / a.fixture
    layers = [int(x) for x in a.layers.split(",")]
    if a.device_root:
        print("device dumps under %s, layers %s, position 64" % (a.device_root, layers))
        return device_dump_mode(layers, [64], Path(a.device_root))
    w = S.Weights(Path(a.model))
    data = {L: load_layer(w, d, L) for L in layers}
    T = data[layers[0]][3]
    pos_all = list(range(T))
    positions = pos_all if a.positions == "all" else [int(x) for x in a.positions.split(",")]
    idx = {p: i for i, p in enumerate(positions)}
    b2_pos = [int(x) for x in a.b2_positions.split(",")] if a.b2_positions != "all" else pos_all
    b2_pos = [p for p in b2_pos if p in idx]
    P = len(positions)
    print("%s: layers %s | %d positions x %d heads x %d dims per layer"
          % (a.fixture, layers, P, N_Q, DH))
    print("device residual to explain (results.md §4): at p=64 -> %s, every delta 1 ULP\n" % OBSERVED)

    tables = {p: tuple(t[0] for t in S.rope_cs(torch.tensor([p]), None)) for p in positions}
    per, names = {}, None
    for L in layers:
        q_half, q_norm, pin, _ = data[L]
        sel = torch.tensor(positions)
        x = q_half[sel].float()
        sq = x * x
        mean0 = sq.mean(-1)                     # before the variants: they are given a copy of sq,
        variants = {k: torch.from_numpy(np.ascontiguousarray(v)) / DH
                    for k, v in order_variants(sq.numpy()).items()}
        names = names or sorted(variants)
        per[L] = dict(x=x, mean=mean0, pin=pin[sel], variants=variants,
                      z=bf16(z_from_mean(mean0, x, q_norm)))

    tot = 0
    for L in layers:
        for i, p in enumerate(positions):
            cos, sin = tables[p]
            got, want = rope(per[L]["z"][i], cos, sin), per[L]["pin"][i]
            if int((got != want).sum()) and a.verbose:
                print("   dbg L%d p%d ndiff %d | z ok vs specimpl: %s | got[0,:3] %s want[0,:3] %s"
                      % (L, p, int((got != want).sum()),
                         torch.equal(per[L]["z"][i],
                                     bf16(S.rmsnorm_zc(data[L][0][p], data[L][1]))),
                         got[0, :3].tolist(), want[0, :3].tolist()))
            tot += int((got != want).sum())
    print("baseline — torch's own mean + torch's table + the pinned policy: %d q_rope elements wrong"
          " over %d comparisons" % (tot, len(layers) * P))
    if tot:
        print("STOP: the oracle path does not reproduce the pin here, so nothing below attributes"
              " anything.")
        return 1

    print("\n== A. re-permuting the reference's own sum of squares ==")
    print("   %-26s %14s %10s %9s %12s %9s"
          % ("summation order", "F32 ULP mean^2", "q_rope off", "in rope", "in pass-thru", "pos off"))
    rows = {}
    for L in layers:
        pl = per[L]
        bits0 = pl["mean"].numpy().astype(np.float32).view(np.int32)
        for name in names:
            mean_v = pl["variants"][name]
            d_ulp = int(np.abs(mean_v.numpy().view(np.int32) - bits0).max())
            z_v = bf16(z_from_mean(mean_v, pl["x"], data[L][1]))
            n_rope = n_pass = n_pos = 0
            where = {}
            for i, p in enumerate(positions):
                cos, sin = tables[p]
                diff = rope(z_v[i], cos, sin) != pl["pin"][i]
                if int(diff.sum()):
                    n_pos += 1
                    nr = int(diff[..., :ROT].sum())
                    npass = int(diff[..., ROT:].sum())
                    n_rope += nr
                    n_pass += npass
                    where[p] = (nr, npass)
            r = rows.setdefault(name, dict(ulp=0, rope=0, passr=0, pos=0, where={}))
            r["ulp"] = max(r["ulp"], d_ulp)
            r["rope"] += n_rope
            r["passr"] += n_pass
            r["pos"] += n_pos
            if where:
                r["where"][L] = where
    for name in names:
        r = rows[name]
        print("   %-26s %14d %10d %9d %12d %9d%s"
              % (name, r["ulp"], r["rope"] + r["passr"], r["rope"], r["passr"], r["pos"],
                 "   <-- layers %s" % sorted(r["where"]) if r["pos"] else ""))
        if a.verbose:
            for L in sorted(r["where"]):
                print("        L%-3d %s" % (L, r["where"][L]))
    a_worst = max(r["rope"] + r["passr"] for r in rows.values())
    a_pass = max(r["passr"] for r in rows.values())
    a_pos = sum(r["pos"] for r in rows.values())
    ker = rows["A1 warp tree (the kernel)"]
    print("   A1, the order the kernel actually uses, per layer at p=64 (rope, pass-through): %s"
          % {L: ker["where"].get(L, {}).get(64, (0, 0)) for L in layers})
    print("   A1 hits at p=64 in %d layer(s); the observation to match is %s"
          % (sum(1 for L in layers if 64 in ker["where"].get(L, {})), OBSERVED))
    print("   %s" % ("every legal order reproduces the pin bit-for-bit at every swept position"
                     if a_worst == 0 else
                     "some order moved %d element(s), %d of them in the pass-through dims, at %d"
                     " positions" % (a_worst, a_pass, a_pos)))

    print("\n== B1. one F32 ULP on r = rsqrtf(mean+eps), on every head of every swept position ==")
    b1 = dict(z=0, rope=0, passr=0, pos=0)
    for L in layers:
        pl = per[L]
        r0 = torch.rsqrt(pl["mean"] + EPS).numpy().astype(np.float32)
        z_v = bf16((pl["x"] * torch.from_numpy(ulp_dir(r0, 1.0))[..., None]) * (1.0 + data[L][1]))
        b1["z"] += int((z_v != pl["z"]).sum())
        for i, p in enumerate(positions):
            cos, sin = tables[p]
            diff = rope(z_v[i], cos, sin) != pl["pin"][i]
            if int(diff.sum()):
                b1["pos"] += 1
                b1["rope"] += int(diff[..., :ROT].sum())
                b1["passr"] += int(diff[..., ROT:].sum())
    print("   r up 1 ULP: z elems moved %d | q_rope off %d (rope %d, pass-thru %d) at %d positions"
          % (b1["z"], b1["rope"] + b1["passr"], b1["rope"], b1["passr"], b1["pos"]))

    print("\n== B2. one BF16 ULP on ONE rope-table frequency (both of its slots), at one position ==")
    print("   §05.3's table is the %d distinct cos/sin values written twice, so one disagreeing" % HR)
    print("   frequency moves TWO of the 128 dumped entries — which is exactly the shape F3b saw on")
    print("   the device (2 of 128 at p=64: cos[3] and cos[35], measured from its own `rope_cs` dump).")
    hits, on64, exact = [], [], []
    for p in b2_pos:
        cos0, sin0 = tables[p]
        i = idx[p]
        for which in ("cos", "sin"):
            for k in range(HR):
                for sign in (+1, -1):
                    cos, sin = cos0.clone(), sin0.clone()
                    # both slots of the frequency: the table is cat([v0..v31, v0..v31])
                    for slot in (k, k + HR):
                        if which == "cos":
                            cos[slot] = bf16_ulp(cos[slot], sign)
                        else:
                            sin[slot] = bf16_ulp(sin[slot], sign)
                    per_layer, nr, np_ = {}, 0, 0
                    for L in layers:
                        pl = per[L]
                        diff = rope(pl["z"][i], cos, sin) != pl["pin"][i]
                        per_layer[L] = int(diff.sum())
                        nr += int(diff[..., :ROT].sum())
                        np_ += int(diff[..., ROT:].sum())
                    if any(per_layer.values()):
                        rec = (p, which, k, sign, per_layer, nr, np_)
                        hits.append(rec)
                        if p == 64:
                            on64.append(rec)
                        if per_layer == OBSERVED and np_ == 0:
                            exact.append(rec)
    print("   %d single-entry perturbations move q_rope anywhere; %d of them sit at p=64"
          % (len(hits), len(on64)))
    print("   %d reproduce the observed per-layer counts %s exactly, nothing in the pass-through dims"
          % (len(exact), OBSERVED))
    for rec in (exact or on64)[:12]:
        print("     p=%s %s[%d] %+d ULP -> per layer %s | rope dims %d, pass-through %d"
              % (rec[0], rec[1], rec[2], rec[3], {L: n for L, n in rec[4].items() if n},
                 rec[5], rec[6]))

    print("\n== verdict material ==")
    a1_64 = {L: sum(ker["where"].get(L, {}).get(64, (0, 0))) for L in layers}
    a1_64 = {L: n for L, n in a1_64.items() if n}
    print("   A  the kernel's own order (A1) at p=64 -> %s ; the observation is %s"
          % (a1_64, OBSERVED))
    print("   A  the worst legal order anywhere moves %d q_rope elements (%d in the pass-through"
          " dims) at %d positions, so a re-associated sum has a real but scattered 1-ULP floor"
          % (a_worst, a_pass, a_pos))
    print("   B1 one ULP of rsqrtf moves %d elements at %d positions (rope %d / pass-thru %d)"
          % (b1["rope"] + b1["passr"], b1["pos"], b1["rope"], b1["passr"]))
    print("   B2 one ULP of one table frequency (both slots) at p=64: %d of the 128 candidates move"
          " something, %d reproduce %s with the pass-through dims clean"
          % (len(on64), len(exact), OBSERVED))
    if a1_64 == OBSERVED:
        print("   VERDICT: REDUCTION ORDER. The kernel's own summation, run on the pinned rows,"
              " reproduces the observed per-layer signature at p=64 exactly.")
    elif exact:
        print("   VERDICT: FUNCTION. Reduction order cannot produce it — the kernel's own order moves"
              " %d elements at p=64 and %d across the whole sweep — while a one-ULP disagreement in"
              " ONE rope-table frequency at p=64 reproduces %s exactly, rope dims only. The device"
              " says so itself under --device-root, where the pass-through dims (a straight copy of"
              " z) are clean at every layer."
              % (sum(a1_64.values()), a_worst, OBSERVED))
    else:
        print("   VERDICT: neither candidate reproduces the signature from the pins alone: A1 gives"
              " %s at p=64, and no single-frequency table nudge gives %s." % (a1_64, OBSERVED))
    print("   (attribution only: no kernel, band or gate was changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
