#!/usr/bin/env python3
"""Per-function gate for prototype/gdn_decode — every kernel mode is checked against the
pinned HF reference dumps before the next one is trusted.

Everything is driven by reference-sourced data, nothing synthesized: the decode step of
the `smoke` fixture at step 1 takes its prior state from step 0's dumps, so

    prior conv window  <- decode0.L00.gdn.conv_state (cols 1..3 = the three taps)
    prior S            <- decode0.L00.gdn.state_new
    new pre-conv raw   <- decode1.L00.gdn.mixed_qkv_pre_conv
    expected outputs   <- decode1.L00.gdn.*  (q/k/v_post_conv, g_decay_log, beta,
                                              o_pre_norm, state_new)

a_raw/b_raw are not dumped, so they are computed here with the reference projections
(§04.1) from the dumped layer input; the script tries the pre-norm and post-norm readings
and keeps whichever reproduces the pinned beta — a wrong choice shows up immediately as a
gate failure rather than as a silent fudge factor.

    .venv/bin/python prototype/gdn_decode/check.py
    .venv/bin/python prototype/gdn_decode/check.py --iters 200 \
        --work-dir /tmp/gdn_in --out-dir /tmp/gdn_out

The fixture this assembles is written to `--work-dir` and left in place, so the raw binary is
runnable against it afterwards — which is what makes the timing rows in
`bench/commit-review-2026-10-11.md` §4 reproducible as written:

    prototype/gdn_decode/gdn_decode --mode all --dir <work-dir> --out <out-dir> --layers 18 \
        --iters 200 --trials 5

`gdn_decode` on its own is not: it reads `q_all.f32`/`S_all.f32`/… and has no fixture of its own, so
`--dir`/`--out` must point at a directory this script filled (`--mode rec` is single-layer by
construction and reads `q.f32`/`S.f32` from the same place).
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "data/refs/smoke"
CKPT = ROOT / "models/Qwen3.5-0.8B"
BIN = Path(__file__).resolve().parent / "gdn_decode"
WORK = Path("/tmp/gdn_in")
OUT = Path("/tmp/gdn_out")

GDN_LAYERS = [l for l in range(24) if l % 4 != 3]      # §00: every 4th layer is full attn
D, NH, CH = 128, 16, 6144
Q_SCALE = 128 ** -0.5


def dump(name: str) -> np.ndarray:
    return np.load(REF / name)


def wkey(key: str) -> torch.Tensor:
    for f in sorted(glob.glob(str(CKPT / "*.safetensors"))):
        with safe_open(f, "pt") as s:
            if key in s.keys():
                return s.get_tensor(key)
    raise KeyError(key)


def bf16(a: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(a)).bfloat16().float().numpy()


def rmsnorm_zc(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    n = xf * torch.rsqrt(torch.tensor(xf).pow(2).mean(-1, keepdim=True) + 1e-6).numpy()
    return (n * (1.0 + w.astype(np.float32))).astype(np.float32)


def spec_recurrence(q, k, v, g, beta, S):
    """numpy mirror of §04.4 in fp32 — the exact math the kernel claims to implement.

    Why a mirror is needed: none, as it turns out — but keep it, it is the tightest possible
    oracle for "did the kernel do the arithmetic in §04.4". The pinned HF `o_pre_norm` looks
    like it disagrees with this block at meanrel 1.45e-3, and that is *not* an accumulation
    difference: HF's recurrent/chunk kernel returns in the input dtype, so the dump is
    BF16-valued stored as fp32. Round this mirror (or the kernel) to bf16 and it is
    **bit-exact** with the dump. Verify with `o_rounded_matches_dump` below before believing
    any claim that this op has an irreducible reference floor.
    """
    qn = q * (1.0 / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)) * Q_SCALE
    kn = k * (1.0 / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6))
    S = S * np.exp(g)[..., None, None]
    r = np.einsum("hkd,hk->hd", S, kn)
    dv = beta[..., None] * (v - r)
    S = S + np.einsum("hk,hd->hkd", kn, dv)
    return np.einsum("hk,hkd->hd", qn, S).astype(np.float32), S.astype(np.float32)


def put(name: str, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr, dtype=np.float32).tofile(WORK / name)


def run(mode: str, layers: int = 18) -> dict:
    proc = subprocess.run(
        [str(BIN), "--mode", mode, "--dir", str(WORK), "--out", str(OUT),
         "--iters", str(ITERS), "--layers", str(layers)],
        capture_output=True, text=True)
    if proc.returncode:
        print(proc.stdout, proc.stderr)
        raise SystemExit("%s failed" % mode)
    fields = {}
    notes = []
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            fields = dict(kv.split("=", 1) for kv in line.split()[1:])
        elif line.startswith("NOTE"):
            notes.append(line)
    fields["_notes"] = notes
    return fields


def bf16_ulp_report(got: np.ndarray, ref: np.ndarray) -> tuple:
    """Distance between `got` (fp32) and the pinned dump, after rounding both to bf16.

    The dumps for this op are BF16-valued (HF's kernel returns in the input dtype) stored as
    fp32, so bit-exactness is the right target. What is left over is fp32 accumulation order:
    measured at 1 element in 2048 on 3 of 18 layers, each exactly 1 ULP - a value sitting on
    the opposite side of a bf16 rounding boundary. Nothing larger has ever appeared, and
    >1 ULP is a real bug, not noise. Diagnostic: ulp_probe.py.
    """
    a = torch.tensor(np.squeeze(got)).to(torch.bfloat16).view(torch.int16).to(torch.int32)
    b = torch.tensor(np.squeeze(ref)).to(torch.bfloat16).view(torch.int16).to(torch.int32)
    # bf16 is sign-magnitude: map to a monotonic integer line before subtracting.
    ma = torch.where(a < 0, -a, a | 0x8000)
    mb = torch.where(b < 0, -b, b | 0x8000)
    d = (ma - mb).abs()
    n_bad = int((d > 1).sum())
    n_diff = int((d != 0).sum())
    return n_bad, n_diff, int(d.max()) if d.numel() else 0, d.numel()


def report(label: str, got: np.ndarray, ref: np.ndarray, tol: float, fails: list) -> None:
    got = np.squeeze(got).astype(np.float64)
    ref = np.squeeze(ref).astype(np.float64)
    cos = float((got.ravel() @ ref.ravel()) /
                (np.linalg.norm(got.ravel()) * np.linalg.norm(ref.ravel()) + 1e-30))
    meanrel = float(np.abs(got - ref).mean() / (np.abs(ref).mean() + 1e-30))
    exact = bool(np.array_equal(got, ref))
    ok = meanrel <= tol
    print("  %-22s max|d| %.6g  meanrel %.3g  cos %.8f  bitexact=%s  %s"
          % (label, np.abs(got - ref).max(), meanrel, cos, exact, "ok" if ok else "FAIL"))
    if not ok:
        fails.append("%s: meanrel %.3g > %.3g" % (label, meanrel, tol))


def main() -> int:
    global ITERS, WORK, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    # The binary takes its fixture by directory; naming the directories here is what lets a doc
    # hand over a raw `gdn_decode …` command that actually runs (see the module docstring).
    ap.add_argument("--work-dir", type=Path, default=WORK,
                    help="where the assembled inputs go — the binary's --dir (default %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help="where kernel output goes — the binary's --out (default %(default)s)")
    args = ap.parse_args()
    ITERS, WORK, OUT = args.iters, args.work_dir, args.out_dir
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        subprocess.run([str(Path(__file__).with_name("build.sh"))], check=True)

    fails: list[str] = []
    prefpref = "L00.gdn."

    # ---------------- F1: S04.3 gates ------------------------------------
    # a_raw/b_raw from §04.1 projections of the layer input; which reading of "the layer
    # input" (raw s_in or its norm) is decided by which one reproduces the pinned beta.
    s_in = np.squeeze(dump("decode1.L00.s_in.npy")).astype(np.float32)
    ln = wkey("model.language_model.layers.0.input_layernorm.weight").float().numpy()
    Wa = wkey("model.language_model.layers.0.linear_attn.in_proj_a.weight").float().numpy()
    Wb = wkey("model.language_model.layers.0.linear_attn.in_proj_b.weight").float().numpy()
    A_log = wkey("model.language_model.layers.0.linear_attn.A_log").float().numpy()
    dt = wkey("model.language_model.layers.0.linear_attn.dt_bias").float().numpy()
    ref_g = np.squeeze(dump("decode1.L00.gdn.g_decay_log.npy"))
    ref_beta = np.squeeze(dump("decode1.L00.gdn.beta.npy"))

    chosen = None
    for label, x in (("norm(s_in)", rmsnorm_zc(s_in, ln)), ("s_in raw", s_in)):
        cand = (x @ Wa.T).astype(np.float32)
        err = float(np.abs((torch.nn.functional.softplus(torch.tensor(cand + dt)) *
                            torch.tensor(A_log.astype(np.float32)).exp()).numpy()
                           - (-ref_g)).max())
        if chosen is None or err < chosen[1]:
            chosen = (label, err, (x @ Wa.T).astype(np.float32), (x @ Wb.T).astype(np.float32))
    label, err, a_raw, b_raw = chosen
    print("F1 input reading: %s (g matches to %.3g)" % (label, err))
    # s_in is dumped for one token, so the projections are already [16]; if a fixture ever
    # dumps a longer prefill, keep the last row.
    last = (lambda t: t[-1] if np.ndim(t) > 1 else t)
    put("a_raw.f32", last(a_raw))
    put("b_raw.f32", last(b_raw))
    put("A_log.f32", A_log)
    put("dt_bias.f32", dt)
    run("gates")
    print("F1 gates (S04.3):")
    # Two levels. (a) unit-exact: the kernel's formula vs torch on the *same* fp32 inputs —
    # this is the check that a wrong softplus/sigmoid/exp would fail. (b) against the pinned
    # dump, which carries the reference's bf16 projection rounding, so it can only be
    # bounded, not matched: a_raw here is fp32 while HF computed a_raw/b_raw with a bf16
    # matmul, and that difference (not the kernel) sets the residual.
    a_last = np.asarray(last(a_raw), dtype=np.float32)
    b_last = np.asarray(last(b_raw), dtype=np.float32)
    gt = (-torch.tensor(A_log).exp() * torch.nn.functional.softplus(
        torch.tensor(a_last) + torch.tensor(dt.astype(np.float32)))).numpy()
    bt = torch.sigmoid(torch.tensor(b_last)).numpy()
    report("g vs torch(unit)", np.fromfile(OUT / "g.f32", dtype=np.float32), gt, 1e-6, fails)
    report("beta vs torch(unit)", np.fromfile(OUT / "beta.f32", dtype=np.float32), bt,
           1e-6, fails)
    report("g vs dump", np.fromfile(OUT / "g.f32", dtype=np.float32), ref_g, 5e-3, fails)
    report("beta vs dump", np.fromfile(OUT / "beta.f32", dtype=np.float32), ref_beta, 5e-3,
           fails)

    # ---------------- F2: S04.2 conv -------------------------------------
    cs0 = dump("decode0.L00.gdn.conv_state.npy")[0]        # [6144,4]
    cs1 = dump("decode1.L00.gdn.conv_state.npy")[0]
    raw1 = np.squeeze(dump("decode1.L00.gdn.mixed_qkv_pre_conv.npy"))  # [6144] (T dropped)
    Cw = wkey("model.language_model.layers.0.linear_attn.conv1d.weight").float().numpy()
    put("raw.f32", raw1)
    put("win.f32", cs0[:, 1:4].T.astype(np.float32))       # three taps, oldest first
    put("convw.f32", Cw.reshape(CH, 4))
    run("conv")
    u = np.fromfile(OUT / "u.f32", dtype=np.float32)
    new_win = np.fromfile(OUT / "new_win.f32", dtype=np.float32).reshape(3, CH)
    print("F2 conv (S04.2):")
    for nm, sl in (("q_post_conv", slice(0, 2048)), ("k_post_conv", slice(2048, 4096)),
                   ("v_post_conv", slice(4096, 6144))):
        report(nm, u[sl].reshape(NH, D), np.squeeze(dump("decode1.L00.gdn.%s.npy" % nm)),
               1e-6, fails)
    report("new conv window", new_win.T, cs1[:, 1:4], 1e-6, fails)

    # ---------------- F3: S04.4 recurrence, one layer --------------------
    q = bf16(np.squeeze(dump("decode1.L00.gdn.q_post_conv.npy")))
    k = bf16(np.squeeze(dump("decode1.L00.gdn.k_post_conv.npy")))
    v = bf16(np.squeeze(dump("decode1.L00.gdn.v_post_conv.npy")))
    g1 = np.squeeze(ref_g).astype(np.float32)
    b1 = np.squeeze(ref_beta).astype(np.float32)
    S0 = dump("decode0.L00.gdn.state_new.npy").astype(np.float32)[0]     # [16,128,128]
    for nm, arr in (("q", q), ("k", k), ("v", v), ("g", g1), ("beta", b1), ("S", S0)):
        n = int(np.isnan(np.asarray(arr)).sum())
        if n:
            fails.append("input %s contains %d NaNs - bad fixture, not a kernel bug" % (nm, n))
    put("q.f32", q); put("k.f32", k); put("v.f32", v)
    put("g.f32", g1); put("beta.f32", b1); put("S.f32", S0)
    run("rec", layers=1)
    o_mine = np.fromfile(OUT / "o.f32", dtype=np.float32).reshape(NH, D)
    S_mine = np.fromfile(OUT / "S_new.f32", dtype=np.float32).reshape(NH, D, D)
    o_ref_np, S_ref_np = spec_recurrence(q, k, v, g1, b1, S0)
    print("F3 recurrence (S04.4), one layer:")
    report("o vs spec mirror", o_mine, o_ref_np, 1e-6, fails)
    report("state vs spec mirror", S_mine, S_ref_np, 1e-6, fails)
    # The real gate against the reference: HF returns this op in bf16, so after the same
    # rounding the only freedom left is fp32 accumulation order, worth at most 1 ULP.
    n_bad, n_diff, mx, nel = bf16_ulp_report(o_mine, dump("decode1.L00.gdn.o_pre_norm.npy"))
    print("  %-22s %d/%d elems differ, max %d ULP  (gate: <=1%% differ, none >1 ULP)"
          % ("o vs HF dump (bf16)", n_diff, nel, mx))
    if n_bad or n_diff > nel // 100:
        fails.append("o vs HF dump: %d elems >1 ULP, %d of %d differ - dump is bf16-valued, "
                     "so >1 ULP is a kernel bug" % (n_bad, n_diff, nel))
    report("state vs HF dump", S_mine, dump("decode1.L00.gdn.state_new.npy")[0], 5e-2, fails)
    r = run("rec", layers=1)
    print("     rec alone: %s ms" % r["ms"])

    # ---------------- F4: all 18 GDN layers ------------------------------
    qs, ks, vs, gs, bs, Ss = [], [], [], [], [], []
    for l in GDN_LAYERS:
        ll = "L%02d" % l
        qs.append(bf16(np.squeeze(dump("decode1.%s.gdn.q_post_conv.npy" % ll)).reshape(NH, D)))
        ks.append(bf16(np.squeeze(dump("decode1.%s.gdn.k_post_conv.npy" % ll)).reshape(NH, D)))
        vs.append(bf16(np.squeeze(dump("decode1.%s.gdn.v_post_conv.npy" % ll)).reshape(NH, D)))
        gs.append(np.squeeze(dump("decode1.%s.gdn.g_decay_log.npy" % ll)).astype(np.float32))
        bs.append(np.squeeze(dump("decode1.%s.gdn.beta.npy" % ll)).astype(np.float32))
        Ss.append(dump("decode0.%s.gdn.state_new.npy" % ll).astype(np.float32)[0])
    put("q_all.f32", np.stack(qs)); put("k_all.f32", np.stack(ks))
    put("v_all.f32", np.stack(vs)); put("g_all.f32", np.stack(gs))
    put("beta_all.f32", np.stack(bs)); put("S_all.f32", np.stack(Ss))
    res = run("all", layers=len(GDN_LAYERS))
    o_all = np.fromfile(OUT / "o.f32", dtype=np.float32).reshape(len(GDN_LAYERS), NH, D)
    S_all = np.fromfile(OUT / "S_new.f32", dtype=np.float32).reshape(len(GDN_LAYERS), NH, D, D)
    print("F4 all %d GDN layers (decode step, worst over layers):" % len(GDN_LAYERS))
    worst_o, worst_S, worst_S_hf = 0.0, 0.0, 0.0
    bad_ulp, diff_layers, max_ulp = 0, 0, 0
    for i, l in enumerate(GDN_LAYERS):
        ll = "L%02d" % l
        ro = np.squeeze(dump("decode1.%s.gdn.o_pre_norm.npy" % ll))
        rs = dump("decode1.%s.gdn.state_new.npy" % ll)[0]
        o_exp, S_exp = spec_recurrence(
            np.stack(qs)[i], np.stack(ks)[i], np.stack(vs)[i], np.stack(gs)[i],
            np.stack(bs)[i], np.stack(Ss)[i])
        mr = lambda a, b: float(np.abs(a - b).mean() / np.abs(b).mean())
        worst_o = max(worst_o, mr(o_all[i], o_exp))
        worst_S = max(worst_S, mr(S_all[i], S_exp))
        nb, nd, mx, nel = bf16_ulp_report(o_all[i], ro)
        bad_ulp += nb
        diff_layers += 1 if nd else 0
        max_ulp = max(max_ulp, mx)
        worst_S_hf = max(worst_S_hf, mr(S_all[i], rs))
    print("  vs spec mirror: worst o %.3g worst state %.3g (tol 1e-5)"
          % (worst_o, worst_S))
    print("  vs HF dump    : o within 1 bf16 ULP on all layers: %s | %d/%d layers have any "
          "1-ULP tie (max %d ULP) | worst state %.3g"
          % (bad_ulp == 0, diff_layers, len(GDN_LAYERS), max_ulp, worst_S_hf))
    if worst_o > 1e-5 or worst_S > 1e-5:
        fails.append("all-layers vs spec mirror: o %.3g state %.3g" % (worst_o, worst_S))
    if bad_ulp:
        fails.append("all-layers: %d elements more than 1 bf16 ULP from the HF dump - the dump "
                     "is bf16-valued, so >1 ULP is a kernel bug" % bad_ulp)
    if worst_S_hf > 5e-2:
        fails.append("all-layers vs HF dump: state %.3g" % worst_S_hf)
    print("  timing %s ms (best of %s trials of %s launches, worst %s, spread %s %%; restore=%s) "
          "over %s MiB of state -> %s GB/s (%s%% of the 237 probe)"
          % (res["ms"], res.get("trials", "?"), res.get("iters", "?"), res.get("worst", "?"),
             res.get("spread_pct", "?"), res.get("restore", "?"),
             res["state_MiB"], res["gbs"], res["pct_of_237"]))
    for note in res.get("_notes", []):
        print("  %s" % note)
    state_ms = float(res["ms"])
    print("  token budget: state traffic alone would cost %.3f ms of the ~6.6 ms roofline token"
          % state_ms)

    print()
    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: all four GDN decode functions match the reference dumps")
    print("\nFixture left assembled in %s (kernel output in %s). The timing rows re-run against it\n"
          "without rebuilding anything — the same three commands the §4 hand-off list carries, and\n"
          "`--mode rec` is one layer whatever --layers says (the binary forces L=1 there):"
          % (WORK, OUT))
    for cmd in ("%s --mode all --dir %s --out %s --layers %d --iters 200 --trials 5"
                % (BIN, WORK, OUT, len(GDN_LAYERS)),
                "%s --mode rec --dir %s --out %s --iters 200 --trials 5" % (BIN, WORK, OUT),
                "%s --mode all --dir %s --out %s --layers %d --iters 1 --trials 20 --restore-each"
                % (BIN, WORK, OUT, len(GDN_LAYERS))):
        print("  " + cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
