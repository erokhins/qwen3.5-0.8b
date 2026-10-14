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
`progress/2026-10-11/commit-review-2026-10-11.md` §4 reproducible as written:

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
D, NH, CH, K = 128, 16, 6144, 4
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


def run_typed(mode: str, work: Path, out: Path, tokens: int, neg: str = "none") -> dict:
    """`run()` with the two P6 switches. Separate helper rather than more defaults on the old one, so
    the decode-step stages above keep their exact command line."""
    cmd = [str(BIN), "--mode", mode, "--dir", str(work), "--out", str(out),
           "--iters", "1", "--tokens", str(tokens)]
    if neg != "none":
        cmd += ["--neg", neg]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        print(proc.stdout, proc.stderr)
        raise SystemExit("%s tokens=%d neg=%s failed" % (mode, tokens, neg))
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return dict(kv.split("=", 1) for kv in line.split()[1:])
    raise SystemExit("no RESULT line from %s" % mode)


def exact(label: str, got: np.ndarray, ref: np.ndarray, fails: list, why: str = "") -> bool:
    """Bit-identity, which is the only honest claim for two launches of the same arithmetic over the
    same bytes. `why` says what is being claimed, because "not bit-identical" is not a diagnosis."""
    same = bool(np.array_equal(got, ref))
    ndiff = int((got != ref).sum())
    print("  %-52s bitexact=%s%s%s" % (label, same, "" if same else "  %d/%d elems differ" % (
        ndiff, got.size), "  " + why if not same and why else ""))
    if not same:
        fails.append("%s: %d/%d elements differ%s" % (label, ndiff, got.size,
                                                      " — " + why if why else ""))
    return same


def moved_rows(a: np.ndarray, b: np.ndarray) -> list:
    """Which ROWS differ — the locality question every negative below answers. A negative that moves
    everything catches itself and proves nothing (`prototype/mlp_gemv/results.md`)."""
    if a.shape != b.shape:
        return [-1]
    return [i for i in range(a.shape[0]) if not np.array_equal(a[i], b[i])]


def tokens_gate(fx: str, nlayers: int, fails: list) -> int:
    """P6: §04.2's conv and §04.3's gates over a [T, CH] tile, against the PREFILL pins.

    The decode-step stages above cannot test any of this: `smoke` is one token per dump, so its conv
    window is always the one the engine rolls by hand and a kernel that ignored the window entirely
    would pass them. A prefill fixture has T tokens per layer, which is what makes four claims testable:

      1. the tile reproduces q/k/v_post_conv for EVERY row, not just the last one (§04.2's causality);
      2. the window it writes is `raw[T-3..T-1]` — compared as CONTENTS, because `u` cannot see what
         the window will be read for (a lagged window produces a correct `u`);
      3. a wave that carries a window reproduces the single-wave result bit for bit (A-04's `T > 1`
         end condition, tested rather than assumed), and zero-padding there instead moves rows 0..2 of
         the second wave and nothing else — the locality proof that the carried window is load-bearing;
      4. `--tokens 1` is bit-identical to the wide launch row by row, which is what licenses keeping
         one kernel for both paths instead of a decode kernel and a prefill kernel.

    Each injected bug is required to appear only where it lives, same rule as the negatives in
    `prototype/mlp_gemv/results.md`.
    """
    global WORK, OUT
    ref = ROOT / "data/refs" / fx
    if not ref.is_dir():
        raise SystemExit("no such prefill fixture: %s" % ref)
    T = int(np.load(ref / "L00.gdn.beta.npy").shape[1])
    # A wave split with a ragged second wave, so the prompt end falls mid-chunk and the carried window
    # has real work to do. 7 and 5 are not divisors of 46 or 277, which is the point: with an even
    # split the last interior boundary would land on a multiple of the wave size and the case most
    # like `p128` (a boundary-aligned prompt end) would be the only one priced.
    splits = [s for s in (T - 7, T - 5, T // 2) if 3 < s < T - 3][:2]
    print("tokens gate on %s: T=%d, layers=%d, wave splits=%s"
          % (fx, T, nlayers, splits))

    for l in GDN_LAYERS[:nlayers]:
        ll = "L%02d" % l
        w = WORK / fx / ll
        o = OUT / fx / ll
        w.mkdir(parents=True, exist_ok=True)
        o.mkdir(parents=True, exist_ok=True)
        raw = np.ascontiguousarray(np.squeeze(np.load(ref / f"{ll}.gdn.mixed_qkv_pre_conv.npy")).T)
        assert raw.shape == (T, CH), (ll, raw.shape)
        z = np.zeros((3, CH), np.float32)     # the request-start window §08.2 specifies
        Cw = wkey(f"model.language_model.layers.{l}.linear_attn.conv1d.weight").float().numpy()
        Cw = np.ascontiguousarray(Cw.reshape(CH, K))
        put_at(w, "raw.f32", raw)
        put_at(w, "win.f32", z)
        put_at(w, "convw.f32", Cw)
        ref_u = np.concatenate([np.squeeze(np.load(ref / f"{ll}.gdn.{n}.npy")).reshape(T, NH * D)
                                for n in ("q_post_conv", "k_post_conv", "v_post_conv")], axis=1)

        # ---- 1. the whole prompt in one wave, window starting at zeros (§08.2's request start) ----
        run_typed("conv", w, o, T)
        u = np.fromfile(o / "u.f32", dtype=np.float32).reshape(T, CH).copy()
        win_new = np.fromfile(o / "new_win.f32", dtype=np.float32).reshape(3, CH).copy()
        print(f"{ll} §04.2 conv, one wave of {T} tokens (window = zeros):")
        for nm, sl in (("q_post_conv", slice(0, 2048)), ("k_post_conv", slice(2048, 4096)),
                       ("v_post_conv", slice(4096, 6144))):
            report(nm, u[:, sl].reshape(T, NH, D),
                   np.squeeze(np.load(ref / f"{ll}.gdn.{nm}.npy")), 1e-6, fails)
        # 2. the window, compared to the pre-conv raw and NOT to anything derived from `u`: a window
        # taken after the conv is the bug, so an `u`-based check could not see it.
        exact("new window == raw[T-3..T-1] (contents)", win_new, raw[-3:], fails,
              "the window the next wave would read is wrong, so the next wave's first three rows"
              " would be computed from the wrong taps (§08.2, A-04)")

        # ---- 3. the same prompt as two waves: the window is the only thing carried ----
        for c in splits:
            put_at(w, "raw.f32", np.ascontiguousarray(raw[:c]))
            put_at(w, "win.f32", z)
            run_typed("conv", w, o, c)
            winA = np.fromfile(o / "new_win.f32", dtype=np.float32).reshape(3, CH).copy()
            put_at(w, "raw.f32", np.ascontiguousarray(raw[c:]))
            put_at(w, "win.f32", winA)
            run_typed("conv", w, o, T - c)
            uB = np.fromfile(o / "u.f32", dtype=np.float32).reshape(T - c, CH).copy()
            winB = np.fromfile(o / "new_win.f32", dtype=np.float32).reshape(3, CH).copy()
            print(f"{ll} two waves, boundary after row {c} (prompt end mid-chunk):")
            exact(f"wave A window == raw[{c-3}:{c}]", winA, raw[c - 3:c], fails,
                  "wave A handed wave B the wrong three taps")
            exact(f"wave B u == one-wave u rows {c}..", uB, u[c:], fails,
                  "the carried window did not reproduce the single-wave conv — A-04's T>1 clause")
            exact("window after two waves == raw[T-3..T-1]", winB, raw[-3:], fails)
            # …and the carry is load-bearing: zero-pad the same second wave and exactly its first
            # three rows move. If this injection changed nothing, the "carry" above would be vacuous.
            put_at(w, "win.f32", winA)
            run_typed("conv", w, o, T - c, neg="zero_pad")
            uBz = np.fromfile(o / "u.f32", dtype=np.float32).reshape(T - c, CH).copy()
            rows = moved_rows(uBz, uB)
            ok = bool(rows) and set(rows) <= {0, 1, 2}
            print(f"  {'':<52} zero_pad injection: rows {rows} moved, want a non-empty subset"
                  f" of [0,1,2]  %s" % ("ok" if ok else "FAIL"))
            if not ok:
                fails.append(f"{ll} split {c}: zero-padding the carried window moved rows {rows}")
        put_at(w, "raw.f32", raw)      # restore for the stages below

        # ---- 4. --tokens 1 must be the launch this op shipped with, row for row ----
        probe = sorted({0, 1, 2, 3, splits[0] - 1, splits[0], splits[0] + 1, T - 2, T - 1})
        bad = []
        for i in probe:
            put_at(w, "raw.f32", np.ascontiguousarray(raw[i:i + 1]))
            # A row before index 3 has fewer than three carried taps and the shortfall is zeros at the
            # FRONT of the window, which is exactly how `win` is indexed (row k of `win` is column
            # k-(K-1-i) of the sequence), so a short prefix pads the front.
            pad = np.zeros((3 - min(i, 3), CH), np.float32)
            put_at(w, "win.f32", np.concatenate([pad, raw[max(i - 3, 0):i]], axis=0)
                   if i else z)
            run_typed("conv", w, o, 1)
            one = np.fromfile(o / "u.f32", dtype=np.float32).reshape(1, CH)
            if not np.array_equal(one, u[i:i + 1]):
                bad.append(i)
        print(f"  %-52s bitexact=%s rows %s of %s probed"
              % ("T=1 launches == the wide launch, row by row", not bad, probe, len(probe)))
        if bad:
            fails.append(f"{ll}: --tokens 1 differs from the tile at rows {bad}")

        # ---- 5. conv negatives, each required to appear only where it lives ----
        put_at(w, "raw.f32", raw)      # the one-row probes above left a one-row file behind
        put_at(w, "win.f32", z)
        # `tap_stride` indexes the tile as if it were still HF's [CH][tokens]. At tokens == 1 the two
        # indexings are the same address, so the bug is invisible on a decode step and only the wide
        # launch can show it — which is the reason this gate exists at all. The two window injections
        # are the opposite case: `u` cannot see either one, so only a direct window comparison can.
        run_typed("conv", w, o, 1, neg="tap_stride")
        one_bad = not np.array_equal(np.fromfile(o / "u.f32", dtype=np.float32).reshape(1, CH),
                                     u[0:1])
        print(f"  {'':<52} tap_stride at tokens=1 is invisible (both indexings are one address)"
              f"  %s" % ("ok" if not one_bad else "FAIL"))
        if one_bad:
            fails.append(f"{ll}: tap_stride moved the tokens=1 launch — the single-token path is"
                         " supposed to be unaffected by the tile indexing bug")
        for neg, claim in (("tap_stride", "every row of the wide launch moves"),
                           ("win_lag", "u unchanged, window moved"),
                           ("win_from_u", "u unchanged, window moved")):
            run_typed("conv", w, o, T, neg=neg)
            un = np.fromfile(o / "u.f32", dtype=np.float32).reshape(T, CH)
            wn = np.fromfile(o / "new_win.f32", dtype=np.float32).reshape(3, CH)
            u_rows, w_same = moved_rows(un, u), np.array_equal(wn, win_new)
            if neg == "tap_stride":
                ok = len(u_rows) == T and w_same
            else:
                ok = not u_rows and not w_same
            print(f"  %-52s %s  u rows moved: %s  window unchanged: %s"
                  % (f"neg {neg} ({claim})", "ok" if ok else "FAIL",
                     f"{len(u_rows)}/{T}" if neg == "tap_stride" else str(u_rows[:6]), w_same))
            if not ok:
                fails.append(f"{ll}: neg {neg} misplaced — u rows {u_rows[:6]} of {T}, window"
                             f" unchanged {w_same}")

        # ---- 6. §04.3 gates over the tile, from the pinned a/b projections ----
        a_raw = np.ascontiguousarray(np.squeeze(np.load(ref / f"{ll}.in_proj_a.out.npy")))
        b_raw = np.ascontiguousarray(np.squeeze(np.load(ref / f"{ll}.in_proj_b.out.npy")))
        A_log = wkey(f"model.language_model.layers.{l}.linear_attn.A_log").float().numpy()
        dt_bias = wkey(f"model.language_model.layers.{l}.linear_attn.dt_bias").float().numpy()
        put_at(w, "a_raw.f32", a_raw)
        put_at(w, "b_raw.f32", b_raw)
        put_at(w, "A_log.f32", A_log)
        put_at(w, "dt_bias.f32", dt_bias)
        run_typed("gates", w, o, T)
        g = np.fromfile(o / "g.f32", dtype=np.float32).reshape(T, NH)
        beta = np.fromfile(o / "beta.f32", dtype=np.float32).reshape(T, NH)
        print(f"{ll} §04.3 gates over {T} tokens:")
        # Two levels, same as F1 above, for the same reason: against the same input bytes the formula
        # is checkable exactly, while the PIN can only be bounded — the reference's `beta` dump is
        # BF16-valued, so the residual there is the dump's rounding and not the kernel's.
        At = torch.tensor(np.asarray(A_log, dtype=np.float32))
        dt = torch.tensor(np.asarray(dt_bias, dtype=np.float32))
        gt = (-At.exp() * torch.nn.functional.softplus(
            torch.tensor(np.asarray(a_raw, dtype=np.float32)) + dt)).numpy()
        bt = torch.sigmoid(torch.tensor(np.asarray(b_raw, dtype=np.float32))).numpy()
        report("g vs torch(unit, same bytes)", g, gt, 1e-6, fails)
        report("beta vs torch(unit, same bytes)", beta, bt, 1e-6, fails)
        report("g vs pin (bounded)", g,
               np.squeeze(np.load(ref / f"{ll}.gdn.g_decay_log.npy")), 5e-3, fails)
        report("beta vs pin (bounded)", beta, np.squeeze(np.load(ref / f"{ll}.gdn.beta.npy")),
               5e-3, fails)
        run_typed("gates", w, o, T, neg="row_zero")
        gz = np.fromfile(o / "g.f32", dtype=np.float32).reshape(T, NH)
        rows = moved_rows(gz, g)
        ok = rows and 0 not in rows
        print(f"  {'':<52} neg row_zero: rows {rows[:4]}… moved, row 0 still exact"
              f"  %s" % ("ok" if ok else "FAIL"))
        if not ok:
            fails.append(f"{ll}: neg row_zero misplaced (rows {rows[:4]})")

    print()
    if fails:
        print("TOKENS GATE FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("TOKENS GATE PASS: the wave conv and gates reproduce the prefill pins row by row, the"
          " carried window is exact across an interior boundary, T=1 is bit-identical to the wide"
          " launch, and every injection stayed where it was put.")
    return 0


def put_at(work: Path, name: str, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr, dtype=np.float32).tofile(work / name)


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
    ap.add_argument("--tokens", metavar="FIXTURE", default="",
                    help="P6: run the WAVE stages against the prefill fixture named (p64 or p300)"
                         " instead of the decode-step stages. Checks §04.2/§04.3 over a [T,6144]"
                         " tile, the carried conv window across a wave boundary, T=1 bit-identity"
                         " against the single-token launch, and that each injected bug appears only"
                         " where it lives")
    ap.add_argument("--layers", type=int, default=2,
                    help="with --tokens: how many GDN layers to price (default 2 — every shipped"
                         " engine fixture already prices all 18 through its logits)")
    args = ap.parse_args()
    ITERS, WORK, OUT = args.iters, args.work_dir, args.out_dir
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        subprocess.run([str(Path(__file__).with_name("build.sh"))], check=True)

    fails: list[str] = []
    if args.tokens:
        return tokens_gate(args.tokens, args.layers, fails)

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
