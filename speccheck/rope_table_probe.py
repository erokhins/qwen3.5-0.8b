#!/usr/bin/env python3
"""Is the pinned rope table reachable from this box? A probe, not a gate.

§05.3 needs cos/sin of `position * theta^(-2i/64)`. The dumps pin the RESULT of applying that
table (`attn.q_rope`), and the pins round the table and each product to BF16 (§05.3), so for every
rope dim the pin is one exact equation:

    q_rope[d] = bf16( bf16(z[d] * c[d]) + bf16(rot[d] * s[d]) ),   rot[d] = -z[d+32] for d < 32

with z = the zero-centred RMSNorm of the pinned q_proj row — known. Eight query heads means eight
independent equations for two unknowns, which is enough to say something stronger than "close":
this script builds every table this machine can produce

  * torch F32      — what speccheck/specimpl.py::rope_cs computes (the oracle's own path)
  * torch F64 -> F32
  * numpy F32
  * numpy F64 -> F32
  * C libm         — cosf/sinf against the same F32 expression, i.e. what the HIP kernel calls
  * the device     — the kernel's own table, read back from prototype/attn_decode's written stages

and reports, per position, how many of the pin's ROT_DIM x N_Q rope elements each one gets wrong.
For a position nothing reproduces, it then searches +/-2 F32 ULP nudges of cos and sin per dim and
reports the offsets that WOULD reproduce the pin — so the answer is "one ULP of F32 in cos at dim d",
not "mismatch".

Read the output as: a position that only the *device* table (or only a nudged table) reproduces is a
reference-generation boundary — the dumps were written by some other libm / theta-power path — and
no rounding policy in a kernel here can recover it. A position where the device table alone fails is
the opposite: our bug. Usage:

    .venv/bin/python speccheck/rope_table_probe.py --fixture p300 --layer 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from speccheck import specimpl as S                                      # noqa: E402

CLIB_SRC = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char **argv) {
    float pos = (float)atof(argv[1]);
    float theta = (float)atof(argv[2]);
    int rot = atoi(argv[3]);
    for (int d = 0; d < rot / 2; d++) {
        float inv = 1.0f / powf(theta, (2.0f * d) / (float)rot);   /* specimpl's expression, in F32 */
        float ang = pos * inv;
        printf("%.9e %.9e\n", cosf(ang), sinf(ang));
    }
    return 0;
}
'''
W = 32   # distinct cos values = distinct sin values = ROT_DIM / 2


def ulp_nudge(x: float, steps: int) -> float:
    """Move an F32 by `steps` ULPs, sign-safely (the bit-order trick bf16_bits uses)."""
    b = int(np.float32(x).view(np.uint32))
    return float(np.uint32(b + (steps if x >= 0 else -steps)).view(np.float32))


def make_tables(pos: int, theta: float, clib, device_tab) -> dict:
    """(cos32, sin32) in F32, one entry per producer."""
    i = torch.arange(W, dtype=torch.float32)
    tabs = {}
    c, s = S.rope_cs(torch.tensor([pos]), None)          # the oracle's own path
    tabs["torch F32"] = (c[0, :W], s[0, :W])
    ang64 = float(pos) * (1.0 / theta ** (2.0 * i.numpy() / S.ROT_DIM))
    ang64 = torch.tensor(ang64, dtype=torch.float64)
    t64 = torch.stack([ang64.cos(), ang64.sin()]).to(torch.float32)
    tabs["torch F64 -> F32"] = (t64[0], t64[1])
    inv32 = np.float32(1.0) / np.float32(theta) ** (np.float32(2.0) * i.numpy() / S.ROT_DIM)
    ang32 = np.float32(pos) * inv32
    tabs["numpy F32"] = (torch.from_numpy(np.cos(ang32)), torch.from_numpy(np.sin(ang32)))
    # The same ANGLE the F32 path uses, but cos/sin evaluated in F64: separates "our angle is one
    # off" from "the transcendental itself disagrees".
    ang_f32_as_f64 = torch.tensor(np.float64(np.float32(pos)) * inv32.astype(np.float64))
    t_ang = torch.stack([ang_f32_as_f64.cos(), ang_f32_as_f64.sin()]).to(torch.float32)
    tabs["F32 angle, F64 cos"] = (t_ang[0], t_ang[1])
    a_np = float(pos) * (1.0 / theta ** (2.0 * i.numpy() / S.ROT_DIM))
    t_np = torch.from_numpy(np.stack([np.cos(a_np), np.sin(a_np)])).to(torch.float32)
    tabs["numpy F64 -> F32"] = (t_np[0], t_np[1])
    if clib is not None:
        tabs["C libm cosf/sinf"] = (torch.tensor(clib[:, 0]), torch.tensor(clib[:, 1]))
    if device_tab is not None:
        # Layout the kernel writes: bf16 cos[ROT_DIM] || sin[ROT_DIM], each half being the ROT_DIM/2
        # distinct values duplicated — the same shape specimpl's rope_cs returns.
        tabs["the device"] = (device_tab[:W].float(), device_tab[S.ROT_DIM:S.ROT_DIM + W].float())
    return tabs


def rope_with(z: torch.Tensor, cos32: torch.Tensor, sin32: torch.Tensor) -> torch.Tensor:
    """S05.3 under the policy the pins demonstrate, on the ROT_DIM leading dims."""
    cb, sb = cos32.to(torch.bfloat16), sin32.to(torch.bfloat16)
    c = torch.cat([cb, cb], -1)[None, :]                              # [1, ROT_DIM]
    s = torch.cat([sb, sb], -1)[None, :]
    zr = z[:, :S.ROT_DIM]
    rot = torch.cat([-z[:, W:S.ROT_DIM], z[:, :W]], -1)               # rotate_half over ROT_DIM
    return zr * c + rot * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="p300")
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--positions", default="", help="comma list; default: every position in the dump")
    ap.add_argument("--verbose", action="store_true", help="print every position, not only the ones that miss")
    ap.add_argument("--device-tables", default="/tmp/attn_decode",
                    help="root of prototype/attn_decode's written stage outputs (may be empty)")
    a = ap.parse_args()

    src = Path("/tmp/ropeclib.c")
    src.write_text(CLIB_SRC)
    subprocess.run(["gcc", "-O2", "-o", "/tmp/ropeclib", str(src), "-lm"], check=True)
    cache: dict[int, np.ndarray] = {}

    def clib(pos):
        if pos not in cache:
            out = subprocess.run(["/tmp/ropeclib", str(float(pos)), str(S.ROPE_THETA),
                                  str(S.ROT_DIM)], capture_output=True, text=True).stdout
            cache[pos] = np.array([[float(r.split()[0]), float(r.split()[1])]
                                   for r in out.strip().splitlines()])
        return cache[pos]

    # The device table, if prototype/attn_decode has been run: <root>/out*/<case>/rope_cs.bin, where
    # a case name carries the position (decodeN is N after dec8's 17-token prefill).
    dev: dict[int, torch.Tensor] = {}
    for f in sorted(Path(a.device_tables).glob("out*/*/rope_cs.bin")) if Path(a.device_tables).exists() else []:
        nm = f.parent.name
        try:
            stem = nm.split(".")[0]
            p = int(stem.replace("decode", "").replace("prefill", "")) \
                + (17 if stem.startswith("decode") else 0)   # dec8's decode steps follow a 17-token prefill
        except ValueError:
            continue
        dev[p] = torch.from_numpy(
            np.frombuffer(f.read_bytes(), dtype=np.uint16)[:2 * S.ROT_DIM].copy()
        ).view(torch.bfloat16)

    d = ROOT / "data/refs" / a.fixture
    ll = "L%02d" % a.layer
    # The dumps are F32 files holding BF16-representable values (§10.5); attention operands are
    # (batch, heads, seq, dim), which is why q_rope is indexed [0][:, pos] and q_proj.out [0].
    q_raw = torch.from_numpy(np.load(d / (ll + ".q_proj.out.npy")))[0].to(torch.bfloat16)
    pin_qr = torch.from_numpy(np.load(d / (ll + ".attn.q_rope.npy")))[0]      # [N_Q, T, D_HEAD]
    q_norm = S.Weights(Path(a.model)).p[
        "model.language_model.layers.%d.self_attn.q_norm.weight" % a.layer]
    T = q_raw.shape[0]
    q_half = q_raw.view(T, S.N_Q, 2 * S.D_HEAD)[..., :S.D_HEAD].contiguous()   # S05.1 per-head q
    z = S.rmsnorm_zc(q_half, q_norm)                                           # [T, N_Q, D_HEAD]
    pos_list = [int(x) for x in a.positions.split(",") if x.strip()] or list(range(T))
    n_want = S.ROT_DIM * S.N_Q

    print("%s layer %d: %d positions, %d rope dims x %d heads each (%d elements per position)"
          % (a.fixture, a.layer, len(pos_list), S.ROT_DIM, S.N_Q, S.ROT_DIM * S.N_Q))
    if dev:
        print("device tables available for %d positions from %s" % (len(dev), a.device_tables))
    else:
        print("no device tables found under %s (run prototype/attn_decode/run.py to get them)"
              % a.device_tables)

    names = set()
    for pos in pos_list:
        names |= set(make_tables(pos, S.ROPE_THETA, clib(pos), dev.get(pos)))
    names = [n for n in ["torch F32", "torch F64 -> F32", "numpy F32", "numpy F64 -> F32",
                         "F32 angle, F64 cos", "C libm cosf/sinf", "the device"] if n in names]
    bad = {n: [] for n in names}
    for pos in pos_list:
        tabs = make_tables(pos, S.ROPE_THETA, clib(pos), dev.get(pos))
        want = pin_qr[:, pos, :][:, :S.ROT_DIM]
        miss = {n: int((rope_with(z[pos], *t) != want).sum()) for n, t in tabs.items()}
        if a.verbose:
            print("  pos %3d  %s" % (pos, "  ".join("%s=%d" % kv for kv in miss.items())))
        for n, m in miss.items():
            if m:
                bad[n].append(pos)
        if max(miss.values()):
            # At least one producer misses. Print every producer's count so it is visible whether
            # the pin is reachable from SOME table (one producer at 0) or from none of them; then
            # ask how far off the best one is, by nudging ONE frequency index's F32 cos or sin.
            best = min(miss, key=miss.get)
            base, few = tabs[best], min(miss.values())
            fixes = []
            for k in range(W):
                for dc in (-2, -1, 0, 1, 2):
                    for ds in (-2, -1, 0, 1, 2):
                        cc, ss = base[0].clone(), base[1].clone()
                        cc[k] = ulp_nudge(float(cc[k]), dc)
                        ss[k] = ulp_nudge(float(ss[k]), ds)
                        n = int((rope_with(z[pos], cc, ss) != want).sum())
                        if n < few:
                            fixes.append((n, k, dc, ds))
            fixes.sort()
            print("  pos %3d  %s   [best: %s]" % (pos, "  ".join("%s=%d" % kv for kv in miss.items()), best))
            for n, k, dc, ds in fixes[:4]:
                print("            %s %d by %+d F32 ULP%s -> %d of %d elements still wrong"
                      % ("cos" if dc else "sin", k, dc or ds, "" if dc or ds else "", n, n_want))
            if not fixes:
                print("            no single frequency index, nudged by up to 2 F32 ULP, helps")
    print("\nper producer, positions with at least one wrong element (of %d sampled):" % len(pos_list))
    for nm in names:
        print("    %-18s %3d of %d%s" % (nm, len(bad[nm]), len(pos_list),
                                         "  " + str(bad[nm][:8]) if bad[nm] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
