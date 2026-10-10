"""How far from bit-exact are the non-matching layers, in bf16 ULPs?

Diagnostic behind the F4 gate: if a layer's `o` differs from the pinned dump after bf16
rounding, the question is whether it is 1 ULP (two fp32 accumulations landing on opposite
sides of a bf16 rounding boundary - benign) or something larger (a real bug). Write to a
file and run it, then delete; the gate itself is check.py.
"""
import numpy as np
import torch

D = "data/refs/smoke/"
GDN = [l for l in range(24) if l % 4 != 3]


def bf(x):
    return torch.tensor(x).to(torch.bfloat16).float().numpy()


def ulpdist(a, b):
    """Distance in bf16 representable steps, using the monotonic sign-magnitude map."""
    ta = torch.tensor(a).to(torch.bfloat16).view(torch.int16).to(torch.int32)
    tb = torch.tensor(b).to(torch.bfloat16).view(torch.int16).to(torch.int32)
    ma = torch.where(ta < 0, -ta, ta | 0x8000).to(torch.int32)
    mb = torch.where(tb < 0, -tb, tb | 0x8000).to(torch.int32)
    return (ma - mb).abs().numpy()


o = np.fromfile("/tmp/gdn_out/o.f32", dtype=np.float32)
bad = 0
for i, l in enumerate(GDN):
    ref = np.squeeze(np.load(D + "decode1.L%02d.gdn.o_pre_norm.npy" % l))
    mine = bf(o[i * 2048:(i + 1) * 2048].reshape(16, 128))
    if not np.array_equal(mine, ref):
        d = ulpdist(mine, ref)
        n = int((d != 0).sum())
        print("L%02d: %d/%d differ (%.2f%%), max ULP %d, meanrel %.3g"
              % (l, n, ref.size, 100.0 * n / ref.size, int(d.max()),
                 np.abs(mine - ref).mean() / np.abs(ref).mean()))
        bad += 1
print("layers differing: %d of %d" % (bad, len(GDN)))
