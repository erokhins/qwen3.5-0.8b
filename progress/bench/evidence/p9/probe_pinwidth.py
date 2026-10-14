"""Scratch: how much of the o_pre_norm gap to the pin is the PIN's own BF16 quantization.

`L**.gdn.o_pre_norm.npy` is stored float32 but every element is BF16-representable (verified),
i.e. the reference cast its output back to the activation dtype. So an F32 candidate compared to
it in F32 scores the pin's rounding, not the candidate's error. This prints, per form: meanrel of
the raw F32 candidate, and meanrel/ULP of the candidate rounded to the pin's own width.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "gdn_chunk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bf16_ulp                                    # noqa: E402
import chunk_scan as cs                            # noqa: E402

torch.set_grad_enabled(False)
for fx in ("p64", "p300"):
    for l in (0, 8):
        d = ROOT / "data" / "refs" / fx
        pin = torch.from_numpy(np.load(d / f"L{l:02d}.gdn.o_pre_norm.npy")).float().squeeze(0)
        ops = {k: torch.from_numpy(np.load(d / f"L{l:02d}.gdn.{k}.npy")).float().squeeze(0)
               for k in ("q_post_conv", "k_post_conv", "v_post_conv", "beta", "g_decay_log")}
        qn, kn = cs.qk_normalized(ops["q_post_conv"], ops["k_post_conv"])
        forms = {
            "seq": cs.recurrent_scan(qn, kn, ops["v_post_conv"], ops["g_decay_log"], ops["beta"]),
            "chunk64": cs.chunked_scan(qn, kn, ops["v_post_conv"], ops["g_decay_log"],
                                       ops["beta"], C=64),
            "chunk64mma": cs.chunked_scan(qn, kn, ops["v_post_conv"], ops["g_decay_log"],
                                          ops["beta"], C=64, mma=True),
        }
        print(f"{fx} L{l:02d} pin BF16-valued="
              f"{bool(torch.equal(pin, pin.to(torch.bfloat16).float()))} "
              f"pin max {pin.abs().max():.3e}")
        for name, (o, _S) in forms.items():
            raw = float((o - pin).abs().mean() / pin.abs().mean())
            r = o.to(torch.bfloat16)
            rr = float((r.float() - pin).abs().mean() / pin.abs().mean())
            u = bf16_ulp.ulp_dist(r, pin.to(torch.bfloat16))
            print(f"    {name:11s} raw-vs-pin meanrel {raw:.3e} | bf16-vs-pin meanrel {rr:.3e} "
                  f"ndiff {int((u > 0).sum())}/{u.size} maxULP {int(u.max())}")
