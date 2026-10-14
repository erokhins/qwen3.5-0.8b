"""Throwaway probe: what cos floor would pinning logits as BF16 (uint16 bitcast) instead of
FP32 introduce, measured on dec8's own pins?"""
import numpy as np

def bf16_round(x):
    u = x.astype(np.float32).view(np.uint32)
    # round-to-nearest-even on the low 16 bits, the way torch's bf16 cast does it
    r = (u >> 16) & 1
    u = u + 0x7FFF + r
    return (u & 0xFFFF0000).view(np.float32)

for s in (0, 3, 7):
    ref = np.load(f"data/refs/dec8/decode{s}.logits_last.npy").astype(np.float64).reshape(-1)
    got = bf16_round(ref.astype(np.float32)).astype(np.float64)
    cos = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
    meanrel = float(np.mean(np.abs(got - ref)) / np.mean(np.abs(ref)))
    print(f"step {s}: cos {cos:.9f} (1-cos {1-cos:.3e})  meanrel {meanrel:.4e}"
          f"  maxabs {np.max(np.abs(got-ref)):.4g}")
