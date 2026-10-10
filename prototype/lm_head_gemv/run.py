#!/usr/bin/env python3
"""Gate the first real HIP kernel against the reference implementation.

Inputs are taken from the pinned reference dumps and the checkpoint, never synthesized:
  pre_final_hidden.npy  -> S03.1 rmsnorm_zc (speccheck's own, already gate-covered)
                         -> the single normed row that is the lm_head input
  embed_tokens.weight   -> the tied lm_head table (bf16, [248320, 1024], 508.6 MB)
  logits_last.npy       -> ground truth for that row

Checks (both kernel modes):
  full  : argmax equality with the reference, cos >= 0.999 and meanrel <= 0.05 (§10.1
          `logits` class), and the top-5 set.
  fused : the in-kernel argmax must equal the reference argmax, since that mode never
          produces logits at all.
Then it times the same op on torch CPU bf16 so the GPU number has a baseline next to it.

    .venv/bin/python prototype/lm_head_gemv/run.py [--iters 20]
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from speccheck.specimpl import rmsnorm_zc  # noqa: E402

WORK = Path("/tmp/lmhead")
REF = ROOT / "data/refs/smoke"
CKPT = ROOT / "models/Qwen3.5-0.8B"
BIN = Path(__file__).resolve().parent / "lm_head_gemv"


def ckpt_tensor(key: str) -> torch.Tensor:
    for f in sorted(glob.glob(str(CKPT / "*.safetensors"))):
        with safe_open(f, "pt") as s:
            if key in s.keys():
                return s.get_tensor(key)
    raise KeyError(key)


def normed_row_from_reference() -> torch.Tensor:
    """The lm_head input, straight from the reference implementation.

    Note on the dumps: the *bare* names in data/refs/smoke (input_ids.npy,
    pre_final_hidden.npy) are the values pinned during the last decode step, because the
    recorder writes bare names in every phase; `logits_last.npy` is the prefill's. So
    norming `pre_final_hidden.npy` does NOT reproduce `logits_last.npy` - pairing them is
    the trap this function exists to avoid. Instead run the prefill of `prefill_ids` and
    hook the output of model.language_model.norm, then self-check that the logits implied
    by that row match the pinned ones.
    """
    import json

    from transformers import AutoModelForCausalLM

    manifest = json.loads((REF / "manifest.json").read_text())
    ids = torch.tensor([manifest["prefill_ids"]])
    model = AutoModelForCausalLM.from_pretrained(str(CKPT), dtype=torch.bfloat16,
                                                attn_implementation="eager")
    seen = {}

    def hook(_m, _inp, out):
        seen["normed"] = out.detach()

    # The wrapper nests the text model differently across revisions
    # (model.model.norm vs model.model.language_model.norm), so find it by shape and
    # name instead of hardcoding the path.
    norm_mod = None
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "norm" and "layers" not in name and "visual" not in name \
                and getattr(mod, "weight", None) is not None \
                and mod.weight.shape[0] == 1024:
            norm_mod = mod
    if norm_mod is None:
        raise SystemExit("could not find the final norm module")
    handle = norm_mod.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, -1, :]
    handle.remove()

    ref = torch.from_numpy(np.load(REF / "logits_last.npy")).float().reshape(-1)
    got = logits.float().reshape(-1)
    cos = float(got @ ref / (got.norm() * ref.norm()))
    if int(got.argmax()) != int(ref.argmax()) or cos < 0.999:
        raise SystemExit("fixture self-check failed: HF now disagrees with the pinned "
                         "logits_last (argmax/cos %.6f) - regenerate the dumps" % cos)
    print("reference self-check: HF prefill reproduces logits_last (argmax %d, cos %.6f)"
          % (int(ref.argmax()), cos))
    return seen["normed"][0, -1, :].cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the input row from HF instead of using the cache")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)

    # 1. the lm_head weight, as raw row-major bf16
    wkey = "model.language_model.embed_tokens.weight"
    W = ckpt_tensor(wkey)
    print("lm_head weight: %s %s (tied to embedding)" % (tuple(W.shape), W.dtype))
    wpath = WORK / "W.bin"
    # torch refuses to numpy() bf16 directly; the bits are what we want anyway, so view
    # the tensor as uint16 and dump raw.
    W.contiguous().view(torch.uint16).numpy().tofile(wpath)
    V, H = W.shape

    # 2. the input row: normed final hidden of the prefill, taken from HF itself
    hpath = WORK / "h.bin"
    if "--refresh" in sys.argv:
        h = normed_row_from_reference()
        h.float().numpy().tofile(hpath)
        sys.argv.remove("--refresh")
    elif not hpath.exists():
        h = normed_row_from_reference()
        h.float().numpy().tofile(hpath)
    else:
        h = None
        print("using cached %s (pass --refresh to rebuild from HF)" % hpath)

    ref = np.load(REF / "logits_last.npy").reshape(-1)
    ref_argmax = int(np.argmax(ref))
    ref_top5 = set(np.argsort(-ref)[:5].tolist())

    if not BIN.exists():
        subprocess.run([str(Path(__file__).with_name("build.sh"))], check=True)

    failures = []
    for mode in ("full", "fused"):
        out = WORK / ("out_%s.bin" % mode)
        cmd = [str(BIN), "--w", str(wpath), "--h", str(hpath), "--out", str(out),
               "--mode", mode, "--iters", str(args.iters), "--rows", str(args.rows)]
        print("\n$ " + " ".join(cmd[cmd.index(str(BIN)):]).replace(str(wpath), str(wpath)))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode:
            print(proc.stderr)
            failures.append("%s: binary failed" % mode)
            continue

        gpu_ms = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT"):
                fields = dict(kv.split("=", 1) for kv in line.split()[1:])
                gpu_ms = float(fields["ms"])
        if gpu_ms is None:
            print("no RESULT line from the binary:\n" + proc.stdout)
            failures.append("%s: no RESULT line" % mode)
            continue
        if mode == "full":
            got = np.fromfile(out, dtype=np.float32)
            cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
            meanrel = float(np.abs(got - ref).mean() / np.abs(ref).mean())
            amax = int(np.argmax(got))
            top5 = set(np.argsort(-got)[:5].tolist())
            print("  argmax %d (ref %d) | cos %.6f | max|d| %.4f | meanrel %.5f | "
                  "top5 match %d/5" % (amax, ref_argmax, cos, np.abs(got - ref).max(),
                                       meanrel, len(top5 & ref_top5)))
            if amax != ref_argmax:
                failures.append("full: argmax %d != ref %d" % (amax, ref_argmax))
            if cos < 0.999 or meanrel > 0.05:
                failures.append("full: cos %.6f meanrel %.5f outside §10.1" % (cos, meanrel))
        else:
            val, idx = np.fromfile(out, dtype=np.float32)
            print("  fused argmax %d (%.4f) vs ref %d (%.4f)"
                  % (int(idx), val, ref_argmax, ref[ref_argmax]))
            if int(idx) != ref_argmax:
                failures.append("fused: argmax %d != ref %d" % (int(idx), ref_argmax))

    # baseline: the same op on torch CPU bf16, which is what speccheck does every step
    Wf = W.contiguous()
    if h is None:
        # h.bin holds fp32 values that came from a bf16 tensor, so convert (do not
        # bit-view: viewing fp32 as bf16 doubles the element count).
        h = torch.from_numpy(np.fromfile(hpath, dtype=np.float32)).bfloat16()
    hf = h.unsqueeze(0)
    torch.set_num_threads(16)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        _ = hf @ Wf.T
        best = min(best, time.perf_counter() - t0)
    print("\nbaseline torch CPU bf16 (16 threads): %.1f ms -> %.1f GB/s"
          % (best * 1e3, W.numel() * 2 / best / 1e9))
    print("reference top-1 = %d (%.4f)" % (ref_argmax, ref[ref_argmax]))

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS: bf16 lm_head GEMV matches the reference logits (§10.1 logits class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
