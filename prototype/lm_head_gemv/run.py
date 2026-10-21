#!/usr/bin/env python3
"""Gate the first real HIP kernel against the reference implementation.

Inputs are taken from the pinned reference dumps and the checkpoint, never synthesized:
  pre_final_hidden.npy  -> S03.1 rmsnorm_zc (speccheck's own, already gate-covered)
                         -> the single normed row that is the lm_head input
  embed_tokens.weight   -> the tied lm_head table (bf16, [248320, 1024], 508.6 MB)
  logits_last.npy       -> ground truth for that row

Checks (both kernel modes):
  full  : argmax equality with the reference, cos >= 0.999 and meanrel <= 0.05 (§10.1
          `logits` class), the top-5 set, AND a per-element rule on the logits themselves —
          `bf16_ulp` distances with a cancellation band, sized by `floor_probe.py` rather than
          typed in. That last part is the closure of `progress/2026-10-11/gate-audit.md` G1: until it existed this
          gate asserted three aggregates over 248 320 values and a dropped term or a misplaced row
          walked straight through.
  fused : the in-kernel argmax must equal the reference argmax, since that mode never
          produces logits at all.
Then it times the same op on torch CPU bf16 so the GPU number has a baseline next to it.

`--selftest` needs no device: it proves the per-element rule discriminates, by feeding it a legal
reassociation (must pass), the row-swap bug and the dropped-term bug (must fail, while still passing
every aggregate the gate used to have).

    .venv/bin/python prototype/lm_head_gemv/run.py [--iters 20]
    .venv/bin/python prototype/lm_head_gemv/run.py --selftest
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
sys.path.insert(0, str(ROOT / "prototype"))          # bf16_ulp, the shared comparator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from speccheck.specimpl import rmsnorm_zc  # noqa: E402
import bf16_ulp as U                                                        # noqa: E402
from floor_probe import bf16_bits, kernel_order                            # noqa: E402

# The per-element rule, DERIVED — `floor_probe.py` recomputes the same 1024 products in eight legal
# addition orders (one of them `lm_head_gemv_full`'s own warp pattern) and scores each against the
# pin with `prototype/bf16_ulp.py`. Over all 248 320 vocabulary rows (measured 2026-10-12):
# every legal order lands at n_bad 0 with ulp_max 1 and cancel_rel 1e-4, and at ndiff <= 0.029 % of
# the rows; the widest cancellation band any legal order needs is 1.01e-05, so 1e-4 carries ten times
# the measured need and the ndiff bound 3.4 times. The device's own logits sit at n_bad 0, ndiff
# 0.023 %, inside the floor rather than under it.
# Narrower would reject a legal kernel. Wider would accept a wrong one: the floor probe scores the
# dropped-term kernel at cos 0.99942 / meanrel 0.0361 with the argmax intact — inside the aggregates
# this gate used to stand on, and at n_bad 212 554 outside this rule.
PER_ELEMENT_ULP_MAX = 1
PER_ELEMENT_CANCEL_REL = 1e-4
PER_ELEMENT_NDIFF_PCT_MAX = 0.10


def per_element_rule(got, ref):
    """(report, violations): the logits compared element by element, not in aggregate.

    `got` is the kernel's F32 logits; the pin holds BF16 values stored as F32, so the candidate is
    rounded through BF16 first — the reference rounds its logits and the kernel does not, and that
    distance is `engine/logits_floor.py`'s business, not a reason to widen a ULP band.

    Four things are asserted, because they fail differently. A ULP bound with a cancellation band
    catches a value that moved where moving matters, and is blind to a whole vector sitting one ULP
    off everywhere; an ndiff bound catches that one; zeros and NaNs are named separately because a
    collapsed output and a merely wrong one can share a ULP count (`bf16_ulp`'s own docstring);
    and the top-5 set is the claim the engine actually spends — every legal order measured keeps it.
    """
    g = bf16_bits(np.asarray(got, dtype=np.float32))
    rep = U.report(g, ref, ulp_max=PER_ELEMENT_ULP_MAX, cancel_rel=PER_ELEMENT_CANCEL_REL)
    zeros, nans = U.zero_nan(g)
    pct = rep["ndiff"] / rep["n"] * 100.0
    top5 = set(np.argsort(-g)[:5].tolist())
    ref_top5 = set(np.argsort(-np.asarray(ref))[:5].tolist())
    bad = []
    if zeros or nans:
        bad.append("%d zeros, %d NaNs in the logits" % (zeros, nans))
    if rep["n_bad"]:
        bad.append("%d elements differ by more than %d ULP outside the %.0e cancellation band "
                   "(worst %d ULP at |ref|/max %.2e)"
                   % (rep["n_bad"], PER_ELEMENT_ULP_MAX, PER_ELEMENT_CANCEL_REL,
                      rep["worst_bad_ulp"], rep["worst_bad_rel"]))
    if pct > PER_ELEMENT_NDIFF_PCT_MAX:
        bad.append("%.3f %% of the vocabulary differs at all (limit %.2f %%, legal floor 0.029 %%)"
                   % (pct, PER_ELEMENT_NDIFF_PCT_MAX))
    if top5 != ref_top5:
        bad.append("top-5 set %s != reference %s" % (sorted(top5), sorted(ref_top5)))
    return rep, bad


def aggregates(got, ref):
    """What the gate asserted before the per-element rule existed, so a selftest can show a bug
    passing them while failing the new check."""
    got = np.asarray(got, dtype=np.float32)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
    meanrel = float(np.abs(got - ref).mean() / np.abs(ref).mean())
    top5 = len(set(np.argsort(-got)[:5].tolist()) & set(np.argsort(-ref)[:5].tolist()))
    return int(np.argmax(got) == int(np.argmax(ref))), cos, meanrel, top5

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


def selftest(W, h, ref) -> int:
    """Does the per-element rule discriminate? CPU-only, no device and no kernel: the clean vector is
    `lm_head_gemv_full`'s own addition order recomputed in F32, which is a legal way to produce these
    logits and must pass. Then each of the two wrong kernels gate-audit G1 names is fed to the same
    rule, and each must fail it while still passing everything this gate asserted beforehand — a
    negative that every check would catch is worth nothing (prototype/mlp_gemv/results.md).
    """
    print("\n== per-element rule selftest (CPU; no device) ==")
    V = ref.size
    legal = np.empty(V, dtype=np.float32)
    dropped = np.empty(V, dtype=np.float32)
    for lo in range(0, V, 4096):
        hi = min(lo + 4096, V)
        P = W[lo:hi].float().numpy() * h[None, :]   # exact BF16 payloads, widened, never rounded
        legal[lo:hi] = kernel_order(P)
        Pd = P.copy()
        Pd[:, -1] = 0.0                     # adding nothing and adding +0.0 are the same F32 op
        dropped[lo:hi] = kernel_order(Pd)
    cases = [("clean: the kernel's own order, on the CPU", True, *per_element_rule(legal, ref), legal)]
    swap = legal.copy()
    order = np.argsort(-ref)
    i, j = int(order[7]), int(order[9])     # two non-argmax rows, inside the top of the table
    swap[i], swap[j] = legal[j], legal[i]
    cases.append(("BUG row %d swapped with row %d (argmax untouched)" % (i, j), False,
                  *per_element_rule(swap, ref), swap))
    cases.append(("BUG last term of every dot missing", False,
                  *per_element_rule(dropped, ref), dropped))

    ok = True
    for name, expect_pass, rep, bad, vec in cases:
        am_ok, cos, meanrel, top5 = aggregates(vec, ref)
        old_ok = am_ok and cos >= 0.999 and meanrel <= 0.05 and top5 == 5
        right = (not bad) == expect_pass
        ok &= right
        print("   %-46s ndiff %7d (%.3f %%) n_bad %6d -> %s %s"
              % (name, rep["ndiff"], rep["ndiff"] / rep["n"] * 100.0, rep["n_bad"],
                 "passes" if not bad else "FAILS",
                 "(as it should)" if right else "<<< WRONG, the rule is not doing its job"))
        for b in bad:
            print("        %s" % b)
        print("        the old aggregates alone: argmax %s, cos %.6f, meanrel %.5f, top5 %d/5 -> %s"
              % ("=" if am_ok else "DIFFERS", cos, meanrel, top5,
                 "would have passed the old gate" if old_ok else "caught even by the aggregates"))
    print("SELFTEST %s" % ("PASS: the rule admits a legal order and rejects both wrong kernels"
                           if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the input row from HF instead of using the cache")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the per-element rule discriminates; CPU-only, no device run")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)

    # 1. the lm_head weight, as raw row-major bf16
    wkey = "model.language_model.embed_tokens.weight"
    W = ckpt_tensor(wkey)
    print("lm_head weight: %s %s (tied to embedding)" % (tuple(W.shape), W.dtype))
    wpath = WORK / "W.bin"
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

    if args.selftest:
        if h is None:
            h = torch.from_numpy(np.fromfile(hpath, dtype=np.float32)).bfloat16()
        return selftest(W, h.float().numpy(), ref)

    # torch refuses to numpy() bf16 directly; the bits are what we want anyway, so view
    # the tensor as uint16 and dump raw. The selftest never needs the file on disk.
    W.contiguous().view(torch.uint16).numpy().tofile(wpath)

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
            rep, bad = per_element_rule(got, ref)
            print("  per element: ndiff %d/%d (%.3f %%) | >%d ULP %d | n_bad %d | worst bad %d ULP at"
                  " |ref|/max %.2e | tightest band this would pass at %.2e"
                  % (rep["ndiff"], rep["n"], rep["ndiff"] / rep["n"] * 100.0,
                     PER_ELEMENT_ULP_MAX, rep["over1"], rep["n_bad"], rep["worst_bad_ulp"],
                     rep["worst_bad_rel"], rep["worst_off_rel"]))
            for b in bad:
                failures.append("full: %s" % b)
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
