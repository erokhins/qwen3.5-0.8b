#!/usr/bin/env python3
"""P3 survival study: how many vocab rows can a Cauchy-Schwarz bound provably exclude?

`lm_head` reads the whole 486 MiB tied table every decode step to produce 248,320 logits when greedy
needs only the argmax. For row r the logit is W_r . x, so by Cauchy-Schwarz

    W_r . x  <=  ||W_r|| * ||x||

with equality iff W_r is parallel to x. Walk rows in DESCENDING ||W_r||, keep the running max of the
logits actually computed, and the moment the bound falls below that running max EVERY remaining row is
provably not the argmax - the walk stops and the rest of the table is never read. This is an exact
argument, not a heuristic; the only slack needed is for F32 accumulation order, which is what --slack
models.

x is rebuilt exactly as the engine's final RMSNorm produces it: x = gamma * (h / rms(h)), with h pinned
as `pre_final_hidden.npy` and gamma read from the shard (`model.language_model.norm.weight`). ‖x‖ is
then known, not assumed - and the whole method's validity rests on it being an UPPER bound, so it is
also checked against the pinned logits: max_j(logits_j) must be <= ‖x‖ * max_j‖W_j‖, or the study has
a wrong x or a wrong table and must fail loudly rather than print a cheerful survival rate.

    .venv/bin/python prototype/vocab_prune/survival_study.py [--fixture dec8] [--slack 1.0]

CPU only. Reads pinned dumps and the checkpoint; touches no device.
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen3.5-0.8B"
EMBED = "model.language_model.embed_tokens.weight"
NORMW = [".model.language_model.norm.weight", "model.language_model.norm.weight"]


def shard_tensor(name: str) -> np.ndarray:
    """Read one tensor from the single-shard checkpoint without loading the rest."""
    with safe_open(str(MODEL / "model.safetensors-00001-of-00001.safetensors"),
                   framework="pt") as f:
        keys = set(f.keys())
        cand = [k for k in NORMW if k in keys] if "norm" in name else [name]
        if not cand:
            raise SystemExit(f"no norm-weight key among {NORMW}; sample keys {sorted(keys)[:5]}")
        t = f.get_tensor(cand[0])
        return t.detach().to(torch.float32).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--slack", type=float, default=1.0,
                    help="stop when bound < running_max / slack; >1 buys margin for accumulation order")
    args = ap.parse_args()

    refs = ROOT / "data" / "refs" / args.fixture
    h = np.load(refs / "pre_final_hidden.npy").astype(np.float32).reshape(-1)
    gamma = shard_tensor("norm")
    if gamma.shape != h.shape:
        raise SystemExit(f"gamma rank {gamma.shape} disagrees with pinned pre-norm {h.shape}")
    # Two candidate x's, and the study refuses to guess which one the table multiplies: the witness is
    # reproducing the pinned logits, not the dump's filename. A name-shaped assumption is exactly how
    # the §10.5 wrong-pair trap reproduced real tensors and passed every self-check.
    cands = {"gamma*RMSnorm(pre_final_hidden)": gamma * (h / np.sqrt(np.mean(h * h)))}
    fh = refs / "final_hidden.npy"
    if fh.exists():
        cands["final_hidden (as pinned)"] = np.load(fh).astype(np.float32).reshape(-1)

    with safe_open(str(MODEL / "model.safetensors-00001-of-00001.safetensors"), framework="pt") as f:
        E = f.get_tensor(EMBED)          # bf16; stays bf16, widened in chunks below
    V, D = E.shape
    if D != h.shape[-1]:
        raise SystemExit(f"table width {D} disagrees with hidden {h.shape[-1]}")

    # row norms AND candidate-logit predictions in one chunked pass: bf16.numpy() is unsupported (AGENTS
    # trap), so each chunk is widened in torch and only that chunk is materialised.
    # The bare dump names are last-writer-wins across phases, so pairing must be SEARCHED, not assumed:
    # AGENTS records `pre_final_hidden.npy` + `logits_last.npy` as an invalid pair that reproduces the
    # reference's own tensors and passes every self-check while being cos 0.25 apart. Every candidate x
    # is therefore scored against every pinned logits row and the study proceeds only on a pair that
    # actually reproduces (cos ~ 1).
    logit_files = sorted(refs.glob("*logits_last.npy"))
    probes = {p.name: np.load(p).astype(np.float32).reshape(-1)[-V:] for p in logit_files}
    wc = np.empty(V, dtype=np.float32)
    acc = {k: np.zeros(V, dtype=np.float64) for k in cands}
    step = 32768
    for a in range(0, V, step):
        blk = E[a:a + step].to(torch.float32)
        wc[a:a + step] = blk.pow(2).sum(dim=1).sqrt().numpy()
        n = blk.shape[0]
        for k, xv in cands.items():
            acc[k][a:a + n] = (blk @ torch.from_numpy(xv)).numpy()
    best, chosen, chosen_lg = -1.0, None, None
    for k in cands:
        for lname, lg in probes.items():
            p = acc[k][: lg.size]
            cos = float(np.dot(p, lg) / (np.linalg.norm(p) * np.linalg.norm(lg)))
            if cos > 0.9 or k == chosen:
                print(f"  pair x={k:<34} x {lname:<26} cos={cos:.9f}")
            if cos > best:
                best, chosen, chosen_lg = cos, k, lg
    if best < 0.9999:
        raise SystemExit(f"no (x, logits) pair reproduces the reference (best cos={best:.6f} over "
                         f"{len(cands)}x{len(probes)} pairs); the dumps in this fixture do not contain "
                         "the lm_head input/output pair - do not quote a survival rate")
    print(f"valid pair found: x={chosen} with {chosen_lg} at cos={best:.9f}")
    x = cands[chosen]
    x_norm = float(np.sqrt(np.dot(x, x)))
    order = np.argsort(-wc, kind="stable")

    rows = sorted((refs).glob("decode*.logits_last.npy")) + [refs / "logits_last.npy"]
    print(f"fixture={args.fixture}  vocab={V} hidden={D}  ||x||={x_norm:.6f}  slack={args.slack}")
    print("safety check (must hold for every step): max(logit) <= ||x|| * max(||W||)")
    print(f"{'step':<22} {'max_logit':>10} {'bound_max':>10} {'ok':>4} {'computed':>9} "
          f"{'survival%':>10} {'bytes%':>8} {'lm_head_ms':>10}")
    lm_head_ms = 2.1685          # quoted: engine/generation-limits.md §1 at 4,032 lanes
    per = []
    for p in rows:
        if not p.exists():
            continue
        lg = np.load(p).astype(np.float32).reshape(-1)
        if lg.size != V:
            lg = lg[-V:]
        bound = wc[order] * x_norm
        perm = lg[order]
        runmax = np.maximum.accumulate(perm)
        # first index where every remaining row's bound is below the running max: stop there
        stop_ok = bound < (runmax / args.slack)
        idx = np.argmax(stop_ok) if stop_ok.any() else V - 1
        k = int(idx) + 1
        ok = bool(lg.max() <= bound[0] + 1e-3)
        per.append(k / V)
        print(f"{p.name:<22} {lg.max():>10.3f} {bound[0]:>10.3f} {str(ok):>4} {k:>9} "
              f"{100.0 * k / V:>10.2f} {100.0 * k / V:>8.1f} {lm_head_ms * (1 - k / V):>10.3f}")
        if not ok:
            raise SystemExit("safety check FAILED: the bound is below an observed logit")
    if per:
        worst = max(per)
        print(f"\nworst-case survival over these steps: {100.0 * worst:.2f} % of rows must be computed;")
        print(f"best-case achievable saving on the lm_head line is then "
              f"{lm_head_ms * worst:.3f} ms of a 10.9357 ms step kept, "
              f"i.e. up to {lm_head_ms * (1 - worst):.3f} ms recovered ({100.0 * lm_head_ms * (1 - worst) / 10.9357:.1f} % of the step).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
