#!/usr/bin/env python3
"""Measure CPU-side memory bandwidth on this box and turn it into a decode ceiling.

Why this exists: §00.5/§07.4 quote ~210-215 GB/s as the platform's attainable
bandwidth, but that is not automatically what a *CPU-resident* kernel can pull, and the
single-stream decode ceiling is `bandwidth / bytes_per_token` — so the two numbers give
very different ceilings (roughly 137 vs 72 tok/s for bf16 Qwen3.5-0.8B). The engine is
supposed to be HIP (GPU-side); until a HIP probe exists, this is the number that bounds
any CPU-resident path, and it is also the number `speccheck` is judged against.

Method: a 6 GiB read-only reduction (far above the 32 MiB L3) is one flop per four
bytes, so it cannot be compute-bound at any plausible bandwidth; a copy is reported too
because it is the read+write traffic a state-updating kernel really pays. `torch.sum`
is a decent but not perfect kernel, so read bandwidth here is a *lower bound* on what a
hand-written kernel can reach.

    .venv/bin/python tools/bw_probe.py --threads 4,8,16
"""
from __future__ import annotations

import argparse
import time

import torch

GiB = 1024 ** 3


def read_bw(nbytes: int, reps: int) -> float:
    a = torch.randn(nbytes // 4, dtype=torch.float32)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.sum(a)
        best = min(best or 1e9, time.perf_counter() - t0)
    return nbytes / best / 1e9, best


def copy_bw(nbytes: int, reps: int) -> float:
    a = torch.randn(nbytes // 4, dtype=torch.float32)
    b = torch.empty_like(a)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        b.copy_(a)
        best = min(best or 1e9, time.perf_counter() - t0)
    return 2 * nbytes / best / 1e9, best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", default="4,8,16")
    ap.add_argument("--size-gib", type=float, default=6.0)
    ap.add_argument("--bytes-per-token", type=float, default=1.565e9,
                    help="decode traffic per token; §07.4 gives 1.55 GB bf16 + state")
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()

    nbytes = int(args.size_gib * GiB)
    print("working set %.1f GiB, decode cost %.3f GB/token"
          % (args.size_gib, args.bytes_per_token / 1e9))
    print("%-8s %14s %14s %12s" % ("threads", "read GB/s", "copy GB/s", "tok/s cap"))
    for t in [int(x) for x in args.threads.split(",")]:
        torch.set_num_threads(t)
        r, rt = read_bw(nbytes, args.reps)
        c, ct = copy_bw(nbytes, args.reps)
        print("%-8d %14.1f %14.1f %12.0f" % (t, r, c, 1e9 * (1 / (args.bytes_per_token)) * r))
    print("\nnote: `tok/s cap` uses read bandwidth, which is what a decode step is; the")
    print("copy column is lower because it pays write traffic as well.")


if __name__ == "__main__":
    main()
