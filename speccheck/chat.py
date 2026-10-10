#!/usr/bin/env python3
"""Interactive chat on the speccheck (specification) path, with speed stats.

torch CPU, bf16, greedy — this is the implementation that proves the specification,
NOT the HIP engine, so the tok/s numbers are "is the reference behaving sanely",
not the G9 target. Each turn exercises the S08 resume path: the engine state (GDN
state + conv window + KV) carries over and only the new turn's tokens are prefilled,
so per-turn prefill stays small. `--no-resume` forces a cold full prefill to compare.

Commands: /stats  /reset  /raw <text> (no chat template)  /quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speccheck import textio                                    # noqa: E402
from speccheck.specimpl import (ATTN_LAYERS, GDN_LAYERS, Forward,  # noqa: E402
                                Weights)
from speccheck.textio import Tokenizer                         # noqa: E402

GIB = 1024 ** 3


def state_mib(f: Forward) -> float:
    """Live per-request state (S08): GDN recurrent state + conv windows + KV."""
    n = 0
    for l in GDN_LAYERS:
        for t in (f.rec.get(l), f.conv_win.get(l)):
            if t is not None:
                n += t.numel() * t.element_size()
    for l in ATTN_LAYERS:
        for name in ("k", "v"):
            for t in f.kv[l].get(name, []):
                n += t.numel() * t.element_size()
    return n / GIB * 2 ** 10          # MiB


def rate(n: int, secs: float) -> str:
    return "%.1f tok/s" % (n / secs) if secs > 0 and n > 0 else "-"


def lcp(a: list[int], b: list[int]) -> int:
    """Longest common prefix length; handy for ad-hoc state debugging."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--gdn-prefill", default="chunk", choices=["recurrent", "chunk"])
    ap.add_argument("--attn-softmax", default="full", choices=["full", "online"])
    ap.add_argument("--prompt", default=None, help="answer this and exit (batch use)")
    ap.add_argument("--no-resume", action="store_true",
                    help="force a cold full prefill every turn (compares against the "
                         "S08 resume path)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    t0 = time.time()
    w = Weights(Path(args.model))
    tok = Tokenizer(args.model)
    f = Forward(w, gdn_form=args.gdn_prefill, attn_softmax=args.attn_softmax)
    wbytes = sum(t.numel() * t.element_size() for t in w.p.values())
    print("speccheck chat | torch CPU bf16 greedy | weights %.2f GiB, %d tensors "
          "(%d skipped) | load %.1fs\ngdn=%s attn=%s stop_ids=%s max_new=%d\n"
          "commands: /stats /reset /raw <text> /quit"
          % (wbytes / GIB, len(w.p), len(w.skipped), time.time() - t0,
             args.gdn_prefill, args.attn_softmax, list(textio.EOS_IDS), args.max_new),
          flush=True)

    messages: list[dict] = []
    state_ids: list[int] = []          # ids actually in the engine state
    last_stop = ""
    agg = dict(pre_n=0, pre_t=0.0, dec_n=0, dec_t=0.0, turns=0)

    def turn(text: str, chat: bool = True):
        nonlocal state_ids, last_stop
        t_start = time.time()
        if not chat:
            delta, mode = tok.encode(text), "raw"
        elif not state_ids:
            messages.append({"role": "user", "content": text})
            delta, mode = tok.render_ids(messages), "first"
        else:
            messages.append({"role": "user", "content": text})
            # Build the continuation incrementally instead of re-rendering the whole
            # conversation. The pinned template's generation tail (the empty think
            # block) is not what a re-render of that assistant turn produces, so a
            # from-scratch render is NOT a token extension of what the engine holds -
            # it diverges at the think block and would force a cold prefill every turn.
            # So: close the assistant turn if generation was truncated, then append the
            # next user turn and the generation header. Added tokens are hard BPE
            # boundaries, so encoding this fragment cannot merge across them.
            close = "" if last_stop == "eos" else textio.IM_END + "\n"
            cont = (close + textio.IM_START + "user\n" + text + textio.IM_END + "\n"
                    + textio.IM_START + "assistant\n" + textio.THINK_START + "\n\n"
                    + textio.THINK_END + "\n\n")
            delta, mode = tok.encode(cont), "resume"
        if args.no_resume and state_ids:
            f.reset()
            delta, mode = list(state_ids) + delta, mode + "+cold"
            state_ids = []
        if not delta:
            print("[nothing new to prefill — use /reset if the state looks wrong]")
            messages.pop()
            return
        tt = time.time()
        logits = f.logits(f.prefill(torch.tensor(delta)))
        pre_t = time.time() - tt
        first = int(logits[-1].argmax())

        out: list[int] = []
        stop = "max_new"
        tt = time.time()
        nxt = first
        while True:
            out.append(nxt)
            print(tok.decode([nxt]), end="", flush=True)
            if nxt in textio.EOS_IDS:
                stop = "eos"
                break
            if len(out) >= args.max_new:
                break
            nxt = int(f.logits(f.decode(nxt))[-1].argmax())
        dec_t = time.time() - tt
        state_ids = state_ids + delta + out
        last_stop = stop
        print()
        if chat:
            messages.append({"role": "assistant",
                             "content": tok.tok.decode(out, skip_special_tokens=True)})
        n_dec = max(len(out) - 1, 0)      # first token came from prefill
        # Every decode step re-reads all the weights, so the implied read rate is
        # bytes-per-step x steps / time. That is the number to compare with the
        # ~210-215 GB/s the §07.4 roofline assumes; torch CPU on 8 threads is far
        # below it, which is exactly why the engine is a separate implementation.
        gbs = wbytes * n_dec / dec_t / 1e9 if n_dec > 0 and dec_t > 0 else None
        print("  [prefill %d tok %.2fs %s | decode %d tok %.2fs %s | %s"
              " | stop=%s | %s | prefix %d | state %.1f MiB | turn %.1fs]"
              % (len(delta), pre_t, rate(len(delta), pre_t), n_dec, dec_t,
                 rate(n_dec, dec_t),
                 "%.1f GB/s weight reads" % gbs if gbs else "-",
                 stop, mode, f.P, state_mib(f), time.time() - t_start), flush=True)
        agg.update(pre_n=agg["pre_n"] + len(delta), pre_t=agg["pre_t"] + pre_t,
                   dec_n=agg["dec_n"] + n_dec, dec_t=agg["dec_t"] + dec_t,
                   turns=agg["turns"] + 1)

    if args.prompt:
        turn(args.prompt, chat=not args.prompt.startswith("/raw "))
        return

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/reset":
            f.reset()
            state_ids, messages[:] = [], []
            print("[state reset]")
            continue
        if line == "/stats":
            print("  turns %d | prefill %d tok @ %s | decode %d tok @ %s | "
                  "prefix %d | state %.1f MiB | %.2f GiB weights"
                  % (agg["turns"], agg["pre_n"], rate(agg["pre_n"], agg["pre_t"]),
                     agg["dec_n"], rate(agg["dec_n"], agg["dec_t"]), f.P,
                     state_mib(f), wbytes / GIB))
            continue
        if line.startswith("/raw "):
            turn(line[5:], chat=False)
            continue
        turn(line)

    print("bye — %d turns | prefill %d tok @ %s | decode %d tok @ %s"
          % (agg["turns"], agg["pre_n"], rate(agg["pre_n"], agg["pre_t"]),
             agg["dec_n"], rate(agg["dec_n"], agg["dec_t"])))


if __name__ == "__main__":
    main()
