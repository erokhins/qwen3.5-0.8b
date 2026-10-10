#!/usr/bin/env python3
"""S09 — the MTP block used as a speculative drafter, plus the greedy spec loop.

There is no HF reference for this block (`qwen3_5` has no MTP class in transformers),
so everything here is falsified by accept rate rather than by tensor parity:
`gates/run_mtp.py` measures the spec variant against the two plausible-but-wrong
readings of §09 (A-12 pre- vs post-final-norm hidden, A-13 `fc` concat order) and
against a deliberately broken control. A drafter that never agrees with the target is
not "wrong at tensor level", it is simply useless, so the measurement is the gate.

Knobs, resolved by accept-rate measurement in gates/run_mtp.py rather than by reading:
  h_source = "post_norm" (measured 0.642 accept) | "pre_norm" (0.232) — so A-12 as
             originally written was WRONG: the MTP block eats the hidden state the
             lm_head sees, i.e. AFTER the final RMSNorm, not before it.
  fc_order = "eh" [emb ‖ hidden] (0.642) | "he" (0.009) — A-13 confirmed as written.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speccheck.specimpl import (DEFAULT_MODEL, EOS_IDS, Forward, Weights,  # noqa: E402
                                attention, mlp, rmsnorm_zc, rope_cs)

MP = "mtp."


class MTPDraft:
    """One draft step = §09.2. Own KV cache (§09.1), shared embedding table."""

    def __init__(self, w: Weights, h_source: str = "post_norm", fc_order: str = "eh",
                 softmax: str = "full"):
        self.w = w
        self.h_source = h_source
        self.fc_order = fc_order
        self.softmax = softmax
        self.reset()

    def reset(self):
        self.kv: dict = {}
        self.n = 0                      # entries in the MTP cache (= drafted steps)

    def snapshot(self):
        return {"k": list(self.kv.get("k", [])), "v": list(self.kv.get("v", []))}

    def restore(self, snap):
        self.kv = {"k": list(snap["k"]), "v": list(snap["v"])}

    def draft(self, y_next: int, h: torch.Tensor, p: int, dbg: dict | None = None):
        """Draft the token at position `p`. `h` is the hidden state the target's lm_head
        saw (post-final-norm, measured; see the knob list above), `y_next` the token
        just emitted. Returns
        (draft id, x_d) where x_d is this step's output hidden, which §09.3 says
        becomes the next step's `h`."""
        w = self.w
        E = w.p["model.language_model.embed_tokens.weight"]
        e = rmsnorm_zc(E[[y_next]], w.p[MP + "pre_fc_norm_embedding.weight"])
        hh = rmsnorm_zc(h, w.p[MP + "pre_fc_norm_hidden.weight"])
        pair = (torch.cat([e, hh], -1) if self.fc_order == "eh"
                else torch.cat([hh, e], -1))              # A-13: [emb ‖ hidden]
        u = pair @ w.p[MP + "fc.weight"].t()              # [1, 2048] -> [1, 1024]
        pre = MP + "layers.0."
        cos, sin = rope_cs(torch.tensor([p], device=u.device), u.device)
        z = u + attention(w, 0, rmsnorm_zc(u, w.p[pre + "input_layernorm.weight"]),
                          cos, sin, self.kv, None, self.softmax,
                          prefix=pre + "self_attn.")
        z = z + mlp(w, 0, rmsnorm_zc(z, w.p[pre + "post_attention_layernorm.weight"]),
                    prefix=pre + "mlp.")
        x_d = rmsnorm_zc(z, w.p[MP + "norm.weight"])
        self.n += 1
        if dbg is not None:
            dbg["mtp.logits_last"] = (x_d @ E.t())
            dbg["mtp.u"] = u
            dbg["mtp.x_d"] = x_d
        return int((x_d @ E.t())[-1].argmax()), x_d


def _timed(fn, timer: dict | None, key: str):
    """Call `fn`, recording its wall time in `timer[key]` when a timer is given, so a
    bench driver can split prefill from speculation without re-implementing the loop."""
    if timer is None:
        return fn()
    t0 = time.monotonic()
    r = fn()
    timer[key] = time.monotonic() - t0
    return r


def greedy_mtp(f: Forward, m: MTPDraft, ids: list[int], steps: int, k: int = 1,
               stop_on_eos: bool = True, verify: str = "batch",
               timer: dict | None = None, sink=None, resume_from=None):
    """Greedy target + k-token MTP draft per target step (§09.4).

    Invariant at the top of each iteration: `t = out[-1]` is the last emitted token at
    position q and has NOT been fed to the target; the target has consumed through
    position q-1 and `h` is the hidden state at q-1, i.e. the one that predicts q+1
    (the drafter's input pair per §09.1). The verify step then consumes [t, d1..dk]
    at positions q..q+k.

    `verify="batch"` consumes them in one forward - what an engine does, and where the
    saving comes from - but a batched GEMM is not bit-identical to a GEMV in bf16, so
    the target's own verification can disagree with what it would have produced alone
    on a near tie, and greedy speculation stops being bit-lossless (measured: G8b).
    `verify="steps"` feeds them one at a time through the ordinary decode path, which
    is the same arithmetic as plain greedy, so G8a can test positions and rollback
    exactly.

    `sink(tok)` is called for every emitted token in order, which is what lets the chat
    driver stream while speculating; `resume_from=(first_tok, normed_row)` skips the
    prefill for callers that already ran one.
    """
    stats = dict(target_forwards=0, drafted=0, accepted=0, bonus=0, verify_tokens=0)

    def feed(seq):
        """Consume `seq` at the current position, return (last normed hidden, per-position
        target ids)."""
        stats["target_forwards"] += 1
        stats["verify_tokens"] += len(seq)
        if verify == "batch":
            n = f.prefill(torch.tensor(seq))
            return n, [int(x.argmax()) for x in f.logits(n)]
        last, tgt = None, []
        for one in seq:
            last = f.decode(one)
            tgt.append(int(f.logits(last)[-1].argmax()))
        return last, tgt

    if resume_from is None:
        normed = _timed(lambda: f.prefill(torch.tensor(ids)), timer, "prefill_s")
        stats["target_forwards"] += 1
        t = int(f.logits(normed)[-1].argmax())
    else:
        # The caller already prefilled (the chat driver keeps the S08 resume path across
        # turns) and hands over the first sampled token plus the normed hidden row.
        t, normed = resume_from
    out = [t]
    emitted = 0
    if sink is not None:
        sink(t)
    emitted = 1
    h = f.x_final[-1:] if m.h_source == "pre_norm" else normed[-1:]
    while len(out) < steps + 1 and not (stop_on_eos and out[-1] in EOS_IDS):
        q = f.P                                   # position of t (not yet consumed)
        snap, msnap = f.snapshot(), m.snapshot()
        drafts, hd, yd = [], h, t
        for j in range(k):
            d, hd = m.draft(yd, hd, q + 1 + j)
            drafts.append(d)
            yd = d
        stats["drafted"] += k
        normed, tgt = feed([t] + drafts)
        a = 0
        while a < k and drafts[a] == tgt[a]:
            a += 1
        if a < k:                                  # roll back, then re-feed what was
            f.restore(snap)                        # accepted (always including t)
            m.restore(msnap)
            m.n = len(msnap["k"])
            normed, _ = feed([t] + drafts[:a])
        h = f.x_final[-1:] if m.h_source == "pre_norm" else normed[-1:]
        stats["accepted"] += a
        stats["bonus"] += 1
        out.extend(drafts[:a])
        out.append(tgt[a] if a < k else tgt[k])
        if sink is not None:
            for one in out[emitted:]:
                sink(one)
        emitted = len(out)
        t = out[-1]                              # next verify starts from here
    f.generated = out
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--mtp-k", type=int, default=1)
    ap.add_argument("--gdn-prefill", default="chunk", choices=["recurrent", "chunk"])
    ap.add_argument("--h-source", default="pre_norm",
                    choices=["pre_norm", "post_norm"])
    ap.add_argument("--fc-order", default="eh", choices=["eh", "he"])
    ap.add_argument("--compare", action="store_true",
                    help="also run plain greedy and print the speed ratio")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from speccheck.textio import Tokenizer
    w = Weights(Path(args.model))
    tok = Tokenizer(args.model)
    ids = tok.encode(args.prompt)

    f = Forward(w, gdn_form=args.gdn_prefill)
    m = MTPDraft(w, h_source=args.h_source, fc_order=args.fc_order)
    t0 = time.perf_counter()
    out, st = greedy_mtp(f, m, ids, args.steps, k=args.mtp_k)
    dt = time.perf_counter() - t0
    print("mtp draft=%d k=%d h_source=%s fc_order=%s" % (len(out), args.mtp_k,
                                                        args.h_source, args.fc_order))
    print("accept %.3f = %d/%d drafted | %d bonus | %d target forwards | "
          "%.2f tokens per target forward | %.2fs -> %.1f tok/s"
          % (st["accepted"] / max(st["drafted"], 1), st["accepted"], st["drafted"],
             st["bonus"], st["target_forwards"],
             (len(out) - 1) / st["target_forwards"], dt, (len(out) - 1) / dt))
    print(tok.decode(out))

    if args.compare:
        f2 = Forward(w, gdn_form=args.gdn_prefill)
        t0 = time.perf_counter()
        out2 = f2.greedy(ids, len(out) - 1)
        dt2 = time.perf_counter() - t0
        print("\nplain greedy %d tok in %.2fs -> %.1f tok/s | MTP speed ratio %.2fx"
              % (len(out2), dt2, (len(out2) - 1) / dt2,
                 ((len(out) - 1) / dt) / ((len(out2) - 1) / dt2)))


if __name__ == "__main__":
    main()
