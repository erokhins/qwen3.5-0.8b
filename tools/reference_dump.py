#!/usr/bin/env python3
"""Reference dump for Qwen3.5-0.8B — the oracle the engine is gated against.

Runs the checkpoint with transformers (eager attention, single sequence, greedy)
and writes every tensor named in specification/10 to data/refs/<tag>/: one .npy
per pinned tensor (always float32) plus manifest.json.

    .venv/bin/python tools/reference_dump.py --prompt "The capital of France is" --decode 2

Flags:
  --model PATH     checkpoint dir                (default models/Qwen3.5-0.8B)
  --out PATH       output dir                    (default data/refs/<tag>)
  --tag NAME       run tag used in the default output dir
  --prompt STR     raw prompt, repeatable        (concatenated as one sequence)
  --message STR    chat message, repeatable      (rendered with the chat template)
  --no-chat        do not append the no-think chat template rendering
  --decode N       greedy continuation steps after prefill (default 1)
  --no-layer-io    omit per-layer input/output dumps
  --dtype          bfloat16 (default, as shipped) | float32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# Special-token literals (built to keep them out of shell/tool quoting).
LT, GT, BAR = chr(60), chr(62), chr(124)
IM_START = LT + BAR + "im_start" + BAR + GT
IM_END = LT + BAR + "im_end" + BAR + GT
ENDOFTEXT = LT + BAR + "endoftext" + BAR + GT
THINK_START = LT + "think" + GT
THINK_END = LT + "/think" + GT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default="default")
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--message", action="append", default=[])
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--decode", type=int, default=1)
    p.add_argument("--no-layer-io", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


class Recorder:
    """Pins tensors by name. Call-order counters map reference calls to layers."""

    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}
        self.meta: dict[str, dict] = {}
        self.step = "prefill"
        self.gdn_i = 0
        self.attn_i = 0

    def pin(self, name: str, t, note: str = ""):
        if t is None:
            return
        if isinstance(t, (list, tuple)):
            for j, item in enumerate(t):
                self.pin(f"{name}[{j}]", item, note)
            return
        if not isinstance(t, torch.Tensor):
            return
        if self.step != "prefill" and name.startswith("L"):
            name = f"{self.step}.{name}"      # keep decode-step pins separate from prefill pins
        self.tensors[name] = t.detach().to(torch.float32).cpu().clone()
        self.meta[name] = {"step": self.step, "note": note,
                           "shape": list(self.tensors[name].shape),
                           "dtype": str(t.dtype).replace("torch.", "")}


def install_patches(M, rec: Recorder, gdn_layers: list[int], attn_layers: list[int]):
    """Wrap the reference helpers so every call records its operands.

    Calls arrive in ascending layer order inside one forward, so a per-forward
    counter is enough to name them.
    """
    orig = {
        "chunk": M.torch_chunk_gated_delta_rule,
        "recur": M.torch_recurrent_gated_delta_rule,
        "conv_fn": M.causal_conv1d_fn,
        "conv_upd": M.causal_conv1d_update,
        "norm_gated": M.Qwen3_5RMSNormGated.forward,
        "rope": M.apply_rotary_pos_emb,
        "eager": M.eager_attention_forward,
    }

    def chunk(q, k, v, g, beta, **kw):
        i = gdn_layers[rec.gdn_i]
        rec.gdn_i += 1
        out, state = orig["chunk"](q, k, v, g, beta, **kw)
        p = f"L{i:02d}.gdn"
        rec.pin(f"{p}.q_post_conv", q, "post-conv SiLU q (pre l2norm/scale)")
        rec.pin(f"{p}.k_post_conv", k, "post-conv SiLU k (pre l2norm)")
        rec.pin(f"{p}.v_post_conv", v, "post-conv SiLU v")
        rec.pin(f"{p}.g_decay_log", g, "g = -exp(A_log)*softplus(a+dt_bias)")
        rec.pin(f"{p}.beta", beta, "beta = sigmoid(b_raw)")
        rec.pin(f"{p}.o_pre_norm", out, "recurrent output before gated RMSNorm")
        rec.pin(f"{p}.state_new", state, "S after this segment")
        return out, state

    def recur(q, k, v, g, beta, **kw):
        i = gdn_layers[rec.gdn_i]
        rec.gdn_i += 1
        out, state = orig["recur"](q, k, v, g, beta, **kw)
        p = f"L{i:02d}.gdn"
        rec.pin(f"{p}.q_post_conv", q, "decode step")
        rec.pin(f"{p}.k_post_conv", k, "decode step")
        rec.pin(f"{p}.v_post_conv", v, "decode step")
        rec.pin(f"{p}.g_decay_log", g, "decode step")
        rec.pin(f"{p}.beta", beta, "decode step")
        rec.pin(f"{p}.o_pre_norm", out, "decode step")
        rec.pin(f"{p}.state_new", state, "decode step")
        return out, state

    def conv_fn(x, weight, bias=None, activation=None, **kw):
        i = gdn_layers[rec.gdn_i]
        rec.pin(f"L{i:02d}.gdn.mixed_qkv_pre_conv", x, "in_proj_qkv output, channels-first")
        return orig["conv_fn"](x, weight, bias, activation, **kw)

    def conv_upd(x, conv_state, weight, bias=None, activation=None):
        i = gdn_layers[rec.gdn_i]
        rec.pin(f"L{i:02d}.gdn.mixed_qkv_pre_conv", x, "decode: 1-token qkv")
        out = orig["conv_upd"](x, conv_state, weight, bias, activation)
        rec.pin(f"L{i:02d}.gdn.conv_state", conv_state, "decode: conv state after shift (last k_conv-1 cols)")
        return out

    def norm_gated(self, hidden_states, gate):
        out = orig["norm_gated"](self, hidden_states, gate)
        # named by the active GDN layer: gdn_i already advanced past the rule call
        i = gdn_layers[max(rec.gdn_i - 1, 0)]
        rec.pin(f"L{i:02d}.gdn.gated_out", out, "RMSNormGated(o, z) before out_proj")
        return out

    def rope(q, k, cos, sin, unsqueeze_dim=1):
        out_q, out_k = orig["rope"](q, k, cos, sin, unsqueeze_dim)
        i = attn_layers[rec.attn_i]
        rec.pin(f"L{i:02d}.attn.q_rope", out_q, "post q_norm + partial rope")
        rec.pin(f"L{i:02d}.attn.k_rope", out_k, "post k_norm + partial rope")
        rec.attn_i += 1
        return out_q, out_k

    def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        i = attn_layers[max(rec.attn_i - 1, 0)]
        out, weights = orig["eager"](module, query, key, value, attention_mask, scaling, dropout, **kw)
        rec.pin(f"L{i:02d}.attn.v", value, "value states (pre repeat_kv)")
        rec.pin(f"L{i:02d}.attn.attn_out_pre_gate", out, "attention output before sigmoid gate")
        return out, weights

    M.torch_chunk_gated_delta_rule = chunk
    M.torch_recurrent_gated_delta_rule = recur
    M.causal_conv1d_fn = conv_fn
    M.causal_conv1d_update = conv_upd
    M.Qwen3_5RMSNormGated.forward = norm_gated
    M.apply_rotary_pos_emb = rope
    M.eager_attention_forward = eager
    return orig


def get_text_model(model):
    m = model.model if hasattr(model, "model") else model
    for _ in range(3):
        if hasattr(m, "layers"):
            return m
        nxt = getattr(m, "language_model", None) or getattr(m, "model", None)
        if nxt is None:
            break
        m = nxt
    raise RuntimeError("text model with .layers not found")


def dump_cache(cache, rec: Recorder, tag: str):
    """Best-effort: pin recurrent/conv states and KV as stored in the cache."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        return
    for i, layer in enumerate(layers):
        for attr in ("recurrent_states", "conv_states"):
            vals = getattr(layer, attr, None)
            if vals:
                for j, v in enumerate(vals):
                    rec.pin(f"{tag}.L{i:02d}.{attr}[{j}]", v, f"cache {attr} after step")
        for attr in ("keys", "values"):
            v = getattr(layer, attr, None)
            if isinstance(v, torch.Tensor) and v.numel():
                rec.pin(f"{tag}.L{i:02d}.cache_{attr}", v, "kv cache after step")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out) if args.out else ROOT / "data" / "refs" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager")
    model.eval()
    tm = get_text_model(model)
    layers = list(tm.layers)
    gdn_layers = [i for i, l in enumerate(layers) if hasattr(l, "linear_attn")]
    attn_layers = [i for i, l in enumerate(layers) if hasattr(l, "self_attn")]
    print(f"loaded in {time.time()-t0:.1f}s | {len(layers)} layers | gdn {gdn_layers} | attn {attn_layers}")

    rec = Recorder()
    install_patches(M, rec, gdn_layers, attn_layers)

    def reset_counters(_m, _inp):
        rec.gdn_i = 0
        rec.attn_i = 0

    tm.register_forward_pre_hook(reset_counters)

    def emb_hook(_m, inp, o):
        rec.pin("input_ids", inp[0], "token ids fed to embedding")
        rec.pin("inputs_embeds", o, "E[input_ids]")

    def final_norm_hook(_m, inp, o):
        rec.pin("pre_final_hidden", inp[0], "input of model.language_model.norm")
        rec.pin("final_hidden", o, "output of RMSNorm_final (pre logits)")

    tm.embed_tokens.register_forward_hook(emb_hook)
    tm.norm.register_forward_hook(final_norm_hook)

    if not args.no_layer_io:
        def make_layer_hook(i):
            def hook(_m, inp, o):
                rec.pin(f"L{i:02d}.s_in", inp[0], f"layer {i} input")
                rec.pin(f"L{i:02d}.s_out", o, f"layer {i} output")
            return hook

        for i, layer in enumerate(layers):
            layer.register_forward_hook(make_layer_hook(i))

    def attn_io_hook(i):
        def hook(_m, inp, o):
            hs = inp[0] if inp else None
            if hs is not None:
                rec.pin(f"L{i:02d}.attn.in", hs, "normed hidden entering attention")
                rec.pin(f"L{i:02d}.attn.out", o[0] if isinstance(o, tuple) else o, "o_proj output")
        return hook

    for i in attn_layers:
        layers[i].self_attn.register_forward_hook(attn_io_hook(i))

    # ---- build the input sequence ------------------------------------------
    seqs, sources = [], []
    for p in args.prompt:
        seqs.append(tok(p, add_special_tokens=False)["input_ids"])
        sources.append(("raw", p))
    for mtext in args.message:
        rendered = tok.apply_chat_template([{"role": "user", "content": mtext}],
                                           add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
        if isinstance(rendered, list):
            rendered = "".join(rendered)
        seqs.append(tok(rendered, add_special_tokens=False)["input_ids"])
        sources.append(("chat", mtext))
    if not seqs:
        seqs.append(tok("The capital of France is", add_special_tokens=False)["input_ids"])
        sources.append(("raw", "The capital of France is"))
    input_ids = torch.tensor([sum(seqs, [])], dtype=torch.long)
    print("sources:", sources)
    print(f"prefill ids ({input_ids.shape[1]}): {input_ids[0].tolist()}")

    gen = []
    with torch.no_grad():
        t0 = time.time()
        o = model(input_ids=input_ids, use_cache=True)
        print(f"prefill {time.time()-t0:.1f}s")
        rec.pin("logits_last", o.logits[:, -1, :], "logits over vocab at final position")
        nxt = int(o.logits[:, -1, :].argmax(-1))
        gen.append(nxt)
        dump_cache(o.past_key_values, rec, "prefill")
        cache = o.past_key_values
        for s in range(args.decode):
            rec.step = f"decode{s}"
            cur = torch.tensor([[nxt]], dtype=torch.long)
            with torch.no_grad():
                o = model(input_ids=cur, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            rec.pin(f"decode{s}.input_ids", cur, "forced input of this step")
            rec.pin(f"decode{s}.logits_last", o.logits[:, -1, :], "logits at final position")
            nxt = int(o.logits[:, -1, :].argmax(-1))
            gen.append(nxt)
            dump_cache(cache, rec, f"decode{s}")

    print("greedy continuation:", gen, "=", repr(tok.decode(gen)))

    for name, t in rec.tensors.items():
        if not torch.isfinite(t).all():
            print(f"WARN non-finite values in {name}")
        np.save(out / f"{name}.npy", t.numpy())
    cfg = model.config.get_text_config()
    manifest = {
        "model": str(args.model), "dtype": args.dtype, "seed": args.seed,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "attn_implementation": "eager",
        "sources": [{"kind": k, "text": txt, "ids": ids} for (k, txt), ids in zip(sources, seqs)],
        "prefill_ids": input_ids[0].tolist(), "greedy_continuation": gen,
        "greedy_text": tok.decode(gen),
        "special_ids": {n: tok.convert_tokens_to_ids(n) for n in [IM_END, IM_START, ENDOFTEXT, THINK_START, THINK_END]},
        "chat_template_render_probe": tok.apply_chat_template(
            [{"role": "user", "content": "PING"}], add_generation_prompt=True,
            tokenize=False, enable_thinking=False),
        "chat_template_render_think": tok.apply_chat_template(
            [{"role": "user", "content": "PING"}], add_generation_prompt=True,
            tokenize=False, enable_thinking=True),
        "text_config": {k: getattr(cfg, k, None) for k in [
            "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps", "hidden_act",
            "attn_output_gate", "linear_num_key_heads", "linear_num_value_heads",
            "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
            "full_attention_interval", "mtp_num_hidden_layers", "max_position_embeddings",
            "layer_types", "rope_parameters"]},
        "pins": rec.meta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"wrote {len(rec.tensors)} tensors + manifest.json to {out}")


if __name__ == "__main__":
    main()
