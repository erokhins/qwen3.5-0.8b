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
  --turn R:TEXT    a whole transcript, one `role:text` per occurrence, rendered as ONE prompt
                   (the multi-turn case `--message` cannot express; mutually exclusive with it and
                   with --prompt). Same renderer as tools/stop_probe.py: tools/chat_render.py.
  --first-stop VAL what tools/stop_probe.py answered for this prompt — an index, or "none in N".
                   Recorded verbatim in the manifest (§10.5 provenance: a deep fixture must state
                   whether its chain reaches its own stop inside the pinned window).
  --no-chat        do not append the no-think chat template rendering
  --decode N       greedy continuation steps after prefill (default 1). Each step feeds the
                   reference's OWN argmax back in, i.e. it is a forced (teacher-forced) chain,
                   not a free run: see "decode_mode" in the manifest and P1 of
                   bench/plan-2026-10-12.md for why depth can only be pinned that way (A-18).
  --no-layer-io    omit per-layer input/output dumps
  --no-mlp-io      omit the per-layer MLP operands (§03.3); they are the widest pins
                   ([T, 3584] x 3 x 24 layers) and exist to gate the MLP kernel
  --dtype          bfloat16 (default, as shipped) | float32
  --lean           pin policy for a DEEP fixture: per step keep only `logits_last`,
                   `input_ids` and the three state-fingerprint arrays, and keep the full
                   per-layer + cache dump only at `--deep-steps`. A naive 64-step pin is
                   ~1.7 GB (dec8 costs 25.9 MB per step); `--lean` is what buys depth.
  --deep-steps L   which decode steps keep the full dump under `--lean`: "63", "0,63", "all"
                   (default "all", which with `--lean` filters nothing).
  --logits-bits B  f32 (default, every existing fixture) | bf16. bf16 stores `logits_last`
                   as the BF16 bit pattern (uint16, bitcast — §10.5) instead of widened F32,
                   which halves the widest pin. Bit-lossless here, not approximately: these
                   logits are already BF16-exact, so the round-trip moves no element (the
                   probe behind that claim is recorded in engine/results.md, dec64f section).
                   A reader with the old pins: `gate.py:load_pin` widens uint16 by <<16.
  --verify DIR     run, write NOTHING into DIR, and instead byte-compare the fresh run with
                   what is already pinned there (§10.5: a fresh run must reproduce the pin).
                   Exits 1 on any difference, including a file that exists on one side only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from chat_render import parse_turn, render_transcript   # noqa: E402

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
    p.add_argument("--turn", action="append", default=[], metavar="ROLE:TEXT",
                   help="render a whole transcript as one prompt (multi-turn fixture)")
    p.add_argument("--first-stop", default=None, metavar="VAL",
                   help="first stop-id index from tools/stop_probe.py, or 'none in N'; recorded"
                        " in the manifest")
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--decode", type=int, default=1)
    p.add_argument("--no-layer-io", action="store_true")
    p.add_argument("--no-mlp-io", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lean", action="store_true",
                   help="deep-fixture pin policy: per step keep logits + ids + state "
                        "fingerprints only, and the full dump only at --deep-steps")
    p.add_argument("--deep-steps", default="all",
                   help='with --lean: which decode steps keep the full per-layer dump, '
                        'e.g. "63" or "0,63" or "all"')
    p.add_argument("--logits-bits", default="f32", choices=["f32", "bf16"],
                   help="store logits_last as F32 (every existing fixture) or as BF16 bits")
    p.add_argument("--verify", default=None, metavar="DIR",
                   help="generate into a scratch dir and byte-compare with DIR instead of "
                        "writing pins (§10.5 reproduction check)")
    p.add_argument("--keep-temp", action="store_true",
                   help="with --verify: leave the scratch dir in place for inspection")
    return p.parse_args()


def is_per_layer_pin(name: str) -> bool:
    """Does this pin belong to one model layer rather than to the step as a whole?

    The rule-hook pins are bare `L07.gdn.beta` at prefill and gain the `decodeN.` prefix inside
    `Recorder.pin`; the cache pins are `decodeN.L03.cache_keys` from birth. Both families are
    what `--lean` drops, and testing the name rather than the call site means a new hook cannot
    slip past the policy by being pinned from a new place.
    """
    return name.startswith("L") or ".L" in name


def parse_step_list(spec: str) -> set[str] | None:
    """`--deep-steps` as step tags. "all" (and the default) means do not filter at all, which
    is every fixture that existed before `--lean` and must stay byte-identical. "none" keeps no full
    dump anywhere: logits + ids + fingerprints only, which is what a 128-step fixture has to do to fit
    the budget (one full step is 26 MiB, 18 MiB of it the GDN state, and 128 steps of bf16 logits are
    already 61 MiB — see `engine/results.md`, the breadth section)."""
    spec = spec.strip()
    if spec in ("", "all"):
        return None
    if spec == "none":
        return set()
    return {f"decode{int(s)}" for s in spec.split(",") if s.strip() != ""}


def cache_state_tensors(layer, attr: str) -> list[torch.Tensor]:
    """The per-layer state a cache layer holds under `attr`, however it is filed.

    In transformers 5.19 `DynamicLayer.recurrent_states` and `.conv_states` are LAZILY
    INITIALISED DICTS ({state_index: tensor}), not lists, so `enumerate(vals)` — which is what
    `dump_cache` does — yields integer KEYS and `Recorder.pin` drops every one of them. That is
    why no fixture has ever carried a cache GDN-state pin: the state reached the dumps through
    the rule/conv hooks (`gdn.state_new`, `gdn.conv_state`) instead. This reader handles both
    shapes because the fingerprint must see the state even when the hook pins are filtered away
    by `--lean`; `dump_cache`'s own behaviour is deliberately unchanged (fixing the dict case
    there would add pins and break every existing fixture's byte-for-byte reproduction).
    """
    vals = getattr(layer, attr, None)
    if vals is None:
        return []
    if isinstance(vals, dict):
        vals = [vals[k] for k in sorted(vals)]
    out = []
    for v in vals if isinstance(vals, (list, tuple)) else [vals]:
        if isinstance(v, torch.Tensor) and v.numel():
            out.append(v)
    return out


def fp_row(parts: list[torch.Tensor], prev: list[torch.Tensor] | None) -> list[float]:
    """One row of a state fingerprint: [sum x², max|x|, mean x, dn2], F64.

    `parts` is the tensors that make up one logical state (the GDN row has one; the attention
    row is K and V together, so it is two) and they are summarised as one pooled row. Parts are
    compared PART-WISE against the previous step's parts, never as one flattened blob: K and V
    grow by one lane per step, so a flat `x[:prev.numel()] - prev` would silently compare K's
    tail against the start of V and report a large delta for a cache that is behaving perfectly.
    (It did, at 1e4 on a clean run, which is why the definition is part-wise.)

    Moment sums, deliberately: they are invariant to element ORDER, and the engine lays its GDN
    window and KV out differently from the reference in places (`state.h`: head-major window,
    `[head][slot][dim]` KV), so an order-sensitive summary would read a layout difference as a
    bug. What they catch is a state that decayed, was zeroed, was scaled, or was left behind —
    which is what a depth scan looks for. What they CANNOT catch is a corruption that preserves
    the multiset of values (a rotation within the tensor); per-element state gating needs the
    full tensors, at 18 MiB per step, which is exactly what a deep fixture cannot afford.

    dn2 = Σ_i Σ_{common extent of dim −2} (x_i − prev_i)², i.e. how much the ALREADY-WRITTEN
    entries moved. Append-only storage therefore has an expected dn2 of exactly 0.0, and a
    non-zero kv_fp dn2 means history was rewritten. No `prev` (§08.4: a fresh request's state is
    zeroed) means the delta is the tensor itself, so the first row's dn2 equals its n2.

    "The common extent of dim −2" is the append axis for a growing cache ([1,H,T,D]) and the
    whole extent for a state that never grows (S [1,16,128,128], conv [1,6144,4]), which is why
    one rule serves all three families. Flattening first and slicing a common PREFIX is wrong for
    a growing tensor: when T changes, the per-head boundaries move, so a flat prefix compares
    head 1 against the tail of head 0 — it read 3.0e3 on a cache that is bit-identical.
    """
    flat = torch.cat([p.reshape(-1) for p in parts]).to(torch.float32)
    n2 = float(torch.sum(flat * flat))
    row = [n2, float(torch.max(torch.abs(flat))), float(torch.mean(flat))]
    if prev is None:
        row.append(n2)
        return row
    dn2 = 0.0
    for x, p in zip(parts, prev):
        x = x.detach().to(torch.float32)
        n = min(x.shape[-2], p.shape[-2])
        d = x[..., :n, :] - p[..., :n, :]
        dn2 += float(torch.sum(d * d))
    row.append(dn2)
    return row


class Recorder:
    """Pins tensors by name. Call-order counters map reference calls to layers.

    `keep_steps` is the `--lean` policy: None keeps every step's per-layer operands (the
    behaviour every fixture before `--lean` was made with); a set of step tags keeps them only
    for those steps. It filters per-layer names only — the step's `logits_last`, `input_ids`
    and fingerprint arrays are what a deep fixture is FOR and are always kept.
    """

    def __init__(self, keep_steps: set[str] | None = None, logits_bits: str = "f32"):
        self.tensors: dict[str, torch.Tensor] = {}
        self.meta: dict[str, dict] = {}
        self.step = "prefill"
        self.gdn_i = 0
        self.attn_i = 0
        self.keep_steps = keep_steps
        self.logits_bits = logits_bits
        # fingerprint bookkeeping: the previous step's state, per (family, layer row), on the
        # host. ~20 MB for the whole model; it is what makes the delta column a DELTA.
        self.fp_prev: dict[tuple[str, int], torch.Tensor] = {}
        self.fp_rows: dict[str, list[int]] = {"state": [], "conv": [], "kv": []}

    def pin(self, name: str, t, note: str = "", logits: bool = False):
        if t is None:
            return
        if isinstance(t, (list, tuple)):
            for j, item in enumerate(t):
                self.pin(f"{name}[{j}]", item, note)
            return
        if not isinstance(t, torch.Tensor):
            return
        # `--lean`: a per-layer operand survives only at a deep step. Tested on the pre-prefix
        # name so the rule is one test for both phases, and on the NAME rather than the call site
        # so a future hook cannot pin its way past the policy.
        if (self.keep_steps is not None and is_per_layer_pin(name)
                and self.step not in self.keep_steps):
            return
        if self.step != "prefill" and name.startswith("L"):
            name = f"{self.step}.{name}"      # keep decode-step pins separate from prefill pins
        if logits and self.logits_bits == "bf16" and t.dtype is torch.bfloat16:
            # Bitcast, never a numeric cast: `.view(torch.uint16)` keeps the bits, `.to(...)`
            # would write the INTEGER the float rounds to and the pin would read back as
            # garbage that looks like unwritten memory (the trap in AGENTS.md, and the reason
            # `gate.py:load_pin` widens a uint16 payload by <<16).
            self.tensors[name] = t.detach().view(torch.uint16).cpu().clone()
            self.meta[name] = {"step": self.step, "note": note,
                               "shape": list(t.shape), "dtype": "bfloat16(bits:uint16)"}
            return
        self.tensors[name] = t.detach().to(torch.float32).cpu().clone()
        self.meta[name] = {"step": self.step, "note": note,
                           "shape": list(self.tensors[name].shape),
                           "dtype": str(t.dtype).replace("torch.", "")}

    def pin_array(self, name: str, arr: np.ndarray, note: str = ""):
        """Pin an array this file computed (the state fingerprints), not a model output.
        Separate from `pin` because those rows are F64 sums, not tensors the model produced."""
        self.tensors[name] = arr
        self.meta[name] = {"step": self.step, "note": note, "shape": list(arr.shape),
                           "dtype": str(arr.dtype)}

    def fingerprint(self, cache, tag: str):
        """Pin the live cache's state summary for this step: `state_fp` (GDN recurrent S),
        `conv_fp` (conv windows) and `kv_fp` (K and V together per attention layer), each
        [n_rows, 4] F64 with `fp_row`'s columns. Rows are in ascending model-layer order and
        the layer index of every row goes in the manifest, because the row index is not the
        layer index (18 GDN + 6 attention out of 24).

        This is the `--lean` stand-in for pinning the state itself, which costs 18 MiB per step
        (64 steps would be 1.15 GB) and buys per-element depth nobody gates today.
        """
        layers = getattr(cache, "layers", None)
        if layers is None:
            return
        rows: dict[str, list[list[float]]] = {"state": [], "conv": [], "kv": []}
        seen: dict[str, list[int]] = {"state": [], "conv": [], "kv": []}
        for i, layer in enumerate(layers):
            for kind, attr in (("state", "recurrent_states"), ("conv", "conv_states")):
                for v in cache_state_tensors(layer, attr):
                    key = (kind, len(rows[kind]))
                    rows[kind].append(fp_row([v], self.fp_prev.get(key)))
                    self.fp_prev[key] = [v.detach().to(torch.float32).cpu().clone()]
                    seen[kind].append(i)
            ks = cache_state_tensors(layer, "keys")
            vs = cache_state_tensors(layer, "values")
            if ks and vs:
                parts = [ks[0].to(torch.float32), vs[0].to(torch.float32)]
                key = ("kv", len(rows["kv"]))
                rows["kv"].append(fp_row(parts, self.fp_prev.get(key)))
                self.fp_prev[key] = [p.cpu().clone() for p in parts]
                seen["kv"].append(i)
        # The row -> model-layer map is the same at every step, so the manifest carries it once
        # rather than 1170 times for a 64-step fixture.
        self.fp_rows = seen
        for kind, rows_for_kind in rows.items():
            if not rows_for_kind:
                continue
            self.pin_array(f"{tag}{kind}_fp", np.asarray(rows_for_kind, dtype=np.float64),
                           "state fingerprint [layer,4] cols n2|maxabs|mean|dn2 (dn2 = delta "
                           "vs previous step, zeros vs §08.4's zeroed fresh state)")


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
    verify_dir = Path(args.verify) if args.verify else None
    if verify_dir is not None:
        # §10.5: a fresh run must reproduce the pin, byte for byte, or the fixture is not
        # pinned. So a verification run writes into a scratch directory and compares; it never
        # touches the pin it is checking.
        if not verify_dir.is_dir():
            raise SystemExit(f"--verify: pinned fixture directory not found: {verify_dir}")
        out = Path(tempfile.mkdtemp(prefix=f"{out.name}.verify."))
        print(f"verifying {verify_dir} against a fresh run in {out}")
    else:
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

    rec = Recorder(keep_steps=(parse_step_list(args.deep_steps) if args.lean else None),
                   logits_bits=args.logits_bits)
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

    # ---- MLP operands (§03.3) ----------------------------------------------
    # HF: down_proj(act_fn(gate_proj(x)) * up_proj(x)). Pinning all four operands lets a
    # kernel be gated in isolation (feed mlp.in, compare gate_pre/up_pre/act/out) with no
    # upstream cascade, which is what G3i needs. `act` is the down_proj input, i.e. the
    # bf16-rounded silu(gate) * up product whose rounding policy A-07 pins.
    if not args.no_mlp_io:
        def mlp_hook(i):
            def hook(_m, inp, o):
                if inp:
                    rec.pin(f"L{i:02d}.mlp.in", inp[0], "post_attention_layernorm output")
                rec.pin(f"L{i:02d}.mlp.out", o, "mlp() = down_proj(act), the residual addend")
            return hook

        def sub_hook(i, which, note, take_output=True):
            def hook(_m, inp, o):
                rec.pin(f"L{i:02d}.mlp.{which}", o if take_output else inp[0], note)
            return hook

        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            mlp.register_forward_hook(mlp_hook(i))
            mlp.gate_proj.register_forward_hook(sub_hook(i, "gate_pre", "gate_proj(x), bf16"))
            mlp.up_proj.register_forward_hook(sub_hook(i, "up_pre", "up_proj(x), bf16"))
            # `act` is down_proj's INPUT, i.e. the operand the engine must reproduce; the
            # hook has to read inp[0] for that, not the output.
            mlp.down_proj.register_forward_hook(
                sub_hook(i, "act", "silu(gate)*up = down_proj input", take_output=False))

    # ---- projection operands (specification/01 §01.2 / §01.3) ----------------
    # The nine text-path projection modules, each pinned at its own output, so a projection
    # kernel is gated against the tensor it produces instead of against a derivative of it:
    # gdn.mixed_qkv_pre_conv is in_proj_qkv transposed for the conv, gdn.o_pre_norm is after
    # the gated RMSNorm, attn.q_rope after q_norm + rope, attn.attn_out_pre_gate *before*
    # o_proj. Same module-hook mechanism as the MLP block above, so these pins exist for
    # prefill (unprefixed) and for every decode step (decodeN.-prefixed) alike.
    def proj_hook(i, which, note):
        def hook(_m, _inp, o):
            rec.pin(f"L{i:02d}.{which}.out", o, note)
        return hook

    GDN_PROJ = [
        ("in_proj_qkv", "in_proj_qkv(x): [q key_dim | k key_dim | v value_dim], the conv operand"),
        ("in_proj_z", "in_proj_z(x): gate of the gated RMSNorm, before the head reshape"),
        ("in_proj_b", "in_proj_b(x): beta pre-activation per value head (sigmoid is downstream)"),
        ("in_proj_a", "in_proj_a(x): decay pre-activation per value head (softplus+dt_bias downstream)"),
        ("out_proj", "out_proj(gated_out): GDN block output, the residual addend"),
    ]
    ATTN_PROJ = [
        ("q_proj", "q_proj(x): q and the output gate, interleaved per head as [8 heads][2 x 256]"),
        ("k_proj", "k_proj(x): before k_norm and rope"),
        ("v_proj", "v_proj(x): value states before the KV-cache write"),
        ("o_proj", "o_proj(attn_out * sigmoid(gate)): attention block output, the residual addend"),
    ]

    for i in gdn_layers:
        for which, note in GDN_PROJ:
            getattr(layers[i].linear_attn, which).register_forward_hook(proj_hook(i, which, note))
    for i in attn_layers:
        for which, note in ATTN_PROJ:
            getattr(layers[i].self_attn, which).register_forward_hook(proj_hook(i, which, note))

    # ---- build the input sequence ------------------------------------------
    seqs, sources = [], []
    for p in args.prompt:
        seqs.append(tok(p, add_special_tokens=False)["input_ids"])
        sources.append(("raw", p))
    for mtext in args.message:
        rendered = render_transcript(tok, [{"role": "user", "content": mtext}], thinking=False)
        seqs.append(tok(rendered, add_special_tokens=False)["input_ids"])
        sources.append(("chat", mtext))
    if args.turn:
        # One transcript, one sequence. `--turn` replaces `--message`/`--prompt` rather than joining
        # them: a second chat header appended to a transcript would be a different fixture than the
        # transcript that was probed, which is the whole thing this flag has to get right.
        if args.message or args.prompt:
            raise SystemExit("--turn renders the whole prompt; do not also pass --message/--prompt"
                             " (make the extra text one of the turns instead)")
        messages = [parse_turn(t) for t in args.turn]
        rendered = render_transcript(tok, messages, thinking=False)
        seqs.append(tok(rendered, add_special_tokens=False)["input_ids"])
        sources.append(("chat_transcript",
                        "  ||  ".join(f"{m['role']}: {m['content']}" for m in messages)))
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
        rec.pin("logits_last", o.logits[:, -1, :], "logits over vocab at final position",
                logits=True)
        nxt = int(o.logits[:, -1, :].argmax(-1))
        gen.append(nxt)
        rec.fingerprint(o.past_key_values, "")
        dump_cache(o.past_key_values, rec, "prefill")
        cache = o.past_key_values
        for s in range(args.decode):
            rec.step = f"decode{s}"
            # FORCED, not free-running: the token fed is the reference's own previous argmax, so
            # the chain cannot diverge from itself and depth costs nothing (A-18 is why a 64-token
            # FREE run could not be pinned at all). `decode{s}.input_ids` is the proof of what was
            # fed, and `gate.py` re-derives the same identity from the engine's side.
            cur = torch.tensor([[nxt]], dtype=torch.long)
            with torch.no_grad():
                o = model(input_ids=cur, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            rec.pin(f"decode{s}.input_ids", cur, "forced input of this step")
            rec.pin(f"decode{s}.logits_last", o.logits[:, -1, :], "logits at final position",
                    logits=True)
            nxt = int(o.logits[:, -1, :].argmax(-1))
            gen.append(nxt)
            rec.fingerprint(cache, f"decode{s}.")
            dump_cache(cache, rec, f"decode{s}")

    print("greedy continuation:", gen, "=", repr(tok.decode(gen)))

    for name, t in rec.tensors.items():
        finite = (torch.isfinite(t).all() if isinstance(t, torch.Tensor)
                  else bool(np.isfinite(t).all()))
        if not finite:
            print(f"WARN non-finite values in {name}")
        np.save(out / f"{name}.npy", t.numpy() if isinstance(t, torch.Tensor) else t)
    cfg = model.config.get_text_config()
    manifest = {
        "model": str(args.model), "dtype": args.dtype, "seed": args.seed,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "attn_implementation": "eager",
        # Every step above fed the reference's own previous argmax, which is what makes a deep
        # pin possible at all: a FREE 64-token run cannot be pinned, because two legal bf16
        # continuations of the same prompt agree only 0.8776 of the way over 24 tokens (A-18).
        "decode_mode": ("forced_greedy: step s is fed the reference's own argmax of step s-1 "
                        "(teacher forcing), so no divergence cascade can enter and depth is "
                        "free. The engine is gated against this chain with the same feed list "
                        "(decode_run --replay), never against its own output."),
        "pin_policy": ("full" if not args.lean else
                       "lean: per-layer and cache operands only at "
                       + ("all steps" if rec.keep_steps is None else
                          ("no step (logits + ids + fingerprints only)" if not rec.keep_steps
                           else ", ".join(sorted(rec.keep_steps))))),
        "logits_storage": args.logits_bits,
        "state_fingerprint": {
            "columns": ["n2=sum x^2", "maxabs", "mean", "dn2=sum (x - previous step)^2 over the "
                        "already-written extent, so a kv_fp dn2 of 0.0 is the expected value "
                        "(the cache is append-only; anything else means history was rewritten) "
                        "and the first row's dn2 equals its n2 because §08.4 zeroes a fresh "
                        "request's state"],
            "rows_are_model_layers": rec.fp_rows,
            "note": "moment sums, so element order does not matter and the engine's differing "
                    "KV/window layout (KV §08.5 vs state.h) is not a false failure",
        },
        "sources": [{"kind": k, "text": txt, "ids": ids} for (k, txt), ids in zip(sources, seqs)],
        # §10.5 provenance for a deep fixture: the standing rule is that the prompt must be probed to
        # run past the pinned window, and a claim about that belongs with the fixture, not only in a
        # results file. Value is `tools/stop_probe.py`'s answer verbatim (an index, or "none in N"),
        # so it can be re-checked by running that probe with the same --message/--turn and --max-new.
        "stop_probe": None if args.first_stop is None else {
            "first_stop": args.first_stop,
            "stop_id": int(tok.convert_tokens_to_ids(IM_END)),
            "note": "index of the FIRST generated token equal to the stop id, counted from the first"
                    " generated token; 'none in N' means no stop id in N generated tokens. Prompt-"
                    " phase occurrences (the transcript carries one per closed turn) are not stops —"
                    " engine/gate.py's §07.5 note is the same distinction on the engine's side.",
        },
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
    nbytes = {n: (t.numpy() if isinstance(t, torch.Tensor) else t).nbytes
              for n, t in rec.tensors.items()}
    mib = 1024 * 1024
    # `rec.tensors` keys are pin NAMES, not file names — `.npy` is added by np.save below.
    logits_b = sum(v for n, v in nbytes.items() if n.endswith("logits_last"))
    fp_b = sum(v for n, v in nbytes.items() if n.endswith(("state_fp", "conv_fp", "kv_fp")))
    print(f"wrote {len(rec.tensors)} tensors + manifest.json to {out}")
    if manifest.get("stop_probe"):
        print(f"  stop probe recorded: first stop id {manifest['stop_probe']['stop_id']}"
              f" at {manifest['stop_probe']['first_stop']}")
    elif args.lean:
        print("  WARNING: a lean (deep) fixture with no recorded --first-stop: the standing rule is"
              " to pre-flight the prompt with tools/stop_probe.py and record the answer. Nothing"
              " here can tell a prompt that runs 300 tokens from one that stops at 12.")
    print(f"  total {sum(nbytes.values()) / mib:.1f} MiB of pins — logits"
          f" {logits_b / mib:.1f} MiB stored as {args.logits_bits}, state fingerprints"
          f" {fp_b / 1024:.1f} KiB, per-layer+cache+ids"
          f" {(sum(nbytes.values()) - logits_b - fp_b) / mib:.1f} MiB")
    if verify_dir is not None:
        rc = verify_against(verify_dir, out)
        if args.keep_temp:
            print(f"scratch kept at {out}")
        else:
            shutil.rmtree(out, ignore_errors=True)
        raise SystemExit(rc)


def verify_against(pin: Path, fresh: Path) -> int:
    """§10.5's reproduction check.

    Every pinned `.npy` must come out of a fresh run with the SAME BYTES. Byte equality rather
    than a numeric tolerance on purpose: a tolerance lets a small drift become permanent, and the
    thing being proven is that the fixture is a fact about the checkpoint, not an accident of one
    run (the trap this retires is dec8's `pre_final_hidden.npy` + `logits_last.npy` pair, which
    are last-writer-wins leftovers of different steps and do not even pair with each other).

    Two things are reported and do NOT fail, because they mean the dumper knows more than the pin
    does rather than that the model moved:
      * files only in the fresh run — e.g. the state fingerprints, added after dec8 was pinned;
      * manifest keys only in the fresh run, same reason. A pinned key whose value CHANGED, or
        that went missing, is a failure.
    """
    want = sorted(p.name for p in pin.iterdir() if p.is_file())
    got = sorted(p.name for p in fresh.iterdir() if p.is_file())
    missing = [n for n in want if n not in got]
    extra = [n for n in got if n not in want]
    differing = [n for n in want if n in got and n != "manifest.json"
                 and (pin / n).read_bytes() != (fresh / n).read_bytes()]
    pinned_man = json.loads((pin / "manifest.json").read_text())
    fresh_man = json.loads((fresh / "manifest.json").read_text())
    man_bad = manifest_mismatches(pinned_man, fresh_man)
    print(f"\nREPRODUCE {pin.name}: {len(want)} pinned files vs {len(got)} from the fresh run")
    for label, items in (("missing from the fresh run", missing),
                         ("byte-differing", differing)):
        if items:
            print(f"  {label}: {len(items)} — " + ", ".join(items[:12])
                  + (" …" if len(items) > 12 else ""))
    if extra:
        print(f"  note, only in the fresh run (newer dumper, not fixture drift): {len(extra)} — "
              + ", ".join(extra[:6]) + (" …" if len(extra) > 6 else ""))
    if man_bad:
        print(f"  manifest values that did not reproduce: {len(man_bad)} — "
              + ", ".join(man_bad[:10]) + (" …" if len(man_bad) > 10 else ""))
    man_added = sorted(k for k in fresh_man if k not in pinned_man)
    if man_added:
        print(f"  note, manifest keys only in the fresh run: {', '.join(man_added)}")
    ok = not (missing or differing or man_bad)
    print("REPRODUCE " + ("PASS — every pinned byte and every pinned manifest value came back"
                          " identical" if ok else
                          "FAIL — the fresh run does not reproduce the pin, so the fixture is"
                          " not pinned"))
    return 0 if ok else 1


def manifest_mismatches(pin, fresh, path="") -> list[str]:
    """Every key of the PINNED manifest must be present in the fresh one with an equal value.
    Returns the dotted paths that broke; keys the fresh manifest added are not a mismatch."""
    if isinstance(pin, dict):
        if not isinstance(fresh, dict):
            return [path or "$"]
        out = []
        for k, v in pin.items():
            if k not in fresh:
                out.append(f"{path}.{k}")
            else:
                out += manifest_mismatches(v, fresh[k], f"{path}.{k}")
        return out
    if isinstance(pin, list):
        if not isinstance(fresh, list) or len(pin) != len(fresh):
            return [path or "$"]
        out = []
        for i, (a, b) in enumerate(zip(pin, fresh)):
            out += manifest_mismatches(a, b, f"{path}[{i}]")
        return out
    return [] if pin == fresh else [path or "$"]


if __name__ == "__main__":
    main()
