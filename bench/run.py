#!/usr/bin/env python3
"""Run the engine-agnostic tests in testdata/ against one engine.

Engines here are *implementations of the same model*: `speccheck` (this repository's
from-scratch specification path) and `hf` (the transformers reference that produced
data/refs). Both are given the same token ids and both are driven the same way - one
forward over the prompt, then one forward per generated token with the cache/state
carried - so prefill and decode speeds are separated honestly instead of being folded
together by generate().

One process = one engine (the libraries do not coexist well), so bench.sh invokes this
once per engine and then prints the combined table with --table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EOS_IDS = (248046, 248044)          # A-10: tokenizer eos, then the legacy stop
GIB = 1024 ** 3


# --------------------------------------------------------------------------- tests
def load_tests(which: str) -> list[dict]:
    index = json.loads((ROOT / "testdata" / "index.json").read_text())
    out = []
    for name in index["tests"]:
        t = json.loads((ROOT / "testdata" / f"{name}.json").read_text())
        text = t.get("instruction", "")
        if t.get("content_file"):
            text += (ROOT / "testdata" / t["content_file"]).read_text()
        t["prompt_text"] = text
        if which != "all" and which not in (t["goal"], t["name"]):
            continue
        out.append(t)
    if not out:
        sys.exit(f"no test matched {which!r} in testdata/index.json")
    return out


def check_shape(t: dict, prompt_tok: int, gen_tok: int) -> str:
    e = t.get("expect", {})
    bad = []
    if "min_prompt_tokens" in e and prompt_tok < e["min_prompt_tokens"]:
        bad.append(f"prompt {prompt_tok} < {e['min_prompt_tokens']}")
    if "max_prompt_tokens" in e and prompt_tok > e["max_prompt_tokens"]:
        bad.append(f"prompt {prompt_tok} > {e['max_prompt_tokens']}")
    if "min_gen_tokens" in e and gen_tok < e["min_gen_tokens"]:
        bad.append(f"gen {gen_tok} < {e['min_gen_tokens']}")
    if "max_gen_tokens" in e and gen_tok > e["max_gen_tokens"]:
        bad.append(f"gen {gen_tok} > {e['max_gen_tokens']}")
    return "ok" if not bad else "SHAPE " + "; ".join(bad)


# ------------------------------------------------------------------------- engines
class SpeccheckEngine:
    name = "speccheck"

    def __init__(self, model_dir: Path, gdn_prefill: str, attn_softmax: str):
        from speccheck import textio
        from speccheck.specimpl import Forward, Weights
        self.textio = textio
        self.tok = textio.Tokenizer(model_dir)
        self.w = Weights(model_dir)
        self.f = Forward(self.w, gdn_form=gdn_prefill, attn_softmax=attn_softmax)
        self.weight_bytes = sum(t.numel() * t.element_size() for t in self.w.p.values())

    def render(self, text: str) -> list[int]:
        return self.tok.render_ids([{"role": "user", "content": text}])

    def reset(self):
        self.f.reset()

    def prefill(self, ids: list[int]) -> int:
        return int(self.f.logits(self.f.prefill(torch.tensor(ids)))[-1].argmax())

    def step(self, tok_id: int) -> int:
        return int(self.f.logits(self.f.decode(tok_id))[-1].argmax())

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)


class HFEngine:
    name = "hf"

    def __init__(self, model_dir: Path, dtype: str = "bfloat16", full_logits: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=getattr(torch, dtype), attn_implementation="eager")
        self.model.eval()
        self.weight_bytes = sum(p.numel() * p.element_size()
                                for p in self.model.parameters())
        # speccheck's lm_head only touches the final position, so HF must not be made
        # to run the 248k-row head over the whole prompt or the prefill number is not
        # the same quantity. --hf-full-logits turns this off to show the difference.
        self.ltk = {} if full_logits else {"logits_to_keep": 1}

    def render(self, text: str) -> list[int]:
        r = self.tok.apply_chat_template([{"role": "user", "content": text}],
                                         add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        if isinstance(r, list):
            r = "".join(r)
        return list(self.tok(r, add_special_tokens=False)["input_ids"])

    def reset(self):
        self.cache = None

    def prefill(self, ids: list[int]) -> int:
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([ids], dtype=torch.long),
                           use_cache=True, **self.ltk)
        self.cache = o.past_key_values
        return int(o.logits[:, -1, :].argmax())

    def step(self, tok_id: int) -> int:
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([[tok_id]], dtype=torch.long),
                           past_key_values=self.cache, use_cache=True, **self.ltk)
        self.cache = o.past_key_values
        return int(o.logits[:, -1, :].argmax())

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------- run
def run_test(eng, t: dict, reps: int, max_new: int | None) -> dict:
    limit = max_new or t["max_new_tokens"]
    # Both engines must render the same text. HF's Qwen template applies `|trim` to
    # message content, so whitespace around the test text is not part of the prompt
    # there; without the same trim here the two engines differ by a whitespace token
    # and the prompt-hash cross-check in --table fires.
    ids = eng.render(t["prompt_text"].strip())
    phash = hashlib.sha256(str(ids).encode()).hexdigest()[:12]
    rows, preview = [], ""
    for r in range(reps):
        eng.reset()
        t0 = time.monotonic()
        nxt = eng.prefill(ids)
        prefill_s = time.monotonic() - t0
        out, stop = [nxt], "max_new"
        t1 = time.monotonic()
        while len(out) < limit:
            if nxt in EOS_IDS:
                stop = "eos"
                break
            nxt = eng.step(nxt)
            out.append(nxt)
        decode_s = time.monotonic() - t1
        n_dec = max(len(out) - 1, 0)
        rows.append({"prompt_tokens": len(ids), "prefill_s": prefill_s,
                     "gen_tokens": len(out), "decode_steps": n_dec,
                     "decode_s": decode_s, "stop": stop,
                     "prefill_tok_s": len(ids) / prefill_s if prefill_s else 0.0,
                     "decode_tok_s": n_dec / decode_s if decode_s else 0.0,
                     "implied_gbps": (eng.weight_bytes * n_dec / decode_s / 1e9
                                      if decode_s else 0.0)})
        if r == 0:
            preview = eng.decode(out)
    med = lambda k: statistics.median(r[k] for r in rows)      # noqa: E731
    last = rows[-1]
    shape = check_shape(t, last["prompt_tokens"], last["gen_tokens"])
    if max_new and shape.startswith("SHAPE gen") and limit < t.get("expect", {}).get(
            "min_gen_tokens", 0):
        shape = "capped (--max-new below the test's own floor)"
    res = {"engine": eng.name, "test": t["name"], "goal": t["goal"], "reps": reps,
           "prompt_tokens": last["prompt_tokens"], "prompt_hash": phash,
           "max_new_tokens": limit, "stop": last["stop"], "shape": shape,
           "prefill_s": round(med("prefill_s"), 3),
           "prefill_tok_s": round(med("prefill_tok_s"), 1),
           "gen_tokens": last["gen_tokens"],
           "decode_steps": last["decode_steps"],
           "decode_s": round(med("decode_s"), 3),
           "decode_tok_s": round(med("decode_tok_s"), 2),
           "implied_gbps": round(med("implied_gbps"), 1),
           "ttft_s": round(med("prefill_s"), 3),
           "weight_GiB": round(eng.weight_bytes / GIB, 3),
           "output_chars": len(preview), "preview": preview[:1200],
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print("%-10s %-18s prompt %5d tok | prefill %7.2fs %7.1f tok/s | gen %5d tok "
          "decode %6.2fs %6.2f tok/s | %5.1f GB/s | stop=%s | %s"
          % (eng.name, t["name"], res["prompt_tokens"], res["prefill_s"],
             res["prefill_tok_s"], res["gen_tokens"], res["decode_s"],
             res["decode_tok_s"], res["implied_gbps"], res["stop"], shape),
          flush=True)
    return res


def print_table(paths: list[str]):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        sys.exit("no result rows to print")
    hdr = ("ENGINE", "TEST", "prompt", "prefill_s", "prefill tok/s", "gen tok",
           "decode_s", "decode tok/s", "GB/s", "stop", "shape")
    cols = ("engine", "test", "prompt_tokens", "prefill_s", "prefill_tok_s",
            "gen_tokens", "decode_s", "decode_tok_s", "implied_gbps", "stop", "shape")
    widths = [max(len(h), *(len(str(r.get(c, ""))) for r in rows)) for h, c in
              zip(hdr, cols)]
    print("  ".join(h.ljust(w) for h, w in zip(hdr, widths)))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
    # A speed comparison between engines that read different tokens is meaningless.
    by = {}
    for r in rows:
        by.setdefault(r["test"], set()).add(r["prompt_hash"])
    for test, hs in sorted(by.items()):
        print("prompt ids identical across engines for %s: %s"
              % (test, "yes" if len(hs) == 1 else "NO (%d variants)" % len(hs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["speccheck", "hf"])
    ap.add_argument("--test", default="all",
                    choices=["all", "prefill", "decode", "prefill_summarize",
                             "decode_poem"])
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--jsonl", default=str(ROOT / "data" / "bench" / "results.jsonl"))
    ap.add_argument("--no-text", action="store_true", help="skip printing the output")
    ap.add_argument("--gdn-prefill", default="chunk", choices=["recurrent", "chunk"])
    ap.add_argument("--attn-softmax", default="full", choices=["full", "online"])
    ap.add_argument("--hf-full-logits", action="store_true",
                    help="let HF run lm_head over every prompt position (slower, "
                         "not the same quantity as speccheck)")
    ap.add_argument("--table", nargs="+", default=None,
                    help="print a combined table from these jsonl files and exit")
    args = ap.parse_args()

    if args.table:
        print_table(args.table)
        return
    if not args.engine:
        sys.exit("--engine is required unless --table is used")

    torch.set_grad_enabled(False)
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "0") or 0))
    t0 = time.monotonic()
    eng = (SpeccheckEngine(Path(args.model), args.gdn_prefill, args.attn_softmax)
           if args.engine == "speccheck" else
           HFEngine(Path(args.model), full_logits=args.hf_full_logits))
    print("# %s engine ready in %.1fs, weights %.2f GiB, threads %s"
          % (eng.name, time.monotonic() - t0, eng.weight_bytes / GIB,
             torch.get_num_threads() or "default"), flush=True)

    out = Path(args.jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    for t in load_tests(args.test):
        res = run_test(eng, t, args.reps, args.max_new)
        with out.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
        if not args.no_text:
            print("--- %s output (%d chars) ---\n%s\n" % (t["name"], res["output_chars"],
                                                          res["preview"]), flush=True)


if __name__ == "__main__":
    main()
