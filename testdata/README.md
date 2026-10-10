# `testdata/` - engine-agnostic benchmark tests

Two tests, deliberately boring and reusable: one that is **prefill-dominated** and one
that is **decode-dominated**, so any engine can be measured the same way - today
`speccheck` (torch CPU) and the HuggingFace reference, later the HIP engine.

| file | what it is |
|---|---|
| `index.json` | test list + the measurement conventions every engine must honour |
| `prefill_summarize.json` | long summarisation prompt (short answer) -> prefill speed |
| `summarize_source.txt` | the article it summarises (~1300 words, original prose) |
| `decode_poem.json` | tiny prompt, long structured generation -> decode speed |

## Test file schema

```json
{
  "schema": 1,
  "name": "prefill_summarize",
  "goal": "prefill",                        // "prefill" | "decode"
  "instruction": "Summarize ...\n\n",      // user turn text
  "content_file": "summarize_source.txt",   // appended to instruction, or null
  "max_new_tokens": 160,
  "expect": { "min_prompt_tokens": 1200, "max_gen_tokens": 160 },
  "notes": "why the test is shaped this way"
}
```

`expect.*` is checked by the runner: a test whose prompt or generation drifts far from
the shape it was written for is not measuring what its name claims, so the runner
reports `SHAPE` instead of a silently meaningless number.

## Conventions (the part that makes results comparable)

From `index.json`, and enforced by `bench/run.py` for every engine:

- **greedy only** - argmax, no sampling, no seeds;
- **rendered as chat** with thinking off, one user turn, generation prompt appended
  (`specification/02.2`; the same rendering gate `run_tok.py` pins);
- **stop** at the first id in `{248046, 248044}` (A-10) or at `max_new_tokens`;
- **timing** - prefill is one forward over the whole prompt; decode is per-token
  forwards with the cache/state carried (no `generate()` shortcut that would fold the
  two together); weights are loaded before timing starts;
- the runner prints the prompt-id hash so two engines can be checked to have consumed
  the *same* tokens - different tokenisation invalidates a speed comparison.

Because greedy decoding is chaotic past a few tokens (A-18), these tests measure
**speed**, not output equality. Text is printed for eyeballing only.

## Running

```bash
./bench.sh                       # both engines, both tests
./bench.sh --engine speccheck    # one engine
./bench.sh --test decode         # one test, 1024-token generation
./bench.sh --reps 3              # repeat and take the median
./bench.sh --max-new 256         # override the per-test generation ceiling
```
`bench.sh` exports one `OMP_NUM_THREADS` for every engine so CPU thread count cannot
skew a comparison, and writes per-run JSON to `data/bench/`.

Two things the runner deliberately gets right, both of which will matter again when the
HIP engine is measured here:

- **the same amount of `lm_head` work.** speccheck applies the head to the final
  position only; HF would run it over all 2177 prompt positions, which is a different
  quantity, so `bench/run.py` passes `logits_to_keep=1` to HF. `--hf-full-logits`
  turns that off if you want to see the difference.
- **the same tokens.** HF's Qwen template applies `|trim` to message content, so
  trailing whitespace in a test is not part of the prompt there. The runner trims
  before rendering in both engines and then prints whether the engines consumed
  identical prompt ids - a speed comparison between engines that read different tokens
  is worthless.

The HF reference runs on torch fallback kernels on this machine (`causal_conv1d` and
`flash-linear-attention` are not installed and are CUDA-only), so its decode number is
a CPU reference number, not an optimised one. That is fine - it is the same
implementation that produced `data/refs`.
