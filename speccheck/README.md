# Speccheck (independent Python implementation)

This is **not** the engine. It is an independent implementation of
`specification/02..07` whose only job is to prove the specification (and later the
HIP kernels) correct — see `../engine/README.md` for the real engine.

Two layers live here, and they are deliberately different beasts:

| file | role |
|---|---|
| `specimpl.py` | torch-CPU implementation of `specification/02..07`, written **from the spec**, not from transformers. It is the executable form of the spec: if it agrees with the HF reference, the spec is right. It is also the numerical contract that real kernels are gated against, and the fallback path while kernels are missing. |
| `gates/*.py` | §10 parity gates: speccheck tensors vs `tools/reference_dump.py` dumps. |

Gate scripts: `run_gates.py` (G0–G4 tensor parity; stops at the first token flip),
`run_g7.py` (G7 sampled ids; `--corpus`/`--floor` for corpus agreement), `run_g6.py`
(blocked online softmax == reference softmax), `run_g5.py` (incremental == batch),
`run_isolated.py` (G3i per-layer from the reference's own input), `run_chunkcheck.py`
(G3 chunked == recurrent GDN), `run_negative.py` (injected bugs must be caught),
`run_tok.py` (A-11 chat render + tokenizer, A-10 stop ids), `run_mtp.py` (G8: MTP
drafter - accept rate across the three readings of §09, plus a bit-exact losslessness
check of the speculation loop). `textio.py` holds the
engine-side renderer/tokenizer contract; `run_all.py` runs the whole suite and exits
non-zero if anything fails.

### Interactive driver

`speccheck/chat.sh` runs a REPL on this path (torch CPU, bf16, greedy) and prints
per-turn stats: prefill tok/s, decode tok/s, implied weight-read bandwidth, stop
reason, prefix length and live state size. Multi-turn uses the §08 resume path — only
the new turn's tokens are prefilled — and `--no-resume` forces a cold full prefill to
compare. `--mtp K` turns on §09 speculative drafting (K draft tokens per target step,
greedy verification); each turn then reports `mtp k=K accept accepted/drafted = rate`
and `tok/forward` — the quantity that decides whether drafting pays — and `/stats` and
the goodbye line aggregate them. Example (this box, accept 0.66, 1.24 tok/forward):

    [prefill 23 tok 0.05s 459.8 tok/s | decode 39 tok 1.80s 21.6 tok/s |
     33.4 GB/s weight reads | stop=max_new | resume | prefix 117 | state 20.7 MiB |
     turn 1.9s | mtp k=1 accept 16/23 = 0.696 | tok/forward 1.30]

Numbers without drafting: ~250 tok/s prefill, ~25 tok/s decode, ~38 GB/s weight reads,
i.e. ~5× below the 210–215 GB/s the §07.4 roofline assumes. That gap is the
reason the engine is a separate implementation, and it is the baseline G9 replaces.
Note that with drafting the chat is bit-lossless but *slower* here (~21 vs ~24 tok/s)
for the reasons §09.5 spells out — the accept stats are there to make that visible, not
to claim a win.

Kernels (HIP/C++, the actual deliverable) will be added per-op under `../prototype/<op>/`
and graduate here once their §10 gate passes. Nothing here is performance code —
`specimpl.py` runs a 17-token prefill + 2 decode steps in ~3 s on CPU with a
Python loop over 17×16 GDN sub-steps.

## Bring-up commands

```sh
# fixtures (HF oracle): smoke/dec8 from the chat message, p64/p300 from prose
.venv/bin/python tools/reference_dump.py --message "The capital of France is" --decode 2 --tag smoke
.venv/bin/python tools/reference_dump.py --message "$(cat data/long.txt)" --decode 0 --tag p300

# engine tensors, then the gates
.venv/bin/python speccheck/specimpl.py --prompt-ids-file data/refs/smoke/manifest.json \
    --decode 2 --gdn-prefill chunk --dump data/eng/smoke_chunk.npz
.venv/bin/python speccheck/gates/run_gates.py     --ref data/refs/smoke --eng data/eng/smoke_chunk.npz
.venv/bin/python speccheck/gates/run_isolated.py  --fixture smoke          # G3i
.venv/bin/python speccheck/gates/run_chunkcheck.py --fixture p300          # G3 self-check
.venv/bin/python speccheck/gates/run_g5.py --split 64 --extend 4 --gdn-prefill chunk
```

`--gdn-prefill` picks S04.4 (recurrent token loop, the definition) or S04.7
(chunked UT-transform scan, the path kernels will implement); both are gated and
both must pass. Every gate exits non-zero on failure; `run_gates.py --filter SUB`
narrows to one layer/op while debugging.

## Status — all four fixtures, both GDN prefill forms

| gate | result |
|---|---|
| G0/G1 loader + shapes | PASS — 488 tensors → 335 loaded, 153 vision skipped, every shape asserted |
| G7 greedy tokens (`run_g7.py`) | PASS — ids identical on `smoke`, `p8`, `p64`, `p300`, `dec8`, `prefix24dec8` (full reference continuation, stop-on-EOS live) |
| G7 corpus (49 prompts × 24 tok) | PASS — speccheck 0.9124 vs the reference's own self-consistency floor 0.8776 (HF-sdpa vs HF-eager), so corpus equality is judged against the floor (A-18) |
| G8 MTP (`run_mtp.py`) | PASS — speculation is bit-lossless when the verify reuses the decode path (output *and* target state), and §09's reading beats both wrong readings on accept rate: 0.642 vs 0.232 (pre-final-norm hidden) and 0.014 (swapped `fc`) |
| A-11/A-10 tokenizer (`run_tok.py`) | PASS — independent renderer reproduces every fixture's prompt ids exactly; pins `248068` and `248069` by id, distinct from `248045` |
| G2–G4 cascade tensors | PASS — 705 / 235 / 235 / 2115 tensors, worst `cos 0.99923`, worst `meanrel 3.3 %` |
| G3i isolated per-layer | PASS — worst `cos 0.99995`, and raw projections **bit-exact** (`meanrel 0`) |
| G3 chunk == recurrent | PASS — `cos ≥ 0.99999` per layer on 1-, 1- and 5-chunk fixtures (A-08) |
| G5 incremental == batch | PASS — splits 1,2,3,8,17,64,65,130; state, KV, logits and argmax agree |
| G6 online softmax | PASS — ids identical, decode logits cos ≥ 0.9999, with an anti-vacuity check that the blocked branch ran (A-17) |
| G-neg negative controls | PASS — de-interleave (canary cos 0.855), conv write-before-read (0.975), GDN state not persisted (0.005); every clean canary ≥ 0.99998 |

## What the numbers say (read this before blaming a kernel)

With the reference's own layer input fed in (G3i), the engine reproduces the
reference: projections bit-exact, worst residual 5–8 ‰ on the attention output
before the output gate. End-to-end the same engine drifts 1.5–3.3 % `meanrel` by
the last layers. That gap is **bf16 rounding amplified by the residual stream**,
not a formula error, and it is a floor rather than a knob — three plausible fixes
were each implemented *and measured against the reference*, and none moved it:
chunked GDN prefill (§04.7), attention scores kept in the activation dtype, F32
accumulation in the causal conv. All three were kept because each is the
reference-faithful or kernel-realistic choice, not because it improved a number.

Practical consequence: **G3i is the gate a kernel is signed off against** (tight:
`cos ≥ 0.9999`), while the cascade gates exist to catch regressions and to keep
G7 token-exact. "Cascade noise" is not an available excuse for a kernel that fails
G3i. See `specification/10-reference-dumps-and-gates.md` §10.2 for the table.

## History

- G0 caught the first spec/code disagreement: 488 checkpoint tensors split
  **335 loaded / 153 vision-skipped**, not the 344/144 first assumed.
- G2–G4 caught a real engine bug on the first run: the GDN recurrent state `S`
  was not persisted across calls, so every decode step restarted from zeros while
  prefill looked perfect (cos 0.99999 prefill, −0.008 decode). Exactly the class
  of bug that is invisible without per-op state pins.
- A-01/A-07 rounding policy was pinned by measurement, not taste: HF applies the
  RMSNorm affine after one cast (`(x*w).to(dtype)`), the conv and the MLP gate
  product accumulate as the reference does — each verified against the dumps.
- G6 caught a **spec-reading** error, not a typo: `(max, sum)` accumulators were
  carried across decode steps. Each step has a new query, so that state belongs to a
  dead query — attention fell to cos 0.876 and matched the *previous* prefix. §07.3
  and A-17 now say explicitly that these accumulators are per-row reduction state,
  because a HIP kernel is one plausible design away from repeating this.
- A-11 caught a **bug in the specification itself**: §02 listed 248068/248069 as
  `<|im_start|>`/`<|im_end|>`. They are `<|im_start|>`/`<|im_end|>`, and because both render like
  an im-start in a terminal the error was invisible — the renderer inherited it and
  tokenization broke at id 13 of 17. §02 is corrected; literals are resolved by id.
- A-18: the corpus half of G7 could never pass as written (95 % absolute). The
  reference disagrees with a legal variant of itself at 0.8776 while speccheck reaches
  0.9124, so the gate is now relative to that floor, and tensor parity stops at the
  first token flip instead of reporting decorrelation as a failure.
- G8 **falsified A-12**: the MTP block takes the hidden state *after* the final norm,
  not before it (accept 0.642 vs 0.232). Nothing but the accept rate could tell us -
  there is no reference dump for this block. Two bugs in my own speculation loop fell
  out of making that measurement: the verify step kept re-feeding the first generated
  token (`t` was never advanced, so output became "Paris Paris Paris…"), and the
  rollback re-fed one token less than the invariant requires. Both are now pinned by a
  bit-exact state comparison against a sequential replay, not by eyeballing text.
