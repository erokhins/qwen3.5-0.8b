# Engine

Two layers live here, and they are deliberately different beasts:

| file | role |
|---|---|
| `specimpl.py` | torch-CPU implementation of `specification/02..07`, written **from the spec**, not from transformers. It is the executable form of the spec: if it agrees with the HF reference, the spec is right. It is also the numerical contract that real kernels are gated against, and the fallback path while kernels are missing. |
| `gates/*.py` | §10 parity gates: engine tensors vs `tools/reference_dump.py` dumps. |

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
.venv/bin/python engine/specimpl.py --prompt-ids-file data/refs/smoke/manifest.json \
    --decode 2 --gdn-prefill chunk --dump data/eng/smoke_chunk.npz
.venv/bin/python engine/gates/run_gates.py     --ref data/refs/smoke --eng data/eng/smoke_chunk.npz
.venv/bin/python engine/gates/run_isolated.py  --fixture smoke          # G3i
.venv/bin/python engine/gates/run_chunkcheck.py --fixture p300          # G3 self-check
.venv/bin/python engine/gates/run_g5.py --split 64 --extend 4 --gdn-prefill chunk
```

`--gdn-prefill` picks S04.4 (recurrent token loop, the definition) or S04.7
(chunked UT-transform scan, the path kernels will implement); both are gated and
both must pass. Every gate exits non-zero on failure; `run_gates.py --filter SUB`
narrows to one layer/op while debugging.

## Status — all four fixtures, both GDN prefill forms

| gate | result |
|---|---|
| G0/G1 loader + shapes | PASS — 488 tensors → 335 loaded, 153 vision skipped, every shape asserted |
| G7 greedy tokens | PASS — identical to HF-ref on `smoke`, `p64`, `p300`, `dec8` (e.g. `760, 6511, 314`) |
| G2–G4 cascade tensors | PASS — 705 / 235 / 235 / 2115 tensors, worst `cos 0.99923`, worst `meanrel 3.3 %` |
| G3i isolated per-layer | PASS — worst `cos 0.99995`, and raw projections **bit-exact** (`meanrel 0`) |
| G3 chunk == recurrent | PASS — `cos ≥ 0.99999` per layer on 1-, 1- and 5-chunk fixtures (A-08) |
| G5 incremental == batch | PASS — splits 1,2,3,8,17,64,65,130; state, KV, logits and argmax agree |

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
