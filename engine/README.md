# Engine

Two layers live here, and they are deliberately different beasts:

| file | role |
|---|---|
| `specimpl.py` | torch-CPU implementation of `specification/02..07`, written **from the spec**, not from transformers. It is the executable form of the spec: if it agrees with the HF reference, the spec is right. It is also the numerical contract that real kernels are gated against, and the fallback path while kernels are missing. |
| `gates/run_gates.py` | §10 parity gates: engine tensors vs `tools/reference_dump.py` dumps. |

Kernels (HIP/C++, the actual deliverable) will be added per-op under `../prototype/<op>/`
and graduate here once their §10 gate passes. Nothing here is performance code —
`specimpl.py` runs a 17-token prefill + 2 decode steps in ~3 s on CPU with a
Python loop over 17×16 GDN sub-steps.

## Bring-up commands

```sh
.venv/bin/python tools/reference_dump.py --message "The capital of France is" \
    --decode 2 --tag smoke                     # oracle dumps -> data/refs/smoke/
.venv/bin/python engine/specimpl.py --dump data/eng/smoke.npz    # engine tensors
.venv/bin/python engine/gates/run_gates.py --json data/gates/smoke.json   # §10 gates
```

`run_gates.py` exits non-zero on any failure; `--report-only` prints numbers
without failing, `--filter SUBSTR` narrows to one layer/op while debugging.

## Status (smoke fixture)

`PASS: 705 tensors within §10 thresholds`; greedy tokens identical to HF-ref
(`760, 6511, 314`). Worst drift: `L21.gdn.gated_out` cos 0.99928, meanrel 3.3 %
(depth-accumulated bf16 + recurrent-vs-chunk GDN difference; layer 0 is 8e-6).
Not compared by convention: `gdn.conv_state` (HF caches `k_conv` columns, engine
keeps the `k_conv-1` history of S08.2) and HF's raw cache tensors, both covered
by G5 instead.

## History

- G0 caught the first spec/code disagreement: 488 checkpoint tensors split
  **335 loaded / 153 vision-skipped**, not the 344/144 first assumed.
- G2–G4 caught a real engine bug on the first run: the GDN recurrent state `S`
  was not persisted across calls, so every decode step restarted from zeros while
  prefill looked perfect (cos 0.99999 prefill, −0.008 decode). Exactly the class
  of bug that is invisible without per-op state pins.
