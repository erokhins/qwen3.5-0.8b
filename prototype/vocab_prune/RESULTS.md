# P3 — bound-based vocab pruning in `lm_head`: measured dead

**The idea.** `lm_head` reads the whole 486 MiB tied table every step to produce 248,320 logits when greedy
needs only the argmax. For row *r*, `logit_r = W_r · x`, so `logit_r ≤ ‖W_r‖·‖x‖` exactly
(Cauchy–Schwarz, equality iff `W_r ∥ x`). Walk rows in descending `‖W_r‖`, keep the running max of the
logits actually computed, and stop where the bound drops below it — every row below is *provably* not the
argmax, with no approximation anywhere. Prizing it: `lm_head` is 2.1685 ms and 46 % of the token's bytes.

**The measurement** (`.venv/bin/python prototype/vocab_prune/survival_study.py --fixture dec8`, CPU only,
no device):

| | value |
|---|---|
| vocab / hidden | 248,320 / 1024 |
| `‖x‖·max‖W‖` (the largest possible bound) | **145.704** |
| largest logit actually observed, over all 9 pinned rows | **26.625** |
| rows that must be computed before the bound falls below the running max | **248,320 of 248,320** |
| recoverable time on the `lm_head` line | **0.000 ms** |

The bound is ~5.5× looser than the truth at the top of the norm ordering — the high-norm rows are simply not
aligned with `x` — and the *minimum* row norm is still large enough that `‖W_min‖·‖x‖` never dips below the
observed max. So the walk never terminates early. Not a weak result: it is a **clean 100 % survival across
every pinned step**, which means no exact bound of this family can prune this table.

**What would be needed instead, and why it is not on the queue.** A tighter exact bound requires per-column
information (`Σ_j |x_j|·max_{r∈block}|W_{r,j}|`), which means reading the table to decide whether to read
the table — self-defeating. Anything else is a *statistical* threshold, i.e. an approximation that can
change the sampled token, and that is a numerics-band decision for the user rather than an optimization to
try. **P3 closes without any device time spent, for one CPU run's cost.**

**Two traps this study walked into and out of, on the record.** (a) `.npy` bf16 tensors will not `.numpy()` —
widen per chunk in torch, never materialise 508 MiB of f32. (b) The study initially paired
`pre_final_hidden.npy` with `logits_last.npy` and got **cos 0.245** — precisely the §10.5 last-writer-wins
trap this repo already documents (bare dump names are not a valid input/output pair). The fix was not to
pick the nicer-looking dump but to **search all (x, logits) pairs and require reproduction**, which is what
`survival_study.py` now does before it is allowed to print a rate. A guard that refuses is worth more than a
plausible number.
