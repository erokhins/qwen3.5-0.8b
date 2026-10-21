
---

# Outcomes, recorded as executed (2026-10-19)

## R - regression vs `origin/master`: shipped, no regression, and the instrument got a noise floor

Executed as designed (paired families in one hold, canaries A/B/C, baseline through its own `gate.py`), result
in ledger **§37**. Headline: `attn` **1.996×**, wave `RESULT` **1.227×** (869.30 → 708.54 ms = 5,781 tok/s),
every untouched family inside **±3.1 %**, and the baseline reads 869.299 against §33's frozen 869.13 (0.02 %)
while HEAD reads 708.544 against §36's independent 708.14 (0.06 %).

Two things worth recording as decisions rather than results:
- **The ±3 % floor is now a rule of this ledger, not a mood.** Four families the round never edited came in at
  −1.8 %/−2.4 %/−3.1 %/−5 %-of-0.09 ms. That is what a paired window on this box cannot resolve, so from now on
  a sub-3 % move on a sub-30 ms row is reported as *nothing has moved*, in either direction.
- **Canary B paid for the whole harness on its first run**: the fallback ids were space-joined while `--ids` is
  comma-separated, so the 1,024-lane side had been feeding the engine ONE token and would have printed a
  perfectly plausible table of nonsense. D30's lesson, third occurrence: a check that a difference MUST exist
  is the only thing that catches a side that silently measured the wrong thing.

## C - llama.cpp by phase: the prediction was right, and it kills the attention reflex

Executed as designed (throughput rows + `rocprofv3` traces in one hold, `tools/trace_phase_split.py` deriving
pass counts from a kernel with a known per-pass site count), result in ledger **§38**. Matched-row ratio at
4,096 lanes: **1.80×** (was 1.87 in §21) - our +30 % against llama's own +25 % on the same build, which is
why the cross-window half of that sentence is worth nothing and only the within-window pairing is quoted.

The pre-registered **C5** prediction - "GDN carries the gap; if so the next bet is the GDN inner walk, not a
third attention pass" - is **confirmed, and by a cleaner witness than asked for**:
- GDN recurrence: **272.07 ms ours vs 47.20 ms llama, on 144 launches each** (identical chunking, 8/layer).
- Launches/token: **ours 0.81 vs llama 1.93.** Launch structure is dead as an explanation, permanently.
- Both engines run the recurrence at **0.80 blocks/CU** on a 20-CU part - the machine is 4/5 idle in *both*, so
  the gap is per-token step cost, not occupancy.
- Attention is now the **second** gap (136.83 vs 51.65 = 2.65×, 85 ms of the difference) where GDN is worth 225.
- We now **win GEMM** (154.19 vs 176.83 ms) - §21 had that phase 3.5× against us. A phase can flip; re-measure.

**Decision recorded:** the next funded campaign is the **per-token GDN step**, not attention, and not launches.
It is not "re-try the drawer": route A's collapse form was measured (112.04 vs 125.36 ms at the 1,024-lane
shape, 10.6 %, dead on its own kill line), so the brief must ask for a cheaper serial step or a form that fills
the idle CUs. Per this file's own clause, it gets a priced bet before implementation - and §4/§5's 11.4-vs-112
ms cost-model failure is why the pricing must be in cycles-per-dependency at realised occupancy, not
instructions per token.

Also recorded, because it changes how the yardstick is used: **llama.cpp's own rows move ±25 % between windows
on the same build** (matched row 8,290 → 10,381; `fa off` 8,999 → 5,414). Any future llama comparison is an
in-one-hold A/B or it is not a comparison.

## L - dispatched

Worker `2ada39b1-22c8-430f-9dc0-d9e2cced4201`, CPU-side: derive the partial-memory curve before choosing the
raise, implement `kAttnMaxSplit` + `attn_part_floats()` in one coupling, write `tools/longctx_check.sh` with
the five L2 checks (finish / finite / `plan_match=1` / no silent clamp / **prefix consistency at row N**),
report at `prototype/attn_wmma/evidence/longctx.md`. Device pass is the main agent's, after the worker lands.

---

**Archived 2026-10-21** from `progress/queue2-2026-10-19.md`, where it sat at the top of `progress/` instead
of in its dated folder, and under a name that collided with the document it answers. This file is the
**outcomes** half of the 2026-10-19 second batch; the **plan** half — the decisions R/L/C as they were raised
— is `progress/2026-10-19/queue2-2026-10-19.md` (its own sibling, unchanged). Rule 3 normally keeps a moved
filename byte-identical; here the collision is exactly the case rule 3 carves time-or-subject out of the
filename for, and no citation named the old path, so the rename costs nothing.
