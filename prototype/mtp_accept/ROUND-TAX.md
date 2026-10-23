# The round tax: two vocabulary-sized round trips that should never leave the device

Measured basis: `ACCEPT-APPLES.md` addendum, meter on the templated 4,152-lane fixture, loop route, per round:
`body 11.914 + head 2.204 + draft 3.639 + verdict_kernel 0.308 + readback 0.014 + rollback 0.037 = 18.115 ms`,
against llama.cpp's `draft 3.423 + verify 13.738 = 17.161`. Normalised to each engine's own step the body and
head terms already match llama.cpp (1.268 vs 1.264, 0.327 vs 0.315), so the round is not slower because the
math is slower — it is slower because of two places where a **whole vocabulary crosses a boundary** that
llama.cpp does not cross. Both are fixable with a byte-identity guarantee, not a band.

## 1. The draft argmaxes on the HOST, over a full-vocabulary blocking D2H

`engine/mtp_draft.cpp:348-355`:

```cpp
std::vector<float> host(kVocab);
MTP_CK(hipMemcpy(host.data(), st.logits, kVocab * sizeof(float), hipMemcpyDeviceToHost));
p.token = argmax_greedy(host.data(), kVocab);
```

**993,280 bytes downloaded per round, and 248,320 floats scanned on a CPU core, on a blocking sync** — inside
the phase the meter bills as `draft`, i.e. up to 0.5–1.0 ms of the 3.639 ms is this and not the draft block.
llama.cpp resolves its draft argmax in its graph and never pays it.

Fix: one device argmax over `st.logits` and read back **one int** (the verdict machinery already does exactly
this shape for the body's rows: `verdict_readback` is 16 B and costs 0.0137 ms). `top_logit` comes back beside
it if the echo needs it; `logits_out` (a debug-only out-parameter — the side-pass dumper) can keep its D2H,
guarded by the caller that asks for it, so no shipping round pays it.

The correctness argument is not "close": `argmax_greedy` and `detail::better` are the same total order —
strict `>` on value, ties to the **lowest index** — and that order is total, so *any* reduction order over the
same elements returns the same id. The proposal is therefore provably identical, and the admission is a
byte-compare of `emitted.txt` + every `ACCEPT` line against the pre-change binary, not a tolerance.

## 2. The verdict kernel is one block per row scanning 248,320 floats with scalar loads

`engine/mtp_verdict_kernels.h`: `kVerdictThreads = 256`, `gridDim.x = nrows`, each thread strides by 256
through `vocab` → **970 sequential iterations per thread, one CU busy out of 40**. 1.987 MB of logits for
0.308 ms is **6.5 GB/s against this box's measured 237 GB/s ceiling** — a 36× miss, and the logits the kernel
reads were just written by the head pass, so they are largely in the 32 MiB big L2 tier anyway.

Fix: split each row across `nchunks` blocks (grid `nrows × nchunks`, `float4` loads inside a thread's strip),
one tiny second kernel combining `nchunks` partials per row with the same `better()` comparator. Same
order-independence argument as (1) — a total order makes the split arbitrary — so verdicts stay byte-identical.
Expect ~0.03–0.05 ms including the extra launch (launch overhead here is single-digit µs; `hipEventElapsedTime`
cannot even resolve it).

Do **not** touch `rollback`: the meter's own note says it "INFLATES HARD — these are `hipMemcpyAsync` today and
the round lets them overlap the next draft, so the meter's wait serializes a copy the real round hides". Its
0.037 ms/round average is an artifact of being metered, not a cost to remove.

## Why this is the right next spend

At 4,152 lanes the round is 18.42 ms and the step 11.14, so every 0.1 ms off the round is **+0.55 %** on the
MTP arm's throughput. The two items above are plausibly 0.5–0.9 ms between them, which is the difference
between today's **+2.85 %** and roughly **+6 to +7 %** — and it lowers the k=2 bar in the same move, because a
k=2 round pays both of these *twice* per unit of accepted text (its draft runs twice). Current k=2 price
estimate is ~2.03 step; this tax is what makes it 2.0.

Pre-registered success line, fixed before the run: **the change is adopted if `round/step` at 4,152 lanes drops
by ≥ 0.03 (≈ 0.33 ms) with every `ACCEPT` line and every `emitted.txt` byte-identical to the pre-change
binary on the templated fixture, the raw fixture and `dec8`, and `ENGINE GATE PASS`.** If the byte-compare
fails, the argmax rule is not what this file claims and the work is reverted, not banded.

## Order of work

1. **Instrument first** (main agent, exclusive window): meter phases `draft_vocab_d2h` and `draft_argmax_host`
   splitting the `draft` phase's 3.639 ms, so the prize is measured before it is optimized — this repo has been
   wrong about the size of a prize before (`BEAT-MTP.md`'s B0 lesson).
2. (1) draft argmax on device. Same commit as its byte-compare witness.
3. (2) verdict kernel split. Same commit as its byte-compare witness.
4. Re-take `mtp_wall_ab.sh` / the templated-fixture three-arm window and update the ledger only if the
   adopted line is cleared.
