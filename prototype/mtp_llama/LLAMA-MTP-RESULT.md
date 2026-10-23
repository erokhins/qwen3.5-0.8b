# llama.cpp's MTP on this box: it works, and it says where our round's cost is

**Question (asked 2026-10-23).** Does speculative decoding with this checkpoint's trained MTP head work in
`llama.cpp`, on requests like ours — and if it works there and not here, why?

**Answer, in one line.** Yes: llama.cpp's `--spec-type draft-mtp` is worth **+9.5 % to +13.6 % tok/s at
4,152 lanes** and **+18.7 % to +20.4 % at 362 lanes** on this box, peaking at `n_max = 2` and already falling
by `n_max = 4`; and it wins not on the accept rate (which agrees with ours) but because its verify round
costs **1.579 × a decode step** where ours costs **2.085 ×**, of which llama.cpp itself spends only
**0.078 step per extra row** in the verify body — the quantity our batched 2-row body cannot buy.

Everything below is from one harness, `prototype/mtp_llama/llama_mtp_ab.sh`, with evidence in
`evidence/llama_mtp_ab-tiny_short-2026-10-23.txt`, `evidence/llama_mtp_ab-short_long-2026-10-23.txt` and
`evidence/llama_mtp_accept_ladder-2026-10-23.txt`.

---

## 0. Setup, and the one thing that had to be built first

The GGUF already in `models/gguf/` **cannot** do MTP: it has 320 tensors and no `nextn` head, so
`--spec-type draft-mtp` has nothing to draft with. Re-converting this repo's HF checkpoint with llama.cpp's
own converter produced `tmp/mtp_llama/Qwen3.5-0.8B.mtp.bf16.gguf` — 335 tensors, `blk.24.nextn.{eh_proj,
enorm,hnorm,shared_head_norm,…}` — and llama.cpp then logs `creating MTP draft context against the target
model`, i.e. the head runs inside the same model file, no sidecar. The command, so this is re-takable and
not folklore (the file itself is under `tmp/` and so is gitignored):

```bash
PYTHONPATH=engines/llama.cpp/gguf-py .venv/bin/python3 engines/llama.cpp/convert_hf_to_gguf.py \
    models/Qwen3.5-0.8B --outtype bf16 --outfile tmp/mtp_llama/Qwen3.5-0.8B.mtp.bf16.gguf
```

No `--no-mtp` flag is needed or wanted: qwen35 exports its MTP head by default, and the absence of `nextn`
tensors in `models/gguf/Qwen3.5-0.8B.bf16.gguf` predates that default rather than opting out of it.

| item | value |
|---|---|
| binary | `engines/llama.cpp/build-hip/bin/llama-cli`, build `b1-797de63` (b11716), ROCm/HIP |
| arms | `none`, and `draft-mtp` at `--spec-draft-n-max` ∈ {1,2,3,4} |
| fixed | `-ngl 99 -fa on --temp 0 -s 42`, `-n 128`, one lock hold per shape, medians of 2, blocking |
| prompt parity | every arm of a shape prefills the identical token count (`prompt tokens seen: 10 4152`) |
| shapes | `tiny` 15 tok, `short` 362 tok (`speccheck/fixtures/corpus.txt`), `long` 4,152 tok |

Two things make this more than a `tok/s` pair. llama.cpp logs its **own** accept counters, and its
statistics line carries the draft time **separately** from the verify time (`dur(b,g,a)`). With
`rounds = predicted_n − accepted_tokens` (each round emits its bonus token plus its accepted ones), the
round cost comes from the program's own arithmetic and not from a wall-time guess — which is the same
quantity `prototype/mtp_accept/M3-RESULT.md` §4 measures for our engine, so the two are comparable.

---

## 1. The measurement

`round/step` = eval-ms per round ÷ eval-ms per step, both medians from the same harness; `draft` is
llama.cpp's `dur(g)` per round; `verify` = round − draft; `tok/round` = emitted ÷ rounds. Parity needs
`tok/round ≥ round/step`, since a round is the only thing a step is not.

**shape = short — 362 lanes, step 10.566 ms, base 94.639 tok/s**

| arm | tok/s | vs base | round ms | draft ms | verify ms (steps) | round/step | tok/round | accept ladder |
|---|---|---|---|---|---|---|---|---|
| `none` | 94.639 | — | 10.566 | — | — | 1.000 | 1.00 | — |
| `mtp1` | 112.341 | **+18.7 %** | 16.048 | 3.337 | 12.711 (1.203) | 1.519 | 1.80 | 0.826 |
| `mtp2` | 113.956 | **+20.4 %** | 20.423 | 6.567 | 13.856 (1.311) | 1.933 | 2.33 | 0.774, 0.604 |
| `mtp3` | 110.531 | +16.8 % | 25.175 | 9.610 | 15.565 (1.473) | 2.383 | 2.78 | 0.818, 0.591, 0.455 |
| `mtp4` | 104.688 | +10.6 % | 28.434 | 12.724 | 15.710 (1.487) | 2.691 | 2.98 | 0.829, 0.561, 0.415, 0.268 |

**shape = long — 4,152 lanes, step 10.865 ms, base 92.038 tok/s** (the row that matters, per the standing
4k reporting rule)

| arm | tok/s | vs base | round ms | draft ms | verify ms (steps) | round/step | tok/round | accept ladder |
|---|---|---|---|---|---|---|---|---|
| `none` | 92.038 | — | 10.865 | — | — | 1.000 | 1.00 | — |
| `mtp1` | 100.797 | **+9.5 %** | 17.161 | 3.423 | 13.738 (1.264) | 1.579 | 1.73 | 0.750 |
| `mtp2` | 104.576 | **+13.6 %** | 21.474 | 6.778 | 14.696 (1.353) | 1.976 | 2.25 | 0.750, 0.518 |
| `mtp3` | 103.836 | +12.8 % | 25.681 | 10.012 | 15.669 (1.442) | 2.364 | 2.67 | 0.723, 0.532, 0.447 |
| `mtp4` | 98.462 | +7.0 % | 29.545 | 13.266 | 16.279 (1.498) | 2.719 | 2.91 | 0.698, 0.512, 0.419, 0.326 |

Two laws fall straight out of the four `n_max` points, in both shapes independently:

- **drafting is serial and context-free**: `(13.266 − 3.423) / 3 = 3.28 ms` per extra drafted token at
  4,152 lanes, `3.13 ms` at 362 — i.e. **≈ 0.30 of a decode step, per token drafted, forever**. There is no
  batching a draft: token *i+1* needs token *i*.
- **the verify body is nearly flat in rows**: verify per round = 1.264, 1.353, 1.442, 1.498 steps for 2, 3,
  4, 5 rows, a slope of **+0.078 step per extra row** (short shape: +0.094). A one-row pass extrapolates to
  ≈ 1.1 steps. That is the whole economics of speculative decoding in one number.

So the gain peaks at `n_max = 2` because of both halves at once: the accept ladder decays (0.750 → 0.518 →
0.447 → 0.326) while each drafted token costs a flat 0.30 step and each verified row costs 0.078.

---

## 2. Why it works there and not here — and what the data *rules out*

Our own loop, same box, same checkpoint, greedy, measured twice now
(`prototype/mtp_accept/mtp_wall_ab.sh` medians of 5, and `prototype/mtp_accept/verify_body_scale.sh` medians
of 2, `evidence/verify_body_scale-2026-10-23.txt`):

| width | our step | our round | our round/step | llama.cpp round/step at the same width |
|---|---|---|---|---|
| short (17–362 lanes) | 9.349–10.566 ms | 15.804–16.658 ms | **1.691 / 1.722** | 1.519 |
| 1,024 lanes | 10.304 ms | 18.400 ms | **1.786** | — |
| 4,032 lanes | 11.223 ms | 23.398 ms | **2.085** (wall A/B: 2.029) | **1.579** |

`verify_body_scale.sh` was written to find out whether the round's cost is the *body* or the *bookkeeping*
around it (the 19.27 MiB mid-round checkpoint, the verdict kernel, the 16 B readback, the draft), and the
answer came back in a form nobody would have guessed:

| lanes | one-row step | `--mtp-verify 1` side pass | accept round |
|---|---|---|---|
| 128 | 9.672 ms | 25.010 ms (marginal **1.586** steps) | 16.658 ms |
| 1,024 | 10.304 ms | 27.691 ms (marginal **1.687** steps) | 18.400 ms |
| 4,032 | 11.223 ms | 33.476 ms (marginal **1.983** steps) | 23.398 ms |

Three conclusions, in order of how much they cost to learn:

1. **The accept rate is not the difference.** llama.cpp's first-position rate on real prose is
   0.750–0.826; our corpus rate is 0.6556 — the same order on different prompt/response pairs (llama.cpp's
   outputs here are chat-template thinking prose, which is more stereotyped than ours). Our loop is not
   guessing worse; it is paying more.
2. **The bookkeeping is not the difference either — and it is bounded from the other side.** llama.cpp does
   the same jobs (draft, verify both rows, accept/reject, one context) inside **1.579 steps**, so the
   checkpoint + verdict + readback cannot be what costs us 1.0 extra step. And on our side the *read-only
   side pass*, which runs the same 2-row body with **no** checkpoint, **no** verdict kernel and **no** draft,
   already costs ≈ 2 steps at 4k. **The cost is in the body.**
3. **What costs the extra step is the verify body's ROUTE, not its geometry.** — *corrected the same day by
   `prototype/mtp_accept/ROW-SCALE.md`, which is the file to believe.* This section originally read the growth
   (1.722 → 1.786 → 2.085) as the signature of per-row traffic, since llama.cpp's 2-row verify grows only
   1.203 → 1.264 steps for a 37× growth in KV. Priced directly, the same `submit_wave_body` at 2–8 rows over
   a populated 4k cache costs **1.09–1.13 of a step** on the amortising route — cheaper per row than
   llama.cpp's 1.26–1.50 — so our engine can amortise fine and the accept path simply is not allowed to: the
   verify body forces `WaveGemm::kHand` + `GdnScan::kSeq` so row 0 stays bit-comparable to the one-row step,
   and that forcing is what costs the ~1.0 step. The observation that motivated it (round cost grows with
   context) stands; the mechanism named here first did not, and the correction is the useful part.

**And a carried number dies with this.** `prototype/mtp_delta/D3-RESULT.md` priced plan A — the batched
2-row body — at **0.1505 of a step**, and that bill is what funded P7 and M2. Measured on this box the same
body's marginal is **between +0.72 and +1.98 steps** (round − step at 4k = 1.086; side-pass − step at 4k =
1.983, the upper bound, because the side pass also pays a shadow-state copy and an argmax report). Either way
it is an order of magnitude above the bill, and the bill is what the route was priced on. The reason the bill
missed is now measured rather than guessed (`ROW-SCALE.md`): the geometry is cheap, and what is not cheap is
the hand-route forcing the verify body carries. This is the same
failure as `prototype/gdn_collapse/results.md` §4–§5 (a bottom-up instruction bill 9.8 × under the machine):
a static launch bill cannot see that a width-2 wave re-pays a weight walk and a KV pass. It is recorded in
the traps index.

---

## 3. What this decides, and what it does not

- **Nothing is adopted and no band moves.** `--mtp-accept` stays default-off; no engine file changed for this
  result; `engine/performance-status.md` is untouched. llama.cpp's numbers are a *reference point*, not our
  ledger, and its base decode (~92.0 tok/s at 4,152 lanes) and ours (88.6–89.3 today, 94.2–94.9 in the
  ledger) are different windows and are not a paired claim in either direction.
- **The route is alive, and it has exactly one object.** M2's accept loop does not need more accept-loop
  work, and it does not need a better-drafted token: it needs the round to run on the route the rest of the
  engine already uses. Measured per-row cost on that route is **+0.25 step for a second row short, +0.43 at
  4k, and ~0 beyond that** (`ROW-SCALE.md` §2), so a 3-row k=2 round sizes at **≈ 1.3–1.4 steps** and
  **+23 % to +45 %** at 4k on plausible accept rates — which is what `ROW-SCALE.md` §4 pre-registers lines
  against. That is a sizing, not a forecast: it presumes the route switch can be made without losing whatever
  row-0 bit-comparability the loop's admission needs.
- **The next measurement is named, and it is small.** Attribute the width-2 body by kernel family. The
  obvious instrument refuses: `decode_run.cpp:1157` refuses `--mtp-verify` together with `--time-stages`,
  because a side pass has no honest per-step stage bill. So the cheap version is the one already here:
  run the `verify` arm at fixed total work across `--wave-rows`/widths and see which stage's launch count
  doubles — or add one stage-clock-eligible body arm. Either way the question is now specific: *which family
  of the 2-row body re-reads its operands.*
- **`n_max` is a free lesson for the design.** Our loop is k=1. llama.cpp's curve says the win on this model
  is at k=2 (+13.6 % vs +9.5 % at 4k), and that k=4 is already worse than k=1 on both shapes. A k=1-only
  route is therefore capped in the low single-digit tens of percent even with a perfect body; whether to
  build k=2 is a scope decision for the user, not for this file.
- **Caveats, named.** llama.cpp ran through its chat template with thinking enabled, so its emitted streams
  are not ours and its accept ladder is a little flattered by stereotyped output; its `dur(g)` is host-timed
  around its own draft calls; the `tiny` shape is in the evidence only as a warning (+45.6 % on "The capital
  of France is" is a fact about that sentence); and all llama.cpp figures are medians of 2 with `--log-verbose`
  on both arms (uniform logging; a non-verbose base run at the `tiny` shape read `Generation: 92.5 t/s`
  against the verbose harness's 95.06/94.87, so logging is same-order on both arms but is not proved free,
  and the MTP arms print ~1 trace line per draft call where base prints none).
