# GDN and attention: what the §38 gap is made of, and the two bets it funds (2026-10-19)

Written before any implementation, because that is the rule this repo's own failures enforce: §4/§5 of
`prototype/gdn_collapse/results.md` priced a route at 11.4 ms and measured 112.04 ms, and AGENTS.md's
conclusion — *"price a serial kernel in cycles-per-dependency at its realised occupancy, or do not price it"*
— is the standard this document tries to meet. Every number below is from an artifact already on disk:
`bench/performance-status.md` §37 (family medians), §38 (the two-engine phase trace),
`prototype/gdn_walk/evidence-tokenbill-2026-10-19.txt` (per-token ISA bill of the shipping walk), and
`prototype/gdn_rec_isa/token_bill.py`, which is the tool that already did this split correctly.

## 1. What the GDN gap is NOT (three levers closed before spending on them)

| candidate | evidence | verdict |
|---|---|---|
| launch structure | 144 launches/pass on **both** engines (§38); we launch 0.81/token, llama 1.93 | **dead**, §38 |
| state traffic (put the state in LDS) | route A measured **112.04 vs 125.36 ms** = 10.6 %, and died on its own 84 ms kill line at the 1,024-lane shape | **already tried, small** |
| loop unrolling / instruction scheduling inside the walk | `#pragma unroll 8` on the serial `i` loops: bit-identical, **112.8 → 113.4 ms** | **dead**, §5 of gdn_collapse |
| chunked / parallel scan over the sequence | priced ≥1.5×, measured **5.9× worse** | **dead**, AGENTS.md |

## 2. What it actually is: per-token bill of the shipping walk

`token_bill.py` on a fresh `-O3 gfx1151` build of `engine/kernels_gdn.cpp`, kernel `rec_loop_kernel`
(the `--gdn-loop 512` form that ships), per wavefront, grid `16 × 128` = 64 wavefronts:

```
PER-TOKEN bill        : 1,327 slots/wavefront
walks per token       : loopA 44×8 = 352,  loopB 90×8 = 720     -> 1,072 slots = 81 % of the bill
per-TOKEN straight    : 255 slots, of which
    xlane.exec-guard 57 | salu.mov 42 | loop.salu 25 | xlane.permute 20 | xlane.add 19
    norm.expf 16 | barrier.sync 5 | barrier.invalidate 5 | xlane.lds-round-trip 5 | range-select 7 ...
ADDRESS ARITHMETIC    : 93 slots = 7.0 % of the per-token bill
```

Two readings, and the second is the one that matters.

**(a) The per-token cross-lane norm machinery is worth ~9 %, not 5.8×.** The per-token straight-line part that
exists only because `rec_loop_kernel` computes the k- and q-row L2-norms *inside* the recurrence is
permute 20 + add 19 + exec-guard 57 + LDS round-trip 5 + barrier 10 + range-select 7 + addr-math 4 ≈ **122 of
1,327 slots = 9.2 %**. Anyone pricing a norm-hoist above ~10 % of `gdn_rec` is pricing something the ISA does
not contain.

**(b) The work-volume difference cannot explain the time difference, so the gap is issue efficiency.** Total
slots per token: ours 64 wavefronts × 1,327 = **84,928**. llama's kernel is 32 threads × 512 grid with
**LDS = 0** (§38's `--geom` census), so 16 wavefronts carrying the same 262,144 state elements, i.e. ~512
elements/thread at ~7 slots each ≈ 57,344 slots/token. **They do ~1.5× fewer slots and finish in 1/5.76 of the
time.** The only way that happens is that we are not issuing: at **0.80 blocks/CU** a 4-wavefront block puts
**3.2 wavefronts per CU, 0.8 per SIMD**, so when a wavefront stalls on a dependent global load — and the walk's
chain is `global k,q → shuffle tree → barrier → LDS write → barrier → LDS read → 128-dependent state FMAs`
— there is literally nothing else for the SIMD to run. `norm.expf 16` and `barrier 10` per token are the
symptom; the 3.2 wavefronts/CU is the disease.

That reframes the whole item: **the prize is in removing serialization from the per-token dependency chain and
in getting more independent work per SIMD**, not in deleting instructions.

## 3. Bet G1 — hoist the per-token norms out of the walk (cheap, bounded, bit-identity required)

`rec_loop_kernel` re-derives, every token, in every head block: two `block_sum_pair_f32` reductions (10+8
`__shfl_down_sync`, one barrier inside), two LDS writes of `ks`/`qs`, then two more barriers before the state
walk may start. The norms of `k` and `q` are **not part of the recurrence** — they are a row-wise map over the
same tiles the `gdn_gates`/`conv` pass already reads — and the walk cannot begin until they land.

Change: compute `knorm[h]`, `qnorm[h]` (or the normalized rows themselves) in an existing per-token pass, and
have the walk read them. What it removes per token: **3 `s_barrier`, 18 shuffles, the LDS round-trip, and the
whole `xlane.*` guard chain**, and — the actual point — the two barriers that separate token *t* from token
*t+1*, which is what forbids the compiler from software-pipelining the walk across iterations.

Hard requirement: **`o` must stay bit-identical to the pins.** `block_sum_pair_f32`'s summation association is
what makes the current bits (AGENTS.md: a hoisted scalar constant moved 93,522 of 131,072 `o` elements while
the state stayed exact), so the hoisting pass must reproduce that association step for step, not "equivalently"
re-reduce the row. If bit-identity cannot be had, this becomes a numerics-band item and stops for the user.

Pre-registered lines, measured by the main agent at 4,096 lanes against §37's **266.24 ms** baseline, same
harness, one lock hold, best-of-3:
- **live** if `gdn_rec` ≤ **250 ms** (≥ 6 % — above the −1.8/−2.4/−3.1 % spread §37 measured on untouched
  families, so a win at that size is not window noise);
- **dead** if > **258 ms** (inside §37's noise floor, i.e. indistinguishable from nothing);
- in between: report as unresolved and do not ship.
The slot bound says 9.2 % is the ceiling *if the machine were issuing*; the latency argument says it could beat
that. Both are stated so the measurement cannot be narrated after the fact.

## 4. Bet G2 — the geometry llama's trace shows (larger, and the only one with 5.8× of room)

One wavefront per block, **no LDS, no `__syncthreads()` at all**, per-thread ownership of several state columns
so the dependent chain per thread has ILP, reductions done by warp butterfly (legal because a head row is
128 elements = 4 per lane in a 32-lane block). That is structurally what `gated_delta_net_cuda` is: 32 threads,
`LDS_Block_Size = 0`, `VGPR = 80`, 16 blocks, 0.80 blocks/CU, 5.76× faster than us at the same launch count.

Cost honestly stated: each thread then walks 4× the column triples, so the walks grow in iterations per thread
while the barriers disappear; and the association of the state dot changes, so `o` moves (§10.1: a chained dot
is judged by cos/meanrel aggregates with a cancellation band sized by a floor probe, **not** by max-ULP). This
is a real rewrite and it is where the size of the gap lives: `gdn_rec` is **41.96 % of our wave at 4,096 lanes**
and 266 ms of a 628 ms wave.

Pre-registered lines at 4,096 lanes: **live ≤ 180 ms** (1.48× on the family, ≈ −86 ms of wave = +15 % prefill
tok/s), **dead > 230 ms**. Numerics: the §10.1 aggregate floors plus the existing `engine/gate.py` admission,
no band widening, no `--ack-spill`.

## 5. Attention: attribute before funding — the 2.65× may be three different problems

§38 puts our attention at **136.83 ms traced vs llama's 51.65** (180 launches vs 48), but that is a family
sum. Our family is at least three kernels — `attn_prep` (48/pass), `attn_fattn_wave` (42/pass, **122.72 ms —
90 % of the family**) and `attn_combine` (42/pass) — and the round that just landed made the middle one 1.996×
faster, so any claim about "attention" as one object is already stale. Post-round our attention sits near
~3.2 TFLOP/s against the WMMA-class roof of **52.63 TFLOP/s** (`prototype/issue_rate_probe/results.md` §1),
i.e. ~6 % of the roof — which says attention here is *not* flop-roof-bound either, and the same
attribution question has to be asked of it: staging, or softmax, or combine.

So the attention item is funded as **A0, an attribution step, not an implementation**: split the 133 ms across
those three kernels on device (`--time-stages` plus the §38 trace's per-kernel ms), price each against the roof
ladder, and only then fund one of them. No third tile retune until A0 says which kernel holds the time — the
§20 lesson (traffic claims must be read out of the source's staging loops, not asserted) applies to whatever
A0 finds.

## 6. Order, and who runs what

1. **G1 first** — one pass change, bit-identity is the admission, bounded at ~9 %+latency, cheap to kill.
2. **A0** while G1 is measured — CPU-side parse of traces I already have, plus one short device window.
3. **G2** last and only if G1 says the serialization hypothesis is real (G1 removing the barriers and moving
   < 6 % would be evidence that barriers were not the stall, which would re-rank G2's premise).

Device work is the main agent's, blocking, zero subagents, one lock hold, per `AGENTS.md`; workers write
kernels, gates and the CPU-side mirrors and hand over command lines. The numbers in §3/§4 are **pre-registered
kill lines**, and a result between the lines is reported as unresolved rather than narrated.

## 7. A0, half of it, taken the same day from the trace already on disk

The attribution step did not need a new device window: §38's own trace already has per-kernel rows. Our
attention family at 4,096 lanes, traced, per pass:

| kernel | ms/pass | launches/pass | ms/launch |
|---|---|---|---|
| `attn_fattn_wave(CoreArgs)` | **122.72** | 42 | **2.92** |
| `attn_slim_wave(CoreArgs)` | 5.27 | 48 | 0.11 |
| `attn_combine(CoreArgs)` | 5.19 | 42 | 0.12 |
| `attn_prep(CoreArgs)` | 3.65 | 48 | 0.076 |
| family total | 136.83 | 180 | — |

**90 % of the attention family is one kernel.** So "attention is 2.65× behind llama" is really the sentence
"`attn_fattn_wave` is 2.65× behind `flash_attn_ext_f16`", and the 12.1 ms of prep+combine+slim is not where a
bet can be won. Two things follow, and neither is a tile retune:

1. **Rate, not ratio.** D56 priced the frozen attention path at **1.59 TFLOP/s** = 7.1 % of the mix roof. The
   round that just landed made the family 1.996× faster (§37), so the shipping figure is **≈3.17 TFLOP/s**, i.e.
   **6 % of the WMMA-class roof of 52.63 TFLOP/s** (`prototype/issue_rate_probe/results.md` §1). llama's traced
   51.65 ms — an *upper* bound on its real time, since traced ≥ real — implies **≥8.1 TFLOP/s**, ~15 % of the
   same roof. Both engines are nowhere near the matrix roof, which means the remaining gap is *not* a flop-roof
   problem in either engine, and any bet expressed as "get closer to the WMMA roof" is unfunded by construction.
2. **The ladder already names the one compile-only question.** `prototype/attn_wmma/fattn_regfile_ladder.md`
   E1: the shipped 3-warp body is **LDS-bound at 6 wf/CU** (178 VGPR, 0 spill, 32,768 B LDS), and the family's
   own ceiling formula puts 7 at NW=7. E1 asks whether **un-staging Q** from the shared stage (drop the `s_q`
   fill, read Q from global at the use site, keep `s_kv` and the 3-warp geometry) yields `spill = 0` and
   **VGPR ≲ 190** — and the ladder states the admission exactly: that pair *means* an 8+ wf/CU zero-spill shape
   exists and is worth a device A/B; `spill > 0` closes the residency axis at 6.

**A1 is therefore E1, and it is a CPU task.** The ladder is explicit that the old un-stage patch no longer
applies (4 of 5 hunks fail; it is a **hand port**, not an apply), and that E1's evidence is a *note-block read*
— `vgpr_count / vgpr_spill_count / private_segment_fixed_size / group_segment` from one `-O3 gfx1151` compile.
No device, no timing, worker-legal by the letter and the spirit of the GPU rule. Its two outcomes:

- **`spill = 0` and VGPR ≲ 190** → the main agent runs the device A/B, and *that* is pre-registered now:
  against §38's traced 122.72 ms / §37's untraced family 132.85 ms, **live if the family drops ≥ 8 %** (the
  ladder's own warning that "residency is not speed" is exactly why this line exists — §31 bought 4→8 wf/CU for
  +20 % of the family *while spilling 50 registers*, and §33 got 1.217× back by *deleting* scratch at lower
  residency); **dead if inside §37's ±3 % floor.**
- **`spill > 0`** → the residency axis is closed at 6 wf/CU, A1 ships nothing, and the attention money moves to
  the instruction/traffic gap §26 sized at 4.6×, which is a different and larger piece of work.

**One thing A0 does NOT establish, recorded so nobody invents it:** why the traced launch counts are `slim 48 /
fattn 42 / combine 42` at 4,096 lanes — i.e. which waves take the one-pass path and which take the streaming
WMMA path — is not derived here. A1's brief must read that out of the launcher (`prototype/attn_decode/
attn_decode_kernels.h` owns the row-block rule since Bet 2) before pricing anything about work split, because a
bet on "the fattn kernel is slow per launch" is a different bet from "too many launches go through it".

## 8. G1 measured (2026-10-19): **UNRESOLVED** — do not ship, and the line itself was the wrong ruler

`--gdn-norm hoist` is implemented (`engine/gdn_rec_hoist_kernels.h`, default stays `inline`), it is
**bit-identical on device** — `base == inline == hoist` on every prefill logit row *and* every `STATEFP`
state fingerprint, which is the admission §3 demanded — and the A/B ran in one `gpu_lock` hold with canaries
A (baseline rejects `--gdn-norm`) and B (each side echoes its form and submits 144 vs 288 planned launches).
Medians of 3 at 4,096 lanes, logs in `progress/bench/evidence/g1ab-2026-10-19/`:

| quantity | base | new `inline` | new `hoist` | hoist vs base |
|---|---|---|---|---|
| **`gdn_rec` (the pre-registered row)** | 266.60 | 264.21 | **250.84** | **−5.91 %** |
| every other family, summed | — | — | — | **+6.06 ms** |
| wave `wall_ms` | 626.75 | 623.45 | 612.87 | **−2.21 %** |
| `RESULT ms_total` | 717.27 | 714.87 | 707.88 | −1.31 % |

The inline-vs-base null is **−0.89 %**, so the shipped path did not move and the hoist number means what it
says. The hoist side's own rep spread is 3.5 % (246.49 / 250.84 / 255.09), and it landed **0.84 ms above the
250 ms live line** — a fifth of its own spread. §3 said in-between is *unresolved and not shipped*, and that is
the verdict: **not shipped, default stays `inline`, no narration.** Two things come out of it that are worth
more than the number.

**The row was the wrong ruler, and this is the correction to carry into G2.** Hoisting removes work from
`gdn_rec` and *puts traffic into the stage*, which is paid in `gdn_gated_norm` (+15.6 %) and `glue` (+8.8 %)
— `+6.06 ms` outside the row that improved, against `−15.75 ms` inside it. So a pre-registered line on one
family row **over-counts a change whose cost lands in its neighbours**, and the honest whole-change figure is
**wave `wall_ms` −2.21 %**. G2's lines must therefore be written on *both* quantities before it is measured:
a family row to catch "did the walk get cheaper" and `wall_ms` to catch "did the engine get faster". That is
a methodology fix, not a moving of the goalposts — the ship/no-ship question was always the second question.

**The barrier claim itself is not disproved, and the experiment says exactly how far it goes.** 3 barriers,
18 shuffles and the LDS round-trip left the per-token chain (verified in the ISA: the hoisted walk issues **0
barriers and 0 LDS ops** against 5 and 39) and the row got 5.9 % cheaper — which is in the 9.2 % slot-share
slot the census predicted, not the multiple the latency story could have been read to promise. The traffic the
hoist *adds* ate most of it, which is what BET.md §3's own traffic correction (the worker's, +32 KB/token/layer
not +16) said would decide it. What this cannot see: whether a *cheaper walk* that does not add stage traffic —
G2's geometry, one wavefront per block and no LDS at all — would keep the 5.9 % and drop the +6.06 ms. That is
the question G2 is now funded to answer.

## 9. A1 measured (2026-10-19): the attention residency axis closes at 6, and it closes on LDS

E1 was the last rung the Q-staged family had, and it is a hand port (`6f59d7d`, note blocks re-derived by the
main agent from fresh compiles of both sides). Shipped `attn_fattn_wave`: **178 VGPR, 0 spill, 32,768 B LDS** →
2 blocks/CU → 6 wf/CU. Q un-staged: **184 VGPR, 0 spill, 32,768 B LDS** → 2 blocks/CU → 6 wf/CU. The register
leg of the ladder's admission passes; the **LDS leg does not exist to pass**, because the Q stage was never
extra — it is the epilogue's F32 transpose staging on loan (`FATTN_MROWS*FATTN_TROW*4 == FATTN_MROWS*DH*2`,
8,192 B per warp either way). So no 8+ wf/CU zero-spill shape exists by this lever, **no device A/B is owed**,
and the ladder's antecedent goes into the record as incomplete: `spill=0 ∧ VGPR≲190` was never sufficient,
because the binding limit is `LDS`, and it was assumed to fall.

Consequences, stated as decisions rather than as news:
- **Attention occupancy is closed.** Further residency work means shrinking the epilogue to ≤ 21,845 B (3
  blocks/CU) by staging the transpose 8 rows at a time or serialising the three through `s_kv`. Both are
  structural changes to a shipping kernel's epilogue with no measurement behind them, and neither is approved
  by this document.
- **A0's loose end is closed and it removed a candidate.** `split = split_for(lanes_last)` alone explains the
  traced 48/48/42/42 (`a0_work_split.py`, RECONCILED): body 0 of a 4,096-lane prompt has `lvalid = 1`, so
  `lanes_last = 512 = MAXSC`, so `split = 1`, so that body pays no `attn_fattn_wave` and no `attn_combine`.
  The WMMA form covers 87.5 % of rows and **98.4 % of the lane-work**. "Too many launches go through it" is
  dead as a framing; the only attention bet left standing is **per-launch cost** — the §26 instruction/traffic
  gap, sized there at 4.6×, at 2.92 ms per launch.

## 10. G2 pre-registered (before any code, per this file's own rule): the barrier-free GDN geometry

The lever: `rec_loop_kernel` runs 16 blocks × 128 threads (4 wavefronts each) with the per-token norm inside
the block, which is what buys it three barriers and an LDS round-trip per token at 0.8 blocks/CU. G2 spells the
same recurrence as **one wavefront per block, no LDS at all, no `__syncthreads()` by construction**, per-thread
ownership of several state columns so a thread's chain carries ILP, and the row reductions as warp butterflies
(legal: a head row is 128 elements = 4 per lane). That is the shape llama.cpp's `gated_delta_net` presents in
§38's trace: 32 threads, `LDS = 0`, 16 blocks, 0.80 blocks/CU, and 5.76× faster on identical launch counts.

**Both denominators are pre-registered, because §40 proved a one-row line can be fooled.** Medians of 3 at
4,096 lanes, one lock hold, three sides (`base` / new-form-off / new-form-on), canaries first:

| metric | base (§40 window) | live | dead |
|---|---|---|---|
| `gdn_rec` family ms | 266.60 | **≤ 180** | **> 230** |
| wave `wall_ms` | 626.75 | **≤ 560** | **> 607** |

Both must clear their live line to ship; either falling past its dead line kills the form; anything else is
**unresolved and not shipped**. The `wall_ms` line is the ship gate — it is the only one of the two that can
see cost that lands in another family, which is exactly how G1's −5.9 % turned into −2.2 %.

Three constraints carried in from G1, as requirements not advice:
1. **No new per-token staging.** G1's +32 KB/token/layer of normed copies ate most of its own win. If the
   geometry needs the norms, it computes them in registers from data it already loads (which the warp-butterfly
   shape does naturally), or it does not ship.
2. **Bit-identity is not required here, and the band is already fixed.** The state dot re-associates, so `o`
   moves: judged by the existing §10.1 aggregates (`cos ≥ 0.9995`, `meanrel ≤ 5 %`) through `engine/gate.py` on
   the pinned fixtures, **no new band, no widening, no `--ack-spill`**. `STATEFP` must move in a way the change
   explains, not in a way the change hopes nobody diffs.
3. **Price it in cycles-per-dependency at realised occupancy** (AGENTS.md), which for this shape means
   wavefronts-per-SIMD before and after, derived from the note block and KFD — not an instruction count.
