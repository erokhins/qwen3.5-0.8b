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
