# A4 candidates — the prior art this tree already holds for the residual ~1.6× of the attention round

Inventory only, requested by the main agent after A3 was killed at the source
(`prototype/attn_percost/A3-result.md`). **No bet lines, no pre-registration, no implementation, no device,
no timings.** Every `ms` and every ratio is quoted with its citation; every claim here is either a repo
artifact or arithmetic on quoted numbers. The question it answers, in the main agent's words: *has a larger-M
or differently-tiled form already been tried and measured here, and what did it cost?*

Premise, quoted rather than restated: A2 §2 bounded the shipped WMMA round at **5.56 % of the WMMA roof** and
**<= 8.8 % of the slot roof**, so 84-94 % of its wall is neither, and A2 §5 leaves standing exactly one
candidate for the residual ~1.6× — **C7, the round's dependent chain**, whose only unpriced quantity is the
useful-vs-issued ratio of the hand-written WMMA body (**17 `v_wmma` sites against llama's 2,592**, A2 §2, at
a tile shape A2 explicitly says nobody has compared).

## 1. The direct answer: re-tiling has been tried four times on this box, three of them measured

| form | what it changed | measured cost | where the evidence is |
|---|---|---|---|
| **M = 2** (`prototype/attn_m2/`, the `attn_m2` body, 2 live (row, head) pairs/warp) | re-tile of the shipped VALU spell, bit-identical, **demanded bytes halved** | **7-10 % slower at every width taken**, priced **15-29 % slower at the funded ruler** | `prototype/attn_m2/results.md` §2 and §8.4, carried as the geometry-tax anchor by `prototype/attn_wmma/results.md` §5 |
| **M = 16, VALU spell** (`prototype/attn_m2/attn_m16.hip.cpp`, `burst_m16.sh`, `gate_m16.py`) | 16 query rows per block, shipped dot and fold called not copied, `ndiff_aout = 0` (40/40 admission) | **2.12× to 14.25× slower** at demands of **0.28-0.55 of the bytes**, in every fixture the harness can build; `GRP=4` (3 blocks/CU) and `GRP=2` (5 blocks/CU parity) differ by **<= 5 %** | `progress/2026-10-21/performance-status-2026-10-21.md` §27 (GRP4) and §28 (GRP2 retake) |
| **M = 16, WMMA** — `attn_fattn_wave`, **shipped** | the one re-tile that removes the dependent VALU dot: score is one `v_wmma` per 16-dim step | **1.985× over the frozen VALU round** (§33), 1.3955× of the family when it shipped (§30); costs 178 VGPR / 0 spill / **32,768 B LDS** ⇒ 6 wf/CU, and A3 has just closed that axis | `progress/2026-10-21/performance-status-2026-10-21.md` §29-§33, §36, §41; `engine/attn_fattn_kernels.h` |
| **M = 32/64 per warp** (several M tiles in one warp) | never built | never measured; priced from two anchors: a full-body M = 16 block's register bill is **above** the re-tile's 134 VGPR because the PV fragments alone are **128 VGPR/lane** (§27's closing note, D30 §2.4), and the shipped form already spends those 128 at 178 total | §27, `prototype/attn_wmma/results.md` §2 ("PV accumulator wall") |

**§28 overturned §27's explanation, and that is the load-bearing fact for anything funded next.** §27 booked
the M = 16 tax as register pressure (134 VGPR ⇒ 3 blocks/CU); §28 re-ran the same harness against the
parity build (`attn_m16_grp2`: 96 VGPR, 5 blocks/CU) and **a 66 % occupancy increase while giving up half
the traffic advantage changed the time by at most 5 %**. So for this kernel family, on this box: registers
refuted, traffic refuted, and what remains is stated in §28 as *"serial-round latency at low interleave —
the m16 wall tracks lanes and only lanes, and is flat in rows"*, with the forward-looking sentence that
matters more than any number here: **"the difference between the two designs is not register capacity but
what is inside a round: theirs pipelines matrix ops with independent accumulators; ours walks a dependent
dot → 5-step butterfly fold → rescale chain per live pair, 16 of them per warp."**

Two re-open conditions §28 named have since **partly been met without anyone noticing**, which narrows what is
left to fund: (1) the WMMA score tile it said nobody had timed now ships (§29/§30); (2) the 512-row fixture it
asked for is the engine's own wave geometry, and `attn_fattn_wave` is measured there (§30, §33). What was
**not** done is the third thing §28 implied and nobody has built: a round whose matrix ops are *independent
of each other*.

## 1b. Two caveats that belong to that prior art, because both will be re-hit otherwise

* **`prototype/attn_valu_slim/results.md` §"Lever 0" is retired, and it is the closest thing this tree has to
  a "re-roll the loop" precedent.** Its lever was exactly L1's cousin: the score dot's trip count is runtime
  (`nd = (a.neg == NEG_DIMHALF) ? 4 : 8`), so LLVM kept it rolled while the PV loop beside it — trip count the
  literal 8 — unrolled and dual-issued; the fix was to emit the loop twice behind a uniform branch. Its
  measured cost is quotable: **+111 static ops, SGPRs 52 → 54, VGPRs unchanged at 47, LDS unchanged at
  12,288 B**. What AGENTS retires is the *occupancy* half of its claim ("occupancy unchanged at 16
  waves/SIMD"), which was read off clang's `Occupancy [waves/SIMD]` remark — a constant that prints 16 for
  every kernel in the TU and therefore cannot fail (`progress/2026-10-21/performance-status-2026-10-21.md` §26, closing paragraph;
  AGENTS' "Counter, occupancy and census traps" row). The lesson for L1/L2 is the instrument rule, not the
  result: a register or live-range change here must be judged with
  `prototype/attn_wmma/fattn_residency_read.py` on the note block plus a driver/counter answer, never with the
  compiler's occupancy remark.
* **`burst_m16.sh` carries one unresolved red case, and it is the harness's, not the geometry's.** Its
  `short-64` fixture (64 rows @ lanes 64-127, `split = 1`, `full = 0`) returns `ndiff_aout_m16 = 131072` —
  every element — and the harness correctly refused a millisecond figure with `NO-TIMING: admission is red`
  (§27 item 2, and the AGENTS row on "a report's bit-identical everywhere claim"). The signature
  (`ndiff == rows x width` exactly) reads as a changed work split at `split = 1` rather than a wrong value.
  Anyone re-using that harness for a new tile shape must close it first, because a form that is only exercised
  below `MAXSC` will meet it again — and because the M = 16 admission set (spans 0/1/3/16, `gate_m16.py`
  40/40) does not contain it.


## 2. Three levers, each with a price and a citation. No lines, no bands, no implementation.

### L1 — make the round's matrix ops independent rather than serial (the `#pragma unroll 1` score roll)

The shipped score leg rolls all 16 k-tiles into **one** accumulator: `engine/attn_fattn_kernels.h:577-578`,
`#pragma unroll 1` over `for (int t = 0; t < FATTN_KTILES; ++t) acc = wmma(K_t, Q_t, acc)`, i.e. a 16-long
**dependent** WMMA chain per round per warp, feeding a fold, then a rescale of 16 accumulators. The pragma's
own justification is a register statement, and the two halves of it were measured on different objects: the
note at `:566-567` says the roll "costs nothing" because there is one accumulator (the witness is the feeder
probe `prototype/attn_wmma/wmma_feeder_probe.cpp`, not the shipping body), and the note at `:648-656` records
that `unroll 8` on the **PV** leg printed `.private_segment_fixed_size 544` — 544 B/thread of scratch — where
the full unroll measured 72 B. So the roll was chosen to keep the accumulator's live range short, and that is
exactly the live range L1 would have to lengthen. llama's config row does the opposite on purpose:
`nbatch_fa = 64`, `Q_in_reg true`, 256 VGPR and **900 B/thread of deliberate scratch** at `occupancy 1`
(A2 §1) — it pays in spill to buy independent matrix work, and its object carries **1,584 + 1,008 `v_wmma`
sites** where ours carries **17** (A2 §2).

* **Price, quoted:** the round body is **>= 1,654 slots** with **96 non-matrix issue slots per matrix slot**
  and the matrix pipe **5.6 % busy** (A2 §2), so a second or third accumulator does not compete with the
  pipes — it competes with **registers**, and the register question on this part is exactly the one A3-result
  §2b declares unanswerable from CPU facts (clang 178 vs rocprofv3 248 for the same kernel, no RF size
  reported). The precedent that prices it: **§33, spilled-at-8 lost to spill-free-at-6**, and §32's Bet-2 NaN
  (independent accumulators across warps was the *coordination* failure, not the register one).
* **Citable first step that costs one compile and no device:** the live-range question has a CPU answer —
  a second accumulator over the score roll (`acc0`/`acc1`, summed once after the roll) is a compile-and-note-block
  experiment with `prototype/attn_wmma/fattn_ship_census.py` as the existing spill gate. It is **not
  bit-identical** (the 16-tile dot is a fixed ascending sum today, which is `:577`'s whole point), so its admission
  would be the existing §10.1 cos/meanrel bands under the delegated-format rule, not bit-identity — which is
  a decision to flag, not to assume.
* **What would falsify it cheaply:** if the ISA shows LLVM already software-pipelines the roll (the chain is
  an *issue* dependency, not a latency one), the lever is empty. `prototype/gdn_rec_isa/token_bill.py`'s
  loop-annotation method is the existing tool for that split; the AGENTS loop-bill traps (function labels,
  `.LBB` "in Loop" annotations, `v_fmac_f32_e32` not `v_fma_f32`) are why it must reuse that tool rather than
  re-derive a span.

### L2 — batch the accumulator rescale over 4 rounds instead of 1 (llama's `nbatch_fa = 64` against our 16)

A2 §5 already lists this as "cheaper follow-up, explicitly *not* A3", and it is the only named lever whose
price A2 computed: the rescale is **128 F32 multiplies per lane per round** (`:620-622`, where A2 §2's round
histogram counts 128 of the round's 145 `fmul` sites), i.e.
**128 of the round's >= 1,654 slots (7.7 %)**, and our 16-lane batching means **4× the rescale traffic of
llama's 64-lane batching** (A2 §2, C7).

* **Price, quoted:** A2's own verdict is the honest one — *"at 8.8 % issue utilisation, cutting issue slots is
  not where the money is — that is exactly what the §2 bounds say, so this is worth a day only as a probe of
  C7, not a funded bet."* Its value is therefore **diagnostic, and it is the cheapest C7 probe in the tree**:
  if removing ~7.7 % of a round's slots moves the family at all, the round is partly slot-bound and A2's C7
  framing needs revising; if it moves nothing, C7 is confirmed to be latency/dependency rather than slots, and
  L1 inherits the whole residual.
* **Citations:** A2 §2 C7 and §5's last paragraph; `engine/attn_fattn_kernels.h:620-622` — A2 quotes that site
  as `:620-623`, one line off the shipped text, and both land on the same loop.
* **Numerics:** not bit-identical either — rescaling less often changes the accumulator's product tree — so it
  belongs to the band-gated path, same caveat as L1.

### L3 — the split/combine scratch round-trip, which is the only *wide-context* piece still on the table

A2 §4 measured this for the first time: the streaming partials write **887,685,120 B/pass at 4,096 lanes**
and `attn_combine` reads exactly that back to write 201,326,592 B — **19 % of the bytes this kernel fetches,
7.26-8.06× the unique K/V bytes it must read anyway**, and constant in W, so it is *part of* the quadratic
term rather than a separate blow-up. Sized against §46's excess `b`: **10-13 % of the excess quadratic**.

* **Price, quoted:** derived upper bound **8 ms/pass at 4,096 (5.3 % of the 155.71 ms family)** and
  **452 ms/pass at 32,768**, from bytes ÷ 237 GB/s and explicitly an upper bound because most of it is
  L2-served (A2 §4); the cross-check through `attn_combine`'s traced 6.15 ms × the split-sum ratio gives
  354 ms = 10 %.
* **Why it stays on the list while the others are closed:** it is **differential** against llama, whose
  `flash_attn_ext_f16` keeps the online softmax in registers (`Q_in_reg true`) and writes its O tile once,
  paying no such round-trip (A2 §4) — and §46 says `b` is the term that owns the widening gap, so a lever that
  only moves `b` is worth more than its 4,096-lane share suggests.
* **Cost of acting on it, already known:** the partial slot is `(2 + DH)` F32 keyed by the FULL head and
  striding by the launch's `split`, and it is the one expression `attn_combine` and three kernels must agree
  on (`:356-358` — the one spelling of `fattn_part_slot` — and §20.1 of `prototype/attn_decode/results.md`) — so
  this lever is an epilogue-and-merge
  change with a §20.1-shaped hazard, not a local edit. The split rule itself is one integer
  (`split_for_lanes`, `prototype/attn_decode/attn_decode_kernels.h:190`, and §41's derivation of 42 from it).

## 3. What is already closed, so nobody re-funds it from this list

* **Instruction class (C1)** — both engines issue `v_wmma_f32_16x16x16_f16`; ours 5.56 %, theirs 15.2 % of the
  same 52.63 TFLOP/s roof (A2 §2, ISA-witnessed in the carved objects). §26's "parity is a matrix-class
  question" is closed by having reached the class.
* **Tile waste (C2)** — 4.2-7.3 % at every body at every width: **1.07× at most** (A2 §2).
* **K/V traffic as the rate gap (C3)** — staged demand 67 GB/s = 7 % of the cache roof, and D57's falsifier
  (an instruction cut made `attn` **1.521× faster with the byte census unchanged**) is the direct disproof that
  demanded bytes move this kernel (A2 §3, `progress/2026-10-21/performance-status-2026-10-21.md` §27).
* **Residency (C6)** — closed twice: on LDS arithmetic (A3-result.md §1, E1-result.md §4) and undecidably on
  register facts (A3-result.md §2b, `fattn_regfile_ladder.md` §3). §31's elasticity (4→8 wf/CU = 1.647×) stays
  the ledger's only measured number on that axis; it is simply no longer reachable.
* **Launch structure / per-launch cost** — 42 fattn + 48 prep + 48 slim + 42 combine per pass reconciles from
  one integer and the WMMA form covers 87.5 % of rows and 98.4 % of lane-work (§41, `a0_work_split.py`), so
  "too many launches go through it" is dead as a framing.
* **M = 2 and M = 16 VALU re-tiles** — measured, both negative (§1's table).

## 4. What this inventory cannot see

* It cannot say **how long the round's dependent chain is in cycles**: that needs a `rocprofv3` VALU-busy /
  SIM-clock pass or `SQ_WAIT_*` on the real kernel (A2 §6's first bullet, unchanged and still the cheapest
  experiment that would adjudicate C7 directly). Nothing in this tree has it.
* It cannot compare the two **tile shapes** (our 48 rows / 3 warps / one K/V stage against llama's 256×256
  256-thread form at `occupancy 1`) beyond A2 §1's geometry table, because llama's tuned kernels are
  unreadable CCO containers — every matrix rate in the ledger belongs to hipBLASLt or to a container
  `llvm-objdump` refuses (AGENTS' census trap; §26).
* It repeats A2's honesty limit: traced and untraced attention numbers are different measurements
  (155.71 traced vs 133 untraced ms/pass) and ratios across them are not the same claim (§46's 2.164× `b`
  ratio vs the traced attention ratio 2.69-3.02×).
* The one number this list most wants — **the useful-vs-issued ratio of our own 17 sites** — is a census, not
  a device result, and `prototype/attn_percost/a2_round_bill.py` plus `prototype/gdn_walk/g3_isa_bill.py`
  already exist to take it (the round span `0x11520..0x13970` in the carved object). Nobody has asked it the
  one question L1 needs: how much of the round's `>= 1,654` slots are **inside the accumulator's dependency
  cone** rather than beside it. That is a CPU afternoon and it is where I would point first.
