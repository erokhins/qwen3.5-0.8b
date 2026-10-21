# Address arithmetic in the looped GDN kernel — what the bill actually contains

Compile-only package. Nothing in this file came from a timing run, and no number here belongs in
`engine/performance-status.md`. Device commands are handed over in `run_screen_recptr.sh`, which its author
did not run.

Scope: the looped GDN recurrence at `211eb95` HEAD — `engine/gdn_rec_loop_kernels.h::rec_loop_kernel` and
its harness mirrors `prototype/gdn_rec_isa/gdn_rec_isa_kernels.h::{rec2c4_kernel, recloop_kernel}`, plus the
new `recptr_kernel` this package adds. Inputs are the `--save-temps` device listings under
`tmp/gcc/gdn_addr/` (engine TU) and `tmp/gdn_isa/gdn_rec_isa.s` (harness TU).

## 1. The shipped per-token bill, re-derived

From the engine's own TU listing, with `prototype/gdn_rec_isa/isa_census.py` (the census `progress/2026-10-21/performance-status-2026-10-21.md` §24
rows were taken with):

| kernel | form | flat slots / token-step |
|---|---|---|
| `rec_fused_kernel` | one token per launch | **1,374** |
| `rec_loop_kernel` | `--gdn-loop` span inside one launch | **1,530** |

Flat delta **+156 slots/token-step**, which reproduces the §24 of `progress/2026-10-21/performance-status-2026-10-21.md`/D55 figure. The brief's line
`cast=97 other=20 total=406` is **not** this census and cannot be reconciled with it: that triple is the
`LaunchCounts` launch-plan bill printed by `decode_run` (`LaunchCounts::describe()`), whose units are
*kernels launched per decode step*, classified cast/non-cast. The two instruments count different things
(slot-pairs inside one kernel vs launch records per step), so the per-token line quoted in the brief is a
launch bill sitting where an instruction bill was assumed. Anything sized from it is sized wrongly.

## 2. The flat bill is not a per-token bill

`token_bill.py` (new, this package) separates the blocks that run once per launch from the blocks inside the
token loop, using LLVM's own loop annotations (`in Loop: Header=BB5_4`, `Parent Loop …`) rather than a
dominator computation, and keeps `isa_census.py`'s walk finder so the flat total stays comparable:

| kernel | per-launch straight | per-token straight | walks per token | flat | **per-token** |
|---|---|---|---|---|---|
| `rec_fused_kernel` | 0 | 294 | 44×8 + 91×8 = 1,080 | 1,374 | 1,374 |
| `rec_loop_kernel` | **203** | 255 | 44×8 + 90×8 = 1,072 | 1,530 | **1,327** |
| `recptr_kernel` | 190 | 311-ish (see §4 of this file) | 1,072 | 1,526 | 1,336 |

So the looped form is **47 slots/token cheaper** than the per-token form on the honest per-token
accounting: the +156 is an artifact of billing the loop's 203 once-per-launch slots (kernarg loads, the
shuffle exit-guard chain, the `tokens<=0` path) at ×1 per token. The instruction-stream half of the loop was
never a cost. That is arithmetic on the listing, not a measurement, and it does not contradict §24 of `progress/2026-10-21/performance-status-2026-10-21.md` '*time*'
result — the looped form's measured win came from submission count, which is what D53/D55 measured.

## 3. How much of the bill is address arithmetic: 93 slots, not 156

Definition used, because the category table in `isa_census.py` cannot answer it (LLVM prints pointer
arithmetic as `addr.math` only when the mnemonic matches, and half of it lands in `unclassified`): an
instruction is address arithmetic if (i) its mnemonic is in a fixed list of pointer/index arithmetic
(`s_add_u32`, `s_addc_u32`, `s_lshl_b64`, `s_mul_i32`, `s_mul_hi_u32`, `v_add_co_u32`,
`v_add_co_ci_u32_e64`, `v_or_b32`, `v_mov_b32_e32`, and the `v_dual_*` slots whose second half is one of
those — the slot is what gets billed, and LLVM pairs a float op with an address op in one slot), and (ii)
every register it writes is read either as a memory address operand or by another such instruction that
reaches one. Condition (ii) is what makes it a use-based census rather than a mnemonic census; without it the
first version of this file billed 46 `v_fmac_f32_e32` slots in the state walk to the address bill, because
`global_store_b32` prints `vaddr, vdata` and the store's *data* looked like an address.

| kernel | straight-line / token | in the walks / token | total | % of flat bill |
|---|---|---|---|---|
| `rec_fused_kernel` | 16 | 80 | **96** | 7.0% |
| `rec_loop_kernel` | 13 | 80 | **93** | **6.1%** |
| `recptr_kernel` | 24 | 80 | 104 | 6.8% |

So the answer to "is the prize ~156 slots/token of address arithmetic" is **no**:

* all address arithmetic, in every form, is **93–96 slots/token-step ≈ 6–7%** of the bill — about 60% of the
  156 the queue line implied, and 60% of it would be removable only by removing every address;
* the looped form has **3 fewer** address-arithmetic slots than the per-token form it replaced, so the +156
  delta contains **none** of address arithmetic. Its content is the loop control and the shuffle
  exit-guard/SGPR re-materialisation chain (`xlane.exec-guard` 82 of the 203 per-launch slots), which an
  address-spelling change cannot touch;
* two thirds of the address arithmetic (80 of 93) is the **state walks**, which are identical in every form
  because the state row advance is already an `s_add_i32` induction the compiler strength-reduced.

`--selftest` on the shipped listing (it must and does pass):

```
SELFTEST append_addr    OK 93 -> 94 (required >93)
SELFTEST append_float   OK 93 -> 93 (required =93)
SELFTEST attached       : 17 counted address lines write a register a memory operand reads as an address (required > 0)
SELFTEST must-not-catch : 0 counted lines that are pure float work (required 0)
```

`append_addr` inserts a self-contained `v_add_co_u32 v9500, …` + `global_load_b32 v9502, v[9500:9501]`
chain into a per-token block and requires the count to grow; `append_float` inserts a dead float add at the
same place and requires it not to. Mutations are additions rather than deletions or renames on purpose: on a
real listing every physical register is reused, so removing or renaming one address instruction also changes
which later reads belong to it, and the effect on the count is not predictable — the first attempt at this
selftest reported `rename_addr WRONG 27 -> 22` for exactly that reason, and a check that moves for the wrong
reason is worse than none.

## 4. The MODE: `recptr` — pointers pre-offset, advanced by their strides

`recptr_kernel` (`gdn_rec_isa_kernels.h`) keeps the loop and the arithmetic of `recloop_kernel` and changes
only the addressing: the six row pointers start at the base tiles before the loop and are advanced at the
bottom of each iteration by `q += CH; k += CH; v += CH; g += NH; beta += NH; o += NH * D;`.

`check_recptr_transcription.py` proves the claim mechanically, in three parts: (a) every line from
`float kq =` through `o[h * D + j] = acc;` — pragmas included — is character-identical between the two
kernels; (b) each of the six pointers has the same base buffer and its step equals the stride
`recloop_kernel` multiplies `t` by, both read out of the source so the checker cannot go stale; (c) with the
declarations and the six steps removed, the two bodies are line-identical. `--selftest` requires seven
defects to be caught (wrong stride, stride ×2, swapped base, dropped step, re-associated product, dropped
pragma, dropped store) and one cosmetic reorder NOT to be caught; both directions pass.

What the compiler did with it:

| | flat | per-launch | per-token | address slots | VGPR | SGPR | LDS | spill | residency (runtime calculator) |
|---|---|---|---|---|---|---|---|---|---|
| `recloop_kernel` | 1,530 | 203 | 1,327 | 93 | 75 | 40 | 1,280 B | 0 | 16 blk/HIP-CU |
| `recptr_kernel` | **1,526** | 190 | **1,336** | **104** | **69** | 44 | 1,280 B | 0 | 16 blk/HIP-CU |

The pointer form **increases** counted address arithmetic (+11: six 64-bit pointer steps cost
`s_add_u32`/`s_addc_u32` pairs) and removes the multiply chain that produced `t*CH`, for a net **−4 flat
slots (−0.26%)** and **+9 per token**.

Residency comes from `occupancy_probe.cpp` (new), which calls the runtime's own
`hipOccupancyMaxActiveBlocksPerMultiprocessor` and launches nothing. Read it in two parts. Its LDS canary is
sound — a 63 KiB-static-LDS stub returns 1 block/CU and a 32 KiB stub returns 2, so the calculator is
reading static LDS (a full 64 KiB stub is refused outright: the runtime reserves a page per block). But its
absolute answer is geometry-bound and inflated by the 2× reporting fault on this SKU (AGENTS.md,
counter/occupancy trap row): HIP says `multiProcessorCount=20`, `maxThreadsPerMultiProcessor=2048`, so
`stub_lds_none` and all four kernels come back at 16 blocks/CU = 64 wavefronts, above the architectural
maximum of 32 wavefronts per CU. What survives from it is the **comparison**, not the magnitude: the two
looped forms get the identical answer at 75 and 69 VGPRs, so the VGPR drop does not move residency in the
model the runtime is using, and the compiler remark `Occupancy [waves/SIMD]: 16` — which prints 16 for every
kernel in the tree, the 63 KiB stub included — is not evidence of anything and was not used. If residency
ever becomes the question, it needs a wavefront counter (`rocprofv3`), which is a device run this package
does not authorise.

## 5. Verdict, and what would change it

Against §24 of `progress/2026-10-21/performance-status-2026-10-21.md` 's own elasticity (−156 slots/token bought 6.7 pp of wave time, i.e. ≈0.043 pp per slot-step),
−4 slots/token is ≈0.17 pp of wave time ≈ 0.5% of `gdn_rec` — under any refusal bar, and the per-token
component went the wrong way. The address spelling is therefore **not a funded lever**: the premise that
~156 slots/token of the looped kernel is address arithmetic does not survive the listing, and the residual
address arithmetic is dominated by state-walk addressing that the shipping form already strength-reduces.

What would change the verdict: a form that attacks the **walks'** 80 slots (two independent row-pointer
inductions, one per walk, instead of the current per-trip index arithmetic), or a change that moves the
82-slot `xlane.exec-guard` per-launch chain, since that — not addressing — is where the loop's flat
surplus actually sits.

## 6. Re-deriving this

```bash
# engine TU listing (compile only)
hipcc -O3 --offload-arch=gfx1151 --save-temps -c engine/kernels_gdn.cpp -o tmp/gcc/gdn_addr/k.o -I. -Iengine
.venv/bin/python prototype/gdn_rec_isa/isa_census.py --asm tmp/gcc/gdn_addr/kernels_gdn-…-gfx1151.s \
    --kernels rec_fused_kernel,rec_loop_kernel
.venv/bin/python prototype/gdn_rec_isa/token_bill.py  --asm …/kernels_gdn-…-gfx1151.s \
    --kernels rec_fused_kernel,rec_loop_kernel --selftest

# harness listing + resources + residency
SAVE_TEMPS=1 ./prototype/gdn_rec_isa/build.sh
.venv/bin/python prototype/gdn_rec_isa/token_bill.py --asm tmp/gdn_isa/gdn_rec_isa.s \
    --kernels rec2c4_kernel,recloop_kernel,recptr_kernel --selftest --dump-addr
.venv/bin/python prototype/gdn_rec_isa/check_recptr_transcription.py --selftest
hipcc -O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib -I. -Iengine \
      prototype/gdn_rec_isa/occupancy_probe.cpp -o tmp/gcc/gdn_addr/occupancy_probe && tmp/gcc/gdn_addr/occupancy_probe
```

`run_screen_recptr.sh` is the handover: pre-flights (transcription + mutation selftests, the per-token bill
lines differing between the two sides, VGPRs 75 vs 69 with LDS and spill equal), then inside one
`gpu_lock.sh` hold the volume witness, the admission at spans 0/1/3/16 (`ndiff_o_vs0=0`,
`ndiff_state_vs0=0`, reruns clean, `admission=PASS`) and the paired burst `rec2c4,recloop,recptr`. **The
behavioural canary is the per-token bill line: if `recloop` and `recptr` bill the same per-token total in
the window, the binary is stale or the MODE collapsed, and the timing means nothing** (AGENTS.md, the
stale-binary row — a self-consistency field compiled into the binary under test is blind to staleness).
