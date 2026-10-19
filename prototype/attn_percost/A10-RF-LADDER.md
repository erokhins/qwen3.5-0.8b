# A10 step 2 — the compile-only register/LDS ladder: R1+R2 as written does not fit, and it is arithmetic, not tightness

**Question asked** (kickoff step 2): can 256 VGPR/thread hold 128 accumulator + Q + scores + addressing at
8 waves × 16 query rows = 128 rows/block, 256 threads, a 64-position KV stage, Q in registers, staging
synchronous? **Answer, compile-only, no device window run: no — not with the atom this engine ships.**
The funded spelling overflows the per-thread ceiling by 0 registers for the accumulator and Q alone and
nothing is left for scores, P or addressing. A source kill, and a cheap one: no `gpu_lock` hold was taken for
this and none is needed for the verdict.

Instruments: `a10_rf_ladder.sh` (builds the canary and the ladder, reads note blocks), `a10_rf_ladder.hip.cpp`
(the geometry variants + LDS stubs), `a10_frag_cost.hip.cpp` (the fragment-cost calibration), `a10_lb_probe.hip.cpp`
(the `__launch_bounds__` question). All four are `-c --save-temps` compiles: **nothing launched, no context
created, no timing taken anywhere in this file.**

---

## 0. Canary: the instrument is calibrated on a number the ledger already holds

`a10_rf_ladder.sh` builds `engine/kernels_attn.cpp` with the same clang and flags before it builds anything of
its own, and refuses to continue if the shipping kernel does not read what §48 and §53 recorded:

```
attn_fattn_wave          vgpr=178 spill=0 lds=32768     (Q staged: the `stage` spelling)
attn_fattn_wave_two      vgpr=200 spill=0 lds=32768
attn_fattn_wave_two_u8   vgpr=221 spill=0 lds=20480
attn_fattn_wave_u8       vgpr=185 spill=0 lds=20480     ← the default; §53's inertness witness read
                                                          `vgpr=185 lds=20480 wf_per_cu=9` — same triple
CANARY PASS
```

Two things that fall out of the family table while it is on screen, neither of them news but both worth having
in one place: the Q-un-staged spellings are **37–43 VGPR cheaper than the staged ones** (185 vs 178 at one
accumulator, 221 vs 200 at two), which is G1's residency purchase seen from the register side, and `two_u8`
at 221 is where A4-L1's two-accumulator spelling sits — i.e. that lever spent registers, not just the −1.67 %
inside the paired floor (§44).

## 1. The unit: what one WMMA fragment costs, measured three ways

Every number in §2 is a multiple of one quantity — the VGPR cost of a single 16×16 F16 operand tuple — so it
was measured rather than assumed.

| method | reading |
|---|---|
| `.vgpr_count` slope of `a10_frag_cost.hip.cpp`'s hold probe, N = 1, 4, 8 fragments | 12 → 36 → 68 ⇒ **8 VGPR per 16×16 F16 fragment** (16 halves/lane, two halves per VGPR) |
| the shipping kernel's own disassembly (`prototype/attn_percost/evidence/fattn_wave_u8_wmma_slice.s`, the tracked slice of the listing `kernels_attn-…-gfx1151.s`; regenerate with the command in §5) | `v_wmma_f32_16x16x16_f16 v[129:136], v[159:166], v[167:174], v[129:136]` ⇒ A, B and C tuples are **8-register spans** |
| the engine's declared types (`engine/attn_fattn_kernels.h:184-185`) | `fattn_ARegs = short ext_vector_type(16)` = 16 halves = **8 VGPR**; `fattn_CRegs = float ext_vector_type(8)` = **8 VGPR** |

**Instrument defect, recorded because the probe lied before it was fixed.** The first hold probe read 82 VGPR
for 32 fragments (256 expected): a pure `__ldg` has no side effects, so clang hoisted the fold above the point
where the fragments were supposed to be simultaneously live and the probe measured address arithmetic instead of
storage. A `volatile` asm with a `memory` clobber between the loads and the fold fixes it, and the linear region
(N ≤ 8) is what the slope above comes from; past N = 8 the same probe saturates again (77 at 16, 82 at 32) and
**must not be read as a cost**. The disassembly is the authority; the hold probe agrees with it inside its valid
range and disagrees outside it, which is the shape of a probe that is right for a reason and wrong for a reason.

Consequence, and it is the whole finding: **resident Q is 16 fragments, not 8.** A wave holds 16 query rows ×
256 dims; one 16×16 operand tuple carries 16 query rows × 16 dims, so 256 dims is 16 tuples × 8 VGPR = **128
VGPR per lane.** `A10-REWRITE-BRIEF.md` §6 priced Q at "~64 more", which is what you get if you count
128 halves per lane and assume they pack two to a VGPR — they do pack, but the operand tuple is defined per
16 dims of K, not per 32, so the packing does not halve the tuple count. §3 now says 128.

## 2. The arithmetic, against the ceiling

Per-thread VGPR ceiling on this part is **256** (RDNA: 65,536 lane-VGPRs per CU ÷ 256 threads at 1 block of 256
per CU = 256/thread, and no code object here reports more). Live set for one wave mid-round:

| quantity | f32 accumulator (our atom) | f16 accumulator (llama's RDNA3 atom) |
|---|---|---|
| P·V accumulator, 16 rows × 256 dims | 16 × 8 f32 tiles = **128** | 8 × `tile<16,16,half2>` = **64** |
| Q resident, 16 rows × 256 dims (§1) | **128** | **128** |
| a 64-position round's scores | 4 × 8 = **32** | **32** |
| P operands, staging temps, addressing | ≥ 20 | ≥ 20 |
| **total** | **≥ 308 — over the ceiling by ≥ 52** | **≈ 244 — 12 registers of headroom for everything else** |

**R1+R2 with the shipped `v_wmma_f32_16x16x16_f16` cannot be built.** That is not "tight", and it is not a
scheduling problem a better compiler would solve: 128 + 128 already equals the ceiling before a single score,
pointer or loop counter is allocated. The brief's own §4 point 2 ("M cannot grow: at 32 rows the accumulator
alone wants 256 registers/lane") was the same observation one step earlier; the missing half was that Q-in-
registers moves the same bound onto the operand side.

The f16 column is llama's shape, and it fits with 12 registers to spare — which is why the honest conclusion is
not "the geometry is impossible" but **"the geometry and the accumulator format are one decision, not two"**, and
the brief lists them as R2 and R8 with R8 downstream of everything.

## 3. What the variant table actually shows, and what it does not

Every 256-thread variant reads `vgpr_count = 256` (the ceiling) with a few hundred VGPRs spilled:

| variant (256 threads, 8 waves, 64-position stage unless noted) | vgpr | spill | scratch B/thread | LDS |
|---|---|---|---|---|
| `a10_r1_f32` — R1+R2+R3, Q resident, our atom | 256 | 415 | 1,108 | 32,768 |
| `a10_r1_f16` — same, f16 accumulator | 256 | 393 | 1,144 | 32,768 |
| `a10_r1_f32_noq` / `a10_r1_f16_noq` — Q re-read each round | 256 / 256 | 415 / 393 | 1,108 / 1,144 | 32,768 |
| `a10_r1_f16_split` — K and V both resident (L-b's mistake at a deeper stage) | 256 | 364 | 988 | **65,536** |
| `a10_cur` — the shipped round shape as a skeleton | 256 | 124 | 340 | 8,192 |
| LDS stubs 0 / 8,192 / 20,480 / 32,768 / 36,096 / 65,536 B | 2 | 0 | 0 | as declared ✓ |

Three things this table cannot say, stated because the temptation is there:

* **Spill volume is not a verdict here.** The `noq` variants spill *exactly* what the resident-Q variants do
  (415 / 393), which is impossible if the Q tuples were really live in one and re-read in the other — so the
  skeleton's round body is being re-folded by the optimiser in a way that makes the Q axis unreadable at this
  granularity, and a skeleton that reproduces the shipped geometry (`a10_cur`) also lands at 256 + 124 spill
  while the real body sits at 185 + 0. **Skeletons are not bodies.** What survives of the table is the ceiling
  and the LDS column, both of which §1–§2 derive independently of the skeleton.
* **LDS does not bind at 64 positions.** The aliased stage is 32,768 B (`a10_stub_32768` → 2 blocks/CU) and even
  llama's whole declared demand, 36,096 B, leaves 1 block/CU by 4 KiB of margin — but the margin is in the stage
  alias, not in the geometry, and the non-aliased spelling (`a10_r1_f16_split`, 65,536 B) is 1 block/CU with the
  entire LDS spent. This confirms §3 SETTLED item 4: after R1 the binding resource is registers, not LDS.
* **The `__launch_bounds__` rung ladder did not ladder, which is a fact about the target, not about the skeleton.**
  Every rung of every variant reports `vgpr_count = 256` regardless of the cap its own hint implies: at 256 threads,
  `minBlocks = 4` means 1,024 threads/CU and therefore 65,536 ÷ 1,024 = **64 VGPR/thread**, and `a10_r1_f32_lb4`
  still reports 256. Same at 96 threads with `minBlocks = 4` (cap 170, reports 256), and `a10_lb_probe.hip.cpp`
  answers rungs with implied caps of 75, 64, 42, 32 and an unreachable 8-blocks-of-256 with **75 VGPR and 0 scratch
  at every one of them** — no spill, no refusal. The one thing the hint does change is scheduling: `lb4` cut the f32
  variant's spill from 415 to 324 and `a10_3w_f32_lb3`'s from 706 to 272, so clang reacts to the hint by
  re-scheduling into scratch rather than by honouring the cap, and it warns when it cannot meet one (`failed to meet
  occupancy target given by 'amdgpu-waves-per-eu' … desired occupancy was 4, final occupancy is 3`, twice in this
  build) and proceeds. Two consequences, both wider than this ladder: **`__launch_bounds__` is not a register lever
  on this target**, so a rung ladder built on it — including A6's compile side — is reading the compiler's mood and
  not the register file, and the register question must come from live-set arithmetic plus the driver-side occupancy
  query; and it settles the §3 provenance open question — llama's `__launch_bounds__(256, occupancy = 2)`
  (`fattn-mma-f16.cuh:164`, `:1797`) asked the allocator for nothing, which is how a kernel declaring `occupancy 2`
  arrives at the traced 256 VGPR/thread of `A2-census.md:78`.

## 4. Verdict, and the two ways forward

**Step 2 does not survive as written, so step 3 is not started.** No knob, no BET, no device window: the
abort line (6.5 TFLOP/s at 32k) is not reachable by a build that cannot be built, and burning a paired window to
discover arithmetic that clang will print for free is the mistake the step ordering exists to prevent.

The rewrite survives only in a form that changes what was funded, and the choice is the user's, not mine:

1. **Keep `v_wmma_f32_16x16x16_f16` and give up R2.** Q keeps coming from the L2/L1 hierarchy every round (the
   shipped G1 arrangement), the deep 64-position stage still cuts rounds per token ~4× and the rescale chain with
   it, and R1 still cuts the LDS cost per served row. Live set: 128 accumulator + 32 scores + ~20 = ~180, which
   fits with room. What is *not* bought is the item the premise rests on — the Q re-read at 898 GB/pass and
   ~1,100× redundancy per layer — so §3's 9.6 → 64 MAC-per-LSU-byte story halves at best, and there is no reason
   to expect the 6.5 TFLOP/s line from it. Priced in registers: comfortable. Priced in promise: weak.
2. **Take the f16-accumulating atom (`v_wmma_f16_16x16x16_f16`, llama's RDNA3 form) and build R1+R2+R3 whole.**
   Live set ~244 of 256, i.e. it fits the way llama's fits — which is to say it also spills on purpose
   (`A2-census.md:78`: 256 VGPR, 900 B/thread scratch, and 1.9× our rate regardless), so the brief's point 6
   ("what are the spilled bytes on the critical path") is now the open question rather than the spill itself.
   The cost is that **R8 stops being downstream**: accumulating P·V in f16 across a 32,768-position sweep is a
   format change on the shipping path, and the atom's issue rate on gfx1151 has never been measured —
   `prototype/issue_rate_probe/results.md` §1 priced only the f32 form (52.63 TFLOP/s). Both are cheap to settle
   and neither is settled here: the rate is one existing-probe run in a lock hold of mine, the numerics are the
   existing gates with no new band, and the probe answer could make the numerics moot by showing the atom is
   slower.

Cheapest next step, in one line: run `prototype/issue_rate_probe` with an f16-accumulant kernel added to the
existing harness, in a single `gpu_lock` hold, before either branch is funded — it is the only number in this
file that can change the recommendation, and it costs minutes rather than a week.

## 5. Reproducing this

```bash
bash prototype/attn_percost/a10_rf_ladder.sh /tmp/a10rf     # canary + variants + note blocks, ~10 s, no device
hipcc -O3 --offload-arch=gfx1151 --save-temps -c prototype/attn_percost/a10_frag_cost.hip.cpp -o k.o
hipcc -O3 --offload-arch=gfx1151 --save-temps -c prototype/attn_percost/a10_lb_probe.hip.cpp -o k.o
python3 prototype/attn_wmma/fattn_residency_read.py --asm <listing> --all
```

Note-block parsing is `prototype/attn_wmma/fattn_ship_census.py`'s, imported, not re-written (AGENTS: list the
census tools before writing one). One instrument defect found while doing this and **not** fixed, because fixing
it would move A6's recorded evidence: `prototype/attn_percost/a6_rf_ladder.sh` no longer builds — its rung
template calls `attn_fattn_body<false>` and G1 added a second template parameter, so every rung errors with
`candidate template ignored: couldn't infer template argument 'kQUnstaged8'`. Anyone re-running A6's ladder needs
`attn_fattn_body<false, false>` (base) and `<false, true>` (g1) in `a6_rf_ladder.py:40`.
