# The residency ladder after Bet 2: what 6 wf/CU is made of, and what 8+ would cost

Prompt: §33's closing line — *"push residency further inside a zero-spill geometry: is there an 8+ wf/CU shape
that does not pay in scratch?"* This doc prices that question and finds it **cannot be answered on this part
with the facts currently measured**, plus one experiment that can be answered immediately. Compile-only and
driver-only throughout: **no kernel launched, no timing taken**, by anyone, in this file.

## 0. Measured facts this stands on (all re-taken or cited, none assumed)

| fact | value | how it is known |
|---|---|---|
| `attn_fattn_wave` note block at tip `01b3709` | **VGPR 178, spill 0, scratch 0 B/lane, LDS 32768 B, SGPR 24** | re-taken this session, compile-only: `hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage -c engine/kernels_attn.cpp` (matches the §33 banner, so the ledger's note-block claims survive a re-take) |
| head dim `DH` | 256 | `prototype/attn_decode/attn_decode_kernels.h:26` |
| live O accumulator | `FattnO::t[FATTN_VTILES]` with `FATTN_VTILES = DH/16 = 16` tiles = **128 VGPR/lane** | `engine/attn_fattn_kernels.h:189` (the header says so itself) |
| LDS per CU | 65536 B | KFD `/sys/.../nodes/1/properties` → `lds_size_in_kb 64`, `simd_per_cu 2`, `simd_count 80` (re-read this session); also `prototype/attn_wmma/evidence/device_facts.txt` |
| waves/SIMD ceiling | 16, `Max Waves Per CU: 32`, 2 SIMDs/CU, 40 CUs | `rocminfo` Agent 2 (re-run this session) |
| 16 blocks/CU at 0 B LDS, 4 at 16 KiB, 1 at 64 KiB | driver-confirmed stub canaries | `prototype/attn_wmma/occupancy_fattn_probe.cpp` as run in `tmp/gcc/fattn_residency.md` §6 (worker, correctness-only, no kernel launched), output in `prototype/attn_wmma/evidence/occupancy-fattn.txt` |
| **VGPR file per CU** | **UNKNOWN on this part** | see §3 — this is the finding |

## 1. The Q-staged family has a ceiling of 7, and it is LDS arithmetic

With `NW` row-blocks in one block, each warp keeping its own 8 KiB Q stage over one shared 8 KiB K/V stage
(the shipped shape), `LDS(block) = 8192·(NW+1)`, so on the driver's LDS rule

```
wf/CU(NW) = NW · floor(65536 / (8192·(NW+1))) = NW · floor(8/(NW+1))
        NW = 1 → 4    2 → 4    3 → 6    4 → 4    5 → 5    6 → 6    7 → 7
```

NW=1,2,3,4,7 are the rows §4 of `tmp/gcc/fattn_residency.md` measured on the skeleton and they match this
formula exactly, which is what makes it safe to fill in 5 and 6: **the best a Q-staged sharing multiple can do
is 7 wf/CU at NW=7** (224 threads, `LDS = 65536` = the entire budget, one block/CU, and a 112-row union the
round loop pays for). The shipped NW=3 at 6 is therefore **LDS-bound, not register-bound** — at 178 VGPR the
registers would happily hold far more, so *no register cut at all moves residency in the shipped geometry.*
Any statement of the form "we will get more wavefronts by slimming the kernel" is false for this family.

## 2. Where the 178 came from, because the handover attributes it wrongly

The natural story — "the Q stage is what the registers pay for" — is **refuted by the ledger's own numbers**:

| body | VGPR / spill / scratch | source |
|---|---|---|
| 1 warp, Q staged (pre-Bet-2 shipping) | 256 cap / **48** / 68 B | `tmp/gcc/fattn_residency.md` §5 table |
| 1 warp, Q un-staged (Bet 1) | 256 cap / **50** / 72 B | same table (+2 registers for un-staging) |
| **3 warps, Q staged (shipped)** | **178 / 0 / 0** | re-taken above |

Un-staging Q costs **+2** spilled registers and buys −32 B of LDS reads: it is nearly register-neutral (§5's
own conclusion). The **−128 VGPR and all of the spill relief came from the widening itself**, not from any
stage — the plausible mechanism is the one the commit note names in passing, that with three warps sharing one
stage "the round's range has to be block-uniform", and a block-uniform trip count gives LLVM shorter live
ranges than a per-row one. That is an inference, not a measurement, and E1 below is what pins it.

**Consequence, and it is the whole answer to the queue's question:** the register pressure of the *3-warp body
with Q un-staged* has never been measured. §4's V6 (`no Q + 3 row-blocks`) is the same shape on the
**skeleton** and it spilled 73, but the skeleton understates register *demand* by roughly 100 registers on the
control geometry alone (V0 prints 196/0 where the shipping 1-warp body printed 256-with-48-spills, i.e. a demand
near 304), and §4's own caveat says *"Any spill claim from this table must be re-measured on the real patch."* Bet 1 was believed only after exactly that re-take; V6 has
not had it. If the widening's −128 VGPR carries to the un-staged form, then the un-staged 3-warp body lands
near **8192 B of LDS with zero spill**, which is 8 blocks/CU of LDS headroom and a residency decided by
registers instead of LDS — i.e. **the 8+ zero-spill rung exists**. If it lands at 256-with-spill, V6's story
holds, the spill tax of §33 applies again, and the residency axis is closed.

So the question is not "is there a geometry" — it is **one compile of an un-staged 3-warp body**, which does not
exist yet: Bet 1's patch no longer applies to it (§4, E1), so the prize is priced by a port of a 102-line patch
and one note-block read.

## 3. The missing fact: the register file per CU has never been measured here

Every rung above 8 wf/CU needs the register bound, and the box has never been asked. Two independent reasons
not to guess:

* The usual AMD figure (65536 VGPR per SIMD ⇒ `wf/CU = 2·floor(2048/vgpr)`, which reproduces `Max Waves Per CU:
  32` at low VGPR) is **contradicted by the one driver answer that could test it**: §4's V6 row reports the
  driver granting **8 blocks/CU at 3 warps and 256 VGPR = 24 wavefronts/CU**, which needs 196 608 VGPRs per CU,
  three times that file. Either `hipOccupancyMaxActiveBlocksPerMultiprocessor` is naive about VGPR on this SKU,
  or the file is bigger, or the row is an artifact of a kernel that does not really run at 256 (it has 73
  spills). All three are possible and they imply different ladders.
* `rocminfo` does not print a register-file size on this stack, and AGENTS.md has already retired HIP's own
  capability answers on this SKU twice (the halved `multiProcessorCount`, the constant `Occupancy
  [waves/SIMD]` remark). A third HIP number cannot be trusted on inheritance.

**Canary that settles it (driver query, creates a context, launches no kernel, ~1 s):** stub kernels at 0 B LDS
with a real VGPR demand driven by a compile-time dummy register count (8, 32, 64, 128, 168, 192, 224, 255),
each asked for `hipOccupancyMaxActiveBlocksPerMultiprocessor` at 32/64/96 threads. The 0-B-LDS stub already
prints 64 blocks at 32 threads, so the sweep is the same probe with the register axis added. Until it prints,
every number in §2 above 8 wf/CU is **model, not measurement**, and must not be quoted as a residency claim.

## 4. The three experiments, in order, and what each decides

None of E1–E3 is a timing. All are compile-only or driver-only, and all are worker-legal under the GPU rule.

* **E1 — the deciding one: the 3-warp body with Q un-staged. It is a port, not an apply.** The obvious version
  — re-apply `tmp/gcc/fattn_unstage_q.patch` transiently as §5 did for Bet 1 — was tried first and **fails**:
  `patch -p1 --dry-run` on a clean snapshot of the tip gives **4 of 5 hunks FAILED in
  `engine/attn_fattn_kernels.h` and 1 of 1 in `engine/kernels_attn.cpp`** (only the score-leg hunk still lands,
  with fuzz), because Bet 2 rewrote exactly the hunks the un-stage patch touched: the Q-stage fill is now the
  per-warp `for (qr = 0; qr < 16; ++qr)` loop over `s_q[w * FATTN_MROWS * DH]`, and the launcher's row-block rule
  moved into `prototype/attn_decode/attn_decode_kernels.h`. So E1 costs a hand port of the un-stage onto the
  per-warp stage (drop `s_q`'s fill and read the Q operand tile from global at the use site, keeping the shared
  `s_kv` and the 3-warp geometry), and then one note-block read:
  `vgpr_count / vgpr_spill_count / private_segment_fixed_size / group_segment`.
  * `spill = 0` and VGPR ≲ 190 ⇒ there **is** an 8+ wf/CU zero-spill shape, and it is worth a device A/B
    against the shipped 6-wf form;
  * spill > 0 ⇒ §33's spill tax re-applies and the residency axis is closed at 6 (with NW=7 as a footnote).
* **E2 — the register canary of §3,** which turns the ladder from model into measurement.
* **E3 — NW=7 priced,** for completeness of the ladder and because it is the only rung the Q-staged family
  still has: same transient-patch method, expect `LDS 65536` and one block/CU, and note (from the note block,
  not from hope) whether it spills at all and what the 112-row union costs in round range on a `wide1024f`
  fixture — that part is arithmetic on `fattn_covers`/the lane union, CPU-side.

## 5. What this does **not** settle, said plainly

* **Residency is not speed.** §31 bought 4→8 wf/CU for +20 % of the family while carrying 50 spilled registers;
  §33 dropped to 6 wf/CU and got another 1.217× back by deleting the scratch. Nothing in this doc predicts a
  millisecond, and a device A/B is still the only thing that may say one.
* Even 15 wf/CU is **47 % of the machine's 32**, so §28's null (interleave stops paying in some band) is not
  escaped by any rung here — it is approached. If E1 says "zero spill at 8+" and the A/B says "nothing moved",
  that is the answer that the serial round chain is the wall, and the money should move to the instruction and
  traffic gap §26 sized at 4.6×, not to a fourth residency bet.
* Nothing here touches the **known-red ship census**; that is a separate task in flight, and a residency change
  would have to move `geometry()` again (it reads the stage declarations, so it must be taught warp count
  first — E1/E3's numbers are only trustworthy once that gate adjudicates warp counts at all).
