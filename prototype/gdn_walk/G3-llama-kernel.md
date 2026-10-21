# G3 — a census of llama.cpp's GDN kernel, diffed against our shipped walk (2026-10-19)

Task: `prototype/gdn_walk/BET.md` §11 funds G3 as **a census, not an implementation** — read
`engines/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`, find what one token's work actually is, and diff it
item-by-item against `engine/gdn_rec_loop_kernels.h::rec_loop_kernel`. §38's 5.76× is the object; every
*shape* explanation is measured dead (BET.md §11's table), so the remaining axis is per-token work.

CPU-only throughout: no device work, no timing of my own. Every ms in this file is §38's traced figure or an
arithmetic consequence of it, cited as such.

## 0. PROVENANCE: is this the kernel that ran? Yes — and §38's geometry reading of it is wrong

Four links, each checked:

1. **Tree revision.** `engines/llama.cpp` is its own git repo (gitignored by the parent at `.gitignore:16`),
   HEAD = `797de63b2fa7396dcc8caca76834c699ac84e1c3`, tag `b11716`, working tree clean
   (`git -C engines/llama.cpp status --short` → empty). That is exactly the revision §38 records
   ("llama.cpp is still `797de63`", `progress/2026-10-21/performance-status-2026-10-21.md:1733`). So the source I can read is the source
   the binary was built from; no local edits.
2. **The binary is built from this tree.** There is one build directory, `engines/llama.cpp/build-hip`
   (`GGML_HIP=ON`), and the HIP target compiles the CUDA source in place:
   `build-hip/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/gated_delta_net.cu.o` (mtime Oct 18 04:13)
   feeds `build-hip/bin/libggml-hip.so.0.27.0` (Oct 18 04:25) and `build-hip/bin/llama-bench` (Oct 18 04:25),
   all older than the §38 window (Oct 19 03:25). **Note the brief's paths are wrong**: there is no
   `engines/llama.cpp/bench/build/bin/` and no `ggml-cuda.so`; the artifacts are `build-hip/bin/llama-bench`
   and `build-hip/bin/libggml-hip.so`. Cosmetic — the right binary is unambiguous, it is the only one.
3. **The instantiation matches.** `nm -C` on that object file lists
   `void gated_delta_net_cuda<128, false, false, 4>(float const*, …)` among 16 instantiations
   (`S_v ∈ {16,32,64,128} × KDA × keep_rs`), and §38's trace names the running kernel as exactly
   `void gated_delta_net_cuda<128, false, false, 4>`
   (`progress/bench/evidence/llamaphase-2026-10-19/trace_llama_pp4096.txt`). `KDA=false` (scalar `g` per head,
   which is Qwen3.5's form), `keep_rs_t=false` (no per-token state snapshot), `S_v=128` (= our `D`),
   `cols_per_warp=4` (`gdn_cols_per_warp`, `gated_delta_net.cu:7`).
4. **The launch geometry in the source reproduces the traced geometry exactly** — once the trace is parsed on
   all three dimensions, which §38's census did not do. See §1.

So: no source/binary mismatch. But the census that §38 used to characterise llama's kernel mis-read it, and the
misreading is load-bearing for BET.md §4/§10 — see the next section. It is the most important thing in this
document.

## 1. FINDING (the big one): llama's GDN kernel does **not** run one wavefront per block — it runs 8× more blocks than we do at our block shape

`tools/trace_phase_split.py --geom` (the census §38 quotes) builds its key from **`Workgroup_Size_X` and
`Grid_Size_X` only** (`tools/trace_phase_split.py:76-84`, `blocks = grid // wgsize` at `:117`). rocprofv3
reports grid size in **threads per dimension**, and llama's launch is three-dimensional, so the X-only read
sees `wgsize 32`, `grid 512` → "16 workgroups, 0.80 blocks/CU" — and that is how §38 came to write
"llama runs `gated_delta_net` at 32 threads × 512 grid and ours at 128 × 2,048, and **both land at 0.80
blocks/CU** — 16 workgroups on a 20-CU part, four CUs idle, in both engines"
(`progress/2026-10-21/performance-status-2026-10-21.md:1758`).

Reading all three dimensions (`prototype/gdn_walk/g3_trace_geom.py`, a 60-line re-parse of the same two CSVs
§38 already produced — CPU-only, no new measurement):

| engine | kernel | workgroup | grid (threads) | **blocks** | **blocks/CU** (÷20, §38's convention) | VGPR (trace / object) | LDS | traced ms/pass |
|---|---|---|---|---|---|---|---|---|
| llama | `gated_delta_net_cuda<128,false,false,4>` | 32×4×1 = **128 thr** | 512×4×8 | **128** | **6.40** | 80 / 78 | **0** | 47.20 (§38) |
| ours | `rec_loop_kernel` | 128×1×1 = 128 thr | 2048×1×1 | 16 | 0.80 | 80 / 75 | 1,536 B | 272.07 (§38) |

**On the CU count**: `blocks/CU` above follows §38's convention of dividing by HIP's `multiProcessorCount = 20`,
which AGENTS.md records as halved against the architecture. On the architectural 40 CUs both numbers halve
(3.2 vs 0.4 blocks/CU; 12.8 vs 3.2 wavefronts/CU) and **the 8× ratio is unchanged**. The figure that needs no
CU-count assumption is wavefronts per SIMD, from KFD `simd_count 80`: **6.40 against 0.80** (of a 16 maximum).
The VGPR column shows the trace's rounded value and the code objects' own `vgpr_count` (8-granularity).

`(grid_x·grid_y·grid_z)/(wg_x·wg_y·wg_z) = (512·4·8)/(32·4·1) = 128 blocks`, and grid_x in threads
(512) / block_x (32) = 16 = `H`, grid_z/block_z = 8. That is precisely the source's
`grid_dims(H, n_seqs, S_v/(cols_per_warp*num_warps))` = (16, 1, 128/(4·4)=8) and
`block_dims(warp_size, num_warps, 1)` = (32, 4, 1) — `gated_delta_net.cu:200-207`, with `num_warps` staying at
its initial 4 because the fill-the-device shrink loop (`:196-199`) tests `16·1·8 = 128 < nsm=20`, which is
false. **The source and the trace agree in every field, including LDS = 0 and VGPR = 80.** The disagreement is
purely in §38's parsing of the CSV.

Consequences, in the order they matter:

- **llama has 512 wavefronts in flight per launch; we have 64 — 8×.** 6.4 wavefronts per SIMD against our 0.8
  (80 SIMDs: KFD `simd_count 80`, AGENTS.md counter-trap row). Both engines are 4 wavefronts per block at the
  same register budget; the entire difference in occupancy is that llama splits the 128 *columns* of the state
  across 8 z-blocks while we give one head's whole 128×128 block to a single 128-thread block.
- **"four CUs idle, in both engines" is false for llama.** Its 128 blocks cover the machine 6.4× over on
  HIP's 20-CU view, 3.2× on the architectural 40; ours cannot cover it on either.
- **BET.md §4's premise for G2 is refuted by the source it was read from.** §4 says G2's shape — "one wavefront
  per block, no LDS, no `__syncthreads()` … per-thread ownership of several state columns" — "is structurally
  what `gated_delta_net_cuda` is". It is not: llama's block is 4 wavefronts (not 1) and each thread owns
  **16** state elements (not "several columns" = 512). G2 therefore tested a geometry that differs from
  llama's in the one axis that §38's own number measures — wavefronts per CU — and moved it **4× the wrong way**
  (0.8 → 0.2 wf/SIMD). §42's conclusion "the GDN gap is not shape" is sound about *that* experiment and
  unsound as a general closure: the occupancy axis was never touched, it was mislabelled.
- What G2 *did* prove: the barriers and the LDS round-trip are worth 5.9 % (G1) and deleting them while cutting
  wavefronts costs 2.2×. That stands. What got closed on a bad premise is the claim that llama is not
  out-occupying us.

**One gate-side follow-up, cheap:** `trace_phase_split.py --geom` should key on all three dims (or refuse) —
it cannot fail on a 3-D launch, which is exactly the shape llama uses. Not fixed here (analysis-only item, and
it is a tool in `tools/`, so it is a separate commit if the main agent wants it).

## 2. Item-by-item decomposition, llama's kernel against our shipped walk

Left = `engine/gdn_rec_loop_kernels.h::rec_loop_kernel` (the `--gdn-loop 512` form that ships, launched at
`engine/kernels_gdn.cpp:128` as `<<<NH=16, D=128>>>`). Right =
`engines/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`, cited below as **`gdn.cu`**, instantiation
`<S_v=128, KDA=false, keep_rs=false, cols_per_warp=4>`. Where a claim is from the compiled object rather than
the source it says so; the object is `gated_delta_net.cu.o` from `build-hip`, carved with
`tools/carve_codeobjects.py`.

| axis | ours, `rec_loop_kernel` | llama, `gated_delta_net_cuda<128,false,false,4>` |
|---|---|---|
| **grid / block** | 16 blocks × 128 threads (`kernels_gdn.cpp:128`), 4 wavefronts/block | grid `(H, n_seqs, S_v/(4·num_warps))` = 16×1×8 = 128 blocks × `(warp_size, num_warps)` = 32×4 = 128 threads, 4 wavefronts/block (`gdn.cu:203-204`) |
| **wavefronts on the device** | **64** → 0.80/SIMD | **512** → 6.40/SIMD |
| **who owns a state column** | one thread owns one whole column: thread `j` owns `S[i][j]`, i = 0…127 (`:129-133,138-145`) — **128 elements/thread** | 8 lanes own one column, each holding every 8th row: `i = r·lanes_per_col + lane_in_col`, r = 0…15 (`gdn.cu:43-44,48-50,68-73`) — **16 elements/thread** |
| **state layout** | `S[h][i][j]`, address `h·D·D + i·D + j`, i.e. `[row i][col j]`, column index contiguous (`:123,129`) | `state[(seq·H + h)·S_v² + col·S_v + i]`, i.e. `[col][i]` — **the transpose of ours**; comment says so at `gdn.cu:66` |
| **per-thread access pattern** | stride `D`·4 B = 512 B between its own elements; a wavefront request is 32 contiguous lanes = 128 B, so perfectly coalesced *per instruction*, no per-thread locality | the 16 elements are stride-8 in `i`; a wavefront request = 4 column-groups × 8 contiguous lanes = 4 × 32 B sectors. Coalesced to full sectors, worse per-thread locality |
| **state residency across tokens** | **none.** Re-read from global twice per token (loop A, loop B) and written once per token (`:128-145`) | **registers.** Loaded into `float s_shard[16]` before the token loop (`gdn.cu:70-73`), written once after it (`gdn.cu:172-177`). Per token: zero state memory ops |
| **state ops per token, whole grid** | 786,432 element accesses (2 reads + 1 write of 262,144) | 1,024 = 2 × 262,144 / 512 tokens, amortized |
| **vector width emitted** | **none wide.** All global traffic `global_load_b32` / `global_store_b32` (37 loads, 17 stores static); `ds_load_b128` × 12 for the norm's LDS round-trip | **none wide.** All global traffic `global_load_b32` × 51, `global_store_b32` × 17. No `b64`/`b128`, no `float4` — a thread's own elements are strided, so LLVM cannot vectorize either side |
| **delta-rule factoring** | rank-1 identity: pass A `r = Σ S[i][j]·k̂[i]`; `dv = β(v − γr)`; pass B fuses `S' = γS + k̂·dv` **and** `o_j = Σ q̂[i]·S'[i][j]` (`:126-145`) — 2 reads + 1 write fused into the update | **the same identity.** `kv = Σ s_shard·k`, `delta = (v − g·kv)·β`, then one fused pass `s_shard = g·s_shard + k·delta; attn += s_shard·q` (`gdn.cu:99-116`). Also 2 reads + 1 write, also one update pass |
| **where the k/q norms come from** | **inside the walk, every token:** `block_sum_pair_f32` (`engine/gdn_rec_fused_kernels.h:82`) = two 5-level `__shfl_down_sync` trees + a `red[4]` LDS hop + a second tree, then `kn`/`qn` written to `__shared__` with 2 more barriers (`:112-120`) | **not in the kernel at all.** The graph does `build_gdn_l2_norm` = `ggml_scale(ggml_rms_norm(x, eps/n), 1/√n)` (`src/models/models.h:14-18`, called at `src/models/qwen35.cpp:441-442`) as separate nodes, which land in §38's `norm` phase (llama 11.43 ms/pass, ours 11.65 ms). The kernel only multiplies the output by `scale = 1/√S_v` (`gdn.cu:122`, `:275`) |
| **row reductions** | 128-thread block-wide: 10 `__shfl_down_sync` + LDS + `__syncthreads` + 10 more, ×2 values, per token (G1 measured this worth −5.9 %) | `warp_reduce_sum<8>` = 3 `__shfl_xor_sync` + 3 adds, twice per token, **no LDS, no barrier** (`ggml/src/ggml-cuda/common.cuh:468-476`; 6 `ds_bpermute_b32` in the object — LLVM lowering of a wave32 butterfly) |
| **global ops per token per thread** | 3 operands (`k`,`q`,`v` `:111,134`) + `g`,`beta` (`:122`) + **384 state accesses** (`:128-145`) + 1 store (`:148`) | 32 k/q + 1 v + 1 g + 1 β = **35 loads** (`gdn.cu:84-94`) + 1 store on 1 of 8 lanes (`:121-122`) + **0 state ops**. ISA witness: the token loop contains exactly 35 `global_load_b32` and 44 `v_fmac_f32_e32`; the 16 state loads are before the loop and the 16 stores after it |
| **k/q read redundancy** | none — each k element is read by exactly one thread | **32×**: `k_t[i]` ignores `col_in_warp` (4× within a wavefront) and every z-block re-reads the same head row (8×) |
| **tokens per block per launch** | `tokens` = the span, 512 at 4,096 lanes (`decode_step.h:121` default, `decode_step.cpp:746,813`) | `n_tokens` = the ubatch, 512 with `-ub 512` (`gdn.cu:75`) — **identical span on both engines** |
| **instructions per token per wavefront** | **1,327** slots (token_bill.py; walks 352 + 720 = 1,072 = 81 %) | **213** instructions in the token loop, measured out of the object (`g3_isa_bill.py`: span `0x14524..0x1499c`, 35 loads / 44 fma / 6 bpermute / 1 exp / 0 barriers) |
| **whole-kernel instruction count** | 835 static | 424 static |
| **instruction class** | 0 matrix, 0 dot-class (the engine has none anywhere, §26) | **0 matrix, 0 dot-class** — `tools/kernel_isa_census.py` on the object: `MATRIX=0 DOT=0 FMA32=47 LDS=6 VMEM=68 SALU=212` |
| **registers / LDS / scratch** | 75 VGPR, 0 spill, 0 scratch, 1,280 B static LDS (trace reports 1,536 = +256 driver reserve) | 78 VGPR, 0 spill, 0 scratch, **0 LDS**, `max_flat_workgroup_size 128`, `wavefront_size 32` (amdhsa metadata of the object). 48 of the 78 are the `s_shard`/`k_reg`/`q_reg` payload — which is the witness that the state really is register-resident |
| **launch structure** | 144 launches/pass | 144 launches/pass (§38) |

Two notes on fairness, because the table flatters us less than it looks.

1. **The algebra is identical, and so is the arithmetic.** Both spellings execute four FMA-class operations per
   state element per token, so the pass-wide demand is the same 7.73 × 10¹⁰ MAC whatever the thread map
   (`g3_work_table.py`). Nothing in this census is a claim that llama computes less: it computes the same
   4 FMA/element and gets 3.28 TFLOP/s out of the same scalar pipe we get 0.57 from.
2. **llama's norms are not free either** — they moved into two extra graph nodes per tensor per layer, which is
   why §38's `norm` phase is 11.43 ms on their side and 11.65 ms on ours. Comparing our `gdn_rec` row against
   their `gdn_recur` row silently bills us for work they put in another row. That is G1's externality lesson
   (BET.md §8) seen from the other engine, and it is one reason the G4 lines below are written on `wall_ms` too.

## 3. The work table, and the ceilings each side is measured against

`prototype/gdn_walk/g3_work_table.py` (output saved to
`prototype/gdn_walk/evidence-g3-worktable-2026-10-19.txt`). Inputs are all cited: §38's traced times,
token_bill.py's per-token bill for our walk, this file's ISA read of llama's token loop, `g3_trace_geom.py`
for both geometries, and the machine roofs from `prototype/issue_rate_probe/results.md` §1 (40 CU, 2.894 GHz,
scalar `v_fma_f32` 7.30 TFLOP/s) and `devices/strix-halo-memory-bandwidth.md` (237 GB/s DRAM plateau; **918 GB/s
at an 18 MiB working set — and that working set is literally the 18 GDN states**).

```
                                  quantity              ours             llama  ratio
                     threads participating             2,048            16,384  8.0x
                                wavefronts                64               512  8.0x
               wavefronts per SIMD (of 16)              0.80              6.40  8.0x
                 state elements per thread               128                16  0.125x
             slots per token per wavefront             1,327               213  0.161x
               slots per token, whole grid            84,928           109,056  1.28x
           global state accesses per token           786,432             1,024  0.0013x
       bytes demanded at the LSU per token           3.18 MB            553 KB  0.174x
     of which cannot come from L1 (unique)           2.13 MB             37 KB  0.0174x
                  demanded GB/s at the LSU               861               864
  as % of the 918 GB/s cache roof (LSU)              94 %              94 %
  as % of the 918 GB/s cache roof (unique)            63 %             6.3 %
   as % of the 237 GB/s DRAM roof (unique)           244 %            24.4 %
                        achieved G slots/s              23.0             170.3  7.40x
       as % of the 231.5 G-slot issue roof             9.9 %            73.6 %
             achieved TFLOP/s (scalar f32)              0.57              3.28
  as % of the 7.30 TFLOP/s scalar fma roof            7.8 %            44.9 %
                      traced ms/pass (S38)            272.07              47.2  5.76x

our slots for the pass: 6.26 G -> at llama's 74 % issue efficiency = 36.8 ms (vs 272.07 ms now); at 50 % = 54.1 ms
a llama-shaped grid (512 wf) carrying ~213 slots/token costs 8.04 G slots;
  at 74 % of roof = 46.9 ms; at 40 % = 86.8 ms; at 25 % = 138.9 ms
state arithmetic floor: 21.2 ms at the 7.30 TFLOP/s roof
```

## 4. Adjudication: does each difference have 5.6× in it?

Every candidate below is checked against a **measured** ceiling, and the ones that need llama to beat a machine
limit are killed in the row. All three live candidates are the *same* structural choice seen three ways, and
§38's `0.80 blocks/CU in both engines` mis-read is what hid it.

| # | difference | magnitude | ceiling check | verdict |
|---|---|---|---|---|
| **D1** | wavefronts per SIMD: **0.80 → 6.40** (8×) | our walk issues at **9.9 %** of the 231.5 G-slot roof; llama at **73.6 %** — a 7.4× issue-efficiency gap, which brackets 5.76× on its own | 6.40 wf/SIMD is 40 % of the part's 16 waves/SIMD (`simd_count 80`, `max_waves_per_simd 16`), and 78 VGPR at 128 threads leaves register headroom for 10 waves. Legal, with room | **alive — the leading candidate** |
| **D2** | state residency: ours re-reads+writes 1 MiB/token/layer (3 LSU passes over every element); llama holds 16 F32/thread in registers and touches memory once per 512 tokens. Per thread per token: **387 state+operand LSU ops → 35**. Distinct bytes/token/layer 2.13 MB → 37 KB | our unique-byte demand is **577 GB/s = 63 % of the measured 918 GB/s cache roof** for an 18 MiB working set (which *is* the GDN state), and **244 % of the DRAM roof**, so it must be cache-served and we are near the only roof this box has measured for it. llama at 58 GB/s is 6 % of that roof | **measured counter-evidence: route A.** Putting the state in LDS for the whole body bought **+10.6 %** (`BET.md` §1). Route A is a sharper witness than it looks: it ran 64 blocks (`dim3(kStrips=4, kNH=16)`, `kernels_gdn.cpp:323`) but "the recurrence runs on warp 0 alone" (`engine/gdn_collapse_kernels.h:31`), so its *useful* wavefront count was 64 — ours exactly — and it traded 384 global ops per thread per token for 384 LDS ones, 1-for-1. It removed bytes below L1 while leaving occupancy and LSU op count alone, and that is worth ~10 %. **That bounds residency-alone at tens of percent, not 5.76×** | **alive but not sufficient alone**; potent only next to D1 |
| **D3** | total instruction count | llama issues **1.28× MORE** slots per token per layer than us (109,056 vs 84,928) despite 6.2× fewer elements per thread, because of D8 | the claim dies on sign, not magnitude: the engine that issues more instructions finishes 5.76× faster | **dead — wrong sign.** Any "our walk has too many instructions" framing is refuted by the census |
| **D4** | vector width | neither side emits a wide global access: ours 37 `global_load_b32`/17 `global_store_b32`, llama 51/17, no `b64`/`b128` on either | not a difference at all. Both thread maps stride each thread's own elements, so LLVM cannot vectorize either | **dead — absent** |
| **D5** | instruction class (matrix / dot) | `tools/kernel_isa_census.py` on the traced object: `MATRIX=0 DOT=0` for all 16 instantiations | llama's 3.28 TFLOP/s is **45 % of the measured scalar `v_fma_f32` roof of 7.30**, so it needs no matrix unit, no `dot2`, nothing exotic | **dead — absent.** This matters: closing to llama needs no new instruction class, unlike the attention item (§26) |
| **D6** | k/q norms inside the walk vs in the graph | G1 measured hoisting at **−5.9 %** of the row; BET §2(a) sized the machinery at 9.2 % of the walk's slots and device-verified it (0 barriers, 0 LDS ops in the hoisted walk) | two orders of magnitude short of 5.76×, and hoisting costs traffic elsewhere (+6.06 ms outside the row) | **dead as the explanation**; a real ~6 % and a *precondition* for D1's map (see G4's constraint 3) |
| **D7** | barriers and LDS | G1 −5.9 %; G2 removed every barrier and all LDS **and** cut wf/SIMD to 0.20 → +121.9 % | G2 is confounded by D1 moving 4× the wrong way at the same time, so the barrier share stays at ≈ G1's 6 % and the −122 % belongs to D1's downside | **dead alone**, and G2's number is now evidence *for* D1, not against shape |
| **D8** | llama's 32× redundant k/q reads (4× within a wavefront because `k_t[i]` ignores `col_in_warp`, 8× because every z-block re-reads the head row) | costs llama ~517 KB/token/layer of L1 demand — most of its 553 KB total | a **cost** llama pays, not a win. Note the coincidence it produces: LSU-level demand is 861 GB/s for us and 864 GB/s for llama, both 94 % of the 918 GB/s roof. Read literally that says both engines are fabric-bound and the gap is the byte ratio (5.8× ≈ 5.76×) — but route A already falsifies that reading, because route A kept the same op count and moved only the bytes and gained 10.6 %. Recording the coincidence, rejecting the story | **dead**, and a warning against `bytes ÷ time` roofline (the §26/D21 lesson) |

**The census's answer is therefore not "a broad instruction-mix gap".** One axis explains the size: **state
elements per thread, 128 → 16.** Halve-and-halve that number and three things follow arithmetically — 8× the
wavefronts (D1), the thread's slice fits in 16 VGPRs so the state stops being memory (D2), and per-thread LSU
ops per token fall 11×. Every bet we have made on this family kept 128 elements per thread: G1 (norm hoist),
G2 (4 and 2 *columns* per lane = 512 and 256 elements per thread, i.e. 4× and 2× the *wrong* way), route A
(128 elements/thread in LDS), the chunked scan, the unroll. **G2's own table is the confirmation**: it moved
this dial from 128 to 512/256 elements per thread and lost 2.2×/1.44×, and it never tested 32 or 16.

## 5. What this census cannot settle

- **Which of D1 and D2 does it.** They move together in G4, so a single A/B cannot attribute. If G4 lands near
  llama's number, no attribution is needed to ship it; if it lands mid-range, the discriminator is a hybrid
  form — 16 elements/thread with the state forced to stay in global (a `volatile` slice or an explicit spill)
  — which supplies D1 without D2. That is a device experiment and a second kernel; recorded, not funded here.
- **The issue-efficiency figures are derived, not counted.** 9.9 % and 73.6 % are slots ÷ (time × roof) with
  §38's traced times. A `rocprofv3` VALU-busy / SIM-clock pass would measure it directly; that is main-agent
  device work and it is the check to run before spending on G4 if the main agent wants the premise confirmed.
- **Traced ms are upper bounds** (§38's own caveat), inflated roughly in proportion to launch count; both
  engines have 144 launches in this family, so the 5.76× is about the cleanest row in §38's table.
- **The 918 GB/s roof was taken with a float4 grid-stride reader** (`devices/strix-halo-memory-bandwidth.md`
  §0b), not with our column-striding pattern, so "63 % of the cache roof" is indicative, not a bound.
- **The 32× k/q redundancy is read out of the source's index arithmetic** (`gdn.cu:48-50,88-94`: the k address
  is `k_t[r·8 + lane_in_col]`, which cannot depend on `col_in_warp`; `col` depends on `blockIdx.z`) — I counted
  the loads in the ISA but did not decode their addresses.
- **§38 may understate llama.** `gated_delta_net.cu:355-361` tries an MMA-based path first, and on HIP it is
  gated at `gated-delta-net-mma.cu:393-397` to `GGML_CUDA_CC_IS_RDNA3_5 && n_tokens >= 2048`. Our §38 window
  ran `-ub 512`, so `n_tokens = 512` and the fallback ran — which is the kernel this file reads, and it is what
  any AMD user gets at a matched batch. But at `-ub 2048`+ llama would take a WMMA schedule we have never
  measured. If llama is ever re-compared, note that `-ub 512` is a *matched* choice (it is our wave size, per
  `kMaxWaveRows`), not a charitable one.
- **Nothing here says the GDN family is 47 ms is reachable in one step.** The model in §3 says a faithful port
  lands between 47 ms (llama's issue efficiency) and 167 ms (25 % of roof with our norm overhead).

## 6. G4 — the one bet this census implies

**G4: ship llama's thread map under our recurrence — 128 blocks × 128 threads, 8 lanes per state column, 16
state elements per thread held in registers for the whole 512-token span.** It is not a shape tweak in the G2
sense: it is the only change that supplies D1, D2 and the LSU-op cut at once, and the map is *forced* by the
arithmetic — `wavefronts × 32 × elements_per_thread = 262,144`, so raising wavefronts means lowering elements,
and 16 elements is what fits in a register file that already holds k̂ and q̂.

Constraints it must satisfy, each carried from a measured failure rather than taste:

1. **`wf × 32 × el/thread == 262,144` asserted in the source**, 128 blocks × 128 threads, 16 state
   elements/thread, `el/thread` a compile-time constant. Concretely llama's own shape: grid
   `dim3(NH, D/16)` = (16, 8) with block `dim3(32, 4)` — 16 columns per block, 8 lanes per column
   (`gdn.cu:203-204`). One address expression shared by the state load, the update and the store — the §20.1
   aliasing trap is exactly what a two-layout branch produces.
2. **State in registers across the span, no LDS, no staging.** The register budget is not a hope: llama's
   object at the same map reports `vgpr_count 78, vgpr_spill_count 0, private_segment_fixed_size 0` with 48 of
   the 78 spent on `s_shard`/`k_reg`/`q_reg`. Our walk is at 75 VGPR today, so the same shape should fit.
3. **Norms stay inside the walk but become 8-lane butterflies.** Under this map the 8 lanes of a column group
   hold the entire 128-element k row between them (and hold it redundantly in all 4 groups), so `Σk̂²` is a
   3-level butterfly over thread-local slices — no block reduction, no barrier, no staging. That removes the
   reason G1 had to write normed copies (its +32 KB/token/layer externality). Cost ≈ 2 × (16 FMA + 3 shfl) ≈
   40 slots/token/wf. **If that shows up in the measurement**, the fallback is to stage only the two row sums
   per head per token — 128 B/token/layer, 256× less than what G1 staged — and multiply the *dots* by the norm
   scalars (mathematically exact since k̂ = k·c and q̂ = q·c with c per (head, token); it re-associates, so it is
   judged by the §10.1 aggregates, not by bit-identity).
4. **The state layout does not have to change.** Ours is `[i][j]`, llama's is `[col][i]`. Under this map a
   thread's 16 elements in our layout are stride-512 B, which costs 32 sectors per wavefront load instead of 4
   — but that load happens **once per 512 tokens**, so it is 8 MiB per launch per layer ≈ 16 KB/token/layer
   against the 3 MiB/token/layer we pay now. Transposing is an option for later, not a precondition, and not
   touching the layout keeps every state consumer (`state.h:107`, `STATEFP`, the decode handoff) unchanged.
5. **Numerics: pre-declared.** The state dot re-associates (16-element partials + an 8-lane butterfly), so `o`
   and `STATEFP` move. Judged by the existing §10.1 aggregates through `engine/gate.py` on the pinned fixtures,
   **no new band, no widening, no `--ack-spill`** (BET §10 rule 2, unchanged).
6. **Canary, asserted inside the timing window** (AGENTS.md, 2026-10-17): the A/B must show `launches=144` on
   both sides (this change must not move submissions) **and** must read the new geometry — 128 blocks × 128
   threads on the new side, 16 × 128 on the base — from a `--geom` pass in the same hold. That read requires
   `trace_phase_split.py --geom` to key on all three grid dimensions first, or it will print 16 blocks for a
   3-D launch and the canary will be blind.

**Pre-registered lines, both denominators, medians of 3 at 4,096 lanes in one lock hold, three sides
(base / new-form-off / new-form-on), per BET §10 and §40.** Bases: `gdn_rec` 264.00 ms and `wall_ms` 626.44 ms
(§42's window; §40's were 266.60 / 626.75 — the lines are written so either base gives the same verdict).

| metric | base | live | dead | why that number |
|---|---|---|---|---|
| `gdn_rec` family ms | 264.00 | **≤ 170** | **> 225** | The model says a faithful port lands 47–167 ms (§3: 8.04 G slots at 74 % / 40 % / 25 % of the issue roof = 46.9 / 86.8 / 138.9 ms, +20 % for the norm butterflies). 170 is just inside the pessimistic end of the model, and is a 1.55× family win (−94 ms of wave ≈ +15 % prefill tok/s). 225 = less than 1.17×, i.e. below what justifies a rewrite of the shipping walk and its numerics risk |
| wave `wall_ms` | 626.44 | **≤ 540** | **> 607** | Row −94 ms lands the wave at 532 ms; the line at 540 allows ≤ 20 ms of externality, which is 3× G1's measured +6.06 ms and 7× G2's +2.69 ms. Dead line = base −3 %, §10's convention |

Both must clear their live line to ship; either past its dead line kills the form; anything else is unresolved
and not shipped. **The `wall_ms` line is the ship gate.**

## 7. Artifacts this file added, and how to re-run them

| file | what it is |
|---|---|
| `prototype/gdn_walk/g3_trace_geom.py` | re-parses §38's two `rocprofv3` CSVs on all three grid dimensions and prints blocks/CU and wf/CU for the GDN kernels. CPU-only, reads files already on disk |
| `prototype/gdn_walk/g3_isa_bill.py` | per-kernel ISA bill over a carved code object: amdhsa resource metadata (VGPR/SGPR/spills/LDS/scratch), instruction-class mix, load/store widths, and the span of every backward branch with the instruction mix inside it — which is how the token loop is found without guessing. Refuses a missing symbol, an empty disassembly, and prints `NO LOOP` rather than a fabricated per-token figure |
| `prototype/gdn_walk/g3_work_table.py` | the §3 table and every ceiling ratio, with each input named and cited in its docstring |
| `prototype/gdn_walk/evidence-g3-worktable-2026-10-19.txt` | its output |
| `prototype/gdn_walk/evidence-g3-isa-2026-10-19.txt` | the carve + both ISA bills + the matrix/dot census, as run |

Commands (all CPU-only, no device, no timings taken by me):

```
git -C engines/llama.cpp log -1 && git -C engines/llama.cpp status --short
nm -C engines/llama.cpp/build-hip/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/gated_delta_net.cu.o | grep delta
python3 prototype/gdn_walk/g3_trace_geom.py
python3 tools/carve_codeobjects.py engines/llama.cpp/build-hip/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/gated_delta_net.cu.o tmp/gcc/g3_isa --target gfx1151
/opt/rocm/llvm/bin/llvm-objdump -d tmp/gcc/g3_isa/co_036864.co > tmp/gcc/g3_isa/gdn.s
python3 prototype/gdn_walk/g3_isa_bill.py tmp/gcc/g3_isa/gdn.s tmp/gcc/g3_isa/co_036864.co "ILi128ELb0ELb0ELi4E"
python3 tools/kernel_isa_census.py tmp/gcc/g3_isa/gdn.s --kernels gated_delta_net_cuda
python3 prototype/gdn_walk/g3_work_table.py
```

`tools/carve_codeobjects.py` was reused rather than re-invented (it already carves AMDGPU objects out of a host
binary and its canary checks the target id per object — AGENTS.md: "before writing a census tool, list the
census tools"). **No device work was run for this file**: no gate, no timing, no `util` reading to report.

## 8. Ledger corrections this census asks for (not applied — `bench/` and `progress/` are cited by line)

1. **§38's `--geom` sentence** (`progress/2026-10-21/performance-status-2026-10-21.md:1758`): "llama runs `gated_delta_net` at 32
   threads × 512 grid and ours at 128 × 2,048, and **both land at 0.80 blocks/CU** — 16 workgroups on a 20-CU
   part, four CUs idle, in both engines" is wrong about llama. Correct: **128 blocks of 128 threads, 6.40
   blocks/CU, 25.6 wavefronts/CU, no idle CU**, from the same CSVs. The traced ms and the 5.76× are unaffected.
2. **BET.md §4 and §10** describe G2's one-wavefront-per-block geometry as "structurally what
   `gated_delta_net_cuda` is". It is not (`gdn.cu:203-204`, and §1 above). G2 was built on that sentence;
   §42's conclusion "the GDN gap is not shape" should be narrowed to "not *that* shape change" — the
   wavefronts-per-CU axis is untouched and G4 is the test of it.
3. **`tools/trace_phase_split.py`** should key `--geom` on all three grid dimensions, or refuse a multi-
   dimensional launch. As written it cannot see a 3-D grid, and any future canary that reads geometry through
   it inherits this bug.
4. **AGENTS.md trap row candidate**: *a profiler column read on one axis is not a geometry.* `rocprofv3`'s
   `Grid_Size_*` are per-dimension thread counts, and a kernel launched `dim3(16,1,8) × dim3(32,4)` reads as
   "32 threads, 512 grid" if only `_X` is read — which then agrees suspiciously well with the launch one is
   already imagining.
