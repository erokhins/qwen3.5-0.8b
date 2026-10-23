# How llama.cpp makes a verify row nearly free — mechanism account

Source read-only against `engines/llama.cpp` @ `797de63` + the `build-hip` cache on disk. No
device work, no build, no timing. Tags: **[source]** = code says it, **[inference]** = deduced.
Snippets ≤5 lines. `llama-cli` here starts the server in-process (`tools/cli/main.cpp:1-5`,
`tools/cli/cli-server.h:7-8`), so the loop that produced our A/B numbers is
`tools/server/server-context.cpp`. **[source]**

## 0. What this build compiles (and a claim in the parent's brief that was wrong)

- `build-hip/CMakeCache.txt`: `GGML_HIP=ON`, `GGML_CUDA=OFF`, `CMAKE_HIP_ARCHITECTURES=gfx1151`,
  `GGML_CUDA_FA=ON`, `GGML_CUDA_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`, `-O3 -DNDEBUG`; CUDA sources are
  compiled as HIP (`build-hip/ggml/src/ggml-hip/…/__/ggml-cuda/ggml-cuda.cu.o`). **[source]**
- `gfx1151` ⇒ `RDNA3_5` **and** `__GFX11__` ⇒ `RDNA3` (`vendors/hip.h:218-228`), and `common.cuh:281`
  `#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))` → `#define AMD_WMMA_AVAILABLE`,
  with `amd_wmma_available(cc) = IS_RDNA4 || IS_RDNA3` (`common.cuh:357`, verified by the parent).
  **So the WMMA fattn body is compiled and selectable here** (`fattn-mma-f16.cuh:1821` lists
  `AMD_WMMA_AVAILABLE`) — but see Q2: for THIS model it is only selected at ≥ 5 query rows, so a k=1
  or k=2 verify never reaches it. Only *sparse* FA (`fattn.cu:129-151`), `cp.async` (`common.cuh:302`)
  and `ggml_cuda_fattn_mma_get_swizzled` stayed CUDA-only, so HIP **pads instead of swizzling**
  (`fattn-mma-f16.cuh:1244-1247` `stride_tile_K = swz ? nbatch_K2 : nbatch_K2 + 4`). **[source]**
- **Correction to this section's first draft.** It opened by saying the above "retires the `AGENTS.md`
  row 'the HIP fattn path is not the WMMA one'" and "the premise of the `LLAMA-PORT-PLAN.md` correction
  banner". Neither is true, and the false premise came from the parent's brief rather than from the
  repo: `AGENTS.md` carries no such row (grep it — the WMMA rows are about a retracted TFLOP/s route and
  about *our own* kernels containing no matrix instructions), and the `LLAMA-PORT-PLAN.md` banner says
  llama *pads rather than swizzles and stages synchronously* on this platform, which this section
  **confirms** rather than refutes. The live fact is narrower than the sentence it replaces: WMMA is
  compiled, and it is still not what a verify-sized batch runs. **[source]**
- Fixture (`prototype/mtp_llama/gguf_head_meta.py`, added as the probe for these shapes, on
  `tmp/mtp_llama/Qwen3.5-0.8B.mtp.bf16.gguf`): `qwen35`, `block_count=25` (24 trunk + 1 nextn),
  `recurrent_layers` ⇒ **18 GDN + 6 full attn**, `head_count=8`/`head_count_kv=2` (gqa 4),
  `key_length=value_length=256` (DKQ=DV=256), `ssm.state_size=128`, `group_count=16`,
  `time_step_rank=16` (= num_v_heads, `qwen35.cpp:355-357`), `conv_kernel=4`, `embedding_length=1024`.
  GDN state/layer = 128·128·16·F32 = **1 MiB** ⇒ **18 MiB/seq** — the 18.0 MiB
  `engine/mtp_verify.h:30` carries. **[source]**
- Harness (`prototype/mtp_llama/llama_mtp_ab.sh:70-72`): `-ngl 99 -fa on -c 5120 --temp 0`,
  `--spec-type draft-mtp --spec-draft-n-max {1..4}`. **[source]**

## Q1. Verify batch shape

One `llama_process(..., DECODE, batch)` per round, `n_tokens = 1+k` rows, **one sequence**,
consecutive positions, every row `output=true` (`server-context.cpp:565-578`):

    spec_i_batch.push_back(batch.size());                      // :567
    add_ok &= batch.add(id, sampled, pos0++, true, false);     // :574
    for (auto token : spec_draft)
        add_ok &= batch.add(this->id, token, pos0++, true, false);   // :576

`prompt.tokens.insert(spec_draft)` advances positions (`:583`); accept/rollback reads
`common_sampler_sample_and_accept_n(smpl, ctx_tgt, spec_i_batch, spec_draft)` (`:4543`,
`common/sampling.cpp:690`). With `equal_seqs()` the recurrent ubatch is
`split_equal(n_ubatch, true, n_rs_seq+1)` (`llama-memory-recurrent.cpp:459-461`) ⇒ 5 rows ≤ 512 is
one ubatch. **[source]**

There is **no separate decode graph**: `llm_graph_type` is per context *type* (`DEFAULT` vs
`DECODER_MTP`, `llama-graph.h:41-46`, `llama-context.cpp:114-115`) and `process_ubatch` is
shape-agnostic (`:1507`). What a multi-token batch buys, and what it does not:
- **whole-graph reuse**: `if (!graph_reuse_disable && gf_res_prev_active == res &&
  res->can_reuse(gparams)) { … n_reused++; }` skips `build_graph` + `sched_alloc_graph`
  (`:1520-1531`); only `res->set_inputs(&ubatch)` runs (`:1563`). Input-level reuse is shape-keyed
  (`can_reuse_kq_mask`, `llama-graph.cpp:49-66`; `n_outputs == params.n_outputs`, `:296-300`), so
  back-to-back rounds of equal draft length rebuild nothing host-side. **[source]**
- `graph_compute(gf, ubatch.n_tokens > 1)` only swaps the **CPU** threadpool (`n_threads_batch`), it
  switches no kernel (`:1570` + `graph_compute`). Cost the batch *adds*: the causal mask is refilled
  into a host `n_kv × n_tokens` tensor every ubatch (`llama-kv-cache.cpp:1761-1798`, F16 when FA is on
  per `llama-graph.cpp:40`), and FA always takes a mask when one exists (`gqa_opt_applies` requires
  `mask`, `fattn.cu:559`). **[source]**

## Q2. Attention

Kernel choice for this build (DKQ=DV=256, gqa 4, F16 KV) is `fattn.cu:540-717`; the AMD WMMA rule
(`:699-701`) is

    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && … &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16))

with `gqa_ratio_eff = 4` (`:667-669`). So **1-4 query rows → `BEST_FATTN_KERNEL_TILE`, ≥5 →
`BEST_FATTN_KERNEL_MMA_F16`** (`:717`, `:761-769`); VEC is unreachable because `gqa_opt_applies`
(`:704-716`). **[source]** Both chosen kernels put several query rows in one block, K/V streamed
once per block:
- TILE: `col_Q_0 = blockIdx.x * ncols1` (`fattn-tile.cuh:851`), KV loop per block
  (`:973` `for (int k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa)`);
  `use_gqa_opt` at `:1253` and `ncols2=4` for gqa 4 at `:1304-1305`; `cols_per_block` 4/8/16 for
  `ne[1]`=1/2/3-4 (`:1186-1224`) ⇒ **1, 2, 4 rows per block**.
- MMA: work = `(iter_k KV tiles)×(iter_j query tiles)×(iter_z_gqa head groups)`
  (`fattn-mma-f16.cuh:1890-1892`), `ncols1` from `fattn.cu:177-186` ⇒ 4 rows/block ≤4 rows,
  8 rows/block at 5-8. Q is loaded **once per tile** (`:1234` `Q_in_reg`, `:1250` `tile_Q`), K/V
  tiles are the inner loop, and `if (ncols1*ncols2 < 16 || ncols2 == 1 …) NO_DEVICE_CODE`
  (`:1861-1865`) keeps the tile wide. **[source]**

**Key answer: K/V DRAM traffic for 2..5 rows over a 4k cache is O(KV) once per (query tile, KV
head), not O(rows × KV)** — TILE covers 2 rows in one block at 2, MMA one block of 8 at 5. Causal
truncation is per *tile*: `flash_attn_mask_to_KV_max` (`fattn-common.cuh:671-723`) reduces the mask
to a per-(seq,tile) bound at `FATTN_KQ_STRIDE = 256` granularity (`:9`), consumed at
`fattn-mma-f16.cuh:1934` / `fattn-tile.cuh:954`. Addressing is slot arithmetic
(`K_h2 = K + nb13*sequence + nb12*z_KV`, `:1920`), no page table. **[source]** `-fa off` emits
`mul_mat`+`softmax`+`mul_mat` with an F32 mask instead — not this path. **[inference]**

## Q3. GDN / gated deltanet

18 recurrent layers via `llm_build_delta_net_base` (`qwen35.cpp:138,172,347-410`; state tensor
`mctx_cur->get_s_l(il)`, `build_rs(inp, ssm_states_all, …)` at `:399`). The form is chosen by
**token count in the ubatch** (`delta-net-base.cpp:422-444`):

    if (n_seq_tokens == 1) { if (cparams.fused_gdn_ar) return build_delta_net_fused(...); … }
    if (cparams.fused_gdn_ch) return build_delta_net_fused(...);

`ggml_gated_delta_net(ctx0, q,k,v,g,b,s, K)` is **one op over all `n_tokens` rows** (`:399`). The
MMA chunked kernel is *not* selected here: `select_gdn_mma_path` (`gated-delta-net-mma.cu:386-397`)
needs `n_tokens >= 64`, and under `GGML_USE_HIP` also `GGML_CUDA_CC_IS_RDNA3_5(cc) && n_tokens >=
2048`; else `gdn_path::ar` and the launcher returns false (`:426-428`). Verify batches therefore run
the **fused sequential kernel** `gated_delta_net_cuda` (`gated_delta_net.cu:10-180`), grid
`(H, n_seqs, ⌈cols/(4·warps)⌉)` (`:203-204`) — one block per head × column slice — with **the token
loop inside the block**:

    for (int t = 0; t < n_tokens; t++) { … }        // :75

state held in `float s_shard[rows_per_lane]` registers read once before the loop (`:65`) and
written once after it (`:172-179`). **Marginal cost of an extra verify row = one intra-block token
step, not a second state round-trip.** The chunked/MMA form (`build_delta_net_chunking`, `CS=64`,
`delta-net-base.cpp:61-85`; `// S = decay(chunk) * S + K^T * (solved * decay to chunk end)`,
`gated-delta-net-mma.cu:333`; "Each block owns V independent value columns and advances by C
tokens", `:123`) is a prefill route, not the verify route. State dtype **F32** (`:317`
`(const float *) src_state->data`), 1 MiB/layer/seq. Conv is one `ggml_ssm_conv` over the rolling
window for all rows (`qwen35.cpp:397-403`, `delta-net-base.cpp:446-490`). **[source]**

## Q4. Reject / rollback

`n_rs_seq` (recurrent snapshots/seq) = `draft.n_max` for `draft-mtp` (`common/common.h:401-407` →
`common/common.cpp:1716`; clamp `src/llama-context.cpp:191-195`; arch support `llama-arch.cpp:1254-1259`
lists `LLM_ARCH_QWEN35`; the **draft** ctx gets 0, `common/common.cpp:1261`). **[source]**
- **KV**: reject is `seq_rm(seq, p0, -1)` — bookkeeping, no bytes moved. **[source]**
- **Recurrent state**: the GDN kernel writes a snapshot per token of the verify batch *as it
  computes* (`keep_rs`, `gated_delta_net.cu:157-168`, `target_slot = n_tokens-1-t` at `:160`); one
  `ggml_cpy` moves the `n_written = min(n_tokens, K)` snapshots into the cache
  (`delta-net-base.cpp:543,561-600`, `K = n_rs_seq+1`), whose tensors are widened to
  `mem_size*(1+n_rs_seq)` rows (`llama-memory-recurrent.cpp:101`). **Rollback is then a scalar**
  (`:200-216`):

       const llama_pos rollback = cell.pos - (p0 - 1);
       if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
           set_rs_idx(seq_id, (uint32_t) rollback); cell.pos = p0 - 1; return true; }

  Guards: single-use, ≤ `n_rs_seq` (`:213`, `server-context.cpp:4553`); a cell shared by several
  seqs cannot move back for one (`:204-207`). **[source]**

Bytes/round, this model, n_max=k: snapshots written `(k+1)×18×1 MiB = 18(k+1) MiB`, copied again
into the cache ⇒ **~36(k+1) MiB written**, plus the `build_rs` view/gather (`qwen35.cpp:399`). Ours
is **19.27 MiB snapshot + 19.27 MiB shadow seed (+19.27 MiB restore on reject)**
(`engine/mtp_verify.h:30-35,130-134`). So llama's rollback is **not** byte-cheaper; it is
row-cheaper because the snapshots come out of the one GDN kernel instead of forcing one launch per
row. (Arithmetic from source; pricing is the main agent's.) **[source]**

## Q5. Draft cost

`common_speculative_impl_draft_mtp` (`common/speculative.cpp:1431`) runs qwen35 in the "neither"
mode — one trained head, own `ctx_dft`, `is_mem_shared=false` (`:1445-1451`, `:1522`). Per round,
**two** draft-side passes:
- `process()` (`:1580-1688`): a **catch-up decode of `ctx_dft` over all `n_tokens` rows**, each
  paired with the target's shifted hidden state `batch.set_embd_state(idx, {h_row,1,n_embd})`
  (`:1630-1637`), added with `output=false` so the head is not priced (`:1630`; out-id selection
  `llama-graph.cpp:277-295`); then the target h-rows are captured (`llama_get_embeddings_nextn_ith`,
  `:1676-1687`).
- `draft()` (`:1693-1840`): `while (n_drafting > 0)` → **one `llama_process(ctx_dft, DECODE, …)`
  per drafted token**, one row per seq, next position `dp.pos0 + i + 1` (`:1726`, `:1755`, `:1830`)
  — the serial shape. **[source]**

The h it conditions on: the target graph sets `res->t_h_nextn = cur` / `ggml_set_output(t_h_nextn)`
(`qwen35.cpp:212-213`, `llama-graph.cpp:1444-1445`) and the context downloads
`n_rows*n_embd*sizeof(float)` to host per call (`llama-context.cpp:2129-2141`, `:1731-1737`) =
20 KiB/round. The draft graph is the nextn block: `enorm`/`hnorm` → concat → `eh_proj [2*n_embd,
n_embd]` → one **dense attention** layer (nextn is flagged non-recurrent, `qwen35.cpp:15-16`) → MLP
→ **shared head** (`nextn.shared_head_head` else `model.output`, `qwen35.cpp:500-572,647-659`), i.e.
one full `n_embd×n_vocab` head walk per drafted token — where its ~0.3 step goes. **[inference from
graph contents; no timing]** Draft and target share no KV: `is_mem_shared` is
`llama_get_ctx_other(ctx_dft) == ctx_tgt`, false here (`:1522`). **[source]**

## Q6. Other row-cheap mechanisms present

| mechanism | exists | effect on verify cost |
|---|---|---|
| whole-graph reuse `can_reuse(gparams)` | yes `llama-context.cpp:1520-1531` | no host build/alloc per same-shape round |
| input reuse (`can_reuse_kq_mask`, out-id `n_outputs`) | yes `llama-graph.cpp:49-66,296-300` | skips re-deriving unchanged inputs |
| `inp_out_ids` head masking | yes `llama-graph.cpp:269-295` | no help for verify (all rows output=true); **big for the draft catch-up** (0 outputs) |
| `KV_max` causal truncation | yes `fattn-common.cuh:671` | drops whole 256-slot KV tiles per query tile |
| multi-token single-seq ubatch (`equal_seqs`) | yes | every per-launch cost paid once per round |
| `split_equal(..., n_keep_tail=n_rs_seq+1)` | yes `llama-memory-recurrent.cpp:459-461` | keeps a round in one ubatch so snapshots stay valid |
| `--parallel` / `n_seq>1`, `kv_keep` | parallel yes, irrelevant here (1 slot); **`kv_keep` is absent in this tree** | n/a |
| CUDA/HIP graphs | `GGML_HIP_GRAPHS=ON`, not traced | unknown |

## Q7. Portability verdict — the three mechanisms I believe matter

1. **Recurrence walked inside one launch, with the kernel writing per-row snapshots.** Ours:
   `engine/decode_step.cpp:928-944` — *"The checkpoint forces the span to 1 … At a span of 2 one
   `gdn_rec` launch walks both rows inside one kernel, so 'the state after row 0' exists in no
   register, buffer or address a copy could read"*. `--gdn-loop` exists
   (`engine/kernels_gdn.cpp:126-167`, `engine/gdn_rec_loop_kernels.h`) and is bit-identical, but
   `ckpt != nullptr` disables it. Porting = give `rec_loop_kernel`/`gdn_rec_collapse` llama's
   `keep_rs` tail (`gated_delta_net.cu:157-168`) into a K-slot state buffer and make rollback select
   a slot instead of restoring bytes (`llama-memory-recurrent.cpp:211-216`). **Obstacle:** our state
   is one live buffer per (layer, head) and `engine/mtp_verify.h:110-125` models rollback as
   full-size copies — snapshot planes are a state-geometry change, and reject must also un-append
   the row-1 lane (`engine/mtp_verify.cpp:445-458`).
2. **Several query rows per block over one streamed K/V pass.** Ours: the wide form is
   32-query-positions/block but for prefill, and `engine/attn_wide_llama_kernels.h`'s own header
   lists what it did **not** take (no padded stage stride, no double-buffered K, no mask tile,
   epilogue carried not shared); `engine/attn_fattn_kernels.h` holds the fragment maps. The missing
   llama property is not the swizzle but per-tile `KV_max` truncation plus wide tiles on the
   **one-to-few-row** path (`fattn-tile.cuh:1186-1224`). **Obstacle:** our prep appends per-row lanes
   unconditionally and the next round anchors on them (`engine/mtp_verify.cpp:445-458`), so a 2-row
   body's lanes are cache positions, not a query tile, and the causal bound must become per-row
   instead of one `j < lanes_of(...)` scalar.
3. **Round-level costs paid once** (graph reuse, one ubatch, one mask fill, one state gather, one
   sync) **and rollback as an index**. **Obstacle:** our step is a hand-built launch plan with no
   graph object and no input-reuse concept (`engine/kernels_attn.cpp`, `engine/launch_stats.h`).

Not mechanisms: llama's rollback is not byte-free (≈36(k+1) MiB/round, §Q4), and `gdn_single`'s
chunked MMA path is off for verify-sized batches on this GPU. **[source]**

## What I could NOT determine from source

- Nothing here is priced; all byte figures are arithmetic, no measurement was taken.
- Whether `gqa_opt_applies` holds at run time (`K->ne[1] % 256 == 0`, all strides %16 — KV padding
  for `-c 5120` not traced). If it fails, `ncols2=1` and the WMMA fattn compiles out
  (`fattn-mma-f16.cuh:1861`), landing on TILE/VEC.
- Whether HIP graph capture is enabled on this path, and how it interacts with draft-mtp.
- Whether `dur(b,g,a)`'s draft timer includes the `process()` catch-up pass — it wraps
  `impl->draft()` only (`common/speculative.cpp:2945`), so the printed draft time may *exclude* it.
- `llama_memory_hybrid` (kv+recurrent) was read only via `llama-memory-recurrent.cpp`; the wrapper's
  `seq_rm`/`seq_cp` ordering was not verified end to end.
- ggml's `mmvq`/`mmvf` m>1 paths — out of scope per the brief (GEMM axis excluded).

## Side by side with what this repo measured

Every llama.cpp figure is from `prototype/mtp_llama/evidence/` (same box, same checkpoint, greedy,
medians); every engine figure from `prototype/mtp_accept/evidence/`. Nothing in this table is
cross-window inside a pair: the ratios are quoted from within single lock holds.

| axis | llama.cpp (measured) | our engine (measured) | which mechanism explains it |
|---|---|---|---|
| cost per drafted token | **+0.30 step** (serial by construction) | **+0.04 step** (≈0.44 ms/round) | Q5 — theirs re-decodes the draft context once per drafted token *after* a catch-up pass over all rows, through a dense nextn block that ends in a full `n_embd×n_vocab` head walk. **We are ~8× better here.** |
| cost per extra VERIFY row at 4k | **+0.078 step** | **+0.96 step** | Q3 (the recurrence is walked inside one launch) and Q2 (one block streams K/V once for several query rows). This is the whole gap. |
| reject/rollback | index flip (`set_rs_idx`), at ≈36(k+1) MiB of snapshot writes | 19.27 MiB copied out and back | Q4 — theirs is **row-cheap and byte-expensive**; ours is the mirror image. Neither is free. |
| round at 4k, k=1 | 1.579 steps → **+9.5 %** tok/s | 1.964 amortised / 2.133 hand → **−16 % to −22 %** | sum of the two rows above |
| best over k | k=2: **+13.6 %** (k=4 already decays to +7.0 %) | k=1 only, and k=2 at 4k would be *worse* than k=1 | accept ladders: pos1 0.750–0.826 (theirs) vs 0.6556 (ours) — same order, so **the accept rate was never the difference** |
| whole-round overheads | graph reuse, one ubatch, one mask fill, one sync | hand-built launch plan per body + shadow copy + verdict + readback + lane revert | Q6 / Q7.3 |

**Reconciling a claim made earlier today.** I reported, from `row_scale.sh`, that "our engine amortises
rows better than llama.cpp". That is true only of the families that instrument reached — the tied
`lm_head` really is flat in rows (2.14 ms for 1 row or 64), and the tiled projections really do amortise.
It is false of the two families that dominate a long-context round: our GDN recurrence is one launch per
row and our attention re-scans K/V per row at 4k. Those are precisely the two things llama.cpp made
row-independent (Q2, Q3), which is why the correct statement is not "we amortise better" but **"we
amortise the cheap families and pay the expensive ones per row."**
