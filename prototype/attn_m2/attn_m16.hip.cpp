/*
 * attn_m16.hip.cpp — D58's deciding measurement: the shipped `attn_slim_wave` spell re-tiled to
 * SIXTEEN query rows per block, scored with the SHIPPED VALU dot and the SHIPPED warp fold. There is
 * no WMMA instruction in this file, on purpose: what is under test is the GEOMETRY TAX of the M = 16
 * re-tile, not the matrix win (`prototype/attn_wmma/` prices the matrix leg).
 *
 * THE QUESTION, AND THE LINE IT IS MEASURED AGAINST
 * `progress/2026-10-16/prefill-decision-log-2026-10-16.md` D58: replacing the shipped pass-1 leg with
 * a WMMA score leg removes ~48 % of `attn_slim_wave`'s measured 89.9 warp-inst/touch, which at this
 * kernel's own measured slot->time elasticity (`slots^0.39-0.44`, D21/D22) is -24 % of `attn` before
 * tax, ~13 % of `attn` wall against D30's kill line of ">15 % of `attn`" — on the line, not above it,
 * and with NO geometry tax in it. `prototype/attn_m2` owns the only measured tax in this repo: +7 % at
 * M = 2 on device (results.md §6), +15-29 % modelled at the funded ruler (§8.4). A tax near +7 % funds
 * item 2.6; a tax near +29 % closes it. This file produces the M = 16 number, on device, in
 * `prototype/attn_m2`'s own harness shape and plan set.
 *
 * ---------------------------------------------------------------------------------------------
 * THE GEOMETRY FACT THAT SIZES THE WHOLE DESIGN: what one 16-row block can hold SIMULTANEOUSLY
 * ---------------------------------------------------------------------------------------------
 * 16 query rows x 4 GQA heads = 64 (row, head) pairs, and every pair carries its own §05.5/§07.3
 * softmax state: `acc[8]` per lane (its 8 dims of a 256-wide head) plus (m, l). All 64 pairs live at
 * once is 64 x 256 f32 = 16,384 f32 = 64 KiB — 128 f32 per lane of a 128-thread block, or the entire
 * LDS of this CU. That is D58's own PV-half figure ("16 f32 accumulator fragments over DH = 256 =
 * 128 VGPR/lane against today's 47"), and it is a property of the VALU spell, not of WMMA: an MMA
 * holds a 16x16 D tile in 8 f32/lane, which is exactly why the M = 16 geometry is resident for WMMA
 * (`m16_bf16gq`, 5 blocks / 20 wavefronts per CU, driver-confirmed in D58) and is not resident for
 * VALU at 64 live pairs. `wmma_m16_probe.cpp`'s `m16_scalar` control never met this because it has no
 * softmax — it keeps one row of q live at a time (25 VGPR); the same loop with every lane's fold
 * result made observable is 152 VGPR and 1 block/CU (`m16_census.output.txt` §1).
 *
 * So the block OWNS 16 rows — grid.z = ceil(nrows/16), one block per 16-row tile, appending, scoring,
 * gating and writing partials for all 16 — and it shares ONE K/V walk with the largest row group the
 * register file can hold live:
 *
 *     live pairs = ATTN_M16_GRP (4 rows) x ATTN_M16_G (4 heads) = 16 pairs = 4 acc[8] per warp
 *
 * the same live-pair count `attn_m2` carries (2 rows x 4 heads at 8 warps). Derived costs, each
 * checked below rather than asserted:
 *   * TRAFFIC: one staged K row and one staged V row feed 16 (row, head) pairs, so bytes demanded is
 *     ~0.25 of the shipped form against m2's measured 0.50. `bytesdemanded_m16` on the RESULT line is
 *     computed from the rounds this kernel actually stages, not from a hoped-for factor.
 *   * REGISTERS: 4 live rows x (acc[8] + m + l) = 40 f32 + q[8] + e[TILE] + addresses. The compiler's
 *     own remark is the number to quote, and the driver query is the honest one (AGENTS.md: clang's
 *     `Occupancy [waves/SIMD]` remark is a constant; D58: a metadata-derived occupancy was 3x wrong on
 *     one m16 variant).
 *   * LDS: `s_sc` is per (row, head) over a ROUND WINDOW, not over a whole slice, and K and V share one
 *     staging — legal here because a round is scored completely before V is staged, so no K row is
 *     live when V lands. `prototype/attn_m2/results.md` §7.1 lists that alias as an open lever; this
 *     form takes it because without it the parity arithmetic fails (two stagings instead of one).
 *   * GLOBAL-A, NOT STAGED Q: the query vector is read from `a.qrope` per (row, head) per round — the
 *     `m16_bf16gq` shape — because D58's driver put that spelling at shipped parity (5 blocks / 20
 *     wavefronts per CU) while staging Q costs blocks (`m16_roll`: 3 / 12), and a 16-row Q stage is
 *     32 KiB, half the LDS, in a block whose LDS is what binds. The price, named: 512 B of q re-read
 *     per row per round, ~64 B per touch against ~16 B of K+V per touch, served from L1/L2 out of a
 *     32 KiB per-block working set. This gate cannot answer whether that is free; `burst_m16.sh`
 *     header names it as the open question, and it is the same trade `m16_bf16gq` made at 16 K-steps
 *     per tile.
 *
 * ---------------------------------------------------------------------------------------------
 * WHY THIS FORM IS A STREAMING-ROUTE FORM, AND REFUSES A ONE-PASS ROW (a refusal, not a gap)
 * ---------------------------------------------------------------------------------------------
 * §05.5's one-pass form (`s_row == 1 && a.full`, i.e. a row of at most MAXSC = 512 lanes) reads its
 * scores as a WHOLE row: max over the row, then `l = sum_j exp(z_j - m)` serial in j, then P.V serial
 * in j with the weight divided exactly once. So a one-pass pair needs its whole slice resident: the
 * shipped block affords that (`s_sc[G][MAXSC]` = 4 KiB) and m2 could at M = 2 (8,320 B). At 16 live
 * pairs the same staging is 16 x 512 x 2 = 16,384 B for ONE head and four times that for the group's
 * heads = the entire LDS. `fmaxf` is exactly associative so the MAX could be rolled across rounds; the
 * sum-of-exp cannot, because it needs the final `m` before its first term; a score spill to global
 * would demand ~64 KiB per block, more than the K+V it saves; and recomputing the dot in pass 2
 * doubles the very leg whose tax is being measured. So one-pass rows are REFUSED at launch
 * (`attn_m16_ok`), the kernel poisons instead of silently streaming them, and `--geom-guard` asserts
 * the refusal agrees with the geometry in both directions (no phantom, no leak — `prototype/attn_m2`
 * §6, where a wrong aggregate in a refusal predicate silently blocked four funded plans).
 *
 * What that costs in reach, stated plainly because it changes how the number must be read:
 *   * a row streams iff `!full || lanes_of(a,row) > MAXSC`. Every width this route is funded for
 *     (1024 / 2048 / 4096 lanes) streams at every split the launcher picks, so the funded ruler is
 *     inside the form.
 *   * a wave at <= MAXSC lanes under the shipping `full = 1` does NOT, and neither does a wave that
 *     straddles the boundary (rows 512..515 mix a one-pass row with streaming ones — the geometry
 *     `prototype/attn_m2` exists to prove). A 16-row tile at the boundary contains a one-pass row, so
 *     it is refused too. `--full 0` forces every row to stream and makes the short widths measurable,
 *     but then BOTH sides of the A/B take the streaming route, so a short-width number taken that way
 *     is a streaming-route number, not a shipping-path number. `burst_m16.sh` says which plans run
 *     under which `--full`.
 *   * the widths `prototype/attn_m2` measured its +7 % on (§4: lanes 1-8 and 511..520) are one-pass or
 *     straddling. Read that as a finding, not a gap: row-tiling at this depth and the short-row route
 *     are disjoint, and a shipping rule for this form would have to be a lane-count gate of the kind
 *     `prototype/attn_m2/results.md` §5.3 already proposes.
 *
 * ---------------------------------------------------------------------------------------------
 * WHAT MAKES BIT-IDENTITY POSSIBLE: every ordered reduction stays per PAIR, in the shipped order
 * ---------------------------------------------------------------------------------------------
 * §05.5 pins three ordered reductions per (row, head): the score dot (8 serial MACs, then the 5-step
 * `__shfl_down_sync` fold, then ONE BF16 round, then x SCALE); `l` serial in j with `m` the row max;
 * and `a = sum_j w_j . v_j` serial in j. §07.3 adds that for a streaming row the tile grouping and the
 * block boundaries are part of the number, because `expf(m - mn)` does not associate. None of the three
 * spans more than one row, so a block may share the WALK and still run private reductions — m2's
 * argument, and it scales to 16 rows unchanged. What has to be respected:
 *   1. BARRIERS ARE BLOCK-UNIFORM. Every quantity bounding a loop that contains a `__syncthreads()` —
 *      the round union `[u_lo, u_hi)`, `kmax` — is a min/max over the rows of the group, a per-BLOCK
 *      value. A row that is inactive, or that has fewer chunks than `kmax`, is skipped by a predicate
 *      INSIDE the loop body, never by skipping a barrier. A barrier executed by some threads of a
 *      block is a queue abort on this part (code 0x1016), not a wrong count — the `diff_rows` comment
 *      in `attn_m2.hip.cpp` records exactly that fault.
 *   2. EACH ROW'S CHUNK IS ANCHORED AT ITS OWN `blo`, one for one with `attn_core`'s and the shipped
 *      tile form's tile k, so this form's chunk k is bit-for-bit the shipped form's chunk k. The staged
 *      union of a round is therefore a tile plus the skew between the group's row spans;
 *      `m16_census` computes it from the same `m16_row_at`/`m16_chunk_of` the kernel runs and
 *      `--geom-guard` walks it; `s_sc`/`s_kv` are sized by the walked worst case, and an over-run
 *      poisons rather than corrupting a neighbour (AGENTS.md, §13/§14: an LDS over-run does not fault
 *      on this part, it silently moves the neighbour's 1,098,790 elements).
 *   3. THE DOT AND THE FOLD ARE CALLED, NOT TRANSCRIBED. `vslim_dot<8|4>`,
 *      `vslim_fold<VSLIM_DOT | VSLIM_FOLD>` and `vslim_permute_addr` come from
 *      `../attn_valu_slim/attn_valu_slim_kernels.h` — the same objects the shipped `attn_slim1_wave`
 *      compiles — and `publish` is the shipped expression `fb(bf(fb(dot) * SCALE))`. Bit-identity is
 *      owed to the SOURCE TEXT's association, not the algebra: `prototype/gdn_collapse/results.md` §2
 *      is the witness (a hoisted scalar constant moved 93,522 of 131,072 `o` elements on device while
 *      leaving the state bit-exact, and a CPU mirror cannot see it, because a mirror reproduces the
 *      formula and two sides that share a formula agree). A re-tile whose dot was re-typed would have
 *      no witness, so there is no second spelling: `check_m16_transcription.py` fails if a local
 *      `vslim`/`slim_dot`/`warp_sum` definition appears in this file.
 *   4. THE SELF LANE'S SOURCE IS PER ROW AND IS NOT SHARED. Lane `pos + row` is this wave's §05.7
 *      append, and both shipped wave forms read it from the STAGE (`krope`/`vg`) rather than the cache
 *      (m2 constraint 4). Rows of different groups land in different rounds, so no per-round slot can
 *      be "the self lane" for a whole group: every live row of a group gets its own LDS row
 *      (`s_sk`/`s_sv`, indexed by the group slot), and every staged slot comes from the CACHE — which
 *      is what keeps `slotk`/`slotv` visible through the read side instead of laundered by a stage.
 *   5. THE PARTIAL SLOT IS THE SHIPPED EXPRESSION, keyed by the row and by the FULL head:
 *      `((row*N_Q + head)*split + y) * (DH + 2)`, and `attn_combine` stays one function that never
 *      branches on which core ran (§19/§20.1 of `prototype/attn_decode/results.md` is the bill for two
 *      self-consistent layouts joined by a branch; a 16-row block adds two fresh ways to alias).
 *   6. AN EMPTY ROW-TILE OR GROUP RETURNS BEFORE ANY BARRIER, and writes nothing. grid.z and the group
 *      count are exact, so no gate would ever have reached that route — `--pad-tiles` launches it on
 *      purpose and the gate requires the padded run to be byte-identical to the unpadded one
 *      (AGENTS.md: "a route no fixture reaches has no tests, including tests of the gate").
 *
 * WHAT THIS FORM MAY NOT BE ASKED FOR (`attn_m16_ok`, one place so launcher and gate refuse alike):
 * `attn_tile_ok`'s list (no `score_f32`: the score staging is BF16 and exact only for the pinned
 * `bf(fb(dot)) * SCALE`; no `fused`: a wave's §05.7 append must land before any row reads the cache;
 * warp 32, so one warp covers a 256-wide head as 32 lanes x 8 dims) plus three of its own: no one-pass
 * row in the launch, a `split` at least `split_for_lanes(lanes_last)`, and no round union past `M16_L`.
 *
 * ADMISSION: bit-identity, `ndiff = 0`, no band. A pure re-tile with identical arithmetic has nowhere
 * to hide (`prototype/gdn_rec_isa/`'s admission is the shape copied: one process, several geometries,
 * a per-case `ndiff` against the reference form, no timing if any case differs, and a case COUNT that
 * must equal the population so an empty parse cannot print PASS).
 *
 * NO TIMING IS TAKEN OR REPORTED BY A WORKER HERE (AGENTS.md). The `--burst` path below exists for the
 * main agent; the correctness path prints counts, geometry and a bytes-demanded census only, and no
 * millisecond figure appears in this file or in the appended `results.md` section that is not already
 * in `bench/`. THIS BINARY DOES NOT SELF-LOCK: the caller holds `tools/gpu_lock.sh` around it.
 *
 * Build:  prototype/attn_m2/build_m16.sh          (compile-only; no device is touched)
 * Gates:  ./prototype/attn_m2/attn_m16 --geom-guard             (host-only, no device, no lock)
 *         .venv/bin/python prototype/attn_m2/check_m16_transcription.py
 *         .venv/bin/python prototype/attn_m2/gate_m16.py plans  (writes the span plans, no device)
 *         .venv/bin/python prototype/attn_m2/gate_m16.py ab     (device, main agent)
 * Burst:  prototype/attn_m2/burst_m16.sh          (main agent, inside its own lock hold)
 */
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// INCLUDED, not copied, for the reason in constraint 3 of the header: a transcription of the dot would
// be a second spelling of one concept, and no per-form check can tell two self-consistent spellings
// apart (§20.1 of prototype/attn_decode/results.md). This include also brings in
// `../attn_decode/attn_decode_kernels.h`, so `attn_tile_wave`, `attn_slim1_wave`, `attn_prep` and
// `attn_combine` are the shipped objects themselves and this file adds exactly one wave form.
#include "../attn_valu_slim/attn_valu_slim_kernels.h"

#define CK(x)                                                                \
    do {                                                                     \
        hipError_t e_ = (x);                                                 \
        if (e_ != hipSuccess) {                                              \
            fprintf(stderr, "HIP %s:%d %s -> %s\n", __FILE__, __LINE__, #x,  \
                    hipGetErrorString(e_));                                  \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---- geometry constants -------------------------------------------------------------------
#define ATTN_M16_M 16              // query rows ONE BLOCK owns. The whole point of the item.
#ifndef ATTN_M16_GRP                // overridable with -DATTN_M16_GRP=2 (or 8) so the register/LDS
#define ATTN_M16_GRP 4             // coupling can be measured rather than asserted: 2 gives the shipped
#endif                             // 8 live pairs and the shipped register profile at half the K reuse,
                                   // 4 is the shape this file ships, 8 is 32 live accumulators.
                                   // rows that share one K/V walk: 4 x GQA = 16 live pairs. See the
                                   // header: 16 live rows would be 128 f32/lane of accumulator.
#define ATTN_M16_G GQA             // 4 query heads of one KV group per block, exactly as shipped
#define ATTN_M16_THREADS (ATTN_M16_G * 32)   // 128 threads = G warps, one warp per head, as shipped
#define ATTN_M16_MAXSPLIT 8        // this prototype's own cap (m2's ATTN_M2_MAXSPLIT; the engine's
                                   // kAttnMaxSplit is 9 and is not reachable from this harness)
// The staged union of a round is TILE lanes plus the skew between the spans of GRP adjacent rows.
// The value is WALKED, not asserted: `--geom-guard` walks every legal geometry and fails if any round
// it would stage exceeds M16_L. The bound is small because span_i = ceil(n_i / s_row) and the rows of a
// group differ by one lane each, so their spans differ by at most ceil((GRP-1)/s_row) + 1 and the skew
// is that spread times `y <= s_row - 1`.
#define M16_SKEW 8
#define M16_L (TILE + M16_SKEW)    // lanes of K, then V, resident per round (K and V ALIASED)
#define M16_PAIRS (ATTN_M16_GRP * ATTN_M16_G)

constexpr int kM16ScBytes = M16_PAIRS * M16_L * 2;         // BF16 score window per live pair
constexpr int kM16KvBytes = M16_L * DH * 2;                // one staging, K then V
constexpr int kM16SelfBytes = 2 * ATTN_M16_GRP * DH * 2;   // this wave's own lane, per live row
constexpr int kM16LdsBytes = kM16ScBytes + kM16KvBytes + kM16SelfBytes;

static_assert(ATTN_M16_M % ATTN_M16_GRP == 0,
              "a 16-row tile is walked in groups of GRP rows; 16 % 4 must be 0 or the tail group of "
              "every tile needs its own geometry");
static_assert(M16_L >= TILE, "a round must hold one row's whole chunk, or a chunk would straddle two "
                             "staged buffers and the per-chunk rescale would read scores that were not "
                             "staged with it");
static_assert(ATTN_M16_THREADS / 32 == ATTN_M16_G, "one warp per query head of the group, as shipped");
static_assert(kM16LdsBytes <= 65536, "a block's own LDS must fit one CU (KFD: lds_size_in_kb 64)");
static_assert(M16_PAIRS * 8 <= 160, "the live accumulator budget: live pairs x 8 f32/lane. At 32 f32 "
                                    "this block already carries 4 live rows' acc state against the "
                                    "shipped block's one; raising GRP is the change that turns the tax "
                                    "into an occupancy story (the 128 f32/lane wall in the header).");

// The two shipped row rules, transcribed character for character from
// `attn_decode_kernels.h::lanes_of` / `::blocks_for_row`, injections included, so a shared injection
// moves this form exactly where it moves the shipped one. Spelled rather than called because the
// shipped pair are `__device__`-only in a frozen header while this file's launcher refusal and
// `--geom-guard` must run on the host — the same reason and the same precedent as
// `attn_m2_kernels.h`'s `m2_lanes_of`/`m2_blocks_for_row`.
__device__ __host__ inline int m16_lanes_of(const CoreArgs& a, int qrow) {
    if (a.neg == NEG_READPAST) return a.cap;
    if (a.neg == NEG_DROPCUR) return a.pos + qrow;
    return a.lvalid + qrow;
}
__device__ __host__ inline int m16_blocks_for_row(const CoreArgs& a, int qrow) {
    if (a.full && m16_lanes_of(a, qrow) <= MAXSC) return 1;
    return a.split > 0 ? a.split : 1;
}
// The one-pass condition of `attn_tile_body`, stated as a predicate so the refusal and the kernel
// branch on ONE spelling of it.
__device__ __host__ inline bool m16_row_onepass(const CoreArgs& a, int qrow) {
    return a.full && m16_lanes_of(a, qrow) <= MAXSC;
}

// Grid shapes and footprint, shared by the kernel, the launcher and the guard.
__device__ __host__ inline int attn_m16_tiles_of(int nrows) {
    return (nrows + ATTN_M16_M - 1) / ATTN_M16_M;
}
__device__ __host__ inline int attn_m16_groups_of(int nrows) {
    return (nrows + ATTN_M16_GRP - 1) / ATTN_M16_GRP;
}
__device__ __host__ inline int attn_m16_split(int lanes_last) { return split_for_lanes(lanes_last); }
// The partial footprint is the SHIPPED layout, slot for slot: [row][head][lane block][2 + DH] F32.
__device__ __host__ inline int attn_m16_part_floats(int nrows, int lanes_last) {
    return nrows * N_Q * split_for_lanes(lanes_last) * (DH + 2);
}

// One row's geometry in lane block `y`. Values, not array slots indexed at run time: `attn_m2_kernels.h`
// measured what a struct array indexed by a rolled loop variable costs (96 B/lane of private scratch in
// a kernel whose whole point is bytes), so every loop over ATTN_M16_GRP in this kernel is
// `#pragma unroll`, which is what keeps `M16Row g[ATTN_M16_GRP]` in registers. ScratchSize in the
// compiler's own remark is the check, and the appended `results.md` section records it.
struct M16Row {
    int row;      // global query row index
    int pos;      // a.pos + row: that row's §05.7 self lane
    int n;        // lanes it may attend (§06.3's causal prefix for THAT row)
    int s_row;    // lane blocks that row promises (the shipped rule)
    int span;     // lanes per lane block for that row (the shipped rule)
    int blo, bhi; // this lane block's slice of that row
    int chunks;   // work units (tiles) this row runs in this lane block
    bool exists;  // row < nrows
    bool active;  // exists, this lane block is within its s_row, and the slice is non-empty
};

__device__ __host__ inline M16Row m16_row_at(const CoreArgs& a, int row, int y, int tile) {
    M16Row g{};
    g.row = row;
    g.exists = row < nrows_of(a);
    g.pos = a.pos + row;
    g.n = g.exists ? m16_lanes_of(a, row) : 0;
    g.s_row = g.exists ? m16_blocks_for_row(a, row) : 0;
    g.span = g.s_row > 0 ? (g.n + g.s_row - 1) / g.s_row : 0;
    const int blo = g.span * y;
    // `std::min`, spelled as the shipped form spells it, because `check_m16_transcription.py` compares
    // this line against `const int bhi = std::min(nrow, blo + span);` and a hand-rolled ternary "equal"
    // to a min is exactly the kind of re-spelling that check exists to catch.
    const int hi = g.span > 0 ? std::min(g.n, blo + g.span) : 0;
    g.blo = blo;
    g.bhi = hi;
    g.active = g.exists && g.s_row > 0 && y < g.s_row && hi > blo;
    g.chunks = g.active ? (hi - blo + tile - 1) / tile : 0;
    return g;
}

// Row i's chunk k, anchored at its OWN blo: for a streaming row this is tile k of `attn_core`'s loop,
// one for one, which is what makes the per-chunk `(max,sum)` rescale land on the shipped bits (§07.3).
__device__ __host__ inline void m16_chunk_of(const M16Row& g, int k, int tile, int* lo, int* hi) {
    if (!g.active || k >= g.chunks) { *lo = 0; *hi = 0; return; }
    const int s0 = g.blo + k * tile;
    *lo = s0;
    *hi = (g.bhi < s0 + tile) ? g.bhi : s0 + tile;
}

// What one LAUNCH would stage, walked with the same functions the kernel runs: the widest round union
// (what M16_L bounds), the widest row slice, the block count, the lane rows staged in total, and
// whether any row of the launch is one-pass (which this form refuses). ONE struct, ONE walk, used by
// the refusal predicate, the guard and the RESULT line's bytes census, so a claim about traffic and a
// claim about staging cannot come from two different geometries (AGENTS.md: comparing a `total` summed
// over blocks against a per-block bound is `prototype/attn_m2/results.md` §2's bug).
struct M16Census {
    int max_round = 0;      // widest staged round, lanes: what M16_L bounds
    int max_slice = 0;      // widest single row slice, lanes
    int round_lanes = 0;    // lane rows staged, summed over every block and round: TRAFFIC DEMANDED
    int blocks = 0;         // (kv group, lane block, row tile) blocks of the launch
    int tiles = 0, groups = 0;
    int onepass_rows = 0;   // rows this form may not serve
    int empty_groups = 0;   // groups with no active row (the grid-overhang route)
    bool over = false;      // a round wider than M16_L
};

inline M16Census m16_census(int lvalid, int pos, int nrows, int split, int full, int tile,
                            int pad_tiles = 0) {
    M16Census C;
    if (nrows <= 0 || split <= 0) return C;
    CoreArgs a{};
    a.lvalid = lvalid; a.pos = pos; a.nrows = nrows; a.split = split; a.full = full;
    a.neg = NEG_NONE; a.tile = tile; a.cap = 1 << 20;
    C.tiles = attn_m16_tiles_of(nrows);
    C.groups = attn_m16_groups_of(ATTN_M16_M);      // groups inside ONE tile, which is what loops
    for (int row = 0; row < nrows; ++row)
        if (m16_row_onepass(a, row)) ++C.onepass_rows;
    M16Row g[ATTN_M16_GRP];
    for (int y = 0; y < split; ++y) {
        for (int t = 0; t < C.tiles + pad_tiles; ++t) {
            for (int gi = 0; gi < C.groups; ++gi) {
                int kmax = 0, active = 0;
                for (int i = 0; i < ATTN_M16_GRP; ++i) {
                    g[i] = m16_row_at(a, ATTN_M16_M * t + ATTN_M16_GRP * gi + i, y, tile);
                    if (g[i].active) {
                        ++active;
                        if (g[i].chunks > kmax) kmax = g[i].chunks;
                        if (g[i].bhi - g[i].blo > C.max_slice) C.max_slice = g[i].bhi - g[i].blo;
                    }
                }
                ++C.blocks;
                if (active == 0) { ++C.empty_groups; continue; }
                for (int k = 0; k < kmax; ++k) {
                    int lo = 0x7fffffff, hi = 0;
                    for (int i = 0; i < ATTN_M16_GRP; ++i) {
                        int cl, ch;
                        m16_chunk_of(g[i], k, tile, &cl, &ch);
                        if (ch > cl) { if (cl < lo) lo = cl; if (ch > hi) hi = ch; }
                    }
                    const int un = hi > lo ? hi - lo : 0;
                    C.round_lanes += un;
                    if (un > C.max_round) C.max_round = un;
                    if (un > M16_L) C.over = true;
                }
            }
        }
    }
    C.blocks *= N_KV;      // one block per KV group, as every wave form launches
    return C;
}

// What this form may not be asked for, in one place so the launcher and the gate refuse alike.
inline bool attn_m16_ok(int score_f32, int fused, int warp_size, int split, int lvalid, int lanes_last,
                        int nrows, int full, const char** why) {
    if (warp_size != 32) { *why = "one warp covers a 256-wide head as 32 lanes x 8 dims"; return false; }
    if (score_f32) { *why = "the score staging is BF16, exact only for the pinned BF16 score"; return false; }
    if (fused) { *why = "a wave's §05.7 append must finish before any row reads the cache"; return false; }
    if (split < attn_m16_split(lanes_last)) {
        *why = "a row's slice would exceed one lane block of the shipped cut (raise --split)";
        return false;
    }
    CoreArgs a{};
    a.lvalid = lvalid; a.split = split; a.full = full; a.neg = NEG_NONE;
    a.pos = lvalid - 1; a.nrows = nrows > 0 ? nrows : 1;
    // Row 0 has the fewest lanes of the wave (`lanes = lvalid + row`), so a monotone test on it IS the
    // whole predicate: if row 0 streams, every row streams.
    if (m16_row_onepass(a, 0)) {
        *why = "a 16-row tile cannot hold a one-pass row's whole-slice score staging (16 x 512 x 2 B "
               "per head = the entire LDS); this form streams, so it refuses rows at or below MAXSC";
        return false;
    }
    const M16Census C = m16_census(lvalid, lvalid - 1, nrows, split, full, TILE);
    if (C.over) { *why = "a staged round would be wider than M16_L"; return false; }
    return true;
}

// The geometries this op actually runs, kept next to the guard that reads them so a plan that starts
// being refused fails the guard in seconds instead of costing a device window (m2's `M2_REAL`, same
// purpose). `streams` says which side of the one-pass line the wave is on; the ones marked false are
// RUN UNDER `--full 0` by `gate_m16.py`, the only honest way to put them on this form's route.
struct M16RealCase { const char* name; int lvalid; int nrows; int split; int full; bool streams; };

inline const M16RealCase* attn_m16_real_cases(int* n_out) {
    static const M16RealCase REAL[] = {
        // The span set the gate is built on: 0 / 1 / 3 / 16 rows in a 16-row tile.
        {"span01-1024",    1020,  1, 2, 1, true},
        {"span03-1024",    1020,  3, 2, 1, true},
        {"span16-1024",    1020, 16, 3, 1, true},
        {"span16-4096",    4080, 16, 8, 1, true},
        {"span01-4096",    4095,  1, 8, 1, true},
        // The waves `prototype/attn_m2` timed, at the same geometries, for a same-harness tax.
        {"wave64-at-1024", 1020, 64, 3, 1, true},
        {"wave64-at-2048", 2044, 64, 5, 1, true},
        {"wave64-at-4096", 4032, 64, 8, 1, true},
        {"wave16-at-4096", 4080, 16, 8, 1, true},
        {"pair-just-above",  512,  8, 2, 0, false},   // lanes_last = 519, but ROW 0 sits at exactly
                                                      // MAXSC, so `full = 1` still refuses this case
        {"pair-widest-1024", 1020, 4, 2, 1, true},
        {"straddles-block-2", 1023, 4, 3, 1, true},
        {"tail-hf-1032",   1032,  1, 3, 1, true},
        // One-pass under the shipping `full = 1`, admissible only when every row is forced to stream.
        // The 0-row span is not in this table: it is a padded grid (`--pad-tiles 1`), which the guard
        // checks the empty-group arithmetic of for EVERY geometry below.
        {"pair-at-maxsc",   511, 4, 2, 0, false},
        {"maxsc-minus-1",   510, 4, 2, 0, false},
        {"pair-shortest",     1, 8, 1, 0, false},
        {"short-64",         64, 64, 1, 0, false},
        {"short-256",       256, 64, 1, 0, false},
    };
    *n_out = (int)(sizeof(REAL) / sizeof(REAL[0]));
    return REAL;
}

// Host-only walk: no HIP call, no device, no lock. The four assertions of `prototype/attn_m2` §6 plus a
// fifth on the grid arithmetic, because "a route no fixture reaches has no tests" is a trap this repo
// has been bitten by repeatedly (an empty group, and a padded row-tile, are exactly those routes).
//   1. BOUND      — no legal launch stages a round wider than M16_L, as the kernel's own geometry
//                   computes it (not as a re-typed closed form hopes);
//   2. NO PHANTOM — the launcher refuses no launch this form could run;
//   3. NO LEAK    — the launcher accepts no launch the kernel would have to poison;
//   4. REAL       — every geometry `gate_m16.py` / `burst_m16.sh` emits is accepted, and every one
//                   listed as one-pass really is refused under the shipping `full = 1`;
//   5. COVER      — tiles x groups cover `nrows` exactly: no empty row-tile exists unless a caller
//                   padded it on purpose, and an empty GROUP only exists when nrows is not a
//                   multiple of GRP.
inline int attn_m16_guard(int verbose) {
    int fails = 0, checked = 0, refused = 0, phantom = 0, leaked = 0;
    int worst_round = 0, worst_slice = 0, worst_lanes = 0, worst_split = 0;

    int lvals[256];
    int nl = 0;
    auto push = [&](int n) {
        for (int i = 0; i < nl; ++i) if (lvals[i] == n) return;
        if (n >= 1 && nl < 256) lvals[nl++] = n;
    };
    for (int n = 1; n <= 64; ++n) push(n);
    for (int n = MAXSC - 6; n <= MAXSC + 10; ++n) push(n);
    for (int k = 1; k <= ATTN_M16_MAXSPLIT; ++k)
        for (int d = -3; d <= 3; ++d) push(MAXSC * k + d);
    for (int n = 65; n <= MAXSC * ATTN_M16_MAXSPLIT; n += 127) push(n);
    push(MAXSC * ATTN_M16_MAXSPLIT);
    static const int NROWS[] = {1, 2, 3, 4, 5, 15, 16, 17, 32, 64};

    for (int li = 0; li < nl; ++li) {
        for (int split = 1; split <= ATTN_M16_MAXSPLIT; ++split) {
            for (int full = 0; full <= 1; ++full) {
                for (int ni = 0; ni < (int)(sizeof(NROWS) / sizeof(NROWS[0])); ++ni) {
                    const int nrows = NROWS[ni];
                    const int lvalid = lvals[li];
                    const int lanes_last = lvalid + nrows - 1;
                    const M16Census C = m16_census(lvalid, lvalid - 1, nrows, split, full, TILE);
                    ++checked;
                    // 5. COVER, stated independently of the walk above.
                    const int tiles = attn_m16_tiles_of(nrows);
                    if (!(tiles * ATTN_M16_M >= nrows && (tiles - 1) * ATTN_M16_M < nrows)) {
                        printf("GUARD FAIL cover: nrows=%d gives %d tiles, which does not cover it "
                               "minimally\n", nrows, tiles);
                        ++fails;
                    }
                    // A group is empty for two legitimate reasons and no others: the tile it lives in
                    // is not completely filled with EXISTING rows (nrows is not a multiple of the
                    // tile), or this lane block is beyond that row's own lane grid. Anything else means
                    // the walk is skipping work it should do.
                    if (C.empty_groups > 0 && nrows % ATTN_M16_M == 0 &&
                        split <= attn_m16_split(lvalid)) {
                        printf("GUARD FAIL cover: nrows=%d split=%d has %d empty groups although every "
                               "tile is filled with existing rows and no row is past its own grid\n",
                               nrows, split, C.empty_groups);
                        ++fails;
                    }
                    const M16Census P =
                        m16_census(lvalid, lvalid - 1, nrows, split, full, TILE, /*pad=*/1);
                    if (P.empty_groups < C.empty_groups + 4 * split) {
                        printf("GUARD FAIL cover: a padded grid (nrows=%d split=%d) gave %d empty "
                               "groups, fewer than the %d the unpadded run plus the padded tile "
                               "require — the empty-tile route is not what --pad-tiles launches\n",
                               nrows, split, P.empty_groups, C.empty_groups + 4 * split);
                        ++fails;
                    }

                    const bool legal_split = split >= attn_m16_split(lanes_last);
                    const char* why = "";
                    const bool ok =
                        attn_m16_ok(0, 0, 32, split, lvalid, lanes_last, nrows, full, &why);
                    if (legal_split && C.onepass_rows == 0) {
                        // 1. BOUND, as the kernel computes it.
                        if (C.max_round > worst_round) {
                            worst_round = C.max_round; worst_lanes = lvalid; worst_split = split;
                        }
                        if (C.max_slice > worst_slice) worst_slice = C.max_slice;
                        if (C.over) {
                            if (verbose)
                                printf("GUARD FAIL bound: lvalid=%d split=%d full=%d nrows=%d stages a "
                                       "round of %d > M16_L %d\n", lvalid, split, full, nrows,
                                       C.max_round, M16_L);
                            ++fails;
                        }
                    }
                    // 2 and 3: the launcher's verdict must agree with the kernel's, both ways.
                    if (!legal_split) {
                        if (ok) {
                            printf("GUARD FAIL leak: lvalid=%d split=%d nrows=%d accepted although "
                                   "split_for_lanes(%d) needs %d\n", lvalid, split, nrows, lanes_last,
                                   attn_m16_split(lanes_last));
                            ++fails; ++leaked;
                        }
                        continue;
                    }
                    if (ok && (C.over || C.onepass_rows)) {
                        printf("GUARD FAIL leak: lvalid=%d split=%d full=%d nrows=%d accepted but the "
                               "kernel would poison (round %d, one-pass rows %d)\n", lvalid, split,
                               full, nrows, C.max_round, C.onepass_rows);
                        ++fails; ++leaked;
                    }
                    if (!ok && !C.over && !C.onepass_rows) {
                        printf("GUARD FAIL phantom refusal: lvalid=%d split=%d full=%d nrows=%d "
                               "refused for \"%s\" but round %d fits and no row is one-pass\n", lvalid,
                               split, full, nrows, why, C.max_round);
                        ++fails; ++phantom;
                    }
                    if (!ok) ++refused;
                }
            }
        }
    }

    // 4. The plans this op runs.
    int nreal = 0;
    const M16RealCase* real = attn_m16_real_cases(&nreal);
    for (int i = 0; i < nreal; ++i) {
        const M16RealCase& c = real[i];
        const int lanes_last = c.lvalid + c.nrows - 1;
        const char* why = "";
        const bool ok = attn_m16_ok(0, 0, 32, c.split, c.lvalid, lanes_last, c.nrows, c.full, &why);
        const M16Census C = m16_census(c.lvalid, c.lvalid - 1, c.nrows, c.split, c.full, TILE);
        const char* why1 = "";
        const bool ok_full1 =
            attn_m16_ok(0, 0, 32, c.split, c.lvalid, lanes_last, c.nrows, 1, &why1);
        bool bad = !ok;
        if (c.streams && !ok_full1) bad = true;      // a streaming plan must survive the shipping mode
        if (!c.streams && ok_full1) bad = true;      // a one-pass plan must be refused by it
        if (bad) {
            printf("GUARD FAIL real plan %s (lvalid=%d nrows=%d split=%d full=%d streams=%d): %s\n",
                   c.name, c.lvalid, c.nrows, c.split, c.full, c.streams ? 1 : 0,
                   ok ? (c.streams ? "refused under full=1" : "accepted under full=1, so it is not "
                                    "one-pass after all") : why);
            ++fails;
        } else if (verbose) {
            printf("  [%s] %-18s lvalid=%4d rows=%2d split=%d full=%d | widest round %2d of %2d, "
                   "slice %3d, %d blocks, %d one-pass rows\n", c.streams ? "ok  " : "full=0",
                   c.name, c.lvalid, c.nrows, c.split, c.full, C.max_round, M16_L, C.max_slice,
                   C.blocks, C.onepass_rows);
        }
    }

    printf("geom-guard m16: %d geometries walked (lvalid census x split 1..%d x full 0/1 x nrows "
           "1/2/3/4/5/15/16/17/32/64) | worst round %d of M16_L %d (at lvalid=%d split=%d) | worst "
           "slice %d | refused %d (phantom %d, leak %d) | %d real plan geometries | LDS %d B/block -> "
           "%d blocks/CU by LDS = %d wavefronts/CU at %d warps/block (the shipped attn_slim_wave: "
           "12,288 B -> 5 blocks = 20)\n",
           checked, ATTN_M16_MAXSPLIT, worst_round, M16_L, worst_lanes, worst_split, worst_slice,
           refused, phantom, leaked, nreal, kM16LdsBytes, 65536 / kM16LdsBytes,
           (65536 / kM16LdsBytes) * (ATTN_M16_THREADS / 32), ATTN_M16_THREADS / 32);
    printf("geom-guard m16: %s\n", fails ? "FAIL" : "PASS");
    return fails;
}

// Loud poison, so a route the launcher failed to refuse cannot become a silent numerics diff
// (`m2_panic`'s precedent; an LDS over-run and a wrong route both look like numerics on this part).
__device__ __forceinline__ void m16_panic(const CoreArgs& a, int row) {
    if (row < 0 || row >= nrows_of(a)) return;
    for (int d = threadIdx.x; d < (int)ROW_H; d += blockDim.x) {
        a.aout[(size_t)ROW_H * row + d] = fb(nanf("m16route"));
        a.gout[(size_t)ROW_H * row + d] = fb(nanf("m16route"));
    }
}

// =============================================================================================
// THE FORM. One block: 16 query rows (grid.z), one lane block (grid.y), one KV group (grid.x),
// G warps = one per query head of the group — the shipped block shape — walking K/V four rows at a
// time so 16 (row, head) pairs share one staged lane row.
// =============================================================================================
__global__ void attn_m16_wave(CoreArgs a) {
    __shared__ hip_bfloat16 s_sc[M16_PAIRS * M16_L];   // exact-BF16 scores, one round window per pair
    __shared__ hip_bfloat16 s_kv[M16_L * DH];          // K rows for the round, then V rows (aliased)
    __shared__ hip_bfloat16 s_sk[ATTN_M16_GRP * DH];   // this wave's own appended lane, per live row
    __shared__ hip_bfloat16 s_sv[ATTN_M16_GRP * DH];

    const int tile = (a.tile > 0 && a.tile <= TILE) ? a.tile : TILE;
    const int kv_read = tile_kv_of(blockIdx.x, a.neg);  // group whose K/V this block reads
    const int kv_write = blockIdx.x;                    // group whose lanes this wave appends
    const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;               // == ATTN_M16_G, by the static_assert
    const int h = w % ATTN_M16_G;                       // warp w owns query head h, as shipped
    const int head = ATTN_M16_G * kv_write + h;         // FULL head index: slots, outputs, gates
    const int y = blockIdx.y;                           // lane block
    const int rt = gridDim.z - 1 - blockIdx.z;          // widest tile first: §2's dispatch tail
    const int nd = (a.neg == NEG_DIMHALF) ? 4 : 8;      // of the 8 BF16 a lane owns

    // The four folded permute addresses (deltas 8, 4, 2, 1): functions of `lane` alone, so
    // loop-invariant across every touch this block makes. The shipped slim form's hoist.
    unsigned pa[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) pa[i] = vslim_permute_addr(lane, 8 >> i);

    constexpr int kGroups = ATTN_M16_M / ATTN_M16_GRP;
    // ROLLED on purpose, and this is the one place the roll is load-bearing for registers: unrolling
    // this loop keeps `st[][]`/`acc[][]` of several groups live at once and the compiler's own remark
    // went from ~60 to 159 VGPR/lane, which on the driver's count is 1 block/CU (`m16_scalar_ob` at
    // 152 VGPR is the same story, `m16_census.output.txt` §1). The inner loops over ATTN_M16_GRP stay
    // unrolled, which is what keeps `M16Row g[]` and the per-row state in registers (ScratchSize 0).
    for (int gi = 0; gi < kGroups; ++gi) {
        M16Row g[ATTN_M16_GRP];
        int kmax = 0, nact = 0, widest = 0;
        bool onepass = false;
#pragma unroll
        for (int i = 0; i < ATTN_M16_GRP; ++i) {
            g[i] = m16_row_at(a, ATTN_M16_M * rt + ATTN_M16_GRP * gi + i, y, tile);
            if (g[i].exists && m16_row_onepass(a, g[i].row)) onepass = true;
            if (g[i].active) {
                ++nact;
                if (g[i].chunks > kmax) kmax = g[i].chunks;
            }
        }
        if (onepass) {            // a launcher that refused nothing gets poison, then a clean exit
            for (int i = 0; i < ATTN_M16_GRP; ++i) m16_panic(a, g[i].row);
            return;               // before any barrier: `onepass` is a per-block OR, so this is uniform
        }
        if (nact == 0) continue;  // the grid-overhang route: nothing staged, no barrier ever run

        // Round bounds must be block-uniform, so every one of them is a min/max over the group. This
        // pre-pass only does integer arithmetic, and it is what lets an over-wide round refuse BEFORE
        // any barrier instead of overrunning a stage that does not fault when it is overrun.
        for (int k = 0; k < kmax; ++k) {
            int lo = 0x7fffffff, hi = 0;
#pragma unroll
            for (int i = 0; i < ATTN_M16_GRP; ++i) {
                int cl, ch;
                m16_chunk_of(g[i], k, tile, &cl, &ch);
                if (ch > cl) { if (cl < lo) lo = cl; if (ch > hi) hi = ch; }
            }
            const int un = hi > lo ? hi - lo : 0;
            if (un > widest) widest = un;
        }
        if (widest > M16_L) {
            for (int i = 0; i < ATTN_M16_GRP; ++i) m16_panic(a, g[i].row);
            return;               // uniform, and still before this group's first barrier
        }

        float st[ATTN_M16_GRP][2];      // (m, l) per live row. Neutral m = 0, l = 0, as shipped.
        float acc[ATTN_M16_GRP][8];     // the live accumulator budget (see the static_assert)
#pragma unroll
        for (int i = 0; i < ATTN_M16_GRP; ++i) {
            st[i][0] = 0.f; st[i][1] = 0.f;
#pragma unroll
            for (int u = 0; u < 8; ++u) acc[i][u] = 0.f;
        }

        // This wave's own appended lane, per live row of this group. Both shipped wave forms read lane
        // `pos + row` from the STAGE and not the cache (m2 constraint 4), and rows of different groups
        // land in different rounds, so each live row gets a row of its own. Indexed by ROW, like
        // `attn_tile_body`/`attn_m2_body`: `attn_prep` wrote `krope[ROW_KV * qrow + kv * DH]`, keyed by
        // the row index; the *test* for the self lane is against the POSITION (`j == g[i].pos`).
        __syncthreads();                 // also closes the previous group's reads of these four rows
        for (int d = threadIdx.x; d < DH; d += blockDim.x) {
#pragma unroll
            for (int i = 0; i < ATTN_M16_GRP; ++i) {
                if (!g[i].active) continue;
                s_sk[i * DH + d] = a.krope[(size_t)ROW_KV * g[i].row + (size_t)kv_write * DH + d];
                s_sv[i * DH + d] = a.vg[(size_t)ROW_KV * g[i].row + (size_t)kv_write * DH + d];
            }
        }
        __syncthreads();

        for (int k = 0; k < kmax; ++k) {
            int lo = 0x7fffffff, hi = 0;
#pragma unroll
            for (int i = 0; i < ATTN_M16_GRP; ++i) {
                int cl, ch;
                m16_chunk_of(g[i], k, tile, &cl, &ch);
                if (ch > cl) { if (cl < lo) lo = cl; if (ch > hi) hi = ch; }
            }
            const int u_lo = hi > lo ? lo : 0, u_hi = hi;

            // ---- stage K for the round's union: one warp per lane row, one `uint4` per lane at
            // `slot * DH + lane * 8`, the shipped staging shape. `slot`, never a lane index.
            for (int j = u_lo + w; j < u_hi; j += nw) {
                const int slot = j - u_lo;
                const hip_bfloat16* ksrc = a.kcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
                *reinterpret_cast<uint4*>(&s_kv[slot * DH + lane * 8]) =
                    *reinterpret_cast<const uint4*>(ksrc + lane * 8);
            }
            __syncthreads();                                   // K sub-block published

            // ---- score every live row's chunk k. One (row, lane, head) touch = one dot + one fold +
            // one BF16 round: the same unit the tile-side census books, so the per-touch rows of the
            // two forms are comparable.
#pragma unroll
            for (int i = 0; i < ATTN_M16_GRP; ++i) {
                int c0, c1;
                m16_chunk_of(g[i], k, tile, &c0, &c1);
                if (c1 <= c0) continue;
                const int off = c0 - u_lo, cnt = c1 - c0, sp = ATTN_M16_G * i + h;
                float q[8];
                const uint4 qu = *reinterpret_cast<const uint4*>(
                    a.qrope + (size_t)ROW_H * g[i].row + (size_t)head * DH + lane * 8);
                const hip_bfloat16* qq = reinterpret_cast<const hip_bfloat16*>(&qu);
#pragma unroll
                for (int t = 0; t < 8; ++t) q[t] = bf(qq[t]);
                auto publish = [&](int slot, float dot) {
                    dot = vslim_fold<VSLIM_DOT | VSLIM_FOLD>(dot, pa);   // LEVER 2, called not copied
                    // One BF16 round, then the scale; exact in BF16 because SCALE = 2^-4, which is
                    // what lets the staging be 2 bytes and still read back the identical F32.
                    if (lane == 0) s_sc[sp * M16_L + slot] = fb(bf(fb(dot)) * SCALE);
                };
                // The dim-half injection gets its own copy of the loop instead of a runtime trip
                // count (the shipped slim form's shape), and a union lane that is THIS row's self lane
                // is read from `s_sk`, which is what keeps `slotk` visible.
                if (nd == 8) {
#pragma unroll
                    for (int u = 0; u < TILE; ++u) {      // cnt <= tile <= TILE, so TILE bounds it
                        if (u >= cnt) continue;
                        const int j = c0 + u;
                        const hip_bfloat16* kp = (j == g[i].pos) ? &s_sk[i * DH + lane * 8]
                                                                 : &s_kv[(off + u) * DH + lane * 8];
                        publish(off + u, vslim_dot<8>(kp, q));
                    }
                } else {
#pragma unroll
                    for (int u = 0; u < TILE; ++u) {
                        if (u >= cnt) continue;
                        const int j = c0 + u;
                        const hip_bfloat16* kp = (j == g[i].pos) ? &s_sk[i * DH + lane * 8]
                                                                 : &s_kv[(off + u) * DH + lane * 8];
                        publish(off + u, vslim_dot<4>(kp, q));
                    }
                }
            }
            __syncthreads();                                   // s_sc published for the round

            // ---- V for the same union, into the SAME staging: no K row of this round is live now,
            // which is the whole licence for aliasing the two (results.md §7.1's open lever).
            for (int j = u_lo + w; j < u_hi; j += nw) {
                const int slot = j - u_lo;
                const hip_bfloat16* vsrc = a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
                *reinterpret_cast<uint4*>(&s_kv[slot * DH + lane * 8]) =
                    *reinterpret_cast<const uint4*>(vsrc + lane * 8);
            }
            __syncthreads();                                   // V published

            // ---- §07.3's streaming step per row, in the shipped order, over the row's OWN chunk,
            // with `attn_core`'s exact per-chunk algebra, so this lands on the bits the shipped form
            // already streams instead of opening a second set.
#pragma unroll
            for (int i = 0; i < ATTN_M16_GRP; ++i) {
                int c0, c1;
                m16_chunk_of(g[i], k, tile, &c0, &c1);
                if (c1 <= c0) continue;
                const int off = c0 - u_lo, nt = c1 - c0, sp = ATTN_M16_G * i + h;
                float mt = -1e30f;
#pragma unroll
                for (int u = 0; u < TILE; ++u) {          // nt <= tile <= TILE
                    if (u >= nt) continue;
                    mt = fmaxf(mt, bf(s_sc[sp * M16_L + off + u]));
                }
                const float mn = fmaxf(st[i][0], mt);
                const float rescale = expf(st[i][0] - mn);
                float e[TILE], ls = 0.f;      // the shipped form's own size: nt <= tile <= TILE,
                                              // and a smaller array is what keeps the four unrolled
                                              // copies of this body inside the register file.
#pragma unroll
                for (int u = 0; u < TILE; ++u) {
                    if (u >= nt) continue;
                    e[u] = expf(bf(s_sc[sp * M16_L + off + u]) - mn);
                    ls += e[u];
                }
                st[i][1] = st[i][1] * rescale + ls;
#pragma unroll
                for (int u = 0; u < 8; ++u) acc[i][u] *= rescale;
#pragma unroll
                for (int u = 0; u < TILE; ++u) {
                    if (u >= nt) continue;
                    const int j = c0 + u;
                    const hip_bfloat16* v = (j == g[i].pos) ? &s_sv[i * DH + lane * 8]
                                                           : &s_kv[(off + u) * DH + lane * 8];
                    for (int t = 0; t < 8; ++t) acc[i][t] += e[u] * bf(v[t]);   // serial in t, §05.5
                }
                st[i][0] = mn;
            }
            __syncthreads();                    // before the next round overwrites the staging
        }

        // ---- epilogue per row: the shipped split-partial slot, keyed by the row and by the FULL
        // head. `attn_combine` merges the lane blocks with the same (max,sum) algebra for every form.
#pragma unroll
        for (int i = 0; i < ATTN_M16_GRP; ++i) {
            if (!g[i].active) continue;
            if (g[i].s_row == 1) { m16_panic(a, g[i].row); continue; }   // unreachable: refused
            float* pp = a.part + (((size_t)g[i].row * N_Q + head) * a.split + y) * (DH + 2);
            if (lane == 0) { pp[0] = st[i][0]; pp[1] = st[i][1]; }
#pragma unroll
            for (int u = 0; u < 8; ++u) pp[2 + lane * 8 + u] = acc[i][u];
        }
    }
}

// ---- the bit-diff kernel -------------------------------------------------------------------
// One block per query row; the whole row (all N_Q heads x DH of `aout`, or of `gated`) is compared by
// BIT as raw 16-bit patterns, so no tolerance can soften the comparison anywhere. The block total is a
// real reduction — warp butterfly, then one shared slot per warp — because a per-thread store to one
// shared slot is not a reduction (the `mixed_bw` trap), and the reduction must run UNIFORMLY because
// `__syncthreads()` inside a divergent branch aborts the queue on this part rather than returning a
// wrong count (`attn_m2.hip.cpp`'s `diff_rows` comment, device 2026-10-16).
__device__ __forceinline__ int m16_block_sum(int v) {
    // Sized for the widest launch this kernel is ever called with (256 threads = 8 warps). A shred
    // array sized for 4 warps while the launch has 8 walks off the end of LDS and the total comes back
    // as HALF the row, which is exactly what the first version of this kernel did: every differing row
    // reported `ndiff = 1024` of 2,048 elements, i.e. a diff kernel that under-counts by blockDim/8 and
    // makes a broken form look half-right.
    __shared__ int warp_part[8];
    const int nw = blockDim.x / 32;
    // The mask must be 64-bit on this HIP (prototype/README.md).
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FULL_MASK, v, off);
    const int warp = threadIdx.x / 32;
    if ((threadIdx.x & 31) == 0) warp_part[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (int i = 0; i < nw; ++i) t += warp_part[i];
        warp_part[0] = t;
    }
    __syncthreads();
    return warp_part[0];
}

__global__ void m16_diff_rows(const hip_bfloat16* x, const hip_bfloat16* y, int* out, int per_row) {
    const int row = blockIdx.x;
    const unsigned short* a = reinterpret_cast<const unsigned short*>(x) + (size_t)row * per_row;
    const unsigned short* b = reinterpret_cast<const unsigned short*>(y) + (size_t)row * per_row;
    int mine = 0;
    for (int i = threadIdx.x; i < per_row; i += blockDim.x) mine += (a[i] != b[i]);
    const int total = m16_block_sum(mine);      // uniform, by construction: no barrier inside a branch
    if (threadIdx.x == 0) out[row] = total;
}

// ---- the harness ----------------------------------------------------------------------------
namespace {

enum Form { F_SHIP = 0, F_SHIPSLIM = 1, F_M16 = 2, NFORMS = 3 };
// `ship` is `attn_tile_wave` (the frozen shipped tiled form), `shipslim` is `attn_slim1_wave` (the
// form the engine ships today), `m16` is the re-tile. Every other active form is bit-diffed against
// the one named in `Pair::b` below, and the control pair (shipslim vs ship) is printed every run so a
// reader can see that the reference of the gate is itself agreeing with the frozen form.
static const char* kFormName[] = {"ship", "shipslim", "m16"};
static const char* kFormKey[] = {"", "_shipslim", "_m16"};

struct FormSet {
    int f[NFORMS] = {0, 0, 0};
    int n = 0;
    bool has(int form) const {
        for (int i = 0; i < n; ++i) if (f[i] == form) return true;
        return false;
    }
    void add(int form) { if (!has(form)) f[n++] = form; }
    std::string names() const {
        std::string s;
        for (int i = 0; i < n; ++i) { if (i) s += ","; s += kFormName[f[i]]; }
        return s;
    }
};

// `ship` is ALWAYS active: it is the anchor of the control pair, and a diff against a form that never
// ran is worse than no diff. Tokens are matched EXACTLY, because a substring test would make
// `--forms shipslim` silently add `ship` by accident rather than by this rule.
static FormSet forms_of(const std::string& spec) {
    FormSet fs;
    fs.add(F_SHIP);
    fs.add(F_SHIPSLIM);
    fs.add(F_M16);
    if (spec.empty()) return fs;
    FormSet pick;
    size_t p = 0;
    while (p < spec.size()) {
        const size_t c = spec.find(',', p);
        const std::string tok = spec.substr(p, c == std::string::npos ? std::string::npos : c - p);
        int found = -1;
        for (int i = 0; i < NFORMS; ++i) if (tok == kFormName[i]) found = i;
        if (found < 0) {
            fprintf(stderr, "--forms: no form \"%s\"; known: ship,shipslim,m16\n", tok.c_str());
            exit(1);
        }
        pick.add(found);
        if (c == std::string::npos) break;
        p = c + 1;
    }
    pick.add(F_SHIP);          // the anchor, always
    return pick;
}

static int neg_from(const std::string& s) {
    return s == "none" ? NEG_NONE
         : s == "qswap" ? NEG_QSWAP
         : s == "gatehalf" ? NEG_GATEHALF
         : s == "ropefull" ? NEG_ROPEFULL
         : s == "ropenone" ? NEG_ROPENONE
         : s == "gqaoff" ? NEG_GQA
         : s == "nomax" ? NEG_NOMAX
         : s == "carry" ? NEG_CARRY
         : s == "readpast" ? NEG_READPAST
         : s == "dropcurrent" ? NEG_DROPCUR
         : s == "dimhalf" ? NEG_DIMHALF
         : s == "slotk" ? NEG_SLOTK
         : s == "slotv" ? NEG_SLOTV : -1;
}

static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    std::vector<char> b((size_t)f.tellg());
    f.seekg(0);
    f.read(b.data(), (std::streamsize)b.size());
    if ((size_t)f.gcount() != b.size()) { fprintf(stderr, "%s: short read\n", p.c_str()); exit(1); }
    return b;
}

// One replica of one file per active form: the §05.7 append writes the cache IN PLACE, so a second
// launch would otherwise read the first form's appended lanes at the first form's slots.
static void upload_forms(const std::string& path, size_t expect_elems, const FormSet& fs,
                         hip_bfloat16** dst) {
    const std::vector<char> raw = slurp(path);
    if (raw.size() != expect_elems * 2) {
        fprintf(stderr, "%s: %zu bytes, expected %zu BF16 elements\n", path.c_str(), raw.size(),
                expect_elems);
        exit(1);
    }
    for (int i = 0; i < fs.n; ++i) {
        hip_bfloat16** d = dst + fs.f[i];
        CK(hipMalloc(d, raw.size()));
        CK(hipMemcpy(*d, raw.data(), raw.size(), hipMemcpyHostToDevice));
    }
}

static hip_bfloat16* upload(const std::string& path, size_t expect_elems) {
    const std::vector<char> raw = slurp(path);
    if (raw.size() != expect_elems * 2) {
        fprintf(stderr, "%s: %zu bytes, expected %zu BF16 elements\n", path.c_str(), raw.size(),
                expect_elems);
        exit(1);
    }
    hip_bfloat16* d = nullptr;
    CK(hipMalloc(&d, raw.size()));
    CK(hipMemcpy(d, raw.data(), raw.size(), hipMemcpyHostToDevice));
    return d;
}

static void dump_file(const std::string& path, const void* dptr, size_t elems) {
    std::vector<char> h(elems * 2);
    CK(hipMemcpy(h.data(), const_cast<void*>(dptr), elems * 2, hipMemcpyDeviceToHost));
    std::ofstream of(path, std::ios::binary);
    of.write(h.data(), (std::streamsize)h.size());
}

struct Case {
    std::string name, outdir;
    int pos = 0, lvalid = 0, cap = 0, nrows = 1;
    hip_bfloat16 *qg = nullptr, *kg = nullptr, *vg = nullptr, *qn = nullptr, *kn = nullptr;
    hip_bfloat16 *kc[NFORMS] = {nullptr, nullptr, nullptr};
    hip_bfloat16 *vc[NFORMS] = {nullptr, nullptr, nullptr};
    hip_bfloat16 *qrope[NFORMS] = {nullptr, nullptr, nullptr};
    hip_bfloat16 *krope[NFORMS] = {nullptr, nullptr, nullptr};
    hip_bfloat16 *gate = nullptr, *rcs = nullptr;
    hip_bfloat16 *aout[NFORMS] = {nullptr, nullptr, nullptr};
    hip_bfloat16 *gated[NFORMS] = {nullptr, nullptr, nullptr};
    float *part[NFORMS] = {nullptr, nullptr, nullptr};
    int *diff = nullptr;
    int lanes_last() const { return lvalid + nrows - 1; }
};

struct Pair { int a, b; const char* key; };

}  // namespace

int main(int argc, char** argv) {
    std::string plan_path, negname = "none", outdir = ".", formspec;
    int split = 0, tile = TILE, nrows_cap = 0, burst = 0, reps = 5, full = 1, pad_tiles = 0;
    int geom_guard = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--plan") plan_path = next();
        else if (a == "--neg") negname = next();
        else if (a == "--split") split = std::atoi(next().c_str());
        else if (a == "--tile") tile = std::atoi(next().c_str());
        else if (a == "--rows") nrows_cap = std::atoi(next().c_str());
        else if (a == "--out") outdir = next();
        else if (a == "--full") full = std::atoi(next().c_str());
        else if (a == "--pad-tiles") pad_tiles = std::atoi(next().c_str());
        else if (a == "--burst") burst = std::atoi(next().c_str());
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--forms") formspec = next();
        else if (a == "--geom-guard") geom_guard = 1;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    // Host-only: no HIP call, no device, no lock.
    if (geom_guard) return attn_m16_guard(1) ? 1 : 0;

    const FormSet fs = forms_of(formspec);
    // The bit-diff pairs: the re-tile against the form the engine ships, and the shipped slim form
    // against the frozen tiled form. The second pair is a CONTROL row: it is expected at 0 for reasons
    // this file did not create (D22's levers), and it is printed every run so the reference of the
    // first pair is seen to be sound rather than trusted.
    std::vector<Pair> pairs;
    if (fs.has(F_M16)) pairs.push_back({F_M16, F_SHIPSLIM, kFormKey[F_M16]});
    if (fs.has(F_SHIPSLIM)) pairs.push_back({F_SHIPSLIM, F_SHIP, kFormKey[F_SHIPSLIM]});
    if (pairs.empty()) { fprintf(stderr, "--forms must name a form besides `ship`\n"); return 1; }
    const int neg = neg_from(negname);
    if (neg < 0) { fprintf(stderr, "unknown --neg %s\n", negname.c_str()); return 1; }
    std::ifstream pf(plan_path);
    if (!pf) { fprintf(stderr, "cannot open plan %s\n", plan_path.c_str()); return 1; }

    std::vector<Case> cases;
    std::string line;
    int ln = 0;
    while (std::getline(pf, line)) {
        ++ln;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string qgf, kgf, vgf, kcf, vcf, qnf, knf;
        Case c;
        if (!(ss >> c.name >> qgf >> kgf >> vgf >> kcf >> vcf >> qnf >> knf >> c.pos >> c.lvalid
              >> c.outdir)) {
            fprintf(stderr, "plan %s line %d: NAME QG KG VG KCACHE VCACHE QNORM KNORM POS LCACHE "
                            "OUTDIR [NROWS]\n", plan_path.c_str(), ln);
            return 1;
        }
        ss >> c.nrows;
        if (c.nrows < 1) c.nrows = 1;
        if (nrows_cap > 0 && c.nrows != nrows_cap) continue;      // run one geometry in isolation
        const std::vector<char> kraw = slurp(kcf);
        c.cap = (int)(kraw.size() / ((size_t)N_KV * DH * 2));
        if (c.lvalid < 0 || c.lanes_last() > c.cap || c.pos + c.nrows - 1 >= c.cap) {
            fprintf(stderr, "%s: pos %d (+%d rows) / lvalid %d do not fit cap %d\n", c.name.c_str(),
                    c.pos, c.nrows - 1, c.lvalid, c.cap);
            return 1;
        }
        const int need = attn_m16_split(c.lanes_last());
        const int use_split = split > 0 ? split : need;
        if (use_split < need) {
            // Not a refusal: a wave's widest row decides the lane grid, and a shallower grid would cut
            // rows the way the shipped form does not. Raised here and echoed, so EVERY form is launched
            // with the SAME `split` and the partial slots cannot mean two things.
            fprintf(stderr, "%s: split %d is shallower than split_for_lanes(%d) = %d\n",
                    c.name.c_str(), use_split, c.lanes_last(), need);
            return 1;
        }
        c.qg = upload(qgf, (size_t)N_Q * 2 * DH * c.nrows);
        c.kg = upload(kgf, (size_t)N_KV * DH * c.nrows);
        c.vg = upload(vgf, (size_t)N_KV * DH * c.nrows);
        c.qn = upload(qnf, DH);
        c.kn = upload(knf, DH);
        upload_forms(kcf, (size_t)N_KV * c.cap * DH, fs, c.kc);
        upload_forms(vcf, (size_t)N_KV * c.cap * DH, fs, c.vc);
        for (int fi = 0; fi < fs.n; ++fi) {
            const int f = fs.f[fi];
            CK(hipMalloc(&c.qrope[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.krope[f], (size_t)N_KV * DH * 2 * c.nrows));
            CK(hipMalloc(&c.aout[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.gated[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMemset(c.aout[f], 0xA5, (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMemset(c.gated[f], 0xA5, (size_t)N_Q * DH * 2 * c.nrows));
            // ONE footprint expression for every form: the partial layout is shared slot for slot, so
            // `attn_combine` stays one function (§20.1).
            const size_t pe = (size_t)attn_m16_part_floats(c.nrows, c.lanes_last()) * 4;
            CK(hipMalloc(&c.part[f], pe));
            CK(hipMemset(c.part[f], 0, pe));
        }
        CK(hipMalloc(&c.gate, (size_t)N_Q * DH * 2 * c.nrows));
        CK(hipMalloc(&c.rcs, (size_t)ROT * 2 * 2));
        CK(hipMalloc(&c.diff, (size_t)pairs.size() * c.nrows * 4 * 2));
        cases.push_back(c);
    }
    if (cases.empty()) { fprintf(stderr, "plan %s has no cases\n", plan_path.c_str()); return 1; }

    int rows_total = 0, worst_lanes = 1;
    for (auto& c : cases) { rows_total += c.nrows; worst_lanes = std::max(worst_lanes, c.lanes_last()); }
    const int use_split = split > 0 ? split : attn_m16_split(worst_lanes);
    const char* why = "";
    // The refusal is per CASE, not per plan, so a mixed plan reports which wave it cannot serve and
    // why, instead of printing a bare exit code. `attn_m16_ok` is the one predicate; the guard walks it.
    for (auto& c : cases) {
        if (!attn_m16_ok(0, 0, 32, use_split, c.lvalid, c.lanes_last(), c.nrows, full, &why)) {
            fprintf(stderr, "REFUSED case=%s rows=%d lanes=%d-%d split=%d full=%d: %s\n",
                    c.name.c_str(), c.nrows, c.lvalid, c.lanes_last(), use_split, full, why);
            return 2;
        }
    }
    printf("plan %s | cases=%zu | rows=%d | split=%d | tile=%d | full=%d | neg=%s | forms=%s"
           " | pad_tiles=%d | MAXSC=%d TILE=%d | M=%d GRP=%d L=%d PAIRS=%d | LDS=%d B -> %d blk/CU |"
           " grid m16=%dx%dx%d | grid tile=%dx%dx%d\n",
           plan_path.c_str(), cases.size(), rows_total, use_split, tile, full, negname.c_str(),
           fs.names().c_str(), pad_tiles, MAXSC, TILE, ATTN_M16_M, ATTN_M16_GRP, M16_L, M16_PAIRS,
           kM16LdsBytes, 65536 / kM16LdsBytes, N_KV, use_split,
           attn_m16_tiles_of(rows_total) + pad_tiles, N_KV, use_split, 1);
    if (neg != NEG_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());

    // ONE place builds a launch plan, used by the correctness path and the burst path alike, so a
    // timing can never be taken on a different launch plan from the one that was gated.
    auto core_args = [&](Case& c, int form) {
        CoreArgs a{};
        a.qg = c.qg; a.kg = c.kg; a.vg = c.vg;
        a.kcache = c.kc[form]; a.vcache = c.vc[form];
        a.kc_as = a.kcache; a.vc_as = a.vcache;      // the append writes in place (§05.7)
        a.qn = c.qn; a.kn = c.kn;
        a.qrope = c.qrope[form]; a.krope = c.krope[form];
        a.aout = c.aout[form]; a.gout = c.gated[form];
        a.rope_out = c.rcs;
        a.part = c.part[form]; a.stale = nullptr;
        a.cap = c.cap; a.pos = c.pos; a.lvalid = c.lvalid; a.nrows = c.nrows;
        a.split = use_split;                         // IDENTICAL for every form, on purpose
        a.neg = neg; a.full = full; a.score_f32 = 0; a.fused = 0; a.tile = tile;
        // Which CORE a one-row wave would take. `attn_combine` does not read it, and this form's
        // partial layout is the shipped one, so 0 is honest here rather than a new value.
        a.tile_kernel = 0;
        return a;
    };
    auto launch_wave = [&](Case& c, int form) {
        CoreArgs a = core_args(c, form);
        // The shipped wave form is ONE BLOCK PER QUERY ROW (grid.z, read back as
        // `row = gridDim.z - 1 - blockIdx.z`), and its lane grid is `max(split, split_for_lanes)`.
        // Launching it with grid.z = 1 does not fail: it computes row 0, leaves every other row's
        // partials at zero, and `attn_combine` then writes 0/0 = NaN for those rows, so the pair
        // comparison reports a NaN-vs-value diff on every row but the first and looks like a numerics
        // failure in the form under test. That is the shape of the first device run of this harness
        // (2026-10-18), and `attn_m2.hip.cpp`'s `tgrid` is the line that says the correct shape.
        const dim3 tgrid(N_KV, std::max(use_split, tile_split_of(c.lanes_last())), c.nrows);
        switch (form) {
            case F_SHIP:     attn_tile_wave<<<tgrid, ATTN_TILE_THREADS>>>(a); break;
            case F_SHIPSLIM: attn_slim1_wave<<<tgrid, ATTN_TILE_THREADS>>>(a); break;
            case F_M16:      attn_m16_wave<<<dim3(N_KV, use_split,
                                       attn_m16_tiles_of(c.nrows) + pad_tiles), ATTN_M16_THREADS>>>(a);
                             break;
            default: fprintf(stderr, "no kernel for form %d\n", form); exit(1);
        }
        CK(hipGetLastError());
    };

    auto run_form = [&](Case& c, int form) {
        CoreArgs a = core_args(c, form);
        CK(hipMemset(c.part[form], 0,
                     (size_t)attn_m16_part_floats(c.nrows, c.lanes_last()) * 4));
        CoreArgs p = a;
        p.qrope_out = c.qrope[form]; p.krope_out = c.krope[form]; p.gate_out = c.gate;
        attn_prep<<<dim3(N_Q + N_KV, c.nrows), THREADS>>>(p);
        a.qrope = c.qrope[form]; a.krope = c.krope[form];
        launch_wave(c, form);
        attn_combine<<<dim3(N_Q, c.nrows), THREADS>>>(a);   // the shipped merge, every form
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
    };

    // BYTES DEMANDED, not DRAM bytes (the GL2C calibration failed by ~100x, decision D9). The shipped
    // forms demand every lane row of every row's own slice, twice (K then V); the re-tile demands the
    // lanes of the rounds it stages, twice, from the SAME walk `attn_m16_ok` refuses on, so the two
    // numbers are arithmetic on the two forms' own loops and not a hoped-for ratio.
    auto bytes_demanded = [&](Case& c, bool m16_form) {
        if (!m16_form) {
            int lanes_total = 0;
            for (int r = 0; r < c.nrows; ++r) lanes_total += c.lvalid + r;
            return (double)lanes_total * N_KV * 2 * DH * 2;
        }
        const M16Census C = m16_census(c.lvalid, c.pos, c.nrows, use_split, full, tile, pad_tiles);
        return (double)C.round_lanes * N_KV * 2 * DH * 2;
    };

    int total_fail = 0;
    for (auto& c : cases) {
        for (int i = 0; i < fs.n; ++i) run_form(c, fs.f[i]);
        for (size_t p = 0; p < pairs.size(); ++p) {
            m16_diff_rows<<<c.nrows, 256>>>(c.aout[pairs[p].a], c.aout[pairs[p].b],
                                            c.diff + p * c.nrows * 2, N_Q * DH);
            m16_diff_rows<<<c.nrows, 256>>>(c.gated[pairs[p].a], c.gated[pairs[p].b],
                                            c.diff + p * c.nrows * 2 + c.nrows, N_Q * DH);
        }
        std::vector<int> dhost((size_t)pairs.size() * c.nrows * 2);
        CK(hipMemcpy(dhost.data(), c.diff, dhost.size() * sizeof(int), hipMemcpyDeviceToHost));
        const double b_ship = bytes_demanded(c, false);
        const double b_m16 = bytes_demanded(c, true);
        const M16Census C = m16_census(c.lvalid, c.pos, c.nrows, use_split, full, tile, pad_tiles);
        printf("RESULT case=%s rows=%d lanes=%d-%d split=%d full=%d forms=%s widest_round=%d"
               " empty_groups=%d",
               c.name.c_str(), c.nrows, c.lvalid, c.lanes_last(), use_split, full,
               fs.names().c_str(), C.max_round, C.empty_groups);
        for (size_t p = 0; p < pairs.size(); ++p) {
            int suma = 0, sumg = 0;
            for (int r = 0; r < c.nrows; ++r) {
                suma += dhost[p * c.nrows * 2 + r];
                sumg += dhost[p * c.nrows * 2 + c.nrows + r];
            }
            printf(" ndiff_aout%s=%d ndiff_gated%s=%d", pairs[p].key, suma, pairs[p].key, sumg);
            if (suma || sumg) {
                for (int r = 0; r < c.nrows; ++r) {
                    const int da = dhost[p * c.nrows * 2 + r];
                    const int dg = dhost[p * c.nrows * 2 + c.nrows + r];
                    if (da || dg)
                        printf("\n  ROWDIFF pair=%s_vs%s case=%s row=%d pos=%d lanes=%d"
                               " ndiff_aout=%d of %d elements ndiff_gated=%d",
                               kFormName[pairs[p].a], kFormName[pairs[p].b], c.name.c_str(), r,
                               c.pos + r, c.lvalid + r, da, N_Q * DH, dg);
                }
                ++total_fail;
            }
        }
        printf(" bytesdemanded_ship=%.0f bytesdemanded_m16=%.0f ratio=%.4f blocks_ship=%d"
               " blocks_m16=%d\n",
               b_ship, b_m16, b_m16 / b_ship, N_KV * use_split * c.nrows, C.blocks);
        if (!outdir.empty()) {
            for (int i = 0; i < fs.n; ++i) {
                const int f = fs.f[i];
                dump_file(outdir + "/" + c.name + "." + kFormName[f] + ".aout.bin", c.aout[f],
                          (size_t)N_Q * DH * c.nrows);
            }
        }
    }
    printf("RESULT mode=ab cases=%zu rows=%d neg=%s full=%d forms=%s cases_with_any_difference=%d"
           " claim=bit-identity-per-row-on-the-per-row-lane-order\n",
           cases.size(), rows_total, negname.c_str(), full, fs.names().c_str(), total_fail);

    // Burst timing: MAIN AGENT ONLY (workers may not take timings, AGENTS.md). Two disciplines kept
    // from `prototype/gdn_rec_isa`: (a) NO TIMING IF ADMISSION IS RED — a millisecond figure taken on
    // a form that does not reproduce the shipped bits measures nothing; (b) every active form is timed
    // INSIDE one rep, so a mid-window disturbance hits all of them and the comparison is one ruler and
    // not two windows. `steady_clock`, never `hipEventElapsedTime`, which returns exactly 0 for a sub-ms
    // kernel on this box. Only the wave launch is looped: `attn_prep` and `attn_combine` are identical
    // for every form, so timing the trio would add the same constant to both sides and dilute the
    // ratio. Pages are warmed first because the first pass over fresh pages runs ~35 GB/s.
    if (burst > 0 && neg == NEG_NONE) {
        if (total_fail) {
            printf("NO-TIMING: admission is red (%d case(s) differ), so no millisecond figure is "
                   "printed for this run\n", total_fail);
            return 1;
        }
        for (auto& c : cases) {
            std::vector<double> best(fs.n, 1e30);
            for (int i = 0; i < fs.n; ++i) launch_wave(c, fs.f[i]);
            CK(hipDeviceSynchronize());
            for (int r = 0; r < reps; ++r) {
                for (int i = 0; i < fs.n; ++i) {
                    const int form = fs.f[i];
                    auto t0 = std::chrono::steady_clock::now();
                    for (int n = 0; n < burst; ++n) launch_wave(c, form);
                    CK(hipDeviceSynchronize());
                    auto t1 = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / burst;
                    if (ms < best[i]) best[i] = ms;
                    printf("BURST case=%s rep=%d form=%s ms=%.4f\n", c.name.c_str(), r,
                           kFormName[form], ms);
                }
            }
            printf("RESULT case=%s burst=%d reps=%d forms=%s", c.name.c_str(), burst, reps,
                   fs.names().c_str());
            for (int i = 0; i < fs.n; ++i)
                printf(" best_%s_ms=%.4f", kFormName[fs.f[i]], best[i]);
            printf(" no_timing_taken=0\n");
            for (int i = 0; i < fs.n; ++i)
                for (int j = i + 1; j < fs.n; ++j)
                    printf("SPEEDUP a=%s b=%s best_a_ms=%.4f best_b_ms=%.4f speedup_a_over_b=%.4f\n",
                           kFormName[fs.f[i]], kFormName[fs.f[j]], best[i], best[j], best[j] / best[i]);
        }
    } else {
        printf("NOT TIMED: no timing taken in this run (pass --burst N to time the wave launch)\n");
    }
    return total_fail ? 1 : 0;
}
