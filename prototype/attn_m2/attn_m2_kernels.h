/*
 * prototype/attn_m2/attn_m2_kernels.h — R1, attention row-tiling at M = 2: TWO query rows share one
 * K/V walk, each row still attends exactly its own prefix, and each row's bits are the shipped form's.
 *
 * WHY (a funded route, not an experiment). `progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md`
 * §3.4 settles that `attn` is byte-bound: its own K/V traffic (Σ_lanes x 2 KiB x 6 layers) sits 7.7x
 * under its measured time at 1024 lanes, while every arithmetic class on this ISA is worth 0.03 ms of
 * it at M = 1. The only lever is re-using one fetched K/V row across more touches. G = 4 (the GQA group)
 * is already spent by the shipped `attn_tile_wave` (`prototype/attn_decode/results.md` §20.5), so the
 * remaining re-use is across QUERY ROWS: at M = 2 one K row load and one V row load feed two rows' dots
 * instead of one. Prize as booked there: 194.6 ms of a 1307.75 ms wave at M = 2, and §3.4's own table
 * says where it stops (M ≈ 7.7, where the scalar-arithmetic floor takes over).
 *
 * THE ADMISSION GATE, and the reason this route needs no user decision: BIT-IDENTITY PER ROW on the
 * per-row lane order (D5 item 3 of progress/2026-10-16/prefill-decision-log-2026-10-16.md; the
 * precedent is `--attn-tile 1` itself, gated at 136/136 and 520/520 with 0 elements differing). If a
 * row's result changes at all, THIS DESIGN IS WRONG — not the band. Nothing here loosens a check.
 *
 * ---------------------------------------------------------------------------------------------
 * WHAT MAKES IT POSSIBLE: the reductions that must be preserved are per ROW, not per BLOCK
 * ---------------------------------------------------------------------------------------------
 * §05.5 of the specification pins, for one (query row, query head), three ordered reductions:
 *   (1) the score dot per lane — 8 serial MACs per thread, then the 5-step `__shfl_down_sync`
 *       butterfly, then ONE BF16 round, then x SCALE;
 *   (2) `l = sum_j exp(z_j - m)` accumulated serially in lane index, with `m = max_j z_j`;
 *   (3) `a = sum_j w_j . v_j` serial in j, with `w_j` rounded to BF16 AFTER the divide on the
 *       one-pass route (§05.5: "computed in F32, then cast"), which is why a partial that has already
 *       divided cannot be re-merged — the rule that killed §12's first attempt.
 * §07.3 adds the streaming form and its scope: the `(max, sum)` pair is reduction state of a SINGLE
 * QUERY ROW and dies with it (A-17); and when a row is cut into blocks the tile grouping and the block
 * boundaries ARE part of the number (the rescale `expf(m - mn)` is not associative), which is exactly
 * why §20 made the tiled form CALL `blocks_for_row` instead of cutting its own windows.
 *
 * None of those reductions involves more than one row. So a block may share the K/V WALK — the fetches
 * — across two rows and still run two private reductions, each over its own row's own slice, in its own
 * index order, with its own (m, l, acc). That is the whole design:
 *
 *   SHARED  the global fetches: K rows and V rows, staged once over UNION(row slices).
 *   PRIVATE the arithmetic: one warp per (row, head-in-group) pair, running `attn_core`'s reductions
 *           verbatim over that row's own [blo, bhi) in that row's own order.
 *
 * THE COLLISION §20.4 named, and how it is dissolved. Two adjacent rows of a wave have n and n+1 lanes,
 * and the shipped lane cut is per row (`span = ceil(n / s_row)`), so their slices for the SAME
 * `blockIdx.y` are not the same interval — at n = 1024, split = 2 they are [512, 1024) and [513, 1025).
 * `static_assert(ATTN_TILE_M == 1)` refuses M > 1 in the shipped header for exactly this reason ("a
 * row-tile would size the K/V stage by the widest PAIR rather than by the staging"). It is sizeable
 * here because the union is bounded and small:
 *
 *   union_y <= span_max + y * (span_max - span_min) <= MAXSC + (split - 1)
 *
 * — the spans of two adjacent rows differ by at most 1, since their lane counts do. `ATTN_M2_STAGE` and
 * `ATTN_M2_VSLOT` are that bound; `attn_m2_ok` refuses a launch that would exceed it; `m2_panic` poisons
 * rather than silently corrupting if a caller gets past both. An out-of-range LDS write does not fault
 * on this part — §13's 1,098,790 moved elements are what that costs — so the sentinel is load-bearing.
 *
 * ---------------------------------------------------------------------------------------------
 * THE FOUR THINGS THAT WOULD BITE, each closed by a named mechanism
 * ---------------------------------------------------------------------------------------------
 * 1. BARRIERS MUST BE BLOCK-UNIFORM. `attn_core` and `attn_tile_wave` may `return` when their row is
 *    out of range, because one row is one block. Here the partner row still needs the stage barriers,
 *    and a divergent `__syncthreads()` is a hang or a torn stage. So every quantity bounding a loop
 *    that contains a barrier — `u_lo`, `u_hi`, `union_n`, `kmax`, the V-chunk bounds — is a min/max over
 *    BOTH rows of the pair: still a per-BLOCK value. Per-row work is skipped by a predicate inside the
 *    loop body, never by skipping the barrier. `m2_pair_geom_at` is where those mins and maxes live, and
 *    it is `__device__ __host__` so the launcher and the CPU mirror price the same geometry.
 * 2. THE PARTIAL SLOT IS THE SHIPPED EXPRESSION, keyed by the row AND the FULL head. §19's failure was
 *    two self-consistent layouts joined by a branch, invisible to both sides (AGENTS.md trap row). M = 2
 *    adds a second way to get it wrong — two rows in one block, so a row term that is dropped or swapped
 *    is a new alias. Hence `NEG_M2_PARTROW` exists as an injected bug with a derived must-catch set, and
 *    `attn_combine` stays one function that never branches on which core ran.
 * 3. THE PAIR IS NOT ONE ROUTE. `blocks_for_row` is per row, so a pair can straddle the one-pass/
 *    streaming boundary: row 0 at 512 lanes is one-pass, row 1 at 513 streams. That is why waves at
 *    exactly MAXSC and MAXSC+1 lanes are mandatory gate cases — M = 2 is what puts the two routes in one
 *    block. Both routes stay alive per row and share only the staging loops. The one-pass row's V
 *    staging is chunked at `tile` width purely as a staging granularity: its weights come from the
 *    whole row, so a chunk boundary is not in its sum. The streaming row's chunks ARE its tiles,
 *    anchored at its own `blo`, one for one with what `attn_core` would have run.
 * 4. THE SELF LANE'S SOURCE IS PER ROW, AND IS NOT SHARED. Lane `pos + row` is this wave's own §05.7
 *    append, and both shipped forms read it from the STAGE (`k_rope`/`v`), not from the cache. In a
 *    shared stage one slot has one source, so the pair's two self lanes get dedicated slots
 *    (`s_kself`, `s_vself`) and everything else comes from the shared stage filled from the CACHE. Every
 *    (row, lane) operand then comes from exactly the memory the shipped form reads it from, so
 *    bit-identity does not have to lean on "the append wrote the same bytes to both places" — which is
 *    true in an honest staged plan and is precisely the assumption `NEG_SLOTK`/`NEG_SLOTV` break. Cost:
 *    one extra K row and one extra V row per row per wave, counted in the census.
 *
 * INJECTIONS THIS FORM ADDS (mirrored and gated in `m2_model.py`; the ones the device can express are
 * runnable through `attn_m2 --neg`):
 *   maskwide   a row attends the pair's UNION instead of its own prefix — the bug the shared walk
 *              invites. Derived blind spots: a row whose own slice already reaches the union end.
 *   partrow    the split-partial address drops the pair-row term, so the two rows alias (§19's class).
 *   partalias  §19's own bug, re-injected: the slot keyed by head-in-group, with no KV-group term.
 *   headcarry  row 1 of a pair seeded from row 0's (m, l) — §07.3/A-17's bug in the one shape M = 2
 *              makes possible (two rows in one block). MIRROR-ONLY, and said so where a device run
 *              would otherwise be silently missing it: with one warp per row the two states are in
 *              different warps, so expressing the leak needs an aliased shared slot whose winner is
 *              whichever warp landed last — an undeterminable value is not a gate witness. The mirror
 *              reproduces the leak deterministically (row 1 seeded from row 0's final state), which is
 *              how §20.6's `headcarry` was gated too.
 *
 * COSTS, priced from these constants (no device touched; the compile census lives in results.md and
 * comes from `hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage -c`):
 *   registers  attn_m2_wave compiles to VGPR 50, SGPR 64, ScratchSize 0 B/lane, LDS 22,656 B/block.
 *              ScratchSize is the number to watch: a stack struct indexed by the row number is local
 *              memory, and the first version measured 96 B/lane — a private-memory stream inside a
 *              kernel whose whole point is bytes. Values-not-arrays in `m2_row_at` is what removed it.
 *   occupancy  blocks/CU = min(65,536 / LDS, 32 waves-per-CU / 8 waves-per-block) = min(2, 4) = 2, so
 *              16 wavefronts/CU of the 32 `rocminfo` reports (devices/strix-halo-official-specs.md
 *              §"Max Waves Per CU: 32"). The shipped tile form is 12,288 B and 4 waves/block, so it
 *              realises min(5, 8) = 5 blocks = 20 wavefronts/CU. M = 2 THEREFORE RESIDES FEWER
 *              WAVEFRONTS PER CU than the form it replaces (16 against 20), and that is the price the
 *              union staging charges; `attn_decode results.md` §20.4's `static_assert` said as much in
 *              advance. The named lever, NOT taken here: `s_k` and `s_v` are never live at the same time
 *              (pass 1 finishes before pass 2 reads a score), so aliasing them into one 8,192 B array
 *              buys 3 blocks/CU = 24 wavefronts — a second change, with its own over-run risk, so it is
 *              listed in results.md as an open item rather than smuggled in beside the admission gate.
 *   work done  exactly the shipped form's (2 rows x G heads x own prefix), plus the pair's union
 *              residue — lanes fetched that only one row of the pair needs. `m2_model.py traffic`
 *              counts that residue per width rather than asserting it away, and names the widths where
 *              it matters (short rows at a wave's start, and a straddling pair).
 *
 * WHAT THIS FORM MAY NOT BE ASKED FOR: `attn_m2_ok` — `attn_tile_ok`'s list (no `score_f32`: the score
 * staging is BF16 and exact only for the pinned `bf(fb(dot)) * SCALE` with SCALE = 2^-4; no `fused`: a
 * wave's §05.7 append must land before any row reads the cache) plus one refusal of its own: a `split`
 * whose slices would push a pair's union past `ATTN_M2_STAGE`.
 *
 * ---------------------------------------------------------------------------------------------
 * THE MODE PARAMETER: the same body, with and without the two per-touch instruction levers
 * ---------------------------------------------------------------------------------------------
 * D22 queued the re-test of M = 2 against a touch bill that is now 45 slots instead of 116, and the
 * measured tile-form gain from making it so (`prototype/attn_valu_slim/results.md` §4 priced it, §7.1
 * measured it at 1.52-1.57x). `attn_m2_body` is therefore a template on MODE, whose
 * bits are `VSLIM_DOT` and `VSLIM_FOLD` — the SAME vocabulary `prototype/attn_valu_slim` uses for the
 * tile form — and two of them are instantiated:
 *
 *   `attn_m2_wave`       MODE 0. The body exactly as it was measured at 2.5402 ms, unchanged.
 *   `attn_m2_slim_wave`  MODE `VSLIM_DOT|VSLIM_FOLD`. Same body, two spellings replaced.
 *
 * WHY THE HELPERS ARE INCLUDED AND NOT TRANSCRIBED. `attn_slim_kernels.h` (the engine's copy) and
 * `attn_valu_slim_kernels.h` (the prototype's) each already spell the lever helpers, and a third copy
 * in this file would be the §20.1 failure in a new place (`prototype/attn_decode/results.md`: two
 * self-consistent spellings of the same dot, joined by a branch, each agreeing only with itself). So this
 * file includes
 * `../attn_valu_slim/attn_valu_slim_kernels.h` and calls `vslim_dot<ND>` / `vslim_fold<MODE>` /
 * `vslim_permute_addr` directly. Those three are the same code as the engine's `attn_slim_dot` /
 * `attn_slim_fold` / `attn_slim_perm_addr`, verified identical apart from the names and comment tails
 * (and the fold's parameter name), and `m2_model.py selftest` requires this body to name them — so
 * "character for character" is not a transcription claim here, there is no transcription.
 * Association is load-bearing (AGENTS.md, the gdn_collapse row: a hoisted constant moved 93,522 of
 * 131,072 elements while leaving the state bit-exact, and a CPU mirror cannot see either the
 * contraction or the association), which is exactly why a shared spelling beats a copied one.
 *
 * HOW m2's touch differs from the tile form's, and what that did to the levers — stated because the
 * two are NOT the same shape, and pretending otherwise is how a port silently adapts a gate:
 *   - THE DOT is the same eight BF16 products in the same serial-in-`t` order into one F32 accumulator,
 *     but it is handed a SELECTED pointer: a union lane that is this row's SELF lane comes from
 *     `s_kself`, everything else from `s_k` (constraint 4 above). `vslim_dot` reads its operand as one
 *     `uint4`, so the select stays and becomes one `v_cndmask` on the LDS offset in front of the one
 *     wide load — the tile form has no such select and therefore one fewer addressing op. Both stage
 *     bases are 16-byte aligned by construction (`(slot or m) * DH + lane * 8` elements = a multiple of
 *     16 bytes), which is what lets the wide load be legal at all.
 *   - THE FOLD is the same 5-step `warp_sum` over one warp holding 32 lanes x 8 dims, once per touch.
 *     Nothing about M = 2 changes its shape, so the lever applies verbatim.
 *   - WHAT NEITHER LEVER TOUCHES, and is m2-only: the per-element pair-walk guard
 *     (`if (j < g.blo || j >= g.bhi) continue;`), which is most of the +8 SALU/touch this form pays over the
 *     tile form, and the union residue. Both stay.
 *
 * Control row, same discipline as the tile-side port: MODE 0 IS the shipped `attn_m2_wave` (one body,
 * so nothing can drift between a copy and the original), and its census row must stay
 * 124 slots / 55 VALU / 14 LDS / 55 SALU per touch with VGPR 50 / SGPR 64 / LDS 22,656 B — the
 * recorded pre-port figure. `attn_m2_slim_wave` should land near the tile form's 45/28/7/10 plus the
 * guard. Reproduce with:
 *
 *     .venv/bin/python prototype/attn_valu_slim/isa_census.py \
 *         --src "$PWD/prototype/attn_m2/attn_m2.hip.cpp" \
 *         --kernels attn_m2_wave,attn_m2_slim_wave,attn_tile_wave,attn_slim1_wave
 *
 * (the `--src` must be absolute: the census compiles inside a scratch directory).
 */
#pragma once

#include "../attn_decode/attn_decode_kernels.h"
#include "../attn_valu_slim/attn_valu_slim_kernels.h"   // the lever helpers, shared not copied (see above)

// ---- geometry constants ---------------------------------------------------------------------
#define ATTN_M2_M 2                  // query rows per K/V walk. 2 is the funded step; M = 4 and M = 8
                                     // are a byte model in `m2_model.py traffic`, not a kernel here.
#define ATTN_M2_G GQA                // 4 query heads of one KV group per block (the shipped lever)
#define ATTN_M2_THREADS 256          // ATTN_M2_M * ATTN_M2_G warps: ONE WARP PER (row, head) pair
#define ATTN_M2_L 8                  // K lane rows staged per pass-1 sub-block (= warps per block)
#define ATTN_M2_MAXSPLIT 8           // this prototype's OWN cap, not the engine's `kAttnMaxSplit`
                                     // (raised to 9 for `wide4096f`); 2.2's numbers were taken at 8.
#define ATTN_M2_STAGE (MAXSC + ATTN_M2_MAXSPLIT)   // worst union of two adjacent rows' slices, lanes
#define ATTN_M2_VSLOT (TILE + ATTN_M2_MAXSPLIT)    // worst union of the pair's two CURRENT tiles
#define ATTN_M2_PAIR (ATTN_M2_M * ATTN_M2_G)       // (row, head) pairs one block holds = warps

constexpr int kM2Warps = ATTN_M2_THREADS / 32;
constexpr int kM2ScBytes = ATTN_M2_M * ATTN_M2_G * ATTN_M2_STAGE * 2;
constexpr int kM2KBytes = ATTN_M2_L * DH * 2;
constexpr int kM2VBytes = ATTN_M2_VSLOT * DH * 2;
constexpr int kM2SelfBytes = 2 * ATTN_M2_M * DH * 2;      // s_kself + s_vself
constexpr int kM2LdsBytes = kM2ScBytes + kM2KBytes + kM2VBytes + kM2SelfBytes;

// Injections this form adds. The shared `enum Neg` belongs to `prototype/attn_decode` and is not
// edited from here; these are further values of the same `CoreArgs::neg` field, disjoint from it.
enum NegM2 {
    NEG_M2_MASKWIDE = 100,   // a row attends the pair's union: the shared walk ate the mask
    NEG_M2_PARTROW = 101,    // the split-partial slot drops the pair-row term: the two rows alias
    NEG_M2_PARTALIAS = 102,  // §19's bug, re-injected: slot keyed by head-in-group, no KV-group term
};

static_assert(kM2Warps == ATTN_M2_PAIR, "one warp per (row, head-in-group) pair: ATTN_M2_THREADS is a "
                                        "literal so `m2_model.py` can read it, and this ties it to M*G");
static_assert(ATTN_M2_M == 2, "this kernel implements a PAIR: `m2_pair_at` folds exactly two rows, and "
                              "M = 4/8 is a byte model in `m2_model.py traffic`, not a deeper tile. "
                              "Raising it means writing the fold out, sizing the stages again, and "
                              "re-deriving the union bound — see the funding doc's M table");
static_assert(ATTN_M2_L >= TILE, "pass 1 stages a sub-block of K and pass 2 a streaming tile of V; an "
                                 "LDS over-run does not fault on this part, it silently corrupts the "
                                 "neighbour (trap: attn_decode results.md §13/§14)");
static_assert(ATTN_M2_VSLOT > TILE, "the union of two rows' tile k is wider than a tile by up to "
                                    "split - 1; VSLOT == TILE would be an off-by-one waiting for a "
                                    "split of 2");
static_assert(ATTN_M2_STAGE == MAXSC + ATTN_M2_MAXSPLIT, "the score staging bounds the union of a "
                                                         "pair's slices; see the derivation above");
static_assert(kM2LdsBytes <= 65536, "a block's own LDS must fit one CU");

// One row's geometry inside a block: its own slice of its own row, and whether this block serves it.
// The two shipped rules this form depends on are spelled ONCE, just below, in `__device__ __host__`
// form; the one-pass/streaming boundary and the slice width are the two things M = 2 is most likely to
// get wrong, so there is exactly one spelling of each inside this kernel (§20.1 is the bill for two).
struct M2Row {
    int row;          // global query row index
    int pos;          // that row's position (a.pos + row): its self lane
    int n;            // lanes it may attend — §06.3's causal prefix for THAT row
    int s_row;        // blocks that row promises (the shipped rule)
    int span;         // lanes per block for that row (the shipped rule)
    int blo, bhi;     // this block's slice of that row; empty unless active
    int chunks;       // pass-2 work units (tiles) this row runs in this block
    bool exists;      // row < nrows
    bool active;      // exists, this block is within its s_row, and the slice is non-empty
    bool one_pass;    // the shipped one-pass condition, per row
    bool self_here;   // this block's union covers this row's self lane
};

// The pair's shared walk: the union of the two rows' slices, and the pass-2 iteration count. Every
// field here is a min/max over the pair, which is what makes it legal to bound a barrier with it.
struct M2Pair {
    int u_lo, u_hi, union_n;   // shared K walk; union-relative lane 0 is `u_lo`
    int kmax;                  // pass-2 iterations = the uniform barrier count
    bool any;
    bool too_wide;             // a union or slice past the staging: refuse, do not overrun
};

// The two shipped rules, transcribed character for character from
// `attn_decode_kernels.h::lanes_of` / `::blocks_for_row`, injections included, so a shared injection
// moves this form exactly where it moves the shipped one. Spelled rather than called because the
// shipped pair are `__device__`-only in a frozen header while this file's census
// (`attn_m2_pair_union`) and launcher refusal (`attn_m2_ok`) must run on the host. The tie to the
// shipped spelling is not a hope: `m2_model.py selftest` requires this form's per-row lane cut to equal
// `attn_tile_model.row_bounds`' cut block for block, and that mirror is device-witnessed against the
// pins (`prototype/attn_decode/results.md` §20, §21). Two self-consistent layouts inside ONE kernel
// joined by a branch is the §20.1 failure; one spelling per file with a CPU gate across the boundary is
// the opposite of it.
__device__ __host__ inline int m2_lanes_of(const CoreArgs& a, int row) {
    if (a.neg == NEG_READPAST) return a.cap;
    if (a.neg == NEG_DROPCUR) return a.pos + row;
    return a.lvalid + row;
}

__device__ __host__ inline int m2_blocks_for_row(const CoreArgs& a, int row) {
    if (a.full && m2_lanes_of(a, row) <= MAXSC) return 1;
    return a.split > 0 ? a.split : 1;
}

// One row's geometry in lane block `y`. Returns by value and takes no array, because the device body
// must never index a local array with a runtime index: `struct { M2Row r[2]; }` on the stack read as
// `r[m]` is local memory, and the first version of this kernel measured ScratchSize 96 B/lane — a
// private-memory stream in the middle of a byte-bound kernel. Values, not slots.
__device__ __host__ inline M2Row m2_row_at(const CoreArgs& a, int row, int y, int tile) {
    M2Row g{};
    const int nrows = nrows_of(a);
    g.row = row;
    g.exists = row < nrows;
    g.pos = a.pos + row;
    g.n = g.exists ? m2_lanes_of(a, row) : 0;
    g.s_row = g.exists ? m2_blocks_for_row(a, row) : 0;
    g.span = g.s_row > 0 ? (g.n + g.s_row - 1) / g.s_row : 0;
    g.one_pass = g.exists && g.s_row == 1 && a.full != 0;
    const int blo = g.span * y;
    const int hi = g.span > 0 ? (g.n < blo + g.span ? g.n : blo + g.span) : 0;
    g.blo = blo;
    g.bhi = hi;
    g.active = g.exists && g.s_row > 0 && hi > blo;
    g.chunks = (hi - blo + tile - 1) / tile;
    g.self_here = false;
    return g;
}

// Fold one row into the pair's shared walk. Called once per row of the pair, unrolled at compile time
// by the callers, so nothing here is indexed.
__device__ __host__ inline void m2_fold(const M2Row& g, M2Pair* p) {
    if (g.active && g.bhi - g.blo > MAXSC) p->too_wide = true;   // a slice past the score staging
    if (!g.active) return;
    if (!p->any) { p->u_lo = g.blo; p->u_hi = g.bhi; p->kmax = g.chunks; p->any = true; return; }
    if (g.blo < p->u_lo) p->u_lo = g.blo;
    if (g.bhi > p->u_hi) p->u_hi = g.bhi;
    if (g.chunks > p->kmax) p->kmax = g.chunks;
}

// `y` is the launch's lane-block index (`blockIdx.y`). It is an argument, not a `blockIdx` read, so the
// host and the CPU mirror can price the walk a launch will perform — and be checked against the kernel.
// Takes the two rows by pointer rather than as an array: this function is compiled into the device body
// as well, where an array would mean local memory.
__device__ __host__ inline void m2_pair_at(const CoreArgs& a, int rt, int y, M2Row* p0, M2Row* p1,
                                           M2Pair* p) {
    const int tile = (a.tile > 0 && a.tile <= TILE) ? a.tile : TILE;
    p->u_lo = 0x7fffffff; p->u_hi = 0; p->kmax = 0; p->any = false; p->too_wide = false;
    *p0 = m2_row_at(a, ATTN_M2_M * rt + 0, y, tile);
    *p1 = m2_row_at(a, ATTN_M2_M * rt + 1, y, tile);
    m2_fold(*p0, p);
    m2_fold(*p1, p);
    p->union_n = p->any ? p->u_hi - p->u_lo : 0;
    if (p->union_n > ATTN_M2_STAGE) p->too_wide = true;
    p0->self_here = p0->active && p0->pos >= p->u_lo && p0->pos < p->u_hi;
    p1->self_here = p1->active && p1->pos >= p->u_lo && p1->pos < p->u_hi;
}

// Grid shapes and footprint, shared with the launcher and the mirror.
__device__ __host__ inline int attn_m2_rtiles_of(int nrows) { return (nrows + ATTN_M2_M - 1) / ATTN_M2_M; }
__device__ __host__ inline int attn_m2_split(int lanes_last) { return split_for_lanes(lanes_last); }
// The partial footprint is the SHIPPED layout, slot for slot: [row][head][lane block][2 + DH] F32.
__device__ __host__ inline int attn_m2_part_floats(int nrows, int lanes_last) {
    return nrows * N_Q * split_for_lanes(lanes_last) * (DH + 2);
}

// What a pair's shared K walk touches. ONE struct with TWO aggregates, because they are different
// quantities and conflating them is a bug that already happened: `total` is lane rows SUMMED OVER THE
// PAIR'S LANE BLOCKS — a traffic census, bounded by nothing except the context length — while
// `max_block` is the widest SINGLE lane block, which is the quantity `ATTN_M2_STAGE` bounds. The first
// version of `attn_m2_ok` compared `total` to STAGE and therefore refused every plan above ~520 lanes
// (a 1024-lane wave sums four ~340-lane blocks to 1032), which is results.md §2's blocked widths.
// `attn_m2_geometry_guard` below is the guard that makes the confusion fail loudly instead of quietly.
struct M2Union {
    int total = 0;        // lane rows over all lane blocks: TRAFFIC
    int max_block = 0;    // widest single lane block: what STAGE bounds
    int max_slice = 0;    // widest single row slice: what MAXSC bounds
    bool over_read = false;   // a block staged past what the row(s) in it have lanes for
};

// Lane rows one pair's shared K walk fetches. `n_lo` is the LOWER row's LANE COUNT; the pair's second
// row has n_lo + 1, which is what `lanes_of` gives for adjacent query rows (lanes = lvalid + row) — the
// census is about a wave, so it does not pretend to price a pair of arbitrary lane counts it can never
// contain. It is a lane count, not a row index; `m2_launch_scan` is the function that takes a launch and
// walks its real rows. Honest-route geometry only: `neg` is left at NEG_NONE, because the traffic claim
// is a claim about the shipping path.
//
// This is a CLOSED FORM over the same rules `m2_pair_at` applies (it calls it), and
// `m2_model.py selftest` requires the C++ and Python spellings of the census to agree on a census of
// (n, split, full) — a traffic model that retypes the geometry is how §12 booked a 4x that was never
// there (§20.5).
inline M2Union attn_m2_pair_union(int n_lo, int split, int full) {
    M2Union u;
    if (n_lo <= 0) return u;
    CoreArgs a{};
    a.split = split > 0 ? split : 1;
    a.full = full;
    a.lvalid = n_lo;
    a.neg = NEG_NONE;
    a.nrows = ATTN_M2_M;
    a.pos = n_lo - 1;                      // row 0's self lane is its last lane, like a real wave
    a.tile = TILE;
    M2Row p0, p1;
    M2Pair p;
    for (int y = 0; y < a.split + 1; ++y) {
        m2_pair_at(a, 0, y, &p0, &p1, &p);
        if (!p.any) break;
        u.total += p.union_n;
        if (p.union_n > u.max_block) u.max_block = p.union_n;
        const int s0 = p0.active ? p0.bhi - p0.blo : 0;
        const int s1 = p1.active ? p1.bhi - p1.blo : 0;
        const int s = s0 > s1 ? s0 : s1;
        if (s > u.max_slice) u.max_slice = s;
        if (p.u_hi > n_lo + 1) u.over_read = true;   // the pair has n_lo + 1 lanes between its rows
    }
    return u;
}

// The same census for one row alone (a wave with an odd row count ends with one).
inline M2Union attn_m2_row_union(int n, int split, int full) {
    M2Union u;
    if (n <= 0) return u;
    CoreArgs a{};
    a.split = split > 0 ? split : 1;
    a.full = full;
    a.lvalid = n;
    a.neg = NEG_NONE;
    a.nrows = 1;
    a.pos = n - 1;
    a.tile = TILE;
    M2Row p0, p1;
    M2Pair p;
    for (int y = 0; y < a.split + 1; ++y) {
        m2_pair_at(a, 0, y, &p0, &p1, &p);
        if (!p.any) break;
        u.total += p.union_n;
        if (p.union_n > u.max_block) u.max_block = p.union_n;
        const int s0 = p0.active ? p0.bhi - p0.blo : 0;
        if (s0 > u.max_slice) u.max_slice = s0;
        if (p.union_n > n) u.over_read = true;        // a lone row must never over-read
    }
    return u;
}

// What one LAUNCH would actually do, walked over every pair and every lane block it can contain. The
// launcher refuses on a prediction built from `lanes_last` alone; this is the ground truth that
// prediction is checked against, and it is why the guard can tell a correct refusal from a phantom one.
struct M2Launch {
    int max_block = 0;        // widest single union the kernel will stage
    int max_slice = 0;        // widest single row slice
    int total_union = 0;      // traffic census, summed over blocks
    int blocks = 0;
    bool too_wide = false;    // a union past STAGE or a slice past MAXSC, as `m2_pair_at` computes it
};

inline M2Launch m2_launch_scan(int lvalid, int pos, int nrows, int split, int full) {
    CoreArgs a{};
    a.lvalid = lvalid; a.pos = pos; a.nrows = nrows;
    a.split = split > 0 ? split : 1; a.full = full; a.neg = NEG_NONE; a.tile = TILE;
    M2Launch L;
    for (int rt = 0; rt < attn_m2_rtiles_of(nrows); ++rt) {
        for (int y = 0; y < a.split + 1; ++y) {
            M2Row r0, r1; M2Pair p;
            m2_pair_at(a, rt, y, &r0, &r1, &p);
            if (!p.any) break;
            ++L.blocks;
            if (p.too_wide) L.too_wide = true;
            if (p.union_n > L.max_block) L.max_block = p.union_n;
            L.total_union += p.union_n;
            const int s0 = r0.active ? r0.bhi - r0.blo : 0;
            const int s1 = r1.active ? r1.bhi - r1.blo : 0;
            const int s = s0 > s1 ? s0 : s1;
            if (s > L.max_slice) L.max_slice = s;
        }
    }
    return L;
}

// What this form may not be asked for, in one place so the launcher and the gate refuse the same list.
inline bool attn_m2_ok(int score_f32, int fused, int warp_size, int split, int lanes_last,
                       const char** why) {
    if (warp_size != 32) { *why = "one warp covers a 256-wide head as 32 lanes x 8 dims"; return false; }
    if (score_f32) { *why = "the score staging is BF16, exact only for the pinned BF16 score"; return false; }
    if (fused) { *why = "a wave's §05.7 append must finish before any row reads the cache"; return false; }
    if (split < attn_m2_split(lanes_last)) {
        *why = "a row's slice would exceed the MAXSC staging the score buffer reserves (raise --split)";
        return false;
    }
    // The pair-union bound at the widest pair the launch can contain: rows (lanes_last - 1, lanes_last).
    // With the split check above this is implied (span_max <= MAXSC, union <= span_max + split - 1 =
    // ATTN_M2_STAGE); it is spelled out because an implied bound is an untested one — and because it was
    // spelled WRONG first (a `total` summed over blocks against a per-block bound), which is what
    // `attn_m2_geometry_guard` now checks instead of a comment. `max_block`, never `total`. Both `full`
    // spellings are priced because this predicate does not take `full`, and a refusal must hold for the
    // launch the caller will actually make whichever one that is.
    if (lanes_last >= 2) {
        const M2Union u0 = attn_m2_pair_union(lanes_last - 1, split, 0);
        const M2Union u1 = attn_m2_pair_union(lanes_last - 1, split, 1);
        const int worst_block = u0.max_block > u1.max_block ? u0.max_block : u1.max_block;
        const int worst_slice = u0.max_slice > u1.max_slice ? u0.max_slice : u1.max_slice;
        if (worst_block > ATTN_M2_STAGE) {
            *why = "the union of a pair's SLICE PAIR in one lane block exceeds ATTN_M2_STAGE";
            return false;
        }
        if (worst_slice > MAXSC) {   // belt and braces on the split check above
            *why = "a row's slice in one lane block exceeds MAXSC";
            return false;
        }
    }
    return true;
}

// The guard DEFECT 1 needed, and the reason it is a guard rather than a patch. `attn_m2_ok` refuses on a
// prediction made from `lanes_last` alone. Its first prediction used the wrong aggregate and refused
// four of the eight funded plans, including both widths the 13.6/194.3 ms prize is booked at, while
// every plan under ~520 lanes passed by luck — `pair-just-above` sat at exactly 520. "The refusal is
// implied by the split check" is a note, not a forcing, so this walks the space and asserts:
//
//   1. BOUND     — no geometry the kernel can be launched on stages a union wider than ATTN_M2_STAGE or
//                  a row slice wider than MAXSC, as the KERNEL computes it (`m2_launch_scan`);
//   2. NO PHANTOM — the launcher never refuses a launch the kernel would run happily (the §2 bug);
//   3. NO LEAK    — the launcher never accepts a launch the kernel would have to poison;
//   4. REAL PLANS — the plan geometries this op actually runs, listed in `M2_REAL`, are accepted.
//
// Host-only in use (no HIP call), so it can run in a minute on any box: `./attn_m2 --geom-guard`.
// Returns the number of failures.
struct M2RealCase { const char* name; int lanes_last; int split; int nrows; };

// The geometries of `run.py`'s plans and the two widths the route is funded for. Kept here, next to the
// guard that reads it, so a plan that starts being refused fails the guard rather than the plan run.
inline const M2RealCase* attn_m2_real_cases(int* n_out) {
    static const M2RealCase REAL[] = {
        {"pair-at-maxsc",     515, 2, 4},   // one-pass/streaming straddle: 512 and 513 lanes
        {"maxsc-minus-1",     514, 2, 4},
        {"pair-just-above",   520, 2, 8},   // sat at exactly the old bound: passed by luck
        {"straddles-block-2", 1027, 3, 4},  // refused before the fix
        {"straddles-block-3", 1539, 4, 4},  // refused before the fix
        {"pair-widest-1024",  1024, 2, 4},  // refused before the fix
        {"tail-hf-1032",      1032, 3, 1},  // the real HF row: refused before the fix
        {"prefill-1024",      1024, 2, 8},  // where the doc books 13.6 ms
        {"prefill-2048",      2048, 4, 8},
        {"prefill-4096",      4096, 8, 8},  // where it books 194.6 ms
        {"decode-640",         640, 2, 1},
        {"decode-2049",       2049, 5, 1},   // 2049 lanes needs ceil(2049/512) = 5 blocks, not 4
        // The four waves `run.py`'s TIMING_SPECS writes — the ones above the bandwidth plateau, i.e. the
        // plans the funded claim will be measured on (results.md §7). Listed here so the guard proves
        // acceptance for every plan the driver emits, in seconds, instead of a device window finding out.
        {"wave64-at-1024",    1084, 3, 64},
        {"wave64-at-2048",    2108, 5, 64},
        {"wave64-at-4096",    4096, 8, 64},
        {"wave16-at-4096",    4096, 8, 16},
    };
    *n_out = (int)(sizeof(REAL) / sizeof(REAL[0]));
    return REAL;
}

inline int attn_m2_geometry_guard(int verbose) {
    int fails = 0, checked = 0, refused = 0, phantom = 0, leaked = 0;
    int worst_block = 0, worst_slice = 0;
    int wb_lanes = 0, wb_split = 0, wb_full = -1;      // the geometry that produced worst_block

    // The lane counts worth walking: every short one, a dense band over the one-pass/streaming boundary,
    // every multiple of MAXSC +/- 2 (where a row changes lane block), a coarse sweep to the engine's lane
    // ceiling MAXSC x MAXSPLIT, and the ceiling itself.
    int lanes[512];
    int nl = 0;
    auto push = [&](int n) {              // deduplicated: the Python mirror walks the SAME set, and two
        for (int i = 0; i < nl; ++i)      // guards that count different populations cannot be compared
            if (lanes[i] == n) return;
        if (n >= 1 && nl < 512) lanes[nl++] = n;
    };
    for (int n = 1; n <= 64; ++n) push(n);
    for (int n = MAXSC - 6; n <= MAXSC + 10; ++n) push(n);
    for (int k = 2; k <= ATTN_M2_MAXSPLIT; ++k)
        for (int d = -2; d <= 2; ++d) push(MAXSC * k + d);
    for (int n = 65; n <= MAXSC * ATTN_M2_MAXSPLIT; n += 127) push(n);
    push(MAXSC * ATTN_M2_MAXSPLIT);

    static const int NROWS[] = {1, 2, 3, 4, 8, 17};
    for (int li = 0; li < nl; ++li) {
        const int lanes_last = lanes[li];
        for (int split = 1; split <= ATTN_M2_MAXSPLIT; ++split) {
            for (int full = 0; full <= 1; ++full) {
                for (int ni = 0; ni < (int)(sizeof(NROWS) / sizeof(NROWS[0])); ++ni) {
                    const int nrows = NROWS[ni];
                    if (lanes_last < nrows) continue;
                    const int lvalid = lanes_last - nrows + 1;
                    const int pos = lvalid - 1;
                    const M2Launch L = m2_launch_scan(lvalid, pos, nrows, split, full);
                    ++checked;
                    if (L.total_union <= 0) continue;
                    // A grid shallower than `split_for_lanes` is not a geometry this form may be
                    // launched on — the split check refuses it — so the bound is not claimed for it. What
                    // IS claimed, and checked below, is that the refusal that fires is the SPLIT one, so a
                    // union refusal can never again be the accidental answer for a too-shallow grid.
                    const bool legal_split = split >= attn_m2_split(lanes_last);
                    if (legal_split) {
                        // 1. BOUND, as the kernel computes it, not as the launcher predicts.
                        if (L.max_block > worst_block) {
                            worst_block = L.max_block;
                            wb_lanes = lanes_last; wb_split = split; wb_full = full;
                        }
                        if (L.max_slice > worst_slice) worst_slice = L.max_slice;
                        if (L.max_block > ATTN_M2_STAGE) {
                            if (verbose)
                                printf("GUARD FAIL bound: lanes_last=%d split=%d full=%d nrows=%d stages "
                                       "union %d > ATTN_M2_STAGE %d\n", lanes_last, split, full, nrows,
                                       L.max_block, ATTN_M2_STAGE);
                            ++fails;
                        }
                        if (L.max_slice > MAXSC) {
                            if (verbose)
                                printf("GUARD FAIL bound: lanes_last=%d split=%d full=%d nrows=%d slices "
                                       "%d > MAXSC %d\n", lanes_last, split, full, nrows, L.max_slice, MAXSC);
                            ++fails;
                        }
                    }
                    // 2 and 3: the launcher's verdict must agree with the kernel's, in both directions.
                    const char* why = "";
                    const bool ok = attn_m2_ok(0, 0, 32, split, lanes_last, &why);
                    if (!legal_split) {
                        if (ok) {
                            if (verbose)
                                printf("GUARD FAIL leak: lanes_last=%d split=%d accepted although "
                                       "split_for_lanes would need %d\n", lanes_last, split,
                                       attn_m2_split(lanes_last));
                            ++fails; ++leaked;
                        } else if (std::string(why).find("--split") == std::string::npos) {
                            if (verbose)
                                printf("GUARD FAIL attribution: lanes_last=%d split=%d refused for "
                                       "\"%s\", not the shallow-grid reason\n", lanes_last, split, why);
                            ++fails;
                        }
                        continue;
                    }
                    if (ok && L.too_wide) {
                        if (verbose)
                            printf("GUARD FAIL leak: lanes_last=%d split=%d full=%d nrows=%d accepted "
                                   "but the kernel would poison (max_block=%d max_slice=%d)\n",
                                   lanes_last, split, full, nrows, L.max_block, L.max_slice);
                        ++fails; ++leaked;
                    }
                    if (!ok && !L.too_wide) {
                        if (verbose)
                            printf("GUARD FAIL phantom refusal: lanes_last=%d split=%d full=%d "
                                   "nrows=%d refused for \"%s\" but max_block=%d max_slice=%d fit\n",
                                   lanes_last, split, full, nrows, why, L.max_block, L.max_slice);
                        ++fails; ++phantom;
                    }
                    if (!ok) ++refused;
                }
            }
        }
    }

    // 4. The plans this op runs must be accepted, and must not be too wide.
    int nreal = 0;
    const M2RealCase* real = attn_m2_real_cases(&nreal);
    for (int i = 0; i < nreal; ++i) {
        const M2RealCase& c = real[i];
        const char* why = "";
        const bool ok = attn_m2_ok(0, 0, 32, c.split, c.lanes_last, &why);
        const M2Launch L = m2_launch_scan(c.lanes_last - c.nrows + 1, c.lanes_last - c.nrows,
                                          c.nrows, c.split, 1);
        if (!ok || L.too_wide) {
            printf("GUARD FAIL real plan %s (lanes_last=%d split=%d nrows=%d): %s%s max_block=%d\n",
                   c.name, c.lanes_last, c.split, c.nrows, ok ? "kernel would poison" : why,
                   ok ? "" : "", L.max_block);
            ++fails;
        } else if (verbose) {
            printf("  [ok  ] %-16s lanes_last=%4d split=%d nrows=%2d | max block union %3d of %d, "
                   "slice %3d of %d, %d blocks\n", c.name, c.lanes_last, c.split, c.nrows,
                   L.max_block, ATTN_M2_STAGE, L.max_slice, MAXSC, L.blocks);
        }
    }

    printf("geom-guard: %d geometries walked (split 1..%d, full 0/1, nrows 1/2/3/4/8/17) | "
           "worst single-block union %d of ATTN_M2_STAGE %d over LEGAL grids (at lanes_last=%d split=%d "
           "full=%d) | worst row slice %d of MAXSC %d | refused %d (phantom %d, leak %d) | "
           "%d real plan geometries\n",
           checked, ATTN_M2_MAXSPLIT, worst_block, ATTN_M2_STAGE, wb_lanes, wb_split, wb_full,
           worst_slice, MAXSC, refused, phantom, leaked, nreal);
    printf("geom-guard: %s\n", fails ? "FAIL" : "PASS");
    return fails;
}

// Loud poison, so a bound a launcher failed to enforce cannot become a silent numerics diff.
__device__ __forceinline__ void m2_panic(const CoreArgs& a, int row) {
    if (row >= 0 && row < nrows_of(a)) {
        a.aout[(size_t)ROW_H * row + threadIdx.x] = fb(nanf("m2stage"));
    }
}

// MODE 0 is the body exactly as it was before the touch levers, and MODE `VSLIM_DOT|VSLIM_FOLD` is the
// same body with the score dot and the warp fold re-spelled by the shared helpers. One body, not a
// copy: the two cannot then disagree about anything except the two spellings.
template <int MODE>
__device__ __forceinline__ void attn_m2_body(const CoreArgs a) {
    __shared__ hip_bfloat16 s_sc[ATTN_M2_M][ATTN_M2_G][ATTN_M2_STAGE];   // exact-BF16 scores
    __shared__ hip_bfloat16 s_k[ATTN_M2_L * DH];                         // shared K rows (pass 1)
    __shared__ hip_bfloat16 s_v[ATTN_M2_VSLOT * DH];                     // shared V rows (pass 2)
    __shared__ hip_bfloat16 s_kself[ATTN_M2_M * DH];                     // this row's §05.7 append
    __shared__ hip_bfloat16 s_vself[ATTN_M2_M * DH];

    const int rt = gridDim.z - 1 - blockIdx.z;      // widest pair first: §2's dispatch tail
    const int kv_read = tile_kv_of(blockIdx.x, a.neg);   // group whose K/V this block READS
    const int kv_write = blockIdx.x;                // group whose lanes this wave appends
    const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;           // == ATTN_M2_PAIR, by the static_assert
    const int m = w / ATTN_M2_G;                    // this warp's row within the pair
    const int h = w % ATTN_M2_G;                    // this warp's head within that group
    const int head = ATTN_M2_G * kv_write + h;      // FULL head index: slots, outputs, gates
    const int tile = (a.tile > 0 && a.tile <= TILE) ? a.tile : TILE;
    const int nd = (a.neg == NEG_DIMHALF) ? 4 : 8;  // of the 8 BF16 a lane owns
    // LEVER 2, part 1 — the four folded permute addresses (deltas 8, 4, 2, 1). Functions of `lane`
    // alone, so they are loop-invariant across every touch this block makes; only the slim MODE reads
    // them. Declared outside the `if constexpr` so the two MODEs differ in exactly the two spellings and
    // nothing else — and the control row of the census is what shows the unused slots are free here
    // (VGPR 50 and LDS 22,656 B on both m2 forms), not a sentence saying so.
    unsigned pa[4] = {0, 0, 0, 0};
    if constexpr (MODE != 0) {
        for (int i = 0; i < 4; ++i) pa[i] = vslim_permute_addr(lane, 8 >> i);   // 8, 4, 2, 1
    }

    M2Row p0, p1;
    M2Pair b;
    m2_pair_at(a, rt, blockIdx.y, &p0, &p1, &b);
    if (b.too_wide) {                               // a launcher that refused nothing gets poison
        m2_panic(a, p0.row);
        m2_panic(a, p1.row);
        return;
    }
    if (!b.any) return;                             // before any barrier: block-uniform

    // This warp's row — selected by a conditional MOVE, never by indexing an array of them: `b.r[m]`
    // on a stack struct is local memory, and the first version of this kernel measured a 96 B/lane
    // private scratch stream in a kernel whose whole point is bytes.
    M2Row g = (m == 0) ? p0 : p1;
    if (a.neg == NEG_M2_MASKWIDE && g.active) {     // the shared walk ate this row's mask
        g.blo = b.u_lo;
        g.bhi = b.u_hi;
        g.chunks = (g.bhi - g.blo + tile - 1) / tile;
    }
    // The pass-2 iteration count must stay block-uniform even under that injection.
    const int kmax = a.neg == NEG_M2_MASKWIDE
                         ? (b.union_n + tile - 1) / tile
                         : b.kmax;

    // ---- the pair's self-lane stage: lane `pos + row` comes from the STAGE, per row, exactly as
    // `attn_core` and `attn_tile_wave` read it. Filled once per block, for the rows this block owns.
    if (p0.self_here || p1.self_here) {
        __syncthreads();                            // nobody may read these before they land
        for (int d = threadIdx.x; d < DH; d += blockDim.x) {
            if (p0.self_here) {
                s_kself[d] = a.krope[(size_t)ROW_KV * p0.row + (size_t)kv_write * DH + d];
                s_vself[d] = a.vg[(size_t)ROW_KV * p0.row + (size_t)kv_write * DH + d];
            }
            if (p1.self_here) {
                s_kself[DH + d] = a.krope[(size_t)ROW_KV * p1.row + (size_t)kv_write * DH + d];
                s_vself[DH + d] = a.vg[(size_t)ROW_KV * p1.row + (size_t)kv_write * DH + d];
            }
        }
        __syncthreads();
    }

    // ---- this warp's query, in registers: lane owns dims lane*8..lane*8+7 — the same 512-byte read
    // a K row needs, and the same chunk `attn_core` scores with.
    float q[8];
    for (int u = 0; u < 8; ++u) q[u] = 0.f;
    if (g.active) {
        const uint4 qu = *reinterpret_cast<const uint4*>(
            a.qrope + (size_t)ROW_H * g.row + (size_t)head * DH + lane * 8);
        const hip_bfloat16* qq = reinterpret_cast<const hip_bfloat16*>(&qu);
        for (int u = 0; u < 8; ++u) q[u] = bf(qq[u]);
    }

    // ---- pass 1: ONE K walk over the union of the pair's slices. `slot` is a sub-block slot, never a
    // slice-absolute lane: indexing the stage by the lane is §13/§14's silent LDS over-run, and
    // `m2_model.py lds` reads this spelling instead of trusting this sentence.
    for (int s_lo = 0; s_lo < b.union_n; s_lo += ATTN_M2_L) {
        const int s_hi = (b.union_n < s_lo + ATTN_M2_L) ? b.union_n : s_lo + ATTN_M2_L;
        for (int rr = s_lo + w; rr < s_hi; rr += nw) {
            const int slot = rr - s_lo;
            const int j = b.u_lo + rr;
            const hip_bfloat16* ksrc = a.kcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
            *reinterpret_cast<uint4*>(&s_k[slot * DH + lane * 8]) =
                *reinterpret_cast<const uint4*>(ksrc + lane * 8);
        }
        __syncthreads();                                   // K sub-block published
        if (g.active) {
            // One (row, lane, head) touch = one dot + one fold + one BF16 round — THE SAME unit the
            // tile-side census counts, so the two forms' per-touch rows are comparable. m2's touch
            // differs from the tile form's in two ways and neither is a lever: the walk is over the
            // PAIR's union, so each warp skips the lanes that are not its own row's (the +8 SALU/touch
            // the census shows), and a union lane that is this row's SELF lane is scored from `s_kself`
            // rather than `s_k` (constraint 4). Both survive both levers unchanged.
            if constexpr (MODE == 0) {
                for (int rr = s_lo; rr < s_hi; ++rr) {
                    const int j = b.u_lo + rr;
                    if (j < g.blo || j >= g.bhi) continue;  // not this ROW's lane: not this warp's work
                    const hip_bfloat16* k = (j == g.pos)
                                                ? &s_kself[m * DH + lane * 8]
                                                : &s_k[(rr - s_lo) * DH + lane * 8];
                    float dot = 0.f;
                    for (int t = 0; t < nd; ++t) dot += bf(k[t]) * q[t];   // serial in t, §05.5
                    dot = warp_sum(dot);                                  // 5-step butterfly
                    // One BF16 round, then the scale; exact in BF16 because SCALE = 2^-4, which is what
                    // lets the staging be 2 bytes and still read back the identical F32.
                    if (lane == 0) s_sc[m][h][rr] = fb(bf(fb(dot)) * SCALE);
                }
            } else {
                // LEVER 1 (`vslim_dot<ND>`) and LEVER 2 (`vslim_fold<MODE>`), spelled by the shared
                // helpers in `prototype/attn_valu_slim/attn_valu_slim_kernels.h` — not transcribed, so
                // there is no second spelling of the dot in this tree to disagree with the first.
                // The dot's operand is a SELECTED pointer here, which the tile form never pays; that is
                // one `v_cndmask` in front of its one wide load, and it is not removable by a lever.
                auto publish = [&](int rr, float dot) {
                    dot = vslim_fold<MODE>(dot, pa);        // LEVER 2: the guards are hoisted, the
                                                            // permute is one inline ds_bpermute_b32
                    // One BF16 round, then the scale; exact in BF16 because SCALE = 2^-4, which is what
                    // lets the staging be 2 bytes and still read back the identical F32.
                    if (lane == 0) s_sc[m][h][rr] = fb(bf(fb(dot)) * SCALE);
                };
                // LEVER 1: `nd` is warp-uniform (a function of `a.neg`), so the dim-half injection gets
                // its own copy of the loop behind one uniform compare instead of lending this one a
                // runtime trip count. Which is how MODE 0 ends up rolling the dot: eight scalar
                // `ds_load_u16`s, eight `v_movrels_b32_e32`s for the dynamically indexed `q[t]` and
                // eight back edges per touch.
                if (nd == 8) {
                    for (int rr = s_lo; rr < s_hi; ++rr) {
                        const int j = b.u_lo + rr;
                        if (j < g.blo || j >= g.bhi) continue;
                        publish(rr, vslim_dot<8>((j == g.pos) ? &s_kself[m * DH + lane * 8]
                                                              : &s_k[(rr - s_lo) * DH + lane * 8], q));
                    }
                } else {
                    for (int rr = s_lo; rr < s_hi; ++rr) {
                        const int j = b.u_lo + rr;
                        if (j < g.blo || j >= g.bhi) continue;
                        publish(rr, vslim_dot<4>((j == g.pos) ? &s_kself[m * DH + lane * 8]
                                                              : &s_k[(rr - s_lo) * DH + lane * 8], q));
                    }
                }
            }
        }
        __syncthreads();                                   // s_sc published for pass 2
    }

    // ---- pass 2: two private reductions over one shared V walk.
    float mn_run = 0.f, l = 0.f, acc[8];
    for (int u = 0; u < 8; ++u) acc[u] = 0.f;
    if (g.active && g.one_pass) {
        // §05.5's reference form for THIS row: max, then sum-of-exp, then the BF16-rounded normalised
        // weight and P·V, each serial in index order over the row's own slice.
        const int ns = g.bhi - g.blo, off = g.blo - b.u_lo;
        for (int t = 0; t < ns; ++t) mn_run = fmaxf(mn_run, bf(s_sc[m][h][off + t]));
        for (int t = 0; t < ns; ++t) l += expf(bf(s_sc[m][h][off + t]) - mn_run);
    }

    for (int k = 0; k < kmax; ++k) {
        // The union of the pair's CURRENT work units. A row's chunk k is anchored at its own `blo`, so
        // for a streaming row chunk k IS tile k of `attn_core`'s loop, one for one; for a one-pass row
        // it is a staging granularity only — its weights come from the whole row.
        int c_lo = 0x7fffffff, c_hi = 0;
        // Written twice, once per row of a pair of fixed size 2, rather than as a loop over an array:
        // same arithmetic, and no local-memory slot for the pair.
        if (p0.active && k < p0.chunks) {
            const int s0 = p0.blo + k * tile;
            const int s1 = (p0.bhi < s0 + tile) ? p0.bhi : s0 + tile;
            if (s0 < c_lo) c_lo = s0;
            if (s1 > c_hi) c_hi = s1;
        }
        if (p1.active && k < p1.chunks) {
            const int s0 = p1.blo + k * tile;
            const int s1 = (p1.bhi < s0 + tile) ? p1.bhi : s0 + tile;
            if (s0 < c_lo) c_lo = s0;
            if (s1 > c_hi) c_hi = s1;
        }
        if (c_hi > c_lo) {
            for (int j = c_lo + w; j < c_hi; j += nw) {
                const int slot = j - c_lo;                 // a SLOT in `s_v`, never a lane index
                // The shared stage comes from the CACHE for every lane, including a partner row's self
                // lane: that is what the shipped form reads there, and it is what keeps
                // NEG_SLOTK/NEG_SLOTV visible through the read side.
                const hip_bfloat16* vsrc = a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
                *reinterpret_cast<uint4*>(&s_v[slot * DH + lane * 8]) =
                    *reinterpret_cast<const uint4*>(vsrc + lane * 8);
            }
        }
        __syncthreads();                                   // V chunk published
        if (g.active && k < g.chunks) {
            const int c0 = g.blo + k * tile;
            const int c1 = (g.bhi < c0 + tile) ? g.bhi : c0 + tile;
            if (g.one_pass) {
                for (int j = c0; j < c1; ++j) {
                    const float wt = bf(fb(expf(bf(s_sc[m][h][j - b.u_lo]) - mn_run) / l));
                    const hip_bfloat16* v = (j == g.pos)
                                                ? &s_vself[m * DH + lane * 8]
                                                : &s_v[(j - c_lo) * DH + lane * 8];
                    for (int u = 0; u < 8; ++u) acc[u] += wt * bf(v[u]);   // serial in t, §05.5
                }
            } else {
                // §07.3's streaming form with `attn_core`'s EXACT tile algebra over the SAME
                // [blo, bhi) it would be handed, including the `expf(m - mn)` rescale, so this route
                // lands on the bits the shipped form already streams rather than opening a second set.
                float mt = -1e30f;
                for (int j = c0; j < c1; ++j) mt = fmaxf(mt, bf(s_sc[m][h][j - b.u_lo]));
                const float mn = fmaxf(mn_run, mt);
                const float rescale = expf(mn_run - mn);
                float e[TILE], ls = 0.f;
                int nt = 0;
                for (int j = c0; j < c1; ++j) {
                    e[nt] = expf(bf(s_sc[m][h][j - b.u_lo]) - mn);
                    ls += e[nt];
                    ++nt;
                }
                l = l * rescale + ls;
                for (int u = 0; u < 8; ++u) acc[u] *= rescale;
                for (int j = c0; j < c1; ++j) {
                    const hip_bfloat16* v = (j == g.pos)
                                                ? &s_vself[m * DH + lane * 8]
                                                : &s_v[(j - c_lo) * DH + lane * 8];
                    for (int u = 0; u < 8; ++u) acc[u] += e[j - c0] * bf(v[u]);
                }
                mn_run = mn;
            }
        }
        __syncthreads();                                   // before the next stage overwrites
    }

    if (g.one_pass) l = 1.0f;    // the weights were already divided; §07.3's rule is one divide ONCE

    // ---- epilogue: `finish_head`'s arithmetic, and the shipped partial slot expression.
    if (!g.active) return;
    if (g.s_row == 1) {
        hip_bfloat16 o[8], g8[8];
        const uint4 gu = *reinterpret_cast<const uint4*>(
            a.qg + (size_t)ROW_Q * g.row + g_off(head, a.neg) + lane * 8);
        const hip_bfloat16* gg = reinterpret_cast<const hip_bfloat16*>(&gu);
        for (int u = 0; u < 8; ++u) {
            const float av = bf(fb(acc[u] / l));
            o[u] = fb(av);
            const float gv = bf(gg[u]);
            g8[u] = fb(av * bf(fb(1.0f / (1.0f + expf(-gv)))));
        }
        *reinterpret_cast<uint4*>(a.aout + (size_t)ROW_H * g.row + (size_t)head * DH + lane * 8) =
            *reinterpret_cast<uint4*>(o);
        *reinterpret_cast<uint4*>(a.gout + (size_t)ROW_H * g.row + (size_t)head * DH + lane * 8) =
            *reinterpret_cast<uint4*>(g8);
    } else {
        // The shipped slot: ((row*N_Q + head)*split + y) * (DH + 2), keyed by the row and by the FULL
        // head. `NEG_M2_PARTROW` drops the row term (the two rows of a pair then write one slot);
        // `NEG_M2_PARTALIAS` drops the KV-group term (§19's failure). Both are gated with must-catch
        // sets DERIVED from the geometry, and `attn_combine` stays one function for both forms.
        int prow = g.row;
        if (a.neg == NEG_M2_PARTROW) prow = ATTN_M2_M * rt;
        int slot_head = head;
        if (a.neg == NEG_M2_PARTALIAS) slot_head = h;
        float* pp = a.part
                  + (((size_t)prow * N_Q + slot_head) * a.split + blockIdx.y) * (DH + 2);
        if (lane == 0) { pp[0] = mn_run; pp[1] = l; }
        for (int u = 0; u < 8; ++u) pp[2 + lane * 8 + u] = acc[u];
    }
}

// MODE 0 is the form every number in `results.md` §2–§7 was taken on; the slim MODE is D22's re-test.
// Both are compiled unconditionally, so `isa_census.py` prints both rows out of one binary and the
// harness's `--slim` decides only which one the LAUNCHER runs, never which one exists.
__global__ void attn_m2_wave(CoreArgs a) { attn_m2_body<0>(a); }
__global__ void attn_m2_slim_wave(CoreArgs a) { attn_m2_body<VSLIM_DOT | VSLIM_FOLD>(a); }

// What `--slim` means here, in one place so the harness and a reader cannot drift: 0 = the two forms of
// record (ship, m2); 1 = adds `attn_m2_slim_wave`, i.e. both levers. Single-lever attribution lives in
// `prototype/attn_valu_slim --slim 2|3`, which is where the two levers were separated.
inline bool attn_m2_slim_ok(int slim) { return slim == 0 || slim == 1; }
