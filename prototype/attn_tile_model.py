#!/usr/bin/env python3
"""attn_tile_model.py — the CPU mirror and CPU-only gate for Phase 2's tiled attention wave.

WHY THIS FILE EXISTS (and why it is not just a rerun of `run.py`). Phase 2 replaces the wave's
`attn_core` with `attn_tile_wave`, whose whole point is that ONE K row is loaded once and fed to
G x M = 16 score dots (§4/§9 of progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md: the wave reads 25.79 GB of
K/V to do 25.79 GFLOP, 1.0 byte/flop, and the byte count alone prices the line). The contract is
§5's: the five ordered reductions of §05.5 survive verbatim, so the tiled form must agree with the
shipped form BIT FOR BIT on both the one-pass route and §07.3's streaming route — no new band, and
no already-banded row smuggled into the exact route.

A worker may not touch the GPU (AGENTS.md), so this file is the part of that claim that can be made
without one:

    geometry   the launch plan and the byte census of both forms, closed-form (extends
               `wave_model.py`'s traffic arithmetic rather than re-deriving it)
    traffic    the same census MEASURED off the mirror's own reads — every K and V row the mirror
               actually reads is counted, and a read outside a row's causal prefix lands on poison
    compare    the mirror's two forms against each other, bit for bit, and against the pinned dumps
    isa        the compile-only resource census of the new kernel (hipcc -c; no device touched)
    lds        what LDS the K stage's INDEX ARITHMETIC can reach, read off the source — a footprint
               claim, which is not the declared-size claim `isa` prints (see §14 of results.md)
    oob-bound  how many elements a stage that overruns `s_k` could move, against a device report —
               turns "the grid moved 1.1 M elements" into a falsified-or-confirmed diagnosis
    selftest   the mirror's own invariants, including the ones a wrong geometry would break
    negative   the injections, each with a must-catch/must-not-catch set derived from the geometry

The arithmetic mirror reproduces `attn_core`'s reductions in the order the kernel performs them
(`reference.attend`'s `softmax="online"` branch is the same trick for the streaming half), and
`compare` is what makes the mirror worth believing: on every pinned span the mirror's SHIPPED form
must reproduce `attn_out_pre_gate` bit-exactly, because if it does then the same code path run with
the tile geometry is a prediction of the device's bits rather than a guess at them.

Run:
    .venv/bin/python prototype/attn_tile_model.py geometry
    .venv/bin/python prototype/attn_tile_model.py traffic --tokens 1024 --wave-rows 512
    .venv/bin/python prototype/attn_tile_model.py isa
    .venv/bin/python prototype/attn_tile_model.py selftest
    .venv/bin/python prototype/attn_tile_model.py gate                 # CPU-only, pinned fixtures
    .venv/bin/python prototype/attn_tile_model.py gate --negative
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "attn_decode")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import reference as R                                            # noqa: E402
import bf16_ulp                                                  # noqa: E402

KERNEL_H = ROOT / "prototype" / "attn_decode" / "attn_decode_kernels.h"
DRAM_READ_GBS = 237.0        # devices/strix-halo-memory-bandwidth.md, pure read stream
MEASURED_ATTN_MS = 109.2     # frozen ledger progress/2026-10-16/performance-status-2026-10-16.md:951, route-(c) attribution (the baseline)
VECTOR_FP32_TF = 2.93        # devices/strix-halo-compute.md: what the vector path sustains
HAND_GEMM_TF = 0.85          # the engine's own hand GEMMs (the rate a vector kernel really hits)
LINES = {"full": 0.0, "stream": 0.0}   # traffic lines read off the mirror, by route
POISON = float("nan")        # see `CacheDoc`: a leaked mask must poison the output, not shrink it


def kernel_consts(path: Path = KERNEL_H) -> dict:
    """The kernel's own constants, read from the header (a copy here would drift). Covers both
    forms: `MAXSC`/`TILE` are the shipped form's and `ATTN_TILE_*` the tiled form's."""
    raw = dict(re.findall(r"^#define\s+(\w+)\s+(\d+)\s*(?://.*)?$", path.read_text(), re.M))
    vals = {k: int(v) for k, v in raw.items()}
    for name in ("N_Q", "N_KV", "DH", "TILE", "MAXSC", "ATTN_TILE_M", "ATTN_TILE_L",
                 "ATTN_TILE_THREADS"):
        if name not in vals:
            raise SystemExit(f"{path.name}: cannot read {name} — this model's copy is stale")
    vals["SCALE"] = 0.0625
    vals["G"] = vals["N_Q"] // vals["N_KV"]   # the header spells GQA as (N_Q / N_KV)
    vals["M"] = vals["ATTN_TILE_M"]
    vals["warps"] = vals["ATTN_TILE_THREADS"] // 32
    vals["reuse"] = vals["G"] * vals["M"]
    vals["score_staging_bytes"] = vals["G"] * vals["M"] * vals["MAXSC"] * 2   # BF16 scores
    vals["k_stage_bytes"] = vals["ATTN_TILE_L"] * vals["DH"] * 2
    vals["v_stage_bytes"] = vals["ATTN_TILE_L"] * vals["DH"] * 2
    vals["block_lds_bytes"] = (vals["score_staging_bytes"] + vals["k_stage_bytes"]
                               + vals["v_stage_bytes"])
    return vals


C = kernel_consts()
N_Q, N_KV, DH, MAXSC, TILEW = (C["N_Q"], C["N_KV"], C["DH"], C["MAXSC"], C["TILE"])

# The places the two lane stagings index their LDS arrays (the stage write and the use site of each).
# The declarations `s_k[ATTN_TILE_L * DH]` / `s_v[ATTN_TILE_L * DH]` are deliberately NOT among them,
# which is why `+ lane * 8` is what identifies an index site rather than just `s_k[`.
S_K_INDEX = re.compile(r"s_[a-z]+\[([^\]]*?)\s*\+\s*lane \* 8\]")
# A slot index is a sub-block slot iff it subtracts the sub-block's own base (`rr - s_lo`, `rr - base`)
# or goes through a name defined that way. `named_slot` is the one definition the file is allowed to
# have, and `>= TILE` is the static_assert that keeps a streaming tile inside the V stage.
SLOT_DEFS = re.compile(r"const int slot = (\w+) - (\w+);")
V_STAGE_GUARD = re.compile(r"static_assert\(ATTN_TILE_L >= TILE")
# The split-partial slot expressions: `a.part + (((size_t)<row> * <mult> + <head>) * <stride> ...)`.
PART_SITE = re.compile(r"a\.part \+ \(\(\(size_t\)(\w+) \* (\w+) \+ (\w+)\)")
# Which lane-window rule `attn_tile_body` uses, read off its own source (the same anti-rot trick as
# `S_K_INDEX`: a model that retypes the kernel's geometry can disagree with it silently).
TILE_FIXED_WINDOW = re.compile(r"const int blo = y \* MAXSC;")
TILE_PER_ROW_SLICE = re.compile(r"blocks_for_row\(a, row\)")
# How one staged lane row is SHARED, read off the staging loops. Exactly one of two shapes may appear:
#   `for (int p = w; p < pairs; p += nw)`  — every (row, head) pair runs its own stage, so the G heads
#       of a group each fetch the identical K row: G global requests per lane row per block, i.e. no
#       re-use beyond what L1 was going to hand out anyway. This is §12 as shipped (§20).
#   `for (int rr = s_lo + w; rr < s_hi; rr += nw)` — one warp stages one lane for the whole block and
#       every head-warp reads that slot: ONE request per lane row. This is the §20 form, and it is the
#       whole traffic claim.
# Traffic is MULTIPLICITY: what counts is how many staging loops touch a lane row, not how many warps
# read it. A model that assumed the block stage while the kernel had the pair stage books a 4x byte cut
# that never happened — which is precisely the §12 mistake in its own column.
PAIR_STAGED = re.compile(r"for \(int p = w; p < pairs; p \+= nw\)")
BLOCK_STAGED = re.compile(r"for \(int rr = s_lo \+ w; rr < s_hi; rr \+= nw\)")


def stage_copies(src: str) -> dict:
    """Times one block asks global memory for the same K (or V) lane row, and the re-use that leaves."""
    pair, block = bool(PAIR_STAGED.search(src)), bool(BLOCK_STAGED.search(src))
    if pair == block:
        raise SystemExit(f"stage_copies: expected exactly one staging shape, found pair={pair} "
                         f"block={block} — this model's patterns are stale")
    copies = G if pair else 1
    return {"level": "pair" if pair else "block", "copies": copies,
            "reuse": N_Q // (N_KV * copies)}   # shipped fetches each row once per query head
SCALE, G, M = C["SCALE"], C["G"], C["M"]


# ============================================================================================
# BF16-valued F32 arithmetic, in the order the kernel performs it.
#
# Every value carried here is an F32 that is exactly representable in BF16 where §05.5 rounds, so a
# `bf16(x)` call is the kernel's `bf(fb(x))` and a store is `fb(x)`. Sums are `numpy.float32`
# accumulated in the kernel's ORDER, because that order is the whole claim: §05.5's `l` and `acc` are
# serial in the lane index, and a vectorised or reordered sum is a different number.
# `reference.attend` already established that this reproduces the pins bit-for-bit (0 of 98,304
# elements, results.md §2), which is what makes it a witness for the tile form too.
# ============================================================================================
def bf16_round(a: np.ndarray) -> np.ndarray:
    """One BF16 round of a float32 array: round-to-nearest-even on the top 16 bits, infinities and
    NaNs passed through untouched (the naive `+0x7FFF` carry would turn -inf into +0)."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = (u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) & np.uint64(0xFFFF0000)
    out = r.astype(np.uint32).view(np.float32)
    keep = (u & np.uint64(0x7F800000)) == np.uint64(0x7F800000)
    return np.where(keep, np.asarray(a, dtype=np.float32), out).astype(np.float32)


def bf16(x) -> np.float32:
    """One BF16 round of a scalar — the kernel's `bf(fb(x))`, and the same rounding torch does."""
    return np.float32(bf16_round(np.array([x], dtype=np.float32))[0])


def vec16(t: torch.Tensor) -> torch.Tensor:
    """One BF16 round of a whole tensor — `fb(x)` read back as F32."""
    return t.to(torch.bfloat16).to(torch.float32)


def expf(x) -> np.float32:
    return np.float32(np.exp(np.float32(x)))


def half_keep() -> np.ndarray:
    """The dim mask a NEG_DIMHALF dot is equivalent to: keep dims where (d % 8) < 4. The kernel's
    `nd = 4` stops each lane after 4 of its own 8, which is dims 0-3, 8-11, 16-19, … — 128 of the 256
    dims, interleaved, NOT the first 128."""
    keep = np.ones(DH, dtype=np.float32)
    keep[(np.arange(DH) % 8) >= 4] = 0.0
    return keep


def dot256(q: np.ndarray, k: np.ndarray) -> float:
    """One 256-term score dot — the kernel's 8 serial MACs per thread plus the 5-step butterfly.
    `reference.attend` gets the same bits from torch's own 256-wide dot (0 of 98,304 elements off the
    pins, results.md §2), so the matmul form is used here too; `score_stream` is its only caller."""
    return float(np.float32(torch.from_numpy(np.asarray(q, dtype=np.float32)).unsqueeze(0)
                           @ torch.from_numpy(np.asarray(k, dtype=np.float32)).t())[0, 0])


class CacheDoc:
    """The K/V the mirror is allowed to read, instrumented.

    Lanes are built from the pins up to the wave's own appends and POISON (`nan`) beyond; every read
    is counted by (kind, kv, lane). A causal-mask bug therefore does not merely change a number, it
    reads a `nan` that propagates into the output and is reported as a wrong element — which is the
    `mixed_bw` lesson (a check that cannot fail is worse than none, and a checksum that is
    pre-filled with the expected value proves nothing): the tripwire has to be a value the correct
    run can never produce, not a count the reader has to trust.
    """

    def __init__(self, k: torch.Tensor, v: torch.Tensor, valid: int, cap: int):
        self.k = torch.full((N_KV, cap, DH), POISON, dtype=torch.float32)
        self.v = torch.full((N_KV, cap, DH), POISON, dtype=torch.float32)
        self.k[:, :valid] = k[:, :valid].to(torch.float32)
        self.v[:, :valid] = v[:, :valid].to(torch.float32)
        self.valid, self.cap = valid, cap
        self.reads: dict[tuple, int] = {}

    def read(self, kind: str, kv: int, lane: int) -> np.ndarray:
        self.reads[(kind, kv, lane)] = self.reads.get((kind, kv, lane), 0) + 1
        return (self.k if kind == "k" else self.v)[kv, lane].numpy()

    @property
    def bytes_read(self) -> int:
        return len(self.reads) * DH * 2

    def poison_rows(self) -> int:
        """How many distinct rows landed on poison — the mask-leak witness, reported always."""
        n = 0
        for (kind, kv, lane), _ in self.reads.items():
            row = (self.k if kind == "k" else self.v)[kv, lane]
            if bool(torch.isnan(row).any()):
                n += 1
        return n


# ============================================================================================
# The two forms' arithmetic.
# ============================================================================================
def score_stream(q: np.ndarray, doc: CacheDoc, kv: int, lanes: list[int], self_lane: int,
                 self_k: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """§05.5's scores for one (row, head) pair, in lane order: gather the row's K lanes (each read
    through `doc`, so the read census is a fact about the mirror rather than an assertion about it),
    one round of the 256-term dot, then `x SCALE` — one BF16 round before the scale, because §05.5
    rounds there and because SCALE = 2^-4 makes the product exact in BF16.

    The self lane comes from the staged `k_rope` (the §05.7 append, which the cache slot may not hold
    yet), exactly as `attn_core` and `attn_tile_wave` both do."""
    qq = q if keep is None else q * keep
    rows = np.empty((len(lanes), DH), dtype=np.float32)
    for i, j in enumerate(lanes):
        k = self_k if j == self_lane else doc.read("k", kv, j)
        rows[i] = k if keep is None else k * keep
    dots = (torch.from_numpy(qq).unsqueeze(0) @ torch.from_numpy(rows).t()).numpy().ravel()
    return bf16_round(dots) * np.float32(SCALE)


def v_stream(doc: CacheDoc, kv: int, lanes: list[int], self_lane: int,
             self_v: np.ndarray) -> torch.Tensor:
    rows = [self_v if j == self_lane else doc.read("v", kv, j) for j in lanes]
    return torch.from_numpy(np.stack(rows)) if rows else torch.zeros(0, DH, dtype=torch.float32)


def one_pass(sc: np.ndarray, V: torch.Tensor, seed_m=0.0, seed_l=None,
             seed_acc=None) -> tuple[float, float, np.ndarray]:
    """§05.5's reference form, verbatim: max in index order, sum-of-exp in index order, the BF16
    round of the NORMALISED weight, then P·V serial in t. Returns (m, l=1, acc) with the weights
    already divided, so the epilogue must not divide again (§07.3's rule).

    The three `seed_*` arguments exist only so the `headcarry` injection can seed a head from the
    previous head's state — §07.3's bug in the tiled form's shape, where consecutive heads are warps
    of ONE block and a shared loop-carried `m`/`l`/`acc` is exactly the defect that form would have.
    Left alone they are the neutral (0, 0, 0) the kernel starts from."""
    m = np.float32(seed_m)
    l = np.float32(0.0 if seed_l is None else seed_l)
    acc = np.zeros(DH, dtype=np.float32) if seed_acc is None else np.array(seed_acc, dtype=np.float32)
    for t in range(len(sc)):
        m = np.maximum(m, sc[t])
    for t in range(len(sc)):
        l = np.float32(l + expf(sc[t] - m))
    for t in range(len(sc)):
        wt = bf16(expf(sc[t] - m) / l)
        acc = acc + np.float32(wt) * V[t].numpy()
    return float(m), 1.0, acc


def online(sc: np.ndarray, V: torch.Tensor, blk: int, seed_m=0.0, seed_l=None,
           seed_acc=None) -> tuple[float, float, np.ndarray]:
    """§07.3's streaming form with `attn_core`'s EXACT tile-of-`blk` algebra (running max rescaled by
    `expf(m - mn)`, weights NOT rounded, the divide left to the merge). `reference.attend`'s
    online branch is the same form; the tile grouping is reproduced here because a different grouping
    of the same numbers is a different F32 sum, and the claim is bit-identity.

    `seed_*` exist for the `headcarry` control only (see `one_pass`): a state that outlived its owner
    has to be injectable on BOTH routes, or the control proves nothing about the route it skips.
    Neutral by default, which is what every honest call passes."""
    m = np.float32(seed_m)
    l = np.float32(0.0 if seed_l is None else seed_l)
    acc = np.zeros(DH, dtype=np.float32) if seed_acc is None else np.array(seed_acc, dtype=np.float32)
    for base in range(0, len(sc), blk):
        sub = sc[base:base + blk]
        mt = np.float32(-1e30)
        for t in range(len(sub)):
            mt = np.maximum(mt, sub[t])
        mn = np.maximum(m, mt)
        rescale = expf(m - mn)
        e = [expf(sub[t] - mn) for t in range(len(sub))]
        ls = np.float32(0.0)
        for t in range(len(sub)):
            ls = np.float32(ls + e[t])
        l = np.float32(l * rescale + ls)
        acc = acc * rescale
        for t in range(len(sub)):
            acc = acc + np.float32(e[t]) * V[base + t].numpy()
        m = mn
    return float(m), float(l), acc


def merge_partials(partials: list[tuple[float, float, np.ndarray]]) -> tuple[float, float, np.ndarray]:
    """`attn_combine`'s merge, in window order. Returns `(m, l, acc)` with `acc` NOT yet divided, because
    §05.5's rule is that a weight is divided exactly ONCE and the streaming route leaves that divide
    to the epilogue (`finish_head`: `bf(fb(acc / l))`).

    It used to return `(m, acc)` and the caller divided by 1. That is §07.3's mis-normalisation — a row
    whose partials were summed but never divided — and it sat invisible in this file until §20 pushed
    the mirror past MAXSC, because no pinned fixture reaches 512 lanes and the one-pass route does its
    own dividing. A path a gate cannot reach is a path with no tests, including tests of the gate.
    """
    m, l, acc = 0.0, np.float32(0.0), np.zeros(DH, dtype=np.float32)
    for ms, ls, avs in partials:
        mn = max(m, ms)
        r0, r1 = expf(m - mn), expf(ms - mn)
        l = np.float32(l * r0 + np.float32(ls) * r1)
        acc = acc * np.float32(r0) + avs * np.float32(r1)
        m = mn
    return m, float(l), acc


def epilogue(acc: np.ndarray, l: float, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """§05.6: `a = bf16(acc/l)`, then `gated = a * bf16(sigmoid(gate))`. `gate` is the head's OWN
    256 gate values, i.e. `qg[..., DH:]` — the q_proj row is [q head | gate] interleaved per head."""
    av = vec16(torch.from_numpy(np.asarray(acc, dtype=np.float32)) / np.float32(l))
    a_out = vec16(av)
    g = vec16(torch.sigmoid(gate.to(torch.float32)))
    return a_out, vec16(av * g)


# ============================================================================================
# Geometry: the two forms' launch plans, and what each one reads.
# ============================================================================================
def windows_of(lanes_last: int) -> int:
    """Lane windows a launch has. THE SAME expression for both forms (`split_for_lanes`), which is
    what keeps the one-pass/streaming boundary — and therefore the bit-identity claim — from drifting
    between them."""
    return (lanes_last + MAXSC - 1) // MAXSC


def tile_windows_of(nrows: int) -> int:
    return (nrows + M - 1) // M


# ---- the per-row lane split, and the split-partial buffer: two ADDRESS claims ---------------
# Both are arithmetic on integers, so both are checkable on CPU with no device and no band, and both
# failed §19 on the device for reasons that had nothing to do with rounding.
def route_of(nrow: int, split: int, full: bool = True) -> int:
    """`blocks_for_row`: one block iff the staging fits the row, otherwise the LAUNCH's `split`.
    The shipped form's rule; the tiled form is only bit-identical to it if it uses this one."""
    return 1 if (full and nrow <= MAXSC) else max(split, 1)


def row_bounds(nrow: int, s_row: int) -> list[tuple[int, int]]:
    """`attn_core`'s slice of one row across `s_row` blocks: `span = ceil(n/s_row)`, block y owns
    `[y*span, min(n, (y+1)*span))`. Fixed MAXSC-wide windows are a DIFFERENT cut (for 513 lanes and
    split 2 they cut at 257/256, windows at 512/1), and a different cut is a different F32 sum."""
    span = (nrow + s_row - 1) // s_row
    return [(b * span, min(nrow, b * span + span)) for b in range(s_row)]


def fixed_window_bounds(nrow: int) -> list[tuple[int, int]]:
    """The §12/§19 tiled form's cut: MAXSC-wide windows, because a window was what a block shared a
    K load over. Kept so the failure can be re-priced after the fix, not to be used by it."""
    return [(lo, min(nrow, lo + MAXSC)) for lo in range(0, nrow, MAXSC)]


def part_offset(row: int, head: int, window: int, split: int) -> int:
    """The one split-partial slot expression: `((row*N_Q + head)*split + window) * (DH + 2)`, the
    head index being the FULL head (0..N_Q), not a head-in-group. Every writer and the merge must
    spell it this way — `part_layout()` proves they do — because the row of a group's tuple is
    multiplied by N_Q and any smaller multiplier makes two KV groups alias."""
    return ((row * N_Q + head) * split + window) * (DH + 2)


def part_offset_collapsed(row: int, head: int, window: int, windows: int) -> int:
    """The layout §12 shipped and §19's device run failed on: `((row*G + head_in_group)*windows + y)`,
    which has no KV-group term, so blocks x=0 and x=1 of one row write the SAME (2 + DH) floats.
    Reachable only from the negative, and kept so the negative can be re-priced (§20)."""
    return ((row * G + head % G) * windows + window) * (DH + 2)


def part_layout(src: str) -> dict:
    """Which partial layout the kernel source actually uses, read off its own `a.part +` sites.

    The rule is one line: a slot expression must multiply the row by `N_Q`, i.e. carry the FULL head
    index, because the heads of a KV group are the block in the tiled form and the grid in the
    shipped one, and a tuple keyed by head-in-group alone cannot tell group 0's head h from group
    1's. Out-of-range writes and aliases do not fault here: the survivor of two writers is whichever
    block landed last, and the merge then reads one of them for BOTH heads.
    """
    sites = PART_SITE.findall(src)
    if len(sites) < 3:
        raise SystemExit(f"{len(sites)} `a.part +` index sites found — this model's pattern is stale")
    collapsed = [(r, mult, h) for r, mult, h in sites if mult != "N_Q"]
    return {"sites": [{"row": r, "mult": mult, "head": h} for r, mult, h in sites],
            "collapsed": collapsed, "ok": not collapsed}


def tile_window_form(src: str) -> str:
    """`per-row-slice` or `fixed-window`, read off `attn_tile_body`'s own source. Same anti-rot rule
    as the LDS census: the model reads the geometry instead of retyping it, so a kernel that goes
    back to fixed windows makes the mirror price fixed windows (and fail) rather than agreeing with
    itself about a form the kernel no longer is."""
    if TILE_FIXED_WINDOW.search(src):
        return "fixed-window"
    if TILE_PER_ROW_SLICE.search(src):
        return "per-row-slice"
    raise SystemExit("attn_tile_body's lane-window rule is neither `y * MAXSC` nor "
                     "`blocks_for_row(a, row)` — this model's patterns are stale")


class PartBuf:
    """The `part` scratch, modelled as an ADDRESS space rather than as a list of per-row partials.

    §19's failure was not arithmetic: the tiled form's per-window partials were merged exactly as the
    shipped form's, but two blocks wrote them to the same address. A mirror that merges a Python list
    per row cannot express that, which is why §12's gate was green and the device was red. This models
    the buffer: writers land on an offset, the last writer wins (block order is grid order, x
    fastest, so the higher KV group wins a slot — on the device the winner is whichever CU landed
    last, i.e. arbitrary, which is worse and is said so in the report), and every alias is recorded
    with both writers named. A reader that has to trust a count is a reader who is being fooled.
    """

    def __init__(self, offset_fn):
        self.off = offset_fn
        self.cells: dict[int, tuple] = {}     # float offset -> (value tuple, writer)
        self.writes: dict[int, list] = {}     # float offset -> every writer that landed on it

    def write(self, row: int, head: int, window: int, split: int, val: tuple, who: str):
        base = self.off(row, head, window, split)
        self.writes.setdefault(base, []).append(who)
        self.cells[base] = val

    def read(self, row: int, head: int, window: int, split: int):
        base = self.off(row, head, window, split)
        if base not in self.cells:
            return (0.0, 0.0, np.zeros(DH, dtype=np.float32))   # zero-filled scratch = the neutral
        return self.cells[base]

    def aliases(self) -> list[tuple[int, list]]:
        return [(off, sorted(set(w))) for off, w in self.writes.items() if len(set(w)) > 1]


def stages_declared(src: str, c: dict) -> dict:
    """The `__shared__` arrays a tile form actually declares, sized from their OWN bracket factors.

    `kernel_consts` computes a size for `s_sc`/`s_k`/`s_v` from the header's constants whatever the
    kernel does with them, and §12's form declared no `s_v` at all — pass 2 read V straight from global.
    A census that prints a size for a stage that does not exist hides the difference between "the stage
    is too small" (a defect to fix) and "there is no stage" (a traffic claim that was never true), so
    every LDS number printed here is read off a declaration.
    """
    named = {"ATTN_TILE_G": c["G"], "ATTN_TILE_M": c["M"], "ATTN_TILE_L": c["ATTN_TILE_L"],
             "kTilePairs": c["G"] * c["M"],   # the §12 spelling of the same product
             "MAXSC": c["MAXSC"], "DH": c["DH"]}
    out: dict[str, int] = {}
    for line in src.splitlines():
        if "__shared__" not in line:
            continue
        for name, brackets in re.findall(r"\b(s_sc|s_k|s_v)\s*((?:\[[^\]]+\])+)", line):
            prod = 1
            for bracket in re.findall(r"\[([^\]]+)\]", brackets):
                for term in bracket.split("*"):
                    term = term.strip()
                    prod *= named[term] if term in named else int(term)
            out[name] = prod * 2          # every stage is BF16
    return out


def k_stage_reach(src: str, c: dict, span_max: int) -> dict:
    """How far the two lane stagings' index arithmetic can walk into LDS, read off the kernel source.

    This is a FOOTPRINT claim, not the declared-size claim the compile census prints. `s_k` and `s_v`
    are `ATTN_TILE_L * DH` BF16 — `L` sub-block SLOTS, because the sub-block loop steps `s_lo` by `L`
    and the streaming route stages one `tile` of V at a time — so the index into them must be the slot
    (`rr - s_lo`, `rr - base`, or a name defined as exactly that), not the window-absolute lane. With
    `rr` the same loop reaches `span * DH * 2` bytes for a row tile whose widest row spans `span`
    lanes: 141,824 B for a 277-lane window and 262,144 B for a MAXSC-wide one, against a kernel that
    reserves 20,480 B of LDS in total, on every sub-block after the first. Out-of-range LDS does not
    fault on this part; the writes land on whatever else the block reserved — here the score staging —
    so the symptom is a numerics difference, not a crash. That is §13's `1,098,790 elements moved`,
    and this census names it without a device (§14).

    The second claim is the one the V stage adds: a staged streaming tile must fit the array, i.e.
    `ATTN_TILE_L >= TILE`, and the census requires the kernel's own `static_assert` for it rather than
    trusting a comment. A `--tile` bigger than `L` is a silent overrun of exactly the §13 kind.

    The spellings are read from the source rather than retyped here so the census fails when the source
    regresses, and both spellings' footprints are reported so the size of the mistake is on the record.
    """
    exprs = S_K_INDEX.findall(src)
    if not exprs:
        raise SystemExit("no `s_*[... + lane * 8]` index site found — this model's pattern is stale")
    # An expression is sub-block-relative iff it subtracts a loop base (`(rr - s_lo) * DH`), or mentions
    # `slot` as a factor — the one name the file defines as such (`const int slot = rr - s_lo;`), which
    # is what the stage write spells (`slot * DH`, a FACTOR, not the whole expression). A bare `rr`, or a
    # bare loop variable whose definition the census cannot see, is treated as absolute.
    slot_names = {"slot"} if SLOT_DEFS.search(src) else set()

    def is_slot(e: str) -> bool:
        return (" - " in e) or any(re.search(rf"\b{n}\b", e) for n in slot_names)

    slot_relative = all(is_slot(e) for e in exprs)
    worst_slot = (c["ATTN_TILE_L"] - 1) if slot_relative else max(0, min(span_max, c["MAXSC"]) - 1)
    bytes_worst = (worst_slot + 1) * c["DH"] * 2
    # Is V staged at all? `kernel_consts` computes an `s_v` SIZE from the constants whatever the kernel
    # does with it, and §12's form had no `s_v` — every pass-2 touch was a global read — so the census
    # has to distinguish "the stage is too small" from "there is no stage", which is the difference
    # between a defect to fix and a traffic claim that was never true. Print the difference.
    decls = stages_declared(src, c)
    v_staged = "s_v" in decls
    index_ok = slot_relative and bytes_worst <= decls.get("s_k", c["k_stage_bytes"])
    v_guard = V_STAGE_GUARD.search(src) is not None
    return {"exprs": sorted(set(exprs)), "slot_relative": slot_relative, "span_max": span_max,
            "bytes_worst": bytes_worst, "declared": decls.get("s_k", 0),
            "v_declared": decls.get("s_v", 0), "v_staged": v_staged,
            "v_tile_guard": v_guard, "declared_stages": decls,
            "block_lds": sum(decls.values()) or c["block_lds_bytes"],
            "index_ok": index_ok, "ok": index_ok and v_staged and v_guard}


def block_touches(lanes: list[int]) -> int:
    """(row, head, lane) triples a launch must produce — the shipped form's work, identical here."""
    return sum(n * N_Q for n in lanes)


def tile_lane_segments(lanes: list[int], nrows: int, rows_per_block: int,
                       window_form: str, split: int) -> list[tuple[int, int]]:
    """The (lo, hi) lane segments each block stages, in block order — one entry per block.

    A row-tile stops at its widest row, because the causal mask is a PREFIX and a masked lane is never
    read at all; that prefix fact is what makes tiling rows legal at all. With the per-row cut (§20) a
    row-tile IS one row, so the segments are that row's own slices and there is no widest-row residue:
    a per-row slice never over-reads a causal tail. Asking for the per-row cut with `rows_per_block > 1`
    is refused rather than modelled, because the kernel's own `static_assert` refuses it: M rows would
    need a stage over the union of M different slices, which is not bounded by MAXSC.
    """
    if rows_per_block > 1 and window_form == "per-row-slice":
        raise SystemExit("tile_lane_segments: the per-row cut cannot be shared by more than one row "
                         "(the union of two rows' slices is not bounded by MAXSC) — that is the "
                         "kernel's own static_assert, so the model declines it too")
    segs: list[tuple[int, int]] = []
    for r0 in range(0, nrows, rows_per_block):
        rows = range(r0, min(nrows, r0 + rows_per_block))
        n_max = max(lanes[r] for r in rows)
        cut = (fixed_window_bounds(n_max) if window_form == "fixed-window"
               else row_bounds(n_max, route_of(n_max, split)))
        segs += [(lo, hi) for lo, hi in cut if hi > lo]
    return segs


def tiled_reads(lanes: list[int], nrows: int, rows_per_block: int,
                k_copies: int = 1, v_copies: int = 1, window_form: str = "per-row-slice",
                split: int | None = None) -> tuple[int, int]:
    """How many BF16 K rows (and V rows) a whole launch asks global for, counted WITH multiplicity
    across blocks and across the heads of a block — traffic is multiplicity, not distinctness.

    `k_copies`/`v_copies` are how many times ONE BLOCK asks global for the same lane row; pass
    `stage_copies(src)`'s number for the form as written. Both are 1 in the §20 form — one warp stages
    one lane of the sub-block into `s_k` (pass 1) or `s_v` (pass 2) and all G head-warps read that slot
    — which is the lever: one K row load feeding G dots, and the same for V, V included because V is
    half the byte wall. §12 staged per pair instead (`copies = G`), so the re-use it booked as "M" was
    nothing at all; pass `k_copies=G, v_copies=G` to price that form again.
    """
    split = split if split is not None else windows_of(max(lanes))
    staged = sum(hi - lo for lo, hi in
                 tile_lane_segments(lanes, nrows, rows_per_block, window_form, split))
    # N_KV blocks per row (one per KV group, `blockIdx.x`), each asking `copies` times.
    return N_KV * k_copies * staged, N_KV * v_copies * staged


def shipped_reads(lanes: list[int], nrows: int) -> tuple[int, int]:
    """The shipped form's census: one block per (row, head), each reading its own row's lanes of its
    own KV group, so a group's row is fetched once per query head of that group — G times as often as
    by a block that holds the whole group."""
    k = sum(lanes) * N_Q
    return k, k

# ============================================================================================
# The mirror: one wave through prep + core + combine, in either form.
# ============================================================================================
def wave_inputs(case: dict) -> dict:
    """Per-row staged operands of a wave case, exactly as `attn_prep` leaves them: q_rope and k_rope
    from the pins (which is what makes this a prediction rather than a re-derivation), and the cache
    built with each row's own append at its own slot and poison past it."""
    p0, p1 = case["span"]
    return {"rows": list(range(p0, p1 + 1)), "n": p1 - p0 + 1,        # q_rope/k_rope/v are the pins
            "q_rope": case["ref"]["q_rope"], "k_rope": case["ref"]["k_rope"],
            "v": case["ref"]["v"], "gate": case["qg"], "cap": case["cap"],
            "p0": p0, "p1": p1}


def route_and_bounds(wform: str, nrow: int, wins: int) -> tuple[int, list]:
    """One row's lane cut, per form. `per-row-slice` is `attn_core`'s own (`span = ceil(n/split)`),
    which the tiled form reaches by CALLING THE SAME FUNCTION; `fixed-window` is §12/§19's tiled cut,
    kept so that failure can be re-priced and separated from the aliasing one. Returns how many blocks
    the row actually uses and their [lo,hi) bounds — and for a row the staging fits, both forms return
    one block holding the whole row, which is the tile-boundary case §20 has to price."""
    if wform == "fixed-window":
        bounds = fixed_window_bounds(nrow)
        return (1 if nrow <= MAXSC else len(bounds)), bounds
    s_row = route_of(nrow, wins)
    return s_row, row_bounds(nrow, s_row)


def wave_out(case: dict, form: str, neg: str = "none", doc: CacheDoc | None = None,
             layout: str | None = None, window_form: str | None = None) -> dict:
    """One pinned wave span through the mirror, returning `attn_pre_gate` and `gated` laid out the
    way the kernel writes them ([rows][N_Q*DH]), plus the read census and the partial buffer.

    `form` is "shipped" (one block per (row, head), `attn_core`) or "tile" (the G heads of one KV
    group of one row per block, `attn_tile_wave`). The two differ in WHICH threads load which K/V row
    and who runs pass 2; the arithmetic per (row, head) is the same five reductions and the per-row
    lane cut is the same function, so identical bits are the claim being tested.

    Two arguments exist to model a FORM THE KERNEL NO LONGER IS, for the negatives and for §20's
    before/after: `layout` forces a partial-slot spelling ("group-dimensioned" = the kernel's, read
    off its source by `part_layout`; "group-collapsed" = §12's aliasing one) and `window_form` forces
    a lane cut. Both default to what the header says, so the mirror cannot quietly agree with it.
    """
    srct = KERNEL_H.read_text()
    if layout is None:
        layout = "group-dimensioned" if part_layout(srct)["ok"] else "group-collapsed"
    if window_form is None:
        window_form = "per-row-slice" if form == "shipped" else tile_window_form(srct)
    if neg == "partalias":
        layout = "group-collapsed"      # the injection IS the layout, so it overrides the source
    w = wave_inputs(case)
    rows, n, p0 = w["rows"], w["n"], w["p0"]
    lvalid = p0 + 1                                     # decode_step.cpp: a.lvalid = pos0 + 1
    lanes = [lvalid + i for i in range(n)]              # row i may attend this many lanes
    lanes_last = lanes[-1]
    wins = windows_of(lanes_last)
    if doc is None:
        doc = CacheDoc(case["kcache"].reshape(N_KV, case["cap"], DH),
                       case["vcache"].reshape(N_KV, case["cap"], DH), p0, case["cap"])
    # The wave's own appends, in row order: row i writes lane p0+i. Lanes past p0+n stay poison.
    for i in range(n):
        doc.k[:, p0 + i] = w["k_rope"][i].to(torch.float32)
        doc.v[:, p0 + i] = w["v"][:, i].to(torch.float32)   # ref["v"] is [N_KV, rows, DH]

    qg = w["gate"].reshape(n, N_Q, 2 * DH)[:, :, DH:]      # the gate half of each q_proj row
    a_pre = torch.zeros(n, N_Q, DH, dtype=torch.float32)
    gated = torch.zeros(n, N_Q, DH, dtype=torch.float32)
    keep = half_keep() if neg == "dimhalf" else None
    buf = PartBuf(part_offset if layout == "group-dimensioned" else part_offset_collapsed)
    todo: list[tuple[int, int, int]] = []        # (row, head, s_row) the merge still owes
    leak_src: dict[int, tuple] = {}                # `headcarry`: the PREVIOUS head's state, and this
                                                   # dict is what resets at a block boundary
    # ---- the core launch: every (row, head) pair, in block order (grid.x fastest = head ascending).
    for i in range(n):
        row_pos = p0 + i
        nrow = lanes[i]
        if neg == "dropcurrent":
            nrow = row_pos                       # NEG_DROPCUR: the appended lane is forgotten
        elif neg == "readpast":
            nrow = case["cap"]                   # NEG_READPAST: the causal mask is gone
        s_row, bounds = route_and_bounds(window_form, nrow, wins)
        for head in range(N_Q):
            kv = head // G
            if neg == "gqaoff":
                kv = (kv + 1) % N_KV             # the whole group reads the neighbour
            elif neg == "gqa_head0":
                kv = (kv + 1) % N_KV if head == 0 else kv
            if neg == "headcarry" and head % G == 0:
                leak_src = {}            # a block's FIRST warp has nothing to leak from, and a leak
                                         # cannot cross blocks: with M=1 a block is one row's G heads
                                         # of one KV group, so the heads a carry can reach are exactly
                                         # those with h%G != 0. That is the must-catch set, and it is
                                         # geometry, not taste.
            q = w["q_rope"][i, head].to(torch.float32).numpy()
            self_k = w["k_rope"][i, kv].to(torch.float32).numpy()
            self_v = case["ref"]["v"][kv, i].to(torch.float32).numpy()
            # The scores and V rows of one (row, head) are the same for every form, layout or injection
            # that does not change what that pair reads, and a straddling wave computes ~8k of them.
            # Caching them on the case is what makes the attribution table (four passes over the same
            # operands) cost one pass of arithmetic.
            cache = case.setdefault("_scorecache", {})
            key = (i, head, kv, nrow, neg == "dimhalf")
            if key not in cache:
                cache[key] = (score_stream(q, doc, kv, list(range(nrow)), row_pos, self_k, keep),
                              v_stream(doc, kv, list(range(nrow)), row_pos, self_v))
            sc_all, V_all = cache[key]
            if s_row == 1:
                # One block, the whole row, §05.5's reference form. Below MAXSC this is where every
                # pinned prefill row in the repo lives — no pinned prompt reaches 512 tokens — and it
                # is the SAME branch in both forms, which is why §12's gate could be green while the
                # device was red: nothing on this path had changed between the two forms.
                seed = (0.0, np.float32(0.0), np.zeros(DH, dtype=np.float32))
                if neg == "headcarry" and (head - 1) in leak_src:
                    seed = leak_src[head - 1]      # state that outlived the head it belonged to
                m, l, acc = one_pass(sc_all, V_all, seed[0], seed[1], seed[2])
                leak_src[head] = (m, l, acc)
                a_pre[i, head], gated[i, head] = epilogue(acc, l, qg[i, head])
            else:
                # Above the staging the row is cut into `s_row` blocks whose (max,sum,sum w·v) partials
                # `attn_combine` merges — and here they go THROUGH A BUFFER, not through a Python list.
                # §19's device failure was an address, not an arithmetic, and a mirror that hands each
                # row its own list cannot express two blocks writing one slot. `window_form` says which
                # cut the form used: `attn_core`'s per-row `span = ceil(n/split)`, or §12's fixed
                # MAXSC-wide windows (both are §07.3's streaming form; only the first is bit-identical
                # to the other form, because a different partial boundary is a different F32 sum).
                seed = (0.0, np.float32(0.0), np.zeros(DH, dtype=np.float32))
                if neg == "headcarry" and (head - 1) in leak_src:
                    seed = leak_src[head - 1]      # the previous head's state, same block, same group
                for y, (lo, hi) in enumerate(bounds):
                    m, l, acc = online(sc_all[lo:hi], V_all[lo:hi], TILEW,
                                       seed[0], seed[1], seed[2])
                    buf.write(i, head, y, wins, (m, l, acc), f"r{i}h{head}")
                leak_src[head] = (m, l, acc)     # the last slice's state is what would outlive it
                todo.append((i, head, s_row))

    # ---- the combine launch: one block per (row, head), reading the same buffer back.
    for i, head, s_row in todo:
        parts = [buf.read(i, head, y, wins) for y in range(s_row)]
        _, l, acc = merge_partials(parts)      # the divide is the epilogue's, at l — not at 1.0
        a_pre[i, head], gated[i, head] = epilogue(acc, l, qg[i, head])
    return {"attn_pre_gate": a_pre, "gated": gated, "doc": doc, "buf": buf,
            "rows": rows, "lanes": lanes, "windows": wins, "layout": layout,
            "window_form": window_form, "rows_per_block": M}


# ============================================================================================
# Commands
# ============================================================================================
def fmt_b(n: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"


def cmd_geometry(args) -> None:
    print(f"constants from {KERNEL_H.name}: N_Q={N_Q} N_KV={N_KV} G={G} DH={DH} MAXSC={MAXSC} "
          f"TILE={TILEW} | tiled: M={M} pairs/block={G * M} L={C['ATTN_TILE_L']}")
    for tokens, rows in ((1024, 512), (277, 64), (300, 512)):
        lanes, lvalid0 = [], 1
        for pos0 in range(0, tokens, rows):
            nr = min(rows, tokens - pos0)
            lanes += [pos0 + lvalid0 + i for i in range(nr)]
        touches = block_touches(lanes)
        sh_k, sh_v = shipped_reads(lanes, len(lanes))
        src_g = KERNEL_H.read_text()
        st = stage_copies(src_g)
        ti_k, ti_v = tiled_reads(lanes, len(lanes), M, k_copies=st["copies"],
                                 v_copies=st["copies"], window_form=tile_window_form(src_g))
        print(f"\n{tokens} tokens, {rows}-row waves: {len(lanes)} rows, "
              f"{touches:,} touches, windows={windows_of(max(lanes))}")
        print(f"  shipped  core blocks {N_Q * windows_of(max(lanes)) * len(lanes):>7}   "
              f"K/V rows read {sh_k + sh_v:>10,} = {fmt_b((sh_k + sh_v) * DH * 2)}")
        print(f"  tiled    core blocks {N_KV * windows_of(max(lanes)) * tile_windows_of(len(lanes)):>7}"
              f"   K/V rows read {ti_k + ti_v:>10,} = {fmt_b((ti_k + ti_v) * DH * 2)}")
        print(f"  re-use   {(sh_k + sh_v) / (ti_k + ti_v):.1f}x fewer K/V row reads "
              f"(stage is {st['level']}-level: {st['copies']} fetches per lane row per block, so the "
              f"claim is {st['reuse']}x = the group share, not G x M)")
        print(f"  staging  {fmt_b(C['score_staging_bytes'])} of scores + "
              f"{fmt_b(C['k_stage_bytes'])} of K + {fmt_b(C['v_stage_bytes'])} of V = "
              f"{fmt_b(C['block_lds_bytes'])} per block; 64 KiB/CU admits "
              f"{int(65536 // C['block_lds_bytes'])} block(s)/CU")
        reach = k_stage_reach(KERNEL_H.read_text(), C, max(lanes))
        print(f"  LDS reach  the K stage's index arithmetic reaches {fmt_b(reach['bytes_worst'])} "
              f"for a span of {reach['span_max']} lanes against {fmt_b(reach['declared'])} declared: "
              f"{'ok' if reach['ok'] else 'OUT OF BOUNDS'} "
              f"(spelling: {', '.join('`' + e + '`' for e in reach['exprs'])})")


def cmd_traffic(args) -> None:
    """The §7 pre-registered numbers, from the closed-form census (no per-element work)."""
    lanes, lvalid0 = [], 1
    for pos0 in range(0, args.tokens, args.wave_rows):
        nr = min(args.wave_rows, args.tokens - pos0)
        lanes += [pos0 + lvalid0 + i for i in range(nr)]
    nrows = len(lanes)
    touches = block_touches(lanes)
    flops = touches * 4 * DH
    src = KERNEL_H.read_text()
    st, wf = stage_copies(src), tile_window_form(src)
    sh_k, sh_v = shipped_reads(lanes, nrows)
    ti_k, ti_v = tiled_reads(lanes, nrows, M, k_copies=st["copies"], v_copies=st["copies"],
                             window_form=wf)
    vp_k, vp_v = tiled_reads(lanes, nrows, M, k_copies=st["copies"], v_copies=G, window_form=wf)
    sh_bytes = (sh_k + sh_v) * DH * 2
    ti_bytes = (ti_k + ti_v) * DH * 2
    vp_bytes = (vp_k + vp_v) * DH * 2
    layers = 6
    print(f"{args.tokens} tokens x {layers} attention layers, {args.wave_rows}-row waves")
    print(f"  touches          {touches * layers:,}   flops {flops * layers / 1e9:.2f} GFLOP "
          f"(the ledger's 25.79)")
    print(f"  shipped K/V read {fmt_b(sh_bytes * layers)}  = "
          f"{sh_bytes * layers / (flops * layers):.3f} byte/flop")
    print(f"  tiled   K/V read {fmt_b(ti_bytes * layers)}  = "
          f"{ti_bytes * layers / (flops * layers):.3f} byte/flop   "
          f"({sh_bytes / ti_bytes:.1f}x less; stage is {st['level']}-level, {st['copies']} fetch"
          f"{'y' if st['copies'] == 1 else 'ies'} per lane row, so the claim is {st['reuse']}x;"
          f" lane cut {wf})")
    print(f"  V-per-pair        {fmt_b(vp_bytes * layers)}  = "
          f"{vp_bytes * layers / (flops * layers):.3f} byte/flop   "
          f"({sh_bytes / vp_bytes:.1f}x less: what hoisting only K to the block would have bought)")
    print(f"\nthe §7 pre-registered expectation, as numbers:")
    print(f"  baseline attn                     {MEASURED_ATTN_MS:.1f} ms  (library route, best-of-3)")
    print(f"  shipped bytes / 237 GB/s            {sh_bytes * layers / 1e9 / DRAM_READ_GBS * 1000:.1f} ms"
          "   <- predicts the measured line")
    print(f"  tiled bytes / 237 GB/s              {ti_bytes * layers / 1e9 / DRAM_READ_GBS * 1000:.1f} ms"
          "   <- the traffic term after the change")
    print(f"  V-per-pair / 237 GB/s               {vp_bytes * layers / 1e9 / DRAM_READ_GBS * 1000:.1f} ms"
          "   <- if only K were hoisted to the block (V still staged per pair)")
    print(f"  arithmetic at the hand-GEMM rate    {flops * layers / 1e12 / HAND_GEMM_TF * 1000:.1f} ms"
          "   <- what binds once bytes stop binding")
    print(f"  arithmetic at the vector rate       {flops * layers / 1e12 / VECTOR_FP32_TF * 1000:.1f} ms")
    print(f"\n  WIN LINE (§7 item 2): attn must land at <= 40 ms absolute.")
    print(f"  FALSIFIER A: if tiled bytes fall {sh_bytes / ti_bytes:.0f}x and attn does not fall at all,")
    print(f"     traffic was not the wall and §4's verdict is wrong -> the redundant pass-2 loops are it.")
    print(f"  FALSIFIER B: if attn lands 40-80 ms, traffic fell but something else binds: report PARTIAL.")
    print(f"  EXPECTED BAND: {max(ti_bytes * layers / 1e9 / DRAM_READ_GBS * 1000, flops * layers / 1e12 / HAND_GEMM_TF * 1000):.0f} ms "
          f"(the larger of the two terms the model leaves) plus whatever issue costs; a number well above")
    print(f"     that is the mirror being wrong about what the kernel reads, not a slower kernel.")
    print("\n  Counter to verify the read census, not assume it (§9's line, plus the byte counter):")
    print("    rocprofv3 --kernel-include-regex \"attn_tile\" --pmc SQ_INSTS SQ_INSTS_VALU "
          "SQ_INSTS_LDS SQ_WAVES GRBM_COUNT GRBM_RUN_BUSY_STATS --output-format csv -d tmp/rocp "
          "-o tile -- ./engine/build/decode_run --ids \"$(cat progress/bench/evidence/q10/ids1024.txt)\" --steps 1")
    print(f"    SQ_INSTS_VALU must NOT rise with the tile form (the softmax dedup should cut it: the")
    print(f"    shipped form runs pass 2 in all {C['warps']} warps of its block; the tile form runs")
    print(f"    it once per head ({C['warps']} warps = {G} heads, one warp each, results.md §20), and the byte")
    print(f"    claim is verified by the ms falling toward the traffic term above, not by a counter")
    print(f"    this box has never produced for DRAM bytes.")


def cmd_lds(args) -> int:
    """The K stage's LDS footprint, priced from a header's own index expression.

    Defaults to the live kernel; `--kernel` points it at another copy of the same file, which is how
    §14's before/after was taken without a device:

        git show HEAD:prototype/attn_decode/attn_decode_kernels.h > /tmp/pre.h
        .venv/bin/python prototype/attn_tile_model.py lds --kernel /tmp/pre.h   # FAIL
        .venv/bin/python prototype/attn_tile_model.py lds                       # PASS
    """
    path = Path(args.kernel) if args.kernel else KERNEL_H
    c, src = kernel_consts(path), path.read_text()
    decls = stages_declared(src, c)
    print(f"{path.name}: L={c['ATTN_TILE_L']} DH={c['DH']} MAXSC={c['MAXSC']} M={c['M']} | declares "
          + (", ".join(f"{k} {fmt_b(v)}" for k, v in sorted(decls.items())) or "no tile stage")
          + f", block LDS {fmt_b(sum(decls.values()))}")
    missing = [k for k in ("s_sc", "s_k", "s_v") if k not in decls]
    if missing:
        print("  NOT DECLARED: " + ", ".join(missing)
              + " — whatever the constants could size, this form does not stage it")
    r0 = k_stage_reach(src, c, 1)
    print("  index spellings: " + ", ".join(f"`{e}`" for e in r0["exprs"])
          + f"  -> {'sub-block slot' if r0['slot_relative'] else 'WINDOW-ABSOLUTE lane'}")
    fails = 0
    for label, span in (("a tile whose widest row is L lanes (the last span an "
                         "absolute index cannot overrun)", c["ATTN_TILE_L"]),
                        ("p300 --wave --wave-rows 64, widest row 277 lanes", 277),
                        ("engine 1024-token wave, one MAXSC-wide window", c["MAXSC"])):
        r = k_stage_reach(src, c, span)
        fails += 0 if r["index_ok"] else 1        # the INDEX claim only; the V claims print below
        print(f"  {'ok  ' if r['index_ok'] else 'FAIL'}  {label:<48} reaches {fmt_b(r['bytes_worst']):>9}"
              f"  (declared {fmt_b(r['declared'])}, block {fmt_b(r['block_lds'])})")
    v_ok = bool(r0["v_staged"] and r0["v_tile_guard"])
    print(f"  {'ok  ' if v_ok else 'FAIL'}  V is staged and a streaming tile fits it (tile <= "
          f"L={c['ATTN_TILE_L']}, `static_assert(ATTN_TILE_L >= TILE)` "
          f"{'present' if r0['v_tile_guard'] else 'ABSENT'}): "
          f"{'yes' if v_ok else ('NO V STAGE AT ALL — every pass-2 V touch is a global read' if not r0['v_staged'] else 'stage exists, unguarded')}"
          f"   (declared {fmt_b(r0['v_declared'])})")
    print(f"lds: {'PASS' if not fails and v_ok else 'FAIL'} — the stage index arithmetic "
          f"{'fits' if not fails and v_ok else 'does not fit'} "
          f"{' + '.join(sorted(r0['declared_stages'])) or 'nothing'}, so "
          f"{fmt_b(sum(r0['declared_stages'].values()))} of stage LDS is "
          f"{'really' if not fails and v_ok else 'NOT really'} what the kernel touches")
    return 1 if fails else 0


def cmd_oob_bound(args) -> int:
    """How many elements the out-of-bounds `s_k` index COULD have moved, against what §13 saw.

    §13 reported `1098790 elements moved when the lane grid was deepened` and read that as grid
    dependence. If the real cause is the stage walking past `s_k` (§14), then the damage should be
    bounded by the set of rows a wide row-tile can reach — a row-tile whose widest row is `<= L` lanes
    cannot overrun at all, so those rows are provably untouched whatever the bug does — and §13's number
    should sit just under that bound rather than somewhere in the middle of the range. That is what this
    prices, from the fixture's own geometry and the header's own constants:

        .venv/bin/python prototype/attn_tile_model.py oob-bound --observed 1098790

    `--observed` is the number from the device report; it is a named input, not a constant, so the line
    can be re-made against a different run.
    """
    dl = ROOT / "data" / "refs" / args.fixture
    layer = R.ATTN_LAYERS[0]
    seq = R.prefill_tokens(dl, layer)
    spans = R.wave_spans(seq, args.wave_rows_gate)
    L, M, DH = C["ATTN_TILE_L"], C["M"], R.DH
    rows = sum(p1 - p0 + 1 for p0, p1 in spans)
    untouched = 0
    wide_tiles = tiles = 0
    for p0, p1 in spans:
        n = p1 - p0 + 1
        lanes = [p0 + 1 + i for i in range(n)]        # lvalid = p0+1; row i sees lvalid+i lanes
        for r0 in range(0, n, M):
            hi = min(n, r0 + M)
            span = max(lanes[r] for r in range(r0, hi))
            tiles += 1
            if span <= L:
                untouched += hi - r0
            else:
                wide_tiles += 1
    # Claim 2b compares the two reduction stages, each N_Q x DH per row (`WAVE_STAGE_ROWS` in run.py
    # compares five; the two that a score change can move are attn_pre_gate and gated).
    per_row = 2 * R.N_Q * DH
    bound = (rows - untouched) * per_row
    total = rows * per_row
    print(f"layer L{layer:02d}  fixture {args.fixture}  prefill rows {seq}  waves of"
          f" {args.wave_rows_gate} -> {spans}")
    print(f"row-tiles with span > L (={L} lanes): {wide_tiles} of {tiles}; rows whose own tile is"
          f" narrow enough to be unreachable: {untouched} of {rows}")
    print(f"reduction elements the control compares: {total:,}")
    print(f"upper bound if ONLY the unreachable-by-construction rows stayed clean: {bound:,}"
          f"  ({100.0 * bound / total:.2f} % of the comparison)")
    if args.observed < 0:
        return 0
    print(f"device report                        : {args.observed:,}")
    print(f"observed / bound                   : {100.0 * args.observed / bound:.4f} %"
          f"   (bound - observed = {bound - args.observed:,} elements that happened to agree)")
    if not 0 <= args.observed <= bound:
        print("oob-bound: NOT EXPLAINED — the report moves MORE elements than an out-of-bounds stage"
              " can reach, so that diagnosis is incomplete and must not be accepted")
        return 1
    share = 100.0 * args.observed / bound
    reach = "essentially all of it" if share > 95.0 else "only part of it"
    tail = ("; the residue is small enough to be elements that coincidentally agreed" if share > 99.0
            else "")
    print(f"oob-bound: CONSISTENT — the reported movement is {share:.2f} % of what a stage that"
          f" overruns `s_k` can reach, so {reach} is explained by the overrun{tail}")
    return 0


def cmd_isa(args) -> None:
    """Compile-only resource census of both cores — `hipcc -c`, no device touched, no timing."""
    tmp = Path("/tmp/attn_tile_isa")
    tmp.mkdir(exist_ok=True)
    (tmp / "one.cpp").write_text('#include "%s"\n' % KERNEL_H)
    r = subprocess.run(["hipcc", "-O3", "--offload-arch=gfx1151",
                        "-Rpass-analysis=kernel-resource-usage", "-c", str(tmp / "one.cpp"),
                        "-o", str(tmp / "one.o")], capture_output=True, text=True, cwd=tmp)
    if r.returncode:
        raise SystemExit("COMPILE FAILED (compile-only; no device touched):\n" + r.stderr[-2500:])
    want = ("Function Name", "VGPRs:", "SGPRs:", "ScratchSize", "Occupancy", "VGPRs Spill", "LDS Size")
    cur: dict[str, dict] = {}
    for line in r.stderr.splitlines():
        if "Function Name" in line:
            name = line.split("_Z")[1].split(" ")[0].split("8")[0]
            cur[name] = {}
        for k in want[1:]:
            if k in line and cur:
                cur[list(cur)[-1]][k] = line.split(":")[-1].split("[")[0].strip()
    for name, res in cur.items():
        print(f"  {name:<16} " + "  ".join(f"{k.strip(':').replace(' ', '_')}={v}"
                                           for k, v in res.items()))
    print(f"  source's own __shared__ sum for the tile core: scores {C['score_staging_bytes']} B "
          f"+ K {C['k_stage_bytes']} B + V {C['v_stage_bytes']} B = {C['block_lds_bytes']} B. If the "
          f"descriptor says exactly that, §3's 6.3x inflation is attn_core-specific.")


def cmd_selftest(args) -> int:
    """The mirror's own invariants — the ones a wrong geometry or a wrong rounding would break. All
    CPU-only; nothing here can pass by construction, which is why several checks assert a NUMERICAL
    identity against an independent expression rather than against the model itself."""
    checks: list[tuple[str, bool]] = []
    # 1. the constants are the kernel's, and the staging arithmetic is what the header asserts.
    checks.append((f"constants read: G={G} M={M} MAXSC={MAXSC} L={C['ATTN_TILE_L']} "
                   f"threads={C['ATTN_TILE_THREADS']} warps={C['warps']}",
                   G == 4 and M == 1 and MAXSC == 512 and C["ATTN_TILE_L"] == 8
                   and C["warps"] == G))
    checks.append((f"score staging {C['score_staging_bytes']} B == G*M*MAXSC*2",
                   C["score_staging_bytes"] == G * M * MAXSC * 2))
    checks.append(("score staging fits one 64 KiB CU", C["score_staging_bytes"] <= 65536))
    # 2. the two forms share ONE window formula (the bit-identity boundary). Compare against
    #    `wave_model.py`'s independent spelling of the same split, not against this file's own.
    import wave_model
    ws = wave_model.kernel_consts()
    same = all(windows_of(n) == (n + ws["MAXSC"] - 1) // ws["MAXSC"]
               for n in (1, 8, 511, 512, 513, 1024, 4095, 4096))
    checks.append(("window count matches wave_model's split_for() at 1..4096 lanes", same))
    # 3. the exact-BF16 staging claim: bf16(x) * 2^-4 is exactly representable, so staging the score
    #    as BF16 loses nothing (the reason M=4 fits at all).
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(50000, generator=g) * torch.exp(torch.randn(50000, generator=g) * 6))
    xb = x.to(torch.bfloat16).to(torch.float32)
    checks.append(("bf16(x) * 1/16 is exact (the staging is lossless)",
                   bool(((xb * 0.0625).to(torch.bfloat16).to(torch.float32) == xb * 0.0625).all())))
    # 4. bf16() here is torch's round-to-nearest-even, not a truncation.
    rnd = np.float32(1.0078127)
    checks.append(("bf16() rounds to nearest even", float(bf16(rnd)) == float(torch.tensor(float(rnd))
                                                                              .to(torch.bfloat16))))
    # 5. `attn_combine`'s merge of ONE partial must be the identity, because a window a row has no
    #    lanes in emits nothing and the merge is then over the windows that did write.
    acc = np.arange(DH, dtype=np.float32)
    _, l_one, merged = merge_partials([(3.0, 2.0, acc)])
    checks.append(("merging one partial is the identity except for the epilogue's divide by l "
                   "(empty windows may emit nothing)",
                   bool(np.allclose(merged / np.float32(l_one), acc / np.float32(2.0), atol=0))
                   and l_one == 2.0))
    # 6. the poison tripwire is not vacuous: a read past the built prefix lands on NaN.
    kd = torch.zeros(N_KV, 16, DH); vd = torch.zeros(N_KV, 16, DH)
    doc = CacheDoc(kd, vd, valid=4, cap=16)
    doc.read("k", 0, 3); doc.read("k", 0, 9)
    checks.append(("a read past the built prefix poisons (the mask-leak witness can fail)",
                   doc.poison_rows() == 1))
    # 7. the traffic claim, taken from the header's own staging loops rather than from a number
    #    assumed here. `stage_copies` reads which shape the kernel has — a per-pair stage fetches each
    #    lane row G times per block and so re-uses NOTHING that counts as traffic, a block-level stage
    #    fetches it once — and the re-use the census may print is whatever that leaves.
    src_h = KERNEL_H.read_text()
    st = stage_copies(src_h)
    wf_src = tile_window_form(src_h)
    lanes = [i + 1 for i in range(277)]
    sh_k, sh_v = shipped_reads(lanes, 277)
    ti_k, ti_v = tiled_reads(lanes, 277, M, k_copies=st["copies"], v_copies=st["copies"],
                             window_form=wf_src)
    pp_k, pp_v = tiled_reads(lanes, 277, M, k_copies=st["copies"], v_copies=G, window_form=wf_src)
    ratio = (sh_k + sh_v) / (ti_k + ti_v)
    checks.append((f"re-use as written = {ratio:.2f}x: the {st['level']}-level stage fetches each lane "
                   f"row {st['copies']}x per block, so the honest claim is {st['reuse']}x "
                   f"({'one load shared by the group' if st['copies'] == 1 else 'NOTHING beyond L1'})",
                   abs(ratio - st["reuse"]) < 1e-9 and st["reuse"] == N_Q // (N_KV * st["copies"])))
    checks.append((f"leaving V per-pair while hoisting K would give only {(sh_k + sh_v) / (pp_k + pp_v):.2f}x "
                   f"against {st['reuse']}x: V is half the byte wall, so half-staged buys a fifth of "
                   f"the lever, not half of it",
                   ratio > (sh_k + sh_v) / (pp_k + pp_v) + 0.5))
    # 7b. the three `__shared__` declarations, sized from their OWN bracket factors instead of from a
    #     number retyped here. The §13/§14 overrun walked off an array whose size is stated in this same
    #     file, so a declaration that disagrees with the constants makes the whole LDS budget fiction.
    #     `ATTN_TILE_L >= TILE` is the companion claim for the V stage: a staged streaming tile must fit
    #     the array it is staged into, and overrunning an LDS array does not fault, it corrupts a sibling.
    sized = stages_declared(src_h, C)
    want = {"s_sc": C["score_staging_bytes"], "s_k": C["k_stage_bytes"], "s_v": C["v_stage_bytes"]}
    checks.append(("every `__shared__` array is the size its own constants say ("
                   + ", ".join(f"{k}={sized.get(k, -1):,} B" for k in want)
                   + "; V-stage guard `ATTN_TILE_L >= TILE` "
                   + ("present" if V_STAGE_GUARD.search(src_h) else "MISSING") + ")",
                   all(sized.get(k) == v for k, v in want.items())
                   and bool(V_STAGE_GUARD.search(src_h))))
    # 8. the K stage indexes its own L slots, so its LDS footprint is L rows deep and not span rows
    #    deep. Read off the source, so this fails when the kernel regresses to a window-absolute
    #    index (the §13 defect); the p300 tile and a full MAXSC window are both priced.
    src = KERNEL_H.read_text()
    for span in (277, MAXSC):
        r = k_stage_reach(src, C, span)
        checks.append((f"K stage reaches {fmt_b(r['bytes_worst'])} for a {span}-lane span, within the "
                       f"{fmt_b(r['declared'])} `s_k` reserves (spelling {r['exprs']})", r["ok"]))
    fails = 0
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
        fails += 0 if ok else 1
    print(f"selftest: {len(checks) - fails}/{len(checks)} passed")
    return 1 if fails else 0


# The injections the mirror can express, each with the set it MUST move and the set it must NOT, at
# (row, head) granularity. The sets are DERIVED from the geometry by `neg_expect()` rather than
# hand-listed, for the reason `mlp_gemv` writes up: a negative that cannot fail is worse than none,
# and one that catches everything measures nothing. A hand-listed set can rot into "everything".
NEG_TILE = {
    "readpast":  "NEG_READPAST: attend `cap` lanes, the causal mask is gone",
    "dropcurrent": "NEG_DROPCUR: attend lanes < pos, the appended lane is forgotten (§05.7)",
    "gqaoff":    "NEG_GQA: the whole group reads the neighbouring KV group (§05.5 kappa)",
    "dimhalf":   "NEG_DIMHALF: the score dot accumulates 128 of the 256 dims",
    "gqa_head0": "one head of the group reads the neighbouring KV group: the shared-load wiring test",
    "headcarry": "a head seeded from the PREVIOUS HEAD's softmax state: §07.3's bug in the shape the "
                 "tiled form gives it, where a group's four heads are warps of ONE block",
    "partalias": "the split-partial slot keyed by head-in-group instead of by head: the two KV groups "
                 "of one row then write one address (§19's failure, §20's root cause)",
}


def neg_expect(neg: str, lanes: list[int], n: int, cap: int) -> tuple[set, set]:
    """{(row, head)} that must move, and the subset that must not, for one injection on one wave.

    Derived from the geometry rather than hand-listed, for the reason `mlp_gemv` writes up: a negative
    that cannot fail is worse than none, and one that catches everything measures nothing — and a
    hand-listed set rots into "everything" the moment the fixture changes. Three derivations are worth
    naming because each came out of a FAIL or a near-miss rather than out of a guess:
      * `readpast` cannot move a row that already attended `cap` lanes — the mask was not a
        constraint there, so the bug is invisible by construction, not by weakness.
      * No score-only bug can move a row with exactly ONE lane: softmax of a singleton is 1.0
        whatever the score arrives as, so `dimhalf` is structurally blind on row 0 of every span.
      * `partalias` cannot move a row the staging fits: a one-pass row finishes inside its own block
        and writes NO partial, so there is no slot to steal. That is the §19 signature read backwards —
        every device failure was above MAXSC and nothing below it moved.
    A control that is blind somewhere must say so, or a reader counts it as coverage.
    """
    all_rh = {(i, h) for i in range(n) for h in range(N_Q)}
    wins = windows_of(max(lanes))
    one_pass_rows = [i for i in range(n) if route_of(lanes[i], wins) == 1]
    stream_rows = [i for i in range(n) if route_of(lanes[i], wins) > 1]
    if neg == "readpast":
        still = {(i, h) for i in range(n) if lanes[i] >= cap for h in range(N_Q)}
        return all_rh - still, still
    if neg == "dimhalf":
        blind = {(i, h) for i in range(n) if lanes[i] == 1 for h in range(N_Q)}
        return all_rh - blind, blind
    if neg in ("dropcurrent", "gqaoff"):
        return all_rh, set()      # gqaoff: N_KV=2 and the offset is 1 mod 2, so both groups move
    if neg == "gqa_head0":
        return {(i, 0) for i in range(n)}, {(i, h) for i in range(n) for h in range(1, N_Q)}
    if neg == "headcarry":
        # M=1 makes a block one row's G heads of ONE KV group, so the leak this control injects — a
        # softmax state that outlived its owner — can only reach head h from head h-1 INSIDE that group.
        # A group's first head (h % G == 0) has no predecessor in its own block and CANNOT move, and no
        # carry can reach another row, because another row is another block. Both routes are injected,
        # so the route is not part of the localisation: what is under test is the block boundary, which
        # is the exact thing §12's row tiling put in jeopardy.
        must = {(i, h) for i in range(n) for h in range(N_Q) if h % G}
        return must, all_rh - must
    if neg == "partalias":
        # The model resolves an aliased slot the way block order does (grid.x fastest, so the higher KV
        # group lands last and its bytes survive), which makes group 0's heads wrong and group 1's
        # heads right. On a device the winner is whichever CU landed last, i.e. arbitrary — the
        # weaker, still-derived claim there is "at least one head of each pair {h, h+G} moves", and
        # `cmd_compare` checks the exact-witness half: the stolen slot reads back as the OTHER group's
        # answer, bit for bit.
        must = {(i, h) for i in stream_rows for h in range(G)}
        blind = ({(i, h) for i in one_pass_rows for h in range(N_Q)}
                 | {(i, h) for i in stream_rows for h in range(G, N_Q)})
        return must, blind
    raise KeyError(neg)


def pin_witness(case: dict, dl: Path, layer: int) -> int:
    """Elements of one wave span that `reference.attend`'s OWN pinned form gets wrong — which must be
    zero, and is (0 of 131,072 on p300 L03 rows 0-63).

    This is the operand-pairing witness, and it is the reason the mirror's own pin diff can be read
    as arithmetic floor rather than as a bad fixture: `reference.attend` is the same torch chain that
    produced the dumps, so if THIS is 0 then the positions, the pre-append cache and the row/head
    layout this whole gate is built on are right, and any residue in the mirror below is the ordering
    floor of §10.4 and nothing else. It costs a few seconds per wave; it is the cheapest thing in the
    gate that could catch a §10.5 mis-pairing."""
    p0, p1 = case["span"]
    n = p1 - p0 + 1
    kc = R.pin(dl, "prefill.L%02d.cache_keys" % layer).reshape(N_KV, -1, DH)
    vc = R.pin(dl, "prefill.L%02d.cache_values" % layer).reshape(N_KV, -1, DH)
    got = torch.zeros(n, N_Q, DH, dtype=torch.bfloat16)
    for i in range(n):
        # The cache row i reads: the carried prefix below p0, plus this wave's own EARLIER rows — the
        # thing a per-row case cannot test at all (§06.3), and what NEG_SLOTK is about.
        kpre = torch.full((N_KV, p0 + i, DH), 1.0, dtype=torch.bfloat16)
        vpre = torch.full((N_KV, p0 + i, DH), 1.0, dtype=torch.bfloat16)
        kpre[:, :p0] = kc[:, :p0]
        vpre[:, :p0] = vc[:, :p0]
        if i:
            kpre[:, p0:p0 + i] = case["ref"]["k_rope"][:i].permute(1, 0, 2)
            vpre[:, p0:p0 + i] = case["ref"]["v"][:, :i]
        got[i] = R.attend(q_rope=case["ref"]["q_rope"][i], k_rope=case["ref"]["k_rope"][i],
                          v=case["ref"]["v"][:, i], kcache=kpre, vcache=vpre, pos=p0 + i,
                          softmax="full")
    return int(bf16_ulp.report(got, case["ref"]["attn_pre_gate"])["ndiff"])


# ============================================================================================
# Cases that reach past MAXSC. No pinned prompt on this box has 512 tokens, so a straddling wave
# has to be BUILT, and the honest way to build it says so in its own name.
# ============================================================================================
def wide_case(dl: Path, layer: int, nrows: int, start: int = 0) -> dict:
    """A SYNTHETIC-WIDE wave: `nrows` query rows at positions `start`..`start+nrows-1`, whose per-row
    operands are the fixture's pinned L03 stages cycled until they fill the wave.

    `start` exists so a wave can be placed ON the one boundary that matters: `start = MAXSC-1` puts its
    first row at exactly 512 lanes (one-pass, no partial) and its second at 513 (two blocks, two
    partials), which is the tile-boundary case §19 could not name because no pinned prompt is that
    long. A straddling wave at `start = 0` prices the geometry; a straddling wave at the boundary
    prices the ROUTE CHOICE.

    Why it is legitimate and where its limit is. Every q_rope/k_rope/v/gate row is a real pinned
    tensor of this model — nothing is invented — and the geometry (one row per position, a causal
    prefix, a per-KV-group cache the wave appends itself) is the engine's. What is NOT real is the
    content: row 512 of this wave is a reused prompt row, so its output is not HF ground truth and
    no pin exists for it (`data/refs/wide{640,1024}f` pin logits rows and decode steps, not
    per-layer prefill stages — route A hit the same gap). So this case prices GEOMETRY, ADDRESSING and
    REDUCTION ORDER, which is what §19's failure was; the numeric anchor for the streaming route is
    the real >MAXSC decode step in `decode_steps()` below, not this.
    """
    seq = R.prefill_tokens(dl, layer)
    base = R.wave_cases(dl, layer, [(0, seq - 1)])[0]
    idx = [j % seq for j in range(nrows)]
    cap = -(-(start + nrows) // R.CAP_ROUND) * R.CAP_ROUND
    cap = max(R.CAP_ROUND, cap)
    zeros = torch.zeros(N_KV, cap, DH, dtype=torch.bfloat16)
    return {"name": f"synthetic-wide{max(nrows,seq)}.L{layer:02d}"
            + (f"@{start}" if start else ""), "kind": "wave", "pos": start,
            "lvalid": start + 1, "nrows": nrows, "cap": cap, "synthetic": True,
            "qg": base["qg"].reshape(seq, N_Q, 2 * DH)[idx].reshape(-1),
            "kcache": zeros.reshape(-1), "vcache": zeros.reshape(-1),
            "ref": {"q_rope": base["ref"]["q_rope"][idx], "k_rope": base["ref"]["k_rope"][idx],
                    "v": base["ref"]["v"][:, idx], "attn_pre_gate": None},
            "span": (start, start + nrows - 1)}


def decode_steps(dl: Path, layer: int, steps=None) -> list:
    """Real pinned decode steps of a wide fixture, re-shaped as one-row waves.

    This is the only HF-anchored >MAXSC attention data on the box: `wide1024f`'s eight decode steps
    sit at positions 1024..1031, so each is a query row of 1025..1032 lanes — above MAXSC, streaming,
    with `attn_out_pre_gate` pinned by HF and the whole cache pinned. A single-row wave is the same
    per-(row,head) arithmetic with s_row = split, so it witnesses the streaming reductions against
    ground truth in a way no synthetic span can.
    """
    out = []
    for step in (steps or [f"decode{i}" for i in range(8)]):
        try:
            c = R.decode_case(dl, step, layer)
        except FileNotFoundError:
            continue
        out.append({"name": c["name"] + "(wide)", "kind": "wave", "pos": c["pos"],
                    "lvalid": c["lvalid"], "nrows": 1, "cap": c["cap"],
                    "qg": c["qg"], "kcache": c["kcache"], "vcache": c["vcache"],
                    "ref": {"q_rope": c["ref"]["q_rope"][None], "k_rope": c["ref"]["k_rope"][None],
                            "v": c["ref"]["v"], "attn_pre_gate": c["ref"]["attn_pre_gate"][None]},
                    "span": (c["pos"], c["pos"])})
    return out


def attend_witness(case: dict, rows: list[int], softmax="online") -> torch.Tensor:
    """`reference.attend` over the same operands — the independent third side. For a synthetic wave
    it cannot be bit-exact with the pins (there are none), but it is a different implementation of the
    same form, so agreement to cos ~1 says the mirror is not agreeing with itself about nothing."""
    p0 = case["span"][0]
    n = case["span"][1] - p0 + 1
    out = torch.zeros(len(rows), N_Q, DH, dtype=torch.bfloat16)
    k = torch.full((N_KV, p0 + n, DH), 1.0, dtype=torch.bfloat16)
    v = torch.full((N_KV, p0 + n, DH), 1.0, dtype=torch.bfloat16)
    for i in rows:
        kk = k.clone()
        vv = v.clone()
        kk[:, p0:p0 + i] = case["ref"]["k_rope"][:i].permute(1, 0, 2)
        vv[:, p0:p0 + i] = case["ref"]["v"][:, :i]
        got = R.attend(q_rope=case["ref"]["q_rope"][i], k_rope=case["ref"]["k_rope"][i],
                       v=case["ref"]["v"][:, i], kcache=kk[:, :p0 + i], vcache=vv[:, :p0 + i],
                       pos=p0 + i, softmax=softmax)
        out[rows.index(i)] = got
    return out


def form_diff(got: torch.Tensor, want: torch.Tensor) -> dict:
    """Bit-for-bit diff of two form outputs, with the per-(row,head) aggregate the floors are keyed on.

    `worst_row_1cos` is the largest per-head `1 - cos` over the whole comparison, vectorised: it is the
    quantity `engine/gate.py` judges a streaming row by (`CASCADE_BAND["row_1cos_max"]`), so a cause
    can be priced against the published floor instead of described. Zero rows are defined to agree
    (cos 1), which is what a masked-out row means."""
    g_bf, w_bf = got.to(torch.bfloat16), want.to(torch.bfloat16)
    rep = bf16_ulp.report(g_bf, w_bf)
    per_head_diff = (g_bf != w_bf).any(-1)
    g2 = g_bf.float().reshape(-1, DH)
    w2 = w_bf.float().reshape(-1, DH)
    denom = g2.norm(dim=-1) * w2.norm(dim=-1)
    cos = torch.where(denom > 0, (g2 * w2).sum(-1) / torch.where(denom > 0, denom, torch.ones_like(denom)),
                      torch.ones_like(denom))
    return {"ndiff": rep["ndiff"], "n": rep["n"], "cos": rep["cos"], "meanrel": rep["meanrel"],
            "pairs": [tuple(x) for x in torch.nonzero(per_head_diff).tolist()],
            "worst_row_1cos": float((1.0 - cos).max())}


def cmd_compare(args) -> int:
    """The two mirrored forms, on spans whose rows straddle MAXSC, with the partial buffer modelled.

    Four things, in this order:
      1. what the mirror believes the kernel IS, read out of the header (window rule + partial layout
         + LDS reach), because a model that retypes the geometry can agree with itself forever;
      2. the pinned spans at or below MAXSC — the §12 gate claim, re-printed so a regression here
         cannot hide behind the new material;
      3. the straddling wave and the real >MAXSC decode steps: per-form geometry, alias report, and the
         per-row detail of anything that differs;
      4. the attribution table, which prices the two candidate causes SEPARATELY (window grouping vs
         partial aliasing) so "why" is a number and not a story.
    """
    src = KERNEL_H.read_text()
    lay, wform = part_layout(src), tile_window_form(src)
    reach = k_stage_reach(src, C, C["MAXSC"])
    print(f"mirror models the kernel as: lane cut {wform}, partial layout {lay['ok'] and 'group-dimensioned' or 'group-COLLAPSED'}"
          f" ({len(lay['sites'])} `a.part +` sites, multipliers "
          f"{sorted({s['mult'] for s in lay['sites']})}), LDS reach {fmt_b(reach['bytes_worst'])}"
          f" of {fmt_b(reach['declared'])} K-stage declared"
          + (f" + {fmt_b(reach['v_declared'])} V-stage" if reach["v_staged"]
             else " and NO V STAGE")
          + f" (block {fmt_b(reach['block_lds'])})")
    if not lay["ok"]:
        print("  ALIASING: " + "; ".join(f"row `{r}` * `{m}` + head `{h}`" for r, m, h in lay["collapsed"])
              + " — no KV-group term, so the two groups of one row write one slot")
    fails: list[str] = []

    # ---- 1. the pinned spans at or below MAXSC
    dl = ROOT / "data" / "refs" / args.fixture
    layer = R.ATTN_LAYERS[0]
    seq = R.prefill_tokens(dl, layer)
    print(f"\n[1] pinned spans, {dl.name} L{layer:02d} ({seq} rows, widest row {seq} lanes"
          f" — nothing here reaches MAXSC={MAXSC}, which is exactly why §12's gate was green):")
    for p0, p1 in R.wave_spans(seq, args.wave_rows_gate)[:args.waves]:
        case = R.wave_cases(dl, layer, [(p0, p1)])[0]
        t, s = wave_out(case, "tile"), wave_out(case, "shipped",
                                               layout="group-dimensioned", window_form="per-row-slice")
        d = form_diff(t["attn_pre_gate"], s["attn_pre_gate"])
        print(f"  wave {p0:>4}-{p1:<4} lanes {t['lanes'][0]:>4}-{t['lanes'][-1]:<4} "
              f"forms differ {d['ndiff']:>4}/{d['n']:,} poison {t['doc'].poison_rows()}")
        if d["ndiff"] or t["buf"].aliases():
            fails.append(f"pinned wave {p0}-{p1}: {d['ndiff']} differing elements"
                         f" / {len(t['buf'].aliases())} aliased partial slots")

    # ---- 2./3. the straddling wave, and the two causes priced apart
    nrows = args.wide
    case = wide_case(dl, layer, nrows)
    # The shipped side is modelled with the shipped form's OWN spelling — `attn_core` is frozen and §19
    # proved it reproduces HF on all 1024 rows, so its lane cut and its slot expression are the
    # reference, whatever the tile form's header currently says. Modelling it with the tile form's
    # layout (which is what the first version of this command did) compared two broken things and
    # reported the fixed geometry as the one that moved 512 rows.
    ship = wave_out(case, "shipped", layout="group-dimensioned", window_form="per-row-slice")
    lanes = ship["lanes"]
    n_stream = sum(1 for x in lanes if x > MAXSC)
    print(f"\n[2] SYNTHETIC-WIDE wave of {nrows} rows (pinned L{layer:02d} stages cycled; geometry and"
          f" addressing real, content reused — NOT HF ground truth). Rows above MAXSC: {n_stream};"
          f" launch windows/split = {ship['windows']}.")
    print(f"    boundary rows priced by BOTH forms: " + "; ".join(
        f"row {r-1}: {r} lanes -> {route_of(r, ship['windows'])} block"
        f"{'' if route_of(r, ship['windows']) == 1 else 's'} {row_bounds(r, route_of(r, ship['windows']))[:2]}"
        for r in (MAXSC, MAXSC + 1)))
    combos = [("per-row-slice", "group-dimensioned", "as the kernel now is (§20)"),
              ("fixed-window", "group-dimensioned", "§12's window grouping ALONE"),
              ("per-row-slice", "group-collapsed", "the aliasing ALONE"),
              ("fixed-window", "group-collapsed", "§12 exactly — what §19 ran on the device")]
    print("\n[3] attribution — the same operands, the two candidate causes priced separately:")
    print(f"    {'lane cut / partial layout':<38} {'rows moved':>11} {'elements moved':>15}"
          f" {'worst row 1-cos':>16}  {'gate floor':>10}")
    band = None
    try:
        band = _load_gate_band()["row_1cos_max"]
    except Exception:                                     # a missing hipcc-free import must not hide
        pass                                              # the numbers, only the verdict word
    for wf, ly, label in combos:
        tile = wave_out(case, "tile", layout=ly, window_form=wf)
        d = form_diff(tile["attn_pre_gate"], ship["attn_pre_gate"])
        verdict = ("inside" if band and d["worst_row_1cos"] <= band else "OUTSIDE") if band else "-"
        print(f"    {wf + ' / ' + ly:<38} {len({i for i, _ in d['pairs']}):>7}/{len(lanes):<3}"
              f" {d['ndiff']:>10}/{d['n']:,} {d['worst_row_1cos']:>16.3e}  {verdict:>10}")
        if label.startswith("as the kernel now is"):
            if d["ndiff"]:
                for i, h in d["pairs"][:8]:
                    print(f"      row {i} head {h}: {lanes[i]} lanes, form "
                          f"{'one-pass' if route_of(lanes[i], ship['windows']) == 1 else 'streaming'}"
                          f" — differs, which is a defect, not a band")
            if tile["buf"].aliases():
                fails.append(f"the kernel's own partial layout aliases {len(tile['buf'].aliases())}"
                             " slots — two blocks wrote one address")
            print(f"      aliases in the kernel's own layout: {len(tile['buf'].aliases())}"
                  f" (of {len(tile['buf'].writes)} distinct slots written)")
        if wf == "fixed-window" and ly == "group-collapsed":
            ali = tile["buf"].aliases()
            print(f"      aliased slots: {len(ali)}; example writers "
                  f"{ali[0][1][:4] if ali else '-'}")
            # Localisation: with the collapsed layout the merge reads ONE slot for both groups, so
            # every streaming row's head h comes out bit-identical to head h+G. That cannot happen in
            # a correct engine (different q, different answer), so it is the witness that the address
            # — not the rounding — moved the row, and it is exact rather than a band argument.
            same = sum(1 for i in range(len(lanes)) if lanes[i] > MAXSC
                       and all(bool((tile["attn_pre_gate"][i, h].to(torch.bfloat16)
                                     == tile["attn_pre_gate"][i, h + G].to(torch.bfloat16)).all())
                               for h in range(G)))
            other = sum(1 for i in range(len(lanes)) if lanes[i] > MAXSC
                        and bool((tile["attn_pre_gate"][i, 0].to(torch.bfloat16)
                                  == ship["attn_pre_gate"][i, G].to(torch.bfloat16)).all()))
            print(f"      streaming rows whose 4 head-pairs came out BIT-IDENTICAL to each other: "
                  f"{same}/{n_stream}   (group 0's answer == group 1's, bit for bit, in "
                  f"{other}/{n_stream} of them: the stolen slot is the other group's)")
            below = sum(1 for i in range(len(lanes)) if lanes[i] <= MAXSC
                        and int((tile["attn_pre_gate"][i].to(torch.bfloat16)
                                 != ship["attn_pre_gate"][i].to(torch.bfloat16)).sum()))
            print(f"      rows at or below MAXSC touched by the aliasing: {below}"
                  f"  — a one-pass row writes no partial, so it CANNOT be hit; that is the derived")
            print("      must-not-catch set, and it is why §19's failing rows were all above 512.")

    # ---- 4. the real >MAXSC anchor: HF-pinned decode rows, above the staging, with a pin
    for name in args.anchor:
        p = Path(name) if Path(name).is_absolute() else ROOT / "data" / "refs" / name
        steps = decode_steps(p, layer)
        if not steps:
            print(f"\n[4] anchor {name}: no per-layer decode pins (expected for wide640f/wide1024f"
                  " if only logits were dumped)")
            continue
        print(f"\n[4] real >MAXSC anchor {p.name} L{layer:02d}: {len(steps)} pinned decode steps,"
              f" lanes {steps[0]['span'][0] + 1}..{steps[-1]['span'][0] + 1} — HF ground truth above")
        print("    the staging: the two FORMS are judged bit-for-bit against each other, and the pins")
        print("    say whether that answer is the reference's answer.")
        for c in steps:
            t = wave_out(c, "tile")
            s = wave_out(c, "shipped", layout="group-dimensioned", window_form="per-row-slice")
            d = form_diff(t["attn_pre_gate"], s["attn_pre_gate"])
            rep = bf16_ulp.report(s["attn_pre_gate"].to(torch.bfloat16),
                                  c["ref"]["attn_pre_gate"])
            band = _load_gate_band()
            lanes_here = c["span"][0] + 1
            deficit = 1.0 - rep["cos"]
            if lanes_here <= MAXSC:
                bar = "one-pass ordering floor (cos >= 0.9999999, meanrel <= 1e-5)"
                bad = rep["cos"] < 0.9999999 or rep["meanrel"] > 1e-5
                tail = "  <-- OUTSIDE " + bar if bad else "  inside " + bar
            else:
                # Streaming against HF: REPORTED, not judged. `engine/attn_form_band.py`'s
                # `verdict_lines` states the published rule for this comparison and it is explicit —
                # above MAXSC the row is judged by `row_1cos_max`, and "meanrel and the ULP columns are
                # REPORTED and not judged: `stage_meanrel_max` is published per stage tile rather than
                # per logits row ... no per-row meanrel or ULP bound exists for a reassociating route,
                # so inventing one here would be the widening this script exists to refuse". Applying
                # `stage_meanrel_max` per row, which an earlier version of this line did, turned a legal
                # §07.3 row into a red one by inventing a per-row bound.
                bar = (f"no per-row bound is published for a streaming row against HF; "
                       f"1-cos {deficit:.3e} printed against `row_1cos_max` "
                       f"{band['row_1cos_max']:.0e} for scale only")
                bad = False
                tail = "  [mirror-vs-pin reported, not judged]"
            print(f"    {c['name']:<24} lanes {lanes_here:>5} s_row "
                  f"{route_of(lanes_here, windows_of(lanes_here))} "
                  f"| forms differ {d['ndiff']:>4}/{d['n']:,} | mirror-vs-pin ndiff {rep['ndiff']:>4}"
                  f"/{rep['n']:,} cos {rep['cos']:.8f} meanrel {rep['meanrel']:.1e}{tail}")
            if d["ndiff"]:
                # §20 raised the tiled form's claim from "banded" to "bit-identical on both routes", so
                # a non-zero difference here is a defect in the form, not a band to be argued about.
                fails.append(f"{c['name']}: the two forms differ in {d['ndiff']} elements at "
                             f"{lanes_here} lanes — §20's claim is bit-identity on the streaming route"
                             f" too, because the tiled form now uses `blocks_for_row`'s own slice")
            if bad:
                fails.append(f"{c['name']}: the mirror's shipped form is off the pins at cos "
                             f"{rep['cos']:.8f} meanrel {rep['meanrel']:.1e}, outside {bar}")
    print(f"\ncompare: {len(fails)} failures")
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


GateBand = None


def _load_gate_band() -> dict:
    """The published cascade band, imported from `engine/gate.py` rather than retyped (the
    `attn_form_band.py` rule: no new numbers in a judge)."""
    global GateBand
    if GateBand is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_atm_gate", ROOT / "engine" / "gate.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_atm_gate"] = mod
        spec.loader.exec_module(mod)
        GateBand = mod.CASCADE_BAND
    return GateBand


def cmd_gate(args) -> int:
    """The CPU-only gate: the tile form's bits against the shipped form's, both against the pins, and
    the injections each caught only where the bug lives. No device, no timing."""
    dl = ROOT / "data" / "refs" / args.fixture
    if not dl.exists():
        raise SystemExit(f"no fixture {dl}")
    layer = R.ATTN_LAYERS[0]
    seq = R.prefill_tokens(dl, layer)
    rows_sel = [int(x) for x in args.rows.split(",") if x.strip()]
    fails: list[str] = []
    all_spans = R.wave_spans(seq, args.wave_rows_gate)
    spans = [s for s in all_spans if s[0] in set(rows_sel) or not rows_sel][: args.waves]
    print(f"fixture {dl.name} L{layer:02d} ({seq} pinned rows) | waves of {args.wave_rows_gate} "
          f"rows at {[s for s in spans]} | G={G} M={M} MAXSC={MAXSC}")
    ncmp = 0
    tot_pin = None
    for p0, p1 in spans:
        case = R.wave_cases(dl, layer, [(p0, p1)])[0]
        n = p1 - p0 + 1
        clean = wave_out(case, "tile")
        ship = wave_out(case, "shipped")
        pin_a = case["ref"]["attn_pre_gate"]
        poison = clean["doc"].poison_rows()
        witness = pin_witness(case, dl, layer)
        ident = int((clean["attn_pre_gate"].to(torch.bfloat16)
                     != ship["attn_pre_gate"].to(torch.bfloat16)).sum())
        ncmp += clean["attn_pre_gate"].numel()
        rep = bf16_ulp.report(clean["attn_pre_gate"].to(torch.bfloat16), pin_a)
        part = {k: rep[k] for k in ("ndiff", "n", "n_band")}
        tot_pin = part if tot_pin is None else {k: tot_pin[k] + part[k] for k in part}
        chain_bad = (rep["cos"] < 0.9999999 or rep["meanrel"] > 1e-5)
        print(f"  wave {p0:>3}-{p0 + n - 1:<3} rows {n:>3} lanes {clean['lanes'][0]:>4}-"
              f"{clean['lanes'][-1]:<4} windows {clean['windows']} "
              f"K/V rows read {len(clean['doc'].reads):>7,} poison {poison} "
              f"| ref-vs-pin {witness:>3} | tile-vs-shipped {ident:>3} | mirror vs pin ndiff "
              f"{rep['ndiff']:>4}/{rep['n']:,} cos {rep['cos']:.8f} meanrel {rep['meanrel']:.1e}")
        if witness:
            fails.append(f"wave {p0}-{p0 + n - 1}: reference.attend's pinned form differs from the "
                         f"pins in {witness} elements. The FIXTURE is mis-paired (§10.5), so nothing "
                         "else printed for this wave means anything")
        if poison:
            fails.append(f"wave {p0}-{p0 + n - 1}: {poison} K/V reads landed on poison — the "
                         "tiled form read a lane some row may not attend")
        if ident:
            fails.append(f"wave {p0}-{p0 + n - 1}: the tiled form differs from the shipped form in "
                         f"{ident} elements. Bit-identity is the §5 contract: either a reduction "
                         "order changed, or the mirror and the kernel disagree")
        if chain_bad:
            fails.append(f"wave {p0}-{p0 + n - 1}: the mirror is {rep['ndiff']} elements off the "
                         f"pins at cos {rep['cos']:.8f} meanrel {rep['meanrel']:.1e}, outside the "
                         "pre-registered ordering floor (cos >= 0.9999999, meanrel <= 1e-5). A real "
                         "arithmetic bug moves thousands of elements and drops the cosine; a 1-ULP "
                         "ordering flip does not")
    print(f"\nmirror vs pins: ndiff {tot_pin['ndiff']}/{tot_pin['n']} "
          f"({100 * tot_pin['ndiff'] / tot_pin['n']:.4f} %), cancellation band excused "
          f"{tot_pin['n_band']}. Those are ULP flips from the mirror summing `l` and `acc` serially "
          "in lane order — which is what the KERNEL does, and is why the pinned torch softmax and the "
          "engine route differ at all (§10.4, A-18). Per AGENTS.md the chain aggregate is the judge "
          "here, not ndiff: `reference.attend` reproduces the pins at ndiff 0 above, so the operands "
          "are right and the residue is order. The claim this gate makes is tile-vs-shipped bit "
          "identity, which is exact.")
    if args.negative:
        print("\nnegative controls (each caught only where the bug lives; `blind` is the derived set")
        print("the injection structurally cannot see, which must stay untouched and is why `caught`")
        print("is not expected to be everything). Span 0 is a whole prompt in one wave; span 1 is a")
        print("carried prefix, which is where a §05.7 append bug can live. The wide anchors are the")
        print("only rows above MAXSC on this box, and without them `partalias` has NO must-catch set:")
        print("every pinned prompt is under the staging, so a streaming-route control run only on")
        print("pinned spans is a check that cannot fail — the defect `mlp_gemv` and `mixed_bw` report.")
        neg_cases = [(f"span {sp[0]}-{sp[1]}", R.wave_cases(dl, layer, [sp])[0])
                     for sp in spans[: args.neg_waves]]
        for anc in args.neg_wide:
            ap = Path(anc) if Path(anc).is_absolute() else ROOT / "data" / "refs" / anc
            for c in decode_steps(ap, layer):
                neg_cases.append((f"{ap.name} {c['name']}", c))
        # The tile-boundary wave: rows of exactly MAXSC, MAXSC+1, MAXSC+2, MAXSC+3 lanes. Below the
        # line a row has no partial to lose, above it it has two, and the route decision is made by the
        # SAME `blocks_for_row` on both sides, so any form that disagrees with the shipped one has to
        # disagree here. This is the case §19 could not name, because no pinned prompt reaches 512.
        neg_cases.append((f"boundary {MAXSC}..{MAXSC + 3} lanes",
                          wide_case(dl, layer, 4, start=MAXSC - 1)))
        print("    cases: " + ", ".join(lbl for lbl, _ in neg_cases))
        for neg in NEG_TILE:
            tot = {"caught": 0, "blind": 0, "wrong": set(), "missing": set(), "poison": 0, "n": 0}
            for _label, case in neg_cases:
                n = case["span"][1] - case["span"][0] + 1
                clean = wave_out(case, "tile")
                inj = wave_out(case, "tile", neg=neg)
                must, must_not = neg_expect(neg, clean["lanes"], n, case["cap"])
                caught = {(i, h) for i in range(n) for h in range(N_Q)
                          if bool((inj["attn_pre_gate"][i, h].to(torch.bfloat16)
                                   != clean["attn_pre_gate"][i, h].to(torch.bfloat16)).any())}
                tot["caught"] += len(caught)
                tot["blind"] += len(must_not)
                tot["n"] += n * N_Q
                tot["wrong"] |= caught & must_not
                tot["missing"] |= must - caught
                tot["poison"] += (inj["doc"].poison_rows() - clean["doc"].poison_rows())
            print(f"    {neg:<12} {NEG_TILE[neg][:56]:<58} caught {tot['caught']:>4}/{tot['n']:<5} "
                  f"blind {tot['blind']:>4}/{tot['n']:<5} must-not-hit {len(tot['wrong']):>2} "
                  f"poison {tot['poison']:>3}{'' if not tot['wrong'] else '   <-- LOCALISED WRONG'}")
            if tot["missing"]:
                fails.append(f"negative {neg} did NOT move {len(tot['missing'])} (row,head) it must "
                             f"move: {sorted(tot['missing'])[:4]} — the control is vacuous there")
            if tot["wrong"]:
                fails.append(f"negative {neg} moved {len(tot['wrong'])} (row,head) it must not "
                             f"touch: {sorted(tot['wrong'])[:4]} — the control localises nothing")
    print(f"\n{ncmp} elements compared, {len(fails)} failures")
    for f in fails[:20]:
        print("  FAIL", f)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="geometry",
                    choices=["geometry", "traffic", "lds", "oob-bound", "isa", "selftest",
                             "gate", "compare"])
    ap.add_argument("--observed", type=int, default=-1,
                    help="for oob-bound: the element count a device report printed, to price the"
                         " diagnosis against it; -1 prints only the bound")
    ap.add_argument("--kernel", default="",
                    help="price the LDS census against another copy of the kernel header "
                         "(for a before/after); empty is the live one")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--wave-rows", type=int, default=512)
    ap.add_argument("--fixture", default="p300")
    ap.add_argument("--wave-rows-gate", type=int, default=64,
                    help="rows per wave for the CPU gate; 64 matches run.py --wave")
    ap.add_argument("--waves", type=int, default=99, help="how many wave spans to price")
    ap.add_argument("--neg-waves", type=int, default=2,
                    help="spans to run each negative over: span 0 is one prompt, span 1 is a carry")
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--neg-wide", action="append", default=["wide640f", "wide1024f"],
                    help="wide fixtures whose pinned decode steps also join the negative cases. "
                         "They are the only rows above MAXSC on this box: without them the "
                         "streaming-route controls have an empty must-catch set and cannot fail.")
    ap.add_argument("--wide", type=int, default=1024,
                    help="for compare: rows in the SYNTHETIC-WIDE wave (positions 0..n-1, pinned"
                         " per-row stages cycled); 1024 puts body 1's rows above MAXSC")
    ap.add_argument("--anchor", action="append", default=["wide1024f", "wide640f", "wide520f"],
                    help="for compare: a fixture whose pinned DECODE steps sit above MAXSC "
                         "(wide1024f, wide640f). Repeatable.")
    ap.add_argument("--rows", default="",
                    help="restrict the gate to wave spans starting at these row indices; empty "
                         "(the default) takes the first --waves spans")
    args = ap.parse_args()
    if args.cmd == "geometry":
        cmd_geometry(args)
    elif args.cmd == "traffic":
        cmd_traffic(args)
    elif args.cmd == "lds":
        return cmd_lds(args)
    elif args.cmd == "oob-bound":
        return cmd_oob_bound(args)
    elif args.cmd == "isa":
        cmd_isa(args)
    elif args.cmd == "selftest":
        return cmd_selftest(args)
    elif args.cmd == "compare":
        return cmd_compare(args)
    else:
        return cmd_gate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
