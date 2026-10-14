#!/usr/bin/env python3
"""attn_tile_model.py — the CPU mirror and CPU-only gate for Phase 2's tiled attention wave.

WHY THIS FILE EXISTS (and why it is not just a rerun of `run.py`). Phase 2 replaces the wave's
`attn_core` with `attn_tile_wave`, whose whole point is that ONE K row is loaded once and fed to
G x M = 16 score dots (§4/§9 of bench/attn-wave-diagnosis-2026-10-14.md: the wave reads 25.79 GB of
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
MEASURED_ATTN_MS = 109.2     # bench/performance-status.md, route-(c) attribution (the baseline)
VECTOR_FP32_TF = 2.93        # devices/strix-halo-compute.md: what the vector path sustains
HAND_GEMM_TF = 0.85          # the engine's own hand GEMMs (the rate a vector kernel really hits)
LINES = {"full": 0.0, "stream": 0.0}   # traffic lines read off the mirror, by route
POISON = float("nan")        # see `CacheDoc`: a leaked mask must poison the output, not shrink it


def kernel_consts() -> dict:
    """The kernel's own constants, read from the header (a copy here would drift). Covers both
    forms: `MAXSC`/`TILE` are the shipped form's and `ATTN_TILE_*` the tiled form's."""
    raw = dict(re.findall(r"^#define\s+(\w+)\s+(\d+)\s*(?://.*)?$", KERNEL_H.read_text(), re.M))
    vals = {k: int(v) for k, v in raw.items()}
    for name in ("N_Q", "N_KV", "DH", "TILE", "MAXSC", "ATTN_TILE_M", "ATTN_TILE_L"):
        if name not in vals:
            raise SystemExit(f"{KERNEL_H.name}: cannot read {name} — this model's copy is stale")
    vals["SCALE"] = 0.0625
    vals["G"] = vals["N_Q"] // vals["N_KV"]   # the header spells GQA as (N_Q / N_KV)
    vals["M"] = vals["ATTN_TILE_M"]
    vals["reuse"] = vals["G"] * vals["M"]
    vals["score_staging_bytes"] = vals["G"] * vals["M"] * vals["MAXSC"] * 2   # BF16 scores
    vals["k_stage_bytes"] = vals["ATTN_TILE_L"] * vals["DH"] * 2
    return vals


C = kernel_consts()
N_Q, N_KV, DH, MAXSC, TILEW = (C["N_Q"], C["N_KV"], C["DH"], C["MAXSC"], C["TILE"])
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


def half_keep() -> np.ndarray:
    """The dim mask a NEG_DIMHALF dot is equivalent to: keep dims where (d % 8) < 4."""
    keep = np.ones(DH, dtype=np.float32)
    keep[(np.arange(DH) % 8) >= 4] = 0.0
    return keep


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

    The three `seed_*` arguments exist only so the NEG_TILECARRY injection can seed a pair from the
    previous row's state — §07.3's bug in the tiled form's new shape, where consecutive queries are
    pairs of ONE block. Left alone they are the neutral (0, 0, 0) the kernel starts from."""
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


def online(sc: np.ndarray, V: torch.Tensor, blk: int) -> tuple[float, float, np.ndarray]:
    """§07.3's streaming form with `attn_core`'s EXACT tile-of-`blk` algebra (running max rescaled by
    `expf(m - mn)`, weights NOT rounded, the divide left to the merge). `reference.attend`'s
    online branch is the same form; the tile grouping is reproduced here because a different grouping
    of the same numbers is a different F32 sum, and the claim is bit-identity."""
    m = np.float32(0.0)
    l = np.float32(0.0)
    acc = np.zeros(DH, dtype=np.float32)
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


def merge_partials(partials: list[tuple[float, float, np.ndarray]]) -> tuple[float, np.ndarray]:
    """`attn_combine`'s merge, in window order."""
    m, l, acc = 0.0, np.float32(0.0), np.zeros(DH, dtype=np.float32)
    for ms, ls, avs in partials:
        mn = max(m, ms)
        r0, r1 = expf(m - mn), expf(ms - mn)
        l = np.float32(l * r0 + np.float32(ls) * r1)
        acc = acc * np.float32(r0) + avs * np.float32(r1)
        m = mn
    return m, acc


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


def block_touches(lanes: list[int]) -> int:
    """(row, head, lane) triples a launch must produce — the shipped form's work, identical here."""
    return sum(n * N_Q for n in lanes)


def tiled_reads(lanes: list[int], nrows: int, rows_per_block: int,
                k_copies: int = G, v_copies: int = G) -> tuple[int, int]:
    """How many BF16 K rows (and V rows) a whole launch asks global for, counted WITH multiplicity
    across blocks and across the pairs of a block — traffic is multiplicity, not distinctness.

    A block covers rows [r0, r0+rows_per_block) and the window [w*MAXSC, (w+1)*MAXSC). What it reads
    stops at the tile's widest row, because the causal mask is a PREFIX and a masked lane is never
    read at all — that prefix fact is what makes tiling rows legal, and the residue against a clean
    G x M division is exactly those masked tails (a tile holding rows with lane counts 1, 2, 3, 277
    reads 277 lanes, not 283).

    `k_copies`/`v_copies` are how many times one block asks for the same row, and the kernel as
    written stages K per PAIR and reads V per pair, so both are G: the four heads of a group read the
    identical rows (the writes land on the same LDS slots with the same bytes, so it is benign, and
    an L1 hit is likely within a sub-block but is NOT something a traffic claim should rest on). That
    is why the re-use this kernel predicts is M = 4x and not G x M = 16x. Hoisting the stage to block
    level — stage the group's rows once, and let each pair override only its OWN appended lane at the
    use site, which is the single lane where pairs differ — takes both to 1x and the ratio to G x M;
    it is ~15 lines in pass 1 and a pass-2 sub-block restructure, and it is the named next lever if
    the device run says bytes still bind. Pass `k_copies=1, v_copies=1` to price it.
    """
    k = v = 0
    for r0 in range(0, nrows, rows_per_block):
        rows = range(r0, min(nrows, r0 + rows_per_block))
        n_max = max(lanes[r] for r in rows)
        for w in range(windows_of(max(lanes))):
            lo, hi = w * MAXSC, min(n_max, (w + 1) * MAXSC)
            if hi > lo:
                # N_KV blocks per row tile (one per KV group, `blockIdx.x`), each asking `copies` times.
                k += N_KV * k_copies * (hi - lo)
                v += N_KV * v_copies * (hi - lo)
    return k, v


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


def wave_out(case: dict, form: str, neg: str = "none",
             doc: CacheDoc | None = None) -> dict:
    """One pinned wave span through the mirror, returning `attn_pre_gate` and `gated` laid out the
    way the kernel writes them ([rows][N_Q*DH]), plus the read census.

    `form` is "shipped" (one block per (row, head), `attn_core`) or "tile" (G heads x M rows per
    block, `attn_tile_wave`). The two differ in WHICH block loads which K row and nothing else: the
    arithmetic per pair is the same five reductions, so identical bits are the claim being tested.
    """
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
    stale: dict[int, tuple[float, np.ndarray]] = {}     # NEG_TILECARRY: the previous row's state
    for i in range(n):
        row_pos = p0 + i
        nrow = lanes[i]
        if neg == "dropcurrent":
            nrow = row_pos                       # NEG_DROPCUR: the appended lane is forgotten
        elif neg == "readpast":
            nrow = case["cap"]                   # NEG_READPAST: the causal mask is gone
        for head in range(N_Q):
            kv = head // G
            if neg == "gqaoff":
                kv = (kv + 1) % N_KV             # the whole group reads the neighbour
            elif neg == "gqa_head0":
                kv = (kv + 1) % N_KV if head == 0 else kv
            q = w["q_rope"][i, head].to(torch.float32).numpy()
            self_k = w["k_rope"][i, kv].to(torch.float32).numpy()
            self_v = case["ref"]["v"][kv, i].to(torch.float32).numpy()
            sc_all = score_stream(q, doc, kv, list(range(nrow)), row_pos, self_k, keep)
            V_all = v_stream(doc, kv, list(range(nrow)), row_pos, self_v)
            if nrow <= MAXSC:
                # One-pass, and the SAME in both forms: one block, the whole row, the reference form.
                # Every pinned row in this repo is here (the longest fixture is 277 lanes), so this is
                # the branch the CPU gate actually prices.
                seed_m, seed_l, seed_acc = (0.0, np.float32(0.0), np.zeros(DH, dtype=np.float32))
                if neg == "tilecarry" and i % M != 0 and head in stale:
                    seed_m, seed_acc = stale[head]     # §07.3's bug, between rows of one block
                    seed_l = np.float32(1.0)
                m, l, acc = one_pass(sc_all, V_all, seed_m, seed_l, seed_acc)
                stale[head] = (m, acc)
                a_pre[i, head], gated[i, head] = epilogue(acc, l, qg[i, head])
            else:
                # Above the staging both forms stream, and they do NOT group identically: the shipped
                # form cuts each row into `wins` equal slices (span = ceil(n/wins), `attn_core`'s
                # `span`), the tiled form owns fixed MAXSC-wide windows because a window is what a
                # block shares a K load over. Both are §07.3's streaming form and both sit inside its
                # pre-authorized band, but they are not bit-identical to each other — see the
                # "WHERE BIT-IDENTITY STOPS" note in the kernel header and in results.md. No pinned
                # fixture reaches 512 lanes, so the CPU gate cannot exercise this; the engine's
                # >512-lane bodies can.
                parts = []
                if form == "shipped":
                    span = (nrow + wins - 1) // wins
                    bounds = [(lo, min(nrow, lo + span)) for lo in range(0, nrow, span)]
                else:
                    bounds = [(lo, min(nrow, lo + MAXSC)) for lo in range(0, nrow, MAXSC)]
                for lo, hi in bounds:
                    parts.append(online(sc_all[lo:hi], V_all[lo:hi], TILEW))
                _, acc = merge_partials(parts)
                a_pre[i, head], gated[i, head] = epilogue(acc, 1.0, qg[i, head])
    return {"attn_pre_gate": a_pre, "gated": gated, "doc": doc,
            "rows": rows, "lanes": lanes, "windows": wins,
            "rows_per_block": 1 if form == "shipped" else M}


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
        ti_k, ti_v = tiled_reads(lanes, len(lanes), M)
        print(f"\n{tokens} tokens, {rows}-row waves: {len(lanes)} rows, "
              f"{touches:,} touches, windows={windows_of(max(lanes))}")
        print(f"  shipped  core blocks {N_Q * windows_of(max(lanes)) * len(lanes):>7}   "
              f"K/V rows read {sh_k + sh_v:>10,} = {fmt_b((sh_k + sh_v) * DH * 2)}")
        print(f"  tiled    core blocks {N_KV * windows_of(max(lanes)) * tile_windows_of(len(lanes)):>7}"
              f"   K/V rows read {ti_k + ti_v:>10,} = {fmt_b((ti_k + ti_v) * DH * 2)}")
        print(f"  re-use   {(sh_k + sh_v) / (ti_k + ti_v):.1f}x fewer K/V row reads "
              f"(G x M = {G * M}; the residue is the row tiles' widest-row extent)")
        print(f"  staging  {fmt_b(C['score_staging_bytes'])} of scores + "
              f"{fmt_b(C['k_stage_bytes'])} of K = {fmt_b(C['score_staging_bytes'] + C['k_stage_bytes'])}"
              f" per block; 64 KiB/CU admits "
              f"{int(65536 // (C['score_staging_bytes'] + C['k_stage_bytes']))} block(s)/CU")


def cmd_traffic(args) -> None:
    """The §7 pre-registered numbers, from the closed-form census (no per-element work)."""
    lanes, lvalid0 = [], 1
    for pos0 in range(0, args.tokens, args.wave_rows):
        nr = min(args.wave_rows, args.tokens - pos0)
        lanes += [pos0 + lvalid0 + i for i in range(nr)]
    nrows = len(lanes)
    touches = block_touches(lanes)
    flops = touches * 4 * DH
    sh_k, _ = shipped_reads(lanes, nrows)
    ti_k, _ = tiled_reads(lanes, nrows, M)
    bs_k, _ = tiled_reads(lanes, nrows, M, k_copies=1, v_copies=1)
    sh_bytes = sh_k * DH * 2 * 2
    ti_bytes = ti_k * DH * 2 * 2
    bs_bytes = bs_k * DH * 2 * 2
    layers = 6
    print(f"{args.tokens} tokens x {layers} attention layers, {args.wave_rows}-row waves")
    print(f"  touches          {touches * layers:,}   flops {flops * layers / 1e9:.2f} GFLOP "
          f"(the ledger's 25.79)")
    print(f"  shipped K/V read {fmt_b(sh_bytes * layers)}  = "
          f"{sh_bytes * layers / (flops * layers):.3f} byte/flop")
    print(f"  tiled   K/V read {fmt_b(ti_bytes * layers)}  = "
          f"{ti_bytes * layers / (flops * layers):.3f} byte/flop   "
          f"({sh_bytes / ti_bytes:.1f}x less, = M: pairs stage per pair)")
    print(f"  block-staged      {fmt_b(bs_bytes * layers)}  = "
          f"{bs_bytes * layers / (flops * layers):.3f} byte/flop   "
          f"({sh_bytes / bs_bytes:.1f}x less, = G*M: the next lever if bytes still bind)")
    print(f"\nthe §7 pre-registered expectation, as numbers:")
    print(f"  baseline attn                     {MEASURED_ATTN_MS:.1f} ms  (library route, best-of-3)")
    print(f"  shipped bytes / 237 GB/s            {sh_bytes * layers / 1e9 / DRAM_READ_GBS * 1000:.1f} ms"
          "   <- predicts the measured line")
    print(f"  tiled bytes / 237 GB/s              {ti_bytes * layers / 1e9 / DRAM_READ_GBS * 1000:.1f} ms"
          "   <- the traffic term after the change")
    print(f"  block-staged / 237 GB/s             {bs_bytes * layers / 1e9 / DRAM_READ_GBS * 1000:.1f} ms"
          "   <- if the K stage is hoisted to the block")
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
          "-o tile -- ./engine/build/decode_run --ids \"$(cat tmp/q10/ids1024.txt)\" --steps 1")
    print(f"    SQ_INSTS_VALU must NOT rise with the tile form (the softmax dedup should cut it: the")
    print(f"    shipped form runs pass 2 in all {C['N_Q'] // G * M}... see results.md), and the byte")
    print(f"    claim is verified by the ms falling toward the traffic term above, not by a counter")
    print(f"    this box has never produced for DRAM bytes.")


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
          f"+ K {C['k_stage_bytes']} B = {C['score_staging_bytes'] + C['k_stage_bytes']} B. If the "
          f"descriptor says exactly that, §3's 6.3x inflation is attn_core-specific.")


def cmd_selftest(args) -> int:
    """The mirror's own invariants — the ones a wrong geometry or a wrong rounding would break. All
    CPU-only; nothing here can pass by construction, which is why several checks assert a NUMERICAL
    identity against an independent expression rather than against the model itself."""
    checks: list[tuple[str, bool]] = []
    # 1. the constants are the kernel's, and the staging arithmetic is what the header asserts.
    checks.append((f"constants read: G={G} M={M} MAXSC={MAXSC} L={C['ATTN_TILE_L']}",
                   G == 4 and M == 4 and MAXSC == 512 and C["ATTN_TILE_L"] == 8))
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
    _, merged = merge_partials([(3.0, 2.0, acc)])
    checks.append(("merging one partial is the identity (empty windows may emit nothing)",
                   bool(np.array_equal(merged, acc * np.float32(np.exp(np.float32(0.0 - 3.0)))
                                       / np.float32(1.0) * 0 + acc * 0
                                       + acc * np.float32(np.exp(np.float32(3.0 - 3.0))) * 1.0))))
    # 6. the poison tripwire is not vacuous: a read past the built prefix lands on NaN.
    kd = torch.zeros(N_KV, 16, DH); vd = torch.zeros(N_KV, 16, DH)
    doc = CacheDoc(kd, vd, valid=4, cap=16)
    doc.read("k", 0, 3); doc.read("k", 0, 9)
    checks.append(("a read past the built prefix poisons (the mask-leak witness can fail)",
                   doc.poison_rows() == 1))
    # 7. the tiled form reads strictly fewer K/V rows than the shipped form, and by about G x M.
    lanes = [i + 1 for i in range(277)]
    sh_k, sh_v = shipped_reads(lanes, 277)
    ti_k, ti_v = tiled_reads(lanes, 277, M)
    bs_k, bs_v = tiled_reads(lanes, 277, M, k_copies=1, v_copies=1)
    ratio = (sh_k + sh_v) / (ti_k + ti_v)
    checks.append((f"re-use as written = {ratio:.2f}x, which is M = {M} (pairs stage per pair)",
                   M * 0.8 < ratio <= M + 1e-9))
    checks.append((f"block-level staging would give {(sh_k + sh_v) / (bs_k + bs_v):.2f}x, just under "
                   f"G*M = {G * M} (the residue is each row tile's masked tail)",
                   G * M * 0.9 < (sh_k + sh_v) / (bs_k + bs_v) <= G * M + 1e-9))
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
    "tilecarry": "a pair seeded from the PREVIOUS ROW's softmax state: §07.3's bug in its new, "
                 "between-rows-of-one-block form",
}


def neg_expect(neg: str, lanes: list[int], n: int, cap: int) -> tuple[set, set]:
    """{(row, head)} that must move, and the subset that must not, for one injection on one wave.

    Derived from the geometry rather than hand-listed, for the reason `mlp_gemv` writes up: a negative
    that cannot fail is worse than none, and one that catches everything measures nothing — and a
    hand-listed set rots into "everything" the moment the fixture changes. Two derivations are worth
    naming because both came out of a FAIL rather than out of a guess:
      * `readpast` cannot move a row that already attended `cap` lanes — the mask was not a
        constraint there, so the bug is invisible by construction, not by weakness.
      * No score-only bug can move a row with exactly ONE lane: softmax of a singleton is 1.0
        whatever the score arrives as, so `dimhalf` is structurally blind on row 0 of every span.
    A control that is blind somewhere must say so, or a reader counts it as coverage.
    """
    all_rh = {(i, h) for i in range(n) for h in range(N_Q)}
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
    if neg == "tilecarry":
        # Only rows that are not the first of their row tile can inherit a state from a sibling.
        return ({(i, h) for i in range(n) if i % M != 0 for h in range(N_Q)},
                {(i, h) for i in range(n) if i % M == 0 for h in range(N_Q)})
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
        print("carried prefix, which is where a §05.7 append bug can live:")
        for neg in NEG_TILE:
            tot = {"caught": 0, "blind": 0, "wrong": set(), "missing": set(), "poison": 0, "n": 0}
            for sp in spans[: args.neg_waves]:
                case = R.wave_cases(dl, layer, [sp])[0]
                n = sp[1] - sp[0] + 1
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
                    choices=["geometry", "traffic", "isa", "selftest", "gate"])
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--wave-rows", type=int, default=512)
    ap.add_argument("--fixture", default="p300")
    ap.add_argument("--wave-rows-gate", type=int, default=64,
                    help="rows per wave for the CPU gate; 64 matches run.py --wave")
    ap.add_argument("--waves", type=int, default=99, help="how many wave spans to price")
    ap.add_argument("--neg-waves", type=int, default=2,
                    help="spans to run each negative over: span 0 is one prompt, span 1 is a carry")
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--rows", default="",
                    help="restrict the gate to wave spans starting at these row indices; empty "
                         "(the default) takes the first --waves spans")
    args = ap.parse_args()
    if args.cmd == "geometry":
        cmd_geometry(args)
    elif args.cmd == "traffic":
        cmd_traffic(args)
    elif args.cmd == "isa":
        cmd_isa(args)
    elif args.cmd == "selftest":
        return cmd_selftest(args)
    else:
        return cmd_gate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
