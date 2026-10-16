#!/usr/bin/env python3
"""m2_model.py — the CPU mirror and CPU-only gates for `prototype/attn_m2` (R1: attention row-tiling
at M = 2, two query rows sharing one K/V walk). No device, no timings.

WHAT IS SETTLED HERE (all of it without a GPU, which is what a worker is allowed to touch):

    geometry   the pair's lane cut, its union, its staging bound — and a SECOND independent spelling of
               the union, so a bug in one is visible in the other
    census     K and V lane rows FETCHED per block, with multiplicity (traffic is multiplicity —
               `attn_decode` results.md §20.5 is the bill for booking a re-use the loop did not have)
    compare    this mirror's M = 2 form against the SHIPPED form's mirror (`attn_tile_model.wave_out`,
               form="shipped"), bit for bit, on pinned spans and on built boundary waves
    negative   the injections, each with a must-catch / must-not-catch set DERIVED from the geometry
    traffic    the byte model at the widths that carry the headline, and where the sharing stops paying
    selftest   the invariants a wrong pair geometry would break, plus source-text guards

WHY THE REDUCTIONS ARE IMPORTED, NOT RETYPED. The three ordered reductions (§05.5, and §07.3's
streaming form) come from `attn_tile_model` — `one_pass`, `online`, `merge_partials`, `epilogue`. They
are already witnessed: `reference.attend` reproduces the pins bit-for-bit (results.md §2, 0 of 98,304)
and that mirror reproduces the shipped form on both routes (results.md §20, §21 device). The M = 2
claim is not about the arithmetic; it is about WHICH LANES a row reduces, in WHICH order, FROM WHICH
memory, into WHICH partial slot. So the geometry below is transcribed from `attn_m2_kernels.h`
expression for expression — the `y * span` anchor, the `min(n, blo + span)` clip, the `self_here`
predicate — and the reduction primitives are handed the per-row ranges that transcription produces.
Two sides that share a formula agree, which is exactly why the gdn_collapse lesson warns that a mirror
which rewrites the expression cannot witness bit-identity: what is under test here is the ranges, and
the ranges are derived the way the kernel derives them.

THE SELF-LANE PROVENANCE IS CHECKED, NOT ASSERTED. Every K and V fetch in the mirror records which
document it came from (`stage` or `cache`) and the check is that it equals the shipped rule —
`lane == row_pos ? stage : cache` — per row. That is the property the shared stage could have broken
(one slot, two candidate sources), and it is what makes `NEG_SLOTK`/`NEG_SLOTV` (which perturb the
cache, not the stage) keep their bite in this form.

Run:
    .venv/bin/python prototype/attn_m2/m2_model.py selftest
    .venv/bin/python prototype/attn_m2/m2_model.py compare
    .venv/bin/python prototype/attn_m2/m2_model.py negative
    .venv/bin/python prototype/attn_m2/m2_model.py traffic
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "attn_decode")):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import attn_tile_model as T          # noqa: E402  the shipped form's mirror + the reduction primitives
import reference as R                # noqa: E402

M2_H = Path(__file__).resolve().parent / "attn_m2_kernels.h"
N_Q, N_KV, DH, MAXSC, TILEW, G = T.N_Q, T.N_KV, T.DH, T.MAXSC, T.TILEW, T.G
DRAM_GBS = T.DRAM_READ_GBS            # 237 GB/s — quoted (devices/strix-halo-memory-bandwidth.md)


def m2_consts(path: Path = M2_H) -> dict:
    """This kernel's own constants, read from its header (a copy here would drift). Several are written
    as expressions over the shipped header's names — `ATTN_M2_G` is `GQA`, `ATTN_M2_STAGE` is
    `MAXSC + ATTN_M2_SPLIT_RES` — so they are resolved in a namespace seeded from the shipped header
    rather than retyped here."""
    ns = {"N_Q": T.N_Q, "N_KV": T.N_KV, "GQA": T.G, "TILE": T.TILEW, "MAXSC": T.MAXSC}
    raw = re.findall(r"^#define\s+(\w+)\s+([\w/+*() -]+?)\s*(?://.*)?$", path.read_text(), re.M)
    vals: dict[str, int] = {}
    for name, expr in raw:
        try:
            vals[name] = int(eval(expr.strip(), {"__builtins__": {}}, ns))  # noqa: S307
        except (SyntaxError, ValueError, NameError, TypeError):
            continue                        # a non-numeric macro: not a constant this mirror needs
        ns[name] = vals[name]
    for name in ("ATTN_M2_M", "ATTN_M2_G", "ATTN_M2_THREADS", "ATTN_M2_L", "ATTN_M2_MAXSPLIT",
                 "ATTN_M2_STAGE", "ATTN_M2_VSLOT"):
        if name not in vals:
            raise SystemExit(f"{path.name}: cannot read {name} — this mirror's copy is stale")
    vals["warps"] = vals["ATTN_M2_THREADS"] // 32
    return vals


C2 = m2_consts()
M2, STAGE, VSLOT, MAXSPLIT = (C2["ATTN_M2_M"], C2["ATTN_M2_STAGE"], C2["ATTN_M2_VSLOT"],
                              C2["ATTN_M2_MAXSPLIT"])
BYTES_PER_LANE_ROW = DH * 2           # one BF16 K (or V) row of one KV head = 512 B
LAYERS = 6                            # §05: attention layers 3,7,11,15,19,23


# ============================================================================================
# Geometry — transcribed from `m2_row_at` / `m2_pair_at`, field for field.
# ============================================================================================
def lanes_of(lvalid: int, pos: int, row: int, neg: str = "none", cap: int = 1 << 30) -> int:
    """The shipped `lanes_of`, including the two wrong-engine injections it carries."""
    if neg == "readpast":
        return cap
    if neg == "dropcurrent":
        return pos + row
    return lvalid + row


def blocks_for_row(n: int, split: int, full: bool) -> int:
    """The shipped `blocks_for_row`: one block iff the staging fits the row, else the launch's split."""
    return 1 if (full and n <= MAXSC) else max(split, 1)


def row_at(lvalid, pos, split, full, row, y, nrows, tile=TILEW, neg="none", cap=1 << 30) -> dict:
    g = {"row": row, "exists": row < nrows, "pos": pos + row}
    g["n"] = lanes_of(lvalid, pos, row, neg, cap) if g["exists"] else 0
    g["s_row"] = blocks_for_row(g["n"], split, full) if g["exists"] else 0
    g["span"] = (g["n"] + g["s_row"] - 1) // g["s_row"] if g["s_row"] > 0 else 0
    g["one_pass"] = bool(g["exists"] and g["s_row"] == 1 and full)
    blo = g["span"] * y
    hi = min(g["n"], blo + g["span"]) if g["span"] > 0 else 0
    g["blo"], g["bhi"] = blo, hi
    g["active"] = bool(g["exists"] and g["s_row"] > 0 and hi > blo)
    g["too_wide"] = bool(g["active"] and (hi - blo) > MAXSC)
    g["chunks"] = (hi - blo + tile - 1) // tile
    g["self_here"] = False
    return g


def pair_at(lvalid, pos, split, full, rt, y, nrows, tile=TILEW, neg="none", cap=1 << 30):
    p = {"u_lo": 1 << 30, "u_hi": 0, "kmax": 0, "any": False, "too_wide": False}
    rows = []
    for m in range(M2):
        g = row_at(lvalid, pos, split, full, M2 * rt + m, y, nrows, tile, neg, cap)
        if g["too_wide"]:
            p["too_wide"] = True
        if g["active"]:
            if not p["any"]:
                p.update(u_lo=g["blo"], u_hi=g["bhi"], kmax=g["chunks"], any=True)
            else:
                p["u_lo"] = min(p["u_lo"], g["blo"])
                p["u_hi"] = max(p["u_hi"], g["bhi"])
                p["kmax"] = max(p["kmax"], g["chunks"])
        rows.append(g)
    p["union_n"] = p["u_hi"] - p["u_lo"] if p["any"] else 0
    if p["union_n"] > STAGE:
        p["too_wide"] = True
    for g in rows:
        g["self_here"] = bool(g["active"] and p["u_lo"] <= g["pos"] < p["u_hi"])
    return rows, p


def rtiles(nrows: int) -> int:
    return (nrows + M2 - 1) // M2


def chunk_span(rows, k, tile):
    """The union of the pair's work unit k — the set of V lane rows that chunk stages."""
    lo, hi = 1 << 30, 0
    for g in rows:
        if not g["active"] or k >= g["chunks"]:
            continue
        s0 = g["blo"] + k * tile
        hi = max(hi, min(g["bhi"], s0 + tile))
        lo = min(lo, s0)
    return (0, 0) if hi <= lo else (lo, hi)


# ---- the fetch census: multiplicity, per block ------------------------------------------------
def wave_census(lanes, split, full, tile=TILEW, form="m2"):
    """(K, V) BF16 lane rows a whole wave asks global memory for, summed over its two KV groups.

    `form="shipped"` is `attn_tile_wave`'s census — one block per (row, KV group, lane block), each
    staging its own row's slice: Σ lanes per group (results.md §20.5, the G re-use). `form="m2"` walks
    pairs: K over the union of the pair's slices (slices partition a row, so the union is the honest
    fetched set) plus one self-lane row per active row; V per pass-2 chunk, where the pair's chunks are
    anchored at their OWN blo, so when the two spans differ the chunk unions OVERLAP and V is re-used
    slightly less than K. That is a property of the geometry, printed, not hidden.
    """
    nrows = len(lanes)
    if form == "shipped":
        k = sum(lanes) * N_KV
        return k, k
    lvalid, pos = lanes[0], lanes[0] - 1
    k = v = 0
    for rt in range(rtiles(nrows)):
        for y in range(split):
            rows, p = pair_at(lvalid, pos, split, full, rt, y, nrows, tile)
            if not p["any"]:
                continue
            k += p["union_n"] + sum(1 for g in rows if g["self_here"])
            for kk in range(p["kmax"]):
                lo, hi = chunk_span(rows, kk, tile)
                v += hi - lo
            v += sum(1 for g in rows if g["self_here"])
    return k * N_KV, v * N_KV


def wave_census_tile(lanes, split, full, m_rows):
    """The same census generalized to a row tile of `m_rows` — M = 4 and M = 8 are priced, not built."""
    lvalid, pos = lanes[0], lanes[0] - 1
    k = v = 0
    for i in range(0, len(lanes), m_rows):
        grp = list(range(i, min(len(lanes), i + m_rows)))
        for y in range(split):
            segs = []
            for row in grp:
                g = row_at(lvalid, pos, split, full, row, y, len(lanes))
                if g["active"]:
                    segs.append((g["blo"], g["bhi"]))
            if not segs:
                continue
            width = max(s[1] for s in segs) - min(s[0] for s in segs)
            k += width + len(segs)
            v += width + len(segs)
    return k * N_KV, v * N_KV


def union_brute(lanes, split, full):
    """Σ over pairs and lane blocks of the union width, spelled a SECOND way: each row's slice list
    comes from `T.row_bounds` (the shipped mirror's own cut), not from `row_at` above. Agreement
    between the two spellings is the check; either alone proves nothing."""
    total = 0
    for i in range(0, len(lanes), M2):
        grp = lanes[i:i + M2]
        bounds = [T.row_bounds(n, blocks_for_row(n, split, full)) for n in grp]
        ys = max(len(b) for b in bounds) if bounds else 0
        for y in range(ys):
            segs = [b[y] for b in bounds if y < len(b) and b[y][1] > b[y][0]]
            if segs:
                total += max(s[1] for s in segs) - min(s[0] for s in segs)
    return total * N_KV


# ============================================================================================
# The mirror.
# ============================================================================================
def case_rows(case: dict) -> dict:
    p0, p1 = case["span"]
    return {"p0": p0, "n": p1 - p0 + 1, "lvalid": p0 + 1, "pos": p0,
            "q_rope": case["ref"]["q_rope"], "k_rope": case["ref"]["k_rope"],
            "v": case["ref"]["v"], "gate": case["qg"], "cap": case["cap"]}


def build_doc(case: dict):
    w = case_rows(case)
    doc = T.CacheDoc(case["kcache"].reshape(N_KV, case["cap"], DH),
                     case["vcache"].reshape(N_KV, case["cap"], DH), w["p0"], case["cap"])
    for i in range(w["n"]):
        doc.k[:, w["p0"] + i] = w["k_rope"][i].to(torch.float32)
        doc.v[:, w["p0"] + i] = w["v"][:, i].to(torch.float32)
    return doc


def read_lanes(doc, kind, kv, lanes, self_lane, self_row):
    """Fetch the lane rows one (row, head) pair needs, recording each one's provenance."""
    out = []
    prov = []
    for j in lanes:
        if j == self_lane:
            out.append(self_row)
            prov.append("stage")
        else:
            out.append(doc.read(kind, kv, j))
            prov.append("cache")
    return np.stack(out), prov


def part_site_m2(src: str) -> list[tuple]:
    """The split-partial address sites in a header, as (row, multiplier, head, split-expr, block-expr).

    `attn_tile_model.part_layout` is calibrated to the shipped file's THREE sites and raises on fewer;
    this form has one writer, because its reader is the shipped `attn_combine` — that reuse is the
    design (§19's aliasing came from a merge with a second, self-consistent layout), so the check here
    is not "my layout is internally consistent" but "my writer and the shipped reader compute the same
    address". That is what `selftest`'s `SLOT-SHAPE` check compares."""
    return re.findall(
        r"a\.part\s*\+\s*\(\(\(size_t\)(\w+)\s*\*\s*(\w+)\s*\+\s*(\w+)\)\s*\*\s*([\w.]+)\s*\+\s*"
        r"([^)]+)\)\s*\*\s*\(DH \+ 2\)", src)


def m2_wave(case: dict, neg: str = "none", doc=None, full: bool = True, tile: int = TILEW) -> dict:
    """One wave through the M = 2 mirror. Block order follows the kernel (pair, KV group, lane block)
    rather than a per-row loop on purpose: if walking it in block order could change a row's bits, the
    reduction was never per-row and the route is not admissible. Block order is also where the census,
    the staging bound and the partial slots become observable."""
    w = case_rows(case)
    p0, n, lvalid, pos = w["p0"], w["n"], w["lvalid"], w["pos"]
    lanes = [lanes_of(lvalid, pos, r, neg, case["cap"]) for r in range(n)]
    split = T.windows_of(max(lanes))
    if doc is None:
        doc = build_doc(case)
    src = M2_H.read_text()
    layout = "group-collapsed" if neg == "partalias" else (
        "group-dimensioned" if all(m == "N_Q" for _, m, _, _, _ in part_site_m2(src))
        and part_site_m2(src) else "group-collapsed")
    qg = w["gate"].reshape(n, N_Q, 2 * DH)[:, :, DH:]
    a_pre = torch.zeros(n, N_Q, DH, dtype=torch.float32)
    gated = torch.zeros(n, N_Q, DH, dtype=torch.float32)
    keep = T.half_keep() if neg == "dimhalf" else None
    buf = T.PartBuf(T.part_offset if layout == "group-dimensioned" else T.part_offset_collapsed)
    prov_bad = 0
    staged = {"k": 0, "v": 0}
    attended: dict[tuple[int, int], tuple[int, int]] = {}
    owe: dict[tuple[int, int], int] = {}
    carry: dict[tuple[int, int], tuple[float, float]] = {}

    for rt in range(rtiles(n)):
        for kv in range(N_KV):
            for y in range(split):
                rows, p = pair_at(lvalid, pos, split, full, rt, y, n, tile, neg, case["cap"])
                if p["too_wide"]:
                    raise RuntimeError(f"pair {rt} block {y}: union {p['union_n']} > staging {STAGE}"
                                       " — the launcher must refuse this split")
                if not p["any"]:
                    continue
                staged["k"] += p["union_n"] + sum(1 for g in rows if g["self_here"])
                for kk in range(p["kmax"]):
                    lo, hi = chunk_span(rows, kk, tile)
                    staged["v"] += hi - lo
                staged["v"] += sum(1 for g in rows if g["self_here"])
                for m in range(M2):
                    g = rows[m]
                    if not g["active"]:
                        continue
                    row = g["row"]
                    lo_r, hi_r = (p["u_lo"], p["u_hi"]) if neg == "maskwide" else (g["blo"], g["bhi"])
                    attended[(row, y)] = (lo_r, hi_r)
                    lane_list = list(range(lo_r, hi_r))
                    for h in range(G):
                        head = G * kv + h
                        # The shipped mirror's spelling: `gqaoff` moves the whole group, `gqa_head0`
                        # moves head 0 of it. Note the SELF operand below keeps the TRUE group, because
                        # that is what both kernels do (`attn_decode_kernels.h:593,628,656` and this
                        # header's `s_kself` fill both index `kv_write`), which is also why these two
                        # injections are not form-neutral here — see NEG_KIND.
                        kv_read = (kv + 1) % N_KV if neg in ("gqaoff",
                                                             "gqa_head0") and (
                                             neg == "gqaoff" or head == 0) else kv
                        q = w["q_rope"][row, head].to(torch.float32).numpy()
                        kmat, pk = read_lanes(doc, "k", kv_read, lane_list, g["pos"],
                                              w["k_rope"][row, kv].to(torch.float32).numpy())
                        vmat, pv = read_lanes(doc, "v", kv_read, lane_list, g["pos"],
                                              case["ref"]["v"][kv, row].to(torch.float32).numpy())
                        # The provenance check: the shipped rule is stage-for-self, cache-for-others,
                        # per row. A shared stage that fed one row the other row's stage bytes would
                        # show here first, and would also be what `NEG_SLOTK`/`NEG_SLOTV` expose.
                        for j, s in zip(lane_list, pk + pv):
                            if s != ("stage" if j == g["pos"] else "cache"):
                                prov_bad += 1
                        if keep is not None:
                            kmat = kmat * keep
                            q = q * keep
                        dots = (torch.from_numpy(q).unsqueeze(0)
                                @ torch.from_numpy(kmat).t()).numpy().ravel()
                        sc = T.bf16_round(dots) * np.float32(T.C["SCALE"])
                        V = torch.from_numpy(vmat)
                        seed_m, seed_l = 0.0, np.float32(0.0)
                        if neg == "headcarry" and m > 0 and (kv, h) in carry:
                            seed_m, seed_l = carry[(kv, h)]
                        if g["one_pass"]:
                            mm, l, acc = T.one_pass(sc, V, seed_m, seed_l)
                            a_pre[row, head], gated[row, head] = T.epilogue(acc, l, qg[row, head])
                        else:
                            mm, l, acc = T.online(sc, V, tile, seed_m, seed_l)
                            # `partrow`: the writer's row term collapses to the pair's first row, while
                            # the READER (`attn_combine`) still asks for the slot of the row it is
                            # merging. Modelling the writer remap and not the reader's is the point: the
                            # first row of the pair then merges its partner's partials and the partner
                            # merges a slot nobody wrote (the kernel memsets it to the neutral (0,0,0)).
                            wrow = M2 * rt if neg == "partrow" else row
                            buf.write(wrow, head, y, split, (mm, l, acc), f"r{row}h{head}")
                            owe[(row, head)] = g["s_row"]
                        carry[(kv, h)] = (mm, l)
    for (row, head), s_row in owe.items():
        parts = [buf.read(row, head, yy, split) for yy in range(s_row)]
        _, l, acc = T.merge_partials(parts)
        a_pre[row, head], gated[row, head] = T.epilogue(acc, l, qg[row, head])
    return {"attn_pre_gate": a_pre, "gated": gated, "doc": doc, "buf": buf, "lanes": lanes,
            "split": split, "layout": layout, "staged": staged, "prov_bad": prov_bad,
            "attended": attended}


def shipped_wave(case: dict, neg: str = "none", doc=None) -> dict:
    """The reference side, spelled by the shipped mirror and never copied here. §20.2's lesson: a check
    whose reference side is derived from the thing under test cannot adjudicate it."""
    return T.wave_out(case, "shipped", neg=neg, doc=doc, layout="group-dimensioned",
                      window_form="per-row-slice")


# ============================================================================================
# Cases. No pinned prompt on this box reaches MAXSC lanes (results.md §20.2), so the boundary waves are
# BUILT from pinned rows and named as synthetic. M = 2 puts a pair of rows in one block, which changes
# which rows can straddle the one-pass/streaming boundary — so these are mandatory, not optional.
# ============================================================================================
def wide_plan(dl: Path, layer: int, start: int, nrows: int, tag: str, src_rows: int = 32) -> dict:
    """A wave placed anywhere in the context, including past the end of the pinned prompt.

    No fixture on this box has a PROMPT long enough to put a pair of rows on the MAXSC boundary
    (`wide1024f` has decode rows above 1024 lanes but no per-layer prefill stages — `wave_cases` cannot
    read it at all), so both the CPU mirror and the device plan need a built case. Why it is legitimate
    rather than a fiction:

      * every q/k/v/gate row is a real pinned tensor of this model, cycled — nothing is invented;
      * the CACHE is filled by cycling the pinned prefill cache over ALL `cap` lanes, NOT left zeros.
        `attn_tile_model.wide_case` leaves it zeros, and that is a real weakness for KV-group controls:
        a wave's first row then attends exactly one non-zero lane (its own self lane) and a group swap
        cannot reach it — a control that scores zero and looks clean, the `partalias`-over-pinned-spans
        trap one level over. A populated cache removes that blind spot;
      * the geometry is the engine's: one row per position, a causal prefix, a wave that appends its own
        lanes (§05.7) into lanes `>= pos`.

    What is NOT real is the content, so no HF pin exists for these rows and no numeric accuracy claim is
    made from them. They price geometry, addressing, reduction order and bit-identity between two forms
    over identical operands — which is what this gate is.
    """
    seq = min(R.prefill_tokens(dl, layer), src_rows)
    base = R.wave_cases(dl, layer, [(0, seq - 1)])[0]
    idx = [(start + j) % seq for j in range(nrows)]
    cap = max(R.CAP_ROUND, -(-(start + nrows) // R.CAP_ROUND) * R.CAP_ROUND)
    ck, cv = base["ref"]["cache_keys"], base["ref"]["cache_values"]
    kc = torch.stack([ck[:, j % ck.shape[1]] for j in range(cap)], dim=1)
    vc = torch.stack([cv[:, j % cv.shape[1]] for j in range(cap)], dim=1)
    return {"name": tag, "kind": "wave", "pos": start, "lvalid": start + 1, "nrows": nrows,
            "cap": cap, "synthetic": True,
            "qg": base["qg"].reshape(seq, N_Q, 2 * DH)[idx].reshape(-1),
            "kg": base["kg"].reshape(seq, N_KV, DH)[idx].reshape(-1),
            "vg": base["vg"].reshape(seq, N_KV, DH)[idx].reshape(-1),
            "qn": base["qn"], "kn": base["kn"],
            "kcache": kc.reshape(-1).to(torch.bfloat16), "vcache": vc.reshape(-1).to(torch.bfloat16),
            "ref": {"q_rope": base["ref"]["q_rope"][idx], "k_rope": base["ref"]["k_rope"][idx],
                    "v": base["ref"]["v"][:, idx], "attn_pre_gate": None},
            "span": (start, start + nrows - 1)}


def boundary_cases(dl: Path, layer: int) -> list[dict]:
    """Waves placed so a PAIR contains rows of exactly MAXSC and MAXSC+1 lanes (one one-pass, one
    streaming), plus waves straddling lane blocks 2 and 3, the widest pair a 1024-token context holds,
    and the shortest pairs a wave has. M = 2 puts two rows in one block, which changes which rows can
    straddle the one-pass/streaming boundary, so these are mandatory cases, not optional extras."""
    specs = ((MAXSC - 1, 4, "pair-at-maxsc"),          # one-pass beside streaming
             (MAXSC - 2, 4, "maxsc-minus-1"),
             (MAXSC, 8, "pair-just-above"),            # unequal spans: the V-overlap case
             (2 * MAXSC - 1, 4, "straddles-block-2"),
             (3 * MAXSC - 1, 4, "straddles-block-3"),
             (1020, 4, "pair-widest-1024"),
             (0, 8, "pair-shortest"))                  # the least a pair can share
    return [wide_plan(dl, layer, start, nrows, f"m2.{tag}.L{layer:02d}")
            for start, nrows, tag in specs]


def pinned_cases(dl: Path, layer: int, wave_rows: int) -> list[dict]:
    seq = R.prefill_tokens(dl, layer)
    return [R.wave_cases(dl, layer, [sp])[0] for sp in R.wave_spans(seq, wave_rows)]


# ============================================================================================
# Derived must-catch / must-not-catch sets.
# ============================================================================================
NEG_KIND = {          # form-neutral: both forms do the SAME wrong thing, so the forms must still agree
    "readpast": "form-neutral", "dropcurrent": "form-neutral", "dimhalf": "form-neutral",
    # These two cannot be form-neutral, and the reason is a finding, not an excuse: the shipped mirror
    # rewrites the SELF operand's KV group along with the group's cache reads, while the shipped KERNEL
    # stages the self row from `kv_write` — the block's own group (`attn_decode_kernels.h:593,628,656`),
    # exactly as this kernel does. So the two mirrors cannot agree bit-for-bit under these injections
    # even though the two kernels do; what is assertable is that both forms' wrongs land on exactly the
    # derived rows, and that the injection does not disturb the stage/cache provenance split.
    "gqaoff": "self-independent", "gqa_head0": "self-independent",
    "maskwide": "m2-range",       # adjudicated on WHICH LANES each row attended, not on output bits
    "partrow": "m2", "partalias": "m2", "headcarry": "m2",
}
NEG_WHY = {
    "readpast": "the causal mask is gone: every row attends `cap` lanes",
    "dropcurrent": "the §05.7 append is forgotten: the row attends only lanes below its own position",
    "dimhalf": "the score dot accumulates 128 of the 256 head dims",
    "gqaoff": "the whole group reads the neighbour's K/V (kappa off by one)",
    "gqa_head0": "only head 0 of a group reads the neighbour: the shared-stage wiring test",
    "maskwide": "a row attends the pair's UNION instead of its own prefix — the bug M = 2's shared walk"
                " invites, and the one the admission gate exists to catch",
    "partrow": "the split-partial slot drops the pair-row term: the two rows of a pair alias — §19's"
               " failure class at a new address",
    "partalias": "the slot is keyed by head-in-group with no KV-group term: §19's failure, re-injected",
    "headcarry": "row 1 of a pair seeded from row 0's (max, sum): §07.3/A-17's state leak in the only"
                 " shape M = 1 could not have. Mirror-only: two rows in one block, and the winner of an"
                 " aliased shared slot is whichever warp landed last, which is not a gate witness.",
}


def maskwide_rows(lanes: list[int], split: int, full: bool, tile: int = TILEW) -> tuple[set, set]:
    """Which rows `maskwide` can reach, derived by walking the geometry instead of arguing about it.

    A row moves iff the pair's union in one of ITS lane blocks extends past that row's own slice. That
    is not the same question as "which row has fewer lanes": at the one-pass/streaming boundary the
    one-pass row walks its WHOLE row in block 0 while its streaming partner walks one slice, so the
    narrower slice is the STREAMING row's and it is the one that gains lanes. Arguing it from the lane
    counts got this derivation backwards on first writing; walking `pair_at` cannot.
    """
    lvalid, pos = lanes[0], lanes[0] - 1
    can, blind = set(), set()
    for row in range(len(lanes)):
        gains = False
        for y in range(split):
            rows, p = pair_at(lvalid, pos, split, full, row // M2, y, len(lanes), tile)
            g = rows[row % M2]
            if g["active"] and (g["blo"], g["bhi"]) != (p["u_lo"], p["u_hi"]):
                gains = True
        (can if gains else blind).add(row)
    return ({(r, h) for r in can for h in range(N_Q)},
            {(r, h) for r in blind for h in range(N_Q)})


def neg_expect(neg: str, lanes: list[int], split: int, full: bool, cap: int) -> tuple[set, set]:
    """{(row, head)} that must move, and the subset that must not — DERIVED from the geometry, never
    hand-listed (a control that cannot fail is worse than none; one that catches everything measures
    nothing; a hand-listed set rots into "everything" the moment the fixture changes).

    The derivations worth naming, because each is a fact about geometry rather than a wish:

      * `maskwide` widens a row to the pair's union. A row sees nothing unless the union extends past
        ITS OWN slice — the last lane block of a pair ends at the pair's max `bhi`, so the row with
        FEWER lanes gains lanes and the other gains none. Rows that already reach `cap` are blind for
        the same reason §20.6's `readpast` set is blind.
      * `partrow` / `partalias` cannot move a one-pass row: a one-pass row finishes inside its own
        block and writes NO partial, so there is no slot to steal. Over pinned spans alone (all under
        MAXSC) this scores 0 of everything — §20.6's `partalias` trap, which is why the boundary waves
        above are mandatory cases of this gate.
      * `headcarry` cannot move a pair's FIRST row: it has no predecessor inside its own block, and a
        carry cannot cross a block because another pair is another block.
    """
    n = len(lanes)
    all_rh = {(i, h) for i in range(n) for h in range(N_Q)}
    if neg == "readpast":
        still = {(i, h) for i in range(n) if lanes[i] >= cap for h in range(N_Q)}
        return all_rh - still, still
    if neg == "dimhalf":
        blind = {(i, h) for i in range(n) if lanes[i] == 1 for h in range(N_Q)}
        return all_rh - blind, blind
    if neg in ("dropcurrent", "gqaoff"):
        return all_rh, set()
    if neg == "gqa_head0":
        return {(i, 0) for i in range(n)}, all_rh - {(i, 0) for i in range(n)}
    if neg == "maskwide":
        return maskwide_rows(lanes, split, full)
    if neg in ("partrow", "partalias"):
        stream = {i for i in range(n) if blocks_for_row(lanes[i], split, full) > 1}
        one = set(range(n)) - stream
        must = {(i, h) for i in stream for h in range(N_Q)}
        blind = {(i, h) for i in one for h in range(N_Q)}
        if neg == "partalias":
            must = {(i, h) for i in stream for h in range(G)}
            blind = blind | {(i, h) for i in stream for h in range(G, N_Q)}
        return must, blind
    if neg == "headcarry":
        must = {(lo + 1, h) for lo in range(0, n, M2) if lo + 1 < n for h in range(N_Q)}
        return must, all_rh - must
    raise KeyError(neg)


# ============================================================================================
# Commands
# ============================================================================================
def fmt_b(n: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"


def cmd_geometry(args) -> None:
    print(f"constants from {M2_H.name}: M={M2} G={G} warps={C2['warps']} L={C2['ATTN_M2_L']} "
          f"MAXSPLIT={MAXSPLIT} STAGE={STAGE} VSLOT={VSLOT} | shipped MAXSC={MAXSC} TILE={TILEW} "
          f"| lane ceiling MAXSC x MAXSPLIT = {MAXSC * MAXSPLIT}")
    print(f"{'n (lower row)':>14} {'split':>5} {'routes':>14} {'union':>6} {'ship':>6} {'m2':>5}"
          f" {'ratio':>7}  slice of the streaming row")
    for n in (1, 2, 8, MAXSC - 1, MAXSC, MAXSC + 1, 1023, 1024, 1025, 2048, 4095, 4096):
        split = max(1, T.windows_of(n + 1))
        rows, p = pair_at(n, n - 1, split, True, 0, 0, 2)
        u = union_brute([n, n + 1], split, True) // N_KV
        ship = 2 * n + 1
        routes = f"{rows[0]['s_row']}+{rows[1]['s_row']} blocks"
        ratio = u / ship
        print(f"{n:>14} {split:>5} {routes:>14} {p['u_lo']:>6} {ship:>6} {u:>5} {ratio:>7.3f}  "
              f"{[(g['blo'], g['bhi']) for g in rows][:2]}")


def cmd_compare(args) -> int:
    """The M = 2 mirror against the shipped mirror, bit for bit, with the fetch census per case."""
    fails: list[str] = []
    dl = ROOT / "data" / "refs" / args.fixture
    layer = R.ATTN_LAYERS[0]
    pin = pinned_cases(dl, layer, args.wave_rows)
    bnd = boundary_cases(dl, layer)
    cases = pin + bnd
    if args.anchor:
        cases += T.decode_steps(ROOT / "data" / "refs" / args.anchor, layer)
    print(f"fixture {dl.name} L{layer:02d}: {len(cases)} cases — {len(pin)} pinned spans (all under"
          f" MAXSC), {len(bnd)} built boundary waves"
          + ("" if not args.anchor else f", {len(cases) - len(pin) - len(bnd)} real >MAXSC decode rows"))
    print(f"{'case':<38} {'lanes':>11} {'split':>5} {'forms differ':>16} {'K+V ship':>11}"
          f" {'K+V m2':>10} {'ratio':>6} {'poison':>6} {'prov':>5}")
    for case in cases:
        n = case["span"][1] - case["span"][0] + 1
        lanes = [case["lvalid"] + i for i in range(n)]
        split = T.windows_of(max(lanes))
        ship = shipped_wave(case)
        m2 = m2_wave(case)
        d = T.form_diff(m2["attn_pre_gate"], ship["attn_pre_gate"])
        sh_k, sh_v = wave_census(lanes, split, True, form="shipped")
        m_k, m_v = wave_census(lanes, split, True)
        poison = m2["doc"].poison_rows()
        diff = f"{d['ndiff']} of {d['n']}"
        ratio = (m_k + m_v) / (sh_k + sh_v)
        print(f"{case['name'][:38]:<38} {f'{lanes[0]}-{lanes[-1]}':>11} {split:>5}"
              f" {diff:>16} {sh_k + sh_v:>11,} {m_k + m_v:>10,} {ratio:>6.3f}"
              f" {poison:>6} {m2['prov_bad']:>5}")
        if d["ndiff"]:
            for i, h in d["pairs"][:5]:
                print(f"      row {i} head {h}: {lanes[i]} lanes — differs. The claim is bit-identity,"
                      " so this is a defect, not a band.")
            fails.append(f"{case['name']}: {d['ndiff']} elements differ from the shipped form"
                         f" (worst row 1-cos {d['worst_row_1cos']:.3e})")
        if m2["buf"].aliases():
            fails.append(f"{case['name']}: the M = 2 partial layout aliases {len(m2['buf'].aliases())}"
                         " slots — two writers, one address (§19's failure class)")
        if poison:
            fails.append(f"{case['name']}: the M = 2 mirror read {poison} poison rows — the causal"
                         " mask leaked past this row's prefix")
        if m2["prov_bad"]:
            fails.append(f"{case['name']}: {m2['prov_bad']} lane reads came from a different document"
                         " than the shipped rule prescribes")
    print(f"\ncompare: {len(fails)} failures")
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


def gqa_reach(doc, n_lanes_row: list[int], self_lanes: list[int], include_self: bool) -> tuple[set, set]:
    """Which (row, head) a KV-group swap can reach, derived from the OPERANDS rather than hoped for.

    A group swap can only change a row's output if some lane that row attends holds different bytes in
    the neighbouring KV group. On a SYNTHETIC-WIDE wave that is a real constraint: `wide_case`'s cache
    is zeros except the rows the wave itself appends, so a wave's FIRST row attends exactly one
    non-zero lane — its own self lane — and a swap that leaves the self lane alone cannot touch it at
    all. A must-catch set that ignored this would be a control that cannot fail, which is the
    `partalias`-over-pinned-spans trap again, one level over: reachability is a property of the fixture
    as well as the geometry.

    `include_self` is where the two mirrors legitimately differ. Both KERNELS stage a row's self lane
    from the block's own KV group (`attn_decode_kernels.h:593,628,656`; `attn_m2_kernels.h` `s_kself`),
    so a swap cannot reach the self lane on the device. The shipped MIRROR rewrites the self operand's
    group along with the rest of the block (`attn_tile_model.wave_out`, the `gqaoff` branch before
    `self_k = ...`), so on that side the self lane does count. Passing True for the shipped side and
    False for this one is not a fudge: it is the difference between the two spellings, and the reason
    these two injections are not usable as form-neutral bit-identity checks.
    """
    dist = torch.zeros(max(n_lanes_row), dtype=torch.bool)   # lanes are 0..n-1, so cap-sized
    for j in range(len(dist)):
        dist[j] = bool(torch.any(doc.k[0, j, :] != doc.k[1, j, :])
                       or torch.any(doc.v[0, j, :] != doc.v[1, j, :]))
    must, blind = set(), set()
    for r, nr in enumerate(n_lanes_row):
        reach = any(dist[j] for j in range(nr) if include_self or j != self_lanes[r])
        for h in range(N_Q):
            (must if reach else blind).add((r, h))
    return must, blind


def cmd_negative(args) -> int:
    """Every injection against its derived sets, on the cases that can reach it."""
    fails: list[str] = []
    dl = ROOT / "data" / "refs" / args.fixture
    layer = R.ATTN_LAYERS[0]
    cases = boundary_cases(dl, layer) + pinned_cases(dl, layer, args.wave_rows)
    for case in cases:
        if args.only and args.only not in case["name"]:
            continue
        n = case["span"][1] - case["span"][0] + 1
        lanes = [case["lvalid"] + i for i in range(n)]
        split = T.windows_of(max(lanes))
        doc = build_doc(case)
        base_m2 = m2_wave(case, doc=doc)
        base_ship = shipped_wave(case, doc=doc)
        for neg, kind in NEG_KIND.items():
            if neg in ("gqaoff", "gqa_head0"):
                must_s, blind_s = gqa_reach(doc, lanes, [case["pos"] + i for i in range(n)], True)
                must_m, blind_m = gqa_reach(doc, lanes, [case["pos"] + i for i in range(n)], False)
                if neg == "gqa_head0":      # only head 0 of the group is rewired, on both sides
                    must_s = {(r, h) for r, h in must_s if h == 0}
                    must_m = {(r, h) for r, h in must_m if h == 0}
                    blind_s = set([(r, h) for r, h in blind_s if h == 0]) | {
                        (r, h) for r in range(len(lanes)) for h in range(N_Q) if h != 0}
                    blind_m = set([(r, h) for r, h in blind_m if h == 0]) | {
                        (r, h) for r in range(len(lanes)) for h in range(N_Q) if h != 0}
                must, blind = must_s, blind_s
            else:
                must, blind = neg_expect(neg, lanes, split, True, case["cap"])
                must_m = must
            inj_m2 = m2_wave(case, neg=neg, doc=build_doc(case)) \
                if kind.startswith("m2") or kind == "self-independent" else None
            if kind.startswith("m2"):
                ref, got = base_m2, inj_m2
            else:
                ref, got = base_ship, shipped_wave(case, neg=neg, doc=build_doc(case))
            moved = set(T.form_diff(got["attn_pre_gate"], ref["attn_pre_gate"])["pairs"])
            bad: list[str] = []
            extra = ""
            if kind == "m2-range":
                # Adjudicated on WHICH LANES each row attended, not on output bits. A widened row can
                # cancel to the same BF16 result (the chaos floor every bf16 gate on this box fights),
                # so "the output happened to move" is not a property of the mask. What must hold exactly
                # is the range set, plus two subset assertions that a range argument alone cannot give.
                hon_at, inj_at = base_m2["attended"], got["attended"]
                range_moved = {r for (r, y) in hon_at if hon_at[(r, y)] != inj_at.get((r, y))}
                extra = f"range {len(range_moved)}"
                if {(r, h) for r in range_moved for h in range(N_Q)} != must:
                    bad.append(f"the attended-range set is not the derived set:"
                               f" +{sorted({r for r in range_moved} - {r for r, _ in must})[:4]}"
                               f" -{sorted({r for r, _ in must} - range_moved)[:4]}")
                if moved - must:
                    bad.append(f"{len(moved - must)} pairs moved whose lanes did NOT change — the"
                               " widening leaked into a row that attends exactly the same lanes")
                quiet = sorted({r for r in range_moved
                                if not any((r, h) in moved for h in range(N_Q))})
                if quiet:
                    bad.append(f"{len(quiet)} rows attended extra lanes and no head moved"
                               f" (e.g. {quiet[:3]}) — the injection never reached the arithmetic")
            else:
                if must - moved:
                    bad.append(f"{len(must - moved)} must-catch pairs did NOT move"
                               f" (e.g. {sorted(must - moved)[:3]}) — {NEG_WHY[neg]}")
                if moved & blind:
                    bad.append(f"moved {len(moved & blind)} derived-blind pairs"
                               f" (e.g. {sorted(moved & blind)[:3]}) — the derivation is wrong")
            if kind == "form-neutral":
                d = T.form_diff(inj_m2["attn_pre_gate"], got["attn_pre_gate"]) \
                    if inj_m2 else T.form_diff(m2_wave(case, neg=neg)["attn_pre_gate"],
                                               got["attn_pre_gate"])
                if d["ndiff"]:
                    bad.append(f"the two forms disagree under their own injection in {d['ndiff']}"
                               " elements — the shared walk changed which memory a row reads from")
            if kind == "self-independent":
                m2_moved = set(T.form_diff(inj_m2["attn_pre_gate"],
                                           base_m2["attn_pre_gate"])["pairs"])
                extra = f"m2 {len(m2_moved)}/{len(must_m)}"
                if m2_moved != must_m:
                    bad.append(f"the M = 2 form's version of this injection lands on different rows"
                               f" than its own derived set: +{sorted(m2_moved - must_m)[:3]}"
                               f" -{sorted(must_m - m2_moved)[:3]}")
                if inj_m2["prov_bad"]:
                    bad.append(f"{inj_m2['prov_bad']} lane reads came from the wrong document under"
                               " the injection")
            print(f"  [{'ok  ' if not bad else 'BAD '}] {case['name'][:32]:<32} {neg:<11}"
                  f" {kind:<15} moved {len(moved):>5}  must {len(must):>5}"
                  f"  blind {len(blind):>5}  {extra}")
            for b in bad:
                fails.append(f"{case['name']} {neg}: {b}")
    print(f"\nnegative: {len(fails)} failures")
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


def cmd_traffic(args) -> None:
    """The byte model, from quoted denominators only. NOTHING here is a measurement: the denominator is
    the 237 GB/s read ceiling, the per-touch byte count is arithmetic on the head geometry, and the ms
    column is bytes / that ceiling. The shipped attn ms rows live in
    progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md §3.4 and are NOT restated as results."""
    print("model: 1 lane row = 512 B (256 BF16); a (row, lane) touch is K+V over 2 KV heads = 2 KiB;")
    print(f"{LAYERS} attention layers (§05); 237 GB/s; wave row cap {args.wave_rows} (engine kMaxWaveRows).")
    print("The shipped column reproduces the funding doc's Σ_lanes x 2 KiB x 6 to the byte.")
    measured = {1024: 27.23, 2048: 104.32, 4096: 412.07}      # §3.4 "Measured" row, quoted
    doc_prize = {1024: 13.6, 2048: 49.9, 4096: 194.6}         # §6 table, M = 2 row, quoted
    doc_floor = {1024: 27.2, 2048: 108.8, 4096: 435.0}        # §3.4 byte floor at M = 1, quoted
    print(f"\n{'T':>6} {'Σlanes':>12} {'M=1 bytes':>11} {'M=1 floor':>10} {'doc floor':>10}"
          f" {'M=2 floor':>10} {'M=4':>8} {'M=8':>8} {'M=2/M=1':>9} {'attn ms':>9}"
          f" {'prize ms':>9} {'doc prize':>10}")
    for tokens in (1024, 2048, 4096):
        lanes = []
        for pos0 in range(0, tokens, args.wave_rows):
            nr = min(args.wave_rows, tokens - pos0)
            lanes += [pos0 + 1 + i for i in range(nr)]
        split = T.windows_of(max(lanes))
        sh_k, sh_v = wave_census(lanes, split, True, form="shipped")
        m2_k, m2_v = wave_census(lanes, split, True)
        m4_k, m4_v = wave_census_tile(lanes, split, True, 4)
        m8_k, m8_v = wave_census_tile(lanes, split, True, 8)
        per = BYTES_PER_LANE_ROW * LAYERS
        f1 = (sh_k + sh_v) * per / (DRAM_GBS * 1e9) * 1e3
        f2 = (m2_k + m2_v) * per / (DRAM_GBS * 1e9) * 1e3
        f4 = (m4_k + m4_v) * per / (DRAM_GBS * 1e9) * 1e3
        f8 = (m8_k + m8_v) * per / (DRAM_GBS * 1e9) * 1e3
        print(f"{tokens:>6} {sum(lanes):>12,} {fmt_b((sh_k + sh_v) * per):>11} {f1:>10.1f}"
              f" {doc_floor[tokens]:>10.1f} {f2:>10.1f} {f4:>8.1f} {f8:>8.1f}"
              f" {f2 / f1:>9.4f} {measured[tokens]:>9.2f}"
              f" {measured[tokens] - f2:>9.1f} {doc_prize[tokens]:>10.1f}")
    print("\n`prize ms` is `attn measured − M = 2 byte floor`, the same formula §6 uses, evaluated here")
    print("from an independent census: it lands on the document's own 13.6 / 49.9 / 194.6 ms within")
    print("0.3 ms, and the M = 1 floor column reproduces §3.4's 27.2 / 108.8 / 435.0. That agreement")
    print("is the census's gate — a traffic claim on this box cannot be argued from counters (D9).")
    print("\nWhere the sharing stops paying (all read off the per-block census, not fitted):")
    for rows, start in ((8, 0), (64, 0), (64, MAXSC - 1), (64, MAXSC), (64, 2 * MAXSC - 1)):
        lanes = [start + 1 + i for i in range(rows)]
        split = T.windows_of(max(lanes))
        sh_k, sh_v = wave_census(lanes, split, True, form="shipped")
        m_k, m_v = wave_census(lanes, split, True)
        print(f"  rows={rows:>3} at pos {start:>5}: lanes {lanes[0]}-{lanes[-1]}, split {split}, "
              f"M=2 bytes / M=1 bytes = {(m_k + m_v) / (sh_k + sh_v):.4f}")
    print("  * a pair whose rows have equal lane counts shares perfectly (ratio -> 1/2); the wave's")
    print("    FIRST pairs are short, and a short pair pays the union residue of both rows' slices;")
    print("  * a pair that STRADDLES the one-pass/streaming boundary shares least: the one-pass row")
    print("    walks its whole row in lane block 0 while its partner walks one slice, so the partner's")
    print("    later slices are shared with nobody;")
    print("  * V is re-used less than K whenever the two rows' spans differ, because the pair's chunk k")
    print("    are anchored at their own blo and the staged unions then overlap;")
    print("  * each row's self lane is staged a second time from `krope`/`vg` (that is what the shipped")
    print("    form reads there, and what keeps NEG_SLOTK/NEG_SLOTV visible), ~2 lane rows per block")
    print("    fetched twice, ~0.4 % at a 512-lane block. Counted, not hidden;")
    print("  * and past the traffic wall there is nothing left to win by depth: the arithmetic floor")
    print("    (3.53 / 14.1 / 56.5 ms, §3.4) sits 7.7x UNDER the byte floor and M = 2 does not move it")
    print("    at all, which is where the M ~ 7.7 crossover comes from (§3.4, width-independent because")
    print("    both bills are quadratic in Σ_lanes). M = 2 is the funded step; M = 4/8 are not built.")


def cmd_census(args) -> None:
    dl = ROOT / "data" / "refs" / args.fixture
    layer = R.ATTN_LAYERS[0]
    c = T.wide_case(dl, layer, args.rows, start=args.start)
    lanes = [c["lvalid"] + i for i in range(c["nrows"])]
    split = T.windows_of(max(lanes))
    sh_k, sh_v = wave_census(lanes, split, True, form="shipped")
    m_k, m_v = wave_census(lanes, split, True)
    closed = union_brute(lanes, split, True)
    print(f"{c['name']}: {args.rows} rows at {args.start}, lanes {lanes[0]}..{lanes[-1]}, split {split}")
    print(f"  shipped K {sh_k:,} V {sh_v:,} lane rows = {fmt_b((sh_k + sh_v) * BYTES_PER_LANE_ROW)}")
    print(f"  M = 2   K {m_k:,} V {m_v:,} lane rows = {fmt_b((m_k + m_v) * BYTES_PER_LANE_ROW)}"
          f"  ({(sh_k + sh_v) / (m_k + m_v):.4f}x)")
    print(f"  K closed form (second spelling) {closed:,} vs per-block walk {m_k:,}: "
          f"{'agree' if closed == m_k else 'DISAGREE'}")


def cmd_selftest(args) -> int:
    fails: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {what}")
        if not ok:
            fails.append(what)

    src = M2_H.read_text()
    # 1. geometry: two spellings of the union agree, and the staging bound holds for every legal launch
    bad, worst = 0, 0
    for split in range(1, MAXSPLIT + 1):
        for n in (1, 2, 7, 8, 63, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096):
            if n > split * MAXSC:
                continue                       # a slice wider than MAXSC is refused by the launcher
            for full in (0, 1):
                walk = 0
                for y in range(MAXSPLIT):
                    rows, p = pair_at(n, n - 1, split, full, 0, y, 2)
                    if not p["any"]:
                        break
                    walk += p["union_n"]
                    worst = max(worst, p["union_n"])
                if walk != union_brute([n, n + 1], split, full) // N_KV:
                    bad += 1
    check(bad == 0, f"pair union: the per-block walk and `row_bounds`' brute union agree ({bad} disagree)")
    check(worst <= STAGE, f"pair union <= ATTN_M2_STAGE for every legal (split, n): worst {worst}/{STAGE}")
    refused = sum(1 for n in (513, 1025, 4097)
                  if pair_at(n, n - 1, 1, True, 0, 0, 2)[1]["too_wide"])
    check(refused == 3, "a split shallower than split_for_lanes is flagged too_wide (3/3)")
    # 2. this mirror's lane cut IS the shipped cut, block for block
    ok = True
    for n in (1, 5, 512, 513, 1024, 1025, 4096):
        for s in range(1, MAXSPLIT + 1):
            s_row = blocks_for_row(n, s, True)
            got = [(row_at(n, n - 1, s, True, 0, y, 1)["blo"], row_at(n, n - 1, s, True, 0, y, 1)["bhi"])
                   for y in range(s_row)]
            ok &= [g for g in got if g[1] > g[0]] == T.row_bounds(n, s_row)
    check(bool(ok), "the per-row lane cut is `T.row_bounds`' cut, block for block")
    # 3. source guards: what the kernel must still SAY, read off its own text
    check(re.search(r"static_assert\(ATTN_M2_M == 2", src) is not None,
          "the kernel static_asserts M == 2 (this mirror prices only what the kernel implements)")
    check(re.search(r"static_assert\(ATTN_M2_L >= TILE", src) is not None,
          "a streaming tile fits the K sub-block stage: static_assert present")
    check(re.search(r"static_assert\(ATTN_M2_VSLOT > TILE", src) is not None,
          "the V stage bounds a tile plus the split residue: static_assert present")
    sites = part_site_m2(src)
    check(len(sites) == 1, f"this form has ONE partial writer — its reader is the shipped "
                           f"`attn_combine` ({len(sites)} sites found)")
    check(bool(sites) and all(m == "N_Q" for _, m, _, _, _ in sites)
          and {h for _, _, h, _, _ in sites} <= {"head", "slot_head"},
          f"the writer slot is keyed by row x N_Q + the FULL head ({sites})")
    ship_sites = part_site_m2(T.KERNEL_H.read_text())
    shapes = {(m, sp) for _, m, _, sp, _ in sites} , {(m, sp) for _, m, _, sp, _ in ship_sites}
    check(bool(ship_sites) and shapes[0] == shapes[1] == {("N_Q", "a.split")},
          f"writer and shipped reader compute the same address shape, row*N_Q + head, striding "
          f"a.split (mine {shapes[0]}, shipped {sorted(shapes[1])})")
    slots = re.findall(r"s_[a-z]+\[\((\w+) - (\w+)\) \* DH", src)
    bases = sorted({b for _, b in slots})
    score_uses = [e for e in re.findall(r"s_sc\[[^\]]*\]\[[^\]]*\]\[([^\]]*)\]", src)
                  if e != "ATTN_M2_STAGE"]
    check(len(slots) >= 3 and set(bases) <= {"s_lo", "c_lo"},
          f"flat stage indices are sub-block slots against a loop base, not lane indices: "
          f"{len(slots)} sites, bases {bases}")
    check(bool(score_uses) and set(score_uses) <= {"rr", "off + t", "j - b.u_lo"},
          f"every score-stage USE is union-relative by a named expression ({len(score_uses)} sites: "
          f"{sorted(set(score_uses))}); a bare lane index here is §13/§14's silent LDS over-run")
    check("const int ns = g.bhi - g.blo, off = g.blo - b.u_lo;" in src,
          "the one-pass read offsets the score stage by THIS row's slice start, not the union's")
    body = src[src.index("void attn_m2_body("):]
    bars = [i for i in range(len(body)) if body.startswith("__syncthreads", i)]
    # A `return` is legal only outside the barrier region, or in a block whose condition cannot mention
    # a lane (the shipped form's own `if (n == 0)` uniform exit has the same shape and the same right).
    late = []
    for m in re.finditer(r"\breturn\b", body):
        if not (bars[0] < m.start() < bars[-1]):
            continue
        ctx = body[max(0, body.rfind("\n", 0, m.start()) - 260):m.start()]
        cond = ctx[ctx.rfind("if ("):]
        if re.match(r"if \((n4 == 0|!any_active)\)", cond) and body[m.start():].startswith("return;"):
            continue
        late.append(m.start())
    check(not late, f"no lane-dependent `return` between the first and last barrier: {len(late)} found")
    def rule_pair(path, name):
        m = re.search(r"%s\(const CoreArgs& a, int (\w+)\) \{(.*?)\n\}" % name,
                      path.read_text(), re.S)
        if not m:
            return None
        return [re.sub(r"\s+", " ", ln.strip())
                for ln in m.group(2).strip().splitlines() if ln.strip()]

    def rule_norm(lines):
        return [re.sub(r"\bm2_", "", ln.replace("qrow", "row")) for ln in lines or []]

    same = all(rule_norm(rule_pair(M2_H, "m2_" + n)) == rule_norm(rule_pair(T.KERNEL_H, n))
               for n in ("lanes_of", "blocks_for_row"))
    check(same, "the two geometry rules are transcribed character for character from the shipped "
                "header's `lanes_of`/`blocks_for_row` (modulo the `m2_` prefix and `qrow`→`row`)")
    check(T.one_pass.__module__ == "attn_tile_model" and T.online.__module__ == "attn_tile_model",
          "one_pass/online are imported from attn_tile_model (one spelling per reduction)")
    # 4. the byte model reproduces the funding doc's closed form on the shipped side
    lanes = [i + 1 for i in range(1024)]
    sh_k, sh_v = wave_census(lanes, T.windows_of(max(lanes)), True, form="shipped")
    sigma = 1024 * 1025 // 2
    check((sh_k + sh_v) * BYTES_PER_LANE_ROW * LAYERS == sigma * 2048 * LAYERS,
          "shipped bytes == Σ_lanes x 2 KiB x 6 layers exactly (the funding doc's denominator)")
    print(f"\nselftest: {len(fails)} failures")
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU mirror and gates for prototype/attn_m2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("geometry", cmd_geometry), ("census", cmd_census), ("compare", cmd_compare),
                     ("negative", cmd_negative), ("traffic", cmd_traffic), ("selftest", cmd_selftest)):
        p = sub.add_parser(name)
        p.add_argument("--fixture", default="p300")
        p.add_argument("--anchor", default=None, help="fixture with >MAXSC decode pins, e.g. wide1024f")
        p.add_argument("--wave-rows", type=int, default=64)
        p.add_argument("--rows", type=int, default=8)
        p.add_argument("--start", type=int, default=0)
        p.add_argument("--only", default=None)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
