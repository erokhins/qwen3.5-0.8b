#!/usr/bin/env python3
"""The SHIPPED-PLACEMENT census for `attn_fattn_wave` — the spill gate and the derived-geometry canary.

WHY THIS EXISTS AND WHAT IT REFUSES TO DO. `prototype/attn_wmma/fattn_census.py` priced the PROBE. The probe
has no head axis, so its LDS figure and its 9.719 pipe-weighted slots/touch describe a kernel that cannot be
routed; carrying those numbers onto the shipping path is how claim (i) stayed believed after it had gone red.
This script reads the note block of the kernel the engine actually launches, fails on any non-zero spill (the
gate the Q placement was decided under), and re-derives its expected instruction sites from the geometry
constants in `engine/attn_fattn_kernels.h` rather than copying them from anywhere — a canary whose expected
values were hand-copied from an older geometry asserts the old geometry and passes vacuously.

    python3 prototype/attn_wmma/fattn_ship_census.py [--asm PATH] [--no-build]

Exit 0 = the shipped placement is spill-free and its instruction skeleton matches its own source. Every
number it prints is a compile-time fact (`.s` note block + static sites); nothing here touches a device.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

HEADER = "engine/attn_fattn_kernels.h"
# The gate's own build, tracked beside the gate. While this lived under gitignored `tmp/` the tracked census
# depended on a scratch script to run at all, so a clone could not reproduce it; the rule that a measurement is
# reproducible with a tool in the tree applies to a compile-only gate exactly as it does to a device one.
BUILD_SH = "prototype/attn_wmma/fattn_census_build.sh"
ASM_DIR = "/tmp/fattn_census2"
NOTE_KEYS = ("vgpr_count", "vgpr_spill_count", "sgpr_count", "sgpr_spill_count",
             "private_segment_fixed_size", "group_segment_fixed_size")


def for_bound(text, var, defines):
    """The bound of `for (int VAR = …; VAR < BOUND; ++VAR)` in the source, with BOUND resolved.

    This is how a loop's shape enters the expected instruction counts: the bound is read from the loop the
    source actually spells (resolving a `#define` through `defines`), so widening a publish block or dropping
    a stage fill moves the expectation, and a number copied out of a listing cannot move at all.
    """
    m = re.search(r"for\s*\(\s*int\s+%s\s*=\s*[\w]+\s*;\s*%s\s*<\s*([A-Za-z_]\w*|\d+)\s*;" % (var, var),
                  text)
    if not m:
        raise AssertionError("no `for (int %s = …; %s < BOUND; …)` loop found — the census names a loop the " \
                             "source no longer spells, which is a FAILURE of the census, not a free pass"
                             % (var, var))
    tok = m.group(1)
    return int(tok) if tok.isdigit() else int(defines[tok])


def enclosing_bound(text, anchor, defines):
    """The bound of the loop that immediately encloses `anchor` (nearest `for` line above it)."""
    head = text[:text.index(anchor)].split("\n")
    for line in reversed(head):
        m = re.search(r"for\s*\([^;]*;\s*(\w+)\s*<\s*([A-Za-z_]\w*|\d+)\s*;", line)
        if m:
            tok = m.group(2)
            return int(tok) if tok.isdigit() else int(defines[tok])
    raise AssertionError("no enclosing for-loop found above %r" % anchor)


def eval_factor(tok, env):
    """One factor of a geometry expression: a literal, `NAME`, or `NAME / literal`.

    A missing name is reported as the same AssertionError the caller already knows how to phrase, so a
    `KeyError` never escapes as a bare traceback from a census that simply has not been taught a constant.
    """
    tok = re.sub(r"u$|ul$|ull$", "", tok.strip())
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    m = re.fullmatch(r"(\w+)\s*/\s*(\d+)", tok)
    if m:
        if m.group(1) not in env:
            raise AssertionError("cannot evaluate %r" % tok)
        return env[m.group(1)] // int(m.group(2))
    if re.fullmatch(r"\w+", tok) and tok in env:
        return env[tok]
    raise AssertionError("cannot evaluate %r" % tok)


def eval_expr(tok, env):
    """Evaluate the expression forms this header's geometry uses: a factor, or a `*` product of factors
    (parentheses allowed, one level of nesting or more as long as they balance).

    A tiny reader rather than `eval`, and deliberately so: a census that evaluated arbitrary text out of a
    source file would be a census that runs whatever a comment happens to look like.

    The product form exists because the warp count arrived in the header's own expressions: the block declares
    `s_q[FATTN_WARPS * FATTN_MROWS * DH]` (one Q stage per warp) and the op header spells the launcher's rule
    as `#define FATTN_ROWS_PER_BLOCK (FATTN_WARPS_PER_BLOCK * 16)`. A reader that could not multiply names
    could not price either without being TOLD the warp count here, and a constant typed into the checker
    asserts the geometry it was typed in for, which is the failure this whole census was rewritten to refuse.
    """
    tok = tok.strip()
    while tok.startswith("(") and tok.endswith(")"):
        inner = tok[1:-1].strip()
        if inner.count("(") != inner.count(")"):
            break
        tok = inner
    product = 1
    for factor in tok.split("*"):
        product *= eval_factor(factor, env)
    return product


def half_loop_tiles(text, env):
    """The C-tiles one half of the epilogue stages, read from that loop's OWN two bounds.

    The loop spells them as `half * (FATTN_VTILES / 2)` and `(half + 1) * (FATTN_VTILES / 2)`, so its per-half
    trip count is the difference of two expressions and neither bound is a constant by itself: taking the upper
    bound alone — the tempting shortcut — reads 16 where the loop runs 8, and doubles every stage-write
    expectation built on it.
    """
    m = re.search(r"for \(int t = half \* \(([^)]+)\); t < \(half \+ 1\) \* \(([^)]+)\)", text)
    if not m:
        raise AssertionError("the epilogue no longer splits its C-tiles by half in the shape this census "
                             "reads. Re-derive the split from the source; do not type the answer here")
    per = re.fullmatch(r"\s*(\w+)\s*/\s*(\d+)\s*", m.group(1))
    assert per and m.group(1) == m.group(2), (
        "the epilogue's two half-bounds (%r, %r) are no longer the same scaled expression, so the split is no "
        "longer uniform and this census cannot price it" % (m.group(1), m.group(2)))
    return env[per.group(1)] // int(per.group(2))


SHARED_TYPES = {"__half": 2, "float": 4, "hip_bfloat16": 2, "char": 1, "unsigned": 4, "int": 4}
# One `__shared__` declaration, as the census reads it: type, name, raw dimension expression. The byte sum
# below and the Q-source classifier in `geometry` both go through this one pattern, so the classifier cannot
# believe in a stage the byte sum did not price — an uncounted stage is traffic hiding from the census.
SHARED_DECL = re.compile(r"__shared__\s+(?:__align__\(\d+\)\s+)?([A-Za-z_]\w*)\s+(\w+)\[([^\]]+)\]")


def shared_decl_names(body):
    """The names of the arrays this body declares `__shared__`.

    `geometry` asks one question of them: is the score leg's Q operand handed one of these names, or a pointer
    from somewhere else? A list, because a block may declare one Q stage per warp; an operand that names none
    of them is not arriving through LDS at all.
    """
    return [m.group(2) for m in SHARED_DECL.finditer(body)]


def func_body(src, name):
    """The text of one function definition, signature to closing brace, by its own name.

    The census uses it to price an operand loader by what the loader DOES (how many wide loads it issues)
    instead of by a byte count typed next to the call site. Definitions only: the `[^;{]*` between the
    argument list and the brace rejects a call, which is closed by a `;`.
    """
    m = re.search(r"\b%s\s*\([^)]*\)[^;{]*\{" % re.escape(name), src)
    if not m:
        raise AssertionError("no definition of `%s` found: the census names a function the source does not "
                             "define. Re-read the source; do not assume its body" % name)
    end = src.index("\n}\n", m.start())
    return src[m.start():end]


def shared_bytes(body, env):
    """The LDS one block declares, summed from its own `__shared__` declarations.

    Wavefronts per CU are what the residency question is about, and on this part LDS per block is what sets
    them, so a census that hard-coded 16384 could not witness a geometry change: it would go red on the very
    variant it was written to justify. An unparseable or missing declaration is a hard failure — guessing an
    element width here is guessing the residency, and a census that found no stage at all has not priced
    anything.
    """
    total, seen = 0, []
    for m in SHARED_DECL.finditer(body):
        typ, name, dims = m.group(1), m.group(2), m.group(3)
        if typ not in SHARED_TYPES:
            raise AssertionError("`%s %s[%s]`: no width for type %r. Add it to SHARED_TYPES with its size; do "
                                 "not assume 2" % (typ, name, dims, typ))
        n = 1
        for tok in dims.split("*"):
            n *= eval_expr(tok, env)
        seen.append("%s=%d" % (name, SHARED_TYPES[typ] * n))
        total += SHARED_TYPES[typ] * n
    if not seen:
        raise AssertionError("no `__shared__` declaration in the kernel body: this census cannot price "
                             "residency without knowing what a block declares")
    return total, ", ".join(seen)


def barrier_sites(text, env):
    """The `s_barrier` instructions the compiler SEES in one function body.

    One per source `__syncthreads()` call site, multiplied by the static trip count of every `#pragma unroll`
    loop that lexically encloses it: an unrolled loop emits its body that many times and a rolled one emits it
    once however many times it runs. That is the same rule this census already applies to the epilogue's LDS
    stores, applied to the barriers a multi-warp block actually pays. Before Bet 2 the whole term was the
    sentence "a one-warp block has its barriers deleted, so 0", and the count was `1 + 3 + 3` typed from an
    older listing; at 96 threads the source spells 11 call sites (two before the round, four in the rolled
    round, five in the epilogue because the two inside the half loop are emitted per unrolled half) and the
    listing agrees with the source, not with the typed 9.
    """
    total, depth, stack, pending = 0, 0, [], None
    for line in text.split("\n"):
        bare = re.sub(r"//.*$", "", line)
        if bare.strip().startswith(("*", "/*")):
            bare = ""                       # comment text must not move the brace depth
        opened, closed = bare.count("{"), bare.count("}")
        if re.search(r"#pragma\s+unroll", bare):
            pending = "roll" if re.search(r"#pragma\s+unroll\s+1\b", bare) else "unroll"
        m = re.search(r"\bfor\s*\(\s*int\s+(\w+)\s*=\s*([^;]+);\s*\1\s*<\s*([^;]+);", bare)
        weight = 1
        if m and pending == "unroll":
            try:
                weight = eval_factor(m.group(3), env) - eval_factor(m.group(2), env)
            except AssertionError:
                weight = 1                  # bounds resolvable only at run time: emitted once
        if m:
            stack.append((depth + opened, weight))
            pending = None          # the pragma belongs to the loop it sits above; consume it there
        if "__syncthreads()" in bare:
            sites = 1
            for _, w in stack:
                sites *= w
            total += sites
        depth += opened - closed
        stack = [entry for entry in stack if entry[0] <= depth]
    return total


def define_value(text, name, env):
    """The value of a `#define NAME …` line, via `eval_expr`.

    Reading the geometry out of the header instead of restating it here is what lets `--selftest` prove the
    derived numbers depend on the source: a constant typed into this file could have been copied from a listing.
    """
    m = re.search(r"^#define\s+%s\s+(.+)$" % name, text, re.M)
    if not m:
        raise AssertionError("no `#define %s` in the source: the census names a constant the kernel no "
                             "longer has" % name)
    tok = re.split(r"//", m.group(1))[0].strip()
    try:
        return eval_expr(tok, env)
    except AssertionError:
        raise AssertionError("cannot evaluate `#define %s %s`: extend eval_expr, do not hard-code the value"
                             % (name, tok))


def geometry(text=None, op=None):
    """The trip counts, read out of the kernel's own source rather than restated here.

    `text` / `op` are the header texts, so `--selftest` can perturb one geometric quantity at a time and
    require the expectation to move. A function that could not be shown to depend on its source is a function
    whose constants might as well have been typed.
    """
    src = open(HEADER).read() if text is None else text
    op = open("prototype/attn_decode/attn_decode_kernels.h").read() if op is None else op
    g = {}
    g["DH"] = define_value(op, "DH", {})
    g["KTILES"] = g["VTILES"] = g["DH"] // 16
    # FATTN_WARPS is read like every other constant, from the header that declares it. It is not a knob this
    # census may hold a number for: the Q stage's dimension expression, the block's LDS, the publish width and
    # the fold's slot count are all products through it, and each of those must be able to move on its own.
    for name in ("FATTN_MROWS", "FATTN_MLANES", "FATTN_THREADS", "FATTN_TBLK", "FATTN_PBLK",
                 "FATTN_TROW", "FATTN_WARPS"):
        g[name] = define_value(src, name, g)
    # THE WAVEFRONT WIDTH, out of the header's own `static_assert(FATTN_THREADS == 32 * FATTN_WARPS)` rather
    # than typed here: that line is the source stating that threads and warps are one fact, so the census
    # reads the statement instead of restating it, and mirrors the assert (bench/performance-status.md 34).
    m = re.search(r"static_assert\(FATTN_THREADS == (\d+) \* FATTN_WARPS", src)
    assert m, ("no `static_assert(FATTN_THREADS == <lanes> * FATTN_WARPS)` in %s: the census prices warps, "
               "so it has to read the wavefront width from the source that asserts it rather than assume 32"
               % HEADER)
    g["warp_lanes"] = int(m.group(1))
    assert g["FATTN_THREADS"] == g["warp_lanes"] * g["FATTN_WARPS"], (
        "`#define FATTN_THREADS %d` and `FATTN_WARPS %d` at %d lanes per warp do not agree: the census "
        "cannot price a block whose thread count and warp count are two different facts (the kernel's own "
        "static_assert would refuse this source too)"
        % (g["FATTN_THREADS"], g["FATTN_WARPS"], g["warp_lanes"]))
    g["one_warp_per_block"] = g["FATTN_THREADS"] == g["warp_lanes"]
    # THE LAUNCHER AND THE KERNEL, ONE FACT (bench/performance-status.md 34: a geometry fact in two headers can
    # be committed in one of them, and no single-file review sees the missing half). `fattn_rtiles_of` counts
    # blocks of FATTN_ROWS_PER_BLOCK rows in the OP header; the kernel carries FATTN_WARPS row-blocks of
    # FATTN_MROWS rows. The kernel static_asserts the two are equal; so does this census, because the census is
    # the gate that runs on a compile of the tree and the assert is the fact a half-committed change breaks.
    warps_per_block = define_value(op, "FATTN_WARPS_PER_BLOCK", g)
    g["FATTN_WARPS_PER_BLOCK"] = warps_per_block   # `FATTN_ROWS_PER_BLOCK` is spelled in terms of it
    rows_per_block = define_value(op, "FATTN_ROWS_PER_BLOCK", g)
    assert warps_per_block == g["FATTN_WARPS"], (
        "`FATTN_WARPS_PER_BLOCK %d` in prototype/attn_decode/attn_decode_kernels.h and `FATTN_WARPS %d` in %s "
        "disagree: the launcher would budget row-blocks this kernel does not own, and `attn_combine` would "
        "merge scratch for the rows nobody wrote (the other half of 20.1)"
        % (warps_per_block, g["FATTN_WARPS"], HEADER))
    assert rows_per_block == g["FATTN_WARPS"] * g["FATTN_MROWS"], (
        "the launcher counts blocks of %d rows and the kernel carries %d x %d: a row neither side owns is "
        "invisible in both" % (rows_per_block, g["FATTN_WARPS"], g["FATTN_MROWS"]))
    body = src[src.index("void attn_fattn_body("):]
    body = body[:body.index("__device__ __host__ inline bool fattn_plan_ok")]
    epilogue = src[src.index("void fattn_transpose_out("):]
    epilogue = epilogue[:epilogue.index("warp-wide min and max")]
    stage_row = src[src.index("void fattn_stage_row("):]
    stage_row = stage_row[:stage_row.index("struct FattnRow")]
    # WHICH ROLL THE CENSUS PRICES: the SHIPPED one. Two spellings live in this body now (`single` and bet
    # A4-L1's `two`, selected by `--attn-fattn-roll`), both under their own `#pragma unroll 1`, so asking the
    # body "is there a rolled loop anywhere" no longer identifies the shipped leg -- it would stay true even if
    # the shipped roll stopped being rolled, which is a check that cannot fail. The shipped roll is the loop
    # that steps `++t`; the two-accumulator loop steps `t += 2`.
    g["score_rolled"] = bool(re.search(
        r"#pragma unroll 1\s*\n\s*for \(int t = 0; t < FATTN_KTILES; \+\+t\)", body))
    # THE HALVES LOOP: the epilogue stages F32 through the dead K/V stage in this many passes, and every
    # publish-side count below is a multiple of it.
    g["transpose_halves"] = for_bound(epilogue, "half", g)
    # THE PUBLISH LOOP: each lane publishes this many blocks per half, each block this many floats.
    g["publish_blocks"] = for_bound(epilogue, "k", g)
    g["publish_block_floats"] = for_bound(epilogue, "u", g)
    # THE PUBLISHING SET'S WIDTH, read out of the coverage assert that states it. The header spells the assert
    # with a literal 32 and says why in its own comment: "The 32 is a warp and not FATTN_THREADS, because the
    # stage and the publishing set are both per warp". At three warps that distinction is load-bearing — a
    # denominator of FATTN_THREADS (96) would divide the stage three ways and make a per-warp publish look like
    # a third of the traffic it is — so the census reads the width from the assert and cross-checks it against
    # the warp count rather than picking one of the two names by taste.
    m = re.search(r"static_assert\(FATTN_MROWS \* FATTN_TROW == (\d+) \* FATTN_PBLK \* FATTN_TBLK", src)
    assert m, ("no `static_assert(FATTN_MROWS * FATTN_TROW == <lanes> * FATTN_PBLK * FATTN_TBLK)` in %s: that "
               "assert is the publish-coverage claim, and its literal is the number of lanes that publish one "
               "stage. Re-read the source; do not type 32 here" % HEADER)
    g["publish_lanes"] = int(m.group(1))
    assert g["publish_lanes"] == g["warp_lanes"], (
        "the publish coverage is written for %d lanes but a warp is %d: the transpose stage is per warp, so "
        "one of the two is stale" % (g["publish_lanes"], g["warp_lanes"]))
    assert g["publish_blocks"] == g["FATTN_PBLK"] and g["publish_block_floats"] == g["FATTN_TBLK"], (
        "the publish loop runs %s x %s but FATTN_PBLK/FATTN_TBLK are %d/%d: the coverage static_assert in "
        "the header is then asserting a shape the loop does not run, which is exactly how the second version "
        "of the epilogue published 16 of every row's 256 dims and left `attn_combine` merging scratch"
        % (g["publish_blocks"], g["publish_block_floats"], g["FATTN_PBLK"], g["FATTN_TBLK"]))
    # THE STAGE WRITE of the same epilogue: one F32 per C-tile register, per C-tile staged per half, per half.
    # The per-half tile count is the split the loop itself spells (see `half_loop_tiles`), and the product is
    # asserted against the accumulator: a half that stopped staging all its tiles would silently leave part of
    # the output unsaved, which is the same class of defect as a partial publish.
    g["FATTN_VTILES"] = g["VTILES"]        # the alias the header's loop bounds use
    g["stage_tiles_per_half"] = half_loop_tiles(epilogue, g)
    g["stage_write_per_tile"] = for_bound(epilogue, "l", g)
    assert g["transpose_halves"] * g["stage_tiles_per_half"] == g["VTILES"], (
        "the epilogue stages %d x %d = %d C-tiles but the P*V leg produces %d: the accumulator is only partly "
        "saved, which is the same class of defect as a partial publish"
        % (g["transpose_halves"], g["stage_tiles_per_half"],
           g["transpose_halves"] * g["stage_tiles_per_half"], g["VTILES"]))
    # THE V-STAGE FILL: one narrow LDS store per dim of the run the lane moves per trip.
    g["v_narrow_stores"] = enclosing_bound(body, "s_kv[fattn_v_addr(lane * 8 + t, kv)]", g)
    # THE CROSS-WARP FOLD, and the traffic it is worth. Three warps cannot agree on the round's range without
    # writing it down, so the warp-level min/max/mask go through `fold[]` slots borrowed from the front of the
    # K/V stage and come back over a barrier. This is NEW traffic that the one-warp form did not have (it is
    # also why the census's `s_barrier` term stopped being 0), and it is small enough that a listing could hide
    # it: 9 int reads and 3 int writes. So it is derived from the fold's own shape - how many slots the writer
    # writes, how many distinct slots the reader reads, and that reader's bound - and asserted slot-for-slot,
    # because a fold slot that is written and never read is a span that never reached the block, which is the
    # dead-lock-on-a-diverged-round defect this geometry exists to be careful about.
    mf = re.search(r"int\*\s*fold\s*=\s*reinterpret_cast<int\*>\(\s*(\w+)\s*\)", body)
    assert mf, ("no `int* fold = reinterpret_cast<int*>(...)` in %s: at %d warps the round range has to be "
                "folded through shared memory and this census prices that fold by reading it. If the fold is "
                "gone, the barriers and this term both have to be re-derived from the source"
                % (HEADER, g["FATTN_WARPS"]))
    g["fold_backing"] = mf.group(1)
    assert g["fold_backing"] in shared_decl_names(body), (
        "the fold borrows `%s`, which the block does not declare `__shared__`: the round range would be folded "
        "through memory no one allocated" % g["fold_backing"])
    b1 = body.index("__syncthreads", mf.end())
    b2 = body.index("__syncthreads", b1 + len("__syncthreads"))
    fold_write, fold_read = body[mf.end():b1], body[b1:b2]
    g["fold_write_slots"] = len(re.findall(r"\bfold\[[^\]]+\]\s*=", fold_write))
    g["fold_reads_per_trip"] = len(set(re.findall(r"\bfold\[[^\]]+\]", fold_read)))
    g["fold_trips"] = for_bound(body, "i", g)
    assert g["fold_trips"] == g["FATTN_WARPS"], (
        "the fold loop runs %d times but the block has %d warps: some warp's span is never folded in, and the "
        "round range it disagrees about is a barrier the other warps wait at forever"
        % (g["fold_trips"], g["FATTN_WARPS"]))
    assert g["fold_write_slots"] == g["fold_reads_per_trip"] and g["fold_write_slots"], (
        "the fold writes %d slots and reads %d: a slot written and not read, or read and not written, means "
        "the block range is not the one the warps computed"
        % (g["fold_write_slots"], g["fold_reads_per_trip"]))
    g["fold_read_B"] = g["fold_reads_per_trip"] * g["fold_trips"] * 4
    g["fold_write_B"] = g["fold_write_slots"] * 4
    # THE BARRIERS the compiler sees: the source's own call sites, weighted by the loops that unroll them.
    # At one warp per block every one of them is deleted (measured 0 in the shipped listing); above that they
    # are real instructions and the round pays them, which is Bet 2's structural price.
    g["barrier_sites"] = barrier_sites(body, g) + barrier_sites(epilogue, g)
    # THE STAGE FILLS: `fattn_stage_row` moves one uint4 of bf16 (8 halves = 16 B) per call, so each call site
    # is one wide global read and one wide LDS write. The transposed V fill is an inline uint4 of its own.
    g["stage_row_halves"] = for_bound(stage_row, "t", g)
    # Call SITES, at statement position: `body` starts at `attn_fattn_body`, so the definition of
    # `fattn_stage_row` (whose line starts with `void`) and the prose that names it are both excluded.
    g["stage_row_sites"] = len(re.findall(r"^\s*fattn_stage_row\(", body, re.M))
    g["global_stage_fills"] = g["stage_row_sites"] + 1
    # THE Q OPERAND'S SOURCE — the one geometric fact about this form that is not a loop bound, and the reason
    # this census was left red by the Q un-staging (bench/performance-status.md §31). The two spellings move two
    # traffic classes in OPPOSITE directions:
    #   staged (placement (B)) — `fattn_load_tile(&s_q[...])`: the operand's bytes are LDS reads, and a
    #                            `fattn_stage_row` fills the stage once per launch (16 B global + 16 B LDS).
    #   global (Bet 1)         — `fattn_load_q_tile(qrow + 16*t)`: the same bytes are GLOBAL reads, one tile per
    #                            rolled score site, and the stage plus its fill are gone.
    # The model asks the source one question rather than being told one: does the operand's own argument name an
    # array the body declared `__shared__`? A banner (`q=global`) and a comment can both lie; the call site that
    # produces the operand cannot, and the byte cost comes out of the loader function that call site names.
    # K is deliberately NOT classified: this form's score leg has exactly one staged operand by construction and
    # un-staging K would make the stage pointless rather than cheaper, so if that ever happens the listing moves
    # out from under this model and the census goes red, which is the correct answer to an unpriced change.
    mq = re.search(r"const fattn_ARegs second\s*=\s*(\w+)\(([^;]*?)\)\s*;", body)
    assert mq, ("no `const fattn_ARegs second = <loader>(...)` in the score leg: this census classifies the Q "
                "operand by its own call site and the source no longer spells one. Re-read the source; do not "
                "type the answer here")
    g["q_producer"] = mq.group(1)
    g["q_stage"] = next((nm for nm in shared_decl_names(body)
                         if re.search(r"\b%s\b" % nm, mq.group(2))), "")
    g["q_from_global"] = 0 if g["q_stage"] else 1
    if g["q_from_global"]:
        # Two `uint4` derefs in the loader = 32 B of global per tile, read off the loader rather than asserted.
        g["q_tile_B"] = 16 * len(re.findall(r"reinterpret_cast<const\s+uint4\s*\*",
                                            func_body(src, g["q_producer"])))
        assert g["q_tile_B"], ("`%s` issues no wide load, so this census has the wrong loader or the Q operand "
                               "is built some other way. Read the source again; do not type 32 here"
                               % g["q_producer"])
    else:
        # A staged operand needs a writer: an LDS operand nothing fills is a slot of scratch, which is §29's NaN
        # one layer down. This is the check that makes `q_stage` a claim and not a name match.
        assert g["q_stage"] and re.search(r"fattn_stage_row\(\s*&%s\[" % g["q_stage"], src), (
            "the model says Q is staged but the operand names no stage it declared (`%s`), or the stage it names "
            "has no writer: an LDS operand with no writer is uninitialised memory, not a placement"
            % (g["q_stage"] or "<none>"))
        g["q_tile_B"] = 0
    # THE PARTIAL HEADER: the (m, l) pair, two F32 per covered row. It is the only global write that is not
    # the publish loop's, and the second version of the epilogue got the publish shape wrong beside it.
    g["part_header_floats"] = 2
    # THE STAGES, as declared. The residency question is a question about this number, so it is read off the
    # declarations and not assumed from a sentence about them.
    g["lds_declared_B"], g["lds_declared"] = shared_bytes(body, g)
    g["launch_bounds"] = "__launch_bounds__(FATTN_THREADS)" in src
    return g


def derive(g):
    """What the listing must show, derived from the geometry above: traffic in bytes/lane, structure in sites.

    Static, not per-round dynamic cost: the score leg is rolled, so its operand reads appear once and run
    KTILES times. Deriving dynamic cost here would fail against the listing for a reason that has nothing to
    do with the kernel — the same class of error as a canary whose expected numbers were copied from a probe
    with a different shape.

    NOTHING HERE IS COPIED FROM A LISTING. Every term is a loop or a declaration the source spells (see
    `geometry`), and `--selftest` fails the census if any term stops depending on the quantity it claims to
    come from, which is the guard against "fixing" a red census by editing the number to match.
    """
    score_sites = 1 if g["score_rolled"] else g["KTILES"]
    # One operand tile is 16 halves = 32 B. The score leg always reads its K operand from the stage; it reads
    # its Q operand from the stage only if the source declares a Q stage (`q_from_global`, read off the
    # operand's own call site in `geometry`). The P*V leg has one staged operand each, because the P operand
    # lives in registers — which is the whole point of the C->B exchange.
    operand_bytes = (score_sites * (2 - g["q_from_global"]) + g["VTILES"]) * 16 * 2
    # THE PUBLISH SIDE — the term the pre-NaN-fix model did not have at all. Per half the stage holds
    # FATTN_MROWS x FATTN_TROW floats; every one of them is read back, exactly once, in blocks of
    # FATTN_TBLK floats and written to the partial slot, so the stage read and the partial write are the same
    # byte count from two different loops.
    publish_bytes = g["transpose_halves"] * g["publish_blocks"] * g["publish_block_floats"] * 4
    # The same number from the OTHER side of the header's coverage assert: the stage's own float count, split
    # across the block's lanes. If the two disagree, the publish loop and the stage's coverage disagree, and a
    # partial publish is exactly the defect that made every covered row NaN.
    publish_bytes_stage = (g["transpose_halves"] * g["FATTN_MROWS"] * g["FATTN_TROW"] * 4
                           // g["publish_lanes"])
    assert publish_bytes == publish_bytes_stage, (
        "the publish loop accounts for %d B per lane but the stage holds %d B per lane: the loop does not "
        "cover the stage it is publishing, so `attn_combine` reads uninitialised slots for the rest"
        % (publish_bytes, publish_bytes_stage))
    return {
        # The LDS a block declares, which on this part is what sets blocks/CU and therefore wavefronts/CU —
        # the quantity the residency question is about. Checked against the note block, not against a sentence.
        "lds_B": g["lds_declared_B"],
        # one v_wmma site in the rolled score loop, one per tile in the unrolled P*V loop
        "v_wmma": score_sites + g["VTILES"],
        # LDS READS: the WMMA operand tiles that arrive through LDS (K always; Q only where it is staged) plus
        # the epilogue's publish of the transpose stage, plus the cross-warp fold's int reads (its own slot
        # count x one trip per warp) — traffic the one-warp form did not have, and small enough that a listing
        # could hide it, so it is read off the fold's own shape in `geometry`.
        "lds_read_B": operand_bytes + publish_bytes + g["fold_read_B"],
        # LDS WRITES: the Q and K stage fills (one uint4 per `fattn_stage_row` call), the transposed V fill
        # (one 16-bit store per dim of its loop), the transpose stage (one F32 per C-tile register per tile per
        # half), and the fold's own writes.
        "lds_write_B": (g["stage_row_sites"] * g["stage_row_halves"] * 2
                        + g["v_narrow_stores"] * 2
                        + g["transpose_halves"] * g["stage_tiles_per_half"]
                        * g["stage_write_per_tile"] * 4
                        + g["fold_write_B"]),
        # GLOBAL READS: the stage fills, one uint4 per lane per trip of each fill loop, plus the Q operand's own
        # wide loads when the score leg builds that operand from global instead of reading it from a stage.
        "global_read_B": g["global_stage_fills"] * 16 + score_sites * g["q_tile_B"],
        # GLOBAL WRITES: the publish loop's bytes into `part`, plus the (m, l) header the epilogue writes
        # beside it. This is the term the broken epilogue got wrong in a way no bound-style assert could see:
        # it wrote 16 of every row's 256 dims, so 240 dims per row of the partial slot were never written.
        "global_write_B": publish_bytes + g["part_header_floats"] * 4,
        # 5 butterflies each in warp_min and warp_max (outside the round), the round's 1 max fold + 8 C->B
        # exchanges, and the 1 deferred rowsum combine: 10 outside, 9 in the round, 1 after it.
        "cross_lane": 10 + 9 + 1,
        # Barrier SITES. At one warp per block a `__syncthreads()` has no second wavefront to sequence, and
        # the measured listing of that geometry contained ZERO `s_barrier` instructions — so the third barrier
        # per round that aliasing V over K costs was free there. Above one warp they are real instructions, and
        # the expectation is derived by `barrier_sites` from the source's own call sites weighted by the loops
        # that statically unroll them (two before the round, four in the rolled round, five in the epilogue),
        # not typed from a listing. The round range must be block-uniform or the block dead-locks, which is why
        # this line is a fact the census insists on rather than a comment.
        "s_barrier": 0 if g["one_warp_per_block"] else g["barrier_sites"],
        # The ordering cost does not disappear, it moves here: LDS RAW within a wavefront is carried by
        # `s_waitcnt lgkmcnt(0)`, which the compiler inserts whether or not a barrier survives.
        "s_wait_lgkmcnt0": 3,
    }


# The traffic classes, as BYTES PER LANE rather than as site counts: whether LLVM moves a staged row with one
# `ds_load_b128` or with two `ds_load_b64` is a codegen choice, and a census that keyed on the width it happened
# to see would go red for a reason that tells the reader nothing. Bytes are what the loops actually demand, so
# bytes are what the model predicts — and any LDS/global mnemonic that is not in one of these tables is a hard
# failure, because an unclassified access is exactly how traffic hides from a census.
LDS_READ_BYTES = {"ds_load_b128": 16, "ds_load_2addr_b128": 32, "ds_load_b64": 8, "ds_load_b32": 4,
                  "ds_load_u16": 2, "ds_load_2addr_b32": 8, "ds_load_2addr_stride64_b32": 8}
LDS_WRITE_BYTES = {"ds_store_b128": 16, "ds_store_2addr_b128": 32, "ds_store_b32": 4,
                   "ds_store_2addr_b32": 8, "ds_store_b16": 2, "ds_store_b16_d16_hi": 2,
                   "ds_store_u16": 2}
GLOBAL_READ_BYTES = {"global_load_b128": 16, "global_load_b64": 8, "global_load_b32": 4,
                     "global_load_u16": 2}
GLOBAL_WRITE_BYTES = {"global_store_b128": 16, "global_store_b64": 8, "global_store_b32": 4,
                      "global_store_u16": 2}
# Cross-lane movement is not traffic and is counted in sites: it is the fragment exchange, and one butterfly
# fewer is a numerics defect (a half denominator per row), not a bandwidth one.
CROSS_LANE_SITES = ("ds_bpermute_b32", "ds_swizzle_b32")


def body_of(asm_text, label_re=r"^(_ZN6engine7kernels\w+):"):
    """(label, body) pairs, from each kernel's label to its `.Lfunc_end`.

    Splitting the file on a `.text` header (an earlier attempt here) leaves no chunk containing the label, so
    every count comes back 0 and the census FAILs a healthy kernel — the counting twin of the empty-parse
    failure in `prototype/gdn_collapse/results.md` §1.
    """
    out = []
    for m in re.finditer(label_re, asm_text, re.M):
        rest = asm_text[m.end():]
        end = rest.find(".Lfunc_end")
        out.append((m.group(1), rest[:end if end > 0 else len(rest)]))
    return out


def n(body, mnemonic):
    return len(re.findall(r"^\s+" + mnemonic + r"(\s|$)", body, re.M))


def count_body(body):
    """This census's view of one kernel body: traffic in bytes/lane, structure in sites, plus the leftovers."""
    def total(classes):
        return sum(n(body, mnemonic) * w for mnemonic, w in classes.items())
    seen = set()
    for line in body.split("\n"):
        m = re.match(r"\s+([a-z][a-z0-9_]*)", line)
        if m and re.match(r"(ds|global)_", m.group(1)):
            seen.add(m.group(1))
    modelled = (set(LDS_READ_BYTES) | set(LDS_WRITE_BYTES) | set(GLOBAL_READ_BYTES)
                | set(GLOBAL_WRITE_BYTES) | set(CROSS_LANE_SITES))
    return {
        "v_wmma": n(body, r"v_wmma_\w+"),
        "lds_read_B": total(LDS_READ_BYTES),
        "lds_write_B": total(LDS_WRITE_BYTES),
        "global_read_B": total(GLOBAL_READ_BYTES),
        "global_write_B": total(GLOBAL_WRITE_BYTES),
        "cross_lane": sum(n(body, mnemonic) for mnemonic in CROSS_LANE_SITES),
        "s_barrier": n(body, r"s_barrier"),
        "s_wait_lgkmcnt0": n(body, r"s_waitcnt lgkmcnt\(0\)"),
        "_unclassified": sorted(m for m in seen if m not in modelled and n(body, m)),
    }


def counts(asm_text):
    return {label: count_body(body) for label, body in body_of(asm_text)}


def note_blocks(asm_text):
    """The AMDHSA YAML note blocks, keyed by mangled and by short name.

    The YAML lists one entry per kernel starting at `  - .args:` and names it later with `.name:`. The
    `.amdhsa_kernel NAME` lines inside `.text` are the ELF section directive, not the note; splitting on
    those (or on `---`) yields zero blocks and a census that reports the kernel is missing when it is right
    there in the file.
    """
    blocks = {}
    if "amdhsa.kernels:" not in asm_text:
        return blocks
    yaml = asm_text[asm_text.index("amdhsa.kernels:"):]
    for blk in re.split(r"^  - \.args:", yaml, flags=re.M)[1:]:
        m = re.search(r"^\s+\.name:\s*([^\s]+)", blk, re.M)
        if not m:
            continue
        full = m.group(1)
        rec = {}
        for k in NOTE_KEYS:
            mm = re.search(r"\.%s:\s+(\S+)" % k, blk)
            rec[k] = int(mm.group(1)) if mm else -1
        blocks[full] = rec
        blocks[re.sub(r"^_ZN6engine7kernels\d+", "", full).split("E")[0]] = rec
    return blocks


def apply_text_edits(src, steps, what):
    """Apply literal source edits, refusing loudly if a needle has gone.

    A mutation whose needle no longer matches is a mutation that stopped testing something, and the history of
    this file is a list of checks that quietly stopped testing things. So it raises, and `selftest` reports the
    raise as an escape rather than a pass.
    """
    for needle, repl in steps:
        if needle not in src:
            raise AssertionError("%s can no longer be spelled: %r is not in %s. Fix the mutation; do not delete "
                                 "the case" % (what, needle[:70], HEADER))
        src = src.replace(needle, repl, 1)
    return src


# THE Q PLACEMENT, as text. The shipped form stages Q (one 8 KiB stage per warp). Bet 1 of
# `tmp/gcc/fattn_residency.md` 5 is the placement that stopped staging it and paid 50 spilled VGPRs, and Bet 2
# refuted doing the same at three warps (V6: 73 spills, 284 B/lane). Both spellings have to stay spellable here:
# the census classifies the Q operand by the call site that PRODUCES it, and a classification no text can
# disagree with is a comment, not a check.
Q_STAGE_DECL = "    __shared__ __align__(16) __half s_q[FATTN_WARPS * FATTN_MROWS * DH];"
Q_FILL_LOOP = """    for (int qr = 0; qr < FATTN_MROWS; ++qr) {           // every slot of THIS warp's stage, exactly once
        const int qrs = qr < qrmax ? qr : qrmax - 1;
        if (qrs < 0) break;                             // this whole row-block is past the last row
        fattn_stage_row(&s_q[w * FATTN_MROWS * DH + fattn_q_addr(qrs, 0)],
                        a.qrope + (size_t)ROW_H * (rb * FATTN_MROWS + qrs) + (size_t)head0 * DH, lane);
    }"""
Q_OPERAND = ("                const fattn_ARegs first = fattn_score_k_tile(s_kv, lane, t);\n"
             "                const fattn_ARegs second = fattn_score_q_tile(s_q, w, lane, t);")
Q_OPERAND_GLOBAL = ("                const fattn_ARegs first = fattn_score_k_tile(s_kv, lane, t);\n"
                    "                const fattn_ARegs second = fattn_load_tile(a.qrope + (size_t)ROW_H "
                    "* (rb * FATTN_MROWS + fattn_ab_row(lane)) + (size_t)head0 * DH + 16 * t);")
# WHY BOTH SPELLINGS OF THE OPERAND NOW CALL A HELPER: `--attn-fattn-roll` put a second roll loop beside this
# one, and two roll spellings each writing their own `&s_q[...]` is §20.1's aliasing shape. `fattn_score_k_tile`
# and `fattn_score_q_tile` are the one spelling of the two staged operands, so the classifier still reads the
# Q operand's source off its own call site -- the helper's ARGUMENT is what names (or fails to name) a
# `__shared__` array, and the flip below replaces that call site with a global row exactly as it did before.



def q_unstaged_operand_text(src):
    """The shipped header with ONLY the score leg's Q operand handed a global row instead of the stage.

    One edit, because it is the smallest spelling of the question the classifier answers: `geometry` decides
    `q_from_global` from the operand's own call site, so only a flip of that call site can falsify it. The stage
    and its fill stay in place deliberately — if the classifier keyed on the presence of a declaration instead of
    on the operand, this text would move neither traffic class and the selftest would say so.
    """
    return apply_text_edits(src, [(Q_OPERAND, Q_OPERAND_GLOBAL)], "the Q operand flip")


def q_unstaged_text(src):
    """The whole Bet-1 placement at three warps: no Q stage, no Q fill, operand built from global.

    Three real edits, so the case cannot rot into a flag. Judging this model against the shipped listing must go
    red on LDS traffic and global traffic in OPPOSITE directions, which is the state 31 left this census in, but
    reproduced rather than remembered.
    """
    return apply_text_edits(src, [
        (Q_STAGE_DECL, ""),
        (Q_FILL_LOOP, "    const hip_bfloat16* qrow = a.qrope + (size_t)ROW_H * (rb * FATTN_MROWS "
                      "+ fattn_ab_row(lane)) + (size_t)head0 * DH;"),
        (Q_OPERAND, Q_OPERAND_GLOBAL),
    ], "the Q un-staging")


def q_shared_stage_text(src):
    """The geometry error of `fattn_ship_results.md` 2, spelled against today's source: ONE Q stage for the
    block's warps instead of one per warp. It is the case that proves `lds_B` really derives from the warp count
    rather than being the number 32768 wearing a formula, and it is the only one of these edits that leaves the
    code compiling-looking while making three warps write over one another's operands.
    """
    return apply_text_edits(src, [(Q_STAGE_DECL.replace("FATTN_WARPS * FATTN_MROWS * DH", "FATTN_MROWS * DH"),
                                   Q_STAGE_DECL)], "the one-stage-per-block error")


def selftest(asm_text):
    """Inject-and-must-fail over BOTH sides of the census; returns the list of escapes (empty = teeth).

    Run it with `--selftest`. This is the required witness that the re-derived expectation is a derivation and
    not the listing's numbers wearing a formula, and that the comparison can still see a defect: a check whose
    expected value was copied from the listing, or whose comparison only bounds one direction, passes every
    injection here and looks identical to a correct one in a normal run.
    """
    g = geometry()
    notes = note_blocks(asm_text)
    clean_note = dict(notes["attn_fattn_wave"], vgpr_spill_count=0, private_segment_fixed_size=0)
    bodies = [body for label, body in body_of(asm_text) if label.endswith("attn_fattn_waveE8CoreArgs")]
    if len(bodies) != 1:
        return ["selftest needs exactly one attn_fattn_wave body to mutate, found %d" % len(bodies)]
    body = bodies[0]
    escapes = []
    src = open(HEADER).read()
    op = open("prototype/attn_decode/attn_decode_kernels.h").read()
    try:
        flip_src = q_unstaged_operand_text(src)
    except AssertionError as exc:
        flip_src = None
        escapes.append(str(exc))

    def red_after(mutated_body, geo=None):
        """The failures a mutation ADDS. A mutation that goes red only because the base was already red proves
        nothing, so the baseline is subtracted here once rather than at each call site."""
        _, fails, _ = compare(geo or g, clean_note, count_body(mutated_body), ack_spill="selftest")
        return [f for f in fails if f not in red_after.base]

    red_after.base = []
    base_fails = compare(g, clean_note, count_body(body), ack_spill="selftest")[1]
    red_after.base = base_fails
    if base_fails:
        # The census is red on the shipped tree for reasons named in its own report, so a naive injection test on
        # top of it would credit every injection with the failures the base already had. `red_after` subtracts
        # the baseline once, so an injection must still ADD a failure that is its own -- and the red lines are
        # PRINTED here rather than hidden, and the run's exit status stays non-zero for them. What this refuses to
        # do is throw away the teeth of the injections because the base is red: a selftest that cannot run on a
        # tree that has a known finding is a selftest that stops witnessing exactly when it is needed.
        print("selftest: NOTE — the base listing is NOT green (%d line(s)): %s"
              % (len(base_fails), " | ".join(f[:110] for f in base_fails)))
        print("selftest:       every injection below must still ADD a failure the base does not already have.")

    def drop(text, mnemonic, count=1):
        return re.sub(r"^(\s+)" + mnemonic + r"(\s|$)", r"\1v_noop\2", text, count=count, flags=re.M)

    def inject(text, line):
        i = text.index("\n")
        return text[:i] + "\n\t" + line + text[i:]

    # ---- 1. THE LISTING SIDE: each injection must turn the census red, and blame the right class.
    CASES = [
        ("one extra ds_load_b128 (an LDS read no loop accounts for)",
         inject(body, "ds_load_b128 v[1:4], v13 offset:8704"), "lds_read_B"),
        ("one ds_load_b128 MISSING (a publish loop that stopped covering the stage)",
         drop(body, "ds_load_b128"), "lds_read_B"),
        ("one extra global_load_b128 (a staged operand started coming from global)",
         inject(body, "global_load_b128 v[1:4], v13"), "global_read_B"),
        ("one extra global_store_b128 (a global write the loops do not account for)",
         inject(body, "global_store_b128 v[1:4], v13"), "global_write_B"),
        ("one ds_bpermute_b32 MISSING (D-4: half a denominator per row)",
         drop(body, "ds_bpermute_b32"), "cross_lane"),
        ("one v_wmma MISSING (a tile of the round went away)",
         drop(body, r"v_wmma_\w+"), "v_wmma"),
        ("one s_barrier APPEARED (the block is no longer one wavefront)",
         inject(body, "s_barrier"), "s_barrier"),
        ("every s_waitcnt lgkmcnt(0) REMOVED (the LDS ordering vanished with the barriers)",
         drop(body, r"s_waitcnt lgkmcnt\(0\)", 0), "s_wait_lgkmcnt0"),
        ("an access mnemonic this census does not model (ds_read_b128) appeared",
         inject(body, "ds_read_b128 v[1:4], v3"), "unclassified"),
    ]
    # DERIVED, not hand-listed: one "one access of this class disappeared" case per access mnemonic the listing
    # actually contains, in each of the four traffic classes. A hand-listed mnemonic can simply stop being
    # emitted — the transpose stage's F32 writes have come out as `ds_store_b32` and as `ds_store_2addr_b32`
    # from two builds of near-identical source, and the (m, l) header as `global_store_b64` beside
    # `global_store_b128` for the publish — and an injection aimed at a mnemonic with zero sites cannot fail.
    # That is the empty-must-catch trap AGENTS.md records for `partalias` and `mixed_bw`, one level down: the
    # must-catch set has to be read off the build, not from a list of what an earlier build did.
    for classes, key, what in ((LDS_READ_BYTES, "lds_read_B", "LDS read"),
                               (LDS_WRITE_BYTES, "lds_write_B", "LDS write"),
                               (GLOBAL_READ_BYTES, "global_read_B", "global read"),
                               (GLOBAL_WRITE_BYTES, "global_write_B", "global write")):
        for mnemonic in sorted(classes):
            if n(body, mnemonic):
                CASES.append(("one %s MISSING (a %d-B %s the loops no longer perform)"
                              % (mnemonic, classes[mnemonic], what), drop(body, mnemonic), key))
    for name, mutated, key in CASES:
        extra = [f for f in red_after(mutated) if f not in base_fails]
        if not extra:
            escapes.append("the census did not notice: %s" % name)
            continue
        joined = " ".join(extra)
        if key == "unclassified":
            if "unclassified" not in joined:
                escapes.append("the census saw an extra access but did not call it unclassified: %s" % name)
        elif key not in joined:
            escapes.append("the census noticed %s but blamed the wrong class: %s" % (name, joined[:140]))

    # ---- 1b. THE Q-SOURCE CROSS-CHECK: judge this listing (Q staged in `s_q`, which is what the shipping header
    # spells) against the model the un-staged-operand header would derive. The model then believes the score leg
    # reads Q from global while the code reads it from LDS, and the two traffic classes must disagree in OPPOSITE
    # directions (LDS short by exactly the operand tile, global long by it) with the census naming both. A Q-source
    # claim a listing cannot contradict is a comment about the source, not a check on it.
    if flip_src is not None:
        cross = red_after(body, geometry(flip_src, op))
        joined = " ".join(cross)
        for k, how in (("lds_read_B", "short by"), ("global_read_B", "long by")):
            if k not in joined:
                escapes.append("judging a staged-Q listing against an un-staged-Q model did not blame %s, which "
                               "the model must be %s exactly the Q operand tile: %s"
                               % (k, how, joined[:160]))

    # ---- 2. THE CONTROL: a benign injection must NOT change the verdict. A check that fails for any reason
    # adjudicates nothing, which is the same defect as one that can never fail.
    if red_after(inject(body, "v_add_f32 v0, v0, v0")):
        escapes.append("the census went red for an unrelated instruction, so its verdict cannot be used to "
                       "adjudicate anything")

    # ---- 3. THE MODEL SIDE: each expectation must move when the quantity it claims to come from moves.
    base = derive(g)
    # Each case is a list of edits to the kernel header and, where the fact is shared, to the op header. The warp
    # count IS a two-header fact now (`FATTN_WARPS` here, `FATTN_WARPS_PER_BLOCK` there, and 34 of
    # bench/performance-status.md is what happens when one half moves alone), so the cases below move both halves
    # the way a real change would — and one case keeps the asymmetry on purpose to check the census refuses it.
    K_FILL = "            fattn_stage_row(&s_kv[fattn_kv_addr(slot, 0)],"
    # (what, edits to engine header, edits to op header, terms that MUST move, terms that MUST NOT move)
    MUTATIONS = [
        ("the halves loop bound", [("for (int half = 0; half < 2; ++half)",
                                   "for (int half = 0; half < 1; ++half)")], [],
         ("lds_read_B", "lds_write_B", "global_write_B"), ()),
        ("the publish loop's block count", [("#define FATTN_PBLK 8", "#define FATTN_PBLK 6")], [],
         ("lds_read_B",), ()),
        ("the publish block's width", [("#define FATTN_TBLK 8", "#define FATTN_TBLK 4")], [],
         ("lds_read_B",), ()),
        ("the C-tile register count the stage write loops over",
         [("for (int l = 0; l < 8; ++l)", "for (int l = 0; l < 4; ++l)")], [],
         ("lds_write_B",), ()),
        ("the transposed V fill's per-dim loop",
         [("for (int t = 0; t < 8; ++t) {\n                s_kv[fattn_v_addr",
           "for (int t = 0; t < 4; ++t) {\n                s_kv[fattn_v_addr")], [],
         ("lds_write_B",), ()),
        # ONE Q STAGE FOR THE WHOLE BLOCK: the geometry error of `fattn_ship_results.md` 2, spelled against today's
        # source. It is the case that proves `lds_B` derives from the warp count rather than being 32768 wearing
        # a formula, and it is the error three warps make the kernel fatal about: each would stage its own head's
        # rows into one buffer and score itself against an interleaving of all three.
        ("one Q stage shared by the block's warps (the 2 geometry error)",
         [(Q_STAGE_DECL.replace("FATTN_WARPS * FATTN_MROWS * DH", "FATTN_MROWS * DH"), Q_STAGE_DECL)], [],
         ("lds_B",), ()),
        ("the whole Q un-staging (Bet 1: no stage, no fill, operand from global)",
         [(Q_STAGE_DECL, ""),
          (Q_FILL_LOOP, "    const hip_bfloat16* qrow = a.qrope + (size_t)ROW_H * (rb * FATTN_MROWS "
                        "+ fattn_ab_row(lane)) + (size_t)head0 * DH;"),
          (Q_OPERAND, Q_OPERAND_GLOBAL)], [],
         ("lds_B", "lds_read_B", "lds_write_B", "global_read_B"), ("global_write_B", "cross_lane", "v_wmma")),
        ("a third stage fill appearing", [(K_FILL, K_FILL + "\n                            a.kcache, lane);\n"
                                           + K_FILL)], [],
         ("lds_write_B", "global_read_B"), ()),
        ("the score leg's roll pragma", [("#pragma unroll 1\n            for (int t = 0; t < FATTN_KTILES; ++t)",
                                         "#pragma unroll 16\n            for (int t = 0; t < FATTN_KTILES; ++t)")],
         [], ("v_wmma", "lds_read_B"), ()),
        # THE WARP COUNT, both headers at once. `lds_B` must move (one fewer 8 KiB Q stage) and `lds_read_B` must
        # move (the fold has one fewer warp's span to read back). The MUST-NOT list is the other half of the
        # witness: three warps execute one kernel body, so the per-warp site counts and the per-row publish are
        # nothing's business but their own, and a model that moved them would be moving a number because the
        # listing moved it.
        ("the warp count per block (3 -> 2)",
         [("#define FATTN_WARPS 3", "#define FATTN_WARPS 2"),
          ("#define FATTN_THREADS 96", "#define FATTN_THREADS 64")],
         [("#define FATTN_WARPS_PER_BLOCK 3", "#define FATTN_WARPS_PER_BLOCK 2")],
         ("lds_B", "lds_read_B"), ("v_wmma", "cross_lane", "global_write_B", "s_barrier")),
        # ...and down to ONE warp, where the barriers go back to 0 because LLVM deletes a lone wavefront's
        # `__syncthreads()`. This is the case that retires the typed `1 + 3 + 3` the old census carried.
        ("the warp count per block (3 -> 1)",
         [("#define FATTN_WARPS 3", "#define FATTN_WARPS 1"),
          ("#define FATTN_THREADS 96", "#define FATTN_THREADS 32")],
         [("#define FATTN_WARPS_PER_BLOCK 3", "#define FATTN_WARPS_PER_BLOCK 1")],
         ("s_barrier", "lds_B"), ()),
        # The two headers DISAGREEING is not a geometry to price, it is the half-committed change of 34, and the
        # census refuses it in `geometry` itself. The case passes by the census raising; it would report an escape
        # if the census ever stopped comparing the two.
        ("the two headers disagreeing on the warp count", [],
         [("#define FATTN_WARPS_PER_BLOCK 3", "#define FATTN_WARPS_PER_BLOCK 2")], (), ()),
    ]
    for what, src_edits, op_edits, keys, stable in MUTATIONS:
        missing = [needle for needle, _ in src_edits if needle not in src] \
            + [needle for needle, _ in op_edits if needle not in op]
        if missing:
            escapes.append("selftest cannot perturb %s: %r is no longer in the headers. Fix the selftest; do "
                           "not delete the case" % (what, missing[0][:70]))
            continue
        text = src
        optext = op
        for needle, repl in src_edits:
            text = text.replace(needle, repl, 1)
        for needle, repl in op_edits:
            optext = optext.replace(needle, repl, 1)
        try:
            moved = derive(geometry(text, optext))
        except AssertionError:
            continue          # the census noticed the inconsistency itself, which is a live dependency
        if moved == base:
            escapes.append("the derived expectation did not move at all when %s changed, so it is a constant "
                           "wearing a formula" % what)
            continue
        for k in keys:
            if k not in moved:
                escapes.append("%s claims to derive %s, which the model no longer produces" % (what, k))
            elif moved[k] == base[k]:
                escapes.append("deriving %s ignored a change to %s" % (k, what))
        for k in stable:
            if k in moved and moved[k] != base[k]:
                escapes.append("deriving %s moved when only %s changed: the warp count moves the LDS a block "
                               "declares and the barriers and fold it pays, not a per-warp or per-row term"
                               % (k, what))
    try:
        moved = derive(geometry(src, op.replace("#define DH 256", "#define DH 128", 1)))
    except AssertionError:
        moved = dict(base, v_wmma=-1)   # the census noticed the inconsistency itself; that counts as moved
    if moved["v_wmma"] == base["v_wmma"]:
        escapes.append("deriving v_wmma ignored DH, so KTILES/VTILES were not read from the op header")

    # ---- 3b. THE Q OPERAND'S SOURCE, BOTH WAYS ROUND. The shipped form stages Q; the flip hands the score leg a
    # global row instead, leaving the stage and its fill alone. So `lds_read_B` must FALL (the operand stops being
    # an LDS read) and `global_read_B` must RISE (it becomes two uint4 global reads priced off the loader the call
    # site names). Directions rather than deltas on purpose: a delta would be a byte count typed in here, which is
    # the thing this selftest exists to refuse. If the classifier keyed on anything other than the operand's own
    # call site — the presence of a declaration, say — one of the two goes still.
    if flip_src is not None:
        mv = derive(geometry(flip_src, op))
        for k, up in (("lds_read_B", False), ("global_read_B", True)):
            if mv[k] == base[k]:
                escapes.append("reading the Q operand from global while a Q stage is still declared did not move "
                               "%s, so the classification does not come from the operand's own call site" % k)
            elif (mv[k] > base[k]) != up:
                escapes.append("un-staging Q moved %s the wrong way (%+d): an operand that stops coming from LDS "
                               "cannot also stop costing global reads" % (k, mv[k] - base[k]))
    return escapes


def exp_of(g):
    """`derive` under a name that reads well inside `compare`. Memoised because compare() is called per key."""
    return derive(g)


COMPARE_KEYS = ("v_wmma", "lds_read_B", "lds_write_B", "global_read_B", "global_write_B", "cross_lane",
                "s_barrier")

# WHAT EACH NUMBER IS PER. Printed beside the comparison, because a count with no denominator is its own bug
# class (AGENTS.md: "an aggregate verdict that does not name its denominator cannot fail"), and because the warp
# count moves exactly the terms whose denominator is the BLOCK -- which is the reviewer's test that this is a
# model and not a mirror of the listing.
DENOMS = {
    "lds_B": "bytes per BLOCK (what sets blocks/CU)",
    "v_wmma": "sites in one kernel body = per WARP per launch (the rolled score leg prices one trip)",
    "lds_read_B": "bytes per LANE, one site per source access",
    "lds_write_B": "bytes per LANE, one site per source access",
    "global_read_B": "bytes per LANE, one site per source access",
    "global_write_B": "bytes per LANE, one site per source access",
    "cross_lane": "sites in one kernel body = per WARP (the three warps run the same body)",
    "s_barrier": "sites per BLOCK (0 at one warp per block, where LLVM deletes them)",
    "s_wait_lgkmcnt0": "FLOOR on sites in one kernel body; LLVM's own conservatism may add to it",
}

# What each comparison is FOR, printed with the failure. A census line that only says "wanted 1088, got 1104"
# gets ``fixed'' by editing the number; a line that names the defect it catches gets fixed in the source.
WHY = {
    "v_wmma": "the tile count is a property of KTILES/VTILES and the roll/unroll pragmas, not of taste",
    "lds_read_B": "the bytes this form reads through LDS are the LDS-resident WMMA operand tiles (K always, Q "
                  "only where the source stages it) plus the publish of the transpose stage. More than that "
                  "means a staged object is being re-read from somewhere (the 20.5 traffic error); less means a "
                  "publish loop that stopped covering the stage it is "
                  "publishing, which is the defect that made every covered row of a wide wave NaN "
                  "(bench/performance-status.md 29's correction)",
    "lds_write_B": "the Q and K stage fills, the transposed V fill, the F32 transpose stage and the cross-warp "
                   "fold: four loops, each with a shape this census reads out of the source. More than the "
                   "source spells means either a staged object is being written from somewhere new, or LLVM "
                   "emitted one rolled fill loop's body more than once (a fill loop states no unroll factor, so "
                   "its site count is a codegen choice the census cannot predict until the source says so)",
    "global_read_B": "one uint4 per lane per stage fill, plus the Q operand's own wide loads when the score leg "
                     "builds it from global (Bet 1) rather than reading it from a stage (placement B). A new "
                     "one that is neither means a staged object started being read "
                     "from global, which is the 20.5 error in its original form; a repeat of a fill the source "
                     "spells once is LLVM emitting a rolled loop's body twice (see the census note on the Q "
                     "fill): same bytes demanded, more sites emitting them, and the census says so rather than "
                     "absorbing it",
    "global_write_B": "the publish loop's bytes into `part` plus the (m, l) header. The broken epilogue "
                      "wrote 16 of every row's 256 dims here and left the merge reading scratch, and no "
                      "bound-style assert could see it because the defect was in the SET of indices touched",
    "cross_lane": "10 warp min/max + 9 in the round + 1 deferred rowsum combine. One fewer is D-4 coming "
                  "back: half a denominator per row, which cosine cannot see at all. Per warp-execution: the "
                  "three warps of a Bet-2 block run the same body, so this count does not triple with them",
    "s_barrier": "derived from the source's own __syncthreads() call sites, weighted by the loops that "
                 "statically unroll them (see `barrier_sites`). At one warp per block LLVM deletes every one "
                 "(measured 0), which is why aliasing V over K was free at that geometry; above a warp they are "
                 "real instructions and the round pays them, so the wavefronts-per-CU gain has to be read "
                 "alongside that price",
}


def compare(g, note, obs, ack_spill=""):
    """Every derived-vs-listing comparison, as a pure function so `--selftest` can drive it on mutated input.

    Returns (lines_to_print, hard_failures, spill_findings). Exact equality on every class: the expectation is
    derived from the source's loops, so a mismatch means the model and the code disagree about what the
    source says, and the fix is to re-read the source — either side. The old form of this check was
    ``observed > derived + 2``, which could not see a publish loop that under-covered the stage, and the
    tolerance was itself a number copied from a listing.
    """
    prints, fails, spill = [], [], []
    if not g["launch_bounds"]:
        fails.append("no __launch_bounds__(FATTN_THREADS): without it LLVM budgets a thread 196,608/1024 = "
                     "192 VGPRs and spills the rest, so a spilling result would be about the declaration and "
                     "not the kernel")
    for k, limit in (("vgpr_spill_count", 0), ("private_segment_fixed_size", 0)):
        if note[k] != limit:
            spill.append("%s=%d, must be %d. That is the spill gate: scratch lands on the same dependent "
                         "chain bench/performance-status.md 28 identifies as the mechanism, so --attn-fattn "
                         "stays default 0 until this is 0" % (k, note[k], limit))
    lds_wanted = exp_of(g)["lds_B"]
    if note["group_segment_fixed_size"] != lds_wanted:
        fails.append("lds_B: group_segment_fixed_size=%d, but the body declares %d B of `__shared__` arrays. "
                     "Either a stage was added or removed in one place and not the other, or LLVM kept or "
                     "merged an object the source does not declare (it has deleted an unused LDS object before, "
                     "which is why the two are compared rather than trusted). Wavefronts per CU come from this "
                     "number, so a change here is a residency change and must be argued, not stumbled into"
                     % (note["group_segment_fixed_size"], lds_wanted))
    prints.append("derived-vs-listing: " + "  ".join("%s want=%d got=%d" % (k, exp_of(g)[k], obs[k])
                                                     for k in COMPARE_KEYS))
    if obs["_unclassified"]:
        fails.append("unclassified LDS/global access(es) %s: this census models an explicit list of access "
                     "mnemonics, and an access it does not name is traffic it has not priced. Add it to the "
                     "class table WITH its width and a sentence on which loop it comes from, or remove it"
                     % ", ".join(obs["_unclassified"]))
    for k in COMPARE_KEYS:
        want = exp_of(g)[k]
        if k == "s_wait_lgkmcnt0":
            continue
        if obs[k] != want:
            fails.append("%s sites %d, derived %d. %s" % (k, obs[k], want, WHY[k]))
    if obs["s_wait_lgkmcnt0"] < exp_of(g)["s_wait_lgkmcnt0"]:
        fails.append("s_wait_lgkmcnt0 sites %d, derived floor %d. The staged Q, the K stage and the transposed "
                     "V each need an LDS read-after-write fence inside the wavefront. Deleting barriers does "
                     "not delete the ordering requirement — if this count ever falls to 0 the transposed-V read "
                     "is unsynchronised and the form is wrong on the wavefront level, not just slow"
                     % (obs["s_wait_lgkmcnt0"], exp_of(g)["s_wait_lgkmcnt0"]))
    if spill and not (ack_spill or "").strip():
        fails.extend(spill)
    return prints, fails, spill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", default=os.path.join(ASM_DIR, "kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s"))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--ack-spill", default="", metavar="REASON",
                    help="D63: measure a spilling placement without clearing it. The spill findings are still "
                         "printed and the placement is still NOT cleared to become the default; this only lets "
                         "a diagnostic window be timed. Set by the operator, never by a worker.")
    ap.add_argument("--selftest", action="store_true",
                    help="inject defects into the listing and into the source, and require the census to "
                         "catch every one of them. A census whose expectation could have been typed in looks "
                         "exactly like a correct one until something perturbs it.")
    a = ap.parse_args()
    if not a.no_build:
        # Every one of these exits is the empty-parse class AGENTS.md records three times: a gate whose input can
        # fail to arrive must be SHOWN to have arrived, or it reports a verdict over nothing.
        if not os.path.isfile(BUILD_SH):
            sys.exit("%s is missing, so this census cannot compile the kernel it adjudicates. Restore it (it is "
                     "tracked); do not instead point the census at a listing left over from another tree, which "
                     "is how a stale binary reads as a clean A/B" % BUILD_SH)
        proc = subprocess.run(["bash", BUILD_SH, ASM_DIR], stdout=subprocess.DEVNULL)
        if proc.returncode != 0:
            sys.exit("%s exited %d: the compiler said no, so there is no listing to adjudicate and no numerics "
                     "verdict to report. Its own output is above this line." % (BUILD_SH, proc.returncode))
        # `--save-temps` writes BOTH the device and the host listing into the CWD; moving whichever
        # `os.listdir` returns last is how this census once read the x86_64 listing and found no note block.
        for f in os.listdir("."):
            if f.startswith("kernels_attn-hip-amdgcn") and f.endswith(".s"):
                shutil.move(f, a.asm)
        if not os.path.isfile(a.asm):
            sys.exit("%s compiled but left no device listing at %s: the census would be reading a listing from a "
                     "previous build. Check the `--save-temps` output, do not reuse one" % (BUILD_SH, a.asm))
    if not os.path.isfile(a.asm):
        sys.exit("no listing at %s, and --no-build was passed: nothing to adjudicate" % a.asm)
    text = open(a.asm).read()
    g = geometry()
    notes, obs = note_blocks(text), counts(text)
    if "attn_fattn_wave" not in notes:
        sys.exit("no note block for attn_fattn_wave in " + a.asm)
    if not obs:
        # The empty-parse trap: a census over a listing it parsed nothing out of has no failures to report and
        # therefore prints PASS. A check whose input list can be empty cannot fail.
        sys.exit("no kernel bodies found in " + a.asm)
    n = notes["attn_fattn_wave"]
    print("geometry (%s): WARPS=%d THREADS=%d MROWS=%d MLANES=%d KTILES=%d VTILES=%d TBLK=%d PBLK=%d TROW=%d "
          "halves=%d publish=%s blocks of %d by %d publishing lanes | stage writes/tile=%d V-narrow=%d "
          "stage fills=%d fold=%dx%d reads +%d writes score_leg=%s launch_bounds=%s"
          % (HEADER, g["FATTN_WARPS"], g["FATTN_THREADS"], g["FATTN_MROWS"], g["FATTN_MLANES"], g["KTILES"],
             g["VTILES"], g["FATTN_TBLK"], g["FATTN_PBLK"], g["FATTN_TROW"], g["transpose_halves"],
             g["publish_blocks"], g["publish_block_floats"], g["publish_lanes"],
             g["stage_write_per_tile"],
             g["v_narrow_stores"], g["global_stage_fills"],
             g["fold_reads_per_trip"], g["fold_trips"], g["fold_write_slots"],
             "rolled" if g["score_rolled"] else "unrolled", "yes" if g["launch_bounds"] else "NO"))
    # THE CLASSIFIER'S ANSWER, printed rather than assumed. Two spellings of the Q operand produce the same
    # v_wmma count and the same publish, so without this line a reader cannot tell which source the model
    # believes it read, and a wrong belief looks like a two-sided traffic error in the kernel.
    print("q operand: %s (%s)" % ("staged in `%s`" % g["q_stage"] if not g["q_from_global"] else "from global",
                                  g["q_producer"]))
    print("derived from those loops: " + "  ".join("%s=%d" % (k, exp_of(g)[k])
                                                   for k in COMPARE_KEYS + ("s_wait_lgkmcnt0",)))
    print("denominators: " + " | ".join("%s: %s" % (k, DENOMS[k]) for k in COMPARE_KEYS))
    print("note block: " + "  ".join("%s=%s" % (k, n[k]) for k in NOTE_KEYS))
    key = [k for k in obs if k.endswith("attn_fattn_waveE8CoreArgs")]
    if not key:
        prints, fails, spill = [], ["no attn_fattn_wave body found in the listing to count sites in"], []
    else:
        prints, fails, spill = compare(g, n, obs[key[0]], getattr(a, "ack_spill", ""))
    for line in prints:
        print(line)
    print("touches per warp-round = %d (MROWS x MLANES), the same denominator fattn_census.py used"
          % (g["FATTN_MROWS"] * g["FATTN_MLANES"]))
    blocks = 65536 // max(n["group_segment_fixed_size"], 1)
    print("LDS-bound blocks per CU = %d (64 KiB / %d B declared: %s), wavefronts/CU = %d at %d warp(s) per "
          "block (driver-confirmed rule: the census never queries the driver, it prints the arithmetic the "
          "driver agreed with)"
          % (blocks, g["lds_declared_B"], g["lds_declared"], blocks * g["FATTN_WARPS"], g["FATTN_WARPS"]))
    if a.selftest:
        st = selftest(text)
        print("selftest: %d injected defect(s) escaped" % len(st))
        for f in st:
            print("  - " + f)
        fails.extend("selftest: " + f for f in st)
    # D63: the spill finding is a REPORTED cost, not a precondition on measurement. Everything else in
    # `fails` (launch bounds, LDS-vs-source, the derived traffic and structure) stays a hard failure, because
    # those are facts about the code, not bets about what the machine charges for them. The spill-vs-ack
    # decision itself lives in `compare`, so `--selftest` judges it on the same terms as every other check.
    if fails:
        print("\nFAIL — the shipped placement is NOT cleared to become the default")
        for f in fails:
            print("  - " + f)
        return 1
    if spill and (a.ack_spill or "").strip():
        print("\nSPILL ACKNOWLEDGED, FOR MEASUREMENT ONLY — operator-supplied reason: " + a.ack_spill)
        for f in spill:
            print("  - " + f)
        print("  -> this placement may be TIMED in this window. It is still NOT cleared to become the")
        print("     default: only a measured >=1.5x cut of the attn family at 4096 lanes flips that (D63).")
        print("     Every timing row taken after this line must carry the spill and the wavefronts/CU printed above.")
        return 0
    print("\nPASS: spill-free, LDS matches the source, instruction skeleton matches its own geometry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
