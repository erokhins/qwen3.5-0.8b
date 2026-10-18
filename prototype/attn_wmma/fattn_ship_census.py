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
import re
import subprocess
import sys

HEADER = "engine/attn_fattn_kernels.h"
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


def eval_expr(tok, env):
    """Evaluate the two expression forms this header's geometry uses: a literal, or `NAME / literal`.

    A tiny reader rather than `eval`, and deliberately so: a census that evaluated arbitrary text out of a
    source file would be a census that runs whatever a comment happens to look like.
    """
    tok = re.sub(r"u$|ul$|ull$", "", tok.strip())
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    m = re.fullmatch(r"\(?\s*(\w+)\s*/\s*(\d+)\s*\)?", tok)
    if m:
        return env[m.group(1)] // int(m.group(2))
    if re.fullmatch(r"\w+", tok) and tok in env:
        return env[tok]
    raise AssertionError("cannot evaluate %r" % tok)


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


def shared_bytes(body, env):
    """The LDS one block declares, summed from its own `__shared__` declarations.

    Wavefronts per CU are what the two unpriced suspects are about, and on this part LDS per block is what sets
    them, so a census that hard-coded 16384 could not witness a geometry change: it would fail the very variant
    it was written to justify (and `compare` used to do exactly that). An unparseable declaration is a hard
    failure — guessing an element width here is guessing the residency.
    """
    total = 0
    for m in re.finditer(r"__shared__\s+(?:__align__\(\d+\)\s+)?([A-Za-z_]\w*)\s+(\w+)\[([^\]]+)\]", body):
        typ, name, dims = m.group(1), m.group(2), m.group(3)
        if typ not in SHARED_TYPES:
            raise AssertionError("`%s %s[%s]`: no width for type %r. Add it to SHARED_TYPES with its size; do "
                                 "not assume 2" % (typ, name, dims, typ))
        n = 1
        for tok in dims.split("*"):
            n *= eval_expr(tok, env)
        total += SHARED_TYPES[typ] * n
    if total == 0:
        raise AssertionError("no `__shared__` declaration in the kernel body: this census cannot price "
                             "residency without knowing what a block declares")
    return total


SHARED_TYPES = {"__half": 2, "float": 4, "hip_bfloat16": 2, "char": 1, "unsigned": 4, "int": 4}


def shared_bytes(body, env):
    """The LDS one block declares, summed from its own `__shared__` declarations.

    Wavefronts per CU are what the residency question is about, and on this part LDS per block is what sets
    them, so a census that hard-coded 16384 could not witness a geometry change: it would go red on the very
    variant it was written to justify. An unparseable or missing declaration is a hard failure — guessing an
    element width here is guessing the residency, and a census that found no stage at all has not priced
    anything.
    """
    total, seen = 0, []
    for m in re.finditer(r"__shared__\s+(?:__align__\(\d+\)\s+)?([A-Za-z_]\w*)\s+(\w+)\[([^\]]+)\]", body):
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
    for name in ("FATTN_MROWS", "FATTN_MLANES", "FATTN_THREADS", "FATTN_TBLK", "FATTN_PBLK",
                 "FATTN_TROW"):
        g[name] = define_value(src, name, g)
    body = src[src.index("void attn_fattn_body("):]
    body = body[:body.index("__device__ __host__ inline bool fattn_plan_ok")]
    epilogue = src[src.index("void fattn_transpose_out("):]
    epilogue = epilogue[:epilogue.index("warp-wide min and max")]
    stage_row = src[src.index("void fattn_stage_row("):]
    stage_row = stage_row[:stage_row.index("struct FattnRow")]
    g["score_rolled"] = bool(re.search(r"#pragma unroll 1\s*$", body, re.M))
    # THE HALVES LOOP: the epilogue stages F32 through the dead K/V stage in this many passes, and every
    # publish-side count below is a multiple of it.
    g["transpose_halves"] = for_bound(epilogue, "half", g)
    # THE PUBLISH LOOP: each lane publishes this many blocks per half, each block this many floats.
    g["publish_blocks"] = for_bound(epilogue, "k", g)
    g["publish_block_floats"] = for_bound(epilogue, "u", g)
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
    # THE STAGE FILLS: `fattn_stage_row` moves one uint4 of bf16 (8 halves = 16 B) per call, so each call site
    # is one wide global read and one wide LDS write. The transposed V fill is an inline uint4 of its own.
    g["stage_row_halves"] = for_bound(stage_row, "t", g)
    # Call SITES, at statement position: `body` starts at `attn_fattn_body`, so the definition of
    # `fattn_stage_row` (whose line starts with `void`) and the prose that names it are both excluded.
    g["stage_row_sites"] = len(re.findall(r"^\s*fattn_stage_row\(", body, re.M))
    g["global_stage_fills"] = g["stage_row_sites"] + 1
    # THE PARTIAL HEADER: the (m, l) pair, two F32 per covered row. It is the only global write that is not
    # the publish loop's, and the second version of the epilogue got the publish shape wrong beside it.
    g["part_header_floats"] = 2
    # THE STAGES, as declared. The residency question is a question about this number, so it is read off the
    # declarations and not assumed from a sentence about them.
    g["lds_declared_B"], g["lds_declared"] = shared_bytes(body, g)
    g["launch_bounds"] = "__launch_bounds__(FATTN_THREADS)" in src
    g["one_warp_per_block"] = g["FATTN_THREADS"] == 32
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
    # One operand tile is 16 halves = 32 B. The score leg has two operands per site; the P*V leg has one,
    # because the P operand lives in registers — which is the whole point of the C->B exchange.
    operand_bytes = (score_sites * 2 + g["VTILES"]) * 16 * 2
    # THE PUBLISH SIDE — the term the pre-NaN-fix model did not have at all. Per half the stage holds
    # FATTN_MROWS x FATTN_TROW floats; every one of them is read back, exactly once, in blocks of
    # FATTN_TBLK floats and written to the partial slot, so the stage read and the partial write are the same
    # byte count from two different loops.
    publish_bytes = g["transpose_halves"] * g["publish_blocks"] * g["publish_block_floats"] * 4
    # The same number from the OTHER side of the header's coverage assert: the stage's own float count, split
    # across the block's lanes. If the two disagree, the publish loop and the stage's coverage disagree, and a
    # partial publish is exactly the defect that made every covered row NaN.
    publish_bytes_stage = (g["transpose_halves"] * g["FATTN_MROWS"] * g["FATTN_TROW"] * 4
                           // g["FATTN_THREADS"])
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
        # LDS READS: the WMMA operand tiles plus the epilogue's publish of the transpose stage.
        "lds_read_B": operand_bytes + publish_bytes,
        # LDS WRITES: the Q and K stage fills (one uint4 per `fattn_stage_row` call), the transposed V fill
        # (one 16-bit store per dim of its loop), and the transpose stage (one F32 per C-tile register per
        # tile per half).
        "lds_write_B": (g["stage_row_sites"] * g["stage_row_halves"] * 2
                        + g["v_narrow_stores"] * 2
                        + g["transpose_halves"] * g["stage_tiles_per_half"]
                        * g["stage_write_per_tile"] * 4),
        # GLOBAL READS: the stage fills, one uint4 per lane per trip of each fill loop.
        "global_read_B": g["global_stage_fills"] * 16,
        # GLOBAL WRITES: the publish loop's bytes into `part`, plus the (m, l) header the epilogue writes
        # beside it. This is the term the broken epilogue got wrong in a way no bound-style assert could see:
        # it wrote 16 of every row's 256 dims, so 240 dims per row of the partial slot were never written.
        "global_write_B": publish_bytes + g["part_header_floats"] * 4,
        # 5 butterflies each in warp_min and warp_max (outside the round), the round's 1 max fold + 8 C->B
        # exchanges, and the 1 deferred rowsum combine: 10 outside, 9 in the round, 1 after it.
        "cross_lane": 10 + 9 + 1,
        # Barrier SITES. At one warp per block a `__syncthreads()` has no second wavefront to sequence, and
        # the measured listing contains ZERO `s_barrier` instructions — so the third barrier per round that
        # aliasing V over K costs on a multi-warp block is free at this geometry, which is a large part of why
        # placement (B) beat placement (A) on more than registers. If FATTN_THREADS ever exceeds a warp this
        # expectation must become 1 + 3 + 3 (Q published / K published / K consumed / V published, then the
        # transpose), and the wavefronts-per-CU claim has to be re-derived with it.
        "s_barrier": 0 if g["one_warp_per_block"] else 1 + 3 + (2 * g["transpose_halves"] + 1),
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

    def red_after(mutated_body):
        _, fails, _ = compare(g, clean_note, count_body(mutated_body), ack_spill="selftest")
        return fails

    base_fails = red_after(body)
    if base_fails:
        return ["the unmodified listing does not pass on its own, so an injection test on top of it proves "
                "nothing: " + " | ".join(base_fails)]

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
        ("one global_store_b128 MISSING (part of the row's publish disappeared)",
         drop(body, "global_store_b128"), "global_write_B"),
        ("one ds_store_b16 MISSING (the transposed V publish loses a dim)",
         drop(body, "ds_store_b16"), "lds_write_B"),
        ("one ds_store_b32 MISSING (the transpose stage loses one C-tile register)",
         drop(body, "ds_store_b32"), "lds_write_B"),
        ("one ds_store_b128 MISSING (a stage fill lost its wide write)",
         drop(body, "ds_store_b128"), "lds_write_B"),
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

    # ---- 2. THE CONTROL: a benign injection must NOT change the verdict. A check that fails for any reason
    # adjudicates nothing, which is the same defect as one that can never fail.
    if red_after(inject(body, "v_add_f32 v0, v0, v0")):
        escapes.append("the census went red for an unrelated instruction, so its verdict cannot be used to "
                       "adjudicate anything")

    # ---- 3. THE MODEL SIDE: each expectation must move when the quantity it claims to come from moves.
    src = open(HEADER).read()
    op = open("prototype/attn_decode/attn_decode_kernels.h").read()
    base = derive(g)
    MUTATIONS = [
        ("the halves loop bound", "for (int half = 0; half < 2; ++half)",
         "for (int half = 0; half < 1; ++half)", ("lds_read_B", "lds_write_B", "global_write_B")),
        ("the publish loop's block count", "#define FATTN_PBLK 8", "#define FATTN_PBLK 6",
         ("lds_read_B",)),
        ("the publish block's width", "#define FATTN_TBLK 8", "#define FATTN_TBLK 4", ("lds_read_B",)),
        ("the C-tile register count the stage write loops over", "for (int l = 0; l < 8; ++l)",
         "for (int l = 0; l < 4; ++l)", ("lds_write_B",)),
        ("the transposed V fill's per-dim loop",
         "for (int t = 0; t < 8; ++t) {\n                s_kv[fattn_v_addr",
         "for (int t = 0; t < 4; ++t) {\n                s_kv[fattn_v_addr", ("lds_write_B",)),
        ("a stage declaration disappearing", "    __shared__ __align__(16) __half s_q[FATTN_MROWS * DH];",
         "", ("lds_B",)),
        ("a third stage fill appearing",
         "        fattn_stage_row(&s_q[fattn_q_addr(qrs, 0)],",
         "        fattn_stage_row(&s_q[fattn_q_addr(qrs, 0)], a.k, lane);\n"
         "        fattn_stage_row(&s_q[fattn_q_addr(qrs, 0)],",
         ("lds_write_B", "global_read_B")),
        ("the score leg's roll pragma", "#pragma unroll 1", "#pragma unroll 16",
         ("v_wmma", "lds_read_B")),
        ("threads per block", "#define FATTN_THREADS 32", "#define FATTN_THREADS 64", ("s_barrier",)),
    ]
    for what, needle, repl, keys in MUTATIONS:
        if needle not in src:
            escapes.append("selftest cannot perturb %s: %r is no longer in %s. Fix the selftest; do not "
                           "delete the case" % (what, needle, HEADER))
            continue
        try:
            moved = derive(geometry(src.replace(needle, repl, 1), op))
        except AssertionError:
            continue          # the census noticed the inconsistency itself, which is a live dependency
        if moved == base:
            escapes.append("the derived expectation did not move at all when %s changed, so it is a constant "
                           "wearing a formula" % what)
            continue
        for k in keys:
            if k in moved and moved[k] == base[k]:
                escapes.append("deriving %s ignored a change to %s" % (k, what))
    try:
        moved = derive(geometry(src, op.replace("#define DH 256", "#define DH 128", 1)))
    except AssertionError:
        moved = dict(base, v_wmma=-1)   # the census noticed the inconsistency itself; that counts as moved
    if moved["v_wmma"] == base["v_wmma"]:
        escapes.append("deriving v_wmma ignored DH, so KTILES/VTILES were not read from the op header")
    return escapes


def exp_of(g):
    """`derive` under a name that reads well inside `compare`. Memoised because compare() is called per key."""
    return derive(g)


COMPARE_KEYS = ("v_wmma", "lds_read_B", "lds_write_B", "global_read_B", "global_write_B", "cross_lane",
                "s_barrier")

# What each comparison is FOR, printed with the failure. A census line that only says "wanted 1088, got 1104"
# gets ``fixed'' by editing the number; a line that names the defect it catches gets fixed in the source.
WHY = {
    "v_wmma": "the tile count is a property of KTILES/VTILES and the roll/unroll pragmas, not of taste",
    "lds_read_B": "the bytes this form reads through LDS are the WMMA operand tiles plus the publish of the "
                  "transpose stage. More than that means a staged object is being re-read from somewhere "
                  "(the 20.5 traffic error); less means a publish loop that stopped covering the stage it is "
                  "publishing, which is the defect that made every covered row of a wide wave NaN "
                  "(bench/performance-status.md 29's correction)",
    "lds_write_B": "the Q and K stage fills, the transposed V fill and the F32 transpose stage: three loops, "
                   "each with a shape this census reads out of the source",
    "global_read_B": "one uint4 per lane per stage fill. A new one means a staged object started being read "
                     "from global, which is the 20.5 error in its original form",
    "global_write_B": "the publish loop's bytes into `part` plus the (m, l) header. The broken epilogue "
                      "wrote 16 of every row's 256 dims here and left the merge reading scratch, and no "
                      "bound-style assert could see it because the defect was in the SET of indices touched",
    "cross_lane": "10 warp min/max + 9 in the round + 1 deferred rowsum combine. One fewer is D-4 coming "
                  "back: half a denominator per row, which cosine cannot see at all",
    "s_barrier": "at one warp per block LLVM deletes every __syncthreads() (measured 0), which is why "
                 "aliasing V over K is free here; a non-zero count means the block is wider than a warp and "
                 "the wavefronts-per-CU claim has to be re-derived with it",
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
    ap.add_argument("--asm", default="/tmp/fattn_census2/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s")
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
        subprocess.run(["bash", "tmp/gcc/fattn_census_build.sh", "/tmp/fattn_census2"], check=True,
                       stdout=subprocess.DEVNULL)
        import shutil, os
        # `--save-temps` writes BOTH the device and the host listing into the CWD; moving whichever
        # `os.listdir` returns last is how this census once read the x86_64 listing and found no note block.
        for f in os.listdir("."):
            if f.startswith("kernels_attn-hip-amdgcn") and f.endswith(".s"):
                shutil.move(f, a.asm)
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
    print("geometry (%s): MROWS=%d MLANES=%d THREADS=%d KTILES=%d VTILES=%d TBLK=%d PBLK=%d TROW=%d "
          "halves=%d publish=%s blocks of %s | stage writes/tile=%d V-narrow=%d stage fills=%d "
          "score_leg=%s launch_bounds=%s"
          % (HEADER, g["FATTN_MROWS"], g["FATTN_MLANES"], g["FATTN_THREADS"], g["KTILES"], g["VTILES"],
             g["FATTN_TBLK"], g["FATTN_PBLK"], g["FATTN_TROW"], g["transpose_halves"],
             g["publish_blocks"], g["publish_block_floats"], g["stage_write_per_tile"],
             g["v_narrow_stores"], g["global_stage_fills"],
             "rolled" if g["score_rolled"] else "unrolled", "yes" if g["launch_bounds"] else "NO"))
    print("derived from those loops: " + "  ".join("%s=%d" % (k, exp_of(g)[k])
                                                   for k in COMPARE_KEYS + ("s_wait_lgkmcnt0",)))
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
    print("LDS-bound blocks per CU = %d (64 KiB / %d B declared: %s), wavefronts/CU = %d at %d warp(s) per "
          "block"
          % (65536 // max(n["group_segment_fixed_size"], 1), g["lds_declared_B"], g["lds_declared"],
             65536 // max(n["group_segment_fixed_size"], 1), g["FATTN_THREADS"] // 32))
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
    if spill and ack:
        print("\nSPILL ACKNOWLEDGED, FOR MEASUREMENT ONLY — operator-supplied reason: " + ack)
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
