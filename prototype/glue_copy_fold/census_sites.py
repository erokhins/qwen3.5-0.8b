#!/usr/bin/env python3
"""census_sites.py — what the 68 `glue` + `copy` launches actually are, derived from engine source.

Route S1's follow-on folds those launches into the kernels that consume their results. Before any of
it can be priced the population has to be NAMED rather than inherited: `prototype/launch_price/census.py`
bills "glue 49 / copy 19" and `engine/generation-limits.md` §1 repeats it, but neither says which 49
and which 19, which buffer each writes, which kernel reads it, or — the question this route turns on —
WHICH OF THEM CAN FOLD AT ALL. Foldability is a property of the CONSUMERS, so the consumers have to be
enumerated before the prize can be sized.

Four rules from AGENTS.md shape this file:

  * "a census must name the files it could read" — every row is read out of `engine/decode_step.cpp`
    (the call texts, their guards and their multiplicity) and `engine/state.h` (the widths), and the
    two launch totals are derived a SECOND time from `planned_launch_counts()`'s own expressions in
    that same file, so 49 and 19 are two independent derivations and not one remembered number.
  * "a parser that can match nothing reports success over an empty list" — every derivation refuses on
    an empty or short match, and `--selftest` proves each one refuses by feeding it a deliberately
    broken copy of its own source.
  * "a traffic claim is a property of the loops the source contains" — the per-site BYTES are compared
    against `tools/step_traffic_model.py`'s rows BY NOTE, not by family total, so a disagreement is
    attributed to one named row rather than dissolved in a family sum. The only tolerance is the
    model's own print precision (`%.3f` MB, so ±500 B), stated as such.
  * the `norms` row that `prototype/launch_price/census.py` folds into the `glue` STAGE — which is
    where §1's 0.407 MB read comes from — re-bills the 49 weight rows that `glue`'s own row already
    counts. That double count is printed, not carried.

Sites are keyed by the CALL TEXT each one is written with, not by a line number: `cast_fold` learned
that on the fold which added comment lines above its own sites.

Exit 0 prints the table, the classifications and `SITES ok`; anything else exits 1 with the reason.

    .venv/bin/python prototype/glue_copy_fold/census_sites.py [--selftest]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STEP_CPP = ROOT / "engine" / "decode_step.cpp"
STATE_H = ROOT / "engine" / "state.h"
TRAFFIC = ROOT / "tools" / "step_traffic_model.py"
MLP_KERNELS = ROOT / "prototype" / "mlp_gemv" / "mlp_gemv_kernels.h"
LMHEAD_KERNELS = ROOT / "prototype" / "lm_head_gemv" / "lm_head_gemv_kernels.h"
# The grouped-launch geometry lives in the launcher, not the kernel header.
GROUP_KERNELS = ROOT / "engine" / "kernels_proj.cpp"

# The per-decode-step sites (a body of ONE row) of the two families. `pattern` is the call text as the
# source spells it and `pattern_total` how often that text must appear; `per` is how many times one
# decode step REACHES it:
#   once    outside the layer loop, so once per body        gdn   under `if (!attn_layer)`, kNumGdnLayers
#   layers  in the layer loop, under no mixer guard, kNumLayers
# `nrows_read` / `nrows_write` are the operand ROWS the launch moves and `row`/`dtype` size them, in
# engine/state.h's names. The norm-only form passes delta == nullptr and s_out == nullptr, so it moves
# two rows in and one out where the fused form moves three and two.
SITES = [
    dict(name="glue_normonly_L0", family="glue",
         pattern=r"glue_add_norm_rows\(s, nullptr,", pattern_total=1, per="once",
         reads="s (== sc.embed), w(layer 0 input_layernorm)", writes="sc.norm",
         nrows_read=2, nrows_write=1, row="kHidden", dtype="BF16",
         consumer="layer 0's INPUT projections, and nothing else: s_out is nullptr, so this launch has"
                  " exactly one result (ONE grouped launch on the shipped --proj-plan packed, 4 under"
                  " --proj-plan split)",
         klass="not-foldable",
         why="the one glue launch whose norm has a SINGLE consumer, so it is the cleanest test of"
             " 'ride the consumer' — and it fails on the same arithmetic as the other 48: the consumer"
             " is a GEMV whose blocks are one output row wide, so riding it recomputes the whole-row"
             " norm once per consumer block. Priced in the FOLD-INTO-CONSUMER PRICE section below."),
    dict(name="glue_addnorm_postattn", family="glue",
         pattern=r"add_norm\(sc\.delta, layer_dev\(weights, layer, leaf::kPostAttnNorm\)\);",
         pattern_total=1, per="layers",
         reads="s (residual), sc.delta (mixer output), w(post_attention_layernorm)",
         writes="s_out (the new residual), sc.norm",
         nrows_read=3, nrows_write=2, row="kHidden", dtype="BF16",
         consumer="s_out -> this layer's SECOND add_norm; sc.norm -> mlp_gateup (one launch)",
         klass="not-foldable",
         why="TWO results, two consumers, and only one of them is per-element. The add could ride the"
             " mixer's output-projection GEMV (block i owns delta[i] and could write s_out[i]); the"
             " norm needs the whole row of the ADDED residual, which no launch between here and"
             " mlp_gateup has. Moving only the add leaves the launch standing — it still owes n_out —"
             " so it saves zero submissions; moving both buys the consumer-side norm priced below."),
    dict(name="glue_addnorm_nextinput", family="glue",
         pattern=r"add_norm\(sc\.delta, next_norm\);", pattern_total=1, per="layers",
         reads="s (residual), sc.delta (MLP output), w(next layer's input_layernorm, or lm.norm)",
         writes="s_out (the new residual), sc.norm",
         nrows_read=3, nrows_write=2, row="kHidden", dtype="BF16",
         consumer="s_out -> the NEXT layer's first add_norm (nothing, after layer 23); sc.norm -> that"
                  " layer's input projection group (one launch shipped), or lm_head after layer 23",
         klass="not-foldable",
         why="same shape as the post-attn site. The layer-23 instance is the only one whose s_out is"
             " dead and whose norm has a single consumer — lm_head, whose one-row form is the most"
             " blocks of any launch in the step: the most expensive consumer of all to park a norm in."),
    dict(name="copy_embed_row", family="copy",
         pattern=r"hipMemcpyAsync\(sc\.embed, table \+ \(size_t\)tokens\[0\] \* kHidden,",
         pattern_total=1, per="once",
         reads="one row of the tied embedding table", writes="sc.embed",
         nrows_read=1, nrows_write=1, row="kHidden", dtype="BF16",
         consumer="`s` of glue_normonly_L0 — the only reader of sc.embed in engine/, and sc.embed is"
                  " never a write target (decode_step.cpp's ping-pong note)",
         klass="foldable-now -> FOLDED in engine/decode_step.cpp",
         why="pure data movement, one reader, no arithmetic at all: point that reader at the table row"
             " the copy read and the copy has nothing left to do. Same values at a different address,"
             " so it owes a byte-compare and not a band."),
    dict(name="copy_conv_rollback", family="copy",
         pattern=r"hipMemcpyAsync\(st\.conv_window_layer\(layer\), sc\.conv_new,", pattern_total=1,
         per="gdn", reads="sc.conv_new, [kConvWindow][kConvChannels] F32",
         writes="st.conv_window_layer(layer)",
         nrows_read=3, nrows_write=3, row="kConvChannels", dtype="F32",
         consumer="the NEXT token's conv_kernel, through st.conv_window_layer(layer)",
         klass="needs-kernel-signature-change -> FOLDED in engine/gdn_conv_roll_kernels.h",
         why="the roll-back exists only so conv_kernel's `win` and `new_win` can both stay"
             " __restrict__. conv_kernel's own comment says the end-condition write 'is safe against"
             " its own reads whatever the relationship between win and new_win is', because a lane owns"
             " the same CONV_VEC columns on both sides — so handing the conv its state window as BOTH"
             " deletes the copy. What it needs is an entry point that does not ASSERT non-aliasing"
             " between the two, which is a new kernel rather than a call-site edit."),
]

# The copy-shaped submissions a WIDE body makes and a decode step never reaches. Named, so the census
# can say what it excluded instead of silently producing 19 out of a bigger set.
WIDE_ONLY = [
    dict(name="copy_ids_upload", pattern=r"hipMemcpyAsync\(ids_dev, ids_pin,",
         note="the wave's token-id H2D (D30 4.2′): the `n != 1` branch only"),
    dict(name="embed_gather", pattern=r"kernels::embed_gather_rows\(",
         note="the wave's single gather over the table (D30 4.2′), ditto"),
]

BYTES_OF = {"BF16": 2, "F32": 4}
POPULATION = 68          # [Q1] §1 / launch_price census: glue 49 + copy 19
PER_LAUNCH_US = 3.10     # [Q2] engine/generation-limits.md §4: measured by deleting 97 launches
READ_PLATEAU_GBS = 237.0  # [Q3] devices/strix-halo-memory-bandwidth.md read probe
MB_PRINT = 500           # step_traffic_model prints `%.3f` MB: ±500 B is its own precision


class CensusError(Exception):
    """A derivation that could not read what it expected. `die()` turns it into exit 1."""


def die(msg: str):
    raise CensusError(msg)


def shapes(state_txt: str | None = None) -> dict[str, int]:
    """The widths engine/state.h defines, resolved rather than typed.

    The same two-pass resolver `cast_fold/census_sites.py` uses: `kConvWindow` is `kConvKernel - 1`
    and `kQkvRows` is a product of constants declared later in the file, so a name-by-name pass
    cannot see its own dependencies, and typing 3 or 6144 to get past that is exactly the stale
    number this function exists to avoid.
    """
    txt = state_txt if state_txt is not None else STATE_H.read_text()
    exprs: dict[str, str] = {}
    for name, expr in re.findall(r"constexpr int\s+(\w+)\s*=\s*([^;]+);", txt):
        exprs[name] = expr.split("//")[0].strip()
    out: dict[str, int] = {}
    pending = list(exprs)
    while pending:
        progressed = False
        for name in list(pending):
            try:
                out[name] = int(exprs[name])
            except ValueError:
                try:
                    out[name] = int(eval(exprs[name], {"__builtins__": {}},  # noqa: S307 our source
                                         {k2: v for k2, v in out.items() if k2 != name}))
                except Exception:  # noqa: BLE001  a dependency is unresolved yet; try next round
                    continue
            progressed = True
            pending.remove(name)
        if not progressed:
            die(f"circular or unresolvable state.h constants: {sorted(pending)}")
    wanted = ("kHidden", "kHeadDim", "kIntermediate", "kQkvRows", "kValueDim", "kQwidth",
              "kQprojRows", "kKvProjRows", "kNumValueHeads", "kConvChannels", "kConvWindow",
              "kNumLayers", "kNumGdnLayers", "kNumAttnLayers", "kVocab")
    missing = [n for n in wanted if n not in out]
    if missing:
        die(f"engine/state.h does not define {missing}")
    # The layer-routing rule, read rather than remembered: `is_full_attention(l) == (l % M == R)`.
    m = re.search(r"is_full_attention\(int layer\)\s*\{\s*return layer % (\d+) == (\d+);", txt)
    if not m:
        die("cannot read the layer routing out of `is_full_attention()` in engine/state.h")
    out["_attn_mod"], out["_attn_rem"] = int(m.group(1)), int(m.group(2))
    return out


def is_attn_layer(layer: int, k: dict[str, int]) -> bool:
    return layer % k["_attn_mod"] == k["_attn_rem"]


def planned_exprs(step_txt: str) -> dict[str, str]:
    """`planned_launch_counts()`'s own arithmetic for the two rows, as text."""
    out: dict[str, str] = {}
    for field in ("glue", "other"):
        m = re.search(rf"c\.{field} = ([^;]+);", step_txt)
        if not m:
            die(f"cannot read `c.{field} = ...;` out of planned_launch_counts(); the launch bill has"
                " no source to derive from")
        out[field] = m.group(1).strip()
    return out


def _split_ternary(expr: str) -> tuple[int, int]:
    """Positions of the top-level `?` and `:` of a C++ conditional, or (-1, -1)."""
    depth, q, c = 0, -1, -1
    for i, ch in enumerate(expr):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "?" and depth == 0 and q < 0:
            q = i
        elif ch == ":" and depth == 0 and q >= 0 and c < 0:
            c = i
    if (q >= 0) != (c >= 0):
        die(f"`{expr}` has a `?` without a `:` or the reverse; this census will not guess at the bill")
    return q, c


def py_expr(expr: str) -> str:
    """Translate one C++ arithmetic expression out of `planned_launch_counts()` into Python.

    The fold rows are conditionals (`copy_fold ? 0 : 1 + kNumGdnLayers`), so a plain `eval` reads the
    bill as a syntax error. Translating is not optional and neither is doing it exactly: a census that
    re-typed the arithmetic would agree with itself instead of with the source, which is the
    empty-parse failure class one level up. Anything this cannot parse is a refusal.
    """
    expr = expr.strip()
    q, c = _split_ternary(expr)
    if q >= 0:
        cond, a, b = expr[:q], expr[q + 1:c], expr[c + 1:]
        return f"(({py_expr(a)}) if ({py_expr(cond)}) else ({py_expr(b)}))"
    out, i = [], 0
    while i < len(expr):
        ch = expr[i]
        if ch in "([":
            depth, j = 1, i + 1
            while j < len(expr) and depth:
                if expr[j] in "([":
                    depth += 1
                elif expr[j] in ")]":
                    depth -= 1
                j += 1
            if depth:
                die(f"unbalanced parentheses in `{expr}`")
            out.append("(" + py_expr(expr[i + 1:j - 1]) + ")")
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def evaluate(expr: str, k: dict[str, int]) -> int:
    """Evaluate one of `planned_launch_counts()`'s expressions ON THE SHIPPED DEFAULTS.

    The bill takes the two route flags as parameters now (`cast_fold`, `copy_fold`), and this census
    prices the population as it ships, so the fold arguments are bound to false here rather than left
    unresolved — a census that silently evaluated the FOLDED bill would report 49 glue launches and 0
    copies and call its own subject gone.
    """
    env = dict(k)
    env.setdefault("cast_fold", False)
    env.setdefault("copy_fold", False)
    try:
        return int(eval(py_expr(expr), {"__builtins__": {}}, env))  # noqa: S307  source is ours
    except CensusError:
        raise
    except Exception as exc:  # noqa: BLE001
        die(f"cannot evaluate planned_launch_counts()'s expression {expr!r}: {exc}")
        raise AssertionError  # unreachable


def analyse(step_txt: str) -> dict[str, list[int]]:
    """Locate every site by call text, then check its control-flow claim. Raises CensusError."""
    src = step_txt.splitlines()
    found: dict[str, list[int]] = {}
    for site in SITES:
        found[site["name"]] = [i for i, t in enumerate(src, 1) if re.search(site["pattern"], t)]
        if len(found[site["name"]]) != site["pattern_total"]:
            die(f"`{site['pattern']}` appears {len(found[site['name']])} time(s) in "
                f"engine/decode_step.cpp, the table claims {site['pattern_total']} — a moved, "
                "duplicated or deleted submission changes this route's population")
    for site in WIDE_ONLY:
        hits = [i for i, t in enumerate(src, 1) if re.search(site["pattern"], t)]
        if len(hits) != 1:
            die(f"wide-body-only site `{site['pattern']}` appears {len(hits)} times; the census"
                " asserts exactly one so that it can NAME what it excludes")
        site["line"] = hits[0]
    check_control_flow(src, found)
    return found


def check_control_flow(src: list[str], found: dict[str, list[int]]) -> dict[str, str]:
    """`per` is a claim about CONTROL FLOW, not about call text.

    `copy_conv_rollback` is billed kNumGdnLayers only because it sits under `if (!attn_layer)`, and
    the two `add_norm` sites are billed kNumLayers only because they sit in the layer loop under no
    mixer guard. A guard that moved is exactly the change that turns 49 into some other number while
    every call text still matches, so re-read it from the source rather than trusting the table.
    """
    # The file has FOUR `for (int layer = 0; ...)` loops — prepare_weight_adapters, submit_wave_body,
    # check_weights and proj_plan_jobs — and taking the first finds the adapter warm-up's, which is
    # ABOVE every glue site and would make the norm-only call read as "inside the layer loop".
    body = next((i for i, t in enumerate(src, 1) if t.startswith("void submit_wave_body")), None)
    if body is None:
        die("cannot bound submit_wave_body() in engine/decode_step.cpp")
    layer_loop = next((i for i, t in enumerate(src, 1)
                       if i > body and re.search(r"for \(int layer = 0; layer < kNumLayers; "
                                                 r"\+\+layer\)", t)), None)
    if layer_loop is None:
        die("no `for (int layer = 0; layer < kNumLayers; ++layer)` inside submit_wave_body()")
    where: dict[str, str] = {}
    for site in SITES:
        line = found[site["name"]][0]
        chain = enclosing(src, line)
        in_loop = any(i == layer_loop for i, _ in chain)
        if site["per"] == "once":
            desc, ok = "outside the layer loop", not in_loop
        elif site["per"] == "layers":
            desc = "in the layer loop, under no mixer guard"
            ok = in_loop and not _inside_gdn_guard(src, line)
        elif site["per"] == "gdn":
            desc = "in the layer loop, under `if (!attn_layer)`"
            ok = in_loop and _inside_gdn_guard(src, line)
        else:
            die(f"site {site['name']} has an unknown per={site['per']!r}")
        if not ok:
            die(f"site {site['name']} is at line {line}, which is not {desc}: the multiplicity"
                f" per={site['per']!r} is wrong, and every count derived from it is wrong with it")
        where[site["name"]] = desc
    # The roll-back must now live on the SHIPPED side of the fold guard: if someone moves it out, the
    # folded path submits it anyway and the `copy` bill this census prices is a lie. Naming the branch
    # is the cheapest witness that the flag and the submission are still in the same `if`.
    roll = found["copy_conv_rollback"][0]
    fold_guards = [i for i, t in enclosing(src, roll) if re.search(r"if \(fold_copy\)", t)]
    if not fold_guards:
        die(f"the conv roll-back at line {roll} is no longer under a `fold_copy` guard, so the route's"
            " flag does not reach the submission this census bills")
    if not any(re.search(r"^\s*\}\s*else\s*\{", src[i]) for i in range(fold_guards[0], roll - 1)):
        die(f"the conv roll-back at line {roll} is on the FOLDED side of `if (fold_copy)` — the shipped"
            " path would stop submitting it")
    # The fused launch is submitted from ONE lambda body, and the TWO `add_norm` call sites are what
    # make it 2 per layer. Both checked: a third call site would move 49 silently.
    lam = [i for i, t in enumerate(src, 1)
           if re.search(r"kernels::glue_add_norm_rows\(s, delta, ", t)]
    if len(lam) != 1:
        die(f"the fused add+norm is submitted from {len(lam)} place(s); expected the one `add_norm`"
            " lambda body")
    calls = [i for i, t in enumerate(src, 1) if re.search(r"\badd_norm\(sc\.delta, ", t)]
    if len(calls) != 2:
        die(f"{len(calls)} `add_norm(sc.delta, ...)` call sites, expected 2 — that number IS the 2 in"
            " `c.glue = 1 + 2 * kNumLayers`")
    return where


def enclosing(src: list[str], line: int, limit: int = 10) -> list[tuple[int, str]]:
    """The block-opening lines that enclose `line`, nearest first, as (lineno, text).

    Walks up counting braces and consuming one unmatched `{` per enclosing level, which is what makes
    it survive a site nested deeper than one `if` — the roll-back now sits under
    `if (!attn_layer) { … if (fold_copy) { … } else { <the copy> } }`, and a checker that only names
    the innermost opener would report that site as ungarded and silently lose the 18.
    """
    depth, out = 0, []
    for i in range(line - 2, -1, -1):
        depth += src[i].count("}") - src[i].count("{")
        while depth < 0:
            out.append((i + 1, src[i]))
            depth += 1
            if len(out) >= limit:
                return out
    return out


def _inside_gdn_guard(src: list[str], line: int) -> bool:
    """True when some enclosing block is the mixer guard `if (!attn_layer)`."""
    return any("if (!attn_layer)" in t for _, t in enclosing(src, line))


def launch_counts(k: dict[str, int]) -> dict[str, int]:
    per = {"once": 1, "layers": k["kNumLayers"], "gdn": k["kNumGdnLayers"]}
    return {site["name"]: per[site["per"]] for site in SITES}


def site_bytes(k: dict[str, int], counts: dict[str, int]) -> dict[str, dict[str, int]]:
    """Per-family bytes, summed from the operand rows each site names."""
    out: dict[str, dict[str, int]] = {}
    for site in SITES:
        wide = k[site["row"]] * BYTES_OF[site["dtype"]]
        n = counts[site["name"]]
        fam = out.setdefault(site["family"], {"read": 0, "write": 0})
        fam["read"] += n * site["nrows_read"] * wide
        fam["write"] += n * site["nrows_write"] * wide
    return out


def traffic_rows() -> dict[str, int]:
    """The traffic model's rows, keyed by their OWN note text, so each comparison is named.

    Keyed by note and not by family because `copy` has two write rows (the roll-back and the
    embedding row) and a family-keyed parser keeps only the last one it saw and reports 0.002 MB for
    a family that moves 1.329 — which is the empty-parse failure class, one level down.
    """
    if not TRAFFIC.exists():
        die(f"{TRAFFIC} missing; the byte side of the table has no source")
    res = subprocess.run([sys.executable, str(TRAFFIC)], capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        die(f"{TRAFFIC.name} exited {res.returncode}: {res.stderr.strip()[:200]}")
    out: dict[str, int] = {}
    for line in res.stdout.splitlines():
        m = re.match(r"^(\S+)\s+(read|write)\s+([0-9]+\.[0-9]+)\s+[0-9.]+%\s+(.*)$", line)
        if m:
            out[f"{m.group(1)}.{m.group(2)}: {m.group(4).strip()}"] = round(float(m.group(3)) * 1e6)
    if len(out) < 20:
        die(f"parsed {len(out)} traffic rows from {TRAFFIC.name}; expected the full family table")
    return out


def find_row(rows: dict[str, int], needle: str) -> int:
    hits = [v for key, v in rows.items() if needle in key]
    if len(hits) != 1:
        die(f"{len(hits)} traffic rows match {needle!r}; the comparison has to name ONE row")
    return hits[0]


def reconcile(k: dict[str, int], counts: dict[str, int], mine: dict[str, dict[str, int]]) -> list[str]:
    """Compare each derived subtotal with the ONE model row it corresponds to.

    Tolerance is the model's print precision (±500 B) and nothing else. Each named difference is then
    explained in the note, so a reader can see WHY the two differ instead of being told they are close.
    """
    notes: list[str] = []
    row = k["kHidden"] * 2                        # one BF16 row of the residual family
    glue_launches = sum(counts[s["name"]] for s in SITES if s["family"] == "glue")

    def cmp(label: str, derived: int, correction: int, note: str, why: str) -> None:
        """`correction` is the EXACT extra the model's row bills over this table's, so the only
        slack left is the model's own print precision and not a fudge factor."""
        want = find_row(traffic_rows(), note)
        expect = derived + correction
        if abs(expect - want) > MB_PRINT:
            die(f"{label}: this table says {derived} B plus a stated {correction:+d} B = {expect} B,"
                f" `{note}` says {want} B — the residue {expect - want} B is larger than the model's"
                f" print precision ({MB_PRINT} B)")
        notes.append(f"{label}: derived {derived} B, model {want} B, explained difference"
                     f" {correction:+d} B (residual {want - expect:+d} B is the model's own %.3f MB"
                     f" rounding, ±{MB_PRINT} B) — {why}")

    cmp("glue read", mine["glue"]["read"], +row, "add+norm reads s, delta, weight",
        "the model bills 49 x 3 rows; layer 0's norm-only call passes delta == nullptr, so exactly one"
        " BF16 row (2048 B) is never read")
    cmp("glue write", mine["glue"]["write"], +row, "s_out + norm_out per glue launch",
        "the same launch passes s_out == nullptr and writes n_out only, so again one row less")
    cmp("copy read (roll-back)", counts["copy_conv_rollback"] * k["kConvWindow"] * k["kConvChannels"]
        * BYTES_OF["F32"], 0, "roll-back D2D copy reads conv_new", "the 18 copies, row for row")
    cmp("copy write (roll-back)",
        counts["copy_conv_rollback"] * k["kConvWindow"] * k["kConvChannels"] * BYTES_OF["F32"], 0,
        "roll-back D2D copy writes the window", "ditto")
    embed = counts["copy_embed_row"] * k["kHidden"] * BYTES_OF["BF16"]
    cmp("copy read, all sites", mine["copy"]["read"], -embed, "roll-back D2D copy reads conv_new",
        f"this table counts the embedding row copy's {embed} B read; the model has no read row for"
        " that copy at all")
    cmp("copy write, all sites", mine["copy"]["write"], -embed, "roll-back D2D copy writes the window",
        f"the model DOES carry `embedding row copy (tied table row -> row 0)` as a {embed} B write"
        " row, but its printer drops every row under 4,096 B"
        " (`tools/step_traffic_model.py`, `if b < 4096: continue`), so it never reaches the table this"
        " census reads — a 2 KB submission is invisible to the traffic print and still costs a full"
        " submission")
    weight_rows = glue_launches * row
    attn_qk = 2 * k["kHeadDim"] * k["kNumAttnLayers"] * 2
    cmp("norms read", weight_rows + attn_qk, 0, "input/post-attn norms, q_norm/k_norm",
        f"the norms row IS the {weight_rows} B of weight vectors the {glue_launches} glue launches"
        f" already read (and the model's glue row counts them too), plus q_norm/k_norm's {attn_qk} B,"
        " which belongs to the attention site and to no glue launch")
    notes.append(f"ATTRIBUTION: §1's glue row reads 0.407 MB because `prototype/launch_price/census.py`"
                 f"'s STAGE_OF folds `norms` into the `glue` stage; {weight_rows} B of that 0.407 is"
                 f" the 49 weight rows counted twice, and {attn_qk} B of it is q_norm/k_norm. On this"
                 f" site table the 49 glue launches move {mine['glue']['read']} B read /"
                 f" {mine['glue']['write']} B written, and the 19 copies move"
                 f" {mine['copy']['read']} B / {mine['copy']['write']} B.")
    return notes


def fold_price(k: dict[str, int], counts: dict[str, int]) -> dict:
    """What riding a row norm on its CONSUMER would cost, bottom-up, per token.

    This is the arithmetic the not-foldable classification rests on, so the block geometries are READ
    from the sources: a consumer block owns one output row, so a norm that rides the consumer is
    recomputed once per block, each copy re-reading the norm's operands and redoing the whole-row
    reduction. `mlp_gemv` ships ROWS_PER_BLOCK, `proj_group_kernels.h` ships kProjGroupRowsPerBlock;
    lm_head's one-row form shares the mlp family's block shape.
    """
    def grab(path: Path, rx: str, what: str) -> int:
        m = re.search(rx, path.read_text())
        if not m:
            die(f"cannot read {what} out of {path.name}; the fold price would have no source")
        return int(m.group(1))

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    mlp_rpb = grab(MLP_KERNELS, r"#define ROWS_PER_BLOCK\s+(\d+)", "ROWS_PER_BLOCK")
    lmh_rpb = grab(LMHEAD_KERNELS, r"#define ROWS_PER_BLOCK\s+(\d+)", "lm_head ROWS_PER_BLOCK")
    grp_rpb = grab(GROUP_KERNELS, r"kProjGroupRowsPerBlock\s*=\s*(\d+)", "kProjGroupRowsPerBlock")
    gateup_blocks = ceil_div(k["kIntermediate"], mlp_rpb)
    gdn_group_blocks = ceil_div(k["kQkvRows"] + k["kValueDim"] + 2 * k["kNumValueHeads"], grp_rpb)
    attn_group_blocks = ceil_div(k["kQprojRows"] + 2 * k["kKvProjRows"], grp_rpb)
    lm_head_blocks = ceil_div(k["kVocab"], lmh_rpb)

    def input_group_blocks(layer: int) -> int:
        return attn_group_blocks if is_attn_layer(layer, k) else gdn_group_blocks

    # Walk the step and name each norm's consumer block, layer by layer, so the count is a walk and not
    # a formula: layer 0's norm-only site, then per layer the post-attn norm and the next-input norm.
    per_block = 3 * k["kHidden"] * 2               # s + delta + w, the norm's operands
    blocks = 0
    walk = []
    blocks += input_group_blocks(0)
    walk.append(("L00 norm-only", "layer 0 input-projection group", input_group_blocks(0)))
    for layer in range(k["kNumLayers"]):
        blocks += gateup_blocks
        walk.append((f"L{layer:02d} post-attn", "mlp_gateup", gateup_blocks))
        if layer + 1 < k["kNumLayers"]:
            n = input_group_blocks(layer + 1)
            blocks += n
            walk.append((f"L{layer:02d} next-input", f"layer {layer+1} input-projection group", n))
        else:
            blocks += lm_head_blocks
            walk.append((f"L{layer:02d} next-input", "lm_head (final norm)", lm_head_blocks))
    return dict(mlp_rpb=mlp_rpb, grp_rpb=grp_rpb, lmh_rpb=lmh_rpb, gateup_blocks=gateup_blocks,
                gdn_group_blocks=gdn_group_blocks, attn_group_blocks=attn_group_blocks,
                lm_head_blocks=lm_head_blocks, consumer_blocks=blocks, per_block_read=per_block,
                extra_read_bytes=blocks * per_block, walk=walk,
                glue_launches=sum(counts[s["name"]] for s in SITES if s["family"] == "glue"))


def render(rows: list[dict]) -> str:
    hdr = (f"{'site':<26}{'family':<7}{'launches':>8}{'read B':>12}{'write B':>12}"
           f"{'rows r/w':>10}  where")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        rw = f"{r['nrows_read']}/{r['nrows_write']}"
        out.append(f"{r['name']:<26}{r['family']:<7}{r['launches']:>8}{r['bytes_read']:>12}"
                   f"{r['bytes_written']:>12}{rw:>10}  {r['where']}")
    out.append("-" * len(hdr))
    out.append(f"{'TOTAL per decode token':<33}{sum(r['launches'] for r in rows):>8}"
               f"{sum(r['bytes_read'] for r in rows):>12}{sum(r['bytes_written'] for r in rows):>12}")
    return "\n".join(out)


def selftest() -> int:
    """Each derivation, fed a source that must make it refuse. A check that cannot fail is decoration."""
    good_state = STATE_H.read_text()
    good_step = STEP_CPP.read_text()
    cases = [
        ("state.h with kConvWindow renamed", lambda: shapes(good_state.replace(
            "constexpr int kConvWindow = kConvKernel - 1;", "constexpr int kRenamedAway = 3;"))),
        ("state.h with an unresolvable product", lambda: shapes(good_state.replace(
            "constexpr int kConvChannels = 6144;", "constexpr int kConvChannels = not_a_name * 2;"))),
        ("state.h whose routing line moved", lambda: shapes(good_state.replace(
            "return layer % 4 == 3;", "return is_attn(layer);"))),
        ("the conv roll-back renamed away", lambda: analyse(good_step.replace(
            "hipMemcpyAsync(st.conv_window_layer(layer), sc.conv_new",
            "hipMemcpyAsync(st.not_the_window, sc.conv_new"))),
        ("a duplicated add_norm call site", lambda: analyse(
            good_step + "\n// a third glue call would be a 50th launch\n"
            "void dup() { add_norm(sc.delta, next_norm); }\n")),
        ("the norm-only call edited away", lambda: analyse(
            good_step.replace("kernels::glue_add_norm_rows(s, nullptr,",
                              "kernels::glue_other(s, nullptr,"))),
        ("the wide-only gather deleted", lambda: analyse(
            good_step.replace("kernels::embed_gather_rows(table, ids_dev, sc.embed, n);", ";"))),
        ("the roll-back leaving its !attn_layer guard", lambda: analyse(
            good_step.replace("        if (!attn_layer) {\n            const int slot = gdn_slot(layer);",
                              "        if (true) {\n            const int slot = gdn_slot(layer);"))),
        ("the roll-back moved onto the folded side of the flag", lambda: analyse(
            good_step.replace("            } else {\n                kernels::gdn_conv(",
                              "            }\n            if (true) {\n                kernels::gdn_conv(",
                              1))),
        ("the fold guard renamed, so the flag no longer reaches the copy", lambda: analyse(
            good_step.replace("if (fold_copy) {", "if (fold_something_else) {"))),
        ("a source with no layer loop inside the body", lambda: analyse(
            good_step.replace("    for (int layer = 0; layer < kNumLayers; ++layer) {\n"
                              "        const bool attn_layer",
                              "    for (int layer = 0; layer < kWhatever; ++layer) {\n"
                              "        const bool attn_layer"))),
        ("a bill row whose ternary has no colon", lambda: evaluate("copy_fold ? 0", {**shapes()})),
        ("planned_launch_counts losing its glue row", lambda: planned_exprs(
            good_step.replace("c.glue = 1 + 2 * kNumLayers;", "// gone"))),
    ]
    failed = 0
    for what, fn in cases:
        try:
            fn()
        except CensusError as exc:
            print(f"  selftest refused as required  [{what}]  -> {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  SELFTEST FAIL [{what}] raised {type(exc).__name__} instead of CensusError: {exc}")
        else:
            failed += 1
            print(f"  SELFTEST FAIL [{what}] returned normally — that check cannot fail")
    # The population assertion: a check that cannot see its own number guards nothing.
    k = shapes()
    counts = launch_counts(k)
    tot = sum(counts.values())
    if tot != POPULATION:
        failed += 1
        print(f"  SELFTEST FAIL: the site table bills {tot} launches, §1 bills {POPULATION}")
    glue = sum(counts[s["name"]] for s in SITES if s["family"] == "glue")
    if glue != 1 + 2 * k["kNumLayers"]:
        failed += 1
        print(f"  SELFTEST FAIL: the glue sites bill {glue}, `1 + 2*kNumLayers` says"
              f" {1 + 2 * k['kNumLayers']}")
    copy = sum(counts[s["name"]] for s in SITES if s["family"] == "copy")
    if copy != 1 + k["kNumGdnLayers"]:
        failed += 1
        print(f"  SELFTEST FAIL: the copy sites bill {copy}, expected 1 embedding row +"
              f" {k['kNumGdnLayers']} rolls")
    print(f"SELFTEST {'PASS' if failed == 0 else f'FAIL ({failed})'}")
    return 0 if failed == 0 else 1


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return selftest()
    k = shapes()
    step_txt = STEP_CPP.read_text()
    found = analyse(step_txt)
    where = check_control_flow(step_txt.splitlines(), found)
    counts = launch_counts(k)
    rows = []
    for site in SITES:
        wide = k[site["row"]] * BYTES_OF[site["dtype"]]
        n = counts[site["name"]]
        rows.append(dict(site, launches=n, where=where[site["name"]], line=found[site["name"]][0],
                         bytes_read=n * site["nrows_read"] * wide,
                         bytes_written=n * site["nrows_write"] * wide))
    bill = planned_exprs(step_txt)
    glue_plan, copy_plan = evaluate(bill["glue"], k), evaluate(bill["other"], k) - 1
    # The FOLDED bill, read from the same source text with the flag argument true: the copies must go to
    # zero and nothing else may move. A `c.other` that folded the lm_head launch too, or a `c.glue` that
    # reacted to the wrong flag, is caught here rather than in a timing window.
    folded_bill = {"glue": evaluate(bill["glue"], {**k, "copy_fold": True}),
                   "other": evaluate(bill["other"], {**k, "copy_fold": True}) - 1}
    print(f"layers: kNumLayers={k['kNumLayers']} kNumGdnLayers={k['kNumGdnLayers']} "
          f"kNumAttnLayers={k['kNumAttnLayers']}   widths: kHidden={k['kHidden']} "
          f"kConvChannels={k['kConvChannels']} kConvWindow={k['kConvWindow']}  (engine/state.h)")
    print(f"second derivation, from planned_launch_counts()'s own text: `c.glue = {bill['glue']}`"
          f" -> {glue_plan}; `c.other = {bill['other']}` -> {copy_plan} copies (other minus lm_head)")
    print("excluded as wide-body-only, a decode step never reaches them: "
          + "; ".join(f"{w['name']} line {w['line']} ({w['note']})" for w in WIDE_ONLY))
    print()
    print(render(rows))
    print()
    for r in rows:
        print(f"  {r['name']} (line {r['line']}, {r['where']}): {r['launches']} launch(es)/step")
        print(f"      reads  {r['reads']}")
        print(f"      writes {r['writes']}")
        print(f"      consumer: {r['consumer']}")
        print(f"      [{r['klass']}]")
        print(f"      why: {r['why']}")
        print()
    folded = sum(r["launches"] for r in rows
                 if r["klass"].startswith("foldable") or "FOLDED" in r["klass"])
    total = sum(r["launches"] for r in rows)
    bytes_by_family = site_bytes(k, counts)
    for note in reconcile(k, counts, bytes_by_family):
        print("RECONCILE: " + note)
    price = fold_price(k, counts)
    print()
    print("FOLD-INTO-CONSUMER PRICE — the arithmetic the 49 not-foldable rows rest on.")
    print("  A consumer block owns ONE output row, so a norm that rides the consumer is recomputed")
    print("  once per block, each copy re-reading the norm's operands and redoing the whole-row")
    print(f"  reduction. Shipped one-row geometries: mlp_gateup rows/block {price['mlp_rpb']} ->"
          f" {price['gateup_blocks']} blocks; input-projection group rows/block {price['grp_rpb']} ->"
          f" {price['gdn_group_blocks']} (GDN layer) / {price['attn_group_blocks']} (attention);"
          f" lm_head rows/block {price['lmh_rpb']} -> {price['lm_head_blocks']}.")
    for what, who, n in price["walk"]:
        print(f"    {what:<18} -> {who:<36} {n:>6} blocks")
    glue_prize = price["glue_launches"] * PER_LAUNCH_US / 1000.0
    extra_ms = price["extra_read_bytes"] / (READ_PLATEAU_GBS * 1e9) * 1000.0
    print(f"  = {price['consumer_blocks']} consumer blocks per token x {price['per_block_read']} B of"
          f" re-read operands = {price['extra_read_bytes']:,} B"
          f" = {price['extra_read_bytes']/1e6:.0f} MB/token of EXTRA reads, plus that many redundant")
    print(f"  whole-row reductions, to delete {price['glue_launches']} launches worth"
          f" {glue_prize:.3f} ms at the measured {PER_LAUNCH_US} us/launch (quoted, §4). The extra")
    print(f"  reads alone are {extra_ms:.3f} ms at the ~{READ_PLATEAU_GBS:.0f} GB/s read plateau"
          f" (quoted) = {extra_ms/glue_prize:.1f}x the prize they buy, before any of the redundant")
    print("  VALU is priced. And it is not bit-identical either: a consumer block is 1024 threads over")
    print("  a 1024-element row, which is a DIFFERENT F32 summation order for the mean of squares than")
    print("  glue's 256 threads x 4 elements — engine/kernels_glue.cpp's own note on exactly that, and")
    print("  prototype/glue/floor_probe.py, which sizes a norm's legal floor at 1 ULP with ndiff as the")
    print("  criterion doing the work. So the fold would buy a band AND pay the reads.")
    print()
    print(f"FOLDED {folded} of {total} launches. Ceiling at {PER_LAUNCH_US} us/launch (quoted):"
          f" {total*PER_LAUNCH_US/1000:.4f} ms for the whole population,"
          f" {folded*PER_LAUNCH_US/1000:.4f} ms for what actually folds.")
    print(f"The folded bill, read back out of `c.other`: copies {copy_plan} -> "
          f"{folded_bill['other']}, glue unchanged at {folded_bill['glue']}.")
    problems = []
    glue_n = sum(r["launches"] for r in rows if r["family"] == "glue")
    copy_n = sum(r["launches"] for r in rows if r["family"] == "copy")
    if glue_n != glue_plan:
        problems.append(f"the glue sites bill {glue_n}/token, planned_launch_counts says {glue_plan}")
    if copy_n != copy_plan:
        problems.append(f"the copy sites bill {copy_n}/token, `other` minus lm_head says {copy_plan}")
    if total != POPULATION:
        problems.append(f"the sites bill {total}/token, §1 bills {POPULATION}")
    if folded_bill["other"] != 0:
        problems.append(f"with copy_fold on, `other` minus lm_head still bills {folded_bill['other']}"
                        " copies; the bill did not fold")
    if folded_bill["glue"] != glue_plan:
        problems.append(f"with copy_fold on, the glue row moved to {folded_bill['glue']}; this route"
                        " deletes no glue launch")
    if problems:
        for p in problems:
            print("CENSUS FAIL: " + p)
        return 1
    print(f"SITES ok: 5 call sites, {total} launches/token ({glue_n} glue + {copy_n} copy),"
          f" {folded} fold and {total - folded} do not; every byte row reconciles against"
          f" {TRAFFIC.name} at its own print precision; the two launch totals agree with"
          " planned_launch_counts() by a second, independent derivation.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CensusError as e:
        print(f"CENSUS FAIL: {e}")
        sys.exit(1)
