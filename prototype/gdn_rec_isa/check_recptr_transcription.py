#!/usr/bin/env python3
"""Is `recptr_kernel` `recloop_kernel` with the addressing moved, and NOTHING else moved?

WHY THIS EXISTS, AND WHY IT IS NOT THE DEVICE GATE. The item claims that a token step costs the same
work whether the compiler derives each row address from the token index (`qtile + (size_t)t * CH`) or
advances a pointer by that stride (`q += CH`). The device gate is the only thing that can show the
COMPILER kept the addresses equal — but it is a device run, and before spending one a reviewer should be
able to check cheaply that the SOURCE moved only where the item says it moved. That is this file.

It is deliberately separate from `check_recloop_transcription.py`. That checker proves the looped body is
the shipping body; this one proves the POINTER body is the LOOPED body, and it must therefore allow
exactly one class of difference — the addressing — which is the whole difficulty. A checker that ignored
the addressing lines would pass while the strides were wrong, and a wrong stride is a silent numerics
change, not a cosmetic one: `o += NH * D * 2` walks into the next head's rows, and a one-token fixture
could not notice at all.

WHAT IS COMPARED, in three parts:
  (a) THE ARITHMETIC CORE — every line from `float kq =` through `o[h * D + j] = acc;`, whitespace
      collapsed, comments stripped, `#pragma unroll` lines INCLUDED. Character-identical between the two
      kernels. Per §19 of `prototype/gdn_collapse/results.md`, bit-identity is owed to the source text's
      ASSOCIATION, not to the algebra, so a re-associated product fails here even though it is the same
      value.
  (b) THE ADDRESSING, keyed by pointer NAME rather than by position. `recloop_kernel` declares each of the
      six rows as `<base> + (size_t)t * <STRIDE>`; `recptr_kernel` declares it as `<base>` and advances it
      with `+= <STRIDE'>`. The check requires STRIDE == STRIDE' and the same `<base>` for every one of
      q, k, v, g, beta, o. Both sides are READ from the sources — this file never lists a stride, so if the
      shipped form's stride ever changes the check follows it instead of going stale. Because the
      comparison is keyed by name, the six `+=` statements may be written in any order (they are
      independent), which is what `MUST_NOT_CATCH` proves.
  (c) NOTHING ELSE — with the declarations and the six steps removed from both bodies, the remaining line
      sequences must be identical. That is what stops a third kind of change (a hoisted load, a moved
      `__syncthreads()`, a deleted store) from sheltering inside the allowed difference.

A CHECK THAT CANNOT FAIL IS DECORATION (`prototype/mlp_gemv`, `tools/mixed_bw`,
`prototype/attn_tile_model.py`, this directory's own `--selftest-lane`). So `--selftest` mutates the
extracted `recptr` body once per defect class and requires each to be REPORTED: a wrong stride, a stride
scaled by two, a swapped base buffer, a re-associated product, a dropped `#pragma unroll`, a dropped
store. It also requires a cosmetic reorder of the six steps NOT to be reported — a checker that fires on
everything adjudicates nothing, which is the `--scope form` lesson from `engine/gate.py`.

Usage:
    prototype/gdn_rec_isa/check_recptr_transcription.py            # the real diff
    prototype/gdn_rec_isa/check_recptr_transcription.py --selftest # the must-catch set
Exit 0 = only the addressing differs, and (under --selftest) every defect class is visible.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "prototype/gdn_rec_isa/gdn_rec_isa_kernels.h"

CORE_START = "float kq ="
CORE_END = "o[h * D + j] = acc;"
MIN_CORE_LINES = 30                       # the shipped body is ~40 code lines; short = parse failure

# The six rows whose addressing the item moves. Names, not offsets: the offsets are what is under test.
POINTERS = ("q", "k", "v", "g", "beta", "o")

DECL_RE = r"(?:const\s+float\s*\*|float\s*\*)\s*__restrict__\s*(%s)\s*=\s*(.+);"
STEP_RE = r"^(%s)\s*\+=\s*(.+);$"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def kernel_body(path: Path, name: str) -> str:
    """The brace-matched body of `__global__ void <name>(...)`."""
    src = strip_comments(path.read_text())
    m = re.search(r"__global__\s+void\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{", src)
    if not m:
        raise SystemExit(f"{path.name}: no `__global__ void {name}(...)` found — the checker would be"
                         " comparing nothing, which is a FAIL, not a pass")
    i = m.end() - 1
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
    raise SystemExit(f"{path.name}: unbalanced braces in {name}")


def code_lines(body: str) -> list[str]:
    return [" ".join(l.split()) for l in body.splitlines() if l.strip()]


def norm_expr(text: str) -> str:
    """`  (NH * D)  ` -> `NH * D`: outer parentheses and runs of spaces are spelling, not arithmetic.

    Only a MATCHED pair of outer parens goes — `(NH * D) + 1` keeps its shape — so a stride written with a
    redundant pair cannot read as a different stride, and a real expression is never flattened."""
    t = " ".join(text.split()).rstrip(";").strip()
    while len(t) > 2 and t.startswith("(") and t.endswith(")"):
        depth = 0
        close = -1
        for i, ch in enumerate(t):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close != len(t) - 1:      # the leading paren closes early: not a redundant pair
            break
        t = t[1:-1].strip()
    return t


def core_of(body: str) -> list[str]:
    """(a) the arithmetic core, inclusive of the two boundary lines."""
    out, inside = [], False
    for line in code_lines(body):
        if not inside and line.startswith(CORE_START):
            inside = True
        if inside:
            out.append(line)
            if line.startswith(CORE_END):
                return out
    raise SystemExit(f"core not found: no `{CORE_START}` ... `{CORE_END}` in a"
                     f" {len(body.splitlines())}-line body — the extraction is broken")


def addressing(lines: list[str]) -> dict[str, dict[str, str | None]]:
    """(b) per pointer name: what it is declared from, what the looped form multiplies `t` by, and what
    the pointer form adds each step. Missing entries are reported as problems by `compare_addressing`,
    never silently skipped."""
    decl_re = DECL_RE % "|".join(POINTERS)
    step_re = STEP_RE % "|".join(POINTERS)
    out: dict[str, dict[str, str | None]] = {}
    for line in lines:
        m = re.match(decl_re, line)
        if m:
            name, rhs = m.group(1), " ".join(m.group(2).split())
            off = re.search(r"\+\s*\(size_t\)t\s*\*\s*(.+?)\s*$", rhs)
            if off:
                out[name] = {"base": norm_expr(rhs[:off.start()].rstrip(" +")),
                             "stride": norm_expr(off.group(1)), "step": None}
            else:
                out[name] = {"base": norm_expr(rhs), "stride": None, "step": None}
            continue
        m = re.match(step_re, line)
        if m:
            out.setdefault(m.group(1), {"base": None, "stride": None, "step": None})
            if out[m.group(1)]["step"] is not None:
                raise SystemExit(f"`{m.group(1)}` is advanced twice in one token step — the two forms"
                                 " are not comparable and the checker refuses to guess")
            out[m.group(1)]["step"] = norm_expr(m.group(2))
    return out


def strip_addressing(lines: list[str]) -> list[str]:
    """(c) the body without the six declarations and the six steps."""
    decl_re = DECL_RE % "|".join(POINTERS)
    step_re = STEP_RE % "|".join(POINTERS)
    return [l for l in lines if not re.match(decl_re, l) and not re.match(step_re, l)]


def compare_addressing(looped: dict, ptr: dict) -> list[str]:
    problems: list[str] = []
    for name in POINTERS:
        if name not in looped:
            problems.append(f"`recloop_kernel` declares no `{name}` — the POINTER list in this file is"
                            " stale and cannot adjudicate the addressing")
            continue
        if name not in ptr:
            problems.append(f"`recptr_kernel` declares no `{name}` — the pointer form lost a row")
            continue
        want, got = looped[name], ptr[name]
        if want["base"] != got["base"]:
            problems.append(f"`{name}` base: recloop `{want['base']}` vs recptr `{got['base']}`")
        if want["stride"] is None:
            problems.append(f"`{name}`: could not parse a `(size_t)t * ...` offset in recloop, so"
                            " nothing can be compared for it")
        if got["step"] is None:
            problems.append(f"`{name}` is never advanced in `recptr_kernel`: every token after the first"
                            " re-reads token 0's row, which a one-token fixture cannot see")
        elif want["stride"] is not None and got["step"] != want["stride"]:
            problems.append(f"`{name}` stride: recloop `(size_t)t * {want['stride']}` vs recptr"
                            f" `+= {got['step']}`")
    for extra in sorted((set(looped) | set(ptr)) - set(POINTERS)):
        problems.append(f"a seventh row pointer `{extra}` appeared — this checker only knows"
                        f" {', '.join(POINTERS)}; extend POINTER after reading what it is for")
    return problems


def diff(a: list[str], b: list[str], an: str, bn: str) -> list[str]:
    problems = []
    if len(a) != len(b):
        problems.append(f"{an} is {len(a)} lines, {bn} is {len(b)} — different shape")
    for k in range(min(len(a), len(b))):
        if a[k] != b[k]:
            problems.append(f"line {k}:\n    {an}: {a[k]}\n    {bn}: {b[k]}")
    return problems


# --- mutations: each one is a defect this checker is OBLIGED to see ----------------------------------
# `where` says which half of the checker has to catch it: `addr` is judged by compare_addressing, `core`
# by the arithmetic-core diff. A mutation is only a real control for the check that is supposed to see it.
def _step(name: str, new: str):
    def f(ls: list[str]) -> list[str]:
        out = list(ls)
        for i, l in enumerate(out):
            if re.match(STEP_RE % name, l):
                out[i] = f"{name} += {new};"
                return out
        raise SystemExit(f"no `{name} +=` line to mutate — the pointer form changed shape")
    return f


def _base(name: str, new: str):
    def f(ls: list[str]) -> list[str]:
        out = list(ls)
        pat = DECL_RE % name
        for i, l in enumerate(out):
            if re.match(pat, l):
                out[i] = f"const float* __restrict__ {name} = {new};"
                return out
        raise SystemExit(f"no `{name}` declaration to mutate — the pointer form changed shape")
    return f


def _reassoc(ls: list[str]) -> list[str]:
    out = [("float qn = qq * (rsqrtf(sq + 1e-6f) * Q_SCALE);"
            if l.startswith("float qn =") else l) for l in ls]
    if out == ls:
        raise SystemExit("no `float qn =` line to re-associate — the core changed shape")
    return out


def _pragma(ls: list[str]) -> list[str]:
    out = [l for l in ls if not l.startswith("#pragma unroll 4")]
    if out == ls:
        raise SystemExit("no `#pragma unroll 4` to drop — the core changed shape")
    return out


def _store(ls: list[str]) -> list[str]:
    out = [(l.replace("s0 += ks[i + 0] * dv; Sh[(i + 0) * D + j] = s0;", "s0 += ks[i + 0] * dv;")
            if l.startswith("float s0 =") else l) for l in ls]
    if out == ls:
        raise SystemExit("the state store did not match — the core changed shape")
    return out


def _drop_pointer(ls: list[str]) -> list[str]:
    return [l for l in ls if not re.match(STEP_RE % "o", l)]


def _reorder_steps(ls: list[str]) -> list[str]:
    """Swap two of the six `+=` lines. They are independent statements, so this cannot change a value."""
    pat = STEP_RE % "|".join(POINTERS)
    at = [i for i, l in enumerate(ls) if re.match(pat, l)]
    if len(at) < 2:
        raise SystemExit("fewer than two pointer steps found — nothing to reorder")
    out = list(ls)
    out[at[0]], out[at[1]] = out[at[1]], out[at[0]]
    return out


MUST_CATCH = {
    "o_stride_x2":    ("addr", _step("o", "NH * D * 2"),
                       "the o row steps twice as far as the index form: walks into the next head"),
    "g_stride_D":     ("addr", _step("g", "D"),
                       "the gate steps by D instead of NH: reads another head's gate"),
    "v_base_ktile":   ("addr", _base("v", "ktile"),
                       "v reads the k buffer: the value is wrong and every address in it is a lie"),
    "o_step_missing": ("addr", _drop_pointer,
                       "the o pointer never advances: every token rewrites token 0's output row"),
    "qscale_reassoc": ("core", _reassoc,
                       "`(qq*rsqrt)*Q_SCALE` re-parenthesised: the `prototype/gdn_collapse/results.md` §19 product-tree class"),
    "pragma_dropped": ("core", _pragma,
                       "`#pragma unroll 4` gone: LLVM fully unrolls and the billed listing is gone"),
    "store_dropped":  ("core", _store,
                       "one state store removed: the state stops being written for that column"),
}
MUST_NOT_CATCH = {
    "steps_reordered": (_reorder_steps,
                        "two independent `+=` statements swapped: keyed by name, so order cannot matter"),
}


def run_mutations(looped_body: str, ptr_body: str) -> int:
    bad = 0
    looped_lines, ptr_lines = code_lines(looped_body), code_lines(ptr_body)
    addr0 = compare_addressing(addressing(looped_lines), addressing(ptr_lines))
    core0 = diff(core_of(looped_body), core_of(ptr_body), "recloop core", "recptr core")
    if addr0 or core0:
        print("NOTE the two kernels do not agree as shipped; the mutations below are still judged, but"
              f" read the real diff first ({len(addr0)} addressing,"
              f" {len(core0)} core problems)")
    for name, (where, fn, why) in MUST_CATCH.items():
        try:
            mutated = fn(ptr_lines)
        except SystemExit as e:
            print(f"MUST-CATCH {name:16} NOT-APPLICABLE {e}")
            bad += 1
            continue
        if where == "addr":
            probs = compare_addressing(addressing(looped_lines), addressing(mutated))
        else:
            body = "\n".join(mutated)
            probs = diff(core_of(looped_body), core_of(body), "recloop core", "mutated recptr core")
        ok = bool(probs)
        bad += 0 if ok else 1
        print(f"MUST-CATCH {name:16} [{'addr' if where == 'addr' else 'core'}] "
              f"{'CAUGHT' if ok else 'MISSED  '}"
              + (f" — {probs[0]}" if probs else "") + f"\n{'':25}{why}")
    for name, (fn, why) in MUST_NOT_CATCH.items():
        mutated = fn(ptr_lines)
        probs = compare_addressing(addressing(looped_lines), addressing(mutated))
        ok = not probs
        bad += 0 if ok else 1
        print(f"MUST-NOT   {name:16} {'CLEAN' if ok else 'FIRED   '}"
              + (f" — {probs[0]}" if probs else "") + f"\n{'':25}{why}")
    print(f"SELFTEST recptr transcription must_catch={len(MUST_CATCH)}"
          f" must_not={len(MUST_NOT_CATCH)} bad={bad} ok={0 if bad else 1}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    looped_body = kernel_body(KERNELS, "recloop_kernel")
    ptr_body = kernel_body(KERNELS, "recptr_kernel")
    core_l, core_p = core_of(looped_body), core_of(ptr_body)
    for tag, core in (("recloop", core_l), ("recptr", core_p)):
        if len(core) < MIN_CORE_LINES:
            raise SystemExit(f"{tag}: only {len(core)} core lines (<{MIN_CORE_LINES}) — refusing to"
                             " certify an empty comparison")

    if args.selftest:
        return run_mutations(looped_body, ptr_body)

    lp, pp = code_lines(looped_body), code_lines(ptr_body)
    problems = diff(core_l, core_p, "recloop_kernel core", "recptr_kernel core")
    problems += compare_addressing(addressing(lp), addressing(pp))
    problems += diff(strip_addressing(lp), strip_addressing(pp),
                     "recloop minus addressing", "recptr minus addressing")
    steps = len([l for l in pp if re.match(STEP_RE % "|".join(POINTERS), l)])
    decls = len([l for l in pp if re.match(DECL_RE % "|".join(POINTERS), l)])
    print(f"TRANSCRIPTION recptr_core={len(core_p)} recloop_core={len(core_l)}"
          f" (must be >= {MIN_CORE_LINES}) pointers={len(POINTERS)} steps_found={steps}"
          f" decls_found={decls}"
          f" lines_compared_outside_addressing={len(strip_addressing(lp))}")
    if steps != len(POINTERS) or decls != len(POINTERS):
        problems.append(f"{decls} declarations and {steps} steps found, {len(POINTERS)} of each expected")
    if problems:
        print("TRANSCRIPTION FAIL — something other than the addressing moved:")
        for p in problems:
            print("  " + p)
        return 1
    print("TRANSCRIPTION OK — every non-addressing line of `recptr_kernel` is `recloop_kernel`'s"
          " character for character (both `#pragma unroll 4` lines included), and all six row pointers"
          " start at the same base buffer and advance by exactly the stride `recloop_kernel` multiplies"
          " `t` by. What this does NOT prove is that the COMPILER emitted the same addresses: the"
          " register allocator may strength-reduce one form and not the other, which is why the device"
          " gate (`--modes rec0,recptr`) is the admission and this file is only the pre-flight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
