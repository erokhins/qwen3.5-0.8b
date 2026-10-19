#!/usr/bin/env python3
"""A4-L1 canary: does one side of a build reproduce the SHIPPED instruction stream of one kernel?

`engine/build.sh` is not bit-reproducible on this box and a paired A/B against a stale or re-shaped binary
reads as a clean null (AGENTS.md, "a paired A/B against a stale binary reads as a clean null"). Both sides of
the A4-L1 build are compiled from COPIES of the engine TU, so the base side must be SHOWN to carry the
shipped code before any base-vs-new delta means anything. This compares the instruction text (addresses
stripped) of one kernel between any two listings.

    python3 prototype/attn_percost/a4_l1_canary.py LISTING_A LISTING_B [--kernel attn_fattn_wave]
    python3 prototype/attn_percost/a4_l1_canary.py --selftest

TWO LISTING FORMS ARE READ, because the bet has needed both:

  * `llvm-objdump -d` output — `000000000000da00 <_ZN6engine7kernels15attn_fattn_waveE8CoreArgs>:` labels
    with the encoded bytes in a trailing comment. This is what `a4_l1_build.sh` produces.
  * the `--save-temps` `.s` — the mangled name as a column-0 label (`_ZN6engine7kernels15attn_fattn_wave
    E8CoreArgs:`), directives and `;` comments interleaved. This is what `fattn_census_build.sh` and
    `engine/build.sh`'s own tree leave behind, and the first version of this canary REFUSED it: it looked for
    a label shaped like the objdump form and reported "no function matching", which is a canary that cannot
    load its input (the check-that-cannot-fail class this tree keeps meeting).

Kernel names are matched by exact short name (`attn_fattn_wave`), exact mangled name
(`_ZN6engine7kernels15attn_fattn_waveE8CoreArgs`), or an UNAMBIGUOUS substring — and ambiguity is a refusal,
not a first-match. That matters now that the engine carries `attn_fattn_wave` AND `attn_fattn_wave_two`: a
substring that could name either kernel, and a canary that read whichever the parser reached first, would
certify one kernel and report it as the other — the two-sides-are-the-same null, from the other direction.

Exit 1 on any instruction difference, with the first few differing lines; exit 1 on a listing or a kernel
name that could not be read at all.
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]

# objdump: `000000000000da00 <LABEL>:` opening a body, and `  <tab>mnemonic ...  // ADDR: BYTES` insns.
OBJ_LABEL = re.compile(r"^[0-9a-fA-F]{8,16} <(.+)>:\s*$")
OBJ_INSN = re.compile(r"^\s+(\S.*?)\s*//\s*[0-9A-Fa-f]{8,16}:")
# LLVM .s: the column-0 label that `.type NAME,@function` declares, and the insns under it.
S_TYPE = re.compile(r"^\s*\.type\s+([^\s,]+),@function\b")
S_LABEL = re.compile(r"^([^\s;/.$][^\s;:]*):\s*(?:;.*)?$")
S_INSN = re.compile(r"^\t([a-z][\w.]*)\b(.*)$")


def demangle(label: str) -> str:
    """`_ZN6engine7kernels15attn_fattn_waveE8CoreArgs` -> `engine::kernels::attn_fattn_wave`.

    Only the subset these listings emit: `_Z` / `_ZN...E` nested names with `<len><name>` components and a
    trailing parameter encoding. Anything else is returned unchanged, so a plain C label (`attn_core`) and an
    unmangled `_Z9attn_core8CoreArgs` both still resolve to a usable short name.
    """
    if not label.startswith("_Z"):
        return label
    body = label[2:]
    names = []
    if body.startswith("N"):
        body = body[1:]
        while body and body[0] != "E":
            m = re.match(r"(\d+)", body)
            if not m:
                return label
            n = int(m.group(1))
            start = m.end()
            if len(body) < start + n:
                return label
            names.append(body[start:start + n])
            body = body[start + n:]
        if body.startswith("E"):
            body = body[1:]
    else:
        m = re.match(r"(\d+)", body)
        if not m:
            return label
        n = int(m.group(1))
        start = m.end()
        if len(body) < start + n:
            return label
        names.append(body[start:start + n])
    return "::".join(names) if names else label


def short(label: str) -> str:
    """The last component of a (possibly qualified) name: what `--kernel` is normally written as."""
    return demangle(label).rsplit("::", 1)[-1]


def functions(text: str) -> dict:
    """{label -> [instruction text]} for every function in either listing form.

    The objdump form is tried first; if it yields nothing the `.s` form is read. A listing that yields zero
    functions raises, because "the parser matched nothing" and "the kernel is absent" must never look alike.
    """
    lines = text.splitlines()
    out, label = {}, None
    for ln in lines:
        m = OBJ_LABEL.match(ln)
        if m:
            label = m.group(1)
            out.setdefault(label, [])
            continue
        if label is not None:
            if OBJ_LABEL.match(ln) or ln.startswith("Disassembly of"):
                label = None
                continue
            mi = OBJ_INSN.match(ln)
            if mi:
                out[label].append(mi.group(1).strip())
    if out:
        return out
    declared = {m.group(1) for m in (S_TYPE.search(ln) for ln in lines) if m}
    out, label = {}, None
    for ln in lines:
        m = S_LABEL.match(ln)
        if m and (m.group(1) in declared or m.group(1).startswith("_Z")):
            label = m.group(1)
            out.setdefault(label, [])
            continue
        if label is None:
            continue
        if ln.startswith("\t.size") or ln.startswith("\t.globl") or ln.startswith("\t.section"):
            continue
        mi = S_INSN.match(ln)
        if mi and not ln.lstrip().startswith("."):
            out[label].append((mi.group(1) + " " + mi.group(2)).strip())
    if not out:
        raise SystemExit("REFUSE: no functions parsed from this listing in either form (objdump `<label>:` "
                         "or `.s` column-0 label). A canary that reads nothing cannot compare anything.")
    return out


def resolve(spec: str, funcs: dict, path: str) -> str:
    """The one function `spec` names, or a refusal naming the candidates."""
    exact = [k for k in funcs if k == spec or short(k) == spec or demangle(k) == spec]
    if len(exact) == 1:
        return exact[0]
    subs = sorted({k for k in funcs if spec in k or spec in short(k) or spec in demangle(k)})
    if len(exact) > 1:
        raise SystemExit(f"REFUSE: {spec!r} matches {len(exact)} labels in {path}: "
                         + ", ".join(sorted(exact)))
    if len(subs) == 1:
        return subs[0]
    raise SystemExit(f"REFUSE: --kernel {spec!r} matched {len(subs)} functions in {path}"
                     + (": " + ", ".join(subs) if subs else "")
                     + ". Name one exactly (short name or mangled). `attn_fattn_wave` is a substring of"
                       " `attn_fattn_wave_two`, so a first-match here could certify either one.")


def compare(a: str, b: str, kernel: str, verbose=True) -> bool:
    """True = the two listings' `kernel` bodies agree instruction-for-instruction."""
    fa, fb = functions(open(a).read()), functions(open(b).read())
    ka, kb = resolve(kernel, fa, a), resolve(kernel, fb, b)
    ba, bb = fa[ka], fb[kb]
    if verbose:
        print(f"{kernel}: {a} [{ka}] = {len(ba)} insns | {b} [{kb}] = {len(bb)} insns")
    if not ba or not bb:
        raise SystemExit(f"REFUSE: {kernel} parsed to {len(ba)}/{len(bb)} instructions"
                         " ({a} / {b}). Empty is not equal, it is unread.")
    if ba == bb:
        if verbose:
            print(f"IDENTICAL (addresses stripped): {len(ba)} instructions agree")
        return True
    if verbose:
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(ba, bb)) if x != y]
        print(f"DIFFER: {len(diff)} of {min(len(ba), len(bb))} aligned lines"
              + ("" if len(ba) == len(bb) else f"  (lengths differ: {len(ba)} vs {len(bb)})"))
        for i, x, y in diff[:6]:
            print(f"  [{i}] - {x}\n      + {y}")
    return False


FIXTURE_A = """
0000000000000000 <_ZN6engine7kernels15attn_fattn_waveE8CoreArgs>:
    0:\tv_mov_b32_e32 v129, 0\t\t\t\t// 0000000000000000: BF810081
    4:\tv_wmma_f32_16x16x16_f16 v[129:136], v[159:166], v[167:174], v[129:136]\t// 0000000000000004: Dxxx
0000000000000100 <_ZN6engine7kernels19attn_fattn_wave_twoE8CoreArgs>:
    0:\tv_mov_b32_e32 v137, 0\t\t\t\t// 0000000000000000: BF810081
"""
FIXTURE_B = """
0000000000000000 <_ZN6engine7kernels15attn_fattn_waveE8CoreArgs>:
    0:\tv_mov_b32_e32 v129, 0\t\t\t\t// 0000000000000000: BF810081
    4:\tv_wmma_f32_16x16x16_f16 v[129:136], v[159:166], v[167:174], v[129:136]\t// 0000000000000004: Dxxx
0000000000000100 <_ZN6engine7kernels19attn_fattn_wave_twoE8CoreArgs>:
    0:\tv_mov_b32_e32 v137, 0\t\t\t\t// 0000000000000000: BF810081
"""
FIXTURE_C = """
0000000000000000 <_ZN6engine7kernels15attn_fattn_waveE8CoreArgs>:
    0:\tv_mov_b32_e32 v129, 1\t\t\t\t// 0000000000000000: BF810081
    4:\tv_wmma_f32_16x16x16_f16 v[129:136], v[159:166], v[167:174], v[129:136]\t// 0000000000000004: Dxxx
0000000000000100 <_ZN6engine7kernels19attn_fattn_wave_twoE8CoreArgs>:
    0:\tv_mov_b32_e32 v137, 0\t\t\t\t// 0000000000000000: BF810081
"""


def selftest() -> int:
    """Both directions, on inputs that exist NOW, plus a synthetic pair that always exists.

    A one-directional canary test is a check that cannot fail: a comparator that reports IDENTICAL for
    everything would pass "the base equals the shipped form" and fail to notice a swapped binary. So this
    witnesses (a) the same listing compared with itself -> equal, (b) a listing with ONE injected instruction
    -> different, and both again on real listings. The real pair is `/tmp/a4l1v/{base,new}/objdump.s` when
    the Gate-1 build is still on disk; otherwise `a4_l1_build.sh` rebuilds it (CPU compile only, a few
    seconds). If neither is available the selftest REFUSES rather than reporting a pass built on the
    synthetic half alone.
    """
    bad = []
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td)
        (p / "a.s").write_text(FIXTURE_A)
        (p / "b.s").write_text(FIXTURE_B)
        (p / "c.s").write_text(FIXTURE_C)
        for what, x, y, want in (
                ("synthetic identical pair", p / "a.s", p / "b.s", True),
                ("synthetic ONE injected instruction", p / "a.s", p / "c.s", False)):
            got = compare(str(x), str(y), "attn_fattn_wave", verbose=False)
            ok = got == want
            print(f"  [{'PASS' if ok else 'FAIL'}] {what}: expected"
                  f" {'IDENTICAL' if want else 'DIFFERENT'}, got"
                  f" {'IDENTICAL' if got else 'DIFFERENT'}")
            if not ok:
                bad.append(what)
        # The ambiguity the engine now creates: `attn_fattn_wave` is a substring of
        # `attn_fattn_wave_two`, so a spec that could name either must be refused rather than first-matched --
        # while the two exact spellings of each real name still resolve.
        for spec, want in (("attn_fattn_wave", "attn_fattn_wave"),
                           ("attn_fattn_wave_two", "attn_fattn_wave_two"),
                           ("fattn_wave", None)):
            funcs = functions((p / "a.s").read_text())
            try:
                got = short(resolve(spec, funcs, "fixture"))
            except SystemExit:
                got = None
            ok = got == want
            print(f"  [{'PASS' if ok else 'FAIL'}] {spec!r} -> {got}"
                  + ("" if want else " (refused: it is a substring of both kernels)"))
            if not ok:
                bad.append(f"name spec {spec!r}")
        for spec in ("attn_fattn_wave", "_ZN6engine7kernels15attn_fattn_waveE8CoreArgs",
                     "engine::kernels::attn_fattn_wave"):
            try:
                key = resolve(spec, functions((p / "a.s").read_text()), "fixture")
                ok = short(key) == "attn_fattn_wave"
            except SystemExit as exc:
                ok, key = False, str(exc)
            print(f"  [{'PASS' if ok else 'FAIL'}] {spec!r} resolves to attn_fattn_wave (got {key})")
            if not ok:
                bad.append(f"name form {spec!r}")

    base = pathlib.Path("/tmp/a4l1v/base/objdump.s")
    new = pathlib.Path("/tmp/a4l1v/new/objdump.s")
    if not (base.exists() and new.exists()):
        print("  real pair not on disk; running a4_l1_build.sh (compile only, no device)")
        proc = subprocess.run(["bash", str(REPO / "prototype/attn_percost/a4_l1_build.sh")],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not base.exists():
            raise SystemExit("REFUSE: no real listing pair and a4_l1_build.sh could not make one: "
                             + (proc.stderr or proc.stdout)[-300:])
    for what, x, y, want in (
            ("real: base vs itself", base, base, True),
            ("real: base vs the two-accumulator build", base, new, False)):
        got = compare(str(x), str(y), "attn_fattn_wave", verbose=False)
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}: expected"
              f" {'IDENTICAL' if want else 'DIFFERENT'}, got {'IDENTICAL' if got else 'DIFFERENT'}")
        if not ok:
            bad.append(what)
    print("SELFTEST PASS: the canary says identical when the streams agree and different when they do not"
          if not bad else f"SELFTEST FAIL: {bad}")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--kernel", default="attn_fattn_wave")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.a or not a.b:
        raise SystemExit("usage: a4_l1_canary.py LISTING_A LISTING_B [--kernel NAME] | --selftest")
    ok = compare(a.a, a.b, a.kernel)
    if not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
