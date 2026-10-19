#!/usr/bin/env python3
"""ceiling_proof.py — CPU witness for a lane-ceiling raise (`kAttnMaxSplit`), currently 9 -> 64.

Usage:  .venv/bin/python tools/attn_ceiling_proof.py [repo-root]     (default root = cwd)

WHY THIS EXISTS. The admission for raising `kAttnMaxSplit` is that every plan the OLD ceiling already
admitted must come out BIT-IDENTICAL, because the raise only enlarges buffers (the first newly admitted
plan, `wide4096f` at 4104 lanes, is new ground and cannot witness it). The small device gates can only
show that some rows are unchanged; the general claim is cheaper and stronger on the CPU, as a statement
about which quantities the constant is allowed to reach. Nothing here touches the GPU and no timing is
taken. The constants below are the raise THIS revision witnesses (CAP_OLD 9 = the shipped 4,608-lane
ceiling, CAP_NEW 64 = 32,768 lanes); retargeting them is the whole of what moves between raises.

WHAT IS CHECKED. Constants are READ out of the files that enforce them, with `engine/gate.py`'s own
two regexes, and each check is paired with an injection that must make it go red — a check that cannot
fail is decoration (AGENTS.md, three times over).

  [A] READERSHIP   every `kAttnMaxSplit` occurrence, classified; none may sit on a line that also
                   mentions `a.part` — no address expression may read the cap.
  [B1] ONE ADDRESS  the three writers (`attn_core`, `attn_tile_body`, `attn_slim_wave`) and the CPU
                   mirror's `part_offset` must parse to the SAME expression, and `attn_combine`'s
                   base must equal them at window 0 with a `(DH + 2)` stride. results.md §20.1's
                   "one address expression, shared" turned from a quoted claim into a check.
  [B2] WRITE=READ   for a streaming plan the union of the writer slot offsets equals the reader's
                   offset set. Must-catch: §20.1's group-collapsed writer (head-in-group).
  [B3] CAP-FREE     the PARSED writer expression, evaluated with CAP_OLD and CAP_NEW, gives identical
                   offsets for every plan the old ceiling admitted. Must-catch: the same text with
                   `* kAttnMaxSplit` injected, which makes the two caps differ.
  [C] BOUNDARY     legality is monotone in the cap, `split` never changes, the newly admitted band is
                   exactly MAXSC*CAP_OLD+1 .. MAXSC*CAP_NEW, and MAXSC*CAP_NEW passes while
                   MAXSC*CAP_NEW+1 is refused.
  [D] SIZING       `attn_part_floats(kMaxWaveRows)` at CAP_OLD, CAP_NEW and the unfunded CAP_TARGET.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path.cwd()).resolve()
KERNEL = ROOT / "prototype" / "attn_decode" / "attn_decode_kernels.h"
SLIM = ROOT / "engine" / "attn_slim_kernels.h"
LAUNCHER = ROOT / "engine" / "kernels_attn.cpp"
MIRROR = ROOT / "prototype" / "attn_tile_model.py"
N_Q, GQA, DH = 8, 4, 256
SLOT = (DH + 2) * 4                       # bytes per (max, sum, acc[DH]) F32 slot
WAVE_ROWS = 512                           # `kMaxWaveRows` — a different knob, NOT touched by this
CAP_OLD, CAP_NEW, CAP_TARGET = 9, 64, 128
CHECKPOINT_B = 1_545_700_000              # AGENTS.md budget table: 1.5457 GB bf16 checkpoint


def src(p):
    return p.read_text().splitlines()


def find(path, needle, want=1):
    hits = [(i, ln.strip()) for i, ln in enumerate(src(path), 1) if needle in ln]
    if len(hits) != want:
        raise SystemExit(f"proof refuses to guess: `{needle}` matched {len(hits)} lines in "
                         f"{path.name}, expected {want} — a check whose input can silently fail to "
                         f"arrive cannot adjudicate anything")
    return hits[0]


def normalise(expr):
    """C++ address arithmetic -> the same arithmetic in Python. One spelling of `row` (`qrow` and
    `row` are the same quantity at the three sites) and of the launch's `split`."""
    e = expr.replace("(size_t)", "").replace("static_cast<size_t>", "")
    return e.replace("a.split", "split").replace("blockIdx.y", "window").replace("qrow", "row")


def compile_expr(expr):
    return compile(ast.parse(normalise(expr), mode="eval"), "<src>", "eval")


def ast_of(expr):
    return ast.dump(ast.parse(normalise(expr), mode="eval").body)


def make_eval(expr):
    code = compile_expr(expr)

    def ev(row, head, window, split, cap):
        return eval(code, {}, {"row": row, "rows": row, "head": head, "window": window,
                               "split": split, "cap": cap, "kAttnMaxSplit": cap, "N_Q": N_Q,
                               "DH": DH, "GQA": GQA})
    return ev


def main():
    bad = []
    staging = int(re.search(r"#define\s+MAXSC\s+(\d+)", KERNEL.read_text()).group(1))
    cap_now = int(re.search(r"constexpr int kAttnMaxSplit\s*=\s*(\d+)", LAUNCHER.read_text()).group(1))
    print(f"[0] constants as enforced: MAXSC = {staging} ({KERNEL.name}), kAttnMaxSplit = {cap_now} "
          f"({LAUNCHER.name}) -> lane ceiling {staging * cap_now} lanes")
    if cap_now != CAP_NEW:
        bad.append(f"kAttnMaxSplit reads {cap_now}, this proof was written for {CAP_NEW}")

    # ---- [A] who is allowed to read the constant ------------------------------------------------
    print("\n[A] every `kAttnMaxSplit` occurrence in engine/ (the cap's whole readership):")
    seen, unclassified, addr_reads = {}, [], 0
    for path in sorted(p for p in (ROOT / "engine").iterdir() if p.suffix in (".cpp", ".h")):
        for i, line in enumerate(src(path), 1):
            if "kAttnMaxSplit" not in line:
                continue
            code = line.split("//")[0].rstrip()      # a trailing comment is not a use
            s = code.strip()
            where = f"{path.relative_to(ROOT)}:{i}"
            if "kAttnMaxSplit" not in code or s.startswith(('"', "*", "//")):
                kind = "comment"                     # prose, or a string literal being concatenated
            elif "std::to_string(" in s:
                kind = "message"                     # error text, not behaviour
            elif "constexpr int kAttnMaxSplit" in s:
                kind = "declaration"
            elif "return" in s and "kAttnMaxSplit * (DH + 2)" in s:
                kind = "allocation"
            elif "if (split > kAttnMaxSplit)" in s:
                kind = "refusal"
            else:
                kind = "UNACCOUNTED"
                unclassified.append(where)
            seen[kind] = seen.get(kind, 0) + 1
            print(f"    {where}  [{kind}]  {line.strip()[:84]}")
            if kind in ("declaration", "allocation", "refusal", "UNACCOUNTED") and "a.part" in code:
                addr_reads += 1
    if unclassified:
        bad.append(f"CODE uses this proof cannot account for: {', '.join(unclassified)}")
    if addr_reads:
        bad.append(f"{addr_reads} functional CODE lines read the cap inside a `part` address "
                   f"expression")
    for kind in ("declaration", "allocation", "refusal"):
        if seen.get(kind) != 1:
            bad.append(f"expected exactly one {kind} use of the cap, saw {seen.get(kind, 0)}")
    print(f"    -> census: {', '.join(f'{k}={v}' for k, v in sorted(seen.items()))}. The cap's only "
          f"functional uses are the declaration, the allocation and the refusal; address expressions "
          f"reading it: {addr_reads} (must be 0)")

    # ---- [B1] the four sites and the mirror spell ONE address ----------------------------------
    sites = (("attn_core  (prototype/attn_decode/attn_decode_kernels.h)", KERNEL,
              "float* p = a.part + (((size_t)qrow * N_Q + head) * a.split + blockIdx.y)"),
             ("attn_tile_body  (prototype/attn_decode/attn_decode_kernels.h)", KERNEL,
              "float* pp = a.part + "),
             ("attn_slim_wave  (engine/attn_slim_kernels.h)", SLIM, "float* pp = a.part + "))
    writers = {}
    print("\n[B1] the ONE address expression, quoted from each site:")
    for label, path, needle in sites:
        ln, text = find(path, needle)
        writers[label] = (ln, text.split("= a.part + ", 1)[1].rstrip(";"))
        print(f"    {label}:{ln}\n        {text}")
    rd_ln, rd_text = find(KERNEL, "const float* p = a.part + ")
    rd_expr = rd_text.split("= a.part + ", 1)[1].rstrip(";")
    mir_ln, mir_text = find(MIRROR, "return ((row * N_Q + head) * split + window) * (DH + 2)")
    mir_expr = mir_text.split("return ", 1)[1].rstrip(";")
    print(f"    attn_combine  (reader, prototype/attn_decode/attn_decode_kernels.h:{rd_ln})\n        {rd_text}")
    print(f"    attn_tile_model.part_offset  (prototype/attn_tile_model.py:{mir_ln})\n        {mir_expr}")
    base = ast_of(writers["attn_core  (prototype/attn_decode/attn_decode_kernels.h)"][1])
    for label, (ln, e) in list(writers.items())[1:] + [("attn_tile_model.part_offset (mirror)", (mir_ln, mir_expr))]:
        if ast_of(e) != base:
            bad.append(f"{label}:{ln} spells a different partial address from attn_core")
    rd = make_eval(rd_expr)
    core_expr = writers["attn_core  (prototype/attn_decode/attn_decode_kernels.h)"][1]
    core = make_eval(core_expr)
    if any(rd(r, h, 0, s, cap_now) != core(r, h, 0, s, cap_now)
           for r in range(4) for h in range(N_Q) for s in (2, 5, 9)):
        bad.append("attn_combine's base differs from a writer's slot at window 0")
    stride_line = find(KERNEL, "const float ms = p[(size_t)s * (DH + 2) + 0];")[1]
    print(f"    -> all three writers + the mirror parse to ONE AST: "
          f"{'NO — FAIL' if bad else 'yes'}; reader base == writer slot at window 0 (checked over "
          f"rows 0..3 x 8 heads x splits 2/5/9), reader stride per block from:\n        "
          f"{stride_line}")

    def writer_set(rows, split, ev):
        return {ev(r, h, b, split, cap_now) * 4 for r in range(rows)
                for h in range(N_Q) for b in range(split)}

    def reader_set(rows, split):
        out = set()
        tail = (0, 4) + tuple((2 + d) * 4 for d in range(DH))     # the 258 floats of one slot
        for r in range(rows):
            for h in range(N_Q):
                b0 = rd(r, h, 0, split, cap_now) * 4
                for s in range(split):
                    out |= {b0 + s * SLOT + o for o in tail}
        return out

    tail = (0, 4) + tuple((2 + d) * 4 for d in range(DH))

    def expected_reader_set(slots):
        """What the merge MUST read if it is reading the writer's own slots: all 258 floats of every
        slot the writers can land on, and nothing else."""
        return {slot + o for slot in slots for o in tail}

    print("\n[B2] writer slot set == reader access set, streaming plans (one case per split):")
    cases = [(rows, sp) for rows in (1, 4) for sp in range(2, CAP_NEW + 1)]
    n_ok = sum(1 for rows, sp in cases
               if expected_reader_set(writer_set(rows, sp, core)) == reader_set(rows, sp))
    if n_ok != len(cases):
        bad.append(f"{len(cases) - n_ok}/{len(cases)} streaming plans have writer != reader slot sets")
    print(f"    {n_ok}/{len(cases)} plans agree (rows in (1,4) x splits 2..{CAP_NEW}; rows enter the "
          f"expression only as `row * N_Q * split`, so the argument is uniform in rows)")
    alias_line, alias_src = find(KERNEL,
                                 "float* p = a.part + (((size_t)qrow * N_Q + head) * a.split + blockIdx.y)")
    alias = make_eval(alias_src.split("= a.part + ", 1)[1].rstrip(";")
                      .replace("N_Q + head", "GQA + head % GQA"))   # §20.1's collapse
    caught = sum(1 for rows in (1, 4, 16)
                 if expected_reader_set(writer_set(rows, 3, alias)) != reader_set(rows, 3))
    print(f"    must-catch (§20.1 group-collapsed writer, head-in-group, re-injected): caught in "
          f"{caught}/3 shapes — {'OK' if caught == 3 else 'THE CHECK IS VACUOUS'}")
    if caught != 3:
        bad.append("the writer/reader set check does not catch §20.1's aliasing")
    alloc_ln, alloc_text = find(LAUNCHER, "return static_cast<size_t>(rows) * N_Q * kAttnMaxSplit * (DH + 2);")
    alloc = make_eval(alloc_text.split("return ", 1)[1].rstrip(";"))
    for rows in (1, 8, WAVE_ROWS):
        for cap in (CAP_OLD, CAP_NEW, CAP_TARGET):
            last = max(writer_set(rows, cap, core)) + SLOT   # widest plan the cap admits
            buf = alloc(rows, 0, 0, 0, cap) * 4
            if last > buf:
                bad.append(f"rows={rows} cap={cap}: last slot {last} B > buffer {buf} B")
    print(f"    containment (max slot + {SLOT} B <= `{alloc_text.strip()}`, {LAUNCHER.name}:{alloc_ln}) "
          f"at rows 1/8/{WAVE_ROWS} "
          f"x caps {CAP_OLD}/{CAP_NEW}/{CAP_TARGET}: checked, no overrun")

    print("\n[B3] cap-independence — the PARSED writer expression evaluated at cap "
          f"{CAP_OLD} and cap {CAP_NEW}:")
    moved = 0
    checked = 0
    for sp in range(1, CAP_OLD + 1):
        for lanes in (sp * staging - 1, sp * staging, sp * staging + 1):
            s2 = (lanes + staging - 1) // staging
            for rows in (1, 4, WAVE_ROWS):
                for r in range(min(rows, 3)):
                    for h in range(N_Q):
                        for b in range(s2):
                            checked += 1
                            if core(r, h, b, s2, CAP_OLD) != core(r, h, b, s2, CAP_NEW):
                                moved += 1
    print(f"    {checked} slot evaluations at the split boundaries of every split the old cap admitted: "
          f"offsets that move with the cap = {moved} (must be 0)")
    if moved:
        bad.append(f"{moved} writer offsets move with the cap")
    inject = make_eval(core_expr.replace("* (DH + 2)", "* kAttnMaxSplit * (DH + 2)"))
    diff = inject(1, 2, 1, 3, CAP_OLD) != inject(1, 2, 1, 3, CAP_NEW)
    print(f"    must-catch (`* kAttnMaxSplit` injected into the writer text): the caps then differ = "
          f"{diff} — {'OK' if diff else 'THE CHECK IS VACUOUS'}")
    if not diff:
        bad.append("cap-independence check cannot see an address that reads the cap")
    if any(core(r, h, b, s, CAP_NEW) != make_eval(mir_expr)(r, h, b, s, CAP_NEW)
           for r in range(3) for h in range(N_Q) for b in range(3) for s in (3, 9)):
        bad.append("the kernel's parsed expression and the mirror's disagree numerically")

    def split_of_n(n):
        return (n + staging - 1) // staging

    print("\n[C] plan legality (split = ceil(lanes / MAXSC); legal iff split <= cap):")
    newly = [n for n in range(1, staging * CAP_NEW + 1) if CAP_OLD < split_of_n(n) <= CAP_NEW]
    lost = [n for n in range(1, staging * CAP_OLD + 1) if split_of_n(n) > CAP_NEW]
    print(f"    newly admitted: {newly[0]}..{newly[-1]} ({len(newly)} widths), all at splits "
          f"{CAP_OLD + 1}..{CAP_NEW}")
    print(f"    legal at {CAP_OLD} and refused at {CAP_NEW}: {len(lost)} (must be 0)")
    if lost:
        bad.append("the raise made a previously legal plan illegal")
    if newly[0] != staging * CAP_OLD + 1 or newly[-1] != staging * CAP_NEW:
        bad.append(f"the newly admitted band is not exactly MAXSC*{CAP_OLD}+1 .. MAXSC*{CAP_NEW}")
    for lanes, want in ((staging * CAP_OLD, True), (staging * CAP_OLD + 1, True), (4104, True),
                        (staging * CAP_NEW, True), (staging * CAP_NEW + 1, False),
                        (staging * CAP_TARGET, False)):
        ok = split_of_n(lanes) <= CAP_NEW
        if ok != want:
            bad.append(f"boundary {lanes} lanes reads the wrong way")
        print(f"    {lanes:6d} lanes -> split {split_of_n(lanes):3d}  "
              f"{'legal   ' if ok else 'REFUSED'} at cap {CAP_NEW}   "
              f"[expected {'legal' if want else 'refused'}]")

    print("\n[D] `attn_part_floats(kMaxWaveRows = 512)` in bytes:")
    for cap, note in ((CAP_OLD, f"old ceiling, {staging} x {CAP_OLD} = {staging*CAP_OLD} lanes"),
                      (CAP_NEW, f"AS SHIPPED NOW, ceiling {staging} x {CAP_NEW} = "
                                f"{staging*CAP_NEW} lanes"),
                      (CAP_TARGET, f"UNFUNDED {staging*CAP_TARGET}-lane target; no pin reaches it")):
        b = WAVE_ROWS * N_Q * cap * (DH + 2) * 4
        print(f"    split {cap:2d}: 512*8*{cap}*258*4 B = {b:>12,} B = {b/1e6:7.2f} MB = "
              f"{b/1048576:7.2f} MiB = {100*b/CHECKPOINT_B:5.2f} % of checkpoint   {note}")
    print(f"    cost of the raise itself: "
          f"{WAVE_ROWS*N_Q*CAP_NEW*(DH+2)*4 - WAVE_ROWS*N_Q*CAP_OLD*(DH+2)*4:,} B")

    print("\n" + ("PROOF FAIL — " + "; ".join(bad) if bad else
                  "PROOF PASS: the cap reaches only the allocation and the refusal; the three writers "
                  "and the mirror share one address and the reader matches it; every plan the old "
                  "ceiling admitted keeps its split, its grid and its byte offsets."))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
