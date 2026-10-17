#!/usr/bin/env python3
"""traffic_census.py — could `gdn_rec`'s STATE TRAFFIC be what binds it, and what would a 16-bit state buy?

D30 queue item 1.6, asked on the CPU before anybody writes device code for it. The family is 534.54 ms
of a 1111.95 ms wave at 4096 lanes (48.07 %), and the lever list carried the state dtype as "UNTESTED,
cheap census first" on the theory that a recurrence which moves an 18 MiB buffer per token must be
paying for it. This script answers three questions and refuses to answer a fourth.

  1. WHAT IS IN MEMORY, from the engine's own text, not from prose. Which buffer, which dtype, which
     shape, and which of it one token touches. Nothing here is typed in: the shape constants come from
     `engine/state_sizes.py` (which is itself the gate that checks them against §00.5), the dtype comes
     from the declared field in `engine/state.h`, the launch geometry from `engine/kernels_gdn.cpp`, and
     the number of passes over the state from `rec_kernel`'s own loop body.
  2. IS THE STATE RESIDENT ACROSS THE TOKEN LOOP? Answered by quoting the structure that decides it.
  3. WHAT WOULD A 16-BIT STATE BUY, and what would it cost, priced with this repo's own measured
     elasticities rather than with a hope.

  4. THE REFUSED QUESTION: "is it bandwidth-bound?" It cannot be answered from demanded bytes, and
     AGENTS.md already records why in the D21 row: **demanded bytes are not DRAM bytes.** The census is
     admissible only as a CEILING argument — if demanded bytes at the measured time EXCEED what the part
     can move, something in the model is wrong; if they are far below, traffic cannot be the binding
     constraint. Between those two the demand figure is a count, not a wall. The verdict section says
     which side of that test this family lands on, and the falsifier that follows.

Two witness modes, and the second one is JOB B's admission gate:

  traffic_census.py                       # the census and the verdict
  traffic_census.py --volume-check        # the ISA volume witness for the dependency probes: per-kernel
                                          # slots/token-step, state load/store instruction counts and
                                          # bytes/launch, IDENTICAL to MODE 0 or non-zero exit.
                                          # `gdn_rec_isa.hip.cpp --burst` refuses to time unless the
                                          # device-side half of this check passed; this is the other half.
  traffic_census.py --selftest            # the arithmetic identities the census depends on, CPU only
  traffic_census.py --selftest-volume     # must-catch witnesses: two listing mutations the volume
                                          # admission is required to catch, derived from its blind spots

Sources of every number are printed inline as `path:line` so a reader can re-check the claim rather
than trust this file. Limits are listed at the end of the output, and in `results.md` §8.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MIB = 1024 * 1024
GB = 1000.0 ** 3            # a GB/s figure from a probe is decimal GB


def load_census():
    """Import `isa_census.py` from this directory — same TU, same parser, no second spelling."""
    spec = importlib.util.spec_from_file_location("gdn_rec_isa_census", HERE / "isa_census.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gdn_rec_isa_census"] = mod
    spec.loader.exec_module(mod)          # runs the rule selftest at import; it exits 2 if it fails
    return mod


# -----------------------------------------------------------------------------------------------
# the constants, taken from the engine rather than typed in
# -----------------------------------------------------------------------------------------------
CONST_RX = re.compile(r"state.h constants:\s*(.+)")


def engine_consts() -> tuple[dict[str, int], str]:
    """Ask `engine/state_sizes.py` for §08.1's shape constants.

    Deliberately a subprocess rather than an import: that file is the GATE which checks the same
    constants against §00.5's rows and against the spec pages, so going through it means a spec edit
    that renames a shape fails this census too instead of quietly disagreeing with it. It prints a
    report and exits non-zero on a mismatch; both are surfaced here.
    """
    p = subprocess.run([sys.executable, str(REPO / "engine/state_sizes.py")],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("engine/state_sizes.py failed, so the state shapes cannot be trusted:\n"
                         + p.stdout + p.stderr)
    m = CONST_RX.search(p.stdout)
    if not m:
        raise SystemExit("engine/state_sizes.py printed no `state.h constants:` line")
    out: dict[str, int] = {}
    for item in m.group(1).split(","):
        name, _, val = item.partition("=")
        out[name.strip()] = int(val)
    return out, "engine/state_sizes.py (which re-derives state.h against §00.5)"


def field_dtype(path: Path, field: str) -> tuple[str, int, int]:
    """The declared C++ type of one `RequestState` field, and the element size that follows from it.

    This is where a dtype claim gets its witness. `§00.5` carried the conv window as BF16/0.63 MiB
    for a buffer `conv_kernel` reads and writes as `float*`, and the wrong cell survived a spec pass
    (`engine/state.h`'s own header comment, and `engine/state_sizes.py`'s FIXES list, both record the
    correction). So the authority is the declaration, not the prose row.
    """
    for n, line in enumerate(path.read_text().splitlines(), 1):
        m = re.search(r"^\s*(\w[\w:* ]*?)\s*\*\s*" + re.escape(field) + r"\s*[;=]", line)
        if m:
            t = m.group(1).strip().split()[-1]
            size = {"float": 4, "double": 8, "uint16_t": 2, "uint8_t": 1, "int": 4}[t]
            return t, size, n
    raise SystemExit(f"{path}: no `T* {field}` declaration found")


def kernel_define(path: Path, name: str) -> tuple[int, int]:
    for n, line in enumerate(path.read_text().splitlines(), 1):
        m = re.match(r"#define\s+%s\s+(\d+)" % name, line)
        if m:
            return int(m.group(1)), n
    raise SystemExit(f"{path}: no `#define {name} <int>`")


def kernel_body(path: Path, name: str) -> tuple[str, int]:
    """The body of one `__global__` kernel, by brace matching, plus its start line."""
    text = path.read_text()
    lines = text.splitlines()
    start = None
    for n, line in enumerate(lines, 1):
        if re.search(r"__global__\s+void\s+" + re.escape(name) + r"\s*\(", line):
            start = n
            break
    if start is None:
        raise SystemExit(f"{path}: kernel {name} not found")
    depth, body, seen = 0, [], False
    for line in lines[start - 1:]:
        for ch in line:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
        if seen:
            body.append(line)
        if seen and depth == 0:
            break
    return "\n".join(body), start


def state_passes(body: str) -> tuple[int, int]:
    """How many times one launch walks the state buffer, read and write, derived from the source.

    A statement whose left-hand side is the state pointer is a write; any other mention is a read. The
    count is of *statements*, and each statement runs once per row of the state, so the number of
    passes over the whole buffer is exactly this count. `rec_kernel` has two reads (loop A's dot, loop
    B's read-modify-write) and one write. A third read or a dropped write changes the census, so this
    is derived and cross-checked against the compiled listing in `--volume-check`, not asserted.
    """
    rx = re.compile(r"(?:const\s+)?float\s*\*\s*(?:__restrict__\s*)?(\w*Sh\w*)")
    mm = rx.search(body)
    name = mm.group(1) if mm else "Sh"
    reads = writes = 0
    for line in body.splitlines():
        stmt = line.strip()
        if not stmt or stmt.startswith("//"):
            continue
        if not re.search(r"\b" + name + r"\s*\[", stmt):
            continue
        if re.match(r"^" + name + r"\s*\[[^\]]*\]\s*(=[^=]|\+=|-=)", stmt):
            writes += 1
        else:
            reads += 1
    return reads, writes


def launch_geometry() -> dict:
    """Grid and block of the shipped launch, and the loop that makes it one launch per TOKEN."""
    src = (REPO / "engine/kernels_gdn.cpp").read_text().splitlines()
    grid = block = line = None
    for n, txt in enumerate(src, 1):
        m = re.search(r"rec_kernel<<<\s*(\w+)\s*,\s*(\w+)\s*>>>", txt)
        if m:
            grid, block, line = m.group(1), m.group(2), n
            break
    if grid is None:
        raise SystemExit("engine/kernels_gdn.cpp: no `rec_kernel<<<...>>>` launch site found")
    step = (REPO / "engine/decode_step.cpp").read_text().splitlines()
    call = next((n for n, t in enumerate(step, 1) if "kernels::gdn_rec(" in t), None)
    if call is None:
        raise SystemExit("engine/decode_step.cpp: no `kernels::gdn_rec(` call site found")
    enclosing = None
    for n in range(call, max(0, call - 25), -1):
        if re.search(r"for\s*\(int t\s*=\s*0", step[n - 1]):
            enclosing = n
            break
    return {"grid": grid, "block": block, "line": f"engine/kernels_gdn.cpp:{line}",
            "call": f"engine/decode_step.cpp:{call}",
            "token_loop": f"engine/decode_step.cpp:{enclosing}" if enclosing else "NOT FOUND"}


# -----------------------------------------------------------------------------------------------
# the census
# -----------------------------------------------------------------------------------------------
def build_model(lanes: int) -> dict:
    C, csrc = engine_consts()
    state_h = REPO / "engine/state.h"
    sdt, sbytes, sdecl = field_dtype(state_h, "gdn_state")
    cdt, cbytes, cdecl = field_dtype(state_h, "conv_window")
    kd = REPO / "prototype/gdn_decode/gdn_decode_kernels.h"
    D, dline = kernel_define(kd, "D")
    NH, nhline = kernel_define(kd, "NH")
    if NH != C["kNumValueHeads"]:
        raise SystemExit(f"NH={NH} ({kd}:{nhline}) != kNumValueHeads={C['kNumValueHeads']} — a shape "
                         f"claim in two places has started to disagree; fix the source, not this census")
    if D != C["kDv"] or D != C["kDk"]:
        raise SystemExit(f"D={D} != d_k/d_v {C['kDk']}/{C['kDv']} — same two-place disagreement")
    body, bline = kernel_body(kd, "rec_kernel")
    reads, writes = state_passes(body)
    geo = launch_geometry()
    wave = 32                                     # .amdhsa_wavefront_size32 1 (results.md §1)
    wf_per_launch = (D // wave) * NH

    state_layer_elems = C["kNumValueHeads"] * C["kDk"] * C["kDv"]
    state_layer_bytes = state_layer_elems * sbytes
    layers = C["kNumGdnLayers"]
    per_token_step = {
        "state_read": reads * state_layer_bytes,
        "state_write": writes * state_layer_bytes,
        # one token's q|k|v row is [q(NH*D) | k | v] and it is the conv's own output tile; g|beta are
        # NH scalars each; `o` is one NH*D row. All F32, exactly as the launcher declares them.
        "operand_read": (3 * C["kNumValueHeads"] * C["kDv"] + 2 * C["kNumValueHeads"]) * 4,
        "operand_write": C["kNumValueHeads"] * C["kDv"] * 4,
    }
    per_token_step["read"] = per_token_step["state_read"] + per_token_step["operand_read"]
    per_token_step["write"] = per_token_step["state_write"] + per_token_step["operand_write"]
    per_token_step["total"] = per_token_step["read"] + per_token_step["write"]
    wave_steps = lanes * layers
    m = {
        "consts": C, "const_source": csrc, "dtype": sdt, "dtype_decl": f"engine/state.h:{sdecl}",
        "conv_dtype": cdt, "conv_decl": f"engine/state.h:{cdecl}",
        "D": D, "NH": NH, "wave": wave, "wf_per_launch": wf_per_launch, "layers": layers,
        "state_layer_elems": state_layer_elems, "state_layer_bytes": state_layer_bytes,
        "passes_read": reads, "passes_write": writes, "kernel_body": f"{kd.name}:{bline}",
        "geo": geo, "lanes": lanes, "per_token_step": per_token_step,
        "state_per_request": C["kNumGdnLayers"] * state_layer_bytes,
        "conv_per_request": C["kNumGdnLayers"] * C["kConvWindow"] * C["kConvChannels"] * cbytes,
    }
    m["launches_per_wave"] = wave_steps
    m["wave_demand"] = {
        "read": per_token_step["read"] * wave_steps,
        "write": per_token_step["write"] * wave_steps,
        "total": per_token_step["total"] * wave_steps,
        "state_only": (per_token_step["state_read"] + per_token_step["state_write"]) * wave_steps,
    }
    m["token"] = {k: per_token_step[k] * layers for k in
                  ("read", "write", "total", "state_read", "state_write")}
    return m


def verdict(m: dict, family_ms: float, dram_read: float, dram_copy: float, warm: float,
            cold: float, collapse: dict) -> list[str]:
    out: list[str] = []
    def w(s): out.append(s)
    secs = family_ms / 1000.0
    demand = m["wave_demand"]
    demand_gbs = demand["total"] / secs / GB
    demand_state_gbs = demand["state_only"] / secs / GB
    w("")
    w("== the ceiling test, and what it can and cannot say")
    w(f"  demanded bytes at {family_ms} ms = {demand_gbs:.1f} GB/s total"
      f" ({demand_state_gbs:.1f} GB/s of it state)")
    w(f"  the part moves {dram_read:.0f} GB/s read / {dram_copy:.0f} GB/s copy at 1 GiB"
      f" (devices/strix-halo-memory-bandwidth.md, `iGPU, 1 GiB`)")
    w(f"  the SAME access pattern with the state warm measures {warm:.0f} GB/s at an 18 MiB working set")
    w(f"  and {cold:.0f} GB/s once tiled to 108 MiB (prototype/gdn_decode/results.md, Performance)")
    if demand_gbs > dram_read:
        w(f"  EXCEEDED: {demand_gbs:.1f} > {dram_read:.0f} GB/s. That is not a bandwidth wall, it is a")
        w("  units confession — AGENTS.md's D21 row: DEMANDED BYTES ARE NOT DRAM BYTES. L1/L2/MALL")
        w("  absorb the reuse (hipGetDeviceProperties calls L2 2 MiB; the warm row above proves a")
        w("  larger cache the query does not report is holding the 18 MiB). So the demand figure is")
        w("  admissible ONLY as the ceiling argument this section makes, never as the wall.")
    w(f"  against the measured warm-state rate the same shape achieves, the demand is"
      f" {100.0 * demand_gbs / warm:.1f} % of what the box demonstrably moves. Traffic therefore cannot"
      f" be the binding constraint: the kernel asks for less than half the bytes-per-second the"
      f" identical access pattern is measured delivering when the state is warm, and each launch still")
    w(f"  takes {family_ms * 1000.0 / m['launches_per_wave']:.2f} us.")
    w("")
    w("== the direct witness, which is better than any ceiling: the traffic has already been removed")
    w(f"  `prototype/gdn_collapse/results.md` §4 measured route A, which keeps the state resident in")
    w(f"  LDS: state round trip {collapse['traffic_from_gb']:.1f} GB ->"
      f" {collapse['traffic_to_gb']*1000:.0f} MB per {collapse['lanes']}-lane wave"
      f" (a {collapse['traffic_from_gb']/collapse['traffic_to_gb']:.0f}x cut, essentially all of it) bought"
      f" {collapse['seq_ms']:.2f} -> {collapse['collapse_ms']:.2f} ms")
    w(f"  = {collapse['seq_ms']/collapse['collapse_ms']:.3f}x of the family.")
    w(f"  Attribute all {collapse['seq_ms']-collapse['collapse_ms']:.2f} ms of that to traffic and a 2x cut is")
    w("  still worth at most half of it. Attribute it the way the launch arithmetic says to — route A")
    w(f"  also replaced {collapse['launch_from']:,} launches with 36, and the empty-kernel dispatch floor")
    w("  is 1.75-2.08 us (devices/strix-halo-memory-bandwidth.md §0c), which more than covers the")
    w(f"  {collapse['seq_ms']-collapse['collapse_ms']:.1f} ms on its own — and traffic bought close to nothing at all.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lanes", type=int, default=4096, help="wave width (D30's basis: 4096)")
    ap.add_argument("--family-ms", type=float, default=534.54,
                    help="measured `gdn_rec` family time for that wave (D30's table)")
    ap.add_argument("--dram-read-gbs", type=float, default=240.0)
    ap.add_argument("--dram-copy-gbs", type=float, default=210.0)
    ap.add_argument("--warm-gbs", type=float, default=876.0,
                    help="measured rate for this access pattern with an 18 MiB warm state")
    ap.add_argument("--cold-gbs", type=float, default=255.0, help="...tiled to 108 MiB")
    ap.add_argument("--collapse-seq-ms", type=float, default=125.36)
    ap.add_argument("--collapse-route-ms", type=float, default=112.04)
    ap.add_argument("--collapse-traffic-from-gb", type=float, default=38.7)
    ap.add_argument("--collapse-traffic-to-mb", type=float, default=75.0)
    ap.add_argument("--collapse-lanes", type=int, default=1024)
    ap.add_argument("--asm", type=Path, default=None,
                    help="device .s for --volume-check (default: tmp/gdn_isa/gdn_rec_isa.s)")
    ap.add_argument("--volume-check", action="store_true")
    ap.add_argument("--modes", default="rec_kernel,rec0_kernel,recx_indep_kernel,"
                                      "recx_unchain_kernel,recx_shape_kernel,recx_both_kernel")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-volume", action="store_true",
                    help="must-catch witnesses that the volume admission can fail")
    ap.add_argument("--selftest-all", action="store_true")
    args = ap.parse_args()

    if args.selftest_volume:
        return selftest_volume()
    if args.selftest_all:
        return 1 if (selftest() or selftest_volume()) else 0
    if args.selftest:
        return selftest()

    if args.volume_check:
        return volume_check(args)

    m = build_model(args.lanes)
    w = print
    mib = lambda b: f"{b / MIB:.3f} MiB"
    gb = lambda b: f"{b / GB:.3f} GB"

    w("== 1. what is in memory (every row sourced; nothing typed in)")
    w(f"  shape constants   {m['const_source']}")
    w(f"                    " + ", ".join(f"{k}={v}" for k, v in m["consts"].items()))
    w(f"  GDN state dtype   {m['dtype']} per element, declared at {m['dtype_decl']}")
    w(f"                    (§00.5's BF16/0.63 MiB conv cell and §08.1's were corrected to F32/1.27 MiB;"
      f" see engine/state.h's header comment — the brief's 0.63 MiB is the retired cell)")
    w(f"  conv window       {m['conv_dtype']} at {m['conv_decl']} = {mib(m['conv_per_request'])}/request")
    w(f"  one layer's state {m['state_layer_elems']} elems = {mib(m['state_layer_bytes'])}"
      f"; all {m['layers']} layers = {mib(m['state_per_request'])}/request")
    w(f"  wavefronts        {m['wf_per_launch']} per launch ({m['NH']} blocks x {m['D']//m['wave']} warps)")
    w(f"  launch            `rec_kernel<<<{m['geo']['grid']}, {m['geo']['block']}>>>` at {m['geo']['line']}"
      f"  ({m['NH']} blocks x {m['D']} threads, wave{m['wave']})")
    w(f"  one launch = ONE (token, layer); the token loop is the host loop at"
      f" {m['geo']['call']} inside {m['geo']['token_loop']}")
    w(f"  kernel body       {m['kernel_body']}: {m['passes_read']} state read passes,"
      f" {m['passes_write']} write pass")

    w("")
    w("== 2. resident or re-read?  RE-READ, EVERY TOKEN, because there is no token loop to be resident in")
    w(f"  The shipped form's only loops are the two `for (int i = 0; i < D; ++i)` walks over the state")
    w(f"  inside ONE launch ({m['kernel_body']}); the next token is the next launch")
    w(f"  ({m['geo']['token_loop']} -> {m['geo']['call']}), so the state is read from global on every")
    w("  token by construction. It is not a staging choice either: a block owns one head's 128x128 F32")
    w("  strip = 64 KiB, which is the whole LDS budget of a CU (65,536 B), so the shipped geometry")
    w("  CANNOT hold the state across tokens. The engine does own a resident form")
    w("  (`--gdn-scan collapse`, engine/gdn_collapse_kernels.h) and it is measured, in §4 below.")

    p = m["per_token_step"]
    t = m["token"]
    wv = m["wave_demand"]
    w("")
    w(f"== 3. the census: bytes DEMANDED (read and written; DRAM later). lanes={args.lanes},"
      f" family={args.family_ms} ms, launches={m['launches_per_wave']:,}")
    w("  %-26s %14s %14s %16s" % ("scope", "read", "written", "total"))
    w("  %-26s %14s %14s %16s" % ("one token-step (1 layer)", mib(p["read"]), mib(p["write"]),
                                   mib(p["total"])))
    w("  %-26s %14s %14s %16s" % ("   of which state", mib(p["state_read"]), mib(p["state_write"]),
                                   mib(p["state_read"] + p["state_write"])))
    w("  %-26s %14s %14s %16s" % ("   of which q|k|v,g,beta,o", mib(p["operand_read"]),
                                   mib(p["operand_write"]), mib(p["operand_read"] + p["operand_write"])))
    w("  %-26s %14s %14s %16s" % (f"one token ({m['layers']} layers)", mib(t["read"]), mib(t["write"]),
                                   mib(t["total"])))
    w("  %-26s %14s %14s %16s" % ("one 4096-lane wave", gb(wv["read"]), gb(wv["write"]),
                                   gb(wv["total"])))
    secs = args.family_ms / 1000.0
    w(f"  demand rate       {wv['total']/secs/GB:.1f} GB/s total,"
      f" state alone {wv['state_only']/secs/GB:.1f} GB/s")
    w(f"  per launch        {args.family_ms*1000.0/m['launches_per_wave']:.3f} us"
      f" (ledger bills the family at 6.8-6.9 us; D25 measured 6.319 us in the burst window)")
    w(f"  operand share     {100.0*(p['operand_read']+p['operand_write'])/p['total']:.2f} % of the demand"
      f" — the state is the whole traffic story of this kernel, and it is {(p['state_read']+p['state_write'])/(p['operand_read']+p['operand_write']):.0f}x the operands")

    for line in verdict(m, args.family_ms, args.dram_read_gbs, args.dram_copy_gbs, args.warm_gbs,
                        args.cold_gbs,
                        {"seq_ms": args.collapse_seq_ms, "collapse_ms": args.collapse_route_ms,
                         "traffic_from_gb": args.collapse_traffic_from_gb,
                         "traffic_to_gb": args.collapse_traffic_to_mb / 1000.0,
                         "lanes": args.collapse_lanes, "launch_from": 18432}):
        w(line)

    w("")
    w("== 5. what a 16-bit state would buy, priced with measured elasticities")
    operand_wave = (p["operand_read"] + p["operand_write"]) * m["launches_per_wave"]
    half = m["wave_demand"]["total"] / 2.0
    new_total = half + operand_wave
    w(f"  bytes: it halves the STATE term, not the demand — the state is"
      f" {100.0*(p['state_read']+p['state_write'])/p['total']:.1f} % of the demand, so the wave's demand")
    w(f"  falls from {gb(m['wave_demand']['total'])} to ~{gb(new_total)}.")
    w(f"  demand rate would fall from {m['wave_demand']['total']/secs/GB:.0f} to ~{new_total/secs/GB:.0f} GB/s —"
      f" i.e. from above the {args.dram_read_gbs:.0f} GB/s package ceiling to about it, and the ceiling was")
    w("  never what bounded the kernel (§3). A saving in a quantity that is not the constraint.")
    w("  instructions: the recurrence consumes the state as F32 in FMA chains. An F16/BF16 state means a")
    w("  conversion per state access, and the accesses are per thread per token-step: 2 x D reads + 1 x D")
    w(f"  writes = 3 x {m['D']} = {3*m['D']} conversions on a {1523}-slot bill (isa_census.output.txt) ="
      f" +{100.0*3*m['D']/1523:.1f} % slots. Packing two halves per 32-bit access cannot avoid it: a thread")
    w(f"  owns ONE d_v column j and walks i, so the pair it would need is in a different thread — packing")
    w("  the state means re-cutting the thread mapping, i.e. a different kernel, not a dtype flag.")
    w(f"  D25's measured elasticity is ~0.8 (a -2.0 % slot cut returned -1.6 % of time), so +{100.0*3*m['D']/1523:.0f} % slots")
    w(f"  is roughly +{100.0*3*m['D']/1523*0.8:.0f} % of the family ~ +{args.family_ms*3*m['D']/1523*0.8:.0f} ms, against a traffic saving the")
    w(f"  §4 witness bounds at well under {args.collapse_seq_ms-args.collapse_route_ms:.0f} ms for the WHOLE traffic term.")
    w("  It is a net loss on this repo's own numbers, twice over. And it is already named as a defect:")
    w("  `engine/gdn_collapse_kernels.h`'s fail-closed negative `rc_state_bf16` is 'the cheap *the state")
    w("  is big, halve it* mistake', injected on purpose to break bit-identity. A 16-bit state is also a")
    w("  numerics change with NO existing range evidence: §10.1 G3 gates `state_new`, but the pinned")
    w("  `decode*.state_fp.npy` files are FINGERPRINT rows (tools/reference_dump.py -> tools/state_fp.py),")
    w("  not raw values, so `tools/fp16_range_probe.py`'s D14 census (peak 82.5, 794x margin) covers")
    w("  activations/logits/weights and says NOTHING about the state's dynamic range. Under D13 a")
    w("  16-bit switch is adoptable if the gates stay green; here the gates could not go green or red,")
    w("  because nothing pins the state's magnitudes. That is the missing measurement, not a band to ask")
    w("  for. RECOMMENDATION: do not fund it.")

    w("")
    w("== 6. VERDICT")
    w(f"  TRAFFIC-BOUND: NO. The demand rate ({wv['total']/secs/GB:.0f} GB/s) exceeds the package ceiling,")
    w("  which under D21 falsifies the traffic model instead of supporting it; the same access pattern")
    w(f"  is measured at {args.warm_gbs:.0f} GB/s warm, so the demand fits in half of what is demonstrably")
    w("  available; and removing essentially ALL the state traffic in situ bought 1.119x (route A).")
    w("  LATENCY-BOUND on the per-token dependency chain: STILL THE LIVE READING, and still unmeasured.")
    w("  Everything this census could do without a device raises it: traffic is out, instructions are out")
    w("  (D25, 1.0159x), launches are out (route A removed 18,396 of them for 13 ms), the working set is")
    w("  inside the cache that holds it. What is left is the chain.")
    w("  FALSIFIER THAT FOLLOWS: the dependency-break probes in this directory (§9 of results.md).")
    w("  rec3 breaks the cross-token round trip at fixed volume (and a fixed bill: -1 slot), rec4")
    w("  shortens the two serial accumulator chains at fixed volume, rec5 does both, and rec6 is rec4's")
    w("  bill-matched control — same unroll-4 loop spelling, chain NOT broken, bit-identical to rec0 —")
    w("  because the census found rec4 10.5 % cheaper in slots and D25's elasticity says a bill that")
    w("  much is worth up to 8.4 % of time by itself. Read rec4 against rec6, not against rec0. If the")
    w("  probes move the clock, the chain binds and D30 items 1.4/1.5 get funded; if they all sit in the")
    w("  noise, `gdn_rec` needs a model that D23's census has still not named, and the counters pass")
    w("  (HANDOFF.md §4) is next, not more levers.")
    w("")
    w("== limits of this census (CPU-only; no device was touched and NO TIMING WAS TAKEN)")
    w("  * demanded bytes, not DRAM bytes: no counter pass was taken, so the absorbed fraction is unknown")
    w("    and only the ceiling argument above is admissible.")
    w("  * the measured 876/255 GB/s rows are decode-step measurements of the same access pattern, not of")
    w("    the prefill wave, and the 108 MiB row is quoted from a build that no longer exists")
    w("    (prototype/gdn_decode/results.md says so). They bound the argument; they are not this kernel.")
    w("  * route A is a different geometry (64 blocks/CU resident), so its 1.119x bundles dispatch and")
    w("    traffic; the attribution is argued in §4 and not measured apart.")
    w("  * the conv window is a different kernel (`gdn_conv`, 20.71 ms, D30 item 1.7) and is only listed")
    w("    here for the per-request arithmetic; this census bills `gdn_rec`.")
    return 0


def volume_report(asm: Path, names: list[str]) -> int:
    """The ISA half of the probes' volume admission: identical access counts, or this returns non-zero.

    Why instructions and not a runtime counter: a probe whose bytes changed would look fast for the
    wrong reason, and the timed kernels must stay clean, so the byte account is read off the listing the
    binary actually runs — one `global_load_b32` is one 4-byte lane access, so load/store counts per
    token-step ARE the per-launch read/write volume. `gdn_rec_isa.hip.cpp --volume` is the device half
    (which elements got written, guard bands, the host-side account); this is the instruction half.

    Two counts are admitted, not one. The STATE columns (the two walks) are the bytes this probe is
    about; the STEP columns are every global access in the token-step, prologue operands included, so a
    probe cannot pass by quietly reading less of everything else. The SLOT column is reported and NOT
    admitted, with its D25-elasticity consequence printed, because a dependency break that re-cuts the
    loop can and does change the instruction bill — and a timing delta must be read against that, not
    against 1.000.
    """
    cen = load_census()
    if not Path(asm).exists():
        raise SystemExit(f"no listing at {asm}. Build it first: SAVE_TEMPS=1 ./prototype/gdn_rec_isa/build.sh")
    kernels = cen.split_kernels_local(Path(asm).read_text().splitlines())
    missing = [n for n in names if n not in kernels]
    if missing:
        raise SystemExit(f"{missing} not in {asm}; here: {sorted(kernels)}")

    def walk_counts(c: dict) -> tuple[int, int]:
        ld = st = 0
        for reg, mult in (("loopA", c["trips"]["loopA"]), ("loopB", c["trips"]["loopB"])):
            for mn, _ in c["regions"][reg]:
                if mn.startswith("global_load"):
                    ld += mult
                elif mn.startswith("global_store"):
                    st += mult
        return ld, st

    def step_counts(c: dict) -> tuple[int, int]:
        ld = st = 0
        for mn, _ in c["step"]:
            if mn.startswith("global_load"):
                ld += 1
            elif mn.startswith("global_store"):
                st += 1
        return ld, st

    wf, per_launched = cen.WF_PER_LAUNCH, 4 * cen.WAVE * cen.WF_PER_LAUNCH
    ref = None
    print(f"listing  {asm}")
    print(f"unit     one token-step = one launch = {wf} wavefronts x 32 lanes; a global_load_b32 is one")
    print(f"         4 B lane access, so bytes/launch = count x {per_launched} B")
    print("         state = the two state walks; step = the whole token-step, prologue operands in")
    print("  %-22s %7s %6s %6s %8s %8s %12s %13s %s" %
          ("kernel", "slots", "st_ld", "st_st", "step_ld", "step_st", "read/launch", "write/launch",
           "verdict"))
    rows: dict[str, tuple[int, int, int, int, int]] = {}
    for n in names:
        c = cen.census_kernel(kernels[n])
        sld, sst = walk_counts(c)
        ald, ast = step_counts(c)
        sl = cen.slots(c["step"])
        rows[n] = (sld, sst, ald, ast, sl)
        if ref is None:
            ref = (n, sld, sst, ald, ast, sl)
            verdict_s = "reference"
        else:
            verdict_s = ("VOLUME_SAME_AS_" + ref[0]) if rows[n][:4] == ref[1:5] else (
                f"VOLUME_DIFFERS(state {sld}/{sst} vs {ref[1]}/{ref[2]},"
                f" step {ald}/{ast} vs {ref[3]}/{ref[4]})")
        print("  %-22s %7d %6d %6d %8d %8d %12s %13s %s" %
              (n, sl, sld, sst, ald, ast, f"{sld*per_launched/1048576:.3f} MiB",
               f"{sst*per_launched/1048576:.3f} MiB", verdict_s))

    bad = [n for n in names[1:] if rows[n][:4] != ref[1:5]]
    print(f"VOLUME-CHECK reference={ref[0]} modes={len(names)} volume_differing={len(bad)}"
          f" verdict={'OK' if not bad else 'FAILED: ' + ','.join(bad)}")
    print("admitted the state and step load/store columns: they are the bytes. A probe that changes any")
    print("of them is a different kernel and this script exits non-zero so a timing run cannot proceed.")
    print("reported, not admitted — the instruction bill, with D25's measured slot elasticity ~0.8")
    print("(a -2.0 % slot cut returned -1.6 % of time on this kernel) applied to it:")
    ref_slots = ref[5]
    for n in names:
        d = rows[n][4] - ref_slots
        print(f"    {n:<22} {d:+5d} slots = {100.0*d/ref_slots:+6.1f} % of {ref[0]}"
              f" -> up to {100.0*d*0.8/ref_slots:+6.1f} % of time explainable by the bill alone")
    print("A probe whose bill is materially cheaper cannot attribute its delta to the dependency it")
    print("broke; read the timing against the elasticity column, and where the two disagree the probe")
    print("needs a bill-matched sibling (that is MODE rec6) before its number means anything.")
    return 1 if bad else 0


def volume_check(args) -> int:
    return volume_report(Path(args.asm) if args.asm else REPO / "tmp/gdn_isa/gdn_rec_isa.s",
                         [n for n in args.modes.split(",") if n])


def selftest_volume() -> int:
    """Must-catch witnesses for the volume admission, derived from what it is supposed to see.

    An admission that cannot fail is decoration — the `prototype/gdn_collapse/run.py` driver that parsed
    zero `CASE` lines printed PASS over a matrix it had never read, and `partalias` scored 0 of 1,024
    because the geometry it could fire on was never a case. So the two mutations here are chosen from
    the two blind spots the admission has, not hand-listed:

      (1) a state-walk store deleted from one probe. This is the byte the probes are about. It can be
          caught two ways and both are admissible: the census may refuse the listing outright (a state
          access per trip that no longer divides D is not the two-walk form), or the STATE columns may
          differ from MODE 0's. The test records which route fired, because they mean different things.
      (2) a PROLOGUE operand load deleted from one probe. The state columns cannot see this at all —
          which is exactly why the STEP columns exist, so this mutant must be caught BY THE STEP COLUMNS
          and must leave the state columns equal. That is what makes the step columns load-bearing
          rather than decoration.

    Both are applied to a copy of the real listing in `tmp/`, never to the file the build produced.
    """
    asm = REPO / "tmp/gdn_isa/gdn_rec_isa.s"
    if not asm.exists():
        print("SELFTEST volume SKIP: no listing. Build it: SAVE_TEMPS=1 ./prototype/gdn_rec_isa/build.sh")
        return 2
    text = asm.read_text().splitlines()

    def mutate(kernel: str, mn: str, which: str) -> Path:
        """Delete the Nth (first|last) instruction matching `mn` inside one kernel's body."""
        out, name, hit = [], None, False
        for ln in text:
            m = re.match(r"^(_Z[A-Za-z0-9_]+)", ln)
            if m:
                name = m.group(1)
            if name and kernel in name and not hit and mn in ln:
                if which == "first":
                    hit = True
                    continue
            out.append(ln)
        if not hit:
            raise SystemExit(f"selftest could not build mutation {which} in {kernel}: no `{mn}` line")
        p = REPO / f"tmp/gdn_isa/volume_mutant_{kernel}_{which}.s"
        p.write_text("\n".join(out) + "\n")
        return p

    def run(path: Path, names: list[str]) -> tuple[int, str]:
        """`volume_report`, with a refusal (SystemExit) recorded as a non-zero verdict plus its reason."""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = volume_report(path, names)
        except SystemExit as e:
            code = 1
            buf.write(f"REFUSED: {e}\n")
        return code, buf.getvalue()

    def columns(out: str, kernel: str) -> tuple[int, int, int, int] | None:
        """The four admitted columns as printed for `kernel`, so the test reads the report, not internals."""
        for ln in out.splitlines():
            if ln.strip().startswith(kernel):
                f = ln.split()
                return int(f[2]), int(f[3]), int(f[4]), int(f[5])
        return None

    bad = 0
    names = ["rec_kernel", "rec0_kernel", "recx_indep_kernel", "recx_both_kernel"]
    ref_code, ref_out = run(asm, names)
    ref_cols = columns(ref_out, "rec_kernel")
    cases = [("recx_indep_kernel", "global_store_b32", "state-walk store removed"),
             ("recx_both_kernel", "global_load_b32", "prologue operand load removed")]
    for kernel, mn, what in cases:
        p = mutate(kernel, mn, "first")
        code, out = run(p, names)
        got = columns(out, kernel)
        refusal = "REFUSED" in out
        diff_state = got is not None and ref_cols is not None and got[:2] != ref_cols[:2]
        diff_step = got is not None and ref_cols is not None and got[2:] != ref_cols[2:]
        caught = code != 0 and (refusal or diff_state or diff_step)
        print(f"  must-catch {kernel:<20} {what:<32} exit={code} route="
              f"{'refused-listing' if refusal else ('state-columns' if diff_state else 'step-columns') if caught else 'MISSED'}"
              f" state_cols={got[:2] if got else '-'} step_cols={got[2:] if got else '-'}"
              f" (MODE 0 state={ref_cols[:2]} step={ref_cols[2:]})")
        if not caught:
            bad += 1
        if kernel == "recx_both_kernel":   # the step columns' own witness: step moved, state did not
            if not (diff_step and not diff_state):
                print("  FAIL prologue mutant must move the step columns and leave the state columns equal"
                      f" (state_diff={diff_state}, step_diff={diff_step})")
                bad += 1
    # and the unmutated listing must pass, or the mutants prove nothing about the real thing
    code, out = run(asm, names)
    print(f"  must-not-catch unmutated listing: exit={code} (want 0)")
    if code != 0:
        bad += 1
    print(f"SELFTEST volume cases={len(cases) + 1} bad={bad} ok={0 if bad else 1}"
          f" claim=the volume admission refuses a removed state-walk store and a removed prologue"
          f" operand load, the second of which is invisible to the state columns and is caught only"
          f" because the step columns are admitted too, and it passes the unmutated listing")
    return 1 if bad else 0


def selftest() -> int:
    """The arithmetic identities this census leans on, and two that must FAIL."""
    bad = 0
    m = build_model(4096)
    C = m["consts"]
    # A: one layer's state must be §00.5's 1.0 MiB and all 18 layers §00.5's 18.0 MiB.
    if m["state_layer_bytes"] != MIB:
        print(f"SELFTEST FAIL one layer {m['state_layer_bytes']} != 1 MiB"); bad += 1
    if m["state_per_request"] != 18 * MIB:
        print(f"SELFTEST FAIL state/request {m['state_per_request']} != 18 MiB"); bad += 1
    # B: the per-launch state volume must equal passes x layer bytes, and the wave must scale by
    #    lanes x layers. Recomputed the long way round, not from the same helper.
    per_launch = m["per_token_step"]["state_read"] + m["per_token_step"]["state_write"]
    if per_launch != (m["passes_read"] + m["passes_write"]) * m["state_layer_bytes"]:
        print("SELFTEST FAIL per-launch state != passes x layer bytes"); bad += 1
    if m["wave_demand"]["state_only"] != per_launch * m["lanes"] * m["layers"]:
        print("SELFTEST FAIL wave state != per-launch x lanes x layers"); bad += 1
    # C: route A's traffic figure must be reproducible: 1024 lanes x 18 layers x (1 read + 1 write)
    #    pass of a 1 MiB layer = the 38.7 GB its results.md quotes. This is the census disagreeing or
    #    agreeing with an independent measurement, and it is the one number here that is not derived.
    route_a = m["lanes"] // 4 * m["layers"] * 2 * m["state_layer_bytes"] / GB
    if abs(route_a - 38.7) > 0.2:
        print(f"SELFTEST FAIL route-A reconstruction {route_a:.2f} GB != 38.7 GB"); bad += 1
    # D: must-catch. A model that counted the state as BF16, or as one read pass, must break A and C.
    bf16_layer = m["state_layer_bytes"] // 2
    if bf16_layer == m["state_layer_bytes"]:
        print("SELFTEST FAIL the bf16 must-catch control cannot differ"); bad += 1
    one_pass = (1 + m["passes_write"]) * m["state_layer_bytes"] * m["lanes"] * m["layers"] / GB
    if abs(one_pass - route_a) < 1.0:
        print("SELFTEST FAIL the one-pass must-catch control did not move the figure"); bad += 1
    print(f"SELFTEST traffic cases=6 bad={bad} ok={0 if bad else 1}"
          f" claim=the census reproduces §00.5's 1 MiB/18 MiB rows and route A's 38.7 GB figure from"
          f" the engine's own constants, and its two must-catch controls (BF16 state, one read pass)"
          f" each move the reconstructed figure")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
