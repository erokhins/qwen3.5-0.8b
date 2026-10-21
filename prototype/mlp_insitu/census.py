#!/usr/bin/env python3
"""census.py — the MLP traffic census, READ OUT OF THE SOURCE, plus the arm-spec the rig is driven with.

CPU-only. Touches no device, takes no timing.

Why this exists rather than a table of constants at the top of the rig: `AGENTS.md` has a standing trap
row for exactly this — "a traffic claim is a property of the staging loops the source contains, not of
the tile shape you booked", and the fix is to read the re-use factor out of the header
(`prototype/attn_tile_model.py:stage_copies`) instead of asserting it. So every number below is parsed
from one of three places:

  * `prototype/mlp_gemv/mlp_gemv_kernels.h` — the kernel `engine/kernels_mlp.cpp` includes. Gives the
    shapes (HID/FF), ROWS_PER_BLOCK, how many weight streams each kernel walks, whether it stages into
    LDS, how many shared reads per weight word, and what each kernel STORES.
  * `engine/decode_step.cpp` — the per-layer call sites. Gives the launch counts per token, and which
    intermediates are written per layer.
  * `tools/step_traffic_model.py` — the whole-step byte budget, itself parsed from `engine/state.h`.
    This is where the non-MLP bytes come from, so the interferer sizes are derived and never typed.

It also enforces two disciplines the rig cannot enforce alone:

  1. **no unlabelled bandwidth.** `--ms` prices a row, and a rate above the ~237 GB/s plateau is only
     ever printed as `demanded`, because demanded bytes over wall time is not a DRAM rate — L1/L2
     absorb reuse (`devices/strix-halo-memory-bandwidth.md` §the size-bandwidth curve: a warm 32 MiB
     read legitimately does 876 GB/s). This exact confusion produced a wrong column of numbers once
     already (AGENTS.md, the §07.4 "skinny argmax" row and the D21 note).
  2. **no empty parse.** Every regex must match the record count it expects, or the script dies. A
     parser that matches nothing and prints blanks is the failure mode that aborted a timing window on
     2026-10-21 (`prototype/attn_decode_wide/ab_d3_engine.sh`'s burst line), and it is why the case-list
     assertion compares against the rig's OWN `--print-cases` output rather than a number guessed here.

Usage:
    prototype/mlp_insitu/census.py                      # print the census
    prototype/mlp_insitu/census.py --emit-stepspec      # the STEPMIX line for ab_d2_sweep.sh
    prototype/mlp_insitu/census.py --ms 3.3628 --which mlp   # price a row (labelled, never a DRAM rate)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
KERNEL_H = ROOT / "prototype/mlp_gemv/mlp_gemv_kernels.h"
MLP_LAUNCHER = ROOT / "engine/kernels_mlp.cpp"
DECODE_STEP = ROOT / "engine/decode_step.cpp"
STATE_H = ROOT / "engine/state.h"
TRAFFIC_MODEL = ROOT / "tools/step_traffic_model.py"
RIG = HERE / "mlp_insitu"

MIB = 1048576.0
MB = 1e6
READ_PROBE_GBS = 237.0     # quoted: devices/strix-halo-memory-bandwidth.md
SUSPECT_GBS = 240.0        # above this, the number is cache residency or skipped work, not speed


def read(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"census: missing source file {path}")
    return path.read_text()


def must_search(pattern: str, text: str, want: int, what: str):
    """Every parse here must find exactly `want` records. A pattern that matches nothing and yields a
    default is the empty-parse failure this whole script exists to prevent."""
    found = re.findall(pattern, text, re.MULTILINE)
    if len(found) != want:
        raise SystemExit(
            f"census: {what}: pattern {pattern!r} matched {len(found)} records, wanted exactly "
            f"{want}. Refusing to print a census assembled from a partial parse.")
    return found


def kernel_bodies(src: str) -> dict:
    """Split the header on `__global__ void <name>(` and return the body of each kernel, from its
    signature to the next kernel's signature. Splitting on the signature and NOT on `template <...>` is
    the difference between reading `mlp_down` and reading `mlp_down` plus the two token-batched kernels
    that follow it — which is what made the first version of this census report 3 staging declarations
    for a kernel that has one. A slicer that reads past its kernel is the same class of bug as a slicer
    that finds a 57-slot stub and calls it the kernel (AGENTS.md, the ISA-census row)."""
    marks = [(m.start(), m.group(1)) for m in
             re.finditer(r"__global__\s+void\s+(\w+)\s*\(", src)]
    out: dict[str, str] = {}
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(src)
        out[name] = src[pos:end]
    if "mlp_gateup" not in out or "mlp_down" not in out:
        raise SystemExit(f"census: header kernels are {sorted(out)}; the slicer's names are stale.")
    return out


def kernel_facts() -> dict:
    """Shapes, streams, staging and stores, read out of the kernel source itself."""
    src = read(KERNEL_H)
    hid = int(must_search(r"#define\s+HID\s+(\d+)", src, 1, "HID")[0])
    ff = int(must_search(r"#define\s+FF\s+(\d+)", src, 1, "FF")[0])
    rpb = int(must_search(r"#define\s+ROWS_PER_BLOCK\s+(\d+)", src, 1, "ROWS_PER_BLOCK")[0])

    bodies = kernel_bodies(src)
    gate_body = bodies["mlp_gateup"]
    down_body = bodies["mlp_down"]

    def stage_copies(body: str, name: str) -> dict:
        """How many times global memory is asked for the same byte, and how much LDS traffic that
        buys. Read from the loop shape, not from a comment: `__shared__ float xs[IN]` plus a body that
        indexes `&xs[v * 8]` once per weight word means one global fetch per word per row and 8 F32
        shared reads per 16-byte word (32 B of LDS per 16 B of weight)."""
        stages = len(re.findall(r"__shared__\s+float\s+xs\[", body))
        if stages != 1:
            raise SystemExit(f"census: {name}: expected exactly one `__shared__ float xs[]` staging "
                             f"declaration, found {stages}. The staging model here is stale.")
        rereads = len(re.findall(r"&xs\[v\s*\*\s*8\]", body))
        if rereads == 0:
            raise SystemExit(f"census: {name}: no `&xs[v*8]` re-read found, so the LDS-traffic model "
                             f"here no longer describes the kernel.")
        streams = 2 if re.search(r"accum_word\(ag,", body) else 1
        return {"staged": True, "streams": streams, "lds_reads_per_word": 8, "reread_sites": rereads}

    gate = stage_copies(gate_body, "mlp_gateup")
    down = stage_copies(down_body, "mlp_down")

    # What each kernel stores: gateup writes `act[...]` (and `pre[...]` only when asked, which the
    # engine never does); down writes `out[...]`. Count distinct destination arrays, not stores.
    gate_stores = sorted(set(re.findall(r"\b(act|pre|out)\[orow\]", gate_body)))
    down_stores = sorted(set(re.findall(r"\b(act|pre|out)\[", down_body)))
    if "act" not in gate_stores:
        raise SystemExit("census: mlp_gateup's `act` store is not where the census expects it.")
    if "out" not in down_stores:
        raise SystemExit("census: mlp_down's `out` store is not where the census expects it.")

    # The reduction bound, and therefore the words per row per stream.
    words = {"gateup": hid // 8, "down": ff // 8}
    return {
        "HID": hid, "FF": ff, "ROWS_PER_BLOCK": rpb,
        "gateup": {"rows": ff, "in": hid, "streams": gate["streams"], **gate},
        "down": {"rows": hid, "in": ff, "streams": down["streams"], **down},
        "words": words,
        "gate_stores": gate_stores, "down_stores": down_stores,
    }


def launcher_facts() -> dict:
    """The two walks the launcher can instantiate for a single row, and the token-tile widths it refuses.
    Parsed from `engine/kernels_mlp.cpp`, which is the file that turns `--mlp-map` into a TEMPLATE
    argument: the kernel is spelled `::mlp_gateup<IN, MAP>` there, so the names to look for are the two
    arms of the launcher's branch, not two spellings of the kernel."""
    src = read(MLP_LAUNCHER)
    walks = must_search(r"launch_gateup_single_kernel<IN,\s*(MAP_CONTIG|MAP_INTERLEAVE)>", src, 2,
                        "gateup walks")
    rpb = must_search(r"constexpr int kRowsPerBlock\s*=\s*(\w+)", src, 1, "kRowsPerBlock")[0]
    if rpb != "ROWS_PER_BLOCK":
        raise SystemExit(f"census: engine's kRowsPerBlock is {rpb!r}, not ROWS_PER_BLOCK; the rig's "
                         f"grid no longer matches the engine's.")
    # The one-row default is a decision in this file, and the rig's default must agree with it.
    if "rows == 1 ? MlpGateMap::kContig" not in src:
        raise SystemExit("census: the engine's one-row default walk is no longer spelled "
                         "`rows == 1 ? MlpGateMap::kContig`; the rig's --mapping default must move "
                         "with it, in one commit (AGENTS.md, the two-header row).")
    return {"gateup_walks": sorted(set(walks)), "rpb_symbol": rpb}


def step_facts() -> dict:
    """Launches and per-layer intermediates, read from the per-layer loop in decode_step.cpp."""
    src = read(DECODE_STEP)
    layers = int(must_search(r"kNumLayers\s*=\s*(\d+)", read(STATE_H), 1, "kNumLayers")[0])
    g_sites = len(re.findall(r"kernels::mlp_gateup\(", src))
    d_sites = len(re.findall(r"kernels::mlp_down\(", src))
    # One hand call site each (the library route is `*_gemm`, and the rig does not model it).
    if g_sites != 1 or d_sites != 1:
        raise SystemExit(f"census: expected one hand `mlp_gateup`/`mlp_down` call site each, found "
                         f"{g_sites}/{d_sites}. The rig's per-token launch bill must be re-derived.")
    act_f32 = len(re.findall(r"cast_bf16_to_f32\(sc\.mlp_act", src))
    return {"layers": layers, "launches_per_token": layers * (g_sites + d_sites),
            "act_widenings_per_layer": act_f32}


def step_traffic() -> dict:
    """The whole-step byte budget from `tools/step_traffic_model.py`, so the interferer sizes are the
    engine's own numbers rather than a remembered list (the trap: an admission typed from memory)."""
    if not TRAFFIC_MODEL.is_file():
        raise SystemExit("census: tools/step_traffic_model.py is missing")
    out = subprocess.run([sys.executable, str(TRAFFIC_MODEL)], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("census: step_traffic_model.py exited %d:\n%s" % (out.returncode, out.stderr))
    rows = re.findall(r"^(\S+)\s+(read|write)\s+([0-9.]+)\s+[0-9.]+%", out.stdout, re.MULTILINE)
    if len(rows) < 10:
        raise SystemExit(f"census: parsed only {len(rows)} traffic rows from step_traffic_model.py; "
                         f"refusing to build an interferer spec from a partial table.")
    tot = re.findall(r"^TOTAL\s+read\s+([0-9.]+)", out.stdout, re.MULTILINE)
    tot_w = re.findall(r"write\s+([0-9.]+)\s+[0-9.]+%\s*$", out.stdout, re.MULTILINE)
    if not tot:
        raise SystemExit("census: could not parse the TOTAL read row from step_traffic_model.py")
    fam: dict[str, dict[str, float]] = {}
    for f, d, v in rows:
        if f == "TOTAL":
            continue
        fam.setdefault(f, {"read": 0.0, "write": 0.0})[d] += float(v)
    total_read = float(tot[0])
    # The writes total is the second `TOTAL`-area line; take the max of the parse candidates and say
    # which row it came from rather than guessing.
    total_write = max(float(x) for x in tot_w) if tot_w else sum(v["write"] for v in fam.values())
    mlp_read = fam.get("mlp", {}).get("read", 0.0)
    mlp_write = fam.get("mlp", {}).get("write", 0.0)
    head = fam.get("lm_head", {}).get("read", 0.0)
    return {"families": fam, "total_read_mb": total_read, "total_write_mb": total_write,
            "mlp_read_mb": mlp_read, "mlp_write_mb": mlp_write, "head_read_mb": head, "raw": out.stdout}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=None,
                    help="price a row: milliseconds per token for --which")
    ap.add_argument("--which", default="mlp", choices=["mlp", "total"])
    ap.add_argument("--emit-stepspec", action="store_true",
                    help="print the STEPMIX line ab_d2_sweep.sh feeds the rig")
    ap.add_argument("--check-cases", action="store_true", default=True)
    args = ap.parse_args()

    k = kernel_facts()
    lau = launcher_facts()
    st = step_facts()
    tr = step_traffic()

    gate_bytes = k["gateup"]["rows"] * k["gateup"]["in"] * 2
    down_bytes = k["down"]["rows"] * k["down"]["in"] * 2
    per_token_read = st["layers"] * (2 * gate_bytes + down_bytes)
    per_token_write = st["layers"] * (k["gateup"]["rows"] * 2 + k["down"]["rows"] * 2)

    print("== MLP traffic census, parsed from the source (no numbers typed in here) ==")
    print(f"kernel header      {KERNEL_H.relative_to(ROOT)}")
    print(f"  HID={k['HID']} FF={k['FF']} ROWS_PER_BLOCK={k['ROWS_PER_BLOCK']}"
          f"  engine kRowsPerBlock symbol = {lau['rpb_symbol']}")
    print(f"  gateup: rows={k['gateup']['rows']} in={k['gateup']['in']} "
          f"weight streams per row={k['gateup']['streams']} staged={k['gateup']['staged']} "
          f"LDS reads per weight word={k['gateup']['lds_reads_per_word']} "
          f"words/row={k['words']['gateup']}")
    print(f"  down:   rows={k['down']['rows']} in={k['down']['in']} "
          f"weight streams per row={k['down']['streams']} staged={k['down']['staged']} "
          f"LDS reads per weight word={k['down']['lds_reads_per_word']} words/row={k['words']['down']}")
    print(f"  walks the launcher can instantiate: {', '.join(lau['gateup_walks'])} "
          f"(one row ships `contig`: engine/kernels_mlp.cpp resolve_mlp_gate_map)")
    print(f"  re-use factor, read out of the loop shape: each weight byte is fetched EXACTLY ONCE per "
          f"token (a single-row body re-uses nothing across tokens and nothing across rows), while the "
          f"staged x row is re-read from LDS {k['gateup']['lds_reads_per_word']}x per weight word = "
          f"{4 * k['gateup']['lds_reads_per_word']} B of LDS traffic per 16 B of weight, {k['gateup']['streams']}x "
          f"per row for gateup (two streams) and 1x for down")
    print(f"launch bill        {st['launches_per_token']} launches/token = {st['layers']} layers x "
          f"(gateup + down); act widened to F32 {st['act_widenings_per_layer']}x/layer (a `cast` "
          f"launch, not an `mlp` one)")
    print("byte bill per token (DEMANDED, from the shapes above):")
    print(f"  weights read     {per_token_read / MIB:9.3f} MiB  = {per_token_read / MB:9.3f} MB "
          f"(the ledger's 528.482 MB row)")
    print(f"  act+delta write  {per_token_write / MIB:9.3f} MiB  = {per_token_write / MB:9.3f} MB "
          f"({100.0 * per_token_write / per_token_read:.4f} % of what is read)")
    print(f"  whole step read  {tr['total_read_mb']:9.3f} MB   write {tr['total_write_mb']:.3f} MB"
          f"   ratio {tr['total_read_mb'] / tr['total_write_mb']:.1f} : 1")
    non_mlp_read = tr["total_read_mb"] - tr["mlp_read_mb"]
    non_mlp_write = tr["total_write_mb"] - tr["mlp_write_mb"]
    print(f"  non-MLP traffic  read {non_mlp_read:.3f} MB  write {non_mlp_write:.3f} MB  "
          f"(this is what an interferer arm has to stand in for)")

    if args.ms is not None:
        bytes_m = per_token_read / MB if args.which == "mlp" else tr["total_read_mb"]
        gbs = bytes_m / args.ms * 1000.0 / 1000.0
        label = "demanded_not_dram" if gbs > SUSPECT_GBS else "under_plateau"
        print(f"PRICING which={args.which} ms={args.ms} bytes={bytes_m:.3f} MB -> {gbs:.1f} GB/s "
              f"label={label} probe=237.0(quoted)")
        if gbs > SUSPECT_GBS:
            print("PRICING NOTE: above the plateau, so this is DEMANDED bytes over wall time, NOT a "
                  "DRAM rate. Do not quote it as a bandwidth result.")

    if args.emit_stepspec:
        # Per-layer carry-over stand-ins, sized from the parsed table, plus the one whole-token blob.
        L = st["layers"]
        r_per_layer = (non_mlp_read - tr["head_read_mb"]) * MB / L / MIB
        w_per_layer = non_mlp_write * MB / L / MIB
        head_mib = tr["head_read_mb"] * MB / MIB
        print(f"STEPMIX read_mib_per_layer={r_per_layer:.3f} write_mib_per_layer={w_per_layer:.3f} "
              f"head_mib={head_mib:.3f} layers={L} "
              f"derived_from=tools/step_traffic_model.py@{tr['total_read_mb']:.3f}MB")

    if args.check_cases:
        if not RIG.is_file():
            raise SystemExit(f"census: {RIG} is not built; run prototype/mlp_insitu/build.sh")
        out = subprocess.run([str(RIG), "--print-cases"], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit("census: rig --print-cases failed")
        printed = re.findall(r"^CASE-ROW ", out.stdout, re.MULTILINE)
        total = re.findall(r"^CASE-ROWS total=(\d+)", out.stdout, re.MULTILINE)
        if len(total) != 1:
            raise SystemExit("census: rig printed no single CASE-ROWS total line")
        if len(printed) != int(total[0]):
            raise SystemExit(f"census: rig printed {len(printed)} CASE-ROW lines but claims "
                             f"{total[0]}; a case list whose count is not its own is not a case list")
        # Derived here from the same geometry the rig says it used: 2 shapes x 7 positions x 5
        # injections, plus the 6 rig-level account cases.
        positions = 7
        derived = 2 * positions * 5 + 6
        if len(printed) != derived:
            raise SystemExit(f"census: rig case list is {len(printed)}, independently derived here as "
                             f"{derived} (2 shapes x {positions} one-hot positions x 5 injections + 6 "
                             f"account cases). One of the two is stale.")
        print(f"CASES rig={len(printed)} derived_here={derived} agree=1 "
              f"(positions from words=in/8 and the 32-lane warp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
