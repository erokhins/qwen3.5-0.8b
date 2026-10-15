#!/usr/bin/env python3
"""Device-busy coverage, dispatch bubbles and per-kernel shares for one prefill, from rocprofv3.

    tools/gpu_lock.sh acquire <tag>
    .venv/bin/python tools/prefill_kernel_trace.py --waves 8 --out tmp/kt4096 -- \
        ./engine/build/decode_run --model models/Qwen3.5-0.8B --ids "$(cat tmp/ids4096.txt)" \
            --steps 1 --out tmp/kt4096_out
    tools/prefill_kernel_trace.py --csv tmp/kt4096/*_kernel_trace.csv --waves 8   # re-read, no device
    tools/gpu_lock.sh release <tag>

WHY THIS EXISTS AND WHAT ONLY IT CAN SEE. `bench/performance-status.md` §3 splits a prefill wave into
families with the engine's own stage clock, and that clock is family-scoped: the time between two kernels
lands inside whichever family's window contains it, which is why its completeness account
(`ungaged_ms=0.0228` of 243.17 ms) says nothing about whether the GPU was ever idle. `gpu_busy_percent`
cannot settle it either — `flash_serve` pins it near 99 % with its polling loop (`tools/gpu_lock.sh`
header), and `tools/prefill_occupancy.py` exists to prove that from a baseline. Kernel *dispatch*
timestamps are immune to both problems: they come from the queue, they name every kernel, and the ratio

    coverage = Σ kernel duration / span of kernel activity

is an occupancy number that a poll loop cannot inflate, because the poll loop is not in this process's
trace. Everything else here is what a reader needs to argue with that one number: the gap distribution
(who waits, how often, for whom), the launch rate, the population of sub-5 µs launches (is the wave a
swarm of tiny dispatches?), and per-kernel grid/workgroup geometry turned into workgroups-per-CU.

CU COUNT: 40, never 20. `devices/strix-halo-compute.md` §"Correction (same day)" settled that HIP's
`multiProcessorCount = 20` is a reporting artifact and that `rocminfo`'s 40 is the device. This tool does
not have to take either on faith: rocprofv3 writes an `*_agent_info.csv` beside the trace carrying
`Cu_Count`, `Max_Waves_Per_Cu`, `Wave_Front_Size`, `Lds_Size_In_Kb` and `Simd_Count`, so `--cus auto`
(the default) reads the machine's own answer from that file and prints where it came from. 40 is only the
fallback when no agent_info file is present.

THE TRACE IS NOT THE RUN. Instrumenting a dispatch costs the launcher something real, so absolute ms
under this tool are not the engine's — do not move a ledger number with it. Shares, ratios, counts and
durations-of-interest (which kernel, how many, how big a grid) are what it is for, and §6's rule about
quoting a pair inside one window applies to any comparison.

IT DOES NOT ACQUIRE `tools/gpu_lock.sh`: same rule as `tools/prefill_occupancy.py` and `decode_run` — the
caller holds the lock, and a self-locking child parks forever inside a lock its parent already owns
(`AGENTS.md` trap index; `grep -c gpu_lock prototype/glue/run.py` for the pattern being avoided).

`--selftest` writes a synthetic trace whose numbers are hand-computed and asserts on them; it needs no
device, no rocprofv3 and no lock.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROCPROFV3 = os.environ.get("ROCPROFV3", "/usr/bin/rocprofv3")
FALLBACK_CUS = 40      # devices/strix-halo-compute.md: 40 CUs, NOT the 20 HIP reports
SMALL_US = 5.0         # "a swarm of tiny launches?" — the threshold the plan names

# rocprofv3 1.3.5 kernel_trace.csv columns, as produced on this box (pinned by --selftest's fixture).
COL_NAME = "Kernel_Name"
COL_START = "Start_Timestamp"
COL_END = "End_Timestamp"
COL_GRID = ("Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z")
COL_WG = ("Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z")
COL_EXTRA = ("LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count")
AGENT_CUS_KEYS = ("Cu_Count",)


# ---- reading --------------------------------------------------------------------------------------
def read_trace(paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Kernel records from one or more kernel_trace CSVs. Returns (records, notes).

    Records are sorted by start. `dur_ns` is taken from the timestamps, which is what rocprofv3 fills
    even when a build declines to emit a duration column.
    """
    rows: list[dict] = []
    notes: list[str] = []
    for path in paths:
        n_before = len(rows)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            missing = [c for c in (COL_NAME, COL_START, COL_END, *COL_GRID, *COL_WG)
                       if c not in header]
            if missing:
                raise SystemExit(f"prefill_kernel_trace: {path} has no column {missing[0]!r}; its"
                                 f" header is: {header[:6]}… — the analyser pins rocprofv3's own"
                                 " names rather than guessing at a positional layout")
            n0 = len(rows)
            n_bad = 0
            for rec in reader:
                try:
                    start = int(rec[COL_START])
                    end = int(rec[COL_END])
                except (TypeError, ValueError):
                    n_bad += 1
                    continue
                blocks = 1
                for c in COL_GRID:
                    blocks *= max(1, int(rec[c] or 1))
                threads = 1
                for c in COL_WG:
                    threads *= max(1, int(rec[c] or 1))
                # Grid_Size_* in this CSV is the thread extent (blocks x workgroup), not the block
                # count: `rec_kernel<<<16, 128>>>` arrives as Grid_Size_X=2048, Workgroup_Size_X=128,
                # and `conv_kernel<<<dim3(6144/256, tokens), 256>>>` as 6144 x tokens. Dividing the two
                # is the only way to get workgroups, and it is exact for every row of a real trace
                # (checked on this box's smoke: 2048/128=16=NH, 6144*64/256=1536=24 x 64).
                workgroups, ragged = divmod(blocks, threads)
                if ragged:
                    workgroups += 1
                    notes.append(f"{path.name}: a grid that is not a whole multiple of its workgroup"
                                 f" ({blocks} threads / {threads}) — rounded up")
                rows.append({
                    "name": rec[COL_NAME],
                    "start": start,
                    "end": max(start, end),
                    "dur": max(0, end - start),
                    "blocks": workgroups,
                    "threads": threads,
                    "grid": tuple(int(rec[c] or 1) for c in COL_GRID),
                    "wg": tuple(int(rec[c] or 1) for c in COL_WG),
                    "extra": {k: int(rec[k] or 0) for k in COL_EXTRA if k in rec},
                    "queue": rec.get("Queue_Id"),
                    "stream": rec.get("Stream_Id"),
                })
            if n_bad:
                notes.append(f"{path.name}: {n_bad} row(s) with unparsable timestamps skipped")
        if len(rows) == n_before:
            # A file that yielded nothing is a parser that matched nothing (AGENTS.md: a check whose input
            # list can be empty cannot fail), so it is an error rather than an empty report.
            raise SystemExit(f"prefill_kernel_trace: no kernel records parsed from {path}")
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows, notes


def read_agent_info(paths: list[Path]) -> dict:
    """The GPU agent row from rocprofv3's agent_info CSV, if it is there."""
    for path in paths:
        try:
            with open(path, newline="") as fh:
                for rec in csv.DictReader(fh):
                    if rec.get("Agent_Type") == "GPU":
                        return rec
        except OSError:
            continue
    return {}


def resolve_cus(agent: dict, requested: str) -> tuple[int, str]:
    if requested != "auto":
        return int(requested), f"--cus {requested} (given by hand)"
    for key in AGENT_CUS_KEYS:
        try:
            val = int(agent.get(key) or 0)
        except ValueError:
            val = 0
        if val > 0:
            return val, (f"agent_info {key}={val} (the trace's own machine row; HIP would say 20"
                         " — devices/strix-halo-compute.md says use this one)")
    return FALLBACK_CUS, f"fallback {FALLBACK_CUS} (no *_agent_info.csv beside the trace)"


# ---- statistics -----------------------------------------------------------------------------------
def pct(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile over an already-sorted list: index ceil(q·(n-1)), 0-based.

    Stated because p99 of 40 samples is only a number once everyone agrees which sample it names, and
    because `round()` cannot be used for it: Python rounds half to EVEN, so `round(0.5*(n-1))` silently
    picks the LOW sample for every even n. The selftest asserts the two-sample p50 so that cannot come
    back quietly.
    """
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def merged_busy(rows: list[dict]) -> list[tuple[int, int]]:
    """Union of [start,end] intervals. The engine submits on one stream, so overlaps are normally
    empty, but a hipBLAS split-K fixup or a copy engine record would overlap, and a coverage ratio that
    double-counted an overlap would be a lie."""
    out: list[list[int]] = []
    for r in rows:
        if out and r["start"] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], r["end"])
        else:
            out.append([r["start"], r["end"]])
    return [(s, e) for s, e in out]


def analyse(rows: list[dict], cus: int, wave_groups: int = 1) -> dict:
    span = (rows[-1]["end"] - rows[0]["start"]) if rows else 0
    total = sum(r["dur"] for r in rows)
    busy = merged_busy(rows)
    busy_ns = sum(e - s for s, e in busy)
    # Gaps between the merged busy intervals, each attributed to the kernel that ended the interval
    # before it: "who was the GPU just finished with when it went idle" is the question dispatch bubbles
    # actually ask.
    gaps: list[tuple[int, str, int]] = []      # (gap_ns, preceding kernel name, gap index)
    # Which kernel ended each busy interval, resolved once: a linear scan per gap would be O(n × gaps),
    # and a 20k-token wave has ~400k records and hundreds of thousands of gaps.
    last_to_end: dict[int, dict] = {}
    for r in rows:
        cur = last_to_end.get(r["end"])
        if cur is None or r["start"] > cur["start"]:
            last_to_end[r["end"]] = r
    for i in range(1, len(busy)):
        gap = busy[i][0] - busy[i - 1][1]
        if gap <= 0:
            continue
        culprit = last_to_end[busy[i - 1][1]]
        gaps.append((gap, culprit["name"], i))
    gap_ns = [g[0] for g in gaps]
    by_culprit: dict[str, list[int]] = {}
    for gap, name, _ in gaps:
        by_culprit.setdefault(name, []).append(gap)
    small = [r for r in rows if r["dur"] < SMALL_US * 1000]
    return {
        "n": len(rows),
        "span_ns": span,
        "total_ns": total,
        "busy_ns": busy_ns,
        "overlap": (total - busy_ns) > 0,
        "coverage_sum": (total / span) if span else 0.0,
        "coverage_union": (busy_ns / span) if span else 0.0,
        "gap_count": len(gaps),
        "gap_total_ns": sum(gap_ns),
        "gap_p50": pct(sorted(gap_ns), 0.50),
        "gap_p90": pct(sorted(gap_ns), 0.90),
        "gap_p99": pct(sorted(gap_ns), 0.99),
        "gap_max": max(gap_ns) if gap_ns else 0.0,
        "gap_by_culprit": sorted(((name, len(v), sum(v)) for name, v in by_culprit.items()),
                                 key=lambda t: -t[2])[:12],
        "top_gaps": sorted(gaps, key=lambda t: -t[0])[:10],
        "launches_per_s": (len(rows) * 1e9 / span) if span else 0.0,
        "small_n": len(small),
        "small_sum_ns": sum(r["dur"] for r in small),
        "by_name": _by_name(rows, cus),
        "waves": _by_wave(rows, wave_groups),
    }


def _by_name(rows: list[dict], cus: int) -> list[dict]:
    total = sum(r["dur"] for r in rows) or 1
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["name"], []).append(r)
    out = []
    for name, rs in groups.items():
        dur = sum(r["dur"] for r in rs)
        shapes = {(r["grid"], r["wg"], tuple(r["extra"].get(k, 0) for k in COL_EXTRA)) for r in rs}
        one = rs[0]
        wf = -(-one["threads"] // 64)          # wavefronts per workgroup (Wave_Front_Size 64)
        out.append({
            "name": name,
            "n": len(rs),
            "sum_ns": dur,
            "mean_ns": dur / len(rs),
            "max_ns": max(r["dur"] for r in rs),
            "share": dur / total,
            "grid": one["grid"], "wg": one["wg"],
            "wgs": one["blocks"], "wf_per_wg": wf,
            "wgs_per_cu": one["blocks"] / cus,
            "wf_per_cu": one["blocks"] * wf / cus,
            "lds": one["extra"].get("LDS_Block_Size", 0),
            "vgpr": one["extra"].get("VGPR_Count", 0),
            "sgpr": one["extra"].get("SGPR_Count", 0),
            # Geometry varies when one name covers several launch shapes — which is exactly the case a
            # single printed row would otherwise average away (a hipBLAS heuristic that picks a
            # different kernel per M shows up here, not in the mean).
            "multi_shape": len(shapes) > 1,
        })
    return sorted(out, key=lambda d: -d["sum_ns"])


def _by_wave(rows: list[dict], waves: int) -> list[dict]:
    """Split the record list into `waves` equal consecutive chunks and price each.

    A prefill is a sequence of identical waves (same rows per wave, hence the same launch count per wave),
    so an equal split by launch index is the wave boundary — and it is the only way to ask "is the FIRST
    wave slower, because it is touching fresh pages?" of a single run instead of two. `AGENTS.md`: a first
    GPU pass over fresh pages runs ~35 GB/s, 7x slow. The tail chunk is ragged (it also carries the priced
    decode step), which is printed rather than hidden.
    """
    if waves < 2 or not rows:
        return []
    n = len(rows)
    size = n // waves
    out = []
    for i in range(waves):
        chunk = rows[i * size:(i + 1) * size] if i < waves - 1 else rows[(waves - 1) * size:]
        if not chunk:
            continue
        busy = merged_busy(chunk)
        span = chunk[-1]["end"] - chunk[0]["start"]
        busy_ns = sum(e - s for s, e in busy)
        out.append({"i": i, "n": len(chunk), "start": chunk[0]["start"], "end": chunk[-1]["end"],
                    "span_ns": span, "sum_ns": sum(r["dur"] for r in chunk),
                    "coverage": (busy_ns / span) if span else 0.0,
                    "gap_ns": span - busy_ns})
    return out


# ---- printing -------------------------------------------------------------------------------------
def us(ns: float) -> str:
    return f"{ns / 1000.0:.3f}"


def ms(ns: float) -> str:
    return f"{ns / 1e6:.4f}"


def report(st: dict, cus: int, cus_src: str, paths: list[Path], notes: list[str]) -> None:
    print("---- prefill_kernel_trace ----")
    print("trace files: " + ", ".join(str(p) for p in paths))
    for n in notes:
        print("note: " + n)
    print(f"CU count: {cus}  [{cus_src}]")
    print(f"kernel records          {st['n']}")
    print(f"span of kernel activity {ms(st['span_ns'])} ms   (first start .. last end)"
          "   <- includes model load and hipBLAS init: read the BY-WAVE table for the wave itself")
    print(f"Σ kernel duration       {ms(st['total_ns'])} ms")
    print(f"Σ busy (merged)         {ms(st['busy_ns'])} ms"
          + ("" if not st["overlap"] else "   <- records overlap; Σ duration double-counts them"))
    print("")
    print(f"DEVICE-BUSY COVERAGE    {st['coverage_union'] * 100:.2f} %  "
          f"(Σ busy / span; Σdur/span would say {st['coverage_sum'] * 100:.2f} %)")
    print(f"Σ idle inside span      {ms(st['span_ns'] - st['busy_ns'])} ms")
    print(f"launch rate             {st['launches_per_s'] / 1000.0:.1f} k launches/s")
    print(f"launches < {g(SMALL_US)} µs   {st['small_n']} of {st['n']}"
          f" ({st['small_n'] / st['n'] * 100:.1f} % of launches) totalling {ms(st['small_sum_ns'])} ms"
          f" = {st['small_sum_ns'] / st['total_ns'] * 100 if st['total_ns'] else 0:.1f} % of Σ duration")
    print("")
    print(f"DISPATCH BUBBLES: {st['gap_count']} gaps, Σ {ms(st['gap_total_ns'])} ms"
          f" ({st['gap_total_ns'] / st['span_ns'] * 100 if st['span_ns'] else 0:.2f} % of span)")
    print(f"  gap µs: p50={us(st['gap_p50'])} p90={us(st['gap_p90'])} p99={us(st['gap_p99'])}"
          f" max={us(st['gap_max'])}    (nearest-rank percentiles)")
    print("  Σ gap attributed to the kernel that preceded each gap:")
    for name, n, tot in st["gap_by_culprit"]:
        print(f"    {ms(tot):>10} ms  n={n:<7} {short(name)}")
    print("  ten largest gaps:")
    for gap, name, _ in st["top_gaps"]:
        print(f"    {us(gap):>10} µs  after {short(name)}")
    print("")
    print(f"PER-KERNEL (Σ duration descending, {len(st['by_name'])} distinct names, shares of Σ dur)"
          "  wgs = workgroups per launch, wgs/CU = wgs / CU count")
    print(f"  {'ms':>10} {'share':>7} {'count':>7} {'mean µs':>9} {'max µs':>9}  "
          f"{'wgs':>9} {'wg':>13} {'wgs/CU':>7} {'wf/CU':>7} {'lds':>7} {'vgpr':>5}  name")
    for d in st["by_name"]:
        wg = f"{d['wg'][0]}x{d['wg'][1]}x{d['wg'][2]}"
        flag = "  <-geometry varies" if d["multi_shape"] else ""
        star = " *" if d["wgs"] < cus else ""    # the grid cannot even cover the machine
        print(f"  {ms(d['sum_ns']):>10} {d['share'] * 100:6.2f}% {d['n']:>7} "
              f"{us(d['mean_ns']):>9} {us(d['max_ns']):>9}  "
              f"{d['wgs']:>9} {wg:>13} {d['wgs_per_cu']:>7.2f}{star} {d['wf_per_cu']:>7.2f} "
              f"{d['lds']:>7} {d['vgpr']:>5}  {short(d['name'])}{flag}")
    if any(d["wgs"] < cus for d in st["by_name"]):
        print("  (*) fewer workgroups in the launch than there are CUs: that kernel cannot fill the"
              " machine however long it runs; wf/CU is the machine-average wavefronts per CU"
              " (Wave_Front_Size 64).")
    if st["waves"]:
        print("")
        print(f"BY WAVE ({len(st['waves'])} equal launch-index chunks — the wave boundary is a launch"
              " count, not a time; the last chunk is ragged and also holds the priced step)")
        base = st["waves"][0]["span_ns"]
        for w in st["waves"]:
            print(f"  wave {w['i']:>2}  n={w['n']:<7} span={ms(w['span_ns']):>9} ms"
                  f"  Σdur={ms(w['sum_ns']):>9} ms  coverage={w['coverage'] * 100:6.2f} %"
                  f"  idle={ms(w['gap_ns']):>8} ms"
                  f"  vs wave 0: {w['span_ns'] / base * 100 if base else 0:6.1f} %")


def short(name: str, width: int = 62) -> str:
    """A table cell for a kernel name.

    hipBLAS/Tensile names run to hundreds of characters and share a 70-character prefix, and
    `(anonymous namespace)::` names BEGIN with a parenthesis, so "cut at the first `(`" — the obvious
    thing — prints an empty cell for a third of the engine's kernels and a blurred one for hipBLAS.
    So: drop only the argument list, keep the last name qualifier, and when the cell must still be cut,
    end it with a stable `#hash` of the full name so two similar Tensile kernels cannot occupy one
    spelling (the full string is always greppable in the CSV).
    """
    anon = "(anonymous namespace)"
    head = name
    idx = head.find("(")
    while idx != -1 and head[idx:idx + len(anon)] == anon:
        idx = head.find("(", idx + 1)          # skip an anonymous-namespace paren, wherever it sits
    if idx != -1:
        head = head[:idx]
    if "::" in head:
        head = head.split("::")[-1] or head
    head = head or name
    if len(head) <= width:
        return head
    digest = hashlib.blake2s(name.encode(), digest_size=3).hexdigest()
    return head[:width - len(digest) - 2].rstrip("_") + "…" + digest


def g(x) -> str:
    try:
        return f"{float(x):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(x)


# ---- selftest -------------------------------------------------------------------------------------
def selftest() -> int:
    """A hand-built trace with hand-computed answers.

    Fixture (µs, one stream, two idle holes):
        alpha  start 0/10/20  dur 10 each   4 blocks of 128 threads  → busy [0,30]
        beta   start 50       dur 900       40 blocks of 256 threads → busy [50,950]
        gamma  start 1050     dur 4         1 block of 32 threads    → busy [1050,1054]
    Σ dur = 30 + 900 + 4 = 934 µs. Span = 0..1054 = 1054 µs. Coverage = 934/1054.
    Gaps: 20 µs after alpha, 100 µs after beta → Σ 120 µs. Nearest-rank p50 of [20,100] is index
    round(0.5·1)=1 → 100 µs, which is stated because a p50 of two samples is otherwise a coin toss.
    Sub-5 µs population: gamma alone (4 µs).
    """
    import tempfile
    fails: list[str] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(what)

    hdr = [COL_NAME, COL_START, COL_END, *COL_GRID, *COL_WG, *COL_EXTRA, "Queue_Id", "Stream_Id"]

    def row(name, start_us, dur_us, gx, gy, gz, wx, wy, wz, lds=0, vgpr=0, sgpr=0):
        return [name, str(int(start_us * 1000)), str(int((start_us + dur_us) * 1000)),
                str(gx), str(gy), str(gz), str(wx), str(wy), str(wz),
                str(lds), "0", str(vgpr), str(sgpr), "1", "0"]

    with tempfile.TemporaryDirectory(prefix="prefill_kernel_trace_selftest_") as td:
        d = Path(td)
        trace = d / "self_kernel_trace.csv"
        with open(trace, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            for st_us in (0, 10, 20):
                w.writerow(row("alpha_kernel", st_us, 10, 512, 1, 1, 128, 1, 1, lds=2048, vgpr=24))
            w.writerow(row("beta_kernel", 50, 900, 10240, 1, 1, 256, 1, 1))
            w.writerow(row("gamma_kernel", 1050, 4, 32, 1, 1, 32, 1, 1))
        agent = d / "self_agent_info.csv"
        with open(agent, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Agent_Type", "Cu_Count", "Wave_Front_Size", "Max_Waves_Per_Cu"])
            w.writerow(["CPU", "32", "0", "0"])
            w.writerow(["GPU", "40", "64", "32"])

        rows, notes = read_trace([trace])
        check(len(rows) == 5, "parses every row of a kernel_trace CSV", f"n={len(rows)}")
        check(notes == [], "no unparsable rows in a well-formed fixture", f"notes={notes}")
        cus, src = resolve_cus(read_agent_info([agent]), "auto")
        check(cus == 40 and "agent_info" in src, "CU count comes from agent_info, not from HIP's 20", src)
        st = analyse(rows, cus=40, wave_groups=2)
        check(st["span_ns"] == 1054_000, "span is first-start..last-end", f"{st['span_ns']} ns")
        check(st["total_ns"] == 934_000, "Σ duration is the sum of the records", f"{st['total_ns']} ns")
        check(abs(st["coverage_union"] - 934 / 1054) < 1e-12, "coverage = Σ busy / span",
              f"{st['coverage_union']:.6f}")
        check(st["gap_count"] == 2 and st["gap_total_ns"] == 120_000,
              "gaps are the holes between busy intervals",
              f"n={st['gap_count']} sum={st['gap_total_ns']}")
        check(st["gap_max"] == 100_000 and st["gap_p50"] == 100_000,
              "gap percentiles use ceil() nearest-rank, not round()'s half-to-even",
              f"p50={st['gap_p50']}")
        check(st["gap_by_culprit"][0][0] == "beta_kernel",
              "a gap is attributed to the kernel that preceded it", str(st["gap_by_culprit"][:2]))
        check(st["small_n"] == 1 and st["small_sum_ns"] == 4_000,
              "sub-5 µs population is counted by count and by Σ",
              f"n={st['small_n']} sum={st['small_sum_ns']}")
        names = {d2["name"]: d2 for d2 in st["by_name"]}
        check(names["beta_kernel"]["share"] > 0.9, "shares divide the whole Σ",
              f"beta={names['beta_kernel']['share']:.3f}")
        check(names["alpha_kernel"]["n"] == 3 and names["alpha_kernel"]["mean_ns"] == 10_000,
              "per-name count and mean")
        check(names["alpha_kernel"]["wgs"] == 4 and names["alpha_kernel"]["wgs_per_cu"] == 4 / 40 and
              names["alpha_kernel"]["wf_per_cu"] == 4 * 2 / 40,
              "workgroups come from grid/threads, and occupancy divides by the 40-CU count",
              f"wgs={names['alpha_kernel']['wgs']} per_cu={names['alpha_kernel']['wgs_per_cu']}")
        check(len(st["waves"]) == 2 and st["waves"][0]["n"] == 2,
              "the launch-index wave split produces equal chunks",
              str([w["n"] for w in st["waves"]]))
        # A malformed header must be refused by name, not parsed into nonsense.
        bad = d / "bad_kernel_trace.csv"
        bad.write_text("kind,start,dur\nKERNEL,1,2\n")
        try:
            read_trace([bad])
            check(False, "a CSV without the pinned columns is refused")
        except SystemExit as exc:
            check("Kernel_Name" in str(exc), "a CSV without the pinned columns is refused", str(exc)[:60])
        empty = d / "empty_kernel_trace.csv"
        empty.write_text(",".join(hdr) + "\n")
        try:
            read_trace([empty])
            check(False, "a header-only CSV is refused rather than reported as a clean zero")
        except SystemExit as exc:
            check("no kernel records" in str(exc), "a header-only CSV is refused", str(exc)[:60])
    print(("SELFTEST PASS" if not fails else f"SELFTEST FAIL: {fails}"))
    return 0 if not fails else 1


# ---- driver ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", default=[],
                    help="analyse an existing kernel_trace.csv instead of running anything"
                         " (repeatable; a glob is allowed)")
    ap.add_argument("--out", default=None, help="directory for rocprofv3 output (default tmp/kt-<pid>)")
    ap.add_argument("--name", default="prefill", help="rocprofv3 -o NAME (default prefill)")
    ap.add_argument("--cus", default="auto", help="CU count for occupancy (default: agent_info, else 40)")
    ap.add_argument("--waves", type=int, default=1,
                    help="split the records into N equal launch-index chunks = the wave count,"
                         " ceil(prompt_tokens / wave_rows) (default 1 = no split)")
    ap.add_argument("--rocprof", default=ROCPROFV3, help=f"rocprofv3 binary (default {ROCPROFV3})")
    ap.add_argument("--rocprof-arg", action="append", default=[],
                    help="extra argument for rocprofv3, repeatable (e.g. --pmc SQ_WAVES)")
    ap.add_argument("--selftest", action="store_true", help="run the CPU-only selftest and exit")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- COMMAND [ARGS...] (usually ./engine/build/decode_run ... --steps 1)")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    paths: list[Path] = []
    for pattern in args.csv:
        hits = sorted(glob.glob(pattern))
        if hits:
            paths.extend(Path(h) for h in hits)
        else:
            paths.append(Path(pattern))   # a literal path with no metacharacters; read_trace will
                                          # complain by name if it is not there
    info_paths: list[Path] = []
    if not paths:
        cmd = list(args.cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            ap.error("nothing to analyse: pass `--csv GLOB` or `-- COMMAND ...`")
        out = Path(args.out or f"tmp/kt-{os.getpid()}")
        out.mkdir(parents=True, exist_ok=True)
        argv = [args.rocprof, "--kernel-trace", "--output-format", "csv",
                "-d", str(out), "-o", args.name, *args.rocprof_arg, "--", *cmd]
        print("$ " + " ".join(argv))
        sys.stdout.flush()
        proc = subprocess.run(argv)          # stdout goes through: the child's RESULT/WAVE lines stay
        if proc.returncode != 0:
            raise SystemExit(f"prefill_kernel_trace: rocprofv3 exited {proc.returncode}")
        paths = sorted(Path(p) for p in glob.glob(str(out / f"{args.name}*kernel_trace*.csv")))
        info_paths = sorted(Path(p) for p in glob.glob(str(out / "*agent_info*.csv")))
        if not paths:
            raise SystemExit(f"prefill_kernel_trace: rocprofv3 produced no"
                             f" {args.name}*kernel_trace*.csv in {out}; it wrote"
                             f" {[p.name for p in sorted(out.iterdir())]}")
    else:
        for p in paths:
            info_paths += sorted(Path(x) for x in glob.glob(str(p.parent / "*agent_info*.csv")))

    rows, notes = read_trace(paths)
    cus, cus_src = resolve_cus(read_agent_info(info_paths), args.cus)
    report(analyse(rows, cus=cus, wave_groups=max(1, args.waves)), cus, cus_src, paths, notes)


if __name__ == "__main__":
    sys.exit(main())
