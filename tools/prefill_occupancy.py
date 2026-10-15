#!/usr/bin/env python3
"""Who is actually using the iGPU while a prefill child runs — and is `gpu_busy_percent` even a signal?

    tools/gpu_lock.sh acquire <tag>
    .venv/bin/python tools/prefill_occupancy.py --out tmp/occ.csv -- \
        ./engine/build/decode_run --model models/Qwen3.5-0.8B --ids "$(cat tmp/ids4096.txt)" \
            --steps 1 --out tmp/occ_out
    tools/gpu_lock.sh release <tag>

WHAT THIS IS FOR. The ledger attributes a prefill wave to kernel families (`bench/performance-status.md`
§3) with the engine's OWN stage clock, whose markers are family-scoped: a gap between two kernels lands
inside whichever family's window contains it, so the clock cannot see an idle GPU at all (its completeness
account — `ungaged_ms=0.0228` of 243.17 — is a statement about the gages, not about the machine). This tool
watches the device from OUTSIDE the process, at a period the engine does not control, and it brackets the
child between two IDLE BASELINE windows, which is the only way to read these counters honestly on a box
whose own model server is running: `tools/gpu_lock.sh`'s header records that `flash_serve` "holds the iGPU
at ~98% utilisation from its polling loop", so a busy% figure taken without a baseline is not a measurement
of the child's work. Baseline-vs-run delta is the whole method, and when the baseline is already pegged the
tool says so and names the counter to use instead (the kernel-dispatch coverage ratio from
`tools/prefill_kernel_trace.py`) rather than printing a mean of 99 % as a result.

WHAT IT SAMPLES. `gpu_busy_percent`, `mem_info_vram_used`, `mem_info_gtt_used`, `mem_info_vis_vram_used`
under one device directory (resolved from `/sys/class/drm/card1/device`, falling back to the PCI path),
plus the child's own `/proc/<pid>/stat` jiffies, `/proc/<pid>/statm` RSS and `/proc/<pid>/status` context
switches — the host side, because "the GPU is idle" and "the host is blocked in a sync" look identical from
the device alone. The two memory counters do not move together on this box (`mem_info_vram_total` is a 2 GiB
BAR window while `hipMalloc` lands in GTT), so the summary prints every total next to every used figure and
names the counters that actually moved during the child's life.

IT DOES NOT ACQUIRE `tools/gpu_lock.sh`. This is a launcher, like `decode_run`, `lm_head_gemv` and
`mlp_gemv`: the caller holds the lock, and a child that acquires the lock its parent already holds parks
silently with no CPU and no GPU work — the deadlock recorded in `AGENTS.md`'s trap index (`grep -c gpu_lock
prototype/glue/run.py` is the pattern this file avoids).

CPU-JIFFIE GRANULARITY: `SC_CLK_TCK` is 100 here, so `/proc` CPU arrives in 10 ms ticks and a per-sample CPU
delta at a 5 ms period is quantised to 0 or 1 tick. Use the TOTALS in the summary (child CPU seconds over
its whole life, and the context-switch counts, which are exact) and do not differentiate the column.

`--selftest` needs no GPU and no device: it points the sampler at a temp directory of fake sysfs files and
runs `sleep` children, once with a pegged baseline and once with a live one, and asserts both verdicts.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PCI_PATH = "/sys/devices/pci0000:00/0000:00:08.1/0000:c2:00.0"
DRM_PATH = "/sys/class/drm/card1/device"
# The four device counters the plan asks for. `mem_info_preempt_used` is deliberately absent: it is the
# in-flight preempted-work figure, and on this box it is nonzero for the poll loop, not for the child.
COUNTERS = ("gpu_busy_percent", "mem_info_vram_used", "mem_info_gtt_used", "mem_info_vis_vram_used")
TOTAL_OF = {
    "mem_info_vram_used": "mem_info_vram_total",
    "mem_info_gtt_used": "mem_info_gtt_total",
    "mem_info_vis_vram_used": "mem_info_vis_vram_total",
}
CSV_FIELDS = (
    "t_rel_ms", "wall", "phase", "gpu_busy_percent", "mem_info_vram_used", "mem_info_gtt_used",
    "mem_info_vis_vram_used", "cpu_ms", "rss_mib", "threads", "vol_ctx", "nonvol_ctx", "exited",
)
MIB = 1024 * 1024


def pct(sorted_vals: list, q: float):
    """Nearest-rank percentile, index ceil(q·(n-1)). `round()` is unusable here: Python rounds half to
    even, so it picks the LOW sample for every even n."""
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, max(0, math.ceil(q * (len(sorted_vals) - 1))))]


# The two thresholds the busy%% verdict is decided by. Both are printed inside the verdict text, because a
# rule nobody can see is a rule nobody can argue with.
BASELINE_TOL = 10.0    # points the two idle windows may disagree by and still be one baseline
RESPONSIVE_TOL = 5.0   # points the child must move the counter by to count as visible


def verdict_busy(pre_median: float | None, post_median: float | None, run_median: float | None,
                 saturate: float) -> str:
    """Is `gpu_busy_percent` worth anything in this window? Four states, and only one of them licenses
    reading the number as load.

    The plan's expectation was saturation: `tools/gpu_lock.sh`'s header says `flash_serve` pins the
    counter near 98 % from its polling loop. What the smoke actually showed is subtler and needs its own
    state — the poll loop is only busy when the session is generating, so a window in which the parent
    agent is blocked reads a LOW baseline, and the two baseline windows can disagree by tens of points
    (measured here: pre 47 %, post 8 %, child 12 %). Reporting "baseline 14 %, child 12 %, therefore
    nothing happened" from that would be a fabricated negative, and reporting the -2 as a result would
    be worse. So: saturated / unstable / insensitive / usable, and three of the four refuse the number.
    """
    if pre_median is None or post_median is None or run_median is None:
        return ("VERDICT busy%: one of the three windows has no samples, so nothing is concluded."
                " Increase --baseline or check that the child really started.")
    base = statistics.median([pre_median, post_median])
    spread = abs(pre_median - post_median)
    delta = run_median - base
    if max(pre_median, post_median) >= saturate:
        return (f"VERDICT busy%: USELESS AS AN OCCUPANCY SIGNAL IN THIS WINDOW. With nothing of ours"
                f" running, the idle baseline median is {g(base)}% (saturation test: at or above"
                f" {g(saturate)}%); during the child it is {g(run_median)}%, a delta of"
                f" {delta:+.2f} points, i.e. indistinguishable from idle. `flash_serve` owns the"
                " counter from its polling loop (`tools/gpu_lock.sh` header), so mean/median busy%"
                " here measures the SERVER, not the child. Use"
                " `tools/prefill_kernel_trace.py`'s kernel-dispatch coverage ratio (Σ busy / span),"
                " which is taken from this process's own queue and cannot be inflated by a poll loop."
                f" Do NOT quote a busy%% number from a window like this one.")
    if spread > BASELINE_TOL:
        return (f"VERDICT busy%: BASELINE UNSTABLE, so the delta cannot be read yet. The two idle"
                f" windows disagree: pre {g(pre_median)}%, post {g(post_median)}%, spread"
                f" {spread:.2f} points against a tolerance of {g(BASELINE_TOL)}. Something was using"
                " the GPU inside a window that was supposed to be idle — most often this session's"
                " own tokens (`flash_serve` wakes when the parent agent generates), which is exactly"
                " why AGENTS.md demands a blocking call and zero subagents. A lengthened --baseline"
                " or a quiet window is the fix; the child-vs-baseline delta"
                f" ({delta:+.2f} points) is meaningless until the two windows agree.")
    if abs(delta) < RESPONSIVE_TOL:
        return (f"VERDICT busy%: INSENSITIVE IN THIS WINDOW. The baseline is stable"
                f" (pre {g(pre_median)}%, post {g(post_median)}%, mean {g(base)}%) but the child moves"
                f" it by only {delta:+.2f} points (< {g(RESPONSIVE_TOL)}). Either the child's GPU duty"
                " is genuinely inside the baseline noise, or it is a short run spending most of its"
                " life loading weights rather than computing — price that with the kernel trace's"
                " BY-WAVE table, which looks only at the span of kernel activity.")
    return (f"VERDICT busy%: USABLE in this window. Stable baseline {g(base)}% (pre/post"
            f" {g(pre_median)}/{g(post_median)}, spread {spread:.2f} ≤ {g(BASELINE_TOL)}) → child"
            f" {g(run_median)}%, {delta:+.2f} points. Read it as a DUTY CYCLE, not a load meter: the"
            " counter says the queue was not empty for that fraction of the window, and says nothing"
            " about how many wavefronts were resident — that is the kernel trace's occupancy column.")


# ---- sysfs ---------------------------------------------------------------------------------------
class DeviceCounters:
    """Reads one device directory's sysfs files. `dir` may be a real amdgpu device dir or a directory of
    plain files with the same names, which is what makes the parser testable without a GPU."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        missing = [c for c in COUNTERS if not (self.dir / c).exists()]
        if missing:
            raise SystemExit(f"prefill_occupancy: {self.dir} has no {missing[0]}; pass --sysfs PATH"
                             " with gpu_busy_percent and the mem_info_* files in it")

    def read(self) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for name in COUNTERS:
            try:
                with open(self.dir / name, "rb") as fh:
                    out[name] = int(fh.read().strip())
            except (OSError, ValueError):
                out[name] = None      # a vanished or mid-write file is a blank cell, never a 0
        return out

    def totals(self) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for used, total in TOTAL_OF.items():
            path = self.dir / total
            try:
                out[used] = int(path.read_text().strip())
            except (OSError, ValueError):
                out[used] = None
        return out


def resolve_device_dir(explicit: str | None) -> tuple[Path, str]:
    """One path, whichever exists: the DRM symlink first (it is stable across PCI re-enumeration), then
    the hard-coded PCI coordinate the plan names."""
    if explicit:
        return Path(explicit), f"--sysfs {explicit}"
    for cand, how in ((Path(DRM_PATH), "resolved from the DRM symlink"),
                      (Path(PCI_PATH), "the PCI path itself")):
        if (cand / "gpu_busy_percent").exists():
            return cand.resolve(), f"{cand} ({how})"
    raise SystemExit(f"prefill_occupancy: no amdgpu device directory at {DRM_PATH} or {PCI_PATH};"
                     " pass --sysfs PATH (or --selftest, which needs no device)")


# ---- the child's own host numbers -----------------------------------------------------------------
CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")


def child_cpu_ms(pid: int) -> float | None:
    """utime+stime of the child's whole thread group, in ms. Parsed after the LAST ')' because the comm
    field may itself contain spaces and parens."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read().decode("ascii", "replace")
    except OSError:
        return None
    rp = data.rfind(")")
    if rp < 0:
        return None
    fields = data[rp + 2:].split()
    # field 14 (utime) and 15 (stime) are index 11 and 12 once pid and comm are dropped.
    try:
        return (int(fields[11]) + int(fields[12])) * 1000.0 / CLK_TCK
    except (IndexError, ValueError):
        return None


def child_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/statm", "rb") as fh:
            return int(fh.read().split()[1]) * PAGE_BYTES
    except (OSError, IndexError, ValueError):
        return None


def child_proc_bits(pid: int) -> tuple[int | None, int | None, int | None]:
    """(threads, voluntary ctxt switches, nonvoluntary) of the thread-group LEADER — measured on this
    box: a 4-thread sleeper reports `Threads: 4` in the same file while its switch counters stay at the
    leader's own handful, so these two columns describe the submitting thread, which is exactly the one
    a dispatch-bound host cares about. They are exact counts, unlike jiffies. The point of carrying them:
    a host thread blocked in a stream sync racks up voluntary switches while its jiffies stay flat, and
    that is the one visible difference between 'the GPU is busy' and 'the host is waiting for it'."""
    try:
        with open(f"/proc/{pid}/status", "rb") as fh:
            text = fh.read().decode("ascii", "replace")
    except OSError:
        return None, None, None
    out: dict[str, int] = {}
    for line in text.splitlines():
        for key in ("Threads", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
            if line.startswith(key + ":"):
                out[key] = int(line.split(":", 1)[1])
    return out.get("Threads"), out.get("voluntary_ctxt_switches"), out.get("nonvoluntary_ctxt_switches")


# ---- the recorder ---------------------------------------------------------------------------------
class Recorder:
    def __init__(self, counters: DeviceCounters, period_s: float):
        self.counters = counters
        self.period = period_s
        self.rows: list[dict] = []
        self.t0 = time.monotonic()

    def one(self, phase: str, pid: int | None, exited: int = 0) -> dict:
        wall = time.monotonic()
        row = {
            "t_rel_ms": (wall - self.t0) * 1000.0,
            "wall": datetime.fromtimestamp(time.time()).isoformat(timespec="milliseconds"),
            "phase": phase,
            "exited": exited,
        }
        row.update(self.counters.read())
        if pid is None or exited:
            row.update(cpu_ms=None, rss_mib=None, threads=None, vol_ctx=None, nonvol_ctx=None)
        else:
            cpu = child_cpu_ms(pid)
            rss = child_rss_bytes(pid)
            thr, vol, nvol = child_proc_bits(pid)
            row.update(cpu_ms=cpu, rss_mib=None if rss is None else rss / MIB, threads=thr,
                       vol_ctx=vol, nonvol_ctx=nvol)
        self.rows.append(row)
        return row

    def run(self, baseline_pre: float, baseline_post: float, cmd: list[str],
            echo_stdout: bool) -> dict:
        """Idle baseline, then the child, then an idle baseline again. Returns the run facts."""
        deadline = self.t0 + baseline_pre
        while time.monotonic() < deadline:
            now = time.monotonic()
            self.one("pre", None)
            _sleep_until(now + self.period, self.period)

        info: dict = {"cmd": list(cmd)}
        stdout_lines: list[str] = []
        t_start = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        reader = _drain(proc.stdout, stdout_lines, echo_stdout)

        exit_t: float | None = None
        final_cpu_ms = None
        peak_rss = None
        while True:
            now = time.monotonic()
            rc = proc.poll()
            if rc is None:
                self.one("run", proc.pid)
                cpu, rss = child_cpu_ms(proc.pid), child_rss_bytes(proc.pid)
                if cpu is not None:
                    final_cpu_ms = cpu
                # PEAK, not last: a child that has exited but not yet been reaped is a zombie, and a
                # zombie's statm reads zero, so the final sample is the one sample that must not be
                # trusted for RSS.
                if rss:
                    peak_rss = rss if peak_rss is None else max(peak_rss, rss)
            else:
                if exit_t is None:
                    exit_t = now
                    info["rc"] = rc
                    info["cpu_ms"] = final_cpu_ms
                    info["rss_bytes"] = peak_rss
                    info["wall_s"] = exit_t - t_start
                    self.one("run", proc.pid, exited=1)
                elif now >= exit_t + baseline_post:
                    break
                else:
                    # The child is gone; this is the after-baseline again, and it is the second half of
                    # the only honest busy% reference this box allows.
                    self.one("post", None)
            # One period per pass, always: the CSV's cadence must not change when the child dies, or the
            # post-baseline is sampled at a different rate from the run it is the reference for.
            _sleep_until(now + self.period, self.period)
        reader.join(timeout=5.0)
        info["stdout_lines"] = stdout_lines
        # Anything still queued after the exit sample belongs to 'post'.
        return info

    # ---- reporting -------------------------------------------------------------
    def stats(self, phase: str, key: str = "gpu_busy_percent") -> dict:
        vals = [r[key] for r in self.rows if r["phase"] == phase and r[key] is not None]
        if not vals:
            return {"n": 0}
        s = sorted(vals)
        return {"n": len(s), "mean": statistics.fmean(s), "median": statistics.median(s),
                "min": s[0], "max": s[-1], "p90": pct(s, 0.90), "p99": pct(s, 0.99)}

    def above(self, threshold: float) -> dict:
        out = {}
        for phase in ("pre", "run", "post"):
            vals = [r["gpu_busy_percent"] for r in self.rows
                    if r["phase"] == phase and r["gpu_busy_percent"] is not None]
            out[phase] = (sum(1 for v in vals if v > threshold), len(vals))
        return out

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: ("" if r.get(k) is None else
                                     (f"{r[k]:.3f}" if isinstance(r[k], float) else r[k]))
                                 for k in CSV_FIELDS})

    def summary(self, info: dict, saturate: float, thresholds: list[float]) -> list[str]:
        lines: list[str] = []
        add = lines.append
        pre, run, post = self.stats("pre"), self.stats("run"), self.stats("post")
        add("---- prefill_occupancy ----")
        for ph, st in (("baseline pre", pre), ("child", run), ("baseline post", post)):
            if not st.get("n"):
                add(f"busy%  {ph:<13} no samples")
                continue
            add(f"busy%  {ph:<13} n={st['n']:<5} mean={st['mean']:6.2f} median={st['median']:6.2f}"
                f" p90={st['p90']:6.2f} max={st['max']:6.2f}")
        run_median = run.get("median")
        pre_median, post_median = pre.get("median"), post.get("median")
        for t in thresholds:
            ab = self.above(t)
            add(f"samples above busy% > {g(t)}: " +
                "  ".join(f"{ph} {ab[ph][0]}/{ab[ph][1]}" for ph in ("pre", "run", "post")))
        add("")
        add(verdict_busy(pre_median, post_median, run_median, saturate))
        add("")
        add("memory counters (MiB). `total` is printed next to every `used` because on this box"
            " mem_info_vram_total is a 2 GiB BAR window while `hipMalloc` lands in GTT, so the two"
            " do not move together and a delta read off the wrong one is meaningless:")
        totals = self.counters.totals()
        for name in COUNTERS:
            if name == "gpu_busy_percent":
                continue        # a percentage, not a memory counter
            vals = [r[name] for r in self.rows if r[name] is not None]
            if not vals:
                continue
            pre_max = max([r[name] for r in self.rows if r["phase"] == "pre" and r[name]] or [0])
            run_max = max([r[name] for r in self.rows if r["phase"] == "run" and r[name]] or [0])
            tot = totals.get(name)
            grew = run_max - pre_max
            add(f"  {name:<24} used_min={min(vals) / MIB:10.1f} used_max={max(vals) / MIB:10.1f}"
                f" total={'?' if tot is None else format(tot / MIB, '.1f'):>10}"
                f"  run-peak-over-pre={grew / MIB:+10.1f}"
                + ("   <- moves" if grew > MIB else "   (flat)")
                + ("  <- ABOVE TOTAL: the counter is not the allocation pool"
                   if tot is not None and max(vals) > tot else ""))
        add("")
        rss = info.get("rss_bytes")
        add("host side (the child): rc={} wall={} ms cpu={} ms rss_peak={} MiB".format(
            info.get("rc"), g((info.get("wall_s") or 0) * 1000), g(info.get("cpu_ms") or 0),
            "?" if rss is None else g(rss / MIB)))
        if info.get("wall_s") and info.get("cpu_ms") is not None:
            frac = (info["cpu_ms"] / 1000.0) / info["wall_s"]
            bound = "single-core bound" if frac > 0.85 else "not single-core bound"
            run_rows = [r for r in self.rows if r["phase"] == "run"]
            vol = [r["vol_ctx"] for r in run_rows if r["vol_ctx"] is not None]
            nvol = [r["nonvol_ctx"] for r in run_rows if r["nonvol_ctx"] is not None]
            add("  cpu fraction of wall = {} ({}). voluntary ctx switches {} / nonvoluntary {}:"
                " many voluntary switches on little CPU is a host BLOCKED in a sync or an"
                " allocation, not a host doing work.".format(
                    g(frac), bound, vol[-1] if vol else "?", nvol[-1] if nvol else "?"))
        add("")
        add("no timing claim here: wall/cpu are the child's own, taken in whatever window this ran,"
            " and this tool never takes best-of-N. A number for the ledger comes from the main agent"
            " blocking in a clean window (bench/performance-status.md §6).")
        return lines


def _sleep_until(deadline: float, period: float) -> None:
    """Sleep to a deadline in period-sized steps. `time.sleep` granularity on this box is tens of
    microseconds, which is nothing against a 5 ms period."""
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, period))


def _drain(pipe, sink: list[str], echo: bool) -> threading.Thread:
    """Drain the child's stdout in the background: not doing so deadlocks the child the moment its
    output exceeds the pipe buffer (a `--time-waves` run prints hundreds of lines)."""
    def loop() -> None:
        for line in pipe:
            sink.append(line.rstrip("\n"))
            if echo:
                sys.stdout.write(line)
        sys.stdout.flush()
        pipe.close()

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th


def g(x) -> str:
    """Print a float without trailing zeros; the summary is a table and 99.00000000000001 is noise."""
    try:
        return f"{float(x):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(x)


def _row_for(text: str, needle: str) -> str:
    """The summary line that names `needle`, for the selftest to assert on instead of guessing."""
    return next((l for l in text.splitlines() if needle in l), "<no such line>")


# ---- selftest: no device, no GPU, no lock ---------------------------------------------------------
def selftest() -> int:
    """Two synthetic sysfs directories and two `sleep` children. Case A reproduces this box (baseline
    pegged at 99 % by the poll loop) and must conclude the counter is useless; case B is a quiet machine
    where the child moves every counter and must conclude it is usable. Asserts the verdicts, the CSV,
    the RESULT passthrough and the child accounting."""
    import tempfile
    fails: list[str] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(what)

    def fake_dir(tmp: Path, name: str, script: list[tuple[float, dict[str, int]]]) -> Path:
        """A directory of plain files named like sysfs counters, mutated by a background thread on a
        schedule. The FIRST entry is written synchronously, so a sample can never observe a counter
        before its opening value and blame the child for the tool's own start-up."""
        d = tmp / name
        d.mkdir(parents=True, exist_ok=True)
        for f in list(COUNTERS) + list(TOTAL_OF.values()):
            (d / f).write_text("0\n")
        (d / "mem_info_vram_total").write_text(str(2 * 1024 ** 3))
        (d / "mem_info_gtt_total").write_text(str(120 * 1024 ** 3))
        (d / "mem_info_vis_vram_total").write_text(str(2 * 1024 ** 3))

        def write(vals: dict[str, int]) -> None:
            for k, v in vals.items():
                (d / k).write_text(f"{v}\n")

        write(script[0][1])

        def mutate() -> None:
            t0 = time.monotonic()
            for delay, vals in script[1:]:
                _sleep_until(t0 + delay, 0.01)
                write(vals)

        th = threading.Thread(target=mutate, daemon=True)
        th.start()
        return d

    def analyze(rec: Recorder, info: dict) -> list[str]:
        return rec.summary(info, saturate=95.0, thresholds=[50.0, 90.0, 99.0])

    with tempfile.TemporaryDirectory(prefix="prefill_occupancy_selftest_") as td:
        tmp = Path(td)
        # --- case A: this box. Baseline 99 %, child 99 %, nothing moves but GTT (weights arrive).
        gib = 1024 ** 3
        dirA = fake_dir(tmp, "pegged", [
            (0.0, dict(gpu_busy_percent=99, mem_info_vram_used=1750 * 1024 ** 2,
                       mem_info_gtt_used=26 * gib, mem_info_vis_vram_used=1750 * 1024 ** 2)),
            (0.25, dict(mem_info_gtt_used=int(27.5 * gib))),
        ])
        recA = Recorder(DeviceCounters(dirA), 0.02)
        infoA = recA.run(0.2, 0.2,
                         [sys.executable, "-c", "import time;time.sleep(0.3);"
                          "print('RESULT mode=fake steps=1 ms_total=245.730')"], echo_stdout=False)
        linesA = analyze(recA, infoA)
        textA = "\n".join(linesA)
        check(recA.rows and {r["phase"] for r in recA.rows} == {"pre", "run", "post"},
              "A: all three phases sampled", f"phases={sorted({r['phase'] for r in recA.rows})}")
        check(infoA["rc"] == 0 and any(l.startswith("RESULT") for l in infoA["stdout_lines"]),
              "A: child rc=0 and its RESULT line reached the caller verbatim")
        check("USELESS AS AN OCCUPANCY SIGNAL" in textA,
              "A: a 99%-pegged baseline is called useless, not reported as a load figure")
        check(_row_for(textA, "mem_info_gtt_used").endswith("<- moves"),
              "A: GTT, which does grow, is named as the counter that moves",
              _row_for(textA, "mem_info_gtt_used"))
        check("(flat)" in _row_for(textA, "mem_info_vram_used"),
              "A: a counter pinned by the 2 GiB BAR window is named as flat",
              _row_for(textA, "mem_info_vram_used"))
        csvA = tmp / "A.csv"
        recA.write_csv(csvA)
        rows = list(csv.DictReader(open(csvA)))
        check(len(rows) == len(recA.rows) and rows[0]["phase"] == "pre",
              "A: CSV rows equal the samples and the header parses", f"n={len(rows)}")
        check(all(r["gpu_busy_percent"] == "99" for r in rows),
              "A: every sampled value made it into the CSV", f"distinct={sorted({r['gpu_busy_percent'] for r in rows})}")

        # --- case B: a quiet machine. Baseline 3 %, child 62 %, GTT and RSS both climb.
        dirB = fake_dir(tmp, "quiet", [
            (0.0, dict(gpu_busy_percent=3, mem_info_vram_used=100 * 1024 ** 2,
                       mem_info_gtt_used=10 * gib, mem_info_vis_vram_used=100 * 1024 ** 2)),
            (0.25, dict(gpu_busy_percent=62, mem_info_gtt_used=int(11.7 * gib),
                        mem_info_vis_vram_used=101 * 1024 ** 2)),
            (0.75, dict(gpu_busy_percent=2, mem_info_gtt_used=10 * gib)),
        ])
        recB = Recorder(DeviceCounters(dirB), 0.02)
        infoB = recB.run(0.2, 0.2,
                         [sys.executable, "-c", "import time;time.sleep(0.5)"], echo_stdout=False)
        linesB = analyze(recB, infoB)
        textB = "\n".join(linesB)
        check("VERDICT busy%: USABLE" in textB, "B: a live, stable baseline is called usable",
              next(l for l in linesB if l.startswith("VERDICT"))[:80])
        st_run, st_pre = recB.stats("run"), recB.stats("pre")
        check(st_run["median"] > st_pre["median"] + 30, "B: child busy median tracks the fake load",
              f"pre={st_pre['median']} run={st_run['median']}")
        check("<- moves" in _row_for(textB, "mem_info_gtt_used") and
              "(flat)" in _row_for(textB, "mem_info_vram_used"),
              "B: the GTT climb is reported as a moving counter and the flat one as flat",
              _row_for(textB, "mem_info_gtt_used"))
        check(recB.rows and {r["phase"] for r in recB.rows} == {"pre", "run", "post"},
              "B: all three phases sampled", f"phases={sorted({r['phase'] for r in recB.rows})}")
        check(infoB["cpu_ms"] is not None and infoB["rss_bytes"] is not None,
              "B: child cpu and RSS were read from /proc",
              f"cpu_ms={infoB['cpu_ms']} rss_mib={g((infoB['rss_bytes'] or 0) / MIB)}")
        check(recB.rows[-1]["exited"] == 1 or any(r["exited"] for r in recB.rows),
              "B: the exit sample is marked")

        # --- the four busy%% verdicts, decided directly so each state is actually reachable.
        check(verdict_busy(99, 99, 99, 95.0).startswith("VERDICT busy%: USELESS"),
              "verdict: a pegged baseline is refused")
        check(verdict_busy(47, 8, 12, 95.0).startswith("VERDICT busy%: BASELINE UNSTABLE"),
              "verdict: two baseline windows that disagree by 39 points refuse the delta",
              "(this is what the real box actually did on the first smoke)")
        check(verdict_busy(10, 12, 13, 95.0).startswith("VERDICT busy%: INSENSITIVE"),
              "verdict: a stable baseline the child does not move is called insensitive")
        check(verdict_busy(10, 12, 62, 95.0).startswith("VERDICT busy%: USABLE"),
              "verdict: a stable baseline the child moves is called usable")
        check(verdict_busy(None, 10, 62, 95.0).startswith("VERDICT busy%: one of the three"),
              "verdict: a missing window is named, not silently averaged")
    print(("SELFTEST PASS" if not fails else f"SELFTEST FAIL: {fails}"))
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tmp/prefill_occupancy.csv",
                    help="CSV sample file to write (default tmp/prefill_occupancy.csv)")
    ap.add_argument("--period", type=float, default=5.0, help="sample period in ms (default 5)")
    ap.add_argument("--baseline", type=float, default=2.0,
                    help="idle baseline seconds BEFORE the child starts (default 2)")
    ap.add_argument("--baseline-post", type=float, default=None,
                    help="idle baseline seconds AFTER the child exits (default: --baseline)")
    ap.add_argument("--saturate", type=float, default=95.0,
                    help="baseline median at or above this declares gpu_busy_percent useless (95)")
    ap.add_argument("--above", default="50,90,99",
                    help="comma list of busy%% thresholds to count samples above (50,90,99)")
    ap.add_argument("--sysfs", default=None, help="device directory override")
    ap.add_argument("--quiet-child", action="store_true",
                    help="do not echo the child's stdout (it is still captured for the RESULT line)")
    ap.add_argument("--selftest", action="store_true", help="run the CPU-only selftest and exit")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- COMMAND [ARGS...] (usually ./engine/build/decode_run ... --steps 1)")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("nothing to run: pass `-- COMMAND ...`, e.g. `-- ./engine/build/decode_run ...`")
    if shutil.which(cmd[0]) is None and not Path(cmd[0]).exists():
        raise SystemExit(f"prefill_occupancy: child program {cmd[0]!r} not found")

    dev, how = resolve_device_dir(args.sysfs)
    counters = DeviceCounters(dev)
    thresholds = [float(x) for x in args.above.split(",") if x.strip()]
    print(f"device dir: {dev}   (resolved from {how})")
    print(f"child: $ {' '.join(cmd)}")
    print(f"baseline {g(args.baseline)} s pre / {g(args.baseline_post or args.baseline)} s post,"
          f" sample period {g(args.period)} ms")
    sys.stdout.flush()
    rec = Recorder(counters, args.period / 1000.0)
    info = rec.run(args.baseline, args.baseline_post if args.baseline_post is not None
                   else args.baseline, cmd, echo_stdout=not args.quiet_child)
    for line in rec.summary(info, args.saturate, thresholds):
        print(line)
    for line in info["stdout_lines"]:
        if line.startswith("RESULT"):
            print("child RESULT: " + line)
    out = Path(args.out)
    rec.write_csv(out)
    print(f"csv: {out}  ({len(rec.rows)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
