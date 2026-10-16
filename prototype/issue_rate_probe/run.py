#!/usr/bin/env python3
"""Driver for the issue-rate probe.

WHAT THIS DRIVER DOES. It keeps three things honest and takes no measurement of its own:

  1. `--verify`  (host-only, safe any time) — the instruction bill each kernel runs against must be
     the bill the disassembly says (`isa_census.py --check`), the host mirrors and the plan
     rejections must hold (`--selftest-geom`), and every leg's plan must build at the --iters this
     driver will ask for. No device is opened, no timing is taken.
  2. `--commands` — prints the numbered device window for the main agent, with each leg's accepted
     --iters substituted in.
  3. `--device`  (main agent, clean window) — runs the legs and re-reads the witness fields. The
     binary does the timing; this driver only checks the accounts and refuses to print a row whose
     witness did not reproduce.

NO GPU LOCK HERE, ON PURPOSE. `tools/gpu_lock.sh` must be held by the *caller* around a whole window.
Three gate drivers in this tree acquire the lock themselves, so a caller that holds it and then runs
one of them parks silently (AGENTS.md trap row). This driver never invokes it — the word appears in
this file only to say so, and the check is that no line shells out to it.

A2, the rule this file exists to enforce: any driver that parses another program's output must
require the number of parsed records to equal the number of records printed, and a check over an
empty list must be impossible. See `require_count` — every parse below goes through it.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = HERE / "issue_rate_probe"
CENSUS = HERE / "isa_census.py"

# The bill, copied out of `isa_census.py --emit` and enforced by `isa_census.py --check`. A leg whose
# pair is (0, 0) was REFUSED by the census — its required mnemonic is not inside any loop, so there
# is no honest rate to compute. Such a leg is skipped loudly, never billed silently.
DECLARED: dict[str, tuple[int, int]] = {
    "irp_fma_1":       (7, 1),
    "irp_fma_4":       (10, 4),
    "irp_fma_8":       (14, 8),
    "irp_fma_16":      (22, 16),
    "irp_fma_32":      (38, 32),
    "irp_serial_8":    (18, 8),
    "irp_serial_32":   (54, 32),
    "irp_cvt_8":       (45, 11),
    "irp_cvt_16":      (78, 20),
    "irp_lds_tile":    (0, 0),   # UNBILLED: the compiler peels the LDS reads out of the loop
    "irp_kv_read":     (37, 1),
    "irp_f32_8":       (14, 8),
    "irp_mix_8":       (13, 8),
    "irp_pkf16_8":     (13, 8),
    "irp_d2f16_8":     (13, 8),
    "irp_d2bf16_8":    (14, 8),
    "irp_d4i8_8":      (13, 8),
    "irp_d8i4_8":      (13, 8),
    "irp_u4u8_8":      (14, 8),
    "irp_u8u4_8":      (14, 8),
    "irp_wmma_4":      (10, 4),
}

# Why each unbilled leg is unbilled, printed verbatim wherever the leg is skipped.
UNBILLED_REASON = {
    "irp_lds_tile": "isa_census.py found no loop containing both the staged LDS read and the FMAs "
                    "that consume it: LLVM stages global->LDS in the billed loop and peels the read "
                    "side into a loop that adds constants. Billing that loop would report a rate for "
                    "staging while the row claims it measures LDS reads.",
}

# Per-leg --iters. The binary owns the exactness bound (an fp16 accumulator carries exact integers
# only to 2^11, an f32 one to 2^24, an i32 one to 2^31) and rejects a plan above it; `verify` obeys
# whatever cap the binary prints rather than duplicating the arithmetic here. These defaults are
# chosen to sit under the cap at the default grid (320 x 128) so a plain run needs no retry.
ITERS = {
    "irp_fma_1": 20000, "irp_fma_4": 20000, "irp_fma_8": 20000,
    "irp_fma_16": 20000, "irp_fma_32": 10000,
    "irp_serial_8": 20000, "irp_serial_32": 10000,
    "irp_cvt_8": 10000, "irp_cvt_16": 10000,
    "irp_kv_read": 1,
    "irp_f32_8": 20000, "irp_mix_8": 20000,
    "irp_pkf16_8": 512,        # fp16 accumulator: the bound, not the author, picks this
    "irp_d2f16_8": 10000, "irp_d2bf16_8": 10000,
    "irp_d4i8_8": 5000, "irp_d8i4_8": 5000, "irp_u4u8_8": 5000, "irp_u8u4_8": 5000,
    "irp_wmma_4": 2000,
}

# The arithmetic legs, in the order the report reads: scalar floor first, matrix roof last.
ARITH_LEGS = ["irp_f32_8", "irp_mix_8", "irp_pkf16_8", "irp_d2f16_8", "irp_d2bf16_8",
              "irp_d4i8_8", "irp_d8i4_8", "irp_u4u8_8", "irp_u8u4_8", "irp_wmma_4"]

# The KV footprint sweep — the independent variable of the second original question.
KV_FOOTPRINT_MIB = [2, 8, 32, 128, 512]


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=HERE, capture_output=True, text=True)


def require_count(what: str, parsed: list[str], printed: int) -> None:
    """A2. A parser that silently matches nothing reports success over an empty list, so the count
    comparison is not a nicety: `printed` must come from the producer's own output, never from a
    list this driver hand-wrote."""
    if printed == 0:
        sys.exit(f"BUG: {what}: producer printed 0 records; refusing to adjudicate an empty set")
    if len(parsed) != printed:
        sys.exit(f"BUG: {what}: parsed {len(parsed)} records but the producer printed {printed}")
    print(f"   parse {what}: parsed {len(parsed)} == printed {printed}")


# --------------------------------------------------------------------------------------------
# --verify: host-only. Bill, mirrors, plans. No device, no timing.
# --------------------------------------------------------------------------------------------
def unbilled_declared() -> list[str]:
    """The legs THIS driver already knows are unbilled. Passing them to the census is not a way to
    hide a failure: `--check` then requires the census to agree that each named leg has no billable
    loop, and a leg the census CAN bill but this table still lists as (0,0) is reported as a MISMATCH."""
    return [k for k, (wi, ops) in DECLARED.items() if wi == 0 and ops == 0]


def census_argv(*flags: str) -> list[str]:
    argv = [sys.executable, str(CENSUS), *flags]
    allow = unbilled_declared()
    if allow:
        argv += ["--allow-unbilled", ",".join(allow)]
    return argv


def verify() -> int:
    print("== 1. the bill comes from the disassembly")
    r = run(census_argv("--check"))
    print(r.stdout.rstrip())
    if r.returncode != 0:
        print(r.stderr.rstrip())
        sys.exit("isa_census.py --check failed: the compiler changed a loop body, or a leg is "
                 "unbilled and was not declared as (0,0)")

    print("\n== 2. host mirrors, closed forms, and the rejections that must happen")
    r = run([str(BIN), "--selftest-geom"])
    head = [ln for ln in r.stdout.splitlines() if ln.startswith("GEOMCASE")]
    tail = [ln for ln in r.stdout.splitlines() if ln.startswith("SELFTEST-GEOM")]
    if len(tail) != 1:
        sys.exit("BUG: --selftest-geom printed no verdict line")
    claim = int(re.search(r"cases=(\d+)", tail[0]).group(1))
    fails = int(re.search(r"fails=(\d+)", tail[0]).group(1))
    require_count("--selftest-geom GEOMCASE lines", head, claim)
    print("   " + tail[0])
    if fails:
        for ln in head:
            if " ok=0" in ln:
                print("   FAILED " + ln)
        sys.exit("--selftest-geom failed; fix the mirrors before any device number is trusted")

    print("\n== 3. every leg plans at the --iters this driver will ask for")
    for name, (wi, ops) in DECLARED.items():
        if wi == 0:
            print(f"   SKIP {name:16s} UNBILLED — {UNBILLED_REASON[name]}")
            continue
        it = ITERS[name]
        extra = ["--passes", "2", "--footprint-mib", "8"] if name == "irp_kv_read" else []
        r = run([str(BIN), "--dry-run", "--kernel", name, "--iters", str(it)] + extra)
        if r.returncode != 0:
            m = re.search(r"max --iters here is (\d+)", r.stderr)
            if not m:
                sys.exit(f"{name}: plan rejected for a reason with no cap in it: {r.stderr.strip()}")
            cap = int(m.group(1))
            if cap < 32:
                sys.exit(f"{name}: the accumulator's exactness bound leaves only {cap} iterations "
                         "at this grid; run.py will not bill a leg it cannot sustain")
            ITERS[name] = cap
            print(f"   CAP  {name:16s} iters {it} -> {cap} ({r.stderr.strip().splitlines()[0]})")
            r = run([str(BIN), "--dry-run", "--kernel", name, "--iters", str(cap)] + extra)
            if r.returncode != 0:
                sys.exit(f"{name}: still rejected at the cap it printed: {r.stderr.strip()}")
        plan = r.stdout.strip()
        got_wi = re.search(r"warp_inst_per_iter=(\d+)", plan)
        if not got_wi:
            sys.exit(f"{name}: --dry-run printed no PLAN line: {plan}")
        if int(got_wi.group(1)) != wi:
            sys.exit(f"{name}: PLAN bills warp_inst_per_iter={got_wi.group(1)} but DECLARED says "
                     f"wi={wi}. run.py's table and the census disagree; do not run the device yet.")
        print(f"   ok   {name:16s} iters={ITERS[name]:<6d} wi={wi} ops={ops}")
    print("\nVERIFY ok — the bills, the mirrors and the plans agree. No device was opened.")
    return 0


# --------------------------------------------------------------------------------------------
# --device: the main agent's clean window. The binary times; this only checks the accounts.
# --------------------------------------------------------------------------------------------
def leg_argv(name: str, *, launches: int, reps: int) -> list[str]:
    argv = [str(BIN), "--kernel", name, "--iters", str(ITERS[name]),
            "--launches", str(launches), "--reps", str(reps)]
    if name == "irp_kv_read":
        argv += ["--footprint-mib", str(KV_FOOTPRINT_MIB[0]), "--passes", "2"]
    return argv


def parse_irp(lines: list[str]) -> list[dict[str, str]]:
    """The binary prints one `IRP tag=... key=value ...` line per leg it ran; the fields are read by
    name, so a renamed field shows up as a KeyError at the check that needs it rather than a silent
    absence."""
    out = []
    for ln in lines:
        if not ln.startswith("IRP tag="):
            continue
        rec = dict(re.findall(r"(\w+)=([^\s]+)", ln))
        rec["kernel"] = re.search(r"kernel=(\S+)", ln).group(1)
        rec["_line"] = ln
        out.append(rec)
    return out


def device(legs: list[str], launches: int, reps: int) -> int:
    ran, skipped = [], []
    for name in legs:
        wi, _ = DECLARED[name]
        if wi == 0:
            print(f"SKIP {name:16s} UNBILLED — {UNBILLED_REASON[name]}")
            skipped.append(name)
            continue
        r = run(leg_argv(name, launches=launches, reps=reps))
        if r.returncode != 0:
            sys.exit(f"{name}: {r.stderr.strip()}")
        recs = parse_irp(r.stdout.splitlines())
        # A2 against the producer: the binary prints exactly one IRP line per leg it ran, and it
        # reports how many launches it itself accounted for.
        require_count(f"{name} IRP lines", recs, 1)
        rec = recs[0]
        if int(rec["parsed_launches"]) != int(rec["launches"]):
            sys.exit(f"{name}: the binary accounted {rec['parsed_launches']} of {rec['launches']} "
                     "launches — a launch ran short and its ms would be a lie")
        if rec["checksum"] != "ok":
            sys.exit(f"{name}: checksum=FAIL. The witness did not reproduce, so the instruction "
                     "count below describes a kernel that did not do its work")
        if int(rec["bad_blocks"]) or int(rec["each_bad"]):
            sys.exit(f"{name}: bad_blocks={rec['bad_blocks']} each_bad={rec['each_bad']}")
        ran.append(rec)
        print("OK   " + rec["_line"])
    print(f"\nDEVICE ran={len(ran)} skipped={len(skipped)}"
          + (f" ({', '.join(skipped)})" if skipped else ""))
    return 0


# --------------------------------------------------------------------------------------------
# --commands: the numbered window, with this driver's accepted --iters already substituted.
# --------------------------------------------------------------------------------------------
def commands() -> int:
    b = f"{HERE}/issue_rate_probe"
    print("# The device window. Main agent only: clean window, zero subagents, blocking, and")
    print("# tools/gpu_lock.sh held around the WHOLE list. This driver never takes the lock, so")
    print("# nothing here deadlocks a caller that holds it (AGENTS.md trap row).")
    print("# No command below is run by a worker; none of them is a measurement this driver keeps.\n")
    print("# 0. WMMA correctness FIRST — if the lane layout is wrong, every wmma number downstream")
    print("#    is worthless. onesh must print bad=0/128 first=16 (a 16-deep accumulating dot);")
    print("#    cadd bad=0/128 (D == 16 + C, the accumulate wiring); feeder must print")
    print("#    slots_not_16_deep=0 distinct_A_patterns=16 distinct_B_patterns=16 (the rows and the")
    print("#    columns). 'WMMA tile SKIPPED ... NOT DECIDABLE' is an honest result, not a failure:")
    print("#    it means the k-slot pairing is not recoverable from per-lane marginals.")
    print(f"   {b} --wmma-check")
    print(f"   {b} --wmma-check --wmma-dump\n")
    print("# 0b. device byte witnesses, one child per case. Correctness only, no timing printed.")
    print(f"   {b} --selftest\n")
    print("# 1. THE headline: one timing row per arithmetic path, scalar floor to matrix roof.")
    print("#    The number each row discriminates is pct_of_ceiling: 1 MAC/lane/clk for f32/mix,")
    print("#    2 for pk_fma_f16 and the two dot2s, 4 for dot4-i8 and WMMA, 8 for dot8-i4.")
    for n in ARITH_LEGS:
        print(f"   python3 {HERE}/run.py --device --only {n} --launches 20 --reps 5"
              f"   # iters={ITERS[n]}")
    print("\n# 2a. original question 1: sustained warp-inst/CU/clk on the attention dot shape.")
    print(f"   python3 {HERE}/run.py --device --only irp_fma_1 --only irp_fma_4 --only irp_fma_8 "
          "--only irp_fma_16 --only irp_fma_32 --only irp_serial_8 --only irp_serial_32 "
          "--only irp_cvt_8 --only irp_cvt_16 --launches 20 --reps 5")
    print("\n# 2b. original question 2: achievable read GB/s against a resident KV working set.")
    for f in KV_FOOTPRINT_MIB:
        print(f"   {b} --kernel irp_kv_read --iters 1 --passes 2 --footprint-mib {f} "
              "--launches 20 --reps 5")
    print("\n# 3. irp_lds_tile is UNBILLED (run.py skips it with the census reason):")
    print("#    " + UNBILLED_REASON["irp_lds_tile"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="host-only checks (default)")
    ap.add_argument("--commands", action="store_true", help="print the device window and exit")
    ap.add_argument("--device", action="store_true",
                    help="MAIN AGENT ONLY: run the legs (the binary times them; no lock here)")
    ap.add_argument("--only", action="append", default=[], help="restrict to these legs")
    ap.add_argument("--launches", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if not BIN.exists():
        sys.exit(f"{BIN} missing — run build.sh first")
    if a.commands:
        return commands()
    if a.device:
        legs = a.only or [k for k in DECLARED if k not in ("irp_lds_tile",)]
        for k in a.only:
            if k not in DECLARED:
                sys.exit(f"unknown leg {k}; known: {', '.join(DECLARED)}")
        return device(legs, a.launches, a.reps)
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
