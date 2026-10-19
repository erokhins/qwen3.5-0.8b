#!/usr/bin/env python3
"""Stub decode_run for validating a7_host_split.sh's assertions WITHOUT the device.

Emits `prototype/perprompt/a7_fixture.log`, but shaped by the A7_* environment the way the
patched engine shapes it: no A7 lines at all unless the flag was set, one `A7REPEAT
kind=repeat` per A7_REPEAT, and one `A7SYNC1` per wave execution (the run's own wave plus
each repeat). That makes it a real test of the driver's count assertions: if the driver
asserted a hand-typed constant instead of the count the environment implies, this stub says
so. `--dry-run` of the driver needs nothing; this covers legs C/D/E.
"""
import os
import sys

here = os.environ.get("A7_STUB_DIR") or os.path.dirname(os.path.abspath(__file__))
lines = open(os.path.join(here, "a7_fixture.log")).read().splitlines()

warm = int(os.environ.get("A7_WARMUP", "0") or 0)
rep = int(os.environ.get("A7_REPEAT", "0") or 0)
body = os.environ.get("A7_BODY") is not None
sync1 = os.environ.get("A7_SYNC1") is not None

for ln in lines:
    # The fixture carries example A7 lines for `a7_parse_selftest.sh`; here every A7 line is
    # regenerated from the environment, so the counts the driver asserts are the env's, not
    # the fixture's. Keeping the fixture's copies would double them and hide an off-by-one.
    if ln.startswith(("A7REPEAT", "A7BODY", "A7SYNC1")):
        continue
    print(ln)

if warm:
    print(f"A7REPEAT kind=warmup idx=0 wall_ms=49.8820 tokens=512")
    if body:
        print("A7BODY wave=0 rows=512 submit_ms=6.402")
    if sync1:
        print("A7SYNC1 wave=0 device_ms=49.9000")
if body:
    print("A7BODY wave=0 rows=512 submit_ms=6.402")   # the run's own wave
if sync1:
    print("A7SYNC1 wave=0 device_ms=113.4100")       # the run's own wave

# Repeat waves the driver asked for, each with its own body/sync lines when those are on.
for i in range(rep):
    print(f"A7REPEAT kind=repeat idx={i} wall_ms=49.0010 tokens=512")
    if body:
        print("A7BODY wave=0 rows=512 submit_ms=5.118")
    if sync1:
        print("A7SYNC1 wave=0 device_ms=48.1020")
sys.exit(0)
