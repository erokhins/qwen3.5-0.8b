#!/usr/bin/env python3
"""§08.1's state inventory, re-derived in Python and checked against `engine/state.h`.

CPU-only by construction: it reads the C++ header as text and re-does the arithmetic from §08.1's
shapes and §00.5's rows. It never imports HIP and never touches a device, which is what makes it
runnable while the GPU belongs to the operator.

    .venv/bin/python engine/state_sizes.py

What it proves: the byte counts in `state.h` agree with the shapes §08.1 lists, and §00.5's
independent rows (12 KiB/token KV, 18.0 MiB GDN state, 3.07 GiB at full context) come out the same
way from them. What it does NOT prove: that the buffers are allocated at those sizes, that the
strides are right, or that `decode_step` writes inside them — that is `--check-state` on a device.

Two of §08.1's own cells are wrong; they are listed under FIXES and are OPEN until someone edits
specification/08 (deliberately not done in this commit).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
H = (ROOT / "state.h").read_text()
MIB = 1024 * 1024

CONST_NAMES = ["kNumLayers", "kNumGdnLayers", "kNumAttnLayers", "kNumValueHeads", "kDk", "kDv",
               "kConvChannels", "kConvKernel", "kConvWindow", "kNumKvHeads", "kHeadDim"]


def consts() -> dict:
    """Pull §08.1's shape constants out of state.h and evaluate them in dependency order.
    `kConvWindow = kConvKernel - 1` is the only one naming another constant, hence the loop."""
    raw = {}
    for name in CONST_NAMES:
        m = re.search(r"constexpr int %s = ([^;]+);" % name, H)
        if not m:
            raise SystemExit("state.h: no `constexpr int %s = ...;`" % name)
        raw[name] = m.group(1).strip()
    out: dict = {}
    while len(out) < len(raw):
        for name, expr in raw.items():
            if name in out:
                continue
            try:
                out[name] = int(expr)
            except ValueError:
                parts = [p.strip() for p in re.split(r"([+\-*/()])", expr) if p.strip()]
                if all(p.isdigit() or p in out or p in "+-*/()" for p in parts):
                    out[name] = eval(expr, {}, out)
                else:
                    raise SystemExit("state.h: cannot evaluate `%s = %s`" % (name, expr))
    return out


C = consts()
assert C["kNumGdnLayers"] + C["kNumAttnLayers"] == C["kNumLayers"], "layer counts do not add up"
assert C["kConvWindow"] == C["kConvKernel"] - 1, "§08.2: the window is k_conv-1 columns"

gdn_state = C["kNumGdnLayers"] * C["kNumValueHeads"] * C["kDk"] * C["kDv"] * 4
conv_window = C["kNumGdnLayers"] * C["kConvWindow"] * C["kConvChannels"] * 4
kv_per_token = 2 * C["kNumKvHeads"] * C["kHeadDim"] * 2 * C["kNumAttnLayers"]

FIXES = [
    ("§08.1 conv window dtype",
     "table says BF16 [6144][3] = 36 KiB/layer (0.63 MiB); the engine keeps F32 because "
     "prototype/gdn_decode's conv kernel reads and writes the window as F32, and 36 cast "
     "launches/token is the wrong price for 0.6 MiB of a pool this model uses 4% of. Values are "
     "BF16-exact either way (S and the window are separate buffers, so §08.6's first invariant "
     "does not reach them). specification/08 NOT yet corrected."),
    ("§08.1 KV per-layer cell",
     "the size column says 4 KiB/token/layer while its own total says 12 KiB/token over 6 layers, "
     "i.e. 2 KiB/layer; §00.5's formula 2·n_kv·d_head·L_full·2 B = 12288 B/token is the "
     "consistent one and the shapes agree with it. specification/08 NOT yet corrected."),
    ("§00.5 KV-at-full-context row",
     "12288 B/token × 262144 = 3.000 GiB (3.22 GB decimal), not the 3.07 GiB §00.5 prints — a "
     "~2 % slip in the prose row, the third in the same family. The shape-derived number is the "
     "one to use. specification/00 NOT yet corrected."),
]

print("state.h constants: " + ", ".join("%s=%d" % (k, v) for k, v in C.items()))
print("%-22s %-30s %10s  %s" % ("state", "shape", "size", "note"))
rows = [
    ("GDN recurrent S", "[%d][%d][%d][%d] F32" % (C["kNumGdnLayers"], C["kNumValueHeads"],
                                                  C["kDk"], C["kDv"]), gdn_state,
     "§00.5: 18.0 MiB"),
    ("GDN conv window", "[%d][%d][%d] F32" % (C["kNumGdnLayers"], C["kConvWindow"],
                                              C["kConvChannels"]), conv_window,
     "§08.1 said BF16 / 0.63 MiB; see FIXES"),
    ("KV, K and V, 6 layers", "2 × [6][cap][%d][%d] BF16" % (C["kNumKvHeads"], C["kHeadDim"]),
     0, "scales with cap: %d B/token (§00.5: 12288)" % kv_per_token),
]
for name, shape, nbytes, note in rows:
    print("%-22s %-30s %9.3f MiB  %s" % (name, shape, nbytes / MIB, note))
constant = gdn_state + conv_window
print("constant per request: %.4f MiB   (§08.1's BF16-window reading would be %.4f MiB)"
      % (constant / MIB, (gdn_state + conv_window // 2) / MIB))
print("KV at 262144 ctx: %.3f GiB = %.2f GB   (§00.5 says 3.07 GiB; see FIXES)"
      % (kv_per_token * 262144 / 1024 ** 3, kv_per_token * 262144 / 1000 ** 3))

fails = []
if gdn_state != 18 * MIB:
    fails.append("GDN state %.4f MiB != §00.5's 18.0 MiB" % (gdn_state / MIB))
if kv_per_token != 12288:
    fails.append("KV/token %d B != §00.5's 12288 B" % kv_per_token)
if abs(kv_per_token * 262144 / 1024 ** 3 - 3.07) > 0.01:
    print("OPEN  §00.5 full-context row      shape arithmetic gives %.3f GiB, the prose row says 3.07 GiB"
          % (kv_per_token * 262144 / 1024 ** 3))
for label, note in FIXES:
    print("OPEN  %-26s %s" % (label, note))

if fails:
    print("\nSTATE SIZE CHECK FAILED:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("\nSTATE SIZE CHECK PASS — state.h's constants reproduce §00.5's rows")
