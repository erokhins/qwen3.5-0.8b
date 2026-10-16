# `prototype/attn_valu_slim`

One question, asked of the disassembly: **how much of the attention wave's per-touch instruction bill
is control flow and format handling, and how much of that can a bit-identity-preserving re-spelling
remove?** D20 item 4 funded it at "24 % of the stream"; `results.md` §1–§4 is the answer,
`isa_census.output.txt` is the count, `HANDOFF.md` is the device pass that has not been run yet.

Start with `results.md`. Then:

| want | file |
|---|---|
| the before/after census, per category, per MODE | `isa_census.py` → `isa_census.output.txt` |
| the two levers, in one body with a MODE switch | `attn_valu_slim_kernels.h` |
| CPU bit-identity, with derived must-catch sets | `touch_probe.py` |
| the device A/B (not yet run) | `attn_valu_slim.hip.cpp`, `build.sh`, `HANDOFF.md` |

`--slim` values: `1` both levers, `2` unrolled score dot only, `3` guard-free fold only. `MODE 0` in
the header is the shipped body copied verbatim and exists so the census can prove the copy is free.

Nothing here is committed and nothing here has touched a device.
