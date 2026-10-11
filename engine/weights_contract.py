#!/usr/bin/env python3
"""Every shape `check_weights` claims, re-read from the shard header instead of assumed.

    .venv/bin/python engine/weights_contract.py [models/Qwen3.5-0.8B] [cpp-to-check]

CPU-only: no device, no engine code, no torch. It reads three independent sources and fails if any
two disagree —

  * the `Expected` table compiled into `engine/decode_step.cpp` (what the engine claims),
  * the §00 constants in `engine/state.h` (how those claims are spelled as expressions),
  * the safetensors header of the shard named by `model.safetensors.index.json` (what is true).

The second argument exists so the check can be shown to bite: point it at a mutated copy of
`decode_step.cpp` and it must name the row it disagrees with. A shape check that has never failed is
a decoration (`prototype/mlp_gemv/results.md`).

Why it exists: the first device run of `decode_run` died in `check_weights` on
`lm.layers.0.linear_attn.conv1d.weight`. The table said rank 2 `[6144,4]`; the shard says rank 3
`[6144,1,4]`. The claim was wrong and the only thing that could have told me was a GPU round trip,
which is the scarcest resource in this repo (`AGENTS.md`: subagents do not touch the GPU, and a
quiescent window is scheduled, not free). Shape claims are not worth one. `state_sizes.py` already
sets the pattern of checking C++ constants against a spec by reading the header as text; this does
the same for the checkpoint.

The checkpoint's rank census is part of the assertion, not a footnote: of 488 tensors, exactly 18 are
rank 3 and all 18 are `linear_attn.conv1d.weight` [6144,1,4]. That is why `Expected` carries an
explicit `rank` field — a "d1 == 0 means 1-D" convention cannot describe this tensor, and folding the
depth-1 dimension away inside a checker is how a checker stops reporting what the store holds.
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPP = ROOT / "engine" / "decode_step.cpp"
STATE_H = ROOT / "engine" / "state.h"

def consts() -> dict:
    """Every `constexpr int kX = ...` in state.h, evaluated in dependency order (kQkvRows names
    kValueDim, which names kNumValueHeads). Same trick state_sizes.py uses on its shorter list."""
    raw = dict(re.findall(r"constexpr int (k\w+) = ([^;]+);", STATE_H.read_text()))
    out: dict = {}
    while len(out) < len(raw):
        for name, expr in raw.items():
            if name in out:
                continue
            try:
                out[name] = int(expr.strip())
            except ValueError:
                parts = [p.strip() for p in re.split(r"([+\-*/()])", expr) if p.strip()]
                if all(p.isdigit() or p in out or p in "+-*/()" for p in parts):
                    out[name] = eval(expr, {}, out)
                else:
                    raise SystemExit(f"state.h: cannot evaluate `{name} = {expr}`")
    return out


def engine_names(text: str) -> dict:
    """`kConvW` → `linear_attn.conv1d.weight`, and likewise for the two top-level names. These are the
    strings `loader.cpp` hands `WeightStore`, so this is the same spelling the engine looks up."""
    names = dict(re.findall(r'constexpr const char\* (k\w+) = "([^"]+)";', text))
    if not names:
        raise SystemExit("decode_step.cpp: no `constexpr const char* k… = \"…\"` names found")
    return names


def parse_entries(text: str, names: dict, C: dict) -> list:
    """The (name, scope, rank, dims, dtype) the C++ declares.

    Two forms are read: the named `constexpr Expected kWant…{…}` lines (which the adapter uses, on
    GDN layers only), and the entries of each `static const Expected k…[] = { … }` list. The list's
    NAME decides which layers an entry applies to — `kEveryLayer` all 24, `kGdnBlock` the 18 linear
    ones, `kAttnBlock` the 6 attention ones — and that matters: checking `self_attn.o_proj` on layer 0
    is not a stricter check, it is a wrong one, and the first version of this script did exactly that.
    Both forms are one uniform six-field shape, so one parser handles them; a new form must be added
    here too, which is the cost of checking a compiled table from outside — the same cost
    `state_sizes.py` accepts for `state.h`.
    """
    def fields(body: str):
        leaf, rank, d0, d1, d2, dtype = [p.strip() for p in body.split(",")]
        dims = tuple(int(C[t]) if t in C else int(t) for t in (d0, d1, d2) if t != "0")
        return leaf, int(rank), dims, dtype.split("::")[1]

    scope_of = {"kEveryLayer": "all", "kGdnBlock": "gdn", "kAttnBlock": "attn"}
    out = []
    named: dict = {}
    for m in re.finditer(r"constexpr Expected (k\w+)\s*\{\s*([^}]+?)\s*\};", text):
        named[m.group(1)] = fields(m.group(2))

    for b in re.finditer(r"static const Expected (k\w+)\[\] = \{(.*?)\n    \};", text, re.S):
        block_scope = scope_of.get(b.group(1))
        if block_scope is None:
            raise SystemExit(f"decode_step.cpp: not an Expected list I know: {b.group(1)}")
        for line in b.group(2).splitlines():
            line = line.split("//")[0].strip().rstrip(",")
            if not line:
                continue
            scope = block_scope
            m = re.match(r"\{\s*(.+?)\s*\}$", line)
            if m:
                leaf, rank, dims, dtype = fields(m.group(1))
            elif line in named:
                leaf, rank, dims, dtype = named[line]
            else:
                raise SystemExit(f"decode_step.cpp: unparseable Expected entry `{line}`")
            out.append((leaf, scope, rank, dims, dtype))
    for m in re.finditer(r"want\((k\w+), [^,]+, \{([^}]+)\}\)", text):
        leaf, rank, dims, dtype = fields(m.group(2))
        out.append((leaf, "top", rank, dims, dtype))

    resolved = []
    for leaf, scope, rank, dims, dtype in out:
        key = leaf.split("::")[1] if "::" in leaf else leaf
        if key not in names:
            raise SystemExit(f"decode_step.cpp: name `{key}` is not declared as a constexpr char*")
        resolved.append((names[key], scope, rank, dims, dtype))
    # Each name is claimed once per scope; the adapter's two kWant* entries repeat the GDN block's, so
    # drop the duplicates rather than printing them twice.
    return list(dict.fromkeys(resolved))


def header_shapes(model_dir: Path) -> dict:
    """name → (dtype, shape) straight from the safetensors header. No torch, no mmap of the data."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    table: dict = {}
    for s in shards:
        with open(model_dir / s, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n).decode())
        head.pop("__metadata__", None)
        table.update({k: (v["dtype"], list(v["shape"])) for k, v in head.items()})
    return table


def to_ckpt(engine_name: str) -> str:
    """Invert §02.5 (`model.language_model.* → lm.*`). If the loader's rule changes, the name this
    builds stops existing in the header and the row fails as missing — which is the right outcome for
    a rename, and the reason the inverse is checked rather than trusted."""
    if engine_name.startswith("lm."):
        return "model.language_model." + engine_name[len("lm."):]
    return engine_name


def main() -> int:
    model_dir = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-0.8B")
    cpp = Path(sys.argv[2]) if len(sys.argv) > 2 else CPP
    text = cpp.read_text()
    C = consts()
    entries = parse_entries(text, engine_names(text), C)
    header = header_shapes(model_dir)

    # §02.5's normalisation, asserted before it is used to look anything up.
    if "model.language_model.layers.0.linear_attn.conv1d.weight" not in header:
        raise SystemExit("the header does not contain what §02.5's rule predicts; check the rule")

    fails: list[str] = []
    seen: set = set()
    rows = []
    interval = json.loads((model_dir / "config.json").read_text())["text_config"][
        "full_attention_interval"]
    attn_layers = [i for i in range(C["kNumLayers"]) if (i + 1) % interval == 0]

    def layers_in(scope):
        if scope == "all":
            return range(C["kNumLayers"])
        if scope == "gdn":
            return [i for i in range(C["kNumLayers"]) if i not in attn_layers]
        if scope == "attn":
            return attn_layers
        return [None]

    for leaf_name, scope, rank, dims, dtype in entries:
        for L in layers_in(scope):
            ename = (f"lm.layers.{L}.{leaf_name}" if L is not None else leaf_name)
            if ename in seen:
                continue
            seen.add(ename)
            cname = to_ckpt(ename)
            if cname not in header:
                fails.append(f"{ename}: not in the shard header as {cname}")
                continue
            hdtype, hshape = header[cname]
            ok = hdtype == dtype and len(hshape) == rank and list(hshape) == list(dims)
            rows.append((ok, ename, dtype, rank, dims, hdtype, hshape))
            if not ok:
                fails.append(f"{ename}: engine claims {dtype} rank-{rank} {dims}, header says "
                             f"{hdtype} rank-{len(hshape)} {hshape}")

    for ok, ename, dtype, rank, dims, hdtype, hshape in sorted(rows, key=lambda r: r[1]):
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {ename:46s} {dtype} rank-{rank} {dims}   header: {hdtype} {hshape}")
    print(f"\n{len(rows)} engine names checked against {len(header)} header tensors "
          f"(claims read from {cpp.name})")

    # The census that makes the rank field meaningful, and the F32 identity §01.6 states. The GDN layer
    # list comes from config.json's own full_attention_interval, which is the third source for
    # state.h's `is_full_attention(l) { return l % 4 == 3; }`.
    gdn = [i for i in range(C["kNumLayers"]) if i not in attn_layers]
    ranks = {}
    for _, (d, s) in header.items():
        ranks[len(s)] = ranks.get(len(s), 0) + 1
    r3 = sorted({tuple(s) for _, (d, s) in header.items() if len(s) == 3})
    f32 = sorted(k for k, (d, _) in header.items() if d == "F32")
    want_f32 = sorted(
        [f"model.language_model.layers.{i}.linear_attn.A_log" for i in gdn]
        + [f"model.language_model.layers.{i}.linear_attn.norm.weight" for i in gdn])
    checks = [
        (r3 == [(6144, 1, 4)], f"the only rank-3 shape in the checkpoint is [6144,1,4] (got {r3})"),
        (len(f32) == 36, f"exactly 36 F32 tensors in the shard (got {len(f32)}) — §01.6"),
        (sorted(f32) == want_f32,
         "those 36 are exactly 18xA_log + 18xlinear_attn.norm.weight, no third kind"),
        (len(gdn) == C["kNumGdnLayers"] and len(attn_layers) == C["kNumAttnLayers"],
         f"config.json's interval {interval} gives the 18/6 split state.h hard-codes "
         f"(got {len(gdn)}/{len(attn_layers)})"),
        ("lm_head.weight" not in header and
         "model.language_model.lm_head.weight" not in header,
         "no lm_head.weight in the checkpoint, so §01.1's aliasing is the only option"),
    ]
    for ok, what in checks:
        print(("  [ok  ] " if ok else "  [FAIL] ") + what)
        if not ok:
            fails.append(what)
    print(f"  [info] rank census: {dict(sorted(ranks.items()))}")

    if fails:
        print(f"\nWEIGHTS CONTRACT FAIL — {len(fails)}")
        for f in fails:
            print("  - " + f)
        return 1
    print("\nWEIGHTS CONTRACT PASS — every shape engine/decode_step.cpp claims is what the shard holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
