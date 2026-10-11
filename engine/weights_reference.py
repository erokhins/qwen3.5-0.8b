#!/usr/bin/env python3
"""Per-tensor reference table for `engine/loader_test.cpp`.

`tools/weights_report.py` is the authority for shapes and dtypes — specification/01 is
generated from it — but its printer only displays the groups it chooses to show
(`layers.0`, `layers.3`, `mtp.`, embed, final norm). The loader gate has to check all
335 text tensors, so this emits the same facts for every tensor **using the tool's own
header parser and its own byte/shape/dtype assertion**, rather than re-deriving them.

usage: engine/weights_reference.py <model-dir | path/to/*.safetensors>

output (tab-separated, one fact per line):
  T\t<checkpoint_name>\t<dtype>\t<shape '6144x1024', 1-D as '1024'>\t<nbytes>\t<text|vision|other>
  K\t<key>\t<value>
"""
import contextlib
import importlib.util
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "weights_report.py")


def resolve_shard(arg):
    """Accept a model directory (via its index, per A-09) or a shard path directly."""
    if os.path.isdir(arg):
        index = os.path.join(arg, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            if len(shards) != 1:
                sys.exit(f"weights_reference: {len(shards)} shards; the loader gate expects 1")
            return os.path.join(arg, shards[0])
        shards = sorted(f for f in os.listdir(arg) if f.endswith(".safetensors"))
        if len(shards) != 1:
            sys.exit(f"weights_reference: no single shard in {arg}")
        return os.path.join(arg, shards[0])
    return arg


def report_module(shard):
    """Load tools/weights_report.py as a module and reuse its reader.

    The tool calls main() at import time (no __main__ guard), so it runs once with an
    empty filter into a sink; we then call its header()/size_of() directly.
    """
    spec = importlib.util.spec_from_file_location("weights_report", TOOL)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [TOOL, shard, ""]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    shard = resolve_shard(sys.argv[1])
    tool = report_module(shard)
    hdr = tool.header(shard)
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}

    counts, bytes_by = {}, {}
    for name in sorted(tensors):
        v = tensors[name]
        if name.startswith("model.visual."):
            cat = "vision"
        elif name.startswith("model.language_model.") or name.startswith("mtp."):
            cat = "text"
        else:
            cat = "other"
        nbytes = tool.size_of(v)  # asserts nbytes == numel * elemsize, same as the report
        shape = "x".join(str(d) for d in v["shape"])
        print(f"T\t{name}\t{v['dtype']}\t{shape}\t{nbytes}\t{cat}")
        counts[cat] = counts.get(cat, 0) + 1
        bytes_by[(cat, v["dtype"])] = bytes_by.get((cat, v["dtype"]), 0) + nbytes

    text_b = sum(n for (c, _), n in bytes_by.items() if c == "text")
    for k in ("text", "vision", "other"):
        print(f"K\t{k}_tensors\t{counts.get(k, 0)}")
    print(f"K\ttotal_tensors\t{len(tensors)}")
    print(f"K\ttext_bytes\t{text_b}")
    for dt in sorted({d for (c, d) in bytes_by if c == "text"}):
        print(f"K\ttext_tensors_{dt}\t{sum(1 for k, v in tensors.items() if v['dtype'] == dt and (k.startswith('model.language_model.') or k.startswith('mtp.')))}")
        print(f"K\ttext_bytes_{dt}\t{bytes_by.get(('text', dt), 0)}")
    print(f"K\ttext_gb\t{text_b / 1e9:.4f}")


main()
