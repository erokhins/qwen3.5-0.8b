"""The one place Python knows how a checkpoint name becomes an engine name (§02.5).

Why this file exists. `engine::WeightStore` is keyed on the **normalised** name: `at()`/`find()`
answer for `lm.layers.14.linear_attn.norm.weight`. Every weight *reader* in `prototype/` instead
opens the checkpoint with the **raw** key `model.language_model.layers.14.linear_attn.norm.weight`.
Both spellings are correct, for different jobs — the raw one reads the file, the normalised one asks
the store for the resident blob — and nothing in the type system keeps a plan writer from using the
wrong one. The first device run of `prototype/gdn_gated_norm` found out by aborting inside a
subprocess:

    terminate called after throwing an instance of 'std::out_of_range'
      what():  WeightStore: no tensor named model.language_model.layers.0.linear_attn.norm.weight

`at()` deliberately refuses instead of normalising what it is handed, and that is the right
behaviour: a lookup that tolerated both spellings would answer for tensors the store does not hold,
and would turn a naming mistake into a silent pointer to the wrong blob (two spellings of one name,
one resolution path, and a name it could not normalise becoming an error only at the third decimal
place of a cosine). So the fix belongs on the **producer** side — and the producer is Python, which
therefore needs these rules. `engine/loader.cpp::normalize_name` remains the authority; if this
mirror and that function ever disagree, that one wins and this file is wrong.

`store_names()` applies the same map to the whole checkpoint, which makes a plan checkable on the
CPU before any subprocess is launched (`check_plan`). It reads `model.safetensors.index.json`, not
the shard header, because the header parser lives in `loader.cpp` behind a `load()` that hipMallocs
and workers do not touch the device (`AGENTS.md`). For this checkpoint the two lists are the same
list: the index resolves to exactly one shard (`A-09`) and the loader classifies that shard's header
key list, which is what `weight_map` is. The tie is independent and asserted in `selftest()`: the
index holds 488 names, 335 load + 153 vision is §01.7's split as pinned by `engine/loader_test.cpp`,
and the store's key set is 335 + the tied `lm_head.weight` (§01.1) = 336, which is
`loader_test.cpp:362`'s `lookup_count() == n_loaded + 1`.

    .venv/bin/python prototype/weight_names.py     # selftest, no device
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "Qwen3.5-0.8B"

# Mirrors `engine::TensorPlan`.
LOAD, SKIP_VISION, UNKNOWN = "load", "skip_vision", "unknown"

# §01.1: the checkpoint carries no `lm_head.weight` — the loader registers it as a second *name*
# for the embedding allocation. No index entry can produce it, so the guard adds it here; leaving
# it out would make every engine plan that reads the head fail a check the store would pass.
ENGINE_ALIASES = frozenset({"lm_head.weight"})

# §01.7 / §10 gate G0, pinned independently by engine/loader_test.cpp:231 ("loaded text tensors
# = 335") and :362 (`lookup_count() == n_loaded + 1`, the +1 being the tied head). A move here means
# either the checkpoint changed or the Python mirror drifted from loader.cpp — both are failures,
# and neither is a number to edit.
EXPECTED_TEXT = 335
EXPECTED_VISION = 153
EXPECTED_KEYS = EXPECTED_TEXT + len(ENGINE_ALIASES)


def normalize_name(checkpoint_name: str) -> tuple[str, str]:
    """(plan, engine_name) for one checkpoint name — the §02.5 classification.

    `plan` is LOAD / SKIP_VISION / UNKNOWN; `engine_name` is non-empty only for LOAD. A name that is
    neither in the load set nor an explicitly skipped vision tensor is UNKNOWN, and the loader fails
    the whole load on one (gate G0) rather than skipping it.
    """
    if checkpoint_name.startswith("model.visual."):
        return SKIP_VISION, ""
    if checkpoint_name.startswith("model.language_model."):
        return LOAD, "lm." + checkpoint_name[len("model.language_model."):]
    if checkpoint_name.startswith("mtp.layers.0."):
        return LOAD, "mtp.layer0." + checkpoint_name[len("mtp.layers.0."):]
    if checkpoint_name.startswith("mtp."):
        return LOAD, checkpoint_name
    return UNKNOWN, ""


def engine_layer(layer: int) -> str:
    """Engine prefix for a decoder layer — the plan-writer's half of `LAYER_PREFIX`."""
    return "lm.layers.%d." % layer


def checkpoint_layer(layer: int) -> str:
    """Checkpoint prefix for a decoder layer, i.e. what `safetensors` needs."""
    return "model.language_model.layers.%d." % layer


@lru_cache(maxsize=4)
def index_names(model_dir: str | Path = CKPT) -> tuple[str, ...]:
    """Every checkpoint name the shards carry, read from `model.safetensors.index.json` (A-09).

    Sorted, so a failure prints the same way twice.
    """
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.exists():
        raise SystemExit(
            "weight_names: no %s. This guard reads the index, not the shard header (module "
            "docstring says why); tools/weights_report.py has a header parser if a header-based "
            "fallback is ever needed." % index)
    return tuple(sorted(json.loads(index.read_text())["weight_map"]))


def store_names(model_dir: str | Path = CKPT) -> frozenset[str]:
    """Exactly the names `WeightStore::at()` answers for, derived without touching the device."""
    known: set[str] = set()
    for name in index_names(model_dir):
        plan, norm = normalize_name(name)
        if plan == UNKNOWN:
            raise SystemExit(
                "weight_names: %s is neither in the §02.5 load set nor a vision tensor; the loader "
                "fails on it too (gate G0), so the §02.5 mirror needs updating." % name)
        if plan == LOAD:
            known.add(norm)
    return frozenset(known | ENGINE_ALIASES)


def check_names(names, *, what: str, known: frozenset[str] | None = None) -> int:
    """Stop now, naming the offenders, instead of later with a SIGABRT from inside a subprocess.

    Returns the number of names checked. Prints one summary line on success so a gate log shows the
    check ran and against how many keys.
    """
    known = store_names() if known is None else known
    wanted = list(names)
    bad = sorted({n for n in wanted if n not in known})
    if bad:
        raise SystemExit(
            "%s asks for %d weight name(s) no WeightStore holds:\n  %s\n"
            "The store is keyed on the §02.5 normalised name (`lm.layers.N.*`), not the checkpoint "
            "key (`model.language_model.layers.N.*`) — see prototype/weight_names.py."
            % (what, len(bad), "\n  ".join(bad)))
    print("  %s: %d weight name(s), all present in the store's %d-name key set"
          % (what, len(wanted), len(known)))
    return len(wanted)


def check_plan(path, weight_field: int, *, known: frozenset[str] | None = None) -> int:
    """Every weight name a plan file asks for must be a store key. CPU only, no allocation.

    `weight_field` is the whitespace-separated column holding the name, passed explicitly so a plan
    format change cannot make this silently read a file path and pass. The two plans that resolve
    weights through the store agree on column 3 today:
    glue is `NAME SFILE DFILE WNAME OUT…` and gdn_gated_norm is `NAME OFILE ZFILE WNAME OUT
    [NORM]`. Plans that carry weight *file paths* (proj_gemv, attn_decode, mlp_gemv, lm_head_gemv,
    gdn_decode) have no name column and must not be pointed at this.
    """
    names = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) <= weight_field:
            raise SystemExit("%s line %d: only %d field(s), so the weight name at column %d is "
                             "missing — not a naming problem, a plan-format problem"
                             % (path, lineno, len(fields), weight_field + 1))
        names.append(fields[weight_field])
    return check_names(names, what="plan %s" % path, known=known)


def selftest() -> int:
    """The §02.5 rules by literal, the tied head, §01.7's split off the real index, and a guard that
    is required to fire — a check that cannot fail is the same hole as no check."""
    rules = [
        # The name whose raw form started this: the checkpoint key and the engine key, both.
        ("model.language_model.layers.14.linear_attn.norm.weight", LOAD,
         "lm.layers.14.linear_attn.norm.weight"),
        ("model.language_model.embed_tokens.weight", LOAD, "lm.embed_tokens.weight"),
        ("model.language_model.norm.weight", LOAD, "lm.norm.weight"),
        ("mtp.layers.0.self_attn.q_proj.weight", LOAD, "mtp.layer0.self_attn.q_proj.weight"),
        ("mtp.fc.weight", LOAD, "mtp.fc.weight"),
        ("model.visual.blocks.0.attn.proj.weight", SKIP_VISION, ""),
        ("totally.unknown.tensor", UNKNOWN, ""),
    ]
    bad_rules = [(n, normalize_name(n)) for n, plan, norm in rules
                 if normalize_name(n) != (plan, norm)]

    names = index_names()
    n_text = sum(1 for n in names if normalize_name(n)[0] == LOAD)
    n_vision = sum(1 for n in names if normalize_name(n)[0] == SKIP_VISION)
    known = store_names()
    # Nothing raw may survive into the key set: `lm.`/`mtp.` only, plus the engine-side head alias.
    leaked = sorted(n for n in known
                    if not (n.startswith("lm.") or n.startswith("mtp.")
                            or n in ENGINE_ALIASES))

    # The guard's own positive-and-negative pair, on a scratch plan so it stays on the CPU: the
    # normalised name must be accepted, the checkpoint key that started this must be refused *by
    # name*, and a short line must be reported as a format problem rather than skipped.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), tempfile.TemporaryDirectory() as td:
        def try_plan(body: str):
            p = Path(td) / "plan.txt"
            p.write_text(body)
            try:
                check_plan(p, 3)
                return "passed"
            except SystemExit as e:
                return str(e)
        guard_good = try_plan("c000 in.o in.z lm.layers.14.linear_attn.norm.weight out.bin\n")
        guard_bad = try_plan("c000 in.o in.z "
                             "model.language_model.layers.14.linear_attn.norm.weight out.bin\n")
        guard_short = try_plan("c000 in.o in.z\n")
        guard_ok = (guard_good == "passed"
                    and "model.language_model.layers.14" in guard_bad
                    and "plan-format" in guard_short)

    ok = (not bad_rules and not leaked and guard_ok
          and n_text == EXPECTED_TEXT and n_vision == EXPECTED_VISION
          and len(known) == EXPECTED_KEYS
          and len(names) == EXPECTED_TEXT + EXPECTED_VISION
          and "lm_head.weight" in known and "lm.embed_tokens.weight" in known
          and "lm.layers.14.linear_attn.norm.weight" in known)
    print("weight_names selftest: %d/%d §02.5 rules wrong | index %d names = %d text + %d vision "
          "(want %d + %d) | store keys %d (want %d, tied lm_head present) | raw names leaked: %s"
          " | guard: accepts normalised=%s, refuses checkpoint key=%s, flags short line=%s -> %s"
          % (len(bad_rules), len(rules), len(names), n_text, n_vision,
             EXPECTED_TEXT, EXPECTED_VISION, len(known), EXPECTED_KEYS, leaked or "none",
             guard_good == "passed", "model.language_model.layers.14" in guard_bad,
             "plan-format" in guard_short, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
