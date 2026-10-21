#!/usr/bin/env python3
"""census_sites.py — what the 97 `cast` launches actually are, derived from engine source.

Route S1 item (1) deletes the per-token BF16->F32 widenings by folding the widen into the kernel that
consumes the result. Before any of that can be priced, the population has to be NAMED rather than
inherited: `prototype/launch_price/census.py` says "cast: 97 launches" and `engine/generation-limits.md`
§1 says the same, but neither says which 97, which buffer each writes, or which kernel reads it. The
foldable set is a property of the CONSUMERS, so the consumers have to be enumerated.

Three rules from AGENTS.md shape this file:

  * "a census must name the files it could read" — every row is read out of `engine/decode_step.cpp`
    (the call sites, their guards and their multiplicity) and `engine/state.h` (the widths). Nothing is
    typed from the census line: the 97 is re-derived and then asserted against it.
  * "a parser that can match nothing reports success over an empty list" — every derivation refuses on
    an empty or short match, and `--selftest` proves each one refuses by feeding it a deliberately broken
    copy of its own source.
  * "a traffic claim is a property of the loops the source contains" — the per-site BYTES are checked
    against `tools/step_traffic_model.py`'s own cast rows (consumed, never retyped), so a site that does
    not account for its share of 0.371/0.741 MB/token is a mismatch, not a rounding.

Sites are keyed by the CALL TEXT each one is written with, not by a line number: an earlier revision
keyed on lines and went red the moment the fold added comment lines above them, which is the same
brittleness every line-number citation in this tree has.

Exit 0 prints the table and `SITES ok`; anything else exits 1 with the reason.

    .venv/bin/python prototype/cast_fold/census_sites.py [--selftest]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STEP_CPP = ROOT / "engine" / "decode_step.cpp"
STATE_H = ROOT / "engine" / "state.h"
TRAFFIC = ROOT / "tools" / "step_traffic_model.py"

# The per-token cast sites. `pattern` is the call text as the source spells it and `pattern_total` how
# many times that text must appear: two sites widen `sc.norm -> sc.norm_f32` with identical call text
# (the mixer-input norm and the MLP norm) and so share one pattern with total 2. `per` is how many times
# one decode step — a ONE-row body — reaches the call:
#   layers  every layer, so kNumLayers times        gdn   the GDN mixers' site only, kNumGdnLayers
#   attn    the attention mixers' site only, kNumAttnLayers
#   once    `finish_token`, once per priced row
# `width` is the state.h constant naming how many elements one call widens.
SITES = [
    dict(name="norm_for_input_proj",
         pattern=r"cast_bf16_to_f32\(sc\.norm, sc\.norm_f32, n \* kHidden\)", pattern_total=2,
         per="layers", width="kHidden", src="sc.norm", dst="sc.norm_f32",
         consumer="proj_rows (grouped, engine/proj_group_kernels.h) on the shipped --proj-plan "
                  "packed; else project()'s hand branch -> proj_gemv",
         guard="!fold_cast && (!lib_gemm || !attn_layer)"),
    dict(name="attn_gated_for_o_proj",
         pattern=r"cast_bf16_to_f32\(sc\.a_gated, sc\.block_out_f32, \(size_t\)n \* kQwidth\)",
         pattern_total=1, per="attn", width="kQwidth", src="sc.a_gated", dst="sc.block_out_f32",
         consumer="project()'s hand branch -> proj_gemv (o_proj)",
         guard="!fold_cast && !lib_gemm"),
    dict(name="gdn_out_for_out_proj",
         pattern=r"cast_bf16_to_f32\(sc\.block_out, sc\.block_out_f32, n \* kValueDim\)",
         pattern_total=1, per="gdn", width="kValueDim", src="sc.block_out", dst="sc.block_out_f32",
         consumer="project()'s hand branch -> proj_gemv (out_proj)", guard="!fold_cast && !lib_gemm"),
    dict(name="norm_for_mlp_gateup",
         pattern=r"cast_bf16_to_f32\(sc\.norm, sc\.norm_f32, n \* kHidden\)", pattern_total=2,
         per="layers", width="kHidden", src="sc.norm", dst="sc.norm_f32",
         consumer="mlp_gateup (hand form; the library form reads sc.norm itself)",
         guard="!fold_cast && !lib_gemm"),
    dict(name="mlp_act_for_mlp_down",
         pattern=r"cast_bf16_to_f32\(sc\.mlp_act, sc\.mlp_act_f32, n \* kIntermediate\)",
         pattern_total=1, per="layers", width="kIntermediate", src="sc.mlp_act",
         dst="sc.mlp_act_f32", consumer="mlp_down (hand form)", guard="!fold_cast && !lib_gemm"),
    dict(name="final_norm_for_lm_head",
         pattern=r"cast_bf16_to_f32\(norm_row_ptr, sc\.norm_f32, kHidden\)", pattern_total=1,
         per="once", width="kHidden", src="norm_row_ptr (row of sc.norm)", dst="sc.norm_f32",
         consumer="lm_head_full", guard="none before the fold — submitted on every route"),
]
# Each pattern is claimed by as many sites as share it, so the multiplicity is checked once per pattern
# and not once per site (otherwise the shared `sc.norm` pattern would demand two occurrences each).
PATTERN_CLAIMS: dict[str, int] = {}
for _s in SITES:
    PATTERN_CLAIMS[_s["pattern"]] = _s["pattern_total"]


class CensusError(Exception):
    """A derivation that could not read what it expected. `die()` turns it into exit 1."""


def die(msg: str):
    raise CensusError(msg)


def shapes(state_txt: str | None = None) -> dict[str, int]:
    """The widths engine/state.h defines that the cast tiles are sized by."""
    txt = state_txt if state_txt is not None else STATE_H.read_text()
    # Read EVERY `constexpr int NAME = expr;` first, then resolve: `kValueDim` and `kQwidth` are
    # products of constants declared later in the file, so a name-by-name pass in the order this census
    # wants them cannot see its own dependencies — and hardcoding 2048 to get past that is exactly the
    # stale number this function exists to avoid.
    exprs: dict[str, str] = {}
    for name, expr in re.findall(r"constexpr int\s+(\w+)\s*=\s*([^;]+);", txt):
        exprs[name] = expr.split("//")[0].strip()
    out: dict[str, int] = {}
    pending = list(exprs)
    while pending:
        progressed = False
        for name in list(pending):
            expr = exprs[name]
            try:
                out[name] = int(expr)
            except ValueError:
                try:
                    out[name] = int(eval(expr, {"__builtins__": {}},
                                         {k: v for k, v in out.items() if k != name}))
                except Exception:  # noqa: BLE001  a dependency is unresolved yet; try next round
                    continue
            progressed = True
            pending.remove(name)
        if not progressed:
            die(f"circular or unresolvable state.h constants: {sorted(pending)}")
    wanted = ("kHidden", "kIntermediate", "kValueDim", "kQwidth",
              "kNumLayers", "kNumGdnLayers", "kNumAttnLayers")
    missing = [n for n in wanted if n not in out]
    if missing:
        die(f"engine/state.h does not define {missing}")
    return out


def split_sites(step_txt: str) -> tuple[list[int], list[int]]:
    """(per-token cast call lines, one-time adapter cast call lines).

    `prepare_weight_adapters()` widens `dt_bias` and `conv1d.weight` once per layer, once per PROCESS.
    Those are not per-token and are never counted in the 97, so they are separated by the function that
    contains them rather than by a remembered count.
    """
    src = step_txt.splitlines()
    hits = [i for i, t in enumerate(src, 1) if re.search(r"kernels::cast_bf16_to_f32\s*\(", t)]
    if not hits:
        die("no `kernels::cast_bf16_to_f32(` call in engine/decode_step.cpp — the file moved or the "
            "family is gone; re-read the source before trusting this table")
    starts = [(i, t) for i, t in enumerate(src, 1) if t.strip().startswith("void ")]
    adapter_start = next((i for i, t in starts if "prepare_weight_adapters" in t), None)
    body_start = next((i for i, t in starts if "submit_wave_body" in t), None)
    if adapter_start is None or body_start is None:
        die("cannot bound prepare_weight_adapters() / submit_wave_body() in the source")
    adapter = [h for h in hits if adapter_start < h < body_start]
    per_token = [h for h in hits if h not in adapter]
    if len(adapter) != 2:
        die(f"expected 2 one-time adapter widenings (dt_bias, conv1d.weight), found {adapter}")
    return per_token, adapter


def check_call_text(step_txt: str) -> dict[str, list[int]]:
    """Every site's call text must appear exactly as often as the table claims."""
    lines = step_txt.splitlines()
    found: dict[str, list[int]] = {}
    for pattern in PATTERN_CLAIMS:
        found[pattern] = [i for i, t in enumerate(lines, 1) if re.search(pattern, t)]
        if len(found[pattern]) != PATTERN_CLAIMS[pattern]:
            die(f"call text `{pattern}` appears {len(found[pattern])} time(s) in "
                f"engine/decode_step.cpp, the table claims {PATTERN_CLAIMS[pattern]} — a moved, "
                "duplicated or deleted widening changes the fold's population")
    total = sum(len(v) for v in found.values())
    if total != len(SITES):
        die(f"the sites' call text accounts for {total} calls, this census lists {len(SITES)}")
    for site in SITES:
        site["lines"] = found[site["pattern"]]
    return found


def launch_count(k: dict[str, int]) -> dict[str, int]:
    per = {"layers": k["kNumLayers"], "gdn": k["kNumGdnLayers"], "attn": k["kNumAttnLayers"],
           "once": 1}
    return {site["name"]: per[site["per"]] for site in SITES}


def traffic_cast_mb() -> tuple[float, float, str]:
    """The traffic model's own cast read/write MB per token, consumed not retyped."""
    if not TRAFFIC.exists():
        die(f"{TRAFFIC} missing; the byte side of the table has no source")
    res = subprocess.run([sys.executable, str(TRAFFIC)], capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        die(f"{TRAFFIC.name} exited {res.returncode}: {res.stderr.strip()[:200]}")
    out: dict[str, float] = {}
    for line in res.stdout.splitlines():
        m = re.match(r"^cast\s+(read|write)\s+([0-9]+\.[0-9]+)\s", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    if len(out) != 2:
        die(f"parsed cast traffic rows {sorted(out)} from {TRAFFIC.name}; expected read and write")
    return out["read"], out["write"], "tools/step_traffic_model.py"


def table(k: dict[str, int]) -> list[dict]:
    counts = launch_count(k)
    out = []
    for site in SITES:
        elems = counts[site["name"]] * k[site["width"]]
        out.append(dict(site, launches=counts[site["name"]], elements=elems,
                        bytes_read=elems * 2, bytes_written=elems * 4))
    return out


def render(rows: list[dict]) -> tuple[str, float, float, int]:
    lines = []
    hdr = (f"{'site':<26}{'src -> dst':<46}{'elems':>8}{'launches':>9}{'read MB':>10}"
           f"{'write MB':>10}  guard")
    lines += [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['name']:<26}{r['src'] + ' -> ' + r['dst']:<46}{r['elements']:>8}"
                     f"{r['launches']:>9}{r['bytes_read'] / 1e6:>10.3f}"
                     f"{r['bytes_written'] / 1e6:>10.3f}  {r['guard']}")
    lines.append("-" * len(hdr))
    lines.append(f"{'TOTAL per decode token':<72}{sum(r['elements'] for r in rows):>8}"
                 f"{sum(r['launches'] for r in rows):>9}"
                 f"{sum(r['bytes_read'] for r in rows) / 1e6:>10.3f}"
                 f"{sum(r['bytes_written'] for r in rows) / 1e6:>10.3f}")
    return ("\n".join(lines), sum(r["bytes_read"] for r in rows) / 1e6,
            sum(r["bytes_written"] for r in rows) / 1e6, sum(r["launches"] for r in rows))


def selftest() -> int:
    """Each derivation, fed a source that must make it refuse. A check that cannot fail is decoration."""
    good_state = STATE_H.read_text()
    good_step = STEP_CPP.read_text()
    cases = [
        ("state.h with no kHidden", lambda: shapes(good_state.replace(
            "constexpr int kHidden = 1024;", ""))),
        ("state.h with an unresolvable product", lambda: shapes(good_state.replace(
            "constexpr int kValueDim = kNumValueHeads * kDv;",
            "constexpr int kValueDim = not_a_name * kDv;"))),
        ("decode_step.cpp with the cast calls renamed away", lambda: split_sites(
            good_step.replace("kernels::cast_bf16_to_f32(", "kernels::not_a_cast("))[0]),
        ("decode_step.cpp missing one per-token call", lambda: check_call_text(
            good_step.replace("kernels::cast_bf16_to_f32(sc.mlp_act, sc.mlp_act_f32", "// gone"))),
        ("decode_step.cpp with one widening duplicated", lambda: check_call_text(
            good_step + "\n// duplicated widening\nvoid dup() { kernels::cast_bf16_to_f32("
            "sc.mlp_act, sc.mlp_act_f32, n * kIntermediate); }\n")),
        ("decode_step.cpp with the norm widening's count changed", lambda: check_call_text(
            good_step.replace("cast_bf16_to_f32(sc.norm, sc.norm_f32, n * kHidden)",
                              "cast_bf16_to_f32(sc.norm, sc.norm_f32, n * kQwidth)"))),
        ("a source with no boundable adapter function", lambda: split_sites(
            good_step.replace("void prepare_weight_adapters(", "void renamed_adapters("))),
    ]
    failed = 0
    for what, fn in cases:
        try:
            fn()
        except CensusError as exc:
            print(f"  selftest refused as required  [{what}]  -> {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  SELFTEST FAIL [{what}] raised {type(exc).__name__} instead of CensusError: {exc}")
        else:
            failed += 1
            print(f"  SELFTEST FAIL [{what}] returned normally — that check cannot fail")
    if len(cases) != 7:
        print(f"SELFTEST FAIL: expected 7 broken-source cases, built {len(cases)}")
        failed += 1
    print("SELFTEST PASS" if failed == 0 else f"SELFTEST FAIL: {failed} of {len(cases)}")
    return 0 if failed == 0 else 1


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return selftest()
    k = shapes()
    step_txt = STEP_CPP.read_text()
    per_token, adapter = split_sites(step_txt)
    check_call_text(step_txt)
    rows = table(k)
    body, read_mb, write_mb, total = render(rows)
    print(f"layers: kNumLayers={k['kNumLayers']} kNumGdnLayers={k['kNumGdnLayers']} "
          f"kNumAttnLayers={k['kNumAttnLayers']}  (engine/state.h)")
    print(f"widths: kHidden={k['kHidden']} kValueDim={k['kValueDim']} kQwidth={k['kQwidth']} "
          f"kIntermediate={k['kIntermediate']}")
    print(f"per-token cast calls read from engine/decode_step.cpp at lines {per_token}; the "
          f"{len(adapter)} one-time weight-adapter widenings at lines {adapter} are NOT per-token and "
          "are not counted here")
    print()
    print(body)
    print()
    for r in rows:
        # Two sites widen `sc.norm -> sc.norm_f32` with identical call text, so a per-site line number would
        # print the same pair twice and read as a duplicate row; the sharing is stated instead.
        shared = PATTERN_CLAIMS[r["pattern"]] > 1
        where = ("shares its call text with the other `sc.norm -> sc.norm_f32` site"
                 if shared else f"line {r['lines'][0]}")
        print(f"  {r['name']} ({where}): consumer = {r['consumer']}")
    print()
    problems = []
    if total != 97:      # [Q1] engine/generation-limits.md §1, prototype/launch_price/census.py
        problems.append(f"the six sites bill {total} launches/token, the §07 bill is 97")
    tmr, tmw, src = traffic_cast_mb()
    for label, mine, theirs in (("read", read_mb, tmr), ("write", write_mb, tmw)):
        if abs(mine - theirs) > 0.005:
            problems.append(f"cast {label} {mine:.3f} MB/token != {src}'s {theirs:.3f}")
    if problems:
        for p in problems:
            print("CENSUS FAIL: " + p)
        return 1
    print(f"SITES ok: 6 call sites, 97 launches/token, {read_mb:.3f} MB read / {write_mb:.3f} MB "
          f"written/token — agrees with engine/generation-limits.md §1 and {src}.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CensusError as e:
        print(f"CENSUS FAIL: {e}")
        sys.exit(1)
