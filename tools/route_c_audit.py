#!/usr/bin/env python3
"""The route (c) audit's arithmetic — A1/A2/A3/A4 of `bench/plan-route-c-audit-2026-001.md`.

Everything here is element-level and bounded against a measured floor, which is the standard the plan
sets ("not 'the aggregates look small'"). No number in this file is a band and no band is set here:
`engine/gate.py` and `engine/route_allowance.py` own bands, and the plan says the A1 columns are the
bound that the rest get judged against, so they are written to a JSON in the audit's own work dir for
the parent to price from — never into a pin or an allowance file.

Every BF16 distance comes from `prototype/bf16_ulp.py`, whose `selftest()` pins the ordering
transform. That is deliberate: a hand-rolled `uint16` comparator here would have the dead-branch sign
bug AGENTS.md records, and it would pass vacuously instead of failing.

    .venv/bin/python tools/route_c_audit.py prompt --tokens 1024      # the audit prompt (stdout)
    .venv/bin/python tools/route_c_audit.py a1 tmp/route_c/floor       # A1 per-shape floors
    .venv/bin/python tools/route_c_audit.py a3 tmp/route_c             # A3 per-stage diff
    .venv/bin/python tools/route_c_audit.py a4 tmp/route_c/floor       # A4 negative control
    .venv/bin/python tools/route_c_audit.py a2-run tmp/route_c/a2      # device: 8 fixtures x 2 routes
    .venv/bin/python tools/route_c_audit.py a2 tmp/route_c/a2 dec8     # A2 per-step diff

CORRECTNESS ONLY. Nothing here times anything, and no output of this file may be quoted in a timing
document (AGENTS.md: worker device runs take no timings).

The one definition each column needs, stated rather than implied:
  * A BF16 ULP is `prototype/bf16_ulp.py`'s: adjacent representable BF16 values are 1 apart, across
    zero included.
  * Logits dumps are F32 (`engine/prefill.h`'s §03.4 full-logit form). The ULP columns for a logits
    row are BF16 ULPs OF THE VALUES ROUNDED TO BF16 — the width a pin carries — and the exact-F32
    column is printed beside them so the rounding step is visible instead of baked in.
  * "near-cancellation" means |value| <= 1e-4 x max|ref|, which is `bf16_ulp.band_usage`'s default
    band. It is the plan's rounding signature; a large-magnitude offender is the bug signature.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))

import bf16_ulp  # the selftested comparator (prototype/bf16_ulp.py, run its selftest first)

REFS = ROOT / "data" / "refs"
# The eight gated fixtures, in `engine/slack_size.py`'s order — the same set the route's gate table
# is taken over.
FIXTURES = ["dec8", "dec64f", "dec64t", "dec300f", "mt32f", "p128", "wide520f", "chunk64b"]
# A-18's bound, read from the gate rather than retyped: past it the gate teacher-forces with
# `--replay`, and the audit's two sides have to be fed identically or a difference could be a
# different token stream rather than a different GEMM.
FREE_RUN_BOUND = 8

TENSOR_DTYPES = {"BF16": 2, "F32": 4}


# ------------------------------------------------------------------ loading bits ---------------
def read_raw(path: Path, dtype: str) -> np.ndarray:
    """Raw dump → (uint16 bit patterns | float32 values), read as the width it was written in."""
    raw = path.read_bytes()
    if dtype == "BF16":
        return np.frombuffer(raw, dtype="<u2").astype(np.uint16)
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def bf16_bits_to_tensor(bits: np.ndarray) -> torch.Tensor:
    """uint16 bit patterns → a BF16 tensor, by VIEW (a `.to(torch.uint16)` would cast, not bitcast)."""
    return torch.from_numpy(np.ascontiguousarray(bits, dtype=np.uint16)).view(torch.bfloat16)


def f32_to_tensor(vals: np.ndarray) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32)


# --------------------------------------------------------------------- statistics --------------
def hist(d: np.ndarray) -> dict:
    """ULP histogram in the plan's buckets: 0 / 1 / 2 / 3 / >3."""
    h = {"0": int((d == 0).sum()), "1": int((d == 1).sum()), "2": int((d == 2).sum()),
         "3": int((d == 3).sum()), ">3": int((d > 3).sum())}
    return h


def bf16_compare(bits_got: np.ndarray, bits_ref: np.ndarray, band: int | None = None,
                 cancel_rel: float = 1e-4, ulp_max: int = 1) -> dict:
    """Element-level BF16 comparison of two dumps, `ref` being the reference side.

    TWO max-ULP numbers are reported because they mean different things, and conflating them is the
    trap `prototype/mlp_gemv/floor_probe.py` and §10.1 F1–F4 write down:

      `max_ulp`      — the raw maximum, wherever it lands. On a long dot that is a element that
                       cancelled to ~1e-8 of the tensor max, where the two F32 summation orders can
                       even disagree about the SIGN (measured: +1.9e-06 vs −7.2e-07 at rel 4.5e-08 of
                       a max of 42, which is ~27 000 "ULP" and is nothing but legal rounding). Used
                       raw, it is a band nothing can fail.
      `max_ulp_large`— the maximum over elements with |ref| > `cancel_rel` × max|ref|, i.e. the
                       elements big enough for a ULP to mean something. THIS is the usable floor,
                       and `n_bad` (offenders outside the band) is what a gate can be judged on.
                       `bf16_ulp.report` computes both; this function adds the histogram and the
                       index-of-max detail the plan asks for and delegates the band arithmetic to
                       `bf16_ulp.band_usage` rather than restating it.

    `band` is the A1 floor this comparison is being judged against, when one exists; it is reported as
    counts and percentages, never as a verdict — the verdict is the parent's band call.
    """
    got = bf16_bits_to_tensor(bits_got)
    ref = bf16_bits_to_tensor(bits_ref)
    d = bf16_ulp.ulp_dist(got, ref).ravel()
    gv = got.float().numpy().astype(np.float64).ravel()
    rv = ref.float().numpy().astype(np.float64).ravel()
    scale = float(np.abs(rv).max()) if rv.size else 0.0
    rel = np.abs(rv) / scale if scale else np.zeros_like(rv)
    i = int(np.argmax(d)) if d.size else 0
    large = rel > cancel_rel
    off = d > ulp_max
    out = {
        "n": int(d.size), "ndiff": int((d > 0).sum()),
        "pct_diff": float((d > 0).mean()) if d.size else 0.0,
        "hist": hist(d),
        "max_ulp": int(d.max()) if d.size else 0,
        "argmax": i,
        "ref_at_max": float(rv[i]) if d.size else 0.0,
        "got_at_max": float(gv[i]) if d.size else 0.0,
        "rel_at_max": float(rel[i]) if d.size else 0.0,
        "n_large": int(large.sum()),
        "max_ulp_large": int(d[large].max()) if large.any() else 0,
        "n_bad": int((off & ~large).sum() if scale else 0),
        "n_off_large": int((off & large).sum()),
        "worst_off_rel": float(rel[off].max()) if off.any() else 0.0,
        "n_band": int((~large).sum()),
        "meanrel": float(np.abs(gv - rv).mean() / np.abs(rv).mean()) if rv.size else 0.0,
        "cos": float(np.dot(gv, rv) / (np.linalg.norm(gv) * np.linalg.norm(rv)))
               if rv.size and np.linalg.norm(gv) and np.linalg.norm(rv) else 1.0,
        "n_nan_got": bf16_ulp.zero_nan(got)[1], "n_nan_ref": bf16_ulp.zero_nan(ref)[1],
    }
    if band is not None:
        out["band_ulp"] = band
        out["n_over_band"] = int((d > band).sum())
        out["pct_over_band"] = float((d > band).mean()) if d.size else 0.0
        out["n_over_band_large"] = int(((d > band) & large).sum())
        out["worst_off_band_rel"] = float(np.where(d > band, rel, 0.0).max()) \
            if (d > band).any() else 0.0
    return out


def f32_compare(got: np.ndarray, ref: np.ndarray) -> dict:
    """F32 stage tiles: no BF16 ULP is meaningful on a genuinely-F32 tensor, so say so in the shape
    of the row — exact difference count, max relative error, cos and meanrel."""
    d = got - ref
    scale = float(np.abs(ref).max()) if ref.size else 0.0
    denom = np.where(np.abs(ref) > 0, np.abs(ref), np.inf)
    return {
        "n": int(ref.size), "ndiff": int((d != 0).sum()),
        "pct_diff": float((d != 0).mean()) if ref.size else 0.0,
        "max_abs": float(np.abs(d).max()) if d.size else 0.0,
        "max_rel": float((np.abs(d) / denom).max()) if ref.size else 0.0,
        "max_abs_over_scale": float(np.abs(d).max() / scale) if scale else 0.0,
        "meanrel": float(np.abs(d).mean() / np.abs(ref).mean()) if ref.size else 0.0,
        "cos": float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
               if np.linalg.norm(got) and np.linalg.norm(ref) else 1.0,
    }


def logit_row_decision(got: np.ndarray, ref: np.ndarray) -> dict:
    """What the row is actually used for: the greedy decision. `gap` is top1 minus top2, i.e. how
    much room the decision has. The rounding signature at the decision level is 'every disagreement
    sits on a thin gap, every fat gap agrees'; a bug flips a decision that had room."""
    def top(vals: np.ndarray):
        i = int(np.argmax(vals))
        v = vals[i]
        second = float(np.max(np.delete(vals, i))) if vals.size > 1 else float("-inf")
        return i, float(v), second

    gi, gv, g2 = top(got)
    ri, rv, r2 = top(ref)
    return {"argmax_got": gi, "argmax_ref": ri, "argmax_match": gi == ri,
            "gap_got": gv - g2, "gap_ref": rv - r2}


def logit_compare(got: np.ndarray, ref: np.ndarray, band: int | None = None,
                  cancel_rel: float = 1e-4) -> dict:
    """A logits row: F32 exactness AND the BF16-ULP view of the same values.

    The logits are a 24-layer cascade's output, so the plan's own discriminator — is the offender a
    cancelled element (rounding) or a large one (bug) — is carried per row as `n_bad` /
    `worst_off_rel` rather than decided here.
    """
    out = f32_compare(got, ref)
    out["pct_diff_f32"] = out["pct_diff"]
    out["ndiff_f32"] = out["ndiff"]
    out["meanrel_f32"] = out["meanrel"]
    out["cos_f32"] = out["cos"]
    gb = f32_round_bits(got)
    rb = f32_round_bits(ref)
    for k, v in bf16_compare(gb, rb, band=band, cancel_rel=cancel_rel).items():
        out[k] = v
    return out


def f32_round_bits(vals: np.ndarray) -> np.ndarray:
    """F32 values → BF16 bit patterns, by the same round-to-nearest-even torch applies."""
    return torch.tensor(vals, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy()# ------------------------------------------------------------------------- A1 ------------------
def a1(floor_dir: Path, out_json: Path | None) -> int:
    """A1: the isolated per-shape floor table, at each width the plan names.

    `maxULP` is the raw maximum and `value@max`/`rel@max` say what kind of element it sits on; every
    one of these raw maxima lands on a cancelled element, which is why `maxULP(large)` — the maximum
    over elements above the 1e-4 cancellation band — `>1ULP_large` (how many large elements moved by
    more than 1 ULP), `n_bad` (offenders inside the band) and `worst_off_rel` (the rel of the worst
    offender, i.e. the smallest band under which nothing offends) are the columns a downstream band
    has to be written against. All of them are in the JSON.
    """
    rows = []
    # `inj_*` bins are A4's injected-bug material, not a legal floor; `a4` lists them.
    names = sorted({re.sub(r"@\d+_(hand|lib)\.bin$", "", p.name)
                    for p in floor_dir.glob("*@*_hand.bin")
                    if not p.name.startswith("inj_")})
    print("A1 — isolated per-shape reassociation floor (hand kernel vs hipBLAS, real weight,"
          " real activation)\n")
    print(f"{'shape':<16} {'M':>5} {'n':>9} {'%diff':>7} {'1':>7} {'2':>5} {'3':>4} {'>3':>4}"
          f" {'maxULP':>7} {'value@max':>12} {'rel@max':>9} {'maxULPlarge':>11}"
          f" {'>1ULP_large':>11} {'n_bad':>6} {'worst_off_rel':>13} {'meanrel':>9} {'cos':>11}")
    for name in names:
        for m in (1, 88, 512, 1024):
            hand = floor_dir / f"{name}@{m}_hand.bin"
            lib = floor_dir / f"{name}@{m}_lib.bin"
            if not (hand.exists() and lib.exists()):
                continue
            st = bf16_compare(read_raw(hand, "BF16"), read_raw(lib, "BF16"))
            rows.append({"shape": name, "m": m, **st})
            h = st["hist"]
            print(f"{name:<16} {m:>5} {st['n']:>9} {100 * st['pct_diff']:>6.3f}% {h['1']:>7}"
                  f" {h['2']:>5} {h['3']:>4} {h['>3']:>4} {st['max_ulp']:>7}"
                  f" {st['ref_at_max']:>12.4g} {st['rel_at_max']:>9.2e} {st['max_ulp_large']:>11}"
                  f" {st['n_off_large']:>11} {st['n_bad']:>6} {st['worst_off_rel']:>13.3e}"
                  f" {st['meanrel']:>9.2e} {st['cos']:>11.9f}")
    if out_json:
        out_json.write_text(json.dumps({"a1": rows}, indent=1))
        print(f"\nwrote {out_json} — the floor columns the parent prices bands from. Nothing here"
              " touches engine/gate.py, an allowance file, or a pin.")
    return 0


# ------------------------------------------------------------------------- A3 ------------------
def a3(dump_dir: Path, a1json: Path | None) -> int:
    """A3: every stage tile, hand vs library, at one fixed body, judged against that width's A1 floor.

    The floor used here is A1's `max_ulp_large` at the same M (the plan: "every stage within 2× its A1
    floor"), and it is applied to the band-excluded elements for the same reason A1 reports them: a
    raw max is a cancelled element and cancels the check.
    """
    manifest = json.loads((dump_dir / "stages.json").read_text())
    floors: dict[int, int] = {}
    if a1json and a1json.exists():
        for r in json.loads(a1json.read_text())["a1"]:
            if not r["shape"].startswith("inj_"):
                floors[r["m"]] = max(floors.get(r["m"], 0), r["max_ulp_large"])
    hand, lib = dump_dir / "hand", dump_dir / "library"
    for body in sorted(p.name for p in hand.glob("body*")):
        rows = manifest["rows"]
        band = floors.get(rows)
        print(f"\nA3 — per-stage element diff, body {body} ({rows} rows/body,"
              f" {manifest['activation_rows']} activation rows captured). A tile holds its LAST"
              f" writer within the body: layer 23 for the attention and MLP tiles, layer 22 for the"
              f" GDN ones. Band = 2 x A1 max_ulp_large at M={rows} = "
              f"{'none (no --a1json)' if band is None else f'{2 * band} ULP (floor {band})'}\n")
        print(f"{'stage':<14} {'dtype':<5} {'n':>9} {'%diff':>7} {'maxULP':>7}"
              f" {'maxULPlarge':>11} {'>band':>6} {'n_bad':>6} {'worst_off_rel':>13}"
              f" {'meanrel':>9} {'cos':>11}")
        rows_seen = []
        for stage in manifest["stages"]:
            name, dt = stage["name"], stage["dtype"]
            hp, lp = hand / body / f"{name}.bin", lib / body / f"{name}.bin"
            if not (hp.exists() and lp.exists()):
                print(f"{name:<14} {dt:<5} {'MISSING DUMP':>9}")
                continue
            if dt == "BF16" and name in LIBRARY_ONLY_STAGES:
                note = ("library-only tile: the hand route never writes it, so the hand dump is"
                        " uninitialised memory - not a comparison")
                print(f"{name:<14} {dt:<5} {stage['rows'] * stage['cols']:>9}  {note}")
                continue
            if dt == "BF16":
                st = bf16_compare(read_raw(hp, dt), read_raw(lp, dt), band=2 * band if band else None)
                rows_seen.append((name, st["meanrel"]))
                print(f"{name:<14} {dt:<5} {st['n']:>9} {100 * st['pct_diff']:>6.3f}%"
                      f" {st['max_ulp']:>7} {st['max_ulp_large']:>11}"
                      f" {st.get('n_over_band', 0):>6} {st['n_bad']:>6}"
                      f" {st['worst_off_rel']:>13.3e} {st['meanrel']:>9.2e} {st['cos']:>11.9f}")
            else:
                st = f32_compare(read_raw(hp, dt), read_raw(lp, dt))
                rows_seen.append((name, st["meanrel"]))
                print(f"{name:<14} {dt:<5} {st['n']:>9} {100 * st['pct_diff']:>6.3f}%"
                      f" {'-':>7} {'-':>11} {'-':>6} {'-':>6} {'-':>13}"
                      f" {st['meanrel']:>9.2e} {st['cos']:>11.9f}")
        ladder = sorted([r for r in rows_seen if r[1] > 0], key=lambda r: r[1])
        if len(ladder) > 1:
            lo, hi = ladder[0], ladder[-1]
            print(f"  stage-graph locality: {len(ladder)} compared stages span meanrel"
                  f" {lo[1]:.2e} ({lo[0]}) .. {hi[1]:.2e} ({hi[0]}) = x{hi[1] / lo[1]:,.1f} top to"
                  " bottom, i.e. every compared stage is the same order of magnitude. That is the"
                  " rounding signature the plan names: a bug is ONE stage far above the stages"
                  " before and after it, while a flat ladder across 24 layers is accumulation"
                  " without a bug. A4's injected bugs sit 10^5-10^6 above their own legal floor,"
                  " for comparison.")
    return 0


# ------------------------------------------------------------------------- A4 ------------------
# The three injections `tools/route_c_stages.cpp --inject` produces, and the legal row each one is
# scored against at the SAME shape with the SAME operands. A separation of 1.0 would mean the
# comparator cannot tell a reassociation from a bug; the plan's bar is >=10x.
INJECTIONS = [
    ("inj_stale_o_proj", "o_proj", "stale tile: o_proj read block_out (the GDN gated norm's)"
                                  " instead of this layer's a_gated"),
    ("inj_rowoffset_q_proj", "q_proj", "row offset: q_proj read activation row r+1 for body row r"),
    ("inj_swapw_k_proj", "k_proj", "swapped weight pointer: k_proj's call got v_proj's weight"),
]
# Statistics separation is taken on. `min` across these is the verdict number, so a bug has to be
# caught by the least flattering statistic, not by the most flattering one (the vacuous-negative trap
# in AGENTS.md is exactly a bug that only the easy statistic sees).
SEP_KEYS = ("ndiff", "max_ulp_large", "meanrel", "n_off_large", "cos_deficit")
# The hand route never writes these two tiles — `allocate_scratch` hipMallocs them and only
# `mlp_gateup_gemm` fills them — so a hand-vs-library diff of them compares a written tile with
# uninitialised memory. Printed as what they are, never as a delta.
LIBRARY_ONLY_STAGES = {"mlp_gate_raw", "mlp_up_raw"}
# Buckets for the greedy-decision flip rate. A BF16 logit is quantised in ~0.004-wide steps near the
# top of a typical row, so anything under ~0.05 of top1-minus-top2 is a coin toss between two tokens
# for either route; 0.1 and above is a decision neither route could lose to rounding alone.
BUCKETS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, float("inf"))]


def _sep(legal: dict, bug: dict) -> dict:
    """Separation per statistic. A statistic whose legal value is zero but whose bug value is not is
    the strongest possible catch (the legal route never produces one and the bug always does), so it
    is recorded as `float('inf')` and prints as such rather than being dropped."""
    out = {}
    for key in SEP_KEYS:
        pick = (lambda d: 1.0 - d.get("cos", 1.0)) if key == "cos_deficit" else \
               (lambda d: d.get(key, 0))
        lg, bg = pick(legal), pick(bug)
        if lg in (0, 0.0):
            out[key] = float("inf") if bg else 0.0
        else:
            out[key] = bg / lg
    return out


def a4(floor_dir: Path, band: int | None) -> int:
    """A4: the negative control. Each injected bug vs the correct hand output at its own shape,
    against the legal hand-vs-library delta at that same shape and those same operands."""
    print("\nA4 — negative control. `legal` = hand vs library at the shape; `bug` = the injected"
          " output vs the correct hand output at the SAME shape, weight and activation. A row whose"
          " separation is ~1 would mean the statistic cannot tell a reassociation from a bug.\n")
    ok = True
    for inj, legal_name, what in INJECTIONS:
        for m in (88, 512, 1024):
            inj_h = floor_dir / f"{inj}@{m}_hand.bin"
            if not inj_h.exists():
                continue
            legal = bf16_compare(read_raw(floor_dir / f"{legal_name}@{m}_hand.bin", "BF16"),
                                 read_raw(floor_dir / f"{legal_name}@{m}_lib.bin", "BF16"))
            bug = bf16_compare(read_raw(inj_h, "BF16"),
                               read_raw(floor_dir / f"{legal_name}@{m}_hand.bin", "BF16"),
                               band=band)
            sep = _sep(legal, bug)
            numeric = [v for v in sep.values() if isinstance(v, float)]
            print(f"  {inj} M={m} — {what}")
            print(f"    legal : ndiff={legal['ndiff']} maxULPlarge={legal['max_ulp_large']}"
                  f" n_off_large={legal['n_off_large']} meanrel={legal['meanrel']:.2e}"
                  f" 1-cos={1 - legal['cos']:.2e}")
            print(f"    bug   : ndiff={bug['ndiff']} maxULPlarge={bug['max_ulp_large']}"
                  f" n_off_large={bug['n_off_large']} meanrel={bug['meanrel']:.2e}"
                  f" 1-cos={1 - bug['cos']:.2e}"
                  + (f"  >band={bug.get('n_over_band', 0)}" if band is not None else ""))
            print("    sep   : " + "  ".join(f"{k}=x{v:,.0f}" if isinstance(v, float)
                                             else f"{k}={v}" for k, v in sep.items()))
            finite = [v for v in numeric if v != float("inf")]
            worst_sep = min(finite) if finite else float("inf")
            if not numeric or worst_sep < 10.0:
                ok = False
                print(f"    VERDICT: NOT CAUGHT BY EVERY STATISTIC — worst separation"
                      f" x{worst_sep:,.1f}, below the plan's 10x. The audit gets strengthened"
                      " rather than the bug being dropped.")
            else:
                print(f"    VERDICT: caught by every statistic, worst separation"
                      f" x{worst_sep:,.1f} (best x{max(numeric):,.0f})")
    print(f"\nA4 overall: {'PASS' if ok else 'FAIL'} — the rule applied is the strict one: every"
          " injected bug must clear 10x on EVERY statistic, not on the most flattering one"
          " (plan §Bands asked for 10x; AGENTS.md's vacuous-negative trap is why the weakest"
          " statistic is the one that decides).")
    return 0 if ok else 1


# ------------------------------------------------------------------------- A2 ------------------
def a2_run(out_root: Path, fixtures: list[str], binary: Path, model: str) -> int:
    """The device half of A2: each fixture under `--gemm hand` and `--gemm library`, with
    `--prefill-logits all`, exactly as `engine/gate.py` would run it (replay past A-18's bound)."""
    for fx in fixtures:
        man = json.loads((REFS / fx / "manifest.json").read_text())
        prompt = man["prefill_ids"]
        greedy = man["greedy_continuation"]
        n_decode = len(greedy) - 1
        for route in ("hand", "library"):
            out = out_root / fx / route
            out.mkdir(parents=True, exist_ok=True)
            for stale in out.glob("logits*.bin"):
                stale.unlink()
            cmd = [str(binary), "--model", model, "--ids", ",".join(map(str, prompt)),
                   "--steps", str(n_decode), "--out", str(out),
                   "--prefill-logits", "all", "--gemm", route]
            if n_decode > FREE_RUN_BOUND:
                cmd += ["--replay", ",".join(str(t) for t in greedy[:-1])]
            print("$ " + " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  rc={proc.returncode} {proc.stderr.strip()[-400:]}")
                return 1
            print("  " + next((ln for ln in proc.stdout.splitlines()
                               if ln.startswith("RESULT")), "(no RESULT line)"))
    return 0


def a2(a2_root: Path, fixtures: list[str], band: int | None) -> int:
    """A2 per-step element diff on logits: hand vs library at every dumped row of each fixture,
    plus the aggregate over all of them (the per-fixture bucket counts are small, and the honest
    statement about flip rate lives in the aggregate)."""
    grand: dict[tuple[float, float], list[int]] = {}
    agree = total = 0
    for fixture in fixtures:
        agree_n, total_n, buckets = _a2_one(a2_root, fixture, band)
        if agree_n is None:
            continue
        agree += agree_n
        total += total_n
        for key, (rows_n, flips) in buckets.items():
            acc = grand.setdefault(key, [0, 0])
            acc[0] += rows_n
            acc[1] += flips
    if total:
        print(f"\nA2 aggregate over {len(fixtures)} fixtures: {agree}/{total} dumped logits rows"
              " agree on the greedy argmax."
              + (f" {total - agree} flips = {100 * (total - agree) / total:.2f}%."
                 if total != agree else " — no flips at all."))
        for key in sorted(grand):
            rows_n, flips = grand[key]
            hi = 999.0 if key[1] == float("inf") else key[1]
            print(f"  room [{key[0]:.2f},{hi:.2f}): {rows_n:5d} rows, {flips:3d} flipped"
                  f" = {100 * flips / rows_n if rows_n else 0.0:5.1f}%")
    return 0


Buckets = dict[tuple[float, float], tuple[int, int]]


def _a2_one(a2_root: Path, fixture: str, band: int | None) -> tuple[int | None, int, Buckets]:
    """One fixture's A2 table. Returns (rows that agree, rows dumped, per-bucket (rows, flips))."""
    man = json.loads((REFS / fixture / "manifest.json").read_text())
    prompt_len = len(man["prefill_ids"])
    hand, lib = a2_root / fixture / "hand", a2_root / fixture / "library"
    idx = sorted({int(m.group(1)) for p in list(hand.glob("logits*.bin"))
                  for m in [re.match(r"logits(\d+)\.bin", p.name)] if m})
    print(f"\nA2 — {fixture}: {len(idx)} dumped rows, prompt_len={prompt_len}"
          f" (rows 0..{prompt_len - 1} are prefill rows, {prompt_len}.. are generation steps)."
          f" band = {band} ULP (max A1 max-ULP at the fixture's body width).\n")
    print(f"{'idx':>5} {'kind':<6} {'n':>8} {'%diffF32':>9} {'%diffBF16':>10} {'maxULP':>7}"
          f" {'maxULPlarge':>11} {'>band':>6} {'n_bad':>6} {'value@max':>11} {'rel@max':>9}"
          f" {'1-cosF32':>9} {'argmax=':>7} {'gap_hand':>9} {'gap_lib':>9}")
    worst, decisions = None, []
    for i in idx:
        hp, lp = hand / f"logits{i}.bin", lib / f"logits{i}.bin"
        if not (hp.exists() and lp.exists()):
            continue
        hv = read_raw(hp, "F32")
        lv = read_raw(lp, "F32")
        st = logit_compare(hv, lv, band=band)
        dec = logit_row_decision(lv, hv)
        decisions.append((i, dec))
        kind = "prefill" if i < prompt_len else "decode"
        print(f"{i:>5} {kind:<6} {st['n']:>8} {100 * st['pct_diff_f32']:>8.3f}%"
              f" {100 * st['pct_diff']:>9.3f}% {st['max_ulp']:>7} {st['max_ulp_large']:>11}"
              f" {st.get('n_over_band', 0):>6} {st['n_bad']:>6} {st['ref_at_max']:>11.5g}"
              f" {st['rel_at_max']:>9.2e} {1 - st['cos_f32']:>9.2e}"
              f" {'yes' if dec['argmax_match'] else 'NO':>7}"
              f" {dec['gap_ref']:>9.3f} {dec['gap_got']:>9.3f}")
        if worst is None or st["max_ulp_large"] > worst[1]["max_ulp_large"]:
            worst = (i, st)
    if worst:
        i, st = worst
        print(f"  worst row {i}: {st['max_ulp_large']} ULP on a band-excluded element, raw max"
              f" {st['max_ulp']} ULP at flat index {st['argmax']} (value {st['ref_at_max']:.6g} at"
              f" rel {st['rel_at_max']:.2e} of the row max); elements beyond 1 ULP that are NOT"
              f" near-cancellation: {st['n_off_large']}; 1-cos = {1 - st['cos_f32']:.3e}")
    if decisions:
        dis = [(i, d) for i, d in decisions if not d["argmax_match"]]
        print(f"  greedy decision: {len(decisions) - len(dis)}/{len(decisions)} rows agree on"
              " argmax. Flip rate bucketed by how much room the decision had — `room` is the"
              " SMALLER of the two sides' top1-minus-top2 gap, so a row in the first bucket was a"
              " coin toss for one of the two routes and cannot be blamed on the other:")
        buckets: Buckets = {}
        for lo, hi in BUCKETS:
            inb = [d for _, d in decisions
                   if lo <= min(d["gap_ref"], d["gap_got"]) < hi]
            flips = sum(1 for d in inb if not d["argmax_match"])
            buckets[(lo, hi)] = (len(inb), flips)
            print(f"    room [{lo:>5.2f},{hi if hi != float('inf') else 999:>6.2f}):"
                  f" {len(inb):>4} rows, {flips:>3} flipped"
                  f" = {100 * flips / len(inb) if inb else 0.0:5.1f}%")
        for i, d in dis:
            print(f"    row {i}: hand argmax {d['argmax_ref']} (gap {d['gap_ref']:.3f})"
                  f" vs library argmax {d['argmax_got']} (gap {d['gap_got']:.3f})")
        return len(decisions) - len(dis), len(decisions), buckets
    return None, 0, {}


def prompt_ids(tokens: int) -> str:
    """The audit's prompt: real token ids from the shipped fixtures' own rendered prompts, in
    fixture order, truncated to `tokens`. Real text through the real chat template, so the activation
    statistics are the ones the engine produces — not a synthetic fill, which is what the plan's
    'real activations' rule is against. The engine caps a body at `kMaxWaveRows` (512), so a 1024-
    token prompt is two bodies and both are captured to give 1024 real activation rows."""
    ids: list[int] = []
    for fx in FIXTURES:
        man = json.loads((REFS / fx / "manifest.json").read_text())
        ids.extend(man["prefill_ids"])
        if len(ids) >= tokens:
            break
    return ",".join(str(t) for t in ids[:tokens])


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompt")
    p.add_argument("--tokens", type=int, default=1024)
    p = sub.add_parser("a1")
    p.add_argument("floor_dir", type=Path)
    p.add_argument("--json", type=Path, default=None)
    p = sub.add_parser("a3")
    p.add_argument("dump_dir", type=Path)
    p.add_argument("--a1json", type=Path, default=None)
    p = sub.add_parser("a4")
    p.add_argument("floor_dir", type=Path)
    p.add_argument("--band", type=int, default=None)
    p = sub.add_parser("a2-run")
    p.add_argument("out_root", type=Path)
    p.add_argument("--fixtures", default=",".join(FIXTURES))
    p.add_argument("--binary", type=Path, default=ROOT / "engine" / "build" / "decode_run")
    p.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    p = sub.add_parser("a2")
    p.add_argument("a2_root", type=Path)
    p.add_argument("fixtures", nargs="*", default=list(FIXTURES))
    p.add_argument("--band", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "prompt":
        print(prompt_ids(a.tokens))
        return 0
    if a.cmd == "a1":
        return a1(a.floor_dir, a.json)
    if a.cmd == "a3":
        return a3(a.dump_dir, a.a1json)
    if a.cmd == "a4":
        return a4(a.floor_dir, a.band)
    if a.cmd == "a2-run":
        return a2_run(a.out_root, a.fixtures.split(","), a.binary, a.model)
    return a2(a.a2_root, a.fixtures, a.band)


if __name__ == "__main__":
    raise SystemExit(main())
