#!/usr/bin/env python3
"""P9 Phase 0 — price the chunked GDN delta-rule scan against the pinned dumps. CPU only.

Four subcommands, one deliverable each, from `progress/2026-10-14/plan-p9-chunked-scan-2026-10-14.md`:

    .venv/bin/python prototype/gdn_chunk/spike.py stage    # per-stage error vs data/refs
    .venv/bin/python prototype/gdn_chunk/spike.py carry    # initial-state / end-condition vs pins
    .venv/bin/python prototype/gdn_chunk/spike.py tokens   # decision-margin-bucketed agreement
    .venv/bin/python prototype/gdn_chunk/spike.py cost     # cost model at the measured rates

`data/refs/` is the reference. The chunked form is never judged against the sequential form
alone — that proves self-consistency and nothing else (§04.7, G3's weak half, and the reason
`bench/gate-audit.md` flags the existing `run_chunkcheck.py` as a cos-only check). Every table
here therefore prints three sides: chunk-vs-pin, sequential-vs-pin (the floor the engine already
pays) and chunk-vs-sequential (the delta a route change would introduce).

Error is reported the way §10.1 (F1–F4) demands: per-stage `meanrel`/`cos`, a BF16 ULP histogram
WITH the cancellation band and how much of the tensor that band excuses, and decision-margin
bucketed token agreement. No single max-ULP number is used as a verdict anywhere: F1–F4 measure
that instrument to be meaningless for a long dot.

Pin dtypes matter and are measured, not assumed: `gdn.o_pre_norm` comes back BF16-valued (HF's
rule casts its output back to the activation dtype), so an F32 candidate compared to it in F32
scores meanrel ~1e-3 of *pin quantization*, not of kernel error. `gdn.state_new` is genuine F32.
The `stage` table marks each stage with the width its pin actually carries and scores the BF16-
valued ones in BF16 ULPs, which is the only width at which those comparisons mean anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "gdn_chunk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bf16_ulp                                                    # noqa: E402
import chunk_scan as cs                                            # noqa: E402
from speccheck import specimpl                                     # noqa: E402

GDN_LAYERS = specimpl.GDN_LAYERS
BAND = 3.0e-5          # §10.1's cancellation band, sized by prototype/mlp_gemv/floor_probe.py
ULP_MAX = 1            # §10.1's per-element ULP bound above that band
CS_LIST = (16, 32, 64)
D11 = dict(meanrel=3.0e-2, row_1cos=2.0e-3, stage_cos=0.9998, room=0.10)   # admitted cascade band
# Forms under test. `mma` rounds every matmul operand to BF16 and keeps the F32 accumulator,
# which is what this box's matrix units do (devices/strix-halo-compute.md); it is a model of a
# kernel, not a kernel, and it is how Phase 0 prices the band a real device form would need.
FORMS = [("seq", "sequential S04.4 (today's engine)"),
         ("chunk16", "chunked C=16, F32 operands"),
         ("chunk32", "chunked C=32, F32 operands"),
         ("chunk64", "chunked C=64, F32 operands (= the reference's own prefill form)"),
         ("chunk16mma", "chunked C=16, BF16 matmul operands (device shape)"),
         ("chunk32mma", "chunked C=32, BF16 matmul operands (device shape)"),
         ("chunk64mma", "chunked C=64, BF16 matmul operands (device shape)")]
FORM_TAGS = ",".join(t for t, _ in FORMS)
STAGES = ("o_pre_norm", "state_new", "gated_out", "out_proj_out")
# Measured widths of the pins, not the dtype of the .npy header: all four are stored float32, three
# of them are BF16-valued, so an F32 candidate scored against those three in F32 measures the pin.
PIN_WIDTH = dict(o_pre_norm="BF16-valued", state_new="true F32", gated_out="BF16-valued",
                 out_proj_out="BF16-valued")


def parse_form(tag: str) -> dict:
    if tag == "seq":
        return dict(form="seq")
    if tag.startswith("chunk"):
        body = tag[5:]
        mma = body.endswith("mma")
        return dict(form="chunk", C=int(body[:-3] if mma else body), mma=mma)
    raise ValueError(tag)


# --------------------------------- pins, candidates, metrics -------------------
def load_gdn_pins(fixture: str, layer: int, seg: slice | None = None) -> dict:
    """One GDN layer's pinned operands and outputs, squeezed to [T, ...]."""
    d = ROOT / "data" / "refs" / fixture
    cut = (slice(0, 1 << 30) if seg is None else seg)
    out = {}
    for key in ("q_post_conv", "k_post_conv", "v_post_conv", "beta", "g_decay_log",
                "o_pre_norm", "state_new"):
        t = torch.from_numpy(np.load(d / f"L{layer:02d}.gdn.{key}.npy")).float().squeeze(0)
        out[key] = t if key == "state_new" else t[cut]
    out["z"] = torch.from_numpy(np.load(d / f"L{layer:02d}.in_proj_z.out.npy")).float()[cut]
    out["gated_out"] = torch.from_numpy(np.load(d / f"L{layer:02d}.gdn.gated_out.npy")).float()
    out["out_proj_out"] = torch.from_numpy(np.load(d / f"L{layer:02d}.out_proj.out.npy")).float()
    return out


def scan(pins: dict, form: str, C: int = 64, mma: bool = False, bug: str = "",
         S0: torch.Tensor | None = None):
    """One arithmetic form of §04.4 fed the pinned post-conv operands → (o, S)."""
    qn, kn = cs.qk_normalized(pins["q_post_conv"], pins["k_post_conv"])
    if form == "seq":
        return cs.recurrent_scan(qn, kn, pins["v_post_conv"], pins["g_decay_log"],
                                 pins["beta"], S=S0, bug=bug)
    return cs.chunked_scan(qn, kn, pins["v_post_conv"], pins["g_decay_log"], pins["beta"],
                           S=S0, C=C, mma=mma, bug=bug)


def downstream(w, layer: int, o: torch.Tensor, z: torch.Tensor):
    """§04.5 + out_proj on a candidate `o`, the same code for every side so a difference
    stays attributable to the scan. Returns (gated_out, out_proj_out) in the widths the
    residual stream actually carries: gated BF16, out F32-of-BF16."""
    p = f"model.language_model.layers.{layer}.linear_attn."
    T = o.shape[0]
    gated = specimpl.rmsnorm_gated(o.reshape(T * cs.H_V, cs.D_V).to(torch.bfloat16),
                                   z.reshape(T * cs.H_V, cs.D_V).to(torch.bfloat16),
                                   w.p[p + "norm.weight"].to(torch.bfloat16)).float()
    return gated, (gated.to(torch.bfloat16).reshape(T, specimpl.VAL_DIM)
                   @ w.p[p + "out_proj.weight"].t()).float()


def stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """§10.2's aggregates: cos over the flattened tensor, mean|Δ|/mean|ref|, max|Δ|/max|ref|."""
    g, r = got.double().reshape(-1), ref.double().reshape(-1)
    denom = float(g.norm() * r.norm())
    return dict(cos=float(g @ r / denom) if denom else 1.0,
                meanrel=float((g - r).abs().mean() / r.abs().mean()),
                maxrel=float((g - r).abs().max() / r.abs().max()))


def ulp_arrays(got: torch.Tensor, ref: torch.Tensor, cancel_rel: float = BAND):
    """(BF16 ULP distance, in-cancellation-band) for every element, at the width `ref` carries.

    Split out of `ulp_line` so a pool over 18 layers is a real pool (concatenate the arrays,
    histogram once) instead of an addition of per-layer bucket counts, which double-counted
    the moment the first layer's bucket list was aliased into the accumulator.
    """
    r = ref.double().reshape(-1)
    scale = float(r.abs().max())
    rel = (r.abs() / scale).numpy() if scale else np.zeros(r.numel(), dtype=np.float64)
    d = bf16_ulp.ulp_dist(got.to(torch.bfloat16), ref.to(torch.bfloat16)).reshape(-1)
    return d, np.asarray(bf16_ulp.band_usage(rel, d, ULP_MAX, cancel_rel)["in_band"])


ULP_EDGES = ((0, 0), (1, 1), (2, 4), (5, 16), (17, 128), (129, 4096), (4097, None))


def ulp_line(got: torch.Tensor, ref: torch.Tensor, cancel_rel: float = BAND) -> dict:
    """§10.1's report on one tensor pair: how much moved, how much of it the cancellation band
    excuses, and the worst offender that the band does NOT excuse (`worst_off_rel` is the
    smallest band under which nothing offends, so the band stays falsifiable)."""
    d, in_band = ulp_arrays(got, ref, cancel_rel)
    off = d > ULP_MAX
    worst_off_rel = 0.0
    if off.any():
        rel = (ref.double().abs() / float(ref.double().abs().max())).reshape(-1)[off]
        worst_off_rel = float(rel.max())
    hist = []
    for lo, hi in ULP_EDGES:
        m = (d >= lo) if hi is None else ((d >= lo) & (d <= hi))
        hist.append((lo, hi, int(m.sum()), int((m & ~in_band).sum())))
    return dict(n=int(d.size), ndiff=int((d > 0).sum()), n_band=int(in_band.sum()),
                n_off=int(off.sum()), n_off_band=int((off & ~in_band).sum()),
                worst_off_ulp=int(d[off].max()) if off.any() else 0,
                worst_off_band_ulp=int(d[off & ~in_band].max())
                if (off & ~in_band).any() else 0,
                worst_off_rel=worst_off_rel, hist=hist)


def fmt_ulp(u: dict) -> str:
    return ("%5.1f%% differ | band %5.1f%% | >%dULP %6d (off-band %5d) | worst_off_rel "
            "%.1e | worst %d ULP, off-band %d"
            % (100.0 * u["ndiff"] / u["n"], 100.0 * u["n_band"] / u["n"], ULP_MAX,
               u["n_off"], u["n_off_band"], u["worst_off_rel"], u["worst_off_ulp"],
               u["worst_off_band_ulp"]))


def worst_layer(per_layer: list[dict], key: str, worst) -> tuple[float, int]:
    best = worst(range(len(per_layer)), key=lambda i: per_layer[i][key])
    return per_layer[best][key], per_layer[best]["layer"]


# --------------------------------- stage ---------------------------------------
def cmd_stage(args) -> int:
    torch.set_grad_enabled(False)
    w = specimpl.Weights(Path(args.model))
    forms = [(t, dict(FORMS)[t]) for t in args.forms.split(",")]
    for fixture in args.fixtures.split(","):
        T = int(np.load(ROOT / f"data/refs/{fixture}/L00.gdn.o_pre_norm.npy",
                       mmap_mode="r").shape[1])
        print(f"\n=== stage: {fixture} — T={T} tokens, {len(GDN_LAYERS)} GDN layers, "
              f"pin widths {PIN_WIDTH}")
        print("    (the reference's own prefill is chunked at C=64: "
              "`torch_chunk_gated_delta_rule`, so chunk-vs-pin is not a form-vs-form proxy)")
        sides: dict[str, dict[int, dict]] = {}
        refs: dict[int, dict] = {}
        for l in GDN_LAYERS:
            pins = load_gdn_pins(fixture, l)
            refs[l] = pins
            for tag, _lbl in forms:
                o, S = scan(pins, **parse_form(tag))
                gated, outp = downstream(w, l, o, pins["z"])
                # a fresh dict per form: one shared dict would make every side point at the
                # last form computed and every row of the table identical.
                sides.setdefault(tag, {})[l] = dict(o_pre_norm=o, state_new=S,
                                                   gated_out=gated, out_proj_out=outp)
            sides.setdefault("ref", {})[l] = dict(
                o_pre_norm=pins["o_pre_norm"], state_new=pins["state_new"],
                gated_out=pins["gated_out"], out_proj_out=pins["out_proj_out"])
            # ctrl = the PIN's own o pushed through this file's §04.5 code. If the reference's
            # downstream rounding is reproduced exactly, ctrl matches the pinned gated/out bit
            # for bit, and every gated/out number below belongs to the scan and nowhere else.
            ctrl_g, ctrl_o = downstream(w, l, pins["o_pre_norm"], pins["z"])
            sides.setdefault("ctrl", {})[l] = dict(o_pre_norm=pins["o_pre_norm"],
                                                  state_new=pins["state_new"],
                                                  gated_out=ctrl_g, out_proj_out=ctrl_o)
        for dst, dst_lbl in (("ref", "vs PINS (data/refs)"),
                             ("seq", "vs SEQUENTIAL (the route delta)"),
                             ("ctrl", "vs PIN-o-THROUGH-THIS-FILE'S S04.5 (downstream control)")):
            print(f"\n  --- {dst_lbl}; worst over the 18 layers, band {BAND:g} of tensor max")
            print("    %-26s %-13s %10s %10s %10s  %s"
                  % ("form", "stage", "cos", "meanrel", "maxrel", "BF16 ULP"))
            for tag, lbl in forms:
                if tag == dst:
                    continue
                for st in STAGES:
                    per = [dict(layer=l, **stats(sides[tag][l][st], sides[dst][l][st]),
                               **ulp_line(sides[tag][l][st], sides[dst][l][st]))
                           for l in GDN_LAYERS]
                    cm, cl = worst_layer(per, "cos", min)
                    mm, ml = worst_layer(per, "meanrel", max)
                    xm, xl = worst_layer(per, "maxrel", max)
                    wm = max(per, key=lambda r: r["worst_off_band_ulp"])
                    print("    %-26s %-13s %10.8f %10.2e %10.2e  %s"
                          % (f"{lbl} (worst L{cl:02d}/L{ml:02d}/L{xl:02d})", st, cm, mm, xm,
                             fmt_ulp(wm)))
        if args.hist:
            print("\n  --- BF16 ULP histogram, all 18 layers pooled into ONE array per stage "
                  "(count total / of those off-band); band is 3e-05 of the tensor max")
            for tag in ("seq", "chunk64", "chunk64mma"):
                for st in ("o_pre_norm", "state_new"):
                    ds, bs = [], []
                    for l in GDN_LAYERS:
                        d, b = ulp_arrays(sides[tag][l][st], refs[l][st])
                        ds.append(d)
                        bs.append(b)
                    d, in_band = np.concatenate(ds), np.concatenate(bs)
                    off = d > ULP_MAX
                    cells = " ".join(
                        f"[{lo}-{hi if hi is not None else 'max'}] "
                        # The band column is only meaningful for elements that actually offend
                        # (d > ULP_MAX): an element that did not move has nothing to excuse, and
                        # printing a band share for the [0-0] bucket reads as if 92 % of the
                        # tensor needed excusing.
                        f"{int(((d >= lo) & ((d <= hi) if hi is not None else True)).sum())}"
                        + ("" if lo <= ULP_MAX else
                           "/" + str(int(((d >= lo) & ((d <= hi) if hi is not None else True)
                                          & off & ~in_band).sum())))
                        for lo, hi in ULP_EDGES)
                    print(f"    {tag:10s} {st:12s} n={d.size:9d} band {100.0*in_band.sum()/d.size:5.1f}% "
                          f">{ULP_MAX}ULP {int(off.sum()):7d} off-band {int((off & ~in_band).sum()):7d} "
                          f"worst off-band {int(d[off & ~in_band].max()) if (off & ~in_band).any() else 0} ULP")
                    print(f"               cells total[/off-band, offending buckets only]: {cells}")
        if args.negatives:
            print("\n  --- injected bugs, and where each one is visible (locality: a negative "
                  "that catches everything proves nothing)")
            pins = load_gdn_pins(fixture, GDN_LAYERS[0])
            C = 64
            good_o, good_S = scan(pins, "chunk64", C=C)
            NC = -(-T // C)
            for bug in ("no_inter_chunk_read", "causality_upper", "no_pairwise_decay",
                        "no_chunk_decay", "drop_last_chunk", "read_before_decay",
                        "read_after_write"):
                fn = "seq" if bug.startswith("read_") else "chunk64"
                o, S = scan(pins, fn, bug=bug)
                fin = bool(torch.isfinite(o).all() and torch.isfinite(S).all())
                per = ([float((o[i * C:(i + 1) * C] - good_o[i * C:(i + 1) * C]).abs().max())
                        for i in range(NC)] if fn == "chunk64" else [0.0])
                st = stats(S, good_S)
                print("    %-22s per-chunk max|Δo| %s | state meanrel %.2e %s"
                      % (bug, " ".join(f"{x:.1e}" for x in per), st["meanrel"],
                         "" if fin else "(non-finite: caught outright)"))
    return 0


# ---------------------------------- carry --------------------------------------
def cmd_carry(args) -> int:
    """§04.7's initial-state and end-condition clauses against the pins.

    The engine prefills in `wave_rows` bodies and resumes requests, so a chunked kernel
    must (a) take a carried-in state and (b) hand out the state the reference would have
    at the same cut. Checked by cutting each fixture's pinned operands at a token count
    that is deliberately NOT a multiple of any chunk size in use, feeding segment 2 the
    chunked end-state of segment 1, and comparing both the concatenated `o` and the final
    `S` against the pins."""
    torch.set_grad_enabled(False)
    rows = []
    for fixture in args.fixtures.split(","):
        T = int(np.load(ROOT / f"data/refs/{fixture}/L00.gdn.o_pre_norm.npy",
                        mmap_mode="r").shape[1])
        for l in (GDN_LAYERS if args.per_layer else GDN_LAYERS[:3]):
            pins = load_gdn_pins(fixture, l)
            for cut in {c for c in (args.cut, T - args.cut2, T // 3) if 0 < c < T}:
                for C in CS_LIST:
                    s1 = load_gdn_pins(fixture, l, slice(0, cut))
                    s2 = load_gdn_pins(fixture, l, slice(cut, T))
                    _, S1 = scan(s1, "chunk", C=C)
                    o2, S2 = scan(s2, "chunk", C=C, S0=S1)
                    o1 = scan(s1, "chunk", C=C)[0]
                    o = stats(torch.cat([o1, o2], 0), pins["o_pre_norm"])
                    s = stats(S2, pins["state_new"])
                    zero = stats(scan(s2, "chunk", C=C)[1], pins["state_new"])
                    rows.append(dict(fixture=fixture, layer=l, C=C, cut=cut,
                                     o_cos=o["cos"], o_meanrel=o["meanrel"],
                                     s_cos=s["cos"], s_meanrel=s["meanrel"],
                                     nozr_zero=s["meanrel"], nozr=zero["meanrel"]))
    bad = [r for r in rows if r["s_cos"] < 0.99999 or r["o_cos"] < 0.99999]
    print(f"carry: {len(rows) - len(bad)}/{len(rows)} splits PASS")
    for key, worst in (("o_cos", min), ("s_cos", min), ("o_meanrel", max), ("s_meanrel", max)):
        print(f"  {key:11s} worst {worst(r[key] for r in rows):.3e}")
    print(f"  a zero-initialised second segment would score state meanrel "
          f"{max(r['nozr'] for r in rows):.3e} — the carry is doing real work, "
          f"not being excused")
    for r in bad[:6]:
        print("   FAIL", r)
    return 1 if bad else 0


# ---------------------------------- tokens -------------------------------------
def run_side(w, ids, mode, steps: int, replay: list[int] | None = None):
    """One side of the decision comparison: greedy from `ids`, recording the logits row that
    decided each token. Row 0 is the LAST PREFILL ROW and decides token 0; row i (i≥1) is the
    logits after feeding token i−1 (or `replay[i-1]`) and decides token i. `gen[i]` pairs with
    `greedy_continuation[i]` in the manifest — the same alignment `engine/chat.py` documents.
    """
    if mode == "seq":
        specimpl.gdn_chunk_scan = _ORIG_CHUNK
        f = specimpl.Forward(w, gdn_form="recurrent")
    else:
        C, mma = mode
        specimpl.gdn_chunk_scan = (lambda qn, kn, v, g, beta, S=None, _C=C:
                                   cs.chunked_scan(qn, kn, v, g, beta, S, C=_C, mma=mma))
        f = specimpl.Forward(w, gdn_form="chunk")
    rows, toks = [], []
    last = f.logits(f.prefill(torch.tensor(ids)))[-1]
    rows.append(last.clone())
    toks.append(int(last.argmax()))
    for i in range(steps):
        tok = replay[i] if replay is not None else toks[-1]
        if replay is None and tok in specimpl.EOS_IDS:
            break
        last = f.logits(f.decode(tok))[-1]
        rows.append(last.clone())
        toks.append(int(last.argmax()))
    return rows, toks


def gap(logits: torch.Tensor) -> float:
    """top-1 minus top-2. A2's `room` is the SMALLER of the two sides' gaps."""
    top = torch.topk(logits.float(), 2)
    return float(top.values[0] - top.values[1])


BUCKETS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.50), (0.50, float("inf")))


def new_tally() -> dict:
    return dict(rows=0, flips=0, buck={b: [0, 0] for b in BUCKETS},
                cos=[float("inf"), 0.0, 0.0], pin_ok=0, pin_all=0, detail=[],
                stage_worst=0.0, pin_side=[0, 0, 0])


def cmd_tokens(args) -> int:
    torch.set_grad_enabled(False)
    w = specimpl.Weights(Path(args.model))
    modes = [(C, mma) for C in CS_LIST for mma in (False, True)] if args.mma else \
        [(C, False) for C in CS_LIST] + [(64, True)]
    tally = {m: new_tally() for m in modes}
    base_pin = [0, 0]
    for src in args.fixtures.split(","):
        items = []
        if src.startswith("corpus:"):
            prompts = json.loads((ROOT / "data/refs/corpus48/manifest.json").read_text())["prompts"]
            for pr in prompts[:args.corpus_n]:
                items.append((pr["ids"], pr["continuation"][:args.steps], f"corpus48[{pr['ids'][:2]}]"))
        else:
            man = json.loads((ROOT / f"data/refs/{src}/manifest.json").read_text())
            pin = man.get("greedy_continuation", [])
            items.append((man["prefill_ids"], pin[:max(args.steps, 1)], src))
        for ids, pin, label in items:
            steps = len(pin) if pin else args.steps
            a_rows, a_toks = run_side(w, ids, "seq", steps)
            for j, want in enumerate(a_toks[:len(pin)]):
                base_pin[0] += int(want == pin[j])
                base_pin[1] += 1
            for mode in modes:
                t = tally[mode]
                b_rows, b_toks = run_side(w, ids, mode, steps, replay=a_toks[:steps])
                for j, (a, b) in enumerate(zip(a_rows, b_rows)):
                    room = min(gap(a), gap(b))
                    cos = float((a.double() @ b.double()
                                 / (a.double().norm() * b.double().norm())))
                    one = 1.0 - cos
                    a_tok, b_tok = int(a.argmax()), int(b.argmax())
                    flip = a_tok != b_tok
                    t["rows"] += 1
                    t["flips"] += flip
                    for lo, hi in BUCKETS:
                        if lo <= room < hi:
                            t["buck"][(lo, hi)][0] += 1
                            t["buck"][(lo, hi)][1] += flip
                    t["cos"] = [min(t["cos"][0], one),
                                (t["cos"][1] * (t["rows"] - 1) + one) / t["rows"],
                                max(t["cos"][2], one)]
                    # Adjudicate a flip against the pinned HF continuation: "a decision changed"
                    # and "a decision changed towards the reference" are different findings, and
                    # D-11's flip rule cannot tell them apart.
                    want = pin[j] if j < len(pin) else None
                    if flip:
                        t["detail"].append((label, j, room, a_tok, b_tok, want,
                                            a_tok == want, b_tok == want))
                for j, want in enumerate(b_toks[:len(pin)]):
                    t["pin_ok"] += int(want == pin[j])
                    t["pin_all"] += 1
    print("\n=== tokens: today's sequential prefill vs the chunked form, same fed tokens "
          "(replay), so every row is comparable")
    print("    sides: row 0 = last prefill row (decides the first generated token), row i = "
          "logits after feeding token i-1")
    print("    D-11 admitted band for comparison: row 1-cos <= %.0e, flip room < %.2f"
          % (D11["row_1cos"], D11["room"]))
    print(f"    the sequential side itself reproduces {base_pin[0]}/{base_pin[1]} pinned "
          f"continuation tokens (today's form, same anchor)")
    for mode, t in tally.items():
        C, mma = mode
        print(f"\n  chunked C={C}{' BF16-operand matmuls' if mma else ''}: {t['rows']} rows, "
              f"{t['flips']} flipped ({100.0 * t['flips'] / max(t['rows'], 1):.2f} %)")
        print("    logits-row 1-cos: min %.2e mean %.2e max %.2e" % tuple(t["cos"]))
        print("    pinned continuation reproduced: %d/%d" % (t["pin_ok"], t["pin_all"]))
        print("    room bucket        rows  flipped    rate")
        for b in BUCKETS:
            n, fl = t["buck"][b]
            print("    %5.2f - %-6s %6d %8d %7.1f %%   %s"
                  % (b[0], ("%.2f" % b[1]) if b[1] < 1 else "inf", n, fl,
                     100.0 * fl / n if n else 0.0,
                     "D-11 admits flips here" if b[1] <= D11["room"] else "D-11 admits NO flip here"))
        for lbl, j, room, a_tok, b_tok, want, a_ok, b_ok in t["detail"]:
            print("    FLIP %s row %d room %.3f: sequential %d -> chunked %d; pinned token %s "
                  "(%s)" % (lbl, j, room, a_tok, b_tok,
                            want if want is not None else "not pinned",
                            "chunked side matches the pin" if b_ok else
                            ("sequential side matches the pin" if a_ok else
                             "neither side matches the pin")))
    return 0


# ----------------------------------- cost --------------------------------------
RATES = [(0.271, "gdn_rec today, effective (125.4 ms over 33.94 GFLOP, measured wave)"),
         (0.85, "the engine's own hand GEMMs, measured 0.8-0.9 TFLOP/s"),
         (4.80, "hipBLAS 24-layer wave mix at M=64, measured, 16-bit C"),
         (18.84, "hipBLAS 24-layer wave mix at M=512, measured, 16-bit C"),
         (33.0, "hipBLAS at gate_proj's own shape, measured best")]
MEASURED_GDN_MS = 125.4


def cmd_cost(args) -> int:
    """Arithmetic cost model at the engine's real shapes, priced only at measured rates."""
    D, H, L = 128, 16, 18
    BODY, WAVE = args.rows, args.tokens
    BODIES = WAVE // BODY
    seq = D * D + 2 * D * D + 3 * D + 2 * D * D + 2 * D * D      # decay, read, delta, write, read
    seq_wave = seq * H * L * WAVE
    traffic_seq = 2 * D * D * 4 * H * L * WAVE
    print(f"shapes: {BODY}-row bodies x {BODIES} = {WAVE} tokens, H={H} v-heads, "
          f"d_k=d_v={D}, {L} GDN layers. Measured baseline: {MEASURED_GDN_MS} ms, 18,432 launches, "
          f"6.8 us each.")
    print(f"\nsequential S04.4: {seq} flops per token-head-layer -> {seq_wave/1e9:.2f} GFLOP/wave"
          f" = {seq_wave/MEASURED_GDN_MS*1e-9:.3f} TFLOP/s effective")
    print(f"  state traffic {traffic_seq/1e9:.2f} GB/wave (S read+write per token-head), "
          f"which at the 240 GB/s DRAM ceiling is {traffic_seq/240e9*1e3:.0f} ms — more than the "
          f"measured {MEASURED_GDN_MS} ms, so the state is being served above DRAM rate (it is "
          f"1 MiB per layer, `hipGetDeviceProperties` says 2 MiB L2) and the op is a "
          f"per-token-latency shape, not a flops shape.")
    print(f"  pure dispatch: 18,432 x 1.8 us (measured eager-launch floor) = "
          f"{18432*1.8e-3:.0f} ms of the {MEASURED_GDN_MS} ms")
    for C in CS_LIST:
        intra = (2 * C * C * D            # UT system (k.beta) k^T, decayed
                 + 2 * C * C * D          # intra attention q k^T, decayed
                 + 4 * C * C * D          # the UT inverse applied to two [C,128] right sides
                 + 2 * C * C * D          # intra @ v_new
                 + 2 * C ** 3 // 3)       # UT inverse built, C^3/3 MACs
        inter = (6 * C * D * D            # kcd@S, q~@S, k^T@v_new — the serial carry
                 + D * D)                 # chunk decay of the state
        per = (intra + inter) // C
        wave = per * H * L * WAVE
        steps = -(-BODY // C)
        print(f"\nchunked C={C}: intra {intra} + inter {inter} per chunk-head = {per} "
              f"flops/token-head-layer ({100.0 * (per / seq - 1):+.1f} % vs sequential)")
        print(f"  {wave/1e9:.2f} GFLOP/wave; serial depth {steps} chunk-steps per body instead of "
              f"{BODY} token-steps; state traffic {traffic_seq/C/1e6:.0f} MB/wave")
        print(f"  launches: a FLA-shaped kernel is ~4 sites per (layer, body) = "
              f"{4*L*BODIES} ({4*L*BODIES*1.8e-3:.1f} ms dispatch); one launch per chunk-step "
              f"would be {L*BODIES*steps} ({L*BODIES*steps*1.8e-3:.1f} ms)")
        print("    %-16s %-22s %-9s %s" % ("rate TF/s", "ms/wave", "speedup", "verdict vs 1.5x"))
        for rate, note in RATES:
            ms = wave / rate / 1e12 * 1e3 + 4 * L * BODIES * 1.8e-3
            print("    %-16s %-22.1f %-9.2f %s" % (rate, ms, MEASURED_GDN_MS / ms,
                                                   ("PASS" if MEASURED_GDN_MS / ms >= 1.5
                                                    else "kill") + "   " + note))
        for want in (1.0, 1.5, 2.0):
            need = wave / ((MEASURED_GDN_MS / want) * 1e-3) / 1e12
            print("    needs %.3f TF/s for %.1fx = %.0f %% of the hand-GEMM rate, %.0f %% of the "
                  "measured M=64 mix rate" % (need, want, 100.0 * need / 0.85, 100.0 * need / 4.80))
    nq, dh = 8, 256
    per_layer = 0
    for b in range(BODIES):
        per_layer += 4 * nq * dh * sum(b * BODY + t + 1 for t in range(BODY))
    attn = per_layer * 6
    print(f"\nattn (deliverable 4): 12 sites (6 attn layers x {BODIES} bodies), 109.2 ms measured, "
          f"causal-triangle geometry {nq} heads x {dh} dims = {attn/1e9:.2f} GFLOP/wave "
          f"= {attn/109.2e-3/1e12:.3f} TFLOP/s effective")
    for rate, note in RATES[1:]:
        print("   at %-6s TF/s that arithmetic is %6.1f ms   (%s)"
              % (rate, attn / rate / 1e12 * 1e3, note))
    print(f"  same question at 4096-token context: the O(T^2) term scales as T, so the same "
          f"sites would cost {(4*nq*dh*sum(4096+t for t in range(4096))*6)/attn:.1f}x the flops")
    return 0


_ORIG_CHUNK = specimpl.gdn_chunk_scan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stage", help="per-stage error vs the pinned dumps")
    p.add_argument("--fixtures", default="p8,p64,p300")
    p.add_argument("--forms", default=FORM_TAGS)
    p.add_argument("--hist", action="store_true")
    p.add_argument("--negatives", action="store_true")
    p.set_defaults(fn=cmd_stage)
    p = sub.add_parser("carry", help="initial-state / end-condition clauses vs the pins")
    p.add_argument("--fixtures", default="p8,p64,p300")
    p.add_argument("--cut", type=int, default=100)
    p.add_argument("--cut2", type=int, default=13)
    p.add_argument("--per-layer", action="store_true")
    p.set_defaults(fn=cmd_carry)
    p = sub.add_parser("tokens", help="decision-margin-bucketed token agreement")
    p.add_argument("--fixtures", default="p8,p64,p300,dec8,p128,wide520f,chunk64b,dec300f")
    p.add_argument("--corpus-n", type=int, default=0,
                   help="also run N prompts of data/refs/corpus48; name them with corpus:")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--mma", action="store_true")
    p.set_defaults(fn=cmd_tokens)
    p = sub.add_parser("cost", help="cost model at the measured rates")
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--rows", type=int, default=512)
    p.set_defaults(fn=cmd_cost)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
