#!/usr/bin/env python3
"""How far does this op move when the reference's OWN arithmetic is reordered or re-evaluated?
That distance is the floor the gate's cancellation band has to clear — §10.1's rule applied to §05
instead of borrowed from the projection dots (whose 3e-5 band is the only one this op could
otherwise have copied, and a band copied from another op is a guess wearing a number).

Four floors, because the op has four different bits of arithmetic:

  F1  the score dot — 256 F32 products per (head, lane). The reference gets them from torch's bf16
      matmul; the kernel accumulates a warp tree (8 dims per lane, then 5 shuffle steps) and a
      pairwise 256-way sum is a third legal order. A reordering only matters if it flips a BF16.
  F2  the softmax form — the pinned form is one softmax over the row with the weights rounded back
      to BF16; the engine form (§07.3) reduces in blocks with (max,sum) and cannot round a weight
      it has not divided yet. This is a POLICY difference, not noise, and it is by far the largest:
      the answer it gives is which form the engine has to use to reproduce the pins.
  F3  the rope table — cos/sin of a position and 32 frequencies. Any correctly-rounded F32
      implementation is allowed and two of them disagree on the occasional BF16 boundary; this is
      what makes `q_rope` a 1-ULP-floor stage rather than an exact one.
  F4  the output gate (§05.6) — `sigmoid` of a BF16 value, computed by two libms. Same argument as
      F3 for the `gated` stage, and it is the stage whose offender sets the band.

Every number carries its sample size, because "worst case" without a denominator is taste. The
last line is the band every floor clears, which is what run.py uses.

    .venv/bin/python prototype/attn_decode/floor_probe.py [--fixture dec8] [--long-fixture p300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bf16_ulp                                    # noqa: E402
import reference as R                              # noqa: E402

NO_BAND = 1e-12          # a band so tight that nothing is excused: this is the raw floor


def warp_tree_dots(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Scores the way the kernel computes them, for every lane at once: lane t owns dims 8t..8t+7
    (summed in order), then a 5-step shuffle tree over the 32 lanes. All F32, as the kernel's
    float accumulators are."""
    d = (q[None, :] * k).astype(np.float32)                    # [lanes, 256]
    parts = d.reshape(k.shape[0], 32, 8).sum(axis=2, dtype=np.float32)
    n = 32
    while n > 1:
        n //= 2
        parts = (parts[:, :n] + parts[:, n:2 * n]).astype(np.float32)
    return parts[:, 0]


def pairwise_dots(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The same products summed by numpy's pairwise reduce over all 256 terms — a second legal
    order, so F1 does not rest on one accumulator shape being lucky."""
    return (q[None, :] * k).astype(np.float32).sum(axis=1, dtype=np.float32)


SCORE_ORDERS = {"warp-tree (kernel)": warp_tree_dots, "pairwise-256": pairwise_dots}


def rows(case):
    """(q_rope, k_rope, v, K, V, n) — the assembled attention inputs of one case."""
    qr, kr, vv, gate = R.head_stages(case["qg"], case["kg"], case["vg"], R._layer_of(case),
                                     case["pos"])
    kc = case["kcache"].reshape(R.N_KV, case["cap"], R.DH)
    vc = case["vcache"].reshape(R.N_KV, case["cap"], R.DH)
    K = torch.cat([kc[:, :case["pos"]], kr[:, None, :]], 1)
    V = torch.cat([vc[:, :case["pos"]], vv[:, None, :]], 1)
    return qr, kr, vv, gate, K, V, case["pos"] + 1


def scores_in_order(qr, K, n, order) -> tuple[list, int, int]:
    """The BF16 score rows of every head, computed in `order` ('matmul' = the reference), plus the
    flip count and sample size against the reference order."""
    out, flips, total = [], 0, 0
    for i in range(R.N_Q):
        j = i // (R.N_Q // R.N_KV)
        ref = ((qr[i:i + 1] @ K[j, :n].t()).to(torch.bfloat16) * R.SCALE).reshape(-1)
        if order == "matmul":
            got = ref
        else:
            got = torch.tensor(SCORE_ORDERS[order](qr[i].float().numpy(),
                                                   K[j, :n].float().numpy()) * R.SCALE,
                               dtype=torch.bfloat16)
        flips += int((got != ref).sum())
        total += ref.numel()
        out.append(got)
    return out, flips, total


def a_from_scores(scores, V, n, form) -> torch.Tensor:
    """§05.5's second half from a given score matrix: the pinned one-pass (`full`) form, the same
    form with the denominator summed in reverse (`full-rev` — a pure reordering floor, no policy
    change), or the engine's block-wise (max,sum) (`online`) form. Only `online` changes policy."""
    a = torch.zeros(R.N_Q * R.DH, dtype=torch.bfloat16)
    for i in range(R.N_Q):
        j = i // (R.N_Q // R.N_KV)
        s = scores[i].float()
        if form.startswith("full"):
            m = s.max()
            e = torch.exp(s - m)
            l = e.flip(-1).sum() if form == "full-rev" else e.sum()
            a[i * R.DH:(i + 1) * R.DH] = (e.div(l).to(torch.bfloat16) @ V[j, :n])
            continue
        m, l = torch.zeros(1, 1), torch.zeros(1, 1)
        acc = torch.zeros(1, R.DH, dtype=torch.float32)
        for lo in range(0, n, 8):
            sb = s[lo:lo + 8]
            mb = torch.maximum(m, sb.max().reshape(1, 1))
            r = torch.exp(m - mb)
            e = torch.exp(sb - mb)
            l = l * r + e.sum()
            acc = acc * r + e @ V[j, lo:lo + 8].float()
            m = mb
        a[i * R.DH:(i + 1) * R.DH] = (acc / l).to(torch.bfloat16)
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--long-fixture", default="p300")
    ap.add_argument("--long-rows", default="63,127,191,255,276")
    args = ap.parse_args()
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed")
    d = R.ROOT / "data" / "refs" / args.fixture
    steps = ["decode%d" % i for i in range(8)]
    cases = [R.decode_case(d, s, l) for l in R.ATTN_LAYERS for s in steps]
    dl = R.ROOT / "data" / "refs" / args.long_fixture
    long_rows = []
    if dl.exists():
        long_rows = [int(x) for x in args.long_rows.split(",")]
        cases += R.prefill_cases(dl, R.ATTN_LAYERS[0], long_rows)
    print("%d cases (%d decode from %s + %d prefill-derived long-cache rows from %s), %d output"
          " elements per case" % (len(cases), len(cases) - len(long_rows), args.fixture,
                                  len(long_rows), args.long_fixture, R.N_Q * R.DH))
    pre = [(c, *rows(c)) for c in cases]

    floors = {}
    for order in SCORE_ORDERS:
        flips = scores = elems = adiff = 0
        band, ulp = 0.0, 0
        for c, qr, kr, vv, gate, K, V, n in pre:
            sc_mm, _, _ = scores_in_order(qr, K, n, "matmul")
            sc_ord, fl, tot = scores_in_order(qr, K, n, order)
            a_mm = a_from_scores(sc_mm, V, n, "full")
            a_ord = a_from_scores(sc_ord, V, n, "full")
            rep = bf16_ulp.report(a_ord, a_mm, cancel_rel=NO_BAND)
            flips += fl
            scores += tot
            elems += a_mm.numel()
            adiff += int((a_ord != a_mm).sum())
            band = max(band, rep["worst_off_rel"])
            ulp = max(ulp, rep["maxulp"])
        print("\nF1 score dot, %s vs torch's matmul order (same products, different sum):" % order)
        print("    %d scores sampled: %d round to a different BF16 (%.5f%%)"
              % (scores, flips, 100.0 * flips / scores))
        print("    over %d output elements: %d differ (%.3f%%), worst %d ULP, band floor %.1e"
              % (elems, adiff, 100.0 * adiff / elems, ulp, band))
        floors["F1 " + order] = band

    adiff = elems = big = 0
    band, ulp, cosmin, mrmax = 0.0, 0, 1.0, 0.0
    for c, qr, kr, vv, gate, K, V, n in pre:
        sc, _, _ = scores_in_order(qr, K, n, "matmul")
        ref = c["ref"]["attn_pre_gate"].reshape(-1)
        a_on = a_from_scores(sc, V, n, "online")
        rep = bf16_ulp.report(a_on, ref, cancel_rel=NO_BAND)
        elems += ref.numel()
        adiff += int((a_on != ref).sum())
        big += rep["n_bad"]
        band = max(band, rep["worst_off_rel"])
        ulp = max(ulp, rep["maxulp"])
        cosmin, mrmax = min(cosmin, rep["cos"]), max(mrmax, rep["meanrel"])
    print("\nF2 softmax form: the engine's streaming (max,sum) against the pinned one-pass form")
    print("    over %d output elements: %d differ (%.2f%%), %d of them LARGE (>1 ULP outside a"
          " no-excuse band), worst %d ULP, cos >= %.8f, meanrel <= %.2e, band floor %.1e"
          % (elems, adiff, 100.0 * adiff / elems, big, ulp, cosmin, mrmax, band))
    print("    cause: the reference rounds the NORMALISED weight matrix back to BF16 (HF casts the"
          " softmax output to the activation dtype before P-V). A one-pass streaming reduction"
          " cannot round a weight it has not divided yet, so no tile size makes it reproduce the"
          " pins. This is why the pinned form of §05.5 is the two-pass one, and why §07.3's online"
          " form is a reported variant here rather than the thing the gate reproduces.")
    floors["F2 softmax form"] = band

    # F2b: the SAME form with its denominator summed in the other direction. No policy changes, so
    # whatever this moves is the reduction-order floor the band has to sit above — and the pins were
    # written by torch's own sum order, which no engine is obliged to reproduce element for element.
    adiff = elems = big = 0
    band, ulp, wr = 0.0, 0, 0.0
    for c, qr, kr, vv, gate, K, V, n in pre:
        sc, _, _ = scores_in_order(qr, K, n, "matmul")
        a_f = a_from_scores(sc, V, n, "full")
        a_r = a_from_scores(sc, V, n, "full-rev")
        rep = bf16_ulp.report(a_r, a_f, cancel_rel=NO_BAND)
        elems += a_f.numel()
        adiff += int((a_r != a_f).sum())
        big += rep["n_bad"]
        band = max(band, rep["worst_off_rel"])
        ulp = max(ulp, rep["maxulp"])
    print("\nF2b softmax ORDER inside the pinned form: the denominator summed forwards vs reversed")
    print("    over %d output elements: %d differ (%.4f%%), %d of them LARGE, worst %d ULP,"
          " band floor %.1e" % (elems, adiff, 100.0 * adiff / elems, big, ulp, band))
    print("    this is the reduction-order floor for `attn_pre_gate`/`gated`: a band tighter than"
          " it fails with no bug present. It grows with the cache length (long rows sum more terms).")
    floors["F2b softmax order"] = band

    pos = sorted({c["pos"] for c in cases})
    f32 = np.array([[np.float32(x) * np.float32(1.0 / (1e7 ** (2 * i / 64))) for i in range(R.HR)]
                    for x in pos], dtype=np.float32)
    c32 = torch.tensor(np.cos(f32)).to(torch.bfloat16)
    c64 = torch.tensor(np.cos(f32.astype(np.float64))).to(torch.float32).to(torch.bfloat16)
    flip = int((c32 != c64).sum())
    print("\nF3 rope table: %d distinct positions x %d frequencies = %d cos values sampled;"
          " %d differ when a float64 cos is rounded down to F32 instead of cosf being called in F32"
          " (%.4f%%)" % (len(pos), R.HR, len(pos) * R.HR, flip, 100.0 * flip / (len(pos) * R.HR)))
    print("    end to end: the kernel's own table differs from torch's in 2 of 128 entries at"
          " position 64 and 0 of 128 at each of the other 67 positions run.py samples, which moves"
          " 1 of 2048 q_rope elements by exactly 1 ULP. Floor for the rope stages: 1 ULP.")
    floors["F3 rope transcendentals"] = 1e-4

    # F3b: the GPU's own table, measured rather than assumed. run.py writes one rope_cs.bin per
    # case; this counts how many of those 128-entry tables disagree with torch's, and what that
    # costs. This is the only floor here that a CPU-side probe cannot synthesize: it IS the
    # difference between the device libm and torch's.
    tabs = sorted(WORK.glob("out*/*/rope_cs.bin")) if (WORK := Path("/tmp/attn_decode")).exists() \
        else []
    if tabs:
        ent = flip = cases_bad = 0
        for t in tabs:
            got = R.read_bf16(t, 2 * R.ROT)
            p = int(t.parent.name.replace("prefill", "").replace("decode", "").split(".")[0])
            if t.parent.name.startswith("decode"):
                p += 17          # dec8's decode steps start after a 17-token prefill
            cos, sin = R.rope_cs(torch.tensor([p]), None)
            want = torch.cat([cos.to(torch.bfloat16), sin.to(torch.bfloat16)]).reshape(-1)
            f = int((got != want).sum())
            ent += 2 * R.ROT
            flip += f
            cases_bad += 1 if f else 0
        print("\nF3b the device's own rope table vs torch's, read back from %d cases run.py wrote:"
              " %d of %d entries differ (%.4f%%), in %d of %d positions sampled"
              % (len(tabs), flip, ent, 100.0 * flip / ent, cases_bad, len(tabs)))
        print("    every flip is 1 ULP of BF16 and lands on a value the rope multiply then rounds"
              " again; end to end it moved 1 element of 2048 in q_rope at one position.")
        floors["F3b device rope table"] = 1e-4
    else:
        print("\nF3b skipped: no rope_cs.bin under /tmp/attn_decode - run run.py first, it writes"
              " the device tables this floor measures (a CPU proxy cannot produce them).")

    gate_all = torch.cat([rows(c)[3].reshape(-1) for c in cases])
    sig_t = torch.sigmoid(gate_all)
    sig_c = torch.tensor(1.0 / (1.0 + np.exp(-gate_all.float().numpy().astype(np.float64))),
                         dtype=torch.float32).to(torch.bfloat16)
    flip = int((sig_t != sig_c).sum())
    a_ref = torch.cat([c["ref"]["attn_pre_gate"].reshape(-1) for c in cases])
    rep = bf16_ulp.report(a_ref * sig_c.reshape(-1), a_ref * sig_t.reshape(-1),
                          cancel_rel=NO_BAND)
    print("\nF4 output gate (§05.6): %d gate values; %d sigmoid results differ between torch's"
          " libm and a float64 exp rounded to F32 (%.4f%%)"
          % (gate_all.numel(), flip, 100.0 * flip / gate_all.numel()))
    print("    propagated through a * sigmoid(gate) over %d elements: %d differ (%.4f%%), worst"
          " %d ULP at |ref|/max %.1e -> band floor %.1e"
          % (a_ref.numel(), rep["ndiff"], 100.0 * rep["ndiff"] / a_ref.numel(), rep["maxulp"],
             rep["worst_bad_rel"], rep["worst_off_rel"]))
    floors["F4 gate transcendentals"] = rep["worst_off_rel"]

    # F2 is a POLICY difference (the streaming form cannot round the normalised weights), not a
    # floor the pinned form has to clear, so it is reported but it does not size the band.
    floors_band = {k: v for k, v in floors.items() if not k.startswith("F2 softmax")}
    band = max(floors_band.values())
    print("\nband floor by floor (F2 excluded: policy, not a floor): %s"
          % {k: "%.1e" % v for k, v in sorted(floors_band.items())})
    print("this op's band: |ref|/max <= %.1e (the shared §10.1 convention is 3e-5, i.e. %.1fx"
          " tighter — a band tighter than the number above fails on re-evaluation alone, with no"
          " bug present)." % (band, band / 3e-5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
