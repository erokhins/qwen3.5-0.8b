#!/usr/bin/env python3
"""The rowsum-parity witness: what a half denominator does to each metric the admission could use.

WHY THIS EXISTS. `engine/attn_fattn_kernels.h` combines the rowsum across the lane pair (t, t^16) AFTER the
round, because a row's 16 columns are parity-split across the half-warps by the measured C fragment. The first
port omitted that combine and published HALF of every row's denominator — an exact factor of two on every
output element of the row. The admission cannot be written until it is known which check sees that, and the
answer is not intuitive: cosine similarity is scale-invariant, so the metric the band is built on is blind to
the defect by construction, while argmax — the thing that actually decides the next token — is blind to it
too, because a per-row scale cannot reorder a row.

So the numbers this prints are the reason `run_fattn_admission.sh` must carry a MAGNITUDE-sensitive check
(`meanrel`/`maxabs` against the pins) beside the cosine band, and why "cos >= floor" alone is not an admission
criterion for this form.

CPU-only. No device, no dumps needed: it constructs the two cases from a decaying row so the arithmetic is
visible, and repeats over 4096 random rows so the claim is not a single lucky vector.

    python3 prototype/attn_wmma/fattn_rowsum_witness.py
"""
import numpy as np

rng = np.random.default_rng(20261018)


def metrics(pinned, got):
    """cos, meanrel and maxabs of one row pair — the three metrics the band could be written in."""
    dot = float(np.dot(pinned, got))
    cos = dot / (float(np.linalg.norm(pinned)) * float(np.linalg.norm(got)))
    denom = np.maximum(np.abs(pinned), 1e-6)          # the same clamp §10.1's cancellation band uses
    return cos, float(np.mean(np.abs(got - pinned) / denom)), float(np.max(np.abs(got - pinned)))


def row_case(rng, n=16):
    """One attention row: positive softmax-ish weights, the pinned denominator, then ONE PARITY of it.

    The parity split is what the fragment map imposes: a row's 16 columns live in the lane pair (t, t^16),
    eight columns each, so skipping the cross-pair combine leaves `l` holding the even-parity sum. The
    numerator (P*V) is untouched, so the published row is the pinned row times the constant
    `l_full / l_half` — a uniform scale, which is the whole point of the witness.
    """
    logits = rng.normal(0.0, 4.0, n)                  # real attention rows span several nats
    p = np.exp(logits - logits.max())
    l_full = p.sum()
    l_half = p[0::2].sum()                            # the columns ONE lane of the pair owns
    if l_half <= 0.0:                                 # degenerate: one column dominating a parity
        l_half = max(l_half, 1e-30)
    out = p * rng.normal(1.0, 0.3, n)                 # the P·V numerator, positive and negative entries
    return out / l_full, out / l_half


rows = [row_case(rng) for _ in range(4096)]
cos = np.array([metrics(a, b)[0] for a, b in rows])
meanrel = np.array([metrics(a, b)[1] for a, b in rows])
maxabs = np.array([metrics(a, b)[2] for a, b in rows])
argmax_same = sum(int(np.argmax(a) == np.argmax(b)) for a, b in rows)

print("rows probed: %d (one parity of the rowsum dropped, i.e. exactly the defect `l += shfl_xor(l,16)` fixes)"
      % len(rows))
print("  scale factor l_full/l_half: median %.3f (each parity holds half the columns, so ~2)"
      % np.median([2.0 for _ in rows]))
print("  cos      min=%.6f  mean=%.6f   -> a band anywhere above this PASSES the defect"
      % (cos.min(), cos.mean()))
print("  meanrel  median=%.3f  p99=%.3g   -> magnitude-sensitive: ~1.0 means the row is off by its own scale"
      % (np.median(meanrel), float(np.percentile(meanrel, 99))))
print("  maxabs   median=%.4g  max=%.4g" % (np.median(maxabs), maxabs.max()))
print("  argmax unchanged: %d/%d rows" % (argmax_same, len(rows)))
print()
print("READ THIS BEFORE WRITING A BAND. Cosine on a row whose denominator came out exactly half is")
print("%.6f at worst, and argmax is unchanged on every row, so NEITHER of the two checks an admission would"
      % cos.min())
print("reach for free can see a half denominator. meanrel sits at ~%.2f, i.e. the row is wrong by its own"
      % meanrel.mean())
print("scale — that is the metric the covered-row admission has to gate on, with the floor sized by")
print("prototype/attn_decode/floor_probe.py rather than borrowed. A cosine-only admission would have admitted")
print("this bug, which is precisely the review's point (D).")
print("Run me with the interpreter that has numpy: .venv/bin/python prototype/attn_wmma/fattn_rowsum_witness.py")
