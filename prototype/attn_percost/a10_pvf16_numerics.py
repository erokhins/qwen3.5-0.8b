"""Compare two engine output dirs: logits cosine / meanrel / argmax, and the sample stream.

Used by the f16 P·V device pass (`run_a10_pvf16_device.sh`): arm-vs-arm on the same prompt, where the only
difference is `--attn-fattn-pvf16`. Reports the aggregates D5 of `A10-F16-IMPL.md` asked for — cos and meanrel
are the §10.1 aggregates, and argmax movement is the decision-relevant one because a token flip is what a
gate actually trips on. No band is applied here; the numbers go to the gates and to the report.
"""
import sys
from pathlib import Path

import numpy as np


def load_logits(p: Path) -> np.ndarray:
    """The engine writes the final logits as F32, one value per vocab entry.

    First version of this loader assumed BF16 because the dumps elsewhere in the tree are BF16, and it was
    silently catastrophic rather than obviously wrong: reading 4-byte floats as pairs of uint16 and widening
    produced values around 1e35, so cos/meanrel/maxabs all came back NaN and `argmax MOVED` on every row while
    the sample stream still agreed 10,261/10,266 — a divergence story invented entirely by the reader. The
    shape is now asserted against a plausible vocab and finiteness is asserted too, because a reader that
    cannot fail is not a reader.
    """
    x = np.fromfile(p, dtype="<f4")
    assert x.size > 100_000, f"{p}: {x.size} floats — too few to be a logits row, wrong dtype?"
    assert np.isfinite(x).all(), f"{p}: non-finite values in the dump — wrong dtype or a broken run"
    return x


def main(a: str, b: str) -> int:
    da, db = Path(a), Path(b)
    names = sorted([p.name for p in da.glob("logits*.bin") if (db / p.name).exists()])
    if not names:
        sys.exit("NO-LOGITS: no shared logits*.bin between the two dirs — not a zero, an absent input")
    for n in names:
        x, y = load_logits(da / n), load_logits(db / n)
        if x.shape != y.shape:
            print(f"{n}: SHAPE MISMATCH {x.shape} vs {y.shape}")
            continue
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        cos = float(x @ y / (nx * ny)) if nx and ny else float("nan")
        denom = np.maximum(np.abs(x), 1e-6)
        meanrel = float(np.mean(np.abs(x - y) / denom))
        amax_x, amax_y = int(np.argmax(x)), int(np.argmax(y))
        ndiff = int(np.sum(x != y))
        print(f"{n:16s} cos={cos:.8f} meanrel={meanrel:.3e} argmax {amax_x}->{amax_y} "
              f"{'SAME' if amax_x == amax_y else 'MOVED'}  ndiff={ndiff}/{x.size} "
              f"maxabs={float(np.max(np.abs(x - y))):.3e}")
    sa, sb = da / "sampled.txt", db / "sampled.txt"
    if sa.exists() and sb.exists():
        la, lb = sa.read_text().split(), sb.read_text().split()
        same = sum(1 for u, v in zip(la, lb) if u == v)
        print(f"sampled.txt: {same}/{max(len(la), len(lb))} positions agree "
              f"(len {len(la)} vs {len(lb)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
