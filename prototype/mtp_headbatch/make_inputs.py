#!/usr/bin/env python3
"""Build and PROVE the inputs for prototype/mtp_headbatch. CPU only — no device, no timing.

Two jobs, in this order, because a rig whose input is wrong prices nothing.

1. THE TIED TABLE. `model.language_model.embed_tokens.weight`, BF16 [248320, 1024] =
   508,559,360 B, dumped raw row-major. Reuses /tmp/lmhead/W.bin when it is present and the right
   size (it is prototype/lm_head_gemv's own dump, and copying 485 MiB twice per session is waste);
   otherwise it extracts from the shard with a rank- and shape-checked read, per the AGENTS rule that
   a shape claim written as a guess is a bug.

2. THE QUERY ROWS. M rows of H=1024 F32, each taken from a **real pinned hidden state**, because a
   synthetic x would price the batched head on magnitudes the engine never sees:
       row 0  data/refs/dec8/final_hidden.npy    (the MTP block's own `h` input, §09.1)
       row 1  data/refs/smoke/final_hidden.npy
       row 2  data/refs/p300/final_hidden.npy    (row 100 of a 277-row prefill tile)
       row 3  data/refs/p128/final_hidden.npy
   All four are F32 in the pins (asserted on load, because a uint16 bit-store read as float is the
   garbage MTP-GATEZERO §2 hit); `p300` is the only multi-row one, which is also why row 2 is taken
   from it rather than two rows from a single-token fixture.
   `final_hidden` is the vector after the final RMSnorm, i.e. exactly the lm_head input (§03.4), so
   these are the same distribution the head is fed in production — and row 0 is the vector
   `prototype/mtp_gatezero/mtp_census.py` uses as the drafter's `h`, so this rig and P7's draft path
   read the same fixture.

   WHY THERE ARE M ROWS AND NOT M COPIES. The failure mode this rig exists to catch is a batched head
   that computes row 1 against row 0's data (see gate.py). That bug is invisible if all rows are
   equal: every output would be correct and every control would pass vacuously. Distinct real rows
   are therefore a correctness requirement, not a nicety, and `--verify` asserts they are distinct.

   THE PAIRING IS PROVED BEFORE ANYTHING IS WRITTEN, with the same discipline as
   `prototype/mtp_gatezero/pin_pair_search.py`: row 0 times the table, in torch F32, must reproduce
   `dec8/decode7.logits_last` (cos >= 0.9999 and matching argmax). The negative half is built in:
   the same row against the fixture's BARE `logits_last.npy` must NOT reproduce it — §10.5's
   last-writer-wins trap says the bare name is a different step's row, and if that check ever passes
   the fixture set has changed under this rig and the gate says so instead of quietly pricing garbage.

    .venv/bin/python prototype/mtp_headbatch/make_inputs.py [--force]
    .venv/bin/python prototype/mtp_headbatch/make_inputs.py --verify
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WORK = Path("/tmp/mtp_headbatch")
EMBED_KEY = "model.language_model.embed_tokens.weight"
CKPT = ROOT / "models/Qwen3.5-0.8B"
V, H = 248320, 1024
SHIPPED_W = Path("/tmp/lmhead/W.bin")      # prototype/lm_head_gemv's dump of the same tensor

# (destination row, pin file, index into that pin). The order is the batch order the rig gets.
X_ROWS = [
    ("dec8/final_hidden.npy", 0),
    ("smoke/final_hidden.npy", 0),
    ("p300/final_hidden.npy", 100),
    ("p128/final_hidden.npy", 0),
]
PAIR_POS = ("dec8/decode7.logits_last.npy", "paired: decode7's row is the one this x produced")
PAIR_NEG = ("dec8/logits_last.npy", "unpaired: the bare name is another step's row (§10.5)")
PAIR_COS = 0.9999


def sha(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_pin(rel: str, idx: int) -> np.ndarray:
    a = np.load(ROOT / "data/refs" / rel)
    if a.dtype != np.float32:
        raise SystemExit(f"{rel}: dtype {a.dtype}; these pins hold F32 hidden rows and a bit-store "
                         f"read as float is garbage that looks like unwritten memory")
    flat = np.asarray(a, dtype=np.float32).reshape(-1, H)
    if idx >= flat.shape[0]:
        raise SystemExit(f"{rel}: asked for row {idx} but it has {flat.shape[0]}")
    row = np.ascontiguousarray(flat[idx])
    if row.shape != (H,):
        raise SystemExit(f"{rel}: row is {row.shape}, expected ({H},)")
    return row


def table(force: bool) -> Path:
    out = WORK / "W.bin"
    if out.exists() and out.stat().st_size == V * H * 2 and not force:
        print(f"table: using existing {out} ({out.stat().st_size} B)")
        return out
    if SHIPPED_W.exists() and SHIPPED_W.stat().st_size == V * H * 2 and not force:
        os.link(SHIPPED_W, out)          # same bytes, no 485 MiB copy
        print(f"table: hard-linked from {SHIPPED_W}")
        return out
    for f in sorted(glob.glob(str(CKPT / "*.safetensors"))):
        with safe_open(f, "pt") as s:
            if EMBED_KEY not in s.keys():
                continue
            t = s.get_tensor(EMBED_KEY)
            if tuple(t.shape) != (V, H) or t.dtype != torch.bfloat16 or t.ndim != 2:
                raise SystemExit(f"{EMBED_KEY}: rank {t.ndim} shape {tuple(t.shape)} dtype "
                                 f"{t.dtype}; this rig expects rank-2 BF16 [{V},{H}]")
            t.contiguous().view(torch.uint16).numpy().tofile(out)
            print(f"table: extracted {EMBED_KEY} from {Path(f).name} -> {out}")
            return out
    raise SystemExit(f"{EMBED_KEY} not found under {CKPT}")


def price_row(x: np.ndarray, W_bits: np.ndarray) -> np.ndarray:
    """x @ W^T in torch F32 — speccheck's own op, an implementation independent of the rig."""
    torch.set_num_threads(16)
    xt = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    wt = torch.from_numpy(W_bits).view(torch.bfloat16).reshape(V, H).float()
    return (xt @ wt.T).reshape(-1).numpy()


def pairing(x: np.ndarray, W_bits: np.ndarray) -> int:
    ok = True
    for rel, note in (PAIR_POS, PAIR_NEG):
        ref = np.load(ROOT / "data/refs" / rel).reshape(-1).astype(np.float32)
        got = price_row(x, W_bits)
        cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
        am = int(np.argmax(got)) == int(np.argmax(ref))
        want = rel == PAIR_POS[0]
        good = (cos >= PAIR_COS and am) if want else (cos < PAIR_COS or not am)
        ok &= good
        print("  pairing %-34s cos %.9f argmax %s -> %s (%s)"
              % (rel, cos, "match" if am else "DIFFER",
                 "reproduces" if cos >= PAIR_COS and am else "does not reproduce",
                 "required" if good else "<<< WRONG"))
    if not ok:
        print("PAIRING FAILED. The fixture pairing this rig's row-0 witness rests on no longer "
              "holds; fix the fixture (or re-point PAIR_POS), do not loosen this check.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-extract the table even if it exists")
    ap.add_argument("--verify", action="store_true", help="re-check what is on disk; write nothing")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    man = WORK / "manifest.json"

    if args.verify:
        if not man.exists():
            raise SystemExit(f"--verify: no {man}; run make_inputs.py first")
        m = json.loads(man.read_text())
        bad = []
        for f, want in m["sizes"].items():
            got = (WORK / f).stat().st_size if (WORK / f).exists() else -1
            if got != want:
                bad.append(f"{f}: {got} != {want}")
        rows = np.fromfile(WORK / "x4.bin", dtype=np.float32).reshape(4, H)
        for i in range(4):
            for j in range(i + 1, 4):
                if np.array_equal(rows[i], rows[j]):
                    bad.append(f"x rows {i} and {j} are identical; a batched head that ignores the "
                               f"row index would be undetectable")
        if bad:
            print("VERIFY FAILED")
            for b in bad:
                print("  -", b)
            return 1
        print("VERIFY ok: %d files, sizes match the manifest; the 4 x rows are distinct; "
              "sizes were re-read, not remembered" % len(m["sizes"]))
        for f in sorted(m["sizes"]):
            print("  %-14s %12d B" % (f, m["sizes"][f]))
        return 0

    W = table(args.force)
    sizes = {"W.bin": W.stat().st_size}

    rows = np.stack([load_pin(rel, idx) for rel, idx in X_ROWS])
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rows[i], rows[j]):
                raise SystemExit(f"x rows {i} and {j} are identical: a batched head that ignores "
                                 f"the row index could not be detected on this input")
    W_bits = np.fromfile(W, dtype=np.uint16)
    print("pairing witness (torch F32, independent of the rig):")
    rc = pairing(rows[0], W_bits)
    if rc:
        return rc

    for m_ in (1, 2, 4):
        f = WORK / f"x{m_}.bin"
        np.ascontiguousarray(rows[:m_]).tofile(f)
        sizes[f.name] = f.stat().st_size
    sizes["x4.bin"] = (WORK / "x4.bin").stat().st_size
    # One file per single row: the cross-M identity check needs "that row, run alone", which is a
    # 1-row launch with a different x, not a re-run of the same batch.
    for i in range(4):
        f = WORK / f"xrow{i}.bin"
        np.ascontiguousarray(rows[i:i + 1]).tofile(f)
        sizes[f.name] = f.stat().st_size
    # A single-row file for the shipped-kernel baseline arm, which reads exactly HID floats.
    rows[:1].tofile(WORK / "x1_shipped.bin")
    sizes["x1_shipped.bin"] = (WORK / "x1_shipped.bin").stat().st_size

    man.write_text(json.dumps({
        "table": {"key": EMBED_KEY, "dtype": "BF16", "shape": [V, H], "bytes": sizes["W.bin"],
                  "sha256": sha(WORK / "W.bin")},
        "x_rows": [{"pin": rel, "index": idx} for rel, idx in X_ROWS],
        "x_sha256": {f"row{i}": sha_of_row(rows[i]) for i in range(4)},
        "pairing": {"positive": PAIR_POS[0], "negative": PAIR_NEG[0], "cos_min": PAIR_COS},
        "sizes": sizes,
    }, indent=1) + "\n")
    print("wrote", man, "and", ", ".join(sorted(sizes)))
    print("next: .venv/bin/python prototype/mtp_headbatch/gate.py --selftest")
    return 0


def sha_of_row(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
