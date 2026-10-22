#!/usr/bin/env python3
"""crossrow_arm1 — does a prefill row depend on a LATER row's token? No MTP code involved.

The question (`prototype/mtp_verify/CROSSROW.md`, arm 1): inside one wave body of `rows` query
positions, can row i's logits be moved by a token that belongs to a row after i? `rows > 1` is not
exotic — it is the shipping prefill path (`engine/kernels_attn.cpp:626` routes rows>1 to the
staged/tiled/wave forms), so a yes here is a causal leak in prefill, not an MTP curiosity.

The design has one job: make every legitimate source of difference identical between the two runs and
change ONLY the token VALUES of rows after i.

  * same prompt LENGTH on both sides, so `rows`, `lanes_last`, `split`, the wide-grid row-tile count and
    the GEMM M are all identical — a difference cannot be the geometry difference (a wave-vs-one-row
    comparison would be confounded by exactly that: `VERIFY-HANDOVER.md` measures the form gap at
    1-cos 1.555e-04, the same size as the effect being hunted);
  * same state: fresh process, prefix 0, no generation steps (`--steps 0`);
  * the same first K token ids on both sides; only positions K..L-1 differ.

Under a correctly causal engine, rows 0..K-1 must be BYTE-identical: row i's attention may read lanes
0..i, its GDN recurrence has seen tokens 0..i, and a batched GEMV/GEMM row's dot product touches
nothing but that row and the weights. Any movement at all is the finding.

Rows are compared as raw bytes first (that is the claim), and any row that differs is re-measured with
`prototype/mtp_verify/rowdiff.py`'s metrics (1-cos, meanrel, argmax) so a leak is sized, not just named.

Subcommands:
    prep     write the prompt CSVs into a work dir (CPU only)
    compare  byte-compare + metric-size the logits dumps of two runs (CPU only)

No GPU work happens in this file. No timing is taken or printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # prototype/, where bf16_ulp lives
import numpy as np  # noqa: E402
import torch  # noqa: E402
import bf16_ulp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VOCAB = 248320


def fixture_ids(name: str) -> list[int]:
    """A pinned fixture's prompt ids, read from its manifest — never typed from memory."""
    man = ROOT / "data" / "refs" / name / "manifest.json"
    if not man.is_file():
        raise SystemExit(f"{man} missing; enumerate data/refs/*/manifest.json (AGENTS' admission trap)")
    ids = json.loads(man.read_text()).get("prefill_ids") or []
    if not ids:
        raise SystemExit(f"{man} has no prefill_ids; pick a fixture that does")
    return [int(t) for t in ids]


def alt_suffix(base: list[int], start: int, pool: list[int], seed: int) -> list[int]:
    """Tokens for positions `start..len(base)-1` that differ from `base` at EVERY position.

    Drawn from a real fixture's own token pool so the run is not fed pathological ids, and stepped by a
    stride coprime-ish with the pool so the picks are spread rather than repeated. A suffix that
    happened to agree with the base at some position would silently shrink the perturbation, which is the
    vacuous-negative shape this repo keeps getting bitten by, so the difference is asserted, not hoped for.
    """
    out = []
    p = 0
    for i in range(start, len(base)):
        while True:
            t = pool[(seed * 7919 + p * 3571) % len(pool)]
            p += 1
            if t != base[i]:
                out.append(t)
                break
    return out


def write_csv(path: Path, ids: list[int]) -> None:
    # COMMA-separated: a space-joined prompt reads as ONE token and the run still succeeds
    # (AGENTS' `--ids-file` row). The count written is asserted, not assumed.
    path.write_text(",".join(str(t) for t in ids) + "\n")
    got = len(path.read_text().strip().split(","))
    if got != len(ids):
        raise SystemExit(f"{path}: wrote {len(ids)} ids, read back {got}; field count must be asserted")


def prep(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    base = fixture_ids(args.fixture)
    L = len(base) if args.length <= 0 else args.length
    base = base[:L]
    pool = fixture_ids(args.pool)
    K = args.prefix

    variants = {
        # R1: the reference prompt.
        "ref": list(base),
        # R2: first K ids identical, EVERY token after K different. Rows 0..K-1 must not move.
        "tail": list(base[:K]) + alt_suffix(base, K, pool, seed=11),
        # R3: only the LAST row's token differs — the sharpest "vary only rows after i" there is,
        #     because i = L-2 has exactly one row after it and that row's token is the only delta.
        "lastonly": list(base[:L - 1]) + alt_suffix(base, L - 1, pool, seed=23),
        # R4: R1 again, the determinism control. A difference vs R1 here is not a leak.
        "again": list(base),
    }
    for name, ids in variants.items():
        write_csv(work / f"{name}.csv", ids)
        diff_at = [i for i in range(L) if ids[i] != base[i]]
        print(f"prep {name:9s} n={len(ids):5d} differs-from-ref at positions "
              f"{diff_at[:6]}{'...' if len(diff_at) > 6 else ''} (count {len(diff_at)})")
    print(f"prep fixture={args.fixture} L={L} prefix_shared={K} pool={args.pool}")
    print(f"prep work={work}")
    return 0


def load_row(path: Path):
    """One logits row as F32, dtype read off the file's BYTE COUNT (AGENTS' dump-dtype row)."""
    raw = path.read_bytes()
    if len(raw) == VOCAB * 4:
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).astype(np.float32).copy()), "f32"
    if len(raw) == VOCAB * 2:
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return torch.from_numpy(u16.copy()).view(torch.bfloat16).float(), "bf16"
    raise SystemExit(f"{path}: {len(raw)} bytes is neither {VOCAB*4} (f32) nor {VOCAB*2} (bf16)")


def perturb(args) -> int:
    """One prompt per perturbed position: `pert<p>.csv` = the ref prompt with ONLY position p changed.

    This is the dependence graph, one edge-set per run: after running pert<p>, every row i < p that
    differs from the ref run is a row that can read position p — and rows i < p that are byte-identical
    are proof that row i cannot reach p. The graph, not a scalar, is what distinguishes a per-lane mask
    from a per-row mask from an aliased scratch slot.
    """
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    base = fixture_ids(args.fixture)
    L = len(base) if args.length <= 0 else args.length
    base = base[:L]
    pool = fixture_ids(args.pool)
    todo = [int(x) for x in args.positions.split(",") if x.strip() != ""]
    for p in todo:
        ids = list(base)
        # same generator as `tail`, seeded per position, and asserted different at p
        ids[p] = alt_suffix(base, p, pool, seed=101 + p)[0]
        assert ids[p] != base[p]
        write_csv(work / f"pert{p}.csv", ids)
    print(f"perturb fixture={args.fixture} L={L} positions={todo} work={work}")
    return 0


def sweep(args) -> int:
    """Print the dependence graph read off the pert<p> runs: rows < p that moved against the ref run."""
    ref = Path(args.ref)
    work = Path(args.work)
    base = fixture_ids(args.fixture)
    L = len(base) if args.length <= 0 else args.length
    todo = [int(x) for x in args.positions.split(",") if x.strip() != ""]
    edges = {}
    for p in todo:
        d = work / f"pert{p}"
        files = sorted(d.glob("logits*.bin"))
        if len(files) != L:
            raise SystemExit(f"{d}: {len(files)} logits files, expected {L}; the sweep population "
                             "must equal the case list (AGENTS' empty-parse row)")
        moved = []
        for i in range(p):                     # rows AFTER p moved legitimately and are not edges
            if (ref / f"logits{i}.bin").read_bytes() != (d / f"logits{i}.bin").read_bytes():
                xa, _ = load_row(ref / f"logits{i}.bin")
                xb, _ = load_row(d / f"logits{i}.bin")
                r = bf16_ulp.report(xa, xb, ulp_max=1)
                am = (int(xa.argmax()), int(xb.argmax()))
                moved.append((i, 1 - r["cos"], r["meanrel"], am))
        # p itself must differ (the perturbation reached the model) or the row is vacuous
        reached = (ref / f"logits{p}.bin").read_bytes() != (d / f"logits{p}.bin").read_bytes()
        edges[p] = (moved, reached)
    n_edges = sum(len(m) for m, _ in edges.values())
    print(f"sweep rows={L} cases={len(todo)} cross-row edges={n_edges}")
    for p in todo:
        moved, reached = edges[p]
        back = [p - i[0] for i in moved]       # distances: "how far back does the influence reach"
        sizes = " ".join(f"row{i}:1-cos={c:.2e} meanrel={mr:.2e} argmax={'SAME' if a[0]==a[1] else 'DIFFER'}"
                         for i, c, mr, a in moved)
        print(f"  perturbed row {p:4d}: reached_row={int(reached)} earlier_rows_moved={len(moved)} "
              f"distances_back={back} {sizes}")
    return 0


def rows_to_check(a_dir: Path, prefix: int, edge_row: int) -> list[int]:
    """The row population a comparison adjudicates, and it can never be empty (AGENTS' empty-parse row)."""
    rows = sorted(int(p.name[6:-4]) for p in a_dir.glob("logits*.bin"))
    if not rows:
        raise SystemExit(f"{a_dir} has no logits*.bin — an empty comparison cannot fail")
    if edge_row >= 0:
        return [r for r in rows if r == edge_row]
    return [r for r in rows if r < prefix]


def compare(args) -> int:
    a_dir, b_dir = Path(args.a), Path(args.b)
    rows_a = sorted(int(p.name[6:-4]) for p in a_dir.glob("logits*.bin"))
    rows_b = sorted(int(p.name[6:-4]) for p in b_dir.glob("logits*.bin"))
    if not rows_a:
        raise SystemExit(f"{a_dir} has no logits*.bin — empty comparison cannot fail (AGENTS)")
    if rows_a != rows_b:
        raise SystemExit(f"row populations differ: {len(rows_a)} vs {len(rows_b)}")
    shared = rows_to_check(a_dir, args.prefix, args.edge_row)
    after = [i for i in rows_a if i > max(shared)]
    print(f"compare a={a_dir} b={b_dir} rows={len(rows_a)} "
          f"adjudicated={len(shared)} row(s) control(rows after them)={len(after)}")

    n_same = 0
    moved = []
    for i in shared:
        pa, pb = a_dir / f"logits{i}.bin", b_dir / f"logits{i}.bin"
        if pa.read_bytes() == pb.read_bytes():
            n_same += 1
            continue
        xa, da = load_row(pa)
        xb, db = load_row(pb)
        r = bf16_ulp.report(xa, xb, ulp_max=1)
        am = int(xa.argmax()), int(xb.argmax())
        moved.append((i, 1 - r["cos"], r["meanrel"], r["ndiff"], am))
    print(f"shared rows byte-identical: {n_same}/{len(shared)}")
    if moved:
        print(f"SHARED ROWS MOVED: {len(moved)} of {len(shared)}  "
              "<- a row changed because a LATER row's token changed")
        for i, c1, mr, nd, am in moved:
            print(f"  row {i:5d}  1-cos={c1:.3e} meanrel={mr:.3e} ndiff={nd} "
                  f"argmax={am[0]}/{am[1]}{' SAME' if am[0] == am[1] else ' DIFFER'}")
    else:
        print("no shared row moved: no prefill-visible cross-row dependence at this geometry")
    # The rows that are SUPPOSED to differ, as the positive control that the perturbation reached the
    # model at all: if these are byte-identical too, the suffix change did nothing and the null above is
    # vacuous (the same "must-catch set" discipline as every negative control in this repo).
    ctl_same = sum(1 for i in after
                   if (a_dir / f"logits{i}.bin").read_bytes() == (b_dir / f"logits{i}.bin").read_bytes())
    print(f"positive control, the {len(after)} row(s) after the adjudicated ones (expected to differ): "
          f"{len(after) - ctl_same}/{len(after)} differ"
          + ("" if ctl_same < len(after) else "   <- VACUOUS: nothing moved anywhere"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--fixture", default="p128")
    p.add_argument("--pool", default="wide640f")
    p.add_argument("--work", default="/tmp/crossrow")
    p.add_argument("--prefix", type=int, default=64)
    p.add_argument("--length", type=int, default=0, help="0 = the whole fixture prompt")
    p.set_defaults(fn=prep)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--prefix", type=int, default=64)
    c.add_argument("--edge-row", type=int, default=-1,
                   help="adjudicate only this one row (the neighbour of a single-position perturbation)")
    c.set_defaults(fn=compare)
    pt = sub.add_parser("perturb")
    pt.add_argument("--fixture", default="p128")
    pt.add_argument("--pool", default="wide640f")
    pt.add_argument("--work", default="/tmp/crossrow")
    pt.add_argument("--length", type=int, default=0)
    pt.add_argument("--positions", required=True, help="comma list of positions to perturb, one file each")
    pt.set_defaults(fn=perturb)
    sw = sub.add_parser("sweep")
    sw.add_argument("--ref", required=True)
    sw.add_argument("--work", required=True)
    sw.add_argument("--fixture", default="p128")
    sw.add_argument("--length", type=int, default=0)
    sw.add_argument("--positions", required=True)
    sw.set_defaults(fn=sweep)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
