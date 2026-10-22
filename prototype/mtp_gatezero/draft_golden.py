#!/usr/bin/env python3
"""The S09.2 draft step: byte-witnessed traffic ledger, pinned inputs, golden intermediates.

P7 gate zero, part 4 — the rig a future MTP kernel gets signed against, and the thing that turns the
draft's traffic claim from arithmetic into a measurement. HF has no MTP class for `qwen3_5`
(`specification/09-mtp-draft-layer.md` header), so there is no reference dump to pin against: what this
writes is a GOLDEN derived from the shipped CPU form (`speccheck/mtp.py`), plus the controls that make
such a golden mean something.

1. INPUT CONTRACT, from pins. `h` comes from a pinned dump (`final_hidden.npy`). The script re-proves
   the claim on the way in: `E @ h` must reproduce the paired pinned logits row at cos >= 0.9999, or
   it aborts. `y_next` and the position are READ OFF the fixture (the argmax of that row, and the count
   of tokens the fixture fed), never typed in.

2. TRAFFIC, WITNESSED NOT ARITHMETIC. The drafter runs through a weights dict that records every key
   it indexes. The output is the set of tensors a draft step actually reads and their bytes, which both
   (a) witnesses the 39.01 MiB block + 485.0 MiB tied table that `c` is priced from, and (b) is a
   must-not-touch witness: a target-layer weight in that set means a drafter reading the wrong book.

3. GOLDEN. For a TWO-STEP chain (§09.3 - the second step attends the MTP cache and needs rope to
   matter), `u`, `x_d` and the draft logits row are dumped bit-exactly. BF16 goes to disk through a
   uint16 view: torch refuses `bf16.numpy()`, and `.to(uint16)` *casts* where `.view(uint16)`
   bitcasts, so confusing them writes garbage that reads like unwritten memory
   (`prototype/gdn_gated_norm/results.md` S8). `--check` re-derives and compares: that is the gate an
   engine-side `mtp_layer` kernel would be admitted against.

4. CONTROLS, DERIVED FROM GEOMETRY (AGENTS.md: a must-catch set that is not derived rots into
   "everything", and a negative test that catches everything catches nothing).
   * MUST-CATCH `h_source=pre_norm` and `fc_order=he` — the two knobs S09.5 settled by accept rate.
   * MUST-CATCH a position shift judged on the LAST step, and MUST-NOT-CATCH the same shift judged on
     step 0: with a one-entry MTP cache the attention row is a singleton and softmax of a singleton is
     1.0 whatever the scores are, so rope cannot move step 0. The first version of this script asserted
     a position shift on a ONE-step case and it failed - the failure was the geometry, and it is the
     reason the gated case is two steps long.
   * MUST-NOT-CATCH zeroing every tensor outside the read set, and perturbing step 2's input cannot
     move step 1's output (the chain feeds forward).

    .venv/bin/python prototype/mtp_gatezero/draft_golden.py [--fixture dec8] [--write] [--check]

CPU only. torch CPU bf16. No device, no timing taken.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speccheck.mtp import MP, MTPDraft                                    # noqa: E402
from speccheck.specimpl import Weights                                    # noqa: E402

EMBED = "model.language_model.embed_tokens.weight"
GOLD = pathlib.Path(__file__).resolve().parent / "golden"
NSTEPS = 2                       # >= 2 so the MTP cache and rope are inside the gated geometry


class RecDict(dict):
    """A weights dict that records every key indexed through it."""

    def __init__(self, src: dict):
        super().__init__(src)
        self.counts: dict[str, int] = {}

    def __getitem__(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1
        return dict.__getitem__(self, k)


class Shim:
    """`Weights` is only ever used as `w.p[...]`, so a shim avoids re-reading the shard."""

    def __init__(self, p: RecDict):
        self.p = p


def save_bf16(path: pathlib.Path, t: torch.Tensor):
    """BF16 to disk as its bits, and only as its bits."""
    bits = t.detach().to(torch.bfloat16).reshape(-1).view(torch.uint16).numpy()
    np.save(path, bits)


def load_bf16(path: pathlib.Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).view(torch.bfloat16)


def pin_row(d: pathlib.Path, name: str) -> np.ndarray:
    """A pinned logits/hidden row as F64, handling both storage forms the refs tree uses."""
    a = np.load(d / f"{name}.npy").reshape(-1)
    if a.dtype == np.uint16:
        return torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16).float().numpy()
    return a.astype(np.float64)


def run_chain(w, h: torch.Tensor, y_next: int, p: int, n: int = NSTEPS,
              h_source="post_norm", fc_order="eh", h_of_step=None, uniform_shift=0,
              last_shift=0):
    """`n` S09.2 draft steps chained as §09.3 says (x_d becomes the next `h`, the proposal the next
    `y_next`, position +1 each step). Returns (per-step records, read ledger).

    `uniform_shift` moves every position together (rope is relative, so it must NOT move anything - a
    derived must-not-catch); `last_shift` moves only the final step's position, which changes its
    relative distance to the cached draft keys and so must move the attention output once the cache
    holds more than one lane.
    """
    rec = RecDict(w.p)
    m = MTPDraft(Shim(rec), h_source=h_source, fc_order=fc_order)
    out = []
    hd, yd = h, y_next
    for j in range(n):
        dbg: dict = {}
        pos = p + j + uniform_shift + (last_shift if j == n - 1 else 0)
        d, x_d = m.draft(yd, hd, pos, dbg)
        out.append({"j": j, "pos": pos, "proposal": d, "u": dbg["mtp.u"], "x_d": x_d,
                    "logits": dbg["mtp.logits_last"]})
        hd, yd = x_d, d
        if h_of_step is not None and j + 1 < n:
            hd = h_of_step(j + 1, hd)
    return out, rec


def restricted_head(x_d: torch.Tensor, E: torch.Tensor, ids: torch.Tensor):
    """The restricted proposal of S1.2(a): argmax over the k table rows in `ids`, nothing else read.

    Returns (proposal, bytes_read). A gather plus a k-row dot, so the traffic is `k * hidden * 2` B
    measured off the tensor this line materialises, not asserted.
    """
    sub = E[ids]                                    # touches k rows and nothing else
    got = (x_d @ sub.t()).float().reshape(-1)
    return int(ids[int(got.argmax())]), sub.numel() * sub.element_size()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--write", action="store_true", help="write/refresh the golden files")
    ap.add_argument("--check", action="store_true", help="compare a re-run against the golden")
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    d = ROOT / "data" / "refs" / args.fixture
    man = json.loads((d / "manifest.json").read_text())
    n_prefill = len(man["prefill_ids"])
    last = max([int(q.name.split(".")[0][len("decode"):])
                for q in d.glob("decode*.logits_last.npy")] or [-1])
    n_decode = last + 1

    w = Weights(pathlib.Path(args.model))
    E = w.p[EMBED]
    V, Hd = E.shape

    # ---- 1. input contract, from pins, with its witness -------------------------------------
    h = torch.from_numpy(pin_row(d, "final_hidden")).to(torch.bfloat16).reshape(1, Hd)
    prev_row = pin_row(d, f"decode{last}.logits_last")
    y_next = int(prev_row.argmax())
    p = n_prefill + n_decode                  # position of the token being drafted
    pred = (h @ E.t()).float().reshape(-1).numpy().astype(np.float64)
    cos = float(np.dot(pred, prev_row) / (np.linalg.norm(pred) * np.linalg.norm(prev_row)))
    cos = min(cos, 1.0)                       # an F64 dot over 248k terms can read a few 1e-6 > 1
    print(f"fixture {args.fixture}: h = final_hidden ({n_decode} decode steps, prefill {n_prefill})")
    print(f"  witness  E @ h vs decode{last}.logits_last: cos={cos:.9f}")
    if cos < 0.9999:
        raise SystemExit("the pinned hidden does not reproduce the pinned logits row; this fixture "
                         "cannot serve as the drafter's input contract")
    print(f"  y_next = {y_next} (argmax of that row), drafting position p = {p}, "
          f"{NSTEPS} chained steps")

    # ---- 2/3. the chain, its traffic, its goldens -------------------------------------------
    steps, rec = run_chain(w, h, y_next, p)
    read = sorted(rec.counts)     # NOT `sorted(rec)`: rec IS the whole weights dict
    bytes_of = {k: w.p[k].numel() * w.p[k].element_size() for k in read}
    touched = sum(bytes_of.values())
    mtp_bytes = sum(v for k, v in bytes_of.items() if k.startswith(MP))
    emb_bytes = bytes_of.get(EMBED, 0)
    print(f"\nread ledger over {NSTEPS} steps: {len(bytes_of)} distinct tensors, "
          f"{touched} B = {touched/2**20:.3f} MiB ({sum(rec.counts.values())} reads as-run)")
    print(f"  mtp block {mtp_bytes} B = {mtp_bytes/2**20:.4f} MiB   "
          f"tied table {emb_bytes} B = {emb_bytes/2**20:.4f} MiB")
    print(f"  => per-step weight traffic is {touched/2**20:.3f} MiB: {100.0*mtp_bytes/touched:.1f} % "
          f"block, {100.0*emb_bytes/touched:.1f} % the tied head re-read")
    per_key = {k: c for k, c in rec.counts.items()}
    if any(c != NSTEPS for c in per_key.values()):
        print(f"  note: not every tensor is read once per step: "
              f"{ {k: c for k, c in per_key.items() if c != NSTEPS} }")
    wrong = sorted(k for k in bytes_of if not (k.startswith(MP) or k == EMBED))
    if wrong:
        raise SystemExit(f"the drafter read {len(wrong)} tensor(s) outside the MTP block + tied "
                         f"table, e.g. {wrong[:3]}; traffic model and must-not-touch both fail")
    print("  must-not-touch: every tensor read is mtp.* or the tied table")
    for k in read:
        print(f"      {k:<58} {bytes_of[k]/2**20:>9.4f} MiB x{rec.counts[k]}")

    for st in steps:
        print(f"\nstep j={st['j']} at position {st['pos']}: proposal {st['proposal']}")
        if st["j"] + 1 < NSTEPS:
            _rp, rb = restricted_head(st["x_d"], E, torch.from_numpy(np.argsort(-prev_row))[:1024])
            print(f"  (restricted head over the pinned row's top-1024 would read {rb/2**20:.3f} MiB "
                  f"of table instead of {emb_bytes/2**20:.1f} MiB)")

    if args.write:
        GOLD.mkdir(exist_ok=True)
        for st in steps:
            tag = f"{args.fixture}.step{st['j']}"
            save_bf16(GOLD / f"{tag}.u.npy", st["u"])
            save_bf16(GOLD / f"{tag}.x_d.npy", st["x_d"])
            save_bf16(GOLD / f"{tag}.draft_logits.npy", st["logits"])
        (GOLD / f"{args.fixture}.json").write_text(json.dumps({
            "fixture": args.fixture, "h_pin": "final_hidden", "y_next": y_next, "pos": p,
            "n_steps": NSTEPS, "h_source": "post_norm", "fc_order": "eh",
            "proposals": [st["proposal"] for st in steps],
            "positions": [st["pos"] for st in steps],
            "storage": "bf16 bits stored in a uint16 array", "vocab": V, "hidden": Hd,
            "witness_cos": cos, "read_bytes_per_run": touched, "mtp_bytes": mtp_bytes,
            "read_tensors": read, "reads_as_run": per_key,
        }, indent=2))
        print(f"\nwrote golden to {GOLD}/{args.fixture}.*")

    if args.check:
        meta = json.loads((GOLD / f"{args.fixture}.json").read_text()) if \
            (GOLD / f"{args.fixture}.json").exists() else None
        if meta is None:
            raise SystemExit(f"no golden for {args.fixture}; run with --write first")
        for st in steps:
            for name, got in (("u", st["u"]), ("x_d", st["x_d"]), ("draft_logits", st["logits"])):
                path = GOLD / f"{args.fixture}.step{st['j']}.{name}.npy"
                want_t = load_bf16(path)
                same = bool(torch.equal(want_t, got.to(torch.bfloat16).reshape(-1)))
                print(f"  golden step{st['j']}.{name:<13} bit-identical: {same}")
                if not same:
                    raise SystemExit(f"GOLDEN MOVED: step{st['j']}.{name} differs. A reference change "
                                     f"is a re-pin decision, not a fix.")
            if meta["proposals"][st["j"]] != st["proposal"]:
                raise SystemExit(f"GOLDEN MOVED: proposal step{st['j']} "
                                 f"{meta['proposals'][st['j']]} -> {st['proposal']}")
        if meta["read_bytes_per_run"] != touched:
            raise SystemExit(f"read census moved: {meta['read_bytes_per_run']} -> {touched}")
        print("  golden check PASS")

    # ---- 4. controls -------------------------------------------------------------------------
    print("\n=== controls ===")
    pre = torch.from_numpy(pin_row(d, "pre_final_hidden")).to(torch.bfloat16).reshape(1, Hd)

    def x0(r):
        return r[0]["x_d"].to(torch.bfloat16)

    def x1(r):
        return r[1]["x_d"].to(torch.bfloat16) if len(r) > 1 else None

    base, _ = run_chain(w, h, y_next, p)
    must_catch = [
        ("h_source=pre_norm (A-12 as first written)",
         lambda: run_chain(w, pre, y_next, p, h_source="pre_norm"), x0),
        ("fc_order=he (A-13 alternative)",
         lambda: run_chain(w, h, y_next, p, fc_order="he"), x0),
        ("position +1 on the LAST step only (relative distance to the cached draft keys changes)",
         lambda: run_chain(w, h, y_next, p, last_shift=1), x1),
    ]
    for label, mk, pick in must_catch:
        r, _ = mk()
        a, b = pick(base), pick(r)
        moved = not torch.equal(a, b) if b is not None else False
        print(f"  MUST-CATCH   {label:<66} moved={moved}")
        if not moved:
            raise SystemExit(f"negative control did not move anything: {label}. A golden that this "
                             f"knob leaves alone witnesses nothing.")

    # Two facts about this geometry, both derived rather than guessed, and both of which the first
    # version of this script got wrong (it asserted a position shift on a ONE-step case and failed):
    #  * with a ONE-entry MTP cache the attention row is a singleton, and softmax of a singleton is 1.0
    #    whatever the scores are, so no score change - rope included - can move step 0;
    #  * rope (S05.3) makes the scores a function of RELATIVE distance, so shifting every position of
    #    the chain together cannot move anything either.
    r, _ = run_chain(w, h, y_next, p, uniform_shift=1)
    inert_all = all(torch.equal(a["x_d"].to(torch.bfloat16), b["x_d"].to(torch.bfloat16))
                    for a, b in zip(base, r))
    print(f"  MUST-NOT-CATCH shift EVERY position +1 (relative-distance invariance of rope): "
          f"all steps unchanged={inert_all}")
    if not inert_all:
        raise SystemExit("a uniform position shift moved the draft, which contradicts rope's "
                         "relative-distance form")
    r, _ = run_chain(w, h, y_next, p, last_shift=1)
    step0_inert = torch.equal(x0(base), x0(r))
    print(f"  MUST-NOT-CATCH last-step shift judged on STEP 0 (singleton MTP cache: scores cannot "
          f"matter): step 0 unchanged={step0_inert}")
    if not step0_inert:
        raise SystemExit("step 0 moved under a last-step position shift, which contradicts the "
                         "one-lane geometry this gate's shape was chosen for")

    outside = {k: torch.zeros_like(v2) for k, v2 in w.p.items() if k not in bytes_of}
    shuffled = Weights.__new__(Weights)
    shuffled.p = dict(w.p)
    shuffled.p.update(outside)
    r, _ = run_chain(shuffled, h, y_next, p)
    same = torch.equal(x0(base), x0(r)) and torch.equal(x1(base), x1(r))
    print(f"  MUST-NOT-CATCH zero every tensor outside the read set ({len(outside)} tensors): "
          f"both steps' x_d identical={same}")
    if not same:
        raise SystemExit("must-not-catch FAILED: the drafter depends on a tensor the read ledger did "
                         "not list, so the traffic census is incomplete")

    def bump_second_h(j, hd):
        return hd + 0.25 if j == 1 else hd

    r, _ = run_chain(w, h, y_next, p, h_of_step=bump_second_h)
    forward_only = torch.equal(x0(base), x0(r)) and not torch.equal(x1(base), x1(r))
    print(f"  MUST-NOT-CATCH perturb step 2's input hidden: step 1 unchanged and step 2 moved = "
          f"{forward_only}")
    if not forward_only:
        raise SystemExit("the chain is not feeding forward as S09.3 says")
    print("\ncontrols PASS: the golden discriminates the knobs S09.5 settled, rope and the MTP cache "
          "are inside its geometry, and nothing else moves it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
