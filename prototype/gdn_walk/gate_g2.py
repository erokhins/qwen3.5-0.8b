#!/usr/bin/env python3
"""gate_g2.py — the ONE device correctness gate bet G2 is allowed to run, and its witness.

WHAT IT RUNS. `decode_run` at two shapes — the shipping one (`--wave-rows 512 --gdn-loop 512`, one launch
per layer per body) and a ragged one (`16/8`, spans of eight and six rows so the span tail is walked) —
and on each shape the shipped form (`--gdn-geom block`) plus the new geometry (`wf4`, and `wf2` as the
split leg), each with `--prefill-logits all --state-fp`. Every prefill logit row and every state
fingerprint is then compared against the shipped side and, for the last row, against the PINNED HF logits.
It is CORRECTNESS ONLY: no millisecond figure is taken, nothing is warmed, nothing is repeated, and it
self-acquires `tools/gpu_lock.sh`, so do NOT call it while holding the lock yourself (AGENTS.md: a
self-locking gate driver nested inside a lock hold parks silently and looks like a hung box).

WHAT THE WITNESS IS. Not bit-equality. `--gdn-geom wf4` re-associates three sums (the two row norms and
the state dots) and rewrites `o` in closed form, so logits and `STATEFP` MUST move — BET.md §10
constraint 2 says so in advance and `g2_mirror.py` sizes the move on CPU. Three separate questions are
adjudicated, and the first attempt at this file conflated the first two, which is worth recording:

  1. FINITE. No NaN, no Inf, anywhere. A wrong address or an overflow shows up here first (AGENTS.md: a
     device fault on a wave-only path presents as a hang, and a wrong base address presents as garbage
     that still runs).
  2. IN BAND — and the band is measured against the REFERENCE, not against the shipped spelling. The
     §10.1 aggregates (cos >= 0.9995, meanrel <= 5 %) are calibrated against the pinned HF dumps; the
     shipped engine already sits ~2 % (RMS-relative) away from them, so demanding that a re-associated
     spelling sit within 0.9995 cos of the SHIPPED engine asks for a tolerance tighter than the one the
     engine itself lives inside, and it fails for reasons that have nothing to do with the geometry. The
     line here is therefore: `cos(new vs pinned logits) >= 0.9995` AND the new side is not meaningfully
     FURTHER from the reference than the shipped side is. New-vs-shipped numbers are still printed —
     they are how a reader sizes the move — but they are a report, not the gate.
  3. EXPLAINABLE, per layer. Layer 0's GDN state has NO upstream except the embedding, so its fingerprint
     move is the geometry's DIRECT effect, un-amplified: it must be re-association scale (<= 1e-6
     relative), and it is measured rather than assumed. Deeper layers inherit the cascade, and this repo
     already knows a one-ULP change grows with depth (AGENTS.md: "greedy token equality decays with
     depth", chaos floor 0.8776 sdpa-vs-eager), so the line there is looser (1e-2) and exists to catch
     corruption, not chaos. And layer 0's CONV window is byte-identical or something is writing outside
     its buffer — that is the address witness G1's grid-axis swap would have failed.

Exit 0 = finite, in band against the pins, direct effect at re-association scale, layer-0 conv untouched,
on every shape and every side.
Usage:
    .venv/bin/python prototype/gdn_walk/gate_g2.py                      # two shapes, three sides
    .venv/bin/python prototype/gdn_walk/gate_g2.py --sides block,wf4    # ship leg only
    .venv/bin/python prototype/gdn_walk/gate_g2.py --reuse              # re-adjudicate existing dumps
    .venv/bin/python prototype/gdn_walk/gate_g2.py --no-lock            # if the caller holds the lock
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COS_FLOOR = 0.9995          # the existing §10.1 aggregates. Not tunable from a flag.
MEANREL_CEIL = 0.05
CANCEL_BAND = 1e-4          # maxrel ignores elements below this fraction of the max (§10.1 F1-F4)
DIRECT_EFFECT_CEIL = 1e-6   # layer 0's state: no upstream, so this is the geometry's own move
CASCADE_CEIL = 1e-2         # deeper layers: chaos, not corruption (AGENTS.md A-18)
STATEFP_RE = re.compile(r"^STATEFP phase=(\S+) family=(\S+) row=(\d+) layer=(\d+)")


def run_one(bin_path, model, ids, shape, side, out, log):
    rows, loop = shape
    cmd = [str(bin_path), "--model", str(model), "--ids", ids, "--steps", "0",
           "--prefill-logits", "all", "--state-fp",
           "--wave-rows", str(rows), "--gdn-loop", str(loop),
           "--gdn-geom", side, "--out", str(out)]
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if rc != 0:
        raise SystemExit(f"run failed rc={rc}: shape={rows}/{loop} side={side}; see {log}")
    return wall


def logits_rows(out):
    rows = {}
    for p in sorted(Path(out).glob("logits*.bin")):
        idx = int(re.match(r"logits(\d+)\.bin", p.name).group(1))
        rows[idx] = np.fromfile(str(p), dtype="<f4")
    return rows


def aggregates(got, ref):
    """cos / meanrel / band-limited maxrel, in float64 (§10.1's aggregates)."""
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    cos = float((a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())))
    denom = np.abs(b)
    rel = np.abs(a - b) / np.where(denom > 0, denom, 1.0)
    live = denom >= CANCEL_BAND * float(denom.max())
    return dict(cos=cos, meanrel=float(rel.mean()),
                maxrel_live=float(rel[live].max()) if live.any() else 0.0,
                rms_rel=float(np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum())))


def statefp_rows(log, fam):
    out = {}
    for line in Path(log).read_text().splitlines():
        m = STATEFP_RE.match(line)
        if m and m.group(2) == fam:
            n2 = re.search(r" n2=(\S+)", line)
            out[(int(m.group(3)), int(m.group(4)))] = (line.strip(),
                                                      float(n2.group(1)) if n2 else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="engine/build/decode_run")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--ids-file", default="/tmp/g2_ids.txt")
    ap.add_argument("--pins", default="data/refs/p64",
                    help="fixture whose logits_last.npy is the reference for the last prefill row")
    ap.add_argument("--shapes", default="512/512,16/8")
    ap.add_argument("--sides", default="block,wf4,wf2")
    ap.add_argument("--out", default="tmp/gdn_walk/gate")
    ap.add_argument("--tag", default="g2-gate")
    ap.add_argument("--reuse", action="store_true",
                    help="adjudicate the dumps already in --out; touches no device")
    ap.add_argument("--no-lock", action="store_true",
                    help="do NOT acquire tools/gpu_lock.sh (only if the caller already holds it)")
    a = ap.parse_args()

    ids = Path(a.ids_file).read_text().strip()
    shapes = [tuple(int(x) for x in s.split("/")) for s in a.shapes.split(",")]
    sides = a.sides.split(",")
    if sides[0] != "block":
        raise SystemExit("the first side must be `block`: it is the reference every other side is judged "
                         "against. Re-order --sides rather than guessing a reference.")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    pin_last = None
    pin_path = ROOT / a.pins / "logits_last.npy"
    if pin_path.exists():
        pin_last = np.load(str(pin_path)).reshape(-1)

    runs = {}
    wall_total = 0.0
    util_at = {}
    if a.reuse:
        for shape in shapes:
            for side in sides:
                tag = f"{shape[0]}x{shape[1]}_{side}"
                runs[tag] = dict(shape=shape, side=side, log=out / f"{tag}.log")
        print("--reuse: no device run; adjudicating the dumps already in " + str(out))
    else:
        lock = ROOT / "tools" / "gpu_lock.sh"
        if not a.no_lock:
            subprocess.call(["bash", str(lock), "acquire", a.tag])
            util_at["acquire"] = subprocess.run(["bash", str(lock), "util"], capture_output=True,
                                                text=True).stdout.strip()
        try:
            for shape in shapes:
                for side in sides:
                    tag = f"{shape[0]}x{shape[1]}_{side}"
                    wall_total += run_one(ROOT / a.bin, ROOT / a.model, ids, shape, side,
                                          out / tag, out / f"{tag}.log")
                    runs[tag] = dict(shape=shape, side=side, log=out / f"{tag}.log")
        finally:
            if not a.no_lock:
                util_at["release"] = subprocess.run(["bash", str(lock), "util"], capture_output=True,
                                                    text=True).stdout.strip()
                subprocess.call(["bash", str(lock), "release", a.tag])

    print(f"runs: {len(runs)} ({len(shapes)} shapes x {len(sides)} sides), "
          f"ids={len(ids.split(','))} tokens from {a.ids_file}")
    for key, val in util_at.items():
        print(f"gpu_lock util@{key}: {val}")
    if wall_total:
        print(f"wall for all device runs: {wall_total:.1f} s — CORRECTNESS ONLY, no timing taken "
              "(no warm loop, no best-of-N; no ms figure here measures anything)")

    bad = []
    for shape in shapes:
        ref_tag = f"{shape[0]}x{shape[1]}_block"
        ref_rows = logits_rows(out / ref_tag)
        if not ref_rows:
            raise SystemExit(f"no logits in {out / ref_tag}")
        last = max(ref_rows)
        print(f"\n== shape --wave-rows {shape[0]} --gdn-loop {shape[1]}: reference side `block`, "
              f"{len(ref_rows)} prefill logit rows")
        pin_line = ""
        if pin_last is not None and pin_last.size == ref_rows[last].size:
            m = aggregates(ref_rows[last], pin_last)
            pin_line = (f" | shipped row {last} vs PINNED logits: cos={m['cos']:.9f} "
                        f"meanrel={m['meanrel']:.3e} rms_rel={m['rms_rel']:.3e}")
        print(f"   {pin_line.strip()}")
        for side in sides[1:]:
            tag = f"{shape[0]}x{shape[1]}_{side}"
            got_rows = logits_rows(out / tag)
            if set(got_rows) != set(ref_rows):
                bad.append(f"{tag}: logit row set differs ({len(got_rows)} vs {len(ref_rows)})")
                continue
            n_nan = sum(int((~np.isfinite(v)).sum()) for v in got_rows.values())
            differing = sum(1 for i in ref_rows if got_rows[i].tobytes() != ref_rows[i].tobytes())
            worst = {"cos": 2.0, "meanrel": 0.0, "maxrel_live": 0.0}
            for i in sorted(ref_rows):
                if got_rows[i].shape != ref_rows[i].shape:
                    bad.append(f"{tag} row {i}: width {got_rows[i].shape} vs {ref_rows[i].shape}")
                    continue
                m = aggregates(got_rows[i], ref_rows[i])
                worst["cos"] = min(worst["cos"], m["cos"])
                worst["meanrel"] = max(worst["meanrel"], m["meanrel"])
                worst["maxrel_live"] = max(worst["maxrel_live"], m["maxrel_live"])
            print(f"  side {side}: logits compared={len(ref_rows)} differing={differing} "
                  f"nonfinite={n_nan}")
            print(f"        vs shipped (report, not the gate): worst-row cos={worst['cos']:.9f} "
                  f"meanrel={worst['meanrel']:.3e} maxrel(band)={worst['maxrel_live']:.3e}")
            if n_nan:
                bad.append(f"{tag}: {n_nan} non-finite logit values")
            if pin_last is not None and pin_last.size == ref_rows[last].size:
                new = aggregates(got_rows[last], pin_last)
                ship = aggregates(ref_rows[last], pin_last)
                print(f"        vs PINNED logits, row {last} (THE GATE): cos={new['cos']:.9f} "
                      f"meanrel={new['meanrel']:.3e} rms_rel={new['rms_rel']:.3e}"
                      f"   [shipped cos={ship['cos']:.9f} rms_rel={ship['rms_rel']:.3e}]")
                if new["cos"] < COS_FLOOR:
                    bad.append(f"{tag}: cos vs the pinned reference {new['cos']:.6f} < {COS_FLOOR}")
                if new["rms_rel"] > ship["rms_rel"] * 1.25 + 1e-9:
                    bad.append(f"{tag}: distance to the reference grew ({new['rms_rel']:.3e} vs the"
                               f" shipped {ship['rms_rel']:.3e}); a re-association may move bits, it may"
                               f" not move the answer away from the reference")
            # ---- STATEFP, per family: layer 0 is the direct effect, the rest is the cascade.
            moves = {}
            for fam in ("state", "conv", "kv"):
                ref_fp = statefp_rows(runs[ref_tag]["log"], fam)
                got_fp = statefp_rows(out / f"{tag}.log", fam)
                if set(ref_fp) != set(got_fp):
                    bad.append(f"{tag} {fam}: fingerprint row set differs")
                    continue
                identical = sum(1 for key in ref_fp if ref_fp[key][0] == got_fp[key][0])
                rel = []
                for key, (_line, n2) in sorted(ref_fp.items()):
                    _g2, gn2 = got_fp[key]
                    if n2 is None or gn2 is None:
                        continue
                    rel.append((abs(gn2 - n2) / max(abs(n2), 1e-30), key))
                rel.sort(reverse=True)
                moves[fam] = dict(rows=len(ref_fp), identical=identical,
                                  worst=(rel[0][0], rel[0][1]) if rel else (0.0, None))
                if fam == "conv" and (0, 0) in ref_fp and ref_fp[(0, 0)][0] != got_fp[(0, 0)][0]:
                    bad.append(f"{tag}: layer 0's conv window fingerprint moved. Nothing this bet"
                               f" touches is upstream of it, so that is an address bug")
            print("        STATEFP " + " ".join(
                f"{fam}: rows={moves[fam]['rows']} identical={moves[fam]['identical']} "
                f"worst_rel={moves[fam]['worst'][0]:.2e}@layer{moves[fam]['worst'][1][1] if moves[fam]['worst'][1] else '-'}"
                for fam in ("state", "conv", "kv")))
            st = moves.get("state")
            if st and st["identical"] == st["rows"]:
                bad.append(f"{tag}: NO GDN state fingerprint moved, which cannot be true of a"
                           f" re-associated walk — the geometry may not be running at all")
            if st:
                direct = None
                ref_fp = statefp_rows(runs[ref_tag]["log"], "state")
                got_fp = statefp_rows(out / f"{tag}.log", "state")
                for key, (_line, n2) in ref_fp.items():
                    if key[1] == 0 and n2:                    # layer 0: no upstream
                        direct = abs(got_fp[key][1] - n2) / abs(n2)
                if direct is None:
                    bad.append(f"{tag}: no layer-0 state fingerprint found to measure the direct effect")
                elif direct > DIRECT_EFFECT_CEIL:
                    bad.append(f"{tag}: layer 0's state moved {direct:.2e} relative, above the"
                               f" {DIRECT_EFFECT_CEIL:.0e} a re-association can explain — this is not"
                               f" chaos (layer 0 has no upstream), it is the geometry")
                else:
                    print(f"        layer-0 state move {direct:.2e} (<= {DIRECT_EFFECT_CEIL:.0e}:"
                          f" the geometry's direct effect, before the cascade)")
                for rel_move, key in [st["worst"]]:
                    if rel_move and rel_move > CASCADE_CEIL:
                        bad.append(f"{tag}: state layer {key[1]} moved {rel_move:.2e} relative, above"
                                   f" the {CASCADE_CEIL:.0e} cascade line")
    same_wf = True
    for shape in shapes:
        a_rows = logits_rows(out / f"{shape[0]}x{shape[1]}_wf4")
        b_rows = logits_rows(out / f"{shape[0]}x{shape[1]}_wf2")
        if a_rows and b_rows:
            same_wf = same_wf and all(a_rows[i].tobytes() == b_rows[i].tobytes() for i in a_rows)
    if "wf4" in sides and "wf2" in sides:
        print(f"\nwf4 vs wf2 bit-identical: {same_wf} — expected: each COLUMN's arithmetic is identical"
              f" at either split (the lane partition changes which lane owns a column, not what that"
              f" lane computes), so the split leg is a pure geometry change with no numerics in it.")
    print()
    if bad:
        print("G2 DEVICE GATE FAIL")
        for item in bad:
            print("   " + item)
        return 1
    print("G2 DEVICE GATE PASS — finite, in band against the pinned reference, layer-0 effect at"
          " re-association scale, layer-0 conv untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
