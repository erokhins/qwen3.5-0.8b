#!/usr/bin/env python3
"""P9 Phase 1 — the chunked GDN delta-rule scan priced against `data/refs/`, CPU only.

Phase 0 (`bench/p9-chunked-scan-2026-10-14.md`) funded this work and set the ceiling the forms here
are judged against; Phase 0's own numbers came from `spike.py`, which this file imports rather than
re-derives (`prototype/gdn_chunk/chunk_scan.py` is the specification of the form — nothing here
re-derives its algebra). What is NEW here, and the reason this file exists next to `spike.py`:

  1. **The budget is judged, not reported.** Phase 0 §5 proposed the numbers; this file is a gate:
     per-stage `meanrel` <= `STAGE_MEANREL_MAX`, per-logits-row `1-cos` <= `ROW_1COS_MAX`, and zero
     argmax moves at any pinned row with >= `FLIP_ROOM_MIN` of top-1/top-2 room. Each line prints
     PASS/FAIL and the process exits 1 on any FAIL. The room rule itself is imported from
     `engine/gate.py` (`flip_admissible`, `CASCADE_BAND`), not retyped.
  2. **The streaming rows exist now.** `wide640f`/`wide1024f` pin EVERY prefill row, so rows above
     `MAXSC` — the rows that just killed the attention tile form (`prototype/attn_decode/results.md`
     §19, room 13.25) — are adjudicable for a GDN recurrence change for the first time. `rows`
     judges them.
  3. **The body hand-off is proved bit-identical, not close.** The engine prefills in `wave_rows`
     bodies, so a chunked scan split at a multiple of C must be *bit-identical* to one segment.
     `split` asserts that, with a non-aligned cut as its negative control.
  4. **Negatives localise.** `negatives` injects three bugs of the shape Phase 1 will plausibly
     contain (a stale carry, an off-by-one in the intra-chunk decay product, a chunk boundary applied
     to the wrong head) and asserts each one is silent EXACTLY where the geometry says it cannot be
     seen and loud everywhere else. Must-catch/must-not-catch sets are DERIVED from (T, C, H) and the
     operand the bug touches — `AGENTS.md`'s rule that a hand-listed set rots into "everything".

CPU only, per `AGENTS.md`: this file never touches the iGPU and takes no timing. The cost model in
`cost` is arithmetic on rates already measured in `bench/` and `devices/`.

    .venv/bin/python prototype/gdn_chunk/phase1.py stage     --fixtures p64,p300
    .venv/bin/python prototype/gdn_chunk/phase1.py split     --fixtures p64,p300
    .venv/bin/python prototype/gdn_chunk/phase1.py negatives --fixture p300
    .venv/bin/python prototype/gdn_chunk/phase1.py side      --fixture wide1024f --form chunk64mma
    .venv/bin/python prototype/gdn_chunk/phase1.py rows      --fixture wide1024f
    .venv/bin/python prototype/gdn_chunk/phase1.py cost      --C 64
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "gdn_chunk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chunk_scan as cs                                                # noqa: E402
import spike                                                           # noqa: E402
from speccheck import specimpl                                         # noqa: E402


def _load_module(name: str, path: Path):
    """Import a tool file that is not a package member (gate.py is a script)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GATE = _load_module("_p1_gate", ROOT / "engine" / "gate.py")

# ---- THE BUDGET — Phase 0's ceilings, quoted, and the D-11 band they sit inside --------------------
# Phase 0 §5 proposed these from measurements already on disk (`bench/p9-chunked-scan-2026-10-14.md`
# §2.1-§2.3). They are a CEILING, not a target: a form that lands under them is inside its budget, a
# form that lands over them stops and reports the number. No band is adopted here and nothing in this
# file is a band — `engine/gate.py` and `tools/band_history.py` are untouched and stay green.
STAGE_MEANREL_MAX = 9.9e-3      # Phase 0 §2.1 worst BF16-operand stage meanrel (vs pins)
STAGE_COS_MIN = 0.9999          # §2.1 worst BF16-operand stage cos, rounded up to a line
ROW_1COS_MAX = 3.2e-4           # Phase 0 §2.3 worst per-logits-row 1-cos
FLIP_ROOM_MIN = GATE.CASCADE_BAND["flip_room_max"]   # 0.10, imported from the file that owns it
D11 = dict(stage_meanrel=GATE.CASCADE_BAND["stage_meanrel_max"],
           row_1cos=GATE.CASCADE_BAND["row_1cos_max"],
           stage_cos=GATE.CASCADE_BAND["stage_cos_min"],
           room=FLIP_ROOM_MIN)
BODY_ROWS = 512                 # engine wave_rows body the device shape is priced at
CS_LIST = spike.CS_LIST         # (16, 32, 64)
# The forms a Phase-1 device kernel could be. `mma` (BF16 operands, F32 accumulator) is the shape the
# matrix units actually run, so it is the form the budget must hold; F32 is kept as the algebra floor.
DEVICE_FORMS = [f"chunk{c}mma" for c in CS_LIST]
FLOOR_FORM = "seq"

SIDES_DIR_DEFAULT = "tmp/p91/sides"


# ---------------------------- shared helpers ----------------------------
def fixture_len(fixture: str) -> int:
    return int(np.load(ROOT / f"data/refs/{fixture}/L00.gdn.o_pre_norm.npy", mmap_mode="r").shape[1])


def row_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """cos and §10.2 meanrel of one logits row pair, float64 accumulators (gate.py's `metrics` shape
    without the argsort — this runs over 1024 rows of 248 320 and only needs the pair's aggregates."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    return dict(cos=(float(x @ y) / (nx * ny)) if nx and ny else 1.0,
                meanrel=float(np.abs(x - y).mean() / np.abs(y).mean()))


def bf16_rows_to_disk(path: Path, rows: torch.Tensor) -> None:
    """Rows as BF16 bits. `.view(torch.uint16)` BITCASTS (`Tensor.to(uint16)` casts numerically — the
    trap in `prototype/gdn_gated_norm/results.md` §8, and the reason a fixture writer gets it wrong)."""
    np.save(path, rows.to(torch.bfloat16).view(torch.uint16).numpy())


def bf16_rows_from_disk(path: Path):
    return np.load(path, mmap_mode="r")


def widen_bits(a: np.ndarray) -> np.ndarray:
    return GATE.widen_pin(np.asarray(a)).astype(np.float32)


# ---------------------------- stage ----------------------------
def cmd_stage(args) -> int:
    """Per-stage aggregates for each form against the PINS, judged against the budget."""
    torch.set_grad_enabled(False)
    w = specimpl.Weights(Path(args.model))
    forms = [FLOOR_FORM] + [f for f in args.forms.split(",") if f]
    fails: list[str] = []
    for fixture in args.fixtures.split(","):
        layers = spike.GDN_LAYERS if args.layers == "all" else \
            [int(x) for x in args.layers.split(",")]
        T = fixture_len(fixture)
        print(f"\n=== stage: {fixture} T={T}, layers {layers[0]}..{layers[-1]}"
              f" ({len(layers)} of 18 GDN layers), forms {forms}")
        print(f"    budget: per-stage meanrel <= {STAGE_MEANREL_MAX:.1e}, cos >= {STAGE_COS_MIN}"
              f"   (D-11 for comparison: {D11['stage_meanrel']:.0e} / {D11['stage_cos']})")
        pins = {l: spike.load_gdn_pins(fixture, l) for l in layers}
        for tag in forms:
            specimpl.gdn_chunk_scan = spike._ORIG_CHUNK      # a stale patch would fake the floor
            if tag != FLOOR_FORM:
                specimpl.gdn_chunk_scan = _patched_scan(spike.parse_form(tag))
            worst = {}
            for l in layers:
                o, S = spike.scan(pins[l], **spike.parse_form(tag))
                gated, outp = spike.downstream(w, l, o, pins[l]["z"])
                got = dict(o_pre_norm=o, state_new=S, gated_out=gated, out_proj_out=outp)
                for st, val in got.items():
                    s = spike.stats(val, pins[l][st])
                    if st not in worst or s["meanrel"] > worst[st][0]:
                        worst[st] = (s["meanrel"], s["cos"], l)
            for st in spike.STAGES:
                rel, cos, l = worst[st]
                ok = rel <= STAGE_MEANREL_MAX and cos >= STAGE_COS_MIN
                fails += [] if ok else [f"{fixture} {tag} {st}: meanrel {rel:.3e} cos {cos:.8f}"]
                print(f"    [{'PASS' if ok else 'FAIL'}] {tag:11s} {st:13s} meanrel {rel:.3e}"
                      f"  cos {cos:.8f}   (worst layer {l:02d})")
    print(f"\nstage: {'PASS' if not fails else 'FAIL'}"
          + ("".join("\n  " + f for f in fails) if fails else
             f" — every stage of every form is inside the Phase-0 budget"))
    return 1 if fails else 0


# ---------------------------- split (the body hand-off) ----------------------------
def cmd_split(args) -> int:
    """The wave-body claim, proved in bits: a chunked scan cut at a multiple of C is bit-identical.

    The engine prefills in `wave_rows` bodies and resumes requests mid-sequence, so the chunked
    kernel will be handed a state and must hand one back. If a split at an aligned cut changed even
    one bit, then the engine's body schedule would be part of the numerics and no per-body gate
    could ever be read as a whole-prompt claim. The negative control (a cut that is NOT a multiple of
    C) is what makes the PASS a claim rather than a tautology: it must NOT be bit-identical.
    """
    torch.set_grad_enabled(False)
    fails, lines = [], []
    for fixture in args.fixtures.split(","):
        T = fixture_len(fixture)
        for l in (spike.GDN_LAYERS[:args.layers_n] if args.layers_n else [spike.GDN_LAYERS[0]]):
            pins = spike.load_gdn_pins(fixture, l)
            for C in CS_LIST:
                # Aligned cuts: one mid-fixture multiple of C, plus the engine's own body boundary
                # (512 rows) when the fixture is long enough to have one. Control: the aligned cut
                # plus one token, which no chunk boundary can land on.
                cuts = {max(1, (T // C) // 2) * C}
                if T > BODY_ROWS:
                    cuts.add((T // BODY_ROWS) * BODY_ROWS)
                for cut in sorted(cuts):
                    if not 0 < cut < T:
                        continue
                    for c, aligned in ((cut, True), (cut + 1, False)):
                        if c >= T:
                            continue
                        o_all, S_all = spike.scan(pins, "chunk", C=C)
                        seg1 = spike.load_gdn_pins(fixture, l, slice(0, c))
                        seg2 = spike.load_gdn_pins(fixture, l, slice(c, T))
                        o1, S1 = spike.scan(seg1, "chunk", C=C)
                        o2, S2 = spike.scan(seg2, "chunk", C=C, S0=S1)
                        same_o = torch.equal(torch.cat([o1, o2], 0), o_all)
                        same_s = torch.equal(S2, S_all)
                        ok = (same_o and same_s) if aligned else not (same_o and same_s)
                        tag = ("engine body boundary" if c == BODY_ROWS and aligned
                               else "aligned" if aligned else "NON-aligned control")
                        lines.append(f"    [{'PASS' if ok else 'FAIL'}] {fixture} L{l:02d} C={C:2d}"
                                     f" cut={c:4d} ({tag}):"
                                     f" o bit-identical {int(same_o)}  S bit-identical {int(same_s)}")
                        if not ok:
                            fails.append(f"{fixture} L{l:02d} C={C} cut={c} aligned={aligned}")
    print("\n=== split: a chunked scan cut at a multiple of C is BIT-IDENTICAL to one segment"
          f" (and a cut that is not must differ, or this check cannot fail)")
    print("\n".join(lines))
    print(f"split: {'PASS' if not fails else 'FAIL'}"
          + ("" if not fails else " — " + "; ".join(fails)))
    return 1 if fails else 0


# ---------------------------- side (one form's prefill rows) ----------------------------
def _patched_scan(pf: dict, bug: str = ""):
    """The engine-side form, substituted for HF's chunk scan inside specimpl's cascade. Same trick
    `spike.run_side` uses: `specimpl.gdn_chunk_scan` is the only seam, so every other stage of the
    cascade is the code speccheck already gates. `pf` is `spike.parse_form(tag)` minus the form tag."""
    C, mma = pf["C"], pf["mma"]

    def scan(qn, kn, v, g, beta, S=None, _C=C):
        return cs.chunked_scan(qn, kn, v, g, beta, S, C=_C, mma=mma, bug=bug)
    return scan


def cmd_side(args) -> int:
    """Run one arithmetic form's PREFILL and dump every prompt row's logits for `rows` to judge."""
    torch.set_grad_enabled(False)
    w = specimpl.Weights(Path(args.model))
    man = json.loads((ROOT / "data" / "refs" / args.fixture / "manifest.json").read_text())
    ids = man["prefill_ids"]
    form = args.form
    specimpl.gdn_chunk_scan = spike._ORIG_CHUNK
    fwd_form = "recurrent"
    if form != FLOOR_FORM:
        pf = spike.parse_form(form)
        specimpl.gdn_chunk_scan = _patched_scan(pf)
        fwd_form = "chunk"
    f = specimpl.Forward(w, gdn_form=fwd_form)
    rows = f.logits(f.prefill(torch.tensor(ids)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{args.fixture}.{form}.rows.npy"
    bf16_rows_to_disk(dst, rows)
    (out / f"{args.fixture}.{form}.json").write_text(json.dumps(
        dict(fixture=args.fixture, form=form, gdn_form=fwd_form, T=len(ids), V=rows.shape[-1],
             prompt_head=ids[:4], pin_rows=man.get("prefill_logit_rows"),
             torch=str(torch.__version__)), indent=1) + "\n")
    print(f"side: {args.fixture} form={form} gdn_form={fwd_form} T={len(ids)} -> {dst}"
          f" ({dst.stat().st_size / (1024 ** 2):.1f} MiB, BF16 bits)")
    return 0


# ---------------------------- rows (the decision-level budget) ----------------------------
def cmd_rows(args) -> int:
    """Every pinned prefill row: chunked vs sequential (the route delta) and vs the PIN (HF decides).

    Three quantities, and only two of them can be blamed on the scan:
      * `1-cos(chunk, seq)` — the budget line Phase 0 measured (`ROW_1COS_MAX`), same instrument,
        now including the rows above MAXSC that Phase 0 had no pin for.
      * argmax moves vs the PIN, adjudicated by `gate.flip_admissible` (room = the smaller side's
        top1-top2 gap). The CPU mirror's own cascade error also moves rows against HF, so the
        SEQUENTIAL side's move count is printed beside the chunked one as the floor: a move the
        shipped form also makes is not this change's.
      * the adjudicated move (Phase 0 §5 clause 2): the chunked side disagrees with HF where the
        sequential side agrees. That is the only class of move that can count against the change.
    """
    torch.set_grad_enabled(False)
    sides = Path(args.sides)
    pin_path = ROOT / "data" / "refs" / args.fixture / "logits_rows.npy"
    man = json.loads((ROOT / "data" / "refs" / args.fixture / "manifest.json").read_text())
    if not pin_path.exists():
        raise SystemExit(f"rows: {args.fixture} pins no row block ({pin_path}); "
                         "only wide640f/wide1024f carry `logits_rows.npy`")
    rows_pin = np.load(pin_path, mmap_mode="r")
    n_rows, vocab = rows_pin.shape
    seq_path = sides / f"{args.fixture}.{FLOOR_FORM}.rows.npy"
    if not seq_path.exists():
        raise SystemExit(f"rows: the floor side is missing ({seq_path}). Run `side --form "
                         f"{FLOOR_FORM}` first — without the shipped form on the same anchor every"
                         " number here is unfloored.")
    seq = bf16_rows_from_disk(seq_path)
    if seq.shape != (n_rows, vocab):
        raise SystemExit(f"rows: floor side is {seq.shape}, pins are {(n_rows, vocab)}"
                         " — a short side is a crashed run, not a band difference")
    chunk_forms = [f for f in args.forms.split(",") if f]
    fails: list[str] = []
    print(f"\n=== rows: {args.fixture} — {n_rows} pinned prefill rows x {vocab} vocab"
          f" (storage {man.get('logits_storage')}, MAXSC {GATE.lane_geometry()[0]})")
    print(f"    budget: per-row 1-cos(chunk,seq) <= {ROW_1COS_MAX:.1e};"
          f" zero moves at room >= {FLIP_ROOM_MIN:.2f} (room rule: gate.flip_admissible)")
    print("    a row's lanes are row+1, so rows above MAXSC are the streaming form §07.3 bands —"
          " the rows that killed the attention tile form at room 13.25")
    for form in chunk_forms:
        cpath = sides / f"{args.fixture}.{form}.rows.npy"
        if not cpath.exists():
            raise SystemExit(f"rows: side {form} missing ({cpath}); run `side --form {form}`")
        ch = bf16_rows_from_disk(cpath)
        if ch.shape != (n_rows, vocab):
            raise SystemExit(f"rows: side {form} is {ch.shape}, pins are {(n_rows, vocab)}")
        staging = GATE.lane_geometry()[0]
        c_min, c_mean, c_max, worst_row = float("inf"), 0.0, 0.0, -1
        region_worst = {"one-pass": (0.0, -1), "streaming": (0.0, -1)}
        n_over, moves_pin, moves_seq, adjudicated, thin = 0, [], [], [], 0
        seq_moves_pin = 0
        big_room_moves = []
        for i in range(n_rows):
            pin = widen_bits(rows_pin[i]).reshape(-1)
            srow = widen_bits(seq[i]).reshape(-1)
            crow = widen_bits(ch[i]).reshape(-1)
            m = row_metrics(crow, srow)
            one = 1.0 - m["cos"]
            c_min = min(c_min, one)
            c_mean += one
            if one > c_max:
                c_max, worst_row = one, i
            region = "one-pass" if i + 1 <= staging else "streaming"
            if one > region_worst[region][0]:
                region_worst[region] = (one, i)
            if one > ROW_1COS_MAX:
                n_over += 1
            pin_id, _, pin_room = GATE.top2(pin)
            seq_id, _, seq_room = GATE.top2(srow)
            ch_id, _, ch_room = GATE.top2(crow)
            if seq_id != pin_id:
                seq_moves_pin += 1
            if ch_id != pin_id:
                room = min(pin_room, ch_room)
                adm = GATE.flip_admissible(pin_room, ch_room)
                moves_pin.append((i, region, pin_id, ch_id, room, adm, seq_id == pin_id))
                if room >= FLIP_ROOM_MIN:
                    big_room_moves.append((i, region, pin_id, ch_id, room))
                if not adm:
                    fails.append(f"{form} row {i}: argmax moved against the pin at room {room:.4g}")
            if ch_id != seq_id:
                moves_seq.append((i, region, seq_id, ch_id, min(seq_room, ch_room)))
            if ch_id != pin_id and seq_id == pin_id:
                adjudicated.append((i, region, pin_id, ch_id, min(pin_room, ch_room)))
            if pin_room < FLIP_ROOM_MIN:
                thin += 1
        c_mean /= n_rows
        print(f"\n  --- {form} vs {FLOOR_FORM} (the route delta) and vs the pin (HF decides)")
        ok_cos = n_over == 0 and c_max <= ROW_1COS_MAX
        if not ok_cos:
            fails.append(f"{form}: per-row 1-cos max {c_max:.3e} > {ROW_1COS_MAX:.1e}"
                         f" ({n_over} rows over)")
        print(f"    [{'PASS' if ok_cos else 'FAIL'}] per-row 1-cos vs {FLOOR_FORM}: min {c_min:.2e}"
              f" mean {c_mean:.2e} max {c_max:.2e} at row {worst_row}"
              f" — {n_over} of {n_rows} rows over the {ROW_1COS_MAX:.1e} ceiling")
        print("        worst row by region (the streaming rows are §19's): "
              + "; ".join(f"{r} row {i} 1-cos {v:.2e}" for r, (v, i) in region_worst.items()
                          if i >= 0))
        print(f"        argmax moved vs {FLOOR_FORM} on {len(moves_seq)} of {n_rows} rows"
              + (": " + "; ".join(f"row {i} ({r}) {a}->{b} room {rm:.4g}"
                                  for i, r, a, b, rm in moves_seq[:12]) if moves_seq else ""))
        print(f"        argmax moved vs the PIN: chunked {len(moves_pin)} rows,"
              f" {FLOOR_FORM} floor {seq_moves_pin} (a move the shipped form also makes is not this"
              f" change's); {len(adjudicated)} adjudicated against this change"
              f" (the {FLOOR_FORM} side agrees with HF there)")
        if moves_pin:
            print("        every move against the pin, with its room and whether the room rule admits it:")
            for i, region, p, c, room, adm, seq_ok in moves_pin:
                print(f"          row {i:5d} ({region:9s}) pin {p} -> {c}  room {room:.4g}"
                      f"  {'admitted (thin room)' if adm else 'NOT ADMITTED'}"
                      f"  {FLOOR_FORM} side {'agrees' if seq_ok else 'also disagrees'} with HF")
        ok_room = not big_room_moves
        if not ok_room:
            fails.append(f"{form}: {len(big_room_moves)} argmax moves at room >= {FLIP_ROOM_MIN}")
        print(f"    [{'PASS' if ok_room else 'FAIL'}] zero argmax moves at room >="
              f" {FLIP_ROOM_MIN:.2f}: {len(big_room_moves)} such moves"
              + (" — " + "; ".join(f"row {i} ({r}) room {rm:.4g}"
                                   for i, r, _, _, rm in big_room_moves) if big_room_moves else ""))
        print(f"        adjudicated moves (chunked disagrees with HF where {FLOOR_FORM} agrees):"
              f" {len(adjudicated)}" + ("" if not adjudicated else
              " — " + "; ".join(f"row {i} room {rm:.4g}" for i, _, _, _, rm in adjudicated[:12])))
        if adjudicated and args.adjudicated_counts:
            fails.append(f"{form}: {len(adjudicated)} adjudicated moves")
        print(f"        the pin's own fragile rows: {thin} of {n_rows} have room under"
              f" {FLIP_ROOM_MIN:.2f} (census: tools/prefill_row_rooms.py)")
    print(f"\nrows: {'PASS' if not fails else 'FAIL'}"
          + ("".join("\n  " + f for f in fails) if fails else
             f" — {chunk_forms} inside the Phase-0 row budget on all {n_rows} pinned rows"))
    return 1 if fails else 0


# ---------------------------- onerow (the decode clause) ----------------------------
def load_gdn_step(fixture: str, layer: int, step: int | None) -> dict:
    """One pinned GDN operand set — `step=None` is the prefill block, `step=7` the seventh decode
    step, which is a ONE-row body and therefore the decode case.

    `spike.load_gdn_pins` reads only the prefill block (`L00.gdn.*`); the decode steps are the same
    names behind a `decode{s}.` prefix, and the wide fixtures carry ONLY those, which is what makes a
    one-row body checkable against the reference at all.
    """
    d = ROOT / "data" / "refs" / fixture
    pre = f"{('decode%d.' % step) if step is not None else ''}L{layer:02d}.gdn."
    out = {}
    for key in ("q_post_conv", "k_post_conv", "v_post_conv", "beta", "g_decay_log",
                "o_pre_norm", "state_new"):
        t = torch.from_numpy(np.load(d / f"{pre}{key}.npy")).float().squeeze(0)
        if step is None and key != "state_new":
            # The prefill block's row 0 IS a one-row body: token 0 from the zero state, and the
            # recurrence cannot see further than its own row.
            t = t[:1]
        out[key] = t
    if step is None:
        # The block's `state_new` is the state after the WHOLE prompt, which a one-row body has no
        # claim on. Row 0's `o` is still exactly comparable, so that is what this clause checks.
        out["state_new"] = None
    return out


def gdn_step_choices(fixture: str) -> list:
    """Which GDN operand sets this fixture actually pins: the prefill block, or decode steps."""
    d = ROOT / "data" / "refs" / fixture
    if (d / "L00.gdn.o_pre_norm.npy").exists():
        return [None]
    steps = sorted({int(p.name.split(".")[0][len("decode"):])
                    for p in d.glob("decode*.L00.gdn.o_pre_norm.npy")})
    return steps or [None]


def cmd_onerow(args) -> int:
    """A one-row body has no scan to chunk — measured, not asserted.

    Two claims, and they need different witnesses:
      * the STATE update of a one-token chunk degenerates to §04.4's exactly (the UT system is a
        1x1 unit matrix, so `U = I` and the solve is the identity), so `S` comes out BIT-IDENTICAL;
      * the READ does not: the chunked form computes `o = Gamma (q.S) + (q.k) dv`, which is §04.4's
        SPLIT read, and §04.4 names the split form as the one its own note measured wrong ("the two
        split terms cancel"). So `o` is not bit-identical — measured here at meanrel ~2e-7 per layer,
        which is small but is not nothing, and a decode step is exactly where the engine promises
        bit-identity.
    Therefore decode bit-identity is a property of the DISPATCH RULE (a one-row body must still run
    `rec_kernel`), not of the algebra, and the rule is load-bearing. The number printed here is the
    size of the damage if the rule is broken, which is what the device decode A/B is checking against
    zero. `vs pin` is only printed where the request starts at that row (the prefill block's row 0, a
    one-row body from the zero state); a mid-sequence decode step's carried-in state is not pinned,
    so scoring its operands from a zero state would print a number about the fixture, not the form.
    """
    torch.set_grad_enabled(False)
    worst, fails = 0.0, []
    print("\n=== onerow: a decode step (T=1) chunked anyway — the price of breaking the dispatch rule")
    print("    claim: with `--gdn-scan chunk`, a one-row body still runs rec_kernel, and that is what")
    print("    makes decode bit-identical. Below: S is bit-identical anyway; o is NOT.")
    for fixture in args.fixtures.split(","):
        for step in gdn_step_choices(fixture)[:args.max_steps]:
            for l in spike.GDN_LAYERS[:args.layers_n]:
                pins = load_gdn_step(fixture, l, step)
                o_ref, S_ref = spike.scan(pins, FLOOR_FORM)
                tag = (f"{fixture} L{l:02d} "
                       + ("prefill row 0 (zero state)" if step is None
                          else f"decode step {step} (carried state)"))
                bits, parts = [], []
                for C in CS_LIST:
                    o, S = spike.scan(pins, "chunk", C=C)
                    d = spike.stats(o, o_ref)
                    worst = max(worst, d["meanrel"])
                    bits.append(float((o - o_ref).abs().max()))
                    parts.append(f"C={C}: S bit-identical {int(torch.equal(S, S_ref))}"
                                 f" o bit-identical {int(torch.equal(o, o_ref))}"
                                 f" | chunk-vs-seq meanrel {d['meanrel']:.2e}"
                                 f" cos {d['cos']:.8f}")
                pin = ("" if step is not None else
                       f"; vs pin o meanrel {spike.stats(o_ref, pins['o_pre_norm'])['meanrel']:.2e}"
                       f" (both forms, indistinguishably)")
                print(f"    {tag}: " + " | ".join(parts) + pin)
                if all(b == 0.0 for b in bits):
                    fails.append(f"{tag}: chunked one-row body was bit-identical at every C,"
                                 " so this check has stopped distinguishing the forms")
    print(f"onerow: {'PASS' if not fails else 'FAIL'} — worst chunk-vs-seq o meanrel on a one-row"
          f" body {worst:.2e}. Decode bit-identity therefore rests on the kernel dispatching"
          " `tokens == 1` to the sequential path, which is what the device decode A/B must show as"
          " byte-identical output.")
    return 1 if fails else 0


# ---------------------------- negatives ----------------------------
# Each entry: the bug, the operand it corrupts, and what the GEOMETRY says about where the damage can
# reach. The reach is expressed as (first reachable chunk, first reachable row-in-chunk) and derived
# by the code below from C/NC/H and that description, never hand-listed per row.
#   carry_stale_state      chunk i reads the state as of chunk i-2. Chunk 0's read is the request's
#                          own incoming state, so the bug cannot be seen in chunk 0 at all.
#   decay_off_by_one       the pairwise intra-chunk decay M_ij uses gamma_{i-1} instead of gamma_i.
#                          Row 0 of a chunk has only the diagonal j=0 below the line, and M_00 is
#                          exp(gamma_0 - gamma_0) = 1 either way, so chunk 0 row 0 is untouched; the
#                          chunk's own writes then carry the error into every later row and chunk.
#   boundary_wrong_head    the chunk-total decay applied to S_in is taken from the neighbouring head.
#                          It multiplies S_in ONLY, so it is invisible wherever S_in is zero (chunk 0
#                          of a request) and still invisible one chunk further (chunk 1 reads the
#                          state chunk 0 wrote, which the decay never touched): first damage at chunk 2.
NEGATIVES = [
    ("carry_stale_state", "the inter-chunk carry", (1, 0)),
    ("decay_off_by_one", "the intra-chunk decay product M_ij", (0, 1)),
    ("boundary_wrong_head", "the chunk-boundary decay of S_in", (2, 0)),
]
# The head-blind sibling of the wrong-head bug: same operand, no head axis involved, so the two must
# produce different per-head profiles or the profile is not evidence about heads.
HEAD_AXIS_CONTROL = {"boundary_wrong_head": "no_chunk_decay"}


def cmd_negatives(args) -> int:
    """Inject each bug, then check it is silent EXACTLY where the geometry says it is unreachable."""
    torch.set_grad_enabled(False)
    C = args.C
    T = fixture_len(args.fixture)
    NC = -(-T // C)
    fails: list[str] = []
    print(f"\n=== negatives: {args.fixture} L{spike.GDN_LAYERS[0]:02d}, T={T}, C={C}"
          f" -> {NC} chunks; H={cs.H_V}; threshold is 100x the legal route delta (Phase 0 §1.4)")
    pins = spike.load_gdn_pins(args.fixture, spike.GDN_LAYERS[0])
    good_o, good_S = spike.scan(pins, f"chunk{C}")
    seq_o, seq_S = spike.scan(pins, FLOOR_FORM)
    # The legal delta is the route change itself, so the negatives are calibrated by measurement.
    legal_o = float((good_o - seq_o).abs().max())
    legal_s = float((good_S - seq_S).abs().max())
    thr = 100.0 * max(legal_o, 1e-30)
    print(f"    legal chunk-vs-sequential max|delta o| {legal_o:.3e}, state {legal_s:.3e}"
          f" -> a bug must move a reachable cell by > {thr:.3e}")
    print("    contract, per bug: (A) the unreachable cells are EXACTLY zero bits of movement,"
          " (B) every chunk it can reach moves above the threshold, (C) the head axis is actually"
          " read (a wrong-head boundary must not look like a missing one), (D) a budget line trips.")
    profiles: dict[str, torch.Tensor] = {}
    for name, what, (first_chunk, first_row) in NEGATIVES:
        o, S = spike.scan(pins, f"chunk{C}", bug=name)
        d = (o - good_o).abs()
        finite = bool(torch.isfinite(o).all() and torch.isfinite(S).all())
        if NC * C > T:                       # the ragged tail's pad rows: no data, no movement
            d = torch.cat([d, d.new_zeros((NC * C - T,) + tuple(d.shape[1:]))], 0)
        dc = d.reshape(NC, C, cs.H_V, cs.D_V)
        per_cell = dc.amax(dim=(2, 3))                     # [NC, C]
        per_head = dc.amax(dim=(0, 1, 3))                  # [H]
        profiles[name] = per_head
        # (A) DERIVED unreachable set: everything before the bug's first reachable cell, in
        # chunk-major order. Exact zeros, not small numbers — the arithmetic path is untouched there.
        unreachable = [(c, r) for c in range(NC) for r in range(C)
                       if (c, r) < (first_chunk, first_row)]
        silent_ok = all(float(per_cell[c, r]) == 0.0 for c, r in unreachable)
        # (B) every reachable chunk moves. Per CHUNK, not per cell: the geometry licenses reach, it
        # does not promise magnitude — a head whose gate decays everything legitimately shows
        # nothing, and demanding per-cell movement would be a threshold by taste.
        loud_ok = all(float(per_cell[c, first_row if c == first_chunk else 0:].max()) > thr
                      for c in range(first_chunk, NC))
        st_o, st_s = spike.stats(o, pins["o_pre_norm"]), spike.stats(S, pins["state_new"])
        vs_good = spike.stats(o, good_o)
        budget_caught = st_o["meanrel"] > STAGE_MEANREL_MAX or st_s["meanrel"] > STAGE_MEANREL_MAX
        ok = finite and silent_ok and loud_ok and budget_caught
        note = ""
        if name in HEAD_AXIS_CONTROL:
            # (C) the head axis is read: the wrong-head profile must differ from the profile of the
            # head-BLIND sibling (`no_chunk_decay`, the same carry with decay 1 everywhere). If a
            # gate cannot tell those apart it is not looking at heads at all.
            sib = spike.scan(pins, f"chunk{C}", bug=HEAD_AXIS_CONTROL[name])[0]
            ds = (sib - good_o).abs()
            if NC * C > T:
                ds = torch.cat([ds, ds.new_zeros((NC * C - T,) + tuple(ds.shape[1:]))], 0)
            profiles[HEAD_AXIS_CONTROL[name]] = ds.reshape(NC, C, cs.H_V, cs.D_V).amax(dim=(0, 1, 3))
            diff = float((per_head - profiles[HEAD_AXIS_CONTROL[name]]).abs().max())
            ok = ok and diff > thr
            note = (f"   head axis: |profile − head-blind sibling's profile| {diff:.1e}"
                    f" > {thr:.1e}: {int(diff > thr)}")
        fails += [] if ok else [f"{name}"]
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:20s} ({what})")
        print("          per-chunk max|delta o|: "
              + " ".join(f"c{c}:{float(dc[c].max()):.1e}" for c in range(NC)))
        print(f"          (A) silent on all {len(unreachable)} unreachable cells, exact zeros"
              f" (first reachable (chunk {first_chunk}, row {first_row})): {int(silent_ok)}"
              f"   (B) every reachable chunk loud: {int(loud_ok)}"
              f"   (D) meanrel ceiling {STAGE_MEANREL_MAX:.0e} tripped: {int(budget_caught)}"
              + ("" if finite else "   NON-FINITE (caught outright)") + note)
        print(f"          vs pins after the bug: o meanrel {st_o['meanrel']:.2e}, state meanrel"
              f" {st_s['meanrel']:.2e}; per-head delta range"
              f" {float(per_head.min()):.1e}..{float(per_head.max()):.1e}"
              + (f"; delta vs correct chunked meanrel {vs_good['meanrel']:.2e}"
                 if args.show_delta else ""))
    print(f"\nnegatives: {'PASS' if not fails else 'FAIL'}"
          + ("" if not fails else " — " + ", ".join(fails)))
    return 1 if fails else 0


# ---------------------------- cost ----------------------------
# Rates measured on this box, with their source. Nothing here is estimated: 0.271 TF/s is today's
# gdn_rec line read back as a rate, 0.85 TF/s is the engine's own worst hand GEMM, and 2.93 TF/s is
# the measured F32 FMA ceiling of the vector path (`devices/strix-halo-compute.md` correction 2).
RATES = [(0.271, "today's gdn_rec line as a rate (125.4 ms over 33.94 GFLOP, measured wave)"),
         (0.85, "the engine's own hand GEMMs, measured 0.8-0.9 TFLOP/s"),
         (2.93, "measured F32 FMA ceiling of the vector path (a hand kernel's own ceiling)"),
         (4.80, "hipBLAS 24-layer wave mix at M=64, measured"),
         (18.84, "hipBLAS 24-layer wave mix at M=512, measured")]
MEASURED_GDN_MS, MEASURED_LAUNCHES, LAUNCH_US = 125.4, 18432, 1.8
DISPATCH_FLOOR_MS = MEASURED_LAUNCHES * LAUNCH_US * 1e-3


def cmd_cost(args) -> int:
    """What the HIP chunked kernel costs at C, against the line it replaces. Arithmetic only."""
    D, H, L = cs.D_K, cs.H_V, len(spike.GDN_LAYERS)
    WAVE, BODY = args.tokens, args.rows
    BODIES = WAVE // BODY
    seq = D * D + 2 * D * D + 3 * D + 2 * D * D + 2 * D * D
    C = args.C
    intra = (2 * C * C * D + 2 * C * C * D + 4 * C * C * D + 2 * C * C * D + 2 * C ** 3 // 3)
    inter = 6 * C * D * D + D * D
    per = (intra + inter) // C
    wave = per * H * L * WAVE
    seq_wave = seq * H * L * WAVE
    steps = -(-BODY // C)
    print(f"\n=== cost: C={C}, {BODY}-row bodies x {BODIES} = {WAVE}-token wave,"
          f" H={H}, d_k=d_v={D}, {L} GDN layers")
    print(f"    line being replaced: gdn_rec {MEASURED_GDN_MS} ms, {MEASURED_LAUNCHES} launches,"
          f" {MEASURED_GDN_MS / MEASURED_LAUNCHES * 1e3:.1f} us each, {seq} flops/token-head-layer")
    print(f"    chunked: intra {intra} + inter {inter} per chunk-head = {per}"
          f" flops/token-head-layer ({100.0 * (per / seq - 1):+.1f} % vs sequential)")
    print(f"    {wave / 1e9:.2f} GFLOP/wave (sequential {seq_wave / 1e9:.2f});"
          f" serial depth {steps} chunk-steps per body instead of {BODY} token-steps")
    print(f"    state traffic per wave: {2 * D * D * 4 * H * L * WAVE / C / 1e6:.0f} MB"
          f" (sequential {2 * D * D * 4 * H * L * WAVE / 1e9:.2f} GB, which exceeds the 240 GB/s"
          " DRAM floor — the sequential line is latency/L2-shaped, not traffic-shaped)")
    for sites, why in ((2, "prep + carry, one launch each per (layer, body)"),
                       (4, "FLA-shaped: 4 sites per (layer, body)")):
        n = sites * L * BODIES
        print(f"    launches: {n} ({why}) replacing {MEASURED_LAUNCHES}"
              f" — dispatch {n * LAUNCH_US * 1e-3:.2f} ms vs {DISPATCH_FLOOR_MS:.1f} ms")
    print("    %-10s %-14s %-9s %s" % ("rate TF/s", "ms/wave", "speedup", "what that rate is"))
    for rate, note in RATES:
        ms = wave / rate / 1e12 * 1e3 + 2 * L * BODIES * LAUNCH_US * 1e-3
        print("    %-10s %-14.1f %-9.2f %s" % (rate, ms, MEASURED_GDN_MS / ms, note))
    for want in (1.0, 1.5, 2.0):
        need = wave / ((MEASURED_GDN_MS / want) * 1e-3) / 1e12
        print(f"    needs {need:.3f} TF/s for {want:.1f}x = {100 * need / 0.85:.0f} % of the hand"
              f" GEMM rate, {100 * need / 2.93:.0f} % of the measured F32 FMA ceiling")
    print("    PRE-REGISTERED WIN LINE (this file, before any measurement): the chunked GDN scan"
          f" line must come in <= {MEASURED_GDN_MS / 1.99:.0f} ms on this wave (1.99x) and is KILLED"
          f" above {MEASURED_GDN_MS / 1.5:.0f} ms (Phase 0's 1.5x kill threshold).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stage", help="per-stage aggregates vs the pins, judged against the budget")
    p.add_argument("--fixtures", default="p64,p300")
    p.add_argument("--forms", default=",".join(DEVICE_FORMS))
    p.add_argument("--layers", default="all", help="'all' or a comma list of GDN layer ids")
    p.set_defaults(fn=cmd_stage)

    p = sub.add_parser("split", help="the wave-body hand-off, proved bit-identical")
    p.add_argument("--fixtures", default="p64,p300")
    p.add_argument("--layers-n", type=int, default=1)
    p.set_defaults(fn=cmd_split)

    p = sub.add_parser("side", help="run one form's prefill and dump every prompt row")
    p.add_argument("--fixture", required=True)
    p.add_argument("--form", required=True, help=f"{FLOOR_FORM} or chunk16/32/64 with optional mma")
    p.add_argument("--out", default=SIDES_DIR_DEFAULT)
    p.set_defaults(fn=cmd_side)

    p = sub.add_parser("rows", help="judge the dumped rows against the budget")
    p.add_argument("--fixture", required=True)
    p.add_argument("--forms", default=",".join(DEVICE_FORMS))
    p.add_argument("--sides", default=SIDES_DIR_DEFAULT)
    p.add_argument("--adjudicated-counts", action="store_true",
                   help="count adjudicated moves (chunked wrong where sequential is right) as FAIL")
    p.set_defaults(fn=cmd_rows)

    p = sub.add_parser("onerow", help="the one-row-body (decode) clause, measured against the pins")
    p.add_argument("--fixtures", default="p300,wide1024f")
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--layers-n", type=int, default=2)
    p.set_defaults(fn=cmd_onerow)

    p = sub.add_parser("negatives", help="localising injected bugs")
    p.add_argument("--fixture", default="p300")
    p.add_argument("--C", type=int, default=64)
    p.add_argument("--show-delta", action="store_true")
    p.set_defaults(fn=cmd_negatives)

    p = sub.add_parser("cost", help="the arithmetic cost model at the engine's shapes")
    p.add_argument("--C", type=int, default=64)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--rows", type=int, default=BODY_ROWS)
    p.set_defaults(fn=cmd_cost)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
