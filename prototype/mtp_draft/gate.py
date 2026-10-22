#!/usr/bin/env python3
"""Gate for the MTP draft block at the real shapes (§09.2) — the thing `prototype/mtp_headbatch/
MTP-IMPL.md` §4 asks for: "reproduce the §09 golden bit-exactly (or within the published ULP bands)
BEFORE it is allowed to affect acceptance".

WHAT THIS GATE IS NOT: a place that invents tolerances. `prototype/mtp_draft/DRAFT-HANDOVER.md` §2 says
why, and `bands()` imports every constant from the op whose floor probe measured it — `prototype/glue`
(norms, residual adds), `prototype/proj_gemv` (GEMVs), `prototype/attn_decode` (the attention row),
`prototype/mlp_gemv` (F1–F4). The rig owns exactly zero numeric constants.

Two references, neither blended:
  * `mirror(policy="oracle")` calls `speccheck.specimpl.attention`/`mlp` verbatim and must reproduce the
    pinned `prototype/mtp_gatezero/golden` bit-for-bit — that witness (`--selftest`) is the mirror's
    licence to exist, and it is the same class of check as `prototype/attn_decode`'s `reference.attend`
    reproducing the pins at ndiff 0.
  * `mirror(policy="pinned")` stages the model's own BF16 rope/score policy, which is what the shipped
    kernels implement and what the oracle column deliberately does not (`speccheck/specimpl.py:164-181`).
    The device is judged against THIS column, per stage, with the imported bands; the gap between the two
    columns is reported as a number, never absorbed into a wider band.

NEGATIVES are derived, not hand-listed: for each injection the must-catch set is what the pinned mirror
measurably moves and the must-not-catch set is what it does not, and the paired DEVICE runs must move
exactly the first and be byte-identical on the second. A control whose derived set is empty is a
vacuous control (the `partalias` 0-of-1,024 lesson) and aborts here rather than scoring a clean pass.

CPU side: builds the fixture from the pinned dumps and the checkpoint, launches the rig, reads its dumps,
runs the witnesses. The rig prints no timings and this script takes none.

USAGE
  python3 prototype/mtp_draft/gate.py --selftest                 # CPU only, no device
  python3 prototype/mtp_draft/gate.py --fixture dec8 --k 1        # device, one lock hold
  python3 prototype/mtp_draft/gate.py --fixture dec8 --k 2 --negatives
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import td_reference as R                                        # noqa: E402

# (dump name, element count as a function of k, band, class). The classes are the imported rule sets;
# "exact" is a bit-for-bit requirement, licensed because the stage's INPUT is bit-exact by the fixture's
# own provenance (a pinned row, or a copy of a dumped half) — not because anyone wished it so.
def obs_table(k: int):
    """(dump name, TOTAL element count across the k steps, band, why). The rig writes one file per step
    for the step-varying observables and one whole-cache file, so the counts here are totals and the
    reader divides by k — a count that cannot disagree with the dump cannot hide a short write."""
    L = k                                            # cache rows: one [2 heads, 256] row per step
    return [
        ("e",       1024 * k, "exact", "the tied-table row of y_next — a pure copy (§09.1)"),
        ("pair",    2048 * k, "exact", "concat(e', h') — the fc column order AND both pre-FC norms, at bit level"),
        ("u",       1024 * k, "proj",  "the shared fc GEMV, [1024,2048] (§09.2)"),
        ("nrm",     1024 * k, "norm",  "mtp input_layernorm"),
        ("qg",      4096 * k, "proj",  "q_proj: the [4096,1024] head-interleaved [q|gate] GEMV (§05.1)"),
        ("kg",      512 * k,  "proj",  "k_proj"),
        ("vg",      512 * k,  "proj",  "v_proj"),
        ("qrop",    2048 * k, "attn",  "q_norm + rope at this step's absolute position (§05.2, §05.3)"),
        ("krop",    512 * k,  "attn",  "k_norm + rope, the value the cache lane must hold"),
        ("gate",    2048 * k, "proj",  "the gate half of q_proj: the same GEMV output, so the same proj "
                                       "band; the de-interleave itself is checked structurally, below"),
        ("aout",    2048 * k, "attn",  "softmax·V per head before the gate (§05.5)"),
        ("gout",    2048 * k, "attn",  "the §05.6 gating"),
        ("attn_o",  1024 * k, "proj",  "the SECOND o_proj GEMV, [1024,2048] (§09.2)"),
        ("z1",      1024 * k, "norm",  "post-attention residual add (§03.2)"),
        ("u2",      1024 * k, "norm",  "post_attention_layernorm"),
        ("act",     3584 * k, "chain", "silu(gate)*up: fed by two GEMVs whose legal 1-ULP disagreement "
                                       "lands on the product, which is F4's argument verbatim "
                                       "(\"an `act` flip puts many ULP onto an `out` element\")"),
        ("mlp_o",   1024 * k, "chain", "down_proj — `prototype/mlp_gemv/run.py:395,493` judges THIS "
                                       "observable with the F4 aggregates and no ULP gate, and that is "
                                       "the band imported here, not a wider one invented here"),
        ("z2",      1024 * k, "chain", "an add whose addend is only aggregate-legal is only "
                                       "aggregate-legal; a 1-ULP band here would fail on a legal act flip"),
        ("xd",      1024 * k, "chain", "the last norm's output: THE round's output, folded through every "
                                       "upstream rounding (F4: aggregates only)"),
        ("cache_k", L * 512,  "attn",  "the dense MTP cache's K lanes, post-rope (§05.7)"),
        ("cache_v", L * 512,  "proj",  "the dense MTP cache's V lanes"),
    ]


WHOLE_CACHE = ("cache_k", "cache_v")

# The golden writes `x_d`; the mirror and the rig's dumps call it `xd`. One map, so the witness cannot
# quietly compare the wrong tensor.
GOLDEN_TO_MIRROR = {"u": "u", "x_d": "xd"}


def read_set(d: pathlib.Path, table, k: int) -> dict:
    """Read one run's dumps. Per-step files are concatenated in step order, so a missing or short step
    file is an error here rather than a silent half-comparison."""
    out = {}
    for nm, n, _b, _w in table:
        if nm in WHOLE_CACHE:
            # The rig dumps the cache the way its kernels index it — plane-major [kv][row][256], the
            # shipped `kcache[(kv*cap + lane)*DH]` — so it is transposed here to the mirror's
            # [row][kv][256]. Comparing without this would have compared a transposed cache.
            raw = read_dump(d / f"{nm}.bin", n)
            out[nm] = raw.view(2, k, 256).permute(1, 0, 2).reshape(-1)
        else:
            per = n // k
            out[nm] = torch.cat([read_dump(d / f"step{j}.{nm}.bin", per) for j in range(k)])
    return out


def bf16_view(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).view(torch.bfloat16)


def read_dump(path: pathlib.Path, n: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != n:
        raise SystemExit(f"{path.name}: {raw.size} elements, gate expected {n}. "
                         "A dtype or width disagreement, not a numerics result.")
    return bf16_view(raw)


def judge(dev: torch.Tensor, ref: torch.Tensor, rule) -> tuple[bool, str]:
    """Per-element adjudication through the repo's own comparator, with the band that class of stage
    measured. The per-element test is `n_bad == 0` — elements far apart AND large — not "max ULP ≤ 1",
    which `prototype/mlp_gemv/floor_probe.py` shows a legal dot can break at a cancelled element.
    The ndiff limit keeps `NDIFF_ABS_MIN = 1` from the same sources, so a 512-element dump is not failed
    for one element because the percentage happens to be large at that size."""
    if rule == "exact":
        n = int((dev != ref).sum())
        return n == 0, f"exact ndiff={n}/{dev.numel()}"
    rep = R.U.report(dev, ref, ulp_max=rule["ulp"] if rule["ulp"] is not None else 1,
                     cancel_rel=rule["cancel"] if rule["cancel"] is not None else 1e-4)
    bad = []
    if rule["ulp"] is not None and rep["n_bad"]:
        bad.append(f"{rep['n_bad']} elements outside {rule['ulp']} ULP (worst {rep['worst_bad_ulp']} "
                   f"ULP at {rep['worst_bad_rel']:.3g} of max)")
    if rule["ndiff_pct"] is not None:
        lim = max(rule["ndiff_pct"] * rep["n"] / 100.0, 1.0)
        if rep["ndiff"] > lim:
            bad.append(f"ndiff {rep['ndiff']}/{rep['n']} > {lim:.0f}")
    if rep["cos"] < rule["cos"]:
        bad.append(f"cos {rep['cos']:.6f} < {rule['cos']}")
    if rep["meanrel"] > rule["meanrel"]:
        bad.append(f"meanrel {rep['meanrel']:.3g} > {rule['meanrel']}")
    detail = (f"ndiff={rep['ndiff']}/{rep['n']} maxulp={rep['maxulp']} over1={rep['over1']} "
              f"n_bad={rep['n_bad']} in_band={rep['n_band']} worst_off_rel={rep['worst_off_rel']:.2g} "
              f"cos={rep['cos']:.6f} meanrel={rep['meanrel']:.3g}")
    return (not bad), ("OK" if not bad else "BAD " + "; ".join(bad)) + " | " + detail


def device_proposal(xd: torch.Tensor, E: torch.Tensor) -> int:
    """The proposal the draft round committed to, from the device's own `x_d` — exactly the product the
    mirror forms (`x_d @ E.t()` in BF16), so the two are comparable rather than approximately so."""
    return int((xd.unsqueeze(0) @ E.t()).float().argmax())


# ── the work directory ──────────────────────────────────────────────────────────────────────────
def build_work(fixture: str, k: int, out_dir: pathlib.Path) -> dict:
    from safetensors import safe_open
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = R.fixture(fixture)
    wp = {}
    with safe_open(str(R.SHARD), framework="pt") as f:
        for name, shape in R.BLOCK:
            t = f.get_tensor(name)
            got = tuple(t.shape)
            if got != shape:
                raise SystemExit(f"{name}: shard shape {got}, gate claimed {shape}. "
                                 "State the rank; do not guess it (engine/weights_contract.py).")
            if t.dtype != torch.bfloat16:
                raise SystemExit(f"{name}: shard dtype {t.dtype}, this rig stages BF16 only")
            wp[name] = t
        wp[R.EMBED] = f.get_tensor(R.EMBED)
        vocab = wp[R.EMBED].shape[0]
    # The block plan: name, rank, dims, element count. The rig re-checks every claim against what the
    # kernel about to read the blob needs, so a wrong rank here fails at the launch site, not at a read.
    lines, blob = [], b""
    for name, shape in R.BLOCK:
        t = wp[name]
        dims = list(t.shape) + [0] * (3 - len(t.shape))
        lines.append(f"{name} {t.dim()} {dims[0]} {dims[1]} {dims[2]} {t.numel()}")
        blob += t.reshape(-1).view(torch.uint16).numpy().tobytes()
    (out_dir / "block.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "block.bin").write_bytes(blob)

    # Per-step inputs. Where the pinned golden covers a step, its `positions` and `proposals` are used
    # verbatim: step 0's (y_next, pos) and each later step's drafted id then come from the same authority
    # that fixed `h`, so a divergence at step j is a numerics fact and not a different token or position.
    steps = [(fx["y_next"], fx["pos"] + j) for j in range(k)]
    gmeta_path = GOLDEN_FOR = R.GOLDEN / f"{fixture}.json"
    if gmeta_path.exists():
        gm = json.loads(gmeta_path.read_text())
        if k > gm["n_steps"]:
            raise SystemExit(f"{fixture}: the pinned golden runs {gm['n_steps']} steps; asking for "
                             f"{k} would make step {gm['n_steps']}'s input nobody's pin")
        steps = [(gm["proposals"][j - 1] if j else gm["y_next"], gm["positions"][j])
                 for j in range(k)]
    (out_dir / "inputs.txt").write_text(
        f"{vocab}\n" + "".join(f"{y} {p}\n" for y, p in steps))
    (out_dir / "h_post.bin").write_bytes(
        fx["h"].reshape(-1).to(torch.bfloat16).view(torch.uint16).numpy().tobytes())
    (out_dir / "h_pre.bin").write_bytes(
        fx["h_pre"].reshape(-1).to(torch.bfloat16).view(torch.uint16).numpy().tobytes())

    # The tied table is 485 MiB. `prototype/mtp_headbatch/make_inputs.py:table()` already owns the
    # "link it out of the shard without a copy" job and states the rank/dtype it expects, so reuse it and
    # hard-link the result into this work directory rather than writing a second 485 MiB file.
    tbl = out_dir / "W.bin"
    if tbl.exists() and tbl.stat().st_size != vocab * 1024 * 2:
        tbl.unlink()
    if not tbl.exists():
        sys.path.insert(0, str(ROOT / "prototype" / "mtp_headbatch"))
        import make_inputs as MI                                        # noqa: E402
        src = MI.table(False)
        try:
            os.link(src, tbl)
        except OSError:
            shutil.copyfile(src, tbl)
    return {"wp": wp, "fx": fx, "steps": steps, "vocab": vocab}


# ── running the rig ─────────────────────────────────────────────────────────────────────────────
def run_rig(work: pathlib.Path, out: pathlib.Path, extra: list) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(HERE / "mtp_draft"), "--work", str(work), "--out", str(out), "--dump"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"rig exited {r.returncode}: {' '.join(extra)}")
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return {"case": line, "log": r.stdout}
    raise SystemExit("rig printed no RESULT line:\n" + r.stdout)


def witness_expect(work: pathlib.Path) -> tuple:
    """The traffic witness's expected triple, computed HERE in numpy's wrapping uint64 from the same
    bytes and the same segment order the kernel walks. This is deliberately NOT the kernel's own number
    reformatted: `tools/mixed_bw`'s first version checked a read against a value the kernel itself
    produced and passed while reading 1/256 of its bytes. The kernel's job is to reproduce THIS."""
    raw = np.frombuffer((work / "block.bin").read_bytes(), dtype=np.uint64)
    lo = (raw & np.uint64(0xFFFFFFFF)).astype(np.uint64)
    hi = (raw >> np.uint64(32)).astype(np.uint64)
    vals = lo + hi                                        # wraps, exactly as the kernel's u64 does
    idx = (2 * np.arange(raw.size, dtype=np.uint64) + np.uint64(1))
    return int(raw.size), int(vals.sum(dtype=np.uint64)), int((vals * idx).sum(dtype=np.uint64))


# ── negative controls, derived ──────────────────────────────────────────────────────────────────
INJECTIONS = [
    # (name, rig flags, mirror kwargs, which block tensor to scale — None for a knob)
    ("fc-swapped",       ["--fc-order", "he"],  dict(fc_order="he"), None),
    ("h-source-pre",     ["--h-source", "pre"], dict(h_source="pre"), None),
    ("pos-off-by-one",   ["--pos-last", "1"],   dict(pos_last=1), None),
    ("fc-scale",         ["--perturb", "mtp.fc.weight:1.25"],
     dict(perturb={"mtp.fc.weight": 1.25}), "mtp.fc.weight"),
    ("qproj-scale",      ["--perturb", "mtp.layers.0.self_attn.q_proj.weight:1.25"],
     dict(perturb={"mtp.layers.0.self_attn.q_proj.weight": 1.25}),
     "mtp.layers.0.self_attn.q_proj.weight"),
    ("qnorm-scale",      ["--perturb", "mtp.layers.0.self_attn.q_norm.weight:1.25"],
     dict(perturb={"mtp.layers.0.self_attn.q_norm.weight": 1.25}),
     "mtp.layers.0.self_attn.q_norm.weight"),
    ("oproj-scale",      ["--perturb", "mtp.layers.0.self_attn.o_proj.weight:1.25"],
     dict(perturb={"mtp.layers.0.self_attn.o_proj.weight": 1.25}),
     "mtp.layers.0.self_attn.o_proj.weight"),
    ("down-scale",       ["--perturb", "mtp.layers.0.mlp.down_proj.weight:1.25"],
     dict(perturb={"mtp.layers.0.mlp.down_proj.weight": 1.25}), "mtp.layers.0.mlp.down_proj.weight"),
    ("norm-scale",       ["--perturb", "mtp.pre_fc_norm_hidden.weight:1.25"],
     dict(perturb={"mtp.pre_fc_norm_hidden.weight": 1.25}), "mtp.pre_fc_norm_hidden.weight"),
]


def mirror_all(wp, fx, steps, policy, fc_order="eh", h_source="post", pos_shift=0, pos_last=0,
               perturb=None):
    w = dict(wp)
    if perturb:
        for name, mult in perturb.items():
            w[name] = (w[name].float() * mult).to(torch.bfloat16)
    h = fx["h"] if h_source == "post" else fx["h_pre"]
    st = [(y, p + pos_shift + (pos_last if j == len(steps) - 1 else 0))
          for j, (y, p) in enumerate(steps)]
    return R.mirror_chain(w, h, st, policy, fc_order=fc_order, E=fx["E"])


def flatten(steps: list, table, k: int) -> dict:
    """One flat {name: tensor} per observable, in the same order the rig's dumps are read (step j's block
    at the j*per-step offset, the cache as the whole lane block)."""
    out = {}
    for name, n, _band, _why in table:
        if name in WHOLE_CACHE:
            # The cache dump is the whole cache as of the last step — earlier steps are its own prefixes,
            # not separate chunks — so exactly one record carries it, and its size is the check.
            src = steps[-1][name].reshape(-1)
            if src.numel() != n:
                raise SystemExit(f"mirror {name} has {src.numel()}, dump expects {n}")
            out[name] = src.to(torch.bfloat16)
            continue
        per = n // k
        chunks = []
        for j, rec in enumerate(steps):
            src = rec[name].reshape(-1)
            if src.numel() != per:
                raise SystemExit(f"mirror {name} step{j} has {src.numel()}, dump expects {per}")
            chunks.append(src.to(torch.bfloat16))
        out[name] = torch.cat(chunks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--work", default=str(R.WORK))
    ap.add_argument("--negatives", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="CPU only, no device")
    ap.add_argument("--arm", default="chain",
                    choices=["chain", "launch", "witness", "xcheck", "stagecheck"])
    a = ap.parse_args()
    work = pathlib.Path(a.work)

    if a.selftest:
        selftest(a, work)
        return

    table = obs_table(a.k)
    info = build_work(a.fixture, a.k, work)
    wp, fx, steps = info["wp"], info["fx"], info["steps"]
    bands = R.bands()

    # ── witness: the mirror's oracle column must BE the shipped golden ────────────────────────────
    gold = None
    gmeta_path = R.GOLDEN / f"{a.fixture}.json"
    if gmeta_path.exists():
        gold = R.golden(a.fixture)
        if gold["meta"]["n_steps"] < a.k:
            gold = None
    oracle = mirror_all(wp, fx, steps, "oracle")
    if gold:
        for j, rec in enumerate(oracle):
            for nm in ("u", "x_d"):
                bits = gold["steps"][j][nm].reshape(-1)
                nd = int((bits != rec[GOLDEN_TO_MIRROR[nm]].reshape(-1)).sum())
                print(f"WITNESS mirror(oracle) vs golden step{j}.{nm}: ndiff={nd}/{bits.numel()} "
                      + ("BIT-EXACT" if nd == 0 else ""))
                if nd:
                    raise SystemExit("The mirror does not reproduce the shipped MTP golden; nothing "
                                     "downstream of this check can adjudicate the device.")
            gp = gold["meta"]["proposals"][j]
            print(f"WITNESS proposal step{j}: mirror={rec['proposal']} golden={gp}")
            if rec["proposal"] != gp:
                raise SystemExit("proposal mismatch between mirror and golden")

    # ── the device, clean ─────────────────────────────────────────────────────────────────────────
    extra = ["--mode", a.arm, "--k", str(a.k)]
    log = run_rig(work, work / "clean", extra)
    print(log["case"])
    if a.arm == "witness":
        fields = {kv.split("=")[0]: kv.split("=")[1] for kv in log["case"].split() if "=" in kv}
        wn, wraw, waddr = witness_expect(work)
        for nm, want in (("uint2_read", wn), ("raw", wraw), ("addr", waddr)):
            got = int(fields[nm])
            print(f"  witness {nm}: device={got} host={want} " + ("MATCH" if got == want else "MISMATCH"))
            if got != want:
                raise SystemExit(f"traffic witness disagrees on {nm}")
        return
    if a.arm == "launch":
        # The launch-overhead arm prints a duration only under --time-burst, which this gate never passes,
        # so a worker running it sees the census and nothing that could be mistaken for a measurement.
        if "launches=14 timed=0" not in log["case"]:
            raise SystemExit(f"launch arm did not report its census: {log['case']}")
        print(f"  {log['case']}")
        return
    if a.arm == "xcheck":
        fields = {}
        for kv in log["case"].split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                if v.isdigit():
                    fields[k] = int(v)
        for nm in ("steps", "compared", "diff_aout", "diff_gout"):
            if nm not in fields:
                raise SystemExit(f"xcheck printed no {nm}")
        print(f"  xcheck compared {fields['compared']} values over {fields['steps']} rows; "
              f"diff_aout={fields['diff_aout']} diff_gout={fields['diff_gout']} "
              + ("BIT-IDENTICAL" if fields["diff_aout"] + fields["diff_gout"] == 0 else "DIVERGES"))
        if fields["compared"] == 0:
            raise SystemExit("xcheck compared nothing — a comparison that cannot fail")
        if fields["diff_aout"] + fields["diff_gout"]:
            raise SystemExit("the transcribed row is not bit-identical to the shipped attn_core")
        return
    if a.arm != "chain":
        print(f"ARM {a.arm}: see the RESULT lines above; the numerics judgement is the chain arm.")
        return
    dev_dir = work / "clean"
    dev = read_set(dev_dir, table, a.k)

    ref = flatten(mirror_all(wp, fx, steps, "pinned"), table, a.k)
    print(f"\nJUDGEMENT device vs mirror(policy=pinned), fixture={a.fixture} k={a.k}")
    fails = 0
    for nm, n, band, why in table:
        rule = "exact" if band == "exact" else bands[band]
        ok, msg = judge(dev[nm], ref[nm], rule)
        fails += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm:8s} {band:6s} {msg}")
    # and the proposal the round actually committed to
    dp = device_proposal(dev["xd"][(a.k - 1) * 1024:], fx["E"])    # Structural checks with NO reference at all: the §05.1 de-interleave is a copy, so the device's own
    # `gate` must equal the second half of its own `qg` bit for bit, and `qrop` must NOT be a copy of the
    # first half (rope actually ran). Reference-free means there is no band to argue about.
    for j in range(a.k):
        qgv = dev["qg"][j * 4096:(j + 1) * 4096].view(-1, 512)
        copy_ok = torch.equal(dev["gate"][j * 2048:(j + 1) * 2048], qgv[:, 256:].reshape(-1))
        qraw = qgv[:, :256].reshape(-1)
        rope_moved = int((dev["qrop"][j * 2048:(j + 1) * 2048] != qraw).sum())
        print(f"  structural step{j}: gate==qg[:,256:] {'PASS' if copy_ok else 'FAIL'}; "
              f"qrop differs from qraw in {rope_moved}/2048 "
              + ("PASS" if rope_moved else "FAIL (rope did not run)"))
        if not copy_ok or not rope_moved:
            fails += 1

    mp = mirror_all(wp, fx, steps, "pinned")[-1]["proposal"]
    print(f"  proposal: device={dp} mirror(pinned)={mp}" + ("" if dp == mp else "  <-- MISMATCH"))
    if dp != mp:
        fails += 1

    if gold:
        gp = gold["meta"]["proposals"][min(a.k, len(gold["meta"]["proposals"])) - 1]
        print(f"  proposal vs golden step{a.k - 1}: device={dp} golden={gp}")
        if dp != gp:
            fails += 1
            print("  FAIL proposal does not reproduce the pinned golden")

    if fails:
        raise SystemExit(f"{fails} judgement(s) red")

    # ── negatives ─────────────────────────────────────────────────────────────────────────────────
    if a.negatives:
        clean_pinned = ref
        print("\nNEGATIVES (sets derived from the pinned mirror; a derived empty set aborts)")
        for name, flags, kw, tensor in INJECTIONS:
            dirty_mirror = mirror_all(wp, fx, steps, "pinned", **kw)
            dirty_flat = flatten(dirty_mirror, table, a.k)
            must, must_not = [], []
            for nm, n, _b, _w in table:
                moved = bool((dirty_flat[nm] != clean_pinned[nm]).any())
                (must if moved else must_not).append(nm)
            if not must:
                raise SystemExit(f"{name}: derived must-catch set is EMPTY — the control is vacuous. "
                                 "Ask what geometry can make it fire (the partalias lesson).")
            run_rig(work, work / ("neg-" + name), extra + flags)
            dev_dirty = read_set(work / ("neg-" + name), table, a.k)
            caught = [nm for nm in must if bool((dev_dirty[nm] != dev[nm]).any())]
            blind = [nm for nm in must if nm not in caught]
            leaked = [nm for nm in must_not if bool((dev_dirty[nm] != dev[nm]).any())]
            print(f"  {name:15s} must-catch {len(must)} caught {len(caught)} "
                  f"blind {blind or '[]'} must-not {len(must_not)} leaked {leaked or '[]'}")
            if blind or leaked:
                raise SystemExit(f"{name}: device disagrees with the derived control set")


def selftest(a, work: pathlib.Path):
    """CPU only. Three witnesses, each of which can fail and is known to be able to."""
    print("SELFTEST (CPU only, no device)")
    bands = R.bands()
    for k, v in bands.items():
        if v["cos"] is None and v["meanrel"] is None:
            raise SystemExit(f"band set {k} resolved empty")
        print(f"  bands[{k}] ulp={v['ulp']} cancel={v['cancel']} cos>={v['cos']} "
              f"meanrel<={v['meanrel']} src={v['src']}")

    info = build_work(a.fixture, max(1, a.k), work)
    wp, fx, steps = info["wp"], info["fx"], info["steps"]
    gold = R.golden(a.fixture)
    oracle = mirror_all(wp, fx, steps, "oracle")
    for j, rec in enumerate(oracle):
        for nm in ("u", "x_d"):
            bits = gold["steps"][j][nm].reshape(-1)
            nd = int((bits != rec[GOLDEN_TO_MIRROR[nm]].reshape(-1)).sum())
            print(f"  witness mirror(oracle)==golden step{j}.{nm}: ndiff={nd}/{bits.numel()}")
            if nd:
                raise SystemExit("mirror is not the shipped form; it cannot adjudicate anything")
        if rec["proposal"] != gold["meta"]["proposals"][j]:
            raise SystemExit("mirror proposal != golden")
    print("  witness: pinned column differs from the oracle column where the policy says it must:")
    pinned = mirror_all(wp, fx, steps, "pinned")
    for nm in ("u", "qrop", "aout", "xd"):
        same = int((pinned[-1][nm] != oracle[-1][nm]).sum())
        print(f"    {nm:6s} ndiff(pinned,oracle)={same}")
    if int((pinned[-1]["u"] != oracle[-1]["u"]).sum()):
        raise SystemExit("u must not depend on the rope/score policy; something upstream is wrong")

    # The traffic witness's expectation is computed HERE, from the bytes, in numpy's wrapping uint64 —
    # the `tools/mixed_bw` rule that a check whose reference side is the thing under test cannot
    # adjudicate it. The witness arm compares the kernel's triple against these numbers.
    wn, wraw, waddr = witness_expect(work)
    print(f"  witness expectation (host, from block.bin): units={wn} raw={wraw} addr={waddr}")
    if wn == 0 or wraw == 0:
        raise SystemExit("witness expectation is vacuously zero — wrong file?")
    print("SELFTEST PASS")


if __name__ == "__main__":
    main()
