#!/usr/bin/env python3
"""Engine gate: does a greedy token come out right?

    .venv/bin/python engine/gate.py                 # everything; runs engine/build/decode_run
    .venv/bin/python engine/gate.py --no-engine     # fixture + state arithmetic only, no device
    .venv/bin/python engine/gate.py --build         # run engine/build.sh first
    .venv/bin/python engine/gate.py --binary P --model DIR --work DIR

Four sections, in the order the task fixed them plus one that has to come first:

  (0) FIXTURE PAIRING, proved rather than assumed (§10.5). See `pairing_checks`.
  (a) SAMPLED IDS — the engine's sample after the prompt's last token plus its 8 decode samples must
      equal the manifest's `greedy_continuation`, exactly. §10.1 G7 requires token-for-token
      exactness on `dec8`, and a token id is an integer: there is no band to argue about.
  (b) THE LOGITS — each step's logits against that step's pinned `logits_last.npy`, inside `BAND`.
  (c) STATE SIZES — the bytes the engine reports for its request state must equal what
      `engine/state_sizes.py` derives from `state.h`; that file is *run*, not re-typed.
  (d) LAUNCH PLAN — one token's launch bill, counted from §07 in Python and compared with the engine's.

Why this file contains no engine arithmetic. Its value is independence. A Python mirror of the
kernels would check the engine against the engine, which is the failure `bench/gate-audit.md` exists
to find. Everything Python-side here is either a §10 metric, the §00/§08 state derivation, or the
§03.4 argmax used to prove the fixture pairs with itself. Nothing here recomputes a model output.

CPU-only in `--no-engine` mode; the default mode runs the device binary as a subprocess and reads its
dumps. It does not import torch (the reference's own dumps are read with numpy), so it runs under the
same `.venv` the rest of the gates use.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REFS = ROOT / "data" / "refs" / "dec8"
DEFAULT_BINARY = ROOT / "engine" / "build" / "decode_run"
WORK = ROOT / "tmp" / "engine_gate"

# ---- the band ----------------------------------------------------------------------------------
# A token's logits vector is the end of a 24-layer cascade, so §10.1/§10.2 have already decided what
# it is allowed to look like, and this gate uses their published numbers instead of inventing one:
#
#   cos >= 0.9995   §10.1 G5, "incremental == batch": prefill T then decode D, compared with the
#                   reference at the same positions, is held to sampled ids exact + logits
#                   cos >= 0.9995. That is the closest published analogue of what this gate does, so
#                   it is the hard floor here.
#   meanrel <= 5 %  §10.2's cascade ceiling ("cos >= 0.999, meanrel <= 5 %"), the band G2/G3/G4 use
#                   for every whole-cascade tensor comparison.
#   cos >= 0.9999   §10.1 G6, decode logits under online softmax — the tighter figure the reference
#                   itself achieves for a decode-step logits vector. REPORTED, not gated: whether
#                   the engine should be held to it is a policy call, and §10.2 warns that cascade
#                   floors are "a floor, not a knob". See engine/results.md "what this gate does not
#                   claim".
#
#   ndiff / max ULP REPORTED, NOT GATED. §10.1 F1-F4 is explicit about why: on a chained dot the
#   LEGAL ndiff is 29-47 %, and a max-ULP bound is blind to a whole-tensor 1-ULP shift. A threshold
#   on either would be theatre.
#
# §10.2 also fixes the DEFINITIONS: cos is over the flattened tensor, meanrel = mean|Δ| / mean|ref|
# (a ratio of means, not a mean of ratios — the latter explodes on the near-zero logits that most of
# a 248 320-wide vocabulary is).
BAND = {
    "cos_hard": 0.9995,      # §10.1 G5
    "meanrel_hard": 0.05,    # §10.2 cascade ceiling
    "cos_report": 0.9999,    # §10.1 G6, decode logits
}

fails: list[str] = []
notes: list[str] = []


def check(ok: bool, name: str, detail: object = "") -> bool:
    # `detail` is stringified rather than typed as str because several call sites pass a list of ids,
    # which is exactly the thing an operator needs to see when a step fails.
    text = detail if isinstance(detail, str) else repr(detail)
    print(("  [PASS] " if ok else "  [FAIL] ") + name + ((" — " + text) if text else ""))
    if not ok:
        fails.append(name + (": " + text if text else ""))
    return bool(ok)


def section(title: str) -> None:
    print("\n== " + title)


def load_pin(name: str) -> np.ndarray:
    """One pinned dump. numpy has no BF16, so a uint16 payload is widened by the bit shift (§10.5:
    a bitcast, never a numeric cast — `Tensor.to(uint16)`-style truncation is the bug that ate the
    first §04.5 device run)."""
    path = REFS / name
    if not path.exists():
        raise SystemExit(f"gate: pinned dump missing: {path}")
    a = np.load(path, allow_pickle=False)
    if a.dtype == np.uint16:
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a


def argmax_low(a: np.ndarray) -> int:
    """§03.4: greedy argmax, ties to the LOWEST id. numpy's argmax returns the first maximum, which
    is the lowest index; spelled out because the C++ side has to do the same with a strict `>`."""
    return int(np.argmax(a))


def metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    g, r = got.astype(np.float64), ref.astype(np.float64)
    d = g - r
    order = np.argsort(r)[::-1]
    maxabs = float(np.max(np.abs(d)))
    return {
        "cos": float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r))),
        "meanrel": float(np.mean(np.abs(d)) / np.mean(np.abs(r))),   # §10.2's definition
        "ndiff": int(np.count_nonzero(d)),
        "maxabs": maxabs,
        "gap12": float(r[order[0]] - r[order[1]]),
        "margin": float((r[order[0]] - r[order[1]]) / max(maxabs, 1e-30)),
    }


# ---- (0) the fixture pairing -------------------------------------------------------------------
def pairing_checks(man: dict) -> tuple[list, list, np.ndarray]:
    section("(0) fixture pairing — proved from the dumps (§10.5)")
    print(f"  manifest: {REFS / 'manifest.json'}")
    prompt_ids = man["sources"][0]["ids"]
    greedy = man["greedy_continuation"]
    print(f"  sources[0].ids        = {prompt_ids}")
    print(f"  greedy_continuation   = {greedy}")

    dec_logits = [load_pin(f"decode{s}.logits_last.npy").reshape(-1) for s in range(len(greedy) - 1)]
    dec_inputs = [load_pin(f"decode{s}.input_ids.npy").reshape(-1) for s in range(len(greedy) - 1)]
    prefill_logits = load_pin("logits_last.npy").reshape(-1)

    for s, (lg, ip) in enumerate(zip(dec_logits, dec_inputs)):
        meta = man["pins"][f"decode{s}.logits_last"]
        check(meta["step"] == f"decode{s}", f"manifest declares decode{s}.logits_last step",
              f'{meta["step"]} / {meta["shape"]} / {meta["dtype"]}')
        check(len(ip) == 1 and int(ip[0]) == greedy[s],
              f"decode{s}.input_ids == greedy_continuation[{s}]", ip.astype(int).tolist())
        check(argmax_low(lg) == greedy[s + 1],
              f"argmax(decode{s}.logits_last) == greedy_continuation[{s + 1}]",
              f"{argmax_low(lg)} vs {greedy[s + 1]}")
    check(argmax_low(prefill_logits) == greedy[0],
          "argmax(bare logits_last.npy) == greedy_continuation[0] — so the bare name is the PREFILL",
          f"{argmax_low(prefill_logits)} vs {greedy[0]}")
    bare_inputs = load_pin("input_ids.npy").reshape(-1)
    check(int(bare_inputs[0]) == greedy[-2],
          "bare input_ids.npy is decode7's input, NOT the prompt — recorded so it is never used as"
          " one (§10.5's last-writer-wins trap)",
          f"{bare_inputs.astype(int).tolist()} == greedy_continuation[-2] == {greedy[-2]}")
    print(f"  [info] prompt ids come from the manifest's sources[0].ids; the prompt's own logits"
          f" come from the bare logits_last.npy, whose argmax was just proved to be"
          f" greedy_continuation[0]. Pinned logits dtype: {dec_logits[0].dtype},"
          f" shape {dec_logits[0].shape}.")
    return prompt_ids, dec_logits, dec_inputs


# ---- (c) state sizes ---------------------------------------------------------------------------
def state_checks(state_line: dict | None) -> None:
    section("(c) state sizes — against engine/state_sizes.py, which is run rather than retyped")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            from engine import state_sizes as ss
        ok = True
        detail = "imported clean; its own §00.5/spec-page checks passed"
    except SystemExit as e:      # state_sizes.py exits 1 when state.h stops matching §00.5
        ok, detail = False, f"state_sizes.py exited with {e.code}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    check(ok, "engine/state_sizes.py PASS", detail)
    if not ok:
        print(buf.getvalue().rstrip())
        return
    check(ss.gdn_state == 18 * 1024 * 1024, "GDN recurrent state is 18.0 MiB", f"{ss.gdn_state} B")
    check(ss.kv_per_token == 2 * 2 * 256 * 2 * 6,
          "KV is 12288 B/token = 2 caches × 2 kv-heads × 256 × 2 B × 6 layers",
          f"{ss.kv_per_token} B")
    if state_line is None:
        notes.append("(c) ran on the Python side only: --no-engine did not start decode_run.")
        return
    cap = int(state_line["cap"])
    want = {
        "gdn": ss.gdn_state,
        "conv": ss.conv_window,
        "kv": ss.kv_per_token * cap,
        "per_token": ss.kv_per_token,
        "total": ss.gdn_state + ss.conv_window + ss.kv_per_token * cap,
    }
    for k, v in sorted(want.items()):
        check(int(state_line[k]) == v, f"decode_run's STATE {k} matches state_sizes.py",
              f"engine {state_line[k]} vs python {v} (cap {cap})")


# ---- (d) the launch plan -----------------------------------------------------------------------
def launch_plan(n_layers: int, n_gdn: int, n_attn: int) -> dict:
    """One token's launch bill, counted from §07 — independent of the C++'s own arithmetic.

    glue   1 + 2·24. Layer 0's mixer norm has no add in front of it (§03.2's first call), and every
           other add is fused with the norm that follows it: mixer add + post-attention norm, MLP add
           + the next layer's input norm, and after layer 23 the MLP add + the FINAL norm. That last
           pairing is why the final norm is not a launch of its own, and why this is the same 49 that
           `prototype/glue/run.py` pins its token plan against — the number §07's audit in
           `bench/gate-audit.md` demanded.
    proj   GDN: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a (§04.1) + out_proj = 5.
           attention: q_proj, k_proj, v_proj (§05.1) + o_proj = 4.
    gdn    conv (§04.2), gates (§04.3), recurrence (§04.4), gated norm (§04.5) = 4.
    attn   one fused §05.1-§05.7 launch per attention layer (§07.3).
    mlp    gate+up in one launch, down in another (§03.3) = 2 per layer.
    cast   the widenings the F32-input kernels force: GDN 5 per layer (mixer norm row, one packed
           qkv+a+b, gated-norm output, MLP norm row, MLP act) and attention 4 (mixer norm row, the
           gated o_proj operand, MLP norm row, MLP act), plus ONE more for the final norm before
           lm_head. Every one of these is a dtype disagreement between graduated kernels, not model
           math — `kernels_cast.cpp` says why they cannot be dropped today.
    other  the embedding row read (§02.3), one conv-window roll-back per GDN layer (§08.2), lm_head.
    """
    return {
        "glue": 1 + 2 * n_layers,
        "proj": 5 * n_gdn + 4 * n_attn,
        "gdn": 4 * n_gdn,
        "attn": n_attn,
        "mlp": 2 * n_layers,
        "cast": 5 * n_gdn + 4 * n_attn + 1,
        "other": 1 + n_gdn + 1,
    }


def parse_run(text: str) -> dict:
    out = {"steps": [], "state": {}, "launch": {}, "result": {}}
    for line in text.splitlines():
        if line.startswith("STEP "):
            out["steps"].append(dict(re.findall(r"(\w+)=([^\s]+)", line)))
        elif line.startswith("STATE ") or line.startswith("LAUNCH "):
            key = line.split()[0].lower()
            out[key] = dict(re.findall(r"(\w+)=([^\s]+)", line))
        elif line.startswith("RESULT "):
            out["result"] = dict(re.findall(r'(\w+)=(".*?"|[^\s]+)', line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--no-engine", action="store_true",
                    help="fixture pairing + state arithmetic only; starts no binary, touches no device")
    ap.add_argument("--build", action="store_true", help="run engine/build.sh first")
    args = ap.parse_args()

    if args.build:
        subprocess.run([str(ROOT / "engine" / "build.sh")], check=True)

    man = json.loads((REFS / "manifest.json").read_text())
    prompt_ids, dec_logits, dec_inputs = pairing_checks(man)
    greedy = man["greedy_continuation"]
    n_decode = len(dec_logits)

    if args.no_engine:
        state_checks(None)
        return report()

    section("(run) decode_run — one decode_step per token: the prompt, then generation")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
           "--steps", str(n_decode), "--out", str(work)]
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        print("      " + line)
    if proc.returncode != 0:
        for line in proc.stderr.splitlines():
            print("  !! " + line)
        check(False, "decode_run exits 0", f"rc={proc.returncode}")
        return report()
    check(True, "decode_run exits 0")
    run = parse_run(proc.stdout)
    check(bool(run["steps"]), "one STEP line per token", f"{len(run['steps'])} lines")

    # ---- (a) sampled ids -----------------------------------------------------------------------
    section("(a) sampled token ids — §03.4 greedy, §10.1 G7 exactness")
    prompt_steps = [s for s in run["steps"] if s["kind"] == "prompt"]
    decode_steps = [s for s in run["steps"] if s["kind"] == "decode"]
    check(len(prompt_steps) == len(prompt_ids), "one step per prompt token",
          f"{len(prompt_steps)} vs {len(prompt_ids)}")
    check(len(decode_steps) == n_decode, f"{n_decode} generation steps", f"{len(decode_steps)}")
    check(int(prompt_steps[-1]["sampled"]) == greedy[0],
          "the prompt's last token samples greedy_continuation[0]",
          f"{prompt_steps[-1]['sampled']} vs {greedy[0]}")
    got_ids = [int(s["sampled"]) for s in decode_steps]
    for j, (g, w) in enumerate(zip(got_ids, greedy[1:])):
        check(g == w, f"generation step {j} samples greedy_continuation[{j + 1}]", f"{g} vs {w}")
    check(got_ids == greedy[1:], f"all {n_decode} generated ids match, in order",
          f"engine {got_ids} / pinned {greedy[1:]}")
    fed = [int(s["input"]) for s in decode_steps]
    check(fed == greedy[:-1],
          "the tokens the engine fed itself are the ones the reference fed (the engine feeds itself,"
          " so this is the chain being re-derived, not replayed)", f"{fed}")
    positions = [int(s["pos_before"]) for s in decode_steps]
    check(positions == list(range(len(prompt_ids), len(prompt_ids) + n_decode)),
          f"positions run {len(prompt_ids)}..{len(prompt_ids) + n_decode - 1} (§02.4's P)",
          f"{positions}")
    if proc.stdout.count("[STOP]") and greedy[-1] not in man["special_ids"].values():
        notes.append("engine hit the stop token but the manifest's continuation does not")

    # ---- (b) logits ------------------------------------------------------------------------------
    section("(b) logits vs the pinned dumps — band from §10.1 G5 and §10.2")
    print(f"  hard: cos >= {BAND['cos_hard']} (G5 incremental==batch), meanrel <= "
          f"{BAND['meanrel_hard']:.0%} (§10.2 cascade ceiling)")
    print(f"  reported: cos vs {BAND['cos_report']} (G6 decode logits); ndiff and max ULP are "
          "reported but not gated (§10.1 F1–F4: a chained dot's legal ndiff is 29–47 %, and a "
          "max-ULP bound is blind to a whole-tensor 1-ULP shift)")
    for j in range(n_decode):
        path = work / f"logits{len(prompt_ids) + j}.bin"
        if not path.exists():
            check(False, f"engine wrote {path.name}", "file missing — check --out")
            continue
        eng = np.fromfile(path, dtype="<f4")
        ref = dec_logits[j]
        if eng.shape != ref.shape:
            check(False, f"step {j}: vocabulary length", f"{eng.shape} vs {ref.shape}")
            continue
        m = metrics(eng, ref)
        check(True, f"step {j}: metrics",
              f"cos {m['cos']:.6f} meanrel {m['meanrel']:.4%} ndiff {m['ndiff']}/{ref.size}"
              f" maxabs {m['maxabs']:.4g} | ref top1−top2 gap {m['gap12']:.3f}"
              f" = {m['margin']:.0f}× the largest element error")
        check(m["cos"] >= BAND["cos_hard"], f"step {j}: cos >= {BAND['cos_hard']} (§10.1 G5)")
        check(m["meanrel"] <= BAND["meanrel_hard"],
              f"step {j}: meanrel <= {BAND['meanrel_hard']:.0%} (§10.2)")
        check(argmax_low(eng) == argmax_low(ref), f"step {j}: argmax agrees with the pin",
              f"{argmax_low(eng)}")
        if m["cos"] < BAND["cos_report"]:
            notes.append(f"step {j}: cos {m['cos']:.6f} is under G6's decode-logits figure "
                         f"{BAND['cos_report']} (still above the G5 floor that is gated). Whether to"
                         " promote G6's number to a hard floor here is an owner decision, not a"
                         " gate edit — see engine/results.md.")
        else:
            print(f"  [info] step {j}: cos {m['cos']:.6f} clears G6's {BAND['cos_report']}")

    state_checks(run["state"])

    # ---- (d) launch plan ------------------------------------------------------------------------
    section("(d) the launch plan — §07's structure, counted in Python")
    cfg = man["text_config"]
    n_layers = cfg["num_hidden_layers"]
    n_gdn = sum(1 for t in cfg["layer_types"] if t == "linear_attention")
    n_attn = sum(1 for t in cfg["layer_types"] if t == "full_attention")
    want = launch_plan(n_layers, n_gdn, n_attn)
    got = {k: int(v) for k, v in run["launch"].items() if k != "total"}
    for k in sorted(want):
        check(got.get(k) == want[k], f"launch count {k}", f"engine {got.get(k)} vs §07 {want[k]}")
    total = int(run["launch"].get("total", -1))
    check(total == sum(want.values()), "launch total", f"{total} vs {sum(want.values())}")
    check(want["glue"] == 49, "glue is the 49 `prototype/glue/run.py` pins its token plan against",
          f"{want['glue']}")
    check(n_gdn + n_attn == n_layers and n_attn == 6,
          "layer routing read from the manifest matches §03.5", f"{n_gdn}+{n_attn} of {n_layers}")
    if "ms_per_step" in run["result"]:
        notes.append(f"decode_run reported {run['result']['ms_per_step']} ms/step. Timing from a"
                     " gate run is a by-product, not a measurement: see bench/performance-status.md"
                     " §6 for what a clean window takes.")
    return report()


def report() -> int:
    section("summary")
    for n in notes:
        print("  [note] " + n)
    if fails:
        print(f"ENGINE GATE FAIL — {len(fails)} failed")
        for f in fails:
            print("  - " + f)
        return 1
    print("ENGINE GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
