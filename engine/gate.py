#!/usr/bin/env python3
"""Engine gate: does a greedy token come out right?

    .venv/bin/python engine/gate.py                 # everything; runs engine/build/decode_run
    .venv/bin/python engine/gate.py --no-engine     # fixture pairing + state sizes, zero device
    .venv/bin/python engine/gate.py --selftest      # proves THIS gate discriminates: no engine, no device
    .venv/bin/python engine/gate.py --dry-run       # every section against a CPU stand-in
    .venv/bin/python engine/gate.py --build         # run engine/build.sh first
    .venv/bin/python engine/gate.py --binary P --model DIR --work DIR

`--selftest` re-runs this file in subprocesses against four injected bugs — a wrong sampled id, a
swapped top-two logits, 5 % logits noise that leaves the argmax alone, and a §07.5 stop check firing
during the prompt phase — and requires each to fail at the step it was injected at, because a negative
that every check would catch proves nothing (`prototype/mlp_gemv/results.md`). It is the only part of
a gate that can be validated before the engine has ever run, and it is why a first device run need not
double as a test of this file.

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

# §07.5 / A-10: the id greedy decoding stops on — the end-of-turn token. Spelled as an id, never as
# template text, and it is the same constant as `engine::kStopToken`. The `dec8` prompt is a rendered
# chat template, so this id occurs INSIDE the prompt (index 8) as well as in a continuation.
STOP_ID = 248046

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
#   cos >= 0.9999   §10.1 G6, decode logits under online softmax. REPORTED, not gated — DECIDED
#                   2026-10-11, do not promote it; the reasoning is below, not in a results file,
#                   because a band comment is the thing a future reader edits.
#
#   ndiff / max ULP REPORTED, NOT GATED. §10.1 F1-F4 is explicit about why: on a chained dot the
#   LEGAL ndiff is 29-47 %, and a max-ULP bound is blind to a whole-tensor 1-ULP shift. A threshold
#   on either would be theatre.
#
# Why G6's 0.9999 is advisory here and G5's 0.9995 is the floor — and why the three §10 cos figures
# are not interchangeable thresholds for the same quantity:
#
#   * G3i (cos >= 0.9999) measures an ISOLATED single layer fed the reference's own `s_in`, and it is
#     the tightest figure precisely because "no upstream cascade can enter" (§10.1). It is the right
#     bar for one kernel and the wrong bar for a token.
#   * G6's 0.9999 is a decode-step logits vector in the REFERENCE, i.e. the whole 24-layer chain
#     already upstream of it, with the softmax form as the only variable. It is a bf16-cascade figure,
#     not a per-op one.
#   * G5's 0.9995 is the incremental-vs-batch invariant: same model, same cascade, two ways of getting
#     to position T+D.
#   The rule, as decided: an end-to-end quantity must not be judged by a per-layer or per-operation
#   bar, because the error it is allowed to carry is the accumulated error of all 24 layers and the
#     whole token chain, not the error of one stage. G3i/G5/G6 therefore compare different things —
#     one stage, two routes to one position, and one form of one stage — and a threshold moved between
#     them is a different measurement, not a stricter or looser one.
#   What actually decides correctness on this gate is **token identity (G7, exact ids)**; the logits
#     band is supporting evidence about *how* the engine got there, which is why a step whose cos sits
#     between the two figures is reported (the G6 note below) and not failed. The first end-to-end
#     pass had exactly that profile: step 1 at cos 0.999886 — under G6, over the G5 floor, with every
#     one of the eight ids exact and the reference's own top-1/top-2 margin 7-23x the largest element
#     error at every step. It stays a documented observation. Two CPU probes price it, both with no
#     engine and no device: `engine/logits_floor.py` is the LAST-OP floor (legal reassociation of the
#     reference's own logits dot, its answer 0.99999862), and `engine/route_price.py` is the
#     END-TO-END floor (the same reference run through two legal GDN prompt forms; its spread is
#     1-cos 4.7e-5..1.9e-4 and it is wider than the engine's deviation at every step). The engine is
#     below the first and inside the second, which is why the flag is informational: a per-stage floor
#     applied to an end-to-end quantity will always fire, and the quantity that could really be wrong
#     is tokens, which is G7. engine/results.md records both tables and the reasoning.
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
    gap = float(r[order[0]] - r[order[1]])
    return {
        "cos": float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r))),
        "meanrel": float(np.mean(np.abs(d)) / np.mean(np.abs(r))),   # §10.2's definition
        "ndiff": int(np.count_nonzero(d)),
        "maxabs": maxabs,
        "gap12": gap,
        # How many times bigger the reference's own top-1/top-2 margin is than our worst element
        # error — i.e. how much further the error would have had to grow to flip the token. Guarded:
        # a bit-exact step has maxabs 0, and dividing by an epsilon there prints a ludicrous number.
        "margin": (float("inf") if maxabs == 0.0 else gap / maxabs),
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
    stop_at = [i for i, t in enumerate(prompt_ids) if t == STOP_ID]
    print(f"  [info] the prompt itself contains the stop id {STOP_ID} at index {stop_at}, so a"
          " prompt-phase step can legitimately sample it. §07.5 is a generation-phase rule and such"
          " a match must be ignored (section (a) asserts both halves of that).")
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


def dry_run(man: dict, prompt_ids: list, dec_logits: list, work: Path, break_ids: int = -1,
            break_logits: int = -1, break_noise: int = -1, break_prompt_stop: int = -1) -> tuple[
                str, int]:
    """A stand-in for `decode_run`, on the CPU, that replays the pins as if the engine were perfect.

    Why this is in a gate: sections (a)-(d) parse a line protocol, read binary dumps, and do float
    maths that has never once run against a real logits file. Discovering a typo there costs a
    quiescent-GPU window, which is the scarcest thing in this repo (`AGENTS.md`). So the plumbing gets
    exercised here, on the CPU, against input built from the same pins the checks will compare with —
    which is exactly why it cannot be read as a result: it passes by construction. Anything it fails
    is a bug in this file, not in the engine. (`--selftest` is what proves it can also FAIL.)

    `break_ids`/`break_logits` inject one deliberate error at that generation step, the way
    `prototype/*/run.py --negative` does: the injected bug has to make the gate fail *at that step*,
    which is the property worth testing — a negative that every check would catch proves nothing
    (`prototype/mlp_gemv/results.md` has that trap written down). `break_prompt_stop` injects the one
    bug the engine actually shipped with: the §07.5 stop check firing during the prompt, which truncates
    the run there and leaves both the rest of the prompt and all generation steps unsaid.
    """
    greedy = man["greedy_continuation"]
    cfg = man["text_config"]
    lines = []
    cap = len(prompt_ids) + len(dec_logits) + 1
    plan = launch_plan(cfg["num_hidden_layers"],
                       sum(1 for t in cfg["layer_types"] if t == "linear_attention"),
                       sum(1 for t in cfg["layer_types"] if t == "full_attention"))
    lines.append(f"STATE cap={cap} gdn={ss_sizes()[0]} conv={ss_sizes()[1]}"
                 f" kv={ss_sizes()[2] * cap} total={ss_sizes()[0] + ss_sizes()[1] + ss_sizes()[2] * cap}"
                 f" per_token={ss_sizes()[2]}")
    lines.append("LAUNCH " + " ".join(f"{k}={v}" for k, v in plan.items())
                 + f" total={sum(plan.values())}")
    for i, t in enumerate(prompt_ids):
        # Only the LAST prompt step's sample is checked by section (a) — that is the one the fixture
        # pins. The earlier ones are the engine's own business; here they echo the next prompt token.
        sampled = greedy[0] if i == len(prompt_ids) - 1 else (prompt_ids[i + 1] if i + 1 < len(prompt_ids) else greedy[0])
        # `stop=` is the §07.5 CHECK being armed and matched — never the raw match. A prompt-phase step
        # reports stop=0 even if it sampled the stop id, because its sample is discarded; that is what
        # decode_run prints and what section (a) asserts. `--break-prompt-stop` is the one case where
        # it is 1, and it reproduces the first device run: the stop match ends the run AT the prompt,
        # so the remaining prompt tokens and all 8 generation steps are simply missing.
        if i == break_prompt_stop:
            # The stop check fires BECAUSE the sample is the stop id, so the stand-in has to say both;
            # `decode_run` prints `stop=1` only under that condition.
            lines.append(f"STEP kind=prompt index={i} input={t} sampled={STOP_ID} top_logit=1"
                         f" pos_before={i} pos_after={i + 1} stop=1")
            lines.append(f"RESULT mode=run steps={cap - 1} ms_total=0 ms_per_step=0"
                         f" last_sampled={STOP_ID} prefix_len={i + 1} stopped=1")
            return "\n".join(lines), 0
        lines.append(f"STEP kind=prompt index={i} input={t} sampled={sampled} top_logit=1"
                     f" pos_before={i} pos_after={i + 1} stop=0")
    for j in range(len(dec_logits)):
        pos = len(prompt_ids) + j
        sampled = greedy[j + 1]
        logits = dec_logits[j]
        if j == break_ids:
            sampled = greedy[j + 1] + 1        # one wrong id, exactly where a real divergence starts
        if j == break_logits:
            # Flip the argmax: raise the runner-up past the top. This is the smallest logits-level
            # failure that matters — a token chosen wrongly, not a vector that is merely noisy.
            logits = logits.copy()
            order = np.argsort(logits)[::-1]
            logits[order[1]] = logits[order[0]] + 1.0
        if j == break_noise:
            # Aggregate corruption that LEAVES the argmax alone: 5 % relative noise puts cos at
            # ≈ 1 − σ²/2 ≈ 0.9987, under G5's 0.9995 and over §10.2's 0.999 cascade floor. This is
            # the case that asks whether the band has teeth of its own, separate from the token id.
            rng = np.random.default_rng(4)   # fixed: the injection must be reproducible
            logits = logits * (1.0 + 0.05 * rng.standard_normal(logits.shape)).astype(np.float32)
        lines.append(f"STEP kind=decode index={pos} input={greedy[j]} sampled={sampled}"
                     f" top_logit=1 pos_before={pos} pos_after={pos + 1} stop=0")
        (work / f"logits{pos}.bin").write_bytes(logits.astype(np.float32).tobytes())
    lines.append("RESULT mode=run steps=25 ms_total=0 ms_per_step=0 last_sampled="
                 f"{greedy[-1]} prefix_len={cap - 1} stopped=0")
    return "\n".join(lines), 0


def ss_sizes() -> list:
    """The three constant state figures, from engine/state_sizes.py (same source section (c) uses)."""
    with contextlib.redirect_stdout(io.StringIO()):
        from engine import state_sizes as ss
    return [ss.gdn_state, ss.conv_window, ss.kv_per_token]


def selftest() -> int:
    """Prove the gate discriminates, on the CPU, before anyone spends a GPU window on it.

    Four runs: the clean dry run must pass; each injected bug must fail, and fail *at its own step*.
    A negative that any check would catch is worth nothing (`prototype/mlp_gemv/results.md`), so the
    assertion is about where the failure landed, not that one happened.
    """
    base = [sys.executable, str(Path(__file__).resolve())]
    cases = [
        ([], 0, None, "clean dry run"),
        (["--break-ids", "3"], 1, "[FAIL] generation step 3 samples greedy_continuation[4]",
         "wrong id injected at step 3"),
        (["--break-logits", "2"], 1, "[FAIL] step 2: argmax agrees with the pin",
         "top two logits swapped at step 2"),
        (["--break-noise", "5"], 1, "[FAIL] step 5: cos >= 0.9995 (§10.1 G5)",
         "5 % noise on step 5's logits (argmax untouched, aggregate wrong)"),
        # The bug the engine actually shipped with, replayed in the stand-in: id 248046 sampled at
        # prompt index 6 ended a 17-token prompt after 7 steps. Both new checks have to name it.
        (["--break-prompt-stop", "6"], 1,
         ["[FAIL] no prompt-phase step fires §07.5's stop check",
          "[FAIL] one step per prompt token",
          "that last step's §07.5 stop check HAD fired"],
         "the stop check fired at prompt index 6 (the first device run's bug: a 17-token prompt"
         " ended after 7 steps and generation never started)"),
    ]
    bad = []
    for extra, want_rc, want_in, what in cases:
        p = subprocess.run(base + ["--dry-run"] + extra, capture_output=True, text=True)
        needles = [] if want_in is None else ([want_in] if isinstance(want_in, str) else list(want_in))
        missing = [n for n in needles if n not in p.stdout]
        ok = p.returncode == want_rc and not missing
        where = "" if not needles else (" — every expected FAIL line present" if not missing
                                        else " — missing: " + "; ".join(missing))
        print(f"  [{'PASS' if ok else 'FAIL'}] {what} — rc={p.returncode} want {want_rc}{where}")
        if not ok:
            print("      " + "\n      ".join(l for l in p.stdout.splitlines() if "FAIL" in l)[:400])
            bad.append(what)
    print("SELFTEST " + ("FAIL — " + "; ".join(bad) if bad else
                        "PASS (the gate passes clean input and catches each injected bug at its"
                        " injected step; this says nothing about the engine)"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--no-engine", action="store_true",
                    help="fixture pairing + state arithmetic only; starts no binary, touches no device")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every section except the engine itself, against a CPU stand-in that"
                         " replays the pins; exercises this file's parsing and metrics. NOT a result.")
    ap.add_argument("--break-ids", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: make that generation step sample a wrong id, and require"
                         " the gate to catch it there")
    ap.add_argument("--break-logits", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: swap that step's top two logits, and require the gate to"
                         " catch it there")
    ap.add_argument("--break-noise", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: put 5 % relative noise on that step's logits, which moves"
                         " cos under the floor without changing the argmax")
    ap.add_argument("--break-prompt-stop", type=int, default=-1, metavar="PROMPT_INDEX",
                    help="with --dry-run: fire the §07.5 stop check at that prompt step and end the"
                         " run there, which is the bug the first device run had")
    ap.add_argument("--selftest", action="store_true",
                    help="re-run this file in subprocesses: clean dry run must pass, each injected"
                         " bug must fail at its own step")
    ap.add_argument("--build", action="store_true", help="run engine/build.sh first")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

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
    # Section (b) reads `logits{pos}.bin` by name, so a run that stops early is judged against whatever
    # the PREVIOUS run left behind. The first device run stopped at prompt index 6, so it wrote
    # logits0..6 — and still reported only six failures, all in section (a), which is only possible if
    # section (b) found logits17..24.bin that run cannot have written (its default `--work` is the same
    # directory `--dry-run` and `--selftest` write into). Clear them: (b) must judge the run it is
    # judging, and a missing dump has to read as missing.
    for stale in work.glob("logits*.bin"):
        stale.unlink()
    for stale in work.glob("logits*.bin"):
        stale.unlink()
    cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
           "--steps", str(n_decode), "--out", str(work)]
    if args.dry_run:
        stdout, rc = dry_run(man, prompt_ids, dec_logits, work, args.break_ids,
                             args.break_logits, args.break_noise, args.break_prompt_stop)
        notes.append("DRY RUN: the gate's own plumbing was exercised against a stand-in that replays"
                     " the pins. NOTHING ENGINE-RELATED WAS MEASURED. It exists so the first device"
                     " run is not also the first time this file parses a STEP line. Section (d) is"
                     " Python comparing with itself in this mode; only a real decode_run makes it a"
                     " cross-check.")
        if args.break_ids >= 0:
            notes.append(f"injected: generation step {args.break_ids} sampled the wrong id")
        if args.break_logits >= 0:
            notes.append(f"injected: generation step {args.break_logits} had its top two logits"
                         " swapped")
        if args.break_noise >= 0:
            notes.append(f"injected: generation step {args.break_noise} got 5 % relative noise"
                         " (wrong in aggregate, argmax untouched)")
        if args.break_prompt_stop >= 0:
            notes.append(f"injected: the §07.5 stop check fired at prompt step"
                         f" {args.break_prompt_stop} and ended the run there")
    else:
        print("  $ " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout, rc = proc.stdout, proc.returncode
        for line in proc.stdout.splitlines():
            print("      " + line)
    if rc != 0:
        check(False, "decode_run exits 0", f"rc={rc}")
        return report()
    check(True, "decode_run exits 0")
    run = parse_run(stdout)
    check(bool(run["steps"]), "one STEP line per token", f"{len(run['steps'])} lines")
    if not run["steps"]:
        notes.append("no STEP lines at all: decode_run printed nothing the gate could parse — read"
                     " the raw log above (a load failure or a thrown HIP error exits before any step)")
        return report()

    # ---- (a) sampled ids -----------------------------------------------------------------------
    section("(a) sampled token ids — §03.4 greedy, §10.1 G7 exactness")
    prompt_steps = [s for s in run["steps"] if s["kind"] == "prompt"]
    decode_steps = [s for s in run["steps"] if s["kind"] == "decode"]
    # WHY the prompt came up short, not just that it did. The run log already says what ended it, and
    # reading that log on a device window is exactly the round trip these two lines exist to skip.
    if len(prompt_steps) != len(prompt_ids):
        last = run["steps"][-1] if run["steps"] else {}
        fired = last.get("stop") == "1"
        why = (f"{len(prompt_steps)} of {len(prompt_ids)} prompt tokens fed — the run ended early."
               f" Last STEP: kind={last.get('kind', 'no STEP line at all')}"
               f" index={last.get('index', '-')} sampled={last.get('sampled', '-')}"
               f" stop={last.get('stop', '-')}; RESULT: {run['result'] or 'no RESULT line'} — "
               + ("that last step's §07.5 stop check HAD fired. A stop match during the prompt phase"
                  " is not a stop: the sample is discarded and the next fed token comes from the"
                  f" fixture, and a chat prompt legitimately contains the stop id {STOP_ID} (dec8's"
                  " does, at index 8). `decode_run.cpp` owns the generation-only rule; it is the"
                  " first device run's bug, reproduced here so it cannot come back quietly"
                  if fired else
                  "no stop check on the last step, so this is an early exit or a thrown error rather"
                  " than §07.5 — read the lines above for the HIP message"))
    else:
        why = f"{len(prompt_steps)} vs {len(prompt_ids)}"
    check(len(prompt_steps) == len(prompt_ids), "one step per prompt token", why)
    # The same bug seen from the other side, as a standing tripwire: no prompt-phase step may be the one
    # that fires §07.5. The raw match is NOT the failure here — prompt-phase steps sampling the stop id is
    # expected on a chat prompt and prints `stop=0` with that id in `sampled=`. `stop=` is what
    # decode_run's human-readable [STOP] marker means (same printf condition), so this covers it too.
    prompt_stop = [s for s in prompt_steps if s.get("stop") == "1"]
    check(not prompt_stop, "no prompt-phase step fires §07.5's stop check",
          "" if not prompt_stop else
          f"kind=prompt index={prompt_stop[0].get('index')}"
          f" sampled={prompt_stop[0].get('sampled')} ended the run; §07.5 arms only during"
          f" generation (and {STOP_ID} inside a prompt is not news — see decode_run.cpp)")
    check(len(decode_steps) == n_decode, f"{n_decode} generation steps", f"{len(decode_steps)}")
    check(int(prompt_steps[-1]["sampled"]) == greedy[0] if prompt_steps else False,
          "the prompt's last token samples greedy_continuation[0]",
          f"{prompt_steps[-1]['sampled'] if prompt_steps else 'no prompt step ran'} vs {greedy[0]}")
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
    if stdout.count("[STOP]") and greedy[-1] not in man["special_ids"].values():
        notes.append(f"engine stopped on id {STOP_ID} in the generation phase but the manifest's"
                     " continuation does not")

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
        margin = ("exact" if m["margin"] == float("inf") else f"{m['margin']:.0f}×")
        check(True, f"step {j}: metrics",
              f"cos {m['cos']:.6f} meanrel {m['meanrel']:.4%} ndiff {m['ndiff']}/{ref.size}"
              f" maxabs {m['maxabs']:.4g} | ref top1−top2 gap {m['gap12']:.3f}"
              f" = {margin} the largest element error")
        check(m["cos"] >= BAND["cos_hard"], f"step {j}: cos >= {BAND['cos_hard']} (§10.1 G5)")
        check(m["meanrel"] <= BAND["meanrel_hard"],
              f"step {j}: meanrel <= {BAND['meanrel_hard']:.0%} (§10.2)")
        check(argmax_low(eng) == argmax_low(ref), f"step {j}: argmax agrees with the pin",
              f"{argmax_low(eng)}")
        if m["cos"] < BAND["cos_report"]:
            notes.append(f"step {j}: cos {m['cos']:.6f} is under G6's decode-logits figure "
                         f"{BAND['cos_report']} (still above the G5 floor that is gated). G6 is"
                         " ADVISORY here by decision, not an open question: an end-to-end token"
                         " carries the whole 24-layer cascade, so it is not judged by a figure"
                         " belonging to a narrower comparison — see the BAND comment above, and"
                         " the two measured floors in engine/results.md: last-op reassociation"
                         " 0.99999862 (engine/logits_floor.py) and end-to-end two-legal-routes"
                         " 1-cos 4.7e-5..1.9e-4 (engine/route_price.py), which brackets this step.")
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
