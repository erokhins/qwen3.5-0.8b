#!/usr/bin/env python3
"""Interactive chat on the HIP engine, with per-turn stats.

    engine/chat.py                      # REPL
    engine/chat.py --prompt "..."       # one answer, then exit
    engine/chat.py --max-new 32         # cap generation length

This is NOT the speccheck driver. `speccheck/chat.py` runs the torch CPU reference;
this one drives the compiled engine — one `decode_step` per token on the iGPU — through
`engine/build/decode_run --session`, and the weights and the `RequestState` stay resident
in that one process between turns. Resident is the whole point: the load is 1.55 GB of
`hipMalloc` + `hipMemcpy` before the first token and the request state cannot cross a
process boundary, so a process per turn would pay the load again and re-feed the whole
transcript. `decode_run --session`'s header comment is the protocol this speaks.

WHAT THE NUMBERS MEAN. The tok/s and TTFT printed here are the engine's own wall clock on
a box that is not provably exclusive — `flash_serve`, the server generating this session's
own tokens, polls the same iGPU. So these are "is the engine behaving sanely" figures, NOT
ledger numbers: `bench/performance-status.md` §6 owns what a clean window takes (zero
subagents, blocking, exclusive) and nothing printed here may be pasted into that ledger.
See also `AGENTS.md`, "GPU execution is serialized and belongs to the main agent".

One device, one holder. The GPU must not be shared with a measurement, so this process
takes `tools/gpu_lock.sh` for its whole life and refuses to start if the lock is already
held (that is also why there is no thread knob here, unlike the torch CPU original: the
engine's parallelism is the kernel's block count, not OMP).

Special tokens are printed as `[id]`, never as their literal text — the rule in `AGENTS.md`
exists because every agent on this box is served by this same checkpoint, and a literal
end-of-turn token in a transcript ends whatever message contains it. `speccheck/chat.py`
streams them literally; this file deliberately does not.
"""
from __future__ import annotations

import argparse
import atexit
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speccheck import textio                                          # noqa: E402
from speccheck.textio import Tokenizer                                # noqa: E402

DEFAULT_BINARY = ROOT / "engine" / "build" / "decode_run"
# The engine stops on 248046 alone (`engine/decode_step.h`, `kStopToken`, §07.5). The torch
# reference additionally treats 248044 as a stop (A-10); it is in this tuple so the token is
# masked from the display, NOT so it ends a turn. See the `stop=` note in the turn stats line.
STOP_IDS = (textio.EOS_PRIMARY_ID, textio.EOS_LEGACY_ID)
# Every id that must never reach the terminal as its literal text (AGENTS.md).
MASKED = (textio.THINK_START_ID, textio.THINK_END_ID, textio.EOS_PRIMARY_ID,
          textio.EOS_LEGACY_ID, 248045)
TERMINAL = ("FED", "END", "RESET", "PREFIX", "ERROR", "BYE")


def rate(n: int, secs: float) -> str:
    return "%.1f tok/s" % (n / secs) if secs > 0 and n > 0 else "-"


class EngineSession:
    """Line-protocol driver for `decode_run --session`.

    Reads are blocking and line-buffered; every command ends at one of `TERMINAL`, so the
    driver can't mistake a streamed `GEN` line for an answer. An `ERROR` reply is raised:
    a step that threw left the state half-advanced (§07.1 runs the layers in order and
    `prefix_len` advances last), and continuing on top of that is the quiet-corruption case.
    """

    def __init__(self, binary: str, model: str, cap: int):
        self.p = subprocess.Popen(
            [binary, "--model", model, "--session", "--cap", str(cap)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        atexit.register(self.close)
        # The engine announces itself before it will read a command. Anything before `READY` is
        # the loader's own banner (tensors, allocation, its own ms) — kept and echoed, never
        # parsed, so a change to that line cannot silently change what `cap` means here.
        self.banners: list[str] = []
        first = self._send("", first_line=True)
        while first["verb"] != "READY":
            self.banners.append(first["raw"])
            first = self._send("", first_line=True)
        self.cap = int(first["cap"])

    def _send(self, line: str, on_token=None, first_line: bool = False) -> dict:
        assert self.p.stdin is not None and self.p.stdout is not None
        if self.p.poll() is not None:
            raise RuntimeError(f"engine exited (rc={self.p.returncode}); see the lines above")
        if not first_line:
            self.p.stdin.write(line + "\n")
            self.p.stdin.flush()
        out: dict = {"gen": []}
        while True:
            raw = self.p.stdout.readline()
            if not raw:
                self.p.wait()
                raise RuntimeError(f"engine closed the session mid-command (rc={self.p.returncode})")
            word, _, rest = raw.strip().partition(" ")
            if word == "GEN":
                gid, pos = rest.split()
                out["gen"].append((int(gid), int(pos)))
                if on_token:
                    on_token(int(gid))
                continue
            out["verb"] = word
            out["raw"] = raw.strip()
            out.update(dict(kv.split("=", 1) for kv in rest.split() if "=" in kv))
            if word == "ERROR":
                raise RuntimeError(out["raw"])
            return out

    def feed(self, ids) -> dict:
        return self._send("feed" + (" " + " ".join(map(str, ids)) if ids else ""))

    def gen(self, n: int, on_token=None) -> dict:
        return self._send(f"gen {n}", on_token)

    def reset(self) -> dict:
        return self._send("reset")

    def stats(self) -> dict:
        return self._send("stats")

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.write("quit\n")
                self.p.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--cap", type=int, default=4096,
                    help="slots to allocate up front; the state is grown-once, not per turn "
                         "(§08.6). A turn that would pass it is refused, not truncated.")
    ap.add_argument("--prompt", default=None, help="answer this and exit (batch use)")
    ap.add_argument("--ignore-lock", action="store_true",
                    help="start even if tools/gpu_lock.sh says the GPU is held (you will be "
                         "contending with someone's measurement)")
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        print(f"engine chat: no engine binary at {binary}\n"
              "  build it with:  bash engine/build.sh", file=sys.stderr)
        return 1
    if not (Path(args.model) / "config.json").exists():
        print(f"engine chat: no checkpoint at {args.model}\n"
              "  get it with:  hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B",
              file=sys.stderr)
        return 1

    lock = ROOT / "tools" / "gpu_lock.sh"
    tag = "engine/chat.py interactive session"
    have_lock = False
    if lock.exists() and not args.ignore_lock:
        held = subprocess.run([str(lock), "status"], capture_output=True, text=True).stdout
        if held.startswith("HELD"):
            print(f"engine chat: the GPU is spoken for — {held.strip()}\n"
                  "chat would contend with that run, and its own tok/s would be a lie either"
                  " way.\nWait for it, or pass --ignore-lock if you are sure it is stale.",
                  file=sys.stderr)
            return 1
        if subprocess.run([str(lock), "acquire-noblock", tag]).returncode != 0:
            print("engine chat: could not take the GPU lock; refusing to contend.",
                  file=sys.stderr)
            return 1
        have_lock = True

    def release():
        if have_lock:
            subprocess.run([str(lock), "release", tag])
    atexit.register(release)

    tok = Tokenizer(args.model)
    t0 = time.time()
    try:
        eng = EngineSession(str(binary), args.model, args.cap)
    except (RuntimeError, ValueError) as exc:
        print(f"engine chat: the engine did not come up: {exc}", file=sys.stderr)
        return 1
    load = time.time() - t0
    for banner in eng.banners:
        print("  " + banner)
    print("HIP engine chat | engine/build/decode_run --session, GPU-resident, bf16 greedy"
          " | one decode_step per token\n"
          "  tok/s here is NOT a ledger number: it is this box while it is also serving this"
          " session. bench/performance-status.md §6 owns clean windows.\n"
          f"  cap={eng.cap} slots | stop id {textio.EOS_PRIMARY_ID} (generation phase only,"
          f" §07.5) | max_new={args.max_new} | up in {load:.1f}s\n"
          "  commands: /stats /reset /raw <text> /quit", flush=True)

    messages: list[dict] = []
    last_stop = ""
    agg = dict(pre_n=0, pre_t=0.0, dec_n=0, dec_t=0.0, ttft=[], turns=0)

    def show(text: str):
        sys.stdout.write(text)
        sys.stdout.flush()

    def turn(text: str, chat: bool = True):
        nonlocal last_stop
        if not chat:
            delta, mode = tok.encode(text), "raw"
        elif not messages:
            messages.append({"role": "user", "content": text})
            delta, mode = tok.render_ids(messages), "first"
        else:
            messages.append({"role": "user", "content": text})
            # Same incremental-continuation rule as `speccheck/chat.py`, for the same reason:
            # re-rendering the whole conversation is NOT a token extension of what the state
            # holds (the pinned generation tail differs from a re-render of the assistant
            # turn), so a re-render would force a cold prefill every turn. Close the assistant
            # turn only if generation was cut off before the stop token, then append the next
            # user turn and the header. The added tokens are hard BPE boundaries, so encoding
            # this fragment cannot merge across them.
            close = "" if last_stop == "eos" else textio.IM_END + "\n"
            cont = (close + textio.IM_START + "user\n" + text + textio.IM_END + "\n"
                    + textio.IM_START + "assistant\n" + textio.THINK_START + "\n\n"
                    + textio.THINK_END + "\n\n")
            delta, mode = tok.encode(cont), "resume"
        t_start = time.perf_counter()
        try:
            fed = eng.feed(delta)
            t_first = time.perf_counter()
        except RuntimeError as exc:
            print(f"  {exc}")
            messages.pop()
            return
        out: list[int] = []
        # Byte-level BPE puts a multi-byte character across several tokens, so decoding one id at
        # a time prints U+FFFD for every partial sequence (the torch driver has this artefact), and
        # re-decoding the whole run and printing by character count does not fix it either: the two
        # bytes of an emoji decode to TWO replacement characters, then collapse to ONE real one, so
        # the string can shrink while it is really growing. What is stable is the prefix before any
        # trailing replacement run — only the tail can still change — so that prefix is what gets
        # printed, and the rest is flushed when the run ends.
        run: list[int] = []
        shown = 0

        def emit(final: bool):
            nonlocal shown
            text = tok.tok.decode(run, skip_special_tokens=True)
            safe = text if final else text.rstrip("\ufffd")
            if len(safe) > shown:
                show(safe[shown:])
                shown = len(safe)

        def collect(token_id: int):
            nonlocal shown
            out.append(token_id)     # the turn's own ids, for the stats and the history
            if token_id in MASKED:
                emit(final=True)
                run.clear()
                shown = 0
                show(f"[{token_id}]")   # never the literal text — AGENTS.md
                return
            run.append(token_id)
            emit(final=False)

        try:
            end = eng.gen(args.max_new, collect)
        except RuntimeError as exc:
            print(f"  {exc}")
            return
        emit(final=True)             # whatever the tail was holding back
        print(flush=True)
        stop = "stop" if end.get("stop") == "1" else "max_new"
        last_stop = "eos" if stop == "stop" else ""
        if chat:
            # skip_special_tokens for the stored history: the transcript the next turn is
            # rendered from must not carry the masking markers the display uses.
            messages.append({"role": "assistant",
                             "content": tok.tok.decode(out, skip_special_tokens=True)})
        ttft = t_first - t_start
        dec_t = time.perf_counter() - t_first
        n_dec = max(len(out) - 1, 0)      # the first token came from the last prefill step
        prefix = int(end.get("prefix_len", -1))
        # ttft is measured from submitting the turn, so it is prefill + the one step that
        # samples the first answer token — which is the honest meaning of "time to first token"
        # on a resident engine, and why it tracks the prompt length rather than the answer's.
        # decode rate then covers the remaining n_dec over the time between first and last token.
        # `fed n` is what the engine actually stepped, which is one more than this turn's own
        # tokens on a resume turn: the last token of the previous answer was emitted but not
        # consumed then (the state cannot consume a token it has not been given), so `feed`
        # consumes it first. Printing both makes the transcript accounting checkable by eye.
        print("  [prompt %d tok (fed %d) %.2fs | ttft %.2fs | generated %d tok | decode %d tok"
              " %.2fs %s | stop=%s | mode=%s | prefix %d/%d | turn %.1fs]"
              % (len(delta), int(fed.get("n", -1)), t_first - t_start, ttft, len(out), n_dec,
                 dec_t,
                 rate(n_dec, dec_t), stop, mode, prefix, eng.cap,
                 time.perf_counter() - t_start), flush=True)
        if prefix < 0:
            print(f"  [state] the END line carried no prefix_len: {end}")
        agg.update(pre_n=agg["pre_n"] + len(delta), pre_t=agg["pre_t"] + (t_first - t_start),
                   dec_n=agg["dec_n"] + n_dec, dec_t=agg["dec_t"] + dec_t,
                   ttft=agg["ttft"] + [ttft], turns=agg["turns"] + 1)

    def summary(prefix="bye"):
        tt = (" | ttft %.2fs avg, %.2fs worst" % (sum(agg["ttft"]) / len(agg["ttft"]),
                                                  max(agg["ttft"]))) if agg["ttft"] else ""
        return ("%s — %d turns | prompt %d tok @ %s | decode %d tok @ %s%s"
                % (prefix, agg["turns"], agg["pre_n"], rate(agg["pre_n"], agg["pre_t"]),
                   agg["dec_n"], rate(agg["dec_n"], agg["dec_t"]), tt))

    if args.prompt:
        turn(args.prompt, chat=not args.prompt.startswith("/raw "))
        print(summary())
        return 0

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/reset":
            print("  " + eng.reset().get("raw", "reset"))
            messages.clear()
            last_stop = ""
            continue
        if line == "/stats":
            st = eng.stats()
            print("  " + summary("session") + f" | state prefix {st.get('p')}/{eng.cap}")
            continue
        if line.startswith("/raw "):
            turn(line[5:], chat=False)
            continue
        turn(line)

    print(summary())
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
