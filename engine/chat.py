#!/usr/bin/env python3
"""Chat and one-shot document summarisation on the HIP engine, with prefill/decode rates.

    engine/chat.py                                  # REPL
    engine/chat.py --prompt "..."                  # one answer, then exit
    engine/chat.py --summary notes.md              # summarize a document
    engine/chat.py --summary a.md b.md --max-new 400
    engine/chat.py --summary - < notes.md          # document on stdin
    engine/chat.py --verbose                        # also echo the engine's diagnostics

This is NOT the speccheck driver. `speccheck/chat.py` runs the torch CPU reference;
this one drives the compiled engine — one `decode_step` per token on the iGPU — through
`engine/build/decode_run --session`, and the weights and the `RequestState` stay resident
in that one process between turns. Resident is the whole point: the load is 1.55 GB of
`hipMalloc` + `hipMemcpy` before the first token and the request state cannot cross a
process boundary, so a process per turn would pay the load again and re-feed the whole
transcript. `decode_run --session`'s header comment is the protocol this speaks.

OUTPUT IS QUIET BY DEFAULT. The engine narrates itself: the loader banner, `STATE`,
`LAUNCH`, `NUMERICS`, `GEMM`, and one `GEMM-LT` line per (shape, M) the hipBLASLt route
prices for the first time — including *mid-turn*, when a turn introduces a width no
earlier turn used. Only `--verbose` echoes any of that. The default view is the answer
plus one rate line per turn, and `--summary` prints the summary plus a two-line
prefill/generation rate block.

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
# The verbs `decode_run --session` ends a reply with (`session_loop` in `engine/decode_run.cpp`).
# This list is the whole protocol boundary, and the reason it is a list rather than "the next
# non-GEN line": the engine writes diagnostics to the same stdout as the protocol, and it writes
# them DURING a command — the hipBLASLt route prints one `GEMM-LT` line per (shape, M) it prices
# for the first time, so the first turn of a new prompt width emits seven of them inside `feed`.
# A driver that takes the first non-GEN line for the reply gets an algo hex back instead of `FED`,
# generates zero tokens, and reports `prefix -1`. Anything not in this list is a diagnostic.
TERMINAL = ("FED", "END", "RESET", "PREFIX", "ERROR", "BYE", "READY")

# The summarisation ask. Phrased for a 0.8B model: one instruction, an explicit shape, and
# no options it has to make a judgement call about. `--instruction` overrides it.
SUMMARY_INSTRUCTION = (
    "Read the document below and write a summary of it.\n"
    "Start with one sentence saying what the document is about, then list its key points"
    " as bullet lines. Write in the same language as the document.\n\n"
    "Document:")


def rate(n: int, secs: float) -> str:
    return "%.1f tok/s" % (n / secs) if secs > 0 and n > 0 else "-"


def read_document(spec: str) -> str:
    """Read a `--summary` operand: a path, or `-` for stdin.

    Decoded with errors="replace" rather than strictly: a summary target is somebody's
    log or export and one stray byte must not lose the whole document.
    """
    if spec == "-":
        return sys.stdin.read()
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"engine chat: no such document: {spec}")
    if p.is_dir():
        raise SystemExit(f"engine chat: {spec} is a directory, not a document")
    return p.read_text(errors="replace")


class EngineSession:
    """Line-protocol driver for `decode_run --session`.

    Reads are blocking and line-buffered; every command ends at one of `TERMINAL`, so the
    driver can't mistake a streamed `GEN` line for an answer — and, equally, can't mistake
    the engine's own chatter for one either (see `TERMINAL`). An `ERROR` reply is raised:
    a step that threw left the state half-advanced (§07.1 runs the layers in order and
    `prefix_len` advances last), and continuing on top of that is the quiet-corruption case.
    """

    def __init__(self, binary: str, model: str, cap: int, verbose: bool = False):
        self.verbose = verbose
        self.diag: list[str] = []
        self.banner: list[str] = []
        self.p = subprocess.Popen(
            [binary, "--model", model, "--session", "--cap", str(cap)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        atexit.register(self.close)
        # The engine announces itself with `READY cap=<n>` before it will read a command.
        # Everything before it is the loader's own report (tensors, allocation, its own ms) —
        # kept, and echoed only under `--verbose`, so a change to those lines cannot silently
        # change what `cap` means here.
        self.cap = self._await_ready()

    def _diag(self, line: str):
        self.diag.append(line)
        if self.verbose:
            print("  ! " + line, flush=True)

    def _read(self) -> str:
        raw = self.p.stdout.readline()
        if not raw:
            self.p.wait()
            raise RuntimeError(f"engine closed the session mid-command (rc={self.p.returncode})")
        return raw.strip()

    def _await_ready(self) -> int:
        while True:
            raw = self._read()
            word, _, rest = raw.partition(" ")
            if word == "READY":
                return int(dict(kv.split("=", 1) for kv in rest.split() if "=" in kv)["cap"])
            self.banner.append(raw)

    def _send(self, line: str, on_token=None) -> dict:
        assert self.p.stdin is not None and self.p.stdout is not None
        if self.p.poll() is not None:
            raise RuntimeError(f"engine exited (rc={self.p.returncode}); see the lines above")
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out: dict = {"gen": []}
        while True:
            raw = self._read()
            word, _, rest = raw.partition(" ")
            if word == "GEN":
                gid, pos = rest.split()
                out["gen"].append((int(gid), int(pos)))
                if on_token:
                    on_token(int(gid))
                continue
            if word not in TERMINAL:
                self._diag(raw)
                continue
            out["verb"] = word
            out["raw"] = raw
            out.update(dict(kv.split("=", 1) for kv in rest.split() if "=" in kv))
            if word == "ERROR":
                raise RuntimeError(raw)
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


def rates_block(label: str, pre_n: int, pre_t: float, dec_n: int, dec_t: float,
                ttft: float = None, stop: str = None) -> str:
    """The two-line prefill/generation rate block, in the vocabulary of the engine's own
    `RESULT` line. Same shape for one turn, one document, and a whole session, so the three
    are comparable by eye."""
    head = f"{label}\n" if label else ""
    t_tf = f"  (ttft {ttft:.2f}s)" if ttft is not None else ""
    t_stop = f"  (stop={stop})" if stop else ""
    return (head
            + "  prefill    %s tok in %6.2fs  =  %s%s\n"
            % (f"{pre_n:,}", pre_t, rate(pre_n, pre_t), t_tf)
            + "  generation %s tok in %6.2fs  =  %s%s"
            % (f"{dec_n:,}", dec_t, rate(dec_n, dec_t), t_stop))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--max-new", type=int, default=None,
                    help="generation cap (default 128 in chat, 512 with --summary)")
    ap.add_argument("--cap", type=int, default=0,
                    help="slots to allocate up front; the state is grown-once, not per turn "
                         "(§08.6). A turn that would pass it is refused, not truncated. "
                         "Default: 4096, or, with --summary, the document plus its answer.")
    ap.add_argument("--max-auto-cap", type=int, default=32768,
                    help="ceiling for the --summary auto-sized cap; past it a document is"
                         " refused with its token count rather than truncated")
    ap.add_argument("--prompt", default=None, help="answer this and exit (batch use)")
    ap.add_argument("--summary", nargs="+", metavar="FILE", default=None,
                    help="summarize each FILE (or '-' for stdin) and exit; prints prefill"
                         " and generation speed for every document")
    ap.add_argument("--instruction", default=SUMMARY_INSTRUCTION,
                    help="override the summarisation instruction sent with each document")
    ap.add_argument("--verbose", action="store_true",
                    help="echo the engine's own banner and mid-turn diagnostics (GEMM-LT,"
                         " STATE, LAUNCH, NUMERICS), and use the detailed per-turn line")
    ap.add_argument("--ignore-lock", action="store_true",
                    help="start even if tools/gpu_lock.sh says the GPU is held (you will be"
                         " contending with someone's measurement)")
    args = ap.parse_args()
    verbose = args.verbose
    max_new = args.max_new if args.max_new is not None else (512 if args.summary else 128)

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
    if args.summary and args.prompt:
        print("engine chat: --summary and --prompt are exclusive", file=sys.stderr)
        return 1

    # Documents are read BEFORE the engine starts, because the cap the state is allocated with
    # has to hold the whole document and its answer: `feed` past the cap throws and §07.1's
    # half-advanced state means the session is poisoned by then, so resizing after the fact is
    # not available. They are rendered to ids as soon as the tokenizer exists, a few lines down.
    docs: list[tuple[str, str]] = []
    if args.summary:
        for spec in args.summary:
            text = read_document(spec)
            if not text.strip():
                print(f"engine chat: {spec} is empty, skipping", file=sys.stderr)
                continue
            docs.append((spec, text))
        if not docs:
            # Every operand was empty. Say so before the lock and the load, not after: there is
            # nothing to summarize and the engine would be starting for nothing.
            print("engine chat: --summary given, but no document had any text", file=sys.stderr)
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

    # Now the cap. An explicit --cap wins; otherwise a chat session keeps the engine's own
    # 4096 default and a summary sizes itself to the document it was handed.
    doc_ids: list[tuple[str, str, list[int]]] = []
    if args.summary:
        for spec, text in docs:
            doc_ids.append((spec, text,
                            tok.render_ids([{"role": "user",
                                             "content": args.instruction + "\n\n" + text}])))
        # The +64 is margin over the rendered prompt (which already carries the template's own
        # tokens) for the wave rounding; the per-document check below is what actually
        # guarantees a turn fits, so an oversized document is refused before anything launches
        # rather than poisoning the session with a §07.1 half-advanced state mid-feed.
        want = max((len(i) for _, _, i in doc_ids), default=0) + max_new + 64
        cap = args.cap or max(4096, ((want + 511) // 512) * 512)
        if verbose and not args.cap:
            print(f"  state sized to the documents: cap={cap} slots "
                  f"(largest prompt {max((len(i) for _, _, i in doc_ids), default=0)} tok"
                  f" + max_new {max_new})")
        for spec, _, ids in doc_ids:
            if len(ids) + max_new > cap:
                print(f"engine chat: {spec} renders to {len(ids)} tokens, which with"
                      f" --max-new {max_new} does not fit the {cap}-slot state.\n"
                      f"  raise --cap (KV is 12 KiB/token, so 32k slots is ~400 MiB),"
                      f" lower --max-new, or shorten the document.", file=sys.stderr)
                return 1
        if cap > args.max_auto_cap and not args.cap:
            print(f"engine chat: this needs a {cap}-slot state, above --max-auto-cap"
                  f" {args.max_auto_cap}; pass --cap explicitly if you want it.",
                  file=sys.stderr)
            return 1
    else:
        cap = args.cap or 4096

    t0 = time.time()
    try:
        eng = EngineSession(str(binary), args.model, cap, verbose=verbose)
    except (RuntimeError, ValueError) as exc:
        print(f"engine chat: the engine did not come up: {exc}", file=sys.stderr)
        return 1
    load = time.time() - t0
    if verbose:
        for banner in eng.banner:
            print("  " + banner)
        print(f"  engine up in {load:.1f}s | {len(eng.diag)} diagnostic lines so far")
    print(f"Qwen3.5-0.8B on the HIP engine | bf16, greedy | weights and state GPU-resident"
          f" (cap {eng.cap} slots, stop id {textio.EOS_PRIMARY_ID}, max_new {max_new})",
          flush=True)
    print("  rates here are this box's wall clock, not a ledger number"
          " (bench/performance-status.md §6).", flush=True)

    messages: list[dict] = []
    last_stop = ""
    agg = dict(pre_n=0, pre_t=0.0, dec_n=0, dec_t=0.0, ttft=[], turns=0)

    def show(text: str):
        sys.stdout.write(text)
        sys.stdout.flush()

    def turn(text: str, chat: bool = True, pre_ids: list[int] | None = None,
             markers: bool = True, stats_line: bool = True):
        """One turn: feed the delta, stream the answer, return this turn's numbers.

        `pre_ids` is the caller's already-rendered token list (used by --summary, where the
        tokens were needed before the engine started to size the cap); otherwise the turn is
        rendered here. Returns None if the engine refused (the transcript is left as it was).
        """
        nonlocal last_stop
        if pre_ids is not None:
            delta, mode = pre_ids, "doc"
        elif not chat:
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
            if chat:
                messages.pop()
            return None
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
            dec = tok.tok.decode(run, skip_special_tokens=True)
            safe = dec if final else dec.rstrip("\ufffd")
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
                if markers:
                    show(f"[{token_id}]")   # never the literal text — AGENTS.md
                return
            run.append(token_id)
            emit(final=False)

        try:
            end = eng.gen(max_new, collect)
        except RuntimeError as exc:
            print(f"  {exc}")
            return None
        emit(final=True)             # whatever the tail was holding back
        print(flush=True)
        stop = "eos" if end.get("stop") == "1" else "max_new"
        last_stop = "eos" if stop == "eos" else ""
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
        n_pre = len(delta)
        t_pre = t_first - t_start
        if verbose:
            print("  [prompt %d tok (fed %d) %.2fs | ttft %.2fs | generated %d tok | decode %d tok"
                  " %.2fs %s | stop=%s | mode=%s | prefix %d/%d | turn %.1fs]"
                  % (n_pre, int(fed.get("n", -1)), t_pre, ttft, len(out), n_dec, dec_t,
                     rate(n_dec, dec_t), stop, mode, prefix, eng.cap,
                     time.perf_counter() - t_start), flush=True)
        elif stats_line:
            # One line per turn in the REPL; --summary skips it and prints the full rate block
            # instead, so the same figures are never on screen twice.
            print("  [%s tok prompt %.2fs %s | %s tok answer %.2fs %s | ttft %.2fs | %s]"
                  % (f"{n_pre:,}", t_pre, rate(n_pre, t_pre), f"{len(out):,}", dec_t,
                     rate(n_dec, dec_t), ttft, stop), flush=True)
        if prefix < 0:
            print(f"  [state] the END line carried no prefix_len: {end}")
        agg.update(pre_n=agg["pre_n"] + n_pre, pre_t=agg["pre_t"] + t_pre,
                   dec_n=agg["dec_n"] + n_dec, dec_t=agg["dec_t"] + dec_t,
                   ttft=agg["ttft"] + [ttft], turns=agg["turns"] + 1)
        return dict(pre_n=n_pre, pre_t=t_pre, dec_n=n_dec, dec_t=dec_t, ttft=ttft,
                    text=tok.tok.decode(out, skip_special_tokens=True), stop=stop)

    def summary(prefix="bye"):
        tt = (" | ttft %.2fs avg, %.2fs worst" % (sum(agg["ttft"]) / len(agg["ttft"]),
                                                  max(agg["ttft"]))) if agg["ttft"] else ""
        return ("%s — %d turns | prompt %d tok @ %s | decode %d tok @ %s%s"
                % (prefix, agg["turns"], agg["pre_n"], rate(agg["pre_n"], agg["pre_t"]),
                   agg["dec_n"], rate(agg["dec_n"], agg["dec_t"]), tt))

    def end_rates(label=""):
        tt = sum(agg["ttft"]) / len(agg["ttft"]) if agg["ttft"] else None
        return rates_block(label, agg["pre_n"], agg["pre_t"], agg["dec_n"], agg["dec_t"],
                           ttft=tt, stop=None)

    # ---- --summary: one fresh request per document, then the rates. -------------------------
    # Each document gets a `reset`: §08.4's fresh request, so nothing from the previous one is
    # in the state, and the prefill figure is that document's own wave rather than a mix.
    if args.summary:
        for spec, text, ids in doc_ids:
            eng.reset()
            messages.clear()
            last_stop = ""
            src = "<stdin>" if spec == "-" else spec
            print(f"\n=== summary of {src} "
                  f"({len(text):,} chars -> {len(ids):,} prompt tok) ===", flush=True)
            r = turn("", chat=False, pre_ids=ids, markers=False, stats_line=False)
            if r is None:
                continue
            print()
            print(rates_block("", r["pre_n"], r["pre_t"], r["dec_n"], r["dec_t"],
                              ttft=r["ttft"], stop=r["stop"]))
        print()
        print(end_rates(f"total — {len(doc_ids)} document(s), {agg['turns']} request(s)"))
        eng.close()
        return 0

    if args.prompt:
        turn(args.prompt, chat=not args.prompt.startswith("/raw "))
        print(end_rates())
        eng.close()
        return 0

    print("  commands: /stats /reset /raw <text> /quit"
          "   (add --verbose for the engine's own lines)", flush=True)
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
        if line == "/help":
            print("  /stats  this session's rates and state prefix")
            print("  /reset  fresh request (§08.4): zeroes the state and the transcript")
            print("  /raw T  send T through the tokenizer with no chat template")
            print("  /quit   exit")
            continue
        if line.startswith("/raw "):
            turn(line[5:], chat=False)
            continue
        turn(line)

    print(end_rates())
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
