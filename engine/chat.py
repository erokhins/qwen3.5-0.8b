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

THE TWO PHASES ARE PRINTED SEPARATELY, because they are two different questions.

  * the LOAD phase — one line per process: wall time split into tokenizer build and engine
    start, and within the engine, what the loader itself reported (tensors, resident GB, ms,
    allocations) versus what the rest of start-up cost (HIP context, state allocation, route
    planning). It used to be visible only under `--verbose`; it is on by default now.
  * the WARM rates — prefill tok/s and decode tok/s of the steady state, which is what "how
    fast is this box" means. A process's FIRST prefill is not steady state: it pays each
    kernel's code-object load on its first launch, and measured 0.09s for a 21-token turn
    against 0.01s for a same-sized second one — a factor of ~6 about the process, not the
    prompt. That turn is bucketed COLD and kept out of the warm figure. A later turn that
    introduces a width no earlier turn used pays hipBLASLt route pricing mid-turn (the
    `GEMM-LT` lines), is labelled `+N priced`, and STAYS in the warm bucket: the same
    measurement read 1,438 tok/s on a turn with seven routes priced inside it, so pricing
    is milliseconds next to the launch cost. `--warmup` (default 512 tokens, one full
    `kMaxWaveRows` wave) buys the cold cost before the first turn you care about and prints
    it as its own line; the state is reset straight after, so nothing of it survives.

OUTPUT IS QUIET BY DEFAULT. The engine narrates itself: the loader banner, `STATE`,
`LAUNCH`, `NUMERICS`, `GEMM`, and one `GEMM-LT` line per (shape, M) the hipBLASLt route
prices for the first time — including *mid-turn*, when a turn introduces a width no
earlier turn used. Only `--verbose` echoes any of that. The default view is the answer
plus one rate line per turn, and `--summary` prints the summary plus a prefill/generation
rate block.

WHAT THE NUMBERS MEAN. The tok/s and TTFT printed here are the engine's own wall clock on
a box that is not provably exclusive — `flash_serve`, the server generating this session's
own tokens, polls the same iGPU. So these are "is the engine behaving sanely" figures, NOT
ledger numbers: `engine/performance-status.md` §6 owns what a clean window takes (zero
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
import re
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
# The loader's own report line, printed before `READY` (`engine/decode_run.cpp`, right after
# `WeightStore::load`). Parsed rather than eyeballed so the load phase can be split into
# "what loading 1.55 GB cost" and "what everything else about start-up cost" without
# `--verbose`, and so a change to the line prints nothing new instead of a silent zero.
LOAD_RE = re.compile(r"loaded (\d+) tensors, ([\d.]+) GB resident in ([\d.]+) ms \((\d+) allocations\)")

# The summarisation ask. Phrased for a 0.8B model: one instruction, an explicit shape, and
# no options it has to make a judgement call about. `--instruction` overrides it.
SUMMARY_INSTRUCTION = (
    "Read the document below and write a summary of it.\n"
    "Start with one sentence saying what the document is about, then list its key points"
    " as bullet lines. Write in the same language as the document.\n\n"
    "Document:")


def rate(n: int, secs: float) -> str:
    return "%.1f tok/s" % (n / secs) if secs > 0 and n > 0 else "-"


def rate_line(label: str, n: int, secs: float, extra: str = "") -> str:
    """One aligned `label  N tok in T.TTs  =  R tok/s  (extra)` row."""
    return f"  {label:<15}{n:>9,} tok in {secs:6.2f}s  =  {rate(n, secs)}{extra}"


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
        # What the loader reported about itself, None if its line ever changes shape.
        self.load_ms: float | None = None
        self.load_gb: str = "?"
        self.load_tensors: str = "?"
        self.load_allocs: str = "?"
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
            if m := LOAD_RE.search(raw):
                self.load_tensors, self.load_gb = m.group(1), m.group(2)
                self.load_ms, self.load_allocs = float(m.group(3)), m.group(4)

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
                ttft: float = None, stop: str = None, cold=None) -> str:
    """The prefill/generation rate block, in the vocabulary of the engine's own `RESULT` line.
    Same shape for one turn, one document, and a whole session, so the three are comparable
    by eye.

    `cold` is the (tokens, seconds) of `pre_*` that the process's first-launch cost was paid
    on. It is already INSIDE `pre_n`/`pre_t`; when it is non-zero it gets its own row and the
    prefill row is restated as the warm remainder, because the warm rate is the one that
    describes the machine. `cold=None` (or zero) means there was nothing cold in the bucket.
    """
    head = f"{label}\n" if label else ""
    t_tf = f"  (ttft {ttft:.2f}s)" if ttft is not None else ""
    t_stop = f"  (stop={stop})" if stop else ""
    cn, ct = cold if cold else (0, 0)
    wn, wt = pre_n - cn, pre_t - ct
    if cn and wn:
        rows = [rate_line("prefill (warm)", wn, wt, t_tf),
                rate_line("  + cold", cn, ct, "  (process's first launches)")]
    elif cn:
        rows = [rate_line("prefill (cold)", cn, ct,
                          t_tf + "  — nothing warm measured; pass --warmup")]
    else:
        rows = [rate_line("prefill (warm)", pre_n, pre_t, t_tf)]
    rows.append(rate_line("generation", dec_n, dec_t, t_stop))
    return head + "\n".join(rows)


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
    ap.add_argument("--warmup", type=int, default=512, metavar="N",
                    help="before the first real turn, prefill N throwaway tokens and generate"
                         " 2, then reset — so every turn you read is a WARM prefill. The"
                         " process's first prefill pays each kernel's code-object load on its"
                         " first launch (default 512 = one full wave of kMaxWaveRows rows;"
                         " 0 disables it, and then the first turn is the cold one and is"
                         " labelled and bucketed as such)")
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

    t_tok = time.perf_counter()
    tok = Tokenizer(args.model)
    tok_s = time.perf_counter() - t_tok

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

    t0 = time.perf_counter()
    try:
        eng = EngineSession(str(binary), args.model, cap, verbose=verbose)
    except (RuntimeError, ValueError) as exc:
        print(f"engine chat: the engine did not come up: {exc}", file=sys.stderr)
        return 1
    eng_s = time.perf_counter() - t0
    load = tok_s + eng_s
    if verbose:
        for banner in eng.banner:
            print("  " + banner)
        print(f"  engine up in {eng_s:.1f}s | {len(eng.diag)} diagnostic lines so far")
    print(f"Qwen3.5-0.8B on the HIP engine | bf16, greedy | weights and state GPU-resident"
          f" (cap {eng.cap} slots, stop id {textio.EOS_PRIMARY_ID}, max_new {max_new})",
          flush=True)
    # The load phase, always: it is a different quantity from a turn rate (one charge per
    # process, not per token) and it used to hide behind --verbose.
    if eng.load_ms is not None:
        rest = eng_s - eng.load_ms / 1000.0
        print(f"  load      {load:.1f}s wall = tokenizer {tok_s:.1f}s + engine {eng_s:.1f}s"
              f" — weights {eng.load_gb} GB / {eng.load_tensors} tensors"
              f" ({eng.load_allocs} allocs) in {eng.load_ms / 1000.0:.2f}s,"
              f" hip init + state + routes {rest:.2f}s", flush=True)
    else:
        print(f"  load      {load:.1f}s wall = tokenizer {tok_s:.1f}s + engine {eng_s:.1f}s"
              f" (loader line not recognised, so the split inside the engine is unknown)",
              flush=True)
    print("  rates here are this box's wall clock, not a ledger number"
          " (engine/performance-status.md §6).", flush=True)

    messages: list[dict] = []
    last_stop = ""
    agg = dict(pre_n=0, pre_t=0.0, dec_n=0, dec_t=0.0, ttft=[], turns=0,
               cold_n=0, cold_t=0.0, warm_n=0, warm_t=0.0, priced=0)

    # ---- the warmup: buy the cold costs before any turn you care about. ----------------------
    # The one-time cost on a process's first prefill is each kernel's code object loading on its
    # first launch: measured 0.09s for a first 21-token turn against 0.01s for a same-sized
    # second one, and 1,767 tok/s against 11,306 tok/s on a 173-token document. hipBLASLt route
    # pricing is the other first-time cost (the GEMM-LT lines), but it is solved per (shape, M)
    # so a fixed-width warmup could not buy it for an arbitrary prompt anyway — and it measures
    # at milliseconds next to the launch cost, so a turn that pays it is labelled `+N priced`
    # and stays in the warm bucket. 512 tokens is one full kMaxWaveRows wave, the width the
    # engine actually emits; the state is reset straight after, so nothing of the warmup is
    # left in it.
    warmed = False
    if args.warmup > 0:
        if args.warmup + 8 > cap:
            print(f"  warmup    skipped: {args.warmup} tokens plus generation does not fit a"
                  f" {cap}-slot state (--cap, or lower --warmup)", flush=True)
        else:
            word = tok.encode(" the quick brown fox jumps over the lazy dog")
            ids = (word * (args.warmup // max(len(word), 1) + 2))[:args.warmup]
            t_w = time.perf_counter()
            try:
                eng.feed(ids)
                t_w_pre = time.perf_counter() - t_w
                eng.gen(2)
            except RuntimeError as exc:
                print(f"  warmup    FAILED: {exc}", flush=True)
                return 1
            t_w_all = time.perf_counter() - t_w
            eng.reset()
            warmed = True
            n_price = sum(1 for l in eng.diag if l.startswith("GEMM-LT"))
            print(f"  warmup    {len(ids):,} tok in {t_w_pre:.2f}s  =  {rate(len(ids), t_w_pre)}"
                  f"  (COLD: first kernel launches"
                  f"{f', {n_price} routes priced' if n_price else ''}; 2 decode steps"
                  f" {t_w_all - t_w_pre:.2f}s) — state reset, every turn below is warm", flush=True)

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
        n_diag0 = len(eng.diag)
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
        # Which bucket this turn's prefill belongs in, and why. The process's FIRST prefill is
        # COLD: it pays each kernel's code-object load on its first launch. Measured on this
        # box with --warmup 0, a 21-token first turn took 0.09s (232 tok/s) and the same-sized
        # second turn 0.01s (1,438 tok/s) — a factor of ~6 that is a property of the process,
        # not the prompt, so it does not belong in a rate that claims to describe the machine.
        # A later turn that introduces a width no earlier turn used makes hipBLASLt price new
        # routes mid-turn (the `GEMM-LT` lines, counted here) and is labelled `+N priced`, but
        # stays in the warm bucket: the same measurement says 18 tok at 1,438 tok/s WITH seven
        # routes priced inside it, so pricing is milliseconds next to the launch cost.
        n_price = sum(1 for l in eng.diag[n_diag0:] if l.startswith("GEMM-LT"))
        first_prefill = agg["turns"] == 0 and not warmed
        cold = first_prefill
        tag = ("cold" if first_prefill else "") + (f" +{n_price} priced" if n_price else
                                                  ("" if first_prefill else "warm"))
        if verbose:
            print("  [prompt %d tok (fed %d) %.2fs %s%s | ttft %.2fs | generated %d tok | decode %d tok"
                  " %.2fs %s | stop=%s | mode=%s | prefix %d/%d | turn %.1fs]"
                  % (n_pre, int(fed.get("n", -1)), t_pre, rate(n_pre, t_pre),
                     " [" + tag.strip() + "]", ttft, len(out), n_dec, dec_t,
                     rate(n_dec, dec_t), stop, mode, prefix, eng.cap,
                     time.perf_counter() - t_start), flush=True)
        elif stats_line:
            # One line per turn in the REPL; --summary skips it and prints the full rate block
            # instead, so the same figures are never on screen twice.
            print("  [%s tok prompt %.2fs %s %-11s | %s tok answer %.2fs %s | ttft %.2fs | %s]"
                  % (f"{n_pre:,}", t_pre, rate(n_pre, t_pre), tag.strip(), f"{len(out):,}", dec_t,
                     rate(n_dec, dec_t), ttft, stop), flush=True)
        if prefix < 0:
            print(f"  [state] the END line carried no prefix_len: {end}")
        agg.update(pre_n=agg["pre_n"] + n_pre, pre_t=agg["pre_t"] + t_pre,
                   dec_n=agg["dec_n"] + n_dec, dec_t=agg["dec_t"] + dec_t,
                   ttft=agg["ttft"] + [ttft], turns=agg["turns"] + 1, priced=agg["priced"] + n_price)
        if cold:
            agg.update(cold_n=agg["cold_n"] + n_pre, cold_t=agg["cold_t"] + t_pre)
        else:
            agg.update(warm_n=agg["warm_n"] + n_pre, warm_t=agg["warm_t"] + t_pre)
        return dict(pre_n=n_pre, pre_t=t_pre, dec_n=n_dec, dec_t=dec_t, ttft=ttft,
                    cold=(n_pre, t_pre) if cold else None,
                    text=tok.tok.decode(out, skip_special_tokens=True), stop=stop)

    def summary(prefix="bye"):
        tt = (" | ttft %.2fs avg, %.2fs worst" % (sum(agg["ttft"]) / len(agg["ttft"]),
                                                  max(agg["ttft"]))) if agg["ttft"] else ""
        warm = (f" (warm {agg['warm_n']:,} tok @ {rate(agg['warm_n'], agg['warm_t'])})"
                if agg["cold_n"] and agg["warm_n"] else "")
        cold = (f" | cold {agg['cold_n']:,} tok @ {rate(agg['cold_n'], agg['cold_t'])}"
                if agg["cold_n"] else "")
        return ("%s — %d turns | prompt %d tok @ %s%s%s | decode %d tok @ %s%s"
                % (prefix, agg["turns"], agg["pre_n"], rate(agg["pre_n"], agg["pre_t"]),
                   warm, cold, agg["dec_n"], rate(agg["dec_n"], agg["dec_t"]), tt))

    def end_rates(label=""):
        tt = sum(agg["ttft"]) / len(agg["ttft"]) if agg["ttft"] else None
        cold = (agg["cold_n"], agg["cold_t"]) if agg["cold_n"] else None
        hint = "" if agg["turns"] else "  (no turn ran)"
        return rates_block(label + hint, agg["pre_n"], agg["pre_t"], agg["dec_n"],
                           agg["dec_t"], ttft=tt, stop=None, cold=cold)

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
                              ttft=r["ttft"], stop=r["stop"], cold=r["cold"]))
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
