# Acceptance, apples to apples: our draft is at llama.cpp's rate; the gap is round cost

## The confound, and why it had to be removed on our side

Every llama.cpp accept number in this repo came from `llama-cli`, whose conversation mode is chosen by the
**presence of a chat template in the GGUF**. This build has **no `-no-cnv`** (rejected: `error: invalid
argument`), and both `--chat-template ""` and `--no-skip-chat-parsing --single-turn` still apply the template
(verified 2026-10-24, tiny prompt, output identical in form to the templated run). `prototype/mtp_llama/llama_mtp_ab.sh`
therefore decodes **template-wrapped thinking prose**, and a `NOCNV=1` hook was added there so the attempt is
recorded rather than repeated.

That meant llama's ladder (0.750 at k=1, long shape) and ours (0.4713 on `tmp/llama_input/ids_long.csv`) were
never the same task: stereotyped prose is easier to draft, and the caveat was already written in
`LLAMA-MTP-RESULT.md` ("its accept ladder is a little flattered by stereotyped output") but never priced. Since
llama cannot be brought down to a raw continuation, the text was brought **up** to llama's form:
`prototype/mtp_accept/ids_from_chat.py` renders the identical bytes as one user turn through
`tools/chat_render.render_transcript` — the same function `tools/reference_dump.py` uses for the pinned
fixtures, so the template is the one the pins were made with — and tokenizes to `--ids-file` CSV. The render
comes out at **4,152 ids**, which is exactly llama's own prompt token count for that file.

## The measured comparison (4,152 lanes, greedy, 256 rounds, main agent, lock held)

| prompt form | our `a` (loop) | our `a` (`--mtp-dense 1 --mtp-prompt-lanes 146`) | llama.cpp `a` |
|---|---|---|---|
| raw continuation (`ids_long.csv`) | 0.4713 (82/174) | 0.5329 (89/167) | not obtainable |
| **chat template + thinking** (`ids_long_chat.csv`) | **0.6784** (173/255) | **0.7349** (183/249) | **0.750** |

So the accept gap was **the prompt form, not the draft block**. On the text llama.cpp actually decoded, our
draft proposes the target's next token at 0.7349 against llama's 0.750 — a 0.015 gap, i.e. parity within the
noise of 249 rounds. Both knobs that this session built (`--mtp-dense`, `--mtp-prompt-lanes`) are what carry our
raw-text 0.4713 up to that: +0.057 accept total, and they are still default-off.

## What that leaves as the real gap

On llama's own text at its own width, at the same time:

| | base tok/s | MTP tok/s | vs base | round/step | `a` |
|---|---|---|---|---|---|
| llama.cpp `mtp1` | 92.0 | 100.797 | **+9.5 %** | 1.579 | 0.750 |
| ours, `dense+fan146` | 90.671 | **91.595** | **+1.02 %** | **1.670** | 0.7349 |

Acceptance is equal and the round cost is not: **18.42 ms/round against llama's 17.16 ms**, on base steps that
are already within 1.5 % of each other (11.029 vs 10.865 ms). Feeding llama's round cost our measured accept
gives `(1+0.7349)/1.579 = +1.035`, i.e. **+9.9 %** — the same place llama.cpp sits. The remaining ~1.3 ms is
inside our round body, not in the draft (our draft is 3.61 ms vs its 3.423 ms) and not in bookkeeping (`B1`
measured all of it at 0.609 ms and was killed on its own line).

Two honest caveats on the wall row: the A/B harness runs `--steps 32`, so its own accept is a handful of rounds
and noisier than the 249-round `a` above, and `+1.02 %` is a median of 5 reps in one window, not a certified
adopt measurement. And this is now the *templated* population — a real chat request, so it is arguably the fair
reporting width, but it is not the same input as the raw-id rows in `CORPUS-WALL.md`, and the two must not be
averaged.

## Consequence for the plan

The funded question was "why is our accept 0.47 when llama gets 0.75?" The answer is that the two were measured
on different prose. On matched prose there is no accept gap, so **no further draft-conditioning work is
funded**, and the only thing between us and llama.cpp's +9.5 % is 1.3 ms in the round body — which `B1`/`B2`
already priced the pieces of and came up short on. That is the number to attack next, with the meter, not the
draft.
