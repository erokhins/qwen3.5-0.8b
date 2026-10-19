# Item L — lifting the 4,608-lane wall: change, memory arithmetic, and the long-context check

Worker task (queue item L, `progress/2026-10-19/queue2-2026-10-19.md`). **No timing appears in this
file**: every device run here was correctness-only, and no millisecond figure was taken or is quoted.

## 0. The geometry as read from the source, not from the brief

| quantity | value | read from |
|---|---|---|
| lanes one block stages (`MAXSC`) | 512 | `prototype/attn_decode/attn_decode_kernels.h` (`#define MAXSC`) |
| max lane blocks per query row (`kAttnMaxSplit`) | **was 9, now 64** | `engine/kernels_attn.cpp` |
| lane ceiling `MAXSC × kAttnMaxSplit` | was 4,608 → **now 32,768** | both files; `engine/gate.py::lane_ceiling()` READS both |
| wave rows (`kMaxWaveRows`) | 512 | `engine/decode_step.h` (`gate.wave_row_ceiling()`) |
| partial slot | `(DH + 2)` F32 = 258 × 4 = 1,032 B | the writers' own `* (DH + 2)` |
| partial floats per token | `N_Q(8) × split × (DH+2)(258)` | `attn_part_floats()` |
| **partial bytes per token** | `8 × 258 × 4 × split` = **8,256 B × split** | arithmetic on the row above |
| state, constant per request | 19.2656 MiB (GDN 18.0 + conv 1.266) | `engine/state_sizes.py` (PASS) |
| state, per token | 12,288 B (+2,048 B/token MTP while drafting) | `engine/state.h` §00.5/§08.1 |

**Two stale figures in the brief are corrected here.** The brief and
`progress/2026-10-19/queue2-2026-10-19.md` quote **64.5 KiB/token at split 9** and **32.25 MiB per
512-row wave**; those are the *split-8* leftovers of the 8→9 raise (`8×8×258×4 = 66,048 B = 64.5 KiB`).
At the shipped split 9 the numbers are `8×9×258×4 = 74,304 B = 72.5625 KiB/token` and
`512 × 74,304 = 38,043,648 B = 36.28 MiB`. The queue's split-64 row ("528 KiB/token = 264 MiB") mixes
1000- and 1024-based units; exact is **528,384 B/token = 516 KiB** and **270,532,608 B = 258 MiB** per
scratch. `engine/kernels_attn.cpp`'s own comment carried the correct 38,043,648 B and has been updated
to the new constant; the wrong ones live in `progress/` (frozen by the standing rule) and are left alone.

## 1. The memory curve, with the arithmetic shown

Required split for a prompt of `L` tokens at `wave_rows = 512`: the last body's widest row has
`lanes_last = L`, so `split = ceil(L / 512)`. Partial scratch per scratch instance is
`attn_part_floats(512) = 512 × 8 × split × 258 × 4 B = 4,227,072 B × split`.

| lanes | split | partial B/token | partial B / scratch | MiB / scratch | MiB × 8 concurrent | state at that cap (19.27 MiB + 12 KiB/token) | scratch + state, 1 req | scratch + state, 8 req |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,608 (was) | 9 | 74,304 | 38,043,648 | 36.28 | 290.3 | 73.29 MiB | 109.6 MiB | 0.85 GiB |
| 8,192 | 16 | 132,096 | 67,633,152 | 64.50 | 516.0 | 115.27 MiB | 179.8 MiB | 1.40 GiB |
| 16,384 | 32 | 264,192 | 135,266,304 | 129.00 | 1,032.0 | 211.27 MiB | 340.3 MiB | 2.66 GiB |
| **32,768 (now)** | **64** | 528,384 | 270,532,608 | **258.00** | 2,064.0 | 403.27 MiB = 422,854,656 B | 661.3 MiB | 5.17 GiB |

Against the measured ~34 GiB device pool and a 1.5457 GB checkpoint, a **single** request at 32,768
lanes costs ~661 MiB of state+partials (~2 % of the pool) and eight of them ~5.2 GiB (~15 %). So the
raise is affordable, and the ceiling chosen is the one decision L2 asked the check to exercise:
**32,768 lanes, `kAttnMaxSplit = 64`**.

**The cost is per scratch instance, not per token, and that is the honest caveat.** `allocate_scratch()`
calls `attn_part_floats(wave_rows)` and does not know the run's cap, so raising the constant charges
**258 MiB of partial scratch to every scratch instance whatever the request's length** — a 20-token
chat pays it as fully as a 32k prompt. Today `decode_run` allocates exactly one scratch per run, so
nothing breaks and nothing regresses; the charge lands the day a concurrent server shares the
allocator, where a 100-request server would hold ~25.2 GiB of `a_part` alone (today: 3.5 GiB). The fix
is to size by `split_for(cap)` rather than by the constant — a **separate decision with its own memory
argument, deliberately not taken here**, because one constant that `gate.py::lane_ceiling()` reads is
what keeps the allocation and the refusal from disagreeing (§20.1's aliasing class). Flagged as a
follow-up needing a call, not folded into this commit.

Checked vs arithmetic: the state column is `engine/state_sizes.py`'s shape arithmetic (it self-checks
against `state.h` and prints PASS), the scratch column is `attn_part_floats()`, and the concurrency
columns are multiplication. No device run produced any number in that table, and none of them is a
timing. What the raise also costs, beyond bytes: partial traffic — pass 2 reads back `rows × 8 × split`
slots per streaming wave per attention layer, so streaming-row traffic grows linearly with the split
for rows that need it. Item L3 is "enable, not optimise"; no timing was taken to price it.

## 2. The change (one topic)

- `engine/kernels_attn.cpp` — `kAttnMaxSplit` 9 → **64**, with its comment rewritten to state the new
  boundary, the byte cost, the per-scratch charging, and the fact that **no pinned fixture reaches past
  4,104 lanes** (§17.1 stands), so the whole band 4,608→32,768 is funded but unpinned.
  `attn_part_floats()` needed no code change: it already multiplies by `kAttnMaxSplit`, which is the
  coupling L1b asked for; its comment's numbers are updated to the new split.
- `engine/decode_step.cpp` — the `a_part` comment's stale "33.8 MB at the full ceiling instead of
  64.5 KiB" → 258 MB vs 516 KiB.
- `engine/decode_run.cpp` — `kSessionDefaultCap`'s comment no longer restates "512 x 9 = 4608"; it now
  *names* the two constants and the gate that reads them, so the number lives in one place again.
- `tools/attn_ceiling_proof.py` — retargeted from the 8→9 raise to this one: `CAP_OLD, CAP_NEW,
  CAP_TARGET = 9, 64, 128`, its boundary expectations and print strings derived from those constants
  instead of hardcoded 4,608/20,480 (20,480 lanes is **legal** now, so the old expectation was false).
  No check was removed and no must-catch injection was weakened — [A]/[B1]/[B2]/[B3]/[C]/[D] all still
  run and all still catch their injections. It self-reports staleness if the constant moves again.
- `tools/longctx_check.sh` — new (see §4), plus this report.

`engine/gate.py` was **not** touched: `lane_ceiling()` reads both constants, so the gate follows the
raise by construction. Proof, from `gate.py --dry-run` after the change:

```
[PASS] the planned run fits the attention stage's lane ceiling — 17 prompt + 8 generated = 25 lanes vs
staging MAXSC 512 x kAttnMaxSplit 64 = 32768 lanes (…; above it the launcher throws rather than
truncating)
```

## 3. Index-arithmetic audit (what had to be true for the raise to be safe)

Every write site and the reader index the partial buffer through ONE expression —
`a.part + (((size_t)row * N_Q + head) * a.split + blockIdx.y) * (DH + 2)`, at
`attn_decode_kernels.h:396` (`attn_core`), `:744` (`attn_tile_body`), `engine/attn_slim_kernels.h:345`
(`attn_slim_wave`), `engine/attn_fattn_kernels.h:358` (`fattn_part_slot`), read back at
`attn_decode_kernels.h:767` with a `(DH + 2)` stride, mirrored in `prototype/attn_tile_model.py`'s
`part_offset`. Since `a.split ≤ kAttnMaxSplit` is enforced by the launcher's throw and the buffer is
sized by `rows × N_Q × kAttnMaxSplit × (DH+2)`, raising the constant cannot overrun an address, and no
address reads the constant at all. That is not a claim in this file — it is `attn_ceiling_proof.py`'s
[A]/[B1]/[B2]/[B3], re-run after the change and reported PASS below. Everything else that scales with
the cap already casts to `size_t` before multiplying (10 cache-index sites in
`attn_decode_kernels.h` / `attn_slim_kernels.h` / `attn_fattn_kernels.h`, and `decode_run.cpp`'s
`src_stride`), so no int32 overflow opens at 32,768: the widest per-request buffer is one KV cache
pair, `2 × cap × DH` = 201 MB, two orders below 2^31 bytes. `tile_part_floats()` (prototype hosts) is
an `int` product whose largest value at the new ceiling is 67,633,152 floats — inside int32, audited
rather than assumed. Grid dimensions at split 64 stay far inside their limits
(`attn_slim_wave` at most `2 × 64 × 512` blocks).

## 4. `tools/longctx_check.sh` — what it checks, and what one width costs

Per width `W` it runs the engine twice (both correctness-only, `--steps 0` / `--steps 1`, one rep, no
`--time-*` flag) and then judges five things:

| check | what it asserts | how it could be wrong without it |
|---|---|---|
| (a) finish | both runs exit 0, print `RESULT`, no throw/fault marker | a device fault on a rows>1-only path presents as a hang, not a crash |
| (b) logits | the priced row is `kVocab` floats, all finite, `distinct > 1000` | an all-zeros row is "finite" and passes a finite-only check |
| (c) plan | the untimed `LAUNCHSTAT family=attn` counters equal `gate.launch_plan(24,18,6)['attn'] × bodies` via `gate.wave_expect()`; prints `plan_match=1` | a refactor that loops the wave body per token changes no output bit |
| (d) no clamp | `PREFILL tokens=W prefix_len=W wave_rows=512`, `STATE cap=W`, `kv == per_token × cap`, priced row index `== W-1` | a silently clamped width would look like a pass at a shorter context |
| (e) prefix consistency | row `W-1` from the wave route vs row `W-1` reached as one replayed step off the truncated prompt, judged by `engine/attn_form_band.py` run verbatim (bands from `engine/gate.py`) | a mis-indexed partial is not a missing row, it is a row with the wrong answer in it |

Why two deep runs per width: with `--prefill-logits last` only a run's **last** row is priced, so the
only cheap way to see row `W-1` twice is to reach it a second way — a one-token-shorter prompt plus one
replayed token (`--steps 1 --replay`, teacher-forced, so the two runs share their whole token history).
`--prefill-logits all` is the alternative and is unusable here: it holds `rows × kVocab × 4 B` of host
logits, ≈ 7.6 GiB at 8,192 and ≈ 30.5 GiB at 32,768. The prefix side therefore also prices its own last
prefill row (row `W-2`); the comparison judges row `W-1` only, through a pair of symlink directories,
and the original dumps are never moved or deleted.

Two real defects in the tool were found by running it against real dumps and are fixed: the differing
file set (which `attn_form_band.py` correctly reads as a crashed run), and a `sampled.txt` column
difference at row `W-2` that is a coverage artifact of the two invocations, not a numerics move.

`--selftest` runs with **no engine and no GPU**: it writes synthetic fixtures for a 1,024-lane pair,
requires the driver to read PASS, then injects four defects (clamped `STATE cap`, a NaN logit, a wrong
`attn` launch count, a diverged prefix row) and requires the driver to read FAIL and exit nonzero. Run
after the fixes: `SELFTEST PASS — the driver turns green on a clean pair and red on every injected
defect`, exit 0. Without this, per this file's own trap table, "a gate driver whose parser silently
matches nothing reports PASS".

Command for the widths that matter (main agent, exclusive window, blocking, `gpu_lock` held — this
script does **not** self-lock, so holding the lock around it is correct and cannot deadlock):

```sh
tools/gpu_lock.sh acquire longctx-L
tools/longctx_check.sh --widths 8192,16384,32768 --work tmp/longctx   # add --with-refusal to also
tools/gpu_lock.sh release longctx-L                                   # prove ceiling+1 still refuses
```

Cost, derived not measured: two deep prefills per width, so the 32,768-lane width dominates — quadratic
attention makes a 32k prefill minutes, and the whole script is a main-agent run, not a worker one. The
refusal leg is opt-in precisely because a refusal at `ceiling+1` happens in the last wave, so it costs
one full deep prefill to observe.

## 5. What was actually run here (and by whom)

| command | result |
|---|---|
| `engine/build.sh` | built `loader_test` + `decode_run`, 17.3 s, no static assert fired |
| `.venv/bin/python tools/attn_ceiling_proof.py` | **PROOF PASS** with the retargeted constants: readership census `allocation=1, declaration=1, refusal=1, message=2, comment=8`, address expressions reading the cap = 0; one AST across three writers + mirror; 126/126 streaming plans writer==reader, §20.1 aliasing caught 3/3; 8,064 slot evaluations with 0 offsets moving with the cap, `* kAttnMaxSplit` injection caught; boundary 32,768 legal / 32,769 refused / 65,536 refused; sizing 38,043,648 → 270,532,608 B |
| `.venv/bin/python engine/gate.py --dry-run` | **ENGINE GATE PASS**, ceiling line reads `512 x kAttnMaxSplit 64 = 32768 lanes` (L1b's re-read) |
| `.venv/bin/python engine/state_sizes.py` | STATE SIZE CHECK PASS (source of the state column) |
| `tools/longctx_check.sh --selftest --work tmp/longctx_selftest` | **SELFTEST PASS** (clean 0, injected nonzero), no GPU |
| `tools/gpu_lock.sh acquire worker-longctx-1024; tools/longctx_check.sh --widths 1024 --work tmp/longctx_devcheck; tools/gpu_lock.sh release worker-longctx-1024` | device gate, correctness only, **4 s wall**, `gpu_lock util` **90 % before / 10 % after**, **no timing taken**. All five checks pass on real dumps at 1,024 lanes: row 1023 is a `streaming` row (split 2), `plan_match=1` both sides, `STATE cap=1024 kv=12,582,912`, and `attn_form_band.py` says "every judged row is inside §07.3's band … no argmax moved" (1−cos 1.166e-04 against `row_1cos_max` 2e-3; pooled meanrel 1.799e-02 against 3e-2) |

The `speccheck/run_all.py` suites were **not** re-run: nothing in `speccheck/` or `engine/gate.py` was
touched, and the standing rule is to budget verification to what a change can reach. `bench/` was not
touched at all, per the line-citation rule and because a worker's device number may not enter the ledger.
Nothing is committed: four modified files and two new ones, left for review.

**Why 8,192 / 16,384 / 32,768 were not run here**: two deep prefills each is far past the ~60 s a worker
may hold the device for, and 32,768 lanes is minutes of GPU. Those three widths are the main agent's
run; the command is above. The 1,024-lane run does exercise the streaming path (lanes > MAXSC ⇒ split 2
⇒ partials written and merged), so the code path is not untested — only its depth is.

## 6. What this check cannot see

- **Depth vs reference.** There are no pinned dumps past 4,104 lanes, so (e) compares two engine routes
  against each other, not against HF. An error shared by the wave route and the one-row route is
  invisible; only a pin at the new depth would catch that, and making one costs a fresh HF run.
- **Interior rows.** Only the deepest row is priced per run. A per-row index bug that leaves the deepest
  row's own slots intact but corrupts rows `W-2 … W-512` would pass. Fixing that costs
  `--prefill-logits all` (≈ 30.5 GiB host at 32k) or a row-selection flag the engine does not have.
- **`split` itself.** Nothing prints the split. A clean finish at `W` lanes is the witness that
  `ceil(W/MAXSC) ≤ kAttnMaxSplit`, because above the ceiling the launcher throws (that is what
  `--with-refusal` proves from the other side).
- **Performance.** No timing, so the extra partial traffic and any quadratic-in-split cost are unpriced
  here; per L3 this is "enable, not optimise", and the ledger rows are the main agent's.
- **Server-shaped memory.** The 258 MB-per-scratch charge is per scratch instance; the check runs one
  scratch at a time and cannot see the concurrency consequence in §1's last paragraph.
- **Long generation.** `--steps 0/1` only, so nothing about the §07.5 stop path, the session REPL, or a
  free run at depth is judged here.

## 7. Residual risks / open items for the reviewer

1. Per-scratch 258 MB (vs 36.28 MB) charged regardless of request length — affordable for one request
   and for eight at 32k, not for a hypothetical 100-request server. Cap-aware sizing is the fix and is a
   decision, not a patch.
2. The band between 4,608 and 32,768 lanes has **no pinned reference**; the new widths are gated for
   self-consistency, not for agreement with HF.
3. The device gate ran while `util` read 90 %, i.e. not a provably exclusive window. Fine for a
   deterministic correctness gate (no atomics in these kernels), and no timing was taken or is quoted.
4. `--with-refusal` was not run here (it costs a full deep prefill); the CPU boundary proof covers the
   32,769 refusal by arithmetic, and the device run of it is a one-flag addition for the main agent.
5. Uncommitted: `engine/kernels_attn.cpp`, `engine/decode_step.cpp`, `engine/decode_run.cpp`,
   `tools/attn_ceiling_proof.py`, `tools/longctx_check.sh`, this file. One topic: raise the attention
   lane ceiling to 32,768 lanes, with the tool and the retargeted CPU proof that follow it.

## 5. The device pass, run by the main agent, and the three harness bugs it found

The five checks passed at 8,192 / 16,384 / 32,768 lanes and 32,769 was refused naming `kAttnMaxSplit=64`
(ledger **§39** carries the table and the cost curve). Getting there took three fixes to *this* check and one
to the engine's argument parsing, and none of them was visible by reading the script:

1. **`--ids` cannot carry a 32,768-token prompt.** Linux caps a single argv slot at `MAX_ARG_STRLEN = 131,072 B`
   regardless of `getconf ARG_MAX` (2 MiB, which misleads everyone); 32,768 ids is 191,588 B, so `exec` returned
   E2BIG — shell **exit 126**, "Argument list too long", in 0 s — which presented as "the engine crashes at
   32k". `decode_run` gained `--ids-file PATH` (same parser, same list) and the check uses it. 16,384 ids is
   95,796 B, which is why the same bug let every narrower width through.
2. **The refusal case was vacuous.** The id pool was generated to `max(requested widths)`, and the refusal
   width is `ceiling + 1`, so it was fed 32,768 ids, ran a *legal* prompt, and reported "the ceiling moved
   somewhere other than the launcher's refusal" — a must-catch case that could not fire, the `partalias`
   lesson one layer over. Pool now covers `max(widths, ceiling+1)`, **and** `ids_file_of` asserts the field
   count it wrote, because a silently short prompt is the actual failure mode.
3. **`--selftest` and a real run shared one work directory.** The selftest writes a synthetic 1,024-id pool
   into `$WORK/ids.csv`; a real run cutting its ids from the same file after a selftest is fed **1,024 ids**
   while claiming a wider width. The first 8k/16k timing set taken against these logs was three reps of a
   1,024-token run wearing an 8,192 label — every field in its `RESULT` line looked plausible. Discarded and
   re-taken from a pool with its field count asserted. A selftest that writes into the directory real runs
   read from is the trap, not the timing.

**No timing in this file.** The §39 cost curve is the main agent's, taken in its own locked window; nothing
here quotes a millisecond figure.
