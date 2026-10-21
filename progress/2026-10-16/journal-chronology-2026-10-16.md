# The journal chronology — the chronicle and retired sections of `engine/results.md`, quoted

Written 2026-10-16, CPU only: no device command, no build, no gate, no timing, no
`tools/gpu_lock.sh`. This is step **S5** of `progress/2026-10-16/archive-review-2026-10-16.md`
§5 — option (a′), the extraction **by quotation** — and it is one of the two things that step
adds, the other being the 5-line pointer appended at the very end of the journal. This file
makes no classification of its own: every class call and every evidence line below is quoted
from that archive review's §1, which is where the calls were made and argued.

**The rule this file obeys is that the journal keeps every line.** Not one line of
`engine/results.md` has been deleted, moved, re-wrapped or re-flowed to produce it; the only
change to the journal is 5 lines appended at its tail. The reason is arithmetic, not taste: the
census counted **96 of its 101 line-form citations living in `progress/`** — the tree rule 2 of
`progress/README.md` freezes to same-line path fixes — and **51 of the 101 address line numbers
in this journal**; the first chronicle block starts at `:81`, so a deletion there renumbers 64
of the 65 line numbers the archive cites (§4 of that review, and its §2.3 for the origin count).
A copy that can drift is the only extraction that costs nothing, and that is why §4 recommends it.
So: **this file is a copy, made for legibility, and the journal stays the record.**

**Drift is checkable, per block and by machine.** Every block below names the lines it was
taken from, so a reader can confirm any one of them by hand — the range printed above a block
must print that block and nothing else:

```bash
sed -n '299,334p' engine/results.md
```

and all of them at once with:

```bash
python3 tools/citation_census.py --verify-quotes \
    progress/2026-10-16/journal-chronology-2026-10-16.md
python3 tools/citation_census.py --selftest   # the mutation cases that prove it can fail
python3 tools/doc_ref_check.py                # every path and line anchor here resolves
```

`tools/doc_ref_check.py` scans `git ls-files`, so it sees this document only once it is tracked;
until then the same resolvers adjudicate it file by file — which is how its citations were
checked before the commit that makes it scannable: **63 line anchors, 58 distinct paths and 3
section labels, all of them resolving.**

`--verify-quotes` refuses to pass on zero blocks, and `--selftest` carries a case where a quote
has drifted by one word and must be rejected, because a check that cannot fail is not a check.

**Ranges, and the arithmetic the census's totals sentence got wrong, stated plainly.** The §1 class
table publishes
14 SUPERSEDED rows and 13 NARRATIVE rows, and its prose adds 3 H3 blocks (`565-601`, `712-769`,
`1048-1057`). Taken as printed, those 30 ranges span **883 lines**, not the **853** that §1's
totals sentence claimed until it was corrected the same day. The gap is the census's own, not a slip
of transcription here: its totals
line read `14 SUPERSEDED (483)` while its 14 SUPERSEDED rows sum to **513**, and
`59 LIVE-DATA (3,745 lines)` against rows summing to **3,716** — **+30 and −29** between the table and
the sentence that summarises it, while NARRATIVE agrees at 265 and the section counts
(59 / 13 / 14) check out; the sentence now says 513 and 3,716, corrected in place, and the census's
§8 records the correction. The residual ±1 is the missing final newline, which makes the table's
rows partition 4,494 lines and the prose 4,493. This file quotes **the rows**. Of the 883 lines
they span, **848 are quoted below** and **35 are not**: every census row ends on the blank line
that separates a section from the next heading, and a quotation has no use for a separator. Each
marker therefore names the block's last line that carries text, which has a second benefit — both
checkers accept every anchor in this file, where the anchor resolver of `tools/doc_ref_check.py`
rejects a cite that lands on a blank line, and §8 of the archive review counts 27 cites dead that
way in a document about dead line cites.

## The index

Grouped by class — SUPERSEDED, then NARRATIVE, then the three H3 blocks — and in the census's
line order within each group. `census §1 row` is the range as `engine/results.md` is sectioned
there; `quoted` is what follows, trimmed of the trailing separator blank.

| id | class | census §1 row | quoted | lines | the section, as its own heading reads |
|---|---|---|---|---:|---|
| S1 | SUPERSEDED | `109-113` | `109-112` | 4 | W5b on disk: `state.{h,cpp}` + the launcher layer — proven, deferred, open |
| S2 | SUPERSEDED | `185-249` | `185-248` | 64 | W5b's token loop on disk: `decode_step` + `decode_run` + `gate.py` — compiled, none of it run |
| S3 | SUPERSEDED | `250-298` | `250-297` | 48 | Two device runs in: the loop runs, and six failed checks had one cause — §07.5 armed during the prompt |
| S4 | SUPERSEDED | `299-335` | `299-334` | 36 | The first ms/step figure, recorded as *not* a measurement |
| S5 | SUPERSEDED | `336-390` | `336-389` | 54 | The three flagged risks, checked statically — no device, no gate loosened |
| S6 | SUPERSEDED | `1201-1223` | `1201-1221` | 21 | What a worker may not decide, left open on purpose |
| S7 | SUPERSEDED | `1431-1458` | `1431-1456` | 26 | The proposal, with numbers — band untouched, decision belongs to the operator |
| S8 | SUPERSEDED | `1650-1677` | `1650-1676` | 27 | Q1 of `progress/2026-10-12/plan-2026-10-12`: band option 1 is in force — the margin is a measurement, and what that costs |
| S9 | SUPERSEDED | `1678-1711` | `1678-1710` | 33 | The derivation, re-derived from the shipped files |
| S10 | SUPERSEDED | `1712-1739` | `1712-1738` | 27 | Why this is a sharpening and not a loosening — the property that makes it defensible |
| S11 | SUPERSEDED | `1740-1766` | `1740-1765` | 26 | The tripwire this derivation used to carry — retired by D-2, kept here because it is how D-2 was found |
| S12 | SUPERSEDED | `1864-1895` | `1864-1894` | 31 | The tripwire fired on `dec300f`, so nothing was priced and the margin did not move |
| S13 | SUPERSEDED | `1963-2004` | `1963-2003` | 41 | What this pass left open, named |
| S14 | SUPERSEDED | `2562-2621` | `2562-2620` | 59 | The handover: what this pass could not run, and the sequence if a band move is ever decided |
| N1 | NARRATIVE | `81-100` | `81-99` | 19 | The D2H discrepancy: measured 27.7 GB/s vs the doc's 63–70 GB/s |
| N2 | NARRATIVE | `101-108` | `101-107` | 7 | What is deliberately not measured here |
| N3 | NARRATIVE | `130-155` | `130-154` | 25 | Deferred, with the price named |
| N4 | NARRATIVE | `156-184` | `156-183` | 28 | Open questions this phase did not decide |
| N5 | NARRATIVE | `1412-1430` | `1412-1429` | 18 | What this does not settle |
| N6 | NARRATIVE | `1767-1778` | `1767-1777` | 11 | What was not chosen, and why it is on the record |
| N7 | NARRATIVE | `1779-1796` | `1779-1794` | 16 | What the pass does not buy |
| N8 | NARRATIVE | `1815-1843` | `1815-1842` | 28 | The pre-flight, and making them |
| N9 | NARRATIVE | `2154-2184` | `2154-2183` | 30 | The pass, and what a worker ran |
| N10 | NARRATIVE | `2185-2203` | `2185-2202` | 18 | What this pass left open, named |
| N11 | NARRATIVE | `2316-2339` | `2316-2338` | 23 | What this pass ran, and how to extend it |
| N12 | NARRATIVE | `2340-2356` | `2340-2355` | 16 | What this left open, named |
| N13 | NARRATIVE | `2549-2561` | `2549-2560` | 12 | What this pass ran, and what was left for the device pass |
| H1 | H3 block | `565-601` | `565-600` | 36 | Levers, ranked by expected ms/token |
| H2 | H3 block | `712-769` | `712-768` | 57 | Commands for the main agent, in the order that changes strategy |
| H3 | H3 block | `1048-1057` | `1048-1054` | 7 | Context, not action: there is no prefill |

The `census §1 row` column is written as a bare range, not a `path:NNN` cite: a range that
ends on a separator blank would be a broken anchor in the file whose whole point is that its
anchors resolve.

## SUPERSEDED — 14 blocks: the text itself says it was retired, refuted, corrected or replaced

The class key is the census's, quoted from §1: *SUPERSEDED — the text itself says it was
retired, refuted, corrected or replaced.* Each block's own §1 evidence line is quoted under
`Why it is in here`, with the line number the census cites as its witness.

### S1. `engine/results.md:109-112` — W5b on disk: `state.{h,cpp}` + the launcher layer — proven, deferred, open

Why it is in here — the census's §1 evidence for the row `109-113`: the census's `:111` —
"Everything in this section is **compiled, never run on a device**." — disproved from
`:250`.

quoted from `engine/results.md:109-112`

````
# W5b on disk: `state.{h,cpp}` + the launcher layer — proven, deferred, open

Everything in this section is **compiled, never run on a device**. No allocation has happened, so
no number here is a measurement of anything except the compiler's willingness.
````

### S2. `engine/results.md:185-248` — W5b's token loop on disk: `decode_step` + `decode_run` + `gate.py` — compiled, none of it run

Why it is in here — the census's §1 evidence for the row `185-249`: the H1 at `:185` —
"…`decode_step` + `decode_run` + `gate.py` — compiled, none of it run".

quoted from `engine/results.md:185-248`

````
# W5b's token loop on disk: `decode_step` + `decode_run` + `gate.py` — compiled, none of it run

2026-10-11. `decode_step.{h,cpp}`, `decode_run.cpp`, `gate.py`, and the four launcher TUs that were
missing (`kernels_cast`, `kernels_mlp`, `kernels_attn`, `kernels_lmhead`). All twelve TUs compile
clean at `-O3` and `engine/build.sh` links `engine/build/{loader_test,decode_run}`. **No engine
binary has been executed**, so nothing in this section is a measurement.

What is settled without a device:

* `check_weights(store)` pins every shape and §01.6 dtype this loop depends to the checkpoint it
  actually loaded — including `lm_head.weight` being the embedding blob rather than a second 485 MiB.
  **This bullet used to claim more than it had earned.** It listed `dt_bias`/`conv1d.weight` as pinned
  BF16, and the first device run proved the `conv1d.weight` entry wrong: the table claimed rank 2
  [6144,4] and the shard holds rank 3 [6144,1,4] (it is one of the checkpoint's 18 Conv1d weights —
  every rank-3 tensor in the file). The store was right, the claim was a guess, and the type system
  made the guess inevitable: `Expected` was `{d0, d1, dtype}` with `d1 == 0` meaning rank 1, so rank 3
  was inexpressible. Fixed in 92d8e5f — `Expected` now carries `rank` explicitly, and the
  [6144,1,4] → flat [6144][4] collapse that `conv_kernel`'s `C[c*K + j]` wants is asserted in
  `prepare_weight_adapters`, where the flattening is spent, not in the checker. `dt_bias` was the
  other guess and was already right: BF16 rank-1 [16].
* `engine/weights_contract.py` is why that class of bug is over. It reads the shard header, `state.h`
  and the `Expected` table as three independent sources and fails on any disagreement — 320 names
  clean, plus the rank/F32/§03.5-18-6 census — and it bites when handed a mutated copy (rank-2 conv,
  dt_bias as [16,1], A_log as BF16 each FAIL at their own row alone). Shape claims are now checked on
  CPU, which is the only place they can be checked cheaply; the alternative was one GPU window per
  wrong rank.
* `gate.py --no-engine` proves the `dec8` fixture pairs with itself — 26 assertions, including that
  the bare `logits_last.npy` is the prefill (argmax 760) and the bare `input_ids.npy` is decode7's —
  and that `state_sizes.py`'s bytes are the bytes `decode_run` will report. That half is green.
* The block plan is agreed between two independent implementations: `gate.py` counts one token's
  launches from §07 in Python, `planned_launch_counts()` derives the same numbers from §00's
  constants in C++, and the gate fails on any mismatch. glue 49 · proj 114 · gdn 72 · attn 6 ·
  mlp 48 · cast 115 · other 20 = **424**.

What a first device run is most likely to break, in the order to look at them:

0. **The weight table.** It already broke once, before a single kernel launched: `check_weights`
   aborted on `conv1d.weight`'s rank. `engine/weights_contract.py` now retires that failure mode on
   CPU, so if it reappears the shard changed rather than the claim.
1. **The KV append inside the fused attention launch.** `stage_head()` gates the §05.7 cache write on
   `q_rope != nullptr` — the same null check as the three stage dumps — so `kernels_attn.cpp` refuses
   null stage pointers rather than silently skipping the append. If the fused path turns out not to
   append at all, every step after the prompt reads empty KV and the first wrong token appears at
   decode step 0 with the prompt steps also wrong.
2. **The q/k/v slice order handed to `rec_kernel`.** `mixed_qkv` is [q|k|v] in 2048-wide blocks
   (§04.1) and `decode_step` passes `g_u`, `g_u+2048`, `g_u+4096`. A k/v transposition shows up as a
   prompt that samples correctly and a chain that diverges at step 1.
3. **The conv-window roll-back.** `conv_kernel` writes the rolled window to scratch and the layer's
   window is updated by one D2D copy; if the copy's size or the slot indexing is off, the drift is
   gradual and shows up several tokens in, not immediately.
4. **The logits band.** `cos ≥ 0.9995` is §10.1 G5's published figure for incremental-vs-batch
   logits, not a number measured against this engine. If the ids come out exactly right and the band
   fails, that is a policy question about which §10 figure governs a token-level logits vector —
   report it, do not loosen the constant. `ndiff` and max-ULP are already excluded on principle
   (§10.1 F1–F4).

* **Fixture: `dec8`, not the `prefix24dec8` `progress/2026-10-15/build-queue.md` W5 proposed.** `dec8` is what the
  task named, it is the fixture whose per-step dumps (`decode{s}.input_ids`,
  `decode{s}.logits_last`) make the pairing provable, and its prompt is 17 tokens rather than 24 — so
  the attention stages run at ≤25 lanes against `MAXSC` 512. Same assertions would run against
  `prefix24dec8`; `gate.py` hard-codes the `dec8` path and nothing else, so that is a second call, not
  a flag.

---
````

### S3. `engine/results.md:250-297` — Two device runs in: the loop runs, and six failed checks had one cause — §07.5 armed during the prompt

Why it is in here — the census's §1 evidence for the row `250-298`: the H1 at `:250` — "six
failed checks had one cause — §07.5 armed during the prompt" (fixed in `5e7114b`, named
there).

quoted from `engine/results.md:250-297`

````
# Two device runs in: the loop runs, and six failed checks had one cause — §07.5 armed during the prompt

2026-10-11, two operator-run foreground device passes (workers do not touch the GPU, `AGENTS.md`).
Nothing here is a timing except where it explicitly says it is not one.

**What the runs established.** The loader loads: 335 tensors, 1.5457 GB resident, ≈1.0 s.
`check_weights: PASS` — the rank-3 `conv1d.weight` claim that aborted the very first attempt is closed
by `92d8e5f` and `engine/weights_contract.py`, which is the CPU-side reason it cannot come back.
The `state:` and `LAUNCH` lines verify against §00/§07 item by item (gate sections (c) and (d) pass),
so the 424-launch plan and the per-request byte counts are agreed between the two independent readings
and the binary. Risk 0 above ("the weight table") is retired, and so is the part of the hand-off that
said no engine code had ever touched the device.

**The six failures, and they were one bug — a phase bug, not numerics.** The run log:

    prompt 6 | fed 9338 | sampled 248046 | top 19.3246193 | P 6 -> 7 [STOP]
    stopped at step 6 (§07.5: id 248046)
    - one step per prompt token: 7 vs 17
    - 8 generation steps: 0
    - all 8 generated ids match, in order: engine [] / pinned [6511, 314, 9338, 369, 2972, 57590, 159034, 271]

`decode_run` applied §07.5's stop check to *every* step, prompt steps included. The fixture prompt is
the manifest's `sources[0].ids` — a rendered chat template — so it legitimately contains the
end-of-turn id **248046 at index 8** (and the think-start/end ids 248068/248069 later); at prompt index
6 the engine's own argmax happened to be 248046, which ended a 17-token prompt after 7 steps. During
the prompt phase the sample is discarded and the next token fed comes from the fixture, so a stop match
there is meaningless. Fixed in `5e7114b`: **the check is armed only in the generation phase**; a
prompt-phase match is still printed, with its own marker, and the `STEP` line gained `stop=0|1` meaning
"the §07.5 check fired" rather than "the sample equals `kStopToken`". Two consequences worth keeping in
mind: `st.stopped` is decode_step's raw match (either phase) while `RESULT stopped=` is the run's
decision, and **no claim about the numerics follows from either run** — nothing was generated, so the
eight pinned ids are still neither matched nor contradicted.

One diagnostic the runs did give for free: the prompt path runs end to end and its samples are ordinary
token ids, not zeros or NaN — `[846, 198, 3710, 2614, 314, 279, 248046]` at top logits 14.70–23.07
(`progress/engine/evidence/engine_gate/sampled.txt`). They are *not* the template's own next tokens (steps 2, 3, 5, 6 are
not), which is expected twice over: the reference prefills the prompt as one chunked forward while the
engine feeds it token-at-a-time through the recurrence (§04.7 vs §04.4, and the README's "What this is
NOT yet"), and a model is free to want something else mid-template. A gate-side note while reading the
log: section (b) can pass on **stale** `logits*.bin` if a `--dry-run` wrote that work dir earlier — the
first run stopped at index 6, so it wrote `logits0..6` only, and its six failures were all in section
(a); `gate.py` now clears the dir so a missing dump reads as missing.

**And the gate is now written to name this bug itself** (`b7b6766` and the stale-dump follow-up):
"one step per prompt token" reports the last STEP's kind/index/sample/stop plus the RESULT line and
says which of the two causes it is; a new check fails if any prompt-phase step fires §07.5;
`--selftest`'s fourth case replays the truncation (`--break-prompt-stop 6`) and requires both checks to
bite at that step. `SELFTEST PASS` (clean dry run + four injections), CPU-only.
````

### S4. `engine/results.md:299-334` — The first ms/step figure, recorded as *not* a measurement

Why it is in here — the census's §1 evidence for the row `299-335`: the census's `:301` —
"**Superseded the same day, twice over.**".

quoted from `engine/results.md:299-334`

````
## The first ms/step figure, recorded as *not* a measurement

**Superseded the same day, twice over.** The `14.631 ms/step` below stayed a non-measurement — it did,
and `progress/2026-10-16/performance-status-2026-10-16.md`:269 (frozen ledger §6) carried the real row at **11.978 / 12.053 / 12.232**.
And item 1 below named HIP graphs as the first perf item, which the measured dispatch curve refutes:
§1's probe replayed 64 captured launches at 1.78 µs against 1.85 µs eager, so a one-token graph buys
≈4 % of the dispatch bill (≈0.06 ms) and not the gap. Both corrections live in
"the first engine throughput row", at the end of this file.

The second run printed `RESULT … ms_per_step=14.631`. **This is an observation, not a number.** It is a
gate run's by-product, which `progress/2026-10-16/performance-status-2026-10-16.md`:244 (frozen ledger §6) refuses as a measurement; it is one
unwarmed first pass over freshly allocated state and scratch, i.e. the first-GPU-touch regime
`devices/strix-halo-unified-memory.md` measures at ~35 GB/s instead of ~240; there is no burst and no
best-of-N; and no `util` was recorded for the window. Do not quote it and do not diff it against
anything as if it were a measurement.

What it is good for is sizing the first perf item, which comes **after** correctness is green. The
ledger's §6 op-sum is **8.03 ms → ~124 tok/s** (7.975 ms before the `gdn_decode` re-take — same six
families, one replaced row), so the unwarmed engine is ~1.8× that. The two most likely causes, in order
of expected size, are both launch-side:

1. **424 launches per token plus one device sync per step.** At the measured 1.85 µs dispatch floor
   (`tools/launch_overhead_hip.cpp`) the launch bill alone is ≈0.78 ms/token, and that assumes a warm,
   deep queue; the step then syncs at the logits, so the host waits on all of it every token. **HIP
   graphs** — capture one token's 424 launches once, replay them — are the tool that removes the
   per-launch cost, and they are now the first perf item, because nothing about correctness still needs
   the launches to be individually visible.
2. The two deferrals named above (packed projections ≈0.10 ms, a BF16-staged MLP to delete 48 of the
   115 casts), which shrink that same bill rather than its overhead.

The honest caveat on the comparison: 8.03 ms is a **sum of per-op cold bursts measured separately**, and
`progress/2026-10-11/commit-review-2026-10-11.md` already graded that sum "overstated as a description of the engine",
so 14.631 − 8.03 is not a measured gap and nobody should defend or attack it as one. The first real
engine timing is a warm burst in a zero-subagent, `gpu_lock.sh`-recorded window, with the same best-of-N
discipline as every other §6 row, and it owes both the ms/token and the tok/s rows the ledger has never
had for the engine.
````

### S5. `engine/results.md:336-389` — The three flagged risks, checked statically — no device, no gate loosened

Why it is in here — the census's §1 evidence for the row `336-390`: the census's `:411` —
"both risks are retired by the ids rather than by reading".

quoted from `engine/results.md:336-389`

````
## The three flagged risks, checked statically — no device, no gate loosened

1. **The q/k/v slice order handed to `rec_kernel`: correct, and worth noting that nothing had ever
   checked it.** `speccheck/specimpl.py:270-272` slices the conv output as `q = u[:, :KEY_DIM]`,
   `k = u[:, KEY_DIM:2*KEY_DIM]`, `v = u[:, 2*KEY_DIM:]` with `KEY_DIM = H_K*D_K = 16×128 = 2048` — the
   same 2048 as `state.h`'s `kValueDim` — so `[q|k|v]` at float offsets 0 / 2048 / 4096, exactly what
   `decode_step.cpp` hands `gdn_rec` (`sc.g_u`, `+kValueDim`, `+2*kValueDim`). Inside a slice
   `rec_kernel` indexes `x[h*D + j]` (`prototype/gdn_decode/gdn_decode_kernels.h:90-120`), i.e.
   head-major with `D = 128`, which matches the reference's `.reshape(T, H_K, D_K)`. Why reading was the
   only way to know: `prototype/gdn_decode`'s own driver fed `rec_kernel` from the **already-sliced**
   pins (`q_post_conv`/`k_post_conv`/`v_post_conv`), so §04's device gate could not have seen a
   transposition in the engine's offsets whatever they were. The first run whose ids come out exact is
   the empirical counterpart of this reading.
2. **The conv-window roll-back: size, roll order and slot indexing all check out.** Reference and kernel
   roll the same way — `new_win = padded[-(K-1):]` (`specimpl.py:260`) is `[win₁, win₂, raw]`, and the
   kernel writes `new_win[j] = win[j+1]` then `new_win[K-2] = raw` (`gdn_decode_kernels.h:82-86`), with
   `K-2` the last valid row so nothing runs past the buffer. The engine's scratch is
   `kConvWindow × kConvChannels` floats (3 × 6144 = 18 432) and the roll-back copy is
   `kConvWindow*kConvChannels*sizeof(float)` = 73 728 B, exactly one layer's stride; and
   `conv_window_layer()` strides by `gdn_slot(layer)` — the *slot*, the same index that selects the
   widened `conv1d.weight`, so the two cannot drift apart by layer (`state.cpp:76-81`). The copy is a
   `hipMemcpyAsync` on the default stream, enqueued between the conv launch and the launches that read
   `g_u`, and the engine uses one stream, so there is no race — a second stream would break it, which is
   worth remembering if a later pass parallelises layers. The rank-3 → flat `[6144][4]` widening of that
   weight is refused unless the rank is exactly what the shard holds (`prepare_weight_adapters`).
3. **The logits band: the citation is the soft spot, not the arithmetic — and the constant stays put.**
   Three separate facts, none of them a gate edit:
   * *What G5's 0.9995 actually is.* §10.1 G5 is "prefill `T` then decode `D` equals one prefill of
     `T+D`" — the reference against **itself**, no engine in the loop. The engine-vs-reference
     comparison this gate makes is strictly harder (a different implementation of every op), and §10's
     published ceiling for a whole-cascade comparison is §10.2's `cos ≥ 0.999, meanrel ≤ 5 %` — whose
     own measured worst cascade cos on `dec8` is **0.99923**, *below* the floor this gate hard-gates.
     So a run with eight exactly-right ids can legitimately land in [0.999, 0.9995) and fail here. When
     it does, the open question is which §10 figure governs engine-vs-reference token logits — an owner
     decision, not this file's and not a constant to edit. `BAND` was not touched.
   * *What a token-level cos can see.* cos over 248 320 entries is dominated by the mass of the
     vocabulary: a 1-ULP-level error profile (bf16 ≈ 0.2 % per element) puts cos at ≈ 1 − σ²/2 ≈
     0.999 998, so the floor is nowhere near binding when the arithmetic is right — while still
     discriminating gross errors (the §10.5 q/gate mis-read measured cos −0.14). What it cannot see is a
     **wrong token**: at a token level the informative statistic is the top-1/top-2 margin against the
     worst element error, which is what `gate.py`'s `metrics()` prints as `gap12` and `margin` next to
     the separate argmax check. When an id is wrong, read those, not the cos.
   * *The other two figures are already in the file.* G6's 0.9999 is reported-not-gated, and speccheck's
     own G6 gate meets it on these same pinned `decode{s}.logits_last.npy`, so the reference side is not
     the risky side; the engine adds what speccheck does not have — bf16 kernels, and a prompt fed
     token-at-a-time through the recurrence rather than the chunked form (§10.1 G3 self-checks
     chunk-vs-recurrent at cos ≥ 0.9999). A first run landing between G5's floor and §10.2's cascade
     ceiling is a reportable finding, not a surprise.

**Hand-off, unchanged and ready:** `./engine/build.sh && .venv/bin/python engine/gate.py`, in a quiescent
window (`tools/gpu_lock.sh acquire`, zero subagents). Sections (0)/(c)/(d) should be green as before,
and (a) will be the first token-level correctness result this repo has had.

---
````

### S6. `engine/results.md:1201-1221` — What a worker may not decide, left open on purpose

Why it is in here — the census's §1 evidence for the row `1201-1223`: the H2 at `:1201` —
"What a worker may not decide, left open on purpose" — decided at `:1459`.

quoted from `engine/results.md:1201-1221`

````
## What a worker may not decide, left open on purpose

1. **Tie policy.** Two of 64 pinned ids are exact ties in the reference. Either they become a named
   exception (proof required from the pin: the engine's id must carry the pinned top-1 logit exactly), or
   `dec64f` stays red at 17/18 and everyone lives with a failing deep gate. Both are defensible; the band
   and the rule are §10's and the operator's.
   → **CLOSED 2026-10-12.** Decided (payload-space predicate, not F32 equality) and implemented; see
   "Decision 1 of the deep gate" at the end of this file. `dec64f` now reports steps 17/18 as TIE and the
   gate is red only on the band half.
2. **Whether the 26–34 excursion needs a bug.** The correlation says scale, not decay, but scale is a
   reading, not an attribution. The measurement that settles it is an engine-side per-step state
   fingerprint (the same four moments the pin carries) printed by `decode_run`, which would make `(e)` a
   two-column comparison instead of a correlation. That is a new engine flag plus a device run, and it is a
   scope decision, so it is written here rather than built.
   → **ANSWERED A DIFFERENT WAY 2026-10-12, and the state fingerprint is still owed.** Pricing the two
   legal reference routes against each other at all 64 steps (`engine/route_price.py --fixture dec64f`)
   puts *them* under G5 at 6 of the engine's 7 failing steps, so no engine bug is needed to explain the
   window — but step 34 is outside the legal spread, and only the engine's own moments can speak to that.
   See "Decision 2 of the deep gate" below for the table, the three band options, and the eos confound.
3. **`MAXSC`.** 512 lanes is fine for everything this repo runs today (64-deep gate, 300-deep smoke, a
   4k-context request is a prefill conversation). Recorded, not touched.
````

### S7. `engine/results.md:1431-1456` — The proposal, with numbers — band untouched, decision belongs to the operator

Why it is in here — the census's §1 evidence for the row `1431-1458`: the H2 at `:1431` —
"The proposal, with numbers — band untouched, decision belongs to the operator" — decided at
`:1650`.

quoted from `engine/results.md:1431-1456`

````
## The proposal, with numbers — band untouched, decision belongs to the operator

The G5/G6 bands were **not** edited, and `engine/gate.py --fixture dec64f` still reports
`ENGINE GATE FAIL — 15 failed`. Options, in the order I would rank them:

1. **Key an allowance on the measured route spread, per fixture and per step.** Gate step j on
   `engine 1−cos ≤ k × route 1−cos(j)` with the route column produced by `engine/route_price.py --fixture …` and
   cached next to the fixture, `k = 4`. dec64f then passes at all 64 steps (worst observed ratio 3.04) while a real
   bug — which would have to be *bigger* than the reference's own legal spread to hide — still fails. Cost: the gate
   gains a CPU preprocessing step per fixture — cheap, 5.3 s wall for the whole 64-step table — and its
   fairness now depends on `specimpl`'s
   own route spread being representative; it is a spread between two forms of *specimpl*, against a pin made by *HF*.
2. **Gate the decision, not the vector**: require per-step argmax agreement (plus the proven-tie exception of
   decision 1), and keep the logits comparison as reported evidence with the §10.2 margin statistic as the check —
   `gap / maxabs >= 1` passes dec64f at every step except steps 17/18, where the pin's own gap is 0 by construction.
   This is the cheapest honest option and it makes the gate say what it actually verifies: the engine picks the
   reference's token. It also gives up any claim on the logits vector's shape, which is today's only defence against
   a wrong-but-argmax-preserving error (e.g. a mis-scaled final norm, which is exactly the class the meanrel column
   is there to catch).
3. **Leave the bands and label the window**: keep 0.9995 hard, and carry a per-fixture list of steps that are known
   to be inside legal spread (dec64f: 27–32 by the route table, 34 not) so the gate prints them as `TIE`-like
   `SPREAD` lines that are reported and counted but do not arm. Zero new arithmetic, one more thing to keep in sync.

Whatever is chosen, the 27–34 rows of `dec64f` are a poor thing to hang a band on, because they are simultaneously
the low-energy window and the post-eos window; the no-crossing fixture (`dec64t`) should exist before the band is
rewritten, and if the dips do not move with the energy curve there, item 1 needs re-measuring rather than adopting.
````

### S8. `engine/results.md:1650-1676` — Q1 of `progress/2026-10-12/plan-2026-10-12`: band option 1 is in force — the margin is a measurement, and what that costs

Why it is in here — the census's §1 evidence for the row `1650-1677`: the census's `:1652` —
"**SUPERSEDED IN PART, same day (D-2).** This section's *rule* stands…".

quoted from `engine/results.md:1650-1676`

````
# Q1 of `progress/2026-10-12/plan-2026-10-12`: band option 1 is in force — the margin is a measurement, and what that costs

> **SUPERSEDED IN PART, same day (D-2).** This section's *rule* stands — judge step *j* against the route
> spread measured at step *j* — but the margin is no longer one global constant: it is now **per fixture**,
> each fixture's file carrying a margin derived from that fixture's own steps. Everything below that says
> "3.50 sets it for everyone" was true when written and is now history; `dec64f` and `dec64t` still carry
> 3.50 / 0.42, `dec300f` is priced at 7.99 / 0.72 and `mt32f` at 2.93 / 0.24, and the reason the global
> constant was retired is in the D-1/D-2 section at the end of this file (which also re-derives every
> number in the table below from the shipped files rather than from memory).

The operator's decision on "Decision 2" above was **option 1**: judge step *j* against the reference's own
measured route spread at step *j*, not against an absolute floor the reference cannot hold against itself.
`5d26bb5` implements it (`engine/gate.py`'s BAND + ROUTE ALLOWANCE blocks, `engine/route_allowance.py`,
the `--engine-log`/`--out` growth in `engine/route_price.py`); `68c4e9d` is the unrelated driver fix from
the same session, noted at the end. **Both deep fixtures now PASS**: `dec64f` with 2 steps admitted as
proved ties, `dec64t` with nothing admitted at all.

The rule, at step *j*:

```
1 - cos(engine_j, pin_j)  <=  3.50  x  route_1cos(j)        (shape)
meanrel(engine_j, pin_j)  <=  0.42  x  route_meanrel(j)     (scale)
```

`route_*(j)` is `engine/route_price.py`'s chunked-vs-recurrent spread for that fixture, cached in
`data/refs/<fixture>/route_allowance.json` with the prompt/feed/specimpl hashes `load_allowance` re-checks,
so a stale spread fails the gate instead of silently becoming the band.
````

### S9. `engine/results.md:1678-1710` — The derivation, re-derived from the shipped files

Why it is in here — the census's §1 evidence for the row `1678-1711`: the same `:1652`
banner governs the whole Q1 block, so it governs this subsection with it.

quoted from `engine/results.md:1678-1710`

````
## The derivation, re-derived from the shipped files

`margin` is not chosen. It is `max over the priced fixture set of (engine deficit / route spread)`, rounded
**up** at two decimals — the tightest constant that admits every step ever measured, with no safety factor.
`engine/route_allowance.py` computes it and **refuses to write** a file whose measurement does not reproduce
the constant in `engine/gate.py`'s BAND, which is what makes the number re-checkable rather than merely
quoted. Re-deriving it from the two shipped allowance files (their `engine_*_when_priced` columns over their
`route_*` columns, 128 steps total):

| | `dec64f` | `dec64t` | sets the margin? |
|---|---|---|---|
| worst engine-deficit / route-spread, cos | **3.0368** at step 32 (2.1555e-3 over 7.0982e-4) | **3.4939** at step 54 (2.3281e-4 over 6.6635e-5) | `dec64t` step 54 → ceil2 = **3.50** |
| worst engine / route, meanrel | 0.3344 at step 54 (1.2596 % over 3.7670 %) | 0.4111 at step 4 (2.0331 % over 4.9452 %) | `dec64t` step 4 → ceil2 = **0.42** |
| steps the flat §10.1 G5 floor would have failed | 7 (27–32, 34) | 0 | — |
| steps the two legal routes themselves fail G5 on | 6 (27–32) | 0 | — |

Two things fall out of that table and both belong on the record.

1. **The proposal in `results.md` guessed k = 4; the measurement says 3.50** — and 3.50 is *not* set by the
   ill-conditioned `dec64f` excursion but by an ordinary, well-conditioned step of `dec64t`. Say the
   inconvenient half plainly: a margin of **3.0 or below leaves `dec64f` step 32 and `dec64t` step 54 RED**,
   and there is no headroom factor hiding that. The constant is a measured property of this engine at these
   two fixtures, so if the engine gets worse the constant moves, and moving it is a visible three-place edit
   (BAND, this comment, every cached file's `margin` field — `load_allowance` fails the gate when a file
   disagrees with the code).
2. **The `meanrel` column got the same treatment rather than being demoted to reported-only**, because cos
   is scale-invariant: a mis-scaled final norm — exactly the class §10.2's meanrel ceiling exists for —
   scores cos 1.000000. Reported-only would have handed that failure class back, which is the hole option 2
   was rejected for. At 0.42 the meanrel line is not simply looser either: it is **tighter than the flat 5 %
   ceiling at 52 of `dec64t`'s and 48 of `dec64f`'s steps** (tightest 1.5836 % and 1.2166 %), and it opens
   only where two legal routes have already blown through 5 % themselves (their median is 8.11 % /
   7.27 %, worst 26.6 % / 60.8 % — the flat ceiling was exceeded by legal reference forms at 44 of 64 steps
   on `dec64f`). Same argument as the cos half, one column over.
````

### S10. `engine/results.md:1712-1738` — Why this is a sharpening and not a loosening — the property that makes it defensible

Why it is in here — the census's §1 evidence for the row `1712-1739`: the same `:1652`
banner governs the whole Q1 block, so it governs this subsection with it.

quoted from `engine/results.md:1712-1738`

````
## Why this is a sharpening and not a loosening — the property that makes it defensible

The allowance follows the reference's own conditioning, so **outside a low-energy window it is tighter than
the floor it replaces**:

| | all-steps cos cap vs the flat 5e-4 | tightest cap | widest cap |
|---|---|---|---|
| `dec64t` | **64 of 64 steps tighter** | 1.2274e-4 (step 41) — 4.1× the floor | 4.4043e-4 (step 0) — still under 5e-4 |
| `dec64f` | **54 of 64 steps tighter** | 9.4788e-05 (step 24) — 5.3× the floor | 5.9983e-3 (step 28) — 12× the floor, inside the excursion |

`dec64t` is the clean case to read: its widest route spread anywhere is 1.2584e-4, so **every** step of it is
judged at `3.5 × 1.26e-4 ≈ 4.4e-4 < 5e-4`, i.e. more strictly than flat G5 was, on a fixture where the engine
never needed an inch of allowance (worst deficit 2.4099e-4). The excursion steps of `dec64f` are where the
band opens, and they are exactly the steps where two legal forms of the reference open it wider still. A band
that is tight where the reference is tight and loose only where the reference is provably loose is a different
instrument from a loosened floor.

What still binds underneath it, unchanged and hard: **token identity (G7) exact per step**, the per-step
**argmax** (the two legal routes agree with each other 64/64 on both fixtures), and the tie policy's proof.
The allowance is a statement about the logits *vector*, never about the token.

And the discriminator still works, which is the only thing that separates this from relabelling:
`engine/gate.py --selftest --fixture dec64t` injects noise on **step 41** — the step whose allowance is
tightest — sized to 1.5× that allowance and still **inside** the flat §10.1 G5 floor, and requires the gate to
FAIL there. That is a failure class the old absolute band could not see at all, and the selftest now proves
the new band sees it. `--break-allow` also refuses a step where 1.5× the allowance would not fit under the
flat floor, so the negative cannot quietly degrade into the ordinary noise test.
````

### S11. `engine/results.md:1740-1765` — The tripwire this derivation used to carry — retired by D-2, kept here because it is how D-2 was found

Why it is in here — the census's §1 evidence for the row `1740-1766`: the H2 at `:1740` —
"The tripwire this derivation used to carry — retired by D-2, kept here because it is how
D-2 was found".

quoted from `engine/results.md:1740-1765`

````
## The tripwire this derivation used to carry — retired by D-2, kept here because it is how D-2 was found

As originally written, `margin` was a **global max over the priced fixture set**, so a newly priced fixture
whose engine-deficit/route-spread ratio exceeded 3.4939 did not just price itself — it raised the constant
and loosened the gate for every existing fixture. The standing rule then was:

> A new fixture whose ratio exceeds 3.4939 is a **finding, not a re-pricing**. Stop and report the step, the
> fixture and the ratio. The answer is either a per-fixture margin (a real change to the rule) or something
> true about that fixture; silently bumping 3.50 is neither.

That tripwire fired for real, on `dec300f` (7.98 at step 86 — see the Q2 section), and the answer it produced
was the first branch: **a per-fixture margin**, decided as D-2. So the tripwire is gone, and with it the
`stale_dir` refusal in `engine/route_allowance.py` that demanded every priced fixture be named in every run:
with per-fixture keying a margin is derived from its own fixture's rows and nothing else, so naming one
fixture is now the normal thing to do, and the tool says out loud which files it did not touch and why they
cannot be affected. `engine/route_allowance.py --selftest` is the test for that property — it prices a
3.49-ratio fixture next to a 7.98-ratio one, requires the first to stay at 3.50, and asserts that a global
max would have answered differently, so reverting the derivation to global fails the test rather than
passing it quietly.

What replaced the tripwire is a different guard, not a weaker one: re-pricing a fixture never moves its band
by itself. If a fresh derivation disagrees with the margin that fixture's file already carries — in either
direction — the tool REFUSES and names the file to delete deliberately, so a band move stays a decision that
shows up in a diff. And the self-reference that per-fixture keying does **not** fix is spelled out in the
D-1/D-2 section: a margin defined as `max(engine deficit / route spread)` puts that fixture's worst step on
the boundary by construction, which is D-3 / Q4b's problem, not this one's.
````

### S12. `engine/results.md:1864-1894` — The tripwire fired on `dec300f`, so nothing was priced and the margin did not move

Why it is in here — the census's §1 evidence for the row `1864-1895`: the census's `:1997` —
"**That `mt32f` FAIL and the "nothing was priced" state were both superseded later the same
day**"; also `:1811` — "**They are superseded two sections down**".

quoted from `engine/results.md:1864-1894`

````
## The tripwire fired on `dec300f`, so nothing was priced and the margin did not move

`engine/route_price.py --fixture dec300f --engine-log …` (CPU, ~4 min, no device) says the reference's two
legal prompt forms are **closer together on this fixture than on either 64-step one** — route 1−cos median
**4.07e-5** vs `dec64t`'s 6.71e-5 and `dec64f`'s 4.93e-5 — 0 of 128 steps under G5 on either side, and both
routes reproduce the pinned ids 128/128. The engine's deficits, though, do not shrink with them:

```
engine deficit / route 1-cos   : median 1.628, p90 2.92, MAX 7.9837 at step 86
        steps over the 3.4939 ceiling: 72 (3.648), 86 (7.984), 89 (3.908), 97 (3.895)
engine meanrel / route meanrel : median 0.246, MAX 0.7198 at step 72 — 16 of 128 steps over 0.42
```

Pricing `dec300f` would therefore derive `margin = 7.99`, `margin_rel = 0.72` — i.e. **more than double the
constant for every fixture at once**, which is exactly the failure mode the Q1 section names as a tripwire.
`engine/route_allowance.py` refuses to write a file whose measurement disagrees with the constant in
`engine/gate.py`, and the instruction was explicit: **report it, do not bump it.** So: no allowance file was
written, `BAND` is untouched, `dec300f` is judged by the flat §10.1 G5 / §10.2 floors (visible as
`rule: flat G5` on all 128 step lines), and it **passes on that basis** — with 3.7e-6 to spare at step 86.

Why the ratio grows with context, as a finding rather than a re-pricing: the route instrument prices *one
choice* — how a prompt gets into the GDN state — and on a long prose prompt that choice matters less, so the
instrument tightens. The engine's end-to-end deficit is not that choice; it is the 24-layer cascade plus an
attention path that grows with lanes. An allowance keyed on the GDN route spread therefore under-sizes a
legitimate deficit once the context is long, and the three possible responses are all operator calls, none
taken here: a per-fixture margin; an instrument that prices the spread at the context in question rather than
at one prompt; or leave long-context fixtures unpriced under the flat floor (today's state, and the flat
floor held). Step 86 by itself is a legitimate object of interest — engine 4.632e-4 where the two legal
routes are 5.802e-5 apart, pinned `S dn2` 131.9 against a run median around 50, argmax still safe at 12×,
one isolated step between neighbours at 9.9e-5 — but it is inside the published floor and it is not a
gate-tuning decision dressed up as a measurement.
````

### S13. `engine/results.md:1963-2003` — What this pass left open, named

Why it is in here — the census's §1 evidence for the row `1963-2004`: that `:1997` banner
sits inside this block.

quoted from `engine/results.md:1963-2003`

````
## What this pass left open, named

1. **`mt32f` is red pending a decision** on tie-policy clause 3 above. Nothing about its numerics is in doubt.
   → **RESOLVED the same day by D-1**: the scope was the bug. The policy now applies only when the engine
   picked an id other than the pin's top-1, and `mt32f` is green with step 11 an ordinary id match.
2. **Nothing was priced.** Pricing `dec300f` would move the global margin from 3.50 to 7.99; `mt32f` and any
   future fixture carry the same risk until the per-fixture-vs-global question is answered.
   → **RESOLVED the same day by D-2**: margins are per fixture, and `dec300f` (7.99/0.72) and `mt32f`
   (2.93/0.24) are priced. The numbers did not move; the keying did.
3. **`dec300f` has no full per-layer operands at depth** (budget arithmetic, above) — if a per-kernel gate at
   400 lanes is ever wanted, it needs either a bigger budget or a knob that dumps operands without the
   18 MiB state.
4. **The engine-side per-step state fingerprint is still unwritten**, and it is the missing instrument for
   both new questions: step 86's 8× and the +3.07e-7/step drift.
5. No second short prose fixture (`prose64f`) was made: `dec300f`'s prompt already runs past 128 tokens and
   serves both jobs, and a second unpriced prose pin would add tripwire exposure without adding a question.
6. **423 lanes passed; 512 was not crossed.** Crossing `MAXSC` is Q5's work, and the lane-ceiling check in
   `gate.py` will refuse it before the kernel does.

### The device passes a worker ran (correctness only, no timing taken)

Lock held by this worker twice; `tools/gpu_lock.sh util` read **92 %** at the first acquire and **91 %** at
the second (discounted per `AGENTS.md`; the serving engine that renders this text is the load). Wall times
are the whole command including the 1.55 GB load:

```
engine/gate.py --fixture dec64f     2 s   PASS (2 steps reported TIE)
engine/gate.py --fixture dec64t     2 s   PASS
engine/gate.py --fixture dec300f    7 s   PASS
engine/gate.py --fixture mt32f      4 s   FAIL — 1 line (tie policy); 290 checks green
decode_run --ids <dec300f prompt> --steps 128      5 s   free run, 423 steps, stopped=0
progress/engine/evidence/q2_session.py mt32f 32                        7 s   feed+gen parity, 32/32
```

**That `mt32f` FAIL and the "nothing was priced" state were both superseded later the same day** — by D-1
(the tie predicate was scoped to the case it exists for, so step 11 is an ordinary id match and `mt32f` is
green) and D-2 (margins became per fixture, so `dec300f` at 7.99/0.72 and `mt32f` at 2.93/0.24 could be
priced without moving anyone else's band). The numbers in this section are what that pass measured and are
still true as history; the section at the end of this file is the current state.

---
````

### S14. `engine/results.md:2562-2620` — The handover: what this pass could not run, and the sequence if a band move is ever decided

Why it is in here — the census's §1 evidence for the row `2562-2621`: the H2 at `:2562` —
"The handover: what this pass could not run, and the sequence if a band move is ever
decided" — that band move has since been decided and run.

quoted from `engine/results.md:2562-2620`

````
## The handover: what this pass could not run, and the sequence if a band move is ever decided

Two things are device-bound and were therefore not run here, and one is deliberately not this pass's to run.

**Re-verifying the fixtures green** (main agent only, blocking, zero subagents — `engine/gate.py` drives the
built binary on the device this session is being served from):

```bash
tools/gpu_lock.sh acquire "D-3 keep: re-verify"; tools/gpu_lock.sh util
for f in dec8 dec64f dec64t dec300f mt32f; do .venv/bin/python engine/gate.py --fixture $f; done
tools/gpu_lock.sh release "D-3 keep: re-verify"
```

Expected, from the recorded D-2 pass: 94 / 550 / 547 / 1059 / 291 checks, zero failures, and the rule line
naming each fixture's own margin (3.50, 3.50, 7.99, 2.93). Nothing here should change that — no input to the
gate moved — so if any of those five goes red after this commit, the cause is something else in the tree and
`route_margin.py` is a useful witness: it will show whether the reference's own routes moved.

**Moving a margin, if a decision is ever taken to do one** (the documented way, which `route_allowance.py`
enforces — it refuses a re-price that differs from the shipped constant until the file is deleted deliberately,
and it writes its own `band_history` line for both the write and the refusal):

```bash
# ONLY under a recorded decision (plan §3, revisit clause (a) or (b)) — D-3 as measured says do not.
rm data/refs/<fixture>/route_allowance.json                       # the deliberate step; it is what makes it visible
.venv/bin/python engine/route_price.py --fixture <fixture> \
    --engine-log tmp/d6_gate_<fixture>.log --out tmp/route_<fixture>.json
.venv/bin/python engine/route_allowance.py --table tmp/route_<fixture>.json --dry-run
.venv/bin/python engine/route_allowance.py --table tmp/route_<fixture>.json
.venv/bin/python tools/band_history.py check                      # must come back clean
```

and the same in `engine/results.md` about what moved and why, which is the half the tool cannot write for you.
If the change decided is instead a *rule shape* (the pin-side denominator above), that sequence is not enough:
it needs a third per-step column in the allowance schema and a matching rule in `engine/gate.py`, which is a
change to what the gate means, not a constant.

Left open, named:

1. **The meanrel definition mismatch is unfixed on purpose.** Making both sides of `margin_rel` use §10.2's
   definition changes four shipped constants by ~4–5×; that is a band move. The measurement needed to expose it
   is in this section and in `route_margin.py`'s two meanrel columns.
2. **`dec64f` and `dec64t` still carry 3.50 / 0.42** from the retired global scheme, while their own derivations
   are 3.04 / 0.34 and 3.50 / 0.42 — i.e. the carried number is *looser* than their own measurement allows.
   Pre-existing, recorded under D-2, untouched here.
3. **`dec64f` step 17 is unexplained by any of this, and it is reference-side.** **Every** route — including
   route a, the pin's own chunked family — picks a different id than the pin at that one step and agrees
   everywhere else (8/8 routes, 63/64). So it is a place where `specimpl` and the HF dump themselves part
   company, not an engine event; it is the single clearest illustration of the oracle-vs-dump term above, and
   related to §07.5's stop-check and D-4's tie finding without being resolved by either.
4. **The pin-side baseline is priced for route a only, and only as a diagnostic.** If the second candidate rule
   above were ever adopted, every legal route's distance to the pin needs pricing per step and caching in the
   allowance file (cheap — the pins are already on disk), and the interaction with A-18's depth decay has to be
   stated rather than assumed.
5. **A 4k-context fixture will want a bigger number, and this section does not make one available.** The honest
   statement for Q5 is that the band there is engine-fitted, a large part of its numerator is the oracle's own
   distance from HF eager, and the two rule shapes that could change what the band means are both priced and
   both rejected here for a reason: the envelope because it fails at every margin-setting step, the pin-side
   denominator because it shrinks the self-reference without removing it.
````

## NARRATIVE — 13 blocks: chronicle and decision-log prose

The class key, again the census's: *NARRATIVE — chronicle / decision-log prose / "what this
pass ran" / "left open".* None of this is retired; it is the record of a pass rather than a
result another file consumes.

### N1. `engine/results.md:81-99` — The D2H discrepancy: measured 27.7 GB/s vs the doc's 63–70 GB/s

Why it is in here — the census's §1 evidence for the row `81-100`: the census's `:84` — "The
loader's copy-back measures 18–28 GB/s. **The doc is probably right and my number is a
different quantity.**" — an attribution the settle-plan (named in the same block) never ran.

quoted from `engine/results.md:81-99`

````
## The D2H discrepancy: measured 27.7 GB/s vs the doc's 63–70 GB/s

`devices/strix-halo-unified-memory.md` records `hipMemcpy H2D / D2H = 76–82 / 63–70 GB/s`.
The loader's copy-back measures 18–28 GB/s. **The doc is probably right and my number is a
different quantity.** The old claim comes from `tools/bw_probe_hip.cpp:155-166`: *one* bulk
`hipMemcpy` of the whole buffer into a pinned host allocation, best-of-N, on a quiet box.
Mine is 335 chunked copies (64 B … 485 MiB) with a `hipDeviceSynchronize()` after each, at
95–99 % util. The arithmetic fits that reading: at 63 GB/s the bytes need 24.5 ms, the
measured 55.8 ms leaves ~31 ms over 335 syncs ≈ 93 µs per sync — and the same window is
where H2D steady still reached 76.7 GB/s, i.e. H2D at spec and D2H at 2.3× below it is hard
to blame on the bus. What would settle it: a chunked-vs-bulk D2H sweep (same total bytes,
(a) one transfer, (b) the loader's 335 chunks with a sync each, (c) 335 chunks with one
sync at the end) at util ≈ 0 and util ≈ 98 %. If (c) ≈ (a) and (b) is ~2× slower, it is the
per-copy sync and the doc's row should be labelled "bulk, per-transfer"; if (a) falls with
util, the row needs a contention caveat instead.

Two notes for whoever runs it: the loader's D2H does **not** need to be fast (verification is
one-time and off by default), and dropping the per-copy sync would change the timing
instrument, not the guarantee — the verify pass would still hash every byte.
````

### N2. `engine/results.md:101-107` — What is deliberately not measured here

Why it is in here — the census's §1 evidence for the row `101-108`: the H2 at `:101` — "What
is deliberately not measured here"; `:103` — "Nothing in `prototype/` was called…".

quoted from `engine/results.md:101-107`

````
## What is deliberately not measured here

Nothing in `prototype/` was called (other agents own those files), so no kernel result
depends on this loader and nothing here depends on a kernel. The dependency is the
checkpoint file only.

---
````

### N3. `engine/results.md:130-154` — Deferred, with the price named

Why it is in here — the census's §1 evidence for the row `130-155`: the H2 at `:130` —
"Deferred, with the price named" (the census writes this row and `156-184` as one evidence
line).

quoted from `engine/results.md:130-154`

````
## Deferred, with the price named

1. **Unpacked projections.** `proj_gemv` is called once per projection (q, k, v, o, in_proj_qkv,
   in_proj_z, in_proj_b, in_proj_a, out_proj…) rather than through the packed multi-weight form the
   kernel supports. Now that `decode_step` exists the real bill is known rather than estimated:
   **424 launches/token, 114 of them projection GEMVs** (`gate.py` section (d) asserts the split).
   The packable part is `in_proj_a`+`in_proj_b` (16 rows, 32 KiB each, same input row) alongside
   `in_proj_qkv`/`in_proj_z`: one packed launch instead of four per GDN layer, so 54 launches ≈
   **0.10 ms/token** at the ≈1.85 µs floor (`tools/launch_overhead_hip.cpp`). The earlier "≈358
   launches, ≈0.12 ms removable" in this file was arithmetic over a plan the loop does not use.
2. **Dtype adapters: 115 cast launches/token.** `mlp_gemv` and `gdn_decode`'s conv/gates/rec take
   F32 activations, while `glue`/`proj_gemv`/`gdn_gated_norm` emit BF16, so the engine widens: 5 per
   GDN layer, 4 per attention layer, 1 before `lm_head`. 48 of the 115 are the MLP's (its input row
   and its `act`, twice across 24 layers). `kernels_cast.cpp` now exists, and the GDN layer's three
   widenings (`in_proj_qkv`, `in_proj_a`, `in_proj_b`) are already folded into ONE launch
   (`cast_bf16_to_f32_3`), which is why a GDN layer costs 5 casts and not 7. The real fix is still a
   BF16-staged form of `mlp_gateup`/`mlp_down` — `proj_gemv` proved that is bit-identical for a dot
   product — which would delete 48 of the 115. Not done: it means touching a graduated kernel and
   re-gating it, and `mlp_gemv`'s gate belongs to another phase. What was done here instead is the
   other half of the old estimate: `conv1d.weight` and `dt_bias` are widened **once per request** by
   `prepare_weight_adapters`, not once per token, so those 36 rows cost 36 launches per request.

Both deferrals are launch-count, not bandwidth: at ≈240 GB/s the token's memory floor is well under
1 ms and 424 launches is ≈0.78 ms, so they are the first thing a perf pass should attack — after the
token matches.
````

### N4. `engine/results.md:156-183` — Open questions this phase did not decide

Why it is in here — the census's §1 evidence for the row `156-184`: the H2 at `:156` — "Open
questions this phase did not decide" (the census writes this row and `130-155` as one
evidence line).

quoted from `engine/results.md:156-183`

````
## Open questions this phase did not decide

* **Conv window dtype — corrected in the spec, 2026-10-11.** `state.h` stores it F32 (1.266 MiB)
  where §08.1's cell used to say BF16 (0.633 MiB), because the graduated `gdn_decode` conv kernel
  takes F32 in and out and 36 cast launches per token is the wrong price for 0.6 MiB on a device
  using 4 % of its pool. The stored *values* are BF16-exact either way. §08.1 and §00.5's conv row
  now say F32 / 1.27 MiB, naming the old cell; `state_sizes.py` greps both pages for that figure, so
  the doc cannot drift back to BF16 without the size check failing.
* **§08.1 / §00.5 doc slips — corrected, 2026-10-11** (all four were found by `state_sizes.py`, and
  its lines are now `FIXED …` rather than `OPEN …`): the "4 KiB/token/layer" KV cell contradicted the
  same table's 12 KiB/token total (2 KiB/layer is what the shapes give, and what the audit's proposed
  8 KiB was not); the MTP KV row had the same 2× slip; §00.5's 3.07 GiB at full context is 3.00 GiB
  from its own 12288 B/token. **Still open:** KV layout is head-major `[head][slot][dim]` per the
  kernel, not §08.5's `[slots][2][256]` — §08.5 says so no more than it did, and `state.h` carries
  the departure.
* **§04.5 / A-01 still state the affine in F32** in `specification/`, and that is a reading the
  pins refuse: with the weight rounded to BF16 and widened back to F32 it misses 74 860 of 294 912
  elements, with the raw F32 checkpoint weight 105 683 (`prototype/gdn_gated_norm/results.md` §1).
  `speccheck.specimpl.rmsnorm_gated` is **not** among those: called the way speccheck calls it it
  is bit-exact, 0 of 294 912, because `Weights` hands it a BF16 weight. An earlier revision of this
  file reported the oracle as 105 683 elements off — that was the mirror probe handing the oracle
  an F32-widened weight, retracted in `fc444d0`, and the probe now asserts its oracle row at 0.
  Correcting the spec wording is a decision for the spec owner, not this phase.
* **`lm_head` full + host argmax**, not the fused single-kernel form: the fused form resolves ties
  to the highest id and §03.4 fixes ties to the lowest. Costs ≈1 MB D2H + a host scan (≈0.04 ms)
  per token and gives the gate its logits. Worth revisiting only with a tie-corrected fused kernel.

---
````

### N5. `engine/results.md:1412-1429` — What this does not settle

Why it is in here — the census's §1 evidence for the row `1412-1430`: its own heading —
"What this does not settle" (the census writes this row with `1767-1778` and `1779-1796` as
one evidence line).

quoted from `engine/results.md:1412-1429`

````
## What this does not settle

Two things, and both are named rather than assumed away.

1. **It does not prove the engine's state is fine.** It shows the symptom is not *attributable* to a bug by any
   measurement in this repo, because the reference's own legal forms produce a larger version of the same dip at
   6 of the 7 steps. The engine-side per-step state fingerprint (the four moments `decode_run` does not print yet)
   is still the only instrument that can say "the engine's `S n2` equals the pinned `S n2`", and it is still
   unwritten.
2. **It does not separate scale from the eos crossing, and it cannot: this fixture has both at once.** The pinned
   chain samples the stop id (248046) at **step 24**, and every under-floor step is after it — route-under-G5 steps
   27–32 sit **6 after, 0 before**; the engine's seven sit **7 after, 0 before**. The post-crossing window is also
   where the reference is emitting its own special tokens (248044, 248045, 248068, 248069 at steps 26–34), which is
   a structurally different regime: a model driven past its own end of turn, on a prompt whose answer ended at
   step 24, is being asked to continue something it had already finished. "ill-conditioned state" and "the model is
   off-distribution past its own stop" make the same curve here, and `dec64f` cannot tell them apart. A second
   fixture whose chain never crosses the stop id can; it is the next instrument, and it is cheap (dump + `--verify`,
   CPU-only, then the same two commands above pointed at it).
````

### N6. `engine/results.md:1767-1777` — What was not chosen, and why it is on the record

Why it is in here — the census's §1 evidence for the row `1767-1778`: its own heading —
"What was not chosen, and why it is on the record".

quoted from `engine/results.md:1767-1777`

````
## What was not chosen, and why it is on the record

* **Option 2 — gate the decision, not the vector** (per-step argmax only; logits demoted to reported
  evidence). Cheapest and honest about what the gate really verifies, but it gives up any claim on the
  vector's shape, i.e. the whole wrong-but-argmax-preserving class. The argmax rule is a hard gate *here*
  too, unchanged — it is just not the only thing this gate checks.
* **Option 3 — keep 0.9995 hard and label the window** (a per-fixture list of steps reported as `SPREAD`
  but not armed). REJECTED, and the reason is the whole reason: **an exempted step is judged by NOTHING**, so
  a real bug landing on an exempted step passes silently — and the list is hand-maintained from a measurement
  that can move. Under the allowance there is no exempted step and no list to keep in sync: every step is
  judged, by a number measured at that step.
````

### N7. `engine/results.md:1779-1794` — What the pass does not buy

Why it is in here — the census's §1 evidence for the row `1779-1796`: its own heading —
"What the pass does not buy".

quoted from `engine/results.md:1779-1794`

````
## What the pass does not buy

* The spread is between two forms of **`specimpl`** (chunked §04.7 vs recurrent §04.4) against a pin made by
  **HF**. That is inherited from the instrument, and it means the allowance measures the GDN route choice
  inside one Python reference, not every legal implementation route. It is the only end-to-end instrument
  this repo has, and it is used as such, with its bias named.
* `dec8` is **not priced** — `route_price.py` prints its 8-step table but no allowance file was written, so
  its 8 steps are judged by the flat floors and every step line says `flat G5`. Pricing it is outstanding,
  and would have to be re-derived against the whole priced set (see the tripwire).
* The engine-side per-step state fingerprint is still unwritten, so "the engine's `S n2` equals the pinned
  `S n2`" remains unmeasurable at depth. Decision 2's open question 1 is untouched by this commit.

*Also from this session, unrelated to the band:* `68c4e9d` makes `decode_run` create its `--out` directory
before loading weights. The failure it retires was losing a device window to `cannot open
tmp/…/sampled.txt` after 1.55 GB of `hipMalloc` + `hipMemcpy` and a full run — now a bad path fails in
milliseconds, and the refusal is printed before any step runs.
````

### N8. `engine/results.md:1815-1842` — The pre-flight, and making them

Why it is in here — the census's §1 evidence for the row `1815-1843`: the H2 at `:1815` —
"The pre-flight, and making them".

quoted from `engine/results.md:1815-1842`

````
## The pre-flight, and making them

```bash
.venv/bin/python tools/stop_probe.py --max-new 300 --message "<the letter prompt>"
#   prompt: 295 tokens; stop ids (248046, 248044) appear at prompt index [286] — prompt-phase, not a stop
#   FIRST STOP (248046) AT STEP: 253      first legacy stop 248044 at step: 255
.venv/bin/python tools/reference_dump.py --message "<the letter prompt>" --decode 128 --tag dec300f \
        --lean --deep-steps none --logits-bits bf16 \
        --first-stop "253 (probed to 300 generated tokens; legacy stop 248044 at 255)"     # ~15 s, CPU
.venv/bin/python tools/reference_dump.py <same args> --verify data/refs/dec300f
#   REPRODUCE PASS — every pinned byte and every pinned manifest value came back identical
```

Three tools grew knobs to make this possible, each committed separately (`tools/chat_render.py`,
`reference_dump --turn/--first-stop/--deep-steps none`, and the `--verify` fix below):

* **`--first-stop` records the probe's answer in the manifest**, which the standing rule asks for and
  nothing used to enforce; a lean fixture pinned without it now prints a warning out loud.
* **`--deep-steps none`** (no full per-layer dump at any step) is what keeps a 128-step fixture inside the
  ~60 MB budget, and the reason it had to be added is arithmetic, not taste: 128 steps × 0.473 MiB of
  BF16 logits = **60.6 MiB is the floor by itself**, while one full step costs 26.4 MiB of which 18 MiB is
  the GDN state. So this fixture has *zero* full per-layer steps, where `dec64f`/`dec64t` have one. State
  carry at depth is carried here by the fingerprints, as it is in those fixtures — it is a pin-policy
  trade, stated rather than hidden.
* `tools/reference_dump.py --verify` had to be fixed first: it **failed on every priced fixture** because
  `route_allowance.json` sits in the fixture directory and a dumper run cannot reproduce it — measured on
  `dec64t`, `REPRODUCE FAIL … missing from the fresh run: route_allowance.json` with all 826 model tensors
  byte-identical. It is now carried-and-named, not compared; its own provenance is `gate.py:load_allowance`.
````

### N9. `engine/results.md:2154-2183` — The pass, and what a worker ran

Why it is in here — the census's §1 evidence for the row `2154-2184`: its own heading — "The
pass, and what a worker ran" (first of the five chronicle rows the census writes in a single
evidence line: `2154-2184`, `2185-2203`, `2316-2339`, `2340-2356`, `2549-2561`.

quoted from `engine/results.md:2154-2183`

````
## The pass, and what a worker ran

All correctness-only, each inside the `<60 s` worker allowance, under `tools/gpu_lock.sh` (tag
"engine gate D-1/D-2 last worker pass (correctness only)"), `gpu_lock.sh util` **92 %** at the first read and
**60 %** at the last — discounted as usual, the serving engine that renders this text is the load and it
busy-polls the iGPU. **No timing taken**: the wall figures below are the cost of running a check, printed only
to show each one fit the worker allowance, and none of it goes near `bench/performance-status.md`. The whole
pass is one command — `progress/engine/evidence/d6_final_pass.py` — and it left the ten logs in `tmp/d6_*.log`. Both helpers live
under `tmp/`, which is gitignored: they are how these figures were made, the tables above are the record.

```
engine/route_allowance.py --selftest                                   CPU    ~0 s   PASS (6 checks)
engine/gate.py --selftest --fixture dec8                               CPU    1.1 s  PASS (tie + allow cases SKIPped out loud)
engine/gate.py --selftest --fixture dec64f                             CPU    5.0 s  PASS (10 cases incl. 3 tie injections)
engine/gate.py --selftest --fixture dec64t                             CPU    3.5 s  PASS
engine/gate.py --selftest --fixture dec300f                            CPU    6.1 s  PASS (5 % noise SKIPped: inside its own allowance)
engine/gate.py --selftest --fixture mt32f                              CPU    3.2 s  PASS (9 cases incl. 3 tie injections)
engine/gate.py --fixture dec8       device  1.1 s  PASS — 94 checks, flat G5 (still unpriced)
engine/gate.py --fixture dec64f     device  2.0 s  PASS — 550 checks, 2 steps TIE, margin 3.50 (unchanged)
engine/gate.py --fixture dec64t     device  2.2 s  PASS — 547 checks, margin 3.50
engine/gate.py --fixture dec300f    device  6.8 s  PASS — 1059 checks, margin 7.99, worst step 86 at 7.984x
engine/gate.py --fixture mt32f      device  4.1 s  PASS — 291 checks, 32/32 ids, margin 2.93, step 11 an ordinary match
engine/route_price.py --fixture mt32f --engine-log … --out …  CPU  3.8 s  the table that priced mt32f
speccheck/run_all.py                                       CPU  7m50s  26 gates run, 0 failed
```

`dec300f` was **not** re-priced on the CPU: `progress/engine/evidence/route_dec300f.json` (from the Q2 pass) was checked step by
step against a fresh gate log with `route_price.py`'s own parser and the engine column is identical at all
128 steps (`progress/engine/evidence/d2_dec300f_still_valid.py`), so the shipped table was cited rather than re-made — the same
rule `AGENTS.md` gives for any measurement already on disk whose inputs have not moved.
````

### N10. `engine/results.md:2185-2202` — What this pass left open, named

Why it is in here — the census's §1 evidence for the row `2185-2203`: its own heading —
"What this pass left open, named", and this is the first of the two blocks carrying that
heading.

quoted from `engine/results.md:2185-2202`

````
## What this pass left open, named

1. **`dec8` is still unpriced**, so its 8 steps are judged by the flat floors and every step line says
   `flat G5`. Under D-2 pricing it would write only its own margin, so the old objection ("it would move the
   shared constant") is gone; what is left is that no question here needs its answer.
2. **D-3 / Q4b**, the engine-independent multiplier, above. D-2 keyed the margin per fixture; it did not make
   the margin engine-independent.
3. **No fixture has a 3-way-or-wider tie**, so "all tied ids share one payload" is exercised at 2-way only.
   It is structural in `pin_tie` (membership *is* payload equality), but it is untested at width.
4. **`margin_basis` is absent on `dec64f`/`dec64t`'s files**, on purpose. If either is ever re-priced the tool
   refuses first, so keeping the carried constants stays a decision rather than an oversight.
5. **Why `dec300f`'s ratio is ~8 while the 64-step fixtures' are ~3 is still unexplained.** D-2 accommodates
   it per fixture; it does not explain it. The candidate cause named in the Q2 section stands — the route
   instrument prices only the GDN prompt-form choice, while the engine's end-to-end deficit includes an
   attention path that grows with lanes — and the missing instrument is still the engine's own per-step state
   moments next to the pinned ones.

---
````

### N11. `engine/results.md:2316-2338` — What this pass ran, and how to extend it

Why it is in here — the census's §1 evidence for the row `2316-2339`: its own heading —
"What this pass ran, and how to extend it".

quoted from `engine/results.md:2316-2338`

````
## What this pass ran, and how to extend it

One device call, the whole corpus, correctness only, under `tools/gpu_lock.sh` (tag "D-4 corpus free run";
`gpu_lock.sh util` read **93 %** at the acquire and the lock showed `free` after the release — discounted per
`AGENTS.md`, the serving engine that renders this text is the load). Wall for the whole command including the
1.55 GB load was 18 s, comfortably inside the worker allowance; **no timing was taken and nothing here
belongs in `bench/performance-status.md`** — the wall figure is the cost of running a check, not a
measurement. The margin pass is CPU-only (~45 s) and the report pass touches neither.

```bash
tools/gpu_lock.sh acquire "D-4 corpus free run"; tools/gpu_lock.sh util
.venv/bin/python tools/corpus_freerun.py                      # device, 18 s: all 49 prompts, then the report
tools/gpu_lock.sh release "D-4 corpus free run"
.venv/bin/python tools/corpus_margin.py --out data/refs/corpus48-margins.json   # CPU, ~45 s
.venv/bin/python tools/corpus_freerun.py --report-only --json progress/engine/evidence/corpus-freerun-2026-10-12.json
```

`--start`/`--limit` chunk the corpus across several calls if a longer budget is ever wanted (a 128-token
corpus budget would be ~4× the steps and still one session; `--steps 128` overrides the corpus' own 24,
which only changes what this tool prints — the pins stay 24 and the comparison caps at the pin length).
The full per-prompt records — first divergence, both margins, the flip's rank in the reference's top-8,
the tie flag — are committed as `progress/engine/evidence/corpus-freerun-2026-10-12.json`; the engine manifest it reads lives
in the gitignored `data/` like every other regenerable fixture.
````

### N12. `engine/results.md:2340-2355` — What this left open, named

Why it is in here — the census's §1 evidence for the row `2340-2356`: its own heading —
"What this left open, named", the second of the two blocks that heading appears on.

quoted from `engine/results.md:2340-2355`

````
## What this left open, named

1. **Tie-aware prefix agreement is a metric with no gate behind it.** If a corpus gate for the engine is ever
   wanted, the tie clause has to be written into it deliberately (this pass measured with it and reported it
   beside the strict number; it changed no predicate).
2. **The steps after a first flip are unmeasured.** Measuring them needs a different question — replay the
   engine along the *reference's* chain at corpus scale (forced, like `dec64f`), which prices per-step
   agreement without chaos, instead of a free run that burns its budget on the first tie.
3. **Tie frequency is now a measured property of this checkpoint and nothing consumes it.** 28/49 chains
   contain a tie inside 24 tokens; any future gate written in "tokens equal" is silently a tie-frequency
   gate too, and the number to quote beside it is this one.
4. **49 prompts × 24 tokens is A-18's budget, not a bigger one.** The corpus was never widened (§10's G7
   intent was "hundreds cheap"); the run costs 18 s of device, so widening it is a fixture-regeneration
   decision (`tools/corpus_dump.py --tag … --steps …` on a wider prompt list), not an engineering one.

---
````

### N13. `engine/results.md:2549-2560` — What this pass ran, and what was left for the device pass

Why it is in here — the census's §1 evidence for the row `2549-2561`: its own heading —
"What this pass ran, and what was left for the device pass".

quoted from `engine/results.md:2549-2560`

````
## What this pass ran, and what was left for the device pass

Everything here is CPU-only: the torch reference passes plus the already-recorded gate logs. No device work was
done and **no timing was taken**; the gates themselves were not re-run, because nothing that feeds them changed
(this section adds one read-only instrument and documentation; no band, no allowance file, no `specimpl`, no
`gate.py`). The recorded D-2 re-verification remains the green state.

```bash
.venv/bin/python engine/route_margin.py --selftest                    # 12 cases, no files touched
.venv/bin/python engine/route_margin.py --fixture dec64f --engine-log progress/engine/evidence/d6_gate_dec64f.log \
    --check-patch --out progress/engine/evidence/route_margin_dec64f.json                  # repeat per fixture
```
````

## The 3 H3 blocks §1's prose adds to the movable set (+105 lines)

These are subsections, so they carry no row of the §1 table; the paragraph under it names
all three and classes them. Two are SUPERSEDED and one is NARRATIVE, and they are kept in
the order the census listed them.

### H1. `engine/results.md:565-600` — Levers, ranked by expected ms/token

Why it is in here — the census's §1 evidence for the row `565-601`: §1 adds the three H3
blocks to the movable set in the prose under its table rather than as rows of it, and
classes this one there: "Three H3 blocks add to the movable set — `565-601` "Levers, ranked
by expected ms/token" (NARRATIVE)".

quoted from `engine/results.md:565-600`

````
### Levers, ranked by expected ms/token

| # | lever | expected | bit-identical? | how to confirm |
|---|---|---|---|---|
| 1 | **`proj` rows/block 8 → 32** — all 114 engine projections launch at 256 threads (`kRowsPerBlock = 8` × 32 lanes), the width the §4 curve prices at 3.75 µs against 2.776 µs at 1024, and the block-starved narrow shapes are the ones that care | **−0.50 ms** — a clean-window pair on the same 48-job plan: 3.039 ms at rpb 8 (§6) vs 2.534 ms at rpb 32 (E2's 0.0528 ms/launch × 48, same window as the engine's 11.978) | **yes, by construction**: one wavefront owns a row (`proj_gemv_kernels.h:116-133` — `row = blockIdx.x*rpb + threadIdx.x/ws`, the K loop strides by lane, `reduce_row` is a within-warp butterfly), so `rpb` moves rows per block and nothing that touches a value | re-take the rpb ladder in one clean window, then A/B the engine (commands below). It is one constant in `engine/kernels_proj.cpp`, and it is a decision, not a fix: 8 is the op's gated default |
| 2 | **Pack the GDN input projections** — `in_proj_qkv` + `z` + `b` + `a` in one launch per GDN layer, −54 launches. The deferral in this file priced it at ≈0.10 ms from launch count; the per-shape table prices the *same bytes* 4× faster merged | **−0.11 ms** (the `b`/`a` fold, as `proj_gemv` measured it) to **−0.57 ms** (the full four-into-one; cross-`tile` comparison, so treat as the upper bound) | same kernel, fewer launches — the outputs are the same dots | `prototype/proj_gemv/run.py --layers all --steps all --time --sweep`; the gap between `PACK_inproj` (181.9 GB/s) and the four separate shapes (141 GB/s) *is* the price |
| 3 | **The 115 cast launches** — upper bound −0.43 ms, and the cheap deletion is not the one this file has been carrying. Folding the widening into the *producer* (`glue_add_norm` holds the norm row in registers and could also write the F32 copy for a 4 KB store) removes 49 of them for near-zero marginal traffic. Folding them into the *consumer*, which is the old `mlp_gateup`/`mlp_down` plan, has a measured precedent against it: `PACK_X` cost **+0.278 ms** over 48 launches to save 48 casts | −0.18 … −0.43 ms, and plausibly net-negative if done the consumer way | yes (every BF16 is exact in F32) | anchor the µs/launch at 256 threads (command below), then decide which producer to touch — both are graduated kernels, so both cost a re-gate |
| 4 | **The per-step logits tail** — 0.128 ms of host scan/alloc + 0.015 ms D2H, always serial with the GPU, plus 0.254 ms when a run dumps logits. The sync itself is **not** the lever: §03.4's lowest-id tie rule put the argmax on the host, and a tie-correct fused argmax would shrink the D2H from 0.993 MB to 4 B but could not remove the wait | −0.14 … −0.40 ms | n/a — no value changes | A/B `decode_run` with and without `--out` (command below); the CPU pieces are already measured above |
| 5 | ~~HIP graphs~~ | **−0.06 ms** | — | **retired by measurement**: §1's probe replays 64 captured launches at 1.78 µs against 1.85 µs eager. This file's earlier "graphs are the first perf item" line is superseded (note added there) |
| — | per-step allocation churn | −0.007 ms | — | measured; not a lever |
| — | stage dumps written every step | −0.254 ms, **only** when `--out` is passed | — | measured on CPU; a serving loop never pays it, so this is a note about how the throughput figure was taken, not an engine item |

**So: the biggest single lever is neither the casts nor the per-step sync. It is the 114 projections at 256
threads** — with the packed-plan fold immediately behind it, the two together worth −0.6 … −1.1 ms, i.e. as
much of the gap as everything else combined. The casts are an upper bound with a measured counter-example,
and the sync is 1.3 % of the step. The biggest *unknown* is the 2.5 ms residue, and it needs one instrument,
not one more measurement of a family in isolation.

**Commands for the top three** (operator-run; a worker may not open a GPU window):

```bash
# 1 · the projections' block width, and the packing question, in one clean window.
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps all --time --sweep

# 2 · the engine A/B once `engine/kernels_proj.cpp`'s kRowsPerBlock says 32 (bit-identical per the
#     kernel geometry above; the proj gate should still be re-run because it owns the default).
#     Same shape as the throughput row: 17 prompt tokens one per step, then 64 generation steps.
IDS=248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198,248068,271,248069,271
./engine/build/decode_run --ids "$IDS" --steps 64            # no --out: driver's dump excluded
./engine/build/decode_run --ids "$IDS" --steps 64 --out tmp/engine_gate_nodump   # with it

# 3 · the dispatch anchor every small-kernel row above is priced from, at the widths the engine uses
#     (256 threads: glue, casts, the narrow projections; 1024 is the width glue prefers).
.venv/bin/python prototype/glue/run.py --time --threads 256 --iters 300
.venv/bin/python prototype/glue/run.py --time --threads 1024 --iters 300
```
````

### H2. `engine/results.md:712-768` — Commands for the main agent, in the order that changes strategy

Why it is in here — the census's §1 evidence for the row `712-769`: the same §1 sentence
classes this one with the next: "`712-769` "Commands for the main agent, in the order that
changes strategy" and `1048-1057` … (both SUPERSEDED: `engine/prefill.cpp` and
`engine/prefill.h` exist)".

quoted from `engine/results.md:712-768`

````
### Commands for the main agent, in the order that changes strategy

Every line is a timing or a ≥1-minute device run, so none of them was run in this pass. `prototype/glue/run.py`,
`prototype/proj_gemv/run.py` and `prototype/gdn_gated_norm/run.py` **take the lock themselves** — do not
hold it across them (AGENTS.md trap row).

```bash
# 0 · A/B the width change on the same day, before anything else. `bc61dce` is the width commit, so
#     the before-binary is one rebuild away and the tree is put back by the same four lines
#     (`git diff --stat` must be empty afterwards). Both sweeps take --binary, and the WAVE path
#     changed width too, so prefill has to be re-taken in the same window as decode.
mkdir -p tmp/q6
git show bc61dce~1:engine/kernels_proj.cpp > progress/engine/evidence/q6/proj_rpb8.cpp
cp engine/kernels_proj.cpp progress/engine/evidence/q6/proj_rpb32.cpp
cp progress/engine/evidence/q6/proj_rpb8.cpp  engine/kernels_proj.cpp && engine/build.sh && cp engine/build/decode_run tmp/q6/decode_run_rpb8
cp progress/engine/evidence/q6/proj_rpb32.cpp engine/kernels_proj.cpp && engine/build.sh
tools/gpu_lock.sh acquire q6-rpb-ab && tools/gpu_lock.sh util
.venv/bin/python tools/context_sweep.py  --binary tmp/q6/decode_run_rpb8  --lanes 128,512,1024,2048,4032 --steps 32 --runs 3
.venv/bin/python tools/context_sweep.py  --binary engine/build/decode_run --lanes 128,512,1024,2048,4032 --steps 32 --runs 3
.venv/bin/python tools/prefill_sweep.py  --binary tmp/q6/decode_run_rpb8  --lanes 64,256,1024,3072 --runs 3
.venv/bin/python tools/prefill_sweep.py  --binary engine/build/decode_run --lanes 64,256,1024,3072 --runs 3
tools/gpu_lock.sh release q6-rpb-ab

# 1 · the 4032-lane point, which §6 quotes at a 37.8 % spread and a median. Five runs, stated as
#     median + spread again; if it stays bimodal the split attention path at 8 blocks/site is the
#     suspect, not the memory system.
tools/gpu_lock.sh acquire q6-4032 && .venv/bin/python tools/context_sweep.py --lanes 4032 --steps 32 --runs 5 && tools/gpu_lock.sh release q6-4032

# 2 · ledger §2's two "not measured on their own" rows, now selectable (this commit's --only).
#     GDN layers are 81 % of the projection bytes and own the 5 %-efficient shapes; if the GDN row
#     comes back at the family's 63 % and attention at 85 %, the narrow-shape work has a target.
#     run.py takes the lock itself.
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps all --time --only gdn
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps all --time --only attn

# 3 · the rows/block ladder in ONE clean window, which is the only way to settle whether the -0.50 ms
#     above is real: the 8-vs-32 pair has so far only ever been compared across windows.
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps all --time --sweep --rows 8

# 4 · the dispatch anchor the small-kernel rows are priced from, at the two widths the engine now
#     uses. This is the number that would replace the curve's 1.8× over-price on small launches.
.venv/bin/python prototype/glue/run.py --time --threads 256  --iters 300
.venv/bin/python prototype/glue/run.py --time --threads 1024 --iters 300

# 5 · the correctness re-runs on the committed tree, since the engine now ships a width that op's gate
#     defaulted away. Both were run in this pass (all-24-layer `--rows 32` and `--rows 8`: 114/0 each;
#     the negatives and the eight engine fixtures as counted above, x1.0000 margins); they are here so
#     the tree is re-checkable by anyone, and `--steps all` makes the first line the full 912-comparison
#     gate rather than the 114-comparison slice this pass had time for.
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps all --negative
.venv/bin/python prototype/proj_gemv/run.py --fixture dec8 --layers all --steps all --rows 32

# 6 · Q6.1, the full-plan burst against the engine's own step: BUILT (see the Q6.1 section below).
#     `decode_run --time-stages` is that experiment and one more besides — it prices the same launches
#     at burst, per-stage and per-kernel submission in one process, with the per-family clock inside
#     `decode_step` that this line asked for if the burst came back under the step.
```
````

### H3. `engine/results.md:1048-1054` — Context, not action: there is no prefill

Why it is in here — the census's §1 evidence for the row `1048-1057`: the third of those
three, superseded for the reason that same sentence gives: "`1048-1057` "Context, not
action: there is no prefill" (both SUPERSEDED: `engine/prefill.cpp` and `engine/prefill.h`
exist)".

quoted from `engine/results.md:1048-1054`

````
### Context, not action: there is no prefill

§07.1 has no chunked prefill, so all 17 prompt steps cost a full decode step — ≈0.2 s of prompt before the
first generated token, and 17 of the 81 steps in every figure above are prompt steps. A prefill pass would
remove most of that from a first-token figure and would not change a token of the steady-state 11.978 ms.
It is recorded here as context for reading the row, not as a work item: the gap this section is about is a
decode-step gap, and prefill does not touch it.
````

## What this file deliberately does not do

* It retires nothing. Every block above is still in `engine/results.md`, at the line numbers
  printed beside it, and a reader after the live contract should read the journal rather than
  the copy. What was retired says so in the journal; it is not gone from it.
* It decides no new classes. The 59 LIVE-DATA sections stay in the journal unread here, and
  the zero-cite measurement blocks that §7.5 of the archive review flags as arguable were
  left alone, so no claim here depends on an arguable call.
* It adds no rule. Rule 3 of `progress/README.md` already covers a new dated document, and §7.2
  of the archive review notes that the archive has no rule for a journal *section* — option
  (a′) sidesteps that by being a copy rather than a move.
* It is not a substitute for `--sections` over the journal. The fence tracker of
  `python3 tools/citation_census.py --sections` toggles one level deep, so run over this file it
  counts the headings that are *inside* quoted blocks (it reports 35 H1s here, of which one is
  this document's own). `--verify-quotes` is the mode written for this file: it consumes a fence as
  a unit, which is why the outer fence is longer than anything it contains.

