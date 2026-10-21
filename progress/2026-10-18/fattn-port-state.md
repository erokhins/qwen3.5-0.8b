# fattn port — continuation state (written before the next dispatch, 2026-10-18)

Purpose of this file: a resumption point. The port worker `d1f79493` expired its 2 h deadline mid-task; this
records what is verified, what is not, and the exact next commands, so no continuation has to re-derive them.
Read `prototype/attn_wmma/fattn_score_handover.md` (§5a–§5e) alongside it.

## 0. Where the tree is

| thing | state |
|---|---|
| branch | **`wip-fattn-port`**, tip `d7057c4` — the engine port is committed HERE, not on `master` |
| `master` | tip `5666adc` (`prototype/attn_wmma: build D62's falsifiable form …`), untouched by the port |
| build | `bash engine/build.sh` → **rc 0** (`loader_test`, `decode_run`) at `d7057c4`. Verified by main agent. |
| gate | **never run** on the port. No admission, no burst, no census at the shipped placement. |
| dirty | `prototype/attn_m2/attn_m16{,_grp2}` binaries and scratch only — deliberately uncommitted |

Working on the branch is intentional: a gated re-port can either split `d7057c4` into real topical commits
(kernel + gate + echo) or revert it whole. Nothing is pushed, so history work is free.

## 1. Engine headline (unchanged all session)

- prefill **5,364 tok/s** (median 763.49 ms at 4096 lanes), decode **71.3 tok/s**.
- defaults: `--gemm lt`, `--gemm-pins all`, `--attn-tile 1`, `--attn-touch 1`, `--gdn-spell fused`,
  **`--gdn-loop 512`**, `kRowsPerBlock = 32`. `--attn-fattn` exists on the branch, default 0 (NOT confirmed
  by gate — confirm before trusting it).
- family shares of the wave: `gdn_rec` 34.9–36.5 %, `attn` 34.0–34.9 % (264.71 ms), matmul ~23 %.

## 2. Why this route is funded at all (D61 → D62, both in the decision log)

- **D59 overturned by D61**: the "+112 % geometry tax" and "the register file refuses it" were both refuted.
  `attn_m16_grp2` (96 VGPR → **5 blocks/CU, shipped parity**) moves the time **≤5 %** vs GRP4 while giving
  up half the traffic win. Mechanism is **serial-round latency at low interleave**: the wall tracks *lanes*
  and is **flat in rows and flat in block count**. §28 of `progress/2026-10-21/performance-status-2026-10-21.md`.
- **D62 funded at ROUND scope**: llama.cpp's traced kernel is **384 `v_wmma`** (zero `v_mfma`, zero `v_dot`),
  **1 cross-lane op per softmax row per round** against our **5-step butterfly per score**, **10 independent
  accumulator chains**, one rescale per 64 KV rows; **3.29 warp-inst/touch raw, 7.16 pipe-weighted** vs our
  **89.9 measured**. They run at **8 wavefronts/CU with 358 spilled VGPRs and beat us ≥5×** ⇒ occupancy is
  not the axis. Prize through the measured elasticity α=0.44: **+30 % of prefill** (ceiling +39 %).
- **Two retired beliefs**: (a) **LDS binds, not registers** — `kernel_occupancy_probe.cpp`'s "512 lane-slots
  /SIMD" was hard-coded arithmetic; driver reports `MaxRegistersPerMultiprocessor = 196,608` ⇒ ~307
  lane-regs/thread at our shape, so **D30's "128 VGPR/lane wall" never existed at 128 threads**.
  (b) Their 5× is **not** the matrix roof (~17 % of the 52.63 TFLOP/s this box sustains).

## 3. Verified on the probe (`m16_fattn_score`, committed at `5666adc`)

- claim (i) **4 blocks/CU, 16 wf/CU**, 193 VGPR, **16,384 B LDS with ZERO slack** (Q 8 KiB + K 8 KiB, V
  aliased over K) — one more byte of `__shared__` ⇒ 3 blocks/CU. Driver probe has a three-control canary
  (65,536 B→1, 0 B→16, **12,288 B→5 = the recorded shipped row**).
- claim (ii) **9.719 pipe-weighted slots/touch** (bar ≤20); 4.855 in D57's waits-excluded currency (bar ≤19).
  Matrix 4.00 / non-matrix 5.72. Gap to llama's 7.16 = bf16→f16 operand conversion (1.20 CVT/touch) + V's
  transposed narrow stores + DCE sink.
- **`m16_census.py:325`'s 512-VGPR-per-SIMD term is RETIRED** (predicts 1 block for `m16_bf16rt`, driver
  recorded 3). Only the LDS / 196,608 reading reproduces all five recorded rows.
- **THE MAP IS SWAPPED** (this is the session's biggest save): `check_fattn_map.py` 26/26 on device shows
  lane *t*'s *l*th C value = `sum_k FIRST[2l + t/16][k] · SECOND[t%16][k]` — first operand's row lands in
  the **column**. The transcription from `mma.cuh:183-215` claimed the opposite. As written the port would
  have shipped **S[q][kv] instead of S[kv][q]** and folded every softmax down the wrong axis, silently, past
  a census that passed. Fix = operand swap: score leg `wmma(K_tile, Q_tile, acc)`, P·V leg
  `wmma(P_operand, Vt_tile, acc)`.
- **C→B witness `check_fattn_cb.py` 27/27**, 7 derived controls behave as derived. **Named blind spot**:
  `cb_no_shfl` (removing the `get_half2` butterfly) leaves the OUTPUT exactly right and only the operand dump
  wrong — the hardware reads each half-warp's own parity, so the butterfly only makes both parities right in
  both half-warps. Say that in the checker header. `cb_v_odd_kv_zero` confirms staged V's k-run is the
  contiguous 16 the form assumes.
- Re-census **after the operand swap, on the probe**: 9.719 unchanged, 4 blocks/CU unchanged
  (`prototype/attn_wmma/evidence/occupancy-fattn-after-swap.txt`).

## 4. Supervisor decisions the continuation must honour

1. **Route (A) — the ONLY approved routing.** `--attn-fattn 1` covers **rows above `MAXSC`=512 only** (the
   streaming regime that already calls `attn_combine`); one-pass rows keep `attn_slim_wave`. Chosen because
   (B) would push currently-pinned bit-exact rows into a streaming band they never had = **user-level band
   relaxation, not mine and not the worker's to spend**; (C) (staging scores for one-pass at M=16) is refuted
   by the LDS budget. If (iii) wins and (B) matters, it comes to the user with numbers.
2. **Inertness asserted, not assumed**: `--attn-fattn 1` vs `0` bit-identical for rows ≤512, and the launcher
   **echoes every fallback** instead of routing silently (the `--attn-tile` trap class).
3. **Covered fraction printed BEFORE the burst**: `fattn_rows=n/<N>` and the covered share of `attn` ms in the
   same log as the `ms`. Under (A) at 4096 lanes roughly the last ~87 % of rows move — that is my arithmetic,
   the script must print the real number. If only 70 % routes over, a 1.5× family cut needs ~2× on covered
   rows, and knowing that in advance is the difference between reading and misreading the result.
4. **Boundary tested both ways**: rows **511 and 513** are mandatory admission cases, and the derived
   must-catch set includes an injection only a straddling row can catch (§20.6 rule: a route no fixture
   reaches has no tests, including tests of the gate).
5. **Claim (iii) stands as written**: `attn` family cut **≥1.5× at 4096 lanes**, best-of-3 paired inside ONE
   lock hold, canary asserted in the same breath. If (i)+(ii) hold and (iii) fails ⇒ matrix class and round
   shape are both exonerated and **2.6 closes on evidence**, not on budget.

## 5. Remaining work, in order

**Worker (compile + CPU only, no milliseconds):**
1. On `wip-fattn-port`: finish route (A) — boundary routing, fallback echo, `attn_fattn=<0|1>` echo beside
   `attn_touch=`, one-time `ATTN_FATTN form=m16_fattn_score vgpr=… lds=… vtiles=…` banner. **`gate.py`'s
   default and `decode_run.cpp`'s move in one commit.**
2. **Re-census at the placement that actually ships** — do NOT carry 9.719/4 blocks over from the probe. The
   staged-Q question is the port's to decide and document: staged Q = 8 KiB of LDS (V aliased, 4 blocks/CU)
   vs register-resident Q + un-aliased V = 24 KiB ⇒ **2 blocks/CU**. Barrier count moves 2→3 per round at
   4 blocks/CU (unpriced by any census).
3. Admission sequence written (§5b): **cos/meanrel sized by a floor probe, never max-ULP** (a 16-bit operand +
   legal reordering passes any max-ULP vacuously; a chained dot's legal ndiff is 29–47 %). Expectation set in
   advance: **this form is NOT bit-identical above 512 lanes**; below it must be identical. Derived must-catch
   set incl. the **lane-half C→B parity injection the current butterfly form structurally cannot see**; the
   single fold shuffle; one of the 16 O tiles; `FATTN_KQ_MAX_OFFSET` (moves every P by ~1e-4, must NOT move
   argmax). Must-NOT-catch: a row with exactly one live lane; any partial-slot injection at lanes ≤ MAXSC.
   Cases generated from geometry, driver asserts `cases == |population|` (D40).
4. Make `check_fattn_map.py` + `check_fattn_cb.py` **permanent parts of admission**, not one-offs.
5. Burst script per §5c with the stale-binary canary (banner + `attn_fattn=` must differ between sides before
   any `ms` is trusted).

**Main agent (device window: blocking, zero subagents, one `gpu_lock` hold, never backgrounded):**
6. `prototype/issue_rate_probe/run_wmma_latency_window.sh` — **both** accumulator widths, ILP 1/2/4/8. Answers
   the f16-accumulator question with data (below).
7. Admission (§5b) then burst (§5c) → claim (iii). Optional in the same window: `SQ_LDS_BANK_CONFLICT` for
   the transposed V store (priced alternative = row-major V + 256 narrow A reads, **+0.75 slots/touch**), and
   realised occupancy by counter (`SQ_WAVE_CYCLES/SQ_BUSY_CYCLES`, max is **32 wf/CU**, not 64).
8. Report at the milestone either way; if it wins, the §4.4 gate work and the 512-row × 4096-lane fixture
   become mandatory, and (B) comes to the user with numbers.

## 6. The open f16-accumulator question (user asked, still unanswered by data)

`v_wmma_f16_16x16x16_f16` has **never been timed on this part** — 52.63 TFLOP/s is the **f32-C** form
(`prototype/issue_rate_probe/results.md:39`). llama.cpp splits 192/192 between the two and pays **358
spilled VGPRs** to hold f16-O over the whole KV span; that was a register escape, and §3.4 shows registers
do not bind us. Error is span-proportional: 4096 rows ⇒ ~256 sequential f16 adds per output element
(ε≈4.9e-4) ⇒ ~1e-3 typical, percent-scale worst case, vs negligible for f32. If the leg shows f16-C materially
faster, the design to adopt is the middle one: **f16 inside a KV tile, drained to f32 at the tile edge**
(~4 adds, not 256). Gates decide it under the standing numeric-format rule, and the instrument must be
cos/meanrel against pins — a max-ULP check would pass vacuously and we would "adopt" an error we never
measured.

## 7. Also closed / parked this session (so nobody re-makes them)

- **D60**: the "−156 slots/token" that priced the pointer-increment addressing item **never existed** — a flat
  ISA census bills the loop kernel's 203 once-per-launch slots per token. True: `rec_loop` flat +156 but
  **per-token −47**; all address arithmetic is **93 slots (6.1 %)**, delta −3. `recptr` came back **+9
  per-token (worse)** ⇒ ~0.17 pp of wave, below resolution ⇒ **no device window spent**. `token_bill.py` is
  the instrument. My brief's error is recorded: `cast=97 other=20 total=406` is `LaunchCounts::describe()`'s
  **launch-plan** bill (records/decode-step), a valid launch-count canary and **not** an instruction census.
- **`AGENTS.md`**: my trap row inserted at line 78 silently shifted `:122/:140/:173/:315`, and
  `citation_rot_check.py` reported **0 rotted on that commit** because its baseline is HEAD. Moved to an
  end-of-file section (`404a405,416`). **Guard: `wc -l` before/after + `diff <(git show <prev>:file) file`.**
- Commits: `af39933` D58 · `d592723`,`9748c15`,`67e00f7` D60 · `1080eb4` AGENTS fix · `d7efb5a`,`6dfd945`
  §28/D61 · `09fb87d` anatomy · `b24faea` D62 · `5666adc` `m16_fattn_score` · `d7057c4` wip port (branch).

## 8. Standing rules that bit us this session (do not re-learn them)

Device work is the main agent's only, **blocking, zero subagents, one lock hold**; workers write commands and
run correctness-only gates under ~60 s with command + wall + `util` + "no timing taken". `--ids` is
**comma-separated** (a space-separated list silently prices a 1-token wave). Never print the chat-template
end-of-turn token literally (refer to id 248046). `engine/performance-status.md` §1–§5 are line-cited:
**edits there must be line-count neutral**, append new sections at the end. A canary must be able to detect a
**stale binary**; a paired A/B against a stale build reads as a clean null. A check whose input list can be
empty cannot fail; a must-catch set must be **derived**, not hand-listed. Do not chain a python stdin heredoc
(`analysis.unsupported-heredoc`) — write the script to a file.

## 9. State after the second port worker (branch tip `0dca934`) — the window is next

Branch `wip-fattn-port` now: `60725fe` engine route (A) + one-warp blocks + ship census · `3dcf434` probe-vs-port
divergence marked in place · `74875d9` admission + WMMA latency leg written-not-run · `0dca934` cite re-points
(`citation_rot_check` = **0 rotted**). `engine/build.sh` = **0**. Nothing device-run: no admission, no burst.

- **Claim (i) is RED and stays red**: best placement 256 VGPR / **49 spill** / 72 B scratch / 16,384 B LDS /
  4 blocks/CU. `fattn_ship_census.py` is section 0a of admission and exits non-zero on any spill. D63 re-scoped
  zero-spill from precondition to reported cost; the flag is default **0** and only a measured ≥1.5× win flips it.
- **Placement that ships**: one warp per block (16 rows × 1 head, head into grid.x) — correct-by-construction.
- **Geometry caveat that must travel with any number**: 4 wf/CU, five times below §28's measured 12–20 band.
  `wf_per_cu` is printed for exactly this reason.
- **Rowsum defect witnessed**: half denominator ⇒ cos 1.000000, argmax unchanged on all 4096 rows, meanrel
  median 0.956. Admission gates on **meanrel**; cos and argmax are both blind to it (`fattn_rowsum_witness.py`).

**Next window (main agent, blocking, zero subagents, one `gpu_lock` hold), in this order:**
1. `prototype/issue_rate_probe/run_wmma_latency_window.sh` — f32-C vs f16-C, ILP 1/2/4/8. Answers the f16 question.
2. `bash prototype/attn_wmma/run_fattn_admission.sh` — refuses on the ship census, so expect it to stop at 0a
   unless the spill refusal is annotated as accepted-for-this-window; that annotation is my call to make and say.
3. `bash prototype/attn_wmma/run_fattn_burst_window.sh` — claim (iii): ≥1.5× cut of `attn` at 4096 lanes,
   paired best-of-3, canary (banner + `attn_fattn=` + `wf_per_cu`) asserted before any `ms` is read.
4. Optional same hold: `SQ_LDS_BANK_CONFLICT` on the transposed V store; realised occupancy by counter
   (`SQ_WAVE_CYCLES/SQ_BUSY_CYCLES`, max **32 wf/CU**).
