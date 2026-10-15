# Repo cleanup audit, part 1 — cited-but-untracked files (2026-10-16)

Scope set by the brief: every path cited by a tracked `*.md` or by a commit message that is **not**
tracked in git, split into (a) what should be committed, (b) what must not be, (c) the cited paths
that are simply gone. CPU-only pass: no build, no gate, no device command, no deletion, no edit of a
tracked document. Everything found here is staged or recommended, never committed.

## 1. Method and commands

`git status --porcelain` was clean except `engine/.pi/`, as briefed — that changed mid-pass: commit
`d7b2813` added `.pi/` to `.gitignore` and corrected the `AGENTS.md` sentence claiming the repo had no
`.pi/` at all, so the ignore list now covers the harness state and my own proposal for it (§5) is
moot. The audit is the reference scan, not a file walk: a file matters only because something tracked
says so.

```bash
python3 /tmp/audit_helper.py           # the brief's pass: 52 present-untracked-ignored, 13 MISSING
python3 /tmp/audit2.py                 # corrected extension set (see finding 1): 54 / 10
python3 /tmp/audit3.py                 # any path-shaped token, any extension (no extension filter)
python3 /tmp/cite.py /tmp/paths.json   # per-path citation count + file:line, docs and commit msgs
python3 /tmp/contentcite.py            # content-level citation: does a doc/commit quote this log?
python3 /tmp/sizes.py                  # stat -c%s per candidate, size-cap check
```

Method, in words:

1. Tokenize every path-shaped reference in the tracked markdown files (`git ls-files '*.md'`, 51
   files under `bench/`, `engine/`, `prototype/`, `tools/`, `specification/`, `devices/`, `speccheck/`
   and the root) and in all 394 commit messages (`git log --format=%s%n%b%n`).
2. Bucket each token: tracked / present-but-untracked-and-ignored / missing.
3. For every present-but-untracked candidate, record `stat -c%s`, the number of doc citations, the
   number of commit-message citations, and the exact `file:line` of the first citing line.
4. Check the citation at the **content** level too (`/tmp/contentcite.py`): a log whose printed
   numbers a commit message quotes verbatim is provenance even when its path is never named.

Two findings about the audit itself, both of which changed the file list:

1. **The original extension list produced two phantom MISSING entries and missed two real files.**
   The pattern's extension alternation had `o` before `out` and `json` before `jsonl`, and the
   preceding `[\w.\-/]*` backtracks, so `bench/band-history.jsonl` was recorded as the missing path
   `bench/band-history.json` and `progress/bench/evidence/p9/stage.out` as the missing path `tmp/p9/stage.o`. Correcting
   the alternation order gives **54** present-untracked-ignored (adds `progress/bench/evidence/p9/stage.out`,
   `progress/devices/evidence/q8/mixed_bw.err`) and **10** genuinely missing. The brief's `tmp/p9/stage*.o → NO` therefore
   referred to a file that does not exist; the file the doc cites is `progress/bench/evidence/p9/stage.out` (§4).
2. **A path citation is not the only kind of citation.** `progress/engine/evidence/prefill_mix_check-2026-10-13.log`
   carries **zero** path citations, but commit `40cb5d7` quotes its table verbatim ("in_proj_qkv
   1842/98304, gate and up 536/57344 each" — lines 14, 17, 18 of that log), and
   `progress/engine/evidence/prefill_mix_det-2026-10-13.log` holds the two hashes the same commit quotes
   (`1731f224b4015b07`, `ede2ef6da48c6e47`). Counting path mentions alone would have dropped the
   certificate the hipBLAS rounding argument rests on.

Deviations from the brief are listed in one place (§2d) rather than spread through the groups.

## 2. Staged for commit — groups 1-3: 47 files, 1,323,865 bytes (1.26 MiB)

This audit document is the 48th staged file and the fourth topical group (§6).

Counts here include the parent's commit-time decisions: the five §2e files were
accepted (four into group 3, `prefill_mix_det` into group 1), so group 1 is 8
files and group 3 is 23.

`docs` / `msgs` = number of citing lines in tracked `*.md` / occurrences in all commit messages.

### 2a. Group 1 — the cited device-measurement logs (8 files, 117,879 B; `msg1-logs.txt`)

| path | bytes | docs | msgs | why it is worth keeping |
|---|---:|---:|---:|---|
| `progress/bench/evidence/fp16_range_audit-2026-10-13.log` | 93,992 | 2 | 0 | `progress/2026-10-13/prefill-route-spike-2026-10-13.md:245` calls it "Full output:" of the fp16 range census. Re-making it needs `data/refs/p300` (5.3 GB, untracked by rule B) and `.venv`; the log is the only tracked-size artifact of that census |
| `progress/bench/evidence/prefill_gemm_shapes-2026-10-13.log` | 11,714 | 3 | 0 | the shape table quoted at `…spike:148,208`; the re-run line at `:427` reads the checkpoint shard, so the table is not reconstructible from the tree alone |
| `progress/bench/evidence/gemm_peak_check-2026-10-13.log` | 921 | 3 | 0 | the §4.3 rounding table, reproduced verbatim by a device run the doc says needs a held lock (`:311`, `:433`) |
| `progress/bench/evidence/gemm_peak_det-2026-10-13.log` | 5,128 | 2 | 0 | sole record of the determinism hashes `b30c25c0e7fd147a` / `48ad8cbb1e7ae419`, quoted verbatim in commit `ade0388` ("`--determinism` reproduces both recorded hashes") |
| `progress/bench/evidence/mfma_isa-2026-10-13.log` | 1,030 | 2 | 0 | compile-only ISA census behind a claim in `devices/` ("the compiler emitted no `v_mfma`"); cheap to re-make, but only against today's toolchain, so it is a dated witness |
| `progress/bench/evidence/mfma_probe-2026-10-13.log` | 1,691 | 2 | 0 | the device half of the same claim |
| `progress/engine/evidence/prefill_mix_check-2026-10-13.log` | 2,332 | 0 | 0 | cited by **content**, not path: commit `40cb5d7` quotes its per-shape bit counts. The `--check` certificate for the hipBLAS call form (§10.1's rounding question), device-run to re-make |
| `progress/engine/evidence/prefill_mix_det-2026-10-13.log` | 1,071 | 0 | 0 (content) | added at commit time: it holds the two mix hashes commit `40cb5d7` quotes verbatim, so it carries the same argument as the row above |

Not staged, and why: `prefill_mix_smoke` (0 citations of either kind), `fp16_audit_p300-2026-10-13.log` (93,799 B, the
superseded first run of the cited audit — it opens with a `RuntimeWarning: invalid value encountered
in cast` the cited log does not have), `dump_smoke.log`, `w1_bf16_rerun.log`,
`speccheck-final.log` / `speccheck-quick2.log` / `speccheck-run-154714.log` (reproducible with
`speccheck/run_all.py`, which is the point of that runner), `download.log`, `pip_tf.log` (0 B),
`pip_torch.log` (0 B).

### 2b. Group 2 — the cited `tmp/` scripts (16 files, 40,575 B; `msg2-scripts.txt`)

Every one is cited as the thing that *produced* a quoted number, and none can be re-made from
tracked content: they read the untracked `data/refs/` pins, the untracked gate logs, or a
`decode_run` binary that is itself build output. Largest is 7,167 B.

| path | bytes | docs | msgs | citing line (first) |
|---|---:|---:|---:|---|
| `progress/engine/evidence/d5_allowance_report.py` | 3,004 | 2 | 0 | `engine/results.md:2090` — "recomputes every figure in this table from the files on disk", so the table is unverifiable without it |
| `progress/engine/evidence/d2_dec300f_still_valid.py` | 2,159 | 1 | 0 | `engine/results.md:2182` — the 128-step dec300f re-check behind "the shipped table was cited rather than re-made" |
| `progress/engine/evidence/d6_final_pass.py` | 2,887 | 1 | 0 | `engine/results.md:2161` — "the final pass is one command" |
| `progress/engine/evidence/d9_sizing/all.sh` | 321 | 1 | 0 | `engine/results.md:4273` — the eight locked device runs, in order |
| `progress/engine/evidence/decode_run_library.sh` | 71 | 3 | 0 | `progress/2026-10-16/performance-status-2026-10-16.md:989`, `engine/results.md:4476,4479` — the only bridge from `gate.py --binary` to the library route; the file that proves the library gate was ever runnable |
| `progress/engine/evidence/gate_ab.sh` | 3,132 | 1 | 1 | `engine/results.md:4427` — "Reproduce: `progress/engine/evidence/gate_ab.sh <hand\|library> <fixture>`" |
| `progress/engine/evidence/gemm_probe.cpp` | 5,238 | 1 | 1 | `engine/results.md:4339` — the probe that localised the wrong-tile defect (cos 0.42) |
| `progress/engine/evidence/q10/decode_run_contig.sh` | 274 | 1 | 0 | `engine/results.md:3739`; see §5 — the same line says "not committed", which this commit makes stale |
| `progress/engine/evidence/q6/proj_rpb32.cpp` | 7,167 | 2 | 0 | `engine/results.md:725,727` — one half of the rpb8↔rpb32 A/B, i.e. the source of the historical ledger row |
| `progress/engine/evidence/q6/proj_rpb8.cpp` | 6,152 | 2 | 0 | `engine/results.md:724,726` — the other half, recovered from `git show bc61dce~1` |
| `progress/bench/evidence/tilefix/repeat_ids.py` | 785 | 1 | 0 | `progress/2026-10-16/performance-status-2026-10-16.md:1094` — how the 3072-lane prompt was built ("repeats a real 1024-token prompt three times") |
| `progress/bench/evidence/w9.sh` | 1,289 | 1 | 0 | `progress/2026-10-14/p9-chunked-scan-2026-10-14.md:435` — the baseline invocation a priced model is judged against |
| `progress/engine/evidence/q2_session.py` | 2,733 | 1 | 0 | `engine/results.md:1994` — the feed+gen parity run (32/32) |
| `progress/prototype/evidence/stub_decode_run.py` | 2,105 | 1 | 0 | `prototype/attn_decode/results.md:1105` — the §18.1 coverage harness; the only thing that made the prefill-row gate testable |
| `progress/bench/evidence/p9/probe_pinwidth.py` | 2,399 | 1 | 0 | `progress/2026-10-14/p9-chunked-scan-2026-10-14.md:52` — "Pin widths are measured, not assumed" |
| `progress/engine/evidence/probe_bf16_logits.py` | 859 | 0 | 1 | commit-message citation only (`max abs difference 0 and cos exactly 1.0 (probe: progress/engine/evidence/probe_bf16_logits.py)`); named in the brief, kept as a probe rather than scratch |

### 2c. Group 3 — the cited `tmp/` evidence (23 files, 1,165,411 B; `msg3-evidence.txt`)

| path | bytes | docs | msgs | why |
|---|---:|---:|---:|---|
| `progress/bench/evidence/q10/ids1024.txt` | 4,677 | 25 | 0 | the id fixture **25 documented commands read** (`progress/2026-10-14/HANDOFF-2026-10-14.md:93`, `progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md:240,244`, `progress/2026-10-14/p9-phase1-2026-10-14.md:257-259`, …). Commit `8c81e21` re-took the headline after the purge specifically to prove these ids still mean what they said |
| `progress/bench/evidence/q10/ids3072.txt` | 14,031 | 3 | 0 | the 3072-lane fixture, `progress/2026-10-14/p9-route-a-2026-10-14.md:204-206` |
| `progress/engine/evidence/engine_gate/sampled.txt` | 2,840 | 1 | 0 | `engine/results.md:285`; the file whose *empty* state has twice masqueraded as a numerics diff (commit `5f05313`) — keeping it is keeping the negative control |
| `progress/bench/evidence/rc_a1/a1.json` | 26,288 | 4 | 0 | `progress/2026-10-14/route-c-audit-2026-10-14.md:42,43,94,327` — per-shape floors in machine-readable form, the input to A3 |
| `progress/bench/evidence/rc_a1/a1.txt` | 7,580 | 1 | 0 | the same census as prose, `route-c-audit:93` |
| `progress/bench/evidence/rc_a2/all.txt` | 244,276 | 2 | 0 | `route-c-audit:150,188` — "1813 lines, every dumped row"; carries the per-row offenders (e.g. `chunk64b` row 70, both ids and both gaps) quoted in prose. **2.4× the brief's 100 KB cap**, kept because the brief names it and the rows exist nowhere else |
| `progress/engine/evidence/route_dec300f.json` | 61,476 | 2 | 0 | `engine/README.md:162`, `engine/results.md:2180` — the table `route_allowance.py --table` reads; the dec300f leg was never re-priced on CPU, so this *is* its price |
| `progress/engine/evidence/route_dec64f.json` | 24,321 | 1 | 0 | `engine/results.md:1296` |
| `progress/engine/evidence/route_dec64f.ra.json` | 31,200 | 1 | 0 | `engine/results.md:2105` — the file whose `exit=2` refusal is the evidence that the allowance check bites |
| `progress/engine/evidence/route_margin_dec64f.json` | 92,489 | 1 | 0 | `engine/results.md:2559` (the D-6 margin pass) |
| `progress/engine/evidence/margin/library_dec300f.log` | 316,659 | 2 | 1 | `progress/2026-10-16/performance-status-2026-10-16.md:988`, `engine/results.md:4478` — the log whose header proved the "library" gate had actually run the hand route. **3.2× the cap**; the misattribution finding is unrecoverable without it |
| `progress/engine/evidence/gate_dec64f.log` | 72,790 | 5 | 0 | the gate run itself: `engine/README.md:246`, `engine/results.md:1157,1158,1294,1296` all read it as *input* to a CPU probe |
| `progress/engine/evidence/gate_dec64t.log` | 72,130 | 2 | 0 | same, `engine/results.md:1574,1575` |
| `progress/engine/evidence/d6_gate_dec64f.log` | 91,465 | 1 | 0 | same, `engine/results.md:2558` |
| `progress/devices/evidence/q8/mixed_bw.log` | 6,742 | 6 | 0 | `devices/strix-halo-mixed-traffic.md:175,178,233,257,259,294` — the named probe file `tools/step_traffic_model.py --probe-file` reads; the whole mixed-traffic device section prices itself off these lines |
| `progress/devices/evidence/q8/mixed_bw2.log` | 20,516 | 1 | 0 | the corrected second sweep, `…mixed-traffic.md:330` |
| `progress/devices/evidence/q9/rows.log` | 6,625 | 2 | 0 | "Raw lines: `progress/devices/evidence/q9/rows.log`", `…mixed-traffic.md:546,595` |
| `progress/engine/evidence/q10/md5_A_int.txt` | 469 | 2 | 0 | the contig-vs-interleaved identity (`engine/results.md:3731-3733`, "no output: 10 files identical") — the witness cannot be re-made without the dumps it hashed |
| `progress/engine/evidence/q10/md5_A_con.txt` | 469 | 2 | 0 | its pair |

### 2d. Deviations from the brief, in one place

| deviation | files | basis | undo |
|---|---|---|---|
| committed although over the stated 100 KB cap | `progress/bench/evidence/rc_a2/all.txt` (244,276), `progress/engine/evidence/margin/library_dec300f.log` (316,659) | explicitly enumerated in the brief, whose only stated skip condition was "missing"; 561 KB together, 45 % of the staged bytes | `git restore --staged progress/bench/evidence/rc_a2/all.txt progress/engine/evidence/margin/library_dec300f.log` |
| committed although cited by content, not by path | `progress/engine/evidence/prefill_mix_check-2026-10-13.log` | enumerated in the brief; commit `40cb5d7` quotes its numbers verbatim | `git restore --staged progress/engine/evidence/prefill_mix_check-2026-10-13.log` |
| enumerated in the brief but does not exist | `tmp/p9/stage*.o` | the audit's `.o`/`.out` truncation; the real cited file is `progress/bench/evidence/p9/stage.out` (§4, §2e) | — |
| added at commit time, by the parent, from §2e | `progress/bench/evidence/p9/stage.out`, `progress/devices/evidence/q8/mixed_bw.err`, `progress/bench/evidence/rc_prompt.ids`, `progress/bench/evidence/rc_a1.log`, `progress/engine/evidence/prefill_mix_det-2026-10-13.log` (69,439 B) | each meets the brief's own test for its group (cited evidence under 100 KB); they were absent from the brief only because its extension set stopped at `.log`/`.json` | `git restore --staged progress/bench/evidence/p9/stage.out progress/devices/evidence/q8/mixed_bw.err progress/bench/evidence/rc_prompt.ids progress/bench/evidence/rc_a1.log progress/engine/evidence/prefill_mix_det-2026-10-13.log` |

### 2e. Cited, present, untracked — found by this pass, **committed at the parent's word** (69,439 B)

These were missed by the brief's list only because its extension set stopped at `.log`/`.json`.
Each is cited evidence under 100 KB, so each meets the brief's own test for its group, and the
parent accepted all five: the four `tmp/` files went into group 3, `prefill_mix_det` into group 1.

| path | bytes | docs | msgs | citation |
|---|---:|---:|---:|---|
| `progress/bench/evidence/p9/stage.out` | 43,637 | 1 | 0 | `progress/2026-10-14/p9-chunked-scan-2026-10-14.md:114` — "Full text in `progress/bench/evidence/p9/stage.out`" is the §1.3 stage table's full text |
| `progress/devices/evidence/q8/mixed_bw.err` | 10,494 | 4 | 1 | `devices/strix-halo-mixed-traffic.md:175,182,230,295` — the stderr half of the committed `mixed_bw.log`, and the corrections section quotes it directly |
| `progress/bench/evidence/rc_prompt.ids` | 4,767 | 2 | 0 | `progress/2026-10-14/route-c-audit-2026-10-14.md:36,40` — an id fixture a documented command reads, same class as `ids1024.txt` |
| `progress/bench/evidence/rc_a1.log` | 9,470 | 1 | 0 | `progress/2026-10-14/route-c-audit-2026-10-14.md:126` — "`poison_survivors=0` on every floor and injection bin in `progress/bench/evidence/rc_a1.log`" |
| `progress/engine/evidence/prefill_mix_det-2026-10-13.log` | 1,071 | 0 | 0 (content) | holds the two hashes commit `40cb5d7` quotes verbatim; the sibling of the committed `prefill_mix_check` |

## 3. Deliberately not committed

| category | size | reason |
|---|---:|---|
| `data/` — pinned HF dumps (`data/refs/*/…npy`), `data/eng/*.npz`, `data/corpus.txt`, `data/long.txt` | 5.3 GB | decision B, and `.gitignore` excludes `data/`, `*.npy`, `*.npz` deliberately. Nine cited paths live here (`data/refs/chunk64b/logits_last.npy`, `data/refs/smoke/pre_final_hidden.npy`, `data/refs/{dec64t,prefix24dec8,smoke}/manifest.json`, `data/refs/corpus48-margins.json`, `data/corpus.txt`, `data/long.txt`, `data/eng/smoke_chunk.npz`) — cited-but-absent is their intended state: `tools/reference_dump.py` re-pins them and `engine/gate.py` re-verifies them. Committing them would also break the "regenerable, large" contract the ignore rule records |
| `models/` — incl. the cited `models/Qwen3.5-0.8B/config.json` | 1.7 GB | decision B; a hub download, and `.gitignore` names `models/` and `*.safetensors` |
| `.venv/` | — | environment, re-created by a documented command; also the home of the `transformers` module cited in `specification/ASSUMPTIONS.md` (§4 row 5) |
| `engine/build/`, `tools/build/` — `decode_run` (622,672, cited 16×), `loader_test`, `gemm_peak`, `mfma_probe`, `mixed_bw` (cited 21×), `route_c_stages` | 1,670,728 B | decision B: build output. The citations name these paths as the *command* in a reproduce block, never as an artifact, so keeping them out of git costs nothing a doc claims. Rebuilt by `tools/build.sh` / `engine/build.sh`, and `.gitignore` already names the per-op binaries exactly after the `prototype/**/<op>` directory trap |
| `prototype/<op>/<op>` and `tmp/` A/B binaries — `prototype/gdn_decode/gdn_decode` (56,952, cited 7×), `prototype/glue/glue` (247,944), `tmp/q6/decode_run_rpb8` (482,848, cited 3×), `tmp/q10b/decode_run_old` (529,816), `tmp/q10/decode_run_contig` (529,832) | ~1.4 MB | decision B. Commit `8c81e21` kept them on disk as "the only built evidence for historical ledger rows"; keeping them on disk but out of git is consistent, because the rebuild inputs are now committed — `progress/engine/evidence/q6/proj_rpb8.cpp` and `progress/engine/evidence/q6/proj_rpb32.cpp` are the two A/B sources and `engine/results.md:724-727` spells out the `cp` + `engine/build.sh` recipe that turns them into those binaries |
| `*.o` anywhere — including **`mlp_gemv.hip-hip-amdgcn-amd-amdhsa-gfx1151.o` at the repo root** (73,488 B) | 72 KB | an accidental `hipcc` device-object artifact of the `prototype/mlp_gemv` build, left in the working-tree root on 2026-10-13 10:20 and matched by two separate `*.o` lines in `.gitignore`, which is why nobody noticed it. **Recommend `rm mlp_gemv.hip-hip-amdgcn-amd-amdhsa-gfx1151.o`.** Not deleted in this pass, and not committed |
| `engine/.pi/subagents.json` (596 B, cited 1×) | 596 B | harness-written config; `AGENTS.md:140`, `progress/2026-10-14/HANDOFF-2026-10-14.md:143` and `progress/2026-10-14/speedup-routes-2026-10-14.md:337` all say keep it out of commits. Commit `d7b2813`, landed during this pass, put `.pi/` in `.gitignore`, so it is now ignored rather than merely untracked — the right outcome, and it closes the one entry `git status` used to show dirty |
| uncited scratch under `tmp/` and `logs/` | remainder of 83 MB | no doc or message refers to them, so no claim depends on them. `tmp/` is 83 MB, unchanged since the purge commit `8c81e21`; largest are `q10b` 22 MB, `q10` 18 MB, `q6` 4.7 MB, `pre-route` 4.4 MB |

## 4. Cited paths that do not exist — correction table

The 13 tokens the brief's pass reported as missing become **10 genuinely missing** plus 3 audit
artifacts (rows 1-3). Nothing below was edited; the "disposition" column is a proposal.

| # | referenced path | exact location(s) in docs / msgs | class | disposition |
|---|---|---|---|---|
| 1 | `bench/band-history.json` (docs=8, msgs=4) | `progress/2026-10-12/plan-2026-10-12.md:239`, `progress/2026-10-12/plan-prefill-2026-10-12.md:14`, `progress/2026-10-14/speedup-routes-2026-10-14.md:332`, `engine/README.md:182`, `engine/results.md:3256,4016,4409`, `prototype/attn_decode/results.md:377` | **audit artifact, no doc error** | Every one of those lines writes `bench/band-history.jsonl`, which **is tracked** (8,575 B, policed by `tools/band_history.py --check`). The `.json`/`.jsonl` truncation is in the scanner, not the prose. No change |
| 2 | `tmp/p9/stage.o` (docs=1) | `progress/2026-10-14/p9-chunked-scan-2026-10-14.md:114` | **audit artifact** | The doc writes `progress/bench/evidence/p9/stage.out` — present, 43,637 B, untracked. → recommend committing it (§2e); no doc change |
| 3 | `engine/prefill.h/.cpp` (docs=1) | `progress/2026-10-12/plan-prefill-2026-10-12.md:367` (`engine/prefill.{h,cpp}`) | **audit artifact** | Brace shorthand for two files that both exist and are tracked. No change |
| 4 | `engine/specimpl.py` (msgs=1) | commit `347ed54`'s message ("`engine/specimpl.py` implements specification/02..07 directly from the spec"); no tracked doc says it | **(i) stale name, moved** | → **`speccheck/specimpl.py`**. Git recorded it: `R083 engine/specimpl.py → speccheck/specimpl.py`. Commit messages are immutable history, so nothing to fix outside this table; all docs already say `speccheck/specimpl.py` (`AGENTS.md:172`, `README.md:186`, `bench/gate-audit.md:40`) |
| 5 | `speccheck/run.py` (docs=1) | `prototype/gdn_gated_norm/results.md:123` — "`speccheck/run.py --keep-f32-scalars` (the A-02 A/B knob)" | **(i) stale name, wrong file** | → **`speccheck/specimpl.py --keep-f32-scalars`**: that flag is specimpl's own (`speccheck/specimpl.py:526`); the runner is `speccheck/run_all.py`, whose only flags are `--python/--quick/--corpus/--floor`. Same stale line is copied into the untracked snapshot `tmp/pre-route/prototype/gdn_gated_norm/results.md:123` |
| 6 | `prototype/gdn_chunk/run.py` (docs=1) | `progress/2026-10-14/p9-phase1c-2026-10-14.md:452` — "check with `grep -c gpu_lock prototype/gdn_chunk/run.py` before holding the lock around it" | **(i) stale name, never existed** | → **`prototype/gdn_chunk/phase1.py`**, or the glob `prototype/gdn_chunk/*.py`. That dir holds `chunk_scan.py`, `gdn_chunk_kernels.h`, `kernel_mirror.py`, `kernel_price.py`, `phase1.py`, `spike.py` — no `run.py`. The sentence's *conclusion* survives the rename: `grep -c gpu_lock` is 0 for all five scripts, so it is indeed not a self-locking driver |
| 7 | `engine/gen_watch.py` (docs=1) | `engine/results.md:496` — "`engine/gen_watch.py --all` is already the comparator" | **(iii) never existed — a stale claim, not a lost file** | No such path in any of 394 commits (`git log --all -- '*gen_watch*'` is empty). The **same sentence's other two names are also absent**: `decode_run` has `--time-stages`, not `--dump-stages` (`engine/decode_run.cpp`'s flag list), and there is no `stage_pairs` anywhere in `speccheck/`. So the sentence describes a stage-dump→pins comparator that was never wired; its own tail — "nothing currently connects the first to the last" — is the only true part. Nearest real comparators: `engine/curve_probe.py`, `engine/pin_bits_probe.py` |
| 8 | `models/qwen3_5/modeling_qwen3_5.py` (docs=1) | `specification/ASSUMPTIONS.md:4` | **(i) shorthand for an external path** | Not a repo file: it is the installed library module, spelled correctly two lines away in `specification/README.md:10` as `transformers/models/qwen3_5/modeling_qwen3_5.py`; on this box it is `.venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` (exists). Suggest prefixing `transformers/` in ASSUMPTIONS.md |
| 9 | `data/refs/p64/prefill_state_mismatch.json` (docs=2) | `progress/2026-10-13/p3-boundary-fixtures-2026-10-13.md:94`, `progress/2026-10-12/plan-prefill-2026-10-12.md:445` | **not an error** | Both sentences say the file "still does not exist". A planned fixture that was never produced, correctly reported; `data/` is excluded by rule B regardless |
| 10 | `tmp/form1024/ship/wave_carry_deferred.json`, `tmp/form1024/tile/wave_carry_deferred.json` (docs=1 each, same line) | `prototype/attn_decode/results.md:1130` (the §18 hand-off diff) | **(ii) transient sidecar, deleted by the recorded purge** | Written by `engine/gate.py --scope form` (`engine/gate.py:710` — "every one of them (pass or fail) goes to `wave_carry_deferred.json` beside the run's dumps") and removed by `8c81e21`. Regenerable, but only by the two device gate runs at `…results.md:1122-1126`. The §18 verdicts they carried are already in prose (`:1089`, `:1104`), and later sidecars from the same route survive at `tmp/tilefix1024f/ship{0,1}/`, `tmp/scope_form/`, `tmp/f/*` |
| 11 | `tmp/gdna/rocprof.txt` (docs=1) | `progress/2026-10-14/p9-route-a-2026-10-14.md:213` (`rocprofv3 --kernel-code-object-code -i tmp/gdna/rocprof.txt …`) | **(ii) input to be authored at run time, never created** | Not lost evidence: the same block notes "counter file needs: SQ_INSTS_VALC, SQ_INSTS_LDS, SQ_INSTS_VMEM, SQ_IO_TIME", i.e. the operator writes it before the counter run — which route A never reached (it died at 112.04 ms, commit `821cec6`). Same status as a `<gate log>` placeholder |
| 12 | `tmp/route_wide520f.json` (docs=2, msgs=2) | `engine/results.md:3251,3252` | **(iii) genuinely lost, but cheap to re-make on CPU** | The one file holding the wide520f margin price behind `results.md:3255` ("worst engine/route 1-cos ratio 2.11916 at step 17 → margin 2.12"). Two things soften it: the applied bands are recorded in tracked `bench/band-history.jsonl` (2 `wide520f` entries, decision D-2), and the input gate log survived untracked — `tmp/gate_wide520f_after.log` (116,010 B) — so `engine/route_price.py --fixture wide520f --engine-log tmp/gate_wide520f_after.log --out tmp/route_wide520f.json` re-makes it, CPU only. Recommended: re-make it and commit it with that gate log |

Summary of the 10: **4 stale names** (rows 4, 5, 6, 8) where a correct path exists, **4 transient or
never-created scratch paths** (rows 9, 10×2, 11), **1 genuinely lost but CPU-re-derivable** (row 12),
**1 never-built tool whose sentence is the bug** (row 7).

## 5. Open items

* **Stray compiler artifact.** `mlp_gemv.hip-hip-amdgcn-amd-amdhsa-gfx1151.o`, 73,488 B, repo root.
  Not a fixture, not a gate artifact: `hipcc`'s device-side object from the `prototype/mlp_gemv`
  build, orphaned in the root and hidden by `*.o`. **Deleted by the parent in this pass** — build
  output, regenerable by `prototype/mlp_gemv/build.sh`, cited by nothing.
* **`engine/.pi/`** (one harness-written file, `subagents.json`, 596 B) — resolved by commit
  `d7b2813` during this pass: `.pi/` is now ignored at any depth, with the caveat recorded in the
  ignore comment that a future authored `.pi/settings.json` must become an explicit allowlist rather
  than grow a negation. Nothing left to do; listed so this audit does not appear to have missed it.
* **Empty directories `engines/` and `thinking/`** — both empty, both ignored (`engines/` for cloned
  reference engines, `thinking/` for cross-session notes). Git cannot track a directory, so there is
  nothing to commit and no diff to clean; `rmdir` would be cosmetic. Left alone. `engines/` being
  empty is also worth noting: no cloned reference engine is present on this box, which is what the
  `[FUTURE]` passages in `devices/` assume.
* **`tmp/` is 83 MB**, unchanged since the purge (`8c81e21`), and the biggest consumers are
  `q10b` (22 MB), `q10` (18 MB), `q6` (4.7 MB), `pre-route` (4.4 MB), `d9_sizing` (3.1 MB),
  `margin` (2.7 MB). Nothing here grows it. Of that 83 MB, this audit moves 1.2 MB into git and
  recommends another 68 KB; the rest is either the built A/B binaries `8c81e21` deliberately kept or
  uncited work dirs.
* **A doc line this commit makes stale**: `engine/results.md:3739` writes
  "`progress/engine/evidence/q10/decode_run_contig.sh`, not committed". Group 2 commits it. Rewrite the parenthetical in
  the next docs pass (not edited here, per the brief).
* **The five files in §2e** — accepted and committed with groups 1 and 3.
* **Second half of the cleanup** (not in scope here): part 2 should decide the empty-body commits the
  audit helper also counts (394 commits, 195 of them with an empty body), which were out of scope for this pass.

### Gitignore proposal — argued, NOT applied

`git add -f` works, but it is invisible: nothing in `.gitignore` records that `logs/*.log` and a list
of `tmp/` files are *intentional* content, so the next session repeats this audit or (worse) a purge
deletes a cited artifact again — which is exactly how `tmp/route_wide520f.json` and the
`wave_carry_deferred.json` sidecars were lost. The `.pi/` half of this proposal became moot mid-pass
(`d7b2813` added `.pi/` with its own comment block), so what remains is the `logs/` half. Options,
cheapest first:

```gitignore
# --- Cited device-measurement logs: provenance for quoted numbers ------------------
# `logs/` itself is not ignored, only `*.log`, so the negation lands; re-ignore the
# noise AFTER the negation. Cited set and reasons:
# progress/2026-10-16/cleanup-audit-2026-10-16.md section 2a.
!logs/*.log
logs/pip_*.log
logs/download.log
```

* **(a) Recommended: the `!logs/*.log` block above.** Four lines, honest about why, and it makes the
  next audit a `git status` instead of a path-token sweep over 394 commit messages. It does not apply
  to the already-tracked files — once staged they are safe either way — it applies to the next log a
  doc decides to cite.
* **(b) Do not try the same for `tmp/`.** Git cannot re-include a file whose parent directory is
  excluded, and `.gitignore` excludes `tmp/` as a directory. Making `!/progress/bench/evidence/q10/ids1024.txt` work
  means rewriting `tmp/` as `tmp/*` and then negating per subdirectory — the same
  directory-vs-file class of bug that already swallowed `prototype/**/<op>` once (see the comment
  block above the binary lines in `.gitignore`). Keep `tmp/` blunt and keep using `-f` there.
* **(c) Reject any broad negation** (`!tmp/**`, `!*.log`): it would surface the 21,816 present-but-
  untracked files — the 5.3 GB of `data/` fixtures and the built binaries among them — into
  `git status`, which inverts the point.
* **(d) Alternative to (a), zero diff:** record the rule in `AGENTS.md` instead — "`tmp/` and `logs/`
  hold cited provenance; `git add -f` when you cite it, never `git clean` it". If (a) is considered
  too much ignore-file surface, this is the fallback; doing neither leaves the next session to
  re-derive this table.

## 6. What was staged

`git add -f`, in three topical groups plus this document; **no `git commit` was run**.

| staged group | files | bytes | message file |
|---|---:|---:|---|
| group 1 — `logs/` | 7 | 116,808 | `/tmp/cleanup-msgs/msg1-logs.txt` |
| group 2 — `tmp/` scripts | 16 | 40,575 | `/tmp/cleanup-msgs/msg2-scripts.txt` |
| group 3 — `tmp/` evidence | 19 | 1,097,043 | `/tmp/cleanup-msgs/msg3-evidence.txt` |
| group 4 — this audit | 1 | (this file) | `/tmp/cleanup-msgs/msg4-audit.txt` |

