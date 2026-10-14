# `progress/` — the archive of finished documents and committed evidence

Created 2026-10-16 by the two-part cleanup. Part 1
(`progress/2026-10-16/cleanup-audit-2026-10-16.md`)
settled which cited files were missing and committed them; part 2
(`progress/2026-10-16/structure-proposal-2026-10-16.md`) surveyed what a reorganization would
cost.
The user overrode part 2's leave-it verdicts: the dated documents and the committed
evidence **do** move here. This file is the index, the naming rule, and the crosswalk.

## The rule

1. **A document moves here when it stops being a live instruction.** A plan, audit, review,
   diagnosis or handoff that has produced its verdict is a record, not a command list any
   more, so it stops sitting next to the live files (`bench/` keeps exactly one ledger,
   `bench/performance-status.md`, plus the two undated docs and the runner — see below).
2. **Nothing in `progress/` is ever edited afterwards, except to fix a broken path.**
   A verdict is not re-litigated here and a dated document is not silently updated: if a
   conclusion changes, the new decision is written as a new dated document (or in the
   `results.md` journal that owns the phase) and it cites the old one. The only edit this
   tree ever receives is a path rewrite, and it must be a same-line substitution, so that
   `file:NNNN` cites in the journals keep pointing at the right line.
3. **Naming.** Date folders are `progress/<YYYY-MM-DD>/`, one per day, and the filename is
   kept **unchanged** so that a reader following a citation from an old commit message or a
   journal line still finds the file by the name it was cited under. Time or period goes in
   the filename (`<name>-2026-10-14b.md`), never in a folder name: two documents on one day
   are rare, and the folder is the part a citation must stay stable against.
4. **Evidence lives at `progress/<owner>/evidence/<path-without-its-old-prefix>`.** The
   owner is the top-level folder whose docs cite the file most — see the ladder below.
   `logs/<name>.log` keeps its basename; `tmp/<sub>/<name>` keeps `<sub>/<name>`.
5. **`progress/` is not scratch.** `tmp/` and `logs/` still are, and are still ignored;
   nothing tracked lives under them any more, so `rm -rf tmp/` can no longer delete
   provenance. A re-run whose command line names a file listed below now writes into
   `progress/` — if you want the committed copy byte-identical, redirect the re-run into
   `tmp/` yourself.

## What stayed live (not archived)

| path | why it is live |
|---|---|
| `bench/performance-status.md` | the ledger: appended to, 82 doc lines cite it |
| `bench/build-queue.md`, `bench/gate-audit.md` | undated, still open |
| `bench/run.py`, `bench.sh`, `testdata/` | the CPU speed harness, untouched by this pass |
| `bench/band-history.jsonl` | data file, default of `tools/band_history.py --check` |
| `bench/corpus-freerun-2026-10-12.json` | data file, 4 doc lines read it |
| `engine/results.md`, `prototype/*/results.md` | the journals: live append-only record; this pass made same-line path fixes only, no line moved |
| `engines/` | unchanged: cloned reference engines, gitignored, absent on this box |

## Moves: dated one-shot documents (18)

`bench/<name>.md` -> `progress/<YYYY-MM-DD>/<name>.md`. The count columns are citation
lines in the **pre-move** tree: `md` counts every tracked `*.md`, `md*` excludes the two
catalogue docs that enumerate every path, `code` counts tracked non-`*.md` files. A file's
own self-naming lines are excluded. Every one of these occurrences was rewritten, in the
catalogue docs included.

| old | new | md | md* | code |
|---|---|---:|---:|---:|
| `bench/commit-review-2026-10-11.md` | `progress/2026-10-11/commit-review-2026-10-11.md` | 3 | 2 | 3 |
| `bench/review-2026-10-11.md` | `progress/2026-10-11/review-2026-10-11.md` | 2 | 2 | 0 |
| `bench/plan-2026-10-12.md` | `progress/2026-10-12/plan-2026-10-12.md` | 12 | 10 | 13 |
| `bench/plan-prefill-2026-10-12.md` | `progress/2026-10-12/plan-prefill-2026-10-12.md` | 13 | 8 | 14 |
| `bench/p3-boundary-fixtures-2026-10-13.md` | `progress/2026-10-13/p3-boundary-fixtures-2026-10-13.md` | 3 | 2 | 0 |
| `bench/prefill-route-spike-2026-10-13.md` | `progress/2026-10-13/prefill-route-spike-2026-10-13.md` | 2 | 1 | 4 |
| `bench/plan-route-c-audit-2026-001.md` | `progress/2026-10-13/plan-route-c-audit-2026-10-13.md` | 2 | 1 | 2 |
| `bench/attn-wave-diagnosis-2026-10-14.md` | `progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md` | 5 | 5 | 4 |
| `bench/HANDOFF-2026-10-14.md` | `progress/2026-10-14/HANDOFF-2026-10-14.md` | 6 | 2 | 1 |
| `bench/p9-chunked-scan-2026-10-14.md` | `progress/2026-10-14/p9-chunked-scan-2026-10-14.md` | 7 | 3 | 2 |
| `bench/p9-phase1-2026-10-14.md` | `progress/2026-10-14/p9-phase1-2026-10-14.md` | 6 | 5 | 10 |
| `bench/p9-phase1c-2026-10-14.md` | `progress/2026-10-14/p9-phase1c-2026-10-14.md` | 2 | 1 | 5 |
| `bench/p9-route-a-2026-10-14.md` | `progress/2026-10-14/p9-route-a-2026-10-14.md` | 3 | 1 | 6 |
| `bench/plan-p9-chunked-scan-2026-10-14.md` | `progress/2026-10-14/plan-p9-chunked-scan-2026-10-14.md` | 2 | 2 | 1 |
| `bench/route-c-audit-2026-10-14.md` | `progress/2026-10-14/route-c-audit-2026-10-14.md` | 12 | 6 | 10 |
| `bench/speedup-routes-2026-10-14.md` | `progress/2026-10-14/speedup-routes-2026-10-14.md` | 5 | 3 | 1 |
| `bench/cleanup-audit-2026-10-16.md` | `progress/2026-10-16/cleanup-audit-2026-10-16.md` | 2 | 0 | 3 |
| `bench/structure-proposal-2026-10-16.md` | `progress/2026-10-16/structure-proposal-2026-10-16.md` | 0 | 0 | 0 |

Totals: **18 documents**, 87 `md` lines and 79 code lines.

The one malformed name in the set, `bench/plan-route-c-audit-2026-001.md`, carried a
sequence number where the date belongs.
`progress/2026-10-14/route-c-audit-2026-10-14.md:12` records that it was planned
on 2026-10-13, so it moved as
`2026-10-13/plan-route-c-audit-2026-10-13.md` — the only rename in the set.

## Moves: committed evidence (47)

47 files, 1.26 MiB — the 8 `logs/*.log` and the 39 `tmp/` files that part 1 committed as
cited provenance. Owner ladder (the ruling of 2026-10-16):

1. majority of doc-citation lines, **catalogue docs excluded** (the `md*` column) —
   `progress/2026-10-16/cleanup-audit-2026-10-16.md` and
   `progress/2026-10-16/structure-proposal-2026-10-16.md` each
   enumerate every path, so counting them would make `bench` the owner of everything by
   construction; part 1's own table did not count them either, because it was untracked
   while it counted;
2. no majority (a tie) or zero doc cites -> the folder that owns the tool which consumes
   the file, or the folder whose claim it certifies — a judgement, so the **why** column is
   mandatory on those rows;
3. `misc` only if neither names anything. **No file needed it.**

| old | new | owner | rung | md | md* | code | why |
|---|---|---|---:|---:|---:|---:|---|
| `logs/fp16_range_audit-2026-10-13.log` | `progress/bench/evidence/fp16_range_audit-2026-10-13.log` | bench | 1 | 3 | 2 | 0 | progress/2026-10-13/prefill-route-spike-2026-10-13.md cites it |
| `logs/gemm_peak_check-2026-10-13.log` | `progress/bench/evidence/gemm_peak_check-2026-10-13.log` | bench | 1 | 4 | 3 | 0 | progress/2026-10-13/prefill-route-spike-2026-10-13.md cites it |
| `logs/gemm_peak_det-2026-10-13.log` | `progress/bench/evidence/gemm_peak_det-2026-10-13.log` | bench | 1 | 3 | 2 | 0 | progress/2026-10-13/prefill-route-spike-2026-10-13.md cites it |
| `logs/mfma_isa-2026-10-13.log` | `progress/bench/evidence/mfma_isa-2026-10-13.log` | bench | 1 | 3 | 2 | 0 | progress/2026-10-13/prefill-route-spike-2026-10-13.md cites it |
| `logs/mfma_probe-2026-10-13.log` | `progress/bench/evidence/mfma_probe-2026-10-13.log` | bench | 1 | 3 | 2 | 0 | progress/2026-10-13/prefill-route-spike-2026-10-13.md cites it |
| `logs/prefill_gemm_shapes-2026-10-13.log` | `progress/bench/evidence/prefill_gemm_shapes-2026-10-13.log` | bench | 1 | 4 | 3 | 0 | progress/2026-10-13/prefill-route-spike-2026-10-13.md cites it |
| `logs/prefill_mix_check-2026-10-13.log` | `progress/engine/evidence/prefill_mix_check-2026-10-13.log` | engine | 2 | 3 | 0 | 0 | supervisor ruling: --check certificate for the hipBLAS call form engine/kernels_gemm.cpp uses; cited by content in an engine route commit |
| `logs/prefill_mix_det-2026-10-13.log` | `progress/engine/evidence/prefill_mix_det-2026-10-13.log` | engine | 2 | 4 | 0 | 0 | supervisor ruling: holds the two mix hashes that certificate is checked against |
| `tmp/d2_dec300f_still_valid.py` | `progress/engine/evidence/d2_dec300f_still_valid.py` | engine | 1 | 2 | 1 | 0 | engine/results.md:2182 re-check script |
| `tmp/d5_allowance_report.py` | `progress/engine/evidence/d5_allowance_report.py` | engine | 1 | 3 | 2 | 0 | engine/results.md:2090 re-computes the cited table |
| `tmp/d6_final_pass.py` | `progress/engine/evidence/d6_final_pass.py` | engine | 1 | 2 | 1 | 0 | engine/results.md:2161 final pass |
| `tmp/d6_gate_dec64f.log` | `progress/engine/evidence/d6_gate_dec64f.log` | engine | 1 | 2 | 1 | 3 | engine/results.md:2558; engine/route_margin.py reads it |
| `tmp/d9_sizing/all.sh` | `progress/engine/evidence/d9_sizing/all.sh` | engine | 1 | 2 | 1 | 0 | engine/results.md:4273 eight locked device runs |
| `tmp/decode_run_library.sh` | `progress/engine/evidence/decode_run_library.sh` | engine | 2 | 4 | 3 | 2 | tie 2/2 broken by consumption: it is the bridge from engine/gate.py --binary to the library route, and engine/results.md:4476,4479 is the claim it certifies |
| `tmp/engine_gate/sampled.txt` | `progress/engine/evidence/engine_gate/sampled.txt` | engine | 1 | 2 | 1 | 0 | engine/results.md:285 negative control |
| `tmp/gate_ab.sh` | `progress/engine/evidence/gate_ab.sh` | engine | 1 | 2 | 1 | 1 | engine/results.md:4427 reproduce script |
| `tmp/gate_dec64f.log` | `progress/engine/evidence/gate_dec64f.log` | engine | 1 | 6 | 5 | 4 | engine 5 cites; engine/curve_probe.py + engine/route_price.py parse it |
| `tmp/gate_dec64t.log` | `progress/engine/evidence/gate_dec64t.log` | engine | 1 | 3 | 2 | 0 | engine/results.md:1574,1575 read it as probe input |
| `tmp/gemm_probe.cpp` | `progress/engine/evidence/gemm_probe.cpp` | engine | 1 | 2 | 1 | 0 | engine/results.md:4339 probe source |
| `tmp/margin/library_dec300f.log` | `progress/engine/evidence/margin/library_dec300f.log` | engine | 2 | 4 | 2 | 1 | supervisor ruling: engine/route_price.py and engine/route_margin.py parse margin logs; bench/performance-status.md:988 only quotes its conclusion |
| `tmp/p9/probe_pinwidth.py` | `progress/bench/evidence/p9/probe_pinwidth.py` | bench | 1 | 2 | 1 | 0 | progress/2026-10-14/p9-chunked-scan-2026-10-14.md:52 |
| `tmp/p9/stage.out` | `progress/bench/evidence/p9/stage.out` | bench | 1 | 9 | 1 | 4 | progress/2026-10-14/p9-chunked-scan-2026-10-14.md:114 full stage table |
| `tmp/probe_bf16_logits.py` | `progress/engine/evidence/probe_bf16_logits.py` | engine | 2 | 1 | 0 | 0 | supervisor ruling: logits fp/bf16 probe, cited only by an engine commit message |
| `tmp/q10/decode_run_contig.sh` | `progress/engine/evidence/q10/decode_run_contig.sh` | engine | 1 | 3 | 1 | 0 | engine/results.md:3739 contig-vs-interleaved run |
| `tmp/q10/ids1024.txt` | `progress/bench/evidence/q10/ids1024.txt` | bench | 1 | 27 | 25 | 4 | bench 15 vs prototype 10 doc lines; 25 documented commands read it (HANDOFF, attn-wave-diagnosis, p9-phase1) |
| `tmp/q10/ids3072.txt` | `progress/bench/evidence/q10/ids3072.txt` | bench | 1 | 4 | 3 | 0 | progress/2026-10-14/p9-route-a-2026-10-14.md:204-206 |
| `tmp/q10/md5_A_con.txt` | `progress/engine/evidence/q10/md5_A_con.txt` | engine | 1 | 3 | 2 | 0 | engine/results.md:3731-3733 identity witness |
| `tmp/q10/md5_A_int.txt` | `progress/engine/evidence/q10/md5_A_int.txt` | engine | 1 | 3 | 2 | 0 | engine/results.md:3731-3733 identity witness |
| `tmp/q2_session.py` | `progress/engine/evidence/q2_session.py` | engine | 1 | 2 | 1 | 0 | engine/results.md:1994 feed+gen parity run |
| `tmp/q6/proj_rpb32.cpp` | `progress/engine/evidence/q6/proj_rpb32.cpp` | engine | 1 | 4 | 2 | 0 | engine/results.md:725,727 half of the rpb8/rpb32 A/B |
| `tmp/q6/proj_rpb8.cpp` | `progress/engine/evidence/q6/proj_rpb8.cpp` | engine | 1 | 4 | 2 | 0 | engine/results.md:724,726 the other half |
| `tmp/q8/mixed_bw.err` | `progress/devices/evidence/q8/mixed_bw.err` | devices | 1 | 7 | 4 | 0 | devices/strix-halo-mixed-traffic.md:175,182,230,295 |
| `tmp/q8/mixed_bw.log` | `progress/devices/evidence/q8/mixed_bw.log` | devices | 1 | 7 | 6 | 1 | devices 6 cites: devices/strix-halo-mixed-traffic.md prices its section off it. NOTE the code consumer is tools/step_traffic_model.py --probe-file (rung 1 decides, and devices is whose claim it certifies) |
| `tmp/q8/mixed_bw2.log` | `progress/devices/evidence/q8/mixed_bw2.log` | devices | 1 | 2 | 1 | 0 | devices/strix-halo-mixed-traffic.md:330 corrected sweep |
| `tmp/q9/rows.log` | `progress/devices/evidence/q9/rows.log` | devices | 1 | 3 | 2 | 0 | devices/strix-halo-mixed-traffic.md:546,595 raw lines |
| `tmp/rc_a1.log` | `progress/bench/evidence/rc_a1.log` | bench | 1 | 3 | 1 | 0 | progress/2026-10-14/route-c-audit-2026-10-14.md:126 |
| `tmp/rc_a1/a1.json` | `progress/bench/evidence/rc_a1/a1.json` | bench | 1 | 5 | 4 | 2 | progress/2026-10-14/route-c-audit-2026-10-14.md:42,43,94,327 per-shape floors |
| `tmp/rc_a1/a1.txt` | `progress/bench/evidence/rc_a1/a1.txt` | bench | 1 | 2 | 1 | 0 | progress/2026-10-14/route-c-audit-2026-10-14.md:93 same census as prose |
| `tmp/rc_a2/all.txt` | `progress/bench/evidence/rc_a2/all.txt` | bench | 1 | 4 | 2 | 0 | progress/2026-10-14/route-c-audit-2026-10-14.md:150,188 per-row offenders |
| `tmp/rc_prompt.ids` | `progress/bench/evidence/rc_prompt.ids` | bench | 1 | 4 | 2 | 0 | progress/2026-10-14/route-c-audit-2026-10-14.md:36,40 id fixture a command reads |
| `tmp/route_dec300f.json` | `progress/engine/evidence/route_dec300f.json` | engine | 1 | 3 | 2 | 4 | engine 2 cites + engine/route_allowance.py --table reads it |
| `tmp/route_dec64f.json` | `progress/engine/evidence/route_dec64f.json` | engine | 2 | 2 | 1 | 1 | tie 1/1 broken by consumption: engine/route_price.py --out/--table |
| `tmp/route_dec64f.ra.json` | `progress/engine/evidence/route_dec64f.ra.json` | engine | 2 | 2 | 1 | 1 | tie 1/1 broken by consumption: engine/route_allowance.py refuses it; engine/results.md:2105 is the claim |
| `tmp/route_margin_dec64f.json` | `progress/engine/evidence/route_margin_dec64f.json` | engine | 1 | 2 | 1 | 1 | engine/results.md:2559 D-6 margin pass; engine/route_margin.py writes/reads it |
| `tmp/stub_decode_run.py` | `progress/prototype/evidence/stub_decode_run.py` | prototype | 1 | 2 | 1 | 0 | prototype/attn_decode/results.md:1105 the 18.1 coverage harness |
| `tmp/tilefix/repeat_ids.py` | `progress/bench/evidence/tilefix/repeat_ids.py` | bench | 1 | 2 | 1 | 0 | bench/performance-status.md:1094 how the 3072-lane prompt was built |
| `tmp/w9.sh` | `progress/bench/evidence/w9.sh` | bench | 1 | 2 | 1 | 0 | progress/2026-10-14/p9-chunked-scan-2026-10-14.md:435 baseline invocation |

Owner totals: **engine 25**, **bench 17**, **devices 4**, **prototype 1** — 47 files.

Two rows are worth reading twice:

- `tmp/q8/mixed_bw.log` — rung 1 decides it (`devices` 6 doc lines vs `bench` 1), and the
  **code** consumer is `tools/step_traffic_model.py --probe-file`. The owner follows the
  claim the file certifies (`devices/strix-halo-mixed-traffic.md` prices its whole section
  off those lines), not the folder holding the parser.
- `tmp/q10/ids1024.txt` — `bench` 15 vs `prototype` 10, so bench wins rung 1 on a narrow
  margin, while `prototype/attn_tile_model.py` prints it in a reproduce command. A sweep
  that priced it for prototype would flip this row; the counts are on the table so a reader
  can see how thin the margin was.

## Gitignore decisions that go with this tree

- `tmp/` and `logs/` stay ignored and now hold **only** scratch; the comment block in
  `.gitignore` says so. Nothing here needs `git add -f` any more.
- `!progress/**/*.log` is required, because the `*.log` rule is an unanchored pattern and
  bites at any depth — the same directory-vs-file class as `results/`, `refs/`, `data/`
  (see the trap rows in `AGENTS.md`). Verified with `git check-ignore`:
  `progress/bench/evidence/probe-new.log` and `progress/engine/evidence/q8/x.log` are not
  ignored, `logs/new.log` and `tmp/newscratch.txt` still are.
- The duplicate `*.o` line that sat at the end of `.gitignore` is gone; `*.o` is now stated
  once, in the build-output block.
- `thinking/` is deleted and its ignore rule removed: the directory was empty and its
  purpose — cross-session notes — is what `progress/` now is. `README.md` pointed a reader
  at `thinking/*.md`; it now points here and at
  `progress/2026-10-14/HANDOFF-2026-10-14.md`.
  `engines/` and its ignore rule are untouched.
- The `.pi/` block from `d7b2813` is untouched as a rule; the only change inside it is the
  two document paths its comment cites, which moved with M1.

## What this pass did NOT fix

- **Commit messages are immutable.** 52 mentions in 37 of them name one of
  these 65 paths (28 of the paths). They now name paths that no
  longer exist; this table is the crosswalk.
- **Transcript bytes are the witness, so four evidence files still name old paths** and
  were deliberately not rewritten:
  `progress/engine/evidence/d6_gate_dec64f.log`,
  `progress/engine/evidence/margin/library_dec300f.log`,
  `progress/bench/evidence/rc_a1/a1.txt`,
  `progress/engine/evidence/route_margin_dec64f.json`. Each one records a command line as
  it was actually run.
- `tools/doc_ref_check.py` keeps `tmp/p9/stage.out`-shaped strings in its regex comments and
  `--selftest` fixtures: those are scanner inputs, not pointers. They were rewritten with
  everything else, so the selftest stays internally consistent either way.
- `README.md` still mentions `thinking/` twice, both naming the **previous** project
  (`../qwen3.8-flash-next-old/thinking/…`), which is a different tree.
- Historical prose inside the journals and the two 2026-10-16 docs that discusses a file by
  its **old** name — e.g. `engine/results.md:3739` ("not committed", made stale by part 1),
  `progress/2026-10-16/structure-proposal-2026-10-16.md:118,215` about the malformed
  `plan-route-c-audit-2026-001.md`,
  and `tools/doc_ref_check.py:102`'s `bench/plan-2026-10-12` shorthand example — was left as
  written: rewriting an account of the past would make the account false.

