# The 2026-10-16 ledger prune — what moved, what stayed, what the re-take changed

Written 2026-10-16 alongside the prune itself. This is the decision record for the split of
`bench/performance-status.md`; the numbers it quotes are the ones now in the live ledger, and every
file:line it names was checked at HEAD `bb7246d`.

## A. The two files, and the evidence dir

| file | lines | role |
|---|---|---|
| `bench/performance-status.md` | **326** (was 1,149) | the live ledger. Current measured state, re-taken 2026-10-16, §1–§6 |
| `progress/2026-10-16/performance-status-2026-10-16.md` | **1,160** = 1,149 frozen + 11 provenance footer | the frozen previous state, byte-identical |
| `progress/bench/evidence/ledger-2026-10-16/commands-and-provenance.md` | 151 | every command, lock tag, `util` pair and spread behind today's rows, plus §4's basis clause and its correction note |
| `progress/bench/evidence/ledger-2026-10-16/stage-wave.txt` | 300 | the filtered `STAGE*`/`WAVE*`/`STAGEPARITY`/`GROUP` rows behind §2 and §3 |
| `tools/citation_rot_check.py` | 250 | added by the review of this pass (§I): fails when the text at a `path:NNN` coordinate differs from `HEAD`, which is the drift `doc_ref_check.py` cannot see |

Why the live ledger is short and the archive is not: the live file had become a chronology. Its first
190 lines were a state table, and the remaining ~960 were the append-only record of how each number
arrived — the clean-window re-take (Q1–Q10c), the route audits, the wave attributions, the headline
tables of four successive eras, and the corrections made in place. Two readers' needs were fighting:
"what is the engine's number today" wanted a table, and "how do I know that number is real" wanted the
whole history. They are now different files.

## B. What moved out of the live ledger, and what stayed

**Moved to the snapshot** (all of it still present in the repo, at the same line numbers):

- the old §1 "measured kernels" table (frozen :27-41), the old §2 "derived floors" (:77-90), the old
  §3 gap and §4 dispatch-wall width table (:120-135);
- the old §5 open-items list, including item 0 (the quiescent-pass procedure, frozen :191-205) and
  items (g) (frozen :203) and 5 (the block-count question, frozen :240);
- the whole §6 clean-window re-take era (:244-560) and everything appended after it: Q6/Q6.1 launch
  bills (:624-700), Q7–Q10c (:700-900), the wave-path attribution (:895-960), the route-(c) audit,
  and the four generations of "headline tables" (:1008-1149).

**Kept in the live ledger**, in this order:

| § | what it answers | source of its numbers |
|---|---|---|
| §1 | what the engine does today — decode, prefill/TTFT, cold start | re-taken 2026-10-16 |
| §2 | where a decode token goes — burst/stage/kernel, exposed gap, in-situ family rows at 128 and 4032 lanes | re-taken 2026-10-16 |
| §3 | where a 1024-token prefill wave goes — wave attribution, `GROUP` lines, width table | re-taken 2026-10-16 |
| §4 | per-op floors in isolation — `lm_head`, MLP by window and walk, `glue`, gated norm, projections, `gdn` state, launch floor, probes | re-taken today; a carried row is labelled as carried and names its frozen line |
| §5 | what the measurements decided — shipped defaults, dead routes with the number that killed each, debts | frozen record, cited by line |
| §6 | what a number in this repo is worth — protocol, bimodality, the `hipEvents` disowning, the lane-count and `util` rules, the lock/self-lock deadlock trap | re-stated; unchanged in substance |

**§6 is still the section that answers "what is a number on this box worth"**, and the numbering of
§1–§6 is a compatibility requirement, not a style choice: `engine/gate.py`, `engine/chat.py`,
`AGENTS.md`, `engine/kernels_glue.cpp` and a dozen `results.md` lines cite `§6` by number for the
clean-window protocol, and every one of them still resolves without an edit. That is also why the live
§1 does not keep the old §1's kernel table under the same number: the old §1 rows were pre-clean-window
floors, and leaving them under a live-looking number is exactly the failure the re-take was for.

## C. Why the snapshot is byte-identical

`progress/2026-10-16/performance-status-2026-10-16.md` is `cp` of the old file plus an appended footer,
and it was verified with `cmp` rather than by eye:

```
cmp <(head -1149 progress/2026-10-16/performance-status-2026-10-16.md) <(git show HEAD:bench/performance-status.md)
→ identical (rc=1 only because the second stream is the git blob; the 1149-line prefix matches byte for byte)
```

Three reasons not to re-type or trim it:

1. **`path:NNN` cites are claims about line numbers.** 96 of the 101 line-form cites in this tree
   originate in `progress/` (rule 2 freezes those files, so they cannot be re-pointed cheaply), and the
   archive's cites name coordinates inside the old ledger. A hand-made excerpt would have silently
   mis-pointed every one of them; a byte-identical copy keeps all of them true at the same coordinates.
2. **Rule 2 of `progress/README.md`.** Nothing in `progress/` may be edited except same-line path
   rewrites. A transcribed ledger would be a new document pretending to be a record; a byte copy is a
   record.
3. **A number that has to be retyped can drift.** The footer states explicitly that everything after
   line 1,149 is provenance about the snapshot and not part of the measurement record, so the marker
   itself is the only addition.

## D. Citation re-point table

Every cite changed by this pass, what it asserted, and where it now points. **L** = live tree, **P** =
`progress/` (same-line substitution only).

| file:line | old target | new target | fact the cite asserts |
|---|---|---|---|
| L `prototype/glue/run.py:65` | old §1 | live §4 | launch floor **1.85 µs/launch**, measured |
| L `prototype/glue/glue.hip.cpp:12`, `:361` | old §1 | live §4 | same 1.85 µs floor used as a divisor |
| L `prototype/glue/results.md:28` | old §1 | live §4 | launches/token and µs/launch, not GB/s, is the finding |
| L `prototype/gdn_decode/gdn_decode.hip.cpp:265` | old §1 | live §4 | **18 MiB of GDN state stays cache-resident**; the L2 attribute under-reports it |
| L `prototype/gdn_gated_norm/results.md:477` | old §1, §6 | live §4, §6 | its own per-launch row's ledger home |
| L `prototype/proj_gemv/proj_gemv.hip.cpp:13` | old §2 | live §4 | **0.4677 GB/token = 29 %** of the token's bytes |
| L `prototype/proj_gemv/proj_gemv.hip.cpp:510` | "ledger §5 item 5" | frozen `:240` | the block-count question (20 CUs, not the 40 `rocminfo` prints) |
| L `prototype/proj_gemv/run.py:40-44` | old §2's two "not measured on its own" rows | live §4 + frozen `:81-82` | the two projection-family rows, now measured (`--only gdn`/`--only attn`) |
| L `prototype/attn_decode/results.md:115-117` | old §5 item (g) | frozen `:203` | position-64 pre-rope reduction attribution, the open item |
| L `prototype/attn_decode/prerope_probe.py:4` | old §5(g) | frozen `:203` | same item, from the probe that closed it |
| L `prototype/gdn_decode/results.md:30-31` | "§1, §2 and §6 now quote 0.0646 ms" | live §4 quotes **0.0648 ms**; the 0.0646 row is frozen `:36,265` | the warm state-launch row, re-taken |
| L `prototype/gdn_decode/results.md:38-39` | old §5 provenance rule, §6 token sum | live §6 + frozen `:271` | the provenance rule; the `0.062` that was carried into the old token sum |
| L `prototype/mlp_gemv/results.md:331-334` | "§5 item 5 owns the follow-up" | frozen `:240` + live §4 | the follow-up's answer: 98 % of probe at the 528 MB contig window |
| L `prototype/mlp_gemv/mlp_gemv.hip.cpp:6-8` | old §2 | frozen `:80` + live §4 | **MLP 0.5285 GB/token = 33 %** of decode bytes |
| L `prototype/attn_decode/wave_model.py:44` | bare "the ledger" route-(c) table | frozen `:951` | `MEASURED_ATTN_MS = 109.2` |
| L `prototype/attn_decode/wave_model.py:50` | bare "the ledger" | frozen `:923` | hand GEMMs at **~0.85 TFLOP/s** |
| L `prototype/attn_tile_model.py:62` | bare "the ledger" route-(c) | frozen `:951` | same 109.2 ms baseline |
| L `engine/kernels_proj.cpp:35` (the line below the one that names the width table) | old §4 width table | frozen `:135`, and live §4 for today's re-take of the direction | a launch costs 3.75 µs at 256 threads and 2.776 µs at 1024; decode issues 114 of these launches. **Line 34 is byte-identical to HEAD** because `progress/2026-10-11/commit-review-2026-10-11.md:90` cites `engine/kernels_proj.cpp:34,38`, so the re-point went on the next line |
| L `engine/kernels.h:48` | §"Wave-path attribution" of the live file | frozen `:912` | matmul **74.4 %** of a 1024-token prefill; hand GEMM ~0.85 vs hipBLAS ~33 TFLOP/s |
| L `engine/kernels.h:171` | §6 "Q10" | frozen `:811` | `contig` recovers **0.945 ms** of the MLP's 2.034 ms in-step gap |
| L `engine/stage_clock.h:4` | §6 | frozen `:629` | the **≈3.4–3.6 ms residue** inside a ~12.9 ms token, which is why the stage clock exists |
| L `engine/README.md:313` (note: **not** :314, which `progress/2026-10-16/archive-review-2026-10-16.md:286,343` cites) | — (new text) | frozen `:267` + live §4 | the 2.936 µs/launch row above it is the frozen era's; live §4 carries today's 2.967 µs/launch re-take |
| L `engine/README.md:323` | §6 | frozen `:263` | the glue `<256`-thread preference is measured (E1: 784 comparisons at 1024 threads, `g1_n` gated 0 / raw 1), not a numerical requirement |
| L `engine/results.md:127` | §6 | live §4 + frozen `:267` | gated-norm launch cost, then and now |
| L `engine/results.md:302` | "§6 now carries 11.978 / 12.053 / 12.232" | frozen `:269` (+ live §1) | the first real ms/token row |
| L `engine/results.md:394` | §6 | frozen `:244` | the clean-window header that refused a gate by-product as a measurement |
| L `engine/results.md:503` | §6 | frozen `:269` | provenance of the three-run row |
| L `engine/results.md:520` | "ledger §6 bills" | frozen `:271` (kept §6) | the old token sum the table reconciles against |
| L `engine/results.md:740` | "ledger §2's two not-measured rows" | live §4 | the command block for the two projection rows the `--only` selector was built for |
| L `engine/results.md:1020` | "ledger §2's two not-measured rows" | live §4 | the hand-off note repeating the same ask, now marked closed by §4 |
| L `bench/gate-audit.md:9` | §5 item 0 of `performance-status.md` | frozen `:191` | the quiescent-pass procedure |
| L `bench/gate-audit.md:226` | §6 | frozen `:267` + live §4 | 2.936 µs/launch, spread 1.4 %, then; 2.967, spread 0.8 %, today |
| L `bench/build-queue.md:69` | §5 item 5 | frozen `:240` | the block-count question expected to land on `out_proj`/`o_proj` |
| L `README.md:47-51` | 87.1 / 3,331 tok/s, "§6 is the live table, §1–§5 are superseded" | **90.6 tok/s @128, 3,443.1 @1024**, §1 is the live table, eras frozen in the snapshot | the on-ramp's headline and section map |
| L `README.md:167` | §6 | frozen `:15` (+ live §1) | M2's 83.9 tok/s / 11.919 ms row |
| L `README.md:246-248` | "§1 gets a measured tok/s row replacing the projected 137–147" | live §4 named as the per-op floor table, old figure labelled pre-engine-era | which numbers are measured vs projected |
| L `devices/strix-halo-mixed-traffic.md:58` | "the ledger, Q6.1 stage medians at 128 lanes" | frozen `:666` | **12.399 ms/token** era row |
| L `tools/instep_family_budget.py:50-54` | "the ledger's Q6.1 table" | frozen `:683-693` | the isolated-burst cost per family per token |
| L `tools/instep_family_budget.py:67-69` | "the ledger Q6's dispatch curve" | frozen `:135` + `:39` | 1.85 µs floor, 2.6–3.8 µs at the widths these families use |
| P `progress/README.md:123`, `:149` | the pre-prune ledger, lines 988 and 1094 | snapshot `:988`, `:1094` | the margin log's identity; how the 3072-lane prompt was built |
| P `progress/2026-10-16/cleanup-audit-2026-10-16.md:96`, `:102`, `:123` | the pre-prune ledger, lines 989, 1094 and 988 | snapshot, same lines | same three facts |
| P `progress/2026-10-11/commit-review-2026-10-11.md:99` | `performance-status.md:65` | snapshot `:65` | the §1 row that read 387 % of probe |
| P `progress/2026-10-11/commit-review-2026-10-11.md:109` | `performance-status.md:19-20` | snapshot `:19-20` | the `0.062 ms / 918 GB/s / 387 %` and `1.344 ms / 255 GB/s` rows |
| P `progress/2026-10-11/commit-review-2026-10-11.md:131` | "ledger §6" | ledger §6 + snapshot `:271` | the 7.975 ms token sum / ~125 tok/s row |
| P `progress/2026-10-11/commit-review-2026-10-11.md:139` | `ledger §5:187` | frozen ledger snapshot `§5:187` | where the re-take was marked "waiting for quiescent pass" |
| P `progress/2026-10-11/review-2026-10-11.md:37` | `performance-status.md §5 item 0` | snapshot `§5 item 0` | the GPU pass, in order |
| P `progress/2026-10-11/review-2026-10-11.md:63` | "`performance-status.md` §6" | snapshot (§6) | the clean-window re-take of the §1 rows |
| P `progress/2026-10-11/review-2026-10-11.md:93` | `ledger §5(g)` | §5(g) + snapshot `:203` | the position-64 pre-rope open item |
| P `progress/2026-10-12/plan-2026-10-12.md:183` | "ledger §2's two" | §2 + snapshot `:81-82` | the two projection rows Q6 item 5 asked to fill |
| P `progress/2026-10-12/plan-prefill-2026-10-12.md:299` | `performance-status.md §6` | snapshot §6 | the 6.10 GB/s effective slope, dispatch-bound |
| P `progress/2026-10-12/plan-prefill-2026-10-12.md:375` | `performance-status.md §6` | §6 + snapshot | where P10's rows were to land |
| P `progress/2026-10-12/plan-prefill-2026-10-12.md:462` | `performance-status.md §6` | snapshot `:431` | the slope line that named MAXSC as the blocker |
| P `progress/2026-10-14/plan-p9-chunked-scan-2026-10-14.md:5-6` | `performance-status.md §"Wave-path attribution"`, `§"Headline tables after the route (c) flip"` | snapshot, same §labels | the 40 %-non-matmul funding threshold and the pre-P9 headline tables |
| P `progress/2026-10-14/speedup-routes-2026-10-14.md:5-6` | "now 1,100+ lines of append-only ledger" | "1,149 lines frozen as the snapshot, appended to ever since" | what the route index points at |
| P `progress/2026-10-16/journal-chronology-2026-10-16.md:272`, `:279` | `bench/performance-status.md §6` (×2) | snapshot `:269`, `:244` | extracts of `engine/results.md` that named the §6 rows |

Deliberately **not** re-pointed, because live §6 still carries the fact they cite (the clean-window
protocol, the bimodality warning, the `util` rule, "timings are the main agent's"):
`engine/gate.py:4454`, `engine/chat.py:19,200`, `engine/kernels_glue.cpp:44`,
`engine/README.md:455,466`, `engine/results.md:5,20,309,1242,2798,2817,3296,3452,3804,4035`,
`AGENTS.md:122`, `prototype/{proj_gemv,glue,mlp_gemv,attn_decode}/*` protocol cites, and
`bench/build-queue.md:114` ("§1 gets a measured tok/s row" — live §1 is that row).

## E. Rule 2 compliance for `progress/`

Every `progress/` edit is a same-line substitution. **The first pass of this prune broke that rule
outside `progress/`** — it added lines to twelve live-tree files that the frozen archive cites by
number, which rotted 27 line-citations while `doc_ref_check.py` stayed green; §I records the fix. The
invariant now holds everywhere and is machine-checked by `tools/citation_rot_check.py`: **every file
this pass touched has the same `wc -l` it has at `HEAD`**, and the only file allowed to change length
is the ledger itself.

| file | HEAD | now | file | HEAD | now |
|---|---|---|---|---|---|
| `AGENTS.md` | 255 | 255 | `prototype/attn_decode/results.md` | 1509 | 1509 |
| `README.md` | 248 | 248 | `prototype/attn_decode/wave_model.py` | 401 | 401 |
| `bench/build-queue.md` | 169 | 169 | `prototype/attn_tile_model.py` | 1629 | 1629 |
| `bench/gate-audit.md` | 266 | 266 | `prototype/gdn_decode/gdn_decode.hip.cpp` | 273 | 273 |
| `devices/strix-halo-mixed-traffic.md` | 639 | 639 | `prototype/gdn_decode/results.md` | 163 | 163 |
| `engine/README.md` | 496 | 496 | `prototype/gdn_gated_norm/results.md` | 517 | 517 |
| `engine/kernels.h` | 381 | 381 | `prototype/glue/glue.hip.cpp` | 402 | 402 |
| `engine/kernels_proj.cpp` | 150 | 150 | `prototype/glue/results.md` | 286 | 286 |
| `engine/results.md` | 4498 | 4498 | `prototype/glue/run.py` | 780 | 780 |
| `engine/stage_clock.h` | 298 | 298 | `prototype/mlp_gemv/mlp_gemv.hip.cpp` | 495 | 495 |
| `progress/README.md` | 207 | 207 | `prototype/mlp_gemv/results.md` | 534 | 534 |
| `progress/2026-10-11/commit-review-2026-10-11.md` | 300 | 300 | `prototype/proj_gemv/proj_gemv.hip.cpp` | 535 | 535 |
| `progress/2026-10-11/review-2026-10-11.md` | 112 | 112 | `prototype/proj_gemv/run.py` | 749 | 749 |
| `progress/2026-10-12/plan-2026-10-12.md` | 271 | 271 | `prototype/attn_decode/prerope_probe.py` | 452 | 452 |
| `progress/2026-10-12/plan-prefill-2026-10-12.md` | 584 | 584 | `prototype/attn_decode/results.md` | 1509 | 1509 |
| `progress/2026-10-14/plan-p9-chunked-scan-2026-10-14.md` | 51 | 51 | `tools/instep_family_budget.py` | 241 | 241 |
| `progress/2026-10-14/speedup-routes-2026-10-14.md` | 337 | 337 | `progress/2026-10-16/cleanup-audit-2026-10-16.md` | 266 | 266 |
| `progress/2026-10-16/journal-chronology-2026-10-16.md` | 1307 | 1307 | `bench/performance-status.md` | **1149** | **337** |

`engine/results.md` — the other append-only journal — was edited under the same rule for the sake of
the same argument: **4,498 lines before, 4,498 after**, seven changed lines, all of them prose inside
existing paragraphs, and `bash engine/build.sh` still builds both binaries. Two lines —
`engine/README.md:314` (cited by `progress/2026-10-16/archive-review-2026-10-16.md:286,343`) and
`engine/kernels_proj.cpp:34` (cited by `progress/2026-10-11/commit-review-2026-10-11.md:90`) — are themselves
line-citation targets, so they were returned to byte-identical and their re-points moved to the line
below; `tools/citation_rot_check.py` is what proved both cites were live.

## F. What today's re-take changed

Rows that moved by more than 5 % against the frozen ledger, with the frozen line to compare against:

| row | frozen | today | Δ | why |
|---|---|---|---|---|
| cold weight-restore, best of 5 | 454.4 ms best / 510.2 median / 77 % spread (`:493`) | **335.8 / 345.8 / 9.2 %** | **−26 % best, −32 % median** | the load path changed (pinned staging, first-pass 8.5 ms); the spread collapse is the real news |
| projection burst, 48-launch plan | 3.039 → 3.081 ms at **63–64 %** of probe (`:83,132`) | **3.2843 ms / 60 %** at rows/block 32 | **+6.6 % ms, −4 pp of probe** | recorded, not reconciled: rows/block 32 and unpacked `x` are not the 2026-10-12 configuration, and a ruler re-taken in another window is not an A/B (§6) |
| in-situ `mlp`, 128 lanes | 4.264 ms (`:685`) | **3.3880 ms** | **−20.5 %** | `--mlp-map auto` shipped after that window; today's row reproduces Q10b's 3.391 ms to 0.1 % |
| in-situ `glue`, 128 lanes | 0.678 ms (`:689`) | 0.6125 ms | −9.7 % | same window construction, and glue is the least reproducible family on this box |
| in-situ `copy`, 128 lanes | 0.208 ms (`:692`) | 0.2186 ms | +5.1 % | never measured in isolation; still a stage-granularity residual |
| stage wall, 128 lanes | 14.399 ms (`:679`) | **13.247 ms** | −8.0 % | `--attn-tile 1` and `--mlp-map auto` both shipped in between |
| exposed gap (`STAGEGAP exposed_ms`) | ≈1.66 ms (`:679`) | **1.98 ms** | **+19 %** | the burst fell faster than the stage; the submission bill is unchanged in kind, which is the point §2 makes |
| prefill at 256 lanes (sweep ruler) | 1,893 tok/s (`:1100` era ruler) | 1,828.5 tok/s | −3.4 % ms-side | inside the bimodal band for short prompts; §1 states the spread and points at the paired A/B |

Rows that **did not** move, stated because the standing rule asks for an unchanged headline in words:
decode headline **90.6 tok/s at 128 lanes** against 89.6 (−1.1 % on ms/token — the headline did not
materially move); prefill 1024 lanes 3,443.1 vs 3,326 tok/s (+3.5 %), 3072 lanes 3,385.8 vs 3,320.9
(+2.0 %); `lm_head` 2.1309 ms vs 2.107–2.110 (+1.0 %); `glue` fused @512 t 0.1496 vs the frozen §6's
0.1499 (−0.2 % — the +10 %-against-§1 reading is against a row the frozen ledger itself had already
superseded at `:33,259`); gated norm 2.967 vs 2.936 µs/launch (+1.1 %); `gdn` state warm 0.0648 vs
0.0646 ms (+0.3 %); MLP at the 132 MB interleave walk 2.647 vs 2.650 ms (−0.1 %); the wave's `attn`
row 26.74 vs 26.70 ms (+0.15 %); `gdn_rec` 126.83 vs 121.57 ms (+4.3 %); `GROUP matmul_ms` 70.30 vs
72.04 ms (−2.4 %). The read probe re-took at **239.3 GB/s**, 1 % above the 237 GB/s every "% of probe"
column in this repo is stated against, so those columns were left on 237 and §4 says so.

Two things today's pass *added* rather than changed: the two projection-family split rows §4 now
carries (GDN 2.5969 ms / 62 %, attention 0.6887 ms / 54 % of probe), which close the old §2's
"not measured on its own" debt, and the MLP window×walk burst grid (132/528 MB × `interleave`/`contig`),
which answers the old §5 item 5's block-count question from the other side: 94 % and 98 % of probe are
reachable for this shape, so 84 % was never this box's ceiling.

## G. Verification actually run

```
cmp <(head -1149 progress/2026-10-16/performance-status-2026-10-16.md) <(git show HEAD:bench/performance-status.md)   # byte-identical prefix
wc -l bench/performance-status.md                                                     # 337 (317 as first written; 20 lines added by the §2/§3 corrections in §I)
python3 tools/citation_rot_check.py --selftest       # SELFTEST PASS — planted shift caught; quiet corpus and below-only edit quiet
python3 tools/citation_rot_check.py                  # scanned 282 tracked files against HEAD; checked 419 line-anchor(s); 0 rotted line-citation(s)
python3 tools/doc_ref_check.py --list-broken         # byte-identical to the pre-edit baseline: 0 new broken paths/anchors
python3 tools/citation_census.py --check             # CHECK PASS — F1 101/101, F2 171/171, F4 16/16, F5 24/24, F5b 95/95, F6 66/66; live-origin F1 expected 5, counted 5
python3 tools/citation_census.py --anchors bench/performance-status.md                # 32 citations parsed, 0 broken
# falsifiability witness on the real tree, not the synthetic one: insert one line at
# devices/strix-halo-mixed-traffic.md:30 and `citation_rot_check` immediately reports
# `ROTTED devices/strix-halo-mixed-traffic.md:175 | cited by progress/README.md:135` (and the two
# cleanup-audit cites); restore the file and it prints 0 again. A check that cannot fail is the trap
# this repo documents three times over, so it was made to fail on demand before it was trusted.
bash engine/build.sh                          # built engine/build/loader_test, engine/build/decode_run (comment-only edits under engine/)
python3 -m py_compile <each edited .py>       # clean
```

**What `doc_ref_check.py` did not catch, and why.** It passed on the first version of this prune while
27 line-citations were pointing at the wrong sentences. Its line-anchor check asks two questions of a
`path:NNN` cite — is `NNN` within the file, and is that line non-blank — and an insertion above a cited
line keeps both answers true while changing the text under it. `tools/citation_rot_check.py` was
written for this gap: it compares the *text* at the cited coordinate between `HEAD` and the working
tree, so it fails on exactly the drift `doc_ref_check` is blind to, and it says so in its own docstring
along with what it is blind to in turn (§labels, absent-at-either-revision targets, edits below a cite).
Both are run; neither is a substitute for the other.

No device command was run for any number in the live ledger: every row in §1–§4 came from the main
agent's windows recorded in `commands-and-provenance.md`.

## H. Handed over, deliberately unresolved

1. **Decided, not handed over:** the frozen snapshot keeps the name
   `progress/2026-10-16/performance-status-2026-10-16.md`. `progress/2026-10-16/archive-review-2026-10-16.md`
   had proposed `ledger-pre-clean-window-2026-10-16.md`, so that path stays in `doc_ref_check`'s broken
   list as the record of a proposal that was not taken — rule 2 forbids editing the archive sentence
   anyway, and renaming would have broken the live header, the snapshot footer and this file for the
   sake of one stale propose-date path.
2. `progress/2026-10-16/cleanup-audit-2026-10-16.md:125` says the ledger-prune decision record "does
   not exist yet". This file closes that gap; the archive line was left as written, per rule 2.
3. The `tmp/ledger/` scratch logs (`stage128_*.log`, `wave1024_*.log`, …) stay uncommitted as scratch.
   The filtered `STAGE*`/`WAVE*` rows and the full command/lock/util provenance are what was carried
   into `progress/bench/evidence/ledger-2026-10-16/`, so no live cite depends on `tmp/`.
4. `engine/results.md:520`'s reconciliation table still bills against the frozen §6 token sum
   (frozen `:271`), because that is the row it was written to reconcile. Live §2 no longer prints a
   comparable sum at all: at stage granularity the family gages are **exhaustive** (Σ families − TOTAL
   ≤ 0.0001 ms in all six runs), so there is nothing left to reconcile inside the step — see §I.
5. Anything still citing "§1" of the ledger for a **pre-clean-window floor** rather than for the
   headline row now lands on §1, which no longer carries those floors. The two found in this pass
   (`prototype/glue/*`, `prototype/gdn_decode/gdn_decode.hip.cpp`) were moved to §4; a reader who finds
   another should apply the same test: is the fact a current floor (§4) or a headline (§1)?

## I. Corrections made in the review of this prune (2026-10-16, same day)

A re-audit of the raw logs found two medians and a sum mis-stated in the live ledger, and one class of
defect in this pass itself. All of it is recorded here because a document that reorganises a
measurement record should not be able to introduce an error silently.

**Numbers.** Live §2's `lm_head` in-situ row is the **median 2.1799 ms**, not the max 2.1838 that was
first printed, so burst→step is **2.3 %** (the same clause in §6's "a burst is not a step" rule). Live
§2's family sum is **13.273 ms against a 13.247 ms stage wall** — it reconciles, and §2 now carries the
reconciliation paragraph instead of the old "12.639 of a 13.247 wall" framing, which was an arithmetic
error in `commands-and-provenance.md`, not a property of the machine. **That correction was itself
wrong, and a third pass fixed it.** Checking the witness file directly (`stage-wave.txt`) shows Σ
families = TOTAL in every run to 0.0001 ms — the stage gages are exhaustive — so the honest statement is
that the step wall has **no** unattributed time at this granularity, and that a sum of per-family
medians cannot be compared to a median wall in the first place (medians do not add across runs; the
0.2 % was arithmetic, not a finding). §2 and the record now say that, and the `12.639` framing and the
`13.273 reconciles` framing are both gone. Q6's ≈3.4–3.6 ms residue
(frozen `:629`) is a different construction — burst numbers against a token, not gages against a wall —
and stands. The lesson generalises: a reconciliation is only meaningful when the parts are not already a
partition, and the exhaustiveness of a gage set is checkable in the log the instrument prints. The 4032-lane `attn` in-situ
median is **2.996 ms** (not 2.993); §6 debt 7 matches. **The wave composites needed a basis, not just a
number.** §3's `gdn` batched, `cast`+`copy`, `lm_head`+`host_tail` and the two share rows — five cells —
were first transcribed from run 1 under a caption that said "medians of 3". They now
carry medians, and both files name the basis, because for a composite the two defensible bases differ
in the third digit: `gdn` batched is **7.1248 → 7.12** as a sum of per-family medians
(4.5194 + 0.4109 + 2.1945) and **7.108** as a median of per-run sums. This ledger prints the former,
`commands-and-provenance.md` §4 says so in its caption and carries a correction note admitting its own
edit, and live §3 carries the same clause so a reader who never opens the record can still reproduce
the row. `lm_head`+`host_tail` is 2.2940 + 0.7455 = **3.0395 → 3.04** with a 0.93 + 0.31 = 1.24 → **1.2 %**
share, and `cast`+`copy` is 6.8252 + 2.6335 = **9.4587 → 9.46** with **3.8 %**; the two printed share rows
are the medians of the printed values (11.00/10.90/10.81 → **10.9 %**; matmul 28.91/28.55/28.29 →
**28.6 %**).

**A third correction, found by rebuilding the witness rather than reading it.** Live §2 first printed the
exposed gap as 1.97 ms (runs 1.881 / 1.973 / 2.197) — which is `TOTAL − burst`, the gaged sum minus the
burst mean. The instrument prints its own exposed figure as `STAGEGAP exposed_ms` = serialized mean −
burst mean, and that is **1.98 ms** (1.889 / 1.981 / 2.208); at 4032 lanes the instrument says **2.26**
where the other pairing said 2.248. The two spellings agree to within the 0.008 ms of ungaged step
remainder, so no conclusion moves — but a cited number must name its subtraction, because the two
pairings are not interchangeable in a comparison. §2 and the record now print the `STAGEGAP` value and
name the alternative beside it. The same check found the committed witness itself was thinner than the
ledger it backs: `stage-wave.txt` carried neither the `LAUNCH`/`NUMERICS` flag echo (§6 requires every
row to name its flags) nor the `STAGEGAP` lines, so those rows could not be re-derived from the file the
record claims they came from. It is now 336 lines, 40 per stage run and 32 per wave run, and every §2/§3
figure is reconstructible from it alone.
**The defect in the pass itself.** The citation migration added lines to twelve live-tree files, which
moved text out from under 27 line-citations that the frozen archive makes into those files — the
failure mode `AGENTS.md` warns about, met by the pass whose job was citation hygiene. Fixed by folding
every inserted line back into the lines already present (§E's table is the proof), with two cited lines
(`engine/README.md:314`, `engine/kernels_proj.cpp:34`) returned to byte-identical text, and their re-points moved to
the following line. `tools/citation_rot_check.py` exists so the next pass finds this out in one
command instead of one review.
