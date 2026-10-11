# Commit review — 2026-10-11 (72 commits)

**Provenance.** Produced by the read-only `reviewer` agent run `08eca213`, transcribed to disk by the
main agent: that runtime had no shell and no write tool, so the audit was delivered as text and its
final third never arrived — the output truncates mid-verdict-table at `c12...`. **§2 (band-tolerance
history), §3 (message-vs-code detail) and §4 (device hand-off list) are MISSING from this file** and
are re-derivable work, tracked as the first item for the next worker. Nothing below was edited except
this header.

Every claim the main agent independently re-verified against HEAD is marked `[re-verified]`: the
105683 figure does still sit in `engine/README.md:83-84` and `prototype/gdn_gated_norm/results.md`;
`engine/state.h:14` does claim a spec correction that did not happen; `cast_bf16_to_f32_3` is declared
at `engine/kernels.h:84` with no definition anywhere; `engine/kernels_glue.cpp:15` does launch
`glue_add_norm<1024>` while `prototype/glue/run.py:471` defaults to 256; `engine/kernels_proj.cpp`
does set `PACK_X = true` while `prototype/proj_gemv/results.md:196` measures that at 3.269 ms (60 %)
against the gated 2.991 ms (66 %). The audit's six WORST FIRST items are all real.

---

# Commit review — 2026-10-11 (72 commits)

Audited by the review subagent, read-only, on the CPU. Built from: HEAD source, `bench/*.md`,
`prototype/*/results.md`, `specification/*`, and `.git/logs/HEAD` (reflog, for commit messages).

**What this audit could not do, and what therefore hangs on §4.** The review runtime had no shell
and no write tool: `speccheck/run_all.py` was **not started** (the task's first action), `git log
-p` was not run, and nothing was compiled or executed. Commit messages come from the reflog, which
carries the messages but not the diffs; the band-history archaeology in §2 is therefore built from
in-code history notes + commit messages + current constants, and needs the one git command in §4-C
to be made airtight. Every claim below is labelled with the evidence class: **[source]** provable
from HEAD files, **[msg]** message-only, **[pending:X]** needs a §4 command.

Verdicts: `CONFIRMED` (code/fixture shows what the message says) · `OVERSTATED` (real, but the
number/claim is bigger than what is judged) · `UNSUPPORTED` (nothing in the tree backs it, or it
contradicts the tree) · `DEVICE-PENDING` (cannot be settled without the device/compile pass).

## WORST FIRST

Six claims on 2026-10-11 are false as written, three of them load-bearing for `engine/`.

1. **`f79fe0f` — `engine/README.md:83-84` repeats a claim `fc444d0` had already retracted two
   commits earlier.** README says §04.5's readings are "refused, including A-01 as literally
   written (105683 differ) and `speccheck.specimpl.rmsnorm_gated` (**the same 105683**)". Called
   the way speccheck calls it — `specimpl.py:304` passes `w.p[...norm.weight]`, and
   `specimpl.py:70-86` casts every model weight to BF16 unless `keep_f32_scalars=True`, which
   defaults to `False` — the oracle is bit-exact (0/294912), which is what `fc444d0` measured and
   what the dtype plumbing shows. `f79fe0f`'s own subject line says "gdn_gated_norm mirror ndiff 0"
   while the file it adds asserts the oracle mismatches. Fix: delete the "(the same 105683)"
   clause; see also item 4. **[source; final number pending §4-B1]**
2. **`27dbaed` — `engine/state.h:14`**: "Two deliberate departures from what §08 used to say, both
   **corrected in the spec in the same commit as this file**". They were not. `specification/08-state-management.md:14`
   still says conv window `BF16 … 0.63 MiB`, `:15` still says `4 KiB/token/layer`, `:20` still
   says `3.07 GiB`. And `engine/state_sizes.py:71,75,79`, added **in the same commit**, prints
   "specification/08 NOT yet corrected" three times. The commit message is honest about this
   ("flags three doc slips"); only the header comment is wrong. Fix: `state.h:14-15` → "flagged in
   the spec, not yet corrected". **[source]**
3. **`f79fe0f` — `engine/results.md:101`**: "`cast_bf16_to_f32_3` **already** folds three regions
   into one launch for exactly this reason." It is declared (`engine/kernels.h:84`) and has **no
   definition anywhere** in the tree; `engine/README.md:61-62`, written in the same commit, lists
   `kernels_cast.cpp` under "**Not written**". The ≈66-casts/token deferral is built on a helper
   that does not exist, described in the present tense. Fix: "is declared for exactly this reason
   and is not written yet". **[source]**
4. **`0a062b5` — headline finding is false as stated**, and the probe that "proves" it encodes the
   false premise in its pass criterion. Message: "A-01's cast order is not what the pins carry …
   refuses A-01-as-written (105683 differ) and specimpl.rmsnorm_gated (same 105683)".
   `mirror_probe.py:45,62` hands the oracle `w.float()` — the **raw F32 checkpoint tensor**
   (`operands.py:55-65` returns `get_tensor` verbatim) — a dtype combination speccheck never
   produces. The kernel's POLICY variant is unaffected and the pin-level assertion is genuinely
   valuable; but the probe's specimpl row is an artifact, and `gdn_gated_norm/results.md:26,30,49`
   and the probe docstring `:16-17` still print the retracted 105683 figure — `fc444d0` amended
   `bench/gate-audit.md` only. **This also resolves the "Unresolved" note at the end of the G2
   amendment**: the three numbers are one formula at three weight dtypes — specimpl-as-called
   (BF16 weight, product rounded in BF16) = **0**; A-01-as-written with the BF16-rounded weight
   widened back to F32 = **74860** (the amendment's number); A-01-as-written with the raw F32
   weight = **105683** (the probe's). Inputs are identical; only the weight dtype moves them.
   Fix: change the probe's specimpl row to pass the BF16-rounded weight and expect ndiff 0 (that
   then closes G2's oracle half as a live assertion), or the row actively forbids the corrected
   answer. **[source + pending §4-B1]**
5. **`27dbaed`/`f79fe0f` vs `bc78262`/`c12937e` — the engine hard-codes two launch configurations
   that its own gates never ran, and one of them is the one the op measured as slower.**
   * `engine/kernels_glue.cpp:15` launches `glue_add_norm<1024>`; the 784-comparison gate ran at
     `--threads 256` (`prototype/glue/run.py:471` default), and `glue/results.md:170` says in
     so many words: "**Re-run the gate at 1024 before W5 hard-codes a width**", still listed open
     at `:230`. This is not cosmetic: `glue_kernels.h:83-102` sets `PER = H/THREADS`, so 256 →
     4 elements/thread and an 8-partial warp tree, 1024 → 1 element/thread and a 32-partial tree —
     **a different F32 summation order**, and per `glue/results.md:113` `ndiff` is "the criterion
     doing the work", i.e. the only detector of the one bug class a norm cannot otherwise see.
     The floor sample (3/393216) was measured on the 256-thread order (`:102` "4 strided/lane …
     8-partial tree"), so the engine's order has no floor measurement either.
   * `engine/kernels_proj.cpp:34,38` instantiate `proj_gemv<IN,true>` (PACK_X). The
     912-comparison gate runs `--mode once` **without** `--x-packed` (`proj_gemv/run.py:402-406`
     adds it only in the timing path), so `<IN,true>` was never compared against a pin. The
     products are bit-identical by construction (every BF16 is exact in F32, `proj_gemv_kernels.h:96-99`),
     so the correctness risk is low — but the **performance** claim in the same comment
     ("it removes a cast per projection") contradicts the op's own measurement:
     `proj_gemv/results.md:196` measures PACK_X at **3.269 ms / 60 %** vs 2.991 ms / 66 % at the
     gated default, concludes "**and it still never wins**", and `:241` tells W5 to take
     "`PACK_X = false`, measured above". The casts PACK_X saves are ≤48 launches (the plan is a
     48-launch plan, `performance-status.md:65`) ≈ 0.089 ms at the measured 1.85 µs floor —
     against a measured +0.279 ms penalty. Ledger §1's 2.990 ms / 66 % projection row is the
     **PACK_X=false** number and does not describe the engine's config.
   Fix: revert `kernels_proj.cpp` to `PACK_X=false`, or record the tradeoff in `engine/results.md`
   with both numbers; run the glue gate at 1024 (§4-D1) before anything links. Neither
   `engine/README.md` nor `engine/results.md` mentions either config divergence. **[source]**
6. **`gdn_decode`'s two ledger rows are the one timing set in the repo that violates every timing
   rule the repo adopted today.** `prototype/gdn_decode/gdn_decode.hip.cpp` (rec mode) is the only
   remaining per-iteration `hipEventElapsedTime` loop with `best = std::min(best, (double)ms)` and
   **no `ms > 0` guard** — the exact timer `d81ff13` and ledger §6 disown ("`0.000 ms → inf GB/s`").
   Its rows in `performance-status.md:19-20` (`0.062 ms / 918 GB/s / 387 %` and
   `1.344 ms / 255 GB/s`) carry no `util` record and no repeat spread, contrary to §5's own
   provenance rule, and **`gdn_decode` is not on §5's re-take list** — yet `0.062` is taken straight
   into §6's token sum (`:193`, "0.062 (GDN state, §1)"). `check.py:299` prints "% of the 237
   probe" but asserts nothing, so a 387 %-of-probe number passes silently; `lm_head/run.py` and
   `proj_gemv/run.py:410,441` do assert the ceiling. The cache-residency explanation
   (`:32-34`) is plausible and is the repo's own load-bearing claim, but the row needs re-taking
   under the burst-clock rule before it stays in a token budget. **[source]**

Runner-up (misleading rather than false): `ded58da`'s "1764 comparisons" — see §3.1.

## 1. Verdict table

Priority commits first, then the rest that carry a measurement, a gate, or a numeric constant.

| sha | claim (message / doc) | what the code does | verdict |
|---|---|---|---|
| `0a062b5` | mirror is bit-exact for the 3-boundary policy; "refuses A-01-as-written (105683) and specimpl.rmsnorm_gated (same 105683)"; kernel "COMPILE CLEAN AND HAVE NEVER RUN ON A DEVICE" | POLICY row is a genuine 6-variant pin comparison (`mirror_probe.py:41-62`); specimpl row is run with a raw-F32 weight, which speccheck never produces; `results.md:26,30,49` still carry the retracted figure | **UNSUPPORTED** (second half) / CONFIRMED (first half) — WORST FIRST 4 |
| `fc444d0` | "the oracle is bit-exact on §04.5 after all — the 105683 was the probe's weight dtype, not the model's" | Mechanistically correct: `specimpl.py:70-86` casts norm weight to BF16 by default; `:146` `n = w * n` then rounds the product in BF16 = the probe's POLICY. The amendment's own "unresolved 74860 vs 105683" is the same formula at two weight dtypes (see WORST FIRST 4) | **CONFIRMED** (one number pending §4-B1). Gap: it amended only `bench/gate-audit.md`; the probe, its results.md and `engine/` still carry the old claim |
| `27dbaed` | state sized from §08.1; four TUs "COMPILE CLEAN with `hipcc -O3` and zero warnings"; `state_sizes.py` reproduces 18.0 MiB and 12288 B/token and "flags three doc slips" | Arithmetic checks out exactly: `18·16·128·128·4 = 18.0 MiB`, `18·3·6144·4 = 1.265 MiB`, `2·2·256·2·6 = 12288 B/token`, `12288·262144 = 3.000 GiB` (not §00.5's 3.07). Slips are indeed flagged in `state_sizes.py:66-79` — and `state.h:14` contradicts them. Compile claim is not reproducible from the repo: `engine/build.sh` builds only `loader_test` | CONFIRMED (arithmetic) / **DEVICE-PENDING** (compile) / **UNSUPPORTED** (`state.h:14` doc line) |
| `f79fe0f` | hand-off README: "CPU commands (state_sizes PASS, gdn_gated_norm mirror ndiff 0)", device commands with expected results, "≈66 cast launches/token", "the two launch-count deferrals" | README structure is good and honest about what cannot run. But `:83-84` asserts the retracted oracle mismatch (WORST FIRST 1) and `results.md:101` credits an unwritten helper (WORST FIRST 3). 66 casts: no cast launcher exists; the ≈358-vs-≈225 unpacked-launch deferral is arithmetic on measured launch cost, which the file says | **UNSUPPORTED** (2 statements) / CONFIRMED (rest) |
| `d81ff13` | "time with burst wall-clock, because hipEvents returned exactly 0.000 ms for a 2 ms kernel" | `lm_head_gemv.hip.cpp` now runs 1 warm-up + `nb=5` bursts of `iters/5` launches, `best` = min per-launch wall time, and prints `RESULT mode=… ms=… gbs=…` key/value; `run.py` parses by **name** (`fields["ms"]`), so the field order is not load-bearing. No event path left in that binary | **CONFIRMED** |
| `4ca456a` | ledger §6: GEMV families reproduce within 2 %, glue does not; token sum 7.975 ms → ~125 tok/s | Sum is internally consistent from its own components (2.108 + 2.605 + 3.039 + 0.150 + 0.062 + ~0.011 = 7.975) and 1000/7.975 = 125.4 ✓. Two of the six addends are not the engine's config: 3.039 ms is PACK_X=false (WORST FIRST 5) and 0.062 ms is the never-re-taken gdn row (WORST FIRST 6) | CONFIRMED (arithmetic) / OVERSTATED (as a description of the engine) |
| `d06b4e3` | revert "apply_rope must round the rope table to the activation dtype" — "the pins are right, this oracle is the wrong place for it" | The revert restores the state the pins are reproduced by in `attn_decode` and un-breaks G6/G3-smoke. But `specimpl.apply_rope` still returns F32 (`rope_cs` is F32 → promotion), and the call-site comment at `specimpl.py:377` still claims the score matmul is in "the activation dtype (bf16)" — so speccheck is green *because of* an unstated F32 promotion. G6 is wide open at HEAD | CONFIRMED (revert is right) / **UNSUPPORTED** (the code comment G6 asked to fix was not fixed) |
| `5779357` (loader) | `engine/loader_test` gate, 52 checks incl. the tied `lm_head` single-allocation claim | `g_checks` is incremented only in `ok()` (`:45`) and `copy_back()` (`:126`); full verify path = 50 + 2 copy-backs = 52 ✓, `--no-verify` = 50 ✓ (matches ledger §5:189). The tied claim is really asserted: `lm_head.weight` is aliased to `embed_tokens`, same device pointer, and allocation count == loaded tensor count | **CONFIRMED** |
| `ce37aa9` (devices D2H) | 63–70 GB/s bulk D2H vs the loader's 27.7 GB/s chunked copy-back is a copy-pattern effect, not contention | `tools/bw_probe_hip.cpp` does bulk `hipMemcpy` into pinned host memory, so the two numbers are measuring different things — that part is mechanical. The attribution (pattern, not contention) is not backed by a chunked-vs-bulk sweep | CONFIRMED (mechanism) / **DEVICE-PENDING** (attribution, §4-D4) |
| glue: `fc33b69` | one fused add+norm kernel, three instantiations, negatives compiled in | `glue_kernels.h` `glue_add_norm<THREADS>` + `residual_add`/`norm_only` unfused pair + 6 `Neg` values; `--split-check` asserts fused == unfused bit-for-bit (`run.py:408-411`) | **CONFIRMED** |
| `bc78262` (glue gate) | "784 comparisons"; "bit-exact residual adds, 384 norm outputs against the s_in/s_out pins and the cross-stage projections" | `ncmp = 4·steps·layers + 2·steps·(0 ∈ layers)` = 4·8·24 + 2·8 = **784** ✓ (= 384 adds + 384 norms). Norm targets are `mlp.in` (a real pin) for `g1_n` and the torch mirror + cross-stage projection for `g2_n` (only the final step has a pin), so "s_in/s_out pins" names the wrong pin for the norm half — the gate is stronger in one place and weaker in the other than the sentence implies | CONFIRMED (count) / OVERSTATED (pin description) |
| `119d533` (glue floor probe) | 384 norms × 1024 = 393,216 values; norm floor is 1 ULP with no cancellation band; `bf16sub` 2.05 %, `bf16acc` 99.93 % @ 45 ULP | Table at `results.md:98-106` and `floor_probe.py` agree; `results.md:111-114` shows `CANCEL_REL=3e-5` excuses **0 of 393216** and the tightest passing band is 0, i.e. the band carries nothing here — that is the provenance of "1 ULP floor, no band", and it is a measurement, not taste. Caveat: it is the **256-thread** reduction order | **CONFIRMED** (provenance real) / scope caveat (WORST FIRST 5) |
| `04caf9e` (glue results) | fusion buys 0.0795 ms/token ≈ 1 % of token; two windows agree; `loader.h` served all 49 norm vectors | 0.2155−0.1360 = 0.0795 ✓, /7.81 = 1.02 % ✓; windows A/B agree to 1.8 %/1.1 % (`:159`); 49 = 24+24+1 norm vectors, 98.0 KiB, `--loader-verify` was used. Ledger §5 later shows these are the **favourable** windows (re-take gave 0.150–0.185 ms), and §6/§5 say so | **CONFIRMED** (arithmetic) — superseded values, documented as such |
| `ccafece`/`9ccad93` (ledger glue rows) | glue fused 0.1336–0.1360 ms/token, unfused 0.2131–0.2155 | Same as above; ledger §5:187 now marks the re-take "worse, and unstable, spread 2.7–269 %" | CONFIRMED + correctly re-marked |
| `ded58da` (attn gate) | "1764 comparisons green on 48 decode + 20 long-cache cases", 12 located negatives | 1764 reproduces exactly: 432 (48×9 judged) + 1152 (3 cross-variant passes × 48×8) + 180 (20×9). But 1152 of 1764 are re-runs of the same 48 cases under other knobs, 768 of those are counted and **not judged** (`gate_it=False`, `run.py:379,385`), and the long-cache rows come from **one** attention layer (`run.py:393` `layers[:1]`). Judged coverage is 996. Negatives: 12, each with must-catch + must-not-catch asserted | **OVERSTATED** |
| `80b880c` (attn floor probe) | reordering this op's reference needs no band (0 of 15592, 0 of 108544); streaming softmax is a policy difference on 26 % | Probe output matches the message; `run.py` prints "tightest band that would pass" so the margin is on the record | **CONFIRMED** |
| `307f6cf` (attn results) | rope policy is BF16-rounded cos/sin (52–97 of 2048 missed by the F32 reading); two-pass softmax; long cache to 277 lanes; `apply_rope` should be fixed in speccheck | `results.md:28-35` measures it and the verify section **requires the misreading to fail** — a positive-and-negative pair, which is the right shape. The "action for speccheck" was not taken (it was reverted in `d06b4e3`, with reason) | **CONFIRMED** |
| `573bf3b`, `7762972` (gate-audit G6/G2/G7 notes) | G6/G3-smoke bands encode plumbing; G2's F32 rope is now a decision | G6 is accurate: `specimpl.apply_rope` returns F32 and the score matmul runs F32, opposite to the comment at the call site. G2's "deliberate and documented" half is true of `apply_rope`'s docstring; the call-site comment remains wrong | CONFIRMED (audit text) / finding **still open** |
| `a72bbf5` (mlp F4 bands) | F4 judged by cos/meanrel with a measured legal floor; ndiff removed as blind for chains | `run.py:81-82`: `F4_COS_MIN=0.9999` ("measured legal worst 0.99999615 → 26×"), `F4_MEANREL_MAX=5e-3` ("measured legal worst 2.01e-3 → 2.5×"). Both moved **up** from 0.9999/1e-3, and the old `ndiff ≤ 15 %` rule was **deleted** because one legal 1-ULP act flip measured 29.1 %. See §2 | **OVERSTATED if read as "no band was moved"** — see §2 answer |
| `f49df71`, `5536d71`, `a008805`, `ea2e175` (mlp negatives / probes / "measurement behind every tolerance") | 6 negatives with must-catch and must-NOT-catch; floor probe sizes the band | `NEG_KERNEL` entries with per-negative must/must-not sets asserted in `run.py`; `ulp_probe.py` has the `uint16` dead-branch trap fixed and a selftest; F4's negative `f32act` measures 4.14e-3, i.e. **inside** the new 5e-3 band | CONFIRMED; the F4 blind spot is real and is named in §2 |
| `c12
