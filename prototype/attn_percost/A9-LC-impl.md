# A9-LC-impl — L-c (cap the attention split) is implemented behind `--attn-split-cap`, and the bet's central risk turned out to be form-dependent

Written against tip `8b0a3b4` by the implementing worker. Bet: `prototype/attn_percost/A9-BET.md`.
Predictions: A9 §2(c)/§4.2 of `prototype/attn_percost/A9-LONGCTX-CENSUS.md`.
**No timing in this file.** Every millisecond figure belongs to the main agent's window; the engine's own
`RESULT`/`WAVEPASS` lines appeared in the logs of the runs below and are deliberately not quoted.

---

## 0. Status board

| item | state |
|---|---|
| the engine knob (`--attn-split-cap N`, default = no cap, byte-identical) | **done**, `engine/build.sh` clean (18.2 s, both binaries) |
| the central question (does a cap break the lane ceiling?) | **answered from source**, §1; witnessed by `a9_lc_geom_check.py` |
| a second refusal the answer forced (VALU tiled wave + a binding cap = LDS overrun) | **done**, §3 — a host-side error, zero registers, zero LDS, no kernel touched |
| CPU verification | `a9_lc_geom_check.py` SELFTEST PASS + INJECT PASS(4) — incl. **[E]**, the merge re-association on the mirror's own reductions judged by the imported `CASCADE_BAND`; `tools/attn_ceiling_proof.py` **PROOF PASS** unmodified; `a9_waste_model.py --selftest` PASS |
| one device gate (correctness only) | **done, PASS**, §5 — capped 4,608-lane run sane, refusal control fires, **no timing taken** |
| the A/B harness `run_ab_a9lc.sh` | **done**, `bash -n` clean, `SELFTEST=1` passes all five scenarios (incl. the canary-abort) |
| `A9-BET.md` | **done** |
| numerics at the candidate cap (`engine/gate.py` etc.) | **handed over**, §6 — they are device gates longer than a worker's minute |
| the paired window | **RUN by the main agent — verdict DISPLACED**, see `A9-LC-verdict.md`: family −4.61 % (line cleared), wave wall −2.38 % (line missed). Not shipped. |
| inertness of the shipping default | **PASS on device** (`tmp/a9lc/inert2.sh`): committed binary with no flag vs `8b0a3b4` at 4,608 lanes → `logits4607.bin` + `sampled.txt` byte-identical, identical geometry line, `SPLITCAP cap=0 span=512`. No timing taken. |
| gate-forwarding of the flag into `engine/gate.py` | **not applied** — moot while the default stays inert, and the rule (§4) is that a shipping default and the gate's default move in one commit. |
| commits | **not made**, by division of labour; proposed split at §8 |

## 1. The question the brief asked first, answered: **(i) for the shipping form, (ii) for the two VALU tiled spellings**

The assumption in A9 §4.2 is that a capped split keeps every lane by making each block's slice wider
(`cap 8` at 32,768 lanes → 4,096 lanes/block, 256 rounds). That is exactly what the row-slicing arithmetic does —
in all three forms:

* `engine/attn_fattn_kernels.h:475`, `engine/attn_slim_kernels.h:169`,
  `prototype/attn_decode/attn_decode_kernels.h:288` and `:598` all compute the same
  `span = (nrow + s_row - 1) / s_row` with `s_row = blocks_for_row(...)`, and the block owns
  `[lo, min(nrow, lo+span))`. Nothing is lost: `a9_lc_geom_check.py` **[A]** proves
  `Σ span_y = lanes` for 1,970 plans (widths `{512 … 40,000}` × caps `{0,2,3,4,5,8,16,32,64,129}`).
* **But** the WMMA wave and the two VALU tiled waves spend that wider slice differently:
  * `attn_fattn_wave` stages `FATTN_MLANES = 16` lanes a round and its round loop runs
    `for (int base = lo_all; base < hi_all; base += FATTN_MLANES)` (`engine/attn_fattn_kernels.h:771`) — bounded
    by the row's own extent, **not** by any constant. Wider slice ⇒ more rounds. This is (i), and it is the
    **shipping** path (`--attn-fattn` defaults to 1, `engine/decode_run.cpp:152`, `engine/decode_step.h`).
  * `attn_slim_wave` declares `__shared__ hip_bfloat16 s_sc[ATTN_TILE_G][MAXSC]`
    (`engine/attn_slim_kernels.h:157`) and writes it at `:232` with the slice-absolute index `rr`, where
    `rr < ns == span`; the frozen `attn_tile_body` has the same buffer
    (`prototype/attn_decode/attn_decode_kernels.h:586`, `span` at `:598`). So `span > MAXSC` is a **shared-memory
    overrun** — and AGENTS records that a device fault on a staged wave path presents as a *hang*, not a crash.
    This is (ii).
  * `attn_core` (every one-row body, and every row under `--attn-tile 0`) declares
    `__shared__ float sc[(MAXSC < TILE) ? TILE : MAXSC]` (`attn_decode_kernels.h:253`) but indexes it by
    `sc[ns + w]` in the one-pass branch (`:323`, reachable only while `s_row == 1`, and `lanes_of ≤ MAXSC`
    there by the throw below the launcher's ceiling) and by the window-relative `sc[w]`, `w < nt ≤ TILE`, in the
    streaming branch (`:360`). **No branch indexes `sc` by the span**, so `attn_core` absorbs a wider slice as
    more rounds — (i).

**Consequence, and it is the whole shape of the change:** a *binding* cap (one that actually lowers the split) is
legal exactly where the WMMA wave owns the streaming rows. The shipping configuration satisfies that, so the
funded A/B is unaffected; the knob is simply refused where it would overrun a staging (§3). No kernel changed, no
register moved, no LDS byte moved.

Ceiling question, separately: **the ceiling does not move.** The launcher's refusal
(`engine/kernels_attn.cpp`, the `split > kAttnMaxSplit` throw) is priced on the **uncapped** `split` and runs
*before* `split_eff` exists, so `MAXSC × kAttnMaxSplit = 32,768` is still the largest lane count the engine will
admit no matter what cap a caller passes. That ordering is asserted in source order by
`a9_lc_geom_check.py` **[D]**, and `tools/attn_ceiling_proof.py` — which censuses every *code* occurrence of
`kAttnMaxSplit` and refuses to pass if one is unaccounted — still prints **PROOF PASS** with no edit, because
the "no cap" sentinel is `kAttnSplitCapOff = 0` rather than a copy of 64, so no new reader of the constant was
introduced (§2).

And `blocks_for_row` (`prototype/attn_decode/attn_decode_kernels.h:220-223`) **reads `a.split`**; it does not
recompute it (`return a.split > 0 ? a.split : 1;`). The second spelling the brief asked about,
`split_for_lanes` (`:190`), exists and is used by `tile_split_of`/`tile_part_floats`, i.e. by the *frozen*
prototype's own geometry helpers and by `lane_ceiling`-style arithmetic — not by the launcher. **One decision
site**, as A9 assumed.

## 2. The engine change

`--attn-fattn-qstage` was the pattern: option → `Options` → `DecodeScratch` → `AttnLayerArgs` field →
one use at the decision site; default inert; help text printed.

| file | what |
|---|---|
| `engine/kernels.h:593` | `constexpr int kAttnSplitCapOff = 0;` — the sentinel is "no cap", deliberately not `64`, so the constant census sees no new `kAttnMaxSplit` reader |
| `engine/kernels.h:700` | `int split_cap = kAttnSplitCapOff;` in `AttnLayerArgs`, with the note about why it never feeds a size |
| `engine/kernels_attn.cpp:95` | `cap_split(split, cap) = cap >= 2 ? min(split, cap) : split` — one expression, and `cap == 0` returns `split_for(lanes)` verbatim |
| `engine/kernels_attn.cpp:227` | `splitcap_echo()` — the A/B canary line (see §4), deduped per geometry exactly like `fattn_echo` |
| `engine/kernels_attn.cpp:336` | refusal of `split_cap == 1` |
| `engine/kernels_attn.cpp:343` | `const int split_eff = cap_split(split, a.split_cap);` — **after** the ceiling throw |
| `engine/kernels_attn.cpp:372` | `c.split = split_eff` — the one field every grid, `blocks_for_row` and `attn_combine` read, so writer and merge cannot disagree |
| `engine/kernels_attn.cpp:428` | the VALU-route refusal (§3) |
| `engine/kernels_attn.cpp:458` | `splitcap_echo(rows, lanes_last, split, split_eff, a.split_cap)` |
| `engine/decode_step.h:116`, `.cpp:729` | `DecodeScratch::attn_split_cap` (default inert) → `a.split_cap`, both the wave and the single-row branch |
| `engine/decode_run.cpp:190/397/1770`, usage `:855`, help `:962` | `--attn-split-cap N`: strict parse (rejects non-numeric, negative, and `1`), `0` = no cap |

Every downstream consumer moved to `split_eff`: the three grid builds (`attn_tile_wave` / `attn_slim_wave` /
`attn_fattn_wave*`), `attn_core`'s `split=1` launch, `wave_fattn`'s `>= 2` test, `fattn_echo`, and the
`if (split_eff > 1)` guard in front of `attn_combine`. Sizes did **not** move: `attn_part_floats(rows)` is sized
from `kAttnMaxSplit` (the ceiling), so a shallower split leaves the tail of each row's slab unwritten —
`a9_lc_geom_check.py` **[B]**: worst reachable slot index **262,143 of 262,144**.

Two witnesses that "zero registers, zero LDS" is a fact and not an intention: `git diff --stat` touches **no
kernel header** (the wave spellings live in `engine/attn_fattn_kernels.h`, `engine/attn_slim_kernels.h` and
`prototype/attn_decode/attn_decode_kernels.h`, all unchanged), and the capped run's own launcher banner prints
`vgpr=185 lds=20480 wf_per_cu=9` — identical to the published G1 figures at `A6-G1-impl.md:130-148`. The new
field is in the **host** struct `AttnLayerArgs`; every `__global__` in this family takes `CoreArgs` (or
`FattnArgs`), which is unchanged, so no kernarg grew either.

## 3. Two refusals, both host-side, both errors (never a silent fallback)

* `--attn-split-cap 1` — refused at parse *and* in the launcher: one lane block on a wide row is the
  mis-normalisation §07.3 exists to make unreachable (a partial that has already divided by its own `l` cannot be
  re-merged), so it is not "the shallowest cap".
* a **binding** cap where the VALU tiled wave owns the streaming rows (`rows > 1 && wave_tile != 0 &&
  wave_fattn == 0`). Message, verbatim from the device gate:

  > `attn_decode_layer: --attn-split-cap 2 binds here: widest row is 1030 lanes, so a split of 2 gives each
  > block 515 lanes against a staging of MAXSC=512. The VALU tiled wave (--attn-fattn 0) stages a whole block
  > slice in s_sc[ATTN_TILE_G][MAXSC], so a block wider than the staging is an LDS overrun, not a slower run.
  > The WMMA wave stages 16 lanes a round and absorbs a wider slice as more rounds: run --attn-fattn 1 (the
  > shipping default) with a cap, or drop the cap.`

  The condition keys on *binding* (`split_eff != split`), so `--attn-fattn 0 --attn-split-cap 64` — which cannot
  bind — still runs, and an existing `--attn-fattn 0` gate cannot go red from a knob it never passes. Precedent:
  `prototype/attn_m2/attn_m16.hip.cpp:1069-1075` refuses `split < split_for_lanes(lanes_last)` for the same
  reason.

## 4. What `engine/gate.py` must know, and what it must not

* **`lane_ceiling()` needs no change.** It regexes `MAXSC` from `attn_decode_kernels.h`, `kAttnMaxSplit` from
  `kernels_attn.cpp`, multiplies them, and checks the run's `ceil(lanes/MAXSC) * MAXSC <= ceiling`
  (`engine/gate.py:397-429`). None of those three numbers moved; the cap cannot raise the ceiling (§1). Editing
  it would have *weakened* the check it exists to make.
* **If gate.py is ever asked to judge a capped run**, it needs the flag forwarded into every engine leg — that is
  gate policy, so it is handed over rather than applied. The pattern to copy is the `--attn-fattn-qstage`
  forwarding: the parser (`engine/gate.py:7247`), the three argv inserts (`:5257`, `:7496`, `:8547`), and the
  NUMERICS-echo assertion (`:7687-7699`) that refuses to judge a binary which did not echo the flag it was asked
  for. i.e.:

  ```diff
  --- a/engine/gate.py
  +++ b/engine/gate.py
  @@ add_engines(...)
       ap.add_argument("--attn-fattn-qstage", default="stage",
                       help="... an A/B switch: every leg gets the same value ...")
  +   ap.add_argument("--attn-split-cap", type=int, default=0,
  +                   help="L-c (A9-BET): an UPPER BOUND on the split, forwarded to EVERY leg. "
  +                        "0 = today's geometry. A/B evidence only: a NUMERICS comparison must not mix "
  +                        "capped and uncapped legs, because the cap re-associates attn_combine.")
  @@ the three leg-building sites (5257 / 7496 / 8547), next to the qstage insert
  +   if args.attn_split_cap:
  +       cmd += ["--attn-split-cap", str(args.attn_split_cap)]
  ```

  If the stronger form is wanted, `qstage`'s precedent is to echo the value on the run's `NUMERICS` line and
  assert it there; the launcher already prints the equivalent fact on its own line (`SPLITCAP split= cap=`),
  which is what `run_ab_a9lc.sh` reads.

  plus, if the flag is ever given a shipping default, the same-commit rule the `--attn-tile` trap row records:
  `gate.py`'s default and `decode_run.cpp`'s must move in one commit, or the gate silently judges a different
  binary than the engine ships.
* `tools/longctx_check.sh` has **no** flag passthrough at all today (only `--engine`, `--model`, `--py`, `--out`,
  `--suite`, `--only`, `--quick`, `--list`), so a wide-context comparison of a capped engine would need
  `--engine` plus a wrapper script that appends the flag. Also handed over, not done.

## 5. What was run

CPU only, no lock needed:

```
$ python3 prototype/attn_percost/a9_lc_geom_check.py            → SELFTEST PASS   ([A] 1,970 plans / 0 defects,
                                                                                 [B] worst slot 262,143/262,144,
                                                                                 [C] span census, [D] source order,
                                                                                 [E] 7 capped merge rows / 0 outside)
$ python3 prototype/attn_percost/a9_lc_geom_check.py --inject   → INJECT PASS (4 controls, all fire)
$ python3 tools/attn_ceiling_proof.py                           → PROOF PASS, rc=0 (unedited; [A] census green)
$ python3 prototype/attn_percost/a9_waste_model.py --selftest    → PASS (A9 §4.2 reproduced)
$ python3 prototype/attn_percost/a9_waste_model.py --caps --cap-width 32768
        → cap 8: 24.500 GB/pass vs 105.457 at 64, 5.87 block-waves, 256 rounds/block, depth +2.13 %
$ python3 prototype/attn_tile_model.py selftest                  → 13/13 PASS (the frozen tiled geometry,
                                                                   re-read from an unchanged header)
$ bash -n prototype/attn_percost/run_ab_a9lc.sh                 → clean
$ SELFTEST=1 bash prototype/attn_percost/run_ab_a9lc.sh          → SELFTEST PASS (ship/falsified/displaced/
                                                                    unpaired + canary-abort, fixtures from the
                                                                    harness's own constants)
$ engine/build.sh                                               → 18.2 s, both binaries, no new warning
                                                  (baseline `kernels_attn.cpp` also warns: cannot tail `attn_core`)
```

**What [E] prices, and why the device number is bigger.** On the mirror's own `online`/`merge_partials`, the
re-association itself is `meanrel ≈ 3–6e-7` and `1-cos ≈ 0` — four orders inside the band. The device's
end-to-end figure below is `1-cos 8.3e-4`, because a 4,608-lane prefill runs 48 layers and rounds to BF16 at
every one of them (the chaos floor, `specification/ASSUMPTIONS.md` A-18). Two consequences kept in view:
the merge is not where the risk is, and — from the injection — a mis-normalised merge is a *scale* error, so it
reads `cos = 1.000000` while `meanrel` jumps to ~2. **Cosine alone cannot adjudicate this route; `meanrel` and
the top-1 room can.**

Device — **one** short correctness-only gate, blocking, one `gpu_lock` hold, script kept at
`tmp/a9lc/gate.sh` (scratch, not committed):

```
$ time bash tmp/a9lc/gate.sh
util@acquire = gpu use: 91 %      util@release = gpu use: 24 %      wall = 4.8 s
A (default)   rc=0   SPLITCAP split=9 cap=0 lanes=4608 natural=9 rows=512 span=512
B (--attn-split-cap 8) rc=0   SPLITCAP split=8 cap=8 lanes=4096 natural=8 rows=512 span=512
                            SPLITCAP split=8 cap=8 lanes=4608 natural=9 rows=512 span=576
   LAUNCHSTAT family=attn calls=54 calls_batched=54 rows_total=27648 rows_max=512   (identical on both legs)
C (--attn-split-cap 2 --attn-fattn 0, 1,030 lanes) rc=1   refusal message as quoted in §3
logits4607.bin  A vs B: n=248,320 ndiff=248,319 maxabs=9.94e-02 meanrel=1.376e-02 cos=0.999917441
                nonfinite: 0 NaN / 0 Inf on both sides; argmax 82 on both (top-1 margin 1.085 vs 1.161)
```

What that buys, and what it does not:

* **PASS** — a binding cap runs to completion on the shipping form at a span of **576 > MAXSC = 512**: the slice
  widened, no fault, no NaN, the attention launch count unchanged. The last-row logits sit inside the published
  band for a reassociating route (`engine/gate.py:892-902`'s `CASCADE_BAND`): `1-cos = 8.26e-4 ≤ 2.0e-3`,
  `meanrel = 1.376e-2 ≤ 3.0e-2`, top-1 id identical (82), zero non-finite on either side.
  Note which field carried the verdict: `cos ≥ 0.9995` (the §10.1 pin floor) is also met, but a whole-vector
  scale error would pass it — see the [E] note above.
* **PASS** — the refusal control (C) fires with the right message and a non-zero exit; the must-catch for §3.
* **NOT proven here** — that the *default* is byte-identical to `8b0a3b4`. The argument is exact (`cap_split`
  returns its input when `cap == 0`, and no size or order moved), and the A/B harness re-checks the geometry in
  the measuring window, but the byte witness is a main-agent command (§6 item 0). A second worker gate would
  have exceeded this worker's device allowance, so it was handed over instead of run; `tmp/a9lc/gate_inert.sh`
  holds the ready-made command.
* No milliseconds were taken, quoted, or recorded anywhere in this file or the bet. The `RESULT`/`WAVEPASS` lines
  in `tmp/a9lc/{A,B,C,D}.log` are the engine's own and are not evidence about this change.

## 6. Handed to the main agent (device; run blocking, zero subagents, one lock hold per item)

```bash
# 0 — the inertness witness (2 s): default output must be byte-identical to the pre-change binary.
#     Base worktree + binary already exist: /tmp/a9lc-base-8b0a3b4 (built by the harness's own build leg).
bash tmp/a9lc/gate_inert.sh          # prints "INERT: logits4607.bin byte-identical" or "DIFFERS"

# 1 — the A9 §4.2 numerics admission at the candidate cap. Read the count, not the grep: if engine/gate.py is
#     ever taught the flag (§4), every leg gets it; otherwise this judges the DEFAULT (inert) path, which is
#     what item 0 already covers — run it as the pre-change-parity check it is.
bash prototype/attn_wmma/run_fattn_admission.sh

# 2 — the wide-context checks at the cap (needs the §4 forwarding or a wrapper; --attn-split-cap is NOT in
#     tools/longctx_check.sh today).
engine/gate.py ... --attn-split-cap 8          # after §4's diff is applied, if gate policy says so

# 3 — the paired window (the bet's verdict). Everything inside one gpu_lock hold; prints SHIP / FALSIFIED /
#     DISPLACED / UNPAIRED plus the externality sum and the canary line.
bash prototype/attn_percost/run_ab_a9lc.sh
#     env: CAP=16 to re-point at another candidate, REPS=3 for a third rep, SKIP_DEVICE=1 to re-parse saved
#     logs. NOTE, corrected after the window: CANARY_ONLY=1 is NOT a pre-flight — it re-judges the canary over
#     logs already in $OUT and, on an empty $OUT, its empty-parse guard aborts (correctly) instead of running the
#     device. Run the window plainly; CANARY_ONLY is for re-checking a window already taken.
```

## 7. Side effects on the tree and the box, disclosed

* `git worktree /tmp/a9lc-base-8b0a3b4` (detached at `8b0a3b4`) + a built binary inside it — created by an
  accidental early invocation of the harness (`DRY_RUN` was unset). It is the right BASE and saves a rebuild;
  remove with `git worktree remove /tmp/a9lc-base-8b0a3b4` if unwanted.
* One **pre-lock** device run happened before I read the GPU rules closely:
  `engine/build/decode_run --ids 1 --steps 8 --attn-split-cap 0` (~0.3 s, `ms_total=101`, `gpu_lock.sh status`
  was free, no timing recorded anywhere). Reported, not hidden.
* Scratch, gitignored and safe to delete: `tmp/a9lc/` (`gate.sh`, `gate_inert.sh`, `cmp_logits.py`,
  `ids4608.csv`, `ids1030.csv`, `A.log`…`D.log`, `unc/ c8/ base/`).
* Nothing under `bench/` or `prototype/perprompt/` touched; no gate, band, dump or `results.md` edited; nothing
  `--ack-spill`ed; nothing staged.

## 8. Proposed commit split (not executed — commit hygiene is the main agent's)

1. `engine: --attn-split-cap, an upper bound on the attention split` — the five engine files (knob + canary +
   both refusals). Builds standalone: the canary and the refusals are part of the knob.
2. `prototype/attn_percost: A9-L-c bet, geometry witness, A/B harness` — `A9-BET.md`, `a9_lc_geom_check.py`,
   `run_ab_a9lc.sh`.
3. `prototype/attn_percost: A9-LC-impl` — this file, after the device verdicts are in, so the status board can
   record them (or land it now and amend).

## 9. What L-c does not do, and what is still open

* **It does not make L-c cheaper for the frozen VALU forms.** If §52 ever wants a cap under `--attn-fattn 0`, that
  is a kernel change (stage `min(span, MAXSC)` a round in `attn_slim_wave`, i.e. stop indexing `s_sc` by
  slice-absolute `rr`) — outside the zero-LDS promise and explicitly not done here.
* **The prediction is unspent.** A9 §4.2's −4.2…−4.7 % of the 32k wave assumed the 88 %-of-roof reading of
  `attn_combine`. LINE 1 of the bet falsifies or confirms exactly that, and D21 is the standing reason to expect
  FALSIFIED.
* **Depth cost is real but small**: +2.13 % serial chain at cap 8 (`a9_waste_model.py --caps`). At 32k the wave
  is still ~41.5 full-width block-waves deep at cap 8, so there is no parallelism cliff; if LINE 1 clears and
  LINE 2 does not, that depth is the first suspect and DISPLACED is the honest word.
* **`span` is unbounded above by construction** (`ceil(lanes/cap)`), bounded only by the ceiling throw
  (`≤ 32,768`) — i.e. at cap 2 the widest legal row is a 16,384-lane slice in one block. Legal for the WMMA form,
  and the harness will say in seconds what it costs; nothing in the bet funds a survey of caps below 8.
