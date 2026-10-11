# Gate audit — 2026-10-11

Adversarial read-only audit of all six graduated gates, asked one question: **is there a check that
would let a wrong kernel pass?** Written down here rather than left in a transcript, because every
finding below is a hole someone could fall into while the engine is being built on top of these gates.
Merge verdict was **OK with notes**, with one item to close before the engine's token loop ships on
`lm_head_gemv`'s gate.

Dispositions are mine (supervisor). "Quiescent pass" = §5 item 0 of `performance-status.md`; those
items need device runs and cannot be closed by a worker under the serialization rule.

## Open — would let a wrong kernel pass

### G1 `lm_head_gemv` has no per-element rule at all — **the engine must not ship on this gate**

`prototype/lm_head_gemv/run.py` asserts argmax, `cos ≥ 0.999` and `meanrel ≤ 0.05`, and **prints**
top-5 match and `max|d|` without asserting them. There is no `bf16_ulp` comparison, no ndiff rule,
no floor probe and no negative suite in that directory. The band is the §10.1 *cascade* class
(0.999 / 5 %) — roughly 36× the kernel's own measured residual against the pin (meanrel 0.00141) —
while §10.1 designates G3i (0.9999 / 1e-2) as the class a kernel is signed off against.

Two concrete wrong kernels pass it today, both named in the audit:

- **row misplacement** — writing row *i* into row *i+1*'s slot, the same `rowswap` bug class that
  `mlp_gemv` and `proj_gemv` declare as must-catch. Swapping two non-argmax rows changes 2 elements
  of 248 320, leaves the argmax alone, meanrel ≈ 1e-5. Passes all three asserted checks.
- **a dropped term in the reduction** — `mlp_gemv`'s measured `tail` (8 of 1024 terms dropped) gives
  meanrel 9.8e-2, so a 1-term drop lands near 1.2e-2, inside the 0.05 band, with argmax and top-5
  intact. Anything between 1.4e-3 and 5e-2 meanrel passes.

This is the gate the engine's logits ship on, so it is the one hole in this list that matters for W6.
**Cheapest closure:** assert top-5 equality and a `bf16_ulp` per-element rule with a band sized by a
floor probe for a K=1024 dot at 248 320 rows (reuse `proj_gemv`'s K=1024 floor as a starting point
but re-measure — it is a different row count and a different accumulator width), plus negatives for
row misplacement, dropped tail and a `rowswap`, each with must-catch and must-NOT-catch declared.
Worker can write all of it; the device run belongs to the quiescent pass.

### G2 `specimpl.rmsnorm_gated` has no bit-level assertion anywhere — same failure mode as `apply_rope`

`speccheck/specimpl.py:142-147` implements §04.5's gated RMSNorm with a cast order "pinned by A-01",
and **nothing checks it at bit or ULP level**. `proj_gemv/operands.py` consumes `gdn.gated_out` as a
raw pin and never recomputes it; `gdn_decode` stops at `o_pre_norm`; the only comparison of any
gated-norm implementation anywhere is `speccheck/gates/run_isolated.py` at `cos ≥ 0.9999`. A 1-ULP
cast-order policy error there is invisible to every gate — which is exactly how `apply_rope`'s wrong
rope policy survived 26 green gates.

Update from the rope work: `apply_rope`'s F32 form is now *deliberate for this oracle* and documented
in its docstring — the model's policy is the BF16 one (§05.3), and bit-exactness against the pins lives
in `prototype/attn_decode`, where the inputs are the pins. Making the oracle bit-faithful to it cost
three end-to-end gates (d06b4e3). That does not soften this item: `rmsnorm_gated` has the same blind
spot, and §04.5 has no pinned-input gate anywhere to fall back on the way rope has one.

`gdn_chunk_scan` (§04.7) has the same weakness: asserted only at cos bands (G3, chunk == recurrent).

**Cheapest closure:** a bit-exact mirror of `rmsnorm_gated` against the pinned `gdn.gated_out` dumps
inside `gdn_decode/check.py` — the pins already exist and that fixture already loads them. This is
the same move `attn_decode/reference.py::verify` makes for rope: prove the helper against the pin
before letting it stand in for one. **CPU-only, so a worker can close it today.**

## Open — weaker than it looks

### G3 `glue`'s norm ndiff band is a borrowed dot number, and it is the only detector of its target bug

`prototype/glue/run.py` uses `NDIFF_PCT_MAX = 2.0`, which `glue/results.md` §7 already admits is "the
inherited 2 %". It matters uniquely because of the trap the floor probe exposed: for a norm, a BF16
subpartial is invisible to every max-ULP criterion (2.05 % of elements, all ≤ 1 ULP), so **ndiff is
the only live detector**. Measured: legal floor 3 / 393 216 = 0.0008 %, the injected bug at 4.0 %
aggregate — a ~2× margin, and a half-severity variant (one warp's partial instead of every thread's)
sits under 2 % and passes everything.

**Cheapest closure:** 2 % → 0.5 % in `glue` only (the floor sample supports ~600× on the legal side
and still 8× under the negative), then re-run the negative suite. Constant change is CPU; the re-run
is a quiescent-pass item.

### G4 `gdn_decode`'s `state` vs HF dump is judged at `meanrel ≤ 5e-2`

The loosest band in any graduated gate, and `results.md` does not print the measured value, so the
check is effectively decorative at that width. Not exploitable today only because F3/F4 judge `state`
against the spec mirror at 1e-5 / 1e-6 — that is what would actually catch a BF16-stored state, the
bug this band would otherwise hide. Worth tightening to match the mirror, or deleting so it stops
implying coverage.

### G5 two bands are at ~1.3–2× over their own target bug

`attn_decode`'s `q_rope` policy detector is a count with a 1.3× margin (F32-cos/sin misreading
measures 52–97 of 2048 differing; limit is 41), and `proj_gemv`'s 2 % ndiff has no bite on the 16-row
`in_proj_b`/`in_proj_a` pins (1 element = 6 %), where `n_bad` is doing the real work. Neither lets a
wrong kernel through — the rope case is caught bit-exactly by `reference.py::verify` upstream — but
both are counts that look stronger than they are. Recorded so nobody tightens the wrong one.

### G6 §10-G6 and §10-G3-smoke's bands are calibrated on an implementation accident, not on reference behaviour

Both bands were tuned while speccheck's score matmul was silently running in **F32**. `apply_rope`
promoted `qh`/`kh` to F32 (an F32 cos/sin table promotes the product), so `qh @ K.t()` was an F32
matmul — the opposite of what §05.5 and the code's own comment at that line claim ("the score matrix
is produced and kept in the activation dtype (bf16)"). The bands therefore encode speccheck's plumbing
rather than anything the reference does.

That is not speculation: it is the only explanation of the numbers. Restoring the pinned BF16 rope
policy — which makes `qh`/`K` genuinely bf16, and is bit-exact against the pins in
`prototype/attn_decode` — moved G6's dec8 decode-logits cos from 0.99991951 to 0.99989544 (threshold
0.9999) and made §10-G3's chunk-vs-recurrent check on `smoke` fail at `L22.gdn.state_new`
cos 0.99988423. Forcing **only the scores** back to F32 while keeping the pinned rope restored both:
G6 0.99991511 PASS, G3-smoke PASS. Same model, same pins, same bands; the dtype moved them.

Consequence: any future dtype change upstream of the score matmul moves these two gates again with no
change to the model. **Cheapest honest closure:** state the score dtype explicitly at the call site —
`(qh.float() @ K.float().t()) * S_SCALE`, or the bf16 form — so it is a decision on the record with
the bands calibrated against it, instead of a type-promotion artifact inherited from rope. No band
value is proposed here, and none should move until the dtype is stated.

### G7 §10-G7's corpus floor has ~0.035 of headroom and two legal variants already sit outside it

The corpus gate requires speccheck's mean prefix agreement over 49 prompts × 24 tokens to be at least
the reference's own self-consistency floor, **0.8776** (an sdpa-vs-eager realization of the reference
against itself). speccheck measured **0.9124** — headroom 0.0348, roughly one token per prompt.

Two correct-but-differently-rounded variants of speccheck now sit *below* that floor: **0.8503**
(pinned BF16 rope, F32 scores) and **0.8121** (pinned BF16 rope, bf16 scores). Neither is a bug: both
keep sampled ids identical on all six short fixtures, and both apply a rounding policy the pins
themselves carry. A gate that any legal rounding change can trip is not measuring what it looks like
it measures — it is measuring distance from one particular rounding realization, and its margin is
smaller than one policy change.

Recorded as **the gate most likely to produce a false alarm at W6**, where prefill adds more rounding
decisions (chunked GDN forms, long score rows). This is not a proposal to widen it: the floor is a
measurement of the reference against itself and widening it would delete the only end-to-end invariant
that short fixtures cannot provide. When it trips, the honest first move is to re-measure the floor
for the realization actually under comparison, not to move the threshold.

## Closed / no finding

- **`engine/gate.py` exists (2026-10-11) and is auditable by construction, but has never seen the
  engine.** It is the engine's own token gate — the `dec8` fixture in, 9 sampled ids and 8 logits
  vectors out — and it holds no engine arithmetic: no torch, no mirror of a kernel, only §10's metric
  definitions, `engine/state_sizes.py`, and the §03.4 argmax used to prove the fixture pairs with
  itself. Its band is borrowed whole from §10 (ids exact per G7, logits `cos ≥ 0.9995` per G5,
  `meanrel ≤ 5 %` per §10.2, ndiff/max-ULP excluded per F1–F4) rather than fitted, and it re-derives
  the token's launch plan in Python instead of trusting the C++'s. Whether it *discriminates* is
  partly answered on the CPU: `engine/gate.py --selftest` injects three bugs (wrong id at a named
  step, top-two logits swapped, 5 % logits noise with the argmax left alone) and requires each to fail
  at its own step — which is the `prototype/mlp_gemv/results.md` rule that a negative everyone would
  catch is worth nothing. What that cannot say is whether the real engine lands inside the band; only
  the device run does.
- **Fixture pairing: clean.** `lm_head` self-checks live HF against `logits_last`; `glue.final_pair()`
  finds the bare-pin step by bit comparison; `attn.verify()` proves `pos = lvalid − 1` from the cache
  lanes; `proj`/`mlp`/`gdn` pins are step-prefixed and pairing fails loudly. The bare-name trap did not
  bite any current gate.
- **Bit-exactness theater: clean.** `attn_decode/reference.py` is validated against the pins before
  the kernel runs, and long-cache cases are individually re-verified against `p300` pins, so the 277-lane
  claims rest on pins and not on the helper.
- **Negatives localize** in the four suites that have them: must-catch and must-NOT-catch both asserted,
  injections load their own operands, and the two known-degenerate declarations (`attn slotk`/`slotv`)
  are stated as limitations rather than fake must-catches.
- **`lm_head_gemv` and `gdn_decode` have no negative suite at all** (vs mlp 6, proj 10, attn 12, glue 8).
  For `gdn` this is mitigated by bands (1e-6, BF16 ≤ 1 ULP) that leave no room for vacuity; for
  `lm_head` it compounds G1.
- Cosmetic: `attn_decode/run.py`'s docstring claims `q_rope`/`k_rope`/`rope_cs` are bit-exact while
  `EXACT_STAGES` deliberately bands them at the measured device-libm floor (p=64). A reader "fixing"
  the code to match the docstring would produce a false FAIL on correct kernels. Docstring line, not a
  kernel change.

## Amendment to G2 (supervisor, after `0a062b5` landed)

`0a062b5` states that §04.5's A-01 cast order "is not what the pins carry" and that
`specimpl.rmsnorm_gated` is refused by the pins on 105 683 of 294 912 elements. Measured
independently here, same pins, same 144 cases (18 GDN layers × 8 dec8 steps), `specimpl.rmsnorm_gated`
called **the way speccheck calls it** — norm weight in BF16, which is how `speccheck.Weights` stores
every model weight:

    w as the model holds it (BF16)      ndiff      0 / 294912
    w widened to F32                    ndiff  74860 / 294912

So the oracle is **bit-exact against the pins**, and the 105 683 figure is an artifact of the probe
handing it an F32-widened weight (`wf = w.float()`), a dtype combination speccheck never produces.
The kernel's POLICY variant is unaffected — it uses `bf16(w)`, the engine holds BF16 weights, and both
readings coincide there.

What survives of the finding: **the pin-level assertion on this op was missing and now exists** (the
mirror probe, six variants, dec8), which was the actual G2 gap. What does not survive: the claim that
A-01 as *implemented in the oracle* disagrees with the reference. It does not. Before anyone goes
hunting a phantom oracle bug, note the dtype the probe passed in.

Unresolved, stated rather than smoothed: my F32-w variant measures 74 860 differing where the probe's
A-01-as-written variant reports 105 683. Same formula on paper, different count, and I did not chase it
before the agents' budgets ran out. Cheap next step: dump one (o, z, w, pin) tuple from both harnesses
and diff them element-by-element — if the inputs differ, the two probes are not comparing the same
thing and one of the two ndiff figures is meaningless.

**Resolution of that unresolved note — measured on CPU, and now asserted in code.** The two figures
are one formula at two different weights, and both were correct measurements. `mirror_probe.py` now
carries both readings side by side on identical `o`, `z` and pins, so the weight dtype is the only
variable, and reproduces each figure exactly (dec8, 144 cases, 294 912 elements):

    w in BF16, product rounded in BF16 (how speccheck calls the oracle)   ndiff      0
    w BF16-rounded, then widened back to F32 (the amendment's F32-w)      ndiff  74 860
    w straight from the F32 checkpoint tensor (the probe as committed)     ndiff 105 683

`operands.py:55-65` returns `get_tensor` verbatim, so the probe's `w.float()` was the third row and
the amendment's was the second; the 30 823-element gap is that difference and nothing else. The gap
is not a symptom of two harnesses measuring different things. G2's oracle half is closed as an
executable assertion, not prose: `mirror_probe.py` passes the BF16 weight to `rmsnorm_gated`,
asserts its row at ndiff 0, asserts the bit-exact set is exactly {kernel, oracle-as-called}, and
asserts the ladder stays ordered. Run it CPU-only with
`.venv/bin/python prototype/gdn_gated_norm/run.py --mirror` (ends `MIRROR GATE: PASS`).

## Second amendment to G2 — the device half is closed too (supervisor's 2026-10-11 clean window)

G2's gap was "no bit-level assertion anywhere on §04.5". The CPU mirror closed the assertion half; the
device half stayed listed as owed (`prototype/gdn_gated_norm` was "written, gated on CPU, never run on a
device" — `bench/review-2026-10-11.md` §5). It has now run, and it passes at the floor the CPU probe
predicted: over 18 GDN layers × 8 `dec8` steps, `gated_out` worst case **ndiff 1/2048** with **max ULP
1**, zeros 0 / NaN 0, and all six located negatives localising (the four epilogue ones leaving
`norm_pre_gate` at 0/294912). "Max ULP 1 with ndiff ≈0" is the signature of a 128-term warp tree
reordering torch's reduction — exactly the error class the audit's own trap row says a max-ULP band is
blind to and `ndiff` is not, so the criterion doing the work here is the one G3 names, and it is now
running against a device output rather than only a numpy mirror.

Two things this does **not** close, so it is not misread as closing G2 in the wrong direction:
`specimpl.rmsnorm_gated` is still asserted only by this op's mirror probe and not inside
`speccheck/gates/`, and the §04.5/A-01 wording in `specification/` remains the spec owner's call. And the
verdict has a history worth knowing before trusting it: the first two attempts failed 144/144 because the
gate's own fixture writer numeric-cast `z` instead of bitcasting it — which is why the writer now
self-verifies, and why this PASS is quoted with its provenance. Write-up:
`prototype/gdn_gated_norm/results.md` §9 (the pass), §7–§8 (the failures). Timing
(`2.936 µs/launch`, spread 1.4 %, ≈0.053 ms/token) is `bench/performance-status.md` §6.

## `engine/weights_contract.py` — the engine's shape claims, audited against the shard (CPU)

`.venv/bin/python engine/weights_contract.py` (320 engine names vs 488 header tensors; ends
`WEIGHTS CONTRACT PASS`).

It exists because a claim failed a device run rather than an audit. `check_weights` asserted
`linear_attn.conv1d.weight` was `[6144,4]`; the checkpoint holds `[6144,1,4]` (one of the 18 Conv1d
weights, and every rank-3 tensor in the file), and the struct behind the claim had no rank field, so
the wrong guess was the only one writable. Finding that out consumed a GPU window — the scarcest
resource in this repo, per `AGENTS.md` — for a fact that was sitting in a file header the whole time.

Method, and why it counts as independent rather than circular: the script parses three sources that
share no code with the engine and compares them — the `Expected` table compiled into
`engine/decode_step.cpp`, the §00 constants in `engine/state.h`, and the safetensors header named by
`model.safetensors.index.json`. It does not call engine code, does not touch the device, does not
import torch. §02.5's name normalisation is inverted and the inverse is itself checked (a rename makes
the row fail as missing rather than passing vacuously), and §03.5's layer routing is applied per list —
`kGdnBlock` entries are checked on the 18 linear layers only — because asserting `self_attn.o_proj` on
layer 0 is not a stricter gate, it is a wrong one. The first draft of the script made that mistake and
its own missing-name failures caught it.

Beyond row-by-row shape/dtype equality it re-derives four structural facts, so a checkpoint change
cannot slip through a table that was merely updated to match: the only rank-3 shape present is
`[6144,1,4]`; exactly 36 tensors are F32 and they are exactly 18×`A_log` +
18×`linear_attn.norm.weight` (§01.6, no third kind); `config.json`'s `full_attention_interval=4`
reproduces the 18/6 split `state.h` hard-codes; and no `lm_head.weight` exists, which is what makes
§01.1's aliasing the only option rather than a convenience.

Discrimination is demonstrated, not assumed — the second argument points the script at a mutated copy
of `decode_step.cpp`. Three mutations were tried: the original rank-2 `conv1d` claim, `dt_bias` as
`[16,1]`, and `A_log` claimed BF16. Each FAILs at its own rows and at no others, which is the part of
the `prototype/mlp_gemv/results.md` trap that generalises ("a negative test that trips for any
injection proves nothing").

What it does **not** cover: it proves the engine's claims agree with the checkpoint, not that the
engine uses them correctly — offsets, slicing order and the flat `[6144][4]` indexing are the device
gate's problem, and `prepare_weight_adapters` re-asserts the collapse at the use site for that reason.
It also reads only the header, so it cannot see whether the loaded bytes match the declared dtype; that
is `loader_test`'s verification pass.
