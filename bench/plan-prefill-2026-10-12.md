# Q5 — prefill implementation plan (written 2026-10-12, Q5a planning worker)

Deliverable of Q5a: a package list the main agent briefs one execution worker at a time, strictly
sequentially (`bench/plan-2026-10-12.md`, Q5b rule: never two packages live). This file is a plan,
not an implementation: no kernels, no patches, no stubs in `engine/`. Every claim about the current
tree carries a file:line reading; every claim about numerics cites the spec or a ledger doc.

Binding for every package, from `AGENTS.md` and `bench/plan-2026-10-12.md` §4:

- Workers run **correctness-only device gates under ~60 s**, report command + wall + `util`, and take
  **no timing**. Every number that enters `bench/performance-status.md` is main-agent-only, blocking,
  zero-subagent, lock-tagged, best-of-N with stated spread.
- **Bands are ledger-policed.** `engine/gate.py` refuses to run when any deciding constant disagrees
  with `bench/band-history.jsonl` (`tools/band_history.py`). Moving a band needs a recorded,
  non-pass-seeking reason plus a `D-x` decision. Per **D-5**, the bands are a bound on *distance from
  the pinned route*; a fixture needing a **larger** margin than it ships is a signal, not a re-price.
- **Never write the chat-template special tokens literally** — ids only: 248046 (end-of-turn / eos),
  248044 (legacy end-of-text), 248068/248069 (think-boundary). `engine::kStopToken` is 248046.
- Never loosen a gate to make it pass. A new constant needs a floor probe. Deep fixtures get
  stop-probed first with the first-stop index recorded (`tools/stop_probe.py`, `--first-stop`).
- Commit by topic with explicit pathspecs; one package = one (or a few) revertable commits.

---

## 1. Current-state audit — where the engine assumes T = 1

### 1.1 There is no prefill pass; the prompt *is* a run of decode steps

- `engine/decode_run.cpp:376-379`: “Steps 0 .. prompt_len-1 feed the prompt … prefill is token-at-a
  time through `decode_step` with no chunked variant … `prefill.{h,cpp}` remains a throughput item”.
  The run loop calls `decode_step` once per prompt token at `:401`; the session `feed` path does the
  same at `:214` and `:219`.
- Consequence (Finding **F4**): the engine’s *numerics* at end-of-prefill are already the §06.1
  recurrent definition applied token-by-token — prefill is a **throughput** project, not a
  correctness project. What no gate has ever checked is the *state end-condition* against the pinned
  dumps (§08.6 inv 1–2), because the engine emits no state observations. That gate is package P1,
  and it can exist before any batching.

### 1.2 The single-token spine in `engine/decode_step.cpp`

| T = 1 assumption | evidence |
|---|---|
| `decode_step(…, int token)` takes **one** token id | `engine/decode_step.h:114` (declaration), `decode_step.cpp:293` |
| position is one scalar, `pos = st.prefix_len` | `decode_step.cpp:305` |
| one embedding row copied per step (D2D memcpy of `kHidden` BF16) | `decode_step.cpp:316` |
| residual ping-pong buffers are one row each (`[1024]`) | `decode_step.h` `DecodeScratch` (`s_a/s_b/norm/norm_f32/embed/delta`) |
| every scratch buffer is sized for **one token** (`g_raw [6144]`, `g_u [6144]`, `g_o [2048]`, `a_q [4096]`, `mlp_act [3584]`, …) | `decode_step.h:34-90`, allocation in `decode_step.cpp:170-215` |
| the whole §07.1 body is one token wide: one glue wave (49 launches), one mixer call per layer, one MLP call, no token loop inside | `decode_step.cpp:318-453` |
| attention is one query row: `a.pos = pos; a.lvalid = pos + 1;` | `decode_step.cpp:362-363` |
| GDN conv rolls **one** window row per step and copies it back with one D2D memcpy per GDN layer | `decode_step.cpp:397-405` |
| logits + host argmax once per step, with the one sync of the token: `hipMemcpy(host_logits…)`, strict `>` scan, then `st.prefix_len = pos + 1` | `decode_step.cpp:455-471` |
| §07.5 stop check fires on the step’s argmax (generation-phase rule lives in the driver: `decode_run.cpp` prompt/replay waiver) | `decode_step.cpp:472`, `decode_run.cpp:405-440` |

`decode_run` pays, per prompt token, the full decode cost: `lm_head_full` (2.13 ms measured),
1 MB logits D2H, host argmax over 248 320 floats, and a sync — all of it discarded except at the
last prompt position (§06.5: only `y0 = argmax logits[T−1]` is ever used).

### 1.3 Per-family T-tolerance of the graduated kernels

The rule from `prototype/README.md` and `prototype/glue/glue_kernels.h`’s header binds every kernel
change: a graduated kernel is edited **in its prototype directory, re-gated by its own `run.py`,
then re-shipped** into `engine/kernels_*.cpp`. Nothing is edited “for the engine’s convenience”.

| family | engine call sites | kernel today | verdict |
|---|---|---|---|
| `glue` (add+norm, 49/token) | `kernels_glue.cpp` → `glue_add_norm<256><<<1,256>>>` | one block, one row of 1024; body is row-local (thread-local `PER = H/THREADS` + warp tree) | **T-1 by launch shape.** A T-row version is `<<<T,256>>>` with `blockIdx.x` as the row index; per-row reduction order unchanged ⇒ bit-exact across rows. Needs `s/delta/out` to become `[T,1024]` strided buffers. |
| `proj_gemv` (9 projections + out_proj/o_proj, 114/token) | `kernels_proj.cpp:26-46`, `PACK_X=false`, `in ∈ {1024,2048}` | GEMV: `x` is one F32 vector of length `in`; rows = blocks×8×warp | **T-1 hardcoded in the kernel.** A batch is a GEMM — a new templated form, its own gate against the prefill projection pins. |
| `mlp` (gateup + down, 48/token) | `kernels_mlp.cpp:44-79` | same shape as `proj_gemv`; `x` one F32 vector (1024 in / 3584 in) | **T-1 hardcoded**; batch form = GEMM, gate vs the prefill MLP pins (§1.4). |
| `gdn` conv / gates / rec (`kernels_gdn.cpp:11-25`) | 4 launches × 18 layers = 72/token | `conv_kernel` consumes one `raw [6144]` + rolls one window row (`prototype/gdn_decode/gdn_decode_kernels.h:70-86`); `gates_kernel` one token’s 16 heads; `rec_kernel` one token’s recurrence, 16 blocks × 128 threads (`:88-114`) | **T-1 hardcoded.** conv + gates are embarrassingly token-parallel (causal conv over a `[T,6144]` tile carrying the window — §06.2 explicitly allows either zero-pad or carried window). `rec_kernel` is the sequential scan: per-token v1, or replaced by the §04.7 chunk form (P9). |
| `gdn_gated_norm` (`kernels_gdn.cpp:27-31`) | 18/token | `<<<GN_HEADS, GN_D_V>>>` over one token’s `[16][128]` | **T-1 by launch shape** (per-(token,head) blocks). |
| `attn` (`kernels_attn.cpp:60-115`) | 6/token, one fused launch per layer | one query row; `split = 1`, one-pass `full=1` softmax (§07.3’s decision); refuses `lvalid > MAXSC` at `:72-79` | **T-1 hardcoded.** The prototype *already contains* the split-K online-softmax form and its `(max,sum)` merge (`prototype/attn_decode/attn_decode_kernels.h:308-335`, host `--split` `attn_decode.hip.cpp:168,265-268`) — the engine just never selects it. |
| `lm_head` (`kernels_lmhead.cpp`) | 1/token | one row → full vocab logits | **T-1**, and correct to be: prefill needs exactly one row (§06.5). |
| `casts` (`kernels_cast.cpp`) | 115/token | elementwise over `n` | **already T-agnostic** — the only family whose signature has no token notion at all; the call sites are T-1 by buffer size. |

### 1.4 Pins that already exist for a T > 1 world (corrects two brief assumptions)

- Finding **F2**: §10.5’s “a prefill still pins only `s_in`/`s_out` per layer” is **stale**.
  `data/refs/p8`, `p64`, `p300` (486 files each) contain, on the prefill pass,
  `L{ll}.mlp.{in,gate_pre,up_pre,act,out}` (120 files in p300),
  `L{ll}.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,q_proj,k_proj,v_proj,o_proj}.out`,
  `L{ll}.gdn.*` including `state_new` (“S after this segment” — the **prefill-end GDN state is
  pinned today**, 18 files), `L{ll}.gdn.conv_state` excepted (decode-only hook), `final_hidden`,
  `logits_last`. Every batched kernel family in §06 has a prefill pin to be gated against already.
- The four deep fixtures (`dec64f`, `dec64t`, `dec300f`, `mt32f`) also pin **state fingerprints**
  with an untagged (prefill-end) `state_fp.npy` / `conv_fp.npy` / `kv_fp.npy` plus per-step copies
  (195 `_fp` files in `dec64f`); `dec8` is the one fixture without them. Fingerprint rows are
  `[n_rows,4]` F64 `[n2, maxabs, mean, dn2]` with the row→model-layer map in the manifest
  (`tools/reference_dump.py::fingerprint`, ~:261-300). This is the raw material for the executable
  G5 (P1); no re-pinning is needed for it.
- Prompt lengths that exist: dec8/dec64f 17, dec64t 27, mt32f 232, dec300f 295, p8 8, p64 46,
  p300 277. Finding **F3**: **no existing prompt ends at a multiple of 64**, so the
  exactly-on-boundary case (empty zero-β padding, `state_new` with zero padding contribution) is
  pinned nowhere. New fixture `p128` covers it (§5).

### 1.5 `MAXSC` — the brief’s location is wrong (Finding **F1**)

`MAXSC = 512` is **not** in `engine/state.h`. `state.h` carries capacity (`RequestState::cap`) and
layout; `decode_run.cpp:46` sets the session default `cap = 4096` and notes the attention ceiling
separately. The authoritative constant is

- `prototype/attn_decode/attn_decode_kernels.h:32` — `#define MAXSC 512` (shared score staging for
  the one-pass `full=1` softmax; staged as `sc[(MAXSC < TILE) ? TILE : MAXSC]`, `:183`),
- enforced in the engine by `engine/kernels_attn.cpp:72-79` (throws above it: “a longer context
  needs the split-K / flash-decoding path, not a wider stage”),
- read, never retyped, by `engine/gate.py:128-136` (`lane_ceiling()` greps the kernel header), and
  preflighted at `gate.py:1398` (smoke) and `:1599` (run).

Highest lane count ever passed: 423 (`dec300f`); the recorded wall is 513 lanes on the 17-token
prompt (`engine/results.md`, “the wall is at 513 lanes”). §08.5’s `[slots][2][256]` vs the engine’s
head-major `[head][slot][dim]` departure is still open in the spec (`state.h` header, §08.1 tail) —
the kernel remains the authority; a batched attention kernel must index the same head-major layout.

### 1.6 Launch accounting — 424 today, and what prefill amortises

`planned_launch_counts()` (`decode_step.cpp:493-503`) derives, and `gate.py` section (d) re-derives:
glue 49 + proj 114 + gdn 72 + attn 6 + mlp 48 + cast 115 + other 20 = **424** per token at the
measured 1.85 µs dispatch floor (~0.78 ms/token), on ~7.5 ms of weight traffic
(`bench/performance-status.md` §6). Finding **F6** (doc rot to fix when P2 touches these files):
`decode_step.h:104` still says “Submits ~423 launches”, and `kernels_cast.cpp`’s header says
“≈90 launches/token” against the table’s 115; `gate.py`’s number is the derived one.

Per prompt token at full chunked-batch (chunk = whole request), what happens to each family:

| family | today /token | chunked | how |
|---|---|---|---|
| glue 49 | 49 | **49 per chunk** | multi-row launch (row-local, bit-exact) |
| proj 114, mlp 48 | 162 | **162 per chunk** | GEMM forms; weight bytes stream once per chunk instead of per token — the actual TTFT win (MLP ≈ 51 % of block weight bytes, projections ≈ 44 %) |
| cast 115 | 115 | ~115 per chunk | elementwise, wider |
| gdn conv/gates/gated_norm 54 | 54 | **54 per chunk** | token-parallel |
| gdn rec 18 | 18 | **18 per token** in v1 (launch floor ⇒ ~33 µs/token of pure dispatch at 300 tokens ≈ 10 ms at T = 300) | the chunked §04.7 form (P9) is what removes this residual |
| attn 6 | 6 | **6 per token** (one query row per launch ⇒ 6·T per chunk) | the split-K form changes lanes-per-query, not queries-per-launch |
| lm_head + D2H + host argmax | 1 + sync | **once per request** | §06.5 |

So a chunked prefill drives the launch bill from `424·T` to `≈ 380 + 24·T` in v1 (the `24·T` is the
18 rec + 6 attn launches a token still needs itself), or `≈ 380 + 6·T` once P9 chunks the scan,
and — more importantly — the 1.037 GB block-weight stream from `T×` to `1×` per chunk. §06.4 is the
roofline reference: compute-bound for `T ≳ 128`; this box’s attainable bf16 FLOPS are **not** in the
ledger, so all prefill-ms predictions stay roofline arithmetic until the main agent measures (§5).

---

## 2. The §06 chunked prefill design

### 2.1 Waves, chunk boundaries, and size

- **Wave** = one batched pass of `T_w` tokens through `x0 → logits[last]`; the prompt is processed
  in waves, largest-wave-first: one wave for `T ≤ 1024`; beyond that, waves of **512 tokens**, always
  at multiples of 64 (except the last). Reasons: (a) scratch stays bounded (`[T_w,6144]` F32 ≈
  24.6 KB/token ⇒ ≤ 75 MB at 512, trivial against the pool, `AGENTS.md` scope table); (b) attention
  lane staging stays bounded; (c) 64-aligned waves keep any future §04.7 chunk kernel’s chunks
  wave-aligned, so zero-β padding only ever occurs inside a wave or at the request end, never at a
  carry point.
- **GDN chunk size = 64** wherever a chunk appears (§04.7 verbatim: HF-ref choice, A-08-verified,
  D-3 priced its width sweep against exactly this baseline). Ragged tails use zero-`β`, zero-`g`
  padding (§04.7) which contributes no state update — the property A-08 names for ragged batches.
- Wave carry is the §08.4 state tuple, unchanged: a wave boundary **is** a prefill/decode boundary
  with no sampling in between. This is what makes multi-wave equal-by-construction to single-wave,
  and it is checkable (P1’s fingerprints) rather than assumed.

### 2.2 Which GDN form runs a chunk

Two decisions, deliberately separated in time:

1. **v1 (P1–P8): recurrent §04.4, per token.** The engine’s current legal route for prompt tokens
   is recurrent (it has always fed the prompt token-by-token), the pinned dumps’ route is HF’s
   chunked scan, and A-08 measured the pair agree at cos ≥ 0.99999 per layer while both stay
   equidistant from the dump. Switching the engine’s scan form *now* would be a **route change**
   under D-3/D-5 — priced separately, see (3) — with no measured need until the per-token scan’s
   launch bill actually shows up in the TTFT curve.
2. **P9 (conditional): chunked §04.7 device kernel** (UT/WY transform: log-decay prefix sums,
   intra-chunk solve, inter-chunk scan; the oracle’s `specimpl.gdn_chunk_scan` is the CPU mirror and
   `run_chunkcheck.py` its G3 gate). Trigger: after P1–P8, the main agent’s TTFT curve attributes a
   material share to the recurrent scan’s launches (18/token residual ⇒ ~33 µs/token of dispatch,
   §1.6). Gate: engine
   chunk-vs-recurrent self-check (§10.1 G3’s second clause, cos ≥ 0.9999 per GDN layer, computed
   inside the engine), plus `L{ll}.gdn.o_pre_norm` + `state_new` vs `p64`/`p300`, **plus a
   route-affordability check before adoption**: re-run `gate.py` on all five fixtures with the
   chunked scan enabled and report every step’s deficit-vs-cap ratio; any cap that would be exceeded
   — or come within the dec300f margin’s 0.08 % slack of being — is a **stop-and-report event**
   (§6), not a re-price.

### 2.3 Attention over the prefix

§06.3: all `T` queries appended and then causal attention over the whole prefix. In the engine:
per-query-row launches remain the shape (one row = one block set, exactly as decode today); the lane
axis splits above `MAXSC` (P7/P8, §3). KV append writes slot = global position `p[t] = t` (§06.1)
— under waves, `P..P+T_w−1`. Rope positions are per token (`pos = t`), which the current kernel
takes as one float argument — the multi-query launcher must pass the position per row (P8 edit
note). KV and values are written post-`k_norm` post-rope, head-major `[head][slot][dim]`
(`state.h` header; §08.2 “never store pre-rope keys”).

### 2.4 Where state is written, and the end-condition that must match the pins

| state | written | end-condition (§08.6) | pinned as |
|---|---|---|---|
| GDN `S` (18 × [16][128][128] F32) | v1: in place per token inside the wave (`rec_kernel` already does this); P9: per chunk | `S` after token `T−1` | untagged `state_fp.npy` at prefill end; full tensor `L{ll}.gdn.state_new` in `p64`/`p300`/`prefix24dec8` |
| conv window (18 × [3][6144] F32, row 0 oldest) | v1: roll + D2D copy per token; batched conv (P6) writes the last-3 window once per wave, with `raw[t<0] = 0` exactly at **request start** (zero window from `reset()`), never at a wave interior | window = `raw[T−3..T−1]` (inv 2; A-04’s falsifier is G5 failing immediately — this is the gate that catches it) | untagged `conv_fp.npy`; `decodeN.L{ll}.gdn.conv_state` at decode steps |
| KV (6 × [2][cap][256] BF16) | append at slot = position, never overwriting `< P` (inv 3) | KV length = `prefix_len` (inv 4) | untagged `kv_fp.npy`; `decode_run` already checks `prefix_len` invariants host-side (`decode_step.cpp` `check_invariants`) |
| `prefix_len` | one `+T_w` per wave instead of `+1` per token | equals total fed count | `gate.py` sections (a)/(c) |

Nothing crosses the prefill/decode boundary except the §08.1 tuple and `prefix_len = T` (§06.5) —
the decode path needs no change at the boundary, which is the correctness of §04.6/§05.7 incrementality, already load-bearing today.

### 2.5 The first sampled token’s timing

`y0 = argmax_v logits[T−1][v]` is produced **inside the last wave** (one `lm_head_full` over the
final normed row + the existing host strict-`>` argmax, ties to lowest id, §03.4). Decode then
continues from `x = E[y0]`. So TTFT = prefill wall + one host roundtrip, and there is no extra
decode step for the first token. The engine must keep §07.5’s phase rule: the stop check is armed
in generation only — a prompt (or wave) whose last argmax is 248046 is a diagnostic, not a stop
(the `dec8` prompt legitimately contains 248046 at index 8; the §07.5 phase fix in
`decode_run.cpp:405-440` is the prior art and its selftest `--break-prompt-stop` must stay green).

### 2.6 G5 as an executable gate

`specification/10` G5: “prefill `T` then decode `D` equals one prefill of `T+D` (states, logits,
sampled ids); sampled ids exact; logits cos ≥ 0.9995.” Executable form, in two layers:

**Layer 1 — engine state vs pinned state (P1; possible today, no new kernels, no new dumps).**
A new `decode_run` flag `--state-fp` prints, after each phase boundary and optionally every step,
one row per pinned fingerprint tensor: engine-side `[n2, maxabs, mean]` for `gdn_state`,
`conv_window`, `kv_keys`, `kv_values`, computed **host-side in F64 from a D2H of the state**
(~26 MB at 63–70 GB/s D2H ⇒ ~0.4 ms, correctness path only — no timing significance). `gate.py`
gains section `(g) G5-STATE`: compare each printed row against the pinned `state_fp`/`conv_fp`/
`kv_fp` (row→layer map from the manifest; mirror of `reference_dump.py::fp_row`, shared, not
restated). Commands:

```bash
# CPU pre-checks (already-green; cheap, and P1 inherits them):
.venv/bin/python speccheck/gates/run_chunkcheck.py --fixture p300          # G3 chunk==recurrent clause
.venv/bin/python speccheck/gates/run_g5.py --prompt-ids-file data/refs/prefix24dec8/manifest.json \
        --split 24 --extend 8 --gdn-prefill chunk                          # oracle-side §08.6 inv 1
# device (correctness-only, seconds per fixture):
.venv/bin/python engine/gate.py --fixture dec64f --state-fp
.venv/bin/python engine/gate.py --fixture dec300f --state-fp               # + dec64t, mt32f
```

**P1 ships a measurement and a tripwire, not a band**: the first run prints engine-vs-pin distance
per tensor per fixture; a PASS line requires the mechanical parts (selftest: recomputing the fp of
the *pinned* tensors reproduces the pinned fp arrays bit-for-bit; injection: perturbing one engine
state buffer in the selftest must light the section up — G-neg convention). Whether the engine’s
recurrent-route state is “close enough” to the dump’s chunk-route state is then decided by the
main agent **against those printed numbers**, in the D-3/D-5 idiom (the natural denominator is the
route spread at the same position, i.e. `route_price` extended to state fingerprints). Any adopted
threshold is a band decision: it enters `BAND` only with a `band_history` line and a `D-x`.

**Layer 2 — the batched path once it exists (inherits Layer 1 + §10.1 G5 verbatim).** With batched
waves: drive `prefill(T+D)` as one wave and `prefill(T) + decode(D)` as two paths; both are already
required to match the same pinned logits/ids/fingerprints (sections (a)/(b)/(g)), which *is* G5 —
no fifth band, since each side is judged against the pin, never against each other. Commands are
the existing fixture commands with the prefill mode on:

```bash
.venv/bin/python engine/gate.py --fixture p128 --prefill --state-fp        # boundary-exact end-condition
.venv/bin/python engine/gate.py --fixture chunk64b --prefill --state-fp    # chain crosses the 64 boundary mid-stream
```

and once P9 lands, `--gdn-form chunk` flips the scan and the same five commands re-run as the
route-affordability check (§2.2(3)).

---

## 3. Removing `MAXSC` with §07.3’s block-parallel lane form

**This is the same lever Q6.3 wants, and it belongs to Q5.** Reason: Q5’s acceptance already
requires “`MAXSC` no longer caps at 512 and a >512-lane fixture passes”, a 4k-context prefill
cannot run without it, and the decode-side 4k *measurement* (Q6.3) needs it to exist at all. What
remains in Q6.3 is the timing follow-up (the lane-curve rows), not the capability.

- **The form.** Per query row, split the lane axis into `split = ceil(lanes / S)` blocks, each
  staging ≤ S lanes (S = whatever the shared staging holds — today `MAXSC` is exactly that staging
  cap); each block emits a partial `(max, sum, Σ w·v)` tuple of `DH + 2` floats to a global `part`
  buffer; a merge step combines them with the online rescaling
  `m = max(m₁,m₂); ℓ = ℓ₁e^{m₁−m} + ℓ₂e^{m₂−m}; acc = acc₁e^{m₁−m} + acc₂e^{m₂−m}`, then divides.
  **None of this needs inventing**: `prototype/attn_decode` already implements `split` partials and
  the merge kernel (“Merge the split partials of each query head with the same (max,sum) algebra,
  then §05.6”, `attn_decode_kernels.h:308-335`), gated with `full=0 --split N`, refused with
  `full=1` by design (`attn_decode.hip.cpp:265-268`). The engine currently pins `c.split = 1`,
  `c.full = 1` (`kernels_attn.cpp:105-107`).
- **Why the merged softmax stays inside G6’s band** (and inside nothing tighter): §07.3 carries the
  measurement — the blocked form cannot round the normalised weights at §05.5’s cast point, so it
  never reproduces the pins bit-for-bit; measured on the long-cache pins it reaches cos ≥ 0.99999134
  (all 53 cases: 0.99999651, meanrel ≤ 1.58e-3) on `attn_out_pre_gate`. G6’s statement
  (`specification/10.1`): sampled ids exact; decode logits cos ≥ 0.9999 (bf16-cascade floor) — an
  end-to-end threshold this op-level spread dilutes into, exactly as §07.3 pre-authorized: “when
  that path is built it must be gated on a cos/meanrel band — §10 G6’s thresholds are already of
  that shape — and not on agreement with the pins”. **A-17 binds**: `(max, sum)` is per-query-row
  reduction state; the merge is inside one query’s lanes and no accumulator ever crosses rows,
  steps, or requests. Anti-vacuity inherits G6’s: the gate must prove the online branch actually ran.
- **Blast-radius rule.** The one-pass form **stays** wherever a row fits the staging (lanes ≤ 512),
  because it is bit-exact with the pins and the five priced fixtures ride it. The split form is
  selected only by necessity (`lanes > staging cap`). This keeps every existing fixture’s route —
  and therefore every shipped margin — untouched; the new route exists only on the new >512-lane
  fixture, which prices its **own** margin per D-2 (new-fixture pricing is a derivation from that
  fixture’s steps, not a band move). D-3 already priced the online-softmax axis (adds ≤ 0.41 to the
  legal multiplier, `M_wide ≤ 2.40` total): if a >512 fixture’s own steps demand a monster margin
  after including that axis, that is §6 stop-condition territory, not a licence.
- **Launch/dispatch picture.** At 4k decode the current form costs its measured 2.014 µs/lane-token
  slope (6.10 GB/s effective — 2.6 % of probe, `performance-status.md` §6: dispatch-bound). The
  split form multiplies blocks-per-launch by `split` (cheap at 1.85 µs each) and turns the slope
  into bandwidth it can actually use; the merge adds one small pass per layer. For prefill the same
  form serves each query row’s lane axis (no new “flash tile” is planned; per-query split-K over the
  cached prefix is the §07.3 shape and this box’s L2 makes the O(T²) KV reads arithmetic, not
  traffic: 12 KiB/token × Σ lanes ≪ weights). Q6.3 re-prices the curve after it lands.
- `gate.py:lane_ceiling()` keeps its read-from-enforcer pattern: after P7 the enforced ceiling is a
  function of the split (staging cap × split over allocated `part` space); the gate must read the
  new expression’s constants from the kernel, not accept a number.

### 3.1 Debt (P8b, 2026-10-13): the 4096-lane ceiling, what lifting it costs, and why it was not lifted

P8 landed with the ceiling where P7 put it: **`MAXSC` 512 × `kAttnMaxSplit` 8 = 4096 lanes**, the two
factors still read out of the files that obey them (`gate.py::lane_geometry()`). The question P8b was
asked is whether the 8 should become 16. **Answer: not yet, and here is the arithmetic that says so.**

- **Scratch is not the reason to refuse.** `attn_part_floats(rows)` = `rows × N_Q × kAttnMaxSplit ×
  (DH + 2)` F32, i.e. **64.5 KiB per wave row at split 8** (8 × 8 × 258 × 4 = 66 048 B) and **129 KiB at
  split 16**. At `kMaxWaveRows` = 512 that is 32.25 MiB per request at 8 and 64.5 MiB at 16 — 0.19 % of
  this box’s ~34 GiB visible pool, and `attn_part_floats()` sizes the allocation from the same constant
  that sizes the refusal, so the pair cannot drift. That is the P7 answer to “is the cap hopeful?”.
- **Concurrency is where split 16 would actually be paid.** The scratch is per in-flight *request*, so a
  100-request server pays 3.1 GiB at split 8 today and 6.3 GiB at 16 (arithmetic on the AGENTS.md scope
  table’s 100 × 69 MiB figure, which does not count `part` at all). Not a constraint for one request, and
  concurrency is explicitly out of scope (plan-2026-10-12 §5), but the number is now wave_rows × split and
  no longer 64.5 KiB, and whoever does a server should know that before they size a pool.
- **Two constants have to move together, and only one of them is in `kernels_attn.cpp`.** The ceiling is
  unreachable below the context the engine hands out: `decode_run.cpp:61`’s `kSessionDefaultCap` is 4096
  and lanes are `pos + 1`, so a default session cannot even *ask* for 4097 lanes. Raising
  `kAttnMaxSplit` to 16 without raising the default cap changes nothing but the scratch bill; raising the
  cap past 4096 without raising `kAttnMaxSplit` makes `attn_decode_layer` throw. Note what that means for
  §5’s 4k row: a request that is 4096 prompt tokens **plus one generated token** needs 4097 lanes, so the
  first generated token of a full 4k-context request is already one lane past the ceiling. The pair is
  (`kAttnMaxSplit`, `kSessionDefaultCap`); `lane_geometry()` follows both without an edit.
- **Why “provable” is the blocker, not cost.** Above 4096 the engine can only be checked against itself —
  there is no pin past 524 lanes (`wide520f`) and making one is P3-class work plus its own priced route
  allowance (D-2), not a worker’s call. The cheap-looking self-check, “same rows at two `--wave-rows`,
  byte-for-byte”, is **not** generally valid above the staging: `split` is taken from a wave’s widest row
  (`split_for(lanes_last)`), so a row can be handed more lane blocks than it needs and its `(max, sum)`
  merge groups differently than the same row run alone. The precise condition is a **straddle** — the two
  agree while a wave’s lane span contains no multiple of `MAXSC` strictly inside it above the staging.
  That is why 64-aligned waves from position 0 (§2.1’s rule, and every shipped fixture) always agree:
  `MAXSC` multiples land on wave endpoints, never in the middle, and it is why the 525-lane row priced in
  `engine/results.md` is byte-identical across `--wave-rows 512 / 5 / 1`. A wave that starts off that
  alignment (a session prompt resumed at a ragged prefix) and straddles, say, 1024 inside itself hands its
  lower rows a 3-deep grouping where they would have used 2, and must be judged against its pin, never
  against another wave width — which is §2.6’s standing rule said out loud.

So: the ceiling stays at 4096, the constants stay where they are obeyed, and a lift is a *fixture* task
(>4096-lane pin + its allowance) rather than a one-line constant change. If the 4k TTFT row in §5 turns
out to want 4k contexts, that fixture is the first thing to brief, not the `16`.

---

## 4. Packages, risk-ordered

Order rationale: the cheapest discriminating gate first. P1 is pure observation on existing pins and
is the only package that can falsify the *state end-condition* assumption before any batched
kernel exists; P3 pins the chunk-boundary cases before anything is written against them; every GEMM
family then lands alone, each gated against its own prefill pins **and** the five-fixture margin
preflight (§6.0), so a margin-saturation surprise kills exactly one revertable commit.

Device minutes are per-package correctness passes a worker may run (each individual run < 60 s);
the “main agent” column marks the timing passes this plan hands over (§5).

| # | goal (one sentence) | files created/edited | spec refs | gate command (inherits) | CPU/dev | ~dev min | depends | risk retired |
|---|---|---|---|---|---|---|---|---|
| **P1** | Make the engine’s §08 state observable and gate it against the pinned prefill-end and per-step fingerprints — the executable half of G5. | `engine/decode_run.cpp` (`--state-fp`, D2H + F64 fp print), `engine/gate.py` (section `(g)`), shared `fp_row` helper (`tools/reference_dump.py` extract), results note | §08.6 inv 1–4, §10 G5, A-04 | §2.6 layer-1 block; plus `gate.py --selftest` new injections (perturbed state must light `(g)`; fp-of-pins identity check) | CPU build; device runs are the 5 fixtures (~1 min total) | 5 | — | the prefill/decode **state end-condition** vs the pins, incl. conv-window roll (A-04) and KV append, at 17/27/232/295-lane boundaries |
| **P2** | Add the prefill wave entry (`engine/prefill.{h,cpp}`) that runs existing kernels per token under one submission stream with one sync, one `lm_head` + argmax at the end, and `--prefill-logits` set to `all` or `last`. | new `engine/prefill.h/.cpp`, `engine/decode_run.cpp` (run-mode prompt phase + session `feed` route through it; `STEP kind=prefill` line), `engine/build.sh` (one TU), `engine/gate.py` (parse prefill lines; keep `--break-prompt-stop` selftest semantics), fix F6 doc drift | §06.1, §06.5, §07.5 phase rule | `engine/gate.py` on all 5 fixtures **unchanged PASS**, plus byte-identity: `--prefill-logits all` logits dumps == the current per-token path’s dumps on `dec8` (same launches ⇒ same numbers) | device ~10 min | 10 | — | prefill↔decode plumbing: positions, `prefix_len` batching, §07.5 phase split, session pending-sample invariants (`chat.sh` `feed`/`gen`), the first TTFT floor |
| **P3** | Pin the unpinned chunk-boundary and >512-lane cases: extend the dumper’s lean policy with `--logit-steps`, then make `p128`, `chunk64b`, `wide520f`. | `tools/reference_dump.py` (`--logit-steps`, reusing the `keep_steps` mechanism), `tools/dump_fixtures.sh` (three new rows), new `data/refs/{p128,chunk64b,wide520f}` (gitignored; recipes committed) | §04.7 (chunk 64, zero-β), §10.0/§10.5 (fixture identity, verify), plan §4 deep-fixture prompt rule | `reference_dump.py … --verify data/refs/<tag>` byte-exact for each; `tools/check_dump_reproduction.py`; `gate.py --no-engine` pairing on each | **CPU-only** (torch dumper) | 0 | — | boundary cases unpinned today (F3: no prompt ends at 64·k; nothing crosses 512) — later packages get gates *before* their kernels exist. **LANDED 2026-10-13**: all three pinned, byte-exact, unpriced, `--no-engine` pairable; `wide520f` is inert for device runs until P7 by design (§5.1 “P3 landed”, `bench/p3-boundary-fixtures-2026-10-13.md`). **No re-pin happened in P6b** — D-7’s “missing pin” premise was refuted (`Recorder.pin` never lets `--logit-steps` drop the bare prefill row), so these three directories are byte-identical to P3’s commit and nothing here was re-dumped, re-priced or re-probed. One fact about them is worth carrying into P7/P8: `chunk64b`’s pinned prefill row is an exact 2-way BF16 tie (760/1596 at 19.875), which makes it the only fixture that exercises a tied `y0` at all, and its step 0 is D-9’s recorded red (`bench/plan-2026-10-12.md`, expected-red list) |
| **P4** | Batch the biggest weight family: multi-row glue + token-batched MLP GEMMs inside the wave (51 % of block-weight bytes). | `prototype/glue/*` (row-indexed `glue_add_norm`; re-gate), `prototype/mlp_gemv/*` (batched gateup/down forms; re-gate incl. all six negatives at rows=1 unchanged), `engine/kernels_glue.cpp`, `engine/kernels_mlp.cpp`, `engine/prefill.cpp`, `engine/DecodeScratch` ([T_w]-sized buffers), **P4b adds** `engine/launch_stats.{h,cpp}` + `engine/gate.py` section (h) | §03.2, §03.3, §06.4 (MLP dominance), §10.1 F1–F4 (method), §10 G3i | per-op: new `mlp_gemv/run.py --tokens` stage (the plan said `--rows-batched`; `--tokens` is what shipped) vs **prefill** pins `L{ll}.mlp.*` on `p64`/`p300` (F1–F4 shapes, aggregate rules unchanged); rows=1 runs of the existing gate must stay green bit-for-bit; then all 5 fixtures on the engine (margin preflight §6.0). **P4b**: `gate.py` (h) counts launches and A/Bs `--wave-rows 512` vs `1` for byte-identical logits, with two `--selftest` injections | device ~15 min | 15 | P2 | reduction-order risk of the largest GEMM family *before* two more GEMM families stack on it. **LANDED 2026-10-13 (P4 `46d95c0`+`eb089a4`+`48d469a`, P4b proof + per-op stage)**: five fixtures 119/601/598/1107/342 green with margins unchanged at x1.000, the batched kernels proven by counter (dec300f prompt wave 97 launches vs 28 615 one-row, logits `cmp`-identical), glue per-op green on `p64`/`p300`, MLP per-op green on `p64` and red on **one** `p300` element that the single-token kernel reproduces exactly (band question for the main agent, nothing touched). See `engine/results.md` “Q5 / P4 + P4b” |
| **P5** | Batch the nine projections + out_proj/o_proj (44 % of the remaining bytes) with GEMM forms of `proj_gemv`. | `prototype/proj_gemv/*` (batched form + gate stage), `engine/kernels_proj.cpp`, `engine/prefill.cpp` | §04.1, §05.1, §06.2 (“one GEMM per projection” is the spec’s own sentence), §10 G3i, §10.5 W1 pins (they exist for prefill) | per-op gate vs prefill `L{ll}.{in_proj_*,q_proj,k_proj,v_proj,o_proj,out_proj}.out` on `p64`/`p300` at G3i (cos ≥ 0.9999, meanrel ≤ 1 %); engine: 5 fixtures + margin preflight | device ~15 min | 15 | P2 (P4 independent of P5 — both need P2; brief whichever order the margin preflights prefer) | GEMM reduction order on the skinny shapes (the narrow-shape ledger rows say these are shape-limited; batching is their only bandwidth lever) |
| **P6** | Batch GDN conv + gates + gated-norm over a wave, carrying the conv window with the §06.2 request-start zero-padding exactly. | `prototype/gdn_decode/*` (windowed causal conv over `[T,6144]`, multi-token gates), `prototype/gdn_gated_norm/*` (T×), `engine/kernels_gdn.cpp`, `engine/prefill.cpp` (window D2D copies collapse to one per wave) | §04.2, §04.3, §04.5, §06.2, §08.6 inv 2 (window = `raw[T−3..T−1]`), §10 G3/G3i | per-op gate vs prefill `L{ll}.gdn.{mixed_qkv_pre_conv,q,k,v}_post_conv,g_decay_log,beta,o_pre_norm` on `p64`/`p300`; `(g)` state-fp for `conv_fp` on all fixtures; `p128` is the end-condition proof | device ~15 min | 15 | P2, P5 (projections produce the `[T,6144]` `raw`) | the conv-window end-condition at T > 1 (A-04’s falsifier) and the ragged-wave carry |
| **P7** | Remove the 512-lane cap: engine selection of the prototype’s existing split-K online-softmax form when lanes exceed the staging, one-pass untouched below. | `prototype/attn_decode/*` (re-gate `--softmax online --split k` at the G6-shape band + anti-vacuity), `engine/kernels_attn.cpp` (choose `split = ceil(lanes/staging)`, `full=0` only when forced; allocate `part` in scratch/state), `engine/gate.py` (`lane_ceiling()` → the split expression; relax the two preflight lines), results/ledger notes | §07.3 (decision paragraph + pre-authorized band), §10 G6, A-17, §08.2 (KV head-major index contract) | per-op: `attn_decode/run.py --softmax online --split {2,4}` vs band (ids exact; logits cos ≥ 0.9999 per G6; “online branch ran” anti-vacuity); engine: **`wide520f` PASS** (its own route allowance priced by `route_allowance.py` — a derivation, ledger-logged) **and the five original fixtures bit-identical** (one-pass route unchanged) | device ~20 min | 20 | P2 (P3 for the fixture) | the 4k blocker (ledger §6: measurement “blocked by `MAXSC = 512`”); the (max,sum) merge risk, contained to the new route only |
| **P8** | Run attention layers of a prefill wave through the same block-parallel path (per-query split-K; positions per row), retiring the last per-token attention launch wave at long prompts. | `prototype/attn_decode/*` (per-row position; multi-query host loop or grid.y over queries), `engine/prefill.cpp`, `engine/kernels_attn.cpp` | §05.5–§05.7, §06.3, §07.3, §10 G4 (de-interleave negative must still fail) | per-op vs prefill `L{ll}.attn.{q_rope,k_rope,attn_out_pre_gate}` on `p64`/`p300` (one-pass lanes) + online band on `wide520f`; engine: all fixtures + `wide520f`; `p128` y0 check (§06.5 first sample) | device ~20 min | 20 | P5, P7 | O(T²) prefill attention at 1024/3072 contexts; KV append at wave positions. **LANDED 2026-10-13 (P8 op `a332fd9`; engine half P8b)**: one attention site per attention layer per body (`blockIdx.z` = query row, `attn_prep` + `attn_core` staged plan), every shipped fixture bit-identical (`margin_delta.py` x1.0000 everywhere), `p128`’s prompt wave now 6 attention launches instead of 768, and `(h)` counts `attn` as a fifth batchable family with an aimed injection. The first engine wiring hung the device on `p128`; the cause and the trap are in `engine/results.md` (“Q5 / P8”) and §3.1 above is the ceiling debt ** |
| **P9** *(conditional)* | Replace the per-token recurrent scan with a §04.7 chunk-64 device kernel **iff** the scan’s launch residual shows in the measured curve, with a route-affordability check first. | new `prototype/gdn_chunk/` (kernel + gate: engine-vs-engine chunk==recurrent per layer, G3 clause), `engine/kernels_gdn.cpp`, `engine/prefill.cpp` (`--gdn-form chunk`) | §04.7, §06.2, A-08, §10 G3; **D-3/D-5** route rule | `run_chunkcheck.py` mirror inside engine (cos ≥ 0.9999/layer); `L{ll}.gdn.o_pre_norm`/`state_new` vs `p64`/`p300`; then §2.2(3) affordability check on all 5 fixtures | CPU kernel dev + device ~20 min | 20 | P1, P6, **main-agent TTFT curve first** | the O(T) scan-launch residual — and, if it lands, the chunk-route distance question, *priced before adopted* |
| **P10** *(not a worker package)* | The measurement + docs pass: TTFT curve, 1k/4k decode rows, `chat.sh` before/after, ledger + results write-ups. | `bench/performance-status.md` §6, `engine/results.md`, this file’s status column | plan Q5 acceptance | §5 measurement block | main agent only, blocking | — | P1–P8 | closes Q5’s acceptance: G5 at several boundaries, >512 fixture green, TTFT visibly down, rows with lock tag + spread |

Dependencies as a chain a single worker can be briefed against:
`P1 → P2 → P3 (anytime, CPU) → P4 → P5 → P6 → P7 → P8 → [curve] → P9?`
P4/P5/P6 are mutually independent given P2; each is one brief, one gate, one revert.

**CPU-only packages: P1 (build+selftests; needs a device pass to produce its first numbers), P3
(dumper/fixtures), and P9’s CPU oracle-mirror self-check before the kernel lands.** Every other
package needs device passes; all device passes listed are correctness-only and each run is
seconds, so a worker may run its own gate.

---

## 5. Fixtures and measurements

### 5.1 New pins (budget: ~60 MB/fixture, lean policy, stop-probe rule)

| tag | content | pin policy | est. size | first-stop rule |
|---|---|---|---|---|
| `p128` | prompt of **exactly 128 tokens** (prose), + 8 replayed steps; pins the chunk-multiple end-condition and `y0` | `--lean --deep-steps 7 --logits-bits bf16 --first-stop "<probe>"` (probe via `tools/stop_probe.py --max-new 130`; short chain — record whatever it says) | ~20 MB — **P3 actual: 32.1 MiB, 546 pins** | record probe result |
| `chunk64b` | prompt ~60 tokens + 12 forced steps ⇒ the chain crosses the 64-chunk boundary mid-stream (the ragged-carry case; stays inside budget — extend steps only if a 128-crossing is later wanted) | `--lean --deep-steps 11 --logits-bits bf16` | ~12 MB — **P3 actual: 43.1 MiB, 656 pins. Shipped as 120 prompt tokens + 30 steps (`--deep-steps 29`), not ~60 + 12: a 128-crossing was wanted, and this way the prompt end is ragged *and* the chain crosses 128. See “P3 landed” below.** | short chain; probe anyway |
| `wide520f` | prompt ~250 tokens + ~275 forced steps ⇒ ~525 lanes, crossing 512 inside the forced chain | `--lean --logit-steps "<~24 steps around the crossing>" --deep-steps none --logits-bits bf16` — the deep prompt rule binds (probe ≥ 128, prefer ≥ 300 with no stop; known-good families: the count-chain, the months-and-lengths list); first-stop index **required** in the manifest | ~30 MB with `--logit-steps`; **requires P3’s dumper extension first** (without it, lean logits alone would be ~130 MB) — **P3 actual: 53.4 MiB, 626 pins, 500 prompt tokens + 24 steps = 524 lanes, `--logit-steps all` + `--deep-steps 23`; the crossing moved prompt-side and the reason is in “P3 landed” below** | hard requirement |

No existing fixture is re-pinned: their bytes are load-bearing for `--verify` (§10.5). The
fingerprint pins P1 needs already exist in the four deep fixtures; `dec8` simply has no `(g)` rows.

### P3 landed (2026-10-13): what the three fixtures are, and what this sketch got wrong

Full write-up: **`bench/p3-boundary-fixtures-2026-10-13.md`** — what each pin can and cannot falsify, the
negative tests, the re-check commands. Recipes are committed in `tools/dump_fixtures.sh`, prompts in
`speccheck/fixtures/`; the pins are gitignored as usual. All three reproduce byte-for-byte (`--verify` and
`tools/check_dump_reproduction.py`), all three pair **every** pinned step with `0 declared gaps`, and **no
device run was taken** — P3 is CPU-only, as planned.

1. **The size arithmetic here was wrong, and it is what reshaped `wide520f`.** Measured off `dec64f`: one
   BF16 logits vector is **0.4738 MiB**, one full `--deep-steps` dump is **25.96 MiB**, so a ~60 MB budget
   buys **~71 pinned steps with a deep step, ~126 without one**. The sketch’s “~250 + ~275 forced steps”
   needs 275 *pinned* steps for `gate.py` to judge them, which is **130 MiB of logits alone** — 2.2× over
   budget; that is why its `--logit-steps` escape was load-bearing. Approved substitution: **500 prompt
   tokens + 24 forced steps = 524 lanes**, 53.4 MiB, crossing at the same lane with a quarter of the chain.
   Side effect, and a useful one: the over-cap lanes are now **prefill-side**, where P8’s wave arithmetic
   lives. In the gate’s 0-based step names, `decode11` reads exactly 512 lanes (the last the one-pass stage
   serves), `decode12` is the first it cannot, `decode12`…`decode23` live above the cap. With option C
   below the sketch’s chain-side shape is now *expressible* (~275 steps with ~24 declared logits ≈ 40 MiB),
   so a later package can still have a crossing inside the forced chain; P3 did not build it, since the
   approved shape covers the same lane counts for less.
2. **`chunk64b` covers both readings of its ambiguous row** instead of choosing: prompt 120 = 64+56 (partial
   final chunk, with a full chunk carried across the last boundary — which `p64`/`p300` never do, their
   prompts sit inside one partial chunk) **and** a chain crossing 64·2 = 128 at `decode8`, ending at
   150 = 128+22. The row said “extend steps only if a 128-crossing is later wanted”; it was wanted.
3. **A gate change was needed, which the row’s file list did not list.** `--logit-steps` reuses the
   `keep_steps` mechanism with `--deep-steps`’ step-tag vocabulary (`all` / explicit list, no ranges), is
   `--lean`-only, and refuses a deep step whose logits were dropped — `--deep-steps N` without `N` in
   `--logit-steps` would pin every intermediate of a step nothing can judge. But `gate.py` section (0)
   `load_pin`s every step and hard-exits on a missing tensor, so a sparse pin was not pairable at all. Under
   the supervisor’s option C: section (0) may skip a step’s logits **only** if `manifest.json` names that
   exact step in `logits_declared_absent` — per step, never a count, range or ratio; an undeclared gap still
   fails closed; the gate prints `paired N of M steps, K declared gaps`; device and `--dry-run` paths refuse
   a gapped fixture rather than silently dropping its `decodeN.input_ids` checks. Six injections in
   `engine/gate.py --selftest` cover it. No band, margin or `route_allowance.py` behaviour moved, and the
   five shipped fixtures still report `N of N steps, 0 declared gaps`.
4. **All three ship unpriced** (no `route_allowance.json`), riding the flat §10.1 floors — D-2, and pricing
   is a main-agent act. Consequence for P5/P7: a first device failure at these tags may be a flat floor too
   tight for a wide route, not a wrong kernel. D-14 before any band moves.
5. **`wide520f` is inert for device gating until P7**, by design: `gate.py` refuses it at the lane-ceiling
   preflight (`500 prompt + 24 generation = 524 lanes vs MAXSC 512`), which is why `--no-engine` is the only
   pairing available for it today. **P5 meets this, not P7** — P5 runs the gate across the attention fixtures
   (§5.4) and will hit the ceiling on its first device pass. Report it; do not shrink the fixture to make P5
   green.
6. **These pins do not settle F3’s open half.** §5.2’s padding probe (full 296-token prefill vs 64+104 chunks,
   `gdn.state_new` at token 295) is still unrun and `data/refs/p64/prefill_state_mismatch.json` still does not
   exist, so these fixtures can say “the engine must agree with HF at these lane counts”, not “padding moves
   the state by X”. And a kernel that pads *everything* to 64 still matches HF at `p128` while doing exactly
   the wasted work F3 is about: correctness is pinned, efficiency is not — the efficiency half needs a timing,
   and a fixture cannot make one.
7. **`p128`’s `y0` expectation (P6/P8 rows) is an engine-side comparison, not a pin property.** The fixture
   pins the first generated row’s logits and the §08 state at every step; whether a wave’s first sample equals
   the single-token decode’s row (§06.5) is something those gates have to compute *at* the fixture.

### 5.2 Measurements — **main agent only, lock, best-of-N ≥ 3 with spread stated, util recorded and discounted, blocking, zero subagents** (AGENTS.md)

1. **TTFT curve**: prefill wall time at contexts **64 / 256 / 1024 / 3072** — `decode_run --ids
   <p64|dec300f|1024-tok|3072-tok prompts> --steps 0` (or `--prefill` mode from P2), wall between
   “state allocated” and the `y0` line; ≥ 5 reps, report min + spread. This number does not exist
   today at all (plan Q5: “we have no TTFT figure”).
2. **decode ms/token vs context**: `dec64t`-style replay at **1k** and **4k** lanes (prompt padded +
   forced steps), best-of-3 + spread — the rows the ledger has been extrapolating (the 2.014 µs/lane
   slope line in `performance-status.md` §6 explicitly names MAXSC as the blocker; P7 unblocks it).
3. **Attention curve** (P7 pricing, before/after): `prototype/attn_decode` timed at 64/256/512/1024
   lanes with `--split {1,2,4}`, one clean window (ledger already owes this curve).
4. **`chat.sh` on success**: `engine/chat.py`’s printed TTFT for a first turn drops from ≈ prompt_len
   × 12 ms to the prefill wall + one step — order-of-magnitude at 1k+ prompts; per-turn `FED n=`
   becomes one line; `gen` still starts with the sample already pending (session invariant,
   `gen[i]` pairs with `greedy[i]`); tok/s line itself barely moves (decode path is unchanged by
   Q5 except beyond 512 lanes, where it should *improve*). These are behaving-sanity figures, never
   ledger rows.
5. Ledger additions land with a lock tag and spread, per `performance-status.md` §6 protocol; no
   worker number (including P-packages’ own gate runs) enters them.

---

## 6. Stop conditions for execution workers — when to stop and report, not proceed

**“Make the gate pass” is never an acceptable outcome.** Every event below is a stop-and-report to
the main agent, with the numbers attached, in the D-3 idiom (“report and stop”); none of them is
license to touch a constant.

0. **Margin preflight (applies to P4–P9, because of a saturation nobody should rediscover
   silently):** the shipped caps leave *almost no slack at the binding step* — `dec300f` step 86
   sits at ratio 7.984 against a 7.99 margin (`engine/gate.py` band comment; D-2/D-5). Any
   legally-different reduction order (a new GEMM, a batched norm) can move that ratio by more than
   the 0.08 % headroom. So after each batched-kernel package, the worker runs all five fixtures and
   reports the **worst step ratio per fixture next to its cap**, green or not. A green run whose
   worst ratio rose materially, or any step going red, is a **stop event** — D-5 says this is a
   signal about the engine, and what to do about signals is the main agent’s decision (D-3 already
   ruled out re-fitting; the only alternative shapes are the two it priced and declined).
1. **Any band or margin that would need moving** — a gate that fails for a reason whose honest fix
   is a number in `BAND`, a fixture `margin`, or `lane_ceiling` semantics. The bands are policed
   mechanically (`tools/band_history.py --check` runs before `gate.py` reads anything); an
   unrecorded move fails the gate, and a recorded-but-thin reason is exactly what the ledger
   refuses. Do not write one. Stop, report the measurement.
2. **A legal-margin re-price that would have to grow an existing fixture** — new fixtures deriving
   their own margins (`route_allowance.py`) are normal and ledger-logged; existing fixtures’
   margins growing is the D-5 signal from (0).
3. **A G5 mismatch that points at the reference**: e.g. `--state-fp`’s own consistency fails (the
   pinned tensors’ recomputed fingerprint does not reproduce the pinned `*_fp` arrays — that is a
   dumper/fixture question, use `tools/check_dump_reproduction.py` and stop), or two independent
   *engine* routes reproduce the pin’s sampled ids while disagreeing with a pinned intermediate the
   pin itself disagrees with by more than §10.2’s cascade figures. Suspect the fixture before the
   kernel is the standing rule (`§10.5`, “a fresh HF run must reproduce them”); documenting is fine,
   adapting engine arithmetic to make a suspicious pin match is not.
4. **A package that turns out to need a spec change** — e.g. §08.5’s open KV-layout departure
   (`[slots][2][256]` vs the engine’s head-major) biting a batched kernel, or §06 vs the F32 conv
   window resolution, or the zero-padding semantics at a wave interior not following from §06.2.
   Spec edits are main-agent work with a speccheck re-run; a worker hits this, stops, and cites
   the clause.
5. **A device gate that cannot finish in ~60 s**, or any temptation to take a timing in a worker
   pass: stop, hand the command to the main agent with the exact line to run (this plan’s §5 block
   is already drafted for that).
6. **Anything touching the special-token hygiene or §07.5 phase semantics** beyond what a package
   states here (ids only; prompt-phase 248046 stays a diagnostic) — the prior damage is documented
   in `AGENTS.md`; if a fixture shows a stop id mid-prompt where this plan did not predict one,
   report rather than special-case.

---

## 7. The single riskiest assumption of this design

**That Q5 can ship without touching a band.** The shipped per-fixture margins are fitted to the
engine’s *current* route with the worst step sitting on the cap by construction (D-2 note;
`dec300f` step 86 has 0.075 % headroom). Every batched kernel is a legal rounding-route change, and
D-3 already measured that legal routes are *not* small — the reference’s own two prompt forms
separate by up to 1.7e-3 1−cos at steps that break the flat floor, and the engine-independent
multiplier (1.19–1.96) is already smaller than what the engine needs. If a GEMM reduction order
pushes a binding step over its cap, the whole “cheapest legal change” spine of P4–P6 stalls at a
D-5 event — and the correct response is still stop-and-report, not re-fit. That is why §6.0 makes
the margin preflight a per-package ritual, and why P1 (no arithmetic change at all) goes first:
it banks the state end-condition truth before any route is allowed to move.

Runner-up risk, stated so it is not discovered at P7: the claim that the prototype’s
`--softmax online --split` path needs only *re-gating*, not rewriting, at engine buffer geometries
(`part` sizing, head-major `cap` strides, per-row positions in P8). The prototype ran it against
pins in its own host (`attn_decode.hip.cpp:168,265-268,319`; kernels `:308-335`), so the arithmetic
exists — the engineering is the engine launcher and the G6-shape band gate.
