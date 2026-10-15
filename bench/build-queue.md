# Build queue — what is left between three kernels and an engine

This is the work list, not a status report. `bench/performance-status.md` is the ledger of what
is measured; `prototype/README.md` is the index of graduated ops; this file says what to build
next, in order, and what each item must show before it is allowed to be called done.

Written 2026-08 after `mlp_gemv` graduated. Ordered by dependency: W1 unblocks W2/W3, and W5 is
where everything gets wired.

## Contract every item in this queue inherits

Read before starting, in this order: `AGENTS.md` (scope + traps), `prototype/README.md`
(the nine measurement rules and the per-op directory contract), the specification file named in
the item, and `prototype/mlp_gemv/` as the most recent complete example of a kernel + gate +
`results.md`.

- **Never loosen a gate to make it pass.** If a criterion is unreachable, measure the floor
  (`prototype/mlp_gemv/floor_probe.py` is the method) and write down why. A gate that cannot
  fail verifies nothing (`--negative` on every injection, located).
- **No GPU execution inside a worker run — at all.** The engine serving every agent on this box
  shares the GPU, so a device run made by a worker is contaminated by construction. Workers write
  the kernel, the gate, and the CPU-side reference work (pins, torch references, band sizing from
  dumps), **compile** if they like, and hand off the exact command lines. The main agent runs every
  device command, and only with zero subagents active. A report saying "gate written, not yet run" is
  a complete deliverable; a report with a device number taken alongside other agents is not.
- **Timings need the box to themselves:** `tools/gpu_lock.sh acquire <tag>` around every timing
  loop, `release` after. Correctness runs do not need the lock; timing runs do — a second
  process moves a cold bandwidth number by 25 % and flips a config ranking.
- `hipMalloc` + `hipMemcpy` for weights, warm the pages before timing, `-O3` via `build.sh`,
  `-Wl,-rpath,/opt/rocm/lib`, `hip_bfloat16` not `__nv_bfloat16`, 64-bit shuffle mask.
- Compare against `bf16`s shared comparator `prototype/bf16_ulp.py` (run its selftest), not a
  hand-rolled ULP function.
- Before committing: `speccheck/run_all.py` must still be **26/26 PASS**, and a fresh
  `tools/check_dump_reproduction.py` run must be byte-identical to the pinned dumps if you
  touched the dumper.
- One commit per topic, subject line says which (`AGENTS.md` standing rules). `results.md` is
  part of the op's commit, not a follow-up. Ignore the built binary by exact path in
  `.gitignore` — a `prototype/**/<op>` pattern swallows the whole op directory.
- Report, do not assume: numbers in `results.md` come from a run whose command is in the file.

## W1 — pin the projection operands (dumper + fixtures)

Blocked by nothing. Unblocks W2 and W3, which cannot be gated without these.

`data/refs/dec8/` pins `gdn.*` and `attn.*` and `mlp.*`, but **not the projections themselves**:
there is no `in_proj_qkvz` / `in_proj_ba` / `out_proj` / `qkv_proj` / `o_proj` output pin. What
exists is one step removed (`gdn.mixed_qkv_pre_conv` is post-reshape, `gdn.o_pre_norm` is
post-norm-gate), so a projection kernel would be gated against a derivative of its own output
rather than its output. Add the module hooks the same way `mlp.*` were added:
`decode{N}.L{ii}.{in_proj_qkvz,in_proj_ba,out_proj,qkv_proj,o_proj}.out`, plus the shared input
(`s_in`/`s_out` already cover the norm side).

Must show: every pre-existing pin byte-identical after re-dump (the hooks must not perturb the
forward); `check_dump_reproduction.py` clean on all six fixtures; `speccheck/run_all.py` 26/26;
`specification/10-reference-dumps-and-gates.md` §10.5 gains the new pins and any limit on them
(same decode-steps-only caveat as `mlp.*`, if it applies).

## W2 — `prototype/proj_gemv`: the projection GEMV family

The largest remaining byte slice: GDN projections 0.3796 GB/token + attention projections
0.0881 GB/token = **29 % of the token**, and it is one kernel shape instantiated five ways
(`in_proj_qkvz` 6144×1024, `in_proj_ba` 2048×1024, `out_proj` 1024×2048, `qkv_proj` 4096×1024,
`o_proj` 1024×2048 — take the authoritative numbers from `tools/weights_report.py`, not from
this line). `mlp_gemv`'s templated `mlp_down<IN>` is the starting point; do not fork a copy per
shape.

Gate: F1-style per output family (reference operands in, compare against the new pins), the
floor probe re-derived for these K values (1024 and 2048, not 3584), negative tests located,
cold timing under the lock. Expect the **block-count question from the frozen ledger (§5 item 5, progress/2026-10-16/performance-status-2026-10-16.md:240) to
land here**: `out_proj`/`o_proj` are 1024-row outputs, the same under-fill as `down_proj`. If
it does not close, write that down as a measured property of this box rather than tuning around
it.

## W3 — `prototype/attn_decode`: §05 decode step

Only 6 of 24 layers (`l mod 4 == 3` → 3, 7, 11, 15, 19, 23 — the pinned directory names are the
check to trust), and 5.5 % of the token's bytes, but the only remaining piece with real
structure: partial RoPE (§05.3), the zero-centered per-head RMSNorms of §05.2, the q/gate
de-interleave of §05.1, GQA attention over a KV cache that must be **appended to** (§05.7) and
read at length, and the output gate (§05.6). Pins exist: `attn.{q_rope,k_rope,v,attn_out_pre_gate,
cache_keys,cache_values}` plus the W1 projection pins. The G3i isolation harness
(`speccheck/gates/run_isolated.py`) already feeds a single attention layer and is the fastest
route to a bit-exact claim on the projections-then-core path.

Gate must discriminate the known trap: a `q`/`gate` de-interleave error (§05.1) has to fail
loudly here (§10.1 G4 note). KV cache write and read must be gated at more than one cache
length — a decode kernel that is right at `seq_len=25` and wrong at 200 is the classic failure
for this shape.

## W4 — `prototype/glue`: norms, residuals, gating

Tiny bytes (~0.005 GB/token) and therefore a launch-cost item, not a bandwidth item: 2 RMSNorms
+ 2 residual adds + gating per layer ≈ 100+ launches/token at the 1.85 µs floor ≈ 0.2 ms/token,
which is 3 % of the budget for kernels that move nothing. Design goal is **fusion into the op
that produces the input** (the `mlp_gateup` pattern), so the count drops rather than the
per-kernel time. Gate against the existing `s_in`/`s_out` pins and the residual chain.

## W5 — `engine/`: residency, loader, and the token loop

The first thing in this repo that is an engine rather than a kernel. Minimum: safetensors →
`hipMalloc`/`hipMemcpy` with the name-normalized loader (§02.5), one warm-up pass over resident
pages, per-request state buffers (GDN 18.0 MiB + conv 0.63 MiB + KV), and a `decode_step()` that
calls the graduated kernels in §07.1 order with no host round-trip per op. `engine/README.md`
says what it is; `engine/` stays free of policy the spec has not fixed.

Gate: the same fixture speccheck runs — `prefix24dec8` in, 8 decode steps out — compared against
`decode{N}.logits_last.npy` and the §10.4 token-equality class, with the drift explained by
A-18's chaos floor rather than by a new tolerance.

## W6 — end-to-end correctness, then the first real tok/s row

`testdata/` (the `bench.sh` fixtures) through `engine/`, greedy, against the HF reference:
token-equality against A-18's floor, then `bench/performance-status.md` §1 gets a **measured**
tok/s row replacing the projected 137–147. If the projection gap from W2 is real, this is where
it shows up as a number, and the ledger gets the correction rather than a footnote.

## W7 — prefill (§06), if and when a prompt has to be served

Decode-only is enough for W6's tok/s number. Serving a real prompt needs chunked GDN prefill
(S04.7 in speccheck) and attention prefill (§06.3), and §06.5 says exactly what prefill must
leave behind for decode. Deliberately last: it is a second implementation of the recurrence, and
decode parity is the thing that gates the engine's existence.

## W8 — MTP on the GPU (§09), optional and gated by its own economics

speccheck's MTP is bit-lossless but *loses* on CPU (~21 vs ~24 tok/s, accept 0.66 at k=1). On
GPU the draft's 0.0409 GB is cheap against 1.605 GB, so the trade may flip. Gate = G8 accept
rate plus a measured tok/s comparison with drafting off. If it does not win on GPU either, say
so in the ledger and stop.

### D-11 — **DONE (2026-10-14): option A shipped, and route (c) is the default** — accepted by the user, executed 2026-10-13/14

**What was actually done, item by item against the acceptance above.**

1. **Frozen `kMaxWaveRows` + hipBLAS version as pin provenance** — `engine/numerics_provenance.json`
   (`wave_rows_cap: 512`, `hipblas_version: 3.6.0`), printed by every run on its `NUMERICS` line and
   asserted by `engine/gate.py:numerics_provenance_checks`; the version is asked of the loaded `.so` with
   `hipblasGetVersion`, not read from a header macro, so a library swap is a numerics change and not a
   rebuild detail (`d710a78`). The record still says it was captured on the hand route, and that is right:
   the pins were produced there.
2. **The cross-wave-count bit-identity gate is replaced by a band check on the library route** —
   `CASCADE_BAND` in `engine/gate.py` (row 1-cos 2e-3, stage meanrel 3e-2, stage cos 0.9998, flip room
   0.10) with the cross-width comparison walking every dumped row (`cross_width_ids`) and the wave band
   checks pooling meanrel/cos per stage. `--gemm hand` keeps bit-identity; nothing about the hand route's
   guarantee was relaxed (`0b30422`, exercised for real in `0fea0db`).
3. **The tie rule was generalised** — an exact tie in a pin is admitted by D-9's sized slack bound instead
   of refusing admission, which is what `chunk64b`'s tie needed (`0b30422`).
4. **The cascade bands are policed** — `tools/band_history.py` now reads every gated dict
   (`POLICED_BLOCKS` covers `CASCADE_BAND`), and all four values carry ledger lines under this decision
   citing the A2/A3 measurement each came from (`ed56b79`). No band entered the tree without a record.
5. **The condition was met**: `progress/2026-10-14/route-c-audit-2026-10-14.md` proves the hand↔library delta to be
   reassociation at the element level (every library shape ≤ 1 ULP above its own cancellation band at
   M = 1, 88, 512, 1024; the three injected bugs separate from the legal delta by ≥ 492× on the least
   flattering statistic), and `--gemm library` is now the default in the binary and in the gate
   together (`0fea0db`), on a first genuine 8-fixture × 2-route matrix that is 16/16 green — see
   `engine/results.md`, route (c), “Admitted”.

**The weakened guarantee, said plainly.** On the library route the wave-boundary check is **inside a band,
not byte-identical**. A reader who comes to the route-(c) sections expecting the pre-route invariant — the
one that made “the same numbers at every width” true — must not read this decision as still granting it.

**Cost actually paid, and the bill that came with it.** Two gate bugs were found only because flipping the
default forced a genuine library run, and both were of the class “a checker that cannot fail”: the route
echo compared the engine's handle spelling (`route=hipblas`) with the judging route while the dry-run
stand-in printed the route word instead, and `cross_width_ids` mapped a decode row one prompt-length past
the dump. They are written up in `engine/results.md` route (c) rather than fixed quietly, with the
incident they hid: the “16/16 on both routes” claimed by `0b30422` was sixteen hand runs, because the A/B
harness put the route in a wrapper binary whose flag the gate's own appended `--gemm` overrode.

