# `prototype/attn_score_dot2` — handoff

Everything in steps 1–4 is **CPU-only**: compiles and reads assembly, launches nothing, takes no
`tools/gpu_lock.sh`, takes no timing. Step 5 is the one device measurement that could overturn the verdict
and is main-agent work by rule. Step 6 is what to do if it does.

The verdict being handed over: **the dot2 prize is 3.9–10.5 % of the wave against a 15 % line, so no kernel
was built** (`results.md` §0). A worker ran none of the device steps; `results.md` quotes no ms of its own
making — every ms in it is D19's measured table or arithmetic on it.

---

## 1. Regenerate the disassembly of both shipped forms (CPU, ~1 min, no device)

`--save-temps` drops the `.s` files in the **current working directory**, so run both compiles from
`/tmp/isap` with absolute source paths — that keeps ~750 KB of assembly out of the tree (`.gitignore` trap: a
`prototype/**/<name>` line can swallow a whole op directory).

```bash
REPO=$PWD            # from the repo root
mkdir -p /tmp/isap && cd /tmp/isap
hipcc -I"$REPO" -O3 --offload-arch=gfx1151 --save-temps -c \
    "$REPO/prototype/attn_score_dot2/isa_tu.cpp"    -o /tmp/isap/tu.o      # attn_tile_wave + attn_prep
hipcc -I"$REPO" -O3 --offload-arch=gfx1151 --save-temps -c \
    "$REPO/prototype/attn_score_dot2/isa_tu_m2.cpp" -o /tmp/isap/tu_m2.o   # attn_m2_wave
cd "$REPO"
```
The two `.cpp` files are two-line TUs that only `#include` the kernel headers — that is the whole reason they
exist, so a header with no `main()` can be compiled standalone. After this, `/tmp/isap` holds
`isa_tu-hip-amdgcn-amd-amdhsa-gfx1151.s` and `isa_tu_m2-hip-amdgcn-amd-amdhsa-gfx1151.s` (the GCN side; the
`-host-x86_64-…` files are the host stubs and are not interesting here).

## 2. Compile the dot2 shape toy (CPU, seconds)

```bash
cd /tmp/isap
REPO=${REPO:-$PWD}   # steps 1–2 each assume $REPO is the repo root; set it once per shell
hipcc -O3 --offload-arch=gfx1151 --save-temps -c \
    "$REPO/prototype/attn_score_dot2/dot2_shape.cpp" -o /tmp/isap/dot2_shape.o
cd "$REPO"
```
`dot2_shape.cpp` is never executed — it is a compile-proof, and its `extern "C"` names survive mangling so a
census script can select them. It leaves `dot2_shape-hip-amdgcn-amd-amdhsa-gfx1151.s` beside the others.

## 3. The census (CPU)

```bash
.venv/bin/python prototype/attn_score_dot2/isa_census.py /tmp/isap/isa_tu-hip-amdgcn-amd-amdhsa-gfx1151.s \
    --kernels attn_tile_wave --loop-has ds_bpermute_b32
.venv/bin/python prototype/attn_score_dot2/isa_census.py /tmp/isap/isa_tu_m2-hip-amdgcn-amd-amdhsa-gfx1151.s \
    --kernels attn_m2_wave --loop-has ds_bpermute_b32
.venv/bin/python prototype/attn_score_dot2/isa_census.py /tmp/isap/isa_tu-hip-amdgcn-amd-amdhsa-gfx1151.s \
    --kernels attn_tile_wave --ops 'v_exp_f32|v_ldexp|v_div_|v_rcp_f32|v_rndne'
.venv/bin/python prototype/attn_score_dot2/isa_census.py /tmp/isap/dot2_shape-hip-amdgcn-amd-amdhsa-gfx1151.s \
    --kernels score_scalar,score_dot2
```
Expected, and what to look at if it differs:

* the toy: `score_dot2` prints **`v_dot2_f32_bf16=4` and conversions=0**; `score_scalar` prints conversions
  and FMAs and no dot2. If the dot2 count is not 4, the builtin did not lower the way §2.2 claims — that is a
  fact about the compiler, and the price changes.
* `--loop-has ds_bpermute_b32` finds the score loop (the shuffle permute is its fingerprint). Its
  `VALU=` line is **approximate by construction**: LLVM unrolls the 5-step butterfly under guards and spreads
  it over blocks that do not all carry the loop annotation, so the per-touch numbers in `results.md` §2 come
  from quoted windows with line numbers, not from this line. Treat a large disagreement as a compiler change,
  not as a census bug.
* `--ops` is the static share of `expf`/divide instructions. It is a share of the *static* stream, so it is
  context for `results.md` §5, not a per-touch cost.

## 4. The price itself (CPU)

```bash
.venv/bin/python prototype/attn_score_dot2/price.py --isa-dir /tmp/isap
```
Prints the pins, MACs by part, the three framings per wave, the 0.246 ms forensics, and the instruction
breakdown of the rest of the wave. `price.output.txt` is a capture of it; a re-run should reproduce it
digit for digit, and if it does not, something's input changed and the verdict must be re-read.

## 5. THE ONE MEASUREMENT THAT CAN OVERTURN THIS (device — main agent, one lock hold, no subagents)

The price borrows one census from the shipped form: **114.6 VALU warp-inst per (row, lane, head) touch**
(`prefill-occupancy-results-2026-10-16.md` §3). The m2 form's own per-touch count has never been measured.
The instruction framing says the prize is `12 / census`, so the verdict has an exact falsification point:

> **if `attn_m2_wave` costs fewer than 80 VALU warp-inst per touch, the bound rises above 15 % and step 2 is
> funded after all.** (12 / 80 = 15 %.)

```bash
tools/gpu_lock.sh acquire dot2-census
tools/gpu_lock.sh util
.venv/bin/python prototype/attn_m2/run.py plans                      # writes the plan files, no device
rocprofv3 --pmc SQ_INSTS_VALU -o m2w -d /tmp/m2wave -- \
    ./prototype/attn_m2/attn_m2 --plan /tmp/attn_m2/plan-wave64-at-4096.txt \
        --neg none --split 8 --out /tmp/m2wave/dump
tools/gpu_lock.sh release dot2-census
```
The `attn_m2` arguments are exactly the ones `run.py run_bin` passes, so the instrumented launch is the same
launch the gate makes; `--split 8` is that plan's own split (printed by `run.py plans`), not a knob to pick.
Then divide, with arithmetic visible:

```
touches(wave64-at-4096) = 260,128 lanes x 8 heads = 2,081,024      # price.py prints this
census = SQ_INSTS_VALU(attn_m2_wave, deduped by Dispatch_Id) / 2,081,024
```
Dedup by `Dispatch_Id`, never `Correlation_Id` (same doc, §"how the counters were taken"). Run it blocking,
zero subagents, and record `util` then discount it. This is a counter, not a timing, but it is still device
work and only the main agent owns it.

**Report back**: `census=`, and whether it is below 80. Do not act on it in the same window; the threshold is
not a number to apply by hand — `price.py` prints it as its `FALSIFIER` line (it is
`(DOT_SCALAR_VALU - DOT_DOT2_VALU) / PRIZE_LINE` over its own constants), so the re-decision is made by
editing nothing and re-reading the script's verdict line, and it belongs to the reviewer.

## 6. If step 5 falsifies the price — the order of work, gate plan first

`results.md` §6 is the gate plan; this is the order to execute it in, CPU before device:

1. `prototype/attn_score_dot2/floor_probe.py` (to be written, pattern: `prototype/mlp_gemv/floor_probe.py`)
   over the pinned `q_rope`/`k_rope` dumps, reproducing the dot2 association on CPU, sample ≥ 1e6 scores
   from the wide anchors — **before any kernel exists**. It decides whether a band is needed at all: F1 in
   `prototype/attn_decode/results.md` §5 says a wider re-association of this same dot flipped 0 of 15,592
   scores, but 0-of-15,592 cannot see the 1-in-1e5 cancellation event AGENTS.md warns about.
2. One `--dot2` association switch in `prototype/attn_tile_model.py` and `prototype/attn_m2/m2_model.py`,
   applied to **both sides** of every comparison (a reference read out of the form under test cannot adjudicate
   it — §20.3), then re-derive the must-catch magnitudes and re-check that every derived injection still moves
   ≥ 10× the dot2's own movement and that the must-not-catch rows stay **bit-exact**. Widening the blind-row
   predicate is the failure mode, not the band on the must-catch side.
3. Only then the kernel, behind a flag, and the device gates in the existing windows:
   `prototype/attn_decode/run.py` (pinned stage bands), `prototype/attn_m2/run.py ab` (bit-identity — and read
   `results.md` §6.5 before assuming it can still pass), `run.py negative` per class.
4. `speccheck/run_all.py --quick` is the right level; a dot2 in this kernel cannot touch
   `speccheck/specimpl.py`, so the full 26-gate run is not owed.

Step 6 has a decision in it that is not yours or mine: if M = 2's bit-identity admission cannot survive the
dot2, that is a gate-policy question for the reviewer, and per the standing rule it is reported, not routed
around.

---

**Archived 2026-10-21** from `prototype/attn_score_dot2/HANDOFF.md`, where it sat beside the rig it briefed. The device pass it hands over is done and its verdict is in that op's `results.md` and in the ledger, so `progress/README.md` rule 1 puts it here. The filename now carries the op name because a dated folder holds handoffs from several ops at once (rule 3's collision clause); citations naming the old path were re-pointed in the same commit.
