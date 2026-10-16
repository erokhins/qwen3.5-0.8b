# HANDOFF — the device pass for the `--attn-touch` port

Everything here was written and verified **without a device**: the two levers were transplanted, the tree
builds, the instruction census on the engine binary says the levers landed, and the CPU probe for both
levers passes. What is **not** done, and cannot be done by a worker, is every number below — this tree's
device gates are the main agent's, run blocking, in one `tools/gpu_lock.sh` hold, with zero subagents.

## What was ported, and where

`prototype/attn_valu_slim/`'s two levers now run in the shipping engine.

| file | change |
|---|---|
| `engine/attn_slim_kernels.h` | new. `attn_tile_body` copied from `prototype/attn_decode/attn_decode_kernels.h` with the two levers applied, exported as `attn_slim_wave`. Header note carries the provenance, both levers, what neither lever touches, and the census command that guards the copy |
| `engine/kernels_attn.cpp` | includes it after the op header; the tiled-wave launch site picks `attn_slim_wave` or `attn_tile_wave` from `AttnLayerArgs::wave_touch`. Same grid, same block extent, same `CoreArgs`, same `part` expression — only the kernel differs |
| `engine/kernels.h` | `AttnLayerArgs::wave_touch = 1` |
| `engine/decode_step.h`, `engine/decode_step.cpp` | `DecodeScratch::attn_touch`, copied into `a.wave_touch` beside `a.wave_tile` |
| `engine/decode_run.cpp` | `--attn-touch 0\|1`, parse-or-throw like `--attn-tile`, **default 1**; echoed as `attn_touch=%d` on the NUMERICS line |
| `engine/gate.py` | `--attn-touch 0\|1`, **default 1**, on the main argv and on every side of the batched A/B, and asserted against the run's own NUMERICS echo — **a binary whose NUMERICS line carries no `attn_touch=` fails the gate**, it is not treated as inert |

`prototype/attn_decode/` and `prototype/attn_m2/` are untouched, so the frozen reference the port was
proved against is still unmodified source, and the A/B is one flag inside one binary.

## The control, before the port (no device)

Census of the engine TU *before* any edit — `isa_census.py` reused unchanged:

| quantity | engine `attn_tile_wave`, before | `prototype/attn_valu_slim` census of the same kernel |
|---|---|---|
| whole-kernel ops / slots / VALU / LDS / VMEM / SALU | 1454 / 1416 / 908 / 17 / 10 / 475 | 1454 / 1416 / 908 / 17 / 10 / 475 |
| one touch: slots / VALU / LDS / SALU | 116 / 55 / 14 / 47 | 116 / 55 / 14 / 47 |
| `dot.lds-load` / `dot.q-indexed` / `fold.exec-guard` / `fold.range-select` | 8 / 8 / 16 / 7 | 8 / 8 / 16 / 7 |
| VGPR / SGPR / LDS / waves-per-SIMD | 47 / 52 / 12288 / 16 | 47 / 52 / 12288 / 16 |

Identical in every cell, which is the answer to "is the engine's touch bill the one the levers were
written against": yes, byte for byte, so the transplant buys what D21 says it buys.

## After the port (no device)

`engine/attn_slim_kernels.h` and `attn_tile_body` are both compiled into `engine/kernels_attn.cpp`
unconditionally — the flag chooses a **launch**, never a build — so one listing carries both sides:

| kernel | ops / slots / VALU / LDS / VMEM / SALU | slots / VALU / LDS / SALU per touch | VGPR / SGPR / LDS / waves |
|---|---|---|---|
| `attn_tile_wave` (frozen side) | 1454 / 1416 / 908 / 17 / 10 / 475 | **116 / 55 / 14 / 47** | 47 / 52 / 12288 / 16 |
| `attn_slim_wave` (both levers) | 1565 / 1524 / 965 / 24 / 10 / 519 | **45 / 28 / 7 / 10** | 47 / 54 / 12288 / 16 |

Those two rows are the same numbers `prototype/attn_valu_slim/isa_census.output.txt` prints for
`attn_tile_wave` and `attn_slim1_wave`, which is the copy-faithfulness witness: the engine's slim kernel
has the prototype's instruction bill, not one of its own. VGPRs unchanged, LDS unchanged, occupancy
unchanged, SGPRs 52 → 54, whole-kernel ops up by 111 because the score loop is emitted twice.

CPU witness, unchanged and re-run: `.venv/bin/python prototype/attn_valu_slim/touch_probe.py` →
`PROBE summary PASS` (dot bit-identical at both trip counts, fold bit-identical in **every** lane,
must-catch 2002/2009 fold and 1000/1000 dot, no must-not-catch violation).

Build witness: the full `engine/build.sh` command line compiles clean with the change,
`decode_run --help` lists `--attn-touch 0|1`, and `--attn-touch 2` refuses rather than defaulting.
`engine/gate.py --selftest` still says SELFTEST PASS. No device was touched by any of that.

---

## The device pass, in the order to run it

Lock protocol per command is marked. **None of these self-lock** — `engine/gate.py`,
`prototype/attn_decode/run.py` and `decode_run` do not acquire `tools/gpu_lock.sh`
(`grep -c gpu_lock engine/gate.py` is 0; `prototype/attn_decode/run.py`'s own `--tile-kernel` help says
it does not), so **the caller holds the lock around all of them**. The three self-locking drivers
(`prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`) are not part
of this pass; if you add one, release first or it parks silently.

### 1. Re-take the census and the probe — no device, no lock

```
REPO=$PWD; mkdir -p /tmp/attn_census && cd /tmp/attn_census
hipcc -O3 -Wno-unused-command-line-argument -I"$REPO" -I"$REPO/engine" --save-temps \
    -c "$REPO/engine/kernels_attn.cpp" -o k.o
cd "$REPO"
.venv/bin/python prototype/attn_valu_slim/isa_census.py \
    --asm /tmp/attn_census/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s \
    --kernels attn_tile_wave,attn_slim_wave
.venv/bin/python prototype/attn_valu_slim/touch_probe.py
```

The `cd` is not tidiness: `--save-temps` writes the device listing into the CWD, not beside `-o` (the
census script's own comment says so), so a `-o /tmp/x.o` run from the repo root drops a 400 KB `.s` in
the working tree and then finds nothing where it looks. The include paths are absolute for the same
reason.

Admissible: `116/55/14/47` and `45/28/7/10`, and `PROBE summary PASS`. If `attn_tile_wave`'s row moved,
the prototype's kernel changed under this copy — stop and re-derive, do not re-book the win.

### 2. Build — no device, no lock

```
./engine/build.sh
```

Do **not** pass `--build` to `gate.py` below unless you want it rebuilt in the middle of the lock hold.

### 3. Bit-identity on the engine path — DEVICE, caller holds the lock, at BOTH values

This is the witness that matters: the prototype's `ndiff=0` proved the *copy*, these prove the *engine*.
Both values, same lock hold, so the two runs see the same machine.

```
tools/gpu_lock.sh acquire attn-touch && tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture dec8 --attn-touch 0
.venv/bin/python engine/gate.py --fixture dec8 --attn-touch 1
.venv/bin/python engine/gate.py --fixture wide520f --attn-touch 0
.venv/bin/python engine/gate.py --fixture wide520f --attn-touch 1
.venv/bin/python engine/gate.py --fixture wide640f --attn-touch 1
.venv/bin/python engine/gate.py --fixture wide1024f --attn-touch 1
tools/gpu_lock.sh release attn-touch
```

Read three things in every log, not just the verdict line:

* `NUMERICS … attn_tile=1 attn_touch=<value>` — `gate.py` aborts the claim if the field is missing or
  disagrees with the flag it was given, so a run that says nothing is a FAIL, not a pass. That is the
  assertion that `--attn-tile`'s default and the engine's must move together, extended to the new flag:
  both defaults are 1 and they are in this one commit.
* the row-coverage line. Section (i) prints `coverage: the engine delivered all N pinned prefill rows`
  **before** it judges a single number, and `--prefill-logits` is forced on the argv by the fixture's
  `prefill_logit_rows` (`gate.py` raises it before assembling `cmd` and then asserts it on `cmd` with
  `cmd_flag`, per §17's defect). A `COVERAGE GAP:` line means the instrument, not the engine, is broken
  and no numerics verdict follows it.
* section (b)/(h)'s bit-identity verdicts. The claim on both values is **bit-identity**, not the cascade
  band: same eight products, same serial order, same butterfly, same clamp. If a fixture comes back
  inside a band but not bit-identical, that is a defect to fix, not a band to widen.

One honest limit on the new assertion: it sits in `gate.py`'s real-run path, so `--dry-run` and
`--selftest` do not exercise it (they printed PASS unchanged). What was checked on the CPU is that the
field is actually parsed out of the engine's line — a key nothing parses is a check that compares `None`
to `None` forever:

```
log = "NUMERICS wave_rows_cap=512 wave_rows=512 gemm=library hipblas=7.1.0 attn_tile=1 attn_touch=1"
      " gdn_scan=seq gdn_chunk=64"
num = gate.parse_run(log + "\n")["numerics"]      # -> attn_touch == "1"
gate.parse_run(pre_port_line_without_the_field)["numerics"].get("attn_touch")   # -> None
```

Both held when run here, so the absence branch really is a `None` and really fails. The script is
`tmp/attn_touch_echo_witness.py` (scratch, gitignored); it is 30 lines and loads `engine/gate.py` by path,
so if you want it permanent the right home is a case in `gate.py`'s `--selftest`, not a new tracked file.

`wide520f`/`wide640f`/`wide1024f` are the above-MAXSC anchors, i.e. the rows that ride the streaming
route and the split partials; `dec8` is the fixture every prior number was taken on. Run the full
fixture set if you want the wider claim — the six above are the ones that answer "did the port move a
bit on the route the levers actually touch".

### 4. The nondeterminism check for the hand-written `s_waitcnt` — DEVICE, caller holds the lock

A missing or misplaced wait is not a wrong-but-stable number, it is run-to-run disagreement. Cheap
version on the engine path, in one hold:

```
tools/gpu_lock.sh acquire attn-touch-race && tools/gpu_lock.sh util
O=progress/bench/evidence/prefill-occupancy-2026-10-16
for i in 1 2 3; do
    ./engine/build/decode_run --model models/Qwen3.5-0.8B \
        --ids "$(cat $O/ids1024.txt)" --steps 8 --out tmp/race_$i \
        --attn-tile 1 --attn-touch 1 > tmp/race_$i.log
done
diff -r tmp/race_1 tmp/race_2 && diff -r tmp/race_1 tmp/race_3 \
    && echo "RACE CHECK: three runs, byte-identical dumps"
diff <(grep -E '^(STEP|RESULT)' tmp/race_1.log) <(grep -E '^(STEP|RESULT)' tmp/race_2.log) \
    && echo "RACE CHECK: identical sampled ids and RESULT lines"
tools/gpu_lock.sh release attn-touch-race
```

A 1024-lane prompt wave runs the folded butterfly on the order of half a million (row, lane, head) touches
before it prints anything, so three byte-identical dumps is a witness rather than a lucky draw. Leave
`--prefill-logits` at its default: `all` would write 1024 full-vocabulary rows just to compare them. Any
difference kills lever 2 as written, and the fix is the `s_waitcnt`, not the gate. The prototype harness
already did this on its own binary — `prototype/attn_valu_slim/results.md` §7 diffed two full 12-plan
runs and got byte-identical logs — but that was not the engine's launch plan.

### 5. The frozen reference still agrees — DEVICE, caller holds the lock

```
tools/gpu_lock.sh acquire attn-ref && tools/gpu_lock.sh util
.venv/bin/python prototype/attn_decode/run.py --wave --tile-kernel
tools/gpu_lock.sh release attn-ref
```

`prototype/attn_decode/` was not edited, so this is the control on the control: it must print exactly
what it printed before this port. If it moved, something outside the port changed and the `--attn-touch`
comparison in step 3 is judging a moving reference.

### 6. The paired wave timing — DEVICE, main agent only, one hold, zero subagents

Both sides inside one window, alternating, so the result is a pair and not two rulers
(`AGENTS.md`: a re-taken ruler is not an A/B). `--time-waves` is the instrument that produced the 412.07
ms of 1307.75 ms row in `progress/2026-10-16/prefill-occupancy-results-2026-10-16.md` §1; the row to
beat is `attn` at each width, and the widths are the ones whose ids are already committed.

```
tools/gpu_lock.sh acquire attn-touch-timing && tools/gpu_lock.sh util
O=progress/bench/evidence/prefill-occupancy-2026-10-16
for rep in 1 2 3; do
  for touch in 0 1; do
    ./engine/build/decode_run --model models/Qwen3.5-0.8B \
        --ids "$(cat $O/ids1024.txt)" --steps 0 --out tmp/t1024_$touch --time-waves \
        --attn-tile 1 --attn-touch $touch | grep -E '^WAVE|^TOTAL'
    ./engine/build/decode_run --model models/Qwen3.5-0.8B \
        --ids "$(cat $O/ids4096.txt)" --steps 0 --out tmp/t4096_$touch --time-waves \
        --attn-tile 1 --attn-touch $touch | grep -E '^WAVE|^TOTAL'
  done
done
tools/gpu_lock.sh release attn-touch-timing
```

`--steps 0` at 4096 on purpose: at 4097 lanes the engine refuses (split 9 > `kAttnMaxSplit` 8), which is
the same refusal that made the original occupancy table a `--steps 0` table. The `attn` family row is the
PROMPT wave either way, so use the same `--steps` on both sides of the pair and say which you used. Quote
the **pair** per width with the spread, and let the 4096 row carry the headline — the 1024 row is
dispatch-bound and bimodal on this box. The prediction to beat, from
`prototype/attn_valu_slim/results.md` §7 on the prototype's own harness: 1.52–1.57× on the attention
kernels; against the engine's wave, 412.07 ms at 4096 lanes means roughly 140 ms of a 1307.75 ms wave if
it reproduces.

Then, if you want the ledger's own TTFT rows rather than the family attribution:

```
.venv/bin/python tools/prefill_sweep.py --lanes 64,256,1024,3072 --runs 3    # main agent, under the lock
```

## What is still open after this pass

* **Nothing here has been run on a device by me.** Steps 1–2 are the only things I executed; steps
  3–6 are handed over deliberately, and their answers are what makes the port shippable rather than
  merely plausible.
* **The copy can drift.** `engine/attn_slim_kernels.h` is a copy of the prototype's body and nothing
  stops the two from separating. The census in step 1 is the guard (a shipped row that moves is visible
  in one column), and the header note says which file to read first. If `attn_tile_body` is ever edited,
  this copy must be re-copied and re-gated, not patched by hand.
* **`--attn-touch 0` is kept on purpose** — it is the frozen side for the census, the timing baseline,
  and any future attribution. If someone later deletes it, step 1's control row goes with it.
* **The injections stayed in the prototype.** The engine never injects `NEG_DIMHALF`, so lever 1's
  `nd == 4` branch is compiled and unreachable in the engine; the injection matrix that proves it agrees
  is `prototype/attn_valu_slim`'s (`--neg dimhalf`), not a gate in this tree.
