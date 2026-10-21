# G4 — llama's thread map under our recurrence: 128 blocks x 128 threads, 16 state elements per thread in registers

Task: `prototype/gdn_walk/BET.md` §12 adopted G4 and set its lines before the code existed;
`prototype/gdn_walk/G3-llama-kernel.md` §6 is the constraint set. This file is the record of what was
built, what was proved on CPU, the one short device gate that was run, and the timing A/B handed to the
main agent. **It contains no millisecond figure taken by this worker.** Every ms below is quoted from
`progress/2026-10-21/performance-status-2026-10-21.md` §42/§43 or from BET.md §12's pre-registration, cited as such.

Kernel: `engine/gdn_rec_reg16_kernels.h::rec_reg16_kernel`. Knob: `--gdn-geom reg16`, **default stays
`block`**.

---

## 1. MILESTONE 0 — the note block, read before any A/B, any gate, any device run

The entire bet rests on "16 elements/thread fits in registers". Read on a fresh `-O3 gfx1151` compile of
`engine/kernels_gdn.cpp` (the listing is the `--save-temps=obj` GCN `.s` for that TU; both kernels come
out of ONE compiler invocation, so the two note blocks are paired by construction):

```
hipcc -O3 -Wno-unused-command-line-argument -I. -Iengine --save-temps=obj \
      -c engine/kernels_gdn.cpp -o tmp/gdn_walk/s/kernels_gdn.o
.venv/bin/python prototype/attn_wmma/fattn_residency_read.py \
      --asm tmp/gdn_walk/s/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s \
      --kernel _Z16rec_reg16_kernelPKfS0_S0_S0_S0_PfS1_ii --threads 128 --warps 4
```

| note field | **`rec_reg16_kernel` (new)** | `rec_loop_kernel` (shipped) | `rec_wf_kernel<4>` (G2, dead) | llama's `gated_delta_net_cuda<128,false,false,4>` (G3 §2) |
|---|---|---|---|---|
| `vgpr_count` | **76** | 75 | 54 | 78 |
| `vgpr_spill_count` | **0** | 0 | 0 | 0 |
| `sgpr_count` | **26** | 40 | 50 | — |
| `sgpr_spill_count` | **0** | 0 | 0 | 0 |
| `private_segment_fixed_size` | **0** | 0 | 0 | 0 |
| `group_segment_fixed_size` | **0** | 1,280 B | 0 | 0 |
| block threads / warps | 128 / 4 | 128 / 4 | 32 / 1 | 128 / 4 |

**The premise holds, and by the margin G3 predicted: 76 VGPR with 0 spills against llama's 78 with 0
spills.** No scratch (`private_segment_fixed_size 0`, and `scratch_load`/`scratch_store` appear **zero**
times in the body), no LDS at all (`group_segment 0`, where the shipped form pays 1,280 B for its
`ks`/`qs`/`red` arrays), so there is no `--ack-spill` to ask for and no spill argument to have. Per
`prototype/attn_wmma/fattn_residency_read.py`'s derivation from KFD (`lds_size_in_kb 64`, `simd_count 80`)
and `rocminfo` (Max Work-item Per CU 1024, Max Waves Per CU 32): the resource capacity is 8 blocks/CU and
32 wf/CU, i.e. **the resources do not bind at this grid** — 128 blocks over 40 CUs is 3.2 blocks/CU
realised, 12.8 wf/CU, **6.40 wf/SIMD against the shipped 0.80** (G3 §1's number, now ours).

Two things the note block cannot see and the ISA read in §3 does: whether the shard is *actually* kept in
registers across the token loop (a `vgpr_spill_count 0` is necessary, not sufficient — a rolled index
into `shard[]` would put it in scratch and the note block would say so, but a *sunk* load would not), and
whether any state traffic survived into the per-token path.

One honest note on the register count: the peak live payload is 3 x 16 floats (state shard + this token's
k slice + its q slice = 48 of the 76), which is llama's payload exactly (G3 counts 48 of their 78 on
`s_shard`/`k_reg`/`q_reg`). That is only true because `kh`/`qh` are **normalised in place** — they hold raw
`k`/`q` until the row norms land and become `k̂`/`q̂` in the same registers. Written as five live arrays
(k, q, k̂, q̂, shard) the same kernel would need 80 floats of payload and this note block would be the
first thing to go red.

## 2. The geometry as built, against `gdn.cu:203-204`

| field | `rec_reg16_kernel` as launched | llama `gated_delta_net_cuda<128,false,false,4>` | ours, shipped `rec_loop_kernel` |
|---|---|---|---|
| grid | `dim3(NH, D/16)` = **(16, 8)** = 128 blocks | `dim3(H, n_seqs, S_v/(4·4))` = (16, 1, 8) = 128 blocks | `<<<16, 128>>>` = 16 blocks |
| block | `dim3(32, 4)` = **128 threads** | `dim3(warp_size, num_warps)` = (32, 4) | 128 threads |
| wavefronts / launch | **512** | 512 | 64 |
| wf/SIMD (KFD `simd_count` 80) | **6.40** | 6.40 | 0.80 |
| columns per block / per warp | **16 / 4** | 16 / 4 | 128 (whole head) / 32 |
| lanes per state column | **8** (`lane & 7`) | 8 (`lane % lanes_per_col`) | 1 (thread `j` owns column `j`) |
| state elements per thread | **16** | 16 | 128 |
| state layout | `[i][j]`, unchanged | `[col][i]` (transposed) | `[i][j]` |
| a thread's 16 elements | 16 consecutive `i`, **stride 512 B** | 16 elements stride-8 in `i`, stride 32 B | one column, stride 512 B, **128 of them** |
| state traffic per token | **0** (registers; 1 load + 1 store per span) | 0 | 3 passes over 1 MiB |

`kGdnR16Elems = 16` is the compile-time constant and every other dimension is derived from it
(`kGdnR16LanesPerCol = D/kGdnR16Elems = 8`, `kGdnR16ColsPerWarp = 4`, `kGdnR16ColsPerBlock = 16`,
`kGdnR16GridY = D/16 = 8`). The map identity is asserted in the source, not commented:

```cpp
static_assert(kGdnR16Waves * kGdnR16BlockX * kGdnR16Elems == NH * D * D,
              "THE map identity: wavefronts x 32 x elements_per_thread must tile the whole"
              " 262,144-element state exactly once.");
```

which evaluates to `512 * 32 * 16 == 262,144`, alongside `kGdnR16Elems == 16`,
`kGdnR16Elems * kGdnR16LanesPerCol == D`, `kGdnR16GridX == 16 && kGdnR16GridY == 8`,
`kGdnR16BlockX == 32 && kGdnR16BlockY == 4` and `kGdnR16Waves == 512`. The launcher
(`engine/kernels_gdn.cpp::gdn_rec_span_reg16`) launches with **those same constants**, in the same TU as
the kernel, and re-asserts `kGdnR16GridX * kGdnR16GridY == 128 && kGdnR16BlockX * kGdnR16BlockY == 128` —
the AGENTS.md 2026-10-19 row (a geometry fact in two headers gets one half committed and the tip does not
build) closed by construction rather than by review.

Difference from llama, deliberate: the state stays `[i][j]`, so a thread's elements are stride-512 B and
there is no vector state load. See §4 — it turned out to be the *cheaper* half of the difference.

## 3. The structural claims, read off the emitted ISA

`prototype/gdn_walk/walk_isa_diff.py --g4` (extended for G4 rather than writing a third ISA census tool —
AGENTS.md 2026-10-19: "before writing a census tool, list the census tools"). It finds the token loop as the
widest backward branch, then censuses instructions by *where they sit relative to that loop*, because the bet
is a claim about residency, not about arithmetic:

```bash
hipcc -O3 -Wno-unused-command-line-argument -I. -Iengine --save-temps=obj \
      -c engine/kernels_gdn.cpp -o tmp/gdn_walk/s/kernels_gdn.o
.venv/bin/python prototype/gdn_walk/walk_isa_diff.py --asm \
      tmp/gdn_walk/s/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s --g4
```

```
kernel              loop lines  in_gld  in_gst  in_wide  in_sload  out_gld  out_gst  barrier  lds_data  perm  scratch
reg16                 103..540       9       1        8         2       16       16        0         0    12        0
block(shipped)         63..597      37      17        0         0        0        0        5        19    20        0
RESULT walk_isa_diff --g4 PASS
```

What each column is paying for, since a census that cannot fail is decoration:

* **`in_gst 1`** — the token loop writes exactly one global address: `o`. No state store per token, which is
  the whole bet (the shipped form writes 17).
* **`in_gld 9`** — the per-token global reads are `k`, `q` (both halves of the row norm and the update), `v`,
  `beta`, `g` and the gate/norm scalars. Eight of the nine are `in_wide` (≥ `dwordx2`), i.e. the loads are
  vectorised, and two are `in_sload` — the uniform `g`/`beta` scalars.
* **`out_gld 16` / `out_gst 16`** — the 16 scalar state loads before the loop and the 16 stores after it:
  one load and one store per element **per span**, not per token. At `--gdn-loop 512` and 512 lanes that is
  one pass in and one pass out where the shipped walk does three passes per token.
* **`barrier 0`, `lds_data 0`** — the row norms and both state dots are 8-lane `ds_permute` butterflies
  (`perm 12`: three xor offsets x four reduction sites, roughly). G1's externality came from staging; this
  form has nothing to stage and therefore nothing to stage wrong.
* **`scratch 0`** — agrees with `vgpr_spill_count 0`: the shard is registers, not a stack frame.

The reference side has teeth: the shipped `rec_loop_kernel` row shows `in_gst 17`, `barrier 5`, `lds_data 19`.
`--selftest` re-proves the census can fail, with injections that must be caught *in the kernel that carries
them*: a state store moved into the loop (`in_gst 2 out_gst_b32 15`) and a shard load sunk into the loop
(`out_gld_b32 15 in_gld 10`), plus the case with no loop at all raising rather than censusing nothing.

## 4. Correct at any span, including the ragged tail — and where the launcher deliberately refuses

The kernel reads the state once at span entry and writes it once at span exit, so a span boundary is a
store/load round-trip of the same bits. On device this is **bit-exact across span spellings**: the
`512/512`, `512/13` and `512/22` runs (§6) produced *identical* logits and identical STATEFP lines, and
46 tokens is 1x46, 3x13+7 and 2x22+2 — so a 46-token single span, a ragged tail of 7 and a ragged tail of 2
all land on the same bits. The shipped form has the same proved property (`--gdn-loop` span invariance, D54),
so the comparison is like-for-like.

**One edge, found and certified rather than assumed, and NOT decided here.** A one-row span is refused by the
launcher, as every geometry launcher refuses it:

```console
$ ./engine/build/decode_run --model models/Qwen3.5-0.8B --ids "10,11,12" --steps 0 \
      --wave-rows 3 --gdn-loop 2 --gdn-geom reg16
decode_run FAILED: gdn_rec_span_reg16: tokens=1; a one-row body is `resolve_gdn_loop`'s to route to the
per-token kernels. Decode bits depend on that routing.
```

Reachability, stated exactly, because it decides how much this matters: a one-row span needs a *sub-wave*
span leaving remainder 1 (`--gdn-loop` < rows in the body and rows ≡ 1 mod span). The default
`--gdn-loop 512` cannot reach it — a body is at most `kMaxWaveRows = 512` rows, so `resolve_gdn_loop` returns
one span and a 1-row *body* is routed to the per-token kernels before any geometry launcher sees it. So the
refusal is unreachable on the shipping path and reachable on an explicit `--gdn-loop` combination. The kernel
itself **is** correct at `tokens == 1` (a single iteration; `g4_mirror.py`'s 2-token cases cover the
one-and-a-half-token shapes), so the remedy is a one-line relaxation of the launcher guard, and that is a
decision about decode routing rather than about G4 — handed back rather than taken. The refusal is loud and
the default is unaffected.

## 5. Numerics — the CPU mirror, against the shipped recurrence

Pre-declared position (BET §12 constraint 5, restated for the record): **G4 cannot be bit-identical to the
shipped walk and is not required to be.** Three sites re-associate: the `l2` of a row read by 8 lanes instead
of 128 threads, the state row-dot `k̂ · S_row` summed as an 8-lane butterfly instead of one 128-long serial
sum, and the output dot `q̂ · S_new` likewise. The admission is therefore the *existing* §10.1 aggregates —
`cos >= 0.9995`, `meanrel <= 5 %` — unchanged, no new band, nothing widened, judged against the shipped side
as the comparator (`prototype/gdn_walk/g2_mirror.py` supplies the shipped mirror and the aggregate helpers;
`g4_mirror.py` adds the reg-per-thread walk).

`prototype/gdn_walk/g4_mirror.py` simulates the map lane by lane — 16-row thread partials, 8-lane butterflies
with xor offsets 4/2/1 — and carries its own geometry witness (`--geom`: `262144 of 262144`, distinct 262144,
8 lanes/column, 16 columns/block) plus injections that must be caught (aliasing two lanes → 131072 distinct;
dropping `col_in_warp` → 65536). `RESULT g4_mirror SELFTEST PASS`.

On `data/refs/p64` at L00 and L01, plus two random cases (`--layers L00,L01 --random 2`, full output in
`tmp/gdn_walk/g4ab/mirror.txt` when the A/B re-runs it):

| comparison | `o` cos | `o` meanrel | state meanrel | reading |
|---|---|---|---|---|
| shipped mirror vs PINNED (L00) | 0.999998585 | 1.411e-03 | 1.044e-04 | the spelling that ships, at its own residual |
| **`reg16` vs PINNED (L00)** | **0.999998585** | **1.411e-03** | **3.082e-05** | **identical `o` residual, closer on state** |
| `reg16` vs shipped mirror (L00) | 1.000000000 | 3.501e-07 | 3.292e-06 | the map's own move |
| reorder-only floor (L00) | 1.000000000 | 2.466e-07 (`o`) | 2.471e-06 | what a pure reordering costs |
| shipped mirror vs PINNED (L01) | 0.999998589 | 1.411e-03 | 2.700e-06 | |
| **`reg16` vs PINNED (L01)** | **0.999998589** | **1.411e-03** | **2.431e-06** | **identical `o` residual, closer on state** |
| `reg16` vs shipped mirror (L01) | 1.000000000 | 1.301e-06 | 6.430e-07 | |
| reorder-only floor (L01) | 1.000000000 | 4.775e-07 (`o`) | 6.238e-07 | |

The row that matters is the second of each pair: **the map's `o` distance to the pinned reference is the
shipped spelling's distance, to four digits, on both layers, and on the state it is slightly closer**
(3.082e-05 against 1.044e-04 at L00, 2.431e-06 against 2.700e-06 at L01). The geometry is a re-grouping of the
same arithmetic, not an addition to its error. `G4 MIRROR PASS — every case inside the existing §10.1 band,
and the map is a bijection; no band was widened here.`

## 6. The one device gate, verbatim

Correctness-only, inside the worker budget, no timing. Command exactly as run, and it now exits 0:

```bash
.venv/bin/python prototype/gdn_walk/gate_g4.py --base-bin /tmp/base-g4/engine/build/decode_run \
  --token-steps 64
```

* `gpu_lock util@acquire: gpu use: 85%`, `util@release: gpu use: 57%`.
* `wall for all device runs: 12.0 s` — 14 `decode_run` launches (12 shape runs + 2 token-parity runs); 13 s
  for the whole script. **CORRECTNESS ONLY, no timing taken** — no warm loop, no best-of-N, and no ms figure
  here may enter `engine/performance-status.md`.
* `--base-bin` is a `git worktree` of `HEAD` (`2aa4d7e`) built into `/tmp/base-g4`, with `data`/`models`
  symlinked in.

Per-shape output, all four shapes (`512/512`, `512/13`, `512/22`, `16/8`), shipped side `block`:

```
TOKEN PARITY, --wave-rows 512 --gdn-loop 512, 64 free-running steps (bet-local, not gate.py):
  shipped emitted 110 ids, reg16 emitted 110; agreeing positions 110/110, IDENTICAL SEQUENCE

== shape --wave-rows 512 --gdn-loop 512: shipped comparator `block`, 46 prefill logit rows
  DEFAULT INERT: base vs new `block`: logit rows differing=0/46  STATEFP lines identical=True (18 state
  rows)  | new echo gdn_geom=block, base echo gdn_geom=block, base refuses `reg16`=1
   shipped row 45 vs PINNED logits: cos=0.999806723 meanrel=1.602e-01 rms_rel=1.973e-02
  side base: logits compared=46 differing=0 nonfinite=0 echo=ok
  side reg16: logits compared=46 differing=46 nonfinite=0 echo=ok
        vs shipped (report, not the gate): worst-row cos=0.999038085 meanrel=4.429e-01 maxrel(band)=1.122e+02
        vs PINNED logits, row 45 (THE GATE): cos=0.999697969 meanrel=2.009e-01 rms_rel=2.467e-02
                                           [shipped cos=0.999806723 rms_rel=1.973e-02]
        [EXCURSION, informative] rms_rel new/ship = 1.250290 against gate_g2's 1.25   exactly-at-the-bound=0
              (new rms_rel=2.466849e-02, shipped=1.973023e-02)
        top-5 contributors to the increase, out of 248320 elements  (positive part = 3.510e-04 of
        rms_rel^2, negative part = -1.317e-04, net = +2.193e-04):
          idx 145324 pin=  -0.9414 shipped=  -1.0460 new=  -1.2317 d(sq)=+4.469e-08 =  0.01% of the +part
          idx  95762 pin=  +2.6094 shipped=  +2.4821 new=  +2.3230 d(sq)=+4.010e-08 =  0.01% of the +part
          idx 132877 pin=  -4.7812 shipped=  -4.8655 new=  -5.0400 d(sq)=+3.647e-08 =  0.01% of the +part
          idx 136285 pin=  -2.5000 shipped=  -2.5852 new=  -2.7522 d(sq)=+3.435e-08 =  0.01% of the +part
          idx 129214 pin=  -3.2344 shipped=  -3.2773 new=  -3.4710 d(sq)=+3.299e-08 =  0.01% of the +part
        elements further from the pin than shipped: 152858 of 248320 (61.56 %) — a broad shift, not a
              clamp on a handful of elements
        STATEFP state: rows=18 identical=0 worst_rel=5.46e-03@layer13 conv: rows=18 identical=2
          worst_rel=1.62e-03@layer20 kv: rows=6 identical=0 worst_rel=3.06e-04@layer23
        layer-0 state move 4.31e-08 (<= 1e-06: the map's direct effect, before the cascade)

== shape --wave-rows 16 --gdn-loop 8:  (spans of 8, ragged tail of 6 in the last body)
   shipped row 45 vs PINNED logits: cos=0.999689165 meanrel=2.014e-01 rms_rel=2.521e-02
  side reg16: logits compared=46 differing=46 nonfinite=0 echo=ok
        vs shipped (report, not the gate): worst-row cos=0.998503153 meanrel=1.784e+00 maxrel(band)=2.507e+02
        vs PINNED logits, row 45 (THE GATE): cos=0.999847564 meanrel=1.381e-01 rms_rel=1.748e-02
                                           [shipped cos=0.999689165 rms_rel=2.521e-02]
        [EXCURSION, informative] rms_rel new/ship = 0.481025 against gate_g2's 1.25   exactly-at-the-bound=0
        elements further from the pin than shipped: 84097 of 248320 (33.87 %)
        STATEFP state: rows=18 identical=0 worst_rel=6.04e-03@layer10 conv: rows=18 identical=2
          worst_rel=2.18e-03@layer21 kv: rows=6 identical=0 worst_rel=8.58e-04@layer23
        layer-0 state move 5.57e-08 (<= 1e-06: the map's direct effect, before the cascade)

FAR FIELD — rms_rel to the pinned last row, new vs shipped, one entry per span  [INFORMATIVE: not one of
                                                                the §10.1 aggregates, does not decide]
     512x512 reg16: new=2.4668e-02 shipped=1.9730e-02 ratio=1.250290  EXCURSION past the 1.25 witness
      512x13 reg16: new=2.4668e-02 shipped=1.9730e-02 ratio=1.250290  EXCURSION past the 1.25 witness
      512x22 reg16: new=2.4668e-02 shipped=1.9730e-02 ratio=1.250290  EXCURSION past the 1.25 witness
        16x8 reg16: new=1.7484e-02 shipped=2.5209e-02 ratio=0.481025  inside the 1.25 witness
   the SAME shipped binary's own rms_rel across these spans: min=1.9730e-02 max=2.5209e-02 swing=27.8% with
   no arithmetic change — that spread is the noise floor of this statistic, and it is the same order as the
   excursion above

G4 DEVICE GATE PASS — finite, in band against the pinned reference on the pre-registered §10.1 aggregates
  with the shipped side as comparator, layer-0 effect at re-association scale, layer-0 conv untouched, and
  the token sequence identical.
```

### 6a. The excursion, characterised before it is set aside

The operator asked for the suspicious part spelled out rather than waved at, so here it is with the numbers.

**Is it at the bound, or does it merely land there?** `1.250290` — it *rounds* to 1.250 and is not equal to
it. `exactly-at-the-bound=0` is computed as `abs(ratio - 1.25) < 1e-9`, and the underlying values are
`new 2.466849e-02 / shipped 1.973023e-02`. So it is not a clamp or a plateau: nothing is pinned to the
threshold, and the check is not passing-or-failing on its own boundary. (An earlier draft of this block
printed `1.563224`, which was the ratio of the *squared* sums — `rms_rel` is a square root, so that number
was the witness squared. Caught by re-deriving it beside the aggregate line it describes; the fix is
`math.sqrt` in `far_field_report`, and the two numbers now agree.)

**What carries the increase?** Not a handful of elements. The five largest positive contributors are
**0.01 % each** of the positive part of the change — a 248,320-element row in which the top contributor is
one part in ten thousand is a broad shift, and the census says so directly: **61.56 % of the elements** are
further from the pin than shipped, 33.87 % at the other shape. The positive part is `3.510e-04` of
`rms_rel^2`, the negative part `-1.317e-04`, net `+2.193e-04`: the two directions nearly cancel, which is
what a re-associated sum over a chaotic cascade looks like and not what a wrong value looks like.

**Why it is informative and not the band.** The shipped binary's own `rms_rel` to these pins moves
`1.9730e-02 → 2.5209e-02` — a **27.8 %** swing between two spans with no arithmetic change whatsoever, and
the sign of the new side's excursion flips with the span (1.250 at `--wave-rows 512`, 0.481 at 16). A witness
whose noise is ~28 % cannot adjudicate a 25 % ratio. It is still computed, still printed, on every span, with
that noise in the same block; it no longer colours the exit code, because BET §10 constraint 2 named the
§10.1 aggregates as the authority. The 1.25 constant is untouched, the check is not skipped, and there is no
`--ack` flag — there is no flag at all, because a witness that cannot change the verdict needs no switch that
could turn a red into a green.

### 6b. The authority BET §10 named, now actually able to name the geometry

`engine/gate.py` had no `--gdn-geom` at all: its argv passed `--gdn-spell`, `--gdn-loop`, `--gdn-scan` and
`--gdn-chunk`, never the shape, so the pre-registered admission could not be exercised for *any* value of the
knob — `wf4` included — and the claim lived in a bet-local script beside the gate. With the operator's
authorisation, a `--gdn-geom` passthrough was added: argv-only, default `block` (the shipped form, so a gate
run that names nothing still judges what ships), no threshold, band, fixture admission or default changed.
The witness is the same one every other form flag uses — the run's own `NUMERICS` echo is asserted against
what the gate asked for, and a binary that predates the field fails rather than being judged as though it
carried the knob:

```console
$ .venv/bin/python engine/gate.py --fixture dec8 --gdn-geom reg16
  NUMERICS ... gdn_spell=fused gdn_loop=512 gdn_norm=inline gdn_geom=reg16
  [PASS] the run ran the gdn_rec map this gate is judging it under — NUMERICS echoed gdn_geom=reg16,
         this gate was run with --gdn-geom reg16
  [PASS] all 8 generated ids match, in order
  ENGINE GATE PASS                                        (wall 7 s, under gpu_lock)

$ .venv/bin/python engine/gate.py --fixture dec64f --gdn-geom reg16
  [PASS] the run ran the gdn_rec map this gate is judging it under — NUMERICS echoed gdn_geom=reg16
  [PASS] all 64 generated ids match, in order — 2 step(s) resolved by the tie policy
  worst step 34 at cos 0.998307, drift ratio 0.59 (last 8 steps BETTER than the first 8)
  ENGINE GATE PASS — with 2 step(s) reported TIE          (wall 13 s, under gpu_lock)
```

Those two lines are the sentence that matters: **on the pinned fixtures, through the named authority, at
`--gdn-geom reg16`, every generated id matches the reference — all 8 of dec8 and all 64 of dec64f** (the two
TIE steps are the pin's own exact ties, decided by the standing tie policy, and they are TIE at `block` too).
`dec8` also passes unchanged at the default `--gdn-geom block`, re-taken after the passthrough landed, so the
passage is proved to add nothing to the default path. Section (j) — the four-way spelling matrix — deliberately
keeps `block`, since `reg16` *is* the looped body and `ship`/`unroll4` have none; the gate prints a note saying
so whenever the run's geometry differs, rather than holding a form silently.

Everything the first version of this section rested on still stands and is stronger now: the §10.1 cos floor
passes on every span (0.999698 / 0.999848 against 0.9995); the layer-0 direct effect is 4.31e-08 / 5.57e-08
against a 1e-6 ceiling; layer-0 conv is bit-identical; deeper-layer moves are 5.5e-03 / 6.0e-03 against the
1e-2 cascade line; and free-running token parity is now 64 steps rather than 16 — **110/110 ids identical**,
with the top logits visibly moved beside it (21.6397 vs 21.6302 at one step). The far-field excursion moves no
decision, and on the forced fixtures the decisions are the pinned ones.

## 7. Default inertness, proved against a pre-G4 tree

`base` is `HEAD` (`2aa4d7e`) built in its own worktree; the new binary is this tree. At every shape:

```
logit rows differing = 0/46        STATEFP lines identical = True (18 state rows)
new echo gdn_geom=block            base echo gdn_geom=block        base refuses `reg16` = 1
```

Byte-for-byte equality of every prefill logit row and every state fingerprint line, plus the two-trees witness
run for real rather than inferred: the baseline binary is *executed* with `--gdn-geom reg16` and must refuse
it (parse-time, no model load). The first version of that check asked whether the baseline's log contained the
string `gdn_geom` — which was true of G2's knob shipping in `8b65fb0` and so called a correct baseline stale.
Also refused, on the new binary: `--gdn-geom bogus` (parse by refusal), `--gdn-geom reg16 --gdn-loop 1`,
`--gdn-geom reg16 --gdn-norm hoist`, `--gdn-geom reg16 --gdn-scan collapse`. The default remains `block`
everywhere, including in `gate.py`'s own argv, which never passes the knob.

## 8. The timing A/B, handed over

`prototype/gdn_walk/run_ab_g4.sh` — **main-agent device window only**, blocking, zero subagents, one
`tools/gpu_lock.sh` hold, self-locking (it acquires and releases; do not hold the lock around it).

```bash
bash prototype/gdn_walk/run_ab_g4.sh                       # REPS=3, 4,096 lanes, sides base/block/reg16
REPS=2 bash prototype/gdn_walk/run_ab_g4.sh
SUMMARIZE=1 bash prototype/gdn_walk/run_ab_g4.sh           # re-judge the logs, touch no device
BASE=<sha> BASEDIR=/tmp/base-g4 bash prototype/gdn_walk/run_ab_g4.sh
```

Legs, in order. **Before** the hold: canary A (the baseline must refuse the *name* `reg16`, for the right
reason — `got 'reg16'` — and the new binary must accept it; `engine/build.sh` is not bit-reproducible here so
two binaries agreeing proves nothing); numerics leg 1 (`base` == `block` bit for bit); numerics leg 2
(`reg16` must MOVE the state fingerprint — an identical STATEFP would mean the flag selected nothing and its
ms would be the shipped form's a second time); the CPU mirror plus its `--geom` bijection; and a reminder
that `gate_g4.py` self-locks so it runs outside the window.

Inside the hold, before any timing: **canary C, the geometry itself, from a `rocprofv3` trace in the same
window** — `tools/prefill_kernel_trace.py` (which does not self-lock) then `tools/trace_phase_split.py --geom
rec_ --calibrate attn_prep --per-pass 48`. It asserts the `block` census contains a `(128,1,1)` row and the
`reg16` census contains a `(32,4,1)` row with **128 blocks/launch** and names `rec_reg16_kernel`. This is the
leg that answers the stale-binary null on the *shape* axis: the launch count is 144 on every side, so canary
B alone cannot tell the two maps apart. Per rep, canary B: each side echoes `gdn_geom=<side>`, every side
shows `family=gdn_rec launches=144 plan_launches=144 counts_match=1`, and `TOTAL … plan_match=1`.

Pre-registered outcome (BET §12 / G3 §6), medians of 3 at 4,096 lanes, base = §42's window (`gdn_rec` 264.00 ms,
`wall_ms` 626.44 ms):

| metric | base | live | dead |
|---|---|---|---|
| `gdn_rec` family ms | 264.00 | **≤ 170** | **> 225** |
| wave `wall_ms` | 626.44 | **≤ 540** | **> 607** |

Both must clear live to ship; either past dead kills; anything else is UNRESOLVED and reported as unresolved,
with no narration. **`wall_ms` is the ship gate**, and the script prints the externality sum over every family
other than `gdn_rec` beside it (§40's lesson, and G4 is the form most likely to buy an externality: it stages
no LDS, so any traffic it loses shows up in a neighbour).

The summary parser was colour-tested before handover, on synthetic logs, and all three verdicts come out:

```bash
SIDES="reg16=0.60" OUT=tmp/gdn_walk/parsecheck_g4_live  .venv/bin/python prototype/gdn_walk/mk_parsecheck.py
SUMMARIZE=1 OUT=tmp/gdn_walk/parsecheck_g4_live  NEW_SIDES=reg16 bash prototype/gdn_walk/run_ab_g4.sh
#   VERDICT LIVE  (gdn_rec 159.96 <= 170, wall 514.96 <= 540)
# scale 0.90 -> VERDICT DEAD      (gdn_rec 239.94 > 225)
# scale 0.80 -> VERDICT UNRESOLVED(gdn_rec 213.28, wall 568.28: between the lines)
```

`mk_parsecheck.py` gained an env-var side list (`SIDES="name=scale …"`, `OUT=…`) and refuses a duplicated side
name, because a side name is a file name there and a duplicate would be last-writer-wins.

## 9. What none of these checks can see

* **FMA contraction and association.** A CPU mirror reproduces the *formula*; two spellings that share a
  formula agree, so the mirror is structurally blind to what the compiler does with each product tree. The
  shipped expressions were transcribed character for character for that reason (AGENTS.md's "a hoisted scalar
  constant is a different number" row), but the mirror cannot *witness* it — only the near-field device
  numbers (4.3e-08) bound how much that could matter.
* **Why the far field reads 1.25 at one span and 0.48 at another.** Consistent with cascade amplification and
  quantified as such in §6a — the shipped side's own swing across the same two spans is 27.8 %, and the
  increase is spread over 61.56 % of the row's elements rather than carried by a few. The pinned fixtures
  settle the question the ratio could not: at `--gdn-geom reg16`, `engine/gate.py` reproduces all 8 ids of
  `dec8` and all 64 of `dec64f` (§6b). What remains genuinely unseen is the *shape* of that noise on a fixture
  whose chain forks differently, which no single-fixture run can bound.
* **Speed, at every level of the word.** No ms in this file was taken by this worker, and none may be copied
  into the ledger from a worker run.
* **The 4,096-lane geometry at scale.** The device gate runs 46-token prompts at `--wave-rows 512` and 16; the
  census says the map is what it claims on a 46-row body, not on a 4,096-row prefill. Canary C closes that on
  the real shape when the A/B runs.
* **Decode.** Every leg is `--steps 0` prefill except the token-parity leg, which free-runs 16 steps. A
  one-row body never reaches this kernel by design, so the decode step is untouched — and untested for this
  knob, which is the same statement as §4's refusal.

## 10. Decisions taken, decisions left

1. **Settled by the operator (2026-10-19): the pre-registered band governs.** BET §10 constraint 2 and §12 name
   the §10.1 aggregates as the authority; `gate_g2`'s 1.25x far-field ratio is an inherited heuristic whose own
   noise (§6a: 27.8 % on the shipped side itself) exceeds its margin, so it is recorded, printed with that noise
   beside it, and does not decide. G4 is **band-clean with a documented excursion** and the timing A/B in §8 is
   worth running. What was *not* done getting there: the constant was not touched, the check was not skipped or
   deleted, and no `--ack` switch was added — the flag that had briefly existed for exactly this was removed,
   on the ground that a witness which does not decide should not carry a dial.
2. **Done under that same authorisation:** `engine/gate.py` gained the `--gdn-geom` passthrough (§6b),
   argv-only, default `block`, with the NUMERICS-echo witness so it cannot silently no-op. Two notes for
   whoever next touches that file: section (j) holds the shape at `block` **on purpose** — three of its four
   spelling sides have no looped body for `reg16` to travel in — and prints a note whenever the run's geometry
   differs, rather than holding a form silently; and `gate.py --help` raises
   `ValueError: unsupported format character 'v'` on a `5 %` sitting in another flag's help string, which is
   **pre-existing at HEAD** (verified by running `git show HEAD:engine/gate.py` in isolation), not something
   this change introduced, and which leaves the usage text unreadable until someone escapes it.
3. **The one-row-span refusal (§4).** Unreachable on the shipping path, loud when reached, one line to relax.
   Not relaxed here.
4. **Nothing is committed and the default does not move.** If the operator decides to carry G4, the kernel
   header and the launcher must land in one commit — `engine/gdn_rec_reg16_kernels.h` owns the constants and
   `engine/kernels_gdn.cpp::gdn_rec_span_reg16` launches with them, and the AGENTS.md 2026-10-19 row is about
   exactly that pair (assert alone is not enough; build the *committed* tree). The `engine/gate.py` passthrough
   is a different topic and should be its own commit, ahead of the kernel work if the order matters.

## 11. Files and re-take commands

New: `engine/gdn_rec_reg16_kernels.h`, `prototype/gdn_walk/g4_mirror.py`, `prototype/gdn_walk/gate_g4.py`,
`prototype/gdn_walk/run_ab_g4.sh`.
Modified: `engine/kernels.h` (the `kReg16` enumerator and its comment), `engine/kernels_gdn.cpp` (the
launcher, the parse/name pair), `engine/decode_step.cpp` (the route), `engine/decode_run.cpp` (the flag,
its refusals and its help), `engine/gate.py` (the `--gdn-geom` passthrough and its echo witness, argv-only,
default unchanged — §6b), `prototype/gdn_walk/walk_isa_diff.py` (`--g4`),
`prototype/gdn_walk/mk_parsecheck.py` (side list via env).

```bash
bash engine/build.sh                                        # both binaries, ~18 s
.venv/bin/python speccheck/run_all.py --quick               # 11 gates, 0 failed
.venv/bin/python prototype/gdn_walk/g4_mirror.py --geom
.venv/bin/python prototype/gdn_walk/g4_mirror.py --selftest
.venv/bin/python prototype/gdn_walk/g4_mirror.py --layers L00,L01 --random 2
.venv/bin/python prototype/gdn_walk/walk_isa_diff.py --asm <listing> --g4
.venv/bin/python prototype/gdn_walk/walk_isa_diff.py --asm <listing> --selftest
.venv/bin/python prototype/gdn_walk/gate_g4.py --base-bin /tmp/base-g4/engine/build/decode_run   # device
```

