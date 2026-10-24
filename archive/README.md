# `archive/` — what was pulled out of `tmp/` before it was cleared (2026-10-24)

`tmp/` grew to **475 GB** and the operator asked for it to be cleared (the system `/tmp` had already been
emptied, which took the scratch binaries and rate-probe outputs with it). Nothing in `tmp/` is tracked, but a
census found **310 paths inside `tmp/` cited by committed files**, and 125 of them were real files — analysis
scripts, `rocprofv3` traces, and `.md` write-ups that `progress/` and `results.md` quote by path. Deleting
them silently would have made those citations point at nothing, which is the failure mode
`tools/citation_rot_check.py` cannot see (it checks paths, and a deleted path is not a rotted line).

## `tmp-cited-2026-10-24/` (tracked, ~11 MiB)

A path-preserving mirror: a citation to `tmp/gcc/fattn_residency.md` resolves to
`archive/tmp-cited-2026-10-24/tmp/gcc/fattn_residency.md`. Every file here is a byte copy of the thing that
was cited and was under 5 MiB. To re-materialise one at its original path:

```sh
cp -a archive/tmp-cited-2026-10-24/tmp/<path> tmp/<path>
```

The harness-fixture CSVs people keep reaching for are in here — `tmp/dec8_ids.csv` (cited 22×),
`tmp/verify_body_scale/ids4032.csv`, `tmp/llama_input/{ids_long,ids_long_chat,p_long}.{csv,txt}` — and they
are also regenerable from committed scripts, which is the better path when in doubt:
`prototype/mtp_accept/ids_from_text.py` (raw text → ids) and `prototype/mtp_accept/ids_from_chat.py`
(chat-template render → ids, byte-for-byte what `ids_long_chat.csv` was).

## `raw/` (gitignored, on disk, ~1.8 GB)

Ten oversized cited artifacts that must not enter git — the `rocprofv3` kernel traces behind the A11/A12/D52
counter claims (5–121 MB each) — plus the llama.cpp comparison fixture that `prototype/mtp_llama/llama_mtp_ab.sh`
needs: `Qwen3.5-0.8B.mtp.bf16.gguf` (1.5 GB, produced by this repo's own converter, `LLAMA-MTP-RESULT.md:22`)
and its prompt files. Same path-preservation rule: `archive/raw/tmp/<path>`. If disk pressure ever demands it,
this directory is the first thing to delete — every number it backs is already transcribed into a committed
`results.md`/`evidence/*.txt`, and the GGUF can be re-converted.

## What was deleted from `tmp/`

The run dumps: `glue_copy_fold/` (105 GB), `cast_fold/` (104 GB), `d3ab/` (70 GB), `flipc{,2,3}/` (60 GB),
`fattn_nan_gate/` (18 GB), `wide4096a.provisional/` (16 GB), `attn_percost/`, `wave_band_floor/`, `proj-d1/`,
`bet{1,2}_gate/` and the `mtp_speed_*` / `meter_rt_*` output trees — logits dumps, stage logs and intermediate
npy files. Their headline numbers live in `engine/results.md`, the per-op `results.md`, `progress/` journals and
`prototype/mtp_accept/evidence/`; the surviving MTP counters from 2026-10-24 are in
`prototype/mtp_accept/evidence/mtp_wall_and_meter_2026-10-24.txt`, and that file says plainly which runs lost
their per-rep logs to a reused default `OUT` directory — a harness lesson recorded there rather than hidden.

## Rule this directory is for

A `tmp/` path cited by a committed document is an artifact of record, not scratch. Either commit it (small) or
move it to `archive/raw/` (large) and say so here — or stop citing it. The deletion of a cited dump is fine
when the number it produced is transcribed and the tool that made it is committed; it is not fine when the dump
*is* the evidence.
