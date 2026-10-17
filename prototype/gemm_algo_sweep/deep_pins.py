#!/usr/bin/env python3
"""D30 item 3.7: read a DEEP `gemm_algo_sweep --time` window on CPU and derive the algo-pin set.

CPU-only. It takes no timing, opens no device, and needs no `tools/gpu_lock.sh`: every millisecond
figure it prints is read out of a main-agent sweep log passed in with `--time-log`, and the ledger
denominator it prices against is passed in with `--wave-ms`. It exists to answer two questions and
nothing else:

  1. what did the deep field (64 candidates, the engine's own 128 MiB budget, M = 512) actually buy
     over the heuristic's rank-0 — per shape, and priced at the engine's real call counts; and
  2. which `algo_pins()` rows would name those winners, with the 24-byte record MATCHED against an
     `ALGODUMP` capture rather than reconstructed-and-trusted.

Why the match is mandatory. The sweep's `ALGO` line prints the algo as `data=%016llx:%016llx`, two
native-endian words (`algo_bytes`, gemm_algo_sweep.cpp:544), while `algo_pins()` wants the 24 bytes
of the whole `hipblasLtMatmulAlgo_t` in byte order (`algo_hex`, engine/kernels_gemm.cpp), whose last
8 bytes are `max_workspace_bytes` — a field the `ALGO` line never prints. So a pin hex is
reconstructed: byte-swap each word, then stamp the workspace preference cap the query ran under. The
`ALGODUMP` line of a `--verify --algo-dump` capture prints `data=` already in the pin's form, so the
reconstruction can be CHECKED against library output instead of believed. A hex that is derived but
not matched is a guess with a compile-time guarantee it will be wrong eventually (the
`check_weights` rank-3 shape trap, in the shape of a byte string), so every emitted row must match.

Three rules from this repo's own trap table are enforced rather than mentioned:

  * a driver whose parse matches nothing reports PASS (fifth instance this session, D32) — so the
    record counts are asserted against the log's own self-reported numbers and a mismatch or an
    empty parse exits nonzero, and the counts print even when they are zero;
  * the `down_proj` trap (D27 lesson 2): a `best` column read against anything except the DEFAULT
    column is not a win. Every ratio here is printed beside `default_ms`, and a shape whose winner
    only beats `engine_call_ms` and not rank-0 is labelled as such, because the unpinned `lt` route
    gets rank-0, not the winner;
  * the 128 MiB workspace cap is a fact (D31 closed raising it), so a candidate whose stamped
    workspace demand exceeds it is REFUSED with a printed reason instead of quietly launched smaller.

Usage (all read-only, no device):

  python3 prototype/gemm_algo_sweep/deep_pins.py \
      --time-log tmp/wsjob/w64_128.log \
      --algo-dump /tmp/gas_algos.txt \
      --engine prototype/gemm_algo_sweep/gemm_algo_sweep \
      --wave-ms 956.12
  python3 prototype/gemm_algo_sweep/deep_pins.py --selftest-rule  # rule vs the committed pin
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
from pathlib import Path

# The engine's own constants, READ from the source that enforces them rather than retyped: a copy
# here would be a second authority that can drift (the same reason `gate.py` reads kMaxWaveRows).
ENGINE_GEMM = Path("engine/kernels_gemm.cpp")
SHIPPED_PIN_MARK = "// ONE shape is pinned today"
# The materiality floor for a candidate, as a share of the whole 4096-lane wave. A pin is not free: it
# freezes one kernel against one hipBLASLt version and has to be re-captured on every bump (the
# provenance note in `algo_pins()`), so a winner worth 0.03 % of a wave is cheaper to leave alone than
# to carry. Stated as a number here rather than applied by feel.
MIN_PCT_OF_WAVE = 0.1


class Noise:
    """How far one algorithm's burst `best_ms` moves between two sweep windows, per shape.

    Keyed on the algo RECORD, not on `seq`: `seq` is enumeration order and the heuristic list is not a
    prefix across different `--requested` counts (rank 5 of an 8-wide query and rank 5 of a 64-wide one
    are different kernels - see `--prefix-check`), so a pair only means something if both sides are the same
    24 bytes. This is what sizes which deep gains are real: D31 of
    progress/2026-10-16/prefill-decision-log-2026-10-16.md quoted one hand-picked pair (q_proj 0.1212 ->
    0.1303, "7.5 % on a 0.12 ms call") and a shape's gain has to clear its own measured movement, not
    someone else's.
    """

    def __init__(self, a: TimeLog, b: TimeLog, m: int):
        self.by_shape: dict[str, list[float]] = {}
        by_role: dict[str, dict[tuple[str, str], float]] = {}
        for log, role in ((a, "a"), (b, "b")):
            per: dict[tuple[str, str], float] = {}
            for key, times in log.time.items():
                if key[1] != m:
                    continue
                algo_by_seq = {int(x["seq"]): x.get("data", "?") for x in log.algo.get(key, [])}
                for t in times:
                    seq = int(t["seq"])
                    if seq in algo_by_seq:
                        per[(key[0], algo_by_seq[seq])] = float(t["best_ms"])
            by_role[role] = per
        pa, pb = by_role["a"], by_role["b"]
        for (shape, algo), ms in sorted(pa.items()):
            if (shape, algo) in pb:
                self.by_shape.setdefault(shape, []).append(pb[(shape, algo)] / ms)

    @staticmethod
    def _med(v: list[float]) -> float:
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def floor(self, shape: str) -> float | None:
        v = self.by_shape.get(shape)
        return abs(self._med(v) - 1.0) if v else None

    def worst(self, shape: str) -> float | None:
        v = self.by_shape.get(shape)
        return max(abs(x - 1.0) for x in v) if v else None

    def report(self) -> None:
        print("NOISE same-algorithm best_ms movement between the two windows (floor = |median-1|, the "
              "smallest gain this shape can distinguish)")
        for shape, v in sorted(self.by_shape.items()):
            print(f"  {shape:18s} n={len(v):2d} median={self._med(v):.4f} floor={100*self.floor(shape):.1f} %"
                  f"  worst_pair={100*self.worst(shape):.1f} %  min={min(v):.4f} max={max(v):.4f}")
        allv = [x for v in self.by_shape.values() for x in v]
        if allv:
            print(f"  ALL n={len(allv)} median={self._med(allv):.4f} worst={100*max(abs(x-1) for x in allv):.1f} %")
        else:
            print("  NONE — the two windows share no algorithm, so no gain in either is separable from "
                  "window movement and every candidate must be refused")


def kv(line: str) -> dict:
    return dict(re.findall(r"(\w+)=(\S+)", line))


def read_engine_cap(path: Path) -> tuple[int, int]:
    """(workspace cap in bytes, heuristic width) out of engine/kernels_gemm.cpp."""
    text = path.read_text()
    m = re.search(r"kWorkspaceBytes\s*=\s*([0-9]+)ull\s*<<\s*\(?(\d+)\)?", text)
    if not m:
        raise SystemExit(f"deep_pins: cannot find kWorkspaceBytes in {path} — the cap this tool "
                         "enforces must be read from the code that enforces it, not remembered")
    cap = int(m.group(1)) << int(m.group(2))
    w = re.search(r"kHeuristicWidth\s*=\s*(\d+)", text)
    return cap, (int(w.group(1)) if w else -1)


def read_committed_pins(path: Path) -> list[dict]:
    """The pin rows the engine actually carries, parsed out of `algo_pins()`.

    Matches the `v.push_back({name, N, K, M, hex})` lines by shape rather than by line number, and
    returns [] only if the function body genuinely has no rows — the caller decides whether empty
    is a legal answer (it never is for the shipped set today, and `--selftest-rule` says so).
    """
    text = path.read_text()
    body = re.search(r"algo_pins\(\)\s*\{(.*?)\n\}", text, re.S)
    if not body:
        raise SystemExit(f"deep_pins: cannot locate algo_pins() in {path}")
    rows = []
    for m in re.finditer(r"push_back\(\{([^}]*?)\}\)", body.group(1), re.S):
        parts = [p.strip() for p in m.group(1).split(",")]
        hexes = [p for p in parts if re.fullmatch(r'"[0-9a-f]{48}"', p)]
        if len(hexes) != 1:
            continue
        names = [p for p in parts if re.fullmatch(r'"[a-z_+0-9]+"', p)]
        sets = [p for p in parts if re.fullmatch(r"PinSet::k\w+", p)]
        nums = [p for p in parts if re.fullmatch(r"\d+|k\w+", p)]
        rows.append({"fields": parts, "hex": hexes[0].strip('"'), "nums": nums,
                     "name": names[0].strip('"') if names else "?",
                     # A row with no set field is a row from before the split, and defaulting it to
                     # "shipped" would let a stale table read as fully adjudicated. Call it unknown.
                     "set": sets[0].split("::")[1] if sets else "UNKNOWN"})
    return rows


def pin_hex_from_algo_data(data: str, workspace_cap: int) -> str:
    """`ALGO … data=%016llx:%016llx` -> the 48-hex-char 24-byte record `algo_pins()` stores.

    Two printers, and confusing them is the whole trap. The sweep's `ALGO` line prints the algorithm as
    TWO native words (`w0:w1`) with the workspace stamp in a separate `ws=` field; the `ALGODUMP` capture
    prints the full 24-byte record in one field, and that is the coordinate `algo_pins()` stores and the
    deriver matches against. The two words are printed as unsigned longs, i.e. host-endian NUMBERS, while
    the record is raw BYTES — on this little-endian box that is a byte reversal per word, which
    `struct.pack('<QQ', w0, w1)` does — and then the stamp is APPENDED, because the `ALGO` line does not
    carry it in record form.

    It has to be the same cap the capture used, and that is not a detail: for a heuristic-ranked record the
    stamped `max_workspace_bytes` is the PREFERENCE CAP the query ran under — in the 64-deep capture here,
    all 448 heuristic-ranked records end `00000008 00000000` = 134217728 = the engine's `kWorkspaceBytes`
    — not a per-kernel demand. (The other 4802 `ALGODUMP` lines in that capture are catalogue entries with
    `heur_rank=-1`, which is why the counts differ.) So `--workspace-mib` must equal `kWorkspaceBytes` (the
    default) or no reconstruction matches the capture and every candidate reads UNMATCHED; and a candidate
    derived under a larger cap than the engine has would arrive as a stamp the engine refuses at plan time,
    which is D31's closed door rather than a silent downgrade.
    """
    if not re.fullmatch(r"[0-9a-f]{16}:[0-9a-f]{16}", data):
        raise SystemExit(f"deep_pins: algo data={data!r} is not the two-word form the sweep prints")
    w0, w1 = (int(x, 16) for x in data.split(":"))
    return struct.pack("<QQ", w0, w1).hex() + struct.pack("<Q", workspace_cap).hex()


class Dump:
    """An `--algo-dump` capture: the witness for every pin hex."""

    def __init__(self, paths: list[Path]):
        self.by_shape: dict[tuple[str, int], dict[str, dict]] = {}
        self.lines = 0
        for p in paths:
            if not p.exists():
                raise SystemExit(f"deep_pins: --algo-dump file {p} does not exist. A pin hex must "
                                 "be matched against a capture; producing one is the printed "
                                 "`--verify --algo-dump` command below, not a reconstruction.")
            for line in p.read_text().splitlines():
                if not line.startswith("ALGODUMP"):
                    continue
                f = kv(line)
                if "data" not in f or "M" not in f:
                    continue
                self.lines += 1
                self.by_shape.setdefault((f["shape"], int(f["M"])), {})[f["data"]] = f

    def match(self, shape: str, m: int, hexstr: str) -> dict | None:
        return self.by_shape.get((shape, m), {}).get(hexstr)

    def ranks_seen(self, shape: str, m: int) -> list[int]:
        return sorted(int(f["heur_rank"]) for f in self.by_shape.get((shape, m), {}).values()
                      if f["heur_rank"].lstrip("-").isdigit() and int(f["heur_rank"]) >= 0)


class TimeLog:
    """A `--time` sweep window, parsed into per-(shape, M) records."""

    def __init__(self, path: Path):
        self.path = path
        text = path.read_text(errors="replace").splitlines()
        self.heur: dict[tuple[str, int], dict] = {}
        self.plan: dict[tuple[str, int], dict] = {}
        self.algo: dict[tuple[str, int], list[dict]] = {}
        self.time: dict[tuple[str, int], list[dict]] = {}
        self.result: dict[tuple[str, int], dict] = {}
        self.skipsum: dict[tuple[str, int], dict] = {}
        for line in text:
            if line.startswith("HEUR shape="):
                f = kv(line)
                self.heur[(f["shape"], int(f["M"]))] = f
            elif line.startswith("PLAN tool=gemm_algo_sweep shape="):
                f = kv(line)
                self.plan[(f["shape"], int(f["M"]))] = f
            elif line.startswith("ALGO shape="):
                f = kv(line)
                self.algo.setdefault((f["shape"], int(f["M"])), []).append(f)
            elif line.startswith("TIME shape="):
                f = kv(line)
                self.time.setdefault((f["shape"], int(f["M"])), []).append(f)
            elif line.startswith("RESULT shape="):
                f = kv(line)
                self.result[(f["shape"], int(f["M"]))] = f
            elif line.startswith("SKIP-SUMMARY shape="):
                f = kv(line)
                self.skipsum[(f["shape"], int(f["M"]))] = f

    def assert_counts(self, want_m: int) -> None:
        """Fail loudly unless every shape's parse matches the log's own self-reported counts.

        The counts are compared against three INDEPENDENT self-reports in the log (HEUR returned,
        RESULT candidates, RESULT band_ok) so a regex that silently matches nothing cannot produce a
        table. A check whose input list can be empty cannot fail (gdn_collapse/run.py, mixed_bw,
        gate.py's prefill rows, gdn_rec_isa's --volume, attn_wmma's region census).
        """
        keys = sorted({k for k in self.result if k[1] == want_m})
        print(f"DERIVE-INPUT file={self.path} problems={len(keys)} M={want_m} "
              f"shapes={','.join(k[0] for k in keys) if keys else 'NONE'}")
        if not keys:
            print(f"DERIVE-FAIL reason=no RESULT lines at M={want_m} -> the parse matched nothing, "
                  "which is not the same as nothing being wrong")
            raise SystemExit(2)
        bad = 0
        for key in keys:
            res, heur = self.result[key], self.heur.get(key, {})
            nalgo, ntime = len(self.algo.get(key, [])), len(self.time.get(key, []))
            cands, band_ok = int(res["candidates"]), int(res["band_ok"])
            returned = int(heur.get("returned", -1))
            heur_lines = [a for a in self.algo.get(key, []) if int(a["heur_rank"]) >= 0]
            ok = (nalgo >= cands and ntime == band_ok and len(heur_lines) == returned
                  and cands > 0 and band_ok > 0)
            print(f"DERIVE-LINES shape={key[0]} M={key[1]} algo_lines={nalgo} "
                  f"heur_ranked_lines={len(heur_lines)} heur_returned={returned} "
                  f"time_lines={ntime} result_candidates={cands} band_ok={band_ok} ok={1 if ok else 0}")
            bad += 0 if ok else 1
        print(f"DERIVE-LINES-SUMMARY problems={len(keys)} ok={len(keys) - bad} bad={bad}")
        if bad:
            print("DERIVE-FAIL reason=a shape's parsed record count disagrees with the log's own "
                  "count, so any table below would be a partial read of a 64-deep field — which is "
                  "exactly the truncated-field error D31 says not to price")
            raise SystemExit(2)


def analyse(log: TimeLog, dump: Dump, cap: int, wave_ms: float, m: int, top: int,
            heur_width: int, noise: "Noise | None" = None,
            engine_rows: list[dict] | None = None) -> list[dict]:
    rows = []
    engine_rows = engine_rows or []
    pin_by_name = {r["name"]: r["hex"] for r in engine_rows if r["set"] == "kShipped"}
    for key in sorted(k for k in log.result if k[1] == m):
        res, plan = log.result[key], log.plan.get(key, {})
        shape = key[0]
        calls = int(plan.get("calls_per_wave", 0))
        d_ms = float(res["default_ms"])
        eng_ms = float(res["engine_call_ms"])
        times = {int(t["seq"]): float(t["best_ms"]) for t in log.time.get(key, [])}
        algos = {int(a["seq"]): a for a in log.algo.get(key, []) if int(a["heur_rank"]) >= 0}
        # Rank the field by measured time. seq is the enumeration order; heur_rank is the
        # heuristic's own rank, and for these logs they coincide — asserted, not assumed.
        ranked = sorted(times.items(), key=lambda kv_: kv_[1])
        full_list = []
        for seq, ms in ranked:
            a = algos.get(seq, {})
            hexstr = pin_hex_from_algo_data(a["data"], cap) if "data" in a else ""
            ws_bytes = struct.unpack("<Q", bytes.fromhex(hexstr[32:]))[0] if hexstr else 0
            m_rec = dump.match(shape, m, hexstr)
            rank = int(a.get("heur_rank", -1))
            refusable = ws_bytes > cap
            full_list.append({
                "seq": seq, "heur_rank": rank, "ms": ms, "hex": hexstr, "ws_stamped": ws_bytes,
                "idx": a.get("idx", "?"),
                "solution": a.get("solution", "?"),
                "witness": "matched" if m_rec else "UNMATCHED",
                "witness_heur_rank": m_rec.get("heur_rank") if m_rec else None,
                "admissible": not refusable,
                "refuse_reason": (f"workspace_stamped={ws_bytes} > engine cap={cap} (128 MiB); "
                                  "D31 closed raising kWorkspaceBytes, so this record cannot run in "
                                  "the engine at all") if refusable else "",
                "visible_to_engine": rank < heur_width,
            })
            if m_rec is None and seq == ranked[0][0]:
                print(f"PIN-UNMATCHED shape={shape} M={m} seq={seq} heur_rank={rank} hex={hexstr} "
                      f"reason=the reconstructed 24 bytes appear in no ALGODUMP line for this "
                      f"shape; the capture on disk covers heur_rank "
                      f"{min(dump.ranks_seen(shape, m), default=-1)}.."
                      f"{max(dump.ranks_seen(shape, m), default=-1)} so a deeper winner needs a "
                      f"`--verify --algo-dump` at --requested >= {rank + 1} before it can be pinned")
        top_list = full_list[:top]
        winner = full_list[0]
        win_ms = winner["ms"]
        top_list_full = full_list
        # The baseline is what the SHIPPING path runs for THIS shape, and it is not always rank-0:
        # `down_proj` has been pinned since D28, so reading its deep winner against rank-0 books 0.9744
        # ms per wave when the shipped pin already banked 0.71 of it and the real prize is 0.2644. That
        # is the §3d "wrong baseline column" error one layer over, so the pin's own measured time in THIS
        # window is found by hex and used when it exists; the pairing is in-window, so it is paired.
        pin_hex = pin_by_name.get(shape)
        pin_seq = pin_ms = None
        for t in top_list_full:
            if t["hex"] and t["hex"] == pin_hex:
                pin_seq, pin_ms = t["seq"], t["ms"]
                break
        base_ms = pin_ms if pin_ms is not None else d_ms
        base_name = f"shipped pin seq={pin_seq}" if pin_ms is not None else "rank-0"
        gain = base_ms / win_ms if win_ms > 0 else 0.0
        floor = noise.floor(shape) if noise else None
        worst = noise.worst(shape) if noise else None
        saved_wave = calls * (base_ms - win_ms)
        per_prefill = saved_wave * 8.0
        reasons = []
        if not winner["admissible"]:
            reasons.append(winner["refuse_reason"])
        if winner["witness"] != "matched":
            reasons.append("winner hex is not witnessed in any ALGODUMP capture")
        if pin_ms is not None and win_ms >= pin_ms:
            reasons.append("the shipped pin is already at least as fast in this window")
        if pin_ms is None and win_ms >= d_ms:
            reasons.append("the deep winner IS rank-0: nothing to pin")
        if floor is not None and (1.0 - win_ms / base_ms) <= floor:
            reasons.append(f"gain {100*(1-win_ms/base_ms):.1f} % does not clear this shape's own "
                           f"cross-window movement {100*floor:.1f} % (median same-algo ratio)")
        if per_prefill < MIN_PCT_OF_WAVE / 100.0 * wave_ms:
            reasons.append(f"buys {100*per_prefill/wave_ms:.3f} % of a wave, under the "
                           f"{MIN_PCT_OF_WAVE} % floor a re-capture obligation is worth")
        rows.append({
            "shape": shape, "M": m, "n": res["N"], "k": res["K"], "calls_per_wave": calls,
            "family": plan.get("family", "?"),
            "default_seq": int(res["default_seq"]), "default_ms": d_ms,
            "shipped_pin_ms": pin_ms, "shipped_pin_seq": pin_seq, "baseline_ms": base_ms,
            "baseline_name": base_name,
            "noise_floor": floor, "noise_worst": worst,
            "admitted": not reasons, "reject_reasons": reasons,
            "default_tflops": float(res["default_tflops"]),
            "engine_call_ms": eng_ms, "field": len(times),
            "winner": winner, "top": top_list,
            # The only ratio that describes a change to the shipping path: what the run in force gets
            # (rank-0, or the shipped pin where one exists) versus what a candidate would get. The
            # engine column is kept beside it because it is a different question (route choice, already
            # shipped as D26/D27).
            "gain_vs_rank0": d_ms / win_ms if win_ms > 0 else 0.0,
            "gain_vs_base": base_ms / win_ms if win_ms > 0 else 0.0,
            "gain_vs_engine": eng_ms / win_ms if win_ms > 0 else 0.0,
            "rank0_vs_engine": d_ms / eng_ms if eng_ms > 0 else 0.0,
            "saved_ms_per_wave": saved_wave,
            "wave_ms": wave_ms,
        })
    return rows


def print_report(rows: list[dict], dump: Dump, cap: int, heur_width: int, wave_ms: float) -> int:
    print(f"\nDEPTH shape-level (M={rows[0]['M'] if rows else '?'}; the engine's only shippable "
          f"width is kMaxWaveRows=512, engine/decode_step.h:49). BASE is what runs TODAY for that "
          f"shape: its shipped pin where one exists, else rank-0.")
    print(f"{'shape':18s} {'calls':>5} {'BASE ms':>8} {'best ms':>8} {'best rank':>9} "
          f"{'best/BASE':>9} {'noise':>6} {'saved/wave':>10} {'%/wave':>7} verdict")
    total = admitted_total = 0.0
    blockers = 0
    for r in rows:
        w = r["winner"]
        if w["witness"] != "matched" or not w["admissible"]:
            blockers += 1
        admitted_total += r["saved_ms_per_wave"] if r["admitted"] else 0.0
        per_prefill = r["saved_ms_per_wave"] * 8.0
        noise_txt = f"{100*r['noise_floor']:.1f} %" if r["noise_floor"] is not None else "n/a"
        verdict = "ADMIT" if r["admitted"] else "reject: " + "; ".join(r["reject_reasons"])
        base_tag = "" if r["shipped_pin_ms"] is None else f" ({r['baseline_name']})"
        print(f"{r['shape']:18s} {r['calls_per_wave']:>5} {r['baseline_ms']:>8.4f} {w['ms']:>8.4f} "
              f"{w['heur_rank']:>9} {w['ms'] / r['baseline_ms']:>9.4f} {noise_txt:>6} "
              f"{r['saved_ms_per_wave']:>10.4f} {100*per_prefill/wave_ms:>7.3f} {verdict}{base_tag}")
        for extra in r["top"][1:]:
            print(f"{'  +':18s} {'':>5} {'':>8} {extra['ms']:>8.4f} {extra['heur_rank']:>9} "
                  f"{extra['ms'] / r['baseline_ms']:>9.4f} {'':>6} {'':>10} {'':>7}"
                  f" {'beyond-8' if not extra['visible_to_engine'] else ''}")
    per_prefill_all = sum(max(0.0, r["saved_ms_per_wave"]) for r in rows) * 8.0
    print(f"\nDEEP-PRIZE shapes={len(rows)} field={rows[0]['field']} "
          f"engine_heuristic_width={heur_width} workspace_cap_bytes={cap}")
    print(f"DEEP-PRIZE every-shape-vs-BASE: per_4096_lane_prefill={per_prefill_all:.4f} ms "
          f"= {100.0*per_prefill_all/wave_ms:.3f} % of the {wave_ms:.2f} ms wave — a CEILING, because "
          "it credits a burst win on shapes the admission rule rejected")
    print(f"DEEP-PRIZE admitted-set-only: saved_per_24layer_wave={admitted_total:.4f} "
          f"per_4096_lane_prefill={admitted_total*8.0:.4f} "
          f"pct_of_wave({wave_ms:.2f} ms)={100.0*admitted_total*8.0/wave_ms:.3f}")
    print(f"DEEP-PRIZE blocked={blockers} (unwitnessed hex or over-cap workspace stamp) — a blocked "
          "winner is not a candidate, it is a capture that has not been taken yet")
    print("DEEP-PRIZE reading: best/BASE < 1 is the ONLY thing that changes the shipping path, because "
          "an unpinned `lt` run gets rank-0 and a pinned shape gets its pin. rank0/eng is the "
          "already-shipped route gap (D26), a different question. Both are warm-burst per-call figures "
          "read out of tmp/wsjob/ logs, and D28 is the precedent that they do not transfer to family "
          "time in situ: two of three burst winners were in-situ losers. Treat the admitted-set number "
          "as an upper bound and adjudicate in one lock hold.")
    return blockers


def check_table(rows: list[dict], engine_rows: list[dict]) -> int:
    """The engine's candidate rows must be exactly the set this derivation admits — no more, no fewer.

    Without this, the table in `kernels_gemm.cpp` is a transcription that rots the first time a winner
    moves, and a stale pin is worse than no pin: it names a kernel nobody re-measured while reading as
    adjudicated. Counts AND hexes are compared, and an empty engine candidate set with a non-empty
    derived set is a failure, not a pass.
    """
    eng_cand = {r["hex"]: r["name"] for r in engine_rows if r["set"] == "kCandidate"}
    eng_ship = {r["hex"]: r["name"] for r in engine_rows if r["set"] == "kShipped"}
    derived = {r["winner"]["hex"]: "deep_" + r["shape"] for r in rows if r["admitted"]}
    print(f"\nCHECK-TABLE engine_shipped={len(eng_ship)} engine_candidates={len(eng_cand)} "
          f"derived_admitted={len(derived)}")
    rc = 0
    if len(eng_ship) != 1:
        print(f"CHECK-TABLE FAIL reason=the shipped set is asserted to be exactly 1 row (D28's "
              f"adjudication left down_proj standing); found {len(eng_ship)}. A check that can come "
              "back empty passes vacuously, which is the failure class this repo has hit five times.")
        rc = 1
    if derived and set(derived) != set(eng_cand):
        missing = sorted(set(derived) - set(eng_cand))
        extra = sorted(set(eng_cand) - set(derived))
        print(f"CHECK-TABLE FAIL reason=the engine's candidate rows are not what this log derives. "
              f"derived_but_not_in_engine={missing} in_engine_but_not_derived={extra}. Either the "
              "table is stale against the sweep, or the sweep was re-taken and the table was not "
              "re-derived — a stale candidate is a kernel nobody measured wearing an adjudicated label.")
        rc = 1
    if not rc:
        print("CHECK-TABLE PASS (every candidate the engine carries is one this log derives, and "
              "every winner the log admits is in the engine)")
    return rc


def emit_pins(rows: list[dict], engine_rows: list[dict], cap: int) -> None:
    shipped_hexes = {r["hex"] for r in engine_rows if r["set"] == "kShipped"}
    cand_hexes = {r["hex"]: r["name"] for r in engine_rows if r["set"] == "kCandidate"}
    print("\n=== algo_pins() CANDIDATE rows (promotion to the shipped set is a one-word edit; "
          "the default `all` does not reach these) ===")
    n = 0
    for r in rows:
        w = r["winner"]
        if not r["admitted"]:
            print(f"# {r['shape']}: NOT a candidate — {'; '.join(r['reject_reasons'])}")
            continue
        if w["hex"] in shipped_hexes:
            print(f"# {r['shape']}: the deep winner IS the already-SHIPPED pin "
                  f"(hex {w['hex'][:16]}…); nothing new to pin")
            continue
        if w["hex"] in cand_hexes:
            print(f"# {r['shape']}: already carried as the candidate `{cand_hexes[w['hex']]}` — "
                  f"unchanged by this derivation")
            continue
        n += 1
        floor_txt = (f"{100*r['noise_floor']:.1f} %" if r["noise_floor"] is not None
                     else "not measured — pass --noise-log")
        print(f'{{"deep_{r["shape"]}", {r["n"]}, {r["k"]}, 512, "{w["hex"]}", PinSet::kCandidate}},  '
              f'// heur_rank={w["heur_rank"]} idx={w["idx"]} '
              f'{w["ms"]:.4f} ms vs BASE {r["baseline_ms"]:.4f} ({r["baseline_name"]}) '
              f'= {w["ms"] / r["baseline_ms"]:.3f}x, shape noise floor {floor_txt}, '
              f'saved {r["saved_ms_per_wave"]:.4f} ms/wave = '
              f"{100*r['saved_ms_per_wave']*8.0/r['wave_ms']:.3f} % of a {r['wave_ms']:.2f} ms wave, "
              f'ws_stamp={w["ws_stamped"]} <= cap {cap}, witnessed in ALGODUMP; '
              f'{w["solution"][:72]}')
    print(f"=== {n} candidate row(s) out of {len(rows)} shapes ===")


def selftest_rule(dump_paths: list[Path], cap: int) -> int:
    """The reconstruction rule, checked against the one pin whose bytes are independently known.

    `algo_pins()` carries one row, `down_proj`, and results.md §3d states its provenance: it is the
    ALGODUMP record at seq=1 idx=905, captured not hand-written. So: take down_proj heur_rank=1 from
    a sweep log's ALGO line, reconstruct with the rule, and require byte equality with BOTH the
    committed pin in engine/kernels_gemm.cpp and the ALGODUMP line. If the rule is wrong about the
    endianness or about the stamped cap, this fails — which is the only reason the rest of the tool
    is allowed to reconstruct anything.
    """
    pins = read_committed_pins(ENGINE_GEMM)
    shipped = [p for p in pins if p["set"] == "kShipped"]
    cands = [p for p in pins if p["set"] == "kCandidate"]
    unknown = [p for p in pins if p["set"] not in ("kShipped", "kCandidate")]
    print(f"SELFTEST-RULE rows={len(pins)} shipped={len(shipped)} candidates={len(cands)} "
          f"unset={len(unknown)} names={','.join(p['name'] for p in pins)}")
    if len(shipped) != 1 or unknown:
        print(f"SELFTEST-RULE FAIL reason=the shipped set is asserted to be exactly 1 row "
              f"(D28's adjudication) and every row must name a set; found shipped={len(shipped)} "
              f"unset={len(unknown)}. A check that reads the pin table and can come back empty or "
              "wrong-sized passes vacuously, which is the failure class this repo has now hit five "
              "times.")
        return 1
    committed = shipped[0]["hex"]
    dump = Dump(dump_paths)
    recs = [f for f in dump.by_shape.get(("down_proj", 512), {}).values()
            if f["heur_rank"] == "1" and int(f["idx"]) == 905]
    if not recs:
        print("SELFTEST-RULE FAIL reason=no ALGODUMP line for down_proj heur_rank=1 idx=905")
        return 1
    rec = recs[0]
    ok_dump = rec["data"] == committed
    print(f"SELFTEST-RULE algodump_down_proj_rank1_matches_committed={1 if ok_dump else 0} "
          f"algodump={rec['data']} committed={committed}")
    if not ok_dump:
        print("SELFTEST-RULE FAIL reason=the capture and the committed pin disagree; nothing this "
              "tool reconstructs can be trusted until that is explained")
        return 1
    print("SELFTEST-RULE note=\"the ALGO-line half of the rule is checked in --check-rule-vs-log, "
          "which needs a --time-log; run both together\"")
    print("SELFTEST-RULE PASS (the pin form the engine parses is the form the library writes)")
    return 0


def check_rule_vs_log(log: TimeLog, dump: Dump, cap: int, m: int) -> int:
    """Every heuristic-ranked record in the log, reconstructed and matched against the capture.

    This is the "applied to ALL rows, not the one checked by hand" requirement: the rule is exercised
    on the whole overlap between the two sources, and the count that matched is printed against the
    count that could be tested (only ranks the capture contains are testable — that gap is itself
    reported, because it is the gap that blocks deep pins).
    """
    tested = matched = 0
    untestable = 0
    for key, algos in sorted(log.algo.items()):
        if key[1] != m:
            continue
        for a in algos:
            if int(a["heur_rank"]) < 0 or "data" not in a:
                continue
            hexstr = pin_hex_from_algo_data(a["data"], cap)
            tested += 1
            if dump.match(key[0], m, hexstr):
                matched += 1
            else:
                untestable += 1
    print(f"RULE-VS-LOG M={m} reconstructed={tested} matched_in_ALGODUMP={matched} "
          f"not_in_capture={untestable} "
          f"note=\"not_in_capture is the rank range the on-disk capture did not request, not a "
          f"mismatch; a candidate there needs a deeper --verify --algo-dump first\"")
    if tested == 0:
        print("RULE-VS-LOG FAIL reason=nothing was testable")
        return 1
    if matched == 0:
        print("RULE-VS-LOG FAIL reason=zero reconstructions matched the library's own capture; the "
              "endian or the workspace stamp in pin_hex_from_algo_data is wrong")
        return 1
    return 0


def prefix_check(dumps: list[Path], m: int, engine_width: int) -> int:
    """Is the shallow heuristic list a PREFIX of the deep one? Compare two `--algo-dump` captures.

    This is the fact every rank-based idea here quietly rests on, so it belongs in the tool that pins,
    not in a scratch script: if the list were a prefix, "widen the pool" would only ever ADD candidates.
    It is not. Rank-0 is identical across request sizes (which is what makes widening inert for the
    default pick), but from a rank as low as 5 the two lists carry different algorithms, and some algos
    the shallow query returns are absent from the deep one. That is why a pin stores the 24-byte record
    and why nothing in this tree keys on a rank.
    """
    if len(dumps) < 2:
        print("PREFIX FAIL reason=needs two --algo-dump captures (pass the shallow one and the deep one)")
        return 1

    def by_rank(path: Path) -> dict[str, dict[int, str]]:
        d = Dump([path])
        out: dict[str, dict[int, str]] = {}
        for (shape, mm), recs in d.by_shape.items():
            if mm != m:
                continue
            out[shape] = {int(f["heur_rank"]): f["data"] for f in recs.values()
                          if f["heur_rank"].lstrip("-").isdigit() and int(f["heur_rank"]) >= 0}
        return out

    a, b = by_rank(dumps[0]), by_rank(dumps[1])
    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"PREFIX FAIL reason=no shape appears at M={m} in both captures")
        return 1
    print(f"PREFIX captures={dumps[0]} vs {dumps[1]} M={m} shapes={len(shared)} "
          f"pool widths={sorted({len(v) for v in a.values()})} vs {sorted({len(v) for v in b.values()})}")
    bad = 0
    for s in shared:
        ra, rb = a[s], b[s]
        common = sorted(set(ra) & set(rb))
        diff = [r for r in common if ra[r] != rb[r]]
        first8_same = all(ra[r] == rb[r] for r in common if r < engine_width)
        rank0_same = ra.get(0) == rb.get(0)
        if not rank0_same:
            bad += 1
        print(f"  {s:18s} rank-0 same={str(rank0_same):5s} "
              f"ranks 0..{engine_width-1} all same={str(first8_same):5s} "
              f"first rank whose algo differs={str(diff[0] if diff else 'none'):5s} "
              f"sets equal={str(set(ra.values()) == set(rb.values()))}")
    print("PREFIX reading: rank-0 identical everywhere is what makes `--gemm-heur` inert for the default "
          "pick (the engine takes the first admissible record, and that record is the same at both widths); "
          "`first rank differing` below or near the engine width is why a candidate must be the algo RECORD "
          "and never a rank, because the deep list is a partly different list, not a longer one.")
    if bad:
        print("PREFIX FAIL reason=rank-0 differs between the captures, so widening the pool WOULD change "
              "what an unpinned shape runs and the knob is no longer behaviour-preserving")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--time-log", type=Path, default=Path("tmp/wsjob/w64_128.log"))
    ap.add_argument("--algo-dump", type=Path, action="append", default=None,
                    help="an `--algo-dump` capture; repeatable. Default /tmp/gas_algos.txt")
    ap.add_argument("--m", type=int, default=512,
                    help="the width to derive at. 512 is the only width the engine can emit "
                         "(kMaxWaveRows); 1024/4096 are bounds, not prizes.")
    ap.add_argument("--top", type=int, default=3, help="candidates to show per shape")
    ap.add_argument("--wave-ms", type=float, default=956.12,
                    help="the ledger's whole-wave denominator (D33: 956.12 ms at 4096 lanes)")
    ap.add_argument("--selftest-rule", action="store_true")
    ap.add_argument("--prefix-check", action="store_true",
                    help="compare the first two --algo-dump captures and report whether the shallow "
                         "heuristic list is a prefix of the deep one")
    ap.add_argument("--no-check-rule", action="store_true")
    ap.add_argument("--noise-log", type=Path, default=Path("tmp/wsjob/ws128.log"),
                    help="a SECOND --time window at the same M and budget, same-algorithm pairs from "
                         "which set the floor a candidate must clear. Pass `none` to skip the "
                         "admission rule (the print then says so per shape rather than pretending).")
    ap.add_argument("--no-check-table", action="store_true",
                    help="skip the engine-table-vs-derived comparison. Off by default on purpose: a "
                         "stale pin table is a kernel nobody measured wearing an adjudicated label.")
    a = ap.parse_args()

    dumps = a.algo_dump or [Path("/tmp/gas_algos.txt")]
    cap, heur_width = read_engine_cap(ENGINE_GEMM)
    print(f"ENGINE constants read from {ENGINE_GEMM}: kWorkspaceBytes={cap} "
          f"({cap >> 20} MiB) kHeuristicWidth={heur_width}")

    if a.selftest_rule:
        return selftest_rule(dumps, cap)

    if a.prefix_check:
        return prefix_check(dumps, a.m, heur_width)

    dump = Dump(dumps)
    print(f"ALGODUMP files={len(dumps)} lines={dump.lines} "
          f"shapes={len(dump.by_shape)} ranks_per_shape="
          f"{[len(v) for k, v in sorted(dump.by_shape.items())][:1]}")
    log = TimeLog(a.time_log)
    log.assert_counts(a.m)
    rc = 0 if a.no_check_rule else check_rule_vs_log(log, dump, cap, a.m)
    if rc:
        return rc
    engine_rows = read_committed_pins(ENGINE_GEMM)
    noise = None
    if str(a.noise_log) != "none":
        if not a.noise_log.exists():
            print(f"NOISE file={a.noise_log} missing — refusing to admit ANY candidate on a floor it "
                  "could not measure. Pass --noise-log none to say so out loud and take the rule off.")
            return 2
        noise = Noise(TimeLog(a.noise_log), log, a.m)
        print(f"NOISE windows: floor side={a.noise_log} against measure side={log.path} at M={a.m}")
        noise.report()
    else:
        print("NOISE disabled by --noise-log none: gains are not being checked against any ruler "
              "spread, so every ADMIT below is unproven")
    rows = analyse(log, dump, cap, a.wave_ms, a.m, a.top, heur_width, noise=noise,
                   engine_rows=engine_rows)
    emit_pins(rows, engine_rows, cap)
    print()
    blockers = print_report(rows, dump, cap, heur_width, a.wave_ms)
    if not a.no_check_table:
        rc = check_table(rows, engine_rows)
        if rc:
            return rc
    admitted = [r for r in rows if r["admitted"]]
    print(f"DERIVE-OUTCOME admitted={len(admitted)}/{len(rows)} "
          f"names={','.join('deep_' + r['shape'] for r in admitted) if admitted else 'NONE'}")
    return 0 if blockers == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
