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
    workspace demand exceeds it is REFUSED with a printed reason instead of quietly launched smaller;
  * the burst window cannot see inside a wave, and what it cannot see has twice been wrong (D35). So the
    in-situ adjudication lives in `insitu_verdicts.tsv` as data, keyed by (shape, algo hex) — shape as
    well as hex because one kernel record can rank in two shapes' fields — and `CHECK-TABLE` requires
    shipped == (pinnable ∩ in-situ wins), candidates == ((pinnable − shipped) ∪ retired pins), and NO
    pinnable shape left unverdicted. There is deliberately no shape allow-list in this file: what may ship
    is read from the verdict rows, so a re-take changes the answer by adding a measurement, not by
    editing a rule.

Usage (all read-only, no device):

  python3 prototype/gemm_algo_sweep/deep_pins.py \
      --time-log tmp/wsjob/w64_128.log \
      --algo-dump /tmp/gas_algos.txt \
      --engine prototype/gemm_algo_sweep/gemm_algo_sweep \
      --wave-ms 956.12
  python3 prototype/gemm_algo_sweep/deep_pins.py --selftest-rule    # rule + witness + verdicts, no window
  python3 prototype/gemm_algo_sweep/deep_pins.py --selftest-checks  # tamper the data, require each guard red
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
# The whole vocabulary a verdict row may use, and `tie` is in it because D36 needed it: the two
# `down_proj` records are not separable in situ, and the honest word for "better than rank-0, not
# separable from the incumbent" is a tie — not `superseded` wearing a mechanism that was refuted. A tie
# may never ship (see `check_verdict_data`), so `shipped == the win rows` keeps holding without anyone
# having to fake a measurement in the losing direction.
VERDICT_WORDS = ("win", "lose", "tie", "superseded")
# How much a verdict outranks another, for the one derived rule that needs an ordering: a shipped row may
# not be dominated by a strictly better-measured record for the same shape.
VERDICT_RANK = {"win": 2, "tie": 1, "lose": 0, "superseded": 0}
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


def pin_shape(name: str) -> str:
    """The sweep's shape name for a pin row: `deep_down_proj` and `down_proj` are both the `down_proj`
    shape. Stripping the derivation prefix, rather than keeping a name list, is what lets the engine name a
    promoted row `deep_*` and a retired row by its shape without the deriver having to be told twice.
    """
    return name.removeprefix("deep_")


def read_verdicts(path: Path) -> dict[tuple[str, str], dict]:
    """The in-situ adjudication, as data — `insitu_verdicts.tsv` holds the column meanings.

    Keyed by **(shape, algo hex)**, and the shape half is not decoration: the same kernel record can rank
    in several shapes' candidate lists, so a verdict keyed on hex alone lets a `q_proj` field winner that
    happens to be the same 24 bytes as a `down_proj` pin read as adjudicated. That is not hypothetical —
    the first version of this check reported `q_proj` as shipped-by-verdict for exactly that reason. A
    missing row is NOT "no verdict yet, ship anyway": both checks below treat that as an error.
    """
    if not path.exists():
        raise SystemExit(f"deep_pins: no in-situ verdict file at {path}. A burst-derived pin cannot be "
                         "shipped on burst evidence alone (D35: two admitted winners ran +26 % / +27 % "
                         "slower in situ), so this tool refuses to guess. The format is "
                         "`prototype/gemm_algo_sweep/insitu_verdicts.tsv`; pass --verdicts to point at "
                         "a different copy of it.")
    out: dict[tuple[str, str], dict] = {}
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cells = line.split("\t")
        if header is None:
            header = cells
            continue
        if len(cells) != len(header):
            raise SystemExit(f"deep_pins: {path} line is not {len(header)} tab-separated fields: "
                             f"{line[:80]!r}")
        rec = dict(zip(header, cells))
        for key in ("own_4096", "own_1024", "inter_4096", "inter_1024",
                    "null_floor_4096_pct", "null_floor_1024_pct"):
            raw = rec[key]
            if raw in ("", "?"):
                rec[key] = None
                continue
            try:
                rec[key] = float(raw)
            except ValueError:
                raise SystemExit(f"deep_pins: {path} {key}={raw!r} is not a number. A verdict needs the "
                                 "null floor its instrument actually measured (§9 reads +3.1 % at 4096, "
                                 "+1.9 % at 1024); `auto` or a remembered guess cannot size one")
        if (rec["null_floor_4096_pct"] is None or rec["null_floor_1024_pct"] is None):
            raise SystemExit(f"deep_pins: {path} row {rec['shape']}/{rec['hex'][:16]}… carries no null "
                             "floor. Every verdict is adjudicated against a measured null, so the floor "
                             "is a required column, not an annotation")
        if rec["verdict"] not in VERDICT_WORDS:
            raise SystemExit(f"deep_pins: {path} verdict {rec['verdict']!r} is not one of "
                             "/".join(sorted(VERDICT_WORDS)) + " — inventing a verdict word is how a "
                             "table quietly stops meaning anything")
        # D36, as a parse rule. `--gemm-pins <csv>` REPLACES the admissible set, so a spec that names a
        # subset silently excludes every shape it does not name; §9's `gu` side named one row and therefore
        # ran `down_proj` unpinned, and the +20 % that produced was recorded as a judgement about an algo.
        # A ratio whose two sides are not spelled out cannot be re-checked, so an unnamed side is a parse
        # error here rather than a hole a later reader falls into.
        for key in ("pins_treated", "pins_reference"):
            if rec.get(key, "") in ("", "?"):
                raise SystemExit(f"deep_pins: {path} row {rec['shape']}/{rec['hex'][:16]}… has "
                                 f"{key}={rec.get(key, '(absent)')!r}. Every ratio is a PAIR of pin sets; "
                                 "name both sides in full (use `none` for an all-rank-0 side) so a "
                                 "coverage effect can never again be recorded as algo physics (D36)")
        if rec["pins_treated"] == rec["pins_reference"]:
            raise SystemExit(f"deep_pins: {path} row {rec['shape']}/{rec['hex'][:16]}… names the same "
                             "pin set on both sides, so its ratio measures nothing")
        if rec["carry"] not in ("own", "interaction"):
            raise SystemExit(f"deep_pins: {path} carry={rec['carry']!r} must name which evidence the "
                             "verdict rests on: own or interaction")
        if rec["carry"] == "interaction" and not rec["inter_4096"]:
            raise SystemExit(f"deep_pins: {path} carry=interaction but inter_4096 is empty")
        key = (rec["shape"], rec["hex"])
        if key in out:
            raise SystemExit(f"deep_pins: {path} has two verdict rows for {key}")
        out[key] = rec
    if not out:
        raise SystemExit(f"deep_pins: {path} has a header and no rows — a verdict table that cannot fail "
                         "is not a check (the vacuous-check class this repo keeps hitting)")
    return out


def carrying_ratio(rec: dict) -> tuple[float, float, str]:
    """(ratio at 4096, ratio at 1024, label) for the evidence a verdict rests on.

    A verdict is only as good as the number it points at. `own` is the family of THIS shape with the
    treated pin set in force over the reference set; `interaction` is a family row this record does not
    own but measurably moves. Naming which pair carries a verdict is what stops an inside-the-noise own-
    ratio being quietly propped up by a big neighbour — and the label now prints both pin sets, because a
    pair that cannot name its two sides is what D36 was about.
    """
    pair = f"{rec['pins_treated']} vs {rec['pins_reference']}"
    if rec["carry"] == "own":
        return rec["own_4096"], rec["own_1024"], f"own ({pair})"
    return rec["inter_4096"], rec["inter_1024"], f"interaction ({pair})"


def check_verdict_data(verdicts: dict[tuple[str, str], dict], engine_rows: list[dict]) -> int:
    """The verdict file against the engine table, with no sweep window needed.

    Four things, each in a direction that can actually fail:
      * every row the engine carries has a verdict for its (shape, hex), and every verdict names a row the
        engine carries — a verdict for a record nobody carries is rotted data, a row with no verdict is an
        unadjudicated pin;
      * `kShipped` is exactly the `win` rows and `kCandidate` is exactly `lose` + `superseded`, so a pin
        cannot move in C++ without moving in the data too;
      * at most one shipped row per shape (two is the runtime refusal, caught at derive time instead);
      * a `win`'s carrying evidence must clear that row's own null-test floor and must be faster at both
        widths — a verdict inside the noise of the instrument that produced it is not a verdict.
    """
    rc = 0
    eng = {r["hex"]: r for r in engine_rows}
    if len(eng) != len(engine_rows):
        print("VERDICTS FAIL reason=two engine rows carry the same algo hex")
        rc = 1
    eng_keys = {(pin_shape(r["name"]), r["hex"]) for r in engine_rows}
    missing = sorted(f"{s}/{h[:16]}…" for (s, h) in eng_keys if (s, h) not in verdicts)
    orphan = sorted(f"{s}/{h[:16]}…" for (s, h) in verdicts if (s, h) not in eng_keys)
    if missing or orphan:
        print(f"VERDICTS FAIL reason=the engine table and the verdict file are not the same set of "
              f"records. in_engine_without_verdict={missing} verdict_without_engine_row={orphan}. Retire "
              "a row as `superseded`, do not delete it: a deleted pin cannot be re-adjudicated without a "
              "re-capture, and a verdict nobody carries is data that has already rotted.")
        rc = 1
    mismatch = []
    for r in engine_rows:
        v = verdicts.get((pin_shape(r["name"]), r["hex"]))
        if v is None:
            continue
        want = "kShipped" if v["verdict"] == "win" else "kCandidate"
        if r["set"] != want:
            mismatch.append(f"{r['name']}(PinSet::{r['set']}, verdict={v['verdict']} wants {want})")
    if mismatch:
        print(f"VERDICTS FAIL reason=PinSet disagrees with the verdicts: {sorted(mismatch)}. A pin moves "
              "by editing the row's PinSet AND its verdict row, in one commit; anything weaker lets the "
              "data and the binary drift apart, which is how a table starts describing a build nobody "
              "ran.")
        rc = 1
    shipped_shapes = [pin_shape(r["name"]) for r in engine_rows if r["set"] == "kShipped"]
    dupe = sorted({s for s in shipped_shapes if shipped_shapes.count(s) > 1})
    if dupe:
        print(f"VERDICTS FAIL reason=two shipped rows for shape(s) {dupe} — the engine refuses two pins on "
              "one (N,K,M) at plan time, so this table could not launch")
        rc = 1
    for v in verdicts.values():
        r4096, r1024, label = carrying_ratio(v)
        f4096 = v["null_floor_4096_pct"] / 100.0
        f1024 = v["null_floor_1024_pct"] / 100.0
        separated = (abs(r4096 - 1.0) > f4096, abs(r1024 - 1.0) > f1024)
        if v["verdict"] == "tie":
            if any(separated):
                print(f"VERDICTS FAIL shape={v['shape']} hex={v['hex'][:16]}… verdict=tie reason=its "
                      f"carrying evidence ({label}: {r4096:.4f} at 4096, {r1024:.4f} at 1024) separates "
                      f"past its own null ({v['null_floor_4096_pct']} % / "
                      f"{v['null_floor_1024_pct']} %). A tie is the word for TWO RECORDS THE INSTRUMENT "
                      "CANNOT TELL APART; once it separates, restate the row as win or lose — do not leave "
                      "a measuring-tie in a table that decides what ships")
                rc = 1
        elif not all(separated):
            print(f"VERDICTS FAIL shape={v['shape']} hex={v['hex'][:16]}… verdict={v['verdict']} "
                  f"reason=its carrying evidence ({label}: {r4096:.4f} at 4096, {r1024:.4f} at 1024) is "
                  f"inside its own null test ({v['null_floor_4096_pct']} % / "
                  f"{v['null_floor_1024_pct']} %). §9's null test — the `proj` family under a config that "
                  "cannot change it — exists so a row like this cannot be read as a result in either "
                  "direction. If it is genuinely not separable, the verdict word is `tie`.")
            rc = 1
        if v["verdict"] == "win" and max(r4096, r1024) >= 1.0:
            print(f"VERDICTS FAIL shape={v['shape']} reason=verdict=win but its carrying ratios are "
                  f"{r4096:.4f}/{r1024:.4f}, i.e. not faster at both widths")
            rc = 1
        if v["verdict"] in ("lose", "superseded") and min(r4096, r1024) <= 1.0:
            print(f"VERDICTS FAIL shape={v['shape']} reason=verdict={v['verdict']} but its carrying ratios "
                  f"are {r4096:.4f}/{r1024:.4f}, i.e. not slower at both widths")
            rc = 1
    # The config a verdict was measured under is part of the verdict, not a comment on it. D36 is the
    # reason: `--gemm-pins <csv>` replaces the admissible set, so a side that named one row silently ran
    # every other shape unpinned, and the +20 % that came out was recorded as physics about an algo.
    eng_names = {r["name"] for r in engine_rows}
    phantom, misown, nocouple = [], [], []
    for v in verdicts.values():
        treated = [s.strip() for s in v["pins_treated"].split(",") if s.strip()]
        reference = [s.strip() for s in v["pins_reference"].split(",") if s.strip()]
        for spec, side in ((treated, "treated"), (reference, "reference")):
            for token in spec:
                if token != "none" and token not in eng_names:
                    phantom.append(f"{v['shape']}/{v['hex'][:8]}… {side} names {token!r}")
        own = [r["name"] for r in engine_rows
               if pin_shape(r["name"]) == v["shape"] and r["hex"] == v["hex"]]
        if not own or own[0] not in treated:
            misown.append(f"{v['shape']}/{v['hex'][:8]}… treated={v['pins_treated']!r} "
                          f"does not name this row ({own[0] if own else 'no such engine row'})")
        if v["carry"] == "interaction" and not any(pin_shape(t) != v["shape"] for t in treated):
            nocouple.append(f"{v['shape']}/{v['hex'][:8]}… carry=interaction but pins_treated="
                            f"{v['pins_treated']!r} names no other shape's pin")
    if phantom or misown or nocouple:
        print(f"VERDICTS FAIL reason=a verdict does not name a pin set the engine can express. "
              f"phantom_pin_names={phantom} treated_side_missing_its_own_row={misown} "
              f"interaction_without_a_co_named_pin={nocouple}. `--gemm-pins` REPLACES the set, so a verdict "
              "keyed to shape+algo without its two pin sets can record a coverage effect as algo physics — "
              "exactly how D35 got the `down_proj` row backwards.")
        rc = 1
    if not rc:
        ship = sorted(r["name"] for r in engine_rows if r["set"] == "kShipped")
        cand = sorted(r["name"] for r in engine_rows if r["set"] == "kCandidate")
        words = ",".join(f"{v['verdict']}:{v['shape']}/{v['hex'][:8]}…" for v in
                         sorted(verdicts.values(), key=lambda v: (v["shape"], v["hex"])))
        print(f"VERDICTS file_verdicts={len(verdicts)} engine_rows={len(engine_rows)} "
              f"shipped={len(ship)} ({','.join(ship)}) candidates={len(cand)} ({','.join(cand)}) "
              f"by_word=[{words}] PASS (every row has a verdict, the sets match the verdicts, every "
              "verdict's carrying evidence is separated-or-tied against its own null, and both sides of "
              "every pair are pin sets the engine can express)")
    return rc


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
    # Keyed by SHAPE (the row's name with the `deep_` derivation prefix stripped), because a promoted row
    # is named `deep_down_proj` while the sweep calls the shape `down_proj`, and a retired row named
    # `down_proj` must not be mistaken for what ships today. Two shipped rows for one shape is the runtime
    # refusal, so stop here rather than pick one.
    pin_by_shape: dict[str, str] = {}
    for r in engine_rows:
        if r["set"] != "kShipped":
            continue
        s = pin_shape(r["name"])
        if s in pin_by_shape:
            raise SystemExit(f"deep_pins: two shipped rows map to shape {s!r} "
                             f"({pin_by_shape[s]} and {r['name']})")
        pin_by_shape[s] = r["hex"]
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
        pin_hex = pin_by_shape.get(shape)
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
        # A SECOND, narrower question, and the reason CHECK-TABLE survives a promotion. `admitted` above is
        # about the *change* — what this window says should move — so after a winner ships, the baseline
        # becomes that winner, the gain is 1.000, and `admitted` correctly goes False. If CHECK-TABLE keyed
        # on `admitted` alone, the shipped set would evaporate the moment its pins were promoted. `pinnable`
        # is the change-independent question — does this shape want a pin at all, and which record is it —
        # measured against rank-0, which is what an shape with NO pin gets. The verdict identity is written
        # over `pinnable`.
        pin_reasons = []
        if not winner["admissible"]:
            pin_reasons.append(winner["refuse_reason"])
        if winner["witness"] != "matched":
            pin_reasons.append("winner hex is not witnessed in any ALGODUMP capture")
        if win_ms >= d_ms:
            pin_reasons.append("the deep winner is rank-0 or slower: this shape wants no pin")
        if floor is not None and (1.0 - win_ms / d_ms) <= floor:
            pin_reasons.append(f"gain over rank-0 {100*(1-win_ms/d_ms):.1f} % does not clear this shape's "
                               f"own cross-window movement {100*floor:.1f} %")
        rank0_prefill = calls * (d_ms - win_ms) * 8.0
        if rank0_prefill < MIN_PCT_OF_WAVE / 100.0 * wave_ms:
            pin_reasons.append(f"worth {100*rank0_prefill/wave_ms:.3f} % of a wave over rank-0, under the "
                               f"{MIN_PCT_OF_WAVE} % floor a re-capture obligation is worth")
        rows.append({
            "shape": shape, "M": m, "n": res["N"], "k": res["K"], "calls_per_wave": calls,
            "family": plan.get("family", "?"),
            "default_seq": int(res["default_seq"]), "default_ms": d_ms,
            "shipped_pin_ms": pin_ms, "shipped_pin_seq": pin_seq, "baseline_ms": base_ms,
            "baseline_name": base_name,
            "noise_floor": floor, "noise_worst": worst,
            "admitted": not reasons, "reject_reasons": reasons,
            "pinnable": not pin_reasons, "unpinnable_reasons": pin_reasons,
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


def print_report(rows: list[dict], dump: Dump, cap: int, heur_width: int, wave_ms: float,
                 verdicts: dict[tuple[str, str], dict]) -> int:
    print(f"\nDEPTH shape-level (M={rows[0]['M'] if rows else '?'}; the engine's only shippable "
          f"width is kMaxWaveRows=512, engine/decode_step.h:49). BASE is what runs TODAY for that "
          f"shape: its shipped pin where one exists, else rank-0. PINNABLE is the change-independent "
          "question (does this shape want a pin at all, measured against rank-0); ADMIT is what this "
          "window says should MOVE; the last column is the in-situ verdict, which outranks both.")
    print(f"{'shape':18s} {'calls':>5} {'BASE ms':>8} {'best ms':>8} {'best rank':>9} "
          f"{'best/BASE':>9} {'noise':>6} {'saved/wave':>10} {'%/wave':>7} verdict")
    total = 0.0
    admitted_total = 0.0
    blockers = 0
    for r in rows:
        w = r["winner"]
        if w["witness"] != "matched" or not w["admissible"]:
            blockers += 1
        admitted_total += r["saved_ms_per_wave"] if r["admitted"] else 0.0
        per_prefill = r["saved_ms_per_wave"] * 8.0
        noise_txt = f"{100*r['noise_floor']:.1f} %" if r["noise_floor"] is not None else "n/a"
        head = "ADMIT" if r["admitted"] else "no change: " + "; ".join(r["reject_reasons"])
        if r["pinnable"]:
            v = verdicts.get((r["shape"], w["hex"]))
            if v is None:
                tail = " | PINNABLE but UNVERDICTED — cannot be shipped until measured in situ"
            else:
                r4096, r1024, label = carrying_ratio(v)
                tail = (f" | in-situ {v['verdict']} ({label} {r4096:.4f}/{r1024:.4f}, "
                        f"floor {v['null_floor_4096_pct']}/{v['null_floor_1024_pct']} %)")
        else:
            tail = " | " + "; ".join(r["unpinnable_reasons"])
        base_tag = "" if r["shipped_pin_ms"] is None else f" ({r['baseline_name']})"
        print(f"{r['shape']:18s} {r['calls_per_wave']:>5} {r['baseline_ms']:>8.4f} {w['ms']:>8.4f} "
              f"{w['heur_rank']:>9} {w['ms'] / r['baseline_ms']:>9.4f} {noise_txt:>6} "
              f"{r['saved_ms_per_wave']:>10.4f} {100*per_prefill/wave_ms:>7.3f} {head}{base_tag}{tail}")
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
    print(f"DEEP-PRIZE wanting-a-change-set: saved_per_24layer_wave={admitted_total:.4f} "
          f"per_4096_lane_prefill={admitted_total*8.0:.4f} "
          f"pct_of_wave({wave_ms:.2f} ms)={100.0*admitted_total*8.0/wave_ms:.3f} — this is what the burst "
          "says should MOVE, and it is not what ships: on today's verdicts every shape in it is an "
          "in-situ loser or a tie, so booking this figure would be booking the +26 % in reverse")
    shipped_burst = sum(r["calls_per_wave"] * (r["default_ms"] - r["winner"]["ms"]) for r in rows
                        if verdicts.get((r["shape"], r["winner"]["hex"]), {}).get("verdict") == "win")
    print(f"DEEP-PRIZE shipped-set-worth-vs-no-pin (burst, vs rank-0, counted on the win rows that are "
          f"also this window's field winner): saved_per_24layer_wave={shipped_burst:.4f} "
          f"pct_of_wave({wave_ms:.2f} ms)={100.0*shipped_burst*8.0/wave_ms:.3f} — two omissions are "
          "deliberate, not a bug: the shipped `down_proj` incumbent is not this window's winner (its tie "
          "challenger is), and its burst value against rank-0 was banked by D28, so re-counting it here "
          "would double-book a win the shipping path already has. §9 measured the delivered set at -0.63 % "
          "of wave in situ, so this burst figure is ~2-3x the delivered win: right about the sign, wrong "
          "about the size. Quote §9 for the wave; this line exists to show the gap rather than hide it.")
    print(f"DEEP-PRIZE blocked={blockers} (unwitnessed hex or over-cap workspace stamp) — a blocked "
          "winner is not a candidate, it is a capture that has not been taken yet")
    print("DEEP-PRIZE reading: best/BASE < 1 is what changes the shipping path, and the verdict in "
          "`insitu_verdicts.tsv` is what decides whether it is allowed to: two shapes admitted here ran "
          "+26 % and +27 % slower in situ, and a third is a tie that may not buy a pin. What D36 settled is "
          "the direction of the arrow — a pin set's COVERAGE costs, not which of two near-equal winners runs "
          "a shape: leaving `down_proj` unpinned moved its family about 20 %, swapping its two winners moved "
          "it inside the null. So a burst number is an upper bound, the in-situ verdict is the decision, "
          "and `wall_ms` cannot adjudicate either way (§9's null test reads +3.1 % / +1.9 % on a family no "
          "pin can touch).")
    return blockers


def check_table(rows: list[dict], engine_rows: list[dict],
                verdicts: dict[tuple[str, str], dict]) -> int:
    """The engine's pin table must equal what this window derives, MODULO the in-situ verdicts.

    Two identities, and they are the whole point of the tool having verdict data at all:

        shipped    ==  the in-situ-win rows
        candidates ==  every non-win verdict (lose ∪ tie ∪ superseded)

    plus three derived rules that only make sense with a window in hand: anything pinnable with no verdict
    is an ERROR rather than a silent absence (a re-take must record a verdict before a pin can move); a
    `tie` may never be shipped and must BE the field winner it claims to tie against; and a shipped row may
    not be dominated by a strictly better-measured record for the same shape. `pinnable` is what this burst
    window says the shape wants at all (rank-0 as the baseline, so it does not move when a pin is promoted),
    and the verdict file is the only thing that can turn a burst winner into a shipped pin.

    `tie` is why the first identity is stated over verdicts and not over `pinnable ∩ wins`: the shipped
    `down_proj` row is an incumbent, and this window's fastest record for that shape is a different algo
    that measures a tie against it (D36). Keying the identity on the field winner alone would demand that
    a tie displace a win, which is the opposite of what a tie means.

    Note what this does NOT do: it does not re-litigate a verdict. The burst window cannot see what a
    neighbouring pin changes, and pretending it could is how `o_proj+out_proj` and `in_proj_qkv` were
    admitted twice and rejected twice.
    """
    # Shape-AND-hex keyed, never hex alone: one kernel record can rank in two shapes' fields (the first
    # version of this check reported `q_proj` as shipped because its fastest record happened to be the same
    # 24 bytes as the `down_proj` pin), and an identity that collapses two shapes into one cannot adjudicate
    # either of them.
    eng_ship = {(pin_shape(r["name"]), r["hex"]) for r in engine_rows if r["set"] == "kShipped"}
    eng_cand = {(pin_shape(r["name"]), r["hex"]) for r in engine_rows if r["set"] == "kCandidate"}
    pinnable = {(r["shape"], r["winner"]["hex"]) for r in rows if r["pinnable"]}
    admitted = sum(1 for r in rows if r["admitted"])
    retired = {(s, h) for (s, h), v in verdicts.items() if v["verdict"] == "superseded"}
    eng_all = eng_ship | eng_cand
    word = {k: v["verdict"] for k, v in verdicts.items()}
    # The one row per shape that this window says is worth anything, and its verdict if it has one.
    field_winner = {r["shape"]: (r["shape"], r["winner"]["hex"]) for r in rows if r["pinnable"]}

    def nm(keys):
        return sorted(f"{s}:{h[:8]}" for s, h in keys)

    unverdicted = sorted(f"{s}:{h[:8]}" for (s, h) in pinnable if (s, h) not in verdicts)
    want_ship = {k for k, w in word.items() if w == "win"}
    want_cand = {k for k, w in word.items() if w != "win"} | (retired & eng_all)
    ties = {k for k, w in word.items() if w == "tie"}
    # A tie is a claim about THIS window's best record, so it can only be checked against this window.
    tie_offfield = sorted(f"{s}:{h[:8]}" for (s, h) in ties
                          if field_winner.get(s) != (s, h))
    tie_shipped = sorted(f"{s}:{h[:8]}" for (s, h) in (ties & eng_ship))
    # A shipped row may not be dominated: if the field's best record for a shape carries a strictly
    # better verdict than the row the engine ships, the better one is sitting unpinned.
    dominated = []
    for key in sorted(eng_ship):
        win_key = field_winner.get(key[0])
        if win_key and win_key != key and VERDICT_RANK.get(word.get(win_key, ""), -1) > \
                VERDICT_RANK.get(word.get(key, ""), -1):
            dominated.append(f"{key[0]}: ships {key[1][:8]}…({word.get(key)}) while the field winner "
                             f"{win_key[1][:8]}… is {word.get(win_key)}")
    print(f"\nCHECK-TABLE derived_pinnable={len(pinnable)} of {len(rows)} shapes "
          f"(wanting_a_change={admitted}) verdicts={len(verdicts)} "
          f"win={len(want_ship)} tie={len(ties)} lose={sum(1 for w in word.values() if w == 'lose')} "
          f"superseded={len(retired)} engine_shipped={len(eng_ship)} engine_candidates={len(eng_cand)}")
    rc = 0
    if unverdicted:
        print(f"CHECK-TABLE FAIL reason=these shapes are pinnable on burst evidence but carry NO in-situ "
              f"verdict: {unverdicted}. That is the hole D35 closed: a burst winner is not a pin until the "
              "in-situ measurement is recorded in `insitu_verdicts.tsv`, and refusing to guess is the only "
              "way this table keeps meaning `measured by someone, in situ`.")
        rc = 1
    if tie_offfield:
        print(f"CHECK-TABLE FAIL reason=these rows claim `tie` but are not their shape's field winner in "
              f"this window: {tie_offfield}. A tie is `best the field has vs the incumbent`; asserted for a "
              "mid-field record it is unverifiable here, and the shape's actual winner would be silently "
              "unadjudicated.")
        rc = 1
    if tie_shipped:
        print(f"CHECK-TABLE FAIL reason=a tie is shipped: {tie_shipped}. A tie means the instrument cannot "
              "separate it from what already ships, so it buys nothing and inherits a re-capture obligation "
              "on every library bump — which is `MIN_PCT_OF_WAVE`'s own reason for refusing a candidate, "
              "applied in situ rather than at burst.")
        rc = 1
    if dominated:
        print(f"CHECK-TABLE FAIL reason=a shipped row is dominated by a better-measured record for the same "
              f"shape: {dominated}. Either the better record is the one that ships, or its verdict is not "
              "a win; shipping the weaker of two separated records is an unexplained edit.")
        rc = 1
    if not want_ship:
        print("CHECK-TABLE FAIL reason=no verdict says win, so the shipped set would be empty. A check "
              "whose expected answer is empty cannot fail, and a pin table with no shipped rows is not "
              "what `--gemm-pins all` means.")
        rc = 1
    if want_ship != eng_ship:
        print(f"CHECK-TABLE FAIL reason=shipped set is not the set of in-situ wins. "
              f"should_be_shipped={nm(want_ship)} missing={nm(want_ship - eng_ship)} "
              f"shipped_without_a_win={nm(eng_ship - want_ship)}. Either the C++ row's PinSet was edited "
              "without the verdict file, or a re-take changed the field and nobody re-adjudicated it.")
        rc = 1
    if want_cand != eng_cand:
        print(f"CHECK-TABLE FAIL reason=candidate set is not every non-win verdict (lose ∪ tie ∪ "
              f"superseded). should_be_candidate={nm(want_cand)} missing={nm(want_cand - eng_cand)} "
              f"in_engine_but_not_derived={nm(eng_cand - want_cand)}. A stale candidate is a kernel nobody "
              "re-measured wearing an adjudicated label; a missing one is a retired pin that was deleted "
              "instead of kept-and-labelled.")
        rc = 1
    if not rc:
        print(f"CHECK-TABLE PASS shipped == the win rows == {nm(want_ship)}; candidates == every non-win "
              f"verdict == {nm(want_cand)}; nothing pinnable is unverdicted; no shipped row is dominated by "
              "its shape's field winner; no tie is shipped")
    return rc


def emit_pins(rows: list[dict], engine_rows: list[dict], cap: int,
              verdicts: dict[tuple[str, str], dict]) -> None:
    """Print what each shape's row should be, set included — derived, so it cannot print a stale table.

    The set on each printed row is what the verdict file says, not what `algo_pins()` currently says, so a
    re-take tells you which rows need editing and in which direction. A pinnable shape with no verdict is
    printed as an error line rather than as a candidate, because "we did not measure it" and "it is a
    candidate" are different claims and only the second one gets copied into a build.
    """
    shipped_now = {r["hex"] for r in engine_rows if r["set"] == "kShipped"}
    winners = {x["winner"]["hex"] for x in rows}
    print("\n=== algo_pins() rows as the derivation + the in-situ verdicts say they should be "
          "(a row moves by editing PinSet AND its verdict row, in one commit) ===")
    n = 0
    for r in rows:
        w = r["winner"]
        if not r["pinnable"]:
            print(f"# {r['shape']}: wants no pin — {'; '.join(r['unpinnable_reasons'])}")
            continue
        v = verdicts.get((r["shape"], w["hex"]))
        if v is None:
            print(f"# {r['shape']}: PINNABLE on burst ({w['ms']:.4f} ms vs rank-0 "
                  f"{r['default_ms']:.4f}) but UNVERDICTED. Adjudicate it in situ and record the row in "
                  "insitu_verdicts.tsv before any pin may move; it is deliberately not emitted as a "
                  "candidate, because an unmeasured record and a measured-then-rejected one are not the "
                  "same thing.")
            continue
        want = "kShipped" if v["verdict"] == "win" else "kCandidate"
        r4096, r1024, label = carrying_ratio(v)
        moved = "" if (want == "kShipped") == (w["hex"] in shipped_now) else "  <-- CHANGES the shipped set"
        fields = '{"deep_%s", %s, %s, 512, "%s", PinSet::%s},' % (
            r["shape"], r["n"], r["k"], w["hex"], want)
        why = ("heur_rank=%s idx=%s %.4f ms vs rank-0 %.4f and vs BASE %.4f (%s); in-situ %s on %s = "
               "%.4f at 4096 / %.4f at 1024 (floor %s %%, reps %s, %s); ws_stamp=%s <= cap %s, "
               "witnessed in ALGODUMP%s") % (
            w["heur_rank"], w["idx"], w["ms"], r["default_ms"], r["baseline_ms"], r["baseline_name"],
            v["verdict"], label, r4096, r1024,
            f"{v['null_floor_4096_pct']}/{v['null_floor_1024_pct']}", v["reps"], v["source"],
            w["ws_stamped"], cap, moved)
        print(fields + "  // " + why)
        n += 1
    for r in engine_rows:
        if r["hex"] in winners:
            continue
        v = verdicts.get((pin_shape(r["name"]), r["hex"]))
        why = (v["note"][:110] if v else "no verdict row")
        print(f"# {r['name']}: carried but not this window's winner (PinSet::{r['set']}) — {why}")
    print(f"=== {n} row(s) the derivation and the verdicts agree on, out of {len(rows)} shapes ===")


def selftest_rule(dump_paths: list[Path], cap: int, verdicts_path: Path) -> int:
    """Three independent things, none of which needs a sweep window or a device.

    1. THE RECONSTRUCTION RULE, against the one record whose bytes are independently known. §3d of
       `prototype/gemm_algo_sweep/results.md` states its provenance: the `down_proj` pin is the ALGODUMP
       record at heur_rank=1 idx=905, captured not hand-written. So take that record from the capture and
       require byte equality with the row the engine compiles — and note the row's SET is no longer part of
       the test: the witness is about the bytes, so it still works now that the row is a retired candidate.
       If the rule were wrong about endianness or about the stamped cap, this is what says so.
    2. EVERY ROW WITNESSED. Each hex in `algo_pins()` must appear in an `ALGODUMP` capture for its own
       shape. One hand-typed character in any row — promoted or retired — fails here, which is the general
       form of the check in (1).
    3. THE SET STRUCTURE: `check_verdict_data`, which derives the expected shipped and candidate sets from
       the verdict FILE and compares them with the table in both directions. The counts printed here are
       counted out of the table; nothing asserts a remembered number, so the assertion cannot be satisfied
       by retyping it — which is the vacuous-check class this repo has now hit five times.
    """
    pins = read_committed_pins(ENGINE_GEMM)
    shipped = [p for p in pins if p["set"] == "kShipped"]
    cands = [p for p in pins if p["set"] == "kCandidate"]
    unknown = [p for p in pins if p["set"] not in ("kShipped", "kCandidate")]
    print(f"SELFTEST-RULE rows={len(pins)} shipped={len(shipped)} candidates={len(cands)} "
          f"unset={len(unknown)} names={','.join(p['name'] for p in pins)}")
    if not pins or not shipped or unknown:
        print(f"SELFTEST-RULE FAIL reason=the table must be non-empty, must ship at least one row, and "
              f"every row must name a set; found rows={len(pins)} shipped={len(shipped)} "
              f"unset={len(unknown)}. A check whose expected answer can be empty passes vacuously.")
        return 1
    rc = check_verdict_data(read_verdicts(verdicts_path), pins)
    if rc:
        return rc
    dump = Dump(dump_paths)
    watched = [p for p in pins
               if p["hex"] in dump.by_shape.get((pin_shape(p["name"]), 512), {})]
    unwatched = sorted(p["name"] for p in pins if p not in watched)
    print(f"SELFTEST-RULE engine_rows_witnessed_in_ALGODUMP={len(watched)}/{len(pins)}")
    if unwatched:
        print(f"SELFTEST-RULE FAIL reason=these rows name bytes that appear in no ALGODUMP capture for "
              f"their shape: {unwatched}. A pin is a capture, not a transcription — regenerate the capture "
              "with `--verify --algo-dump` or fix the row.")
        return 1
    by_name = {p["name"]: p for p in pins}
    anchor = by_name.get("down_proj")
    if anchor is None:
        print("SELFTEST-RULE FAIL reason=the table no longer carries a `down_proj` row at all, which was "
              "the anchor for the byte rule; pick a new record whose provenance is written down before "
              "deleting it")
        return 1
    recs = [f for f in dump.by_shape.get(("down_proj", 512), {}).values()
            if f["heur_rank"] == "1" and int(f["idx"]) == 905]
    if not recs:
        print("SELFTEST-RULE FAIL reason=no ALGODUMP line for down_proj heur_rank=1 idx=905")
        return 1
    rec = recs[0]
    ok_dump = rec["data"] == anchor["hex"]
    print(f"SELFTEST-RULE algodump_down_proj_rank1_matches_committed={1 if ok_dump else 0} "
          f"algodump={rec['data']} committed={anchor['hex']} "
          f"row_set={anchor['set']} "
          f"note=\"keyed by row NAME, not shape: `down_proj` and `deep_down_proj` are the same shape and "
          "different records, so a shape-keyed lookup would anchor on whichever row came last\"")
    if not ok_dump:
        print("SELFTEST-RULE FAIL reason=the capture and the committed pin disagree; nothing this "
              "tool reconstructs can be trusted until that is explained")
        return 1
    print("SELFTEST-RULE note=\"the ALGO-line half of the rule is checked in --check-rule-vs-log, "
          "which needs a --time-log; run both together\"")
    print("SELFTEST-RULE PASS (the pin form the engine parses is the form the library writes, every row is "
          "witnessed, and the sets match the recorded verdicts)")
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


def selftest_checks(verdicts_path: Path) -> int:
    """Prove the two new checks discriminate, in-process, with no sweep window and no device.

    A guard nobody has watched fail is decoration — the trap this repo has written up five times, and the
    reason `gate.py` carries `--break-*` injections. Each scenario below is a small tamper on the real
    verdict file or the real pin table (so the numbers stay honest), and must FAIL for its own reason while
    the untouched configuration PASSES. If someone later softens a check into a no-op, this is what goes
    red. The scenarios:

      clean                       — the shipped state, must pass both checks;
      an unverdicted pinnable shape — the re-take case D35 closed: burst says pin it, nobody measured it;
      a drifted PinSet            — a shipped row whose verdict says lose, i.e. C++ edited without the data;
      a win inside its floor      — an own ratio the null test cannot resolve, passed off as a win;
      a tie promoted to shipped   — D36's boundary: a tie may be a good algo, but it may not take a pin, so
                                    flipping only the C++ set must be caught;
      a tie promoted by relabel   — the other way round: calling the tie a `win` without moving its numbers
                                    must be caught, so the verdict cannot be faked to buy a pin;
      a verdict that lost its pair — an interaction verdict whose treated side names no other shape, and a
                                    row whose treated side does not name itself, which is the exact shape
                                    of D36's error (a coverage effect labelled as algo physics);
    """
    verdicts = read_verdicts(verdicts_path)
    engine_rows = read_committed_pins(ENGINE_GEMM)

    def make_rows(verd):
        """One synthetic window row per shape, for `check_table`, which needs a field winner.

        There is no sweep window in a selftest, so the field winner is modelled as the lowest-ranked
        verdict present for the shape — which in the real table IS the burst winner: the `tie` row beats the
        incumbent it ties against, the `lose` rows are their shapes' field winners, and the gateup `win` is
        its shape's. Modelling it any other way would make the dominance rule test nothing.
        """
        best: dict[str, tuple[int, str, dict]] = {}
        for order, ((s, h), v) in enumerate(verd.items()):
            rank = VERDICT_RANK[v["verdict"]]
            if s not in best or rank < best[s][0]:
                best[s] = (rank, h, v)
        return [{"shape": s, "winner": {"hex": h, "ms": 0.1}, "default_ms": 0.12,
                 "baseline_ms": 0.1, "baseline_name": "selftest", "calls_per_wave": 24,
                 "saved_ms_per_wave": 0.48, "wave_ms": 956.12, "admitted": v["verdict"] == "lose",
                 "pinnable": True, "unpinnable_reasons": [], "reject_reasons": []}
                for s, (_rank, h, v) in best.items()]

    rc_all = 0
    ok = check_table(make_rows(verdicts), engine_rows, verdicts) == 0 \
        and check_verdict_data(verdicts, engine_rows) == 0
    print(f"SELFTEST-CHECKS case=clean expect_fail=False got_fail={not ok} {'ok' if ok else 'NOT OK'}")
    rc_all |= 0 if ok else 1

    verd = dict(verdicts)
    extra = {"in_proj_z": "f0" + "0" * 46}
    rows = make_rows(verdicts) + [{"shape": s, "winner": {"hex": h, "ms": 0.1}, "default_ms": 0.2,
                                   "baseline_ms": 0.2, "baseline_name": "selftest",
                                   "calls_per_wave": 18, "saved_ms_per_wave": 0.288,
                                   "wave_ms": 956.12, "admitted": True, "pinnable": True,
                                   "unpinnable_reasons": [], "reject_reasons": []}
                                  for s, h in extra.items()]
    got = check_table(rows, engine_rows, verd)
    ok = bool(got)
    print(f"SELFTEST-CHECKS case=an unverdicted pinnable shape expect_fail=True got_fail={ok} "
          f"{'ok' if ok else 'NOT OK'}")
    rc_all |= 0 if ok else 1

    drifted = [dict(r) for r in engine_rows]
    for r in drifted:
        if (pin_shape(r["name"]), r["hex"]) in {(s, h) for (s, h), v in verdicts.items()
                                                if v["verdict"] == "win"}:
            r["set"] = "kCandidate"
            break
    got = check_verdict_data(verdicts, drifted) or check_table(make_rows(verdicts), drifted, verdicts)
    ok = bool(got)
    print(f"SELFTEST-CHECKS case=a drifted PinSet (shipped row whose verdict is not a win) "
          f"expect_fail=True got_fail={ok} {'ok' if ok else 'NOT OK'}")
    rc_all |= 0 if ok else 1

    import copy
    tampered = copy.deepcopy(verdicts)
    key = next((k for k, v in tampered.items() if v["verdict"] == "win"), None)
    if key is None:
        print("SELFTEST-CHECKS FAIL reason=no win verdict in the file to tamper with")
        return 1
    inside = 1.0 - tampered[key]["null_floor_4096_pct"] / 400.0
    for f in ("own_4096", "own_1024", "inter_4096", "inter_1024"):
        if tampered[key][f] is not None:
            tampered[key][f] = inside
    got = check_verdict_data(tampered, engine_rows)
    ok = bool(got)
    print(f"SELFTEST-CHECKS case=a win pulled inside its own null test expect_fail=True got_fail={ok} "
          f"{'ok' if ok else 'NOT OK'}")
    rc_all |= 0 if ok else 1

    # D36's boundary, in the C++ direction: promoting a tie by editing its PinSet alone.
    tie_key = next((k for k, v in verdicts.items() if v["verdict"] == "tie"), None)
    if tie_key is None:
        print("SELFTEST-CHECKS FAIL reason=no tie verdict in the file, so the "
              "\"a tie cannot ship\" guard has nothing to test. Delete the guard, or measure a tie.")
        return 1
    promoted = [dict(r) for r in engine_rows
                if not (pin_shape(r["name"]) == tie_key[0] and r["hex"] == tie_key[1])]
    for r in engine_rows:
        if pin_shape(r["name"]) == tie_key[0] and r["hex"] == tie_key[1]:
            r2 = dict(r)
            r2["set"] = "kShipped"
            promoted.append(r2)
    promoted.sort(key=lambda r: r["name"])
    got = check_verdict_data(verdicts, promoted) or check_table(make_rows(verdicts), promoted, verdicts)
    ok = bool(got)
    print(f"SELFTEST-CHECKS case=a tie promoted to shipped in C++ expect_fail=True got_fail={ok} "
          f"{'ok' if ok else 'NOT OK'}")
    rc_all |= 0 if ok else 1

    # ...and in the data direction: relabelling the tie a `win` while its ratios stay inside the null.
    relabel = copy.deepcopy(verdicts)
    relabel[tie_key] = dict(relabel[tie_key], verdict="win")
    got = check_verdict_data(relabel, engine_rows) or check_table(make_rows(relabel), engine_rows, relabel)
    ok = bool(got)
    print(f"SELFTEST-CHECKS case=a tie relabelled win without moving its numbers expect_fail=True "
          f"got_fail={ok} {'ok' if ok else 'NOT OK'}")
    rc_all |= 0 if ok else 1

    # The D36 error itself, twice: an interaction verdict naming no co-named pin, and a row whose treated
    # side does not name the row it claims to adjudicate. Both are label-only tampers — same numbers.
    orphan_pair = copy.deepcopy(verdicts)
    ikey = next(iter(verdicts))
    own_name = [r["name"] for r in engine_rows
                if pin_shape(r["name"]) == ikey[0] and r["hex"] == ikey[1]][0]
    rec = dict(orphan_pair[ikey], carry="interaction")
    rec["inter_4096"], rec["inter_1024"] = rec["own_4096"], rec["own_1024"]
    # Names its OWN row and nothing else: the treated set is well-formed and self-consistent, but an
    # interaction verdict that names no second shape is an interaction with nothing.
    rec["pins_treated"] = own_name
    orphan_pair[ikey] = rec
    got = check_verdict_data(orphan_pair, engine_rows)
    ok = bool(got)
    print(f"SELFTEST-CHECKS case=an interaction verdict naming no other shape's pin expect_fail=True "
          f"got_fail={ok} {'ok' if ok else 'NOT OK'}")
    rc_all |= 0 if ok else 1

    selfless = copy.deepcopy(verdicts)
    selfless[tie_key] = dict(selfless[tie_key], pins_treated="deep_gate_proj+up_proj")
    got = check_verdict_data(selfless, engine_rows)
    ok = bool(got)
    print(f"SELFTEST-CHECKS case=a verdict whose treated side omits its own row expect_fail=True "
          f"got_fail={ok} {'ok' if ok else 'NOT OK'}")
    rc_all |= 0 if ok else 1

    if rc_all:
        print("SELFTEST-CHECKS FAIL reason=a scenario did not fail when it should have, which means the "
              "check behind it cannot fail and is therefore not a check")
        return 1
    print("SELFTEST-CHECKS PASS (each guard fires on its own scenario and passes the clean one)")
    return 0



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
    ap.add_argument("--selftest-checks", action="store_true",
                    help="tamper with the verdict file and the pin table in memory and require each guard "
                         "to go red for its own reason — the injections that prove the verdict checks are "
                         "not decoration")
    ap.add_argument("--prefix-check", action="store_true",
                    help="compare the first two --algo-dump captures and report whether the shallow "
                         "heuristic list is a prefix of the deep one")
    ap.add_argument("--no-check-rule", action="store_true")
    ap.add_argument("--noise-log", type=Path, default=Path("tmp/wsjob/ws128.log"),
                    help="a SECOND --time window at the same M and budget, same-algorithm pairs from "
                         "which set the floor a candidate must clear. Pass `none` to skip the "
                         "admission rule (the print then says so per shape rather than pretending).")
    ap.add_argument("--verdicts", type=Path,
                    default=Path("prototype/gemm_algo_sweep/insitu_verdicts.tsv"),
                    help="the in-situ adjudication as data: per (shape, algo hex) a verdict of "
                         "win/lose/superseded with the measured family ratios, the null-test floor they "
                         "must clear, and the ledger section they came from. The burst window cannot see "
                         "what happens inside a wave (D35: two admitted winners ran +26 % in situ), so "
                         "this is the only input allowed to turn a derived winner into a shipped pin. A "
                         "missing file is an error, not a pass.")
    ap.add_argument("--no-check-table", action="store_true",
                    help="skip the engine-table-vs-derived comparison. Off by default on purpose: a "
                         "stale pin table is a kernel nobody measured wearing an adjudicated label.")
    a = ap.parse_args()

    dumps = a.algo_dump or [Path("/tmp/gas_algos.txt")]
    cap, heur_width = read_engine_cap(ENGINE_GEMM)
    print(f"ENGINE constants read from {ENGINE_GEMM}: kWorkspaceBytes={cap} "
          f"({cap >> 20} MiB) kHeuristicWidth={heur_width}")

    if a.selftest_rule:
        return selftest_rule(dumps, cap, a.verdicts)

    if a.selftest_checks:
        return selftest_checks(a.verdicts)

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
    verdicts = read_verdicts(a.verdicts)
    print(f"VERDICTS file={a.verdicts} rows={len(verdicts)} "
          + " ".join(f"{w}={sum(1 for v in verdicts.values() if v['verdict'] == w)}"
                      for w in VERDICT_WORDS))
    rc_verdicts = check_verdict_data(verdicts, engine_rows)
    if rc_verdicts:
        return rc_verdicts
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
    emit_pins(rows, engine_rows, cap, verdicts)
    print()
    blockers = print_report(rows, dump, cap, heur_width, a.wave_ms, verdicts)
    if not a.no_check_table:
        rc = check_table(rows, engine_rows, verdicts)
        if rc:
            return rc
    admitted = [r for r in rows if r["admitted"]]
    shipped_now = sorted({s for (s, _h), v in verdicts.items() if v["verdict"] == "win"})
    print(f"DERIVE-OUTCOME wants_a_change={len(admitted)}/{len(rows)} "
          f"names={','.join('deep_' + r['shape'] for r in admitted) if admitted else 'NONE'} "
          "note=\"burst-admitted, not ship-admitted: a name here whose winner's verdict is `lose` or "
          "`tie` was admitted at burst and refused in situ, which is the whole reason two lists exist\"")
    print(f"DERIVE-OUTCOME shipped_per_verdicts={len(shipped_now)} "
          f"names={','.join(shipped_now) if shipped_now else 'NONE'} — these are the shapes whose "
          "in-situ verdict is a win, counted on the VERDICT ROWS and not on this window's winners, because "
          "a shipped incumbent need not be the field's fastest record: the `down_proj` pin is a win "
          "against running that shape with no pin at all, while this window's fastest `down_proj` algo "
          "measures a tie against it (D36). The burst admits more than this on purpose: a burst winner is "
          "a candidate, an in-situ winner is a pin (D35).")
    return 0 if blockers == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
