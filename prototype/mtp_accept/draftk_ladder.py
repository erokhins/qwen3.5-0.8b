#!/usr/bin/env python3
"""draftk_ladder.py — the draft block's accept LADDER, in both definitions, with denominators.

THE QUESTION. Whether a k=2 speculative round is worth building turns on ONE number: how often the
draft block's SECOND row is right. `prototype/mtp_accept/ACCEPT-APPLES.md` (addendum) has the funding
arithmetic; this measures the term it is missing. The knob is `decode_run --mtp-draft-k 1|2|3|4`, which
runs the §09 block k rows per drafted position, feeding each row's own proposal back as the next row's
`y_next` and the block's own output row `x_d` as its `h` (`engine/decode_run.cpp`, and the same pair of
objects llama.cpp feeds back at `engines/llama.cpp/common/speculative.cpp:1770-1783`).

THE THREE DEFINITIONS, and why this file prints all three. `engine/gate.py`'s oldest lesson is that a
comparison across mismatched definitions is not a comparison, and llama's ladder and ours were about to
be compared that way:

  MARGINAL    m_i = P(hit_i)                      over every adjudicable row. Position i's own rate,
                                                counted even when an earlier row missed. This is the
                                                only one of the three a SIDE PASS can overstate,
                                                because here the block does propose row i whether or
                                                not row i-1 was right — an accept loop would never
                                                have asked.
  CONDITIONAL c_i = P(hit_i | hit_1 .. hit_{i-1}) the rate an accept loop actually realises at row i.
  NESTED      n_i = P(hit_1 and .. and hit_i)     the share of rounds in which position i was ACCEPTED,
                                                which is llama.cpp's `#acc rate/pos` — its counter is
                                                `n_acc_tokens_per_pos[i] / n_call_accept` over
                                                `i < n_accepted` (`common/speculative.cpp:3020-3031`),
                                                and a position is only inside `n_accepted` when every
                                                position before it was too. n_i is therefore the column
                                                that sits beside llama's 0.750 / 0.518 / 0.447 / 0.326,
                                                and the only one with the identity
                                                    tok/round = 1 + sum_i n_i
                                                on the same side. n_i = n_{i-1} * c_i, so with nested
                                                hits (no row i hit while an earlier one missed) n_i
                                                collapses onto m_i and c_2 = m_2 / m_1. THIS FILE
                                                COUNTS THE NON-NESTED ROWS AND PRINTS THEM, because
                                                whether that collapse holds is a measurement, not a
                                                definition.

RELATIONS PRINTED: c_2 = n_2 / n_1 always; c_2 = m_2 / m_1 only if the non-nested count is zero.

THE JOIN, and who checks it. A draft anchored at `pos_before = q` proposes the token at q+2; its first
proposal is settled by the step at q+1, its second by q+2, and its i-th by q+i (the shipped k=1 side
pass prints `adjudicated_by_pred_pos` for exactly this reason, and AGENTS.md records a false 0/128 that
came from aligning on `[0:]` instead). The engine prints `t`/`hit` itself, because by the time it has
the target token it can print every field on every line; THIS FILE THEN RE-DERIVES EVERY `t` AND `hit`
FROM THE RUN'S OWN `STEP kind=decode ... pos_before=` LINES AND ASSERTS THEY AGREE. Two independent
implementations of the join, in two languages, on the same run — a silent one-off cannot survive it.

A FIELD THAT DOES NOT EXIST IS `-1`, ON EVERY LINE. The last k-1 anchors of a run have no target behind
their later rows and a row the cache ceiling refused has no proposal; both read `t=-1`/`c=-1`/`hit=-1`
and are counted OUT of that column's denominator with the exclusion count printed. Every `DRAFTK` line
carries all fifteen fields in a fixed order and a line that does not aborts the run rather than being
parsed leniently — a reader that silently got fewer fields is the failure this repo keeps recording (a
check whose input list can be empty cannot fail).

CROSS-CHECKS, all of them aborting:
  * the number of parsed `DRAFTK` lines == `MTPDRAFTK lines=` == `MTPDRAFT rounds=` (the census, not a
    hope), and none of the three may be zero;
  * `hit1 != -1` count == `MTPDRAFT adjudicated` and `hit1 == 1` count == `MTPDRAFT accepted`, i.e. the
    ladder's first column IS the shipped k=1 accept rate of the same run, not a near relation to it;
  * every engine-printed `tN`/`hitN` equals the value re-derived from `STEP` lines;
  * `MTPDRAFTK identity=ok` (the block's own row counter == anchors + extra rows).

NO TIMINGS. Nothing here prints or reads a millisecond figure; this is a token-agreement instrument and a
worker-run device number is contaminated by the session on the same GPU (AGENTS.md). The shell wrapper
prints its own harness wall in SECONDS, labelled as that and nothing more.

GPU PROTOCOL: like every other `decode_run` caller, this tool does NOT take `tools/gpu_lock.sh`; the
caller holds it (`prototype/mtp_accept/draftk_ladder.sh`).

USAGE
    .venv/bin/python3 prototype/mtp_accept/draftk_ladder.py --fixture tmp/llama_input/ids_long_chat.csv \
        --k 4 --steps 256 --tag templated --json /tmp/draftk/templated.json
    .venv/bin/python3 prototype/mtp_accept/draftk_ladder.py --corpus 12 --k 2 --steps 32 --tag corpus12
    # the shipped-rate control, one column, defaults (the way 0.6641 was taken):
    .venv/bin/python3 prototype/mtp_accept/draftk_ladder.py --corpus 12 --k 1 --steps 32 --gdn-loop '' \
        --tag control-k1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "speccheck" / "fixtures" / "corpus.txt"
REFS = ROOT / "data" / "refs" / "corpus48" / "manifest.json"
STOP_ID = 248046                      # end-of-turn; referred to by id, never rendered (AGENTS.md)

# The exact field order `engine/decode_run.cpp:print_ladder_row` writes. Checked positionally, so a line
# whose fields are present but reordered cannot be read as a good line.
LADDER_KEYS = ["pos", "anchor", "k"] + [f"{n}{i}" for i in (1, 2, 3, 4)
                                        for n in ("t", "c", "hit")]

KV = re.compile(r"^([a-z_0-9]+)=([^\s]+)$")      # value domain is NOT numeric-only: see AGENTS.md


def die(msg: str):
    raise SystemExit(f"draftk_ladder: {msg}")


def parse_kv(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in line.split()[1:]:
        m = KV.match(part)
        if not m:
            die(f"field {part!r} is not key=value in: {line}")
        out[m.group(1)] = m.group(2)
    return out


def load_corpus(start: int, limit: int) -> list[tuple[str, list[int]]]:
    """The corpus lines, DERIVED from the two files that define them and cross-checked.

    Same derivation `run_accept.py` uses, for the same reason: an admission over a fixture list typed
    from memory is not an admission (seven of nine names on one such list did not exist).
    """
    lines = [ln.strip() for ln in CORPUS.read_text().splitlines() if ln.strip()]
    man = json.loads(REFS.read_text())
    prompts = man.get("prompts") or []
    if len(prompts) != len(lines):
        die(f"{REFS} carries {len(prompts)} prompts but {CORPUS} has {len(lines)} lines")
    for i, (p, text) in enumerate(zip(prompts, lines)):
        if p.get("text") != text:
            die(f"prompt {i}: manifest text != corpus line")
        if not p.get("ids"):
            die(f"prompt {i} has no token ids in {REFS}")
    if start + limit > len(lines):
        die(f"--corpus {limit} from --start {start} exceeds the population of {len(lines)} lines")
    return [(f"corpus{i}", p["ids"])
            for i, p in enumerate(prompts[start:start + limit], start=start)]


class Run:
    """One `decode_run` process, and the three record types its stdout carries for this instrument."""

    def __init__(self, name: str, cmd: list[str], log: pathlib.Path):
        self.name = name
        self.cmd = cmd
        self.log = log
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
            die(f"{name}: decode_run exited {r.returncode}; tail: "
                f"{' | '.join((r.stdout + r.stderr).strip().splitlines()[-3:])}")
        log.write_text(r.stdout)
        self.lines = r.stdout.splitlines()

    def ladder_rows(self) -> list[dict]:
        rows = []
        for ln in self.lines:
            if not ln.startswith("DRAFTK "):
                continue
            parts = ln.split()
            if len(parts) != 1 + len(LADDER_KEYS):
                die(f"{name}: a DRAFTK line has {len(parts) - 1} fields, want {len(LADDER_KEYS)}: {ln}")
            got = [p.split("=", 1)[0] for p in parts[1:]]
            if got != LADDER_KEYS:
                die(f"{name}: a DRAFTK line's fields are {got}, want {LADDER_KEYS}: {ln}")
            kv = parse_kv(ln)
            rows.append({k: int(kv[k]) for k in LADDER_KEYS})
        return rows

    def step_targets(self) -> dict[int, int]:
        """`pos_before -> the token this step predicted`: the run's own greedy continuation."""
        out: dict[int, int] = {}
        for ln in self.lines:
            if not ln.startswith("STEP kind=decode "):
                continue
            kv = parse_kv(ln)
            pos, tok = int(kv["pos_before"]), int(kv["sampled"])
            if pos in out and out[pos] != tok:
                die(f"{name}: position {pos} was predicted twice with different tokens")
            out[pos] = tok
        return out

    def census(self, tag: str) -> dict[str, str] | None:
        last = None
        for ln in self.lines:
            if ln.startswith(tag + " "):
                last = ln
        return parse_kv(last) if last else None

    def count_prefix(self, needle: str) -> int:
        return sum(1 for ln in self.lines if ln.startswith(needle))


def verify_join(run: Run, rows: list[dict]) -> None:
    """Re-derive every `tN`/`hitN` from the STEP lines and assert the engine agrees."""
    by_pos = run.step_targets()
    if not by_pos:
        die(f"{run.name}: the run printed no `STEP kind=decode` lines, so nothing can adjudicate a "
            "proposal and a ladder printed here would be unjudged")
    for r in rows:
        for i in (1, 2, 3, 4):
            if i > r["k"]:
                # Past this row's width the fields must read -1 (no proposal, therefore no target, therefore
                # no hit). Asserted rather than skipped: a leaked value here means the engine printed a `t`
                # for a row it never drafted, which is the wrong-index failure this check exists to catch.
                if (r[f"t{i}"], r[f"c{i}"], r[f"hit{i}"]) != (-1, -1, -1):
                    die(f"{run.name}: pos={r['pos']} k={r['k']} but column {i} is not the -1 triple: "
                        f"t{r[f't{i}']} c{r[f'c{i}']} hit{r[f'hit{i}']}")
                continue
            want_t = by_pos.get(r["anchor"] + i, -1)
            if r[f"t{i}"] != want_t:
                die(f"{run.name}: pos={r['pos']} t{i}={r[f't{i}']} but the STEP lines put token "
                    f"{want_t} at position {r['anchor'] + i}. The join disagrees with the run.")
            want_hit = (-1 if want_t < 0 or r[f"c{i}"] < 0
                        else 1 if r[f"c{i}"] == want_t else 0)
            if r[f"hit{i}"] != want_hit:
                die(f"{run.name}: pos={r['pos']} hit{i}={r[f'hit{i}']} but t{i}={want_t} c{i}="
                    f"{r[f'c{i}']} gives {want_hit}")


def tally(rows: list[dict], k: int) -> dict:
    """m_i / c_i / n_i with their denominators, plus the non-nested census that decides which is which."""
    cols = []
    for i in range(1, k + 1):
        hi = f"hit{i}"
        adj = [r for r in rows if r[hi] != -1]
        # NESTED denominator: every position up to i is adjudicable, so the prefix event is well defined.
        nested_pool = [r for r in rows if all(r[f"hit{j}"] != -1 for j in range(1, i + 1))]
        nested_hit = [r for r in nested_pool if all(r[f"hit{j}"] == 1 for j in range(1, i + 1))]
        if i == 1:
            cond_pool, cond_hit = adj, [r for r in adj if r[hi] == 1]
        else:
            cond_pool = [r for r in adj
                         if all(r[f"hit{j}"] == 1 for j in range(1, i))]
            cond_hit = [r for r in cond_pool if r[hi] == 1]
        nonnested = [r for r in adj if r[hi] == 1
                     and any(r[f"hit{j}"] == 0 for j in range(1, i))]
        cols.append({
            "i": i,
            "adjudicable": len(adj),
            "marginal_hits": sum(1 for r in adj if r[hi] == 1),
            "marginal": (sum(1 for r in adj if r[hi] == 1) / len(adj)) if adj else None,
            "nested_denom": len(nested_pool),
            "nested_hits": len(nested_hit),
            "nested": (len(nested_hit) / len(nested_pool)) if nested_pool else None,
            "cond_denom": len(cond_pool),
            "cond_hits": len(cond_hit),
            "conditional": (len(cond_hit) / len(cond_pool)) if cond_pool else None,
            "nonnested_hits": len(nonnested),
            "excluded_no_target": sum(1 for r in rows if r[hi] == -1 and r[f"c{i}"] >= 0),
            "excluded_no_proposal": sum(1 for r in rows if r[f"c{i}"] == -1),
        })
    return {"rows": len(rows), "cols": cols}


def fmt_rate(hits: int, denom: int) -> str:
    return "   n/a  " if denom == 0 else f"{hits / denom:.4f}"


def print_table(name: str, t: dict, engine_note: str) -> None:
    print(f"\n=== {name} — {t['rows']} drafted positions ===")
    print(f"{'pos':>4} {'MARGINAL m_i':>22} {'NESTED n_i (= llama)':>26} "
          f"{'CONDITIONAL c_i':>20} {'non-nested':>11}")
    for c in t["cols"]:
        print(f"{c['i']:>4} "
              f"{fmt_rate(c['marginal_hits'], c['adjudicable']):>10} "
              f"{c['marginal_hits']:>4}/{c['adjudicable']:<4} "
              f"{fmt_rate(c['nested_hits'], c['nested_denom']):>10} "
              f"{c['nested_hits']:>4}/{c['nested_denom']:<4} "
              f"{fmt_rate(c['cond_hits'], c['cond_denom']):>10} "
              f"{c['cond_hits']:>4}/{c['cond_denom']:<4} "
              f"{c['nonnested_hits']:>6}")
    print(f"     exclusions (rows with no target / no proposal, per column): "
          + ", ".join(f"c{c['i']}:{c['excluded_no_target']}/{c['excluded_no_proposal']}"
                      for c in t["cols"]))
    print(f"     {engine_note}")


def verdict(k: int, t: dict, ship_m1: float, out: dict) -> None:
    """The two bars, stated against NESTED rates (the ones with the tok/round identity) and marginals."""
    cols = {c["i"]: c for c in t["cols"]}
    if 2 not in cols or cols[2]["nested"] is None:
        return
    n1, n2 = cols[1]["nested"], cols[2]["nested"]
    m1, m2 = cols[1]["marginal"], cols[2]["marginal"]
    c2 = cols[2]["conditional"]
    # The bars handed down with the task, in step units and percentages only — no milliseconds exist here.
    # A k=2 round costs ~2.03 step against our k=1's 1.65, so it must beat 1.031 tok/step, i.e.
    # tok/round > 2.093, i.e. n1 + n2 > 1.093; the +10 % line needs n1 + n2 > 1.233.
    bars = (("beat k=1", 1.093, 0.487), ("reach +10 %", 1.233, 0.678))
    print(f"\n--- verdict on {k} rows (n1={n1:.4f} n2={n2:.4f} m1={m1:.4f} m2={m2:.4f} c2={c2:.4f})")
    print(f"    tok/round at k=2 = 1 + n1 + n2 = {1 + n1 + n2:.4f}"
          f"   (nested sum; the marginal sum would read {1 + m1 + m2:.4f})")
    for label, need_sum, need_c2 in bars:
        ok_nested = (n1 + n2) > need_sum
        # Projecting onto the SHIPPING geometry: its first position accepts at `ship_m1` (an accept-loop
        # number, D8 says a different h), so the loop's second-position share is ship_m1 * c2.
        n2_ship = ship_m1 * c2
        ok_ship = (ship_m1 + n2_ship) > need_sum
        print(f"    {label}: needs n1+n2 > {need_sum} (c2 > {need_c2})"
              f" | measured n1+n2 = {n1 + n2:.4f} -> {'CLEARS' if ok_nested else 'DOES NOT CLEAR'}"
              f" | projected on the shipping m1={ship_m1:.4f}: {ship_m1:.4f} + {n2_ship:.4f} "
              f"= {ship_m1 + n2_ship:.4f} -> {'CLEARS' if ok_ship else 'DOES NOT CLEAR'}")
    out["bars"] = [{"bar": b[0], "need_sum": b[1], "need_c2": b[2],
                    "nested_sum": n1 + n2, "clears_nested": (n1 + n2) > b[1],
                    "projected_ship": ship_m1 + ship_m1 * c2,
                    "clears_projected": (ship_m1 + ship_m1 * c2) > b[1]} for b in bars]


def main() -> int:
    ap = argparse.ArgumentParser()
    pop = ap.add_mutually_exclusive_group(required=True)
    pop.add_argument("--fixture", help="one --ids-file CSV of token ids")
    pop.add_argument("--corpus", type=int, metavar="N",
                     help=f"the first N lines of {CORPUS.name} (population derived and asserted)")
    ap.add_argument("--start", type=int, default=0,
                    help="first corpus line index, so a 49-line population fits in short device windows")
    ap.add_argument("--k", type=int, default=4, help="draft rows per position, 1..4")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--sched", default="loop", choices=["loop", "every"],
                    help="which round sequence the side pass walks; `loop` is the shipped one and the "
                         "one whose k=1 rate is the carried 0.6641")
    ap.add_argument("--gdn-loop", default="1",
                    help="passed to the engine; the task's ladder runs use 1, the shipped-rate control "
                         "runs at the engine default — pass '' for that")
    ap.add_argument("--stop-rule", default="no-stop", choices=["no-stop", "engine"],
                    help="`no-stop` (the default, and what every rate probe in this directory uses) keeps a "
                         "stop id from ending the run early and shrinking the population; `engine` lets "
                         "§07.5 fire, which is how the session protocol of `run_accept.py` took the "
                         "carried 0.6641 — the k=1 control must be taken that way to be the same run")
    ap.add_argument("--expect-m1", default="",
                    help="assert the first column equals HITS/DENOM (e.g. 615/926, the carried side-pass "
                         "rate). This is the invariant that proves k>1 leaves row 1 alone: row 1 is the "
                         "shipped call, so its rate must not move when rows 2..k are added. The k=1 run "
                         "alone cannot show this, because at k=1 the extra-row path never runs at all")
    ap.add_argument("--extra", default="", help="extra decode_run flags, one argv entry")
    ap.add_argument("--binary", default=str(ROOT / "engine/build/decode_run"))
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.5-0.8B"))
    ap.add_argument("--tag", default="ladder")
    ap.add_argument("--ship-m1", type=float, default=0.7008,
                    help="the measured k=1 accept of the SHIPPING accept loop on this fixture; the "
                         "verdict projects the ladder's c_2 onto it (D8: a different h, so it is a "
                         "projection and labelled one)")
    ap.add_argument("--out", default="/tmp/draftk")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not 1 <= args.k <= 4:
        die("--k takes 1..4 (the DRAFTK line carries exactly four triples)")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.corpus:
        population = load_corpus(args.start, args.corpus)
    else:
        f = pathlib.Path(args.fixture)
        if not f.is_file():
            die(f"no fixture at {f}")
        ids = [int(x) for x in f.read_text().strip().split(",")]
        population = [(f.stem, ids)]
        print(f"fixture {f}: {len(ids)} ids (asserted by field count, not by the filename)")

    cmd_common = [args.binary, "--model", args.model, "--steps", str(args.steps),
                  "--mtp-draft", "1", "--mtp-draft-k", str(args.k), "--mtp-sched", args.sched]
    if args.stop_rule == "no-stop":
        cmd_common.append("--no-stop")
    if args.gdn_loop:
        cmd_common += ["--gdn-loop", args.gdn_loop]
    if args.extra:
        cmd_common += args.extra.split()

    all_rows: list[dict] = []
    census_tot = {"rounds": 0, "adjudicated": 0, "accepted": 0, "refused": 0, "lines": 0,
                  "extra_rows": 0, "row_refused": 0, "runs": 0}
    identities = []
    for name, ids in population:
        csv = out / f"{args.tag}_{name}.csv"
        csv.write_text(",".join(str(i) for i in ids) + "\n")
        # A one-field CSV is this repo's recorded silent failure (one token parses, the run "succeeds").
        if len(csv.read_text().strip().split(",")) != len(ids):
            die(f"{name}: wrote {len(ids)} ids but the CSV reads back as "
                f"{len(csv.read_text().strip().split(','))} fields")
        run = Run(name, cmd_common + ["--ids-file", str(csv)], out / f"{args.tag}_{name}.log")
        rows = run.ladder_rows()
        if not rows:
            die(f"{name}: the run printed ZERO DRAFTK lines. An empty ladder cannot adjudicate anything; "
                "read the log before believing any rate near this.")
        verify_join(run, rows)
        dr = run.census("MTPDRAFT")
        dk = run.census("MTPDRAFTK")
        if dr is None or dk is None:
            die(f"{name}: missing MTPDRAFT or MTPDRAFTK census line")
        if dk["identity"] != "ok":
            die(f"{name}: MTPDRAFTK identity={dk['identity']} — the row tally does not add up, so no "
                "rate from this run is readable")
        # The census is the denominator's source, not a courtesy: parsed lines must equal what the engine
        # says it printed, and what it printed must equal what it drafted.
        if len(rows) != int(dk["lines"]) or int(dk["lines"]) != int(dk["anchors"]):
            die(f"{name}: parsed {len(rows)} DRAFTK lines, census says lines={dk['lines']} "
                f"anchors={dk['anchors']}")
        if int(dk["anchors"]) != int(dr["rounds"]):
            die(f"{name}: anchors={dk['anchors']} but MTPDRAFT rounds={dr['rounds']}")
        h1 = [r for r in rows if r["hit1"] != -1]
        if len(h1) != int(dr["adjudicated"]) or sum(1 for r in h1 if r["hit1"] == 1) != int(dr["accepted"]):
            die(f"{name}: the ladder's first column ({sum(1 for r in h1 if r['hit1']==1)}/{len(h1)}) is "
                f"not the run's own k=1 tally ({dr['accepted']}/{dr['adjudicated']}). The two joins of "
                "this run disagree; do not read a ladder out of it.")
        nskips = run.count_prefix("DRAFTK-SKIP")
        if nskips != int(dk["row_refused"]):
            die(f"{name}: {nskips} DRAFTK-SKIP lines vs census row_refused={dk['row_refused']}")
        for key, src in (("rounds", dr), ("adjudicated", dr), ("accepted", dr), ("refused", dr),
                         ("lines", dk), ("extra_rows", dk), ("row_refused", dk)):
            census_tot[key] += int(src[key])
        census_tot["runs"] += 1
        identities.append(f"{name}:{len(rows)}")
        all_rows += rows

    t = tally(all_rows, args.k)
    if args.expect_m1:
        want_h, want_d = args.expect_m1.split("/")
        c1 = t["cols"][0]
        if (c1["marginal_hits"], c1["adjudicable"]) != (int(want_h), int(want_d)):
            die(f"first column is {c1['marginal_hits']}/{c1['adjudicable']}, expected "
                f"{args.expect_m1}. At k>1 that means the extra rows moved the shipped first row — do "
                "not read a ladder out of this run.")
        print(f"first column == {args.expect_m1}: the extra rows did not move row 1")
    note = (f"engine census over {census_tot['runs']} run(s): rounds={census_tot['rounds']} "
            f"adjudicated={census_tot['adjudicated']} accepted={census_tot['accepted']} "
            f"refused={census_tot['refused']} extra_rows={census_tot['extra_rows']} "
            f"row_refused={census_tot['row_refused']} lines={census_tot['lines']}")
    print_table(args.tag, t, note)
    result = {"tag": args.tag, "k": args.k, "sched": args.sched, "steps": args.steps,
              "stop_rule": args.stop_rule,
              "gdn_loop": args.gdn_loop or "engine-default", "population": [n for n, _ in population],
              "ladder": t, "census": census_tot, "identity_runs": len(identities)}
    verdict(args.k, t, args.ship_m1, result)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
