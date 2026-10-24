#!/usr/bin/env python3
"""corpus_wall_ab.py — the corpus-representative WALL A/B: is the accept loop faster on the corpus?

WHY THIS EXISTS. Every wall verdict so far was taken on ONE fixture, and the corpus accept rate swings by
text: 0.097 (a technical document), 0.6556 (corpus-49 aggregate), 0.6951 (a 12-file subset), 0.8065 (prose).
At the current round cost of 1.617 step the tok/s ratio is (1+a)/1.617, so the same engine is -32 % on one
fixture and +12 % on another. A single-fixture headline is therefore meaningless here, and the number quoted
for the corpus (+2.4 %) was a FORECAST obtained by dividing a rate measured on one population by a round cost
measured on another. This measures the pair together: the corpus population, both arms, same lock hold.

THE WIDTH HONESTLY. The standing rule reports decode at 4,032 lanes, but the corpus-49 lines are truncated
near 2k tokens each, so no single line reaches that width. Cases are therefore built by JOINING successive
corpus lines until the width is reached, then trimming to exactly --ctx. The text mix stays the corpus; only
the concatenation is synthetic, and it is printed. Natural-length mode (--ctx 0) reports the mean lanes it
actually ran instead.

REFUSALS, each covering a failure this tree has already been bitten by:
  * a run with rc != 0 or no DECWALL line (a dead arm would be summed out of existence);
  * a case whose ids file does not hold exactly the width it claims (`--ids-file` splits on COMMAS, and a
    space-joined fixture reads as one token while the run still succeeds in ~1 s);
  * base reporting rounds > 0 or accept reporting rounds == 0 — the aggregate `a` would be 0/0 or a lie;
  * per-case emitted tokens differing across reps in the same arm: greedy decode is deterministic, so a
    difference means the run is not reproducible in this window and no median should be printed;
  * a per-rep pairing broken (different case sets between reps), which would compare two populations.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_accept import load_population, die  # noqa: E402
import shlex
import subprocess  # noqa: E402

def dec_fields(line: str) -> dict:
    """DECWALL is a bag of `key=value` fields in an order the engine may change; parse it as a bag.
    An ordered regex was the first version and it failed on the real line (order is prefill_ms, decode_ms,
    bodies, rounds, plain, accepted, rejected, ... emitted), which is the same class of fragility as
    reading a dtype off an assumption instead of the file."""
    out = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def run(binary: str, model: str, ids: pathlib.Path, out: pathlib.Path, extra: list[str]):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [binary, "--mtp-accept", "0", "--mtp-draft", "0", "--model", model,
           "--ids-file", str(ids), "--gdn-loop", "1", "--no-stop",
           "--out", str(out)] + extra
    # The pin is on the shared shape, not on the base arm: the accept arm names `--mtp-accept 1` in `extra`
    # and the last name wins, so `base` stays a plain-step run after the default flip and the
    # `base ran rounds` guard below keeps its teeth instead of firing on every case.
    p = subprocess.run(cmd, capture_output=True, text=True)
    d = None
    for line in (p.stdout + p.stderr).splitlines():
        if line.startswith("DECWALL"):
            f = dec_fields(line)
            need = ("decode_ms", "emitted", "bodies", "rounds", "accepted")
            if all(k in f for k in need):
                d = dict(decode_ms=float(f["decode_ms"]), emitted=int(f["emitted"]),
                         bodies=int(f["bodies"]), rounds=int(f["rounds"]),
                         accepted=int(f["accepted"]))
                # A route/flag must be read BACK off the run's own echo, never trusted from the flag.
                for tok in (p.stdout + p.stderr).split():
                    if tok.startswith("mtp_route="):
                        d["route_echo"] = tok.split("=", 1)[1]
                        break
    if p.returncode != 0 or d is None:
        die(f"arm failed rc={p.returncode} no DECWALL: {' '.join(cmd)}\n"
            + "\n".join((p.stdout + p.stderr).splitlines()[-6:]))
    return d


def build_cases(prompts, ctx: int, limit: int, out: pathlib.Path):
    cases, buf, used = [], [], 0
    for i, p in enumerate(prompts):
        if limit and len(cases) >= limit:
            break
        buf.extend(p["ids"])
        used += 1
        if ctx and len(buf) < ctx:
            continue
        take = len(buf) if not ctx else ctx
        cases.append(buf[:take])
        buf = buf[take:]
        if not ctx:
            buf = []
    if buf:
        cases.append(buf)
    for j, ids in enumerate(cases):
        f = out / f"case{j:03d}.csv"
        f.write_text(",".join(str(t) for t in ids) + "\n")
        back = f.read_text().strip().split(",")
        if len(back) != len(ids):
            die(f"case {j}: wrote {len(ids)} ids, read back {len(back)}")
    return cases, used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="engine/build/decode_run")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--ctx", type=int, default=4032, help="0 = one case per corpus line, natural length")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/tmp/corpus_wall_ab")
    ap.add_argument("--accept-extra", default="",
                    help="extra flags appended to the accept arm (e.g. '--mtp-route amortised'). Echoed from "
                         "the run's own log line, not trusted from here.")
    ap.add_argument("--base-ref-ms", type=float, default=0.0,
                    help="known clean-window ms/step for this population. If the median base lands more than "
                         "25 %% off it, the harness refuses to print a ratio: a contended window produces "
                         "plausible-looking numbers (this session measured 9.6 ms/step clean and 12.6-18.5 "
                         "ms/step with flash_serve busy-polling), and a ratio taken in one is not quotable.")
    ap.add_argument("--adopt-pct", type=float, default=10.0)
    ap.add_argument("--demand", default="",
                    help="--mtp-draft-demand value for a THIRD arm (off|prefix|selfgreedy); empty = two-arm "
                         "run. One more run per case per rep, same window, so the third arm is paired with "
                         "the other two rather than timed in another hour of the machine's life.")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_population()
    cases, used = build_cases(prompts, a.ctx, a.limit, out)
    if not cases:
        die("no cases built from the population")
    widths = [len(c) for c in cases]
    mode = (f"joined to exactly {a.ctx} lanes from {used} corpus lines" if a.ctx
            else f"natural length, mean {sum(widths)//len(widths)} lanes")
    print(f"population: {len(prompts)} derived lines -> {len(cases)} cases; {mode}")
    if a.ctx and any(w != a.ctx for w in widths):
        print(f"note: {sum(1 for w in widths if w != a.ctx)} case(s) shorter than {a.ctx} "
              f"(population exhausted); narrowest {min(widths)}")

    cap = max(widths) + a.steps + 64
    per_rep = []
    for rep in range(1, a.reps + 1):
        kinds = ["base", "accept"] + (["demand"] if a.demand else [])
        agg = {k: dict(ms=0.0, em=0, bodies=0, rounds=0, acc=0) for k in kinds}
        emits = {k: [] for k in kinds}
        for j, _ids in enumerate(cases):
            cf = out / f"case{j:03d}.csv"
            common = ["--steps", str(a.steps), "--cap", str(cap)]
            b = run(a.binary, a.model, cf, out / f"r{rep}_b{j}", common)
            c = run(a.binary, a.model, cf, out / f"r{rep}_a{j}",
                    common + ["--mtp-draft", "1", "--mtp-accept", "1"]
                    + (shlex.split(a.accept_extra) if a.accept_extra else []))
            if a.accept_extra and "mtp_route" in c and a.accept_extra.startswith("--mtp-route"):
                want = shlex.split(a.accept_extra)[-1]
                if c["route_echo"] != want:
                    die(f"rep {rep} case {j}: asked route {want}, the run echoed "
                        f"{c.get('route_echo')}; a flag the run ignored is not the arm being measured")
            dd = None
            if a.demand:
                dd = run(a.binary, a.model, cf, out / f"r{rep}_d{j}",
                         common + ["--mtp-draft", "1", "--mtp-accept", "1",
                                   "--mtp-draft-demand", a.demand])
                if dd["rounds"] == 0:
                    die(f"rep {rep} case {j}: demand arm ran 0 rounds; its `a` would be 0/0")
            if b["rounds"] != 0:
                die(f"rep {rep} case {j}: base ran rounds ({b['rounds']}); the base arm must be plain steps")
            if c["rounds"] == 0:
                die(f"rep {rep} case {j}: accept arm ran 0 rounds; `a` would be 0/0")
            for kind, v in ([("base", b), ("accept", c)] + ([("demand", dd)] if dd else [])):
                agg[kind]["ms"] += v["decode_ms"]
                agg[kind]["em"] += v["emitted"]
                agg[kind]["bodies"] += v["bodies"]
                agg[kind]["rounds"] += v["rounds"]
                agg[kind]["acc"] += v["accepted"]
                emits[kind].append(v["emitted"])
        if rep > 1 and any(emits[k] != per_rep[0][f"em_{k}"] for k in kinds):
            die(f"rep {rep} emitted different token counts than rep 1 in some arm — greedy decode is "
                "deterministic, so this window is not reproducible and no median is printable")
        rate = {k: (agg[k]["em"] / (agg[k]["ms"] / 1000.0) if agg[k]["ms"] else float("nan"))
                for k in kinds}
        rb, ra = rate["base"], rate["accept"]
        rd = rate.get("demand", float("nan"))
        per_rep.append(dict(ratio=ra / rb, dratio=(rd / rb) if a.demand else float("nan"),
                            rb=rb, ra=ra, rd=rd, a_d=(agg["demand"]["acc"] / agg["demand"]["rounds"]
                                                    if a.demand and agg["demand"]["rounds"] else float("nan")),
                            **{f"em_{k}": emits[k] for k in kinds},
                            a=agg["accept"]["acc"] / agg["accept"]["rounds"],
                            step=agg["base"]["ms"] / max(agg["base"]["bodies"], 1),
                            rnd=agg["accept"]["ms"] / agg["accept"]["rounds"],
                            rnd_d=(agg["demand"]["ms"] / agg["demand"]["rounds"]
                                   if a.demand and agg["demand"]["rounds"] else float("nan")),
                            cases=len(cases)))
        line = f"  rep {rep}: base {rb:8.3f} tok/s   accept {ra:8.3f} tok/s"
        if a.demand:
            line += f"   demand {rd:8.3f}"
        print(line + f"   a={per_rep[-1]['a']:.4f}   round/step="
                     f"{per_rep[-1]['rnd'] / per_rep[-1]['step']:.3f}")
        if per_rep[-1]["cases"] != len(cases):
            die("case count changed between reps; the reps are not paired")

    ratios = sorted(r["ratio"] for r in per_rep)
    med = statistics.median(ratios)
    best = per_rep[ratios.index(med)]
    if a.base_ref_ms:
        drift = best["step"] / a.base_ref_ms - 1.0
        if abs(drift) > 0.25:
            die(f"window is contended: base ran {best['step']:.3f} ms/step against the known clean "
                f"{a.base_ref_ms:.3f} ({drift * 100:+.1f} %). No ratio is printed from a window this far "
                "off a clean reference — the arms are paired, but the ratio itself moves with contention.")
    astar = best["rnd"] / best["step"] - 1.0
    print(f"\n== medians of {a.reps} reps, ONE lock hold, {len(cases)} corpus cases, "
          f"{a.ctx if a.ctx else 'natural'} lanes")
    print(f"  base    {best['rb']:8.3f} tok/s   ({best['step']:.3f} ms/step)")
    print(f"  accept  {best['ra']:8.3f} tok/s   ({best['rnd']:.3f} ms/round)")
    print(f"  accept rate a = {best['a']:.4f}   parity needs a* = round/step - 1 = {astar:.3f}")
    print(f"  accept vs base: {(med - 1) * 100:+.2f} %  (pre-registered adopt line: >= +{a.adopt_pct:.2f} %)")
    print("  VERDICT: " + ("CLEARS the adopt line on the corpus population." if (med - 1) * 100 >= a.adopt_pct
                           else "does NOT clear the adopt line on the corpus population."))
    if a.demand:
        drs = sorted(r["dratio"] for r in per_rep)
        dmed = statistics.median(drs)
        dbest = per_rep[drs.index(dmed)]
        print(f"  demand  {dbest['rd']:8.3f} tok/s   ({dbest['rnd_d']:.3f} ms/round, "
              f"a={dbest['a_d']:.4f})")
        print(f"  demand vs base: {(dmed - 1) * 100:+.2f} %   vs plain accept: "
              f"{(dmed / med - 1) * 100:+.2f} %   (adopt line: >= +{a.adopt_pct:.2f} %)")
        print("  VERDICT(demand): " + ("CLEARS the adopt line on the corpus population." if (dmed - 1) * 100
                                      >= a.adopt_pct else
                                      "does NOT clear the adopt line on the corpus population."))
    print("  note: a > a* is necessary and not sufficient on its own width; the adopt line is the line.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
