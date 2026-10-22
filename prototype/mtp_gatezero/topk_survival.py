#!/usr/bin/env python3
"""Study A - candidate-set survival of the TARGET's own next token, measured on the pinned dumps.

P7 gate zero, part 2 (route (a) of MTP-GATEZERO.md S1.2). A k=1 MTP draft has to turn `x_d` into a
token id, and the tied head makes that a second full 485 MiB walk (2.1685 ms quoted). The restricted
route instead asks: the target already computed a full logits row on the PREVIOUS step - the row whose
argmax was the token we are drafting behind. If the drafted token is usually inside that row's top-k,
then the draft's head is a `k`-row gather (k*2048 B) instead of the whole table.

So the quantity that decides route (a) is

    survival(k) = P( target's token at position i+1 is inside top-k of the row that produced token i )

which is exactly what the pinned `decode{N}.logits_last` rows can answer, with no MTP weights touched.

ALIGNMENT. The free-run index trap (`AGENTS.md`: a free run compared against the wrong index reads
0/128 and looks like total divergence) applies directly, so nothing here assumes an index. For each
fixture the script FIRST proves the pairing on the fixture's own manifest:

    argmax(logits_last)                        == greedy_continuation[0]
    argmax(decode{s}.logits_last)              == greedy_continuation[s+1]
    decode{s}.input_ids                        == greedy_continuation[s]

and refuses to report a survival rate for a fixture whose pairing does not reproduce. Row `decode{s}`
produces `g = greedy[s+1]`, so the token a drafter standing on that row would propose is
`greedy[s+2]`; rows with no `s+2` available are counted and reported as dropped, never guessed at.

That guard earned its keep on the first run: the `logits_storage = bf16` fixtures store the row as raw
BF16 BITS in a uint16 array, and comparing those as numbers is silently wrong (uint16 order is not
float order for negatives), so the naive read produced a confident survival table over garbage. Rows
are therefore decoded by the dtype the manifest declares, and `pin_pair_search.py` is what proves
`final_hidden` is the lm_head's input (cos 0.999989 against the paired logits row).

Ranks are computed both ways against BF16 ties: `rank_strict` = #{logits strictly greater} + 1
(the best case for membership) and `rank_tie` = #{logits >= the target's value} (the worst), because
the pins are BF16 and a boundary tie can land either side of a cut.

    .venv/bin/python prototype/mtp_gatezero/topk_survival.py [--fixtures dec8,prefix24dec8,...]
                                                             [--ks 1,2,4,...,65536]

CPU only, no device, no timing. Cost figures printed are QUOTED from engine/generation-limits.md and
the block size comes from mtp_census.py's JSON.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen3.5-0.8B"
SHARD = MODEL / "model.safetensors-00001-of-00001.safetensors"
NORMW = "model.language_model.norm.weight"

# QUOTED: engine/generation-limits.md S1 (paired window, 4,032 lanes). Never re-derived here.
QUOTED = {"step_ms": 10.9357, "lm_head_ms": 2.1685, "lm_head_mib": 509.552,
          "plateau_gibs": 237.0, "launch_us": 3.10}

DEFAULT_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536]


def declared_dtype(man: dict, pin: str) -> str:
    """The manifest's own storage claim for one pin ('bfloat16', 'bfloat16(bits:uint16)', ...)."""
    p = man.get("pins", {}).get(pin)
    if not isinstance(p, dict) or "dtype" not in p:
        raise SystemExit(f"manifest has no dtype claim for pin {pin!r}; refusing to guess a "
                         f"storage form from the file extension")
    return str(p["dtype"])


def load_row(path: pathlib.Path, vocab: int, want: str) -> np.ndarray:
    """A pinned logits row as F64, decoded by the dtype the MANIFEST declares for it.

    Two storage forms exist in `data/refs/` and they are one letter apart while meaning different
    things: the old fixtures hold BF16 values widened to F32 (993,408 B for 248,320 rows), the
    `logits_storage = bf16` ones hold raw BF16 *bits* in a uint16 array (496,768 B). Reading the
    uint16 form as numbers is silently wrong rather than loud: uint16 order is not float order for
    negatives, so argmax and every top-k rank come out garbage that still looks like data - the same
    failure as the `bf16_ulp` dead-branch trap in AGENTS.md. So: decode by declaration, and refuse if
    the file does not match the declaration.
    """
    raw = np.load(path).reshape(-1)
    if raw.size != vocab:
        raise SystemExit(f"{path.name}: {raw.size} elements, expected {vocab}")
    bits_form = "bits:uint16" in want
    if bits_form:
        if raw.dtype != np.uint16:
            raise SystemExit(f"{path.name}: manifest declares {want!r} but the file holds "
                             f"{raw.dtype}; refusing")
        vals = torch.from_numpy(np.ascontiguousarray(raw)).view(torch.bfloat16).float().numpy()
    elif want.startswith("bfloat16"):
        if raw.dtype not in (np.float32, np.float64):
            raise SystemExit(f"{path.name}: manifest declares {want!r} but the file holds "
                             f"{raw.dtype}; refusing")
        vals = raw.astype(np.float64)
    elif want.startswith("float32"):
        if raw.dtype != np.float32:
            raise SystemExit(f"{path.name}: manifest declares {want!r} but the file holds "
                             f"{raw.dtype}; refusing")
        vals = raw.astype(np.float64)
    else:
        raise SystemExit(f"{path.name}: unhandled declared dtype {want!r}")
    return vals


def load_pin(d: pathlib.Path, man: dict, name: str) -> np.ndarray:
    """Any pin by name, decoded by the dtype its manifest entry declares (see load_row's note)."""
    path = d / f"{name}.npy"
    if not path.exists():
        raise SystemExit(f"{d.name}: no pin file for {name}")
    raw = np.load(path).reshape(-1)
    want = declared_dtype(man, name)
    if "bits:uint16" in want or want.startswith("bfloat16"):
        if raw.dtype == np.uint16:
            return torch.from_numpy(np.ascontiguousarray(raw)).view(torch.bfloat16).float().numpy()
        return raw.astype(np.float64)
    if want.startswith("int"):
        return raw.astype(np.int64)
    return raw.astype(np.float64)


def embed_meta() -> tuple[int, int, int]:
    with safe_open(str(SHARD), framework="pt") as f:
        t = f.get_slice("model.language_model.embed_tokens.weight")
        shape = t.get_shape()
        dt = t.get_dtype()
    return int(shape[0]), int(shape[1]), {"BF16": 2, "F16": 2, "F32": 4}[dt]


def gamma_norm(h_bits: torch.Tensor) -> torch.Tensor:
    """The §03.1 zero-centred RMSNorm the engine's final norm performs, in the shape the pin is stored.

    `h -> gamma * (h * rsqrt(mean(h^2)))` with the reduction in F32 and one BF16 rounding at the end,
    which is what makes it comparable bit for bit with a BF16 pin.
    """
    with safe_open(str(SHARD), framework="pt") as f:
        g = f.get_tensor(NORMW).to(torch.float32)
    x = h_bits.to(torch.float32)
    y = g * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)))
    return y.to(torch.bfloat16)


def fixture_rows(tag: str, vocab: int, verbose: bool) -> tuple[list[tuple[np.ndarray, int]], list[str]]:
    """Returns (rows, notes): rows = [(logits row, the token it emits, the NEXT token wanted)].

    The third element is what a drafter standing on that row proposes, so the list is built only from
    pairs the manifest confirms.
    """
    d = ROOT / "data" / "refs" / tag
    notes: list[str] = []
    man = json.loads((d / "manifest.json").read_text())
    gre = man.get("greedy_continuation") or []
    if not gre:
        return [], [f"{tag}: manifest has no greedy_continuation"]

    rows: list[tuple[np.ndarray, int, int]] = []

    def check(label: str, got, want):
        good = got == want
        if not good:
            notes.append(f"{tag}: ALIGNMENT FAILED {label}: got {got} want {want}")
        return good

    pre = d / "logits_last.npy"
    if pre.exists() and len(gre) >= 2:
        lg = load_row(pre, vocab, declared_dtype(man, "logits_last"))
        if check("argmax(logits_last) vs greedy[0]", int(lg.argmax()), gre[0]):
            rows.append((lg, gre[0], gre[1]))

    s = 0
    while (d / f"decode{s}.logits_last.npy").exists():
        p = d / f"decode{s}.logits_last.npy"
        if s + 2 >= len(gre):
            notes.append(f"{tag}: dropped {p.name}: no greedy[{s + 2}] to compare against "
                         f"(continuation is {len(gre)} long)")
            s += 1
            continue
        lg = load_row(p, vocab, declared_dtype(man, f"decode{s}.logits_last"))
        if check(f"argmax(decode{s}.logits_last) vs greedy[{s + 1}]", int(lg.argmax()), gre[s + 1]):
            rows.append((lg, gre[s + 1], gre[s + 2]))
        inp = d / f"decode{s}.input_ids.npy"
        if inp.exists():
            check(f"decode{s}.input_ids vs greedy[{s}]", int(load_pin(d, man, f"decode{s}.input_ids")[0]), gre[s])
        s += 1

    # S09.1 input contract: `h` is the vector the lm_head sees, i.e. the pinned final norm output.
    pf, fh = d / "pre_final_hidden.npy", d / "final_hidden.npy"
    if pf.exists() and fh.exists():
        # REPORTED, NOT A WITNESS. `pin_pair_search.py` is the witness for the S09.1 input contract:
        # `final_hidden x decode{last}.logits_last` reproduces at cos 0.999989, so `final_hidden` IS
        # the vector the lm_head saw. Recomputing gamma*RMSnorm from the pinned `pre_final_hidden`
        # does NOT reproduce it (cos 0.9987 on dec8, median elementwise difference 22 %), which is the
        # bare-name trap of S10.5 again: the two files are one step apart in a fixture that has decode
        # steps, and a norm recomputation across them is a fact about the pairing, not about the norm.
        want = torch.from_numpy(load_pin(d, man, "final_hidden")).to(torch.float32).reshape(1, -1)
        got = gamma_norm(torch.from_numpy(load_pin(d, man, "pre_final_hidden"))
                         .to(torch.float32).reshape(1, -1).to(torch.bfloat16))
        exact = bool(torch.equal(got.reshape(-1), want.to(torch.bfloat16).reshape(-1)))
        cos = float(torch.nn.functional.cosine_similarity(got.float(), want, dim=-1).item())
        notes.append(f"{tag}: note recomputed norm(pre_final_hidden) vs final_hidden cos={cos:.6f} "
                     f"bit-exact={exact}; not a witness - see pin_pair_search.py for what `h` is")
    if verbose:
        for nn in notes:
            print("  note:", nn)
    return ([] if any("ALIGNMENT FAILED" in n or "does NOT reproduce" in n for n in notes) else rows), notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="", help="comma list; default = every fixture with decode rows")
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    vocab, hidden, ebytes = embed_meta()
    ks = sorted(int(x) for x in args.ks.split(","))
    if fixtures_arg := args.fixtures:
        tags = [t for t in fixtures_arg.split(",") if t]
    else:
        tags = sorted(p.parent.name for p in (ROOT / "data" / "refs").glob("*/decode0.logits_last.npy"))

    strict: list[int] = []
    tie: list[int] = []
    gaps: list[float] = []
    gap_frac: list[float] = []
    per_fix: list[dict] = []
    dropped = 0

    for tag in tags:
        try:
            rows, notes = fixture_rows(tag, vocab, args.verbose)
        except SystemExit as exc:
            print(f"{tag:<16} SKIPPED - unreadable: {exc}")
            continue
        if any("ALIGNMENT FAILED" in n for n in notes):
            print(f"{tag:<16} SKIPPED - alignment failed (rerun with --verbose for the rows)")
            continue
        if not rows:
            print(f"{tag:<16} no usable (row, next-token) pairs")
            continue
        for lg, emitted, want in rows:
            v = lg[want]
            strict.append(int(np.count_nonzero(lg > v)) + 1)
            tie.append(int(np.count_nonzero(lg >= v)))
            top2 = np.partition(lg, lg.size - 2)[-2:]
            top2.sort()
            gaps.append(float(top2[-1] - top2[-2]))
            gap_frac.append(float((top2[-1] - top2[-2]) / (abs(top2[-1]) + 1e-30)))
        per_fix.append({"fixture": tag, "pairs": len(rows)})
        print(f"{tag:<16} pairs={len(rows):>4}")

    if not strict:
        raise SystemExit("no fixture produced a confirmed (row, next-token) pair; nothing to price")
    st = np.array(strict)
    tt = np.array(tie)
    n = st.size
    print(f"\n=== Study A: target-side survival of the NEXT token in the PREVIOUS row's top-k ===")
    print(f"confirmed pairs: {n} over {len(per_fix)} fixtures   vocab={vocab}")
    print(f"rank of the next token inside the previous row (1 = it was the row's own argmax):")
    for name, arr in (("strict", st), ("tie-worst", tt)):
        q = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 99])
        print(f"  {name:<10} median {q[4]:>8.0f}  p75 {q[5]:>8.0f}  p90 {q[6]:>8.0f}  "
              f"p95 {np.percentile(arr, 95):>8.0f}  p99 {q[7]:>8.0f}  max {arr.max():>7.0f}")

    QD = QUOTED
    block_mib = 39.0107           # from mtp_census.py: mtp.* tensors, measured from the shard
    plateau = QD["plateau_gibs"] * 2**30
    # a byte at the copy plateau, in ms: the block walk is 0.161 ms this way and 0.21-0.23 ms at the
    # ledger's measured GEMV efficiency (mlp 70 %, proj 75 %), so treat every c below as a FLOOR.
    def ms_of(mib: float) -> float:
        return mib * 2**20 / plateau * 1e3

    launches_ms = QD["launch_us"] * 18 / 1e3      # one §05 block + MLP + norms + fc, ~18 launches
    print(f"\nsurvival(k) and what each k buys.  head cost = block {block_mib:.2f} MiB "
          f"({ms_of(block_mib):.4f} ms at the plateau, +{QD['launch_us']:.2f} us x 18 launches) + "
          f"k rows x {hidden} x {ebytes} B; a FULL second tied pass is the quoted "
          f"{QD['lm_head_ms']:.4f} ms.")
    print(f"{'k':>7} {'surv_strict':>11} {'surv_tieworst':>13} {'head MiB':>9} {'c_route':>8} "
          f"{'speedup@a=0.642':>15} {'speedup@a=0.50':>15}")
    table = []
    for k in ks:
        s_s = float(np.mean(st <= k))
        s_t = float(np.mean(tt <= k))
        head = block_mib + k * hidden * ebytes / 2**20
        c = (ms_of(head) + launches_ms) / QD["step_ms"]
        print(f"{k:>7} {100*s_s:>10.2f}% {100*s_t:>12.2f}% {head:>9.3f} {c:>8.4f} "
              f"{(1 + 0.642*s_s)/(1 + c):>15.3f} {(1 + 0.50*s_s)/(1 + c):>15.3f}")
        table.append({"k": k, "survival_strict": s_s, "survival_tie_worst": s_t,
                      "head_mib": head, "c": c})
    full_c = (QD["lm_head_ms"] + ms_of(block_mib) + launches_ms) / QD["step_ms"]
    print(f"\nFULL tied-head route: c={full_c:.4f} -> speedup at a=0.642: {1.642/(1+full_c):.3f}x, "
          f"at a=0.50: {1.50/(1+full_c):.3f}x")

    g = np.array(gaps)
    print(f"\n=== decision room of the target's own argmax (does `a` measured on one bit stream "
          f"transfer to another?) ===")
    print(f"top1-top2 logit gap over the same {n} rows: median {np.median(g):.4f}  "
          f"p10 {np.percentile(g,10):.4f}  p1 {np.percentile(g,1):.4f}  min {g.min():.4f}")
    gf = np.array(gap_frac)
    print(f"relative gap (gap/|top1|): median {np.median(gf):.4g}  p10 {np.percentile(gf,10):.4g}  "
          f"p1 {np.percentile(gf,1):.4g}")
    bf16_ulp = 2 ** -8      # relative spacing of BF16 near 1.0
    print(f"a legal 1-ULP BF16 change at the top of a row is ~{bf16_ulp:.4g} relative, so the share "
          f"of rows whose argmax a legal rounding change could flip is at least "
          f"{100.0*float(np.mean(gf < bf16_ulp)):.2f} % (relative gap under one ULP).")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({
            "pairs": int(n), "fixtures": per_fix,
            "rank_strict": {"median": float(np.median(st)), "p90": float(np.percentile(st, 90)),
                            "p99": float(np.percentile(st, 99))},
            "survival": table, "full_route_c": full_c,
            "gap_median": float(np.median(g)), "gap_p10": float(np.percentile(g, 10)),
            "flippable_share": float(np.mean(gf < bf16_ulp)),
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
