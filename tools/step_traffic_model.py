#!/usr/bin/env python3
"""Q8: how many bytes does one decode token actually move, and what ceiling does that imply?

CPU-only arithmetic. It takes NO timing and touches NO device: every shape comes from
`engine/state.h` (parsed, not retyped — same reason `engine/weights_contract.py` gives: a shape
copied into a second file is a guess that will be wrong the day the first one changes), and every
bandwidth number it uses is either already committed in `devices/strix-halo-memory-bandwidth.md`
or is passed in from a `tools/mixed_bw` log with `--probe-file`.

Why it exists. Q7 killed the fresh-window explanation for the ~3-6 ms residue between the decode
step and the roofline (M1: the same MLP kernels at a 528 MB cold window run at 93 % of the probe in
isolation and 52 % inside the step). The surviving traffic-shaped candidate is that the step's
WRITE traffic taxes its READ traffic, so the 237 GB/s read roofline is the wrong denominator. This
file decides whether that story can even be true by arithmetic, before any measurement:

  * it counts every byte a token moves, read and written, by family — weights, state, KV, casts,
    logits, glue, activation tiles — so the step's own read:write ratio is a number and not a
    guess ("roughly 20:1" was the starting guess; it is ~60:1, which matters because the more
    read-dominant the step is, the less any write penalty can cost);
  * it prices the residue under the no-cliff (byte-weighted) model: if the memory system charges a
    mix by its byte fractions, what write rate would be needed to produce the observed 12.4 ms?
    This is the cheap falsifier — if it needs an absurd write rate, mixed traffic is dead whatever
    the probe says, and the probe's job becomes confirming the flat curve rather than finding a cliff;
  * it inverts the question into one number the probe can answer directly: the achieved GB/s at the
    step's own ratio and geometry that would put the engine at 90 % of ceiling;
  * it ranks byte-reduction by MB/token with its price in ms, which turns out to be a launch-count
    argument wearing a byte-reduction costume.

Usage:
    tools/step_traffic_model.py                       # budget + parametric ceilings
    tools/step_traffic_model.py --lanes 4032          # long-context row
    tools/step_traffic_model.py --probe-file progress/devices/evidence/q8/mixed_bw.log   # price it with measured rows
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instep_family_budget import state_constants          # noqa: E402  (shared parser, no duplication)

MB = 1e6
MIB = 1048576.0

# Committed device numbers (`devices/strix-halo-memory-bandwidth.md`). The write-only rate has
# never been measured on this box — `bw_probe_hip.cpp` does read and copy only — which is one of
# the things tools/mixed_bw.cpp exists to fix.
READ_GBS = 237.0        # iGPU grid-stride read, best of sweep
COPY_GBS = 210.0        # iGPU copy, 1:1 read+write, at 1 GiB
D2H_GBS = 66.0          # host->device 70.6, device->host 62.8; the logits D2H uses the latter
GDN_STATE_GBS = 876.0   # the 18 MiB F32 recurrent state read warm: cache-resident, not a DRAM rate
SUSPECT_GBS = 240.0     # above this on this box means skipped work or a cache-resident footprint

# The step as it runs today: `engine/results.md` Q6.1 stage-clock medians at 128 lanes. The
# in-situ column is the honest one for "what the step costs"; the burst column is what the isolated
# kernel bursts say the same family should cost.
ENGINE_MS = {128: 12.399, 4032: 15.353}
FAMILY_INSITU_MS = {
    "mlp": 4.260, "proj": 4.237, "lm_head": 2.196, "host_tail": 0.718, "glue": 0.691,
    "gdn": 0.642, "cast": 0.914, "attn": 0.484, "copy": 0.208,
}
LAUNCHES = {"mlp": 48, "proj": 114, "lm_head": 1, "glue": 49, "gdn": 72, "cast": 115,
            "attn": 6, "copy": 19, "host_tail": 0}


def rows(k: dict[str, int], lanes: int) -> list[tuple[str, str, int, int, str]]:
    """(family, direction, bytes, weight-or-state flag, note) for one token, n = 1.

    Every row is derived from state.h shapes. A decode step reads each weight row exactly once
    (x is one row), so weight bytes = 2 * rows * in_features; the activation traffic is derived
    from the launch plan in `engine/decode_step.cpp`'s header comment (49 glue / 114 proj /
    48 mlp / 72 gdn / 115 cast / 20 other = 424 launches).
    """
    hid, ff = k["kHidden"], k["kIntermediate"]
    ng, na, nl = k["kNumGdnLayers"], k["kNumAttnLayers"], k["kNumLayers"]
    qkv, zval, qproj, kvproj, qwidth = (k["kQkvRows"], k["kValueDim"], k["kQprojRows"],
                                        k["kKvProjRows"], k["kQwidth"])
    nh, hd = k["kNumQueryHeads"], k["kHeadDim"]
    kvh = k["kNumKvHeads"]
    cvc, cvk, cvw = k["kConvChannels"], k["kConvKernel"], k["kConvWindow"]
    vh = k["kNumValueHeads"]
    dk, dv = k["kDk"], k["kDv"]
    vocab = k["kVocab"]
    r: list[tuple[str, str, int, int, str]] = []
    add = r.append

    # ---- reads: the weights, which are 96 % of the step -----------------------------------
    mlp_w = 2 * nl * (2 * ff * hid + hid * ff)
    gdn_w = 2 * ng * ((qkv + zval + 2 * vh) * hid + hid * zval)
    attn_w = 2 * na * ((qproj + 2 * kvproj) * hid + hid * qwidth)
    add(("mlp", "read", mlp_w, 1, "24 layers x (gate+up+down) x [3584,1024] bf16"))
    add(("proj", "read", gdn_w, 1, "18 GDN layers: qkv/z/a/b in, out_proj"))
    add(("proj", "read", attn_w, 1, "6 attn layers: q/k/v in, o_proj"))
    add(("lm_head", "read", 2 * vocab * hid, 1, "one row of the tied 248320x1024 bf16 table"))
    add(("norms", "read", 2 * hid * nl * 2 + 2 * hd * na * 2 + hid * 2, 1,
         "input/post-attn norms, q_norm/k_norm [256], final norm"))
    add(("gdn", "read", ng * vh * dk * dv * 4, 1,
         "§08.1 recurrent state [18][16][128][128] F32 read — warm at ~876 GB/s, cache-resident"))
    add(("gdn", "read", ng * cvc * cvk * 4, 0,
         "conv1d weight read as F32: the step widens [6144,1,4] BF16 once into g_conv_w"))
    add(("gdn", "read", ng * cvw * cvc * 4, 0, "conv window (k_conv-1) rows read by conv_kernel"))
    add(("copy", "read", ng * cvw * cvc * 4, 0, "roll-back D2D copy reads conv_new"))
    add(("gdn", "read", ng * (3 * zval) * 4, 0, "conv output tile u: q,k,v as F32 into rec_kernel"))
    add(("gdn", "read", ng * zval * 4 + ng * zval * 2, 0, "gated_norm reads o (F32) and z (BF16)"))
    add(("cast", "read", ng * (hid + (qkv + 2 * vh) + zval + hid + ff) * 2
                     + na * (hid + qwidth + hid + ff) * 2 + hid * 2, 0,
         "the 115 cast inputs, BF16 tiles: norm rows, packed qkv+a+b, block_out, MLP act"))
    add(("proj", "read", (ng * 5 + na * 4) * hid * 4 + nl * ff * 4, 0,
         "GEMV x inputs as F32 (the widening is why this is 4 B/elem, not 2)"))
    add(("mlp", "read", nl * ff * 4, 0, "mlp_down reads mlp_act_f32"))
    add(("glue", "read", 49 * 3 * hid * 2, 0, "add+norm reads s, delta, weight (BF16) x 49"))
    add(("attn", "read", na * 2 * nh * hd * 2, 0, "attn_prep reads q_raw + k_raw (post-norm BF16)"))
    add(("attn", "read", lanes * kvh * hd * 2 * 2 * na, 0,
         f"KV read at {lanes} lanes: 6 layers x (K,V) x 2 heads x 256 x 2 B = 12 KiB/lane-token"))
    add(("gdn", "read", ng * (2 * vh) * 4, 0, "A_log + dt_bias, F32 [16] each"))
    add(("gdn", "read", ng * 128 * 4, 0, "gated-norm weight [128] F32"))

    # ---- writes -----------------------------------------------------------------------------
    add(("gdn", "write", ng * vh * dk * dv * 4, 1, "rec_kernel writes S back: same 18.87 MB it read"))
    add(("gdn", "write", ng * cvw * cvc * 4, 0, "conv_kernel writes conv_new"))
    add(("copy", "write", ng * cvw * cvc * 4, 0, "roll-back D2D copy writes the window"))
    add(("gdn", "write", ng * zval * 4 + ng * vh * 4 * 2, 0, "u tile + g/beta stage rows"))
    add(("gdn", "write", ng * zval * 2, 0, "gated_norm BF16 out"))
    add(("cast", "write", ng * (hid + (qkv + 2 * vh) + zval + hid + ff) * 4
                      + na * (hid + qwidth + hid + ff) * 4 + hid * 4, 0,
         "the 115 cast outputs as F32 — twice the bytes they read"))
    add(("proj", "write", ng * (qkv + zval + 2 * vh + hid) * 2
                     + na * (qproj + 2 * kvproj + hid) * 2, 0,
         "GEMV BF16 outputs: g_raw/g_z/g_a/g_b/delta, q/k/v/delta"))
    add(("mlp", "write", nl * (ff + hid) * 2, 0, "act (BF16) + delta per layer"))
    add(("glue", "write", 49 * 2 * hid * 2, 0, "s_out + norm_out per glue launch"))
    add(("attn", "write", na * (nh * hd + kvh * hd + nh * hd + qwidth) * 2, 0,
         "q_rope, k_rope, gate, gated stage tiles"))
    add(("lm_head", "write", vocab * 4, 0, "logits [248320] F32 — written every token"))
    add(("lm_head", "read", vocab * 4, 0, "the same logits read back by the D2H copy (see D2H rate)"))
    add(("attn", "write", kvh * hd * 2 * 2 * na, 0, "KV append: one lane, 12 KiB/token"))
    add(("copy", "write", hid * 2, 0, "embedding row copy (tied table row -> row 0)"))
    return r


def ms(bytes_: float, gbs: float) -> float:
    """Milliseconds to move `bytes_` at `gbs` GB/s. One place, because getting this wrong by 1000x
    is exactly the kind of error that would make a roofline look like a discovery."""
    return bytes_ / (gbs * 1e6)


def summarize(rows_: list[tuple[str, str, int, int, str]]) -> tuple[int, int]:
    rd = sum(b for _, d, b, _, _ in rows_ if d == "read")
    wr = sum(b for _, d, b, _, _ in rows_ if d == "write")
    return rd, wr


def ms_to_gbs(bytes_: float, ms_: float) -> float:
    """Inverse of ms(): what rate `bytes_` in `ms_` milliseconds is."""
    return bytes_ / (ms_ * 1e6)


def mix_nocliff_gbs(read_frac: float, write_frac: float,
                    read_gbs: float = READ_GBS, write_gbs: float | None = None) -> float:
    """Byte-weighted harmonic mean: what a mix costs if the box charges each stream on its own terms.

    1/BW = f_read/BW_read + f_write/BW_write. That is the NO-CLIFF model — the one that says the
    only thing a mix costs is its own bytes. `tools/mixed_bw` either confirms it or does not.
    """
    if write_gbs is None:
        # No measured write-only rate exists for this box. 1:1 copy is 210 GB/s total, i.e. 105
        # GB/s of read + 105 of write, which puts a pure write stream at ~105-210; the no-cliff
        # bound is only useful if it is GENEROUS to the story being tested, so take the generous end.
        write_gbs = COPY_GBS
    return 1.0 / (read_frac / read_gbs + write_frac / write_gbs)


def parse_probe(path: Path) -> list[dict[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith("MIXEDBW"):
            continue
        fields = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        out.append(fields)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lanes", type=int, default=128, help="context lanes of the step being priced")
    ap.add_argument("--engine-ms", type=float, default=0.0,
                    help="step cost to price against the ceiling (default: the ledger's for --lanes)")
    ap.add_argument("--probe-file", default="", help="file of MIXEDBW lines from tools/mixed_bw")
    ap.add_argument("--read-gbs", type=float, default=READ_GBS)
    args = ap.parse_args(argv)

    k = state_constants()
    rr = rows(k, args.lanes)
    eng_ms = args.engine_ms or ENGINE_MS.get(args.lanes, 12.399)
    rd, wr = summarize(rr)
    tot = rd + wr
    ratio = rd / wr

    print(f"== one decode token, n=1, {args.lanes} lanes: the traffic budget ==")
    print(f"{'family':10} {'dir':6} {'MB/token':>10} {'%total':>7}  where it comes from")
    for fam, d, b, _, note in sorted(rr, key=lambda r: -r[2]):
        if b < 4096:
            continue
        print(f"{fam:10} {d:6} {b/MB:10.3f} {100.0*b/tot:6.2f}%  {note}")
    print(f"{'TOTAL':10} {'read':6} {rd/MB:10.3f} {100.0*rd/tot:6.2f}%")
    print(f"{'':10} {'write':6} {wr/MB:10.3f} {100.0*wr/tot:6.2f}%")
    print(f"{'':10} {'TOTAL':6} {tot/MB:10.3f}   step read:write = {ratio:.1f}:1"
          f"   (writes are {100.0*wr/tot:.2f}% of the bytes)")
    print()
    print("The step is far more read-dominant than the 20:1 the question assumed: every byte-reduction")
    print("argument about activations is therefore arguing over ~1% of the traffic, and any explanation")
    print("of the residue by write traffic has to charge a write tens of times more than a read.")
    print()

    # ---- what the residue would require ---------------------------------------------------
    print(f"== pricing the engine's {eng_ms:.3f} ms/token ==")
    pure_read_ms = ms(rd, args.read_gbs)
    for wg in (COPY_GBS, 105.0):
        bw = mix_nocliff_gbs(rd / tot, wr / tot, args.read_gbs, wg)
        pred = ms(tot, bw)
        print(f"  no-cliff model, reads at {args.read_gbs:.0f} / writes at {wg:.0f} GB/s"
              f"  ->  {bw:6.1f} GB/s  = {pred:6.3f} ms/token"
              f"   the engine reaches {100.0*pred/eng_ms:5.1f}% of that ceiling")
    print(f"  reads alone at {args.read_gbs:.0f} GB/s (writes free)          "
          f"  ->           = {pure_read_ms:6.3f} ms/token"
          f"   the engine reaches {100.0*pure_read_ms/eng_ms:5.1f}% of that ceiling")
    write_ms_avail = eng_ms - pure_read_ms
    write_gbs_needed = ms_to_gbs(wr, write_ms_avail)
    print(f"\n  If reads run at {args.read_gbs:.0f} GB/s, the writes get {write_ms_avail:.3f} ms for "
          f"{wr/MB:.1f} MB:")
    print(f"    that is {write_gbs_needed:.2f} GB/s, i.e. a write would have to cost "
          f"{args.read_gbs/write_gbs_needed:.0f}x what a read costs. No memory system does that to a")
    print(f"    64 B-granular write stream; {COPY_GBS:.0f} GB/s copy traffic (1:1) is already the "
          "measured bound.")
    for pct in (1.0, 0.9):
        need = pct * tot / (eng_ms * 1e6)
        print(f"  the engine sits at {100*pct:.0f}% of ceiling iff the box sustains "
              f"{need:6.1f} GB/s at {ratio:.0f}:1 and this geometry")
    print(f"  ({SUSPECT_GBS:.0f} GB/s is the measured ceiling: anything the probe reports above it is")
    print("   either skipped work or a cache-resident footprint, never a new record.)")
    print()

    # ---- measured rows, if provided -------------------------------------------------------
    if args.probe_file:
        probe_path = Path(args.probe_file)
        probe = parse_probe(probe_path) if probe_path.exists() else []
        if not probe:
            print(f"--probe-file {args.probe_file}: no MIXEDBW lines there (yet). Nothing is priced "
                  f"from measured rows until the sweep in devices/strix-halo-mixed-traffic.md runs.")
        else:
            print("== measured points: predicted step time at each ==")
            print(f"{'tag':22} {'mode':10} {'ratio':>8} {'pages':6} {'gbs':>7} "
                  f"{'pred_ms':>8} {'pct_of_ceil':>11} {'susp':5} {'cksum':10}")
            for f in probe:
                gbs = float(f.get("gbs", 0.0))
                if gbs <= 0:
                    continue
                pred = ms(tot, gbs)
                print(f"{f.get('tag','-'):22} {f.get('mode','-'):10} "
                      f"{float(f.get('ratio_act', 0)):8.2f} {f.get('pages','-'):6} {gbs:7.1f} "
                      f"{pred:8.3f} {100.0*pred/eng_ms:10.1f}% {f.get('suspect','?'):5} "
                      f"{f.get('checksum','-'):10}")
            print()

    # ---- byte-reduction ladder -----------------------------------------------------------
    print("== byte-reduction ladder, ranked by MB/token ==")
    st = k["kNumGdnLayers"] * k["kNumValueHeads"] * k["kDk"] * k["kDv"] * 4
    ladder = [
        ("GDN state read+write", 2 * st, 0,
         "NOT REDUCIBLE: it is the recurrence. F32 is what the reference uses; BF16 would halve it "
         "and re-open §08.1's numerics. Warm it already runs at ~876 GB/s, cache-resident."),
        ("conv window read+write+roll-back copy", 3 * k["kNumGdnLayers"] * k["kConvWindow"]
         * k["kConvChannels"] * 4, 18,
         "kill the roll-back copy by writing the window in place (ring), §08.2's alternative"),
        ("cast round-trip (115 launches)", sum(b for fam, d, b, *_ in rr if fam == "cast"), 115,
         "BF16 staging in mlp/proj deletes ~48 of them; the prize is the launches, not the bytes"),
        ("logits write + D2H", int(k["kVocab"]) * 4 * 2, 0,
         "device argmax would delete the D2H read, but must keep §02.5's tie rule (lowest index) "
         "and the gate's logits dump"),
        ("conv1d weight as F32", k["kNumGdnLayers"] * k["kConvChannels"] * k["kConvKernel"] * 4, 0,
         "read BF16 and convert in-kernel instead of pre-widening: halves this row"),
        ("activation tiles read by GEMVs", sum(b for fam, d, b, wt, _ in rr
                                               if d == "read" and fam in ("proj", "mlp", "glue",
                                                                          "attn")
                                               and b < 1 << 20), 0,
         "the F32 x inputs: BF16 staging halves them"),
        ("KV read (context-dependent)", args.lanes * k["kNumKvHeads"] * k["kHeadDim"] * 2 * 2
         * k["kNumAttnLayers"], 0,
         "the only row that grows with context; at long context the fix is the attn kernel, which "
         "measures ~13.7 GB/s against its own byte count, not fewer bytes"),
    ]
    bw_now = mix_nocliff_gbs(rd / tot, wr / tot, args.read_gbs)
    print(f"{'MB/token':>10} {'ms@ceil':>8} {'launches':>9}  note")
    for name, bytes_, launches, note in sorted(ladder, key=lambda x: -x[1]):
        print(f"{bytes_/MB:10.3f} {ms(bytes_, bw_now):8.4f} {launches:9}  {name}")
        print(f"{'':10} {'':8} {'':9}  {note}")
    conv_bytes = 3 * k["kNumGdnLayers"] * k["kConvWindow"] * k["kConvChannels"] * 4
    nonweight = 2 * st + conv_bytes
    weight_read = sum(b for fam, d, b, wt, _ in rr if d == "read" and wt)
    weight_all = sum(b for fam, d, b, wt, _ in rr if wt)
    print(f"\n  Every non-weight row together is {nonweight/MB:.1f} MB (dominated by the state) = "
          f"{ms(2*st, bw_now):.4f} ms at the no-cliff ceiling.")
    print(f"  Weights are {weight_read/rd*100:.1f}% of what the step reads, and "
          f"{weight_all/tot*100:.1f}% of all traffic.")
    print("  => byte reduction cannot close a multi-millisecond residue at this model size. The")
    print("     only byte lever of the right shape is weight dtype (M3, and not for fitting reasons),")
    print("     which is why the real question is the RATE, and that is tools/mixed_bw.cpp's job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
