#!/usr/bin/env python3
"""Fixtures for gemm_algo_sweep --selftest-sys. Hand-authored, NOT a device capture: their job is to
show the predicates fire on a legal list and on each illegal one. The real audit input is
`--verify --algo-dump`. Asserts the algo field is exactly sizeof(hipblasLtMatmulAlgo_t)=24 bytes,
because the tool rejects any other length and a fixture that silently mismatches proves nothing.
Also writes the floor-calibration fixtures (--calibration), whose parser must be tested too.
"""
import struct

def algo_bytes(data16: bytes, max_ws: int) -> str:
    assert len(data16) == 16, len(data16)
    raw = data16 + struct.pack("<Q", max_ws)
    assert len(raw) == 24, len(raw)
    return raw.hex()

def line(shape, M, N, K, seq, idx, state=0, sup=0, supp=1, ws=0, max_ws=0, sol="Cijk_11507_bf16_A_A",
         data16=b"\x01\x00\x00\x00\x24\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00"):
    s = (f"ALGODUMP shape={shape} M={M} N={N} K={K} seq={seq} idx={idx} heur_rank={seq} "
         f"state={state} support_status={sup} supported={supp} ws={ws} heur_ws={ws} "
         f"data={algo_bytes(data16, max_ws)} solution={sol}")
    assert len(s.split("data=")[1].split(" ")[0]) == 48
    return s

ok = [
    line("q_proj", 512, 4096, 1024, 0, 36),
    line("q_proj", 512, 4096, 1024, 1, 44, ws=8388608, max_ws=8388608, sol="TensileAsset_bf16_splitk"),
    line("k_proj+v_proj", 512, 512, 1024, 0, 36),
    line("down_proj", 512, 1024, 3584, 0, 36, ws=134217728, max_ws=134217728),
]

# Each line is the shape of bad argument list the launch site must refuse; --selftest-sys has to name
# exactly one violation tag per line, so the count is a check, not a hope.
bad = [
    line("q_proj", 512, 4096, 1024, 0, 36, state=5, sol="Cijk_state_not_success"),
    line("q_proj", 512, 4096, 1024, 1, 44, data16=b"\x00" * 16, sol="TensileAsset_algo_all_zero"),
    line("q_proj", 512, 4096, 1024, 2, 51, ws=268435456, sol="TensileAsset_ws_request_over_allocation"),
    line("q_proj", 512, 4096, 1024, 3, 52, sup=14, sol="Cijk_support_status_invalid_value"),
    line("in_proj_qkv", 512, 6144, 1024, 0, 36, max_ws=268435456, sol="Cijk_algo_max_ws_over_allocation"),
]

open("/tmp/fx_ok.txt", "w").write("\n".join(ok) + "\n")
open("/tmp/fx_bad.txt", "w").write("\n".join(bad) + "\n")
print(f"wrote /tmp/fx_ok.txt ({len(ok)} lines) and /tmp/fx_bad.txt ({len(bad)} lines)")

# --------------------------------------------------------------------------- calibration fixtures
# The floor file --time reads its cos floor from. Hand-authored here so the parse, the key match and
# the refusal are all testable without a device; the real one comes from `--floor-probe`.
#
# The numbers are the shape of what the first device window measured (worst cos 0.9999979743, margin
# 1e-05) and of what mlp_gemv's cancellation band came out at (worst_off_rel 8.2e-06 against a band of
# 3e-05), not real measurements of this build; the STRUCTURE is what these fixtures test.
# mustcatch_max_cos 0.98 is the hardest REACHABLE injection (a dropped split-K partial), which sits
# about 4 orders of magnitude below the floor - the headroom v1's one-term-short injection lacked.
CENSUS = [
    ("q_proj", 4096, 1024), ("k_proj+v_proj", 512, 1024), ("o_proj+out_proj", 1024, 2048),
    ("in_proj_qkv", 6144, 1024), ("in_proj_z", 2048, 1024), ("gate_proj+up_proj", 3584, 1024),
    ("down_proj", 1024, 3584),
]
ROWS = [512, 1024, 4096]
CHECK_ROWS = [8, 16]


def cal(shape, M, N, K, check_rows, seeds=4, floor=0.999987974322, version=2,
        cancel_rel=3.3e-05, cancel_tol=2.4e-05):
    return (f"CAL version={version} shape={shape} M={M} N={N} K={K} check_rows={check_rows} "
            f"seeds={seeds} worst_algo_cos=0.999997974322 worst_engine_cos=0.999997974322 "
            f"ref_reorder_worst_cos=0.999997974322 license_worst_cos=0.999997974322 "
            f"quant_cos_deficit=2.026e-06 ndiff_spread=79-236 margin=1.000000e-05 "
            f"cos_floor={floor:.12f} cancel_rel={cancel_rel:.6e} cancel_tol={cancel_tol:.6e} "
            f"license_worst_off_rel=8.200e-06 license_n_off=4 license_worst_excess=6.000e-06 "
            f"device_worst_off_rel=5.900e-06 device_n_off=9 device_n_off_in_band=9 "
            f"device_worst_excess=1.100e-05 mustcatch_max_cos=0.980000000 "
            f"one_term_short_cos=0.999998400 one_term_over_quant=0.79 device=\"Radeon 8060S\"")


full = [cal(s, M, N, K, cr) for cr in CHECK_ROWS for M in ROWS for (s, N, K) in CENSUS]
# Complete minus exactly one key: if the tool's own parser were broken, it would report all 21 missing
# instead of one, so this fixture distinguishes "parsed and matched" from "parsed nothing".
partial = [ln for ln in full
           if not ("shape=q_proj " in ln and " M=512 " in ln and "check_rows=16" in ln)]
# An unuseable threshold. cos_floor=0.0 makes `cos >= floor` unfailable and cancel_rel=0.0 excuses no
# cancellation at all, so each of these has to be REFUSED rather than skipped - a row the parser walks
# past is a row whose shape then gets judged by nothing.
badrow = full[:2] + [cal("o_proj+out_proj", 512, 1024, 2048, 16, floor=0.0)] + full[3:]
badband = full[:2] + [cal("o_proj+out_proj", 512, 1024, 2048, 16, cancel_rel=0.0)] + full[3:]
# A v1-format row, which is what is on disk from the previous window. It must be refused as a format
# mismatch, not accepted: v1 never measured a cancellation band, so any default for it would be a
# threshold nobody measured.
v1line = ('CAL shape=q_proj M=512 N=4096 K=1024 check_rows=8 seeds=4 '
          'worst_algo_cos=0.999997974322 worst_engine_cos=0.999997974322 '
          'ref_reorder_worst_cos=0.999997974322 worst_ulp=1.0000 mustcatch_max_cos=0.999999000 '
          'ndiff_spread=79-236 margin=1.000000e-05 cos_floor=0.999987974322 device="Radeon 8060S"')

open("/tmp/cal_full.tsv", "w").write("\n".join(full) + "\n")
open("/tmp/cal_partial.tsv", "w").write("\n".join(partial) + "\n")
open("/tmp/cal_badrow.tsv", "w").write("\n".join(badrow) + "\n")
open("/tmp/cal_badband.tsv", "w").write("\n".join(badband) + "\n")
open("/tmp/cal_v1.tsv", "w").write(v1line + "\n")
print(f"wrote /tmp/cal_full.tsv ({len(full)} CAL lines), /tmp/cal_partial.tsv ({len(partial)}, "
      f"missing q_proj|M=512|check_rows=16), /tmp/cal_badrow.tsv (cos_floor=0), "
      f"/tmp/cal_badband.tsv (cancel_rel=0), /tmp/cal_v1.tsv (one v1-format row)")

