#!/usr/bin/env python3
"""Witness that the rewritten check_source() can actually fail — a check that cannot fail is decoration.

Runs check_source() against scratch copies of the two headers, with five mutations:
  A  both headers as they now stand                     -> expect PASS
  B  one of the four frozen sites reverted to `t < nd`  -> expect FAIL (regressed)
  C  all four frozen sites reverted                     -> expect FAIL (regressed)
  D  an unrelated edit in the variant body              -> expect FAIL (source drift)
  E  a line deleted from the variant body               -> expect FAIL (line count)
Nothing here touches the device or the tree: the real headers are copied to tmp/ and only the copies
are mutated. CPU only, no timing.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "attn_decode_wide"))
import gate  # noqa: E402

SRC_FROZEN = ROOT / "prototype" / "attn_decode" / "attn_decode_kernels.h"
SRC_VARIANT = ROOT / "prototype" / "attn_decode_wide" / "attn_decode_wide_kernels.h"
WORK = ROOT / "tmp" / "d3_check_source_witness"   # scratch copies only; the real headers are never touched

LIT = "for (int t = 0; t < 8; ++t) if (t < nd) dot += "
RT = "for (int t = 0; t < nd; ++t) dot += "


def revert(text: str, how_many: int) -> str:
    """Turn the last `how_many` literal-trip dot sites back into a runtime-bound loop. Matching on the
    loop header alone covers both bodies (self-row `sq*sk` and cache-row `bf(k[t])*qreg[t]`)."""
    out, done = [], 0
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines) if LIT in l]
    want = set(hits[-how_many:])
    for i, l in enumerate(lines):
        out.append(l.replace(LIT, RT) if i in want else l)
        done += 1 if i in want else 0
    assert done == how_many, f"only reverted {done} of {how_many}"
    return "\n".join(out) + "\n"


def build(mutate_frozen=None, mutate_variant=None) -> int:
    (WORK / "op").mkdir(parents=True, exist_ok=True)
    (WORK / "wide").mkdir(parents=True, exist_ok=True)
    f = SRC_FROZEN.read_text()
    v = SRC_VARIANT.read_text()
    if mutate_frozen:
        f = mutate_frozen(f)
    if mutate_variant:
        v = mutate_variant(v)
    (WORK / "op" / "attn_decode_kernels.h").write_text(f)
    (WORK / "wide" / "attn_decode_wide_kernels.h").write_text(v)
    gate.OP, gate.HERE = WORK / "op", WORK / "wide"
    return gate.check_source()


CASES = [
    ("A as shipped", None, None, 0),
    ("B one site reverted", lambda t: revert(t, 1), None, 1),
    ("C all four reverted", lambda t: revert(t, 4), None, 1),
    ("D variant body edited", None, lambda t: t.replace("    const int qrow = blockIdx.z;",
                                                        "    const int qrow = blockIdx.z * 1;"), 1),
    ("E variant line deleted", None,
     lambda t: t.replace("    const float pos = (float)row_pos;\n", ""), 1),
]

bad = 0
for name, mf, mv, want in CASES:
    got = build(mf, mv)
    ok = (got != 0) == (want != 0)
    bad += 0 if ok else 1
    print(f"\n[{name}] expected {'FAIL' if want else 'PASS'}, got {'FAIL' if got else 'PASS'} "
          f"-> {'ok' if ok else 'THE CHECK CANNOT FAIL'}\n")
print(f"\n{'WITNESS FAIL' if bad else 'WITNESS PASS'}: {len(CASES) - bad}/{len(CASES)} cases behaved "
      f"as required (a mutation that does not turn check_source red is a decoration)")
sys.exit(1 if bad else 0)
