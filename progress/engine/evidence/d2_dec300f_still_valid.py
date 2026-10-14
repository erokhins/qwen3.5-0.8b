"""Check whether the already-priced `dec300f` route table still matches today's gate log.

`engine/route_price.py` is a multi-minute torch CPU run per fixture, and `AGENTS.md` says to cite a
measurement already on disk rather than re-make it when nothing that feeds it moved. What could have
moved: the ENGINE column, which this script reads out of a gate log. The route column comes from the
reference, and `route_allowance.py` re-derives the provenance hashes at write time from the fixture and
`speccheck/specimpl.py`, not from this table.

So: compare every step's `engine_one_cos` / `engine_meanrel` in progress/engine/evidence/route_dec300f.json against the same
steps parsed out of the fresh gate log with route_price's own regex.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import route_price as rp   # noqa: E402  (the parser that produced the table in the first place)

table = json.loads((ROOT / "tmp" / "route_dec300f.json").read_text())
log = ROOT / "tmp" / "d2_gate_dec300f.log"
eng = rp.engine_metrics(log)["steps"]
print(f"table: fixture={table['fixture']} rows={len(table['rows'])}")
print(f"fresh log: {log.name} steps parsed={len(eng)}")

bad = []
for r in table["rows"]:
    j = int(r["step"])
    if j not in eng:
        bad.append(f"step {j}: not in the fresh log")
        continue
    if abs(eng[j]["one_cos"] - r["engine_one_cos"]) > 0:
        bad.append(f"step {j}: table {r['engine_one_cos']!r} vs fresh {eng[j]['one_cos']!r}")
    if abs(eng[j]["meanrel"] - r["engine_meanrel"]) > 0:
        bad.append(f"step {j}: meanrel table {r['engine_meanrel']!r} vs fresh {eng[j]['meanrel']!r}")
print("engine column identical at every step:", not bad)
for line in bad[:10]:
    print("  ", line)

worst_c = max(r["engine_one_cos"] / r["route_one_cos"] for r in table["rows"])
worst_r = max(r["engine_meanrel"] / r["route_meanrel"] for r in table["rows"])
print(f"worst cos ratio {worst_c:.4f} -> ceil2 {__import__('math').ceil(worst_c*100)/100:.2f};"
      f" worst meanrel ratio {worst_r:.4f} -> ceil2"
      f" {__import__('math').ceil(worst_r*100)/100:.2f}")
