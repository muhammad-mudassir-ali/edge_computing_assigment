"""Turns results/*.csv into Markdown tables ready to paste into the technical report."""
import csv

from common import RESULTS_DIR

out = []
for name in ("quantization_benchmark", "pruning_benchmark", "onnx_benchmark",
             "unstructured_pruning_log", "structured_pruning_log"):
    p = RESULTS_DIR / f"{name}.csv"
    if not p.exists():
        continue
    rows = list(csv.reader(open(p)))
    out.append(f"### {name}\n\n| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]))
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    out.append("")
(RESULTS_DIR / "report_tables.md").write_text("\n".join(out))
print("\n".join(out))
