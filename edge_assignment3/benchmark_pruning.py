"""Question 2 / Deliverable 7 - benchmark original vs unstructured-pruned vs structured-pruned."""
import argparse

from inference import *


def extras(name, model, path):
    return dict(params_M=round(count_params(model) / 1e6, 3),
                nonzero_params_M=round(count_nonzero_params(model) / 1e6, 3),
                weight_sparsity_pct=round(weight_sparsity(model), 2),
                dense_MACs_M=round(count_macs(model) / 1e6, 1),
                gzip_size_mb=round(gzip_size_mb(path), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    summary, preds = benchmark_models(["fp32", "unstructured", "structured"], a.threads, extras)
    write_csv(RESULTS_DIR / "pruning_benchmark.csv", summary)
    write_csv(RESULTS_DIR / "pruning_predictions.csv", preds)
