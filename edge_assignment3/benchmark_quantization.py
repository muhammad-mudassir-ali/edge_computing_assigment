"""Question 1 / Deliverable 7 - benchmark FP32 vs PTQ-INT8 vs QAT-INT8 on the CIFAR-10 test set."""
import argparse

from inference import *

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    summary, preds = benchmark_models(["fp32", "ptq", "qat"], a.threads)
    write_csv(RESULTS_DIR / "quantization_benchmark.csv", summary)
    write_csv(RESULTS_DIR / "quantization_predictions.csv", preds)
