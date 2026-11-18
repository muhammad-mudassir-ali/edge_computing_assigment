"""Deliverable 8 - CPU inference + benchmark for every ONNX model with ONNX Runtime.

  python onnx_inference.py                     # benchmark all onnx/*.onnx on CIFAR-10 test set
  python onnx_inference.py --image cat.png     # classify one image with every ONNX model
"""
import argparse
import time

import numpy as np
import onnxruntime as ort

from common import *
from inference import preprocess_image, write_csv


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    models = sorted(ONNX_DIR.glob("*.onnx"))
    if not models:
        raise SystemExit("no ONNX models found - run export_onnx.py (and qat.py) first")

    if args.image:
        x = preprocess_image(args.image).numpy()
        for p in models:
            logits = session(p, args.threads).run(None, {"input": x})[0][0]
            prob = np.exp(logits - logits.max()); prob /= prob.sum()
            print(f"{p.name:34s} -> {CLASSES[int(prob.argmax())]} ({prob.max():.2f})")
        return

    X, y = load_test_tensors()
    X, y = X.numpy(), y.numpy()
    rows = []
    for p in models:
        s = session(p, args.threads)
        s.run(None, {"input": X[:128]})                                  # warm-up
        preds, t_total = [], 0.0
        for i in range(0, len(X), 128):
            t0 = time.perf_counter()
            out = s.run(None, {"input": X[i:i + 128]})[0]
            t_total += time.perf_counter() - t0
            preds.append(out.argmax(1))
        lat = []
        for i in range(200):
            t0 = time.perf_counter()
            s.run(None, {"input": X[i % len(X):i % len(X) + 1]})
            lat.append((time.perf_counter() - t0) * 1000)
        rows.append(dict(onnx_model=p.name, accuracy_pct=round(100 * float((np.concatenate(preds) == y).mean()), 2),
                         size_mb=round(file_size_mb(p), 3), latency_bs1_mean_ms=round(float(np.mean(lat)), 3),
                         throughput_img_per_s=round(len(X) / t_total, 1), threads=args.threads))
        print(rows[-1])
    write_csv(RESULTS_DIR / "onnx_benchmark.csv", rows)


if __name__ == "__main__":
    main()
