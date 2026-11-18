"""
Deliverable 6 - unified inference pipeline for all five models + shared benchmarking engine.

  python inference.py --model all --num-samples 8          # predict random CIFAR-10 test images
  python inference.py --model ptq --image my_picture.png   # predict one image file
Models: fp32 | ptq | qat | unstructured | structured | all   (everything runs on CPU)
"""
import argparse
import csv
import time

import numpy as np
import torch
from PIL import Image

from common import *

REGISTRY = {  # name -> (checkpoint file, kind)
    "fp32": ("resnet18_fp32.pth", "float"),
    "ptq": ("resnet18_ptq_int8.pt", "jit"),
    "qat": ("resnet18_qat_int8.pt", "jit"),
    "unstructured": ("resnet18_unstructured_pruned.pth", "float"),
    "structured": ("resnet18_structured_pruned.pth", "float"),
}


def checkpoint_path(name):
    return CKPT_DIR / REGISTRY[name][0]


def load_model(name):
    path = checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the matching script first (see README).")
    if REGISTRY[name][1] == "float":
        return load_float_model(path)
    set_quant_backend()               # must match the backend used when the INT8 model was built
    return torch.jit.load(str(path), map_location="cpu").eval()


def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean, std = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]
    return ((x - mean) / std).unsqueeze(0)


@torch.no_grad()
def run_predictions(model, X, batch_size=128):
    """Returns predictions, confidences and pure forward-pass time (seconds)."""
    for _ in range(2):                                   # warm-up
        model(X[:batch_size])
    preds, confs, elapsed = [], [], 0.0
    for i in range(0, len(X), batch_size):
        t0 = time.perf_counter()
        out = model(X[i:i + batch_size])
        elapsed += time.perf_counter() - t0
        conf, pred = out.softmax(1).max(1)
        preds.append(pred)
        confs.append(conf)
    return torch.cat(preds), torch.cat(confs), elapsed


@torch.no_grad()
def latency_single(model, X, warmup=20, runs=200):
    """Batch-size-1 latency (ms): mean, median, p95 - the edge-device use case."""
    for i in range(warmup):
        model(X[i % len(X)][None])
    ts = []
    for i in range(runs):
        x = X[i % len(X)][None]
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.median(ts)), float(np.percentile(ts, 95))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def benchmark_models(names, threads=4, extra_fn=None):
    """Evaluate each model on the full CIFAR-10 test set. Returns (summary_rows, prediction_rows)."""
    torch.set_num_threads(threads)
    X, y = load_test_tensors()
    summary = []
    pred_rows = [dict(image_index=i, true_label=int(y[i]), true_class=CLASSES[int(y[i])])
                 for i in range(len(y))]
    for name in names:
        print(f"--- benchmarking {name}")
        model = load_model(name)
        preds, confs, elapsed = run_predictions(model, X)
        mean_ms, p50_ms, p95_ms = latency_single(model, X)
        path = checkpoint_path(name)
        row = dict(model=name, accuracy_pct=round(100 * (preds == y).float().mean().item(), 2),
                   size_mb=round(file_size_mb(path), 3),
                   latency_bs1_mean_ms=round(mean_ms, 3), latency_bs1_p50_ms=round(p50_ms, 3),
                   latency_bs1_p95_ms=round(p95_ms, 3),
                   throughput_img_per_s=round(len(X) / elapsed, 1),
                   total_inference_s=round(elapsed, 2), threads=threads)
        if extra_fn:
            row.update(extra_fn(name, model, path))
        summary.append(row)
        for i, r in enumerate(pred_rows):
            r[f"pred_{name}"] = int(preds[i])
            r[f"conf_{name}"] = round(float(confs[i]), 4)
        print({k: v for k, v in row.items() if k != "model"})
    base = summary[0]
    for r in summary:
        r["accuracy_drop_vs_" + base["model"]] = round(base["accuracy_pct"] - r["accuracy_pct"], 2)
        r["size_reduction_x"] = round(base["size_mb"] / r["size_mb"], 2)
        r["speedup_bs1_x"] = round(base["latency_bs1_mean_ms"] / r["latency_bs1_mean_ms"], 2)
        r["speedup_throughput_x"] = round(r["throughput_img_per_s"] / base["throughput_img_per_s"], 2)
    return summary, pred_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=list(REGISTRY) + ["all"])
    ap.add_argument("--image", default=None)
    ap.add_argument("--num-samples", type=int, default=8)
    args = ap.parse_args()

    names = list(REGISTRY) if args.model == "all" else [args.model]
    if args.image:
        X, truth = preprocess_image(args.image), None
    else:
        Xt, yt = load_test_tensors()
        idx = torch.randperm(len(Xt))[:args.num_samples]
        X, truth = Xt[idx], yt[idx]
    for name in names:
        try:
            model = load_model(name)
        except FileNotFoundError as e:
            print(f"[{name}] skipped: {e}")
            continue
        preds, confs, _ = run_predictions(model, X)
        print(f"\n[{name}]")
        for i in range(len(X)):
            t = f"  (true: {CLASSES[int(truth[i])]})" if truth is not None else ""
            print(f"  sample {i}: {CLASSES[int(preds[i])]:<10s} conf {confs[i]:.2f}{t}")


if __name__ == "__main__":
    main()
