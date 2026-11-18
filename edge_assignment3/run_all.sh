#!/usr/bin/env bash
# Full reproducible pipeline (GPU recommended for the training steps; ~1 h on a single GPU).
set -e
python train_baseline.py --epochs 30
python ptq.py
python qat.py --epochs 10
python prune_unstructured.py --target-sparsity 0.8 --iterations 5 --ft-epochs 3
python prune_structured.py --prune-ratio 0.5 --ft-epochs 15
python export_onnx.py
python benchmark_quantization.py
python benchmark_pruning.py
python onnx_inference.py
python make_report_tables.py
