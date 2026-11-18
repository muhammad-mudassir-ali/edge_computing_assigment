#!/usr/bin/env bash
# Verifies the whole pipeline end-to-end in a few minutes on tiny random data (accuracy is meaningless).
set -e
export FAKE_DATA=1
python train_baseline.py --epochs 1
python ptq.py --calib-samples 100
python qat.py --epochs 2
python prune_unstructured.py --iterations 2 --ft-epochs 1
python prune_structured.py --ft-epochs 1
python export_onnx.py
python benchmark_quantization.py
python benchmark_pruning.py
python onnx_inference.py
python inference.py --model all --num-samples 3
python make_report_tables.py
