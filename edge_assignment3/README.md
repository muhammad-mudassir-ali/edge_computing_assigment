# Assignment 3 - Edge Computing & Intelligent Systems
Model optimisation pipeline for CIFAR-10 / ResNet-18: **PTQ, QAT, unstructured pruning, structured pruning, ONNX deployment.**

## 1. Installation
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt                       # PyTorch >= 2.3 (install the CUDA build for GPU training)
```
CIFAR-10 is downloaded automatically into `data/` on first run. Quantized (INT8) models run on **CPU only**.

## 2. Quick verification (minutes, no download, random data)
```bash
bash smoke_test.sh        # sets FAKE_DATA=1; checks that every script runs end to end
```
Delete `checkpoints/`, `results/`, `onnx/` contents afterwards, then run the real pipeline.

## 3. Full reproducible pipeline
```bash
bash run_all.sh           # or run the steps below one by one
```
| Step | Command | Output |
|---|---|---|
| 1 Baseline FP32 | `python train_baseline.py --epochs 30` | `checkpoints/resnet18_fp32.pth` |
| 2 PTQ (static INT8) | `python ptq.py` | `checkpoints/resnet18_ptq_int8.pt` |
| 3 QAT | `python qat.py --epochs 10` | `checkpoints/resnet18_qat_int8.pt`, `onnx/resnet18_qat_qdq.onnx` |
| 4 Unstructured pruning | `python prune_unstructured.py` | `checkpoints/resnet18_unstructured_pruned.pth`, `results/unstructured_pruning_log.csv` |
| 5 Structured pruning | `python prune_structured.py` | `checkpoints/resnet18_structured_pruned.pth`, `results/structured_pruning_log.csv` |
| 6 ONNX export | `python export_onnx.py` | `onnx/*.onnx` |
| 7 Benchmarks | `python benchmark_quantization.py`, `python benchmark_pruning.py`, `python onnx_inference.py` | `results/*.csv` |
| 8 Report tables | `python make_report_tables.py` | `results/report_tables.md` |

## 4. Inference
```bash
python inference.py --model all --num-samples 8        # fp32, ptq, qat, unstructured, structured
python inference.py --model structured --image cat.png  # any image (resized to 32x32)
python onnx_inference.py --image cat.png                # ONNX Runtime, CPU
```

## 5. File overview
| File | Purpose |
|---|---|
| `common.py` | data, quantization-friendly ResNet-18, fusion, training loop, statistics, checkpoint I/O |
| `train_baseline.py` | FP32 training |
| `ptq.py` / `qat.py` | Q1 scripts (fuse -> qconfig/backend -> calibrate or fake-quant fine-tune -> convert -> save) |
| `prune_unstructured.py` / `prune_structured.py` | Q2 scripts |
| `inference.py` | unified inference for all 5 models + benchmarking engine |
| `benchmark_quantization.py` / `benchmark_pruning.py` | automated benchmarks -> CSV |
| `export_onnx.py` / `onnx_inference.py` | ONNX export and CPU inference |

## 6. Design notes
* **Backend**: `fbgemm` on x86, `qnnpack` on ARM (auto-selected). Load INT8 models on the same CPU family they were built on.
* **INT8 checkpoints** are TorchScript files (`.pt`) - executable without the model class; the quantized `state_dict` is saved alongside.
* **Unstructured**: global L1 magnitude pruning, 5 geometric iterations to 80 % sparsity, fine-tuning after each, masks folded with `prune.remove`. Dense kernels do not get faster from zeros; the benefit is sparsity / compressed size (see `gzip_size_mb`).
* **Structured**: per-block L1 filter ranking on `conv1`; removes `conv1` filters + `bn1` channels + `conv2` input channels and **rebuilds a smaller `ResNet18`** (checkpoint stores `mid_channels`). Residual-stream widths are kept so the skip additions stay valid. `check_equivalence` proves the rebuilt network equals the zero-masked original.
* **ONNX**: FP32 + both pruned models exported directly; INT8 ONNX via ONNX Runtime static quantization of the FP32 graph; QAT exported as a QDQ (fake-quant) graph (best effort, wrapped in try/except).
* Seeds are fixed (`--seed`), but GPU training is not bit-wise deterministic.
