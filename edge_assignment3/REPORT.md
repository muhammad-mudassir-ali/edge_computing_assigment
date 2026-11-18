# Technical Report - Assignment 3: Edge Computing & Intelligent Systems
**Name:** ________  **Roll No:** ________  **Section:** ____

> Fill every `<...>` from `results/*.csv` (run `python make_report_tables.py`), add screenshots of the terminal output, then export to PDF/Word.

## 1. Setup
Hardware (CPU/GPU, RAM), PyTorch / ONNX Runtime versions, quantization backend, threads, epochs, seed: `<...>`

## 2. Baseline (FP32)
CIFAR-10 CIFAR-style ResNet-18 (3x3 stem, no max-pool, 11.2 M params), SGD+Nesterov, OneCycle LR 0.1, 30 epochs, crop+flip augmentation. Test accuracy: `<..>%`. Screenshot: `<training log>`

## 3. Quantization (Question 1)
**3.1 PTQ** - fusion of Conv+BN(+ReLU) (stem, both convs of every block, 1x1 shortcuts), FloatFunctional for residual adds, default static qconfig for the backend, calibration on `<1000>` training images, `convert()` to INT8.
**3.2 QAT** - initialised from FP32, `fuse_modules_qat`, `prepare_qat` (FakeQuantize), `<10>` epochs SGD lr 0.01 cosine, BN statistics frozen after 40 % and observers after 60 % of epochs, `convert()`.

| Model | Acc % | Size MB | Latency bs1 ms | Throughput img/s | Size reduction | Speed-up |
|---|---|---|---|---|---|---|
| FP32 | | | | | 1.0x | 1.0x |
| PTQ-INT8 | | | | | | |
| QAT-INT8 | | | | | | |

Discussion: accuracy drop of PTQ vs QAT, why QAT recovers accuracy (weights adapt to rounding noise), ~4x size reduction, speed-up source (INT8 kernels, fused ops). `<...>`

## 4. Pruning (Question 2)
**4.1 Unstructured** - global L1 magnitude pruning of all Conv2d/Linear weights, 5 iterations to 80 % sparsity (geometric schedule), 3 fine-tuning epochs after each, `prune.remove` for permanence. Iteration log: `<paste unstructured_pruning_log.csv>`
**4.2 Structured** - L1 filter importance in each block's `conv1`; removing filter k also removes `bn1[k]` and `conv2` input channel k; rebuilt smaller architecture `<mid channels>`; equivalence check error `<..>`; 15 fine-tuning epochs.

| Model | Acc % | Params M | Non-zero M | Sparsity % | MACs M | Size MB | gzip MB | Latency bs1 ms | Throughput |
|---|---|---|---|---|---|---|---|---|---|
| Original | | | | | | | | | |
| Unstructured | | | | | | | | | |
| Structured | | | | | | | | | |

Discussion: unstructured pruning gives high sparsity/compressibility but no dense-CPU speed-up; structured pruning gives real MAC/latency reduction at larger accuracy cost; trade-offs for edge devices. `<...>`

## 5. ONNX deployment
Exported models, ORT CPU accuracy/latency (`results/onnx_benchmark.csv`), parity check vs PyTorch: `<...>`

## 6. Conclusions & limitations
`<...>`
