"""Question 1 / Deliverable 2 - Post-Training Static Quantization (FP32 -> INT8)."""
import argparse

import torch

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-samples", type=int, default=1000)
    args = ap.parse_args()

    seed_everything(0)
    cpu = torch.device("cpu")
    backend = set_quant_backend()                                   # 1. backend
    print("quantization backend:", backend)
    _, test_loader = get_loaders(num_workers=0)

    model = load_float_model(CKPT_DIR / "resnet18_fp32.pth").eval()  # 2. pretrained FP32
    fp32_acc = evaluate(model, test_loader, cpu)

    fuse_model(model, is_qat=False)                                  # 3. Conv+BN+ReLU fusion
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(model, inplace=True)               # 4. insert observers

    calib = get_calibration_loader(args.calib_samples)               # 5. calibration (train subset)
    with torch.no_grad():
        for x, _ in calib:
            model(x)
    print(f"calibrated on {len(calib.dataset)} training images")

    int8 = torch.ao.quantization.convert(model, inplace=False)       # 6. -> INT8
    int8_acc = evaluate(int8, test_loader, cpu)

    path = CKPT_DIR / "resnet18_ptq_int8.pt"                         # 7. save for deployment
    save_quantized(int8, path)
    print(f"FP32 acc {fp32_acc:.2f}% | PTQ-INT8 acc {int8_acc:.2f}% | "
          f"size {file_size_mb(CKPT_DIR / 'resnet18_fp32.pth'):.1f}MB -> {file_size_mb(path):.1f}MB")
    print("saved", path)


if __name__ == "__main__":
    main()
