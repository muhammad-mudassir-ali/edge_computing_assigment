"""Question 1 / Deliverable 3 - Quantization-Aware Training initialised from the FP32 model."""
import argparse
import copy

import torch

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    backend = set_quant_backend()
    device = get_device()
    train_loader, test_loader = get_loaders()

    model = load_float_model(CKPT_DIR / "resnet18_fp32.pth")        # pretrained init
    model.train()
    fuse_model(model, is_qat=True)                                   # fuse (train-mode variant)
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(model, inplace=True)           # insert FakeQuantize modules

    bn_freeze = max(1, int(0.4 * args.epochs))                       # freeze BN stats, then observers
    obs_freeze = max(1, int(0.6 * args.epochs))

    def callback(ep, m):
        if ep >= bn_freeze:
            m.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        if ep >= obs_freeze:
            m.apply(torch.ao.quantization.disable_observer)

    fit(model, train_loader, test_loader, device, args.epochs, args.lr, weight_decay=1e-4,
        schedule="cosine", epoch_callback=callback, tag="QAT")

    model.cpu().eval()
    fake_quant_model = copy.deepcopy(model)                          # keep for QDQ-ONNX export
    int8 = torch.ao.quantization.convert(model, inplace=False)       # real INT8 model
    cpu = torch.device("cpu")
    int8_acc = evaluate(int8, test_loader, cpu)

    path = CKPT_DIR / "resnet18_qat_int8.pt"
    save_quantized(int8, path)
    print(f"QAT-INT8 test accuracy {int8_acc:.2f}% | size {file_size_mb(path):.1f}MB | saved {path}")

    try:                                                             # best-effort QDQ ONNX export
        export_onnx_model(fake_quant_model, ONNX_DIR / "resnet18_qat_qdq.onnx")
        print("exported", ONNX_DIR / "resnet18_qat_qdq.onnx")
    except Exception as e:                                           # noqa
        print("QDQ ONNX export skipped:", repr(e)[:200])


if __name__ == "__main__":
    main()
