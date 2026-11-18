"""Deliverable 1 - train the FP32 ResNet-18 baseline on CIFAR-10 and save the checkpoint."""
import argparse

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    train_loader, test_loader = get_loaders(args.batch_size)
    model = ResNet18()
    print(f"device={device}  params={count_params(model) / 1e6:.2f}M")
    hist = fit(model, train_loader, test_loader, device, args.epochs, args.lr,
               schedule="onecycle", amp=True, tag="FP32")
    model.cpu()
    path = CKPT_DIR / "resnet18_fp32.pth"
    save_float_model(model, path)
    print(f"Final test accuracy: {hist[-1]['test_acc']:.2f}%  -> saved {path}")


if __name__ == "__main__":
    main()
