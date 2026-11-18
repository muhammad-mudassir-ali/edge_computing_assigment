"""Question 2 / Deliverable 5 - structured channel pruning with physical architecture reconstruction.

Strategy (Li et al., "Pruning Filters for Efficient ConvNets"): in every residual block rank the
filters of conv1 by L1-norm and keep the strongest ones. Removing conv1 filter k forces removal of
  * bn1 channel k,   * input channel k of conv2.
The block's output width is unchanged, so the residual additions stay shape-consistent.
A NEW, physically smaller ResNet18 is built and the surviving weights are copied into it.
"""
import argparse
import csv

import torch

from common import *


def build_pruned_model(model: ResNet18, ratio: float, min_keep: int = 8):
    blocks = get_blocks(model)
    keep_idx, new_mid = [], []
    for b in blocks:
        scores = b.conv1.weight.detach().abs().sum(dim=(1, 2, 3))          # L1 norm per filter
        n_keep = max(min_keep, int(round(scores.numel() * (1 - ratio))))
        keep = torch.argsort(scores, descending=True)[:n_keep].sort().values
        keep_idx.append(keep)
        new_mid.append(n_keep)

    pruned = ResNet18(mid_channels=new_mid)
    for name in ("conv1", "bn1", "fc"):                                     # untouched layers
        getattr(pruned, name).load_state_dict(getattr(model, name).state_dict())
    for old, new, keep in zip(blocks, get_blocks(pruned), keep_idx):
        with torch.no_grad():
            new.conv1.weight.copy_(old.conv1.weight[keep])
            for attr in ("weight", "bias", "running_mean", "running_var"):
                getattr(new.bn1, attr).copy_(getattr(old.bn1, attr)[keep])
            new.bn1.num_batches_tracked.copy_(old.bn1.num_batches_tracked)
            new.conv2.weight.copy_(old.conv2.weight[:, keep])
        new.bn2.load_state_dict(old.bn2.state_dict())
        new.shortcut.load_state_dict(old.shortcut.state_dict())
    return pruned, keep_idx, new_mid


@torch.no_grad()
def check_equivalence(model, pruned, keep_idx):
    """Zero-masked original must give the same logits as the physically smaller model."""
    import copy
    masked = copy.deepcopy(model).eval()
    for b, keep in zip(get_blocks(masked), keep_idx):
        drop = torch.ones(b.conv1.out_channels, dtype=torch.bool)
        drop[keep] = False
        b.conv1.weight[drop] = 0
        b.bn1.weight[drop] = 0
        b.bn1.bias[drop] = 0
    x = torch.randn(8, 3, 32, 32)
    return (masked(x) - pruned.eval()(x)).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune-ratio", type=float, default=0.5, help="fraction of conv1 filters removed per block")
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    cpu = torch.device("cpu")
    train_loader, test_loader = get_loaders()

    model = load_float_model(CKPT_DIR / "resnet18_fp32.pth")
    base_acc = evaluate(model, test_loader, cpu)
    pruned, keep_idx, new_mid = build_pruned_model(model, args.prune_ratio)
    print("mid-channels per block:", model.mid_channels, "->", new_mid)
    print(f"max |logit diff| masked-vs-reconstructed: {check_equivalence(model, pruned, keep_idx):.2e}")

    acc_before = evaluate(pruned, test_loader, cpu)
    p0, p1 = count_params(model), count_params(pruned)
    m0, m1 = count_macs(model), count_macs(pruned)
    print(f"params {p0 / 1e6:.2f}M -> {p1 / 1e6:.2f}M | MACs {m0 / 1e6:.0f}M -> {m1 / 1e6:.0f}M | "
          f"acc {base_acc:.2f}% -> {acc_before:.2f}% (before fine-tuning)")

    hist = fit(pruned, train_loader, test_loader, device, args.ft_epochs, args.lr, tag="struct-ft")
    pruned.cpu()
    final_acc = evaluate(pruned, test_loader, cpu)
    path = CKPT_DIR / "resnet18_structured_pruned.pth"
    save_float_model(pruned, path)           # stores mid_channels so the slim architecture can be rebuilt

    with open(RESULTS_DIR / "structured_pruning_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    print(f"FINAL: accuracy {final_acc:.2f}% (baseline {base_acc:.2f}%) | saved {path}")


if __name__ == "__main__":
    main()
