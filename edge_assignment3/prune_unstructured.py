"""Question 2 / Deliverable 4 - iterative global magnitude (L1) unstructured pruning + fine-tuning."""
import argparse
import csv

import torch.nn as nn
import torch.nn.utils.prune as prune

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-sparsity", type=float, default=0.8)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--ft-epochs", type=int, default=3, help="fine-tune epochs per iteration")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    train_loader, test_loader = get_loaders()
    model = load_float_model(CKPT_DIR / "resnet18_fp32.pth").to(device)
    base_acc = evaluate(model, test_loader, device)
    print(f"baseline accuracy {base_acc:.2f}%")

    targets = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    # Geometric schedule: after n steps the remaining fraction is (1 - target).
    # PyTorch applies `amount` to the still-unpruned weights at every call.
    step_amount = 1 - (1 - args.target_sparsity) ** (1 / args.iterations)

    log = []
    for it in range(1, args.iterations + 1):
        prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=step_amount)
        sp = weight_sparsity(model)                       # masks are applied to `weight`
        acc_pruned = evaluate(model, test_loader, device)
        hist = fit(model, train_loader, test_loader, device, args.ft_epochs, args.lr,
                   tag=f"prune-it{it}")                   # masks stay active -> zeros stay zero
        log.append(dict(iteration=it, sparsity_pct=round(sp, 2),
                        acc_after_pruning=round(acc_pruned, 2),
                        acc_after_finetune=round(hist[-1]["test_acc"], 2)))
        print(f"== iteration {it}: sparsity {sp:.1f}% | {acc_pruned:.2f}% -> {hist[-1]['test_acc']:.2f}%")

    for m, name in targets:                               # make pruning permanent (fold mask)
        prune.remove(m, name)
    assert not any(n.endswith("weight_orig") for n, _ in model.named_parameters())

    model.cpu()
    final_acc = evaluate(model, test_loader, torch.device("cpu"))
    path = CKPT_DIR / "resnet18_unstructured_pruned.pth"
    save_float_model(model, path)
    with open(RESULTS_DIR / "unstructured_pruning_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"FINAL: sparsity {weight_sparsity(model):.2f}% | accuracy {final_acc:.2f}% "
          f"(baseline {base_acc:.2f}%) | saved {path}")


if __name__ == "__main__":
    main()
