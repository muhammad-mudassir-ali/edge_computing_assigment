"""
common.py - shared utilities for Assignment 3 (Edge Computing & Intelligent Systems)

Contains: paths, data loading (CIFAR-10), a quantization-friendly CIFAR ResNet-18,
training/evaluation helpers, layer fusion, model statistics, checkpoint I/O.

Set the environment variable FAKE_DATA=1 to run the whole pipeline on tiny random
data (smoke test only - no internet / no CIFAR-10 download needed).
"""
import os
import platform
import random
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub
from torch.utils.data import DataLoader, Subset

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
ONNX_DIR = ROOT / "onnx"
for _d in (CKPT_DIR, RESULTS_DIR, ONNX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------------- misc
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_quant_backend() -> str:
    """Select the quantized-kernel backend: fbgemm/x86 on PCs, qnnpack on ARM."""
    machine = platform.machine().lower()
    wanted = "qnnpack" if ("arm" in machine or "aarch64" in machine) else "fbgemm"
    supported = torch.backends.quantized.supported_engines
    if wanted not in supported:
        wanted = "x86" if "x86" in supported else "qnnpack"
    torch.backends.quantized.engine = wanted
    return wanted


# ----------------------------------------------------------------------------- data
def _dataset(train: bool, transform):
    if os.environ.get("FAKE_DATA") == "1":
        return torchvision.datasets.FakeData(
            size=1024 if train else 512, image_size=(3, 32, 32), num_classes=10,
            transform=transform, random_offset=0 if train else 10_000)
    return torchvision.datasets.CIFAR10(str(DATA_DIR), train=train, download=True,
                                        transform=transform)


def _tf(train: bool):
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    return T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])


def get_loaders(batch_size=128, num_workers=2):
    train = DataLoader(_dataset(True, _tf(True)), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True)
    test = DataLoader(_dataset(False, _tf(False)), batch_size=256, shuffle=False,
                      num_workers=num_workers)
    return train, test


def get_calibration_loader(num_samples=1000, batch_size=50, seed=0):
    """Random subset of the TRAIN set (no augmentation) used to calibrate observers."""
    ds = _dataset(True, _tf(False))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:min(num_samples, len(ds))].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)


def load_test_tensors():
    """Whole CIFAR-10 test set as (X float32 [N,3,32,32], y int64 [N]) tensors."""
    loader = DataLoader(_dataset(False, _tf(False)), batch_size=512, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


# ----------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    """ResNet basic block. `mid` = channels between conv1 and conv2 (prunable)."""

    def __init__(self, in_planes, planes, stride=1, mid=None):
        super().__init__()
        mid = mid or planes
        self.conv1 = nn.Conv2d(in_planes, mid, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        self.skip_add = FloatFunctional()   # quantization-friendly residual add
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.skip_add.add(out, self.shortcut(x))
        return self.relu2(out)


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18 (3x3 stem, no max-pool) with Quant/DeQuant stubs."""
    DEFAULT_MID = [64, 64, 128, 128, 256, 256, 512, 512]

    def __init__(self, num_classes=10, mid_channels=None):
        super().__init__()
        self.mid_channels = list(mid_channels or self.DEFAULT_MID)
        self.quant, self.dequant = QuantStub(), DeQuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        in_planes, k = 64, 0
        for i, (planes, stride) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)], 1):
            blocks = []
            for s in (stride, 1):
                blocks.append(BasicBlock(in_planes, planes, s, self.mid_channels[k]))
                in_planes, k = planes, k + 1
            setattr(self, f"layer{i}", nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.dequant(self.fc(x))


def get_blocks(model):
    return [m for m in model.modules() if isinstance(m, BasicBlock)]


def fuse_model(model: ResNet18, is_qat: bool = False):
    """Fuse Conv+BN(+ReLU). PTQ: model in eval(); QAT: model in train()."""
    fuse = (torch.ao.quantization.fuse_modules_qat if is_qat
            else torch.ao.quantization.fuse_modules)
    fuse(model, [["conv1", "bn1", "relu"]], inplace=True)
    for name, m in list(model.named_modules()):
        if isinstance(m, BasicBlock):
            groups = [[f"{name}.conv1", f"{name}.bn1", f"{name}.relu1"],
                      [f"{name}.conv2", f"{name}.bn2"]]
            if len(m.shortcut) > 0:
                groups.append([f"{name}.shortcut.0", f"{name}.shortcut.1"])
            fuse(model, groups, inplace=True)
    return model


# ----------------------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def fit(model, train_loader, test_loader, device, epochs, lr, weight_decay=5e-4,
        schedule="cosine", amp=False, epoch_callback=None, tag=""):
    """SGD training loop used for baseline training, fine-tuning and QAT."""
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay, nesterov=True)
    total = max(1, epochs * len(train_loader))
    if schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.2)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    for ep in range(epochs):
        if epoch_callback:
            epoch_callback(ep, model)
        model.train()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            n += y.numel()
        test_acc = evaluate(model, test_loader, device)
        history.append(dict(epoch=ep + 1, loss=loss_sum / n, train_acc=100 * correct / n,
                            test_acc=test_acc))
        print(f"[{tag}] epoch {ep + 1:3d}/{epochs} loss {loss_sum / n:.4f} "
              f"train {100 * correct / n:5.2f}% test {test_acc:5.2f}% ({time.time() - t0:.0f}s)")
    return history


# ----------------------------------------------------------------------------- statistics
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_nonzero_params(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def weight_sparsity(model) -> float:
    """% of exactly-zero weights over all Conv2d / Linear layers."""
    zeros = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += (m.weight == 0).sum().item()
            total += m.weight.numel()
    return 100.0 * zeros / max(1, total)


@torch.no_grad()
def count_macs(model, input_size=(1, 3, 32, 32)) -> int:
    """Dense multiply-accumulates of Conv2d + Linear layers for one image."""
    macs, hooks = [0], []

    def conv_hook(m, i, o):
        macs[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def fc_hook(m, i, o):
        macs[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    model.eval()
    model(torch.randn(*input_size))
    for h in hooks:
        h.remove()
    return macs[0]


def file_size_mb(path) -> float:
    return os.path.getsize(path) / 1e6


def gzip_size_mb(path) -> float:
    """Size after zlib-9 compression - shows the storage benefit of sparsity."""
    with open(path, "rb") as f:
        return len(zlib.compress(f.read(), 9)) / 1e6


# ----------------------------------------------------------------------------- checkpoints
def save_float_model(model, path, extra=None):
    ck = {"state_dict": model.state_dict(), "mid_channels": model.mid_channels}
    ck.update(extra or {})
    torch.save(ck, path)


def load_float_model(path) -> ResNet18:
    ck = torch.load(path, map_location="cpu")
    model = ResNet18(mid_channels=ck.get("mid_channels"))
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def save_quantized(model, path):
    """Save INT8 model as self-contained TorchScript (runs without the class code)."""
    model = model.cpu().eval()
    traced = torch.jit.trace(model, torch.randn(1, 3, 32, 32), check_trace=False)
    torch.jit.save(traced, str(path))
    torch.save(model.state_dict(), str(path).replace(".pt", "_state_dict.pth"))


def export_onnx_model(model, path, opset=13):
    model = model.cpu().eval()
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    dummy = torch.randn(1, 3, 32, 32)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
    except TypeError:                      # older torch without `dynamo` argument
        torch.onnx.export(model, dummy, str(path), **kw)
