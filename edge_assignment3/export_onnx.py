"""Deliverable 8 - export ONNX models (FP32, pruned, INT8) and verify them against PyTorch."""
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)

from inference import load_model
from common import *


class CalibReader(CalibrationDataReader):
    def __init__(self, loader):
        self.batches = iter([{"input": x.numpy()} for x, _ in loader])

    def get_next(self):
        return next(self.batches, None)


def main():
    for name in ("fp32", "unstructured", "structured"):
        model = load_model(name)
        path = ONNX_DIR / f"resnet18_{name}.onnx"
        export_onnx_model(model, path)
        x = torch.randn(4, 3, 32, 32)
        ref = model(x).detach().numpy()
        out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
        print(f"exported {path.name:38s} max|torch-onnx| = {np.abs(ref - out).max():.2e}")

    # INT8 ONNX: static PTQ (QDQ format) of the FP32 ONNX graph with the same calibration subset
    int8_path = ONNX_DIR / "resnet18_ptq_int8.onnx"
    quantize_static(str(ONNX_DIR / "resnet18_fp32.onnx"), str(int8_path),
                    CalibReader(get_calibration_loader(500, 50)),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True)
    print("exported", int8_path.name, "(QAT QDQ model is exported by qat.py)")


if __name__ == "__main__":
    main()
