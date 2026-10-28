"""
Vision Edge — Member 3, Week 1, Step 1
Export a pretrained YOLOv10 checkpoint (PyTorch) to ONNX format.

Why this step exists:
    TensorRT cannot compile a PyTorch model directly. ONNX is the universal
    intermediate representation that bridges "PyTorch land" to "TensorRT land".
    Get this step right and clean, and the TensorRT compile step becomes trivial.

Usage:
    python 01_export_onnx.py --weights models/yolov10n.pt --imgsz 640 --simplify

Output:
    models/<weights_name>.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv10 (.pt) to ONNX")
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to the .pt checkpoint, e.g. models/yolov10n.pt")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Square input resolution the model expects (default: 640)")
    parser.add_argument("--opset", type=int, default=17,
                         help="ONNX opset version. 17 is the safe modern default for TensorRT 10.x")
    parser.add_argument("--batch", type=int, default=1,
                         help="Export batch size. Use 1 for a single stream; "
                              "raise this later if you batch multiple camera feeds together")
    parser.add_argument("--dynamic", action="store_true",
                         help="Allow dynamic batch size in the exported graph "
                              "(useful once the Orchestration Engineer starts running "
                              "multiple concurrent streams)")
    parser.add_argument("--simplify", action="store_true",
                         help="Run onnx-simplifier after export to fold constants and "
                              "clean up redundant ops (recommended, makes TensorRT's job easier)")
    parser.add_argument("--half", action="store_true",
                         help="Export weights in FP16. Usually leave this OFF here — "
                              "do precision conversion at the TensorRT stage instead, "
                              "so you keep one FP32 ONNX file as your 'source of truth'.")
    return parser.parse_args()


def validate_onnx_forward_pass(onnx_path: Path, imgsz: int, batch: int) -> None:
    """
    Load the exported ONNX file with onnxruntime and push a dummy tensor through it.
    This catches export bugs (bad shapes, unsupported ops, broken dynamic axes)
    immediately, instead of discovering them later inside a cryptic TensorRT error.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime not installed — skipping forward-pass validation. "
              "Install onnxruntime-gpu to enable this safety check.")
        return

    print(f"[check] Loading {onnx_path.name} with onnxruntime for a sanity forward pass...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)

    outputs = session.run(None, {input_name: dummy_input})
    print(f"[check] Forward pass succeeded. Output tensor shapes: "
          f"{[o.shape for o in outputs]}")
    print("[check] ONNX export looks structurally valid.\n")


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"[error] Checkpoint not found at {weights_path}. "
              f"Download one first, e.g.:\n"
              f"    python -c \"from ultralytics import YOLO; YOLO('yolov10n.pt')\"")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"[info] Loading checkpoint: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[info] Exporting to ONNX  |  imgsz={args.imgsz}  batch={args.batch}  "
          f"opset={args.opset}  dynamic={args.dynamic}  simplify={args.simplify}")

    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
    )

    export_path = Path(export_path)
    print(f"\n[success] ONNX model written to: {export_path}")

    validate_onnx_forward_pass(export_path, args.imgsz, args.batch)

    print("[next] Feed this file into 02_build_tensorrt_engine.py to compile the "
          "TensorRT engine:")
    print(f"    python scripts/02_build_tensorrt_engine.py --onnx {export_path} --fp16")


if __name__ == "__main__":
    main()
