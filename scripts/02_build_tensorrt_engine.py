"""
Vision Edge — Member 3, Week 1, Step 2
Compile an ONNX graph into a serialized, GPU-specific TensorRT engine.

Why this step exists:
    TensorRT does hardware-aware graph optimization (layer fusion, kernel
    auto-tuning, precision calibration) specifically for the GPU it's run on.
    The output ".engine" file is NOT portable across different GPU models —
    that's expected and correct, not a bug.

Usage:
    python 02_build_tensorrt_engine.py --onnx models/yolov10n.onnx --fp16
    python 02_build_tensorrt_engine.py --onnx models/yolov10n.onnx --fp16 --workspace 4096

Output:
    engines/<onnx_name>_<precision>.engine
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX file")
    parser.add_argument("--onnx", type=str, required=True,
                         help="Path to the input .onnx file")
    parser.add_argument("--output_dir", type=str, default="engines",
                         help="Directory to write the compiled .engine file into")
    parser.add_argument("--fp16", action="store_true",
                         help="Enable FP16 precision. Recommended default on consumer GPUs — "
                              "roughly 2x smaller and faster than FP32 with negligible accuracy loss")
    parser.add_argument("--workspace", type=int, default=2048,
                         help="Max GPU scratch memory (in MB) TensorRT can use while searching "
                              "for the fastest kernel implementations. Raise this if you hit "
                              "'insufficient workspace' errors and have spare VRAM.")
    parser.add_argument("--min_batch", type=int, default=1)
    parser.add_argument("--opt_batch", type=int, default=1,
                         help="The batch size to optimize kernel selection for — "
                              "set this to whatever your 'typical' runtime batch size will be")
    parser.add_argument("--max_batch", type=int, default=1)
    return parser.parse_args()


def build_engine(onnx_path: Path, output_path: Path, fp16: bool, workspace_mb: int,
                  min_batch: int, opt_batch: int, max_batch: int) -> None:
    try:
        import tensorrt as trt
    except ImportError:
        print("[error] tensorrt is not installed in this environment.")
        print("        pip install tensorrt   (or use the NVIDIA TensorRT container —")
        print("        see docs/TROUBLESHOOTING.md for the exact docker run command)")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[info] Parsing ONNX file: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[error] Failed to parse ONNX file. Errors:")
            for i in range(parser.num_errors):
                print(f"    {parser.get_error(i)}")
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[info] FP16 mode enabled.")
        else:
            print("[warn] This GPU doesn't report fast FP16 support — "
                  "continuing anyway, TensorRT will fall back where needed.")
            config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic shape profile — matters once the Orchestration Engineer starts
    # batching multiple concurrent camera streams together in later weeks.
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape  # e.g. (-1, 3, 640, 640) or (1, 3, 640, 640)

    if -1 in input_shape or min_batch != max_batch:
        profile = builder.create_optimization_profile()
        c, h, w = input_shape[1], input_shape[2], input_shape[3]
        profile.set_shape(
            input_tensor.name,
            min=(min_batch, c, h, w),
            opt=(opt_batch, c, h, w),
            max=(max_batch, c, h, w),
        )
        config.add_optimization_profile(profile)
        print(f"[info] Dynamic batch profile set: min={min_batch} opt={opt_batch} max={max_batch}")

    print("[info] Building TensorRT engine — this can take a few minutes while "
          "TensorRT benchmarks candidate kernels on your specific GPU...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("[error] Engine build failed. Common causes: unsupported ONNX ops, "
              "insufficient workspace memory, or an incompatible opset version. "
              "See docs/TROUBLESHOOTING.md.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    print(f"\n[success] TensorRT engine written to: {output_path}")
    print(f"[info] Engine size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    print("[next] Verify it actually runs:")
    print(f"    python scripts/03_verify_engine.py --engine {output_path}")


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)

    if not onnx_path.exists():
        print(f"[error] ONNX file not found at {onnx_path}. "
              f"Run 01_export_onnx.py first.")
        sys.exit(1)

    precision_tag = "fp16" if args.fp16 else "fp32"
    output_path = Path(args.output_dir) / f"{onnx_path.stem}_{precision_tag}.engine"

    build_engine(
        onnx_path=onnx_path,
        output_path=output_path,
        fp16=args.fp16,
        workspace_mb=args.workspace,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
    )


if __name__ == "__main__":
    main()
