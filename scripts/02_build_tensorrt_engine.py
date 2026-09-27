"""
Vision Edge — Member 3, Week 1, Step 2
Compile an ONNX graph into a serialized, GPU-specific TensorRT engine.
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX file")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="engines")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace", type=int, default=2048)
    parser.add_argument("--min_batch", type=int, default=1)
    parser.add_argument("--opt_batch", type=int, default=1)
    parser.add_argument("--max_batch", type=int, default=1)
    return parser.parse_args()


def build_engine(onnx_path: Path, output_path: Path, fp16: bool, workspace_mb: int,
                  min_batch: int, opt_batch: int, max_batch: int) -> None:
    try:
        import tensorrt as trt
    except ImportError:
        print("[error] tensorrt is not installed in this environment.")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
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
        print("[warn] This TensorRT version's Python API does not expose a BuilderFlag.FP16 "
              "flag. Building in default precision instead. FP16/precision tuning can be "
              "revisited in Week 2 using strongly-typed ONNX export.")

    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

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

    print("[info] Building TensorRT engine — this can take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("[error] Engine build failed.")
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
        print(f"[error] ONNX file not found at {onnx_path}.")
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
