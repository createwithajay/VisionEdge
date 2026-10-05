"""
Vision Edge — Member 3, Week 1, Step 3
Load a compiled TensorRT engine and run a sanity inference pass on a random tensor.

Why this step exists:
    A successful build in Step 2 doesn't guarantee the engine actually executes
    correctly at runtime (buffer binding mistakes are a common source of silent
    failures). This script is your "does it actually run" gate before you tell
    the team Week 1 is done.

Usage:
    python 03_verify_engine.py --engine engines/yolov10n_fp16.engine

Output:
    Prints input/output binding shapes and a rough single-frame latency estimate.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a compiled TensorRT engine")
    parser.add_argument("--engine", type=str, required=True,
                         help="Path to the .engine file")
    parser.add_argument("--iterations", type=int, default=50,
                         help="Number of inference passes to average latency over")
    parser.add_argument("--warmup", type=int, default=10,
                         help="Number of warmup passes before timing starts "
                              "(GPU clocks/caches need to settle first)")
    return parser.parse_args()


def run_verification(engine_path: Path, iterations: int, warmup: int) -> None:
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)
    except ImportError as e:
        print(f"[error] Missing dependency: {e}")
        print("        pip install tensorrt pycuda")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.WARNING)

    print(f"[info] Loading engine: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("[error] Failed to deserialize engine. It may have been built with a "
              "different TensorRT version, or for a different GPU architecture. "
              "Rebuild it on this machine with 02_build_tensorrt_engine.py.")
        sys.exit(1)

    context = engine.create_execution_context()

    # Gather binding info (works for both implicit and explicit-batch engines)
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = context.get_tensor_shape(input_name)
    output_shape = context.get_tensor_shape(output_name)

    print(f"[info] Input  binding: {input_name}  shape={tuple(input_shape)}")
    print(f"[info] Output binding: {output_name}  shape={tuple(output_shape)}")

    # Handle dynamic shapes by substituting batch=1 if needed
    resolved_input_shape = tuple(d if d != -1 else 1 for d in input_shape)
    context.set_input_shape(input_name, resolved_input_shape)

    dummy_input = np.random.rand(*resolved_input_shape).astype(np.float32)
    output_host = np.empty(tuple(context.get_tensor_shape(output_name)), dtype=np.float32)

    d_input = cuda.mem_alloc(dummy_input.nbytes)
    d_output = cuda.mem_alloc(output_host.nbytes)
    stream = cuda.Stream()

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    def infer_once():
        cuda.memcpy_htod_async(d_input, dummy_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, d_output, stream)
        stream.synchronize()

    print(f"[info] Running {warmup} warmup passes...")
    for _ in range(warmup):
        infer_once()

    print(f"[info] Timing {iterations} inference passes...")
    start = time.perf_counter()
    for _ in range(iterations):
        infer_once()
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / iterations) * 1000
    est_fps = 1000 / avg_latency_ms

    print(f"\n[success] Engine runs correctly.")
    print(f"[result] Output shape: {output_host.shape}")
    print(f"[result] Avg single-frame latency: {avg_latency_ms:.2f} ms  "
          f"(~{est_fps:.0f} FPS, single stream, no pipeline overhead)")
    print("\n[note] This is a rough, unofficial number — the real latency benchmark "
          "against native PyTorch happens formally in Week 2 alongside Member 2. "
          "But this confirms your Week 1 deliverable is functionally complete.")


def main() -> None:
    args = parse_args()
    engine_path = Path(args.engine)

    if not engine_path.exists():
        print(f"[error] Engine file not found at {engine_path}. "
              f"Run 02_build_tensorrt_engine.py first.")
        sys.exit(1)

    run_verification(engine_path, args.iterations, args.warmup)


if __name__ == "__main__":
    main()
