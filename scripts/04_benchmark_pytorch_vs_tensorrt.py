"""
Vision Edge — Member 3, Week 2, Step 1
Performance audit: prove TensorRT achieves at least a 3x FPS speedup over
native PyTorch inference, on the same GPU, same model, same input size.

Why this matters:
    This is the formal proof the SOP asks for. It's not enough to say
    "TensorRT is faster" — this script produces a reproducible number,
    a JSON result file your teammates/mentor can check, and a chart image
    for the Mid-Review slide.

Usage:
    python 04_benchmark_pytorch_vs_tensorrt.py \
        --weights models/yolov10n.pt \
        --engine engines/yolov10n_fp16.engine \
        --iterations 200

Output:
    results/benchmark_results.json
    results/benchmark_chart.png
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark native PyTorch vs TensorRT")
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to the .pt checkpoint (native PyTorch baseline)")
    parser.add_argument("--engine", type=str, required=True,
                         help="Path to the compiled .engine file (TensorRT)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=200,
                         help="Number of timed inference passes per backend")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def benchmark_pytorch(weights_path: str, imgsz: int, iterations: int, warmup: int) -> dict:
    import torch
    from ultralytics import YOLO

    print("[info] Benchmarking native PyTorch...")
    model = YOLO(weights_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] CUDA not available — PyTorch baseline will run on CPU. "
              "This will make the TensorRT speedup look artificially large; "
              "note this clearly if you report the number.")
    model.to(device)

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    print(f"[info] Warming up ({warmup} passes)...")
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, verbose=False, device=device)

    if device == "cuda":
        torch.cuda.synchronize()

    print(f"[info] Timing {iterations} passes...")
    start = time.perf_counter()
    for _ in range(iterations):
        model.predict(dummy, imgsz=imgsz, verbose=False, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / iterations) * 1000
    fps = 1000 / avg_latency_ms
    return {"backend": "pytorch", "device": device, "avg_latency_ms": avg_latency_ms, "fps": fps}


def benchmark_tensorrt(engine_path: str, iterations: int, warmup: int) -> dict:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    print("[info] Benchmarking TensorRT engine...")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    input_shape = context.get_tensor_shape(input_name)
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

    print(f"[info] Warming up ({warmup} passes)...")
    for _ in range(warmup):
        infer_once()

    print(f"[info] Timing {iterations} passes...")
    start = time.perf_counter()
    for _ in range(iterations):
        infer_once()
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / iterations) * 1000
    fps = 1000 / avg_latency_ms
    return {"backend": "tensorrt", "device": "cuda", "avg_latency_ms": avg_latency_ms, "fps": fps}


def make_chart(pytorch_result: dict, tensorrt_result: dict, speedup: float, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping chart generation. "
              "pip install matplotlib to enable this.")
        return

    labels = ["Native PyTorch", "TensorRT Engine"]
    fps_values = [pytorch_result["fps"], tensorrt_result["fps"]]
    colors = ["#94a3b8", "#22c55e"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, fps_values, color=colors, width=0.5)
    ax.set_ylabel("Frames Per Second (FPS)")
    ax.set_title(f"Vision Edge — Inference Speedup\n{speedup:.2f}x faster with TensorRT")

    for bar, val in zip(bars, fps_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(fps_values) * 0.02,
                 f"{val:.1f} FPS", ha="center", fontweight="bold")

    ax.axhline(y=pytorch_result["fps"] * 3, color="#ef4444", linestyle="--", linewidth=1,
               label="3x target")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[success] Chart saved to: {output_path}")


def main() -> None:
    args = parse_args()

    pytorch_result = benchmark_pytorch(args.weights, args.imgsz, args.iterations, args.warmup)
    tensorrt_result = benchmark_tensorrt(args.engine, args.iterations, args.warmup)

    speedup = tensorrt_result["fps"] / pytorch_result["fps"]

    print("\n" + "=" * 60)
    print(" BENCHMARK RESULTS — Native PyTorch vs TensorRT")
    print("=" * 60)
    print(f" PyTorch   ({pytorch_result['device']}):  "
          f"{pytorch_result['avg_latency_ms']:.2f} ms/frame  "
          f"({pytorch_result['fps']:.1f} FPS)")
    print(f" TensorRT  (cuda):  "
          f"{tensorrt_result['avg_latency_ms']:.2f} ms/frame  "
          f"({tensorrt_result['fps']:.1f} FPS)")
    print("-" * 60)
    print(f" Speedup: {speedup:.2f}x")
    print(f" Target met (>= 3.0x): {'YES ✅' if speedup >= 3.0 else 'NOT YET ❌'}")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "pytorch": pytorch_result,
        "tensorrt": tensorrt_result,
        "speedup": speedup,
        "target_met": speedup >= 3.0,
        "iterations": args.iterations,
        "imgsz": args.imgsz,
    }
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[success] Results saved to: {results_path}")

    make_chart(pytorch_result, tensorrt_result, speedup, output_dir / "benchmark_chart.png")

    if speedup < 3.0:
        print("\n[note] Speedup is below 3x. Common causes: PyTorch baseline also running on "
              "GPU with warm cache (fair comparison, but TensorRT's edge shrinks on tiny "
              "models); engine built without FP16 (see docs/WEEK2_NOTES.md); or GPU thermal "
              "throttling during the run. This is still a legitimate, honest result to report "
              "— document it rather than hide it.")


if __name__ == "__main__":
    main()
