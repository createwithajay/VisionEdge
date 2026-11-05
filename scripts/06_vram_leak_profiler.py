"""
Vision Edge — Member 3, Week 2, Step 3
Profile GPU memory usage over an extended continuous inference run to catch
VRAM leaks before they become a 3am problem during the Final Review demo.

Why this step exists:
    A leak that grows by 2MB every 1000 frames is invisible in a 10-frame
    test but will crash the pipeline after an hour of real streaming. This
    script runs thousands of inference passes and plots memory over time —
    a flat line means you're clean; a rising line means you have a leak to
    chase down (usually: a buffer being re-allocated every loop instead of
    reused, or a reference being held onto unintentionally).

Usage:
    python 06_vram_leak_profiler.py --engine engines/yolov10n_fp16.engine --iterations 3000

Output:
    results/vram_profile_log.csv
    results/vram_profile_chart.png
    Console verdict: LEAK DETECTED or CLEAN
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile VRAM usage during continuous inference")
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=3000,
                         help="Number of inference passes to run. Higher = more confidence "
                              "in the leak-detection slope, but takes longer.")
    parser.add_argument("--sample_every", type=int, default=20,
                         help="Record a memory sample every N iterations "
                              "(sampling every frame would itself distort timing)")
    parser.add_argument("--leak_threshold_mb", type=float, default=5.0,
                         help="Flag a leak if total VRAM growth exceeds this many MB "
                              "over the full run")
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def get_gpu_memory_mb() -> float:
    """Query current GPU memory usage in MB via pycuda."""
    import pycuda.driver as cuda
    free_bytes, total_bytes = cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return used_bytes / (1024 ** 2)


def main() -> None:
    args = parse_args()

    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine not found at {engine_path}. Run Week 1's build step first.")

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = context.get_tensor_shape(input_name)
    resolved_shape = tuple(d if d != -1 else 1 for d in input_shape)
    context.set_input_shape(input_name, resolved_shape)

    output_host = np.empty(tuple(context.get_tensor_shape(output_name)), dtype=np.float32)
    # Allocate buffers ONCE outside the loop — reusing them is the correct pattern.
    # (A common source of real leaks is allocating these fresh every iteration instead.)
    d_input = cuda.mem_alloc(int(np.prod(resolved_shape)) * 4)
    d_output = cuda.mem_alloc(output_host.nbytes)
    stream = cuda.Stream()
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    dummy_frame = np.random.rand(*resolved_shape).astype(np.float32)

    print(f"[info] Running {args.iterations} continuous inference passes, "
          f"sampling VRAM every {args.sample_every} iterations...")
    print("[info] This intentionally reuses GPU buffers across iterations — "
          "the correct pattern for a long-running stream. A leak here would "
          "indicate a real TensorRT/driver-level issue worth escalating, "
          "not just sloppy Python buffer handling.")

    samples = []
    start_time = time.perf_counter()

    for i in range(args.iterations):
        cuda.memcpy_htod_async(d_input, dummy_frame, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, d_output, stream)
        stream.synchronize()

        if i % args.sample_every == 0:
            mem_mb = get_gpu_memory_mb()
            samples.append((i, mem_mb))

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  [progress] {i + 1}/{args.iterations} iterations "
                  f"({elapsed:.1f}s elapsed), current VRAM: {samples[-1][1]:.1f} MB")

    # --- Leak analysis: simple linear regression slope of memory over iteration count ---
    iterations_arr = np.array([s[0] for s in samples])
    memory_arr = np.array([s[1] for s in samples])

    # Ignore the first few samples — initial allocations/caching settle out early
    # and would otherwise look like a false "leak" at the start of the run.
    settle_idx = max(3, len(samples) // 10)
    slope, intercept = np.polyfit(iterations_arr[settle_idx:], memory_arr[settle_idx:], 1)
    total_growth_mb = memory_arr[-1] - memory_arr[settle_idx]

    print("\n" + "=" * 60)
    print(" VRAM LEAK PROFILE SUMMARY")
    print("=" * 60)
    print(f" Iterations run:         {args.iterations}")
    print(f" Samples collected:      {len(samples)}")
    print(f" Starting VRAM (settled):{memory_arr[settle_idx]:.1f} MB")
    print(f" Ending VRAM:            {memory_arr[-1]:.1f} MB")
    print(f" Total growth:           {total_growth_mb:+.2f} MB")
    print(f" Growth slope:           {slope * 1000:+.4f} MB per 1000 iterations")

    leak_detected = total_growth_mb > args.leak_threshold_mb
    print(f" Verdict:                {'⚠️  LEAK DETECTED' if leak_detected else '✅ CLEAN — no leak'}")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "vram_profile_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "vram_used_mb"])
        writer.writerows(samples)
    print(f"\n[success] VRAM profile log saved to: {log_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations_arr, memory_arr, color="#3b82f6", linewidth=1.5, label="VRAM usage")
        trend_line = slope * iterations_arr + intercept
        ax.plot(iterations_arr, trend_line, color="#ef4444", linestyle="--",
                 label=f"Trend ({slope*1000:+.3f} MB / 1000 iter)")
        ax.set_xlabel("Inference iteration")
        ax.set_ylabel("GPU memory used (MB)")
        ax.set_title("VRAM Usage During Continuous Streaming\n"
                      f"{'⚠️ Leak detected' if leak_detected else '✅ Clean — stable over full run'}")
        ax.legend()
        plt.tight_layout()
        chart_path = output_dir / "vram_profile_chart.png"
        plt.savefig(chart_path, dpi=150)
        print(f"[success] Chart saved to: {chart_path}")
    except ImportError:
        print("[warn] matplotlib not installed — skipping chart. pip install matplotlib.")

    if leak_detected:
        print("\n[note] A positive slope over a long run usually points to one of: "
              "input/output buffers being re-allocated inside the loop instead of reused, "
              "a Python-side reference to old output arrays never being released, or "
              "(rarer) a genuine driver/TensorRT-level allocation issue worth reporting "
              "to Member 2 before the Mid-Review.")


if __name__ == "__main__":
    main()
