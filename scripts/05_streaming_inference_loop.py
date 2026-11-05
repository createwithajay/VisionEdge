"""
Vision Edge — Member 3, Week 2, Step 2
Connect a continuous stream of frames to the TensorRT inference loop and
measure real sustained latency — not just a single warm-cache pass.

Why this step exists:
    A single inference call (Week 1's verification) proves the engine works.
    It doesn't prove it holds up under continuous load, the way it will once
    Member 2's real decoder is feeding it frames non-stop. This script
    simulates that: a tight loop pulling "frames" (synthetic for now, or a
    real video file if you point --source at one) and running them through
    the engine back-to-back, the way the real pipeline will.

Usage:
    # synthetic frames (no dependency on Member 2's decoder being ready yet)
    python 05_streaming_inference_loop.py --engine engines/yolov10n_fp16.engine --frames 500

    # or against a real video file, once you have one
    python 05_streaming_inference_loop.py --engine engines/yolov10n_fp16.engine --source sample_data/test.mp4

Output:
    results/streaming_latency_log.csv
    Console summary: avg / p50 / p95 / p99 latency, sustained FPS
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream frames into the TensorRT inference loop")
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--frames", type=int, default=500,
                         help="Number of frames to push through the loop "
                              "(ignored if --source is a finite video file)")
    parser.add_argument("--source", type=str, default=None,
                         help="Optional path to a real video file. If omitted, "
                              "synthetic random frames simulate the stream — "
                              "useful for testing before Member 2's decoder is ready.")
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def frame_generator(source: str, n_frames: int, imgsz: int):
    """Yields frames as (imgsz, imgsz, 3) uint8 arrays, from a video file or synthetic noise."""
    if source is None:
        for _ in range(n_frames):
            yield np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        return

    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (imgsz, imgsz))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
        count += 1
    cap.release()
    print(f"[info] Read {count} frames from {source}")


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
    imgsz = resolved_shape[2]

    output_host = np.empty(tuple(context.get_tensor_shape(output_name)), dtype=np.float32)
    d_input = cuda.mem_alloc(int(np.prod(resolved_shape)) * 4)  # float32
    d_output = cuda.mem_alloc(output_host.nbytes)
    stream = cuda.Stream()
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    print(f"[info] Streaming {'synthetic frames' if args.source is None else args.source} "
          f"into the TensorRT inference loop...")

    latencies = []
    for i, frame in enumerate(frame_generator(args.source, args.frames, imgsz)):
        frame_f32 = frame.astype(np.float32) / 255.0
        frame_f32 = np.transpose(frame_f32, (2, 0, 1))[np.newaxis, ...]  # HWC -> NCHW
        frame_f32 = np.ascontiguousarray(frame_f32)

        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_input, frame_f32, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, d_output, stream)
        stream.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

        if (i + 1) % 100 == 0:
            print(f"  [progress] {i + 1} frames processed, "
                  f"running avg latency: {np.mean(latencies):.2f} ms")

    latencies = np.array(latencies)
    avg = latencies.mean()
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    sustained_fps = 1000 / avg

    print("\n" + "=" * 60)
    print(" STREAMING INFERENCE SUMMARY")
    print("=" * 60)
    print(f" Frames processed:   {len(latencies)}")
    print(f" Avg latency:        {avg:.2f} ms")
    print(f" p50 latency:        {p50:.2f} ms")
    print(f" p95 latency:        {p95:.2f} ms")
    print(f" p99 latency:        {p99:.2f} ms")
    print(f" Sustained FPS:      {sustained_fps:.1f}")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "streaming_latency_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "latency_ms"])
        for i, lat in enumerate(latencies):
            writer.writerow([i, f"{lat:.4f}"])
    print(f"\n[success] Per-frame latency log saved to: {log_path}")
    print("[next] Hand this sustained-latency number to Member 2 for pipeline timing, "
          "and to Member 4 for WebRTC buffering decisions.")


if __name__ == "__main__":
    main()
