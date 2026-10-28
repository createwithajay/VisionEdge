"""VisionEdge Week 2: Honest CPU profiling baseline.

This script measures video-frame conversion and lightweight CPU preprocessing.
It does not claim to benchmark YOLO, PyTorch, TensorRT, or GPU VRAM.
"""

import json
import os
import time
from pathlib import Path

import av
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


VIDEO_PATH = Path("sample_video.mp4")
MAX_FRAMES = 150


def get_process_ram_mb():
    """Return this process's resident RAM usage, if psutil is available."""
    if psutil is None:
        return None

    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def preprocess_frame(raw_bgr_frame):
    """Run a small, measurable CPU preprocessing workload."""
    resized = raw_bgr_frame[::2, ::2, :]
    rgb = resized[:, :, ::-1]
    normalized = rgb.astype(np.float32) / 255.0

    # Lightweight CPU operation, not AI model inference.
    checksum = float(np.mean(normalized))
    return checksum


def run_week2_pipeline(video_path=VIDEO_PATH):
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    latencies_ms = []
    memory_samples_mb = []
    frames_processed = 0
    start_memory_mb = get_process_ram_mb()

    print("=" * 60)
    print("VISIONEDGE WEEK 2 — CPU PROFILING BASELINE")
    print("Device: CPU; GPU acceleration is not verified")
    print("Workload: Video frame conversion and lightweight preprocessing")
    print("=" * 60)

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in: {video_path}")

        video_stream = container.streams.video[0]

        for frame in container.decode(video_stream):
            start = time.perf_counter()

            raw_bgr = frame.to_ndarray(format="bgr24")
            _ = preprocess_frame(raw_bgr)

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
            frames_processed += 1

            if frames_processed % 25 == 0:
                ram_mb = get_process_ram_mb()

                if ram_mb is not None:
                    memory_samples_mb.append(ram_mb)
                    print(
                        f"Frames: {frames_processed:03d} | "
                        f"CPU workload: {elapsed_ms:.2f} ms/frame | "
                        f"Process RAM: {ram_mb:.2f} MB"
                    )
                else:
                    print(
                        f"Frames: {frames_processed:03d} | "
                        f"CPU workload: {elapsed_ms:.2f} ms/frame"
                    )

            if frames_processed >= MAX_FRAMES:
                break

    if frames_processed == 0:
        raise ValueError(f"No video frames decoded from: {video_path}")

    end_memory_mb = get_process_ram_mb()

    average_ms = float(np.mean(latencies_ms))
    minimum_ms = float(np.min(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    workload_fps = (
        1000.0 / average_ms if average_ms > 0 else 0.0
    )

    ram_growth_mb = None
    if start_memory_mb is not None and end_memory_mb is not None:
        ram_growth_mb = end_memory_mb - start_memory_mb

    report = {
        "project": "VisionEdge",
        "phase": "Week 2 - CPU profiling baseline",
        "role": "Member 1 - Orchestration Engineer",
        "video_file": str(video_path),
        "device": "CPU",
        "gpu_acceleration_verified": False,
        "workload": (
            "Video frame conversion and lightweight NumPy preprocessing; "
            "not YOLO or neural-network inference"
        ),
        "frames_processed": frames_processed,
        "cpu_workload_latency_ms": {
            "average": round(average_ms, 3),
            "minimum": round(minimum_ms, 3),
            "p95": round(p95_ms, 3),
        },
        "cpu_workload_fps_estimate": round(workload_fps, 2),
        "memory_profiling": {
            "measurement": "Process resident RAM, not GPU VRAM",
            "psutil_available": psutil is not None,
            "start_ram_mb": (
                round(start_memory_mb, 2)
                if start_memory_mb is not None else None
            ),
            "end_ram_mb": (
                round(end_memory_mb, 2)
                if end_memory_mb is not None else None
            ),
            "ram_growth_mb": (
                round(ram_growth_mb, 2)
                if ram_growth_mb is not None else None
            ),
            "memory_leak_verified": False,
        },
        "pytorch_tensorRT_comparison": {
            "status": "NOT BENCHMARKED",
            "pytorch_fps": None,
            "tensorrt_fps": None,
            "speedup_factor": None,
            "reason": (
                "No actual model-based PyTorch and TensorRT benchmark "
                "was run by this script."
            ),
        },
        "target_10_streams_4k_60fps": "NOT VERIFIED",
    }

    output = Path("week2_audit_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- CPU Profiling Results ---")
    print(f"Frames processed: {frames_processed}")
    print(f"Average workload latency: {average_ms:.3f} ms/frame")
    print(f"Minimum workload latency: {minimum_ms:.3f} ms/frame")
    print(f"95th percentile latency: {p95_ms:.3f} ms/frame")
    print(f"CPU workload FPS estimate: {workload_fps:.2f}")
    print("\nPyTorch/TensorRT speedup: NOT BENCHMARKED")
    print("GPU memory and 10-stream target: NOT VERIFIED")
    print(f"Report saved to: {output.resolve()}")


if __name__ == "__main__":
    run_week2_pipeline()