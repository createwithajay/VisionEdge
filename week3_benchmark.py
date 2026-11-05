
import asyncio
import json
import time
from pathlib import Path

from orchestrator.stream_manager import StreamManager, decode_and_process


VIDEO = "sample_video.mp4"
STREAM_COUNT = 4
FRAMES_PER_STREAM = 60


async def run_benchmark():
    if not Path(VIDEO).is_file():
        raise FileNotFoundError(f"Missing benchmark video: {VIDEO}")

    # Sequential baseline: process each stream one after another.
    sequential_start = time.perf_counter()
    sequential_frames = 0

    for _ in range(STREAM_COUNT):
        sequential_frames += await asyncio.to_thread(
            decode_and_process, VIDEO, FRAMES_PER_STREAM
        )

    sequential_seconds = time.perf_counter() - sequential_start

    # Concurrent run: process independent streams together.
    manager = StreamManager(max_concurrent_streams=STREAM_COUNT)
    streams = [
        (f"Stream-{i:02d}", VIDEO)
        for i in range(1, STREAM_COUNT + 1)
    ]

    concurrent_start = time.perf_counter()
    results = await manager.run_streams(streams, FRAMES_PER_STREAM)
    concurrent_seconds = time.perf_counter() - concurrent_start

    concurrent_frames = sum(r.frames_processed for r in results)
    all_completed = all(r.status == "completed" for r in results)

    speedup = (
        sequential_seconds / concurrent_seconds
        if concurrent_seconds > 0 else 0
    )

    report = {
        "project": "VisionEdge",
        "phase": "Week 3 benchmark",
        "device": "CPU baseline; GPU acceleration not verified",
        "stream_count": STREAM_COUNT,
        "frames_per_stream": FRAMES_PER_STREAM,
        "sequential": {
            "frames": sequential_frames,
            "elapsed_seconds": round(sequential_seconds, 4),
            "aggregate_fps": round(
                sequential_frames / sequential_seconds, 2
            ) if sequential_seconds else 0,
        },
        "concurrent": {
            "frames": concurrent_frames,
            "elapsed_seconds": round(concurrent_seconds, 4),
            "aggregate_fps": round(
                concurrent_frames / concurrent_seconds, 2
            ) if concurrent_seconds else 0,
            "all_streams_completed": all_completed,
        },
        "speedup_ratio": round(speedup, 2),
        "interpretation": (
            "Concurrent execution was faster."
            if speedup > 1
            else "Concurrent execution was not faster in this run."
        ),
    }

    Path("week3_benchmark_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nReport saved to week3_benchmark_report.json")


if __name__ == "__main__":
    asyncio.run(run_benchmark())