
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from orchestrator.stream_manager import StreamManager


STREAM_COUNT = 10
FRAMES_PER_STREAM = 60
VIDEO_PATH = Path("sample_video.mp4")


async def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Sample video not found: {VIDEO_PATH.resolve()}")

    manager = StreamManager(
        max_concurrent_streams=STREAM_COUNT,
        max_retries=1,
        retry_delay=0.25,
    )

    streams = [
        (f"Camera-{i:02d}", str(VIDEO_PATH))
        for i in range(1, STREAM_COUNT + 1)
    ]

    print("=" * 60)
    print("VISIONEDGE - MEMBER 1 ORCHESTRATION TEST")
    print("10 independent video streams | CPU baseline")
    print("=" * 60)
    print(f"Concurrency limit : {STREAM_COUNT}")
    print(f"Frames per stream : {FRAMES_PER_STREAM}")
    print(f"Video file        : {VIDEO_PATH.resolve()}")
    print("\nStarting test...")

    start = time.perf_counter()
    results = await manager.run_streams(
        streams,
        max_frames=FRAMES_PER_STREAM,
    )
    elapsed = time.perf_counter() - start

    total_frames = sum(r.frames_processed for r in results)
    successful = sum(r.status == "completed" for r in results)
    failed = sum(r.status == "failed" for r in results)

    report = {
        "project": "VisionEdge",
        "role": "Member 1 - Orchestration Engineer",
        "test": "10-stream concurrent orchestration",
        "execution_device": "CPU baseline; NVIDIA GPU not verified",
        "concurrency_limit": STREAM_COUNT,
        "requested_frames_per_stream": FRAMES_PER_STREAM,
        "total_streams": len(results),
        "successful_streams": successful,
        "failed_streams": failed,
        "total_frames_processed": total_frames,
        "total_elapsed_seconds": round(elapsed, 4),
        "aggregate_fps": round(total_frames / elapsed, 2) if elapsed else 0,
        "stream_results": [asdict(r) for r in results],
        "gpu_validation": "NOT PERFORMED",
        "4k_60fps_validation": "NOT PERFORMED",
    }

    output = Path("week4_10_stream_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- RESULTS ---")
    for result in results:
        print(
            f"{result.stream_id}: {result.status} | "
            f"{result.frames_processed} frames | "
            f"{result.fps:.2f} FPS | "
            f"attempts: {result.attempts}"
        )

    print("\n--- SUMMARY ---")
    print(f"Successful streams : {successful}/{STREAM_COUNT}")
    print(f"Failed streams     : {failed}")
    print(f"Total frames       : {total_frames}")
    print(f"Elapsed time       : {elapsed:.2f} seconds")
    print(f"Aggregate FPS      : {report['aggregate_fps']:.2f}")
    print(f"Report saved       : {output.resolve()}")
    print("\nNote: This is a CPU orchestration test, not GPU validation.")


if __name__ == "__main__":
    asyncio.run(main())
