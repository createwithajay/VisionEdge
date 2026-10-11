
"""Week 3: Concurrent video stream orchestration.

CPU-testable baseline for VisionEdge.
GPU inference can be integrated through the processor callback later.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import av
import numpy as np


@dataclass
class StreamResult:
    stream_id: str
    status: str
    frames_processed: int = 0
    elapsed_seconds: float = 0.0
    fps: float = 0.0
    error: Optional[str] = None


def decode_and_process(video_path: str, max_frames: int) -> int:
    """Decode frames and perform a small CPU workload."""
    path = Path(video_path)

    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    frames_processed = 0

    with av.open(str(path)) as container:
        video_stream = container.streams.video[0]

        for frame in container.decode(video_stream):
            bgr = frame.to_ndarray(format="bgr24")

            # Lightweight CPU workload; this is NOT YOLO inference.
            sample = bgr[::16, ::16, :].astype(np.float32)
            _ = float(np.mean(sample))

            frames_processed += 1

            if frames_processed >= max_frames:
                break

    return frames_processed


class StreamManager:
    """Runs independent video streams concurrently."""

    def __init__(self, max_concurrent_streams: int = 4):
        if max_concurrent_streams < 1:
            raise ValueError("max_concurrent_streams must be at least 1")

        self.semaphore = asyncio.Semaphore(max_concurrent_streams)

    async def process_stream(
        self, stream_id: str, video_path: str, max_frames: int = 90
    ) -> StreamResult:
        async with self.semaphore:
            started = time.perf_counter()

            try:
                count = await asyncio.to_thread(
                    decode_and_process, video_path, max_frames
                )
                elapsed = time.perf_counter() - started

                return StreamResult(
                    stream_id=stream_id,
                    status="completed",
                    frames_processed=count,
                    elapsed_seconds=round(elapsed, 4),
                    fps=round(count / elapsed, 2) if elapsed else 0.0,
                )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                elapsed = time.perf_counter() - started

                return StreamResult(
                    stream_id=stream_id,
                    status="failed",
                    elapsed_seconds=round(elapsed, 4),
                    error=f"{type(exc).__name__}: {exc}",
                )

    async def run_streams(
        self, streams: list[tuple[str, str]], max_frames: int = 90
    ) -> list[StreamResult]:
        tasks = [
            self.process_stream(stream_id, path, max_frames)
            for stream_id, path in streams
        ]
        return await asyncio.gather(*tasks)


async def main():
    manager = StreamManager(max_concurrent_streams=4)

    # Repeat the sample video as independent test streams.
    streams = [
        (f"Stream-{i:02d}", "sample_video.mp4")
        for i in range(1, 5)
    ]

    started = time.perf_counter()
    results = await manager.run_streams(streams, max_frames=90)
    total_elapsed = time.perf_counter() - started

    report = {
        "project": "VisionEdge",
        "phase": "Week 3",
        "implementation": "asyncio concurrent stream orchestration",
        "execution_device": "CPU baseline; GPU not verified",
        "total_streams": len(results),
        "successful_streams": sum(r.status == "completed" for r in results),
        "failed_streams": sum(r.status == "failed" for r in results),
        "total_elapsed_seconds": round(total_elapsed, 4),
        "aggregate_fps": round(
            sum(r.frames_processed for r in results) / total_elapsed, 2
        ) if total_elapsed else 0.0,
        "streams": [asdict(result) for result in results],
    }

    output = Path("week3_orchestration_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {output.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())