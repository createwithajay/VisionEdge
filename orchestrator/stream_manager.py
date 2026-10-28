
"""Week 4: Concurrent video stream orchestration with retries.

CPU-testable baseline for VisionEdge.
GPU acceleration is not verified by this module.
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
    attempts: int = 0


def decode_and_process(video_path: str, max_frames: int) -> int:
    """Decode video frames and perform a lightweight CPU workload."""
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    path = Path(video_path)

    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    frames_processed = 0

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found: {path}")

        video_stream = container.streams.video[0]

        for frame in container.decode(video_stream):
            bgr = frame.to_ndarray(format="bgr24")

            # Lightweight CPU workload; this is NOT YOLO inference.
            sample = bgr[::16, ::16, :].astype(np.float32)
            _ = float(np.mean(sample))

            frames_processed += 1

            if frames_processed >= max_frames:
                break

    if frames_processed == 0:
        raise ValueError(f"No video frames decoded: {path}")

    return frames_processed


class StreamManager:
    """Runs independent streams with a concurrency limit and retries."""

    def __init__(
        self,
        max_concurrent_streams: int = 4,
        max_retries: int = 2,
        retry_delay: float = 0.25,
    ):
        if max_concurrent_streams < 1:
            raise ValueError("max_concurrent_streams must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_delay < 0:
            raise ValueError("retry_delay cannot be negative")

        self.max_concurrent_streams = max_concurrent_streams
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max_concurrent_streams)

    async def process_stream(
        self, stream_id: str, video_path: str, max_frames: int = 90
    ) -> StreamResult:
        started = time.perf_counter()
        attempts = 0
        last_error = None

        async with self.semaphore:
            for attempt in range(1, self.max_retries + 2):
                attempts = attempt

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
                        attempts=attempts,
                    )

                except asyncio.CancelledError:
                    raise

                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"

                    if attempt <= self.max_retries and self.retry_delay:
                        await asyncio.sleep(self.retry_delay)

        elapsed = time.perf_counter() - started

        return StreamResult(
            stream_id=stream_id,
            status="failed",
            elapsed_seconds=round(elapsed, 4),
            error=last_error,
            attempts=attempts,
        )

    async def run_streams(
        self,
        streams: list[tuple[str, str]],
        max_frames: int = 90,
    ) -> list[StreamResult]:
        """Run streams concurrently, preserving the input order of results."""
        tasks = [
            self.process_stream(stream_id, path, max_frames)
            for stream_id, path in streams
        ]
        return await asyncio.gather(*tasks)


async def main():
    manager = StreamManager(
        max_concurrent_streams=4,
        max_retries=2,
    )

    streams = [
        (f"Stream-{i:02d}", "sample_video.mp4")
        for i in range(1, 5)
    ]

    started = time.perf_counter()
    results = await manager.run_streams(streams, max_frames=90)
    total_elapsed = time.perf_counter() - started
    total_frames = sum(result.frames_processed for result in results)

    
    report = {
        "project": "VisionEdge",
        "phase": "Week 4 orchestration with retries",
        "execution_device": "CPU baseline; GPU not verified",
        "max_concurrent_streams": manager.max_concurrent_streams,
        "max_retries": manager.max_retries,
        "total_streams": len(results),
        "successful_streams": sum(
            result.status == "completed" for result in results
        ),
        "failed_streams": sum(
            result.status == "failed" for result in results
        ),
        "total_frames_processed": total_frames,
        "total_elapsed_seconds": round(total_elapsed, 4),
        "aggregate_fps": round(total_frames / total_elapsed, 2)
        if total_elapsed
        else 0.0,
        "streams": [asdict(result) for result in results],
    }

    output = Path("week4_orchestration_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {output.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
