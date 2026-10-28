
import asyncio
from orchestrator.stream_manager import StreamManager


async def main():
    manager = StreamManager(max_concurrent_streams=3)

    streams = [
        ("Stream-01", "sample_video.mp4"),
        ("Stream-02", "missing_video.mp4"),
        ("Stream-03", "sample_video.mp4"),
    ]

    results = await manager.run_streams(streams, max_frames=20)

    for result in results:
        print(
            f"{result.stream_id}: {result.status}, "
            f"frames={result.frames_processed}, error={result.error}"
        )

    successful = sum(r.status == "completed" for r in results)
    failed = sum(r.status == "failed" for r in results)

    print(f"\nSuccessful streams: {successful}")
    print(f"Failed streams: {failed}")

    assert successful == 2, "Expected two successful streams"
    assert failed == 1, "Expected one failed stream"
    assert results[1].stream_id == "Stream-02"
    assert results[1].status == "failed"

    print("PASS: One stream failed without stopping the other streams.")


if __name__ == "__main__":
    asyncio.run(main())