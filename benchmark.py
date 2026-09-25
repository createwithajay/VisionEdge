import asyncio
import time


STREAM_COUNT = 4
FRAMES_PER_STREAM = 20
FRAME_DELAY = 0.05


def process_stream_sync(stream_id: int):
    for _ in range(FRAMES_PER_STREAM):
        time.sleep(FRAME_DELAY)


async def process_stream_async(stream_id: int):
    for _ in range(FRAMES_PER_STREAM):
        await asyncio.sleep(FRAME_DELAY)


def run_sequential():
    start = time.perf_counter()

    for stream_id in range(1, STREAM_COUNT + 1):
        process_stream_sync(stream_id)

    return time.perf_counter() - start


async def run_concurrent():
    start = time.perf_counter()

    tasks = [
        asyncio.create_task(
            process_stream_async(stream_id)
        )
        for stream_id in range(1, STREAM_COUNT + 1)
    ]

    await asyncio.gather(*tasks)

    return time.perf_counter() - start


async def main():
    print("========================================")
    print(" VisionEdge - Concurrency Benchmark")
    print("========================================")
    print(f"Streams: {STREAM_COUNT}")
    print(f"Frames per stream: {FRAMES_PER_STREAM}")
    print(f"Frame delay: {FRAME_DELAY}s\n")

    sequential_time = run_sequential()

    concurrent_time = await run_concurrent()

    speedup = sequential_time / concurrent_time

    print("--------------- Results ----------------")
    print(f"Sequential time : {sequential_time:.3f} seconds")
    print(f"Asyncio time    : {concurrent_time:.3f} seconds")
    print(f"Time reduction  : {sequential_time - concurrent_time:.3f} seconds")
    print(f"Concurrency ratio: {speedup:.2f}x")
    print("-----------------------------------------")

    print("\nWeek 1 concurrency test completed.")


if __name__ == "__main__":
    asyncio.run(main())