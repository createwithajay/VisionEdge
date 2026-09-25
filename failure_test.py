import asyncio


async def process_stream(stream_id: str, should_fail: bool = False):
    print(f"[{stream_id}] Started")

    try:
        for frame in range(1, 11):
            await asyncio.sleep(0.1)

            if should_fail and frame == 5:
                raise RuntimeError("Simulated stream failure")

            print(f"[{stream_id}] Frame {frame} processed")

        print(f"[{stream_id}] Completed")

    except Exception as error:
        print(f"[{stream_id}] ERROR: {error}")
        print(f"[{stream_id}] Stream stopped safely")


async def main():
    print("========================================")
    print(" VisionEdge - Failure Isolation Test")
    print("========================================\n")

    tasks = [
        asyncio.create_task(
            process_stream("Stream-01")
        ),
        asyncio.create_task(
            process_stream("Stream-02", should_fail=True)
        ),
        asyncio.create_task(
            process_stream("Stream-03")
        ),
        asyncio.create_task(
            process_stream("Stream-04")
        ),
    ]

    await asyncio.gather(*tasks)

    print("\n========================================")
    print(" Failure isolation test completed")
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())