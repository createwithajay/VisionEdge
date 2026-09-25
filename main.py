import asyncio

from orchestrator.stream_manager import StreamManager


async def main():
    manager = StreamManager()

    stream_ids = [
        "Stream-01",
        "Stream-02",
        "Stream-03",
        "Stream-04",
    ]

    print("========================================")
    print(" VisionEdge - Asyncio Stream Orchestrator")
    print("========================================")
    print(f"Starting {len(stream_ids)} concurrent streams...\n")

    await manager.run_streams(stream_ids)

    manager.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
