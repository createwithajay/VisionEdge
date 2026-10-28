import asyncio
import time
import unittest

from orchestrator.stream_manager import StreamManager


class Week4OrchestrationTests(unittest.IsolatedAsyncioTestCase):

    async def test_concurrency_limit(self):
        manager = StreamManager(max_concurrent_streams=2)
        active = 0
        max_active = 0

        async def fake_stream(stream_id, video_path, max_frames=90):
            nonlocal active, max_active

            async with manager.semaphore:
                active += 1
                max_active = max(max_active, active)

                try:
                    await asyncio.sleep(0.05)
                finally:
                    active -= 1

        tasks = [
            fake_stream(f"Stream-{i}", "unused.mp4")
            for i in range(5)
        ]

        await asyncio.gather(*tasks)

        self.assertLessEqual(max_active, 2)
        self.assertEqual(active, 0)

    async def test_invalid_stream_is_reported(self):
        manager = StreamManager(max_concurrent_streams=2)

        results = await manager.run_streams(
            [
                ("Stream-01", "sample_video.mp4"),
                ("Stream-02", "missing_week4_video.mp4"),
            ],
            max_frames=5,
        )

        self.assertEqual(results[0].status, "completed")
        self.assertEqual(results[1].status, "failed")
        self.assertIn("FileNotFoundError", results[1].error)

    async def test_invalid_concurrency_limit(self):
        with self.assertRaises(ValueError):
            StreamManager(max_concurrent_streams=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)