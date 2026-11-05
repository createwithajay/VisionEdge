
import asyncio
import unittest
from unittest.mock import patch

from orchestrator.stream_manager import StreamManager


class Week4OrchestrationTests(unittest.IsolatedAsyncioTestCase):

    async def test_concurrency_limit(self):
        manager = StreamManager(max_concurrent_streams=2)
        active = 0
        max_active = 0

        async def fake_stream():
            nonlocal active, max_active
            async with manager.semaphore:
                active += 1
                max_active = max(max_active, active)
                try:
                    await asyncio.sleep(0.03)
                finally:
                    active -= 1

        await asyncio.gather(*(fake_stream() for _ in range(5)))
        self.assertLessEqual(max_active, 2)
        self.assertEqual(active, 0)

    async def test_invalid_concurrency_limit(self):
        with self.assertRaises(ValueError):
            StreamManager(max_concurrent_streams=0)

    async def test_invalid_stream_is_reported(self):
        manager = StreamManager(max_concurrent_streams=2, max_retries=0)
        results = await manager.run_streams([
            ("Stream-01", "sample_video.mp4"),
            ("Stream-02", "missing_week4_video.mp4"),
        ], max_frames=5)

        self.assertEqual(results[0].status, "completed")
        self.assertEqual(results[1].status, "failed")
        self.assertIn("FileNotFoundError", results[1].error)

    async def test_transient_failure_recovers(self):
        manager = StreamManager(
            max_concurrent_streams=1,
            max_retries=1,
            retry_delay=0,
        )

        with patch(
            "orchestrator.stream_manager.decode_and_process",
            side_effect=[OSError("temporary error"), 5],
        ):
            result = await manager.process_stream(
                "Stream-Retry", "unused.mp4", max_frames=5
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.frames_processed, 5)
        self.assertEqual(result.attempts, 2)

    async def test_retry_limit_is_respected(self):
        manager = StreamManager(
            max_concurrent_streams=1,
            max_retries=2,
            retry_delay=0,
        )

        with patch(
            "orchestrator.stream_manager.decode_and_process",
            side_effect=OSError("persistent error"),
        ) as mocked:
            result = await manager.process_stream(
                "Stream-Fail", "unused.mp4", max_frames=5
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(mocked.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
