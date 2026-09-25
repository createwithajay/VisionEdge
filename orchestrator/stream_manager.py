import asyncio
import time
from dataclasses import dataclass


@dataclass
class StreamState:
    stream_id: str
    status: str = "CREATED"
    frames_processed: int = 0
    start_time: float = 0.0


class StreamManager:
    """
    Manages multiple independent video stream tasks
    using Python asyncio.
    """

    def __init__(self):
        self.streams: dict[str, StreamState] = {}

    async def process_stream(self, stream_id: str, frame_count: int = 20):
        """
        Simulates asynchronous processing of one video stream.
        """

        state = StreamState(
            stream_id=stream_id,
            status="RUNNING",
            start_time=time.perf_counter(),
        )

        self.streams[stream_id] = state

        print(f"[{stream_id}] Started")

        try:
            for frame_number in range(1, frame_count + 1):

                # Simulate frame processing.
                await asyncio.sleep(0.05)

                state.frames_processed += 1

                print(
                    f"[{stream_id}] "
                    f"Frame {frame_number} processed"
                )

            state.status = "COMPLETED"

            elapsed = time.perf_counter() - state.start_time

            fps = (
                state.frames_processed / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"[{stream_id}] Completed | "
                f"Frames: {state.frames_processed} | "
                f"FPS: {fps:.2f}"
            )

        except asyncio.CancelledError:
            state.status = "CANCELLED"
            print(f"[{stream_id}] Cancelled")
            raise

        except Exception as error:
            state.status = "ERROR"
            print(f"[{stream_id}] Error: {error}")

    async def run_streams(self, stream_ids: list[str]):
        """
        Creates and runs multiple stream tasks concurrently.
        """

        tasks = [
            asyncio.create_task(
                self.process_stream(stream_id)
            )
            for stream_id in stream_ids
        ]

        await asyncio.gather(*tasks)

    def print_summary(self):
        """
        Displays the final state of all streams.
        """

        print("\n========== STREAM SUMMARY ==========")

        for stream_id, state in self.streams.items():
            print(
                f"{stream_id} | "
                f"Status: {state.status} | "
                f"Frames: {state.frames_processed}"
            )

        print("====================================")