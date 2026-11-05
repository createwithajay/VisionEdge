"""VisionEdge Week 1: Safe video frame decoding."""

import sys
from pathlib import Path

import av


def extract_raw_frames(video_path="sample_video.mp4", max_frames=90):
    path = Path(video_path)

    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    print(f"Opening video: {path.resolve()}")
    frame_count = 0

    # Use standard decoding unless hardware acceleration is explicitly
    # configured and verified on a supported machine.
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found: {path}")

        video_stream = container.streams.video[0]
        print("Video stream found. Starting frame decoding...")

        for frame in container.decode(video_stream):
            raw_frame_array = frame.to_ndarray(format="bgr24")
            frame_count += 1

            if frame_count == 1:
                print(
                    "First frame decoded successfully. "
                    f"Array shape: {raw_frame_array.shape}"
                )

            if frame_count % 30 == 0:
                print(f"Successfully decoded {frame_count} frames.")

            if frame_count >= max_frames:
                break

    if frame_count == 0:
        raise ValueError(f"No video frames could be decoded: {path}")

    print(f"\nWeek 1 complete: decoded {frame_count} frames.")
    return frame_count


if __name__ == "__main__":
    try:
        extract_raw_frames()
    except (FileNotFoundError, ValueError, av.FFmpegError) as exc:
        print(f"Decoding failed: {exc}", file=sys.stderr)
        sys.exit(1)