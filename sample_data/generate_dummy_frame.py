"""
Optional helper: generate a synthetic test image if you don't yet have access
to a real RTSP stream or sample video (Member 2 owns the real stream ingestion,
but you don't need to wait on them to test your export/compile pipeline).

Usage:
    python generate_dummy_frame.py --imgsz 640 --out sample_data/dummy.jpg
"""

import argparse

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=str, default="sample_data/dummy.jpg")
    args = parser.parse_args()

    # Random noise image is enough to sanity-check that a model runs end to end;
    # it won't produce meaningful detections, only structural correctness.
    frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    cv2.imwrite(args.out, frame)
    print(f"[success] Wrote dummy test frame to {args.out}")


if __name__ == "__main__":
    main()
