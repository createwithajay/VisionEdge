# VisionEdge — Video Stream Orchestration

## Overview

VisionEdge is a CPU-testable baseline for concurrent video stream orchestration using Python's `asyncio`. It demonstrates concurrent processing, per-stream failure isolation, bounded retries, and benchmark reporting.

**Current status:** CPU baseline tested. NVIDIA GPU acceleration and the production target of 10 simultaneous 4K streams at 60 FPS have not been verified.

## Implementation

- **Concurrency:** `asyncio` and `asyncio.gather()`
- **Thread execution:** `asyncio.to_thread()`
- **Stream management:** `StreamManager` with a configurable concurrency limit
- **Video decoding:** PyAV
- **Frame processing:** Lightweight NumPy-based CPU workload
- **Reliability:** Per-stream error reporting and bounded retries
- **Reporting:** JSON benchmark and orchestration reports

## Tests and Results

### Week 1 — Video Decoding

- Successfully decoded 90 frames from `sample_video.mp4`.
- Confirmed the decoded frame array had shape `(432, 768, 3)`.
- Added checks for missing files, missing video streams, and empty decoding results.

### Week 2 — CPU Profiling Baseline

- Frames processed: 150
- Average workload latency: 2.045 ms per frame
- 95th percentile workload latency: 2.682 ms
- Process RAM at the end of the run: approximately 50.73 MB
- Reciprocal latency estimate: approximately 488.98 workload operations per second

These measurements describe the lightweight CPU workload, not YOLO or TensorRT inference. PyTorch-versus-TensorRT speedup and GPU memory usage were not benchmarked.

Report: `week2_audit_report.json`

### Week 3 — Sequential vs. Concurrent Processing

Latest recorded benchmark run:

| Metric | Sequential | Concurrent |
|---|---:|---:|
| Streams | 4 | 4 |
| Total frames | 240 | 240 |
| Elapsed time | 0.4173 s | 0.1501 s |
| Aggregate workload throughput | 575.16 FPS | 1,598.53 FPS |

**Observed speedup: 2.78×.**

The result applies to this particular CPU-based benchmark run. It does not establish real-time video performance or the target FPS for each stream.

Report: `week3_benchmark_report.json`

### Week 3 — Failure Isolation

Tested three streams, with one stream pointing to a nonexistent video file.

- Successful streams: 2
- Failed streams: 1
- Result: The two valid streams completed even though the other stream failed.

### Week 4 — Orchestration Reliability

Five automated tests passed:

1. Concurrency limit enforcement
2. Invalid concurrency configuration
3. Failed-stream reporting
4. Recovery after a transient failure
5. Retry-limit enforcement

Command used:

```cmd
python -m unittest week4_orchestration_test -v
```

## Running the Tests

Run these commands from the project root:

```cmd
python week1_decoder.py
python week2_inference_profiling.py
python week3_benchmark.py
python week3_failure_test.py
python -m unittest week4_orchestration_test -v
```

The scripts require their relevant dependencies and sample video file to be available.

## Generated Reports

- `week2_audit_report.json`
- `week3_benchmark_report.json`
- `week3_orchestration_report.json`
- `week4_orchestration_test_report.json`

## Limitations and Next Steps

- NVIDIA GPU acceleration has not been verified.
- The current frame-processing workload is not actual YOLO or TensorRT inference.
- CuPy zero-copy integration and CUDA bounding-box kernels are not implemented in this baseline.
- The 10-stream, 4K, 60-FPS target has not been validated.
- Production validation requires compatible GPU hardware, representative independent video streams, sustained-load testing, and accurate per-stream FPS and latency measurements.

## Summary

The current baseline demonstrates asynchronous stream scheduling, failure isolation, bounded retries, and improved throughput in a tested CPU workload. GPU acceleration and production performance remain unverified.
