# Week 2 Checklist — Member 3

## Prerequisites
- [ ] Week 1 complete: `engines/yolov10n_fp16.engine` exists and verified
- [ ] `models/yolov10n.pt` still present (needed for the PyTorch baseline comparison)
- [ ] `pip install matplotlib` if not already installed (for the chart outputs)

## Stage 1 — Benchmark (3x speedup proof)
- [ ] `04_benchmark_pytorch_vs_tensorrt.py` runs successfully
- [ ] `results/benchmark_results.json` exists with both backend numbers
- [ ] `results/benchmark_chart.png` exists and is legible
- [ ] You know your actual speedup number (whatever it is — honestly reported)
- [ ] If under 3x: you understand why (see README_WEEK2.md's honesty note)
      and can explain it in one sentence

## Stage 2 — Streaming inference loop
- [ ] `05_streaming_inference_loop.py` runs successfully on at least 500 frames
- [ ] `results/streaming_latency_log.csv` exists
- [ ] You have p50/p95/p99 latency numbers, not just a single average
- [ ] You've shared the sustained latency number with Member 2 (they need it
      to plan decoder-to-inference buffering) and Member 4 (WebRTC timing)

## Stage 3 — VRAM leak profile
- [ ] `06_vram_leak_profiler.py` runs for at least 3,000 iterations
- [ ] `results/vram_profile_chart.png` shows a flat (or near-flat) trend line
- [ ] Verdict printed is "CLEAN" — if "LEAK DETECTED," you've investigated
      and either fixed it or documented the suspected cause before the
      Mid-Review (don't walk in with an unexplained leak on the slide)

## Mid-Review Readiness
- [ ] You can state your real speedup number without checking notes
- [ ] You can explain what p95 latency means and why it matters more than
      a single best-case number for a streaming system
- [ ] You can explain how the VRAM profile works (continuous loop, reused
      buffers, memory sampled over time, slope fitted to detect drift)
- [ ] All three chart/log files are either committed to the repo or ready
      to screen-share live

## Commit Hygiene (remember: 10/14 days needed before Mid-Review)
- [ ] See `docs/WEEK2_COMMIT_PLAN.md` for a natural, honest commit cadence
      that covers this week's work
