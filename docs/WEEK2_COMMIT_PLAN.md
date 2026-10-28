# Week 2 Commit Plan — Member 3 (max 15 commits)

Same principle as Week 1: commit at real working checkpoints, not at
arbitrary end-of-day dumps. Below is a natural 15-commit breakdown of this
week's actual work — use it as a guide, not a script to copy blindly. If
your real work goes in a slightly different order, commit in *that* order;
an honest commit history beats a tidy fake one.

```
1.  chore: scaffold week2 benchmark and profiling scripts
2.  feat: add native PyTorch baseline benchmark function
3.  feat: add TensorRT engine benchmark function
4.  feat: compute speedup ratio and print comparison summary
5.  feat: save benchmark results to JSON for downstream reporting
6.  feat: add matplotlib chart generation for benchmark results
7.  fix: correct warmup iteration count affecting early benchmark noise
8.  feat: add synthetic frame generator for streaming loop (no decoder dependency)
9.  feat: implement continuous streaming inference loop with per-frame timing
10. feat: add p50/p95/p99 latency percentile reporting
11. feat: add VRAM leak profiler with periodic memory sampling
12. feat: add linear regression slope detection for leak verdict
13. feat: add VRAM trend chart with fitted line overlay
14. feat: add run_week2.sh to orchestrate full performance audit in one command
15. docs: write week2 README, checklist, and honest reporting notes for mid-review
```

## Why this order makes sense (so it doesn't feel arbitrary)
You genuinely can't write the chart-generation commit before the benchmark
numbers exist to chart (6 depends on 2-5). You can't write the leak
*verdict* logic before you have raw samples to analyze (12 depends on 11).
Following dependency order isn't just good practice for hitting commit
targets — it's also just how this code actually has to be built.

## If you finish faster than 15 commits' worth of natural checkpoints
Don't pad with filler commits — that's more obvious to a reviewer than an
honest 11-commit week. Instead, use remaining time on polish that's
genuinely useful going into Week 3:
- Add a `--imgsz` sweep to the benchmark script (how does speedup change at
  different resolutions? directly useful once Member 1 starts batching
  multiple streams)
- Add a simple `results/SUMMARY.md` auto-generated from the JSON, so
  Member 4/5 can read your numbers without running the scripts themselves

## If you hit a wall (TensorRT API mismatch, speedup under 3x, etc.)
Commit the investigation anyway:
```
fix: wip - investigating sub-3x speedup, suspect default-precision build
```
This is genuinely more valuable in your history than silence, and it's
exactly the kind of commit message that makes a Mid-Review conversation
easier, not harder — you're showing the debugging process, not just the
clean result.
