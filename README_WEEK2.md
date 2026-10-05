# Vision Edge — Member 3 — Week 2
### Performance Audit: Connecting Frames, Proving the Speedup, Profiling for Leaks

Week 1 built the engine. Week 2 proves it's actually good — with numbers,
not vibes. This is the part of the project a mentor can actually look at
and immediately see the work is real: a reproducible benchmark, a sustained
streaming test, and a memory-leak profile with a chart, not just a claim.

---

## What this week delivers (per the SOP)

> *"Connect the frames to the TensorRT inference loop and measure initial
> latency. Prove the 3x FPS speedup and profile for VRAM memory leaks
> during continuous streaming."*

Three scripts, each producing hard evidence:

| Script | Proves |
|---|---|
| `04_benchmark_pytorch_vs_tensorrt.py` | TensorRT is at least 3x faster than native PyTorch, same model, same GPU |
| `05_streaming_inference_loop.py` | The engine holds up under continuous frame-by-frame load, not just a single warm call |
| `06_vram_leak_profiler.py` | VRAM usage stays flat over thousands of iterations — no leak waiting to crash the Final Review demo |

Running all three (`run_week2.sh`) produces a `results/` folder with JSON,
CSV, and PNG chart outputs — exactly what you'd drop into a Mid-Review
slide or hand to Member 2/Member 4 to plan their own integration timing.

---

## Why this is the strongest deliverable to show a mentor

Most teams at this stage can *say* their pipeline is fast. This gives you
three things most won't have:
1. **A side-by-side chart** (`benchmark_chart.png`) showing PyTorch vs
   TensorRT FPS bars, with the 3x target line drawn in — immediately
   legible in a 5-second glance, no explanation needed.
2. **A sustained-load latency log**, not just a single best-case number —
   p50/p95/p99 latency over hundreds of frames is what a real production
   system reports, not a cherry-picked single run.
3. **A VRAM trend chart with a fitted slope** — "clean" isn't a claim here,
   it's a measured near-zero slope over 3,000 iterations. If a mentor asks
   "how do you know it doesn't leak," you have an actual answer.

---

## How to run it

Assumes Week 1 is already done (`engines/yolov10n_fp16.engine` exists —
see `README.md` if not).

```bash
bash scripts/run_week2.sh models/yolov10n.pt engines/yolov10n_fp16.engine
```

Or run each stage individually:

```bash
# Stage 1 — prove the speedup
python scripts/04_benchmark_pytorch_vs_tensorrt.py \
    --weights models/yolov10n.pt --engine engines/yolov10n_fp16.engine --iterations 200

# Stage 2 — connect frames, measure sustained latency
python scripts/05_streaming_inference_loop.py \
    --engine engines/yolov10n_fp16.engine --frames 500

# Stage 3 — profile for VRAM leaks
python scripts/06_vram_leak_profiler.py \
    --engine engines/yolov10n_fp16.engine --iterations 3000
```

All outputs land in `results/`.

---

## A note on honesty (read this before the Mid-Review)

If your TensorRT build doesn't fall cleanly on FP16 (some TensorRT 10+
installs no longer expose the old `BuilderFlag.FP16` the same way — see
`docs/WEEK1_NOTES_TRT_API.md` if you hit this in Week 1), your speedup
number may land a bit under 3x rather than comfortably over it. **Report
the real number.** A mentor trusts a team more for an honest 2.6x with a
clear explanation of why (default-precision build, small model, tiny
input) than a suspiciously round "3.1x" that doesn't hold up to a follow-up
question. The scripts here print an explicit note when the target isn't
met, and the check is automatic and visible — not something you have to
remember to mention. If you do land under 3x, the honest fix for later is
revisiting FP16 via strongly-typed ONNX export, which is a legitimate,
well-scoped thing to put on the Week 3 backlog rather than something to
hide now.

---

## What to say in 60 seconds at the Mid-Review

*"I connected the TensorRT engine to a continuous frame loop and benchmarked
it against native PyTorch under identical conditions. [State your actual
speedup number]x faster, with p95 latency of [X] ms under sustained load.
I also ran 3,000 continuous inference passes while tracking GPU memory —
it stayed flat, confirming no leak, which matters because the real pipeline
will run for hours, not seconds."*

That's it. Chart up, number stated, done.
