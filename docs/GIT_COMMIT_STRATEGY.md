# Git Commit Strategy — Member 3

The Team Lead needs to show commits on **10 of the 14 days** before the Mid
Review (and later, a strict 20-day unbroken streak before the Final Review).
That's a team-wide requirement, not busywork specific to you — but you can make
it trivial to satisfy honestly, without gaming it.

## The honest way to hit this naturally

Real engineering work on this pipeline breaks into small, commit-worthy chunks
almost by default:

```
day 1: chore: scaffold onnx export script skeleton
day 2: feat: implement basic yolov10 -> onnx export
day 3: fix: correct dynamic axis naming for batch dimension
day 4: test: add onnxruntime sanity check after export
day 5: feat: implement tensorrt engine builder (fp32 baseline)
day 6: feat: add fp16 precision flag to engine builder
day 7: fix: increase default workspace size to fix build failure
day 8: feat: add engine verification + latency script
day 9: docs: write week1 checklist and troubleshooting notes
day 10: chore: clean up script arg parsing, add run_week1.sh convenience runner
```

Notice none of these are fake — they're the actual natural checkpoints of doing
this work properly. If you build sequentially and commit at each *working*
checkpoint (not just at the end of the day), you'll clear 10/14 without
thinking about it.

## Two rules worth following
1. **Commit when something works, not when you stop for the day.** A commit
   that captures "export now handles dynamic batch correctly" is far more
   useful (to you, in Week 2, when you need to remember what changed and why)
   than a vague end-of-day dump.
2. **Write commit messages that would make sense to Member 2 or Member 4**,
   since they depend on your engine file. "fixed stuff" tells them nothing;
   "fix: engine now accepts variable input resolution" tells them exactly
   what changed and why they might care.

## If you get stuck for a day or two
Commit anyway — even a small "wip: debugging tensorrt workspace memory error,
see notes" commit keeps the streak alive *and* is genuinely useful documentation
of where you got stuck, in case you need to explain a delay at the Mid Review.
Silence for 48 hours is what triggers the Team Lead's escalation clause in the
SOP — a small commit costs you two minutes and avoids that entirely.
