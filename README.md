# Vision Edge — Member 3 Toolkit
### Role: AI Inference & CUDA Specialist | Phase: Week 1

This folder is your personal execution kit for the first week of the Vision Edge
project. It's built directly off the SOP you were given, so nothing here is generic —
every script maps to a specific line item in your role description.

---

## 1. Where This Fits In The Project

Quick recap so you always know why you're doing what you're doing:

| Week | Your Job |
|------|----------|
| **Week 1 (you are here)** | Export YOLOv10 from PyTorch → ONNX, then compile the ONNX graph into a TensorRT engine |
| Week 2 | Connect frames to the TensorRT inference loop, measure latency, prove 3x FPS speedup vs PyTorch, profile for VRAM leaks (joint with Member 2) |
| Week 3 | Build the CuPy Zero-Copy Bridge (GPU-to-GPU frame handoff, no system RAM), draw bounding boxes with CUDA kernels |
| Week 4 | Support integration, performance audit polish, final review prep |

Your Week 1 output (a working `.engine` file) is the single hard dependency Member 2
and Member 4 need before they can plug inference into the video pipeline in Week 2.
Get this right and the whole team's Week 2 unblocks on time.

---

## 2. Week 1 Deliverables (Definition of Done)

By the end of this week you should be able to check off all four:

- [ ] YOLOv10 model exported to `.onnx` with a validated forward pass
- [ ] ONNX graph compiled into a serialized TensorRT `.engine` file
- [ ] Engine loads and runs a sanity inference without crashing
- [ ] At least one commit per day in your branch (Team Lead is tracking this for the Mid Review — 10 commit-days out of 14 needed)

---

## 3. Folder Structure

```
vision_edge_member3/
├── README.md                     ← you are here
├── requirements.txt               ← exact package list
├── scripts/
│   ├── 01_export_onnx.py          ← PyTorch → ONNX
│   ├── 02_build_tensorrt_engine.py← ONNX → TensorRT engine
│   ├── 03_verify_engine.py        ← sanity-check the compiled engine
│   └── run_week1.sh               ← runs all three in order
├── models/                        ← put your .pt weights here
├── engines/                       ← compiled .engine files land here
├── sample_data/                   ← dummy input generator for testing
└── docs/
    ├── WEEK1_CHECKLIST.md
    ├── TROUBLESHOOTING.md
    └── GIT_COMMIT_STRATEGY.md     ← how to hit the 10/14-day commit rule painlessly
```

---

## 4. Prerequisites

You need access to a machine with:
- An NVIDIA GPU (any consumer GPU with ≥6GB VRAM works for YOLOv10-small/medium)
- NVIDIA driver + CUDA Toolkit installed (`nvidia-smi` should run cleanly)
- Python 3.9–3.11

If you don't have a personal GPU box, use **Google Colab (T4/L4 runtime)** or a
university lab machine — TensorRT ships with the CUDA base images so you won't need
to install drivers from scratch there.

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `ultralytics` (for YOLOv10 + built-in ONNX export),
`onnx`, `onnxruntime-gpu`, `onnxsim`, and `tensorrt` (or use the NVIDIA TensorRT
Docker image if pip install gives you trouble — see Troubleshooting).

---

## 5. Step-by-Step: From Zero to Working Engine

### Step 1 — Get a model checkpoint
Download a pretrained YOLOv10 checkpoint (start small, prove the pipeline works,
then scale up):

```bash
python -c "from ultralytics import YOLO; YOLO('yolov10n.pt')"
```
This auto-downloads `yolov10n.pt` into your working directory — move it into
`models/`.

### Step 2 — Export to ONNX
```bash
python scripts/01_export_onnx.py --weights models/yolov10n.pt --imgsz 640 --simplify
```
This produces `models/yolov10n.onnx`. The script also runs a quick ONNX Runtime
inference on a dummy tensor to confirm the export isn't broken before you move on —
**never compile a TensorRT engine from an ONNX file you haven't validated first**,
it wastes hours of debugging later.

### Step 3 — Compile the TensorRT engine
```bash
python scripts/02_build_tensorrt_engine.py --onnx models/yolov10n.onnx --fp16
```
This builds `engines/yolov10n_fp16.engine`. FP16 mode is on by default because it's
the standard "free" speedup on consumer GPUs (this is also your first checkpoint
toward the 3x FPS goal you'll be proving formally in Week 2).

### Step 4 — Verify the engine actually runs
```bash
python scripts/03_verify_engine.py --engine engines/yolov10n_fp16.engine
```
This loads the engine, allocates GPU buffers, runs inference on a random tensor,
and prints output shapes + a rough single-frame latency number. If this runs
clean, your Week 1 deliverable is done.

### One-shot version
```bash
bash scripts/run_week1.sh
```
Runs all three steps back-to-back with your weights file as the only input.

---

## 6. Daily Commit Plan (for the Mid-Review requirement)

The Team Lead needs commits on **10 of the 14 days** before the Mid Review. Don't
dump all your work in one commit the night before — that's an obvious red flag in
the commit history and puts your team at risk. Suggested natural commit points:

| Day | Commit |
|-----|--------|
| 1 | `chore: scaffold member3 inference branch` |
| 2 | `feat: add ONNX export script` |
| 3 | `test: validate ONNX export with onnxruntime` |
| 4 | `feat: add TensorRT engine builder` |
| 5 | `fix: resolve dynamic batch axis issue in export` |
| 6 | `feat: add engine verification script` |
| 7 | `docs: add week1 checklist and notes` |
| ... | (natural fixes/tweaks as you actually hit issues) |

See `docs/GIT_COMMIT_STRATEGY.md` for more detail — this is genuinely useful, not
just box-checking, since a clean commit trail is what you'll want to point to in
the performance audit report in Week 2 anyway.

---

## 7. What "Good" Looks Like Going Into the Mid-Review

- `engines/yolov10n_fp16.engine` exists and is committed (or its build command is
  documented, since engine files are GPU/driver-specific and shouldn't always be
  committed as binaries — see note in `TROUBLESHOOTING.md`)
- You can explain, in one sentence each: why ONNX is the intermediate format, why
  TensorRT needs a GPU-specific compile step, and why FP16 is a safe first
  optimization
- You have a rough latency number ready (even an unofficial one) — Member 2 will
  ask you for it to plan the Week 2 integration

Good luck — this is the piece the whole pipeline hangs off. Ship it clean, commit
often, and you'll walk into the Mid Review with the strongest deliverable of the
five roles.
