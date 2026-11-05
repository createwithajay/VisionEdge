# Troubleshooting — Member 3 Week 1

Real problems you're likely to actually hit, in the order you'll hit them.

---

### "pip install tensorrt" fails or can't find a matching wheel
TensorRT's pip packaging is version-sensitive to your CUDA install. Two reliable
fallbacks:

1. **NVIDIA TensorRT Docker image** (most reliable, recommended if you're stuck):
   ```bash
   docker run --gpus all -it --rm \
     -v $(pwd):/workspace \
     nvcr.io/nvidia/tensorrt:24.08-py3
   cd /workspace
   pip install ultralytics onnx onnxsim onnxruntime-gpu
   ```
   This container already has TensorRT + CUDA correctly matched.

2. **Google Colab**: Colab's GPU runtimes generally have TensorRT preinstalled
   or one version-matched pip command away. Check `!nvidia-smi` and
   `!python -c "import tensorrt"` first before installing anything.

---

### ONNX export succeeds but `03_verify_engine.py` gives garbage output shapes
Almost always a dynamic-axis mismatch. Re-run the export with `--imgsz` matching
exactly what you pass at inference time, or use `--dynamic` consistently through
both the export and engine-build steps rather than mixing static and dynamic.

---

### "insufficient workspace" or engine build silently returns None
Raise `--workspace` in `02_build_tensorrt_engine.py` (try 4096, then 8192 MB)
if you have VRAM headroom. If you're on a small GPU (≤6GB), stick with
`yolov10n` (nano) rather than larger variants — `yolov10l`/`yolov10x` can
genuinely be too large to compile comfortably on entry-level cards.

---

### Engine built on my laptop doesn't run on the lab machine / teammate's GPU
This is expected, not a bug. TensorRT engines are compiled for a specific
GPU architecture + TensorRT version + CUDA version combination. **Never commit
`.engine` files to git as the "source of truth"** — commit the `.onnx` file and
the build script instead, and have each machine compile its own local engine.
Document this clearly for Member 4, since they'll be building the "swap engine
files on the fly" feature in Week 4 and need to know engines aren't portable
across different deployment GPUs.

---

### `pycuda` import fails / CUDA context errors in `03_verify_engine.py`
Make sure only one CUDA context is being created. If you're running this
inside a Jupyter notebook that already imported `pycuda.autoinit` elsewhere,
restart the kernel. This script is designed to run as a standalone `python`
process, not interleaved with other GPU-context-creating code.

---

### My FPS number looks worse than expected
Don't panic yet — this script measures raw single-frame TensorRT latency with
no batching, no CUDA streams overlap, and no zero-copy frame handoff. The real,
representative benchmark happens in **Week 2** once frames are coming in live
from Member 2's decoder and you're comparing against a proper native-PyTorch
baseline. A "good enough for now" number here is fine; don't over-optimize
prematurely.

---

### General debugging order
When something breaks, check in this order — it's the order things usually
actually go wrong in:
1. Is the `.onnx` file even valid? (`onnxruntime` sanity check in step 1)
2. Did `.onnx` → `.engine` compile without silent failure? (check the build log)
3. Does the engine load and produce *any* output? (step 3)
4. Only then worry about whether the output values/shapes are semantically correct
