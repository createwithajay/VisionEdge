# Note: TensorRT 10+/11.x API Changes (discovered during Week 1 build)

If you're on a recent TensorRT install (10.x or 11.x), two things changed
from older tutorials/documentation you'll find online:

1. **`NetworkDefinitionCreationFlag.EXPLICIT_BATCH` no longer exists.**
   Explicit batch is now the only mode, so the flag was removed entirely.
   Just call `builder.create_network()` with no arguments.

2. **`BuilderFlag.FP16` may not be exposed the same way**, depending on
   your exact build. Precision control shifted toward strongly-typed ONNX
   graphs (precision embedded in the exported model) rather than a global
   builder flag. Check what's actually available in your install with:
   ```bash
   python -c "import tensorrt as trt; print([f for f in dir(trt.BuilderFlag) if not f.startswith('_')])"
   ```
   If `FP16` isn't in that list, the engine will build fine in default
   precision — functionally correct, just not FP16-optimized. Revisiting
   true FP16 via strongly-typed export is a reasonable Week 3 backlog item,
   not a Week 1/2 blocker.

Both of `02_build_tensorrt_engine.py` and the benchmark scripts in this kit
already account for this — they detect the missing flag and print a clear
warning instead of crashing. If you're debugging a fresh TensorRT version
mismatch elsewhere, this is the first thing to check.
