# Week 1 Checklist — Member 3

Print this or pin it. Check off as you go. This is what "done" looks like for
your Mid-Review contribution.

## Environment
- [ ] `nvidia-smi` runs and shows your GPU
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python -c "import tensorrt; print(tensorrt.__version__)"` prints a version

## Model
- [ ] YOLOv10 checkpoint downloaded (`yolov10n.pt` recommended to start — nano
      variant, fastest to iterate with)
- [ ] Checkpoint moved into `models/`

## Export (Step 1)
- [ ] `01_export_onnx.py` runs successfully
- [ ] The script's built-in onnxruntime sanity check passes (prints output shapes,
      no exceptions)
- [ ] `models/yolov10n.onnx` exists and is a non-trivial file size (tens of MB,
      not a few KB — a tiny file usually means the export silently failed)

## Compile (Step 2)
- [ ] `02_build_tensorrt_engine.py --fp16` runs successfully
- [ ] `engines/yolov10n_fp16.engine` exists
- [ ] You understand *why* this file won't run on a teammate's different GPU
      model (it's compiled for your specific hardware — this is expected)

## Verify (Step 3)
- [ ] `03_verify_engine.py` runs successfully
- [ ] Output shape matches what you'd expect for YOLO detection output
      (roughly `[batch, num_boxes, 4 + num_classes]` depending on export head)
- [ ] You have a rough latency number written down somewhere (even a screenshot)
      — Member 2 will want this to plan Week 2 integration timing

## Git / Process
- [ ] Your own branch exists and you're the only one pushing to it
- [ ] At least one commit made (ideally: today, and most days this week)
- [ ] You've told the Team Lead your Week 1 deliverable status honestly —
      if you're blocked on GPU access or a broken install, flag it *now*,
      not two days before the Mid Review

## Mid-Review Readiness
- [ ] You can explain your pipeline in under 60 seconds: "I exported YOLOv10
      to ONNX, validated the export, compiled it into a TensorRT engine with
      FP16 precision, and confirmed it runs inference correctly."
- [ ] You know your rough single-frame latency number
- [ ] You know what you're doing in Week 2 (connecting this engine to real
      video frames from Member 2's decoder, and formally proving the 3x
      speedup vs native PyTorch)
