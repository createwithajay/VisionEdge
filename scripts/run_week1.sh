#!/usr/bin/env bash
# Vision Edge — Member 3, Week 1 — one-shot pipeline runner
#
# Usage:
#   bash scripts/run_week1.sh models/yolov10n.pt
#
# Runs export -> compile -> verify back to back. Stops immediately if any
# step fails so you're not debugging a downstream error caused upstream.

set -e  # exit on first error

WEIGHTS="${1:-models/yolov10n.pt}"
IMGSZ="${2:-640}"

echo "=================================================================="
echo " Vision Edge — Member 3 — Week 1 pipeline"
echo " Weights: ${WEIGHTS}"
echo " Image size: ${IMGSZ}"
echo "=================================================================="

echo ""
echo "[1/3] Exporting to ONNX..."
python scripts/01_export_onnx.py --weights "${WEIGHTS}" --imgsz "${IMGSZ}" --simplify

ONNX_PATH="models/$(basename "${WEIGHTS}" .pt).onnx"

echo ""
echo "[2/3] Building TensorRT engine..."
python scripts/02_build_tensorrt_engine.py --onnx "${ONNX_PATH}" --fp16

ENGINE_PATH="engines/$(basename "${ONNX_PATH}" .onnx)_fp16.engine"

echo ""
echo "[3/3] Verifying engine..."
python scripts/03_verify_engine.py --engine "${ENGINE_PATH}"

echo ""
echo "=================================================================="
echo " Week 1 pipeline complete."
echo " Engine ready at: ${ENGINE_PATH}"
echo "=================================================================="
