#!/usr/bin/env bash
# Vision Edge — Member 3, Week 2 — one-shot pipeline runner
#
# Usage:
#   bash scripts/run_week2.sh models/yolov10n.pt engines/yolov10n_fp16.engine
#
# Runs: benchmark (3x speedup proof) -> streaming inference loop -> VRAM leak profile.
# Produces everything needed for the Mid-Review performance audit in one command.

set -e

WEIGHTS="${1:-models/yolov10n.pt}"
ENGINE="${2:-engines/yolov10n_fp16.engine}"

echo "=================================================================="
echo " Vision Edge — Member 3 — Week 2 performance audit"
echo " Weights: ${WEIGHTS}"
echo " Engine:  ${ENGINE}"
echo "=================================================================="

if [ ! -f "${ENGINE}" ]; then
    echo "[error] Engine not found at ${ENGINE}. Run Week 1's pipeline first:"
    echo "    bash scripts/run_week1.sh ${WEIGHTS}"
    exit 1
fi

echo ""
echo "[1/3] Benchmarking native PyTorch vs TensorRT (3x speedup proof)..."
python scripts/04_benchmark_pytorch_vs_tensorrt.py --weights "${WEIGHTS}" --engine "${ENGINE}" --iterations 200

echo ""
echo "[2/3] Running streaming inference loop (sustained latency)..."
python scripts/05_streaming_inference_loop.py --engine "${ENGINE}" --frames 500

echo ""
echo "[3/3] Profiling VRAM for leaks during continuous inference..."
python scripts/06_vram_leak_profiler.py --engine "${ENGINE}" --iterations 3000

echo ""
echo "=================================================================="
echo " Week 2 performance audit complete."
echo " Results saved in: results/"
echo "   - benchmark_results.json / benchmark_chart.png"
echo "   - streaming_latency_log.csv"
echo "   - vram_profile_log.csv / vram_profile_chart.png"
echo "=================================================================="
