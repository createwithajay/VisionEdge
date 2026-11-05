import subprocess
import sys
import time

def run_phase(phase_name, script_file):
    print("\n" + "=" * 70)
    print(f"EXECUTING: {phase_name} ({script_file})")
    print("=" * 70)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, script_file], capture_output=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"Error during {phase_name}. Exit code: {result.returncode}")
        sys.exit(result.returncode)
    print(f"Completed {phase_name} in {elapsed:.2f}s")

def main():
    print("=" * 70)
    print("AXLERO SOLUTIONS - VISIONEDGE COMPLETE END-TO-END PIPELINE")
    print("Hardware Edge Engineering & Accelerated AI Inference")
    print("=" * 70)

    run_phase("Phase 1: PyAV Stream Extraction", "week1_decoder.py")
    run_phase("Phase 2: Inference Latency & Leak Profiling", "week2_inference_profiling.py")
    run_phase("Phase 3: Zero-Copy Bridge Baseline", "week3_zero_copy_bridge.py")
    run_phase("Phase 4: MultiStream Orchestration (asyncio)", "week4_multistream_pipeline.py")
    run_phase("Phase 5: Member 3 CuPy Zero-Copy CUDA Kernels", "member3_zero_copy_cuda.py")
    run_phase("Phase 6: Member 3 TensorRT Dynamic Engine Hot-Swap", "member3_tensorrt_dynamic_engine.py")

    print("\n" + "=" * 70)
    print("ALL VISIONEDGE MODULES (MEMBER 2 + REALLOCATED MEMBER 3) EXECUTED SUCCESSFULLY")
    print("All JSON Audit Reports Ready for Evaluation.")
    print("=" * 70)

if __name__ == "__main__":
    main()
