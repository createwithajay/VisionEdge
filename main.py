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
        print(f"❌ Error during {phase_name}. Exit code: {result.returncode}")
        sys.exit(result.returncode)
    print(f" Completed {phase_name} in {elapsed:.2f}s")

def main():
    print("=" * 70)
    print("AXLERO SOLUTIONS - VISIONEDGE END-TO-END PIPELINE DEMONSTRATION")
    print("Hardware-Accelerated Computer Vision & Edge Profiling System")
    print("=" * 70)

    # Week 1: Hardware-Accelerated Video Decoding (PyAV / NVDEC)
    run_phase("Phase 1: PyAV Stream Extraction", "week1_decoder.py")

    # Week 2: Inference Loop Latency & 3x Speedup Performance Audit
    run_phase("Phase 2: Inference Latency & Leak Profiling", "week2_inference_profiling.py")

    # Week 3: CuPy Zero-Copy Bridge & GPU In-Place Rendering
    run_phase("Phase 3: Zero-Copy Bridge & CUDA Kernels", "week3_zero_copy_bridge.py")

    # Week 4: Asyncio MultiStream Orchestration & Dynamic Model Swapping
    run_phase("Phase 4: MultiStream Orchestration (asyncio)", "week4_multistream_pipeline.py")

    print("\n" + "=" * 70)
    print("ALL VISIONEDGE PIPELINE PHASES EXECUTED SUCCESSFULLY")
    print("Audit Artifacts Ready: week2_audit_report.json, week3_zero_copy_report.json, week4_final_pipeline_report.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
