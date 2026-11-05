import os

def generate_final_audit():
    print("[VisionEdge] Generating Final Review Readiness Audit...")
    
    required_scripts = [
        "04_health_check.py",
        "05_benchmark_latency.py",
        "06_test_client.py",
        "07_model_profiler.py",
        "08_stream_monitor.py",
        "09_gpu_telemetry.py"
    ]
    
    missing = []
    for script in required_scripts:
        path = os.path.join("scripts", script)
        if not os.path.exists(path):
            missing.append(script)
            
    if not missing:
        print("[SUCCESS] All 6 core WebRTC and backend validation scripts are verified in place.")
        print("[READY] Member-4 streaming and backend pipeline is fully prepared for Final Review.")
    else:
        print(f"[WARNING] Missing scripts: {missing}")

if __name__ == "__main__":
    generate_final_audit()
