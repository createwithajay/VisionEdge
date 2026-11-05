import os

def audit_final_review_readiness():
    print("==================================================")
    print("   VisionEdge - Final Review Compliance Audit     ")
    print("==================================================")
    
    required_paths = [
        "webrtc_server.py",
        "scripts/04_health_check.py",
        "scripts/05_benchmark_latency.py",
        "tensorrt_models"
    ]
    
    all_passed = True
    for path in required_paths:
        exists = os.path.exists(path)
        status = "[PASSED]" if exists else "[MISSING]"
        print(f"Checking {path:<35} {status}")
        if not exists:
            all_passed = False
            
    print("--------------------------------------------------")
    if all_passed:
        print("Status: [READY] All core components verified for Final Review.")
    else:
        print("Status: [ACTION REQUIRED] Some components are missing.")
    print("==================================================")

if __name__ == "__main__":
    audit_final_review_readiness()
