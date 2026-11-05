import random

def monitor_gpu_telemetry():
    print("[VisionEdge] Monitoring Edge GPU & VRAM Utilization...")
    
    # Simulate GPU metrics for edge hardware inference
    vram_used = round(random.uniform(2.4, 4.1), 2)  # GB
    vram_total = 8.0                                # GB
    gpu_util = round(random.uniform(65.0, 92.5), 1) # percentage
    
    print(f"GPU Core Utilization: {gpu_util}%")
    print(f"VRAM Memory Usage: {vram_used} GB / {vram_total} GB")
    
    if vram_used < 6.0 and gpu_util < 95.0:
        print("Status: [PASSED] GPU memory footprint and core load are within optimal range.")
    else:
        print("Status: [WARNING] High resource utilization detected.")

if __name__ == "__main__":
    monitor_gpu_telemetry()
