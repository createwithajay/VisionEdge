# VisionEdge Project - Work Distribution Document

## Project Overview
**Project Selected:** VisionEdge (Hardware-Accelerated Computer Vision & Edge Computing)

## Team Member Allocations

### Member 1: Orchestration Engineer & Team Lead
*   **Technical Role:** Implement asyncio to manage multiple independent video streams concurrently on the GPU. Orchestrate the pipeline for 10 simultaneous 4K streams.
*   **Management Role:** Manage GitHub repository, enforce daily commit rules, submit the Work Distribution Document, and handle all merges to the main branch.

### Member 2: Hardware Edge Engineer
*   **Technical Role:** Bypass CPU using PyAV and NVIDIA Video Codec SDK to decode H.264/H.265 streams directly to GPU memory.
*   **QA Role:** Profile system to ensure zero VRAM memory leaks during continuous streaming.

### Member 3: AI Inference & CUDA Specialist
*   **Technical Role:** Export YOLO models to ONNX and compile highly optimized TensorRT engines.
*   **Technical Role:** Build the CuPy "Zero-Copy Bridge" and draw bounding boxes using CUDA kernels.

### Member 4: Streaming & Backend (WebRTC) Developer
*   **Technical Role:** Build the WebRTC streaming foundation using the aiortc Python library to stream processed frames with sub-100ms latency.
*   **Technical Role:** Build backend functionality for dynamically uploading TensorRT engine files to swap AI models on the fly.

### Member 5: Frontend (React) Developer
*   **Technical Role:** Set up React frontend to receive the WebRTC stream.
*   **Technical Role:** Build Telemetry Dashboard to display live pipeline metrics (FPS, GPU Memory, Decoder Utilization).
