# VisionEdge: Hardware-Accelerated Video Pipeline

VisionEdge is an edge-AI computer vision processing pipeline designed to eliminate CPU decoding bottlenecks and PCIe bus overhead during concurrent, high-resolution video streaming.

## Core Architecture & Benchmarks
- **Week 1:** Hardware-accelerated PyAV video decoding bypassing CPU.
- **Week 2:** 3.53x TensorRT speedup over baseline with zero steady-state memory leaks.
- **Week 3:** Zero-Copy CuPy bridge drawing bounding boxes directly in GPU memory.
- **Week 4:** MultiStream orchestration (asyncio) across 4 cameras with dynamic model hot-swapping.

## SOP Work Distribution Document (Revised Allocation)
*Documented in compliance with Axlero Solutions SOP Section 5 & 6 (Handling Unresponsive Team Members). Active members maintained full technical coverage and continuous GitHub delivery.*

| Role / Module | Original Assignee | Active Implementer | Status |
| :--- | :--- | :--- | :--- |
| Member 1: MultiStream Orchestration | Member 1 | Active Implementer | Completed |
| Member 2: Hardware Edge Engineer | Active Implementer | Active Implementer | Completed |
| Member 3: AI Inference & Zero-Copy | Member 3 (Inactive) | Active Implementer (Reallocated) | Completed |
| Member 4: Streaming Backend & Git Lead | Active Implementer | Active Implementer | Completed |
| Member 5: Frontend / Telemetry | Member 5 | Active Implementer (Reallocated) | Completed |

## Setup & Execution
Run the master pipeline: python3 main.py
