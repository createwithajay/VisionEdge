\# VisionEdge - Member 1: Orchestration Engineer



\## Week 1 - Async Stream Orchestration



This module implements the Week 1 orchestration foundation for VisionEdge.



The responsibility of Member 1 is to manage multiple independent video stream processing tasks concurrently using Python asyncio.



\## Objective



Build an asynchronous stream orchestration layer that can:



\- Start multiple independent stream tasks

\- Process streams concurrently

\- Maintain individual stream states

\- Handle stream failures without stopping other streams

\- Provide basic execution and concurrency measurements



\## Architecture



Multiple Video Streams

&#x20;       |

&#x20;       v

&#x20; StreamManager

&#x20;       |

&#x20;       +-- Stream-01 task

&#x20;       +-- Stream-02 task

&#x20;       +-- Stream-03 task

&#x20;       +-- Stream-04 task

&#x20;       |

&#x20;       v

&#x20;Asyncio Event Loop

&#x20;       |

&#x20;       v

&#x20;Concurrent Processing



\## Stream Lifecycle



CREATED

&#x20;  |

&#x20;  v

RUNNING

&#x20;  |

&#x20;  +----> COMPLETED

&#x20;  |

&#x20;  +----> ERROR

&#x20;  |

&#x20;  +----> CANCELLED



\## Project Structure



VisionEdge-member1/

|

+-- main.py

+-- benchmark.py

+-- failure\_test.py

+-- orchestrator/

|   +-- \_\_init\_\_.py

|   +-- stream\_manager.py

|

+-- .gitignore

+-- README.md



\## Main Components



\### StreamManager



orchestrator/stream\_manager.py contains the StreamManager class.



It is responsible for:



\- Creating asynchronous stream tasks

\- Running multiple streams concurrently

\- Tracking stream status

\- Counting processed frames

\- Handling cancellation and errors

\- Printing a stream summary



\### main.py



Runs four simulated video streams concurrently.



\### benchmark.py



Compares sequential processing with asyncio-based concurrent processing using a simulated frame workload.



A representative local run showed approximately:



\- Sequential: 4 seconds

\- Asyncio: 1.2 seconds

\- Concurrency ratio: about 3.4x



This is a simulated concurrency benchmark and is not a GPU performance measurement.



\### failure\_test.py



Tests failure isolation.



One stream is intentionally stopped with a simulated error while the other streams continue processing.



\## Testing



Run the main orchestration test:



python main.py



Run the concurrency benchmark:



python benchmark.py



Run the failure isolation test:



python failure\_test.py



\## Technologies



\- Python

\- asyncio

\- Dataclasses

\- time.perf\_counter



\## Week 1 Scope



This implementation focuses only on the orchestration foundation.



It does not claim implementation of:



\- PyAV/NVDEC hardware decoding

\- TensorRT inference

\- CUDA/CuPy processing

\- WebRTC streaming

\- React dashboard

\- Real 4K 60 FPS GPU benchmarking



Those components belong to other team members and later integration stages.



\## Future Integration



The orchestration layer is designed so that future stream-processing components can be connected to each asynchronous stream task.



Target architecture:



Video Input

&#x20;   |

&#x20;   v

Stream Task

&#x20;   |

&#x20;   +--> Decode

&#x20;   |

&#x20;   +--> AI Inference

&#x20;   |

&#x20;   +--> Processing

&#x20;   |

&#x20;   +--> Streaming

&#x20;   |

&#x20;   v

Monitoring



The Week 1 implementation provides the asynchronous foundation required for this architecture.
