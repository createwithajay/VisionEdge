import asyncio
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

# Setup standard routing for the application
app = web.Application()

if __name__ == "__main__":
    # Run the asynchronous server on port 8080
    print("Starting VisionEdge WebRTC server on http://0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)
