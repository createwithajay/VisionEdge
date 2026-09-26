import asyncio
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

class ProcessedVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.frame_count = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # Generate a blank 1080p frame and draw a frame counter on it
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(img, f"VisionEdge Stream - Frame {self.frame_count}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        self.frame_count += 1

        # Convert the OpenCV numpy array into a WebRTC VideoFrame
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

# --- DAY 6 UPDATE: WebRTC Offer Handler ---
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    
    # Attach our video track to the WebRTC peer connection
    pc.addTrack(ProcessedVideoTrack())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

# Setup routing for the application
app = web.Application()
app.router.add_post("/offer", offer)

if __name__ == "__main__":
    print("Starting VisionEdge WebRTC server on http://0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)
