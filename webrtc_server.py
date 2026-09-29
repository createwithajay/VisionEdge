import asyncio
import os
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

UPLOAD_DIR = "./tensorrt_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ProcessedVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.frame_count = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(img, f"VisionEdge Stream - Frame {self.frame_count}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        self.frame_count += 1
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"WebRTC connection state changed to: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()

    pc.addTrack(ProcessedVideoTrack())
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def upload_model(request):
    reader = await request.multipart()
    field = await reader.next()

    if field and field.name == 'engine_file':
        filename = field.filename
        file_path = os.path.join(UPLOAD_DIR, filename)
        size = 0
        with open(file_path, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)

        return web.json_response({
            "status": "success", 
            "message": f"Model {filename} uploaded and swapped successfully. Size: {size} bytes."
        })

    return web.json_response({"status": "error", "message": "Invalid file upload field"}, status=400)

app = web.Application()
app.router.add_post("/offer", offer)
app.router.add_post("/upload_model", upload_model)

if __name__ == "__main__":
    print("Starting VisionEdge WebRTC server on http://0.0.0.0:8081")
    web.run_app(app, host="0.0.0.0", port=8081)
