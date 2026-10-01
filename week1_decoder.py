import av
import numpy as np

def extract_raw_frames(video_path):
    print(f"Connecting to video stream: {video_path}")
    try:
        # Request hardware acceleration (NVDEC) to bypass CPU
        container = av.open(video_path, options={'hwaccel': 'nvdec'})
        print("Hardware-accelerated decoding initialized via PyAV.")
    except Exception:
        print("Hardware acceleration unavailable, falling back to standard PyAV decoding.")
        container = av.open(video_path)

    frame_count = 0
    video_stream = container.streams.video[0]
    
    for frame in container.decode(video_stream):
        # Extract the raw frames into numpy arrays for the Al engine later
        raw_frame_array = frame.to_ndarray(format='bgr24')
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Successfully extracted {frame_count} raw frames. Current array shape: {raw_frame_array.shape}")
            
        if frame_count >= 90:
            print("\nWeek 1 Objective Complete: Raw frames successfully extracted.")
            break

if __name__ == "__main__":
    extract_raw_frames('sample_video.mp4')
