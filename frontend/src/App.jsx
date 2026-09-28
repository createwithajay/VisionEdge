import { useRef, useEffect, useState } from 'react'

function App() {
  const videoRef = useRef(null)
  const peerConnectionRef = useRef(null)

  const [fps, setFps] = useState('--')
  const [gpuMemory, setGpuMemory] = useState('--')
  const [decoderUtilization, setDecoderUtilization] = useState('--')

  useEffect(() => {
    const peerConnection = new RTCPeerConnection()

    peerConnectionRef.current = peerConnection

    console.log('WebRTC Peer Connection created')

    return () => {
      peerConnection.close()
      console.log('WebRTC Peer Connection closed')
    }
  }, [])

  return (
    <div>
      <header>
        <h1>Vision Edge</h1>
        <p>AI-Powered Vision System</p>
      </header>

      <main>
        <section>
          <h2>Video Stream</h2>

          <p>Status: WebRTC connection ready</p>

          <div>
            <video
              ref={videoRef}
              controls
              width="640"
              height="360"
            >
              Your browser does not support video playback.
            </video>
          </div>
        </section>

        <section>
          <h2>Telemetry</h2>

          <div>
            <h3>FPS</h3>
            <p>{fps}</p>
          </div>

          <div>
            <h3>GPU Memory</h3>
            <p>{gpuMemory}</p>
          </div>

          <div>
            <h3>Decoder Utilization</h3>
            <p>{decoderUtilization}</p>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App