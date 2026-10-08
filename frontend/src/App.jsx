import { useRef, useEffect, useState } from 'react'

function App() {
  const videoRef = useRef(null)
  const peerConnectionRef = useRef(null)

  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [fps, setFps] = useState('--')
  const [gpuMemory, setGpuMemory] = useState('--')
  const [decoderUtilization, setDecoderUtilization] = useState('--')

  useEffect(() => {
    const peerConnection = new RTCPeerConnection()

    peerConnectionRef.current = peerConnection

    peerConnection.onconnectionstatechange = () => {
      const state = peerConnection.connectionState

      if (state === 'connected') {
        setConnectionStatus('Connected')
      } else if (state === 'connecting' || state === 'new') {
        setConnectionStatus('Connecting...')
      } else if (state === 'disconnected') {
        setConnectionStatus('Disconnected')
      } else if (state === 'failed') {
        setConnectionStatus('Connection failed')
      } else if (state === 'closed') {
        setConnectionStatus('Connection closed')
      }
    }

    peerConnection.ontrack = (event) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0]
      }
    }

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

          <p>Status: {connectionStatus}</p>

          <div>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              width="640"
              height="360"
              style={{ backgroundColor: 'black' }}
              onError={() => setConnectionStatus('Video error')}
              onLoadStart={() => setConnectionStatus('Video loading...')}
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