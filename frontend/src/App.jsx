
import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const videoRef = useRef(null)
  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [hasVideo, setHasVideo] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [fps] = useState('--')
  const [gpuMemory] = useState('--')
  const [decoderUtilization] = useState('--')

  useEffect(() => {
    let active = true
    let peerConnection

    async function connectToBackend() {
      try {
        setConnectionStatus('Connecting...')
        setErrorMessage('')

        peerConnection = new RTCPeerConnection()

        peerConnection.onconnectionstatechange = () => {
          if (!active) return

          const state = peerConnection.connectionState

          if (state === 'connected') {
            setConnectionStatus('Connected')
          } else if (state === 'failed') {
            setConnectionStatus('Connection failed')
          } else if (state === 'disconnected') {
            setConnectionStatus('Disconnected')
          } else if (state === 'closed') {
            setConnectionStatus('Connection closed')
          }
        }

        peerConnection.ontrack = (event) => {
          if (!active || !videoRef.current) return

          const stream =
            event.streams[0] || new MediaStream([event.track])

          videoRef.current.srcObject = stream
          setHasVideo(true)
          setConnectionStatus('Connected')
        }

        peerConnection.addTransceiver('video', {
          direction: 'recvonly',
        })

        const offer = await peerConnection.createOffer()
        await peerConnection.setLocalDescription(offer)

        // Wait until the browser has gathered its ICE candidates.
        if (peerConnection.iceGatheringState !== 'complete') {
          await new Promise((resolve, reject) => {
            const timeout = setTimeout(
              () => reject(new Error('ICE gathering timed out')),
              10000,
            )

            const checkState = () => {
              if (peerConnection.iceGatheringState === 'complete') {
                clearTimeout(timeout)
                peerConnection.removeEventListener(
                  'icegatheringstatechange',
                  checkState,
                )
                resolve()
              }
            }

            peerConnection.addEventListener(
              'icegatheringstatechange',
              checkState,
            )
            checkState()
          })
        }

        const response = await fetch('/offer', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(peerConnection.localDescription),
        })

        if (!response.ok) {
          throw new Error(`Backend returned HTTP ${response.status}`)
        }

        const answer = await response.json()
        await peerConnection.setRemoteDescription(answer)
      } catch (error) {
        console.error('WebRTC connection error:', error)

        if (active) {
          setConnectionStatus('Connection failed')
          setHasVideo(false)
          setErrorMessage(
            `${error.message}. Check that the Python backend is running on port 8081.`,
          )
        }
      }
    }

    connectToBackend()

    return () => {
      active = false

      if (peerConnection) {
        peerConnection.close()
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
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

          <p>
            <strong>Status: {connectionStatus}</strong>
          </p>

          <div
            style={{
              width: '640px',
              maxWidth: '100%',
              backgroundColor: '#000',
            }}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              width="640"
              height="360"
              style={{
                display: 'block',
                width: '100%',
                height: 'auto',
                backgroundColor: '#000',
              }}
            />

            {!hasVideo && (
              <p
                role="status"
                style={{ color: 'white', padding: '12px' }}
              >
                {errorMessage ||
                  'Waiting for video from the Python backend...'}
              </p>
            )}
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

          <p>Telemetry will be available when the backend provides it.</p>
        </section>
      </main>
    </div>
  )
}

export default App
