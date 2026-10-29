import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const videoRef = useRef(null)
  const peerConnectionRef = useRef(null)

  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [hasVideo, setHasVideo] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [fps, setFps] = useState('--')
  const [gpuMemory, setGpuMemory] = useState('Unavailable')
  const [decoderUtilization, setDecoderUtilization] =
    useState('Unavailable')

  useEffect(() => {
    let active = true
    let peerConnection
    let statsInterval
    let previousFrames = null
    let previousTime = null

    async function connectToBackend() {
      try {
        setConnectionStatus('Connecting...')
        setErrorMessage('')

        peerConnection = new RTCPeerConnection()
        peerConnectionRef.current = peerConnection

        peerConnection.onconnectionstatechange = () => {
          if (!active) return

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
          if (!active || !videoRef.current) return

          const stream =
            event.streams[0] || new MediaStream([event.track])

          videoRef.current.srcObject = stream
          setHasVideo(true)
          setConnectionStatus('Connected')

          event.track.onended = () => {
            if (active) setHasVideo(false)
          }
        }

        // Receive video from the Python WebRTC backend.
        peerConnection.addTransceiver('video', {
          direction: 'recvonly',
        })

        const offer = await peerConnection.createOffer()
        await peerConnection.setLocalDescription(offer)

        // Wait for ICE gathering to finish before sending the offer.
        if (peerConnection.iceGatheringState !== 'complete') {
          await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
              peerConnection.removeEventListener(
                'icegatheringstatechange',
                checkState,
              )
              reject(new Error('ICE gathering timed out'))
            }, 10000)

            function checkState() {
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

        // Read video statistics from the browser once per second.
        statsInterval = setInterval(async () => {
          if (
            !active ||
            !peerConnection ||
            peerConnection.connectionState === 'closed'
          ) {
            return
          }

          try {
            const stats = await peerConnection.getStats()

            stats.forEach((report) => {
              if (
                report.type !== 'inbound-rtp' ||
                !(
                  report.kind === 'video' ||
                  report.mediaType === 'video'
                )
              ) {
                return
              }

              const now = performance.now()

              if (typeof report.framesPerSecond === 'number') {
                setFps(report.framesPerSecond.toFixed(1))
              } else if (
                typeof report.framesReceived === 'number'
              ) {
                if (
                  previousFrames !== null &&
                  previousTime !== null
                ) {
                  const elapsed = (now - previousTime) / 1000
                  const received =
                    report.framesReceived - previousFrames

                  if (elapsed > 0 && received >= 0) {
                    setFps((received / elapsed).toFixed(1))
                  }
                }

                previousFrames = report.framesReceived
                previousTime = now
              }
            })
          } catch (error) {
            console.error('Could not read video statistics:', error)
          }
        }, 1000)
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

      if (statsInterval) {
        clearInterval(statsInterval)
      }

      if (peerConnection) {
        peerConnection.close()
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null
      }

      peerConnectionRef.current = null
    }
  }, [])

  return (
    <div className="app">
      <header>
        <h1>Vision Edge</h1>
        <p>AI-Powered Vision System</p>
      </header>

      <main>
        <section>
          <h2>Video Stream</h2>

          <p>
            Status:{' '}
            <strong>{connectionStatus}</strong>
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
          <h2>Telemetry Dashboard</h2>
          <p>Live video performance statistics</p>

          <div className="telemetry-grid">
            <article className="telemetry-card">
              <h3>FPS</h3>
              <p>{fps}</p>
              <small>Received video frames per second</small>
            </article>

            <article className="telemetry-card">
              <h3>GPU Memory</h3>
              <p>{gpuMemory}</p>
              <small>Requires backend GPU metrics</small>
            </article>

            <article className="telemetry-card">
              <h3>Decoder Utilization</h3>
              <p>{decoderUtilization}</p>
              <small>Requires backend decoder metrics</small>
            </article>
          </div>

          <p>
            FPS is measured using browser WebRTC statistics when
            available. GPU memory and decoder utilization will
            appear when the backend provides those values.
          </p>
        </section>
      </main>
    </div>
  )
}

export default App