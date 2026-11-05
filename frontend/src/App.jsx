
import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const videoRef = useRef(null)

  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [hasVideo, setHasVideo] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [fps, setFps] = useState('--')
  const [gpuMemory, setGpuMemory] = useState('Unavailable')
  const [decoderUtilization, setDecoderUtilization] = useState('Unavailable')

  useEffect(() => {
    let active = true
    let pc
    let statsTimer
    let telemetryTimer
    let previousFrames = null
    let previousTime = null

    async function connectToBackend() {
      try {
        setConnectionStatus('Connecting...')
        setErrorMessage('')

        pc = new RTCPeerConnection()
        pc.addTransceiver('video', { direction: 'recvonly' })

        pc.onconnectionstatechange = () => {
          if (!active) return

          if (pc.connectionState === 'connected') {
            setConnectionStatus('Connected')
          } else if (
            pc.connectionState === 'connecting' ||
            pc.connectionState === 'new'
          ) {
            setConnectionStatus('Connecting...')
          } else if (pc.connectionState === 'disconnected') {
            setConnectionStatus('Disconnected')
          } else if (pc.connectionState === 'failed') {
            setConnectionStatus('Connection failed')
          } else if (pc.connectionState === 'closed') {
            setConnectionStatus('Connection closed')
          }
        }

        pc.ontrack = async (event) => {
          if (!active || !videoRef.current) return

          const video = videoRef.current
          video.srcObject =
            event.streams[0] || new MediaStream([event.track])

          try {
            await video.play()
          } catch (error) {
            console.error('Video playback error:', error)
          }
        }

        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        if (pc.iceGatheringState !== 'complete') {
          await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
              pc.removeEventListener('icegatheringstatechange', checkState)
              reject(new Error('ICE gathering timed out'))
            }, 10000)

            function checkState() {
              if (pc.iceGatheringState === 'complete') {
                clearTimeout(timeout)
                pc.removeEventListener('icegatheringstatechange', checkState)
                resolve()
              }
            }

            pc.addEventListener('icegatheringstatechange', checkState)
            checkState()
          })
        }

        const response = await fetch('/offer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(pc.localDescription),
        })

        if (!response.ok) {
          throw new Error(`Backend returned HTTP ${response.status}`)
        }

        const answer = await response.json()

        if (!answer.sdp || !answer.type) {
          throw new Error(answer.error || 'Invalid WebRTC answer')
        }

        await pc.setRemoteDescription(answer)

        statsTimer = setInterval(async () => {
          if (!active || !pc || pc.connectionState === 'closed') return

          try {
            const stats = await pc.getStats()

            stats.forEach((report) => {
              if (
                report.type !== 'inbound-rtp' ||
                !(report.kind === 'video' || report.mediaType === 'video')
              ) {
                return
              }

              const now = performance.now()

              if (typeof report.framesPerSecond === 'number') {
                setFps(report.framesPerSecond.toFixed(1))
              } else if (typeof report.framesReceived === 'number') {
                if (previousFrames !== null && previousTime !== null) {
                  const elapsed = (now - previousTime) / 1000
                  const received = report.framesReceived - previousFrames

                  if (elapsed > 0 && received >= 0) {
                    setFps((received / elapsed).toFixed(1))
                  }
                }

                previousFrames = report.framesReceived
                previousTime = now
              }
            })
          } catch (error) {
            console.error('Video statistics error:', error)
          }
        }, 1000)
      } catch (error) {
        console.error('WebRTC connection error:', error)

        if (active) {
          setConnectionStatus('Connection failed')
          setErrorMessage(error.message)
        }
      }
    }

    async function fetchTelemetry() {
      try {
        const response = await fetch('/telemetry')

        if (!response.ok) {
          throw new Error(`Telemetry returned HTTP ${response.status}`)
        }

        const data = await response.json()

        if (!active) return

        if (
          typeof data.gpu_memory_percent === 'number' &&
          typeof data.gpu_memory_used_mb === 'number' &&
          typeof data.gpu_memory_total_mb === 'number'
        ) {
          setGpuMemory(
            `${data.gpu_memory_percent}% (${data.gpu_memory_used_mb} / ${data.gpu_memory_total_mb} MB)`
          )
        } else {
          setGpuMemory('Unavailable')
        }

        setDecoderUtilization(
          typeof data.decoder_utilization === 'number'
            ? `${data.decoder_utilization}%`
            : 'Unavailable'
        )
      } catch (error) {
        if (active) setGpuMemory('Unavailable')
        console.error('Telemetry fetch error:', error)
      }
    }

    function updateVideoStatus() {
      if (!active || !videoRef.current) return

      const video = videoRef.current

      if (video.videoWidth > 0 && video.videoHeight > 0 && !video.paused) {
        setHasVideo(true)
        setConnectionStatus('Connected')
        setErrorMessage('')
      } else {
        setHasVideo(false)
      }
    }

    connectToBackend()
    fetchTelemetry()
    telemetryTimer = setInterval(fetchTelemetry, 2000)

    const video = videoRef.current
    video?.addEventListener('playing', updateVideoStatus)
    video?.addEventListener('pause', updateVideoStatus)
    video?.addEventListener('error', updateVideoStatus)

    return () => {
      active = false
      clearInterval(statsTimer)
      clearInterval(telemetryTimer)

      video?.removeEventListener('playing', updateVideoStatus)
      video?.removeEventListener('pause', updateVideoStatus)
      video?.removeEventListener('error', updateVideoStatus)

      if (pc) pc.close()
      if (video) video.srcObject = null
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
            Status: <strong>{hasVideo ? 'Video playing' : connectionStatus}</strong>
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
              <p role="status" style={{ color: 'white', padding: '12px' }}>
                {errorMessage || 'Waiting for video frames...'}
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
              <small>Live data from the backend</small>
            </article>

            <article className="telemetry-card">
              <h3>Decoder Utilization</h3>
              <p>{decoderUtilization}</p>
              <small>Unavailable if the driver does not report it</small>
            </article>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
