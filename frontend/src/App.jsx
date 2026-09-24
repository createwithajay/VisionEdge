function App() {
  return (
    <div>
      <header>
        <h1>Vision Edge</h1>
        <p>AI-Powered Vision System</p>
      </header>

      <main>
        <section>
          <h2>Video Stream</h2>

          <p>Status: Waiting for video stream...</p>

          <div>
            <video
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
            <p>--</p>
          </div>

          <div>
            <h3>GPU Memory</h3>
            <p>--</p>
          </div>

          <div>
            <h3>Decoder Utilization</h3>
            <p>--</p>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App