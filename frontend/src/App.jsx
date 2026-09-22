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

          <p>FPS: --</p>
          <p>GPU Memory: --</p>
          <p>Decoder Utilization: --</p>
        </section>
      </main>
    </div>
  )
}

export default App