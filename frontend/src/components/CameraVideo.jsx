export default function CameraVideo({ videoRef, stream, recognition }) {
  const width = recognition?.image_width || 1
  const height = recognition?.image_height || 1

  return (
    <div className={`camera-video-wrap ${stream ? 'active' : ''}`}>
      <video ref={videoRef} autoPlay muted playsInline aria-label="Live parking camera" />
      {!stream && (
        <div className="camera-video-empty">
          <span aria-hidden="true">◎</span>
          <p>Start the camera to begin plate recognition.</p>
        </div>
      )}
      {stream && (
        <div className="capture-zone" aria-hidden="true">
          <span>Position license plate inside this area</span>
        </div>
      )}
      {stream && recognition?.detections.map((detection, index) => {
        const { x1, y1, x2, y2 } = detection.bbox
        const style = {
          left: `${(x1 / width) * 100}%`,
          top: `${(y1 / height) * 100}%`,
          width: `${((x2 - x1) / width) * 100}%`,
          height: `${((y2 - y1) / height) * 100}%`,
        }
        return (
          <div className="camera-detection-box" style={style} key={`${x1}-${y1}-${index}`}>
            <span>
              {detection.normalized_text || 'Unread'} · {Math.round(detection.detection_confidence * 100)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}
