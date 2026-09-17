export default function ImageDetectionOverlay({ previewUrl, recognition, selectedIndex, onSelect }) {
  const width = recognition?.image_width || 1
  const height = recognition?.image_height || 1

  return (
    <div className="detection-image-wrap">
      <img className="upload-preview" src={previewUrl} alt="Selected vehicle" />
      {recognition?.detections.map((detection, index) => {
        const { x1, y1, x2, y2 } = detection.bbox
        const boxStyle = {
          left: `${(x1 / width) * 100}%`,
          top: `${(y1 / height) * 100}%`,
          width: `${((x2 - x1) / width) * 100}%`,
          height: `${((y2 - y1) / height) * 100}%`,
        }
        return (
          <button
            className={`detection-box ${selectedIndex === index ? 'selected' : ''}`}
            key={`${x1}-${y1}-${index}`}
            style={boxStyle}
            onClick={() => onSelect(index)}
            aria-label={`Select detection ${index + 1}`}
          >
            <span>
              {detection.normalized_text || 'Unread'} · {Math.round(detection.detection_confidence * 100)}%
            </span>
          </button>
        )
      })}
    </div>
  )
}
