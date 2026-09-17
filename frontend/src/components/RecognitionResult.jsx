function variantLabel(value) {
  return value
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

export default function RecognitionResult({ detection, index, selected, onUse }) {
  return (
    <article className={`recognition-card ${selected ? 'selected' : ''}`}>
      <div className="recognition-card-heading">
        <div>
          <span className="result-label">Detected plate {index + 1}</span>
          <strong>{detection.normalized_text || detection.raw_text || 'Unread'}</strong>
        </div>
        <span className={`validity-badge ${detection.is_valid ? 'valid' : 'invalid'}`}>
          {detection.is_valid ? 'Valid' : 'Needs correction'}
        </span>
      </div>
      <div className="recognition-metrics">
        <span>YOLO <strong>{Math.round(detection.detection_confidence * 100)}%</strong></span>
        <span>OCR <strong>{Math.round(detection.ocr_confidence * 100)}%</strong></span>
        <span>Variant <strong>{variantLabel(detection.preprocessing_variant)}</strong></span>
      </div>
      <button className={`btn btn-sm ${selected ? 'btn-success' : 'btn-outline-primary'}`} onClick={onUse}>
        {selected ? 'Selected for check-in' : detection.is_valid ? 'Use this plate' : 'Use / Edit'}
      </button>
    </article>
  )
}
