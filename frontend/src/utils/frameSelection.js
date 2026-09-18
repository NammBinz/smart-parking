export const FRAME_BUFFER_SIZE = 5
export const MIN_QUALITY_FRAMES = 3
export const OCR_EVIDENCE_WINDOW_SIZE = 3
export const OCR_REQUIRED_MATCHES = 2
export const STRONG_OCR_CONFIDENCE = 0.65

function bounded(value) {
  return Math.max(0, Math.min(1, Number(value) || 0))
}

export function bboxIoU(left, right) {
  if (!left || !right) return 0
  const intersectionWidth = Math.max(0, Math.min(left.x2, right.x2) - Math.max(left.x1, right.x1))
  const intersectionHeight = Math.max(0, Math.min(left.y2, right.y2) - Math.max(left.y1, right.y1))
  const intersection = intersectionWidth * intersectionHeight
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1)
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1)
  const union = leftArea + rightArea - intersection
  return union > 0 ? intersection / union : 0
}

export function isSamePlateTrack(previous, current) {
  const left = previous?.bbox
  const right = current?.bbox
  if (!left || !right) return false
  if (bboxIoU(left, right) >= 0.25) return true

  const leftWidth = Math.max(1, left.x2 - left.x1)
  const leftHeight = Math.max(1, left.y2 - left.y1)
  const rightWidth = Math.max(1, right.x2 - right.x1)
  const rightHeight = Math.max(1, right.y2 - right.y1)
  const widthRatio = rightWidth / leftWidth
  const heightRatio = rightHeight / leftHeight
  if (widthRatio < 0.55 || widthRatio > 1.8 || heightRatio < 0.55 || heightRatio > 1.8) return false

  const leftCenterX = (left.x1 + left.x2) / 2
  const leftCenterY = (left.y1 + left.y2) / 2
  const rightCenterX = (right.x1 + right.x2) / 2
  const rightCenterY = (right.y1 + right.y2) / 2
  const distance = Math.hypot(rightCenterX - leftCenterX, rightCenterY - leftCenterY)
  const averageDiagonal = (Math.hypot(leftWidth, leftHeight) + Math.hypot(rightWidth, rightHeight)) / 2
  return distance / averageDiagonal <= 0.65
}

export function selectPrimaryQualityDetection(response) {
  return [...(response?.detections || [])]
    .sort((left, right) => (
      Number(right.camera_status === 'candidate') - Number(left.camera_status === 'candidate')
      || Number(right.center_bonus || 0) - Number(left.center_bonus || 0)
      || Number(right.quality_score || 0) - Number(left.quality_score || 0)
      || Number(right.bbox_area || 0) - Number(left.bbox_area || 0)
      || Number(right.detection_confidence || 0) - Number(left.detection_confidence || 0)
    ))[0] || null
}

export function updateFrameBuffer(buffer, item, maxSize = FRAME_BUFFER_SIZE) {
  if (!item?.detection || item.detection.camera_status !== 'candidate') {
    return { buffer, reset: false, accepted: false }
  }
  const previous = buffer.at(-1)
  const reset = Boolean(previous && !isSamePlateTrack(previous.detection, item.detection))
  const next = reset ? [item] : [...buffer, item].slice(-maxSize)
  return { buffer: next, reset, accepted: true }
}

export function rankBestFrames(buffer) {
  const candidates = buffer.filter(({ detection }) => detection?.camera_status === 'candidate')
  if (!candidates.length) return []
  const sharpnessValues = candidates.map(({ detection }) => Number(detection.sharpness) || 0)
  const minimum = Math.min(...sharpnessValues)
  const maximum = Math.max(...sharpnessValues)

  return candidates
    .map((entry) => {
      const rawSharpness = Number(entry.detection.sharpness) || 0
      const sharpnessQuality = maximum > minimum
        ? (rawSharpness - minimum) / (maximum - minimum)
        : rawSharpness / (rawSharpness + 120)
      const qualityScore = (
        0.28 * bounded(entry.detection.detection_confidence)
        + 0.25 * bounded(entry.detection.size_quality)
        + 0.25 * bounded(sharpnessQuality)
        + 0.12 * bounded(entry.detection.center_bonus)
        + 0.10 * bounded(entry.detection.brightness_quality)
      )
      return { ...entry, qualityScore, sharpnessQuality }
    })
    .sort((left, right) => right.qualityScore - left.qualityScore || right.id - left.id)
}
