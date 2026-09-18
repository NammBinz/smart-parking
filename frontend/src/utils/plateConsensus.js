export const CONSENSUS_WINDOW_SIZE = 5
export const CONSENSUS_REQUIRED_MATCHES = 3

export const DEFAULT_CAPTURE_ZONE = Object.freeze({
  x1: 0.15,
  y1: 0.25,
  x2: 0.85,
  y2: 0.80,
})

export function detectionObservation(detection) {
  if (
    !detection
    || !detection.is_valid
    || detection.ocr_status !== 'ok'
    || !detection.normalized_text
  ) return null

  return {
    plate: detection.normalized_text,
    ocrConfidence: Number(detection.ocr_confidence) || 0,
    detectionConfidence: Number(detection.detection_confidence) || 0,
  }
}

export function advanceObservationWindow(window, observation, size = CONSENSUS_WINDOW_SIZE) {
  return [...window, observation].slice(-size)
}

export function summarizeConsensus(window, required = CONSENSUS_REQUIRED_MATCHES) {
  const aggregates = new Map()
  window.forEach((observation, index) => {
    if (!observation?.plate) return
    const current = aggregates.get(observation.plate) || {
      plate: observation.plate,
      count: 0,
      ocrTotal: 0,
      detectionTotal: 0,
      latestIndex: -1,
    }
    current.count += 1
    current.ocrTotal += observation.ocrConfidence || 0
    current.detectionTotal += observation.detectionConfidence || 0
    current.latestIndex = index
    aggregates.set(observation.plate, current)
  })

  const ranked = [...aggregates.values()]
    .map((candidate) => ({
      plate: candidate.plate,
      count: candidate.count,
      averageOcrConfidence: candidate.ocrTotal / candidate.count,
      averageDetectionConfidence: candidate.detectionTotal / candidate.count,
      latestIndex: candidate.latestIndex,
    }))
    .sort((left, right) => (
      right.count - left.count
      || right.averageOcrConfidence - left.averageOcrConfidence
      || right.averageDetectionConfidence - left.averageDetectionConfidence
      || right.latestIndex - left.latestIndex
    ))

  const leader = ranked[0] || null
  return {
    leader,
    stable: Boolean(leader && leader.count >= required),
    required,
    sampledFrames: window.length,
  }
}

function isInsideCaptureZone(detection, imageWidth, imageHeight, zone) {
  if (!imageWidth || !imageHeight) return false
  const centerX = (detection.bbox.x1 + detection.bbox.x2) / 2 / imageWidth
  const centerY = (detection.bbox.y1 + detection.bbox.y2) / 2 / imageHeight
  return centerX >= zone.x1 && centerX <= zone.x2 && centerY >= zone.y1 && centerY <= zone.y2
}

export function selectPrimaryDetection(response, zone = DEFAULT_CAPTURE_ZONE) {
  const detections = response?.detections || []
  const valid = detections
    .map((detection, index) => ({
      detection,
      index,
      insideZone: isInsideCaptureZone(
        detection,
        response.image_width,
        response.image_height,
        zone,
      ),
      area: Math.max(0, detection.bbox.x2 - detection.bbox.x1)
        * Math.max(0, detection.bbox.y2 - detection.bbox.y1),
    }))
    .filter(({ detection }) => detectionObservation(detection) !== null)

  valid.sort((left, right) => (
    Number(right.insideZone) - Number(left.insideZone)
    || left.index - right.index
    || right.area - left.area
    || right.detection.detection_confidence - left.detection.detection_confidence
  ))
  return valid[0]?.detection || null
}
