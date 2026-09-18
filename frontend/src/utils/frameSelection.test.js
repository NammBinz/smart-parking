import assert from 'node:assert/strict'
import test from 'node:test'

import {
  bboxIoU,
  isSamePlateTrack,
  rankBestFrames,
  selectPrimaryQualityDetection,
  updateFrameBuffer,
} from './frameSelection.js'

function detection(overrides = {}) {
  return {
    bbox: { x1: 100, y1: 100, x2: 300, y2: 180 },
    camera_status: 'candidate',
    detection_confidence: 0.9,
    size_quality: 0.8,
    sharpness: 100,
    center_bonus: 1,
    brightness_quality: 0.9,
    quality_score: 0.8,
    bbox_area: 16000,
    ...overrides,
  }
}

test('overlap and nearby similarly sized boxes stay on the same track', () => {
  const first = detection()
  const overlapping = detection({ bbox: { x1: 120, y1: 105, x2: 320, y2: 185 } })
  const nearby = detection({ bbox: { x1: 205, y1: 110, x2: 405, y2: 190 } })
  assert.ok(bboxIoU(first.bbox, overlapping.bbox) > 0.25)
  assert.equal(isSamePlateTrack(first, overlapping), true)
  assert.equal(isSamePlateTrack(first, nearby), true)
})

test('a distant vehicle resets the short frame buffer', () => {
  const original = [{ id: 1, detection: detection() }]
  const next = {
    id: 2,
    detection: detection({ bbox: { x1: 700, y1: 400, x2: 900, y2: 480 } }),
  }
  const result = updateFrameBuffer(original, next)
  assert.equal(result.reset, true)
  assert.deepEqual(result.buffer.map(({ id }) => id), [2])
})

test('too-small frames are not accepted into the OCR candidate buffer', () => {
  const result = updateFrameBuffer([], {
    id: 1,
    detection: detection({ camera_status: 'too_small' }),
  })
  assert.equal(result.accepted, false)
  assert.deepEqual(result.buffer, [])
})

test('relative sharpness makes a clear frame rank ahead of a blurred frame', () => {
  const ranked = rankBestFrames([
    { id: 1, detection: detection({ sharpness: 8 }) },
    { id: 2, detection: detection({ sharpness: 340 }) },
    { id: 3, detection: detection({ sharpness: 40 }) },
  ])
  assert.equal(ranked[0].id, 2)
  assert.ok(ranked[0].qualityScore > ranked[2].qualityScore)
})

test('primary quality selection prefers an OCR-ready centered candidate', () => {
  const primary = selectPrimaryQualityDetection({
    detections: [
      detection({ camera_status: 'move_closer', quality_score: 0.99 }),
      detection({ camera_status: 'candidate', center_bonus: 0, quality_score: 0.7 }),
      detection({ camera_status: 'candidate', center_bonus: 1, quality_score: 0.6 }),
    ],
  })
  assert.equal(primary.camera_status, 'candidate')
  assert.equal(primary.center_bonus, 1)
})
