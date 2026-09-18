import assert from 'node:assert/strict'
import test from 'node:test'

import {
  advanceObservationWindow,
  selectPrimaryDetection,
  summarizeConsensus,
} from './plateConsensus.js'

const observation = (plate, ocrConfidence = 0.9, detectionConfidence = 0.9) => (
  plate ? { plate, ocrConfidence, detectionConfidence } : null
)

function consensusFor(plates) {
  const window = plates.reduce(
    (current, plate) => advanceObservationWindow(current, observation(plate)),
    [],
  )
  return summarizeConsensus(window)
}

test('three exact observations become stable', () => {
  const result = consensusFor(['29Z158344', '29Z158344', '29Z158344'])
  assert.equal(result.stable, true)
  assert.equal(result.leader.plate, '29Z158344')
})

test('three matching observations beat one competing plate', () => {
  const result = consensusFor(['29Z158344', '29Z158344', '51A4032', '29Z158344'])
  assert.equal(result.stable, true)
  assert.equal(result.leader.plate, '29Z158344')
})

test('null samples evict stale recognition', () => {
  const result = consensusFor(['29Z158344', null, null, null, null])
  assert.equal(result.stable, false)
  assert.equal(result.leader.count, 1)
})

test('A A B B A makes A stable', () => {
  const result = consensusFor(['A', 'A', 'B', 'B', 'A'])
  assert.equal(result.stable, true)
  assert.equal(result.leader.plate, 'A')
})

test('A A B B null has no stable candidate', () => {
  const result = consensusFor(['A', 'A', 'B', 'B', null])
  assert.equal(result.stable, false)
})

test('primary selection ignores invalid OCR and prefers the capture zone', () => {
  const detection = (plate, bbox, valid = true) => ({
    normalized_text: plate,
    is_valid: valid,
    ocr_status: valid ? 'ok' : 'unreadable',
    ocr_confidence: 0.9,
    detection_confidence: 0.9,
    bbox,
  })
  const response = {
    image_width: 1000,
    image_height: 500,
    detections: [
      detection('INVALID', { x1: 300, y1: 180, x2: 600, y2: 320 }, false),
      detection('OUTSIDE', { x1: 0, y1: 0, x2: 120, y2: 80 }),
      detection('INSIDE', { x1: 400, y1: 180, x2: 600, y2: 300 }),
    ],
  }
  assert.equal(selectPrimaryDetection(response).normalized_text, 'INSIDE')
})
