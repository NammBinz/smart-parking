import { useCallback, useEffect, useRef, useState } from 'react'

import useCameraStream from '../hooks/useCameraStream'
import useRecognitionLoop from '../hooks/useRecognitionLoop'
import { errorMessage, recognizeFrame } from '../services/api'
import { normalizePlate } from '../utils/format'
import {
  advanceObservationWindow,
  detectionObservation,
  selectPrimaryDetection,
  summarizeConsensus,
} from '../utils/plateConsensus'
import {
  MIN_QUALITY_FRAMES,
  OCR_EVIDENCE_WINDOW_SIZE,
  OCR_REQUIRED_MATCHES,
  STRONG_OCR_CONFIDENCE,
  rankBestFrames,
  selectPrimaryQualityDetection,
  updateFrameBuffer,
} from '../utils/frameSelection'
import CameraControls from './CameraControls'
import CameraVideo from './CameraVideo'

export const CAMERA_STATES = Object.freeze({
  IDLE: 'IDLE',
  CAMERA_ACTIVE: 'CAMERA_ACTIVE',
  SCANNING: 'SCANNING',
  STABLE: 'STABLE',
  PROCESSING_ENTRY: 'PROCESSING_ENTRY',
  PROCESSING_EXIT: 'PROCESSING_EXIT',
  ERROR: 'ERROR',
})

const DEFAULT_RECOGNITION_INTERVAL_MS = 1500
const DEFAULT_RESUME_DELAY_MS = 2000
const CAMERA_DEBUG = import.meta.env.VITE_CAMERA_DEBUG === 'true'

export default function CameraRecognition({
  onStablePlate,
  onManualPlate,
  onResetPlate,
  workflowResetKey,
  recognitionIntervalMs = DEFAULT_RECOGNITION_INTERVAL_MS,
  resumeDelayMs = DEFAULT_RESUME_DELAY_MS,
}) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const observationsRef = useRef([])
  const frameBufferRef = useRef([])
  const frameIdRef = useRef(0)
  const missedFramesRef = useRef(0)
  const ocrCallsRef = useRef(0)
  const scanStartedRef = useRef(performance.now())
  const stableRef = useRef(null)
  const previousResetKey = useRef(workflowResetKey)
  const camera = useCameraStream()
  const [mode, setMode] = useState('ENTRY')
  const [machineState, setMachineState] = useState(CAMERA_STATES.IDLE)
  const [latestRecognition, setLatestRecognition] = useState(null)
  const [latestPlate, setLatestPlate] = useState('')
  const [consensus, setConsensus] = useState(summarizeConsensus([], OCR_REQUIRED_MATCHES))
  const [stablePlate, setStablePlate] = useState(null)
  const [recognitionError, setRecognitionError] = useState('')
  const [cameraFeedback, setCameraFeedback] = useState('Searching for plate')

  useEffect(() => {
    if (videoRef.current) videoRef.current.srcObject = camera.stream
  }, [camera.stream])

  const clearRecognition = useCallback((notifyParent = true) => {
    observationsRef.current = []
    frameBufferRef.current = []
    missedFramesRef.current = 0
    ocrCallsRef.current = 0
    scanStartedRef.current = performance.now()
    stableRef.current = null
    setLatestRecognition(null)
    setLatestPlate('')
    setConsensus(summarizeConsensus([], OCR_REQUIRED_MATCHES))
    setStablePlate(null)
    setRecognitionError('')
    setCameraFeedback('Searching for plate')
    if (notifyParent) onResetPlate?.()
  }, [onResetPlate])

  const scanAgain = useCallback((notifyParent = true) => {
    clearRecognition(notifyParent)
    setMachineState(camera.stream ? CAMERA_STATES.SCANNING : CAMERA_STATES.IDLE)
  }, [camera.stream, clearRecognition])

  useEffect(() => {
    if (previousResetKey.current === workflowResetKey) return undefined
    previousResetKey.current = workflowResetKey
    const timeout = window.setTimeout(() => scanAgain(false), resumeDelayMs)
    return () => window.clearTimeout(timeout)
  }, [resumeDelayMs, scanAgain, workflowResetKey])

  const publishStablePlate = useCallback(async (stable, selectedMode = mode) => {
    if (!stable) return
    if (selectedMode === 'EXIT') setMachineState(CAMERA_STATES.PROCESSING_EXIT)
    else setMachineState(CAMERA_STATES.PROCESSING_ENTRY)
    try {
      await onStablePlate?.({
        mode: selectedMode,
        plate: stable.plate,
        manuallyCorrected: stable.manuallyCorrected || false,
        detectionConfidence: stable.manuallyCorrected ? null : stable.averageDetectionConfidence,
        ocrConfidence: stable.manuallyCorrected ? null : stable.averageOcrConfidence,
      })
      setMachineState(CAMERA_STATES.STABLE)
    } catch (error) {
      setRecognitionError(errorMessage(error))
      setMachineState(CAMERA_STATES.STABLE)
    }
  }, [mode, onStablePlate])

  const handleFrameAnalysis = useCallback(async (response, blob) => {
    setLatestRecognition(response)
    const primary = selectPrimaryQualityDetection(response)
    if (CAMERA_DEBUG) {
      console.debug('[camera] analysis', {
        yoloMs: Math.round(response.yolo_inference_time * 1000),
        qualityMs: Math.round(response.quality_scoring_time * 1000),
        primary,
      })
    }

    if (!primary) {
      missedFramesRef.current += 1
      setCameraFeedback('Searching for plate')
      if (missedFramesRef.current >= 2) {
        frameBufferRef.current = []
        observationsRef.current = []
        ocrCallsRef.current = 0
        scanStartedRef.current = performance.now()
        setConsensus(summarizeConsensus([], OCR_REQUIRED_MATCHES))
        setLatestPlate('')
      }
      return
    }

    missedFramesRef.current = 0
    if (primary.camera_status !== 'candidate') {
      setCameraFeedback(
        primary.camera_status === 'too_small'
          ? 'Too far — move the plate closer'
          : primary.camera_status === 'too_blurry'
            ? 'Image is blurry — hold steady'
            : 'Move the plate closer and hold steady',
      )
      return
    }

    const update = updateFrameBuffer(frameBufferRef.current, {
      id: ++frameIdRef.current,
      blob,
      detection: primary,
    })
    frameBufferRef.current = update.buffer
    if (update.reset) {
      observationsRef.current = []
      ocrCallsRef.current = 0
      scanStartedRef.current = performance.now()
      setConsensus(summarizeConsensus([], OCR_REQUIRED_MATCHES))
      setLatestPlate('')
    }
    if (frameBufferRef.current.length < MIN_QUALITY_FRAMES) {
      setCameraFeedback('Plate detected — waiting for a clearer image')
      return
    }

    setCameraFeedback('Reading plate')
    const ranked = rankBestFrames(frameBufferRef.current)
    let nextWindow = observationsRef.current
    for (const [rankIndex, candidate] of ranked.slice(0, 3).entries()) {
      const ocrStarted = performance.now()
      const ocrResponse = await recognizeFrame(candidate.blob)
      ocrCallsRef.current += 1
      const detection = selectPrimaryDetection(ocrResponse)
      const observation = detectionObservation(detection)
      if (detection?.normalized_text) setLatestPlate(detection.normalized_text)
      nextWindow = advanceObservationWindow(nextWindow, observation, OCR_EVIDENCE_WINDOW_SIZE)
      if (CAMERA_DEBUG) {
        console.debug('[camera] selected frame OCR', {
          qualityScore: candidate.qualityScore,
          sharpness: candidate.detection.sharpness,
          selectedBestFrameIndex: candidate.id,
          selectedRank: rankIndex + 1,
          ocrCalls: ocrCallsRef.current,
          elapsedMs: Math.round(performance.now() - ocrStarted),
          plate: observation?.plate || null,
          ocrConfidence: observation?.ocrConfidence || 0,
        })
      }
      if (observation && observation.ocrConfidence >= STRONG_OCR_CONFIDENCE) break
    }

    // Release the entire batch so OCR cannot run on every sampled camera frame.
    frameBufferRef.current = []
    observationsRef.current = nextWindow
    const nextConsensus = summarizeConsensus(nextWindow, OCR_REQUIRED_MATCHES)
    setConsensus(nextConsensus)
    if (nextConsensus.stable && !stableRef.current) {
      const stable = { ...nextConsensus.leader, manuallyCorrected: false }
      stableRef.current = stable
      setStablePlate(stable)
      setCameraFeedback('Stable plate detected')
      setMachineState(CAMERA_STATES.STABLE)
      if (CAMERA_DEBUG) {
        console.debug('[camera] stable recognition', {
          finalOcr: stable.plate,
          ocrCalls: ocrCallsRef.current,
          timeToStableMs: Math.round(performance.now() - scanStartedRef.current),
        })
      }
      await publishStablePlate(stable)
    } else {
      setCameraFeedback('Hold steady for recognition')
    }
  }, [publishStablePlate])

  const handleRecognitionError = useCallback((error) => {
    setRecognitionError(
      error?.response ? 'Recognition server unavailable.' : errorMessage(error),
    )
    setMachineState(CAMERA_STATES.ERROR)
  }, [])

  const { isProcessing } = useRecognitionLoop({
    enabled: Boolean(camera.stream) && machineState === CAMERA_STATES.SCANNING,
    videoRef,
    canvasRef,
    intervalMs: recognitionIntervalMs,
    onResponse: handleFrameAnalysis,
    onError: handleRecognitionError,
  })

  const startCamera = async () => {
    try {
      await camera.start()
      clearRecognition()
      setMachineState(CAMERA_STATES.SCANNING)
    } catch {
      setMachineState(CAMERA_STATES.ERROR)
    }
  }

  const stopCamera = () => {
    camera.stop()
    clearRecognition()
    setMachineState(CAMERA_STATES.IDLE)
  }

  const changeMode = (nextMode) => {
    if (nextMode === mode) return
    setMode(nextMode)
    clearRecognition()
    setMachineState(camera.stream ? CAMERA_STATES.SCANNING : CAMERA_STATES.IDLE)
  }

  const editStablePlate = (value) => {
    const normalized = normalizePlate(value)
    const updated = {
      ...stableRef.current,
      plate: normalized,
      manuallyCorrected: true,
      averageOcrConfidence: null,
    }
    stableRef.current = updated
    setStablePlate(updated)
    setRecognitionError('')
    onManualPlate?.(normalized)
  }

  const statusError = camera.error || recognitionError
  const leaderCount = consensus.leader?.count || 0

  return (
    <div className="camera-recognition">
      <CameraControls
        mode={mode}
        onModeChange={changeMode}
        cameraStatus={camera.status}
        devices={camera.devices}
        selectedDeviceId={camera.selectedDeviceId}
        onDeviceChange={camera.setSelectedDeviceId}
        onStart={startCamera}
        onStop={stopCamera}
        active={Boolean(camera.stream)}
        starting={camera.status === 'STARTING'}
      />

      <CameraVideo
        videoRef={videoRef}
        stream={camera.stream}
        recognition={latestRecognition}
      />
      <canvas ref={canvasRef} className="camera-capture-canvas" aria-hidden="true" />

      {statusError && <div className="alert alert-danger camera-alert">{statusError}</div>}
      {machineState === CAMERA_STATES.ERROR && camera.stream && (
        <button className="btn btn-outline-primary w-100 mb-3" type="button" onClick={() => scanAgain(false)}>
          Resume Scanning
        </button>
      )}

      <div className="camera-recognition-status" aria-live="polite">
        <div>
          <span>Recognition</span>
          <strong>{stablePlate?.plate || latestPlate || (camera.stream ? 'Scanning...' : 'Not started')}</strong>
        </div>
        <div>
          <span>Stability</span>
          <strong>{leaderCount} / {OCR_REQUIRED_MATCHES} votes</strong>
          <small>{leaderCount} matches in last {consensus.sampledFrames} selected frames</small>
        </div>
        <div>
          <span>Status</span>
          <strong className={stablePlate ? 'stable-text' : ''}>
            {stablePlate ? '✓ Stable' : isProcessing ? cameraFeedback : machineState === CAMERA_STATES.ERROR ? 'Paused after error' : camera.stream ? cameraFeedback : 'Idle'}
          </strong>
        </div>
      </div>

      {stablePlate && (
        <div className="stable-plate-card">
          <p>Stable plate detected · {mode}</p>
          <input
            className="form-control plate-input"
            aria-label="Edit stable camera plate"
            value={stablePlate.plate}
            onChange={(event) => editStablePlate(event.target.value)}
          />
          {!stablePlate.manuallyCorrected && (
            <small>
              OCR {Math.round(stablePlate.averageOcrConfidence * 100)}% · YOLO {Math.round(stablePlate.averageDetectionConfidence * 100)}%
            </small>
          )}
          {stablePlate.manuallyCorrected && <small>Manually corrected · OCR confidence not applied</small>}
          <p className="stable-guidance">
            {mode === 'ENTRY'
              ? 'Select an empty parking slot, then confirm check-in.'
              : 'Review the active parking session and confirm payment before checkout.'}
          </p>
          <div className="camera-action-row">
            {mode === 'EXIT' && (
              <button className="btn btn-success" type="button" onClick={() => publishStablePlate(stablePlate)}>
                Review Checkout
              </button>
            )}
            <button className="btn btn-outline-primary" type="button" onClick={() => scanAgain()}>
              Scan Again
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
