import { useEffect, useRef, useState } from 'react'

import { recognizeFrame } from '../services/api'

const MAX_CAPTURE_WIDTH = 1280
const MAX_CAPTURE_HEIGHT = 720
const JPEG_QUALITY = 0.85

export function captureVideoFrame(video, canvas) {
  if (!video?.videoWidth || !video?.videoHeight || !canvas) return Promise.resolve(null)
  const scale = Math.min(
    1,
    MAX_CAPTURE_WIDTH / video.videoWidth,
    MAX_CAPTURE_HEIGHT / video.videoHeight,
  )
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Unable to capture a camera frame.'))),
      'image/jpeg',
      JPEG_QUALITY,
    )
  })
}

export default function useRecognitionLoop({
  enabled,
  videoRef,
  canvasRef,
  intervalMs,
  onResponse,
  onError,
}) {
  const [isProcessing, setIsProcessing] = useState(false)
  const processingRef = useRef(false)
  const responseRef = useRef(onResponse)
  const errorRef = useRef(onError)

  useEffect(() => { responseRef.current = onResponse }, [onResponse])
  useEffect(() => { errorRef.current = onError }, [onError])

  useEffect(() => {
    if (!enabled) {
      setIsProcessing(false)
      return undefined
    }

    let cancelled = false
    let timeoutId = null
    let controller = null

    const schedule = (delay) => {
      if (!cancelled) timeoutId = window.setTimeout(run, delay)
    }

    const run = async () => {
      if (cancelled) return
      if (processingRef.current) {
        schedule(intervalMs)
        return
      }
      const started = performance.now()
      controller = new AbortController()
      processingRef.current = true
      setIsProcessing(true)
      try {
        const blob = await captureVideoFrame(videoRef.current, canvasRef.current)
        if (!blob) return
        const response = await recognizeFrame(blob, { signal: controller.signal })
        if (!cancelled) await responseRef.current?.(response)
      } catch (error) {
        if (!cancelled && error?.name !== 'CanceledError' && error?.name !== 'AbortError') {
          errorRef.current?.(error)
        }
      } finally {
        processingRef.current = false
        if (!cancelled) {
          setIsProcessing(false)
          const remaining = Math.max(0, intervalMs - (performance.now() - started))
          schedule(remaining)
        }
      }
    }

    schedule(0)
    return () => {
      cancelled = true
      if (timeoutId) window.clearTimeout(timeoutId)
      controller?.abort()
      processingRef.current = false
      setIsProcessing(false)
    }
  }, [canvasRef, enabled, intervalMs, videoRef])

  return { isProcessing }
}
