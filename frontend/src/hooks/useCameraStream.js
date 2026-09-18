import { useCallback, useEffect, useState } from 'react'

function cameraErrorMessage(error) {
  if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') {
    return 'Camera permission was denied.'
  }
  if (error?.name === 'NotFoundError' || error?.name === 'OverconstrainedError') {
    return 'No usable camera was found.'
  }
  if (error?.name === 'NotReadableError' || error?.name === 'AbortError') {
    return 'Unable to access the selected camera. It may be in use by another application.'
  }
  return 'Unable to access the selected camera.'
}

export default function useCameraStream() {
  const [stream, setStream] = useState(null)
  const [devices, setDevices] = useState([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [status, setStatus] = useState('IDLE')
  const [error, setError] = useState('')

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return []
    const available = await navigator.mediaDevices.enumerateDevices()
    const cameras = available.filter((device) => device.kind === 'videoinput')
    setDevices(cameras)
    setSelectedDeviceId((current) => (
      current && cameras.some((device) => device.deviceId === current)
        ? current
        : cameras[0]?.deviceId || ''
    ))
    return cameras
  }, [])

  const stop = useCallback(() => {
    setStream((current) => {
      current?.getTracks().forEach((track) => track.stop())
      return null
    })
    setStatus('IDLE')
  }, [])

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      const message = 'This browser does not support camera access.'
      setError(message)
      setStatus('ERROR')
      throw new Error(message)
    }

    stop()
    setStatus('STARTING')
    setError('')
    const video = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }
    try {
      const nextStream = await navigator.mediaDevices.getUserMedia({ audio: false, video })
      nextStream.getVideoTracks().forEach((track) => {
        track.addEventListener('ended', () => {
          setStream(null)
          setStatus('ERROR')
          setError('The camera was disconnected.')
        }, { once: true })
      })
      setStream(nextStream)
      setStatus('ACTIVE')
      await refreshDevices()
      return nextStream
    } catch (cameraError) {
      const message = cameraErrorMessage(cameraError)
      setStatus('ERROR')
      setError(message)
      throw new Error(message)
    }
  }, [refreshDevices, selectedDeviceId, stop])

  useEffect(() => {
    refreshDevices().catch(() => {})
    const mediaDevices = navigator.mediaDevices
    mediaDevices?.addEventListener?.('devicechange', refreshDevices)
    return () => mediaDevices?.removeEventListener?.('devicechange', refreshDevices)
  }, [refreshDevices])

  useEffect(() => stop, [stop])

  return {
    stream,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    status,
    error,
    start,
    stop,
    refreshDevices,
  }
}
