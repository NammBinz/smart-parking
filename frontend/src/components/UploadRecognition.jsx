import { useEffect, useRef, useState } from 'react'
import { errorMessage, recognizeImage } from '../services/api'
import ImageDetectionOverlay from './ImageDetectionOverlay'
import RecognitionResult from './RecognitionResult'

export default function UploadRecognition({ onSelectPlate, onResetSelection, resetKey }) {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [recognition, setRecognition] = useState(null)
  const [selectedIndex, setSelectedIndex] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const previewRef = useRef('')

  const clearLocal = () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    previewRef.current = ''
    setFile(null)
    setPreviewUrl('')
    setRecognition(null)
    setSelectedIndex(null)
    setError('')
  }

  useEffect(() => () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
  }, [])

  useEffect(() => {
    if (resetKey > 0) clearLocal()
  }, [resetKey])

  const selectImage = (event) => {
    const nextFile = event.target.files?.[0]
    if (!nextFile) return
    clearLocal()
    onResetSelection()
    const objectUrl = URL.createObjectURL(nextFile)
    previewRef.current = objectUrl
    setFile(nextFile)
    setPreviewUrl(objectUrl)
  }

  const chooseDetection = (index, response = recognition) => {
    const detection = response?.detections[index]
    if (!detection) return
    setSelectedIndex(index)
    onSelectPlate(detection, response.image_path)
  }

  const runRecognition = async () => {
    if (!file) return
    setRunning(true)
    setError('')
    setRecognition(null)
    setSelectedIndex(null)
    onResetSelection()
    try {
      const response = await recognizeImage(file)
      setRecognition(response)
      const onlyDetection = response.detections.length === 1 ? response.detections[0] : null
      if (
        onlyDetection?.is_valid
        && onlyDetection.ocr_confidence >= 0.5
        && onlyDetection.detection_confidence >= 0.5
      ) {
        chooseDetection(0, response)
      }
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="upload-recognition">
      <label className="form-label fw-semibold" htmlFor="vehicle-image">Select vehicle image</label>
      <input
        key={resetKey}
        className="form-control"
        id="vehicle-image"
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        onChange={selectImage}
      />
      {previewUrl ? (
        <ImageDetectionOverlay
          previewUrl={previewUrl}
          recognition={recognition}
          selectedIndex={selectedIndex}
          onSelect={(index) => chooseDetection(index)}
        />
      ) : (
        <p className="helper-text mt-3">Choose a JPG, JPEG, or PNG image up to 10 MB.</p>
      )}

      {file && (
        <button className="btn btn-primary w-100 mt-3" onClick={runRecognition} disabled={running}>
          {running && <span className="spinner-border spinner-border-sm me-2" aria-hidden="true" />}
          {running ? 'Detecting license plate...' : 'Recognize License Plate'}
        </button>
      )}
      {error && <div className="alert alert-danger mt-3 mb-0">{error}</div>}
      {recognition?.message && <div className="alert alert-warning mt-3 mb-0">{recognition.message}</div>}
      {recognition?.detections.length > 1 && (
        <p className="helper-text mt-3 mb-2">Multiple plates found. Select the vehicle you want to check in.</p>
      )}
      {recognition?.detections.length > 0 && (
        <div className="recognition-results">
          {recognition.detections.map((detection, index) => (
            <RecognitionResult
              key={`${detection.bbox.x1}-${detection.bbox.y1}-${index}`}
              detection={detection}
              index={index}
              selected={selectedIndex === index}
              onUse={() => chooseDetection(index)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
