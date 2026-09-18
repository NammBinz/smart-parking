import { useCallback, useEffect, useState } from 'react'
import CheckInModal from './components/CheckInModal'
import CheckoutModal from './components/CheckoutModal'
import DashboardStats from './components/DashboardStats'
import Header from './components/Header'
import InputPanel from './components/InputPanel'
import Notification from './components/Notification'
import ParkingGrid from './components/ParkingGrid'
import ReportsModal from './components/ReportsModal'
import SettingsModal from './components/SettingsModal'
import VehicleDetailModal from './components/VehicleDetailModal'
import {
  checkIn,
  checkout,
  errorMessage,
  getCheckoutPreview,
  getDashboard,
  getSettings,
  getSlots,
} from './services/api'

const emptyDashboard = {
  total_slots: 0,
  occupied_slots: 0,
  empty_slots: 0,
  disabled_slots: 0,
  vehicles_today: 0,
  revenue_today: 0,
}

export default function App() {
  const [dashboard, setDashboard] = useState(emptyDashboard)
  const [slots, setSlots] = useState([])
  const [currency, setCurrency] = useState('VND')
  const [plate, setPlate] = useState('')
  const [aiMetadata, setAiMetadata] = useState(null)
  const [recognitionResetKey, setRecognitionResetKey] = useState(0)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [selectedSlot, setSelectedSlot] = useState(null)
  const [checkInSlot, setCheckInSlot] = useState(null)
  const [checkoutPreview, setCheckoutPreview] = useState(null)
  const [showReports, setShowReports] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [notification, setNotification] = useState(null)

  const notify = useCallback((message, type = 'success') => {
    setNotification({ message, type })
    window.setTimeout(() => setNotification(null), 4500)
  }, [])

  const refresh = useCallback(async () => {
    const [dashboardData, slotData] = await Promise.all([getDashboard(), getSlots()])
    setDashboard(dashboardData)
    setSlots(slotData)
  }, [])

  useEffect(() => {
    Promise.all([refresh(), getSettings().then((data) => setCurrency(data.currency))])
      .catch((err) => notify(errorMessage(err), 'error'))
      .finally(() => setLoading(false))
  }, [notify, refresh])

  const handleSlotClick = (slot) => {
    if (slot.status === 'OCCUPIED') setSelectedSlot(slot)
    if (slot.status === 'EMPTY' && plate) setCheckInSlot(slot)
  }

  const handlePlateChange = (value) => {
    setPlate(value)
    if (aiMetadata && value !== aiMetadata.originalPlate) {
      setAiMetadata({
        ...aiMetadata,
        entry_ocr_confidence: null,
        manuallyCorrected: true,
      })
    }
  }

  const selectAiPlate = (detection, imagePath) => {
    setPlate(detection.normalized_text)
    setAiMetadata({
      originalPlate: detection.normalized_text,
      entry_image: imagePath,
      entry_detection_confidence: detection.detection_confidence,
      entry_ocr_confidence: detection.ocr_confidence,
      manuallyCorrected: false,
      source: 'upload',
    })
  }

  const handleCameraStable = useCallback(async ({
    mode,
    plate: cameraPlate,
    manuallyCorrected,
    detectionConfidence,
    ocrConfidence,
  }) => {
    setPlate(cameraPlate)
    setAiMetadata({
      originalPlate: cameraPlate,
      entry_image: null,
      entry_detection_confidence: detectionConfidence,
      entry_ocr_confidence: manuallyCorrected ? null : ocrConfidence,
      manuallyCorrected,
      source: 'camera',
    })

    if (mode !== 'EXIT') return
    setBusy(true)
    try {
      const preview = await getCheckoutPreview(cameraPlate)
      setCheckoutPreview(preview)
    } catch (err) {
      const message = errorMessage(err)
      notify(message, 'error')
      throw new Error(message)
    } finally {
      setBusy(false)
    }
  }, [notify])

  const resetAiSelection = () => {
    if (aiMetadata) setPlate('')
    setAiMetadata(null)
  }

  const clearPlate = () => {
    setPlate('')
    setAiMetadata(null)
  }

  const resetCameraWorkflow = useCallback(() => {
    setPlate('')
    setAiMetadata(null)
    setCheckoutPreview(null)
  }, [])

  const confirmCheckIn = async () => {
    setBusy(true)
    try {
      const payload = { plate_number: plate, slot_id: checkInSlot.id }
      if (aiMetadata?.entry_image) payload.entry_image = aiMetadata.entry_image
      if (aiMetadata?.entry_detection_confidence != null) {
        payload.entry_detection_confidence = aiMetadata.entry_detection_confidence
      }
      if (aiMetadata?.entry_ocr_confidence != null) {
        payload.entry_ocr_confidence = aiMetadata.entry_ocr_confidence
      }
      await checkIn(payload)
      setCheckInSlot(null)
      setPlate('')
      setAiMetadata(null)
      setRecognitionResetKey((value) => value + 1)
      await refresh()
      notify(`Vehicle checked in successfully. Slot ${checkInSlot.code}.`)
    } catch (err) {
      notify(errorMessage(err), 'error')
    } finally {
      setBusy(false)
    }
  }

  const beginCheckout = async () => {
    setBusy(true)
    try {
      const preview = await getCheckoutPreview(selectedSlot.current_parking.plate_number)
      setSelectedSlot(null)
      setCheckoutPreview(preview)
    } catch (err) {
      notify(errorMessage(err), 'error')
    } finally {
      setBusy(false)
    }
  }

  const confirmCheckout = async (paymentMethod) => {
    setBusy(true)
    try {
      await checkout({ plate_number: checkoutPreview.plate_number, payment_method: paymentMethod })
      setCheckoutPreview(null)
      setPlate('')
      setAiMetadata(null)
      setRecognitionResetKey((value) => value + 1)
      await refresh()
      notify('Payment completed and vehicle checked out.')
    } catch (err) {
      notify(errorMessage(err), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app-shell">
      <Header onReports={() => setShowReports(true)} onSettings={() => setShowSettings(true)} />
      <main>
        <DashboardStats dashboard={dashboard} currency={currency} />
        <div className="content-layout">
          <InputPanel
            plate={plate}
            plateSource={aiMetadata ? (
              aiMetadata.manuallyCorrected
                ? `${aiMetadata.source === 'camera' ? 'Camera' : 'AI image'} · manually corrected`
                : aiMetadata.source === 'camera' ? 'Stable camera consensus' : 'Recognized from uploaded image'
            ) : ''}
            onPlateChange={handlePlateChange}
            onAiPlate={selectAiPlate}
            onResetAiSelection={resetAiSelection}
            onClear={clearPlate}
            recognitionResetKey={recognitionResetKey}
            onCameraStable={handleCameraStable}
            onCameraPlateEdit={handlePlateChange}
            onCameraReset={resetCameraWorkflow}
          />
          <ParkingGrid slots={slots} plate={plate} onSlotClick={handleSlotClick} loading={loading} />
        </div>
      </main>

      {checkInSlot && <CheckInModal plate={plate} slot={checkInSlot} onClose={() => setCheckInSlot(null)} onConfirm={confirmCheckIn} busy={busy} />}
      {selectedSlot && <VehicleDetailModal slot={selectedSlot} onClose={() => setSelectedSlot(null)} onCheckout={beginCheckout} busy={busy} />}
      {checkoutPreview && <CheckoutModal preview={checkoutPreview} currency={currency} onClose={() => setCheckoutPreview(null)} onConfirm={confirmCheckout} busy={busy} />}
      {showReports && <ReportsModal currency={currency} onClose={() => setShowReports(false)} />}
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} onSaved={(data) => { setCurrency(data.currency); notify('Settings updated.') }} />}
      <Notification notification={notification} onClose={() => setNotification(null)} />
    </div>
  )
}
