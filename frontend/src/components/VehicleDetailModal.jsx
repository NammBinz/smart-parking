import ModalShell from './ModalShell'
import { formatDate, formatDuration } from '../utils/format'

export default function VehicleDetailModal({ slot, onClose, onCheckout, busy }) {
  const entryTime = slot.current_parking?.entry_time
  const duration = entryTime ? (Date.now() - new Date(`${entryTime}Z`).getTime()) / 1000 : 0
  return (
    <ModalShell
      title="Vehicle Details"
      onClose={onClose}
      footer={(
        <>
          <button className="btn btn-light" onClick={onClose}>Close</button>
          <button className="btn btn-danger" onClick={onCheckout} disabled={busy}>
            {busy ? 'Calculating…' : 'Proceed to Checkout'}
          </button>
        </>
      )}
    >
      <dl className="detail-list">
        <div><dt>Plate Number</dt><dd className="plate-text">{slot.current_parking?.plate_number}</dd></div>
        <div><dt>Parking Slot</dt><dd>{slot.code}</dd></div>
        <div><dt>Entry Time</dt><dd>{formatDate(entryTime)}</dd></div>
        <div><dt>Current Parking Duration</dt><dd>{formatDuration(duration)}</dd></div>
      </dl>
    </ModalShell>
  )
}
