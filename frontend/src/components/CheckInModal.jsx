import ModalShell from './ModalShell'
import { formatDate } from '../utils/format'

export default function CheckInModal({ plate, slot, onClose, onConfirm, busy }) {
  return (
    <ModalShell
      title="Check-in Confirmation"
      onClose={onClose}
      footer={(
        <>
          <button className="btn btn-light" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary" onClick={onConfirm} disabled={busy}>
            {busy ? 'Checking in…' : 'Confirm Check-in'}
          </button>
        </>
      )}
    >
      <dl className="detail-list">
        <div><dt>Plate</dt><dd className="plate-text">{plate}</dd></div>
        <div><dt>Slot</dt><dd>{slot.code}</dd></div>
        <div><dt>Current time</dt><dd>{formatDate(new Date().toISOString())}</dd></div>
      </dl>
    </ModalShell>
  )
}
