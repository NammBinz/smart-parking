import { useState } from 'react'
import ModalShell from './ModalShell'
import { formatDate, formatDuration, formatMoney } from '../utils/format'

export default function CheckoutModal({ preview, currency, onClose, onConfirm, busy }) {
  const [method, setMethod] = useState('CASH')
  return (
    <ModalShell
      title="Payment & Checkout"
      onClose={onClose}
      footer={(
        <>
          <button className="btn btn-light" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-success" onClick={() => onConfirm(method)} disabled={busy}>
            {busy ? 'Processing…' : 'Confirm Payment & Checkout'}
          </button>
        </>
      )}
    >
      <dl className="detail-list">
        <div><dt>Plate Number</dt><dd className="plate-text">{preview.plate_number}</dd></div>
        <div><dt>Parking Slot</dt><dd>{preview.slot}</dd></div>
        <div><dt>Entry Time</dt><dd>{formatDate(preview.entry_time)}</dd></div>
        <div><dt>Current Time</dt><dd>{formatDate(preview.current_time)}</dd></div>
        <div><dt>Actual Duration</dt><dd>{formatDuration(preview.duration_seconds)}</dd></div>
        <div><dt>Billed Hours</dt><dd>{preview.billed_hours}</dd></div>
        <div><dt>Price Per Hour</dt><dd>{formatMoney(preview.price_per_hour, currency)}</dd></div>
        <div className="total-line"><dt>Total Amount</dt><dd>{formatMoney(preview.amount, currency)}</dd></div>
      </dl>
      <fieldset className="mt-4">
        <legend className="form-label fw-semibold">Payment method</legend>
        <div className="payment-options">
          <label><input type="radio" name="method" value="CASH" checked={method === 'CASH'} onChange={(e) => setMethod(e.target.value)} /> Cash</label>
          <label><input type="radio" name="method" value="TRANSFER" checked={method === 'TRANSFER'} onChange={(e) => setMethod(e.target.value)} /> Bank Transfer</label>
        </div>
      </fieldset>
    </ModalShell>
  )
}
