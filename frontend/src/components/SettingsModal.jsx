import { useEffect, useState } from 'react'
import { errorMessage, getSettings, updateSettings } from '../services/api'
import ModalShell from './ModalShell'

export default function SettingsModal({ onClose, onSaved }) {
  const [form, setForm] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    getSettings().then(setForm).catch((err) => setError(errorMessage(err)))
  }, [])

  const change = (event) => setForm({ ...form, [event.target.name]: event.target.value })
  const save = async () => {
    setBusy(true)
    setError('')
    try {
      const saved = await updateSettings({
        price_per_hour: Number(form.price_per_hour),
        minimum_hours: Number(form.minimum_hours),
        yolo_confidence: Number(form.yolo_confidence),
        currency: form.currency,
      })
      onSaved(saved)
      onClose()
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <ModalShell
      title="System Settings"
      onClose={onClose}
      footer={form && (
        <>
          <button className="btn btn-light" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={save} disabled={busy}>{busy ? 'Saving…' : 'Save Settings'}</button>
        </>
      )}
    >
      {error && <div className="alert alert-danger">{error}</div>}
      {!form ? <p>Loading settings…</p> : (
        <div className="settings-grid">
          <label>Price Per Hour<input className="form-control" type="number" min="0" name="price_per_hour" value={form.price_per_hour} onChange={change} /></label>
          <label>Minimum Hours<input className="form-control" type="number" min="1" name="minimum_hours" value={form.minimum_hours} onChange={change} /></label>
          <label>YOLO Confidence<input className="form-control" type="number" min="0" max="1" step="0.01" name="yolo_confidence" value={form.yolo_confidence} onChange={change} /></label>
          <label>Currency<input className="form-control" maxLength="10" name="currency" value={form.currency} onChange={change} /></label>
          <p className="helper-text settings-note">YOLO confidence is stored for Phase 2; no model is loaded in this phase.</p>
        </div>
      )}
    </ModalShell>
  )
}
