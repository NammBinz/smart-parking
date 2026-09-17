import { useEffect, useState } from 'react'
import { normalizePlate } from '../utils/format'

const tabs = [
  ['camera', 'Camera'],
  ['upload', 'Upload Image'],
  ['manual', 'Manual Input'],
]

export default function InputPanel({ plate, onPlateChange, onClear }) {
  const [activeTab, setActiveTab] = useState('manual')
  const [previewUrl, setPreviewUrl] = useState('')

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl])

  const selectImage = (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(file))
  }

  return (
    <section className="panel input-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Vehicle input</p>
          <h2>Identify vehicle</h2>
        </div>
        <span className="phase-badge">Phase 1</span>
      </div>
      <div className="tab-list" role="tablist">
        {tabs.map(([key, label]) => (
          <button
            className={activeTab === key ? 'active' : ''}
            key={key}
            onClick={() => setActiveTab(key)}
            role="tab"
            aria-selected={activeTab === key}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="tab-content-area">
        {activeTab === 'camera' && (
          <div className="placeholder-state">
            <div className="placeholder-icon">◎</div>
            <p>Camera integration will be implemented in Phase 2.</p>
          </div>
        )}
        {activeTab === 'upload' && (
          <div>
            <label className="form-label fw-semibold" htmlFor="vehicle-image">Select vehicle image</label>
            <input
              className="form-control"
              id="vehicle-image"
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              onChange={selectImage}
            />
            {previewUrl ? (
              <img className="upload-preview" src={previewUrl} alt="Selected vehicle" />
            ) : (
              <p className="helper-text mt-3">JPG, JPEG, and PNG files are supported. AI processing arrives in Phase 2.</p>
            )}
          </div>
        )}
        {activeTab === 'manual' && (
          <div>
            <label className="form-label fw-semibold" htmlFor="plate-input">License plate number</label>
            <input
              className="form-control form-control-lg plate-input"
              id="plate-input"
              value={plate}
              onChange={(event) => onPlateChange(normalizePlate(event.target.value))}
              placeholder="e.g. 29-H1 999.99"
              autoComplete="off"
            />
            <div className="detected-plate">
              <span>Detected / Entered Plate</span>
              <strong>{plate || '—'}</strong>
            </div>
            <button className="btn btn-sm btn-outline-secondary mt-3" onClick={onClear} disabled={!plate}>
              Clear plate
            </button>
          </div>
        )}
      </div>
    </section>
  )
}
