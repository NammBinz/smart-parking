import { useState } from 'react'
import { normalizePlate } from '../utils/format'
import UploadRecognition from './UploadRecognition'

const tabs = [
  ['camera', 'Camera'],
  ['upload', 'Upload Image'],
  ['manual', 'Manual Input'],
]

export default function InputPanel({
  plate,
  plateSource,
  onPlateChange,
  onAiPlate,
  onResetAiSelection,
  onClear,
  recognitionResetKey,
}) {
  const [activeTab, setActiveTab] = useState('manual')

  return (
    <section className="panel input-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Vehicle input</p>
          <h2>Identify vehicle</h2>
        </div>
        <span className="phase-badge">Phase 2</span>
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
            <p>Live camera integration will be implemented in Phase 3.</p>
          </div>
        )}
        {activeTab === 'upload' && (
          <UploadRecognition
            resetKey={recognitionResetKey}
            onResetSelection={onResetAiSelection}
            onSelectPlate={(detection, imagePath) => {
              onAiPlate(detection, imagePath)
              if (!detection.is_valid) setActiveTab('manual')
            }}
          />
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
              {plateSource && <small>{plateSource}</small>}
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
