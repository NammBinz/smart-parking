export default function CameraControls({
  mode,
  onModeChange,
  cameraStatus,
  devices,
  selectedDeviceId,
  onDeviceChange,
  onStart,
  onStop,
  active,
  starting,
}) {
  return (
    <div className="camera-controls">
      <div className="camera-control-row">
        <div>
          <span className="camera-control-label">Camera status</span>
          <span className={`camera-status ${active ? 'active' : ''}`}>
            <i aria-hidden="true" /> {starting ? 'Starting' : active ? 'Active' : cameraStatus === 'ERROR' ? 'Error' : 'Inactive'}
          </span>
        </div>
        <div>
          <span className="camera-control-label">Mode</span>
          <div className="camera-mode-toggle" role="group" aria-label="Camera workflow mode">
            {['ENTRY', 'EXIT'].map((value) => (
              <button
                type="button"
                className={mode === value ? 'active' : ''}
                key={value}
                onClick={() => onModeChange(value)}
              >
                {value}
              </button>
            ))}
          </div>
        </div>
      </div>

      <label className="form-label camera-device-select">
        Video input
        <select
          className="form-select"
          value={selectedDeviceId}
          onChange={(event) => onDeviceChange(event.target.value)}
          disabled={active || starting}
        >
          {devices.length === 0 && <option value="">Default camera</option>}
          {devices.map((device, index) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Camera ${index + 1}`}
            </option>
          ))}
        </select>
      </label>

      <div className="camera-action-row">
        <button className="btn btn-primary" type="button" onClick={onStart} disabled={active || starting}>
          {starting ? 'Starting Camera...' : 'Start Camera'}
        </button>
        <button className="btn btn-outline-secondary" type="button" onClick={onStop} disabled={!active && !starting}>
          Stop Camera
        </button>
      </div>
    </div>
  )
}
