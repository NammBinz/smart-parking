export default function Header({ onReports, onSettings }) {
  return (
    <header className="app-header">
      <div>
        <p className="eyebrow mb-1">Operations dashboard</p>
        <h1>Smart Parking Management</h1>
      </div>
      <div className="d-flex gap-2">
        <button className="btn btn-outline-light" onClick={onReports}>Reports</button>
        <button className="btn btn-light" onClick={onSettings}>Settings</button>
      </div>
    </header>
  )
}
