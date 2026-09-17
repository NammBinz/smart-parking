import ParkingSlotCard from './ParkingSlotCard'

export default function ParkingGrid({ slots, plate, onSlotClick, loading }) {
  return (
    <section className="panel parking-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Live layout</p>
          <h2>Parking spaces</h2>
        </div>
        <div className="legend">
          <span><i className="dot empty" /> Available</span>
          <span><i className="dot occupied" /> Occupied</span>
          <span><i className="dot disabled" /> Disabled</span>
        </div>
      </div>
      {loading ? (
        <div className="loading-state">Loading parking spaces…</div>
      ) : (
        <div className="parking-grid">
          {slots.map((slot) => (
            <ParkingSlotCard
              key={slot.id}
              slot={slot}
              canCheckIn={Boolean(plate)}
              onClick={onSlotClick}
            />
          ))}
        </div>
      )}
    </section>
  )
}
