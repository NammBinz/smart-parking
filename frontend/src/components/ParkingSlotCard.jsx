import { formatDate } from '../utils/format'

export default function ParkingSlotCard({ slot, canCheckIn, onClick }) {
  const clickable = slot.status === 'OCCUPIED' || (slot.status === 'EMPTY' && canCheckIn)
  return (
    <button
      className={`slot-card slot-${slot.status.toLowerCase()}`}
      onClick={() => onClick(slot)}
      disabled={!clickable}
      aria-label={`${slot.code}: ${slot.status}`}
    >
      <span className="slot-code">{slot.code}</span>
      {slot.status === 'EMPTY' && <span>Available</span>}
      {slot.status === 'DISABLED' && <span>Disabled</span>}
      {slot.status === 'OCCUPIED' && (
        <>
          <strong>{slot.current_parking?.plate_number}</strong>
          <small>{formatDate(slot.current_parking?.entry_time)}</small>
        </>
      )}
    </button>
  )
}
