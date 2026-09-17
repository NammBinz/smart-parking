import { formatMoney } from '../utils/format'

const cards = [
  ['total_slots', 'Total Slots', 'total'],
  ['occupied_slots', 'Occupied', 'occupied'],
  ['empty_slots', 'Available', 'available'],
  ['revenue_today', "Today's Revenue", 'revenue'],
]

export default function DashboardStats({ dashboard, currency }) {
  return (
    <section className="stats-grid" aria-label="Parking overview">
      {cards.map(([key, label, style]) => (
        <article className={`stat-card stat-${style}`} key={key}>
          <span>{label}</span>
          <strong>{key === 'revenue_today' ? formatMoney(dashboard[key], currency) : dashboard[key] ?? '—'}</strong>
        </article>
      ))}
    </section>
  )
}
