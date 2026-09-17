import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { errorMessage, getHistory } from '../services/api'
import { formatDate, formatMoney } from '../utils/format'
import ModalShell from './ModalShell'

export default function ReportsModal({ currency, onClose }) {
  const [history, setHistory] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    getHistory().then(setHistory).catch((err) => setError(errorMessage(err)))
  }, [])

  const chartData = useMemo(() => {
    const totals = history.reduce((acc, item) => {
      const method = item.payment_method || 'Unpaid'
      acc[method] = (acc[method] || 0) + Number(item.amount || 0)
      return acc
    }, {})
    return Object.entries(totals).map(([method, amount]) => ({ method, amount }))
  }, [history])

  return (
    <ModalShell title="Parking Reports" onClose={onClose} size="xl">
      {error && <div className="alert alert-danger">{error}</div>}
      {chartData.length > 0 && (
        <div className="report-chart">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} margin={{ left: 15, right: 15 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="method" />
              <YAxis />
              <Tooltip formatter={(value) => formatMoney(value, currency)} />
              <Bar dataKey="amount" fill="#2f7d63" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="table-responsive">
        <table className="table align-middle reports-table">
          <thead>
            <tr>
              <th>Plate Number</th><th>Parking Slot</th><th>Entry Time</th><th>Exit Time</th>
              <th>Billed Hours</th><th>Amount</th><th>Payment Method</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item) => (
              <tr key={item.id}>
                <td className="plate-text">{item.plate_number}</td><td>{item.slot}</td>
                <td>{formatDate(item.entry_time)}</td><td>{formatDate(item.exit_time)}</td>
                <td>{item.billed_hours ?? '—'}</td>
                <td>{item.amount ? formatMoney(item.amount, currency) : '—'}</td>
                <td>{item.payment_method || '—'}</td>
              </tr>
            ))}
            {!history.length && !error && <tr><td colSpan="7" className="text-center text-secondary py-4">No parking history yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </ModalShell>
  )
}
