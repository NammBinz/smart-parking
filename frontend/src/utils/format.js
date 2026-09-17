export function normalizePlate(value) {
  return value.trim().toUpperCase().replace(/[\s.-]+/g, '')
}

export function formatDate(value) {
  if (!value) return '—'
  return new Intl.DateTimeFormat('en-GB', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(`${value}${value.endsWith('Z') ? '' : 'Z'}`))
}

export function formatMoney(value, currency = 'VND') {
  return new Intl.NumberFormat('vi-VN', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(Number(value || 0))
}

export function formatDuration(seconds) {
  const safeSeconds = Math.max(0, Math.floor(seconds || 0))
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  return `${hours}h ${minutes}m`
}
