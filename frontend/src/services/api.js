import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  headers: { 'Content-Type': 'application/json' },
  timeout: 10000,
})

export const getDashboard = () => api.get('/api/dashboard').then(({ data }) => data)
export const getSlots = () => api.get('/api/slots').then(({ data }) => data)
export const checkIn = (payload) => api.post('/api/parking/check-in', payload).then(({ data }) => data)
export const getCheckoutPreview = (plateNumber) =>
  api.post('/api/parking/checkout-preview', { plate_number: plateNumber }).then(({ data }) => data)
export const checkout = (payload) => api.post('/api/parking/checkout', payload).then(({ data }) => data)
export const getHistory = (params = {}) => api.get('/api/history', { params }).then(({ data }) => data)
export const getSettings = () => api.get('/api/settings').then(({ data }) => data)
export const updateSettings = (payload) => api.patch('/api/settings', payload).then(({ data }) => data)

export function errorMessage(error) {
  return error.response?.data?.detail || error.message || 'Something went wrong.'
}

export default api
