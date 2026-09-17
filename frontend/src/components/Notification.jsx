export default function Notification({ notification, onClose }) {
  if (!notification) return null
  return (
    <div className={`app-toast toast-${notification.type}`} role="status">
      <span>{notification.message}</span>
      <button onClick={onClose} aria-label="Dismiss">×</button>
    </div>
  )
}
