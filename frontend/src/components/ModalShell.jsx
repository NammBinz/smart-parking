export default function ModalShell({ title, children, footer, onClose, size = '' }) {
  return (
    <div className="modal-layer" role="presentation" onMouseDown={onClose}>
      <div
        className={`modal-card ${size ? `modal-${size}` : ''}`}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modal-card-header">
          <h2>{title}</h2>
          <button className="close-button" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="modal-card-body">{children}</div>
        {footer && <div className="modal-card-footer">{footer}</div>}
      </div>
    </div>
  )
}
