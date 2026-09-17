# Smart Parking Management System

A university web application for managing parking spaces, vehicle check-in, billed checkout, payments, reports, and system settings. This repository currently implements **Phase 1** only: the application architecture, API, SQLite persistence, core parking rules, basic React interface, and automated integration tests.

## Technology stack

- Frontend: React, Vite, Axios, Bootstrap, Recharts
- Backend: Python, FastAPI, SQLAlchemy, Pydantic, Uvicorn
- Database: SQLite (`backend/data/parking.db`)
- Future AI integration: YOLO11, OpenCV, and EasyOCR (not loaded or implemented in Phase 1)

## Project structure

```text
SmartParking/
├── backend/
│   ├── app/
│   │   ├── ai/             # Reserved for Phase 2
│   │   ├── api/            # FastAPI routers
│   │   ├── database/       # Engine, sessions, and idempotent seeding
│   │   ├── models/         # SQLAlchemy database models
│   │   ├── schemas/        # Request and response models
│   │   ├── services/       # Parking rules and reusable normalization
│   │   └── main.py
│   ├── data/               # SQLite database
│   ├── models/best.pt      # Existing trained model, reserved for Phase 2
│   ├── tests/
│   ├── uploads/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── styles/
│   │   ├── utils/
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
├── .gitignore
└── README.md
```

## Backend installation

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- API: <http://127.0.0.1:8000>
- Swagger documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

Tables are created without dropping existing data when the application starts. Missing default settings and any missing codes among slots A1–A10 and B1–B10 are seeded idempotently.

Run backend tests from `backend/`:

```powershell
pytest
```

## Frontend installation

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The frontend expects the API at `http://127.0.0.1:8000`.

Create a production build with:

```powershell
npm run build
```

## Phase 1 features

- Dashboard KPIs sourced from the API
- Responsive parking grid with available, occupied, and disabled states
- Manual plate entry with shared normalization behavior
- Image selection and local preview (no AI processing)
- Transactional vehicle check-in with slot and duplicate-session validation
- Non-mutating checkout preview with rounded-up hourly billing
- Cash or bank-transfer checkout and persisted payments
- Parking history table and payment-method revenue visualization
- Editable price, minimum hours, future confidence, and currency settings
- Slot management API with occupied-slot safety rules
- CORS restricted to the Vite development origin

## Future Phase 2

Phase 2 can add YOLO license plate detection using `backend/models/best.pt`, OpenCV preprocessing, EasyOCR, and live camera processing. No model training, model loading, camera streaming, OCR, or image inference is present in Phase 1.
