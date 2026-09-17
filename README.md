# Smart Parking Management System

A university web application for parking-space management, vehicle check-in, billed checkout, payments, reporting, and license plate recognition. The repository currently implements **Phase 2: license plate recognition from uploaded images** while preserving all Phase 1 manual workflows.

## Technology stack

- Frontend: React, Vite, Axios, Bootstrap, Recharts
- Backend: Python, FastAPI, SQLAlchemy, Pydantic, Uvicorn
- Database: SQLite (`backend/data/parking.db`)
- Image recognition: YOLO11, OpenCV headless, EasyOCR

The supplied YOLO weights are stored at `backend/models/best.pt`. Do not rename, replace, or retrain this file.

## Project structure

```text
SmartParking/
├── backend/
│   ├── app/
│   │   ├── ai/             # Model manager, detection, preprocessing, OCR, validation, pipeline
│   │   ├── api/            # FastAPI routers
│   │   ├── database/       # Engine, sessions, and idempotent seeding
│   │   ├── models/         # SQLAlchemy database models
│   │   ├── schemas/        # Request and response models
│   │   ├── services/       # Parking rules and plate normalization
│   │   └── main.py
│   ├── data/               # Runtime SQLite database
│   ├── models/best.pt      # Existing trained YOLO11 model
│   ├── scripts/test_recognition.py
│   ├── tests/
│   ├── uploads/            # Generated upload names; runtime files are ignored by Git
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
- Swagger: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>
- AI status: <http://127.0.0.1:8000/api/ai/status>

The first real OCR request may download EasyOCR model weights. CUDA is optional; the recognition pipeline works on CPU.

Tables are created without dropping existing data. Missing default settings and missing codes among A1–A10 and B1–B10 are seeded idempotently.

Run all backend tests from `backend/`:

```powershell
pytest
```

AI tests mock model inference and therefore do not download OCR weights.

## Image recognition API

Send a JPEG or PNG of at most 10 MB as the `file` field of a multipart request:

```text
POST /api/ai/recognize-image
Content-Type: multipart/form-data
```

The endpoint saves the original image under `backend/uploads/` using a UUID filename. It returns every YOLO detection, bounding boxes, class information, OCR text and confidence, selected preprocessing variant, validation status, and the best candidate index. A valid image with no detections returns HTTP 200 with an empty detection list.

To exercise the same production pipeline manually with a real image:

```powershell
cd backend
python scripts/test_recognition.py path/to/car.jpg
```

The script prints every detection, bounding box, YOLO confidence, raw and normalized OCR text, OCR confidence, validation status, and preprocessing variant. It does not check a vehicle in.

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

## Phase 2 features

- All Phase 1 manual check-in, checkout, reporting, settings, and slot features
- Lazy, process-wide YOLO and EasyOCR instances
- Detection from both trained model classes without a class filter
- Five-percent padded plate crops
- Upscaled color, grayscale, CLAHE, and adaptive-threshold OCR variants
- Spatial ordering for one-line and two-line OCR fragments
- Confidence-weighted OCR aggregation and conservative plate validation
- Ranked multi-vehicle results without discarding invalid OCR candidates
- Safe UUID upload storage and static image serving
- Responsive browser-side detection-box overlays
- Explicit candidate selection and manual correction before check-in
- Optional source image and confidence metadata stored with parking sessions

## Phase 3

Live camera and video processing are intentionally not included. Phase 3 can add camera modes, continuous-frame processing, multi-frame voting, cooldown logic, and real-time streaming. Automatic camera check-in or checkout is not part of Phase 2.
