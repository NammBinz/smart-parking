# Smart Parking Management System

A university web application for parking-space management, vehicle check-in, billed checkout, payments, reporting, uploaded-image recognition, and **Phase 3.5 stabilized live-camera license plate recognition**. All Phase 1 manual workflows and Phase 2 image recognition remain available.

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

The endpoint saves the original image under `backend/uploads/` using a UUID filename. It returns every YOLO detection, bounding boxes, class information, OCR text and confidence, selected preprocessing variant, validation status, OCR status, and the best candidate index. A valid image with no detections returns HTTP 200 with an empty detection list. Detections smaller than the conservative 40×20 OCR threshold remain in the response with `ocr_status: "too_small"`, empty OCR text, and an invalid result so users can still enter the plate manually.

To exercise the same production pipeline manually with a real image:

```powershell
cd backend
python scripts/test_recognition.py path/to/car.jpg
```

The script prints every detection, bounding box, YOLO confidence, raw and normalized OCR text, OCR confidence, validation status, and preprocessing variant. It does not check a vehicle in.

For exhaustive OCR diagnostics, including every padding/variant candidate, fragment geometry, candidate timing, split-row status, and intermediate images under `backend/debug_output/`, run:

```powershell
python scripts/test_recognition.py path/to/car.jpg --debug
```

Normal two-line inference uses a progressive row-level ensemble. It starts with four targeted greedy OCR calls, reuses lazily computed preprocessing, and stops early when both rows have strong structural evidence. At most two targeted grayscale or beam-search fallbacks are added for weak rows. Top and bottom rows may come from different preprocessing variants or full-crop fragments, and conservative character alternatives require independent OCR evidence. Debug mode remains exhaustive, evaluating all seven preprocessing variants at 5%, 10%, and 15% padding with beam search for up to 63 OCR operations per suitable two-line detection.

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

Run the frontend consensus tests with:

```powershell
npm test
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

## Phase 3 live camera

Open the **Camera** tab and grant browser camera permission. Camera access uses `navigator.mediaDevices.getUserMedia()` and supports available video-input selection where the browser exposes device labels. Leaving the Camera tab or pressing **Stop Camera** stops every media track.

The browser captures a JPEG frame approximately every 1.5 seconds and first sends it to the YOLO-only quality endpoint:

```text
POST /api/ai/analyze-frame
Content-Type: multipart/form-data
```

Only one camera request is active at a time. The endpoint decodes each frame in memory, runs YOLO, and returns bounding-box size, area, Laplacian sharpness, brightness, center-zone bonus, weighted quality score, and separate YOLO/quality timings. It does not run OCR. Camera frames and continuous video are never written to `backend/uploads/` or persisted elsewhere.

The browser keeps a maximum of five JPEG blobs in a short in-memory buffer for one spatially associated vehicle. Tiny, distant, or blurry crops are rejected with actionable feedback. Once three usable frames are available, relative sharpness and the other quality signals select the best frame for `POST /api/ai/recognize-frame`. A second or third candidate is tried only if the preceding OCR is invalid or weak; the full batch is then released, so OCR does not run on every sampled frame. Camera constraints prefer 1920×1080 and request continuous autofocus when the browser and camera support it.

The camera workflow provides explicit **ENTRY** and **EXIT** modes. Exact normalized OCR results vote in a rolling three-selected-frame evidence window; two matching results are required for stability. Empty, invalid, unreadable, and `too_small` results do not cast positive votes. A materially different bounding box resets both the quality buffer and OCR evidence. For frames containing multiple vehicles, all YOLO boxes are displayed while the centered OCR-ready detection is tracked. This remains optimized for a single-vehicle parking gate rather than general multi-vehicle tracking.

When a plate becomes stable, scanning pauses:

- **ENTRY:** the user selects an empty slot and confirms the existing check-in modal.
- **EXIT:** the existing checkout preview is loaded, then the user selects payment and confirms checkout.

Neither workflow mutates parking state automatically. Stable plates remain editable, and manual correction removes the OCR-confidence association from the edited text. After a successful workflow, scanning resumes after a short cooldown so the same vehicle does not trigger repeatedly.

## Phase 3.5 accuracy evaluation

The normal OCR pipeline remains bounded to at most six EasyOCR calls per detected plate. If ordinary candidates are weak, one conservative four-corner perspective rectification may replace a fallback attempt; unsafe or ambiguous geometry returns the original crop unchanged. The supplied YOLO weights are unchanged and no training is performed.

Run the reproducible labeled-image benchmark from `backend/`:

```powershell
python scripts/benchmark_recognition.py
python scripts/benchmark_recognition.py --output-json benchmark/results.json
```

Ground truth is in `backend/benchmark/ground_truth.csv` and contains only manually known plates from the repository images. The report separates detector success from exact OCR accuracy and includes edit distance, character accuracy, error category, YOLO time, quality-gate time, OCR time, and total recognition time. After model warm-up on the current CPU environment, the Phase 3.5 baseline is 4/4 detected plates, 3/4 exact matches (75%), and 88.9% mean character accuracy; the remaining labeled plate is a missing-character OCR error.

The CSV format is:

```csv
filename,plate
image1.jpg,29Z158344
```

PaddleOCR is an optional benchmark comparison and is not a production dependency or automatic dual-OCR path:

```powershell
pip install -r requirements-paddle.txt
python scripts/benchmark_recognition.py --compare-paddle
```

The OCR adapter reads `PRIMARY_OCR_ENGINE` (default `easyocr`) and `ENABLE_SECONDARY_OCR` (default `false`) for explicitly integrated experiments, but the production API intentionally does not call the secondary fallback because no benchmark advantage has yet been established.

Enable camera timing and quality diagnostics in the browser console with `VITE_CAMERA_DEBUG=true`. Debug logging contains numeric metrics and recognized candidates only; it does not save frames.
