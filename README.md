# Warrior Blood

A Sickle Cell Disease (SCD) patient monitoring system built as an MSc Software Engineering dissertation project. The platform connects Community Health Workers (CHWs) with patients through a risk-stratified triage dashboard and a self-service patient portal, backed by a machine-learning VOC (Vaso-Occlusive Crisis) risk predictor.

---

## Features

- **VOC risk prediction** — LightGBM classifier (AUROC 0.808) served via ONNX for fast inference, with SHAP-based plain-English explanations
- **CHW dashboard** — triage view sorted by risk tier (HIGH / MODERATE / LOW), patient history charts, and the ability to log data on behalf of patients
- **Patient portal** — daily check-in, pain diary, and hydration diary; record linkage by email or UUID
- **Weather integration** — OpenWeatherMap alerts for heat stress, cold stress, and poor air quality, with a searchable dropdown covering 65 cities across SCD-prevalent regions
- **Hydration assessment** — WHO 2 000 ml daily target with Armstrong urine colour scale
- **Secure PHI handling** — Fernet encryption for all personal data at rest; JWT authentication with role-based access control

---

## Quick start

### Prerequisites

| Software | Version |
|----------|---------|
| Git | Any recent |
| Docker Desktop | 4.x or later |

Docker Desktop must be running before executing any `docker` commands.

### 1. Clone and enter the repository

```bash
git clone https://github.com/reDevState/warrior-blood-app.git
cd warrior-blood-app
```

### 2. Create the `.env` file

Copy the template below into a file named `.env` in the project root:

```env
DATABASE_URL=mysql+aiomysql://warrior:blood@db:3306/warriorblood?charset=utf8mb4
FERNET_KEY=your-fernet-key-here
SECRET_KEY=warrior-blood-mvp-dev-key-change-in-production-32c
OPENWEATHERMAP_API_KEY=your-owm-key-here
```

Generate a permanent Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 3. Build and start

```bash
docker compose build
docker compose up -d
```

### 4. First-time database setup

```bash
docker compose exec api alembic upgrade head
docker compose exec api bash -c "cd /app && PYTHONPATH=/app python backend/seed.py"
```

### 5. Open the application

| Interface | URL |
|-----------|-----|
| Dashboard | http://localhost:8501 |
| API docs (Swagger) | http://localhost:8000/docs |
| API health check | http://localhost:8000/health |

**Demo credentials:**

| Username | Password | Role |
|----------|----------|------|
| `test_chw` | `chwpassword` | Community Health Worker |
| `test_patient` | `testpassword` | Patient |

---

## Project structure

```
warrior-blood-app/
├── alembic/              # Database migrations
├── backend/              # FastAPI application
│   ├── auth.py           # JWT, bcrypt, Fernet PHI helpers
│   ├── database.py       # Async SQLAlchemy engine and session
│   ├── hydration.py      # Hydration scoring logic
│   ├── main.py           # API routes
│   ├── ml_predictor.py   # ONNX inference + SHAP explanations
│   ├── models.py         # ORM models
│   ├── pain_analysis.py  # Breakthrough pain detection
│   ├── schemas.py        # Pydantic request / response schemas
│   └── weather.py        # OpenWeatherMap client with 6 h cache
├── dashboard/
│   └── chw_dashboard.py  # Streamlit CHW and Patient dashboards
├── docs/
│   ├── technical_document.md
│   └── user_guide.md
├── ml/
│   ├── features.py       # Feature engineering
│   └── train_model.py    # LightGBM training pipeline
├── models/               # Trained ONNX and LGB model files
├── tests/
│   └── test_mvp.py       # 53 pytest tests
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Architecture

```
Browser
  │
  ├── http://localhost:8501  →  Streamlit Dashboard (CHW / Patient)
  │                                     │
  └── http://localhost:8000  →  FastAPI  ├── MySQL 8.0 (port 3307)
                                         ├── Redis (alert queue)
                                         └── ONNX Runtime (in-process)
```

All patient-identifiable fields (name, phone, email) are Fernet-encrypted before being written to the database. JWT tokens (HS256, 30-minute expiry) carry a `role` claim used to route users to the CHW or Patient dashboard.

---

## ML pipeline

To retrain the VOC predictor from scratch:

```bash
docker compose exec api bash -c "cd /app && PYTHONPATH=/app python ml/train_model.py"
```

The pipeline generates 29 200 synthetic SCD patient-days, applies SMOTE, trains a calibrated LightGBM classifier, and exports both a native `.lgb` file and an ONNX model for runtime inference.

Expected performance:

```
AUROC:       0.808  (target ≥ 0.80)
Brier score: 0.105  (target ≤ 0.10)
```

---

## Running the tests

```bash
docker compose exec api bash -c "cd /app && PYTHONPATH=/app pytest tests/ -v --cov=backend --cov=ml --cov-report=term-missing"
```

Expected: **53 tests passed**.

---

## Documentation

- [User guide](docs/user_guide.md) — step-by-step setup, walkthrough of every page, and troubleshooting
- [Technical document](docs/technical_document.md) — architecture, data models, API reference, ML pipeline, and clinical references

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI 0.111, Python 3.12 |
| ORM | SQLAlchemy 2 (async) |
| Database | MySQL 8.0 (aiomysql driver) |
| Migrations | Alembic |
| ML | LightGBM, ONNX Runtime, SHAP |
| Dashboard | Streamlit |
| Auth | JWT (python-jose), bcrypt (passlib), Fernet (cryptography) |
| Cache | Redis 7 |
| Container | Docker Compose |

---

## Clinical references

- Machado et al. (2024) — LightGBM on SCD prediction
- Brandow et al. (2020) — pain trajectories in SCD
- Smith et al. (2008) — breakthrough pain and hospitalisation risk
- Yallop et al. (2007) — dehydration, AQI and SCD hospitalisation
- Nolan et al. (2008) — temperature and SCD hospitalisation
- Wahl et al. (2018) — interpretability in LMIC health AI
- Lundberg & Lee (2017) — SHAP values for model explanation
- Armstrong (1994) — urine colour hydration scale
- WHO (2005) — oral rehydration guidelines

---

## Licence

MIT — see [LICENSE](LICENSE).
