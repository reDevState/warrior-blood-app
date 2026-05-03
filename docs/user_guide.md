# Warrior Blood — User Guide

**Version:** 0.8.0  
**Last updated:** May 2026

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Getting the application](#2-getting-the-application)
3. [Environment configuration](#3-environment-configuration)
4. [Running with Docker](#4-running-with-docker)
5. [First-time database setup](#5-first-time-database-setup)
6. [Accessing the application](#6-accessing-the-application)
7. [CHW dashboard walkthrough](#7-chw-dashboard-walkthrough)
8. [Patient portal walkthrough](#8-patient-portal-walkthrough)
9. [Connecting with MySQL Workbench](#9-connecting-with-mysql-workbench)
10. [Running the ML training pipeline](#10-running-the-ml-training-pipeline)
11. [Running the test suite](#11-running-the-test-suite)
12. [Stopping and restarting](#12-stopping-and-restarting)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

Install the following before continuing:

| Software | Version | Download |
|----------|---------|----------|
| Git | Any recent | https://git-scm.com |
| Docker Desktop | 4.x or later | https://www.docker.com/products/docker-desktop |
| MySQL Workbench (optional) | 8.0+ | https://dev.mysql.com/downloads/workbench |

Docker Desktop must be running before you execute any `docker` commands. On Windows, ensure WSL 2 is enabled (Docker Desktop will prompt you on first launch).

---

## 2. Getting the Application

### Clone from GitHub

```bash
git clone https://github.com/reDevState/warrior-blood-app.git
cd warrior-blood-app
```

### Check out the stable branch

```bash
git checkout main
```

To use the latest development version:

```bash
git checkout develop
```

### Verify the project structure

After cloning you should see:

```
warrior-blood-app/
├── alembic/              # Database migrations
├── backend/              # FastAPI application
│   ├── auth.py
│   ├── database.py
│   ├── hydration.py
│   ├── main.py
│   ├── ml_predictor.py
│   ├── models.py
│   ├── pain_analysis.py
│   ├── schemas.py
│   └── weather.py
├── dashboard/
│   └── chw_dashboard.py  # Streamlit dashboard (CHW + Patient)
├── docs/                 # This guide and the technical document
├── ml/                   # ML training pipeline
│   ├── features.py
│   └── train_model.py
├── models/               # Trained model files (ONNX + LGB)
├── tests/
│   └── test_mvp.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 3. Environment Configuration

The application reads secrets from a `.env` file in the project root. This file is **not** committed to the repository.

### Create the .env file

In the project root, create a file named `.env` with the following content:

```env
# Database connection (used by the API container — do not change the host)
DATABASE_URL=mysql+aiomysql://warrior:blood@db:3306/warriorblood?charset=utf8mb4

# Fernet key for PHI encryption
# Generate a new one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
FERNET_KEY=your-fernet-key-here

# JWT signing secret — change this in production
SECRET_KEY=warrior-blood-mvp-dev-key-change-in-production-32c

# OpenWeatherMap API key (optional — weather data is skipped if absent)
OPENWEATHERMAP_API_KEY=your-owm-key-here
```

> **Note on the Fernet key:** If you omit `FERNET_KEY`, a session key is generated each time the container starts. Patient PHI (name, phone, email) encrypted in one session cannot be decrypted in the next. For any persistent data, always supply a fixed Fernet key.
>
> To generate a permanent key, run once on your machine:
> ```bash
> python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
> ```
> Copy the output into your `.env` file.

> **OpenWeatherMap API key:** Register for a free account at https://openweathermap.org/api. The free tier allows 1 000 calls/day, which is sufficient for the MVP. If no key is provided, check-ins still work but weather alerts are not generated.

---

## 4. Running with Docker

### Build the images

From the project root:

```bash
docker compose build
```

This builds both the `api` and `dashboard` images from the `Dockerfile`. Expect 2–5 minutes on the first build while Python packages are downloaded.

### Start all services

```bash
docker compose up -d
```

This starts four containers in dependency order:

| Container | Role | Port |
|-----------|------|------|
| `warrior-blood-app-db-1` | MySQL 8.0 database | 3307 (host) → 3306 (container) |
| `warrior-blood-app-redis-1` | Redis (alert queue stub) | 6379 |
| `warrior-blood-app-api-1` | FastAPI backend | 8000 |
| `warrior-blood-app-dashboard-1` | Streamlit dashboard | 8501 |

The API container waits for the database health check to pass before starting.

### Verify the containers are running

```bash
docker compose ps
```

All four containers should show `Up` or `Up (healthy)`.

### Check the API is responding

```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"api": "ok", "database": "ok", "timestamp": "2026-..."}
```

---

## 5. First-Time Database Setup

### Run migrations

On first startup, apply all schema migrations:

```bash
docker compose exec api alembic upgrade head
```

This creates all tables in the `warriorblood` MySQL database. You should see output similar to:

```
INFO  [alembic.runtime.migration] Running upgrade  -> 3a04fbb7ef3b, initial schema
...
INFO  [alembic.runtime.migration] Running upgrade e1f2a3b4c5d6 -> f1a2b3c4d5e6, add email_enc to patients
```

> **Important:** If the tables were already created by the application startup (e.g., on a fresh install), use `alembic stamp head` instead to mark the schema as current without re-running migrations:
> ```bash
> docker compose exec api alembic stamp head
> ```

### Seed default accounts

Create the demo CHW and patient user accounts:

```bash
docker compose exec api bash -c "cd /app && PYTHONPATH=/app python backend/seed.py"
```

This creates two accounts if they do not already exist:

| Username | Password | Role |
|----------|----------|------|
| `test_chw` | `chwpassword` | Community Health Worker |
| `test_patient` | `testpassword` | Patient |

> These are demo credentials for development and testing only. In production, create accounts through a secure admin interface and use strong passwords.

---

## 6. Accessing the Application

| Interface | URL |
|-----------|-----|
| Dashboard (CHW + Patient) | http://localhost:8501 |
| API documentation (Swagger) | http://localhost:8000/docs |
| API documentation (ReDoc) | http://localhost:8000/redoc |
| API health check | http://localhost:8000/health |

---

## 7. CHW Dashboard Walkthrough

Log in at http://localhost:8501 with credentials `test_chw / chwpassword`.

### Patient triage

The triage page lists all registered patients sorted by risk tier (HIGH first).

- **Red (HIGH)** — Patient requires immediate attention
- **Yellow (MODERATE)** — Monitor closely
- **Green (LOW)** — No immediate concern
- **Grey** — No check-in data yet

Click on any patient row to expand their record. Four tabs are available:

| Tab | Content |
|-----|---------|
| Overview | 30-day pain score and VOC risk timeline chart. High-risk days are shaded. |
| Pain history | Pain diary chart with breakthrough event markers (⭐). Last 5 entries listed. |
| Hydration | 7-day bar chart of fluid intake vs 8-glass daily target. |
| Send alert | Send an SMS message to the patient (logged to database; requires Africa's Talking integration for live SMS). |

Click **Refresh** at the top of the triage page to reload all patient data including entries logged by patients themselves.

### Log for patient

Use this page to record data on behalf of a patient who does not have access to the patient portal (e.g., during a clinic visit).

1. Select the patient from the dropdown
2. Choose the appropriate tab: **Check-in & Weather**, **Pain diary**, or **Hydration diary**
3. Complete the form and submit

### Register patient

1. Enter the patient's full name (required)
2. Enter their email address — this allows the patient to log in to the patient portal and link their record without needing to know their UUID
3. Optionally add phone number, date of birth, and diagnosis type
4. Click **Register patient**

After successful registration, the confirmation message shows either:
- The email address the patient should use to link their account (if email was provided)
- The Patient UUID (fallback, if no email was provided)

---

## 8. Patient Portal Walkthrough

Log in at http://localhost:8501 with credentials `test_patient / testpassword`.

### Linking your patient record (first login only)

Before you can use the portal, you must link your account to your clinical record. Two options are available:

**Option A — Link by email (recommended)**
1. Select the **Link by email** tab
2. Enter the email address your CHW registered for you
3. Click **Find my record**

**Option B — Link by Patient ID**
1. Select the **Link by Patient ID** tab
2. Enter the UUID your CHW gave you at registration
3. Click **Link record**

Your record is linked for the duration of the browser session. To unlink, click **Change record** in the sidebar.

### Check-in & Weather

Submit a daily symptom report:

1. **Pain score** — slide to your current pain level (0 = none, 10 = worst imaginable)
2. **Fluid intake** — glasses of fluid consumed today
3. **Urine colour** — select from the Armstrong colour scale (1 = pale straw, 8 = dark brown)
4. **Body temperature** — enter in degrees Celsius
5. **Fever / Medication** — tick as appropriate
6. **Sleep** — select hours (0–12), minutes (00/15/30/45), and AM/PM
7. **Location** — select your city from the searchable dropdown to include weather data

After submitting, you will see:
- Your current **VOC risk score** as a gauge (LOW / MODERATE / HIGH)
- The **top risk factors** from the SHAP explanation
- A personalised **suggestion** in plain English
- **Weather alerts** if your location has heat stress, cold stress, or poor air quality
- Your current **hydration status**

### Pain diary

**Log pain** tab — record a detailed pain entry:
- Pain score slider
- Location multiselect (Chest, Back, Abdomen, Left/Right arm, Left/Right leg, Head, Other)
- Trigger checkboxes (Cold, Stress, Exercise, Infection, Dehydration, Other)
- Medication taken (Paracetamol, Ibuprofen, Opioid) with relief rating
- Free-text notes

After submitting, you will see whether a breakthrough pain event was detected and receive a personalised suggestion.

**My pain history** tab — view a chart of your pain score over the selected period, with breakthrough events marked.

### Hydration diary

**Log a drink** tab — record each drink you consume:
- Select drink type (Water, Juice, Milk, Tea, Coffee, Soda, Other)
- Enter volume in millilitres
- Optionally record urine colour, thirst level, and symptoms (dry mouth, dizziness, headache)

After submitting, a gauge shows your total fluid intake for the day vs the WHO 2 000 ml daily target.

**Hydration guide** tab — view the Armstrong urine colour chart and WHO guidelines specific to SCD patients.

### My history

View a combined 30-day (or custom period) timeline of all diary entries and pain records side by side.

---

## 9. Connecting with MySQL Workbench

The MySQL database is accessible from your host machine on port **3307** (mapped from the container's internal port 3306).

### Connection settings

| Setting | Value |
|---------|-------|
| Connection method | Standard TCP/IP |
| Hostname | 127.0.0.1 |
| Port | 3307 |
| Username | warrior |
| Password | blood |
| Default schema | warriorblood |

### Steps

1. Open MySQL Workbench
2. Click the **+** button next to "MySQL Connections"
3. Enter the settings above
4. Click **Test Connection** — you should see "Successfully made the MySQL connection"
5. Click **OK** then open the connection

### Key tables to inspect

```sql
USE warriorblood;

SHOW TABLES;

-- View registered patients (PHI columns are Fernet-encrypted ciphertext)
SELECT id, diagnosis_type, enrolled_at FROM patients;

-- View recent diary entries
SELECT patient_id, entry_date, pain_score, risk_tier, hydration_status
FROM diary_entries
ORDER BY created_at DESC
LIMIT 20;

-- View pain diary entries
SELECT patient_id, pain_score, is_breakthrough, chest_pain_alert, recorded_at
FROM pain_diary_entries
ORDER BY recorded_at DESC
LIMIT 20;
```

> **Note on encrypted fields:** The `name_enc`, `phone_enc`, and `email_enc` columns store Fernet ciphertext. The values are only readable through the application layer using the `FERNET_KEY` from your `.env` file.

---

## 10. Running the ML Training Pipeline

The trained model files (`models/voc_model.onnx`, `models/voc_model.lgb`) are included in the repository. To retrain from scratch:

```bash
docker compose exec api bash -c "cd /app && PYTHONPATH=/app python ml/train_model.py"
```

The pipeline will:
1. Generate 29 200 synthetic SCD patient-days
2. Split temporally (train before 2024-10-01, test after)
3. Apply SMOTE to balance the training set
4. Train a LightGBM classifier and calibrate it
5. Compute SHAP feature importance on the test set
6. Print AUROC, Brier score, and sensitivity metrics
7. Save `models/voc_model.lgb` (native LightGBM)
8. Save `models/voc_model.onnx` (ONNX for fast inference)
9. Save `ml/outputs/metrics.json` and `ml/outputs/shap_importance.json`

Expected output (approx.):
```
Generated 29,200 patient-days | VOC rate: 17.7%
After SMOTE — train samples: 35660, positive rate: 50.0%
Top SHAP features:
  pain_score: 0.7505
  pain_slope_3d: 0.7026
==================================================
AUROC:       0.808  (target ≥ 0.80)
Brier score: 0.105  (target ≤ 0.10)
==================================================
Saved: models/voc_model.lgb
Saved: models/voc_model.onnx via onnxmltools (895 KB)
```

---

## 11. Running the Test Suite

```bash
docker compose exec api bash -c "cd /app && PYTHONPATH=/app pytest tests/ -v --cov=backend --cov=ml --cov-report=term-missing"
```

Expected result: **53 tests passed**.

To run a single test:
```bash
docker compose exec api bash -c "cd /app && PYTHONPATH=/app pytest tests/test_mvp.py::test_checkin_high_risk -v"
```

---

## 12. Stopping and Restarting

### Stop all containers (keep data)

```bash
docker compose down
```

### Stop and delete all data (full reset)

```bash
docker compose down -v
```

> **Warning:** The `-v` flag deletes the `mysql_data` volume. All database content will be permanently lost. You will need to re-run migrations and seed after a full reset.

### Restart after stopping

```bash
docker compose up -d
```

No migration or seed step is needed unless you used `down -v`.

### Rebuild after code changes

```bash
docker compose build
docker compose up -d
```

If you made database schema changes, apply the new migration:
```bash
docker compose exec api alembic upgrade head
```

---

## 13. Troubleshooting

### Docker containers fail to start

Check the logs:
```bash
docker compose logs api
docker compose logs db
```

Common causes:
- Port 8000, 8501, or 3307 already in use — stop the conflicting process or change the port mapping in `docker-compose.yml`
- Missing `.env` file — create it as described in Section 3
- Docker Desktop not running — start it from the system tray

### "Table already exists" on alembic upgrade

The application creates tables on startup. Use `stamp` instead:
```bash
docker compose exec api alembic stamp head
```

### Patient login — "No record found for that email"

The patient email was either not registered by the CHW, or the FERNET_KEY changed between the registration session and the current session. Ensure the same `FERNET_KEY` is in `.env` and the container was restarted after the key was set.

### Weather alerts not appearing

- Ensure `OPENWEATHERMAP_API_KEY` is set in `.env`
- Restart the API container after adding the key: `docker compose restart api`
- Verify the key is valid at https://openweathermap.org/api

### Test suite fails

Ensure you are running tests inside the container (not on your host machine, which may have a different Python environment):
```bash
docker compose exec api bash -c "cd /app && PYTHONPATH=/app pytest tests/ -v"
```

### MySQL Workbench "Can't connect to MySQL server"

- Verify the database container is running: `docker compose ps`
- Use host `127.0.0.1` (not `localhost`) — some systems route `localhost` to IPv6
- Use port `3307` (not 3306)
