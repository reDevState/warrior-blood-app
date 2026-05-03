# Warrior Blood — Technical Document

**Project:** MSc Software Engineering Dissertation  
**Author:** Renee Tucker  
**Institution:** University of Greater Manchester  
**Supervisor:** Aamir Abbas  
**Academic Year:** 2025–26  
**Version:** 0.8.0

---

## 1. Overview and Purpose

Warrior Blood is a clinical decision-support system for monitoring patients living with Sickle Cell Disease (SCD). The system enables Community Health Workers (CHWs) to track patient risk in real time and allows patients to self-report symptoms through a dedicated portal.

### 1.1 Clinical Context

Sickle Cell Disease is an inherited blood disorder characterised by abnormal haemoglobin that causes red blood cells to sickle under physiological stress. The most acute complication is a Vaso-Occlusive Crisis (VOC), marked by severe pain and requiring hospital admission. VOC accounts for the majority of SCD-related emergency visits globally and disproportionately affects populations in sub-Saharan Africa, the Caribbean, and diaspora communities in Europe and North America (Yallop et al., 2007).

VOC is significantly associated with:
- Dehydration (OR ≈ 2.1; Yallop et al., 2007)
- Ambient temperature extremes — both heat (> 35 °C) and cold (< 15 °C) (Nolan et al., 2008)
- Poor air quality (AQI ≥ 3; Yallop et al., 2007)
- Medication non-adherence and breakthrough pain (Brandow et al., 2020)

The Warrior Blood system addresses these risk factors through daily diary logging, environmental monitoring, and a machine learning predictor trained on these clinical features.

### 1.2 System Goals

- Provide CHWs with a real-time risk triage dashboard showing which patients need immediate attention
- Allow patients to self-report symptoms and receive personalised guidance
- Predict 72-hour VOC risk using a LightGBM model with SHAP-based explanations
- Flag weather and air quality conditions that increase VOC risk
- Maintain patient data privacy through encryption of all personally identifiable information (PHI)

---

## 2. Architecture

The system follows a three-tier architecture:

```
┌─────────────────────────────────────────────────────────┐
│                   Presentation Layer                     │
│         Streamlit Dashboard (port 8501)                  │
│   CHW Triage View  │  Patient Self-Service Portal        │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP (REST)
┌──────────────────────────▼──────────────────────────────┐
│                   Application Layer                      │
│              FastAPI Backend (port 8000)                 │
│  Auth │ Patients │ Check-in │ Pain │ Hydration │ Alerts  │
│         LightGBM ONNX Predictor │ SHAP Explainer         │
│         Weather Client (OpenWeatherMap)                   │
└──────────────────────────┬──────────────────────────────┘
                           │ SQLAlchemy async ORM
┌──────────────────────────▼──────────────────────────────┐
│                     Data Layer                           │
│              MySQL 8.0 (port 3306 / 3307)               │
│   patients │ diary_entries │ pain_diary_entries           │
│   hydration_entries │ alert_log │ chw_actions │ users     │
└─────────────────────────────────────────────────────────┘
```

All services are containerised with Docker and orchestrated via Docker Compose. A Redis container is included for future Celery task-queue integration (alert dispatching).

---

## 3. Technology Stack

### 3.1 Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| Web framework | FastAPI | 0.111.0 |
| ASGI server | Uvicorn | 0.30.1 |
| ORM | SQLAlchemy (async) | 2.0.36 |
| Database | MySQL 8.0 | 8.0 |
| Async MySQL driver | aiomysql | 0.2.0 |
| Sync MySQL driver (Alembic) | PyMySQL | 1.1.1 |
| Schema migrations | Alembic | 1.13.1 |
| Data validation | Pydantic v2 | 2.10.6 |
| Authentication | python-jose (HS256 JWT) | 3.3.0 |
| Password hashing | passlib + bcrypt | 1.7.4 / 4.1.3 |
| PHI encryption | cryptography (Fernet) | 42.0.7 |
| Async SQLite (tests) | aiosqlite | 0.20.0 |

### 3.2 Machine Learning

| Component | Technology | Version |
|-----------|-----------|---------|
| Classifier | LightGBM | 4.3.0 |
| Class balancing | imbalanced-learn (SMOTE) | 0.12.3 |
| Calibration | scikit-learn CalibratedClassifierCV | 1.5.0 |
| Explainability | SHAP TreeExplainer | 0.45.0 |
| Model export | ONNX (via onnxmltools) | 1.16.1 |
| Inference | ONNX Runtime | 1.18.0 |
| Numerics | NumPy / Pandas | 1.26.4 / 2.2.2 |

### 3.3 Dashboard

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Streamlit | 1.35.0 |
| Charts | Plotly | 5.22.0 |
| HTTP client | requests | 2.32.2 |

### 3.4 Infrastructure

| Component | Technology |
|-----------|-----------|
| Containerisation | Docker (python:3.12-slim base) |
| Orchestration | Docker Compose |
| Message broker (stub) | Redis 7 Alpine |
| External weather API | OpenWeatherMap (current weather + air pollution) |

### 3.5 Testing

| Component | Technology | Version |
|-----------|-----------|---------|
| Test runner | pytest | 8.2.1 |
| Async test support | pytest-asyncio | 0.23.7 |
| HTTP test client | httpx / ASGI transport | 0.27.0 |
| Coverage | coverage + pytest-cov | 7.3.4 / 5.0.0 |
| Test database | SQLite in-memory | — |

---

## 4. Data Models

### 4.1 Entity Relationship Overview

```
users ──────────────────────────────── (auth only, no FK to patients in MVP)

patients ─┬─── diary_entries
          ├─── pain_diary_entries
          ├─── hydration_entries
          ├─── alert_log
          └─── chw_actions
```

### 4.2 Patient

| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | UUID primary key |
| name_enc | TEXT | Fernet-encrypted full name |
| phone_enc | TEXT | Fernet-encrypted phone number |
| email_enc | TEXT | Fernet-encrypted email address |
| dob | DATE | Date of birth |
| diagnosis_type | VARCHAR(64) | HbSS / HbSC / HbS/β-thal / Other |
| enrolled_at | DATETIME | Registration timestamp |

### 4.3 DiaryEntry (daily check-in)

| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | UUID PK |
| patient_id | VARCHAR(36) | FK → patients |
| entry_date | DATE | Calendar date of check-in |
| pain_score | INT | 0–10 |
| body_temp_c | FLOAT | |
| fever_present | BOOL | |
| fluid_intake_glasses | INT | Glasses of fluid |
| urine_colour | INT | Armstrong 1994 scale, 1–8 |
| med_taken | BOOL | |
| sleep_hours | FLOAT | |
| ambient_temp_c | FLOAT | From OpenWeatherMap |
| humidity_pct | FLOAT | |
| aqi | INT | Air Quality Index 1–5 |
| risk_score | FLOAT | LightGBM predicted probability |
| risk_tier | VARCHAR(16) | LOW / MODERATE / HIGH |
| hydration_status | VARCHAR(32) | WELL_HYDRATED / MILD_RISK / … |
| created_at | DATETIME | |

### 4.4 PainDiaryEntry

| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | UUID PK |
| patient_id | VARCHAR(36) | FK → patients |
| pain_score | INT | 0–10 |
| pain_locations | TEXT | JSON-encoded list: CHEST/BACK/ABDOMEN/L_ARM/R_ARM/L_LEG/R_LEG/HEAD/OTHER |
| trigger_cold / stress / exercise / infection / dehydration | BOOL | |
| trigger_other | TEXT | Free text |
| took_paracetamol / ibuprofen / opioid | BOOL | |
| pain_relief_rating | INT | 0–3 |
| is_breakthrough | BOOL | Computed: score ≥ 7 for first time in 24 h |
| chest_pain_alert | BOOL | Auto-queues CHW alert |
| pain_slope_3d | FLOAT | Linear regression slope over last 3 entries |
| notes | TEXT | |
| recorded_at | DATETIME | |

### 4.5 HydrationEntry

| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | UUID PK |
| patient_id | VARCHAR(36) | FK → patients |
| drink_type | VARCHAR(16) | WATER/JUICE/MILK/TEA/COFFEE/SODA/OTHER |
| drink_volume_ml | INT | 50–2 000 |
| effective_volume_ml | INT | 80% for tea/coffee (diuretic adjustment) |
| daily_total_ml | INT | Running daily total |
| urine_colour | INT | 1–8 |
| thirst_level | INT | 1–4 |
| dry_mouth / dizziness / headache | BOOL | |
| hydration_status | VARCHAR(32) | |
| logged_at | DATETIME | |

### 4.6 User (authentication)

| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | UUID PK |
| username | VARCHAR(64) | Unique, indexed |
| hashed_password | TEXT | bcrypt |
| role | VARCHAR(16) | chw / admin / patient |
| is_active | BOOL | |
| patient_id | VARCHAR(36) | Nullable FK — links user to patient record |

---

## 5. Authentication and Security

### 5.1 JWT Authentication

The API uses HS256 JSON Web Tokens issued at `/auth/token`. Tokens carry `{"sub": username, "role": role}` and expire after 30 minutes. The `require_chw` FastAPI dependency enforces CHW/admin-only access on sensitive routes. Patient tokens can access diary logging and weather check-in endpoints.

### 5.2 PHI Encryption

All personally identifiable fields — name, phone number, and email address — are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) before any database write. Decryption occurs only at the application layer. The Fernet key is loaded from the `FERNET_KEY` environment variable; if absent, a session-scoped key is generated (data does not survive container restarts in that case).

The email field supports patient self-service record linking: the backend decrypts all patient emails at query time to find a match by email address. This is acceptable at MVP scale.

### 5.3 Password Storage

Passwords are hashed with bcrypt via `passlib.CryptContext`. Plaintext passwords are never logged or returned by any endpoint.

---

## 6. Machine Learning Pipeline

### 6.1 Feature Set

The predictor uses 16 features derived from the patient's diary entry and computed history:

| Feature | Description | Source |
|---------|-------------|--------|
| pain_score | Self-reported pain (0–10) | Diary |
| pain_slope_3d | Linear slope of pain over last 3 days | Computed |
| pain_max_7d | Maximum pain in last 7 days | Computed |
| body_temp_c | Body temperature | Diary |
| fever_present | Binary fever flag | Diary |
| fluid_intake_glasses | Fluid consumed | Diary |
| urine_colour | Armstrong 1994 scale (1–8) | Diary |
| hydration_risk_score | Composite 0–7 hydration proxy | Computed |
| med_taken | Medication adherence | Diary |
| med_adherence_7d | 7-day medication adherence rate | Computed |
| sleep_hours | Sleep duration | Diary |
| ambient_temp_c | Outdoor temperature | OpenWeatherMap |
| humidity_pct | Relative humidity | OpenWeatherMap |
| aqi | Air quality index (1–5) | OpenWeatherMap |
| prior_voc_30d | Any VOC in the preceding 30 days | Computed |
| days_since_last_voc | Days since most recent VOC event | Computed |

### 6.2 Training Pipeline

1. **Data generation** — Synthetic SCD patient diary dataset (80 patients × 365 days = 29 200 records) with a clinical VOC rate of ~18% generated using logistic functions parameterised from Yallop et al. (2007) and Machado et al. (2024).

2. **Temporal split** — Training set: entries before 2024-10-01. Test set: entries from 2024-10-01 onwards. This prevents data leakage from future-to-past and mirrors real-world deployment (Machado et al., 2024).

3. **SMOTE** — Synthetic Minority Over-sampling Technique applied to the training split only, balancing the positive class to 50%. `scale_pos_weight` is set to 1 when SMOTE is active.

4. **LightGBM** — Gradient boosted tree classifier (`n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`).

5. **Calibration** — Isotonic regression calibration (`CalibratedClassifierCV`, `cv="prefit"`) fitted on the original training distribution (not the SMOTE-balanced set) to produce well-calibrated probability outputs.

6. **SHAP** — TreeExplainer generates feature attributions on the test set. Mean absolute SHAP values are exported to `ml/outputs/shap_importance.json` and used to explain individual predictions in plain English.

7. **ONNX export** — The calibrated model is exported to ONNX format via `onnxmltools` for fast, dependency-light inference at prediction time.

### 6.3 Model Performance

| Metric | Value | Target |
|--------|-------|--------|
| AUROC | 0.808 | ≥ 0.80 |
| Brier score | 0.105 | ≤ 0.10 |
| Sensitivity at 0.5 threshold | 0.325 | — |

### 6.4 Top SHAP Features

From the test set (mean |SHAP|):

1. pain_score — 0.7505
2. pain_slope_3d — 0.7026
3. urine_colour — 0.4968
4. fever_present — 0.4791
5. ambient_temp_c — 0.4065

### 6.5 Risk Tiers

| Tier | Risk score threshold |
|------|---------------------|
| LOW | < 0.30 |
| MODERATE | 0.30–0.59 |
| HIGH | ≥ 0.60 |

HIGH risk automatically creates an alert log entry for CHW review.

---

## 7. API Endpoints

All endpoints require a Bearer token unless noted.

| Method | Path | Role | Description |
|--------|------|------|-------------|
| GET | `/` | Any | Version string |
| GET | `/health` | Any | DB connectivity check |
| POST | `/auth/token` | — | Issue JWT |
| POST | `/patients` | CHW | Register patient |
| GET | `/patients` | CHW | List patients (risk-sorted) |
| GET | `/patients/lookup?email=` | Any | Find patient ID by email |
| GET | `/patients/{id}/history` | Any | 30-day diary history |
| GET | `/patients/{id}/pain` | Any | Pain diary history |
| POST | `/checkin` | Any | Daily check-in + VOC prediction |
| POST | `/pain` | Any | Pain diary entry |
| POST | `/hydration` | Any | Hydration log entry |
| POST | `/alerts` | CHW | Send SMS alert |

Full interactive documentation: `http://localhost:8000/docs`

---

## 8. Weather Integration

Weather data is fetched from OpenWeatherMap at check-in time if the patient provides GPS coordinates. Results are cached per coordinate (rounded to 2 decimal places) for 6 hours to remain within the free-tier API limit (1 000 calls/day).

Clinical risk flags applied:
- **Heat stress** — ambient temperature > 35 °C (Nolan et al., 2008)
- **Cold stress** — ambient temperature < 15 °C (Nolan et al., 2008)
- **AQI alert** — Air Quality Index ≥ 3 (Yallop et al., 2007)

The dashboard includes a searchable city dropdown covering 65 cities across the primary SCD burden regions.

---

## 9. Hydration Assessment

The hydration module implements WHO oral rehydration guidelines (WHO, 2005) and the Armstrong (1994) urine colour scale:

| Status | Criteria |
|--------|----------|
| WELL_HYDRATED | ≥ 2 000 ml/day AND urine ≤ 4 |
| MILD_RISK | 1 500–1 999 ml/day OR urine 5–6 |
| MODERATE_RISK | 1 000–1 499 ml/day OR urine 6–7 |
| SEVERE_RISK | < 1 000 ml/day OR urine ≥ 7 |

Tea and coffee are recorded at 80% of their stated volume to account for mild diuretic effects. Daily totals reset at midnight UTC.

---

## 10. Database Migrations

Schema changes are managed with Alembic. The migration chain is:

```
3a04fbb7ef3b → initial schema
9fa4de292d7f → user model and core tables
b6f2810500ba → user model and core tables (revised)
c1a2b3d4e5f6 → pain_diary_entries table
d1e2f3a4b5c6 → hydration_entries table
e1f2a3b4c5d6 → weather fields on diary_entries
f1a2b3c4d5e6 → email_enc on patients
```

Every migration includes a `downgrade()` function that precisely reverses the change.

---

## 11. Testing

The test suite (`tests/test_mvp.py`) contains 53 tests covering authentication, check-in, pain diary, hydration, alerts, weather, and ML predictor paths.

**Key testing conventions:**
- Test database: SQLite in-memory (`sqlite+aiosqlite:///:memory:`) — never the dev MySQL container
- All DB access goes through `app.dependency_overrides[get_db]`
- Real bcrypt password verification via seeded `_SEED_USERS`
- External services (weather, SMS) return offline/stub values when API keys are absent
- No mocking of `authenticate_user`

Run the test suite:
```bash
docker compose exec api bash -c "cd /app && PYTHONPATH=/app pytest tests/ -v --cov=backend --cov=ml"
```

---

## 12. Clinical References

- Machado, R. F. et al. (2024). LightGBM-based prediction of sickle cell disease vaso-occlusive crisis.
- Brandow, A. M. et al. (2020). Pain trajectories and hospitalisation in sickle cell disease. *Blood*.
- Smith, W. R. et al. (2008). Breakthrough pain and hospitalisation rates in sickle cell disease. *Journal of Pain*.
- Yallop, D. et al. (2007). The associations between air quality and the number of hospital admissions for acute pain and sickle cell disease in an urban environment. *British Journal of Haematology*, 139(5), 789–796.
- Nolan, V. G. et al. (2008). Association of single nucleotide polymorphisms in NRF2 with hospitalisation for infection in sickle cell anaemia. *British Journal of Haematology*, 143(4), 589–592.
- Wahl, B. et al. (2018). Artificial intelligence (AI) and global health: how can AI contribute to health in resource-poor settings? *BMJ Global Health*, 3(4).
- Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Armstrong, L. E. (1994). Urinary indices of hydration status. *International Journal of Sport Nutrition*, 4(3), 265–279.
- World Health Organization. (2005). *The treatment of diarrhoea: A manual for physicians and other senior health workers*. WHO.
