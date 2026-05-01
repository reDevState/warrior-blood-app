"""
main.py — Warrior Blood FastAPI application (MVP).

Routes:
    GET  /             → health + version
    GET  /health       → service health check
    POST /auth/token   → issue JWT
    POST /patients     → register patient
    GET  /patients     → list all patients (CHW view, sorted by risk tier)
    POST /checkin      → daily diary entry → risk score + SHAP
    GET  /patients/{id}/history → 30-day diary history
    POST /alerts       → queue an SMS alert (logged; SMS stub in MVP)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import authenticate_user, create_access_token, decode_token, decrypt_phi, encrypt_phi, pwd_context
from backend.database import AsyncSessionLocal, create_tables, get_db
from backend.hydration import hydration_risk
from backend.ml_predictor import predict
from backend.models import AlertLog, DiaryEntry, HydrationEntry, PainDiaryEntry, Patient, User
from backend.pain_analysis import compute_pain_trend
from backend.weather import WeatherData, get_weather
from backend.schemas import (
    AlertRequest,
    AlertResponse,
    CheckinRequest,
    CheckinResponse,
    DiaryEntryResponse,
    HydrationLogRequest,
    HydrationLogResponse,
    PainDiaryRequest,
    PainDiaryResponse,
    PatientCreate,
    PatientResponse,
    SHAPFactor,
    TokenResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warrior_blood")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


_DEFAULT_USERS = [
    {"username": "test_patient", "password": "testpassword", "role": "patient", "patient_id": None},
    {"username": "test_chw",     "password": "chwpassword",  "role": "chw",     "patient_id": None},
]


async def _seed_users() -> None:
    """Create default MVP users if they don't already exist."""
    async with AsyncSessionLocal() as db:
        for u in _DEFAULT_USERS:
            exists = await db.execute(select(User).where(User.username == u["username"]))
            if not exists.scalar_one_or_none():
                db.add(User(
                    username=u["username"],
                    hashed_password=pwd_context.hash(u["password"]),
                    role=u["role"],
                    patient_id=u["patient_id"],
                ))
        await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables and seed default users on startup."""
    await create_tables()
    await _seed_users()
    logger.info("Warrior Blood API v0.1.0 — MVP ready")
    yield


app = FastAPI(
    title="Warrior Blood API",
    version="0.1.0-mvp",
    description=(
        "SCD patient VOC risk prediction API. "
        "MVP uses MySQL + LightGBM ONNX predictor."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict:
    """FastAPI dependency: decode JWT and return the user payload."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        username: str = payload.get("sub", "")
        if not username:
            raise credentials_exc
    except JWTError:
        raise credentials_exc
    return payload


async def require_chw(user: Annotated[dict, Depends(get_current_user)]) -> dict:
    """Dependency: require CHW or admin role."""
    if user.get("role") not in ("chw", "admin"):
        raise HTTPException(status_code=403, detail="CHW access required")
    return user


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
async def root():
    """API root — returns version string."""
    return {"message": "Warrior Blood API v0.1.0-mvp - Hi Renee!", "status": "ok"}


@app.get("/health", tags=["health"])
async def health(db: AsyncSession = Depends(get_db)):
    """Service health check — confirms DB is reachable."""
    try:
        await db.execute(select(Patient).limit(1))
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"
    return {"api": "ok", "database": db_status, "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
):
    """
    Issue a JWT access token.

    MVP users: test_patient / testpassword, test_chw / chwpassword.
    """
    user = await authenticate_user(form.username, form.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(access_token=token)


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

@app.post("/patients", response_model=PatientResponse, status_code=201, tags=["patients"])
async def register_patient(
    data: PatientCreate,
    _: Annotated[dict, Depends(require_chw)],
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new SCD patient (CHW only).

    Name and phone are encrypted with Fernet before DB write — PHI never stored in plaintext.
    """
    patient = Patient(
        name_enc=encrypt_phi(data.name),
        phone_enc=encrypt_phi(data.phone) if data.phone else None,
        dob=data.dob,
        diagnosis_type=data.diagnosis_type,
    )
    db.add(patient)
    await db.flush()
    return PatientResponse(
        id=patient.id,
        name=data.name,
        diagnosis_type=patient.diagnosis_type,
        enrolled_at=patient.enrolled_at,
    )


@app.get("/patients", response_model=list[PatientResponse], tags=["patients"])
async def list_patients(
    _: Annotated[dict, Depends(require_chw)],
    db: AsyncSession = Depends(get_db),
):
    """
    List all patients sorted by latest risk tier (HIGH first).

    CHW-only endpoint. Returns anonymised view (no decrypted PHI).
    """
    result = await db.execute(select(Patient).order_by(Patient.enrolled_at.desc()))
    patients = result.scalars().all()

    out = []
    for p in patients:
        # Get latest diary entry to show current risk tier
        entry_result = await db.execute(
            select(DiaryEntry)
            .where(DiaryEntry.patient_id == p.id)
            .order_by(desc(DiaryEntry.created_at))
            .limit(1)
        )
        latest = entry_result.scalar_one_or_none()
        out.append(
            PatientResponse(
                id=p.id,
                name=decrypt_phi(p.name_enc) if p.name_enc else None,
                diagnosis_type=p.diagnosis_type,
                enrolled_at=p.enrolled_at,
                latest_risk_tier=latest.risk_tier if latest else None,
            )
        )

    # Sort: HIGH first, then MODERATE, then LOW, then None
    tier_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, None: 3}
    out.sort(key=lambda x: tier_order.get(x.latest_risk_tier, 3))
    return out


# ---------------------------------------------------------------------------
# Check-in
# ---------------------------------------------------------------------------

@app.post("/checkin", response_model=CheckinResponse, tags=["checkin"])
async def patient_checkin(
    data: CheckinRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a daily symptom diary entry.

    Runs ML risk prediction and hydration scoring, stores the result,
    and returns risk_tier + top-3 plain-English SHAP factors.

    A HIGH risk result automatically logs an alert entry for CHW review.
    """
    # Validate patient exists
    patient = await db.get(Patient, data.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Fetch weather — degrades gracefully when offline or lat/lon absent
    weather: WeatherData = WeatherData()
    if data.latitude and data.longitude:
        weather = get_weather(data.latitude, data.longitude)

    # Collect weather alerts for patient-facing response
    weather_alerts: list[str] = []
    if weather.cold_stress_alert:
        weather_alerts.append(
            f"Cold stress alert: {weather.ambient_temp_c}°C — keep warm, dress in layers."
        )
    if weather.heat_stress_alert:
        weather_alerts.append(
            f"Heat stress alert: {weather.ambient_temp_c}°C — drink extra fluids, stay in shade."
        )
    if weather.aqi_alert:
        weather_alerts.append(
            "Poor air quality today — avoid outdoor exertion and keep windows closed."
        )

    # Run ML prediction (weather features forwarded for future ONNX model)
    prediction = predict({
        "pain_score": data.pain_score,
        "body_temp_c": data.body_temp_c,
        "fever_present": data.fever_present,
        "fluid_intake_glasses": data.fluid_intake_glasses,
        "urine_colour": data.urine_colour,
        "med_taken": data.med_taken,
        "sleep_hours": data.sleep_hours,
        "ambient_temp_c": weather.ambient_temp_c,
        "humidity_pct": weather.humidity_pct,
        "aqi": weather.aqi,
    })

    # Run hydration risk scoring
    hydration = hydration_risk(
        fluid_intake_glasses=data.fluid_intake_glasses,
        urine_colour=data.urine_colour,
        thirst_level=data.thirst_level,
        dry_mouth=data.dry_mouth,
    )

    # Persist diary entry with weather snapshot
    entry = DiaryEntry(
        patient_id=data.patient_id,
        entry_date=date.today(),
        pain_score=data.pain_score,
        body_temp_c=data.body_temp_c,
        fever_present=data.fever_present,
        fluid_intake_glasses=data.fluid_intake_glasses,
        urine_colour=data.urine_colour,
        med_taken=data.med_taken,
        sleep_hours=data.sleep_hours,
        ambient_temp_c=weather.ambient_temp_c,
        feels_like_c=weather.feels_like_c,
        humidity_pct=weather.humidity_pct,
        aqi=weather.aqi,
        pm25_ugm3=weather.pm25_ugm3,
        cold_stress_alert=weather.cold_stress_alert,
        heat_stress_alert=weather.heat_stress_alert,
        aqi_alert=weather.aqi_alert,
        risk_score=prediction.risk_score,
        risk_tier=prediction.risk_tier,
        hydration_status=hydration.status,
    )
    db.add(entry)
    await db.flush()

    # Auto-log alert if HIGH risk
    if prediction.risk_tier == "HIGH":
        alert = AlertLog(
            patient_id=data.patient_id,
            channel="sms",
            message=(
                f"HIGH VOC risk detected (score={prediction.risk_score:.2f}). "
                f"Top factor: {prediction.shap_factors[0]['factor'] if prediction.shap_factors else 'N/A'}. "
                "CHW review required."
            ),
            delivered=False,
        )
        db.add(alert)
        logger.info("HIGH risk alert queued for patient [REDACTED]")

    return CheckinResponse(
        entry_id=entry.id,
        patient_id=data.patient_id,
        risk_score=prediction.risk_score,
        risk_tier=prediction.risk_tier,
        shap_factors=[SHAPFactor(**f) for f in prediction.shap_factors],
        suggestion=prediction.suggestion,
        hydration_status=hydration.status,
        hydration_message=hydration.message,
        hydration_advice=hydration.advice,
        weather_alerts=weather_alerts,
        created_at=entry.created_at,
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@app.get(
    "/patients/{patient_id}/history",
    response_model=list[DiaryEntryResponse],
    tags=["patients"],
)
async def patient_history(
    patient_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    days: int = 30,
):
    """
    Return the last N days of diary entries for a patient.

    Args:
        patient_id: Patient UUID.
        days: Number of days of history to return (default 30).
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        select(DiaryEntry)
        .where(
            DiaryEntry.patient_id == patient_id,
            DiaryEntry.created_at >= cutoff,
        )
        .order_by(DiaryEntry.entry_date.asc())
    )
    entries = result.scalars().all()
    return [DiaryEntryResponse.model_validate(e) for e in entries]


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@app.post("/alerts", response_model=AlertResponse, tags=["alerts"])
async def send_alert(
    data: AlertRequest,
    _: Annotated[dict, Depends(require_chw)],
    db: AsyncSession = Depends(get_db),
):
    """
    Manually queue an SMS alert for a patient (CHW only).

    MVP: logs to DB and prints to console.
    Production: dispatches via Africa's Talking SDK through Celery.
    """
    patient = await db.get(Patient, data.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    alert = AlertLog(
        patient_id=data.patient_id,
        channel=data.channel,
        message=data.message,
        delivered=False,
    )
    db.add(alert)
    await db.flush()

    # MVP stub: log to console (replace with Celery task in production)
    logger.info(f"[ALERT STUB] {data.channel.upper()} queued — message length: {len(data.message)} chars")

    return AlertResponse(
        alert_id=alert.id,
        patient_id=data.patient_id,
        channel=data.channel,
        message=data.message,
        queued_at=alert.sent_at,
    )


# ---------------------------------------------------------------------------
# Pain diary
# ---------------------------------------------------------------------------

def _pain_suggestion(trend, data: PainDiaryRequest) -> str:
    """Return a plain-English self-care suggestion based on the pain trend."""
    if trend.is_chest_pain:
        return (
            "Chest pain can be a sign of acute chest syndrome — a medical emergency. "
            "Go to your nearest hospital immediately or call your CHW."
        )
    if trend.is_breakthrough:
        return (
            "You are having a breakthrough pain event. "
            "Take your prescribed analgesics now, drink 2 glasses of water, "
            "rest, and contact your CHW."
        )
    if trend.is_rising:
        return (
            "Your pain has been rising over the past 3 days. "
            "Increase fluids to 8+ glasses today, avoid cold and exertion, "
            "and take your medication as prescribed."
        )
    if data.pain_score >= 7:
        return (
            "High pain today. Take your prescribed pain relief, "
            "rest, and keep warm. Contact your CHW if this continues."
        )
    if data.pain_score <= 2:
        return (
            "Pain is well controlled today — great. "
            "Keep up your fluids and medication to maintain this."
        )
    return (
        "Moderate pain today. Drink at least 8 glasses of water, "
        "take your medication, and rest if possible."
    )


@app.post("/pain", response_model=PainDiaryResponse, tags=["pain"])
async def log_pain_entry(
    data: PainDiaryRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Log a detailed pain diary entry.

    Computes pain trend, detects breakthrough events and chest pain,
    and returns a plain-English suggestion.
    Chest pain or a breakthrough event auto-queues a CHW alert.
    """
    patient = await db.get(Patient, data.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    cutoff = datetime.utcnow() - timedelta(days=7)
    result = await db.execute(
        select(PainDiaryEntry)
        .where(
            PainDiaryEntry.patient_id == data.patient_id,
            PainDiaryEntry.recorded_at >= cutoff,
        )
        .order_by(PainDiaryEntry.recorded_at.asc())
    )
    recent_scores = [e.pain_score for e in result.scalars().all()]

    locs = data.pain_locations or []
    trend = compute_pain_trend(recent_scores, data.pain_score, locs)
    suggestion = _pain_suggestion(trend, data)

    entry = PainDiaryEntry(
        patient_id=data.patient_id,
        pain_score=data.pain_score,
        pain_locations=",".join(locs) if locs else None,
        trigger_cold=data.trigger_cold,
        trigger_stress=data.trigger_stress,
        trigger_exercise=data.trigger_exercise,
        trigger_infection=data.trigger_infection,
        trigger_dehydration=data.trigger_dehydration,
        trigger_other=data.trigger_other,
        took_paracetamol=data.took_paracetamol,
        took_ibuprofen=data.took_ibuprofen,
        took_opioid=data.took_opioid,
        pain_relief_rating=data.pain_relief_rating,
        is_breakthrough=trend.is_breakthrough,
        notes=data.notes,
    )
    db.add(entry)
    await db.flush()

    if trend.is_chest_pain or trend.is_breakthrough:
        alert_msg = (
            "URGENT: Chest pain reported"
            if trend.is_chest_pain
            else f"Breakthrough pain event — score {data.pain_score}/10"
        )
        db.add(AlertLog(
            patient_id=data.patient_id,
            channel="sms",
            message=alert_msg,
            delivered=False,
        ))

    return PainDiaryResponse(
        entry_id=entry.id,
        patient_id=data.patient_id,
        pain_score=data.pain_score,
        pain_locations=locs or None,
        is_breakthrough=trend.is_breakthrough,
        chest_pain_alert=trend.is_chest_pain,
        pain_slope_3d=trend.slope_3d,
        suggestion=suggestion,
        recorded_at=entry.recorded_at,
    )


@app.get("/patients/{patient_id}/pain", tags=["pain"])
async def pain_history(
    patient_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    days: int = 30,
):
    """Return last N days of pain diary entries for a patient."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        select(PainDiaryEntry)
        .where(
            PainDiaryEntry.patient_id == patient_id,
            PainDiaryEntry.recorded_at >= cutoff,
        )
        .order_by(PainDiaryEntry.recorded_at.asc())
    )
    return [
        {
            "recorded_at": str(e.recorded_at)[:10],
            "pain_score": e.pain_score,
            "locations": e.pain_locations,
            "is_breakthrough": e.is_breakthrough,
        }
        for e in result.scalars().all()
    ]


# ---------------------------------------------------------------------------
# Hydration diary
# ---------------------------------------------------------------------------

def _hydration_suggestion(status: str, remaining_ml: int, drink_type: str) -> str:
    """Return a plain-English hydration suggestion based on current status."""
    glasses_left = remaining_ml // 250
    if status == "SEVERE_RISK":
        return (
            "You are severely dehydrated. Drink water NOW — at least 2 large glasses. "
            "Contact your CHW immediately if you feel dizzy or cannot drink."
        )
    if status == "MODERATE_RISK":
        return (
            f"You need more fluids. Try to drink {glasses_left} more glasses of "
            "water before bedtime."
        )
    if drink_type in ("COFFEE", "TEA"):
        return "Coffee and tea have a mild diuretic effect. Follow this with a glass of water."
    if status == "WELL_HYDRATED":
        return "Great hydration today! Keep sipping water regularly."
    return f"Keep going — aim for {glasses_left} more glasses of water today."


@app.post("/hydration", response_model=HydrationLogResponse, tags=["hydration"])
async def log_hydration(
    data: HydrationLogRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """
    Log an individual drink to the hydration diary.

    Computes the running daily total, runs hydration risk assessment,
    and returns a personalised hydration suggestion.
    Coffee and tea are stored at 80% of their volume (mild diuretic effect).
    """
    patient = await db.get(Patient, data.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    effective_ml = (
        int(data.drink_volume_ml * 0.8)
        if data.drink_type in ("COFFEE", "TEA")
        else data.drink_volume_ml
    )

    today = datetime.utcnow().date()
    result = await db.execute(
        select(HydrationEntry)
        .where(
            HydrationEntry.patient_id == data.patient_id,
            HydrationEntry.entry_date == today,
        )
    )
    prior_ml = sum(e.drink_volume_ml for e in result.scalars().all())
    daily_total = prior_ml + effective_ml

    glasses = daily_total // 250
    h = hydration_risk(
        fluid_intake_glasses=glasses,
        urine_colour=data.urine_colour or 3,
        thirst_level=data.thirst_level,
        dry_mouth=data.dry_mouth,
    )

    remaining_ml = max(0, 2000 - daily_total)
    suggestion = _hydration_suggestion(h.status, remaining_ml, data.drink_type)

    entry = HydrationEntry(
        patient_id=data.patient_id,
        entry_date=today,
        drink_type=data.drink_type,
        drink_volume_ml=effective_ml,
        daily_total_ml=daily_total,
        urine_colour=data.urine_colour,
        thirst_level=data.thirst_level,
        dry_mouth=data.dry_mouth,
        dizziness=data.dizziness,
        headache=data.headache,
        dark_urine_flag=(data.urine_colour or 0) >= 6,
    )
    db.add(entry)
    await db.flush()

    return HydrationLogResponse(
        entry_id=entry.id,
        patient_id=data.patient_id,
        drink_type=data.drink_type,
        drink_volume_ml=effective_ml,
        daily_total_ml=daily_total,
        daily_total_glasses=round(daily_total / 250, 1),
        hydration_status=h.status,
        hydration_message=h.message,
        hydration_advice=h.advice,
        suggestion=suggestion,
        logged_at=entry.logged_at,
    )
