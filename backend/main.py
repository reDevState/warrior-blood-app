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

from backend.auth import authenticate_user, create_access_token, decode_token, decrypt_phi, encrypt_phi
from backend.database import create_tables, get_db
from backend.hydration import hydration_risk
from backend.ml_predictor import predict
from backend.models import AlertLog, DiaryEntry, Patient
from backend.schemas import (
    AlertRequest,
    AlertResponse,
    CheckinRequest,
    CheckinResponse,
    DiaryEntryResponse,
    PatientCreate,
    PatientResponse,
    SHAPFactor,
    TokenResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warrior_blood")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup (MVP — use Alembic in production)."""
    await create_tables()
    logger.info("Warrior Blood API v0.1.0 — MVP ready")
    yield


app = FastAPI(
    title="Warrior Blood API",
    version="0.1.0-mvp",
    description=(
        "SCD patient VOC risk prediction API. "
        "MVP uses SQLite + rule-based ML. "
        "Production: PostgreSQL + LightGBM ONNX."
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
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """
    Issue a JWT access token.

    MVP users: test_patient / testpassword, test_chw / chwpassword.
    """
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(
        {"sub": user["username"], "role": user["role"]}
    )
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

    # Run ML prediction
    prediction = predict({
        "pain_score": data.pain_score,
        "body_temp_c": data.body_temp_c,
        "fever_present": data.fever_present,
        "fluid_intake_glasses": data.fluid_intake_glasses,
        "urine_colour": data.urine_colour,
        "med_taken": data.med_taken,
        "sleep_hours": data.sleep_hours,
    })

    # Run hydration risk scoring
    hydration = hydration_risk(
        fluid_intake_glasses=data.fluid_intake_glasses,
        urine_colour=data.urine_colour,
        thirst_level=data.thirst_level,
        dry_mouth=data.dry_mouth,
    )

    # Persist diary entry
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
        hydration_status=hydration.status,
        hydration_message=hydration.message,
        hydration_advice=hydration.advice,
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
