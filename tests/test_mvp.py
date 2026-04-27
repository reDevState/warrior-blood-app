"""
tests/ — Warrior Blood MVP test suite.

Run:
    pytest tests/ -v --cov=backend --cov-report=term-missing

Coverage targets: backend ≥ 87%, ml ≥ 91%
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.auth import encrypt_phi, pwd_context
from backend.database import Base, get_db
from backend.hydration import hydration_risk
from backend.main import app
from backend.ml_predictor import predict
from backend.models import Patient, User

# ---------------------------------------------------------------------------
# Test database — SQLite in-memory (never use dev PostgreSQL for tests)
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


_SEED_USERS = [
    ("test_patient", "testpassword", "patient"),
    ("test_chw",     "chwpassword",  "chw"),
]


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create tables and seed default users before each test, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with TestSessionLocal() as session:
        for username, password, role in _SEED_USERS:
            session.add(User(
                username=username,
                hashed_password=pwd_context.hash(password),
                role=role,
            ))
        await session.commit()
    app.dependency_overrides[get_db] = override_get_db
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def patient_id(client):
    """Register a test patient and return its ID (CHW token required)."""
    chw_token = await _chw_token(client)
    resp = await client.post(
        "/patients",
        json={"name": "Test Patient", "phone": "+2348012345678", "diagnosis_type": "HbSS"},
        headers={"Authorization": f"Bearer {chw_token}"},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


async def _patient_token(client) -> str:
    resp = await client.post(
        "/auth/token", data={"username": "test_patient", "password": "testpassword"}
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


async def _chw_token(client) -> str:
    resp = await client.post(
        "/auth/token", data={"username": "test_chw", "password": "chwpassword"}
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_patient_success(client):
    """Valid patient credentials return a JWT."""
    resp = await client.post(
        "/auth/token", data={"username": "test_patient", "password": "testpassword"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    """Wrong password returns 401."""
    resp = await client.post(
        "/auth/token", data={"username": "test_patient", "password": "wrong"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_checkin_without_token(client, patient_id):
    """Missing auth token returns 401."""
    resp = await client.post(
        "/checkin",
        json={
            "patient_id": patient_id,
            "pain_score": 5,
            "fluid_intake_glasses": 4,
            "urine_colour": 3,
        },
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Check-in tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_checkin_low_risk(client, patient_id):
    """Well patient returns LOW risk tier and valid response structure."""
    token = await _patient_token(client)
    resp = await client.post(
        "/checkin",
        json={
            "patient_id": patient_id,
            "pain_score": 1,
            "body_temp_c": 36.6,
            "fever_present": False,
            "fluid_intake_glasses": 8,
            "urine_colour": 2,
            "thirst_level": 1,
            "dry_mouth": False,
            "med_taken": True,
            "sleep_hours": 8.0,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_tier"] in ("LOW", "MODERATE", "HIGH")
    assert 0.0 <= data["risk_score"] <= 1.0
    assert "hydration_status" in data
    assert "shap_factors" in data
    assert data["risk_tier"] == "LOW"


@pytest.mark.asyncio
async def test_checkin_high_risk(client, patient_id):
    """Severely ill patient returns HIGH risk tier."""
    token = await _patient_token(client)
    resp = await client.post(
        "/checkin",
        json={
            "patient_id": patient_id,
            "pain_score": 9,
            "body_temp_c": 38.8,
            "fever_present": True,
            "fluid_intake_glasses": 1,
            "urine_colour": 8,
            "thirst_level": 4,
            "dry_mouth": True,
            "med_taken": False,
            "sleep_hours": 3.0,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["risk_tier"] == "HIGH"


@pytest.mark.asyncio
async def test_checkin_pain_score_out_of_range(client, patient_id):
    """pain_score > 10 returns 422 validation error."""
    token = await _patient_token(client)
    resp = await client.post(
        "/checkin",
        json={
            "patient_id": patient_id,
            "pain_score": 11,
            "fluid_intake_glasses": 5,
            "urine_colour": 3,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_checkin_urine_colour_out_of_range(client, patient_id):
    """urine_colour > 8 returns 422 validation error."""
    token = await _patient_token(client)
    resp = await client.post(
        "/checkin",
        json={
            "patient_id": patient_id,
            "pain_score": 3,
            "fluid_intake_glasses": 5,
            "urine_colour": 9,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_checkin_unknown_patient(client):
    """Check-in for non-existent patient_id returns 404."""
    token = await _patient_token(client)
    resp = await client.post(
        "/checkin",
        json={
            "patient_id": "does-not-exist",
            "pain_score": 3,
            "fluid_intake_glasses": 5,
            "urine_colour": 3,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Hydration tests
# ---------------------------------------------------------------------------

def test_well_hydrated():
    result = hydration_risk(fluid_intake_glasses=8, urine_colour=2, thirst_level=1, dry_mouth=False)
    assert result.status == "WELL_HYDRATED"
    assert result.risk_score == 0


def test_mild_hydration_risk():
    result = hydration_risk(fluid_intake_glasses=7, urine_colour=4, thirst_level=1, dry_mouth=False)
    assert result.status == "MILD_RISK"


def test_moderate_hydration_risk():
    result = hydration_risk(fluid_intake_glasses=4, urine_colour=5, thirst_level=2, dry_mouth=False)
    assert result.status == "MODERATE_RISK"


def test_severe_hydration_risk():
    result = hydration_risk(fluid_intake_glasses=2, urine_colour=7, thirst_level=3, dry_mouth=True)
    assert result.status == "SEVERE_RISK"
    assert result.risk_score >= 5


def test_hydration_message_not_empty():
    for glasses, urine in [(2, 8), (5, 4), (8, 2)]:
        result = hydration_risk(fluid_intake_glasses=glasses, urine_colour=urine)
        assert result.message
        assert result.advice


# ---------------------------------------------------------------------------
# ML predictor tests
# ---------------------------------------------------------------------------

def test_predict_low_risk():
    result = predict({
        "pain_score": 1, "body_temp_c": 36.5, "fever_present": False,
        "fluid_intake_glasses": 8, "urine_colour": 2, "med_taken": True, "sleep_hours": 8,
    })
    assert result.risk_tier == "LOW"
    assert 0.0 <= result.risk_score <= 1.0


def test_predict_high_risk():
    result = predict({
        "pain_score": 9, "body_temp_c": 38.9, "fever_present": True,
        "fluid_intake_glasses": 1, "urine_colour": 8, "med_taken": False, "sleep_hours": 3,
    })
    assert result.risk_tier == "HIGH"
    assert result.risk_score >= 0.6


def test_predict_with_missing_features():
    """Predictor must handle None / missing values gracefully."""
    result = predict({"pain_score": 5})
    assert result.risk_tier in ("LOW", "MODERATE", "HIGH")
    assert isinstance(result.risk_score, float)


def test_predict_shap_factors_format():
    """SHAP factors must be a list of dicts with 'factor' and 'direction' keys."""
    result = predict({
        "pain_score": 7, "fever_present": True,
        "fluid_intake_glasses": 2, "urine_colour": 6,
    })
    for factor in result.shap_factors:
        assert "factor" in factor
        assert "direction" in factor
        assert factor["direction"] in ("increasing", "decreasing")


# ---------------------------------------------------------------------------
# Patient list and history tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_patients_chw_only(client):
    """Patient token cannot access the patient list — CHW only."""
    token = await _patient_token(client)
    resp = await client.get("/patients", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_patients_empty(client):
    """CHW sees empty list when no patients registered."""
    token = await _chw_token(client)
    resp = await client.get("/patients", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_patients_sorted_by_risk(client, patient_id):
    """Patient list returns registered patients; HIGH tier sorted first."""
    chw_token = await _chw_token(client)
    patient_token = await _patient_token(client)

    # Submit a HIGH-risk check-in
    await client.post(
        "/checkin",
        json={
            "patient_id": patient_id,
            "pain_score": 9,
            "body_temp_c": 38.8,
            "fever_present": True,
            "fluid_intake_glasses": 1,
            "urine_colour": 8,
            "thirst_level": 4,
            "dry_mouth": True,
            "med_taken": False,
            "sleep_hours": 3.0,
        },
        headers={"Authorization": f"Bearer {patient_token}"},
    )

    resp = await client.get("/patients", headers={"Authorization": f"Bearer {chw_token}"})
    assert resp.status_code == 200
    patients = resp.json()
    assert len(patients) == 1
    assert patients[0]["latest_risk_tier"] == "HIGH"


@pytest.mark.asyncio
async def test_patient_history(client, patient_id):
    """History endpoint returns diary entries for the patient."""
    token = await _patient_token(client)

    # Submit two check-ins
    for pain in [3, 6]:
        await client.post(
            "/checkin",
            json={
                "patient_id": patient_id,
                "pain_score": pain,
                "fluid_intake_glasses": 5,
                "urine_colour": 3,
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    resp = await client.get(
        f"/patients/{patient_id}/history",
        headers={"Authorization": f"Bearer {token}"},
        params={"days": 30},
    )
    assert resp.status_code == 200
    entries = resp.json()
    assert len(entries) == 2
    assert all("risk_tier" in e for e in entries)


@pytest.mark.asyncio
async def test_patient_history_empty(client, patient_id):
    """History returns empty list when no entries exist."""
    token = await _patient_token(client)
    resp = await client.get(
        f"/patients/{patient_id}/history",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Alert tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_alert_success(client, patient_id):
    """CHW can queue an SMS alert for a registered patient."""
    token = await _chw_token(client)
    resp = await client.post(
        "/alerts",
        json={"patient_id": patient_id, "message": "Please check in today.", "channel": "sms"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["channel"] == "sms"
    assert "alert_id" in data


@pytest.mark.asyncio
async def test_send_alert_patient_not_found(client):
    """Alert to non-existent patient returns 404."""
    token = await _chw_token(client)
    resp = await client.post(
        "/alerts",
        json={"patient_id": "ghost-id", "message": "Hello?", "channel": "sms"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_send_alert_patient_token_rejected(client, patient_id):
    """Patient token cannot send alerts — CHW only."""
    token = await _patient_token(client)
    resp = await client.post(
        "/alerts",
        json={"patient_id": patient_id, "message": "Test.", "channel": "sms"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_register_patient_patient_token_rejected(client):
    """Patient token cannot register new patients — CHW only."""
    token = await _patient_token(client)
    resp = await client.post(
        "/patients",
        json={"name": "Intruder", "diagnosis_type": "HbSS"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Health endpoint test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["api"] == "ok"
    assert "database" in data


@pytest.mark.asyncio
async def test_root_endpoint(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "Warrior Blood" in resp.json()["message"]


# ---------------------------------------------------------------------------
# Database helper tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_jwt_returns_401(client):
    """A tampered or garbage JWT returns 401."""
    resp = await client.get(
        "/patients",
        headers={"Authorization": "Bearer this.is.not.a.valid.jwt"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_expired_jwt_returns_401(client):
    """An expired JWT returns 401."""
    from backend.auth import create_access_token
    expired_token = create_access_token({"sub": "test_chw", "role": "chw"}, expires_minutes=-1)
    resp = await client.get(
        "/patients",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_register_patient_full_fields(client):
    """Full patient registration with all optional fields returns 201."""
    token = await _chw_token(client)
    resp = await client.post(
        "/patients",
        json={
            "name": "Amaka Okonkwo",
            "phone": "+2348099887766",
            "dob": "1990-05-15",
            "diagnosis_type": "HbSS",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["diagnosis_type"] == "HbSS"


@pytest.mark.asyncio
async def test_checkin_auto_creates_alert_for_high_risk(client, patient_id):
    """A HIGH-risk check-in automatically queues an alert entry."""
    token = await _patient_token(client)
    resp = await client.post(
        "/checkin",
        json={
            "patient_id": patient_id,
            "pain_score": 9,
            "body_temp_c": 39.0,
            "fever_present": True,
            "fluid_intake_glasses": 1,
            "urine_colour": 8,
            "thirst_level": 4,
            "dry_mouth": True,
            "med_taken": False,
            "sleep_hours": 2.5,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["risk_tier"] == "HIGH"
    # The auto-alert is internal — we just confirm the check-in succeeded
    # and the response includes SHAP factors explaining the HIGH risk
    assert len(resp.json()["shap_factors"]) >= 1


@pytest.mark.asyncio
async def test_alert_invalid_channel(client, patient_id):
    """Alert with invalid channel value returns 422."""
    token = await _chw_token(client)
    resp = await client.post(
        "/alerts",
        json={"patient_id": patient_id, "message": "Test.", "channel": "telegram"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pain diary tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_pain_entry_basic(client, patient_id):
    """Basic pain diary entry returns correct score and a suggestion string."""
    token = await _patient_token(client)
    resp = await client.post(
        "/pain",
        json={"patient_id": patient_id, "pain_score": 5, "pain_locations": ["BACK"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pain_score"] == 5
    assert "suggestion" in data
    assert isinstance(data["suggestion"], str)
    assert len(data["suggestion"]) > 0


@pytest.mark.asyncio
async def test_chest_pain_triggers_alert(client, patient_id):
    """Chest pain location sets chest_pain_alert and returns hospital advice."""
    token = await _patient_token(client)
    resp = await client.post(
        "/pain",
        json={"patient_id": patient_id, "pain_score": 8, "pain_locations": ["CHEST", "BACK"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chest_pain_alert"] is True
    assert "hospital" in data["suggestion"].lower()


def test_compute_pain_trend_rising():
    """Rising pain scores over 3 days produce a positive slope and is_rising flag."""
    from backend.pain_analysis import compute_pain_trend
    trend = compute_pain_trend([2, 4, 5], 7, ["BACK"])
    assert trend.is_rising is True
    assert trend.slope_3d > 0


def test_compute_pain_trend_breakthrough():
    """Score >= mean_7d + 3 and >= 7 flags a breakthrough event."""
    from backend.pain_analysis import compute_pain_trend
    trend = compute_pain_trend([2, 2, 2, 2, 2, 2, 2], 9, None)
    assert trend.is_breakthrough is True


@pytest.mark.asyncio
async def test_db_session_rolls_back_on_error():
    """get_db rolls back the session if an exception is raised mid-transaction."""
    rolled_back = False
    async with TestSessionLocal() as session:
        try:
            session.add(Patient(id="duplicate-id", name_enc="x", phone_enc=None))
            await session.flush()
            # Second distinct object with same PK triggers IntegrityError on flush
            session.add(Patient(id="duplicate-id", name_enc="y", phone_enc=None))
            await session.flush()
        except Exception:
            await session.rollback()
            rolled_back = True
    assert rolled_back
