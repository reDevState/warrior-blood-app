"""
schemas.py — Pydantic v2 request/response schemas for Warrior Blood API.

All clinical field ranges are enforced here at the API boundary.
Invalid data (pain_score > 10, urine_colour > 8) is rejected before
reaching the database or ML predictor.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------------------------------------------------------------------------
# Patient registration
# ---------------------------------------------------------------------------

class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    phone: Optional[str] = Field(None, max_length=20)
    dob: Optional[date] = None
    diagnosis_type: Optional[str] = Field(None, max_length=64)


class PatientResponse(BaseModel):
    id: str
    name: Optional[str] = None
    diagnosis_type: Optional[str]
    enrolled_at: datetime
    latest_risk_tier: Optional[str] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Daily check-in
# ---------------------------------------------------------------------------

class CheckinRequest(BaseModel):
    patient_id: str
    pain_score: int = Field(..., ge=0, le=10, description="Self-reported pain 0–10")
    body_temp_c: Optional[float] = Field(None, ge=35.0, le=43.0)
    fever_present: bool = False
    fluid_intake_glasses: int = Field(..., ge=0, le=30)
    urine_colour: int = Field(..., ge=1, le=8, description="Armstrong 1994 urine colour scale")
    thirst_level: int = Field(1, ge=1, le=4)
    dry_mouth: bool = False
    med_taken: bool = True
    sleep_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)

    @field_validator("urine_colour")
    @classmethod
    def validate_urine_colour(cls, v: int) -> int:
        """Enforce Armstrong 1994 colour chart range (1–8)."""
        if not 1 <= v <= 8:
            raise ValueError("urine_colour must be 1–8 (Armstrong 1994 scale)")
        return v

    @field_validator("pain_score")
    @classmethod
    def validate_pain_score(cls, v: int) -> int:
        """Enforce NRS pain scale range (0–10)."""
        if not 0 <= v <= 10:
            raise ValueError("pain_score must be 0–10 (NRS scale)")
        return v


class SHAPFactor(BaseModel):
    factor: str
    direction: str  # increasing | decreasing


class CheckinResponse(BaseModel):
    entry_id: str
    patient_id: str
    risk_score: float
    risk_tier: str           # LOW | MODERATE | HIGH
    shap_factors: list[SHAPFactor]
    hydration_status: str    # WELL_HYDRATED | MILD_RISK | MODERATE_RISK | SEVERE_RISK
    hydration_message: str
    hydration_advice: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Diary history
# ---------------------------------------------------------------------------

class DiaryEntryResponse(BaseModel):
    id: str
    entry_date: date
    pain_score: int
    body_temp_c: Optional[float]
    fever_present: bool
    fluid_intake_glasses: int
    urine_colour: int
    med_taken: bool
    risk_score: Optional[float]
    risk_tier: Optional[str]
    hydration_status: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class AlertRequest(BaseModel):
    patient_id: str
    message: str = Field(..., min_length=1, max_length=160)
    channel: str = Field("sms", pattern="^(sms|push)$")


class AlertResponse(BaseModel):
    alert_id: str
    patient_id: str
    channel: str
    message: str
    queued_at: datetime


# ---------------------------------------------------------------------------
# Pain diary
# ---------------------------------------------------------------------------

class PainDiaryRequest(BaseModel):
    patient_id: str
    pain_score: int = Field(..., ge=0, le=10)
    pain_locations: Optional[list[str]] = None
    trigger_cold: bool = False
    trigger_stress: bool = False
    trigger_exercise: bool = False
    trigger_infection: bool = False
    trigger_dehydration: bool = False
    trigger_other: Optional[str] = None
    took_paracetamol: bool = False
    took_ibuprofen: bool = False
    took_opioid: bool = False
    pain_relief_rating: Optional[int] = Field(None, ge=0, le=3)
    notes: Optional[str] = None

    @field_validator("pain_locations")
    @classmethod
    def validate_locations(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        valid = {"CHEST", "BACK", "ABDOMEN", "L_ARM", "R_ARM", "L_LEG", "R_LEG", "HEAD", "OTHER"}
        if v:
            invalid = [x for x in v if x not in valid]
            if invalid:
                raise ValueError(f"Invalid pain locations: {invalid}")
        return v


class PainDiaryResponse(BaseModel):
    entry_id: str
    patient_id: str
    pain_score: int
    pain_locations: Optional[list[str]]
    is_breakthrough: bool
    chest_pain_alert: bool
    pain_slope_3d: Optional[float]
    suggestion: str
    recorded_at: datetime
