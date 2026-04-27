"""
models.py — SQLAlchemy 2.0 ORM models for Warrior Blood.

All PKs are UUID. PHI fields (name, phone) are stored encrypted
via Fernet (see auth.py). DiaryEntry stores the ML risk output alongside
the raw diary inputs so the CHW dashboard can query without re-running inference.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Patient(Base):
    """Registered SCD patient."""

    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name_enc: Mapped[str] = mapped_column(Text, nullable=False)        # Fernet-encrypted
    phone_enc: Mapped[str] = mapped_column(Text, nullable=True)        # Fernet-encrypted
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    diagnosis_type: Mapped[str] = mapped_column(String(64), nullable=True)
    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    diary_entries: Mapped[list[DiaryEntry]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    chw_actions: Mapped[list[CHWAction]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    alert_logs: Mapped[list[AlertLog]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    pain_entries: Mapped[list["PainDiaryEntry"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class DiaryEntry(Base):
    """
    Daily patient symptom check-in.

    Raw inputs are stored alongside the ML risk output so the CHW
    dashboard can display trends without re-running inference.
    """

    __tablename__ = "diary_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id"), nullable=False, index=True
    )
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Clinical inputs
    pain_score: Mapped[int] = mapped_column(Integer, nullable=False)          # 0-10
    body_temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    fever_present: Mapped[bool] = mapped_column(Boolean, default=False)
    fluid_intake_glasses: Mapped[int] = mapped_column(Integer, nullable=False)
    urine_colour: Mapped[int] = mapped_column(Integer, nullable=False)        # 1-8
    med_taken: Mapped[bool] = mapped_column(Boolean, default=True)
    sleep_hours: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Environmental (from OpenWeatherMap — may be None if offline)
    ambient_temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ML output
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_tier: Mapped[str | None] = mapped_column(String(16), nullable=True)  # LOW|MODERATE|HIGH
    hydration_status: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    patient: Mapped[Patient] = relationship(back_populates="diary_entries")


class CHWAction(Base):
    """Record of a CHW triage action on a patient."""

    __tablename__ = "chw_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id"), nullable=False, index=True
    )
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    patient: Mapped[Patient] = relationship(back_populates="chw_actions")


class AlertLog(Base):
    """Log of every SMS or push alert sent to a patient or CHW."""

    __tablename__ = "alert_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id"), nullable=False, index=True
    )
    channel: Mapped[str] = mapped_column(String(16), nullable=False)   # sms | push
    message: Mapped[str] = mapped_column(Text, nullable=False)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    patient: Mapped[Patient] = relationship(back_populates="alert_logs")


class PainDiaryEntry(Base):
    """
    Detailed pain diary entry — one or more per day per patient.
    Captures location, triggers, analgesic use, and breakthrough events.
    Ref: Brandow et al. (2020), Smith et al. (2008)
    """

    __tablename__ = "pain_diary_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id"), nullable=False, index=True
    )
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    pain_score: Mapped[int] = mapped_column(Integer, nullable=False)

    # Body locations stored as comma-separated codes
    # Valid: CHEST BACK ABDOMEN L_ARM R_ARM L_LEG R_LEG HEAD OTHER
    pain_locations: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Trigger factors
    trigger_cold: Mapped[bool] = mapped_column(Boolean, default=False)
    trigger_stress: Mapped[bool] = mapped_column(Boolean, default=False)
    trigger_exercise: Mapped[bool] = mapped_column(Boolean, default=False)
    trigger_infection: Mapped[bool] = mapped_column(Boolean, default=False)
    trigger_dehydration: Mapped[bool] = mapped_column(Boolean, default=False)
    trigger_other: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Analgesic use
    took_paracetamol: Mapped[bool] = mapped_column(Boolean, default=False)
    took_ibuprofen: Mapped[bool] = mapped_column(Boolean, default=False)
    took_opioid: Mapped[bool] = mapped_column(Boolean, default=False)
    pain_relief_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0–3

    # Breakthrough: sudden spike >= 3 points above 7-day mean and score >= 7
    is_breakthrough: Mapped[bool] = mapped_column(Boolean, default=False)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="pain_entries")


class User(Base):
    """Persistent CHW / admin / patient account (replaces in-memory _USERS dict)."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)   # patient | chw | admin
    patient_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("patients.id"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
