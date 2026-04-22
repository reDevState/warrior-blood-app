"""
hydration.py — Rule-based hydration risk scoring for Warrior Blood.

Thresholds based on WHO oral rehydration guidelines and
Yallop et al. (2007) dehydration + SCD hospitalisation study.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HydrationResult:
    """Result of hydration risk assessment."""
    status: str          # WELL_HYDRATED | MILD_RISK | MODERATE_RISK | SEVERE_RISK
    risk_score: int      # 0-7 composite score
    message: str         # Plain-English message for CHW display
    advice: str          # Action recommendation


def hydration_risk(
    fluid_intake_glasses: int,
    urine_colour: int,
    thirst_level: int = 1,
    dry_mouth: bool = False,
) -> HydrationResult:
    """
    Classify hydration status from daily diary inputs.

    Thresholds follow WHO oral rehydration guidelines (WHO, 2005) and
    Yallop et al. (2007) SCD-specific dehydration risk factors.

    Args:
        fluid_intake_glasses: Number of glasses of fluid consumed today (0–20).
        urine_colour: Self-reported urine colour on 1–8 scale
                      (1=pale straw, 8=dark brown). Armstrong 1994 scale.
        thirst_level: Self-reported thirst (1=none, 2=mild, 3=moderate, 4=severe).
        dry_mouth: Whether patient reports dry/sticky mouth.

    Returns:
        HydrationResult with status, composite risk_score, message, and advice.

    Example (SEVERE_RISK — classic dehydrated SCD patient):
        >>> hydration_risk(fluid_intake_glasses=2, urine_colour=7,
        ...                thirst_level=3, dry_mouth=True)
        HydrationResult(status='SEVERE_RISK', risk_score=7, ...)
    """
    score = 0

    # Fluid intake contribution (WHO recommends 8+ glasses/day for SCD patients)
    if fluid_intake_glasses < 4:
        score += 3
    elif fluid_intake_glasses < 6:
        score += 2
    elif fluid_intake_glasses < 8:
        score += 1

    # Urine colour contribution (Armstrong 1994 colour chart)
    if urine_colour >= 7:
        score += 3
    elif urine_colour >= 5:
        score += 2
    elif urine_colour >= 4:
        score += 1

    # Thirst and dry mouth (subjective but clinically significant for SCD)
    if thirst_level >= 3:
        score += 1
    if dry_mouth:
        score += 1

    # Classify
    if score >= 5:
        return HydrationResult(
            status="SEVERE_RISK",
            risk_score=score,
            message="Severely dehydrated — urgent action needed",
            advice="Drink at least 2 glasses of water immediately. Contact your CHW if pain worsens.",
        )
    elif score >= 3:
        return HydrationResult(
            status="MODERATE_RISK",
            risk_score=score,
            message="Moderately dehydrated",
            advice="Drink 2–3 more glasses of water in the next hour. Avoid exertion.",
        )
    elif score >= 1:
        return HydrationResult(
            status="MILD_RISK",
            risk_score=score,
            message="Mildly dehydrated",
            advice="Drink an extra glass of water now. Try to reach 8 glasses today.",
        )
    else:
        return HydrationResult(
            status="WELL_HYDRATED",
            risk_score=score,
            message="Well hydrated",
            advice="Keep it up! Aim for 8+ glasses throughout the day.",
        )
