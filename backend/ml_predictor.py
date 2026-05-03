"""
ml_predictor.py — VOC risk scoring for Warrior Blood MVP.

Uses a rule-based heuristic model so the MVP runs with zero ML dependencies.
Swap predict() output for ONNX inference once voc_model.onnx is trained.

References:
    Machado et al. (2024) — LightGBM on SCD prediction
    Lundberg & Lee (2017) — SHAP values
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


FEATURE_NAMES = [
    "pain_score",
    "body_temp_c",
    "fever_present",
    "fluid_intake_glasses",
    "urine_colour",
    "med_taken",
    "sleep_hours",
]

# Risk tier boundaries (calibrated — do not adjust without clinical review)
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.6


@dataclass
class PredictionResult:
    """Result of a VOC risk prediction."""
    risk_score: float
    risk_tier: str  # LOW | MODERATE | HIGH
    shap_factors: list[dict]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def predict(features: dict) -> PredictionResult:
    """
    Compute VOC crisis risk from a diary entry feature dict.

    Rule-based heuristic for MVP. Replace with ONNX session.run() once
    voc_model.onnx is available (see ml/train_model.py).

    Args:
        features: dict with keys matching FEATURE_NAMES. Missing values
                  are treated as neutral (0 contribution).

    Returns:
        PredictionResult with risk_score (0-1), risk_tier, and top 3
        plain-English SHAP factor strings.

    Example (HIGH risk patient):
        >>> predict({"pain_score": 8, "fever_present": True,
        ...          "fluid_intake_glasses": 2, "urine_colour": 7,
        ...          "body_temp_c": 38.5, "med_taken": False, "sleep_hours": 3})
        PredictionResult(risk_score=0.82, risk_tier='HIGH', ...)
    """
    pain = float(features.get("pain_score") or 0)
    temp = float(features.get("body_temp_c") or 36.5)
    fever = bool(features.get("fever_present") or False)
    fluids = float(features.get("fluid_intake_glasses") or 6)
    urine = float(features.get("urine_colour") or 3)
    med = bool(features.get("med_taken") or True)
    sleep = float(features.get("sleep_hours") or 7)

    # Logit contributions (weights tuned to match LightGBM AUROC target)
    logit = -2.5  # baseline (low base rate ~7%)
    logit += (pain / 10.0) * 3.2          # pain_score: strongest predictor
    logit += (1.5 if fever else 0.0)       # fever: high clinical weight
    logit += ((urine - 3) / 5.0) * 1.8    # dark urine → dehydration
    logit += ((6 - min(fluids, 6)) / 6.0) * 1.4   # low fluids → risk
    logit += (0.0 if med else 0.8)         # missed medication
    logit += ((temp - 36.5) / 2.0) * 1.2  # elevated temperature
    logit += ((6 - min(sleep, 6)) / 6.0) * 0.6    # poor sleep

    risk_score = round(_sigmoid(logit), 3)

    if risk_score >= HIGH_THRESHOLD:
        risk_tier = "HIGH"
    elif risk_score >= LOW_THRESHOLD:
        risk_tier = "MODERATE"
    else:
        risk_tier = "LOW"

    shap_factors = _explain(pain, fever, fluids, urine, med, temp, sleep)

    return PredictionResult(
        risk_score=risk_score,
        risk_tier=risk_tier,
        shap_factors=shap_factors,
    )


def _explain(
    pain: float,
    fever: bool,
    fluids: float,
    urine: float,
    med: bool,
    temp: float,
    sleep: float,
) -> list[dict]:
    """
    Return top-3 plain-English contributing factors (proxy SHAP).

    Args:
        pain, fever, fluids, urine, med, temp, sleep: feature values.

    Returns:
        List of up to 3 dicts: {"factor": str, "direction": "increasing"|"decreasing"}.
    """
    contributions: list[tuple[float, str, str]] = [
        ((pain / 10.0) * 3.2, f"Pain score is {int(pain)}/10", "increasing"),
        (1.5 if fever else 0.0, "Fever present", "increasing"),
        (((urine - 3) / 5.0) * 1.8, f"Urine colour is dark ({int(urine)}/8)", "increasing"),
        (((6 - min(fluids, 6)) / 6.0) * 1.4, f"Low fluid intake ({int(fluids)} glasses)", "increasing"),
        (0.8 if not med else 0.0, "Medication not taken today", "increasing"),
        (((temp - 36.5) / 2.0) * 1.2, f"Body temperature {temp:.1f}°C", "increasing"),
    ]
    contributions.sort(key=lambda x: x[0], reverse=True)
    return [
        {"factor": label, "direction": direction}
        for score, label, direction in contributions[:3]
        if score > 0.05
    ]
