"""
ml_predictor.py — Production VOC predictor using LightGBM ONNX + SHAP.

Loads voc_model.onnx for fast inference (<10ms on ARM).
Loads voc_model.lgb for SHAP TreeExplainer (exact, not approximate).
Falls back to the rule-based heuristic when models are not yet trained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.features import FEATURE_NAMES

MODEL_PATH = Path("models/voc_model.onnx")
LGB_PATH   = Path("models/voc_model.lgb")

# Loaded once at module import — not per request
# Only import heavy native libraries when model files actually exist
_session    = None
_input_name = None
_lgb_model  = None
_explainer  = None

if MODEL_PATH.exists():
    try:
        import onnxruntime as rt
        _session    = rt.InferenceSession(str(MODEL_PATH))
        _input_name = _session.get_inputs()[0].name
    except Exception:
        pass

if LGB_PATH.exists():
    try:
        import lightgbm as lgb
        import shap as shap_lib
        _lgb_model = lgb.Booster(model_file=str(LGB_PATH))
        _explainer = shap_lib.TreeExplainer(_lgb_model)
    except Exception:
        pass

LOW_THRESHOLD  = 0.30
HIGH_THRESHOLD = 0.60

_SHAP_LABELS: dict[str, str] = {
    "pain_slope_3d":       "Pain rising over 3 days",
    "prior_voc_30d":       "Recent crisis in last 30 days",
    "hydration_risk_score":"Dehydration risk",
    "fluid_intake_glasses":"Low fluid intake",
    "urine_colour":        "Dark urine",
    "fever_present":       "Fever present",
    "body_temp_c":         "Elevated body temperature",
    "med_taken":           "Medication not taken",
    "ambient_temp_c":      "Extreme outdoor temperature",
    "aqi":                 "Poor air quality",
    "pain_score":          "High pain score",
    "pain_max_7d":         "Pain elevated over 7 days",
    "med_adherence_7d":    "Low medication adherence",
    "days_since_last_voc": "Days since last crisis",
}


@dataclass
class PredictionResult:
    risk_score:   float
    risk_tier:    str             # LOW | MODERATE | HIGH
    shap_factors: list[dict]
    suggestion:   str = ""


def predict(features: dict) -> PredictionResult:
    """
    Predict VOC risk using ONNX model + SHAP TreeExplainer.

    Falls back to heuristic if models are not yet trained.
    """
    x = np.array(
        [[(float(v) if (v := features.get(f)) is not None else math.nan) for f in FEATURE_NAMES]],
        dtype=np.float32,
    )

    if _session:
        prob = float(_session.run(None, {_input_name: x})[1][0][1])
    else:
        prob = _heuristic(features)

    tier = (
        "HIGH"     if prob >= HIGH_THRESHOLD else
        "MODERATE" if prob >= LOW_THRESHOLD  else
        "LOW"
    )

    shap_factors = _shap_factors(x) if _explainer else _heuristic_shap_factors(features)
    suggestion   = _build_suggestion(tier, shap_factors, features)

    return PredictionResult(
        risk_score=round(prob, 3),
        risk_tier=tier,
        shap_factors=shap_factors,
        suggestion=suggestion,
    )


def _shap_factors(x: np.ndarray) -> list[dict]:
    """Return top-3 SHAP factors as plain-English dicts (ONNX path)."""
    shap_vals = _explainer.shap_values(x)
    contributions = list(zip(FEATURE_NAMES, shap_vals[0].tolist()))
    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
    result = []
    for feat, val in contributions[:3]:
        if abs(val) > 0.02:
            result.append({
                "factor":       _SHAP_LABELS.get(feat, feat),
                "contribution": round(val, 3),
                "direction":    "increasing" if val > 0 else "decreasing",
            })
    return result


def _heuristic_shap_factors(features: dict) -> list[dict]:
    """Heuristic SHAP-proxy factors used before ONNX model is trained."""
    pain   = float(features.get("pain_score") or 0)
    fever  = bool(features.get("fever_present") or False)
    fluids = float(features.get("fluid_intake_glasses") or 6)
    urine  = float(features.get("urine_colour") or 3)
    med    = bool(features.get("med_taken") if features.get("med_taken") is not None else True)
    temp   = float(features.get("body_temp_c") or 36.5)

    contributions: list[tuple[float, str, str]] = [
        ((pain / 10.0) * 3.2,            f"Pain score is {int(pain)}/10",          "increasing"),
        (1.5 if fever else 0.0,           "Fever present",                           "increasing"),
        (((urine - 3) / 5.0) * 1.8,      f"Urine colour dark ({int(urine)}/8)",     "increasing"),
        (((6 - min(fluids, 6)) / 6.0) * 1.4, f"Low fluid intake ({int(fluids)} glasses)", "increasing"),
        (0.8 if not med else 0.0,         "Medication not taken today",              "increasing"),
        (((temp - 36.5) / 2.0) * 1.2,    f"Body temperature {temp:.1f}°C",          "increasing"),
    ]
    contributions.sort(key=lambda x: x[0], reverse=True)
    return [
        {"factor": label, "contribution": round(score, 3), "direction": direction}
        for score, label, direction in contributions[:3]
        if score > 0.05
    ]


def _build_suggestion(tier: str, shap_factors: list[dict], features: dict) -> str:
    """Generate a personalised plain-English suggestion from tier + SHAP factors."""
    top = [f["factor"] for f in shap_factors if f.get("direction") == "increasing"]
    if tier == "HIGH":
        reasons = " and ".join(top[:2]) if top else "multiple risk factors"
        return (
            f"Your VOC risk is HIGH today, mainly because of {reasons}. "
            "Contact your CHW immediately and rest. "
            "Drink 2 large glasses of water now. "
            "Take your medication if not already done."
        )
    if tier == "MODERATE":
        reasons = top[0] if top else "elevated risk indicators"
        return (
            f"Moderate VOC risk today — {reasons} is a concern. "
            "Drink 8+ glasses of water, avoid cold or exertion, "
            "and take your medication."
        )
    return (
        "Your VOC risk is LOW today. "
        "Keep up your fluids, take your medication, and rest well."
    )


def _heuristic(features: dict) -> float:
    """Fallback heuristic used before ONNX model is trained."""
    pain   = float(features.get("pain_score") or 0)
    temp   = float(features.get("body_temp_c") or 36.5)
    fever  = bool(features.get("fever_present") or False)
    fluids = float(features.get("fluid_intake_glasses") or 6)
    urine  = float(features.get("urine_colour") or 3)
    med    = bool(features.get("med_taken") if features.get("med_taken") is not None else True)
    sleep  = float(features.get("sleep_hours") or 7)

    logit = -2.5
    logit += (pain / 10.0) * 3.2
    logit += (1.5 if fever else 0.0)
    logit += ((urine - 3) / 5.0) * 1.8
    logit += ((6 - min(fluids, 6)) / 6.0) * 1.4
    logit += (0.0 if med else 0.8)
    logit += ((temp - 36.5) / 2.0) * 1.2
    logit += ((6 - min(sleep, 6)) / 6.0) * 0.6
    return 1.0 / (1.0 + math.exp(-logit))
