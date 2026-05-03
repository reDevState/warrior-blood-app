"""
features.py — Feature engineering for the Warrior Blood ML pipeline.

Computes the 16 derived features used by the LightGBM model from
raw diary history. Called at training time and at inference time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


FEATURE_NAMES = [
    "pain_score",            # Today's NRS pain score
    "pain_slope_3d",         # Linear slope of pain over 3 days (Brandow 2020)
    "pain_max_7d",           # Max pain score in last 7 days
    "body_temp_c",           # Body temperature
    "fever_present",         # Binary fever flag
    "fluid_intake_glasses",  # Daily fluid intake in glasses (250 ml each)
    "urine_colour",          # Armstrong 1994 colour chart 1-8
    "hydration_risk_score",  # Composite 0-7 from hydration module
    "med_taken",             # Hydroxyurea / medication taken today
    "med_adherence_7d",      # Medication adherence rate last 7 days (0-1)
    "sleep_hours",           # Hours of sleep last night
    "ambient_temp_c",        # Outdoor temperature from OWM
    "humidity_pct",          # Outdoor humidity from OWM
    "aqi",                   # Air quality index (WHO 1-5 scale)
    "prior_voc_30d",         # Binary: VOC in last 30 days
    "days_since_last_voc",   # Days since most recent VOC (365 = no prior VOC)
]


@dataclass
class FeatureVector:
    """16 ML features for one patient at one point in time."""
    pain_score: float = float("nan")
    pain_slope_3d: float = float("nan")
    pain_max_7d: float = float("nan")
    body_temp_c: float = float("nan")
    fever_present: float = float("nan")
    fluid_intake_glasses: float = float("nan")
    urine_colour: float = float("nan")
    hydration_risk_score: float = float("nan")
    med_taken: float = float("nan")
    med_adherence_7d: float = float("nan")
    sleep_hours: float = float("nan")
    ambient_temp_c: float = float("nan")
    humidity_pct: float = float("nan")
    aqi: float = float("nan")
    prior_voc_30d: float = float("nan")
    days_since_last_voc: float = float("nan")

    def to_list(self) -> list[float]:
        return [getattr(self, f) for f in FEATURE_NAMES]
