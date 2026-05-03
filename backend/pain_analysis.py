"""
pain_analysis.py — Pain trend analysis for Warrior Blood.

Computes derived features from the pain diary used by the ML predictor
and the plain-English suggestion engine.

References:
    Brandow et al. (2020) — pain trajectories in SCD
    Smith et al. (2008) — breakthrough pain and hospitalisation
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class PainTrend:
    slope_3d: float        # Linear slope over last 3 days (positive = rising)
    mean_7d: float         # 7-day rolling mean
    max_24h: int           # Maximum pain score in last 24 hours
    is_breakthrough: bool  # Score >= mean_7d + 3 and score >= 7
    is_chest_pain: bool    # CHEST in pain_locations
    is_rising: bool        # slope_3d > 0.5


def compute_pain_trend(
    recent_scores: list[int],
    current_score: int,
    pain_locations: list[str] | None,
) -> PainTrend:
    """
    Compute pain trend features from recent diary history.

    Args:
        recent_scores: Daily pain scores oldest-first, up to 7 days.
                       May be empty for new patients.
        current_score: Today's pain score (0–10).
        pain_locations: Body location codes for today's entry.

    Returns:
        PainTrend with derived features used by the suggestion engine.
    """
    all_scores = recent_scores + [current_score]

    last3 = all_scores[-3:] if len(all_scores) >= 3 else all_scores
    if len(last3) >= 2:
        n = len(last3)
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(last3) / n
        denom = sum((x - x_mean) ** 2 for x in xs)
        slope_3d = (
            sum((xs[i] - x_mean) * (last3[i] - y_mean) for i in range(n)) / denom
            if denom > 0 else 0.0
        )
    else:
        slope_3d = 0.0

    mean_7d = statistics.mean(all_scores[-7:]) if all_scores else float(current_score)
    max_24h = current_score

    is_breakthrough = (current_score >= mean_7d + 3) and (current_score >= 7)
    is_chest_pain = bool(pain_locations and "CHEST" in pain_locations)
    is_rising = slope_3d > 0.5

    return PainTrend(
        slope_3d=round(slope_3d, 2),
        mean_7d=round(mean_7d, 2),
        max_24h=max_24h,
        is_breakthrough=is_breakthrough,
        is_chest_pain=is_chest_pain,
        is_rising=is_rising,
    )
