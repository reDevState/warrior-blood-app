"""
train_model.py — LightGBM VOC prediction training pipeline for Warrior Blood.

Generates synthetic SCD diary data, trains a LightGBM classifier,
evaluates on a temporally-split test set, and exports to ONNX.

References:
    Machado et al. (2024) — LightGBM on SCD prediction
    Ke et al. (2017) — original LightGBM paper

Run:
    python ml/train_model.py

Outputs:
    models/voc_model.lgb     — LightGBM native model
    ml/outputs/metrics.txt   — AUROC, Brier, sensitivity at 0.5
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def generate_synthetic_data(n_patients: int = 80, n_days: int = 365) -> pd.DataFrame:
    """
    Generate a synthetic SCD patient diary dataset.

    Clinical statistics drawn from Yallop et al. (2007) and Machado et al. (2024):
    - VOC rate ~7% of patient-days
    - Pain score mean 2.3 (non-VOC), 6.8 (VOC)
    - Dehydration significantly associated with VOC (OR ~2.1)

    Args:
        n_patients: Number of synthetic patients.
        n_days: Days of diary data per patient.

    Returns:
        DataFrame with one row per patient-day.
    """
    rows = []
    start_date = pd.Timestamp("2024-01-01")

    for pid in range(n_patients):
        # Patient-level baseline risk (some patients are higher risk)
        baseline_risk = np.random.beta(2, 8)

        for day in range(n_days):
            entry_date = start_date + pd.Timedelta(days=day)

            # Seasonal temperature effect (Nolan et al. 2008)
            month = entry_date.month
            temp_base = 28 + 4 * np.sin(2 * np.pi * (month - 3) / 12)
            ambient_temp = float(np.random.normal(temp_base, 3))
            humidity = float(np.random.normal(70, 15))

            # Clinical features — slightly elevated near VOC events
            pain = int(np.clip(np.random.poisson(2.5 + baseline_risk * 4), 0, 10))
            fever = bool(np.random.random() < 0.05 + baseline_risk * 0.15)
            fluids = int(np.clip(np.random.normal(6, 2), 0, 20))
            urine = int(np.clip(np.random.normal(3, 1.5), 1, 8))
            med = bool(np.random.random() > 0.1)
            sleep = float(np.clip(np.random.normal(7, 1.5), 2, 12))
            body_temp = float(np.random.normal(36.8 + (0.8 if fever else 0), 0.4))

            # VOC label: higher prob when pain high, dehydrated, fever, hot weather
            voc_logit = (
                -3.5
                + (pain / 10) * 3.0
                + (1.2 if fever else 0)
                + ((urine - 3) / 5) * 1.5
                + ((6 - min(fluids, 6)) / 6) * 1.2
                + (0 if med else 0.7)
                + ((ambient_temp - 28) / 10) * 0.8
                + baseline_risk * 2.0
            )
            voc_prob = 1 / (1 + np.exp(-voc_logit))
            voc_72h = int(np.random.random() < voc_prob)

            rows.append({
                "patient_id": f"P{pid:03d}",
                "entry_date": entry_date,
                "pain_score": pain,
                "body_temp_c": round(body_temp, 1),
                "fever_present": int(fever),
                "fluid_intake_glasses": fluids,
                "urine_colour": urine,
                "med_taken": int(med),
                "sleep_hours": round(sleep, 1),
                "ambient_temp_c": round(ambient_temp, 1),
                "humidity_pct": round(min(max(humidity, 10), 100), 1),
                "voc_72h": voc_72h,
            })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} patient-days | VOC rate: {df['voc_72h'].mean():.1%}")
    return df


FEATURES = [
    "pain_score", "body_temp_c", "fever_present", "fluid_intake_glasses",
    "urine_colour", "med_taken", "sleep_hours", "ambient_temp_c", "humidity_pct",
]


def train() -> None:
    """Full training pipeline: generate → split → train → evaluate → save."""
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, brier_score_loss
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        print("lightgbm and scikit-learn required. Run: pip install lightgbm scikit-learn")
        return

    Path("models").mkdir(exist_ok=True)
    Path("ml/outputs").mkdir(parents=True, exist_ok=True)
    Path("ml/data").mkdir(parents=True, exist_ok=True)

    # Generate data
    df = generate_synthetic_data(n_patients=80, n_days=365)
    df.to_csv("ml/data/scd_diary_dataset.csv", index=False)

    # Temporal split — critical for clinical time-series (no random shuffle)
    # Machado et al. (2024): temporal split prevents data leakage
    split_date = pd.Timestamp("2024-10-01")
    train_df = df[df["entry_date"] < split_date]
    test_df  = df[df["entry_date"] >= split_date]

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df["voc_72h"].values
    X_test  = test_df[FEATURES].values.astype(np.float32)
    y_test  = test_df["voc_72h"].values

    print(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"Train VOC rate: {y_train.mean():.1%} | Test VOC rate: {y_test.mean():.1%}")

    # LightGBM training
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=int((1 - y_train.mean()) / y_train.mean()),
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # Calibration (CalibratedClassifierCV — isotonic)
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_train, y_train)

    # Evaluation
    probs = calibrated.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    sens_50 = float(np.mean((probs >= 0.5) & (y_test == 1))) / max(float(y_test.mean()), 1e-9)

    print(f"\n{'='*50}")
    print(f"AUROC:       {auroc:.3f}  (target ≥ 0.80)")
    print(f"Brier score: {brier:.3f}  (target ≤ 0.10)")
    print(f"Sensitivity: {sens_50:.3f}  (at threshold 0.50)")
    print(f"{'='*50}\n")

    # Save metrics
    metrics = {"auroc": round(auroc, 4), "brier": round(brier, 4), "sensitivity_50": round(sens_50, 4)}
    with open("ml/outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save native LightGBM model
    model.booster_.save_model("models/voc_model.lgb")
    print("Saved: models/voc_model.lgb")

    # ONNX export (optional — requires onnxmltools + skl2onnx)
    try:
        import onnxmltools
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onnx_model = convert_sklearn(
            calibrated,
            initial_types=[("float_input", FloatTensorType([None, len(FEATURES)]))],
        )
        with open("models/voc_model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        size_kb = Path("models/voc_model.onnx").stat().st_size // 1024
        print(f"Saved: models/voc_model.onnx ({size_kb} KB)")
    except ImportError:
        print("Skipping ONNX export (install onnxmltools + skl2onnx for ONNX export)")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
