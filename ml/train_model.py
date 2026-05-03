"""
train_model.py — LightGBM VOC prediction training pipeline for Warrior Blood.

Generates synthetic SCD diary data, trains a LightGBM classifier,
evaluates on a temporally-split test set, and exports to ONNX.

References:
    Machado et al. (2024) — LightGBM on SCD prediction
    Ke et al. (2017) — original LightGBM paper
    Lundberg & Lee (2017) — SHAP values

Run:
    python ml/train_model.py

Outputs:
    models/voc_model.lgb          — LightGBM native model (for SHAP)
    models/voc_model.onnx         — ONNX model (for fast inference)
    ml/outputs/metrics.json       — AUROC, Brier, sensitivity at 0.5
    ml/outputs/shap_importance.json — Mean |SHAP| per feature
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as `python ml/train_model.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ml.features import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def generate_synthetic_data(n_patients: int = 80, n_days: int = 365) -> pd.DataFrame:
    """
    Generate a synthetic SCD patient diary dataset with 16 ML features.

    Clinical statistics drawn from Yallop et al. (2007) and Machado et al. (2024):
    - VOC rate ~7% of patient-days
    - Pain score mean 2.3 (non-VOC), 6.8 (VOC)
    - Dehydration significantly associated with VOC (OR ~2.1)
    """
    rows = []
    start_date = pd.Timestamp("2024-01-01")

    for pid in range(n_patients):
        baseline_risk = np.random.beta(2, 8)
        patient_rows: list[dict] = []

        for day in range(n_days):
            entry_date = start_date + pd.Timedelta(days=day)

            # Seasonal temperature effect (Nolan et al. 2008)
            month = entry_date.month
            temp_base = 28 + 4 * np.sin(2 * np.pi * (month - 3) / 12)
            ambient_temp = float(np.random.normal(temp_base, 3))
            humidity = float(np.random.normal(70, 15))
            aqi = int(np.clip(np.random.poisson(1.5 + baseline_risk * 1.5), 1, 5))

            # Clinical features
            pain = int(np.clip(np.random.poisson(2.5 + baseline_risk * 4), 0, 10))
            fever = bool(np.random.random() < 0.05 + baseline_risk * 0.15)
            fluids = int(np.clip(np.random.normal(6, 2), 0, 20))
            urine = int(np.clip(np.random.normal(3, 1.5), 1, 8))
            med = bool(np.random.random() > 0.1)
            sleep = float(np.clip(np.random.normal(7, 1.5), 2, 12))
            body_temp = float(np.random.normal(36.8 + (0.8 if fever else 0), 0.4))

            # VOC label: higher prob when pain high, dehydrated, fever, hot weather
            # Intercept -4.5 targets ~7% base VOC rate before derived feature boost
            base_logit = (
                -4.5
                + (pain / 10) * 3.0
                + (1.2 if fever else 0)
                + ((urine - 3) / 5) * 1.5
                + ((6 - min(fluids, 6)) / 6) * 1.2
                + (0 if med else 0.7)
                + ((ambient_temp - 28) / 10) * 0.8
                + baseline_risk * 2.0
            )
            voc_prob = 1 / (1 + np.exp(-base_logit))
            voc_72h = int(np.random.random() < voc_prob)

            patient_rows.append({
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
                "aqi": aqi,
                "_base_logit": base_logit,
                "voc_72h": voc_72h,
            })

        # Compute lagged / rolling features for this patient's history
        pain_scores = [r["pain_score"] for r in patient_rows]
        med_flags   = [r["med_taken"]   for r in patient_rows]
        voc_labels  = [r["voc_72h"]     for r in patient_rows]
        fluids_list = [r["fluid_intake_glasses"] for r in patient_rows]
        urine_list  = [r["urine_colour"] for r in patient_rows]

        for i, row in enumerate(patient_rows):
            # pain_slope_3d — linear slope over last 3 days; 0 when insufficient history
            if i >= 2:
                y = [float(pain_scores[j]) for j in range(i - 2, i + 1)]
                row["pain_slope_3d"] = round(float(np.polyfit([0.0, 1.0, 2.0], y, 1)[0]), 3)
            else:
                row["pain_slope_3d"] = 0.0

            # pain_max_7d
            start_idx = max(0, i - 6)
            row["pain_max_7d"] = float(max(pain_scores[start_idx : i + 1]))

            # hydration_risk_score — composite 0-7 proxy matching backend/hydration.py thresholds
            f, u = fluids_list[i], urine_list[i]
            hr = (3 if f < 4 else 2 if f < 6 else 1 if f < 8 else 0)
            hr += (3 if u >= 7 else 2 if u >= 5 else 1 if u >= 4 else 0)
            row["hydration_risk_score"] = float(hr)

            # med_adherence_7d
            start7 = max(0, i - 6)
            row["med_adherence_7d"] = float(np.mean(med_flags[start7 : i + 1]))

            # prior_voc_30d — any VOC in the 30 days before today
            start30 = max(0, i - 30)
            row["prior_voc_30d"] = float(1 if any(v == 1 for v in voc_labels[start30:i]) else 0)

            # days_since_last_voc — 365 sentinel when no prior VOC
            past_voc = [j for j in range(i) if voc_labels[j] == 1]
            row["days_since_last_voc"] = float(i - past_voc[-1]) if past_voc else 365.0

            # Recompute voc_72h with derived feature signal so model can learn from them
            enhanced_logit = (
                row["_base_logit"]
                + float(np.clip(row["pain_slope_3d"], -2, 2)) * 0.5
                + row["prior_voc_30d"] * 0.8
                + (row["hydration_risk_score"] / 7) * 0.5
                - (row["med_adherence_7d"] - 0.9) * 0.4
            )
            row["voc_72h"] = int(np.random.random() < 1 / (1 + np.exp(-enhanced_logit)))
            del row["_base_logit"]

        rows.extend(patient_rows)

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} patient-days | VOC rate: {df['voc_72h'].mean():.1%}")
    return df


def train() -> None:
    """Full training pipeline: generate → split → SMOTE → train → SHAP → evaluate → save."""
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

    # Temporal split — prevents data leakage (Machado et al. 2024)
    split_date = pd.Timestamp("2024-10-01")
    train_df = df[df["entry_date"] < split_date]
    test_df  = df[df["entry_date"] >= split_date]

    X_train = train_df[FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df["voc_72h"].values
    X_test  = test_df[FEATURE_NAMES].values.astype(np.float32)
    y_test  = test_df["voc_72h"].values

    print(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"Train VOC rate: {y_train.mean():.1%} | Test VOC rate: {y_test.mean():.1%}")

    # SMOTE — applied on training set ONLY after temporal split (Machado et al. 2024)
    smote_applied = False
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=SEED, k_neighbors=5)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        smote_applied = True
        print(
            f"After SMOTE — train samples: {len(X_train_res)}, "
            f"positive rate: {y_train_res.mean():.1%}"
        )
    except ImportError:
        print("imbalanced-learn not installed — skipping SMOTE")
        X_train_res, y_train_res = X_train, y_train

    # scale_pos_weight only when SMOTE has NOT already balanced the classes
    pos_weight = 1 if smote_applied else int((1 - y_train.mean()) / max(y_train.mean(), 1e-9))

    # LightGBM training
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=pos_weight,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)])

    # Calibration — isotonic on original training distribution
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_train, y_train)

    # SHAP feature importance on test set
    try:
        import shap as shap_lib
        explainer = shap_lib.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        # Binary classification returns [neg_class, pos_class]; take positive class
        shap_values = sv[1] if isinstance(sv, list) else sv
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = sorted(
            zip(FEATURE_NAMES, mean_abs_shap.tolist()), key=lambda x: -x[1]
        )
        print("\nTop SHAP features:")
        for feat, val in importance[:5]:
            print(f"  {feat}: {val:.4f}")
        with open("ml/outputs/shap_importance.json", "w") as f:
            json.dump({k: round(float(v), 4) for k, v in importance}, f, indent=2)
        print("Saved: ml/outputs/shap_importance.json")
    except Exception as e:
        print(f"shap unavailable ({e.__class__.__name__}) — skipping SHAP importance")

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

    metrics = {
        "auroc": round(auroc, 4),
        "brier": round(brier, 4),
        "sensitivity_50": round(sens_50, 4),
    }
    with open("ml/outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save native LightGBM model (used by SHAP TreeExplainer in production)
    model.booster_.save_model("models/voc_model.lgb")
    print("Saved: models/voc_model.lgb")

    # ONNX export — try convert_sklearn on calibrated model first
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onnx_model = convert_sklearn(
            calibrated,
            initial_types=[("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))],
        )
        with open("models/voc_model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        size_kb = Path("models/voc_model.onnx").stat().st_size // 1024
        print(f"Saved: models/voc_model.onnx ({size_kb} KB)")
    except Exception as e:
        print(f"skl2onnx export failed ({e}) — trying onnxmltools direct LGB export...")
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

            lgb_onnx = convert_lightgbm(
                model.booster_,
                initial_types=[("float_input", OnnxFloat([None, len(FEATURE_NAMES)]))],
            )
            with open("models/voc_model.onnx", "wb") as f:
                f.write(lgb_onnx.SerializeToString())
            size_kb = Path("models/voc_model.onnx").stat().st_size // 1024
            print(f"Saved: models/voc_model.onnx via onnxmltools ({size_kb} KB)")
        except Exception as e2:
            print(f"Both ONNX exports failed: {e2} — using LGB native model only")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
