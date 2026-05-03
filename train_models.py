"""
train_models.py — Train all three XGBoost models using best hyperparameters
found during RandomizedSearchCV (100 iter, 5-fold CV) in the model notebooks.

Run once before starting the app:
    python3 train_models.py

Saves to models/:
    {target}_model.json        — XGBoost native format (no sklearn dependency at load time)
    {target}_feature_cols.pkl  — ordered list of 45 feature names
    {target}_transform.pkl     — inverse-transform metadata
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGETS = ["acc_total", "total_distance", "vel_total"]

BASE_FEATURES = [
    "height", "weight", "age",
    "has_G", "has_TAC", "has_BP", "has_TEC", "has_MATCH", "n_session_types",
    "pos_central_back", "pos_central_midfielder", "pos_forward",
    "pos_full_back", "pos_winger",
    "days_since_start", "days_since_last_activity", "days_since_last_match",
]

# Hyperparameters found via RandomizedSearchCV (100 iter, 5-fold CV) in model notebooks; not hand-tuned
BEST_PARAMS = {
    "acc_total": dict(
        objective="reg:tweedie", tweedie_variance_power=1.9,
        n_estimators=1600, learning_rate=0.005, max_depth=4,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        colsample_bylevel=0.6, gamma=0.3, reg_alpha=1, reg_lambda=5,
    ),
    "total_distance": dict(
        objective="reg:squarederror",
        n_estimators=600, learning_rate=0.02, max_depth=8,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.5,
        colsample_bylevel=0.6, gamma=0.5, reg_alpha=0.01, reg_lambda=0.5,
    ),
    "vel_total": dict(
        objective="reg:squarederror",
        n_estimators=200, learning_rate=0.07, max_depth=3,
        min_child_weight=2, subsample=1.0, colsample_bytree=1.0,
        colsample_bylevel=0.8, gamma=0.3, reg_alpha=10, reg_lambda=5,
    ),
}

TRANSFORMS = {
    "acc_total":      {"type": "none",  "loss": "tweedie"},
    "total_distance": {"type": "log1p", "inverse": "expm1"},
    "vel_total":      {"type": "none",  "loss": "mse"},
}


def main():
    print("=" * 60)
    print("Real Madrid ACWR — Model Training")
    print("=" * 60)

    print("\nLoading model_data.parquet...")
    df = pd.read_parquet(DATA_DIR / "processed" / "model_data.parquet")
    print(f"  Rows: {len(df)}  Players: {df['player_id'].nunique()}")

    player_dummies = pd.get_dummies(df["player_id"], prefix="pid", dtype=int)
    df = pd.concat([df, player_dummies], axis=1)
    pid_features = sorted(c for c in df.columns if c.startswith("pid_"))
    feature_cols = BASE_FEATURES + pid_features
    print(f"  Features: {len(feature_cols)} ({len(BASE_FEATURES)} base + {len(pid_features)} player IDs)")

    for target in TARGETS:
        print(f"\n{'─' * 60}")
        print(f"  Target: {target}")

        X = df[feature_cols].copy()
        y = df[target].copy()

        if TRANSFORMS[target]["type"] == "log1p":
            # total_distance is right-skewed (skew ~3.5); log1p normalises the distribution,
            # preventing large outliers from dominating MSE during training
            y_fit = np.log1p(y)
            print(f"  Transform: log1p  (skew {y.skew():.2f} → {y_fit.skew():.2f})")
        else:
            y_fit = y

        # shuffle=True intentionally: players have overlapping seasons, so a naive time split
        # would leak future load patterns of other players into training; shuffled split avoids this
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_fit, test_size=0.20, random_state=42, shuffle=True
        )

        params = {
            **BEST_PARAMS[target],
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)

        print(f"  Fitting on {len(X_train)} samples...", end=" ", flush=True)
        model.fit(X_train, y_train)
        print("done")

        y_pred_raw = model.predict(X_test)
        if TRANSFORMS[target]["type"] == "log1p":
            y_pred = np.clip(np.expm1(y_pred_raw), 0, None)
            y_true = np.expm1(y_test)
        else:
            y_pred = np.clip(y_pred_raw, 0, None)
            y_true = y_test

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"  Test MAE: {mae:.3f}  R²: {r2:.3f}")

        # Must call on the underlying Booster: XGBRegressor.save_model() triggers sklearn type detection and fails
        model.get_booster().save_model(str(MODELS_DIR / f"{target}_model.json"))
        with open(MODELS_DIR / f"{target}_feature_cols.pkl", "wb") as f:
            pickle.dump(feature_cols, f)
        with open(MODELS_DIR / f"{target}_transform.pkl", "wb") as f:
            pickle.dump(TRANSFORMS[target], f)

        print(f"  Saved: {target}_model.json, {target}_feature_cols.pkl, {target}_transform.pkl")

    print(f"\n{'=' * 60}")
    print(f"All models saved to: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
