"""
train.py — Train three load-forecasting models (accelerations, sprint_distance, total_distance)
using date-blocked CV with XGBoost, MinMaxScaler, and log1p transform.

Run once before starting the app:
    python train_models.py

Saves to models/xgboost/{target}/:
    bundle.joblib  — dict with model, scaler, feature_cols, ewma_spans

Also saves to data/processed/:
    daily.parquet  — raw daily frame used for history at inference time
"""

from __future__ import annotations

import warnings
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor  # noqa: E402

from real_madrid_acwr.config import DAILY_PARQUET, DATA_DIR, MODEL_ARTIFACTS_DIR

warnings.filterwarnings("ignore")

RAW_CSV = DATA_DIR / "raw" / "data_acute_vs_chronic.csv"
RAW_ZIP = DATA_DIR / "data_acute_vs_chronic.zip"

TARGETS = ["accelerations", "sprint_distance", "total_distance"]

# Columns to drop per target before selecting feature_cols (beyond player_id, position)
EXTRA_DROPS: dict[str, list[str]] = {
    "accelerations":  ["total_distance"],
    "sprint_distance": ["total_distance"],
    "total_distance": [],
}


def _load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        if not RAW_ZIP.exists():
            raise FileNotFoundError(
                f"Missing raw data. Expected {RAW_CSV} or bootstrap archive {RAW_ZIP}."
            )
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RAW_ZIP) as archive:
            archive.extract(RAW_CSV.name, path=RAW_CSV.parent)
    return pd.read_csv(RAW_CSV)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "velocity_band6plus7_total_distance": "sprint_distance",
        "acc_band7plus_total_effort_count":   "accelerations",
    })
    df["is_official_match"] = df["is_official_match"].fillna(0)
    df["player_id"] = df["player_id"].astype("category")
    df["period_name"]   = df["period_name"].fillna("MATCH")
    df["exercise_type"] = df["period_name"].str.split(" ").str[0]
    df = df.drop(columns=["is_official_match"])

    df["period_start_time"] = pd.to_datetime(df["period_start_time"]).dt.tz_localize(None).dt.normalize()
    df["date_of_birth"]     = pd.to_datetime(df["date_of_birth"]).dt.tz_localize(None).dt.normalize()
    df["age"] = ((df["period_start_time"] - df["date_of_birth"]).dt.days / 365.25).round(0)

    df = df.dropna(subset=["height", "weight", "position_name_en", "date_of_birth"]).reset_index(drop=True)

    # Fix anomalous match distance for player 94884
    median_94884 = df[
        (df["player_id"] == 94884) &
        (df["exercise_type"] == "MATCH") &
        (df["total_distance"] <= 20000)
    ]["total_distance"].median()
    df.loc[(df["player_id"] == 94884) & (df["total_distance"] >= 20000), "total_distance"] = median_94884

    # Remove trialist placeholders (weight == 200)
    trialists = df[df["weight"] == 200]["player_id"].unique()
    df = df[~df["player_id"].isin(trialists)].reset_index(drop=True)

    return df


def _build_daily(df: pd.DataFrame) -> pd.DataFrame:
    et_dummies = pd.get_dummies(df["exercise_type"], prefix="has").astype(int)
    df_enc = pd.concat([df[["player_id", "period_start_time"]], et_dummies], axis=1)
    et_daily = (
        df_enc.groupby(["player_id", "period_start_time"], observed=True)
        .sum()
        .reset_index()
    )

    daily = (
        df.groupby(["player_id", "period_start_time"], observed=True)
        .agg(
            total_distance   = ("total_distance",   "sum"),
            accelerations    = ("accelerations",    "sum"),
            sprint_distance  = ("sprint_distance",  "sum"),
            n_periods        = ("activity_id",      "count"),
            n_exercise_types = ("exercise_type",    "nunique"),
            height           = ("height",           "first"),
            weight           = ("weight",           "first"),
            age              = ("age",              "first"),
            position         = ("position_name_en", "first"),
        )
        .reset_index()
        .merge(et_daily, on=["player_id", "period_start_time"])
        .rename(columns={"period_start_time": "date"})
        .sort_values(["player_id", "date"])
        .reset_index(drop=True)
    )

    # Ensure standard has_ columns are present even if absent in data
    for col in ["has_G", "has_TAC", "has_BP", "has_TEC", "has_MATCH"]:
        if col not in daily.columns:
            daily[col] = 0

    # Build continuous date spine per player, filling gaps with zero load
    zero_cols   = ["total_distance", "accelerations", "sprint_distance", "n_periods", "n_exercise_types"] + \
                  [c for c in daily.columns if c.startswith("has_")]
    static_cols = ["height", "weight", "age", "position"]

    player_date_range = daily.groupby("player_id", observed=True)["date"].agg(["min", "max"])
    spines = [
        pd.DataFrame({"player_id": pid, "date": pd.date_range(row["min"], row["max"], freq="D")})
        for pid, row in player_date_range.iterrows()
    ]
    spine = pd.concat(spines, ignore_index=True)
    spine["player_id"] = spine["player_id"].astype(daily["player_id"].dtype)

    daily = (
        spine
        .merge(daily, on=["player_id", "date"], how="left")
        .sort_values(["player_id", "date"])
        .reset_index(drop=True)
    )
    daily[zero_cols]   = daily[zero_cols].fillna(0)
    daily[static_cols] = (
        daily.groupby("player_id", observed=True)[static_cols]
        .transform(lambda s: s.ffill().bfill())
    )

    return daily


def _add_features(daily_df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = daily_df.copy().sort_values(["player_id", "date"]).reset_index(drop=True)

    d["day_of_week"] = d["date"].dt.dayofweek

    d["_mc"] = d["date"].dt.to_period("W-SUN")
    _mc_grp  = d.groupby(["player_id", "_mc"], observed=True)[target]
    _mc_mean = _mc_grp.transform(lambda s: s.shift(1).fillna(0).expanding().mean())
    _mc_std  = _mc_grp.transform(lambda s: s.shift(1).fillna(0).expanding().std()).fillna(0)
    d["microcycle_load_sum"]     = _mc_grp.transform(lambda s: s.shift(1).fillna(0).cumsum())
    d["microcycle_load_std_dev"] = _mc_std
    d["monotony"] = np.where(_mc_std == 0, 0, _mc_mean / _mc_std)
    d["strain"]   = d["microcycle_load_sum"] * d["monotony"]
    d = d.drop(columns=["_mc"])

    d["acute_load"] = (
        d.groupby("player_id", observed=True)[target]
        .transform(lambda s: s.shift(1).ewm(span=7, min_periods=1).mean())
    )
    d["chronic_load"] = (
        d.groupby("player_id", observed=True)[target]
        .transform(lambda s: s.shift(1).ewm(span=28, min_periods=1).mean())
    )
    d["training_stress_balance"] = d["chronic_load"] - d["acute_load"]
    d["acwr"] = d["acute_load"] / d["chronic_load"].replace(0, np.nan)

    for lag in [1, 3, 5, 7, 14]:
        d[f"load_lag_{lag}"] = (
            d.groupby("player_id", observed=True)[target]
            .transform(lambda s, n=lag: s.shift(n))
        )

    for window in [3, 7, 14]:
        d[f"load_ma_{window}"] = (
            d.groupby("player_id", observed=True)[target]
            .transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        )

    num_cols = d.select_dtypes(include="number").columns
    d[num_cols] = d[num_cols].fillna(0)

    return d


def _date_blocked_cv(dates: np.ndarray, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    dates        = np.asarray(dates)
    unique_dates = np.sort(pd.unique(dates))
    blocks       = np.array_split(unique_dates, n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        train_dates = np.concatenate(blocks[:i])
        val_dates   = blocks[i]
        train_idx   = np.flatnonzero(np.isin(dates, train_dates))
        val_idx     = np.flatnonzero(np.isin(dates, val_dates))
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def _train_target(
    daily: pd.DataFrame,
    target: str,
    target_dir: Path,
    n_iter: int = 25,
    random_state: int = 42,
) -> None:
    all_dates = np.sort(daily["date"].unique())
    cutoff    = all_dates[int(len(all_dates) * 0.8) - 1]

    train_base = daily[daily["date"] <= cutoff].copy().reset_index(drop=True)
    test_base  = daily[daily["date"] >  cutoff].copy().reset_index(drop=True)

    train = _add_features(train_base, target)
    test  = _add_features(test_base,  target)

    # log1p transform the target
    train[target] = np.log1p(train[target])
    test[target]  = np.log1p(test[target])

    # Drop non-feature columns
    drop_cols = ["player_id", "position", "date"] + EXTRA_DROPS[target]
    drop_cols = [c for c in drop_cols if c in train.columns]
    train = train.drop(columns=drop_cols)
    test  = test.drop(columns=drop_cols)

    feature_cols = [c for c in train.select_dtypes(include="number").columns if c != target]

    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    X_tr, y_tr = train[feature_cols].values, train[target].values
    X_te, y_te = test[feature_cols].values,  test[target].values

    cv = _date_blocked_cv(train_base["date"].values)

    param_space: dict[str, Any] = {
        "n_estimators":     [200, 400, 600, 800],
        "learning_rate":    [0.01, 0.02, 0.05, 0.1],
        "max_depth":        [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 10],
        "subsample":        [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 1.0],
        "gamma":            [0, 0.1, 0.3, 0.5],
        "reg_lambda":       [0.1, 0.5, 1.0, 5.0],
    }
    base_model = XGBRegressor(
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_space,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        error_score="raise",
    )
    print(f"  Fitting on {len(X_tr)} samples (date-blocked CV)...", end=" ", flush=True)
    search.fit(X_tr, y_tr)
    print("done")

    model = search.best_estimator_

    y_pred_log = model.predict(X_te)
    y_pred     = np.clip(np.expm1(y_pred_log), 0, None)
    y_true     = np.expm1(y_te)

    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"  Test MAE: {mae:.3f}  R²: {r2:.3f}")

    target_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "ewma_spans":   {"acute": 7, "chronic": 28},
    }
    joblib.dump(bundle, target_dir / "bundle.joblib")
    print(f"  Saved: {(target_dir / 'bundle.joblib').relative_to(MODEL_ARTIFACTS_DIR.parent)}")


def main() -> None:
    print("=" * 60)
    print("Real Madrid ACWR — Model Training")
    print("=" * 60)

    print("\nLoading raw data...")
    df = _load_raw()
    print(f"  Raw shape: {df.shape}  ({df['player_id'].nunique()} players)")

    df = _preprocess(df)
    print(f"  After preprocessing: {df.shape}")

    daily = _build_daily(df)
    print(f"  Daily frame: {daily.shape}  ({daily['player_id'].nunique()} players)")

    # Save daily parquet for use at inference time
    DAILY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PARQUET, index=False)
    print(f"  Saved daily.parquet → {DAILY_PARQUET.relative_to(DATA_DIR.parent)}")

    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        print(f"\n{'─' * 60}")
        print(f"  Target: {target}")
        _train_target(daily, target, MODEL_ARTIFACTS_DIR / target)

    print(f"\n{'=' * 60}")
    print(f"All bundles saved to: {MODEL_ARTIFACTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
