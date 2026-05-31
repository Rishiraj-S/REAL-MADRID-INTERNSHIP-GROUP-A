"""
datapipeline.py — Shared preprocessing pipeline for all three targets.

Exports
-------
load_data, clean_data, treat_outliers, aggregate_daily, spine_fill,
add_features, encode_dow, scale_train, prepare_test,
split_data, run_pipeline, build_full_daily
"""

from __future__ import annotations

import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from real_madrid_acwr.config import DATA_DIR

warnings.filterwarnings("ignore")

RAW_CSV = DATA_DIR / "raw" / "data_acute_vs_chronic.csv"
RAW_ZIP = DATA_DIR / "data_acute_vs_chronic.zip"


# =============================================================================
# 0. Data Loading
# =============================================================================

def load_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        if not RAW_ZIP.exists():
            raise FileNotFoundError(
                f"Missing raw data. Expected {RAW_CSV} or bootstrap archive {RAW_ZIP}."
            )
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RAW_ZIP) as archive:
            archive.extract(RAW_CSV.name, path=RAW_CSV.parent)
    df = pd.read_csv(RAW_CSV)
    print(f"Raw shape: {df.shape}  ({df['player_id'].nunique()} players)")
    return df


# =============================================================================
# 1. Data Cleaning
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "velocity_band6plus7_total_distance": "sprint_distance",
        "acc_band7plus_total_effort_count":   "accelerations",
    })
    df["is_official_match"] = df["is_official_match"].fillna(0)
    df["player_id"]         = df["player_id"].astype("category")
    df["period_name"]       = df["period_name"].fillna("MATCH")
    df["exercise_type"]     = df["period_name"].str.split(" ").str[0]
    df = df.drop(columns=["is_official_match"])

    df["period_start_time"] = (
        pd.to_datetime(df["period_start_time"]).dt.tz_localize(None).dt.normalize()
    )
    df["date_of_birth"] = (
        pd.to_datetime(df["date_of_birth"]).dt.tz_localize(None).dt.normalize()
    )
    df["age"] = ((df["period_start_time"] - df["date_of_birth"]).dt.days / 365.25).round(0)
    print(f"Age range: {df['age'].min():.0f}–{df['age'].max():.0f}  |  null ages: {df['age'].isna().sum()}")

    affected = df[df["height"].isna()]
    print(f"Dropping {len(affected)} rows from {affected['player_id'].nunique()} player(s) with missing metadata")
    df = df.dropna(subset=["height", "weight", "position_name_en", "date_of_birth"]).reset_index(drop=True)
    assert df[["player_id", "height", "weight", "position_name_en", "date_of_birth"]].isna().sum().sum() == 0
    print(f"Clean period-level shape: {df.shape}")
    return df


# =============================================================================
# 2. Outlier Treatment
# =============================================================================

def _iqr_cap(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return series.clip(upper=q3 + 3 * (q3 - q1))


def treat_outliers(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """IQR-cap total_distance (always) and the target (if different). Drop weight==200 trialists."""
    cols_to_cap = ["total_distance"]
    if target != "total_distance":
        cols_to_cap.append(target)

    for col in cols_to_cap:
        before  = df[col].sum()
        df[col] = df.groupby(["player_id", "exercise_type"], observed=True)[col].transform(_iqr_cap)
        print(f"IQR cap {col}: {int(before - df[col].sum()):,} units reduced")

    weight_200 = df[df["weight"] == 200]["player_id"].unique()
    for pid in weight_200:
        rows = df[df["player_id"] == pid]
        print(
            f"  Trialist {pid}: {len(rows)} rows  "
            f"{rows['period_start_time'].min().date()} → {rows['period_start_time'].max().date()}"
        )
    before = len(df)
    df = df[~df["player_id"].isin(weight_200)].reset_index(drop=True)
    print(f"Dropped {before - len(df)} trialist rows across {len(weight_200)} players")
    assert df["weight"].max() <= 100
    print(f"Final period-level dataset: {df.shape}")
    return df


# =============================================================================
# 3. Daily Aggregation
# =============================================================================

def aggregate_daily(df: pd.DataFrame, target: str) -> tuple:
    """Aggregate to player-day level. Returns (daily_df, HAS_COLS)."""
    et_dummies_all = pd.get_dummies(df["exercise_type"], prefix="has").astype(int)
    df_enc         = pd.concat([df, et_dummies_all], axis=1)
    HAS_COLS       = sorted(et_dummies_all.columns.tolist())

    et_daily = (
        df_enc.groupby(["player_id", "period_start_time"], observed=True)[HAS_COLS]
        .sum().reset_index()
    )

    agg_dict: dict[str, Any] = {target: (target, "sum")}
    if target != "total_distance":
        agg_dict["total_distance"] = ("total_distance", "sum")
    agg_dict.update({
        "n_periods":        ("activity_id",      "count"),
        "n_exercise_types": ("exercise_type",    "nunique"),
        "height":           ("height",           "first"),
        "weight":           ("weight",           "first"),
        "age":              ("age",              "first"),
        "position":         ("position_name_en", "first"),
    })

    daily = (
        df_enc.groupby(["player_id", "period_start_time"], observed=True)
        .agg(**agg_dict)
        .reset_index()
        .merge(et_daily, on=["player_id", "period_start_time"])
        .rename(columns={"period_start_time": "date"})
        .sort_values(["player_id", "date"])
        .reset_index(drop=True)
    )
    print(f"Daily shape: {daily.shape}")
    return daily, HAS_COLS


# =============================================================================
# 3.1 Spine Fill — Rest Days
# =============================================================================

def spine_fill(daily: pd.DataFrame, target: str, HAS_COLS: list) -> pd.DataFrame:
    """Insert rest-day zero rows so each player has a contiguous date range."""
    zero_cols = [target]
    if target != "total_distance" and "total_distance" in daily.columns:
        zero_cols.append("total_distance")
    zero_cols += ["n_periods", "n_exercise_types"] + HAS_COLS

    static_cols = ["height", "weight", "age", "position"]

    player_date_range = daily.groupby("player_id", observed=True)["date"].agg(["min", "max"])
    spines = [
        pd.DataFrame({"player_id": pid, "date": pd.date_range(row["min"], row["max"], freq="D")})
        for pid, row in player_date_range.iterrows()
    ]
    spine = pd.concat(spines, ignore_index=True)
    spine["player_id"] = spine["player_id"].astype(daily["player_id"].dtype)

    daily = (
        spine.merge(daily, on=["player_id", "date"], how="left")
        .sort_values(["player_id", "date"]).reset_index(drop=True)
    )
    daily[zero_cols]   = daily[zero_cols].fillna(0)
    daily[static_cols] = (
        daily.groupby("player_id", observed=True)[static_cols]
        .transform(lambda s: s.ffill().bfill())
    )
    print(f"Daily shape after spine fill: {daily.shape}  |  nulls: {daily.isna().sum().sum()}")
    return daily


# =============================================================================
# 4. Feature Engineering
# =============================================================================

def add_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week (integer 0 = Mon … 6 = Sun)."""
    d = daily_df.copy()
    d["day_of_week"] = d["date"].dt.dayofweek
    return d


def encode_dow(df: pd.DataFrame) -> pd.DataFrame:
    """OHE day_of_week → dow_0 … dow_6 with fixed categories."""
    dow     = pd.Categorical(df["day_of_week"], categories=range(7))
    dummies = pd.get_dummies(dow, prefix="dow").astype(int)
    dummies.index = df.index
    return pd.concat([df.drop(columns=["day_of_week"]), dummies], axis=1)


# =============================================================================
# 5. Target Transform & Feature Scaling
# =============================================================================

def scale_train(train: pd.DataFrame, target: str) -> tuple:
    """log1p-transform target, drop total_distance covariate, MinMaxScale features.

    Returns (train_scaled, scaler, feature_cols).
    """
    train = train.copy()
    train[target] = np.log1p(train[target])

    if target != "total_distance" and "total_distance" in train.columns:
        train = train.drop(columns=["total_distance"])

    NON_FEATURES = {target, "total_distance", "player_id", "position", "date",
                    "period_start_time", "date_of_birth", "day_of_week"}
    feature_cols = [
        c for c in train.columns
        if c not in NON_FEATURES and train[c].dtype != object
    ]

    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])

    print(f"{len(feature_cols)} features: {feature_cols}")
    print(f"Train null count: {train.isna().sum().sum()}")
    return train, scaler, feature_cols


def prepare_test(test_df: pd.DataFrame, target: str,
                 scaler: MinMaxScaler, feature_cols: list) -> pd.DataFrame:
    """Apply the same transforms fitted on train. Call at evaluation time only."""
    t = encode_dow(add_features(test_df))
    t[target] = np.log1p(t[target])
    if target != "total_distance" and "total_distance" in t.columns:
        t = t.drop(columns=["total_distance"])
    t[feature_cols] = scaler.transform(t[feature_cols])
    return t


# =============================================================================
# 6. Train / Test Split  (random — matches notebook split_data)
# =============================================================================

def split_data(daily: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    train_base, test_base = train_test_split(daily, test_size=test_size,
                                             random_state=random_state)
    train_base = train_base.reset_index(drop=True)
    test_base  = test_base.reset_index(drop=True)
    print(f"Train rows: {len(train_base)}  |  Test rows (held out): {len(test_base)}")
    return train_base, test_base


# =============================================================================
# 7. run_pipeline  (matches notebooks/datapipeline.py run_pipeline exactly)
# =============================================================================

def run_pipeline(target: str, test_size: float = 0.2,
                 random_state: int = 42) -> dict:
    """Run the full preprocessing pipeline for the given target column.

    Returns
    -------
    dict with keys:
        train, train_base, test_base  — DataFrames
        X_tr_np, y_tr_np             — numpy arrays (scaled features, log1p target)
        scaler                        — fitted MinMaxScaler
        feature_cols                  — list[str]
        HAS_COLS                      — list[str] of has_* column names
        daily                         — full spine-filled daily DataFrame
    """
    df = load_data()
    df = clean_data(df)
    df = treat_outliers(df, target)

    daily, HAS_COLS = aggregate_daily(df, target)
    daily           = spine_fill(daily, target, HAS_COLS)

    train_base, test_base = split_data(daily, test_size, random_state)

    train = encode_dow(add_features(train_base))
    print(f"Train shape after feature engineering: {train.shape}")

    train, scaler, feature_cols = scale_train(train, target)

    return {
        "train":        train,
        "train_base":   train_base,
        "test_base":    test_base,
        "X_tr_np":      train[feature_cols].to_numpy(),
        "y_tr_np":      train[target].to_numpy(),
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "HAS_COLS":     HAS_COLS,
        "daily":        daily,
    }


# =============================================================================
# 8. build_full_daily  (all 3 metrics — for saving daily.parquet used by the app)
# =============================================================================

def build_full_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all three load metrics to player-day level for the app's daily.parquet."""
    et_dummies = pd.get_dummies(df["exercise_type"], prefix="has").astype(int)
    df_enc     = pd.concat([df[["player_id", "period_start_time"]], et_dummies], axis=1)
    et_daily   = (
        df_enc.groupby(["player_id", "period_start_time"], observed=True)
        .sum().reset_index()
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

    for col in ["has_G", "has_TAC", "has_BP", "has_TEC", "has_MATCH"]:
        if col not in daily.columns:
            daily[col] = 0

    zero_cols = (
        ["total_distance", "accelerations", "sprint_distance", "n_periods", "n_exercise_types"]
        + [c for c in daily.columns if c.startswith("has_")]
    )
    static_cols = ["height", "weight", "age", "position"]

    player_date_range = daily.groupby("player_id", observed=True)["date"].agg(["min", "max"])
    spines = [
        pd.DataFrame({"player_id": pid, "date": pd.date_range(row["min"], row["max"], freq="D")})
        for pid, row in player_date_range.iterrows()
    ]
    spine = pd.concat(spines, ignore_index=True)
    spine["player_id"] = spine["player_id"].astype(daily["player_id"].dtype)

    daily = (
        spine.merge(daily, on=["player_id", "date"], how="left")
        .sort_values(["player_id", "date"]).reset_index(drop=True)
    )
    daily[zero_cols]   = daily[zero_cols].fillna(0)
    daily[static_cols] = (
        daily.groupby("player_id", observed=True)[static_cols]
        .transform(lambda s: s.ffill().bfill())
    )
    print(f"Full daily shape: {daily.shape}  |  nulls: {daily.isna().sum().sum()}")
    return daily
