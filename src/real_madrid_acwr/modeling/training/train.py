"""
train.py — Orchestrator: saves daily.parquet then trains all three models.

Run via:
    python train_models.py
"""

from __future__ import annotations

from real_madrid_acwr.config import DAILY_PARQUET, DATA_DIR
from real_madrid_acwr.modeling.datapipeline import (
    build_full_daily,
    clean_data,
    load_data,
    treat_outliers,
)
from real_madrid_acwr.modeling.training.acceleration_model_train import main as train_accelerations
from real_madrid_acwr.modeling.training.sprint_distance_model_train import (
    main as train_sprint_distance,
)
from real_madrid_acwr.modeling.training.total_distance_model_train import (
    main as train_total_distance,
)


def main() -> None:
    print("Building daily.parquet for app inference…")
    df    = load_data()
    df    = clean_data(df)
    df    = treat_outliers(df, "total_distance")
    daily = build_full_daily(df)
    DAILY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PARQUET, index=False)
    print(f"Saved daily.parquet → {DAILY_PARQUET.relative_to(DATA_DIR.parent)}\n")

    train_accelerations()
    train_sprint_distance()
    train_total_distance()


if __name__ == "__main__":
    main()
