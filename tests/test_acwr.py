import numpy as np
import pandas as pd
import pytest

from real_madrid_acwr.acwr import (
    classify_acwr_zone,
    compute_acwr,
    compute_acwr_for_all_players,
    compute_acwr_with_forecast,
)


def test_compute_acwr_masks_warmup_and_keeps_expected_columns() -> None:
    loads = [0.0, 10.0, 20.0, 30.0, 40.0]

    result = compute_acwr(loads, acute_window=2, chronic_window=4, warmup_days=2)

    assert list(result.columns) == ["load", "acute", "chronic", "acwr"]
    assert result["load"].tolist() == loads
    assert result["acwr"].iloc[:2].isna().all()
    assert result["acwr"].iloc[2:].notna().all()


def test_compute_acwr_rejects_negative_loads() -> None:
    with pytest.raises(ValueError, match="negative"):
        compute_acwr(np.array([1.0, -1.0, 2.0]))


def test_compute_acwr_with_forecast_marks_forecast_rows() -> None:
    result = compute_acwr_with_forecast(
        historical_loads=np.array([10.0, 20.0, 30.0]),
        forecast_loads=np.array([40.0, 50.0]),
        warmup_days=0,
    )

    assert len(result) == 5
    assert result["is_forecast"].tolist() == [False, False, False, True, True]
    assert result["load"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_compute_acwr_for_all_players_sorts_and_computes_per_player() -> None:
    grid = pd.DataFrame(
        {
            "player_id": [2, 1, 1, 2],
            "period_start_time": pd.to_datetime(
                ["2025-01-02", "2025-01-01", "2025-01-02", "2025-01-01"]
            ),
            "total_distance": [20.0, 10.0, 30.0, 40.0],
        }
    )

    result = compute_acwr_for_all_players(
        grid,
        metric_col="total_distance",
        warmup_days=0,
    )

    assert result["player_id"].tolist() == [1, 1, 2, 2]
    assert result["period_start_time"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-01",
        "2025-01-02",
    ]
    assert result[["acute", "chronic", "acwr"]].notna().all().all()


@pytest.mark.parametrize(
    ("value", "zone"),
    [
        (np.nan, "unknown"),
        (0.79, "undertraining"),
        (0.8, "optimal"),
        (1.29, "optimal"),
        (1.3, "caution"),
        (1.49, "caution"),
        (1.5, "danger"),
    ],
)
def test_classify_acwr_zone_boundaries(value: float, zone: str) -> None:
    assert classify_acwr_zone(value) == zone
