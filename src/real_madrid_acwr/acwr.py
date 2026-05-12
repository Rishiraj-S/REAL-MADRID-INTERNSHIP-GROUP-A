"""
EWMA-based Acute:Chronic Workload Ratio (ACWR) computation.

Conventions:
- Acute window:   7-day EWMA, alpha = 2 / (7+1) = 0.25
- Chronic window: 28-day EWMA, alpha = 2 / (28+1) ≈ 0.0690
- Uncoupled: acute and chronic are computed independently from the same load series
  (acute is NOT subtracted from chronic).
- Rest days: load = 0. Rest days contribute to EWMA decay; missing days do not exist
  on the daily grid.
- Warmup: ACWR is masked (NaN) for the first `warmup_days` rows of each player's
  history because chronic load isn't stable yet. Default: warmup = chronic_window.

References:
- Definition: https://www.scienceforsport.com/acutechronic-workload-ratio
- EWMA formulation: Williams et al., 2017. "Better way to determine the
  acute:chronic workload ratio?"
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Default windows (in days)
ACUTE_WINDOW = 7
CHRONIC_WINDOW = 28


def _ewma_alpha(window: int) -> float:
    """Convert a window size to an EWMA smoothing factor.
    
    alpha = 2 / (N + 1) is the standard pandas/sport-science convention.
    """
    return 2.0 / (window + 1.0)


def compute_acwr(
    daily_loads: np.ndarray | pd.Series | Sequence[float],
    acute_window: int = ACUTE_WINDOW,
    chronic_window: int = CHRONIC_WINDOW,
    warmup_days: int | None = None,
) -> pd.DataFrame:
    """Compute EWMA-based ACWR for a single player's daily load series.
    
    Parameters
    ----------
    daily_loads : array-like, length n_days
        One load value per calendar day. Rest days should be 0 (not NaN, not skipped).
        The series should be on a complete daily grid — no gaps.
    acute_window : int, default 7
        Acute EWMA window in days.
    chronic_window : int, default 28
        Chronic EWMA window in days.
    warmup_days : int or None, default None
        Number of leading days to mask as NaN in the output ACWR. If None, uses
        `chronic_window` (the standard convention — chronic isn't stable until the
        window has been filled).
    
    Returns
    -------
    pd.DataFrame, length n_days, columns:
        - load:    input load (passed through)
        - acute:   7-day EWMA of load
        - chronic: 28-day EWMA of load
        - acwr:    acute / chronic, with warmup masked and chronic==0 → NaN
    """
    if warmup_days is None:
        warmup_days = chronic_window
    
    # Coerce to pandas Series to use ewm
    if isinstance(daily_loads, pd.Series):
        loads = pd.Series(daily_loads.values, index=daily_loads.index, dtype=float)
    else:
        loads = pd.Series(daily_loads, dtype=float)
    
    # Validation
    if loads.isna().any():
        raise ValueError(
            "daily_loads contains NaN. Rest days must be encoded as 0, not NaN. "
            "Missing calendar days should be filled before calling compute_acwr."
        )
    if (loads < 0).any():
        raise ValueError("daily_loads contains negative values; loads must be ≥ 0.")
    
    # EWMA — uncoupled: both computed from the same raw series
    acute = loads.ewm(alpha=_ewma_alpha(acute_window), adjust=False).mean()
    chronic = loads.ewm(alpha=_ewma_alpha(chronic_window), adjust=False).mean()
    
    # Ratio. Mask where chronic is 0 (would otherwise divide by zero).
    acwr = acute / chronic
    acwr = acwr.where(chronic > 0, np.nan)
    
    # Warmup mask: chronic isn't trustworthy until enough history accumulates.
    if warmup_days > 0 and len(loads) > warmup_days:
        acwr.iloc[:warmup_days] = np.nan
    elif warmup_days >= len(loads):
        acwr[:] = np.nan
    
    return pd.DataFrame({
        "load": loads,
        "acute": acute,
        "chronic": chronic,
        "acwr": acwr,
    }, index=loads.index)


def compute_acwr_for_all_players(
    daily_grid: pd.DataFrame,
    metric_col: str,
    player_col: str = "player_id",
    date_col: str = "period_start_time",
    acute_window: int = ACUTE_WINDOW,
    chronic_window: int = CHRONIC_WINDOW,
    warmup_days: int | None = None,
) -> pd.DataFrame:
    """Compute ACWR per player over a long-format daily grid.
    
    Each player must have a complete daily grid (one row per calendar day,
    rest days encoded as load = 0). The function does NOT fill missing dates;
    the caller must supply a complete grid.
    
    Parameters
    ----------
    daily_grid : pd.DataFrame
        Long-format with one row per (player, date). Rest days have metric = 0.
    metric_col : str
        Column name of the load metric (e.g., 'total_distance', 'acc_total', 'vel_total').
    player_col : str, default 'player_id'
    date_col : str, default 'period_start_time'
    acute_window, chronic_window, warmup_days : see compute_acwr
    
    Returns
    -------
    pd.DataFrame
        Same as input, plus columns: acute, chronic, acwr.
        Indexed sequentially after sort by (player, date).
    """
    if metric_col not in daily_grid.columns:
        raise KeyError(f"metric_col '{metric_col}' not in daily_grid")
    if player_col not in daily_grid.columns:
        raise KeyError(f"player_col '{player_col}' not in daily_grid")
    if date_col not in daily_grid.columns:
        raise KeyError(f"date_col '{date_col}' not in daily_grid")
    
    df = daily_grid.sort_values([player_col, date_col]).reset_index(drop=True).copy()
    
    acute_list = []
    chronic_list = []
    acwr_list = []
    
    for _, group in df.groupby(player_col, observed=True, sort=False):
        result = compute_acwr(
            group[metric_col].to_numpy(dtype=float, na_value=np.nan),
            acute_window=acute_window,
            chronic_window=chronic_window,
            warmup_days=warmup_days,
        )
        acute_list.append(pd.Series(result["acute"].values, index=group.index))
        chronic_list.append(pd.Series(result["chronic"].values, index=group.index))
        acwr_list.append(pd.Series(result["acwr"].values, index=group.index))
    
    df["acute"] = pd.concat(acute_list).sort_index()
    df["chronic"] = pd.concat(chronic_list).sort_index()
    df["acwr"] = pd.concat(acwr_list).sort_index()
    
    return df


def compute_acwr_with_forecast(
    historical_loads: np.ndarray | pd.Series | Sequence[float],
    forecast_loads: np.ndarray | pd.Series | Sequence[float],
    acute_window: int = ACUTE_WINDOW,
    chronic_window: int = CHRONIC_WINDOW,
    warmup_days: int | None = None,
) -> pd.DataFrame:
    """Compute ACWR over a stitched (historical + forecast) load series.
    
    This is the dashboard's primary use case: take a player's actual past
    loads, append the planned/predicted future loads, and compute the full
    ACWR series including the forecast horizon. The EWMA naturally carries
    history into the forecast region.
    
    Parameters
    ----------
    historical_loads : array-like, length n_history
        Past actual daily loads (rest days = 0).
    forecast_loads : array-like, length n_forecast
        Future predicted daily loads (rest days = 0).
    acute_window, chronic_window, warmup_days : see compute_acwr
    
    Returns
    -------
    pd.DataFrame, length n_history + n_forecast, columns:
        - load:        stitched load series
        - acute, chronic, acwr: as in compute_acwr
        - is_forecast: True for the last n_forecast rows, False for history
    """
    historical = np.asarray(historical_loads, dtype=float)
    forecast = np.asarray(forecast_loads, dtype=float)
    
    stitched = np.concatenate([historical, forecast])
    result = compute_acwr(
        stitched,
        acute_window=acute_window,
        chronic_window=chronic_window,
        warmup_days=warmup_days,
    )
    
    is_forecast = np.zeros(len(stitched), dtype=bool)
    is_forecast[len(historical):] = True
    result["is_forecast"] = is_forecast
    
    return result


# Optional: ACWR risk zones (sport-science convention).
# These are commonly cited but not universally agreed; treat as guidelines.
ACWR_ZONES = {
    "undertraining": (0.0, 0.8),
    "sweet_spot":    (0.8, 1.3),
    "danger":        (1.5, np.inf),
}


def classify_acwr_zone(acwr_value: float) -> str:
    """Classify a single ACWR value into a risk zone.
    
    Returns one of: 'undertraining', 'optimal', 'caution', 'danger', or 'unknown'
    (for NaN). The 1.3-1.5 band is 'caution' — elevated but not yet dangerous.
    """
    if pd.isna(acwr_value):
        return "unknown"
    if acwr_value < 0.8:
        return "undertraining"
    elif acwr_value < 1.3:
        return "optimal"
    elif acwr_value < 1.5:
        return "caution"
    else:
        return "danger"
