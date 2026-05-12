"""Compatibility wrapper for ACWR utilities.

New code should import from `real_madrid_acwr.acwr`.
"""

from real_madrid_acwr.acwr import (  # noqa: F401
    ACUTE_WINDOW,
    ACWR_ZONES,
    CHRONIC_WINDOW,
    classify_acwr_zone,
    compute_acwr,
    compute_acwr_for_all_players,
    compute_acwr_with_forecast,
)
