# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

End-to-end ACWR (Acute:Chronic Workload Ratio) prediction tool for Real Madrid's fitness coaching staff. The goal is to let coaches enter a planned session composition and see predicted 15-day ACWR trajectories for each squad player. Development is in **Databricks**; the canonical pipeline lives in `data_engineering.py` for local iteration.

The raw dataset (`data_acute_vs_chronic.csv`) is gitignored and must be present locally to run the pipeline.

## Running the pipeline

```bash
python data_engineering.py          # runs run_pipeline(), saves full_grid.parquet
```

To run interactively, call `run_pipeline()` directly:

```python
from data_engineering import run_pipeline
full_grid = run_pipeline("data_acute_vs_chronic.csv")
```

No test framework is configured. Correctness checks are inline `assert` statements within each step function.

## Pipeline architecture (`data_engineering.py`)

Nine sequential steps, each a standalone function, wired together by `run_pipeline()`:

| Step | Function | Input → Output |
|---|---|---|
| 1 | `load_raw` | CSV → raw DataFrame (3,903 rows × 13 cols) |
| 2 | `fix_dtypes` | cast `is_official_match` → bool, `player_id` → category |
| 3 | `parse_exercise_type` | extract `exercise_type` prefix from `period_name`; NaN → `'MATCH'` |
| 4 | `parse_datetimes` | parse timestamps, extract `date`, compute float `age` |
| 5 | `drop_missing_metadata` | drop player 50333 (no metadata, 19 rows) |
| 6 | `fix_outlier_total_distance` + `fix_suspect_players` | replace outlier with player median; drop 5 trialists; null placeholder height/weight for player 15795 |
| 7 | `aggregate_to_daily` | period rows → one row per `(player_id, date)`; load cols renamed to `total_distance`, `acc_total`, `vel_total`; `exercise_types` stored as `frozenset` |
| 8 | `expand_to_full_grid` + `add_modelable_flag` | continuous daily grid, zero-filled rest days, `is_rest` / `rest_streak` flags, `is_modelable` (≥28 active days) |
| 9 | `add_acwr_to_grid` | EWMA-ACWR for all 3 metrics; warmup 28 days masked to NaN |

**Output shape:** 8,872 rows × 27 columns; 29 players; 76.1% rest days.

## Key data facts

- **Granularity shift:** raw data is one row per *period* (drill); pipeline aggregates to one row per *player-day*.
- **Load metrics** (three, computed independently): `total_distance` (m), `acc_total` (high-intensity acceleration count), `vel_total` (high-speed running, m).
- **EWMA parameters:** λ_acute = 2/(7+1) = 0.25; λ_chronic = 2/(28+1) ≈ 0.069. Initialized from 0.
- **Modelable players:** 20 of 29 have ≥28 active days and can support model training. The other 9 need a position-based fallback at inference time.
- **`exercise_types` column** is a `frozenset` (e.g., `{'G', 'TAC', 'BP'}`). Serialize to string with `'+'.join(sorted(s))` before writing Parquet/CSV.
- Safe ACWR range: 0.8–1.3; elevated-risk threshold: ≥1.5.

## What comes next

The pipeline is complete. Remaining work:
1. **Load prediction model** — predict `total_distance`, `acc_total`, `vel_total` for a given player-day given planned `exercise_types` composition and player history features.
2. **15-day roll-forward simulation** — feed predicted loads back through `compute_acwr_for_player` to project ACWR trajectories.
3. **Interactive app** — coaches select session templates → app shows predicted ACWR per player.

## Decision log

`data_decisions.md` documents every cleaning and methodology choice (player exclusions, outlier treatment, EWMA formula, warmup masking). Check it before changing pipeline logic. Open items are marked `[OPEN]`.
