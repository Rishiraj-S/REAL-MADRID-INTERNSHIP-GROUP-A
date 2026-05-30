# AGENTS.md

This file provides guidance to Codex and other coding agents when working with
this repository.

## Project

End-to-end ACWR (Acute:Chronic Workload Ratio) prediction tool for Real Madrid's
fitness coaching staff. Coaches enter a planned 15-day session composition and
see predicted ACWR trajectories for each squad player across three load metrics.
The interactive app is a Streamlit application (`main.py`).

The repository keeps `data/data_acute_vs_chronic.zip` as the bootstrap archive.
The extracted raw CSV lives at `data/raw/data_acute_vs_chronic.csv` and is
ignored by git.

## Repository layout

```text
main.py                                     <- Streamlit app (run: streamlit run main.py)
train_models.py                             <- Compatibility wrapper for model training
pyproject.toml                              <- Package metadata, dependencies, tool config
Makefile                                    <- Common local commands

src/real_madrid_acwr/acwr.py                <- EWMA-ACWR utilities
src/real_madrid_acwr/config.py              <- Shared project paths
src/real_madrid_acwr/modeling/train.py      <- Model training implementation

notebooks/data_pipeline.ipynb               <- Produces data/processed/model_data.parquet (legacy)
notebooks/acceleration_model.ipynb          <- EDA + model exploration (accelerations)
notebooks/sprint_distance_model.ipynb       <- EDA + model exploration (sprint_distance)
notebooks/total_distance.ipynb              <- EDA + model exploration (total_distance)

data/processed/model_data.parquet           <- Bridge between pipeline and app/models
models/                                     <- XGBoost JSON artifacts + pkl metadata
static/img/                                 <- App image assets
references/                                 <- Reference documents and briefs
reports/figures/                            <- Generated report figures
tests/                                      <- Pytest suite

data_decisions.md                           <- Cleaning/methodology decision log
```

## Environment

`pyproject.toml` is the single dependency source. Do not reintroduce
`requirements.txt` unless the project intentionally moves back to that workflow.

```bash
python -m pip install -e ".[dev]"
```

Notebook-only dependencies are separate:

```bash
python -m pip install -e ".[notebooks]"
```

## Common commands

```bash
make quality      # ruff + mypy + pytest
make run          # streamlit run main.py
make train        # python train_models.py
```

Equivalent direct commands:

```bash
ruff check main.py train_models.py src tests utils
mypy
pytest
streamlit run main.py
python train_models.py
```

## Model artifacts

Targets: `accelerations`, `sprint_distance`, `total_distance`.

Each target has one artifact file:

| File | Content |
|---|---|
| `models/xgboost/{target}/bundle.joblib` | `joblib`-serialised dict: `model` (XGBRegressor), `scaler` (MinMaxScaler), `feature_cols` (list[str]), `ewma_spans` (dict) |

The app loads bundles with `joblib.load()`. Prediction uses `model.predict(scaler.transform(X))` with a log1p inverse transform (`np.expm1`).

A processed daily history file is also required at inference time:

| File | Content |
|---|---|
| `data/processed/daily.parquet` | Raw daily frame per player with all load columns and session flags |

Both files are produced by `python train_models.py`.

## Important conventions

- Keep production Python code under `src/real_madrid_acwr/`.
- Keep `utils/acwr.py` only as a compatibility wrapper for older imports.
- Keep private raw data out of git.
- Check `data_decisions.md` before changing pipeline or modeling methodology.
- Prefer small, test-backed changes over notebook-only edits for production
  behavior.
