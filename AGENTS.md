# AGENTS.md

This file provides guidance to Codex and other coding agents when working with
this repository.

## Project

End-to-end ACWR (Acute:Chronic Workload Ratio) prediction tool for Real Madrid's
fitness coaching staff. Coaches enter a planned 15-day session composition and
see predicted ACWR trajectories for each squad player across three load metrics.
The interactive app is a Streamlit application (`app.py`).

The repository keeps `data/data_acute_vs_chronic.zip` as the bootstrap archive.
The extracted raw CSV lives at `data/raw/data_acute_vs_chronic.csv` and is
ignored by git.

## Repository layout

```text
app.py                                      <- Streamlit app (run: streamlit run app.py)
train_models.py                             <- Compatibility wrapper for model training
pyproject.toml                              <- Package metadata, dependencies, tool config
Makefile                                    <- Common local commands

src/real_madrid_acwr/acwr.py                <- EWMA-ACWR utilities
src/real_madrid_acwr/config.py              <- Shared project paths
src/real_madrid_acwr/modeling/train.py      <- Model training implementation

notebooks/data_pipeline.ipynb               <- Produces data/processed/model_data.parquet
notebooks/acc_total.ipynb                   <- EDA + model exploration (acc_total)
notebooks/total_distance.ipynb              <- EDA + model exploration (total_distance)
notebooks/vel_total.ipynb                   <- EDA + model exploration (vel_total)

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
make run          # streamlit run app.py
make train        # python train_models.py
```

Equivalent direct commands:

```bash
ruff check app.py train_models.py src tests utils
mypy
pytest
streamlit run app.py
python train_models.py
```

## Model artifacts

Each of the three targets has three files:

| File | Content |
|---|---|
| `models/xgboost/{target}/model.json` | XGBoost `Booster` in native JSON format |
| `models/xgboost/{target}/feature_cols.pkl` | Ordered list of 45 feature names |
| `models/xgboost/{target}/transform.pkl` | Dict with `type` (`log1p` or `none`) for inverse-transform |

The app loads models with `xgb.Booster().load_model()` and does not load sklearn
Pipeline pickles at runtime.

## Important conventions

- Keep production Python code under `src/real_madrid_acwr/`.
- Keep `utils/acwr.py` only as a compatibility wrapper for older imports.
- Keep private raw data out of git.
- Check `data_decisions.md` before changing pipeline or modeling methodology.
- Prefer small, test-backed changes over notebook-only edits for production
  behavior.
