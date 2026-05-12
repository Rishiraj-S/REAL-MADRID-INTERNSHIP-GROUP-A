# data/

Data is organized using the Cookiecutter Data Science convention:

| Directory | Purpose |
|---|---|
| `raw/` | Immutable source data kept local and out of git |
| `external/` | Third-party data kept local and out of git |
| `interim/` | Intermediate transformed data |
| `processed/` | Final model-ready datasets |

The repository keeps `data/data_acute_vs_chronic.zip` as the bootstrap archive.
The notebook pipeline reads `data/raw/data_acute_vs_chronic.csv` and extracts it
from the ZIP if the CSV is not already present locally.
