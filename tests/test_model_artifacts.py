import pickle
from pathlib import Path

TARGETS = ("acc_total", "total_distance", "vel_total")
BASE_FEATURES = {
    "height",
    "weight",
    "age",
    "has_G",
    "has_TAC",
    "has_BP",
    "has_TEC",
    "has_MATCH",
    "n_session_types",
    "pos_central_back",
    "pos_central_midfielder",
    "pos_forward",
    "pos_full_back",
    "pos_winger",
    "days_since_start",
    "days_since_last_activity",
    "days_since_last_match",
}


def test_model_artifact_triads_exist() -> None:
    models_dir = Path("models") / "xgboost"

    for target in TARGETS:
        assert (models_dir / target / "model.json").is_file()
        assert (models_dir / target / "feature_cols.pkl").is_file()
        assert (models_dir / target / "transform.pkl").is_file()


def test_feature_columns_match_current_contract() -> None:
    models_dir = Path("models") / "xgboost"

    for target in TARGETS:
        with (models_dir / target / "feature_cols.pkl").open("rb") as file:
            feature_cols = pickle.load(file)

        assert len(feature_cols) == 45
        assert len(set(feature_cols)) == len(feature_cols)
        assert BASE_FEATURES.issubset(feature_cols)
        assert len([col for col in feature_cols if col.startswith("pid_")]) == 28


def test_transform_metadata_matches_current_contract() -> None:
    models_dir = Path("models") / "xgboost"
    expected = {
        "acc_total": {"type": "none", "loss": "tweedie"},
        "total_distance": {"type": "log1p", "inverse": "expm1"},
        "vel_total": {"type": "none", "loss": "mse"},
    }

    for target, expected_transform in expected.items():
        with (models_dir / target / "transform.pkl").open("rb") as file:
            transform = pickle.load(file)

        assert transform == expected_transform
