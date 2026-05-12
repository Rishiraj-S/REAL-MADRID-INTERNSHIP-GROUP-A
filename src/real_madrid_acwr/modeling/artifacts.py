"""Load and validate deployable model artifacts."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from real_madrid_acwr.config import MODEL_ARTIFACTS_DIR

TARGETS = ("total_distance", "acc_total", "vel_total")

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

VALID_TRANSFORMS = (
    {"type": "none", "loss": "tweedie"},
    {"type": "none", "loss": "mse"},
    {"type": "log1p", "inverse": "expm1"},
)


class ArtifactLoadError(RuntimeError):
    """Raised when deployable model artifacts are missing or invalid."""


@dataclass(frozen=True)
class ModelArtifact:
    """Validated artifacts for one target model."""

    target: str
    model: xgb.Booster
    feature_cols: list[str]
    transform: dict[str, str]


def load_model_artifacts(
    base_dir: Path = MODEL_ARTIFACTS_DIR,
) -> dict[str, ModelArtifact]:
    """Load all target model artifacts or raise a detailed contract error."""
    errors: list[str] = []
    artifacts: dict[str, ModelArtifact] = {}

    for target in TARGETS:
        try:
            artifacts[target] = _load_target_artifact(base_dir, target)
        except ArtifactLoadError as exc:
            errors.append(str(exc))

    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ArtifactLoadError(f"Model artifact contract failed:\n{joined}")

    return artifacts


def _load_target_artifact(base_dir: Path, target: str) -> ModelArtifact:
    artifact_dir = base_dir / target
    model_path = artifact_dir / "model.json"
    feature_cols_path = artifact_dir / "feature_cols.pkl"
    transform_path = artifact_dir / "transform.pkl"

    missing = [
        path.relative_to(base_dir)
        for path in (model_path, feature_cols_path, transform_path)
        if not path.is_file()
    ]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise ArtifactLoadError(f"{target}: missing artifact file(s): {missing_list}")

    model = xgb.Booster()
    try:
        model.load_model(str(model_path))
    except xgb.core.XGBoostError as exc:
        raise ArtifactLoadError(f"{target}: invalid XGBoost model: {exc}") from exc

    feature_cols = _load_pickle(feature_cols_path, target, "feature columns")
    transform = _load_pickle(transform_path, target, "transform metadata")

    validated_feature_cols = _validate_feature_cols(feature_cols, target)
    validated_transform = _validate_transform(transform, target)

    return ModelArtifact(
        target=target,
        model=model,
        feature_cols=validated_feature_cols,
        transform=validated_transform,
    )


def _load_pickle(path: Path, target: str, description: str) -> Any:
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise ArtifactLoadError(f"{target}: could not load {description}: {exc}") from exc


def _validate_feature_cols(feature_cols: Any, target: str) -> list[str]:
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ArtifactLoadError(f"{target}: feature columns must be a non-empty list")
    if not all(isinstance(col, str) for col in feature_cols):
        raise ArtifactLoadError(f"{target}: every feature column must be a string")
    if len(feature_cols) != 45:
        raise ArtifactLoadError(f"{target}: expected 45 feature columns, found {len(feature_cols)}")
    if len(set(feature_cols)) != len(feature_cols):
        raise ArtifactLoadError(f"{target}: feature columns contain duplicates")

    missing_base = sorted(BASE_FEATURES.difference(feature_cols))
    if missing_base:
        raise ArtifactLoadError(f"{target}: missing base feature(s): {', '.join(missing_base)}")

    pid_cols = [col for col in feature_cols if col.startswith("pid_")]
    if len(pid_cols) != 28:
        raise ArtifactLoadError(f"{target}: expected 28 pid_* columns, found {len(pid_cols)}")

    return feature_cols


def _validate_transform(transform: Any, target: str) -> dict[str, str]:
    if not isinstance(transform, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in transform.items()
    ):
        raise ArtifactLoadError(f"{target}: transform metadata must be a dict[str, str]")

    if transform not in VALID_TRANSFORMS:
        raise ArtifactLoadError(f"{target}: unsupported transform metadata: {transform}")

    return transform
