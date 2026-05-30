"""Load and validate deployable model artifacts (forecast bundles)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from real_madrid_acwr.config import MODEL_ARTIFACTS_DIR

TARGETS = ("total_distance", "accelerations", "sprint_distance")

EWMA_SPANS = {"acute": 7, "chronic": 28}


class ArtifactLoadError(RuntimeError):
    """Raised when deployable model artifacts are missing or invalid."""


@dataclass(frozen=True)
class ForecastBundle:
    """Validated forecast bundle for one target model."""

    target: str
    model: Any
    scaler: Any
    feature_cols: list[str]
    ewma_spans: dict[str, int]


def load_model_artifacts(
    base_dir: Path = MODEL_ARTIFACTS_DIR,
) -> dict[str, ForecastBundle]:
    """Load all target forecast bundles or raise a detailed contract error."""
    errors: list[str] = []
    bundles: dict[str, ForecastBundle] = {}

    for target in TARGETS:
        try:
            bundles[target] = _load_target_bundle(base_dir, target)
        except ArtifactLoadError as exc:
            errors.append(str(exc))

    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ArtifactLoadError(f"Model artifact contract failed:\n{joined}")

    return bundles


def _load_target_bundle(base_dir: Path, target: str) -> ForecastBundle:
    bundle_path = base_dir / target / "bundle.joblib"

    if not bundle_path.is_file():
        raise ArtifactLoadError(
            f"{target}: missing bundle file: {bundle_path.relative_to(base_dir)}"
        )

    try:
        raw = joblib.load(bundle_path)
    except Exception as exc:
        raise ArtifactLoadError(f"{target}: could not load bundle: {exc}") from exc

    for key in ("model", "scaler", "feature_cols", "ewma_spans"):
        if key not in raw:
            raise ArtifactLoadError(f"{target}: bundle missing key '{key}'")

    feature_cols = raw["feature_cols"]
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ArtifactLoadError(f"{target}: feature_cols must be a non-empty list")
    if not all(isinstance(c, str) for c in feature_cols):
        raise ArtifactLoadError(f"{target}: every feature_col must be a string")
    if len(set(feature_cols)) != len(feature_cols):
        raise ArtifactLoadError(f"{target}: feature_cols contains duplicates")

    ewma_spans = raw["ewma_spans"]
    if not isinstance(ewma_spans, dict) or set(ewma_spans) != {"acute", "chronic"}:
        raise ArtifactLoadError(f"{target}: ewma_spans must be a dict with keys 'acute' and 'chronic'")

    return ForecastBundle(
        target=target,
        model=raw["model"],
        scaler=raw["scaler"],
        feature_cols=feature_cols,
        ewma_spans=ewma_spans,
    )
