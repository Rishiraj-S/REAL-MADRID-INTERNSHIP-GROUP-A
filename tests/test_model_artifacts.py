import shutil
from pathlib import Path

import pytest

from real_madrid_acwr.config import MODEL_ARTIFACTS_DIR
from real_madrid_acwr.modeling.artifacts import (
    TARGETS,
    ArtifactLoadError,
    ForecastBundle,
    load_model_artifacts,
)


def _copy_artifacts(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "xgboost"
    shutil.copytree(MODEL_ARTIFACTS_DIR, artifact_dir)
    return artifact_dir


def test_current_model_artifacts_load_successfully() -> None:
    bundles = load_model_artifacts()

    assert set(bundles) == set(TARGETS)
    for target, bundle in bundles.items():
        assert isinstance(bundle, ForecastBundle)
        assert bundle.target == target
        assert isinstance(bundle.feature_cols, list)
        assert len(bundle.feature_cols) > 0
        assert len(set(bundle.feature_cols)) == len(bundle.feature_cols)
        assert set(bundle.ewma_spans) == {"acute", "chronic"}


def test_missing_target_directory_raises_artifact_error(tmp_path: Path) -> None:
    artifact_dir = _copy_artifacts(tmp_path)
    shutil.rmtree(artifact_dir / "accelerations")

    with pytest.raises(ArtifactLoadError, match="accelerations: missing bundle file"):
        load_model_artifacts(artifact_dir)


def test_missing_bundle_file_raises_artifact_error(tmp_path: Path) -> None:
    artifact_dir = _copy_artifacts(tmp_path)
    (artifact_dir / "total_distance" / "bundle.joblib").unlink()

    with pytest.raises(ArtifactLoadError, match="total_distance: missing bundle file"):
        load_model_artifacts(artifact_dir)


def test_corrupted_bundle_raises_artifact_error(tmp_path: Path) -> None:
    artifact_dir = _copy_artifacts(tmp_path)
    (artifact_dir / "sprint_distance" / "bundle.joblib").write_bytes(b"not a valid joblib file")

    with pytest.raises(ArtifactLoadError, match="sprint_distance: could not load bundle"):
        load_model_artifacts(artifact_dir)


def test_bundle_missing_key_raises_artifact_error(tmp_path: Path) -> None:
    import joblib

    artifact_dir = _copy_artifacts(tmp_path)
    bundle_path = artifact_dir / "accelerations" / "bundle.joblib"
    raw = joblib.load(bundle_path)
    del raw["scaler"]
    joblib.dump(raw, bundle_path)

    with pytest.raises(ArtifactLoadError, match="accelerations: bundle missing key 'scaler'"):
        load_model_artifacts(artifact_dir)


def test_duplicate_feature_cols_raise_artifact_error(tmp_path: Path) -> None:
    import joblib

    artifact_dir = _copy_artifacts(tmp_path)
    bundle_path = artifact_dir / "accelerations" / "bundle.joblib"
    raw = joblib.load(bundle_path)
    raw["feature_cols"] = raw["feature_cols"] + [raw["feature_cols"][0]]
    joblib.dump(raw, bundle_path)

    with pytest.raises(ArtifactLoadError, match="accelerations: feature_cols contains duplicates"):
        load_model_artifacts(artifact_dir)
