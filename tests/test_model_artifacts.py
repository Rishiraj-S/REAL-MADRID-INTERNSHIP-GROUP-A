import pickle
import shutil
from pathlib import Path

import pytest

from real_madrid_acwr.config import MODEL_ARTIFACTS_DIR
from real_madrid_acwr.modeling.artifacts import (
    BASE_FEATURES,
    TARGETS,
    ArtifactLoadError,
    load_model_artifacts,
)


def _copy_artifacts(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "xgboost"
    shutil.copytree(MODEL_ARTIFACTS_DIR, artifact_dir)
    return artifact_dir


def _load_feature_cols(artifact_dir: Path, target: str) -> list[str]:
    with (artifact_dir / target / "feature_cols.pkl").open("rb") as file:
        return pickle.load(file)


def _write_pickle(path: Path, value: object) -> None:
    with path.open("wb") as file:
        pickle.dump(value, file)


def test_current_model_artifacts_load_successfully() -> None:
    artifacts = load_model_artifacts()

    assert set(artifacts) == set(TARGETS)
    for target, artifact in artifacts.items():
        assert artifact.target == target
        assert len(artifact.feature_cols) == 45
        assert BASE_FEATURES.issubset(artifact.feature_cols)
        assert len([col for col in artifact.feature_cols if col.startswith("pid_")]) == 28


def test_missing_target_directory_raises_artifact_error(tmp_path: Path) -> None:
    artifact_dir = _copy_artifacts(tmp_path)
    shutil.rmtree(artifact_dir / "acc_total")

    with pytest.raises(ArtifactLoadError, match="acc_total: missing artifact file"):
        load_model_artifacts(artifact_dir)


def test_missing_file_in_artifact_triad_raises_artifact_error(tmp_path: Path) -> None:
    artifact_dir = _copy_artifacts(tmp_path)
    (artifact_dir / "total_distance" / "transform.pkl").unlink()

    with pytest.raises(ArtifactLoadError, match="total_distance: missing artifact file"):
        load_model_artifacts(artifact_dir)


def test_bad_transform_metadata_raises_artifact_error(tmp_path: Path) -> None:
    artifact_dir = _copy_artifacts(tmp_path)
    _write_pickle(artifact_dir / "vel_total" / "transform.pkl", {"type": "sqrt"})

    with pytest.raises(ArtifactLoadError, match="vel_total: unsupported transform metadata"):
        load_model_artifacts(artifact_dir)


def test_duplicate_feature_columns_raise_artifact_error(tmp_path: Path) -> None:
    artifact_dir = _copy_artifacts(tmp_path)
    feature_cols = _load_feature_cols(artifact_dir, "acc_total")
    feature_cols[-1] = feature_cols[-2]
    _write_pickle(artifact_dir / "acc_total" / "feature_cols.pkl", feature_cols)

    with pytest.raises(ArtifactLoadError, match="acc_total: feature columns contain duplicates"):
        load_model_artifacts(artifact_dir)


def test_missing_base_feature_raises_artifact_error(tmp_path: Path) -> None:
    artifact_dir = _copy_artifacts(tmp_path)
    feature_cols = _load_feature_cols(artifact_dir, "total_distance")
    feature_cols[feature_cols.index("height")] = "unexpected_feature"
    _write_pickle(artifact_dir / "total_distance" / "feature_cols.pkl", feature_cols)

    with pytest.raises(ArtifactLoadError, match="total_distance: missing base feature"):
        load_model_artifacts(artifact_dir)
