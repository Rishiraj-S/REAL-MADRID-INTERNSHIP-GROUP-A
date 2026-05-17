"""Repository contract tests for entrypoints and local developer workflows."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = "main.py"
LEGACY_ENTRYPOINT = "app.py"


def _read_repo_file(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def test_makefile_quality_targets_existing_streamlit_entrypoint() -> None:
    """The quality workflow should lint the real app entrypoint, not a removed file."""
    makefile = _read_repo_file("Makefile")

    assert (REPO_ROOT / ENTRYPOINT).exists()
    assert f"ruff check {ENTRYPOINT} train_models.py src tests utils" in makefile
    assert f"streamlit run {ENTRYPOINT}" in makefile
    assert f"ruff check {LEGACY_ENTRYPOINT}" not in makefile


def test_core_project_docs_reference_main_py_entrypoint() -> None:
    """Top-level developer docs should advertise the same runnable app entrypoint."""
    for file_name in ["AGENTS.md", "README.md"]:
        text = _read_repo_file(file_name)

        assert f"`{ENTRYPOINT}`" in text or f"run {ENTRYPOINT}" in text
        assert f"streamlit run {LEGACY_ENTRYPOINT}" not in text

