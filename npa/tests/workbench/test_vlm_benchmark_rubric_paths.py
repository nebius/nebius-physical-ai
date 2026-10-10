"""Preserve inline and file rubric boundaries in the public benchmark caller."""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pytest

from npa.workbench import vlm_eval


@pytest.mark.parametrize(
    "rubric",
    [vlm_eval.DEFAULT_RUBRIC, ("Require visible terminal evidence. " * 40).strip()],
)
def test_public_benchmark_accepts_long_inline_rubrics(rubric: str) -> None:
    assert len(rubric.encode()) > 255
    dataset = str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH)
    items = vlm_eval.load_benchmark_dataset(dataset).items
    report = vlm_eval.benchmark_vlm_eval(
        dataset=dataset,
        backend="stub",
        rubrics=[rubric],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )
    assert report.item_count == len(items) == 5
    assert report.best_config.config.rubric == rubric
    assert {
        case.item_id: case.expected_label for case in report.best_config.results
    } == {item.id: item.expected_label for item in items}


@pytest.mark.parametrize("prefix", ["", "@"])
@pytest.mark.parametrize("relative", [False, True])
def test_benchmark_rubric_file_resolution_is_preserved(
    prefix: str, relative: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "explicit-rubric.txt"
    path.write_text("Require a visible terminal state.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rubric = prefix + (path.name if relative else str(path))
    report = vlm_eval.benchmark_vlm_eval(
        dataset=str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH),
        backend="stub",
        rubrics=[rubric],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )
    assert report.best_config.config.rubric == "Require a visible terminal state."


@pytest.mark.parametrize("rubric", ["default", "dataset-named"])
def test_dataset_named_rubrics_keep_precedence(rubric: str) -> None:
    assert vlm_eval._resolve_benchmark_rubrics(
        [rubric], dataset_rubrics={rubric: "Named fixture rubric"}, dataset_path=""
    ) == [(rubric, "Named fixture rubric")]


def test_explicit_missing_rubric_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(vlm_eval.VlmEvalError, match="rubric file does not exist"):
        vlm_eval.benchmark_vlm_eval(
            dataset=str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH),
            backend="stub",
            rubrics=["@" + str(tmp_path / "missing.txt")],
        )


def test_explicit_overlong_rubric_file_is_not_inline_text() -> None:
    with pytest.raises(vlm_eval.VlmEvalError, match="Unable to read rubric file"):
        vlm_eval.benchmark_vlm_eval(
            dataset=str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH),
            backend="stub",
            rubrics=["@" + vlm_eval.DEFAULT_RUBRIC],
        )


@pytest.mark.parametrize("prefix", ["", "@"])
def test_rubric_disappearing_after_stat_is_an_unreadable_file(
    prefix, monkeypatch, tmp_path
):
    """Preserve read failures after proving a rubric path exists.

    Args:
        prefix: Optional explicit-file marker.
        monkeypatch: Isolated read failure injection.
        tmp_path: Test-owned directory.
    Returns:
        None.
    Raises:
        AssertionError: A failed file read becomes a missing path or inline text.
    """
    path = tmp_path / "rubric.txt"
    path.write_text("Visible completion required.")

    def removed_file(*args, **kwargs):
        raise FileNotFoundError("Synthetic removal after stat")

    monkeypatch.setattr(Path, "read_text", removed_file)
    with pytest.raises(vlm_eval.VlmEvalError, match="Unable to read rubric file"):
        vlm_eval._resolve_benchmark_rubrics(
            [prefix + str(path)], dataset_rubrics={}, dataset_path=""
        )


@pytest.mark.parametrize("prefix", ["", "@"])
def test_rubric_stat_permission_error_is_not_inline_text(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inaccessible(_path: Path):
        raise PermissionError(errno.EACCES, "Synthetic permission denial")

    monkeypatch.setattr(Path, "stat", inaccessible)
    with pytest.raises(vlm_eval.VlmEvalError, match="Unable to read rubric file"):
        vlm_eval._resolve_benchmark_rubrics(
            [prefix + "protected-rubric.txt"], dataset_rubrics={}, dataset_path=""
        )


@pytest.mark.parametrize("invalid_utf8", [False, True])
def test_explicit_unreadable_rubric_file_fails_closed(
    invalid_utf8: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unreadable.txt"
    path.write_bytes(b"\xff" if invalid_utf8 else b"Never substitute inline text.")
    if not invalid_utf8:
        original_read = Path.read_text

        def denied_read(candidate: Path, *args: Any, **kwargs: Any) -> str:
            if candidate == path:
                raise PermissionError(errno.EACCES, "Synthetic read permission denial")
            return original_read(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied_read)
    with pytest.raises(vlm_eval.VlmEvalError, match="Unable to read rubric file"):
        vlm_eval.benchmark_vlm_eval(
            dataset=str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH),
            backend="stub",
            rubrics=["@" + str(path)],
        )
