"""Unit tests for the published-image GPU coverage chart renderer.

The chart is a picture of two source-of-truth files - the compatibility matrix
and the public publishing plan - so the risk it carries is that someone edits
one of those and the committed SVG keeps showing the old shape. These tests are
the gate the script's own --check flag was written for, plus the property that
makes the picture trustworthy: an unrecognized cell is an error, never a guess.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "npa/scripts/render_gpu_coverage_chart.py"
COMMITTED_SVG = REPO_ROOT / "docs/assets/image-gpu-coverage.svg"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_gpu_coverage_chart", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chart() -> ModuleType:
    return _load()


def _without_render_date(svg: str) -> str:
    return re.sub(r"Rendered \d{4}-\d{2}-\d{2}", "Rendered", svg)


def test_committed_chart_matches_the_matrix_and_the_publishing_plan(
    chart: ModuleType,
) -> None:
    """The one check that stops the picture from outliving the table."""
    rendered = chart.render(chart.published_images(), chart.matrix_rows())
    assert _without_render_date(COMMITTED_SVG.read_text()) == _without_render_date(
        rendered
    ), (
        "docs/assets/image-gpu-coverage.svg is stale; re-run "
        "npa/scripts/render_gpu_coverage_chart.py"
    )


def test_every_published_image_has_a_matrix_row(chart: ModuleType) -> None:
    rows = chart.matrix_rows()
    missing = sorted(set(chart.published_images()) - set(rows))
    assert not missing, f"published but absent from the compatibility matrix: {missing}"


def test_an_unrecognized_cell_is_an_error_rather_than_a_guess(
    chart: ModuleType,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        chart.classify("probably fine on this one", "npa-example")
    assert "cannot classify" in str(excinfo.value)


def test_each_cell_keeps_its_matrix_wording_as_a_tooltip(chart: ModuleType) -> None:
    """A one-word swatch must not be the only surviving record of the reason."""
    published = chart.published_images()
    svg = chart.render(published, chart.matrix_rows())
    assert svg.count("<title>") == len(published) * len(chart.COLUMNS)
    assert "cu128 NVRTC cannot JIT" in svg
