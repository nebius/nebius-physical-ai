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
CATALOG = REPO_ROOT / "docs/workbench/container-image-catalog.md"


def _load() -> ModuleType:
    """Load the renderer as a module without making scripts a package."""
    spec = importlib.util.spec_from_file_location("render_gpu_coverage_chart", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chart() -> ModuleType:
    """Provide the loaded chart renderer once for this test module."""
    return _load()


def _without_render_date(svg: str) -> str:
    """Remove the one intentionally volatile part of generated SVG output."""
    return re.sub(r"Rendered \d{4}-\d{2}-\d{2}", "Rendered", svg)


def _catalog_band(text: str, heading: str) -> tuple[int, set[str]]:
    """Extract a count and image names from one catalog summary bullet."""
    pattern = rf"- \*\*(\d+) {re.escape(heading)}\*\*:(.*?)(?=\n- \*\*|\n\nTwo gaps)"
    match = re.search(pattern, text, flags=re.DOTALL)
    assert match, f"missing catalog band: {heading}"
    return int(match.group(1)), set(re.findall(r"`(npa-[^`]+)`", match.group(2)))


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
    """Every release in the publishing plan must have chart source data."""
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
    assert "accepted records" in svg
    assert "[accepted records](" not in svg


def test_every_current_public_cell_has_an_explicit_classification(
    chart: ModuleType,
) -> None:
    """New matrix vocabulary must fail the test until the chart handles it."""
    rows = chart.matrix_rows()
    for image in chart.published_images():
        assert len(rows[image]) == len(chart.COLUMNS)
        for cell in rows[image]:
            chart.classify(cell, image)


def test_release_specific_evidence_stays_tied_to_current_pins(
    chart: ModuleType,
) -> None:
    """Do not present runs on superseded images as current release evidence."""
    rows = chart.matrix_rows()
    expected = {
        "npa-detection-training": (
            "runtime-v1-20260905",
            ["supported", "historical", "verified", "historical", "historical"],
        ),
        "npa-cosmos2-transfer": (
            "2.5.1-sim2real-coherent-20260904",
            ["supported", "supported", "supported", "historical", "blocked"],
        ),
        "npa-envgen": (
            "0.1.2-sim2real-coherent-20260904",
            ["supported", "historical", "historical", "historical", "historical"],
        ),
        "npa-isaac-lab": (
            "3.0.0b2.post1-sim2real-coherent-20260904",
            ["supported", "supported", "verified", "blocked", "blocked"],
        ),
    }
    published = chart.published_images()
    for image, (tag, kinds) in expected.items():
        assert published[image] == tag
        assert [chart.classify(cell, image)[0] for cell in rows[image]] == kinds


def test_catalog_summary_matches_generated_band_membership(chart: ModuleType) -> None:
    """Keep prose counts and image lists synchronized with generated bands."""
    rows = chart.matrix_rows()
    grid = chart._classified_grid(chart.published_images(), rows)
    expected = {
        key: set(members) for _label, key, members in chart._ordered_bands(grid)
    }
    catalog = CATALOG.read_text(encoding="utf-8")
    headings = {
        "clean": "GPU images have no known blocked platform",
        "blocked": "public images are blocked on at least one platform",
        "cpu": "are CPU-only and GPU-agnostic",
    }
    assert set(expected) == set(headings)
    for key, heading in headings.items():
        count, members = _catalog_band(catalog, heading)
        assert count == len(expected[key])
        assert members == expected[key]
