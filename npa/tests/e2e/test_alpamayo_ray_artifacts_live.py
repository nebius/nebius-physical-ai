"""Independently reopen real Alpamayo Ray reports and their S3 inference artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image
import pytest

from npa.clients.storage import StorageClient
from npa.workbench.alpamayo2_super.runtime import DEFAULT_DATASET_REVISION, DEFAULT_MODEL_REVISION


pytestmark = [
    pytest.mark.e2e_pipeline,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1"
        or not os.environ.get("NPA_ALPAMAYO_RAY_REPORT_URI"),
        reason="requires an explicit completed live Ray report in authorized S3 storage",
    ),
]


def _read(client, uri, path):
    client.download_file(uri, str(path))
    return json.loads(path.read_text())


@pytest.fixture
def live_report(tmp_path):
    client = StorageClient.from_environment()
    uri = os.environ["NPA_ALPAMAYO_RAY_REPORT_URI"]
    report = _read(client, uri, tmp_path / "report.json")
    client.download_file(uri.rsplit("/", 1)[0] + "/SHA256SUMS", str(tmp_path / "SHA256SUMS"))
    expected = (tmp_path / "SHA256SUMS").read_text().split()[0]
    assert hashlib.sha256((tmp_path / "report.json").read_bytes()).hexdigest() == expected
    assert report["status"] == "complete"
    assert report["revisions"] == {
        "model_revision": DEFAULT_MODEL_REVISION, "dataset_revision": DEFAULT_DATASET_REVISION,
    }
    return client, report


def test_live_report_has_exact_case_coverage(live_report):
    _, report = live_report
    fields = ("sample_index", "seed", "diffusion_steps")
    requested = {tuple(case[field] for field in fields) for case in report["cases"]}
    observed = [tuple(row[field] for field in fields) for row in report["measurements"]]
    assert len(observed) == len(requested)
    assert set(observed) == requested
    for row in report["measurements"]:
        assert row["ray_actor_id"] and row["ray_node_id"]
        assert row["elapsed_seconds"] > 0
        assert all(math.isfinite(row[name]) and row[name] >= 0 for name in ("min_ade_m", "min_fde_m"))


def _verify_case(client, row, directory):
    directory.mkdir()
    result = _read(client, row["artifacts"]["result.json"], directory / "result.json")
    metadata = _read(client, row["artifacts"]["trajectory.json"], directory / "trajectory.json")
    client.download_file(row["artifacts"]["trajectory.png"], str(directory / "trajectory.png"))
    assert result["status"] == "ok"
    assert result["model"]["revision"] == DEFAULT_MODEL_REVISION
    assert result["dataset"]["revision"] == DEFAULT_DATASET_REVISION
    assert result["runtime"]["weights_baked"] is False
    assert result["runtime"]["dataset_baked"] is False
    assert "@sha256:" in result["runtime"]["image"]
    assert result["sample"] == row["sample"]
    assert metadata["clip_id"] == row["sample"]["clip_id"]
    assert metadata["t0_us"] == row["sample"]["t0_us"]
    assert metadata["seed"] == row["seed"]
    assert metadata["pred_xyz_shape"] == [1, 1, 1, 64, 3]
    assert metadata["projection_available"] is True
    for metric in ("min_ade_m", "min_fde_m"):
        assert metadata[metric] == result["metrics"][metric] == row[metric]
    with Image.open(directory / "trajectory.png") as picture:
        assert picture.format == "PNG"
        picture.load()
        assert picture.width > 0 and picture.height > 0


def test_live_case_artifacts_decode_and_match_the_requested_samples(live_report, tmp_path: Path):
    client, report = live_report
    for index, row in enumerate(report["measurements"]):
        _verify_case(client, row, tmp_path / f"case-{index}")
