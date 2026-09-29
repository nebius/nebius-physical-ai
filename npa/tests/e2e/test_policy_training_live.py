"""Opt-in real FiftyOne curation and operator-configured Slurm workflow coverage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from npa.workbench.dataset.storage import read_json_uri, write_json_uri
from npa.workflows.policy_training.data import curate, split

pytestmark = pytest.mark.e2e


def test_real_fiftyone_curation_and_group_split(tmp_path):
    pytest.importorskip("fiftyone")
    source = tmp_path / "dataset"
    write_json_uri(
        str(source / "meta/info.json"),
        {
            "codebase_version": "v3.0",
            "total_episodes": 41,
        },
    )
    episodes = []
    for index in range(41):
        preview = tmp_path / f"preview-{index}.png"
        pixels = (np.indices((16, 16)).sum(axis=0) % 2 * 255).astype("uint8")
        if index == 40:
            pixels[:] = 0
        Image.fromarray(pixels).save(preview)
        episodes.append(
            {
                "dataset_uri": str(source),
                "episode_index": index,
                "group_id": f"generated-{index}",
                "source": "real",
                "preview_uri": str(preview),
                "detections": ["object"],
                "inhouse_keep": True,
            }
        )
    _run_curation(tmp_path, episodes)


def _run_curation(tmp_path, episodes):
    input_uri, policy_uri = (
        str(tmp_path / "episodes.json"),
        str(tmp_path / "policy.json"),
    )
    output_uri, split_uri = (
        str(tmp_path / "curated.json"),
        str(tmp_path / "split/index.json"),
    )
    write_json_uri(
        input_uri, {"schema": "npa.policy.episodes.v1", "episodes": episodes}
    )
    write_json_uri(
        policy_uri,
        {
            "min_brightness": 0.1,
            "max_brightness": 0.9,
            "min_sharpness": 0.01,
            "min_detections": 1,
        },
    )
    curate(input_uri, output_uri, policy_uri)
    report = read_json_uri(output_uri)
    assert report["engine"] == "fiftyone"
    assert report["selected_count"] == 40
    assert report["inhouse_disagreement_count"] == 1
    split(output_uri, split_uri, "generated-test")
    counts = {
        name: item["groups"]
        for name, item in read_json_uri(split_uri)["partitions"].items()
    }
    assert counts == {"train": 36, "holdout_1": 2, "holdout_2": 2}


def test_operator_slurm_pipeline(tmp_path):
    spec = os.environ.get("NPA_E2E_POLICY_TRAINING_SPEC", "")
    if not spec:
        pytest.skip(
            "requires a private runtime spec with staged non-customer inputs and Slurm scripts"
        )
    run_id = os.environ["NPA_E2E_POLICY_TRAINING_RUN_ID"]
    command = [
        str(Path(sys.executable).with_name("npa")),
        "workbench",
        "workflow",
        "submit",
        spec,
        "--runtime",
        "--stage-src",
        "--run-id",
        run_id,
        "--json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        "live submission failed; inspect private operator evidence"
    )
    # The configured terminal URI identifies this exact run; a successful submit
    # alone is insufficient when a controller reports asynchronous acceptance.
    output_uri = os.environ["NPA_E2E_POLICY_TRAINING_RESULT_URI"]
    report = read_json_uri(output_uri)
    assert report["status"] == "completed"
    assert report["request"]["run_id"] == run_id
    assert report["request"]["stage"] == "deploy"
    assert report["policy_test_report_uri"]
