"""Live FiftyOne temporal-tag to LeRobot export coverage.

Prepare a FiftyOne 1.22 workbench with a fully covered LeRobot dataset, then set
``NPA_E2E_FIFTYONE_SUBTASK_DATASET`` and an empty
``NPA_E2E_FIFTYONE_SUBTASK_OUTPUT_PATH`` before enabling the E2E suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.e2e
def test_fiftyone_exports_reviewed_subtasks_to_lerobot() -> None:
    dataset_name = os.environ.get("NPA_E2E_FIFTYONE_SUBTASK_DATASET", "")
    output_path = os.environ.get("NPA_E2E_FIFTYONE_SUBTASK_OUTPUT_PATH", "")
    project = os.environ.get("NPA_E2E_PROJECT", "")
    workbench = os.environ.get("NPA_E2E_FIFTYONE_WORKBENCH", "")
    if not all((dataset_name, output_path, project, workbench)):
        pytest.skip("prepared FiftyOne subtask-label E2E inputs are not configured")

    command = [
        sys.executable,
        "-m",
        "npa",
        "workbench",
        "fiftyone",
        "-p",
        project,
        "-n",
        workbench,
        "export-lerobot-subtasks",
        "--dataset-name",
        dataset_name,
        "--output-path",
        output_path,
        "--output-format",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "exported"
    assert report["segment_count"] > 0
    assert report["unlabeled_frame_count"] == 0
    assert report["uploaded_files"] > 0
