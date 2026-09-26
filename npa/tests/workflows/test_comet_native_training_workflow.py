import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]


def test_workflow_is_real_and_has_control_science_inputs():
    value = yaml.safe_load(
        (
            ROOT / "workflows/testing/behavior-comet-native-full-training.yaml"
        ).read_text()
    )
    assert value["apiVersion"] == "npa.workflow/v0.0.1"
    assert (
        "python -m npa.workflows.behavior_challenge.comet_native_workflow preflight"
        in value["states"]["preflight"]["run"]["shell"]
    )
    assert "NPA_WORKFLOW_ATTEMPT_ID" in value["states"]["preflight"]["run"]["shell"]
    assert (
        "runtime_manifest_uri" in value["config"] and "worker_sha256" in value["config"]
    )
    assert value["initial"] == "preflight"
    assert value["states"]["preflight"]["next"] == "train"
    assert value["states"]["preflight"]["resources"] == "cpu"
    assert value["states"]["train"]["resources"] == "gpu"
    assert value["states"]["train"]["terminal"] is True


def test_workflow_module_has_no_placeholder_success_path():
    text = (
        ROOT / "npa/src/npa/workflows/behavior_challenge/comet_native_workflow.py"
    ).read_text()
    assert "contract_ready" not in text and "fake" not in text.lower()
    assert "prepare_runtime(" in text and "StorageClient.from_environment" in text


def test_public_admission_example_declares_both_capacity_floors():
    import json

    value = json.loads(
        (
            ROOT / "workflows/implementations/behavior-comet12/ADMISSION.example.json"
        ).read_text()
    )
    assert value["minimum_materialization_free_bytes"] > 0
    assert value["minimum_checkpoint_free_bytes"] > 0


def _execute_stages(value, environment):
    for stage in ("preflight", "train"):
        shell = re.sub(
            r"\{\{[^}]+\}\}", "fixture", value["states"][stage]["run"]["shell"]
        )
        done = subprocess.run(
            ["/bin/sh", "-c", shell],
            check=False,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert done.returncode == 0, done.stderr


def _assert_complete_argv(rows):
    flags = (
        "--prefix",
        "--openpi-uri",
        "--runtime-manifest-uri",
        "--worker-package-root",
        "--worker-manifest-sha256",
        "--scientific-runtime-receipt-sha256",
        "--attempt-id",
        "--final-step",
        "--milestones",
    )
    assert len(rows) == 2
    for operation, row in zip(("preflight", "train"), rows, strict=True):
        assert f"comet_native_workflow {operation}" in row
        assert all(row.count(flag) == 1 for flag in flags)


def test_workflow_shell_executes_each_stage_as_one_complete_argv(tmp_path):
    value = yaml.safe_load(
        (
            ROOT / "workflows/testing/behavior-comet-native-full-training.yaml"
        ).read_text()
    )
    binary = tmp_path / "python"
    capture = tmp_path / "argv.log"
    binary.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CAPTURE"\n')
    binary.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "CAPTURE": str(capture),
    }
    _execute_stages(value, environment)
    _assert_complete_argv(capture.read_text().splitlines())
