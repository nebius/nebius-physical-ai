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
    preflight_argv = value["states"]["preflight"]["run"]["argv"]
    assert preflight_argv[:2] == ["python3", "-c"]
    assert "comet_native_workflow" in preflight_argv[2]
    assert "NPA_WORKFLOW_ATTEMPT_ID" in preflight_argv[2]
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


def _assert_complete_argv(rows):
    flags = (
        "--prefix",
        "--openpi-uri",
        "--runtime-manifest-uri",
        "--worker-package-root",
        "--worker-manifest-sha256",
        "--scientific-runtime-receipt-sha256",
        "--final-step",
        "--milestones",
    )
    assert len(rows) == 2
    for operation, row in zip(("preflight", "train"), rows, strict=True):
        assert row[:2] == ["python3", "-c"]
        assert "comet_native_workflow" in row[2]
        assert '"--attempt-id", attempt' in row[2]
        assert row[3] == operation
        assert all(row.count(flag) == 1 for flag in flags)


def test_workflow_stage_argv_keeps_each_operation_and_flag_complete():
    value = yaml.safe_load(
        (
            ROOT / "workflows/testing/behavior-comet-native-full-training.yaml"
        ).read_text()
    )
    rows = [
        [str(argument) for argument in value["states"][stage]["run"]["argv"]]
        for stage in ("preflight", "train")
    ]
    _assert_complete_argv(rows)
