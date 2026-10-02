"""Opt-in full navigation runtime acceptance using operator-owned data and adapters."""

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest
import yaml

from npa.workflows.field_failure.adapters import _configured_identity
from npa.workflows.field_failure.artifacts import _read
from npa.workflows.field_failure.contracts import _Bundle
from npa.workflows.field_failure.stages import run_stage

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1"
        or os.environ.get("NPA_FIELD_FAILURE_LIVE") != "1",
        reason="Requires NPA_INTEGRATION_E2E=1 and NPA_FIELD_FAILURE_LIVE=1; never fabricates navigation inputs.",
    ),
]
_ROOT = Path(__file__).resolve().parents[3]
_SPEC = _ROOT / "workflows/testing/field-failure-policy-improvement.yaml"


def _configuration():
    path = os.environ.get("NPA_FIELD_FAILURE_LIVE_CONFIG")
    assert path, (
        "Set NPA_FIELD_FAILURE_LIVE_CONFIG to the operator's private JSON configuration"
    )
    config = json.loads(Path(path).read_text())
    required = {
        "bucket",
        "bundle_uri",
        "bundle_sha256",
        "reconstruct_adapter",
        "train_adapter",
        "evaluate_adapter",
        "reconstruction_image",
        "training_image",
        "evaluation_image",
    }
    assert set(config) == required, (
        "Live config must contain exactly the documented workflow input keys"
    )
    bundle = _Bundle.model_validate(
        _read(config["bundle_uri"], config["bundle_sha256"])[0]
    )
    for operation, profile in [
        ("reconstruct", "reconstruction"),
        ("train", "training"),
        ("evaluate", "evaluation"),
    ]:
        _configured_identity(
            bundle.adapters[operation],
            config[operation + "_adapter"],
            config[profile + "_image"],
        )
    return config, bundle


def _submit(path, run_id):
    infra = os.environ.get("NPA_FIELD_FAILURE_INFRA")
    assert infra, "Set NPA_FIELD_FAILURE_INFRA to the authorized RTX Kubernetes target"
    command = [
        sys.executable,
        "-m",
        "npa",
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--run-id",
        run_id,
        "--runtime",
        "--stage-src",
        "--infra",
        infra,
        "--json",
    ]
    for name in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "NGC_API_KEY",
        "HF_TOKEN",
    ]:
        if os.environ.get(name):
            command.extend(["--secret-env", name])
    subprocess.run(command, check=True, cwd=_ROOT)


def test_full_navigation_workflow_and_durable_comparison(tmp_path):
    config, bundle = _configuration()
    run_id = "field-failure-" + uuid.uuid4().hex
    spec = yaml.safe_load(_SPEC.read_text())
    spec["config"].update(config)
    path = tmp_path / _SPEC.name
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    _submit(path, run_id)
    root = f"s3://{config['bucket']}/field-failure/{run_id}"
    before, digest = _read(root + "/decision.json")
    assert before["status"] == "verified_comparison"
    assert before["episodes_per_policy"] == sum(len(s.seeds) for s in bundle.held_out)
    assert before["deployment_authorized"] is False
    assert before["evaluator"] == bundle.adapters["evaluate"].model_dump()
    run_stage("compare", config["bundle_uri"], config["bundle_sha256"], root, run_id)
    assert _read(root + "/decision.json")[1] == digest
