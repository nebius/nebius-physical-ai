"""Opt-in live proof for independent runtime-control storage.

Run from an operator checkout after staging ``NPA_SRC_S3_URI``::

    NPA_INTEGRATION_E2E=1 \
    NPA_E2E_RUNTIME_STORAGE=1 \
    NPA_E2E_S3_PREFIX=<fresh-relative-science-prefix> \
    NPA_E2E_RUNTIME_STORAGE_CONTROL_PREFIX=<owner-relative-prefix> \
    NPA_E2E_RUNTIME_STORAGE_ISOLATED_ROOT=<fresh-absolute-owner-only-dir> \
    NPA_E2E_SKY_BIN=<absolute-pinned-sky-path> \
    NPA_E2E_KUBECONTEXT=<selected-context> \
    NPA_E2E_PROJECT=<configured-project> \
    npa/.venv/bin/python -B -m pytest \
      npa/tests/e2e/test_workflow_runtime_storage_live_e2e.py -q -s

``NPA_E2E_S3_BUCKET``, ``NPA_E2E_SKYPILOT_CONFIG_PATH``, and the forwarded
``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` retain their normal live-suite
meanings. No project, bucket, node, PVC, or host path is embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
import uuid

import pytest
from typer.testing import CliRunner, Result

from npa.cli.main import app
from npa.clients.project_credentials import s3_client_for_project
from npa.deploy.images import DEFAULT_PUBLIC_CONTAINER_REGISTRY
from npa.orchestration.npa_workflow import load_spec
from npa.orchestration.npa_workflow.run_state import (
    RUN_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
)
from npa.orchestration.npa_workflow.submission_state import submission_state_path
from .npa_workflow_live_argv import (
    _owned_empty_isolation_root,
    _safe_relative_workflow_prefix,
    _workflow_prefixes_disjoint,
    plan_submit_args,
    runtime_submit_args,
)
from .npa_workflow_live_helpers import (
    live_bucket,
    materialize_live_spec,
    seed_live_workflow_inputs,
    write_runtime_evidence,
)

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot]
RUNNER = CliRunner()
SPEC = "insights-smoke.yaml"


@dataclass(frozen=True)
class _LiveRun:
    run_id: str
    spec_path: Path
    project: str | None
    bucket: str
    registry: str
    science_prefix: str
    control_parent: str
    control_prefix: str
    run_root: Path
    isolated_config_dir: Path
    sky_bin: Path
    infra: str
    launch_evidence: Path


@pytest.fixture(autouse=True)
def _require_live_storage_test() -> None:
    if os.environ.get("NPA_E2E_RUNTIME_STORAGE") != "1":
        pytest.skip("Set NPA_E2E_RUNTIME_STORAGE=1 for this live storage proof")


def _control_parent() -> str:
    value = os.environ.get("NPA_E2E_RUNTIME_STORAGE_CONTROL_PREFIX", "")
    try:
        return _safe_relative_workflow_prefix(value)
    except ValueError:
        pytest.fail(
            "NPA_E2E_RUNTIME_STORAGE_CONTROL_PREFIX must be a safe relative key"
        )


def _require_disjoint(science: str, control: str) -> None:
    if not _workflow_prefixes_disjoint(science, control):
        pytest.fail("scientific and runtime-control prefixes must be disjoint")


def _required_sky_bin() -> Path:
    value = (
        os.environ.get("NPA_E2E_SKY_BIN", "").strip()
        or os.environ.get("NPA_SKYPILOT_BIN", "").strip()
    )
    path = Path(value)
    if not value or not path.is_absolute() or not path.is_file():
        pytest.fail("set NPA_E2E_SKY_BIN to an absolute existing SkyPilot binary")
    resolved = path.resolve(strict=True)
    if not os.access(resolved, os.X_OK):
        pytest.fail("NPA_E2E_SKY_BIN must resolve to an executable")
    return resolved


def _selected_infra() -> str:
    context = os.environ.get("NPA_E2E_KUBECONTEXT", "").strip()
    if not context or any(character.isspace() for character in context):
        pytest.fail("set NPA_E2E_KUBECONTEXT to one exact configured context")
    return f"k8s/{context}"


def _isolation_root() -> Path:
    value = os.environ.get("NPA_E2E_RUNTIME_STORAGE_ISOLATED_ROOT", "").strip()
    try:
        return _owned_empty_isolation_root(value)
    except (OSError, ValueError):
        pytest.fail(
            "NPA_E2E_RUNTIME_STORAGE_ISOLATED_ROOT must be fresh, absolute, "
            "canonical, owner-only, and owned by the current user"
        )


def _build_run(tmp_path: Path, project: str | None) -> _LiveRun:
    run_id = f"npa-wf-runtime-storage-{uuid.uuid4().hex[:10]}"
    bucket = live_bucket(project)
    path = materialize_live_spec(tmp_path, SPEC, bucket=bucket, run_id=run_id)
    science = str(load_spec(path).config["prefix"]).strip("/")
    parent = _control_parent()
    control = f"{parent}/{run_id}"
    _require_disjoint(science, control)
    isolation_root = _isolation_root()
    sky_bin = _required_sky_bin()
    infra = _selected_infra()
    run_root = isolation_root / run_id
    run_root.mkdir(mode=0o700)
    registry = os.environ.get("NPA_E2E_REGISTRY", "").strip()
    return _LiveRun(
        run_id,
        path,
        project,
        bucket,
        registry or DEFAULT_PUBLIC_CONTAINER_REGISTRY,
        science,
        parent,
        control,
        run_root,
        run_root / "sky-controller",
        sky_bin,
        infra,
        run_root / "launch-coordinates.json",
    )


def _secret_args() -> list[str]:
    names = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.fail(f"live runtime worker requires environment variable {missing[0]}")
    return [item for name in names for item in ("--secret-env", name)]


def _sky_lifecycle_args(run: _LiveRun) -> list[str]:
    args = [
        "--s3-bucket",
        run.bucket,
        "--sky-bin",
        str(run.sky_bin),
        "--isolated-config-dir",
        str(run.isolated_config_dir),
        "--controller-backend",
        "kubernetes",
        "--infra",
        run.infra,
    ]
    value = os.environ.get("NPA_E2E_SKYPILOT_CONFIG_PATH", "").strip()
    if not value:
        return args
    path = Path(value)
    if not path.is_file():
        pytest.fail("NPA_E2E_SKYPILOT_CONFIG_PATH must name an existing file")
    return [*args, "--config-path", str(path.resolve(strict=True))]


def _plan(run: _LiveRun) -> dict[str, Any]:
    args = plan_submit_args(
        run.spec_path,
        run_id=run.run_id,
        registry=run.registry,
        project=run.project,
        workflow_s3_prefix=run.control_parent,
        skypilot_config_args=_sky_lifecycle_args(run),
    )
    args.append("--runtime")
    return _json_result(RUNNER.invoke(app, args), expected_exit=0)


def _runtime_args(run: _LiveRun, *, prefix: str, resume: bool = False) -> list[str]:
    return runtime_submit_args(
        run.spec_path,
        run_id=run.run_id,
        registry=run.registry,
        project=run.project,
        poll_seconds=30,
        max_wait_seconds=0,
        cancel_on_timeout=False,
        secret_env_args=_secret_args(),
        skypilot_config_args=_sky_lifecycle_args(run),
        resume=resume,
        workflow_s3_prefix=prefix,
    )


def _json_result(result: Result, *, expected_exit: int) -> dict[str, Any]:
    assert result.exit_code == expected_exit, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def _prefix_inventory(client: Any, bucket: str, prefix: str) -> tuple[tuple, ...]:
    rows = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/"
    ):
        rows.extend(
            (str(item["Key"]), int(item["Size"]), str(item.get("ETag") or ""))
            for item in page.get("Contents") or []
        )
    return tuple(sorted(rows))


def _read_object(client: Any, run: _LiveRun, key: str) -> bytes:
    response = client.get_object(Bucket=run.bucket, Key=key)
    with response["Body"] as body:
        return body.read()


def _assert_success_objects(client: Any, run: _LiveRun) -> None:
    expected_uri = f"s3://{run.bucket}/{run.control_prefix}"
    manifest_key = f"{run.control_prefix}/npa-workflow/manifest.json"
    runtime_key = f"{run.control_prefix}/npa-workflow/runtime.json"
    manifest = json.loads(_read_object(client, run, manifest_key))
    runtime = json.loads(_read_object(client, run, runtime_key))
    assert manifest["schema_version"] == RUN_SCHEMA_VERSION
    assert manifest["run_id"] == run.run_id
    assert manifest["run_prefix_uri"] == expected_uri
    assert runtime["schema_version"] == RUNTIME_SCHEMA_VERSION
    assert runtime["run_id"] == run.run_id
    assert runtime["run_prefix_uri"] == expected_uri
    assert runtime["status"] == "succeeded"
    dashboard = _read_object(
        client, run, f"{run.science_prefix}/dashboard/dashboard.html"
    )
    assert dashboard and b"<html" in dashboard.lower()
    assert not _prefix_inventory(
        client, run.bucket, f"{run.science_prefix}/npa-workflow"
    )


def _assert_mismatch_is_read_only(client: Any, run: _LiveRun) -> None:
    mismatch_parent = f"{run.control_parent}-mismatch"
    mismatch_prefix = f"{mismatch_parent}/{run.run_id}"
    _require_disjoint(run.science_prefix, mismatch_prefix)
    receipt = submission_state_path(run.project or "default", run.run_id)
    before = (
        _prefix_inventory(client, run.bucket, run.control_prefix),
        _prefix_inventory(client, run.bucket, run.science_prefix),
        _prefix_inventory(client, run.bucket, mismatch_prefix),
        receipt.read_bytes(),
    )
    assert not before[2]
    result = RUNNER.invoke(app, _runtime_args(run, prefix=mismatch_parent, resume=True))
    failure = _json_result(result, expected_exit=1)
    assert failure["result"] == "error"
    after = (
        _prefix_inventory(client, run.bucket, run.control_prefix),
        _prefix_inventory(client, run.bucket, run.science_prefix),
        _prefix_inventory(client, run.bucket, mismatch_prefix),
        receipt.read_bytes(),
    )
    assert after == before


def _write_launch_coordinates(run: _LiveRun) -> None:
    payload = {
        "control_prefix_uri": f"s3://{run.bucket}/{run.control_prefix}",
        "infra": run.infra,
        "isolated_config_dir": str(run.isolated_config_dir),
        "run_id": run.run_id,
        "science_prefix_uri": f"s3://{run.bucket}/{run.science_prefix}",
        "schema": "npa.test.runtime-storage-live-coordinates.v1",
        "sky_bin": str(run.sky_bin),
        "spec_path": str(run.spec_path),
        "status": "ready_for_plan_and_launch",
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(run.launch_evidence, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)


def test_runtime_control_storage_is_independent_and_resume_safe(
    tmp_path: Path, e2e_project: str | None
) -> None:
    run = _build_run(tmp_path, e2e_project)
    _write_launch_coordinates(run)
    expected_uri = f"s3://{run.bucket}/{run.control_prefix}"
    assert _plan(run)["run_prefix_uri"] == expected_uri
    seed_live_workflow_inputs(
        spec_name=SPEC,
        bucket=run.bucket,
        run_id=run.run_id,
        e2e_project=e2e_project,
    )
    result = RUNNER.invoke(app, _runtime_args(run, prefix=run.control_parent))
    payload = _json_result(result, expected_exit=0)
    assert payload["status"] == "succeeded"
    assert payload["run_prefix_uri"] == expected_uri
    write_runtime_evidence(run.run_id, payload)
    client = s3_client_for_project(e2e_project, allow_host_creds=True)
    _assert_success_objects(client, run)
    _assert_mismatch_is_read_only(client, run)
