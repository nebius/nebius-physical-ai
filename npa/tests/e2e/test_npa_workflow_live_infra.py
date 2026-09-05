"""Live infra tests for npa.workflow guide runbook paths (real S3 + CLI)."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any
from uuid import uuid4

from botocore.exceptions import ClientError
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.project_credentials import s3_client_for_project
from npa.orchestration.npa_workflow import build_plan, load_spec, run_workflow
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.run_state import RunStateStore
from .npa_workflow_live_helpers import (
    ALL_GOLDEN_SPECS,
    DYNAMIC_SPECS,
    assert_no_credential_leakage,
    assume_decision_for,
    live_bucket,
    live_credential_markers,
    materialize_live_spec,
    parse_json_payload,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1",
        reason="Set NPA_INTEGRATION_E2E=1 to run live NPA workflow infra checks.",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
RUNNER = CliRunner()


@pytest.fixture(scope="module")
def forbidden_markers() -> list[str]:
    return live_credential_markers()


@pytest.fixture
def live_run_id(e2e_project: str | None, tmp_path: Path) -> Iterator[str]:
    """Archive and remove only this test's uniquely owned storage objects."""
    run_id = f"live-infra-{uuid4().hex}"
    bucket = live_bucket(e2e_project)
    client = s3_client_for_project(e2e_project)
    try:
        yield run_id
    finally:
        _archive_and_cleanup_live_prefix(
            client, bucket=bucket, run_id=run_id,
            evidence_dir=tmp_path / "live-s3-evidence",
        )


def _archive_and_cleanup_live_prefix(
    client: Any, *, bucket: str, run_id: str, evidence_dir: Path,
) -> dict[str, Any]:
    """Keep private byte/hash evidence before deleting exact UUID-owned keys.

    Cleanup failures remain visible after attempting every owned object. An
    object whose bytes could not be archived is preserved for manual recovery.
    No bucket lifecycle operations or parent-prefix listings are performed.
    """
    if not re.fullmatch(r"live-infra-[0-9a-f]{32}", run_id):
        raise ValueError("Live cleanup requires this test's UUID run identifier")
    prefix = f"npa-workflow-e2e/{run_id}/"
    evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence_dir.chmod(0o700)
    receipt: dict[str, Any] = {
        "bucket": bucket, "prefix": prefix, "objects": [], "errors": [],
        "all_owned_objects_absent": False,
    }

    def persist_receipt() -> None:
        descriptor = os.open(
            evidence_dir / "cleanup.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def error(operation: str, exc: Exception, key: str | None = None) -> None:
        row = {"operation": operation, "error_type": type(exc).__name__}
        if key is not None:
            row["key"] = key
        if isinstance(exc, ClientError):
            row["provider_code"] = str(exc.response.get("Error", {}).get("Code", ""))
        receipt["errors"].append(row)

    persist_receipt()
    try:
        keys = []
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not isinstance(key, str) or not key.startswith(prefix):
                    raise ValueError("Storage returned an object outside the owned test prefix")
                keys.append(key)
    except Exception as exc:
        error("list owned prefix", exc)
        persist_receipt()
        raise RuntimeError("Live S3 cleanup inventory failed; see private cleanup.json") from exc

    for key in sorted(set(keys)):
        row: dict[str, Any] = {"key": key, "archived": False, "deleted": False}
        receipt["objects"].append(row)
        try:
            filename = hashlib.sha256(key.encode()).hexdigest() + ".bin"
            archive = evidence_dir / filename
            body = client.get_object(Bucket=bucket, Key=key)["Body"]
            digest = hashlib.sha256()
            try:
                descriptor = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    for chunk in iter(lambda: body.read(1024 * 1024), b""):
                        stream.write(chunk)
                        digest.update(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                body.close()
            persisted = archive.read_bytes()
            if hashlib.sha256(persisted).digest() != digest.digest():
                raise ValueError("Archived object bytes failed hash verification")
            row.update(archived=True, archive_file=filename, sha256=digest.hexdigest(),
                       size_bytes=len(persisted))
            persist_receipt()
        except Exception as exc:
            error("archive owned object", exc, key)
            persist_receipt()
            continue
        try:
            client.delete_object(Bucket=bucket, Key=key)
            try:
                client.head_object(Bucket=bucket, Key=key)
            except ClientError as exc:
                if str(exc.response.get("Error", {}).get("Code", "")) not in {
                    "404", "NoSuchKey", "NotFound",
                }:
                    raise
            else:
                raise RuntimeError("Owned object remains after deletion")
            row["deleted"] = True
        except Exception as exc:
            error("delete and verify owned object", exc, key)
        persist_receipt()
    try:
        remaining = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        receipt["all_owned_objects_absent"] = not remaining.get("Contents")
    except Exception as exc:
        error("verify owned prefix absence", exc)
    persist_receipt()
    if receipt["errors"] or not receipt["all_owned_objects_absent"]:
        raise RuntimeError("Live S3 cleanup incomplete; see private cleanup.json")
    return receipt


def _live_store(e2e_project: str | None, *, bucket: str, prefix: str) -> RunStateStore:
    client = s3_client_for_project(e2e_project)

    def reader(b: str, key: str) -> str:
        response = client.get_object(Bucket=b, Key=key)
        return response["Body"].read().decode("utf-8")

    def writer(b: str, key: str, body: bytes) -> None:
        client.put_object(Bucket=b, Key=key, Body=body, ContentType="application/json")

    return RunStateStore(bucket=bucket, prefix=prefix, reader=reader, writer=writer)


def _artifact_checker(e2e_project: str | None):
    client = s3_client_for_project(e2e_project)

    def checker(bucket: str, key: str) -> bool:
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except client.exceptions.ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound", "403", "AccessDenied"}:
                return False
            raise

    return checker


def _write_live_spec(tmp_path: Path, *, bucket: str, run_id: str, shell: str) -> Path:
    path = tmp_path / "live-fail.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            apiVersion: npa.workflow/v0.0.1
            kind: Workflow

            metadata:
              name: live-fail

            config:
              bucket: {bucket}
              prefix: "npa-workflow-e2e/{run_id}"

            initial: fail-step

            states:
              fail-step:
                run:
                  shell: |
                    {shell}
                terminal: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_guide_sim2real_promote_plans_finalize_once(
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    live_bucket(e2e_project)
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPECS / "sim2real.yaml"),
            "--run-id",
            "guide-promote-live",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )
    payload = parse_json_payload(result, forbidden_markers)
    states = [step["state"] for step in payload["steps"]]
    assert states.count("stage-14-visualize") == 1, states


def test_guide_run_spec_scheduler_and_persist_state(
    e2e_project: str | None,
    tmp_path: Path,
    forbidden_markers: list[str],
    live_run_id: str,
) -> None:
    bucket = live_bucket(e2e_project)
    run_id = live_run_id
    spec_path = _write_live_spec(
        tmp_path,
        bucket=bucket,
        run_id=run_id,
        shell='echo "plan-only persist probe"',
    )

    scheduler = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "run-spec",
            str(spec_path),
            "--run-id",
            run_id,
            "--plan-only",
            "--scheduler-plan",
            "--json",
        ],
    )
    scheduler_payload = parse_json_payload(scheduler, forbidden_markers)
    assert scheduler_payload["scheduler"]["tasks"], scheduler_payload
    assert_no_credential_leakage(
        json.dumps(scheduler_payload), extra_forbidden=forbidden_markers
    )

    spec = load_spec(spec_path)
    store = _live_store(e2e_project, bucket=bucket, prefix=f"npa-workflow-e2e/{run_id}")
    report = run_workflow(
        spec,
        run_id=run_id,
        execute=False,
        state_store=store,
    )
    assert report["status"] == "planned"
    manifest = store.read_manifest()
    assert manifest is not None
    assert manifest.status == "planned"
    assert manifest.run_id == run_id


def test_guide_failed_execute_persists_failed_manifest_on_real_s3(
    e2e_project: str | None,
    tmp_path: Path,
    live_run_id: str,
) -> None:
    bucket = live_bucket(e2e_project)
    run_id = live_run_id
    spec_path = _write_live_spec(
        tmp_path,
        bucket=bucket,
        run_id=run_id,
        shell="exit 42",
    )
    spec = load_spec(spec_path)
    store = _live_store(e2e_project, bucket=bucket, prefix=f"npa-workflow-e2e/{run_id}")
    with pytest.raises(NpaWorkflowError):
        run_workflow(spec, run_id=run_id, execute=True, state_store=store)

    manifest = store.read_manifest()
    assert manifest is not None
    assert manifest.status == "failed"
    assert manifest.steps
    assert manifest.steps[0]["status"] == "failed"


def test_guide_require_inputs_fails_on_missing_artifact(
    e2e_project: str | None,
    tmp_path: Path,
    live_run_id: str,
) -> None:
    bucket = live_bucket(e2e_project)
    run_id = live_run_id
    path = tmp_path / "require-inputs.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            apiVersion: npa.workflow/v0.0.1
            kind: Workflow

            metadata:
              name: require-inputs

            config:
              bucket: {bucket}
              prefix: "npa-workflow-e2e/{run_id}"
              rollouts_uri: "s3://{bucket}/npa-workflow-e2e/{run_id}/missing/rollouts/"
              scores_uri: "s3://{bucket}/npa-workflow-e2e/{run_id}/scores/"
              vlm_backend: self-hosted

            initial: score

            states:
              score:
                run:
                  shell: "echo should-not-run"
                inputs:
                  - uri: "s3://{bucket}/npa-workflow-e2e/{run_id}/does-not-exist/manifest.json"
                terminal: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    spec = load_spec(path)
    with pytest.raises(NpaWorkflowError, match="missing required input"):
        run_workflow(
            spec,
            run_id=run_id,
            execute=True,
            require_inputs=True,
            artifact_checker=_artifact_checker(e2e_project),
        )


def test_guide_loop_back_assume_expands_outer_loop(e2e_project: str | None) -> None:
    live_bucket(e2e_project)
    spec = load_spec(SPECS / "sim2real.yaml")
    plan = build_plan(spec, run_id="guide-loop-back", assume_decision="loop_back")
    states = [step.state for step in plan.steps]
    inner = int(spec.config["inner_iterations"])
    outer = int(spec.config["outer_iterations"])
    assert states.count("stage-07-rollouts") == inner * outer


def test_guide_cosmos_gate_loop_back_expands_refinement(
    e2e_project: str | None,
) -> None:
    live_bucket(e2e_project)
    spec = load_spec(SPECS / "tokenfactory-cosmos-gate.yaml")
    plan = build_plan(spec, run_id="guide-cosmos-loop", assume_decision="loop_back")
    states = [step.state for step in plan.steps]
    assert states.count("vlm-critique") == spec.config["refinement_iterations"]


@pytest.mark.parametrize("name", ALL_GOLDEN_SPECS)
def test_live_golden_full_cli_on_real_bucket(
    name: str,
    e2e_project: str | None,
    tmp_path: Path,
    forbidden_markers: list[str],
) -> None:
    """All golden YAMLs: validate, plan, scheduler JSON on live bucket materialization."""

    bucket = live_bucket(e2e_project)
    run_id = f"live-golden-{name.replace('.yaml', '')}"
    path = materialize_live_spec(tmp_path, name, bucket=bucket, run_id=run_id)

    validate = RUNNER.invoke(
        app, ["workbench", "workflow", "validate-spec", str(path), "--json"]
    )
    assert parse_json_payload(validate, forbidden_markers)["status"] == "valid"

    plan_args = [
        "workbench",
        "workflow",
        "plan-spec",
        str(path),
        "--run-id",
        run_id,
        "--json",
    ]
    assume = assume_decision_for(name)
    if assume:
        plan_args.extend(["--assume-decision", assume])
    plan = RUNNER.invoke(app, plan_args)
    plan_payload = parse_json_payload(plan, forbidden_markers)
    assert plan_payload["steps"], name

    run_args = [
        "workbench",
        "workflow",
        "run-spec",
        str(path),
        "--run-id",
        run_id,
        "--plan-only",
        "--scheduler-plan",
        "--json",
    ]
    if assume:
        run_args.extend(["--assume-decision", assume])
    run = RUNNER.invoke(app, run_args)
    run_payload = parse_json_payload(run, forbidden_markers)
    assert run_payload.get("scheduler", {}).get("tasks"), name


@pytest.mark.parametrize("name", ALL_GOLDEN_SPECS)
def test_live_golden_persist_manifest_on_real_s3(
    name: str,
    e2e_project: str | None,
    tmp_path: Path,
    forbidden_markers: list[str],
    live_run_id: str,
) -> None:
    """Plan-only persist-state for every golden spec against real S3."""

    bucket = live_bucket(e2e_project)
    run_id = live_run_id
    path = materialize_live_spec(tmp_path, name, bucket=bucket, run_id=run_id)
    spec = load_spec(path)
    prefix = str(spec.config.get("prefix") or run_id)
    store = _live_store(e2e_project, bucket=bucket, prefix=prefix)

    assume = assume_decision_for(name) or "promote_checkpoint"
    report = run_workflow(
        spec,
        run_id=run_id,
        execute=False,
        assume_decision=assume,
        state_store=store,
    )
    assert report["status"] == "planned"
    assert_no_credential_leakage(json.dumps(report), extra_forbidden=forbidden_markers)

    manifest = store.read_manifest()
    assert manifest is not None
    assert manifest.status == "planned"
    assert manifest.run_id == run_id
    assert manifest.steps


@pytest.mark.parametrize("name", sorted(DYNAMIC_SPECS))
def test_live_dynamic_golden_loop_back_cli(
    name: str,
    e2e_project: str | None,
    tmp_path: Path,
    forbidden_markers: list[str],
) -> None:
    bucket = live_bucket(e2e_project)
    run_id = f"live-loop-{name.replace('.yaml', '')}"
    path = materialize_live_spec(tmp_path, name, bucket=bucket, run_id=run_id)
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(path),
            "--run-id",
            run_id,
            "--assume-decision",
            "loop_back",
            "--json",
        ],
    )
    payload = parse_json_payload(result, forbidden_markers)
    assert payload["steps"], name
