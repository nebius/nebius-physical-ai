"""Live infra tests for npa.workflow guide runbook paths (real S3 + CLI)."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from urllib.parse import urlparse

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import resolve_project_storage
from npa.clients.project_credentials import s3_client_for_project
from npa.orchestration.npa_workflow import build_plan, load_spec, run_workflow
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.run_state import RunStateStore

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


def _live_bucket(e2e_project: str | None) -> str:
    storage = resolve_project_storage(e2e_project)
    raw = storage.checkpoint_bucket or ""
    if not raw:
        pytest.fail("checkpoint_bucket is not configured for live npa.workflow tests")
    parsed = urlparse(raw if "://" in raw else f"s3://{raw}")
    bucket = parsed.netloc if parsed.scheme == "s3" else raw.split("/")[0]
    if not bucket:
        pytest.fail(f"could not resolve live bucket from {raw!r}")
    return bucket


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


def test_guide_sim2real_promote_plans_finalize_once(e2e_project: str | None) -> None:
    _live_bucket(e2e_project)
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPECS / "sim2real-vlm-rl.yaml"),
            "--run-id",
            "guide-promote-live",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    states = [step["state"] for step in payload["steps"]]
    assert states.count("finalize") == 1, states


def test_guide_run_spec_scheduler_and_persist_state(
    e2e_project: str | None,
    tmp_path: Path,
) -> None:
    bucket = _live_bucket(e2e_project)
    run_id = "guide-persist-live"
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
    assert scheduler.exit_code == 0, scheduler.output
    scheduler_payload = json.loads(scheduler.output)
    assert scheduler_payload["scheduler"]["tasks"], scheduler_payload

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
) -> None:
    bucket = _live_bucket(e2e_project)
    run_id = "guide-fail-live"
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
) -> None:
    bucket = _live_bucket(e2e_project)
    run_id = "guide-require-inputs"
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
    _live_bucket(e2e_project)
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    plan = build_plan(spec, run_id="guide-loop-back", assume_decision="loop_back")
    states = [step.state for step in plan.steps]
    inner = spec.config["inner_iterations"]
    outer = spec.config["outer_iterations"]
    assert states.count("rollouts") == inner * outer
