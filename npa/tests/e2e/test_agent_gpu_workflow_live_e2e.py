"""Agent-confirmed live GPU workflow proof.

This test is excluded from ordinary CI with the existing live markers and an
explicit opt-in.  It uses a real Token Factory planner, the production action
digest gate, a real SkyPilot/Nebius GPU submit, S3 artifact retrieval, Insights
ingestion, and a deployed-agent observation turn.  No provider is mocked.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import time
import uuid

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.agent_actions import (
    STOP_DONE,
    STOP_NEEDS_CONFIRMATION,
    ToolSpec,
    run_action_loop,
    summarize_observations,
)
from npa.cli.agent_routing import CHEAP_MODEL, STANDARD_MODEL
from npa.cli.main import app
from npa.clients.project_credentials import s3_client_for_project
from npa.clients.token_factory import TokenFactoryClient
from npa.orchestration.npa_workflow.runtime import _resolved_config
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.skypilot.workflow import workflow_status
from npa.workbench.insights.analytics import query_metrics
from npa.workbench.insights.schemas import IngestRunRequest, QueryRequest
from npa.workbench.insights.store import ingest_run

from .agent_live_helpers import load_agent_live_context
from .npa_workflow_live_helpers import (
    assert_no_credential_leakage,
    live_bucket,
    live_credential_markers,
    materialize_live_spec,
    parse_json_payload,
    seed_live_workflow_inputs,
)
from .test_npa_workflow_submit_live_e2e import (
    TERMINAL_OK,
    _image_args,
    _is_terminal_fail,
    _secret_env_args,
)
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_skypilot,
    pytest.mark.gpu,
    pytest.mark.agent_live,
    pytest.mark.token_factory_e2e,
]

RUNNER = CliRunner()
SPEC_NAME = "vlm-eval-single.yaml"


@pytest.fixture(autouse=True)
def _require_agent_gpu_live() -> None:
    required = {
        "NPA_INTEGRATION_E2E": "1",
        "NPA_AGENT_GPU_LIVE": "1",
        "NPA_AGENT_LIVE": "1",
    }
    missing = [name for name, value in required.items() if os.environ.get(name) != value]
    if missing:
        pytest.skip("agent GPU live proof requires " + ", ".join(missing))


def _completion(decision: dict) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(decision)}}],
        "usage": {"total_tokens": 0},
    }


def _agent_model() -> tuple[TokenFactoryClient, str]:
    client = TokenFactoryClient()
    available = client.list_models()
    for candidate in (CHEAP_MODEL, STANDARD_MODEL):
        if candidate in available:
            return client, candidate
    pytest.fail("Token Factory key exposes neither the cheap nor standard agent model")


def _workflow_submit_allowlist() -> dict[str, ToolSpec]:
    spec = ToolSpec(
        "workflow_submit",
        read_only=False,
        requires_confirmation=True,
        summary=(
            "Submit the named npa.workflow through SkyPilot and wait for its real "
            "terminal status. This is GPU-spending and requires confirmation."
        ),
        params=("spec", "run_id"),
    )
    return {spec.name: spec}


def _requested_accelerator(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(((payload.get("resources") or {}).get("gpu") or {}).get("accelerators") or "")


def _write_evidence(payload: dict) -> None:
    configured = os.environ.get("NPA_AGENT_GPU_EVIDENCE_PATH", "").strip()
    if configured:
        path = Path(configured)
    else:
        path = Path.home() / "npa-live-e2e-logs" / "agent-gpu-workflow-evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_agent_confirmation_to_real_gpu_artifact_and_grounded_answer(
    tmp_path: Path,
    e2e_project: str | None,
) -> None:
    registry = (
        os.environ.get("NPA_E2E_REGISTRY")
        or "ghcr.io/nebius/nebius-physical-ai"
    ).strip()

    case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == SPEC_NAME)
    run_id = f"agent-gpu-{uuid.uuid4().hex[:12]}"
    bucket = live_bucket(e2e_project)
    spec_path = materialize_live_spec(
        tmp_path,
        SPEC_NAME,
        bucket=bucket,
        run_id=run_id,
    )
    seed_live_workflow_inputs(
        spec_name=SPEC_NAME,
        bucket=bucket,
        run_id=run_id,
        e2e_project=e2e_project,
    )
    requested_accelerator = _requested_accelerator(spec_path)
    assert requested_accelerator, "live proof must request a GPU accelerator"

    client, planner_model = _agent_model()
    allowlist = _workflow_submit_allowlist()
    goal = (
        f"Submit GPU workflow {SPEC_NAME} with run_id {run_id}. "
        "Use workflow_submit with exactly those values."
    )
    provider_responses: list[dict] = []

    def _live_planner(messages, *, tier="cheap"):
        del tier
        response = client.chat_completion(
            model=planner_model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra={"chat_template_kwargs": {"thinking": False}},
        )
        provider_responses.append(response)
        return response

    launches: list[dict] = []

    def _must_not_launch(_args):  # pragma: no cover - a failure calls this
        launches.append({"unexpected": True})
        raise AssertionError("confirmation gate allowed an unconfirmed launch")

    proposed = run_action_loop(
        goal,
        tools={"workflow_submit": _must_not_launch},
        model_call=_live_planner,
        allowlist=allowlist,
    )
    assert proposed["stopped_reason"] == STOP_NEEDS_CONFIRMATION, proposed
    assert proposed["needs_confirmation"] is True
    assert launches == []
    action = proposed["proposed_action"]
    assert action["tool"] == "workflow_submit"
    assert action["args"] == {"spec": SPEC_NAME, "run_id": run_id}
    assert action["digest"]
    assert provider_responses, "proposal must come from the live Token Factory planner"

    canonical_plan = _completion(
        {
            "thought": "submit the exact operator-confirmed workflow",
            "tool": action["tool"],
            "args": action["args"],
        }
    )
    session_token = secrets.token_urlsafe(24)
    mismatched = run_action_loop(
        goal,
        tools={"workflow_submit": _must_not_launch},
        model_call=lambda _messages, tier="cheap": canonical_plan,
        allowlist=allowlist,
        confirm_token="mismatched-token",
        session_token=session_token,
        confirm_digest=action["digest"],
    )
    assert mismatched["stopped_reason"] == STOP_NEEDS_CONFIRMATION
    assert launches == []

    forbidden = live_credential_markers()
    launch_evidence: dict = {}
    last_status = "NOT_SUBMITTED"
    job_id = ""

    def _execute(args: dict) -> dict:
        nonlocal job_id, last_status, launch_evidence
        assert args == {"spec": SPEC_NAME, "run_id": run_id}
        assert not launches, "one confirmation token must launch at most once"
        launches.append({"run_id": run_id})
        submit_args = [
            "workbench",
            "workflow",
            "submit",
            str(spec_path),
            "--run-id",
            run_id,
            "--registry",
            registry,
            "--output-format",
            "json",
        ]
        submit_args.extend(_image_args(case, registry))
        submit_args.extend(_secret_env_args(case))
        result = RUNNER.invoke(app, submit_args)
        payload = parse_json_payload(result, forbidden)
        assert payload.get("status") in {"SUBMITTED", "RUNNING", "PENDING", "STARTING"}
        job_id = str(payload.get("job_id") or run_id)
        last_status = str(payload.get("status") or "SUBMITTED").upper()
        while True:
            current = workflow_status(job_id)
            last_status = str(current.status or "UNKNOWN").upper()
            assert_no_credential_leakage(
                (current.stdout or "") + (current.stderr or ""),
                extra_forbidden=forbidden,
            )
            if last_status in TERMINAL_OK:
                launch_evidence = {
                    "job_id": job_id,
                    "run_id": run_id,
                    "terminal_status": last_status,
                    "stdout_tail": (current.stdout or "")[-1000:],
                    "stderr_tail": (current.stderr or "")[-1000:],
                }
                return dict(launch_evidence)
            if _is_terminal_fail(last_status):
                pytest.fail(
                    f"{SPEC_NAME} reached terminal failure status={last_status} "
                    f"job_id={job_id} detail={((current.stderr or current.stdout or '')[-1000:])}"
                )
            time.sleep(30)

    planner_calls = {"count": 0}

    def _confirmed_planner(_messages, *, tier="cheap"):
        del tier
        planner_calls["count"] += 1
        if planner_calls["count"] == 1:
            return canonical_plan
        if not launch_evidence:
            return _completion(
                {
                    "thought": "the workflow executor did not return terminal evidence",
                    "final": "Workflow execution failed before terminal success.",
                }
            )
        return _completion(
            {
                "thought": "the terminal observation confirms completion",
                "final": (
                    f"Workflow {launch_evidence['run_id']} reached terminal status "
                    f"{launch_evidence['terminal_status']} as SkyPilot job "
                    f"{launch_evidence['job_id']}."
                ),
            }
        )

    confirmed = run_action_loop(
        goal,
        tools={"workflow_submit": _execute},
        model_call=_confirmed_planner,
        allowlist=allowlist,
        confirm_token=session_token,
        session_token=session_token,
        confirm_digest=action["digest"],
    )
    assert confirmed["stopped_reason"] == STOP_DONE, confirmed
    assert launch_evidence, confirmed
    assert launches == [{"run_id": run_id}]
    assert confirmed["tools_used"] == ["workflow_submit"]

    resolved = _resolved_config(load_spec(spec_path), run_id)
    run_prefix = str(resolved["prefix"]).strip("/")
    artifact_key = f"{run_prefix}/scores/vlm_eval_stub.json"
    manifest_key = f"{run_prefix}/npa-workflow/manifest.json"
    s3 = s3_client_for_project(e2e_project, allow_host_creds=True)
    artifact = json.loads(s3.get_object(Bucket=bucket, Key=artifact_key)["Body"].read())
    manifest = json.loads(s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read())
    assert artifact["backend"] == "self-hosted"
    assert artifact["dry_run"] is False
    assert int(artifact["frame_count"]) > 0
    assert artifact["model"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert isinstance(artifact["score"], int | float)
    accelerators = [
        str((step.get("resources_profile") or {}).get("accelerators") or "")
        for step in manifest.get("steps") or []
    ]
    assert requested_accelerator in accelerators

    input_uri = f"s3://{bucket}/{run_prefix}/"
    store_uri = f"s3://{bucket}/npa-workflow-e2e/{run_id}/agent-insights/"
    ingested = ingest_run(
        IngestRunRequest(
            input_uri=input_uri,
            output_uri=store_uri,
            workflow="vlm-eval-single",
            workflow_run=run_id,
        )
    )
    assert ingested.recorded_count > 0
    observed = query_metrics(QueryRequest(input_uri=store_uri, run_id=run_id))
    records = [dict(record) for record in observed.records]
    by_metric = {str(record["metric_name"]): record for record in records}
    assert by_metric["gpus"]["value"] >= 1.0
    assert by_metric["score"]["value"] == pytest.approx(float(artifact["score"]))

    deterministic_reply = summarize_observations(
        [{"tool": "insights_query", "result": observed.model_dump(mode="json")}]
    )
    assert run_id in deterministic_reply

    agent = load_agent_live_context()
    response = agent.post(
        "/api/agent/act",
        json={
            "goal": (
                "Use insights_query with input_uri "
                f"{store_uri} and run_id {run_id}. Report the run id and observed "
                "gpus and score metric values; do not estimate missing values."
            )
        },
        timeout=None,
    )
    response.raise_for_status()
    agent_answer = response.json()
    assert "insights_query" in (agent_answer.get("tools_used") or []), agent_answer
    assert run_id in str(agent_answer.get("reply") or ""), agent_answer
    assert_no_credential_leakage(
        json.dumps(agent_answer, sort_keys=True),
        extra_forbidden=forbidden,
    )

    deployed_reply = str(agent_answer.get("reply") or "").replace(bucket, "<bucket>")

    _write_evidence(
        {
            "schema": "npa.agent.gpu_live_e2e.v1",
            "planner_provider": "Nebius Token Factory",
            "planner_model": planner_model,
            "workflow": SPEC_NAME,
            "tool_ref": "workbench.vlm_eval.run",
            "cloud": "kubernetes",
            "accelerator": requested_accelerator,
            "run_id": run_id,
            "job_id": job_id,
            "terminal_status": last_status,
            "artifact_uri": f"s3://<bucket>/{artifact_key}",
            "artifact": {
                "backend": artifact["backend"],
                "model": artifact["model"],
                "frame_count": artifact["frame_count"],
                "score": artifact["score"],
                "status": artifact["status"],
            },
            "insights_store_uri": store_uri.replace(bucket, "<bucket>"),
            "insights_metrics": {
                "gpus": by_metric["gpus"]["value"],
                "score": by_metric["score"]["value"],
            },
            "gate": {
                "missing_token_launches": 0,
                "mismatched_token_launches": 0,
                "confirmed_launches": len(launches),
                "action_digest": action["digest"],
            },
            "grounded_reply": deterministic_reply,
            "deployed_agent_reply": deployed_reply,
            "deployed_agent_tools": agent_answer.get("tools_used") or [],
        }
    )
