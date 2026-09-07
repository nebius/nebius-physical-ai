from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from npa.benchmarks.sim2real_model_agent import (
    BASE_TOOLS,
    PREPARED_ACTION_MARKER,
    TOOLS,
    _active_tools,
    _context_checkpoint,
    _load_transcript,
    _prepared_action_consumed_state,
    _run_tool,
    _submitted_workflow_state,
    _workflow_submission_block_reason,
)
from npa.benchmarks.sim2real_prepared_action import (
    IMAGE_ROLES,
    PreparedActionContext,
    PreparedActionError,
    _append_private_jsonl,
    create_receipt_from_request,
    execute_prepared_action,
    recover_occurrence,
    validate_receipt,
)


@pytest.fixture()
def prepared(tmp_path: Path) -> dict[str, object]:
    private = tmp_path / "private"
    evidence = private / "trial"
    control = private / "prepared-actions" / "trial"
    workspace = tmp_path / "workspace"
    evidence.mkdir(parents=True)
    control.mkdir(parents=True)
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (workspace / "tracked.txt").write_text("base\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=NPA Test",
            "-c",
            "user.email=npa@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", "-q"], cwd=workspace, check=True)
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    spec = evidence / "sim2real.yaml"
    spec.write_text("apiVersion: npa.workflow/v0.0.1\nkind: Workflow\n")
    manifest = evidence / "staged-dataset-manifest.json"
    manifest.write_text(json.dumps({"dataset_id": "prepared-fixture"}))
    preflight_dir = evidence / "preflight"
    preflight_dir.mkdir()
    preflight_names = (
        "health_preflight",
        "model_access",
        "skypilot_verify",
        "workflow_gpu_scheduler",
        "workflow_validate",
        "workflow_plan",
        "image_pullability",
        "submit_plan_only",
    )
    preflights = {}
    for name in preflight_names:
        path = preflight_dir / f"{name}.json"
        path.write_text(json.dumps({"name": name, "passed": True}))
        preflights[name] = str(path)
    images = {
        role: f"registry.invalid/{role}@sha256:{str(index + 1) * 64}"
        for index, role in enumerate(IMAGE_ROLES)
    }
    run_id = "prepared-run-1"
    project = "private-alias"
    project_infra = "k8s/private-context"
    source = base
    spec_sandbox = "/tmp/npa-private-evidence/sim2real.yaml"
    argv = [
        "npa/.venv/bin/npa",
        "workbench",
        "workflow",
        "submit",
        spec_sandbox,
        "--run-id",
        run_id,
        "--project",
        project,
        "--infra",
        project_infra,
        "--runtime",
        "--resume",
        "--accept-eula",
        "--preset",
        "public-franka-lift",
        "--assume-decision",
        "promote_checkpoint",
        "--max-wait-seconds",
        "0",
        "--output-format",
        "json",
        "--var",
        f"source_sha={source}",
        "--var",
        "require_baked_npa=1",
        "--var",
        "bucket=private-bucket-identity",
    ]
    for role, image in images.items():
        argv.extend(["--var", f"{role}_image={image}"])
    argv.extend(["--var", "isaac_cache_pvc=private-cache-identity"])
    secret_names = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "NEBIUS_TOKEN_FACTORY_KEY",
    ]
    for name in secret_names:
        argv.extend(["--secret-env", name])
    request = {
        "schema": "npa.sim2real.prepared_workflow_action.request.v1",
        "action_id": "submit-prepared-run-1",
        "workspace": str(workspace),
        "evidence_dir": str(evidence),
        "private_root": str(private),
        "canonical_spec_path": str(spec),
        "source_commit": source,
        "benchmark_base": base,
        "run_id": run_id,
        "project_alias": project,
        "project_infra": project_infra,
        "staged_manifest_path": str(manifest),
        "staged_input_identity": {"dataset_id": "prepared-fixture"},
        "images": images,
        "runtime_policy": {"runtime": True, "resume": True, "max_wait_seconds": 0},
        "accepted_eulas": ["isaac"],
        "required_secret_env_names": secret_names,
        "preflight_evidence": preflights,
        "argv": argv,
    }
    request_path = control / "prepared-action-request.json"
    request_path.write_text(json.dumps(request))
    os.chmod(request_path, 0o600)
    receipt_path = control / "prepared-action-receipt.json"
    receipt = create_receipt_from_request(request_path, receipt_path)
    environment = {
        "NPA_PROJECT": project,
        "NPA_SIM2REAL_INFRA": project_infra,
        "NPA_SIM2REAL_RUN_ID": run_id,
        "NPA_SIM2REAL_SOURCE_SHA": source,
        **{name: "present-in-test-only" for name in secret_names},
    }
    context = PreparedActionContext(
        workspace=workspace,
        evidence=evidence,
        control_dir=control,
        private_root=private,
        environment=environment,
        isolation={
            "evidence": evidence,
            "private_root": private,
            "controller_repo": tmp_path / "controller",
        },
    )
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "request": request,
        "context": context,
        "workspace": workspace,
        "evidence": evidence,
        "control": control,
    }


def test_receipt_tampering_is_rejected(prepared: dict[str, object]) -> None:
    path = prepared["receipt_path"]
    assert isinstance(path, Path)
    payload = json.loads(path.read_text())
    payload["run"]["run_id"] = "tampered-run"
    path.write_text(json.dumps(payload))
    os.chmod(path, 0o600)

    with pytest.raises(PreparedActionError, match="digest") as error:
        validate_receipt(
            path,
            requested_action_id="submit-prepared-run-1",
            context=prepared["context"],
        )
    assert error.value.classification == "receipt_tampered"


def test_receipt_binds_one_explicit_resume_retry(
    prepared: dict[str, object],
) -> None:
    request = dict(prepared["request"])
    request["action_id"] = "resume-prepared-run-2"
    request["runtime_policy"] = {
        "runtime": True,
        "resume": True,
        "max_wait_seconds": 0,
        "retries": 1,
    }
    argv = list(request["argv"])
    insert_at = argv.index("--max-wait-seconds")
    argv[insert_at:insert_at] = ["--retries", "1"]
    request["argv"] = argv
    control = prepared["control"]
    assert isinstance(control, Path)
    request_path = control / "resume-action-request.json"
    receipt_path = control / "resume-action-receipt.json"
    request_path.write_text(json.dumps(request))
    os.chmod(request_path, 0o600)

    receipt = create_receipt_from_request(request_path, receipt_path)

    assert receipt["runtime_policy"]["retries"] == 1
    assert receipt["argv"].count("--retries") == 1
    validate_receipt(
        receipt_path,
        requested_action_id="resume-prepared-run-2",
        context=prepared["context"],
    )


def test_value_bearing_request_inside_agent_mount_is_rejected(
    prepared: dict[str, object]
) -> None:
    request = json.loads(json.dumps(prepared["request"]))
    request["action_id"] = "visible-request"
    request_path = prepared["evidence"] / "visible-request.json"
    receipt_path = prepared["control"] / "visible-request-receipt.json"
    request_path.write_text(json.dumps(request))
    os.chmod(request_path, 0o600)
    with pytest.raises(PreparedActionError) as error:
        create_receipt_from_request(request_path, receipt_path)
    assert error.value.classification == "receipt_location_invalid"


def test_generic_command_cannot_read_operator_control_receipt(
    prepared: dict[str, object]
) -> None:
    context = prepared["context"]
    assert isinstance(context, PreparedActionContext)
    result = _run_tool(
        "run_command",
        {"command": f"test -r {prepared['receipt_path']}"},
        context.workspace,
        context.environment,
        context.isolation,
    )
    assert result["exit_code"] != 0


def test_source_mismatch_fails_closed(prepared: dict[str, object]) -> None:
    workspace = prepared["workspace"]
    assert isinstance(workspace, Path)
    (workspace / "untracked.txt").write_text("changed\n")
    with pytest.raises(PreparedActionError) as error:
        validate_receipt(
            prepared["receipt_path"],
            requested_action_id="submit-prepared-run-1",
            context=prepared["context"],
        )
    assert error.value.classification == "source_mismatch"


def test_same_dirty_status_entry_with_changed_content_fails_closed(
    prepared: dict[str, object]
) -> None:
    workspace = prepared["workspace"]
    assert isinstance(workspace, Path)
    tracked = workspace / "tracked.txt"
    tracked.write_text("dirty version one\n")
    request = json.loads(json.dumps(prepared["request"]))
    request["action_id"] = "content-bound-action"
    request_path = prepared["control"] / "content-bound-request.json"
    receipt_path = prepared["control"] / "content-bound-receipt.json"
    request_path.write_text(json.dumps(request))
    os.chmod(request_path, 0o600)
    create_receipt_from_request(request_path, receipt_path)

    tracked.write_text("dirty version two\n")
    with pytest.raises(PreparedActionError) as error:
        validate_receipt(
            receipt_path,
            requested_action_id="content-bound-action",
            context=prepared["context"],
        )
    assert error.value.classification == "source_mismatch"


def test_missing_secrets_are_names_only(prepared: dict[str, object]) -> None:
    context = prepared["context"]
    assert isinstance(context, PreparedActionContext)
    missing = PreparedActionContext(
        workspace=context.workspace,
        evidence=context.evidence,
        control_dir=context.control_dir,
        private_root=context.private_root,
        environment={
            key: value
            for key, value in context.environment.items()
            if key != "HF_TOKEN"
        },
        isolation=context.isolation,
    )
    with pytest.raises(PreparedActionError) as error:
        validate_receipt(
            prepared["receipt_path"],
            requested_action_id="submit-prepared-run-1",
            context=missing,
        )
    assert error.value.classification == "missing_required_secrets"
    assert "present-in-test-only" not in str(error.value)


def test_argv_injection_is_rejected_even_with_recomputed_digests(
    prepared: dict[str, object], tmp_path: Path
) -> None:
    request = json.loads(json.dumps(prepared["request"]))
    request["action_id"] = "injected-action"
    request["argv"] = ["bash", "-lc", "npa workbench workflow submit spec.yaml"]
    request_path = prepared["control"] / "injected-request.json"
    request_path.write_text(json.dumps(request))
    os.chmod(request_path, 0o600)
    receipt_path = prepared["control"] / "injected-receipt.json"

    with pytest.raises(PreparedActionError) as error:
        create_receipt_from_request(request_path, receipt_path)
    assert error.value.classification == "argv_contract_invalid"


def test_valid_submission_executes_exact_argv_once_and_is_bounded(
    prepared: dict[str, object]
) -> None:
    calls: list[list[str]] = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(
                {
                    "run_id": "prepared-run-1",
                    "status": "RUNNING",
                    "details": "x" * 100_000,
                }
            ),
            "private-log" * 20_000,
        )

    result = execute_prepared_action(
        prepared["receipt_path"],
        requested_action_id="submit-prepared-run-1",
        occurrence_id="occurrence-1",
        context=prepared["context"],
        runner=runner,
    )

    assert calls == [prepared["receipt"]["argv"]]
    assert result["submission_accepted"] is True
    assert result["safe_run_reference"] == "prepared-run-1"
    assert result["status"] == "RUNNING"
    assert "private-log" not in json.dumps(result)
    assert len(json.dumps(result)) < 2_000
    assert len(result["evidence"]["full_output_sha256"]) == 64
    assert (prepared["control"] / "prepared-action-output.jsonl").stat().st_mode & 0o777 == 0o600

    duplicate = execute_prepared_action(
        prepared["receipt_path"],
        requested_action_id="submit-prepared-run-1",
        occurrence_id="occurrence-2",
        context=prepared["context"],
        runner=runner,
    )
    assert duplicate["error"]["classification"] == "duplicate_submission_prevented"
    assert len(calls) == 1


def test_exit_zero_without_authoritative_run_identity_is_indeterminate(
    prepared: dict[str, object]
) -> None:
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps({"status": "RUNNING"}), "")

    result = execute_prepared_action(
        prepared["receipt_path"],
        requested_action_id="submit-prepared-run-1",
        occurrence_id="invalid-success",
        context=prepared["context"],
        runner=runner,
    )
    assert result["submission_accepted"] is False
    assert result["action_consumed"] is True
    assert result["status"] == "INDETERMINATE"
    assert result["error"]["classification"] == "workflow_submit_response_invalid"


@pytest.mark.parametrize(
    "status",
    ["failed", "FAILED_SETUP", "cancelled", "STOPPED"],
)
def test_nonzero_authoritative_terminal_failure_is_an_accepted_submission(
    prepared: dict[str, object], status: str
) -> None:
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            1,
            json.dumps({"run_id": "prepared-run-1", "status": status}),
            "private terminal failure detail",
        )

    result = execute_prepared_action(
        prepared["receipt_path"],
        requested_action_id="submit-prepared-run-1",
        occurrence_id="terminal-failure",
        context=prepared["context"],
        runner=runner,
    )

    assert result["submission_accepted"] is True
    assert result["action_consumed"] is True
    assert result["safe_run_reference"] == "prepared-run-1"
    assert result["status"] == status.upper()
    assert result["error"] == {
        "classification": "workflow_terminal_failure",
        "retryable": False,
        "action": "inspect the terminal workflow failure and private evidence; do not replay this prepared action",
    }
    assert "private terminal failure detail" not in json.dumps(result)


@pytest.mark.parametrize("status", ["RUNNING", "PENDING", "SUCCEEDED", "BOGUS_STATUS"])
def test_nonzero_nonfailure_status_is_indeterminate(
    prepared: dict[str, object], status: str
) -> None:
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            1,
            json.dumps({"run_id": "prepared-run-1", "status": status}),
            "private contradictory detail",
        )

    result = execute_prepared_action(
        prepared["receipt_path"],
        requested_action_id="submit-prepared-run-1",
        occurrence_id="contradictory-nonzero",
        context=prepared["context"],
        runner=runner,
    )

    assert result["submission_accepted"] is False
    assert result["action_consumed"] is True
    assert result["status"] == "INDETERMINATE"
    assert result["error"]["classification"] == "workflow_submit_indeterminate"
    assert "private contradictory detail" not in json.dumps(result)


def test_crash_before_and_after_exec_recovery_never_replays(
    prepared: dict[str, object]
) -> None:
    state = prepared["control"] / "prepared-action-state.jsonl"
    assert recover_occurrence(
        state,
        action_id="submit-prepared-run-1",
        occurrence_id="before-exec",
    ) == ("not_started", None)

    _append_private_jsonl(
        state,
        {
            "schema": "npa.sim2real.prepared_workflow_action.state.v1",
            "phase": "execution_started",
            "action_id": "submit-prepared-run-1",
            "occurrence_id": "after-exec",
        },
    )
    assert recover_occurrence(
        state,
        action_id="submit-prepared-run-1",
        occurrence_id="after-exec",
    ) == ("indeterminate", None)

    called = False

    def forbidden_runner(argv, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not replay")

    result = execute_prepared_action(
        prepared["receipt_path"],
        requested_action_id="submit-prepared-run-1",
        occurrence_id="retry",
        context=prepared["context"],
        runner=forbidden_runner,
    )
    assert result["error"]["classification"] == "indeterminate_prior_execution"
    assert called is False


def test_controller_wal_recovers_only_pre_exec_occurrence(
    prepared: dict[str, object]
) -> None:
    transcript = prepared["evidence"] / "transcript.jsonl"
    transcript.write_text("")
    journal = prepared["evidence"] / "tool-results.jsonl"
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "typed-call",
                "type": "function",
                "function": {
                    "name": "submit_prepared_workflow",
                    "arguments": json.dumps({"action_id": "submit-prepared-run-1"}),
                },
            }
        ],
    }
    journal.write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.tool_execution.v2",
                "response_id": "response-before-exec",
                "assistant": assistant,
                "tool_call_id": "typed-call",
                "tool_name": "submit_prepared_workflow",
                "prepared_action_id": "submit-prepared-run-1",
                "occurrence_id": "before-exec",
                "phase": "intent",
            }
        )
        + "\n"
    )
    loaded = _load_transcript(
        transcript,
        journal,
        prepared["control"] / "prepared-action-state.jsonl",
    )
    assert json.loads(loaded[1]["content"])["classification"] == "prepared_action_not_started"


def test_compaction_preserves_prepared_action_and_durable_submit() -> None:
    marker = {
        "role": "user",
        "content": PREPARED_ACTION_MARKER + "\nAction ID: action-1",
    }
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "submit",
                "type": "function",
                "function": {
                    "name": "submit_prepared_workflow",
                    "arguments": json.dumps({"action_id": "action-1"}),
                },
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": "submit",
        "content": json.dumps(
            {
                "schema": "npa.sim2real.prepared_workflow_action.result.v1",
                "submission_accepted": True,
                "action_consumed": True,
                "safe_run_reference": "prepared-run-1",
                "status": "RUNNING",
            }
        ),
    }
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        marker,
        assistant,
        result,
        {"role": "assistant", "content": "x" * 20_000},
    ]
    _, checkpoint = _context_checkpoint(messages, max_recent_chars=512)
    assert PREPARED_ACTION_MARKER in checkpoint["content"]
    assert '"submitted":true' in checkpoint["content"]
    assert '"consumed":true' in checkpoint["content"]
    assert '"available":false' in checkpoint["content"]
    assert "Typed action available: none" in checkpoint["content"]
    assert "Typed action available: submit_prepared_workflow" not in checkpoint["content"]
    assert _prepared_action_consumed_state(messages[2:]) is True
    assert _submitted_workflow_state(messages[2:]) == (True, ["prepared-run-1"])


def test_consumed_prepared_action_does_not_hide_new_action() -> None:
    messages = [
        {
            "role": "user",
            "content": f"{PREPARED_ACTION_MARKER}\nAction ID: action-1",
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "schema": "npa.sim2real.prepared_workflow_action.result.v1",
                    "action_consumed": True,
                }
            ),
        },
        {
            "role": "user",
            "content": (
                f"{PREPARED_ACTION_MARKER}\n"
                "Typed action available: submit_prepared_workflow. "
                "Action ID: action-2."
            ),
        },
    ]

    assert _prepared_action_consumed_state(messages) is True
    assert _prepared_action_consumed_state(messages, "action-1") is True
    assert _prepared_action_consumed_state(messages, "action-2") is False

    checkpoint_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *messages,
        {"role": "assistant", "content": "x" * 20_000},
    ]
    _, checkpoint = _context_checkpoint(checkpoint_messages, max_recent_chars=512)
    assert '"action_id":"action-2"' in checkpoint["content"]
    assert '"available":true' in checkpoint["content"]
    assert "Action ID: action-2." in checkpoint["content"]
    assert "Action ID: action-2.." not in checkpoint["content"]


@pytest.mark.parametrize("durable_state", ["finished", "indeterminate"])
def test_generic_submit_consumption_removes_typed_action_from_checkpoint(
    durable_state: str,
) -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "user",
            "content": (
                f"{PREPARED_ACTION_MARKER}\n"
                "Typed action available: submit_prepared_workflow. Action ID: action-1."
            ),
        },
        {"role": "assistant", "content": "x" * 20_000},
    ]
    _, checkpoint = _context_checkpoint(
        messages,
        max_recent_chars=512,
        durable_prepared_state=durable_state,
    )
    assert '"consumed":true' in checkpoint["content"]
    assert '"available":false' in checkpoint["content"]
    assert "Typed action available: none" in checkpoint["content"]
    assert "Typed action available: submit_prepared_workflow" not in checkpoint["content"]


@pytest.mark.parametrize("model_index", [0, 1, 2])
def test_typed_action_schema_has_parity_across_all_benchmark_models(
    model_index: int,
) -> None:
    models_path = (
        Path(__file__).parents[2]
        / "benchmarks/sim2real-three-model/models.json"
    )
    models = json.loads(models_path.read_text())["models"]
    typed = [
        tool
        for tool in TOOLS
        if tool["function"]["name"] == "submit_prepared_workflow"
    ]
    model = models[model_index]
    assert model["repository"]
    assert model["tool_call_parser"]
    assert len(typed) == 1
    assert typed[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"action_id": {"type": "string"}},
        "required": ["action_id"],
        "additionalProperties": False,
    }
    assert all(
        tool["function"]["name"] != "submit_prepared_workflow"
        for tool in BASE_TOOLS
    )


def test_unprepared_legacy_trials_keep_the_original_tool_schema(tmp_path: Path) -> None:
    assert _active_tools(None) is BASE_TOOLS
    assert _active_tools(tmp_path / "receipt.json") is TOOLS


@pytest.mark.parametrize("extra", ["--help", "-h", "--version", "--unknown"])
def test_closed_argv_rejects_introspection_and_unknown_options(
    prepared: dict[str, object], extra: str
) -> None:
    request = json.loads(json.dumps(prepared["request"]))
    request["action_id"] = "closed-argv"
    request["argv"].append(extra)
    request_path = prepared["control"] / f"{extra.lstrip('-')}-request.json"
    receipt_path = prepared["control"] / f"{extra.lstrip('-')}-receipt.json"
    request_path.write_text(json.dumps(request))
    os.chmod(request_path, 0o600)
    with pytest.raises(PreparedActionError) as error:
        create_receipt_from_request(request_path, receipt_path)
    assert error.value.classification == "argv_contract_invalid"


def test_preflight_tampering_is_rejected(prepared: dict[str, object]) -> None:
    preflight = prepared["evidence"] / "preflight" / "workflow_plan.json"
    preflight.write_text(json.dumps({"exit_code": 1, "passed": False}))
    with pytest.raises(PreparedActionError) as error:
        validate_receipt(
            prepared["receipt_path"],
            requested_action_id="submit-prepared-run-1",
            context=prepared["context"],
        )
    assert error.value.classification == "preflight_evidence_mismatch"


def test_contradictory_explicit_preflight_failure_is_rejected(
    prepared: dict[str, object]
) -> None:
    preflight = prepared["evidence"] / "preflight" / "workflow_plan.json"
    preflight.write_text(json.dumps({"exit_code": 0, "passed": False}))
    request = json.loads(json.dumps(prepared["request"]))
    request["action_id"] = "contradictory-preflight"
    request_path = prepared["control"] / "contradictory-request.json"
    receipt_path = prepared["control"] / "contradictory-receipt.json"
    request_path.write_text(json.dumps(request))
    os.chmod(request_path, 0o600)
    with pytest.raises(PreparedActionError) as error:
        create_receipt_from_request(request_path, receipt_path)
    assert error.value.classification == "preflight_not_passed"


@pytest.mark.parametrize("phase", ["execution_started", "execution_finished"])
def test_durable_prepared_state_blocks_generic_submit_without_wal(
    prepared: dict[str, object], phase: str
) -> None:
    _append_private_jsonl(
        prepared["control"] / "prepared-action-state.jsonl",
        {
            "schema": "npa.sim2real.prepared_workflow_action.state.v1",
            "phase": phase,
            "action_id": "submit-prepared-run-1",
            "occurrence_id": "lost-wal-intent",
        },
    )
    assert (
        _workflow_submission_block_reason(
            [],
            tool_name="run_command",
            arguments={
                "command": "npa workbench workflow submit spec.yaml --runtime"
            },
            durable_prepared_state=(
                "indeterminate" if phase == "execution_started" else "finished"
            ),
        )
        == "DuplicateWorkflowSubmissionBlocked"
    )


def test_different_action_id_cannot_bypass_run_wide_consumed_state(
    prepared: dict[str, object]
) -> None:
    _append_private_jsonl(
        prepared["control"] / "prepared-action-state.jsonl",
        {
            "schema": "npa.sim2real.prepared_workflow_action.state.v1",
            "phase": "execution_started",
            "action_id": "replacement-action",
            "occurrence_id": "replacement-occurrence",
        },
    )
    called = False

    def runner(argv, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not execute")

    result = execute_prepared_action(
        prepared["receipt_path"],
        requested_action_id="submit-prepared-run-1",
        occurrence_id="original-occurrence",
        context=prepared["context"],
        runner=runner,
    )
    assert result["error"]["classification"] == "indeterminate_prior_execution"
    assert called is False
