"""Regression tests for the structured stdout boundary."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
import json

import pytest
import typer

from npa.lifecycle_intent import json_stdout_contract


class OutputFormat(str, Enum):
    json = "json"


@pytest.mark.parametrize("output_format", ["json", OutputFormat.json])
def test_json_contract_recognizes_string_and_enum_formats(
    capsys: pytest.CaptureFixture[str], output_format: str | OutputFormat
) -> None:
    @json_stdout_contract
    def command(*, output_format: str | OutputFormat) -> None:
        print("progress")
        print(json.dumps({"status": "succeeded", "run_id": "run-1"}))

    command(output_format=output_format)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "succeeded", "run_id": "run-1"}
    assert "progress" not in captured.out
    assert "diagnostics were removed" in captured.err


@pytest.mark.parametrize(
    "raised",
    [lambda: typer.Exit(0), lambda: SystemExit(0)],
    ids=["typer-exit-zero", "system-exit-zero"],
)
def test_strict_contract_preserves_success_on_deliberate_zero_exit(
    capsys: pytest.CaptureFixture[str], raised: Callable[[], BaseException]
) -> None:
    @json_stdout_contract(fail_closed_on_exception=True)
    def command(*, output_format: str) -> None:
        print(json.dumps({"status": "succeeded", "run_id": "run-zero"}))
        raise raised()

    with pytest.raises(BaseException) as caught:
        command(output_format="json")

    assert json.loads(capsys.readouterr().out) == {
        "status": "succeeded",
        "run_id": "run-zero",
    }
    failure = caught.value
    code = getattr(failure, "exit_code", getattr(failure, "code", None))
    assert code == 0


def test_strict_contract_replaces_stale_success_on_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @json_stdout_contract(fail_closed_on_exception=True)
    def command(*, output_format: str) -> None:
        print(json.dumps({"status": "succeeded", "run_id": "stale"}))
        raise RuntimeError("failed after progress")

    with pytest.raises(RuntimeError, match="failed after progress"):
        command(output_format="json")

    assert json.loads(capsys.readouterr().out) == {
        "error_type": "RuntimeError",
        "mutation_state": "unknown",
        "result": "error",
    }


@pytest.mark.parametrize(
    "raised",
    [lambda: RuntimeError("failed after launch"), lambda: typer.Exit(1)],
    ids=["unexpected-error", "nonzero-exit"],
)
def test_strict_contract_does_not_claim_no_mutation_without_a_document(
    capsys: pytest.CaptureFixture[str], raised: Callable[[], BaseException]
) -> None:
    launched_jobs = []

    @json_stdout_contract(fail_closed_on_exception=True)
    def command(*, output_format: str) -> None:
        launched_jobs.append("job-1")
        raise raised()

    with pytest.raises(BaseException) as caught:
        command(output_format="json")

    assert launched_jobs == ["job-1"]
    assert json.loads(capsys.readouterr().out) == {
        "error_type": type(caught.value).__name__,
        "mutation_state": "unknown",
        "result": "error",
    }


@pytest.mark.parametrize("status", ["blocked", "failed"])
def test_strict_contract_preserves_typed_failure_identity(
    capsys: pytest.CaptureFixture[str], status: str
) -> None:
    payload = {
        "status": status,
        "run_id": "run-recoverable",
        "resume_command": "npa workbench workflow submit spec --resume",
        "infrastructure_recovery": {"used": 2, "limit": 2, "exhausted": True},
        "launch_transaction": {
            "logical_launch_id": "logical-1",
            "job_id": "42",
            "recovery_decision": "operator_review",
        },
    }

    @json_stdout_contract(fail_closed_on_exception=True)
    def command(*, output_format: str) -> None:
        print(json.dumps(payload))
        raise typer.Exit(1)

    with pytest.raises(typer.Exit):
        command(output_format="json")

    assert json.loads(capsys.readouterr().out) == payload


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ("partial", "partial_cancellation"),
        ("VERIFICATION_UNAVAILABLE", "verification_failed"),
    ],
)
def test_strict_contract_preserves_failure_outcome_with_unknown_status(
    capsys: pytest.CaptureFixture[str], status: str, outcome: str
) -> None:
    payload = {"status": status, "outcome": outcome, "run_id": "run-typed"}

    @json_stdout_contract(fail_closed_on_exception=True)
    def command(*, output_format: str) -> None:
        print(json.dumps(payload))
        raise typer.Exit(1)

    with pytest.raises(typer.Exit):
        command(output_format="json")

    assert json.loads(capsys.readouterr().out) == payload


def test_strict_contract_rejects_success_status_even_with_error_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @json_stdout_contract(fail_closed_on_exception=True)
    def command(*, output_format: str) -> None:
        print(json.dumps({"status": "succeeded", "error": "stale"}))
        raise RuntimeError("failed after stale success")

    with pytest.raises(RuntimeError, match="failed after stale success"):
        command(output_format="json")

    assert json.loads(capsys.readouterr().out) == {
        "error_type": "RuntimeError",
        "mutation_state": "unknown",
        "result": "error",
    }
