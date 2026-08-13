"""Exit-code contract for ``python -m npa.workflows.sim2real run``.

A command that orchestrates sub-operations (the staged loop + S3 upload) must
exit non-zero when one of them fails. A "blocked"/"failed" upload recorded in
the JSON report is not a substitute for the exit code: the submit/monitor
wrappers and CI gates rely on it.
"""

from __future__ import annotations

import pytest

from npa.workflows.sim2real import cli


class _FakeWorkflow:
    """Stand-in for Sim2RealWorkflow that returns a canned report."""

    report: dict = {}
    config = None

    def __init__(self, config) -> None:  # noqa: D401 - test double
        self.config = config
        type(self).config = config

    def run_staged(self, *, upload=None, initial_quality=None) -> dict:
        return type(self).report


def _run_with_report(monkeypatch, report: dict) -> int:
    _FakeWorkflow.report = report
    monkeypatch.setattr(cli, "Sim2RealWorkflow", _FakeWorkflow)
    return cli.main(["run", "--run-id", "exit-code-test"])


def test_run_exits_nonzero_on_blocked_upload(monkeypatch, capsys) -> None:
    code = _run_with_report(
        monkeypatch,
        {"upload": {"status": "blocked", "reason": "S3 upload failed: boom"}},
    )
    assert code == 1
    assert "blocked" in capsys.readouterr().err


def test_run_exits_nonzero_on_failed_upload(monkeypatch) -> None:
    code = _run_with_report(monkeypatch, {"upload": {"status": "failed"}})
    assert code == 1


@pytest.mark.parametrize("status", ["uploaded", "skipped"])
def test_run_exits_zero_on_successful_or_skipped_upload(monkeypatch, status) -> None:
    code = _run_with_report(monkeypatch, {"upload": {"status": status}})
    assert code == 0


def test_run_exits_zero_when_only_rerun_serve_blocked(monkeypatch) -> None:
    # rerun-serve is best-effort viz (engine records it WARN/CONTINUE); it must
    # not fail the run on its own.
    code = _run_with_report(
        monkeypatch,
        {"upload": {"status": "uploaded"}, "rerun_serve": {"status": "blocked"}},
    )
    assert code == 0


def test_run_cli_defaults_learning_rate_and_timeout(monkeypatch) -> None:
    monkeypatch.delenv("LEARNING_RATE", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", raising=False)
    _run_with_report(monkeypatch, {"upload": {"status": "skipped"}})
    assert _FakeWorkflow.config.learning_rate == 0.08
    assert _FakeWorkflow.config.k8s_job_timeout_s == 0


def test_run_cli_propagates_learning_rate_and_positive_timeout(monkeypatch) -> None:
    _FakeWorkflow.report = {"upload": {"status": "skipped"}}
    monkeypatch.setattr(cli, "Sim2RealWorkflow", _FakeWorkflow)
    code = cli.main(
        [
            "run",
            "--run-id",
            "override-test",
            "--learning-rate",
            "0.12",
            "--k8s-job-timeout-s",
            "3600",
        ]
    )
    assert code == 0
    assert _FakeWorkflow.config.learning_rate == 0.12
    assert _FakeWorkflow.config.k8s_job_timeout_s == 3600


def test_run_cli_help_documents_uncapped_timeout_and_learning_rate_scope(
    capsys,
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "0 is" in output and "uncapped." in output
    assert "does not override" in output
    assert "Isaac RSL-RL PPO optimizer" in output
