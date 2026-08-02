"""Teardown gaps found while cleaning up after a PAIDF run."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli import cleanup as cleanup_cli
from npa.cli.main import app
from npa.clients.config import ConfigError, _resolve_project_section
from npa.cluster.drain import (
    blocking_pod_disruption_budgets,
    describe_drain_expectation,
)


runner = CliRunner()


# --- `npa cleanup` -----------------------------------------------------------


@pytest.fixture()
def npa_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / ".npa").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_cleanup_is_registered_as_one_command() -> None:
    # The teardown runbook is six ordered steps; without a single entry point it
    # is easy to miss the hung Sky job or the leftover service account.
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "cleanup" in result.output


def test_cleanup_reports_local_leftovers_without_removing_them(npa_home: Path) -> None:
    (npa_home / ".npa" / "skypilot-venv").mkdir()
    (npa_home / ".npa" / "terraform-plugin-cache").mkdir()
    (npa_home / ".npa" / "clusters" / "gone").mkdir(parents=True)

    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert result.exit_code == 0, result.output
    assert "skypilot-venv" in result.output
    assert "terraform-plugin-cache" in result.output
    assert "clusters/gone" in result.output
    assert (npa_home / ".npa" / "skypilot-venv").exists()


def test_cleanup_yes_removes_only_local_caches(npa_home: Path) -> None:
    venv = npa_home / ".npa" / "skypilot-venv"
    venv.mkdir()
    sky = npa_home / ".sky"
    sky.mkdir()

    result = runner.invoke(app, ["cleanup", "--skip-jobs", "--yes"])

    assert result.exit_code == 0, result.output
    assert not venv.exists()
    assert not sky.exists()


def test_cleanup_keep_sky_leaves_skypilot_state(npa_home: Path) -> None:
    venv = npa_home / ".npa" / "skypilot-venv"
    venv.mkdir()
    sky = npa_home / ".sky"
    sky.mkdir()

    result = runner.invoke(app, ["cleanup", "--skip-jobs", "--yes", "--keep-sky"])

    assert result.exit_code == 0, result.output
    assert not venv.exists()
    assert sky.exists()


def test_cleanup_reports_service_accounts_but_never_deletes_them(npa_home: Path) -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "lerobot-training" in result.output
    assert "never deleted" in result.output
    # `npa configure` creates it and no destroy path removes it, so it must be
    # named -- but it is frequently shared, so npa must not remove it either.
    assert "reported, not deleted" in result.output


def test_cleanup_names_a_managed_job_that_still_blocks_teardown(
    monkeypatch: pytest.MonkeyPatch, npa_home: Path
) -> None:
    monkeypatch.setattr(
        cleanup_cli, "_nonterminal_jobs", lambda sky_bin: (["2"], "")
    )

    result = runner.invoke(app, ["cleanup"])

    assert "managed jobs still non-terminal: 2" in result.output
    assert "blocks `sky down`" in result.output
    assert "stays PENDING forever" in result.output


def test_cleanup_prints_the_ordered_runbook(npa_home: Path) -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "Full teardown order:" in result.output
    for step in ("sky jobs cancel", "agent destroy", "cluster down"):
        assert step in result.output


def test_cleanup_json_report(npa_home: Path) -> None:
    (npa_home / ".npa" / "config.yaml").write_text(
        yaml.safe_dump({"projects": {"demo": {}}}), encoding="utf-8"
    )

    result = runner.invoke(app, ["cleanup", "--skip-jobs", "--json"])

    payload = json.loads(result.output)
    assert payload["projects"] == ["demo"]
    assert "lerobot-training" in payload["service_accounts"]
    assert payload["runbook"]


# --- PodDisruptionBudget drain guidance --------------------------------------


def _pdb(namespace: str, name: str, allowed: int, *, desired: int = 1, current: int = 1) -> dict:
    return {
        "metadata": {"namespace": namespace, "name": name},
        "status": {
            "disruptionsAllowed": allowed,
            "desiredHealthy": desired,
            "currentHealthy": current,
        },
    }


def _pdb_runner(payload: dict, *, returncode: int = 0, stderr: str = ""):
    def run(cmd, **kwargs):  # noqa: ANN001 - test stub
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=json.dumps(payload), stderr=stderr
        )

    return run


def test_only_budgets_that_allow_no_evictions_are_reported() -> None:
    payload = {
        "items": [
            _pdb("kube-system", "coredns", 0),
            _pdb("kube-system", "metrics-server", 1),
            _pdb("kube-system", "cilium-operator", 0),
        ]
    }

    blockers, error = blocking_pod_disruption_budgets(runner=_pdb_runner(payload))

    assert error == ""
    assert [blocker.name for blocker in blockers] == ["cilium-operator", "coredns"]


def test_the_guidance_names_the_budgets_and_sets_expectations() -> None:
    payload = {"items": [_pdb("kube-system", "coredns", 0)]}
    blockers, _ = blocking_pod_disruption_budgets(runner=_pdb_runner(payload))

    guidance = describe_drain_expectation(blockers)

    assert "kube-system/coredns" in guidance
    # The reported symptom was a ~6 minute silence that looked like a hang.
    assert "look stalled" in guidance
    assert "expected" in guidance


def test_no_blocking_budgets_produces_no_guidance() -> None:
    payload = {"items": [_pdb("kube-system", "coredns", 1)]}
    blockers, _ = blocking_pod_disruption_budgets(runner=_pdb_runner(payload))

    assert blockers == []
    assert describe_drain_expectation(blockers) == ""


def test_an_unreachable_cluster_is_reported_not_assumed_clean() -> None:
    blockers, error = blocking_pod_disruption_budgets(
        runner=_pdb_runner({}, returncode=1, stderr="connection refused")
    )

    assert blockers == []
    assert "connection refused" in error


def test_cluster_down_previews_the_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli.cluster import terraform_lifecycle

    seen: list[str] = []
    monkeypatch.setattr(
        terraform_lifecycle,
        "_report_drain_blockers",
        lambda kubeconfig: seen.append("called"),
    )
    monkeypatch.setattr(terraform_lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(terraform_lifecycle, "_terraform_env", lambda nebius_bin: {})
    monkeypatch.setattr(
        terraform_lifecycle, "_run_stream", lambda *args, **kwargs: None
    )
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)

    result = runner.invoke(
        app, ["cluster", "down", "--force", "--terraform-dir", str(tf_dir)]
    )

    assert result.exit_code == 0, result.output
    assert seen == ["called"]


# --- explicit project alias that no longer exists ----------------------------


def test_an_explicit_alias_is_reported_when_no_projects_remain() -> None:
    # After teardown removes the alias, passing --project used to be ignored: the
    # command failed later complaining it could not tell which Nebius project to
    # use, never mentioning the alias the operator had passed.
    with pytest.raises(ConfigError) as excinfo:
        _resolve_project_section({"projects": {}}, "test-rtx")

    message = str(excinfo.value)
    assert "test-rtx" in message
    assert "--project-id" in message


def test_an_explicit_alias_is_reported_when_config_is_empty() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _resolve_project_section({}, "test-rtx")

    assert "test-rtx" in str(excinfo.value)


def test_an_explicit_alias_that_exists_still_resolves() -> None:
    section = _resolve_project_section({"projects": {"demo": {"project_id": "p1"}}}, "demo")

    assert section == {"project_id": "p1"}


def test_no_alias_still_falls_back_to_legacy_config() -> None:
    # Legacy flat configs have no `projects` map at all; they must keep working
    # when no alias is requested.
    section = _resolve_project_section({"workbenches": {"default": {"a": 1}}}, None)

    assert section["workbenches"] == {"default": {"a": 1}}


# --- a PENDING managed job explains itself in `workflow status` --------------


def test_status_explains_a_pending_job_whose_pod_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.workbench import workflow as workflow_cli
    from npa.orchestration.skypilot.job_blockers import JobBlockerReport, PodBlocker

    monkeypatch.setattr(
        workflow_cli,
        "_resolve_sky_bin",
        lambda sky_bin="": "/tmp/sky",
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_task_statuses",
        lambda job_id, **kwargs: [{"cluster_name": "sky-abc"}],
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.job_blockers.inspect_job_blockers",
        lambda **kwargs: JobBlockerReport(
            job_id="2",
            cluster_name="sky-abc",
            blockers=[PodBlocker(pod="worker-0", phase="Pending", reason="ImagePullBackOff")],
        ),
    )

    blockers = workflow_cli._stalled_job_blockers("2", "PENDING")

    assert blockers[0]["reason"] == "ImagePullBackOff"
    assert "retries this forever" in blockers[0]["remedy"]


def test_a_running_job_is_not_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.cli.workbench import workflow as workflow_cli

    def explode(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - must not run
        raise AssertionError("a healthy job must not be probed")

    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_task_statuses", explode
    )

    assert workflow_cli._stalled_job_blockers("2", "RUNNING") == []
    assert workflow_cli._stalled_job_blockers("", "PENDING") == []


def test_status_output_shows_the_blocked_pods() -> None:
    from npa.cli.workbench.workflow import OutputFormat, _emit_workflow_status

    result = _emit_workflow_status
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result(
            {
                "run_id": "paidf-1",
                "status": "PENDING",
                "sky_job_id": "2",
                "run_prefix_uri": "s3://b/p",
                "blockers": [
                    {
                        "pod": "worker-0",
                        "reason": "ImagePullBackOff",
                        "message": "403 Forbidden",
                        "remedy": "check pull permission",
                    }
                ],
            },
            OutputFormat.text,
        )

    output = buffer.getvalue()
    assert "blocked: 1 pod(s) cannot start" in output
    assert "worker-0: ImagePullBackOff - 403 Forbidden" in output
    assert "Suggested action: check pull permission" in output
