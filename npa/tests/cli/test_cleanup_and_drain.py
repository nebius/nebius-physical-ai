"""Teardown gaps found while cleaning up after a PAIDF run."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

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
    assert "SkyPilot venv" in result.output
    assert "Terraform provider cache" in result.output
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


def test_cleanup_reports_iam_but_never_deletes_it(npa_home: Path) -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    # `npa configure` provisions a service account that no destroy path removes,
    # but it is frequently shared, so npa must name it rather than delete it.
    assert "not removed" in result.output
    assert "nebius iam service-account delete" in result.output


def test_cleanup_names_a_managed_job_that_still_blocks_teardown(
    monkeypatch: pytest.MonkeyPatch, npa_home: Path
) -> None:
    monkeypatch.setattr(
        cleanup_cli, "_nonterminal_jobs", lambda sky_bin: (["2"], "")
    )

    result = runner.invoke(app, ["cleanup"])

    assert "Managed jobs still non-terminal: 2" in result.output
    assert "block `sky down`" in result.output
    assert "stays PENDING forever" in result.output


def test_cleanup_prints_the_ordered_runbook(npa_home: Path) -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "Full teardown order" in result.output
    for step in ("sky jobs cancel", "agent destroy", "cluster down"):
        assert step in result.output


def test_cleanup_prints_the_runbook_when_it_finds_nothing(npa_home: Path) -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "No local NPA/SkyPilot residue" in result.output
    assert "Full teardown order" in result.output


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


def test_cluster_down_reports_the_budgets_that_will_block_the_drain(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.cli.cluster import terraform_lifecycle

    payload = {"items": [_pdb("kube-system", "coredns", 0)]}
    monkeypatch.setattr(
        "npa.cluster.drain.blocking_pod_disruption_budgets",
        lambda **kwargs: blocking_pod_disruption_budgets(runner=_pdb_runner(payload)),
    )

    terraform_lifecycle._report_drain_blockers(None)

    err = capsys.readouterr().err
    assert "drain-preview" in err
    assert "kube-system/coredns" in err


def test_cluster_down_preview_is_quiet_when_the_cluster_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.cli.cluster import terraform_lifecycle

    monkeypatch.setattr(
        "npa.cluster.drain.blocking_pod_disruption_budgets",
        lambda **kwargs: ([], "connection refused"),
    )

    terraform_lifecycle._report_drain_blockers(None)

    assert "skipped" in capsys.readouterr().err


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


# --- FTUE gaps from the third README walkthrough ------------------------------


def _option_names(command_path: list[str]) -> set[str]:
    import typer.main

    command = typer.main.get_command(app)
    for name in command_path:
        command = command.commands[name]  # type: ignore[attr-defined]
    names: set[str] = set()
    for param in command.params:
        # Click keeps the negative half of a `--x/--no-x` flag in secondary_opts,
        # so opts alone silently misses every off-switch.
        names.update(getattr(param, "opts", ()))
        names.update(getattr(param, "secondary_opts", ()))
    return names


def test_provision_if_absent_accepts_the_same_node_flags_as_cluster_up() -> None:
    # The README steers first-timers at provision-if-absent, but a 2x1-GPU shape
    # needed `cluster up --gpu-nodes 2`, which was not in the copy-paste path.
    assert "--gpu-nodes" in _option_names(["provision-if-absent"])
    assert "--cpu-nodes" in _option_names(["provision-if-absent"])


def test_node_flags_reach_the_cluster_up_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    from npa import provisioning
    from npa.cli.cluster.terraform_lifecycle import up_cmd

    expected_params = set(inspect.signature(up_cmd).parameters)
    seen: dict[str, object] = {}
    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *a, **k: False)

    def fake_up(**kwargs):  # noqa: ANN003 - test stub
        seen.update(kwargs)

    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle.up_cmd", fake_up, raising=False
    )
    monkeypatch.setattr(
        provisioning,
        "_resolve_project_runtime",
        lambda project: (
            "demo",
            type("E", (), {"project_id": "p", "tenant_id": "t"})(),
            type("S", (), {"checkpoint_bucket": "b", "prefix": "p", "endpoint_url": ""})(),
            "cr.example.invalid/reg",
        ),
    )
    monkeypatch.setattr(provisioning, "_runtime_env", lambda *a, **k: __import__("contextlib").nullcontext())

    provisioning.provision_if_absent(skip_s3=True, gpu_nodes=2, cpu_nodes=1)

    assert seen["gpu_nodes"] == 2
    assert seen["cpu_nodes"] == 1
    # Every Typer parameter must be passed explicitly, or omitted ones arrive as
    # OptionInfo sentinels and reach the Terraform overrides as objects.
    assert expected_params <= set(seen)


def test_dry_run_reports_the_requested_node_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa import provisioning

    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *a, **k: False)
    monkeypatch.setattr(
        provisioning,
        "_resolve_project_runtime",
        lambda project: (
            "demo",
            type("E", (), {"project_id": "p", "tenant_id": "t"})(),
            type("S", (), {"checkpoint_bucket": "b", "prefix": "p", "endpoint_url": ""})(),
            "cr.example.invalid/reg",
        ),
    )

    result = provisioning.provision_if_absent(
        skip_s3=True, dry_run=True, gpu_nodes=2, cpu_nodes=1
    )

    assert any("gpu_nodes=2" in action and "cpu_nodes=1" in action for action in result.actions)


def test_an_unavailable_capacity_api_does_not_advertise_the_dead_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `nebius capacity resource-advice list` can answer Unavailable; telling the
    # operator to run it is then the one thing that cannot help.
    import subprocess

    from npa.cli.cluster import capacity

    def capture(args):  # noqa: ANN001 - test stub
        if "resource-advice" in args:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="Unavailable")
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"items": [{"metadata": {"name": "q"}, "spec": {"limit": "2"}, "status": {"usage": "2"}}]}',
            stderr="",
        )

    monkeypatch.setattr(capacity, "gpu_quota_headroom", lambda *a, **k: (2, 2))

    message = capacity.gpu_capacity_error(
        capture,
        nebius_bin="nebius",
        tenant_id="tenant-x",
        region="us-central1",
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        required_gpus=2,
    )

    assert message is not None
    assert "did not answer" in message
    assert "see what is available with" not in message


def test_the_runbook_keeps_cleanup_before_the_skypilot_uninstall(npa_home: Path) -> None:
    """Order matters: cleanup reads the managed-job queue through SkyPilot.

    Uninstalling SkyPilot first turns that safety check into "SkyPilot is not
    installed", so a job still holding the controller goes unnoticed.
    """

    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    order = result.output.index("npa cleanup --yes")
    assert order < result.output.index("npa skypilot uninstall")


def test_the_report_says_it_does_not_touch_the_cloud(npa_home: Path) -> None:
    # `--yes` only clears local caches; the runbook made it look like teardown.
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "does NOT run these" in result.output


def test_an_empty_managed_job_queue_is_not_presented_as_a_failure(npa_home: Path) -> None:
    # `sky jobs cancel -a` raises ClusterNotUpError("No in-progress managed jobs")
    # when no controller exists; the runbook says so rather than leaving an
    # operator to wonder whether teardown already broke.
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "that is success" in result.output


def test_forgetting_the_last_project_leaves_no_dangling_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yaml as _yaml

    from npa.clients import config as config_module

    path = tmp_path / "config.yaml"
    path.write_text(
        _yaml.safe_dump({"default_project": "test-rtx", "projects": {"test-rtx": {"project_id": "p"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", path)

    assert config_module.forget_project("test-rtx") is True

    stored = _yaml.safe_load(path.read_text(encoding="utf-8"))
    assert stored.get("projects") == {}
    # Pointing at the literal "default" would name a project that does not exist.
    assert "default_project" not in stored


def test_forgetting_one_of_several_projects_repoints_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yaml as _yaml

    from npa.clients import config as config_module

    path = tmp_path / "config.yaml"
    path.write_text(
        _yaml.safe_dump(
            {"default_project": "a", "projects": {"a": {"project_id": "1"}, "b": {"project_id": "2"}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", path)

    config_module.forget_project("a")

    stored = _yaml.safe_load(path.read_text(encoding="utf-8"))
    assert stored["default_project"] == "b"


# --- a purge that Nebius scheduled but did not carry out ----------------------


def _bucket(state: str, purge_at: str) -> dict:
    return {
        "metadata": {"id": "bucket-1", "name": "npa-bucket-78978bfd"},
        "status": {"state": state, "purge_at": purge_at},
    }


def test_a_stalled_purge_is_not_reported_as_merely_slow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Past purge_at with the objects still there is a stall, not a slow wait."""

    from npa.cli import storage as storage_cli

    monkeypatch.setattr(
        "npa.clients.nebius.get_bucket_by_name",
        lambda project, name: _bucket("SCHEDULED_FOR_DELETION", "2000-01-01T00:00:00Z"),
    )
    monkeypatch.setattr(storage_cli, "_bucket_item", lambda p, n: _bucket(
        "SCHEDULED_FOR_DELETION", "2000-01-01T00:00:00Z"
    ))

    storage_cli._wait_for_bucket_gone("p", "npa-bucket-78978bfd", "target", 0)

    out = capsys.readouterr().out
    assert "purge_at has already passed" in out
    assert "stalled" in out
    assert "will be removed by" not in out


def test_a_purge_still_within_its_window_reads_as_slow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.cli import storage as storage_cli

    monkeypatch.setattr(
        "npa.clients.nebius.get_bucket_by_name",
        lambda project, name: _bucket("SCHEDULED_FOR_DELETION", "2999-01-01T00:00:00Z"),
    )
    monkeypatch.setattr(storage_cli, "_bucket_item", lambda p, n: _bucket(
        "SCHEDULED_FOR_DELETION", "2999-01-01T00:00:00Z"
    ))

    storage_cli._wait_for_bucket_gone("p", "npa-bucket-78978bfd", "target", 0)

    out = capsys.readouterr().out
    assert "will be removed by" in out
    assert "stalled" not in out


def test_the_reported_state_is_the_one_the_api_returns(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.cli import storage as storage_cli

    monkeypatch.setattr(storage_cli, "_bucket_item", lambda p, n: _bucket(
        "SCHEDULED_FOR_DELETION", "2999-01-01T00:00:00Z"
    ))

    assert storage_cli._scheduled_deletion_state("p", "b") == "SCHEDULED_FOR_DELETION"
    assert storage_cli._purge_is_overdue("p", "b") is False


def test_a_missing_bucket_has_no_state_and_is_not_overdue(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.cli import storage as storage_cli

    monkeypatch.setattr(storage_cli, "_bucket_item", lambda p, n: None)

    assert storage_cli._scheduled_deletion_state("p", "b") == ""
    assert storage_cli._purge_is_overdue("p", "b") is False


def test_provisioning_exposes_preemptible_like_cluster_up() -> None:
    # Getting two GPUs required TF_VAR_gpu_nodes_preemptible=true, which neither
    # the README path nor provision-if-absent exposed.
    for path in (["cluster", "up"], ["provision-if-absent"]):
        names = _option_names(path)
        assert "--preemptible" in names, path
        assert "--on-demand" in names, path


def test_preemptible_reaches_terraform_as_a_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa import provisioning

    seen: dict[str, object] = {}
    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *a, **k: False)
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle.up_cmd",
        lambda **kwargs: seen.update(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        provisioning,
        "_resolve_project_runtime",
        lambda project: (
            "demo",
            type("E", (), {"project_id": "p", "tenant_id": "t"})(),
            type("S", (), {"checkpoint_bucket": "b", "prefix": "p", "endpoint_url": ""})(),
            "cr.example.invalid/reg",
        ),
    )
    monkeypatch.setattr(
        provisioning, "_runtime_env", lambda *a, **k: __import__("contextlib").nullcontext()
    )

    provisioning.provision_if_absent(skip_s3=True, preemptible=True)

    assert seen["preemptible"] is True


def test_dry_run_reports_the_preemptible_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa import provisioning

    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *a, **k: False)
    monkeypatch.setattr(
        provisioning,
        "_resolve_project_runtime",
        lambda project: (
            "demo",
            type("E", (), {"project_id": "p", "tenant_id": "t"})(),
            type("S", (), {"checkpoint_bucket": "b", "prefix": "p", "endpoint_url": ""})(),
            "cr.example.invalid/reg",
        ),
    )

    result = provisioning.provision_if_absent(skip_s3=True, dry_run=True, preemptible=True)

    assert any("preemptible=true" in action for action in result.actions)


def test_configure_show_leads_with_what_is_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Leading with the blank template made an operator read `hf_REPLACE_ME` and
    # conclude nothing had been configured.
    monkeypatch.setattr("npa.clients.config.CONFIG_PATH", tmp_path / "config.yaml")

    result = runner.invoke(app, ["configure", "--show"])

    assert result.exit_code == 0, result.output
    assert result.output.index("Current configuration") < result.output.index("Credential setup")
