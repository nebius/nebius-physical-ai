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
    DrainInventory,
    DrainPreviewIssue,
    blocking_pod_disruption_budgets,
    classify_drain_preview_failure,
    drain_inventory,
    describe_drain_expectation,
    describe_preview_unavailable,
)
from npa.provisioning_preflight import (
    GIB,
    NETWORK_SSD_BYTES_QUOTA,
    ExistingCapacity,
    QuotaObservation,
)


runner = CliRunner()


def _ample_quota_observations(_tenant, _region, names):
    return {
        name: QuotaObservation(
            name=name,
            used=0,
            limit=(20_000 * GIB if name == NETWORK_SSD_BYTES_QUOTA else 100),
            state="known",
        )
        for name in names
    }


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


def test_cleanup_yes_removes_only_local_caches(monkeypatch: pytest.MonkeyPatch, npa_home: Path) -> None:
    venv = npa_home / ".npa" / "skypilot-venv"
    venv.mkdir()
    sky = npa_home / ".sky"
    sky.mkdir()

    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: ([], ""))
    result = runner.invoke(app, ["cleanup", "--yes"])

    assert result.exit_code == 0, result.output
    assert not venv.exists()
    assert sky.exists()
    assert "Preserved shared SkyPilot state" in result.output


def test_exact_never_submitted_project_cleanup_with_keep_sky_skips_default_queue(
    monkeypatch: pytest.MonkeyPatch, npa_home: Path
) -> None:
    from npa.orchestration.npa_workflow.submission_state import update_submission_state

    update_submission_state("demo", "reserved", {"launch_state": "reserved"})
    monkeypatch.setattr(
        cleanup_cli,
        "_nonterminal_jobs",
        lambda _sky: (_ for _ in ()).throw(
            AssertionError("exact safe shortcut must not touch default SkyPilot state")
        ),
    )

    result = runner.invoke(app, ["cleanup", "--project", "demo", "--keep-sky"])

    assert result.exit_code == 0, result.output


def test_destructive_global_sky_cleanup_still_verifies_queue(
    monkeypatch: pytest.MonkeyPatch, npa_home: Path
) -> None:
    from npa.orchestration.npa_workflow.submission_state import update_submission_state

    update_submission_state("demo", "reserved", {"launch_state": "reserved"})
    (npa_home / ".sky").mkdir()
    calls: list[str] = []
    monkeypatch.setattr(
        cleanup_cli,
        "_nonterminal_jobs",
        lambda sky: calls.append(sky) or ([], "", "verified_empty"),
    )

    result = runner.invoke(app, ["cleanup", "--project", "demo", "--yes"])

    assert result.exit_code == 0, result.output
    assert calls == [""]


def test_cleanup_keep_sky_leaves_skypilot_state(monkeypatch: pytest.MonkeyPatch, npa_home: Path) -> None:
    venv = npa_home / ".npa" / "skypilot-venv"
    venv.mkdir()
    sky = npa_home / ".sky"
    sky.mkdir()

    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: ([], ""))
    result = runner.invoke(app, ["cleanup", "--yes", "--keep-sky"])

    assert result.exit_code == 0, result.output
    assert not venv.exists()
    assert sky.exists()


@pytest.mark.parametrize(
    ("selected", "owner_project", "expected"),
    [
        ("project-a", "", "no controller owner"),
        ("project-a", "project-b", "another or unselected project"),
        ("project-a", "project-a", "selected project owns a controller"),
    ],
)
def test_global_sky_state_is_never_authorized_by_controller_ownership(
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
    owner_project: str,
    expected: str,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "npa.controller_ownership.controller_owner",
        lambda: SimpleNamespace(project_id=owner_project) if owner_project else None,
    )

    reason = cleanup_cli._shared_sky_preservation_reason(selected)

    assert expected in reason
    assert "do not prove exclusive ownership" in reason


def test_cleanup_reports_iam_but_never_deletes_it(npa_home: Path) -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    # `npa configure` provisions a service account that no destroy path removes,
    # but it is frequently shared, so npa must name it rather than delete it.
    assert "not removed" in result.output
    assert "npa storage service-account reconcile" in result.output
    assert "nebius iam service-account delete" not in result.output


def test_cleanup_names_a_managed_job_that_still_blocks_teardown(
    monkeypatch: pytest.MonkeyPatch, npa_home: Path
) -> None:
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: (["2"], ""))

    result = runner.invoke(app, ["cleanup"])

    assert "Managed jobs still non-terminal: 2" in result.output
    assert "block `npa skypilot cleanup-controller`" in result.output
    assert "stays PENDING forever" in result.output


def test_cleanup_prints_the_ordered_runbook(npa_home: Path) -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "Full teardown order" in result.output
    for step in ("workflow cancel", "agent destroy", "cluster down"):
        assert step in result.output
    assert "npa skypilot cleanup-controller --yes" in result.output


def test_cleanup_prints_the_runbook_when_it_finds_nothing(npa_home: Path) -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "No local NPA/SkyPilot residue" in result.output
    assert "Full teardown order" in result.output


# --- PodDisruptionBudget drain guidance --------------------------------------


def _pdb(
    namespace: str,
    name: str,
    allowed: int,
    *,
    desired: int = 1,
    current: int = 1,
    min_available: int | None = 1,
) -> dict:
    spec = {} if min_available is None else {"minAvailable": min_available}
    return {
        "metadata": {"namespace": namespace, "name": name},
        "spec": spec,
        "status": {
            "disruptionsAllowed": allowed,
            "desiredHealthy": desired,
            "currentHealthy": current,
        },
    }


def _pdb_runner(payload: dict, *, returncode: int = 0, stderr: str = ""):
    def run(cmd, **kwargs):  # noqa: ANN001 - test stub
        return subprocess.CompletedProcess(cmd, returncode, stdout=json.dumps(payload), stderr=stderr)

    return run


def test_only_budgets_that_allow_no_evictions_are_reported() -> None:
    payload = {
        "items": [
            _pdb("kube-system", "coredns", 0),
            _pdb("kube-system", "metrics-server", 1),
            _pdb("kube-system", "cilium-operator", 0),
        ]
    }

    blockers, issue = blocking_pod_disruption_budgets(runner=_pdb_runner(payload))

    assert issue is None
    assert [blocker.name for blocker in blockers] == ["cilium-operator", "coredns"]


def test_the_guidance_names_the_budgets_and_sets_expectations() -> None:
    payload = {"items": [_pdb("kube-system", "coredns", 0)]}
    blockers, _ = blocking_pod_disruption_budgets(runner=_pdb_runner(payload))

    guidance = describe_drain_expectation(blockers)

    assert "kube-system/coredns" in guidance
    # The reported symptom was a ~6 minute silence that looked like a hang.
    assert "look stalled" in guidance
    assert "expected" in guidance
    assert "retry/wait" in guidance


def test_no_blocking_budgets_produces_no_guidance() -> None:
    payload = {"items": [_pdb("kube-system", "coredns", 1)]}
    blockers, _ = blocking_pod_disruption_budgets(runner=_pdb_runner(payload))

    assert blockers == []
    assert describe_drain_expectation(blockers) == ""


def test_an_unreachable_cluster_is_reported_not_assumed_clean() -> None:
    blockers, issue = blocking_pod_disruption_budgets(runner=_pdb_runner({}, returncode=1, stderr="connection refused"))

    assert blockers == []
    assert issue is not None
    assert issue.kind == "api"


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("error: You must be logged in to the server (Unauthorized)", "authentication"),
        (
            'poddisruptionbudgets.policy is forbidden: User "operator" cannot list resource',
            "authorization",
        ),
        ("error loading config file: context demo was not found", "kubeconfig"),
        ("Unable to connect to the server: dial tcp: connection refused", "api"),
    ],
)
def test_drain_preview_failure_causes_are_classified(message: str, kind: str) -> None:
    assert classify_drain_preview_failure(message).kind == kind


def test_noninteractive_preview_disables_browser_auth_in_the_kubeconfig(
    tmp_path: Path,
) -> None:
    import yaml

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Config",
                "current-context": "npa-cluster",
                "users": [
                    {
                        "name": "npa-user",
                        "user": {
                            "exec": {
                                "apiVersion": "client.authentication.k8s.io/v1beta1",
                                "command": "/usr/local/bin/nebius",
                                "args": [
                                    "mk8s",
                                    "v1",
                                    "cluster",
                                    "get-token",
                                    "--profile",
                                    "npa-service-account",
                                    "--format",
                                    "json",
                                ],
                                "interactiveMode": "IfAvailable",
                            }
                        },
                    }
                ],
            }
        )
    )
    observed: dict[str, object] = {}

    def run(cmd, **kwargs):  # noqa: ANN001 - subprocess test double
        generated = Path(kwargs["env"]["KUBECONFIG"])
        rendered = yaml.safe_load(generated.read_text())
        exec_config = rendered["users"][0]["user"]["exec"]
        observed.update(
            {
                "stdin": kwargs.get("stdin"),
                "interactive_mode": exec_config["interactiveMode"],
                "args": exec_config["args"],
            }
        )
        return subprocess.CompletedProcess(cmd, 0, stdout='{"items": []}', stderr="")

    blockers, issue = blocking_pod_disruption_budgets(kubeconfig=str(kubeconfig), runner=run)

    assert blockers == []
    assert issue is None
    assert observed["stdin"] is subprocess.DEVNULL
    assert observed["interactive_mode"] == "Never"
    assert observed["args"][0] == "--no-browser"
    assert "npa-service-account" in observed["args"]


def test_preview_unavailable_explains_scope_safety_and_operator_action() -> None:
    issue = DrainPreviewIssue(
        kind="authorization",
        summary="Kubernetes RBAC denied listing policy/v1 PodDisruptionBudgets",
    )

    message = describe_preview_unavailable(issue)

    assert "best-effort drain preview" in message
    assert "teardown will continue" in message
    assert "not verified" in message
    assert "only the preview" in message
    assert "read access" in message
    assert "will not broaden RBAC" in message
    assert "policy/v1 PodDisruptionBudgets" in message


def _node(name: str) -> dict:
    return {
        "kind": "Node",
        "metadata": {
            "name": name,
            "labels": {"nebius.com/node-group-id": "cpu-pool"},
        },
    }


def _pod(name: str, labels: dict[str, str], owner: str) -> dict:
    return {
        "kind": "Pod",
        "metadata": {
            "namespace": "kube-system",
            "name": name,
            "labels": labels,
            "ownerReferences": [{"kind": "Deployment", "name": owner, "controller": True}],
        },
        "spec": {"nodeName": "cpu-0"},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def _selected_pdb(name: str, labels: dict[str, str], allowed: int) -> dict:
    item = _pdb("kube-system", name, allowed)
    item["kind"] = "PodDisruptionBudget"
    item["spec"]["selector"] = {"matchLabels": labels}
    return item


def test_shared_inventory_finds_system_pdbs_on_one_node_cpu_pool() -> None:
    payload = {
        "items": [
            _node("cpu-0"),
            _pod("cilium-0", {"app": "cilium-operator"}, "cilium-operator"),
            _pod("coredns-0", {"k8s-app": "kube-dns"}, "coredns"),
            _pod("coredns-1", {"k8s-app": "kube-dns"}, "coredns"),
            _pod("autoscaler-0", {"app": "coredns-autoscaler"}, "coredns-autoscaler"),
            _pod("metrics-0", {"k8s-app": "metrics-server"}, "metrics-server"),
            _selected_pdb("cilium-operator", {"app": "cilium-operator"}, 0),
            _selected_pdb("coredns", {"k8s-app": "kube-dns"}, 1),
            _selected_pdb("coredns-autoscaler", {"app": "coredns-autoscaler"}, 0),
            _selected_pdb("metrics-server", {"k8s-app": "metrics-server"}, 0),
        ]
    }
    calls: list[list[str]] = []

    def run(cmd, **kwargs):  # noqa: ANN001 - subprocess test double
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    inventory, issue = drain_inventory(runner=run)

    assert issue is None
    assert inventory is not None
    assert inventory.nodes == ("cpu-0",)
    assert [item.name for item in inventory.blockers] == [
        "cilium-operator",
        "coredns",
        "coredns-autoscaler",
        "metrics-server",
    ]
    assert all(item.one_node_pool for item in inventory.blockers)
    coredns = next(item for item in inventory.blockers if item.name == "coredns")
    assert coredns.disruptions_allowed == 1
    assert len(coredns.matching_pods) == 2
    assert coredns.workloads == ("kube-system/Deployment/coredns",)
    assert calls == [
        [
            "kubectl",
            "get",
            "nodes,pods,poddisruptionbudgets",
            "--all-namespaces",
            "-o",
            "json",
        ]
    ]


def test_unhealthy_always_allow_pod_is_not_a_false_blocker() -> None:
    pod = _pod("metrics-0", {"k8s-app": "metrics-server"}, "metrics-server")
    pod["status"]["conditions"][0]["status"] = "False"
    pdb = _selected_pdb("metrics-server", {"k8s-app": "metrics-server"}, 0)
    pdb["spec"]["unhealthyPodEvictionPolicy"] = "AlwaysAllow"

    inventory, issue = drain_inventory(runner=_pdb_runner({"items": [_node("cpu-0"), pod, pdb]}))

    assert issue is None
    assert inventory is not None
    assert inventory.blockers == ()


def test_unhealthy_if_healthy_budget_uses_reported_health() -> None:
    pod = _pod("metrics-0", {"k8s-app": "metrics-server"}, "metrics-server")
    pod["status"]["conditions"][0]["status"] = "False"
    pdb = _selected_pdb("metrics-server", {"k8s-app": "metrics-server"}, 0)

    healthy_payload = {"items": [_node("cpu-0"), pod, pdb]}
    healthy_inventory, issue = drain_inventory(runner=_pdb_runner(healthy_payload))

    assert issue is None
    assert healthy_inventory is not None
    assert healthy_inventory.blockers == ()

    pdb["status"]["currentHealthy"] = 0
    unhealthy_inventory, issue = drain_inventory(runner=_pdb_runner({"items": [_node("cpu-0"), pod, pdb]}))

    assert issue is None
    assert unhealthy_inventory is not None
    assert [item.name for item in unhealthy_inventory.blockers] == ["metrics-server"]


def test_cluster_down_reports_the_budgets_that_will_block_the_drain(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.cli.cluster import terraform_lifecycle

    payload = {"items": [_pdb("kube-system", "coredns", 0)]}
    monkeypatch.setattr(
        "npa.cluster.drain.drain_inventory",
        lambda **kwargs: drain_inventory(runner=_pdb_runner(payload)),
    )

    terraform_lifecycle._report_drain_blockers(None)

    err = capsys.readouterr().err
    assert "drain-preview" in err
    assert "kube-system/coredns" in err


def test_cluster_down_preview_explains_when_the_cluster_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.cli.cluster import terraform_lifecycle

    monkeypatch.setattr(
        "npa.cluster.drain.drain_inventory",
        lambda **kwargs: (
            None,
            DrainPreviewIssue(kind="api", summary="the Kubernetes API endpoint could not be reached"),
        ),
    )

    terraform_lifecycle._report_drain_blockers(None)

    message = capsys.readouterr().err
    assert "teardown will continue" in message
    assert "not verified" in message
    assert "Kubernetes API" in message


def test_cluster_down_explains_backoff_without_mutating_any_pdb(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.cli.cluster import terraform_lifecycle

    blockers, issue = blocking_pod_disruption_budgets(
        runner=_pdb_runner(
            {
                "items": [
                    _pdb("kube-system", "coredns", 0),
                    _pdb("customer", "orders", 0),
                ]
            }
        )
    )
    assert issue is None
    monkeypatch.setattr(
        "npa.cluster.drain.drain_inventory",
        lambda **kwargs: (
            DrainInventory((), 0, len(blockers), tuple(blockers)),
            None,
        ),
    )

    terraform_lifecycle._report_drain_blockers(None)

    message = capsys.readouterr().err
    assert "will deny at least one eviction" in message
    assert "kube-system/coredns" in message
    assert "customer/orders" in message
    assert "will not patch budgets" in message
    assert "force-delete" in message


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

    monkeypatch.setattr("npa.orchestration.skypilot.workflow.workflow_task_statuses", explode)

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
    monkeypatch.setattr(
        "npa.provisioning_preflight.read_provider_quotas",
        _ample_quota_observations,
    )
    monkeypatch.setattr(
        "npa.provisioning_preflight.discover_existing_capacity",
        lambda **kwargs: ExistingCapacity(),
    )

    def fake_up(**kwargs):  # noqa: ANN003 - test stub
        seen.update(kwargs)

    monkeypatch.setattr("npa.cli.cluster.terraform_lifecycle.up_cmd", fake_up, raising=False)
    monkeypatch.setattr(
        provisioning,
        "_resolve_project_runtime",
        lambda project: (
            "demo",
            type("E", (), {"project_id": "p", "tenant_id": "t", "region": "r"})(),
            type("S", (), {"checkpoint_bucket": "b", "prefix": "p", "endpoint_url": ""})(),
            "cr.example.invalid/reg",
        ),
    )
    monkeypatch.setattr(
        provisioning,
        "_runtime_env",
        lambda *a, **k: __import__("contextlib").nullcontext(),
    )

    provisioning.provision_if_absent(skip_s3=True, gpu_nodes=2, cpu_nodes=1)

    assert seen["gpu_nodes"] == 2
    assert seen["cpu_nodes"] == 1
    # Every Typer parameter must be passed explicitly, or omitted ones arrive as
    # OptionInfo sentinels and reach the Terraform overrides as objects.
    assert expected_params <= set(seen)


def test_nested_cluster_preflight_inherits_exact_outer_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import provisioning
    from npa.provisioning_preflight import (
        WholePathPreflightPlan,
        current_resolved_plan,
        resolve_topology,
    )

    plan = WholePathPreflightPlan(
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="region-a",
        topology=resolve_topology(cluster_name="provider-cluster", gpu_nodes=2),
        decision="ready",
    )
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(provisioning, "_build_provision_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *_a, **_k: False)
    monkeypatch.setattr(
        provisioning,
        "_resolve_project_runtime",
        lambda _project: (
            "demo",
            type("E", (), {"project_id": "project-a", "tenant_id": "tenant-a", "region": "region-a"})(),
            type(
                "S",
                (),
                {
                    "checkpoint_bucket": "bucket",
                    "endpoint_url": "",
                    "aws_access_key_id": "",
                    "aws_secret_access_key": "",
                },
            )(),
            "registry",
        ),
    )
    seen: list[object] = []

    def fake_up(**_kwargs):
        seen.append(current_resolved_plan())

    monkeypatch.setattr("npa.cli.cluster.terraform_lifecycle.up_cmd", fake_up)
    result = provisioning.provision_if_absent(
        project="demo",
        cluster_name="provider-cluster",
        context_name="explicit-context",
        skip_s3=True,
    )

    assert seen == [plan]
    assert current_resolved_plan() is None
    assert result.preflight == plan.to_dict()


def test_provisioning_rollback_uses_explicit_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import provisioning
    from npa.provisioning_preflight import WholePathPreflightPlan, resolve_topology

    plan = WholePathPreflightPlan(
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="region-a",
        topology=resolve_topology(cluster_name="provider-cluster"),
        decision="ready",
    )
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(provisioning, "_build_provision_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *_a, **_k: False)
    monkeypatch.setattr(
        provisioning,
        "_resolve_project_runtime",
        lambda _project: (
            "demo",
            type("E", (), {"project_id": "project-a", "tenant_id": "tenant-a", "region": "region-a"})(),
            type(
                "S",
                (),
                {
                    "checkpoint_bucket": "bucket",
                    "endpoint_url": "",
                    "aws_access_key_id": "",
                    "aws_secret_access_key": "",
                },
            )(),
            "registry",
        ),
    )
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle.up_cmd",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("apply failed")),
    )
    rollback: dict[str, object] = {}
    monkeypatch.setattr(
        provisioning,
        "_rollback_owned_cluster",
        lambda _operation, **kwargs: rollback.update(kwargs) or False,
    )

    with pytest.raises(RuntimeError, match="apply failed"):
        provisioning.provision_if_absent(
            project="demo",
            cluster_name="provider-cluster",
            context_name="explicit-context",
            skip_s3=True,
        )

    assert rollback["context"] == "explicit-context"


def test_dry_run_reports_the_requested_node_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = provisioning.provision_if_absent(skip_s3=True, dry_run=True, gpu_nodes=2, cpu_nodes=1)

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


def test_the_runbook_does_not_run_skypilot_uninstall_after_cleanup(
    npa_home: Path,
) -> None:
    """Cleanup already removes the isolated venv, so a later uninstall is dead."""

    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "npa cleanup --full --yes" in result.output
    assert "npa skypilot uninstall" not in result.output


def test_the_report_says_it_does_not_touch_the_cloud(npa_home: Path) -> None:
    # `--yes` only clears local caches; the runbook made it look like teardown.
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "cleanup never implies the preceding cloud steps" in result.output


def test_an_empty_managed_job_queue_is_not_presented_as_a_failure(
    npa_home: Path,
) -> None:
    # The NPA-only sequence makes both workflow and controller cleanup explicit
    # and repeat-safe instead of prescribing a raw global SkyPilot cancel.
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert "repeat-safe" in result.output
    assert "sky jobs cancel" not in result.output
    assert "sky down" not in result.output
    assert "kubectl delete" not in result.output
    assert "nebius iam" not in result.output


def test_forgetting_the_last_project_leaves_no_dangling_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yaml as _yaml

    from npa.clients import config as config_module

    path = tmp_path / "config.yaml"
    path.write_text(
        _yaml.safe_dump(
            {
                "default_project": "test-rtx",
                "projects": {"test-rtx": {"project_id": "p"}},
            }
        ),
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
            {
                "default_project": "a",
                "projects": {"a": {"project_id": "1"}, "b": {"project_id": "2"}},
            }
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
    monkeypatch.setattr(
        storage_cli,
        "_bucket_item",
        lambda p, n: _bucket("SCHEDULED_FOR_DELETION", "2000-01-01T00:00:00Z"),
    )

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
    monkeypatch.setattr(
        storage_cli,
        "_bucket_item",
        lambda p, n: _bucket("SCHEDULED_FOR_DELETION", "2999-01-01T00:00:00Z"),
    )

    storage_cli._wait_for_bucket_gone("p", "npa-bucket-78978bfd", "target", 0)

    out = capsys.readouterr().out
    assert "will be removed by" in out
    assert "stalled" not in out


def test_the_reported_state_is_the_one_the_api_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import storage as storage_cli

    monkeypatch.setattr(
        storage_cli,
        "_bucket_item",
        lambda p, n: _bucket("SCHEDULED_FOR_DELETION", "2999-01-01T00:00:00Z"),
    )

    assert storage_cli._scheduled_deletion_state("p", "b") == "SCHEDULED_FOR_DELETION"
    assert storage_cli._purge_is_overdue("p", "b") is False


def test_a_missing_bucket_has_no_state_and_is_not_overdue(
    monkeypatch: pytest.MonkeyPatch,
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


def test_preemptible_reaches_terraform_as_a_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa import provisioning

    seen: dict[str, object] = {}
    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *a, **k: False)
    monkeypatch.setattr(
        "npa.provisioning_preflight.read_provider_quotas",
        _ample_quota_observations,
    )
    monkeypatch.setattr(
        "npa.provisioning_preflight.discover_existing_capacity",
        lambda **kwargs: ExistingCapacity(),
    )
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
            type("E", (), {"project_id": "p", "tenant_id": "t", "region": "r"})(),
            type("S", (), {"checkpoint_bucket": "b", "prefix": "p", "endpoint_url": ""})(),
            "cr.example.invalid/reg",
        ),
    )
    monkeypatch.setattr(
        provisioning,
        "_runtime_env",
        lambda *a, **k: __import__("contextlib").nullcontext(),
    )

    provisioning.provision_if_absent(skip_s3=True, preemptible=True)

    assert seen["preemptible"] is True


def test_dry_run_reports_the_preemptible_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_configure_show_leads_with_what_is_saved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Leading with the blank template made an operator read `hf_REPLACE_ME` and
    # conclude nothing had been configured.
    monkeypatch.setattr("npa.clients.config.CONFIG_PATH", tmp_path / "config.yaml")

    result = runner.invoke(app, ["configure", "--show"])

    assert result.exit_code == 0, result.output
    assert result.output.index("Current configuration") < result.output.index("Credential setup")
