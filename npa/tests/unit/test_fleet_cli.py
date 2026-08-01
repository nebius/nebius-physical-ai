"""Unit tests for `npa fleet`: spec merge/validation, tfvars rendering, CLI wiring.

These tests must not touch real infrastructure: they exercise pure spec/tfvars
logic and the Typer command surface (help + validation + plan), mocking the
lifecycle at the call site for deploy/destroy paths.
"""

from __future__ import annotations

import json
import textwrap

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.fleet import spec_from_mapping
from npa.fleet.spec import (
    ClusterSpec,
    FleetSpec,
    FleetSpecError,
    NodePoolSpec,
    ProjectSpec,
    load_spec,
)
from npa.fleet.tfvars import patch_provider_domain, provider_domain, render_tfvars

runner = CliRunner()


def _rtx_profile() -> dict:
    return {
        "cpu_nodes": {"count": 1, "platform": "cpu-d3", "preset": "48vcpu-192gb"},
        "gpu_nodes": {"count": 1, "platform": "gpu-rtx6000", "preset": "1gpu-24vcpu-218gb"},
        "enable_filestore": True,
    }


def _base_mapping() -> dict:
    return {
        "apiVersion": "npa.fleet/v0.0.1",
        "name": "fleet1-test",
        "tenant_id": "tenant-x",
        "region": "us-central1",
        "project_prefix": "fleet1-test-",
        "defaults": _rtx_profile(),
        "projects": [{"name": "a"}, {"name": "b"}],
    }


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_help() -> None:
    result = runner.invoke(app, ["fleet", "--help"])
    assert result.exit_code == 0
    assert "fleet" in result.output.lower()


def test_deploy_help_documents_spec_and_projects() -> None:
    result = runner.invoke(app, ["fleet", "deploy", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output
    assert "--create-projects" in result.output
    assert "--k8s-training-ref" in result.output


# --------------------------------------------------------------------------- #
# Spec parsing / defaults merge (identical vs custom vs mixed)
# --------------------------------------------------------------------------- #
def test_identical_fleet_applies_defaults_to_every_project() -> None:
    spec = spec_from_mapping(_base_mapping())
    spec.validate()
    assert [p.display_name(spec.project_prefix) for p in spec.projects] == [
        "fleet1-test-a",
        "fleet1-test-b",
    ]
    targets = spec.cluster_targets()
    assert len(targets) == 2
    for _project, cluster in targets:
        assert cluster.cpu_count() == 1
        assert cluster.gpu_nodes.platform == "gpu-rtx6000"
        assert cluster.gpu_nodes.preset == "1gpu-24vcpu-218gb"
        assert cluster.name == "cluster"


def test_custom_and_mixed_clusters() -> None:
    data = _base_mapping()
    data["projects"] = [
        {"name": "a"},  # identical
        {
            "name": "b",
            "clusters": [
                {
                    "name": "train",
                    "gpu_nodes": {"count": 2, "platform": "gpu-h200-sxm", "preset": "8gpu-128vcpu-1600gb"},
                    "enable_gpu_cluster": True,
                    "infiniband_fabric": "us-central1-a",
                },
                {"name": "infer"},  # inherits defaults
            ],
        },
        {"project_id": "project-existing", "clusters": [{}]},
    ]
    spec = spec_from_mapping(data)
    spec.validate()
    a, b, existing = spec.projects
    assert [c.name for c in a.clusters] == ["cluster"]
    assert [c.name for c in b.clusters] == ["train", "infer"]
    train = b.clusters[0]
    assert train.gpu_nodes.count == 2
    assert train.resolved_enable_gpu_cluster() is True
    # infer inherits the RTX default profile.
    assert b.clusters[1].gpu_nodes.preset == "1gpu-24vcpu-218gb"
    # Existing project referenced by id -> not created.
    assert existing.project_id == "project-existing"
    assert existing.display_name(spec.project_prefix) == "project-existing"


def test_single_gpu_preset_auto_disables_gpu_cluster() -> None:
    spec = spec_from_mapping(_base_mapping())
    cluster = spec.projects[0].clusters[0]
    # RTX single-GPU preset must not enable GPU clustering.
    assert cluster.resolved_enable_gpu_cluster() is False


def test_enable_gpu_cluster_requires_8gpu_preset_and_fabric() -> None:
    cluster = ClusterSpec(
        name="c",
        gpu_nodes=NodePoolSpec(count=1, platform="gpu-rtx6000", preset="1gpu-24vcpu-218gb"),
        enable_gpu_cluster=True,
    )
    with pytest.raises(FleetSpecError, match="8-GPU preset"):
        cluster.validate()


def test_cluster_needs_at_least_one_node() -> None:
    cluster = ClusterSpec(name="empty")
    with pytest.raises(FleetSpecError, match="at least one CPU or GPU node"):
        cluster.validate()


def test_project_needs_name_or_id() -> None:
    spec = FleetSpec(name="f", projects=[ProjectSpec(clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])])
    with pytest.raises(FleetSpecError, match="name.*project_id"):
        spec.validate()


def test_bad_apiversion_rejected() -> None:
    data = _base_mapping()
    data["apiVersion"] = "npa.fleet/v9"
    with pytest.raises(FleetSpecError, match="unsupported apiVersion"):
        spec_from_mapping(data)


def test_prefix_not_double_applied() -> None:
    data = _base_mapping()
    data["projects"] = [{"name": "fleet1-test-a"}]
    spec = spec_from_mapping(data)
    assert spec.projects[0].display_name(spec.project_prefix) == "fleet1-test-a"


# --------------------------------------------------------------------------- #
# tfvars / main.tf rendering
# --------------------------------------------------------------------------- #
def test_render_tfvars_rtx_single_gpu() -> None:
    spec = spec_from_mapping(_base_mapping())
    cluster = spec.projects[0].clusters[0]
    tf = render_tfvars(cluster, ssh_public_key="ssh-ed25519 AAAA me")
    assert 'cluster_name = "cluster"' in tf
    assert "cpu_nodes_fixed_count = 1" in tf
    assert 'cpu_nodes_preset = "48vcpu-192gb"' in tf
    assert "gpu_nodes_fixed_count_per_group = 1" in tf
    assert "gpu_node_groups = 1" in tf
    assert 'gpu_nodes_platform = "gpu-rtx6000"' in tf
    assert 'gpu_nodes_preset = "1gpu-24vcpu-218gb"' in tf
    # Single-GPU preset must not enable GPU clustering.
    assert "enable_gpu_cluster = false" in tf
    assert "enable_filestore = true" in tf
    # existing_filestore must always be emitted as "" (the recipe defaults it to
    # null and branches on == "", so an unset value would read a phantom FS).
    assert 'existing_filestore = ""' in tf
    # loki has no recipe default and must be emitted, plus o11y stays off.
    assert "loki = { enabled = false" in tf
    assert "enable_grafana           = false" in tf
    assert 'ssh_public_key = { key = "ssh-ed25519 AAAA me" }' in tf


def test_render_tfvars_8gpu_cluster_emits_fabric() -> None:
    cluster = ClusterSpec(
        name="train",
        gpu_nodes=NodePoolSpec(count=2, platform="gpu-h200-sxm", preset="8gpu-128vcpu-1600gb"),
        enable_gpu_cluster=True,
        infiniband_fabric="us-central1-a",
    )
    tf = render_tfvars(cluster)
    assert "enable_gpu_cluster = true" in tf
    assert 'infiniband_fabric = "us-central1-a"' in tf
    assert "gpu_nodes_fixed_count_per_group = 2" in tf


def test_patch_provider_domain_region_aware() -> None:
    provider_tf = 'provider "nebius" {\n  domain = "api.eu.nebius.cloud:443"\n}\n'
    us = patch_provider_domain(provider_tf, "us-central1")
    assert "api.nebius.cloud:443" in us
    assert "api.eu.nebius.cloud:443" not in us
    eu = patch_provider_domain(provider_tf, "eu-north1")
    assert "api.eu.nebius.cloud:443" in eu


def test_provider_domain_eu_vs_global() -> None:
    assert provider_domain("eu-north1") == "api.eu.nebius.cloud:443"
    assert provider_domain("us-central1") == "api.nebius.cloud:443"


# --------------------------------------------------------------------------- #
# plan (no infra)
# --------------------------------------------------------------------------- #
def test_plan_json(tmp_path) -> None:
    path = tmp_path / "fleet.yaml"
    path.write_text(
        textwrap.dedent(
            """
            apiVersion: npa.fleet/v0.0.1
            name: fleet1-test
            tenant_id: tenant-x
            region: us-central1
            project_prefix: "fleet1-test-"
            defaults:
              cpu_nodes: {count: 1, platform: cpu-d3, preset: 48vcpu-192gb}
              gpu_nodes: {count: 1, platform: gpu-rtx6000, preset: 1gpu-24vcpu-218gb}
            projects:
              - name: a
              - name: b
            """
        )
    )
    result = runner.invoke(app, ["fleet", "plan", "--spec", str(path), "--output", "json"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["project_count"] == 2
    assert plan["cluster_count"] == 2
    assert plan["tenant_id"] == "tenant-x"
    names = sorted(p["display_name"] for p in plan["projects"])
    assert names == ["fleet1-test-a", "fleet1-test-b"]
    assert all(p["will_create"] for p in plan["projects"])


def test_load_spec_from_yaml(tmp_path) -> None:
    path = tmp_path / "fleet.yaml"
    path.write_text(
        textwrap.dedent(
            """
            apiVersion: npa.fleet/v0.0.1
            name: fromyaml
            region: us-central1
            defaults:
              gpu_nodes: {count: 1, platform: gpu-rtx6000, preset: 1gpu-24vcpu-218gb}
            projects:
              - name: only
            """
        )
    )
    spec = load_spec(path)
    assert spec.name == "fromyaml"
    assert spec.projects[0].clusters[0].gpu_count() == 1


# --------------------------------------------------------------------------- #
# lifecycle helpers (mocked, no infra)
# --------------------------------------------------------------------------- #
def test_resolve_project_id_existing_by_id() -> None:
    from npa.fleet import lifecycle

    project = ProjectSpec(project_id="project-abc", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])
    pid, created = lifecycle.resolve_project_id(
        "nebius", "tenant-x", project, prefix="fleet1-test-", create=True, env={}
    )
    assert pid == "project-abc"
    assert created is False


def test_resolve_project_id_creates_when_absent(monkeypatch) -> None:
    from npa.fleet import lifecycle

    monkeypatch.setattr(lifecycle, "_list_projects", lambda *a, **k: [])
    created_names: list[str] = []

    def fake_create(nebius_bin, tenant_id, name, env, *, region="", profile=""):
        created_names.append((name, region))
        return "project-new"

    monkeypatch.setattr(lifecycle, "_create_project", fake_create)
    project = ProjectSpec(name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])
    pid, created = lifecycle.resolve_project_id(
        "nebius", "tenant-x", project, prefix="fleet1-test-", create=True, env={}, region="us-central1"
    )
    assert pid == "project-new"
    assert created is True
    assert created_names == [("fleet1-test-a", "us-central1")]


def test_resolve_project_id_reuses_existing_by_name(monkeypatch) -> None:
    from npa.fleet import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_list_projects",
        lambda *a, **k: [{"metadata": {"name": "fleet1-test-a", "id": "project-found"}}],
    )
    project = ProjectSpec(name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])
    pid, created = lifecycle.resolve_project_id(
        "nebius", "tenant-x", project, prefix="fleet1-test-", create=False, env={}
    )
    assert pid == "project-found"
    assert created is False


def test_resolve_project_id_errors_when_absent_and_no_create(monkeypatch) -> None:
    from npa.fleet import lifecycle

    monkeypatch.setattr(lifecycle, "_list_projects", lambda *a, **k: [])
    project = ProjectSpec(name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])
    with pytest.raises(ValueError, match="creation is disabled"):
        lifecycle.resolve_project_id(
            "nebius", "tenant-x", project, prefix="fleet1-test-", create=False, env={}
        )


def test_infra_fleet_deploy_toolref_registered() -> None:
    from npa.orchestration.npa_workflow.catalog import argv_for_tool

    argv = argv_for_tool("infra.fleet.deploy")
    assert argv[:3] == ["npa", "fleet", "deploy"]
    assert "{{config.fleet_spec}}" in argv


def test_deploy_help_documents_yes_and_only_clusters() -> None:
    result = runner.invoke(app, ["fleet", "deploy", "--help"])
    assert result.exit_code == 0
    assert "--yes" in result.output
    assert "--only-clusters" in result.output


def test_destroy_help_documents_yes_force_and_only_clusters() -> None:
    result = runner.invoke(app, ["fleet", "destroy", "--help"])
    assert result.exit_code == 0
    assert "--yes" in result.output
    assert "--force" in result.output
    assert "--only-clusters" in result.output


# --------------------------------------------------------------------------- #
# Confirmation gates (creation + removal)
# --------------------------------------------------------------------------- #
def _spec_file(tmp_path):
    path = tmp_path / "fleet.yaml"
    path.write_text(
        textwrap.dedent(
            """
            apiVersion: npa.fleet/v0.0.1
            name: fleet1-test
            tenant_id: tenant-x
            region: us-central1
            project_prefix: "fleet1-test-"
            defaults:
              gpu_nodes: {count: 1, platform: gpu-rtx6000, preset: 1gpu-24vcpu-218gb}
            projects:
              - name: a
              - name: b
            """
        )
    )
    return path


def test_deploy_aborts_on_declined_confirmation(tmp_path, monkeypatch) -> None:
    import npa.cli.fleet as fleetcli

    called = {"deploy": False}
    monkeypatch.setattr(
        fleetcli, "_load", fleetcli._load
    )  # keep real loader
    import npa.fleet.lifecycle as L

    def _boom(*a, **k):
        called["deploy"] = True
        return {}

    monkeypatch.setattr(L, "deploy_fleet", _boom)
    # Answer "n" to the confirmation prompt.
    result = runner.invoke(app, ["fleet", "deploy", "--spec", str(_spec_file(tmp_path))], input="n\n")
    assert result.exit_code == 1
    assert "Aborted." in result.output
    assert called["deploy"] is False  # deploy must not run when declined


def test_deploy_yes_flag_skips_prompt_and_runs(tmp_path, monkeypatch) -> None:
    import npa.fleet.lifecycle as L

    captured = {}

    def _fake_deploy(spec, **kwargs):
        captured.update(kwargs)
        return {"name": spec.name, "region": "us-central1", "tenant_id": "t", "deployed": 2, "failed": 0, "clusters": []}

    monkeypatch.setattr(L, "deploy_fleet", _fake_deploy)
    result = runner.invoke(app, ["fleet", "deploy", "--spec", str(_spec_file(tmp_path)), "--yes"])
    assert result.exit_code == 0, result.output
    assert "2 deployed" in result.output


def test_destroy_aborts_on_declined_confirmation(tmp_path, monkeypatch) -> None:
    import npa.fleet.lifecycle as L

    called = {"destroy": False}

    def _boom(*a, **k):
        called["destroy"] = True
        return {}

    monkeypatch.setattr(L, "destroy_fleet", _boom)
    result = runner.invoke(app, ["fleet", "destroy", "--spec", str(_spec_file(tmp_path))], input="n\n")
    assert result.exit_code == 1
    assert "Aborted." in result.output
    assert "torn down" in result.output  # teardown/reclaim warning is shown
    assert called["destroy"] is False


def test_destroy_force_and_yes_skip_prompt(tmp_path, monkeypatch) -> None:
    import npa.fleet.lifecycle as L

    monkeypatch.setattr(L, "destroy_fleet", lambda spec, **k: {"name": spec.name, "clusters": []})
    for flag in ("--yes", "--force"):
        result = runner.invoke(app, ["fleet", "destroy", "--spec", str(_spec_file(tmp_path)), flag])
        assert result.exit_code == 0, result.output
        assert "Destroyed fleet" in result.output


def test_confirmation_lists_only_targeted_clusters(tmp_path, monkeypatch) -> None:
    import npa.fleet.lifecycle as L

    seen = {}

    def _fake_deploy(spec, **kwargs):
        seen.update(kwargs)
        return {"name": spec.name, "region": "r", "tenant_id": "t", "deployed": 0, "failed": 0, "clusters": []}

    monkeypatch.setattr(L, "deploy_fleet", _fake_deploy)
    result = runner.invoke(
        app,
        ["fleet", "deploy", "--spec", str(_spec_file(tmp_path)), "--only-projects", "a", "--yes"],
    )
    assert result.exit_code == 0, result.output
    # Only project a's cluster is listed/targeted, not b's.
    assert "fleet1-test-a / cluster" in result.output
    assert "fleet1-test-b" not in result.output
    assert seen["only_projects"] == ["a"]


def test_dns_name_rejects_over_63_chars() -> None:
    long_name = "a" * 64
    cluster = ClusterSpec(name=long_name, cpu_nodes=NodePoolSpec(count=1))
    with pytest.raises(FleetSpecError, match="DNS-1123 label"):
        cluster.validate()


# --------------------------------------------------------------------------- #
# ensure_subnet (mocked, no infra)
# --------------------------------------------------------------------------- #
class _Cap:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def test_ensure_subnet_reuses_existing_and_reports_no_created_network(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_list_subnets", lambda *a, **k: [{"metadata": {"id": "subnet-x"}}])
    assert L.ensure_subnet("neb", "proj", name_stem="c", env={}) == ("subnet-x", "")


def test_ensure_subnet_creates_network_and_subnet(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_list_subnets", lambda *a, **k: [])

    def cap(cmd, *, env=None, check=True, timeout=None, cwd=None):
        if "network" in cmd and "create" in cmd:
            return _Cap(json.dumps({"metadata": {"id": "net-9"}}))
        if "subnet" in cmd and "create" in cmd:
            return _Cap(json.dumps({"metadata": {"id": "sub-9"}}))
        return _Cap("{}")

    monkeypatch.setattr(L, "_run_capture", cap)
    assert L.ensure_subnet("neb", "proj", name_stem="c", env={}) == ("sub-9", "net-9")


# --------------------------------------------------------------------------- #
# _prepare_install_dir provider-domain patch (loud no-op)
# --------------------------------------------------------------------------- #
def _fake_recipe(tmp_path, provider_body: str):
    root = tmp_path / "recipe"
    (root / "k8s-training").mkdir(parents=True)
    (root / "modules").mkdir()
    (root / "k8s-training" / "provider.tf").write_text(provider_body)
    (root / "k8s-training" / "variables.tf").write_text("")
    return root


def test_prepare_install_dir_patches_eu_domain(tmp_path) -> None:
    from npa.fleet import lifecycle as L

    root = _fake_recipe(tmp_path, 'provider "nebius" { domain = "api.eu.nebius.cloud:443" }\n')
    msgs: list[str] = []
    wd = L._prepare_install_dir(
        tmp_path / "inst",
        recipe_root=root,
        region="us-central1",
        cluster=ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1)),
        ssh_public_key="k",
        on_status=msgs.append,
    )
    txt = (wd / "provider.tf").read_text()
    assert "api.nebius.cloud:443" in txt and "api.eu.nebius.cloud:443" not in txt
    assert not any("not patched" in m for m in msgs)


def test_prepare_install_dir_warns_when_provider_domain_not_matched(tmp_path) -> None:
    from npa.fleet import lifecycle as L

    # Recipe drift: the EU domain string is absent, so the literal replace is a
    # no-op. For a non-EU region this must warn loudly instead of silently using
    # the wrong endpoint.
    root = _fake_recipe(tmp_path, 'provider "nebius" { domain = "renamed-domain:443" }\n')
    msgs: list[str] = []
    L._prepare_install_dir(
        tmp_path / "inst",
        recipe_root=root,
        region="us-central1",
        cluster=ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1)),
        ssh_public_key="k",
        on_status=msgs.append,
    )
    assert any("not patched" in m for m in msgs)


# --------------------------------------------------------------------------- #
# _deploy_one_cluster sidecar status transitions (mocked)
# --------------------------------------------------------------------------- #
def _mock_deploy_boundary(monkeypatch, *, apply_fails: bool = False):
    from npa.fleet import lifecycle as L

    def fake_prepare(install_dir, **k):
        install_dir.mkdir(parents=True, exist_ok=True)
        return install_dir / "k8s-training"

    monkeypatch.setattr(L, "ensure_subnet", lambda *a, **k: ("subnet-1", "net-1"))
    monkeypatch.setattr(L, "_prepare_install_dir", fake_prepare)
    monkeypatch.setattr(L, "_cluster_tf_env", lambda *a, **k: {})

    def fake_stream(*a, **k):
        if apply_fails:
            raise RuntimeError("terraform boom")
        return None

    monkeypatch.setattr(L, "_run_stream", fake_stream)
    monkeypatch.setattr(
        L, "_terraform_outputs", lambda *a, **k: {"kube_cluster": {"value": {"id": "mk8s-1"}}}
    )
    monkeypatch.setattr(L, "_write_kubeconfig", lambda *a, **k: None)
    return L


def _run_one_cluster(L, tmp_path, *, profile: str = ""):
    spec = FleetSpec(name="f")
    project = ProjectSpec(name="a")
    cluster = ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))
    return L._deploy_one_cluster(
        spec=spec,
        project=project,
        cluster=cluster,
        project_id="p1",
        project_created=False,
        region="us-central1",
        tenant_id="t",
        ssh_public_key="k",
        fleet_root=tmp_path,
        recipe_root=tmp_path,
        terraform_bin="terraform",
        nebius_bin="nebius",
        profile=profile,
        timeout_minutes=1,
        on_status=None,
    )


def test_deploy_one_cluster_success_promotes_sidecar_and_records_network(tmp_path, monkeypatch) -> None:
    L = _mock_deploy_boundary(monkeypatch)
    res = _run_one_cluster(L, tmp_path)
    assert res["status"] == "deployed"
    assert res["cluster_id"] == "mk8s-1"
    sidecar = json.loads((tmp_path / "a" / "c" / L._ENV_SIDECAR).read_text())
    assert sidecar["status"] == "deployed"
    assert sidecar["created_network_id"] == "net-1"
    assert sidecar["cluster_id"] == "mk8s-1"


def test_deploy_one_cluster_failure_leaves_sidecar_provisioning(tmp_path, monkeypatch) -> None:
    L = _mock_deploy_boundary(monkeypatch, apply_fails=True)
    res = _run_one_cluster(L, tmp_path)
    assert res["status"] == "error"
    # Sidecar was written before apply; a failed apply must NOT read as deployed.
    sidecar = json.loads((tmp_path / "a" / "c" / L._ENV_SIDECAR).read_text())
    assert sidecar["status"] == "provisioning"


# --------------------------------------------------------------------------- #
# destroy_fleet reclaims a fleet-created network (mocked)
# --------------------------------------------------------------------------- #
def _setup_destroy(tmp_path, monkeypatch, sidecar_extra: dict):
    from npa.fleet import lifecycle as L

    install = tmp_path / "f" / "a" / "c"
    (install / L._K8S_TRAINING_SUBDIR).mkdir(parents=True)
    L._write_env_sidecar(
        install,
        {
            "tenant_id": "t",
            "project_id": "p1",
            "region": "us-central1",
            "subnet_id": "sub-9",
            "cluster_name": "c",
            "status": "deployed",
            **sidecar_extra,
        },
    )
    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_terraform_env", lambda b, **k: {})
    monkeypatch.setattr(L, "_run_stream", lambda *a, **k: None)
    deleted: list[str] = []

    def cap(cmd, *, env=None, check=True, timeout=None, cwd=None):
        if "delete" in cmd and ("subnet" in cmd or "network" in cmd):
            deleted.append(cmd[cmd.index("--id") + 1])
        return _Cap("", 0)

    monkeypatch.setattr(L, "_run_capture", cap)
    return L, deleted


def test_destroy_reclaims_created_network_and_subnet(tmp_path, monkeypatch) -> None:
    L, deleted = _setup_destroy(tmp_path, monkeypatch, {"created_network_id": "net-9"})
    spec = FleetSpec(
        name="f",
        projects=[ProjectSpec(name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])],
    )
    L.destroy_fleet(spec, work_root=tmp_path)
    # Subnet first, then network.
    assert deleted == ["sub-9", "net-9"]


def test_destroy_leaves_reused_subnet_untouched(tmp_path, monkeypatch) -> None:
    # No created_network_id -> the subnet was pre-existing; must not be deleted.
    L, deleted = _setup_destroy(tmp_path, monkeypatch, {"created_network_id": ""})
    spec = FleetSpec(
        name="f",
        projects=[ProjectSpec(name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])],
    )
    L.destroy_fleet(spec, work_root=tmp_path)
    assert deleted == []


# --------------------------------------------------------------------------- #
# add/remove one or multiple clusters (targeting + state reconciliation)
# --------------------------------------------------------------------------- #
def _two_cluster_project_spec() -> FleetSpec:
    return FleetSpec(
        name="f",
        projects=[
            ProjectSpec(
                name="a",
                clusters=[
                    ClusterSpec(name="c1", cpu_nodes=NodePoolSpec(count=1)),
                    ClusterSpec(name="c2", cpu_nodes=NodePoolSpec(count=1)),
                ],
            )
        ],
    )


def test_deploy_only_clusters_targets_a_single_cluster(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_resolve_tenant_id", lambda *a, **k: "tenant-x")
    monkeypatch.setattr(L, "_resolve_region", lambda *a, **k: "us-central1")
    monkeypatch.setattr(L, "_resolve_ssh_public_key", lambda *a, **k: "ssh-key")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *a, **k: tmp_path / "recipe")
    monkeypatch.setattr(L, "_nebius_cli_env", lambda: {})
    monkeypatch.setattr(L, "resolve_project_id", lambda *a, **k: ("proj-1", False))
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 1))
    built: list[str] = []

    def fake_one(**kwargs):
        name = kwargs["cluster"].name
        built.append(name)
        return {"project_key": "a", "cluster_name": name, "status": "deployed", "cluster_id": f"id-{name}"}

    monkeypatch.setattr(L, "_deploy_one_cluster", fake_one)
    res = L.deploy_fleet(_two_cluster_project_spec(), work_root=tmp_path, only_clusters=["c2"])
    assert built == ["c2"]  # only the targeted cluster is (re)deployed
    assert res["deployed"] == 1


def test_deploy_help_documents_concurrency() -> None:
    result = runner.invoke(app, ["fleet", "deploy", "--help"])
    assert result.exit_code == 0
    assert "--concurrency" in result.output


def _mock_deploy_fleet_boundary(monkeypatch, tmp_path):
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_resolve_tenant_id", lambda *a, **k: "t")
    monkeypatch.setattr(L, "_resolve_region", lambda *a, **k: "us-central1")
    monkeypatch.setattr(L, "_resolve_ssh_public_key", lambda *a, **k: "k")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *a, **k: tmp_path / "recipe")
    monkeypatch.setattr(L, "_nebius_cli_env", lambda: {})
    monkeypatch.setattr(L, "resolve_project_id", lambda *a, **k: ("proj-1", False))
    # No quota API in unit tests: an unreadable allowance list skips the preflight.
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 1))
    return L


def test_deploy_fleet_parallel_runs_all_targets_and_prewarms_once(tmp_path, monkeypatch) -> None:
    import threading

    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    prewarm = {"n": 0}
    monkeypatch.setattr(L, "_prewarm_plugin_cache", lambda *a, **k: prewarm.__setitem__("n", prewarm["n"] + 1))
    ran: list[str] = []
    log_paths: list = []
    lock = threading.Lock()

    def fake_one(**kwargs):
        with lock:
            ran.append(kwargs["cluster"].name)
            log_paths.append(kwargs.get("log_path"))
        return {
            "project_key": kwargs["project"].key(),
            "cluster_name": kwargs["cluster"].name,
            "status": "deployed",
            "cluster_id": "id",
        }

    monkeypatch.setattr(L, "_deploy_one_cluster", fake_one)
    res = L.deploy_fleet(_two_cluster_project_spec(), work_root=tmp_path, concurrency=2)
    assert sorted(ran) == ["c1", "c2"]  # both targets applied
    assert res["deployed"] == 2
    assert prewarm["n"] == 1  # plugin cache pre-warmed exactly once for parallel
    assert all(lp is not None for lp in log_paths)  # parallel -> per-cluster log files


def test_deploy_fleet_sequential_skips_prewarm_and_streams(tmp_path, monkeypatch) -> None:
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    prewarm = {"n": 0}
    monkeypatch.setattr(L, "_prewarm_plugin_cache", lambda *a, **k: prewarm.__setitem__("n", prewarm["n"] + 1))
    log_paths: list = []

    def fake_one(**kwargs):
        log_paths.append(kwargs.get("log_path"))
        return {"project_key": "a", "cluster_name": kwargs["cluster"].name, "status": "deployed"}

    monkeypatch.setattr(L, "_deploy_one_cluster", fake_one)
    L.deploy_fleet(_two_cluster_project_spec(), work_root=tmp_path, concurrency=1)
    assert prewarm["n"] == 0  # no pre-warm when sequential
    assert log_paths == [None, None]  # sequential -> stream to stdout, no log file


def test_destroy_fleet_parallel_runs_all_and_prunes(tmp_path, monkeypatch) -> None:
    import threading

    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    ran: list[str] = []
    lock = threading.Lock()

    def fake_destroy_one(**kwargs):
        with lock:
            ran.append(kwargs["cluster"].name)
        return {"project_key": "a", "cluster_name": kwargs["cluster"].name, "status": "destroyed"}

    pruned = {}
    monkeypatch.setattr(L, "_destroy_one_cluster", fake_destroy_one)
    monkeypatch.setattr(L, "_prune_fleet_state", lambda fr, keys: pruned.update({"keys": keys}))
    L.destroy_fleet(_two_cluster_project_spec(), work_root=tmp_path, concurrency=2)
    assert sorted(ran) == ["c1", "c2"]
    assert pruned["keys"] == {("a", "c1"), ("a", "c2")}


def test_run_to_log_writes_and_raises(tmp_path) -> None:
    from npa.fleet import lifecycle as L

    log = tmp_path / "deploy.log"
    L._run_to_log(["true"], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, timeout=30, log_path=log)
    assert log.exists() and "$ true" in log.read_text()
    with pytest.raises(RuntimeError, match="command failed"):
        L._run_to_log(["false"], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, timeout=30, log_path=log)


def test_upsert_and_prune_fleet_state_roundtrip(tmp_path) -> None:
    from npa.fleet import lifecycle as L

    fleet_root = tmp_path / "f"
    fleet_root.mkdir()
    base = {"name": "f", "tenant_id": "t", "region": "r", "project_prefix": "", "k8s_training_source": "x"}
    # Deploy c1, then add c2 -- both must be present (upsert must not clobber c1).
    L._upsert_fleet_state(fleet_root, base, [{"project_key": "a", "cluster_name": "c1", "status": "deployed"}])
    L._upsert_fleet_state(fleet_root, base, [{"project_key": "a", "cluster_name": "c2", "status": "deployed"}])
    state = L._load_fleet_state(fleet_root)
    assert sorted(c["cluster_name"] for c in state["clusters"]) == ["c1", "c2"]
    assert state["deployed"] == 2
    # Remove c2 -- only c1 remains.
    L._prune_fleet_state(fleet_root, {("a", "c2")})
    state = L._load_fleet_state(fleet_root)
    assert [c["cluster_name"] for c in state["clusters"]] == ["c1"]
    assert state["deployed"] == 1


# --------------------------------------------------------------------------- #
# --profile / spec profile: multi-tenant principal selection
# --------------------------------------------------------------------------- #
def test_spec_parses_profile() -> None:
    spec = spec_from_mapping({**_base_mapping(), "profile": "sd"})
    assert spec.profile == "sd"


def test_spec_profile_defaults_to_active() -> None:
    assert spec_from_mapping(_base_mapping()).profile == ""


def test_nebius_argv_adds_profile_only_when_set() -> None:
    from npa.fleet import lifecycle as L

    assert L._nebius_argv("nebius") == ["nebius"]
    assert L._nebius_argv("nebius", "sd") == ["nebius", "--profile", "sd"]


def test_resolve_tenant_id_uses_named_profile(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(
        L,
        "_nebius_config",
        lambda: {
            "default": "other",
            "profiles": {"other": {"tenant-id": "tenant-other"}, "sd": {"tenant-id": "tenant-sd"}},
        },
    )
    assert L._resolve_tenant_id("nebius", "", "sd") == "tenant-sd"
    assert L._resolve_tenant_id("nebius", "") == "tenant-other"


def test_resolve_tenant_id_rejects_profile_without_tenant(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    # Falling back to the active profile's tenant here would deploy the fleet
    # into the wrong tenant, so this must fail loudly instead.
    monkeypatch.setattr(
        L,
        "_nebius_config",
        lambda: {"default": "other", "profiles": {"other": {"tenant-id": "tenant-other"}, "sd": {}}},
    )
    with pytest.raises(ValueError, match="has no 'tenant-id'"):
        L._resolve_tenant_id("nebius", "", "sd")


def test_list_projects_passes_profile_to_cli(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    seen: list[list[str]] = []
    monkeypatch.setattr(
        L, "_run_capture", lambda cmd, **k: (seen.append(cmd), _Cap('{"items": []}', 0))[1]
    )
    L._list_projects("nebius", "tenant-x", {}, "sd")
    assert seen[0][:3] == ["nebius", "--profile", "sd"]


def test_deploy_fleet_cli_profile_overrides_spec(tmp_path, monkeypatch) -> None:
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    seen: list[str] = []

    def fake_one(**kwargs):
        seen.append(kwargs["profile"])
        return {"project_key": "a", "cluster_name": kwargs["cluster"].name, "status": "deployed"}

    monkeypatch.setattr(L, "_deploy_one_cluster", fake_one)
    spec = FleetSpec(
        name="f",
        profile="from-spec",
        projects=[ProjectSpec(name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])],
    )
    res = L.deploy_fleet(spec, work_root=tmp_path)
    assert seen == ["from-spec"]
    assert res["profile"] == "from-spec"
    seen.clear()
    res = L.deploy_fleet(spec, work_root=tmp_path, profile="from-cli")
    assert seen == ["from-cli"]
    assert res["profile"] == "from-cli"


def test_deploy_one_cluster_records_profile_in_sidecar(tmp_path, monkeypatch) -> None:
    L = _mock_deploy_boundary(monkeypatch)
    _run_one_cluster(L, tmp_path, profile="sd")
    sidecar = json.loads((tmp_path / "a" / "c" / L._ENV_SIDECAR).read_text())
    assert sidecar["profile"] == "sd"


def test_destroy_falls_back_to_sidecar_profile(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    install = tmp_path / "f" / "a" / "c"
    (install / L._K8S_TRAINING_SUBDIR).mkdir(parents=True)
    L._write_env_sidecar(
        install,
        {"tenant_id": "t", "project_id": "p1", "region": "us-central1", "subnet_id": "s",
         "cluster_name": "c", "profile": "sd", "status": "deployed"},
    )
    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_run_stream", lambda *a, **k: None)
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 0))
    seen: list[str] = []
    monkeypatch.setattr(
        L, "_terraform_env", lambda b, **k: (seen.append(k.get("profile", "")), {})[1]
    )
    spec = FleetSpec(
        name="f",
        projects=[ProjectSpec(name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))])],
    )
    L.destroy_fleet(spec, work_root=tmp_path)
    assert seen == ["sd"]  # teardown authenticates as the deploying principal


def test_deploy_and_destroy_help_document_profile() -> None:
    for cmd in ("deploy", "destroy", "plan"):
        result = runner.invoke(app, ["fleet", cmd, "--help"])
        assert result.exit_code == 0
        assert "--profile" in result.output


def test_deploy_json_output_keeps_stdout_pure_json(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    # Regression: progress + confirmation banner went to stdout even in json
    # mode, so `npa fleet deploy --output json` (what the toolRef runs) emitted a
    # stream that would not parse.
    spec_file = tmp_path / "fleet.yaml"
    spec_file.write_text(
        textwrap.dedent(
            """
            apiVersion: npa.fleet/v0.0.1
            name: f
            region: us-central1
            projects:
              - name: a
                clusters:
                  - name: c
                    cpu_nodes: { count: 1, platform: cpu-d3, preset: 4vcpu-16gb }
            """
        )
    )
    monkeypatch.setattr(
        L,
        "deploy_fleet",
        lambda *a, **k: (
            k["on_status"]("a status line that must not land on stdout"),
            {"name": "f", "region": "us-central1", "tenant_id": "t", "deployed": 1,
             "failed": 0, "clusters": []},
        )[1],
    )
    result = runner.invoke(
        app, ["fleet", "deploy", "--spec", str(spec_file), "--yes", "--output", "json"]
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["deployed"] == 1  # stdout parses as JSON


def test_deploy_text_output_keeps_progress_on_stdout(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    spec_file = tmp_path / "fleet.yaml"
    spec_file.write_text(
        textwrap.dedent(
            """
            apiVersion: npa.fleet/v0.0.1
            name: f
            region: us-central1
            projects:
              - name: a
                clusters:
                  - name: c
                    cpu_nodes: { count: 1, platform: cpu-d3, preset: 4vcpu-16gb }
            """
        )
    )
    monkeypatch.setattr(
        L,
        "deploy_fleet",
        lambda *a, **k: (
            k["on_status"]("visible progress"),
            {"name": "f", "region": "us-central1", "tenant_id": "t", "deployed": 1,
             "failed": 0, "clusters": []},
        )[1],
    )
    result = runner.invoke(app, ["fleet", "deploy", "--spec", str(spec_file), "--yes"])
    assert result.exit_code == 0
    assert "visible progress" in result.stdout
    assert "About to create/update" in result.stdout


def test_fleet_deploy_toolref_is_non_interactive() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    # A workflow state cannot answer the confirmation prompt.
    assert "--yes" in TOOL_CATALOG["infra.fleet.deploy"].argv_template


# --------------------------------------------------------------------------- #
# tenant quota preflight (mk8s accepts node groups it cannot fill)
# --------------------------------------------------------------------------- #
def _allowance(name: str, region: str, limit, unit: str = "count") -> dict:
    return {
        "metadata": {"name": name},
        "spec": {"region": region, "limit": limit},
        "status": {"unit": unit},
    }


def test_required_quotas_counts_nodes_vcpu_gpus_and_filesystem() -> None:
    from npa.fleet.quotas import required_quotas

    needed = required_quotas(
        [
            ClusterSpec(
                name="c",
                cpu_nodes=NodePoolSpec(count=2, platform="cpu-d3", preset="48vcpu-192gb"),
                gpu_nodes=NodePoolSpec(
                    count=2, platform="gpu-rtx6000", preset="8gpu-192vcpu-1744gb"
                ),
                enable_gpu_cluster=False,
                enable_filestore=True,
                filestore_disk_size_gibibytes=2048,
            )
        ]
    )
    assert needed["compute.instance.count"] == 4
    assert needed["compute.disk.count"] == 4
    assert needed["compute.instance.non-gpu.vcpu"] == 96  # GPU-node vCPUs excluded
    assert needed["compute.instance.gpu.rtx6000"] == 16
    assert needed["compute.filesystem.count"] == 1
    assert needed["compute.filesystem.size.network-ssd"] == 2048 * 1024**3
    assert needed["mk8s.cluster.count"] == 1
    assert "compute.gpucluster.count" not in needed


def test_required_quotas_counts_gpu_cluster_and_skips_existing_filestore() -> None:
    from npa.fleet.quotas import required_quotas

    needed = required_quotas(
        [
            ClusterSpec(
                name="c",
                gpu_nodes=NodePoolSpec(
                    count=2, platform="gpu-h200-sxm", preset="8gpu-128vcpu-1600gb"
                ),
                enable_gpu_cluster=True,
                infiniband_fabric="us-central1-a",
                enable_filestore=True,
                existing_filestore="computefilesystem-abc",
            )
        ]
    )
    assert needed["compute.instance.gpu.h200"] == 16
    assert needed["compute.gpucluster.count"] == 1
    # Reusing a filesystem consumes no new filesystem quota.
    assert "compute.filesystem.count" not in needed


def test_parse_allowances_filters_region_and_skips_unset_limits() -> None:
    from npa.fleet.quotas import parse_allowances

    payload = json.dumps(
        {
            "items": [
                _allowance("compute.instance.count", "us-central1", "10"),
                _allowance("compute.instance.count", "eu-north1", "99"),
                # Unset limit = "no limit at this container", not zero.
                _allowance("compute.instance.gpu.rtx6000", "us-central1", None),
            ]
        }
    )
    parsed = parse_allowances(payload, "us-central1")
    assert parsed["compute.instance.count"]["limit"] == 10
    assert "compute.instance.gpu.rtx6000" not in parsed


def test_find_shortfalls_flags_zero_limit_and_ignores_unadvertised() -> None:
    from npa.fleet.quotas import find_shortfalls, parse_allowances

    allowances = parse_allowances(
        json.dumps(
            {
                "items": [
                    _allowance("compute.instance.gpu.rtx6000", "us-central1", "0"),
                    _allowance("compute.instance.count", "us-central1", "10"),
                ]
            }
        ),
        "us-central1",
    )
    needed = {
        "compute.instance.gpu.rtx6000": 16,
        "compute.instance.count": 4,
        "mk8s.cluster.count": 1,  # not advertised for the region -> not asserted
    }
    shortfalls = find_shortfalls(needed, allowances, "us-central1")
    assert [s.name for s in shortfalls] == ["compute.instance.gpu.rtx6000"]
    assert "needs 16" in shortfalls[0].describe()


def test_gpu_family_maps_platform_to_quota_family() -> None:
    from npa.fleet.quotas import gpu_family

    assert gpu_family("gpu-rtx6000") == "rtx6000"
    assert gpu_family("gpu-h200-sxm") == "h200"
    assert gpu_family("cpu-d3") == ""


def _preflight_boundary(monkeypatch, tmp_path, allowances_json: str, *, rc: int = 0):
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap(allowances_json, rc))
    deployed: list[str] = []
    monkeypatch.setattr(
        L,
        "_deploy_one_cluster",
        lambda **kw: (
            deployed.append(kw["cluster"].name),
            {"project_key": "a", "cluster_name": kw["cluster"].name, "status": "deployed"},
        )[1],
    )
    return L, deployed


def _rtx_cluster_spec() -> FleetSpec:
    return FleetSpec(
        name="f",
        projects=[
            ProjectSpec(
                name="a",
                clusters=[
                    ClusterSpec(
                        name="c",
                        gpu_nodes=NodePoolSpec(
                            count=2, platform="gpu-rtx6000", preset="8gpu-192vcpu-1744gb"
                        ),
                        enable_gpu_cluster=False,
                    )
                ],
            )
        ],
    )


def test_deploy_preflight_blocks_on_zero_gpu_quota(tmp_path, monkeypatch) -> None:
    payload = json.dumps(
        {"items": [_allowance("compute.instance.gpu.rtx6000", "us-central1", "0")]}
    )
    L, deployed = _preflight_boundary(monkeypatch, tmp_path, payload)
    with pytest.raises(ValueError, match="compute.instance.gpu.rtx6000"):
        L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)
    assert deployed == []  # nothing applied


def test_deploy_no_preflight_skips_the_check(tmp_path, monkeypatch) -> None:
    payload = json.dumps(
        {"items": [_allowance("compute.instance.gpu.rtx6000", "us-central1", "0")]}
    )
    L, deployed = _preflight_boundary(monkeypatch, tmp_path, payload)
    L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path, preflight=False)
    assert deployed == ["c"]


def test_deploy_preflight_passes_when_quota_is_sufficient(tmp_path, monkeypatch) -> None:
    payload = json.dumps(
        {"items": [_allowance("compute.instance.gpu.rtx6000", "us-central1", "16")]}
    )
    L, deployed = _preflight_boundary(monkeypatch, tmp_path, payload)
    L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)
    assert deployed == ["c"]


def test_deploy_preflight_unreadable_quota_api_does_not_block(tmp_path, monkeypatch) -> None:
    # Losing the preflight (no quota read permission) must not block a deploy.
    L, deployed = _preflight_boundary(monkeypatch, tmp_path, "", rc=1)
    msgs: list[str] = []
    L.deploy_fleet(
        _rtx_cluster_spec(), work_root=tmp_path, on_status=lambda m: msgs.append(m)
    )
    assert deployed == ["c"]
    assert any("skipping quota preflight" in m for m in msgs)


def test_preflight_runs_before_any_project_is_created(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    # Regression: the preflight used to run *after* project resolution, so a
    # quota-blocked deploy left a freshly created, empty project behind.
    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_resolve_tenant_id", lambda *a, **k: "t")
    monkeypatch.setattr(L, "_resolve_region", lambda *a, **k: "us-central1")
    monkeypatch.setattr(L, "_resolve_ssh_public_key", lambda *a, **k: "k")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *a, **k: tmp_path / "recipe")
    monkeypatch.setattr(L, "_nebius_cli_env", lambda: {})
    resolved: list[str] = []
    monkeypatch.setattr(
        L,
        "resolve_project_id",
        lambda *a, **k: (resolved.append("resolved"), ("proj-1", True))[1],
    )
    monkeypatch.setattr(
        L,
        "_run_capture",
        lambda *a, **k: _Cap(
            json.dumps(
                {"items": [_allowance("compute.instance.gpu.rtx6000", "us-central1", "0")]}
            ),
            0,
        ),
    )
    monkeypatch.setattr(L, "_deploy_one_cluster", lambda **kw: {"status": "deployed"})
    with pytest.raises(ValueError, match="quota is too low"):
        L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)
    assert resolved == []  # no project touched


def test_plan_resolves_tenant_from_named_profile(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    # plan must show the tenant deploy would use, not "(resolve-at-deploy)".
    monkeypatch.setattr(
        L,
        "_nebius_config",
        lambda: {"default": "other", "profiles": {"sd": {"tenant-id": "tenant-sd"}}},
    )
    spec = spec_from_mapping({**_base_mapping(), "tenant_id": "", "profile": "sd"})
    plan = L.plan_fleet(spec)
    assert plan["tenant_id"] == "tenant-sd"
    assert plan["profile"] == "sd"
    # An explicit spec tenant_id still wins over the profile's.
    pinned = spec_from_mapping({**_base_mapping(), "tenant_id": "tenant-pinned", "profile": "sd"})
    assert L.plan_fleet(pinned)["tenant_id"] == "tenant-pinned"


def test_fleet_spec_profile_is_declared_after_projects() -> None:
    import dataclasses

    # Adding `profile` must not shift the positional order of pre-existing SDK
    # fields, so it has to come after `projects`.
    names = [f.name for f in dataclasses.fields(FleetSpec)]
    assert names.index("profile") > names.index("projects")
    positional = FleetSpec("f", "t", "r", "p", "ssh", [ProjectSpec(name="a")])
    assert positional.projects[0].name == "a"
    assert positional.profile == ""


def test_deploy_help_documents_preflight() -> None:
    result = runner.invoke(app, ["fleet", "deploy", "--help"])
    assert result.exit_code == 0
    assert "--no-preflight" in result.output


def test_destroy_only_clusters_removes_install_dir_and_prunes_state(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    # Two deployed clusters recorded in state + install dirs; destroy only c2.
    fleet_root = tmp_path / "f"
    for name in ("c1", "c2"):
        d = fleet_root / "a" / name
        (d / L._K8S_TRAINING_SUBDIR).mkdir(parents=True)
        L._write_env_sidecar(
            d,
            {"tenant_id": "t", "project_id": "p1", "region": "us-central1", "subnet_id": "s",
             "cluster_name": name, "status": "deployed"},
        )
    base = {"name": "f", "tenant_id": "t", "region": "r", "project_prefix": "", "k8s_training_source": "x"}
    L._upsert_fleet_state(
        fleet_root, base,
        [{"project_key": "a", "cluster_name": "c1", "status": "deployed"},
         {"project_key": "a", "cluster_name": "c2", "status": "deployed"}],
    )
    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_terraform_env", lambda b, **k: {})
    monkeypatch.setattr(L, "_run_stream", lambda *a, **k: None)
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 0))

    spec = _two_cluster_project_spec()
    L.destroy_fleet(spec, work_root=tmp_path, only_clusters=["c2"])

    assert not (fleet_root / "a" / "c2").exists()  # removed
    assert (fleet_root / "a" / "c1").exists()  # untouched
    state = L._load_fleet_state(fleet_root)
    assert [c["cluster_name"] for c in state["clusters"]] == ["c1"]
