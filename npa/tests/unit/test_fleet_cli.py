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

    def fake_create(nebius_bin, tenant_id, name, env, *, region=""):
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


def _run_one_cluster(L, tmp_path):
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
    monkeypatch.setattr(L, "_terraform_env", lambda b: {})
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
