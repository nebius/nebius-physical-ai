"""Unit tests for `npa fleet`: spec merge/validation, tfvars rendering, CLI wiring.

These tests must not touch real infrastructure: they exercise pure spec/tfvars
logic and the Typer command surface (help + validation + plan), mocking the
lifecycle at the call site for deploy/destroy paths.
"""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

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
        "gpu_nodes": {
            "count": 1,
            "platform": "gpu-rtx6000",
            "preset": "1gpu-24vcpu-218gb",
        },
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
                    "gpu_nodes": {
                        "count": 2,
                        "platform": "gpu-h200-sxm",
                        "preset": "8gpu-128vcpu-1600gb",
                    },
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
        gpu_nodes=NodePoolSpec(
            count=1, platform="gpu-rtx6000", preset="1gpu-24vcpu-218gb"
        ),
        enable_gpu_cluster=True,
    )
    with pytest.raises(FleetSpecError, match="8-GPU preset"):
        cluster.validate()


def test_capacity_block_group_parses_on_gpu_pool_and_rejects_cpu_pool() -> None:
    data = _base_mapping()
    data["defaults"]["gpu_nodes"]["capacity_block_group"] = " capacityblockgroup-test "
    spec = spec_from_mapping(data)
    assert (
        spec.projects[0].clusters[0].gpu_nodes.capacity_block_group
        == "capacityblockgroup-test"
    )

    cluster = ClusterSpec(
        name="c",
        cpu_nodes=NodePoolSpec(count=1, capacity_block_group="capacityblockgroup-test"),
    )
    with pytest.raises(FleetSpecError, match="only valid for gpu_nodes"):
        cluster.validate()


def test_filestore_mount_contract_parses_and_validates() -> None:
    data = _base_mapping()
    data["defaults"]["filestore_mount_path"] = "/mnt/data"
    data["defaults"]["filestore_mount_tag"] = "npa-shared-fs"
    cluster = spec_from_mapping(data).projects[0].clusters[0]
    cluster.validate()
    assert cluster.filestore_mount_path == "/mnt/data"
    assert cluster.filestore_mount_tag == "npa-shared-fs"

    cluster.filestore_mount_path = "mnt/data"
    with pytest.raises(FleetSpecError, match="must be absolute"):
        cluster.validate()

    cluster.filestore_mount_path = "/mnt/data"
    cluster.filestore_mount_tag = "bad tag"
    with pytest.raises(FleetSpecError, match="without whitespace or commas"):
        cluster.validate()


def test_cluster_needs_at_least_one_node() -> None:
    cluster = ClusterSpec(name="empty")
    with pytest.raises(FleetSpecError, match="at least one CPU or GPU node"):
        cluster.validate()


def test_project_needs_name_or_id() -> None:
    spec = FleetSpec(
        name="f",
        projects=[
            ProjectSpec(
                clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))]
            )
        ],
    )
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
    assert 'filestore_mount_path = "/mnt/data"' in tf
    assert 'filestore_mount_tag = "data"' in tf
    assert 'filesystem_csi = { chart_version = "0.1.6"' in tf
    # loki has no recipe default and must be emitted, plus o11y stays off.
    assert "loki = { enabled = false" in tf
    assert "enable_grafana           = false" in tf
    assert 'ssh_public_key = { key = "ssh-ed25519 AAAA me" }' in tf


def test_render_tfvars_8gpu_cluster_emits_fabric() -> None:
    cluster = ClusterSpec(
        name="train",
        gpu_nodes=NodePoolSpec(
            count=2, platform="gpu-h200-sxm", preset="8gpu-128vcpu-1600gb"
        ),
        enable_gpu_cluster=True,
        infiniband_fabric="us-central1-a",
    )
    tf = render_tfvars(cluster)
    assert "enable_gpu_cluster = true" in tf
    assert 'infiniband_fabric = "us-central1-a"' in tf
    assert "gpu_nodes_fixed_count_per_group = 2" in tf


def test_vendored_filestore_contract_matches_official_guide() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    recipe = repo_root / "deploy/cluster/vendor/nebius-solutions-library"
    cloud_init = (recipe / "modules/cloud-init/k8s-cloud-init.tftpl").read_text()
    main_tf = (recipe / "k8s-training/main.tf").read_text()
    egress_tf = (recipe / "modules/cilium-egress-gateway/main.tf").read_text()
    mount_validation = (
        recipe
        / "k8s-training/filesystem-csi-validation/01-verify-node-filesystem-mounts.sh"
    ).read_text()
    rwx_validation = (
        recipe
        / "k8s-training/filesystem-csi-validation/03-run-csi-rwx-cross-node-test.sh"
    ).read_text()
    rwx_manifest = (
        recipe
        / "k8s-training/filesystem-csi-validation/manifests/02-csi-rwx-cross-node.yaml"
    ).read_text()
    variables_tf = (recipe / "k8s-training/variables.tf").read_text()

    assert (
        "[ ${filestore_mount_tag}, ${filestore_mount_path}, virtiofs, "
        '"defaults,nofail", 0, 2 ]' in cloud_init
    )
    assert main_tf.count('attach_mode = "READ_WRITE"') == 2
    assert main_tf.count("mount_tag   = local.filestore.mount_tag") == 2
    assert main_tf.count("filestore_mount_tag  = local.filestore.mount_tag") == 2
    assert 'filestore_mount_tag  = "data"' in egress_tf
    assert 'grep -Fq \\' in mount_validation
    # kubectl 1.36 suppresses attached output and the created pod name under
    # --quiet, defeating both evidence checking and debugger-pod cleanup.
    assert "--quiet" not in mount_validation
    assert "completed without the required shared-filesystem success evidence" in (
        mount_validation
    )
    assert "nodeName:" not in rwx_manifest
    assert rwx_manifest.count("kubernetes.io/hostname:") == 2
    assert rwx_validation.count("awk '{print \\$1}'") == 4
    assert "awk '{print \\\\$1}'" not in rwx_validation
    assert (
        'chart_version                       = optional(string, "0.1.6")'
        in variables_tf
    )


def test_render_tfvars_capacity_block_is_strict() -> None:
    cluster = ClusterSpec(
        name="reserved",
        gpu_nodes=NodePoolSpec(
            count=1,
            platform="gpu-rtx6000",
            preset="8gpu-192vcpu-1744gb",
            capacity_block_group="capacityblockgroup-test",
        ),
        enable_gpu_cluster=False,
    )
    tf = render_tfvars(cluster)
    assert (
        'gpu_nodes_reservation_policy = { policy = "STRICT", '
        'reservation_ids = ["capacityblockgroup-test"] }' in tf
    )


def test_render_tfvars_b200_uses_managed_driver_image() -> None:
    b200 = render_tfvars(
        ClusterSpec(
            name="b200",
            gpu_nodes=NodePoolSpec(
                count=2, platform="gpu-b200-sxm", preset="8gpu-160vcpu-1792gb"
            ),
            enable_gpu_cluster=True,
            infiniband_fabric="us-central1-b",
        )
    )
    assert "gpu_nodes_driverfull_image   = true" in b200

    b200_alias = render_tfvars(
        ClusterSpec(
            name="b200-alias",
            gpu_nodes=NodePoolSpec(
                count=1, platform="gpu-b200-sxm-a", preset="8gpu-160vcpu-1792gb"
            ),
            enable_gpu_cluster=True,
            infiniband_fabric="ramon",
        )
    )
    assert "gpu_nodes_driverfull_image   = true" in b200_alias

    h200 = render_tfvars(
        ClusterSpec(
            name="h200",
            gpu_nodes=NodePoolSpec(
                count=1, platform="gpu-h200-sxm", preset="8gpu-128vcpu-1600gb"
            ),
            enable_gpu_cluster=True,
            infiniband_fabric="us-central1-a",
        )
    )
    assert "gpu_nodes_driverfull_image   = false" in h200


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
    result = runner.invoke(
        app, ["fleet", "plan", "--spec", str(path), "--output", "json"]
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["project_count"] == 2
    assert plan["cluster_count"] == 2
    assert plan["tenant_id"] == "tenant-x"
    names = sorted(p["display_name"] for p in plan["projects"])
    assert names == ["fleet1-test-a", "fleet1-test-b"]
    assert all(p["will_create"] for p in plan["projects"])


def test_plan_reports_strict_reservation_without_echoing_capacity_block_id() -> None:
    from npa.fleet.lifecycle import plan_fleet

    data = _base_mapping()
    data["defaults"]["gpu_nodes"]["capacity_block_group"] = (
        "capacityblockgroup-runtime-only"
    )
    plan = plan_fleet(spec_from_mapping(data))
    cluster_plan = plan["projects"][0]["clusters"][0]
    assert cluster_plan["gpu_reservation"] == "strict"
    assert cluster_plan["enable_filestore"] is True
    assert cluster_plan["filestore_disk_size_gibibytes"] == 1024
    assert cluster_plan["filestore_mount_path"] == "/mnt/data"
    assert cluster_plan["filestore_mount_tag"] == "data"
    assert "capacityblockgroup-runtime-only" not in json.dumps(plan)


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

    project = ProjectSpec(
        project_id="project-abc",
        clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))],
    )
    pid, created = lifecycle.resolve_project_id(
        "nebius", "tenant-x", project, prefix="fleet1-test-", create=True, env={}
    )
    assert pid == "project-abc"
    assert created is False


def test_resolve_project_id_creates_when_absent(monkeypatch, tmp_path: Path) -> None:
    from npa.fleet import lifecycle
    from npa.provisioning_journal import list_operations

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(lifecycle, "_list_projects", lambda *a, **k: [])
    created_names: list[str] = []

    def fake_create(nebius_bin, tenant_id, name, env, *, region="", profile=""):
        created_names.append((name, region))
        return "project-new"

    monkeypatch.setattr(lifecycle, "_create_project", fake_create)
    project = ProjectSpec(
        name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))]
    )
    pid, created = lifecycle.resolve_project_id(
        "nebius",
        "tenant-x",
        project,
        prefix="fleet1-test-",
        create=True,
        env={},
        region="us-central1",
    )
    assert pid == "project-new"
    assert created is True
    assert created_names == [("fleet1-test-a", "us-central1")]
    [ownership] = list_operations(project_id="project-new", resource_type="project")
    resource = ownership.read()["resources"][0]
    assert resource["ownership_source"] == "provider-create-response"
    assert resource["labels"] == {"tenant_id": "tenant-x", "region": "us-central1"}


def test_resolve_project_id_reuses_existing_by_name(monkeypatch) -> None:
    from npa.fleet import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_list_projects",
        lambda *a, **k: [
            {"metadata": {"name": "fleet1-test-a", "id": "project-found"}}
        ],
    )
    project = ProjectSpec(
        name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))]
    )
    pid, created = lifecycle.resolve_project_id(
        "nebius", "tenant-x", project, prefix="fleet1-test-", create=False, env={}
    )
    assert pid == "project-found"
    assert created is False


def test_resolve_project_id_errors_when_absent_and_no_create(monkeypatch) -> None:
    from npa.fleet import lifecycle

    monkeypatch.setattr(lifecycle, "_list_projects", lambda *a, **k: [])
    project = ProjectSpec(
        name="a", clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))]
    )
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
    monkeypatch.setattr(fleetcli, "_load", fleetcli._load)  # keep real loader
    import npa.fleet.lifecycle as L

    def _boom(*a, **k):
        called["deploy"] = True
        return {}

    monkeypatch.setattr(L, "deploy_fleet", _boom)
    # Answer "n" to the confirmation prompt.
    result = runner.invoke(
        app, ["fleet", "deploy", "--spec", str(_spec_file(tmp_path))], input="n\n"
    )
    assert result.exit_code == 1
    assert "Aborted." in result.output
    assert called["deploy"] is False  # deploy must not run when declined


def test_deploy_yes_flag_skips_prompt_and_runs(tmp_path, monkeypatch) -> None:
    import npa.fleet.lifecycle as L

    captured = {}

    def _fake_deploy(spec, **kwargs):
        captured.update(kwargs)
        return {
            "name": spec.name,
            "region": "us-central1",
            "tenant_id": "t",
            "deployed": 2,
            "failed": 0,
            "clusters": [],
        }

    monkeypatch.setattr(L, "deploy_fleet", _fake_deploy)
    result = runner.invoke(
        app, ["fleet", "deploy", "--spec", str(_spec_file(tmp_path)), "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "2 deployed" in result.output


def test_destroy_aborts_on_declined_confirmation(tmp_path, monkeypatch) -> None:
    import npa.fleet.lifecycle as L

    called = {"destroy": False}

    def _boom(*a, **k):
        called["destroy"] = True
        return {}

    monkeypatch.setattr(L, "destroy_fleet", _boom)
    result = runner.invoke(
        app, ["fleet", "destroy", "--spec", str(_spec_file(tmp_path))], input="n\n"
    )
    assert result.exit_code == 1
    assert "Aborted." in result.output
    assert "torn down" in result.output  # teardown/reclaim warning is shown
    assert called["destroy"] is False


def test_destroy_force_and_yes_skip_prompt(tmp_path, monkeypatch) -> None:
    import npa.fleet.lifecycle as L

    monkeypatch.setattr(
        L, "destroy_fleet", lambda spec, **k: {"name": spec.name, "clusters": []}
    )
    for flag in ("--yes", "--force"):
        result = runner.invoke(
            app, ["fleet", "destroy", "--spec", str(_spec_file(tmp_path)), flag]
        )
        assert result.exit_code == 0, result.output
        assert "Destroyed fleet" in result.output


def test_confirmation_lists_only_targeted_clusters(tmp_path, monkeypatch) -> None:
    import npa.fleet.lifecycle as L

    seen = {}

    def _fake_deploy(spec, **kwargs):
        seen.update(kwargs)
        return {
            "name": spec.name,
            "region": "r",
            "tenant_id": "t",
            "deployed": 0,
            "failed": 0,
            "clusters": [],
        }

    monkeypatch.setattr(L, "deploy_fleet", _fake_deploy)
    result = runner.invoke(
        app,
        [
            "fleet",
            "deploy",
            "--spec",
            str(_spec_file(tmp_path)),
            "--only-projects",
            "a",
            "--yes",
        ],
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


def test_ensure_subnet_reuses_existing_and_reports_no_created_network(
    monkeypatch,
) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(
        L, "_list_subnets", lambda *a, **k: [{"metadata": {"id": "subnet-x"}}]
    )
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

    root = _fake_recipe(
        tmp_path, 'provider "nebius" { domain = "api.eu.nebius.cloud:443" }\n'
    )
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
    root = _fake_recipe(
        tmp_path, 'provider "nebius" { domain = "renamed-domain:443" }\n'
    )
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


def _add_quiet_filesystem_verifier(recipe_root: Path) -> Path:
    verifier = (
        recipe_root
        / "k8s-training"
        / "filesystem-csi-validation"
        / "01-verify-node-filesystem-mounts.sh"
    )
    verifier.parent.mkdir(parents=True, exist_ok=True)
    verifier.write_text(
        "#!/usr/bin/env bash\n"
        "kubectl debug node/test \\\n"
        "  --attach=true \\\n"
        "  --quiet \\\n"
        "  --image=ubuntu -- true\n"
    )
    return verifier


def test_prepare_install_dir_patches_external_recipe_debug_quiet(tmp_path) -> None:
    from npa.fleet import lifecycle as L

    root = _fake_recipe(
        tmp_path, 'provider "nebius" { domain = "api.eu.nebius.cloud:443" }\n'
    )
    source = _add_quiet_filesystem_verifier(root)
    workdir = L._prepare_install_dir(
        tmp_path / "install",
        recipe_root=L._resolve_recipe_root(
            root, ref=None, work_root=tmp_path / "clones", on_status=None
        ),
        region="us-central1",
        cluster=ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1)),
        ssh_public_key="k",
    )
    installed = (
        workdir
        / "filesystem-csi-validation"
        / "01-verify-node-filesystem-mounts.sh"
    )
    assert "--quiet" in source.read_text()
    assert "--quiet" not in installed.read_text()


def test_prepare_install_dir_patches_fetched_recipe_debug_quiet(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle as L

    def clone(args, **_kwargs):
        clone_root = Path(args[-1])
        (clone_root / "k8s-training").mkdir(parents=True)
        (clone_root / "modules").mkdir()
        (clone_root / "k8s-training" / "variables.tf").write_text("")
        _add_quiet_filesystem_verifier(clone_root)

    monkeypatch.setattr(L, "_require_bin", lambda name: name)
    monkeypatch.setattr(L, "_run_stream", clone)
    root = L._resolve_recipe_root(
        None, ref="upstream-test", work_root=tmp_path / "fetch", on_status=None
    )
    workdir = L._prepare_install_dir(
        tmp_path / "install",
        recipe_root=root,
        region="us-central1",
        cluster=ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1)),
        ssh_public_key="k",
    )
    assert "--quiet" not in (
        workdir
        / "filesystem-csi-validation"
        / "01-verify-node-filesystem-mounts.sh"
    ).read_text()


# --------------------------------------------------------------------------- #
# _deploy_one_cluster sidecar status transitions (mocked)
# --------------------------------------------------------------------------- #
def _mock_deploy_boundary(monkeypatch, *, apply_fails: bool = False):
    from npa.fleet import lifecycle as L

    def fake_prepare(install_dir, **k):
        install_dir.mkdir(parents=True, exist_ok=True)
        return install_dir / "k8s-training"

    monkeypatch.setattr(L, "_prepare_install_dir", fake_prepare)
    monkeypatch.setattr(L, "_cluster_tf_env", lambda *a, **k: {})

    def fake_stream(*a, **k):
        if apply_fails:
            raise RuntimeError("terraform boom")
        return None

    monkeypatch.setattr(L, "_run_stream", fake_stream)
    monkeypatch.setattr(
        L,
        "_terraform_outputs",
        lambda *a, **k: {"kube_cluster": {"value": {"id": "mk8s-1"}}},
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
        subnet_id="subnet-1",
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


def test_deploy_one_cluster_success_promotes_sidecar(tmp_path, monkeypatch) -> None:
    L = _mock_deploy_boundary(monkeypatch)
    res = _run_one_cluster(L, tmp_path)
    assert res["status"] == "deployed"
    assert res["cluster_id"] == "mk8s-1"
    sidecar = json.loads((tmp_path / "a" / "c" / L._ENV_SIDECAR).read_text())
    assert sidecar["status"] == "deployed"
    assert sidecar["subnet_id"] == "subnet-1"
    assert sidecar["cluster_id"] == "mk8s-1"


def test_deploy_one_cluster_failure_leaves_sidecar_provisioning(
    tmp_path, monkeypatch
) -> None:
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
    monkeypatch.setattr(L, "_assert_terraform_version", lambda b: "1.12.0")
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
        projects=[
            ProjectSpec(
                name="a",
                clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))],
            )
        ],
    )
    L.destroy_fleet(spec, work_root=tmp_path)
    # Subnet first, then network.
    assert deleted == ["sub-9", "net-9"]


def test_destroy_leaves_reused_subnet_untouched(tmp_path, monkeypatch) -> None:
    # No created_network_id -> the subnet was pre-existing; must not be deleted.
    L, deleted = _setup_destroy(tmp_path, monkeypatch, {"created_network_id": ""})
    spec = FleetSpec(
        name="f",
        projects=[
            ProjectSpec(
                name="a",
                clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))],
            )
        ],
    )
    L.destroy_fleet(spec, work_root=tmp_path)
    assert deleted == []


def test_destroy_retains_project_ownership_after_partial_network_cleanup(
    tmp_path, monkeypatch
) -> None:
    L, _deleted = _setup_destroy(
        tmp_path, monkeypatch, {"created_network_id": "network-test"}
    )
    calls: list[str] = []

    def cap(cmd, **kwargs):
        kind = "subnet" if "subnet" in cmd else "network"
        calls.append(kind)
        return _Cap("busy", 5 if kind == "subnet" else 0)

    monkeypatch.setattr(L, "_run_capture", cap)
    spec = FleetSpec(
        name="f",
        projects=[
            ProjectSpec(
                name="a",
                clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))],
            )
        ],
    )
    result = L.destroy_fleet(spec, work_root=tmp_path)
    assert calls == ["subnet", "network"]
    assert result["failed"] == 1
    assert result["networks"][0]["status"] == "destroy-incomplete"
    assert (tmp_path / "f" / "a" / L._PROJECT_NETWORK_STATE).exists()


def _destroy_one_with_mocked_terraform(tmp_path, monkeypatch, *, destroy_fails: bool):
    from npa.fleet import lifecycle as L

    fleet_root = tmp_path / "f"
    install = fleet_root / "a" / "c"
    (install / L._K8S_TRAINING_SUBDIR).mkdir(parents=True)
    L._write_env_sidecar(
        install,
        {
            "tenant_id": "t",
            "project_id": "p1",
            "region": "us-central1",
            "subnet_id": "sub-9",
            "created_network_id": "net-9",
            "cluster_name": "c",
            "status": "deployed",
        },
    )
    calls: list[list[str]] = []

    def fake_tf_run(args, **kwargs):
        calls.append(args)
        if destroy_fails and "destroy" in args:
            raise RuntimeError("terraform destroy failed")

    monkeypatch.setattr(L, "_tf_run", fake_tf_run)
    monkeypatch.setattr(L, "_cluster_tf_env", lambda *a, **k: {})
    monkeypatch.setattr(L, "_find_cluster_id_by_name", lambda *a, **k: "")
    monkeypatch.setattr(L, "_reclaim_created_network", lambda *a, **k: None)
    result = L._destroy_one_cluster(
        spec=FleetSpec(name="f"),
        project=ProjectSpec(name="a"),
        cluster=ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1)),
        fleet_root=fleet_root,
        terraform_bin="terraform",
        nebius_bin="nebius",
        timeout_minutes=1,
        on_status=None,
    )
    return L, install, result, calls


def test_destroy_terraform_failure_retains_exact_state_for_retry(
    tmp_path, monkeypatch
) -> None:
    L, install, result, _calls = _destroy_one_with_mocked_terraform(
        tmp_path, monkeypatch, destroy_fails=True
    )

    assert result["status"] == "destroy-incomplete"
    assert "--only-projects a --only-clusters c" in result["retry_command"]
    assert install.exists()
    assert (install / L._ENV_SIDECAR).exists()


def test_destroy_success_removes_local_state(tmp_path, monkeypatch) -> None:
    _L, install, result, calls = _destroy_one_with_mocked_terraform(
        tmp_path, monkeypatch, destroy_fails=False
    )

    assert result["status"] == "destroyed"
    assert not install.exists()
    assert any("destroy" in call for call in calls)


def test_destroy_recovery_retries_retained_terraform_state(
    tmp_path, monkeypatch
) -> None:
    L, install, first, _calls = _destroy_one_with_mocked_terraform(
        tmp_path, monkeypatch, destroy_fails=True
    )
    assert first["status"] == "destroy-incomplete"
    assert install.exists()

    monkeypatch.setattr(L, "_tf_run", lambda *a, **k: None)
    second = L._destroy_one_cluster(
        spec=FleetSpec(name="f"),
        project=ProjectSpec(name="a"),
        cluster=ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1)),
        fleet_root=tmp_path / "f",
        terraform_bin="terraform",
        nebius_bin="nebius",
        timeout_minutes=1,
        on_status=None,
    )
    assert second["status"] == "destroyed"
    assert not install.exists()


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
    monkeypatch.setattr(L, "_assert_terraform_version", lambda b: "1.12.0")
    monkeypatch.setattr(L, "_resolve_tenant_id", lambda *a, **k: "tenant-x")
    monkeypatch.setattr(L, "_resolve_region", lambda *a, **k: "us-central1")
    monkeypatch.setattr(L, "_resolve_ssh_public_key", lambda *a, **k: "ssh-key")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *a, **k: tmp_path / "recipe")
    monkeypatch.setattr(L, "_nebius_cli_env", lambda: {})
    monkeypatch.setattr(L, "resolve_project_id", lambda *a, **k: ("proj-1", False))
    monkeypatch.setattr(L, "ensure_subnet", lambda *a, **k: ("subnet-1", ""))
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 1))
    built: list[str] = []

    def fake_one(**kwargs):
        name = kwargs["cluster"].name
        built.append(name)
        return {
            "project_key": "a",
            "cluster_name": name,
            "status": "deployed",
            "cluster_id": f"id-{name}",
        }

    monkeypatch.setattr(L, "_deploy_one_cluster", fake_one)
    res = L.deploy_fleet(
        _two_cluster_project_spec(),
        work_root=tmp_path,
        only_clusters=["c2"],
        preflight=False,
    )
    assert built == ["c2"]  # only the targeted cluster is (re)deployed
    assert res["deployed"] == 1


def test_deploy_help_documents_concurrency() -> None:
    result = runner.invoke(app, ["fleet", "deploy", "--help"])
    assert result.exit_code == 0
    assert "--concurrency" in result.output


def _mock_deploy_fleet_boundary(monkeypatch, tmp_path):
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_assert_terraform_version", lambda b: "1.12.0")
    monkeypatch.setattr(L, "_resolve_tenant_id", lambda *a, **k: "t")
    monkeypatch.setattr(L, "_resolve_region", lambda *a, **k: "us-central1")
    monkeypatch.setattr(L, "_resolve_ssh_public_key", lambda *a, **k: "k")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *a, **k: tmp_path / "recipe")
    monkeypatch.setattr(L, "_nebius_cli_env", lambda: {})
    monkeypatch.setattr(L, "_list_projects", lambda *a, **k: [])
    monkeypatch.setattr(L, "resolve_project_id", lambda *a, **k: ("proj-1", False))
    monkeypatch.setattr(L, "ensure_subnet", lambda *a, **k: ("subnet-1", ""))
    # Tests outside the preflight section opt out explicitly when this generic
    # boundary does not model the quota API.
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 1))
    return L


def test_deploy_fleet_parallel_runs_all_targets_and_prewarms_once(
    tmp_path, monkeypatch
) -> None:
    import threading

    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    prewarm = {"n": 0}
    monkeypatch.setattr(
        L,
        "_prewarm_plugin_cache",
        lambda *a, **k: prewarm.__setitem__("n", prewarm["n"] + 1),
    )
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
    res = L.deploy_fleet(
        _two_cluster_project_spec(),
        work_root=tmp_path,
        concurrency=2,
        preflight=False,
    )
    assert sorted(ran) == ["c1", "c2"]  # both targets applied
    assert res["deployed"] == 2
    assert prewarm["n"] == 1  # plugin cache pre-warmed exactly once for parallel
    assert all(lp is not None for lp in log_paths)  # parallel -> per-cluster log files


def test_deploy_fleet_sequential_skips_prewarm_and_streams(
    tmp_path, monkeypatch
) -> None:
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    prewarm = {"n": 0}
    monkeypatch.setattr(
        L,
        "_prewarm_plugin_cache",
        lambda *a, **k: prewarm.__setitem__("n", prewarm["n"] + 1),
    )
    log_paths: list = []

    def fake_one(**kwargs):
        log_paths.append(kwargs.get("log_path"))
        return {
            "project_key": "a",
            "cluster_name": kwargs["cluster"].name,
            "status": "deployed",
        }

    monkeypatch.setattr(L, "_deploy_one_cluster", fake_one)
    L.deploy_fleet(
        _two_cluster_project_spec(),
        work_root=tmp_path,
        concurrency=1,
        preflight=False,
    )
    assert prewarm["n"] == 0  # no pre-warm when sequential
    assert log_paths == [None, None]  # sequential -> stream to stdout, no log file


def test_destroy_fleet_parallel_runs_all_and_prunes(tmp_path, monkeypatch) -> None:
    import threading

    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_assert_terraform_version", lambda b: "1.12.0")
    ran: list[str] = []
    lock = threading.Lock()

    def fake_destroy_one(**kwargs):
        with lock:
            ran.append(kwargs["cluster"].name)
        return {
            "project_key": "a",
            "cluster_name": kwargs["cluster"].name,
            "status": "destroyed",
        }

    pruned = {}
    monkeypatch.setattr(L, "_destroy_one_cluster", fake_destroy_one)
    monkeypatch.setattr(
        L, "_prune_fleet_state", lambda fr, keys: pruned.update({"keys": keys})
    )
    L.destroy_fleet(_two_cluster_project_spec(), work_root=tmp_path, concurrency=2)
    assert sorted(ran) == ["c1", "c2"]
    assert pruned["keys"] == {("a", "c1"), ("a", "c2")}


def test_run_to_log_writes_and_raises(tmp_path) -> None:
    from npa.fleet import lifecycle as L

    log = tmp_path / "deploy.log"
    L._run_to_log(
        ["true"], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, timeout=30, log_path=log
    )
    assert log.exists() and "$ true" in log.read_text()
    with pytest.raises(RuntimeError, match="command failed"):
        L._run_to_log(
            ["false"],
            cwd=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            timeout=30,
            log_path=log,
        )


def test_upsert_and_prune_fleet_state_roundtrip(tmp_path) -> None:
    from npa.fleet import lifecycle as L

    fleet_root = tmp_path / "f"
    fleet_root.mkdir()
    base = {
        "name": "f",
        "tenant_id": "t",
        "region": "r",
        "project_prefix": "",
        "k8s_training_source": "x",
    }
    # Deploy c1, then add c2 -- both must be present (upsert must not clobber c1).
    L._upsert_fleet_state(
        fleet_root,
        base,
        [{"project_key": "a", "cluster_name": "c1", "status": "deployed"}],
    )
    L._upsert_fleet_state(
        fleet_root,
        base,
        [{"project_key": "a", "cluster_name": "c2", "status": "deployed"}],
    )
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
            "profiles": {
                "other": {"tenant-id": "tenant-other"},
                "sd": {"tenant-id": "tenant-sd"},
            },
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
        lambda: {
            "default": "other",
            "profiles": {"other": {"tenant-id": "tenant-other"}, "sd": {}},
        },
    )
    with pytest.raises(ValueError, match="has no 'tenant-id'"):
        L._resolve_tenant_id("nebius", "", "sd")


def test_list_projects_passes_profile_to_cli(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    seen: list[list[str]] = []
    monkeypatch.setattr(
        L,
        "_run_capture",
        lambda cmd, **k: (seen.append(cmd), _Cap('{"items": []}', 0))[1],
    )
    L._list_projects("nebius", "tenant-x", {}, "sd")
    assert seen[0][:3] == ["nebius", "--profile", "sd"]


def test_deploy_fleet_cli_profile_overrides_spec(tmp_path, monkeypatch) -> None:
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    seen: list[str] = []

    def fake_one(**kwargs):
        seen.append(kwargs["profile"])
        return {
            "project_key": "a",
            "cluster_name": kwargs["cluster"].name,
            "status": "deployed",
        }

    monkeypatch.setattr(L, "_deploy_one_cluster", fake_one)
    spec = FleetSpec(
        name="f",
        profile="from-spec",
        projects=[
            ProjectSpec(
                name="a",
                clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))],
            )
        ],
    )
    res = L.deploy_fleet(spec, work_root=tmp_path, preflight=False)
    assert seen == ["from-spec"]
    assert res["profile"] == "from-spec"
    seen.clear()
    res = L.deploy_fleet(
        spec, work_root=tmp_path, profile="from-cli", preflight=False
    )
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
        {
            "tenant_id": "t",
            "project_id": "p1",
            "region": "us-central1",
            "subnet_id": "s",
            "cluster_name": "c",
            "profile": "sd",
            "status": "deployed",
        },
    )
    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_assert_terraform_version", lambda b: "1.12.0")
    monkeypatch.setattr(L, "_run_stream", lambda *a, **k: None)
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 0))
    seen: list[str] = []
    monkeypatch.setattr(
        L, "_terraform_env", lambda b, **k: (seen.append(k.get("profile", "")), {})[1]
    )
    spec = FleetSpec(
        name="f",
        projects=[
            ProjectSpec(
                name="a",
                clusters=[ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1))],
            )
        ],
    )
    L.destroy_fleet(spec, work_root=tmp_path)
    assert seen == ["sd"]  # teardown authenticates as the deploying principal


def test_deploy_and_destroy_help_document_profile() -> None:
    for cmd in ("deploy", "destroy", "plan"):
        result = runner.invoke(app, ["fleet", cmd, "--help"])
        assert result.exit_code == 0
        assert "--profile" in result.output


_FAKE_TERRAFORM_MARKER = "UNMISTAKABLE_TERRAFORM_SUBPROCESS_MARKER"
_FAKE_IAM_TOKEN = "sentinel-iam-token-that-must-be-redacted"


def _fake_terraform_executable(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-terraform"
    executable.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            action="${{1:-}}"
            case "$action" in
              version)
                printf '{{"terraform_version":"1.12.3"}}\\n'
                ;;
              output)
                printf '{{"kube_cluster":{{"value":{{"id":"mk8s-test"}}}}}}\\n'
                ;;
              init|apply|destroy)
                printf '{_FAKE_TERRAFORM_MARKER} stdout %s\\n' "$action"
                printf '{_FAKE_TERRAFORM_MARKER} stderr %s\\n' "$action" >&2
                printf 'credential=%s\\n' "${{TF_VAR_iam_token:-missing}}"
                printf 'credential=%s\\n' "${{TF_VAR_iam_token:-missing}}" >&2
                if [ "${{NPA_FAKE_TF_FAIL_ACTION:-}}" = "$action" ]; then
                  exit 17
                fi
                ;;
              *)
                printf 'unexpected fake terraform action: %s\\n' "$action" >&2
                exit 19
                ;;
            esac
            """
        )
    )
    executable.chmod(0o700)
    return executable


def _subprocess_fleet_spec(tmp_path: Path, cluster_names: tuple[str, ...]) -> Path:
    path = tmp_path / "fleet-subprocess.yaml"
    path.write_text(
        json.dumps(
            {
                "apiVersion": "npa.fleet/v0.0.1",
                "name": "f",
                "region": "us-central1",
                "projects": [
                    {
                        "name": "a",
                        "clusters": [
                            {
                                "name": name,
                                "cpu_nodes": {
                                    "count": 1,
                                    "platform": "cpu-d3",
                                    "preset": "4vcpu-16gb",
                                },
                            }
                            for name in cluster_names
                        ],
                    }
                ],
            }
        )
    )
    return path


def _patch_infra_free_subprocess_boundaries(tmp_path: Path, monkeypatch):
    from npa.fleet import lifecycle as L

    terraform = _fake_terraform_executable(tmp_path)
    recipe = tmp_path / "recipe"
    (recipe / "k8s-training").mkdir(parents=True)
    (recipe / "modules").mkdir()
    work_root = tmp_path / "work"

    monkeypatch.setenv("NPA_TERRAFORM_BIN", str(terraform))
    monkeypatch.setenv("NPA_NEBIUS_BIN", "/bin/true")
    monkeypatch.setattr(L, "_default_work_root", lambda: work_root)
    monkeypatch.setattr(L, "_resolve_tenant_id", lambda *a, **k: "tenant-test")
    monkeypatch.setattr(L, "_resolve_region", lambda *a, **k: "us-central1")
    monkeypatch.setattr(L, "_resolve_ssh_public_key", lambda *a, **k: "ssh-test")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *a, **k: recipe)
    monkeypatch.setattr(L, "_nebius_cli_env", lambda: {})
    monkeypatch.setattr(L, "resolve_project_id", lambda *a, **k: ("project-test", False))
    monkeypatch.setattr(L, "ensure_subnet", lambda *a, **k: ("subnet-test", ""))
    monkeypatch.setattr(L, "_find_cluster_id_by_name", lambda *a, **k: "")

    def fake_tf_env(*args, **kwargs):
        return {
            "PATH": os.environ["PATH"],
            "TF_VAR_iam_token": _FAKE_IAM_TOKEN,
            "NEBIUS_IAM_TOKEN": _FAKE_IAM_TOKEN,
            "NPA_FAKE_TF_FAIL_ACTION": os.environ.get(
                "NPA_FAKE_TF_FAIL_ACTION", ""
            ),
        }

    def fake_kubeconfig(_bin, _cluster_id, path, _context, _env, _profile=""):
        path.write_text("apiVersion: v1\n")

    monkeypatch.setattr(L, "_terraform_env", fake_tf_env)
    monkeypatch.setattr(L, "_write_kubeconfig", fake_kubeconfig)
    return L, work_root


def _prepare_destroy_state(L, work_root: Path, cluster_names: tuple[str, ...]) -> None:
    for cluster_name in cluster_names:
        install_dir = work_root / "f" / "a" / cluster_name
        (install_dir / L._K8S_TRAINING_SUBDIR).mkdir(parents=True)
        L._write_env_sidecar(
            install_dir,
            {
                "tenant_id": "tenant-test",
                "project_id": "project-test",
                "region": "us-central1",
                "subnet_id": "subnet-test",
                "cluster_name": cluster_name,
                "status": "deployed",
            },
        )


@pytest.mark.parametrize(
    ("command", "extra_args", "fail_action"),
    [
        ("deploy", [], ""),  # default concurrency=1
        ("destroy", ["-j", "4"], ""),  # one target is still non-parallel
        ("deploy", [], "apply"),
        ("destroy", [], "destroy"),
    ],
)
def test_json_lifecycle_subprocess_is_logged_redacted_and_stdout_stays_one_document(
    command, extra_args, fail_action, tmp_path, monkeypatch
) -> None:
    L, work_root = _patch_infra_free_subprocess_boundaries(tmp_path, monkeypatch)
    cluster_names = ("c",)
    spec_file = _subprocess_fleet_spec(tmp_path, cluster_names)
    if command == "destroy":
        _prepare_destroy_state(L, work_root, cluster_names)
    if fail_action:
        monkeypatch.setenv("NPA_FAKE_TF_FAIL_ACTION", fail_action)

    fleet_root = work_root / "f"
    if command == "deploy":
        expected_log = fleet_root / "a" / "c" / "deploy.log"
    else:
        expected_log = fleet_root / ".logs" / "a" / "c" / "destroy.log"
    expected_log.parent.mkdir(parents=True, exist_ok=True)
    expected_log.write_text("PREEXISTING_DIAGNOSTIC\n")
    expected_log.chmod(0o644)

    args = [
        "fleet",
        command,
        "--spec",
        str(spec_file),
        "--yes",
        "--output",
        "json",
        *extra_args,
    ]
    if command == "deploy":
        args.append("--no-preflight")
    result = runner.invoke(app, args)

    assert result.exit_code == (1 if fail_action else 0), result.output
    payload = json.loads(result.stdout)
    assert payload["failed"] == (1 if fail_action else 0)
    assert _FAKE_TERRAFORM_MARKER not in result.stdout
    assert _FAKE_IAM_TOKEN not in result.stdout
    assert _FAKE_IAM_TOKEN not in result.stderr

    entry = payload["clusters"][0]
    assert Path(entry["terraform_log"]) == expected_log
    log_text = expected_log.read_text()
    assert "PREEXISTING_DIAGNOSTIC" in log_text  # append, never truncate
    assert _FAKE_TERRAFORM_MARKER in log_text
    assert "<redacted>" in log_text
    assert _FAKE_IAM_TOKEN not in log_text
    assert stat.S_IMODE(expected_log.stat().st_mode) == 0o600
    current = expected_log.parent
    while current != fleet_root.parent:
        assert stat.S_IMODE(current.stat().st_mode) == 0o700
        if current == fleet_root:
            break
        current = current.parent


def test_json_multiple_targets_remain_sequential_and_each_subprocess_is_logged(
    tmp_path, monkeypatch
) -> None:
    _L, work_root = _patch_infra_free_subprocess_boundaries(tmp_path, monkeypatch)
    spec_file = _subprocess_fleet_spec(tmp_path, ("c1", "c2"))
    result = runner.invoke(
        app,
        [
            "fleet",
            "deploy",
            "--spec",
            str(spec_file),
            "--yes",
            "--no-preflight",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["deployed"] == 2
    assert _FAKE_TERRAFORM_MARKER not in result.stdout
    for entry in payload["clusters"]:
        log_path = Path(entry["terraform_log"])
        assert log_path.is_file()
        assert _FAKE_TERRAFORM_MARKER in log_path.read_text()
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
        assert work_root / "f" in log_path.parents


def test_json_parallel_prewarm_and_targets_never_write_terraform_to_stdout(
    tmp_path, monkeypatch
) -> None:
    _L, work_root = _patch_infra_free_subprocess_boundaries(tmp_path, monkeypatch)
    spec_file = _subprocess_fleet_spec(tmp_path, ("c1", "c2"))
    result = runner.invoke(
        app,
        [
            "fleet",
            "deploy",
            "--spec",
            str(spec_file),
            "--yes",
            "--no-preflight",
            "--output",
            "json",
            "-j",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["deployed"] == 2
    assert _FAKE_TERRAFORM_MARKER not in result.stdout
    prewarm_log = work_root / "f" / ".logs" / "terraform-prewarm.log"
    for log_path in [
        prewarm_log,
        *(Path(entry["terraform_log"]) for entry in payload["clusters"]),
    ]:
        text = log_path.read_text()
        assert _FAKE_TERRAFORM_MARKER in text
        assert _FAKE_IAM_TOKEN not in text
        assert "<redacted>" in text
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("link_kind", "error_text"),
    [
        ("symlink", "securely open Terraform diagnostics log"),
        ("hardlink", "multiple hard links"),
    ],
)
def test_json_deploy_refuses_unsafe_preexisting_log_link_without_touching_target(
    link_kind, error_text, tmp_path, monkeypatch
) -> None:
    _L, work_root = _patch_infra_free_subprocess_boundaries(tmp_path, monkeypatch)
    spec_file = _subprocess_fleet_spec(tmp_path, ("c",))
    victim = tmp_path / "victim"
    victim.write_text("DO_NOT_TOUCH\n")
    victim_mode = stat.S_IMODE(victim.stat().st_mode)
    log_path = work_root / "f" / "a" / "c" / "deploy.log"
    log_path.parent.mkdir(parents=True)
    if link_kind == "symlink":
        log_path.symlink_to(victim)
    else:
        os.link(victim, log_path)

    result = runner.invoke(
        app,
        [
            "fleet",
            "deploy",
            "--spec",
            str(spec_file),
            "--yes",
            "--no-preflight",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["failed"] == 1
    assert error_text in payload["clusters"][0]["error"]
    assert log_path.is_symlink() is (link_kind == "symlink")
    assert victim.read_text() == "DO_NOT_TOUCH\n"
    assert stat.S_IMODE(victim.stat().st_mode) == victim_mode
    assert _FAKE_IAM_TOKEN not in result.stdout
    assert _FAKE_IAM_TOKEN not in result.stderr


@pytest.mark.parametrize("command", ["deploy", "destroy"])
def test_json_handled_lifecycle_error_still_emits_one_document(
    command, tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle as L

    def fail(*args, **kwargs):
        raise ValueError("sanitized preflight failure")

    monkeypatch.setattr(L, f"{command}_fleet", fail)
    result = runner.invoke(
        app,
        [
            "fleet",
            command,
            "--spec",
            str(_spec_file(tmp_path)),
            "--yes",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["failed"] == 1
    assert payload["error"] == "sanitized preflight failure"
    assert "sanitized preflight failure" in result.stderr


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
            {
                "name": "f",
                "region": "us-central1",
                "tenant_id": "t",
                "deployed": 1,
                "failed": 0,
                "clusters": [],
            },
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
            {
                "name": "f",
                "region": "us-central1",
                "tenant_id": "t",
                "deployed": 1,
                "failed": 0,
                "clusters": [],
            },
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
    assert "--output" in TOOL_CATALOG["infra.fleet.deploy"].argv_template
    assert "json" in TOOL_CATALOG["infra.fleet.deploy"].argv_template


# --------------------------------------------------------------------------- #
# tenant quota preflight (mk8s accepts node groups it cannot fill)
# --------------------------------------------------------------------------- #
def _allowance(
    name: str,
    region: str,
    limit,
    unit: str = "count",
    *,
    parent_id: str = "",
) -> dict:
    metadata = {"name": name}
    if parent_id:
        metadata["parent_id"] = parent_id
    return {
        "metadata": metadata,
        "spec": {"region": region, "limit": limit},
        "status": {"unit": unit, "usage_percentage": "0.00"},
    }


def _complete_allowances(*overrides: dict) -> str:
    """Return sufficient evidence for every quota in the RTX fleet fixture."""

    defaults = [
        _allowance("mk8s.cluster.count", "us-central1", 100),
        _allowance("compute.instance.count", "us-central1", 100),
        _allowance("compute.disk.count", "us-central1", 100),
        _allowance(
            "compute.disk.size.network-ssd", "us-central1", 100 * 1024**4, "byte"
        ),
        _allowance("compute.instance.non-gpu.vcpu", "us-central1", 1000),
        _allowance("compute.instance.gpu.rtx6000", "us-central1", 100),
        _allowance("vpc.allocation.count", "us-central1", 1000),
        _allowance("vpc.network.count", "us-central1", 100),
        _allowance("vpc.pool.count", "us-central1", 100),
        _allowance("vpc.route.count", "us-central1", 100),
        _allowance("vpc.routetable.count", "us-central1", 100),
        _allowance("vpc.subnet.count", "us-central1", 100),
    ]
    indexed = {item["metadata"]["name"]: item for item in defaults}
    indexed.update({item["metadata"]["name"]: item for item in overrides})
    return json.dumps({"items": list(indexed.values())})


def _capacity_block(
    *,
    reservation_id: str = "capacityblockgroup-test",
    tenant_id: str = "t",
    region: str = "us-central1",
    platform: str = "gpu-rtx6000",
    fabric: str = "",
    limit: str = "16",
    usage: str | None = None,
    usage_percentage: str = "0.00",
    state: str = "STATE_ACTIVE",
) -> dict:
    compute_v1 = {"platform": platform}
    if fabric:
        compute_v1["fabric"] = fabric
    status = {
        "region": region,
        "resource_affinity": {"compute_v1": compute_v1},
        "state": state,
        "current_limit": limit,
        "usage_percentage": usage_percentage,
        "usage_state": (
            "USAGE_STATE_NOT_USED" if usage_percentage == "0.00" else "USAGE_STATE_USED"
        ),
    }
    if usage is not None:
        status["usage"] = usage
    return {
        "metadata": {"id": reservation_id, "parent_id": tenant_id},
        "status": status,
    }


def test_required_quotas_counts_nodes_vcpu_gpus_and_filesystem() -> None:
    from npa.fleet.quotas import required_quotas

    needed = required_quotas(
        [
            ClusterSpec(
                name="c",
                cpu_nodes=NodePoolSpec(
                    count=2, platform="cpu-d3", preset="48vcpu-192gb"
                ),
                gpu_nodes=NodePoolSpec(
                    count=2, platform="gpu-rtx6000", preset="8gpu-192vcpu-1744gb"
                ),
                enable_gpu_cluster=False,
                enable_filestore=True,
                filestore_disk_size_gibibytes=2048,
            )
        ]
    )
    # Managed control-plane etcd consumes neither customer VM nor disk quota.
    assert needed["compute.instance.count"] == 4
    assert needed["compute.disk.count"] == 4
    assert needed["compute.disk.size.network-ssd"] == (2 * 128 + 2 * 1023) * 1024**3
    assert needed["compute.instance.non-gpu.vcpu"] == 96  # GPU-node vCPUs excluded
    assert needed["compute.instance.gpu.rtx6000"] == 16
    assert needed["compute.filesystem.count"] == 1
    assert needed["compute.filesystem.size.network-ssd"] == 2048 * 1024**3
    assert needed["mk8s.cluster.count"] == 1
    assert needed["vpc.allocation.count"] == 8
    assert "compute.gpucluster.count" not in needed

    with_project_topology = required_quotas(
        [ClusterSpec(name="cpu", cpu_nodes=NodePoolSpec(count=1))], new_projects=1
    )
    assert with_project_topology["vpc.network.count"] == 1
    assert with_project_topology["vpc.pool.count"] == 2
    assert with_project_topology["vpc.route.count"] == 1
    assert with_project_topology["vpc.routetable.count"] == 1
    assert with_project_topology["vpc.subnet.count"] == 1


def test_required_quotas_excludes_managed_control_plane_compute_resources() -> None:
    from npa.fleet.quotas import required_quotas

    needed = required_quotas(
        [
            ClusterSpec(
                name="c",
                cpu_nodes=NodePoolSpec(
                    count=1, platform="cpu-d3", preset="4vcpu-16gb"
                ),
            )
        ]
    )
    # The service-owned etcd/control plane uses neither customer VM nor disk
    # quota. Only the single node-group VM and its boot disk are counted.
    assert needed["compute.instance.count"] == 1
    assert needed["compute.disk.count"] == 1
    assert needed["compute.disk.size.network-ssd"] == 128 * 1024**3


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


def test_required_quotas_excludes_strictly_reserved_gpu_but_keeps_other_quotas() -> (
    None
):
    from npa.fleet.quotas import required_quotas, required_reservations

    cluster = ClusterSpec(
        name="c",
        cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="48vcpu-192gb"),
        gpu_nodes=NodePoolSpec(
            count=2,
            platform="gpu-rtx6000",
            preset="8gpu-192vcpu-1744gb",
            capacity_block_group="capacityblockgroup-test",
        ),
        enable_gpu_cluster=False,
    )
    needed = required_quotas([cluster])
    assert "compute.instance.gpu.rtx6000" not in needed
    assert needed["compute.instance.count"] == 3
    assert needed["compute.disk.count"] == 3
    assert needed["compute.instance.non-gpu.vcpu"] == 48
    reservations = required_reservations([cluster], "us-central1")
    assert reservations["capacityblockgroup-test"].required_gpus == 16


def test_reservation_capacity_parser_and_validation_match_region_platform_fabric() -> (
    None
):
    from npa.fleet.quotas import (
        ReservationRequirement,
        find_reservation_shortfalls,
        parse_capacity_blocks,
    )

    blocks = parse_capacity_blocks(
        json.dumps(
            {
                "items": [
                    _capacity_block(
                        platform="gpu-b200-sxm", fabric="us-central1-b", limit="40"
                    )
                ]
            }
        )
    )
    requirement = ReservationRequirement(
        reservation_id="capacityblockgroup-test",
        region="us-central1",
        platform="gpu-b200-sxm",
        fabric="us-central1-b",
        required_gpus=16,
    )
    assert (
        find_reservation_shortfalls(
            {requirement.reservation_id: requirement}, blocks, "t"
        )
        == []
    )
    wrong_fabric = ReservationRequirement(
        **{**requirement.__dict__, "fabric": "us-central1-a"}
    )
    shortfalls = find_reservation_shortfalls(
        {wrong_fabric.reservation_id: wrong_fabric}, blocks, "t"
    )
    assert "fabric" in shortfalls[0].reason


def test_reservation_capacity_parser_treats_live_usage_percentage_as_fraction() -> None:
    from npa.fleet.quotas import parse_capacity_blocks

    blocks = parse_capacity_blocks(
        json.dumps(
            {
                "items": [
                    _capacity_block(limit="48", usage_percentage="0.17"),
                ]
            }
        )
    )
    # The live API emits 0.17 for about 17%, not 0.17%.
    assert blocks["capacityblockgroup-test"]["available_gpus"] == 39

    exact = parse_capacity_blocks(
        json.dumps(
            {
                "items": [
                    _capacity_block(limit="48", usage="8", usage_percentage="0.17"),
                ]
            }
        )
    )
    assert exact["capacityblockgroup-test"]["available_gpus"] == 40


def test_parse_allowances_filters_region_and_preserves_unlimited() -> None:
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
    assert parsed["compute.instance.gpu.rtx6000"]["unlimited"] is True


@pytest.mark.parametrize("tenant_first", [True, False])
def test_parse_allowances_tenant_finite_cannot_be_shadowed_by_project_unlimited(
    tenant_first: bool,
) -> None:
    from npa.fleet.quotas import find_shortfalls, parse_allowances

    tenant = _allowance(
        "compute.disk.count",
        "us-central1",
        3,
        parent_id="tenant-test",
    )
    project = _allowance(
        "compute.disk.count",
        "us-central1",
        None,
        parent_id="project-test",
    )
    project.pop("status")  # Unlimited wire entries may omit status entirely.
    items = [tenant, project] if tenant_first else [project, tenant]

    parsed = parse_allowances(
        json.dumps({"items": items}),
        "us-central1",
        required={"compute.disk.count"},
        container_id="tenant-test",
    )

    assert parsed["compute.disk.count"]["unlimited"] is False
    assert parsed["compute.disk.count"]["limit"] == 3
    assert len(
        find_shortfalls(
            {"compute.disk.count": 4}, parsed, "us-central1"
        )
    ) == 1


@pytest.mark.parametrize("unlimited_first", [True, False])
def test_parse_allowances_unscoped_finite_wins_concrete_unlimited_collision(
    unlimited_first: bool,
) -> None:
    from npa.fleet.quotas import parse_allowances

    finite = _allowance("compute.instance.count", "us-central1", 8)
    unlimited = _allowance("compute.instance.count", "us-central1", None)
    unlimited.pop("status")
    items = [unlimited, finite] if unlimited_first else [finite, unlimited]

    parsed = parse_allowances(json.dumps({"items": items}), "us-central1")

    assert parsed["compute.instance.count"]["limit"] == 8
    assert parsed["compute.instance.count"]["unlimited"] is False


def test_parse_allowances_authoritative_tenant_scope_beats_project_finite() -> None:
    from npa.fleet.quotas import parse_allowances

    tenant = _allowance(
        "compute.instance.count",
        "us-central1",
        None,
        parent_id="tenant-test",
    )
    tenant.pop("status")
    project = _allowance(
        "compute.instance.count",
        "us-central1",
        0,
        parent_id="project-test",
    )

    parsed = parse_allowances(
        json.dumps({"items": [project, tenant]}),
        "us-central1",
        container_id="tenant-test",
    )

    assert parsed["compute.instance.count"]["unlimited"] is True


def test_parse_allowances_duplicate_finite_candidates_are_deterministic_or_fail_closed() -> (
    None
):
    from npa.fleet.quotas import parse_allowances

    first = _allowance(
        "compute.instance.count", "us-central1", 10, parent_id="tenant-test"
    )
    duplicate = json.loads(json.dumps(first))
    parsed = parse_allowances(
        json.dumps({"items": [duplicate, first]}),
        "us-central1",
        container_id="tenant-test",
    )
    assert parsed["compute.instance.count"]["limit"] == 10

    conflicting = _allowance(
        "compute.instance.count", "us-central1", 11, parent_id="tenant-test"
    )
    with pytest.raises(ValueError, match="ambiguous finite candidates"):
        parse_allowances(
            json.dumps({"items": [first, conflicting]}),
            "us-central1",
            container_id="tenant-test",
        )


@pytest.mark.parametrize(
    ("name", "unit"),
    [
        ("compute.disk.size.network-ssd", "byte"),
        ("compute.disk.count", "count"),
        ("compute.instance.non-gpu.vcpu", "vcpu"),
        ("compute.instance.non-gpu.vcpu", "count"),
        ("compute.instance.gpu.b200", "gpu"),
        ("compute.instance.gpu.b200", "count"),
    ],
)
def test_parse_allowances_accepts_only_known_compatible_units(
    name: str, unit: str
) -> None:
    from npa.fleet.quotas import parse_allowances

    parsed = parse_allowances(
        json.dumps({"items": [_allowance(name, "us-central1", 10, unit)]}),
        "us-central1",
    )
    assert parsed[name]["unit"] == unit


@pytest.mark.parametrize(
    ("name", "unit"),
    [
        ("compute.disk.size.network-ssd", "gibibyte"),
        ("compute.disk.count", "vcpu"),
        ("compute.instance.non-gpu.vcpu", "byte"),
        ("compute.instance.gpu.b200", "vcpu"),
    ],
)
def test_parse_allowances_rejects_incompatible_units(name: str, unit: str) -> None:
    from npa.fleet.quotas import parse_allowances

    with pytest.raises(ValueError, match="incompatible unit"):
        parse_allowances(
            json.dumps({"items": [_allowance(name, "us-central1", 10, unit)]}),
            "us-central1",
        )


def test_parse_allowances_unlimited_may_omit_status_but_finite_may_not() -> None:
    from npa.fleet.quotas import parse_allowances

    unlimited = _allowance("compute.instance.count", "us-central1", None)
    unlimited.pop("status")
    parsed = parse_allowances(json.dumps({"items": [unlimited]}), "us-central1")
    assert parsed["compute.instance.count"]["unlimited"] is True

    finite = _allowance("compute.instance.count", "us-central1", 10)
    finite.pop("status")
    with pytest.raises(ValueError, match="unusable capacity evidence"):
        parse_allowances(json.dumps({"items": [finite]}), "us-central1")


@pytest.mark.parametrize("missing_status_first", [True, False])
def test_parse_allowances_duplicate_unlimited_candidates_are_deterministic(
    missing_status_first: bool,
) -> None:
    from npa.fleet.quotas import parse_allowances

    without_status = _allowance("compute.instance.count", "us-central1", None)
    without_status.pop("status")
    with_status = _allowance("compute.instance.count", "us-central1", None)
    items = (
        [without_status, with_status]
        if missing_status_first
        else [with_status, without_status]
    )

    parsed = parse_allowances(json.dumps({"items": items}), "us-central1")

    assert parsed["compute.instance.count"]["unlimited"] is True
    assert parsed["compute.instance.count"]["unit"] == "count"

    with_status["status"]["unit"] = "byte"
    with pytest.raises(ValueError, match="incompatible unit"):
        parse_allowances(
            json.dumps({"items": [without_status, with_status]}), "us-central1"
        )


def test_parse_allowances_finite_status_variations_remain_fail_closed() -> None:
    from npa.fleet.quotas import parse_allowances

    not_used = _allowance("compute.instance.count", "us-central1", 10)
    not_used["status"] = {
        "unit": "count",
        "usage_state": "USAGE_STATE_NOT_USED",
    }
    parsed = parse_allowances(json.dumps({"items": [not_used]}), "us-central1")
    assert parsed["compute.instance.count"]["available"] == 10
    assert parsed["compute.instance.count"]["usage_source"] == "not-used"

    over_limit = _allowance("compute.instance.count", "us-central1", 10)
    over_limit["status"]["usage_percentage"] = "1.25"
    parsed = parse_allowances(json.dumps({"items": [over_limit]}), "us-central1")
    assert parsed["compute.instance.count"]["available"] == 0

    no_usage = _allowance("compute.instance.count", "us-central1", 10)
    no_usage["status"] = {"unit": "count"}
    with pytest.raises(ValueError, match="unusable capacity evidence"):
        parse_allowances(json.dumps({"items": [no_usage]}), "us-central1")

    frozen = _allowance("compute.instance.count", "us-central1", 10)
    frozen["status"]["state"] = "STATE_FROZEN"
    with pytest.raises(ValueError, match="unusable capacity evidence"):
        parse_allowances(json.dumps({"items": [frozen]}), "us-central1")


@pytest.mark.parametrize(
    "broken",
    [
        {"metadata": {"name": "compute.instance.count"}, "spec": []},
        {
            "metadata": {"name": "compute.instance.count"},
            "spec": {"region": "us-central1", "limit": 1.5},
            "status": {"unit": "count", "usage": 0},
        },
        {
            "metadata": {"name": "compute.instance.count"},
            "spec": {"region": "us-central1", "limit": 10},
            "status": "active",
        },
    ],
)
def test_parse_allowances_rejects_malformed_relevant_candidates(broken: dict) -> None:
    from npa.fleet.quotas import parse_allowances

    with pytest.raises(ValueError):
        parse_allowances(
            json.dumps({"items": [broken]}),
            "us-central1",
            required={"compute.instance.count"},
        )


def test_parse_allowances_rejects_incomplete_paginated_response() -> None:
    from npa.fleet.quotas import parse_allowances

    with pytest.raises(ValueError, match="incomplete after paginated read"):
        parse_allowances(
            json.dumps(
                {
                    "items": [
                        _allowance("compute.instance.count", "us-central1", 10)
                    ],
                    "next_page_token": "still-more",
                }
            ),
            "us-central1",
        )


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
    if rc == 0:
        supplied = json.loads(allowances_json or "{}").get("items", [])
        allowances_json = _complete_allowances(*supplied)
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap(allowances_json, rc))
    deployed: list[str] = []
    monkeypatch.setattr(
        L,
        "_deploy_one_cluster",
        lambda **kw: (
            deployed.append(kw["cluster"].name),
            {
                "project_key": "a",
                "cluster_name": kw["cluster"].name,
                "status": "deployed",
            },
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
                            count=2,
                            platform="gpu-rtx6000",
                            preset="8gpu-192vcpu-1744gb",
                        ),
                        enable_gpu_cluster=False,
                    )
                ],
            )
        ],
    )


def _reserved_rtx_cluster_spec() -> FleetSpec:
    spec = _rtx_cluster_spec()
    spec.projects[0].clusters[
        0
    ].gpu_nodes.capacity_block_group = "capacityblockgroup-test"
    return spec


def _reserved_preflight_boundary(
    monkeypatch, tmp_path, capacity_payload: str, *, rc: int = 0
):
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    calls: list[list[str]] = []

    def capture(args, **_kwargs):
        calls.append(args)
        if "capacity-block-group" in args:
            return _Cap(capacity_payload, rc)
        return _Cap(
            _complete_allowances(),
            0,
        )

    monkeypatch.setattr(L, "_run_capture", capture)
    deployed: list[str] = []
    monkeypatch.setattr(
        L,
        "_deploy_one_cluster",
        lambda **kw: (
            deployed.append(kw["cluster"].name),
            {
                "project_key": "a",
                "cluster_name": kw["cluster"].name,
                "status": "deployed",
            },
        )[1],
    )
    return L, deployed, calls


def test_reserved_preflight_uses_capacity_and_ignores_public_gpu_quota(
    tmp_path, monkeypatch
) -> None:
    payload = json.dumps({"items": [_capacity_block()]})
    L, deployed, calls = _reserved_preflight_boundary(monkeypatch, tmp_path, payload)
    messages: list[str] = []
    L.deploy_fleet(
        _reserved_rtx_cluster_spec(),
        work_root=tmp_path,
        on_status=messages.append,
    )
    assert deployed == ["c"]
    assert any("capacity-block-group" in call for call in calls)
    assert any("quota-allowance" in call and "--all" in call for call in calls)
    assert any("ordinary GPU quota excluded" in message for message in messages)
    assert any("compute.disk.count" in message for message in messages)


def test_reserved_preflight_fails_closed_when_block_unreadable_or_too_small(
    tmp_path, monkeypatch
) -> None:
    L, deployed, _calls = _reserved_preflight_boundary(monkeypatch, tmp_path, "", rc=1)
    with pytest.raises(ValueError, match="refusing to bypass STRICT"):
        L.deploy_fleet(_reserved_rtx_cluster_spec(), work_root=tmp_path)
    assert deployed == []

    payload = json.dumps({"items": [_capacity_block(limit="8")]})
    L, deployed, _calls = _reserved_preflight_boundary(monkeypatch, tmp_path, payload)
    with pytest.raises(ValueError, match="only 8 reserved GPUs remain"):
        L.deploy_fleet(_reserved_rtx_cluster_spec(), work_root=tmp_path)
    assert deployed == []


def test_deploy_preflight_blocks_on_zero_gpu_quota(tmp_path, monkeypatch) -> None:
    payload = json.dumps(
        {"items": [_allowance("compute.instance.gpu.rtx6000", "us-central1", "0")]}
    )
    L, deployed = _preflight_boundary(monkeypatch, tmp_path, payload)
    with pytest.raises(ValueError, match="compute.instance.gpu.rtx6000"):
        L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)
    assert deployed == []  # nothing applied


def test_deploy_preflight_blocks_when_four_workers_have_only_three_disk_slots(
    tmp_path, monkeypatch
) -> None:
    disk_allowance = _allowance("compute.disk.count", "us-central1", "25")
    disk_allowance["status"]["usage"] = "22"
    L, deployed = _preflight_boundary(
        monkeypatch,
        tmp_path,
        json.dumps({"items": [disk_allowance]}),
    )
    spec = _rtx_cluster_spec()
    spec.projects[0].clusters[0].cpu_nodes = NodePoolSpec(
        count=2, platform="cpu-d3", preset="16vcpu-64gb"
    )

    with pytest.raises(
        ValueError,
        match=r"compute\.disk\.count .*needs 4 count, 3 available.*limit 25",
    ):
        L.deploy_fleet(spec, work_root=tmp_path)
    assert deployed == []


def test_deploy_no_preflight_skips_the_check(tmp_path, monkeypatch) -> None:
    payload = json.dumps(
        {"items": [_allowance("compute.instance.gpu.rtx6000", "us-central1", "0")]}
    )
    L, deployed = _preflight_boundary(monkeypatch, tmp_path, payload)
    L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path, preflight=False)
    assert deployed == ["c"]


def test_deploy_preflight_passes_when_quota_is_sufficient(
    tmp_path, monkeypatch
) -> None:
    payload = json.dumps(
        {"items": [_allowance("compute.instance.gpu.rtx6000", "us-central1", "16")]}
    )
    L, deployed = _preflight_boundary(monkeypatch, tmp_path, payload)
    L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)
    assert deployed == ["c"]


def test_deploy_preflight_unreadable_quota_api_fails_closed(
    tmp_path, monkeypatch
) -> None:
    # Losing the preflight is not proof of capacity and must block mutation.
    L, deployed = _preflight_boundary(monkeypatch, tmp_path, "", rc=1)
    with pytest.raises(ValueError, match="refusing to create projects"):
        L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)
    assert deployed == []


def test_deploy_preflight_missing_required_allowance_fails_closed(
    tmp_path, monkeypatch
) -> None:
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    monkeypatch.setattr(
        L,
        "_run_capture",
        lambda *a, **k: _Cap(
            json.dumps(
                {
                    "items": [
                        _allowance(
                            "compute.instance.gpu.rtx6000", "us-central1", 100
                        )
                    ]
                }
            ),
            0,
        ),
    )
    with pytest.raises(ValueError, match="omitted required allowances"):
        L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)


def test_preflight_allows_only_explicit_optional_unadvertised_topology_quotas() -> (
    None
):
    from npa.fleet.quotas import preflight_region, required_quotas

    cluster = ClusterSpec(
        name="cpu",
        cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="4vcpu-16gb"),
    )
    needed = required_quotas([cluster], new_projects=1)
    optional = {
        "vpc.network.count",
        "vpc.subnet.count",
        "vpc.pool.count",
        "vpc.routetable.count",
        "vpc.route.count",
    }
    items = [
        _allowance(
            name,
            "us-central1",
            10 * 1024**4 if name == "compute.disk.size.network-ssd" else 100,
            "byte" if name == "compute.disk.size.network-ssd" else "count",
            parent_id="tenant-test",
        )
        for name in sorted(set(needed) - optional)
    ]
    messages: list[str] = []
    shortfalls = preflight_region(
        nebius_bin="nebius",
        tenant_id="tenant-test",
        region="us-central1",
        clusters=[cluster],
        env={},
        new_projects=1,
        run_capture=lambda *a, **k: _Cap(json.dumps({"items": items})),
        nebius_argv=lambda binary, profile: [binary],
        on_status=messages.append,
    )
    assert shortfalls == []
    assert sum("unadvertised optional quota" in message for message in messages) == 5

    missing_core = [
        item
        for item in items
        if item["metadata"]["name"] != "compute.instance.count"
    ]
    with pytest.raises(ValueError, match="omitted required allowances"):
        preflight_region(
            nebius_bin="nebius",
            tenant_id="tenant-test",
            region="us-central1",
            clusters=[cluster],
            env={},
            new_projects=1,
            run_capture=lambda *a, **k: _Cap(
                json.dumps({"items": missing_core})
            ),
            nebius_argv=lambda binary, profile: [binary],
        )


def test_preflight_runs_before_any_project_is_created(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    # Regression: the preflight used to run *after* project resolution, so a
    # quota-blocked deploy left a freshly created, empty project behind.
    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_assert_terraform_version", lambda b: "1.12.0")
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
            _complete_allowances(
                _allowance("compute.instance.gpu.rtx6000", "us-central1", "0")
            ),
            0,
        ),
    )
    monkeypatch.setattr(L, "_deploy_one_cluster", lambda **kw: {"status": "deployed"})
    with pytest.raises(ValueError, match="quota is too low"):
        L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)
    assert resolved == []  # no project touched


def _project_quota_accounting_boundary(monkeypatch, tmp_path, projects):
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    list_calls: list[str] = []
    preflight_calls: list[dict[str, int]] = []
    resolve_calls: list[str] = []
    deployed: list[str] = []

    def list_projects(*_args, **_kwargs):
        list_calls.append("list")
        if isinstance(projects, BaseException):
            raise projects
        return projects

    monkeypatch.setattr(L, "_list_projects", list_projects)
    monkeypatch.setattr(
        L,
        "_preflight_quotas",
        lambda *a, **k: preflight_calls.append(dict(k["new_projects_by_region"])),
    )

    def resolve(_binary, _tenant, project, **_kwargs):
        resolve_calls.append(project.key())
        return f"resolved-{project.key()}", not bool(project.project_id)

    monkeypatch.setattr(L, "resolve_project_id", resolve)
    monkeypatch.setattr(
        L,
        "_deploy_one_cluster",
        lambda **k: (
            deployed.append(k["project"].key()),
            {
                "project_key": k["project"].key(),
                "cluster_name": k["cluster"].name,
                "status": "deployed",
            },
        )[1],
    )
    return L, list_calls, preflight_calls, resolve_calls, deployed


def test_preflight_project_accounting_existing_named_and_genuinely_new(
    tmp_path, monkeypatch
) -> None:
    existing_spec = _rtx_cluster_spec()
    existing_name = existing_spec.projects[0].display_name(
        existing_spec.project_prefix
    )
    L, list_calls, preflight_calls, resolve_calls, deployed = (
        _project_quota_accounting_boundary(
            monkeypatch,
            tmp_path,
            [{"metadata": {"name": existing_name, "id": "project-existing"}}],
        )
    )
    L.deploy_fleet(existing_spec, work_root=tmp_path)
    assert list_calls == ["list"]
    assert preflight_calls == [{}]
    assert resolve_calls == []
    assert deployed == ["a"]

    new_root = tmp_path / "new"
    L, list_calls, preflight_calls, resolve_calls, deployed = (
        _project_quota_accounting_boundary(monkeypatch, new_root, [])
    )
    L.deploy_fleet(_rtx_cluster_spec(), work_root=new_root)
    assert list_calls == ["list"]
    assert preflight_calls == [{"us-central1": 1}]
    assert resolve_calls == ["a"]
    assert deployed == ["a"]


def test_preflight_project_accounting_explicit_id_and_mixed_projects(
    tmp_path, monkeypatch
) -> None:
    cluster = ClusterSpec(
        name="c", cpu_nodes=NodePoolSpec(count=1, preset="4vcpu-16gb")
    )
    explicit_only = FleetSpec(
        name="f",
        region="us-central1",
        projects=[ProjectSpec(project_id="project-explicit", clusters=[cluster])],
    )
    L, list_calls, preflight_calls, _resolve_calls, _deployed = (
        _project_quota_accounting_boundary(monkeypatch, tmp_path / "explicit", [])
    )
    L.deploy_fleet(explicit_only, work_root=tmp_path / "explicit")
    assert list_calls == []
    assert preflight_calls == [{}]

    mixed = FleetSpec(
        name="mixed",
        region="us-central1",
        projects=[
            ProjectSpec(name="existing", clusters=[cluster]),
            ProjectSpec(name="new", region="eu-north1", clusters=[cluster]),
            ProjectSpec(project_id="project-explicit", clusters=[cluster]),
        ],
    )
    L, list_calls, preflight_calls, resolve_calls, deployed = (
        _project_quota_accounting_boundary(
            monkeypatch,
            tmp_path / "mixed",
            [{"metadata": {"name": "existing", "id": "project-existing"}}],
        )
    )
    L.deploy_fleet(mixed, work_root=tmp_path / "mixed")
    assert list_calls == ["list"]
    assert preflight_calls == [{"eu-north1": 1}]
    assert resolve_calls == ["new", "project-explicit"]
    assert sorted(deployed) == ["existing", "new", "project-explicit"]


def test_preflight_project_lookup_failure_cannot_undercount(
    tmp_path, monkeypatch
) -> None:
    L, _list_calls, preflight_calls, resolve_calls, deployed = (
        _project_quota_accounting_boundary(
            monkeypatch, tmp_path, RuntimeError("project inventory unavailable")
        )
    )
    with pytest.raises(RuntimeError, match="project inventory unavailable"):
        L.deploy_fleet(_rtx_cluster_spec(), work_root=tmp_path)
    assert preflight_calls == []
    assert resolve_calls == []
    assert deployed == []


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
    pinned = spec_from_mapping(
        {**_base_mapping(), "tenant_id": "tenant-pinned", "profile": "sd"}
    )
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


def test_destroy_only_clusters_removes_install_dir_and_prunes_state(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle as L

    # Two deployed clusters recorded in state + install dirs; destroy only c2.
    fleet_root = tmp_path / "f"
    for name in ("c1", "c2"):
        d = fleet_root / "a" / name
        (d / L._K8S_TRAINING_SUBDIR).mkdir(parents=True)
        L._write_env_sidecar(
            d,
            {
                "tenant_id": "t",
                "project_id": "p1",
                "region": "us-central1",
                "subnet_id": "s",
                "cluster_name": name,
                "status": "deployed",
            },
        )
    base = {
        "name": "f",
        "tenant_id": "t",
        "region": "r",
        "project_prefix": "",
        "k8s_training_source": "x",
    }
    L._upsert_fleet_state(
        fleet_root,
        base,
        [
            {"project_key": "a", "cluster_name": "c1", "status": "deployed"},
            {"project_key": "a", "cluster_name": "c2", "status": "deployed"},
        ],
    )
    monkeypatch.setattr(L, "_require_bin", lambda b: b)
    monkeypatch.setattr(L, "_assert_terraform_version", lambda b: "1.12.0")
    monkeypatch.setattr(L, "_terraform_env", lambda b, **k: {})
    monkeypatch.setattr(L, "_run_stream", lambda *a, **k: None)
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 0))

    spec = _two_cluster_project_spec()
    L.destroy_fleet(spec, work_root=tmp_path, only_clusters=["c2"])

    assert not (fleet_root / "a" / "c2").exists()  # removed
    assert (fleet_root / "a" / "c1").exists()  # untouched
    state = L._load_fleet_state(fleet_root)
    assert [c["cluster_name"] for c in state["clusters"]] == ["c1"]


# --------------------------------------------------------------------------- #
# Follow-up review regressions: recovery, versioning, concurrency, and JSON
# --------------------------------------------------------------------------- #
def test_terraform_version_accepts_supported_and_newer_prerelease(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    for version in ("1.12.0", "1.12.3", "1.13.0-rc1", "2.0.0"):
        monkeypatch.setattr(
            L,
            "_run_capture",
            lambda *a, version=version, **k: _Cap(
                json.dumps({"terraform_version": version}), 0
            ),
        )
        assert L._assert_terraform_version("terraform") == version


@pytest.mark.parametrize("version", ["1.11.9", "1.12.0-rc1"])
def test_terraform_version_rejects_old_or_not_yet_final(monkeypatch, version) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(
        L,
        "_run_capture",
        lambda *a, **k: _Cap(json.dumps({"terraform_version": version}), 0),
    )
    with pytest.raises(ValueError, match=r"Terraform >= 1\.12.*found"):
        L._assert_terraform_version("terraform")


def test_terraform_version_rejects_malformed_and_command_failure(monkeypatch) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("not-json", 0))
    with pytest.raises(ValueError, match="could not parse"):
        L._assert_terraform_version("terraform")
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 7))
    with pytest.raises(ValueError, match="exited 7"):
        L._assert_terraform_version("terraform")


def test_parallel_deploy_resolves_one_project_subnet_before_workers(
    tmp_path, monkeypatch
) -> None:
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    calls: list[Path] = []

    def one_subnet(*args, **kwargs):
        calls.append(kwargs["network_state_path"])
        return "shared-subnet", "owned-network"

    monkeypatch.setattr(L, "ensure_subnet", one_subnet)
    monkeypatch.setattr(L, "_prewarm_plugin_cache", lambda *a, **k: None)
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        L,
        "_deploy_one_cluster",
        lambda **kw: (
            seen.append((kw["cluster"].name, kw["subnet_id"])),
            {
                "project_key": "a",
                "cluster_name": kw["cluster"].name,
                "status": "deployed",
            },
        )[1],
    )
    L.deploy_fleet(
        _two_cluster_project_spec(),
        work_root=tmp_path,
        concurrency=2,
        preflight=False,
    )
    assert calls == [tmp_path / "f" / "a" / L._PROJECT_NETWORK_STATE]
    assert sorted(seen) == [("c1", "shared-subnet"), ("c2", "shared-subnet")]


def test_explicit_subnet_override_bypasses_shared_project_subnet(
    tmp_path, monkeypatch
) -> None:
    L = _mock_deploy_fleet_boundary(monkeypatch, tmp_path)
    spec = _two_cluster_project_spec()
    spec.projects[0].clusters[0].subnet_id = "explicit-subnet"
    monkeypatch.setattr(L, "ensure_subnet", lambda *a, **k: ("shared-subnet", ""))
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        L,
        "_deploy_one_cluster",
        lambda **kw: (
            seen.__setitem__(kw["cluster"].name, kw["subnet_id"]),
            {
                "project_key": "a",
                "cluster_name": kw["cluster"].name,
                "status": "deployed",
            },
        )[1],
    )
    L.deploy_fleet(spec, work_root=tmp_path, preflight=False)
    assert seen == {"c1": "explicit-subnet", "c2": "shared-subnet"}


def test_ensure_subnet_persists_project_ownership_before_subnet_create(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_list_subnets", lambda *a, **k: [])
    state_path = tmp_path / L._PROJECT_NETWORK_STATE

    def cap(cmd, **kwargs):
        if "network" in cmd:
            return _Cap(json.dumps({"metadata": {"id": "network-test"}}), 0)
        raise RuntimeError("subnet create failed")

    monkeypatch.setattr(L, "_run_capture", cap)
    with pytest.raises(RuntimeError, match="subnet create failed"):
        L.ensure_subnet(
            "nebius",
            "project-test",
            name_stem="a",
            env={},
            network_state_path=state_path,
        )
    state = json.loads(state_path.read_text())
    assert state == {
        "project_id": "project-test",
        "created_network_id": "network-test",
        "subnet_id": "",
        "profile": "",
    }


@pytest.mark.parametrize("command", ["deploy", "destroy"])
def test_empty_scope_json_emits_one_document_without_lifecycle(
    command, tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle as L

    def unexpected(*args, **kwargs):
        raise AssertionError("lifecycle must not run for an empty scope")

    monkeypatch.setattr(L, f"{command}_fleet", unexpected)
    result = runner.invoke(
        app,
        [
            "fleet",
            command,
            "--spec",
            str(_spec_file(tmp_path)),
            "--only-projects",
            "missing",
            "--yes",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["clusters"] == []
    assert payload["failed"] == 0
    assert result.stdout.count("{") == 1


def test_write_kubeconfig_fails_on_command_error_or_missing_file(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 9))
    with pytest.raises(RuntimeError, match="exited 9"):
        L._write_kubeconfig("nebius", "cluster-test", tmp_path / "kube", "ctx", {})
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 0))
    with pytest.raises(RuntimeError, match="without a kubeconfig"):
        L._write_kubeconfig("nebius", "cluster-test", tmp_path / "kube", "ctx", {})
    assert not (tmp_path / "kube").exists()


def test_deploy_kubeconfig_failure_is_partial_and_retains_state(
    tmp_path, monkeypatch
) -> None:
    L = _mock_deploy_boundary(monkeypatch)
    monkeypatch.setattr(
        L,
        "_write_kubeconfig",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("denied")),
    )
    result = _run_one_cluster(L, tmp_path)
    assert result["status"] == "deployed-credentials-failed"
    assert result["kubeconfig"] == ""
    sidecar = json.loads((tmp_path / "a" / "c" / L._ENV_SIDECAR).read_text())
    assert sidecar["status"] == "deployed-credentials-failed"
    assert sidecar["cluster_id"] == "mk8s-1"


def test_destroy_fallback_failure_is_reported_and_state_retained(
    tmp_path, monkeypatch
) -> None:
    L, install, _result, _calls = _destroy_one_with_mocked_terraform(
        tmp_path, monkeypatch, destroy_fails=True
    )
    monkeypatch.setattr(L, "_find_cluster_id_by_name", lambda *a, **k: "cluster-test")
    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("permission denied", 7))
    result = L._destroy_one_cluster(
        spec=FleetSpec(name="f"),
        project=ProjectSpec(name="a"),
        cluster=ClusterSpec(name="c", cpu_nodes=NodePoolSpec(count=1)),
        fleet_root=tmp_path / "f",
        terraform_bin="terraform",
        nebius_bin="nebius",
        timeout_minutes=1,
        on_status=None,
    )
    assert result["status"] == "destroy-incomplete"
    assert any("fallback delete failed" in error for error in result["errors"])
    assert install.exists()


def test_network_cleanup_attempts_later_steps_and_retains_ownership_on_failure(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle as L

    calls: list[str] = []

    def cap(cmd, **kwargs):
        kind = "subnet" if "subnet" in cmd else "network"
        calls.append(kind)
        return _Cap("busy", 4 if kind == "subnet" else 0)

    monkeypatch.setattr(L, "_run_capture", cap)
    errors = L._reclaim_created_network(
        "nebius", "project-test", "network-test", "subnet-test", {}, None, "a"
    )
    assert calls == ["subnet", "network"]
    assert errors == ["subnet delete failed (nebius exited 4)"]

    calls.clear()

    def raising_cap(cmd, **kwargs):
        kind = "subnet" if "subnet" in cmd else "network"
        calls.append(kind)
        if kind == "subnet":
            raise RuntimeError("temporary API failure")
        return _Cap("", 0)

    monkeypatch.setattr(L, "_run_capture", raising_cap)
    errors = L._reclaim_created_network(
        "nebius", "project-test", "network-test", "subnet-test", {}, None, "a"
    )
    assert calls == ["subnet", "network"]
    assert errors == ["subnet delete failed: RuntimeError: temporary API failure"]


def test_quota_filesystem_byte_unit_usage_and_drift() -> None:
    from npa.fleet.quotas import find_shortfalls, parse_allowances

    allowance = _allowance(
        "compute.filesystem.size.network-ssd", "us-central1", 1000, "byte"
    )
    allowance["status"]["usage_percentage"] = "0.25"
    parsed = parse_allowances(json.dumps({"items": [allowance]}), "us-central1")
    assert parsed["compute.filesystem.size.network-ssd"]["available"] == 750
    assert (
        find_shortfalls(
            {"compute.filesystem.size.network-ssd": 750}, parsed, "us-central1"
        )
        == []
    )
    assert (
        find_shortfalls(
            {"compute.filesystem.size.network-ssd": 751}, parsed, "us-central1"
        )[0].available
        == 750
    )

    zero = _allowance("compute.filesystem.size.network-ssd", "us-central1", 0, "byte")
    parsed_zero = parse_allowances(json.dumps({"items": [zero]}), "us-central1")
    assert (
        len(
            find_shortfalls(
                {"compute.filesystem.size.network-ssd": 1}, parsed_zero, "us-central1"
            )
        )
        == 1
    )

    drift = _allowance(
        "compute.filesystem.size.network-ssd", "us-central1", 1, "gibibyte"
    )
    with pytest.raises(ValueError, match="expected.*byte"):
        parse_allowances(json.dumps({"items": [drift]}), "us-central1")


def test_nebius_discovery_fails_closed_and_state_write_logs(
    tmp_path, monkeypatch, caplog
) -> None:
    from npa.fleet import lifecycle as L

    monkeypatch.setattr(L, "_run_capture", lambda *a, **k: _Cap("", 8))
    with pytest.raises(RuntimeError, match="could not list projects"):
        L._list_projects("nebius", "tenant-test", {})
    with pytest.raises(RuntimeError, match="could not list subnets"):
        L._list_subnets("nebius", "project-test", {})

    monkeypatch.setattr(
        L,
        "_write_json_file",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    L._write_fleet_state(tmp_path, {"name": "f"})
    assert "could not persist fleet summary" in caplog.text


def test_nebius_config_parse_failure_logs_without_content(tmp_path, monkeypatch, caplog) -> None:
    from npa.fleet import lifecycle as L

    config_dir = tmp_path / ".nebius"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("profiles: [secret-token\n")
    monkeypatch.setattr(L.Path, "home", lambda: tmp_path)
    assert L._nebius_config() == {}
    assert "could not parse Nebius config" in caplog.text
    assert "secret-token" not in caplog.text


def test_spec_rejects_quota_inputs_that_cannot_be_counted() -> None:
    with pytest.raises(FleetSpecError, match="cannot be negative"):
        ClusterSpec(
            name="c", cpu_nodes=NodePoolSpec(count=-1), gpu_nodes=NodePoolSpec(count=2)
        ).validate()
    with pytest.raises(FleetSpecError, match="platform must start"):
        ClusterSpec(
            name="c", gpu_nodes=NodePoolSpec(count=1, platform="cpu-d3")
        ).validate()
    with pytest.raises(FleetSpecError, match="positive GPU count"):
        ClusterSpec(
            name="c", gpu_nodes=NodePoolSpec(count=1, platform="gpu-h200", preset="bad")
        ).validate()
