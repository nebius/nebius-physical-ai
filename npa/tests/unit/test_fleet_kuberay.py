"""The CPU RayCluster contract must reach actual resources, or fail before apply."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from npa.cluster_backends.kuberay import (
    KubeRaySpec, RAY_IMAGE, RAY_VERSION, validate_recipe_kuberay_compatibility,
    kuberay_materialized_digest, validate_kuberay_destroyed_state,
)
from npa.cluster_backends.mk8s import MK8sApplyRequest, MK8sBackend
from npa.cluster_backends import mk8s_execution as E
from npa.cluster_backends.mk8s_model import as_mk8s_desired
from npa.cluster_backends.mk8s_render import render_tfvars
from npa.fleet import lifecycle as L
from npa.fleet.spec import ClusterSpec, FleetSpec, NodePoolSpec, ProjectSpec, spec_from_mapping
from npa.sdk import fleet


def cluster(policy=None):
    return ClusterSpec(name="cpu-ray", cpu_nodes=NodePoolSpec(count=1), kuberay=policy)


def spec(policy):
    return spec_from_mapping({
        "name": "ray-example", "defaults": {
            "cpu_nodes": {"count": 1, "platform": "cpu-d3", "preset": "16vcpu-64gb"},
            "kuberay": policy,
        }, "projects": [{"name": "example"}],
    })


@pytest.fixture
def recipe(tmp_path):
    source = L._find_vendored_recipe_root()
    assert source is not None
    root = tmp_path / "recipe"
    shutil.copytree(source, root)
    return root


def test_defaults_remain_disabled_and_legacy_tfvars_identical(tmp_path):
    baseline = render_tfvars(cluster())
    assert baseline == render_tfvars(cluster(KubeRaySpec()))
    assert "enable_kuberay_cluster   = false" in baseline
    assert "enable_kuberay_service   = false" in baseline
    assert "kuberay_cpu_cluster" not in baseline
    validate_recipe_kuberay_compatibility(cluster(), tmp_path)


def test_policy_reaches_backend_sdk_plan_and_actual_root_variables(recipe):
    desired = cluster(KubeRaySpec(True, 3, 4, 8))
    desired.validate()
    backend = as_mk8s_desired(desired)
    assert backend.kuberay == desired.kuberay
    assert fleet.KubeRaySpec is KubeRaySpec
    rendered = render_tfvars(backend, recipe_dir=recipe / "k8s-training")
    config = json.loads(next(line.split("=", 1)[1] for line in rendered.splitlines() if line.startswith("kuberay_cpu_cluster =")))
    assert config == {"worker_replicas": 3, "worker_cpus": 4, "worker_memory_gib": 8}
    assert "enable_kuberay_cluster   = true" in rendered
    assert "enable_kuberay_service   = false" in rendered
    assert MK8sBackend().plan(backend)["kuberay"] == desired.kuberay.plan()
    declaration = FleetSpec(name="ray", projects=[ProjectSpec(name="example", clusters=[desired])])
    assert L.plan_fleet(declaration)["projects"][0]["clusters"][0]["kuberay"]["worker_replicas"] == 3


@pytest.mark.parametrize("policy", [
    [], True, "cluster", {"enabled": "true"}, {"enabled": 1},
    {"enabled": True, "mode": "service"}, {"enabled": True, "gpu_workers": 1},
    {"enabled": True, "image": "rayproject/ray:latest"},
    {"enabled": True, "min_replicas": 1}, {"worker_cpus": 2},
    {"enabled": False, "worker_replicas": 1},
    *[{"enabled": True, field: value} for field in ("worker_replicas", "worker_cpus", "worker_memory_gib") for value in (0, -1, True, 1.5, "2", None)],
])
def test_invalid_mapping_refused(policy):
    with pytest.raises(ValueError, match="kuberay"):
        spec(policy).validate()


def test_sdk_cannot_bypass_type_or_cpu_pool_validation():
    for policy in (KubeRaySpec(enabled="yes"), KubeRaySpec(True, True), KubeRaySpec(False, 3), {}):
        with pytest.raises(ValueError, match="kuberay"):
            cluster(policy).validate()
    for pool in (None, NodePoolSpec(count=0), NodePoolSpec(count=1, platform="gpu-test"), NodePoolSpec(count=1, preset="")):
        with pytest.raises(ValueError, match="kuberay"):
            replace(cluster(KubeRaySpec(True)), cpu_nodes=pool).validate()
    with pytest.raises(ValueError, match="mk8s"):
        replace(cluster(KubeRaySpec(True)), backend="soperator").validate()


def test_disable_replaces_default_policy_and_explicit_envelope_works():
    declaration = {
        "name": "example", "defaults": {
            "cpu_nodes": {"count": 1, "preset": "16vcpu-64gb"},
            "kuberay": {"enabled": True, "worker_replicas": 3},
        }, "projects": [{"name": "example", "clusters": [
            {"name": "off", "kuberay": {"enabled": False}},
            {"name": "on", "backend": "mk8s", "mk8s": {"kuberay": {"enabled": True, "worker_replicas": 2}}},
        ]}],
    }
    desired = spec_from_mapping(declaration)
    desired.validate()
    off, on = desired.projects[0].clusters
    assert off.kuberay == KubeRaySpec()
    assert on.kuberay == KubeRaySpec(True, 2)


@pytest.mark.parametrize("path,old,new", [
    ("k8s-training/variables.tf", 'variable "kuberay_cpu_cluster"', 'variable "unused"'),
    ("k8s-training/applications.tf", "cpu_cluster = var.kuberay_cpu_cluster", "cpu_cluster = null"),
    ("modules/kuberay/main.tf", "ray-values-cpu.yaml.tftpl", "ray-values.yaml.tftpl"),
    ("modules/kuberay/main.tf", 'policy_types = ["Ingress"]', 'policy_types = []'),
    ("modules/kuberay/files/ray-values-cpu.yaml.tftpl", 'rayVersion: "2.58.0"', 'rayVersion: "2.46.0"'),
    ("modules/kuberay/files/ray-values-cpu.yaml.tftpl", "${worker_cpus}", "2"),
    ("k8s-training/main.tf", "platform = local.cpu_nodes_platform", 'platform = "cpu-different"'),
    ("k8s-training/locals.tf", "cpu_nodes_platform", "changed_platform"),
    ("modules/cloud-init/k8s-cloud-init.tftpl", "package_update: true", "package_update: false"),
    ("modules/o11y/versions.tf", "terraform", "changed"),
])
def test_recipe_mutants_refused_before_any_cloud_mutation(recipe, path, old, new, monkeypatch):
    target = recipe / path
    original = target.read_text()
    assert old in original
    target.write_text(original.replace(old, new))
    desired = cluster(KubeRaySpec(True))
    declaration = FleetSpec(name="ray", tenant_id="tenant-test", region="us-central1", ssh_public_key="ssh-test", projects=[ProjectSpec(project_id="project-test", clusters=[desired])])
    monkeypatch.setattr(L, "_require_bin", lambda name: name)
    monkeypatch.setattr(L, "_assert_terraform_version", lambda _: "1.13.3")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *a, **kw: recipe)
    monkeypatch.setattr(L, "resolve_project_id", lambda *a, **kw: pytest.fail("project mutation"))
    monkeypatch.setattr(L, "preflight_region", lambda *a, **kw: pytest.fail("quota preflight"))
    with pytest.raises(ValueError, match="reviewed CPU KubeRay contract"):
        L.deploy_fleet(declaration, work_root=recipe.parent / "state")
    with pytest.raises(ValueError, match="reviewed CPU KubeRay contract"):
        render_tfvars(desired, recipe_dir=recipe / "k8s-training")


@pytest.mark.parametrize("path,content", [
    ("k8s-training/override.tf", 'module "kuberay" { cpu_cluster = null }'),
    ("k8s-training/extra_override.tf.json", '{"module":{"kuberay":{"cpu_cluster":null}}}'),
    ("modules/kuberay/override.tf", 'resource "kubernetes_network_policy_v1" "cpu_cluster" { count = 0 }'),
    ("modules/o11y/extra.tf", 'resource "terraform_data" "unexpected" {}'),
    ("k8s-training/terraform.tfvars.json", '{"kuberay_cpu_cluster":null}'),
    ("k8s-training/extra.auto.tfvars", "kuberay_cpu_cluster = null"),
    ("k8s-training/extra.auto.tfvars.json", '{"enable_kuberay_cluster":false}'),
])
def test_additional_effective_recipe_inputs_are_rejected(recipe, path, content, tmp_path):
    (recipe / path).write_text(content)
    destination = tmp_path / "must-not-materialize"
    with pytest.raises(ValueError, match="reviewed CPU KubeRay contract"):
        L._prepare_install_dir(destination, recipe_root=recipe, region="eu-north1",
                               cluster=cluster(KubeRaySpec(True)), ssh_public_key="ssh-test")
    assert not destination.exists()


def test_recipe_symlink_is_rejected_even_when_bytes_match(recipe):
    path = recipe / "modules/kuberay/main.tf"
    saved = recipe.parent / "saved.tf"
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(ValueError, match="Symlinks"):
        validate_recipe_kuberay_compatibility(cluster(KubeRaySpec(True)), recipe / "k8s-training")


@pytest.mark.parametrize("relative", ["k8s-training/override.tf", "modules/kuberay/extra.tf"])
def test_special_source_fails_before_provider_or_retained_file_mutation(recipe, tmp_path, monkeypatch, relative):
    desired, destination, deploy = _isolated_deploy(recipe, tmp_path, monkeypatch)
    work = destination / "k8s-training"
    cache = work / ".terraform/modules"
    cache.mkdir(parents=True)
    (cache / "modules.json").write_text("retained cache")
    (work / "terraform.tfvars").write_text("retained variables")
    os.mkfifo(recipe / relative)
    with pytest.raises(ValueError, match="Special source entries"):
        deploy()
    with pytest.raises(ValueError, match="Special source entries"):
        E._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                               cluster=desired, ssh_public_key="ssh-test")
    assert (work / "terraform.tfvars").read_text() == "retained variables"
    assert (cache / "modules.json").read_text() == "retained cache"


@pytest.mark.parametrize("subtree", ["k8s-training", "modules"])
def test_source_inventory_requires_actual_directory_roots(recipe, subtree):
    shutil.rmtree(recipe / subtree)
    (recipe / subtree).write_text("not a directory")
    with pytest.raises(ValueError, match="Source roots must be directories"):
        validate_recipe_kuberay_compatibility(cluster(KubeRaySpec(True)), recipe / "k8s-training")


def test_source_traversal_errors_are_not_silently_ignored(recipe, monkeypatch):
    original = os.scandir
    hidden = recipe / "modules/kuberay"

    def denied(path):
        if Path(path) == hidden:
            raise PermissionError("synthetic unreadable source directory")
        return original(path)

    monkeypatch.setattr(os, "scandir", denied)
    with pytest.raises(ValueError, match="Unreadable source"):
        validate_recipe_kuberay_compatibility(cluster(KubeRaySpec(True)), recipe / "k8s-training")


@pytest.mark.parametrize("relative", [
    "k8s-training/terraform.tfstate.d", "k8s-training/.terraform/terraform.tfstate",
    "k8s-training/terraform.tfstate", "k8s-training/terraform.tfvars",
    "k8s-training/unrecorded-empty", "modules/kuberay/unrecorded-empty",
])
def test_unrecorded_source_directories_fail_before_provider_and_replacement(recipe, tmp_path, monkeypatch, relative):
    desired, destination, deploy = _isolated_deploy(recipe, tmp_path, monkeypatch)
    (recipe / relative).mkdir(parents=True)
    work = destination / "k8s-training"
    cache = work / ".terraform/modules"
    cache.mkdir(parents=True)
    (cache / "modules.json").write_text("retained mapping")
    (work / "terraform.tfvars").write_text("retained variables")
    (work / "terraform.tfstate").write_text('{"serial":123}')
    with pytest.raises(ValueError, match="Unexpected source directories"):
        deploy()
    with pytest.raises(ValueError, match="Unexpected source directories"):
        E._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                               cluster=desired, ssh_public_key="ssh-test")
    assert (cache / "modules.json").read_text() == "retained mapping"
    assert (work / "terraform.tfvars").read_text() == "retained variables"
    assert (work / "terraform.tfstate").read_text() == '{"serial":123}'


@pytest.mark.parametrize("region", ["eu-north1", "us-central1"])
def test_materialization_validates_pristine_recipe_and_preserves_owned_state(recipe, tmp_path, region):
    destination = tmp_path / "installation"
    state = destination / "k8s-training/terraform.tfstate"
    state.parent.mkdir(parents=True)
    state.write_text('{"serial":123}')
    work = L._prepare_install_dir(destination, recipe_root=recipe, region=region,
                                  cluster=cluster(KubeRaySpec(True)), ssh_public_key="ssh-test")
    assert state.read_text() == '{"serial":123}'
    assert "enable_kuberay_cluster   = true" in (work / "terraform.tfvars").read_text()
    domain = "api.eu.nebius.cloud:443" if region.startswith("eu") else "api.nebius.cloud:443"
    assert domain in (work / "provider.tf").read_text()
    validate_recipe_kuberay_compatibility(cluster(KubeRaySpec(True)), recipe / "k8s-training")


def _isolated_deploy(recipe, tmp_path, monkeypatch):
    desired = cluster(KubeRaySpec(True))
    project = ProjectSpec(project_id="project-test", clusters=[desired])
    declaration = FleetSpec(name="ray", tenant_id="tenant-test", region="us-central1",
                            ssh_public_key="ssh-test", projects=[project])
    monkeypatch.setattr(L, "_require_bin", lambda name: name)
    monkeypatch.setattr(L, "_assert_terraform_version", lambda _: "1.13.3")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *a, **kw: recipe)
    monkeypatch.setattr(L, "resolve_project_id", lambda *a, **kw: pytest.fail("project mutation"))
    monkeypatch.setattr(L, "preflight_region", lambda *a, **kw: pytest.fail("quota preflight"))
    root = tmp_path / "state"
    destination = root / declaration.name / project.key() / desired.name
    return desired, destination, lambda: L.deploy_fleet(declaration, work_root=root)


@pytest.mark.parametrize("name,content", [
    ("terraform.tfstate_override.tf", 'module "kuberay" { cpu_cluster = null }'),
    ("terraform.tfstate.auto.tfvars", "enable_kuberay_cluster = false"),
    ("terraform.tfstate.auto.tfvars.json", '{"kuberay_cpu_cluster":null}'),
    ("override.tf", 'module "kuberay" { cpu_cluster = null }'),
    ("extra_override.tf.json", '{"module":{"kuberay":{"cpu_cluster":null}}}'),
    ("extra.auto.tfvars", "kuberay_cpu_cluster = null"),
    ("terraform.tfvars.json", '{"kuberay_cpu_cluster":null}'),
])
def test_retained_inputs_fail_before_cloud_and_preserve_state(recipe, tmp_path, monkeypatch, name, content):
    desired, destination, deploy = _isolated_deploy(recipe, tmp_path, monkeypatch)
    work = destination / "k8s-training"
    work.mkdir(parents=True)
    state = work / "terraform.tfstate"
    state.write_text('{"serial":123}')
    extra = work / name
    extra.write_text(content)
    with pytest.raises(ValueError, match="KubeRay"):
        deploy()
    with pytest.raises(ValueError, match="KubeRay"):
        L._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                               cluster=desired, ssh_public_key="ssh-test")
    assert state.read_text() == '{"serial":123}'
    assert extra.read_text() == content
    assert not (work / "main.tf").exists()


@pytest.mark.parametrize("key,value", [
    ("TF_CLI_ARGS", "-var-file=unreviewed.tfvars"),
    ("TF_CLI_ARGS_apply", "-var=kuberay_cpu_cluster=null"),
    ("TF_CLI_ARGS_plan", "-var=enable_kuberay_cluster=false"),
    ("TF_CLI_ARGS_init", "-backend-config=unreviewed.hcl"),
    ("TF_WORKSPACE", "unreviewed"),
    ("TF_WORKSPACE", " default "),
    ("TF_DATA_DIR", "unreviewed-data"),
    ("TF_DATA_DIR", " "),
])
def test_ambient_overrides_fail_before_provisioning(recipe, tmp_path, monkeypatch, key, value):
    _, destination, deploy = _isolated_deploy(recipe, tmp_path, monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="KubeRay"):
        deploy()
    assert not destination.exists()


def test_backend_reapply_rejects_retained_override_before_unchanged_target_shortcut(recipe, tmp_path, monkeypatch):
    desired = as_mk8s_desired(cluster(KubeRaySpec(True)))
    project = E.MK8sProjectIdentity(project_key="example")
    root = tmp_path / "state"
    work = root / project.key() / desired.name / "k8s-training"
    work.mkdir(parents=True)
    (work / "terraform.tfstate_override.tf").write_text('module "kuberay" { cpu_cluster = null }')
    monkeypatch.setattr(E, "is_verified_unchanged_target", lambda **kw: pytest.fail("unchanged-target shortcut"))
    request = MK8sApplyRequest(
        recipe_root=recipe, fleet_root=root, project=project,
        scope=E.MK8sExecutionScope(fleet_name="ray"), provider_preflight=True,
        nebius_bin="nebius", tenant_id="tenant-test", region="us-central1", provider_env={},
    )
    with pytest.raises(ValueError, match="KubeRay"):
        MK8sBackend().preflight(desired, request)


def test_actual_execution_environment_is_checked_before_state_or_terraform(recipe, tmp_path, monkeypatch):
    desired = as_mk8s_desired(cluster(KubeRaySpec(True)))
    monkeypatch.setattr(E, "_cluster_tf_env", lambda *a, **kw: {"TF_CLI_ARGS_apply": "-var=kuberay_cpu_cluster=null"})
    monkeypatch.setattr(E, "_write_env_sidecar", lambda *a, **kw: pytest.fail("sidecar mutation"))
    monkeypatch.setattr(E, "_tf_run", lambda *a, **kw: pytest.fail("Terraform mutation"))
    result = E._deploy_one_cluster(
        spec=E.MK8sExecutionScope(fleet_name="ray"),
        project=E.MK8sProjectIdentity(project_key="example"), cluster=desired,
        project_id="project-test", project_created=False, subnet_id="subnet-test",
        region="us-central1", tenant_id="tenant-test", ssh_public_key="ssh-test",
        fleet_root=tmp_path / "state", recipe_root=recipe, terraform_bin="terraform",
        nebius_bin="nebius", timeout_minutes=120, on_status=None,
    )
    assert result["status"] == "error"
    assert "TF_CLI_ARGS" in result["error"]


@pytest.mark.parametrize("policy", [None, KubeRaySpec(), KubeRaySpec(True)])
def test_deploy_binds_current_materialization_before_first_terraform_command(recipe, tmp_path, monkeypatch, policy):
    desired = as_mk8s_desired(cluster(policy))
    project = E.MK8sProjectIdentity(project_key="example")
    root = tmp_path / "state"
    install = root / project.key() / desired.name
    if not policy or not policy.enabled:
        install.mkdir(parents=True)
        E._write_env_sidecar(install, {"kuberay_managed": True, "kuberay_materialized_sha256": "old"})
    monkeypatch.setattr(E, "_cluster_tf_env", lambda *a, **kw: dict(os.environ))
    observed = []

    def first_command(*a, **kw):
        saved = E._load_env_sidecar(install)
        assert saved["kuberay_managed"] is True
        assert saved["kuberay_materialized_sha256"] == kuberay_materialized_digest(install)
        observed.append(saved)
        raise RuntimeError("synthetic initialization failure")

    monkeypatch.setattr(E, "_tf_run", first_command)
    result = E._deploy_one_cluster(
        spec=E.MK8sExecutionScope(fleet_name="ray"), project=project, cluster=desired,
        project_id="project-test", project_created=False, subnet_id="subnet-test",
        region="us-central1", tenant_id="tenant-test", ssh_public_key="ssh-test",
        fleet_root=root, recipe_root=recipe, terraform_bin="terraform",
        nebius_bin="nebius", timeout_minutes=120, on_status=None,
    )
    assert result["status"] == "error"
    assert result["error"] == "synthetic initialization failure"
    assert len(observed) == 1
    assert E._load_env_sidecar(install)["kuberay_managed"] is True


@pytest.mark.parametrize("relative", [
    "k8s-training", "modules", "k8s-training/filesystem-csi-validation",
    "k8s-training/terraform.tfstate", "k8s-training/terraform.tfstate.backup",
    "k8s-training/.terraform", "k8s-training/.terraform/environment",
    "k8s-training/.terraform/modules",
    "k8s-training/.terraform/terraform.tfstate",
])
def test_destination_symlinks_fail_before_copy_even_when_dangling(recipe, tmp_path, relative):
    destination = tmp_path / "installation"
    path = destination / relative
    path.parent.mkdir(parents=True)
    path.symlink_to(tmp_path / "absent-target")
    with pytest.raises(ValueError, match="KubeRay"):
        L._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                               cluster=cluster(KubeRaySpec(True)), ssh_public_key="ssh-test")
    assert path.is_symlink()
    assert not (tmp_path / "absent-target").exists()


@pytest.mark.parametrize("name,kind", [
    ("terraform.tfstate", "directory"), ("terraform.tfstate.backup", "directory"),
    ("terraform.tfstate.d", "directory"), (".terraform", "file"),
    (".terraform/environment", "directory"),
    (".terraform/modules", "file"),
])
def test_invalid_retained_state_types_are_rejected(recipe, tmp_path, name, kind):
    destination = tmp_path / "installation"
    path = destination / "k8s-training" / name
    path.parent.mkdir(parents=True)
    path.mkdir() if kind == "directory" else path.write_text("invalid")
    with pytest.raises(ValueError, match="KubeRay"):
        L._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                               cluster=cluster(KubeRaySpec(True)), ssh_public_key="ssh-test")
    assert path.exists()


def test_default_workspace_and_provider_cache_links_remain_supported(recipe, tmp_path, monkeypatch):
    destination = tmp_path / "installation"
    work = destination / "k8s-training"
    data = work / ".terraform"
    (data / "providers").mkdir(parents=True)
    cache = tmp_path / "provider-cache"
    cache.mkdir()
    (data / "providers/cached").symlink_to(cache, target_is_directory=True)
    (data / "environment").write_text("default")
    (work / "terraform.tfstate.backup").write_text('{"serial":122}')
    monkeypatch.setenv("TF_WORKSPACE", "default")
    L._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                           cluster=cluster(KubeRaySpec(True)), ssh_public_key="ssh-test")
    assert (work / "terraform.tfstate.backup").read_text() == '{"serial":122}'
    assert (data / "providers/cached").is_symlink()
    (data / "environment").write_text("unreviewed")
    with pytest.raises(ValueError, match="default retained Terraform workspace"):
        L._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                               cluster=cluster(KubeRaySpec(True)), ssh_public_key="ssh-test")


def test_disabled_materialization_retains_existing_behavior(recipe, tmp_path, monkeypatch):
    destination = tmp_path / "installation"
    extra = destination / "k8s-training/terraform.tfstate_override.tf"
    extra.parent.mkdir(parents=True)
    extra.write_text("# existing operator override")
    monkeypatch.setenv("TF_CLI_ARGS_apply", "-var-file=operator.tfvars")
    monkeypatch.setenv("TF_WORKSPACE", "operator")
    monkeypatch.setenv("TF_DATA_DIR", "operator-data")
    work = L._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                                  cluster=cluster(), ssh_public_key="ssh-test")
    assert extra.read_text() == "# existing operator override"
    assert "enable_kuberay_cluster   = false" in (work / "terraform.tfvars").read_text()


def test_reapply_rebuilds_module_manifest_without_touching_redirected_directory(recipe, tmp_path):
    destination = tmp_path / "installation"
    work = destination / "k8s-training"
    cache = work / ".terraform/modules"
    cache.mkdir(parents=True)
    external = tmp_path / "unreviewed-module"
    external.mkdir()
    source = external / "main.tf"
    source.write_text('output "unexpected" { value = true }')
    (cache / "modules.json").write_text(json.dumps({"Modules": [{
        "Key": "kuberay", "Source": "../modules/kuberay", "Dir": str(external),
    }]}))
    (cache / "external-link").symlink_to(external, target_is_directory=True)
    L._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                           cluster=cluster(KubeRaySpec(True)), ssh_public_key="ssh-test")
    assert not cache.exists()
    assert source.read_text() == 'output "unexpected" { value = true }'
    assert (destination / "modules/kuberay/main.tf").read_bytes() == (recipe / "modules/kuberay/main.tf").read_bytes()


@pytest.mark.parametrize("backend", [{"type": "cloud"}, {"type": "local", "config": {"path": "outside.tfstate"}}, {}])
def test_retained_backend_metadata_is_preserved_and_rejected(recipe, tmp_path, backend):
    destination = tmp_path / "installation"
    metadata = destination / "k8s-training/.terraform/terraform.tfstate"
    metadata.parent.mkdir(parents=True)
    content = json.dumps({"backend": backend})
    metadata.write_text(content)
    with pytest.raises(ValueError, match="implicit local Terraform state"):
        L._prepare_install_dir(destination, recipe_root=recipe, region="us-central1",
                               cluster=cluster(KubeRaySpec(True)), ssh_public_key="ssh-test")
    assert metadata.read_text() == content


@pytest.fixture
def live_harness():
    path = Path(__file__).parents[1] / "e2e/test_fleet_kuberay_live.py"
    definition = importlib.util.spec_from_file_location("kuberay_live_harness", path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def _destroy_fixture(tmp_path, monkeypatch, policy=None):
    desired = as_mk8s_desired(cluster(policy))
    project = E.MK8sProjectIdentity(project_key="example", project_id="project-test")
    root = tmp_path / "state"
    install = root / project.key() / desired.name
    work = install / "k8s-training"
    work.mkdir(parents=True)
    (install / "modules").mkdir()
    (work / "main.tf").write_text('resource "terraform_data" "owned" { input = "owned synthetic value" }')
    (work / "terraform.tfvars").write_text("")
    saved = {
        "backend": "mk8s", "cluster_name": desired.name, "project_id": "project-test",
        "tenant_id": "tenant-test", "region": "us-central1", "context": "test-ray-context",
        "cluster_id": "cluster-test", "kuberay_managed": True,
        "kuberay_materialized_sha256": kuberay_materialized_digest(install),
    }
    E._write_env_sidecar(install, saved)
    monkeypatch.setattr(E, "_cluster_tf_env", lambda *a, **kw: dict(os.environ))
    monkeypatch.setattr(E, "_run_capture", lambda *a, **kw: pytest.fail("provider fallback must not run"))
    kwargs = dict(
        spec=E.MK8sExecutionScope(fleet_name="ray", tenant_id="tenant-test", region="us-central1"),
        project=project, cluster=desired, fleet_root=root, terraform_bin="terraform",
        nebius_bin="nebius", timeout_minutes=120, on_status=None,
    )
    return install, work, kwargs


@pytest.mark.parametrize("policy", [None, KubeRaySpec()])
def test_legacy_apply_cannot_bypass_historical_opt_in(tmp_path, monkeypatch, policy):
    _, work, kwargs = _destroy_fixture(tmp_path, monkeypatch, policy)
    request = MK8sApplyRequest(
        terraform_command=("terraform", "apply"), terraform_cwd=work, terraform_env={},
        command_runner=lambda *a, **kw: pytest.fail("legacy Terraform must not run"),
    )
    with pytest.raises(ValueError, match="requires native mk8s"):
        MK8sBackend().apply(kwargs["cluster"], request)


@pytest.mark.parametrize("policy", [None, KubeRaySpec(), KubeRaySpec(True)])
@pytest.mark.parametrize("actual_env", [False, True])
def test_destroy_refuses_overrides_even_when_desired_policy_omitted(tmp_path, monkeypatch, policy, actual_env):
    install, work, kwargs = _destroy_fixture(tmp_path, monkeypatch, policy)
    (work / "terraform.tfstate").write_text('{"version":4,"resources":[]}')
    if actual_env:
        monkeypatch.setattr(E, "_cluster_tf_env", lambda *a, **kw: {"TF_CLI_ARGS_destroy": "-target=terraform_data.missing"})
    else:
        monkeypatch.setenv("TF_CLI_ARGS_destroy", "-target=terraform_data.missing")
    monkeypatch.setattr(E, "_tf_run", lambda *a, **kw: pytest.fail("Terraform must not run"))
    before = (install / E._ENV_SIDECAR).read_bytes()
    result = E._destroy_one_cluster(**kwargs)
    assert result["status"] == "destroy-incomplete"
    assert "TF_CLI_ARGS" in result["errors"][0]
    assert (install / E._ENV_SIDECAR).read_bytes() == before
    assert (work / "terraform.tfstate").exists()


@pytest.mark.parametrize("mutation", ["main.tf", "terraform.tfvars", "modules/extra.tf", "missing-digest", "invalid-marker"])
def test_destroy_refuses_unbound_materialization_without_fallback(tmp_path, monkeypatch, mutation):
    install, work, kwargs = _destroy_fixture(tmp_path, monkeypatch)
    if mutation in ("missing-digest", "invalid-marker"):
        saved = E._load_env_sidecar(install)
        if mutation == "missing-digest":
            saved.pop("kuberay_materialized_sha256")
        else:
            saved["kuberay_managed"] = "true"
        E._write_json_file(install / E._ENV_SIDECAR, saved)
    else:
        target = install / mutation if mutation.startswith("modules/") else work / mutation
        target.write_text("changed effective input")
    monkeypatch.setattr(E, "_tf_run", lambda *a, **kw: pytest.fail("Terraform must not run"))
    result = E._destroy_one_cluster(**kwargs)
    assert result["status"] == "destroy-incomplete"
    assert install.exists()


@pytest.mark.parametrize("policy", [None, KubeRaySpec()])
def test_disabled_reapply_and_sidecar_rewrites_preserve_historical_protection(recipe, tmp_path, monkeypatch, policy):
    install, work, _ = _destroy_fixture(tmp_path, monkeypatch, policy)
    (work / ".terraform.tfstate.lock.info").write_text("retained lock")
    E._prepare_install_dir(install, recipe_root=recipe, region="us-central1",
                           cluster=as_mk8s_desired(cluster(policy)), ssh_public_key="ssh-test")
    assert "enable_kuberay_cluster   = false" in (work / "terraform.tfvars").read_text()
    assert (work / ".terraform.tfstate.lock.info").read_text() == "retained lock"
    E._write_env_sidecar(install, {"status": "provisioning"})
    assert E._load_env_sidecar(install)["kuberay_managed"] is True
    monkeypatch.setenv("TF_CLI_ARGS_apply", "-var-file=override.tfvars")
    with pytest.raises(ValueError, match="TF_CLI_ARGS"):
        E._prepare_install_dir(install, recipe_root=recipe, region="us-central1",
                               cluster=as_mk8s_desired(cluster(policy)), ssh_public_key="ssh-test")


@pytest.mark.parametrize("payload", [None, "{", "[]", '{}', '{"version":4,"resources":{}}',
                                     '{"version":4,"resources":[{}]}',
                                     '{"version":4,"resources":[{"mode":"managed","instances":[]}]}'])
def test_destroy_state_postcondition_fails_closed(tmp_path, payload):
    if payload is not None:
        (tmp_path / "terraform.tfstate").write_text(payload)
    with pytest.raises((OSError, ValueError)):
        validate_kuberay_destroyed_state(tmp_path)


def test_materialized_digest_excludes_only_exact_runtime_paths(tmp_path, monkeypatch):
    install, work, _ = _destroy_fixture(tmp_path, monkeypatch)
    initial = kuberay_materialized_digest(install)
    for name in ("terraform.tfstate", "terraform.tfstate.backup", ".terraform.lock.hcl", ".terraform.tfstate.lock.info"):
        (work / name).write_text("runtime")
    (work / ".terraform/providers").mkdir(parents=True)
    (work / ".terraform/providers/cache").symlink_to(tmp_path / "external-cache")
    receipts = work / "filesystem-csi-validation/.state"
    receipts.mkdir(parents=True)
    (receipts / "receipt").write_text("runtime")
    assert kuberay_materialized_digest(install) == initial
    (install / "modules/terraform.tfstate").write_text("nested source")
    assert kuberay_materialized_digest(install) != initial


@pytest.mark.parametrize("partial", [False, True])
def test_native_destroy_preserves_recovery_until_state_is_empty(tmp_path, monkeypatch, partial):
    terraform = shutil.which("terraform")
    if terraform is None:
        pytest.skip("terraform required for native teardown regression")
    install, work, kwargs = _destroy_fixture(tmp_path, monkeypatch)
    for command in ([terraform, "init", "-input=false"], [terraform, "apply", "-auto-approve", "-input=false"]):
        run = subprocess.run(command, cwd=work, text=True, capture_output=True)
        assert run.returncode == 0, run.stderr + run.stdout
    from npa.cluster import state as S
    monkeypatch.setattr(S, "CLUSTERS_DIR", tmp_path / "identities")
    S.save_cluster_state(S.ClusterState(
        name="test-ray-context", cluster_id="cluster-test", project_id="project-test",
        region="us-central1", node_count=1, node_platform="cpu-d3", node_preset="16vcpu-64gb",
        k8s_version="1.34", subnet_id="subnet-test", created_at="2026-01-01T00:00:00Z",
    ))
    receipt = tmp_path / "native-destroy-receipt.json"
    wrapper = tmp_path / "terraform-probe"
    # Fault injection: an otherwise successful command can leave managed state.
    # All product validation, native Terraform execution and recovery I/O stay real.
    wrapper.write_text(
        f"#!{sys.executable}\nimport json,subprocess,sys\nfrom pathlib import Path\n"
        f"args=[{terraform!r},*sys.argv[1:]]\n"
        f"if sys.argv[1]=='destroy' and {partial!r}: args.append('-target=terraform_data.missing')\n"
        "result=subprocess.run(args)\n"
        "if sys.argv[1]=='destroy':\n"
        f" Path({str(receipt)!r}).write_text(json.dumps({{'exit_code':result.returncode,'state':json.loads(Path('terraform.tfstate').read_text())}}))\n"
        "raise SystemExit(result.returncode)\n"
    )
    wrapper.chmod(0o700)
    kwargs["terraform_bin"] = str(wrapper)
    result = E._destroy_one_cluster(**kwargs)
    proof = json.loads(receipt.read_text())
    assert proof["exit_code"] == 0
    remaining = [r for r in proof["state"]["resources"] if r["mode"] == "managed"]
    assert bool(remaining) is partial
    assert result["status"] == ("destroy-incomplete" if partial else "destroyed")
    assert install.exists() is partial
    assert (S.load_cluster_state("test-ray-context") is not None) is partial


def test_live_harness_records_real_command_failure_privately(live_harness, tmp_path):
    with pytest.raises(AssertionError, match="probe failed"):
        live_harness._record_command(tmp_path, "probe", [sys.executable, "-c", "print('retained failure'); raise SystemExit(7)"])
    receipt = tmp_path / "probe.json"
    assert receipt.stat().st_mode & 0o777 == 0o600
    data = json.loads(receipt.read_text())
    assert data["exit_code"] == 7
    assert data["stdout"] == "retained failure\n"


def test_live_harness_cleanup_survives_stop_failure_and_keeps_primary(live_harness, tmp_path):
    calls = []

    def run(name, args):
        calls.append(name)
        if name == "stop":
            raise RuntimeError("job did not register")

    with pytest.raises(RuntimeError, match="Native proof and cleanup failures") as failure:
        with live_harness._job_cleanup(run, [], str(tmp_path / "owned-test"), "owned-test"):
            raise ValueError("primary staging failure")
    assert calls == ["stop", "remove-source"]
    assert [str(error) for error in failure.value.args[1]] == ["primary staging failure", "job did not register"]
    assert isinstance(failure.value.__cause__, ValueError)


def test_live_harness_allocates_distinct_private_worker_directories(live_harness, tmp_path, monkeypatch):
    """Execute the remote allocator locally to verify exclusive private staging."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    def run(name, arguments):
        assert name == "create-source-directory"
        result = subprocess.run([sys.executable, *arguments[1:]], check=True, capture_output=True, text=True)
        return result.stdout

    directories = []
    try:
        for _ in range(2):
            directory = Path(live_harness._create_worker_source_directory(run, []))
            directories.append(directory)
            assert directory.parent == tmp_path
            assert directory.name.startswith("npa-cpu-proof-")
            assert directory.stat().st_mode & 0o777 == 0o700
        assert directories[0] != directories[1]
    finally:
        for directory in directories:
            shutil.rmtree(directory)


def test_missing_recipe_cannot_pass_backend_preflight():
    with pytest.raises(ValueError, match="selected recipe"):
        MK8sBackend().preflight(as_mk8s_desired(cluster(KubeRaySpec(True))), MK8sApplyRequest())


def test_legacy_execution_cannot_silently_ignore_kuberay(tmp_path):
    request = MK8sApplyRequest(
        terraform_command=("terraform", "apply"),
        terraform_cwd=tmp_path,
        terraform_env={},
        command_runner=lambda *args, **kwargs: pytest.fail("legacy apply executed"),
    )
    with pytest.raises(ValueError, match="legacy standalone Terraform"):
        MK8sBackend().apply(as_mk8s_desired(cluster(KubeRaySpec(True))), request)


def test_real_terraform_template_decodes_requested_resource_values(recipe, tmp_path):
    terraform = shutil.which("terraform")
    if terraform is None:
        pytest.skip("terraform required for native template evaluation")
    template = recipe / "modules/kuberay/files/ray-values-cpu.yaml.tftpl"
    work = tmp_path / "evaluation"
    work.mkdir()
    (work / "main.tf").write_text(
        'output "values" {\n value = templatefile(' + json.dumps(str(template)) + ', {\n'
        'cpu_platform = "cpu-d3"\nworker_replicas = 3\nworker_cpus = 4\nworker_memory_gib = 8\n})\n}\n'
    )
    result = subprocess.run([terraform, "apply", "-auto-approve", "-no-color"], cwd=work, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = subprocess.run([terraform, "output", "-json", "values"], cwd=work, capture_output=True, text=True, check=True)
    values = yaml.safe_load(json.loads(output.stdout))
    assert values["kube-prometheus-stack"]["enabled"] is False
    assert "configure" not in values["configuration"]
    ray = yaml.safe_load(values["configuration"]["putspec"]["yamlInput"])["spec"]
    assert ray["rayVersion"] == RAY_VERSION
    assert ray["enableInTreeAutoscaling"] is False
    head = ray["headGroupSpec"]
    assert head["serviceType"] == "ClusterIP"
    assert head["rayStartParams"]["num-cpus"] == "0"
    assert len(ray["workerGroupSpecs"]) == 1
    worker = ray["workerGroupSpecs"][0]
    assert [worker[key] for key in ("replicas", "minReplicas", "maxReplicas")] == [3, 3, 3]
    assert worker["rayStartParams"] == {"num-cpus": "4"}
    for group, expected in [(head, {"cpu": "1", "memory": "4Gi"}), (worker, {"cpu": "4", "memory": "8Gi"})]:
        pod = group["template"]["spec"]
        assert pod["nodeSelector"] == {"node.kubernetes.io/instance-type": "cpu-d3"}
        assert pod["automountServiceAccountToken"] is False
        assert pod["securityContext"]["runAsNonRoot"] is True
        container = pod["containers"][0]
        assert container["image"] == RAY_IMAGE
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
        assert container["resources"] == {"requests": expected, "limits": expected}
