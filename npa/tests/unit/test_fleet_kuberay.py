"""The CPU RayCluster contract must reach actual resources, or fail before apply."""

from dataclasses import replace
import json
import shutil
import subprocess

import pytest
import yaml

from npa.cluster_backends.kuberay import (
    KubeRaySpec, RAY_IMAGE, RAY_VERSION, validate_recipe_kuberay_compatibility,
)
from npa.cluster_backends.mk8s import MK8sApplyRequest, MK8sBackend
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


def test_missing_recipe_cannot_pass_backend_preflight():
    with pytest.raises(ValueError, match="selected recipe"):
        MK8sBackend().preflight(as_mk8s_desired(cluster(KubeRaySpec(True))), MK8sApplyRequest())


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
