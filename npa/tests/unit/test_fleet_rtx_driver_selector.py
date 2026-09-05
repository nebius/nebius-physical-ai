"""Exact rendering selectors must survive recipe validation and Helm updates."""

import shutil

import pytest

from npa.cluster.gpu_driver import GpuDriverStrategyError
from npa.cluster_backends.mk8s_render import (
    render_tfvars,
    validate_recipe_rtx_compatibility,
)
from npa.fleet import lifecycle as L
from npa.fleet.spec import ClusterSpec, FleetSpec, NodePoolSpec, ProjectSpec


def rendering(platform="gpu-rtx6000-a", preset="8gpu-192vcpu-1744gb"):
    return ClusterSpec(
        name="render", gpu_workload_profile="rtx-rendering",
        gpu_nodes=NodePoolSpec(count=1, platform=platform, preset=preset),
    )


@pytest.fixture
def recipe(tmp_path):
    shipped = L._find_vendored_recipe_root()
    assert shipped is not None
    root = tmp_path / "recipe"
    (root / "k8s-training").mkdir(parents=True)
    for name in ("variables.tf", "helm.tf"):
        shutil.copy2(shipped / "k8s-training" / name, root / "k8s-training" / name)
    shutil.copytree(shipped / "modules" / "gpu-operator", root / "modules" / "gpu-operator")
    return root


@pytest.mark.parametrize("platform", ["gpu-rtx6000", "gpu-rtx6000-a"])
@pytest.mark.parametrize("preset", ["1gpu-24vcpu-218gb", "8gpu-192vcpu-1744gb"])
def test_rendering_preserves_exact_operator_selector(recipe, platform, preset):
    output = render_tfvars(rendering(platform, preset), recipe_dir=recipe / "k8s-training")
    assert 'gpu_operator_rtx_driver_profile = {"platform": "' + platform in output
    assert '"preset": "' + preset + '"}' in output
    assert "custom_driver                = false" in output


@pytest.mark.parametrize("file,old", [
    ("k8s-training/variables.tf", 'variable "gpu_operator_rtx_driver_profile"'),
    ("k8s-training/helm.tf", "var.gpu_operator_rtx_driver_profile"),
    ("modules/gpu-operator/variables.tf", 'variable "rtx_driver_profile"'),
    ("modules/gpu-operator/main.tf", "local.rtx_driver_values"),
])
def test_unsupported_recipe_fails_before_project_mutation(recipe, file, old, monkeypatch):
    path = recipe / file
    path.write_text(path.read_text().replace(old, "unsupported"))
    spec = FleetSpec(
        name="rtx", tenant_id="tenant-test", region="uk-south2", ssh_public_key="ssh-test",
        projects=[ProjectSpec(project_id="project-test", clusters=[rendering()])],
    )
    monkeypatch.setattr(L, "_require_bin", lambda name: name)
    monkeypatch.setattr(L, "_assert_terraform_version", lambda _: "1.13.3")
    monkeypatch.setattr(L, "_resolve_recipe_root", lambda *args, **kwargs: recipe)
    monkeypatch.setattr(L, "resolve_project_id", lambda *args, **kwargs: pytest.fail("mutated project"))
    with pytest.raises(GpuDriverStrategyError, match="exact RTX rendering driver selector"):
        L.deploy_fleet(spec, work_root=recipe.parent / "work", preflight=False)


def test_unprofiled_recipe_does_not_require_or_select_rtx_override(tmp_path):
    cluster = ClusterSpec(name="cpu", cpu_nodes=NodePoolSpec(count=1))
    validate_recipe_rtx_compatibility(cluster, tmp_path)
    assert "gpu_operator_rtx_driver_profile" not in render_tfvars(cluster)
