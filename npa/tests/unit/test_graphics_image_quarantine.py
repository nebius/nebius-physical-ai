"""Govern graphics probe defaults at the Fleet and mk8s mutation boundaries."""

import pytest
import yaml

from npa.cluster.gpu_health import DEFAULT_GRAPHICS_SMOKE_IMAGE
from npa.cluster_backends.mk8s import MK8sApplyRequest, MK8sBackend, desired_state
from npa.fleet import lifecycle
from npa.fleet.spec import ClusterSpec, FleetSpec, ProjectSpec, load_spec


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/graphics:operator",
        "registry.example/graphics@sha256:" + "1" * 64,
    ],
)
def test_fleet_yaml_preserves_explicit_graphics_image_through_backend(tmp_path, image):
    path = tmp_path / "fleet.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "npa.fleet/v0.0.1",
                "name": "graphics-test",
                "tenant_id": "tenant-test",
                "region": "region-test",
                "projects": [
                    {
                        "name": "team",
                        "clusters": [
                            {
                                "name": "render",
                                "gpu_workload_profile": "rtx-rendering",
                                "gpu_graphics_smoke_image": image,
                            }
                        ],
                    }
                ],
            }
        )
    )

    spec = load_spec(path)
    cluster = spec.projects[0].clusters[0]

    assert cluster.gpu_graphics_smoke is True
    assert cluster.gpu_graphics_smoke_image == image
    assert desired_state(cluster)["gpu_graphics_smoke_image"] == image


def test_fleet_default_refusal_precedes_project_network_and_backend_calls(
    monkeypatch, tmp_path
):
    spec = FleetSpec(
        name="graphics-test",
        tenant_id="tenant-test",
        region="region-test",
        projects=[
            ProjectSpec(
                name="team",
                clusters=[
                    ClusterSpec(name="render", gpu_workload_profile="rtx-rendering")
                ],
            ),
        ],
    )

    def unexpected_operation(*_args, **_kwargs):
        pytest.fail(
            "quarantine must fail before project/network/provider/backend operations"
        )

    for name in (
        "_require_bin",
        "_run_capture",
        "resolve_project_id",
        "ensure_subnet",
        "_deploy_mk8s_fleet",
    ):
        monkeypatch.setattr(lifecycle, name, unexpected_operation)

    with pytest.raises(ValueError, match="quarantined public release"):
        lifecycle.deploy_fleet(spec, work_root=tmp_path)


def test_backend_default_refusal_precedes_native_apply(monkeypatch, tmp_path):
    cluster = ClusterSpec(name="render", gpu_workload_profile="rtx-rendering")
    request = MK8sApplyRequest(
        terraform_command=("terraform", "apply"),
        terraform_cwd=tmp_path,
        terraform_env={},
        command_runner=lambda *_args, **_kwargs: pytest.fail("unsafe apply"),
    )

    with pytest.raises(ValueError, match="quarantined public release"):
        MK8sBackend().apply(cluster, request)


def test_backend_skip_validation_does_not_consume_graphics_image(monkeypatch, tmp_path):
    cluster = ClusterSpec(name="render", gpu_workload_profile="rtx-rendering")
    observed = []
    monkeypatch.setattr(
        "npa.cluster_backends.mk8s.resolve_graphics_smoke_image",
        lambda *_args: pytest.fail("skipped health must not resolve images"),
    )
    request = MK8sApplyRequest(
        terraform_command=("terraform", "apply"),
        terraform_cwd=tmp_path,
        terraform_env={},
        post_deploy_validation="skip",
        command_runner=lambda command, **_kwargs: observed.append(command),
    )

    assert MK8sBackend().apply(cluster, request)["status"] == "applied"
    assert observed == [["terraform", "apply"]]
    assert cluster.gpu_graphics_smoke_image == DEFAULT_GRAPHICS_SMOKE_IMAGE


def test_readonly_plan_retains_governed_default_without_consuming_it():
    cluster = ClusterSpec(name="render", gpu_workload_profile="rtx-rendering")

    assert desired_state(cluster)["gpu_graphics_smoke_image"] == "tool://sonic"
