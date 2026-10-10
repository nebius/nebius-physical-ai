"""Hermetic contracts for the opt-in standalone RTX graphics harness."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


LIVE_HARNESS = Path(__file__).parents[1] / "e2e" / "test_mk8s_gpu_health_live.py"
IMMUTABLE_IMAGE = "registry.example/graphics@sha256:" + "a" * 64


def _harness() -> dict[str, object]:
    return runpy.run_path(str(LIVE_HARNESS))


def _configure_rtx_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")
    monkeypatch.setenv("NPA_E2E_MK8S_RTX_RENDERING", "1")
    monkeypatch.setenv("NPA_E2E_MK8S_GPU_KUBECONFIG", str(kubeconfig))
    monkeypatch.setenv("NPA_E2E_MK8S_GPU_NODES", "1")
    monkeypatch.setenv("NPA_E2E_MK8S_GPU_PLATFORM", "gpu-rtx6000")
    monkeypatch.setenv("NPA_E2E_MK8S_GPU_PRESET", "1gpu-24vcpu-218gb")
    monkeypatch.setenv("NPA_E2E_MK8S_GPU_DRIVER_MODE", "operator")


def test_standalone_rtx_harness_forwards_immutable_graphics_image_before_health(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_rtx_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("NPA_E2E_MK8S_GRAPHICS_SMOKE_IMAGE", IMMUTABLE_IMAGE)
    test = _harness()["test_rtx_rendering_profile_passes_live_graphics_readiness_gate"]
    captured = []

    def fake_validate(*_args, **kwargs):
        captured.append(kwargs["config"])
        return {
            "status": "healthy",
            "cuda_smokes": [{}],
            "graphics_smokes": [{"vulkan_physical_devices": 1}],
        }

    monkeypatch.setitem(test.__globals__, "validate_gpu_health", fake_validate)
    test(tmp_path)

    assert [config.graphics_smoke_image for config in captured] == [IMMUTABLE_IMAGE]


def test_standalone_rtx_harness_requires_graphics_image_before_health(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_rtx_environment(monkeypatch, tmp_path)
    test = _harness()["test_rtx_rendering_profile_passes_live_graphics_readiness_gate"]

    def unexpected_health(*_args, **_kwargs):
        pytest.fail("missing operator image must not invoke live health")

    monkeypatch.setitem(test.__globals__, "validate_gpu_health", unexpected_health)
    with pytest.raises(pytest.skip.Exception, match="NPA_E2E_MK8S_GRAPHICS_SMOKE_IMAGE"):
        test(tmp_path)


@pytest.mark.parametrize("image", ["tool://sonic", "registry.example/graphics:tag"])
def test_standalone_rtx_harness_refuses_nonimmutable_image_before_health(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image: str
) -> None:
    _configure_rtx_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("NPA_E2E_MK8S_GRAPHICS_SMOKE_IMAGE", image)
    test = _harness()["test_rtx_rendering_profile_passes_live_graphics_readiness_gate"]

    def unexpected_health(*_args, **_kwargs):
        pytest.fail("nonimmutable image must be rejected before live health")

    monkeypatch.setitem(test.__globals__, "validate_gpu_health", unexpected_health)
    with pytest.raises(ValueError, match="NPA_E2E_MK8S_GRAPHICS_SMOKE_IMAGE"):
        test(tmp_path)
