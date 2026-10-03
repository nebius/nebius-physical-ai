"""Unit tests for serverless e2e image/platform resolution helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HELPER = Path(__file__).resolve().parent.parent / "e2e" / "_serverless_images.py"
_SPEC = importlib.util.spec_from_file_location("npa_e2e_serverless_images", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def test_resolve_image_ignores_generic_registry(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.invalid/operator/private")
    monkeypatch.delenv("NPA_E2E_REGISTRY", raising=False)
    assert (
        _mod.resolve_image(
            "ghcr.io/nebius/nebius-physical-ai/"
            "npa-cosmos:cu128-torch27-sm100-1.0.9-20260803T002017Z"
        )
        == "ghcr.io/nebius/nebius-physical-ai/"
        "npa-cosmos:cu128-torch27-sm100-1.0.9-20260803T002017Z"
    )


def test_resolve_serverless_gpu_maps_legacy_l40s(monkeypatch) -> None:
    monkeypatch.delenv("NPA_E2E_SERVERLESS_GPU_TYPE", raising=False)
    assert _mod.resolve_serverless_gpu_type("gpu-l40s-d") == "gpu-rtx6000"
    assert _mod.resolve_serverless_gpu_type("l40s") == "gpu-rtx6000"
    monkeypatch.setenv("NPA_E2E_SERVERLESS_GPU_TYPE", "gpu-h200-sxm")
    assert _mod.resolve_serverless_gpu_type("gpu-l40s-d") == "gpu-h200-sxm"


def test_resolve_serverless_gpu_preset_remaps_for_rtx6000(monkeypatch) -> None:
    monkeypatch.delenv("NPA_E2E_SERVERLESS_PRESET", raising=False)
    monkeypatch.delenv("NPA_E2E_SERVERLESS_GPU_TYPE", raising=False)
    assert (
        _mod.resolve_serverless_gpu_preset("1gpu-16vcpu-200gb", platform="gpu-rtx6000")
        == "1gpu-24vcpu-218gb"
    )
    assert (
        _mod.resolve_serverless_gpu_preset("1gpu-40vcpu-160gb", platform="gpu-rtx6000")
        == "1gpu-24vcpu-218gb"
    )
    assert (
        _mod.resolve_serverless_gpu_preset("1gpu-16vcpu-200gb", platform="gpu-h200-sxm")
        == "1gpu-16vcpu-200gb"
    )
    monkeypatch.setenv("NPA_E2E_SERVERLESS_PRESET", "8gpu-192vcpu-1744gb")
    assert (
        _mod.resolve_serverless_gpu_preset("1gpu-16vcpu-200gb", platform="gpu-rtx6000")
        == "8gpu-192vcpu-1744gb"
    )
