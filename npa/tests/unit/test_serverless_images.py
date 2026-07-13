"""Unit tests for serverless e2e image/platform resolution helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HELPER = Path(__file__).resolve().parent.parent / "e2e" / "_serverless_images.py"
_SPEC = importlib.util.spec_from_file_location("npa_e2e_serverless_images", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def test_resolve_image_rewrites_placeholder_registry(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw")
    monkeypatch.delenv("NPA_E2E_REGISTRY", raising=False)
    assert (
        _mod.resolve_image("cr.eu-north1.nebius.cloud/your-registry-id/npa-cosmos:1.0.9")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-cosmos:1.0.9"
    )


def test_resolve_serverless_gpu_maps_legacy_l40s(monkeypatch) -> None:
    monkeypatch.delenv("NPA_E2E_SERVERLESS_GPU_TYPE", raising=False)
    assert _mod.resolve_serverless_gpu_type("gpu-l40s-d") == "gpu-h200-sxm"
    assert _mod.resolve_serverless_gpu_type("l40s") == "gpu-h200-sxm"
    monkeypatch.setenv("NPA_E2E_SERVERLESS_GPU_TYPE", "gpu-rtx6000")
    assert _mod.resolve_serverless_gpu_type("gpu-l40s-d") == "gpu-rtx6000"
