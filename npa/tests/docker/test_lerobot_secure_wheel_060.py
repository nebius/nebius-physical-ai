"""Regression coverage for the reviewed 0.6 W&B security integration."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa/docker/workbench/lerobot/prepare_secure_wheel_060.py"


def _module():
    spec = importlib.util.spec_from_file_location("secure_lerobot_060", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_rewrite_is_the_reviewed_secure_dependency_closure():
    module = _module()
    expected = {
        "torch<2.12.0,>=2.7": "torch<2.14.0,>=2.13.0",
        "torchvision<0.27.0,>=0.22.0": "torchvision<0.29.0,>=0.28.0",
        "torchcodec<0.12.0,>=0.3.0": "torchcodec<0.17.0,>=0.16.0",
        "torchcodec<0.12.0,>=0.7.0": "torchcodec<0.17.0,>=0.16.0",
        "torchcodec<0.12.0,>=0.11.0": "torchcodec<0.17.0,>=0.16.0",
        "diffusers<0.36.0,>=0.27.2": "diffusers<0.39.0,>=0.38.0",
        "Pillow<13.0.0,>=10.0.0": "Pillow<14.0.0,>=13.0.0",
        "setuptools<81.0.0,>=71.0.0": "setuptools<85.0.0,>=84.1.0",
        "transformers<5.6.0,>=5.4.0": "transformers<5.11.0,>=5.10.0",
        "wandb<0.28.0,>=0.24.0": "wandb<0.31.0,>=0.30.0",
        "datasets<5.0.0,>=4.7.0": "datasets<5.1.0,>=5.0.1",
    }
    assert module.DEPENDENCIES == expected
    markers = {
        "torch<2.12.0,>=2.7": "",
        "torchvision<0.27.0,>=0.22.0": "",
        "torchcodec<0.12.0,>=0.3.0": '; ((sys_platform == "linux" and (platform_machine == "x86_64" or platform_machine == "AMD64")) or (sys_platform == "darwin" and platform_machine == "arm64")) and extra == "dataset"',
        "torchcodec<0.12.0,>=0.7.0": '; sys_platform == "win32" and extra == "dataset"',
        "torchcodec<0.12.0,>=0.11.0": '; (sys_platform == "linux" and (platform_machine == "aarch64" or platform_machine == "arm64")) and extra == "dataset"',
        "diffusers<0.36.0,>=0.27.2": '; extra == "diffusers-dep"',
        "Pillow<13.0.0,>=10.0.0": "",
        "setuptools<81.0.0,>=71.0.0": "",
        "transformers<5.6.0,>=5.4.0": '; extra == "transformers-dep"',
        "wandb<0.28.0,>=0.24.0": '; extra == "training"',
        "datasets<5.0.0,>=4.7.0": '; extra == "dataset"',
    }
    upstream = b"Metadata-Version: 2.4\nVersion: 0.6.0\n" + b"".join(
        f"Requires-Dist: {dependency}{markers[dependency]}\n".encode()
        for dependency in expected
    )
    rewritten = module._metadata(upstream)
    assert b"Version: 0.6.0+npa.secure4\n" in rewritten
    for dependency, replacement in expected.items():
        assert (
            f"Requires-Dist: {replacement}{markers[dependency]}\n".encode() in rewritten
        )
        assert f"Requires-Dist: {dependency};".encode() not in rewritten


def test_metadata_rewrite_refuses_an_unreviewed_upstream_edge():
    module = _module()
    with pytest.raises(RuntimeError, match="reviewed dependency edges changed"):
        module._metadata(b"Version: 0.6.0\n")


def test_logger_rewrite_rejects_the_removed_wandb_api_at_the_real_call_site(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()
    upstream = b"logger = wandb.run.get_url()\n"
    monkeypatch.setattr(module, "LOGGER_SHA256", hashlib.sha256(upstream).hexdigest())
    rewritten = module._logger_source(upstream)
    assert b"wandb.run.get_url()" not in rewritten
    assert b"wandb.run.url" in rewritten
    assert rewritten.startswith(module.LOGGER_MODIFICATION_NOTICE)


def test_native_source_claim_is_derived_from_the_patch_paths() -> None:
    module = _module()
    assert module.LOGGER_PATH == "lerobot/common/wandb_utils.py"
    assert module._native_model_source_unchanged({module.LOGGER_PATH: {}})
    assert not module._native_model_source_unchanged(
        {"lerobot/policies/act/modeling_act.py": {}}
    )
