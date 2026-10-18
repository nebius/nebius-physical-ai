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


def test_metadata_rewrite_is_exactly_the_reviewed_wandb_security_edge():
    module = _module()
    upstream = (
        b"Metadata-Version: 2.4\n"
        b"Version: 0.6.0\n"
        b'Requires-Dist: wandb<0.28.0,>=0.24.0; extra == "training"\n'
        b'Requires-Dist: datasets<5.0.0,>=4.7.0; extra == "dataset"\n'
    )
    rewritten = module._metadata(upstream)
    assert b"Version: 0.6.0+npa.secure2\n" in rewritten
    assert b'Requires-Dist: wandb<0.31.0,>=0.30.0; extra == "training"\n' in rewritten
    assert b'Requires-Dist: datasets<5.1.0,>=5.0.1; extra == "dataset"\n' in rewritten
    assert b"wandb<0.28.0,>=0.24.0" not in rewritten


def test_metadata_rewrite_refuses_an_unreviewed_upstream_edge():
    module = _module()
    with pytest.raises(RuntimeError, match="W&B dependency edge"):
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
