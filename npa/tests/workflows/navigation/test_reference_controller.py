"""Reject changed controller bytes before invoking any native model decoder."""

import hashlib
import io
import sys
from types import SimpleNamespace

import pytest

from npa.workflows.navigation import reference_controller as controller


def test_changed_controller_is_rejected_before_native_import(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.manager_based.navigation.mdp.pre_trained_policy_action",
        None,
    )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        controller._load_verified(None, None, b"changed", "0" * 64)


def test_exact_verified_controller_is_the_file_actually_decoded(monkeypatch):
    torch = pytest.importorskip("torch")
    module = torch.jit.trace(torch.nn.Linear(3, 2), torch.ones(1, 3))
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    cfg = SimpleNamespace(policy_path=controller.CONTROLLER_URL)
    env = SimpleNamespace(cfg=SimpleNamespace(npa_reference_controller_sha256=digest))
    decoded = []

    def load(local_cfg, native_env):
        from pathlib import Path

        assert native_env is env
        assert Path(local_cfg.policy_path).read_bytes() == payload
        decoded.append(local_cfg.policy_path)
        return torch.jit.load(local_cfg.policy_path)

    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.assets",
        SimpleNamespace(read_file=lambda _: io.BytesIO(payload)),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.manager_based.navigation.mdp.pre_trained_policy_action",
        SimpleNamespace(PreTrainedPolicyAction=load),
    )
    loaded = controller.create_action(cfg, env)
    assert torch.equal(loaded(torch.ones(1, 3)), module(torch.ones(1, 3)))
    assert decoded and cfg.policy_path == controller.CONTROLLER_URL
    report = controller.controller_evidence(
        env, SimpleNamespace(reference_controller_sha256=digest)
    )
    assert report["sha256"] == digest and report["bytes"] == len(payload)
    assert report["verified_before_load"]
    with pytest.raises(ValueError, match="omitted"):
        controller.controller_evidence(
            env, SimpleNamespace(reference_controller_sha256="0" * 64)
        )


def test_reference_requires_explicit_controller_pin(recipe):
    from npa.workflows.navigation.contract import Recipe

    values = recipe.model_dump()
    values["adapter_module"] = "npa.workflows.navigation.reference"
    with pytest.raises(ValueError, match="reference_controller_sha256"):
        Recipe.model_validate(values)
    values["reference_controller_sha256"] = controller.CONTROLLER_SHA256
    assert (
        Recipe.model_validate(values).reference_controller_sha256
        == controller.CONTROLLER_SHA256
    )
