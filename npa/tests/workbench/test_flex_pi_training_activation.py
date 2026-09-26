"""CPU contract checks for explicit activation recomputation policies."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import yaml

from npa.sdk.workbench.flex_pi import train
from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.service import create_app
from npa.workbench.flex_pi.training_activation import (
    activation_checkpointing_overrides,
    activation_checkpointing_receipt,
)


def _model(video=True, action=True, mixed_attention=True):
    return SimpleNamespace(
        mot=SimpleNamespace(
            mixtures={
                "video": SimpleNamespace(use_gradient_checkpointing=video),
                "action": SimpleNamespace(use_gradient_checkpointing=action),
            },
            mot_checkpoint_mixed_attn=mixed_attention,
        )
    )


@pytest.mark.parametrize("mode", ["on", "off"])
def test_resolved_policy_controls_all_three_paths_and_records_actual_flags(mode):
    resolved = dict(
        value.removeprefix("+").split("=", 1)
        for value in activation_checkpointing_overrides(mode)
    )
    cfg = {key: yaml.safe_load(value) for key, value in resolved.items()}
    model = _model(
        cfg["model.video_dit_config.use_gradient_checkpointing"],
        cfg["model.action_dit_config.use_gradient_checkpointing"],
        cfg["model.mot_checkpoint_mixed_attn"],
    )
    receipt = activation_checkpointing_receipt(cfg, model)
    assert receipt == {
        "policy": mode,
        "video": mode == "on",
        "action": mode == "on",
        "mixed_attention": mode == "on",
    }
    assert ("npa_activation_checkpointing" in cfg) is (mode == "off")


@pytest.mark.parametrize("mode", ["on", "off"])
@pytest.mark.parametrize("path", ["video", "action", "mixed_attention"])
@pytest.mark.parametrize("incorrect", [None, "true", "false", 0, 1])
def test_loaded_flag_must_be_an_actual_boolean_matching_policy(mode, path, incorrect):
    flags = dict.fromkeys(("video", "action", "mixed_attention"), mode == "on")
    flags[path] = incorrect
    with pytest.raises(FlexPiError, match="loaded model differs"):
        activation_checkpointing_receipt(
            {"npa_activation_checkpointing": mode}, _model(**flags)
        )


@pytest.mark.parametrize("mode", ["on", "off"])
@pytest.mark.parametrize("path", ["video", "action", "mixed_attention"])
def test_one_ignored_upstream_flag_is_rejected(mode, path):
    flags = dict.fromkeys(("video", "action", "mixed_attention"), mode == "on")
    flags[path] = mode != "on"
    with pytest.raises(FlexPiError, match="loaded model differs"):
        activation_checkpointing_receipt(
            {"npa_activation_checkpointing": mode}, _model(**flags)
        )


@pytest.mark.parametrize("mode", ["auto", "false", "", False])
def test_invalid_policy_fails_before_vendor_execution(mode, monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: pytest.fail("must not start a vendor process"),
    )
    with pytest.raises(FlexPiError, match="activation-checkpointing must"):
        train(output_path="s3://example-bucket/run", activation_checkpointing=mode)


@pytest.mark.parametrize("mode, status", [("on", 200), ("off", 200), ("auto", 422)])
def test_authenticated_training_service_preserves_policy(tmp_path, mode, status):
    manifest = tmp_path / "input.json"
    manifest.write_text("{}")
    service = create_app(
        token="test-token",
        output_root="s3://example-bucket/run",
        input_manifest=str(manifest),
    )
    with TestClient(service) as client:
        response = client.post(
            "/train",
            json={
                "output_path": "candidate",
                "activation_checkpointing": mode,
                "dry_run": True,
            },
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == status
    if status == 200:
        assert response.json()["execution"]["activation_checkpointing"] == mode
