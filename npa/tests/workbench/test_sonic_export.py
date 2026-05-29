from __future__ import annotations

# ruff: noqa: E402

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from npa.workbench.sonic import export_onnx, load_export_metadata, validate_onnx_parity


class TinySonicPolicy(torch.nn.Module):
    observation_dim = 4
    action_dim = 2
    control_dt = 0.02
    obs_spec = {
        "name": "actor_obs",
        "shape": [4],
        "fields": [
            {"name": "base_ang_vel", "dim": 2, "units": "rad/s"},
            {"name": "joint_pos", "dim": 2, "units": "rad"},
        ],
    }
    action_spec = {
        "name": "joint_targets",
        "shape": [2],
        "fields": [
            {"name": "left_hip_pitch", "dim": 1, "units": "rad"},
            {"name": "right_hip_pitch", "dim": 1, "units": "rad"},
        ],
    }

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.5, -0.25, 0.75, 0.1], [-0.4, 0.2, 0.3, 0.6]])
            )
            self.linear.bias.copy_(torch.tensor([0.05, -0.15]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


def test_sonic_export_round_trip_and_parity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_policy.pt"
    output = tmp_path / "tiny_policy.onnx"
    policy = TinySonicPolicy().eval()
    torch.save(policy, checkpoint)

    result = export_onnx(
        checkpoint=str(checkpoint),
        output=str(output),
        verify=True,
        parity_atol=1e-4,
    )

    assert result.status == "exported"
    assert result.opset == 17
    assert result.axes == "dynamic"
    assert result.normalize == "baked"
    assert result.metadata == "sidecar"
    assert result.parity is not None
    assert result.parity["passed"] is True

    metadata = load_export_metadata(result.metadata_path)
    assert metadata["deterministic_action_path"] == "mean"
    assert metadata["obs_spec"]["fields"][0]["name"] == "base_ang_vel"
    assert metadata["action_spec"]["shape"] == [2]
    assert metadata["control_dt"] == 0.02

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    obs = np.random.default_rng(7).normal(size=(3, 4)).astype(np.float32)
    action = session.run(["action"], {"obs": obs})[0]
    assert action.shape == (3, 2)
    assert action.dtype == np.float32

    parity = validate_onnx_parity(
        policy=policy,
        onnx_path=str(output),
        observations=torch.as_tensor(obs),
        atol=1e-4,
    )
    assert parity.passed
    assert parity.max_abs_diff <= 1e-4


def test_sonic_export_loads_state_dict_checkpoint_from_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state_dict.pt"
    output = tmp_path / "state_dict.onnx"
    policy = TinySonicPolicy().eval()
    torch.save({"actor_model_state_dict": policy.state_dict()}, checkpoint)

    result = export_onnx(
        checkpoint=str(checkpoint),
        output=str(output),
        config={
            "policy": {
                "class": f"{TinySonicPolicy.__module__}.TinySonicPolicy",
                "kwargs": {},
            }
        },
        verify=True,
    )

    assert result.status == "exported"
    assert result.parity is not None
    assert result.parity["passed"] is True


def test_sonic_export_bakes_normalization_and_embeds_metadata(tmp_path: Path) -> None:
    output = tmp_path / "normalized.onnx"
    policy = TinySonicPolicy().eval()
    config = {
        "control_dt": 0.01,
        "normalization": {
            "mean": [1.0, 2.0, 3.0, 4.0],
            "var": [4.0, 4.0, 4.0, 4.0],
            "epsilon": 1e-5,
        },
    }

    result = export_onnx(
        checkpoint="in-memory-policy",
        output=str(output),
        policy=policy,
        config=config,
        metadata="embedded",
        sample_observation=torch.zeros(1, 4),
    )

    assert result.metadata_path == ""
    assert result.normalize == "baked"
    obs = torch.tensor([[2.0, 4.0, 6.0, 8.0]], dtype=torch.float32)
    denom = torch.sqrt(torch.tensor(config["normalization"]["var"]) + 1e-5)
    expected = policy((obs - torch.tensor(config["normalization"]["mean"])) / denom)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(["action"], {"obs": obs.numpy()})[0]
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=1e-4)

    import onnx

    model = onnx.load(str(output))
    props = {item.key: item.value for item in model.metadata_props}
    assert props["npa.sonic.format"] == "npa_sonic_onnx_export_v1"
    embedded = json.loads(props["npa.sonic.export"])
    assert embedded["metadata"] == "embedded"
    assert "normalization" not in embedded


def test_sonic_export_sidecar_keeps_normalization_stats(tmp_path: Path) -> None:
    output = tmp_path / "sidecar.onnx"
    result = export_onnx(
        checkpoint="in-memory-policy",
        output=str(output),
        policy=TinySonicPolicy().eval(),
        config={
            "normalization": {
                "mean": [0.0, 0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0, 1.0],
            }
        },
        normalize="sidecar",
    )

    metadata = load_export_metadata(result.metadata_path)
    assert metadata["normalize"] == "sidecar"
    assert metadata["normalization"]["std"] == [1.0, 1.0, 1.0, 1.0]
