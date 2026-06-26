"""Unit tests for sim2real ONNX policy export.

The pure helpers (``build_policy_contract``, ``infer_mlp_dims``,
``actor_weight_shapes``, ``load_state_dict_from_checkpoint``) run with no torch.
``test_export_policy_onnx_shapes_mocked`` mocks the torch-backed pieces so the
export wiring/contract is exercised without torch installed. A torch+onnxruntime
round-trip is gated behind ``importorskip`` so it runs where those are available
and skips in the light unit env.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.sim2real import policy_export as pe
from npa.workflows.sim2real.policy_export import (
    PolicyExportError,
    actor_weight_shapes,
    build_policy_contract,
    infer_mlp_dims,
    load_state_dict_from_checkpoint,
)


def _state_dict(obs_dim: int = 6, act_dim: int = 4, hidden=(16, 8)) -> dict:
    """A fake rsl_rl ActorCritic model_state_dict (numpy arrays expose .shape)."""

    dims = [obs_dim, *hidden, act_dim]
    state: dict = {"std": np.zeros((act_dim,), dtype=np.float32)}
    idx = 0
    for in_f, out_f in zip(dims, dims[1:]):
        state[f"actor.{idx}.weight"] = np.zeros((out_f, in_f), dtype=np.float32)
        state[f"actor.{idx}.bias"] = np.zeros((out_f,), dtype=np.float32)
        state[f"critic.{idx}.weight"] = np.zeros((out_f, in_f), dtype=np.float32)
        state[f"critic.{idx}.bias"] = np.zeros((out_f,), dtype=np.float32)
        idx += 2  # rsl_rl interleaves activation modules at odd indices
    return state


# --------------------------------------------------------------------------- #
# Pure dim inference.
# --------------------------------------------------------------------------- #
def test_infer_mlp_dims_basic() -> None:
    shapes = actor_weight_shapes(_state_dict(obs_dim=36, act_dim=8, hidden=(256, 128, 64)))
    obs_dim, act_dim, hidden = infer_mlp_dims(shapes)
    assert obs_dim == 36
    assert act_dim == 8
    assert hidden == [256, 128, 64]


def test_actor_weight_shapes_ignores_critic_and_std() -> None:
    shapes = actor_weight_shapes(_state_dict())
    assert all(name.startswith("actor.") and name.endswith(".weight") for name in shapes)
    assert "std" not in shapes
    assert not any("critic" in name for name in shapes)


def test_infer_mlp_dims_empty_raises() -> None:
    with pytest.raises(PolicyExportError, match="no actor"):
        infer_mlp_dims({})


def test_infer_mlp_dims_dim_mismatch_raises() -> None:
    # actor.0 out=16 but actor.2 in=8 -> broken topology.
    shapes = {"actor.0.weight": (16, 6), "actor.2.weight": (4, 8)}
    with pytest.raises(PolicyExportError, match="dim mismatch"):
        infer_mlp_dims(shapes)


def test_infer_mlp_dims_non_2d_raises() -> None:
    with pytest.raises(PolicyExportError, match="not 2-D"):
        infer_mlp_dims({"actor.0.weight": (16,)})


def test_load_state_dict_from_checkpoint_variants() -> None:
    sd = _state_dict()
    assert load_state_dict_from_checkpoint({"model_state_dict": sd}) is sd
    # Bare state dict (actor.* at top level) is accepted.
    assert load_state_dict_from_checkpoint(sd) is sd
    with pytest.raises(PolicyExportError, match="neither"):
        load_state_dict_from_checkpoint({"optimizer_state_dict": {}})


# --------------------------------------------------------------------------- #
# Pure contract builder.
# --------------------------------------------------------------------------- #
def test_build_policy_contract_opaque_layout_and_caveat() -> None:
    contract = build_policy_contract(
        obs_dim=36,
        act_dim=8,
        isaac_task="Isaac-Lift-Cube-Franka-v0",
        hidden_dims=[256, 128, 64],
        checkpoint={"train_iter": 975},
        created_at="2026-06-26T00:00:00+00:00",
    )
    assert contract["format"] == pe.POLICY_EXPORT_FORMAT
    assert contract["obs"]["dim"] == 36
    assert contract["obs"]["layout"]["kind"] == "opaque"
    assert contract["action"]["dim"] == 8
    # Known task hint is applied.
    assert contract["action"]["type"] == "joint_position"
    assert "ActionManager" in contract["action"]["note"]
    assert contract["network"]["hidden_dims"] == [256, 128, 64]
    # Honest sim-to-real caveat is always present.
    assert "does NOT bridge sim-to-real" in contract["sim_to_real_caveat"]
    assert contract["checkpoint"]["train_iter"] == 975


def test_build_policy_contract_ordered_terms() -> None:
    terms = [
        {"name": "joint_pos", "dim": 9},
        {"name": "joint_vel", "dim": 9},
        {"name": "object_pose", "dim": 7},
        {"name": "target", "dim": 11},
    ]
    contract = build_policy_contract(obs_dim=36, act_dim=8, obs_terms=terms)
    assert contract["obs"]["layout"]["kind"] == "ordered_terms"
    assert contract["obs"]["layout"]["terms"] == terms


def test_build_policy_contract_terms_sum_mismatch_raises() -> None:
    with pytest.raises(PolicyExportError, match="sum to"):
        build_policy_contract(
            obs_dim=36, act_dim=8, obs_terms=[{"name": "x", "dim": 10}]
        )


def test_build_policy_contract_unknown_task_action_opaque() -> None:
    contract = build_policy_contract(obs_dim=12, act_dim=3, isaac_task="Some-Unknown-Task")
    assert contract["action"]["type"] == "opaque"


def test_build_policy_contract_action_type_override() -> None:
    contract = build_policy_contract(
        obs_dim=12, act_dim=3, isaac_task="Isaac-Lift-Cube-Franka-v0",
        action_type="joint_velocity",
    )
    assert contract["action"]["type"] == "joint_velocity"


@pytest.mark.parametrize("bad", [0, -1])
def test_build_policy_contract_rejects_nonpositive_dims(bad: int) -> None:
    with pytest.raises(PolicyExportError):
        build_policy_contract(obs_dim=bad, act_dim=4)
    with pytest.raises(PolicyExportError):
        build_policy_contract(obs_dim=4, act_dim=bad)


# --------------------------------------------------------------------------- #
# Export wiring with torch mocked out (runs without torch installed).
# --------------------------------------------------------------------------- #
def test_export_policy_onnx_shapes_mocked(tmp_path: Path, monkeypatch) -> None:
    ckpt = tmp_path / "model_975.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")  # provenance hashes the bytes
    out_dir = tmp_path / "export"

    state = _state_dict(obs_dim=36, act_dim=8, hidden=(256, 128, 64))
    checkpoint = {"model_state_dict": state, "iter": 975}

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_export(model, sample, path, **kwargs):
        Path(path).write_bytes(b"ONNX")  # single self-contained file, no .data

    fake_torch = SimpleNamespace(
        load=lambda *a, **k: checkpoint,
        zeros=lambda *shape, **k: SimpleNamespace(shape=tuple(shape)),
        no_grad=_NoGrad,
        float32="float32",
        onnx=SimpleNamespace(export=_fake_export),
    )

    monkeypatch.setattr(pe, "_import_torch", lambda: fake_torch)
    monkeypatch.setattr(pe, "_build_actor_module", lambda t, sd, act: "ACTOR")
    monkeypatch.setattr(
        pe, "_detect_normalization", lambda t, c, d: (None, {"type": "none"})
    )
    monkeypatch.setattr(
        pe,
        "_make_forward_module",
        lambda t, actor, norm: (lambda obs: SimpleNamespace(shape=(1, 8))),
    )

    result = pe.export_policy_onnx(
        str(ckpt),
        out_dir=str(out_dir),
        isaac_task="Isaac-Lift-Cube-Franka-v0",
        checkpoint_source="s3://bucket/run/model_975.pt",
    )

    assert result["status"] == "success"
    assert result["obs_dim"] == 36
    assert result["act_dim"] == 8
    assert result["hidden_dims"] == [256, 128, 64]
    assert (out_dir / "policy.onnx").is_file()
    contract = json.loads((out_dir / "policy_contract.json").read_text())
    assert contract["obs"]["dim"] == 36
    assert contract["action"]["dim"] == 8
    assert contract["checkpoint"]["train_iter"] == 975
    assert contract["checkpoint"]["source"] == "s3://bucket/run/model_975.pt"


def test_export_policy_onnx_dim_override_mismatch_raises(tmp_path: Path, monkeypatch) -> None:
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    checkpoint = {"model_state_dict": _state_dict(obs_dim=36, act_dim=8)}
    monkeypatch.setattr(
        pe, "_import_torch", lambda: SimpleNamespace(load=lambda *a, **k: checkpoint)
    )
    with pytest.raises(PolicyExportError, match="disagrees with checkpoint"):
        pe.export_policy_onnx(str(ckpt), out_dir=str(tmp_path / "o"), obs_dim=99)


# --------------------------------------------------------------------------- #
# Real torch -> ONNX -> onnxruntime round-trip (gated; the whole point).
# --------------------------------------------------------------------------- #
def test_export_real_torch_onnxruntime_roundtrip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")

    obs_dim, act_dim = 12, 4
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 16),
        torch.nn.ELU(),
        torch.nn.Linear(16, act_dim),
    )
    # rsl_rl-style keys: actor.0, actor.2.
    state = {
        "actor.0.weight": actor[0].weight.detach(),
        "actor.0.bias": actor[0].bias.detach(),
        "actor.2.weight": actor[2].weight.detach(),
        "actor.2.bias": actor[2].bias.detach(),
        "std": torch.ones(act_dim),
    }
    ckpt = tmp_path / "model_5.pt"
    torch.save({"model_state_dict": state, "iter": 5}, str(ckpt))

    result = pe.export_policy_onnx(
        str(ckpt), out_dir=str(tmp_path / "export"), isaac_task="Isaac-Reach-Franka-v0"
    )
    assert result["obs_dim"] == obs_dim
    assert result["act_dim"] == act_dim
    # Single self-contained file (no external-data sidecar).
    onnx_path = Path(result["onnx_path"])
    assert onnx_path.is_file()
    assert not onnx_path.with_name(onnx_path.name + ".data").exists()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    obs = np.zeros((1, obs_dim), dtype=np.float32)
    action = sess.run(["action"], {"obs": obs})[0]
    assert action.shape == (1, act_dim)

    # Parity with the torch actor.
    with torch.no_grad():
        expected = actor(torch.from_numpy(obs)).numpy()
    assert np.allclose(action, expected, atol=1e-4)
