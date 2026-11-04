"""Verify weighted flow gradients, split selection, and runtime refusal boundaries."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.groot.offline_feedback import OfflineManifest, PreparedEpisode
from npa.workbench.groot.offline_training import (
    FeedbackCollator,
    FeedbackDataset,
    _training_trace,
    _terminal_mask_evidence,
    _validate_cuda_runtime,
    _load_native_runtime,
    _load_training_dataset,
    checkpoint_digest,
    train_offline,
    weighted_flow_loss,
)


@pytest.fixture(autouse=True)
def torch_runtime():
    return pytest.importorskip("torch")


def _manifest(**overrides):
    return OfflineManifest(
        dataset_sha256="a" * 64,
        feedback_sha256="b" * 64,
        mode="outcome",
        action_horizon=2,
        temperature=1.0,
        episodes=[
            PreparedEpisode(
                episode_index=0, split="train", length=3, weights=[2.0, 1.0, 0.0]
            ),
            PreparedEpisode(
                episode_index=1, split="eval", length=2, weights=[1.0, 1.0]
            ),
        ],
        **overrides,
    )


def _native_loss_model(torch):
    class NativeReduction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, inputs):
            losses = (self.prediction - inputs["action"]) ** 2
            mask = inputs["action_mask"]
            return {"loss": (losses * mask).sum() / mask.sum()}

    return NativeReduction()


def test_weights_match_unreduced_native_gradient(torch_runtime):
    torch = torch_runtime
    model = _native_loss_model(torch)
    actions = torch.tensor([[[1.0], [2.0]], [[3.0], [8.0]]])
    mask = torch.tensor([[[1], [1]], [[1], [0]]], dtype=torch.bool)
    weights = torch.tensor([2.0, 0.5])
    batch = {"action": actions, "action_mask": mask, "offline_weight": weights}
    loss, _ = weighted_flow_loss(model, batch)
    expected = (
        ((model.prediction - actions) ** 2) * mask * weights[:, None, None]
    ).sum() / mask.sum()
    assert loss.item() == pytest.approx(expected.item())
    expected_gradient = torch.autograd.grad(expected, model.prediction)[0]
    loss.backward()
    assert model.prediction.grad.item() == pytest.approx(expected_gradient.item())
    assert batch["action"] is actions and batch["action_mask"] is mask
    assert "offline_weight" in batch


@pytest.mark.parametrize("weight", [0.0, 0.2, 2.0])
def test_weight_still_applies_with_batch_size_one(torch_runtime, weight):
    torch = torch_runtime
    model = _native_loss_model(torch)
    batch = {
        "action": torch.ones(1, 2, 1),
        "action_mask": torch.ones(1, 2, 1),
        "offline_weight": torch.tensor([weight]),
    }
    loss, _ = weighted_flow_loss(model, batch)
    assert loss.item() == pytest.approx(0.75**2 * weight)
    loss.backward()
    assert model.prediction.grad.item() == pytest.approx(-1.5 * weight)


@pytest.mark.parametrize(
    "weights", [[-1.0], [float("nan")], [float("inf")], [[1.0]], [1.0, 2.0]]
)
def test_invalid_loss_weights_are_rejected(torch_runtime, weights):
    torch = torch_runtime
    batch = {
        "action": torch.ones(1, 2, 1),
        "action_mask": torch.ones(1, 2, 1),
        "offline_weight": torch.tensor(weights),
    }
    with pytest.raises(ValueError, match="weight"):
        weighted_flow_loss(_native_loss_model(torch), batch)


def test_dataset_selects_train_and_masks_terminal_actions():
    class Native:
        all_steps = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        mask = np.ones((2, 3), dtype=bool)

        def __getitem__(self, index):
            return {"action_mask": self.mask, "action": np.full((2, 3), index)}

    native = Native()
    dataset = FeedbackDataset(native, _manifest())
    assert len(dataset) == 3
    final = dataset[2]
    assert final["offline_weight"] == 0
    assert final["action_mask"].tolist() == [[True] * 3, [False] * 3]
    np.testing.assert_array_equal(final["action"], np.full((2, 3), 2))
    assert native.mask.all()
    assert len(FeedbackDataset(native, _manifest(), split="eval")) == 2


def test_dataset_rejects_misaligned_native_steps():
    with pytest.raises(ValueError, match="frames differ"):
        FeedbackDataset(SimpleNamespace(all_steps=[(0, 0)]), _manifest())


def test_terminal_mask_evidence_decodes_observation_and_excludes_padding():
    class Native:
        all_steps = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

        def __getitem__(self, index):
            return {
                "action_mask": np.ones((2, 3), dtype=bool),
                "action": np.full((2, 3), index),
                "video.front": np.zeros((2, 2, 3), dtype=np.uint8),
            }

    evidence = _terminal_mask_evidence(FeedbackDataset(Native(), _manifest()))
    assert evidence == {
        "episode_index": 0,
        "frame_index": 2,
        "valid_action_horizon_steps": 1,
        "masked_action_horizon_steps": 1,
        "decoded_observation_shapes": {"video.front": [2, 2, 3]},
    }


def test_collator_does_not_feed_weight_to_native_processor(torch_runtime):
    def native(features):
        assert all("offline_weight" not in sample for sample in features)
        return {"action": "preserved"}

    features = [{"offline_weight": 0.5, "action": "preserved"}]
    result = FeedbackCollator(native)(features)
    assert result["offline_weight"].tolist() == [0.5]
    assert result["action"] == "preserved"
    assert features[0]["offline_weight"] == 0.5


def test_changed_dataset_refuses_before_native_import(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"gr00t_n1_5"}')
    path = tmp_path / "manifest.json"
    path.write_text(_manifest().model_dump_json())
    monkeypatch.setattr(
        "npa.workbench.groot.offline_training._load_native_runtime",
        lambda *_: pytest.fail("native runtime must not be loaded"),
    )
    with pytest.raises(ValueError, match="dataset bytes changed"):
        train_offline(
            dataset_root=dataset,
            manifest_path=path,
            groot_source=tmp_path / "native",
            base_model=checkpoint,
            output_dir=tmp_path / "run",
            data_config="synthetic",
            embodiment_tag="new_embodiment",
            max_steps=1,
        )
    assert not (tmp_path / "run").exists()


def test_distributed_runtime_refuses_before_source_access(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one GPU"):
        _load_native_runtime(tmp_path)


@pytest.mark.parametrize("visible_devices", [0, 2])
def test_cuda_runtime_refuses_non_single_visible_device(visible_devices):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: visible_devices,
        is_bf16_supported=lambda: True,
    )
    with pytest.raises(ValueError, match="exactly one visible CUDA GPU"):
        _validate_cuda_runtime(SimpleNamespace(cuda=cuda))


def test_cuda_runtime_reports_one_visible_device():
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        is_bf16_supported=lambda: True,
    )
    assert _validate_cuda_runtime(SimpleNamespace(cuda=cuda)) == 1


def test_training_trace_requires_ordered_finite_optimizer_losses():
    trainer = SimpleNamespace(
        state=SimpleNamespace(
            log_history=[
                {"step": 1, "loss": 1.25},
                {"step": 2, "loss": 0.75, "learning_rate": 0.0001},
            ]
        )
    )
    assert _training_trace(trainer) == [
        {"step": 1, "loss": 1.25},
        {"step": 2, "loss": 0.75},
    ]
    trainer.state.log_history = [{"step": 2, "loss": 1.0}, {"step": 1, "loss": 0.5}]
    with pytest.raises(ValueError, match="step ordered"):
        _training_trace(trainer)


def test_native_statistics_cannot_be_silently_recomputed(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/stats.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        _load_training_dataset({}, tmp_path, _manifest(), "unused", "new_embodiment")


def test_checkpoint_digest_binds_weights_and_normalization(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gr00t_n1_5"}))
    (tmp_path / "model.safetensors").write_bytes(b"synthetic-identity-only")
    initial = checkpoint_digest(tmp_path)
    (tmp_path / "experiment_cfg").mkdir()
    (tmp_path / "experiment_cfg/metadata.json").write_text("{}")
    assert checkpoint_digest(tmp_path) != initial
