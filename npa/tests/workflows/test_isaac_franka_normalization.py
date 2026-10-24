"""Verify native frozen training statistics, checkpoint restoration, and literal audit guards."""

from types import SimpleNamespace

import pytest

from npa.workflows.sim2real.isaac_franka_normalization import (
    checkpoint_normalization_metadata,
    freeze_training_normalization,
    restore_checkpoint_normalization,
)


@pytest.fixture
def native_statistics():
    torch = pytest.importorskip("torch")
    normalization = pytest.importorskip("rsl_rl.modules").EmpiricalNormalization
    torch.manual_seed(37)
    actor = SimpleNamespace(obs_normalizer=normalization((39,)))
    actor.obs_normalizer.train()
    actor.obs_normalizer.update(torch.randn(128, 39))
    return torch, actor, normalization


def test_native_training_updates_cannot_move_frozen_statistics_or_actor_inputs(
    native_statistics,
):
    torch, actor, _ = native_statistics
    normalizer = actor.obs_normalizer
    probe = torch.randn(8, 39)
    metadata = freeze_training_normalization(actor)
    before = normalizer(probe).detach().clone()
    state = {name: value.clone() for name, value in normalizer.state_dict().items()}
    assert metadata["count"] == metadata["until"] == 128
    normalizer.train()
    normalizer.update(torch.randn(256, 39) + 50)
    torch.testing.assert_close(normalizer(probe), before)
    for name, value in state.items():
        assert torch.equal(normalizer.state_dict()[name], value)
    assert checkpoint_normalization_metadata(actor) == metadata


def test_exact_native_checkpoint_restores_freeze_after_loading_statistic_tensors(
    native_statistics,
):
    torch, actor, normalization = native_statistics
    metadata = freeze_training_normalization(actor)
    restored = SimpleNamespace(obs_normalizer=normalization((39,)))
    restored.obs_normalizer.load_state_dict(actor.obs_normalizer.state_dict())
    assert restored.obs_normalizer.until is None
    restore_checkpoint_normalization(
        restored, {"npa_stock_franka_normalization": metadata}
    )
    restored.obs_normalizer.train()
    restored.obs_normalizer.update(torch.randn(256, 39) - 50)
    assert checkpoint_normalization_metadata(restored) == metadata
    for name, value in actor.obs_normalizer.state_dict().items():
        assert torch.equal(restored.obs_normalizer.state_dict()[name], value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("scope", "validation"),
        ("gold_used", True),
        ("gold_used", 0),
        ("frozen", 1),
        ("count", 128.0),
        ("count", True),
        ("count", 127),
        ("until", 128.0),
        ("until", True),
        ("until", None),
    ],
)
def test_native_load_rejects_changed_or_coerced_normalization_metadata(
    native_statistics, field, value
):
    _, actor, _ = native_statistics
    metadata = freeze_training_normalization(actor)
    metadata[field] = value
    actor.obs_normalizer.until = None
    with pytest.raises(RuntimeError):
        restore_checkpoint_normalization(
            actor, {"npa_stock_franka_normalization": metadata}
        )
    assert actor.obs_normalizer.until is None


def test_native_load_rejects_missing_metadata_and_changed_loaded_count(
    native_statistics,
):
    torch, actor, _ = native_statistics
    metadata = freeze_training_normalization(actor)
    with pytest.raises(RuntimeError, match="metadata"):
        restore_checkpoint_normalization(actor, {})
    actor.obs_normalizer.until = None
    actor.obs_normalizer.update(torch.randn(8, 39))
    with pytest.raises(RuntimeError, match="loaded training statistics"):
        restore_checkpoint_normalization(
            actor, {"npa_stock_franka_normalization": metadata}
        )


def test_explicit_unbootstrapped_checkpoint_preserves_native_statistic_updates(
    native_statistics,
):
    torch, actor, normalization = native_statistics
    metadata = checkpoint_normalization_metadata(actor)
    assert metadata["frozen"] is False and metadata["until"] is None
    restored = SimpleNamespace(obs_normalizer=normalization((39,)))
    restored.obs_normalizer.load_state_dict(actor.obs_normalizer.state_dict())
    restore_checkpoint_normalization(
        restored, {"npa_stock_franka_normalization": metadata}
    )
    restored.obs_normalizer.train()
    restored.obs_normalizer.update(torch.randn(128, 39))
    assert restored.obs_normalizer.count.item() == 256


def test_uninitialized_or_changed_freeze_bound_cannot_be_saved(native_statistics):
    _, actor, normalization = native_statistics
    with pytest.raises(RuntimeError, match="initialized training"):
        freeze_training_normalization(
            SimpleNamespace(obs_normalizer=normalization((39,)))
        )
    freeze_training_normalization(actor)
    actor.obs_normalizer.until = 127
    with pytest.raises(RuntimeError, match="changed after"):
        checkpoint_normalization_metadata(actor)
