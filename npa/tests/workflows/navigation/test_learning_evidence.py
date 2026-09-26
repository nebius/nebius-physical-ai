"""Preserve native pre-learning state and drain final asynchronous scalar events."""

import json
from types import SimpleNamespace

import pytest

from npa.workflows.navigation.learning_evidence import (
    _learning_curves,
    _save_initial_checkpoint,
)


def test_initial_checkpoint_needs_no_initialized_logger(tmp_path):
    torch = pytest.importorskip("torch")
    state = {
        "actor_state_dict": {"weight": torch.tensor([1.25])},
        "critic_state_dict": {"weight": torch.tensor([-2.0])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
    }
    runner = SimpleNamespace(
        alg=SimpleNamespace(save=lambda: state.copy()),
        current_learning_iteration=37,
        logger=SimpleNamespace(),
        save=lambda *_: pytest.fail("native logger is not initialized before learn"),
    )
    path = tmp_path / "reference.pt"
    _save_initial_checkpoint(runner, path)
    result = torch.load(path, weights_only=True)
    assert result["actor_state_dict"]["weight"].item() == 1.25
    assert result["critic_state_dict"]["weight"].item() == -2.0
    assert result["optimizer_state_dict"] == state["optimizer_state_dict"]
    assert result["iter"] == 37 and result["infos"] is None


def test_export_includes_last_async_tensorboard_scalar(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(tmp_path / "checkpoints"), flush_secs=3600)
    for step in range(1500):
        writer.add_scalar("Train/mean_reward", step / 100.0, step)
    runner = SimpleNamespace(logger=SimpleNamespace(writer=writer))
    _learning_curves(runner, tmp_path)
    rows = json.loads((tmp_path / "learning-curves.json").read_text())[
        "Train/mean_reward"
    ]
    assert len(rows) == 1500
    assert rows[-1]["step"] == 1499
    assert rows[-1]["value"] == pytest.approx(14.99)
    assert writer.file_writer is None
