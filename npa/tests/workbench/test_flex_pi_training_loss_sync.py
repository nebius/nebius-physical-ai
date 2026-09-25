"""Keep loss readback asynchronous without changing updates or hiding its cost."""

from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def engine(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    packages = {
        name: ModuleType(name)
        for name in (
            "flexpi",
            "flexpi.datasets",
            "flexpi.datasets.lerobot",
            "flexpi.datasets.lerobot.robot_video_dataset",
            "flexpi.trainer",
        )
    }
    packages["flexpi.datasets.lerobot"].base_lerobot_dataset = ModuleType("base")
    packages["flexpi.datasets.lerobot.robot_video_dataset"].RobotVideoDataset = object
    packages["flexpi.trainer"].Wan22Trainer = object
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    from npa.workbench.flex_pi import training

    path = Path(training.__file__).with_name("training_engine.py")
    spec = importlib.util.spec_from_file_location("_verified_training_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Accelerator:
    def __init__(self, torch, accumulation):
        self.torch = torch
        self.accumulation = accumulation
        self.step = 0
        self.sync_gradients = False
        self.last_batch = False
        self.optimizer_step_was_skipped = False
        self.device = torch.device("cpu")

    @contextmanager
    def accumulate(self, model):
        self.step += 1
        self.sync_gradients = self.step % self.accumulation == 0 or self.last_batch
        yield

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        (loss / self.accumulation).backward()

    def clip_grad_norm_(self, parameters, limit):
        parameters = list(parameters)
        self.gradients = [parameter.grad.detach().clone() for parameter in parameters]
        self.clipping_norm = self.torch.nn.utils.clip_grad_norm_(parameters, limit)
        return self.clipping_norm


def _trainer(engine, width):
    torch = engine.torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, dtype=torch.float64)

        def forward(self, sample):
            loss = (self.linear(sample["inputs"]) - sample["targets"]).square().mean()
            loss = loss + sample.get("poison", 0.0)
            return loss, {"total": loss}

    torch.manual_seed(42)
    trainer = engine.VerifiedTrainer()
    trainer.model = Model()
    trainer.batch_size = width
    trainer.gradient_accumulation_steps = 24 // width
    trainer.accelerator = _Accelerator(torch, trainer.gradient_accumulation_steps)
    trainer.max_grad_norm = 1.0
    trainer.global_step = 0
    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(), lr=1e-4, betas=(0.9, 0.95)
    )
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )
    return trainer


@pytest.mark.parametrize("width", [1, 3])
def test_full_and_tail_updates_preserve_losses_gradients_and_optimizer(engine, width):
    from npa.workbench.flex_pi.training_state import state_digest

    torch = engine.torch
    actual, reference = _trainer(engine, width), _trainer(engine, width)
    generator = torch.Generator().manual_seed(17)
    seen = []
    expected = []
    for position in range(33 // width):
        sample = {
            "inputs": torch.randn(width, 2, dtype=torch.float64, generator=generator),
            "targets": torch.randn(width, 1, dtype=torch.float64, generator=generator),
        }
        divisor = (24 if position < 24 // width else 9) // width
        actual.accelerator.last_batch = position == 33 // width - 1
        loss = actual._backward_impl(sample, divisor)
        assert isinstance(loss, torch.Tensor) and not loss.requires_grad
        seen.append(loss)

        original_loss, _ = reference.model(sample)
        expected.append(float(original_loss.detach()))
        scaled = original_loss * (reference.gradient_accumulation_steps / divisor)
        (scaled / reference.gradient_accumulation_steps).backward()
        if actual.accelerator.sync_gradients:
            assert state_digest(actual.accelerator.gradients) == state_digest(
                [parameter.grad for parameter in reference.model.parameters()]
            )
            reference_norm = torch.nn.utils.clip_grad_norm_(
                reference.model.parameters(), 1.0
            )
            assert state_digest(actual.accelerator.clipping_norm) == state_digest(
                reference_norm
            )
            reference.optimizer.step()
            reference.scheduler.step()
            reference.optimizer.zero_grad(set_to_none=True)
            assert state_digest(actual.model.state_dict()) == state_digest(
                reference.model.state_dict()
            )
            assert state_digest(actual.optimizer.state_dict()) == state_digest(
                reference.optimizer.state_dict()
            )
            assert actual.scheduler.state_dict() == reference.scheduler.state_dict()
            assert actual._pending_losses == []
    assert actual.global_step == 2
    assert torch.stack(seen).tolist() == expected


def test_no_microbatch_scalar_reads_before_optimizer_boundary(engine, monkeypatch):
    torch = engine.torch
    trainer = _trainer(engine, 1)
    conversions = []
    for name in ("__float__", "__bool__"):
        original = getattr(torch.Tensor, name)

        def guarded(value, original=original, name=name):
            assert trainer.accelerator.sync_gradients, name
            conversions.append(name)
            return original(value)

        monkeypatch.setattr(torch.Tensor, name, guarded)
    sample = {
        "inputs": torch.ones(1, 2, dtype=torch.float64),
        "targets": torch.zeros(1, 1, dtype=torch.float64),
    }
    for _ in range(23):
        trainer._backward_impl(sample, 24)
    assert conversions == []
    trainer._backward_impl(sample, 24)
    assert conversions and trainer.global_step == 1


@pytest.mark.parametrize("position", [0, 11, 23])
@pytest.mark.parametrize("poison", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_earlier_loss_cannot_reach_optimizer_step(
    engine, monkeypatch, position, poison
):
    from npa.workbench.flex_pi.training_state import state_digest

    torch = engine.torch
    trainer = _trainer(engine, 1)
    before = state_digest(trainer.model.state_dict())
    monkeypatch.setattr(
        trainer.optimizer,
        "step",
        lambda *args, **kwargs: pytest.fail("nonfinite loss reached optimizer"),
    )
    with pytest.raises(RuntimeError, match="nonfinite training loss"):
        for index in range(24):
            trainer._backward_impl(
                {
                    "inputs": torch.ones(1, 2, dtype=torch.float64),
                    "targets": torch.zeros(1, 1, dtype=torch.float64),
                    # This additive constant leaves finite parameter gradients.
                    "poison": poison if index == position else 0.0,
                },
                24,
            )
    assert trainer.global_step == 0 and not trainer.optimizer.state
    assert state_digest(trainer.model.state_dict()) == before


def test_loss_readback_preserves_python_sum_and_stays_inside_update_timer(
    engine, monkeypatch
):
    torch = engine.torch
    trainer = _trainer(engine, 1)
    trainer.global_step = 1
    events, recorded = [], []
    original = torch.Tensor.tolist

    def readback(value):
        events.append("loss_readback")
        return original(value)

    def reduce(value, op=None):
        if op is None:
            value.mul_(4)

    monkeypatch.setattr(torch.Tensor, "tolist", readback)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("cuda_sync"))
    monkeypatch.setattr(
        engine.time, "perf_counter", lambda: events.append("timer") or 10
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    trainer._record = recorded.append
    losses = [torch.tensor(value) for value in (1e8, 1.0, -1e8, 0.25)]
    trainer._record_update(2.0, losses, 0.01)
    assert events[:3] == ["loss_readback", "cuda_sync", "timer"]
    assert recorded[0]["seconds"] == 8.0
    assert recorded[0]["samples"] == 16
    assert recorded[0]["loss"] == 0.3125


def test_nonfinite_remote_rank_prevents_local_optimizer_update(engine, monkeypatch):
    from npa.workbench.flex_pi.training_state import state_digest

    torch = engine.torch
    trainer = _trainer(engine, 1)
    before = state_digest(trainer.model.state_dict())
    collectives = []

    def remote_failure(value, op):
        assert op == torch.distributed.ReduceOp.MIN
        assert value.dtype == torch.int32 and value.numel() == 1
        collectives.append(True)
        value.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", remote_failure)
    monkeypatch.setattr(
        trainer.optimizer,
        "step",
        lambda *args, **kwargs: pytest.fail("remote nonfinite loss reached optimizer"),
    )
    with pytest.raises(RuntimeError, match="nonfinite training loss"):
        for _ in range(24):
            trainer._backward_impl(
                {
                    "inputs": torch.ones(1, 2, dtype=torch.float64),
                    "targets": torch.zeros(1, 1, dtype=torch.float64),
                },
                24,
            )
    assert collectives == [True]
    assert trainer.global_step == 0 and not trainer.optimizer.state
    assert state_digest(trainer.model.state_dict()) == before
