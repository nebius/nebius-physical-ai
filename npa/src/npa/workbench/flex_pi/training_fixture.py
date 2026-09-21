"""Record exact full-batch and true-tail native DDP diagnostic updates."""

import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from npa.workbench.flex_pi.training import TRAIN_FRAMES, VALIDATION_FRAMES
from npa.workbench.flex_pi.training_memory import observation_with_memory_fill
from npa.workbench.flex_pi.training_state import rng_digest, state_digest


def fixture_indices():
    """Select the real first96 and final36 anchors without a resumed sampler.

    Args:
        None; the frozen public split and seed define both ordered fixtures.
    Returns:
        Distinct global anchor indices in their original epoch order.
    Raises:
        RuntimeError: The fixed fixtures overlap or have an unexpected size.
    """
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(TRAIN_FRAMES, generator=generator).tolist()
    indices = permutation[:96] + permutation[-36:]
    if len(indices) != 132 or len(set(indices)) != 132:
        raise RuntimeError("qualification requires 132 distinct real anchors")
    return indices


def _finite_digest(tensors):
    for value in tensors.values():
        if value is None or not torch.isfinite(value).all():
            raise RuntimeError("qualification found a missing or nonfinite tensor")
    return state_digest(tensors)


class FixtureRecorder:
    """Hold hash-only evidence for two untimed native optimizer updates.

    Args:
        None; each disposable rank creates its own recorder.
    Returns:
        A recorder retaining hashes instead of data or model tensor payloads.
    Raises:
        RuntimeError: A recorded numeric value is missing or nonfinite.
    """

    def __init__(self):
        self.updates = []
        self.losses = []
        self.gradients = {}

    @observation_with_memory_fill()
    def _loss(self, loss, components):
        if any(not np.isfinite(float(value)) for value in components.values()):
            raise RuntimeError("qualification found a nonfinite component loss")
        self.losses.append(state_digest({"total": loss, "components": components}))

    @observation_with_memory_fill()
    def _gradients(self, trainer, stage):
        tensors = {
            name: parameter.grad
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }
        self.gradients[stage] = _finite_digest(tensors)

    @observation_with_memory_fill()
    def _update(self, trainer, norm):
        model = trainer.accelerator.unwrap_model(trainer.model)
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.updates.append(
            {
                "losses_sha256": state_digest(self.losses),
                "local_samples": len(self.losses),
                "gradients_sha256": dict(self.gradients),
                "clipping_norm_sha256": state_digest(norm),
                "trainable_parameters_sha256": _finite_digest(parameters),
                "model_sha256": state_digest(model.state_dict()),
                "buffers_sha256": state_digest(dict(model.named_buffers())),
                "optimizer_sha256": _optimizer_digest(trainer.optimizer),
                "scheduler_sha256": state_digest(trainer.scheduler.state_dict()),
                "rng_sha256": rng_digest(),
                "global_step": trainer.global_step,
                "accelerator_step": trainer.accelerator.step,
            }
        )
        self.losses = []
        self.gradients = {}


def _optimizer_digest(optimizer):
    state = optimizer.state_dict()
    tensors = {
        f"{parameter}/{key}": value
        for parameter, values in state["state"].items()
        for key, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    _finite_digest(tensors)
    return state_digest(state)


def _fixture_loader(trainer, indices):
    loader = DataLoader(
        trainer.train_dataset,
        batch_size=1,
        sampler=indices,
        num_workers=trainer.num_workers,
        pin_memory=True,
    )
    loader = trainer.accelerator.prepare(loader)
    loader.batch_sampler.even_batches = False
    if not trainer.accelerator.gradient_state.sync_with_dataloader:
        raise RuntimeError("qualification tail requires native loader-end sync")
    return loader


def _fixture_updates(trainer, loader, recorder):
    indices, inputs = [], []
    initial_step = trainer.global_step
    for position, sample in enumerate(loader, 1):
        indices.extend(sample.pop("npa_sample_index").detach().cpu().tolist())
        inputs.append(state_digest(sample))
        trainer._backward(sample, 24 if position <= 24 else 9)
        if trainer.accelerator.sync_gradients != (position in {24, 33}):
            raise RuntimeError("qualification changed native accumulation boundaries")
    if len(indices) != 33 or trainer.global_step != initial_step + 2:
        raise RuntimeError("qualification must execute exactly 96 and 36 samples")
    if [row["local_samples"] for row in recorder.updates] != [24, 9]:
        raise RuntimeError("qualification optimizer updates have incorrect divisors")
    return indices, state_digest(inputs)


def run_fixture_qualification(trainer):
    """Execute two diagnostic updates in this disposable native-DDP process.

    Args:
        trainer: Full-workload trainer, initialized or loaded before qualification.
    Returns:
        Hash-only evidence for inputs, losses, gradients, updates and state.
    Raises:
        RuntimeError: Sample order, accumulation, finiteness or counts differ.
    """
    started = time.perf_counter()
    if trainer.accelerator.step % trainer.gradient_accumulation_steps:
        raise RuntimeError("qualification requires an optimizer-update boundary")
    initial = trainer._training_state_receipt()
    model = trainer.accelerator.unwrap_model(trainer.model)
    initial_buffers = state_digest(dict(model.named_buffers()))
    global_indices = fixture_indices()
    recorder = FixtureRecorder()
    trainer._fixture_recorder = recorder
    indices, inputs = _fixture_updates(
        trainer, _fixture_loader(trainer, global_indices), recorder
    )
    if indices != global_indices[trainer.accelerator.process_index :: 4]:
        raise RuntimeError("qualification loader changed the ordered real anchors")
    trainer._verify_indices(indices, 132)
    local = {
        "rank": trainer.accelerator.process_index,
        "initial_buffers_sha256": initial_buffers,
        "ordered_anchors_sha256": state_digest(indices),
        "transformed_inputs_sha256": inputs,
        "updates": recorder.updates,
        "validation": _validation_fixture(trainer),
    }
    ranks = [None] * 4
    torch.distributed.all_gather_object(ranks, local)
    trainer._qualification_seconds = time.perf_counter() - started
    return {"initial_training_state": initial, "ranks": ranks}


def _validation_fixture(trainer):
    indices = list(range(4)) + list(range(VALIDATION_FRAMES - 4, VALIDATION_FRAMES))
    loader = DataLoader(
        trainer.val_dataset,
        batch_size=1,
        sampler=indices[trainer.accelerator.process_index :: 4],
        num_workers=trainer.num_workers,
        pin_memory=True,
    )
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.eval()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    rows = []
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for sample in loader:
            index = int(sample.pop("npa_sample_index").item())
            torch.manual_seed(1000000 + index)
            torch.cuda.manual_seed(1000000 + index)
            inputs = state_digest(sample)
            loss, components = trainer._validation_forward(model, sample)
            if not torch.isfinite(loss):
                raise RuntimeError("qualification validation loss is nonfinite")
            if any(not np.isfinite(float(value)) for value in components.values()):
                raise RuntimeError("qualification validation component is nonfinite")
            rows.append(
                {
                    "index": index,
                    "inputs_sha256": inputs,
                    "loss_sha256": state_digest(loss),
                    "components_sha256": state_digest(components),
                    "rng_sha256": rng_digest(),
                }
            )
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    trainer._set_dit_only_train_mode()
    if len(rows) != 2:
        raise RuntimeError("qualification requires all eight held-out fixture anchors")
    return rows
