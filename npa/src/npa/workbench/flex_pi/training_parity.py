"""Qualify larger microbatches on real anchors before timed training begins."""

from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate

from npa.workbench.flex_pi.training_parity_metrics import ParityComparison
from npa.workbench.flex_pi.training_compile import candidate_compilation
from npa.workbench.flex_pi.training_randomness import AnchorRandomness
from npa.workbench.flex_pi.training_state import state_digest


@contextmanager
def _preserve_random_generators():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _validate_architecture(trainer, model):
    if trainer.global_step != 0 or trainer.optimizer.state:
        raise RuntimeError("microbatch parity requires fresh optimizer state")
    if getattr(model, "_glue_cache_enabled", False):
        raise RuntimeError("microbatch parity requires disabled inference caching")
    configuration = model.flex_joint
    if any(
        not getattr(configuration, f"cross_modal_predict_{stream}")
        for stream in ("video", "dino", "pointmap")
    ):
        raise RuntimeError("microbatch parity requires all four loss streams")
    if model._pointmap_globally_off:
        raise RuntimeError("microbatch parity requires the real pointmap stream")
    if len(trainer.cfg.data.train.dataset_dirs) != 1:
        raise RuntimeError("microbatch parity requires one immutable dataset")
    if trainer.cfg.data.train.get("exterior_view_aug_prob", 0) != 0:
        raise RuntimeError("microbatch parity requires fixed camera intrinsics")
    for module in model.modules():
        if (
            isinstance(module, torch.nn.modules.dropout._DropoutNd)
            and module.training
            and module.p
        ):
            raise RuntimeError("active dropout is outside the pinned parity contract")
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.training
        ):
            raise RuntimeError("training batch normalization prevents decomposition")


@contextmanager
def _preserve_model_attributes(model):
    names = ("_camera_intrinsics", "_batch_flex")
    saved = {name: getattr(model, name) for name in names if hasattr(model, name)}
    try:
        yield
    finally:
        for name in names:
            if name in saved:
                setattr(model, name, saved[name])
            elif hasattr(model, name):
                delattr(model, name)
        model.vae.model.clear_cache()


def _fixture(trainer, global_samples):
    permutation = list(trainer.train_sampler)
    indices = permutation[:96] if global_samples == 96 else permutation[-36:]
    indices = indices[trainer.accelerator.process_index :: 4]
    trainer._verify_indices(indices, global_samples)
    samples = [trainer.train_dataset[index] for index in indices]
    first = samples[0]["camera_intrinsics"]
    if any(not torch.equal(first, sample["camera_intrinsics"]) for sample in samples):
        raise RuntimeError("batched anchors have different camera intrinsics")
    return samples, indices


def _restore_parameters(parameters, initial):
    with torch.no_grad():
        for name, parameter in parameters:
            parameter.copy_(initial[name])
            parameter.grad = None


def _forward_backward(trainer, samples, width, tape):
    from accelerate.utils import send_to_device
    import flexpi.models.flexpi as upstream

    losses = {}
    original = upstream.sample_flex_batch_flags
    replacement = partial(tape.flags, original)
    for offset in range(0, len(samples), width):
        group = list(range(offset, offset + width))
        sample = default_collate([samples[index] for index in group])
        sample.pop("npa_sample_index")
        sample = send_to_device(sample, trainer.accelerator.device)
        tape.begin(group)
        with tape, patch.object(upstream, "sample_flex_batch_flags", replacement):
            with trainer.model.no_sync(), trainer.accelerator.autocast():
                loss, streams = trainer.model(sample)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite fixed-batch parity loss")
                (loss * (width / len(samples))).backward()
            tape.finish()
        for key, value in {"total": float(loss.detach()), **streams}.items():
            losses[key] = losses.get(key, 0.0) + value * width / len(samples)
    return losses


def _synchronize_gradients(parameters, comparison):
    for name, parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("parity found a missing trainable gradient")
        parameter.grad.div_(4)
        torch.distributed.all_reduce(parameter.grad)
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("nonfinite fixed-batch parity gradient")
        comparison.observe("gradient", name, parameter.grad)


def _observe_update(parameters, initial, optimizer, comparison):
    for name, parameter in parameters:
        updated = parameter.detach().cpu().float() - initial[name].float()
        comparison.observe("update", name, updated)
        state = optimizer.state[parameter]
        comparison.observe("first_moment", name, state["exp_avg"])
        comparison.observe("second_moment", name, state["exp_avg_sq"])
        comparison.observe("optimizer_step", name, state["step"])


def _parity_pass(trainer, parameters, initial, samples, width, tape, comparison):
    started = time.perf_counter()
    _parity_progress(trainer, comparison, width, "started", started)
    _restore_parameters(parameters, initial)
    optimizer = trainer._build_optimizer([parameter for _, parameter in parameters])
    losses = _forward_backward(trainer, samples, width, tape)
    _synchronize_gradients(parameters, comparison)
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in parameters], trainer.max_grad_norm
    )
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite parity clipping norm")
    optimizer.step()
    _observe_update(parameters, initial, optimizer, comparison)
    report = comparison.finish()
    report.update(
        losses=losses,
        clipping_norm=float(norm),
        microbatch=width,
        peak_learning_rate=trainer.learning_rate,
    )
    del optimizer
    _parity_progress(trainer, comparison, width, "completed", started)
    return report


def _parity_progress(trainer, comparison, width, event, started):
    if trainer.accelerator.is_main_process:
        phase = (
            "capture"
            if comparison.capture
            else "replay"
            if comparison.exact
            else "candidate"
        )
        print(
            json.dumps(
                {
                    "parity_phase": phase,
                    "event": event,
                    "microbatch": width,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )


def _compare_losses(reference, candidate, *, exact):
    checks = {}
    for key, value in reference["losses"].items():
        other = candidate["losses"][key]
        error = abs(other - value)
        checks[key] = error == 0 if exact or value == 0 else error / abs(value) <= 0.01
    norm, other_norm = reference["clipping_norm"], candidate["clipping_norm"]
    norm_error = abs(other_norm - norm)
    checks["clipping_norm"] = (
        norm_error == 0 if exact or norm == 0 else norm_error / norm <= 0.02
    )
    candidate["loss_and_norm_checks"] = checks
    candidate["passed"] &= all(checks.values())


def _qualify_fixture(trainer, parameters, initial, global_samples):
    samples, indices = _fixture(trainer, global_samples)
    entries, reference = {}, {}
    captured = _parity_pass(
        trainer,
        parameters,
        initial,
        samples,
        1,
        AnchorRandomness(entries, capture=True),
        ParityComparison(reference, capture=True),
    )
    replayed = _parity_pass(
        trainer,
        parameters,
        initial,
        samples,
        1,
        AnchorRandomness(entries, capture=False),
        ParityComparison(reference, exact=True),
    )
    _compare_losses(captured, replayed, exact=True)
    candidate = _candidate_pass(
        trainer, parameters, initial, samples, entries, reference
    )
    _compare_losses(captured, candidate, exact=False)
    return {
        **_fixture_identity(samples, indices, global_samples),
        "capture": captured,
        "transparent_replay": replayed,
        "candidate": candidate,
        "passed": replayed["passed"] and candidate["passed"],
    }


def _candidate_pass(trainer, parameters, initial, samples, entries, reference):
    model = trainer.accelerator.unwrap_model(trainer.model)
    compiled = trainer.cfg.get("npa_compile_mode", "off") == "rmsnorm"
    with candidate_compilation(model, compiled):
        report = _parity_pass(
            trainer,
            parameters,
            initial,
            samples,
            trainer.batch_size,
            AnchorRandomness(entries, capture=False),
            ParityComparison(reference),
        )
    report["compile_mode"] = "rmsnorm" if compiled else "off"
    return report


def _fixture_identity(samples, indices, global_samples):
    return {
        "global_samples": global_samples,
        "local_anchor_count": len(indices),
        "anchor_order_sha256": state_digest(indices),
        "transformed_inputs_sha256": state_digest(samples),
    }


def _publish_parity(trainer, fixtures):
    local = {"rank": trainer.accelerator.process_index, "fixtures": fixtures}
    reports = [None] * 4
    torch.distributed.all_gather_object(reports, local)
    passed = all(
        fixture["passed"] for report in reports for fixture in report["fixtures"]
    )
    body = json.dumps({"passed": passed, "ranks": reports}, allow_nan=False).encode()
    if trainer.accelerator.is_main_process:
        (Path(trainer.output_dir) / "parity.json").write_bytes(body)
    summary = {
        "passed": passed,
        "global_fixture_sizes": [96, 36],
        "reference_microbatch": 1,
        "candidate_microbatch": trainer.batch_size,
        "candidate_compile_mode": trainer.cfg.get("npa_compile_mode", "off"),
        "report_sha256": hashlib.sha256(body).hexdigest(),
        "random_tensors_exported": False,
        "peak_learning_rate": trainer.learning_rate,
    }
    if not passed:
        raise RuntimeError("microbatch parity rejected; thresholds were not relaxed")
    return summary


def verify_execution_parity(trainer):
    """Qualify execution against exact replay of real eager microbatch-one inputs.

    Args:
        trainer: Four-rank native DDP trainer before any timed optimizer update.
    Returns:
        Accepted fixture receipt; complete summaries are saved as parity.json.
    Raises:
        RuntimeError: Architecture, replay, numerical or distributed checks fail.
    """
    model = trainer.accelerator.unwrap_model(trainer.model)
    _validate_architecture(trainer, model)
    parameters = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    initial = {name: value.detach().cpu().clone() for name, value in parameters}
    original_digest = trainer._synchronized_digest()
    try:
        with _preserve_random_generators(), _preserve_model_attributes(model):
            fixtures = [
                _qualify_fixture(trainer, parameters, initial, count)
                for count in (96, 36)
            ]
    finally:
        _restore_parameters(parameters, initial)
        trainer.optimizer.zero_grad(set_to_none=True)
    if trainer._synchronized_digest() != original_digest:
        raise RuntimeError("parity altered the initial model state")
    return _publish_parity(trainer, fixtures)
