"""Independently replay SGD in scalar CPU arithmetic and reject invalid evidence."""

import math

from regression_contract import IMAGE, INITIAL, RECIPE, TORCH_VERSION, _close, _dataset


def _loss_gradient(parameters, rows, targets):
    gradient = [0.0] * 9
    squared_error = 0.0
    for row, target in zip(rows, targets, strict=True):
        error = sum(parameters[index] * row[index] for index in range(8))
        error += parameters[8] - target
        squared_error += error * error
        for index, value in enumerate(row + [1.0]):
            gradient[index] += 2 * error * value / len(rows)
    return squared_error / len(rows), gradient


def _oracle():
    rows, targets = _dataset(RECIPE["seed"], RECIPE["samples"])
    parameters, momentum, journal = INITIAL.copy(), [0.0] * 9, []
    for step in range(1, RECIPE["steps"] + 1):
        loss, gradient = _loss_gradient(parameters, rows, targets)
        momentum = [
            RECIPE["momentum"] * old + new
            for old, new in zip(momentum, gradient, strict=True)
        ]
        movement = [-RECIPE["learning_rate"] * value for value in momentum]
        parameters = [
            old + change for old, change in zip(parameters, movement, strict=True)
        ]
        journal.append(
            {
                "optimizer_step": step,
                "loss": loss,
                "gradient_norm": math.sqrt(sum(value * value for value in gradient)),
                "parameter_delta": math.sqrt(sum(value * value for value in movement)),
            }
        )
    return parameters, momentum, journal


def _validate_binding(manifest, metrics, checkpoint, expected):
    for document in (manifest, metrics, checkpoint):
        for key, value in expected.items():
            if document.get(key) != value:
                raise ValueError(f"artifact binding differs: {key}")
    if metrics.get("recipe") != RECIPE or checkpoint.get("recipe") != RECIPE:
        raise ValueError("training recipe differs")
    if metrics.get("image") != IMAGE or metrics.get("torch_version") != TORCH_VERSION:
        raise ValueError("runtime image or Torch version differs")
    if metrics.get("device_type") != "cuda" or not metrics.get("device_name"):
        raise ValueError("CUDA device evidence missing")
    capability = metrics.get("compute_capability")
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("CUDA compute capability missing")
    if any(type(value) is not int or value < 0 for value in capability):
        raise ValueError("CUDA compute capability invalid")


def _validate_journal(journal, expected):
    if len(journal) != RECIPE["steps"]:
        raise ValueError("optimizer journal is incomplete")
    for actual, reference in zip(journal, expected, strict=True):
        if type(actual.get("optimizer_step")) is not int:
            raise ValueError("optimizer step must be an integer")
        for key, value in reference.items():
            _close(actual.get(key), value, f"journal {key}")
        for key in ("elapsed_seconds", "samples_per_second"):
            if not math.isfinite(actual.get(key, float("nan"))) or actual[key] <= 0:
                raise ValueError(f"invalid measured {key}")
        _close(
            actual["samples_per_second"] * actual["elapsed_seconds"],
            RECIPE["samples"],
            "measured throughput",
        )


def _validate_parameters(checkpoint, parameters, momentum):
    for key, expected in (
        ("parameters", parameters),
        ("momentum", momentum),
        ("initial_parameters", INITIAL),
    ):
        actual = checkpoint.get(key, [])
        if len(actual) != len(expected):
            raise ValueError(f"checkpoint {key} shape differs")
        for observed, reference in zip(actual, expected, strict=True):
            _close(observed, reference, f"checkpoint {key}")
    if checkpoint.get("optimizer_step") != RECIPE["steps"]:
        raise ValueError("checkpoint optimizer step differs")


def _validate_artifacts(manifest, metrics, checkpoint, expected):
    _validate_binding(manifest, metrics, checkpoint, expected)
    parameters, momentum, journal = _oracle()
    _validate_journal(metrics.get("journal", []), journal)
    _validate_parameters(checkpoint, parameters, momentum)
    rows, targets = _dataset(RECIPE["seed"] + 1, RECIPE["heldout_samples"])
    heldout_loss, _ = _loss_gradient(parameters, rows, targets)
    _close(metrics.get("heldout_loss"), heldout_loss, "held-out loss")
    rows, targets = _dataset(RECIPE["seed"], RECIPE["samples"])
    final_loss, _ = _loss_gradient(parameters, rows, targets)
    _close(metrics.get("final_train_loss"), final_loss, "final training loss")
    if not 0 < final_loss < journal[0]["loss"] * 0.02:
        raise ValueError("training loss did not meet the frozen improvement gate")
    return {
        "optimizer_steps_verified": len(journal),
        "heldout_loss": heldout_loss,
        "initial_train_loss": journal[0]["loss"],
        "final_train_loss": final_loss,
        "training_loss_ratio": final_loss / journal[0]["loss"],
        "optimizer_momentum_verified": True,
        "method": "independent scalar CPU analytic-gradient SGD replay",
    }
