"""Fit and export the fixed linear and exact-GELU conditional gates."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifact import ARTIFACT_SCHEMA, GateArtifact, predict_scaled
from .fit_data import FitData


@dataclass(frozen=True)
class FitResult:
    """Hold fitted models, scalers, and selection evidence.

    Args:
        models: Torch linear-control and MLP modules.
        scalers: Exact float32 fit-only scalers.
        logs: Per-model epoch metric rows.
        selection: Earliest strict validation-minimum epoch per model.
    Returns:
        None.
    Raises:
        None.
    """

    models: dict[str, Any]
    scalers: dict[str, np.ndarray]
    logs: dict[str, list[dict[str, float | int]]]
    selection: dict[str, dict[str, float | int]]


def fit_scalers(data: FitData) -> dict[str, np.ndarray]:
    """Fit population scalers on fit rows only using float64 reductions.

    Args:
        data: Canonical validated TRAIN data.
    Returns:
        Float32 input and target scaler arrays.
    Raises:
        ValueError: If fit and validation memberships are incomplete.
    """
    data.validated()
    selected = data.partitions == "fit"
    if not selected.any() or not (data.partitions == "validation").any():
        raise ValueError("fit and validation partitions are required")
    features = data.features[selected].astype(np.float64)
    target = (data.continue_error - data.fresh_error)[selected]
    input_std = np.maximum(features.std(axis=0), 1e-6)
    target_std = np.maximum(np.asarray([target.std()]), 1e-6)
    return {
        "input_mean": features.mean(axis=0).astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "target_mean": np.asarray([target.mean()], np.float32),
        "target_std": target_std.astype(np.float32),
    }


def make_models(torch: Any) -> dict[str, Any]:
    """Construct the qualified seed-1033 Torch model pair.

    Args:
        torch: Imported Torch module supplied by the execution environment.
    Returns:
        Linear-control and exact-GELU MLP modules.
    Raises:
        RuntimeError: If Torch cannot enable deterministic execution.
    """
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1033)
    nn = torch.nn
    return {
        "linear_control": nn.Linear(1233, 1),
        "mlp": nn.Sequential(
            nn.Linear(1233, 256),
            nn.GELU(approximate="none"),
            nn.Linear(256, 64),
            nn.GELU(approximate="none"),
            nn.Linear(64, 1),
        ),
    }


def fit_models(torch: Any, data: FitData) -> FitResult:
    """Fit both preregistered models with fixed AdamW and early selection.

    Args:
        torch: Imported Torch module supplied by the execution environment.
        data: Canonical TRAIN-only fit data.
    Returns:
        Fitted models, scalers, logs, and selected epochs.
    Raises:
        ValueError: If input rows violate the fit contract.
    """
    _require_membership(data)
    scalers = fit_scalers(data)
    x, y = scaled_rows(data, scalers)
    models, logs, selection = make_models(torch), {}, {}
    for name, model in models.items():
        logs[name], selection[name] = _train_model(torch, model, x, y, data.partitions)
    return FitResult(models, scalers, logs, selection)


def _require_membership(data: FitData) -> None:
    data.validated()
    expected = {"fit": (1_920, 160), "validation": (240, 20)}
    for name, (rows, episodes) in expected.items():
        selected = data.partitions == name
        observed = data.episodes[selected]
        if selected.sum() != rows or len(np.unique(observed)) != episodes:
            raise ValueError("fit membership differs from fixed 160/20 split")
        counts = np.unique(observed, return_counts=True)[1]
        if not np.array_equal(counts, np.full(episodes, 12)):
            raise ValueError("fit rows per episode differ")
    fit = set(data.episodes[data.partitions == "fit"].tolist())
    validation = set(data.episodes[data.partitions == "validation"].tolist())
    if fit & validation or len(fit | validation) != 180:
        raise ValueError("fit membership episodes must be globally unique")


def scaled_rows(
    data: FitData, scalers: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Scale canonical features and error-difference targets.

    Args:
        data: Validated fit data.
        scalers: Fit-only population scalers.
    Returns:
        Float32 scaled feature matrix and column target.
    Raises:
        ValueError: If required scaler arrays are absent.
    """
    x = ((data.features - scalers["input_mean"]) / scalers["input_std"]).astype(
        np.float32
    )
    target = (data.continue_error - data.fresh_error).astype(np.float32)
    y = ((target - scalers["target_mean"][0]) / scalers["target_std"][0]).astype(
        np.float32
    )
    return x, y[:, None]


def export_result(result: FitResult, metadata: dict[str, Any]) -> GateArtifact:
    """Convert fitted Torch state into the exact NumPy artifact schema.

    Args:
        result: Completed fit result.
        metadata: Public provenance and selection metadata.
    Returns:
        Validated NumPy-only gate artifact.
    Raises:
        ValueError: If metadata schema or export parity differs.
    """
    if metadata.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("artifact metadata schema differs")
    expected_selection = {
        name: row.get("selected_epoch") for name, row in result.selection.items()
    }
    if metadata.get("selection") != expected_selection:
        raise ValueError("artifact selection differs from fitted result")
    arrays = {
        name: np.ascontiguousarray(value) for name, value in result.scalers.items()
    }
    for model_name, model in result.models.items():
        for name, tensor in model.state_dict().items():
            arrays[f"{model_name}.{name}"] = np.ascontiguousarray(
                tensor.detach().cpu().numpy()
            )
    artifact = GateArtifact(arrays, metadata).validated()
    _verify_export(result, artifact)
    return artifact


def _train_model(
    torch: Any, model: Any, x: np.ndarray, y: np.ndarray, partitions: np.ndarray
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    optimizer = _optimizer(torch, model)
    fit, validation = (
        np.flatnonzero(partitions == "fit"),
        np.flatnonzero(partitions == "validation"),
    )
    generator = torch.Generator(device="cpu").manual_seed(1729)
    best, stale, logs = None, 0, []
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    fit_indices = torch.from_numpy(fit)
    for epoch in range(1, 201):
        order = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        _train_epoch(torch, model, optimizer, tx, ty, order)
        loss = _validation_loss(torch, model, tx, ty, validation)
        logs.append({"epoch": epoch, "validation_mean_huber": loss})
        best, stale = _select_best(model, best, stale, loss, epoch)
        if stale >= 20:
            break
    assert best is not None
    model.load_state_dict(best[2])
    return logs, {
        "selected_epoch": best[1],
        "validation_mean_huber": best[0],
        "epochs_executed": len(logs),
    }


def _optimizer(torch: Any, model: Any) -> Any:
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def _validation_loss(
    torch: Any, model: Any, x: Any, y: Any, indices: np.ndarray
) -> float:
    model.eval()
    with torch.no_grad():
        value = torch.nn.functional.huber_loss(model(x[indices]), y[indices], delta=1.0)
    return float(value.item())


def _select_best(
    model: Any, best: Any, stale: int, loss: float, epoch: int
) -> tuple[Any, int]:
    if best is None or loss < best[0]:
        return (loss, epoch, copy.deepcopy(model.state_dict())), 0
    return best, stale + 1


def _train_epoch(
    torch: Any, model: Any, optimizer: Any, x: Any, y: Any, order: np.ndarray
) -> None:
    model.train()
    for start in range(0, len(order), 64):
        index = order[start : start + 64]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.huber_loss(model(x[index]), y[index], delta=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _verify_export(result: FitResult, artifact: GateArtifact) -> None:
    sample = np.zeros((2, 1233), np.float32)
    scaled = (
        (sample - artifact.arrays["input_mean"]) / artifact.arrays["input_std"]
    ).astype(np.float32)
    for name, model in result.models.items():
        torch_values = (
            model(next(model.parameters()).new_tensor(scaled)).detach().cpu().numpy()
        )
        numpy_values = predict_scaled(artifact, sample, name)
        delta = torch_values.astype(np.float64) - numpy_values.astype(np.float64)
        if np.max(np.abs(delta)) > 1e-5 or np.sqrt(np.mean(np.square(delta))) > 1e-6:
            raise ValueError(f"Torch-to-NumPy export parity differs: {name}")
