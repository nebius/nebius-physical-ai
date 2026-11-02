"""Preserve learned Franka observation statistics through PPO and exact checkpoint loads."""

from __future__ import annotations

from typing import Any

_METADATA_KEY = "npa_stock_franka_normalization"


def _normalizer(actor: Any) -> Any:
    from npa.workflows.sim2real.isaac_franka_policy import stock_franka_policy_contract

    observation_dim = stock_franka_policy_contract()["observation_dim"]
    normalizer = getattr(actor, "obs_normalizer", None)
    if (
        type(normalizer).__name__ != "EmpiricalNormalization"
        or normalizer.mean.numel() != observation_dim
        or type(normalizer.count.item()) is not int
        or normalizer.count.item() <= 0
    ):
        raise RuntimeError(
            "stock-Franka normalization lacks initialized training statistics"
        )
    return normalizer


def checkpoint_normalization_metadata(actor: Any) -> dict[str, Any]:
    """Bind the actual observation statistics and update bound to a checkpoint.

    Args:
        actor: Native stock-Franka actor with initialized empirical normalization.
    Returns:
        Literal checkpoint metadata for frozen or explicitly unbootstrapped statistics.
    Raises:
        RuntimeError: Statistics are uninitialized or the update bound is inconsistent.
    """
    normalizer = _normalizer(actor)
    count, until = normalizer.count.item(), normalizer.until
    if until is not None and (type(until) is not int or until != count):
        raise RuntimeError(
            "stock-Franka normalization changed after its training freeze"
        )
    return {
        "schema": "npa.sim2real.franka_normalization.v1",
        "scope": "sealed_training_only",
        "gold_used": False,
        "frozen": until is not None,
        "count": count,
        "until": until,
    }


def freeze_training_normalization(actor: Any) -> dict[str, Any]:
    """Freeze native empirical statistics after the complete learned-actor handoff.

    Args:
        actor: Native stock-Franka actor fitted only on consumed training scenarios.
    Returns:
        The literal normalization state bound to every subsequent saved checkpoint.
    Raises:
        RuntimeError: Training statistics have not been initialized.
    """
    normalizer = _normalizer(actor)
    normalizer.until = normalizer.count.item()
    return checkpoint_normalization_metadata(actor)


def restore_checkpoint_normalization(actor: Any, infos: Any) -> None:
    """Restore the native update bound only from exact checkpoint-bound statistics.

    Args:
        actor: Native actor after loading its saved observation-statistic tensors.
        infos: Metadata returned by the native runner's checkpoint load API.
    Returns:
        None; the normalizer receives the verified saved update bound.
    Raises:
        RuntimeError: Metadata is missing, coerced, or differs from loaded statistics.
    """
    normalizer = _normalizer(actor)
    metadata = infos.get(_METADATA_KEY) if isinstance(infos, dict) else None
    if not isinstance(metadata, dict) or type(metadata.get("frozen")) is not bool:
        raise RuntimeError(
            "checkpoint lacks literal stock-Franka normalization metadata"
        )
    count = normalizer.count.item()
    expected = {
        "schema": "npa.sim2real.franka_normalization.v1",
        "scope": "sealed_training_only",
        "gold_used": False,
        "frozen": metadata["frozen"],
        "count": count,
        "until": count if metadata["frozen"] else None,
    }
    if metadata != expected or any(
        type(metadata[key]) is not type(value) for key, value in expected.items()
    ):
        raise RuntimeError(
            "checkpoint normalization differs from its loaded training statistics"
        )
    normalizer.until = expected["until"]
