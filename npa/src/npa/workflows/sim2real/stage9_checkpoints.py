"""Bind canonical validation sweeps to the native trainer's saved checkpoint pool."""

from __future__ import annotations

from typing import Any


def validation_checkpoints(
    update: dict[str, Any], *, configured_updates: int
) -> list[dict[str, Any]]:
    """Validate every published periodic checkpoint without shortening the PPO pass.

    Args:
        update: Completed native trainer output with its durable checkpoint pool.
        configured_updates: Full requested update count for this training pass.
    Returns:
        Ordered checkpoint identities; older producers retain their final candidate.
    Raises:
        RuntimeError: A checkpoint identity, index, or pool is invalid or duplicated.
    """
    _validate_completed_pass(update, configured_updates)
    periodic = update.get("periodic_checkpoints")
    if periodic is None:
        periodic = [
            {
                "checkpoint_uri": update.get("checkpoint_path"),
                "training_iteration": configured_updates,
            }
        ]
    if not isinstance(periodic, list) or not periodic:
        raise RuntimeError("Stage 9 trainer published an empty checkpoint pool")
    checkpoints = [_checkpoint_identity(row) for row in periodic]
    indices = [row["training_iteration"] for row in checkpoints]
    uris = [row["checkpoint_uri"] for row in checkpoints]
    if len(set(indices)) != len(indices) or len(set(uris)) != len(uris):
        raise RuntimeError("Stage 9 trainer published duplicate checkpoint identities")
    return sorted(checkpoints, key=lambda row: row["training_iteration"])


def _validate_completed_pass(update, configured_updates):
    if "ppo_telemetry" not in update:
        return
    telemetry = update["ppo_telemetry"]
    observed = (
        telemetry.get("observed_iterations") if isinstance(telemetry, dict) else None
    )
    if type(observed) is not int or observed != configured_updates:
        raise RuntimeError("Stage 9 validation requires the complete measured PPO pass")


def _checkpoint_identity(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise RuntimeError("Stage 9 checkpoint identity must be an object")
    uri = row.get("checkpoint_uri")
    index = row.get("training_iteration")
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise RuntimeError("Stage 9 checkpoint requires a durable S3 identity")
    if type(index) is not int or index < 0:
        raise RuntimeError("Stage 9 checkpoint requires a literal native update index")
    if "checkpoint_completed_updates" in row or "checkpoint_iteration_semantics" in row:
        completed = row.get("checkpoint_completed_updates")
        if (
            type(completed) is not int
            or completed != index + 1
            or row.get("checkpoint_iteration_semantics") != "last_completed_zero_based"
        ):
            raise RuntimeError("Stage 9 checkpoint update accounting is inconsistent")
    return dict(row)


def validate_replay_checkpoint_pool(
    update: dict[str, Any], candidates: list[dict[str, Any]]
) -> None:
    """Require a replay to contain exactly the completed trainer's candidate pool.

    Args:
        update: Archived completed trainer output.
        candidates: Same-iteration durable validation candidates.
    Returns:
        None when every expected checkpoint has one bound validation candidate.
    Raises:
        RuntimeError: Replay candidates omit, duplicate, or change checkpoint lineage.
    """
    if "periodic_checkpoints" not in update:
        if len(candidates) != 1 or candidates[0].get("checkpoint_uri") != update.get(
            "checkpoint_path"
        ):
            raise RuntimeError("Stage 9 replay checkpoint lineage is inconsistent")
        return
    expected = validation_checkpoints(
        update, configured_updates=(update.get("ppo") or {}).get("iterations", 0)
    )
    identities = [
        (row["training_iteration"], row["checkpoint_uri"]) for row in expected
    ]
    actual = [_checkpoint_identity(row) for row in candidates]
    actual_identities = [
        (row["training_iteration"], row["checkpoint_uri"]) for row in actual
    ]
    if len(set(actual_identities)) != len(actual_identities) or sorted(
        actual_identities
    ) != sorted(identities):
        raise RuntimeError("Stage 9 replay checkpoint pool is partial or inconsistent")
