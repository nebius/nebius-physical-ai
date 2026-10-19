"""Fit and evaluate a small Cartesian reaching policy on generated LeRobot episodes."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pyarrow.parquet as pq

from npa.workbench.dataset.storage import read_bytes_uri, read_json_uri, write_bytes_uri
from .contracts import digest


def episodes(request):
    """Load only the request's hash-bound partition from LeRobot Parquet.

    Args:
        request: Batch request with an immutable partition reference.
    Returns:
        Mapping of episode indices to state and action arrays.
    Raises:
        ValueError: The partition changed or is not the generated reference dataset.
    """
    partition = read_json_uri(request["dataset"]["uri"])
    if digest(partition) != request["dataset"]["sha256"]:
        raise ValueError("reference partition digest mismatch")
    rows = {}
    for episode in partition["episodes"]:
        root = episode["dataset_uri"]
        info = read_json_uri(root + "/meta/info.json")
        if info.get("robot_type") != "npa_generated_planar_reacher":
            raise ValueError("reference worker accepts generated planar data only")
        index = episode["episode_index"]
        table = pq.read_table(
            io.BytesIO(read_bytes_uri(root + "/data/chunk-000/file-000.parquet")),
            filters=[("episode_index", "=", index)],
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=float)
        actions = np.asarray(table["action"].to_pylist(), dtype=float)
        if states.shape != (24, 4) or actions.shape != (24, 2):
            raise ValueError("invalid reference episode shape")
        rows[index] = (states, actions)
    return rows


def load_weights(identity):
    """Read safe numeric checkpoint bytes and verify their recorded digest.

    Args:
        identity: Checkpoint URI and SHA256, or None for an untrained policy.
    Returns:
        Finite 3-by-2 policy weight array.
    Raises:
        ValueError: Checkpoint bytes, shape, or values are invalid.
    """
    if identity is None:
        return np.zeros((3, 2))
    content = read_bytes_uri(identity["uri"])
    if hashlib.sha256(content).hexdigest() != identity["sha256"]:
        raise ValueError("reference checkpoint digest mismatch")
    weights = np.load(io.BytesIO(content), allow_pickle=False)
    if weights.shape != (3, 2) or not np.isfinite(weights).all():
        raise ValueError("invalid reference checkpoint weights")
    return weights


def train(request, records, weights):
    """Take one damped least-squares behavior-cloning update on training data.

    Args:
        request: Pretrain or finetune request bound to the training partition.
        records: Selected episode state/action arrays.
        weights: Previous candidate's weights, or zero initialization.
    Returns:
        Checkpoint identity and measured training loss before and after the update.
    Raises:
        ValueError: A non-training partition is supplied.
    """
    if request["partition"] != "train":
        raise ValueError("reference training cannot consume holdouts")
    states = np.concatenate([record[0] for record in records.values()])
    targets = np.concatenate([record[1] for record in records.values()])
    if request["stage"] == "finetune":
        # The refinement task halves actuator gain; relabel only training states.
        targets = targets * 2
    features = np.column_stack([states[:, 2:] - states[:, :2], np.ones(len(states))])
    optimum = np.linalg.lstsq(features, targets, rcond=None)[0]
    updated = weights + 0.4 * (optimum - weights)
    if np.max(np.abs(updated - weights)) < 1e-12:
        raise ValueError("reference optimizer converged without a passing gate")
    losses = [float(np.mean((features @ w - targets) ** 2)) for w in (weights, updated)]
    buffer = io.BytesIO()
    np.save(buffer, updated, allow_pickle=False)
    content = buffer.getvalue()
    uri = request["result_uri"].rsplit("/", 1)[0] + "/checkpoint.npy"
    write_bytes_uri(uri, content)
    return {"uri": uri, "sha256": hashlib.sha256(content).hexdigest()}, losses


def rollout(weights, start, target, gain):
    """Execute a learned policy for a sixteen-step analytical reaching task.

    Args:
        weights: Learned Cartesian policy weights.
        start: Initial end-effector coordinates in meters.
        target: Target coordinates in meters.
        gain: Actuator displacement per unit action.
    Returns:
        Actual positions, final distance, and success within four centimeters.
    Raises:
        None.
    """
    position = np.asarray(start, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    positions = [position.tolist()]
    for _ in range(16):
        action = np.r_[target - position, 1.0] @ weights
        position += gain * np.clip(action, -0.25, 0.25)
        positions.append(position.tolist())
    distance = float(np.linalg.norm(target - position))
    return {
        "positions": positions,
        "target": target.tolist(),
        "distance": distance,
        "success": distance <= 0.04,
    }


def evaluate(request, records, weights):
    """Measure success on held-out goals and four fixed initial-state variations.

    Args:
        request: Evaluation or terminal policy-test request.
        records: Held-out source episodes.
        weights: Checkpoint weights whose bytes were verified.
    Returns:
        Per-system counts and every measured trajectory.
    Raises:
        ValueError: A training partition is supplied.
    """
    if request["partition"] == "train":
        raise ValueError("reference evaluation requires a holdout")
    gains = {"nominal": 1.0, "perturbed": 0.85}
    if request["stage"] != "evaluate-pretrain":
        gains = {"slow-actuator": 0.5}
    systems, trajectories = {}, {}
    offsets = ((0, 0), (0.03, 0), (0, -0.03), (-0.03, 0.03))
    for system, gain in gains.items():
        trials = []
        for index, (states, _) in records.items():
            for variation, offset in enumerate(offsets):
                trial = rollout(weights, states[0, :2] + offset, states[0, 2:], gain)
                trials.append({**trial, "episode": index, "variation": variation})
        systems[system] = {
            "successes": sum(t["success"] for t in trials),
            "trials": len(trials),
        }
        trajectories[system] = trials
    return systems, trajectories
