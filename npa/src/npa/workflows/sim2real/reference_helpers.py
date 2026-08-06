"""Small reference-path helpers shared by legacy and staged Sim2Real engines.

These helpers implement deterministic CPU/reference behavior only.  Keeping
them outside the orchestration modules prevents the already-large engines from
absorbing more leaf-level data and scoring utilities.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, cast

from npa.workflows.sim2real.constants import CORRECTIVE_TARGETS
from npa.workflows.sim2real.models import Sim2RealLoopError
from npa.workflows.sim2real.utils import _write_json_artifact


def _write_env_manifest(root: Path, *, count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    envs = [
        {
            "env_id": f"env-{index:04d}",
            "seed": rng.randrange(1, 2**31 - 1),
            "asset_ref": f"asset-{index:04d}",
            "physics": {
                "friction": round(0.5 + rng.random() * 0.5, 4),
                "mass_scale": round(0.85 + rng.random() * 0.3, 4),
                "lighting": round(0.4 + rng.random() * 0.5, 4),
            },
        }
        for index in range(count)
    ]
    return _write_json_artifact(
        root / "manifest.json",
        {"schema": "npa.sim2real.env_manifest.v1", "stage": 4, "envs": envs},
    )


def _write_train_heldout_split(
    root: Path,
    *,
    raw_envs: dict[str, Any],
    train_count: int,
    heldout_count: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    envs = list(raw_envs["payload"]["envs"])
    expected = train_count + heldout_count
    if len(envs) != expected:
        raise Sim2RealLoopError(
            f"raw env count {len(envs)} must equal train+heldout count {expected}"
        )
    rng = random.Random(seed)
    rng.shuffle(envs)
    train = envs[:train_count]
    heldout = envs[train_count : train_count + heldout_count]
    if len(train) != train_count or len(heldout) != heldout_count:
        raise Sim2RealLoopError("train/heldout split did not preserve requested counts")
    train_record = _write_json_artifact(
        root / "train" / "manifest.json",
        {
            "schema": "npa.sim2real.env_split.v1",
            "stage": 5,
            "split": "train",
            "envs": train,
        },
    )
    heldout_record = _write_json_artifact(
        root / "heldout" / "manifest.json",
        {
            "schema": "npa.sim2real.env_split.v1",
            "stage": 5,
            "split": "heldout",
            "envs": heldout,
        },
    )
    return train_record, heldout_record


def _tags_for_quality(quality: float, *, step: int) -> list[str]:
    if quality < 0.45:
        return ["missed_target", "unstable"] if step % 2 == 0 else ["late_grasp"]
    if quality < 0.65:
        return ["minor_alignment"] if step % 2 == 0 else ["late_grasp"]
    if quality < 0.8:
        return ["minor_alignment"]
    return ["ok"]


def _critique_for_tags(tags: list[str], *, quality: float) -> str:
    if tags == ["ok"]:
        return f"Step is stable; estimated rollout quality {quality:.2f}."
    corrections = [
        CORRECTIVE_TARGETS.get(tag, CORRECTIVE_TARGETS["minor_alignment"])[
            "nl_correction"
        ]
        for tag in tags
    ]
    return " ".join(str(correction) for correction in corrections)


def _merge_targets(tags: list[str]) -> dict[str, Any]:
    corrections = [
        CORRECTIVE_TARGETS.get(tag, CORRECTIVE_TARGETS["minor_alignment"])
        for tag in tags
    ]
    action_dim = max(len(item["action_delta"]) for item in corrections)
    merged = [0.0 for _ in range(action_dim)]
    for item in corrections:
        for index, value in enumerate(item["action_delta"]):
            merged[index] += float(cast(Any, value)) / float(len(corrections))
    return {
        "nl_correction": " ".join(str(item["nl_correction"]) for item in corrections),
        "action_delta": [round(value, 6) for value in merged],
    }


def _signal_mean_reward(signal: dict[str, Any]) -> float:
    steps = signal.get("per_step") or []
    return sum(float(step["reward"]) for step in steps) / float(len(steps))


def _heldout_env_score(
    distance_score: float, reward_score: float, *, env_success: bool
) -> float:
    """Map per-env distance/reward to a continuous held-out score."""

    quality = max(0.0, min(1.0, 0.7 * distance_score + 0.3 * reward_score))
    if env_success:
        return round(0.75 + 0.25 * quality, 6)
    return round(0.6 * quality, 6)


def _signal_diversity_report(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether VLM-to-RL credit varies across rollouts."""

    scores = [round(float(signal.get("score") or 0.0), 4) for signal in signals]
    mean_rewards = [round(_signal_mean_reward(signal), 4) for signal in signals]
    distinct_scores = sorted(set(scores))
    distinct_rewards = sorted(set(mean_rewards))
    total = len(signals)
    coherent = total > 1 and len(distinct_scores) > 1 and len(distinct_rewards) > 1
    return {
        "total_rollouts": total,
        "distinct_scores": len(distinct_scores),
        "distinct_mean_rewards": len(distinct_rewards),
        "score_values": distinct_scores,
        "mean_reward_values": distinct_rewards,
        "coherent": coherent,
        "degenerate": not coherent,
    }


def _write_ppm(path: Path, *, red: int, green: int, blue: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 32
    height = 32
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixel = bytes(
        [max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue))]
    )
    path.write_bytes(header + pixel * width * height)
