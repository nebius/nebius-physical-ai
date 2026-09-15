"""Join separately rendered Franka capture cases while verifying policy and reset pairing."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from npa.workflows.lerobot_transfer_data import file_sha256, write_json
from npa.workflows.franka_rl_embodiments import validate_capture_embodiment


def merge_captures(evaluation: Path, recipe: dict) -> None:
    """Flatten capture cases into a LeRobot-compatible dataset with per-episode lineage.

    Args:
        evaluation: Completed evaluation and independently captured arm/condition cases.
        recipe: Sealed capture grid and expected physical reset streams.
    Returns:
        None.
    Raises:
        ValueError: Case coverage, policy identity, or paired initial states disagree.
        OSError: Capture arrays or metadata cannot be read or moved.
    """
    output = evaluation / "trajectories"
    output.mkdir()
    rows, metadata, identity = [], None, None
    for arm in recipe["visual_eval"]["arms"]:
        expected = file_sha256(evaluation / ("initial.pt" if arm == "initial" else "selected.pt"))
        for condition in recipe["conditions"]:
            source = evaluation / "captures" / f"{arm}-{condition}"
            metadata = json.loads((source / "meta.json").read_text())
            validate_capture_embodiment(metadata, recipe)
            if identity is not None and identity != metadata.get("embodiment"):
                raise ValueError("Capture cases contain different robot embodiments")
            identity = metadata.get("embodiment")
            if metadata["checkpoint_sha256"] != expected or metadata["num_episodes"] != recipe["capture_episodes"]:
                raise ValueError("Franka capture checkpoint or episode coverage differs from protocol")
            for index, row in enumerate(metadata["episode_results"]):
                if (row["arm"], row["condition"], row["capture_index"]) != (arm, condition, index):
                    raise ValueError("Franka capture case identity differs from protocol")
                shutil.move(source / f"episode_{index:06d}", output / f"episode_{len(rows):06d}")
                row["applied_physics"] = metadata["physics"]
                rows.append(row)
    for index in range(recipe["capture_episodes"]):
        hashes = {row["initial_state_sha256"] for row in rows if row["capture_index"] == index}
        if len(hashes) != 1:
            raise ValueError("Franka visual comparison did not begin from paired physical states")
    metadata.update(episode_results=rows, num_episodes=len(rows), multiple_checkpoints=True,
                    checkpoint_sha256=None, episode_lengths=[row["length"] for row in rows],
                    total_frames=sum(row["length"] for row in rows),
                    rgb_frame_count=sum(row["length"] for row in rows))
    write_json(output / "meta.json", metadata)
