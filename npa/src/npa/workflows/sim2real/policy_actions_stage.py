"""Stage 7 action-conditioned scenario component entrypoint."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def run_policy_actions_component_from_s3(
    *,
    train_envs_uri: str,
    output_uri: str,
    policy_image: str,
    limit: int,
    seed: int,
    run_id: str,
    rollout_count: int,
    steps_per_rollout: int,
) -> dict[str, Any]:
    """Run the swappable LeRobot policy container contract."""

    from npa.clients.storage import StorageClient
    from npa.workflows.sim2real_envgen import (
        EnvGenConfig,
        write_action_conditioned_envs,
    )

    config = EnvGenConfig(
        run_id=run_id or "sim2real-policy",
        output_uri=output_uri.rsplit("/actions/", 1)[0],
        env_count=max(limit, rollout_count),
        seed=seed,
    )
    with tempfile.TemporaryDirectory(prefix="npa-policy-actions-") as tmp:
        result = write_action_conditioned_envs(
            config,
            Path(tmp),
            policy_image=policy_image,
            limit=min(limit, rollout_count),
            train_envs_uri=train_envs_uri,
            actions_uri=output_uri.rsplit("/", 1)[0] + "/",
        )
    result_local = Path("/tmp/policy-actions-result.json")
    result_local.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    StorageClient.from_environment().upload_file(str(result_local), output_uri)
    return result
