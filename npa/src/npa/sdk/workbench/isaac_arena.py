"""SDK surface for Isaac Lab-Arena evaluation."""

from typing import Any

from npa.workbench.isaac_arena import IsaacArenaRequest, evaluate as _evaluate


def evaluate(
    *,
    output_path: str,
    environment: str = "cube_goal_pose",
    policy_type: str = "zero_action",
    input_path: str = "",
    num_episodes: int = 1,
    num_envs: int = 1,
    seed: int = 42,
    embodiment: str = "",
    object_name: str = "",
    record_video: bool = False,
    run_id: str = "",
    runtime_image: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the same evaluation implementation used by CLI and workflows."""

    return _evaluate(
        IsaacArenaRequest(
            output_path=output_path,
            environment=environment,
            policy_type=policy_type,
            input_path=input_path,
            num_episodes=num_episodes,
            num_envs=num_envs,
            seed=seed,
            embodiment=embodiment,
            object_name=object_name,
            record_video=record_video,
            run_id=run_id,
            runtime_image=runtime_image,
            dry_run=dry_run,
        )
    )


__all__ = ["IsaacArenaRequest", "evaluate"]
