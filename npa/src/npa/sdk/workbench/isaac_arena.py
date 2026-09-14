"""SDK surface for Isaac Lab-Arena evaluation."""

from typing import Any

from npa.workbench.isaac_arena import (
    IsaacArenaRequest,
    capabilities as _capabilities,
    evaluate as _evaluate,
)


def capabilities() -> dict[str, Any]:
    """Return the pinned upstream surface and honest NPA support status.

    Args:
        None.

    Returns:
        The capability inventory with implementation and qualification scope.

    Raises:
        None.
    """

    return _capabilities()


def evaluate(
    *,
    output_path: str,
    environment: str = "cube_goal_pose",
    policy_type: str = "zero_action",
    input_path: str = "",
    execution_device: str = "cuda:0",
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
    """Run the same evaluation implementation used by CLI and workflows.

    Args:
        output_path: Local result directory or S3 output prefix.
        environment: Registered Arena environment name.
        policy_type: ``zero_action``, ``replay``, or ``rsl_rl`` adapter.
        input_path: Local or S3 replay file/checkpoint input, when required.
        execution_device: ``cuda:0`` or diagnostic-only ``cpu`` physics device.
        num_episodes: Number of scored episodes; replay requires one.
        num_envs: Number of simulation environments; replay requires one.
        seed: Upstream random seed.
        embodiment: Optional registered robot override.
        object_name: Optional task-compatible object override.
        record_video: Require viewport video and its visual qualification.
        run_id: Optional identifier binding the resulting evidence.
        runtime_image: Optional exact image identity recorded in the manifest.
        dry_run: Build the request without executing the simulator.

    Returns:
        The result manifest and publication location, or a dry-run request.

    Raises:
        IsaacArenaError: The request, runtime result, or evidence is invalid.
        OSError: A local input/output operation fails.
    """

    return _evaluate(
        IsaacArenaRequest(
            output_path=output_path,
            environment=environment,
            policy_type=policy_type,
            input_path=input_path,
            execution_device=execution_device,
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


__all__ = ["IsaacArenaRequest", "capabilities", "evaluate"]
