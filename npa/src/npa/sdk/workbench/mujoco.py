"""mujoco workbench SDK; mirrors the npa mujoco CLI stage.

Thin, real wrapper over the pipeline stage function in
``npa.workflows.byof.mujoco_pipeline``: no ``NotImplementedError``, no CLI
indirection.
"""

from __future__ import annotations

from typing import Any, Mapping

from npa.cli.path_contract import PathContractError, validate_write_path


def run(
    *,
    task: str,
    policy: str,
    output_path: str = "",
    output_uri: str | None = None,
    episodes: int = 10,
    seed: int = 0,
    max_steps: int = 500,
) -> Mapping[str, Any]:
    """Run N episodes of a scripted policy; write trajectories + metrics."""
    from npa.workflows.byof.mujoco_pipeline import run as _run

    if output_path and output_uri and output_path != output_uri:
        raise ValueError("output_path and compatibility output_uri must match")
    try:
        output_path = validate_write_path(
            output_path or output_uri or "",
            tool="mujoco SDK run",
            option="output_path",
            required=True,
        )
    except PathContractError as exc:
        raise ValueError(str(exc)) from exc
    return _run(
        task=task,
        policy=policy,
        output_path=output_path,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )


__all__ = ["run"]
