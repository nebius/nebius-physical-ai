"""mujoco workbench SDK; mirrors the npa mujoco CLI stage.

Thin, real wrapper over the pipeline stage function in
``npa.workflows.byof.mujoco_pipeline``: no ``NotImplementedError``, no CLI
indirection.
"""

from __future__ import annotations

from typing import Any, Mapping


def run(
    *,
    task: str,
    policy: str,
    output_uri: str,
    episodes: int = 10,
    seed: int = 0,
    max_steps: int = 500,
) -> Mapping[str, Any]:
    """Run N episodes of a scripted policy; write trajectories + metrics."""
    from npa.workflows.byof.mujoco_pipeline import run as _run

    return _run(
        task=task,
        policy=policy,
        output_uri=output_uri,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )


__all__ = ["run"]
