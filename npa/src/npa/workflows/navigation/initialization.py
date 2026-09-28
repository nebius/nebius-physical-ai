"""Load a sealed baseline into the real native learner before any policy update."""

from npa.workflows.navigation.artifacts import file_sha256


def initialize_runner(runner, recipe, source) -> dict:
    """Resume the exact native checkpoint, including optimizer and iteration state.

    Args:
        runner: Newly constructed RSL-RL learner matching the registered task.
        recipe: Sealed recipe with optional initial checkpoint.
        source: Verified local input bundle.
    Returns:
        Initialization provenance for the training record.
    Raises:
        ValueError: Checkpoint identity or native state is invalid.
        RuntimeError: Strict checkpoint loading fails.
    """
    initial = recipe.initial_checkpoint
    if initial is None:
        return {"mode": "from_scratch", "initial_iteration": 0}
    path = source / initial.file
    if file_sha256(path) != initial.sha256:
        raise ValueError("initial checkpoint changed before native loading")
    load_native_checkpoint(runner, path)
    return {
        "mode": "resume_native_checkpoint",
        "checkpoint_sha256": initial.sha256,
        "initial_iteration": runner.current_learning_iteration,
    }


def load_native_checkpoint(runner, path):
    """Restore complete native PPO state with safe tensor decoding and strict keys.

    Args:
        runner: Newly constructed native RSL-RL runner.
        path: Previously checksum-verified native checkpoint path.
    Returns:
        None after policy, optimizer and iteration are restored.
    Raises:
        ValueError: Iteration or complete native state is missing.
        RuntimeError: Safe decoding or strict native state loading fails.
    """
    import torch

    payload = torch.load(path, map_location="cuda:0", weights_only=True)
    if not isinstance(payload, dict) or type(payload.get("iter")) is not int:
        raise ValueError("navigation checkpoint lacks native PPO iteration state")
    if payload["iter"] < 0 or not runner.alg.load(payload, load_cfg=None, strict=True):
        raise ValueError(
            "navigation checkpoint did not restore complete native PPO state"
        )
    runner.current_learning_iteration = payload["iter"]
