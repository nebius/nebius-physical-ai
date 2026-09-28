"""Retain native reference checkpoints, learning curves and measured GPU throughput."""

import time

from npa.workflows.navigation.artifacts import file_sha256, write_json


def learn(runner, env, recipe, output):
    """Run native PPO while retaining the actual pre-update checkpoint and timing.

    Args:
        runner: Initialized or resumed native RSL-RL runner.
        env: Actual vectorized Isaac environment.
        recipe: Explicit robot population and learning iteration settings.
        output: Artifact directory including native TensorBoard checkpoint logs.
    Returns:
        Measured training throughput, memory and initial checkpoint identity.
    Raises:
        RuntimeError: Native learning or scalar-log extraction fails.
    """
    import torch

    reference = output / "reference_checkpoint.pt"
    _save_initial_checkpoint(runner, reference)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    runner.learn(num_learning_iterations=recipe.iterations, init_at_random_ep_len=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    transitions = (
        recipe.iterations * runner.cfg["num_steps_per_env"] * env.unwrapped.num_envs
    )
    if recipe.adapter_module == "npa.workflows.navigation.reference":
        _learning_curves(runner, output)
    return {
        "reference_checkpoint_sha256": file_sha256(reference),
        "learning_wall_seconds": elapsed,
        "control_transitions": transitions,
        "control_transitions_per_second": transitions / elapsed,
        "physics_steps_per_control": env.unwrapped.cfg.decimation,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _save_initial_checkpoint(runner, path):
    import torch

    # RSL-RL 5.0.1 creates logger.writer inside learn(), so runner.save() is
    # unavailable here. Preserve its exact native checkpoint payload instead.
    payload = runner.alg.save()
    payload.update(iter=runner.current_learning_iteration, infos=None)
    torch.save(payload, path)


def _learning_curves(runner, output):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # RSL-RL stops external loggers but leaves TensorBoard's async writer open.
    runner.logger.writer.flush()
    runner.logger.writer.close()
    events = EventAccumulator(str(output / "checkpoints"), size_guidance={"scalars": 0})
    events.Reload()
    curves = {
        tag: [
            {"step": row.step, "value": row.value, "wall_time": row.wall_time}
            for row in events.Scalars(tag)
        ]
        for tag in events.Tags()["scalars"]
    }
    if not curves:
        raise RuntimeError("native PPO completed without TensorBoard learning curves")
    write_json(output / "learning-curves.json", curves)
