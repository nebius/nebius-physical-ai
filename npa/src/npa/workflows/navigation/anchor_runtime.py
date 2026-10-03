"""Bind the teacher after exact native initialization and report complete anchor accounting."""

from npa.workflows.navigation.anchor_config import anchor_enabled
from npa.workflows.navigation.artifacts import file_sha256, write_json


def bind_after_snapshot(runner, recipe, reference, source):
    """Authenticate initial state before creating the frozen training teacher.

    Args:
        runner: Initialized native runner with an enabled anchor extension.
        recipe: Sealed executable recipe.
        reference: Already written native pre-update checkpoint.
        source: Sealed input directory containing both checkpoint identities.
    Returns:
        None; disabled recipes execute no native imports or teacher operations.
    Raises:
        ValueError: Source, snapshot or post-binding native state differs.
    """
    if not anchor_enabled(recipe):
        return
    from npa.workflows.navigation.anchor_state import state_digest
    import torch

    initial = _load_source(source, recipe.initial_checkpoint, runner.alg.device)
    saved = torch.load(reference, map_location=runner.alg.device, weights_only=True)
    digest = state_digest(_learning_state(initial))
    if state_digest(_learning_state(saved)) != digest:
        raise ValueError(
            "anchor pre-update snapshot differs from native initialization"
        )
    baseline = _load_source(
        source, recipe.baseline_anchor.checkpoint, runner.alg.device
    )
    random = state_digest(_binding_rng(runner.alg.device))
    runner.alg.bind_baseline(baseline, recipe.initial_checkpoint.sha256)
    current = {**runner.alg.save(), "iter": runner.current_learning_iteration}
    if state_digest(_learning_state(current)) != digest:
        raise ValueError("teacher binding changed native learning state")
    _record_binding(runner, reference, digest, random)


def anchor_evidence(runner, recipe, output):
    """Verify frozen teacher and full minibatch accounting after native training.

    Args:
        runner: Completed native runner.
        recipe: Sealed full-budget recipe.
        output: Result directory for measured anchor evidence.
    Returns:
        Optional source-bound anchor evidence without checkpoint tensors.
    Raises:
        ValueError: Teacher state changed or update/sample accounting is incomplete.
    """
    if not anchor_enabled(recipe):
        return None
    from npa.workflows.navigation.anchor_state import ANCHOR_KEY

    algorithm = runner.alg
    record = algorithm.save()[ANCHOR_KEY]
    steps = algorithm.anchor_steps - algorithm.initial_anchor_steps
    samples = algorithm.anchor_samples - algorithm.initial_anchor_samples
    epochs = algorithm.num_learning_epochs
    expected_steps = recipe.iterations * epochs * algorithm.num_mini_batches
    expected_samples = (
        recipe.iterations * epochs * recipe.num_envs * runner.cfg["num_steps_per_env"]
    )
    if steps != expected_steps or samples != expected_samples:
        raise ValueError("baseline anchor did not cover every native PPO minibatch")
    evidence = {
        key: value for key, value in record.items() if key != "teacher_state_dict"
    }
    evidence.update(
        optimizer_steps_this_run=steps, samples_this_run=samples, teacher_unchanged=True
    )
    evidence["binding"] = algorithm.anchor_binding
    write_json(output / "baseline-anchor.json", evidence)
    return evidence


def _load_source(source, identity, device):
    import torch

    path = source / identity.file
    if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
        raise ValueError("anchor checkpoint escapes input bundle")
    if file_sha256(path) != identity.sha256:
        raise ValueError("anchor checkpoint changed before native binding")
    return torch.load(path, map_location=device, weights_only=True)


def _binding_rng(device):
    import torch

    states = {"cpu": torch.get_rng_state()}
    if str(device).startswith("cuda"):
        states["cuda"] = torch.cuda.get_rng_state(device)
    return states


def _record_binding(runner, reference, native_digest, random_digest):
    from npa.workflows.navigation.anchor_state import state_digest

    if state_digest(_binding_rng(runner.alg.device)) != random_digest:
        raise ValueError("teacher binding changed native random generator state")
    runner.alg.anchor_binding = {
        "reference_checkpoint_sha256": file_sha256(reference),
        "native_initial_state_sha256": native_digest,
        "random_state_sha256": random_digest,
        "native_state_unchanged": True,
        "random_state_unchanged": True,
    }


def _learning_state(payload):
    return {
        key: payload[key]
        for key in (
            "actor_state_dict",
            "critic_state_dict",
            "optimizer_state_dict",
            "iter",
        )
    }
