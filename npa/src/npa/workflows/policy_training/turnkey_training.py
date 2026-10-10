"""Train, evaluate, gate and export real public SmolVLA candidates through durable stages."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import sys

from .contracts import digest
from .public_vla_data import fetch_inputs, local_policy, training_view, write_json
from .public_vla_export import export_policy, file_sha256
from .turnkey_runtime import native_runtime, partial_recovery, run_native
from .turnkey_store import inherit, materialize, read, record, require_identity


def train(args, workspace: Path, output: Path) -> None:
    """Train the selected curated episodes, continuing failed candidates with full recovery.

    Args:
        args: Split, phase, attempt, parent and output locations.
        workspace: Private GPU workspace.
        output: Candidate artifact directory.
    Returns:
        None.
    Raises:
        ValueError: Lineage, recovery or model update evidence is invalid.
        RuntimeError: Native CUDA training fails.
    """
    prepared = materialize(args.input_uri, workspace / "prepared")
    inherit(prepared, output)
    recipe, corpus = read(prepared, "recipe.json"), read(prepared, "corpus.json")
    attempt = _attempt(args)
    native_runtime(workspace)
    inputs = fetch_inputs(workspace)
    parent, recovery, parent_gate = _parent(args, workspace, recipe, corpus)
    if parent is not None:
        inherit(parent, output)
        inherit(parent_gate, output)
    phase, source, selection, increment = _training_selection(
        args, workspace, recipe, corpus, inputs, parent
    )
    _advance(
        args,
        workspace,
        output,
        recipe,
        phase,
        source,
        selection,
        increment * attempt,
        recovery,
    )


def _advance(
    args, workspace, output, recipe, phase, source, selection, steps, recovery
):
    recovery_uri = args.phase_uri.rstrip("/") + f"/recovery/{args.iteration}"
    interrupted = partial_recovery(
        recovery_uri, workspace / "partial", digest(recipe), digest(selection)
    )
    final = _train(
        workspace,
        recipe,
        phase,
        selection,
        source,
        recovery,
        interrupted,
        steps,
        recovery_uri,
    )
    _candidate(
        workspace,
        output,
        final,
        source,
        recipe,
        selection,
        args,
        recovery or interrupted,
    )


def _training_selection(args, workspace, recipe, corpus, inputs, parent):
    phase = "generalist" if args.phase == "pretrain" else "specialist"
    source = inputs["model"] if parent is None else parent / "exported-policy/policy"
    source = local_policy(source, inputs["backbone"], workspace / "base-policy")
    task = None if args.phase == "pretrain" else _task(recipe)
    selection = training_view(
        inputs["dataset"], corpus, workspace / "views" / phase, task
    )
    increment = math.ceil(
        selection["frames"] * recipe[phase + "_epochs"] / recipe["batch_size"]
    )
    return phase, source, selection, increment


def _attempt(args):
    attempt = int(args.iteration)
    if attempt < 1 or args.phase not in {"pretrain", "finetune"}:
        raise ValueError("training phase and positive iteration are required")
    return attempt


def _parent(args, workspace, recipe, corpus):
    attempt = int(args.iteration)
    uri = args.parent_uri
    if attempt > 1:
        uri = args.phase_uri.rstrip("/") + f"/{attempt - 1}/gate/"
    if not uri:
        if args.phase != "pretrain" or attempt != 1:
            raise ValueError("fine-tuning and retries require their preceding gate")
        return None, None, None
    gate = materialize(uri, workspace / "parent-gate")
    require_identity(gate, recipe, corpus)
    decision = read(gate, "decision.json")
    expected = "loop_back" if attempt > 1 else "promote_checkpoint"
    if decision["decision"] != expected:
        raise ValueError("training parent has the wrong gate decision")
    parent = materialize(decision["candidate_uri"], workspace / "parent-candidate")
    require_identity(parent, recipe, corpus)
    _weights(parent, decision["checkpoint_sha256"])
    return parent, parent / "recovery" if attempt > 1 else None, gate


def _task(recipe):
    from libero.libero import benchmark

    return (
        benchmark.get_benchmark_dict()[recipe["suite"]]()
        .get_task(recipe["task_id"])
        .language
    )


def _train(
    root, recipe, phase, selection, source, recovery, interrupted, steps, recovery_uri
):
    from .public_vla import _train_arguments

    checkpoint = interrupted or recovery
    if checkpoint is not None:
        if _adopt(root, checkpoint, recipe, phase, selection, steps):
            return root / phase / "checkpoints/recovered/pretrained_model"
    arguments = _train_arguments(root, recipe, source, selection, phase, steps)
    environment = os.environ | {
        "NPA_VLA_EVIDENCE": str(root / "evidence" / phase),
        "NPA_VLA_RECOVERY_URI": recovery_uri,
        "NPA_VLA_RECIPE_SHA256": digest(recipe),
        "NPA_VLA_SELECTION_SHA256": digest(selection),
    }
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "-m",
        "npa.workflows.policy_training.public_vla_train",
        *arguments,
    ]
    run_native(command, root / "logs/training.log", environment)
    final = root / phase / "checkpoints" / f"{steps:06d}" / "pretrained_model"
    if not (final / "model.safetensors").is_file():
        raise ValueError("native trainer did not save its requested final checkpoint")
    return final


def _adopt(root, checkpoint, recipe, phase, selection, steps):
    _resume_checkpoint(root, checkpoint, recipe, phase, selection, steps)
    if (checkpoint / "npa-evidence").is_dir():
        shutil.copytree(
            checkpoint / "npa-evidence", root / "evidence" / phase, dirs_exist_ok=True
        )
    saved = read(checkpoint, "training_state/training_step.json")["step"]
    if saved > steps:
        raise ValueError("recovery checkpoint exceeds the requested training target")
    return saved == steps


def _resume_checkpoint(root, checkpoint, recipe, phase, selection, steps):
    from .public_vla_data import _localize_tokenizer

    destination = root / phase / "checkpoints/recovered"
    shutil.copytree(checkpoint, destination)
    last = destination.parent / "last"
    last.symlink_to(destination.name, target_is_directory=True)
    model = destination / "pretrained_model"
    config = read(model, "train_config.json")
    config.update(output_dir=str(root / phase), steps=steps, resume=True)
    config["dataset"].update(
        root=str(root / "views" / phase), episodes=selection["episodes"]
    )
    config["policy"].update(
        vlm_model_name=str(root / "inputs/backbone"), pretrained_path=str(model)
    )
    if config.get("scheduler") and "num_decay_steps" in config["scheduler"]:
        config["scheduler"]["num_decay_steps"] = steps
    write_json(model / "train_config.json", config)
    value = read(model, "config.json")
    value.update(vlm_model_name=str(root / "inputs/backbone"))
    write_json(model / "config.json", value)
    for path in model.glob("policy_*processor.json"):
        value = read(model, path.name)
        _localize_tokenizer(value, root / "inputs/backbone")
        write_json(path, value)


def _candidate(root, output, final, source, recipe, selection, args, recovery):
    initial = file_sha256(source / "model.safetensors")
    changed = file_sha256(final / "model.safetensors")
    if initial == changed:
        raise ValueError("candidate training did not change its input weights")
    exported = export_policy(root, final)
    shutil.copytree(root / "exported-policy", output / "exported-policy")
    shutil.copytree(final.parent, output / "recovery")
    phase = "generalist" if args.phase == "pretrain" else "specialist"
    evidence = root / "evidence" / phase
    _training_history(evidence, output, args.phase, args.iteration)
    runtime = read(evidence, "runtime-rank-0.json")
    result = {
        "engine": "lerobot-smolvla-torchrun",
        "phase": args.phase,
        "iteration": int(args.iteration),
        "candidate_uri": args.output_uri,
        "input_checkpoint_sha256": initial,
        "checkpoint_sha256": changed,
        "selection": selection,
        "runtime": runtime,
        "resumed": recovery is not None,
        "training_kind": recipe["training_kind"],
        "export": exported,
    }
    write_json(output / "candidate.json", result)
    record(output, f"{args.phase}-{args.iteration}-train", result)


def _training_history(evidence, output, phase, iteration):
    path = evidence / "metrics-rank-0.jsonl"
    metrics = [json.loads(line) for line in path.read_text().splitlines()]
    if not metrics or not all(math.isfinite(row["loss"]) for row in metrics):
        raise ValueError("native training requires finite measured optimizer history")
    write_json(
        output / "history" / f"{phase}-{iteration}-metrics.json", {"metrics": metrics}
    )
    shutil.copytree(evidence / "checkpoints", output / "checkpoint-events")


def evaluate(args, workspace: Path, output: Path) -> None:
    """Measure reserved-demonstration loss and native LIBERO task completion.

    Args:
        args: Exact candidate and evaluation phase.
        workspace: Private GPU workspace.
        output: Evaluation evidence and native videos.
    Returns:
        None.
    Raises:
        ValueError: Candidate or evaluation evidence is invalid.
        RuntimeError: Native benchmark fails.
    """
    parent = materialize(args.input_uri, workspace / "candidate")
    inherit(parent, output)
    recipe, candidate = read(parent, "recipe.json"), read(parent, "candidate.json")
    if candidate["phase"] != args.phase:
        raise ValueError("evaluation phase does not match the candidate")
    _weights(parent, candidate["checkpoint_sha256"])
    native_runtime(workspace)
    inputs = fetch_inputs(workspace)
    policy = local_policy(
        parent / "exported-policy/policy", inputs["backbone"], workspace / "policy"
    )
    metrics, offline, offset = _evaluate_candidate(
        workspace, parent, policy, recipe, args, output
    )
    evaluation = _evaluation_result(args, candidate, metrics, offline, offset)
    write_json(output / "evaluation.json", evaluation)
    record(output, f"{args.phase}-{candidate['iteration']}-evaluate", evaluation)


def _evaluate_candidate(workspace, parent, policy, recipe, args, output):
    from .public_vla import _evaluation_command, _evaluation_metrics
    from .turnkey_holdout import holdout_loss

    partition = "validation" if args.phase == "pretrain" else "test"
    offline = holdout_loss(
        workspace,
        parent,
        policy,
        partition,
        _task(recipe) if args.phase == "finetune" else None,
    )
    offset = 0 if args.phase == "pretrain" else recipe["evaluation_episodes"]
    run_native(
        _evaluation_command(recipe, policy, output / "benchmark"),
        workspace / "logs/evaluation.log",
        os.environ | {"NPA_VLA_INIT_OFFSET": str(offset)},
    )
    metrics = _evaluation_metrics(output / "benchmark", recipe)
    return metrics, offline, offset


def _evaluation_result(args, candidate, metrics, offline, offset):
    count = metrics["n_episodes"]
    successes = round(metrics["pc_success"] * count / 100)
    if not math.isclose(successes / count * 100, metrics["pc_success"], abs_tol=1e-6):
        raise ValueError(
            "benchmark success percentage does not represent integer counts"
        )
    return {
        "engine": "lerobot-libero-native",
        "phase": args.phase,
        "iteration": candidate["iteration"],
        "candidate_uri": args.input_uri,
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "successes": successes,
        "trials": count,
        "success_rate": successes / count,
        "initial_state_offset": offset,
        "reserved_demonstrations": offline,
    }


def gate(args, workspace: Path, output: Path) -> None:
    """Emit a measured promotion or retry decision bound to the exact checkpoint.

    Args:
        args: Evaluation and gate output URIs.
        workspace: Private worker directory.
        output: Durable decision directory.
    Returns:
        None.
    Raises:
        ValueError: Evaluation counts, lineage or engine are invalid.
    """
    parent = materialize(args.input_uri, workspace / "evaluation")
    inherit(parent, output)
    evaluation = read(parent, "evaluation.json")
    decision = decide(evaluation, read(parent, "recipe.json"))
    write_json(output / "decision.json", decision)
    record(output, f"{evaluation['phase']}-{evaluation['iteration']}-gate", decision)


def decide(evaluation: dict, recipe: dict) -> dict:
    """Validate integer benchmark counts and compare the sealed success threshold.

    Args:
        evaluation: Native evaluation artifact.
        recipe: Immutable run recipe.
    Returns:
        Checkpoint-bound gate decision.
    Raises:
        ValueError: Measurements or checkpoint identity are invalid.
    """
    if evaluation.get("engine") != "lerobot-libero-native":
        raise ValueError("gate requires native LIBERO measurements")
    successes, trials = evaluation["successes"], evaluation["trials"]
    if (
        type(successes) is not int
        or type(trials) is not int
        or trials != recipe["evaluation_episodes"]
    ):
        raise ValueError("gate requires the complete integer trial counts")
    if not 0 <= successes <= trials or not math.isclose(
        evaluation["success_rate"], successes / trials
    ):
        raise ValueError("gate success rate differs from the measured counts")
    checksum = evaluation["checkpoint_sha256"]
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("gate requires a saved checkpoint identity")
    return evaluation | {
        "decision": "promote_checkpoint"
        if successes / trials >= recipe["minimum_success"]
        else "loop_back",
        "minimum_success": recipe["minimum_success"],
    }


def export(args, workspace: Path, output: Path) -> None:
    """Export only the specialist approved by both real evaluation gates.

    Args:
        args: Final gate and promoted export output URIs.
        workspace: Private worker directory.
        output: Portable model and gate lineage.
    Returns:
        None.
    Raises:
        ValueError: Either gate is missing or checkpoint identity changed.
    """
    parent = materialize(args.input_uri, workspace / "gate")
    decision = read(parent, "decision.json")
    _require_both_gates(parent)
    if decision["decision"] != "promote_checkpoint" or decision["phase"] != "finetune":
        raise ValueError("export requires the final promoted specialist")
    candidate = materialize(decision["candidate_uri"], workspace / "candidate")
    require_identity(
        candidate, read(parent, "recipe.json"), read(parent, "corpus.json")
    )
    _weights(candidate, decision["checkpoint_sha256"])
    inherit(parent, output)
    shutil.copytree(candidate / "exported-policy", output / "exported-policy")
    write_json(output / "promotion.json", decision)
    record(
        output,
        "export",
        {
            "engine": "portable-smolvla",
            "checkpoint_sha256": decision["checkpoint_sha256"],
        },
    )


def _weights(root, expected):
    actual = file_sha256(root / "exported-policy/policy/model.safetensors")
    if actual != expected:
        raise ValueError("candidate checkpoint bytes differ from their lineage")


def _require_both_gates(parent):
    history = [
        read(parent / "history", p.name)
        for p in (parent / "history").glob("*-gate.json")
    ]
    for phase in ("pretrain", "finetune"):
        if not any(
            r["phase"] == phase and r["decision"] == "promote_checkpoint"
            for r in history
        ):
            raise ValueError("export requires both measured gates")
