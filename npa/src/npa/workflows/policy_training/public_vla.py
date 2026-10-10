"""Execute public SmolVLA continued training, task adaptation and measured LIBERO evaluation."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .public_vla_data import (
    fetch_inputs,
    local_policy,
    prepare_corpus,
    training_view,
    write_json,
)


def main() -> None:
    """Run the complete public reference pipeline in a GPU container or Slurm allocation.

    Args:
        None; the process accepts --output-path and optional --input-path recipe.
    Returns:
        None.
    Raises:
        ValueError: A recipe, checkpoint or evaluation contract is invalid.
        subprocess.CalledProcessError: Native training or evaluation fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--input-path", type=Path)
    args = parser.parse_args()
    recipe = _recipe(args.input_path)
    root = args.output_path.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _environment(root)
    _bind_recipe(root, recipe)
    _run_pipeline(root, recipe)


def _recipe(path):
    recipe = {
        "seed": 42,
        "batch_size": 64,
        "generalist_epochs": 1,
        "specialist_epochs": 10,
        "suite": "libero_spatial",
        "task_id": 0,
        "evaluation_episodes": 10,
        "minimum_success": 0.7,
        "workers": 8,
    }
    if path is not None:
        incoming = json.loads(path.read_text())
        if not isinstance(incoming, dict):
            raise ValueError("public VLA recipe must be a JSON object")
        if set(incoming) - set(recipe):
            raise ValueError("unknown public VLA recipe keys")
        recipe.update(incoming)
    _validate_recipe(recipe)
    return recipe


def _validate_recipe(recipe):
    for key in (
        "batch_size",
        "generalist_epochs",
        "specialist_epochs",
        "evaluation_episodes",
        "workers",
    ):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    threshold = recipe["minimum_success"]
    if (
        recipe["evaluation_episodes"] > 25
        or type(threshold) not in (int, float)
        or not 0 < threshold <= 1
    ):
        raise ValueError(
            "evaluation needs disjoint states and a success threshold in (0, 1]"
        )
    if type(recipe["seed"]) is not int or recipe["seed"] < 0:
        raise ValueError("seed must be nonnegative")
    if recipe["suite"] not in {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    }:
        raise ValueError("unknown LIBERO suite")
    if type(recipe["task_id"]) is not int or not 0 <= recipe["task_id"] < 10:
        raise ValueError("task_id must select one task in the suite")


def _environment(root):
    settings = {
        "HF_HOME": str(root / "hf-cache"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "WANDB_MODE": "disabled",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "TOKENIZERS_PARALLELISM": "false",
        "LIBERO_CONFIG_PATH": str(root / "libero-config"),
        "OMP_NUM_THREADS": "4",
        "ACCELERATE_MIXED_PRECISION": "bf16",
        "NPA_VLA_ASSETS": str(root / "inputs" / "assets"),
    }
    os.environ.update(settings)
    _record_execution(root)
    _rendering_backend(root)
    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise RuntimeError("the pinned LeRobot image with LIBERO extras is required")
    package = Path(next(iter(spec.submodule_search_locations))) / "libero"
    config = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "assets": str(root / "inputs" / "assets"),
        "datasets": str(root / "inputs" / "dataset"),
    }
    write_json(root / "libero-config" / "config.yaml", config)


def _record_execution(root):
    scheduler = (
        "slurm"
        if os.environ.get("SLURM_JOB_ID")
        else "kubernetes"
        if os.environ.get("KUBERNETES_SERVICE_HOST")
        else "direct"
    )
    write_json(
        root / "execution.json",
        {
            "scheduler": scheduler,
            "training_launcher": "torchrun",
            "training_nodes": 1,
        },
    )


def _rendering_backend(root):
    try:
        ctypes.CDLL("libEGL_nvidia.so.0")
        backend = "nvidia-egl"
    except OSError:
        os.environ.update(
            LIBGL_ALWAYS_SOFTWARE="1",
            MESA_LOADER_DRIVER_OVERRIDE="llvmpipe",
            EGL_PLATFORM="surfaceless",
        )
        backend = "mesa-software-egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(_egl_device())
    write_json(root / "rendering.json", {"backend": backend, "policy_device": "cuda"})


def _egl_device():
    from mujoco.egl import egl_ext as egl
    from OpenGL.error import GLError

    for index, device in enumerate(egl.eglQueryDevicesEXT()):
        display = egl.eglGetPlatformDisplayEXT(
            egl.EGL_PLATFORM_DEVICE_EXT, device, None
        )
        try:
            initialized = egl.eglInitialize(display, None, None)
        except GLError:
            continue
        if initialized:
            egl.eglTerminate(display)
            return index
    raise RuntimeError(
        "no usable headless EGL device; install a graphics-capable runtime"
    )


def _bind_recipe(root, recipe):
    path = root / "recipe.json"
    if path.exists() and json.loads(path.read_text()) != recipe:
        raise ValueError(
            "run directory belongs to a different recipe; use a new output path"
        )
    write_json(path, recipe)


def _run_pipeline(root, recipe):
    from libero.libero import benchmark

    inputs = fetch_inputs(root)
    corpus = prepare_corpus(inputs["dataset"], root / "corpus.json", recipe["seed"])
    suite = benchmark.get_benchmark_dict()[recipe["suite"]]()
    task = suite.get_task(recipe["task_id"]).language
    if not any(
        row["partition"] == "train" and task in row["tasks"]
        for row in corpus["episodes"]
    ):
        raise ValueError(
            "the public corpus has no training episodes for the selected simulator task"
        )
    write_json(
        root / "task.json",
        {"suite": recipe["suite"], "id": recipe["task_id"], "instruction": task},
    )
    base = local_policy(inputs["model"], inputs["backbone"], root / "base-policy")
    _evaluate(root, recipe, base, "baseline", 0)
    generalist = _train_phase(root, recipe, inputs, corpus, base, "generalist", None)
    _evaluate(root, recipe, generalist, "generalist", 0)
    specialist = _train_phase(
        root, recipe, inputs, corpus, generalist, "specialist", task
    )
    _evaluate(root, recipe, specialist, "specialist", 0)
    chosen = _select_candidate(root, recipe, generalist, specialist)
    _evaluate(root, recipe, chosen, "test", recipe["evaluation_episodes"])
    _finish(root, chosen)


def _finish(root, chosen):
    from .public_vla_export import export_policy
    from .public_vla_report import build_report

    selection = _require_quality_gate(root, chosen)
    if not (root / "execution.json").exists():
        _record_execution(root)
    exported = export_policy(root, chosen)
    _execute(
        [
            sys.executable,
            "-m",
            "npa.workflows.policy_training.public_vla_verify",
            "--input-path",
            str(root),
        ],
        root / "logs/export-verification.log",
        os.environ.copy(),
    )
    selection["checkpoint_sha256"] = exported["trained_weights_sha256"]
    write_json(root / "selection.json", selection)
    write_json(
        root / "completed.json",
        {
            "training_executed": True,
            "evaluation_executed": True,
            "selected_phase": selection["selected"],
        },
    )
    build_report(root, root / "report")


def _require_quality_gate(root, chosen):
    recipe = _recipe(root / "recipe.json")
    selection = json.loads((root / "selection.json").read_text())
    phase = selection["selected"]
    if phase not in {"generalist", "specialist"}:
        raise ValueError("unknown selected training phase")
    _require_evaluation_identity(root / "evaluation" / phase, recipe, chosen, 0)
    _require_evaluation_identity(
        root / "evaluation/test", recipe, chosen, recipe["evaluation_episodes"]
    )
    threshold = recipe["minimum_success"]
    validation = _evaluation_metrics(root / "evaluation" / phase, recipe)
    test = _evaluation_metrics(root / "evaluation/test", recipe)
    selection.update(
        validation_pass=validation["pc_success"] / 100 >= threshold,
        minimum_success=threshold,
        test_success=test["pc_success"] / 100,
    )
    selection["quality_gate_passed"] = bool(
        selection["validation_pass"] and selection["test_success"] >= threshold
    )
    write_json(root / "selection.json", selection)
    if not selection["quality_gate_passed"]:
        raise ValueError("quality gate failed; checkpoint export is blocked")
    return selection


def _train_phase(root, recipe, inputs, corpus, source, phase, task):
    selection = training_view(inputs["dataset"], corpus, root / "views" / phase, task)
    steps = math.ceil(
        selection["frames"] * recipe[phase + "_epochs"] / recipe["batch_size"]
    )
    plan = selection | {"steps": steps, "phase": phase, "task": task}
    write_json(root / "plans" / f"{phase}.json", plan)
    output = root / phase
    final = output / "checkpoints" / f"{steps:06d}" / "pretrained_model"
    if final.is_dir() and (root / "evidence" / phase / "complete.json").exists():
        return final
    args = _train_arguments(root, recipe, source, selection, phase, steps)
    env = os.environ | {"NPA_VLA_EVIDENCE": str(root / "evidence" / phase)}
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "-m",
        "npa.workflows.policy_training.public_vla_train",
        *args,
    ]
    _execute(command, root / "logs" / f"{phase}.log", env)
    if not (final / "model.safetensors").is_file():
        raise ValueError("trainer completed without the expected checkpoint")
    write_json(root / "evidence" / phase / "complete.json", plan)
    return final


def _train_arguments(root, recipe, source, selection, phase, steps):
    checkpoint = root / phase / "checkpoints" / "last" / "pretrained_model"
    if checkpoint.is_dir():
        return [f"--config_path={checkpoint / 'train_config.json'}", "--resume=true"]
    if (root / phase).exists():
        raise ValueError(
            "incomplete training has no resumable checkpoint; retain it and use a new run directory"
        )
    return [
        f"--policy.path={source}",
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        "--policy.train_expert_only=true",
        "--policy.freeze_vision_encoder=true",
        "--policy.optimizer_lr=0.00001",
        f"--policy.scheduler_warmup_steps={min(100, steps // 10)}",
        f"--policy.scheduler_decay_steps={steps}",
        "--dataset.repo_id=lerobot/libero",
        f"--dataset.root={root / 'views' / phase}",
        "--dataset.use_imagenet_stats=false",
        f"--dataset.episodes={json.dumps(selection['episodes'])}",
        "--dataset.video_backend=torchcodec",
        f"--output_dir={root / phase}",
        f"--steps={steps}",
        f"--batch_size={recipe['batch_size']}",
        f"--seed={recipe['seed']}",
        f"--num_workers={recipe['workers']}",
        "--env_eval_freq=0",
        "--log_freq=25",
        f"--save_freq={max(1, steps // 4)}",
        "--wandb.enable=false",
    ]


def _evaluate(root, recipe, policy, phase, offset):
    output = root / "evaluation" / phase
    if (output / "eval_info.json").exists():
        _require_evaluation_identity(output, recipe, policy, offset)
        _evaluation_metrics(output, recipe)
        return
    write_json(output / "request.json", _evaluation_request(recipe, policy, offset))
    env = os.environ | {"NPA_VLA_INIT_OFFSET": str(offset)}
    _execute(
        _evaluation_command(recipe, policy, output),
        root / "logs" / f"evaluate-{phase}.log",
        env,
    )
    _require_evaluation_identity(output, recipe, policy, offset)
    _evaluation_metrics(output, recipe)


def _evaluation_command(recipe, policy, output):
    return [
        sys.executable,
        "-m",
        "npa.workflows.policy_training.public_vla_eval",
        f"--policy.path={policy}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--env.type=libero",
        f"--env.task={recipe['suite']}",
        f"--env.task_ids=[{recipe['task_id']}]",
        "--env.init_states=true",
        "--env.max_parallel_tasks=1",
        f"--eval.n_episodes={recipe['evaluation_episodes']}",
        f"--eval.batch_size={recipe['evaluation_episodes']}",
        "--eval.use_async_envs=false",
        f"--seed={recipe['seed']}",
        f"--output_dir={output}",
    ]


def _evaluation_request(recipe, policy, offset):
    from .public_vla_export import file_sha256

    if not (policy / "model.safetensors").is_file():
        raise ValueError("evaluation requires a saved policy checkpoint")
    return {
        "schema": "npa.public-vla.evaluation-request.v1",
        "policy_files": {
            path.name: file_sha256(path)
            for path in sorted(policy.iterdir())
            if path.is_file()
            and path.suffix in {".json", ".safetensors"}
            and path.name != "train_config.json"
        },
        "suite": recipe["suite"],
        "task_id": recipe["task_id"],
        "seed": recipe["seed"],
        "episodes": recipe["evaluation_episodes"],
        "initial_state_offset": offset,
    }


def _require_evaluation_identity(output, recipe, policy, offset):
    path = output / "request.json"
    if not path.is_file():
        raise ValueError(
            "evaluation has no checkpoint identity; rerun in a new directory"
        )
    if json.loads(path.read_text()) != _evaluation_request(recipe, policy, offset):
        raise ValueError("evaluation checkpoint, task or holdout identity changed")


def _evaluation_metrics(output, recipe):
    info = json.loads((output / "eval_info.json").read_text())["overall"]
    count, success = info["n_episodes"], info["pc_success"]
    if (
        type(count) is not int
        or count != recipe["evaluation_episodes"]
        or type(success) not in (int, float)
        or not math.isfinite(success)
        or not 0 <= success <= 100
    ):
        raise ValueError("incomplete or nonfinite simulator evaluation")
    videos = list((output / "videos").rglob("*.mp4"))
    if len(videos) != min(10, recipe["evaluation_episodes"]):
        raise ValueError(
            "native simulator evaluation has an incomplete rollout video set"
        )
    return info


def _select_candidate(root, recipe, generalist, specialist):
    scores = {
        phase: _evaluation_metrics(root / "evaluation" / phase, recipe)["pc_success"]
        / 100
        for phase in ("generalist", "specialist")
    }
    chosen = max(scores, key=lambda phase: (scores[phase], phase == "specialist"))
    write_json(
        root / "selection.json",
        {
            "scores": scores,
            "selected": chosen,
            "validation_pass": scores[chosen] >= recipe["minimum_success"],
            "minimum_success": recipe["minimum_success"],
        },
    )
    return {"generalist": generalist, "specialist": specialist}[chosen]


def _execute(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": log.stem, "status": "running"}), flush=True)
    with log.open("a") as stream:
        subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
        )
    print(json.dumps({"stage": log.stem, "status": "completed"}), flush=True)


if __name__ == "__main__":
    main()
