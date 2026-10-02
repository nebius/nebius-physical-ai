"""Evaluate a verified native Cosmos checkpoint in the matching LIBERO simulator."""

from __future__ import annotations

import json
import http.client
import socket
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from npa.clients.storage import safe_s3_download_target
from npa.workbench.cosmos.policy_artifacts import (
    file_digest,
    materialize_bundle,
    policy_workspace,
    publish_bundle,
    write_local_json,
)
from npa.workbench.cosmos.policy_contract import (
    ACTION_CONTRACT,
    EXPERIMENT,
    FRAMEWORK_REVISION,
    LIBERO_REVISION,
    EvalSettings,
    validate_summary,
)
from npa.workbench.cosmos.policy_runtime import (
    prepare_simulation_runtime,
    prepare_training_runtime,
    run_native,
)
from npa.workbench.cosmos.policy_train import TRAIN_SCHEMA

EVAL_SCHEMA = "npa.cosmos3.policy-eval.v1"


def _inference_overrides(bundle: Path) -> list[str]:
    # The server introspects the dataloader's action/prompt settings but never
    # instantiates it. Clear only its unused training-data root so native config
    # resolution does not depend on LIBERO_ROOT from a previous worker.
    return [
        f"model.config.tokenizer.vae_path={bundle / 'Wan2.2_VAE.pth'}",
        "dataloader_train.dataloader.datasets.libero.dataset.root=null",
    ]


def _preflight_configuration(
    repo: Path, bundle: Path, env: dict[str, str], artifacts: Path
) -> None:
    code = (
        "import sys; from pathlib import Path; "
        "from cosmos_framework.inference.common.config import load_config, save_config; "
        "config = load_config('cosmos_framework/configs/base/config.py', sys.argv[1], "
        "overrides=sys.argv[3:]); save_config(config, Path(sys.argv[2])); "
        "print('Native inference configuration resolved')"
    )
    run_native(
        [
            str(repo / ".venv/bin/python"),
            "-c",
            code,
            EXPERIMENT,
            str(artifacts / "inference-config"),
            *_inference_overrides(bundle),
        ],
        cwd=repo,
        env=env,
        log=artifacts / "inference-configuration.log",
    )


def server_argv(
    repo: Path, bundle: Path, checkpoint: Path, port: int, seed: int
) -> list[str]:
    """Bind the native policy server to loopback with matching training semantics.

    Args:
        repo: Pinned framework checkout.
        bundle: Verified training artifact directory.
        checkpoint: Verified trained DCP directory.
        port: Private local port.
        seed: Model sampling seed.
    Returns:
        Native server argument vector.
    Raises:
        None.
    """
    return [
        str(repo / ".venv/bin/python"),
        "-m",
        "cosmos_framework.scripts.action_policy_server_libero",
        "--experiment",
        EXPERIMENT,
        "--checkpoint-path",
        str(checkpoint / "model"),
        "--experiment-overrides",
        *_inference_overrides(bundle),
        "--action-normalization",
        "quantile_rot",
        "--action-stats-path",
        str(bundle / "action_stats.json"),
        "--raw-action-dim",
        "10",
        "--fps",
        "20",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--seed",
        str(seed),
    ]


def evaluation_argv(
    python: Path, repo: Path, output: Path, port: int, settings: EvalSettings
) -> list[str]:
    """Build the matching native closed-loop simulator invocation.

    Args:
        python: Separate simulator interpreter.
        repo: Pinned framework checkout.
        output: Native results directory.
        port: Loopback model port.
        settings: Task selection, trials, and seed.
    Returns:
        Native simulator argument vector retaining complete rollout GIFs.
    Raises:
        None.
    """
    return [
        str(python),
        str(repo / "cosmos_framework/simulation/libero/closed_loop_eval.py"),
        "--server_url",
        f"http://127.0.0.1:{port}",
        "--task_suite",
        "libero_10",
        "--num_trials_per_task",
        str(settings.trials_per_task),
        "--num_envs",
        "1",
        "--camera",
        "agentview,wrist",
        "--image_size",
        "256",
        "--action_dim",
        "10",
        "--action_space",
        "frame_wise_relative",
        "--rotation_space",
        "6d",
        "--gripper_mode",
        "zero_one",
        "--mujoco_gl",
        "osmesa",
        "--task_ids",
        ",".join(map(str, settings.task_ids)),
        "--seed",
        str(settings.seed),
        "--save_gifs",
        "--output_dir",
        str(output),
    ]


def _wait_ready(process: subprocess.Popen, port: int) -> dict[str, Any]:
    while process.poll() is None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request("GET", "/info")
            response = connection.getresponse()
            if response.status == 200:
                return json.load(response)
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(1)
    raise RuntimeError("native policy server exited before model readiness")


def _evaluate_native(
    repo: Path,
    bundle: Path,
    checkpoint: Path,
    python: Path,
    env: dict[str, str],
    artifacts: Path,
    settings: EvalSettings,
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    argv = server_argv(repo, bundle, checkpoint, port, settings.seed)
    with (artifacts / "server.log").open("wb") as log:
        process = subprocess.Popen(argv, cwd=repo, env=env, stdout=log, stderr=log)
        try:
            info = _wait_ready(process, port)
            if (
                Path(info.get("checkpoint", "")).resolve()
                != (checkpoint / "model").resolve()
            ):
                raise ValueError("policy server loaded a different checkpoint")
            write_local_json(artifacts / "server-info.json", info)
            run_native(
                evaluation_argv(python, repo, artifacts / "rollouts", port, settings),
                cwd=repo,
                env=env,
                log=artifacts / "evaluation.log",
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def evaluate_policy(
    *,
    input_path: str,
    output_path: str,
    trials_per_task: int = 50,
    task_ids: str = "0,1,2,3,4,5,6,7,8,9",
    seed: int = 0,
) -> dict[str, Any]:
    """Run real robot rollouts against the exact trained checkpoint.

    Args:
        input_path: Completed training.json artifact URI.
        output_path: Private local/S3 result prefix.
        trials_per_task: Trials for every selected task, using native initial states.
        task_ids: Comma-separated LIBERO-10 task IDs.
        seed: Simulator and model seed.
    Returns:
        Validated summary, checkpoint identity, protocol, and rollout artifacts.
    Raises:
        ValueError: Invalid settings, checkpoint mismatch, or incomplete evaluation.
        subprocess.CalledProcessError: Native setup or evaluation failed.
    """
    settings = EvalSettings(
        trials_per_task=trials_per_task,
        task_ids=[int(v) for v in task_ids.split(",")],
        seed=seed,
    )
    with policy_workspace(output_path, "eval") as root:
        artifacts = root / "artifacts"
        artifacts.mkdir()
        # Complete dependency checks before transferring a full distributed
        # checkpoint. Failed setup must not waste a large artifact readback.
        repo, env = prepare_training_runtime(root, guardrails=True)
        _preflight_configuration(repo, root / "bundle", env, artifacts)
        python = prepare_simulation_runtime(root, repo, env)
        for log in root.glob("*.log"):
            shutil.copyfile(log, artifacts / log.name)
        report = materialize_bundle(input_path, root / "bundle", TRAIN_SCHEMA)
        _verify_contract(report)
        if file_digest(root / "bundle/action_stats.json") != report["stats_sha256"]:
            raise ValueError("checkpoint normalization statistics mismatch")
        checkpoint = safe_s3_download_target(root / "bundle", report["checkpoint"], "")
        _evaluate_native(
            repo, root / "bundle", checkpoint, python, env, artifacts, settings
        )
        summary = json.loads((artifacts / "rollouts/summary.json").read_text())
        validate_summary(summary, settings)
        result = {
            "schema": EVAL_SCHEMA,
            "status": "succeeded",
            "summary": summary,
            "settings": settings.model_dump(),
            "training_manifest": input_path,
            "training": report,
            "framework_revision": FRAMEWORK_REVISION,
            "libero_revision": LIBERO_REVISION,
            "action_contract": ACTION_CONTRACT,
            "stats_sha256": file_digest(root / "bundle/action_stats.json"),
        }
        return publish_bundle(artifacts, output_path, result, "evaluation.json")


def _verify_contract(report: dict[str, Any]) -> None:
    if report.get("framework_revision") != FRAMEWORK_REVISION:
        raise ValueError("checkpoint requires a different framework revision")
    if (
        report.get("action_contract") != ACTION_CONTRACT
        or report.get("experiment") != EXPERIMENT
    ):
        raise ValueError("checkpoint action contract does not match LIBERO evaluation")
