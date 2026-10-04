"""Execute a pinned Isaac-GR00T LIBERO evaluation inside its native runtimes.

This file intentionally imports no NPA modules.  ``groot_libero_x`` starts it
with Isaac-GR00T's Python 3.12 LIBERO environment after materialising the
reviewed upstream source.  Keeping the evaluator on that side of the runtime
boundary prevents the older bootstrap image's GR00T packages from being
silently substituted for the reviewed native entrypoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LIBERO_X_BDDL_LEVELS = frozenset({"LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4"})


def _read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("native evaluator config must be a JSON object")
    return payload


def _native_libero_x_env_name(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode()).hexdigest()[:32]
    return f"libero_sim/npa_groot_libero_x_{digest}"


def _validate_libero_x_task(task: dict[str, Any]) -> str:
    """Validate the native-side copy of the task-to-BDDL contract."""

    task_id = str(task.get("task_id") or "").strip()
    relative = str(task.get("libero_x_bddl_path") or "").strip()
    path = PurePosixPath(relative)
    if (
        not TASK_ID.fullmatch(task_id)
        or path.is_absolute()
        or len(path.parts) != 5
        or path.parts[:3] != ("libero", "libero_x", "bddl")
        or path.parts[3] not in LIBERO_X_BDDL_LEVELS
        or path.suffix != ".bddl"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError("native LIBERO-X task has an unsafe task ID or BDDL path")
    return _native_libero_x_env_name(task_id)


def _configure_libero_x_source(config: dict[str, Any]) -> Path | None:
    """Prioritize the pinned LIBERO-X fork before importing GR00T's wrapper."""

    raw = str(config.get("libero_x_source") or "").strip()
    if not raw:
        return None
    source = Path(raw).resolve()
    package_root = source / "libero"
    if not (source / "libero/libero_x/bddl").is_dir():
        raise RuntimeError("pinned LIBERO-X evaluator source lacks BDDL assets")
    package_text = str(package_root)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    return source


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_server(process: subprocess.Popen[str], host: str, port: int) -> None:
    """Wait for the server to bind, stopping promptly if it exits.

    There is intentionally no invented wall-clock limit here.  The workflow
    runner owns scheduling/cancellation, while a model server may take a
    legitimate amount of time to load a checkpoint.  A failed server process
    is surfaced as soon as it exits.
    """

    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"GR00T model server exited before readiness ({process.returncode})"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _parse_rollout(result: Any) -> dict[str, Any]:
    if (
        not isinstance(result, tuple)
        or len(result) < 3
        or not isinstance(result[0], str)
    ):
        raise RuntimeError(
            "Isaac-GR00T native LIBERO rollout returned an unsupported result"
        )
    env_name, successes, info = result[:3]
    values = [bool(value) for value in list(successes or [])]
    if not values:
        raise RuntimeError(
            "Isaac-GR00T native LIBERO rollout returned no completed episodes"
        )
    info_map = dict(info or {})
    lengths = [int(value) for value in list(info_map.get("episode_lengths") or [])]
    rewards = [float(value) for value in list(info_map.get("episode_rewards") or [])]
    if lengths and len(lengths) != len(values):
        raise RuntimeError(
            "native rollout lengths do not align with completed episodes"
        )
    return {
        "native_env_name": str(env_name),
        "successes": values,
        "episode_lengths": lengths,
        "episode_rewards": rewards,
        "completed_episodes": len(values),
        "success_rate": sum(values) / len(values),
    }


def _run(config: dict[str, Any]) -> dict[str, Any]:
    libero_x_source = _configure_libero_x_source(config)
    # These imports must resolve from the checked-out Isaac-GR00T revision,
    # via the .pth that upstream setup_libero.sh writes into this interpreter.
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.eval.open_loop_eval import evaluate_single_trajectory
    from gr00t.eval.rollout_policy import run_gr00t_sim_policy
    from gr00t.policy.server_client import PolicyClient

    host = "127.0.0.1"
    port = _reserve_loopback_port()
    server_python = str(config["server_python"])
    server_script = str(config["server_script"])
    server_log = Path(str(config["server_log"]))
    server_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        server_python,
        server_script,
        "--model-path",
        str(config["model_path"]),
        "--embodiment-tag",
        "LIBERO_PANDA",
        "--use-sim-policy-wrapper",
        "--host",
        host,
        "--port",
        str(port),
    ]
    with server_log.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        try:
            _wait_for_server(server, host, port)
            policy = PolicyClient(host=host, port=port)
            modality_config = policy.get_modality_config()
            loader = LeRobotEpisodeLoader(str(config["dataset_path"]), modality_config)
            action_rows: list[dict[str, Any]] = []
            action_mse_weighted = 0.0
            action_mae_weighted = 0.0
            action_samples = 0
            action_forward_calls = 0
            task_rows: list[dict[str, Any]] = []
            evaluation_tasks = config.get("evaluation_tasks")
            if not isinstance(evaluation_tasks, list) or not evaluation_tasks:
                raise RuntimeError(
                    "native evaluator requires non-empty evaluation_tasks"
                )
            for index, task in enumerate(evaluation_tasks):
                if not isinstance(task, dict):
                    raise RuntimeError("native evaluator task must be an object")
                task_id = str(task.get("task_id") or "").strip()
                env_name = str(task.get("env_name") or "").strip()
                trajectory_ids = task.get("trajectory_ids")
                if (
                    not task_id
                    or not env_name
                    or not isinstance(trajectory_ids, list)
                    or not trajectory_ids
                ):
                    raise RuntimeError(
                        "native evaluator task lacks task_id, env_name, or trajectory_ids"
                    )
                if env_name.startswith("libero_x/"):
                    if libero_x_source is None:
                        raise RuntimeError(
                            "LIBERO-X task requires the pinned evaluator source"
                        )
                    native_env_name = _validate_libero_x_task(task)
                elif env_name.startswith("libero_sim/"):
                    native_env_name = env_name
                else:
                    raise RuntimeError(
                        "native evaluator task is not a LIBERO or LIBERO-X environment"
                    )
                for trajectory_id in trajectory_ids:
                    if (
                        not isinstance(trajectory_id, int)
                        or trajectory_id < 0
                        or trajectory_id >= len(loader)
                    ):
                        raise RuntimeError(
                            f"trajectory {trajectory_id!r} is unavailable for task {task_id}"
                        )
                    steps = min(
                        int(config["max_episode_steps"]),
                        int(loader.get_episode_length(trajectory_id)),
                    )
                    if steps < 1:
                        raise RuntimeError(
                            f"trajectory {trajectory_id} for task {task_id} has no steps"
                        )
                    mse, mae = evaluate_single_trajectory(
                        policy,
                        loader,
                        trajectory_id,
                        EmbodimentTag.LIBERO_PANDA,
                        steps=steps,
                        execution_horizon=int(config["n_action_steps"]),
                        save_plot_path=str(
                            Path(config["plot_dir"]) / f"{task_id}-{trajectory_id}.jpeg"
                        ),
                    )
                    if not math.isfinite(float(mse)) or not math.isfinite(float(mae)):
                        raise RuntimeError(
                            "native GR00T open-loop evaluator returned a non-finite metric"
                        )
                    action_rows.append(
                        {
                            "task_id": task_id,
                            "trajectory_id": trajectory_id,
                            "mse": float(mse),
                            "mae": float(mae),
                            "steps": steps,
                        }
                    )
                    action_mse_weighted += float(mse) * steps
                    action_mae_weighted += float(mae) * steps
                    action_samples += steps
                    action_forward_calls += math.ceil(
                        steps / int(config["n_action_steps"])
                    )
                video_dir = Path(config["video_dir"]) / f"{index:03d}-{task_id}"
                native = run_gr00t_sim_policy(
                    env_name=native_env_name,
                    n_episodes=int(config["episodes_per_task"]),
                    max_episode_steps=int(config["max_episode_steps"]),
                    model_path="",
                    policy_client_host=host,
                    policy_client_port=port,
                    n_envs=int(config["n_envs"]),
                    n_action_steps=int(config["n_action_steps"]),
                    video_dir=str(video_dir),
                    seed=int(config["seed"]) + index,
                )
                rollout = _parse_rollout(native)
                if rollout["native_env_name"] != native_env_name:
                    raise RuntimeError(
                        "native GR00T rollout did not preserve the registered environment"
                    )
                task_rows.append(
                    {
                        "task_id": task_id,
                        "env_name": env_name,
                        "native_env_name": native_env_name,
                        "trajectory_ids": trajectory_ids,
                        **{
                            key: value
                            for key, value in rollout.items()
                            if key != "native_env_name"
                        },
                        "video_dir": str(video_dir),
                    }
                )
        finally:
            _stop_server(server)
    if action_samples < 1:
        raise RuntimeError("native GR00T evaluator produced no action-error samples")
    return {
        "native_entrypoint": "gr00t.eval.rollout_policy.run_gr00t_sim_policy",
        "open_loop_action_error": {
            "mse": action_mse_weighted / action_samples,
            "mae": action_mae_weighted / action_samples,
            "sample_count": action_samples,
            "forward_calls": action_forward_calls,
            "trajectories": action_rows,
        },
        "tasks": task_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = _run(_read_config(Path(args.config)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
