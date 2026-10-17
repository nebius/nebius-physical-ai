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
from typing import Any, Callable, Mapping


TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LIBERO_X_BDDL_LEVELS = frozenset({"LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4"})
NATIVE_WORKER_PROVENANCE_SCHEMA = "npa.groot_libero_x.native_worker_provenance.v1"
POLICY_SERVER_EVIDENCE_SCHEMA = "npa.groot_libero_x.native_policy_server_evidence.v1"


def _read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("native evaluator config must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular native source file."""

    if not path.is_file():
        raise RuntimeError("native provenance source is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_record(module: Any) -> dict[str, str]:
    """Return an actual imported module path and content digest."""

    path = Path(str(getattr(module, "__file__", ""))).resolve()
    return {
        "module": str(module.__name__),
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _read_server_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require device and checkpoint evidence written after policy construction."""

    path = Path(str(config.get("server_evidence_path") or ""))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "native policy server did not publish readable evidence"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != POLICY_SERVER_EVIDENCE_SCHEMA
    ):
        raise RuntimeError("native policy server evidence has an unknown schema")
    model = payload.get("model")
    device = payload.get("device")
    imports = payload.get("imported_modules")
    if (
        payload.get("model_load_completed") is not True
        or not isinstance(model, dict)
        or not isinstance(model.get("checkpoint_files"), list)
        or not model["checkpoint_files"]
        or not isinstance(device, dict)
        or device.get("cuda_available") is not True
        or not str(device.get("selected_model_tensor_device") or "").startswith("cuda")
        or int(device.get("device_count") or 0) < 1
        or not isinstance(imports, list)
        or not imports
    ):
        raise RuntimeError("native policy server evidence lacks model or CUDA proof")
    return payload


def _source_episode_indices(
    task: Mapping[str, Any], input_provenance: Mapping[str, Any]
) -> list[int]:
    """Resolve logical task trajectories to the source episodes the loader consumes."""

    trajectories = task.get("trajectory_ids")
    if (
        not isinstance(trajectories, list)
        or not trajectories
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in trajectories
        )
    ):
        raise RuntimeError("native evaluator task has invalid trajectory identities")
    mapping = input_provenance.get("episode_index_mapping")
    if mapping is None:
        return list(trajectories)
    if not isinstance(mapping, Mapping):
        raise RuntimeError("native evaluator has invalid episode identity mapping")
    logical = mapping.get("workflow_trajectory_id")
    source = mapping.get("source_episode_index")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (logical, source)
    ):
        raise RuntimeError("native evaluator has invalid episode identity mapping")
    reference = task.get("reference_observation")
    if logical in trajectories and (
        not isinstance(reference, Mapping)
        or reference.get("source_episode_index") != source
    ):
        raise RuntimeError(
            "native evaluator episode identity mapping conflicts with task provenance"
        )
    return [
        source if trajectory_id == logical else trajectory_id
        for trajectory_id in trajectories
    ]


def _native_libero_x_env_name(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode()).hexdigest()[:32]
    return f"libero_sim/npa_groot_libero_x_{digest}"


def _validate_libero_x_task(task: dict[str, Any], source: Path | None = None) -> str:
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
    expected_hash = str(task.get("libero_x_bddl_sha256") or "").lower()
    expected_language = str(task.get("language") or "").strip()
    if bool(expected_hash) != bool(expected_language):
        raise RuntimeError("native LIBERO-X task has incomplete BDDL provenance")
    if source is not None:
        bddl_root = (source / "libero/libero_x/bddl").resolve()
        bddl_path = (source / path).resolve()
        if bddl_root not in bddl_path.parents or not bddl_path.is_file():
            raise RuntimeError("native LIBERO-X BDDL path escapes evaluator source")
        if expected_hash:
            actual_hash = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError("native LIBERO-X BDDL hash differs from preparation")
            from libero.libero.utils.parse_bddl import parse_bddl_file

            parsed = parse_bddl_file(str(bddl_path))
            if str(parsed.get("language") or "").strip() != expected_language:
                raise RuntimeError(
                    "native LIBERO-X BDDL language differs from preparation"
                )
    return _native_libero_x_env_name(task_id)


def _configure_libero_x_source(config: dict[str, Any]) -> Path | None:
    """Prioritize the pinned LIBERO-X fork before importing GR00T's wrapper."""

    raw = str(config.get("libero_x_source") or "").strip()
    if not raw:
        return None
    source = Path(raw).resolve()
    if not (source / "libero/libero_x/bddl").is_dir():
        raise RuntimeError("pinned LIBERO-X evaluator source lacks BDDL assets")
    # LIBERO-X keeps the importable ``libero`` package one level below its
    # checkout root.  Adding that inner directory makes ``libero`` resolve to
    # ``libero/libero`` and breaks the upstream ``libero.libero`` imports used
    # by GR00T's registration bridge.  Keep the evaluator root first instead,
    # so its namespace package retains the expected two-level layout.
    package_text = str(source)
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
    if len(lengths) != len(values) or len(rewards) != len(values):
        raise RuntimeError(
            "native rollout episode metrics do not align with completed episodes"
        )
    completion_identities = _completion_identities(info_map, len(values))
    outcomes = [
        {
            **identity,
            "success": values[index],
            "episode_length": lengths[index],
            "episode_reward": rewards[index],
        }
        for index, identity in enumerate(completion_identities)
    ]
    return {
        "native_env_name": str(env_name),
        "successes": values,
        "episode_lengths": lengths,
        "episode_rewards": rewards,
        "completed_episode_outcomes": outcomes,
        "completed_episodes": len(values),
        "success_rate": sum(values) / len(values),
    }


def _completion_identities(
    info: Mapping[str, Any], completed_episodes: int
) -> list[dict[str, int]]:
    """Require upstream completion records for each emitted episode outcome."""

    environment_indices = info.get("npa_environment_indices")
    completion_ordinals = info.get("npa_completion_ordinals")
    if (
        not isinstance(environment_indices, list)
        or not isinstance(completion_ordinals, list)
        or len(environment_indices) != completed_episodes
        or len(completion_ordinals) != completed_episodes
    ):
        raise RuntimeError(
            "native rollout lacks completion identities for its episode outcomes"
        )
    identities: list[dict[str, int]] = []
    for environment_index, completion_ordinal in zip(
        environment_indices, completion_ordinals, strict=True
    ):
        if (
            isinstance(environment_index, bool)
            or not isinstance(environment_index, int)
            or environment_index < 0
            or isinstance(completion_ordinal, bool)
            or not isinstance(completion_ordinal, int)
            or completion_ordinal < 0
        ):
            raise RuntimeError("native rollout completion identity is invalid")
        identities.append(
            {
                "environment_index": environment_index,
                "completion_ordinal": completion_ordinal,
            }
        )
    return identities


def _episode_video_records(
    *,
    task_video_dir: Path,
    wave_dir: Path,
    task_id: str,
    trajectory_ids: list[int],
    initial_state_protocol: str,
    reset_seeds: list[int],
    outcomes: list[dict[str, Any]],
    wave_index: int,
) -> list[dict[str, Any]]:
    """Join upstream completion identities to one video and requested seed each."""

    outcomes_by_environment = _seeded_outcomes_by_environment(outcomes, reset_seeds)
    records: list[dict[str, Any]] = []
    for environment_index, reset_seed in enumerate(reset_seeds):
        outcome = outcomes_by_environment.get(environment_index)
        if outcome is None:
            raise RuntimeError("native rollout is missing a seeded environment outcome")
        videos = sorted((wave_dir / f"env-{environment_index:03d}").rglob("*.mp4"))
        if len(videos) != 1:
            raise RuntimeError(
                "native rollout cannot bind one final MP4 to each seeded environment"
            )
        video = videos[0]
        try:
            relative = video.relative_to(task_video_dir).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                "native rollout video escapes its task workspace"
            ) from exc
        records.append(
            {
                "task_id": task_id,
                "wave_index": wave_index,
                "environment_index": environment_index,
                "reset_seed": reset_seed,
                "completion_ordinal": outcome["completion_ordinal"],
                "initial_state": {"protocol": initial_state_protocol},
                "trajectory_ids": trajectory_ids,
                "success": outcome["success"],
                "episode_length": outcome["episode_length"],
                "episode_reward": outcome["episode_reward"],
                "video_relative_path": relative,
            }
        )
    return records


def _seeded_outcomes_by_environment(
    outcomes: list[dict[str, Any]], reset_seeds: list[int]
) -> dict[int, dict[str, Any]]:
    """Require one first-completion outcome for every requested environment."""

    if len(outcomes) != len(reset_seeds):
        raise RuntimeError(
            "native rollout does not have one outcome per seeded environment"
        )
    outcomes_by_environment: dict[int, dict[str, Any]] = {}
    for outcome in outcomes:
        environment_index = outcome.get("environment_index")
        completion_ordinal = outcome.get("completion_ordinal")
        if (
            isinstance(environment_index, bool)
            or not isinstance(environment_index, int)
            or environment_index not in range(len(reset_seeds))
            or isinstance(completion_ordinal, bool)
            or not isinstance(completion_ordinal, int)
            or completion_ordinal != 0
        ):
            raise RuntimeError(
                "native rollout completion does not describe an initial seeded episode"
            )
        if environment_index in outcomes_by_environment:
            raise RuntimeError(
                "native rollout has duplicate seeded environment outcomes"
            )
        outcomes_by_environment[environment_index] = outcome
    return outcomes_by_environment


def _run_seeded_episode_waves(
    *,
    native_env_name: str,
    rollout_runner: Callable[..., Any],
    policy_host: str,
    policy_port: int,
    video_dir: Path,
    task_id: str,
    trajectory_ids: list[int],
    initial_state_protocol: str,
    task_index: int,
    episodes_per_task: int,
    n_envs: int,
    max_episode_steps: int,
    n_action_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run fixed reset-seed waves so policy arms see identical initial states."""

    if episodes_per_task % n_envs:
        raise RuntimeError("episodes_per_task must divide evenly into n_envs waves")
    successes: list[bool] = []
    lengths: list[int] = []
    rewards: list[float] = []
    reset_seed_batches: list[list[int]] = []
    episodes: list[dict[str, Any]] = []
    for offset in range(0, episodes_per_task, n_envs):
        wave_seed = seed + task_index * episodes_per_task + offset
        wave_index = offset // n_envs
        wave_dir = video_dir / f"wave-{wave_index:03d}"
        parsed = _parse_rollout(
            rollout_runner(
                env_name=native_env_name,
                n_episodes=n_envs,
                max_episode_steps=max_episode_steps,
                model_path="",
                policy_client_host=policy_host,
                policy_client_port=policy_port,
                n_envs=n_envs,
                n_action_steps=n_action_steps,
                video_dir=str(wave_dir),
                seed=wave_seed,
            )
        )
        if parsed["native_env_name"] != native_env_name:
            raise RuntimeError(
                "native rollout did not preserve the registered environment"
            )
        if parsed["completed_episodes"] != n_envs:
            raise RuntimeError(
                "native rollout did not complete its seeded episode wave"
            )
        successes.extend(parsed["successes"])
        lengths.extend(parsed["episode_lengths"])
        rewards.extend(parsed["episode_rewards"])
        reset_seeds = list(range(wave_seed, wave_seed + n_envs))
        reset_seed_batches.append(reset_seeds)
        episodes.extend(
            _episode_video_records(
                task_video_dir=video_dir,
                wave_dir=wave_dir,
                task_id=task_id,
                trajectory_ids=trajectory_ids,
                initial_state_protocol=initial_state_protocol,
                reset_seeds=reset_seeds,
                outcomes=parsed["completed_episode_outcomes"],
                wave_index=wave_index,
            )
        )
    return {
        "successes": successes,
        "episode_lengths": lengths,
        "episode_rewards": rewards,
        "completed_episodes": len(successes),
        "success_rate": sum(successes) / len(successes),
        "initial_reset_seed_batches": reset_seed_batches,
        "episodes": episodes,
    }


def _flatten_open_loop_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Translate GR00T's nested open-loop observation to the sim-server protocol.

    Args:
        observation: Nested GR00T observation emitted by ``open_loop_eval``.

    Returns:
        Flat simulator observation accepted by ``Gr00tSimPolicyWrapper``.

    Raises:
        RuntimeError: If a required modality has an unsupported shape.
    """
    flattened: dict[str, Any] = {}
    for modality in ("video", "state"):
        values = observation.get(modality)
        if not isinstance(values, dict) or not values:
            raise RuntimeError(f"open-loop observation lacks {modality} values")
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise RuntimeError(
                    f"open-loop observation has an invalid {modality} key"
                )
            flattened[f"{modality}.{key}"] = value
    language = observation.get("language")
    if not isinstance(language, dict) or not language:
        raise RuntimeError("open-loop observation lacks language values")
    for key, values in language.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(values, list)
            or any(
                not isinstance(item, list)
                or len(item) != 1
                or not isinstance(item[0], str)
                for item in values
            )
        ):
            raise RuntimeError(
                "open-loop language observation has an unsupported shape"
            )
        flattened[key] = tuple(item[0] for item in values)
    return flattened


def _unprefix_sim_actions(actions: Any) -> dict[str, Any]:
    """Translate simulator-wrapper action keys back to open-loop policy keys.

    Args:
        actions: Flat action mapping returned by the simulator-wrapper server.

    Returns:
        Action mapping expected by Isaac-GR00T's open-loop evaluator.

    Raises:
        RuntimeError: If the server action response is not a flat simulator action map.
    """
    if not isinstance(actions, dict) or not actions:
        raise RuntimeError("simulator policy server returned no actions")
    prefix = "action."
    normalized: dict[str, Any] = {}
    for key, value in actions.items():
        if not isinstance(key, str) or not key.startswith(prefix) or key == prefix:
            raise RuntimeError("simulator policy server returned an invalid action key")
        normalized[key.removeprefix(prefix)] = value
    return normalized


class _OpenLoopSimServerPolicy:
    """Adapt nested open-loop requests to the retained flat simulator server contract."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], Any]:
        """Request one action chunk through the simulator-wrapper policy server.

        Args:
            observation: Nested observation produced by the open-loop evaluator.
            options: Optional inference options forwarded to the policy client.

        Returns:
            Open-loop action keys and upstream response metadata.

        Raises:
            RuntimeError: If the server response cannot satisfy the open-loop contract.
        """
        result = self._client.get_action(
            _flatten_open_loop_observation(observation), options
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "simulator policy server returned an invalid action response"
            )
        actions, info = result
        return _unprefix_sim_actions(actions), info


def _worker_provenance(
    *,
    config: Mapping[str, Any],
    server_evidence: Mapping[str, Any],
    rollout_module: Any,
    open_loop_module: Any,
    server_client_module: Any,
) -> dict[str, Any]:
    """Bind the imported simulator consumer to its model and input evidence."""

    input_provenance = config.get("input_provenance")
    if not isinstance(input_provenance, Mapping):
        raise RuntimeError("native evaluator lacks digest-bound input provenance")
    evaluator_path = Path(__file__).resolve()
    return {
        "schema": NATIVE_WORKER_PROVENANCE_SCHEMA,
        "worker_execution": {
            "executable": sys.executable,
            "python_version": sys.version,
            "sanitized_server_argv": list(config.get("sanitized_server_argv") or []),
        },
        "executed_native_entrypoint": {
            "module": "gr00t.eval.rollout_policy.run_gr00t_sim_policy",
            "path": str(Path(rollout_module.__file__).resolve()),
            "sha256": _sha256_file(Path(rollout_module.__file__).resolve()),
        },
        "native_evaluator": {
            "path": str(evaluator_path),
            "sha256": _sha256_file(evaluator_path),
        },
        "imported_modules": [
            _module_record(rollout_module),
            _module_record(open_loop_module),
            _module_record(server_client_module),
        ],
        "input_provenance": dict(input_provenance),
        "policy_server": dict(server_evidence),
    }


def _run(config: dict[str, Any]) -> dict[str, Any]:
    libero_x_source = _configure_libero_x_source(config)
    input_provenance = config.get("input_provenance")
    rollout_protocol = (
        input_provenance.get("rollout_protocol")
        if isinstance(input_provenance, Mapping)
        else None
    )
    initial_state_protocol = (
        str(rollout_protocol.get("initial_state_protocol") or "")
        if isinstance(rollout_protocol, Mapping)
        else ""
    )
    if not initial_state_protocol:
        raise RuntimeError("native evaluator lacks the prepared initial-state protocol")
    # These imports must resolve from the checked-out Isaac-GR00T revision,
    # via the .pth that upstream setup_libero.sh writes into this interpreter.
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.eval import open_loop_eval, rollout_policy
    from gr00t.eval.open_loop_eval import evaluate_single_trajectory
    from gr00t.eval.rollout_policy import run_gr00t_sim_policy
    from gr00t.policy import server_client
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
        "--npa-evidence-input",
        str(config["server_evidence_input"]),
        "--npa-evidence-output",
        str(config["server_evidence_path"]),
        "--npa-upstream-server",
        str(config["upstream_server_script"]),
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
            server_evidence = _read_server_evidence(config)
            policy = PolicyClient(host=host, port=port)
            modality_config = policy.get_modality_config()
            loader = LeRobotEpisodeLoader(str(config["dataset_path"]), modality_config)
            open_loop_policy = _OpenLoopSimServerPolicy(policy)
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
                    native_env_name = _validate_libero_x_task(task, libero_x_source)
                elif env_name.startswith("libero_sim/"):
                    native_env_name = env_name
                else:
                    raise RuntimeError(
                        "native evaluator task is not a LIBERO or LIBERO-X environment"
                    )
                source_episode_indices = _source_episode_indices(
                    task, config["input_provenance"]
                )
                for trajectory_id, source_episode_index in zip(
                    trajectory_ids, source_episode_indices, strict=True
                ):
                    if source_episode_index >= len(loader):
                        raise RuntimeError(
                            "mapped source episode "
                            f"{source_episode_index!r} is unavailable for task {task_id}"
                        )
                    steps = min(
                        int(config["max_episode_steps"]),
                        int(loader.get_episode_length(source_episode_index)),
                    )
                    if steps < 1:
                        raise RuntimeError(
                            "mapped source episode "
                            f"{source_episode_index} for task {task_id} has no steps"
                        )
                    mse, mae = evaluate_single_trajectory(
                        open_loop_policy,
                        loader,
                        source_episode_index,
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
                            "source_episode_index": source_episode_index,
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
                rollout = _run_seeded_episode_waves(
                    native_env_name=native_env_name,
                    rollout_runner=run_gr00t_sim_policy,
                    policy_host=host,
                    policy_port=port,
                    video_dir=video_dir,
                    task_id=task_id,
                    trajectory_ids=trajectory_ids,
                    initial_state_protocol=initial_state_protocol,
                    task_index=index,
                    episodes_per_task=int(config["episodes_per_task"]),
                    n_envs=int(config["n_envs"]),
                    max_episode_steps=int(config["max_episode_steps"]),
                    n_action_steps=int(config["n_action_steps"]),
                    seed=int(config["seed"]),
                )
                task_rows.append(
                    {
                        "task_id": task_id,
                        "env_name": env_name,
                        "native_env_name": native_env_name,
                        "trajectory_ids": trajectory_ids,
                        "task_input": {
                            "task_id": task_id,
                            "env_name": env_name,
                            "trajectory_ids": trajectory_ids,
                            "libero_x_bddl_path": str(
                                task.get("libero_x_bddl_path") or ""
                            ),
                            "libero_x_bddl_sha256": str(
                                task.get("libero_x_bddl_sha256") or ""
                            ),
                            "language": str(task.get("language") or ""),
                        },
                        **rollout,
                        "video_dir": str(video_dir),
                    }
                )
        finally:
            _stop_server(server)
    if action_samples < 1:
        raise RuntimeError("native GR00T evaluator produced no action-error samples")
    return {
        "native_entrypoint": "gr00t.eval.rollout_policy.run_gr00t_sim_policy",
        "worker_provenance": _worker_provenance(
            config=config,
            server_evidence=server_evidence,
            rollout_module=rollout_policy,
            open_loop_module=open_loop_eval,
            server_client_module=server_client,
        ),
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
