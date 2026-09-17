"""Shared execution path for genuine Isaac Lab-Arena policy evaluation.

The CLI, SDK, and ``npa.workflow`` toolRef all call this module.  The worker
image contains the immutable Apache-2.0 Arena source, but Isaac Sim and Isaac
Lab remain operator-authorized runtime downloads in the inherited NPA cache.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from importlib import import_module
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable
from urllib.parse import urlparse

from npa.clients.storage import StorageClient

from .capabilities import capabilities as capabilities
from .identity import CAPABILITIES_SCHEMA as CAPABILITIES_SCHEMA
from .identity import REGISTERED_ENVIRONMENTS as REGISTERED_ENVIRONMENTS
from .identity import (
    ISAAC_ARENA_VERSION,
    ISAAC_ARENA_REVISION,
    LIGHTWHEEL_SDK_VERSION,
    ISAAC_ARENA_ROOT,
    ARTIFACT_SCHEMA,
    SUPPORTED_POLICIES,
    UNSUPPORTED_ENVIRONMENTS,
    SUPPORTED_ENVIRONMENTS,
)
from .viewport_graphics import _prepare_viewport_graphics
from .replay_input import (
    _prepare_replay_execution_input,
    _input_evidence,
    _checkpoint_input,
)
from .errors import IsaacArenaError
from .hashing import file_sha256 as _sha256
from .video_evidence import probe_mp4 as _probe_mp4, denoise_mp4 as _denoise_mp4
from .video_evidence import verify_capture_evidence as _verify_capture_evidence
from .ground_truth import simulator_ground_truth as _simulator_ground_truth
from .acceptance import qualify_visual_acceptance as _qualify_visual_acceptance
from .action_evidence import read_action_evidence as _read_action_evidence
from .simulator_video import legacy_rtx_kit_args
from .runtime_identity import assert_runtime_identity

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class IsaacArenaRequest:
    """One reproducible policy evaluation request.

    Args:
        output_path: Local directory or operator-owned S3 artifact prefix.
        environment: Registered scored upstream environment.
        policy_type: Upstream zero_action, replay, or rsl_rl adapter.
        input_path: Operator replay HDF5 or checkpoint with its agent configuration.
        execution_device: Simulator CPU or first CUDA device; image needs CUDA.
        num_episodes: Requested completed scored episodes.
        num_envs: Concurrent simulator environments.
        seed: Upstream random seed.
        embodiment: Optional compatible upstream robot selector.
        object_name: Optional compatible upstream object selector.
        record_video: Capture and verify one environment's real viewport.
        run_id: Searchable operator run identifier.
        runtime_image: Exact candidate image coordinate for provenance.
        dry_run: Render the invocation without downloading or executing inputs.
    """

    output_path: str
    environment: str = "cube_goal_pose"
    policy_type: str = "zero_action"
    input_path: str = ""
    execution_device: str = "cuda:0"
    num_episodes: int = 1
    num_envs: int = 1
    seed: int = 42
    embodiment: str = ""
    object_name: str = ""
    record_video: bool = False
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


def _validate_output(request: IsaacArenaRequest) -> None:
    parsed = urlparse(request.output_path)
    if not request.output_path.strip():
        raise IsaacArenaError("output_path is required")
    if parsed.scheme and parsed.scheme != "s3":
        raise IsaacArenaError("output_path must be a local path or s3:// prefix")
    if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
        raise IsaacArenaError("output_path must include an S3 bucket and prefix")


def _validate(request: IsaacArenaRequest) -> None:
    _validate_output(request)
    if request.policy_type not in SUPPORTED_POLICIES:
        raise IsaacArenaError(
            "policy_type must be one of: " + ", ".join(sorted(SUPPORTED_POLICIES))
        )
    if request.environment not in SUPPORTED_ENVIRONMENTS:
        if request.environment in UNSUPPORTED_ENVIRONMENTS:
            raise IsaacArenaError(
                "environment has no upstream scored task and is unsupported by the NPA evaluation contract"
            )
        raise IsaacArenaError(
            "environment must be a registered NPA Arena environment; inspect the capabilities command"
        )
    for field, value in (
        ("environment", request.environment),
        ("embodiment", request.embodiment),
        ("object_name", request.object_name),
    ):
        if value and _NAME.fullmatch(value) is None:
            raise IsaacArenaError(f"{field} contains unsupported characters")
    if request.num_episodes < 1 or request.num_envs < 1:
        raise IsaacArenaError("num_episodes and num_envs must be positive")
    if request.record_video and (request.num_envs != 1 or request.num_episodes != 1):
        raise IsaacArenaError(
            "video qualification requires exactly one environment and one episode"
        )
    if request.num_envs > request.num_episodes:
        raise IsaacArenaError("num_envs cannot exceed num_episodes")
    if request.policy_type == "replay" and (
        request.num_envs != 1 or request.num_episodes != 1
    ):
        raise IsaacArenaError("replay requires exactly one environment and one episode")
    if request.policy_type == "zero_action" and request.input_path:
        raise IsaacArenaError("zero_action does not accept input_path")
    if request.policy_type != "zero_action" and not request.input_path:
        raise IsaacArenaError(f"{request.policy_type} requires input_path")
    if request.execution_device not in {"cpu", "cuda:0"}:
        raise IsaacArenaError("execution_device must be cpu or cuda:0")


def _local_input(request: IsaacArenaRequest, root: Path) -> Path | None:
    if not request.input_path:
        return None
    if request.input_path.startswith("s3://"):
        target = root / "input"
        downloaded = StorageClient.from_environment().download_path(
            request.input_path, str(target)
        )
        return Path(downloaded).resolve()
    parsed = urlparse(request.input_path)
    if parsed.scheme:
        raise IsaacArenaError("input_path must be a local path or s3:// URI")
    path = Path(request.input_path).expanduser().resolve()
    if not path.exists():
        raise IsaacArenaError(f"input_path does not exist: {path}")
    return path


def build_evaluation_argv(
    request: IsaacArenaRequest, *, output_dir: Path, local_input: Path | None = None
) -> list[str]:
    """Build argv for upstream's genuine ``policy_runner.py``.

    Args:
        request: Validated evaluation options.
        output_dir: Fresh upstream artifact directory.
        local_input: Privately materialized replay or checkpoint input.
    Returns:
        Ordered process arguments for the pinned upstream runner.
    Raises:
        IsaacArenaError: The policy input is missing or incomplete.
    """

    argv = _runner_options(request, output_dir)
    if request.record_video:
        # Keep the launcher preset explicit. Capture overrides it with the
        # required real-time reconstruction settings and checks their readback.
        argv.extend(
            [
                "--rendering_mode",
                "balanced",
                "--kit_args",
                legacy_rtx_kit_args(),
                "--record_viewport_video",
            ]
        )
    argv.extend(_policy_input_argv(request, local_input))
    # Upstream's subparsers require global and policy flags before the environment.
    argv.append(request.environment)
    if request.embodiment:
        argv.extend(["--embodiment", request.embodiment])
    if request.object_name:
        argv.extend(["--object", request.object_name])
    return argv


def _runner_options(request: IsaacArenaRequest, output_dir: Path) -> list[str]:
    return [
        os.environ.get("ISAAC_ARENA_PYTHON", "/isaac-sim/python.sh"),
        f"{ISAAC_ARENA_ROOT}/isaaclab_arena/evaluation/policy_runner.py",
        "--headless",
        "--device",
        request.execution_device,
        "--policy_type",
        request.policy_type,
        "--num_episodes",
        str(request.num_episodes),
        "--num_envs",
        str(request.num_envs),
        "--seed",
        str(request.seed),
        "--output_base_dir",
        str(output_dir),
    ]


def _policy_input_argv(
    request: IsaacArenaRequest, local_input: Path | None
) -> list[str]:
    if request.dry_run and request.policy_type in {"replay", "rsl_rl"}:
        flag = (
            "--replay_file_path"
            if request.policy_type == "replay"
            else "--checkpoint_path"
        )
        return [flag, "<operator-input>"]
    if request.policy_type == "replay":
        if local_input is None or not local_input.is_file():
            raise IsaacArenaError("replay input_path must resolve to one HDF5 file")
        return ["--replay_file_path", str(local_input)]
    if request.policy_type == "rsl_rl":
        if local_input is None:
            raise IsaacArenaError("rsl_rl requires a checkpoint input")
        checkpoint, _agent_config = _checkpoint_input(local_input)
        return ["--checkpoint_path", str(checkpoint)]
    return []


def _subprocess_env(*, viewport_only: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    # Inputs are materialized and outputs are published by NPA, so the simulator
    # gets no cloud credentials or HTTP admission secrets.  This also keeps its
    # captured log safe to retain as an evaluation artifact.
    for key in tuple(env):
        upper = key.upper()
        if upper in {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "HF_TOKEN",
            "NGC_API_KEY",
            "NEBIUS_IAM_TOKEN",
            "NEBIUS_TOKEN_FACTORY_KEY",
        } or upper.endswith(
            (
                "_API_KEY",
                "_SECRET",
                "_TOKEN",
                "_PASSWORD",
                "_ACCESS_KEY_ID",
                "_SECRET_ACCESS_KEY",
                "_SECRET_KEY",
            )
        ):
            env.pop(key, None)
    env.setdefault("ACCEPT_EULA", "Y")
    env.pop("NPA_ISAAC_ARENA_VIEWPORT_ONLY", None)
    if viewport_only:
        env["NPA_ISAAC_ARENA_VIEWPORT_ONLY"] = "1"
    return env


def _numeric_metrics(log_text: str) -> dict[str, float]:
    """Extract upstream's plain-Python metric mapping from its retained log."""

    for line in reversed(log_text.splitlines()):
        marker = "Metrics:"
        if marker not in line:
            continue
        candidate = line.split(marker, 1)[1].strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        result: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                result[str(key)] = number
        if result:
            return result
    return {}


def _episode_records(run_dir: Path) -> tuple[list[dict], list[Path]]:
    result_files = sorted(run_dir.glob("episode_results_rank*.jsonl"))
    if not result_files:
        raise IsaacArenaError("upstream evaluation wrote no episode-results JSONL")
    records: list[dict[str, Any]] = []
    for path in result_files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IsaacArenaError(
                    f"invalid episode JSONL at {path.name}:{number}"
                ) from exc
            if not isinstance(record, dict) or not isinstance(
                record.get("success"), bool
            ):
                raise IsaacArenaError(
                    f"episode record at {path.name}:{number} has no boolean success"
                )
            records.append(record)
    if not records:
        raise IsaacArenaError("upstream evaluation completed no scored episodes")
    return records, result_files


def _summarize(run_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    records, result_files = _episode_records(run_dir)
    lengths = [record.get("episode_length") for record in records]
    if any(type(length) is not int or length <= 0 for length in lengths):
        raise IsaacArenaError("upstream evaluation recorded an invalid episode length")
    successes = sum(bool(record["success"]) for record in records)
    progress = [record.get("progress") for record in records]
    progress = [item for item in progress if isinstance(item, dict)]
    scores = [
        float(item.get("overall_score", 0.0))
        for item in progress
        if isinstance(item.get("overall_score", 0.0), (int, float))
    ]
    events = sum(
        len(item.get("events", []))
        for item in progress
        if isinstance(item.get("events", []), list)
    )
    return (
        {
            "episodes": len(records),
            "successes": successes,
            "success_rate": successes / len(records),
            "mean_episode_length": sum(lengths) / len(lengths),
            "max_progress_score": max(scores, default=0.0),
            "progress_event_count": events,
        },
        result_files,
    )


def _gpu_info() -> dict[str, Any]:
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise IsaacArenaError("PyTorch import failed because a dependency is missing") from exc
        return {"available": False, "device_name": "", "compute_capability": []}
    except ImportError as exc:
        raise IsaacArenaError("PyTorch could not be imported for CUDA inspection") from exc
    try:
        available = bool(torch.cuda.is_available())
        if not available:
            return {"available": False, "device_name": "", "compute_capability": []}
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
    except (AssertionError, OSError, RuntimeError) as exc:
        raise IsaacArenaError("CUDA driver/device query failed during Arena evidence capture") from exc


def _publish(local_dir: Path, output_path: str) -> str:
    if output_path.startswith("s3://"):
        return StorageClient.from_environment().upload_directory(
            str(local_dir), output_path
        )
    target = Path(output_path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for source in sorted(local_dir.iterdir()):
        destination = target / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
    return str(target)


def _prepare_inputs(
    request: IsaacArenaRequest, private_dir: Path
) -> tuple[Path | None, dict | None]:
    if request.dry_run:
        return None, None
    local_input = _local_input(request, private_dir)
    evidence = _input_evidence(request, local_input)
    if request.policy_type != "replay":
        return local_input, evidence
    assert local_input is not None and evidence is not None
    environment = next(
        item
        for item in capabilities()["environments"]
        if item["name"] == request.environment
    )
    execution, execution_evidence = _prepare_replay_execution_input(
        local_input,
        private_dir,
        source_evidence=evidence,
        embodiment=request.embodiment or environment["default_embodiment"],
    )
    evidence["execution"] = execution_evidence
    return execution, evidence


def _runtime_metadata(request: IsaacArenaRequest) -> dict[str, Any]:
    return {
        "image": request.runtime_image or os.environ.get("NPA_TASK_IMAGE", ""),
        "execution_device": request.execution_device,
        "viewport_renderer_gpu_required": request.record_video,
        "viewport_graphics": {
            "mode": "deferred" if request.record_video else "not_requested",
            "validated": False,
        },
        "isaac_runtime_fetch": True,
        "runtime_identity": assert_runtime_identity(),
        "lightwheel_sdk": {
            "version": LIGHTWHEEL_SDK_VERSION,
            "baked": True,
            "license": "Apache-2.0",
        },
        "lightwheel_registry_assets": {
            "baked": False,
            "runtime_fetch": request.environment
            in {
                "franka_put_and_close_door",
                "gr1_open_microwave",
                "press_button",
                "put_item_in_fridge_and_close_door",
            },
            "license": "upstream-provider-controlled",
            "redistribution": False,
        },
        "model_baked": False,
        "dataset_baked": False,
        "input_payload_scope": "Arena policy weights and evaluation datasets",
        "inherited_dependency_fixtures": "Public dependency test fixtures are described separately in the image's third-party notices.",
    }


def _public_argv(argv: list[str]) -> list[str]:
    public = list(argv)
    for flag, replacement in (
        ("--replay_file_path", "<operator-input>"),
        ("--checkpoint_path", "<operator-input>"),
        ("--output_base_dir", "<run-output>"),
    ):
        if flag in public:
            public[public.index(flag) + 1] = replacement
    return public


def _private_input_log(log_text: str, argv: list[str]) -> str:
    for flag in ("--replay_file_path", "--checkpoint_path"):
        if flag not in argv:
            continue
        path = argv[argv.index(flag) + 1]
        log_text = log_text.replace(path, "<operator-input>")
        if flag == "--checkpoint_path" and Path(path).parent != Path("/"):
            log_text = log_text.replace(
                str(Path(path).parent), "<operator-input-directory>"
            )
    return log_text


def _prepare_evaluation(
    request: IsaacArenaRequest, root: Path
) -> tuple[Path, Path, dict, list[str]]:
    artifact_root, private_dir = root / "artifacts", root / "private"
    output_root = artifact_root / "upstream"
    output_root.mkdir(parents=True)
    execution_input, evidence = _prepare_inputs(request, private_dir)
    argv = build_evaluation_argv(
        request, output_dir=output_root, local_input=execution_input
    )
    public_request = asdict(request)
    public_request["output_path"] = "<operator-output>"
    if public_request["input_path"]:
        public_request["input_path"] = "<operator-input>"
    base = {
        "schema": ARTIFACT_SCHEMA,
        "upstream": {
            "repository": "https://github.com/isaac-sim/IsaacLab-Arena",
            "version": ISAAC_ARENA_VERSION,
            "revision": ISAAC_ARENA_REVISION,
        },
        "request": public_request,
        "runtime": _runtime_metadata(request),
        "input": evidence,
        "argv": _public_argv(argv),
    }
    return artifact_root, private_dir, base, argv


def _require_optix_runtime(log_text: str) -> None:
    for line in log_text.lower().splitlines():
        if (
            "unable to load denoiser weights" in line
            or "optix error: optix_error_" in line
            or ("[error]" in line and "[rtx.optixdenoising.plugin]" in line)
            or ("optixdenoisercreate(" in line and "failed" in line)
        ):
            raise IsaacArenaError(
                "Arena viewport reported an OptiX runtime failure; "
                "renderer settings alone do not prove denoising"
            )


def _execute_upstream(
    request, artifact_root, private_dir, base, runner, graphics_preparer, argv
):
    sim_env = _subprocess_env(viewport_only=request.record_video)
    if request.record_video:
        base["runtime"]["viewport_graphics"] = graphics_preparer(
            private_dir / "viewport-graphics",
            sim_env,
        )
    if runner is subprocess.run:
        from functools import partial
        from .phase_liveness import run_supervised

        runner = partial(run_supervised, artifact_root=artifact_root, private_dir=private_dir)
    completed = runner(
        argv,
        cwd=ISAAC_ARENA_ROOT,
        env=sim_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_text = _private_input_log(completed.stdout or "", argv)
    (artifact_root / "evaluation.log").write_text(log_text, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(log_text.splitlines()[-40:])
        raise IsaacArenaError(
            f"upstream policy_runner failed ({completed.returncode}):\n{tail}"
        )
    if request.record_video:
        _require_optix_runtime(log_text)
    run_dirs = sorted(
        path for path in (artifact_root / "upstream").iterdir() if path.is_dir()
    )
    if len(run_dirs) != 1:
        raise IsaacArenaError(
            "upstream evaluation did not create exactly one run directory"
        )
    return run_dirs[0], log_text


def _validated_summary(
    request: IsaacArenaRequest, run_dir: Path, log_text: str, evidence: dict | None
) -> dict:
    summary, _ = _summarize(run_dir)
    if summary["episodes"] < request.num_episodes:
        raise IsaacArenaError(
            "upstream evaluation completed fewer episodes than requested"
        )
    if (request.record_video or request.policy_type == "replay") and summary[
        "episodes"
    ] != 1:
        raise IsaacArenaError(
            "replay and video qualification require exactly one completed episode"
        )
    if request.record_video and request.policy_type == "replay":
        if ((evidence or {}).get("trajectory") or {}).get(
            "nonzero_actions"
        ) is not True:
            raise IsaacArenaError(
                "replay visual qualification requires measured nonzero source actions"
            )
    summary["metrics"] = _numeric_metrics(log_text)
    success_rate = summary["metrics"].get("success_rate")
    if success_rate is None or not math.isclose(
        success_rate, summary["success_rate"], abs_tol=1e-9, rel_tol=0.0
    ):
        raise IsaacArenaError(
            "upstream numeric success metric disagrees with episode JSONL"
        )
    steps = ((evidence or {}).get("execution") or {}).get("prepared_steps")
    require_task_success = request.record_video and request.policy_type != "zero_action"
    ground_truth = _simulator_ground_truth(
        run_dir,
        environment=request.environment,
        policy_type=request.policy_type,
        expected_episodes=summary["episodes"],
        expected_successes=summary["successes"],
        expected_action_steps=steps,
        require_task_success=require_task_success,
    )
    ground_truth["action_evidence"] = (
        _read_action_evidence(
            run_dir,
            policy_type=request.policy_type,
            expected_steps=ground_truth["episodes"][0]["episode_length"],
        )
        if require_task_success
        else None
    )
    summary["simulator_ground_truth"] = ground_truth
    return summary


def _behavior_evidence(request: IsaacArenaRequest, summary: dict) -> dict:
    ground_truth = summary["simulator_ground_truth"]
    task_success = summary["successes"] > 0 and summary["metrics"]["success_rate"] > 0
    progress = bool(
        summary["max_progress_score"] > 0
        or summary["progress_event_count"]
        or (ground_truth.get("task_motion") or {}).get("task_success")
    )
    observed = task_success or progress
    if request.record_video and request.policy_type != "zero_action" and not observed:
        raise IsaacArenaError(
            "nonzero policy produced no task success or upstream progress evidence"
        )
    return {
        "policy_is_nonzero_adapter": request.policy_type != "zero_action",
        "output_behavior_observed": observed,
        "task_success": task_success,
        "task_progress": progress,
        "positive_metrics": {
            key: value for key, value in summary["metrics"].items() if value > 0
        },
        "movement_metrics_are_outcome_claims": False,
        "meaningful": request.policy_type != "zero_action" and observed,
    }


def _video_binding(
    request: IsaacArenaRequest, run_dir: Path, evidence: dict | None, ground_truth: dict
) -> dict:
    source_hash = str((evidence or {}).get("sha256") or "")
    motion = ground_truth.get("task_motion") or {}
    actions = ground_truth.get("action_evidence") or {}
    return {
        "run_id": request.run_id or run_dir.name,
        "upstream_run_directory": run_dir.name,
        "environment": request.environment,
        "policy_type": request.policy_type,
        "episode": motion.get("episode"),
        "action_steps": motion.get("episode_length"),
        "input_sha256": source_hash,
        "execution_input_sha256": str(
            ((evidence or {}).get("execution") or {}).get("prepared_sha256")
            or source_hash
        ),
        "executed_action_evidence_sha256": ((actions.get("file") or {}).get("sha256")),
        "executed_action_sequence_sha256": actions.get("sequence_sha256"),
        "simulator_ground_truth_sha256": [
            item["sha256"] for item in ground_truth["files"]
        ],
    }


def _capture_context(request: IsaacArenaRequest, ground_truth: dict) -> dict:
    motion = ground_truth.get("task_motion")
    if motion and motion.get("visual_progress_qualified"):
        return motion
    if request.policy_type != "zero_action":
        raise IsaacArenaError(
            "visual task qualification requires a supported simulator task-progress binding"
        )
    capture = ground_truth["episodes"][0].get("video_capture")
    if not capture:
        raise IsaacArenaError("viewport capture has no simulator action-step binding")
    terminal = capture["terminal_action_step"]
    return {
        "video_capture": capture,
        "progress_interval": {
            "start_action_step": 0,
            "end_action_step": terminal,
            "total_action_steps": terminal,
        },
    }


def _video_artifacts(
    request, run_dir, evidence, summary, video_preparer
) -> dict[Path, dict]:
    ground_truth = summary["simulator_ground_truth"]
    videos = sorted(run_dir.rglob("*.mp4"))
    if request.record_video and (
        not videos or any(path.stat().st_size == 0 for path in videos)
    ):
        raise IsaacArenaError(
            "video recording was requested but no non-empty MP4 was written"
        )
    result = {}
    task_motion = ground_truth.get("task_motion") or {}
    interval = task_motion.get("visual_interval")
    progress_interval = task_motion.get("progress_interval")
    progress_signal = task_motion.get("visual_progress_signal")
    progress_region = task_motion.get("visual_progress_region")
    progress_radius = task_motion.get("visual_association_radius_fraction")
    for source in videos:
        capture = _verify_capture_evidence(
            run_dir,
            source,
            task_motion=_capture_context(request, ground_truth),
            expected_steps=(ground_truth.get("task_motion") or {}).get(
                "episode_length"
            ),
        )
        video, derivation = video_preparer(source)
        metadata = _probe_mp4(
            video,
            evidence_interval=interval,
            progress_interval=progress_interval if interval is not None else None,
            progress_signal=progress_signal if interval is not None else None,
            progress_region=progress_region if interval is not None else None,
            progress_association_radius_fraction=(
                progress_radius if interval is not None else None
            ),
        )
        metadata["sha256"] = _sha256(video)
        metadata["derivation"] = derivation
        metadata["binding"] = _video_binding(request, run_dir, evidence, ground_truth)
        metadata["simulator_capture"] = capture
        acceptance = None
        if request.policy_type != "zero_action":
            acceptance = _qualify_visual_acceptance(
                environment=request.environment,
                policy_type=request.policy_type,
                evidence=evidence,
                summary=summary,
                ground_truth=ground_truth,
                capture=capture,
                video=metadata,
            )
        metadata["acceptance"] = acceptance
        metadata["task_qualified"] = bool(acceptance and acceptance["qualified"])
        result[video] = {"video": metadata}
        result[source] = {
            "visual_source": {
                "role": "raw_upstream_source",
                "evidence_derivative": video.name,
                "sha256": derivation["source_sha256"],
            }
        }
    return result


def _artifact_entries(artifact_root: Path, metadata: dict[Path, dict]) -> list[dict]:
    return [
        {
            "path": str(path.relative_to(artifact_root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            **metadata.get(path, {}),
        }
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file()
    ]


def _evaluation_result(
    request, artifact_root, run_dir, log_text, base, video_preparer
) -> dict:
    summary = _validated_summary(request, run_dir, log_text, base["input"])
    behavior = _behavior_evidence(request, summary)
    report = run_dir / "index.html"
    if not report.is_file() or report.stat().st_size == 0:
        raise IsaacArenaError("upstream evaluation report is missing")
    videos = _video_artifacts(request, run_dir, base["input"], summary, video_preparer)
    gpu = _gpu_info()
    if not gpu["available"]:
        raise IsaacArenaError("Arena evaluation returned without a CUDA device")
    return {
        **base,
        "status": "ok",
        "run_id": request.run_id or run_dir.name,
        "upstream_run_directory": run_dir.name,
        "summary": summary,
        "behavior": behavior,
        "gpu": gpu,
        "artifacts": _artifact_entries(artifact_root, videos),
    }


def _save_result(
    request: IsaacArenaRequest, artifact_root: Path, manifest: dict
) -> dict:
    (artifact_root / "result.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        destination = _publish(artifact_root, request.output_path)
    except Exception as exc:
        retained = Path(tempfile.mkdtemp(prefix="npa-isaac-arena-unpublished-"))
        retained.chmod(0o700)
        shutil.copytree(artifact_root, retained / "artifacts")
        raise IsaacArenaError(
            f"artifact publication failed; evidence retained privately at {retained}"
        ) from exc
    return {**manifest, "published_to": destination}


def evaluate(
    request: IsaacArenaRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    graphics_preparer: Callable[
        [Path, dict[str, str]], dict[str, Any]
    ] = _prepare_viewport_graphics,
    video_preparer: Callable[[Path], tuple[Path, dict[str, Any]]] = _denoise_mp4,
) -> dict[str, Any]:
    """Execute real Arena evaluation and retain truthful scored or failed evidence.

    Args:
        request: Scored policy evaluation and optional visual qualification.
        runner: Upstream process executor, injectable for offline contract tests.
        graphics_preparer: Verify or privately prepare native graphics userspace.
        video_preparer: Create the declared denoised derivative without altering source.
    Returns:
        Published evaluation manifest with metrics and independently hashed artifacts.
    Raises:
        IsaacArenaError: Invalid request or runtime/evidence failure; runtime artifacts are retained.
    """
    _validate(request)
    with tempfile.TemporaryDirectory(prefix="npa-isaac-arena-") as scratch:
        artifact_root, private_dir, base, argv = _prepare_evaluation(
            request, Path(scratch)
        )
        if request.dry_run:
            return {**base, "status": "dry_run", "artifacts": {}}
        try:
            run_dir, log_text = _execute_upstream(
                request,
                artifact_root,
                private_dir,
                base,
                runner,
                graphics_preparer,
                argv,
            )
            manifest = _evaluation_result(
                request, artifact_root, run_dir, log_text, base, video_preparer
            )
        except IsaacArenaError as exc:
            failure = {
                **base,
                "status": "failed",
                "run_id": request.run_id,
                "error": str(exc),
                "artifacts": _artifact_entries(artifact_root, {}),
            }
            _save_result(request, artifact_root, failure)
            raise
        return _save_result(request, artifact_root, manifest)
