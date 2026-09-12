"""Shared execution path for genuine Isaac Lab-Arena policy evaluation.

The CLI, SDK, and ``npa.workflow`` toolRef all call this module.  The worker
image contains the immutable Apache-2.0 Arena source, but Isaac Sim and Isaac
Lab remain operator-authorized runtime downloads in the inherited NPA cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable
from urllib.parse import urlparse

from npa.clients.storage import StorageClient

ISAAC_ARENA_VERSION = "0.3.0"
ISAAC_ARENA_REVISION = "ed0fd12be862078be316c73eb7cf423ba9b1c5cd"
ISAAC_ARENA_ROOT = "/opt/isaac-arena"
ARTIFACT_SCHEMA = "npa.workbench.isaac_arena.evaluation.v1"
SUPPORTED_POLICIES = frozenset({"zero_action", "replay", "rsl_rl"})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class IsaacArenaError(RuntimeError):
    """Raised when an Arena request or upstream evaluation is invalid."""


@dataclass(frozen=True)
class IsaacArenaRequest:
    """One reproducible policy evaluation request."""

    output_path: str
    environment: str = "cube_goal_pose"
    policy_type: str = "zero_action"
    input_path: str = ""
    num_episodes: int = 1
    num_envs: int = 1
    seed: int = 42
    embodiment: str = ""
    object_name: str = ""
    record_video: bool = False
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


def _validate(request: IsaacArenaRequest) -> None:
    parsed = urlparse(request.output_path)
    if not request.output_path.strip():
        raise IsaacArenaError("output_path is required")
    if parsed.scheme and parsed.scheme != "s3":
        raise IsaacArenaError("output_path must be a local path or s3:// prefix")
    if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
        raise IsaacArenaError("output_path must include an S3 bucket and prefix")
    if request.policy_type not in SUPPORTED_POLICIES:
        raise IsaacArenaError(
            "policy_type must be one of: " + ", ".join(sorted(SUPPORTED_POLICIES))
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
    if request.num_envs > request.num_episodes:
        raise IsaacArenaError("num_envs cannot exceed num_episodes")
    if request.policy_type == "zero_action" and request.input_path:
        raise IsaacArenaError("zero_action does not accept input_path")
    if request.policy_type != "zero_action" and not request.input_path:
        raise IsaacArenaError(f"{request.policy_type} requires input_path")


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
    """Build argv for upstream's genuine ``policy_runner.py``."""

    argv = [
        os.environ.get("ISAAC_ARENA_PYTHON", "/isaac-sim/python.sh"),
        f"{ISAAC_ARENA_ROOT}/isaaclab_arena/evaluation/policy_runner.py",
        "--headless",
        "--device",
        "cuda:0",
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
    if request.record_video:
        argv.append("--record_viewport_video")
    if request.policy_type == "replay":
        if local_input is None or not local_input.is_file():
            raise IsaacArenaError("replay input_path must resolve to one HDF5 file")
        argv.extend(["--replay_file_path", str(local_input)])
    elif request.policy_type == "rsl_rl":
        if local_input is None:
            raise IsaacArenaError("rsl_rl requires a checkpoint input")
        checkpoint = local_input
        if checkpoint.is_dir():
            candidates = sorted(checkpoint.glob("model*.pt"))
            if not candidates:
                candidates = sorted(checkpoint.rglob("model*.pt"))
            if len(candidates) != 1:
                raise IsaacArenaError(
                    "rsl_rl input directory must contain exactly one model*.pt checkpoint"
                )
            checkpoint = candidates[0]
        if checkpoint.suffix != ".pt" or not checkpoint.is_file():
            raise IsaacArenaError("rsl_rl input_path must resolve to a .pt checkpoint")
        if not (checkpoint.parent / "params" / "agent.yaml").is_file():
            raise IsaacArenaError(
                "rsl_rl checkpoint requires sibling params/agent.yaml"
            )
        argv.extend(["--checkpoint_path", str(checkpoint)])
    # Upstream uses an argparse subparser per environment and explicitly makes
    # its argv order part of the interface: global and policy options first,
    # then the environment name, then environment-specific options.
    argv.append(request.environment)
    if request.embodiment:
        argv.extend(["--embodiment", request.embodiment])
    if request.object_name:
        argv.extend(["--object", request.object_name])
    return argv


def _subprocess_env() -> dict[str, str]:
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
        } or upper.endswith(("_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD")):
            env.pop(key, None)
    env.setdefault("ACCEPT_EULA", "Y")
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize(run_dir: Path) -> tuple[dict[str, Any], list[Path]]:
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
    try:
        lengths = [int(record["episode_length"]) for record in records]
    except (KeyError, TypeError, ValueError) as exc:
        raise IsaacArenaError(
            "upstream evaluation recorded no integer episode length"
        ) from exc
    if any(length <= 0 for length in lengths):
        raise IsaacArenaError("upstream evaluation recorded an invalid episode length")
    return (
        {
            "episodes": len(records),
            "successes": sum(bool(record["success"]) for record in records),
            "success_rate": sum(bool(record["success"]) for record in records)
            / len(records),
            "mean_episode_length": sum(lengths) / len(lengths),
        },
        result_files,
    )


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        return {
            "available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "",
            "compute_capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else [],
        }
    except Exception:
        return {"available": False, "device_name": "", "compute_capability": []}


def _probe_mp4(path: Path) -> dict[str, Any]:
    """Require a decodable video stream with dimensions and positive duration."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout or "{}")
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
        codec = str(stream["codec_name"])
        width = int(stream["width"])
        height = int(stream["height"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IsaacArenaError(f"invalid viewport MP4: {path.name}") from exc
    if (
        completed.returncode != 0
        or not codec
        or width <= 0
        or height <= 0
        or duration <= 0
    ):
        raise IsaacArenaError(f"invalid viewport MP4: {path.name}")
    return {
        "codec": codec,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


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


def evaluate(
    request: IsaacArenaRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run upstream Arena and publish its raw artifacts plus verified summary."""

    _validate(request)
    with tempfile.TemporaryDirectory(prefix="npa-isaac-arena-") as scratch:
        root = Path(scratch)
        output_root = root / "upstream"
        output_root.mkdir()
        local_input = None if request.dry_run else _local_input(request, root)
        argv = build_evaluation_argv(
            request, output_dir=output_root, local_input=local_input
        )
        base: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "upstream": {
                "repository": "https://github.com/isaac-sim/IsaacLab-Arena",
                "version": ISAAC_ARENA_VERSION,
                "revision": ISAAC_ARENA_REVISION,
            },
            "request": asdict(request),
            "runtime": {
                "image": request.runtime_image or os.environ.get("NPA_TASK_IMAGE", ""),
                "isaac_runtime_fetch": True,
                "model_baked": False,
                "dataset_baked": False,
            },
            "argv": argv,
        }
        if request.dry_run:
            return {**base, "status": "dry_run", "artifacts": {}}

        completed = runner(
            argv,
            cwd=ISAAC_ARENA_ROOT,
            env=_subprocess_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = root / "evaluation.log"
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout or "").splitlines()[-40:])
            raise IsaacArenaError(
                f"upstream policy_runner failed ({completed.returncode}):\n{tail}"
            )
        run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
        if len(run_dirs) != 1:
            raise IsaacArenaError(
                "upstream evaluation did not create exactly one run directory"
            )
        run_dir = run_dirs[0]
        summary, result_files = _summarize(run_dir)
        # Upstream writes the top-level report beside the JSONL journal. The
        # ``report/`` directory contains its linked task/job detail pages.
        report = run_dir / "index.html"
        if not report.is_file() or report.stat().st_size == 0:
            raise IsaacArenaError("upstream evaluation report is missing")
        videos = sorted(run_dir.rglob("*.mp4"))
        if request.record_video and (
            not videos or any(path.stat().st_size == 0 for path in videos)
        ):
            raise IsaacArenaError(
                "video recording was requested but no non-empty MP4 was written"
            )
        video_metadata = {path: _probe_mp4(path) for path in videos}

        # Publish the complete upstream report tree, and bind every retained
        # byte—not just its top-level index—to the result manifest.
        artifacts = sorted(path for path in root.rglob("*") if path.is_file())
        manifest = {
            **base,
            "status": "ok",
            "summary": summary,
            "gpu": _gpu_info(),
            "artifacts": [
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    **(
                        {"video": video_metadata[path]}
                        if path in video_metadata
                        else {}
                    ),
                }
                for path in artifacts
            ],
        }
        if not manifest["gpu"]["available"]:
            raise IsaacArenaError("Arena evaluation returned without a CUDA device")
        result_path = root / "result.json"
        result_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        destination = _publish(root, request.output_path)
        return {**manifest, "published_to": destination}
