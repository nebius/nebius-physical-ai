"""Real OpenArm workloads shared by every public surface.

Heavy simulator imports are deferred so client-side CLI and SDK imports do not
require MuJoCo or Isaac. The Isaac subprocess runs only through the repository's
runtime-fetch shim; this module never installs or silently substitutes a simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workbench.openarm.schemas import (
    OpenArmQualificationRequest,
    OpenArmQualificationResponse,
    OpenArmRunRequest,
    OpenArmSystemInfo,
)

MUJOCO_COMMIT = "a8c979629f2591ad035d99d338ce114969e6cddc"
ISAAC_COMMIT = "bad82e23716e6941c2de78ccb978f57c78b37734"


class OpenArmError(RuntimeError):
    """A real OpenArm simulator workload failed."""


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def system_info() -> OpenArmSystemInfo:
    """Return the packaged stack identity without triggering Isaac download."""
    return OpenArmSystemInfo(
        python=platform.python_version(),
        platform=platform.platform(),
        mujoco_version=_version("mujoco"),
        openarm_mujoco_version=_version("openarm-mujoco"),
        openarm_mujoco_commit=os.environ.get("OPENARM_MUJOCO_COMMIT", MUJOCO_COMMIT),
        openarm_isaac_commit=os.environ.get("OPENARM_ISAAC_COMMIT", ISAAC_COMMIT),
        isaac_sim_version=os.environ.get("ISAAC_SIM_VERSION", ""),
        isaac_lab_version=os.environ.get("ISAAC_LAB_VERSION", ""),
    )


def manifest_sha256(request: OpenArmRunRequest) -> str:
    """Return a stable hash of one validated workload request."""
    payload = json.dumps(
        request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def run_id(request: OpenArmRunRequest) -> str:
    """Derive a stable run id from the request."""
    return f"openarm-{request.simulator}-{manifest_sha256(request)[:12]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _arm_actuator_indices(model: Any) -> Any:
    """Resolve the 16 bimanual actuators without commanding the cell lifter."""
    import mujoco
    import numpy as np

    names = [
        *(f"right_joint{index}_ctrl" for index in range(1, 8)),
        "right_finger1_ctrl",
        *(f"left_joint{index}_ctrl" for index in range(1, 8)),
        "left_finger1_ctrl",
    ]
    indices = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in names
        ],
        dtype=np.intp,
    )
    if np.any(indices < 0):
        raise OpenArmError("OpenArm bimanual actuator mapping is incomplete")
    return indices


def _sample_targets(model: Any, phase: float) -> Any:
    import numpy as np

    indices = _arm_actuator_indices(model)
    lower = np.asarray(model.actuator_ctrlrange[indices, 0], dtype=float)
    upper = np.asarray(model.actuator_ctrlrange[indices, 1], dtype=float)
    center = (lower + upper) / 2.0
    amplitude = np.minimum((upper - lower) * 0.12, 0.18)
    offsets = np.sin(phase + np.arange(indices.size) * 0.31)
    return np.clip(center + amplitude * offsets, lower, upper)


def _render_video(model: Any, data: Any, commands: list[Any], output: Path) -> None:
    import imageio.v2 as imageio
    import mujoco

    frames = []
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        mujoco.mj_resetData(model, data)
        actuator_indices = _arm_actuator_indices(model)
        for index, command in enumerate(commands):
            data.ctrl[actuator_indices] = command
            mujoco.mj_step(model, data)
            if index % 5 == 0:
                renderer.update_scene(data)
                frames.append(renderer.render().copy())
        if not frames:
            raise OpenArmError("MuJoCo renderer produced no frames")
        imageio.mimwrite(output, frames, fps=30, codec="libx264")
    finally:
        renderer.close()


def _step_mujoco(
    model: Any, data: Any, resolver: Any, steps: int
) -> tuple[list[Any], list[Any], list[float]]:
    import mujoco
    import numpy as np

    commands: list[Any] = []
    samples: list[Any] = []
    energies: list[float] = []
    for step in range(steps):
        command = _sample_targets(model, step * 0.025)
        resolver.set_ctrl(data.ctrl, command[:8], "right")
        resolver.set_ctrl(data.ctrl, command[8:], "left")
        mujoco.mj_step(model, data)
        commands.append(command.copy())
        if step % max(1, steps // 100) == 0 or step + 1 == steps:
            right, _ = resolver.get_driver(data.qpos, "right")
            left, _ = resolver.get_driver(data.qpos, "left")
            samples.append(np.concatenate((right, left)))
            energies.append(float(np.dot(data.qvel, data.qvel)))
    return commands, samples, energies


def _write_mujoco_trace(
    output_dir: Path, commands: list[Any], samples: list[Any], energies: list[float]
) -> Path:
    import numpy as np

    trace = output_dir / "mujoco_trajectory.npz"
    np.savez_compressed(
        trace,
        joint_position=np.asarray(samples),
        command=np.asarray(commands),
        velocity_energy=np.asarray(energies),
    )
    return trace


def _mujoco_result(
    request: OpenArmRunRequest,
    model_path: Path,
    model: Any,
    data: Any,
    samples: list[Any],
    energies: list[float],
    trace: Path,
) -> dict[str, Any]:
    import numpy as np

    if not np.isfinite(np.asarray(samples)).all() or float(data.time) <= 0:
        raise OpenArmError("MuJoCo produced non-finite state or did not advance time")
    artifact = {
        "path": trace.name,
        "bytes": trace.stat().st_size,
        "sha256": _sha256(trace),
    }
    return {
        "schema": "npa.openarm.mujoco_rollout.v1",
        "status": "completed",
        "simulator": "mujoco",
        "upstream_commit": os.environ.get("OPENARM_MUJOCO_COMMIT", MUJOCO_COMMIT),
        "model": model_path.name,
        "model_dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu},
        "steps": request.steps,
        "simulation_seconds": float(data.time),
        "sample_count": len(samples),
        "final_joint_l2": float(np.linalg.norm(samples[-1])),
        "max_velocity_energy": max(energies),
        "artifact": artifact,
    }


def _run_mujoco(request: OpenArmRunRequest, output_dir: Path) -> dict[str, Any]:
    import mujoco
    import openarm_mujoco.v2 as openarm

    model_path = Path(openarm.openarm_demo_xml()).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    resolver = openarm.JointResolver(model)
    commands, samples, energies = _step_mujoco(model, data, resolver, request.steps)
    trace = _write_mujoco_trace(output_dir, commands, samples, energies)
    result = _mujoco_result(request, model_path, model, data, samples, energies, trace)
    if request.render:
        video = output_dir / "mujoco_rollout.mp4"
        _render_video(model, data, commands, video)
        result["artifact"]["video"] = {
            "path": video.name,
            "bytes": video.stat().st_size,
            "sha256": _sha256(video),
        }
    return result


def _isaac_command(request: OpenArmRunRequest, output_dir: Path) -> list[str]:
    python = os.environ.get("ISAAC_LAB_PYTHON", "/isaac-sim/python.sh")
    runner = os.environ.get("OPENARM_ISAAC_RUNNER", "/opt/npa/openarm/isaac_runner.py")
    command = [
        python,
        runner,
        "--mode",
        request.isaac_mode,
        "--task",
        request.task,
        "--num-envs",
        str(request.num_envs),
        "--steps",
        str(request.steps),
        "--seed",
        str(request.seed),
        "--output-dir",
        str(output_dir),
        "--headless",
    ]
    if request.isaac_mode == "train":
        command.extend(("--max-iterations", str(request.max_iterations)))
    return command


def _run_isaac(request: OpenArmRunRequest, output_dir: Path) -> dict[str, Any]:
    command = _isaac_command(request, output_dir)
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        raise OpenArmError(
            f"cannot launch Isaac runtime-fetch interpreter: {exc}"
        ) from exc
    (output_dir / "isaac_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "isaac_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-2000:]
        raise OpenArmError(f"OpenArm Isaac Lab {request.isaac_mode} failed: {tail}")
    result_path = output_dir / "isaac_result.json"
    if not result_path.is_file():
        raise OpenArmError(
            "Isaac workload exited successfully without isaac_result.json"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("simulator") != "isaac-lab":
        raise OpenArmError(
            f"Isaac workload returned an invalid terminal result: {result}"
        )
    return result


def _upload(output_dir: Path, output_uri: str) -> str:
    client = StorageClient.from_environment()
    return client.upload_directory(str(output_dir), output_uri)


def _load_result(root: Path, stage: str, expected_schema: str) -> dict[str, Any]:
    result_path = root / stage / "result.json"
    if not result_path.is_file():
        raise OpenArmError(f"qualification input is missing {stage}/result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != expected_schema or result.get("status") != "completed":
        raise OpenArmError(f"{stage} has no completed {expected_schema} result")
    if result.get("finite_metrics") is not True:
        raise OpenArmError(f"{stage} did not certify finite metrics")
    return result


def _artifact_record(root: Path, stage: str, relative: str) -> dict[str, Any]:
    stage_root = (root / stage).resolve()
    artifact = (stage_root / relative).resolve()
    if not artifact.is_relative_to(stage_root) or not artifact.is_file():
        raise OpenArmError(f"{stage} artifact is missing or unsafe: {relative}")
    size = artifact.stat().st_size
    if size <= 0:
        raise OpenArmError(f"{stage} artifact is empty: {relative}")
    _validate_artifact_content(artifact, stage)
    return {
        "stage": stage,
        "path": relative,
        "bytes": size,
        "sha256": _sha256(artifact),
    }


def _validate_npz(path: Path, stage: str) -> None:
    import numpy as np

    expected = {
        "mujoco": {"joint_position", "command", "velocity_energy"},
        "isaac-rollout": {"reward", "policy_observation"},
    }[stage]
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not expected.issubset(payload.files):
                raise OpenArmError(f"{stage} NPZ is missing arrays: {sorted(expected)}")
            for key in expected:
                values = payload[key]
                if values.size == 0 or not np.isfinite(values).all():
                    raise OpenArmError(f"{stage} NPZ has invalid {key} values")
    except (OSError, ValueError) as exc:
        raise OpenArmError(f"{stage} NPZ is unreadable: {exc}") from exc


def _validate_checkpoint(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise OpenArmError(
            f"Isaac training checkpoint is not a Torch archive: {path.name}"
        )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if not any(name.endswith("data.pkl") for name in names):
        raise OpenArmError(
            f"Isaac training checkpoint has no serialized state: {path.name}"
        )


def _validate_video(path: Path) -> None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise OpenArmError(f"cannot probe MuJoCo video: {exc}") from exc
    try:
        streams = json.loads(completed.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise OpenArmError(f"MuJoCo video probe was invalid: {exc}") from exc
    if completed.returncode or not streams:
        raise OpenArmError("MuJoCo video has no decodable video stream")


def _validate_artifact_content(path: Path, stage: str) -> None:
    if stage in {"mujoco", "isaac-rollout"} and path.suffix == ".npz":
        _validate_npz(path, stage)
    elif stage == "mujoco" and path.suffix == ".mp4":
        _validate_video(path)
    elif stage == "isaac-training":
        _validate_checkpoint(path)


def _verify_declaration(record: dict[str, Any], declared: dict[str, Any]) -> None:
    for key in ("bytes", "sha256"):
        expected = declared.get(key)
        if expected is not None and record[key] != expected:
            raise OpenArmError(
                f"artifact {record['path']} does not match declared {key}"
            )


def _declared_artifacts(
    root: Path, stage: str, result: dict[str, Any]
) -> list[dict[str, Any]]:
    if stage == "isaac-training":
        declarations = [{"path": path} for path in result.get("checkpoints", [])]
    else:
        artifact = result.get("artifact", {})
        declarations = [artifact]
        video = artifact.get("video", {})
        if stage == "mujoco" and video:
            declarations.append(video)
    paths = [row.get("path", "") for row in declarations]
    if not paths or not all(isinstance(path, str) and path for path in paths):
        raise OpenArmError(f"{stage} result declares no usable artifacts")
    records = [_artifact_record(root, stage, path) for path in paths]
    for record, declared in zip(records, declarations, strict=True):
        _verify_declaration(record, declared)
    return records


def _validate_qualification_tree(root: Path) -> list[dict[str, Any]]:
    expected = {
        "mujoco": "npa.openarm.mujoco_rollout.v1",
        "isaac-rollout": "npa.openarm.isaac_lab_rollout.v1",
        "isaac-training": "npa.openarm.isaac_lab_training.v1",
    }
    artifacts: list[dict[str, Any]] = []
    for stage, schema in expected.items():
        result = _load_result(root, stage, schema)
        artifacts.extend(_declared_artifacts(root, stage, result))
    return artifacts


def qualify(request: OpenArmQualificationRequest) -> OpenArmQualificationResponse:
    """Validate every declared artifact from a completed dual-simulator run."""
    client = StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-openarm-qualification-") as temporary:
        root = Path(temporary) / "input"
        report_root = Path(temporary) / "report"
        root.mkdir()
        report_root.mkdir()
        client.download_directory(request.input_uri, str(root))
        artifacts = _validate_qualification_tree(root)
        response = OpenArmQualificationResponse(
            schema="npa.openarm.qualification.v1",
            status="completed",
            input_uri=request.input_uri,
            output_uri=request.output_uri,
            artifacts=artifacts,
        )
        _write_json(
            report_root / "qualification.json", response.model_dump(by_alias=True)
        )
        client.upload_directory(str(report_root), request.output_uri)
    return response


def _all_finite(value: Any) -> bool:
    """Recursively reject non-finite numbers in result-shaped values."""
    import numpy as np

    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, np.ndarray):
        try:
            return bool(np.isfinite(value).all())
        except TypeError:
            return _all_finite(value.tolist())
    if isinstance(value, np.generic):
        try:
            return bool(np.isfinite(value))
        except TypeError:
            return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def run(
    request: OpenArmRunRequest, *, output_dir: Path | None = None
) -> dict[str, Any]:
    """Execute one workload, validate artifacts, and upload the whole evidence tree."""
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="npa-openarm-") as temporary:
            return run(request, output_dir=Path(temporary))
    output_dir.mkdir(parents=True, exist_ok=True)
    if request.simulator == "mujoco":
        result = _run_mujoco(request, output_dir)
    else:
        result = _run_isaac(request, output_dir)
    result["request_sha256"] = manifest_sha256(request)
    result["output_uri"] = request.output_uri
    result["finite_metrics"] = _all_finite(result)
    _write_json(output_dir / "result.json", result)
    _upload(output_dir, request.output_uri)
    return result
