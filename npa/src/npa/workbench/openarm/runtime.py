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
from importlib import metadata
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workbench.openarm.schemas import OpenArmRunRequest, OpenArmSystemInfo

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


def _sample_targets(model: Any, phase: float) -> Any:
    import numpy as np

    lower = np.asarray(model.actuator_ctrlrange[:, 0], dtype=float)
    upper = np.asarray(model.actuator_ctrlrange[:, 1], dtype=float)
    center = (lower + upper) / 2.0
    amplitude = np.minimum((upper - lower) * 0.12, 0.18)
    offsets = np.sin(phase + np.arange(model.nu) * 0.31)
    return np.clip(center + amplitude * offsets, lower, upper)


def _render_video(model: Any, data: Any, commands: list[Any], output: Path) -> None:
    import imageio.v2 as imageio
    import mujoco

    frames = []
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        mujoco.mj_resetData(model, data)
        for index, command in enumerate(commands):
            data.ctrl[:] = command
            mujoco.mj_step(model, data)
            if index % 5 == 0:
                renderer.update_scene(data)
                frames.append(renderer.render().copy())
        if not frames:
            raise OpenArmError("MuJoCo renderer produced no frames")
        imageio.mimwrite(output, frames, fps=30, codec="libx264")
    finally:
        renderer.close()


def _run_mujoco(request: OpenArmRunRequest, output_dir: Path) -> dict[str, Any]:
    import mujoco
    import numpy as np
    import openarm_mujoco.v2 as openarm

    model_path = Path(openarm.openarm_demo_xml()).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    resolver = openarm.JointResolver(model)
    commands: list[Any] = []
    samples: list[Any] = []
    energies: list[float] = []
    for step in range(request.steps):
        command = _sample_targets(model, step * 0.025)
        data.ctrl[:] = command
        mujoco.mj_step(model, data)
        commands.append(command.copy())
        if step % max(1, request.steps // 100) == 0 or step + 1 == request.steps:
            right, _ = resolver.get_driver(data.qpos, "right")
            left, _ = resolver.get_driver(data.qpos, "left")
            samples.append(np.concatenate((right, left)))
            energies.append(float(np.dot(data.qvel, data.qvel)))
    trace = output_dir / "mujoco_trajectory.npz"
    np.savez_compressed(
        trace,
        joint_position=np.asarray(samples),
        command=np.asarray(commands),
        velocity_energy=np.asarray(energies),
    )
    if not np.isfinite(np.asarray(samples)).all() or float(data.time) <= 0:
        raise OpenArmError("MuJoCo produced non-finite state or did not advance time")
    artifact = {
        "path": trace.name,
        "bytes": trace.stat().st_size,
        "sha256": _sha256(trace),
    }
    if request.render:
        video = output_dir / "mujoco_rollout.mp4"
        _render_video(model, data, commands, video)
        artifact["video"] = {
            "path": video.name,
            "bytes": video.stat().st_size,
            "sha256": _sha256(video),
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
    result["finite_metrics"] = all(
        not isinstance(value, float) or math.isfinite(value)
        for value in result.values()
    )
    _write_json(output_dir / "result.json", result)
    _upload(output_dir, request.output_uri)
    return result
