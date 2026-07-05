"""Cold-start multi-node SkyPilot burst jobs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, TextIO

import yaml

from npa.orchestration.skypilot._bin import (
    REQUIRED_SKYPILOT_VERSION,
    SkyBin,
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    resolve_config,
)
from npa.orchestration.skypilot.cleanup import sky_environment
from npa.orchestration.skypilot.controller import (
    DEFAULT_CONTROLLER_BACKEND,
    ControllerBackend,
    apply_controller_override,
)


_ACCELERATOR_SPEC_RE = re.compile(r"^[A-Za-z0-9_.-]+:[1-9][0-9]*$")
_JOB_NAME_RE = re.compile(r"^[a-zA-Z0-9]+(?:[._-]{1,2}[a-zA-Z0-9]+)*$")
_DEFAULT_MASTER_PORT = 29500


class BurstConfigError(ValueError):
    """Raised when a burst request cannot be represented as a SkyPilot task."""


class BurstSubmitError(RuntimeError):
    """Raised when the SkyPilot Python API rejects a burst operation."""


@dataclass(frozen=True)
class BurstSpec:
    """Single gang-scheduled multi-node burst job request."""

    image: str
    num_nodes: int
    gpu_per_node: str
    entrypoint: str
    name: str = "npa-burst"
    cloud: str = "nebius"
    master_port: int = _DEFAULT_MASTER_PORT


@dataclass(frozen=True)
class BurstJobHandle:
    """Serializable handle for a submitted SkyPilot managed job."""

    job_id: str
    name: str
    submitted_yaml_path: str = ""
    config_path: str = ""
    isolated_config_dir: str = ""
    sky_bin: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "BurstJobHandle":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise BurstConfigError("Burst handle JSON must be an object")
        return cls(
            job_id=str(data.get("job_id") or ""),
            name=str(data.get("name") or ""),
            submitted_yaml_path=str(data.get("submitted_yaml_path") or ""),
            config_path=str(data.get("config_path") or ""),
            isolated_config_dir=str(data.get("isolated_config_dir") or ""),
            sky_bin=str(data.get("sky_bin") or ""),
            raw=dict(data.get("raw") or {}),
        )


@dataclass(frozen=True)
class BurstStatus:
    """Status response for a burst job."""

    job_id: str
    status: str
    records: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class BurstLogs:
    """Log response for a burst job."""

    job_id: str
    text: str
    exit_code: int | None = None
    error: str = ""


def build_task_spec(spec: BurstSpec) -> dict[str, Any]:
    """Build a SkyPilot 0.12.2 task spec for one coupled multi-node job."""

    _validate_spec(spec)
    return {
        "name": spec.name,
        "num_nodes": spec.num_nodes,
        "resources": {
            "cloud": spec.cloud,
            "accelerators": spec.gpu_per_node,
            "image_id": _docker_image_id(spec.image),
        },
        "envs": {
            "BURST_ENTRYPOINT": spec.entrypoint,
            "BURST_MASTER_PORT": str(spec.master_port),
        },
        "setup": _setup_script(),
        "run": _run_script(),
    }


def task_yaml(spec: BurstSpec) -> str:
    """Render the generated SkyPilot task YAML."""

    return yaml.safe_dump(build_task_spec(spec), sort_keys=False)


def submit(
    *,
    image: str,
    num_nodes: int,
    gpu_per_node: str,
    entrypoint: str,
    name: str = "npa-burst",
    sky_bin: SkyBin = None,
    config_path: Path | str | None = None,
    isolated_config_dir: Path | str | None = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
) -> BurstJobHandle:
    """Submit a burst job through SkyPilot's Python managed-jobs API."""

    spec = BurstSpec(
        image=image,
        num_nodes=num_nodes,
        gpu_per_node=gpu_per_node,
        entrypoint=entrypoint,
        name=name,
    )
    runtime = _prepare_runtime(
        name=name,
        sky_bin=sky_bin,
        config_path=config_path,
        isolated_config_dir=isolated_config_dir,
        controller_backend=controller_backend,
    )
    yaml_path = runtime.submission_dir / "burst-task.yaml"
    yaml_path.write_text(task_yaml(spec), encoding="utf-8")

    result = _run_sky_api(
        runtime,
        "launch",
        {"yaml_path": str(yaml_path), "name": name},
    )
    job_ids = result.get("job_ids") or []
    job_id = str(job_ids[0]) if job_ids else ""
    if not job_id:
        raise BurstSubmitError(f"SkyPilot launch did not return a job id: {result!r}")
    return BurstJobHandle(
        job_id=job_id,
        name=name,
        submitted_yaml_path=str(yaml_path),
        config_path=str(runtime.config_path),
        isolated_config_dir=str(runtime.isolated_config_dir or ""),
        sky_bin=str(runtime.sky_bin),
        raw=result,
    )


def status(handle: BurstJobHandle | str, **kwargs: Any) -> BurstStatus:
    """Return SkyPilot managed-job status for a burst handle or job id."""

    handle = _coerce_handle(handle)
    runtime = _prepare_runtime_from_handle(handle, **kwargs)
    result = _run_sky_api(
        runtime,
        "queue",
        {"job_id": int(handle.job_id), "refresh": True, "skip_finished": False},
    )
    records = _records_from_queue_result(result)
    state = _status_from_records(records, handle.job_id)
    return BurstStatus(job_id=handle.job_id, status=state or "UNKNOWN", records=records)


def logs(
    handle: BurstJobHandle | str,
    *,
    follow: bool = False,
    tail: int | None = None,
    output_stream: TextIO | None = None,
    **kwargs: Any,
) -> BurstLogs:
    """Return or stream logs for a burst managed job."""

    handle = _coerce_handle(handle)
    runtime = _prepare_runtime_from_handle(handle, **kwargs)
    result = _run_sky_api(
        runtime,
        "logs",
        {
            "job_id": int(handle.job_id),
            "follow": follow,
            "refresh": True,
            "tail": tail,
        },
    )
    text = str(result.get("logs") or "")
    if output_stream is not None and text:
        output_stream.write(text)
        output_stream.flush()
    exit_code = result.get("exit_code")
    return BurstLogs(
        job_id=handle.job_id,
        text=text,
        exit_code=int(exit_code) if isinstance(exit_code, int) else None,
    )


def _validate_spec(spec: BurstSpec) -> None:
    if not spec.image.strip():
        raise BurstConfigError("image must be non-empty")
    if not isinstance(spec.num_nodes, int) or spec.num_nodes <= 0:
        raise BurstConfigError("num_nodes must be a positive integer")
    if not _ACCELERATOR_SPEC_RE.fullmatch(spec.gpu_per_node.strip()):
        raise BurstConfigError(
            "gpu_per_node must be a SkyPilot accelerator spec such as '<GPU_TYPE>:<COUNT>'"
        )
    if not spec.entrypoint.strip():
        raise BurstConfigError("entrypoint must be non-empty")
    if not _JOB_NAME_RE.fullmatch(spec.name):
        raise BurstConfigError("name must be a valid SkyPilot task name")
    if not isinstance(spec.master_port, int) or spec.master_port <= 0:
        raise BurstConfigError("master_port must be a positive integer")


def _docker_image_id(image: str) -> str:
    value = image.strip()
    if value.startswith("docker:"):
        return value
    return f"docker:{value}"


def _setup_script() -> str:
    return """set -euo pipefail
command -v torchrun >/dev/null 2>&1 || {
  echo "npa burst requires torchrun in the selected image" >&2
  exit 2
}
"""


def _run_script() -> str:
    return """set -euo pipefail

mapfile -t __npa_burst_node_ips <<< "${SKYPILOT_NODE_IPS}"
MASTER_ADDR="${__npa_burst_node_ips[0]}"
MASTER_PORT="${BURST_MASTER_PORT:-29500}"
WORLD_SIZE=$((SKYPILOT_NUM_NODES * SKYPILOT_NUM_GPUS_PER_NODE))

export MASTER_ADDR MASTER_PORT WORLD_SIZE
export RANK="${SKYPILOT_NODE_RANK}"
export LOCAL_WORLD_SIZE="${SKYPILOT_NUM_GPUS_PER_NODE}"

printf 'NPA_BURST_DISTRIBUTED rank=%s world_size=%s master_addr=%s master_port=%s node_ips=%q gpus_per_node=%s\\n' \
  "${SKYPILOT_NODE_RANK}" "${WORLD_SIZE}" "${MASTER_ADDR}" "${MASTER_PORT}" \
  "${SKYPILOT_NODE_IPS}" "${SKYPILOT_NUM_GPUS_PER_NODE}"

exec torchrun \
  --nnodes="${SKYPILOT_NUM_NODES}" \
  --nproc-per-node="${SKYPILOT_NUM_GPUS_PER_NODE}" \
  --node-rank="${SKYPILOT_NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  bash -lc "${BURST_ENTRYPOINT}"
"""


@dataclass(frozen=True)
class _Runtime:
    sky_bin: Path
    sky_python: Path
    config_path: Path
    isolated_config_dir: Path | None
    submission_dir: Path
    env: dict[str, str]
    cwd: str


def _prepare_runtime(
    *,
    name: str,
    sky_bin: SkyBin = None,
    config_path: Path | str | None = None,
    isolated_config_dir: Path | str | None = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
) -> _Runtime:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    sky_python = _sky_python_from_bin(runtime_config.sky_bin)
    submission_dir = _submission_dir(name, runtime_config.isolated_config_dir)
    generated_config_path = submission_dir / "skypilot-config.yaml"
    base_config = _load_base_config(runtime_config.global_config_path)
    generated_config_path.write_text(
        yaml.safe_dump(
            apply_controller_override(base_config, controller_backend=controller_backend),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env = sky_environment(runtime_config.isolated_config_dir)
    env["SKYPILOT_GLOBAL_CONFIG"] = str(generated_config_path)
    env["PYTHONPATH"] = _pythonpath_for_bridge(env)
    return _Runtime(
        sky_bin=runtime_config.sky_bin,
        sky_python=sky_python,
        config_path=generated_config_path,
        isolated_config_dir=runtime_config.isolated_config_dir,
        submission_dir=submission_dir,
        env=env,
        cwd=_stable_cwd(runtime_config.isolated_config_dir),
    )


def _prepare_runtime_from_handle(handle: BurstJobHandle, **kwargs: Any) -> _Runtime:
    return _prepare_runtime(
        name=handle.name or f"npa-burst-{handle.job_id}",
        sky_bin=kwargs.pop("sky_bin", None) or handle.sky_bin or None,
        config_path=kwargs.pop("config_path", None) or handle.config_path or None,
        isolated_config_dir=(
            kwargs.pop("isolated_config_dir", None) or handle.isolated_config_dir or None
        ),
        controller_backend=kwargs.pop("controller_backend", DEFAULT_CONTROLLER_BACKEND),
    )


def _run_sky_api(runtime: _Runtime, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    command = [str(runtime.sky_python), "-m", "npa.burst._sky_api", action]
    result = subprocess.run(
        command,
        input=json.dumps(dict(payload)),
        env=runtime.env,
        cwd=runtime.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise BurstSubmitError(f"SkyPilot Python API {action} failed: {detail}")
    try:
        decoded = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise BurstSubmitError(
            f"SkyPilot Python API {action} returned non-json output: {result.stdout!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise BurstSubmitError(f"SkyPilot Python API {action} returned invalid payload: {decoded!r}")
    return decoded


def _sky_python_from_bin(sky_bin: Path) -> Path:
    bin_dir = sky_bin.resolve().parent
    python = bin_dir / ("python.exe" if os.name == "nt" else "python")
    if not python.is_file():
        raise SkyPilotNotInstalledError(
            f"SkyPilot Python executable not found next to sky binary: {python}"
        )
    result = subprocess.run(
        [str(python), "-c", "import sky; print(getattr(sky, '__version__', 'unknown'))"],
        capture_output=True,
        text=True,
        check=False,
    )
    actual = result.stdout.strip()
    if result.returncode != 0:
        raise SkyPilotVersionError(f"Unable to import SkyPilot via {python}: {result.stderr.strip()}")
    if actual != REQUIRED_SKYPILOT_VERSION:
        raise SkyPilotVersionError(
            f"SkyPilot version mismatch: expected {REQUIRED_SKYPILOT_VERSION}, got {actual}"
        )
    return python


def _load_base_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SkyPilotConfigError(f"SkyPilot global config must be a mapping: {config_path}")
    return data


def _submission_dir(name: str, isolated_config_dir: Path | None) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "npa-burst"
    if isolated_config_dir is None:
        root = Path(tempfile.mkdtemp(prefix=f"npa-burst-{safe_name}-"))
    else:
        root = Path(isolated_config_dir) / "burst-submissions" / safe_name
        root.mkdir(parents=True, exist_ok=True)
    return root


def _stable_cwd(isolated_config_dir: Path | None) -> str:
    for candidate in (isolated_config_dir, Path.home()):
        if candidate is not None and Path(candidate).is_dir():
            return str(candidate)
    return str(Path.home())


def _pythonpath_for_bridge(env: Mapping[str, str]) -> str:
    src_root = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH", "")
    if existing:
        return f"{src_root}{os.pathsep}{existing}"
    return str(src_root)


def _coerce_handle(handle: BurstJobHandle | str) -> BurstJobHandle:
    if isinstance(handle, BurstJobHandle):
        return handle
    text = str(handle).strip()
    if text.startswith("{"):
        return BurstJobHandle.from_json(text)
    return BurstJobHandle(job_id=text, name=f"npa-burst-{text}")


def _records_from_queue_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = result.get("records") or result.get("jobs") or []
    if isinstance(records, list):
        return [dict(record) for record in records if isinstance(record, dict)]
    return []


def _status_from_records(records: list[dict[str, Any]], job_id: str) -> str:
    for record in records:
        if str(record.get("job_id") or record.get("id") or "") == str(job_id):
            return str(record.get("status") or "").upper()
    return ""
