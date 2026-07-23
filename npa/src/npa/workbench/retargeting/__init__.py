"""Real SONIC motion retargeting/preprocess helpers for Workbench workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient


UPSTREAM_REPO_URL = "https://github.com/NVlabs/GR00T-WholeBodyControl.git"
UPSTREAM_REPO_REF = "a9d20b2ac0949244d94461a1a3263f38c5027c4a"
CONVERTER_SCRIPT = "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py"
BVH_SOMA_SCRIPT = "gear_sonic/data_process/extract_soma_joints_from_bvh.py"
SUPPORTED_SOURCE_FORMATS = (
    "auto",
    "soma-csv",
    "bones-seed-csv",
    "deploy-pkl",
    "teleop-pkl",
    "motion-lib",
    "bvh",
)
SUPPORTED_EMBODIMENTS = ("unitree-g1", "g1", "unitree-g1-sonic")
ROBOT_MOTION_FIELDS = frozenset({"root_trans_offset", "pose_aa", "dof", "root_rot", "fps"})
SOMA_MOTION_FIELDS = frozenset({"soma_joints", "soma_root_quat", "fps", "joint_names"})
METADATA_FILE_NAME = "retargeting_result.json"


class RetargetingError(ValueError):
    """Raised when a retargeting request is invalid or preprocessing fails."""


@dataclass(frozen=True)
class RetargetingResult:
    status: str
    backend: str
    artifact_kind: str
    input_path: str
    output_path: str
    artifact_uri: str
    metadata_uri: str
    metadata_written_uri: str
    source_format: str
    embodiment: str
    retarget_map: str
    frame_rate: int
    source_frame_rate: int
    max_frames: int
    individual: bool
    num_workers: int
    motion_count: int
    output_files: list[str]
    command: list[str]
    upstream_repo: str
    upstream_ref: str
    generated_at: str


__all__ = [
    "RetargetingResult",
    "metadata_uri_for",
    "result_uri_for",
    "run_retargeting",
    "validate_motion_lib",
    "write_metadata",
    "write_result",
]


def run_retargeting(
    *,
    input_path: str,
    output_path: str,
    source_format: str = "auto",
    embodiment: str = "unitree-g1",
    retarget_map: str = "",
    frame_rate: int = 30,
    source_frame_rate: int = 0,
    max_frames: int = 0,
    individual: bool = True,
    num_workers: int = 4,
    dry_run: bool = False,
    sonic_home: str = "",
    storage_client: "StorageClient | None" = None,
) -> RetargetingResult:
    """Run upstream SONIC preprocessors and produce real motion-lib PKLs.

    Supported robot motion-lib inputs are retargeted SOMA/G1 CSV directories,
    Bones-SEED G1 CSV directories, and deploy/teleop PKLs containing G1 joint
    trajectories. Raw BVH can be converted to SONIC's SOMA skeleton PKLs, but
    upstream SONIC does not include the robot retargeter needed for BVH -> G1
    motion-lib conversion.
    """

    normalized_format = _normalize_source_format(source_format)
    normalized_embodiment = _normalize_embodiment(embodiment)
    _validate_common(
        input_path=input_path,
        output_path=output_path,
        source_format=normalized_format,
        embodiment=normalized_embodiment,
        frame_rate=frame_rate,
        source_frame_rate=source_frame_rate,
        max_frames=max_frames,
        num_workers=num_workers,
    )

    metadata_uri = metadata_uri_for(output_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    artifact_kind = "soma_skeleton" if normalized_format == "bvh" else "robot_motion_lib"
    backend = "sonic-bvh-soma-extractor" if normalized_format == "bvh" else "sonic-motion-lib-converter"

    if dry_run:
        command = _planned_command(
            source_format=normalized_format,
            input_path="<staged-input>",
            output_path="<staged-output>",
            frame_rate=frame_rate,
            source_frame_rate=source_frame_rate,
            individual=individual,
            num_workers=num_workers,
        )
        return RetargetingResult(
            status="planned",
            backend=backend,
            artifact_kind=artifact_kind,
            input_path=input_path,
            output_path=output_path,
            artifact_uri=result_uri_for(output_path),
            metadata_uri=metadata_uri,
            metadata_written_uri="",
            source_format=normalized_format,
            embodiment=normalized_embodiment,
            retarget_map=retarget_map,
            frame_rate=frame_rate,
            source_frame_rate=source_frame_rate,
            max_frames=max_frames,
            individual=individual,
            num_workers=num_workers,
            motion_count=0,
            output_files=[],
            command=command,
            upstream_repo=UPSTREAM_REPO_URL,
            upstream_ref=_upstream_ref(),
            generated_at=generated_at,
        )

    with tempfile.TemporaryDirectory(prefix="npa-retargeting-") as tmp:
        work_dir = Path(tmp)
        local_input = _stage_input(input_path, work_dir / "input", storage_client)
        local_output = _local_output_target(
            output_path=output_path,
            work_dir=work_dir,
            source_format=normalized_format,
            input_path=local_input,
            individual=individual,
        )
        upstream_root = _resolve_upstream_root(sonic_home, work_dir)

        if normalized_format == "motion-lib":
            _copy_motion_lib(local_input, local_output)
            command: list[str] = ["copy-motion-lib", str(local_input), str(local_output)]
        else:
            command = _run_upstream_preprocess(
                upstream_root=upstream_root,
                source_format=normalized_format,
                input_path=local_input,
                output_path=local_output,
                frame_rate=frame_rate,
                source_frame_rate=source_frame_rate,
                individual=individual,
                num_workers=num_workers,
            )

        inspect_root = local_output if local_output.is_dir() else local_output.parent
        if max_frames:
            _truncate_motion_files(inspect_root, max_frames=max_frames, artifact_kind=artifact_kind)
        motion_count, output_files = validate_motion_lib(inspect_root, artifact_kind=artifact_kind)
        artifact_uri = _publish_output(local_output, output_path, storage_client)

        result = RetargetingResult(
            status="retargeted" if artifact_kind == "robot_motion_lib" else "soma_extracted",
            backend=backend,
            artifact_kind=artifact_kind,
            input_path=input_path,
            output_path=output_path,
            artifact_uri=artifact_uri,
            metadata_uri=metadata_uri,
            metadata_written_uri="",
            source_format=normalized_format,
            embodiment=normalized_embodiment,
            retarget_map=retarget_map,
            frame_rate=frame_rate,
            source_frame_rate=source_frame_rate,
            max_frames=max_frames,
            individual=individual,
            num_workers=num_workers,
            motion_count=motion_count,
            output_files=output_files,
            command=command,
            upstream_repo=UPSTREAM_REPO_URL,
            upstream_ref=_upstream_ref(),
            generated_at=generated_at,
        )
        payload = _as_payload(result)
        metadata_written_uri = write_metadata(payload, result_uri=metadata_uri, storage_client=storage_client)
        return RetargetingResult(**{**payload, "metadata_written_uri": metadata_written_uri})


def metadata_uri_for(output_path: str) -> str:
    """Return the metadata sidecar URI for a retargeting output path."""

    if output_path.startswith("s3://"):
        if output_path.endswith(".json"):
            return output_path
        return output_path.rstrip("/") + f"/{METADATA_FILE_NAME}"

    path = Path(output_path)
    if path.suffix:
        return str(path.with_suffix(".retargeting_result.json"))
    return str(path / METADATA_FILE_NAME)


def result_uri_for(output_path: str) -> str:
    """Return the real artifact URI for a retargeting output path."""

    if output_path.startswith("s3://"):
        return output_path.rstrip("/") + "/"
    path = Path(output_path)
    return str(path if path.suffix else path)


def write_metadata(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Write retargeting metadata to local disk or S3."""

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-retargeting-metadata-") as tmp:
            local_path = Path(tmp) / METADATA_FILE_NAME
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def write_result(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Compatibility alias for writing the retargeting metadata sidecar."""

    return write_metadata(payload, result_uri=result_uri, storage_client=storage_client)


def validate_motion_lib(path: str | Path, *, artifact_kind: str = "robot_motion_lib") -> tuple[int, list[str]]:
    """Validate generated SONIC motion PKLs and return motion count plus files."""

    root = Path(path)
    files = [root] if root.is_file() and root.suffix == ".pkl" else sorted(root.rglob("*.pkl"))
    if not files:
        raise RetargetingError(f"SONIC preprocess produced no PKL files under {root}")

    required = SOMA_MOTION_FIELDS if artifact_kind == "soma_skeleton" else ROBOT_MOTION_FIELDS
    motion_count = 0
    rel_files: list[str] = []
    for file_path in files:
        data = _load_joblib(file_path)
        if not isinstance(data, dict) or not data:
            raise RetargetingError(f"Invalid motion PKL {file_path}: expected non-empty dict")
        for motion_name, entry in data.items():
            if not isinstance(entry, dict):
                raise RetargetingError(
                    f"Invalid motion PKL {file_path}: {motion_name!r} is not a dict"
                )
            missing = sorted(required - set(entry))
            if missing:
                joined = ", ".join(missing)
                raise RetargetingError(
                    f"Invalid motion PKL {file_path}: {motion_name!r} missing {joined}"
                )
            motion_count += 1
        rel_files.append(str(file_path.relative_to(root if root.is_dir() else root.parent)))

    return motion_count, rel_files


def _validate_common(
    *,
    input_path: str,
    output_path: str,
    source_format: str,
    embodiment: str,
    frame_rate: int,
    source_frame_rate: int,
    max_frames: int,
    num_workers: int,
) -> None:
    if not input_path:
        raise RetargetingError("input_path is required")
    if not output_path:
        raise RetargetingError("output_path is required")
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        supported = ", ".join(SUPPORTED_SOURCE_FORMATS)
        raise RetargetingError(f"--source-format must be one of: {supported}")
    if embodiment not in SUPPORTED_EMBODIMENTS:
        supported = ", ".join(SUPPORTED_EMBODIMENTS)
        raise RetargetingError(
            f"SONIC's bundled converter is G1-only; --embodiment must be one of: {supported}"
        )
    if frame_rate <= 0:
        raise RetargetingError("--frame-rate must be positive")
    if source_frame_rate < 0:
        raise RetargetingError("--source-frame-rate must be non-negative")
    if max_frames < 0:
        raise RetargetingError("--max-frames must be non-negative")
    if num_workers <= 0:
        raise RetargetingError("--num-workers must be positive")


def _normalize_source_format(source_format: str) -> str:
    normalized = source_format.strip().lower().replace("_", "-")
    aliases = {
        "bones-seed": "bones-seed-csv",
        "bones": "bones-seed-csv",
        "csv": "soma-csv",
        "pkl": "deploy-pkl",
        "soma-pkl": "deploy-pkl",
        "g1-pkl": "deploy-pkl",
        "already-retargeted": "motion-lib",
    }
    return aliases.get(normalized, normalized)


def _normalize_embodiment(embodiment: str) -> str:
    normalized = embodiment.strip().lower().replace("_", "-")
    if normalized == "unitree-g1-sonic":
        return "unitree-g1"
    return normalized


def _stage_input(
    input_path: str,
    local_dir: Path,
    storage_client: "StorageClient | None",
) -> Path:
    if input_path.startswith("s3://"):
        from npa.clients.storage import StorageClient

        local_dir.mkdir(parents=True, exist_ok=True)
        client = storage_client or StorageClient.from_environment()
        downloaded = Path(client.download_path(input_path, str(local_dir)))
        if downloaded.is_dir() and not any(downloaded.rglob("*")):
            raise RetargetingError(f"S3 input contains no objects: {input_path}")
        return downloaded

    path = Path(input_path)
    if not path.exists():
        raise RetargetingError(f"input_path does not exist: {input_path}")
    return path


def _local_output_target(
    *,
    output_path: str,
    work_dir: Path,
    source_format: str,
    input_path: Path,
    individual: bool,
) -> Path:
    if output_path.startswith("s3://"):
        if source_format in {"deploy-pkl", "teleop-pkl"} or input_path.is_file():
            return work_dir / "output" / "motion_lib.pkl"
        return work_dir / "output"

    path = Path(output_path)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    if source_format in {"deploy-pkl", "teleop-pkl"} or input_path.is_file():
        return path / "motion_lib.pkl"
    return path


def _publish_output(
    local_output: Path,
    output_path: str,
    storage_client: "StorageClient | None",
) -> str:
    if not output_path.startswith("s3://"):
        return str(local_output)

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    upload_root = local_output if local_output.is_dir() else local_output.parent
    return client.upload_directory(str(upload_root), output_path)


def _copy_motion_lib(source: Path, destination: Path) -> None:
    if source.is_dir():
        if destination.exists() and destination.is_file():
            raise RetargetingError("--output-path must be a directory when copying a motion-lib directory")
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        shutil.copy2(source, destination / source.name)
    elif destination.suffix:
        shutil.copy2(source, destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)


def _resolve_upstream_root(explicit_home: str, work_dir: Path) -> Path:
    candidates: list[Path] = []
    for value in (
        explicit_home,
        os.environ.get("SONIC_HOME", ""),
        os.environ.get("NPA_RETARGETING_SONIC_HOME", ""),
        "/opt/sonic",
    ):
        if value:
            candidates.append(Path(value))
    cwd = Path.cwd()
    candidates.extend([cwd, cwd.parent])

    for candidate in candidates:
        if (candidate / CONVERTER_SCRIPT).exists() or (candidate / BVH_SOMA_SCRIPT).exists():
            return candidate

    if not _env_truthy("NPA_RETARGETING_AUTO_FETCH", default=True):
        raise RetargetingError(
            "Upstream SONIC preprocessors were not found. Run in an image with "
            "NVlabs/GR00T-WholeBodyControl available, set SONIC_HOME, or enable "
            "NPA_RETARGETING_AUTO_FETCH=1."
        )

    target = work_dir / "upstream-sonic"
    command = [
        "git",
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        UPSTREAM_REPO_URL,
        str(target),
    ]
    _run_checked(command, cwd=work_dir)
    _run_checked(["git", "checkout", _upstream_ref()], cwd=target)
    for script in (CONVERTER_SCRIPT, BVH_SOMA_SCRIPT):
        if not (target / script).exists():
            raise RetargetingError(f"Upstream SONIC script missing after fetch: {script}")
    return target


def _ensure_converter_deps() -> None:
    """Ensure the upstream SONIC motion-lib converter's deps are importable.

    ``convert_soma_csv_to_motion_lib.py`` imports joblib, pandas and scipy. When
    retargeting runs in a lean base image (no dedicated workbench image), install
    them on demand -- the same runtime-provisioning pattern as the upstream repo
    fetch -- so the CPU converter works without pulling the retargeting image.
    """

    missing = []
    for module in ("joblib", "pandas", "scipy"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        _run_checked([_python_executable(), "-m", "pip", "install", "-q", *missing])


def _run_upstream_preprocess(
    *,
    upstream_root: Path,
    source_format: str,
    input_path: Path,
    output_path: Path,
    frame_rate: int,
    source_frame_rate: int,
    individual: bool,
    num_workers: int,
) -> list[str]:
    script_name = BVH_SOMA_SCRIPT if source_format == "bvh" else CONVERTER_SCRIPT
    script = upstream_root / script_name
    if not script.exists():
        raise RetargetingError(f"Upstream SONIC script not found: {script}")

    _ensure_converter_deps()

    command = [
        _python_executable(),
        str(script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--fps",
        str(frame_rate),
    ]
    if source_format == "bvh":
        command.extend(["--num_workers", str(num_workers)])
    else:
        if source_frame_rate:
            command.extend(["--fps_source", str(source_frame_rate)])
        if individual and input_path.is_dir() and source_format not in {"deploy-pkl", "teleop-pkl"}:
            command.extend(["--individual", "--num_workers", str(num_workers)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
    _run_checked(command, cwd=upstream_root)
    return command


def _planned_command(
    *,
    source_format: str,
    input_path: str,
    output_path: str,
    frame_rate: int,
    source_frame_rate: int,
    individual: bool,
    num_workers: int,
) -> list[str]:
    if source_format == "motion-lib":
        return ["copy-motion-lib", input_path, output_path]
    command = [
        _python_executable(),
        BVH_SOMA_SCRIPT if source_format == "bvh" else CONVERTER_SCRIPT,
        "--input",
        input_path,
        "--output",
        output_path,
        "--fps",
        str(frame_rate),
    ]
    if source_format == "bvh":
        command.extend(["--num_workers", str(num_workers)])
    elif source_frame_rate:
        command.extend(["--fps_source", str(source_frame_rate)])
    if individual and source_format not in {"bvh", "deploy-pkl", "teleop-pkl"}:
        command.extend(["--individual", "--num_workers", str(num_workers)])
    return command


def _run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RetargetingError(f"Required command is not available: {command[0]}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"return code {completed.returncode}"
        raise RetargetingError(f"SONIC preprocess failed: {detail[-2000:]}")
    return completed


def _python_executable() -> str:
    for value in (
        os.environ.get("NPA_SONIC_PYTHON", ""),
        os.environ.get("ISAAC_LAB_PYTHON", ""),
        "/isaac-sim/python.sh",
        sys.executable,
        "python3",
    ):
        if not value:
            continue
        if "/" in value:
            if Path(value).exists():
                return value
        elif shutil.which(value):
            return value
    return sys.executable


def _truncate_motion_files(root: Path, *, max_frames: int, artifact_kind: str) -> None:
    files = [root] if root.is_file() and root.suffix == ".pkl" else sorted(root.rglob("*.pkl"))
    if not files:
        return
    required = SOMA_MOTION_FIELDS if artifact_kind == "soma_skeleton" else ROBOT_MOTION_FIELDS
    for file_path in files:
        data = _load_joblib(file_path)
        changed = False
        for entry in data.values():
            if not isinstance(entry, dict) or not required <= set(entry):
                continue
            frame_count = _frame_count(entry)
            if frame_count <= max_frames:
                continue
            for key, value in list(entry.items()):
                if _has_matching_frames(value, frame_count):
                    entry[key] = value[:max_frames]
                    changed = True
        if changed:
            _dump_joblib(data, file_path)


def _frame_count(entry: dict[str, Any]) -> int:
    for key in ("root_trans_offset", "soma_joints", "dof", "pose_aa"):
        value = entry.get(key)
        if hasattr(value, "shape") and value.shape:
            return int(value.shape[0])
        try:
            return len(value)
        except TypeError:
            continue
    return 0


def _has_matching_frames(value: Any, frame_count: int) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if hasattr(value, "shape") and value.shape:
        return int(value.shape[0]) == frame_count
    try:
        return len(value) == frame_count
    except TypeError:
        return False


def _load_joblib(path: Path) -> Any:
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RetargetingError("Validating SONIC motion libs requires joblib") from exc
    return joblib.load(path)


def _dump_joblib(data: Any, path: Path) -> None:
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RetargetingError("Writing SONIC motion libs requires joblib") from exc
    joblib.dump(data, path, compress=True)


def _env_truthy(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _upstream_ref() -> str:
    return os.environ.get("NPA_RETARGETING_UPSTREAM_REF", UPSTREAM_REPO_REF)


def _as_payload(result: RetargetingResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "backend": result.backend,
        "artifact_kind": result.artifact_kind,
        "input_path": result.input_path,
        "output_path": result.output_path,
        "artifact_uri": result.artifact_uri,
        "metadata_uri": result.metadata_uri,
        "metadata_written_uri": result.metadata_written_uri,
        "source_format": result.source_format,
        "embodiment": result.embodiment,
        "retarget_map": result.retarget_map,
        "frame_rate": result.frame_rate,
        "source_frame_rate": result.source_frame_rate,
        "max_frames": result.max_frames,
        "individual": result.individual,
        "num_workers": result.num_workers,
        "motion_count": result.motion_count,
        "output_files": result.output_files,
        "command": result.command,
        "upstream_repo": result.upstream_repo,
        "upstream_ref": result.upstream_ref,
        "generated_at": result.generated_at,
    }
