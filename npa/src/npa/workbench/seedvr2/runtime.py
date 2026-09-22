"""Run pinned SeedVR2 inference with verified S3 input and output artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable
from uuid import uuid4

from npa.clients.storage import StorageClient
from npa.workbench.storage_scope import authorize_uri

from .conditioning import (
    configuration_identity,
    prepare_configuration,
    verify_configuration,
)
from .hardware import require_b200_build_inventory, validate_gpu
from .schemas import (
    MODEL_FILES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PROBE_SCHEMA,
    RESULT_SCHEMA,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    RestoreRequest,
)

SOURCE_REVISION_PATH = Path("/opt/npa-source-revision")
MAX_SOURCE_PIXELS = 1920 * 1080
SOURCE_ROOT = Path("/opt/seedvr2")
RUNTIME_TEMP_ROOT = Path("/workspace/tmp")
UPSTREAM_FAILURE_TAIL_BYTES = 16 * 1024
_UPSTREAM_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)([\"']?\b(?:AUTHORIZATION|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|"
    r"AWS_SESSION_TOKEN|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|NGC_API_KEY|"
    r"OPENAI_API_KEY|TOKEN_FACTORY_API_KEY)\b[\"']?\s*[:=]\s*)"
    r"[^\r\n]*"
)
_UPSTREAM_TOKEN_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"hf_[A-Za-z0-9_=-]{8,}"),
    re.compile(r"nvapi-[A-Za-z0-9_=-]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_=-]{20,}"),
    re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
)


class SeedVR2Error(RuntimeError):
    """Raised when a SeedVR2 request, execution, or artifact is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _require_canonical_s3_uri(
    value: str,
    target: Any,
    *,
    label: str,
    allow_trailing_slash: bool = False,
) -> None:
    canonical = f"s3://{target.bucket}/{target.key}"
    accepted = {canonical}
    if allow_trailing_slash:
        accepted.add(canonical + "/")
    if value not in accepted:
        raise SeedVR2Error(f"{label} must use one canonical, unescaped S3 key")


def _validate_request(request: RestoreRequest) -> None:
    source = authorize_uri(request.input_path, operation="read SeedVR2 input")
    destination = authorize_uri(request.output_path, operation="write SeedVR2 output")
    if source.kind != "s3" or not source.key.lower().endswith(".mp4"):
        raise SeedVR2Error("input_path must be one exact s3:// MP4 object")
    if destination.kind != "s3" or not destination.key:
        raise SeedVR2Error("output_path must be an s3:// bucket prefix")
    _require_canonical_s3_uri(request.input_path, source, label="input_path")
    _require_canonical_s3_uri(
        request.output_path,
        destination,
        label="output_path",
        allow_trailing_slash=True,
    )
    if not request.dry_run and not request.probe_path:
        raise SeedVR2Error("non-dry SeedVR2 execution requires probe_path")
    if not request.probe_path:
        return
    probe = authorize_uri(request.probe_path, operation="read SeedVR2 probe")
    if probe.kind != "s3" or not probe.key.lower().endswith(".json"):
        raise SeedVR2Error("probe_path must be one exact s3:// JSON object")
    _require_canonical_s3_uri(request.probe_path, probe, label="probe_path")


def _verify_probe_contract(
    request: RestoreRequest,
    storage: Any,
    directory: Path,
    *,
    input_hash: str,
    source_probe: dict[str, Any],
) -> dict[str, str] | None:
    if not request.probe_path:
        return None
    probe_path = directory / "probe.json"
    storage.download_file(request.probe_path, str(probe_path))
    try:
        document = json.loads(probe_path.read_text())
        probe_input = document["input"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise SeedVR2Error(
            "probe_path does not contain a valid probe document"
        ) from exc
    if not isinstance(document, dict) or not isinstance(probe_input, dict):
        raise SeedVR2Error("probe_path does not contain a valid probe document")
    valid = (
        document.get("schema") == PROBE_SCHEMA
        and document.get("status") == "ok"
        and document.get("run_id") == request.run_id
        and probe_input.get("uri") == request.input_path
        and probe_input.get("sha256") == input_hash
        and probe_input.get("media") == source_probe
    )
    if not valid:
        raise SeedVR2Error("probe document does not bind the exact restore input")
    return {"uri": request.probe_path, "sha256": _sha256(probe_path)}


def _validate_source_geometry(
    request: RestoreRequest, source_probe: dict[str, Any]
) -> None:
    if source_probe["width"] * source_probe["height"] > MAX_SOURCE_PIXELS:
        raise SeedVR2Error("source video area must not exceed 1920x1080")
    source_ratio = Fraction(source_probe["width"], source_probe["height"])
    output_ratio = Fraction(request.output_width, request.output_height)
    if source_ratio != output_ratio:
        raise SeedVR2Error("requested output aspect ratio must match the source video")


def _stable_run_name(run_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-") or "run"
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    return f"{readable[:48]}-{digest}"


def _safe_run_name(run_id: str) -> str:
    return f"{_stable_run_name(run_id)}-{uuid4().hex[:8]}"


def _create_work_directory(run_id: str) -> Path:
    root = Path(os.environ.get("NPA_SEEDVR2_WORK_DIR", "/workspace/seedvr2-runs"))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    directory = root / _safe_run_name(run_id)
    directory.mkdir(mode=0o700)
    return directory


def _probe_video_header(path: Path) -> tuple[int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "mov",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height:format=format_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=_media_environment(),
        )
    except OSError as exc:
        raise SeedVR2Error("ffprobe is unavailable") from exc
    if completed.returncode != 0:
        raise SeedVR2Error("ffprobe could not decode the video artifact")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        formats = str(payload["format"]["format_name"]).split(",")
        width, height = int(stream["width"]), int(stream["height"])
        if "mp4" not in formats or stream["codec_name"] != "h264":
            raise ValueError("not an H.264 MP4")
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise SeedVR2Error("video is not one supported H.264 MP4 stream") from exc
    if width < 1 or height < 1:
        raise SeedVR2Error("video contains no positive-size video stream")
    return width, height


def _probe_video(path: Path) -> dict[str, Any]:
    width, height = _probe_video_header(path)
    if width * height > MAX_SOURCE_PIXELS:
        raise SeedVR2Error("video area must not exceed 1920x1080")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "mov",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames:"
            "format=format_name,duration,size"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=_media_environment(),
        )
    except OSError as exc:
        raise SeedVR2Error("ffprobe is unavailable") from exc
    if completed.returncode != 0:
        raise SeedVR2Error("ffprobe could not decode the video artifact")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        formats = str(payload["format"]["format_name"]).split(",")
        if "mp4" not in formats or stream["codec_name"] != "h264":
            raise ValueError("not an H.264 MP4")
        frames = int(stream["nb_read_frames"])
        fps = str(stream["avg_frame_rate"])
        result = {
            "codec": str(stream["codec_name"]),
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "duration_seconds": float(payload["format"]["duration"]),
            "bytes": int(payload["format"]["size"]),
        }
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise SeedVR2Error("ffprobe returned incomplete video metadata") from exc
    if frames < 1 or result["width"] < 1 or result["height"] < 1:
        raise SeedVR2Error("video contains no decodable positive-size frames")
    return result


def _decoded_frame_hashes(path: Path) -> list[str]:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "mov",
            "-i",
            str(path),
            "-f",
            "framemd5",
            "-",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=_media_environment(),
    )
    if completed.returncode != 0:
        raise SeedVR2Error("ffmpeg could not decode every output frame")
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in completed.stdout.splitlines()
        if line and not line.startswith("#")
    ]


def _seedvr_python() -> str:
    return os.environ.get("SEEDVR2_PYTHON", "/opt/seedvr2-venv/bin/python")


def _media_environment() -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "TMPDIR": str(RUNTIME_TEMP_ROOT),
    }


def _model_fetch_environment() -> dict[str, str]:
    environment = {
        "HOME": "/workspace",
        "HF_HOME": os.environ.get("HF_HOME", "/workspace/.cache/huggingface"),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(RUNTIME_TEMP_ROOT),
    }
    for name in ("HF_TOKEN", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _resolve_model_files() -> dict[str, Path]:
    allow_patterns = sorted(MODEL_FILES)
    code = (
        "from huggingface_hub import snapshot_download; "
        "import os; "
        f"path=snapshot_download(repo_id={MODEL_REPOSITORY!r}, "
        f"revision={MODEL_REVISION!r}, allow_patterns={allow_patterns!r}, "
        "token=True if os.environ.get('HF_TOKEN') else None); "
        "print('NPA_SEEDVR2_SNAPSHOT=' + path)"
    )
    completed = subprocess.run(
        [_seedvr_python(), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=_model_fetch_environment(),
    )
    if completed.returncode != 0:
        raise SeedVR2Error("the pinned public SeedVR2 model download failed")
    marker = "NPA_SEEDVR2_SNAPSHOT="
    snapshot_value = next(
        (
            line.removeprefix(marker)
            for line in reversed((completed.stdout or "").splitlines())
            if line.startswith(marker)
        ),
        "",
    )
    snapshot = Path(snapshot_value)
    if not snapshot.is_dir():
        raise SeedVR2Error("Hugging Face did not return a model snapshot")
    return _verify_model_files(snapshot)


def _verify_model_files(snapshot: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, (expected_size, expected_hash) in MODEL_FILES.items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != expected_size:
            raise SeedVR2Error(f"pinned model payload size mismatch: {name}")
        if _sha256(path) != expected_hash:
            raise SeedVR2Error(f"pinned model payload hash mismatch: {name}")
        resolved[name] = path
    return resolved


def _prepare_upstream_workspace(
    directory: Path, model_files: dict[str, Path], conditioning_mode: str = "sample"
) -> Path:
    required_files = (
        SOURCE_ROOT / "projects" / "inference_seedvr2_3b.py",
        SOURCE_ROOT / "configs_3b" / "main.yaml",
        SOURCE_ROOT / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml",
    )
    if not all(path.is_file() for path in required_files):
        raise SeedVR2Error("the pinned SeedVR2 source tree is unavailable")
    workspace = directory / "upstream"
    workspace.mkdir()
    prepare_configuration(SOURCE_ROOT, workspace, conditioning_mode)
    for name in ("models", "projects"):
        (workspace / name).symlink_to(SOURCE_ROOT / name, target_is_directory=True)
    checkpoints = workspace / "ckpts"
    checkpoints.mkdir()
    for name, path in model_files.items():
        (checkpoints / name).symlink_to(path)
    for name in ("pos_emb.pt", "neg_emb.pt"):
        (workspace / name).symlink_to(model_files[name])
    return workspace


def build_restore_argv(request: RestoreRequest, workspace: Path) -> list[str]:
    """Build the genuine upstream one-GPU SeedVR2 inference command."""

    return [
        str(Path(_seedvr_python()).with_name("torchrun")),
        "--standalone",
        "--nproc-per-node=1",
        str(SOURCE_ROOT / "projects" / "inference_seedvr2_3b.py"),
        "--video_path",
        str(workspace.parent / "input"),
        "--output_dir",
        str(workspace.parent / "generated"),
        "--seed",
        str(request.seed),
        "--res_h",
        str(request.output_height),
        "--res_w",
        str(request.output_width),
        "--sp_size",
        "1",
    ]


def _inference_environment() -> dict[str, str]:
    environment = {
        "HOME": "/workspace",
        "HF_HOME": os.environ.get("HF_HOME", "/workspace/.cache/huggingface"),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": str(SOURCE_ROOT),
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(RUNTIME_TEMP_ROOT),
    }
    for name in (
        "CUDA_HOME",
        "CUDA_MODULE_LOADING",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _upstream_failure_tail(log_path: Path) -> str:
    with log_path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        snapshot_size = stream.tell()
        start = max(0, snapshot_size - UPSTREAM_FAILURE_TAIL_BYTES)
        preceding = b""
        if start:
            stream.seek(start - 1)
            preceding = stream.read(1)
        stream.seek(start)
        payload = stream.read(snapshot_size - start)
    if start and preceding not in {b"\n", b"\r"}:
        payload = b"".join(payload.splitlines(keepends=True)[1:])
    text = payload.decode("utf-8", errors="replace")
    for pattern in _UPSTREAM_TOKEN_PATTERNS:
        text = pattern.sub("<redacted>", text)
    text = _UPSTREAM_SENSITIVE_ASSIGNMENT.sub(r"\1<redacted>", text)
    if start:
        text = (
            "[truncated to complete lines within final "
            f"{UPSTREAM_FAILURE_TAIL_BYTES} bytes]\n{text}"
        )
    return "".join(
        character for character in text if character in "\n\r\t" or ord(character) >= 32
    ).strip()


def _execute_upstream(
    argv: list[str],
    workspace: Path,
    log_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> None:
    with log_path.open("wb") as log:
        completed = runner(
            argv,
            cwd=workspace,
            env=_inference_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = _upstream_failure_tail(log_path)
        detail = f"; redacted log tail:\n{tail}" if tail else ""
        raise SeedVR2Error(
            f"official SeedVR2 inference exited {completed.returncode}; "
            f"retained log: {log_path}{detail}"
        )


def _validate_output(
    output: Path,
    source_probe: dict[str, Any],
    request: RestoreRequest,
) -> tuple[dict[str, Any], int]:
    probe = _probe_video(output)
    if (probe["height"], probe["width"]) != (
        request.output_height,
        request.output_width,
    ):
        raise SeedVR2Error("SeedVR2 output dimensions differ from the request")
    if probe["frames"] != source_probe["frames"]:
        raise SeedVR2Error("SeedVR2 output frame count differs from the input")
    if Fraction(probe["fps"]) != Fraction(source_probe["fps"]):
        raise SeedVR2Error("SeedVR2 output frame rate differs from the input")
    period = 1.0 / float(Fraction(source_probe["fps"]))
    if abs(probe["duration_seconds"] - source_probe["duration_seconds"]) > period:
        raise SeedVR2Error("SeedVR2 output duration differs by more than one frame")
    frame_hashes = _decoded_frame_hashes(output)
    if len(frame_hashes) != probe["frames"]:
        raise SeedVR2Error("decoded output frame count is inconsistent")
    if len(frame_hashes) > 1 and len(set(frame_hashes)) == 1:
        raise SeedVR2Error("SeedVR2 output repeats one identical frame")
    return probe, len(set(frame_hashes))


def _gpu_inventory() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap,mig.mode.current",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=_media_environment(),
        )
    except OSError:
        return {"status": "unavailable"}
    if completed.returncode != 0:
        return {"status": "unavailable"}
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        return {"status": "unavailable"}
    fields = [item.strip() for item in lines[0].split(",")]
    if len(fields) != 5:
        return {"status": "unavailable"}
    return {
        "status": "available",
        "name": fields[0],
        "memory_mib": fields[1],
        "driver_version": fields[2],
        "compute_capability": fields[3],
        "mig_mode": fields[4],
        "count": "1",
    }


def _runtime_identity(expected_gpu: str = "H100") -> dict[str, Any]:
    image = os.environ.get("NPA_TASK_IMAGE", "")
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
        raise SeedVR2Error("NPA_TASK_IMAGE must bind an immutable image digest")
    try:
        source_revision = SOURCE_REVISION_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SeedVR2Error("image lacks its baked NPA source revision") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise SeedVR2Error("image has an invalid baked NPA source revision")
    gpu = _gpu_inventory()
    try:
        validate_gpu(gpu, expected_gpu)
        if expected_gpu == "B200":
            require_b200_build_inventory()
    except ValueError as exc:
        raise SeedVR2Error(str(exc)) from exc
    return {
        "image": image,
        "image_digest": image.rsplit("@", 1)[1],
        "npa_source_revision": source_revision,
        "gpu": gpu,
        "sequence_parallel_size": 1,
        "color_fix": False,
    }


def _artifact_uri(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def _ensure_artifacts_absent(storage: Any, uris: list[str]) -> None:
    for uri in uris:
        if storage.read_bytes_with_etag(uri) is not None:
            raise SeedVR2Error(f"output artifact already exists: {uri}")


def _publish_verified(storage: Any, source: Path, uri: str, readback_root: Path) -> str:
    expected = _sha256(source)
    storage.put_bytes_conditional(source.read_bytes(), uri, if_none_match=True)
    readback_root.mkdir(parents=True, exist_ok=True)
    readback = readback_root / source.name
    storage.download_file(uri, str(readback))
    if _sha256(readback) != expected:
        raise SeedVR2Error(f"uploaded artifact readback hash mismatch: {uri}")
    return expected


def _planned_result(request: RestoreRequest) -> dict[str, Any]:
    workspace = (
        Path("/workspace/seedvr2-runs") / f"{_stable_run_name(request.run_id)}-dry-run"
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": "dry_run",
        "request": request.model_dump(),
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "argv": build_restore_argv(request, workspace / "upstream"),
        "artifacts": {},
    }


def _result_document(
    request: RestoreRequest,
    argv: list[str],
    *,
    started_at: str,
    media_identity: dict[str, Any],
    output_hash: str,
    upstream_log_hash: str,
    artifacts: dict[str, str],
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "status": "ok",
        "run_id": request.run_id,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        **media_identity,
        "request": request.model_dump(exclude={"dry_run"}),
        "runtime": runtime_identity,
        "configuration": configuration_identity(SOURCE_ROOT, request.conditioning_mode),
        "argv": argv,
        "artifacts": artifacts,
        "artifact_hashes": {
            "restored_video": output_hash,
            "upstream_log": upstream_log_hash,
        },
        "limitations": [
            "Generated detail is a review aid, not observed sensor truth.",
            "Heavy degradation and large motion can fail or create unpleasant detail.",
            "Light degradation and small inputs can be oversharpened.",
        ],
    }


def _model_and_media(
    request,
    source_probe,
    output_probe,
    input_hash,
    output_hash,
    unique_frames,
    probe_identity,
):
    return {
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "files": {
                name: {"bytes": size, "sha256": digest}
                for name, (size, digest) in MODEL_FILES.items()
            },
            "weights_baked": False,
        },
        "input": {
            "uri": request.input_path,
            "sha256": input_hash,
            "media": source_probe,
            "probe": probe_identity,
        },
        "output": {
            "sha256": output_hash,
            "media": output_probe,
            "unique_decoded_frames": unique_frames,
            "derived_sensor_truth": False,
        },
    }


def restore(
    request: RestoreRequest,
    *,
    storage_factory: Callable[[], Any] = StorageClient.from_environment,
    inference_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    model_resolver: Callable[[], dict[str, Path]] = _resolve_model_files,
    runtime_identity_resolver: Callable[[str], dict[str, Any]] = _runtime_identity,
) -> dict[str, Any]:
    """Run official SeedVR2-3B and publish readback-verified artifacts.

    Args:
        request: Immutable S3 input, output, and inference controls.
        storage_factory: Build the request-scoped object-storage client.
        inference_runner: Execute the upstream ``torchrun`` command.
        model_resolver: Fetch and verify the exact public model payloads.
        runtime_identity_resolver: Verify the immutable image and requested GPU identity.
    Returns:
        Complete run provenance and artifact URIs.
    Raises:
        SeedVR2Error: Validation, inference, decoding, or readback fails.
        OSError: A private run-evidence directory cannot be created or written.
    """

    _validate_request(request)
    if request.dry_run:
        return _planned_result(request)
    started_at = _utc_now()
    directory = _create_work_directory(request.run_id)
    with _retain_failure(directory):
        return _restore_in_directory(
            request,
            directory,
            started_at,
            storage_factory(),
            inference_runner,
            model_resolver,
            runtime_identity_resolver,
        )


@contextmanager
def _retain_failure(directory):
    try:
        yield
    except Exception as exc:
        failure = {"status": "failed", "at": _utc_now(), "error": type(exc).__name__}
        (directory / "failure.json").write_bytes(_canonical_json(failure))
        if isinstance(exc, SeedVR2Error):
            raise
        raise SeedVR2Error(
            f"SeedVR2 failed; private evidence retained at {directory}"
        ) from exc


def _restore_in_directory(
    request: RestoreRequest,
    directory: Path,
    started_at: str,
    storage: Any,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
    model_resolver: Callable[[], dict[str, Path]],
    runtime_identity_resolver: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    source, source_probe, input_hash, probe_identity = _prepare_restore_input(
        request, directory, storage
    )
    runtime_identity = runtime_identity_resolver(request.expected_gpu)
    validate_gpu(runtime_identity.get("gpu", {}), request.expected_gpu)
    workspace = _prepare_upstream_workspace(
        directory, model_resolver(), request.conditioning_mode
    )
    argv = build_restore_argv(request, workspace)
    log_path = directory / "upstream.log"
    verify_configuration(SOURCE_ROOT, workspace, request.conditioning_mode)
    _execute_upstream(argv, workspace, log_path, runner)
    verify_configuration(SOURCE_ROOT, workspace, request.conditioning_mode)
    output = directory / "generated" / source.name
    output_probe, unique_frames = _validate_output(output, source_probe, request)
    return _publish_result(
        request,
        directory,
        started_at,
        storage,
        argv,
        source_probe,
        output_probe,
        input_hash,
        output,
        unique_frames,
        log_path,
        probe_identity,
        runtime_identity,
    )


def _prepare_restore_input(request, directory, storage):
    input_dir = directory / "input"
    generated_dir = directory / "generated"
    readback_dir = directory / "readback"
    for path in (input_dir, generated_dir, readback_dir):
        path.mkdir()
    source = input_dir / "input.mp4"
    storage.download_file(request.input_path, str(source))
    source_probe = _probe_video(source)
    input_hash = _sha256(source)
    _validate_source_geometry(request, source_probe)
    probe_identity = _verify_probe_contract(
        request,
        storage,
        directory,
        input_hash=input_hash,
        source_probe=source_probe,
    )
    return source, source_probe, input_hash, probe_identity


def _publish_result(
    request: RestoreRequest,
    directory: Path,
    started_at: str,
    storage: Any,
    argv: list[str],
    source_probe: dict[str, Any],
    output_probe: dict[str, Any],
    input_hash: str,
    output: Path,
    unique_frames: int,
    log_path: Path,
    probe_identity: dict[str, str] | None,
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    uris, output_hash, upstream_log_hash = _upload_result_media(
        request, directory, storage, output, log_path
    )
    result = _result_document(
        request,
        argv,
        started_at=started_at,
        media_identity=_model_and_media(
            request,
            source_probe,
            output_probe,
            input_hash,
            output_hash,
            unique_frames,
            probe_identity,
        ),
        output_hash=output_hash,
        upstream_log_hash=upstream_log_hash,
        artifacts=uris,
        runtime_identity=runtime_identity,
    )
    return _publish_result_document(result, directory, storage, uris["result"])


def _upload_result_media(request, directory, storage, output, log_path):
    uris = {
        "restored_video": _artifact_uri(request.output_path, "restored.mp4"),
        "upstream_log": _artifact_uri(request.output_path, "upstream.log"),
        "result": _artifact_uri(request.output_path, "result.json"),
    }
    _ensure_artifacts_absent(storage, list(uris.values()))
    readback = directory / "readback"
    output_hash = _publish_verified(storage, output, uris["restored_video"], readback)
    upstream_log_hash = _publish_verified(
        storage, log_path, uris["upstream_log"], readback
    )
    return uris, output_hash, upstream_log_hash


def _publish_result_document(result, directory, storage, result_uri):
    readback = directory / "readback"
    result_path = directory / "result.json"
    result_path.write_bytes(_canonical_json(result))
    _publish_verified(storage, result_path, result_uri, readback)
    shutil.rmtree(directory)
    return result


__all__ = ["SeedVR2Error", "build_restore_argv", "restore"]
