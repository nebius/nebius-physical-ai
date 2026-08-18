"""LeRobot v3 episode recording and immutable S3 publication for LeIsaac."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)

DATASET_SCHEMA = "npa.leisaac.dataset.v1"
EPISODE_SCHEMA = "npa.leisaac.episode.v1"
DERIVED_SCHEMA = "npa.leisaac.derived-dataset.v1"
LEROBOT_CODEBASE_VERSION = "v3.0"
LEROBOT_TARGET_VERSION = "0.5.1"
VIDEO_KEY = "observation.images.front"
# Cosmos Transfer 2.5's audited input fixture uses 16 fps. Recording at the
# same rate preserves frame count/timestamps through input-conditioned PAIDF
# augmentation when the model returns the source sequence unchanged in length.
FPS = 16
DEFAULT_S3_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
STATE_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "delta_shoulder_pan",
    "delta_gripper",
]
_OUTCOMES = frozenset({"success", "failure"})


class DatasetError(RuntimeError):
    """Raised when an episode cannot be safely recorded or published."""


def resolve_s3_endpoint(
    explicit: str | None = None,
    *,
    config_endpoint: str | None = None,
    environ: dict[str, str] | None = None,
) -> str:
    """Resolve LeIsaac storage as explicit > environment > config > deployment default."""

    env = os.environ if environ is None else environ
    return str(
        explicit
        or env.get("AWS_ENDPOINT_URL_S3")
        or env.get("NEBIUS_S3_ENDPOINT")
        or env.get("AWS_ENDPOINT_URL")
        or env.get("NPA_STORAGE_ENDPOINT")
        or config_endpoint
        or DEFAULT_S3_ENDPOINT
    ).rstrip("/")


def _ffmpeg_executable() -> str:
    """Resolve a real ffmpeg binary without requiring a system package."""

    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise DatasetError(
            "ffmpeg is unavailable; install the npa runtime dependencies"
        ) from exc
    if not executable or not Path(executable).is_file():
        raise DatasetError("the packaged ffmpeg executable is unavailable")
    return executable


def _ffprobe_executable() -> str | None:
    """Return ffprobe when present; callers retain a packaged-ffmpeg fallback."""

    executable = shutil.which("ffprobe")
    if executable:
        return executable
    sibling = Path(_ffmpeg_executable()).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def split_s3_uri(uri: str, *, label: str = "dataset URI") -> tuple[str, str]:
    parsed = urlparse(str(uri or "").strip())
    prefix = parsed.path.strip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not prefix:
        raise DatasetError(f"{label} must be s3://BUCKET/PREFIX")
    if any(part in {"", ".", ".."} for part in prefix.split("/")):
        raise DatasetError(f"{label} has an unsafe prefix")
    return parsed.netloc, prefix.rstrip("/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        secret = os.environ.get(name, "")
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:500]


def _scalar(value: Any) -> Any:
    """Convert a one-environment torch/numpy value into JSON-safe Python data."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _vector(value: Any, size: int, label: str) -> list[float]:
    result = _scalar(value)
    if not isinstance(result, list) or len(result) != size:
        raise DatasetError(f"real simulator {label} must be a {size}-element vector")
    try:
        values = [float(item) for item in result]
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"real simulator {label} contains a non-number") from exc
    if not all(math.isfinite(item) for item in values):
        raise DatasetError(f"real simulator {label} contains a non-finite value")
    return values


def _bool(value: Any) -> bool:
    return bool(_scalar(value))


def _float(value: Any) -> float:
    result = float(_scalar(value))
    if not math.isfinite(result):
        raise DatasetError("real simulator reward contains a non-finite value")
    return result


def extract_step(
    step_result: Any,
    action: Any,
    *,
    sim_step: int,
) -> dict[str, Any]:
    """Extract only values returned by the real one-environment Isaac step."""

    if not isinstance(step_result, tuple) or len(step_result) != 5:
        raise DatasetError(
            "Isaac environment step must return obs/reward/terminated/truncated/info"
        )
    observations, reward, terminated, truncated, _info = step_result
    if not isinstance(observations, dict):
        raise DatasetError("Isaac environment did not return an observation mapping")
    policy = observations.get("policy", observations)
    if not isinstance(policy, dict) or "joint_pos" not in policy:
        raise DatasetError("Isaac policy observation is missing real joint_pos")
    mono_ns = time.monotonic_ns()
    wall_ns = time.time_ns()
    terminated_value = _bool(terminated)
    truncated_value = _bool(truncated)
    return {
        "observation.state": _vector(policy["joint_pos"], 6, "joint_pos"),
        "action": _vector(action, 8, "applied action"),
        "reward": _float(reward),
        "terminated": terminated_value,
        "truncated": truncated_value,
        "done": terminated_value or truncated_value,
        "sim_step": int(sim_step),
        "monotonic_ns": mono_ns,
        "wall_clock_ns": wall_ns,
    }


class EpisodeRecorder:
    """Thread-safe local episode state consumed by the patched Isaac loop."""

    def __init__(
        self,
        *,
        root: Path,
        output_uri: str,
        task: str,
        environment_id: str,
        environment_index: int,
        seed: int,
        run_id: str,
        source_commit: str,
        source_version: str = "",
        isaac_sim_version: str = "",
        isaac_lab_version: str = "",
        image: str = "",
        registry_fingerprint: str = "",
        camera_ids: tuple[str, ...] = ("primary",),
        provenance: dict[str, Any] | None = None,
        publisher: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
        status_loader: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        split_s3_uri(output_uri, label="output path")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_uri = output_uri.rstrip("/")
        self.task = task
        self.environment_id = environment_id
        self.environment_index = int(environment_index)
        self.seed = int(seed)
        self.run_id = run_id
        self.source_commit = source_commit
        self.source_version = source_version
        self.isaac_sim_version = isaac_sim_version
        self.isaac_lab_version = isaac_lab_version
        self.image = image
        self.registry_fingerprint = registry_fingerprint
        self.camera_ids = self._validated_camera_ids(camera_ids)
        self.provenance = dict(provenance) if isinstance(provenance, dict) else {}
        self.publisher: Callable[[Path, dict[str, Any]], dict[str, Any]]
        self._status_loader: Callable[[], dict[str, Any]] | None
        if publisher is None:
            store = S3DatasetStore(self.output_uri)
            self.publisher = store.publish_episode
            self._status_loader = store.resume_status
        else:
            self.publisher = publisher
            self._status_loader = status_loader
        self.control_path = self.root / "control.jsonl"
        self.status_path = self.root / "status.json"
        self.pending_command_path = self.root / "pending-command.json"
        self._control_offset = 0
        self._lock = threading.RLock()
        self._state = "idle"
        self._outcome = ""
        self._episode_uuid = ""
        self._episode_dir: Path | None = None
        self._latest_step: dict[str, Any] | None = None
        self._last_recorded_step = -1
        self._frames = 0
        self._completed = 0
        self._active_episode: str | None = None
        self._last_episode_index: int | None = None
        self._last_outcome = ""
        self._last_upload_status = "never"
        self._last_error = ""
        self._dataset_version_uri = ""
        self._last_episode_commit_uri = ""
        self._last_command_id = ""
        self._last_command = ""
        self._command_revision = 0
        self._recovery_error = ""
        self._processed_commands: dict[str, str] = {}
        self._pending_camera_groups: dict[str, dict[str, Any]] = {}
        # Publication performs MP4 assembly and immutable object-store writes.
        # A single worker keeps that work off Isaac's render/control thread while
        # preserving strict episode order and bounded resource use.
        self._publication_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="leisaac-episode-publish",
        )
        self._finalize_future: Future[dict[str, Any]] | None = None
        self._finalize_reset: Callable[[], Any] | None = None
        try:
            prior_status = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior_status = {}
        if not isinstance(prior_status, dict):
            prior_status = {}
        prior_commands = prior_status.get("processed_commands", {})
        if isinstance(prior_commands, dict):
            self._processed_commands = {
                str(request_id): str(command)
                for request_id, command in prior_commands.items()
                if request_id and command
            }
        self._last_command_id = str(prior_status.get("last_command_id") or "")
        self._last_command = str(prior_status.get("last_command") or "")
        self._command_revision = int(prior_status.get("command_revision") or 0)
        if self._last_command_id and self._last_command:
            self._processed_commands.setdefault(
                self._last_command_id, self._last_command
            )
        self.control_path.touch(exist_ok=True)
        self._recover_dataset_status()
        self._write_status()

    @staticmethod
    def _validated_camera_ids(camera_ids: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(item or "").strip() for item in camera_ids)
        if (
            not normalized
            or len(normalized) > 4
            or len(set(normalized)) != len(normalized)
            or any(
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", item)
                for item in normalized
            )
        ):
            raise DatasetError("camera IDs must be one to four unique safe names")
        return normalized

    def configure_capture_schema(
        self,
        camera_ids: tuple[str, ...],
        *,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Configure the next episode without mutating an active schema."""

        normalized = self._validated_camera_ids(camera_ids)
        with self._lock:
            if self._state != "idle":
                if normalized != self.camera_ids:
                    raise DatasetError(
                        "recording camera changes apply only at episode boundaries"
                    )
                return
            self.camera_ids = normalized
            if provenance is not None:
                self.provenance = dict(provenance)
            self._write_status()
            self._write_status()

    def _recover_dataset_status(self) -> None:
        if self._status_loader is None:
            return
        try:
            recovered = self._status_loader()
            self._completed = int(recovered["completed_episode_count"])
            self._last_episode_index = recovered.get("last_episode_index")
            self._last_outcome = str(recovered.get("last_outcome") or "")
            self._last_upload_status = str(
                recovered.get("last_upload_status") or "never"
            )
            self._dataset_version_uri = str(recovered.get("dataset_version_uri") or "")
            self._last_episode_commit_uri = str(
                recovered.get("last_episode_commit_uri") or ""
            )
            self._recovery_error = ""
            self._last_error = ""
        except Exception as exc:
            self._recovery_error = _safe_error(exc)
            self._last_error = (
                "Dataset state recovery failed; retry Start episode after storage "
                f"recovers: {self._recovery_error}"
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "dataset_uri": self.output_uri,
                "dataset_version_uri": self._dataset_version_uri,
                "last_episode_commit_uri": self._last_episode_commit_uri,
                "task": self.task,
                "environment_id": self.environment_id,
                "environment_index": self.environment_index,
                "seed": self.seed,
                "active_episode": self._active_episode,
                "last_episode_index": self._last_episode_index,
                "frame_count": self._frames,
                "completed_episode_count": self._completed,
                "pending_outcome": self._outcome,
                "last_outcome": self._last_outcome,
                "last_upload_status": self._last_upload_status,
                "last_error": self._last_error,
                "pending_command_id": self._pending_command_id(),
                "last_command_id": self._last_command_id,
                "last_command": self._last_command,
                "command_revision": self._command_revision,
                "cameras": list(self.camera_ids),
                "display_view_mode": str(
                    self.provenance.get("display_view_mode") or ""
                ),
                "recording_camera_mode": str(
                    self.provenance.get("recording_camera_mode") or ""
                ),
                # The session server consumes this durable ledger directly. The
                # public agent status allowlist deliberately omits it.
                "processed_commands": dict(self._processed_commands),
            }

    def _pending_command_id(self) -> str:
        try:
            payload = json.loads(self.pending_command_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return str(payload.get("request_id") or "") if isinstance(payload, dict) else ""

    def _clear_pending_command(self, request_id: str) -> None:
        try:
            payload = json.loads(self.pending_command_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(payload, dict) and payload.get("request_id") == request_id:
            self.pending_command_path.unlink(missing_ok=True)

    def _write_status(self) -> None:
        _atomic_json(self.status_path, self.status())

    def start(self) -> None:
        with self._lock:
            if self._recovery_error:
                self._recover_dataset_status()
            if self._recovery_error:
                raise DatasetError(self._last_error)
            if self._state != "idle":
                raise DatasetError("an episode is already active")
            self._episode_uuid = uuid.uuid4().hex
            self._episode_dir = self.root / "active" / self._episode_uuid
            (self._episode_dir / "frames").mkdir(parents=True, exist_ok=False)
            for camera_id in self.camera_ids[1:]:
                (self._episode_dir / f"frames-{camera_id}").mkdir(
                    parents=True, exist_ok=False
                )
            self._state = "recording"
            self._active_episode = self._episode_uuid
            self._outcome = ""
            self._frames = 0
            self._latest_step = None
            self._last_recorded_step = -1
            self._pending_camera_groups.clear()
            self._last_error = ""
            self._last_upload_status = "recording"
            self._write_status()

    def mark(self, outcome: str) -> None:
        with self._lock:
            if self._state not in {"recording", "outcome-pending"}:
                raise DatasetError("no active episode to mark")
            if outcome not in _OUTCOMES:
                raise DatasetError("episode outcome must be success or failure")
            self._outcome = outcome
            self._state = "outcome-pending"
            self._last_error = ""
            self._write_status()

    def observe(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._state == "recording":
                self._latest_step = dict(record)

    def frame(
        self,
        jpeg: bytes,
        *,
        camera_id: str = "primary",
        capture_group: str = "",
    ) -> None:
        with self._lock:
            if self._state != "recording" or self._latest_step is None:
                return
            if not jpeg.startswith(b"\xff\xd8") or len(jpeg) < 1024:
                raise DatasetError("captured RTX frame is not a non-empty JPEG")
            selected_camera = str(camera_id or "")
            if selected_camera not in self.camera_ids:
                raise DatasetError("captured frame camera is not configured")
            group_id = str(capture_group or self._latest_step["sim_step"])
            if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", group_id):
                raise DatasetError("captured frame group ID is invalid")
            group = self._pending_camera_groups.setdefault(
                group_id,
                {"step": dict(self._latest_step), "frames": {}},
            )
            frames = group["frames"]
            if selected_camera in frames:
                return
            frames[selected_camera] = bytes(jpeg)
            if any(camera not in frames for camera in self.camera_ids):
                # At most two render cycles may be pending.  A camera callback
                # that never arrives must not create an unbounded recording cache.
                while len(self._pending_camera_groups) > 2:
                    self._pending_camera_groups.pop(
                        next(iter(self._pending_camera_groups))
                    )
                return
            step = dict(group["step"])
            sim_step = int(step["sim_step"])
            self._pending_camera_groups.pop(group_id, None)
            if sim_step <= self._last_recorded_step:
                return
            assert self._episode_dir is not None
            index = self._frames
            frame_name = f"frame-{index:06d}.jpg"
            camera_checksums: dict[str, str] = {}
            for camera_index, configured_camera in enumerate(self.camera_ids):
                content = frames[configured_camera]
                directory = (
                    "frames" if camera_index == 0 else f"frames-{configured_camera}"
                )
                frame_path = self._episode_dir / directory / frame_name
                temporary = frame_path.with_suffix(".jpg.tmp")
                temporary.write_bytes(content)
                temporary.replace(frame_path)
                camera_checksums[configured_camera] = hashlib.sha256(
                    content
                ).hexdigest()
            step.update(
                {
                    "source_frame_index": index,
                    "timestamp": float(index / FPS),
                    "frame_file": frame_name,
                    "frame_sha256": camera_checksums[self.camera_ids[0]],
                    "camera_frame_sha256": camera_checksums,
                    "success": False,
                    "reset_reason": "",
                    "task": self.task,
                    "environment_id": self.environment_id,
                    "environment_index": self.environment_index,
                    "seed": self.seed,
                }
            )
            with (self._episode_dir / "records.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(step, sort_keys=True, separators=(",", ":")) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            self._frames += 1
            self._last_recorded_step = sim_step
            self._write_status()

    def finalize(self) -> dict[str, Any]:
        with self._lock:
            if (
                self._state not in {"outcome-pending", "upload-failed"}
                or self._outcome not in _OUTCOMES
            ):
                raise DatasetError(
                    "mark success or failure before finalizing the episode"
                )
            if self._frames < 2 or self._episode_dir is None:
                raise DatasetError("an episode needs at least two synchronized frames")
            metadata_path = self._episode_dir / "episode.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise DatasetError(
                        "persisted episode metadata is unreadable"
                    ) from exc
                expected = {
                    "episode_uuid": self._episode_uuid,
                    "run_id": self.run_id,
                    "outcome": self._outcome,
                    "frame_count": self._frames,
                }
                if not isinstance(metadata, dict) or any(
                    metadata.get(key) != value for key, value in expected.items()
                ):
                    raise DatasetError(
                        "persisted episode metadata does not match the active episode"
                    )
            else:
                records_path = self._episode_dir / "records.jsonl"
                records = _load_records(records_path)
                records[-1]["success"] = self._outcome == "success"
                records[-1]["reset_reason"] = self._outcome
                records_path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in records
                    ),
                    encoding="utf-8",
                )
                metadata = {
                    "schema": EPISODE_SCHEMA,
                    "episode_uuid": self._episode_uuid,
                    "run_id": self.run_id,
                    "task": self.task,
                    "environment_id": self.environment_id,
                    "environment_index": self.environment_index,
                    "seed": self.seed,
                    "outcome": self._outcome,
                    "frame_count": self._frames,
                    "fps": FPS,
                    "source_commit": self.source_commit,
                    "cameras": list(self.camera_ids),
                    "provenance": {
                        "leisaac_version": self.source_version,
                        "leisaac_commit": self.source_commit,
                        "isaac_sim_version": self.isaac_sim_version,
                        "isaac_lab_version": self.isaac_lab_version,
                        "image": self.image,
                        "task_registry_fingerprint": self.registry_fingerprint,
                        **self.provenance,
                    },
                    "recorded_at": utc_now(),
                    "visual_source": "real RTX viewport JPEG capture",
                    "alignment": "each JPEG is paired with the most recent completed real env.step",
                }
                _atomic_json(metadata_path, metadata)
            self._state = "uploading"
            self._last_upload_status = "uploading"
            self._last_error = ""
            self._write_status()
        try:
            result = self.publisher(self._episode_dir, metadata)
        except Exception as exc:
            with self._lock:
                self._state = "upload-failed"
                self._last_upload_status = "failed"
                self._last_error = _safe_error(exc)
                self._write_status()
            raise
        with self._lock:
            self._completed = int(result["completed_episode_count"])
            self._last_episode_index = int(result["episode_index"])
            self._dataset_version_uri = str(result["dataset_version_uri"])
            self._last_episode_commit_uri = str(result.get("episode_commit_uri") or "")
            self._last_outcome = self._outcome
            self._last_upload_status = "uploaded"
            self._last_error = ""
            self._state = "idle"
            self._active_episode = None
            self._outcome = ""
            self._frames = 0
            self._episode_dir = None
            self._latest_step = None
            self._pending_camera_groups.clear()
            self._write_status()
            return result

    def process_commands(self, reset: Callable[[], Any] | None = None) -> None:
        finalize_future = self._finalize_future
        if finalize_future is not None and finalize_future.done():
            try:
                finalize_future.result()
            except Exception as exc:
                # finalize() already persisted a bounded, redacted failure in
                # recorder status. The operator can retry with a new command ID.
                LOGGER.debug(
                    "LeIsaac finalize failure was recorded in recorder status: %s",
                    _safe_error(exc),
                )
            else:
                finalize_reset = self._finalize_reset
                if finalize_reset is not None:
                    finalize_reset()
            finally:
                self._finalize_future = None
                self._finalize_reset = None
        try:
            with self.control_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._control_offset)
                records = handle.readlines()
                self._control_offset = handle.tell()
        except OSError:
            return
        for raw in records:
            request_id = ""
            command = ""
            try:
                payload = json.loads(raw)
                command = str(payload.get("command") or "")
                request_id = str(payload.get("request_id") or uuid.uuid4().hex)
                with self._lock:
                    processed_command = self._processed_commands.get(request_id)
                    if processed_command is not None:
                        if command != processed_command:
                            raise DatasetError(
                                "recorder request ID was reused for a different command"
                            )
                        self._clear_pending_command(request_id)
                        continue
                if command == "start":
                    self.start()
                elif command == "mark-success":
                    self.mark("success")
                elif command == "mark-failure":
                    self.mark("failure")
                elif command == "finalize":
                    if self._finalize_future is not None:
                        raise DatasetError("episode publication is already active")
                    self._finalize_reset = reset
                    self._finalize_future = self._publication_executor.submit(
                        self.finalize
                    )
                else:
                    raise DatasetError("unsupported recorder command")
            except Exception as exc:
                with self._lock:
                    self._last_error = _safe_error(exc)
                    self._write_status()
            finally:
                if request_id:
                    with self._lock:
                        if request_id not in self._processed_commands:
                            self._processed_commands[request_id] = command
                            self._last_command_id = request_id
                            self._last_command = command
                            self._command_revision += 1
                    self._clear_pending_command(request_id)
                    with self._lock:
                        self._write_status()

    def shutdown(self, *, wait: bool = True) -> None:
        """Drain the bounded publisher during an orderly simulator shutdown."""

        self._publication_executor.shutdown(wait=wait, cancel_futures=not wait)


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise DatasetError("episode records contain fewer than two frames")
    required = {
        "observation.state",
        "action",
        "reward",
        "terminated",
        "truncated",
        "done",
        "sim_step",
        "monotonic_ns",
        "wall_clock_ns",
        "frame_sha256",
    }
    previous_sim_step = -1
    previous_monotonic_ns = -1
    for row in rows:
        if not required.issubset(row):
            raise DatasetError("episode record is missing real simulator fields")
        _vector(row["observation.state"], 6, "joint_pos")
        _vector(row["action"], 8, "applied action")
        sim_step = int(row["sim_step"])
        monotonic_ns = int(row["monotonic_ns"])
        wall_clock_ns = int(row["wall_clock_ns"])
        if sim_step <= previous_sim_step or monotonic_ns <= previous_monotonic_ns:
            raise DatasetError(
                "episode simulator and monotonic timestamps must increase"
            )
        if wall_clock_ns <= 0:
            raise DatasetError("episode wall-clock timestamp must be positive")
        if not re.fullmatch(r"[a-f0-9]{64}", str(row["frame_sha256"])):
            raise DatasetError("episode record has an invalid frame checksum")
        camera_checksums = row.get("camera_frame_sha256")
        if camera_checksums is not None and (
            not isinstance(camera_checksums, dict)
            or not camera_checksums
            or any(
                not isinstance(camera, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", camera)
                or not re.fullmatch(r"[a-f0-9]{64}", str(checksum))
                for camera, checksum in camera_checksums.items()
            )
        ):
            raise DatasetError("episode record has invalid camera frame checksums")
        previous_sim_step = sim_step
        previous_monotonic_ns = monotonic_ns
    return rows


def _encode_frames(frame_dir: Path, destination: Path, fps: int = FPS) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame-%06d.jpg"),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-g",
        str(max(2, fps)),
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if (
        result.returncode
        or not destination.is_file()
        or destination.stat().st_size < 1024
    ):
        raise DatasetError(f"ffmpeg could not encode episode: {result.stderr.strip()}")


def _fraction(value: str) -> float:
    numerator, separator, denominator = str(value or "").partition("/")
    try:
        result = (
            float(numerator) / float(denominator) if separator else float(numerator)
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise DatasetError("encoded video has invalid frame-rate metadata") from exc
    if not math.isfinite(result) or result <= 0:
        raise DatasetError("encoded video has invalid frame-rate metadata")
    return result


def _validated_media_metadata(
    *,
    codec: str,
    pix_fmt: str,
    width: int,
    height: int,
    measured_fps: float,
    frames: int,
    duration: float,
    timestamps: list[float],
    expected_frames: int,
    fps: int,
) -> dict[str, Any]:
    expected_duration = expected_frames / fps
    if (
        codec != "h264"
        or pix_fmt != "yuv420p"
        or width <= 0
        or height <= 0
        or frames != expected_frames
        or len(timestamps) != expected_frames
        or abs(measured_fps - fps) > 0.001
        or abs(duration - expected_duration) > (1.0 / fps)
        or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
        or any(
            abs(timestamp - index / fps) > (0.5 / fps)
            for index, timestamp in enumerate(timestamps)
        )
    ):
        raise DatasetError("encoded video does not match episode frames and timestamps")
    return {
        "codec": "h264",
        "pix_fmt": "yuv420p",
        "width": width,
        "height": height,
        "fps": float(fps),
        "frames": frames,
        "duration": duration,
        "timestamps": timestamps,
        "has_audio": False,
    }


def _probe_video_with_ffmpeg(
    path: Path, *, expected_frames: int, fps: int
) -> dict[str, Any]:
    """Decode and inspect media with imageio's packaged ffmpeg when ffprobe is absent."""

    command = [
        _ffmpeg_executable(),
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        "showinfo",
        "-vsync",
        "0",
        "-f",
        "null",
        "-",
    ]
    environment = dict(os.environ)
    environment.update({"LANG": "C", "LC_ALL": "C"})
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode:
        raise DatasetError("ffmpeg could not decode and validate encoded episode")

    input_report = result.stderr.split("Stream mapping:", 1)[0]
    stream_lines = re.findall(r"^\s*Stream #0:\d+[^\n]*$", input_report, re.MULTILINE)
    video_lines = [line for line in stream_lines if ": Video:" in line]
    if len(video_lines) != 1 or len(stream_lines) != 1:
        raise DatasetError(
            "episode media must contain exactly one video stream and no audio"
        )
    stream = video_lines[0]
    stream_match = re.search(
        r": Video:\s*([^,\s]+)[^,]*,\s*([A-Za-z0-9_]+)(?:\([^)]*\))?,\s*"
        r"(\d+)x(\d+)(?:\s|\[)",
        stream,
    )
    rate_match = re.search(r",\s*([0-9]+(?:\.[0-9]+)?)\s+fps(?:,|\s)", stream)
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", input_report)
    frame_matches = re.findall(
        r"\bn:\s*(\d+)\s+pts:.*?\bpts_time:([0-9.eE+\-]+).*?"
        r"\bfmt:([A-Za-z0-9_]+).*?\bs:(\d+)x(\d+)",
        result.stderr,
    )
    if not stream_match or not rate_match or not duration_match or not frame_matches:
        raise DatasetError("ffmpeg returned malformed media metadata")

    codec, pix_fmt, width_text, height_text = stream_match.groups()
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    decoded = {
        int(index): (float(timestamp), frame_pix_fmt, int(width), int(height))
        for index, timestamp, frame_pix_fmt, width, height in frame_matches
    }
    indexes = sorted(decoded)
    if indexes != list(range(len(indexes))):
        raise DatasetError("ffmpeg returned non-contiguous decoded frame metadata")
    if any(
        item[1:] != (pix_fmt, int(width_text), int(height_text))
        for item in decoded.values()
    ):
        raise DatasetError("decoded frame metadata changes within the episode")
    return _validated_media_metadata(
        codec=codec,
        pix_fmt=pix_fmt,
        width=int(width_text),
        height=int(height_text),
        measured_fps=float(rate_match.group(1)),
        frames=len(indexes),
        duration=duration,
        timestamps=[decoded[index][0] for index in indexes],
        expected_frames=expected_frames,
        fps=fps,
    )


def _probe_video(path: Path, *, expected_frames: int, fps: int) -> dict[str, Any]:
    """Validate the encoded bytes and return LeRobot-compatible media metadata."""

    ffprobe = _ffprobe_executable()
    if ffprobe is None:
        return _probe_video_with_ffmpeg(path, expected_frames=expected_frames, fps=fps)
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-show_frames",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames,duration:format=duration:frame=media_type,best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise DatasetError("ffprobe could not validate encoded episode")
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetError("ffprobe returned malformed media metadata") from exc
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    if len(video_streams) != 1 or any(
        item.get("codec_type") != "video" for item in streams
    ):
        raise DatasetError(
            "episode media must contain exactly one video stream and no audio"
        )
    stream = video_streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
        frames = int(stream["nb_read_frames"])
        duration = float(stream.get("duration") or payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError("encoded video metadata is incomplete") from exc
    measured_fps = _fraction(str(stream.get("avg_frame_rate") or ""))
    timestamps = [
        float(item["best_effort_timestamp_time"])
        for item in payload.get("frames", [])
        if item.get("media_type") == "video"
        and item.get("best_effort_timestamp_time") is not None
    ]
    return _validated_media_metadata(
        codec=str(stream.get("codec_name") or ""),
        pix_fmt=str(stream.get("pix_fmt") or ""),
        width=width,
        height=height,
        measured_fps=measured_fps,
        frames=frames,
        duration=duration,
        timestamps=timestamps,
        expected_frames=expected_frames,
        fps=fps,
    )


def _validate_camera_media(
    videos: dict[str, Path], *, expected_frames: int, fps: int
) -> dict[str, dict[str, Any]]:
    probes = {
        camera: _probe_video(path, expected_frames=expected_frames, fps=fps)
        for camera, path in videos.items()
    }
    reference = next(iter(probes.values()))["timestamps"]
    tolerance = 0.25 / fps
    for probe in probes.values():
        if len(probe["timestamps"]) != len(reference) or any(
            abs(left - right) > tolerance
            for left, right in zip(reference, probe["timestamps"])
        ):
            raise DatasetError("recorded camera videos are not timestamp-aligned")
    return probes


def _stats(values: list[Any]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def build_lerobot_dataset(
    episodes: list[dict[str, Any]],
    destination: Path,
    *,
    episode_index_offset: int = 0,
    global_index_offset: int = 0,
    task_catalog: list[str] | None = None,
) -> dict[str, Any]:
    """Build an exact LeRobotDataset v3 tree from immutable raw episode bundles."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if not episodes:
        raise DatasetError("cannot build an empty dataset")
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    episode_tasks = {str(ep["metadata"]["task"]) for ep in episodes}
    tasks = sorted(set(task_catalog or []) | episode_tasks)
    task_indexes = {task: index for index, task in enumerate(tasks)}
    episode_meta: list[dict[str, Any]] = []
    stats_values: dict[str, list[Any]] = {
        "observation.state": [],
        "action": [],
        "reward": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
        "terminated": [],
        "truncated": [],
        "done": [],
        "success": [],
        "observation.timestamp.monotonic_ns": [],
        "observation.timestamp.wall_clock_ns": [],
    }
    global_index = global_index_offset
    for local_episode_index, episode in enumerate(episodes):
        episode_index = episode_index_offset + local_episode_index
        rows = _load_records(Path(episode["records_path"]))
        meta = episode["metadata"]
        if (
            meta.get("schema") != EPISODE_SCHEMA
            or meta.get("outcome") not in _OUTCOMES
            or int(meta.get("frame_count", -1)) != len(rows)
            or int(meta.get("fps", -1)) != FPS
        ):
            raise DatasetError(
                "episode metadata does not match the recorded frame sequence"
            )
        for row in rows:
            if (
                row.get("task") != meta.get("task")
                or row.get("environment_id") != meta.get("environment_id")
                or int(row.get("environment_index", -1))
                != int(meta.get("environment_index", -2))
                or int(row.get("seed", -1)) != int(meta.get("seed", -2))
            ):
                raise DatasetError(
                    "episode frame identity does not match its provenance"
                )
        task_index = task_indexes[str(meta["task"])]
        columns: dict[str, list[Any]] = {
            "observation.state": [],
            "action": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
            "done": [],
            "success": [],
            "reset_reason": [],
            "observation.timestamp.monotonic_ns": [],
            "observation.timestamp.wall_clock_ns": [],
            "environment.id": [],
            "environment.index": [],
            "seed": [],
            "sim_step": [],
            "source.frame_sha256": [],
            "episode_index": [],
            "frame_index": [],
            "timestamp": [],
            "index": [],
            "task_index": [],
        }
        dataset_from = global_index
        for frame_index, row in enumerate(rows):
            final = frame_index == len(rows) - 1
            values = {
                "observation.state": row["observation.state"],
                "action": row["action"],
                "reward": float(row["reward"]),
                "terminated": bool(row["terminated"]),
                "truncated": bool(row["truncated"]),
                "done": bool(row["done"]),
                "success": bool(final and meta["outcome"] == "success"),
                "reset_reason": str(meta["outcome"] if final else ""),
                "observation.timestamp.monotonic_ns": int(row["monotonic_ns"]),
                "observation.timestamp.wall_clock_ns": int(row["wall_clock_ns"]),
                "environment.id": str(meta["environment_id"]),
                "environment.index": int(meta["environment_index"]),
                "seed": int(meta["seed"]),
                "sim_step": int(row["sim_step"]),
                "source.frame_sha256": str(row["frame_sha256"]),
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": float(frame_index / FPS),
                "index": global_index,
                "task_index": task_index,
            }
            for key, value in values.items():
                columns[key].append(value)
            for key in stats_values:
                stats_values[key].append(values[key])
            global_index += 1
        data_path = destination / f"data/chunk-000/file-{episode_index:03d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        types: dict[str, Any] = {
            "observation.state": pa.list_(pa.float32(), 6),
            "action": pa.list_(pa.float32(), 8),
            "reward": pa.float32(),
            "terminated": pa.bool_(),
            "truncated": pa.bool_(),
            "done": pa.bool_(),
            "success": pa.bool_(),
            "reset_reason": pa.string(),
            "observation.timestamp.monotonic_ns": pa.int64(),
            "observation.timestamp.wall_clock_ns": pa.int64(),
            "environment.id": pa.string(),
            "environment.index": pa.int64(),
            "seed": pa.int64(),
            "sim_step": pa.int64(),
            "source.frame_sha256": pa.string(),
            "episode_index": pa.int64(),
            "frame_index": pa.int64(),
            "timestamp": pa.float32(),
            "index": pa.int64(),
            "task_index": pa.int64(),
        }
        pq.write_table(
            pa.table(
                {
                    key: pa.array(values, type=types[key])
                    for key, values in columns.items()
                }
            ),
            data_path,
            compression="snappy",
        )
        video_path = (
            destination / f"videos/{VIDEO_KEY}/chunk-000/file-{episode_index:03d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(episode["video_path"], video_path)
        episode_meta.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": 0,
                "data/file_index": episode_index,
                "dataset_from_index": dataset_from,
                "dataset_to_index": global_index,
                "length": len(rows),
                "tasks": [str(meta["task"])],
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": episode_index,
                f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
                f"videos/{VIDEO_KEY}/to_timestamp": len(rows) / FPS,
            }
        )
    meta_root = destination / "meta"
    episodes_path = meta_root / "episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(episode_meta), episodes_path, compression="snappy"
    )
    pq.write_table(
        pa.table(
            {
                "task": pa.array(tasks, type=pa.string()),
                "task_index": pa.array(range(len(tasks)), type=pa.int64()),
            }
        ),
        meta_root / "tasks.parquet",
        compression="snappy",
    )
    (meta_root / "stats.json").write_text(
        json.dumps(
            {key: _stats(value) for key, value in stats_values.items()}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    primary_media = [
        _probe_video(
            Path(episode["video_path"]),
            expected_frames=len(_load_records(Path(episode["records_path"]))),
            fps=FPS,
        )
        for episode in episodes
    ]
    first_media = primary_media[0]
    if any(
        (item["width"], item["height"], item["fps"], item["codec"], item["pix_fmt"])
        != (
            first_media["width"],
            first_media["height"],
            first_media["fps"],
            first_media["codec"],
            first_media["pix_fmt"],
        )
        for item in primary_media[1:]
    ):
        raise DatasetError("episode videos have inconsistent media metadata")
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [8], "names": ACTION_NAMES},
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        "terminated": {"dtype": "bool", "shape": [1], "names": None},
        "truncated": {"dtype": "bool", "shape": [1], "names": None},
        "done": {"dtype": "bool", "shape": [1], "names": None},
        "success": {"dtype": "bool", "shape": [1], "names": None},
        "reset_reason": {"dtype": "string", "shape": [1], "names": None},
        "observation.timestamp.monotonic_ns": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "observation.timestamp.wall_clock_ns": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "environment.id": {"dtype": "string", "shape": [1], "names": None},
        "environment.index": {"dtype": "int64", "shape": [1], "names": None},
        "seed": {"dtype": "int64", "shape": [1], "names": None},
        "sim_step": {"dtype": "int64", "shape": [1], "names": None},
        "source.frame_sha256": {"dtype": "string", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [first_media["height"], first_media["width"], 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": first_media["height"],
                "video.width": first_media["width"],
                "video.fps": first_media["fps"],
                "video.codec": first_media["codec"],
                "video.pix_fmt": first_media["pix_fmt"],
                "video.is_depth_map": False,
                "video.channels": 3,
                "has_audio": False,
            },
        },
    }
    info = {
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "robot_type": "so101",
        "total_episodes": episode_index_offset + len(episodes),
        "total_frames": global_index,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{episode_index_offset + len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "features": features,
    }
    _atomic_json(meta_root / "info.json", info)
    return info


class S3DatasetStore:
    """Append episode commits and publish immutable, resumable dataset versions."""

    def __init__(
        self,
        output_uri: str,
        *,
        client: Any | None = None,
        endpoint_url: str | None = None,
        config_endpoint: str | None = None,
    ) -> None:
        self.bucket, self.prefix = split_s3_uri(output_uri, label="output path")
        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=resolve_s3_endpoint(
                    endpoint_url, config_endpoint=config_endpoint
                ),
                region_name=os.environ.get("AWS_REGION") or "eu-north1",
            )
        self.client = client

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}/{suffix.lstrip('/')}"

    @staticmethod
    def _is_missing_object(exc: Exception) -> bool:
        if isinstance(exc, KeyError):
            return True
        response = getattr(exc, "response", {})
        error = response.get("Error", {}) if isinstance(response, dict) else {}
        return str(error.get("Code") or "") in {"404", "NoSuchKey", "NotFound"}

    @staticmethod
    def _is_precondition_failure(exc: Exception) -> bool:
        response = getattr(exc, "response", {})
        error = response.get("Error", {}) if isinstance(response, dict) else {}
        code = str(error.get("Code") or "")
        return code in {
            "409",
            "412",
            "ConditionalRequestConflict",
            "PreconditionFailed",
        } or ("precondition" in str(exc).lower())

    def _read_latest(self) -> tuple[dict[str, Any] | None, str]:
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=self._key("latest.json")
            )
        except Exception as exc:
            if self._is_missing_object(exc):
                return None, ""
            raise DatasetError("could not read the current dataset pointer") from exc
        try:
            payload = json.loads(response["Body"].read())
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError("current dataset pointer is malformed") from exc
        try:
            valid = (
                isinstance(payload, dict)
                and payload.get("schema") == "npa.leisaac.dataset-latest.v1"
                and int(payload.get("episode_count", -1)) >= 0
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise DatasetError("current dataset pointer is malformed")
        if not str(response.get("ETag") or ""):
            raise DatasetError("current dataset pointer has no concurrency token")
        return payload, str(response.get("ETag") or "")

    def _manifest_from_latest(self, latest: dict[str, Any]) -> dict[str, Any]:
        manifest_uri = str(latest.get("manifest_uri") or "")
        bucket, key = split_s3_uri(manifest_uri, label="latest manifest URI")
        if bucket != self.bucket or not key.startswith(self.prefix + "/versions/"):
            raise DatasetError("current dataset pointer escaped the output prefix")
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            manifest = json.loads(response["Body"].read())
        except Exception as exc:
            raise DatasetError("current dataset manifest is unreadable") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != DATASET_SCHEMA
            or int(manifest.get("episode_count", -1)) != int(latest["episode_count"])
        ):
            raise DatasetError("current dataset manifest does not match latest pointer")
        return manifest

    def _publish_latest_monotonic(
        self, latest: dict[str, Any], manifest: dict[str, Any]
    ) -> dict[str, Any]:
        body = (json.dumps(latest, indent=2, sort_keys=True) + "\n").encode()
        key = self._key("latest.json")
        for attempt in range(64):
            current, etag = self._read_latest()
            if current is not None and int(current["episode_count"]) >= int(
                latest["episode_count"]
            ):
                return self._manifest_from_latest(current)
            condition = (
                {"IfMatch": etag} if current is not None else {"IfNoneMatch": "*"}
            )
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=body,
                    Metadata={"sha256": hashlib.sha256(body).hexdigest()},
                    **condition,
                )
                return manifest
            except Exception as exc:
                if self._is_precondition_failure(exc):
                    time.sleep(min(0.5, 0.01 * (attempt + 1)))
                    continue
                raise DatasetError(
                    "could not publish the latest dataset pointer"
                ) from exc
        raise DatasetError(
            "latest dataset pointer concurrency did not converge after 64 attempts"
        )

    def _put_file(self, key: str, path: Path) -> dict[str, Any]:
        checksum = sha256_file(path)
        size = path.stat().st_size
        object_key = self._key(key)
        try:
            with path.open("rb") as handle:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=handle,
                    Metadata={"sha256": checksum},
                    IfNoneMatch="*",
                )
        except Exception as exc:
            if not self._is_precondition_failure(exc):
                raise DatasetError(
                    "could not publish immutable dataset object"
                ) from exc
            try:
                current = self.client.head_object(Bucket=self.bucket, Key=object_key)
                current_checksum = str(
                    (current.get("Metadata") or {}).get("sha256") or ""
                )
                current_size = int(current.get("ContentLength", -1))
            except Exception as head_exc:
                raise DatasetError(
                    "could not verify immutable object collision"
                ) from head_exc
            if current_checksum != checksum or current_size != size:
                raise DatasetError(
                    "immutable dataset object already exists with different bytes"
                ) from exc
        return {"key": object_key, "sha256": checksum, "bytes": size}

    def resume_status(self) -> dict[str, Any]:
        """Load the recorder's last committed state from the authoritative pointer."""

        latest, _etag = self._read_latest()
        if latest is None:
            return {
                "completed_episode_count": 0,
                "last_episode_index": None,
                "last_outcome": "",
                "last_upload_status": "never",
                "dataset_version_uri": "",
                "last_episode_commit_uri": "",
            }
        manifest = self._manifest_from_latest(latest)
        completed = int(manifest["episode_count"])
        commit_uris = manifest.get("episode_commits")
        commit_uri = str(manifest.get("new_episode_commit") or "")
        if not commit_uri and isinstance(commit_uris, list) and commit_uris:
            commit_uri = str(commit_uris[-1] or "")
        if completed <= 0 or not commit_uri:
            raise DatasetError("current dataset manifest has invalid episode commits")
        bucket, key = split_s3_uri(commit_uri, label="latest episode commit URI")
        if bucket != self.bucket or not key.startswith(self.prefix + "/commits/"):
            raise DatasetError("latest episode commit escaped the output prefix")
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            commit = json.loads(response["Body"].read())
        except Exception as exc:
            raise DatasetError("latest episode commit is unreadable") from exc
        metadata = commit.get("metadata") if isinstance(commit, dict) else None
        outcome = (
            str(metadata.get("outcome") or "") if isinstance(metadata, dict) else ""
        )
        episode_index = (
            int(commit.get("episode_index", -1)) if isinstance(commit, dict) else -1
        )
        schema = commit.get("schema") if isinstance(commit, dict) else None
        if (
            schema != "npa.leisaac.episode-commit.v1"
            or episode_index != completed - 1
            or outcome not in _OUTCOMES
        ):
            raise DatasetError("latest episode commit does not match dataset version")
        return {
            "completed_episode_count": completed,
            "last_episode_index": episode_index,
            "last_outcome": outcome,
            "last_upload_status": "uploaded",
            "dataset_version_uri": str(manifest["dataset_uri"]),
            "last_episode_commit_uri": commit_uri,
        }

    def _commits(self) -> list[dict[str, Any]]:
        """Legacy migration reader; normal finalization uses bounded indexes."""
        prefix = self._key("commits/episode-")
        result: list[dict[str, Any]] = []
        token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self.client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                key = str(item["Key"])
                if re.search(r"episode-\d{6}\.json$", key):
                    body = self.client.get_object(Bucket=self.bucket, Key=key)[
                        "Body"
                    ].read()
                    result.append(json.loads(body))
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        result = sorted(result, key=lambda item: int(item["episode_index"]))
        for expected, commit in enumerate(result):
            if (
                commit.get("schema") != "npa.leisaac.episode-commit.v1"
                or int(commit.get("episode_index", -1)) != expected
                or not isinstance(commit.get("metadata"), dict)
                or not isinstance(commit.get("objects"), dict)
            ):
                raise DatasetError(
                    "existing episode commits are stale, malformed, or non-contiguous"
                )
        return result

    def _commit_for_uuid(self, episode_uuid: str) -> dict[str, Any] | None:
        index_key = self._key(f"episode-uuids/{episode_uuid}.json")
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=index_key)
        except Exception as exc:
            if self._is_missing_object(exc):
                return None
            raise DatasetError("could not read the episode UUID index") from exc
        try:
            pointer = json.loads(response["Body"].read())
            commit_key = str(pointer["commit_key"])
            commit_response = self.client.get_object(Bucket=self.bucket, Key=commit_key)
            commit = json.loads(commit_response["Body"].read())
        except Exception as exc:
            raise DatasetError("episode UUID index is malformed") from exc
        if commit.get("episode_uuid") != episode_uuid or commit_key != self._key(
            f"commits/episode-{int(commit.get('episode_index', -1)):06d}.json"
        ):
            raise DatasetError("episode UUID index does not match its commit")
        return commit

    def _commit_at_index(self, episode_index: int) -> dict[str, Any] | None:
        key = self._key(f"commits/episode-{episode_index:06d}.json")
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_missing_object(exc):
                return None
            raise DatasetError("could not read the episode commit") from exc
        try:
            commit = json.loads(response["Body"].read())
        except Exception as exc:
            raise DatasetError("episode commit is malformed") from exc
        if (
            not isinstance(commit, dict)
            or commit.get("schema") != "npa.leisaac.episode-commit.v1"
            or int(commit.get("episode_index", -1)) != episode_index
        ):
            raise DatasetError("episode commit index is malformed")
        return commit

    @staticmethod
    def _validate_retry_sources(
        commit: dict[str, Any], records: Path, primary_video: Path
    ) -> None:
        objects = commit.get("objects")
        videos = objects.get("videos") if isinstance(objects, dict) else None
        storage = objects.get("camera_storage") if isinstance(objects, dict) else None
        primary = (
            str(storage.get("primary_camera") or "")
            if isinstance(storage, dict)
            else ""
        )
        record_ref = objects.get("records") if isinstance(objects, dict) else None
        video_ref = videos.get(primary) if isinstance(videos, dict) else None
        if (
            not isinstance(record_ref, dict)
            or not isinstance(video_ref, dict)
            or str(record_ref.get("sha256") or "") != sha256_file(records)
            or str(video_ref.get("sha256") or "") != sha256_file(primary_video)
        ):
            raise DatasetError("episode UUID retry has different source bytes")

    def _publish_uuid_index(self, commit: dict[str, Any]) -> None:
        episode_uuid = str(commit["episode_uuid"])
        commit_key = self._key(
            f"commits/episode-{int(commit['episode_index']):06d}.json"
        )
        payload = {
            "schema": "npa.leisaac.episode-uuid-index.v1",
            "episode_uuid": episode_uuid,
            "commit_key": commit_key,
            "metadata_sha256": hashlib.sha256(
                json.dumps(
                    commit["metadata"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        body = (json.dumps(payload, sort_keys=True) + "\n").encode()
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._key(f"episode-uuids/{episode_uuid}.json"),
                Body=body,
                Metadata={"sha256": hashlib.sha256(body).hexdigest()},
                IfNoneMatch="*",
            )
        except Exception as exc:
            if not self._is_precondition_failure(exc):
                raise DatasetError("could not publish the episode UUID index") from exc
            existing = self._commit_for_uuid(episode_uuid)
            if existing is None or existing.get("metadata") != commit.get("metadata"):
                raise DatasetError(
                    "episode UUID already exists with different provenance"
                ) from exc

    def publish_episode(
        self, episode_dir: Path, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        raw_cameras = metadata.get("cameras")
        camera_ids = (
            [str(camera) for camera in raw_cameras]
            if isinstance(raw_cameras, list) and raw_cameras
            else ["primary"]
        )
        if (
            len(camera_ids) > 4
            or len(set(camera_ids)) != len(camera_ids)
            or any(
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", camera)
                for camera in camera_ids
            )
        ):
            raise DatasetError("episode metadata has invalid cameras")
        videos: dict[str, Path] = {}
        for camera_index, camera in enumerate(camera_ids):
            frame_dir = (
                episode_dir / "frames"
                if camera_index == 0
                else episode_dir / f"frames-{camera}"
            )
            video = (
                episode_dir / "episode.mp4"
                if camera_index == 0
                else episode_dir / f"episode-{camera}.mp4"
            )
            _encode_frames(frame_dir, video, int(metadata.get("fps") or FPS))
            videos[camera] = video
        records = episode_dir / "records.jsonl"
        rows = _load_records(records)
        primary_frame_paths = sorted((episode_dir / "frames").glob("frame-*.jpg"))
        if len(primary_frame_paths) != len(rows) or [
            sha256_file(path) for path in primary_frame_paths
        ] != [str(row["frame_sha256"]) for row in rows]:
            raise DatasetError("raw primary frames do not match episode records")
        fps = int(metadata.get("fps") or FPS)
        media = _validate_camera_media(videos, expected_frames=len(rows), fps=fps)
        primary_camera = camera_ids[0]
        existing = self._commit_for_uuid(str(metadata["episode_uuid"]))
        if existing is not None:
            if existing.get("metadata") != metadata:
                raise DatasetError(
                    "episode UUID already exists with different provenance"
                )
            latest, _etag = self._read_latest()
            episode_index = int(existing["episode_index"])
            if latest is not None and int(latest["episode_count"]) > episode_index:
                version = self._manifest_from_latest(latest)
            else:
                if (int(latest["episode_count"]) if latest else 0) != episode_index:
                    raise DatasetError("episode UUID index is ahead of dataset history")
                self._validate_retry_sources(existing, records, videos[primary_camera])
                previous = self._manifest_from_latest(latest) if latest else None
                version = self._publish_version(
                    previous,
                    existing,
                    records_path=records,
                    video_path=videos[primary_camera],
                )
            return {
                "episode_index": episode_index,
                "completed_episode_count": int(version["episode_count"]),
                "dataset_version_uri": version["dataset_uri"],
                "episode_commit_uri": f"s3://{self.bucket}/{self._key(f'commits/episode-{episode_index:06d}.json')}",
            }
        bundle = f"episodes/by-id/{metadata['episode_uuid']}"
        objects: dict[str, Any] = {
            "records": self._put_file(f"{bundle}/records.jsonl", records),
            "metadata": self._put_file(
                f"{bundle}/episode.json", episode_dir / "episode.json"
            ),
            "videos": {
                camera: self._put_file(f"{bundle}/videos/{camera}.mp4", videos[camera])
                for camera in camera_ids
            },
            "frames_by_camera": {
                camera: [
                    self._put_file(f"{bundle}/frames/{camera}/{path.name}", path)
                    for path in sorted(
                        (
                            episode_dir / "frames"
                            if index == 0
                            else episode_dir / f"frames-{camera}"
                        ).glob("frame-*.jpg")
                    )
                ]
                for index, camera in enumerate(camera_ids)
            },
            "camera_storage": {
                "schema": "npa.leisaac.camera-storage.v1",
                "primary_camera": primary_camera,
                "videos": "objects.videos",
                "frames": "objects.frames_by_camera",
                "stored_once": True,
            },
        }
        previous_manifest: dict[str, Any] | None = None
        for concurrency_attempt in range(64):
            latest, _etag = self._read_latest()
            previous_manifest = self._manifest_from_latest(latest) if latest else None
            episode_index = int(latest["episode_count"]) if latest else 0
            commit = {
                "schema": "npa.leisaac.episode-commit.v1",
                "episode_index": episode_index,
                "episode_uuid": metadata["episode_uuid"],
                "committed_at": utc_now(),
                "metadata": metadata,
                "objects": objects,
                "media": media,
            }
            commit_bytes = (
                json.dumps(commit, indent=2, sort_keys=True) + "\n"
            ).encode()
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=self._key(f"commits/episode-{episode_index:06d}.json"),
                    Body=commit_bytes,
                    Metadata={"sha256": hashlib.sha256(commit_bytes).hexdigest()},
                    IfNoneMatch="*",
                )
                break
            except Exception as exc:
                if not self._is_precondition_failure(exc):
                    raise DatasetError("could not publish episode commit") from exc
                conflicting = self._commit_at_index(episode_index)
                if (
                    conflicting is not None
                    and conflicting.get("episode_uuid") == metadata["episode_uuid"]
                ):
                    if conflicting.get("metadata") != metadata:
                        raise DatasetError(
                            "episode UUID already exists with different provenance"
                        ) from exc
                    self._validate_retry_sources(
                        conflicting, records, videos[primary_camera]
                    )
                    self._publish_uuid_index(conflicting)
                    version = self._publish_version(
                        previous_manifest,
                        conflicting,
                        records_path=records,
                        video_path=videos[primary_camera],
                    )
                    return {
                        "episode_index": episode_index,
                        "completed_episode_count": int(version["episode_count"]),
                        "dataset_version_uri": version["dataset_uri"],
                        "episode_commit_uri": f"s3://{self.bucket}/{self._key(f'commits/episode-{episode_index:06d}.json')}",
                    }
                concurrent = self._commit_for_uuid(str(metadata["episode_uuid"]))
                if concurrent is not None:
                    if concurrent.get("metadata") != metadata:
                        raise DatasetError(
                            "episode UUID already exists with different provenance"
                        ) from exc
                    for _ in range(64):
                        concurrent_latest, _etag = self._read_latest()
                        if concurrent_latest and int(
                            concurrent_latest["episode_count"]
                        ) > int(concurrent["episode_index"]):
                            version = self._manifest_from_latest(concurrent_latest)
                            return {
                                "episode_index": int(concurrent["episode_index"]),
                                "completed_episode_count": int(
                                    version["episode_count"]
                                ),
                                "dataset_version_uri": version["dataset_uri"],
                                "episode_commit_uri": "s3://"
                                + self.bucket
                                + "/"
                                + self._key(
                                    f"commits/episode-{int(concurrent['episode_index']):06d}.json"
                                ),
                            }
                        time.sleep(0.05)
                    raise DatasetError(
                        "concurrent episode commit did not publish its dataset version"
                    ) from exc
                time.sleep(min(0.5, 0.01 * (concurrency_attempt + 1)))
        else:
            raise DatasetError("episode commit concurrency did not converge")
        self._publish_uuid_index(commit)
        version = self._publish_version(
            previous_manifest,
            commit,
            records_path=records,
            video_path=videos[primary_camera],
        )
        return {
            "episode_index": episode_index,
            "completed_episode_count": int(version["episode_count"]),
            "dataset_version_uri": version["dataset_uri"],
            "episode_commit_uri": f"s3://{self.bucket}/{self._key(f'commits/episode-{episode_index:06d}.json')}",
        }

    def _download(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(destination))

    def _publish_version(
        self,
        previous: dict[str, Any] | None,
        commit: dict[str, Any],
        *,
        records_path: Path,
        video_path: Path,
    ) -> dict[str, Any]:
        """Publish only this episode's immutable LeRobot shard and a small index."""

        with tempfile.TemporaryDirectory(prefix="npa-leisaac-version-") as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            previous_info = previous.get("info", {}) if previous else {}
            task_catalog = sorted(
                set(previous.get("task_catalog", []) if previous else [])
                | {str(commit["metadata"]["task"])}
            )
            shard_info = build_lerobot_dataset(
                [
                    {
                        "records_path": records_path,
                        "video_path": video_path,
                        "metadata": commit["metadata"],
                    }
                ],
                dataset_root,
                episode_index_offset=int(commit["episode_index"]),
                global_index_offset=int(previous_info.get("total_frames") or 0),
                task_catalog=task_catalog,
            )
            episode_count = int(commit["episode_index"]) + 1
            version_id = f"v{episode_count:06d}-{uuid.uuid4().hex}"
            version_prefix = f"versions/{version_id}"
            manifest_key = self._key(f"{version_prefix}/npa-dataset.json")
            manifest_uri = f"s3://{self.bucket}/{manifest_key}"
            shard_prefix = f"lerobot-shards/{commit['episode_uuid']}"
            shard_files: list[dict[str, Any]] = []
            for path in sorted(
                item for item in dataset_root.rglob("*") if item.is_file()
            ):
                shard_files.append(
                    self._put_file(
                        f"{shard_prefix}/{path.relative_to(dataset_root).as_posix()}",
                        path,
                    )
                )
            commit_uri = f"s3://{self.bucket}/" + self._key(
                f"commits/episode-{int(commit['episode_index']):06d}.json"
            )
            info = dict(shard_info)
            manifest = {
                "schema": DATASET_SCHEMA,
                "lerobot_version": LEROBOT_TARGET_VERSION,
                "lerobot_codebase_version": LEROBOT_CODEBASE_VERSION,
                "dataset_uri": f"s3://{self.bucket}/{self._key(version_prefix)}",
                "output_prefix": f"s3://{self.bucket}/{self.prefix}",
                "version": version_id,
                "created_at": utc_now(),
                "episode_count": episode_count,
                # v2 manifests are constant-sized parent-linked snapshots. Old
                # v1 manifests with cumulative episode_commits/files remain
                # readable, but finalization never recopies those arrays.
                "index_layout": "parent-linked-v2",
                "info": info,
                "task_catalog": task_catalog,
                "files": shard_files,
                "storage_layout": "immutable-episode-shards-v1",
                "new_episode_commit": commit_uri,
                "new_episode_files": shard_files,
                "parent_manifest_uri": (
                    str(previous.get("manifest_uri") or "") if previous else ""
                ),
                "manifest_uri": manifest_uri,
            }
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode()
            self.client.put_object(
                Bucket=self.bucket,
                Key=manifest_key,
                Body=manifest_bytes,
                Metadata={"sha256": hashlib.sha256(manifest_bytes).hexdigest()},
                IfNoneMatch="*",
            )
            latest = {
                "schema": "npa.leisaac.dataset-latest.v1",
                "dataset_uri": manifest["dataset_uri"],
                "manifest_uri": manifest_uri,
                "episode_count": episode_count,
                "updated_at": utc_now(),
            }
            return self._publish_latest_monotonic(latest, manifest)
