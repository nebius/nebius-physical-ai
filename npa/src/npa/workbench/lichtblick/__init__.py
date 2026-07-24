"""Lichtblick web-viewer + frames->MCAP export for Workbench.

Lichtblick is an open-source, Foxglove-compatible MCAP / ROS-bag / robotics log
viewer (MPL-2.0). This module is the single source of truth for the
``lichtblick`` tool. The CLI (``npa workbench lichtblick ...``) and SDK
(``npa.sdk.workbench.lichtblick``) both call into these functions.

What it does that is tangible:

- ``build_mcap_from_frames`` turns a sequence of PNG/JPEG robot camera frames
  (e.g. the Sim2Real ``augment/frames`` artifacts, or any PNG/JPEG rollout camera
  export) into a real MCAP of ``foxglove.CompressedImage`` messages that a
  Foxglove-compatible viewer can play back. Raw ``.ppm`` rollout frames are not
  browser-decodable as CompressedImage and are skipped (convert to PNG first).
- ``stage_input_to_mcap`` stages an artifact from S3 (or local): either an
  existing ``.mcap`` (downloaded as-is) or a camera-frames prefix (downloaded and
  packed into an MCAP via the exporter).
- ``serve_viewer`` resolves the ``npa-lichtblick`` image, co-serves the staged
  MCAP from the viewer's own origin, and (when ``execute``) runs the container so
  the log is actually viewable at the returned deep-linked URL.

Cross-tool data flows through S3 only, so every command takes ``--input-path`` /
``--output-path`` S3 (or local) URIs. This module is import-safe: boto3, mcap and
subprocess are imported lazily so unit tests can exercise it with injected fakes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import quote

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
# Container-internal port the caddy static server binds (see the Dockerfile).
CONTAINER_PORT = 8080
# Path inside the container where a staged log artifact is served from.
SERVED_DATA_DIR = "/srv/data"
# Recognized robotics log containers the viewer can open directly.
SUPPORTED_SUFFIXES = (".mcap", ".bag", ".db3")
# Camera-frame image types the frames->MCAP exporter accepts.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
DEFAULT_CAMERA_TOPIC = "/camera"
DEFAULT_FPS = 10.0

# foxglove.CompressedImage well-known schema (jsonschema encoding). Lichtblick /
# Foxglove render this in the Image panel; bytes fields are base64 in JSON.
_COMPRESSED_IMAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "foxglove.CompressedImage",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "frame_id": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
    },
}


class LichtblickError(ValueError):
    """Raised when a Lichtblick viewer request is invalid."""


@dataclass(frozen=True)
class LichtblickLaunchPlan:
    """A fully-resolved description of how to serve a log in Lichtblick."""

    status: str
    input_path: str
    artifact_name: str
    host: str
    port: int
    image: str
    served_artifact_path: str
    viewer_url: str
    docker_command: str
    output_path: str
    staged: bool
    message_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# MCAP export: robot camera frames -> foxglove.CompressedImage MCAP
# --------------------------------------------------------------------------- #
def build_mcap_from_frames(
    frame_paths: list[str],
    output_path: str,
    *,
    topic: str = DEFAULT_CAMERA_TOPIC,
    fps: float = DEFAULT_FPS,
    frame_id: str = "camera",
) -> dict[str, Any]:
    """Pack an ordered list of image files into an MCAP of CompressedImage msgs.

    Deterministic ``log_time`` at ``fps`` so the viewer shows a real playback
    timeline. Returns a summary dict (mcap_path, message_count, topic, fps).
    """

    import base64
    import json

    from mcap.writer import Writer

    if not frame_paths:
        raise LichtblickError("no camera frames found to export to MCAP.")
    if fps <= 0:
        raise LichtblickError(f"--fps must be > 0, got {fps}.")
    period_ns = int(1_000_000_000 / fps)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "wb") as handle:
        writer = Writer(handle)
        writer.start(profile="", library="npa-lichtblick")
        schema_id = writer.register_schema(
            name="foxglove.CompressedImage",
            encoding="jsonschema",
            data=json.dumps(_COMPRESSED_IMAGE_SCHEMA).encode("utf-8"),
        )
        channel_id = writer.register_channel(
            topic=topic, message_encoding="json", schema_id=schema_id
        )
        for index, path in enumerate(frame_paths):
            with open(path, "rb") as frame:
                payload = frame.read()
            fmt = "png" if path.lower().endswith(".png") else "jpeg"
            stamp = index * period_ns
            message = {
                "timestamp": {"sec": stamp // 1_000_000_000, "nsec": stamp % 1_000_000_000},
                "frame_id": frame_id,
                "data": base64.b64encode(payload).decode("ascii"),
                "format": fmt,
            }
            writer.add_message(
                channel_id=channel_id,
                log_time=stamp,
                publish_time=stamp,
                data=json.dumps(message).encode("utf-8"),
            )
        writer.finish()
    return {
        "mcap_path": output_path,
        "message_count": len(frame_paths),
        "topic": topic,
        "fps": fps,
    }


# --------------------------------------------------------------------------- #
# Staging: S3 (or local) -> local .mcap ready to serve
# --------------------------------------------------------------------------- #
def _split_s3(uri: str) -> tuple[str, str]:
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise LichtblickError(f"invalid s3 uri: {uri}")
    return bucket, key


def _default_s3_client() -> Any:
    """Build an S3 client honoring npa-managed config/credentials.

    Resolves endpoint + HMAC keys via ``resolve_project_storage`` (i.e.
    ``~/.npa/config.yaml`` + ``~/.npa/credentials.yaml``), matching how the other
    viewers/workflows reach object storage, then falls back to boto3's ambient
    credential chain. Unit tests inject a fake client, so this path is not
    exercised against real infra.
    """

    import logging

    import boto3

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL") or ""
    access_key = ""
    secret_key = ""
    try:
        from npa.clients.config import resolve_project_storage

        storage = resolve_project_storage()
        endpoint = endpoint or (storage.endpoint_url or "")
        access_key = storage.aws_access_key_id or ""
        secret_key = storage.aws_secret_access_key or ""
    except Exception:
        # Best-effort: fall back to boto3's ambient credential chain / env.
        logging.getLogger(__name__).debug("npa storage config unavailable", exc_info=True)

    kwargs: dict[str, Any] = {}
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def _list_frame_keys(prefix_uri: str, *, s3_client: Any) -> list[tuple[str, str]]:
    bucket, prefix = _split_s3(prefix_uri.rstrip("/") + "/")
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3_client.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []) or []:
            key = obj["Key"]
            if key.lower().endswith(IMAGE_SUFFIXES):
                keys.append(key)
        if response.get("IsTruncated"):
            token = response.get("NextContinuationToken")
        else:
            break
    keys.sort()
    return [(bucket, key) for key in keys]


def _collect_local_frames(directory: str) -> list[str]:
    entries = [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(IMAGE_SUFFIXES)
    ]
    return entries


def stage_input_to_mcap(
    input_path: str,
    workdir: str,
    *,
    from_frames: bool = False,
    topic: str = DEFAULT_CAMERA_TOPIC,
    fps: float = DEFAULT_FPS,
    s3_client: Any | None = None,
) -> tuple[str, int | None]:
    """Return ``(local_mcap_path, message_count)`` for ``input_path``.

    - ``from_frames``: ``input_path`` is a camera-frames prefix (S3) or directory
      (local); frames are downloaded/collected and packed into an MCAP.
    - otherwise: ``input_path`` is an existing ``.mcap`` (downloaded/copied as-is).
    """

    value = (input_path or "").strip()
    if not value:
        raise LichtblickError("--input-path is required.")
    os.makedirs(workdir, exist_ok=True)

    if from_frames:
        frames_dir = os.path.join(workdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        if value.startswith("s3://"):
            client = s3_client or _default_s3_client()
            frame_refs = _list_frame_keys(value, s3_client=client)
            if not frame_refs:
                raise LichtblickError(f"no camera frames ({', '.join(IMAGE_SUFFIXES)}) under {value}")
            local_frames: list[str] = []
            for index, (bucket, key) in enumerate(frame_refs):
                dest = os.path.join(frames_dir, f"{index:06d}{PurePosixPath(key).suffix.lower()}")
                client.download_file(bucket, key, dest)
                local_frames.append(dest)
        else:
            if not os.path.isdir(value):
                raise LichtblickError(f"--from-frames expects a directory or s3 prefix, got {value!r}")
            local_frames = _collect_local_frames(value)
            if not local_frames:
                raise LichtblickError(f"no camera frames ({', '.join(IMAGE_SUFFIXES)}) in {value}")
        out = os.path.join(workdir, "camera.mcap")
        info = build_mcap_from_frames(local_frames, out, topic=topic, fps=fps)
        return out, int(info["message_count"])

    name = _validate_artifact(value)
    out = os.path.join(workdir, name)
    if value.startswith("s3://"):
        client = s3_client or _default_s3_client()
        bucket, key = _split_s3(value)
        client.download_file(bucket, key, out)
    else:
        if not os.path.isfile(value):
            raise LichtblickError(f"local artifact not found: {value}")
        shutil.copyfile(value, out)
    return out, None


def _validate_artifact(input_path: str) -> str:
    """Return the artifact basename for a valid MCAP/bag S3 or local URI."""

    value = (input_path or "").strip()
    if not value:
        raise LichtblickError("--input-path is required (S3 or local MCAP/bag artifact).")
    if "://" in value and not value.startswith("s3://"):
        scheme = value.split("://", 1)[0]
        raise LichtblickError(
            f"--input-path must be an s3:// URI or a local path, got scheme {scheme!r}."
        )
    if value.startswith("s3://"):
        _, key = _split_s3(value)
        name = PurePosixPath(key).name
    else:
        name = PurePosixPath(value).name
    if not name:
        raise LichtblickError(f"could not derive an artifact name from {value!r}.")
    if PurePosixPath(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(SUPPORTED_SUFFIXES)
        raise LichtblickError(
            f"unsupported artifact {name!r}; Lichtblick opens {supported} logs "
            "(use --from-frames to pack a camera-frame sequence into MCAP)."
        )
    return name


def _resolve_image(image: str, *, registry: str | None, tag: str | None) -> str:
    if image.strip():
        return image.strip()
    # Imported lazily so unit tests importing this module never require the
    # deploy stack or registry resolution.
    from npa.deploy.images import container_image_for_tool

    return container_image_for_tool("lichtblick", registry=registry, tag=tag)


def build_launch_plan(
    *,
    input_path: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    image: str = "",
    output_path: str = "",
    registry: str | None = None,
    tag: str | None = None,
    artifact_name: str = "",
    message_count: int | None = None,
    staged: bool = False,
) -> LichtblickLaunchPlan:
    """Return a validated plan for serving a log in Lichtblick.

    ``artifact_name`` overrides the served filename (used after staging, when the
    local file name differs from the input URI). The MCAP is co-served from the
    viewer's own origin, so the browser fetch is same-origin (no CORS/presign).
    """

    name = artifact_name.strip() or _validate_artifact(input_path)
    resolved_host = (host or DEFAULT_HOST).strip() or DEFAULT_HOST
    if port <= 0 or port > 65535:
        raise LichtblickError(f"--port must be in 1..65535, got {port}.")
    resolved_image = _resolve_image(image, registry=registry, tag=tag)
    served_artifact_path = f"{SERVED_DATA_DIR}/{name}"

    # A wildcard bind (0.0.0.0/::) is not a navigable browser host, so the deep
    # link uses a loopback connect host while the container still binds the
    # wildcard for reachability.
    connect_host = "127.0.0.1" if resolved_host in ("0.0.0.0", "::", "*") else resolved_host
    served_url = f"http://{connect_host}:{port}/data/{name}"
    viewer_url = (
        f"http://{connect_host}:{port}/?ds=remote-file"
        f"&ds.url={quote(served_url, safe='')}"
    )
    docker_command = (
        f"docker run --rm -p {resolved_host}:{port}:{CONTAINER_PORT} "
        f"-v <local-artifact>:{served_artifact_path}:ro {resolved_image}"
    )
    return LichtblickLaunchPlan(
        status="planned",
        input_path=input_path.strip(),
        artifact_name=name,
        host=resolved_host,
        port=port,
        image=resolved_image,
        served_artifact_path=served_artifact_path,
        viewer_url=viewer_url,
        docker_command=docker_command,
        output_path=output_path.strip(),
        staged=staged,
        message_count=message_count,
    )


def launch_viewer(
    plan: LichtblickLaunchPlan,
    *,
    local_artifact: str,
    runner: Callable[[list[str]], Any],
    detach: bool = True,
    container_name: str = "",
) -> LichtblickLaunchPlan:
    """Run the viewer container via an injected ``runner`` (e.g. subprocess.run)."""

    artifact = (local_artifact or "").strip()
    if not artifact:
        raise LichtblickError("local_artifact is required to launch the viewer.")
    argv = ["docker", "run", "--rm"]
    if detach:
        argv.append("-d")
    if container_name.strip():
        argv.extend(["--name", container_name.strip()])
    argv.extend(
        [
            "-p",
            f"{plan.host}:{plan.port}:{CONTAINER_PORT}",
            "-v",
            f"{artifact}:{plan.served_artifact_path}:ro",
            plan.image,
        ]
    )
    runner(argv)
    return LichtblickLaunchPlan(**{**plan.to_dict(), "status": "launched", "staged": True})


def serve_viewer(
    *,
    input_path: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    image: str = "",
    output_path: str = "",
    from_frames: bool = False,
    topic: str = DEFAULT_CAMERA_TOPIC,
    fps: float = DEFAULT_FPS,
    execute: bool = False,
    registry: str | None = None,
    tag: str | None = None,
    workdir: str = "",
    s3_client: Any | None = None,
    runner: Callable[[list[str]], Any] | None = None,
    container_name: str = "npa-lichtblick",
) -> LichtblickLaunchPlan:
    """Stage a log (and, with ``execute``, run the viewer container).

    Without ``execute`` this returns a plan (infra-free) and, for the frames path,
    the MCAP is not built. With ``execute`` it stages ``input_path`` from S3, packs
    camera frames into MCAP when ``from_frames``, and runs the container so the log
    is live at ``plan.viewer_url``.
    """

    if not execute:
        # Plan only. For frames, describe the MCAP that would be produced.
        artifact_name = "camera.mcap" if from_frames else _validate_artifact(input_path)
        return build_launch_plan(
            input_path=input_path,
            host=host,
            port=port,
            image=image,
            output_path=output_path,
            registry=registry,
            tag=tag,
            artifact_name=artifact_name,
        )

    work = workdir or tempfile.mkdtemp(prefix="npa-lichtblick-")
    local_mcap, message_count = stage_input_to_mcap(
        input_path,
        work,
        from_frames=from_frames,
        topic=topic,
        fps=fps,
        s3_client=s3_client,
    )
    plan = build_launch_plan(
        input_path=input_path,
        host=host,
        port=port,
        image=image,
        output_path=output_path,
        registry=registry,
        tag=tag,
        artifact_name=os.path.basename(local_mcap),
        message_count=message_count,
        staged=True,
    )
    exec_runner = runner or _default_docker_runner
    return launch_viewer(
        plan,
        local_artifact=os.path.abspath(local_mcap),
        runner=exec_runner,
        detach=True,
        container_name=container_name,
    )


def _default_docker_runner(argv: list[str]) -> Any:
    import subprocess

    return subprocess.run(argv, check=True, text=True, capture_output=True)


__all__ = [
    "CONTAINER_PORT",
    "DEFAULT_CAMERA_TOPIC",
    "DEFAULT_FPS",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "IMAGE_SUFFIXES",
    "LichtblickError",
    "LichtblickLaunchPlan",
    "SUPPORTED_SUFFIXES",
    "build_launch_plan",
    "build_mcap_from_frames",
    "launch_viewer",
    "serve_viewer",
    "stage_input_to_mcap",
]
