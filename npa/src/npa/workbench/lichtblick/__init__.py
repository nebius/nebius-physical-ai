"""Lichtblick web-viewer + frames->MCAP export for Workbench.

Lichtblick is an open-source, Foxglove-compatible MCAP / ROS-bag / robotics log
viewer (MPL-2.0). This module is the single source of truth for the
``lichtblick`` tool. The CLI (``npa workbench lichtblick ...``) and SDK
(``npa.sdk.workbench.lichtblick``) both call into these functions.

What it does that is tangible:

- ``build_mcap_from_frames`` turns a sequence of PNG/JPEG robot camera frames
  (e.g. the Sim2Real ``augment/frames`` artifacts, or any PNG/JPEG rollout camera
  export) into a real MCAP of ``foxglove.CompressedImage`` messages that a
  Foxglove-compatible viewer can play back. Raw ``.ppm`` rollout frames (and other
  PIL-readable formats) are transcoded to PNG bytes first via ``pillow`` so
  genuine rollout cameras render.
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

import logging
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
# Camera-frame image types a browser can decode directly as a CompressedImage;
# these are packed byte-for-byte with no re-encoding.
NATIVE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# Additional PIL-readable camera-frame types that browsers cannot decode as a
# CompressedImage (e.g. the Sim2Real rollout raw ``.ppm`` dumps). The exporter
# transcodes these to PNG bytes before packing so genuine rollout cameras render.
CONVERTIBLE_IMAGE_SUFFIXES = (".ppm", ".pgm", ".pnm", ".bmp", ".webp", ".tif", ".tiff", ".gif")
# Camera-frame image types the frames->MCAP exporter accepts.
IMAGE_SUFFIXES = NATIVE_IMAGE_SUFFIXES + CONVERTIBLE_IMAGE_SUFFIXES
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


def encode_frame_to_compressed_bytes(path: str) -> tuple[bytes, str]:
    """Return ``(payload, format)`` for a camera frame file, browser-decodable.

    PNG/JPEG frames are returned byte-for-byte. Any other PIL-readable frame
    (e.g. the Sim2Real rollout raw ``.ppm`` dumps) is transcoded to PNG bytes,
    because a browser cannot decode those directly inside a CompressedImage.
    """

    lower = path.lower()
    if lower.endswith(".png"):
        with open(path, "rb") as handle:
            return handle.read(), "png"
    if lower.endswith((".jpg", ".jpeg")):
        with open(path, "rb") as handle:
            return handle.read(), "jpeg"
    import io

    from PIL import Image

    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), "png"


def compressed_image_message(
    payload: bytes, *, fmt: str, stamp_ns: int, frame_id: str
) -> dict[str, Any]:
    """Build a ``foxglove.CompressedImage`` JSON message (base64 ``data``)."""

    import base64

    return {
        "timestamp": {"sec": stamp_ns // 1_000_000_000, "nsec": stamp_ns % 1_000_000_000},
        "frame_id": frame_id,
        "data": base64.b64encode(payload).decode("ascii"),
        "format": fmt,
    }


# foxglove.PointCloud well-known schema. Lichtblick / Foxglove render this in the
# GPU-accelerated 3D panel. ``data`` is a base64 packed byte buffer; ``fields``
# describe the per-point layout (x/y/z FLOAT32 + red/green/blue/alpha UINT8 here).
_POINTCLOUD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "foxglove.PointCloud",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "frame_id": {"type": "string"},
        "pose": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                },
                "orientation": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "w": {"type": "number"},
                    },
                },
            },
        },
        "point_stride": {"type": "integer"},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "offset": {"type": "integer"},
                    "type": {"type": "integer"},
                },
            },
        },
        "data": {"type": "string", "contentEncoding": "base64"},
    },
}

# foxglove PackedElementField NumericType values used below.
_FOXGLOVE_UINT8 = 1
_FOXGLOVE_FLOAT32 = 7
# Packed layout: x,y,z float32 (12 bytes) + red,green,blue,alpha uint8 (4 bytes).
#
# ``alpha`` is required for the cloud to keep its own colours: the viewer only
# offers its ``rgba-fields`` colour mode when ALL FOUR of red/green/blue/alpha are
# declared. Without it that mode is unavailable, and the 3D panel silently falls
# back to a synthetic colormap (turbo) — verified on a deployed agent: the same
# reconstruction rendered in yellow/red/blue ramp colours instead of the captured
# RGB. The cloud is still drawn, so this is fidelity, not visibility.
_POINTCLOUD_ALPHA_OPAQUE = 255
_POINTCLOUD_POINT_STRIDE = 16
_POINTCLOUD_FIELDS = [
    {"name": "x", "offset": 0, "type": _FOXGLOVE_FLOAT32},
    {"name": "y", "offset": 4, "type": _FOXGLOVE_FLOAT32},
    {"name": "z", "offset": 8, "type": _FOXGLOVE_FLOAT32},
    {"name": "red", "offset": 12, "type": _FOXGLOVE_UINT8},
    {"name": "green", "offset": 13, "type": _FOXGLOVE_UINT8},
    {"name": "blue", "offset": 14, "type": _FOXGLOVE_UINT8},
    {"name": "alpha", "offset": 15, "type": _FOXGLOVE_UINT8},
]


def pack_pointcloud_bytes(points: Any, colors: Any) -> bytes:
    """Pack ``(N,3)`` float xyz + ``(N,3)`` uint8 rgb into foxglove point bytes.

    Emits a fully-opaque ``alpha`` channel per point so the viewer's
    ``rgba-fields`` colour mode is available and the points keep their captured
    RGB instead of being re-coloured by a fallback colormap.
    """

    import numpy as np

    xyz = np.ascontiguousarray(np.asarray(points, dtype="<f4").reshape(-1, 3))
    rgb = np.ascontiguousarray(np.asarray(colors, dtype=np.uint8).reshape(-1, 3))
    count = min(xyz.shape[0], rgb.shape[0])
    buffer = np.zeros((count, _POINTCLOUD_POINT_STRIDE), dtype=np.uint8)
    buffer[:, 0:12] = xyz[:count].view(np.uint8).reshape(count, 12)
    buffer[:, 12:15] = rgb[:count]
    buffer[:, 15] = _POINTCLOUD_ALPHA_OPAQUE
    return buffer.tobytes()


def pointcloud_message(
    points: Any, colors: Any, *, stamp_ns: int, frame_id: str
) -> dict[str, Any]:
    """Build a ``foxglove.PointCloud`` JSON message (identity pose, RGBA points)."""

    import base64

    payload = pack_pointcloud_bytes(points, colors)
    return {
        "timestamp": {"sec": stamp_ns // 1_000_000_000, "nsec": stamp_ns % 1_000_000_000},
        "frame_id": frame_id,
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "point_stride": _POINTCLOUD_POINT_STRIDE,
        "fields": list(_POINTCLOUD_FIELDS),
        "data": base64.b64encode(payload).decode("ascii"),
    }


# foxglove.FrameTransform well-known schema. A viewer's 3D panel needs a
# coordinate frame to place a PointCloud; without a transform defining the frame
# referenced by the point cloud, the 3D panel renders nothing. We emit a single
# static transform so the sim2real frame is well-defined and the cloud renders.
_FRAME_TRANSFORM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "foxglove.FrameTransform",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "parent_frame_id": {"type": "string"},
        "child_frame_id": {"type": "string"},
        "translation": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "rotation": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"},
            },
        },
    },
}


def frame_transform_message(
    *,
    parent_frame_id: str,
    child_frame_id: str,
    stamp_ns: int,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Build a ``foxglove.FrameTransform`` JSON message (identity by default)."""

    tx, ty, tz = translation
    rx, ry, rz, rw = rotation
    return {
        "timestamp": {"sec": stamp_ns // 1_000_000_000, "nsec": stamp_ns % 1_000_000_000},
        "parent_frame_id": parent_frame_id,
        "child_frame_id": child_frame_id,
        "translation": {"x": float(tx), "y": float(ty), "z": float(tz)},
        "rotation": {"x": float(rx), "y": float(ry), "z": float(rz), "w": float(rw)},
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

    import json

    from mcap.writer import CompressionType, Writer

    if not frame_paths:
        raise LichtblickError("no camera frames found to export to MCAP.")
    if fps <= 0:
        raise LichtblickError(f"--fps must be > 0, got {fps}.")
    period_ns = int(1_000_000_000 / fps)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "wb") as handle:
        # Uncompressed chunks: valid MCAP that needs no lz4/zstandard C-extension,
        # so the exporter works in minimal environments.
        writer = Writer(handle, compression=CompressionType.NONE)
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
            payload, fmt = encode_frame_to_compressed_bytes(path)
            stamp = index * period_ns
            message = compressed_image_message(
                payload, fmt=fmt, stamp_ns=stamp, frame_id=frame_id
            )
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


# --------------------------------------------------------------------------- #
# MCAP -> Rerun: decode foxglove/JSON messages into native Rerun archetypes
# --------------------------------------------------------------------------- #
_MCAP_RERUN_TIMELINE = "mcap_time"


def _import_rerun_sdk() -> Any:
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - exercised via injected fake
        raise LichtblickError(
            "rerun-sdk is not installed; cannot convert MCAP to a Rerun .rrd "
            "(install the 'viz' extra: pip install 'npa[viz]')."
        ) from exc
    return rr


def _rr_set_time(rr: Any, recording: Any, seconds: float) -> None:
    for attempt in (
        lambda: rr.set_time(_MCAP_RERUN_TIMELINE, timestamp=seconds, recording=recording),
        lambda: rr.set_time_seconds(_MCAP_RERUN_TIMELINE, seconds, recording=recording),
        lambda: rr.set_time(_MCAP_RERUN_TIMELINE, duration=seconds, recording=recording),
    ):
        try:
            attempt()
            return
        except Exception:  # noqa: BLE001 - try the next SDK-version signature
            continue


def _rr_scalar(rr: Any, value: float) -> Any:
    if hasattr(rr, "Scalars"):
        return rr.Scalars(value)
    return rr.Scalar(value)


def build_rerun_rrd_from_mcap(
    mcap_path: str,
    output_rrd: str,
    *,
    rr: Any | None = None,
    application_id: str = "npa_mcap_to_rerun",
) -> dict[str, Any]:
    """Decode a foxglove/JSON MCAP into a native Rerun ``.rrd`` recording.

    Rerun's built-in MCAP loader keeps JSON-encoded foxglove messages as raw blobs
    (its decoders target ROS2/protobuf), so it opens the file but does not render
    our camera/critique/signal streams. This converter maps our well-known schemas
    to native Rerun archetypes so the same MCAP renders with full fidelity:

    - ``foxglove.CompressedImage`` -> ``rr.EncodedImage`` (Spatial2D image panel),
    - ``foxglove.Log`` -> ``rr.TextLog`` (text-log panel),
    - any JSON message with a numeric ``value`` -> ``rr.Scalars`` (time-series),
    - anything else -> ``rr.TextDocument`` (raw JSON), so nothing is dropped.

    ``rr`` may be injected for testing. Returns a summary dict.
    """

    import base64
    import json as _json

    if not str(mcap_path).strip():
        raise LichtblickError("mcap_path is required.")
    if not str(output_rrd).lower().endswith(".rrd"):
        raise LichtblickError(f"output path must end in .rrd, got: {output_rrd}")

    from mcap.reader import make_reader

    rr = rr or _import_rerun_sdk()
    os.makedirs(os.path.dirname(os.path.abspath(output_rrd)) or ".", exist_ok=True)

    recording = rr.RecordingStream(application_id) if hasattr(rr, "RecordingStream") else None
    rr.save(output_rrd, recording=recording)

    counts = {"images": 0, "scalars": 0, "logs": 0, "other": 0}
    with open(mcap_path, "rb") as handle:
        reader = make_reader(handle)
        for schema, channel, message in reader.iter_messages():
            topic = (channel.topic or "mcap").strip("/") or "mcap"
            schema_name = (schema.name if schema is not None else "") or ""
            _rr_set_time(rr, recording, message.log_time / 1_000_000_000)
            try:
                payload = _json.loads(message.data)
            except (ValueError, TypeError):
                continue
            if schema_name == "foxglove.CompressedImage":
                raw = base64.b64decode(str(payload.get("data", "")) or "")
                fmt = str(payload.get("format", "png")).lower()
                media_type = "image/jpeg" if fmt in ("jpeg", "jpg") else "image/png"
                rr.log(topic, rr.EncodedImage(contents=raw, media_type=media_type), recording=recording)
                counts["images"] += 1
            elif schema_name == "foxglove.Log":
                rr.log(topic, rr.TextLog(str(payload.get("message", ""))), recording=recording)
                counts["logs"] += 1
            elif isinstance(payload, dict) and isinstance(payload.get("value"), (int, float)):
                rr.log(topic, _rr_scalar(rr, float(payload["value"])), recording=recording)
                counts["scalars"] += 1
            else:
                rr.log(
                    topic,
                    rr.TextDocument(_json.dumps(payload)[:4000]),
                    recording=recording,
                )
                counts["other"] += 1

    disconnect = getattr(rr, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect(recording=recording)
        except Exception:  # noqa: BLE001 - best-effort flush
            logging.getLogger(__name__).debug("rerun disconnect failed", exc_info=True)

    total = sum(counts.values())
    if total == 0:
        raise LichtblickError(f"no decodable messages found in MCAP: {mcap_path}")
    return {
        "output_rrd_path": output_rrd,
        "message_count": total,
        "image_count": counts["images"],
        "scalar_count": counts["scalars"],
        "log_count": counts["logs"],
        "other_count": counts["other"],
    }


def convert_mcap_to_rerun(
    input_path: str,
    output_rrd: str,
    *,
    workdir: str = "",
    s3_client: Any | None = None,
    rr: Any | None = None,
) -> dict[str, Any]:
    """Stage an MCAP (S3 or local) and decode it into a native Rerun ``.rrd``.

    Downloads ``input_path`` from S3 when needed (via npa-managed credentials),
    then calls :func:`build_rerun_rrd_from_mcap`. Returns the converter summary.
    """

    value = (input_path or "").strip()
    if not value:
        raise LichtblickError("--input-path is required.")
    if value.startswith("s3://"):
        work = workdir or tempfile.mkdtemp(prefix="npa-mcap2rrd-")
        os.makedirs(work, exist_ok=True)
        client = s3_client or _default_s3_client()
        bucket, key = _split_s3(value)
        local_mcap = os.path.join(work, PurePosixPath(key).name or "input.mcap")
        client.download_file(bucket, key, local_mcap)
    else:
        if not os.path.isfile(value):
            raise LichtblickError(f"local MCAP not found: {value}")
        local_mcap = value
    summary = build_rerun_rrd_from_mcap(local_mcap, output_rrd, rr=rr)
    summary["input_path"] = value
    return summary


__all__ = [
    "CONTAINER_PORT",
    "CONVERTIBLE_IMAGE_SUFFIXES",
    "DEFAULT_CAMERA_TOPIC",
    "DEFAULT_FPS",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "IMAGE_SUFFIXES",
    "NATIVE_IMAGE_SUFFIXES",
    "LichtblickError",
    "LichtblickLaunchPlan",
    "SUPPORTED_SUFFIXES",
    "build_launch_plan",
    "build_mcap_from_frames",
    "build_rerun_rrd_from_mcap",
    "compressed_image_message",
    "convert_mcap_to_rerun",
    "encode_frame_to_compressed_bytes",
    "frame_transform_message",
    "launch_viewer",
    "serve_viewer",
    "stage_input_to_mcap",
]
