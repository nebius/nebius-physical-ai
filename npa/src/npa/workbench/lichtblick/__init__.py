"""Lichtblick web-viewer launch helpers for Workbench.

Lichtblick is an open-source, Foxglove-compatible MCAP / ROS-bag / robotics log
viewer (MPL-2.0). This module is the single source of truth for the
``lichtblick`` tool. The CLI (``npa workbench lichtblick ...``) and SDK
(``npa.sdk.workbench.lichtblick``) both call into these functions; no launch
logic is duplicated across access paths.

The viewer is a static web app that opens log artifacts staged from S3. Cross-tool
data flows through S3 only, so every command takes ``--input-path`` /
``--output-path`` S3 (or local) URIs.

This module is import-safe: it pulls in no GPU, Docker, or network dependencies
at import time, so it is usable from unit tests and the CLI alike. Actually
launching a container is opt-in and goes through an injected runner.
"""

from __future__ import annotations

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
# Recognized robotics log containers the viewer can open.
SUPPORTED_SUFFIXES = (".mcap", ".bag", ".db3")


class LichtblickError(ValueError):
    """Raised when a Lichtblick viewer launch request is invalid."""


@dataclass(frozen=True)
class LichtblickLaunchPlan:
    """A fully-resolved, infra-free description of how to launch the viewer."""

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        without_scheme = value[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            raise LichtblickError(f"invalid s3 input path: {value}")
        name = PurePosixPath(key).name
    else:
        name = PurePosixPath(value).name
    if not name:
        raise LichtblickError(f"could not derive an artifact name from {value!r}.")
    if PurePosixPath(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(SUPPORTED_SUFFIXES)
        raise LichtblickError(
            f"unsupported artifact {name!r}; Lichtblick opens {supported} logs."
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
) -> LichtblickLaunchPlan:
    """Return a validated, infra-free plan for serving ``input_path`` in Lichtblick.

    The plan resolves the ``npa-lichtblick`` image, the container run command, and
    a deep-linked viewer URL that opens the served artifact. Nothing is executed;
    :func:`launch_viewer` performs the actual (opt-in) container run.
    """

    name = _validate_artifact(input_path)
    resolved_host = (host or DEFAULT_HOST).strip() or DEFAULT_HOST
    if port <= 0 or port > 65535:
        raise LichtblickError(f"--port must be in 1..65535, got {port}.")
    resolved_image = _resolve_image(image, registry=registry, tag=tag)
    served_artifact_path = f"{SERVED_DATA_DIR}/{name}"

    # Lichtblick opens a remote file via the ds=remote-file data source (the same
    # deep-link scheme Foxglove-compatible viewers accept).
    served_url = f"http://{resolved_host}:{port}/data/{name}"
    viewer_url = (
        f"http://{resolved_host}:{port}/?ds=remote-file"
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
        staged=False,
    )


def launch_viewer(
    plan: LichtblickLaunchPlan,
    *,
    local_artifact: str,
    runner: Callable[[list[str]], Any],
) -> LichtblickLaunchPlan:
    """Execute the viewer container via an injected ``runner`` (opt-in).

    ``runner`` receives the argv list (typically ``subprocess.run``). Injecting it
    keeps unit tests infra-free: they pass a fake runner and assert on the argv.
    """

    artifact = (local_artifact or "").strip()
    if not artifact:
        raise LichtblickError("local_artifact is required to launch the viewer.")
    argv = [
        "docker",
        "run",
        "--rm",
        "-p",
        f"{plan.host}:{plan.port}:{CONTAINER_PORT}",
        "-v",
        f"{artifact}:{plan.served_artifact_path}:ro",
        plan.image,
    ]
    runner(argv)
    return LichtblickLaunchPlan(**{**plan.to_dict(), "status": "launched", "staged": True})


__all__ = [
    "CONTAINER_PORT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LichtblickError",
    "LichtblickLaunchPlan",
    "SUPPORTED_SUFFIXES",
    "build_launch_plan",
    "launch_viewer",
]
