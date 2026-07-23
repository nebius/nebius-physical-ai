"""Foxglove (Lichtblick OSS) web-viewer launch helpers for Workbench.

Single source of truth for the ``foxglove`` tool. The CLI
(``npa workbench foxglove ...``) and SDK (``npa.sdk.workbench.foxglove``) both
call into these functions; no launch logic is duplicated across access paths.

The viewer is a static web app (Lichtblick, MPL-2.0) that opens MCAP / ROS-bag /
robotics log artifacts staged from S3. Cross-tool data flows through S3 only, so
every command takes ``--input-path`` / ``--output-path`` S3 (or local) URIs.

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


class FoxgloveError(ValueError):
    """Raised when a Foxglove viewer launch request is invalid."""


@dataclass(frozen=True)
class FoxgloveLaunchPlan:
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
        raise FoxgloveError("--input-path is required (S3 or local MCAP/bag artifact).")
    if "://" in value and not value.startswith("s3://"):
        scheme = value.split("://", 1)[0]
        raise FoxgloveError(
            f"--input-path must be an s3:// URI or a local path, got scheme {scheme!r}."
        )
    if value.startswith("s3://"):
        without_scheme = value[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            raise FoxgloveError(f"invalid s3 input path: {value}")
        name = PurePosixPath(key).name
    else:
        name = PurePosixPath(value).name
    if not name:
        raise FoxgloveError(f"could not derive an artifact name from {value!r}.")
    if PurePosixPath(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(SUPPORTED_SUFFIXES)
        raise FoxgloveError(
            f"unsupported artifact {name!r}; Foxglove opens {supported} logs."
        )
    return name


def _resolve_image(image: str, *, registry: str | None, tag: str | None) -> str:
    if image.strip():
        return image.strip()
    # Imported lazily so unit tests importing this module never require the
    # deploy stack or registry resolution.
    from npa.deploy.images import container_image_for_tool

    return container_image_for_tool("foxglove", registry=registry, tag=tag)


def build_launch_plan(
    *,
    input_path: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    image: str = "",
    output_path: str = "",
    registry: str | None = None,
    tag: str | None = None,
) -> FoxgloveLaunchPlan:
    """Return a validated, infra-free plan for serving ``input_path`` in Foxglove.

    The plan resolves the ``npa-foxglove`` image, the container run command, and a
    deep-linked viewer URL that opens the served artifact. Nothing is executed;
    :func:`launch_viewer` performs the actual (opt-in) container run.
    """

    name = _validate_artifact(input_path)
    resolved_host = (host or DEFAULT_HOST).strip() or DEFAULT_HOST
    if port <= 0 or port > 65535:
        raise FoxgloveError(f"--port must be in 1..65535, got {port}.")
    resolved_image = _resolve_image(image, registry=registry, tag=tag)
    served_artifact_path = f"{SERVED_DATA_DIR}/{name}"

    # Foxglove/Lichtblick opens a remote file via the ds=remote-file data source.
    served_url = f"http://{resolved_host}:{port}/data/{name}"
    viewer_url = (
        f"http://{resolved_host}:{port}/?ds=remote-file"
        f"&ds.url={quote(served_url, safe='')}"
    )

    docker_command = (
        f"docker run --rm -p {resolved_host}:{port}:{CONTAINER_PORT} "
        f"-v <local-artifact>:{served_artifact_path}:ro {resolved_image}"
    )

    return FoxgloveLaunchPlan(
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
    plan: FoxgloveLaunchPlan,
    *,
    local_artifact: str,
    runner: Callable[[list[str]], Any],
) -> FoxgloveLaunchPlan:
    """Execute the viewer container via an injected ``runner`` (opt-in).

    ``runner`` receives the argv list (typically ``subprocess.run``). Injecting it
    keeps unit tests infra-free: they pass a fake runner and assert on the argv.
    """

    artifact = (local_artifact or "").strip()
    if not artifact:
        raise FoxgloveError("local_artifact is required to launch the viewer.")
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
    return FoxgloveLaunchPlan(**{**plan.to_dict(), "status": "launched", "staged": True})


__all__ = [
    "CONTAINER_PORT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "FoxgloveError",
    "FoxgloveLaunchPlan",
    "SUPPORTED_SUFFIXES",
    "build_launch_plan",
    "launch_viewer",
]
