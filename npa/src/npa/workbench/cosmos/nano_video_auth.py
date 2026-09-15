"""Keep Ray administrative access separate from the video inference API."""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def require_management_auth() -> None:
    """Refuse an application builder on a Ray cluster without explicit auth."""
    if os.environ.get("RAY_AUTH_MODE") != "token":
        raise ValueError("Ray management authentication must be enabled before Ray starts")
    token = os.environ.get("RAY_AUTH_TOKEN", "").strip()
    if not token and (token_path := os.environ.get("RAY_AUTH_TOKEN_PATH")):
        fd = os.open(token_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ValueError("Ray management credential must be an owner-only regular file")
            token = stream.read().strip()
    if not token:
        raise ValueError("An explicit Ray management credential is required")
    if token == os.environ.get("NPA_COSMOS3_VIDEO_TOKEN", "").strip():
        raise ValueError("Ray management and inference credentials must be different")


@contextmanager
def local_management_auth() -> Iterator[None]:
    """Give a standalone Ray runtime a fresh private credential for its lifetime."""
    names = ("RAY_AUTH_MODE", "RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN_PATH")
    previous = {name: os.environ.get(name) for name in names}
    with tempfile.TemporaryDirectory(prefix="npa-video-ray-") as directory:
        path = Path(directory) / "management-token"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_hex(32))
        os.environ.pop("RAY_AUTH_TOKEN", None)
        os.environ["RAY_AUTH_MODE"] = "token"
        os.environ["RAY_AUTH_TOKEN_PATH"] = str(path)
        try:
            require_management_auth()
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def serve_local() -> None:
    """Start authenticated Ray locally; expose only the inference HTTP listener."""
    with local_management_auth():
        import ray
        from ray import serve

        from .nano_video_server import app

        try:
            ray.init(
                address="local", include_dashboard=False, _node_ip_address="127.0.0.1"
            )
            serve.start(http_options={"host": "0.0.0.0", "port": 8000})
            serve.run(app(), blocking=True)
        finally:
            ray.shutdown()
