"""Stage the ``npa`` package source to S3 for image-less SkyPilot steps.

Tasks that run on SkyPilot's default image (Token Factory tools, ``run.shell``
states) have no baked ``npa``. Their generated setup script installs it by
syncing ``$NPA_SRC_S3_URI`` into ``/tmp/npa-src`` and running ``pip install -e``
on it — SkyPilot's own ``file_mounts`` cannot be used here because they create
new buckets, which fails on Nebius.

Nothing shipped that copy, so a first submit dead-ended on::

    planned step 'annotate-original' has no workbench image and NPA_SRC_S3_URI
    is unset

with no command to produce one. :func:`stage_npa_source` uploads the package
tree to ``s3://<bucket>/<prefix>/`` and returns the URI to export.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_SRC_PREFIX = "npa-src/npa"

#: Directory names never worth uploading (virtualenvs, caches, build output).
EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".terraform",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

#: File suffixes that are build artifacts rather than source.
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".log")


class SrcStagingError(RuntimeError):
    """Raised when the npa package source cannot be located or uploaded."""


def find_npa_package_root(start: Path | None = None) -> Path:
    """Return the ``npa`` package root (the directory holding ``pyproject.toml``).

    Walks up from this module so it works from a source checkout and from an
    editable install. Raises :class:`SrcStagingError` for a non-editable install,
    where there is no package tree to upload.
    """

    here = (start or Path(__file__)).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "npa").is_dir():
            return candidate
    raise SrcStagingError(
        "Could not locate the npa package source (no pyproject.toml with src/npa "
        "above this module). Staging needs a source checkout or an editable "
        "install; otherwise pass --image <registry>/npa-<tool>:<tag> instead."
    )


def _is_excluded(relative: Path) -> bool:
    if EXCLUDED_DIR_NAMES & set(relative.parts):
        return True
    if any(part.endswith(".egg-info") for part in relative.parts):
        return True
    return relative.name.endswith(EXCLUDED_SUFFIXES)


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield the package files worth uploading, relative to *root*."""

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_excluded(relative):
            continue
        yield relative


def stage_npa_source(
    *,
    bucket: str,
    prefix: str = DEFAULT_SRC_PREFIX,
    endpoint_url: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    source_root: Path | None = None,
    client: Any | None = None,
    on_status: Callable[[str], None] | None = None,
    max_workers: int = 8,
) -> str:
    """Upload the npa package tree and return the ``NPA_SRC_S3_URI`` to use.

    *bucket* accepts ``name`` or ``s3://name`` (any trailing path is ignored in
    favor of *prefix*, so the destination is predictable).
    """

    bucket_name = str(bucket or "").strip().removeprefix("s3://").strip("/").split("/", 1)[0]
    if not bucket_name:
        raise SrcStagingError(
            "A bucket is required to stage the npa source. Pass --bucket "
            "<your-bucket> (or --var bucket=<your-bucket> on submit)."
        )
    key_prefix = str(prefix or DEFAULT_SRC_PREFIX).strip("/") or DEFAULT_SRC_PREFIX

    root = source_root or find_npa_package_root()
    if not (root / "pyproject.toml").is_file():
        raise SrcStagingError(f"{root} does not look like the npa package (no pyproject.toml)")

    if client is None:
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment(
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    destination = f"s3://{bucket_name}/{key_prefix}/"
    if on_status:
        on_status(f"staging {root} -> {destination}")

    files = list(iter_source_files(root))
    if not files:
        raise SrcStagingError(f"No source files found under {root}")

    # ~950 small objects: serial PUTs make this a minute-plus of latency, so
    # upload with a small pool. Order does not matter (the worker syncs the
    # whole prefix before installing).
    def _upload(relative: Path) -> None:
        client.upload_file(str(root / relative), f"{destination}{relative.as_posix()}")

    if max_workers and max_workers > 1 and len(files) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(max_workers, len(files))) as pool:
            for _ in pool.map(_upload, files):
                pass
    else:
        for relative in files:
            _upload(relative)
    uploaded = len(files)
    if on_status:
        on_status(f"staged {uploaded} files")
    return destination


def resolve_src_uri_from_env() -> str:
    """Return the configured npa-source URI from the environment, or ""."""

    return (
        os.environ.get("NPA_SRC_S3_URI", "")
        or os.environ.get("NPA_E2E_NPA_SRC_S3_URI", "")
    ).strip()
