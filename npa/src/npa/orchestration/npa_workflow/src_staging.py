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
tree to a content-addressed ``s3://<bucket>/<prefix>/<sha256>/`` prefix.  A
manifest written last is the commit marker, so an interrupted upload is never
mistaken for a reusable source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Iterable

DEFAULT_SRC_PREFIX = "npa-src/npa"
SOURCE_MANIFEST_NAME = ".npa-source.json"
SOURCE_MANIFEST_SCHEMA = "npa.source.v1"

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

#: Suffixes and names that must never reach a bucket or a worker, even when a
#: secret was accidentally committed to the git index.
SENSITIVE_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".tfstate",
    ".tfstate.backup",
    ".tfvars",
    ".kubeconfig",
)
SENSITIVE_NAMES = frozenset(
    {
        ".env",
        "auth.env",
        "credentials.yaml",
        "credentials.yml",
        "credentials.json",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)
SENSITIVE_DIR_NAMES = frozenset({".aws", ".kube"})


class SrcStagingError(RuntimeError):
    """Raised when the npa package source cannot be located or uploaded."""


@dataclass(frozen=True)
class SourceStageResult:
    """A verified immutable source upload."""

    uri: str
    fingerprint: str
    file_count: int
    reused: bool


def find_npa_package_root(start: Path | None = None) -> Path:
    """Return the ``npa`` package root (the directory holding ``pyproject.toml``).

    Walks up from this module so it works from a source checkout and from an
    editable install. Raises :class:`SrcStagingError` for a non-editable install,
    where there is no package tree to upload.
    """

    here = (start or Path(__file__)).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "npa"
        ).is_dir():
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


def _is_sensitive(relative: Path) -> bool:
    lowered = Path(*(part.casefold() for part in relative.parts))
    if SENSITIVE_DIR_NAMES & set(lowered.parts):
        return True
    if lowered.name in SENSITIVE_NAMES or lowered.name.startswith(".env."):
        return True
    return lowered.name.endswith(SENSITIVE_SUFFIXES)


def _is_safe_regular_source(root: Path, relative: Path) -> bool:
    """Apply one normalized filter to git-index and filesystem-walk candidates."""

    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return False
    normalized = Path(relative.as_posix())
    if normalized != relative or _is_excluded(normalized) or _is_sensitive(normalized):
        return False
    candidate = root / normalized
    current = root
    try:
        for part in normalized.parts:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                return False
        return stat.S_ISREG(candidate.lstat().st_mode)
    except OSError:
        return False


def _git_source_files(root: Path) -> list[Path] | None:
    """Return tracked plus non-ignored untracked source under *root*.

    Live validation and development runs must execute the preserved dirty tree,
    including newly added modules that have not been staged in git yet.  Git's
    exclude machinery remains the authority for local state, and the explicit
    sensitive-file filter is applied again below.
    """

    import subprocess

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    source_files = [Path(name) for name in result.stdout.split("\0") if name]
    return source_files or None


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield the package files worth uploading, relative to *root*.

    Prefers git's tracked plus non-ignored-untracked view. This includes newly
    created dirty implementation files while excluding local state covered by
    ``.gitignore``; explicit sensitive-name filtering is applied independently.
    The directory walk remains the fallback outside a checkout.
    """

    git_files = _git_source_files(root)
    if git_files is not None:
        for relative in sorted(git_files):
            if _is_safe_regular_source(root, relative):
                yield relative
        return

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _is_safe_regular_source(root, relative):
            yield relative


def source_fingerprint(root: Path, files: Iterable[Path] | None = None) -> str:
    """Return a stable digest of paths, modes, and contents in the source tree."""

    digest = hashlib.sha256()
    for relative in files if files is not None else iter_source_files(root):
        path = root / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"x" if os.access(path, os.X_OK) else b"-")
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    value = str(uri or "").strip()
    if not value.startswith("s3://"):
        raise SrcStagingError(f"Expected an s3:// source URI, got {value or '<empty>'}")
    bucket_and_key = value.removeprefix("s3://").split("/", 1)
    if len(bucket_and_key) != 2 or not all(bucket_and_key):
        raise SrcStagingError(f"Expected s3://<bucket>/<prefix>, got {value}")
    return bucket_and_key[0], bucket_and_key[1].rstrip("/")


def verify_staged_source(
    uri: str,
    *,
    client: Any,
    expected_fingerprint: str = "",
) -> dict[str, Any]:
    """Read and validate the upload's commit manifest.

    This intentionally verifies through S3 rather than trusting a local ledger:
    a deleted object, wrong endpoint, or revoked permission must be actionable
    before SkyPilot starts workers.
    """

    bucket, prefix = _parse_s3_uri(uri)
    key = f"{prefix}/{SOURCE_MANIFEST_NAME}"
    try:
        response = client.s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - include the exact provider failure
        raise SrcStagingError(
            f"Source verification failed for {uri}: could not read s3://{bucket}/{key}: "
            f"{exc}. Safely restage with `npa workbench workflow stage-src "
            f"--bucket {bucket}`."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA
    ):
        raise SrcStagingError(
            f"Source verification failed for {uri}: {SOURCE_MANIFEST_NAME} has an "
            f"unsupported schema. Safely restage with `npa workbench workflow "
            f"stage-src --bucket {bucket}`."
        )
    actual = str(payload.get("fingerprint", ""))
    if expected_fingerprint and actual != expected_fingerprint:
        raise SrcStagingError(
            f"Source verification failed for {uri}: expected fingerprint "
            f"{expected_fingerprint}, found {actual or '<missing>'}. Safely restage with "
            f"`npa workbench workflow stage-src --bucket {bucket}`."
        )
    return payload


def _storage_client(
    *,
    endpoint_url: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
):
    """Build a StorageClient from explicit values, env, then ``~/.npa``.

    ``StorageClient.from_environment`` only looks at AWS_* env vars, so staging
    failed with "Unable to locate credentials" on a machine that was fully
    configured through ``npa configure``. Fall back to the saved credentials the
    rest of the CLI uses.
    """

    from npa.clients.credentials import load_credentials
    from npa.clients.storage import StorageClient

    endpoint = endpoint_url.strip()
    key = aws_access_key_id.strip()
    secret = aws_secret_access_key.strip()
    if not (endpoint and key and secret):
        try:
            saved = load_credentials()
        except Exception:  # noqa: BLE001 - staging must not require ~/.npa
            saved = None
        if saved is not None:
            endpoint = endpoint or saved.s3_endpoint
            key = key or saved.s3_access_key_id
            secret = secret or saved.s3_secret_access_key
    return StorageClient.from_environment(
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def ensure_npa_source(
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
    force: bool = False,
) -> SourceStageResult:
    """Return a verified source prefix, uploading it exactly once when absent.

    *bucket* accepts ``name`` or ``s3://name`` (any trailing path is ignored in
    favor of *prefix*, so the destination is predictable).
    """

    bucket_name = (
        str(bucket or "").strip().removeprefix("s3://").strip("/").split("/", 1)[0]
    )
    if not bucket_name:
        raise SrcStagingError(
            "A bucket is required to stage the npa source. Pass --bucket "
            "<your-bucket> (or --var bucket=<your-bucket> on submit)."
        )
    key_prefix = str(prefix or DEFAULT_SRC_PREFIX).strip("/") or DEFAULT_SRC_PREFIX

    root = source_root or find_npa_package_root()
    if not (root / "pyproject.toml").is_file():
        raise SrcStagingError(
            f"{root} does not look like the npa package (no pyproject.toml)"
        )

    files = list(iter_source_files(root))
    if not files:
        raise SrcStagingError(f"No source files found under {root}")
    fingerprint = source_fingerprint(root, files)
    destination = f"s3://{bucket_name}/{key_prefix}/{fingerprint}/"
    if client is None:
        try:
            client = _storage_client(
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as SrcStagingError below
            raise SrcStagingError(
                f"Cannot reach object storage to stage the npa source to {destination}: "
                f"{exc}. Set the S3 endpoint and keys with `npa configure` (or pass "
                "--s3-endpoint / AWS_* env vars)."
            ) from exc

    if not force:
        try:
            verify_staged_source(
                destination,
                client=client,
                expected_fingerprint=fingerprint,
            )
        except SrcStagingError as exc:
            if on_status:
                on_status(f"source cache miss ({exc}); staging {root} -> {destination}")
        else:
            if on_status:
                on_status(f"reusing verified staged source {destination}")
            return SourceStageResult(
                uri=destination,
                fingerprint=fingerprint,
                file_count=len(files),
                reused=True,
            )
    elif on_status:
        on_status(f"force-restaging {root} -> {destination}")

    # ~950 small objects: serial PUTs make this a minute-plus of latency, so
    # upload with a small pool. Order does not matter (the worker syncs the
    # whole prefix before installing).
    def _upload(relative: Path) -> None:
        client.upload_file(str(root / relative), f"{destination}{relative.as_posix()}")

    # Both callers only handle SrcStagingError; an unconfigured bucket, expired
    # keys or AccessDenied would otherwise surface as a bare "Unexpected error"
    # with none of the staging guidance.
    try:
        if max_workers and max_workers > 1 and len(files) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(max_workers, len(files))) as pool:
                for _ in pool.map(_upload, files):
                    pass
        else:
            for relative in files:
                _upload(relative)
        manifest = {
            "schema_version": SOURCE_MANIFEST_SCHEMA,
            "fingerprint": fingerprint,
            "file_count": len(files),
        }
        descriptor, raw_manifest = tempfile.mkstemp(
            prefix="npa-source-", suffix=".json"
        )
        manifest_path = Path(raw_manifest)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True)
                handle.write("\n")
            client.upload_file(
                str(manifest_path), f"{destination}{SOURCE_MANIFEST_NAME}"
            )
        finally:
            manifest_path.unlink(missing_ok=True)
        verify_staged_source(
            destination,
            client=client,
            expected_fingerprint=fingerprint,
        )
    except Exception as exc:  # noqa: BLE001 - normalized above for callers
        raise SrcStagingError(
            f"Failed to stage the npa source to {destination}: {exc}. The incomplete "
            "content-addressed prefix is safe to retry; no workflow will use it without "
            f"{SOURCE_MANIFEST_NAME}."
        ) from exc
    uploaded = len(files)
    if on_status:
        on_status(f"staged {uploaded} files")
    return SourceStageResult(
        uri=destination,
        fingerprint=fingerprint,
        file_count=uploaded,
        reused=False,
    )


def stage_npa_source(**kwargs: Any) -> str:
    """Compatibility wrapper returning only the exact staged source URI."""

    return ensure_npa_source(**kwargs).uri


def resolve_src_uri_from_env() -> str:
    """Return the configured npa-source URI from the environment, or ""."""

    return (
        os.environ.get("NPA_SRC_S3_URI", "")
        or os.environ.get("NPA_E2E_NPA_SRC_S3_URI", "")
    ).strip()
