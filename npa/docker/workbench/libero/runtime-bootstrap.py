#!/usr/bin/env python3
"""Materialize the pinned LIBERO runtime into an operator-owned cache.

The public image contains this fetcher and immutable manifests, not LIBERO,
PyTorch/CUDA wheels, demonstrations, task assets, models, or populated caches.
Download success is never treated as permission: ``ensure`` refuses before the
first cache mutation unless a manager-issued decision is present and hash-bound.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import grp
import hashlib
import hmac
import http.client
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

SCHEMA = "npa.libero.runtime-manifest.v1"
DECISION_SCHEMA = "npa.libero.runtime-use-decision.v2"
COMPLETE_SCHEMA = "npa.libero.runtime-cache.v1"
INVENTORY_SCHEMA = "npa.libero.runtime-cache-inventory.v1"
EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "boundaries",
        "demonstration",
        "governing_terms",
        "language_model",
        "runtime_artifact_count",
        "runtime_artifacts",
        "runtime_python",
        "runtime_use_decision",
        "schema",
        "solution",
        "source",
        "task",
    }
)
EXPECTED_RUNTIME_PYTHON = {
    "version": "3.10",
    "abi": "cp310",
    "platform": "linux_x86_64",
}
EXPECTED_RUNTIME_DECISION_METADATA = {
    "schema": DECISION_SCHEMA,
    "required": True,
    "accepted_by_download": False,
}
EXPECTED_BOUNDARIES = {
    "cache": "operator-owned non-root atomic runtime cache",
    "outputs": "NPA_SMOKE_OUTPUT_DIR only",
    "rendering": False,
}
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "2db4ca50fa3c324bf60eeb6a4f9d9ba9f36fc377a958ef3458e1bf97a51b83ea"
)
EXPECTED_RUNTIME_REQUIREMENTS_SHA256 = (
    "8504f236dcad67ad0e2f5959b916c93aa7ccbd02567c6e323e480366d0f23b99"
)
DEFAULT_MANIFEST = Path("/opt/npa/libero/runtime-manifest.json")
DEFAULT_REQUIREMENTS = Path("/opt/npa/libero/runtime-requirements.txt")
DEFAULT_CACHE = Path("/workspace/.cache/npa/libero")
MANAGER_ACCEPTANCE_PUBLIC_KEY = Path(
    "/opt/npa/libero/manager-acceptance-public-key.b64"
)
MANAGER_ACCEPTANCE_PUBLIC_KEY_OWNER_UID = 0
MANAGER_ACCEPTANCE_NAMESPACE = b"npa.libero.acceptance"
ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "files.pythonhosted.org",
        "download-r2.pytorch.org",
        "huggingface.co",
    }
)
ALLOWED_DOWNLOAD_REDIRECT_HOSTS = frozenset(
    {
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
        "cas-bridge.xethub.hf.co",
        "us.aws.cdn.hf.co",
        "us.gcp.cdn.hf.co",
    }
)
SEALED_DIRECTORY_MODE = stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
SEALED_EXECUTABLE_MODE = SEALED_DIRECTORY_MODE
SEALED_REGULAR_MODE = stat.S_IRUSR | stat.S_IRGRP
CACHE_ROOT_MODE = SEALED_DIRECTORY_MODE | stat.S_IWUSR
ALLOWED_TERMS_HOSTS = frozenset(
    {
        "raw.githubusercontent.com",
        "creativecommons.org",
        "www.apache.org",
        "docs.nvidia.com",
        "www.nvidia.com",
    }
)
EXPECTED_DECISION_BOUNDARIES = frozenset(
    {"source", "runtime_packages", "demonstration", "task_inputs", "language_model"}
)
EXPECTED_GOVERNING_TERMS = frozenset(
    {
        "libero-mit",
        "dataset-cc-by-4.0",
        "bert-apache-2.0",
        "pytorch-bsd",
        "cuda-eula",
        "nvidia-software-license",
        "cudnn-eula",
    }
)
MAX_RUNTIME_CACHE_DOWNLOAD_BYTES = 32 * 1024 * 1024 * 1024
RUNTIME_EXECUTION_GROUP = "npa-libero-exec"
STORAGE_SECRET_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
        "NPA_LIBERO_MANAGER_ACCEPTANCE_B64",
    }
)
RUNTIME_MATERIALIZATION_PASSTHROUGH_ENV_NAMES = frozenset(
    {
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "SOURCE_DATE_EPOCH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "TZ",
    }
)
RUNTIME_EXECUTION_PASSTHROUGH_ENV_NAMES = frozenset(
    {
        "BYOF_CAPABILITY_NAME",
        "BYOF_IMAGE",
        "BYOF_REPO_ROOT",
        "BYOF_SMOKE_ARTIFACT_NAME",
        "BYOF_SMOKE_COMMAND",
        "BYOF_SOLUTION_NAME",
        "CUDA_VISIBLE_DEVICES",
        "HOSTNAME",
        "KUBERNETES_SERVICE_HOST",
        "KUBERNETES_SERVICE_PORT",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "NPA_BYOF_RUN_ID",
        "NPA_LIBERO_EXPECTED_ACCEPTANCE_ID",
        "NPA_LIBERO_EXPECTED_ALLOWED_NODE_SHA256",
        "NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256",
        "NPA_LIBERO_EXPECTED_CLUSTER_IDENTITY_SHA256",
        "NPA_LIBERO_EXPECTED_EXECUTION_KUBECONFIG_SHA256",
        "NPA_LIBERO_EXPECTED_EXTERNAL_RBAC_INVENTORY_SHA256",
        "NPA_LIBERO_EXPECTED_INFRASTRUCTURE_BUNDLE_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_INVENTORY_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_UID_SHA256",
        "NPA_LIBERO_EXPECTED_PAYLOAD_KUBECONFIG_SHA256",
        "NPA_LIBERO_EXPECTED_PUBLICATION_BUNDLE_SHA256",
        "NPA_LIBERO_EXPECTED_RBAC_SPEC_SHA256",
        "NPA_LIBERO_EXPECTED_ROLE_BINDING_UID_SHA256",
        "NPA_LIBERO_EXPECTED_ROLE_UID_SHA256",
        "NPA_LIBERO_EXPECTED_SERVICE_ACCOUNT_UID_SHA256",
        "NPA_LIBERO_EXPECTED_SKYPILOT_CONFIG_SHA256",
        "NPA_LIBERO_BOOTSTRAP_RECEIPT",
        "NPA_LIBERO_RUNTIME_USE_DECISION_SHA256",
        "NPA_SMOKE_OUTPUT_DIR",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TZ",
    }
)
OUTPUT_SIZE_LIMITS = {
    "libero-bc-rnn-smoke.pth": 256 * 1024 * 1024,
    "libero-smoke.json": 8 * 1024 * 1024,
    "npa_byof_summary.json": 1024 * 1024,
    "npa_runtime_bootstrap.json": 8 * 1024 * 1024,
    "npa_runtime_metadata.json": 8 * 1024 * 1024,
    "nvidia_smi.txt": 1024 * 1024,
    "nvidia_smi_list.txt": 1024 * 1024,
    "solution_smoke_stderr.log": 16 * 1024 * 1024,
    "solution_smoke_stdout.log": 16 * 1024 * 1024,
}
MAX_OUTPUT_BYTES = 320 * 1024 * 1024


class BootstrapRefusal(RuntimeError):
    """A fail-closed identity, permission, or boundary refusal."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal(f"cannot read required JSON {path}") from exc
    if not isinstance(value, dict):
        raise BootstrapRefusal(f"required JSON is not an object: {path}")
    return value


def _is_private_regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _validate_manifest(
    path: Path, *, require_runtime_closure: bool = True
) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise BootstrapRefusal("runtime manifest must be a regular image file")
    manifest_sha256 = _sha256(path)
    if manifest_sha256 != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise BootstrapRefusal("runtime manifest bytes differ from the image contract")
    manifest = _load_json(path)
    if manifest.get("schema") != SCHEMA or manifest.get("solution") != "libero":
        raise BootstrapRefusal("runtime manifest identity is invalid")
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise BootstrapRefusal("runtime manifest top-level schema is not closed")
    if manifest.get("runtime_python") != EXPECTED_RUNTIME_PYTHON:
        raise BootstrapRefusal("runtime Python identity is invalid")
    if manifest.get("runtime_use_decision") != EXPECTED_RUNTIME_DECISION_METADATA:
        raise BootstrapRefusal("runtime-use decision metadata is stale or invalid")
    if manifest.get("boundaries") != EXPECTED_BOUNDARIES:
        raise BootstrapRefusal("runtime cache/output boundary metadata is invalid")
    governing_terms = manifest.get("governing_terms")
    if not isinstance(governing_terms, list) or len(governing_terms) != len(
        EXPECTED_GOVERNING_TERMS
    ):
        raise BootstrapRefusal("governing terms inventory is incomplete")
    term_ids: set[str] = set()
    term_boundaries: set[str] = set()
    for term in governing_terms:
        if not isinstance(term, dict) or set(term) != {
            "id",
            "boundary",
            "url",
            "size_bytes",
            "sha256",
        }:
            raise BootstrapRefusal("governing terms entry is not closed")
        term_id = str(term.get("id") or "")
        boundary = str(term.get("boundary") or "")
        if (
            term_id in term_ids
            or boundary not in EXPECTED_DECISION_BOUNDARIES
            or not isinstance(term.get("size_bytes"), int)
            or term["size_bytes"] <= 0
            or not _is_hex(term.get("sha256"), 64)
        ):
            raise BootstrapRefusal("governing terms identity is invalid")
        _validate_terms_url(str(term.get("url") or ""))
        term_ids.add(term_id)
        term_boundaries.add(boundary)
    if (
        term_ids != EXPECTED_GOVERNING_TERMS
        or not {
            "source",
            "runtime_packages",
            "demonstration",
            "language_model",
        }
        <= term_boundaries
    ):
        raise BootstrapRefusal("governing terms inventory is incomplete")
    source = manifest.get("source") or {}
    if (
        source.get("repository")
        != "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
        or not _is_hex(source.get("revision"), 40)
        or not _is_hex(source.get("tree"), 40)
        or source.get("license") != "MIT"
        or not _is_hex(source.get("license_sha256"), 64)
    ):
        raise BootstrapRefusal("pinned LIBERO source contract is invalid")
    sparse_paths = source.get("sparse_paths")
    if (
        not isinstance(sparse_paths, list)
        or len(sparse_paths) != len(set(sparse_paths))
        or not all(isinstance(item, str) and item for item in sparse_paths)
        or "libero/libero/assets" in sparse_paths
        or source.get("forbidden_paths") != ["libero/libero/assets", ".git"]
    ):
        raise BootstrapRefusal("LIBERO sparse-source boundary is invalid")
    demonstration = manifest.get("demonstration") or {}
    if (
        demonstration.get("repository") != "yifengzhu-hf/LIBERO-datasets"
        or not _is_hex(demonstration.get("revision"), 40)
        or demonstration.get("license") != "CC-BY-4.0"
        or not str(demonstration.get("attribution") or "").strip()
        or not str(demonstration.get("filename") or "").endswith(".hdf5")
        or not _is_hex(demonstration.get("sha256"), 64)
        or not isinstance(demonstration.get("size_bytes"), int)
        or demonstration["size_bytes"] <= 0
    ):
        raise BootstrapRefusal("official demonstration contract is invalid")
    _validate_download_url(str(demonstration.get("url") or ""))
    task = manifest.get("task") or {}
    if (
        task.get("suite") != "libero_spatial"
        or not str(task.get("name") or "").strip()
        or not str(task.get("description") or "").strip()
    ):
        raise BootstrapRefusal("LIBERO task identity is invalid")
    for key in ("bddl", "initial_states", "embedding_source"):
        record = task.get(key) or {}
        if not str(record.get("path") or "").strip() or not _is_hex(
            record.get("sha256"), 64
        ):
            raise BootstrapRefusal(f"LIBERO task identity is invalid: {key}")
    model = manifest.get("language_model") or {}
    model_files = model.get("files")
    if (
        model.get("repository") != "google-bert/bert-base-cased"
        or not _is_hex(model.get("revision"), 40)
        or model.get("license") != "Apache-2.0"
        or not isinstance(model_files, list)
        or not model_files
    ):
        raise BootstrapRefusal("task language-model contract is invalid")
    model_names: set[str] = set()
    for item in model_files:
        if not isinstance(item, dict):
            raise BootstrapRefusal("task language-model file is not an object")
        filename = str(item.get("filename") or "")
        if (
            not filename
            or filename in model_names
            or "/" in filename
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] <= 0
            or not _is_hex(item.get("sha256"), 64)
        ):
            raise BootstrapRefusal("task language-model file identity is invalid")
        model_names.add(filename)
    artifacts = manifest.get("runtime_artifacts")
    if (
        manifest.get("runtime_artifact_count") != 135
        or not isinstance(artifacts, list)
        or len(artifacts) != manifest["runtime_artifact_count"]
    ):
        raise BootstrapRefusal(
            "runtime artifact inventory must contain exactly 135 items"
        )
    names: set[str] = set()
    filenames: set[str] = set()
    total_size_bytes = 0
    for item in artifacts:
        if not isinstance(item, dict):
            raise BootstrapRefusal("runtime artifact entry is not an object")
        name = str(item.get("name") or "")
        filename = str(item.get("filename") or "")
        if name in names or filename in filenames or not name or not filename:
            raise BootstrapRefusal(
                "runtime artifact names and filenames must be unique"
            )
        names.add(name)
        filenames.add(filename)
        _validate_download_url(str(item.get("url") or ""))
        if not _is_hex(item.get("sha256"), 64):
            raise BootstrapRefusal(f"invalid runtime artifact hash for {name}")
        size_bytes = item.get("size_bytes")
        license_expression = str(item.get("license_expression") or "").strip()
        if require_runtime_closure and (
            not isinstance(size_bytes, int)
            or size_bytes <= 0
            or not license_expression
        ):
            raise BootstrapRefusal(
                f"runtime artifact size/license review is incomplete for {name}"
            )
        if isinstance(size_bytes, int) and size_bytes > 0:
            total_size_bytes += size_bytes
    if require_runtime_closure and total_size_bytes > MAX_RUNTIME_CACHE_DOWNLOAD_BYTES:
        raise BootstrapRefusal("runtime artifact inventory exceeds the cache budget")
    return manifest, manifest_sha256


def _validate_requirements(
    path: Path, manifest: dict[str, Any]
) -> tuple[list[str], str]:
    if not path.is_file() or path.is_symlink():
        raise BootstrapRefusal("runtime requirements must be a regular image file")
    requirements_sha256 = _sha256(path)
    if requirements_sha256 != EXPECTED_RUNTIME_REQUIREMENTS_SHA256:
        raise BootstrapRefusal(
            "runtime requirements bytes differ from the image contract"
        )
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    artifacts = manifest["runtime_artifacts"]
    if len(lines) != len(artifacts):
        raise BootstrapRefusal("runtime requirements and artifact inventory differ")
    for line, artifact in zip(lines, artifacts, strict=True):
        expected_prefix = (
            f"{artifact['name']}=={artifact['version']} "
            f"--hash=sha256:{artifact['sha256']} # {artifact['url']}"
        )
        if line != expected_prefix:
            raise BootstrapRefusal(
                "runtime requirements do not bind the manifest order"
            )
    return lines, requirements_sha256


def _is_hex(value: object, length: int) -> bool:
    text = str(value or "")
    return len(text) == length and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS
        or parsed.fragment
    ):
        raise BootstrapRefusal(
            "runtime download URL is outside the immutable allowlist"
        )


def _validate_terms_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname not in ALLOWED_TERMS_HOSTS
        or parsed.fragment
    ):
        raise BootstrapRefusal("governing terms URL is outside its allowlist")


def _validate_redirect_url(
    url: str, *, terms: bool = False
) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname or ""
    allowed_hosts = ALLOWED_TERMS_HOSTS if terms else ALLOWED_DOWNLOAD_HOSTS
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or not (
            hostname in allowed_hosts
            or (not terms and hostname in ALLOWED_DOWNLOAD_REDIRECT_HOSTS)
        )
        or parsed.fragment
    ):
        raise BootstrapRefusal("runtime redirect left the download allowlist")
    return parsed


def _open_https_download(
    url: str, *, terms: bool = False
) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    current_url = url
    for _ in range(6):
        parsed = _validate_redirect_url(current_url, terms=terms)
        connection = http.client.HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=60
        )
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request(
            "GET", target, headers={"User-Agent": "npa-libero-runtime/1"}
        )
        response = connection.getresponse()
        if response.status not in {301, 302, 303, 307, 308}:
            if response.status != 200:
                status = response.status
                response.close()
                connection.close()
                raise BootstrapRefusal(
                    f"runtime download returned unexpected HTTP status {status}"
                )
            return connection, response
        location = response.getheader("Location")
        response.close()
        connection.close()
        if not location:
            raise BootstrapRefusal("runtime redirect omitted its target")
        current_url = urllib.parse.urljoin(current_url, location)
    raise BootstrapRefusal("runtime download exceeded the redirect limit")


def _validate_cache_root(cache_root: Path, output_dir: Path | None) -> Path:
    if cache_root.is_symlink():
        raise BootstrapRefusal("runtime cache root may not be a symlink")
    resolved = cache_root.resolve(strict=False)
    if not resolved.is_absolute() or resolved == Path("/"):
        raise BootstrapRefusal("runtime cache root must be a narrow absolute path")
    if output_dir is not None:
        output = output_dir.resolve(strict=False)
        if (
            output == resolved
            or output in resolved.parents
            or resolved in output.parents
        ):
            raise BootstrapRefusal("runtime cache and output boundaries overlap")
    return resolved


def _cache_entry_identity(path: Path) -> tuple[int, int] | None:
    """Return a real directory's identity without following the final component."""

    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise BootstrapRefusal("cannot inspect the runtime cache entry") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise BootstrapRefusal("runtime cache entry must be a real directory")
    return info.st_dev, info.st_ino


def _require_cache_entry_identity(path: Path, expected: tuple[int, int]) -> None:
    if _cache_entry_identity(path) != expected:
        raise BootstrapRefusal("runtime cache entry changed during validation")


@contextmanager
def _open_cache_entry(path: Path, *, expected: tuple[int, int]) -> Iterator[Path]:
    """Retain the no-follow cache leaf while validating through its descriptor."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise BootstrapRefusal("runtime cache requires no-follow directory support")
    try:
        descriptor = os.open(path, os.O_RDONLY | directory | nofollow | os.O_CLOEXEC)
    except OSError as exc:
        raise BootstrapRefusal(
            "runtime cache entry could not be opened without following links"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected
            or stat.S_IMODE(opened.st_mode) & 0o222
        ):
            raise BootstrapRefusal("runtime cache entry identity or mode is invalid")
        _require_cache_entry_identity(path, expected)
        stable_root = Path("/proc/self/fd") / str(descriptor)
        if not stable_root.is_dir():
            raise BootstrapRefusal("runtime cache descriptor is unavailable")
        yield stable_root
        _require_cache_entry_identity(path, expected)
    finally:
        os.close(descriptor)


def _publish_current_cache_link(
    cache_root: Path, final: Path, expected: tuple[int, int]
) -> None:
    """Publish ``current`` only while the validated leaf keeps its identity."""

    current = cache_root / "current"
    temporary_link = cache_root / f".current-{os.getpid()}"
    temporary_link.unlink(missing_ok=True)
    try:
        _require_cache_entry_identity(final, expected)
        temporary_link.symlink_to(final.name)
        _require_cache_entry_identity(final, expected)
        temporary_link.replace(current)
        _require_cache_entry_identity(final, expected)
    except Exception:
        temporary_link.unlink(missing_ok=True)
        _remove_current_cache_link(cache_root, final)
        raise


def _remove_current_cache_link(cache_root: Path, final: Path) -> None:
    current = cache_root / "current"
    if current.is_symlink() and os.readlink(current) == final.name:
        current.unlink()


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _canonical_unsigned_acceptance(payload: dict[str, Any]) -> bytes:
    unsigned = json.loads(json.dumps(payload))
    acceptance = unsigned.get("acceptance")
    if not isinstance(acceptance, dict):
        raise BootstrapRefusal("manager acceptance record is unavailable")
    acceptance.pop("manager_signature", None)
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def _manager_signature_payload(payload: dict[str, Any]) -> bytes:
    canonical = _canonical_unsigned_acceptance(payload)
    return b"".join(
        (
            b"SSHSIG",
            _ssh_string(MANAGER_ACCEPTANCE_NAMESPACE),
            _ssh_string(b""),
            _ssh_string(b"sha512"),
            _ssh_string(hashlib.sha512(canonical).digest()),
        )
    )


def _manager_sshsig(public_key: bytes, signature: bytes) -> bytes:
    public_key_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public_key)
    signature_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(signature)
    payload = b"".join(
        (
            b"SSHSIG",
            struct.pack(">I", 1),
            _ssh_string(public_key_blob),
            _ssh_string(MANAGER_ACCEPTANCE_NAMESPACE),
            _ssh_string(b""),
            _ssh_string(b"sha512"),
            _ssh_string(signature_blob),
        )
    )
    encoded = base64.b64encode(payload).decode("ascii")
    lines = [encoded[index : index + 70] for index in range(0, len(encoded), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    ).encode()


def _trusted_manager_public_key() -> bytes:
    """Load the image-baked trust root without accepting a payload selector."""

    try:
        metadata = MANAGER_ACCEPTANCE_PUBLIC_KEY.lstat()
        encoded = MANAGER_ACCEPTANCE_PUBLIC_KEY.read_bytes()
    except OSError as exc:
        raise BootstrapRefusal("manager acceptance trust root is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != MANAGER_ACCEPTANCE_PUBLIC_KEY_OWNER_UID
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or encoded != encoded.strip()
    ):
        raise BootstrapRefusal("manager acceptance trust root is mutable or invalid")
    try:
        public_key = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise BootstrapRefusal("manager acceptance trust root is invalid") from exc
    if len(public_key) != 32:
        raise BootstrapRefusal("manager acceptance trust root is invalid")
    return public_key


def _verify_manager_signature(
    payload: dict[str, Any], signature_record: dict[str, Any]
) -> None:
    public_key = _trusted_manager_public_key()
    claimed_fingerprint = str(signature_record.get("public_key_sha256") or "")
    if (
        not _is_hex(claimed_fingerprint, 64)
        or hashlib.sha256(public_key).hexdigest() != claimed_fingerprint
    ):
        # The root-owned key baked into the image is authoritative. This signed
        # field is only a consistency assertion; it can never select a key.
        raise BootstrapRefusal("manager acceptance trust root differs")
    try:
        signature = base64.b64decode(
            str(signature_record.get("signature_b64") or ""), validate=True
        )
    except ValueError as exc:
        raise BootstrapRefusal("manager acceptance signature is invalid") from exc
    if len(signature) != 64:
        raise BootstrapRefusal("manager acceptance signature is invalid")
    canonical = _canonical_unsigned_acceptance(payload)
    public_key_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public_key)
    allowed_signer = (
        "npa-manager ssh-ed25519 "
        + base64.b64encode(public_key_blob).decode("ascii")
        + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="npa-libero-manager-signature-") as root:
        root_path = Path(root)
        allowed_path = root_path / "allowed-signers"
        signature_path = root_path / "acceptance.sig"
        allowed_path.write_text(allowed_signer, encoding="ascii")
        signature_path.write_bytes(_manager_sshsig(public_key, signature))
        os.chmod(allowed_path, 0o600)
        os.chmod(signature_path, 0o600)
        completed = subprocess.run(
            [
                "/usr/bin/ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_path),
                "-I",
                "npa-manager",
                "-n",
                MANAGER_ACCEPTANCE_NAMESPACE.decode("ascii"),
                "-s",
                str(signature_path),
            ],
            input=canonical,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
            check=False,
        )
    if completed.returncode:
        raise BootstrapRefusal("manager acceptance signature is invalid")


def _validate_manager_acceptance(
    path: Path, *, manifest_sha256: str, decision_sha256: str
) -> dict[str, Any]:
    """Verify the image-local decision against a non-substitutable trust root."""

    if not _is_private_regular_file(path):
        raise BootstrapRefusal("manager acceptance must be an owner-only regular file")
    payload = _load_json(path)
    acceptance = payload.get("acceptance")
    signature_record = (
        acceptance.get("manager_signature") if isinstance(acceptance, dict) else None
    )
    if (
        payload.get("schema") != "npa.workbench.image-manifest.v1"
        or payload.get("tool") != "libero"
        or payload.get("image_name") != "npa-libero"
        or payload.get("runtime_payloads_baked") is not False
        or payload.get("runtime_use_decision_required") is not True
        or not isinstance(acceptance, dict)
        or acceptance.get("schema") != "npa.libero.qualification-acceptance.v3"
        or acceptance.get("status") != "accepted"
        or not isinstance(signature_record, dict)
        or set(signature_record)
        != {"algorithm", "public_key_sha256", "signature_b64"}
        or signature_record.get("algorithm") != "ed25519"
        or acceptance.get("runtime_manifest_sha256") != manifest_sha256
        or acceptance.get("runtime_use_decision_sha256") != decision_sha256
    ):
        raise BootstrapRefusal("manager acceptance does not authorize this runtime")
    infrastructure = acceptance.get("infrastructure")
    if not isinstance(infrastructure, dict):
        raise BootstrapRefusal("manager acceptance infrastructure is invalid")
    observed_infrastructure = hashlib.sha256(
        json.dumps(
            {"schema": "npa.libero.infrastructure-bundle.v1", **infrastructure},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if observed_infrastructure != acceptance.get("infrastructure_bundle_sha256"):
        raise BootstrapRefusal("manager acceptance infrastructure differs")
    try:
        accepted_at = datetime.fromisoformat(
            str(acceptance["accepted_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(acceptance["expires_at"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BootstrapRefusal("manager acceptance timestamps are invalid") from exc
    now = datetime.now(timezone.utc)
    if (
        accepted_at.tzinfo is None
        or expires_at.tzinfo is None
        or accepted_at > now + timedelta(minutes=5)
        or accepted_at >= expires_at
        or expires_at <= now
        or expires_at - accepted_at > timedelta(days=7)
    ):
        raise BootstrapRefusal("manager acceptance is expired or replayable")
    _verify_manager_signature(payload, signature_record)
    return acceptance


def _validate_decision(
    path: Path,
    expected_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    acceptance: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if not _is_hex(expected_sha256, 64):
        raise BootstrapRefusal("expected decision SHA-256 is required")
    if not _is_private_regular_file(path):
        raise BootstrapRefusal(
            "runtime-use decision must be an owner-only regular file"
        )
    observed = _sha256(path)
    if observed != expected_sha256:
        raise BootstrapRefusal("runtime-use decision hash does not match")
    decision = _load_json(path)
    source = manifest["source"]
    infrastructure = acceptance["infrastructure"]
    boundaries = decision.get("authorized_boundaries")
    expected_keys = {
        "schema",
        "solution",
        "decision",
        "runtime_fetch_authorized",
        "acceptance_id",
        "candidate_image",
        "publication_bundle_sha256",
        "infrastructure_bundle_sha256",
        "runtime_manifest_sha256",
        "upstream_source_revision",
        "authorized_boundaries",
        "run_id",
        "namespace_sha256",
        "issuer",
        "issued_at",
        "expires_at",
        "nonce",
    }
    if (
        set(decision) != expected_keys
        or decision.get("schema") != DECISION_SCHEMA
        or decision.get("solution") != "libero"
        or decision.get("decision") != "authorized"
        or decision.get("runtime_fetch_authorized") is not True
        or decision.get("acceptance_id")
        != acceptance.get("acceptance_id")
        or decision.get("candidate_image") != acceptance.get("candidate_image")
        or decision.get("publication_bundle_sha256")
        != acceptance.get("publication_bundle_sha256")
        or decision.get("infrastructure_bundle_sha256")
        != acceptance.get("infrastructure_bundle_sha256")
        or decision.get("runtime_manifest_sha256") != manifest_sha256
        or decision.get("upstream_source_revision") != source["revision"]
        or decision.get("run_id") != infrastructure.get("run_id")
        or decision.get("namespace_sha256")
        != infrastructure.get("namespace_sha256")
        or os.environ.get("BYOF_IMAGE") != acceptance.get("candidate_image")
        or os.environ.get("NPA_BYOF_RUN_ID") != infrastructure.get("run_id")
        or decision.get("issuer") != "npa-manager"
        or not isinstance(boundaries, list)
        or frozenset(boundaries) != EXPECTED_DECISION_BOUNDARIES
        or len(boundaries) != len(EXPECTED_DECISION_BOUNDARIES)
        or not 32 <= len(str(decision.get("nonce") or "")) <= 128
        or not all(
            character.isalnum() or character in "_-"
            for character in str(decision.get("nonce") or "")
        )
    ):
        raise BootstrapRefusal("runtime-use decision does not bind the exact contract")
    try:
        issued_at = datetime.fromisoformat(
            str(decision["issued_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(decision["expires_at"]).replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise BootstrapRefusal("runtime-use decision timestamps are invalid") from exc
    now = datetime.now(timezone.utc)
    if (
        issued_at.tzinfo is None
        or expires_at.tzinfo is None
        or issued_at > now + timedelta(minutes=5)
        or expires_at <= now
        or expires_at - issued_at > timedelta(hours=24)
    ):
        raise BootstrapRefusal("runtime-use decision is expired or replayable")
    return decision, observed


def _download_verified(
    destination: Path,
    *,
    url: str,
    sha256: str,
    size: int,
    terms: bool = False,
) -> None:
    if size <= 0 or size > MAX_RUNTIME_CACHE_DOWNLOAD_BYTES:
        raise BootstrapRefusal("runtime download has no valid expected size")
    (_validate_terms_url if terms else _validate_download_url)(url)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    observed_size = 0
    connection: http.client.HTTPSConnection | None = None
    response: http.client.HTTPResponse | None = None
    try:
        connection, response = _open_https_download(url, terms=terms)
        content_length = response.getheader("Content-Length")
        if content_length is not None and (
            not content_length.isdigit() or int(content_length) != size
        ):
            raise BootstrapRefusal(
                "runtime download Content-Length differs from expected size"
            )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if observed_size + len(chunk) > size:
                    raise BootstrapRefusal("runtime download exceeded expected size")
                stream.write(chunk)
                digest.update(chunk)
                observed_size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
    if digest.hexdigest() != sha256 or observed_size != size:
        temporary.unlink(missing_ok=True)
        raise BootstrapRefusal(
            "runtime download bytes do not match their immutable identity"
        )
    temporary.replace(destination)


def _governing_terms_identity(manifest: dict[str, Any]) -> str:
    identities = [
        {
            "id": term["id"],
            "boundary": term["boundary"],
            "sha256": term["sha256"],
            "size_bytes": term["size_bytes"],
            "url": term["url"],
        }
        for term in manifest["governing_terms"]
    ]
    encoded = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_governing_terms(manifest: dict[str, Any]) -> str:
    """Resolve and hash every governing terms source before cold cache mutation."""

    with tempfile.TemporaryDirectory(prefix="npa-libero-terms-") as temporary:
        root = Path(temporary)
        for index, term in enumerate(manifest["governing_terms"]):
            destination = root / f"term-{index}"
            _download_verified(
                destination,
                url=term["url"],
                sha256=term["sha256"],
                size=int(term["size_bytes"]),
                terms=True,
            )
    return _governing_terms_identity(manifest)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> None:
    selected_environment = dict(os.environ if environment is None else environment)
    selected_environment["GIT_TERMINAL_PROMPT"] = "0"
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        env=selected_environment,
        stdout=sys.stderr,
    )


def _runtime_materialization_environment(root: Path) -> dict[str, str]:
    """Return the credential-free environment shared by every fetch/build child."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if name in RUNTIME_MATERIALIZATION_PASSTHROUGH_ENV_NAMES
    }
    environment.update(
        {
            "GIT_ASKPASS": "/bin/false",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(root),
        }
    )
    return environment


def _fetch_source(root: Path, source: dict[str, Any]) -> None:
    destination = root / "source"
    destination.mkdir(mode=0o700)
    environment = _runtime_materialization_environment(root)
    _run(["git", "init", "--quiet"], cwd=destination, environment=environment)
    _run(
        ["git", "remote", "add", "origin", source["repository"]],
        cwd=destination,
        environment=environment,
    )
    _run(
        ["git", "config", "remote.origin.promisor", "true"],
        cwd=destination,
        environment=environment,
    )
    _run(
        ["git", "config", "remote.origin.partialclonefilter", "blob:none"],
        cwd=destination,
        environment=environment,
    )
    _run(
        [
            "git",
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            source["revision"],
        ],
        cwd=destination,
        environment=environment,
    )
    fetched = subprocess.check_output(
        ["git", "rev-parse", "FETCH_HEAD^{commit}"],
        cwd=destination,
        env=environment,
        text=True,
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "FETCH_HEAD^{tree}"],
        cwd=destination,
        env=environment,
        text=True,
    ).strip()
    if fetched != source["revision"] or tree != source["tree"]:
        raise BootstrapRefusal(
            "fetched LIBERO source identity differs from the manifest"
        )
    _run(
        ["git", "sparse-checkout", "init", "--no-cone"],
        cwd=destination,
        environment=environment,
    )
    _run(
        ["git", "sparse-checkout", "set", "--no-cone", "--", *source["sparse_paths"]],
        cwd=destination,
        environment=environment,
    )
    _run(
        ["git", "checkout", "--quiet", "--detach", fetched],
        cwd=destination,
        environment=environment,
    )
    if _sha256(destination / source["license_file"]) != source["license_sha256"]:
        raise BootstrapRefusal(
            "LIBERO source license does not match the reviewed MIT file"
        )
    if (destination / "libero" / "libero" / "assets").exists():
        raise BootstrapRefusal("forbidden LIBERO render payload survived sparse fetch")
    shutil.rmtree(destination / ".git")


def _install_runtime(
    root: Path,
    artifacts: list[dict[str, Any]],
    requirement_lines: list[str],
) -> None:
    materialization_environment = _runtime_materialization_environment(root)
    wheelhouse = root / "downloads"
    wheelhouse.mkdir(mode=0o700)
    for item in artifacts:
        _download_verified(
            wheelhouse / item["filename"],
            url=item["url"],
            sha256=item["sha256"],
            size=int(item["size_bytes"]),
        )
    venv = root / "venv"
    _run(
        [sys.executable, "-m", "venv", "--copies", str(venv)],
        environment=materialization_environment,
    )
    bootstrap_names = {"pip", "setuptools", "wheel"}
    bootstrap_lock = root / ".bootstrap-requirements.txt"
    runtime_lock = root / ".runtime-requirements.txt"
    bootstrap_lock.write_text(
        "\n".join(
            line
            for line, item in zip(requirement_lines, artifacts, strict=True)
            if item["name"] in bootstrap_names
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_lock.write_text(
        "\n".join(
            line
            for line, item in zip(requirement_lines, artifacts, strict=True)
            if item["name"] not in bootstrap_names
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(bootstrap_lock, 0o400)
    os.chmod(runtime_lock, 0o400)
    for artifact in wheelhouse.iterdir():
        os.chmod(artifact, 0o400)
    os.chmod(wheelhouse, 0o500)
    pip = str(venv / "bin" / "python")
    common = [
        pip,
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--no-index",
        "--no-deps",
        "--no-cache-dir",
        "--find-links",
        str(wheelhouse),
    ]
    _run(
        [*common, "-r", str(bootstrap_lock)],
        environment=materialization_environment,
    )
    _run(
        [*common, "--no-build-isolation", "-r", str(runtime_lock)],
        environment=materialization_environment,
    )
    site_packages = subprocess.check_output(
        [pip, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
        env=materialization_environment,
    ).strip()
    Path(site_packages, "npa-libero-source.pth").write_text(
        str(root / "source") + "\n", encoding="utf-8"
    )
    os.chmod(wheelhouse, 0o700)
    shutil.rmtree(wheelhouse)
    bootstrap_lock.unlink()
    runtime_lock.unlink()


def _fetch_inputs(root: Path, manifest: dict[str, Any]) -> None:
    demonstration = manifest["demonstration"]
    _download_verified(
        root / "data" / demonstration["filename"],
        url=demonstration["url"],
        sha256=demonstration["sha256"],
        size=int(demonstration["size_bytes"]),
    )
    model = manifest["language_model"]
    model_root = root / "models" / f"bert-base-cased-{model['revision']}"
    for item in model["files"]:
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{item['filename']}?download=true"
        _download_verified(
            model_root / item["filename"],
            url=url,
            sha256=item["sha256"],
            size=int(item["size_bytes"]),
        )
    model_record = {
        "repository": model["repository"],
        "revision": model["revision"],
        "files": {item["filename"]: item["sha256"] for item in model["files"]},
    }
    record_path = model_root / "npa-language-model.json"
    record_path.write_text(
        json.dumps(model_record, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(record_path, 0o600)


def _validate_task_inputs(root: Path, manifest: dict[str, Any]) -> None:
    source = root / "source"
    task = manifest["task"]
    for key in ("bddl", "initial_states", "embedding_source"):
        record = task[key]
        path = source / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise BootstrapRefusal(
                f"genuine upstream task input failed identity: {key}"
            )


def _inventory_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    excluded = {".complete.json", ".content-inventory.json"}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise BootstrapRefusal(
                f"runtime cache may not contain symlinks: {relative}"
            )
        elif stat.S_ISDIR(info.st_mode):
            entries.append({"path": relative, "type": "directory"})
        elif stat.S_ISREG(info.st_mode):
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": info.st_size,
                    "sha256": _sha256(path),
                }
            )
        else:
            raise BootstrapRefusal(
                f"runtime cache contains unsupported filesystem entry: {relative}"
            )
    return entries


def _write_content_inventory(root: Path) -> tuple[str, int]:
    entries = _inventory_entries(root)
    path = root / ".content-inventory.json"
    path.write_text(
        json.dumps({"schema": INVENTORY_SCHEMA, "entries": entries}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return _sha256(path), len(entries)


def _seal_cache_tree(root: Path) -> None:
    try:
        execution_gid = grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid
    except KeyError as exc:
        raise BootstrapRefusal("runtime execution group is unavailable") from exc
    paths = sorted(
        (root, *root.rglob("*")),
        key=lambda item: len(item.relative_to(root).parts),
        reverse=True,
    )
    for path in paths:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise BootstrapRefusal("runtime cache may not contain symlinks")
        os.chown(path, -1, execution_gid)
        if stat.S_ISDIR(info.st_mode):
            os.chmod(path, SEALED_DIRECTORY_MODE)
        elif stat.S_ISREG(info.st_mode):
            os.chmod(
                path,
                SEALED_EXECUTABLE_MODE
                if info.st_mode & 0o111
                else SEALED_REGULAR_MODE,
            )
        else:
            raise BootstrapRefusal("runtime cache contains an unsupported entry")


def _validate_read_only_tree(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        # _open_cache_entry intentionally exposes the stable root through a
        # descriptor symlink under /proc/self/fd. Validate that descriptor's
        # target; no descendant symlink is part of the cache contract.
        info = path.stat() if path == root else path.lstat()
        if path != root and stat.S_ISLNK(info.st_mode):
            raise BootstrapRefusal("runtime cache may not contain symlinks")
        if stat.S_IMODE(info.st_mode) & 0o222:
            raise BootstrapRefusal("existing runtime cache is writable")


def _complete_record(
    root: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    decision_sha256: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    source = manifest["source"]
    demonstration = manifest["demonstration"]
    inventory_sha256, inventory_entry_count = _write_content_inventory(root)
    record = {
        "schema": COMPLETE_SCHEMA,
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "decision_sha256": decision_sha256,
        "runtime_requirements_sha256": requirements_sha256,
        "governing_terms_sha256": governing_terms_sha256,
        "governing_terms_count": len(manifest["governing_terms"]),
        "source_revision": source["revision"],
        "source_tree": source["tree"],
        "source_license_sha256": source["license_sha256"],
        "runtime_artifact_count": len(manifest["runtime_artifacts"]),
        "demonstration_sha256": demonstration["sha256"],
        "demonstration_size_bytes": demonstration["size_bytes"],
        "task_bddl_sha256": manifest["task"]["bddl"]["sha256"],
        "task_initial_states_sha256": manifest["task"]["initial_states"]["sha256"],
        "language_model_revision": manifest["language_model"]["revision"],
        "render_assets_present": False,
        "git_objects_present": False,
        "cache_uploaded": False,
        "content_inventory_sha256": inventory_sha256,
        "content_inventory_entry_count": inventory_entry_count,
    }
    complete = root / ".complete.json"
    complete.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(complete, 0o600)
    return record


def _validate_complete(
    root: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    decision_sha256: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    _validate_read_only_tree(root)
    record = _load_json(root / ".complete.json")
    expected = _complete_record_values(
        manifest,
        manifest_sha256,
        decision_sha256,
        requirements_sha256,
        governing_terms_sha256,
    )
    dynamic_keys = {"content_inventory_sha256", "content_inventory_entry_count"}
    if set(record) != set(expected) | dynamic_keys or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise BootstrapRefusal(
            "existing runtime cache completion record does not match"
        )
    inventory_path = root / ".content-inventory.json"
    if (
        not inventory_path.is_file()
        or inventory_path.is_symlink()
        or not _is_hex(record["content_inventory_sha256"], 64)
        or _sha256(inventory_path) != record["content_inventory_sha256"]
    ):
        raise BootstrapRefusal("existing runtime cache inventory is invalid")
    inventory = _load_json(inventory_path)
    entries = inventory.get("entries")
    if (
        inventory.get("schema") != INVENTORY_SCHEMA
        or not isinstance(entries, list)
        or record["content_inventory_entry_count"] != len(entries)
        or entries != _inventory_entries(root)
    ):
        raise BootstrapRefusal("existing runtime cache content differs from inventory")
    if (root / "source" / "libero" / "libero" / "assets").exists() or (
        root / "source" / ".git"
    ).exists():
        raise BootstrapRefusal(
            "existing runtime cache contains forbidden source payload"
        )
    if not (root / "venv" / "bin" / "python").is_file():
        raise BootstrapRefusal("existing runtime cache is incomplete")
    demonstration = manifest["demonstration"]
    data = root / "data" / demonstration["filename"]
    if (
        not data.is_file()
        or data.stat().st_size != demonstration["size_bytes"]
        or _sha256(data) != demonstration["sha256"]
    ):
        raise BootstrapRefusal("existing demonstration cache does not match")
    model = manifest["language_model"]
    model_root = root / "models" / f"bert-base-cased-{model['revision']}"
    for item in model["files"]:
        path = model_root / item["filename"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _sha256(path) != item["sha256"]
        ):
            raise BootstrapRefusal("existing task language-model cache does not match")
    _validate_task_inputs(root, manifest)
    return record


def _validate_and_publish_cache(
    *,
    cache_root: Path,
    final: Path,
    identity: tuple[int, int],
    manifest: dict[str, Any],
    manifest_sha256: str,
    decision_sha256: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    published = False
    try:
        with _open_cache_entry(final, expected=identity) as stable_final:
            record = _validate_complete(
                stable_final,
                manifest,
                manifest_sha256,
                decision_sha256,
                requirements_sha256,
                governing_terms_sha256,
            )
            _publish_current_cache_link(cache_root, final, identity)
            published = True
    except Exception:
        if published:
            _remove_current_cache_link(cache_root, final)
        raise
    return record


def _complete_record_values(
    manifest: dict[str, Any],
    manifest_sha256: str,
    decision_sha256: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    source = manifest["source"]
    demonstration = manifest["demonstration"]
    return {
        "schema": COMPLETE_SCHEMA,
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "decision_sha256": decision_sha256,
        "runtime_requirements_sha256": requirements_sha256,
        "governing_terms_sha256": governing_terms_sha256,
        "governing_terms_count": len(manifest["governing_terms"]),
        "source_revision": source["revision"],
        "source_tree": source["tree"],
        "source_license_sha256": source["license_sha256"],
        "runtime_artifact_count": len(manifest["runtime_artifacts"]),
        "demonstration_sha256": demonstration["sha256"],
        "demonstration_size_bytes": demonstration["size_bytes"],
        "task_bddl_sha256": manifest["task"]["bddl"]["sha256"],
        "task_initial_states_sha256": manifest["task"]["initial_states"]["sha256"],
        "language_model_revision": manifest["language_model"]["revision"],
        "render_assets_present": False,
        "git_objects_present": False,
        "cache_uploaded": False,
    }


def ensure(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest, manifest_sha256 = _validate_manifest(manifest_path)
    decision_path = Path(args.decision)
    acceptance = _validate_manager_acceptance(
        Path(args.acceptance),
        manifest_sha256=manifest_sha256,
        decision_sha256=args.decision_sha256,
    )
    _, decision_sha256 = _validate_decision(
        decision_path,
        args.decision_sha256,
        manifest,
        manifest_sha256,
        acceptance,
    )
    requirement_lines, requirements_sha256 = _validate_requirements(
        Path(args.requirements), manifest
    )
    output = Path(args.output_dir) if args.output_dir else None
    cache_root = _validate_cache_root(Path(args.cache_root), output)
    final = cache_root / manifest_sha256
    initial_identity = _cache_entry_identity(final)
    governing_terms_sha256 = (
        _governing_terms_identity(manifest)
        if initial_identity is not None
        else _verify_governing_terms(manifest)
    )
    cache_root.mkdir(mode=CACHE_ROOT_MODE, parents=True, exist_ok=True)
    try:
        os.chown(cache_root, -1, grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid)
    except KeyError as exc:
        raise BootstrapRefusal("runtime execution group is unavailable") from exc
    os.chmod(cache_root, CACHE_ROOT_MODE)
    lock_path = cache_root / ".bootstrap.lock"
    with lock_path.open("a+b") as lock:
        os.chown(lock_path, -1, grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid)
        os.chmod(lock_path, 0o640)
        fcntl.flock(lock, fcntl.LOCK_EX)
        locked_identity = _cache_entry_identity(final)
        if initial_identity is not None and locked_identity != initial_identity:
            raise BootstrapRefusal("runtime cache entry changed before validation")
        warm_reuse = locked_identity is not None
        if warm_reuse:
            assert locked_identity is not None
            record = _validate_and_publish_cache(
                cache_root=cache_root,
                final=final,
                identity=locked_identity,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                decision_sha256=decision_sha256,
                requirements_sha256=requirements_sha256,
                governing_terms_sha256=governing_terms_sha256,
            )
        else:
            partial = Path(
                tempfile.mkdtemp(prefix=f".{manifest_sha256}.partial-", dir=cache_root)
            )
            os.chmod(partial, 0o700)
            try:
                _fetch_source(partial, manifest["source"])
                _validate_task_inputs(partial, manifest)
                _install_runtime(
                    partial, manifest["runtime_artifacts"], requirement_lines
                )
                _fetch_inputs(partial, manifest)
                record = _complete_record(
                    partial,
                    manifest,
                    manifest_sha256,
                    decision_sha256,
                    requirements_sha256,
                    governing_terms_sha256,
                )
                _seal_cache_tree(partial)
                partial.replace(final)
                locked_identity = _cache_entry_identity(final)
                if locked_identity is None:
                    raise BootstrapRefusal("materialized runtime cache is unavailable")
                record = _validate_and_publish_cache(
                    cache_root=cache_root,
                    final=final,
                    identity=locked_identity,
                    manifest=manifest,
                    manifest_sha256=manifest_sha256,
                    decision_sha256=decision_sha256,
                    requirements_sha256=requirements_sha256,
                    governing_terms_sha256=governing_terms_sha256,
                )
            except Exception:
                shutil.rmtree(partial, ignore_errors=True)
                raise
    return {
        **record,
        "cache_path": str(final),
        "warm_reuse": warm_reuse,
        "governing_terms_fetched_this_invocation": not warm_reuse,
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_sha256 = _validate_manifest(
        Path(args.manifest), require_runtime_closure=False
    )
    _, requirements_sha256 = _validate_requirements(Path(args.requirements), manifest)
    cache_root = _validate_cache_root(Path(args.cache_root), None)
    final = cache_root / manifest_sha256
    identity = _cache_entry_identity(final)
    materialized = False
    if identity is not None:
        with _open_cache_entry(final, expected=identity) as stable_final:
            record = _load_json(stable_final / ".complete.json")
            decision_sha256 = str(record.get("decision_sha256") or "")
            if not _is_hex(decision_sha256, 64):
                raise BootstrapRefusal(
                    "existing runtime cache decision identity is invalid"
                )
            _validate_complete(
                stable_final,
                manifest,
                manifest_sha256,
                decision_sha256,
                requirements_sha256,
                _governing_terms_identity(manifest),
            )
            materialized = True
    return {
        "schema": "npa.libero.runtime-status.v1",
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "materialized": materialized,
        "source_revision": manifest["source"]["revision"],
    }


def _runtime_execution_environment(stable_root: Path) -> dict[str, str]:
    return {
        **{
            name: value
            for name, value in os.environ.items()
            if name in RUNTIME_EXECUTION_PASSTHROUGH_ENV_NAMES
        },
        "HOME": "/nonexistent",
        "LIBERO_RUNTIME_ROOT": str(stable_root),
    }


def execute() -> int:
    """Run the fixed smoke from one locked, descriptor-stable cache snapshot."""

    manifest, manifest_sha256 = _validate_manifest(DEFAULT_MANIFEST)
    _, requirements_sha256 = _validate_requirements(DEFAULT_REQUIREMENTS, manifest)
    decision_sha256 = os.environ.get(
        "NPA_LIBERO_RUNTIME_USE_DECISION_SHA256", ""
    ).strip()
    if not _is_hex(decision_sha256, 64):
        raise BootstrapRefusal("accepted decision SHA-256 is unavailable")
    cache_root = _validate_cache_root(DEFAULT_CACHE, None)
    final = cache_root / manifest_sha256
    identity = _cache_entry_identity(final)
    if identity is None:
        raise BootstrapRefusal("runtime cache is not materialized")
    current = cache_root / "current"
    if not current.is_symlink() or os.readlink(current) != final.name:
        raise BootstrapRefusal("runtime current link differs from the accepted cache")
    governing_terms_sha256 = _governing_terms_identity(manifest)
    lock_path = cache_root / ".bootstrap.lock"
    with lock_path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        descriptor = os.open(
            final,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != identity:
                raise BootstrapRefusal("runtime cache descriptor identity changed")
            stable_root = Path("/proc/self/fd") / str(descriptor)
            _validate_complete(
                stable_root,
                manifest,
                manifest_sha256,
                decision_sha256,
                requirements_sha256,
                governing_terms_sha256,
            )
            environment = _runtime_execution_environment(stable_root)
            completed = subprocess.run(
                ["/opt/npa/libero/smoke.sh"],
                check=False,
                env=environment,
                pass_fds=(descriptor,),
            )
            _validate_complete(
                stable_root,
                manifest,
                manifest_sha256,
                decision_sha256,
                requirements_sha256,
                governing_terms_sha256,
            )
            _require_cache_entry_identity(final, identity)
            if not current.is_symlink() or os.readlink(current) != final.name:
                raise BootstrapRefusal("runtime current link changed during execution")
            return completed.returncode
        finally:
            os.close(descriptor)


def _parse_utc(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise BootstrapRefusal(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise BootstrapRefusal(f"{label} must include a timezone")
    return parsed


def _storage_authorization(output_prefix: str, run_id: str) -> dict[str, Any]:
    encoded = os.environ.get("NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64", "").strip()
    expected = os.environ.get(
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256", ""
    ).strip()
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise BootstrapRefusal("output storage authorization is not valid Base64") from exc
    if not _is_hex(expected, 64) or hashlib.sha256(payload).hexdigest() != expected:
        raise BootstrapRefusal("output storage authorization is not manager-accepted")
    try:
        authorization = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal("output storage authorization is not valid JSON") from exc
    keys = {
        "schema",
        "issuer",
        "run_id",
        "output_prefix",
        "access_key_id_sha256",
        "secret_access_key_sha256",
        "session_token_sha256",
        "policy_sha256",
        "issued_at",
        "expires_at",
        "nonce",
    }
    if not isinstance(authorization, dict) or set(authorization) != keys:
        raise BootstrapRefusal("output storage authorization schema is not closed")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.environ.get("AWS_SESSION_TOKEN", "")
    prefix_sha256 = hashlib.sha256(output_prefix.encode()).hexdigest()
    issued_at = _parse_utc(authorization.get("issued_at"), "authorization issued_at")
    expires_at = _parse_utc(authorization.get("expires_at"), "authorization expires_at")
    now = datetime.now(timezone.utc)
    valid = (
        authorization.get("schema") == "npa.libero.output-storage-authorization.v1"
        and authorization.get("issuer") == "npa-manager"
        and authorization.get("run_id") == run_id
        and authorization.get("output_prefix") == output_prefix
        and prefix_sha256
        == os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_PREFIX_SHA256", "")
        and authorization.get("policy_sha256")
        == os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_POLICY_SHA256", "")
        and all((access_key, secret_key, session_token))
        and authorization.get("access_key_id_sha256")
        == hashlib.sha256(access_key.encode()).hexdigest()
        and authorization.get("secret_access_key_sha256")
        == hashlib.sha256(secret_key.encode()).hexdigest()
        and authorization.get("session_token_sha256")
        == hashlib.sha256(session_token.encode()).hexdigest()
        and issued_at <= now + timedelta(minutes=5)
        and issued_at < expires_at
        and expires_at > now
        and expires_at - issued_at <= timedelta(hours=24)
    )
    if not valid:
        raise BootstrapRefusal("output storage authorization is invalid or expired")
    return authorization


def _sigv4_request(
    method: str,
    url: str,
    *,
    payload: bytes = b"",
    extra_headers: dict[str, str] | None = None,
) -> tuple[dict[str, str], bytes]:
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    session_token = os.environ["AWS_SESSION_TOKEN"]
    region = os.environ.get("AWS_DEFAULT_REGION", "us-central1")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BootstrapRefusal("output storage endpoint must be a query-free HTTPS URL")
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    headers = {
        "host": parsed.netloc,
        "x-amz-content-sha256": payload_sha256,
        "x-amz-date": amz_date,
        "x-amz-security-token": session_token,
        **{key.lower(): value for key, value in (extra_headers or {}).items()},
    }
    signed_names = ";".join(sorted(headers))
    canonical_headers = "".join(
        f"{name}:{' '.join(headers[name].split())}\n" for name in sorted(headers)
    )
    canonical_uri = urllib.parse.quote(
        urllib.parse.unquote(parsed.path), safe="/-_.~"
    )
    if parsed.path != canonical_uri:
        raise BootstrapRefusal("output storage object path is not canonical")
    canonical_request = "\n".join(
        (
            method,
            canonical_uri,
            "",
            canonical_headers,
            signed_names,
            payload_sha256,
        )
    )
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        )
    )

    def sign(key: bytes, value: str) -> bytes:
        return hmac.new(key, value.encode(), hashlib.sha256).digest()

    signing_key = sign(
        sign(sign(sign(("AWS4" + secret_key).encode(), date_stamp), region), "s3"),
        "aws4_request",
    )
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    request_headers = {
        **headers,
        "authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_names}, Signature={signature}"
        ),
    }
    connection = http.client.HTTPSConnection(
        parsed.hostname, parsed.port or 443, timeout=120
    )
    target = urllib.parse.urlunsplit(("", "", canonical_uri or "/", "", ""))
    try:
        connection.request(
            method,
            target,
            body=payload if method == "PUT" else None,
            headers=request_headers,
        )
        response = connection.getresponse()
        try:
            if response.status != 200:
                raise BootstrapRefusal(
                    f"output storage {method} refused with HTTP {response.status}"
                )
            body = response.read(MAX_OUTPUT_BYTES + 1)
            response_headers = {
                key.lower(): value for key, value in response.getheaders()
            }
        finally:
            response.close()
    except (http.client.HTTPException, OSError) as exc:
        raise BootstrapRefusal(f"output storage {method} request failed") from exc
    finally:
        connection.close()
    if len(body) > MAX_OUTPUT_BYTES:
        raise BootstrapRefusal("output storage response exceeds the aggregate size budget")
    return response_headers, body


def _s3_object_url(endpoint: str, bucket: str, object_key: str) -> str:
    """Encode each S3 key segment while preserving separators on the wire."""

    segments = object_key.split("/")
    if (
        not bucket
        or bucket in {".", ".."}
        or not object_key
        or object_key.startswith("/")
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise BootstrapRefusal("output storage object identity is invalid")
    path = "/".join(
        urllib.parse.quote(segment, safe="-_.~")
        for segment in (bucket, *segments)
    )
    return f"{endpoint}/{path}"


def upload_outputs(smoke_exit_code: int) -> dict[str, Any]:
    """Upload through image-owned stdlib code after untrusted runtime execution ends."""

    run_id = os.environ.get("NPA_BYOF_RUN_ID", "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id):
        raise BootstrapRefusal("output upload requires the accepted run ID")
    output_prefix = os.environ.get("S3_OUTPUT_PREFIX", "").rstrip("/") + "/"
    parsed = urllib.parse.urlsplit(output_prefix)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise BootstrapRefusal("output prefix must be an S3 URI")
    _storage_authorization(output_prefix, run_id)
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT")
        or f"https://s3.{os.environ.get('AWS_DEFAULT_REGION', 'us-central1')}.amazonaws.com"
    ).rstrip("/")
    endpoint_parts = urllib.parse.urlsplit(endpoint)
    if endpoint_parts.scheme != "https" or not endpoint_parts.netloc:
        raise BootstrapRefusal("output storage endpoint must be HTTPS")
    root = Path(f"/workspace/byof-runs/{run_id}")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    summary = {
        "status": "success" if smoke_exit_code == 0 else "failed",
        "tool": "byof",
        "workload": "solution-smoke-libero-b200",
        "run_id": run_id,
        "image": os.environ.get("BYOF_IMAGE", ""),
        "solution_name": os.environ.get("BYOF_SOLUTION_NAME", ""),
        "capability_name": os.environ.get("BYOF_CAPABILITY_NAME", ""),
        "smoke_artifact_name": os.environ.get("BYOF_SMOKE_ARTIFACT_NAME", ""),
        "smoke_exit_code": smoke_exit_code,
        "runtime_cache_uploaded": False,
        "rendering_invoked": False,
        "created_unix": round(datetime.now(timezone.utc).timestamp(), 3),
    }
    summary_payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    summary_fd = os.open(
        "npa_byof_summary.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP,
        dir_fd=root_fd,
    )
    with os.fdopen(summary_fd, "wb") as stream:
        stream.write(summary_payload)

    def immutable_bytes(directory_fd: int, name: str, limit: int) -> tuple[bytes, str]:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise BootstrapRefusal("output violates its type, link, or size boundary")
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        with os.fdopen(file_fd, "rb") as stream:
            opened_before = os.fstat(stream.fileno())
            payload = stream.read(limit + 1)
            opened_after = os.fstat(stream.fileno())
        if len(payload) > limit or (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
        ) != (
            opened_before.st_dev,
            opened_before.st_ino,
            len(payload),
            opened_before.st_mtime_ns,
        ):
            raise BootstrapRefusal("output changed or exceeded its budget during snapshot")
        return payload, hashlib.sha256(payload).hexdigest()

    prefix = parsed.path.lstrip("/")

    def upload_and_read_back(name: str, payload: bytes, digest: str) -> dict[str, Any]:
        checksum = base64.b64encode(bytes.fromhex(digest)).decode()
        url = _s3_object_url(endpoint, parsed.netloc, prefix + name)
        _sigv4_request(
            "PUT",
            url,
            payload=payload,
            extra_headers={"if-none-match": "*", "x-amz-checksum-sha256": checksum},
        )
        headers, observed = _sigv4_request(
            "GET", url, extra_headers={"x-amz-checksum-mode": "ENABLED"}
        )
        if (
            observed != payload
            or hashlib.sha256(observed).hexdigest() != digest
            or headers.get("x-amz-checksum-sha256") != checksum
        ):
            raise BootstrapRefusal("output storage checksum/readback differs")
        return {"name": name, "size_bytes": len(payload), "sha256": digest}

    try:
        if set(os.listdir(root_fd)) != set(OUTPUT_SIZE_LIMITS):
            raise BootstrapRefusal("output differs from the exact artifact allowlist")
        observed_total = sum(
            os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_size
            for name in OUTPUT_SIZE_LIMITS
        )
        if observed_total > MAX_OUTPUT_BYTES:
            raise BootstrapRefusal("output exceeds the aggregate size budget")
        snapshots = []
        for name, limit in sorted(OUTPUT_SIZE_LIMITS.items()):
            payload, digest = immutable_bytes(root_fd, name, limit)
            snapshots.append((name, payload, digest))
        receipts = [
            upload_and_read_back(name, payload, digest)
            for name, payload, digest in snapshots
        ]
        receipt_payload = (
            json.dumps(
                {
                    "schema": "npa.libero.s3-upload-readback.v1",
                    "run_id": run_id,
                    "status": "verified",
                    "artifacts": receipts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        receipt_name = "npa_upload_receipt.json"
        receipt_fd = os.open(
            receipt_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IRUSR | stat.S_IWUSR,
            dir_fd=root_fd,
        )
        with os.fdopen(receipt_fd, "wb") as stream:
            stream.write(receipt_payload)
        receipt_bytes, receipt_sha256 = immutable_bytes(root_fd, receipt_name, 8 * 1024 * 1024)
        upload_and_read_back(receipt_name, receipt_bytes, receipt_sha256)
    finally:
        os.close(root_fd)
    return {"schema": "npa.libero.output-upload.v1", "status": "verified"}


def _run_output_root(run_id: str) -> Path:
    return Path(f"/workspace/byof-runs/{run_id}")


def _bootstrap_receipt_path(cache_root: Path, run_id: str) -> Path:
    return cache_root / "run-receipts" / f"{run_id}.json"


def _immutable_supervisor_bytes(path: Path, limit: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise BootstrapRefusal("supervisor evidence is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o027
        or len(payload) > limit
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or before.st_size != len(payload)
    ):
        raise BootstrapRefusal("supervisor evidence is mutable or invalid")
    return payload


def _execution_uid_processes() -> list[int]:
    try:
        execution_uid = 1001
        discovered = []
        for process in Path("/proc").iterdir():
            if not process.name.isdigit():
                continue
            try:
                status_lines = (process / "status").read_text(
                    encoding="utf-8"
                ).splitlines()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            uid_line = next(
                (line for line in status_lines if line.startswith("Uid:")), None
            )
            if uid_line is None:
                raise BootstrapRefusal("execution process identity is unavailable")
            fields = uid_line.split()
            if len(fields) != 5:
                raise BootstrapRefusal("execution process identity is invalid")
            if int(fields[2]) == execution_uid:
                discovered.append(int(process.name))
        return sorted(discovered)
    except OSError as exc:
        raise BootstrapRefusal("execution process inventory is unavailable") from exc


def _materialize_supervisor_artifact(
    root_fd: int, name: str, payload: bytes
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP,
        dir_fd=root_fd,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def execute_and_upload() -> int:
    """Hold the accepted cache snapshot lock through smoke output readback."""

    manifest, manifest_sha256 = _validate_manifest(DEFAULT_MANIFEST)
    _, requirements_sha256 = _validate_requirements(DEFAULT_REQUIREMENTS, manifest)
    decision_sha256 = os.environ.get(
        "NPA_LIBERO_RUNTIME_USE_DECISION_SHA256", ""
    ).strip()
    if not _is_hex(decision_sha256, 64):
        raise BootstrapRefusal("accepted decision SHA-256 is unavailable")
    cache_root = _validate_cache_root(DEFAULT_CACHE, None)
    final = cache_root / manifest_sha256
    identity = _cache_entry_identity(final)
    if identity is None:
        raise BootstrapRefusal("runtime cache is not materialized")
    current = cache_root / "current"
    governing_terms_sha256 = _governing_terms_identity(manifest)
    run_id = os.environ.get("NPA_BYOF_RUN_ID", "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id):
        raise BootstrapRefusal("execution requires the accepted run ID")
    output_root = _run_output_root(run_id)
    root_fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        root_info = os.fstat(root_fd)
        if (
            root_info.st_uid != os.getuid()
            or stat.S_IMODE(root_info.st_mode) != 0o1770
        ):
            raise BootstrapRefusal("execution output staging directory is invalid")
        bootstrap_receipt = _bootstrap_receipt_path(cache_root, run_id)
        bootstrap_payload = _immutable_supervisor_bytes(
            bootstrap_receipt, OUTPUT_SIZE_LIMITS["npa_runtime_bootstrap.json"]
        )
        if _execution_uid_processes():
            raise BootstrapRefusal("runtime execution UID is already active")
    except Exception:
        os.close(root_fd)
        raise
    lock_path = cache_root / ".bootstrap.lock"
    try:
        stdout_fd = os.open(
            "solution_smoke_stdout.log",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP,
            dir_fd=root_fd,
        )
        try:
            stderr_fd = os.open(
                "solution_smoke_stderr.log",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP,
                dir_fd=root_fd,
            )
        except Exception:
            os.close(stdout_fd)
            raise
        with lock_path.open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            descriptor = os.open(
                final,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != identity:
                    raise BootstrapRefusal("runtime cache descriptor identity changed")
                stable_root = Path("/proc/self/fd") / str(descriptor)
                _validate_complete(
                    stable_root,
                    manifest,
                    manifest_sha256,
                    decision_sha256,
                    requirements_sha256,
                    governing_terms_sha256,
                )
                runtime_metadata = _immutable_supervisor_bytes(
                    stable_root / ".complete.json",
                    OUTPUT_SIZE_LIMITS["npa_runtime_metadata.json"],
                )
                with os.fdopen(stdout_fd, "wb") as stdout, os.fdopen(
                    stderr_fd, "wb"
                ) as stderr:
                    environment = _runtime_execution_environment(final)
                    environment["NPA_LIBERO_BOOTSTRAP_RECEIPT"] = str(
                        bootstrap_receipt
                    )
                    completed = subprocess.run(
                        [
                            "sudo",
                            "--user=npa-libero-exec",
                            "/opt/npa/libero/runtime-bootstrap.py",
                            "execute",
                        ],
                        check=False,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                    )
                smoke_exit_code = completed.returncode
                remaining_processes = _execution_uid_processes()
                if remaining_processes:
                    raise BootstrapRefusal(
                        "runtime execution left processes behind: "
                        + ",".join(str(pid) for pid in remaining_processes)
                    )
                os.fchmod(root_fd, 0o700)
                _materialize_supervisor_artifact(
                    root_fd, "npa_runtime_bootstrap.json", bootstrap_payload
                )
                _materialize_supervisor_artifact(
                    root_fd, "npa_runtime_metadata.json", runtime_metadata
                )
                artifact_name = os.environ.get("BYOF_SMOKE_ARTIFACT_NAME", "")
                if artifact_name not in OUTPUT_SIZE_LIMITS or not (
                    output_root / artifact_name
                ).is_file():
                    with (output_root / "solution_smoke_stderr.log").open(
                        "a", encoding="utf-8"
                    ) as stderr:
                        stderr.write(
                            f"missing required smoke artifact: {artifact_name}\n"
                        )
                    smoke_exit_code = 1
                upload_outputs(smoke_exit_code)
                _validate_complete(
                    stable_root,
                    manifest,
                    manifest_sha256,
                    decision_sha256,
                    requirements_sha256,
                    governing_terms_sha256,
                )
                _require_cache_entry_identity(final, identity)
                if not current.is_symlink() or os.readlink(current) != final.name:
                    raise BootstrapRefusal(
                        "runtime current link changed before output readback completed"
                    )
                return smoke_exit_code
            finally:
                os.close(descriptor)
    finally:
        os.close(root_fd)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("ensure", "status", "execute", "execute-and-upload")
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--decision", default="")
    parser.add_argument("--decision-sha256", default="")
    parser.add_argument("--acceptance", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--smoke-exit-code", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "execute":
            return execute()
        if args.command == "execute-and-upload":
            return execute_and_upload()
        else:
            payload = ensure(args) if args.command == "ensure" else status(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "npa.libero.runtime-bootstrap-result.v1",
                    "status": "refused",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({**payload, "status": "ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
