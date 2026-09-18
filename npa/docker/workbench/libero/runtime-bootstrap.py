#!/usr/bin/env python3
"""Materialize the pinned LIBERO runtime into an operator-owned cache.

The public image contains this fetcher and immutable manifests, not LIBERO,
PyTorch/CUDA wheels, demonstrations, task assets, models, or populated caches.
Download success is never treated as permission: ``ensure`` refuses before the
first network or cache effect unless a customer/run authorization is present,
hash-bound, and signed directly by a customer-controlled key. The NPA control
plane authenticates the caller and transports and validates the evidence; it
does not accept or sign the customer's terms assertion.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import grp
import hashlib
import hmac
import http.client
import json
import os
import pwd
import re
import shutil
import signal
import ssl
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
from uuid import uuid4
from pathlib import Path
from typing import Any

PAYLOAD_RELEASE_ANNOTATION = "npa.nebius.com/libero-release"

SCHEMA = "npa.libero.runtime-manifest.v1"
CUSTOMER_AUTHORIZATION_SCHEMA = "npa.libero.customer-runtime-authorization.v2"
OUTPUT_STORAGE_AUTHORIZATION_SCHEMA = "npa.libero.output-storage-authorization.v3"
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
        "customer_runtime_authorization",
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
EXPECTED_CUSTOMER_AUTHORIZATION_METADATA = {
    "schema": CUSTOMER_AUTHORIZATION_SCHEMA,
    "required": True,
    "credentials_establish_acceptance": False,
}
EXPECTED_BOUNDARIES = {
    "cache": "operator-owned non-root atomic runtime cache",
    "outputs": "NPA_SMOKE_OUTPUT_DIR only",
    "rendering": False,
}
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "a319f5ac5fbd0eeb1390940eeda62828d621cd7dd0dd828789182024562c5b44"
)
EXPECTED_RUNTIME_REQUIREMENTS_SHA256 = (
    "8504f236dcad67ad0e2f5959b916c93aa7ccbd02567c6e323e480366d0f23b99"
)
DEFAULT_MANIFEST = Path("/opt/npa/libero/runtime-manifest.json")
DEFAULT_REQUIREMENTS = Path("/opt/npa/libero/runtime-requirements.txt")
DEFAULT_CACHE = Path("/workspace/.cache/npa/libero")
OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY = Path(
    "/opt/npa/libero/output-storage-authorization-public-key.b64"
)
OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_OWNER_UID = 0
CUSTOMER_AUTHORIZATION_NAMESPACE = b"npa.libero.customer-authorization"
OUTPUT_STORAGE_AUTHORIZATION_NAMESPACE = b"npa.libero.output-storage-authorization"
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
INHERITED_CACHE_DESCRIPTOR = 200
INHERITED_AUTHORIZATION_DESCRIPTOR = 201
INHERITED_EXECUTION_LOCK_DESCRIPTOR = 202
INHERITED_BOOTSTRAP_LOCK_DESCRIPTOR = 203
PROTECTED_AUTHORIZATION_NAME = ".npa-customer-authorization.json"
ALLOWED_TERMS_HOSTS = frozenset(
    {
        "raw.githubusercontent.com",
        "creativecommons.org",
        "www.apache.org",
        "docs.nvidia.com",
        "www.nvidia.com",
    }
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
RUNTIME_EXECUTION_USER = "npa-libero-exec"
RUNTIME_SUPERVISOR_USER = "ubuntu"
STORAGE_SECRET_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64",
    }
)
RUNTIME_MATERIALIZATION_PASSTHROUGH_ENV_NAMES = frozenset(
    {
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
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
        "NPA_BYOF_RUN_ID",
        "NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT",
        "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256",
        "NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256",
        "NPA_LIBERO_EXPECTED_ALLOWED_NODE_SHA256",
        "NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256",
        "NPA_LIBERO_EXPECTED_CLUSTER_IDENTITY_SHA256",
        "NPA_LIBERO_EXPECTED_EXECUTION_KUBECONFIG_SHA256",
        "NPA_LIBERO_EXPECTED_EXTERNAL_RBAC_INVENTORY_SHA256",
        "NPA_LIBERO_EXPECTED_INFRASTRUCTURE_BUNDLE_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_INVENTORY_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_UID_SHA256",
        "NPA_LIBERO_EXPECTED_OUTPUT_LEASE_ETAG",
        "NPA_LIBERO_EXPECTED_OUTPUT_LEASE_KEY",
        "NPA_LIBERO_EXPECTED_OUTPUT_LEASE_VERSION_ID",
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256",
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_POLICY_SHA256",
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_PREFIX_SHA256",
        "NPA_LIBERO_EXPECTED_PAYLOAD_KUBECONFIG_SHA256",
        "NPA_LIBERO_EXPECTED_PUBLICATION_BUNDLE_SHA256",
        "NPA_LIBERO_EXPECTED_RBAC_SPEC_SHA256",
        "NPA_LIBERO_EXPECTED_ROLE_BINDING_UID_SHA256",
        "NPA_LIBERO_EXPECTED_ROLE_UID_SHA256",
        "NPA_LIBERO_EXPECTED_SERVICE_ACCOUNT_UID_SHA256",
        "NPA_LIBERO_EXPECTED_SKYPILOT_CONFIG_SHA256",
        "NPA_LIBERO_BOOTSTRAP_RECEIPT",
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256",
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
        "NPA_SMOKE_OUTPUT_DIR",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TZ",
    }
)
OUTPUT_ARTIFACT_SIZE_LIMITS = {
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
# The commit receipt is deliberately not a workload artifact. Keep the
# workload allowlist separate so the receipt can only be created after every
# ordinary artifact has been snapshotted and uploaded.
OUTPUT_SIZE_LIMITS = OUTPUT_ARTIFACT_SIZE_LIMITS
FAILURE_OUTPUT_REQUIRED_SIZE_LIMITS = {
    name: limit
    for name, limit in OUTPUT_ARTIFACT_SIZE_LIMITS.items()
    if name not in {"libero-bc-rnn-smoke.pth", "libero-smoke.json"}
}
FAILURE_OUTPUT_OPTIONAL_SIZE_LIMITS = {
    name: OUTPUT_ARTIFACT_SIZE_LIMITS[name]
    for name in ("libero-bc-rnn-smoke.pth", "libero-smoke.json")
}
MAX_OUTPUT_BYTES = 320 * 1024 * 1024
OUTPUT_RECEIPT_NAME = "npa_upload_receipt.json"
OUTPUT_RECEIPT_SCHEMA = "npa.libero.s3-upload-readback.v2"


class BootstrapRefusal(RuntimeError):
    """A fail-closed identity, permission, or boundary refusal."""


class CustomerAcceptanceRequired(BootstrapRefusal):
    """The customer must use the authenticated acceptance surface."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason.replace("_", " "))


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


def _read_private_regular_bytes(path: Path, *, limit: int, input_name: str) -> bytes:
    """Read one stable owner-private file through a no-follow descriptor."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError as exc:
        raise CustomerAcceptanceRequired("authorization_missing") from exc
    except OSError as exc:
        raise BootstrapRefusal(f"{input_name} is unavailable or invalid") from exc
    try:
        return _read_private_regular_descriptor(
            descriptor,
            limit=limit,
            owner_uid=os.geteuid(),
            input_name=input_name,
        )
    finally:
        os.close(descriptor)


def _read_private_regular_descriptor(
    descriptor: int, *, limit: int, owner_uid: int, input_name: str
) -> bytes:
    """Read stable private bytes without changing an inherited descriptor offset."""

    try:
        before = os.fstat(descriptor)
        payload = os.pread(descriptor, limit + 1, 0)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise BootstrapRefusal(f"{input_name} descriptor is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != owner_uid
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o077
        or len(payload) > limit
        or before.st_size != len(payload)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise BootstrapRefusal(f"{input_name} is not stable owner-private")
    return payload


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
    if (
        manifest.get("customer_runtime_authorization")
        != EXPECTED_CUSTOMER_AUTHORIZATION_METADATA
    ):
        raise BootstrapRefusal(
            "customer runtime-authorization metadata is stale or invalid"
        )
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
            "name",
            "boundary",
            "version",
            "url",
            "size_bytes",
            "sha256",
        }:
            raise BootstrapRefusal("governing terms entry is not closed")
        term_id = str(term.get("id") or "")
        boundary = str(term.get("boundary") or "")
        if (
            term_id in term_ids
            or boundary
            not in {
                "source",
                "runtime_packages",
                "demonstration",
                "task_inputs",
                "language_model",
            }
            or not str(term.get("name") or "").strip()
            or term.get("version") != f"sha256:{term.get('sha256')}"
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
        if (
            name in names
            or filename in filenames
            or not name
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,254}", filename) is None
            or Path(filename).name != filename
            or filename in {".", ".."}
        ):
            raise BootstrapRefusal(
                "runtime artifact names and filenames must be unique"
            )
        names.add(name)
        filenames.add(filename)
        _validate_download_url(str(item.get("url") or ""))
        version = item.get("version")
        if (
            not isinstance(version, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]*", version) is None
        ):
            raise BootstrapRefusal(f"invalid runtime artifact version for {name}")
        if not _is_hex(item.get("sha256"), 64):
            raise BootstrapRefusal(f"invalid runtime artifact hash for {name}")
        size_bytes = item.get("size_bytes")
        license_expression = str(item.get("license_expression") or "").strip()
        if require_runtime_closure and (
            not isinstance(size_bytes, int) or size_bytes <= 0 or not license_expression
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


def wait_for_release() -> dict[str, str]:
    """Block all payload setup until the manager releases this exact Pod UID."""

    host = os.environ.get("KUBERNETES_SERVICE_HOST", "")
    port = int(os.environ.get("KUBERNETES_SERVICE_PORT", "443"))
    name = os.environ.get("HOSTNAME", "")
    service_account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    namespace = (service_account / "namespace").read_text(encoding="utf-8").strip()
    token = (service_account / "token").read_text(encoding="utf-8").strip()
    if not host or not name or not namespace or not token:
        raise BootstrapRefusal("payload release identity is incomplete")
    context = ssl.create_default_context(cafile=str(service_account / "ca.crt"))
    path = (
        "/api/v1/namespaces/"
        + urllib.parse.quote(namespace, safe="")
        + "/pods/"
        + urllib.parse.quote(name, safe="")
    )
    while True:
        connection = http.client.HTTPSConnection(
            host, port, timeout=10, context=context
        )
        try:
            connection.request(
                "GET", path, headers={"Authorization": f"Bearer {token}"}
            )
            response = connection.getresponse()
            payload = response.read(1024 * 1024)
            status = response.status
        finally:
            connection.close()
        if status == 403:
            time.sleep(1)
            continue
        if status != 200:
            raise BootstrapRefusal("payload release observation failed")
        try:
            pod = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapRefusal("payload release response is invalid") from exc
        metadata = pod.get("metadata") or {}
        uid = str(metadata.get("uid") or "")
        expected = hashlib.sha256(
            f"{uid}:{name}:npa-libero-release-v1".encode()
        ).hexdigest()
        observed = str(
            (metadata.get("annotations") or {}).get(PAYLOAD_RELEASE_ANNOTATION, "")
        )
        if (
            metadata.get("name") != name
            or metadata.get("namespace") != namespace
            or not uid
            or not hmac.compare_digest(observed, expected)
        ):
            time.sleep(1)
            continue
        return {"pod_uid_sha256": hashlib.sha256(uid.encode()).hexdigest()}
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
    raw = os.fspath(cache_root)
    if not os.path.isabs(raw) or os.path.normpath(raw) in {"/", "."}:
        raise BootstrapRefusal("runtime cache root must be a narrow absolute path")
    if any(part in {"", ".", ".."} for part in Path(raw).parts[1:]):
        raise BootstrapRefusal("runtime cache root must be canonical")
    resolved = Path(os.path.normpath(raw))
    if output_dir is not None:
        output = Path(os.path.normpath(os.fspath(output_dir)))
        if (
            output == resolved
            or output in resolved.parents
            or resolved in output.parents
        ):
            raise BootstrapRefusal("runtime cache and output boundaries overlap")
    return resolved


def _create_cache_root_component(
    parent_descriptor: int,
    name: str,
    before_create: Callable[[int, str], None] | None,
) -> int:
    """Create and bind the final root component without trusting a pathname."""

    if before_create is not None:
        before_create(parent_descriptor, name)
    try:
        os.mkdir(name, CACHE_ROOT_MODE, dir_fd=parent_descriptor)
    except FileExistsError:
        raise BootstrapRefusal("runtime cache root appeared during creation") from None
    except OSError as exc:
        raise BootstrapRefusal("runtime cache root could not be created") from exc
    child: int | None = None
    try:
        created = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        child = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(child)
    except OSError as exc:
        if child is not None:
            os.close(child)
        raise BootstrapRefusal("runtime cache root changed during creation") from exc
    if not stat.S_ISDIR(created.st_mode) or (created.st_dev, created.st_ino) != (
        opened.st_dev,
        opened.st_ino,
    ):
        os.close(child)
        raise BootstrapRefusal("runtime cache root changed during creation")
    return child


@contextmanager
def _open_cache_root_descriptor(
    cache_root: Path,
    *,
    create: bool,
    owner_uid: int | None = None,
    before_create: Callable[[int, str], None] | None = None,
) -> Iterator[int]:
    """Walk an absolute cache root without following any path component."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise BootstrapRefusal("runtime cache requires no-follow directory support")
    parts = cache_root.parts
    descriptor = os.open("/", os.O_RDONLY | directory | nofollow | os.O_CLOEXEC)
    try:
        for index, part in enumerate(parts[1:], start=1):
            final = index == len(parts) - 1
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | directory | nofollow | os.O_CLOEXEC,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not (create and final):
                    raise BootstrapRefusal(
                        "runtime cache root is unavailable"
                    ) from None
                child = _create_cache_root_component(descriptor, part, before_create)
            except OSError as exc:
                raise BootstrapRefusal(
                    "runtime cache root contains a link or invalid component"
                ) from exc
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        expected_owner = os.getuid() if owner_uid is None else owner_uid
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_nlink < 1
            or info.st_uid != expected_owner
        ):
            raise BootstrapRefusal("runtime cache root ownership or links are invalid")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _cache_lock(
    cache_root_descriptor: int,
    name: str,
    *,
    exclusive: bool,
    create: bool,
) -> Iterator[int]:
    """Open and hold one validated cache lock through the retained root descriptor."""

    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(name, flags, 0o640, dir_fd=cache_root_descriptor)
    except OSError as exc:
        raise BootstrapRefusal("runtime cache lock is unavailable") from exc
    try:
        root = os.fstat(cache_root_descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != root.st_uid
        ):
            raise BootstrapRefusal("runtime cache lock identity is invalid")
        if create:
            try:
                group_id = grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid
            except KeyError as exc:
                raise BootstrapRefusal(
                    "runtime execution group is unavailable"
                ) from exc
            os.fchown(descriptor, -1, group_id)
            os.fchmod(descriptor, 0o640)
        elif stat.S_IMODE(opened.st_mode) != 0o640:
            raise BootstrapRefusal("runtime cache lock mode is invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield descriptor
    finally:
        os.close(descriptor)


def _require_inherited_lock(
    cache_root_descriptor: int, name: str, inherited_descriptor: int
) -> None:
    """Bind an inherited lock descriptor to the exact retained cache root."""

    try:
        named = os.stat(name, dir_fd=cache_root_descriptor, follow_symlinks=False)
        inherited = os.fstat(inherited_descriptor)
    except OSError as exc:
        raise BootstrapRefusal("inherited runtime lock is unavailable") from exc
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) != 0o640
        or (named.st_dev, named.st_ino) != (inherited.st_dev, inherited.st_ino)
    ):
        raise BootstrapRefusal("inherited runtime lock identity is invalid")


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


def _cache_entry_identity_at(
    parent_descriptor: int, name: str
) -> tuple[int, int] | None:
    """Inspect one no-follow cache child through the trusted parent descriptor."""

    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
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


def _remove_cache_tree(path: Path) -> list[str]:
    """Best-effort no-follow removal that attempts every discovered descendant."""

    errors: list[str] = []
    try:
        descendants = sorted(
            path.rglob("*"),
            key=lambda item: len(item.relative_to(path).parts),
            reverse=True,
        )
    except Exception:  # noqa: BLE001 - rollback records every ambiguity
        return ["descendant inventory failed"]
    for item in (path, *reversed(descendants)):
        try:
            info = item.lstat()
            if stat.S_ISDIR(info.st_mode):
                os.chmod(item, 0o700)
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - continue with independent descendants
            errors.append(f"chmod failed:{item.name}")
    for item in descendants:
        try:
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or stat.S_ISREG(info.st_mode):
                item.unlink()
            elif stat.S_ISDIR(info.st_mode):
                item.rmdir()
            else:
                errors.append(f"unsupported descendant:{item.name}")
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - attempt the rest before reporting
            errors.append(f"remove failed:{item.name}")
    try:
        path.rmdir()
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - caller retries once after child failures
        errors.append(f"remove failed:{path.name}")
    return errors


def _discard_new_cache_entry(
    cache_root: Path,
    final: Path,
    expected: tuple[int, int],
    *,
    parent_descriptor: int,
) -> list[str]:
    """Isolate and remove only the exact entry created by this transaction."""

    errors: list[str] = []
    quarantine_root: Path | None = None
    isolated: Path | None = None
    try:
        observed = _cache_entry_identity_at(parent_descriptor, final.name)
    except Exception:  # noqa: BLE001 - identity ambiguity is a fail-closed result
        errors.append("final identity acquisition failed")
        observed = None
    if observed is None:
        return errors
    if observed != expected:
        errors.append("final identity drifted")
        return errors
    try:
        quarantine_root = Path(
            tempfile.mkdtemp(
                prefix=f".{final.name}.failed-{uuid4().hex}-", dir=cache_root
            )
        )
        os.chmod(quarantine_root, 0o700)
        isolated = quarantine_root / "entry"
        os.rename(
            final.name,
            f"{quarantine_root.name}/entry",
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
    except Exception:  # noqa: BLE001 - direct exact-inode removal is the fallback
        errors.append("quarantine creation or rename failed")
        if quarantine_root is not None:
            _remove_cache_tree(quarantine_root)
        try:
            if _cache_entry_identity_at(parent_descriptor, final.name) == expected:
                isolated = final
            else:
                return errors
        except Exception:  # noqa: BLE001 - never remove an unverified replacement
            errors.append("fallback identity acquisition failed")
            return errors
    assert isolated is not None
    try:
        if _cache_entry_identity(isolated) != expected:
            errors.append("isolated cache identity drifted")
            return errors
    except Exception:  # noqa: BLE001 - retain isolated evidence on ambiguity
        errors.append("isolated cache identity acquisition failed")
        return errors
    first_pass = _remove_cache_tree(isolated)
    if isolated.exists():
        errors.extend(first_pass)
        errors.extend(_remove_cache_tree(isolated))
    if quarantine_root is not None and quarantine_root.exists():
        errors.extend(_remove_cache_tree(quarantine_root))
    return errors


def _rollback_new_cache_entry(
    cache_root: Path,
    final: Path,
    expected: tuple[int, int],
    *,
    parent_descriptor: int,
) -> tuple[str, ...]:
    """Independently retract publication and the renamed entry, aggregating errors."""

    errors: list[str] = []
    for _attempt in range(2):
        try:
            _remove_current_cache_link(cache_root, final)
        except Exception:  # noqa: BLE001 - entry cleanup must still run
            errors.append("current-link removal failed")
        errors.extend(
            _discard_new_cache_entry(
                cache_root,
                final,
                expected,
                parent_descriptor=parent_descriptor,
            )
        )
    try:
        remaining = _cache_entry_identity_at(parent_descriptor, final.name)
    except Exception:  # noqa: BLE001
        errors.append("final absence check failed")
    else:
        if remaining == expected:
            errors.append("renamed cache entry remains")
        elif remaining is not None:
            errors.append("different cache entry occupies final name")
    current = cache_root / "current"
    try:
        if current.is_symlink() and os.readlink(current) == final.name:
            errors.append("current link remains")
    except Exception:  # noqa: BLE001
        errors.append("current-link absence check failed")
    return tuple(dict.fromkeys(errors))


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _canonical_unsigned_customer_authorization(payload: dict[str, Any]) -> bytes:
    unsigned = json.loads(json.dumps(payload))
    unsigned.pop("signature", None)
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def _customer_authorization_signature_payload(payload: dict[str, Any]) -> bytes:
    canonical = _canonical_unsigned_customer_authorization(payload)
    return _sshsig_signature_payload(CUSTOMER_AUTHORIZATION_NAMESPACE, canonical)


def _sshsig_signature_payload(namespace: bytes, canonical: bytes) -> bytes:
    return b"".join(
        (
            b"SSHSIG",
            _ssh_string(namespace),
            _ssh_string(b""),
            _ssh_string(b"sha512"),
            _ssh_string(hashlib.sha512(canonical).digest()),
        )
    )


def _customer_authorization_sshsig(public_key: bytes, signature: bytes) -> bytes:
    return _sshsig_envelope(CUSTOMER_AUTHORIZATION_NAMESPACE, public_key, signature)


def _sshsig_envelope(namespace: bytes, public_key: bytes, signature: bytes) -> bytes:
    public_key_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public_key)
    signature_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(signature)
    payload = b"".join(
        (
            b"SSHSIG",
            struct.pack(">I", 1),
            _ssh_string(public_key_blob),
            _ssh_string(namespace),
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


def _trusted_public_key(path: Path, *, owner_uid: int, label: str) -> bytes:
    """Load one immutable image-baked Ed25519 verification key."""

    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise BootstrapRefusal(f"{label} trust root is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            encoded = stream.read()
        after = os.fstat(descriptor)
    except OSError as exc:
        raise BootstrapRefusal(f"{label} trust root is unavailable") from exc
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != owner_uid
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o444
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
        )
        or encoded != encoded.strip()
    ):
        raise BootstrapRefusal(f"{label} trust root is mutable or invalid")
    try:
        public_key = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise BootstrapRefusal(f"{label} trust root is invalid") from exc
    if len(public_key) != 32:
        raise BootstrapRefusal(f"{label} trust root is invalid")
    return public_key


def _trusted_output_storage_authorization_public_key() -> bytes:
    storage_key = _trusted_public_key(
        OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY,
        owner_uid=OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_OWNER_UID,
        label="output-storage-authorization",
    )
    customer_fingerprint = os.environ.get(
        "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256", ""
    )
    if not _is_hex(customer_fingerprint, 64) or hmac.compare_digest(
        customer_fingerprint, hashlib.sha256(storage_key).hexdigest()
    ):
        raise BootstrapRefusal("output-storage trust root is not independent")
    return storage_key


def _verify_customer_authorization_signature(
    payload: dict[str, Any], signature_record: dict[str, Any]
) -> None:
    try:
        public_key = base64.b64decode(
            str(payload.get("customer_signer_public_key_b64") or ""), validate=True
        )
    except ValueError as exc:
        raise BootstrapRefusal("customer signer identity is invalid") from exc
    claimed_fingerprint = str(signature_record.get("public_key_sha256") or "")
    expected_fingerprint = os.environ.get(
        "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256", ""
    )
    if (
        len(public_key) != 32
        or not _is_hex(claimed_fingerprint, 64)
        or hashlib.sha256(public_key).hexdigest() != claimed_fingerprint
        or not hmac.compare_digest(claimed_fingerprint, expected_fingerprint)
    ):
        raise BootstrapRefusal("customer signer identity differs")
    try:
        signature = base64.b64decode(
            str(signature_record.get("signature_b64") or ""), validate=True
        )
    except ValueError as exc:
        raise BootstrapRefusal("customer authorization signature is invalid") from exc
    if len(signature) != 64:
        raise BootstrapRefusal("customer authorization signature is invalid")
    canonical = _canonical_unsigned_customer_authorization(payload)
    public_key_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public_key)
    allowed_signer = (
        "customer ssh-ed25519 "
        + base64.b64encode(public_key_blob).decode("ascii")
        + "\n"
    )
    with tempfile.TemporaryDirectory(
        prefix="npa-libero-customer-authorization-signature-"
    ) as root:
        root_path = Path(root)
        allowed_path = root_path / "allowed-signers"
        signature_path = root_path / "authorization.sig"
        allowed_path.write_text(allowed_signer, encoding="ascii")
        signature_path.write_bytes(
            _customer_authorization_sshsig(public_key, signature)
        )
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
                "customer",
                "-n",
                CUSTOMER_AUTHORIZATION_NAMESPACE.decode("ascii"),
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
        raise BootstrapRefusal("customer authorization signature is invalid")


def _verify_output_storage_authorization_signature(
    payload: dict[str, Any], signature_record: dict[str, Any]
) -> None:
    public_key = _trusted_output_storage_authorization_public_key()
    fingerprint = hashlib.sha256(public_key).hexdigest()
    if signature_record.get("public_key_sha256") != fingerprint:
        raise BootstrapRefusal("output storage authorization trust root differs")
    try:
        signature = base64.b64decode(
            str(signature_record.get("signature_b64") or ""), validate=True
        )
    except ValueError as exc:
        raise BootstrapRefusal(
            "output storage authorization signature is invalid"
        ) from exc
    if len(signature) != 64:
        raise BootstrapRefusal("output storage authorization signature is invalid")
    unsigned = json.loads(json.dumps(payload))
    unsigned.pop("signature", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    allowed_signer = (
        "npa-output-storage-control-plane ssh-ed25519 "
        + base64.b64encode(
            _ssh_string(b"ssh-ed25519") + _ssh_string(public_key)
        ).decode("ascii")
        + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="npa-libero-storage-signature-") as root:
        allowed_path = Path(root) / "allowed-signers"
        signature_path = Path(root) / "authorization.sig"
        allowed_path.write_text(allowed_signer, encoding="ascii")
        signature_path.write_bytes(
            _sshsig_envelope(
                OUTPUT_STORAGE_AUTHORIZATION_NAMESPACE, public_key, signature
            )
        )
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
                "npa-output-storage-control-plane",
                "-n",
                OUTPUT_STORAGE_AUTHORIZATION_NAMESPACE.decode("ascii"),
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
        raise BootstrapRefusal("output storage authorization signature is invalid")


def _customer_acceptance_notification(
    manifest: dict[str, Any], *, manifest_sha256: str, reason: str
) -> dict[str, Any]:
    customer_identity = os.environ.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", "")
    return {
        "schema": "npa.libero.customer-acceptance-notification.v1",
        "status": "needs_customer_acceptance",
        "solution": "libero",
        "reason": reason,
        "runtime_manifest_sha256": manifest_sha256,
        "candidate_image": os.environ.get("BYOF_IMAGE") or None,
        "run_id": os.environ.get("NPA_BYOF_RUN_ID") or None,
        "customer_identity": (
            f"sha256:{customer_identity[:12]}…"
            if _is_hex(customer_identity, 64)
            else None
        ),
        "terms": [
            {
                "id": term["id"],
                "name": term["name"],
                "official_url": term["url"],
                "version": term["version"],
            }
            for term in manifest["governing_terms"]
        ],
        "acknowledgement": {
            "required": True,
            "instructions": (
                "Review every listed official term and authorize this exact "
                "customer and run using a customer-controlled signing key. "
                "The authenticated NPA surface transports and validates that "
                "evidence but does not accept or sign the terms assertion."
            ),
            "refusal": (
                "Decline or omit authorization to stop before runtime fetch, "
                "installation, cache mutation, or workload execution."
            ),
        },
        "credentials": {
            "purpose": "upstream_access_only",
            "establish_terms_acceptance": False,
        },
    }


def _validate_customer_authorization(
    path: Path,
    expected_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify a customer/run authorization before all external effects."""

    if not _is_hex(expected_sha256, 64):
        raise CustomerAcceptanceRequired("authorization_missing")
    try:
        authorization_bytes = _read_private_regular_bytes(
            path,
            limit=1024 * 1024,
            input_name="customer authorization file",
        )
    except CustomerAcceptanceRequired:
        raise
    except BootstrapRefusal as exc:
        raise CustomerAcceptanceRequired("authorization_file_invalid") from exc
    return _validate_customer_authorization_bytes(
        authorization_bytes, expected_sha256, manifest, manifest_sha256
    )


def _validate_customer_authorization_bytes(
    authorization_bytes: bytes,
    expected_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Validate exact signed bytes, classifying customer-actionable failures."""

    observed_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise CustomerAcceptanceRequired("authorization_hash_mismatch")
    try:
        authorization = json.loads(authorization_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CustomerAcceptanceRequired("authorization_malformed") from exc
    if not isinstance(authorization, dict):
        raise CustomerAcceptanceRequired("authorization_malformed")
    signature_record = authorization.get("signature")
    expected_keys = {
        "schema",
        "solution",
        "status",
        "authorization_id",
        "customer_identity_sha256",
        "run_id",
        "candidate_image",
        "runtime_manifest_sha256",
        "workflow_profile_sha256",
        "upstream_source_revision",
        "terms",
        "issuer",
        "evidence_type",
        "customer_signer_public_key_b64",
        "acknowledged_at",
        "issued_at",
        "expires_at",
        "nonce",
        "signature",
    }
    expected_terms = [
        {"id": term["id"], "version": term["version"]}
        for term in manifest["governing_terms"]
    ]
    if (
        set(authorization) != expected_keys
        or authorization.get("schema") != CUSTOMER_AUTHORIZATION_SCHEMA
        or authorization.get("solution") != "libero"
        or authorization.get("status") not in {"authorized", "denied"}
        or re.fullmatch(
            r"[a-z0-9][a-z0-9-]{15,79}",
            str(authorization.get("authorization_id") or ""),
        )
        is None
        or not _is_hex(authorization.get("customer_identity_sha256"), 64)
        or authorization.get("customer_identity_sha256")
        != os.environ.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256")
        or authorization.get("run_id") != os.environ.get("NPA_BYOF_RUN_ID")
        or re.fullmatch(
            r"[a-z0-9][a-z0-9-]{15,62}",
            str(authorization.get("run_id") or ""),
        )
        is None
        or authorization.get("candidate_image") != os.environ.get("BYOF_IMAGE")
        or re.fullmatch(
            r"ghcr\.io/nebius/nebius-physical-ai/npa-libero@sha256:[0-9a-f]{64}",
            str(authorization.get("candidate_image") or ""),
        )
        is None
        or authorization.get("runtime_manifest_sha256") != manifest_sha256
        or re.fullmatch(
            r"[0-9a-f]{64}", str(authorization.get("workflow_profile_sha256") or "")
        ) is None
        or authorization.get("workflow_profile_sha256")
        != os.environ.get("NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256")
        or authorization.get("upstream_source_revision")
        != (manifest.get("source") or {}).get("revision")
        or authorization.get("terms") != expected_terms
        or authorization.get("issuer") != "customer"
        or authorization.get("evidence_type") != "customer-controlled-signature"
        or not isinstance(signature_record, dict)
        or set(signature_record) != {"algorithm", "public_key_sha256", "signature_b64"}
        or signature_record.get("algorithm") != "ed25519"
        or re.fullmatch(r"[A-Za-z0-9_-]{32,128}", str(authorization.get("nonce") or ""))
        is None
    ):
        raise CustomerAcceptanceRequired("authorization_wrong_scope")
    try:
        acknowledged_at = datetime.fromisoformat(
            str(authorization["acknowledged_at"]).replace("Z", "+00:00")
        )
        issued_at = datetime.fromisoformat(
            str(authorization["issued_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(authorization["expires_at"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CustomerAcceptanceRequired("authorization_timestamps_invalid") from exc
    now = datetime.now(timezone.utc)
    if (
        acknowledged_at.tzinfo is None
        or issued_at.tzinfo is None
        or expires_at.tzinfo is None
        or acknowledged_at > issued_at
        or issued_at - acknowledged_at > timedelta(minutes=5)
        or issued_at > now + timedelta(minutes=5)
        or issued_at >= expires_at
        or expires_at <= now
        or expires_at - issued_at > timedelta(hours=24)
    ):
        raise CustomerAcceptanceRequired("authorization_expired_or_replayable")
    expected_expires_raw = os.environ.get(
        "NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT", ""
    )
    try:
        expected_expires_at = datetime.fromisoformat(
            expected_expires_raw.replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise CustomerAcceptanceRequired("authorization_expected_expiry_invalid") from exc
    if (
        expected_expires_at.tzinfo is None
        or expires_at != expected_expires_at
    ):
        raise CustomerAcceptanceRequired("authorization_expected_expiry_mismatch")
    try:
        _verify_customer_authorization_signature(authorization, signature_record)
    except BootstrapRefusal as exc:
        if str(exc) in {
            "customer-authorization trust root is unavailable",
            "customer-authorization trust root is mutable or invalid",
            "customer-authorization trust root is invalid",
        }:
            raise
        raise CustomerAcceptanceRequired("authorization_signature_invalid") from exc
    if authorization["status"] == "denied":
        raise CustomerAcceptanceRequired("authorization_denied")
    return authorization, observed_sha256


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
            "PATH": "/usr/bin:/bin",
        }
    )
    return environment


def _fetch_source(root: Path, source: dict[str, Any]) -> None:
    destination = root / "source"
    destination.mkdir(mode=0o700)
    environment = _runtime_materialization_environment(root)
    _run(["/usr/bin/git", "init", "--quiet"], cwd=destination, environment=environment)
    _run(
        ["/usr/bin/git", "remote", "add", "origin", source["repository"]],
        cwd=destination,
        environment=environment,
    )
    _run(
        ["/usr/bin/git", "config", "remote.origin.promisor", "true"],
        cwd=destination,
        environment=environment,
    )
    _run(
        ["/usr/bin/git", "config", "remote.origin.partialclonefilter", "blob:none"],
        cwd=destination,
        environment=environment,
    )
    _run(
        [
            "/usr/bin/git",
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
        ["/usr/bin/git", "rev-parse", "FETCH_HEAD^{commit}"],
        cwd=destination,
        env=environment,
        text=True,
    ).strip()
    tree = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "FETCH_HEAD^{tree}"],
        cwd=destination,
        env=environment,
        text=True,
    ).strip()
    if fetched != source["revision"] or tree != source["tree"]:
        raise BootstrapRefusal(
            "fetched LIBERO source identity differs from the manifest"
        )
    _run(
        ["/usr/bin/git", "sparse-checkout", "init", "--no-cone"],
        cwd=destination,
        environment=environment,
    )
    _run(
        [
            "/usr/bin/git",
            "sparse-checkout",
            "set",
            "--no-cone",
            "--",
            *source["sparse_paths"],
        ],
        cwd=destination,
        environment=environment,
    )
    _run(
        ["/usr/bin/git", "checkout", "--quiet", "--detach", fetched],
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
    *,
    published_root: Path,
) -> None:
    materialization_environment = _runtime_materialization_environment(root)
    wheelhouse = root / "downloads"
    wheelhouse.mkdir(mode=0o700)
    resolved_wheelhouse = wheelhouse.resolve(strict=True)
    for item in artifacts:
        destination = wheelhouse / item["filename"]
        if destination.resolve(strict=False).parent != resolved_wheelhouse:
            raise BootstrapRefusal("runtime artifact destination leaves the wheelhouse")
        _download_verified(
            destination,
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
        "--only-binary=:all:",
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
        str(published_root / "source") + "\n", encoding="utf-8"
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
                SEALED_EXECUTABLE_MODE if info.st_mode & 0o111 else SEALED_REGULAR_MODE,
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
    authorization_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
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
        "customer_authorization_sha256": authorization_sha256,
        "customer_identity_sha256": customer_identity_sha256,
        "run_id": run_id,
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
    authorization_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    _validate_read_only_tree(root)
    record = _load_json(root / ".complete.json")
    expected = _complete_record_values(
        manifest,
        manifest_sha256,
        authorization_sha256,
        customer_identity_sha256,
        run_id,
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
    authorization_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    try:
        with _open_cache_entry(final, expected=identity) as stable_final:
            record = _validate_complete(
                stable_final,
                manifest,
                manifest_sha256,
                authorization_sha256,
                customer_identity_sha256,
                run_id,
                requirements_sha256,
                governing_terms_sha256,
            )
            _publish_current_cache_link(cache_root, final, identity)
    except Exception:
        _remove_current_cache_link(cache_root, final)
        raise
    return record


def _complete_record_values(
    manifest: dict[str, Any],
    manifest_sha256: str,
    authorization_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    source = manifest["source"]
    demonstration = manifest["demonstration"]
    return {
        "schema": COMPLETE_SCHEMA,
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "customer_authorization_sha256": authorization_sha256,
        "customer_identity_sha256": customer_identity_sha256,
        "run_id": run_id,
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


def _runtime_cache_scope_sha256(
    manifest_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    authorization_sha256: str,
) -> str:
    if (
        not _is_hex(manifest_sha256, 64)
        or not _is_hex(customer_identity_sha256, 64)
        or not _is_hex(authorization_sha256, 64)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id) is None
    ):
        raise BootstrapRefusal("runtime cache scope is invalid")
    return hashlib.sha256(
        json.dumps(
            {
                "schema": "npa.libero.customer-runtime-cache-scope.v2",
                "customer_authorization_sha256": authorization_sha256,
                "customer_identity_sha256": customer_identity_sha256,
                "run_id": run_id,
                "runtime_manifest_sha256": manifest_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def ensure(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_sha256 = _validate_manifest(Path(args.manifest))
    authorization, authorization_sha256 = _validate_customer_authorization(
        Path(args.authorization), args.authorization_sha256, manifest, manifest_sha256
    )
    customer_identity_sha256 = authorization["customer_identity_sha256"]
    run_id = authorization["run_id"]
    requirement_lines, requirements_sha256 = _validate_requirements(
        Path(args.requirements), manifest
    )
    output = Path(args.output_dir) if args.output_dir else None
    cache_root = _validate_cache_root(Path(args.cache_root), output)
    scope_sha256 = _runtime_cache_scope_sha256(
        manifest_sha256,
        customer_identity_sha256,
        run_id,
        authorization_sha256,
    )
    display_final = cache_root / scope_sha256
    governing_terms_sha256: str | None = None
    cache_root_created = False

    def verify_terms_before_cache_creation(_parent_descriptor: int, _name: str) -> None:
        nonlocal cache_root_created, governing_terms_sha256
        governing_terms_sha256 = _verify_governing_terms(manifest)
        cache_root_created = True

    with _open_cache_root_descriptor(
        cache_root,
        create=True,
        before_create=verify_terms_before_cache_creation,
    ) as root_fd:
        initial_identity = _cache_entry_identity_at(root_fd, scope_sha256)
        if cache_root_created and initial_identity is not None:
            raise BootstrapRefusal("runtime cache entry appeared during creation")
        if initial_identity is not None:
            governing_terms_sha256 = _governing_terms_identity(manifest)
        elif governing_terms_sha256 is None:
            governing_terms_sha256 = _verify_governing_terms(manifest)
        try:
            group_id = grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid
        except KeyError as exc:
            raise BootstrapRefusal("runtime execution group is unavailable") from exc
        os.fchown(root_fd, -1, group_id)
        os.fchmod(root_fd, CACHE_ROOT_MODE)
        stable_cache_root = Path("/proc/self/fd") / str(root_fd)
        final = stable_cache_root / scope_sha256
        assert governing_terms_sha256 is not None
        with _cache_lock(root_fd, ".bootstrap.lock", exclusive=True, create=True):
            locked_identity = _cache_entry_identity_at(root_fd, scope_sha256)
            if locked_identity != initial_identity:
                raise BootstrapRefusal("runtime cache entry changed before validation")
            warm_reuse = locked_identity is not None
            if warm_reuse:
                assert locked_identity is not None
                record = _validate_and_publish_cache(
                    cache_root=stable_cache_root,
                    final=final,
                    identity=locked_identity,
                    manifest=manifest,
                    manifest_sha256=manifest_sha256,
                    authorization_sha256=authorization_sha256,
                    customer_identity_sha256=customer_identity_sha256,
                    run_id=run_id,
                    requirements_sha256=requirements_sha256,
                    governing_terms_sha256=governing_terms_sha256,
                )
            else:
                partial = Path(
                    tempfile.mkdtemp(
                        prefix=f".{manifest_sha256}.partial-", dir=stable_cache_root
                    )
                )
                os.chmod(partial, 0o700)
                renamed = False
                committed = False
                renamed_identity: tuple[int, int] | None = None
                try:
                    _fetch_source(partial, manifest["source"])
                    _validate_task_inputs(partial, manifest)
                    _install_runtime(
                        partial,
                        manifest["runtime_artifacts"],
                        requirement_lines,
                        published_root=display_final,
                    )
                    _fetch_inputs(partial, manifest)
                    record = _complete_record(
                        partial,
                        manifest,
                        manifest_sha256,
                        authorization_sha256,
                        customer_identity_sha256,
                        run_id,
                        requirements_sha256,
                        governing_terms_sha256,
                    )
                    _seal_cache_tree(partial)
                    renamed_identity = _cache_entry_identity(partial)
                    if renamed_identity is None:
                        raise BootstrapRefusal(
                            "materialized runtime cache is unavailable"
                        )
                    os.rename(
                        partial.name,
                        scope_sha256,
                        src_dir_fd=root_fd,
                        dst_dir_fd=root_fd,
                    )
                    renamed = True
                    locked_identity = _cache_entry_identity_at(root_fd, scope_sha256)
                    if locked_identity != renamed_identity:
                        raise BootstrapRefusal(
                            "materialized runtime cache is unavailable"
                        )
                    record = _validate_and_publish_cache(
                        cache_root=stable_cache_root,
                        final=final,
                        identity=locked_identity,
                        manifest=manifest,
                        manifest_sha256=manifest_sha256,
                        authorization_sha256=authorization_sha256,
                        customer_identity_sha256=customer_identity_sha256,
                        run_id=run_id,
                        requirements_sha256=requirements_sha256,
                        governing_terms_sha256=governing_terms_sha256,
                    )
                    committed = True
                except Exception as exc:
                    rollback_errors: tuple[str, ...] = ()
                    if renamed and not committed and renamed_identity is not None:
                        rollback_errors = _rollback_new_cache_entry(
                            stable_cache_root,
                            final,
                            renamed_identity,
                            parent_descriptor=root_fd,
                        )
                    partial_errors = _remove_cache_tree(partial)
                    if partial.exists():
                        partial_errors.append("partial cache remains")
                    cleanup_errors = (*rollback_errors, *partial_errors)
                    if cleanup_errors:
                        cleanup_kind = (
                            "runtime cache rollback"
                            if rollback_errors
                            else "runtime cache partial cleanup"
                        )
                        raise BootstrapRefusal(
                            f"{cleanup_kind} was incomplete: "
                            + ", ".join(dict.fromkeys(cleanup_errors))
                        ) from exc
                    raise
    return {
        **record,
        "cache_path": str(display_final),
        "warm_reuse": warm_reuse,
        "governing_terms_fetched_this_invocation": not warm_reuse,
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_sha256 = _validate_manifest(
        Path(args.manifest), require_runtime_closure=False
    )
    _, requirements_sha256 = _validate_requirements(Path(args.requirements), manifest)
    cache_root = _validate_cache_root(Path(args.cache_root), None)
    materialized = False
    current = cache_root / "current"
    if current.is_symlink():
        target = os.readlink(current)
        if not _is_hex(target, 64):
            raise BootstrapRefusal("runtime current link has an invalid scope")
        final = cache_root / target
        identity = _cache_entry_identity(final)
        if identity is None:
            raise BootstrapRefusal("runtime current link target is unavailable")
        with _open_cache_entry(final, expected=identity) as stable_final:
            record = _load_json(stable_final / ".complete.json")
            authorization_sha256 = str(
                record.get("customer_authorization_sha256") or ""
            )
            customer_identity_sha256 = str(record.get("customer_identity_sha256") or "")
            run_id = str(record.get("run_id") or "")
            if not _is_hex(authorization_sha256, 64):
                raise BootstrapRefusal(
                    "existing runtime cache authorization identity is invalid"
                )
            _validate_complete(
                stable_final,
                manifest,
                manifest_sha256,
                authorization_sha256,
                customer_identity_sha256,
                run_id,
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
        "PATH": "/usr/bin:/bin",
    }


def _execution_authorization_values(
    manifest_sha256: str,
) -> tuple[str, str, str, str]:
    authorization_sha256 = os.environ.get(
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256", ""
    ).strip()
    customer_identity_sha256 = os.environ.get(
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", ""
    ).strip()
    run_id = os.environ.get("NPA_BYOF_RUN_ID", "").strip()
    if not _is_hex(authorization_sha256, 64):
        raise BootstrapRefusal("customer authorization SHA-256 is unavailable")
    scope_sha256 = _runtime_cache_scope_sha256(
        manifest_sha256,
        customer_identity_sha256,
        run_id,
        authorization_sha256,
    )
    return authorization_sha256, customer_identity_sha256, run_id, scope_sha256


def _require_current_cache_link(
    cache_root_descriptor: int, expected: str, refusal: str
) -> None:
    """Require the descriptor-relative current link to name the expected cache."""

    try:
        current = os.readlink("current", dir_fd=cache_root_descriptor)
    except OSError:
        raise BootstrapRefusal(refusal) from None
    if current != expected:
        raise BootstrapRefusal(refusal)


def execute(
    cache_descriptor: int = INHERITED_CACHE_DESCRIPTOR,
    authorization_descriptor: int = INHERITED_AUTHORIZATION_DESCRIPTOR,
    execution_lock_descriptor: int = INHERITED_EXECUTION_LOCK_DESCRIPTOR,
    bootstrap_lock_descriptor: int = INHERITED_BOOTSTRAP_LOCK_DESCRIPTOR,
) -> int:
    """Run the fixed smoke from one locked, descriptor-stable cache snapshot."""

    manifest, manifest_sha256 = _validate_manifest(DEFAULT_MANIFEST)
    _, requirements_sha256 = _validate_requirements(DEFAULT_REQUIREMENTS, manifest)
    expected_authorization_sha256, expected_customer, expected_run, scope_sha256 = (
        _execution_authorization_values(manifest_sha256)
    )
    try:
        supervisor_uid = pwd.getpwnam(RUNTIME_SUPERVISOR_USER).pw_uid
    except KeyError as exc:
        raise BootstrapRefusal("runtime supervisor account is unavailable") from exc
    try:
        authorization_bytes = _read_private_regular_descriptor(
            authorization_descriptor,
            limit=1024 * 1024,
            owner_uid=supervisor_uid,
            input_name="customer authorization file",
        )
    except BootstrapRefusal as exc:
        raise CustomerAcceptanceRequired("authorization_file_invalid") from exc
    authorization, authorization_sha256 = _validate_customer_authorization_bytes(
        authorization_bytes,
        expected_authorization_sha256,
        manifest,
        manifest_sha256,
    )
    customer_identity_sha256 = authorization["customer_identity_sha256"]
    run_id = authorization["run_id"]
    if customer_identity_sha256 != expected_customer or run_id != expected_run:
        raise CustomerAcceptanceRequired("authorization_wrong_scope")
    cache_root = _validate_cache_root(DEFAULT_CACHE, None)
    with _open_cache_root_descriptor(
        cache_root, create=False, owner_uid=supervisor_uid
    ) as root_fd:
        identity = _cache_entry_identity_at(root_fd, scope_sha256)
        if identity is None:
            raise BootstrapRefusal("runtime cache is not materialized")
        _require_inherited_lock(root_fd, ".execution.lock", execution_lock_descriptor)
        _require_inherited_lock(root_fd, ".bootstrap.lock", bootstrap_lock_descriptor)
        _require_current_cache_link(
            root_fd,
            scope_sha256,
            "runtime current link differs from the accepted cache",
        )
        governing_terms_sha256 = _governing_terms_identity(manifest)
        try:
            opened = os.fstat(cache_descriptor)
        except OSError as exc:
            raise BootstrapRefusal(
                "inherited runtime cache descriptor is unavailable"
            ) from exc
        if (opened.st_dev, opened.st_ino) != identity or not stat.S_ISDIR(
            opened.st_mode
        ):
            raise BootstrapRefusal("runtime cache descriptor identity changed")
        stable_root = Path("/proc/self/fd") / str(cache_descriptor)
        _validate_complete(
            stable_root,
            manifest,
            manifest_sha256,
            authorization_sha256,
            customer_identity_sha256,
            run_id,
            requirements_sha256,
            governing_terms_sha256,
        )
        environment = _runtime_execution_environment(stable_root)
        completed = subprocess.run(
            ["/opt/npa/libero/smoke.sh"],
            check=False,
            env=environment,
            pass_fds=(cache_descriptor,),
            start_new_session=True,
        )
        _validate_complete(
            stable_root,
            manifest,
            manifest_sha256,
            authorization_sha256,
            customer_identity_sha256,
            run_id,
            requirements_sha256,
            governing_terms_sha256,
        )
        if _cache_entry_identity_at(root_fd, scope_sha256) != identity:
            raise BootstrapRefusal("runtime cache entry changed during execution")
        _require_current_cache_link(
            root_fd,
            scope_sha256,
            "runtime current link changed during execution",
        )
        return completed.returncode


def _parse_utc(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise BootstrapRefusal(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise BootstrapRefusal(f"{label} must include a timezone")
    return parsed


def _storage_authorization(
    output_prefix: str, run_id: str, endpoint: str
) -> dict[str, Any]:
    encoded = os.environ.get("NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64", "").strip()
    expected = os.environ.get(
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256", ""
    ).strip()
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise BootstrapRefusal(
            "output storage authorization is not valid Base64"
        ) from exc
    if not _is_hex(expected, 64) or hashlib.sha256(payload).hexdigest() != expected:
        raise BootstrapRefusal(
            "output storage authorization is not control-plane-bound"
        )
    try:
        authorization = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal(
            "output storage authorization is not valid JSON"
        ) from exc
    keys = {
        "schema",
        "issuer",
        "capability_id",
        "customer_identity_sha256",
        "run_id",
        "candidate_image",
        "runtime_manifest_sha256",
        "output_prefix",
        "endpoint_url",
        "access_key_id_sha256",
        "secret_access_key_sha256",
        "session_token_sha256",
        "policy_sha256",
        "issued_at",
        "expires_at",
        "nonce",
        "signature",
    }
    if not isinstance(authorization, dict) or set(authorization) != keys:
        raise BootstrapRefusal("output storage authorization schema is not closed")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.environ.get("AWS_SESSION_TOKEN", "")
    prefix_sha256 = hashlib.sha256(output_prefix.encode()).hexdigest()
    _, runtime_manifest_sha256 = _validate_manifest(DEFAULT_MANIFEST)
    issued_at = _parse_utc(authorization.get("issued_at"), "authorization issued_at")
    expires_at = _parse_utc(authorization.get("expires_at"), "authorization expires_at")
    customer_authorization_expires_at = _parse_utc(
        os.environ.get("NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT"),
        "customer authorization expires_at",
    )
    now = datetime.now(timezone.utc)
    valid = (
        authorization.get("schema") == OUTPUT_STORAGE_AUTHORIZATION_SCHEMA
        and authorization.get("issuer") == "npa-output-storage-control-plane"
        and re.fullmatch(
            r"[a-z0-9][a-z0-9-]{15,79}",
            str(authorization.get("capability_id") or ""),
        )
        is not None
        and authorization.get("customer_identity_sha256")
        == os.environ.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256")
        and authorization.get("run_id") == run_id
        and authorization.get("candidate_image") == os.environ.get("BYOF_IMAGE")
        and authorization.get("runtime_manifest_sha256") == runtime_manifest_sha256
        and authorization.get("output_prefix") == output_prefix
        and authorization.get("endpoint_url") == endpoint
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
        and expires_at <= customer_authorization_expires_at
        and isinstance(authorization.get("signature"), dict)
        and set(authorization["signature"])
        == {"algorithm", "public_key_sha256", "signature_b64"}
        and authorization["signature"].get("algorithm") == "ed25519"
    )
    if not valid:
        raise BootstrapRefusal("output storage authorization is invalid or expired")
    _verify_output_storage_authorization_signature(
        authorization, authorization["signature"]
    )
    return authorization


def _output_version_query(method: str, query: str) -> str:
    """Permit only one canonical immutable-version selector for object reads/deletes."""

    if not query:
        return ""
    name, separator, encoded = query.partition("=")
    if method not in {"GET", "HEAD", "DELETE"} or name != "versionId" or not separator:
        raise BootstrapRefusal("output storage version query is not allowed")
    try:
        version_id = urllib.parse.unquote(encoded, errors="strict")
    except UnicodeError as exc:
        raise BootstrapRefusal("output storage version query is not canonical") from exc
    if (
        not version_id
        or version_id == "null"
        or len(version_id.encode()) > 1024
        or any(ord(character) < 32 or ord(character) == 127 for character in version_id)
    ):
        raise BootstrapRefusal("output storage immutable version identity is invalid")
    canonical = "versionId=" + urllib.parse.quote(version_id, safe="-_.~")
    if query != canonical:
        raise BootstrapRefusal("output storage version query is not canonical")
    return canonical


def _sigv4_request(
    method: str,
    url: str,
    *,
    payload: bytes = b"",
    extra_headers: dict[str, str] | None = None,
    expected_statuses: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, str], bytes]:
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
        or parsed.fragment
    ):
        raise BootstrapRefusal(
            "output storage endpoint must be a credential-free HTTPS URL"
        )
    canonical_query = _output_version_query(method, parsed.query)
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
    canonical_uri = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/-_.~")
    if parsed.path != canonical_uri:
        raise BootstrapRefusal("output storage object path is not canonical")
    canonical_request = "\n".join(
        (
            method,
            canonical_uri,
            canonical_query,
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
    signature = hmac.new(
        signing_key, string_to_sign.encode(), hashlib.sha256
    ).hexdigest()
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
    target = urllib.parse.urlunsplit(
        ("", "", canonical_uri or "/", canonical_query, "")
    )
    try:
        connection.request(
            method,
            target,
            body=payload if method == "PUT" else None,
            headers=request_headers,
        )
        response = connection.getresponse()
        try:
            if response.status not in expected_statuses:
                raise BootstrapRefusal(
                    f"output storage {method} refused with HTTP {response.status}"
                )
            status_code = response.status
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
        raise BootstrapRefusal(
            "output storage response exceeds the aggregate size budget"
        )
    return status_code, response_headers, body


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
        urllib.parse.quote(segment, safe="-_.~") for segment in (bucket, *segments)
    )
    return f"{endpoint}/{path}"


def _immutable_output_bytes(
    directory_fd: int, name: str, limit: int
) -> tuple[bytes, str]:
    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
        raise BootstrapRefusal("output violates its type, link, or size boundary")
    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    with os.fdopen(file_fd, "rb") as stream:
        opened_before = os.fstat(stream.fileno())
        payload = stream.read(limit + 1)
        opened_after = os.fstat(stream.fileno())
    stable_identity = (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
    ) == (
        opened_before.st_dev,
        opened_before.st_ino,
        len(payload),
        opened_before.st_mtime_ns,
    )
    if len(payload) > limit or not stable_identity:
        raise BootstrapRefusal("output changed or exceeded its budget during snapshot")
    return payload, hashlib.sha256(payload).hexdigest()


def _verified_output_put(
    *,
    endpoint: str,
    bucket: str,
    object_key: str,
    name: str,
    payload: bytes,
    digest: str,
    attempted: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checksum = base64.b64encode(bytes.fromhex(digest)).decode()
    url = _s3_object_url(endpoint, bucket, object_key)
    try:
        _, created_headers, _ = _sigv4_request(
            "PUT",
            url,
            payload=payload,
            extra_headers={"if-none-match": "*", "x-amz-checksum-sha256": checksum},
        )
    except Exception:
        _recover_ambiguous_output_put(
            url=url,
            payload=payload,
            digest=digest,
            checksum=checksum,
            attempted=attempted,
            object_key=object_key,
        )
        raise
    version_id = created_headers.get("x-amz-version-id", "")
    etag = created_headers.get("etag", "")
    if not version_id or re.fullmatch(r'"[^\"]+"', etag) is None:
        # A successful PUT can omit immutable identity headers. Recover the
        # object through a bounded HEAD before refusing so rollback can delete
        # exactly the object that was created.
        if not _recover_ambiguous_output_put(
            url=url,
            payload=payload,
            digest=digest,
            checksum=checksum,
            attempted=attempted,
            object_key=object_key,
        ):
            raise BootstrapRefusal(
                "output storage did not return immutable creation identity"
            )
        raise BootstrapRefusal(
            "output storage did not return immutable creation identity"
        )
    attempted[object_key] = {
        "size_bytes": len(payload),
        "checksum": checksum,
        "etag": etag,
        "version_id": version_id,
    }
    version_url = url + "?versionId=" + urllib.parse.quote(version_id, safe="")
    _, headers, observed = _sigv4_request(
        "GET", version_url, extra_headers={"x-amz-checksum-mode": "ENABLED"}
    )
    if (
        observed != payload
        or hashlib.sha256(observed).hexdigest() != digest
        or headers.get("x-amz-checksum-sha256") != checksum
        or headers.get("x-amz-version-id") != version_id
        or headers.get("etag") != etag
    ):
        raise BootstrapRefusal("output storage checksum/readback differs")
    return {
        "name": name,
        "object_key": object_key,
        "size_bytes": len(payload),
        "sha256": digest,
        "etag": etag,
        "version_id": version_id,
    }


def _cleanup_object_etag(
    headers: dict[str, str], *, size_bytes: int, checksum: str
) -> str:
    if headers.get("x-amz-checksum-sha256") != checksum:
        return ""
    try:
        observed_size = int(headers.get("content-length", "-1"))
    except ValueError:
        return ""
    etag = headers.get("etag", "")
    if observed_size != size_bytes or re.fullmatch(r'"[^"]+"', etag) is None:
        return ""
    return etag


def _recover_ambiguous_output_put(
    *,
    url: str,
    payload: bytes,
    digest: str,
    checksum: str,
    attempted: dict[str, dict[str, Any]],
    object_key: str,
) -> bool:
    """Record an exactly identifiable object after an ambiguous PUT failure."""

    try:
        status, headers, _ = _sigv4_request(
            "HEAD",
            url,
            extra_headers={"x-amz-checksum-mode": "ENABLED"},
            expected_statuses=frozenset({200, 404}),
        )
    except (BootstrapRefusal, OSError, ValueError):
        return False
    if status != 200:
        return False
    try:
        observed_size = int(headers.get("content-length", "-1"))
    except ValueError:
        return False
    version_id = headers.get("x-amz-version-id", "")
    etag = headers.get("etag", "")
    if (
        observed_size != len(payload)
        or headers.get("x-amz-checksum-sha256") != checksum
        or not version_id
        or re.fullmatch(r'"[^\"]+"', etag) is None
        or hashlib.sha256(payload).hexdigest() != digest
    ):
        return False
    attempted[object_key] = {
        "size_bytes": len(payload),
        "checksum": checksum,
        "etag": etag,
        "version_id": version_id,
    }
    return True


def _cleanup_output_attempts(
    *, endpoint: str, bucket: str, attempted: dict[str, dict[str, Any]]
) -> None:
    failures: list[str] = []
    for object_key, identity in reversed(attempted.items()):
        url = _s3_object_url(endpoint, bucket, object_key)
        version_id = str(identity["version_id"])
        version_url = url + "?versionId=" + urllib.parse.quote(version_id, safe="")
        key_hash = hashlib.sha256(object_key.encode()).hexdigest()
        try:
            status_code, headers, _ = _sigv4_request(
                "HEAD",
                version_url,
                extra_headers={"x-amz-checksum-mode": "ENABLED"},
                expected_statuses=frozenset({200, 404}),
            )
            if status_code == 404:
                continue
            etag = _cleanup_object_etag(
                headers,
                size_bytes=int(identity["size_bytes"]),
                checksum=str(identity["checksum"]),
            )
            if (
                not etag
                or etag != identity["etag"]
                or headers.get("x-amz-version-id") != version_id
            ):
                failures.append(key_hash)
                continue
            delete_status, _, _ = _sigv4_request(
                "DELETE",
                version_url,
                extra_headers={"if-match": str(identity["etag"])},
                expected_statuses=frozenset({200, 204, 412}),
            )
            if delete_status == 412:
                failures.append(key_hash)
                continue
            status_code, _, _ = _sigv4_request(
                "HEAD", version_url, expected_statuses=frozenset({200, 404})
            )
            if status_code != 404:
                failures.append(key_hash)
        except (BootstrapRefusal, OSError, ValueError):
            failures.append(key_hash)
    if failures:
        raise BootstrapRefusal(
            "output transaction cleanup is incomplete for exact key hashes: "
            + ",".join(sorted(set(failures)))
        )


def _verify_output_lease(*, endpoint: str, bucket: str) -> tuple[str, str, str]:
    """Require the exact manager-created, version-bound output lease."""

    key = os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_KEY", "")
    etag = os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_ETAG", "")
    version_id = os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_VERSION_ID", "")
    if not key or re.fullmatch(r'"[^\"]+"', etag) is None or not version_id:
        raise BootstrapRefusal("output transaction lease binding is incomplete")
    url = _s3_object_url(endpoint, bucket, key)
    version_url = url + "?versionId=" + urllib.parse.quote(version_id, safe="")
    status, headers, _ = _sigv4_request(
        "HEAD", version_url, expected_statuses=frozenset({200, 404})
    )
    if (
        status != 200
        or headers.get("etag") != etag
        or headers.get("x-amz-version-id") != version_id
    ):
        raise BootstrapRefusal("output transaction lease changed or disappeared")
    return key, etag, version_id


def _release_output_lease(
    *, endpoint: str, bucket: str, key: str, etag: str, version_id: str
) -> None:
    url = _s3_object_url(endpoint, bucket, key)
    version_url = url + "?versionId=" + urllib.parse.quote(version_id, safe="")
    status, _, _ = _sigv4_request(
        "DELETE",
        version_url,
        extra_headers={"if-match": etag},
        expected_statuses=frozenset({200, 204, 412}),
    )
    if status == 412:
        raise BootstrapRefusal("output transaction lease identity changed")
    status, _, _ = _sigv4_request(
        "HEAD", version_url, expected_statuses=frozenset({200, 404})
    )
    if status != 404:
        raise BootstrapRefusal("output transaction lease cleanup is incomplete")


def upload_outputs(smoke_exit_code: int, *, root_fd: int) -> dict[str, Any]:
    """Upload through image-owned stdlib code after untrusted runtime execution ends."""

    run_id = os.environ.get("NPA_BYOF_RUN_ID", "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id):
        raise BootstrapRefusal("output upload requires the accepted run ID")
    output_prefix = os.environ.get("S3_OUTPUT_PREFIX", "").rstrip("/") + "/"
    parsed = urllib.parse.urlsplit(output_prefix)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise BootstrapRefusal("output prefix must be an S3 URI")
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT") or ""
    ).rstrip("/")
    if not endpoint:
        raise BootstrapRefusal("output storage endpoint is required")
    endpoint_parts = urllib.parse.urlsplit(endpoint)
    if endpoint_parts.scheme != "https" or not endpoint_parts.netloc:
        raise BootstrapRefusal("output storage endpoint must be HTTPS")
    storage_authorization = _storage_authorization(output_prefix, run_id, endpoint)
    root_info = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or stat.S_IMODE(root_info.st_mode) != 0o700
    ):
        raise BootstrapRefusal("output upload descriptor is not sealed")
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

    prefix = parsed.path.lstrip("/")
    transaction_id = uuid4().hex
    transaction_prefix = prefix + ".npa-transactions/" + transaction_id + "/"
    attempted: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    committed = False
    lease_released = False
    try:
        lease_key, lease_etag, lease_version_id = _verify_output_lease(
            endpoint=endpoint, bucket=parsed.netloc
        )
        if smoke_exit_code == 0:
            output_limits = OUTPUT_ARTIFACT_SIZE_LIMITS
        else:
            observed_names = set(os.listdir(root_fd))
            required_names = set(FAILURE_OUTPUT_REQUIRED_SIZE_LIMITS)
            optional_names = set(FAILURE_OUTPUT_OPTIONAL_SIZE_LIMITS)
            if not required_names <= observed_names or not observed_names <= (
                required_names | optional_names
            ):
                raise BootstrapRefusal(
                    "output differs from the exact failure artifact allowlist"
                )
            output_limits = {
                **FAILURE_OUTPUT_REQUIRED_SIZE_LIMITS,
                **{
                    name: FAILURE_OUTPUT_OPTIONAL_SIZE_LIMITS[name]
                    for name in sorted(optional_names & observed_names)
                },
            }
        observed_total = sum(
            os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_size
            for name in output_limits
        )
        if observed_total > MAX_OUTPUT_BYTES:
            raise BootstrapRefusal("output exceeds the aggregate size budget")
        snapshots = []
        for name, limit in sorted(output_limits.items()):
            payload, digest = _immutable_output_bytes(root_fd, name, limit)
            snapshots.append((name, payload, digest))
        for name, payload, digest in snapshots:
            current_authorization = _storage_authorization(
                output_prefix, run_id, endpoint
            )
            if (
                current_authorization["capability_id"]
                != storage_authorization["capability_id"]
            ):
                raise BootstrapRefusal(
                    "output storage authorization changed during upload"
                )
            receipts.append(
                _verified_output_put(
                    endpoint=endpoint,
                    bucket=parsed.netloc,
                    object_key=transaction_prefix + name,
                    name=name,
                    payload=payload,
                    digest=digest,
                    attempted=attempted,
                )
            )
        receipt_payload = (
            json.dumps(
                {
                    "schema": OUTPUT_RECEIPT_SCHEMA,
                    "run_id": run_id,
                    "customer_identity_sha256": os.environ.get(
                        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", ""
                    ),
                    "candidate_image": os.environ.get("BYOF_IMAGE", ""),
                    "output_storage_capability_id": storage_authorization[
                        "capability_id"
                    ],
                    "output_storage_authorization_sha256": os.environ.get(
                        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256",
                        "",
                    ),
                    "transaction_id": transaction_id,
                    "transaction_prefix": transaction_prefix,
                    "status": "verified",
                    "commit_marker": OUTPUT_RECEIPT_NAME,
                    "artifacts": receipts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        receipt_fd = os.open(
            OUTPUT_RECEIPT_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IRUSR | stat.S_IWUSR,
            dir_fd=root_fd,
        )
        with os.fdopen(receipt_fd, "wb") as stream:
            stream.write(receipt_payload)
        _immutable_output_bytes(root_fd, OUTPUT_RECEIPT_NAME, 8 * 1024 * 1024)
        current_authorization = _storage_authorization(output_prefix, run_id, endpoint)
        if (
            current_authorization["capability_id"]
            != storage_authorization["capability_id"]
        ):
            raise BootstrapRefusal("output storage authorization changed before commit")
        _verified_output_put(
            endpoint=endpoint,
            bucket=parsed.netloc,
            object_key=prefix + OUTPUT_RECEIPT_NAME,
            name=OUTPUT_RECEIPT_NAME,
            payload=receipt_payload,
            digest=receipt_sha256,
            attempted=attempted,
        )
        _verify_output_lease(endpoint=endpoint, bucket=parsed.netloc)
        _release_output_lease(
            endpoint=endpoint,
            bucket=parsed.netloc,
            key=lease_key,
            etag=lease_etag,
            version_id=lease_version_id,
        )
        lease_released = True
        committed = True
    except Exception as exc:
        cleanup_errors: list[str] = []
        if not committed:
            try:
                _cleanup_output_attempts(
                    endpoint=endpoint, bucket=parsed.netloc, attempted=attempted
                )
            except BootstrapRefusal as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        if not lease_released and "lease_key" in locals():
            try:
                _release_output_lease(
                    endpoint=endpoint,
                    bucket=parsed.netloc,
                    key=lease_key,
                    etag=lease_etag,
                    version_id=lease_version_id,
                )
            except BootstrapRefusal as lease_exc:
                cleanup_errors.append(str(lease_exc))
        if cleanup_errors:
            raise BootstrapRefusal(
                "output transaction cleanup is incomplete: "
                + "; ".join(cleanup_errors)
            ) from exc
        raise
    return {
        "schema": "npa.libero.output-upload.v2",
        "status": "verified",
        "transaction_id": transaction_id,
        "commit_sha256": receipt_sha256,
    }


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
        try:
            execution_uid = pwd.getpwnam(RUNTIME_EXECUTION_USER).pw_uid
        except KeyError as exc:
            raise BootstrapRefusal("runtime execution account is unavailable") from exc
        discovered = []
        for process in Path("/proc").iterdir():
            if not process.name.isdigit():
                continue
            try:
                status_lines = (
                    (process / "status").read_text(encoding="utf-8").splitlines()
                )
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


def _execution_process_group_ids(processes: list[int]) -> list[int]:
    """Read process groups for the exact execution-UID processes we own."""

    groups: set[int] = set()
    for pid in processes:
        try:
            stat_text = (Path("/proc") / str(pid) / "stat").read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError) as exc:
            raise BootstrapRefusal(
                "execution process group identity is unavailable"
            ) from exc
        closing = stat_text.rfind(")")
        if closing < 0:
            raise BootstrapRefusal("execution process group identity is invalid")
        fields = stat_text[closing + 2 :].split()
        if len(fields) < 3:
            raise BootstrapRefusal("execution process group identity is invalid")
        try:
            groups.add(int(fields[2]))
        except ValueError as exc:
            raise BootstrapRefusal(
                "execution process group identity is invalid"
            ) from exc
    return sorted(groups)


def _terminate_execution_processes(processes: list[int]) -> None:
    """Terminate every owned execution process and its process groups."""

    initial = sorted(set(processes))
    if not initial:
        return
    groups = _execution_process_group_ids(initial)
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise BootstrapRefusal(
                "runtime execution process-group termination failed"
            ) from exc
    for pid in initial:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise BootstrapRefusal(
                "runtime execution termination failed"
            ) from exc

    deadline = time.monotonic() + 5.0
    remaining = _execution_uid_processes()
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = _execution_uid_processes()
    if remaining:
        for group in _execution_process_group_ids(remaining):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise BootstrapRefusal(
                    "runtime execution process-group kill failed"
                ) from exc
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise BootstrapRefusal("runtime execution kill failed") from exc
        deadline = time.monotonic() + 5.0
        while remaining and time.monotonic() < deadline:
            time.sleep(0.05)
            remaining = _execution_uid_processes()
    if remaining:
        raise BootstrapRefusal(
            "runtime execution cleanup is incomplete: "
            + ",".join(str(pid) for pid in remaining)
        )


def _materialize_supervisor_artifact(root_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP,
        dir_fd=root_fd,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


@contextmanager
def _bind_inherited_descriptors(sources: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Expose exact retained descriptors at the sudo close-from boundary."""

    targets = tuple(
        range(INHERITED_CACHE_DESCRIPTOR, INHERITED_CACHE_DESCRIPTOR + len(sources))
    )
    preserved = [os.dup(source) for source in sources]
    try:
        for source, target in zip(preserved, targets, strict=True):
            os.dup2(source, target, inheritable=True)
        yield targets
    finally:
        for target in targets:
            try:
                os.close(target)
            except OSError:
                pass
        for descriptor in preserved:
            os.close(descriptor)


def execute_and_upload() -> int:
    """Hold exact authorization and exclusive UID ownership through readback."""

    manifest, manifest_sha256 = _validate_manifest(DEFAULT_MANIFEST)
    _, requirements_sha256 = _validate_requirements(DEFAULT_REQUIREMENTS, manifest)
    authorization_sha256, customer_identity_sha256, run_id, scope_sha256 = (
        _execution_authorization_values(manifest_sha256)
    )
    cache_root = _validate_cache_root(DEFAULT_CACHE, None)
    output_root = _run_output_root(run_id)
    output_fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    authorization_fd = -1
    try:
        output_info = os.fstat(output_fd)
        if (
            output_info.st_uid != os.getuid()
            or stat.S_IMODE(output_info.st_mode) != 0o1770
        ):
            raise BootstrapRefusal("execution output staging directory is invalid")
        try:
            authorization_fd = os.open(
                PROTECTED_AUTHORIZATION_NAME,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=output_fd,
            )
        except OSError as exc:
            raise CustomerAcceptanceRequired(
                "authorization_missing_at_execution"
            ) from exc
        try:
            authorization_bytes = _read_private_regular_descriptor(
                authorization_fd,
                limit=1024 * 1024,
                owner_uid=os.getuid(),
                input_name="customer authorization file",
            )
        except BootstrapRefusal as exc:
            raise CustomerAcceptanceRequired("authorization_file_invalid") from exc
        authorization, observed_authorization_sha256 = (
            _validate_customer_authorization_bytes(
                authorization_bytes,
                authorization_sha256,
                manifest,
                manifest_sha256,
            )
        )
        if (
            observed_authorization_sha256 != authorization_sha256
            or authorization["customer_identity_sha256"] != customer_identity_sha256
            or authorization["run_id"] != run_id
        ):
            raise CustomerAcceptanceRequired("authorization_wrong_scope")
        bootstrap_receipt = _bootstrap_receipt_path(cache_root, run_id)
        bootstrap_payload = _immutable_supervisor_bytes(
            bootstrap_receipt, OUTPUT_SIZE_LIMITS["npa_runtime_bootstrap.json"]
        )
        with _open_cache_root_descriptor(cache_root, create=False) as cache_root_fd:
            identity = _cache_entry_identity_at(cache_root_fd, scope_sha256)
            if identity is None:
                raise BootstrapRefusal("runtime cache is not materialized")
            if os.readlink("current", dir_fd=cache_root_fd) != scope_sha256:
                raise BootstrapRefusal(
                    "runtime current link differs from accepted cache"
                )
            with _cache_lock(
                cache_root_fd, ".execution.lock", exclusive=True, create=True
            ) as execution_lock_fd:
                if _execution_uid_processes():
                    raise BootstrapRefusal("runtime execution UID is already active")
                with _cache_lock(
                    cache_root_fd, ".bootstrap.lock", exclusive=False, create=False
                ) as bootstrap_lock_fd:
                    descriptor = os.open(
                        scope_sha256,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=cache_root_fd,
                    )
                    try:
                        opened = os.fstat(descriptor)
                        if (opened.st_dev, opened.st_ino) != identity:
                            raise BootstrapRefusal(
                                "runtime cache descriptor identity changed"
                            )
                        stable_root = Path("/proc/self/fd") / str(descriptor)
                        governing_terms_sha256 = _governing_terms_identity(manifest)
                        _validate_complete(
                            stable_root,
                            manifest,
                            manifest_sha256,
                            authorization_sha256,
                            customer_identity_sha256,
                            run_id,
                            requirements_sha256,
                            governing_terms_sha256,
                        )
                        runtime_metadata = _immutable_supervisor_bytes(
                            stable_root / ".complete.json",
                            OUTPUT_SIZE_LIMITS["npa_runtime_metadata.json"],
                        )
                        stdout_fd = os.open(
                            "solution_smoke_stdout.log",
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o640,
                            dir_fd=output_fd,
                        )
                        try:
                            stderr_fd = os.open(
                                "solution_smoke_stderr.log",
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o640,
                                dir_fd=output_fd,
                            )
                        except Exception:
                            os.close(stdout_fd)
                            raise
                        with (
                            os.fdopen(stdout_fd, "wb") as stdout,
                            os.fdopen(stderr_fd, "wb") as stderr,
                            _bind_inherited_descriptors(
                                (
                                    descriptor,
                                    authorization_fd,
                                    execution_lock_fd,
                                    bootstrap_lock_fd,
                                )
                            ) as inherited,
                        ):
                            _validate_customer_authorization_bytes(
                                authorization_bytes,
                                authorization_sha256,
                                manifest,
                                manifest_sha256,
                            )
                            environment = _runtime_execution_environment(
                                Path("/proc/self/fd") / str(INHERITED_CACHE_DESCRIPTOR)
                            )
                            environment["NPA_LIBERO_BOOTSTRAP_RECEIPT"] = str(
                                bootstrap_receipt
                            )
                            try:
                                completed = subprocess.run(
                                    [
                                        "/usr/bin/sudo",
                                        "--close-from",
                                        str(INHERITED_BOOTSTRAP_LOCK_DESCRIPTOR + 1),
                                        "--user=npa-libero-exec",
                                        "/opt/npa/libero/runtime-bootstrap.py",
                                        "execute",
                                    ],
                                    check=False,
                                    env=environment,
                                    stdout=stdout,
                                    stderr=stderr,
                                    pass_fds=inherited,
                                    start_new_session=True,
                                )
                            finally:
                                remaining_processes = _execution_uid_processes()
                                if remaining_processes:
                                    _terminate_execution_processes(remaining_processes)
                                    raise BootstrapRefusal(
                                        "runtime execution left processes behind: "
                                        + ",".join(
                                            str(pid) for pid in remaining_processes
                                        )
                                    )
                        smoke_exit_code = completed.returncode
                        if smoke_exit_code == 3:
                            # The child emits this code only for a customer-actionable
                            # authorization refusal. Revalidate the same retained bytes
                            # so the supervisor returns the structured acceptance state
                            # and never publishes it as a workload failure.
                            try:
                                _validate_customer_authorization_bytes(
                                    authorization_bytes,
                                    authorization_sha256,
                                    manifest,
                                    manifest_sha256,
                                )
                            except CustomerAcceptanceRequired:
                                raise
                            raise BootstrapRefusal(
                                "runtime child reported an inconsistent authorization refusal"
                            )
                        current_output = output_root.stat(follow_symlinks=False)
                        if not stat.S_ISDIR(current_output.st_mode) or (
                            current_output.st_dev,
                            current_output.st_ino,
                        ) != (output_info.st_dev, output_info.st_ino):
                            raise BootstrapRefusal(
                                "execution output staging directory changed during execution"
                            )
                        os.fchmod(output_fd, 0o700)
                        authorization_info = os.fstat(authorization_fd)
                        named_authorization = os.stat(
                            PROTECTED_AUTHORIZATION_NAME,
                            dir_fd=output_fd,
                            follow_symlinks=False,
                        )
                        if (authorization_info.st_dev, authorization_info.st_ino) != (
                            named_authorization.st_dev,
                            named_authorization.st_ino,
                        ):
                            raise BootstrapRefusal(
                                "protected authorization identity changed"
                            )
                        os.unlink(PROTECTED_AUTHORIZATION_NAME, dir_fd=output_fd)
                        _materialize_supervisor_artifact(
                            output_fd, "npa_runtime_bootstrap.json", bootstrap_payload
                        )
                        _materialize_supervisor_artifact(
                            output_fd, "npa_runtime_metadata.json", runtime_metadata
                        )
                        artifact_name = os.environ.get("BYOF_SMOKE_ARTIFACT_NAME", "")
                        try:
                            artifact_info = os.stat(
                                artifact_name, dir_fd=output_fd, follow_symlinks=False
                            )
                            artifact_is_regular = (
                                stat.S_ISREG(artifact_info.st_mode)
                                and artifact_info.st_nlink == 1
                            )
                        except OSError:
                            artifact_is_regular = False
                        if (
                            artifact_name not in OUTPUT_ARTIFACT_SIZE_LIMITS
                            or not artifact_is_regular
                        ):
                            error_fd = os.open(
                                "solution_smoke_stderr.log",
                                os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW,
                                dir_fd=output_fd,
                            )
                            with os.fdopen(error_fd, "a", encoding="utf-8") as stderr:
                                stderr.write(
                                    f"missing required smoke artifact: {artifact_name}\n"
                                )
                            smoke_exit_code = 1
                        _validate_customer_authorization_bytes(
                            authorization_bytes,
                            authorization_sha256,
                            manifest,
                            manifest_sha256,
                        )
                        upload_outputs(smoke_exit_code, root_fd=output_fd)
                        _validate_complete(
                            stable_root,
                            manifest,
                            manifest_sha256,
                            authorization_sha256,
                            customer_identity_sha256,
                            run_id,
                            requirements_sha256,
                            governing_terms_sha256,
                        )
                        if (
                            _cache_entry_identity_at(cache_root_fd, scope_sha256)
                            != identity
                        ):
                            raise BootstrapRefusal(
                                "runtime cache changed before readback"
                            )
                        if os.readlink("current", dir_fd=cache_root_fd) != scope_sha256:
                            raise BootstrapRefusal(
                                "runtime current link changed before readback"
                            )
                        return smoke_exit_code
                    finally:
                        os.close(descriptor)
    finally:
        if authorization_fd >= 0:
            os.close(authorization_fd)
        os.close(output_fd)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "wait-for-release",
            "ensure",
            "status",
            "execute",
            "execute-and-upload",
        ),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--authorization", default="")
    parser.add_argument("--authorization-sha256", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--smoke-exit-code", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "wait-for-release":
            print(
                json.dumps({**wait_for_release(), "status": "released"}, sort_keys=True)
            )
            return 0
        if args.command == "execute":
            return execute()
        if args.command == "execute-and-upload":
            return execute_and_upload()
        else:
            payload = ensure(args) if args.command == "ensure" else status(args)
    except CustomerAcceptanceRequired as exc:
        try:
            manifest, manifest_sha256 = _validate_manifest(Path(args.manifest))
            notification = _customer_acceptance_notification(
                manifest, manifest_sha256=manifest_sha256, reason=exc.reason
            )
        except Exception:
            notification = {
                "schema": "npa.libero.customer-acceptance-notification.v1",
                "status": "needs_customer_acceptance",
                "solution": "libero",
                "reason": exc.reason,
            }
        print(json.dumps(notification, sort_keys=True), file=sys.stderr)
        return 3
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
