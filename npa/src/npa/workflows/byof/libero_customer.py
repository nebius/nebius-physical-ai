"""The bounded customer-operated LIBERO profile and direct customer handoff.

This mode uses the same customer signature as hosted execution. The customer
provides its verification key directly; no caller or storage signing service is
required because the workload receives neither a service-account token nor
storage credentials. It does not turn a mode selector into terms acceptance.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

MODE = "customer-run-v1"
MODE_ENV = "NPA_LIBERO_RUNTIME_DELIVERY"
KEY_FILE_ENV = "NPA_LIBERO_CUSTOMER_RUN_PUBLIC_KEY_FILE"
IMAGE_FILE_ENV = "NPA_LIBERO_CUSTOMER_IMAGE_MANIFEST_FILE"
PROFILE = Path(__file__).parent / "profiles/byof-solution-smoke-libero-customer-b200-gpu.yaml"
SECRET_NAMES = (
    "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64",
    "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256",
    "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
)
FORBIDDEN_WORKER_ENV = frozenset({
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "KUBECONFIG",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NGC_API_KEY",
    "NPA_LIBERO_AUTHENTICATED_CALLER_B64",
    "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
})


def selected(documents: Sequence[Mapping[str, Any]]) -> bool:
    modes = {
        str((document.get("envs") or {}).get(MODE_ENV) or "")
        for document in documents if document.get("resources")
    }
    if not modes or modes == {""}:
        return False
    if modes != {MODE}:
        raise ValueError("LIBERO customer profile has mixed or unknown delivery modes")
    return True


def validate_profile(documents: Sequence[Mapping[str, Any]]) -> None:
    """Allow this single fixed profile, including its credentialless pod contract."""
    tasks = [item for item in documents if item.get("resources")]
    template = list(yaml.safe_load_all(PROFILE.read_text()))[1]
    if len(tasks) != 1 or not selected(tasks):
        raise ValueError("LIBERO customer execution requires its single packaged profile")
    task = tasks[0]
    if set(task) - (set(template) | {"config"}):
        raise ValueError("LIBERO customer task adds an unsupported field")
    if set(task.get("config") or {}) - {"kubernetes"}:
        raise ValueError("LIBERO customer task adds an unsupported configuration")
    for field in ("name", "setup", "run"):
        if task.get(field) != template.get(field):
            raise ValueError(f"LIBERO customer profile {field} differs")
    resources = copy.deepcopy(task.get("resources") or {})
    expected = copy.deepcopy(template["resources"])
    resources.pop("region", None)
    resources.pop("image_id", None)
    expected.pop("image_id", None)
    # NPA lifts task Kubernetes configuration before the shared preflight.
    kubernetes = resources.pop("kubernetes", None)
    configured = (task.get("config") or {}).get("kubernetes")
    if kubernetes is not None and configured is not None and kubernetes != configured:
        raise ValueError("LIBERO customer pod declarations differ")
    if (configured or kubernetes) != expected.pop("kubernetes") or resources != expected:
        raise ValueError("LIBERO customer resource or immutable mount contract differs")
    envs = task.get("envs") or {}
    if set(envs) != set(template["envs"]) or any(name in envs for name in FORBIDDEN_WORKER_ENV):
        raise ValueError("LIBERO customer environment is not the closed credentialless profile")
    dynamic = {"NPA_BYOF_RUN_ID", "BYOF_IMAGE", "NPA_LIBERO_EXECUTABLE_PROFILE_B64"}
    for name, value in envs.items():
        if name not in dynamic and not name.startswith("NPA_LIBERO_EXPECTED_"):
            if value != template["envs"][name]:
                raise ValueError("LIBERO customer executable environment differs")


def validate_authorization(
    process_env: Mapping[str, str], *, image_manifest: dict[str, Any],
    run_id: str, profile_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify the same v2 evidence using the actual customer's private handoff."""
    from npa.deploy.images import (
        _libero_trust_root_bytes,
        validate_libero_customer_runtime_authorization,
    )
    key_path = process_env.get(KEY_FILE_ENV, "")
    key = _libero_trust_root_bytes(key_path, label="customer-run signer")
    payload = base64.b64decode(
        process_env.get("NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64", ""), validate=True
    )
    identity = process_env.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", "")
    if not identity:
        raise ValueError("actual customer identity is absent from the private handoff")
    return validate_libero_customer_runtime_authorization(
        payload, image_manifest=image_manifest, run_id=run_id,
        customer_identity_sha256=identity,
        customer_signer_public_key_sha256=hashlib.sha256(key).hexdigest(),
        executable_profile_sha256=profile_sha256,
        public_key_file=key_path, customer_run=True,
    )


def private_bytes(path: Path, *, limit: int = 2 * 1024 * 1024) -> bytes:
    """Read a stable owner-private handoff without following a symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
        or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) & 0o077
        or len(data) > limit or len(data) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("customer-run private handoff is mutable or invalid")
    return data


def image_manifest(process_env: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read reviewed technical image evidence, independently of customer consent."""
    from npa.deploy.images import (
        libero_image_manifest, libero_publication_lineage_values,
        validate_libero_qualified_image_manifest,
    )
    supplied = process_env.get(IMAGE_FILE_ENV, "")
    manifest = json.loads(private_bytes(Path(supplied))) if supplied else libero_image_manifest()
    canonical = libero_image_manifest()
    for field in ("customer_acceptance", "runtime_manifest_sha256", "runtime_requirements_sha256", "runtime_artifact_review"):
        if manifest.get(field) != canonical.get(field):
            raise ValueError("customer-run image evidence differs from reviewed runtime")
    qualification = validate_libero_qualified_image_manifest(manifest, customer_run=True)
    repo = Path(__file__).resolve().parents[5]
    libero_publication_lineage_values(
        qualification, repo, development_sha=qualification["development_sha"]
    )
    return manifest, qualification
