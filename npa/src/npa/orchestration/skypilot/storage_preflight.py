"""Verify NebiusStore's named AWS profile before Sky creates storage or compute.

SkyPilot 0.12.2 explicitly selects boto3's ``nebius`` profile and copies only
the default AWS files to its controller. Environment access keys therefore do
not prove the mount's principal. The managed-runtime probe below performs no
storage requests and returns only comparison outcomes, never credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


def _fingerprint(access: str, secret: str, token: str = "") -> str:
    return hashlib.sha256(json.dumps([access, secret, token]).encode()).hexdigest()


def nebius_mount_destinations(documents: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Return writable Nebius mount locations with explicit directory semantics."""
    from npa.execution_preflight import ExecutionPreflightError, _destination

    destinations = {}
    for document in documents:
        mounts = document.get("file_mounts") or {}
        if not isinstance(mounts, Mapping):
            raise ExecutionPreflightError("storage_mount", "file mounts must be a resolved mapping")
        env = document.get("envs") or {}
        for mount_path, declaration in mounts.items():
            # SkyPilot Task treats scalar cloud URIs as copy-only inputs.
            mount = declaration if isinstance(declaration, Mapping) else {"source": declaration, "mode": "COPY"}
            source = mount.get("source", "")
            stores = mount.get("store") or []
            stores = [stores] if isinstance(stores, str) else stores
            if not isinstance(stores, (list, tuple)):
                raise ExecutionPreflightError("storage_mount", "storage provider selection is unreadable")
            native = isinstance(source, str) and source.startswith("nebius://")
            native = native or "NEBIUS" in [str(store).upper() for store in stores]
            if not native:
                continue
            if not isinstance(source, str) or not source.startswith("nebius://"):
                raise ExecutionPreflightError("storage_mount", "Nebius mounts require an explicit existing nebius:// source")
            uri = "s3://" + source.removeprefix("nebius://")
            _destination(uri, "directory")
            if str(mount.get("mode") or "MOUNT").upper() == "COPY" or mount.get("read_only") is True:
                continue
            # Raw --durable-s3 mounts the bucket root, then the supported
            # instrumentation writes only its explicit durable run prefix.
            if (mount_path == env.get("NPA_WORKFLOW_MOUNT_ROOT")
                    and urlparse(uri).netloc == env.get("NPA_WORKFLOW_S3_BUCKET")
                    and not urlparse(uri).path.strip("/")
                    and env.get("NPA_WORKFLOW_S3_PREFIX")):
                uri = uri.rstrip("/") + "/" + str(env["NPA_WORKFLOW_S3_PREFIX"]).strip("/")
            destinations[uri] = "directory"
    return destinations


def _has_nebius_mount(documents: Sequence[Mapping[str, Any]]) -> bool:
    for document in documents:
        for declaration in (document.get("file_mounts") or {}).values():
            mount = declaration if isinstance(declaration, Mapping) else {"source": declaration}
            stores = mount.get("store") or []
            stores = [stores] if isinstance(stores, str) else stores
            if str(mount.get("source") or "").startswith("nebius://") or "NEBIUS" in [str(s).upper() for s in stores]:
                return True
    return False


def _profile_probe(request: Mapping[str, str]) -> dict[str, str]:
    """Actual boto3 named-profile resolution; constructing a client is read-only."""
    try:
        import boto3

        home = Path.home()
        for variable, name in (("AWS_SHARED_CREDENTIALS_FILE", "credentials"), ("AWS_CONFIG_FILE", "config")):
            default = home / ".aws" / name
            if os.environ.get(variable) and Path(os.environ[variable]).expanduser().resolve() != default.resolve():
                return {"status": "fail", "reason": "file_override"}
            if not default.is_file():
                return {"status": "unknown", "reason": "missing_profile"}
            # Sky's credential-file propagation requires both exact headers.
            header = "[nebius]" if name == "credentials" else "[profile nebius]"
            if header not in default.read_text():
                return {"status": "unknown", "reason": "missing_profile"}
        session = boto3.session.Session(profile_name="nebius")
        profile = session._session.full_config.get("profiles", {}).get("nebius", {})
        if any(key in profile for key in ("role_arn", "source_profile", "credential_source",
                                          "credential_process", "web_identity_token_file", "sso_session", "sso_start_url")):
            return {"status": "unknown", "reason": "dynamic_profile"}
        if not profile.get("aws_access_key_id") or not profile.get("aws_secret_access_key"):
            return {"status": "unknown", "reason": "missing_profile"}
        credentials = session.get_credentials()
        if credentials is None or credentials.method not in {"shared-credentials-file", "config-file"}:
            return {"status": "unknown", "reason": "dynamic_profile"}
        frozen = credentials.get_frozen_credentials()
        if frozen.token or _fingerprint(frozen.access_key, frozen.secret_key) != request["fingerprint"]:
            return {"status": "fail", "reason": "principal"}
        client = session.client("s3")
        if client.meta.endpoint_url.rstrip("/") != request["endpoint"].rstrip("/"):
            return {"status": "fail", "reason": "endpoint"}
        signed = client._request_signer._credentials.get_frozen_credentials()
        if signed.token or _fingerprint(signed.access_key, signed.secret_key) != request["fingerprint"]:
            return {"status": "fail", "reason": "principal"}
        return {"status": "pass"}
    except Exception:
        return {"status": "unknown", "reason": "unavailable"}


def verify_nebius_mount_principal(documents: Sequence[Mapping[str, Any]], *, target: Any,
                                  sky_bin: str, environment: Mapping[str, str], cwd: str | None) -> None:
    from npa.execution_preflight import ExecutionPreflightError

    if not _has_nebius_mount(documents):
        return
    def verify_overrides(value: Any) -> None:
        if isinstance(value, Mapping):
            mounts = value.get("file_mounts") or {}
            if isinstance(mounts, Mapping) and any(".aws" in Path(str(path)).parts for path in mounts):
                raise ExecutionPreflightError("storage_mount", "task file mounts cannot replace the verified AWS profile")
            files = ("AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE")
            if any(value.get(key) for key in files) or (value.get("name") in files and (value.get("value") or value.get("valueFrom"))):
                raise ExecutionPreflightError("storage_mount", "worker AWS file overrides cannot preserve the verified mount profile")
            for item in value.values():
                verify_overrides(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                verify_overrides(item)

    verify_overrides(documents)
    interpreter = Path(sky_bin).absolute().parent / "python"
    request = {"fingerprint": _fingerprint(target.credentials.access_key_id, target.credentials.secret_access_key),
               "endpoint": target.credentials.endpoint_url}
    try:
        with tempfile.TemporaryDirectory(prefix="npa-mount-preflight-") as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            request_path, result_path = root / "request.json", root / "result.json"
            with open(request_path, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as handle:
                json.dump(request, handle)
            result = subprocess.run([str(interpreter), str(Path(__file__).absolute()), str(request_path), str(result_path)],
                                    env=dict(environment), cwd=cwd, capture_output=True, text=True, check=False)
            payload = json.loads(result_path.read_text()) if result.returncode == 0 else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        payload = {}
    if payload.get("status") == "pass":
        return
    reasons = {
        "file_override": "SkyPilot copies default AWS profile files; nondefault file overrides are unsupported for Nebius mounts",
        "missing_profile": "the executing SkyPilot home requires a complete static nebius AWS profile matching selected storage credentials",
        "dynamic_profile": "the executing Nebius mount profile cannot prove one static principal",
        "principal": "the executing nebius AWS profile disagrees with selected storage credentials",
        "endpoint": "the executing Nebius mount endpoint disagrees with selected storage endpoint",
    }
    raise ExecutionPreflightError("storage_mount", reasons.get(payload.get("reason"), "executing Nebius mount profile cannot be verified"),
                                  status=payload.get("status") if payload.get("status") in {"fail", "unknown"} else "unknown")


if __name__ == "__main__":
    payload = _profile_probe(json.loads(Path(sys.argv[1]).read_text()))
    with open(sys.argv[2], "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as handle:
        json.dump(payload, handle)
