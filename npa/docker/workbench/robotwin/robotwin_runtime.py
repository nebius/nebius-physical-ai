#!/usr/bin/env python3
"""Zero-payload RoboTwin bootstrap with an intentionally closed runtime gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping


AUTH_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_RUNTIME_AUTH_V1"
AUTH_SCHEMA = "npa.byof.robotwin.inner-launch-capability.v1"
LOCK_PATH = Path("/opt/npa/robotwin/runtime-lock.json")
CAPABILITY_STATE_DIR = Path("/home/ubuntu/.local/state/npa/robotwin/capabilities")
SOURCE_REVISION = "96c1feab536306b50c26af200044fcdf126e8904"
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ASSET_REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
WORKFLOW_SHA256 = "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3"
RUNTIME_LOCK_SHA256 = "dda9bfebe81250247d25259d655589f8f3b95af7d8629d31b49c59a6af3150ee"
CAPABILITY_FIELDS = frozenset(
    {
        "schema_version",
        "capability_id",
        "runtime_manifest_sha256",
        "workflow_sha256",
        "source_revision",
        "curobo_revision",
        "asset_revision",
        "bootstrap_image_sha256",
        "runtime_binding_sha256",
        "expires_at",
    }
)
REFUSAL_PATH_ENV_KEYS = (
    "NPA_ROBOTWIN_SOURCE_DIR",
    "NPA_ROBOTWIN_ASSETS_DIR",
    "NPA_ROBOTWIN_CACHE_DIR",
    "NPA_ROBOTWIN_OUTPUT_DIR",
    "NPA_SMOKE_OUTPUT_DIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "HF_HOME",
    "TORCH_HOME",
    "WARP_CACHE_PATH",
    "PIP_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TMPDIR",
)


@dataclass(frozen=True)
class RecoveryContext:
    """Redacted marker rollback receipt safe for owner-only diagnostics."""

    cleanup_outcomes: tuple[tuple[str, str], ...]
    residual_names: tuple[str, ...]
    directory_fsync: str


class Refusal(RuntimeError):
    """Fixed-category refusal safe for logs."""

    def __init__(self, message: str, *, recovery_context: RecoveryContext | None = None) -> None:
        super().__init__(message)
        self.recovery_context = recovery_context


def _refuse(
    category: str, *, recovery_context: RecoveryContext | None = None
) -> None:
    raise Refusal(
        f"ROBOTWIN_RUNTIME_REFUSED:{category}",
        recovery_context=recovery_context,
    )


def _text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        _refuse(f"context-{name}-invalid")
    return value.strip()


def _decode_authorization(environ: Mapping[str, str]) -> dict[str, Any]:
    """Decode only the fixed minimal inner-capability schema."""

    encoded = str(environ.get(AUTH_ENV) or "")
    if not encoded:
        _refuse("authorization-missing")
    try:
        envelope = json.loads(encoded)
    except (TypeError, json.JSONDecodeError):
        envelope = None
    if not isinstance(envelope, dict) or set(envelope) != CAPABILITY_FIELDS:
        _refuse("authorization-envelope-invalid")
    if envelope.get("schema_version") != AUTH_SCHEMA:
        _refuse("authorization-version-mismatch")
    return envelope


def _validate_authorization(environ: Mapping[str, str]) -> dict[str, Any]:
    capability = _decode_authorization(environ)
    capability_id = _text(capability, "capability_id")
    if re.fullmatch(r"robotwin-inner-[0-9a-f]{20}", capability_id) is None:
        _refuse("capability-id-invalid")
    if _text(capability, "source_revision") != SOURCE_REVISION:
        _refuse("source-revision-mismatch")
    if _text(capability, "curobo_revision") != CUROBO_REVISION:
        _refuse("curobo-revision-mismatch")
    if _text(capability, "asset_revision") != ASSET_REVISION:
        _refuse("asset-revision-mismatch")
    expected_digests = {
        "workflow_sha256": WORKFLOW_SHA256,
        "runtime_manifest_sha256": RUNTIME_LOCK_SHA256,
    }
    for name, expected in expected_digests.items():
        if _text(capability, name) != expected:
            _refuse(f"{name.replace('_', '-')}-mismatch")
    expiry_text = _text(capability, "expires_at")
    expiry: datetime | None = None
    try:
        expiry = datetime.strptime(expiry_text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        pass
    if expiry is None:
        _refuse("capability-expiry-invalid")
    if expiry <= datetime.now(timezone.utc):
        _refuse("capability-stale")
    image = str(environ.get("BYOF_IMAGE") or "")
    if re.fullmatch(r"[^\s]+/npa-robotwin@sha256:[0-9a-f]{64}", image) is None:
        _refuse("bootstrap-image-not-immutable")
    if hashlib.sha256(image.encode("utf-8")).hexdigest() != _text(
        capability, "bootstrap_image_sha256"
    ):
        _refuse("bootstrap-image-binding-mismatch")
    binding = {
        "bootstrap_image": image,
        "bucket": str(environ.get("NPA_S3_BUCKET") or ""),
        "output_prefix": str(environ.get("S3_OUTPUT_PREFIX") or ""),
        "run_id": str(environ.get("NPA_BYOF_RUN_ID") or ""),
    }
    if any(not value for value in binding.values()):
        _refuse("runtime-binding-mismatch")
    binding_sha256 = hashlib.sha256(
        json.dumps(binding, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if binding_sha256 != _text(capability, "runtime_binding_sha256"):
        _refuse("runtime-binding-mismatch")
    return capability


def _consume_capability(
    capability: Mapping[str, Any], *, state_dir: Path | None = None
) -> None:
    """Atomically consume the opaque capability immediately before runtime work."""

    state_dir = CAPABILITY_STATE_DIR if state_dir is None else state_dir
    metadata = None
    try:
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = state_dir.stat(follow_symlinks=False)
    except OSError:
        pass
    if metadata is None:
        _refuse("capability-state-unavailable")
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        _refuse("capability-state-not-owner-only")
    marker = (
        state_dir
        / hashlib.sha256(_text(capability, "capability_id").encode("utf-8")).hexdigest()
    )
    directory_descriptor = -1
    descriptor = -1
    created = False
    committed = False
    failure = ""
    cleanup_outcomes: list[tuple[str, str]] = []
    residual_names: tuple[str, ...] = ()
    directory_fsync = "not-attempted"
    try:
        directory_descriptor = os.open(
            state_dir,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        opened = os.fstat(directory_descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            _refuse("capability-state-changed")
        descriptor = os.open(
            marker.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        created = True
        payload = b"consumed\n"
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("capability marker short write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.fsync(directory_descriptor)
        committed = True
    except FileExistsError:
        failure = "capability-replayed"
    except OSError:
        failure = "capability-state-unavailable"
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if failure and created and not committed and directory_descriptor >= 0:
            try:
                os.unlink(marker.name, dir_fd=directory_descriptor)
                cleanup_outcomes.append((marker.name, "removed"))
            except OSError as exc:
                errno = getattr(exc, "errno", "unknown")
                cleanup_outcomes.append(
                    (marker.name, f"error:{type(exc).__name__}:{errno}")
                )
            try:
                os.fsync(directory_descriptor)
            except OSError as exc:
                errno = getattr(exc, "errno", "unknown")
                directory_fsync = f"error:{type(exc).__name__}:{errno}"
            else:
                directory_fsync = "synced"
            try:
                residual_names = tuple(sorted(os.listdir(directory_descriptor)))
            except OSError:
                residual_names = ("<unavailable>",)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
    if failure:
        _refuse(
            failure,
            recovery_context=RecoveryContext(
                cleanup_outcomes=tuple(cleanup_outcomes),
                residual_names=residual_names,
                directory_fsync=directory_fsync,
            ),
        )


def _load_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw: bytes | None = None
    try:
        raw = path.read_bytes()
    except OSError:
        pass
    if raw is None:
        _refuse("runtime-lock-unreadable")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _refuse("runtime-lock-digest-mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "npa.robotwin.runtime-lock.v1"
    ):
        _refuse("runtime-lock-invalid")
    return payload


def run(*, lock_path: Path, environ: Mapping[str, str]) -> int:
    """Refuse before effects until runtime delivery and authority are complete."""

    authorization = _validate_authorization(environ)
    lock = _load_lock(lock_path, _text(authorization, "runtime_manifest_sha256"))
    bootstrap = lock.get("bootstrap")
    if not isinstance(bootstrap, dict) or bootstrap.get("status") != "complete":
        _refuse("runtime-lock-incomplete")
    delivery = lock.get("runtime_delivery")
    if not isinstance(delivery, dict) or delivery.get("status") != "complete":
        _refuse("runtime-delivery-technical-gates-incomplete")
    _consume_capability(authorization)
    _refuse("runtime-fetch-not-implemented")


def _refusal_runtime_paths(root: Path, temporary: Path) -> dict[str, Path]:
    paths = {
        "NPA_ROBOTWIN_SOURCE_DIR": root / "source",
        "NPA_ROBOTWIN_ASSETS_DIR": root / "assets",
        "NPA_ROBOTWIN_CACHE_DIR": root / "cache",
        "NPA_ROBOTWIN_OUTPUT_DIR": root / "output",
        "NPA_SMOKE_OUTPUT_DIR": root / "output" / "smoke",
        "XDG_CACHE_HOME": root / "cache" / "xdg",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "HF_HOME": root / "cache" / "huggingface",
        "TORCH_HOME": root / "cache" / "torch",
        "WARP_CACHE_PATH": root / "cache" / "warp",
        "PIP_CACHE_DIR": root / "cache" / "pip",
        "CUDA_CACHE_PATH": root / "cache" / "cuda",
        "TMPDIR": temporary,
    }
    if tuple(paths) != REFUSAL_PATH_ENV_KEYS:
        raise RuntimeError("RoboTwin refusal path environment is inconsistent")
    return paths


def _refusal_environment(
    *, home: Path, runtime_paths: Mapping[str, Path]
) -> dict[str, str]:
    environment = {
        "HOME": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment.update({name: str(path) for name, path in runtime_paths.items()})
    return environment


def _isolated_tree(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                str(path.relative_to(root)),
                "directory" if path.is_dir() else "entry",
            )
            for path in root.rglob("*")
        )
    )


def assert_refusal() -> int:
    """Exercise the genuine first gate and prove no work directories are created."""

    with tempfile.TemporaryDirectory(prefix="robotwin-refusal-") as raw_root:
        root = Path(raw_root)
        home = root / "home"
        cwd = root / "cwd"
        temporary = root / "tmp"
        home.mkdir(mode=0o700)
        cwd.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
        runtime_paths = _refusal_runtime_paths(root, temporary)
        before = _isolated_tree(root)
        environment = _refusal_environment(home=home, runtime_paths=runtime_paths)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "run"],
            env=environment,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            completed.returncode != 78
            or "ROBOTWIN_RUNTIME_REFUSED:authorization-missing" not in completed.stderr
        ):
            raise RuntimeError("RoboTwin missing-authorization refusal did not execute")
        if _isolated_tree(root) != before:
            raise RuntimeError("RoboTwin refusal changed its isolated runtime tree")
    print("robotwin bootstrap refusal verified; no simulator capability claimed")
    return 0


def status(lock_path: Path) -> int:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lock = {"status": "unreadable"}
    print(
        json.dumps(
            {
                "solution": "robotwin",
                "bootstrap": "zero-vendor-payload",
                "runtime_lock": lock.get("status", "invalid"),
                "capability_validated": False,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("health", "status", "assert-refusal", "run")
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args(argv)
    if args.command == "health":
        print("ok: robotwin zero-payload bootstrap")
        return 0
    if args.command == "status":
        return status(args.lock)
    if args.command == "assert-refusal":
        return assert_refusal()
    try:
        return run(lock_path=args.lock, environ=os.environ)
    except Refusal as exc:
        print(str(exc), file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
