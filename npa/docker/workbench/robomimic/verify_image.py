#!/usr/bin/env python3
"""Fail-closed verification for the neutral image and external CUDA runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any


SOURCE_REVISION = "d309eaecc18acf4152a830a895a6984b8ac71b05"
SOURCE_LICENSE_SHA256 = (
    "7cdbfab482b23a4d925d59ff169ab0bc5f8c97ceb0db79f9fd5bf46ef8aa1556"
)
BAKED_LOCK_SHA256 = "910b762eb9fa6bb31bfb05d339845d8c68c81f0b3e85ab95ec3eefbca29cad71"
BAKED_ARTIFACT_COUNT = 48
RUNTIME_ROOT_DEFAULT = "/opt/npa-runtime/robomimic"
RUNTIME_REFUSAL_STATUS = 78


class VerificationError(RuntimeError):
    """A boundary or immutable identity did not verify."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise VerificationError(f"required regular file is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected JSON object at {path}")
    return value


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _locked_baked_packages(lock_path: Path) -> dict[str, str]:
    raw = lock_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BAKED_LOCK_SHA256:
        raise VerificationError("baked dependency lock hash mismatch")
    lines = raw.decode("utf-8").splitlines()
    if len(lines) != BAKED_ARTIFACT_COUNT:
        raise VerificationError("baked dependency lock count mismatch")
    packages: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})$")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise VerificationError(f"malformed baked dependency lock line: {line!r}")
        name = _canonical_name(match.group(1))
        if name in packages:
            raise VerificationError(f"duplicate baked dependency: {name}")
        packages[name] = match.group(2)
    return packages


def verify_neutral_image(
    *,
    source_root: Path,
    metadata_path: Path,
    baked_lock_path: Path,
    baked_deps_path: Path,
    empty_boundary_paths: tuple[Path, ...],
) -> dict[str, Any]:
    metadata = _json_object(metadata_path)
    expected_metadata = {
        "schema": "npa.robomimic.source.v1",
        "repo": "https://github.com/ARISE-Initiative/robomimic.git",
        "ref": SOURCE_REVISION,
        "observed_head": SOURCE_REVISION,
        "license_sha256": SOURCE_LICENSE_SHA256,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise VerificationError(f"source metadata {key!r} mismatch")
    tree_hash = metadata.get("tree_archive_sha256")
    if (
        not isinstance(tree_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", tree_hash) is None
    ):
        raise VerificationError("source tree archive hash is absent or malformed")
    if not (source_root / "robomimic" / "__init__.py").is_file():
        raise VerificationError("robomimic source package is absent")
    if _sha256(source_root / "LICENSE") != SOURCE_LICENSE_SHA256:
        raise VerificationError("robomimic source license hash mismatch")

    locked = _locked_baked_packages(baked_lock_path)
    observed = {
        _canonical_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions(
            path=[str(baked_deps_path)]
        )
        if distribution.metadata.get("Name")
    }
    if observed != locked:
        missing = sorted(set(locked) - set(observed))
        extra = sorted(set(observed) - set(locked))
        mismatched = sorted(
            name
            for name in set(locked) & set(observed)
            if locked[name] != observed[name]
        )
        raise VerificationError(
            f"baked distribution inventory mismatch: missing={missing} extra={extra} "
            f"version_mismatches={mismatched}"
        )
    forbidden = {
        name
        for name in observed
        if name in {"torch", "torchvision", "triton"} or name.startswith("nvidia-")
    }
    if forbidden:
        raise VerificationError(
            f"forbidden runtime distributions are baked: {sorted(forbidden)}"
        )
    nonempty_boundaries = [
        str(path)
        for path in empty_boundary_paths
        if path.exists() and any(path.iterdir())
    ]
    if nonempty_boundaries:
        raise VerificationError(
            f"runtime/data/output boundary is populated: {nonempty_boundaries}"
        )
    return {
        "schema": "npa.robomimic.neutral-image-verification.v1",
        "baked_dependency_count": len(observed),
        "baked_lock_sha256": BAKED_LOCK_SHA256,
        "source_revision": SOURCE_REVISION,
        "runtime_payload_baked": False,
        "weights_baked": False,
        "data_baked": False,
        "outputs_baked": False,
    }


def _safe_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise VerificationError(f"invalid inventory path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise VerificationError(f"unsafe inventory path: {value!r}")
    if path.parts[0] != "payload":
        raise VerificationError(f"runtime inventory entry escapes payload/: {value!r}")
    return path


def _checked_entries(value: Any, *, symlinks: bool) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise VerificationError("runtime inventory entries must be arrays")
    result: dict[str, dict[str, Any]] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise VerificationError("runtime inventory entry must be an object")
        path = str(_safe_relative_path(entry.get("path")))
        if path in result:
            raise VerificationError(f"duplicate runtime inventory path: {path}")
        if symlinks:
            target = entry.get("target")
            if not isinstance(target, str) or not target or os.path.isabs(target):
                raise VerificationError(f"unsafe symlink target for {path}: {target!r}")
        else:
            if (
                not isinstance(entry.get("size"), int)
                or entry["size"] < 0
                or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))) is None
            ):
                raise VerificationError(f"invalid file identity for {path}")
        result[path] = entry
    return result


def _checked_artifacts(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, list):
        raise VerificationError("runtime artifact inventory must be an array")
    result: dict[str, dict[str, str]] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise VerificationError(
                "runtime artifact inventory entry must be an object"
            )
        name = _canonical_name(str(entry.get("name") or ""))
        version = entry.get("version")
        filename = entry.get("filename")
        source = entry.get("source")
        digest = entry.get("sha256")
        if (
            not name
            or not isinstance(version, str)
            or not version
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(source, str)
            or not source.startswith("https://")
            or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
        ):
            raise VerificationError(f"invalid runtime artifact identity: {entry!r}")
        if name in result:
            raise VerificationError(f"duplicate runtime artifact: {name}")
        result[name] = {
            "version": version,
            "filename": filename,
            "source": source,
            "sha256": str(digest),
        }
    return result


def verify_external_runtime(
    *, runtime_root: Path, runtime_lock_path: Path
) -> dict[str, Any]:
    lock = _json_object(runtime_lock_path)
    lock_hash = _sha256(runtime_lock_path)
    marker_path = runtime_root / ".ready.json"
    inventory_path = runtime_root / "inventory.json"
    marker = _json_object(marker_path)
    inventory = _json_object(inventory_path)
    expected_common = {
        "runtime_id": lock.get("runtime_id"),
        "lock_sha256": lock_hash,
        "source_revision": lock.get("source_revision"),
    }
    if marker.get("schema") != "npa.robomimic.runtime-ready.v1":
        raise VerificationError("runtime ready marker schema mismatch")
    if inventory.get("schema") != "npa.robomimic.runtime-inventory.v1":
        raise VerificationError("runtime inventory schema mismatch")
    for key, expected in expected_common.items():
        if marker.get(key) != expected or inventory.get(key) != expected:
            raise VerificationError(f"runtime {key} mismatch")
    if marker.get("inventory_sha256") != _sha256(inventory_path):
        raise VerificationError("runtime inventory hash mismatch")
    if inventory.get("abi") != lock.get("abi"):
        raise VerificationError("runtime ABI inventory mismatch")
    if inventory.get("packages") != lock.get("packages"):
        raise VerificationError("runtime package inventory mismatch")
    artifacts = _checked_artifacts(inventory.get("artifacts"))
    locked_packages = lock.get("packages")
    if not isinstance(locked_packages, dict):
        raise VerificationError("runtime lock package map is invalid")
    if set(artifacts) != set(locked_packages) or any(
        artifacts[name]["version"] != version
        for name, version in locked_packages.items()
    ):
        raise VerificationError("runtime artifact closure does not match package lock")

    files = _checked_entries(inventory.get("files"), symlinks=False)
    links = _checked_entries(inventory.get("symlinks"), symlinks=True)
    if set(files) & set(links):
        raise VerificationError("runtime path declared as both file and symlink")
    payload_root = runtime_root / "payload"
    if not payload_root.is_dir() or payload_root.is_symlink():
        raise VerificationError("runtime payload directory is absent")
    declared = set(files) | set(links)
    observed: set[str] = set()
    for path in payload_root.rglob("*"):
        relative = path.relative_to(runtime_root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            raise VerificationError(
                f"unsupported runtime filesystem object: {relative}"
            )
        observed.add(relative)
    if observed != declared:
        raise VerificationError(
            f"runtime payload inventory mismatch: missing={sorted(declared - observed)} "
            f"extra={sorted(observed - declared)}"
        )
    root_resolved = runtime_root.resolve()
    for relative, entry in files.items():
        path = runtime_root / relative
        if not path.is_file() or path.is_symlink():
            raise VerificationError(f"declared runtime file is not regular: {relative}")
        if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
            raise VerificationError(f"runtime file identity mismatch: {relative}")
    for relative, entry in links.items():
        path = runtime_root / relative
        if not path.is_symlink() or os.readlink(path) != entry["target"]:
            raise VerificationError(f"runtime symlink identity mismatch: {relative}")
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise VerificationError(
                f"runtime symlink escapes runtime root: {relative}"
            ) from exc
    interpreter = payload_root / "bin" / "python"
    if "payload/bin/python" not in declared or not os.access(interpreter, os.X_OK):
        raise VerificationError(
            "verified runtime interpreter is absent or not executable"
        )
    return {
        "schema": "npa.robomimic.external-runtime-verification.v1",
        "runtime_id": lock["runtime_id"],
        "runtime_lock_sha256": lock_hash,
        "runtime_inventory_sha256": marker["inventory_sha256"],
        "package_count": len(lock["packages"]),
        "artifact_count": len(artifacts),
        "payload_file_count": len(files),
        "payload_symlink_count": len(links),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    image = subparsers.add_parser("image")
    image.add_argument("--source-root", type=Path, default=Path("/opt/robomimic"))
    image.add_argument(
        "--metadata", type=Path, default=Path("/opt/byof/npa_source_metadata.json")
    )
    image.add_argument(
        "--baked-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/baked-requirements.lock"),
    )
    image.add_argument("--baked-deps", type=Path, default=Path("/opt/robomimic-deps"))
    runtime = subparsers.add_parser("runtime")
    runtime.add_argument(
        "--runtime-root", type=Path, default=Path(RUNTIME_ROOT_DEFAULT)
    )
    runtime.add_argument(
        "--runtime-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/runtime-requirements.lock"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "image":
            result = verify_neutral_image(
                source_root=args.source_root,
                metadata_path=args.metadata,
                baked_lock_path=args.baked_lock,
                baked_deps_path=args.baked_deps,
                empty_boundary_paths=(
                    Path(RUNTIME_ROOT_DEFAULT),
                    Path("/workspace/byof-inputs"),
                    Path("/workspace/byof-runs"),
                ),
            )
        else:
            result = verify_external_runtime(
                runtime_root=args.runtime_root,
                runtime_lock_path=args.runtime_lock,
            )
    except VerificationError as exc:
        print(f"NPA_ROBOMIMIC_RUNTIME_REFUSED: {exc}", file=sys.stderr)
        return RUNTIME_REFUSAL_STATUS if args.mode == "runtime" else 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
