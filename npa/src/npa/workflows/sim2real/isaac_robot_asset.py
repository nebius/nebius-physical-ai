"""Resolve or fetch the exact Stage-2 robot asset inside an Isaac task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from npa.clients.storage import StorageError, safe_s3_tree_relative_path


class IsaacRobotAssetError(RuntimeError):
    """Raised when Isaac cannot honor the immutable robot asset contract."""


def _spec() -> dict:
    try:
        payload = json.loads(os.environ["NPA_BYO_ROBOT_SPEC_JSON"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise IsaacRobotAssetError(
            "NPA_BYO_ROBOT_SPEC_JSON is missing or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise IsaacRobotAssetError("NPA_BYO_ROBOT_SPEC_JSON must contain an object")
    return payload


def _s3_parts(uri: str, *, prefix: bool = False) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or (not key and not prefix):
        kind = "prefix" if prefix else "object"
        raise IsaacRobotAssetError(f"expected an exact s3:// {kind}, got {uri!r}")
    return parsed.netloc, key.rstrip("/") + ("/" if prefix else "")


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(item for item in root.rglob("*") if item.is_file())
    if not paths:
        raise IsaacRobotAssetError("downloaded robot source bundle is empty")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _download(uri: str, destination: Path) -> None:
    bucket, key = _s3_parts(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _s3().download_file(bucket, key, str(destination))


def _download_tree(uri: str, destination: Path) -> None:
    bucket, prefix = _s3_parts(uri, prefix=True)
    client = _s3()
    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            try:
                relative = safe_s3_tree_relative_path(key, prefix)
            except StorageError as exc:
                raise IsaacRobotAssetError(
                    f"robot asset_root_uri returned an unsafe object key: {key!r}"
                ) from exc
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    if count == 0:
        raise IsaacRobotAssetError(f"robot asset_root_uri contains no files: {uri}")


def _upload(path: Path, uri: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise IsaacRobotAssetError(f"refusing to upload missing/empty file: {path}")
    bucket, key = _s3_parts(uri)
    _s3().upload_file(str(path), bucket, key)


def _asset_tree_uri(resolved_usd_uri: str, relative_path: str) -> str:
    """Return the immutable object URI for one converted-asset tree member."""

    return f"{resolved_usd_uri}.assets/{relative_path}"


def _safe_relative_path(value: object) -> str:
    relative = str(value or "")
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or relative != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise IsaacRobotAssetError(
            f"resolved robot manifest contains an unsafe asset path: {relative!r}"
        )
    return relative


def _asset_tree_manifest(root: Path) -> tuple[list[dict], str]:
    entries: list[dict] = []
    digest = hashlib.sha256()
    symlinks = sorted(item for item in root.rglob("*") if item.is_symlink())
    if symlinks:
        relative = symlinks[0].relative_to(root).as_posix()
        raise IsaacRobotAssetError(
            f"refusing to publish symlink in resolved robot asset tree: {relative}"
        )
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item != root / "manifest.json"
    ):
        relative = _safe_relative_path(path.relative_to(root).as_posix())
        sha256 = _sha256(path)
        size = path.stat().st_size
        if size == 0:
            raise IsaacRobotAssetError(
                f"refusing to publish empty resolved robot asset: {relative}"
            )
        entries.append({"path": relative, "sha256": sha256, "size_bytes": size})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256.encode())
        digest.update(b"\0")
    if not entries:
        raise IsaacRobotAssetError("resolved robot asset tree is empty")
    return entries, digest.hexdigest()


def _validated_asset_entries(manifest: dict) -> list[dict]:
    if manifest.get("schema") != "npa.sim2real.resolved_robot_asset.v2":
        raise IsaacRobotAssetError(
            "resolved robot manifest does not describe a complete asset tree"
        )
    raw_entries = manifest.get("asset_files")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise IsaacRobotAssetError("resolved robot manifest asset_files is empty")
    entries: list[dict] = []
    seen: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise IsaacRobotAssetError("resolved robot manifest has an invalid asset entry")
        relative = _safe_relative_path(raw.get("path"))
        sha256 = str(raw.get("sha256") or "")
        size = raw.get("size_bytes")
        if relative in seen:
            raise IsaacRobotAssetError(
                f"resolved robot manifest repeats asset path: {relative}"
            )
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise IsaacRobotAssetError(
                f"resolved robot manifest has an invalid digest for {relative}"
            )
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise IsaacRobotAssetError(
                f"resolved robot manifest has an invalid size for {relative}"
            )
        seen.add(relative)
        entries.append({"path": relative, "sha256": sha256, "size_bytes": size})
    if manifest.get("entrypoint") != "robot.usd" or "robot.usd" not in seen:
        raise IsaacRobotAssetError(
            "resolved robot manifest is missing the robot.usd entrypoint"
        )
    return entries


def _validate_spec(spec: dict) -> None:
    required = (
        "asset_root_uri",
        "source_relative_path",
        "source_tree_sha256",
        "source_sha256",
        "source_format",
        "embodiment_digest",
        "resolved_usd_uri",
        "resolved_manifest_uri",
    )
    missing = [name for name in required if not spec.get(name)]
    if missing:
        raise IsaacRobotAssetError(
            "resolved RobotSpec is missing immutable asset fields: "
            + ", ".join(missing)
        )


def _validate_usd(path: Path, spec: dict) -> None:
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise IsaacRobotAssetError(f"Isaac produced an unreadable USD: {path}")
    names = {prim.GetName() for prim in stage.Traverse()}
    required_links = {
        str(spec.get("base_link") or ""),
        str(spec.get("ee_link") or ""),
        *[str(item) for item in spec.get("finger_links") or []],
    }
    missing_links = sorted(
        name for name in required_links if name and name not in names
    )
    if missing_links:
        raise IsaacRobotAssetError(
            "converted USD is missing RobotSpec links: " + ", ".join(missing_links)
        )


def prepare_with_running_app() -> dict:
    """Convert and publish while the caller's Isaac application is running."""

    spec = _spec()
    _validate_spec(spec)
    work = Path(os.environ.get("NPA_ROBOT_WORK_DIR", "/tmp/npa_robot"))
    source_root = work / "source"
    resolved = work / "resolved" / "robot.usd"
    manifest_path = work / "resolved" / "manifest.json"
    shutil.rmtree(source_root, ignore_errors=True)
    shutil.rmtree(resolved.parent, ignore_errors=True)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _download_tree(str(spec["asset_root_uri"]), source_root)
    actual_tree = _tree_sha256(source_root)
    if actual_tree != spec["source_tree_sha256"]:
        raise IsaacRobotAssetError(
            "robot source bundle digest mismatch: "
            f"expected={spec['source_tree_sha256']} actual={actual_tree}"
        )
    source = source_root / str(spec["source_relative_path"])
    if not source.is_file() or _sha256(source) != spec["source_sha256"]:
        raise IsaacRobotAssetError(
            "robot source file is missing or has the wrong digest"
        )

    try:
        if spec["source_format"] == "urdf":
            os.environ["ROS_PACKAGE_PATH"] = str(source_root)
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

            converter = UrdfConverter(
                UrdfConverterCfg(
                    asset_path=str(source),
                    usd_dir=str(resolved.parent),
                    usd_file_name=resolved.name,
                    force_usd_conversion=True,
                    fix_base=True,
                    merge_fixed_joints=False,
                    joint_drive=UrdfConverterCfg.JointDriveCfg(
                        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                            stiffness=None
                        )
                    ),
                )
            )
            produced = Path(str(getattr(converter, "usd_path", "")))
            if not produced.is_file():
                raise IsaacRobotAssetError(
                    f"Isaac UrdfConverter produced no USD for {source.name}"
                )
            if produced.resolve() != resolved.resolve():
                shutil.copy2(produced, resolved)
        elif spec["source_format"] == "usd":
            shutil.copy2(source, resolved)
        else:
            raise IsaacRobotAssetError(
                f"unsupported robot source format {spec['source_format']!r}"
            )
        _validate_usd(resolved, spec)

        # The caller keeps Kit alive through publication and the Stage 7 rollout.
        # Train/eval only start after both exact objects are durable.
        usd_sha256 = _sha256(resolved)
        asset_files, asset_tree_sha256 = _asset_tree_manifest(resolved.parent)
        manifest = {
            "schema": "npa.sim2real.resolved_robot_asset.v2",
            "embodiment_digest": spec["embodiment_digest"],
            "source_tree_sha256": spec["source_tree_sha256"],
            "source_sha256": spec["source_sha256"],
            "source_format": spec["source_format"],
            "usd_uri": spec["resolved_usd_uri"],
            "usd_sha256": usd_sha256,
            "usd_size_bytes": resolved.stat().st_size,
            "entrypoint": "robot.usd",
            "asset_files": asset_files,
            "asset_tree_sha256": asset_tree_sha256,
            "expected_action_dim": int(spec["expected_action_dim"]),
            "expected_observation_dim": int(spec["expected_observation_dim"]),
            "converter": (
                "isaaclab.sim.converters.UrdfConverter"
                if spec["source_format"] == "urdf"
                else "identity"
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        for entry in asset_files:
            relative = str(entry["path"])
            _upload(
                resolved.parent / relative,
                _asset_tree_uri(str(spec["resolved_usd_uri"]), relative),
            )
        # Preserve the exact entrypoint object for existing artifact consumers.
        _upload(resolved, str(spec["resolved_usd_uri"]))
        # Publish the manifest last: its presence attests that the full tree is durable.
        _upload(manifest_path, str(spec["resolved_manifest_uri"]))
        print("ROBOT_ASSET_RESOLVED " + json.dumps(manifest, sort_keys=True), flush=True)
        return manifest
    except IsaacRobotAssetError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface Isaac converter diagnostics
        raise IsaacRobotAssetError(
            f"Isaac robot conversion failed for {source.name}: {exc}"
        ) from exc


def prepare() -> dict:
    """Standalone compatibility entrypoint for conversion and publication."""

    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=True,
        kit_args=os.environ.get(
            "NPA_ISAAC_KIT_ARGS", "--portable-root /tmp/npa-isaac-kit"
        ),
    ).app
    try:
        return prepare_with_running_app()
    finally:
        app.close()


def fetch() -> dict:
    """Fetch Stage 7's exact resolved USD and fail closed on parity mismatch."""

    spec = _spec()
    _validate_spec(spec)
    work = Path(os.environ.get("NPA_ROBOT_WORK_DIR", "/tmp/npa_robot")) / "resolved"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest.json"
    _download(str(spec["resolved_manifest_uri"]), manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise IsaacRobotAssetError("resolved robot manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise IsaacRobotAssetError("resolved robot manifest must contain an object")
    mismatches = {
        name: {"expected": spec[name], "actual": manifest.get(name)}
        for name in (
            "embodiment_digest",
            "source_tree_sha256",
            "expected_action_dim",
            "expected_observation_dim",
        )
        if manifest.get(name) != spec[name]
    }
    if mismatches:
        raise IsaacRobotAssetError(f"resolved robot parity mismatch: {mismatches}")
    entries = _validated_asset_entries(manifest)
    for entry in entries:
        relative = str(entry["path"])
        destination = work / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _download(
            _asset_tree_uri(str(spec["resolved_usd_uri"]), relative), destination
        )
        actual_sha256 = _sha256(destination)
        actual_size = destination.stat().st_size
        if (
            actual_sha256 != entry["sha256"]
            or actual_size != entry["size_bytes"]
        ):
            raise IsaacRobotAssetError(
                "resolved robot asset mismatch for "
                f"{relative}: expected_sha256={entry['sha256']} "
                f"actual_sha256={actual_sha256} expected_size={entry['size_bytes']} "
                f"actual_size={actual_size}"
            )
    actual_entries, actual_tree_sha256 = _asset_tree_manifest(work)
    if actual_entries != entries or actual_tree_sha256 != manifest.get(
        "asset_tree_sha256"
    ):
        raise IsaacRobotAssetError(
            "resolved robot asset tree digest or inventory mismatch"
        )
    usd_entry = next(entry for entry in entries if entry["path"] == "robot.usd")
    if (
        usd_entry["sha256"] != manifest.get("usd_sha256")
        or usd_entry["size_bytes"] != manifest.get("usd_size_bytes")
    ):
        raise IsaacRobotAssetError(
            "resolved robot entrypoint does not match the manifest USD evidence"
        )
    print("ROBOT_ASSET_FETCHED " + json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "fetch"))
    args = parser.parse_args(argv)
    prepare() if args.operation == "prepare" else fetch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
