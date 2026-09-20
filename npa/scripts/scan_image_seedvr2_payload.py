#!/usr/bin/env python3
"""Fail closed when a built SeedVR2 image layer contains runtime-only payload."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import zipfile


@dataclass(frozen=True)
class Finding:
    """One forbidden path or credential-shaped value in an immutable layer."""

    kind: str
    layer: str
    path: str


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "model_weight",
        re.compile(
            r"^(?!.*site-packages/[^/]+\.pth$).*\.(?:safetensors|ckpt|gguf|pth|pt)$",
            re.I,
        ),
    ),
    (
        "model_weight",
        re.compile(
            r"(?:^|/)(?:pytorch_model|diffusion_pytorch_model|model)-?\d*.*\.bin$", re.I
        ),
    ),
    (
        "populated_hf_cache",
        re.compile(r"(?:^|/)(?:\.cache/)?huggingface/(?:hub|datasets)/.+[^/]$", re.I),
    ),
    (
        "sensor_or_output_media",
        re.compile(r"^.+\.(?:mp4|mov|avi|mkv|mcap)$", re.I),
    ),
    (
        "cudnn_sdk_payload",
        re.compile(
            r"(?:^|/)(?:usr/include/cudnn[^/]*|"
            r"(?:site-packages/)?nvidia/cudnn/include/.+|"
            r"(?:site-packages/)?nvidia/cudnn/lib/[^/]+\.a)$",
            re.I,
        ),
    ),
    (
        "nvshmem_sdk_payload",
        re.compile(
            r"(?:^|/)(?:(?:site-packages/)?nvidia/nvshmem/include/.+|"
            r"(?:site-packages/)?nvidia/nvshmem/lib/[^/]+\.(?:a|bc))$",
            re.I,
        ),
    ),
    (
        "credential_file",
        re.compile(
            r"(?:^|/)(?:\.aws/credentials|\.git-credentials|\.docker/config\.json|"
            r"kubeconfig|ssh_host_(?:rsa|ecdsa|ed25519)_key)$",
            re.I,
        ),
    ),
)

SECRET_CONTENT = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"hf_[A-Za-z0-9]{24,}"),
)
NESTED_TAR_SUFFIXES = (".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz")
NESTED_ZIP_SUFFIXES = (".zip", ".whl", ".egg")
UNSUPPORTED_NESTED_ARCHIVE_SUFFIXES = (
    ".7z",
    ".conda",
    ".jar",
    ".rar",
    ".tar.zst",
    ".tzst",
)
MAX_NESTED_ARCHIVE_BYTES = 256 * 1024**2
MAX_NESTED_ARCHIVE_DEPTH = 2


def _is_application_content(name: str) -> bool:
    logical_name = name.rsplit("!/", 1)[-1]
    return logical_name.startswith(
        ("opt/seedvr2/", "opt/npa-src/")
    ) or logical_name in {
        "usr/local/bin/seedvr2-entrypoint",
        "usr/share/doc/npa-seedvr2/REDISTRIBUTION.md",
        "usr/share/doc/npa-seedvr2/THIRD_PARTY_NOTICES.md",
    }


def _path_findings(name: str, *, layer: str) -> list[Finding]:
    logical_name = name.rsplit("!/", 1)[-1]
    return [
        Finding(kind, layer, name)
        for kind, pattern in FORBIDDEN_PATHS
        if pattern.search(name) or pattern.search(logical_name)
    ]


def _member_name(name: str) -> tuple[str, bool]:
    unsafe = "\\" in name
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    unsafe = unsafe or any(part == ".." for part in normalized.split("/"))
    return normalized, unsafe


def _scan_nested_tar(
    payload: bytes, *, layer: str, path: str, depth: int
) -> list[Finding]:
    nested_layer = f"{layer}:{path}"
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            return _scan_layer_archive(
                archive,
                layer=nested_layer,
                prefix=f"{path}!/",
                depth=depth,
            )
    except tarfile.TarError:
        return [Finding("unreadable_nested_archive", layer, path)]


def _scan_nested_zip(
    payload: bytes, *, layer: str, path: str, depth: int
) -> list[Finding]:
    nested_layer = f"{layer}:{path}"
    findings: list[Finding] = []
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative_name, unsafe = _member_name(member.filename)
                name = f"{path}!/{relative_name}"
                if unsafe:
                    findings.append(Finding("unsafe_archive_path", nested_layer, name))
                findings.extend(_path_findings(name, layer=nested_layer))
                nested_kind = _nested_archive_kind(relative_name)
                member_payload: bytes | None = None
                if nested_kind:
                    if depth >= MAX_NESTED_ARCHIVE_DEPTH:
                        findings.append(
                            Finding("nested_archive_depth_exceeded", nested_layer, name)
                        )
                        continue
                    if member.file_size > MAX_NESTED_ARCHIVE_BYTES:
                        findings.append(
                            Finding("oversized_nested_archive", nested_layer, name)
                        )
                        continue
                    member_payload = archive.read(member)
                    findings.extend(
                        _scan_nested_archive(
                            member_payload,
                            layer=nested_layer,
                            path=name,
                            depth=depth + 1,
                            kind=nested_kind,
                        )
                    )
                if _is_application_content(name) and member.file_size <= 16 * 1024**2:
                    if member_payload is None:
                        member_payload = archive.read(member)
                    if any(
                        pattern.search(member_payload) for pattern in SECRET_CONTENT
                    ):
                        findings.append(
                            Finding("credential_content", nested_layer, name)
                        )
    except (
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ):
        return [Finding("unreadable_nested_archive", layer, path)]
    return findings


def _nested_archive_kind(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(NESTED_TAR_SUFFIXES):
        return "tar"
    if lowered.endswith(NESTED_ZIP_SUFFIXES):
        return "zip"
    if lowered.endswith(UNSUPPORTED_NESTED_ARCHIVE_SUFFIXES):
        return "unsupported"
    return ""


def _scan_nested_archive(
    payload: bytes, *, layer: str, path: str, depth: int, kind: str
) -> list[Finding]:
    if kind == "tar":
        return _scan_nested_tar(payload, layer=layer, path=path, depth=depth)
    if kind == "unsupported":
        return [Finding("unsupported_nested_archive", layer, path)]
    return _scan_nested_zip(payload, layer=layer, path=path, depth=depth)


def _scan_layer_archive(
    archive: tarfile.TarFile, *, layer: str, prefix: str = "", depth: int = 0
) -> list[Finding]:
    findings: list[Finding] = []
    for member in archive:
        relative_name, unsafe = _member_name(member.name)
        name = prefix + relative_name
        if unsafe:
            findings.append(Finding("unsafe_archive_path", layer, name))
        if member.isdir():
            continue
        findings.extend(_path_findings(name, layer=layer))
        payload: bytes | None = None
        nested_kind = _nested_archive_kind(relative_name)
        if member.isfile() and nested_kind:
            if depth >= MAX_NESTED_ARCHIVE_DEPTH:
                findings.append(Finding("nested_archive_depth_exceeded", layer, name))
                continue
            if member.size > MAX_NESTED_ARCHIVE_BYTES:
                findings.append(Finding("oversized_nested_archive", layer, name))
                continue
            stream = archive.extractfile(member)
            payload = stream.read() if stream is not None else b""
            findings.extend(
                _scan_nested_archive(
                    payload,
                    layer=layer,
                    path=name,
                    depth=depth + 1,
                    kind=nested_kind,
                )
            )
        if (
            member.isfile()
            and _is_application_content(name)
            and member.size <= 16 * 1024**2
        ):
            if payload is None:
                stream = archive.extractfile(member)
                payload = stream.read() if stream is not None else b""
            if any(pattern.search(payload) for pattern in SECRET_CONTENT):
                findings.append(Finding("credential_content", layer, name))
    return findings


def scan_layer(path: Path, *, layer: str) -> list[Finding]:
    """Scan one layer archive, including bytes hidden by later whiteouts."""

    with tarfile.open(path) as archive:
        return _scan_layer_archive(archive, layer=layer)


def scan_saved_image(image_tar: Path) -> tuple[list[Finding], int]:
    """Scan config plus every immutable layer in one docker-save archive."""

    findings, identity = inspect_saved_image(image_tar)
    return findings, len(identity["layers"])


def inspect_saved_image(image_tar: Path) -> tuple[list[Finding], dict]:
    """Scan and bind one docker-save archive to config and layer identities."""

    findings: list[Finding] = []
    with tarfile.open(image_tar) as archive:
        manifest_stream = archive.extractfile("manifest.json")
        if manifest_stream is None:
            raise RuntimeError("docker-save archive has no manifest.json")
        manifests = json.loads(manifest_stream.read())
        if len(manifests) != 1:
            raise RuntimeError("expected exactly one image manifest")
        config_name = manifests[0].get("Config", "")
        config_stream = archive.extractfile(config_name) if config_name else None
        if config_stream is None:
            raise RuntimeError("docker-save archive has no image config")
        config_payload = config_stream.read()
        config_digest = "sha256:" + hashlib.sha256(config_payload).hexdigest()
        try:
            config = json.loads(config_payload)
            diff_ids = config["rootfs"]["diff_ids"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("docker-save config has no rootfs diff IDs") from exc
        if not isinstance(diff_ids, list) or not all(
            isinstance(item, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", item)
            for item in diff_ids
        ):
            raise RuntimeError("docker-save config has invalid rootfs diff IDs")
        if any(pattern.search(config_payload) for pattern in SECRET_CONTENT):
            findings.append(Finding("credential_content", "image-config", config_name))
        layer_identities = []
        for relative in manifests[0]["Layers"]:
            layer_stream = archive.extractfile(relative)
            if layer_stream is None:
                raise RuntimeError(f"docker-save archive has no layer {relative}")
            digest = hashlib.sha256()
            size = 0
            while block := layer_stream.read(1024 * 1024):
                digest.update(block)
                size += len(block)
            layer_digest = "sha256:" + digest.hexdigest()
            layer_identities.append(
                {"path": relative, "diff_id": layer_digest, "bytes": size}
            )
            layer_stream = archive.extractfile(relative)
            if layer_stream is None:
                raise RuntimeError(f"docker-save archive has no layer {relative}")
            with tarfile.open(fileobj=layer_stream, mode="r|*") as layer_archive:
                findings.extend(_scan_layer_archive(layer_archive, layer=relative))
    if [item["diff_id"] for item in layer_identities] != diff_ids:
        raise RuntimeError("saved layer hashes do not match ordered rootfs diff IDs")
    return findings, {
        "image_config_digest": config_digest,
        "rootfs_diff_ids": diff_ids,
        "layers": layer_identities,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Local Docker image reference.")
    parser.add_argument("--tarball", type=Path, help="Existing docker-save tarball.")
    parser.add_argument(
        "--expected-image-id",
        help="Required immutable sha256 image ID for a supplied tarball.",
    )
    args = parser.parse_args()
    if bool(args.image) == bool(args.tarball):
        parser.error("provide exactly one image or --tarball")
    if args.tarball and not re.fullmatch(
        r"sha256:[0-9a-f]{64}", args.expected_image_id or ""
    ):
        parser.error("--tarball requires --expected-image-id")
    if args.tarball:
        source = str(args.tarball)
        expected_image_id = args.expected_image_id
        findings, identity = inspect_saved_image(args.tarball)
    else:
        source = str(args.image)
        expected_image_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format={{.Id}}", source],
            text=True,
        ).strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_id):
            raise RuntimeError("docker inspect did not return an immutable image ID")
        with tempfile.TemporaryDirectory(prefix="npa-seedvr2-image-") as scratch:
            tarball = Path(scratch) / "image.tar"
            subprocess.run(["docker", "save", source, "-o", str(tarball)], check=True)
            findings, identity = inspect_saved_image(tarball)
    if identity["image_config_digest"] != expected_image_id:
        raise RuntimeError("saved config digest differs from the expected image ID")
    report = {
        "format": "npa_seedvr2_payload_scan_v1",
        "source": source,
        "scan_complete": True,
        "image_id": expected_image_id,
        "image_config_digest": identity["image_config_digest"],
        "rootfs_diff_ids": identity["rootfs_diff_ids"],
        "layers": identity["layers"],
        "layers_scanned": len(identity["layers"]),
        "verdict": "clean" if not findings else "runtime-only-payload-detected",
        "findings": [asdict(item) for item in findings],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
