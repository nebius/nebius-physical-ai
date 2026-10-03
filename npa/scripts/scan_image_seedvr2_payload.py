#!/usr/bin/env python3
"""Fail closed when a built SeedVR2 image layer contains runtime-only payload."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
import zlib


@dataclass(frozen=True)
class Finding:
    """One forbidden path or credential-shaped value in an immutable layer."""

    kind: str
    layer: str
    path: str


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "model_weight",
        re.compile(r"^.*\.(?:safetensors|ckpt|gguf|pt)$", re.I),
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
GZIP_MAGIC = b"\x1f\x8b"
CONTENT_ADDRESSED_BLOB = re.compile(r"^blobs/sha256/([0-9a-f]{64})$")
MANIFEST_MEDIA_TYPES = {
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
}
CONFIG_MEDIA_TYPES = {
    "application/vnd.docker.container.image.v1+json",
    "application/vnd.oci.image.config.v1+json",
}
LAYER_MEDIA_TYPES = {
    "gzip": {
        "application/vnd.docker.image.rootfs.diff.tar.gzip",
        "application/vnd.oci.image.layer.v1.tar+gzip",
    },
    "none": {
        "application/vnd.docker.image.rootfs.diff.tar",
        "application/vnd.oci.image.layer.v1.tar",
    },
}
SETUPTOOLS_DISTUTILS_PATH_FILE = (
    b"import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = "
    b"os.environ.get(var, 'local') == 'local'; enabled and "
    b"__import__('_distutils_hack').add_shim(); \n"
)
ALLOWED_PYTHON_PATH_FILES = {
    (
        "usr/share/python-wheels/setuptools-68.1.2-py3-none-any.whl!/"
        "distutils-precedence.pth"
    ): SETUPTOOLS_DISTUTILS_PATH_FILE,
    (
        "opt/seedvr2-venv/lib/python3.12/site-packages/distutils-precedence.pth"
    ): SETUPTOOLS_DISTUTILS_PATH_FILE,
    "opt/npa-venv/lib/python3.12/site-packages/rerun_sdk.pth": b"rerun_sdk\n",
}


def _strict_json(payload: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(
        payload,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


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
    findings = [
        Finding(kind, layer, name)
        for kind, pattern in FORBIDDEN_PATHS
        if pattern.search(name) or pattern.search(logical_name)
    ]
    if name.lower().endswith(".pth") and name not in ALLOWED_PYTHON_PATH_FILES:
        findings.append(Finding("model_weight", layer, name))
    return findings


def _allowed_path_file_findings(
    name: str, payload: bytes, *, layer: str
) -> list[Finding]:
    expected = ALLOWED_PYTHON_PATH_FILES.get(name)
    if expected is not None and payload != expected:
        return [Finding("model_weight", layer, name)]
    return []


def _member_name(name: str) -> tuple[str, bool]:
    unsafe = "\\" in name or name.startswith("/")
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
                if (
                    name.lower().endswith(".pth")
                    and name in ALLOWED_PYTHON_PATH_FILES
                    and stat.S_ISLNK(member.external_attr >> 16)
                ):
                    findings.append(Finding("model_weight", nested_layer, name))
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
                if name.lower().endswith(".pth"):
                    if member_payload is None:
                        member_payload = archive.read(member)
                    findings.extend(
                        _allowed_path_file_findings(
                            name, member_payload, layer=nested_layer
                        )
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
        findings.extend(_path_findings(name, layer=layer))
        if (
            name.lower().endswith(".pth")
            and name in ALLOWED_PYTHON_PATH_FILES
            and not member.isfile()
        ):
            findings.append(Finding("model_weight", layer, name))
        if member.isdir():
            continue
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
        if member.isfile() and name.lower().endswith(".pth"):
            if payload is None:
                stream = archive.extractfile(member)
                payload = stream.read() if stream is not None else b""
            findings.extend(_allowed_path_file_findings(name, payload, layer=layer))
    return findings


def scan_layer(path: Path, *, layer: str) -> list[Finding]:
    """Scan one layer archive, including bytes hidden by later whiteouts."""

    with tarfile.open(path) as archive:
        return _scan_layer_archive(archive, layer=layer)


def scan_saved_image(image_tar: Path) -> tuple[list[Finding], int]:
    """Scan config plus every immutable layer in one docker-save archive."""

    findings, identity = inspect_saved_image(image_tar)
    return findings, len(identity["layers"])


def _blob_path(digest: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise RuntimeError("OCI descriptor has an invalid digest")
    return "blobs/" + digest.replace(":", "/", 1)


def _regular_member_stream(
    archive: tarfile.TarFile, name: str, *, description: str
) -> tarfile.ExFileObject:
    try:
        member = archive.getmember(name)
    except KeyError as exc:
        raise RuntimeError(f"docker-save archive has no {description}") from exc
    if not member.isfile():
        raise RuntimeError(f"docker-save {description} is not a regular file")
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError(f"docker-save archive has an unreadable {description}")
    return stream


def _validate_outer_archive(archive: tarfile.TarFile) -> None:
    seen: set[str] = set()
    for member in archive.getmembers():
        if member.name in seen:
            raise RuntimeError(
                f"docker-save archive has a duplicate member: {member.name}"
            )
        seen.add(member.name)
        normalized, unsafe = _member_name(member.name)
        if unsafe or normalized != member.name:
            raise RuntimeError(
                f"docker-save archive has an unsafe member path: {member.name}"
            )


def _oci_identity(
    archive: tarfile.TarFile,
    *,
    compatibility: dict,
    config_digest: str,
    config_size: int,
) -> tuple[dict | None, list[dict] | None]:
    """Validate an optional OCI identity graph emitted by modern docker save."""

    try:
        archive.getmember("index.json")
    except KeyError:
        return None, None
    index_stream = _regular_member_stream(
        archive, "index.json", description="OCI index"
    )
    try:
        index = _strict_json(index_stream.read())
        if not isinstance(index, dict):
            raise TypeError
        descriptors = index["manifests"]
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("docker-save OCI index is invalid") from exc
    if (
        index.get("schemaVersion") != 2
        or index.get("mediaType") != "application/vnd.oci.image.index.v1+json"
        or not isinstance(descriptors, list)
        or len(descriptors) != 1
        or not isinstance(descriptors[0], dict)
    ):
        raise RuntimeError("docker-save OCI index must select exactly one image")
    layout_stream = _regular_member_stream(
        archive, "oci-layout", description="OCI layout"
    )
    try:
        layout = _strict_json(layout_stream.read())
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("docker-save OCI layout is invalid") from exc
    if layout != {"imageLayoutVersion": "1.0.0"}:
        raise RuntimeError("docker-save OCI layout version is unsupported")

    descriptor = descriptors[0]
    digest = descriptor.get("digest")
    size = descriptor.get("size")
    media_type = descriptor.get("mediaType")
    if (
        not isinstance(digest, str)
        or type(size) is not int
        or size < 0
        or media_type not in MANIFEST_MEDIA_TYPES
    ):
        raise RuntimeError("docker-save OCI manifest descriptor is invalid")
    manifest_name = _blob_path(digest)
    manifest_stream = _regular_member_stream(
        archive, manifest_name, description="OCI image manifest"
    )
    manifest_payload = manifest_stream.read()
    if (
        len(manifest_payload) != size
        or "sha256:" + hashlib.sha256(manifest_payload).hexdigest() != digest
    ):
        raise RuntimeError("docker-save OCI image manifest differs from its descriptor")
    try:
        manifest = _strict_json(manifest_payload)
        if not isinstance(manifest, dict):
            raise TypeError
        config_descriptor = manifest["config"]
        layer_descriptors = manifest["layers"]
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("docker-save OCI image manifest is invalid") from exc
    if (
        manifest.get("schemaVersion") != 2
        or manifest.get("mediaType") not in MANIFEST_MEDIA_TYPES
        or not isinstance(config_descriptor, dict)
        or not isinstance(layer_descriptors, list)
        or not all(isinstance(item, dict) for item in layer_descriptors)
    ):
        raise RuntimeError("docker-save OCI image manifest schema is invalid")
    if (
        config_descriptor.get("mediaType") not in CONFIG_MEDIA_TYPES
        or config_descriptor.get("digest") != config_digest
        or config_descriptor.get("size") != config_size
        or compatibility.get("Config") != _blob_path(config_digest)
    ):
        raise RuntimeError("docker-save OCI config identity differs")
    annotations = descriptor.get("annotations", {})
    if (
        not isinstance(annotations, dict)
        or annotations.get("config.digest", config_digest) != config_digest
    ):
        raise RuntimeError("docker-save OCI config annotation differs")
    try:
        descriptor_paths = [_blob_path(item["digest"]) for item in layer_descriptors]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("docker-save OCI layer descriptor is invalid") from exc
    if compatibility.get("Layers") != descriptor_paths:
        raise RuntimeError("docker-save OCI and compatibility layer orders differ")
    return (
        {
            "manifest_digest": digest,
            "manifest_bytes": size,
            "manifest_media_type": media_type,
        },
        layer_descriptors,
    )


def _inspect_saved_image(image_tar: Path) -> tuple[list[Finding], dict]:
    """Scan and bind one docker-save archive to config and layer identities."""

    findings: list[Finding] = []
    try:
        archive_context = tarfile.open(image_tar, mode="r:")
    except tarfile.TarError as exc:
        raise RuntimeError(
            "saved image is not a readable tar archive "
            "(outer compression is unsupported)"
        ) from exc
    with archive_context as archive:
        _validate_outer_archive(archive)
        manifest_stream = _regular_member_stream(
            archive, "manifest.json", description="compatibility manifest"
        )
        try:
            manifests = _strict_json(manifest_stream.read())
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(
                "saved image compatibility manifest is invalid JSON"
            ) from exc
        if (
            not isinstance(manifests, list)
            or len(manifests) != 1
            or not isinstance(manifests[0], dict)
        ):
            raise RuntimeError("expected exactly one image manifest")
        compatibility = manifests[0]
        config_name = compatibility.get("Config", "")
        if not isinstance(config_name, str) or not config_name:
            raise RuntimeError("docker-save compatibility manifest has no image config")
        config_stream = _regular_member_stream(
            archive, config_name, description="image config"
        )
        config_payload = config_stream.read()
        config_digest = "sha256:" + hashlib.sha256(config_payload).hexdigest()
        config_address = CONTENT_ADDRESSED_BLOB.fullmatch(config_name)
        if config_address and config_digest != f"sha256:{config_address.group(1)}":
            raise RuntimeError(
                "saved image config differs from its content-addressed path"
            )
        try:
            config = _strict_json(config_payload)
            if not isinstance(config, dict):
                raise TypeError
            diff_ids = config["rootfs"]["diff_ids"]
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError("docker-save config has no rootfs diff IDs") from exc
        if not isinstance(diff_ids, list) or not all(
            isinstance(item, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", item)
            for item in diff_ids
        ):
            raise RuntimeError("docker-save config has invalid rootfs diff IDs")
        layers = compatibility.get("Layers")
        if not isinstance(layers, list) or not all(
            isinstance(item, str) and item for item in layers
        ):
            raise RuntimeError("docker-save compatibility manifest has invalid layers")
        if len(layers) != len(diff_ids):
            raise RuntimeError(
                "saved layer count does not match ordered rootfs diff IDs"
            )
        oci_identity, layer_descriptors = _oci_identity(
            archive,
            compatibility=compatibility,
            config_digest=config_digest,
            config_size=len(config_payload),
        )
        if any(pattern.search(config_payload) for pattern in SECRET_CONTENT):
            findings.append(Finding("credential_content", "image-config", config_name))
        layer_identities = []
        for index, (relative, expected_diff_id) in enumerate(
            zip(layers, diff_ids, strict=True)
        ):
            layer_stream = _regular_member_stream(
                archive, relative, description=f"layer {relative}"
            )
            blob_hash = hashlib.sha256()
            blob_size = 0
            while block := layer_stream.read(1024 * 1024):
                blob_hash.update(block)
                blob_size += len(block)
            blob_digest = "sha256:" + blob_hash.hexdigest()
            content_address = CONTENT_ADDRESSED_BLOB.fullmatch(relative)
            if content_address and blob_digest != f"sha256:{content_address.group(1)}":
                raise RuntimeError(
                    f"saved layer blob differs from content-addressed path: {relative}"
                )

            layer_stream = _regular_member_stream(
                archive, relative, description=f"layer {relative}"
            )
            magic = layer_stream.read(len(GZIP_MAGIC))
            layer_stream = _regular_member_stream(
                archive, relative, description=f"layer {relative}"
            )

            compression = "gzip" if magic == GZIP_MAGIC else "none"
            if layer_descriptors is not None:
                descriptor = layer_descriptors[index]
                if (
                    descriptor.get("mediaType") not in LAYER_MEDIA_TYPES[compression]
                    or descriptor.get("digest") != blob_digest
                    or descriptor.get("size") != blob_size
                ):
                    raise RuntimeError(
                        f"saved OCI layer descriptor differs: {relative}"
                    )
            if compression == "gzip":
                with tempfile.TemporaryFile() as uncompressed:
                    diff_hash = hashlib.sha256()
                    uncompressed_size = 0
                    try:
                        with gzip.GzipFile(fileobj=layer_stream) as decoded:
                            while block := decoded.read(1024 * 1024):
                                diff_hash.update(block)
                                uncompressed.write(block)
                                uncompressed_size += len(block)
                    except (EOFError, OSError, zlib.error) as exc:
                        raise RuntimeError(
                            f"saved layer gzip is invalid: {relative}"
                        ) from exc
                    diff_id = "sha256:" + diff_hash.hexdigest()
                    if diff_id != expected_diff_id:
                        raise RuntimeError(
                            "saved layer hashes do not match ordered rootfs diff IDs"
                        )
                    uncompressed.seek(0)
                    try:
                        with tarfile.open(
                            fileobj=uncompressed, mode="r:*"
                        ) as layer_archive:
                            findings.extend(
                                _scan_layer_archive(layer_archive, layer=relative)
                            )
                    except tarfile.TarError as exc:
                        raise RuntimeError(
                            f"saved layer is not a readable tar archive: {relative}"
                        ) from exc
            else:
                diff_id = blob_digest
                uncompressed_size = blob_size
                if diff_id != expected_diff_id:
                    raise RuntimeError(
                        "saved layer hashes do not match ordered rootfs diff IDs"
                    )
                try:
                    with tarfile.open(
                        fileobj=layer_stream, mode="r|*"
                    ) as layer_archive:
                        findings.extend(
                            _scan_layer_archive(layer_archive, layer=relative)
                        )
                except tarfile.TarError as exc:
                    raise RuntimeError(
                        f"saved layer is not a readable tar archive: {relative}"
                    ) from exc

            layer_identities.append(
                {
                    "path": relative,
                    "compression": compression,
                    "blob_digest": blob_digest,
                    "blob_bytes": blob_size,
                    "diff_id": diff_id,
                    "uncompressed_bytes": uncompressed_size,
                }
            )
    return findings, {
        "image_id": (
            oci_identity["manifest_digest"]
            if oci_identity is not None
            else config_digest
        ),
        "image_config_digest": config_digest,
        "oci": oci_identity,
        "rootfs_diff_ids": diff_ids,
        "layers": layer_identities,
    }


def inspect_saved_image(image_tar: Path) -> tuple[list[Finding], dict]:
    """Normalize unexpected structural parser failures at the scanner boundary."""

    try:
        return _inspect_saved_image(image_tar)
    except RecursionError as exc:
        raise RuntimeError(
            "saved image contains excessively nested structural data"
        ) from exc
    except RuntimeError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, tarfile.TarError) as exc:
        raise RuntimeError("saved image contains malformed structural data") from exc


def _file_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _inspect_stable_saved_image(image_tar: Path) -> tuple[list[Finding], dict, dict]:
    archive_identity = _file_identity(image_tar)
    findings, identity = inspect_saved_image(image_tar)
    if _file_identity(image_tar) != archive_identity:
        raise RuntimeError("docker-save archive changed while it was being scanned")
    return findings, identity, archive_identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Local Docker image reference.")
    parser.add_argument("--tarball", type=Path, help="Existing docker-save tarball.")
    parser.add_argument(
        "--expected-image-id",
        help="Required immutable sha256 image ID for a supplied tarball.",
    )
    args = parser.parse_args(argv)
    if bool(args.image) == bool(args.tarball):
        parser.error("provide exactly one image or --tarball")
    if args.tarball and not re.fullmatch(
        r"sha256:[0-9a-f]{64}", args.expected_image_id or ""
    ):
        parser.error("--tarball requires --expected-image-id")
    if args.tarball:
        source = {"kind": "docker-save-tarball", "filename": args.tarball.name}
        expected_image_id = args.expected_image_id
        findings, identity, archive_identity = _inspect_stable_saved_image(args.tarball)
    else:
        source = {"kind": "local-docker-image"}
        expected_image_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format={{.Id}}", args.image],
            text=True,
        ).strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_id):
            raise RuntimeError("docker inspect did not return an immutable image ID")
        with tempfile.TemporaryDirectory(prefix="npa-seedvr2-image-") as scratch:
            tarball = Path(scratch) / "image.tar"
            subprocess.run(
                ["docker", "save", args.image, "-o", str(tarball)], check=True
            )
            findings, identity, archive_identity = _inspect_stable_saved_image(tarball)
    if expected_image_id not in {
        identity["image_id"],
        identity["image_config_digest"],
    }:
        raise RuntimeError("saved image identity differs from the expected image ID")
    report = {
        "format": "npa_seedvr2_payload_scan_v2",
        "source": source,
        "archive": archive_identity,
        "scanner_sha256": _file_identity(Path(__file__))["sha256"],
        "scan_complete": True,
        "expected_image_id": expected_image_id,
        "image_id": identity["image_id"],
        "image_config_digest": identity["image_config_digest"],
        "oci": identity["oci"],
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
