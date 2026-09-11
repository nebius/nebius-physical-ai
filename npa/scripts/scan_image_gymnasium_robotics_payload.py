#!/usr/bin/env python3
"""Inspect every config/history/layer/rootfs byte of a Docker-save archive.

The scanner is product-specific defense in depth. It cannot authorize release:
Phase A has no accepted manifest or native-byte policy, and incomplete locks are
always fatal.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tarfile
from typing import Any
import zipfile
import zlib

REQUIRED = {
    "opt/npa/gymnasium-robotics/source-lock.json",
    "opt/npa/gymnasium-robotics/apt-runtime.lock.json",
    "opt/npa/gymnasium-robotics/corresponding-source.lock.json",
    "opt/npa/gymnasium-robotics/asset-lock.json",
    "opt/npa/gymnasium-robotics/requirements.lock",
    "opt/npa/gymnasium-robotics/capability_smoke.py",
    "usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md",
    "usr/share/doc/npa-gymnasium-robotics/REDISTRIBUTION.md",
    "usr/share/source/npa-gymnasium-robotics/final-rootfs-manifest.json",
}
FORBIDDEN_PATH = re.compile(
    r"(^|/)(\.git|\.cache|pip-cache|apt/lists|apt/archives|\.aws|\.docker|\.ssh)(/|$)|"
    r"(^|/)(workspace/byof-runs|root/\.cache)(/|$)|"
    r"(^|/)(usr/local/cuda|opt/nvidia)(/|$)|"
    r"(^|/)[^/]*(?:nvidia|isaac|omniverse|ngc)[^/]*(/|$)|"
    r"(^|/)(libcuda[^/]*|libnvcuvid[^/]*|libnvoptix[^/]*)$|"
    r"\.(pt|pth|ckpt|safetensors|onnx|engine)$",
    re.IGNORECASE,
)
# This candidate has no NVIDIA runtime or driver payload. Rejecting any whole
# path component containing "nvidia" is intentionally broader than a package
# inventory so an unexpected vendor byte fails closed before publication.
SECRET_TEXT = re.compile(
    rb"BEGIN (?:RSA |OPENSSH )?PRIVATE"
    rb" KEY|"
    rb"(?i:(?:api[_-]?key|secret[_-]?key|password)\s*[=:]\s*[^\s]{8,})"
)
VENDOR_TEXT = re.compile(
    rb"(?i:(?:nvcr\.io|isaacsim|omniverse[/\\]kit|accept_eula\s*[=:]\s*(?:1|yes|true)))"
)
MAX_NESTED_ARCHIVE = 512 * 1024 * 1024
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
COMPRESSION_SIGNATURES = {
    "bzip2": b"BZh",
    "gzip": b"\x1f\x8b",
    "xz": b"\xfd7zXZ\x00",
}
DECLARED_COMPRESSION_SUFFIXES = (
    (".tar.bz2", "bzip2"),
    (".tar.gz", "gzip"),
    (".tar.xz", "xz"),
    (".tbz2", "bzip2"),
    (".tgz", "gzip"),
    (".txz", "xz"),
    (".bz2", "bzip2"),
    (".gz", "gzip"),
    (".xz", "xz"),
)
COMPRESSED_TAR_SUFFIXES = (
    ".tar.bz2",
    ".tar.gz",
    ".tar.xz",
    ".tbz2",
    ".tgz",
    ".txz",
)
DELIVERY_DIGEST_ROLES = {
    "farama-gymnasium-robotics": {"archive_sha256": "source_archive"},
    "shadow-hand-xml-mesh-texture-assets": {
        "build_instructions_sha256": "build_instructions",
        "preferred_form_archive_sha256": "preferred_form_archive",
        "transformation_manifest_sha256": "transformation_manifest",
    },
    "ubuntu-runtime-closure": {
        "binary_manifest_sha256": "binary_manifest",
        "build_materials_sha256": "build_materials",
        "source_manifest_sha256": "source_manifest",
    },
}
EXPECTED_SOURCE = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
EXPECTED_MUJOCO_COMMIT = "13827e9ee56f097f57acf69ae52b078f9839682d"
EXPECTED_SHADOW_COMMIT = "59d6bdf35bd9cf53185a20eb63413fdfe57fe77c"
EXPECTED_ASSET_LOCK = "e22eb62fc690a5e1d1ea931bab950392ca480caf3d51c7f16fd8cb4133d65568"
EXPECTED_BASE = {
    "image": "ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61",
    "config_digest": "sha256:b2b7ea366714195a1e1c5b2b578ece85c0b3920381a8654d038d9684f009613c",
    "layer_digest": "sha256:e51aee9c82ec5dd5ba2add49c45c6d85d460512757e2615b69bcdf9469c7cb58",
    # Filled only by a separately authorized transaction over the exact base
    # layer. Docker-save archives retain this uncompressed digest (diff ID),
    # not the registry's compressed layer digest.
    "uncompressed_layer_digest": None,
}
# A complete lock is trusted only after its exact bytes are independently
# reviewed and pinned here. Phase A deliberately has no such approved bytes,
# so changing a lock's status locally can never turn the scanner green.
EXPECTED_COMPLETE_LOCK_SHA256: dict[str, str | None] = {
    "source-lock.json": None,
    "apt-runtime.lock.json": None,
    "corresponding-source.lock.json": None,
    "requirements.lock": None,
}
EXPECTED_SOURCE_FIELDS = {
    "farama_gymnasium_robotics": {
        "archive_sha256": "ad8771ed6e9dd772b1101a25310ea46dd0f6f0044fbd6ea9af01af7b7c52c2c7",
        "commit": EXPECTED_SOURCE,
        "license": "MIT",
        "license_sha256": "00668424e12956742815eb1d8e15c7be543192561511df5fde119ae1188315ef",
        "repository": "https://github.com/Farama-Foundation/Gymnasium-Robotics",
        "version": "1.4.2",
    },
    "mujoco": {
        "commit": EXPECTED_MUJOCO_COMMIT,
        "license": "Apache-2.0",
        "license_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "repository": "https://github.com/google-deepmind/mujoco",
        "third_party_notices_sha256": "aec5167579b94d6926340175b4f764b5159f4933657556d89bbfb8238d3b3eb8",
        "version": "3.12.0",
        "wheel_sha256": "7ec16ce408871a0a9157cc556958ab66cd34db9fc1dccd3ef07717170163a4e0",
    },
    "shadow_sr_common": {
        "commit": EXPECTED_SHADOW_COMMIT,
        "license": "GPL-2.0-only",
        "license_sha256": "f9c375a1be4a41f7b70301dd83c91cb89e41567478859b77eef375a52d782505",
        "repository": "https://github.com/shadow-robot/sr_common",
    },
}
EXPECTED_PYTHON_DISTRIBUTIONS = {
    "absl-py": "2.5.0",
    "boto3": "1.43.91",
    "botocore": "1.43.91",
    "cloudpickle": "3.1.2",
    "etils": "1.14.0",
    "farama-notifications": "0.0.6",
    "fsspec": "2026.7.0",
    "glfw": "2.10.2",
    "gymnasium": "1.3.0",
    "imageio": "2.37.4",
    "jinja2": "3.1.6",
    "jmespath": "1.1.0",
    "markupsafe": "3.0.3",
    "mujoco": "3.12.0",
    "numpy": "2.5.3",
    "packaging": "26.3",
    "pettingzoo": "1.27.0",
    "pillow": "12.3.0",
    "pyopengl": "3.1.10",
    "python-dateutil": "2.9.0.post0",
    "s3transfer": "0.19.2",
    "setuptools": "84.0.0",
    "six": "1.17.0",
    "typing-extensions": "4.16.0",
    "urllib3": "2.7.0",
    "zipp": "4.1.0",
}
ROOTFS_MANIFEST = "usr/share/source/npa-gymnasium-robotics/final-rootfs-manifest.json"
ROOTFS_MANIFEST_EXEMPT = {
    ROOTFS_MANIFEST,
    "opt/npa/gymnasium-robotics/corresponding-source.lock.json",
}
ROOTFS_CLASSES = {
    "apt-runtime",
    "base",
    "corresponding-source",
    "farama-source",
    "license-notice",
    "npa-runtime",
    "python-runtime",
}


def _safe(name: str) -> str:
    original = PurePosixPath(name)
    if original.is_absolute() or ".." in original.parts:
        raise ValueError(f"unsafe archive path: {name}")
    normalized = name[2:] if name.startswith("./") else name
    path = PurePosixPath(normalized)
    if not path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return str(path)


def _raw_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"archive member is not a regular file: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"missing archive member: {name}")
    return stream.read()


def _scan_policy_bytes(label: str, content: bytes) -> None:
    if SECRET_TEXT.search(content):
        raise ValueError(f"forbidden secret signature: {label}")
    if VENDOR_TEXT.search(content):
        raise ValueError(f"forbidden vendor payload signature: {label}")


def _looks_like_tar(content: bytes) -> bool:
    """Recognize a valid first tar header, including pre-ustar archives."""

    if len(content) < 512 or not any(content[:512]):
        return False
    checksum_field = content[148:156].rstrip(b"\0 ").lstrip(b" ")
    try:
        expected = int(checksum_field or b"0", 8)
    except ValueError:
        return False
    actual = sum(content[:148]) + (8 * ord(" ")) + sum(content[156:512])
    return expected == actual and bool(content[:100].rstrip(b"\0"))


def _validated_tar_members(path: str, content: bytes) -> list[tarfile.TarInfo]:
    """Parse exactly one uncompressed tar stream with only zero end padding."""

    if len(content) < 1024 or len(content) % 512:
        raise ValueError(f"unaccounted tar bytes: {path}")
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            members = archive.getmembers()
    except tarfile.TarError as error:
        raise ValueError(f"unreadable nested tar archive: {path}") from error
    cursor = 0
    for member in members:
        if member.offset != cursor:
            raise ValueError(f"unaccounted tar bytes: {path}")
        cursor = ((member.offset_data + member.size + 511) // 512) * 512
    tail = content[cursor:]
    if len(tail) < 1024 or any(tail):
        raise ValueError(f"unaccounted tar bytes: {path}")
    return members


def _validated_zip_infos(path: str, content: bytes) -> list[zipfile.ZipInfo]:
    """Parse one prefix/suffix-free non-ZIP64 stream with no local-data gaps."""

    eocd_offset = content.rfind(b"PK\x05\x06", max(0, len(content) - 65_557))
    if eocd_offset < 0 or eocd_offset + 22 > len(content):
        raise ValueError(f"unaccounted zip bytes: {path}")
    (
        signature,
        disk,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", content, eocd_offset)
    if signature != b"PK\x05\x06" or eocd_offset + 22 + comment_size != len(content):
        raise ValueError(f"unaccounted zip bytes: {path}")
    if (
        disk != 0
        or central_disk != 0
        or disk_entries != total_entries
        or 0xFFFF in (disk_entries, total_entries)
        or 0xFFFFFFFF in (central_size, central_offset)
        or central_offset + central_size != eocd_offset
    ):
        raise ValueError(f"unsupported or prefixed zip archive: {path}")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
    except (RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError(f"unreadable nested zip archive: {path}") from error
    if len(infos) != total_entries:
        raise ValueError(f"zip entry count changed: {path}")
    ordered = sorted(infos, key=lambda item: item.header_offset)
    cursor = 0
    for index, info in enumerate(ordered):
        if info.header_offset != cursor or cursor + 30 > central_offset:
            raise ValueError(f"unaccounted zip bytes: {path}")
        (
            local_signature,
            _version,
            local_flags,
            local_compression,
            _mtime,
            _mdate,
            local_crc,
            local_compressed_size,
            local_size,
            filename_size,
            extra_size,
        ) = struct.unpack_from("<4s5H3L2H", content, cursor)
        if (
            local_signature != b"PK\x03\x04"
            or local_flags != info.flag_bits
            or local_compression != info.compress_type
            or local_flags & 1
        ):
            raise ValueError(f"unsupported zip local header: {path}")
        data_start = cursor + 30 + filename_size + extra_size
        data_end = data_start + info.compress_size
        next_offset = (
            ordered[index + 1].header_offset
            if index + 1 < len(ordered)
            else central_offset
        )
        descriptor = content[data_end:next_offset]
        if local_flags & 0x08:
            if len(descriptor) not in (12, 16) or (
                len(descriptor) == 16 and not descriptor.startswith(b"PK\x07\x08")
            ):
                raise ValueError(f"unsupported zip data descriptor: {path}")
        elif descriptor or (
            local_crc != info.CRC
            or local_compressed_size != info.compress_size
            or local_size != info.file_size
        ):
            raise ValueError(f"unaccounted zip bytes: {path}")
        cursor = next_offset
    if cursor != central_offset:
        raise ValueError(f"unaccounted zip bytes: {path}")
    return infos


def _declared_compression(path: str) -> str | None:
    lowered = path.lower()
    return next(
        (
            kind
            for suffix, kind in DECLARED_COMPRESSION_SUFFIXES
            if lowered.endswith(suffix)
        ),
        None,
    )


def _decompress(path: str, content: bytes, kind: str) -> bytes:
    """Expand one recognized stream with a strict output-size bound."""

    try:
        if kind == "gzip":
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            expanded = decompressor.decompress(content, MAX_NESTED_ARCHIVE + 1)
            if (
                not decompressor.eof
                or decompressor.unconsumed_tail
                or decompressor.unused_data
            ):
                raise ValueError(f"ambiguous compressed stream: {path}")
        elif kind == "bzip2":
            decompressor = bz2.BZ2Decompressor()
            expanded = decompressor.decompress(
                content, max_length=MAX_NESTED_ARCHIVE + 1
            )
            if not decompressor.eof or decompressor.unused_data:
                raise ValueError(f"ambiguous compressed stream: {path}")
        else:
            decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
            expanded = decompressor.decompress(
                content, max_length=MAX_NESTED_ARCHIVE + 1
            )
            if not decompressor.eof or decompressor.unused_data:
                raise ValueError(f"ambiguous compressed stream: {path}")
    except (EOFError, OSError, lzma.LZMAError, zlib.error) as error:
        raise ValueError(f"unreadable compressed stream: {path}") from error
    if len(expanded) > MAX_NESTED_ARCHIVE:
        raise ValueError(f"expanded stream exceeds scan bound: {path}")
    return expanded


def _nested_archive_members(path: str, content: bytes, *, depth: int = 0) -> int:
    """Inspect retained archives by validated bytes, never only by filename."""

    if depth > 8:
        raise ValueError(f"nested archive depth exceeds scan bound: {path}")
    _scan_policy_bytes(f"nested archive bytes: {path}", content)
    lowered = path.lower()
    declared_zip = lowered.endswith((".whl", ".zip"))
    declared_tar = lowered.endswith((".tar", *COMPRESSED_TAR_SUFFIXES))
    declared_plain_tar = lowered.endswith(".tar")
    declared_compression = _declared_compression(path)
    is_tar = _looks_like_tar(content)
    is_zip = not is_tar and zipfile.is_zipfile(io.BytesIO(content))
    compression_kind = next(
        (
            kind
            for kind, signature in COMPRESSION_SIGNATURES.items()
            if content.startswith(signature)
        ),
        None,
    )
    if declared_compression is not None and compression_kind != declared_compression:
        raise ValueError(f"declared compression does not match bytes: {path}")
    if declared_zip and not is_zip:
        raise ValueError(f"declared ZIP does not match bytes: {path}")
    if declared_plain_tar and not is_tar:
        raise ValueError(f"declared tar does not match bytes: {path}")
    archive_like = (
        is_zip
        or compression_kind is not None
        or is_tar
        or declared_zip
        or declared_tar
        or content.startswith(ZIP_SIGNATURES)
    )
    if archive_like and len(content) > MAX_NESTED_ARCHIVE:
        raise ValueError(f"nested archive exceeds scan bound: {path}")
    if compression_kind is not None:
        expanded = _decompress(path, content, compression_kind)
        _scan_policy_bytes(f"expanded {compression_kind} stream: {path}", expanded)
        if lowered.endswith(COMPRESSED_TAR_SUFFIXES) and not _looks_like_tar(expanded):
            raise ValueError(f"compressed tar payload is not a tar archive: {path}")
        return _nested_archive_members(
            f"{path}:expanded-{compression_kind}", expanded, depth=depth + 1
        )
    if is_zip:
        infos = _validated_zip_infos(path, content)
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                count = 0
                expanded_total = 0
                for member in infos:
                    safe = _safe(member.filename)
                    count += 1
                    if FORBIDDEN_PATH.search(safe):
                        raise ValueError(
                            f"forbidden nested archive member: {path}:{safe}"
                        )
                    if member.is_dir():
                        continue
                    if member.file_size > MAX_NESTED_ARCHIVE:
                        raise ValueError(
                            f"nested archive member exceeds scan bound: {path}:{safe}"
                        )
                    expanded_total += member.file_size
                    if expanded_total > MAX_NESTED_ARCHIVE:
                        raise ValueError(
                            f"nested archive expansion exceeds scan bound: {path}"
                        )
                    nested_content = archive.read(member)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        target = nested_content.decode(
                            "utf-8", errors="surrogateescape"
                        )
                        resolved = _resolved_link_target(safe, target, relative=True)
                        if FORBIDDEN_PATH.search(
                            target.lstrip("/")
                        ) or FORBIDDEN_PATH.search(resolved):
                            raise ValueError(
                                f"forbidden nested archive link: {path}:{safe}"
                            )
                        _scan_policy_bytes(
                            f"nested archive link: {path}:{safe}", nested_content
                        )
                        continue
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                        raise ValueError(
                            f"unsupported nested archive member type: {path}:{safe}"
                        )
                    _scan_policy_bytes(
                        f"nested archive member: {path}:{safe}", nested_content
                    )
                    count += _nested_archive_members(
                        f"{path}:{safe}", nested_content, depth=depth + 1
                    )
                return count
        except zipfile.BadZipFile as error:
            raise ValueError(f"unreadable nested zip archive: {path}") from error
    if content.startswith(ZIP_SIGNATURES) or declared_zip:
        raise ValueError(f"unreadable nested zip archive: {path}")
    if is_tar:
        _validated_tar_members(path, content)
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as archive:
                count = 0
                expanded_total = 0
                for member in archive:
                    safe = _safe(member.name)
                    count += 1
                    if FORBIDDEN_PATH.search(safe):
                        raise ValueError(
                            f"forbidden nested archive member: {path}:{safe}"
                        )
                    if member.isfile():
                        if member.size > MAX_NESTED_ARCHIVE:
                            raise ValueError(
                                f"nested archive member exceeds scan bound: {path}:{safe}"
                            )
                        expanded_total += member.size
                        if expanded_total > MAX_NESTED_ARCHIVE:
                            raise ValueError(
                                f"nested archive expansion exceeds scan bound: {path}"
                            )
                        stream = archive.extractfile(member)
                        nested_content = stream.read() if stream is not None else b""
                        if stream is None:
                            raise ValueError(
                                f"forbidden nested archive bytes: {path}:{safe}"
                            )
                        _scan_policy_bytes(
                            f"nested archive member: {path}:{safe}", nested_content
                        )
                        count += _nested_archive_members(
                            f"{path}:{safe}", nested_content, depth=depth + 1
                        )
                    elif member.issym() or member.islnk():
                        target_text = member.linkname
                        target = target_text.encode("utf-8", errors="surrogateescape")
                        _scan_policy_bytes(
                            f"nested archive link: {path}:{safe}", target
                        )
                        resolved = _resolved_link_target(
                            safe, target_text, relative=member.issym()
                        )
                        if FORBIDDEN_PATH.search(
                            target_text.lstrip("/")
                        ) or FORBIDDEN_PATH.search(resolved):
                            raise ValueError(
                                f"forbidden nested archive link: {path}:{safe}"
                            )
                    elif not member.isdir():
                        raise ValueError(
                            f"unsupported nested archive member type: {path}:{safe}"
                        )
                return count
        except tarfile.TarError as error:
            raise ValueError(f"unreadable nested tar archive: {path}") from error
    if declared_tar:
        raise ValueError(f"unreadable nested tar archive: {path}")
    return 0


def _resolved_link_target(path: str, target: str, *, relative: bool) -> str:
    candidate = PurePosixPath(target)
    parts = (
        list(PurePosixPath(path).parent.parts)
        if relative and not candidate.is_absolute()
        else []
    )
    for part in candidate.parts:
        if part in ("", ".", "/"):
            continue
        if part == "..":
            if not parts:
                raise ValueError(
                    f"archive link escapes the image root: {path} -> {target}"
                )
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise ValueError(f"archive link has an empty root target: {path} -> {target}")
    return str(PurePosixPath(*parts))


def _remove_path(
    rootfs: dict[str, bytes], entries: dict[str, dict[str, Any]], target: str
) -> None:
    for key in tuple(entries):
        if key == target or key.startswith(target + "/"):
            entries.pop(key, None)
            rootfs.pop(key, None)


def _entry_metadata(item: tarfile.TarInfo, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "mode": item.mode,
        "uid": item.uid,
        "gid": item.gid,
        "mtime": item.mtime,
        "uname": item.uname or "",
        "gname": item.gname or "",
        "pax_headers": dict(sorted(item.pax_headers.items())),
    }


def _whiteout_metadata(
    layer: tarfile.TarFile,
    item: tarfile.TarInfo,
    path: str,
    layer_name: str,
) -> dict[str, Any]:
    """Validate one OCI whiteout before applying its filesystem semantics."""

    if (
        not item.isfile()
        or item.size != 0
        or item.linkname
        or (item.devmajor or 0) != 0
        or (item.devminor or 0) != 0
        or item.pax_headers
    ):
        raise ValueError(f"invalid whiteout entry: {path}")
    payload = layer.extractfile(item)
    if payload is None or payload.read() != b"":
        raise ValueError(f"invalid whiteout entry: {path}")
    return {
        **_entry_metadata(item, "whiteout"),
        "layer": layer_name,
        "path": path,
    }


def _layers(
    archive: tarfile.TarFile, names: list[str]
) -> tuple[
    dict[str, bytes],
    dict[str, dict[str, Any]],
    int,
    int,
    list[str],
    list[dict[str, Any]],
]:
    rootfs: dict[str, bytes] = {}
    entries: dict[str, dict[str, Any]] = {}
    total = 0
    nested = 0
    diff_ids: list[str] = []
    whiteouts: list[dict[str, Any]] = []
    for layer_name in names:
        raw = _raw_member(archive, layer_name)
        _scan_policy_bytes(f"raw layer bytes: {layer_name}", raw)
        _validated_tar_members(f"raw layer: {layer_name}", raw)
        diff_ids.append("sha256:" + hashlib.sha256(raw).hexdigest())
        current_rootfs: dict[str, bytes] = {}
        current_entries: dict[str, dict[str, Any]] = {}
        current_order: list[str] = []
        member_paths: set[str] = set()
        opaque_parents: set[str] = set()
        deleted_targets: set[str] = set()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as layer:
            for item in layer:
                path = _safe(item.name)
                if path in member_paths:
                    raise ValueError(f"duplicate normalized layer path: {path}")
                member_paths.add(path)
                total += 1
                leaf = PurePosixPath(path).name
                if leaf == ".wh..wh..opq":
                    whiteouts.append(_whiteout_metadata(layer, item, path, layer_name))
                    parent = str(PurePosixPath(path).parent)
                    opaque_parents.add(parent)
                    continue
                if leaf.startswith(".wh."):
                    whiteouts.append(_whiteout_metadata(layer, item, path, layer_name))
                    target = str(
                        PurePosixPath(path).with_name(leaf.removeprefix(".wh."))
                    )
                    deleted_targets.add(target)
                    continue
                if FORBIDDEN_PATH.search(path):
                    raise ValueError(f"forbidden image path: {path}")
                current_order.append(path)
                if item.isfile():
                    payload = layer.extractfile(item)
                    if payload is None:
                        raise ValueError(f"unreadable layer file: {path}")
                    content = payload.read()
                    _scan_policy_bytes(path, content)
                    nested += _nested_archive_members(path, content)
                    _remove_path(current_rootfs, current_entries, path)
                    current_rootfs[path] = content
                    current_entries[path] = {
                        **_entry_metadata(item, "regular"),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                elif item.isdir():
                    current_rootfs.pop(path, None)
                    current_entries[path] = _entry_metadata(item, "directory")
                elif item.issym() or item.islnk():
                    target = item.linkname
                    target_bytes = target.encode("utf-8", errors="surrogateescape")
                    _scan_policy_bytes(f"link target: {path}", target_bytes)
                    resolved = _resolved_link_target(
                        path, target, relative=item.issym()
                    )
                    if FORBIDDEN_PATH.search(
                        target.lstrip("/")
                    ) or FORBIDDEN_PATH.search(resolved):
                        raise ValueError(
                            f"forbidden image link target: {path} -> {target}"
                        )
                    _remove_path(current_rootfs, current_entries, path)
                    current_rootfs.pop(path, None)
                    kind = "symlink" if item.issym() else "hardlink"
                    current_entries[path] = {
                        **_entry_metadata(item, kind),
                        "link_target": target,
                        "resolved_link_target": resolved,
                    }
                else:
                    raise ValueError(f"unsupported image member type: {path}")
        for parent in opaque_parents:
            if parent == ".":
                rootfs.clear()
                entries.clear()
            else:
                for key in tuple(entries):
                    if key.startswith(parent + "/"):
                        entries.pop(key, None)
                        rootfs.pop(key, None)
        for target in deleted_targets:
            _remove_path(rootfs, entries, target)
        for path in current_order:
            record = current_entries.get(path)
            if record is None:
                continue
            if record["kind"] == "directory":
                rootfs.pop(path, None)
            else:
                _remove_path(rootfs, entries, path)
            entries[path] = record
            if record["kind"] == "regular":
                rootfs[path] = current_rootfs[path]
    for path, record in entries.items():
        if record["kind"] == "hardlink":
            target = record["resolved_link_target"]
            if target not in entries or entries[target]["kind"] != "regular":
                raise ValueError(
                    f"hardlink target is not a retained regular file: {path}"
                )
    return rootfs, entries, total, nested, diff_ids, whiteouts


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not value or any(_missing(item) for item in value)
    if isinstance(value, dict):
        return not value or any(_missing(item) for item in value.values())
    return False


def _normalize_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _locked_python_distributions(raw: bytes) -> dict[str, str]:
    text = raw.decode("utf-8")
    if "# status: complete" not in text:
        raise ValueError("Python lock is incomplete")
    logical: list[str] = []
    pending = ""
    for source_line in text.splitlines():
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending:
        raise ValueError("Python lock ends with an incomplete continuation")
    distributions: dict[str, str] = {}
    pattern = re.compile(
        r"(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?"
        r"==(?P<version>[^\s]+)"
        r"(?:\s+--hash=sha256:[0-9a-f]{64})+"
    )
    for requirement in logical:
        match = pattern.fullmatch(requirement)
        if match is None:
            raise ValueError(f"unhashed or malformed Python lock entry: {requirement}")
        name = _normalize_distribution(match.group("name"))
        if name in distributions:
            raise ValueError(f"duplicate Python lock distribution: {name}")
        distributions[name] = match.group("version")
    return distributions


def _classified_rootfs(
    rootfs: dict[str, bytes],
    entries: dict[str, dict[str, Any]],
    expected_digest: str,
) -> None:
    raw = rootfs[ROOTFS_MANIFEST]
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("final rootfs classification manifest changed")
    manifest = json.loads(raw)
    if manifest.get("schema") != "npa.gymnasium-robotics.rootfs-manifest.v2":
        raise ValueError("final rootfs classification manifest schema changed")
    classified = manifest.get("entries")
    if not isinstance(classified, dict):
        raise ValueError("final rootfs classification manifest has no entry map")
    expected_paths = set(entries)
    if set(classified) != expected_paths:
        raise ValueError("final rootfs contains missing or unclassified entries")
    for exempt in ROOTFS_MANIFEST_EXEMPT:
        if exempt not in rootfs or entries.get(exempt, {}).get("kind") != "regular":
            raise ValueError(
                f"rootfs classification trust file is not regular: {exempt}"
            )
    for path, record in classified.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid rootfs classification: {path}")
        source = record.get("source")
        if (
            record.get("class") not in ROOTFS_CLASSES
            or not isinstance(source, str)
            or not source
        ):
            raise ValueError(f"invalid rootfs classification: {path}")
        expected_entry = dict(entries[path])
        if path in ROOTFS_MANIFEST_EXEMPT:
            # The exact contents are independently pinned: the corresponding
            # lock by EXPECTED_COMPLETE_LOCK_SHA256, and this manifest by that
            # lock. Omitting only their content hashes avoids a digest cycle;
            # every other retained attribute remains classified.
            expected_entry.pop("sha256", None)
        expected = {**expected_entry, "class": record["class"], "source": source}
        if record != expected:
            raise ValueError(f"classified rootfs entry metadata changed: {path}")


def _locked_assets(rootfs: dict[str, bytes]) -> None:
    raw_lock = rootfs["opt/npa/gymnasium-robotics/asset-lock.json"]
    if hashlib.sha256(raw_lock).hexdigest() != EXPECTED_ASSET_LOCK:
        raise ValueError("approved asset lock bytes changed")
    lock = json.loads(raw_lock)
    if lock.get("source_commit") != EXPECTED_SOURCE:
        raise ValueError("asset lock source commit changed")
    groups = (lock.get("directly_loaded_xml"), lock.get("directly_loaded_mesh_texture"))
    if not all(isinstance(group, dict) for group in groups):
        raise ValueError("asset lock groups are missing")
    if len(groups[0]) != 5 or len(groups[1]) != 14:
        raise ValueError(
            "asset lock is not the exact 5 XML plus 14 mesh/texture closure"
        )
    prefix = "opt/venv/lib/python3.12/site-packages/gymnasium_robotics/envs/assets/"
    for group in groups:
        for name, digest in group.items():
            content = rootfs.get(prefix + name)
            if content is None or hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"installed upstream asset hash changed: {name}")
    notice = rootfs.get(prefix + "LICENSE.md")
    if notice is None or hashlib.sha256(notice).hexdigest() != lock.get(
        "asset_notice_sha256"
    ):
        raise ValueError("installed Shadow Hand asset notice changed")


def _complete_locks(
    rootfs: dict[str, bytes],
    entries: dict[str, dict[str, Any]],
    layer_diff_ids: list[str],
) -> None:
    parsed: dict[str, dict[str, Any]] = {}
    json_locks = (
        "source-lock.json",
        "apt-runtime.lock.json",
        "corresponding-source.lock.json",
    )
    for name in (*json_locks, "requirements.lock"):
        path = f"opt/npa/gymnasium-robotics/{name}"
        expected_lock = EXPECTED_COMPLETE_LOCK_SHA256[name]
        if not isinstance(expected_lock, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_lock
        ):
            raise ValueError(f"reviewed complete lock digest is not configured: {name}")
        if hashlib.sha256(rootfs[path]).hexdigest() != expected_lock:
            raise ValueError(f"reviewed complete lock bytes changed: {name}")
    for name in json_locks:
        path = f"opt/npa/gymnasium-robotics/{name}"
        payload = json.loads(rootfs[path])
        parsed[name] = payload
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete evidence lock in image: {name}")
        if _missing(payload):
            raise ValueError(f"complete evidence lock contains missing values: {name}")
    components = parsed["source-lock.json"].get("components", {})
    if set(components) != set(EXPECTED_SOURCE_FIELDS):
        raise ValueError("source lock component closure changed")
    for name, expected in EXPECTED_SOURCE_FIELDS.items():
        if any(components[name].get(key) != value for key, value in expected.items()):
            raise ValueError(f"source lock identity changed: {name}")
    apt = parsed["apt-runtime.lock.json"]
    if _missing(EXPECTED_BASE):
        raise ValueError("reviewed Ubuntu base diff ID is not configured")
    if apt.get("base") != EXPECTED_BASE:
        raise ValueError("Ubuntu base identity changed")
    if (
        not layer_diff_ids
        or layer_diff_ids[0] != EXPECTED_BASE["uncompressed_layer_digest"]
    ):
        raise ValueError(
            "saved image does not begin with the reviewed Ubuntu base layer"
        )
    if not apt.get("resolved_binary_packages") or not apt.get(
        "resolved_source_packages"
    ):
        raise ValueError("APT binary/source closure is empty")
    requirements = rootfs["opt/npa/gymnasium-robotics/requirements.lock"]
    if _locked_python_distributions(requirements) != EXPECTED_PYTHON_DISTRIBUTIONS:
        raise ValueError("Python distribution closure changed")
    annex = "usr/share/source/npa-gymnasium-robotics/"
    if not any(path.startswith(annex) for path in rootfs):
        raise ValueError("corresponding-source annex is empty")
    source_lock = parsed["source-lock.json"]
    shadow = components["shadow_sr_common"]
    for field in ("preferred_form_sha256", "transformation_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(shadow.get(field) or "")):
            raise ValueError(f"Shadow preferred-form closure is incomplete: {field}")
    deliveries = parsed["corresponding-source.lock.json"].get("deliveries", [])
    expected_deliveries = {
        "farama-gymnasium-robotics",
        "shadow-hand-xml-mesh-texture-assets",
        "ubuntu-runtime-closure",
    }
    if {
        delivery.get("binary_component") for delivery in deliveries
    } != expected_deliveries:
        raise ValueError("corresponding-source delivery closure changed")
    delivery_map = {delivery["binary_component"]: delivery for delivery in deliveries}
    farama_delivery = delivery_map["farama-gymnasium-robotics"]
    shadow_delivery = delivery_map["shadow-hand-xml-mesh-texture-assets"]
    ubuntu_delivery = delivery_map["ubuntu-runtime-closure"]
    if (
        farama_delivery.get("archive_sha256")
        != source_lock["components"]["farama_gymnasium_robotics"]["archive_sha256"]
        or farama_delivery.get("repository")
        != EXPECTED_SOURCE_FIELDS["farama_gymnasium_robotics"]["repository"]
        or farama_delivery.get("source_commit") != EXPECTED_SOURCE
    ):
        raise ValueError("conveyed Farama source identity changed")
    shadow_fields = {
        "build_instructions_sha256": None,
        "preferred_form_archive_sha256": shadow["preferred_form_sha256"],
        "source_commit": EXPECTED_SHADOW_COMMIT,
        "transformation_manifest_sha256": shadow["transformation_manifest_sha256"],
    }
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(shadow_delivery.get(field) or ""))
        if expected is None
        else shadow_delivery.get(field) != expected
        for field, expected in shadow_fields.items()
    ):
        raise ValueError("Shadow preferred-form delivery mapping is incomplete")
    for field in (
        "binary_manifest_sha256",
        "build_materials_sha256",
        "source_manifest_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(ubuntu_delivery.get(field) or "")):
            raise ValueError(
                f"Ubuntu corresponding-source closure is incomplete: {field}"
            )
    artifact_paths: set[str] = set()
    artifact_digests: set[str] = set()
    for delivery in deliveries:
        component = delivery["binary_component"]
        digest_roles = DELIVERY_DIGEST_ROLES[component]
        expected_roles = set(digest_roles.values())
        if set(delivery.get("required_artifact_roles") or []) != expected_roles:
            raise ValueError(
                f"corresponding-source required roles changed: {component}"
            )
        artifacts = delivery.get("artifacts")
        if not artifacts:
            raise ValueError("corresponding-source delivery has no exact artifacts")
        roles = [artifact.get("role") for artifact in artifacts]
        if len(roles) != len(set(roles)) or set(roles) != expected_roles:
            raise ValueError(
                f"corresponding-source artifact roles changed: {component}"
            )
        artifacts_by_role = {artifact["role"]: artifact for artifact in artifacts}
        for digest_field, role in digest_roles.items():
            digest = delivery.get(digest_field)
            if (
                not re.fullmatch(r"[0-9a-f]{64}", str(digest or ""))
                or artifacts_by_role[role].get("sha256") != digest
            ):
                raise ValueError(
                    f"corresponding-source role digest changed: {component}:{role}"
                )
        for artifact in artifacts:
            artifact_path = _safe(str(artifact.get("path") or ""))
            if artifact_path in artifact_paths:
                raise ValueError(
                    f"corresponding-source artifact path is reused: {artifact_path}"
                )
            artifact_paths.add(artifact_path)
            artifact_digest = str(artifact.get("sha256") or "")
            if artifact_digest in artifact_digests:
                raise ValueError(
                    f"corresponding-source artifact digest is reused: {artifact_digest}"
                )
            artifact_digests.add(artifact_digest)
            content = rootfs.get(annex + artifact_path)
            if content is None or hashlib.sha256(content).hexdigest() != artifact.get(
                "sha256"
            ):
                raise ValueError("corresponding-source annex artifact changed")
    if not any(
        artifact.get("sha256") == farama_delivery["archive_sha256"]
        for artifact in farama_delivery["artifacts"]
    ):
        raise ValueError("exact Farama source archive is absent from the annex")
    if not any(
        artifact.get("sha256") == shadow_delivery["preferred_form_archive_sha256"]
        for artifact in shadow_delivery["artifacts"]
    ):
        raise ValueError("exact Shadow preferred form is absent from the annex")
    manifest_digest = parsed["corresponding-source.lock.json"].get(
        "final_rootfs_manifest_sha256"
    )
    _classified_rootfs(rootfs, entries, manifest_digest)
    _locked_assets(rootfs)


def scan(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    archive_bytes = path.read_bytes()
    _scan_policy_bytes("complete Docker-save archive", archive_bytes)
    _validated_tar_members("Docker-save archive", archive_bytes)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        outer_members = archive.getmembers()
        outer_by_name = {_safe(member.name): member for member in outer_members}
        if len(outer_by_name) != len(outer_members):
            raise ValueError("Docker save contains duplicate normalized member paths")
        outer_names = set(outer_by_name)
        manifest_raw = _raw_member(archive, "manifest.json")
        _scan_policy_bytes("Docker-save manifest", manifest_raw)
        manifest = json.loads(manifest_raw)
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise ValueError("Docker save must contain exactly one image")
        entry = manifest[0]
        if not isinstance(entry, dict):
            raise ValueError("Docker save manifest entry must be an object")
        config_name = _safe(str(entry["Config"]))
        config_raw = _raw_member(archive, config_name)
        config_digest = hashlib.sha256(config_raw).hexdigest()
        if config_name != f"{config_digest}.json":
            raise ValueError(
                "Docker save config filename does not bind its exact bytes"
            )
        _scan_policy_bytes("exact image config", config_raw)
        config = json.loads(config_raw)
        if not isinstance(config, dict):
            raise ValueError("Docker save config must be an object")
        runtime_config = config.get("config")
        if (
            not isinstance(runtime_config, dict)
            or runtime_config.get("User") != "ubuntu"
        ):
            raise ValueError("final image must declare the non-root ubuntu user")
        layers = [_safe(str(name)) for name in entry.get("Layers", [])]
        if not layers:
            raise ValueError("Docker save contains no layers")
        if len(layers) != len(set(layers)):
            raise ValueError("Docker save repeats an ordered layer")
        allowed_outer = {"manifest.json", config_name, *layers, "repositories"}
        allowed_directories = {
            str(parent)
            for name in allowed_outer
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
        unexpected = sorted(outer_names - allowed_outer - allowed_directories)
        if unexpected:
            raise ValueError(f"unexpected Docker-save members: {unexpected}")
        for directory in outer_names & allowed_directories:
            if not outer_by_name[directory].isdir():
                raise ValueError(
                    f"Docker-save parent member is not a directory: {directory}"
                )
        if "repositories" in outer_names:
            _scan_policy_bytes(
                "Docker-save repositories", _raw_member(archive, "repositories")
            )
        (
            rootfs,
            entries,
            layer_members,
            nested_members,
            layer_diff_ids,
            whiteouts,
        ) = _layers(archive, layers)
    config_rootfs = config.get("rootfs")
    if (
        not isinstance(config_rootfs, dict)
        or config_rootfs.get("type") != "layers"
        or config_rootfs.get("diff_ids") != layer_diff_ids
    ):
        raise ValueError(
            "image config rootfs diff IDs do not match ordered layer bytes"
        )
    missing = sorted(REQUIRED - rootfs.keys())
    if missing:
        raise ValueError(f"required image files absent: {missing}")
    _complete_locks(rootfs, entries, layer_diff_ids)
    return {
        "schema": "npa.gymnasium-robotics.payload-scan.v1",
        "status": "passed",
        "archive_sha256": archive_sha256,
        "config_sha256": config_digest,
        "layer_count": len(layers),
        "layer_member_count": layer_members,
        "nested_archive_member_count": nested_members,
        "ordered_layer_diff_ids": layer_diff_ids,
        "whiteout_entry_count": len(whiteouts),
        "whiteout_metadata_sha256": hashlib.sha256(
            json.dumps(whiteouts, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "final_entry_count": len(entries),
        "final_regular_file_count": len(rootfs),
        "unresolved_findings": 0,
        "accepted_manifest_present": False,
        "release_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-save", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = scan(args.docker_save)
    except (
        KeyError,
        OSError,
        ValueError,
        tarfile.TarError,
        json.JSONDecodeError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
