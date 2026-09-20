#!/usr/bin/env python3
"""Inspect every config/history/layer/rootfs byte of a container archive.

The scanner is product-specific defense in depth. It cannot authorize release:
The neutral candidate has no accepted manifest or native-byte policy. It must
contain no upstream solution, Shadow, MuJoCo/Python workload, vendor runtime,
cache, credential, dataset, checkpoint, or output byte.
"""

from __future__ import annotations

import argparse
import bz2
from datetime import datetime
import hashlib
import io
import json
import lzma
import math
import os
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
    "opt/npa/gymnasium-robotics/runtime-bootstrap.py",
    "opt/npa/gymnasium-robotics/capability_smoke.py",
    "opt/npa/gymnasium-robotics/verify_image.py",
    "opt/npa/gymnasium-robotics/runtime-fetch-manifest.json",
    "usr/local/bin/npa-gymnasium-entrypoint",
    "usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md",
    "usr/share/doc/npa-gymnasium-robotics/REDISTRIBUTION.md",
}
FORBIDDEN_PATH = re.compile(
    r"(^|/)(\.git|\.cache|pip-cache|apt/lists|apt/archives|\.aws|\.docker|\.ssh)(/|$)|"
    r"(^|/)(workspace/byof-runs|root/\.cache)(/|$)|"
    r"(^|/)(opt/venv|wheelhouse|runtime-cache)(/|$)|"
    r"(^|/)usr/share/source/npa-gymnasium-robotics(/|$)|"
    r"(^|/)gymnasium_robotics(/|$)|"
    r"(^|/)[^/]*(?:shadow[_-]?hand|mujoco)[^/]*(/|$)|"
    r"(^|/)(usr/local/cuda|opt/nvidia)(/|$)|"
    r"(^|/)(?:nvidia|isaac|isaacsim|omniverse|ngc)(?:[-_.][^/]*)?(?:/|$)|"
    r"(^|/)[^/]*(?:[-_.](?:nvidia|isaac|isaacsim|omniverse|ngc))(?:[-_.][^/]*)?(?:/|$)|"
    r"(^|/)(libcuda[^/]*|libnvcuvid[^/]*|libnvoptix[^/]*)$|"
    r"\.(whl|pt|pth|ckpt|safetensors|onnx|engine)$",
    re.IGNORECASE,
)
UPSTREAM_TREE_PATH = re.compile(
    r"(^|/)Gymnasium-Robotics-[0-9a-f]{7,40}(/|$)|"
    r"(^|/)gymnasium_robotics(/|$)",
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
KNOWN_FORBIDDEN_CONTENT_SHA256 = frozenset(
    {
        "ad8771ed6e9dd772b1101a25310ea46dd0f6f0044fbd6ea9af01af7b7c52c2c7",
        "7ec16ce408871a0a9157cc556958ab66cd34db9fc1dccd3ef07717170163a4e0",
        "c1004adf05daea7b57ee57643f8e94945393a2e12b34b18a244be3c1776776b2",
        "9b6f70c49c8bb043ce3e52d181a0797c514a3b85f44f0cf59600d9f906df9f64",
        "fbc404e52def67e38222a395fcc39c051eacc503dbf8e22a22a5bcce36992524",
        "95a9153e6bba4c555ad7cdcea746520eab38e0d57415fa9b3bfaf987dd0418f1",
        "248bc2cb73920c88903786842aaa5476262d550bd01c2c716dd5f2ee642f3318",
        "a2d35742067f71e4888954d1aa55d043dbc6ec0c63f4cc715f339cb45fd13734",
        "0eeb932dcc102dd1fc6bef55fe83f6a74d97aebd32c34d6ee7020c19647306c5",
        "d11a521ae498e947491ba947df8f999136fdf1401d1ba86d594eec20da656ad1",
        "83fd93c9e4c1bf240aa6def2e5fdf5b1adcf4e341c146b05313aeaa8c54fd36a",
        "98e26cf013cd8d7f6ed890d3402ba0aceb11ec5c78779e388390ca2cddc7daf2",
        "80aae0002a6684428278cf214a31040f7cd7aeb3e2e648bd4b68baa375c2d2ad",
        "01aca61837d9db13c52dd170245cacb6f1cedc8238d66fe7239602c80c7a0130",
        "874ae2ac813c6c9d873d04f872cb13de3cabd427cb24a8924619d0a1719d6da7",
        "a8a9c1659aa4391291531b46997ba5eeda1b36e597070ae8a3cd824159937aae",
        "b6901da4d4b93059c02b8803650ad484b3f31aebce46badfa411897d521a4877",
        "12593bcba03bbf278cd7d7db3ca79ba753ada3cb58ee93e725603e9b5e29fd0c",
        "47d0f252ae456541993746ff76d7022135b1bf54263f8f83b19d61f6f94871c9",
        "d39ffb85a87d00c346764191e38ecac3135f2d6f690f64a8ee5da4783ef18e76",
        "3649cb94a9a5f74751d15c0f38291dd666b7eecc286977888b64b6c3c626c9d3",
    }
)
# These are bootstrap implementation bytes installed by the two exact Ubuntu
# packages named below, not runtime-acquired workload wheels.  They are allowed
# only at these paths and only with these extracted-file hashes; their .deb
# hashes remain independently bound by apt-runtime.lock.json.
EXPECTED_SYSTEM_WHEEL_FILES = {
    "usr/share/python-wheels/pip-24.0-py3-none-any.whl": {
        "sha256": "e995a37590643450898cfa5bd5113831a547506cd545c335a409339d0c1e87ab",
        "package": "python3-pip-whl",
        "package_sha256": "4b7c50db8f261b208c1d9cde8db148c1f682cc516957b986ada5088cfcee1359",
    },
    "usr/share/python-wheels/setuptools-68.1.2-py3-none-any.whl": {
        "sha256": "fcfc63a09d24f6195a4c89e8e55323331857ff3711f9f0f574152e76b6f7d8ba",
        "package": "python3-setuptools-whl",
        "package_sha256": "edfa94cc1f6a33af99cfaf6ebfe35dbcd9c4bdd8555b90c0d8e78479faf5c8f0",
    },
}
MAX_NESTED_ARCHIVE = 512 * 1024 * 1024
MAX_NESTED_ARCHIVE_EXPANDED_BYTES = MAX_NESTED_ARCHIVE
MAX_NESTED_ARCHIVE_WORK_BYTES = 2 * MAX_NESTED_ARCHIVE
# Runtime acquisition permits 100,000 top-level archive members. Recursive
# complete-byte scanning uses a stricter limit to bound ZipInfo allocation and
# sorting at every nesting depth before zipfile.ZipFile is constructed.
MAX_NESTED_ARCHIVE_MEMBERS = 10_000
MAX_DOCKER_SAVE_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_DOCKER_SAVE_OUTER_MEMBERS = 4_096
MAX_DOCKER_SAVE_METADATA_BYTES = 16 * 1024 * 1024
MAX_ORDERED_LAYERS = 256
MAX_OCI_DESCRIPTOR_GRAPH_VISITS = MAX_DOCKER_SAVE_OUTER_MEMBERS
MAX_LAYER_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ORDERED_LAYER_BYTES = 1024 * 1024 * 1024
MAX_LAYER_MEMBERS = 100_000
MAX_TOTAL_LAYER_MEMBERS = 250_000
MAX_LAYER_MEMBER_BYTES = 256 * 1024 * 1024
MAX_MATERIALIZED_LAYER_BYTES = 2 * 1024 * 1024 * 1024
ALLOWED_EMPTY_BASE_PATHS = frozenset(
    {
        "var/cache/apt/archives",
        "var/cache/apt/archives/lock",
        "var/cache/apt/archives/partial",
        "var/lib/apt/lists",
    }
)
OCI_INDEX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)
OCI_MANIFEST_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    }
)
OCI_CONFIG_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.config.v1+json",
        "application/vnd.docker.container.image.v1+json",
    }
)
OCI_LAYER_MEDIA_TYPES = {
    "application/vnd.oci.image.layer.v1.tar": "identity",
    "application/vnd.oci.image.layer.v1.tar+gzip": "gzip",
    "application/vnd.docker.image.rootfs.diff.tar.gzip": "gzip",
}
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
EXPECTED_SOURCE = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
EXPECTED_MUJOCO_COMMIT = "13827e9ee56f097f57acf69ae52b078f9839682d"
EXPECTED_SHADOW_COMMIT = "59d6bdf35bd9cf53185a20eb63413fdfe57fe77c"
EXPECTED_ASSET_LOCK = "e22eb62fc690a5e1d1ea931bab950392ca480caf3d51c7f16fd8cb4133d65568"
EXPECTED_BASE = {
    "image": "ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61",
    "config_digest": "sha256:b2b7ea366714195a1e1c5b2b578ece85c0b3920381a8654d038d9684f009613c",
    "layer_digest": "sha256:e51aee9c82ec5dd5ba2add49c45c6d85d460512757e2615b69bcdf9469c7cb58",
    # Docker/OCI config binds the uncompressed base layer diff ID; the registry
    # manifest independently binds the compressed layer digest above.
    "uncompressed_layer_digest": "sha256:6078cde548a521a729def2ee7875e9f65513c18f0d4bac4db817417617d7006a",
}
# These externally reviewed anchors close every candidate-added byte. The
# config binds history/runtime metadata and the ordered diff IDs; the diff IDs
# bind every raw layer. They come from the independently inspected neutral
# Linux/amd64 reference build and do not identify its private registry location.
EXPECTED_IMAGE_CONFIG_SHA256 = (
    "d80e779c65744d8c710cb981f67223340ac539312ef205f679682e0359b36a2e"
)
EXPECTED_ORDERED_LAYER_DIFF_IDS = (
    "sha256:6078cde548a521a729def2ee7875e9f65513c18f0d4bac4db817417617d7006a",
    "sha256:f2cea436a3738d53d631bf57f322c35c6a1be6b21460843a2ec047b056b8c4e1",
    "sha256:b1bcf1bd6dfe330dd4c41bcf0293e5bc26b812efc0659f9a24cc0eb9773e8545",
    "sha256:c56dc218ffc8d025a332e9ceae8962b632d8b0782230f6525e49906cbd3e1053",
    "sha256:ed036594e9aa162a66f47d90d6084eaa259d775dcbbe7467994a4f0bd8a977d6",
    "sha256:2ba04a6cc5b69ee205d249aae25e1de60a6e63f32d3d4eb6e52dd48ffff0309c",
    "sha256:001ab717894637e9d7ff7b6cbb8a56dca3907cc253e510b026f5a4ad0ba34bc8",
    "sha256:4beb089e6b62a23f68ad02adab33168dfc954f37f437613f849a6dc4d35eaecc",
    "sha256:1699ab172e299b1a3cc27c7658ebb2c444fbb916e1795eb9e6e5b1434393c358",
    "sha256:a990387443c5fc8fc8b7be383bf89dcc90cfa22a495bd96280f129c50d9647d8",
    "sha256:cebb133929934ca8e210947de1acd5eb932db7dc4f35e2fcba361beb2a9c31e4",
    "sha256:02fa7282faca1fd7a8ee7909a382b1c5ffe58b67a23651c277c897615b286abd",
    "sha256:c4388878d7778da2fefb4d74d07f413152de51eab4f50d15dcb49af4ad7d172b",
    "sha256:945d9722436c90417305671ca1b0cd5617698b6ff54e3c39fbac1b9ba5db0b6a",
    "sha256:b4b2f3cea99c226a0d25f42f0f5711e88e065ab57c8fc413c2465b3aabbd4661",
    "sha256:3c4745d1270db78d70751371a79aeace32b81f1ae2e0e168c03a22fef193ad80",
    "sha256:9181e0f7a816835fc416e754168371b457afd3ac9b96fbff8bcb40ce3a552fb9",
    "sha256:b4c53f6d2fba920eab0a5cdbe1f6bf2bf05792b38da59885e3fdb24aefbb3066",
    "sha256:90874ca0c277012cee42ac0e4c18b63b2bb5880f8811078f7da1e240318b5974",
    "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef",
    "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef",
)
# Neutral files are trusted only after their exact bytes are independently
# reviewed and pinned here. The reference-build config and ordered layer graph
# above are scanner inputs, not an accepted image digest or current-head proof;
# a local status edit cannot turn the scanner green.
EXPECTED_NEUTRAL_FILE_SHA256: dict[str, str | None] = {
    "source-lock.json": "3318043e3d3fec10b233b212b8e7bd97391f48f20b629dbdb3319981010b6ca9",
    "apt-runtime.lock.json": "6e1df9be2187010e9d4ee12dc2a4d95e4f0aa799ff321c70d86ec2d8772b855e",
    "corresponding-source.lock.json": "10ea8843b7b68c70b38a137a1683f46fcb6de8517d1863684f5419b21145967a",
    "requirements.lock": "30d48e4b2bfcf0c590b47ed569393104dd759476d720a608aa9f441cd9976e4a",
    "runtime-bootstrap.py": "ec1b843ff18606f64a3238ce86d4eef00ac4f0c14f93ba2801c17cbc003ae0e2",
    "capability_smoke.py": "f91683fa5955882e29e2ac8e6ba9f4d92f2a25eb71621275fa3c45b26828d6d6",
    "verify_image.py": "8529bab9b0cc15072a86f4c3cb84c864410ecbefe2302d27596cd3f1ad0108f6",  # gitleaks:allow; public file-content SHA-256
    "runtime-fetch-manifest.json": "08a628dd444d52dcaa7e60b41f1b416a42a4a7b85cde6dce0ab0887f1ea78f17",
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
    "cloudpickle": "3.1.2",
    "etils": "1.14.0",
    "farama-notifications": "0.0.6",
    "fsspec": "2026.7.0",
    "glfw": "2.10.2",
    "gymnasium": "1.3.0",
    "imageio": "2.37.4",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.3",
    "mujoco": "3.12.0",
    "numpy": "2.5.3",
    "packaging": "26.3",
    "pettingzoo": "1.27.0",
    "pillow": "12.3.0",
    "pyopengl": "3.1.10",
    "setuptools": "84.0.0",
    "typing-extensions": "4.16.0",
    "zipp": "4.1.0",
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


def _docker_save_config_digest(config_name: str) -> str:
    """Return the digest encoded by one supported Docker-save config path."""

    legacy = re.fullmatch(r"([0-9a-f]{64})\.json", config_name)
    oci = re.fullmatch(r"blobs/sha256/([0-9a-f]{64})", config_name)
    match = legacy or oci
    if match is None:
        raise ValueError("Docker save config filename is not a supported digest path")
    return match.group(1)


def _raw_member(
    archive: tarfile.TarFile, name: str, *, max_bytes: int
) -> bytes:
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"archive member is not a regular file: {name}")
    if member.size > max_bytes:
        raise ValueError(f"archive member exceeds scan bound: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"missing archive member: {name}")
    content = stream.read(max_bytes + 1)
    if len(content) != member.size or len(content) > max_bytes:
        raise ValueError(f"archive member exceeds scan bound: {name}")
    return content


def _repeated_docker_layer_member_is_identical(
    archive_bytes: bytes,
    first: tarfile.TarInfo,
    duplicate: tarfile.TarInfo,
    name: str,
) -> bool:
    """Accept Docker 29's repeated empty-layer alias only when byte-bound."""

    if re.fullmatch(r"blobs/sha256/[0-9a-f]{64}", name) is None:
        return False
    if not first.isfile() or not duplicate.isfile() or first.size != duplicate.size:
        return False
    first_raw = archive_bytes[first.offset_data : first.offset_data + first.size]
    duplicate_raw = archive_bytes[
        duplicate.offset_data : duplicate.offset_data + duplicate.size
    ]
    digest = name.rsplit("/", 1)[-1]
    return first_raw == duplicate_raw and hashlib.sha256(first_raw).hexdigest() == digest


def _archive_path_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _archive_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        *_archive_path_identity(metadata),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _trusted_archive_chain(path: Path) -> tuple[Path, tuple[tuple[int, ...], ...]]:
    """Bind every existing archive ancestor without following a symlink."""

    absolute = Path(os.path.abspath(path))
    identities: list[tuple[int, ...]] = []
    for ancestor in reversed(absolute.parent.parents):
        metadata = ancestor.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid not in {0, os.geteuid()}
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError("container archive has an unsafe parent directory")
        identities.append(_archive_path_identity(metadata))
    parent = absolute.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid not in {0, os.geteuid()}
        or stat.S_IMODE(parent.st_mode) & 0o022
    ):
        raise ValueError("container archive has an unsafe parent directory")
    identities.append(_archive_path_identity(parent))
    return absolute, tuple(identities)


def _open_archive_at_parent(
    path: Path, parent_identity: tuple[int, ...]
) -> tuple[int, int, os.stat_result]:
    try:
        parent = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        if _archive_path_identity(os.fstat(parent)) != parent_identity:
            raise ValueError("container archive parent changed before open")
        before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent,
        )
    except BaseException:
        if "parent" in locals():
            os.close(parent)
        raise
    return parent, descriptor, before


def _container_archive_bytes(path: Path) -> bytes:
    """Read one stable owner-controlled archive through bound descriptors."""

    try:
        absolute, before_chain = _trusted_archive_chain(path)
        parent, descriptor, before = _open_archive_at_parent(
            absolute, before_chain[-1]
        )
    except (OSError, ValueError) as error:
        raise ValueError("container archive is unsafe or unavailable") from error
    try:
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("container archive is not a regular file")
        if before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) & 0o022:
            raise ValueError("container archive is not trusted and owner-controlled")
        if before.st_size > MAX_DOCKER_SAVE_ARCHIVE_BYTES:
            raise ValueError("container archive exceeds scan bound")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            opened = os.fstat(stream.fileno())
            if _archive_file_identity(opened) != _archive_file_identity(before):
                raise ValueError("container archive path and descriptor differ")
            content = stream.read(MAX_DOCKER_SAVE_ARCHIVE_BYTES + 1)
            final_descriptor = os.fstat(stream.fileno())
        after = os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
        _absolute, after_chain = _trusted_archive_chain(absolute)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)
    if len(content) > MAX_DOCKER_SAVE_ARCHIVE_BYTES:
        raise ValueError("container archive exceeds scan bound")
    identity = _archive_file_identity(before)
    if (
        before_chain != after_chain
        or identity != _archive_file_identity(final_descriptor)
        or identity != _archive_file_identity(after)
        or len(content) != before.st_size
    ):
        raise ValueError("container archive changed while being read")
    return content


def _docker_save_bytes(path: Path) -> bytes:
    """Compatibility name for tests and callers of the Docker-save scanner."""

    return _container_archive_bytes(path)


def _scan_raw_blob_bytes(
    label: str, content: bytes, *, allowed_system_wheel_path: str | None = None
) -> None:
    """Bind opaque or structural bytes without treating their encoding as text."""

    digest = hashlib.sha256(content).hexdigest()
    if digest in KNOWN_FORBIDDEN_CONTENT_SHA256:
        raise ValueError(f"forbidden upstream/runtime byte: {label}")
    system_wheel_paths = {
        record["sha256"]: path for path, record in EXPECTED_SYSTEM_WHEEL_FILES.items()
    }
    reviewed_path = system_wheel_paths.get(digest)
    if reviewed_path is not None and allowed_system_wheel_path != reviewed_path:
        raise ValueError(f"system bootstrap wheel at unauthorized path: {label}")
    if allowed_system_wheel_path is not None:
        expected = EXPECTED_SYSTEM_WHEEL_FILES[allowed_system_wheel_path]["sha256"]
        if digest != expected:
            raise ValueError(
                f"reviewed system bootstrap wheel changed: {allowed_system_wheel_path}"
            )


def _scan_archive_representation_bytes(
    label: str,
    content: bytes,
    *,
    allowed_system_wheel_path: str | None = None,
    skip_text_policy: bool = False,
) -> None:
    """Apply policy to one archive's stored representation."""

    _scan_raw_blob_bytes(
        label,
        content,
        allowed_system_wheel_path=allowed_system_wheel_path,
    )
    if allowed_system_wheel_path is not None:
        return
    if not skip_text_policy and SECRET_TEXT.search(content):
        raise ValueError(f"forbidden secret signature: {label}")
    if not skip_text_policy and VENDOR_TEXT.search(content):
        raise ValueError(f"forbidden vendor payload signature: {label}")


# Exact signed-Ubuntu authentication implementations contain parser markers,
# password prompts and credential-variable assignments, not secret values.
# These dispositions bind both the reviewed path and complete file contents.
REVIEWED_CREDENTIAL_SOURCE_SHA256 = {
    'usr/bin/gpasswd': 'b9ef1bde0bde1b6839a0f55d43a606645723436932f072e3bf71e4c1a0ed4c06',  # gitleaks:allow; public file-content SHA-256
    'usr/bin/newgrp': '572ab15948df3969e929b902641c3a42e76dce7d71a22fbc58011ca89d8d965e',
    'usr/lib/x86_64-linux-gnu/libgnutls.so.30.37.1': 'ddbae2995750875c07bc218d12a062d73f8250678f54dedda7e9be5068781c98',
    'usr/lib/x86_64-linux-gnu/libpam.so.0.85.1': '9e1c85dcceeedd56568874afa04c52bbd051fb8bac52c3e1f5feeb1eed14238a',
    'usr/lib/x86_64-linux-gnu/security/pam_exec.so': '1ac881f3eb2383762917afb8197ecd0122dccf355156b70787c73d2947c5c6b1',
    'usr/lib/x86_64-linux-gnu/security/pam_extrausers.so': '612255329df214295d2e8eca01d79b830fe8c94ce7d87178e371c19c01afc0a2',
    'usr/lib/x86_64-linux-gnu/security/pam_stress.so': '4418b929b99608b048f138af777f2a2b9a6fe5a8d4af7242929308f9f1b9c68b',
    'usr/lib/x86_64-linux-gnu/security/pam_userdb.so': '38843ca2240952995c74582fabb65614e846427eb5e99fe362d8edd736bed3d9',
    'usr/share/pam-configs/unix': '66528ec667294fd2eaa1418ad4372f51e0c9a8fbe628275242276ca070639276',
    'usr/bin/cvtsudoers': '193f1e6bacae37805201b8f3bd06daad7b344d2e4c58e3dbd1b36fe55da272b1',
    'usr/bin/openssl': '30cc7c491903d6d8bca54406889c0334a167777397458214b5bd498c51b6fd97',
    'usr/bin/ssh': '31430165113e51a046f3b9fa2a60ec98e073cc6c0411a081b3b1d76ed6a4eb6a',
    'usr/bin/ssh-add': 'd8f8bc11af29c8f61b545191c981b1ea9d1c6c2bfdf61f4841ad5588af783067',
    'usr/bin/ssh-keygen': '5175ddce2146fc8a03ab8e1ef25a1b0382dd3cb209484f7ae11ac782171c0d04',  # gitleaks:allow; public file-content SHA-256
    'usr/lib/openssh/ssh-keysign': '7a7bf85b1dbe297da7508fc389d1ff27c54678d502e83214a7f9040d86ac4022',  # gitleaks:allow; public file-content SHA-256
    'usr/lib/python3/dist-packages/botocore/credentials.py': 'd4b5d5f206cee8142b12e8a19a30db201e43dac83d60259120168d626e00e671',  # gitleaks:allow; public file-content SHA-256
    'usr/lib/python3/dist-packages/botocore/data/acm/2015-12-08/service-2.json': '660c100f2f5f23e37e592e1681ac9f6a2f9a0a09b8e89b2aa566d1778f41ad22',
    'usr/lib/python3/dist-packages/botocore/httpsession.py': 'f0e0b9664e5af63fb762800ccbad17708e8f9cd97a0ca8c20faee9ab8a47157e',
    'usr/lib/python3/dist-packages/botocore/session.py': '290ec2132af23fd9da4ec9c76b224476139b500e004bfd7c5496138241050371',
    'usr/lib/python3/dist-packages/requests/adapters.py': 'bff1668d4e4a67bea4f98b6d4a1658079469ac8ce184bf18df3816f69e1e050f',
    'usr/lib/python3/dist-packages/requests/auth.py': '87e1cb955c7d8fcaca57985f480c9c3f60293928254f3efb474b73eea09b6c41',  # gitleaks:allow; public file-content SHA-256
    'usr/lib/python3/dist-packages/requests/sessions.py': 'f8bbd3ceb3ed7ad493ad1ddbbb1bb85e176032b2452c1d6ae43ecffbe2f65e1c',
    'usr/lib/python3/dist-packages/urllib3/connection.py': '470683be25f74bf9781a6a884d04001982ed46dc8e79edc4d4e6243ba26bc771',
    'usr/lib/python3/dist-packages/urllib3/connectionpool.py': '576d2852bfc99cadc1852e0a609b98cb4d243f5c6619a5a02a71f29dc69e9ddb',
    'usr/lib/python3/dist-packages/urllib3/contrib/_securetransport/low_level.py': 'd780e1a7f8e299ae49f36e286df157e6804e4588809d42ed509fd307c084b1c6',
    'usr/lib/python3/dist-packages/urllib3/contrib/pyopenssl.py': '97a5b3a5cef4d6cf200e7620f94ae26a45ae49f254d2f7cd323d3757dc9c0947',
    'usr/lib/python3/dist-packages/urllib3/contrib/socks.py': 'c5baacf8ff941c7e4be5afdd2afc4a7ad17257d94abb191c57d2bd3a2c9dfa02',
    'usr/lib/python3/dist-packages/urllib3/util/__pycache__/ssl_.cpython-312.pyc': 'cda36f1be5a92c7263f6213f312b9da7dd798b79ed5484813a6b2a7a97658184',
    'usr/lib/python3/dist-packages/urllib3/util/ssl_.py': '485473cdc8046fe82b32768b3e5506f2a8daca0e5582fbdff61997cb87b876d2',
    'usr/lib/python3.12/__pycache__/getpass.cpython-312.pyc': 'b536111082a164d2b58fb74024074c2e5d34c52f858e921b1898558b6294d644',
    'usr/lib/python3.12/__pycache__/nntplib.cpython-312.pyc': '8897e42279db5be80d880b4e834a4a3fad9e90c27d4a192fab619d0993ef0ff0',
    'usr/lib/python3.12/logging/handlers.py': '9bee4f9ee58c3a8858ef8a5f5ba597600667eddaa404e2653dfd1079b219e6bc',
    'usr/lib/python3.12/netrc.py': '229da6d0cbe6a10295be6c64ac3806420ea018124c09e4887410548fc2fc8b5d',
    'usr/lib/python3.12/nntplib.py': '6a76a94b951b273aa87335d7c9c4d7273e4c59485c784b057f681443b32d9004',
    'usr/lib/python3.12/urllib/parse.py': 'a67b5694763137dbac085adbf0c22821d435ec8ca22d3bd3786d2c9f4ee748d5',
    'usr/lib/python3.12/urllib/request.py': '96ae4b051b87eb7694da24f5fa75bf3931486b49b755699934bf15ab0ca5829d',
    'usr/libexec/sudo/sudoers.so': '4955ef472c43e5749b1076ec902164d43f23bc095266d87358b424aaa7a3a4a0',
    'usr/sbin/sshd': 'f30333b455e6fe10536d544992ea3dfa87cc1d83b8af7945a815c86f8b4f6f6d',
    'usr/sbin/visudo': '0fc25209fe0348b2a52cd117f540c05ce4235f8a18bac011de6381c89f3f1c6e',
}

def _scan_decoded_member_bytes(
    label: str,
    content: bytes,
    *,
    allowed_system_wheel_path: str | None = None,
    skip_text_policy: bool = False,
) -> None:
    """Apply exact-byte and textual policy to one actionable decoded member."""

    _scan_raw_blob_bytes(
        label,
        content,
        allowed_system_wheel_path=allowed_system_wheel_path,
    )
    if skip_text_policy:
        return
    path = label.removeprefix("decoded member: ")
    reviewed_source = REVIEWED_CREDENTIAL_SOURCE_SHA256.get(path) == hashlib.sha256(content).hexdigest()
    if SECRET_TEXT.search(content) and not reviewed_source:
        raise ValueError(f"forbidden secret signature: {label}")
    if VENDOR_TEXT.search(content):
        raise ValueError(f"forbidden vendor payload signature: {label}")


class _NestedArchiveBudget:
    """Track cumulative resource use across one retained archive graph."""

    def __init__(self) -> None:
        self.expanded_bytes = 0
        self.member_count = 0
        self.work_bytes = 0

    def reserve_members(self, path: str, count: int) -> None:
        self.member_count += count
        if self.member_count > MAX_NESTED_ARCHIVE_MEMBERS:
            raise ValueError(f"nested archive member budget exceeded: {path}")

    def account_archive(self, path: str, size: int) -> None:
        self.work_bytes += size
        if self.work_bytes > MAX_NESTED_ARCHIVE_WORK_BYTES:
            raise ValueError(f"nested archive work budget exceeded: {path}")

    def account_expanded(self, path: str, size: int) -> None:
        self.expanded_bytes += size
        self.work_bytes += size
        if self.expanded_bytes > MAX_NESTED_ARCHIVE_EXPANDED_BYTES:
            raise ValueError(f"nested archive expanded-byte budget exceeded: {path}")
        if self.work_bytes > MAX_NESTED_ARCHIVE_WORK_BYTES:
            raise ValueError(f"nested archive work budget exceeded: {path}")


class _OciDescriptorGraphBudget:
    """Bound every descriptor occurrence visited during one OCI scan."""

    def __init__(self) -> None:
        self.visits = 0
        self.validated_blobs: dict[str, bytes] = {}
        self.text_policy_scanned: dict[str, bool] = {}

    def reserve(self, label: str) -> None:
        self.visits += 1
        if self.visits > MAX_OCI_DESCRIPTOR_GRAPH_VISITS:
            raise ValueError(f"OCI descriptor graph work budget exceeded: {label}")

    def remember(
        self, digest: str, content: bytes, *, text_policy_scanned: bool
    ) -> None:
        self.validated_blobs[digest] = content
        self.text_policy_scanned[digest] = text_policy_scanned


def _looks_like_tar(content: bytes) -> bool:
    """Recognize a valid first tar header, including pre-ustar archives."""

    if len(content) < 512 or not any(content[:512]):
        return False
    # Avoid treating arbitrary binary members (notably Python bytecode) as a
    # tar stream merely because their random checksum bytes happen to match.
    name = content[:100].rstrip(b"\0")
    if not name or any(byte < 0x20 or byte > 0x7E for byte in name):
        return False
    checksum_field = content[148:156].rstrip(b"\0 ").lstrip(b" ")
    try:
        expected = int(checksum_field or b"0", 8)
    except ValueError:
        return False
    actual = sum(content[:148]) + (8 * ord(" ")) + sum(content[156:512])
    return expected == actual and bool(content[:100].rstrip(b"\0"))


def _validated_tar_members(
    path: str,
    content: bytes,
    *,
    max_members: int,
    budget: _NestedArchiveBudget | None = None,
) -> list[tarfile.TarInfo]:
    """Parse one tar stream and account for every structural byte."""

    if len(content) < 1024 or len(content) % 512:
        raise ValueError(f"unaccounted tar bytes: {path}")
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            members = []
            for member in archive:
                if len(members) >= max_members:
                    raise ValueError(f"tar archive member count exceeds limit: {path}")
                members.append(member)
    except tarfile.TarError as error:
        raise ValueError(f"unreadable nested tar archive: {path}") from error
    cursor = 0
    for index, member in enumerate(members):
        if member.offset != cursor:
            raise ValueError(f"unaccounted tar bytes: {path}")
        if member.offset_data < member.offset + 512:
            raise ValueError(f"invalid tar metadata boundary: {path}")
        _scan_decoded_member_bytes(
            f"tar metadata: {path}:member-{index}",
            content[member.offset : member.offset_data],
        )
        if (member.issym() or member.islnk()) and member.size != 0:
            raise ValueError(f"nonzero tar link body: {path}:member-{index}")
        data_end = member.offset_data + member.size
        cursor = ((data_end + 511) // 512) * 512
        if any(content[data_end:cursor]):
            raise ValueError(f"nonzero tar member padding: {path}:member-{index}")
    tail = content[cursor:]
    if len(tail) < 1024 or any(tail):
        raise ValueError(f"unaccounted tar bytes: {path}")
    if budget is not None:
        budget.reserve_members(path, len(members))
    return members


def _validate_tar_directory(
    archive: tarfile.TarFile, member: tarfile.TarInfo, label: str
) -> None:
    """Require one tar directory to carry no body or conflicting metadata."""

    if (
        not member.isdir()
        or member.size != 0
        or member.linkname
        or (member.devmajor or 0) != 0
        or (member.devminor or 0) != 0
    ):
        raise ValueError(f"invalid tar directory: {label}")
    payload = archive.extractfile(member)
    if payload is not None and payload.read(1):
        raise ValueError(f"invalid tar directory: {label}")


def _validate_exact_outer_layout(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    regular_files: set[str],
    *,
    label: str,
) -> None:
    """Require only graph files and their zero-body parent directories."""

    directories = {
        str(parent)
        for name in regular_files
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    if set(members) != regular_files | directories:
        raise ValueError(f"{label} has missing or unreferenced members")
    for name in regular_files:
        if not members[name].isfile():
            raise ValueError(f"{label} file is not regular: {name}")
    for name in directories:
        _validate_tar_directory(archive, members[name], f"{label}:{name}")


def _validate_zip_directory(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, label: str
) -> None:
    """Require one ZIP directory to be stored with no body or file type."""

    mode = member.external_attr >> 16
    if (
        not member.is_dir()
        or member.file_size != 0
        or member.compress_size != 0
        or member.CRC != 0
        or member.compress_type != zipfile.ZIP_STORED
        or stat.S_IFMT(mode) not in (0, stat.S_IFDIR)
        or archive.read(member) != b""
    ):
        raise ValueError(f"invalid nested ZIP directory: {label}")


def _validate_zip_data_descriptor(
    path: str, descriptor: bytes, info: zipfile.ZipInfo
) -> None:
    """Bind a classic ZIP data descriptor to its central-directory record."""

    if len(descriptor) == 12:
        fields = descriptor
    elif len(descriptor) == 16 and descriptor.startswith(b"PK\x07\x08"):
        fields = descriptor[4:]
    else:
        raise ValueError(f"unsupported zip data descriptor: {path}")
    crc, compressed_size, file_size = struct.unpack("<3L", fields)
    if (crc, compressed_size, file_size) != (
        info.CRC,
        info.compress_size,
        info.file_size,
    ):
        raise ValueError(f"zip data descriptor does not match central directory: {path}")


def _zip_central_filename_bytes(path: str, info: zipfile.ZipInfo) -> bytes:
    """Recover the exact supported central-directory filename encoding."""

    encoding = "utf-8" if info.flag_bits & 0x800 else "cp437"
    try:
        return info.orig_filename.encode(encoding)
    except UnicodeEncodeError as error:
        raise ValueError(f"unsupported zip filename encoding: {path}") from error


def _validate_zip_compressed_stream(
    path: str, info: zipfile.ZipInfo, content: bytes
) -> None:
    """Require one ZIP member stream to consume all declared stored bytes."""

    if info.file_size > MAX_NESTED_ARCHIVE:
        raise ValueError(f"nested archive member exceeds scan bound: {path}")
    if info.compress_type == zipfile.ZIP_STORED:
        expanded = content
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        try:
            stream = zlib.decompressobj(-zlib.MAX_WBITS)
            expanded = stream.decompress(content, MAX_NESTED_ARCHIVE + 1)
        except zlib.error as error:
            raise ValueError(f"unreadable zip compressed stream: {path}") from error
        if len(expanded) > MAX_NESTED_ARCHIVE:
            raise ValueError(f"nested archive member exceeds scan bound: {path}")
        if not stream.eof or stream.unconsumed_tail or stream.unused_data:
            raise ValueError(f"ambiguous zip compressed stream: {path}")
    else:
        raise ValueError(f"unsupported zip compression: {path}")
    if len(expanded) != info.file_size or zlib.crc32(expanded) != info.CRC:
        raise ValueError(f"zip compressed stream does not match central directory: {path}")


def _reserve_zip_expansion(
    path: str, infos: list[zipfile.ZipInfo], budget: _NestedArchiveBudget
) -> None:
    """Reserve every declared ZIP expansion before decoding any member."""

    for info in infos:
        if info.file_size > MAX_NESTED_ARCHIVE:
            raise ValueError(f"nested archive member exceeds scan bound: {path}")
        budget.account_expanded(path, info.file_size)


def _validated_zip_infos(
    path: str,
    content: bytes,
    *,
    budget: _NestedArchiveBudget | None = None,
) -> list[zipfile.ZipInfo]:
    """Parse one prefix/suffix-free ZIP and account for structural metadata."""

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
    if comment_size:
        raise ValueError(f"unsupported zip archive comment: {path}")
    if max(disk_entries, total_entries) > MAX_NESTED_ARCHIVE_MEMBERS:
        raise ValueError(f"zip archive member count exceeds limit: {path}")
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
    if any(info.comment for info in infos):
        raise ValueError(f"unsupported zip member comment: {path}")
    _scan_decoded_member_bytes(
        f"zip central directory metadata: {path}",
        content[central_offset:eocd_offset],
    )
    if budget is not None:
        budget.reserve_members(path, total_entries)
    ordered = sorted(infos, key=lambda item: item.header_offset)
    if budget is not None:
        _reserve_zip_expansion(path, ordered, budget)
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
        filename_start = cursor + 30
        filename_end = filename_start + filename_size
        data_start = filename_end + extra_size
        if data_start > central_offset:
            raise ValueError(f"unaccounted zip bytes: {path}")
        _scan_decoded_member_bytes(
            f"zip local header metadata: {path}:member-{index}",
            content[cursor:data_start],
        )
        if content[filename_start:filename_end] != _zip_central_filename_bytes(
            path, info
        ):
            raise ValueError(
                f"zip local filename does not match central directory: {path}"
            )
        # This scanner supports classic non-ZIP64 archives without extra-field
        # semantics. Exact equality would still leave an unparsed field able to
        # carry contradictory metadata, so reject unparsed fields entirely.
        if extra_size or info.extra:
            raise ValueError(f"unsupported zip extra field: {path}")
        data_end = data_start + info.compress_size
        next_offset = (
            ordered[index + 1].header_offset
            if index + 1 < len(ordered)
            else central_offset
        )
        if data_end > next_offset:
            raise ValueError(f"unaccounted zip bytes: {path}")
        _validate_zip_compressed_stream(path, info, content[data_start:data_end])
        descriptor = content[data_end:next_offset]
        if local_flags & 0x08:
            if (local_crc, local_compressed_size, local_size) != (0, 0, 0):
                raise ValueError(
                    f"zip local descriptor metadata is not zero: {path}"
                )
            _validate_zip_data_descriptor(path, descriptor, info)
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


def _decompress(
    path: str,
    content: bytes,
    kind: str,
    *,
    max_bytes: int = MAX_NESTED_ARCHIVE,
) -> bytes:
    """Expand one recognized stream with a strict output-size bound."""

    try:
        if kind == "gzip":
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            expanded = decompressor.decompress(content, max_bytes + 1)
            if len(expanded) > max_bytes:
                raise ValueError(f"expanded stream exceeds scan bound: {path}")
            if (
                not decompressor.eof
                or decompressor.unconsumed_tail
                or decompressor.unused_data
            ):
                raise ValueError(f"ambiguous compressed stream: {path}")
        elif kind == "bzip2":
            decompressor = bz2.BZ2Decompressor()
            expanded = decompressor.decompress(
                content, max_length=max_bytes + 1
            )
            if len(expanded) > max_bytes:
                raise ValueError(f"expanded stream exceeds scan bound: {path}")
            if not decompressor.eof or decompressor.unused_data:
                raise ValueError(f"ambiguous compressed stream: {path}")
        else:
            decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
            expanded = decompressor.decompress(
                content, max_length=max_bytes + 1
            )
            if len(expanded) > max_bytes:
                raise ValueError(f"expanded stream exceeds scan bound: {path}")
            if not decompressor.eof or decompressor.unused_data:
                raise ValueError(f"ambiguous compressed stream: {path}")
    except (EOFError, OSError, lzma.LZMAError, zlib.error) as error:
        raise ValueError(f"unreadable compressed stream: {path}") from error
    return expanded


def _nested_archive_members(
    path: str,
    content: bytes,
    *,
    depth: int = 0,
    allowed_system_wheel_path: str | None = None,
    budget: _NestedArchiveBudget | None = None,
    skip_text_policy: bool = False,
) -> int:
    """Inspect retained archives by validated bytes, never only by filename."""

    budget = budget or _NestedArchiveBudget()
    starting_member_count = budget.member_count
    if depth > 8:
        raise ValueError(f"nested archive depth exceeds scan bound: {path}")
    lowered = path.lower()
    declared_zip = lowered.endswith((".whl", ".zip"))
    declared_tar = lowered.endswith((".tar", *COMPRESSED_TAR_SUFFIXES))
    declared_plain_tar = lowered.endswith(".tar")
    declared_compression = _declared_compression(path)
    is_tar = _looks_like_tar(content)
    is_zip = (
        not is_tar
        and content.startswith(ZIP_SIGNATURES)
        and zipfile.is_zipfile(io.BytesIO(content))
    )
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
    if archive_like:
        budget.account_archive(path, len(content))
        _scan_archive_representation_bytes(
            f"raw archive member: {path}",
            content,
            allowed_system_wheel_path=allowed_system_wheel_path,
            skip_text_policy=skip_text_policy,
        )
    else:
        _scan_decoded_member_bytes(
            f"decoded member: {path}",
            content,
            allowed_system_wheel_path=allowed_system_wheel_path,
            skip_text_policy=skip_text_policy,
        )
    if archive_like and len(content) > MAX_NESTED_ARCHIVE:
        raise ValueError(f"nested archive exceeds scan bound: {path}")
    if compression_kind is not None:
        expanded = _decompress(path, content, compression_kind)
        budget.account_expanded(path, len(expanded))
        if lowered.endswith(COMPRESSED_TAR_SUFFIXES) and not _looks_like_tar(expanded):
            raise ValueError(f"compressed tar payload is not a tar archive: {path}")
        return _nested_archive_members(
            f"{path}:expanded-{compression_kind}",
            expanded,
            depth=depth + 1,
            budget=budget,
            skip_text_policy=skip_text_policy,
        )
    if is_zip:
        infos = _validated_zip_infos(path, content, budget=budget)
        normalized_names = [_safe(member.filename) for member in infos]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError(f"duplicate normalized nested ZIP member: {path}")
        reviewed_system_wheel = allowed_system_wheel_path is not None
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                for member in infos:
                    safe = _safe(member.filename)
                    allowed_empty_base_dir = (
                        member.is_dir()
                        and member.file_size == 0
                        and safe in ALLOWED_EMPTY_BASE_PATHS
                    )
                    if (
                        FORBIDDEN_PATH.search(safe)
                        and not reviewed_system_wheel
                        and not allowed_empty_base_dir
                    ) or UPSTREAM_TREE_PATH.search(safe):
                        raise ValueError(
                            f"forbidden nested archive member: {path}:{safe}"
                        )
                    if member.is_dir():
                        _validate_zip_directory(archive, member, f"{path}:{safe}")
                        continue
                    if member.file_size > MAX_NESTED_ARCHIVE:
                        raise ValueError(
                            f"nested archive member exceeds scan bound: {path}:{safe}"
                        )
                    nested_content = archive.read(member)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        target = nested_content.decode(
                            "utf-8", errors="surrogateescape"
                        )
                        resolved = _resolved_link_target(safe, target, relative=True)
                        if not reviewed_system_wheel and (
                            FORBIDDEN_PATH.search(target.lstrip("/"))
                            or FORBIDDEN_PATH.search(resolved)
                            or UPSTREAM_TREE_PATH.search(target.lstrip("/"))
                            or UPSTREAM_TREE_PATH.search(resolved)
                        ):
                            raise ValueError(
                                f"forbidden nested archive link: {path}:{safe}"
                            )
                        _scan_decoded_member_bytes(
                            f"nested archive link: {path}:{safe}",
                            nested_content,
                            skip_text_policy=skip_text_policy,
                        )
                        continue
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                        raise ValueError(
                            f"unsupported nested archive member type: {path}:{safe}"
                        )
                    if reviewed_system_wheel:
                        if (
                            hashlib.sha256(nested_content).hexdigest()
                            in KNOWN_FORBIDDEN_CONTENT_SHA256
                        ):
                            raise ValueError(
                                f"forbidden upstream/runtime byte: {path}:{safe}"
                            )
                        # The exact outer wheel digest binds every member.  Do
                        # not apply generic credential-source regexes to pip's
                        # own authentication implementation.
                        continue
                    _nested_archive_members(
                        f"{path}:{safe}",
                        nested_content,
                        depth=depth + 1,
                        budget=budget,
                        skip_text_policy=skip_text_policy,
                    )
                return budget.member_count - starting_member_count
        except zipfile.BadZipFile as error:
            raise ValueError(f"unreadable nested zip archive: {path}") from error
    if content.startswith(ZIP_SIGNATURES) or declared_zip:
        raise ValueError(f"unreadable nested zip archive: {path}")
    if is_tar:
        members = _validated_tar_members(
            path, content, max_members=MAX_NESTED_ARCHIVE_MEMBERS
        )
        budget.reserve_members(path, len(members))
        normalized_names = [_safe(member.name) for member in members]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError(f"duplicate normalized nested tar member: {path}")
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as archive:
                for member in archive:
                    safe = _safe(member.name)
                    allowed_empty_base_dir = (
                        (member.isdir() or (member.isfile() and safe == "var/cache/apt/archives/lock"))
                        and member.size == 0
                        and safe in ALLOWED_EMPTY_BASE_PATHS
                    )
                    if (
                        FORBIDDEN_PATH.search(safe) and not allowed_empty_base_dir
                    ) or UPSTREAM_TREE_PATH.search(safe):
                        raise ValueError(
                            f"forbidden nested archive member: {path}:{safe}"
                        )
                    budget.account_expanded(f"{path}:{safe}", member.size)
                    if member.isfile():
                        if member.size > MAX_NESTED_ARCHIVE:
                            raise ValueError(
                                f"nested archive member exceeds scan bound: {path}:{safe}"
                            )
                        stream = archive.extractfile(member)
                        nested_content = stream.read() if stream is not None else b""
                        if stream is None:
                            raise ValueError(
                                f"forbidden nested archive bytes: {path}:{safe}"
                            )
                        _nested_archive_members(
                            f"{path}:{safe}",
                            nested_content,
                            depth=depth + 1,
                            budget=budget,
                            skip_text_policy=skip_text_policy,
                        )
                    elif member.issym() or member.islnk():
                        target_text = member.linkname
                        target = target_text.encode("utf-8", errors="surrogateescape")
                        _scan_decoded_member_bytes(
                            f"nested archive link: {path}:{safe}",
                            target,
                            skip_text_policy=skip_text_policy,
                        )
                        resolved = _resolved_link_target(
                            safe, target_text, relative=member.issym()
                        )
                        if FORBIDDEN_PATH.search(
                            target_text.lstrip("/")
                        ) or FORBIDDEN_PATH.search(resolved) or UPSTREAM_TREE_PATH.search(
                            target_text.lstrip("/")
                        ) or UPSTREAM_TREE_PATH.search(resolved):
                            raise ValueError(
                                f"forbidden nested archive link: {path}:{safe}"
                            )
                    elif member.isdir():
                        _validate_tar_directory(archive, member, f"{path}:{safe}")
                    else:
                        raise ValueError(
                            f"unsupported nested archive member type: {path}:{safe}"
                        )
                return budget.member_count - starting_member_count
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
    return _layers_from_bytes(
        [
            (
                layer_name,
                _raw_member(
                    archive, layer_name, max_bytes=MAX_LAYER_ARCHIVE_BYTES
                ),
            )
            for layer_name in names
        ]
    )


def _layers_from_bytes(
    layers: list[tuple[str, bytes]],
    *,
    nested_budget: _NestedArchiveBudget | None = None,
    pre_scanned_text_digests: set[str] | None = None,
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
    ordered_layer_bytes = 0
    materialized_layer_bytes = 0
    if nested_budget is None:
        nested_budget = _NestedArchiveBudget()
    if pre_scanned_text_digests is None:
        pre_scanned_text_digests = set()
    for layer_name, raw in layers:
        if len(raw) > MAX_LAYER_ARCHIVE_BYTES:
            raise ValueError(f"archive member exceeds scan bound: {layer_name}")
        ordered_layer_bytes += len(raw)
        if ordered_layer_bytes > MAX_ORDERED_LAYER_BYTES:
            raise ValueError("ordered layer bytes exceed scan bound")
        _scan_raw_blob_bytes(f"raw layer bytes: {layer_name}", raw)
        layer_members = _validated_tar_members(
            f"raw layer: {layer_name}", raw, max_members=MAX_LAYER_MEMBERS
        )
        if total + len(layer_members) > MAX_TOTAL_LAYER_MEMBERS:
            raise ValueError("total member count exceeds scan bound")
        total += len(layer_members)
        layer_digest = hashlib.sha256(raw).hexdigest()
        diff_ids.append("sha256:" + layer_digest)
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
                allowed_system_wheel = path in EXPECTED_SYSTEM_WHEEL_FILES
                if (
                    FORBIDDEN_PATH.search(path)
                    and not allowed_system_wheel
                    and not (
                        (item.isdir() or (item.isfile() and path == "var/cache/apt/archives/lock"))
                        and item.size == 0
                        and path in ALLOWED_EMPTY_BASE_PATHS
                    )
                    or UPSTREAM_TREE_PATH.search(path)
                ):
                    raise ValueError(f"forbidden image path: {path}")
                current_order.append(path)
                if item.isfile():
                    if item.size > MAX_LAYER_MEMBER_BYTES:
                        raise ValueError(f"layer member exceeds scan bound: {path}")
                    materialized_layer_bytes += item.size
                    if materialized_layer_bytes > MAX_MATERIALIZED_LAYER_BYTES:
                        raise ValueError("materialized layer bytes exceed scan bound")
                    payload = layer.extractfile(item)
                    if payload is None:
                        raise ValueError(f"unreadable layer file: {path}")
                    content = payload.read(MAX_LAYER_MEMBER_BYTES + 1)
                    if len(content) != item.size:
                        raise ValueError(f"layer member exceeds scan bound: {path}")
                    if layer_digest not in pre_scanned_text_digests:
                        nested += _nested_archive_members(
                            path,
                            content,
                            allowed_system_wheel_path=(
                                path if allowed_system_wheel else None
                            ),
                            budget=nested_budget,
                        )
                    _remove_path(current_rootfs, current_entries, path)
                    current_rootfs[path] = content
                    current_entries[path] = {
                        **_entry_metadata(item, "regular"),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                elif item.isdir():
                    _validate_tar_directory(layer, item, f"layer:{path}")
                    current_rootfs.pop(path, None)
                    current_entries[path] = _entry_metadata(item, "directory")
                elif item.issym() or item.islnk():
                    target = item.linkname
                    target_bytes = target.encode("utf-8", errors="surrogateescape")
                    _scan_decoded_member_bytes(f"link target: {path}", target_bytes)
                    resolved = _resolved_link_target(
                        path, target, relative=item.issym()
                    )
                    if FORBIDDEN_PATH.search(
                        target.lstrip("/")
                    ) or FORBIDDEN_PATH.search(resolved) or UPSTREAM_TREE_PATH.search(
                        target.lstrip("/")
                    ) or UPSTREAM_TREE_PATH.search(resolved):
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
        for parent in PurePosixPath(path).parents:
            parent_name = str(parent)
            if parent_name == ".":
                break
            ancestor = entries.get(parent_name)
            if ancestor is not None and ancestor["kind"] in {"symlink", "hardlink"}:
                raise ValueError(
                    f"retained descendant has a link ancestor: {path}"
                )
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


# Independent member closure: reviewed reference plus the explicit deterministic
# build cleanup and reviewed NPA files. Every member is bound, including files
# hidden by later layers. Only timestamps and the separately verified revision vary.
EXPECTED_CANONICAL_CONFIG_SHA256 = 'e2f05d5d405492d028e3e2a9682aa38821b47a7957daa7d19bab47d892c83361'
EXPECTED_CANONICAL_LAYER_SHA256 = (
    '9c9ba303f1155de79657da48833d0b1c80ab4cde0449b9bb8013096336b992f1',
    '73e6fc4e984caa7bb8593375c1c3b06e138906656ffb35f06192691861188243',
    '49a4cfc771493bea5109ca9f9c5f682864aa10e0cf6c445f005b21f0636c820a',
    'b12df185fe3b2a74ba243d30bde009f7ce7d9f4efedb28fafaaabd0cf6b35c57',
    '728fa6a381173f1082dc63ead9aa8cbe3dda12a980d4a43f40dcd9016d69e8d8',
    'f2454738af88ecc1e948b21d29cf737c33b8cc7fd362fd16d235672b81abf3bc',
    '94db80991c8735f6649b289cfc25d325797d5c564a0df43d8ee74e09acb6ba44',
    '80416bd425ffda334527def953b935f52badbc8e2d614851a5f2c279ce4bd449',
    'c1dace677b83aa54d1b5a5d650b766d04dba80dbf5db7678069424107e908912',
    '4ca8f3361b402401501aac431c4741c8b42c3a96b508f5ed5794a1a41d44840c',
    '73daaeeafb9d494ffad6c0b142dbc03785abb760a3fdd7f67448422ec8304d20',
    '1f9be8fd8a7021eb3aa1f1bfc67f93918adba351dde96f9cbf0f610ac9807325',
    '89eb8f2ff39a515d4e8886126d4bad1cf39b76ab0eae9a696fc87445a543204d',
    '3b6f97d60680fb30327f10fd2ae39c3aa3005e2913fca3ade3f2a549253493cc',
    '9a30e527d765b38364c3d5373b5ab70a2d86bc4ca2fd13161c8549d68056a6d9',
    '6a1402467a860af7d89392ef3a268f4694601f009816e4ad104484b0568858eb',
    '195b0ef104b25550a277a3ed8cf01468d7c57896e5b635343b93d31063fcce28',
    '44d00cd636ee5afae681bfc3aa9831345b248196c963a5fbd29454e2afd4e9f4',
    'be1bbe789b60202a6e8666435ffd8ac6ef607d595e9a51999e201e306aca1731',
    '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
    '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
)

def _validate_tar_timestamps(item: tarfile.TarInfo) -> None:
    timestamps = {"mtime": item.mtime}
    timestamps.update({
        key: value for key, value in item.pax_headers.items()
        if key in {"mtime", "atime", "ctime"}
    })
    for value in timestamps.values():
        if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", str(value)) is None:
            raise ValueError("invalid canonical TAR timestamp")
        if not math.isfinite(float(value)):
            raise ValueError("nonfinite canonical TAR timestamp")


def _discard_oci_timestamp(record: dict[str, object]) -> None:
    if "created" not in record:
        return
    value = record.pop("created")
    pattern = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?(?:Z|[+-][0-9]{2}:[0-9]{2})"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError("invalid canonical OCI timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid canonical OCI timestamp") from error


def _canonical_layer_sha256(raw: bytes) -> str:
    """Bind all ordered members, omitting only archive timestamps."""
    records = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as layer:
        for item in layer:
            _validate_tar_timestamps(item)
            record = item.get_info()
            record["type"] = item.type.hex()
            record.pop("mtime")
            record.pop("chksum")  # Already validated; it includes the timestamp.
            record["pax_headers"] = {
                key: value
                for key, value in item.pax_headers.items()
                if key not in {"mtime", "atime", "ctime"}
            }
            if item.isfile():
                payload = layer.extractfile(item)
                if payload is None:
                    raise ValueError("missing canonical layer member")
                record["sha256"] = hashlib.sha256(payload.read()).hexdigest()
            records.append(record)
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_config_sha256(config: dict[str, object]) -> str:
    """Preserve config semantics while separating time and graph identity."""
    canonical = json.loads(json.dumps(config))
    revision = canonical["config"]["Labels"]["org.opencontainers.image.revision"]
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("canonical image revision must be a full Git SHA")
    canonical["config"]["Labels"]["org.opencontainers.image.revision"] = "<revision>"
    _discard_oci_timestamp(canonical)
    canonical["rootfs"]["diff_ids"] = "<independently-bound-ordered-layers>"
    for entry in canonical["history"]:
        _discard_oci_timestamp(entry)
        entry["created_by"] = re.sub(
            r"org\.opencontainers\.image\.revision=(?:[0-9a-f]{40})?(?= |$)",
            "org.opencontainers.image.revision=<revision>",
            entry["created_by"],
        )
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reviewed_image_closure(
    config: dict[str, object], layers: list[tuple[str, bytes]]
) -> None:
    """Require the independent member closure in addition to current graph IDs."""
    if _canonical_config_sha256(config) != EXPECTED_CANONICAL_CONFIG_SHA256:
        raise ValueError("reviewed neutral image config semantics changed")
    observed = tuple(_canonical_layer_sha256(raw) for _, raw in layers)
    if observed != EXPECTED_CANONICAL_LAYER_SHA256:
        raise ValueError("reviewed neutral ordered layer member closure changed")


def _reviewed_image_graph(
    config_digest: str,
    layer_diff_ids: list[str],
    layer_descriptors: list[dict[str, object]] | None = None,
    expected_config_digest: str | None = None,
    expected_layer_diff_ids: list[str] | None = None,
) -> None:
    expected_config = (
        EXPECTED_IMAGE_CONFIG_SHA256
        if expected_config_digest is None
        else expected_config_digest
    )
    if not isinstance(expected_config, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_config
    ):
        raise ValueError("reviewed neutral image config digest is not configured")
    expected_layers = tuple(
        EXPECTED_ORDERED_LAYER_DIFF_IDS
        if expected_layer_diff_ids is None
        else expected_layer_diff_ids
    )
    if (
        not isinstance(expected_layers, tuple)
        or not expected_layers
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            for digest in expected_layers
        )
    ):
        raise ValueError("reviewed neutral ordered layer graph is not configured")
    if config_digest != expected_config:
        raise ValueError("reviewed neutral image config bytes changed")
    if layer_diff_ids != list(expected_layers):
        raise ValueError("reviewed neutral ordered layer bytes changed")
    if layer_descriptors is not None:
        if len(layer_descriptors) != len(layer_diff_ids):
            raise ValueError("OCI layer descriptors do not match ordered layers")
        if layer_descriptors[0].get("digest") != EXPECTED_BASE["layer_digest"]:
            raise ValueError("OCI image does not begin with the reviewed Ubuntu blob")


def _neutral_candidate(
    rootfs: dict[str, bytes],
    layer_diff_ids: list[str],
) -> None:
    """Bind a future neutral image without accepting runtime payload bytes."""

    for name, expected in EXPECTED_NEUTRAL_FILE_SHA256.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"reviewed neutral file digest is not configured: {name}")
        path = f"opt/npa/gymnasium-robotics/{name}"
        if hashlib.sha256(rootfs[path]).hexdigest() != expected:
            raise ValueError(f"reviewed neutral image file changed: {name}")
    for path, record in EXPECTED_SYSTEM_WHEEL_FILES.items():
        content = rootfs.get(path)
        if content is None or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"reviewed system bootstrap wheel changed: {path}")
    if (
        hashlib.sha256(rootfs["opt/npa/gymnasium-robotics/asset-lock.json"]).hexdigest()
        != EXPECTED_ASSET_LOCK
    ):
        raise ValueError("approved asset provenance metadata changed")
    if _missing(EXPECTED_BASE) or (
        not layer_diff_ids
        or layer_diff_ids[0] != EXPECTED_BASE["uncompressed_layer_digest"]
    ):
        raise ValueError("saved image does not begin with the reviewed Ubuntu base")

    source = json.loads(rootfs["opt/npa/gymnasium-robotics/source-lock.json"])
    if (
        source.get("schema") != "npa.gymnasium-robotics.runtime-fetch-lock.v2"
        or source.get("status") != "complete"
        or source.get("source_commit") != EXPECTED_SOURCE
        or source.get("mujoco_version") != "3.12.0"
    ):
        raise ValueError("complete runtime-fetch lock identity changed")
    components = source.get("components", {})
    if set(components) != set(EXPECTED_SOURCE_FIELDS):
        raise ValueError("runtime source component inventory changed")
    for name, expected_fields in EXPECTED_SOURCE_FIELDS.items():
        if any(
            components[name].get(key) != value for key, value in expected_fields.items()
        ):
            raise ValueError(f"runtime source identity changed: {name}")
    if source.get("delivery") != {
        "source": "operator-owned-runtime-cache",
        "baked_runtime": "neutral-bootstrap-only",
        "weights": "none",
        "data_assets": "runtime-cache-only",
        "runtime_cache": "operator-owned-and-external",
        "outputs": "operator-owned-run-artifacts",
    }:
        raise ValueError("six-boundary runtime delivery classification changed")
    artifacts = source.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != 20
        or source.get("expected_python_distribution_count") != 19
        or source.get("resolved_python_artifact_count") != 19
    ):
        raise ValueError("runtime artifact closure is incomplete")
    if sum(item.get("role") == "solution-source" for item in artifacts) != 1:
        raise ValueError("runtime source archive closure changed")
    if sum(item.get("role") == "python-wheel" for item in artifacts) != 19:
        raise ValueError("runtime wheel closure changed")
    if not any(
        item.get("name") == "gymnasium-robotics-source"
        and item.get("sha256")
        == EXPECTED_SOURCE_FIELDS["farama_gymnasium_robotics"]["archive_sha256"]
        for item in artifacts
    ):
        raise ValueError("exact runtime Gymnasium-Robotics source is absent")
    if not any(
        item.get("name") == "mujoco-3.12.0-cp312-linux-x86_64"
        and item.get("sha256") == EXPECTED_SOURCE_FIELDS["mujoco"]["wheel_sha256"]
        for item in artifacts
    ):
        raise ValueError("exact runtime MuJoCo wheel is absent")

    requirements = rootfs["opt/npa/gymnasium-robotics/requirements.lock"]
    if _locked_python_distributions(requirements) != EXPECTED_PYTHON_DISTRIBUTIONS:
        raise ValueError("runtime Python distribution closure changed")
    if (
        source.get("requirements_lock_sha256")
        != hashlib.sha256(requirements).hexdigest()
    ):
        raise ValueError("runtime lock does not bind requirements bytes")

    apt = json.loads(rootfs["opt/npa/gymnasium-robotics/apt-runtime.lock.json"])
    if (
        apt.get("schema") != "npa.gymnasium-robotics.neutral-bootstrap-apt-lock.v2"
        or apt.get("status") != "complete"
        or apt.get("base") != EXPECTED_BASE
        or not apt.get("resolved_binary_packages")
        or not apt.get("resolved_source_packages")
    ):
        raise ValueError("neutral bootstrap APT/source closure is incomplete")
    binaries = {
        item.get("package"): item for item in apt["resolved_binary_packages"]
    }
    for record in EXPECTED_SYSTEM_WHEEL_FILES.values():
        package = binaries.get(record["package"])
        if not package or package.get("sha256") != record["package_sha256"]:
            raise ValueError("system bootstrap wheel package closure changed")
    corresponding = json.loads(
        rootfs["opt/npa/gymnasium-robotics/corresponding-source.lock.json"]
    )
    if (
        corresponding.get("schema")
        != "npa.gymnasium-robotics.baked-corresponding-source-lock.v2"
        or corresponding.get("status") != "complete"
        or corresponding.get("scope") != "candidate-image-layers-only"
        or len(corresponding.get("deliveries") or []) != 1
        or corresponding["deliveries"][0].get("binary_component")
        != "ubuntu-neutral-bootstrap-closure"
        or not corresponding["deliveries"][0].get("artifacts")
    ):
        raise ValueError("neutral image corresponding-source closure is incomplete")


def _oci_blob(
    archive: tarfile.TarFile,
    descriptor: object,
    *,
    label: str,
    referenced: set[str],
    nested_budget: _NestedArchiveBudget,
    graph_budget: _OciDescriptorGraphBudget,
    member_aliases: dict[str, str] | None = None,
    scan_layer_text_policy: bool = True,
) -> tuple[bytes, str]:
    if not isinstance(descriptor, dict):
        raise ValueError(f"OCI {label} descriptor is not an object")
    media_type = descriptor.get("mediaType")
    digest = descriptor.get("digest")
    size = descriptor.get("size")
    if (
        not isinstance(media_type, str)
        or not media_type
        or not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        raise ValueError(f"OCI {label} descriptor is incomplete")
    graph_budget.reserve(label)
    name = f"blobs/sha256/{digest.removeprefix('sha256:')}"
    member_name = (member_aliases or {}).get(name, name)
    raw = graph_budget.validated_blobs.get(digest)
    if raw is None:
        raw = _raw_member(
            archive,
            member_name,
            max_bytes=MAX_LAYER_ARCHIVE_BYTES,
        )
    if len(raw) != size or (
        digest not in graph_budget.validated_blobs
        and "sha256:" + hashlib.sha256(raw).hexdigest() != digest
    ):
        raise ValueError(f"OCI {label} descriptor does not bind its blob")
    referenced.add(member_name)
    if digest not in graph_budget.validated_blobs:
        # A compressed runtime layer is an opaque representation, not an
        # actionable text member.  Running the text regex over its compressed
        # bytes creates false positives when an ordinary decoded file happens
        # to compress to a matching byte sequence.  Raw-byte identities and
        # decoded TAR-member policy remain mandatory below; descriptor,
        # config, and attestation blobs still receive direct text scanning.
        _scan_decoded_member_bytes(
            f"raw OCI {label} blob",
            raw,
            skip_text_policy=media_type in OCI_LAYER_MEDIA_TYPES,
        )
        # Runtime layer blobs are decompressed and semantically scanned by the
        # selected-manifest layer path below.  Applying generic nested-archive
        # text policy to their compressed bytes here creates false positives on
        # ordinary rootfs files; descriptor/config/attestation blobs still get
        # complete recursive archive inspection at this boundary.
        if media_type not in OCI_LAYER_MEDIA_TYPES:
            _nested_archive_members(
                f"raw OCI {label} blob", raw, budget=nested_budget
            )
        elif scan_layer_text_policy:
            _nested_archive_members(
                f"raw OCI {label} blob",
                raw,
                budget=nested_budget,
                skip_text_policy=not scan_layer_text_policy,
            )
        graph_budget.remember(
            digest,
            raw,
            text_policy_scanned=scan_layer_text_policy
            or media_type not in OCI_LAYER_MEDIA_TYPES,
        )
    elif (
        scan_layer_text_policy
        and media_type in OCI_LAYER_MEDIA_TYPES
        and not graph_budget.text_policy_scanned.get(digest, False)
    ):
        # A shared layer may first be reached through a runnable manifest and
        # later through an attestation manifest. Upgrade the cache entry to the
        # stricter decoded-member policy instead of treating the digest cache
        # as proof that the stronger scan already ran.
        _nested_archive_members(
            f"raw OCI {label} blob",
            raw,
            budget=nested_budget,
        )
        graph_budget.text_policy_scanned[digest] = True
    return raw, media_type


def _oci_manifest_candidates(
    archive: tarfile.TarFile,
    descriptors: object,
    *,
    referenced: set[str],
    nested_budget: _NestedArchiveBudget,
    graph_budget: _OciDescriptorGraphBudget,
    member_aliases: dict[str, str] | None = None,
    inherited_platform: dict[str, object] | None = None,
    depth: int = 0,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Validate the complete descriptor graph and return runnable manifests."""

    if depth > 8 or not isinstance(descriptors, list) or not descriptors:
        raise ValueError("OCI index descriptor graph is absent or too deep")
    if len(descriptors) > MAX_ORDERED_LAYERS:
        raise ValueError("OCI index descriptor count exceeds scan bound")
    candidates: list[tuple[dict[str, object], dict[str, object]]] = []
    for index, descriptor in enumerate(descriptors):
        raw, media_type = _oci_blob(
            archive,
            descriptor,
            label=f"graph descriptor {depth}:{index}",
            referenced=referenced,
            nested_budget=nested_budget,
            graph_budget=graph_budget,
            member_aliases=member_aliases,
        )
        if media_type not in OCI_INDEX_MEDIA_TYPES | OCI_MANIFEST_MEDIA_TYPES:
            raise ValueError("OCI index contains an unsupported descriptor media type")
        _scan_decoded_member_bytes(f"decoded OCI descriptor {depth}:{index}", raw)
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("OCI descriptor JSON is malformed") from error
        if not isinstance(document, dict) or document.get("schemaVersion") != 2:
            raise ValueError("OCI descriptor document is malformed")
        platform = descriptor.get("platform")
        if platform is None:
            platform = inherited_platform
        elif not isinstance(platform, dict):
            raise ValueError("OCI descriptor platform is malformed")
        if media_type in OCI_INDEX_MEDIA_TYPES:
            if document.get("mediaType") not in (None, media_type):
                raise ValueError("OCI index media type changed inside its blob")
            candidates.extend(
                _oci_manifest_candidates(
                    archive,
                    document.get("manifests"),
                    referenced=referenced,
                    nested_budget=nested_budget,
                    graph_budget=graph_budget,
                    member_aliases=member_aliases,
                    inherited_platform=platform,
                    depth=depth + 1,
                )
            )
            continue
        if document.get("mediaType") not in (None, media_type):
            raise ValueError("OCI manifest media type changed inside its blob")
        config_raw, _config_media = _oci_blob(
            archive,
            document.get("config"),
            label=f"manifest config {depth}:{index}",
            referenced=referenced,
            nested_budget=nested_budget,
            graph_budget=graph_budget,
            member_aliases=member_aliases,
        )
        _scan_decoded_member_bytes(
            f"decoded OCI manifest config {depth}:{index}", config_raw
        )
        layers = document.get("layers")
        if not isinstance(layers, list) or len(layers) > MAX_ORDERED_LAYERS:
            raise ValueError("OCI manifest layer descriptor list is malformed")
        annotations = descriptor.get("annotations") or {}
        if not isinstance(annotations, dict):
            raise ValueError("OCI descriptor annotations are malformed")
        attestation = annotations.get("vnd.docker.reference.type") == "attestation-manifest"
        for layer_index, layer_descriptor in enumerate(layers):
            _oci_blob(
                archive,
                layer_descriptor,
                label=f"manifest layer {depth}:{index}:{layer_index}",
                referenced=referenced,
                nested_budget=nested_budget,
                graph_budget=graph_budget,
                member_aliases=member_aliases,
                scan_layer_text_policy=attestation,
            )
        if not attestation:
            document = dict(document)
            document["_npa_platform"] = platform
            candidates.append((descriptor, document))
    return candidates


def _strict_oci_layer(
    label: str, raw: bytes, *, media_type: str
) -> bytes:
    compression = OCI_LAYER_MEDIA_TYPES.get(media_type)
    if compression is None:
        raise ValueError(f"unsupported OCI runtime layer media type: {label}")
    if compression == "identity":
        decoded = raw
    else:
        try:
            stream = zlib.decompressobj(16 + zlib.MAX_WBITS)
            decoded = stream.decompress(raw, MAX_LAYER_ARCHIVE_BYTES + 1)
        except zlib.error as error:
            raise ValueError(f"unreadable OCI layer stream: {label}") from error
        if not stream.eof or stream.unconsumed_tail or stream.unused_data:
            raise ValueError(f"ambiguous compressed stream: {label}")
        if len(decoded) > MAX_LAYER_ARCHIVE_BYTES:
            raise ValueError(f"OCI layer exceeds scan bound: {label}")
    empty_tar = len(decoded) == 1024 and not any(decoded)
    if len(decoded) > MAX_LAYER_ARCHIVE_BYTES or not (empty_tar or _looks_like_tar(decoded)):
        raise ValueError(f"OCI layer is not one bounded tar stream: {label}")
    return decoded


def _selected_oci_manifest(
    candidates: list[tuple[dict[str, object], dict[str, object]]],
) -> dict[str, object]:
    if len(candidates) != 1:
        raise ValueError("OCI layout must contain exactly one runnable image")
    _descriptor, manifest = candidates[0]
    platform = manifest.get("_npa_platform")
    if platform is None:
        return manifest
    if not isinstance(platform, dict) or (
        platform.get("architecture") != "amd64" or platform.get("os") != "linux"
    ):
        raise ValueError("OCI image platform must be linux/amd64")
    return manifest


def _finalize_scan(
    *,
    archive_sha256: str,
    config_raw: bytes,
    config: dict[str, object],
    layers: list[tuple[str, bytes]],
    layer_descriptors: list[dict[str, object]] | None,
    archive_format: str,
    nested_budget: _NestedArchiveBudget | None = None,
    expected_config_digest: str | None = None,
    distributed_blob_scan_complete: bool = False,
    expected_layer_diff_ids: list[str] | None = None,
    pre_scanned_text_digests: set[str] | None = None,
) -> dict[str, Any]:
    runtime_config = config.get("config")
    if not isinstance(runtime_config, dict) or runtime_config.get("User") != "ubuntu":
        raise ValueError("final image must declare the non-root ubuntu user")
    (
        rootfs,
        entries,
        layer_members,
        nested_members,
        layer_diff_ids,
        whiteouts,
    ) = _layers_from_bytes(
        layers,
        nested_budget=nested_budget,
        pre_scanned_text_digests=pre_scanned_text_digests,
    )
    config_rootfs = config.get("rootfs")
    if (
        not isinstance(config_rootfs, dict)
        or config_rootfs.get("type") != "layers"
        or config_rootfs.get("diff_ids") != layer_diff_ids
    ):
        raise ValueError(
            "image config rootfs diff IDs do not match ordered layer bytes"
        )
    config_digest = hashlib.sha256(config_raw).hexdigest()
    _reviewed_image_graph(
        config_digest,
        layer_diff_ids,
        layer_descriptors,
        expected_config_digest,
        expected_layer_diff_ids,
    )
    if expected_config_digest is not None or expected_layer_diff_ids is not None:
        _reviewed_image_closure(config, layers)
    missing = sorted(REQUIRED - rootfs.keys())
    if missing:
        raise ValueError(f"required image files absent: {missing}")
    _neutral_candidate(rootfs, layer_diff_ids)
    return {
        "schema": "npa.gymnasium-robotics.payload-scan.v1",
        "status": "passed",
        "archive_format": archive_format,
        "archive_sha256": archive_sha256,
        "config_sha256": config_digest,
        "layer_count": len(layers),
        "layer_member_count": layer_members,
        "nested_archive_member_count": nested_members,
        "ordered_layer_diff_ids": layer_diff_ids,
        "ordered_layer_descriptors": layer_descriptors or [],
        "distributed_blob_scan_complete": distributed_blob_scan_complete,
        "whiteout_entry_count": len(whiteouts),
        "whiteout_metadata_sha256": hashlib.sha256(
            json.dumps(whiteouts, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "final_entry_count": len(entries),
        "final_regular_file_count": len(rootfs),
        "unresolved_findings": 0,
        "upstream_runtime_payload_count": 0,
        "shadow_asset_count": 0,
        "runtime_cache_entry_count": 0,
        "accepted_manifest_present": False,
        "release_authorized": False,
    }


def scan_oci_layout(
    path: Path,
    *,
    expected_config_digest: str | None = None,
    expected_layer_diff_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Bind a complete OCI distribution graph, including compressed blobs."""

    archive_bytes = _container_archive_bytes(path)
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    _scan_raw_blob_bytes("complete OCI layout archive", archive_bytes)
    outer_members = _validated_tar_members(
        "OCI layout archive",
        archive_bytes,
        max_members=MAX_DOCKER_SAVE_OUTER_MEMBERS,
    )
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        outer_by_name = {_safe(member.name): member for member in outer_members}
        if len(outer_by_name) != len(outer_members):
            raise ValueError("OCI layout contains duplicate normalized member paths")
        layout_raw = _raw_member(
            archive, "oci-layout", max_bytes=MAX_DOCKER_SAVE_METADATA_BYTES
        )
        index_raw = _raw_member(
            archive, "index.json", max_bytes=MAX_DOCKER_SAVE_METADATA_BYTES
        )
        _scan_decoded_member_bytes("OCI layout metadata", layout_raw)
        _scan_decoded_member_bytes("OCI root index", index_raw)
        try:
            layout = json.loads(layout_raw)
            index = json.loads(index_raw)
        except json.JSONDecodeError as error:
            raise ValueError("OCI layout metadata is malformed") from error
        if layout != {"imageLayoutVersion": "1.0.0"}:
            raise ValueError("OCI layout version is unsupported")
        if (
            not isinstance(index, dict)
            or index.get("schemaVersion") != 2
            or index.get("mediaType") not in (None, *OCI_INDEX_MEDIA_TYPES)
        ):
            raise ValueError("OCI root index is malformed")
        referenced: set[str] = set()
        nested_budget = _NestedArchiveBudget()
        graph_budget = _OciDescriptorGraphBudget()
        candidates = _oci_manifest_candidates(
            archive,
            index.get("manifests"),
            referenced=referenced,
            nested_budget=nested_budget,
            graph_budget=graph_budget,
        )
        manifest = _selected_oci_manifest(candidates)
        config_descriptor = manifest.get("config")
        config_raw, config_media = _oci_blob(
            archive,
            config_descriptor,
            label="selected image config",
            referenced=referenced,
            nested_budget=nested_budget,
            graph_budget=graph_budget,
        )
        if config_media not in OCI_CONFIG_MEDIA_TYPES:
            raise ValueError("OCI image config media type is unsupported")
        config = json.loads(config_raw)
        if not isinstance(config, dict):
            raise ValueError("OCI image config must be an object")
        layer_descriptors = manifest.get("layers")
        if not isinstance(layer_descriptors, list) or not layer_descriptors:
            raise ValueError("OCI image contains no layers")
        layers: list[tuple[str, bytes]] = []
        normalized_descriptors: list[dict[str, object]] = []
        pre_scanned_text_digests: set[str] = set()
        for position, descriptor in enumerate(layer_descriptors):
            raw, media_type = _oci_blob(
                archive,
                descriptor,
                label=f"selected layer {position}",
                referenced=referenced,
                nested_budget=nested_budget,
                graph_budget=graph_budget,
                scan_layer_text_policy=False,
            )
            assert isinstance(descriptor, dict)
            decoded = _strict_oci_layer(
                f"layer-{position}", raw, media_type=media_type
            )
            digest = descriptor.get("digest") if isinstance(descriptor, dict) else None
            if isinstance(digest, str) and graph_budget.text_policy_scanned.get(digest, False):
                pre_scanned_text_digests.add(hashlib.sha256(decoded).hexdigest())
            layers.append((f"layer-{position}.tar", decoded))
            normalized_descriptors.append(
                {
                    "mediaType": media_type,
                    "size": len(raw),
                    "digest": descriptor["digest"],
                }
            )
        allowed_files = {"oci-layout", "index.json", *referenced}
        _validate_exact_outer_layout(
            archive,
            outer_by_name,
            allowed_files,
            label="OCI layout",
        )
    return _finalize_scan(
        archive_sha256=archive_sha256,
        config_raw=config_raw,
        config=config,
        layers=layers,
        layer_descriptors=normalized_descriptors,
        archive_format="oci-layout",
        nested_budget=nested_budget,
        expected_config_digest=expected_config_digest,
        distributed_blob_scan_complete=True,
        expected_layer_diff_ids=expected_layer_diff_ids,
        pre_scanned_text_digests=pre_scanned_text_digests,
    )


def _bind_docker_save_oci_graph(
    archive: tarfile.TarFile,
    outer_by_name: dict[str, tarfile.TarInfo],
    config_name: str,
    layer_names: list[str],
) -> tuple[set[str], bool]:
    """Bind Docker-save's optional OCI index and every reachable descriptor.

    Docker 29 may place an OCI index, attestations, and descriptor blobs beside
    the classic ``manifest.json``.  Those bytes are part of the image graph,
    not arbitrary auxiliary files: every descriptor must resolve to a hashed
    member, the selected runnable manifest must describe the classic config and
    layers byte-for-byte, and no blob may remain outside the graph.
    """

    outer_names = set(outer_by_name)
    auxiliary = {
        name
        for name in outer_names
        if re.fullmatch(r"blobs/sha256/[0-9a-f]{64}", name)
        and name != config_name
        and name not in set(layer_names)
    }
    if "index.json" not in outer_names:
        if "oci-layout" in outer_names:
            raise ValueError(
                "Docker-save OCI layout marker requires an index descriptor graph"
            )
        if auxiliary:
            raise ValueError(
                "unexpected Docker-save members: auxiliary OCI blobs require an index descriptor graph"
            )
        return set(), False
    index_raw = _raw_member(
        archive, "index.json", max_bytes=MAX_DOCKER_SAVE_METADATA_BYTES
    )
    _scan_decoded_member_bytes("Docker-save OCI root index", index_raw)
    try:
        index = json.loads(index_raw)
    except json.JSONDecodeError as error:
        raise ValueError("Docker-save OCI root index is malformed") from error
    if (
        not isinstance(index, dict)
        or index.get("schemaVersion") != 2
        or index.get("mediaType") not in (None, *OCI_INDEX_MEDIA_TYPES)
    ):
        raise ValueError("Docker-save OCI root index is malformed")
    if "oci-layout" not in outer_names:
        raise ValueError("Docker-save OCI index is missing oci-layout")
    layout_raw = _raw_member(
        archive, "oci-layout", max_bytes=MAX_DOCKER_SAVE_METADATA_BYTES
    )
    _scan_decoded_member_bytes("Docker-save OCI layout", layout_raw)
    try:
        layout = json.loads(layout_raw)
    except json.JSONDecodeError as error:
        raise ValueError("Docker-save OCI layout is malformed") from error
    if layout != {"imageLayoutVersion": "1.0.0"}:
        raise ValueError("Docker-save OCI layout version is unsupported")

    config_digest = _docker_save_config_digest(config_name)
    config_aliases = {
        f"blobs/sha256/{config_digest}": config_name
    }
    graph_referenced: set[str] = set()
    nested_budget = _NestedArchiveBudget()
    graph_budget = _OciDescriptorGraphBudget()
    candidates = _oci_manifest_candidates(
        archive,
        index.get("manifests"),
        referenced=graph_referenced,
        nested_budget=nested_budget,
        graph_budget=graph_budget,
        member_aliases=config_aliases,
    )
    selected = _selected_oci_manifest(candidates)
    selected_config = selected.get("config")
    if (
        not isinstance(selected_config, dict)
        or selected_config.get("digest") != f"sha256:{config_digest}"
    ):
        raise ValueError(
            "Docker-save OCI runnable manifest does not bind manifest.json config"
        )
    selected_layers = selected.get("layers")
    if not isinstance(selected_layers, list) or len(selected_layers) != len(layer_names):
        raise ValueError(
            "Docker-save OCI runnable manifest does not bind manifest.json layers"
        )
    for position, (descriptor, layer_name) in enumerate(
        zip(selected_layers, layer_names, strict=True)
    ):
        if not isinstance(descriptor, dict):
            raise ValueError("Docker-save OCI layer descriptor is malformed")
        raw = _raw_member(
            archive, layer_name, max_bytes=MAX_LAYER_ARCHIVE_BYTES
        )
        expected_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if descriptor.get("digest") != expected_digest:
            raise ValueError(
                f"Docker-save OCI layer descriptor does not bind layer {position}"
            )
        media_type = descriptor.get("mediaType")
        if media_type not in OCI_LAYER_MEDIA_TYPES:
            raise ValueError("Docker-save OCI layer media type is unsupported")
        _strict_oci_layer(
            f"Docker-save OCI layer {position}",
            raw,
            media_type=media_type,
        )
    if auxiliary - graph_referenced:
        raise ValueError("Docker-save contains unreferenced OCI graph blobs")
    return graph_referenced, True


def scan(
    path: Path,
    *,
    expected_config_digest: str | None = None,
    expected_layer_diff_ids: list[str] | None = None,
) -> dict[str, Any]:
    archive_bytes = _docker_save_bytes(path)
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    _scan_raw_blob_bytes("complete Docker-save archive", archive_bytes)
    outer_members = _validated_tar_members(
        "Docker-save archive",
        archive_bytes,
        max_members=MAX_DOCKER_SAVE_OUTER_MEMBERS,
    )
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        outer_by_name: dict[str, tarfile.TarInfo] = {}
        for member in outer_members:
            name = _safe(member.name)
            prior = outer_by_name.get(name)
            if prior is not None and not _repeated_docker_layer_member_is_identical(
                archive_bytes, prior, member, name
            ):
                raise ValueError("Docker save contains duplicate normalized member paths")
            outer_by_name.setdefault(name, member)
        outer_names = set(outer_by_name)
        manifest_raw = _raw_member(
            archive, "manifest.json", max_bytes=MAX_DOCKER_SAVE_METADATA_BYTES
        )
        _scan_decoded_member_bytes("Docker-save manifest", manifest_raw)
        manifest = json.loads(manifest_raw)
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise ValueError("Docker save must contain exactly one image")
        entry = manifest[0]
        if not isinstance(entry, dict):
            raise ValueError("Docker save manifest entry must be an object")
        config_name = _safe(str(entry["Config"]))
        config_raw = _raw_member(
            archive, config_name, max_bytes=MAX_DOCKER_SAVE_METADATA_BYTES
        )
        config_digest = hashlib.sha256(config_raw).hexdigest()
        if _docker_save_config_digest(config_name) != config_digest:
            raise ValueError(
                "Docker save config filename does not bind its exact bytes"
            )
        _scan_decoded_member_bytes("exact image config", config_raw)
        config = json.loads(config_raw)
        if not isinstance(config, dict):
            raise ValueError("Docker save config must be an object")
        runtime_config = config.get("config")
        if (
            not isinstance(runtime_config, dict)
            or runtime_config.get("User") != "ubuntu"
        ):
            raise ValueError("final image must declare the non-root ubuntu user")
        layer_names = entry.get("Layers")
        if not isinstance(layer_names, list) or not layer_names:
            raise ValueError("Docker save contains no layers")
        if len(layer_names) > MAX_ORDERED_LAYERS:
            raise ValueError("ordered layer count exceeds scan bound")
        layers = [_safe(str(name)) for name in layer_names]
        for name in set(layers):
            if layers.count(name) > 1:
                member = outer_by_name.get(name)
                if member is None or not _repeated_docker_layer_member_is_identical(
                    archive_bytes, member, member, name
                ):
                    raise ValueError("Docker save repeats an ordered layer")
        graph_referenced, graph_complete = _bind_docker_save_oci_graph(
            archive,
            outer_by_name,
            config_name,
            layers,
        )
        allowed_outer = {
            "manifest.json",
            "index.json",
            "oci-layout",
            config_name,
            *layers,
            *graph_referenced,
            "repositories",
        }
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
            member = outer_by_name[directory]
            if not member.isdir():
                raise ValueError(
                    f"Docker-save parent member is not a directory: {directory}"
                )
            _validate_tar_directory(archive, member, f"Docker-save:{directory}")
        if "repositories" in outer_names:
            _scan_decoded_member_bytes(
                "Docker-save repositories",
                _raw_member(
                    archive,
                    "repositories",
                    max_bytes=MAX_DOCKER_SAVE_METADATA_BYTES,
                ),
            )
        raw_layers: list[tuple[str, bytes]] = []
        for name in layers:
            stored = _raw_member(
                archive, name, max_bytes=MAX_LAYER_ARCHIVE_BYTES
            )
            _scan_raw_blob_bytes(f"stored layer bytes: {name}", stored)
            if stored.startswith(b"\x1f\x8b"):
                decoded = _decompress(
                    name, stored, "gzip", max_bytes=MAX_LAYER_ARCHIVE_BYTES
                )
                raw_layers.append((name, decoded))
            else:
                raw_layers.append((name, stored))
    return _finalize_scan(
        archive_sha256=archive_sha256,
        config_raw=config_raw,
        config=config,
        layers=raw_layers,
        layer_descriptors=None,
        archive_format="docker-save",
        expected_config_digest=expected_config_digest,
        distributed_blob_scan_complete=graph_complete,
        expected_layer_diff_ids=expected_layer_diff_ids,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--docker-save", type=Path)
    inputs.add_argument("--oci-layout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-config-sha256",
        help="Exact config digest from the build metadata being scanned",
    )
    parser.add_argument(
        "--expected-layer-diff-ids-json",
        help="JSON list of ordered RootFS layer DiffIDs from the exact image",
    )
    args = parser.parse_args(argv)
    try:
        if args.expected_config_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", args.expected_config_sha256
        ):
            raise ValueError("expected config digest must be a lowercase SHA-256")
        expected_layer_diff_ids = None
        if args.expected_layer_diff_ids_json is not None:
            parsed_layers = json.loads(args.expected_layer_diff_ids_json)
            if (
                not isinstance(parsed_layers, list)
                or not parsed_layers
                or any(
                    not isinstance(value, str)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                    for value in parsed_layers
                )
            ):
                raise ValueError("expected layer DiffIDs must be a non-empty JSON list")
            expected_layer_diff_ids = parsed_layers
        result = (
            scan_oci_layout(
                args.oci_layout,
                expected_config_digest=args.expected_config_sha256,
                expected_layer_diff_ids=expected_layer_diff_ids,
            )
            if args.oci_layout is not None
            else scan(
                args.docker_save,
                expected_config_digest=args.expected_config_sha256,
                expected_layer_diff_ids=expected_layer_diff_ids,
            )
        )
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
